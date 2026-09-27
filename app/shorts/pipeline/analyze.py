"""Per-clip analysis, cached next to the upload: speech (Whisper word timings), scenes, motion,
faces, sharpness, loudness, and frame descriptions from an Ollama vision model."""
import json
import subprocess
import threading

import cv2
import numpy as np

from .. import config, llm, media

VERSION = 2
SCAN_FPS = 10
SCAN_LONG = 640
FACE_EVERY = 2  # detect faces on every 2nd scanned frame (5 per second)

_whisper = None
_whisper_lock = threading.Lock()
_face_net = None


def _face_detector():
    global _face_net
    if _face_net is None:
        _face_net = cv2.dnn.readNetFromCaffe(str(config.FACE_PROTO), str(config.FACE_MODEL))
    return _face_net


def detect_faces(frame, min_conf=0.55):
    """Faces as [cx, cy, w, h, conf], normalized to the frame. SSD face detector (OpenCV DNN, CPU)."""
    h, w = frame.shape[:2]
    # Keep the aspect ratio (no squashing) with the short side at 300 px, the size the model was trained on.
    scale = 300 / min(h, w)
    bw, bh = min(600, int(round(w * scale))), min(600, int(round(h * scale)))
    net = _face_detector()
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (bw, bh), (104.0, 177.0, 123.0), swapRB=False, crop=False))
    det = net.forward()
    faces = []
    for i in range(det.shape[2]):
        c = float(det[0, 0, i, 2])
        if c < min_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in det[0, 0, i, 3:7]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue
        faces.append([round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4), round(x2 - x1, 4), round(y2 - y1, 4), round(c, 3)])
    return faces


def _frames(src, meta, fps, long_side, start=None, duration=None):
    """Yield (index, bgr frame) decoded by ffmpeg (auto-rotated, HDR tone-mapped)."""
    w, h = meta["width"], meta["height"]
    s = long_side / max(w, h)
    ow, oh = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)
    # Downscale before tone-mapping: zscale in float at 4K is the slow part.
    vf = media.join_filters(f"fps={fps}", f"scale={ow}:{oh}:flags=area", media.sdr_filter(meta), "format=bgr24")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *media.hw_decode_args()]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", vf, "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = ow * oh * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def scan(src, meta, frames_dir, cancel=None):
    """One pass over the clip at 10 fps: motion, motion centroid, scene-cut score, sharpness, brightness, faces.
    Saves one JPEG per second for the vision model."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = {"fps": SCAN_FPS, "motion": [], "mcx": [], "mcy": [], "cut": [], "sharp": [], "bright": [], "faces": []}
    prev_small = prev_hist = None
    for i, frame in _frames(src, meta, SCAN_FPS, SCAN_LONG):
        if cancel and i % 50 == 0 and cancel():
            raise media.Canceled()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (96, max(2, int(96 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_small is None:
            motion, mcx, mcy, cut = 0.0, 0.5, 0.5, 0.0
        else:
            diff = np.abs(small - prev_small)
            motion = float(diff.mean() / 255)
            tot = float(diff.sum())
            if tot > 0:
                ys, xs = np.indices(diff.shape)
                mcx = float((diff * xs).sum() / tot / diff.shape[1])
                mcy = float((diff * ys).sum() / tot / diff.shape[0])
            else:
                mcx = mcy = 0.5
            cut = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_small, prev_hist = small, hist
        out["motion"].append(round(motion, 5))
        out["mcx"].append(round(mcx, 4))
        out["mcy"].append(round(mcy, 4))
        out["cut"].append(round(cut, 4))
        out["sharp"].append(round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 1))
        out["bright"].append(round(float(gray.mean() / 255), 3))
        out["faces"].append(detect_faces(frame) if i % FACE_EVERY == 0 else None)
        if i % SCAN_FPS == 0:
            cv2.imwrite(str(frames_dir / f"{i // SCAN_FPS:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def scenes_from_scan(sc, duration, thr=0.42, min_len=1.0):
    cuts = [0.0]
    for i, c in enumerate(sc["cut"]):
        t = i / sc["fps"]
        if c > thr and t - cuts[-1] >= min_len and duration - t >= min_len:
            cuts.append(round(t, 2))
    cuts.append(round(duration, 2))
    return [[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)]


def extract_audio(src, dest, rate=16000, channels=1, cancel=None):
    media.ffmpeg("-i", src, "-vn", "-ac", channels, "-ar", rate, "-c:a", "pcm_s16le", dest, cancel=cancel, timeout=3600)


def loudness_curve(wav16k, hop=0.1):
    import soundfile as sf
    x, sr = sf.read(str(wav16k), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    n = int(sr * hop)
    if len(x) < n:
        return []
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return [round(float(v), 1) for v in 20 * np.log10(rms)]


def _torch_device():
    if config.WHISPER_DEVICE == "cpu":
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds of PyTorch report AMD GPUs as "cuda"


def _get_whisper(log):
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            if config.WHISPER_ENGINE == "faster":
                from faster_whisper import WhisperModel
                device = "cpu" if config.WHISPER_DEVICE == "cpu" else "cuda"
                log(f"Loading faster-whisper '{config.WHISPER_MODEL}' on {device} (downloads once, then cached)")
                _whisper = ("faster", WhisperModel(config.WHISPER_MODEL, device=device, compute_type=config.WHISPER_COMPUTE,
                                                   download_root=str(config.MODELS_DIR / "whisper")))
            else:
                import whisper
                device = _torch_device()
                if device == "cpu" and config.WHISPER_DEVICE != "cpu":
                    log("No GPU visible to PyTorch (check /dev/kfd and /dev/dri are mapped); Whisper is running on the CPU")
                log(f"Loading Whisper '{config.WHISPER_MODEL}' on {device} (downloads once, then cached)")
                _whisper = ("torch", whisper.load_model(config.WHISPER_MODEL, device=device,
                                                        download_root=str(config.MODELS_DIR / "whisper")))
        return _whisper


def release_whisper():
    """Free Whisper's GPU memory (before the video generator needs it)."""
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            return
        _whisper = None
        try:
            import gc

            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def transcribe(wav16k, log):
    engine, model = _get_whisper(log)
    sentences = []
    if engine == "faster":
        segments, info = model.transcribe(str(wav16k), word_timestamps=True, vad_filter=True,
                                          language=config.WHISPER_LANGUAGE or None, beam_size=5)
        language = info.language
        raw = [(seg.text, [(w.start, w.end, w.word) for w in (seg.words or [])]) for seg in segments]
    else:
        res = model.transcribe(str(wav16k), word_timestamps=True, language=config.WHISPER_LANGUAGE or None,
                               condition_on_previous_text=False, fp16=next(model.parameters()).is_cuda)
        language = res.get("language")
        raw = [(seg["text"], [(w["start"], w["end"], w["word"]) for w in seg.get("words", [])])
               for seg in res.get("segments", []) if seg.get("no_speech_prob", 0) < 0.8]
    for text, ws in raw:
        words = [[round(a, 3), round(b, 3), w.strip()] for a, b, w in ws if w.strip()]
        if words:
            sentences.append({"s": words[0][0], "e": words[-1][1], "text": text.strip(), "words": words})
    return {"language": language, "sentences": sentences}


VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "subject": {"type": "string"},
        "action": {"type": "string"},
        "shot": {"type": "string", "enum": ["close-up", "medium", "wide", "detail", "other"]},
        "interest": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["description", "subject", "action", "shot", "interest"],
}


def pick_keyframes(sc, scenes, duration, limit):
    """One sharp frame per scene (plus one every ~8 s in long scenes), at most `limit`, as whole seconds."""
    secs = []
    for s, e in scenes:
        n = max(1, int((e - s) // 8))
        for k in range(n):
            a = s + (e - s) * (k + 0.25) / n
            b = s + (e - s) * (k + 0.75) / n
            cand = [t for t in range(int(np.ceil(a)), int(b) + 1) if t < duration]
            if not cand:
                cand = [min(int(s), max(0, int(duration) - 1))]
            best = max(cand, key=lambda t: sc["sharp"][min(len(sc["sharp"]) - 1, t * sc["fps"])])
            secs.append(best)
    secs = sorted(set(secs))
    if len(secs) > limit:
        idx = np.linspace(0, len(secs) - 1, limit).round().astype(int)
        secs = [secs[i] for i in sorted(set(idx))]
    return secs


def describe_frames(frames_dir, secs, name, note, log, cancel=None):
    out = []
    system = ("You help a short-form video editor. Look at one frame from a phone video and answer only with JSON. "
              "Be concrete and brief. 'interest' rates how eye-catching the frame is as the opening shot of a TikTok/Reel (1-10).")
    for t in secs:
        if cancel and cancel():
            raise media.Canceled()
        p = frames_dir / f"{t:05d}.jpg"
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = 512 / max(h, w)
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        user = (f"Frame at {t}s of the clip '{name}'." + (f" The creator says about this clip: {note}." if note else "") +
                " Return JSON with: description (max 20 words), subject (main subject), action (max 8 words), "
                "shot (close-up|medium|wide|detail|other), interest (1-10).")
        try:
            d = llm.chat_json(config.VISION_MODEL, system, user, schema=VISION_SCHEMA, images=[buf.tobytes()], temperature=0.2, num_ctx=4096)
        except llm.LLMError as e:
            log(f"Vision model unavailable, continuing without frame descriptions ({e})")
            return out
        try:
            interest = int(d.get("interest", 5))
        except (TypeError, ValueError):
            interest = 5
        out.append({"t": t, "description": str(d.get("description", ""))[:200], "subject": str(d.get("subject", ""))[:80],
                    "action": str(d.get("action", ""))[:80], "shot": str(d.get("shot", "other"))[:20],
                    "interest": max(1, min(10, interest))})
    return out


def cache_key():
    return {"version": VERSION, "whisper": config.WHISPER_MODEL, "vision": config.VISION_MODEL}


def analyze(upload, log=print, cancel=None, progress=None):
    """Analyze one uploaded clip (cached). Returns the analysis dict."""
    d = config.UPLOADS_DIR / upload["id"]
    cache = d / "analysis.json"
    meta = upload["meta"]
    src = d / ("source" + upload["ext"])
    name = upload["name"]
    step = progress or (lambda f, msg=None: None)
    if cache.is_file():
        try:
            a = json.loads(cache.read_text())
        except ValueError:
            a = {}
        if a.get("key") == cache_key():
            # Redo only the AI steps that could not run last time (e.g. Ollama was offline).
            if a.get("transcript_error") and (d / "audio16k.wav").is_file():
                step(0.3, f"{name}: transcribing speech (Whisper)")
                try:
                    a["transcript"], a["transcript_error"] = transcribe(d / "audio16k.wav", log), None
                except Exception as e:  # noqa: BLE001
                    log(f"{name}: speech transcription skipped ({str(e)[:200]})")
            if a.get("vision_missing"):
                step(0.7, f"{name}: describing key frames ({config.VISION_MODEL})")
                secs = pick_keyframes(a["scan"], a["scenes"], meta["duration"], config.VISION_MAX_FRAMES)
                a["vision"] = describe_frames(d / "frames", secs, name, upload.get("note", ""), log, cancel)
                a["vision_missing"] = not a["vision"]
            cache.write_text(json.dumps(a))
            return a
    a = {"key": cache_key(), "duration": meta["duration"], "fps": meta["fps"], "width": meta["width"], "height": meta["height"]}

    step(0.05, f"{name}: reading audio")
    wav = d / "audio16k.wav"
    a["transcript"] = {"language": None, "sentences": []}
    a["energy"] = []
    if meta.get("has_audio"):
        extract_audio(src, wav, cancel=cancel)
        a["energy"] = loudness_curve(wav)
        step(0.15, f"{name}: transcribing speech (Whisper)")
        try:
            a["transcript"] = transcribe(wav, log)
            n = sum(len(s["words"]) for s in a["transcript"]["sentences"])
            log(f"{name}: {n} words transcribed ({a['transcript']['language'] or '?'})")
        except Exception as e:  # noqa: BLE001 - a missing model or odd audio must not sink the whole short
            a["transcript_error"] = str(e)[:200]
            log(f"{name}: speech transcription skipped ({str(e)[:200]})")

    step(0.45, f"{name}: finding scenes, motion and faces")
    sc = scan(src, meta, d / "frames", cancel=cancel)
    a["scan"] = sc
    a["scenes"] = scenes_from_scan(sc, meta["duration"])
    face_frames = [f for f in sc["faces"] if f is not None]
    a["face_ratio"] = round(sum(1 for f in face_frames if f) / max(1, len(face_frames)), 3)
    log(f"{name}: {len(a['scenes'])} scene(s), faces in {a['face_ratio'] * 100:.0f}% of frames")

    step(0.7, f"{name}: describing key frames ({config.VISION_MODEL})")
    secs = pick_keyframes(sc, a["scenes"], meta["duration"], config.VISION_MAX_FRAMES)
    a["vision"] = describe_frames(d / "frames", secs, name, upload.get("note", ""), log, cancel)
    a["vision_missing"] = not a["vision"]
    cache.write_text(json.dumps(a))
    step(1.0, f"{name}: analyzed")
    return a
