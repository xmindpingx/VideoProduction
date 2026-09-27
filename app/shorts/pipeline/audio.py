"""Dialogue clean-up, voice continuity across clips, music beats, and the final mix.

- clean-up: RNNoise speech denoiser (ffmpeg arnndn), FFT denoise, or Demucs vocal isolation
- continuity: every speaking segment is brought to the same loudness, and clips recorded in different places get a
  gentle EQ match toward the average voice tone, so the cut from one clip to the next does not jump in level or colour
- music: beat grid (librosa) for cutting on the beat; ducked under speech with a sidechain compressor
- master: EBU R128 loudness to -14 LUFS, true peak -1.5 dB (typical for TikTok/Reels/Shorts)
"""
import sys

import numpy as np
import soundfile as sf

from .. import config, media

SR = 48000
_demucs_warned = False
DIALOG_LUFS = -18.0
BANDS = [100, 200, 400, 800, 1600, 3200, 6400, 12000]


def cleanup_filter(mode):
    if mode == "basic":
        return "highpass=f=80,afftdn=nr=12:nf=-40:tn=1"
    if mode == "rnnoise" and config.RNNOISE_MODEL.is_file():
        return f"highpass=f=70,arnndn=m='{config.RNNOISE_MODEL}':mix=0.9"
    if mode in ("rnnoise", "demucs"):
        return "highpass=f=80,afftdn=nr=12:nf=-40:tn=1"
    return ""


def demucs_available():
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def isolate_vocals(upload_dir, src, log, cancel=None):
    """Demucs two-stem separation of the whole clip (cached). Returns the vocals wav, or None."""
    out = upload_dir / "vocals.wav"
    if out.is_file():
        return out
    if (upload_dir / "vocals.failed").exists():
        return None
    if not demucs_available():
        global _demucs_warned
        if not _demucs_warned:
            log("Demucs is not installed in this image (build with INSTALL_DEMUCS=true); using the RNNoise denoiser instead")
            _demucs_warned = True
        return None
    full = upload_dir / "audio48k.wav"
    media.ffmpeg("-i", src, "-vn", "-ac", 2, "-ar", SR, "-c:a", "pcm_s16le", full, cancel=cancel, timeout=3600)
    sep = upload_dir / "demucs"
    log("Isolating dialogue with Demucs (htdemucs)")
    try:
        media.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(sep), str(full)],
                  cancel=cancel, timeout=7200)
    except media.MediaError:
        (upload_dir / "vocals.failed").touch()  # do not retry for every segment of this clip
        raise
    found = next(sep.rglob("vocals.wav"), None)
    if not found:
        return None
    found.replace(out)
    return out


def render_segment_audio(src, meta, start, end, speed, seconds, dest, mode, cancel=None):
    """Segment audio as 48 kHz stereo WAV, exactly `seconds` long. Slowed segments are silent (music carries them)."""
    if speed != 1.0 or not meta.get("has_audio", True):
        media.ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo", "-t", f"{seconds:.4f}", "-c:a", "pcm_s16le", dest, cancel=cancel)
        return False
    af = media.join_filters(cleanup_filter(mode), f"apad,atrim=0:{seconds:.4f}",
                            "afade=t=in:d=0.012", f"afade=t=out:st={max(0.0, seconds - 0.012):.4f}:d=0.012")
    media.ffmpeg("-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", src, "-vn", "-ac", 2, "-ar", SR, "-af", af,
                 "-c:a", "pcm_s16le", dest, cancel=cancel, timeout=600)
    return True


def _ltas(x):
    """Long-term average spectrum in BANDS (dB)."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    n = 4096
    if len(mono) < n:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(mono, n)[:: n // 2]
    spec = (np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    out = []
    for c in BANDS:
        band = (freqs >= c / 1.41) & (freqs < c * 1.41)
        out.append(10 * np.log10(spec[band].mean() + 1e-12))
    return np.array(out)


def match_voices(items, log):
    """items: [{path, speech: bool, clip: int}]. Loudness-match every speaking segment; EQ-match across different clips."""
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    speech = [it for it in items if it["speech"]]
    spectra = {}
    for it in speech:
        x, _ = sf.read(str(it["path"]), dtype="float32", always_2d=True)
        try:
            it["lufs"] = meter.integrated_loudness(x)
        except ValueError:
            it["lufs"] = float("-inf")
        s = _ltas(x)
        if s is not None:
            spectra.setdefault(it["clip"], []).append(s - s.mean())
    ref = None
    if len(spectra) >= 2:  # only worth matching when voices come from different recordings
        per_clip = {k: np.mean(v, axis=0) for k, v in spectra.items()}
        ref = np.median(np.stack(list(per_clip.values())), axis=0)
    for it in speech:
        filters = []
        if ref is not None and it["clip"] in spectra:
            corr = np.clip((ref - np.mean(spectra[it["clip"]], axis=0)) * 0.7, -6, 6)
            if np.max(np.abs(corr)) > 0.75:
                entries = ";".join(f"entry({f},{g:.2f})" for f, g in zip(BANDS, corr))
                filters.append(f"firequalizer=gain_entry='{entries}'")
        if np.isfinite(it.get("lufs", float("-inf"))) and it["lufs"] > -70:
            gain = float(np.clip(DIALOG_LUFS - it["lufs"], -12, 15))
            filters.append(f"volume={gain:.2f}dB")
        if filters:
            tmp = it["path"].with_suffix(".m.wav")
            filters.append("alimiter=limit=0.95:level=disabled")
            media.ffmpeg("-i", it["path"], "-af", ",".join(filters), "-c:a", "pcm_s16le", tmp)
            tmp.replace(it["path"])
    if ref is not None:
        log(f"Voice continuity: loudness and EQ matched across {len(spectra)} recordings")
    elif speech:
        log("Voice continuity: loudness matched across segments")


# ---------------------------------------------------------------- music
def prepare_music(src, dest, log):
    """Decode music to 48 kHz stereo WAV starting where it gets going; returns beat times (s) from that start."""
    import librosa
    y, sr = librosa.load(str(src), sr=22050, mono=True)
    if len(y) < sr:
        raise media.MediaError("music file is too short")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = np.asarray(beats, dtype=float)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    loud = np.nonzero(rms >= np.percentile(rms, 90) * 0.6)[0]
    start = float(t_rms[loud[0]]) if len(loud) else 0.0
    if len(beats):
        start = float(beats[np.argmin(np.abs(beats - start))])
    start = min(start, max(0.0, len(y) / sr - 15))  # keep at least 15 s of music
    media.ffmpeg("-ss", f"{start:.3f}", "-i", src, "-vn", "-ac", 2, "-ar", SR, "-c:a", "pcm_s16le", dest, timeout=600)
    rel = [round(b - start, 3) for b in beats if b >= start]
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    log(f"Music: {bpm:.0f} BPM, {len(rel)} beats, starting at {start:.1f}s of the track")
    return rel


def whoosh(seconds=0.6, pan_from=-0.6, pan_to=0.6, seed=0):
    """A soft transition whoosh: filtered noise with a rising-then-falling brightness and a stereo sweep."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.linspace(0, 1, n)
    env = np.where(t < 0.45, (t / 0.45) ** 2, ((1 - t) / 0.55) ** 1.5)
    cutoff = 300 + 5000 * env
    noise = rng.standard_normal(n)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):  # one-pole low-pass with a moving cutoff
        a = 1 - np.exp(-2 * np.pi * cutoff[i] / SR)
        acc += a * (noise[i] - acc)
        out[i] = acc
    out = out / (np.abs(out).max() + 1e-9) * env * 0.25
    pan = pan_from + (pan_to - pan_from) * t
    return np.stack([out * np.sqrt((1 - pan) / 2), out * np.sqrt((1 + pan) / 2)], axis=1).astype(np.float32)


def write_sfx(events, total, dest):
    """events: [(time_s, kind)] -> stereo WAV with a whoosh at each time."""
    buf = np.zeros((int(total * SR) + SR, 2), dtype=np.float32)
    for k, (t, kind) in enumerate(events):
        w = whoosh(0.5 if kind == "whip" else 0.8, *((-0.7, 0.7) if k % 2 == 0 else (0.7, -0.7)), seed=k)
        i = max(0, int((t - 0.3) * SR))
        j = min(len(buf), i + len(w))
        buf[i:j] += w[: j - i]
    sf.write(str(dest), buf[: int(total * SR)], SR, subtype="PCM_16")


def final_mix(dialog, music, sfx, total, mode, dest, cancel=None):
    """Mix dialogue + ducked music + transition sounds, then master to -14 LUFS. Returns AAC-ready WAV path."""
    inputs = ["-i", str(dialog)]
    graph = []
    use_dialog = mode != "music_only"
    use_music = music is not None and mode != "original_only"
    graph.append(f"[0:a]{'volume=1' if use_dialog else 'volume=0'},apad,atrim=0:{total:.4f}[dlg0]")
    mix_inputs = ["[dlg]"]
    idx = 1
    if use_music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        bed = -20 if use_dialog else -14
        graph.append(f"[{idx}:a]atrim=0:{total:.4f},loudnorm=I={bed}:TP=-2:LRA=11,aresample={SR},"
                     f"afade=t=in:d=0.4,afade=t=out:st={max(0, total - 1.5):.3f}:d=1.5[mus0]")
        if use_dialog:  # duck the music whenever someone speaks
            graph.append("[dlg0]asplit=2[dlg][key]")
            graph.append("[mus0][key]sidechaincompress=threshold=0.015:ratio=6:attack=15:release=400[mus]")
        else:
            graph.append("[dlg0]anull[dlg]")
            graph.append("[mus0]anull[mus]")
        mix_inputs.append("[mus]")
        idx += 1
    else:
        graph.append("[dlg0]anull[dlg]")
    if sfx is not None:
        inputs += ["-i", str(sfx)]
        graph.append(f"[{idx}:a]apad,atrim=0:{total:.4f}[sfx]")
        mix_inputs.append("[sfx]")
        idx += 1
    graph.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0:duration=first,"
                 f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample={SR}[out]")
    media.ffmpeg(*inputs, "-filter_complex", ";".join(graph), "-map", "[out]", "-ac", 2, "-c:a", "pcm_s16le", dest,
                 cancel=cancel, timeout=1800)
    return dest


def silence(seconds, dest):
    media.ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo", "-t", f"{seconds:.4f}", "-c:a", "pcm_s16le", dest)


def concat_audio_with_joins(parts, joins, dest):
    """parts: wav paths; joins: [(kind, seconds)] between them (acrossfade for overlaps, plain concat for cuts)."""
    inputs = []
    for p in parts:
        inputs += ["-i", str(p)]
    graph, cur = [], "[0:a]"
    for k, (kind, d) in enumerate(joins, start=1):
        out = f"[a{k}]"
        if d > 0:
            graph.append(f"{cur}[{k}:a]acrossfade=d={d:.3f}:c1=tri:c2=tri{out}")
        else:
            graph.append(f"{cur}[{k}:a]concat=n=2:v=0:a=1{out}")
        cur = out
    if not graph:
        media.ffmpeg("-i", parts[0], "-c:a", "pcm_s16le", dest)
        return dest
    media.ffmpeg(*inputs, "-filter_complex", ";".join(graph), "-map", cur, "-c:a", "pcm_s16le", dest, timeout=1800)
    return dest

