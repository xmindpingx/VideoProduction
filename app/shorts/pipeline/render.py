"""One job, start to finish: analyze clips -> plan the edit -> cut on the beat -> reframe every segment to 9:16 ->
clean and match dialogue -> End-Frame Motion transitions -> assemble -> captions + film finish + mix -> short.mp4"""
import json
import shutil
import time

from .. import config, db, llm, media
from . import analyze, audio, captions, director, look, reframe, transitions

XFADE = {"crossfade": "fade", "whip": "hblur", "flash": "fadewhite", "zoom": "zoomin", "dip": "fadeblack", "ai_motion": "fade"}
SFX_JOINS = {"whip", "zoom", "flash", "ai_motion"}


def snap_to_beats(plan, clips, beats, fps, log, window=0.35):
    """Nudge each cut onto the nearest music beat (within `window` s) without splitting a word."""
    if not beats:
        return 0
    t, moved = 0.0, 0
    for j, seg in enumerate(plan["segments"][:-1]):
        clip = clips[seg["clip"]]
        k = director.slow_factor(clip, fps) if seg["slowmo"] else 1.0
        join = plan["joins"][j]
        d = director.JOIN_SECONDS.get(join["type"], 0)
        cut_t = t + (seg["end"] - seg["start"]) * k - d / 2
        b = min(beats, key=lambda x: abs(x - cut_t))
        delta = b - cut_t
        if 0.02 < abs(delta) <= window:
            new_end = seg["end"] + delta / k
            words = [w for s in clip["analysis"]["transcript"]["sentences"] for w in s["words"]]
            inside = any(ws < new_end < we for ws, we, _ in words) and not seg["slowmo"]
            if not inside and new_end <= clip["analysis"]["duration"] and new_end - seg["start"] >= (0.6 if seg["slowmo"] else 1.0):
                seg["end"] = round(new_end, 3)
                moved += 1
        t += (seg["end"] - seg["start"]) * k - d + (director.AI_CLIP_SECONDS if join["type"] == "ai_motion" else 0)
    log(f"Cut on the beat: {moved} of {len(plan['segments']) - 1} cuts moved onto a beat")
    return moved


def assemble_video(items, joins, fps, dest, cancel=None):
    """items: [{video, seconds}], joins: [(xfade name or None, seconds)]. Returns start time of every item."""
    inputs, graph, starts = [], [], []
    for i, it in enumerate(items):
        inputs += ["-i", str(it["video"])]
        graph.append(f"[{i}:v]fps={fps},settb=AVTB,format=yuv420p,setsar=1[v{i}]")
    cur, length = "[v0]", items[0]["seconds"]
    starts.append(0.0)
    for k, (name, d) in enumerate(joins, start=1):
        out = f"[x{k}]"
        if name and d > 0:
            graph.append(f"{cur}[v{k}]xfade=transition={name}:duration={d:.3f}:offset={length - d:.4f}{out}")
            starts.append(length - d)
            length += items[k]["seconds"] - d
        else:
            graph.append(f"{cur}[v{k}]concat=n=2:v=1:a=0{out}")
            starts.append(length)
            length += items[k]["seconds"]
        cur = out
    tail, codec = media.encoder(14)
    graph.append(f"{cur}{tail}[vout]")
    media.ffmpeg(*media.hw_init_args(), *inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", *codec,
                 "-r", fps, dest, cancel=cancel, timeout=3600)
    return starts, length


def run(job, log, progress, cancel):
    jid, brief = job["id"], job["brief"]
    fps = brief["fps"]
    work = config.JOBS_DIR / jid
    shutil.rmtree(work, ignore_errors=True)
    (work / "seg").mkdir(parents=True)
    out = work / "out"
    out.mkdir()
    warnings = []

    # 1. analyze every clip (cached per upload, so remixes skip this)
    clips = []
    uploads = [db.get_upload(u) for u in job["upload_ids"]]
    n = len(uploads)
    for i, u in enumerate(uploads):
        if not u:
            raise RuntimeError("A clip was deleted while the short was waiting")
        db.update_upload(u["id"], analysis="running")
        try:
            a = analyze.analyze(u, log, cancel, progress=lambda f, msg=None, i=i: progress(0.02 + 0.38 * (i + f) / n, msg))
        except BaseException as e:
            db.update_upload(u["id"], analysis="none" if isinstance(e, media.Canceled) else "error")
            raise
        db.update_upload(u["id"], analysis="done")
        d = config.UPLOADS_DIR / u["id"]
        clips.append({"id": u["id"], "name": u["name"], "note": u["note"], "slowmo": u["slowmo"], "meta": u["meta"],
                      "analysis": a, "src": d / ("source" + u["ext"]), "dir": d})

    analyze.release_whisper()  # hand the GPU memory to the LLM / video generator

    # 2. music beats
    beats, music_wav = [], None
    if job.get("music_id") and brief["music_mode"] != "original_only":
        progress(0.41, "Finding the beat in your music")
        m = db.get_upload(job["music_id"])
        if m:
            music_wav = work / "music.wav"
            try:
                beats = audio.prepare_music(config.UPLOADS_DIR / m["id"] / ("source" + m["ext"]), music_wav, log)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Music could not be used: {str(e)[:120]}")
                log(warnings[-1])
                music_wav = None

    # 3. plan the edit
    progress(0.44, f"Planning the edit ({config.TEXT_MODEL})")
    plan = director.make_plan(clips, brief, log)
    if plan["source"] == "rules":
        warnings.append("The AI editor (Ollama) was not reachable, so the rule-based editor picked the moments.")
    if config.OLLAMA_UNLOAD_AFTER:
        for m in {config.TEXT_MODEL, config.VISION_MODEL}:
            llm.unload(m)  # free GPU memory for the video generator
    if beats:
        snap_to_beats(plan, clips, beats, fps, log)

    # 4. reframe + render each segment, with its dialogue
    segs = []
    punch = False
    for k, seg in enumerate(plan["segments"]):
        clip = clips[seg["clip"]]
        speed = director.slow_factor(clip, fps) if seg["slowmo"] else 1.0
        progress(0.48 + 0.27 * k / len(plan["segments"]),
                 f"Reframing shot {k + 1} of {len(plan['segments'])}" + (f" (slow motion x{speed:g})" if speed > 1 else ""))
        # Punch-in on alternate talking shots hides jump cuts and keeps the eye busy.
        talking = clip["analysis"]["face_ratio"] > 0.3 and not seg["slowmo"]
        zoom = 1.0
        if brief["punch_in"] and talking:
            punch = not punch
            zoom = 1.12 if punch else 1.0
        vpath = work / "seg" / f"{k:02d}.mp4"
        frames = reframe.render_segment(clip["src"], clip["meta"], clip["analysis"], seg["start"], seg["end"], speed, fps, vpath,
                                        zoom=zoom, shutter=brief["look"] == "film", cancel=cancel)
        seconds = frames / fps
        apath = work / "seg" / f"{k:02d}.wav"
        a_src = clip["src"]
        mode = config.AUDIO_CLEANUP if brief["clean_audio"] else "off"
        if mode == "demucs" and not seg["slowmo"]:
            try:
                v = audio.isolate_vocals(clip["dir"], clip["src"], log, cancel)
            except media.MediaError as e:
                log(f"Demucs failed ({str(e)[:160]}); using the RNNoise denoiser instead")
                v = None
            if v:
                a_src, mode = v, "off"
            else:
                mode = "rnnoise"
        speech = audio.render_segment_audio(a_src, clip["meta"], seg["start"], seg["end"], speed, seconds, apath, mode, cancel)
        speech = speech and any(seg["start"] - 0.1 <= w[0] and w[1] <= seg["end"] + 0.1
                                for s in clip["analysis"]["transcript"]["sentences"] for w in s["words"])
        segs.append({"video": vpath, "audio": apath, "seconds": seconds, "clip": seg["clip"], "seg": seg, "speech": speech,
                     "speed": speed, "zoom": zoom})
    audio.match_voices([{"path": s["audio"], "speech": s["speech"], "clip": s["clip"]} for s in segs], log)

    # 5. transitions (End-Frame Motion for ai_motion joins)
    items, joins, sfx_times_idx = [segs[0]], [], []
    made = {"ai": 0, "still": 0}
    ai_joins = [j for j in plan["joins"] if j["type"] == "ai_motion"]
    done_ai = 0
    for j, join in enumerate(plan["joins"]):
        nxt = segs[j + 1]
        if join["type"] == "ai_motion":
            done_ai += 1
            progress(0.76 + 0.09 * done_ai / max(1, len(ai_joins)), f"End-Frame Motion transition {done_ai} of {len(ai_joins)}")
            last_png, first_png = work / "seg" / f"{j:02d}_last.png", work / "seg" / f"{j + 1:02d}_first.png"
            reframe.extract_frame(segs[j]["video"], last_png, "last")
            reframe.extract_frame(nxt["video"], first_png, "first")
            gpath, gwav = work / "seg" / f"{j:02d}_gen.mp4", work / "seg" / f"{j:02d}_gen.wav"
            kind = transitions.make_transition(last_png, first_png, join, director.AI_CLIP_SECONDS, fps, gpath, work / f"gen{j}",
                                               log, cancel, allow_ai=brief["ai_transitions"])
            made[kind] += 1
            audio.silence(director.AI_CLIP_SECONDS, gwav)
            bridge = kind == "ai" and transitions.comfy.status().get("first_last_frame")
            sfx_times_idx.append(len(items))
            items.append({"video": gpath, "audio": gwav, "seconds": director.AI_CLIP_SECONDS, "gen": kind})
            joins.append(("cut", None, 0.0))
            joins.append(("cut", None, 0.0) if bridge else ("crossfade", "fade", 0.3))
        else:
            d = director.JOIN_SECONDS.get(join["type"], 0.0)
            joins.append((join["type"], XFADE.get(join["type"]), d))
            if join["type"] in SFX_JOINS:
                sfx_times_idx.append(len(items))
        items.append(nxt)
    if made["ai"]:
        transitions.comfy.free()  # let Ollama have the GPU back for the next job
    if made["still"] and brief["ai_transitions"]:
        warnings.append(f"{made['still']} transition(s) used still-frame motion because ComfyUI was not available.")

    # 6. assemble picture and dialogue
    progress(0.86, "Assembling the timeline")
    starts, total = assemble_video(items, [(name, d) for _, name, d in joins], fps, work / "timeline.mp4", cancel)
    audio.concat_audio_with_joins([it["audio"] for it in items], [(t, d) for t, _, d in joins], work / "dialog.wav")

    # 7. captions (Whisper word timings mapped onto the timeline)
    words, emphasis = [], []
    for it, start in zip(items, starts):
        if "seg" not in it or it["speed"] != 1.0:
            continue
        seg = it["seg"]
        emphasis += seg.get("emphasis", [])
        for s in clips[seg["clip"]]["analysis"]["transcript"]["sentences"]:
            for ws, we, w in s["words"]:
                if ws >= seg["start"] - 0.05 and we <= seg["end"] + 0.1:
                    words.append((start + ws - seg["start"], start + we - seg["start"], w))
    words.sort()
    hook_end = min(2.8, items[0]["seconds"] - 0.1) if brief["hook_title"] else 0
    ass, srt = work / "captions.ass", out / "captions.srt"
    captions.build(words, brief["captions"], plan.get("hook_text", "") if brief["hook_title"] else "", hook_end, total,
                   emphasis, ass, srt)
    if brief["captions"] != "off" and not words:
        warnings.append("No speech was transcribed, so there are no spoken-word captions.")

    # 8. mix: dialogue + ducked music + transition whooshes, mastered to -14 LUFS
    progress(0.9, "Mixing sound")
    sfx = None
    if sfx_times_idx and brief["vibe"] not in ("emotional",):
        sfx = work / "sfx.wav"
        audio.write_sfx([(starts[i], "whip") for i in sfx_times_idx], total, sfx)
    mix = audio.final_mix(work / "dialog.wav", music_wav, sfx, total, brief["music_mode"], work / "mix.wav", cancel)

    # 9. finish: grade + halation + grain under clean captions, encode for upload
    progress(0.93, "Film finish, captions and final encode")
    lk = look.look_filter(brief["look"]) or "null"
    tail, codec = media.encoder(16, final=True)
    if not media.HW["enc"]:
        codec = media.x264(crf=18, tune="grain" if brief["look"] != "off" else None) + ["-profile:v", "high", "-level:v", "4.2",
                                                                                       "-maxrate", "20M", "-bufsize", "40M"]
    vf = f"[0:v]{lk}[lk];[lk]ass='{ass}':fontsdir='{config.FONTS_DIR}',{tail}[v]"
    final = out / "short.mp4"
    media.ffmpeg(*media.hw_init_args(), "-i", work / "timeline.mp4", "-i", mix, "-filter_complex", vf, "-map", "[v]", "-map", "1:a",
                 *codec, "-r", fps,
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", f"{total:.3f}", final,
                 cancel=cancel, timeout=7200)
    media.ffmpeg("-ss", f"{min(1.0, total / 3):.2f}", "-i", final, "-frames:v", "1", "-q:v", "3", out / "cover.jpg")

    edit = {"plan": plan, "timeline": [{"start": round(s, 3), "seconds": round(it["seconds"], 3),
                                        **({"clip": clips[it["seg"]["clip"]]["name"], "from": it["seg"]["start"], "to": it["seg"]["end"],
                                            "slowmo": it["speed"], "zoom": it["zoom"]} if "seg" in it else {"generated": it.get("gen")})}
                                       for it, s in zip(items, starts)],
            "joins": [t for t, _, _ in joins], "brief": brief}
    (out / "edit.json").write_text(json.dumps(edit, indent=1))
    meta = media.probe(final)
    shutil.rmtree(work / "seg", ignore_errors=True)
    for f in ("timeline.mp4", "dialog.wav", "mix.wav", "sfx.wav", "music.wav"):
        (work / f).unlink(missing_ok=True)
    return {"title": plan.get("title", ""), "hook_text": plan.get("hook_text", ""), "post_caption": plan.get("post_caption", ""),
            "hashtags": plan.get("hashtags", []), "planner": plan["source"], "seconds": round(meta["duration"], 2),
            "segments": len(plan["segments"]), "transitions": made, "words": len(words), "warnings": warnings,
            "made_at": time.time()}
