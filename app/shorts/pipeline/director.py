"""The edit plan: which moments of which clips, in what order, with which transitions.

An Ollama text model plans the edit from the clips' transcripts, scene descriptions and energy; a rule-based editor
fills in (or takes over entirely) when the model is missing or returns something unusable. Either way the result is
cleaned up the same way: no cut mid-word, no reused footage, slow-motion only where the frames exist, length on target.
"""
import re

import numpy as np

from .. import config, llm

CAMERA_MOVES = ["slow push-in", "slow pull-out", "orbit left", "orbit right", "pan left", "pan right",
                "tilt up", "tilt down", "crane up", "handheld drift", "whip pan", "dolly zoom"]
EFFECTS = {
    "camera": ["none"],
    "atmosphere": ["none", "light leak", "lens flare", "dust drifting in light beams", "soft rolling fog", "rain", "rack focus"],
    "fantastical": ["none", "glowing particles rising", "floating embers", "god rays breaking through", "sparkling magic dust",
                    "swirling petals", "lightning flicker", "shattering into slow-motion shards"],
}
JOINS = ["cut", "crossfade", "whip", "flash", "zoom", "dip", "ai_motion"]
JOIN_SECONDS = {"cut": 0.0, "crossfade": 0.4, "whip": 0.3, "flash": 0.3, "zoom": 0.35, "dip": 0.5, "ai_motion": 0.3}
AI_CLIP_SECONDS = 2.0
MAX_SLOWMO = 4.0

VIBE_PACE = {"hype": (1.2, 3.5), "funny": (1.5, 5.0), "product": (1.5, 4.0), "story": (2.0, 8.0),
             "emotional": (2.5, 8.0), "cinematic": (2.5, 7.0)}
VIBE_JOINS = {
    "hype": ["cut", "whip", "cut", "flash", "cut", "zoom"],
    "funny": ["cut"],
    "product": ["cut", "zoom", "cut", "whip"],
    "story": ["cut", "cut", "crossfade"],
    "emotional": ["crossfade", "dip", "crossfade"],
    "cinematic": ["crossfade", "cut", "dip"],
}


# ---------------------------------------------------------------- helpers
def slow_factor(clip, out_fps):
    return max(1.0, min(MAX_SLOWMO, clip["meta"]["fps"] / out_fps))


def can_slowmo(clip, out_fps):
    """The frames exist for a smooth slow-down (60 fps and up)."""
    return clip["slowmo"] != "off" and slow_factor(clip, out_fps) >= 1.9


def auto_slowmo(clip, out_fps):
    """Slow down without being asked: 120/240 fps captures (iPhone Slo-mo), or the clip is set to 'on'."""
    return can_slowmo(clip, out_fps) and (clip["slowmo"] == "on" or clip["meta"]["fps"] >= 90)


def _words(a):
    return [w for s in a["transcript"]["sentences"] for w in s["words"]]


def snap_to_words(start, end, words, duration):
    """Move cut points out of words: a start inside a word goes to that word's start, an end inside a word to its end."""
    for ws, we, _ in words:
        if ws < start < we:
            start = ws
        if ws < end < we:
            end = we
    return max(0.0, start - 0.05), min(duration, end + 0.12)


def _series(a, key):
    return np.asarray(a["scan"][key], dtype=float) if a["scan"][key] else np.zeros(1)


def window_stats(a, s, e):
    fps = a["scan"]["fps"]
    i0, i1 = int(s * fps), max(int(s * fps) + 1, int(e * fps))
    motion = _series(a, "motion")[i0:i1]
    sharp = _series(a, "sharp")[i0:i1]
    faces = [f for f in a["scan"]["faces"][i0:i1] if f is not None]
    en = np.asarray(a["energy"][int(s * 10):max(int(s * 10) + 1, int(e * 10))] or [-60.0])
    vis = [v["interest"] for v in a.get("vision", []) if s - 2 <= v["t"] <= e + 2]
    return {"motion": float(motion.mean()) if motion.size else 0.0, "sharp": float(np.median(sharp)) if sharp.size else 0.0,
            "faces": sum(1 for f in faces if f) / max(1, len(faces)), "loud": float(np.percentile(en, 90)),
            "interest": float(np.mean(vis)) if vis else 5.0}


def timeline_seconds(segments, joins, clips, out_fps):
    total = 0.0
    for seg in segments:
        k = slow_factor(clips[seg["clip"]], out_fps) if seg.get("slowmo") else 1.0
        total += (seg["end"] - seg["start"]) * k
    for j in joins:
        total -= JOIN_SECONDS.get(j["type"], 0)
        if j["type"] == "ai_motion":
            total += AI_CLIP_SECONDS
    return total


# ---------------------------------------------------------------- rule-based editor
HOOK_WORDS = re.compile(r"\b(wait|watch|look|never|nobody|secret|crazy|insane|how|why|what|best|worst|first|last|stop|this is)\b", re.I)


def candidates(ci, clip, brief):
    a = clip["analysis"]
    dur = a["duration"]
    lo, hi = VIBE_PACE.get(brief["vibe"], (1.5, 5.0))
    out = []
    sentences = a["transcript"]["sentences"]
    if sentences:
        i = 0
        while i < len(sentences):  # merge very short sentences so a segment carries a full thought
            s, e, text = sentences[i]["s"], sentences[i]["e"], sentences[i]["text"]
            j = i + 1
            while e - s < lo and j < len(sentences) and sentences[j]["e"] - s <= hi * 1.6:
                e, text = sentences[j]["e"], text + " " + sentences[j]["text"]
                j += 1
            st = window_stats(a, s, e)
            rate = len(text.split()) / max(0.5, e - s)
            score = (0.25 * min(rate / 3.0, 1.0) + 0.2 * st["faces"] + 0.25 * st["interest"] / 10
                     + 0.15 * np.clip((st["loud"] + 40) / 30, 0, 1) + 0.15 * min(st["motion"] * 20, 1.0))
            if HOOK_WORDS.search(text) or "?" in text or "!" in text:
                score += 0.12
            out.append({"clip": ci, "start": s, "end": e, "score": float(score), "text": text, "slowmo": False})
            i = j
    if not sentences or a["face_ratio"] < 0.2:  # visual moments: action peaks
        slow = auto_slowmo(clip, brief["fps"])
        win = 1.5 if slow else (lo + hi) / 2
        t = 0.0
        while t + win <= dur + 1e-6:
            st = window_stats(a, t, t + win)
            score = (0.35 * min(st["motion"] * 25, 1.0) + 0.3 * st["interest"] / 10 + 0.15 * min(st["sharp"] / 300, 1.0)
                     + 0.2 * np.clip((st["loud"] + 40) / 30, 0, 1))
            if sentences:
                score *= 0.8
            out.append({"clip": ci, "start": t, "end": t + win, "score": float(score), "text": "", "slowmo": slow})
            t += max(0.5, win / 2)
        if not out and dur > 0.6:
            out.append({"clip": ci, "start": 0.0, "end": min(dur, win), "score": 0.1, "text": "", "slowmo": False})
    return out


def _overlaps(c, chosen, pad=0.3):
    return any(c["clip"] == o["clip"] and c["start"] < o["end"] + pad and o["start"] < c["end"] + pad for o in chosen)


def heuristic_plan(clips, brief):
    target = brief["length"]
    pools = [sorted(candidates(i, c, brief), key=lambda x: -x["score"]) for i, c in enumerate(clips)]
    chosen = []
    for pool in pools:  # every clip gets its best moment first, so the short uses all the footage
        if pool:
            chosen.append(pool[0])
    rest = sorted((c for pool in pools for c in pool[1:]), key=lambda x: -x["score"])
    for c in rest:
        if timeline_seconds(chosen, [], clips, brief["fps"]) >= target:
            break
        if not _overlaps(c, chosen):
            chosen.append(c)
    hook = max(chosen, key=lambda c: c["score"])
    others = [c for c in chosen if c is not hook]
    if brief["vibe"] in ("hype", "funny", "product"):
        by_clip = {}
        for c in sorted(others, key=lambda c: (c["clip"], c["start"])):
            by_clip.setdefault(c["clip"], []).append(c)
        order = []
        while any(by_clip.values()):  # alternate between clips for pace
            for k in sorted(by_clip):
                if by_clip[k]:
                    order.append(by_clip[k].pop(0))
    else:
        order = sorted(others, key=lambda c: (c["clip"], c["start"]))
    segments = [hook] + order
    joins = default_joins(segments, brief)
    hook_text = ""
    if hook["text"]:
        hook_text = " ".join(hook["text"].split()[:7]).strip(" ,.;:").upper()
    return {"title": "", "hook_text": hook_text, "post_caption": "", "hashtags": [], "music_mood": "",
            "segments": [{k: c[k] for k in ("clip", "start", "end", "slowmo")} | {"reason": "rule-based pick", "emphasis": []}
                         for c in segments],
            "joins": joins, "source": "rules"}


def default_joins(segments, brief):
    pattern = VIBE_JOINS.get(brief["vibe"], ["cut"])
    moves = CAMERA_MOVES[:10]
    joins = []
    for i in range(len(segments) - 1):
        t = pattern[i % len(pattern)]
        if brief["ai_transitions"] and segments[i]["clip"] != segments[i + 1]["clip"] and brief["vibe"] in ("cinematic", "emotional", "product", "story"):
            t = "ai_motion"
        joins.append({"type": t, "camera_move": moves[i % len(moves)], "effect": "none"})
    return joins


# ---------------------------------------------------------------- LLM editor
def _digest(i, clip, budget):
    a, meta = clip["analysis"], clip["meta"]
    head = f'CLIP {i}: "{clip["name"]}", {a["duration"]:.1f}s, {meta["fps"]:.0f} fps'
    if auto_slowmo(clip, 30):
        head += " (HIGH FRAME RATE: can be slowed down smoothly)"
    if clip.get("note"):
        head += f'. Creator note: {clip["note"]}'
    lines = [head]
    vis = a.get("vision", [])
    for s, e in a["scenes"]:
        descs = [f'{v["description"]} [{v["shot"]}, interest {v["interest"]}/10]' for v in vis if s <= v["t"] < e]
        lines.append(f"  scene {s:.1f}-{e:.1f}s: " + (" | ".join(descs[:3]) if descs else "(no description)"))
    for sen in a["transcript"]["sentences"]:
        lines.append(f'  speech {sen["s"]:.1f}-{sen["e"]:.1f}s: "{sen["text"]}"')
    en = np.asarray(a["energy"] or [-60.0])
    mo = np.asarray(a["scan"]["motion"] or [0.0])
    if en.size > 10:
        peaks = sorted(np.argsort(en)[-4:] / 10.0)
        lines.append("  loudest moments at: " + ", ".join(f"{p:.1f}s" for p in peaks))
    if mo.size > 10:
        peaks = sorted(np.argsort(mo)[-4:] / a["scan"]["fps"])
        lines.append("  most movement at: " + ", ".join(f"{p:.1f}s" for p in peaks))
    text = "\n".join(lines)
    return text if len(text) <= budget else text[:budget] + "\n  (truncated)"


def plan_schema(effects):
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "hook_text": {"type": "string"},
            "post_caption": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "music_mood": {"type": "string"},
            "segments": {"type": "array", "items": {"type": "object", "properties": {
                "clip": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"},
                "slowmo": {"type": "boolean"}, "reason": {"type": "string"},
                "emphasis": {"type": "array", "items": {"type": "string"}},
                "transition_in": {"type": "string", "enum": JOINS},
                "camera_move": {"type": "string", "enum": CAMERA_MOVES},
                "effect": {"type": "string", "enum": EFFECTS[effects]},
            }, "required": ["clip", "start", "end"]}},
        },
        "required": ["title", "hook_text", "segments"],
    }


SYSTEM = """You are a senior short-form video editor who makes vertical TikTok/Reels/Shorts that people watch to the end.
Rules you always follow:
- HOOK: the first segment must grab attention in the first second: the most surprising line, a bold claim, a question, or the peak action. Never open with greetings, intros or setup.
- PACE: change shots often enough to keep attention; each segment is one complete thought or one visual beat.
- Cut on sentence boundaries: start and end times must not split words. Use the speech timings given.
- STORY: after the hook, build toward a payoff; end on a strong line or visual that makes people rewatch (loopable if possible).
- Never reuse the same seconds of footage twice.
- slowmo=true only for clips marked HIGH FRAME RATE, only for action moments without important speech.
- transition_in describes the join INTO a segment (ignored for the first). ai_motion = an AI-generated camera move continuing the previous shot's final frame; use it sparingly for big changes of place or mood.
- camera_move and effect describe ONLY camera motion or a visual effect. Never invent new people, objects or text.
- hook_text: 2-7 punchy words shown on screen at the start. title: a short title. post_caption: one line for the post. hashtags: 3-6 relevant tags without '#'.
Answer only with JSON."""


def llm_plan(clips, brief, log):
    budget = max(1500, 22000 // max(1, len(clips)))
    digests = "\n\n".join(_digest(i, c, budget) for i, c in enumerate(clips))
    lo, hi = VIBE_PACE.get(brief["vibe"], (1.5, 5.0))
    effects_note = {"camera": "effect must be 'none' (camera moves only).",
                    "atmosphere": "effect may add subtle light/atmosphere.",
                    "fantastical": "effect may add a fantastical visual element."}[brief["effects"]]
    user = (f"Make a {brief['length']}-second vertical short. Vibe: {brief['vibe']}. "
            f"Typical segment length for this vibe: {lo}-{hi}s (slow-motion segments count {int(MAX_SLOWMO)}x longer on screen). "
            + (f"Creator's direction: {brief['prompt']}\n" if brief["prompt"] else "\n")
            + (f"Use ai_motion for at most {config.COMFY_MAX_CLIPS} transitions. " if brief["ai_transitions"] else "Do not use ai_motion. ")
            + effects_note + "\n\nFOOTAGE:\n" + digests +
            "\n\nReturn JSON: title, hook_text, post_caption, hashtags, music_mood, and segments in playback order "
            "(each: clip, start, end, slowmo, reason, emphasis (1-2 key words said in it), transition_in, camera_move, effect).")
    raw = llm.chat_json(config.TEXT_MODEL, SYSTEM, user, schema=plan_schema(brief["effects"]), temperature=0.6)
    segs, joins = [], []
    for k, s in enumerate(raw.get("segments") or []):
        try:
            seg = {"clip": int(s["clip"]), "start": float(s["start"]), "end": float(s["end"]), "slowmo": bool(s.get("slowmo")),
                   "reason": str(s.get("reason", ""))[:200], "emphasis": [str(w)[:30] for w in (s.get("emphasis") or [])][:3]}
        except (KeyError, TypeError, ValueError):
            continue
        if segs:
            t = s.get("transition_in") if s.get("transition_in") in JOINS else "cut"
            move = s.get("camera_move") if s.get("camera_move") in CAMERA_MOVES else CAMERA_MOVES[len(joins) % len(CAMERA_MOVES)]
            eff = s.get("effect") if s.get("effect") in EFFECTS[brief["effects"]] else "none"
            joins.append({"type": t, "camera_move": move, "effect": eff})
        segs.append(seg)
    tags = [re.sub(r"[^\w]", "", str(t)).lower() for t in (raw.get("hashtags") or [])][:8]
    return {"title": str(raw.get("title", ""))[:120], "hook_text": str(raw.get("hook_text", ""))[:60],
            "post_caption": str(raw.get("post_caption", ""))[:300], "hashtags": [t for t in tags if t],
            "music_mood": str(raw.get("music_mood", ""))[:80], "segments": segs, "joins": joins, "source": "ollama"}


# ---------------------------------------------------------------- clean-up shared by both editors
def normalize(plan, clips, brief, log):
    out_fps = brief["fps"]
    segs, joins = [], []
    for idx, seg in enumerate(plan["segments"]):
        if not 0 <= seg["clip"] < len(clips):
            continue
        clip = clips[seg["clip"]]
        a = clip["analysis"]
        dur = a["duration"]
        s, e = max(0.0, min(seg["start"], dur)), max(0.0, min(seg["end"], dur))
        if e < s:
            s, e = e, s
        slow = bool(seg.get("slowmo")) and can_slowmo(clip, out_fps)
        if clip["slowmo"] == "on" and can_slowmo(clip, out_fps):
            slow = True
        if not slow:
            s, e = snap_to_words(s, e, _words(a), dur)
        min_len = 0.6 if slow else 1.0
        if e - s < min_len:
            e = min(dur, s + min_len)
            s = max(0.0, e - min_len)
        e = min(e, s + (8.0 if slow else 15.0))
        cand = {**seg, "start": round(s, 3), "end": round(e, 3), "slowmo": slow}
        clash = next((o for o in segs if o["clip"] == cand["clip"] and cand["start"] < o["end"] and o["start"] < cand["end"]), None)
        if clash:  # no reused footage: trim the overlap, or drop the segment if little is left
            if cand["start"] < clash["start"]:
                cand["end"] = clash["start"]
            else:
                cand["start"] = clash["end"]
            if cand["end"] - cand["start"] < min_len:
                continue
        if segs and idx - 1 < len(plan["joins"]):
            joins.append(dict(plan["joins"][idx - 1]))
        elif segs:
            joins.append({"type": "cut", "camera_move": CAMERA_MOVES[0], "effect": "none"})
        segs.append(cand)
    if not segs:
        raise ValueError("no usable segments")
    plan = {**plan, "segments": segs, "joins": joins[: len(segs) - 1]}

    # Length: trim from the end, or top up with the best unused moments.
    target = brief["length"]

    def has_speech(seg):
        return any(seg["start"] <= w[0] and w[1] <= seg["end"] for w in _words(clips[seg["clip"]]["analysis"]))

    while len(plan["segments"]) > 1 and timeline_seconds(plan["segments"], plan["joins"], clips, out_fps) > target * 1.15:
        # Prefer shortening a visual (no speech) shot over dropping a whole shot.
        excess = timeline_seconds(plan["segments"], plan["joins"], clips, out_fps) - target
        visual = sorted((s for s in plan["segments"] if not has_speech(s)), key=lambda s: s["start"] - s["end"])
        shrunk = False
        for seg in visual:
            k = slow_factor(clips[seg["clip"]], out_fps) if seg["slowmo"] else 1.0
            floor = 0.6 if seg["slowmo"] else 1.5
            room = (seg["end"] - seg["start"]) - floor
            if room > 0.3:
                cut = min(room, excess / k)
                seg["start"] = round(seg["start"] + cut / 2, 3)  # keep the middle of the action
                seg["end"] = round(seg["end"] - cut / 2, 3)
                shrunk = True
                break
        if not shrunk:
            plan["segments"].pop()
            plan["joins"] = plan["joins"][: len(plan["segments"]) - 1]
    if timeline_seconds(plan["segments"], plan["joins"], clips, out_fps) > target * 1.15:
        seg = plan["segments"][0]  # one long segment: shorten it, ending on a word boundary when possible
        k = slow_factor(clips[seg["clip"]], out_fps) if seg["slowmo"] else 1.0
        new_end = seg["start"] + target / k
        words = _words(clips[seg["clip"]]["analysis"])
        ends = [w[1] for w in words if seg["start"] + 1 < w[1] <= new_end]
        seg["end"] = round(ends[-1] + 0.12 if ends else new_end, 3)
    if timeline_seconds(plan["segments"], plan["joins"], clips, out_fps) < target * 0.7:
        pool = sorted((c for i, clip in enumerate(clips) for c in candidates(i, clip, brief)), key=lambda x: -x["score"])
        for c in pool:
            if timeline_seconds(plan["segments"], plan["joins"], clips, out_fps) >= target * 0.9:
                break
            if not _overlaps(c, plan["segments"]):
                plan["segments"].append({k: c[k] for k in ("clip", "start", "end", "slowmo")} | {"reason": "added to reach length", "emphasis": []})
                plan["joins"].append(default_joins(plan["segments"][-2:], brief)[0])

    ai = 0
    for j in plan["joins"]:
        if j["type"] == "ai_motion":
            ai += 1
            if not brief["ai_transitions"] or ai > config.COMFY_MAX_CLIPS:
                j["type"] = "crossfade"
        if brief["effects"] == "camera":
            j["effect"] = "none"
    plan["seconds"] = round(timeline_seconds(plan["segments"], plan["joins"], clips, out_fps), 2)
    return plan


def make_plan(clips, brief, log):
    """clips: list of {name, note, slowmo, meta, analysis}. Returns a normalized plan."""
    plan = None
    try:
        plan = llm_plan(clips, brief, log)
        if not plan["segments"]:
            raise ValueError("the model returned no segments")
        plan = normalize(plan, clips, brief, log)
        log(f"Edit planned by {config.TEXT_MODEL}: {len(plan['segments'])} segments, {plan['seconds']:.1f}s")
    except (llm.LLMError, ValueError, KeyError, TypeError) as e:
        log(f"AI editor unavailable ({str(e)[:160]}); using the rule-based editor")
        plan = normalize(heuristic_plan(clips, brief), clips, brief, log)
        log(f"Rule-based edit: {len(plan['segments'])} segments, {plan['seconds']:.1f}s")
    return plan
