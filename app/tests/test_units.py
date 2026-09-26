"""Pure logic: captions, the ComfyUI workflow inspector, the edit-plan clean-up, beat snapping, file sniffing."""
from shorts import comfy, media
from shorts.pipeline import captions, director, render


def fake_clip(duration=20.0, fps=30.0, words=(), slowmo="auto", face_ratio=0.8):
    sentences = []
    if words:
        sentences = [{"s": words[0][0], "e": words[-1][1], "text": " ".join(w[2] for w in words), "words": [list(w) for w in words]}]
    n = int(duration * 10)
    return {"name": "c.mov", "note": "", "slowmo": slowmo, "meta": {"fps": fps, "duration": duration},
            "analysis": {"duration": duration, "transcript": {"sentences": sentences}, "face_ratio": face_ratio,
                         "scan": {"fps": 10, "motion": [0.01] * n, "sharp": [100.0] * n, "faces": [[]] * n, "mcx": [0.5] * n,
                                  "mcy": [0.5] * n, "cut": [0.0] * n, "bright": [0.5] * n},
                         "energy": [-20.0] * n, "vision": [], "scenes": [[0.0, duration]]}}


BRIEF = {"vibe": "hype", "length": 15, "prompt": "", "captions": "pop", "look": "subtle", "fps": 30, "ai_transitions": True,
         "effects": "camera", "music_mode": "mix", "clean_audio": True, "punch_in": True, "hook_title": True}


def test_cuts_never_split_words():
    words = [(1.0, 1.4, "hello"), (1.5, 2.2, "world"), (2.3, 3.0, "again")]
    plan = {"segments": [{"clip": 0, "start": 1.2, "end": 2.0, "slowmo": False}], "joins": []}
    out = director.normalize(plan, [fake_clip(words=words)], {**BRIEF, "length": 2}, print)
    seg = out["segments"][0]
    assert seg["start"] <= 1.0 and seg["end"] >= 2.2


def test_no_reused_footage_and_valid_slowmo():
    plan = {"segments": [{"clip": 0, "start": 2, "end": 6, "slowmo": True}, {"clip": 0, "start": 4, "end": 9, "slowmo": False}],
            "joins": [{"type": "ai_motion", "camera_move": "orbit left", "effect": "embers"}]}
    out = director.normalize(plan, [fake_clip(fps=30)], {**BRIEF, "length": 8}, print)
    a, b = out["segments"]
    assert a["slowmo"] is False  # a 30 fps clip has no frames to slow down
    assert b["start"] >= a["end"]
    assert out["joins"][0]["effect"] == "none"  # "camera only" brief strips effects


def test_high_fps_clip_gets_slowmo():
    plan = {"segments": [{"clip": 0, "start": 1, "end": 2.5, "slowmo": True}], "joins": []}
    out = director.normalize(plan, [fake_clip(fps=240)], {**BRIEF, "length": 6}, print)
    assert out["segments"][0]["slowmo"] is True
    assert director.slow_factor(fake_clip(fps=240), 30) == 4.0


def test_rule_based_plan_uses_every_clip():
    clips = [fake_clip(words=[(1, 1.5, "wait"), (1.6, 3.0, "what?")]), fake_clip(fps=240, face_ratio=0.0), fake_clip(face_ratio=0.0)]
    plan = director.normalize(director.heuristic_plan(clips, BRIEF), clips, BRIEF, print)
    assert {s["clip"] for s in plan["segments"]} == {0, 1, 2}
    assert len(plan["joins"]) == len(plan["segments"]) - 1
    assert plan["seconds"] <= BRIEF["length"] * 1.15


def test_captions_highlight_each_word(tmp_path):
    words = [(0.5, 0.8, "This"), (0.85, 1.0, "is"), (1.05, 1.5, "wild!"), (2.4, 2.9, "Next")]
    n = captions.build(words, "pop", "Watch this", 2.0, 5.0, ["wild"], tmp_path / "c.ass", tmp_path / "c.srt")
    ass = (tmp_path / "c.ass").read_text()
    assert n == 1 + 4  # hook + one event per word
    assert "WILD!" in ass and captions.GOLD in ass
    assert "Style: Cap,Anton" in ass
    srt = (tmp_path / "c.srt").read_text()
    assert "00:00:00,500 --> " in srt and "NEXT" in srt


def test_comfy_inspect_follows_conditioning():
    wf = {  # shape of a Wan image-to-video workflow exported with "Export (API)"
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["38", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["38", 0]}},
        "52": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "50": {"class_type": "WanImageToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "width": 832, "height": 480,
                                                           "start_image": ["52", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["50", 0], "negative": ["50", 1], "seed": 1}},
    }
    info = comfy.inspect(wf)
    assert info["positive"] == ("6", "text")
    assert info["negative"] == ("7", "text")
    assert info["image_nodes"] == ["52"]


def test_beat_snap_moves_cut_onto_beat():
    clip = fake_clip(duration=20)
    plan = {"segments": [{"clip": 0, "start": 0.0, "end": 3.1, "slowmo": False}, {"clip": 0, "start": 5, "end": 8, "slowmo": False}],
            "joins": [{"type": "cut"}]}
    render.snap_to_beats(plan, [clip], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 30, print)
    assert abs(plan["segments"][0]["end"] - 3.0) < 1e-6


def test_magic_bytes():
    assert media.magic_ok(b"\x00\x00\x00\x14ftypqt  ", "video")
    assert media.magic_ok(b"ID3\x04\x00\x00\x00\x00", "music")
    assert not media.magic_ok(b"MZ\x90\x00\x03\x00\x00\x00", "video")
