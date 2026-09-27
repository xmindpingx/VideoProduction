"""Whole pipeline on synthetic clips, with the mock Ollama API standing in for the models and a fixed transcript
standing in for Whisper (whose models are not downloaded in tests)."""
import json
import shutil

import pytest

import mock_ollama
from shorts import config, db, media, worker
from shorts.pipeline import analyze


@pytest.fixture()
def ollama(monkeypatch):
    srv = mock_ollama.start()
    monkeypatch.setattr(config, "OLLAMA_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield mock_ollama.CALLS
    srv.shutdown()


@pytest.fixture()
def fake_whisper(monkeypatch):
    words = [[0.4, 0.8, "Welcome"], [0.85, 1.0, "to"], [1.05, 1.2, "the"], [1.25, 1.8, "studio."],
             [2.2, 2.4, "This"], [2.45, 2.6, "is"], [2.65, 2.8, "the"], [2.85, 3.3, "moment"]]

    def transcribe(wav, log):
        return {"language": "en", "sentences": [{"s": 0.4, "e": 1.8, "text": "Welcome to the studio.", "words": words[:4]},
                                                {"s": 2.2, "e": 3.3, "text": "This is the moment", "words": words[4:]}]}
    monkeypatch.setattr(analyze, "transcribe", transcribe)


def add(path, kind="video", slowmo="auto"):
    u = db.create_upload(kind, path.name, path.stat().st_size, path.suffix.lower())
    d = config.UPLOADS_DIR / u["id"]
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, d / ("source" + path.suffix.lower()))
    db.update_upload(u["id"], status="ready", received=u["size"], meta=media.probe(d / ("source" + path.suffix.lower())), slowmo=slowmo)
    return u["id"]


def test_full_short(assets, clips, ollama, fake_whisper):
    db.init()
    ids = [add(clips / "talk60.mp4"), add(clips / "slomo240.mov"), add(clips / "hdr_portrait.mov")]
    music = add(clips / "beat120.mp3", kind="music")
    brief = {"vibe": "cinematic", "length": 15, "prompt": "", "captions": "pop", "look": "film", "fps": 24, "ai_transitions": True,
             "effects": "camera", "music_mode": "mix", "clean_audio": True, "punch_in": True, "hook_title": True}
    job = db.create_job(ids, music, brief)
    worker.run_job(db.claim_next_job())
    j = db.get_job(job["id"])
    assert j["status"] == "done", j["error"] or j["log"]
    r = j["result"]
    assert r["planner"] == "ollama" and r["title"] == "Mock title"
    assert r["hashtags"] == ["studio", "shorts"]
    assert r["words"] > 0
    assert r["transitions"]["still"] == 1  # the planned ai_motion join, rendered without ComfyUI

    out = config.JOBS_DIR / job["id"] / "out"
    m = media.probe(out / "short.mp4")
    assert (m["width"], m["height"]) == (1080, 1920)
    assert round(m["fps"]) == 24 and m["has_audio"]
    assert 8 < m["duration"] < 20
    edit = json.loads((out / "edit.json").read_text())
    assert any(t.get("slowmo") == 4.0 for t in edit["timeline"])  # the 240 fps clip was slowed down
    assert (out / "captions.srt").read_text().count("-->") >= 2
    assert any(p == "/api/generate" for p, _ in ollama)  # models unloaded to free the GPU
