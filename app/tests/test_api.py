"""API: sign-in, resumable chunked upload, probing, jobs, files."""
import pytest
from fastapi.testclient import TestClient

from shorts import api, config, db

H = {"X-Shorts": "1"}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def login(c):
    r = c.post("/api/login", json={"token": config.TOKEN})
    assert r.status_code == 200


def upload(c, path, kind="video", chunk=700_000):
    data = path.read_bytes()
    r = c.post("/api/uploads", json={"name": path.name, "size": len(data), "kind": kind}, headers=H)
    assert r.status_code == 200, r.text
    uid = r.json()["id"]
    off = 0
    while off < len(data):
        r = c.put(f"/api/uploads/{uid}?offset={off}", content=data[off:off + chunk], headers=H)
        assert r.status_code == 200, r.text
        off = r.json()["received"]
    r = c.post(f"/api/uploads/{uid}/complete", headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def test_requires_sign_in(client):
    client.cookies.clear()
    assert client.get("/api/uploads").status_code == 401
    assert client.post("/api/login", json={"token": "wrong-token-000000"}).status_code == 401
    assert client.get("/").status_code == 200  # the page itself is public; the API is not


def test_bearer_token(client):
    client.cookies.clear()
    r = client.get("/api/jobs", headers={"Authorization": "Bearer " + config.TOKEN})
    assert r.status_code == 200


def test_cookie_writes_need_header(client):
    login(client)
    r = client.post("/api/uploads", json={"name": "a.mov", "size": 10, "kind": "video"})
    assert r.status_code == 403


def test_chunked_upload_resume_and_probe(client, clips):
    login(client)
    data = (clips / "slomo240.mov").read_bytes()
    r = client.post("/api/uploads", json={"name": "slomo240.mov", "size": len(data), "kind": "video"}, headers=H)
    uid = r.json()["id"]
    half = len(data) // 2
    assert client.put(f"/api/uploads/{uid}?offset=0", content=data[:half], headers=H).json()["received"] == half
    # a retried chunk at the wrong offset is told where to resume
    r = client.put(f"/api/uploads/{uid}?offset=0", content=data[:10], headers=H)
    assert r.status_code == 409 and r.json()["received"] == half
    assert client.post(f"/api/uploads/{uid}/complete", headers=H).status_code == 409
    client.put(f"/api/uploads/{uid}?offset={half}", content=data[half:], headers=H)
    u = client.post(f"/api/uploads/{uid}/complete", headers=H).json()
    assert u["status"] == "ready"
    assert u["meta"]["high_fps"] is True and round(u["meta"]["fps"]) == 240
    assert client.get(f"/api/uploads/{uid}/thumb.jpg").headers["content-type"] == "image/jpeg"
    r = client.patch(f"/api/uploads/{uid}", json={"slowmo": "on", "note": "skate trick"}, headers=H)
    assert r.json()["slowmo"] == "on" and r.json()["note"] == "skate trick"


def test_rejects_non_video(client, tmp_path):
    login(client)
    p = tmp_path / "fake.mov"
    p.write_bytes(b"this is not a video at all" * 100)
    data = p.read_bytes()
    uid = client.post("/api/uploads", json={"name": p.name, "size": len(data), "kind": "video"}, headers=H).json()["id"]
    client.put(f"/api/uploads/{uid}?offset=0", content=data, headers=H)
    r = client.post(f"/api/uploads/{uid}/complete", headers=H)
    assert r.status_code == 422
    assert client.post("/api/uploads", json={"name": "x.exe", "size": 5, "kind": "video"}, headers=H).status_code == 415


def test_hdr_portrait_probe(client, clips):
    login(client)
    u = upload(client, clips / "hdr_portrait.mov")
    assert (u["meta"]["width"], u["meta"]["height"]) == (1080, 1920)
    assert u["meta"]["hdr"] == "hlg"


def test_jobs_lifecycle(client, clips):
    login(client)
    u = upload(client, clips / "talk60.mp4")
    m = upload(client, clips / "beat120.mp3", kind="music")
    r = client.post("/api/jobs", json={"upload_ids": [u["id"]], "music_id": m["id"], "brief": {"vibe": "nope", "length": 500, "fps": 25}},
                    headers=H)
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "queued"
    assert job["brief"]["vibe"] == "hype" and job["brief"]["length"] == 90 and job["brief"]["fps"] == 30  # cleaned
    assert client.delete(f"/api/uploads/{u['id']}", headers=H).status_code == 409  # in use
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=H).status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "canceled"
    remix = client.post(f"/api/jobs/{job['id']}/remix", json={"brief": {"vibe": "cinematic"}}, headers=H).json()
    assert remix["brief"]["vibe"] == "cinematic" and remix["music_id"] == m["id"]
    db.update_job(remix["id"], status="canceled")
    assert client.get(f"/api/jobs/{job['id']}/files/short.mp4").status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}", headers=H).status_code == 200


def test_health_shape(client):
    login(client)
    h = client.get("/api/health").json()
    assert h["ollama"]["ok"] is False  # nothing listens in the test
    assert h["comfyui"] == {"enabled": False}
    assert "free_gb" in h["disk"]
