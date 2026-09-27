"""HTTP API + the mobile page. Run: uvicorn shorts.api:app --host 0.0.0.0 --port 8590 --proxy-headers"""
import hmac
import json
import shutil
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__, comfy, config, db, media


@asynccontextmanager
async def lifespan(_app):
    if len(config.TOKEN) < 16:
        raise RuntimeError("SHORTS_TOKEN must be set to at least 16 characters (see .env.example)")
    db.init()
    yield


app = FastAPI(title="Shorts Lab", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".webm", ".mkv", ".3gp"}
MUSIC_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mov"}
COOKIE = "shorts_session"

VIBES = {"hype", "cinematic", "funny", "emotional", "story", "product"}
CAPTIONS = {"pop", "karaoke", "clean", "off"}
LOOKS = {"off", "subtle", "film"}
EFFECTS = {"camera", "atmosphere", "fantastical"}
MUSIC_MODES = {"mix", "music_only", "original_only"}


# ---------------------------------------------------------------- auth
_fails = defaultdict(deque)


def _client_ip(request: Request):
    return (request.headers.get("cf-connecting-ip")
            or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
            or (request.client.host if request.client else "?"))


def _auth_state(request: Request):
    """'ok', 'none' (not signed in) or 'csrf' (cookie present but the request lacks our header)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), config.TOKEN.encode()):
        return "ok"
    if db.session_valid(request.cookies.get(COOKIE)):
        # Cookie auth on a state-changing request must carry our header: a cross-site form cannot set it.
        if request.method not in ("GET", "HEAD") and request.headers.get("x-shorts") != "1":
            return "csrf"
        return "ok"
    return "none"


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in ("/api/login", "/api/ping"):
        state = await run_in_threadpool(_auth_state, request)
        if state == "none":
            return JSONResponse({"error": "Sign in first"}, status_code=401)
        if state == "csrf":
            return JSONResponse({"error": "Missing X-Shorts header"}, status_code=403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _secure(request: Request):
    if config.COOKIE_SECURE in ("true", "false"):
        return config.COOKIE_SECURE == "true"
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


@app.get("/api/ping")
def ping():
    return {"ok": True}


@app.post("/api/login")
async def login(request: Request, response: Response):
    ip, now = _client_ip(request), time.time()
    q, everyone = _fails[ip], _fails["*"]  # per address, plus a cap for all addresses (forwarded IPs can be spoofed)
    for d in (q, everyone):
        while d and now - d[0] > 600:
            d.popleft()
    if len(q) >= 8 or len(everyone) >= 40:
        raise HTTPException(429, "Too many tries. Wait 10 minutes.")
    try:
        body = await request.json()
    except ValueError:
        body = {}
    token = str(body.get("token", ""))
    if not hmac.compare_digest(token.encode(), config.TOKEN.encode()):
        q.append(now)
        everyone.append(now)
        raise HTTPException(401, "That access code is not right.")
    sid = await run_in_threadpool(db.create_session)
    response.set_cookie(COOKIE, sid, max_age=config.SESSION_DAYS * 86400, httponly=True, samesite="strict",
                        secure=_secure(request), path="/")
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    db.delete_session(request.cookies.get(COOKIE) or "")
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me():
    return {"ok": True, "version": __version__}


# ---------------------------------------------------------------- health
def _ollama_status():
    from . import llm
    base, _ = config.ollama_base()
    try:
        names = llm.list_models()
    except Exception as e:  # noqa: BLE001 - any connection problem is just "offline"
        return {"ok": False, "url": base, "error": str(e)[:200]}
    return {"ok": True, "url": base, "via_openwebui": base.endswith("/ollama"),
            "text_model": config.TEXT_MODEL, "text_ok": llm.has_model(names, config.TEXT_MODEL),
            "vision_model": config.VISION_MODEL, "vision_ok": llm.has_model(names, config.VISION_MODEL)}


@app.get("/api/health")
def health():
    du = shutil.disk_usage(config.DATA_DIR)
    return {
        "version": __version__,
        "ollama": _ollama_status(),
        "comfyui": comfy.status(),
        "whisper": {"engine": config.WHISPER_ENGINE, "model": config.WHISPER_MODEL, "device": config.WHISPER_DEVICE},
        "gpu": _gpu_status(),
        "audio_cleanup": config.AUDIO_CLEANUP,
        "disk": {"free_gb": round(du.free / 1024**3, 1), "total_gb": round(du.total / 1024**3, 1)},
        "limits": {"max_upload_bytes": config.MAX_UPLOAD_BYTES, "chunk_bytes": config.CHUNK_BYTES},
        "worker": _worker_status(),
    }


def _gpu_status():
    try:
        return json.loads((config.DATA_DIR / "worker_gpu.json").read_text())
    except (OSError, ValueError):
        return {}


def _worker_status():
    beat = config.DATA_DIR / "worker.heartbeat"
    try:
        age = time.time() - beat.stat().st_mtime
    except OSError:
        return {"ok": False, "error": "Worker has not started"}
    return {"ok": age < 60, "last_seen_s": round(age)}


# ---------------------------------------------------------------- uploads
def _upload_dir(uid):
    return config.UPLOADS_DIR / uid


def _source(u):
    return _upload_dir(u["id"]) / ("source" + u["ext"])


def _public_upload(u):
    keys = ("id", "kind", "name", "size", "received", "status", "meta", "note", "slowmo", "analysis", "error", "created")
    return {k: u.get(k) for k in keys}


def _get_upload_or_404(uid):
    u = db.get_upload(uid)
    if not u:
        raise HTTPException(404, "No such upload")
    return u


@app.post("/api/uploads")
async def create_upload(request: Request):
    body = await request.json()
    name = str(body.get("name", "clip")).replace("/", "_").replace("\\", "_")[:180] or "clip"
    kind = body.get("kind", "video")
    if kind not in ("video", "music"):
        raise HTTPException(400, "kind must be video or music")
    try:
        size = int(body.get("size", 0))
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        raise HTTPException(400, "Empty file")
    if size > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is bigger than the {config.MAX_UPLOAD_BYTES // 1024**3} GB limit")
    ext = Path(name).suffix.lower()
    allowed = VIDEO_EXT if kind == "video" else MUSIC_EXT
    if ext not in allowed:
        raise HTTPException(415, f"Unsupported file type {ext or '(none)'}")
    du = shutil.disk_usage(config.DATA_DIR)
    if du.free < size * 3 + 2 * 1024**3:  # source + working files + headroom
        raise HTTPException(507, "Not enough free disk space on the server for this file")
    u = await run_in_threadpool(db.create_upload, kind, name, size, ext)
    _upload_dir(u["id"]).mkdir(parents=True, exist_ok=True)
    _source(u).touch()
    return {**_public_upload(u), "chunk_size": config.CHUNK_BYTES}


@app.get("/api/uploads")
def list_uploads():
    return [_public_upload(u) for u in db.list_uploads()]


@app.get("/api/uploads/{uid}")
def get_upload(uid: str):
    return _public_upload(_get_upload_or_404(uid))


@app.put("/api/uploads/{uid}")
async def put_chunk(uid: str, offset: int, request: Request):
    u = await run_in_threadpool(_get_upload_or_404, uid)
    if u["status"] != "uploading":
        raise HTTPException(409, "Upload already finished")
    if offset != u["received"]:
        return JSONResponse({"error": "offset mismatch", "received": u["received"]}, status_code=409)
    limit = min(config.CHUNK_BYTES * 2, u["size"] - offset)
    got = 0
    path = _source(u)
    with open(path, "r+b") as f:
        f.seek(offset)
        f.truncate()
        async for part in request.stream():
            got += len(part)
            if got > limit:
                f.truncate(offset)
                raise HTTPException(413, "Chunk too large")
            f.write(part)
    await run_in_threadpool(db.update_upload, uid, received=offset + got)
    return {"received": offset + got}


def _finish_upload(u):
    path = _source(u)
    with open(path, "rb") as f:
        head = f.read(16)
    if not media.magic_ok(head, u["kind"]):
        raise ValueError("This does not look like a video/audio file")
    meta = media.probe(path)
    if u["kind"] == "video" and not meta.get("has_video"):
        raise ValueError("No video track found")
    if u["kind"] == "music" and not meta.get("has_audio"):
        raise ValueError("No audio track found")
    if meta.get("duration", 0) < 0.5:
        raise ValueError("Clip is too short")
    if u["kind"] == "video":
        try:
            media.thumbnail(path, meta, _upload_dir(u["id"]) / "thumb.jpg")
        except media.MediaError:
            pass
    return meta


@app.post("/api/uploads/{uid}/complete")
async def complete_upload(uid: str):
    u = await run_in_threadpool(_get_upload_or_404, uid)
    if u["status"] == "ready":
        return _public_upload(u)
    if u["received"] != u["size"]:
        raise HTTPException(409, f"Upload incomplete: {u['received']} of {u['size']} bytes")
    try:
        meta = await run_in_threadpool(_finish_upload, u)
    except (ValueError, media.MediaError) as e:
        await run_in_threadpool(db.update_upload, uid, status="error", error=str(e)[:300])
        raise HTTPException(422, str(e)[:300])
    await run_in_threadpool(db.update_upload, uid, status="ready", meta=meta, error=None)
    return _public_upload(db.get_upload(uid))


@app.patch("/api/uploads/{uid}")
async def patch_upload(uid: str, request: Request):
    _get_upload_or_404(uid)
    body = await request.json()
    fields = {}
    if "note" in body:
        fields["note"] = str(body["note"])[:500]
    if "slowmo" in body:
        if body["slowmo"] not in ("auto", "on", "off"):
            raise HTTPException(400, "slowmo must be auto, on or off")
        fields["slowmo"] = body["slowmo"]
    if fields:
        db.update_upload(uid, **fields)
    return _public_upload(db.get_upload(uid))


@app.delete("/api/uploads/{uid}")
def delete_upload(uid: str):
    _get_upload_or_404(uid)
    if db.jobs_using_upload(uid):
        raise HTTPException(409, "A short that uses this clip is still being made")
    shutil.rmtree(_upload_dir(uid), ignore_errors=True)
    db.delete_upload(uid)
    return {"ok": True}


@app.get("/api/uploads/{uid}/thumb.jpg")
def upload_thumb(uid: str):
    p = _upload_dir(uid) / "thumb.jpg"
    if not p.is_file():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(p, media_type="image/jpeg")


# ---------------------------------------------------------------- jobs
def _clean_brief(b):
    def pick(key, allowed, default):
        v = b.get(key, default)
        return v if v in allowed else default

    try:
        length = int(b.get("length", 30))
    except (TypeError, ValueError):
        length = 30
    try:
        fps = int(b.get("fps", 30))
    except (TypeError, ValueError):
        fps = 30
    return {
        "vibe": pick("vibe", VIBES, "hype"),
        "length": min(max(length, 8), 90),
        "prompt": str(b.get("prompt", ""))[:800],
        "captions": pick("captions", CAPTIONS, "pop"),
        "look": pick("look", LOOKS, "subtle"),
        "fps": fps if fps in (24, 30, 60) else 30,
        "ai_transitions": bool(b.get("ai_transitions", True)),
        "effects": pick("effects", EFFECTS, "camera"),
        "music_mode": pick("music_mode", MUSIC_MODES, "mix"),
        "clean_audio": bool(b.get("clean_audio", True)),
        "punch_in": bool(b.get("punch_in", True)),
        "hook_title": bool(b.get("hook_title", True)),
    }


def _public_job(j, log=False):
    d = {k: j.get(k) for k in ("id", "status", "stage", "progress", "upload_ids", "music_id", "brief", "result", "error", "created", "started", "finished")}
    if log:
        d["log"] = j.get("log", "")[-8000:]
    return d


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    ids = body.get("upload_ids") or []
    if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
        raise HTTPException(400, "Pick between 1 and 20 clips")
    for uid in ids:
        u = db.get_upload(str(uid))
        if not u or u["kind"] != "video" or u["status"] != "ready":
            raise HTTPException(400, "Every clip must be fully uploaded first")
    music_id = body.get("music_id") or None
    if music_id:
        m = db.get_upload(str(music_id))
        if not m or m["kind"] != "music" or m["status"] != "ready":
            raise HTTPException(400, "The music file is not ready")
    job = db.create_job([str(i) for i in ids], music_id, _clean_brief(body.get("brief") or {}))
    return _public_job(job)


@app.get("/api/jobs")
def list_jobs():
    return [_public_job(j) for j in db.list_jobs()]


def _get_job_or_404(jid):
    j = db.get_job(jid)
    if not j:
        raise HTTPException(404, "No such job")
    return j


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _public_job(_get_job_or_404(jid), log=True)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    j = _get_job_or_404(jid)
    if j["status"] == "queued":
        db.update_job(jid, status="canceled", finished=time.time())
    elif j["status"] == "running":
        db.update_job(jid, cancel=1)
    return {"ok": True}


@app.post("/api/jobs/{jid}/remix")
async def remix_job(jid: str, request: Request):
    j = _get_job_or_404(jid)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    brief = {**(j["brief"] or {}), **(body.get("brief") or {})}
    for uid in j["upload_ids"]:
        if not db.get_upload(uid):
            raise HTTPException(409, "A clip from that short was deleted")
    music_id = body.get("music_id", j["music_id"])
    if music_id and not db.get_upload(music_id):
        music_id = None
    return _public_job(db.create_job(j["upload_ids"], music_id, _clean_brief(brief)))


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    j = _get_job_or_404(jid)
    if j["status"] in ("queued", "running"):
        raise HTTPException(409, "Cancel it first")
    shutil.rmtree(config.JOBS_DIR / jid, ignore_errors=True)
    db.delete_job(jid)
    return {"ok": True}


JOB_FILES = {"short.mp4": "video/mp4", "cover.jpg": "image/jpeg", "captions.srt": "application/x-subrip", "edit.json": "application/json"}


@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid: str, name: str, download: int = 0):
    if name not in JOB_FILES:
        raise HTTPException(404, "No such file")
    _get_job_or_404(jid)
    p = config.JOBS_DIR / jid / "out" / name
    if not p.is_file():
        raise HTTPException(404, "Not made yet")
    filename = f"short-{jid}{Path(name).suffix}" if download else None
    return FileResponse(p, media_type=JOB_FILES[name], filename=filename)


# ---------------------------------------------------------------- page
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
