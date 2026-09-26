"""Background worker: makes queued shorts one at a time, pre-analyzes new uploads while idle, and prunes old files.
Run: python -m shorts.worker"""
import json
import shutil
import signal
import time
import traceback

from . import config, db, media
from .pipeline import analyze, render

_stop = False


def _on_term(*_):
    global _stop
    _stop = True


def _beat():
    (config.DATA_DIR / "worker.heartbeat").touch()


def run_job(job):
    jid = job["id"]
    last = {"t": 0.0, "cancel": False}

    def log(line):
        print(f"[{jid}] {line}", flush=True)
        db.append_log(jid, line)

    def progress(frac, stage=None):
        fields = {"progress": round(max(0.0, min(1.0, frac)), 3)}
        if stage:
            fields["stage"] = stage
            db.append_log(jid, stage)
        db.update_job(jid, **fields)
        _beat()

    def cancel():
        now = time.time()
        if now - last["t"] > 1.0:
            last["t"] = now
            j = db.get_job(jid)
            last["cancel"] = bool(j and j["cancel"]) or _stop
        return last["cancel"]

    log(f"Making a {job['brief']['length']}s {job['brief']['vibe']} short from {len(job['upload_ids'])} clip(s)")
    try:
        result = render.run(job, log, progress, cancel)
    except media.Canceled:
        if _stop:
            db.update_job(jid, status="error", error="Interrupted: the worker was stopped. Press Remix to run it again.",
                          stage="Interrupted", finished=time.time())
        else:
            db.update_job(jid, status="canceled", stage="Canceled", finished=time.time())
        log("Stopped")
        return
    except Exception as e:  # noqa: BLE001 - report every failure to the page
        traceback.print_exc()
        msg = str(e) or e.__class__.__name__
        db.update_job(jid, status="error", error=msg[:500], stage="Failed", finished=time.time())
        log("Failed: " + msg[:500])
        return
    db.update_job(jid, status="done", stage="Done", progress=1.0, result=result, finished=time.time())
    log(f"Done: {result['seconds']}s short")


def pre_analyze(uid):
    u = db.get_upload(uid)
    if not u:
        return
    db.update_upload(uid, analysis="running")
    try:
        analyze.analyze(u, log=lambda m: print(f"[upload {uid}] {m}", flush=True), cancel=lambda: _stop)
        db.update_upload(uid, analysis="done")
    except media.Canceled:
        db.update_upload(uid, analysis="none")
    except Exception as e:  # noqa: BLE001
        print(f"[upload {uid}] analysis failed: {e}", flush=True)
        db.update_upload(uid, analysis="error")


def prune():
    """Delete uploads and finished shorts older than RETENTION_DAYS (0 keeps everything)."""
    if config.RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - config.RETENTION_DAYS * 86400
    for j in db.list_jobs(limit=10000):
        if j["status"] not in ("queued", "running") and (j["finished"] or j["created"]) < cutoff:
            shutil.rmtree(config.JOBS_DIR / j["id"], ignore_errors=True)
            db.delete_job(j["id"])
    for u in db.list_uploads():
        if u["updated"] < cutoff and not db.jobs_using_upload(u["id"]):
            shutil.rmtree(config.UPLOADS_DIR / u["id"], ignore_errors=True)
            db.delete_upload(u["id"])


def _torch_gpu():
    """Name of the GPU PyTorch sees (AMD ROCm or NVIDIA), or why there is none."""
    try:
        import torch
    except ImportError:
        return "PyTorch not installed in this image"
    if not torch.cuda.is_available():
        return "no GPU visible to PyTorch"
    hip = getattr(torch.version, "hip", None)
    return f"{torch.cuda.get_device_name(0)} ({'ROCm ' + hip if hip else 'CUDA'})"


def main():
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    db.init()
    db.fail_interrupted_jobs()
    hw = media.hw_selftest(log=lambda m: print(m, flush=True))
    gpu = {"video": hw["note"], "video_enc": hw["enc"], "video_dec": hw["dec"], "torch": _torch_gpu()}
    (config.DATA_DIR / "worker_gpu.json").write_text(json.dumps(gpu))
    print(f"worker ready: {gpu}", flush=True)
    last_prune = 0.0
    while not _stop:
        _beat()
        job = db.claim_next_job()
        if job:
            run_job(job)
            continue
        uid = db.next_unanalyzed_upload()
        if uid:
            pre_analyze(uid)
            continue
        if time.time() - last_prune > 3600:
            last_prune = time.time()
            try:
                prune()
            except Exception as e:  # noqa: BLE001
                print(f"prune failed: {e}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
