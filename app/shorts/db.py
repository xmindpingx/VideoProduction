"""SQLite store shared by the API and the worker (WAL mode, one short-lived connection per call)."""
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,              -- video | music
  name TEXT NOT NULL,
  size INTEGER NOT NULL,
  received INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,            -- uploading | ready | error
  ext TEXT NOT NULL,
  meta TEXT,                       -- ffprobe summary (JSON)
  note TEXT NOT NULL DEFAULT '',
  slowmo TEXT NOT NULL DEFAULT 'auto',   -- auto | on | off
  analysis TEXT NOT NULL DEFAULT 'none', -- none | running | done | error
  error TEXT,
  created REAL NOT NULL,
  updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,            -- queued | running | done | error | canceled
  stage TEXT NOT NULL DEFAULT '',
  progress REAL NOT NULL DEFAULT 0,
  upload_ids TEXT NOT NULL,
  music_id TEXT,
  brief TEXT NOT NULL,
  result TEXT,
  log TEXT NOT NULL DEFAULT '',
  error TEXT,
  cancel INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  started REAL,
  finished REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  expires REAL NOT NULL
);
"""


def connect():
    con = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    return con


@contextmanager
def conn():
    con = connect()
    try:
        yield con
    finally:
        con.close()


def init():
    config.ensure_dirs()
    with conn() as c:
        c.executescript(SCHEMA)


def new_id():
    return secrets.token_hex(8)


def _row(r, json_fields=()):
    if r is None:
        return None
    d = dict(r)
    for f in json_fields:
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except ValueError:
                d[f] = None
    return d


# ---------------------------------------------------------------- uploads
def create_upload(kind, name, size, ext):
    uid, now = new_id(), time.time()
    with conn() as c:
        c.execute("INSERT INTO uploads (id, kind, name, size, status, ext, created, updated) VALUES (?,?,?,?,?,?,?,?)",
                  (uid, kind, name, size, "uploading", ext, now, now))
    return get_upload(uid)


def get_upload(uid):
    with conn() as c:
        return _row(c.execute("SELECT * FROM uploads WHERE id=?", (uid,)).fetchone(), ("meta",))


def list_uploads():
    with conn() as c:
        return [_row(r, ("meta",)) for r in c.execute("SELECT * FROM uploads ORDER BY created DESC").fetchall()]


def update_upload(uid, **fields):
    if "meta" in fields and not isinstance(fields["meta"], (str, type(None))):
        fields["meta"] = json.dumps(fields["meta"])
    fields["updated"] = time.time()
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE uploads SET {keys} WHERE id=?", (*fields.values(), uid))


def delete_upload(uid):
    with conn() as c:
        c.execute("DELETE FROM uploads WHERE id=?", (uid,))


def next_unanalyzed_upload():
    with conn() as c:
        r = c.execute("SELECT id FROM uploads WHERE kind='video' AND status='ready' AND analysis='none' ORDER BY created LIMIT 1").fetchone()
        return r["id"] if r else None


# ---------------------------------------------------------------- jobs
def create_job(upload_ids, music_id, brief):
    jid, now = new_id(), time.time()
    with conn() as c:
        c.execute("INSERT INTO jobs (id, status, upload_ids, music_id, brief, created) VALUES (?,?,?,?,?,?)",
                  (jid, "queued", json.dumps(upload_ids), music_id, json.dumps(brief), now))
    return get_job(jid)


def get_job(jid):
    with conn() as c:
        return _row(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone(), ("upload_ids", "brief", "result"))


def list_jobs(limit=50):
    with conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r, ("upload_ids", "brief", "result")) for r in rows]


def update_job(jid, **fields):
    for f in ("result",):
        if f in fields and not isinstance(fields[f], (str, type(None))):
            fields[f] = json.dumps(fields[f])
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), jid))


def append_log(jid, line):
    stamp = time.strftime("%H:%M:%S")
    with conn() as c:
        # keep the stored log bounded
        c.execute("UPDATE jobs SET log = substr(log || ?, -60000) WHERE id=?", (f"[{stamp}] {line}\n", jid))


def claim_next_job():
    """Atomically move the oldest queued job to running and return it."""
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        r = c.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not r:
            c.execute("COMMIT")
            return None
        c.execute("UPDATE jobs SET status='running', started=?, stage='Starting', progress=0 WHERE id=?", (time.time(), r["id"]))
        c.execute("COMMIT")
    return get_job(r["id"])


def jobs_using_upload(uid):
    with conn() as c:
        rows = c.execute("SELECT id, upload_ids, music_id, status FROM jobs WHERE status IN ('queued','running')").fetchall()
    return [r["id"] for r in rows if uid in json.loads(r["upload_ids"]) or r["music_id"] == uid]


def delete_job(jid):
    with conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (jid,))


def fail_interrupted_jobs():
    with conn() as c:
        c.execute("UPDATE jobs SET status='error', error='Interrupted: the worker restarted. Press Remix to run it again.', finished=? "
                  "WHERE status='running'", (time.time(),))
        c.execute("UPDATE uploads SET analysis='none' WHERE analysis='running'")


# ---------------------------------------------------------------- sessions
def _sid_hash(sid):
    return hashlib.sha256(sid.encode()).hexdigest()  # only the hash is stored


def create_session():
    sid = secrets.token_urlsafe(32)
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        c.execute("INSERT INTO sessions (id, expires) VALUES (?,?)", (_sid_hash(sid), time.time() + config.SESSION_DAYS * 86400))
    return sid


def session_valid(sid):
    if not sid:
        return False
    with conn() as c:
        r = c.execute("SELECT expires FROM sessions WHERE id=?", (_sid_hash(sid),)).fetchone()
    return bool(r) and r["expires"] > time.time()


def delete_session(sid):
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE id=?", (_sid_hash(sid),))
