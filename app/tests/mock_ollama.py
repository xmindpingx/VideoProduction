"""A stand-in for Ollama's HTTP API (/api/tags, /api/chat, /api/generate) for tests. It answers vision prompts with a
fixed frame description and edit-plan prompts with a plan built from the clip list in the prompt. It proves our client
and the plan clean-up work; it says nothing about how well a real model edits.

Run standalone:  python tests/mock_ollama.py 11434
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS = []


def plan_for(prompt):
    clips = [(int(i), float(d), "HIGH FRAME RATE" in line)
             for line in prompt.splitlines()
             for i, d in re.findall(r'^CLIP (\d+): ".*?", ([\d.]+)s', line)]
    segs = []
    moves = ["orbit left", "whip pan", "slow push-in"]
    joins = ["ai_motion", "whip", "crossfade"]
    for n, (i, dur, hfr) in enumerate(clips):
        start = min(0.5, dur / 4)
        seg = {"clip": i, "start": start, "end": min(dur, start + (1.2 if hfr else 3.2)), "slowmo": hfr,
               "reason": "mock pick", "emphasis": ["studio"]}
        if n:
            seg.update(transition_in=joins[(n - 1) % 3], camera_move=moves[(n - 1) % 3], effect="none")
        segs.append(seg)
    return {"title": "Mock title", "hook_text": "Watch this", "post_caption": "Mock caption",
            "hashtags": ["#studio", "shorts"], "music_mood": "upbeat", "segments": segs}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send({"models": [{"name": "qwen2.5:7b"}, {"name": "qwen2.5vl:7b"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        CALLS.append((self.path, body))
        if self.path == "/api/generate":
            return self._send({"done": True})
        if self.path != "/api/chat":
            return self._send({"error": "not found"}, 404)
        msg = body["messages"][-1]
        if msg.get("images"):
            content = {"description": "a person talking in a studio", "subject": "person", "action": "talking",
                       "shot": "medium", "interest": 7}
        else:
            content = plan_for(msg["content"])
        self._send({"model": body["model"], "message": {"role": "assistant", "content": json.dumps(content)}, "done": True})


def start(port=0, host="127.0.0.1"):
    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 11434, "0.0.0.0")
    print(f"mock ollama on :{srv.server_address[1]}", flush=True)
    threading.Event().wait()
