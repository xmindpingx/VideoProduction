"""
title: VideoProduction Shorts
description: Check on and remix the vertical shorts made by your self-hosted Shorts Lab (VideoProduction).
version: 1.0.0
license: see the VideoProduction repository
"""
# Install: Open WebUI > Workspace > Tools > + (Create), paste this file, save, then set the Valves (gear icon).
# Open WebUI must be able to reach the Shorts Lab "web" service, e.g. after
#   docker network connect videoproduction_default <your open-webui container>
# the default base_url http://web:8590 works.

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        base_url: str = Field("http://web:8590", description="Shorts Lab address as seen from the Open WebUI container")
        public_url: str = Field("http://localhost:8590", description="Address you open Shorts Lab at (used for links)")
        access_token: str = Field("", description="SHORTS_TOKEN from the VideoProduction .env file")

    def __init__(self):
        self.valves = self.Valves()

    def _get(self, path):
        r = requests.get(self.valves.base_url.rstrip("/") + "/api/" + path,
                         headers={"Authorization": "Bearer " + self.valves.access_token}, timeout=20)
        r.raise_for_status()
        return r.json()

    def _post(self, path, body):
        r = requests.post(self.valves.base_url.rstrip("/") + "/api/" + path, json=body,
                          headers={"Authorization": "Bearer " + self.valves.access_token}, timeout=20)
        r.raise_for_status()
        return r.json()

    def _link(self, job_id):
        return f"{self.valves.public_url.rstrip('/')}/api/jobs/{job_id}/files/short.mp4"

    def list_recent_shorts(self) -> str:
        """
        List the most recent shorts made in Shorts Lab, with their status and a link to each finished video.
        """
        try:
            jobs = self._get("jobs")[:10]
        except requests.RequestException as e:
            return f"Could not reach Shorts Lab: {e}"
        if not jobs:
            return "No shorts have been made yet."
        lines = []
        for j in jobs:
            r = j.get("result") or {}
            b = j.get("brief") or {}
            name = r.get("title") or f"{b.get('length')}s {b.get('vibe')} short"
            state = j["status"] if j["status"] != "running" else f"running {round(j['progress'] * 100)}% ({j.get('stage', '')})"
            link = f" - {self._link(j['id'])}" if j["status"] == "done" else ""
            lines.append(f"- {j['id']}: {name} [{state}]{link}")
        return "\n".join(lines)

    def short_details(self, job_id: str) -> str:
        """
        Show one short's status, any warnings, and the suggested post caption and hashtags.
        :param job_id: The short's id, as shown by list_recent_shorts.
        """
        try:
            j = self._get(f"jobs/{job_id}")
        except requests.RequestException as e:
            return f"Could not load that short: {e}"
        r = j.get("result") or {}
        out = [f"Status: {j['status']} {j.get('stage') or ''}".strip()]
        if j.get("error"):
            out.append(f"Error: {j['error']}")
        if r:
            out += [f"Title: {r.get('title') or '-'}", f"Hook: {r.get('hook_text') or '-'}", f"Length: {r.get('seconds')}s",
                    f"Edited by: {'AI (Ollama)' if r.get('planner') == 'ollama' else 'rules'}",
                    f"Post caption: {r.get('post_caption') or '-'}",
                    "Hashtags: " + (" ".join('#' + t for t in r.get("hashtags", [])) or "-"),
                    f"Video: {self._link(job_id)}"]
            out += [f"Warning: {w}" for w in r.get("warnings", [])]
        return "\n".join(out)

    def remix_short(self, job_id: str, direction: str, vibe: str = "", length_seconds: int = 0) -> str:
        """
        Make a new version of a short from the same clips with new creative direction.
        :param job_id: The short to remix.
        :param direction: What the new version should focus on, e.g. "open on the crowd, end on the trophy".
        :param vibe: Optional: hype, cinematic, funny, emotional, story or product.
        :param length_seconds: Optional target length in seconds (8-90).
        """
        brief = {"prompt": direction}
        if vibe:
            brief["vibe"] = vibe
        if length_seconds:
            brief["length"] = int(length_seconds)
        try:
            j = self._post(f"jobs/{job_id}/remix", {"brief": brief})
        except requests.RequestException as e:
            return f"Could not start the remix: {e}"
        return f"Remix queued as {j['id']}. Ask me for short_details({j['id']}) in a few minutes."
