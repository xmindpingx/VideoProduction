"""ComfyUI client for "End-Frame Motion": send a reference frame + a camera-only prompt, get a short clip back.

Works with any image-to-video workflow you exported from ComfyUI with "Export (API)" (Wan, LTX-Video, ...):
- the first LoadImage node gets the start frame; a second LoadImage node (first/last-frame workflows) gets the end frame
- the text node feeding the sampler's "positive" input gets the prompt (image-only workflows such as SVD also work)
- every seed / noise_seed is randomized, and width/height are set to COMFY_WIDTH x COMFY_HEIGHT (9:16)
"""
import json
import random
import time
import uuid
from pathlib import Path

import requests

from . import config
from .media import Canceled

VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")


class ComfyError(RuntimeError):
    pass


def enabled():
    return bool(config.COMFYUI_URL)


def load_workflow(path=None):
    path = Path(path or config.COMFY_WORKFLOW)
    if not path.is_file():
        raise ComfyError(f"No workflow at {path}. In ComfyUI open an image-to-video workflow, then Workflow > Export (API), and save it there.")
    wf = json.loads(path.read_text())
    if "nodes" in wf and "links" in wf:
        raise ComfyError("That workflow was saved in the normal format. Use Workflow > Export (API) instead.")
    if not isinstance(wf, dict) or not all(isinstance(n, dict) and "class_type" in n for n in wf.values()):
        raise ComfyError("Workflow is not in ComfyUI API format")
    return wf


def _link_source(value):
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def _trace_text_node(wf, node_id, branch, depth=0):
    """Follow conditioning links (staying on the positive or negative branch) back to the node holding the prompt string."""
    node = wf.get(node_id)
    if not node or depth > 8:
        return None
    for key in ("text", "prompt"):
        if isinstance(node["inputs"].get(key), str):
            return node_id, key
    for key in (branch, "conditioning", "conditioning_1", "conditioning_to"):
        src = _link_source(node["inputs"].get(key))
        if src:
            found = _trace_text_node(wf, src, branch, depth + 1)
            if found:
                return found
    return None


def inspect(wf):
    image_nodes = sorted((nid for nid, n in wf.items() if n["class_type"] == "LoadImage"), key=lambda x: (len(x), x))
    pos = neg = None
    for nid, n in wf.items():
        for key, target in (("positive", "pos"), ("negative", "neg")):
            src = _link_source(n["inputs"].get(key))
            if src:
                found = _trace_text_node(wf, src, key)
                if found and target == "pos" and not pos:
                    pos = found
                if found and target == "neg" and not neg:
                    neg = found
    return {"image_nodes": image_nodes, "positive": pos, "negative": neg}


def status():
    if not enabled():
        return {"enabled": False}
    res = {"enabled": True, "url": config.COMFYUI_URL}
    try:
        r = requests.get(config.COMFYUI_URL + "/system_stats", timeout=4)
        r.raise_for_status()
        res["ok"] = True
    except requests.RequestException as e:
        res.update(ok=False, error=str(e)[:200])
    try:
        info = inspect(load_workflow())
        res["workflow_ok"] = bool(info["image_nodes"])
        res["first_last_frame"] = len(info["image_nodes"]) >= 2
        res["takes_prompt"] = bool(info["positive"])
        if not res["workflow_ok"]:
            res["workflow_error"] = "Workflow needs a LoadImage node for the reference frame"
    except (ComfyError, ValueError) as e:
        res.update(workflow_ok=False, workflow_error=str(e)[:300])
    return res


def _upload(path):
    with open(path, "rb") as f:
        r = requests.post(config.COMFYUI_URL + "/upload/image", files={"image": (Path(path).name, f, "image/png")},
                          data={"overwrite": "true", "type": "input"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]


def free():
    """Ask ComfyUI to unload its models so the GPU memory is available to Ollama again."""
    try:
        requests.post(config.COMFYUI_URL + "/free", json={"unload_models": True, "free_memory": True}, timeout=30)
    except requests.RequestException:
        pass


def _interrupt(prompt_id):
    for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {})):
        try:
            requests.post(config.COMFYUI_URL + path, json=body, timeout=10)
        except requests.RequestException:
            pass


def generate(start_frame, prompt, out_dir, end_frame=None, cancel=None, log=print):
    """Run the workflow; return a list of downloaded output files (one video, or an image sequence)."""
    wf = load_workflow()
    info = inspect(wf)
    if not info["image_nodes"]:
        raise ComfyError("Workflow needs a LoadImage node for the reference frame")
    wf[info["image_nodes"][0]]["inputs"]["image"] = _upload(start_frame)
    if end_frame and len(info["image_nodes"]) >= 2:
        wf[info["image_nodes"][1]]["inputs"]["image"] = _upload(end_frame)
    if info["positive"]:
        nid, key = info["positive"]
        wf[nid]["inputs"][key] = prompt
    else:
        log("This workflow has no text prompt (e.g. Stable Video Diffusion); the camera prompt is not used")
    for n in wf.values():
        inp = n["inputs"]
        for k in ("seed", "noise_seed"):
            if isinstance(inp.get(k), int):
                inp[k] = random.randint(1, 2**48)
        if config.COMFY_SET_SIZE and isinstance(inp.get("width"), int) and isinstance(inp.get("height"), int):
            inp["width"], inp["height"] = config.COMFY_WIDTH, config.COMFY_HEIGHT

    r = requests.post(config.COMFYUI_URL + "/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex}, timeout=60)
    if r.status_code != 200:
        raise ComfyError(f"ComfyUI rejected the workflow: {r.text[:400]}")
    prompt_id = r.json()["prompt_id"]
    log(f"ComfyUI job {prompt_id} queued")

    start = time.time()
    while True:
        if cancel and cancel():
            _interrupt(prompt_id)
            raise Canceled()
        if time.time() - start > config.COMFY_TIMEOUT:
            _interrupt(prompt_id)
            raise ComfyError(f"ComfyUI took longer than {config.COMFY_TIMEOUT}s")
        time.sleep(2)
        h = requests.get(f"{config.COMFYUI_URL}/history/{prompt_id}", timeout=30).json()
        if prompt_id not in h:
            continue
        entry = h[prompt_id]
        st = entry.get("status", {})
        if st.get("status_str") == "error":
            msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
            detail = msgs[0][1].get("exception_message", "") if msgs else ""
            raise ComfyError(f"ComfyUI error: {detail[:300]}")
        if st.get("completed") or entry.get("outputs"):
            break

    files = []
    for node_out in entry.get("outputs", {}).values():
        for items in node_out.values():
            if isinstance(items, list):
                files += [i for i in items if isinstance(i, dict) and i.get("filename")]
    outputs = [f for f in files if f.get("type") == "output"] or files
    videos = [f for f in outputs if f["filename"].lower().endswith(VIDEO_EXT)]
    chosen = videos[:1] if videos else sorted((f for f in outputs if f["filename"].lower().endswith(".png")), key=lambda f: f["filename"])
    if not chosen:
        raise ComfyError("The workflow finished but produced no video or images")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(chosen):
        r = requests.get(config.COMFYUI_URL + "/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                                              "type": f.get("type", "output")}, timeout=120)
        r.raise_for_status()
        p = out_dir / (f"gen{Path(f['filename']).suffix.lower()}" if len(chosen) == 1 else f"frame_{i:05d}.png")
        p.write_bytes(r.content)
        paths.append(p)
    log(f"ComfyUI finished in {time.time() - start:.0f}s")
    return paths
