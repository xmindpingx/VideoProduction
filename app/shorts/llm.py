"""Minimal Ollama client (native API), used directly or through Open WebUI's /ollama proxy."""
import base64
import json
import re

import requests

from . import config


class LLMError(RuntimeError):
    pass


def _url(path):
    base, headers = config.ollama_base()
    return base + path, headers


def list_models(timeout=5):
    url, headers = _url("/api/tags")
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return [m.get("name") or m.get("model") for m in r.json().get("models", [])]


def has_model(names, model):
    want = model if ":" in model else model + ":latest"
    return any(n == model or n == want for n in names)


def _extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no JSON object in reply")


def chat_json(model, system, user, schema=None, images=None, temperature=0.5, num_ctx=None):
    """Ask for a JSON object. Uses Ollama structured outputs when the server supports them."""
    url, headers = _url("/api/chat")
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = [base64.b64encode(b).decode() for b in images]
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, msg],
        "stream": False,
        "format": schema or "json",
        "options": {"temperature": temperature, "num_ctx": num_ctx or config.OLLAMA_NUM_CTX},
    }
    last = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=config.OLLAMA_TIMEOUT)
            if r.status_code == 400 and schema and payload["format"] != "json":
                payload["format"] = "json"  # older Ollama without JSON-schema support
                continue
            if r.status_code == 404:
                raise LLMError(f"Model {model} is not installed in Ollama. Run: ollama pull {model}")
            r.raise_for_status()
            content = r.json().get("message", {}).get("content", "")
            return _extract_json(content)
        except LLMError:
            raise
        except (requests.RequestException, ValueError) as e:
            last = e
    raise LLMError(f"{model}: {last}")


def unload(model):
    """Ask Ollama to free the model's memory now (keep_alive 0) so a video generator can use the GPU."""
    url, headers = _url("/api/generate")
    try:
        requests.post(url, json={"model": model, "keep_alive": 0}, headers=headers, timeout=30)
    except requests.RequestException:
        pass
