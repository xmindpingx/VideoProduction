"""Settings, all from the environment (see ../../.env.example)."""
import os
from pathlib import Path


def _env(name, default=""):
    return os.environ.get(name, default).strip()


def _int(name, default):
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name, default=False):
    v = _env(name, "true" if default else "false").lower()
    return v in ("1", "true", "yes", "on")


TOKEN = _env("SHORTS_TOKEN")
DATA_DIR = Path(_env("DATA_DIR", "/data"))
ASSETS_DIR = Path(_env("ASSETS_DIR", "/opt/shorts"))  # fonts + small models baked into the image
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

COOKIE_SECURE = _env("COOKIE_SECURE", "auto").lower()  # auto | true | false
SESSION_DAYS = _int("SESSION_DAYS", 30)
MAX_UPLOAD_BYTES = int(_float("MAX_UPLOAD_GB", 10) * 1024**3)
CHUNK_BYTES = _int("CHUNK_MB", 16) * 1024**2
RETENTION_DAYS = _int("RETENTION_DAYS", 14)

# Ollama: direct, or through Open WebUI's /ollama proxy when OPENWEBUI_URL + OPENWEBUI_API_KEY are set.
OLLAMA_URL = _env("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
OPENWEBUI_URL = _env("OPENWEBUI_URL").rstrip("/")
OPENWEBUI_API_KEY = _env("OPENWEBUI_API_KEY")
TEXT_MODEL = _env("TEXT_MODEL", "qwen2.5:7b")
VISION_MODEL = _env("VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 16384)
OLLAMA_TIMEOUT = _int("OLLAMA_TIMEOUT", 600)
VISION_MAX_FRAMES = _int("VISION_MAX_FRAMES", 16)  # per clip
OLLAMA_UNLOAD_AFTER = _bool("OLLAMA_UNLOAD_AFTER", True)  # free VRAM for ComfyUI once planning is done

# Whisper: "torch" = openai-whisper on PyTorch (runs on AMD ROCm or NVIDIA GPUs), "faster" = faster-whisper (CPU / NVIDIA only)
WHISPER_ENGINE = _env("WHISPER_ENGINE", "torch")
WHISPER_MODEL = _env("WHISPER_MODEL", "turbo")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "gpu")  # gpu | cpu  (ROCm PyTorch exposes the AMD GPU as "cuda")
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "int8")  # faster-whisper only
WHISPER_LANGUAGE = _env("WHISPER_LANGUAGE")  # blank = detect

# Hardware video decode/encode on the GPU through VA-API (AMD Radeon via Mesa). Falls back to software if the self-test fails.
HWACCEL = _env("HWACCEL", "vaapi")  # vaapi | none
VAAPI_DEVICE = _env("VAAPI_DEVICE", "/dev/dri/renderD128")

COMFYUI_URL = _env("COMFYUI_URL").rstrip("/")
COMFY_WORKFLOW = Path(_env("COMFY_WORKFLOW", str(DATA_DIR / "comfy" / "i2v_api.json")))
COMFY_WIDTH = _int("COMFY_WIDTH", 480)
COMFY_HEIGHT = _int("COMFY_HEIGHT", 832)
COMFY_SET_SIZE = _bool("COMFY_SET_SIZE", True)
COMFY_TIMEOUT = _int("COMFY_TIMEOUT", 1200)
COMFY_MAX_CLIPS = _int("COMFY_MAX_CLIPS", 3)  # generated transitions per short
COMFY_PROMPT_SUFFIX = _env("COMFY_PROMPT_SUFFIX")

AUDIO_CLEANUP = _env("AUDIO_CLEANUP", "rnnoise")  # off | basic | rnnoise | demucs
HDR_TONEMAP = _env("HDR_TONEMAP", "hable")
ENCODE_PRESET = _env("ENCODE_PRESET", "medium")
FFMPEG_THREADS = _int("FFMPEG_THREADS", 0)

OUT_W, OUT_H = 1080, 1920

FONTS_DIR = ASSETS_DIR / "fonts"
FACE_PROTO = ASSETS_DIR / "models" / "deploy.prototxt"
FACE_MODEL = ASSETS_DIR / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
RNNOISE_MODEL = ASSETS_DIR / "models" / "bd.rnnn"

UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "shorts.db"
MODELS_DIR = DATA_DIR / "models"


def ollama_base():
    """(base_url, headers) for Ollama's native API, directly or via Open WebUI."""
    if OPENWEBUI_URL and OPENWEBUI_API_KEY:
        return OPENWEBUI_URL + "/ollama", {"Authorization": "Bearer " + OPENWEBUI_API_KEY}
    return OLLAMA_URL, {}


def ensure_dirs():
    for d in (DATA_DIR, UPLOADS_DIR, JOBS_DIR, MODELS_DIR, DATA_DIR / "comfy"):
        d.mkdir(parents=True, exist_ok=True)
