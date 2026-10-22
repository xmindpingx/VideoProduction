"""Check the model files for the LTX-2.3 image + audio to video workflow before running it.

    python3 check_models.py path/to/ComfyUI/models

Reads only file headers (standard library, no GPU, nothing loaded). For each model the template uses, it reports:
missing, a Git LFS pointer or HTML page saved instead of the model, a download cut short, and for the Gemma text
encoder, whether the file carries the tokenizer ("spiece_model"). Without the tokenizer, "Load LTXV Audio Text Encoder"
fails with "ValueError: invalid tokenizer". It also lists any .gguf models in diffusion_models/ and unet/.
"""
import json
import struct
import sys
from pathlib import Path

GEMMA = "gemma_3_12B_it_fp4_mixed.safetensors"

# (models subfolder, file name, download URL), as listed in the official template (video_ltx2_3_ia2v)
EXPECTED = [
    ("checkpoints", "ltx-2.3-22b-dev-fp8.safetensors",
     "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors"),
    ("loras", "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"),
    ("text_encoders", GEMMA,
     "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"),
    ("loras", "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
     "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"),
    ("latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
     "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
]

GGUF_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gib(n):
    return f"{n / 1024 ** 3:.2f} GiB"


def not_a_model(head):
    if head.startswith(b"version https://git-lfs"):
        return "a Git LFS pointer, not the model (download the file itself, e.g. with the URL below)"
    if head.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        return "an HTML page, not the model (the download saved an error or login page)"
    return None


def check_safetensors(path):
    """Returns (problem or None, tensor names)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(8)
        problem = not_a_model(head + f.read(64))
        if problem:
            return problem, []
        if len(head) < 8:
            return "empty or too small to be a model", []
        (n,) = struct.unpack("<Q", head)
        if n > size - 8 or n > 512 * 1024 ** 2:
            return "not a valid .safetensors file (bad header)", []
        f.seek(8)
        try:
            header = json.loads(f.read(n))
        except ValueError:
            return "not a valid .safetensors file (unreadable header)", []
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = 8 + n + max((v["data_offsets"][1] for v in tensors.values()), default=0)
    if size < end:
        return f"cut short: {size:,} bytes on disk, the header says {end:,}; download it again", list(tensors)
    return None, list(tensors)


def gguf_architecture(path):
    """Reads general.architecture from a GGUF header (ComfyUI-GGUF refuses architectures it does not know)."""
    with path.open("rb") as f:
        def read(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(read("<Q")).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                item, count = read("<I"), read("<Q")
                return [value(item) for _ in range(count)]
            return read(GGUF_TYPES[t])

        if f.read(4) != b"GGUF":
            return None
        read("<I")  # version
        read("<Q")  # tensor count
        for _ in range(read("<Q")):
            key, t = string(), read("<I")
            v = value(t)
            if key == "general.architecture":
                return v
    return "(not set)"


def find(models, folder, name):
    base = models / folder
    direct = base / name
    if direct.is_file():
        return direct
    # ComfyUI also lists models in subfolders
    return next(iter(sorted(base.rglob(name))), None) if base.is_dir() else None


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    models = Path(sys.argv[1]).expanduser()
    if not models.is_dir():
        raise SystemExit(f"{models} is not a folder. Pass ComfyUI's models folder (it contains checkpoints/, loras/, ...).")

    problems = 0
    for folder, name, url in EXPECTED:
        path = find(models, folder, name)
        if path is None:
            problems += 1
            print(f"MISSING  {folder}/{name}\n         download: {url}")
            continue
        problem, tensors = check_safetensors(path)
        if problem is None and name == GEMMA and "spiece_model" not in tensors:
            problem = ('no tokenizer inside (no "spiece_model" tensor); this is what causes "ValueError: invalid '
                       'tokenizer" in Load LTXV Audio Text Encoder. Replace it with the official file')
        rel = path.relative_to(models)
        if problem:
            problems += 1
            print(f"PROBLEM  {rel}\n         {problem}\n         download: {url}")
        else:
            extra = ', tokenizer inside' if name == GEMMA else ''
            print(f"OK       {rel} ({gib(path.stat().st_size)}, {len(tensors)} tensors{extra})")

    ggufs = sorted(p for folder in ("diffusion_models", "unet") if (models / folder).is_dir()
                   for p in (models / folder).rglob("*.gguf"))
    for path in ggufs:
        rel = path.relative_to(models)
        try:
            arch = gguf_architecture(path)
        except (struct.error, KeyError, UnicodeDecodeError, OverflowError, MemoryError):
            arch = None
        if arch is None:
            with path.open("rb") as f:
                problem = not_a_model(f.read(72)) or "not a valid GGUF file"
            problems += 1
            print(f"PROBLEM  {rel}\n         {problem}")
        else:
            print(f"GGUF     {rel} ({gib(path.stat().st_size)}, architecture: {arch})")

    print("\nAll model files look complete." if problems == 0 else f"\n{problems} problem(s) found.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
