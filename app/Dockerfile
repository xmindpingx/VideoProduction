# One image for the web page/API, the worker, and ComfyUI. Built for AMD Radeon GPUs (ROCm PyTorch + VA-API video).
#   docker compose build            (see ../docker-compose.yml and ../.env.example)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    ASSETS_DIR=/opt/shorts DATA_DIR=/data HOME=/data XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/models/huggingface TORCH_HOME=/data/models/torch LIBVA_DRIVER_NAME=radeonsi

# Python 3.12, ffmpeg 6.1 (zscale, libass, arnndn, VA-API), Mesa VA-API driver for Radeon hardware video, git for ComfyUI
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ffmpeg mesa-va-drivers vainfo fontconfig curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Small models and fonts, pinned and checksum-verified:
#   OpenCV SSD face detector, RNNoise speech model (beguiling-drafter: voice in recordings), Anton + Archivo Black (OFL)
ARG FONTS_REV=23e54b51ddffbc7713c583748e3bd86f62b1fa4a
RUN set -eux; mkdir -p /opt/shorts/models /opt/shorts/fonts; cd /opt/shorts; \
    curl -fsSL -o models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt; \
    curl -fsSL -o models/res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel; \
    curl -fsSL -o models/bd.rnnn https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn; \
    curl -fsSL -o fonts/Anton-Regular.ttf https://raw.githubusercontent.com/google/fonts/${FONTS_REV}/ofl/anton/Anton-Regular.ttf; \
    curl -fsSL -o fonts/Anton-OFL.txt https://raw.githubusercontent.com/google/fonts/${FONTS_REV}/ofl/anton/OFL.txt; \
    curl -fsSL -o fonts/ArchivoBlack-Regular.ttf https://raw.githubusercontent.com/google/fonts/${FONTS_REV}/ofl/archivoblack/ArchivoBlack-Regular.ttf; \
    curl -fsSL -o fonts/ArchivoBlack-OFL.txt https://raw.githubusercontent.com/google/fonts/${FONTS_REV}/ofl/archivoblack/OFL.txt; \
    printf '%s\n' \
      "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9  models/deploy.prototxt" \
      "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d  models/res10_300x300_ssd_iter_140000.caffemodel" \
      "ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b  models/bd.rnnn" \
      "a4ba3a92350ebb031da0cb47630ac49eb265082ca1bc0450442f4a83ab947cab  fonts/Anton-Regular.ttf" \
      "dd9a89a019b4849f66ab75455fe7bdf931311042cbb0f0f97acc061539703180  fonts/ArchivoBlack-Regular.ttf" \
    | sha256sum -c -; \
    fc-cache -f /opt/shorts/fonts

# PyTorch for the GPU. Default: ROCm build for AMD Radeon (RX 6800 = gfx1030). Leave TORCH_INDEX empty for a build without
# PyTorch (Whisper then needs WHISPER_ENGINE=faster on the CPU, and ComfyUI is skipped).
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4
ARG TORCH_VERSION=2.8.0
RUN set -eux; touch /opt/torch-constraints.txt; \
    if [ -n "$TORCH_INDEX" ]; then \
      pip install "torch==${TORCH_VERSION}" torchvision "torchaudio==${TORCH_VERSION}" --index-url "$TORCH_INDEX"; \
      pip freeze | grep -iE '^(torch|torchvision|torchaudio|pytorch-triton-rocm|triton)==' > /opt/torch-constraints.txt; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -c /opt/torch-constraints.txt

# GPU AI tools on top of that PyTorch: Whisper (speech + word timings) and Demucs (dialogue isolation).
# --no-deps keeps pip from swapping the ROCm PyTorch for another build.
ARG INSTALL_DEMUCS=true
RUN set -eux; if [ -n "$TORCH_INDEX" ]; then \
      pip install --no-deps openai-whisper==20250625 && \
      pip install tiktoken more-itertools tqdm numba -c /opt/torch-constraints.txt; \
      if [ "$INSTALL_DEMUCS" = "true" ]; then \
        pip install --no-deps demucs==4.1.0 julius==0.2.7 && \
        pip install einops huggingface-hub lameenc pyyaml safetensors sphn tqdm -c /opt/torch-constraints.txt; \
      fi; \
    fi

# ComfyUI (the image-to-video generator for End-Frame Motion) in its own venv; a .pth file lets it use the ROCm PyTorch
# installed above instead of downloading another copy.
ARG INSTALL_COMFYUI=true
ARG COMFYUI_REF=master
RUN set -eux; if [ -n "$TORCH_INDEX" ] && [ "$INSTALL_COMFYUI" = "true" ]; then \
      git clone --depth 1 --branch "$COMFYUI_REF" https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI && \
      python3 -m venv /opt/comfy-venv && \
      /opt/venv/bin/python -c "import site; print(site.getsitepackages()[0])" \
        > "$(/opt/comfy-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/shared-torch.pth" && \
      /opt/comfy-venv/bin/pip install -r /opt/ComfyUI/requirements.txt -c /opt/torch-constraints.txt; \
    fi

WORKDIR /opt/app
COPY shorts ./shorts
COPY static ./static
EXPOSE 8590
CMD ["uvicorn", "shorts.api:app", "--host", "0.0.0.0", "--port", "8590", "--proxy-headers", "--forwarded-allow-ips", "*"]
