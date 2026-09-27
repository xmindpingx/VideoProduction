# VideoProduction · Shorts Lab

Upload clips from your iPhone (normal, 60 fps, or Slo-mo), and get back a vertical 9:16 short: the best moments picked by AI,
the speaker or subject kept in frame, animated captions, cuts on the beat, AI-generated "End-Frame Motion" transitions,
cleaned-up dialogue, and a film finish. Everything runs on your own machine with free, open-source software, on an
**AMD Radeon RX 6800** (ROCm + VA-API), next to the **Ollama** and **Open WebUI** you already run.

## What it does

| Step | How |
|---|---|
| Mobile upload | Phone-first page. Uploads go in 16 MB pieces, so they resume after a dropped connection and pass Cloudflare's 100 MB request limit. |
| iPhone formats | HEVC/H.264, Dolby Vision / HLG HDR (tone-mapped to SDR), portrait rotation, variable frame rate. 120/240 fps Slo-mo is slowed down smoothly (up to 4×); 60 fps can be slowed 2×. |
| Understanding the footage | **Whisper** (GPU) gives word-level timings. A scan finds scenes, motion, faces, sharpness and loudness. An **Ollama vision model** describes key frames and rates how eye-catching they are. |
| Picking the moments | An **Ollama text model** plans the edit as an editor would: the hook first, then segments on sentence boundaries that build to a payoff. It also picks the transitions and a camera-only prompt for each AI transition, and writes a title, hook text, post caption and hashtags. If Ollama is unreachable, a rule-based editor takes over. Either way, cuts never split a word and no footage is used twice. |
| Smart 9:16 reframing | Face tracking, with a motion-tracking fallback, drives a virtual camera operator. The camera stays locked off when the subject barely moves, pans in a straight line when it drifts, and follows smoothly when it moves around. Groups that don't fit a 9:16 crop get a "fit + blurred fill" layout. Alternate talking shots get a punch-in zoom. |
| End-Frame Motion | The last frame of shot A (and the first frame of shot B, if your workflow takes two images) goes to your **ComfyUI** image-to-video workflow. The prompt describes **only the camera move or effect** (e.g. `slow push-in camera move, smooth natural motion, same scene`). The result is color-matched to the real frame and conformed to the short's size and frame rate. Without ComfyUI, the frame is animated with the same camera move instead. |
| Dialogue | **Demucs** isolates the voice (GPU), with RNNoise as the fallback. Every speaking segment is brought to the same loudness, and clips from different recordings get a gentle EQ match so the voice doesn't change color between cuts. |
| Music | Your track (one you have the rights to). Beats are detected and cuts moved onto them. The music ducks under speech, with a soft whoosh on the fast transitions. Master: -14 LUFS, true peak -1.5 dB. |
| Film finish | One grade across every shot (real and generated), highlight roll-off, halation, vignette, and fine grain that is identical on all footage. Captions and titles sit on top, clean. Optional 24 fps with a 180°-style shutter blur from 60 fps sources. |
| Captions | Pop (1-3 big words, the spoken word lights up gold, AI-picked emphasis words in blue), Karaoke, or Clean. Positioned above the TikTok/Reels/Shorts UI. An `.srt` file is included. |

About trending sounds: a platform's trending sound only counts as that sound when it's added inside the TikTok or
Instagram app, so post the short there and add the sound in the app. Music you drop in here gets burned into the video.

## What runs where

```
iPhone ──(page + chunked upload)──> web  (FastAPI :8590)
                                      │ SQLite queue, shared ./data volume
                                      ▼
                                   worker ──> Ollama (yours)       text + vision models   ─┐
                                      │   ──> Whisper (PyTorch)     speech → words         ├─ RX 6800 via ROCm
                                      │   ──> Demucs (PyTorch)      dialogue isolation     │
                                      │   ──> comfyui (:8188)       End-Frame Motion       ─┘
                                      └──> ffmpeg  decode/encode on the RX 6800 via VA-API
```

The web page, worker and ComfyUI all come from one image (`app/Dockerfile`). The worker and ComfyUI get the GPU
(`/dev/kfd`, `/dev/dri`). The GPU is shared, so models are handed off: Ollama's models are unloaded after planning,
Whisper is freed after analysis, and ComfyUI's models are freed after the transitions are made.

These pieces stay on the CPU: face detection on small 640 px frames, HDR tone mapping, the reframing warps and the audio
filters. They are light next to the AI steps, but the CPU still does real work.

## Setup

Host requirements: Docker with Compose v2, and the `amdgpu` kernel driver for the RX 6800. The ROCm user-space runs
inside the container, but the host must expose `/dev/kfd` and `/dev/dri`.

```bash
git clone https://github.com/xmindpingx/VideoProduction.git
cd VideoProduction
cp .env.example .env
#   SHORTS_TOKEN=$(openssl rand -hex 24)
#   RENDER_GID=$(getent group render | cut -d: -f3)
#   VIDEO_GID=$(getent group video | cut -d: -f3)
nano .env
docker compose up -d --build        # the first build downloads several GB (ROCm PyTorch)
docker compose logs -f worker       # look for "GPU video: VA-API encode + decode" and "worker ready"
```

Open `http://localhost:8590` and sign in with `SHORTS_TOKEN`. The status dots at the top show AI (Ollama), Gen (ComfyUI)
and GPU; tap one for details.

### Ollama models

```bash
ollama pull qwen2.5:7b      # the editor
ollama pull qwen2.5vl:7b    # looks at frames
```

If your Ollama runs in Docker, run these as `docker exec -it <ollama-container> ollama pull ...`. For it to use the RX 6800,
Ollama must run its ROCm build (the `ollama/ollama:rocm` image, with `/dev/kfd` and `/dev/dri` passed in). To check, look
for the Radeon in Ollama's startup log, or run `ollama ps` while a short is at the *Planning the edit* stage (the
PROCESSOR column should say GPU; the models are unloaded again right after planning).

**Reaching Ollama.** The default `OLLAMA_URL=http://host.docker.internal:11434` works when Ollama publishes port 11434 on
the host. If it only lives on a Docker network, attach it and point at it by name:

```bash
docker network connect videoproduction_default <ollama-container>
# .env: OLLAMA_URL=http://<ollama-container>:11434
```

To go through **Open WebUI** instead (its `/ollama` proxy), create an API key in Open WebUI (Settings > Account) and set
`OPENWEBUI_URL` and `OPENWEBUI_API_KEY`.

### ComfyUI workflow (for AI transitions)

1. Open `http://localhost:8188`. Go to Workflow > Browse Templates > Video, and pick an **image-to-video** template (Wan, LTX-Video, …).
2. Download the models the template asks for into `data/comfyui/models/...`. Pick ones that fit the RX 6800's 16 GB;
   smaller models (for example LTX-Video, or the 5B Wan 2.2 model) are the ones to try first.
3. Run it once in ComfyUI to confirm it works on your card.
4. Use Workflow > **Export (API)** and save the file as `data/comfy/i2v_api.json`.

The Gen dot turns green when the workflow loads. Shorts Lab fills in the first `LoadImage` node (a second `LoadImage`
gets the next shot's first frame, for first/last-frame workflows), the prompt node that feeds the sampler's `positive`
input, a random seed, and a 9:16 size (`COMFY_WIDTH` × `COMFY_HEIGHT`). Image-only workflows such as Stable Video
Diffusion also work; the camera prompt is simply not used.

Already running ComfyUI elsewhere? Set `COMFYUI_URL` to it and remove the `comfyui` service.

### Using it from your phone

- **Anywhere:** in your Cloudflare Tunnel, add a public hostname (e.g. `shorts.yourdomain.com`) pointing at
  `http://localhost:8590`. Uploads are chunked, so Cloudflare's per-request limit doesn't matter.
- **Home Wi-Fi (faster):** set `BIND_ADDR=0.0.0.0` in `.env`, run `docker compose up -d`, and open `http://<server-ip>:8590`.

Tip: if the iPhone photo picker says *"Compressing Video…"*, the phone is shrinking the file. For full quality, save the
clip to Files first and pick it with *Choose File*.

### Open WebUI tool (optional)

`openwebui/videoproduction_tool.py` lets you list shorts, read the AI's post caption, and start a remix with new
direction from a chat. In Open WebUI, go to Workspace > Tools > +, paste the file, and set the valves: `access_token` =
`SHORTS_TOKEN`, and `public_url` = the address you open Shorts Lab at. Then connect the containers:
`docker network connect videoproduction_default <open-webui-container>` (the default `base_url` is `http://web:8590`).

## Settings

Everything is in `.env` (see `.env.example`, where every option is documented). The ones you are most likely to change:

| Setting | Default | |
|---|---|---|
| `TEXT_MODEL` / `VISION_MODEL` | `qwen2.5:7b` / `qwen2.5vl:7b` | any Ollama models; the vision one must accept images |
| `WHISPER_MODEL` | `turbo` | `small`, `medium`, `large-v3`, … |
| `AUDIO_CLEANUP` | `demucs` | `rnnoise`, `basic`, `off` |
| `HWACCEL` | `vaapi` | `none` forces software video |
| `COMFY_MAX_CLIPS` | `3` | AI transitions per short |
| `RETENTION_DAYS` | `14` | uploads and shorts older than this are deleted (`0` keeps them) |
| `DATA_PATH` | `./data` | put it on a big disk, e.g. `/mnt/ssd/videoproduction-data` |

## Troubleshooting

- **`RENDER_GID` error on `docker compose up`:** set it in `.env` (`getent group render | cut -d: -f3`).
- **GPU dot says "no GPU visible to PyTorch":** check `ls -l /dev/kfd /dev/dri` on the host and the group ids in `.env`.
  `docker compose exec worker python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  should print `True` and the Radeon's name.
- **"GPU video: … failed":** the worker falls back to software (x264) encoding by itself. To see what VA-API reports:
  `docker compose exec worker vainfo --display drm --device /dev/dri/renderD128`.
- **AI dot is red:** Ollama isn't reachable from the container (see *Reaching Ollama*). The short is still made, with the rule-based editor.
- **Logs:** each short's **Details** button on the page, or `docker compose logs worker`.

## Development

```bash
cd app
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest httpx
python -m pytest tests        # needs ffmpeg; downloads the small face/RNNoise models and fonts once
```

`tests/mock_ollama.py` stands in for Ollama's API, so the AI-editor path can be tested without a model. It checks the
plumbing, not the editing taste of a real model.

## Tested so far, and not yet

Tested in a cloud sandbox (no GPU):
- all 17 tests: API, chunked/resumable upload, probing, job lifecycle, the plan clean-up, captions, and a full short
  from synthetic 60 fps, 240 fps and rotated-HDR clips, with the mock Ollama
- face tracking with a real face photo
- the ComfyUI client against a real ComfyUI 0.37.0 (upload, queue, history, download of WebP and MP4 output, conform + color match)
- the Docker image build, without the ROCm PyTorch layer: ffmpeg 6.1.1, the `h264_vaapi` encoder and the Radeon VA-API driver are present

Not tested, because the sandbox had no GPU and could not download the files:
- the ROCm PyTorch build, Whisper transcription, Demucs, and VA-API encoding on a real RX 6800
- real Ollama models
- a real image-to-video model in ComfyUI

Expect the first run on your machine to be the real test of those parts. The worker logs say which GPU paths came up.

## Third-party software

ffmpeg (Ubuntu build, GPL), ComfyUI (GPL-3.0), openai-whisper (MIT), faster-whisper (MIT), Demucs (MIT), PyTorch
(BSD-style), OpenCV and its SSD face-detector sample model, RNNoise models by GregorR, the Anton and Archivo Black fonts
(SIL Open Font License), FastAPI, librosa and pyloudnorm. Ollama and video models each have their own license (see
each model's page); some video models restrict commercial use. Use music and footage you have the rights to.
