# LTX-2.3 image + audio to video on a 14 GB budget (RX 6800)

ComfyUI's official template **LTX-2.3: image + audio to video** (`video_ltx2_3_ia2v`), set up to fit 14 GB of VRAM on
the RX 6800, with everything still on the GPU.

| File | What it is |
|---|---|
| `ltx23_ia2v_14gb.json` | The template with lower-memory settings. Same model files. |
| `ltx23_ia2v_14gb_gguf.json` | The same, with the 22B model loaded from a GGUF file. Needs the ComfyUI-GGUF custom node. |
| `check_models.py` | Checks your model files, including the cause of `ValueError: invalid tokenizer`. |
| `make_14gb_workflows.py` | Rebuilds both workflows from the template, or from your own export of it. |

Load a workflow by dragging the `.json` onto ComfyUI, or with Workflow > Open.

The two workflows are made from version 0.11.70 of the template package (`comfyui-workflow-templates`, MIT license).

## 1. Fix "Load LTXV Audio Text Encoder failed: invalid tokenizer" first

ComfyUI reads the Gemma tokenizer from a tensor named `spiece_model` inside `gemma_3_12B_it_fp4_mixed.safetensors`.
If the file has no such tensor, ComfyUI has no tokenizer and stops with `invalid tokenizer`. This is in ComfyUI's
`comfy/sd.py` (LTXV text encoder branch) and `comfy/text_encoders/spiece_tokenizer.py`. It happens when the file is a
different Gemma file under that name, or when the download didn't finish.

Check your files (no GPU needed, it only reads file headers):

```bash
python3 check_models.py /path/to/ComfyUI/models
```

If it reports the Gemma file, replace it with the official one, in `models/text_encoders/`:
https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors

## 2. What the 14 GB workflow changes

| Setting | Template | Your run | 14 GB file | Why |
|---|---|---|---|---|
| width × height | 1280 × 720 | 320 × 480 | **384 × 576** | Same 2:3 portrait shape as yours. Must be a multiple of 64 (see below). The first frame is center-cropped to this shape. |
| fps | 24 | 15 | **24** | At 24 fps every whole number of seconds is a valid LTX frame count. |
| duration | 9 s | 30 s | **10 s** | Memory grows with the number of frames. Do longer audio in 10 s parts (section 4). |
| prompt_enhance | on | on | **off** | When on, the 12B Gemma model first writes a longer prompt (up to 2048 tokens): an extra generation pass. Turn it back on if you want it. |
| VAE decode (tile, overlap, frames, frame overlap) | 768, 64, 4096, 4 | | **512, 64, 64, 8** | The template decodes the whole clip as one chunk. 64-frame chunks lower the peak at the end of the run. |
| Models, text encoder device | | | unchanged | The text encoder stays on the GPU (`device: default`). |

**Why not 320 × 480 at 15 fps.** LTX works in 32 px blocks and 8-frame steps, and ComfyUI rounds other values down
without telling you:

- The template renders a first pass at half size. 480 / 2 = 240 rounds down to 224, so a 320 × 480 run comes out
  **320 × 448**. Multiples of 64 avoid this.
- Frames must be 8n + 1. 30 s × 15 fps + 1 = 451 frames, which becomes **449**. At 15 fps, only multiples of 8 s line
  up exactly; at 24 fps, any whole number of seconds does.

## 3. Launch options

Add these to ComfyUI's command line. With this repo's docker-compose, put them in `.env` as `COMFY_ARGS=...` and run
`docker compose up -d comfyui`.

- `--reserve-vram 2` keeps 2 GiB free for the desktop and other apps, so ComfyUI plans for the other 14 GiB of the
  RX 6800's 16 GiB. Without it, ComfyUI reserves 400 MiB on Linux, and 700 MiB on Windows with a 16 GB card
  (`comfy/model_management.py`).
  Lightricks also recommends `--reserve-vram` for low VRAM in the
  [ComfyUI-LTXVideo README](https://github.com/Lightricks/ComfyUI-LTXVideo#low-vram).
- If you still run out of memory: `--disable-smart-memory` (ComfyUI moves models to system RAM instead of keeping
  them in VRAM) and `--use-split-cross-attention` (ComfyUI's log suggests it for memory or speed problems when it uses
  its default sub-quadratic attention, which it does on this card).
- **Don't use `--lowvram`** for this: when dynamic VRAM is off, it runs the text encoders on the CPU (its help text in
  `comfy/cli_args.py`).

## 4. Longer audio: chain 10 s parts

1. Run part 1 with `audio_start` = 0.
2. Take its last frame: `ffmpeg -sseof -1 -i part_01.mp4 -update 1 last_01.png`
3. Run part 2 with `last_01.png` as the first frame and `audio_start` = 10. Keep the prompt and seed. Repeat for the next parts.
4. Join the parts: list them in `parts.txt` (`file 'part_01.mp4'`, one per line), then
   `ffmpeg -f concat -safe 0 -i parts.txt -c copy joined.mp4`. Each part starts on the frame the previous one ended
   on, so that one frame shows twice at each join.

## 5. The GGUF version

`ltx23_ia2v_14gb_gguf.json` loads the 22B model with **Unet Loader (GGUF)** from
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (install it with ComfyUI-Manager or clone it into
`custom_nodes/`). Put the `.gguf` in `models/diffusion_models/` and pick it in that node. The file name preset in the
node is `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`, as published in
[QuantStack/LTX-2.3-GGUF](https://huggingface.co/QuantStack/LTX-2.3-GGUF) (Unsloth also publishes LTX-2.3 GGUFs:
[unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)).

- The distilled LoRA node is **bypassed**, because that GGUF is already the distilled model. If you use a *dev* GGUF
  instead, select the LoRA node and press Ctrl+B to turn it back on.
- The FP8 checkpoint is still needed: it supplies the video VAE, the audio VAE and the text-encoder connector.
- GGUF files are smaller than the FP8 model, so less of it has to sit in system RAM. Check the size on the download
  page: anything over your VRAM budget is still partly offloaded.
- The results won't match the template exactly: the template runs the dev model with the distilled LoRA at strength 0.5.

## What happens on the RX 6800

From ComfyUI's source (version 0.37.0). The startup and model-load lines in ComfyUI's log show what it picked on your machine.

- **The model doesn't fit in 14 GB.** LTX-2.3 has 22 billion parameters (the `22b` in the file name). At FP8, one byte
  each, that is more than 14 GB of weights before any activations. ComfyUI keeps part of the model in system RAM and
  streams it to the GPU while sampling. This works, but it is slower and needs plenty of system RAM. Lightricks lists
  32 GB+ of VRAM for its own LTX ComfyUI nodes
  ([ComfyUI-LTXVideo README](https://github.com/Lightricks/ComfyUI-LTXVideo#prerequisites)).
- **FP8 is storage only on this card.** ComfyUI turns on FP8 math on AMD only for gfx1200, gfx1201, gfx950, gfx1170 and
  gfx1171. The RX 6800 is gfx1030, so FP8 weights are converted layer by layer as they are used.
- **The math runs in FP32.** ComfyUI treats RDNA2 (gfx1030) as having no bf16, and LTX-2.3 supports only bf16 and FP32
  in ComfyUI. Activations take twice the memory of bf16, which is a reason this card needs a smaller size or fewer
  frames than an NVIDIA card with the same VRAM. The log line is
  `model weight dtype torch.float8_e4m3fn, manual cast: torch.float32`.
- **Dynamic VRAM is off.** ComfyUI enables it on AMD only when PyTorch reports ROCm 7.14 or newer. This repo's image
  uses ROCm 6.4 PyTorch wheels (`TORCH_INDEX` in `.env.example`), so ComfyUI uses its older loader. `--reserve-vram`
  works with both.

## Tested, and not

Tested in a cloud sandbox without a GPU or the real model files:
- both workflows (and the original template, for comparison) load in ComfyUI 0.37.0's frontend, convert to a prompt,
  and pass ComfyUI's `/prompt` validation, with ComfyUI-GGUF installed for the GGUF version
- `check_models.py` against made-up files: complete, missing, a Git LFS pointer, an HTML page, a cut-short file, a
  Gemma file without `spiece_model`, and a GGUF file

Not tested: generating a video. That needs the real models and your RX 6800, so the speed, memory use and look of
both versions on your machine are unknown until you run them.
