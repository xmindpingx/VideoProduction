"""End-Frame Motion transitions.

The last frame of the outgoing shot is the visual anchor. With ComfyUI, that frame (plus the first frame of the next
shot, when the workflow takes two images) goes to an image-to-video model with a prompt that describes only the camera
move / effect, so the generated clip continues the real footage instead of inventing a new scene. The result is then
colour-matched to the real frame and conformed to the short's size and frame rate, which is most of what separates a
seamless AI transition from an obvious one.

Without a generator, the same anchor frame is animated on the CPU with the chosen camera move (eased, motion-blurred).
"""
import subprocess

import cv2
import numpy as np

from .. import comfy, config, media
from .reframe import encode_cmd


def camera_prompt(move, effect):
    parts = [f"{move} camera move"]
    if effect and effect != "none":
        parts.append(effect)
    parts.append("smooth natural motion, same scene")
    if config.COMFY_PROMPT_SUFFIX:
        parts.append(config.COMFY_PROMPT_SUFFIX)
    return ", ".join(parts)


def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def _move_params(move, t, w, h):
    """(scale, dx, dy, degrees) of the virtual camera at progress t (0..1)."""
    e = _ease(t)
    if move == "slow pull-out":
        return 1.14 - 0.14 * e, 0, 0, 0
    if move == "pan left":
        return 1.12, (0.05 - 0.10 * e) * w, 0, 0
    if move == "pan right":
        return 1.12, (-0.05 + 0.10 * e) * w, 0, 0
    if move == "tilt up":
        return 1.12, 0, (-0.04 + 0.08 * e) * h, 0
    if move == "tilt down":
        return 1.12, 0, (0.04 - 0.08 * e) * h, 0
    if move == "crane up":
        return 1.06 + 0.08 * e, 0, (-0.03 + 0.06 * e) * h, 0
    if move in ("orbit left", "orbit right"):
        sgn = 1 if move == "orbit left" else -1
        return 1.14, sgn * (0.04 - 0.08 * e) * w, 0, sgn * (-1.5 + 3.0 * e)
    if move == "handheld drift":
        return 1.08, (np.sin(t * 5.1) * 0.006 + np.sin(t * 2.3) * 0.01) * w, (np.sin(t * 3.7) * 0.008) * h, np.sin(t * 2.9) * 0.4
    if move == "whip pan":
        return 1.1, (-0.5 + 1.0 * e) * w * 0.35, 0, 0
    if move == "dolly zoom":
        return 1.0 + 0.2 * e, 0, 0, 0
    return 1.0 + 0.12 * e, 0, 0, 0  # slow push-in (default)


def still_motion(frame_path, move, seconds, out_fps, dest, cancel=None):
    """CPU End-Frame Motion: animate one frame with a camera move, with temporal super-sampling for motion blur."""
    img = cv2.imread(str(frame_path))
    img = cv2.resize(img, (config.OUT_W, config.OUT_H), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    n = max(2, int(round(seconds * out_fps)))
    sub = 6 if move == "whip pan" else 3
    enc = subprocess.Popen(encode_cmd(dest, out_fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(n):
            if cancel and i % 15 == 0 and cancel():
                raise media.Canceled()
            acc = np.zeros((h, w, 3), np.float32)
            for k in range(sub):
                t = (i + k / sub) / (n - 1) if n > 1 else 0
                s, dx, dy, deg = _move_params(move, t, w, h)
                m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, s)
                m[:, 2] += (dx, dy)
                acc += cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            enc.stdin.write((acc / sub).clip(0, 255).astype(np.uint8).tobytes())
    finally:
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        enc.wait()
    if enc.returncode != 0:
        raise media.MediaError("encode failed: " + err[-300:])


def _lab_stats(frames):
    labs = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames]
    stack = np.concatenate([lab.reshape(-1, 3) for lab in labs])
    return stack.mean(axis=0), stack.std(axis=0) + 1e-3


def _webp_to_frames(path, out_dir):
    from PIL import Image
    im = Image.open(path)
    frames = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        p = out_dir / f"frame_{i:05d}.png"
        im.convert("RGB").save(p)
        frames.append(p)
    return frames


def conform_generated(files, ref_frame, seconds, out_fps, dest, bridge, cancel=None, strength=0.85):
    """Scale/crop a generated clip to 1080x1920, retime it to `seconds` (trim for a continuation, time-fit for a
    first/last-frame bridge), and colour-match it to the real reference frame (Reinhard transfer in Lab)."""
    work = dest.parent
    if len(files) == 1 and files[0].suffix == ".webp":
        files = _webp_to_frames(files[0], work)
    if len(files) > 1:
        src_args = ["-framerate", "16", "-i", str(files[0].parent / "frame_%05d.png")]
    else:
        src_args = ["-i", str(files[0])]
    probe_src = files[0] if len(files) == 1 else None
    gen_dur = media.probe(probe_src)["duration"] if probe_src else len(files) / 16
    retime = f"setpts={seconds / gen_dur:.5f}*PTS," if bridge and gen_dur > 0 else ""
    vf = (f"{retime}scale={config.OUT_W}:{config.OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={config.OUT_W}:{config.OUT_H},framerate=fps={out_fps},format=bgr24")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *src_args, "-t", f"{seconds:.3f}", "-an", "-vf", vf,
           "-f", "rawvideo", "-"]
    raw = media.run(cmd, cancel=cancel, capture=True, timeout=600)
    fsize = config.OUT_W * config.OUT_H * 3
    frames = [np.frombuffer(raw[i:i + fsize], np.uint8).reshape(config.OUT_H, config.OUT_W, 3) for i in range(0, len(raw) - fsize + 1, fsize)]
    if not frames:
        raise media.MediaError("generated clip had no frames")
    n = int(round(seconds * out_fps))
    frames = (frames + [frames[-1]] * n)[:n]  # hold the last frame if the model returned a shorter clip
    ref = cv2.resize(cv2.imread(str(ref_frame)), (config.OUT_W, config.OUT_H), interpolation=cv2.INTER_AREA)
    mu_r, sd_r = _lab_stats([ref])
    mu_g, sd_g = _lab_stats(frames[:: max(1, len(frames) // 8)])
    enc = subprocess.Popen(encode_cmd(dest, out_fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
            matched = (lab - mu_g) / sd_g * sd_r + mu_r
            lab = lab + (matched - lab) * strength
            out = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            enc.stdin.write(out.tobytes())
    finally:
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        enc.wait()
    if enc.returncode != 0:
        raise media.MediaError("encode failed: " + err[-300:])


def make_transition(last_frame, next_first_frame, join, seconds, out_fps, dest, work, log, cancel=None, allow_ai=True):
    """Returns 'ai' when ComfyUI generated it, 'still' when the CPU fallback was used."""
    move, effect = join.get("camera_move", "slow push-in"), join.get("effect", "none")
    if allow_ai and comfy.enabled():
        prompt = camera_prompt(move, effect)
        log(f"End-Frame Motion via ComfyUI: \"{prompt}\"")
        try:
            gen_dir = work / "gen"
            files = comfy.generate(last_frame, prompt, gen_dir, end_frame=next_first_frame, cancel=cancel, log=log)
            bridge = len(comfy.inspect(comfy.load_workflow())["image_nodes"]) >= 2
            conform_generated(files, last_frame, seconds, out_fps, dest, bridge, cancel=cancel)
            return "ai"
        except media.Canceled:
            raise
        except Exception as e:  # noqa: BLE001 - fall back to the CPU move rather than failing the short
            log(f"ComfyUI transition failed ({str(e)[:200]}); using still-frame motion instead")
    still_motion(last_frame, move, seconds, out_fps, dest, cancel=cancel)
    return "still"
