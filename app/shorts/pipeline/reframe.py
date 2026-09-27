"""Smart 9:16 reframing with a virtual camera operator.

For every segment we build a crop path from the analysis (faces first, then motion), then smooth it the way a camera
operator would (offline, so there is no lag): a locked-off shot when the subject barely moves, a straight pan when it
drifts, or a smoothed follow when it moves around. A group that cannot fit a 9:16 crop gets the "fit + blurred fill"
layout instead of cutting someone out. Frames are rendered with sub-pixel warps (no jitter) and piped to x264.
"""
import subprocess

import cv2
import numpy as np

from .. import config, media


def crop_geometry(w, h):
    """Largest 9:16 window inside a w x h frame, as (crop_w, crop_h)."""
    target = config.OUT_W / config.OUT_H
    if w / h > target:
        return h * target, float(h)
    return float(w), w / target


def _interp_samples(times, values, t_out, default):
    if not times:
        return np.full(len(t_out), default, dtype=float)
    return np.interp(t_out, times, values)


def _gauss(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def subject_track(a, start, end, crop_frac_w, crop_frac_h):
    """Sample the subject centre (normalized) over [start, end] from faces, else motion. Returns times, cx, cy, spread, conf."""
    sc = a["scan"]
    fps = sc["fps"]
    times, cxs, cys, spreads, confs = [], [], [], [], []
    prev = None
    for i in range(max(0, int(start * fps) - 1), min(len(sc["motion"]), int(end * fps) + 2)):
        t = i / fps
        faces = sc["faces"][i]
        if faces is None:
            continue
        if faces:
            big = max(f[2] * f[3] for f in faces)
            main = [f for f in faces if f[2] * f[3] >= big * 0.35 and f[4] >= 0.6] or faces
            # Stick with the person we were already on unless someone else clearly dominates the frame.
            def pick_score(f):
                s = f[2] * f[3] * f[4]
                if prev is not None and abs(f[0] - prev) < crop_frac_w * 0.3:
                    s *= 1.6
                return s
            target = max(main, key=pick_score)
            xs = [f[0] for f in main]
            spread = (max(xs) - min(xs)) + max(f[2] for f in main)
            # Head room: put the eyes about a third down the frame, not dead centre.
            cy = target[1] + target[3] * 0.35
            times.append(t)
            cxs.append(target[0])
            cys.append(cy)
            spreads.append(spread if len(main) > 1 else 0.0)
            confs.append(1.0)
            prev = target[0]
        elif sc["motion"][i] > 0.004:
            times.append(t)
            cxs.append(0.5 + (sc["mcx"][i] - 0.5) * 0.8)  # motion centroid, pulled a little toward centre
            cys.append(0.5 + (sc["mcy"][i] - 0.5) * 0.6)
            spreads.append(0.0)
            confs.append(0.4)
    return times, cxs, cys, spreads, confs


def plan_camera(a, start, end, n_frames, speed, out_fps, src_w, src_h, zoom=1.0):
    """Per-output-frame crop centres (source pixels), crop size, and layout ('crop' or 'fit')."""
    cw, ch = crop_geometry(src_w, src_h)
    cw, ch = cw / zoom, ch / zoom
    t_out = start + np.arange(n_frames) / out_fps / speed  # source time of each output frame
    fw, fh = cw / src_w, ch / src_h
    times, cxs, cys, spreads, confs = subject_track(a, start, end, fw, fh)

    # A group wider than the crop for most of the shot: show everyone (fit + blurred fill) instead of cropping someone out.
    if spreads and np.mean([s > fw * 0.95 for s in spreads]) > 0.5 and fw < 0.9:
        return {"layout": "fit", "cw": cw, "ch": ch, "cx": np.full(n_frames, src_w / 2), "cy": np.full(n_frames, src_h / 2)}

    face_share = np.mean([c == 1.0 for c in confs]) if confs else 0.0
    cx = _interp_samples(times, cxs, t_out, 0.5)
    cy = _interp_samples(times, cys, t_out, 0.45 if face_share > 0.3 else 0.5)

    def operate(x, frac):
        """Locked-off, linear pan, or smoothed follow, like a camera operator would choose for this shot."""
        if frac >= 0.999:
            return np.full_like(x, 0.5)
        lo, hi = frac / 2, 1 - frac / 2
        x = np.clip(x, lo, hi)
        if np.ptp(x) < frac * 0.25:
            return np.full_like(x, float(np.median(x)))
        tt = np.arange(len(x))
        slope, icpt = np.polyfit(tt, x, 1)
        lin = slope * tt + icpt
        if np.max(np.abs(x - lin)) < frac * 0.18:
            return np.clip(lin, lo, hi)
        sm = _gauss(x, sigma=0.45 * out_fps)
        # cap pan speed so fast subjects do not whip the frame around
        max_step = frac * 1.2 / out_fps
        out = np.empty_like(sm)
        out[0] = sm[0]
        for i in range(1, len(sm)):
            out[i] = out[i - 1] + np.clip(sm[i] - out[i - 1], -max_step, max_step)
        return np.clip(_gauss(out, sigma=0.15 * out_fps), lo, hi)

    cx = operate(cx, fw) * src_w
    cy = operate(cy, fh) * src_h
    return {"layout": "crop", "cw": cw, "ch": ch, "cx": cx, "cy": cy}


def _decode_cmd(src, meta, start, dur, speed, out_fps, pipe_w, pipe_h, shutter):
    vf = []
    if speed != 1.0:
        vf.append(f"setpts={speed:.4f}*PTS")
    scale = f"scale={pipe_w}:{pipe_h}:flags=lanczos"
    if shutter and meta["fps"] / speed >= out_fps * 2:
        # ~180-degree shutter motion blur from the spare frames (60 fps -> 24/30); scale first so 4K is not blended
        vf += [scale, "tmix=frames=2", f"fps={out_fps}"]
    else:
        vf += [f"fps={out_fps}", scale]
    vf += [media.sdr_filter(meta), "format=bgr24"]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *media.hw_decode_args(),
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an", "-vf", media.join_filters(*vf), "-f", "rawvideo", "-"]


def encode_cmd(dest, out_fps):
    """Raw BGR frames on stdin -> 1080x1920 H.264 (GPU VA-API when available, else x264)."""
    tail, codec = media.encoder(15)
    return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *media.hw_init_args(),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{config.OUT_W}x{config.OUT_H}", "-r", str(out_fps), "-i", "-",
            "-vf", tail, *codec, "-r", str(out_fps), str(dest)]


def render_segment(src, meta, a, start, end, speed, out_fps, dest, zoom=1.0, shutter=False, cancel=None):
    """Render [start, end] of the source as a 1080x1920 clip (no audio). Returns the number of frames written."""
    n_frames = max(1, int(round((end - start) * speed * out_fps)))
    src_w, src_h = meta["width"], meta["height"]
    cam = plan_camera(a, start, end, n_frames, speed, out_fps, src_w, src_h, zoom)
    ow, oh = config.OUT_W, config.OUT_H

    # Decode at a size where the crop window maps ~1:1 onto 1080x1920, so warps never upscale twice.
    cw0, ch0 = crop_geometry(src_w, src_h)
    s = oh / ch0 * zoom if cam["layout"] == "crop" else min(1.0, 1920 / max(src_w, src_h))
    pipe_w, pipe_h = int(round(src_w * s / 2) * 2), int(round(src_h * s / 2) * 2)
    s_real = pipe_w / src_w

    dec = subprocess.Popen(_decode_cmd(src, meta, start, end - start, speed, out_fps, pipe_w, pipe_h, shutter),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(encode_cmd(dest, out_fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    size = pipe_w * pipe_h * 3
    written = 0
    last = None
    try:
        while written < n_frames:
            if cancel and written % 30 == 0 and cancel():
                raise media.Canceled()
            buf = dec.stdout.read(size)
            if len(buf) < size:
                if last is None:
                    break
                frame = last  # source ran a frame short: hold the last frame so audio and video stay the same length
            else:
                frame = np.frombuffer(buf, np.uint8).reshape(pipe_h, pipe_w, 3)
                last = frame
            i = min(written, n_frames - 1)
            if cam["layout"] == "fit":
                out = _fit_layout(frame, ow, oh)
            else:
                cw, ch = cam["cw"] * s_real, cam["ch"] * s_real
                k = ow / cw
                x0 = cam["cx"][i] * s_real - cw / 2
                y0 = cam["cy"][i] * s_real - ch / 2
                x0 = min(max(x0, 0), pipe_w - cw)
                y0 = min(max(y0, 0), pipe_h - ch)
                m = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
                out = cv2.warpAffine(frame, m, (ow, oh), flags=cv2.INTER_CUBIC if k > 1.01 else cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_REPLICATE)
            enc.stdin.write(out.tobytes())
            written += 1
    finally:
        dec.stdout.close()
        dec.kill()
        dec.wait()
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        err = enc.stderr.read().decode("utf-8", "replace")
        enc.wait()
    if enc.returncode != 0:
        raise media.MediaError("encode failed: " + err[-400:])
    if written == 0:
        raise media.MediaError("no frames decoded for segment")
    return written


def _fit_layout(frame, ow, oh):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA)
    s = max(ow / w, oh / h)
    bg = cv2.resize(small, (int(w * s) + 2, int(h * s) + 2), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (bg.shape[0] - oh) // 2, (bg.shape[1] - ow) // 2
    bg = cv2.GaussianBlur(bg[y0:y0 + oh, x0:x0 + ow], (0, 0), 25)
    bg = (bg.astype(np.float32) * 0.45).astype(np.uint8)
    fs = ow / w
    fg = cv2.resize(frame, (ow, int(h * fs)), interpolation=cv2.INTER_AREA if fs < 1 else cv2.INTER_CUBIC)
    top = int(oh * 0.42 - fg.shape[0] / 2)
    top = max(0, min(oh - fg.shape[0], top))
    bg[top:top + fg.shape[0]] = fg
    return bg


def extract_frame(video, dest, which="last"):
    """Save the first or last frame of a rendered segment (the reference image for End-Frame Motion)."""
    if which == "first":
        media.ffmpeg("-i", video, "-frames:v", "1", dest, timeout=120)
    else:
        media.ffmpeg("-sseof", "-0.5", "-i", video, "-update", "1", "-q:v", "1", dest, timeout=120)
