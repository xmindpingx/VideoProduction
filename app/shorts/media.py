"""ffprobe / ffmpeg helpers: probing iPhone clips (HDR, rotation, 60/120/240 fps) and running jobs cancelably."""
import json
import subprocess
import time
from fractions import Fraction

from . import config


class MediaError(RuntimeError):
    pass


class Canceled(RuntimeError):
    pass


def run(cmd, cancel=None, timeout=None, stdin=None, capture=False):
    """Run a command; raise MediaError with the stderr tail on failure, Canceled if cancel() turns true."""
    proc = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    start = time.time()
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if cancel and cancel():
                    proc.kill()
                    proc.wait()
                    raise Canceled()
                if timeout and time.time() - start > timeout:
                    proc.kill()
                    proc.wait()
                    raise MediaError(f"{cmd[0]} timed out after {timeout}s")
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        raise
    if proc.returncode != 0:
        tail = (err or b"").decode("utf-8", "replace").strip().splitlines()[-12:]
        raise MediaError(f"{cmd[0]} failed ({proc.returncode}): " + " | ".join(tail))
    return out


def ffmpeg(*args, cancel=None, timeout=None):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if config.FFMPEG_THREADS:
        cmd += ["-threads", str(config.FFMPEG_THREADS)]
    return run(cmd + [str(a) for a in args], cancel=cancel, timeout=timeout)


def _fps(s):
    try:
        f = Fraction(s)
        return float(f) if f > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path):
    """Summarize a media file. Width/height are display size (after rotation)."""
    out = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], capture=True, timeout=120)
    info = json.loads(out)
    fmt = info.get("format", {})
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(fmt.get("duration") or (v or a or {}).get("duration") or 0)
    res = {"duration": round(duration, 3), "format": fmt.get("format_name", ""), "has_video": bool(v), "has_audio": bool(a)}
    if a:
        res["audio"] = {"codec": a.get("codec_name"), "rate": int(a.get("sample_rate") or 0), "channels": a.get("channels")}
    if not v:
        return res
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and v.get("tags", {}).get("rotate"):
        rotation = int(v["tags"]["rotate"])
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if abs(rotation) % 180 == 90:
        w, h = h, w
    avg = _fps(v.get("avg_frame_rate", "0/1"))
    rfr = _fps(v.get("r_frame_rate", "0/1"))
    nb = int(v.get("nb_frames") or 0)
    if nb and duration:  # iPhone files can be variable frame rate; count frames when we can
        avg = nb / duration
    fps = avg or rfr
    trc = v.get("color_transfer", "")
    hdr = "hlg" if trc == "arib-std-b67" else "pq" if trc == "smpte2084" else None
    dovi = any("DOVI" in (sd.get("side_data_type") or "").upper() or "dolby" in (sd.get("side_data_type") or "").lower()
               for sd in v.get("side_data_list", []) or [])
    res.update({
        "width": w, "height": h, "rotation": rotation, "fps": round(fps, 3), "r_fps": round(rfr, 3),
        "codec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "hdr": hdr, "dolby_vision": dovi,
        "color_primaries": v.get("color_primaries"), "color_space": v.get("color_space"),
        "high_fps": fps >= 90,  # 120/240 fps capture: can be slowed down cleanly
    })
    return res


def sdr_filter(meta):
    """Tone-map iPhone HDR (HLG / Dolby Vision base layer, or PQ) to SDR BT.709; empty string for SDR input."""
    hdr = meta.get("hdr")
    if not hdr:
        return ""
    tin = "arib-std-b67" if hdr == "hlg" else "smpte2084"
    return (f"format=yuv420p10le,zscale=tin={tin}:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100,format=gbrpf32le,"
            f"zscale=p=bt709,tonemap=tonemap={config.HDR_TONEMAP}:desat=0,"
            f"zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def join_filters(*parts):
    return ",".join(p for p in parts if p)


def video_tags():
    return ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def x264(crf=16, preset=None, tune=None):
    args = ["-c:v", "libx264", "-preset", preset or config.ENCODE_PRESET, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if tune:
        args += ["-tune", tune]
    return args + video_tags()


# ---------------------------------------------------------------- GPU (VA-API) video decode / encode
HW = {"checked": False, "enc": False, "dec": False, "note": "not checked"}


def hw_selftest(log=print):
    """Probe VA-API once per process: encode a few frames with h264_vaapi and decode HEVC with -hwaccel vaapi."""
    HW["checked"] = True
    if config.HWACCEL != "vaapi":
        HW["note"] = "disabled (HWACCEL=none)"
        return HW
    import os
    import tempfile
    if not os.path.exists(config.VAAPI_DEVICE):
        HW["note"] = f"{config.VAAPI_DEVICE} not found: map /dev/dri into the container"
        log("GPU video: " + HW["note"] + "; using software encode/decode")
        return HW
    dev = config.VAAPI_DEVICE
    with tempfile.TemporaryDirectory() as tmp:
        sample = os.path.join(tmp, "t.mp4")
        try:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", f"vaapi=va:{dev}", "-filter_hw_device", "va",
                 "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=0.5", "-vf", "format=nv12,hwupload",
                 "-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", "20", sample], timeout=60)
            HW["enc"] = True
        except MediaError as e:
            HW["note"] = f"h264_vaapi encode failed: {e}"
        if HW["enc"]:
            try:
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "vaapi", "-hwaccel_device", dev,
                     "-i", sample, "-f", "null", "-"], timeout=60)
                HW["dec"] = True
            except MediaError as e:
                HW["note"] = f"vaapi decode failed: {e}"
    if HW["enc"]:
        HW["note"] = "VA-API encode" + (" + decode" if HW["dec"] else "") + f" on {dev}"
    log("GPU video: " + HW["note"])
    return HW


def hw_init_args():
    """Global args that must precede the inputs when a VA-API encoder is used."""
    return ["-init_hw_device", f"vaapi=va:{config.VAAPI_DEVICE}", "-filter_hw_device", "va"] if HW["enc"] else []


def hw_decode_args():
    """Per-input args: decode on the GPU, frames come back to system memory (falls back to software if unsupported)."""
    return ["-hwaccel", "vaapi", "-hwaccel_device", config.VAAPI_DEVICE] if HW["dec"] else []


def encoder(quality=16, final=False):
    """(filter tail, codec args). Tail converts to BT.709 and, for VA-API, uploads frames to the GPU."""
    if HW["enc"]:
        return ("scale=out_color_matrix=bt709:out_range=tv,format=nv12,hwupload",
                ["-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", str(quality + (2 if final else 0)), "-profile:v", "high",
                 *video_tags()])
    return ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            x264(crf=quality + (2 if final else 0), preset=None if final else "fast", tune="grain" if final else None))


def thumbnail(src, meta, dest, at=None, width=360):
    at = at if at is not None else min(1.0, max(0.0, meta.get("duration", 0) / 3))
    vf = join_filters(sdr_filter(meta), f"scale={width}:-2")
    ffmpeg("-ss", f"{at:.3f}", "-i", src, "-frames:v", "1", "-vf", vf, "-q:v", "4", dest, timeout=120)


def magic_ok(head, kind):
    """Cheap container check before ffprobe: MP4/MOV (ftyp/moov/mdat/wide), Matroska/WebM, and for music MP3/WAV/AAC/FLAC/OGG."""
    if len(head) >= 8 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        return True
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    if kind == "music":
        if head[:3] == b"ID3" or head[:4] in (b"fLaC", b"OggS") or (head[:4] == b"RIFF" and head[8:12] == b"WAVE"):
            return True
        if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:  # MPEG audio / ADTS AAC frame sync
            return True
    return False
