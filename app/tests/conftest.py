"""Test setup: an isolated DATA_DIR, the small models/fonts (downloaded once into tests/.assets), synthetic clips."""
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ASSETS = HERE / ".assets"
FILES = {
    "models/deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "models/bd.rnnn": "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn",
    "fonts/Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/23e54b51ddffbc7713c583748e3bd86f62b1fa4a/ofl/anton/Anton-Regular.ttf",
    "fonts/ArchivoBlack-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/23e54b51ddffbc7713c583748e3bd86f62b1fa4a/ofl/archivoblack/ArchivoBlack-Regular.ttf",
}

_tmp = Path(tempfile.mkdtemp(prefix="vp-test-"))
os.environ.update({
    "DATA_DIR": str(_tmp / "data"), "ASSETS_DIR": os.environ.get("TEST_ASSETS_DIR", str(ASSETS)),
    "SHORTS_TOKEN": "test-token-0123456789", "OLLAMA_URL": "http://127.0.0.1:9", "COMFYUI_URL": "",
    "HWACCEL": "none", "AUDIO_CLEANUP": "rnnoise", "RETENTION_DAYS": "0",
})
sys.path.insert(0, str(HERE.parent))


def _fetch_assets():
    root = Path(os.environ["ASSETS_DIR"])
    for rel, url in FILES.items():
        p = root / rel
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="session")
def assets():
    _fetch_assets()
    return Path(os.environ["ASSETS_DIR"])


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Synthetic phone-like clips: 60 fps with speech, 240 fps (slow-mo capable), and rotated 10-bit HLG HDR."""
    d = tmp_path_factory.mktemp("media")
    ff("-f", "lavfi", "-i", "gradients=s=1920x1080:c0=0x203040:c1=0x604020:speed=0.02:d=10:r=60",
       "-f", "lavfi", "-i", "flite=text='Welcome to the studio. This is the moment everything changed. Watch what happens next.':voice=slt",
       "-filter_complex", "[0:v]drawbox=x='200+900*(0.5-0.5*cos(t*0.6))':y=300:w=300:h=400:color=white@0.9:t=fill,format=yuv420p[v];"
                          "[1:a]aresample=48000,apad=whole_dur=10[a]",
       "-map", "[v]", "-map", "[a]", "-t", 10, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", d / "talk60.mp4")
    ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=240:d=3", "-f", "lavfi", "-i", "sine=f=440:d=3",
       "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", d / "slomo240.mov")
    ff("-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=5", "-f", "lavfi", "-i", "anoisesrc=d=5:c=pink:a=0.05",
       "-vf", "format=yuv420p10le", "-c:v", "libx265", "-x265-params", "log-level=error", "-tag:v", "hvc1",
       "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-c:a", "aac", d / "hdr_tmp.mov")
    ff("-display_rotation", 90, "-i", d / "hdr_tmp.mov", "-c", "copy", d / "hdr_portrait.mov")
    ff("-f", "lavfi", "-i", "aevalsrc='0.8*sin(2*PI*60*t)*exp(-12*mod(t,0.5))':s=44100:d=30", "-c:a", "libmp3lame", d / "beat120.mp3")
    return d


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("KEEP_TEST_DATA"):
        print(f"\ntest data kept in {_tmp}")
        return
    shutil.rmtree(_tmp, ignore_errors=True)
