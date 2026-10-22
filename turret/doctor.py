"""`turret doctor`: check everything the system depends on, and say how to fix each failure.

v1 failed on the Jetson in ways that took an evening each to diagnose (pip
replacing JetPack's OpenCV, a camera stuck in the wrong mode, exposure capping
the frame rate). Every one of those is now a check here that runs before
anything starts, and every failure prints the fix.
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config, load_config

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 127, str(e)


def check_python() -> list[Check]:
    out = []
    v = sys.version_info
    out.append(
        Check(
            "python",
            PASS if v >= (3, 10) else FAIL,
            f"{v.major}.{v.minor}.{v.micro}",
            "Use the container (tv up) or Python 3.10+.",
        )
    )
    try:
        import numpy as np

        major = int(np.__version__.split(".")[0])
        out.append(
            Check(
                "numpy",
                PASS if major < 2 else WARN,
                np.__version__,
                "numpy 2 breaks TensorRT 10.3 bindings (needed in M4). The container pins numpy<2.",
            )
        )
    except ImportError as e:
        out.append(Check("numpy", FAIL, str(e), "pip install -e . (or use the container)"))
    try:
        import cv2
        import numpy as np

        img = np.zeros((800, 1280, 3), np.uint8)
        img[300:500, 500:800] = 200
        ok, buf = cv2.imencode(".jpg", img)
        half = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)
        good = ok and half is not None and half.shape == (400, 640)
        out.append(
            Check(
                "opencv",
                PASS if good else FAIL,
                f"{cv2.__version__}, reduced JPEG decode {'ok' if good else 'BROKEN'}",
                "Reinstall opencv-python-headless in the container.",
            )
        )
    except ImportError as e:
        out.append(Check("opencv", FAIL, str(e), "pip install -e . (or use the container)"))
    missing = []
    for mod in ("fastapi", "uvicorn", "websockets", "msgpack", "pydantic", "serial"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    out.append(
        Check(
            "python packages",
            PASS if not missing else FAIL,
            "all present" if not missing else f"missing: {', '.join(missing)}",
            "pip install -e . (the container already has them)",
        )
    )
    return out


def check_platform(cfg: Config) -> list[Check]:
    out = []
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        text = rel.read_text()
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", text)
        ver = f"R{m.group(1)}.{m.group(2)}" if m else text.splitlines()[0]
        good = bool(m) and m.group(1) == "36"
        out.append(
            Check("jetpack", PASS if good else WARN, ver, "Expected JetPack 6 (L4T R36.x). Do not upgrade mid-project.")
        )
    else:
        out.append(Check("jetpack", SKIP, "not a Jetson (fine for the simulator)"))
    data = Path(cfg.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(data).free / 1e9
    status = PASS if free_gb >= 50 else WARN if free_gb >= 10 else FAIL
    if status == FAIL and not rel.exists():
        status = WARN  # only the Jetson records sessions; elsewhere low disk is not fatal
    out.append(
        Check(
            "disk",
            status,
            f"{free_gb:.1f} GB free at {data.resolve()}",
            "Recording needs room: install the NVMe SSD and put data/ on it (plan: M0).",
        )
    )
    return out


def _parse_formats(text: str) -> list[tuple[str, int, int, float]]:
    modes = []
    fmt = None
    size = None
    for line in text.splitlines():
        m = re.search(r"'(\w+)'", line)
        if "[" in line and m:
            fmt = m.group(1)
        m = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and fmt and size:
            modes.append((fmt, size[0], size[1], float(m.group(1))))
    return modes


def check_camera(cfg: Config) -> list[Check]:
    cc = cfg.camera
    dev = cc.device
    if not Path(dev).exists():
        cams = sorted(glob.glob("/dev/video*"))
        return [
            Check(
                "camera",
                FAIL if Path("/etc/nv_tegra_release").exists() else SKIP,
                f"{dev} not found (video devices: {', '.join(cams) or 'none'})",
                "Plug in the OV9782 USB camera. If it shows up under another /dev/videoN, set camera.device.",
            )
        ]
    if shutil.which("v4l2-ctl") is None:
        return [Check("camera", WARN, f"{dev} present; v4l2-ctl missing", "apt install v4l-utils (in the container)")]
    code, text = _run(["v4l2-ctl", "-d", dev, "--list-formats-ext"])
    if code != 0:
        return [Check("camera", FAIL, text[-200:], "Camera present but not answering; replug it.")]
    modes = _parse_formats(text)
    want = [m for m in modes if m[0] == "MJPG" and m[1] == cc.width and m[2] == cc.height]
    best = max((m[3] for m in want), default=0.0)
    out = [
        Check(
            "camera mode",
            PASS if best >= cc.trigger_hz else FAIL,
            f"MJPG {cc.width}x{cc.height} up to {best:.0f} fps"
            if want
            else f"no MJPG {cc.width}x{cc.height} mode ({len(modes)} modes offered)",
            "The OV9782 does 100 fps only in MJPG; YUYV is 10 fps. Check camera.width/height.",
        )
    ]
    code, text = _run(["v4l2-ctl", "-d", dev, "--list-ctrls"])
    ctrls = set(re.findall(r"^\s*(\w+)\s+0x", text, re.M))
    needed = {"exposure_time_absolute", "auto_exposure"}
    missing = needed - ctrls
    out.append(
        Check(
            "camera controls",
            PASS if not missing else WARN,
            "manual exposure available" if not missing else f"missing: {', '.join(sorted(missing))}",
            "Manual exposure is required (auto exposure caps fps and fakes motion).",
        )
    )
    trig = "exposure_auto_priority" in ctrls
    out.append(
        Check(
            "trigger control",
            PASS if trig else WARN,
            "exposure_auto_priority present (trigger mode switch)" if trig else "no trigger-mode control",
            "Without trigger mode we fall back to free-run + gyro correlation sync (plan, Vision step 1).",
        )
    )
    return out


def find_serial_ports() -> list[str]:
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def check_esp32(cfg: Config) -> list[Check]:
    ports = find_serial_ports() if cfg.link.port == "auto" else [cfg.link.port]
    if not ports:
        return [
            Check(
                "esp32",
                WARN,
                "no serial port found",
                "Plug in the ESP32 by USB (it appears as /dev/ttyUSB0 with the CP2102).",
            )
        ]
    try:
        from .link import messages as m
        from .link.link import Link
        from .link.transport import SerialTransport

        t = SerialTransport(ports[0], cfg.link.baud)
    except Exception as e:
        return [Check("esp32", WARN, f"{ports[0]}: {e}", "Port busy? Stop the running system (tv down) first.")]
    link = Link(t, time.monotonic)
    end = time.monotonic() + 3.0
    while time.monotonic() < end and not (link.connected or link.stats.hash_mismatch):
        link.poll()
        time.sleep(0.005)
    while link.connected and time.monotonic() < end + 1.0:
        link.poll()
        time.sleep(0.005)
    link.close()
    if link.stats.hash_mismatch:
        return [Check("esp32", FAIL, f"{ports[0]}: firmware speaks another protocol version", "Run: tv flash")]
    if not link.connected:
        return [
            Check(
                "esp32",
                FAIL,
                f"{ports[0]}: no answer to Hello",
                "Flash the v2 firmware (tv flash). If flashed, check the baud rate (921600).",
            )
        ]
    sync = link.clock
    out = [Check("esp32", PASS, f"{ports[0]}: firmware v{link.stats.fw_version}, protocol 0x{m.PROTO_HASH:08X}")]
    if sync.valid:
        res_ms = sync.residual_s * 1000
        out.append(
            Check(
                "clock sync",
                PASS if res_ms < 0.5 else WARN,
                f"residual {res_ms:.3f} ms, min round trip {sync.min_delay_s * 1000:.2f} ms",
                "Sync error over 0.5 ms: try another USB port or cable.",
            )
        )
    return out


def check_compute(cfg: Config, frames: int = 60) -> list[Check]:
    """Time the real per-frame vision work on synthetic frames."""
    try:
        import cv2
        import numpy as np

        from .geometry import cam_to_base, rotation_homography, scale_intrinsics
        from .sim.plane import Room
        from .sim.render import Renderer, encode_jpeg
        from .vision.mosaic import BackgroundMosaic
        from .vision.motion import MotionDetector
    except ImportError as e:
        return [Check("compute", FAIL, str(e))]
    cc = cfg.camera
    rnd = Renderer(cc.width, cc.height, cc.hfov_deg, cc.tilt_deg, Room(), 1, 2.0)
    jpegs = [encode_jpeg(rnd.render([y], None, []), 85) for y in np.linspace(0, 6, frames)]
    Ks = scale_intrinsics(rnd.K, 1 / cc.decode_scale)
    det = MotionDetector(cfg.motion, cc.decode_scale)
    mos = BackgroundMosaic(Ks, cc.width // cc.decode_scale, cc.height // cc.decode_scale, cc.tilt_deg)
    hist = []
    times = []
    for i, j in enumerate(jpegs):
        yaw = 6.0 * i / (frames - 1)
        t0 = time.perf_counter()
        g = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        g = cv2.GaussianBlur(g, (3, 3), 0.8)
        bg, known = mos.predict(yaw)
        if len(hist) >= 2:
            R = cam_to_base(yaw, cc.tilt_deg)
            H1 = rotation_homography(Ks, cam_to_base(hist[-1][1], cc.tilt_deg), R)
            H2 = rotation_homography(Ks, cam_to_base(hist[-2][1], cc.tilt_deg), R)
            det.detect(g, hist[-1][0], H1, hist[-2][0], H2, bg, known, None)
        if i % 2 == 0:
            mos.update(g, yaw)
        hist.append((g, yaw))
        times.append(time.perf_counter() - t0)
    t = sorted(times[5:])
    p50 = t[len(t) // 2] * 1000
    budget = 1000.0 / cc.trigger_hz
    status = PASS if p50 < 0.6 * budget else WARN if p50 < 0.9 * budget else FAIL
    return [
        Check(
            "compute",
            status,
            f"vision per frame {p50:.1f} ms (p50), budget {budget:.1f} ms at {cc.trigger_hz:.0f} fps",
            "Over budget: check power mode (sudo nvpmodel -m 2 for MAXN_SUPER) and sudo jetson_clocks.",
        )
    ]


def run_all(cfg: Config | None = None, quick: bool = False) -> list[Check]:
    cfg = cfg or load_config()
    checks = check_python() + check_platform(cfg) + check_camera(cfg)
    if not quick:
        checks += check_esp32(cfg)
        checks += check_compute(cfg)
    return checks


_COLORS = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", SKIP: "\033[90m"}


def main(as_json: bool = False, quick: bool = False) -> int:
    checks = run_all(quick=quick)
    if as_json:
        print(json.dumps([asdict(c) for c in checks], indent=1))
    else:
        tty = sys.stdout.isatty()
        for c in checks:
            tag = c.status.upper().ljust(4)
            if tty:
                tag = f"{_COLORS[c.status]}{tag}\033[0m"
            print(f"  {tag}  {c.name:<16} {c.detail}")
            if c.status in (WARN, FAIL) and c.fix:
                print(f"        {'':16} -> {c.fix}")
        bad = sum(c.status == FAIL for c in checks)
        warn = sum(c.status == WARN for c in checks)
        if bad:
            print(f"\n{bad} check(s) failed, {warn} warning(s).")
        else:
            print(f"\nNo failures{f', {warn} warning(s)' if warn else '. All green'}.")
    return 1 if any(c.status == FAIL for c in checks) else 0
