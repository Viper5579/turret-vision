"""UVC camera controls via v4l2-ctl.

WHY v4l2-ctl instead of cv2.VideoCapture.set(CAP_PROP_EXPOSURE, ...): OpenCV's
V4L2 property mapping is backend/version-dependent and silently lossy — the
auto-exposure menu in particular round-trips wrong on many builds. v4l2-ctl
talks to the driver's control interface directly and is the tool every
troubleshooting doc in this repo already uses, so what the UI does and what
the docs say are the same thing.

Used two ways:
- V4L2Camera.start() applies `camera.v4l2_ctrls` from config so a saved tune
  (exposure/WB locked to manual) survives reboots and replugs.
- The tuning UI's Camera section reads and writes controls live.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

# name  0x009a0901 (menu) : min=0 max=3 default=3 value=3 flags=...
_CTRL_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")

# Autos must be switched to manual BEFORE their manual counterparts are set,
# or the driver rejects the write with "control inactive".
_APPLY_FIRST = ("auto_exposure", "white_balance_automatic")

Runner = Callable[[list[str]], str]


def _default_runner(args: list[str]) -> str:
    r = subprocess.run(["v4l2-ctl", *args], capture_output=True, text=True, timeout=5)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip())
    return r.stdout


class UvcControls:
    """Parsed control table for one device, with cached values."""

    def __init__(self, device: str, runner: Runner | None = None):
        self._device = device
        self._run = runner or _default_runner
        self.controls: dict[str, dict] = {}
        self.refresh()

    def refresh(self) -> None:
        out = self._run(["-d", self._device, "--list-ctrls"])
        ctrls: dict[str, dict] = {}
        for line in out.splitlines():
            m = _CTRL_LINE.match(line)
            if not m:
                continue
            name, typ, rest = m.groups()
            fields = dict(kv.split("=", 1) for kv in rest.split() if "=" in kv)
            ctrl: dict = {"type": typ, "flags": fields.get("flags", "")}
            for k in ("min", "max", "step", "default", "value"):
                if k in fields:
                    ctrl[k] = int(fields[k])
            ctrls[name] = ctrl
        self.controls = ctrls

    def get(self, name: str) -> int:
        return self.controls[name]["value"]

    def set(self, name: str, value: int) -> None:
        try:
            self._run(["-d", self._device, "-c", f"{name}={int(value)}"])
        except RuntimeError as e:
            # Typical cause: writing exposure_time_absolute while auto_exposure
            # is still in auto (control is flags=inactive). Not fatal.
            print(f"[uvc] could not set {name}={int(value)}: {e} "
                  f"(is the matching auto mode still on?)")
            return
        self.controls[name]["value"] = int(value)


def probe(device: str, runner: Runner | None = None) -> UvcControls | None:
    """UvcControls for the device, or None when v4l2-ctl/the device is absent."""
    if runner is None and shutil.which("v4l2-ctl") is None:
        return None
    try:
        cam = UvcControls(device, runner)
    except (RuntimeError, OSError):
        return None
    return cam if cam.controls else None


def apply_ctrls(device: str, ctrls: dict, runner: Runner | None = None) -> None:
    """Apply a config {name: value} mapping at startup (autos first)."""
    if not ctrls:
        return
    if runner is None and shutil.which("v4l2-ctl") is None:
        print("[uvc] camera.v4l2_ctrls configured but v4l2-ctl is not installed; skipping")
        return
    run = runner or _default_runner
    ordered = sorted(ctrls.items(), key=lambda kv: kv[0] not in _APPLY_FIRST)
    for name, value in ordered:
        try:
            run(["-d", device, "-c", f"{name}={int(value)}"])
        except (RuntimeError, OSError) as e:
            print(f"[uvc] could not apply {name}={value}: {e}")
    print(f"[uvc] applied {len(ordered)} camera control(s) from config: "
          + ", ".join(f"{k}={v}" for k, v in ordered))
