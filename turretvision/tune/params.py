"""Registry of live-tunable parameters for the tuning UI.

WHY a registry instead of poking objects from HTTP handlers: writes arrive on
server threads but the pipeline reads the values in its own hot loop. Every
change is queued here and applied by the pipeline thread itself (apply_pending
at the top of each frame), so a slider drag can never observe a half-applied
detector state mid-detect. It also gives one place that knows each knob's
config path, bounds, and how to push it into the live object.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    key: str                  # dotted config path, doubles as the wire id
    label: str
    group: str
    vmin: float
    vmax: float
    step: float
    kind: str                 # "int" | "float" | "toggle"
    help: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None]
    # For "toggle": the raw values the two switch states map to. Not always
    # 1/0 — UVC auto_exposure is a menu where 3=auto and 1=manual.
    on_value: int = 1
    off_value: int = 0

    def coerce(self, value: Any) -> Any:
        v = float(value)
        if self.kind == "toggle":
            return self.off_value if v == float(self.off_value) else self.on_value
        v = min(max(v, self.vmin), self.vmax)
        return int(round(v)) if self.kind == "int" else round(v, 6)


@dataclass
class ParamRegistry:
    specs: dict[str, ParamSpec] = field(default_factory=dict)
    _pending: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, spec: ParamSpec) -> None:
        self.specs[spec.key] = spec

    def queue(self, key: str, value: Any) -> Any:
        """Queue a change from any thread. Returns the coerced value."""
        spec = self.specs[key]  # KeyError -> handler answers 404
        v = spec.coerce(value)
        with self._lock:
            self._pending[key] = v
        return v

    def apply_pending(self) -> None:
        """Apply queued changes; call from the pipeline thread only."""
        with self._lock:
            if not self._pending:
                return
            pending, self._pending = self._pending, {}
        for key, v in pending.items():
            self.specs[key].setter(v)

    def value(self, key: str) -> Any:
        return self.specs[key].getter()

    def snapshot(self) -> dict[str, Any]:
        return {k: s.getter() for k, s in self.specs.items()}

    def describe(self) -> list[dict]:
        return [{"key": s.key, "label": s.label, "group": s.group,
                 "min": s.vmin, "max": s.vmax, "step": s.step, "kind": s.kind,
                 "help": s.help, "value": s.getter(),
                 "on": s.on_value, "off": s.off_value} for s in self.specs.values()]


def _attr(obj: Any, name: str) -> tuple[Callable[[], Any], Callable[[Any], None]]:
    return (lambda: getattr(obj, name)), (lambda v: setattr(obj, name, v))


def build_registry(detector, filt, tracker, min_conf_ref: dict) -> ParamRegistry:
    """Wire up the tunable set for the frame_diff + alpha-beta pipeline.

    min_conf_ref is a {"value": float} holder because min_confidence_output is
    consumed by the main loop, not owned by any stage object.
    """
    from ..detect.frame_diff import FrameDiffDetector

    reg = ParamRegistry()
    if isinstance(detector, FrameDiffDetector):
        det = "Detector · frame_diff"
        g, s = _attr(detector, "threshold")
        reg.add(ParamSpec("detection.frame_diff.threshold", "Threshold", det, 1, 120, 1, "int",
                          "Abs-diff intensity cutoff. Lower catches dim/small movers; "
                          "raise if noise fires it.", g, s))
        g, s = _attr(detector, "min_area_px")
        reg.add(ParamSpec("detection.frame_diff.min_area_px", "Min area (px²)", det, 1, 2000, 1, "int",
                          "Blobs smaller than this are ignored. Raise to reject speckle.", g, s))
        g, s = _attr(detector, "max_area_px")
        reg.add(ParamSpec("detection.frame_diff.max_area_px", "Max area (px²)", det,
                          1000, 200000, 500, "int",
                          "Blobs bigger than this are ignored (e.g. global lighting shifts).", g, s))
        g, s = _attr(detector, "morph_kernel")
        reg.add(ParamSpec("detection.frame_diff.morph_kernel", "Morph kernel", det, 1, 9, 1, "int",
                          "Open/dilate kernel size. Bigger removes more speckle but "
                          "eats small targets.", g, s))
        g, s = _attr(detector, "static_gate_dps")
        reg.add(ParamSpec("detection.frame_diff.static_gate_dps", "Static gate (°/s)", det,
                          0, 60, 0.5, "float",
                          "Suppress detection while commanded turret rate exceeds this.", g, s))

    trk = "Tracker"
    g, s = _attr(filt, "alpha")
    reg.add(ParamSpec("tracking.alpha", "Alpha (position gain)", trk, 0.05, 1.0, 0.01, "float",
                      "Trust in each new position. High = responsive but jittery; "
                      "low = smooth but laggy.", g, s))
    g, s = _attr(filt, "beta")
    reg.add(ParamSpec("tracking.beta", "Beta (velocity gain)", trk, 0.01, 1.0, 0.01, "float",
                      "Trust in each velocity correction. Lower this second if "
                      "output is jittery.", g, s))
    g, s = _attr(tracker, "gate_px")
    reg.add(ParamSpec("tracking.gate_px", "Gate (px)", trk, 10, 600, 5, "int",
                      "Max distance from prediction for a detection to update the track.", g, s))
    g, s = _attr(tracker, "max_coast_frames")
    reg.add(ParamSpec("tracking.max_coast_frames", "Max coast frames", trk, 0, 60, 1, "int",
                      "Predict through this many missed frames before dropping the track.", g, s))

    out = "Output"
    reg.add(ParamSpec("tracking.min_confidence_output", "Min confidence", out, 0.0, 1.0, 0.01, "float",
                      "Below this the link reports target_valid=false.",
                      lambda: min_conf_ref["value"],
                      lambda v: min_conf_ref.__setitem__("value", v)))
    return reg


def add_camera_params(reg: ParamRegistry, cam) -> int:
    """Expose UVC driver controls (uvc_ctrl.UvcControls) that this camera
    actually reports. Keys live under camera.v4l2_ctrls so a Save persists the
    RAW driver values, which V4L2Camera re-applies on every startup.

    Returns the number of controls added.
    """
    grp = "Camera (UVC driver)"
    added = 0

    def _cam_attr(name):
        return (lambda: cam.get(name)), (lambda v: cam.set(name, v))

    def toggle(name, label, help_, on, off):
        nonlocal added
        if name not in cam.controls:
            return
        g, s = _cam_attr(name)
        reg.add(ParamSpec(f"camera.v4l2_ctrls.{name}", label, grp,
                          min(on, off), max(on, off), 1, "toggle", help_, g, s,
                          on_value=on, off_value=off))
        added += 1

    def slider(name, label, help_, max_cap=None):
        nonlocal added
        c = cam.controls.get(name)
        if not c or "min" not in c or "max" not in c:
            return
        vmax = min(c["max"], max_cap) if max_cap else c["max"]
        g, s = _cam_attr(name)
        reg.add(ParamSpec(f"camera.v4l2_ctrls.{name}", label, grp,
                          c["min"], vmax, c.get("step", 1), "int", help_, g, s))
        added += 1

    # UVC auto_exposure is a menu: 1 = Manual Mode, 3 = Aperture Priority (auto).
    toggle("auto_exposure", "Auto exposure",
           "Turn OFF for tracking. AE hunting shifts global brightness (false "
           "frame_diff motion) and long auto exposures cap fps below the mode rate.",
           on=3, off=1)
    # 5000 raw = 500ms — useless for tracking; cap the slider at a sane range.
    slider("exposure_time_absolute", "Exposure time",
           "Units of 100 µs (80 = 8 ms). Must fit the frame period: at 100 fps "
           "anything above ~100 caps the frame rate itself.", max_cap=500)
    slider("gain", "Gain",
           "Sensor gain. Raise THIS (not exposure) if the image is too dark at "
           "a short exposure; noise is cheaper than motion blur here.")
    toggle("white_balance_automatic", "Auto white balance",
           "Turn OFF for tracking; AWB hunting shifts colors globally between "
           "frames, which frame_diff reads as motion.", on=1, off=0)
    slider("white_balance_temperature", "WB temperature (K)",
           "Fixed color temperature, applies once auto WB is off.")
    slider("backlight_compensation", "Backlight comp",
           "Leave at 0 for tracking; it fights the manual exposure lock.")
    return added
