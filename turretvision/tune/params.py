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
    kind: str                 # "int" | "float"
    help: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None]

    def coerce(self, value: Any) -> Any:
        v = float(value)
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
                 "help": s.help, "value": s.getter()} for s in self.specs.values()]


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
