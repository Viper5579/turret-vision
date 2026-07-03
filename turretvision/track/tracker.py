"""Single-target tracker: association gating, coasting, confidence.

WHY single-target: the system's job is intercepting ONE designated object; a
multi-target association solver (Hungarian etc.) is complexity with no customer.
The area-sorted detection list means we still behave sanely with clutter: the
gate rejects far-away blobs, and among in-gate blobs the biggest mover wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..detect.base import Detection
from .filters import StateEstimator


@dataclass
class TrackState:
    x: float
    y: float
    vx: float
    vy: float
    t: float
    confidence: float
    coasting: bool
    valid: bool


class SingleTargetTracker:
    def __init__(self, estimator: StateEstimator, gate_px: float = 120.0,
                 max_coast_frames: int = 8):
        self._est = estimator
        self._gate = gate_px
        self._max_coast = max_coast_frames
        self._coast = 0
        self._conf = 0.0
        self._has_track = False

    def step(self, detections: list[Detection], t: float) -> TrackState | None:
        chosen: Detection | None = None
        if self._has_track:
            pred = self._est.predict(t)
            for d in detections:
                # WHY gate on distance-to-PREDICTION, not to last position: a fast
                # target moves several gate-widths between frames; gating on the
                # stale position would reject its own target at exactly the speeds
                # this project cares about.
                if pred and math.hypot(d.cx - pred.x, d.cy - pred.y) <= self._gate:
                    chosen = d
                    break
        elif detections:
            chosen = detections[0]  # acquisition: biggest mover

        if chosen is not None:
            s = self._est.update(chosen.cx, chosen.cy, t)
            self._has_track = True
            self._coast = 0
            # WHY EMA instead of hit-counter: confidence should decay smoothly on
            # misses and rebuild smoothly on hits, so the min_confidence_output
            # threshold produces hysteresis-ish behavior instead of flapping the
            # target_valid bit on the wire every other frame.
            self._conf = 0.85 * self._conf + 0.15 * 1.0
            return TrackState(s.x, s.y, s.vx, s.vy, t, self._conf, False, True)

        if self._has_track:
            self._coast += 1
            self._conf *= 0.80
            if self._coast > self._max_coast:
                # WHY hard drop: coasting forever means confidently aiming at a
                # ghost. Better to declare no-target and reacquire.
                self._has_track = False
                self._est.reset()
                self._conf = 0.0
                return None
            p = self._est.predict(t)
            if p:
                return TrackState(p.x, p.y, p.vx, p.vy, t, self._conf, True, True)
        return None
