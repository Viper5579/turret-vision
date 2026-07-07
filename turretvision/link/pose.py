"""Turret pose history + interpolation (Phase 4.5 infrastructure).

Ego-motion compensation needs the pose delta BETWEEN TWO FRAME TIMESTAMPS,
and telemetry arrives on its own ~100 Hz clock — so the pose at an arbitrary
frame time is a lookup + linear interpolation over recent telemetry samples.

Timestamps: samples are stored against the Jetson-side receive stamp (t_rx).
WHY t_rx and not the ESP32's t_ms + offset: at 100 Hz telemetry the receive
jitter (~1 ms) moves an interpolated pose by ~0.2 deg at worst slew speed —
acceptable for v1 warp compensation; the min-filtered clock offset
(SerialLink.clock_offset) exists for when it isn't.
"""
from __future__ import annotations

from collections import deque

from .base import Telemetry


class PoseHistory:
    def __init__(self, maxlen: int = 256):
        self._buf: deque[tuple[float, float, float]] = deque(maxlen=maxlen)  # t, yaw, pitch

    def add(self, telem: Telemetry) -> None:
        # Ignore duplicate polls of the same telemetry packet.
        if self._buf and telem.t_rx <= self._buf[-1][0]:
            return
        self._buf.append((telem.t_rx, telem.yaw_deg, telem.pitch_deg))

    def pose_at(self, t: float) -> tuple[float, float] | None:
        """(yaw_deg, pitch_deg) at time t, linearly interpolated. Clamps to the
        history's ends (extrapolating a pose invents motion that didn't happen)."""
        if not self._buf:
            return None
        buf = self._buf
        if t <= buf[0][0]:
            return buf[0][1], buf[0][2]
        if t >= buf[-1][0]:
            return buf[-1][1], buf[-1][2]
        # Walk from the newest end: frame times trail the newest telemetry
        # by less than a frame period, so this loop is ~1-2 steps.
        for i in range(len(buf) - 1, 0, -1):
            t0, y0, p0 = buf[i - 1]
            t1, y1, p1 = buf[i]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return y0 + a * (y1 - y0), p0 + a * (p1 - p0)
        return buf[-1][1], buf[-1][2]

    def __len__(self) -> int:
        return len(self._buf)
