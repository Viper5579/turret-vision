"""Clock sync between the ESP32 (microseconds since boot) and the Jetson (monotonic s).

Every camera frame's pose comes from an ESP32 timestamp, so this mapping is
what puts frames, encoder, gyro, and commands on one time axis.

Method (NTP-style): the Jetson sends a ping at t1 (its clock), the ESP32
stamps receive t2 and reply t3 (its clock), the Jetson receives at t4.

    offset = ((t2 - t1) + (t3 - t4)) / 2     # esp - jetson, if the link is symmetric
    delay  = (t4 - t1) - (t3 - t2)           # round trip spent on the wire

USB-serial delay is asymmetric and jittery, but jitter only ever ADDS delay,
so the samples with the smallest round trip are the most trustworthy. We keep
a sliding window, take the lowest-delay quarter, and fit offset = a + b * t
so a slowly drifting crystal (tens of ppm) is tracked, not averaged away.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SyncSample:
    t_jetson: float  # midpoint of t1..t4, seconds
    offset: float  # esp - jetson, seconds
    delay: float  # round trip on the wire, seconds


class ClockSync:
    def __init__(self, window: int = 128, keep_fraction: float = 0.25) -> None:
        self._samples: deque[SyncSample] = deque(maxlen=window)
        self._keep = keep_fraction
        self._a = 0.0
        self._b = 0.0
        self._t0 = 0.0
        self.valid = False
        self.residual_s = float("inf")
        self.min_delay_s = float("inf")

    def add(self, t1: float, t2_us: int, t3_us: int, t4: float) -> SyncSample:
        t2 = t2_us * 1e-6
        t3 = t3_us * 1e-6
        s = SyncSample(
            t_jetson=0.5 * (t1 + t4),
            offset=0.5 * ((t2 - t1) + (t3 - t4)),
            delay=(t4 - t1) - (t3 - t2),
        )
        self._samples.append(s)
        self._fit()
        return s

    def _fit(self) -> None:
        if len(self._samples) < 3:
            if self._samples:
                best = min(self._samples, key=lambda s: s.delay)
                self._t0, self._a, self._b = best.t_jetson, best.offset, 0.0
                self.valid = True
                self.min_delay_s = best.delay
            return
        ordered = sorted(self._samples, key=lambda s: s.delay)
        k = max(3, int(len(ordered) * self._keep))
        best = ordered[:k]
        t = np.array([s.t_jetson for s in best])
        y = np.array([s.offset for s in best])
        self._t0 = float(t.mean())
        tc = t - self._t0
        span = float(tc.max() - tc.min())
        if span > 2.0:
            b, a = np.polyfit(tc, y, 1)
        else:  # too short a baseline to estimate drift honestly
            a, b = float(y.mean()), 0.0
        self._a, self._b = float(a), float(b)
        pred = self._a + self._b * tc
        self.residual_s = float(np.sqrt(np.mean((y - pred) ** 2)))
        self.min_delay_s = best[0].delay
        self.valid = True

    def offset_at(self, t_jetson: float) -> float:
        return self._a + self._b * (t_jetson - self._t0)

    def esp_to_jetson(self, t_esp_us: int | float) -> float:
        """Convert an ESP32 timestamp (us) to Jetson monotonic seconds."""
        t_esp = t_esp_us * 1e-6
        # Solve t_esp = t_j + a + b (t_j - t0) for t_j.
        return (t_esp - self._a + self._b * self._t0) / (1.0 + self._b)

    def jetson_to_esp_us(self, t_jetson: float) -> int:
        return int(round((t_jetson + self.offset_at(t_jetson)) * 1e6))

    @property
    def drift_ppm(self) -> float:
        return self._b * 1e6
