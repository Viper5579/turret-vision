"""Monotonic timing + per-stage latency stats.

WHY monotonic everywhere: wall-clock time jumps (NTP, timezone) corrupt every
dt-based computation downstream -- velocity, filter gains, FPS. time.monotonic()
cannot go backwards, which is the only property the pipeline actually needs.
"""
from __future__ import annotations

import time
from collections import deque


def now() -> float:
    return time.monotonic()


class RollingRate:
    """FPS over a sliding window (window avg beats instantaneous 1/dt, which is jittery)."""

    def __init__(self, window: int = 60):
        self._stamps: deque[float] = deque(maxlen=window)

    def tick(self, t: float | None = None) -> None:
        self._stamps.append(t if t is not None else now())

    @property
    def hz(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class StageTimer:
    """Accumulates per-stage latency; report() gives rolling means in ms."""

    def __init__(self, window: int = 120):
        self._hist: dict[str, deque[float]] = {}
        self._window = window
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = now()

    def mark(self, stage: str) -> None:
        t = now()
        self._hist.setdefault(stage, deque(maxlen=self._window)).append((t - self._t0) * 1000.0)
        self._t0 = t

    def report(self) -> dict[str, float]:
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self._hist.items()}
