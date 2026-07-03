"""Synthetic moving-target source with known ground truth.

WHY: (1) lets the whole pipeline run and be scored with no camera; (2) ground
truth is the only way to make "is the tracker right?" a number instead of a
vibe. Target flies a gentle sinusoidal arc (paper-airplane-ish) and wraps.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Frame, FrameSource


class SyntheticSource(FrameSource):
    def __init__(self, width: int = 800, height: int = 600, fps: int = 60,
                 speed_px: float = 8.0, radius_px: int = 14, noise: int = 6,
                 n_frames: int | None = None, realtime: bool = False):
        self._w, self._h, self._fps = width, height, fps
        self._speed, self._r, self._noise = speed_px, radius_px, noise
        self._n = n_frames
        self._realtime = realtime
        self._idx = 0
        self._rng = np.random.default_rng(42)  # WHY seeded: reproducible tests
        self.truth: list[tuple[float, float, float]] = []  # (t, x, y)

    def start(self) -> None:
        pass

    def _pos(self, i: int) -> tuple[float, float]:
        x = (60 + i * self._speed) % (self._w - 120) + 30
        y = self._h / 2 + 90 * math.sin(i * self._speed / 90.0)
        return x, y

    def read(self) -> Frame | None:
        if self._n is not None and self._idx >= self._n:
            return None
        if self._realtime:
            import time
            time.sleep(1.0 / self._fps)
        img = self._rng.integers(40, 40 + self._noise, (self._h, self._w, 3), dtype=np.uint8)
        x, y = self._pos(self._idx)
        cv2.circle(img, (int(x), int(y)), self._r, (30, 120, 250), -1)  # bright orange blob
        t = self._idx / self._fps
        self.truth.append((t, x, y))
        f = Frame(img=img, t=t, idx=self._idx)
        self._idx += 1
        return f

    def stop(self) -> None:
        pass

    @property
    def resolution(self) -> tuple[int, int]:
        return self._w, self._h
