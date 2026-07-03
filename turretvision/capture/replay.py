"""Replay a recorded video (with optional sidecar timestamps) as a FrameSource.

WHY this exists: it makes every bug reproducible. Record once, replay the exact
same frames through any code change, diff the CSV state logs. Also lets the full
pipeline run with zero hardware attached.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .base import Frame, FrameSource


class ReplaySource(FrameSource):
    def __init__(self, path: str, realtime: bool = True):
        self._path = Path(path)
        self._realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._stamps: list[float] | None = None
        self._idx = 0
        self._t_prev: float | None = None
        self._fps = 30.0
        self._res = (0, 0)

    def start(self) -> None:
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open replay file {self._path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._res = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        sidecar = self._path.with_suffix(".timestamps.jsonl")
        if sidecar.exists():
            # WHY recorded stamps beat synthesized ones: real capture has jitter and
            # drops; replaying the true dt sequence exercises the filter the same
            # way the live run did. Synthesized stamps hide timing bugs.
            self._stamps = [json.loads(line)["t"] for line in sidecar.read_text().splitlines()]

    def read(self) -> Frame | None:
        assert self._cap is not None
        ok, img = self._cap.read()
        if not ok:
            return None
        t = self._stamps[self._idx] if self._stamps and self._idx < len(self._stamps) \
            else self._idx / self._fps
        if self._realtime and self._t_prev is not None:
            time.sleep(max(0.0, t - self._t_prev))
        self._t_prev = t
        f = Frame(img=img, t=t, idx=self._idx)
        self._idx += 1
        return f

    def stop(self) -> None:
        if self._cap:
            self._cap.release()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res
