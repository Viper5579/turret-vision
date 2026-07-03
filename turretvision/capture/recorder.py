"""Tee frames + grab-timestamps to disk for later ReplaySource use.

WHY sidecar jsonl instead of relying on video fps metadata: the container fps
is a constant; real capture has jitter and dropped frames. Losing the true
timestamps makes replays lie about dt, and dt is what the filter eats.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .base import Frame


class Recorder:
    def __init__(self, out_path: str, fps: float, resolution: tuple[int, int]):
        self._path = Path(out_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, resolution)
        self._stamps = open(self._path.with_suffix(".timestamps.jsonl"), "w")

    def write(self, frame: Frame) -> None:
        self._writer.write(frame.img)
        self._stamps.write(json.dumps({"idx": frame.idx, "t": frame.t}) + "\n")

    def close(self) -> None:
        self._writer.release()
        self._stamps.close()
