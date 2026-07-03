from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    img: np.ndarray
    t: float      # monotonic seconds, stamped AT GRAB (see v4l2.py for why)
    idx: int


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def read(self) -> Frame | None:
        """Newest available frame, or None if the source is exhausted/stopped."""

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height)"""
