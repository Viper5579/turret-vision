from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..capture.base import Frame


@dataclass
class Detection:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    kind: str
    t: float
    # ArUco only: the 4 marker corner pixels, ordered as detected. Carried so
    # ranging/aruco_pose can solvePnP the exact pose instead of re-detecting.
    corners: list[tuple[float, float]] | None = None


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]: ...

    def set_turret_rate(self, dps: float) -> None:  # noqa: B027 (optional hook by design)
        """Commanded turret angular speed hint. Default: ignore.

        WHY it's on the base class: frame_diff needs it for the quasi-static
        gate (and later ego-motion comp); giving every detector the same hook
        means main.py doesn't special-case detector types.
        """
