"""Range estimation contract.

The estimator gets the detection list plus the track's pixel position and
picks its own evidence (the detection nearest the track — the tracked object
is the thing we want the range TO, not whatever blob happens to be first).
Returns None when it has nothing trustworthy this frame; the lead predictor
holds its last estimate rather than aiming on garbage.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..detect.base import Detection


@dataclass(frozen=True)
class RangeEstimate:
    dist_m: float
    sigma_m: float      # honest uncertainty; consumers weight by it later
    method: str


class RangeEstimator(ABC):
    @abstractmethod
    def estimate(self, detections: list[Detection],
                 track_xy: tuple[float, float] | None) -> RangeEstimate | None: ...


def nearest_detection(detections: list[Detection],
                      track_xy: tuple[float, float] | None,
                      max_dist_px: float = 150.0) -> Detection | None:
    """The detection that belongs to the tracked object, if any."""
    if not detections:
        return None
    if track_xy is None:
        return detections[0]
    best = min(detections, key=lambda d: math.hypot(d.cx - track_xy[0], d.cy - track_xy[1]))
    if math.hypot(best.cx - track_xy[0], best.cy - track_xy[1]) > max_dist_px:
        return None
    return best
