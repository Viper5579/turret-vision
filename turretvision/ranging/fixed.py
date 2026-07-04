"""Configured-constant range (test mode / bring-up).

WHY it exists: lead prediction needs SOME range on day one, and indoors the
engagement distance is roughly known anyway. sigma is set huge on purpose so
anything downstream that weights by uncertainty treats it as the guess it is.
"""
from __future__ import annotations

from ..detect.base import Detection
from .base import RangeEstimate, RangeEstimator


class FixedRange(RangeEstimator):
    def __init__(self, fixed_distance_m: float = 4.0, **_unused):
        self._d = fixed_distance_m

    def estimate(self, detections: list[Detection],
                 track_xy: tuple[float, float] | None) -> RangeEstimate:
        return RangeEstimate(dist_m=self._d, sigma_m=self._d * 0.5, method="fixed")
