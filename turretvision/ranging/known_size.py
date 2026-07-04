"""Range from apparent size: dist = f_px * real_size / pixel_size.

The pinhole similar-triangles relation, valid for any object of roughly known
physical extent (paper airplane wingspan). WHY the bbox's LARGER side: a
banking airplane forshortens one axis long before the other; the larger
projected extent is the more stable proxy for wingspan. Error budget per
SPEC 9: ~25% — size varies with aspect, so sigma reflects that honestly.
"""
from __future__ import annotations

from ..detect.base import Detection
from .base import RangeEstimate, RangeEstimator, nearest_detection


class KnownSizeRange(RangeEstimator):
    def __init__(self, focal_px: float, target_size_m: float = 0.30,
                 min_px: float = 4.0, **_unused):
        self._f = focal_px
        self._size = target_size_m
        self._min_px = min_px

    def estimate(self, detections: list[Detection],
                 track_xy: tuple[float, float] | None) -> RangeEstimate | None:
        det = nearest_detection(detections, track_xy)
        if det is None:
            return None
        px = float(max(det.bbox[2], det.bbox[3]))
        if px < self._min_px:
            return None    # a 3px blob's "size" is mostly threshold noise
        d = self._f * self._size / px
        return RangeEstimate(dist_m=d, sigma_m=d * 0.25, method="known_size")
