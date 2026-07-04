"""Exact range from ArUco marker pose (solvePnP with known marker size).

This is the ground-truth instrument (D1): a marker of known physical size
plus calibrated intrinsics gives full 6-DoF pose, and pose gives range with
~cm accuracy — it's what the <15%-at-2/4/6m exit criterion is measured with,
and what validates the known_size estimator against tape-measure truth.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:      # pragma: no cover
    cv2 = None

from ..detect.base import Detection
from .base import RangeEstimate, RangeEstimator, nearest_detection


class ArucoPoseRange(RangeEstimator):
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray | None,
                 marker_size_m: float = 0.10, **_unused):
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self._D = (np.zeros(5) if dist_coeffs is None
                   else np.asarray(dist_coeffs, dtype=np.float64))
        s = marker_size_m / 2.0
        # Marker corners in its own frame, matching cv2.aruco corner order
        # (top-left, top-right, bottom-right, bottom-left), z=0 plane.
        self._obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                             dtype=np.float64)

    def estimate(self, detections: list[Detection],
                 track_xy: tuple[float, float] | None) -> RangeEstimate | None:
        det = nearest_detection([d for d in detections if d.corners], track_xy)
        if det is None or det.corners is None:
            return None
        img = np.array(det.corners, dtype=np.float64)
        ok, _rvec, tvec = cv2.solvePnP(self._obj, img, self._K, self._D,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        d = float(np.linalg.norm(tvec))
        if d <= 0:
            return None
        return RangeEstimate(dist_m=d, sigma_m=max(0.02, d * 0.03), method="aruco_pose")
