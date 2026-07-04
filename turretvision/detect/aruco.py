"""ArUco detector.

Role per design decision D1: calibration and ground-truth tool, NOT the primary
tracking detector. A marker of known size is the only thing in this project
that gives free, exact pose -- which is what validates the range estimator and
solves the boresight. Pose estimation itself lands in Phase 4 (needs intrinsics).
"""
from __future__ import annotations

import cv2

from ..capture.base import Frame
from .base import Detection, Detector


class ArucoDetector(Detector):
    def __init__(self, dictionary: str = "DICT_4X4_50", marker_size_m: float = 0.10, **_unused):
        dict_id = getattr(cv2.aruco, dictionary)
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dict_id),
            cv2.aruco.DetectorParameters(),
        )
        self.marker_size_m = marker_size_m

    def detect(self, frame: Frame) -> list[Detection]:
        gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        dets: list[Detection] = []
        if ids is None:
            return dets
        for quad in corners:
            pts = quad.reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            dets.append(Detection(cx=float(cx), cy=float(cy),
                                  area=float((x1 - x0) * (y1 - y0)),
                                  bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                                  kind="aruco", t=frame.t,
                                  corners=[(float(px), float(py)) for px, py in pts]))
        return dets
