"""HSV color-mask detector: static-target sanity check.

WHY it exists at all: it detects a *stationary* bright target, which frame
differencing by definition cannot. That makes it the tool for verifying the
geometry/overlay/link chain with a target taped to the wall.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..capture.base import Frame
from .base import Detection, Detector


class ColorMaskDetector(Detector):
    def __init__(self, hsv_low=(5, 120, 120), hsv_high=(25, 255, 255),
                 min_area_px: int = 60, **_unused):
        self._lo = np.array(hsv_low, dtype=np.uint8)
        self._hi = np.array(hsv_high, dtype=np.uint8)
        self._min_a = min_area_px
        self._kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lo, self._hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[Detection] = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self._min_a:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(Detection(cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"],
                                  area=a, bbox=(x, y, w, h), kind="color", t=frame.t))
        dets.sort(key=lambda d: -d.area)
        return dets
