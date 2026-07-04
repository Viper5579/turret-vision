"""Primary detector: frame differencing.

WHY this is the primary detector (design decision D1): the target's defining
property is that it's the only fast-moving object in the scene. Differencing
exploits exactly that with zero training and trivially runs at 100 fps.

Known quirk documented on purpose: two-frame differencing produces TWO blobs
per moving object (where it was + where it is). We take the largest contour,
whose centroid sits near the leading edge; the alpha-beta filter downstream
smooths the residual wobble. Three-frame differencing (AND of consecutive
diffs) localizes cleanly on the middle frame at the cost of one frame of
latency -- available via config if the wobble ever matters.

v1 limitation (D1 phasing): while the turret itself rotates, EVERYTHING moves
in the image and differencing lights up the whole frame. The quasi-static gate
suppresses output above a commanded-rate threshold. Ego-motion compensation
(warp prev frame by telemetry pose delta) replaces the gate in Phase 4.5.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..capture.base import Frame
from .base import Detection, Detector


class FrameDiffDetector(Detector):
    def __init__(self, threshold: int = 25, min_area_px: int = 40, max_area_px: int = 40000,
                 morph_kernel: int = 3, static_gate_dps: float = 5.0,
                 three_frame: bool = False, **_unused):
        self._thresh = threshold
        self._min_a, self._max_a = min_area_px, max_area_px
        self._kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        self._gate_dps = static_gate_dps
        self._three = three_frame
        self._prev: np.ndarray | None = None
        self._prev2: np.ndarray | None = None
        self._turret_rate = 0.0

    def set_turret_rate(self, dps: float) -> None:
        self._turret_rate = abs(dps)

    # -- live-tunable knobs (tuning UI) -----------------------------------
    # Setters exist so the tuning server has a public surface that keeps
    # derived state (the morphology kernel) consistent with the value.

    @property
    def threshold(self) -> int:
        return self._thresh

    @threshold.setter
    def threshold(self, v: int) -> None:
        self._thresh = int(v)

    @property
    def min_area_px(self) -> int:
        return self._min_a

    @min_area_px.setter
    def min_area_px(self, v: int) -> None:
        self._min_a = int(v)

    @property
    def max_area_px(self) -> int:
        return self._max_a

    @max_area_px.setter
    def max_area_px(self, v: int) -> None:
        self._max_a = int(v)

    @property
    def morph_kernel(self) -> int:
        return self._kernel.shape[0]

    @morph_kernel.setter
    def morph_kernel(self, v: int) -> None:
        v = max(1, int(v))
        self._kernel = np.ones((v, v), np.uint8)

    @property
    def static_gate_dps(self) -> float:
        return self._gate_dps

    @static_gate_dps.setter
    def static_gate_dps(self, v: float) -> None:
        self._gate_dps = float(v)

    def detect(self, frame: Frame) -> list[Detection]:
        gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # WHY: kills single-pixel sensor noise
                                                  # before thresholding, cheaper than
                                                  # cleaning it up with morphology after
        prev, prev2 = self._prev, self._prev2
        self._prev2, self._prev = prev, gray

        if prev is None:
            return []
        if self._turret_rate > self._gate_dps:
            # Quasi-static gate: differencing is meaningless mid-slew (v1).
            return []

        d1 = cv2.absdiff(gray, prev)
        if self._three and prev2 is not None:
            d2 = cv2.absdiff(prev, prev2)
            diff = cv2.bitwise_and(d1, d2)
        else:
            diff = d1

        _, mask = cv2.threshold(diff, self._thresh, 255, cv2.THRESH_BINARY)
        # WHY open-then-dilate: open removes remaining speckle; dilate re-merges a
        # blob that thresholding split (e.g. a wing crossing a bright background).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: list[Detection] = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (self._min_a <= a <= self._max_a):
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(Detection(cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"],
                                  area=a, bbox=(x, y, w, h), kind="frame_diff", t=frame.t))
        # WHY sort by area desc: single-target assumption -> biggest mover is the
        # best candidate; tracker association gets first pick of the likeliest one.
        dets.sort(key=lambda d: -d.area)
        return dets
