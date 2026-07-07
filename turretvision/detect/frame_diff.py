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
suppresses output above a commanded-rate threshold.

Phase 4.5 (ego_motion_comp=true): the previous frame is WARPED by the camera
pose delta between the two frame timestamps (from telemetry, via set_ego_pose)
before differencing, cancelling the turret's own motion — detection then works
mid-slew and the gate is bypassed whenever a pose is available. The warp is a
pure translation: for a narrow-FOV camera rotating by fractions of a degree
per frame, the rotation-induced pixel flow is uniform to sub-pixel accuracy,
and a translation costs microseconds where a full homography warp costs
milliseconds. The strip of pixels the warp cannot fill is masked out of the
diff instead of hallucinating edge motion.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..capture.base import Frame
from .base import Detection, Detector


class FrameDiffDetector(Detector):
    def __init__(self, threshold: int = 25, min_area_px: int = 40, max_area_px: int = 40000,
                 morph_kernel: int = 3, static_gate_dps: float = 5.0,
                 three_frame: bool = False, ego_motion_comp: bool = False, **_unused):
        self._thresh = threshold
        self._min_a, self._max_a = min_area_px, max_area_px
        self._kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        self._gate_dps = static_gate_dps
        self._three = three_frame
        self._prev: np.ndarray | None = None
        self._prev2: np.ndarray | None = None
        self._turret_rate = 0.0
        self._ego = ego_motion_comp
        self.px_per_deg = 0.0            # main.py sets this from the mapper
        self._pose: tuple[float, float] | None = None        # current frame's pose
        self._prev_pose: tuple[float, float] | None = None   # aligned with _prev
        self._prev2_pose: tuple[float, float] | None = None  # aligned with _prev2

    def set_turret_rate(self, dps: float) -> None:
        self._turret_rate = abs(dps)

    def set_ego_pose(self, yaw_deg: float, pitch_deg: float) -> None:
        """Camera pose at the NEXT frame's timestamp (call before detect())."""
        self._pose = (yaw_deg, pitch_deg)

    def _shift_px(self, from_pose: tuple[float, float]) -> tuple[float, float]:
        """Pixel translation aligning a frame taken at from_pose to the current
        pose. Camera yaw + (pans right) moves the scene LEFT in the image;
        pitch + (up) moves it DOWN (rows grow downward)."""
        dyaw = self._pose[0] - from_pose[0]
        dpitch = self._pose[1] - from_pose[1]
        return -dyaw * self.px_per_deg, dpitch * self.px_per_deg

    def _warp(self, img: np.ndarray, from_pose: tuple[float, float]) -> np.ndarray:
        dx, dy = self._shift_px(from_pose)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return img
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _mask_uncovered(self, diff: np.ndarray, from_pose: tuple[float, float]) -> None:
        """Zero the strip the warp couldn't fill — replicated border pixels
        differenced against real scene read as a wall of fake motion."""
        dx, dy = self._shift_px(from_pose)
        h, w = diff.shape[:2]
        mx, my = min(int(abs(dx)) + 2, w), min(int(abs(dy)) + 2, h)
        if dx > 0.25:
            diff[:, :mx] = 0
        elif dx < -0.25:
            diff[:, w - mx:] = 0
        if dy > 0.25:
            diff[:my, :] = 0
        elif dy < -0.25:
            diff[h - my:, :] = 0

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
        prev_pose, prev2_pose = self._prev_pose, self._prev2_pose
        self._prev2, self._prev = prev, gray
        self._prev2_pose, self._prev_pose = prev_pose, self._pose

        if prev is None:
            return []
        comp = (self._ego and self.px_per_deg > 0
                and self._pose is not None and prev_pose is not None)
        if not comp and self._turret_rate > self._gate_dps:
            # Quasi-static gate: differencing is meaningless mid-slew unless
            # the warp (below) is available to cancel the ego motion.
            return []

        if comp:
            prev = self._warp(prev, prev_pose)
        d1 = cv2.absdiff(gray, prev)
        if comp:
            self._mask_uncovered(d1, prev_pose)
        if self._three and prev2 is not None and (not comp or prev2_pose is not None):
            # prev is already in current-frame coords when compensating; bring
            # prev2 there too so the AND compares apples to apples.
            prev2 = self._warp(prev2, prev2_pose) if comp else prev2
            d2 = cv2.absdiff(prev, prev2)
            if comp:
                self._mask_uncovered(d2, prev2_pose)
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
