"""Pixel <-> angle conversion + boresight offsets.

WHY angles this early in the pipeline (data-flow decision in SPEC 3): once the
target is expressed as (azimuth, elevation) degrees, every downstream consumer
(range, lead, protocol, logs) is in physical units. Change the lens or the
resolution and exactly ONE module changes -- this one.

Sign conventions (pinned here, nowhere else):
  azimuth  +right  (pixel u increasing)
  elevation +up    (pixel v DEcreasing -- image rows grow downward)
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml


class PixelAngleMapper:
    def __init__(self, width: int, height: int,
                 intrinsics_file: str | None = None,
                 fallback_hfov_deg: float = 70.0,
                 boresight_yaw_deg: float = 0.0,
                 boresight_pitch_deg: float = 0.0,
                 boresight_file: str | None = None,
                 undistort_points: bool = True):
        self._w, self._h = width, height
        self._bs_yaw, self._bs_pitch = boresight_yaw_deg, boresight_pitch_deg
        bs = Path(boresight_file) if boresight_file else None
        if bs and bs.exists():
            # tools/calibrate_boresight.py output stacks ON TOP of any manual
            # config offsets (which default to 0 and normally stay there).
            data = yaml.safe_load(bs.read_text()) or {}
            self._bs_yaw += float(data.get("boresight_yaw_deg", 0.0))
            self._bs_pitch += float(data.get("boresight_pitch_deg", 0.0))
            print(f"[calib] boresight offsets loaded from {bs}: "
                  f"yaw {self._bs_yaw:+.2f} pitch {self._bs_pitch:+.2f} deg")
        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._undistort = undistort_points
        self.calibrated = False

        p = Path(intrinsics_file) if intrinsics_file else None
        if p and p.exists():
            data = yaml.safe_load(p.read_text())
            self._K = np.array(data["camera_matrix"], dtype=np.float64)
            self._D = np.array(data["dist_coeffs"], dtype=np.float64)
            self.calibrated = True
            self._fx, self._fy = self._K[0, 0], self._K[1, 1]
            self._cx, self._cy = self._K[0, 2], self._K[1, 2]
        else:
            # Fallback: estimate focal length from horizontal FOV.
            # WHY this is acceptable pre-calibration: fx = (w/2)/tan(hfov/2) from a
            # datasheet FOV is ~5% wrong, which shifts angles proportionally --
            # fine for bring-up, replaced by ChArUco calibration in Phase 4.
            self._fx = (width / 2.0) / math.tan(math.radians(fallback_hfov_deg) / 2.0)
            self._fy = self._fx  # square pixels assumption, also fixed by calibration
            self._cx, self._cy = width / 2.0, height / 2.0

    def pixel_to_angles(self, u: float, v: float) -> tuple[float, float]:
        """(azimuth_deg, elevation_deg) relative to the LAUNCHER axis."""
        if self.calibrated and self._undistort and self._D is not None:
            # WHY undistort the POINT and not the frame (design decision D5): the
            # only place lens distortion corrupts the math is right here; a full
            # 100fps frame remap buys nothing but milliseconds of latency.
            pt = np.array([[[u, v]]], dtype=np.float64)
            und = cv2.undistortPoints(pt, self._K, self._D, P=self._K)
            u, v = float(und[0, 0, 0]), float(und[0, 0, 1])
        # WHY atan and not the small-angle linear map: at the edge of a 70 deg FOV
        # the linear approximation is off by multiple degrees -- a guaranteed miss.
        az = math.degrees(math.atan2(u - self._cx, self._fx))
        el = math.degrees(math.atan2(self._cy - v, self._fy))
        return az + self._bs_yaw, el + self._bs_pitch

    def angles_to_pixel(self, az_deg: float, el_deg: float) -> tuple[float, float]:
        """Inverse map (overlay rendering of angular quantities)."""
        az = math.radians(az_deg - self._bs_yaw)
        el = math.radians(el_deg - self._bs_pitch)
        u = self._cx + self._fx * math.tan(az)
        v = self._cy - self._fy * math.tan(el)
        return u, v

    @property
    def deg_per_px(self) -> float:
        """Approx scale at image center (used by ego-motion comp later)."""
        return math.degrees(1.0 / self._fx)

    @property
    def focal_px(self) -> float:
        """fx (calibrated or FOV-fallback) — ranging.known_size consumes this."""
        return self._fx

    @property
    def camera_matrix(self) -> np.ndarray | None:
        """3x3 K when calibrated, else None (aruco_pose ranging requires it)."""
        return self._K

    @property
    def dist_coeffs(self) -> np.ndarray | None:
        return self._D
