"""Frames, rotations, and the pinhole camera.

Conventions (fixed for the whole codebase):
  Base frame: X forward (yaw = 0), Y left, Z up. Origin at the yaw axis.
  Yaw: positive turns left (counter-clockwise seen from above).
  Azimuth/elevation of a direction: az = atan2(Y, X), el = atan2(Z, hypot(X, Y)).
  Camera frame (OpenCV): x right, y down, z forward (optical axis).
  The camera rides the yaw stage with a fixed upward tilt; it never pitches.

Because camera motion is a pure rotation, the background moves between two
exposures by the homography H = K R_now^T R_then K^-1 (see vision/egomotion.py).
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Ideal pinhole K. Replaced by a calibrated K once intrinsics exist (M2).

    OpenCV puts pixel centers on integers, so the image center is ((w-1)/2, (h-1)/2).
    """
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def scale_intrinsics(K: np.ndarray, scale: float) -> np.ndarray:
    """K for an image resized by `scale` (0.5 = half resolution)."""
    S = np.diag([scale, scale, 1.0])
    Ks = S @ K
    # pixel centers: x_s = s * (x + 0.5) - 0.5
    Ks[0, 2] = scale * (K[0, 2] + 0.5) - 0.5
    Ks[1, 2] = scale * (K[1, 2] + 0.5) - 0.5
    return Ks


def cam_to_base(yaw_deg: float, tilt_deg: float) -> np.ndarray:
    """Rotation whose columns are the camera's x (right), y (down), z (forward) axes in base frame."""
    p = math.radians(yaw_deg)
    t = math.radians(tilt_deg)
    cp, sp, ct, st = math.cos(p), math.sin(p), math.cos(t), math.sin(t)
    right = [sp, -cp, 0.0]
    down = [st * cp, st * sp, -ct]
    fwd = [ct * cp, ct * sp, st]
    return np.array([right, down, fwd]).T


def dir_from_azel(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def azel_from_dir(v: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))


def pixel_to_azel(u: float, v: float, K: np.ndarray, R_cb: np.ndarray) -> tuple[float, float]:
    ray = np.linalg.solve(K, np.array([u, v, 1.0]))
    return azel_from_dir(R_cb @ ray)


def azel_to_pixel(az: float, el: float, K: np.ndarray, R_cb: np.ndarray) -> tuple[float, float] | None:
    d = R_cb.T @ dir_from_azel(az, el)
    if d[2] <= 1e-6:
        return None
    p = K @ (d / d[2])
    return float(p[0]), float(p[1])


def rotation_homography(K: np.ndarray, R_then: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    """Maps pixels of the earlier image to where the same (distant) scene point is now."""
    H = K @ R_now.T @ R_then @ np.linalg.inv(K)
    return H / H[2, 2]


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a
