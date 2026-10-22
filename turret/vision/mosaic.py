"""Online panoramic background model for a camera that only rotates about one axis.

Because the camera rides the yaw stage and never translates, every pixel it
will ever see of the (distant enough) room lies on a cylinder of (azimuth,
elevation). We keep that cylinder as an image and paint each new frame into
it. Then, for any yaw, the expected background is one remap away, and the
target is simply "where this frame differs from the room".

Why this beats frame differencing alone: differencing only sees where the
plane moved, so a slow plane shows only its edges and the overlap of its two
previous positions leaves a ghost behind it (measured: centroid about one
frame of motion behind truth). Subtracting a background gives the whole
silhouette, an unbiased center, and still sees a plane hovering in a stall.

Unseen regions (just panned into) have no background yet; the caller falls
back to three-frame differencing there.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..geometry import cam_to_base


class BackgroundMosaic:
    N_FRAC = 4  # fractional-shift variants of the inverse map (1/8 px worst-case error)

    def __init__(self, K: np.ndarray, width: int, height: int, tilt_deg: float, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.ppd = K[0, 0] * math.pi / 180.0  # mosaic pixels per degree = image scale at center
        self.W = int(round(360.0 * self.ppd))
        vfov = 2 * math.degrees(math.atan(height / 2 / K[1, 1]))
        self.el_max = tilt_deg + vfov / 2 + 3
        el_min = tilt_deg - vfov / 2 - 3
        self.H = int(math.ceil((self.el_max - el_min) * self.ppd))
        self.bg = np.zeros((self.H, self.W), np.float32)
        self.count = np.zeros((self.H, self.W), np.uint8)
        self.w, self.h = width, height

        # Forward map: camera pixel (yaw 0) -> mosaic coordinates.
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        rays = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(K).T @ cam_to_base(0.0, tilt_deg).T
        az = np.degrees(np.arctan2(rays[..., 1], rays[..., 0]))
        el = np.degrees(np.arctan2(rays[..., 2], np.hypot(rays[..., 0], rays[..., 1])))
        self._fx0 = ((az + 180.0) * self.ppd).astype(np.float32)
        self._fy = ((self.el_max - el) * self.ppd).astype(np.float32)

        # Inverse maps: mosaic window (centered on yaw 0) -> camera pixel, for N_FRAC sub-shifts.
        half = int(math.ceil(az.max() * self.ppd)) + 2
        self._c0 = int(round(180.0 * self.ppd)) - half  # window start column at yaw 0
        self._win = 2 * half
        Rcb = cam_to_base(0.0, tilt_deg)
        jj, ii = np.meshgrid(np.arange(self._win, dtype=np.float64), np.arange(self.H, dtype=np.float64))
        el_m = np.radians(self.el_max - ii / self.ppd)
        self._inv = []
        for k in range(self.N_FRAC):
            frac = k / self.N_FRAC
            az_m = np.radians((self._c0 + jj - frac) / self.ppd - 180.0)
            d = np.stack([np.cos(el_m) * np.cos(az_m), np.cos(el_m) * np.sin(az_m), np.sin(el_m)], -1)
            c = d @ Rcb  # camera coordinates
            z = np.where(c[..., 2] > 1e-6, c[..., 2], np.nan)
            u = K[0, 0] * c[..., 0] / z + K[0, 2]
            v = K[1, 1] * c[..., 1] / z + K[1, 2]
            bad = ~np.isfinite(u) | (u < 0) | (u > width - 1) | (v < 0) | (v > height - 1)
            u[bad] = -10
            v[bad] = -10
            self._inv.append((u.astype(np.float32), v.astype(np.float32)))
        self._inside = [((u >= 0).astype(np.uint8) * 255) for u, _ in self._inv]

    def _window(self, yaw_deg: float) -> tuple[int, int]:
        """(first mosaic column of the window, which fractional inverse map to use)."""
        shift = yaw_deg * self.ppd
        s_int = math.floor(shift)
        k = int(round((shift - s_int) * self.N_FRAC))
        if k == self.N_FRAC:
            s_int, k = s_int + 1, 0
        return (self._c0 + s_int) % self.W, k

    def predict(self, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Expected background (uint8) for a frame at `yaw_deg`, and per-pixel update counts (0 = unknown)."""
        mx = self._fx0 + np.float32(yaw_deg * self.ppd)
        bg = cv2.remap(self.bg, mx, self._fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        known = cv2.remap(self.count, mx, self._fy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        return cv2.convertScaleAbs(bg), known

    def update(self, frame: np.ndarray, yaw_deg: float, exclude: np.ndarray | None = None) -> None:
        """Paint `frame` into the mosaic, skipping `exclude` (nonzero = target, keep out of background)."""
        c0, k = self._window(yaw_deg)
        u, v = self._inv[k]
        sample = cv2.remap(frame, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = self._inside[k]
        if exclude is not None:
            ex = cv2.remap(exclude, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(ex))
        # The window may wrap past 360 deg: split it into at most two in-place column slices.
        first = min(self._win, self.W - c0)
        for (m0, m1), (w0, w1) in (((c0, c0 + first), (0, first)), ((0, self._win - first), (first, self._win))):
            if m1 <= m0:
                continue
            bg = self.bg[:, m0:m1]
            cnt = self.count[:, m0:m1]
            smp = sample[:, w0:w1]
            msk = np.ascontiguousarray(mask[:, w0:w1])
            fresh = (cnt == 0) & (msk > 0)
            cv2.accumulateWeighted(smp, bg, self.alpha, mask=msk)
            np.copyto(bg, smp, where=fresh, casting="unsafe")
            cv2.add(cnt, 1, dst=cnt, mask=msk)
