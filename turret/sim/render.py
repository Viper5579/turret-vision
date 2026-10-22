"""Synthetic camera images for the simulator.

Background: the room box is ray-cast once into an equirectangular panorama
(azimuth x elevation). Because the camera only rotates about the vertical
axis, a frame at any yaw is a single cv2.remap of that panorama with the
x-coordinates shifted by the yaw. That is exact for a rotating camera and
costs a few milliseconds.

Plane: its two polygons are projected through the real pinhole model at three
instants across the exposure and averaged, which gives honest motion blur.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import cam_to_base, intrinsics_from_fov
from .plane import Plane, Room

PANO_W = 4096
EL_MIN, EL_MAX = -70.0, 70.0


def _wall_texture(rng: np.random.Generator, u: np.ndarray, v: np.ndarray, base: tuple[int, int, int]) -> np.ndarray:
    """Procedural wall: base color, soft shading, a few posters/shelves, fine grain."""
    h = u.shape
    img = np.empty((*h, 3), np.float32)
    shade = 0.85 + 0.08 * np.sin(u * 1.3 + 0.7) * np.cos(v * 0.9) + 0.05 * np.sin(u * 4.1)
    for c in range(3):
        img[..., c] = base[c] * shade
    for _ in range(9):
        cu, cv_ = rng.uniform(-3, 3), rng.uniform(0.3, 2.2)
        w, hh = rng.uniform(0.25, 1.0), rng.uniform(0.2, 0.8)
        color = rng.integers(30, 230, 3)
        mask = (np.abs(u - cu) < w / 2) & (np.abs(v - cv_) < hh / 2)
        img[mask] = color
        edge = mask & ((np.abs(u - cu) > w / 2 - 0.02) | (np.abs(v - cv_) > hh / 2 - 0.02))
        img[edge] = 25
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    return img


def build_panorama(room: Room, seed: int = 0, desk_height: float = 0.8) -> np.ndarray:
    """Ray-cast the room from the turret (origin) into an equirectangular BGR panorama."""
    rng = np.random.default_rng(seed)
    pano_h = int(round(PANO_W * (EL_MAX - EL_MIN) / 360.0))
    az = np.linspace(-180.0, 180.0, PANO_W, endpoint=False) + 180.0 / PANO_W
    el = np.linspace(EL_MAX, EL_MIN, pano_h)
    AZ, EL = np.meshgrid(np.radians(az), np.radians(el))
    d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1)

    bounds = [room.x, room.y, room.z]
    t_hit = np.full(AZ.shape, np.inf)
    face = np.full(AZ.shape, -1)
    for axis in range(3):
        for side in range(2):
            plane_pos = bounds[axis][side]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = plane_pos / d[..., axis]
            ok = (t > 0) & (t < t_hit)
            t_hit = np.where(ok, t, t_hit)
            face = np.where(ok, axis * 2 + side, face)
    p = d * t_hit[..., None]
    out = np.zeros((*AZ.shape, 3), np.float32)
    palette = [(150, 170, 185), (175, 160, 140), (140, 165, 150), (185, 180, 165)]
    for f in range(6):
        sel = face == f
        if not sel.any():
            continue
        axis = f // 2
        if axis == 2:  # floor or ceiling
            if f == 4:  # floor: wood planks
                stripes = (np.floor(p[..., 0][sel] * 6.0) % 2) * 18
                grain = rng.normal(0, 8, stripes.shape)
                col = np.stack([60 + stripes + grain, 95 + stripes + grain, 135 + stripes + grain], -1)
            else:  # ceiling with light panels
                panel = (np.abs((p[..., 0][sel] % 1.2) - 0.6) < 0.3) & (np.abs((p[..., 1][sel] % 1.2) - 0.6) < 0.3)
                v = 205 + 40 * panel
                col = np.stack([v, v, v], -1).astype(np.float32)
            out[sel] = col
        else:
            u = p[..., 1 - axis][sel] if axis == 0 else p[..., 0][sel]
            v = p[..., 2][sel] + desk_height
            out[sel] = _wall_texture(rng, u, v, palette[f % 4])
            base = (v > 0.0) & (v < 0.12)  # baseboard
            tmp = out[sel]
            tmp[base] = 235
            out[sel] = tmp
    pano = np.clip(out, 0, 255).astype(np.uint8)
    # A real lens never delivers single-pixel-sharp detail; without this slight blur the
    # resampled room aliases (shimmering thin lines) in a way no real camera does.
    return cv2.GaussianBlur(pano, (0, 0), 0.7)


class Renderer:
    def __init__(self, width: int, height: int, hfov_deg: float, tilt_deg: float, room: Room, seed: int, noise: float):
        self.w, self.h = width, height
        self.K = intrinsics_from_fov(width, height, hfov_deg)
        self.tilt = tilt_deg
        self.pano = build_panorama(room, seed)
        ph = self.pano.shape[0]
        # Rays for every pixel at yaw 0 -> panorama coordinates.
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        rays = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(self.K).T
        rays = rays @ cam_to_base(0.0, tilt_deg).T
        az = np.degrees(np.arctan2(rays[..., 1], rays[..., 0]))
        el = np.degrees(np.arctan2(rays[..., 2], np.hypot(rays[..., 0], rays[..., 1])))
        self._map_x0 = ((az + 180.0) / 360.0 * PANO_W - 0.5).astype(np.float32)
        self._map_y = ((EL_MAX - el) / (EL_MAX - EL_MIN) * (ph - 1)).astype(np.float32)
        rng = np.random.default_rng(seed + 99)
        self._noise = []
        for _ in range(4):
            n = rng.normal(0, noise, (height, width, 3))
            self._noise.append((np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8)))
        self._k = 0

    def background(self, yaw_deg: float) -> np.ndarray:
        shift = (yaw_deg / 360.0 * PANO_W) % PANO_W
        return cv2.remap(
            self.pano,
            self._map_x0 + np.float32(shift),
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

    def render(
        self, yaw_samples: list[float], plane: Plane | None, plane_samples: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """yaw_samples / plane_samples: camera yaw and plane (pos, R_body) at instants across the exposure."""
        img = self.background(yaw_samples[len(yaw_samples) // 2])
        if plane is not None and plane_samples:
            self._draw_plane(img, yaw_samples, plane, plane_samples)
        pos, neg = self._noise[self._k]
        self._k = (self._k + 1) % len(self._noise)
        cv2.add(img, pos, dst=img)
        cv2.subtract(img, neg, dst=img)
        return img

    def _draw_plane(self, img, yaw_samples, plane: Plane, plane_samples) -> None:
        polys_all = []
        for yaw, (pos, R) in zip(yaw_samples, plane_samples, strict=True):
            Rcb = cam_to_base(yaw, self.tilt)
            polys = []
            for poly in plane.outline_body():
                world = pos + poly @ R.T
                cam = world @ Rcb  # = (Rcb^T world^T)^T
                if np.any(cam[:, 2] < 0.15):
                    polys = []
                    break
                px = cam @ self.K.T
                polys.append(px[:, :2] / px[:, 2:3])
            polys_all.append(polys)
        pts = [p for polys in polys_all for p in polys]
        if not pts:
            return
        allp = np.concatenate(pts)
        x0, y0 = np.floor(allp.min(0)).astype(int) - 2
        x1, y1 = np.ceil(allp.max(0)).astype(int) + 3
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > self.w * self.h // 4:
            return
        shape = (y1 - y0, x1 - x0)

        def raster(poly: np.ndarray) -> np.ndarray:
            mask = np.zeros(shape, np.uint8)
            pts16 = np.round((poly - [x0, y0]) * 16).astype(np.int32)
            cv2.fillPoly(mask, [pts16], 255, lineType=cv2.LINE_AA, shift=4)
            return mask.astype(np.float32) / 255.0

        cov_sum = np.zeros(shape, np.float32)
        col_sum = np.zeros(shape, np.float32)
        n = 0
        for polys in polys_all:
            if not polys:
                continue
            n += 1
            wing = raster(polys[0])
            keel = raster(polys[1]) * (1.0 - wing)  # keel hidden where the wing covers it
            cov_sum += wing + keel
            col_sum += wing * 238.0 + keel * 200.0
        if n == 0:
            return
        alpha = np.clip(cov_sum / n, 0.0, 1.0)[..., None]
        color = (col_sum / np.maximum(cov_sum, 1e-6))[..., None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(alpha * color + (1 - alpha) * roi, 0, 255).astype(np.uint8)


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
