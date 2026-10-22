"""Stage A detector: training-free, works while the turret slews.

Two complementary tests, both computed in the current frame's view at decode
scale (half resolution):

* Background subtraction against the panoramic mosaic (vision/mosaic.py):
  gives the whole silhouette and an unbiased center. A foreground component
  only counts if it also shows motion (or sits on the tracked target), so a
  chair someone moved does not become a target.
* Ego-motion-compensated three-frame neighborhood differencing: previous
  frames are warped into the current view with the measured rotation, a pixel
  counts as changed only if it lies outside the [min, max] of the warped
  frame's k x k neighborhood (tolerates 1 px misalignment), and it must have
  changed against BOTH t-1 and t-2 (kills most of the ghost). Used alone
  where the mosaic has not seen the room yet, e.g. right after panning.

Both inputs are lightly blurred by the caller so a bilinear-warped (soft)
image is compared with a sharp one at equal sharpness.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MotionCfg

KNOWN_MIN = 3  # mosaic updates before a background pixel is trusted


@dataclass
class Blob:
    cx: float  # full-resolution pixels
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    area: int  # pixels at decode scale
    strength: float  # mean difference inside the blob
    source: str  # "bg" or "diff"


class MotionDetector:
    def __init__(self, cfg: MotionCfg, decode_scale: int) -> None:
        self.cfg = cfg
        self.scale = decode_scale
        k = cfg.neighborhood
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self._join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Bilinear warping mixes the replicated border into the first pixels past the
        # valid region, so trust a warped frame only k + 2 px inside its edge.
        self._margin = cv2.getStructuringElement(cv2.MORPH_RECT, (k + 4, k + 4))
        self.target_mask: np.ndarray | None = None  # accepted blobs, kept out of the mosaic
        self.moving_mask: np.ndarray | None = None  # anything that moved, also kept out

    def _nbhd_diff(self, cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
        hi = cv2.dilate(ref, self._kernel)
        lo = cv2.erode(ref, self._kernel)
        return cv2.max(cv2.subtract(cur, hi), cv2.subtract(lo, cur))

    def _changed(self, cur: np.ndarray, prev: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = cur.shape
        warped = cv2.warpPerspective(prev, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Where the previous frame actually covers this view: its 4 corners mapped by H.
        # (A rotation homography maps the image border to a convex quad.)
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        valid = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(valid, np.round(quad * 16).astype(np.int32), 255, shift=4)
        valid = cv2.erode(valid, self._margin, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return self._nbhd_diff(cur, warped), valid

    def _blob(self, labels, stats, cents, i, score_img, source) -> Blob | None:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < self.cfg.min_area_px or area > self.cfg.max_area_px:
            return None
        x, y, w, h = (int(v) for v in stats[i, :4])
        sel = labels[y : y + h, x : x + w] == i
        s = self.scale

        def full(p: float) -> float:
            return (p + 0.5) * s - 0.5

        return Blob(
            cx=full(float(cents[i, 0])),
            cy=full(float(cents[i, 1])),
            x0=x * s,
            y0=y * s,
            x1=(x + w) * s,
            y1=(y + h) * s,
            area=area,
            strength=float(score_img[y : y + h, x : x + w][sel].mean()),
            source=source,
        )

    def detect(
        self,
        cur: np.ndarray,
        prev1: np.ndarray,
        H1: np.ndarray,
        prev2: np.ndarray,
        H2: np.ndarray,
        bg: np.ndarray | None = None,
        known: np.ndarray | None = None,
        hint: tuple[float, float] | None = None,
    ) -> list[Blob]:
        thr = self.cfg.threshold
        d1, v1 = self._changed(cur, prev1, H1)
        d2, v2 = self._changed(cur, prev2, H2)
        valid = cv2.bitwise_and(v1, v2)
        moving = cv2.bitwise_and(cv2.threshold(cv2.max(d1, d2), thr, 255, cv2.THRESH_BINARY)[1], valid)
        both = cv2.min(d1, d2)
        and_mask = cv2.bitwise_and(cv2.threshold(both, thr, 255, cv2.THRESH_BINARY)[1], valid)

        blobs: list[Blob] = []
        target = np.zeros_like(cur)
        if bg is not None and known is not None:
            known_ok = (known >= KNOWN_MIN).astype(np.uint8) * 255
            dbg = self._nbhd_diff(cur, bg)
            fg = cv2.bitwise_and(cv2.threshold(dbg, thr, 255, cv2.THRESH_BINARY)[1], known_ok)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._join)
            n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
            if n > 1:
                moving_px = np.bincount(labels[moving > 0], minlength=n)
                areas = stats[:, cv2.CC_STAT_AREA]
                for i in np.nonzero((areas >= self.cfg.min_area_px) & (areas <= self.cfg.max_area_px))[0]:
                    if i == 0:
                        continue
                    x, y, w, h = (int(v) for v in stats[i, :4])
                    on_hint = hint is not None and (x - 4 <= hint[0] <= x + w + 4) and (y - 4 <= hint[1] <= y + h + 4)
                    if moving_px[i] == 0 and not on_hint:
                        continue
                    b = self._blob(labels, stats, cents, int(i), dbg, "bg")
                    if b is not None:
                        blobs.append(b)
                        target[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 255
            # Fallback where the room has not been seen yet.
            and_mask = cv2.bitwise_and(and_mask, cv2.bitwise_not(known_ok))
        and_mask = cv2.dilate(and_mask, self._join)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(and_mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        for i in np.nonzero((areas >= self.cfg.min_area_px) & (areas <= self.cfg.max_area_px))[0]:
            if i == 0:
                continue
            b = self._blob(labels, stats, cents, int(i), both, "diff")
            if b is not None:
                blobs.append(b)
                x, y, w, h = (int(v) for v in stats[i, :4])
                target[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i] = 255
        self.target_mask = target
        self.moving_mask = moving
        blobs.sort(key=lambda b: b.area * b.strength, reverse=True)
        return blobs[:12]
