"""Debug overlay. Draws onto a copy; safe headless (never touches a window)."""
from __future__ import annotations

import cv2
import numpy as np

from ..track.tracker import TrackState


class Overlay:
    def __init__(self, draw_trail: bool = True, trail_len: int = 32):
        self._trail: list[tuple[int, int]] = []
        self._draw_trail = draw_trail
        self._trail_len = trail_len

    def render(self, img: np.ndarray, track: TrackState | None,
               az_el: tuple[float, float] | None,
               fps: float, stage_ms: dict[str, float]) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        cv2.drawMarker(out, (w // 2, h // 2), (128, 128, 128),
                       cv2.MARKER_CROSS, 20, 1)  # boresight reference

        if track is not None:
            c = (0, 165, 255) if track.coasting else (0, 255, 0)
            p = (int(track.x), int(track.y))
            cv2.circle(out, p, 12, c, 2)
            # WHY the velocity vector is drawn scaled to ~0.25s of travel: it makes
            # "where will it be next quarter second" directly visible, which is the
            # quantity the lead predictor will consume.
            tip = (int(track.x + track.vx * 0.25), int(track.y + track.vy * 0.25))
            cv2.arrowedLine(out, p, tip, c, 2, tipLength=0.3)
            label = f"conf {track.confidence:.2f}" + (" COAST" if track.coasting else "")
            cv2.putText(out, label, (p[0] + 16, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            if az_el:
                cv2.putText(out, f"az {az_el[0]:+.2f}  el {az_el[1]:+.2f} deg",
                            (p[0] + 16, p[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            if self._draw_trail:
                self._trail.append(p)
                self._trail = self._trail[-self._trail_len:]
                for a, b in zip(self._trail, self._trail[1:], strict=False):
                    cv2.line(out, a, b, (200, 200, 0), 1)
        elif self._draw_trail:
            self._trail.clear()

        y = 22
        cv2.putText(out, f"{fps:5.1f} fps", (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        for k, v in stage_ms.items():
            y += 18
            cv2.putText(out, f"{k}: {v:5.2f} ms", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out
