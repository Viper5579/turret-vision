"""M0 placeholder tracker: constant-velocity Kalman filter on bearing (az, el).

Deliberately simple so the whole loop runs end to end now. M3 replaces it
with the 3D IMM filter (glide / straight / maneuver models) from the plan;
this class keeps the same inputs and outputs so the swap is local.

State per axis: [angle deg, rate deg/s]. Track lifecycle: tentative until
`confirm_hits` detections land inside the gate within `confirm_window`
frames, then confirmed; deleted after `coast_s` without a detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import TrackerCfg
from ..geometry import wrap_deg


@dataclass
class TrackState:
    t: float
    az: float
    el: float
    az_rate: float
    el_rate: float
    sigma_deg: float
    confirmed: bool
    age_s: float
    hits: int

    def predict(self, t: float) -> tuple[float, float]:
        dt = t - self.t
        return wrap_deg(self.az + self.az_rate * dt), self.el + self.el_rate * dt


class BearingTracker:
    def __init__(self, cfg: TrackerCfg) -> None:
        self.cfg = cfg
        self._x: np.ndarray | None = None  # [az, az_rate, el, el_rate]
        self._P = np.eye(4)
        self._t = 0.0
        self._born = 0.0
        self._last_hit = 0.0
        self._recent: deque[bool] = deque(maxlen=cfg.confirm_window)
        self.confirmed = False
        self.hits = 0

    def reset(self) -> None:
        self._x = None
        self.confirmed = False
        self._recent.clear()
        self.hits = 0

    # -------------------------------------------------------------- KF pieces
    def _F_Q(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        F1 = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.cfg.accel_sigma_dps2**2
        Q1 = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        F = np.zeros((4, 4))
        Q = np.zeros((4, 4))
        F[:2, :2] = F[2:, 2:] = F1
        Q[:2, :2] = Q[2:, 2:] = Q1
        return F, Q

    _Hm = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])

    def _predict(self, t: float) -> None:
        assert self._x is not None
        dt = max(t - self._t, 0.0)
        F, Q = self._F_Q(dt)
        self._x = F @ self._x
        self._x[0] = wrap_deg(self._x[0])
        self._P = F @ self._P @ F.T + Q
        self._t = t

    def _innov(self, z: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        assert self._x is not None
        y = np.array([wrap_deg(z[0] - self._x[0]), z[1] - self._x[2]])
        R = np.eye(2) * self.cfg.meas_sigma_deg**2
        S = self._Hm @ self._P @ self._Hm.T + R
        return y, S

    # -------------------------------------------------------------- public
    def update(self, t: float, bearings: list[tuple[float, float]]) -> TrackState | None:
        if self._x is None:
            if bearings:
                az, el = bearings[0]
                self._x = np.array([az, 0.0, el, 0.0])
                self._P = np.diag([self.cfg.meas_sigma_deg**2, 200.0**2, self.cfg.meas_sigma_deg**2, 200.0**2])
                self._t = self._born = self._last_hit = t
                self._recent.clear()
                self._recent.append(True)
                self.hits = 1
                self.confirmed = False
            return self.state()

        self._predict(t)
        best = None
        best_d2 = self.cfg.gate_sigma**2
        for z in bearings:
            y, S = self._innov(z)
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 < best_d2:
                best, best_d2 = z, d2
        if best is not None:
            y, S = self._innov(best)
            K = self._P @ self._Hm.T @ np.linalg.inv(S)
            self._x = self._x + K @ y
            self._x[0] = wrap_deg(self._x[0])
            self._P = (np.eye(4) - K @ self._Hm) @ self._P
            self._last_hit = t
            self.hits += 1
        self._recent.append(best is not None)
        if not self.confirmed and sum(self._recent) >= self.cfg.confirm_hits:
            self.confirmed = True
        lost = (t - self._last_hit) > (self.cfg.coast_s if self.confirmed else 0.05)
        if lost:
            self.reset()
            # The detection that failed to associate may be the real target; seed from it.
            if bearings and best is None:
                return self.update(t, bearings[:1])
            return None
        return self.state()

    def state(self) -> TrackState | None:
        if self._x is None:
            return None
        sig = float(np.sqrt(max(self._P[0, 0], self._P[2, 2])))
        return TrackState(
            t=self._t,
            az=float(self._x[0]),
            el=float(self._x[2]),
            az_rate=float(self._x[1]),
            el_rate=float(self._x[3]),
            sigma_deg=sig,
            confirmed=self.confirmed,
            age_s=self._t - self._born,
            hits=self.hits,
        )
