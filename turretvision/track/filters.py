"""State estimators. v1: alpha-beta filter (design decision D6).

WHY alpha-beta before Kalman: two gains you can reason about by hand.
- alpha: how much you trust the new measurement's POSITION (1.0 = raw
  measurements, jittery; 0.1 = smooth but laggy).
- beta:  same for VELOCITY, derived from the position residual.
A Kalman filter needs honest process/measurement noise covariances, which you
can't write down until you've MEASURED your detection noise -- which needs a
working pipeline first. Same interface, so Kalman drops in later via config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class KinematicState:
    x: float
    y: float
    vx: float
    vy: float
    t: float


class StateEstimator(ABC):
    @abstractmethod
    def update(self, x: float, y: float, t: float) -> KinematicState: ...

    @abstractmethod
    def predict(self, t: float) -> KinematicState | None:
        """Extrapolate without a measurement (used while coasting)."""

    @abstractmethod
    def reset(self) -> None: ...


class AlphaBetaFilter(StateEstimator):
    def __init__(self, alpha: float = 0.5, beta: float = 0.2):
        self._a, self._b = alpha, beta
        self._s: KinematicState | None = None

    def update(self, x: float, y: float, t: float) -> KinematicState:
        if self._s is None:
            self._s = KinematicState(x, y, 0.0, 0.0, t)
            return self._s
        dt = t - self._s.t
        if dt <= 0:
            return self._s
        # Predict forward, then correct by a fraction of the residual.
        px = self._s.x + self._s.vx * dt
        py = self._s.y + self._s.vy * dt
        rx, ry = x - px, y - py
        self._s = KinematicState(
            x=px + self._a * rx,
            y=py + self._a * ry,
            # WHY beta/dt: the residual is a position error; dividing by dt converts
            # it into the velocity correction that would have explained it.
            vx=self._s.vx + (self._b / dt) * rx,
            vy=self._s.vy + (self._b / dt) * ry,
            t=t,
        )
        return self._s

    def predict(self, t: float) -> KinematicState | None:
        if self._s is None:
            return None
        dt = t - self._s.t
        return KinematicState(self._s.x + self._s.vx * dt, self._s.y + self._s.vy * dt,
                              self._s.vx, self._s.vy, t)

    def reset(self) -> None:
        self._s = None
