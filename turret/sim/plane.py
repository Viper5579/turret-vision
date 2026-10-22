"""Paper airplane flight model for the simulator.

A point mass with lift and drag. Lift acts perpendicular to the velocity and is
tilted by the bank angle; drag opposes the velocity. With a fixed lift
coefficient this model naturally produces the phugoid: the plane trades
height for speed and back, so it bobs up and down instead of gliding in a
straight line, which is exactly the motion that breaks a constant-velocity
tracker. Bank angle wanders randomly (Ornstein-Uhlenbeck) to make it turn.

Trim numbers: about 5 m/s glide speed (measured for real paper gliders) and a
glide ratio of about 5, which is lower than wind-tunnel values because real
folded paper is draggy and imperfect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81
RHO = 1.2


@dataclass
class PlaneParams:
    mass_kg: float = 0.005
    area_m2: float = 0.02
    trim_speed_mps: float = 5.0
    glide_ratio: float = 5.0
    wingspan_m: float = 0.20
    length_m: float = 0.22
    bank_sigma_deg: float = 18.0  # stationary std of the wandering bank angle
    bank_tau_s: float = 0.6

    @property
    def cl(self) -> float:
        return 2 * self.mass_kg * G / (RHO * self.trim_speed_mps**2 * self.area_m2)

    @property
    def cd(self) -> float:
        return self.cl / self.glide_ratio


@dataclass
class PlaneState:
    pos: np.ndarray  # m, base frame
    vel: np.ndarray  # m/s
    bank_rad: float = 0.0
    alive: bool = True


@dataclass
class Room:
    """Box around the turret (origin), meters. Turret sits on a desk 0.8 m up."""

    x: tuple[float, float] = (-1.5, 6.5)
    y: tuple[float, float] = (-3.5, 3.5)
    z: tuple[float, float] = (-0.8, 1.9)

    def contains(self, p: np.ndarray) -> bool:
        return self.x[0] < p[0] < self.x[1] and self.y[0] < p[1] < self.y[1] and self.z[0] < p[2] < self.z[1]


@dataclass
class Plane:
    params: PlaneParams = field(default_factory=PlaneParams)
    room: Room = field(default_factory=Room)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    state: PlaneState | None = None

    def throw(self) -> PlaneState:
        """A throw from a person standing 2.5 to 4 m in front of the turret, roughly across its view."""
        r = self.rng
        x0 = r.uniform(2.2, 3.5)
        side = r.choice([-1.0, 1.0])
        y0 = side * r.uniform(1.0, 2.0)
        z0 = r.uniform(0.5, 0.8)  # hand height above the desk the turret sits on
        speed = r.uniform(4.5, 7.0)
        heading = math.atan2(-y0, r.uniform(-1.2, 0.4))  # toward the far side, a bit toward/away
        climb = math.radians(r.uniform(-4.0, 10.0))
        vel = speed * np.array(
            [math.cos(climb) * math.cos(heading), math.cos(climb) * math.sin(heading), math.sin(climb)]
        )
        self.state = PlaneState(pos=np.array([x0, y0, z0]), vel=vel, bank_rad=0.0, alive=True)
        return self.state

    def _accel(self, vel: np.ndarray, bank: float) -> np.ndarray:
        p = self.params
        v = float(np.linalg.norm(vel))
        if v < 1e-3:
            return np.array([0.0, 0.0, -G])
        vhat = vel / v
        q = 0.5 * RHO * v * v * p.area_m2
        drag = -q * p.cd * vhat
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(vhat, up)
        n = np.linalg.norm(side)
        if n < 1e-6:  # flying straight up or down: no defined lift plane
            lift_dir = np.zeros(3)
        else:
            side /= n
            lift_up = np.cross(side, vhat)  # perpendicular to v, in the vertical plane
            lift_dir = math.cos(bank) * lift_up + math.sin(bank) * side
        lift = q * p.cl * lift_dir
        return (lift + drag) / p.mass_kg + np.array([0.0, 0.0, -G])

    def step(self, dt: float) -> None:
        s = self.state
        if s is None or not s.alive:
            return
        p = self.params
        # Ornstein-Uhlenbeck bank angle: mean-reverting random wander.
        a = math.exp(-dt / p.bank_tau_s)
        sig = math.radians(p.bank_sigma_deg) * math.sqrt(1 - a * a)
        s.bank_rad = a * s.bank_rad + sig * float(self.rng.standard_normal())
        # Midpoint (RK2) integration.
        a1 = self._accel(s.vel, s.bank_rad)
        v_mid = s.vel + 0.5 * dt * a1
        a2 = self._accel(v_mid, s.bank_rad)
        s.pos = s.pos + dt * v_mid
        s.vel = s.vel + dt * a2
        if not self.room.contains(s.pos):
            s.alive = False

    def attitude(self) -> np.ndarray:
        """Rotation body -> base. Body: x nose, y left wing, z up through the canopy."""
        s = self.state
        assert s is not None
        v = s.vel
        vhat = v / max(float(np.linalg.norm(v)), 1e-6)
        up = np.array([0.0, 0.0, 1.0])
        left = np.cross(up, vhat)
        n = np.linalg.norm(left)
        left = left / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
        top = np.cross(vhat, left)
        c, sn = math.cos(s.bank_rad), math.sin(s.bank_rad)
        # Positive bank tilts lift (and the canopy) toward the right wing, matching _accel.
        left_b = c * left + sn * top
        top_b = c * top - sn * left
        return np.array([vhat, left_b, top_b]).T

    def outline_body(self) -> list[np.ndarray]:
        """Polygons (body frame) drawn by the renderer: the wing dart and the keel."""
        p = self.params
        L, W = p.length_m, p.wingspan_m
        wing = np.array([[0.55 * L, 0, 0], [-0.45 * L, 0.5 * W, 0], [-0.30 * L, 0, 0], [-0.45 * L, -0.5 * W, 0]])
        keel = np.array([[0.55 * L, 0, 0], [-0.45 * L, 0, 0], [-0.40 * L, 0, -0.18 * W]])
        # Center the drawing on the tracked point so ground truth means "middle of the plane".
        c = np.vstack([wing, keel]).mean(axis=0)
        return [wing - c, keel - c]
