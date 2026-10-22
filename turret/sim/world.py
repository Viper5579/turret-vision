"""The simulated world: virtual clock, plane, ESP32, camera, and the wires between them.

Everything the Jetson would see from real hardware comes out of here with the
same shape and timing quirks: frames arrive as JPEG bytes ~9 ms after the
trigger with jitter and occasional drops, camera sequence numbers do not line
up with ESP32 trigger numbers, and the ESP32 clock has its own offset and drift.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..geometry import azel_from_dir
from ..link.transport import PipeEnd, sim_wire
from .esp32 import SimESP32
from .plane import Plane, PlaneParams, Room
from .render import Renderer, encode_jpeg


@dataclass
class SimFrame:
    cam_seq: int
    jpeg: bytes
    arrival: float  # Jetson clock, s


@dataclass
class FrameTruth:
    t_mid: float
    yaw_deg: float
    plane_azel: tuple[float, float] | None
    plane_range_m: float | None


class VirtualClock:
    def __init__(self) -> None:
        self.t = 100.0  # Jetson monotonic clock never starts at zero either

    def __call__(self) -> float:
        return self.t


class SimWorld:
    TICK_S = 0.001  # ESP32 control loop period

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        sc = cfg.sim
        self.clock = VirtualClock()
        self.rng = np.random.default_rng(sc.seed)
        self.room = Room()
        self.plane = Plane(PlaneParams(), self.room, np.random.default_rng(sc.seed + 1))
        self.jetson_end, esp_end = sim_wire(self.clock, sc.link_delay_ms * 1e-3, sc.link_jitter_ms * 1e-3, sc.seed)
        self.esp = SimESP32(
            esp_end,
            self.clock,
            sc.esp_clock_offset_s,
            sc.esp_drift_ppm,
            np.random.default_rng(sc.seed + 2),
            on_trigger=self._on_trigger,
        )
        cc = cfg.camera
        self.renderer = Renderer(cc.width, cc.height, cc.hfov_deg, cc.tilt_deg, self.room, sc.seed, sc.pixel_noise)
        self.cam_seq = 0
        self._pending: deque[tuple[float, SimFrame]] = deque()
        self.truth: dict[int, FrameTruth] = {}
        self._next_throw = self.clock.t + 0.5
        self.throws = 0

    @property
    def transport(self) -> PipeEnd:
        return self.jetson_end

    def now(self) -> float:
        return self.clock.t

    # ---------------------------------------------------------------- stepping
    def advance(self, dt: float) -> None:
        """Advance virtual time by dt in 1 ms ticks."""
        steps = max(1, int(round(dt / self.TICK_S)))
        for _ in range(steps):
            self.clock.t += self.TICK_S
            self._tick()

    def _tick(self) -> None:
        t = self.clock.t
        st = self.plane.state
        if st is not None and st.alive:
            self.plane.step(self.TICK_S)
        elif st is not None:  # just landed or hit a wall
            self.plane.state = None
            self._next_throw = t + self.cfg.sim.throw_interval_s
        elif t >= self._next_throw:
            self.plane.throw()
            self.throws += 1
        self.esp.update()

    # ---------------------------------------------------------------- camera
    def _on_trigger(self, trigger_seq: int, t_trigger: float) -> None:
        cc, sc = self.cfg.camera, self.cfg.sim
        self.cam_seq += 1
        exp = cc.exposure_us * 1e-6
        t0 = t_trigger + cc.exposure_delay_us * 1e-6
        t_mid = t0 + exp / 2
        # Render "now" using the trajectory around the exposure. The turret is
        # simulated at 1 ms resolution, so extrapolate yaw with the current rate.
        yaw_now = self.esp.truth.yaw_deg
        rate = self.esp.truth.yaw_rate_dps
        yaws = [yaw_now + rate * (t0 + k * exp / 2 - self.clock.t) for k in range(3)]
        plane_samples = []
        st = self.plane.state
        truth_azel = None
        truth_range = None
        if st is not None and st.alive:
            R = self.plane.attitude()
            for k in range(3):
                dtk = t0 + k * exp / 2 - self.clock.t
                plane_samples.append((st.pos + st.vel * dtk, R))
            mid = st.pos + st.vel * (t_mid - self.clock.t)
            truth_azel = azel_from_dir(mid)
            truth_range = float(np.linalg.norm(mid))
        img = self.renderer.render(yaws, self.plane if plane_samples else None, plane_samples)
        self.truth[self.cam_seq] = FrameTruth(t_mid, yaws[1], truth_azel, truth_range)
        if len(self.truth) > 4000:
            self.truth.pop(next(iter(self.truth)))
        if self.rng.random() < sc.frame_drop_prob:
            return
        arrival = t0 + exp + sc.usb_delay_ms * 1e-3 + float(self.rng.random()) * sc.usb_jitter_ms * 1e-3
        frame = SimFrame(self.cam_seq, encode_jpeg(img, sc.jpeg_quality), arrival)
        self._pending.append((arrival, frame))

    def frames_ready(self) -> list[SimFrame]:
        out = []
        now = self.clock.t
        while self._pending and self._pending[0][0] <= now:
            out.append(self._pending.popleft()[1])
        return out


class Pacer:
    """Keeps virtual time from running ahead of the wall clock (for the live GUI)."""

    def __init__(self, sim_now: float) -> None:
        self._wall0 = time.monotonic()
        self._sim0 = sim_now
        self.speed = 1.0

    def wait(self, sim_now: float) -> None:
        ahead = (sim_now - self._sim0) - (time.monotonic() - self._wall0)
        if ahead > 0.002:
            time.sleep(ahead)
        elif ahead < -0.5:  # we are slower than real time; don't try to catch up forever
            self._wall0 = time.monotonic()
            self._sim0 = sim_now
        wall = time.monotonic() - self._wall0
        self.speed = (sim_now - self._sim0) / wall if wall > 0.5 else 1.0
