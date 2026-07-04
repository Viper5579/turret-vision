"""Simulated turret: full closed loop with nothing plugged in (D4).

Behaves like the future ESP32 firmware so the pipeline (and later, ego-motion
compensation) can be developed and regression-tested on the bench:
- yaw: trapezoidal motion profile (AccelStepper-like velocity+accel limits)
- pitch: rate-limited servo
- clamps setpoints to travel limits (never trusts the wire)
- stops while the e-stop flag is set
- enters safe-hold when no AIM frame arrives within the timeout (link-dead is
  distinguishable from no-target because the Jetson heartbeats target_valid=0)

WHY it round-trips real bytes: every send_aim is packed to the wire format and
re-parsed, every telemetry is packed and re-parsed, so using MockLink at all
continuously exercises protocol.py exactly as the serial line would. The
counters (aim_packets, parser errors) let tests assert byte-level integrity.

Time is injectable (time_fn) so simulations can run faster than real time
(tools/sim_target.py) and tests are deterministic.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from ..util.timing import now
from . import protocol
from .base import AimOutput, Telemetry, TurretLink


def _step_axis(pos: float, vel: float, target: float, dt: float,
               vmax: float, amax: float, ff_dps: float = 0.0) -> tuple[float, float]:
    """Advance one axis with velocity+acceleration limits toward target.
    Classic trapezoid: command the velocity that could still stop in time
    (sqrt(2*a*d)), capped at vmax, approached at amax. The rate feedforward
    from the AIM packet is added on top — without it, chasing a target moving
    at constant v carries a permanent v^2/(2a) lag (the position term only
    produces velocity when there's error to burn)."""
    err = target - pos
    v_des = math.copysign(min(vmax, math.sqrt(2.0 * amax * abs(err))), err) + ff_dps
    v_des = max(-vmax, min(vmax, v_des))
    dv = max(-amax * dt, min(amax * dt, v_des - vel))
    vel += dv
    pos += vel * dt
    return pos, vel


class MockLink(TurretLink):
    def __init__(self, yaw_limits: tuple[float, float] = (-170.0, 170.0),
                 pitch_limits: tuple[float, float] = (-10.0, 60.0),
                 yaw_vmax_dps: float = 240.0, yaw_amax_dps2: float = 900.0,
                 pitch_rate_dps: float = 180.0,
                 aim_timeout_s: float = 0.5,
                 time_fn: Callable[[], float] = now, **_unused):
        self._yaw_lim, self._pitch_lim = yaw_limits, pitch_limits
        self._vmax, self._amax = yaw_vmax_dps, yaw_amax_dps2
        self._pitch_rate = pitch_rate_dps
        self._timeout = aim_timeout_s
        self._time = time_fn
        self._t = time_fn()
        self._t0 = self._t
        self._yaw = self._yaw_v = 0.0
        self._pitch = 0.0
        self._yaw_target = self._pitch_target = 0.0
        self._yaw_ff = 0.0
        self._estop = False
        self._last_aim_t: float | None = None
        self._parser = protocol.Parser()       # RX side of the "firmware"
        self._telem_parser = protocol.Parser()  # RX side of the "Jetson"
        self.aim_packets = 0
        self.safe_hold = False

    # -- firmware-side behavior ---------------------------------------------
    def send_aim(self, aim: AimOutput) -> None:
        t = self._time()
        self._advance(t)
        wire = protocol.pack_aim(protocol.AimPacket(
            t_ms=int((aim.t) * 1000), target_valid=aim.valid,
            estop=aim.estop,
            mode=1 if aim.valid else 0,
            yaw_deg=aim.az_deg, pitch_deg=aim.el_deg,
            yaw_rate_dps=aim.az_rate_dps, pitch_rate_dps=aim.el_rate_dps,
            confidence=aim.confidence,
            range_mm=int((aim.range_m or 0) * 1000)))
        for _, payload in self._parser.feed(wire):   # real bytes, real parse
            pkt = protocol.unpack_aim(payload)
            self.aim_packets += 1
            self._last_aim_t = t
            self.safe_hold = False
            self._estop = pkt.estop
            if pkt.estop:
                continue
            if pkt.target_valid:
                # firmware clamps to travel limits: never trust the wire
                self._yaw_target = min(max(pkt.yaw_deg, self._yaw_lim[0]), self._yaw_lim[1])
                self._pitch_target = min(max(pkt.pitch_deg, self._pitch_lim[0]),
                                         self._pitch_lim[1])
                self._yaw_ff = pkt.yaw_rate_dps
            else:
                # target_valid=0 heartbeat: hold last setpoints (SPEC 5), no ff
                self._yaw_ff = 0.0

    def _advance(self, t: float) -> None:
        dt = t - self._t
        self._t = t
        if dt <= 0:
            return
        if self._last_aim_t is not None and t - self._last_aim_t > self._timeout:
            # link-dead safe-hold: stop where you are until frames resume
            self.safe_hold = True
        hold = self._estop or self.safe_hold
        yaw_target = self._yaw if hold else self._yaw_target
        pitch_target = self._pitch if hold else self._pitch_target
        if hold:
            self._yaw_v = 0.0
        else:
            self._yaw, self._yaw_v = _step_axis(self._yaw, self._yaw_v, yaw_target,
                                                dt, self._vmax, self._amax, self._yaw_ff)
            step = self._pitch_rate * dt
            self._pitch += min(max(pitch_target - self._pitch, -step), step)

    # -- Jetson-side API -----------------------------------------------------
    def poll_telemetry(self) -> Telemetry | None:
        t = self._time()
        self._advance(t)
        status = protocol.STATUS_HOMED
        eps = 0.05  # trapezoid approaches the clamp asymptotically; near = at
        if self._yaw <= self._yaw_lim[0] + eps or self._yaw >= self._yaw_lim[1] - eps:
            status |= protocol.STATUS_YAW_AT_LIMIT
        if self._pitch <= self._pitch_lim[0] + eps or self._pitch >= self._pitch_lim[1] - eps:
            status |= protocol.STATUS_PITCH_AT_LIMIT
        if self._estop:
            status |= protocol.STATUS_ESTOPPED
        wire = protocol.pack_telem(protocol.TelemPacket(
            t_ms=int((t - self._t0) * 1000), yaw_deg=self._yaw, pitch_deg=self._pitch,
            yaw_rate_dps=self._yaw_v, status=status))
        for _, payload in self._telem_parser.feed(wire):
            pkt = protocol.unpack_telem(payload)
            return Telemetry(t_esp_ms=pkt.t_ms, yaw_deg=pkt.yaw_deg,
                             pitch_deg=pkt.pitch_deg, yaw_rate_dps=pkt.yaw_rate_dps,
                             status=pkt.status, t_rx=t)
        return None

    @property
    def wire_errors(self) -> int:
        return (self._parser.crc_errors + self._parser.oversize_errors
                + self._telem_parser.crc_errors + self._telem_parser.oversize_errors)
