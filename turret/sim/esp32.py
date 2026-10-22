"""Simulated ESP32: the firmware's behavior model, speaking the real wire protocol.

It decodes real frames from the Jetson and answers with real frames, so the
simulator exercises the exact protocol, clock sync, and frame matching code
the hardware uses. Keep it in step with firmware/src/main.cpp: when firmware
behavior changes, this model changes in the same commit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..link import messages as m
from ..link.framing import FrameDecoder, encode_frame
from ..link.transport import Transport

FW_VERSION = 1
FEAT_ENCODER, FEAT_IMU, FEAT_TRIGGER, FEAT_LASER = 1, 2, 4, 8

ST_LINKED, ST_CONFIGURED, ST_ESTOP, ST_ARMED, ST_YAW_LIMIT, ST_TIMEOUT = 1, 2, 4, 8, 16, 32


@dataclass
class TurretTruth:
    yaw_deg: float = 0.0
    yaw_rate_dps: float = 0.0
    pitch_deg: float = 0.0
    gyro_angle_deg: float = 0.0  # what an ideal integrated gyro would read


class SimESP32:
    def __init__(
        self,
        transport: Transport,
        now: Callable[[], float],
        clock_offset_s: float,
        drift_ppm: float,
        rng: np.random.Generator,
        on_trigger: Callable[[int, float], None] | None = None,
    ) -> None:
        self._t = transport
        self._now = now
        self._offset = clock_offset_s
        self._drift = drift_ppm * 1e-6
        self._rng = rng
        self._dec = FrameDecoder()
        self.on_trigger = on_trigger
        self.cfg: m.Config | None = None
        self.traj: m.Traj | None = None
        self.traj_rx = -1e9
        self.truth = TurretTruth()
        self.gyro_meas_angle = 0.0
        self.status = 0
        self.trigger_seq = 1000  # arbitrary boot value; the Jetson must not assume alignment
        self._next_trigger: float | None = None
        self._next_telem = 0.0
        self._last_t: float | None = None
        self._enc_bias = float(rng.uniform(-0.05, 0.05))
        self._yaw_ref = 0.0

    # ---------------------------------------------------------------- clock
    def esp_us(self, t: float | None = None) -> int:
        t = self._now() if t is None else t
        return int((t * (1.0 + self._drift) + self._offset) * 1e6)

    def _send(self, msg: m.Message) -> None:
        self._t.write(encode_frame(msg.ID, msg.pack()))

    # ---------------------------------------------------------------- rx
    def _handle(self, msg_id: int, payload: bytes) -> None:
        cls = m.BY_ID.get(msg_id)
        if cls is None or len(payload) != cls.SIZE:
            return
        msg = cls.unpack(payload)
        now = self._now()
        if isinstance(msg, m.Hello):
            self.status |= ST_LINKED
            self._send(
                m.HelloAck(
                    proto_hash=m.PROTO_HASH,
                    fw_version=FW_VERSION,
                    uptime_ms=self.esp_us() // 1000 & 0xFFFFFFFF,
                    features=FEAT_ENCODER | FEAT_IMU | FEAT_TRIGGER | FEAT_LASER,
                )
            )
        elif isinstance(msg, m.TimePing):
            t_rx = self.esp_us(now)
            self._send(m.TimePong(seq=msg.seq, t_rx_us=t_rx, t_tx_us=t_rx + 40))
        elif isinstance(msg, m.Config):
            self.cfg = msg
            self.status |= ST_CONFIGURED
            if msg.trigger_hz > 0 and self._next_trigger is None:
                self._next_trigger = now + 0.01
            if msg.trigger_hz <= 0:
                self._next_trigger = None
        elif isinstance(msg, m.Traj):
            self.traj = msg
            self.traj_rx = now

    # ---------------------------------------------------------------- 1 kHz loop
    def update(self) -> None:
        now = self._now()
        data = self._t.read()
        if data:
            for msg_id, payload in self._dec.feed(data):
                self._handle(msg_id, payload)
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        if dt > 0:
            self._control(now, dt)
        if self._next_trigger is not None and self.cfg is not None and now >= self._next_trigger:
            self._trigger(now)
            self._next_trigger += 1.0 / self.cfg.trigger_hz
        if now >= self._next_telem:
            self._telem(now)
            self._next_telem = now + 0.005

    def _control(self, now: float, dt: float) -> None:
        cfg = self.cfg
        tr = self.truth
        if cfg is None:
            return
        timed_out = self.traj is None or (now - self.traj_rx) * 1000.0 > cfg.timeout_ms
        estop = self.traj is not None and bool(self.traj.flags & 1)
        self.status = (
            (self.status & ~(ST_TIMEOUT | ST_ESTOP)) | (ST_TIMEOUT if timed_out else 0) | (ST_ESTOP if estop else 0)
        )
        if timed_out or estop:
            want_rate = 0.0
            pitch_ref = tr.pitch_deg
        else:
            tj = self.traj
            age = (self.esp_us(now) - tj.valid_at_us) * 1e-6
            ref = tj.yaw_deg + tj.yaw_rate_dps * age
            ref = min(max(ref, cfg.yaw_min_deg), cfg.yaw_max_deg)
            target = ref + tj.yaw_rate_dps * cfg.lead_s
            want_rate = tj.yaw_rate_dps + cfg.kp * (target - tr.yaw_deg)
            pitch_ref = min(max(tj.pitch_deg + tj.pitch_rate_dps * age, cfg.pitch_min_deg), cfg.pitch_max_deg)
            self._yaw_ref = ref
        want_rate = max(-cfg.max_rate_dps, min(cfg.max_rate_dps, want_rate))
        dv = want_rate - tr.yaw_rate_dps
        max_dv = cfg.max_accel_dps2 * dt
        tr.yaw_rate_dps += max(-max_dv, min(max_dv, dv))
        tr.yaw_deg += tr.yaw_rate_dps * dt
        if tr.yaw_deg > cfg.yaw_max_deg or tr.yaw_deg < cfg.yaw_min_deg:
            tr.yaw_deg = min(max(tr.yaw_deg, cfg.yaw_min_deg), cfg.yaw_max_deg)
            tr.yaw_rate_dps = 0.0
        # Hobby servo: slews at a fixed speed toward its command.
        step = 400.0 * dt
        tr.pitch_deg += max(-step, min(step, pitch_ref - tr.pitch_deg))
        tr.gyro_angle_deg += tr.yaw_rate_dps * dt
        gyro_noise = 0.05 * float(self._rng.standard_normal())  # dps, ICM-42688 class
        self.gyro_meas_angle += (tr.yaw_rate_dps + gyro_noise) * dt

    def _encoder(self) -> float:
        lsb = 360.0 / 4096.0
        raw = self.truth.yaw_deg + self._enc_bias + 0.02 * float(self._rng.standard_normal())
        return round(raw / lsb) * lsb

    def _trigger(self, now: float) -> None:
        self.trigger_seq += 1
        self._send(
            m.FrameSync(
                seq=self.trigger_seq,
                t_us=self.esp_us(now),
                yaw_deg=self._encoder(),
                yaw_rate_dps=self.truth.yaw_rate_dps,
                gyro_yaw_deg=self.gyro_meas_angle,
                pitch_deg=self.truth.pitch_deg,
            )
        )
        if self.on_trigger:
            self.on_trigger(self.trigger_seq, now)

    def _telem(self, now: float) -> None:
        tr = self.truth
        self._send(
            m.Telem(
                t_us=self.esp_us(now),
                yaw_enc_deg=self._encoder(),
                yaw_step_deg=tr.yaw_deg,
                yaw_rate_dps=tr.yaw_rate_dps,
                yaw_ref_deg=self._yaw_ref,
                pitch_cmd_deg=tr.pitch_deg,
                gyro_z_dps=tr.yaw_rate_dps,
                status=self.status,
                faults=0,
            )
        )
