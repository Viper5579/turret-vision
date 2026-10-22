"""Typed configuration. Defaults live here, in one place; overrides live in data/local.yaml.

Rule carried over from v1: if changing a value should not require reading code,
it is config. Every field says what it does and its unit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraCfg(_M):
    device: str = Field("/dev/video0", description="V4L2 device of the OV9782")
    width: int = Field(1280, description="px")
    height: int = Field(800, description="px")
    hfov_deg: float = Field(70.0, description="horizontal field of view until calibrated")
    tilt_deg: float = Field(10.0, description="camera pitched up from horizontal, fixed mount")
    trigger_hz: float = Field(90.0, description="ESP32 trigger rate; 90 is the OV9782 limit in trigger mode")
    exposure_us: int = Field(2000, description="manual exposure")
    exposure_delay_us: int = Field(0, description="trigger edge to exposure start, from the latency test")
    decode_scale: Literal[1, 2, 4] = Field(2, description="JPEG decoded at 1/scale for the motion detector")


class MotionCfg(_M):
    threshold: int = Field(16, description="gray levels; min change counted as motion")
    min_area_px: int = Field(4, description="at decode scale")
    max_area_px: int = Field(6000, description="at decode scale")
    neighborhood: int = Field(3, description="odd px; tolerance to warp misalignment")


class TrackerCfg(_M):
    meas_sigma_deg: float = Field(0.15, description="bearing measurement noise")
    accel_sigma_dps2: float = Field(400.0, description="target angular acceleration noise")
    gate_sigma: float = Field(4.0, description="Mahalanobis gate for association")
    confirm_hits: int = Field(3, description="hits within confirm_window frames to confirm a track")
    confirm_window: int = Field(5)
    coast_s: float = Field(0.25, description="keep predicting this long without detections")


class TurretCfg(_M):
    motor_steps_per_rev: int = Field(200)
    microsteps: int = Field(16, description="must match the A4988 MS1-3 jumpers")
    yaw_min_deg: float = -170.0
    yaw_max_deg: float = 170.0
    pitch_min_deg: float = -10.0
    pitch_max_deg: float = 60.0
    max_rate_dps: float = Field(300.0)
    max_accel_dps2: float = Field(3000.0)
    kp: float = Field(25.0, description="1/s, ESP32 position-error gain")
    lead_s: float = Field(0.01, description="ESP32 rate feedforward lookahead")
    latency_comp_s: float = Field(0.03, description="Jetson predicts targets this far ahead")
    home_yaw_deg: float = Field(0.0, description="where Track mode waits for the next throw")
    home_after_s: float = Field(1.5, description="Track mode returns home this long after losing the target")
    servo_us_min: int = 500
    servo_us_max: int = 2500

    @property
    def steps_per_deg(self) -> float:
        return self.motor_steps_per_rev * self.microsteps / 360.0


class LinkCfg(_M):
    port: str = Field("auto", description="serial device, or auto")
    baud: int = 921600
    timeout_ms: int = Field(200, description="ESP32 safe-stops after this long without Traj")
    traj_hz: float = Field(100.0, description="heartbeat rate when no frames arrive")


class SimCfg(_M):
    seed: int = 7
    realtime: bool = Field(True, description="pace sim time to the wall clock (GUI); false = as fast as possible")
    esp_clock_offset_s: float = Field(12.3456, description="ESP32 boot offset vs Jetson clock")
    esp_drift_ppm: float = Field(35.0)
    link_delay_ms: float = 1.0
    link_jitter_ms: float = 0.6
    usb_delay_ms: float = Field(9.0, description="trigger to frame arrival on the Jetson")
    usb_jitter_ms: float = 1.5
    frame_drop_prob: float = 0.002
    throw_interval_s: float = Field(1.0, description="pause between throws")
    pixel_noise: float = Field(2.0, description="gray-level sigma")
    jpeg_quality: int = 85


class ServerCfg(_M):
    host: str = "0.0.0.0"
    port: int = 8080
    video_fps: float = Field(30.0, description="max frames/s forwarded to each browser")


class Config(_M):
    camera: CameraCfg = CameraCfg()
    motion: MotionCfg = MotionCfg()
    tracker: TrackerCfg = TrackerCfg()
    turret: TurretCfg = TurretCfg()
    link: LinkCfg = LinkCfg()
    sim: SimCfg = SimCfg()
    server: ServerCfg = ServerCfg()
    data_dir: str = Field("data", description="sessions, local.yaml, engines")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Defaults, overlaid with data/local.yaml (or TURRET_CONFIG, or `path`)."""
    data_dir = Path(os.environ.get("TURRET_DATA", "data"))
    p = Path(path) if path else Path(os.environ.get("TURRET_CONFIG", data_dir / "local.yaml"))
    base = Config().model_dump()
    if p.exists():
        over = yaml.safe_load(p.read_text()) or {}
        base = _merge(base, over)
    base["data_dir"] = str(data_dir)
    return Config.model_validate(base)
