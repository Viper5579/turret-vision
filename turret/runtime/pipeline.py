"""The Jetson's per-frame loop: match -> pose -> warp + detect -> track -> command -> publish.

The same Pipeline runs against the simulator and (from M1) real hardware; only
the frame source and the link transport differ. Nothing in here sleeps or
blocks: the owner calls `on_frame()` when a JPEG arrives and `tick()` often.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np

from ..config import Config
from ..geometry import (
    azel_to_pixel,
    cam_to_base,
    intrinsics_from_fov,
    pixel_to_azel,
    rotation_homography,
    scale_intrinsics,
    wrap_deg,
)
from ..link import messages as m
from ..link.link import Link, LinkCallbacks
from ..track.bearing import BearingTracker, TrackState
from ..vision.mosaic import BackgroundMosaic
from ..vision.motion import Blob, MotionDetector
from .bus import Bus, FramePacket
from .frames import FrameMatcher

log = logging.getLogger(__name__)

_DECODE_FLAGS = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}


class Mode(IntEnum):
    IDLE = 0
    MANUAL = 1
    TRACK = 2
    ENGAGE = 3  # gated until M5; requested ENGAGE behaves as TRACK


@dataclass
class FrameInfo:
    cam_seq: int
    t_mid: float  # Jetson clock, middle of the exposure
    yaw_abs: float  # degrees, encoder-based, for bearings
    yaw_rel: float  # degrees, gyro-based when available, for the warp
    gray: np.ndarray


class Pipeline:
    def __init__(self, cfg: Config, link_transport, now, bus: Bus | None = None) -> None:
        self.cfg = cfg
        self.now = now
        self.bus = bus or Bus()
        tc = cfg.turret
        cc = cfg.camera
        config_msg = m.Config(
            steps_per_deg=tc.steps_per_deg,
            yaw_min_deg=tc.yaw_min_deg,
            yaw_max_deg=tc.yaw_max_deg,
            pitch_min_deg=tc.pitch_min_deg,
            pitch_max_deg=tc.pitch_max_deg,
            max_rate_dps=tc.max_rate_dps,
            max_accel_dps2=tc.max_accel_dps2,
            kp=tc.kp,
            lead_s=tc.lead_s,
            trigger_hz=cc.trigger_hz,
            timeout_ms=cfg.link.timeout_ms,
            servo_us_min=tc.servo_us_min,
            servo_us_max=tc.servo_us_max,
        )
        self.link = Link(
            link_transport,
            now,
            LinkCallbacks(on_telem=self._on_telem, on_frame_sync=self._on_sync),
            config_msg,
        )
        self.matcher = FrameMatcher(lambda t_us: self.link.clock.esp_to_jetson(t_us))
        self.K = intrinsics_from_fov(cc.width, cc.height, cc.hfov_deg)
        self.Ks = scale_intrinsics(self.K, 1.0 / cc.decode_scale)
        self.detector = MotionDetector(cfg.motion, cc.decode_scale)
        self.mosaic = BackgroundMosaic(self.Ks, cc.width // cc.decode_scale, cc.height // cc.decode_scale, cc.tilt_deg)
        self._exclude_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.tracker = BearingTracker(cfg.tracker)
        self._hist: deque[FrameInfo] = deque(maxlen=3)
        self._waiting: deque[tuple[int, bytes, float]] = deque()
        self.mode = Mode.IDLE
        self.estop = False
        self._jog = (0.0, 0.0)
        self._hold_yaw: float | None = None
        self._hold_pitch = 0.0
        self.telem: m.Telem | None = None
        self._last_traj = -1.0
        self._last_lock = -1e9
        self._fps_times: deque[float] = deque(maxlen=90)
        self._lat: deque[float] = deque(maxlen=90)
        self._proc: deque[float] = deque(maxlen=90)
        self.last_track: TrackState | None = None
        self.last_blobs: list[Blob] = []
        self.frames_processed = 0
        self.extra_telemetry: dict = {}

    # ------------------------------------------------------------ link events
    def _on_telem(self, t: m.Telem, t_rx: float) -> None:
        self.telem = t

    def _on_sync(self, fs: m.FrameSync) -> None:
        self.matcher.add_sync(fs)

    # ------------------------------------------------------------ frames
    def on_frame(self, cam_seq: int, jpeg: bytes, arrival: float) -> None:
        self._waiting.append((cam_seq, jpeg, arrival))
        self._drain_frames()

    def _drain_frames(self) -> None:
        if not self.link.synced:
            # No clock yet: frames cannot be placed in time. Keep only the newest few.
            while len(self._waiting) > 3:
                self._waiting.popleft()
            return
        now = self.now()
        while self._waiting:
            cam_seq, jpeg, arrival = self._waiting[0]
            res = self.matcher.match(cam_seq, arrival, now - arrival)
            if res == "wait":
                return
            self._waiting.popleft()
            if res is None:
                continue
            assert isinstance(res, m.FrameSync)
            self._process(cam_seq, jpeg, res)

    def _pose(self, fs: m.FrameSync) -> tuple[float, float, float]:
        cc = self.cfg.camera
        t_trig = self.link.clock.esp_to_jetson(fs.t_us)
        dt_mid = (cc.exposure_delay_us + cc.exposure_us / 2) * 1e-6
        t_mid = t_trig + dt_mid
        yaw_abs = fs.yaw_deg + fs.yaw_rate_dps * dt_mid
        base_rel = fs.gyro_yaw_deg if not math.isnan(fs.gyro_yaw_deg) else fs.yaw_deg
        yaw_rel = base_rel + fs.yaw_rate_dps * dt_mid
        return t_mid, yaw_abs, yaw_rel

    def _process(self, cam_seq: int, jpeg: bytes, fs: m.FrameSync) -> None:
        t_start = time.perf_counter()
        cc = self.cfg.camera
        gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), _DECODE_FLAGS[cc.decode_scale])
        if gray is None:
            return
        # Light blur so the warped (bilinear-resampled, slightly soft) previous frames and
        # the sharp current frame are compared at the same sharpness.
        gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
        t_mid, yaw_abs, yaw_rel = self._pose(fs)
        info = FrameInfo(cam_seq, t_mid, yaw_abs, yaw_rel, gray)
        R_now = cam_to_base(yaw_rel, cc.tilt_deg)
        R_abs = cam_to_base(yaw_abs, cc.tilt_deg)
        blobs: list[Blob] = []
        # The background mosaic is indexed by the absolute (encoder) yaw so it never drifts
        # with the gyro; frame-to-frame warps use the smoother gyro yaw.
        bg, known = self.mosaic.predict(yaw_abs)
        exclude = None
        if len(self._hist) >= 2:
            p1, p2 = self._hist[-1], self._hist[-2]
            H1 = rotation_homography(self.Ks, cam_to_base(p1.yaw_rel, cc.tilt_deg), R_now)
            H2 = rotation_homography(self.Ks, cam_to_base(p2.yaw_rel, cc.tilt_deg), R_now)
            hint = None
            if self.last_track is not None:
                pa, pe = self.last_track.predict(t_mid)
                hp = azel_to_pixel(pa, pe, self.Ks, R_abs)
                hint = hp if hp is not None else None
            blobs = self.detector.detect(gray, p1.gray, H1, p2.gray, H2, bg, known, hint)
        # The room changes slowly, so the background only needs every other frame (halves its cost).
        # Keep the target, and anything still moving, out of it.
        if self.frames_processed % 2 == 0 or len(self._hist) < 2:
            if self.detector.target_mask is not None and len(self._hist) >= 2:
                exclude = cv2.dilate(
                    cv2.max(self.detector.target_mask, self.detector.moving_mask), self._exclude_kernel
                )
            self.mosaic.update(gray, yaw_abs, exclude)
        self._hist.append(info)

        bearings = [pixel_to_azel(b.cx, b.cy, self.K, R_abs) for b in blobs]
        if self.last_track is not None:
            # Offer the detection nearest the prediction first.
            pa, pe = self.last_track.predict(t_mid)
            order = sorted(
                range(len(bearings)), key=lambda i: abs(wrap_deg(bearings[i][0] - pa)) + abs(bearings[i][1] - pe)
            )
            bearings = [bearings[i] for i in order]
            blobs = [blobs[i] for i in order]
        track = self.tracker.update(t_mid, bearings)
        self.last_track = track
        self.last_blobs = blobs
        self.frames_processed += 1

        self._command(frame_time=t_mid)
        done = self.now()
        self._fps_times.append(done)
        self._lat.append(done - t_mid)
        self._proc.append(time.perf_counter() - t_start)
        self._publish(cam_seq, jpeg, R_abs, t_mid, track, blobs)

    # ------------------------------------------------------------ commands
    def handle_commands(self) -> None:
        for cmd in self.bus.drain_commands():
            kind = cmd.get("t")
            if kind == "mode":
                try:
                    self.mode = Mode[str(cmd.get("mode", "IDLE")).upper()]
                except KeyError:
                    continue
                self._hold_yaw = None
                if self.mode == Mode.ENGAGE:
                    log.warning("ENGAGE is gated until M5; tracking only")
            elif kind == "estop":
                self.estop = bool(cmd.get("on", True))
                if self.estop:
                    self.mode = Mode.IDLE
            elif kind == "jog":
                self._jog = (float(cmd.get("yaw", 0.0)), float(cmd.get("pitch", 0.0)))

    def _current_yaw(self) -> float:
        if self.telem is None:
            return 0.0
        enc = self.telem.yaw_enc_deg
        return enc if not math.isnan(enc) else self.telem.yaw_step_deg

    def _command(self, frame_time: float | None = None) -> None:
        if not self.link.synced:
            return
        now = self.now()
        tc = self.cfg.turret
        yaw, yaw_rate, pitch, pitch_rate = self._current_yaw(), 0.0, self._hold_pitch, 0.0
        track = self.last_track
        if self.mode in (Mode.TRACK, Mode.ENGAGE) and track is not None and track.confirmed:
            t_aim = now + tc.latency_comp_s
            yaw, pitch = track.predict(t_aim)
            yaw_rate, pitch_rate = track.az_rate, track.el_rate
            self._hold_yaw, self._hold_pitch = yaw, pitch
            self._last_lock = now
        elif self.mode in (Mode.TRACK, Mode.ENGAGE) and now - self._last_lock > tc.home_after_s:
            # Target gone: wait for the next throw facing the usual throwing area.
            self._hold_yaw, self._hold_pitch = tc.home_yaw_deg, 0.0
            yaw, pitch = self._hold_yaw, self._hold_pitch
        elif self.mode == Mode.MANUAL:
            if self._hold_yaw is None:
                self._hold_yaw = self._current_yaw()
            dt = 0.0 if self._last_traj < 0 else min(now - self._last_traj, 0.05)
            self._hold_yaw += self._jog[0] * dt
            self._hold_pitch += self._jog[1] * dt
            yaw, yaw_rate = self._hold_yaw, self._jog[0]
            pitch, pitch_rate = self._hold_pitch, self._jog[1]
        else:
            if self._hold_yaw is None:
                self._hold_yaw = self._current_yaw()
            yaw = self._hold_yaw
            pitch = self._hold_pitch
        self._hold_yaw = min(max(self._hold_yaw if self._hold_yaw is not None else yaw, tc.yaw_min_deg), tc.yaw_max_deg)
        self._hold_pitch = min(max(self._hold_pitch, tc.pitch_min_deg), tc.pitch_max_deg)
        yaw = min(max(yaw, tc.yaw_min_deg), tc.yaw_max_deg)
        pitch = min(max(pitch, tc.pitch_min_deg), tc.pitch_max_deg)
        self.link.send_traj(
            m.Traj(
                valid_at_us=self.link.clock.jetson_to_esp_us(now),
                yaw_deg=yaw,
                yaw_rate_dps=yaw_rate,
                pitch_deg=pitch,
                pitch_rate_dps=pitch_rate,
                mode=int(min(self.mode, Mode.TRACK)),
                flags=1 if self.estop else 0,
            )
        )
        self._last_traj = now
        self._cmd = (yaw, pitch)

    # ------------------------------------------------------------ periodic
    def tick(self) -> None:
        self.link.poll()
        self.handle_commands()
        if self._waiting:
            self._drain_frames()
        now = self.now()
        if now - self._last_traj > 1.0 / self.cfg.link.traj_hz:
            self._command()

    # ------------------------------------------------------------ outputs
    def stats(self) -> dict:
        ft = self._fps_times
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0
        lat = sorted(self._lat)
        proc = sorted(self._proc)
        return {
            "fps": fps,
            "latency_ms_p50": 1000 * lat[len(lat) // 2] if lat else None,
            "latency_ms_p95": 1000 * lat[int(len(lat) * 0.95)] if lat else None,
            "process_ms_p50": 1000 * proc[len(proc) // 2] if proc else None,
            "matched": self.matcher.stats.matched,
            "unmatched": self.matcher.stats.unmatched,
            "usb_delay_ms": 1000 * self.matcher.stats.delay_s,
        }

    def _publish(self, cam_seq, jpeg, R_abs, t_mid, track: TrackState | None, blobs: list[Blob]) -> None:
        ov: dict = {
            "w": self.cfg.camera.width,
            "h": self.cfg.camera.height,
            "blobs": [[round(b.x0), round(b.y0), round(b.x1), round(b.y1)] for b in blobs[:8]],
        }
        if track is not None:
            px = azel_to_pixel(track.az, track.el, self.K, R_abs)
            if px is not None:
                ov["track"] = {
                    "x": round(px[0], 1),
                    "y": round(px[1], 1),
                    "r": max(6.0, track.sigma_deg * self.K[0, 0] / 57.2958 * 3),
                    "confirmed": track.confirmed,
                }
            pa, pe = track.predict(t_mid + self.cfg.turret.latency_comp_s)
            pp = azel_to_pixel(pa, pe, self.K, R_abs)
            if pp is not None:
                ov["aim"] = {"x": round(pp[0], 1), "y": round(pp[1], 1)}
        self.bus.publish_frame(FramePacket(cam_seq, jpeg, ov))
        self.bus.publish_telemetry(self.telemetry())

    def telemetry(self) -> dict:
        t = self.telem
        tr = self.last_track
        ls = self.link.stats
        out = {
            "time": self.now(),
            "mode": self.mode.name,
            "estop": self.estop,
            "link": {
                "connected": self.link.connected,
                "synced": self.link.synced,
                "hash_mismatch": ls.hash_mismatch,
                "fw_version": ls.fw_version,
                "sync_residual_ms": 1000 * self.link.clock.residual_s if self.link.clock.valid else None,
                "drift_ppm": self.link.clock.drift_ppm,
            },
            "turret": None
            if t is None
            else {
                "yaw_enc": None if math.isnan(t.yaw_enc_deg) else t.yaw_enc_deg,
                "yaw_step": t.yaw_step_deg,
                "yaw_ref": t.yaw_ref_deg,
                "yaw_rate": t.yaw_rate_dps,
                "pitch": t.pitch_cmd_deg,
                "status": t.status,
                "faults": t.faults,
            },
            "track": None
            if tr is None
            else {
                "az": tr.az,
                "el": tr.el,
                "az_rate": tr.az_rate,
                "el_rate": tr.el_rate,
                "sigma": tr.sigma_deg,
                "confirmed": tr.confirmed,
                "age": tr.age_s,
            },
            "detections": len(self.last_blobs),
            "pipeline": self.stats(),
        }
        out.update(self.extra_telemetry)
        return out
