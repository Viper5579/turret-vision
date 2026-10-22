"""Drive the pipeline against the simulator, live (GUI) or headless (tests, CI).

Headless runs score the result against simulator ground truth, which is how CI
checks that a change did not quietly break tracking.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

from ..config import Config
from ..geometry import wrap_deg
from ..sim.world import Pacer, SimWorld
from .bus import Bus
from .pipeline import Mode, Pipeline

log = logging.getLogger(__name__)


@dataclass
class SimScore:
    sim_seconds: float = 0.0
    frames: int = 0
    visible_frames: int = 0
    tracked_frames: int = 0
    bearing_err_deg: list[float] = field(default_factory=list)
    pointing_err_deg: list[float] = field(default_factory=list)
    throws: int = 0
    sync_residual_ms: float | None = None
    false_track_frames: int = 0

    def summary(self) -> dict:
        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs)
            return s[min(len(s) - 1, int(q * len(s)))]

        return {
            "sim_seconds": round(self.sim_seconds, 2),
            "frames": self.frames,
            "throws": self.throws,
            "visible_frames": self.visible_frames,
            "track_coverage": round(self.tracked_frames / self.visible_frames, 3) if self.visible_frames else None,
            "bearing_err_deg_p50": pct(self.bearing_err_deg, 0.5),
            "bearing_err_deg_p95": pct(self.bearing_err_deg, 0.95),
            "pointing_err_deg_p50": pct(self.pointing_err_deg, 0.5),
            "pointing_err_deg_p80": pct(self.pointing_err_deg, 0.8),
            "false_track_frames": self.false_track_frames,
            "sync_residual_ms": self.sync_residual_ms,
        }


class SimRunner:
    def __init__(self, cfg: Config, bus: Bus | None = None, mode: Mode = Mode.TRACK) -> None:
        self.cfg = cfg
        self.world = SimWorld(cfg)
        self.pipeline = Pipeline(cfg, self.world.transport, self.world.now, bus)
        self.pipeline.mode = mode
        self.score = SimScore()
        self._last_hits = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self) -> None:
        w, p = self.world, self.pipeline
        w.advance(0.001)
        p.tick()
        for f in w.frames_ready():
            before = p.frames_processed
            p.on_frame(f.cam_seq, f.jpeg, f.arrival)
            if p.frames_processed > before:
                self._score_frame(f.cam_seq)

    def _score_frame(self, cam_seq: int) -> None:
        truth = self.world.truth.get(cam_seq)
        p = self.pipeline
        sc = self.score
        sc.frames += 1
        tr = p.last_track
        if truth is None:
            return
        hits_before = self._last_hits
        self._last_hits = tr.hits if tr is not None else 0
        if truth.plane_azel is None:
            # A confirmed track that keeps getting NEW detections with no plane in the air
            # is a false track. Coasting briefly after the plane lands is not.
            if tr is not None and tr.confirmed and tr.hits > hits_before:
                sc.false_track_frames += 1
            return
        az_t, el_t = truth.plane_azel
        # "Visible" = inside the camera's view at that instant.
        cam_az = truth.yaw_deg
        half_h = self.cfg.camera.hfov_deg / 2 - 2
        half_v = math.degrees(
            math.atan(math.tan(math.radians(half_h)) * self.cfg.camera.height / self.cfg.camera.width)
        )
        in_view = abs(wrap_deg(az_t - cam_az)) < half_h and abs(el_t - self.cfg.camera.tilt_deg) < half_v
        if not in_view:
            return
        sc.visible_frames += 1
        if tr is not None and tr.confirmed:
            sc.tracked_frames += 1
            sc.bearing_err_deg.append(math.hypot(wrap_deg(tr.az - az_t), tr.el - el_t))
            sc.pointing_err_deg.append(abs(wrap_deg(cam_az - az_t)))

    def run_headless(self, seconds: float) -> SimScore:
        end = self.world.now() + seconds
        while self.world.now() < end:
            self.step()
        self._finish()
        return self.score

    def _finish(self) -> None:
        self.score.sim_seconds = self.world.now() - 100.0
        self.score.throws = self.world.throws
        clk = self.pipeline.link.clock
        self.score.sync_residual_ms = 1000 * clk.residual_s if clk.valid else None

    # ------------------------------------------------------------ live mode
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="sim", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        pacer = Pacer(self.world.now())
        n = 0
        try:
            while not self._stop.is_set():
                self.step()
                n += 1
                if n % 5 == 0 and self.cfg.sim.realtime:
                    pacer.wait(self.world.now())
                if n % 100 == 0:
                    self.pipeline.extra_telemetry["sim"] = {
                        "speed": round(pacer.speed, 2),
                        "throws": self.world.throws,
                        **{
                            k: v
                            for k, v in self.score.summary().items()
                            if k in ("track_coverage", "bearing_err_deg_p50")
                        },
                    }
        except Exception:
            log.exception("simulator loop crashed")
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
