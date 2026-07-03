"""Pipeline assembly + main loop. Contains NO logic on purpose (SPEC 2):
every stage is built from config and wired here, so any stage can be swapped
or tested in isolation without dragging the rest along.

Run examples:
  python -m turretvision.main                                  # live camera
  python -m turretvision.main --source synthetic --headless --max-frames 300
  python -m turretvision.main --source replay --replay logs/run1.mp4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .calib.geometry import PixelAngleMapper
from .capture.base import FrameSource
from .capture.replay import ReplaySource
from .capture.synthetic import SyntheticSource
from .capture.v4l2 import V4L2Camera
from .detect.aruco import ArucoDetector
from .detect.base import Detector
from .detect.color_mask import ColorMaskDetector
from .detect.frame_diff import FrameDiffDetector
from .link.base import AimOutput, TurretLink
from .link.console_link import ConsoleLink
from .track.filters import AlphaBetaFilter
from .track.tracker import SingleTargetTracker
from .ui.overlay import Overlay
from .util.config import Config
from .util.timing import RollingRate, StageTimer

DETECTORS = {"frame_diff": FrameDiffDetector, "aruco": ArucoDetector, "color_mask": ColorMaskDetector}


def build_source(cfg: Config, args) -> FrameSource:
    backend = args.source or cfg.get("camera.backend")
    if backend == "v4l2":
        return V4L2Camera(cfg.get("camera.device"), cfg.get("camera.width"),
                          cfg.get("camera.height"), cfg.get("camera.fps"),
                          cfg.get("camera.fourcc", "MJPG"))
    if backend == "replay":
        path = args.replay or cfg.get("camera.replay_path")
        if not path:
            sys.exit("replay backend needs --replay <file> or camera.replay_path")
        return ReplaySource(path, realtime=cfg.get("camera.replay_realtime", True))
    if backend == "synthetic":
        return SyntheticSource(n_frames=args.max_frames, realtime=not args.headless)
    sys.exit(f"unknown camera backend: {backend}")


def build_detector(cfg: Config) -> Detector:
    mode = cfg.get("detection.mode")
    if mode not in DETECTORS:
        sys.exit(f"unknown detection mode: {mode}")
    return DETECTORS[mode](**cfg.section(f"detection.{mode}"))


def build_link(cfg: Config) -> TurretLink:
    backend = cfg.get("link.backend")
    if backend == "console":
        return ConsoleLink(console_rate_hz=cfg.get("link.console_rate_hz", 5.0))
    # mock | serial arrive in Phase 3; failing loudly beats silently printing.
    sys.exit(f"link backend '{backend}' not implemented until Phase 3 (use 'console')")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="turret vision pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", choices=["v4l2", "replay", "synthetic"], default=None,
                    help="override camera.backend")
    ap.add_argument("--replay", default=None, help="video file for --source replay")
    ap.add_argument("--headless", action="store_true", help="no window (SSH/tests)")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    src = build_source(cfg, args)
    src.start()
    w, h = src.resolution

    mapper = PixelAngleMapper(
        w, h,
        intrinsics_file=cfg.get("calibration.intrinsics_file", None),
        fallback_hfov_deg=cfg.get("calibration.fallback_hfov_deg", 70.0),
        boresight_yaw_deg=cfg.get("calibration.boresight_yaw_deg", 0.0),
        boresight_pitch_deg=cfg.get("calibration.boresight_pitch_deg", 0.0),
        undistort_points=cfg.get("calibration.undistort_points", True),
    )
    if not mapper.calibrated:
        print("[warn] no intrinsics file -> using fallback FOV focal estimate "
              "(~5% angle error; run tools/calibrate_camera.py in Phase 4)")

    detector = build_detector(cfg)
    tracker = SingleTargetTracker(
        AlphaBetaFilter(cfg.get("tracking.alpha"), cfg.get("tracking.beta")),
        gate_px=cfg.get("tracking.gate_px"),
        max_coast_frames=cfg.get("tracking.max_coast_frames"),
    )
    link = build_link(cfg)
    overlay = Overlay(cfg.get("ui.draw_trail", True), cfg.get("ui.trail_len", 32))
    min_conf = cfg.get("tracking.min_confidence_output")

    fps = RollingRate()
    timer = StageTimer()

    csv_writer = None
    csv_file = None
    if cfg.get("logging.csv_state_log", False):
        Path(cfg.get("logging.run_log_dir", "logs")).mkdir(exist_ok=True)
        csv_file = open(Path(cfg.get("logging.run_log_dir", "logs")) / "state.csv", "w", newline="")
        csv_writer = csv.writer(csv_file)
        # WHY a per-frame CSV: it turns "the tracker feels laggy" into a plot, and
        # it's the regression artifact for replay diffing.
        csv_writer.writerow(["t", "valid", "px_x", "px_y", "az", "el",
                             "az_rate", "el_rate", "conf", "coasting"])

    show = cfg.get("ui.show_window", True) and not args.headless
    frames = 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                if isinstance(src, V4L2Camera):
                    continue  # camera: no NEW frame yet, keep polling
                break         # replay/synthetic: exhausted
            timer.start()
            dets = detector.detect(frame)
            timer.mark("detect")
            track = tracker.step(dets, frame.t)
            timer.mark("track")

            if track is not None:
                az, el = mapper.pixel_to_angles(track.x, track.y)
                # Angular rates via the local scale. WHY not per-axis exact math:
                # at these FOVs the center-scale approximation errs <5% and keeps
                # the hot loop trig-free; revisit if edge-of-frame rates matter.
                az_rate = track.vx * mapper.deg_per_px
                el_rate = -track.vy * mapper.deg_per_px
                valid = track.confidence >= min_conf
                aim = AimOutput(frame.t, valid, az, el, az_rate, el_rate, track.confidence)
            else:
                az = el = az_rate = el_rate = 0.0
                aim = AimOutput(frame.t, False, 0.0, 0.0, 0.0, 0.0, 0.0)
            link.send_aim(aim)
            timer.mark("link")

            fps.tick()
            if csv_writer:
                if track is not None:
                    csv_writer.writerow([f"{frame.t:.4f}", int(aim.valid),
                                         f"{track.x:.1f}", f"{track.y:.1f}",
                                         f"{az:.3f}", f"{el:.3f}",
                                         f"{az_rate:.2f}", f"{el_rate:.2f}",
                                         f"{track.confidence:.3f}", int(track.coasting)])
                else:
                    csv_writer.writerow([f"{frame.t:.4f}", 0, "", "", "", "", "", "", "0.0", ""])

            if show:
                import cv2
                img = overlay.render(frame.img, track,
                                     (az, el) if track else None, fps.hz, timer.report())
                cv2.imshow("turret-vision", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        src.stop()
        link.close()
        if csv_file:
            csv_file.close()
        if show:
            import cv2
            cv2.destroyAllWindows()

    r = timer.report()
    print(f"[done] {frames} frames | {fps.hz:.1f} fps | "
          + " | ".join(f"{k} {v:.2f}ms" for k, v in r.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
