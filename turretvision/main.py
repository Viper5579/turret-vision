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
from .capture.gstreamer import GstCamera
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
                          cfg.get("camera.fourcc", "MJPG"),
                          ctrls=cfg.get("camera.v4l2_ctrls", None) or {})
    if backend == "gstreamer":
        return GstCamera(cfg.get("camera.device"), cfg.get("camera.width"),
                         cfg.get("camera.height"), cfg.get("camera.fps"),
                         ctrls=cfg.get("camera.v4l2_ctrls", None) or {},
                         pipeline=cfg.get("camera.gst_pipeline", None))
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


def build_ranger(cfg: Config, mapper: PixelAngleMapper):
    mode = cfg.get("ranging.mode", "fixed")
    if mode == "fixed":
        from .ranging.fixed import FixedRange
        return FixedRange(cfg.get("ranging.fixed_distance_m", 4.0))
    if mode == "known_size":
        from .ranging.known_size import KnownSizeRange
        return KnownSizeRange(focal_px=mapper.focal_px,
                              target_size_m=cfg.get("ranging.target_size_m", 0.30))
    if mode == "aruco_pose":
        from .ranging.aruco_pose import ArucoPoseRange
        if mapper.camera_matrix is None:
            sys.exit("ranging.mode aruco_pose needs calibrated intrinsics "
                     "(run tools/calibrate_camera.py first)")
        return ArucoPoseRange(mapper.camera_matrix, mapper.dist_coeffs,
                              marker_size_m=cfg.get("detection.aruco.marker_size_m", 0.10))
    sys.exit(f"unknown ranging mode: {mode} (fixed | known_size | aruco_pose)")


def build_link(cfg: Config) -> TurretLink:
    backend = cfg.get("link.backend")
    if backend == "console":
        return ConsoleLink(console_rate_hz=cfg.get("link.console_rate_hz", 5.0))
    if backend == "mock":
        from .link.mock_link import MockLink
        return MockLink(yaw_limits=(cfg.get("turret.yaw_min_deg", -170.0),
                                    cfg.get("turret.yaw_max_deg", 170.0)),
                        pitch_limits=(cfg.get("turret.pitch_min_deg", -10.0),
                                      cfg.get("turret.pitch_max_deg", 60.0)),
                        aim_timeout_s=cfg.get("link.telemetry_timeout_s", 0.5))
    if backend == "serial":
        from .link.serial_link import SerialLink
        return SerialLink(cfg.get("link.port"), cfg.get("link.baud", 115200),
                          aim_rate_hz=cfg.get("link.aim_rate_hz", 50))
    sys.exit(f"unknown link backend: {backend} (console | mock | serial)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="turret vision pipeline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--source", choices=["v4l2", "gstreamer", "replay", "synthetic"],
                    default=None, help="override camera.backend")
    ap.add_argument("--replay", default=None, help="video file for --source replay")
    ap.add_argument("--headless", action="store_true", help="no window (SSH/tests)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tune", action="store_true",
                    help="serve the browser tuning UI (works headless; see README)")
    ap.add_argument("--tune-port", type=int, default=None,
                    help="tuning UI port (default: ui.tune_port from config, or 8089)")
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
        boresight_file=cfg.get("calibration.boresight_file", None),
        undistort_points=cfg.get("calibration.undistort_points", True),
    )
    if not mapper.calibrated:
        print("[warn] no intrinsics file -> using fallback FOV focal estimate "
              "(~5% angle error; expected until the Phase 4 calibration tool lands)")

    detector = build_detector(cfg)
    ranger = build_ranger(cfg, mapper)
    lead = None
    if cfg.get("lead.enabled", False):
        from .lead.predictor import LeadPredictor
        lead = LeadPredictor(
            projectile_speed_mps=cfg.get("lead.projectile_speed_mps", 50.0),
            gravity_comp=cfg.get("lead.gravity_comp", True),
            max_lead_deg=cfg.get("lead.max_lead_deg", 15.0))
    tracker = SingleTargetTracker(
        AlphaBetaFilter(cfg.get("tracking.alpha"), cfg.get("tracking.beta")),
        gate_px=cfg.get("tracking.gate_px"),
        max_coast_frames=cfg.get("tracking.max_coast_frames"),
    )
    link = build_link(cfg)
    overlay = Overlay(cfg.get("ui.draw_trail", True), cfg.get("ui.trail_len", 32))
    yaw_lim = (cfg.get("turret.yaw_min_deg", -170.0), cfg.get("turret.yaw_max_deg", 170.0))
    pitch_lim = (cfg.get("turret.pitch_min_deg", -10.0), cfg.get("turret.pitch_max_deg", 60.0))
    # Holder (not a bare float) so the tuning UI can adjust it live; the loop
    # reads it fresh each frame either way.
    min_conf_ref = {"value": cfg.get("tracking.min_confidence_output")}

    tuner = None
    if args.tune:
        from .tune.params import add_camera_params, build_registry
        from .tune.server import TuningServer
        registry = build_registry(detector, tracker.estimator, tracker, min_conf_ref)
        if isinstance(src, V4L2Camera):
            from .capture import uvc_ctrl
            cam = uvc_ctrl.probe(src.device)
            if cam:
                n = add_camera_params(registry, cam)
                print(f"[tune] exposing {n} UVC camera controls via v4l2-ctl")
            else:
                print("[tune] no UVC controls (v4l2-ctl missing or device won't enumerate); "
                      "camera section disabled")
        tuner = TuningServer(registry, config_path=args.config,
                             port=args.tune_port or cfg.get("ui.tune_port", 8089),
                             source_desc=getattr(src, "mode_desc", args.source or
                                                 cfg.get("camera.backend")))
        tuner.start()

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
            if tuner:
                tuner.apply_pending()  # slider changes land here, pipeline-thread only
            timer.start()
            # Telemetry first: the quasi-static gate (D1) needs the turret's
            # actual rate BEFORE this frame's differencing, and the absolute
            # setpoint (D2) is turret pose + tracked error.
            telem = link.poll_telemetry()
            if telem is not None:
                detector.set_turret_rate(telem.yaw_rate_dps)
            dets = detector.detect(frame)
            timer.mark("detect")
            track = tracker.step(dets, frame.t)
            timer.mark("track")

            turret_yaw = telem.yaw_deg if telem else 0.0
            turret_pitch = telem.pitch_deg if telem else 0.0
            lead_sol = None
            rng_est = None
            if track is not None:
                az, el = mapper.pixel_to_angles(track.x, track.y)
                # Angular rates via the local scale. WHY not per-axis exact math:
                # at these FOVs the center-scale approximation errs <5% and keeps
                # the hot loop trig-free; revisit if edge-of-frame rates matter.
                az_rate = track.vx * mapper.deg_per_px
                el_rate = -track.vy * mapper.deg_per_px
                valid = track.confidence >= min_conf_ref["value"]
                rng_est = ranger.estimate(dets, (track.x, track.y))
                aim_az, aim_el = az, el
                if lead and rng_est is not None:
                    lead_sol = lead.solve(az, el, az_rate, el_rate, rng_est.dist_m)
                    if lead_sol is not None:
                        aim_az, aim_el = lead_sol.yaw_deg, lead_sol.pitch_deg
                # Absolute setpoint, clamped Jetson-side too (defense in depth:
                # the firmware clamps as well, D2) — never send an illegal command.
                yaw_set = min(max(turret_yaw + aim_az, yaw_lim[0]), yaw_lim[1])
                pitch_set = min(max(turret_pitch + aim_el, pitch_lim[0]), pitch_lim[1])
                aim = AimOutput(frame.t, valid, yaw_set, pitch_set,
                                az_rate, el_rate, track.confidence,
                                range_m=rng_est.dist_m if rng_est else None)
            else:
                az = el = az_rate = el_rate = 0.0
                aim = AimOutput(frame.t, False, turret_yaw, turret_pitch,
                                0.0, 0.0, 0.0)
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

            if show or tuner:
                lead_px = (mapper.angles_to_pixel(lead_sol.yaw_deg, lead_sol.pitch_deg)
                           if lead_sol else None)
                img = overlay.render(frame.img, track,
                                     (az, el) if track else None, fps.hz, timer.report(),
                                     detections=dets, lead_px=lead_px,
                                     range_est=rng_est)
                if tuner:
                    r = timer.report()
                    tuner.publish(img, {
                        "fps": fps.hz, "n_dets": len(dets),
                        "detect_ms": r.get("detect", 0.0), "track_ms": r.get("track", 0.0),
                        "tracking": track is not None,
                        "coasting": bool(track.coasting) if track else False,
                        "az": az, "el": el, "az_rate": az_rate, "el_rate": el_rate,
                        "conf": track.confidence if track else 0.0,
                    })
                if show:
                    import cv2
                    cv2.imshow("turret-vision", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        src.stop()
        link.close()
        if tuner:
            tuner.stop()
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
