#!/usr/bin/env python3
"""Phase 3 exit criterion: drive the FULL pipeline against MockLink and score it.

Synthetic target -> frame_diff -> tracker -> pixel->angle -> absolute setpoint
-> AIM bytes -> (parse, clamp, trapezoidal dynamics) -> TELEM bytes -> pipeline.
Every packet crosses the real wire format, so this run is also a protocol soak
test: any framing/CRC bug shows up in the wire-error counter.

Runs faster than real time: MockLink gets a simulated clock (time_fn), stepped
by exactly 1/fps per frame, so scoring is deterministic.

Usage: python tools/sim_target.py [--frames 300] [--fps 60]
Exit code 0 = all criteria pass (CI-friendly).
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.calib.geometry import PixelAngleMapper  # noqa: E402
from turretvision.capture.synthetic import SyntheticSource  # noqa: E402
from turretvision.detect.frame_diff import FrameDiffDetector  # noqa: E402
from turretvision.link.base import AimOutput  # noqa: E402
from turretvision.link.mock_link import MockLink  # noqa: E402
from turretvision.track.filters import AlphaBetaFilter  # noqa: E402
from turretvision.track.tracker import SingleTargetTracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--fps", type=int, default=60)
args = ap.parse_args()

sim_t = 0.0
link = MockLink(time_fn=lambda: sim_t)
src = SyntheticSource(n_frames=args.frames, fps=args.fps, realtime=False)
src.start()
w, h = src.resolution
mapper = PixelAngleMapper(w, h, intrinsics_file=None, fallback_hfov_deg=70.0)
detector = FrameDiffDetector(threshold=25, min_area_px=40, max_area_px=40000)
tracker = SingleTargetTracker(AlphaBetaFilter(0.5, 0.2), gate_px=120, max_coast_frames=8)

# Camera-frame geometry note: the synthetic camera is WORLD-FIXED (frames do
# not rotate when the mock turret moves), so this sim models a fixed camera
# commanding a separate turret: the setpoint is the target's camera-frame
# angle directly, and the quasi-static gate must NOT be fed the turret rate
# (the gate models a co-mounted camera that pans with the turret — feeding it
# here suppresses detection for a camera that never actually moved). The
# pose+error setpoint math in main.py is for the real co-mounted rig.
track_frames = 0
px_errs: list[float] = []
turret_errs: list[float] = []
telem_count = 0
total = 0
catch_up = 0  # frames of turret-scoring grace after each target wrap

while True:
    frame = src.read()
    if frame is None:
        break
    total += 1
    sim_t = frame.t

    telem = link.poll_telemetry()
    if telem is not None:
        telem_count += 1

    dets = detector.detect(frame)
    track = tracker.step(dets, frame.t)

    if track is not None:
        az, el = mapper.pixel_to_angles(track.x, track.y)
        aim = AimOutput(frame.t, track.confidence >= 0.4, az, el,
                        track.vx * mapper.deg_per_px, -track.vy * mapper.deg_per_px,
                        track.confidence)
    else:
        aim = AimOutput(frame.t, False, 0.0, 0.0, 0.0, 0.0, 0.0)
    link.send_aim(aim)

    # ---- scoring against ground truth --------------------------------------
    _, gx, gy = src.truth[frame.idx]
    # target wraps horizontally each lap: skip the teleport frame for pixel
    # scoring and give the turret a catch-up window (it must physically slew
    # back across the whole FOV — that's dynamics, not tracking error)
    if frame.idx > 0 and abs(src.truth[frame.idx][1] - src.truth[frame.idx - 1][1]) > 50:
        # budget: ~8 coast + ~7 reacquire frames for the tracker, then a
        # full-FOV slew (64 deg at vmax 240/amax 900 ~= 0.4s = 24 frames)
        catch_up = 45
        continue
    if total < 20:      # acquisition transient
        continue
    if track is not None and not track.coasting:
        track_frames += 1
        px_errs.append(math.hypot(track.x - gx, track.y - gy))
    truth_az, _ = mapper.pixel_to_angles(gx, gy)
    if catch_up > 0:
        catch_up -= 1
    elif telem is not None:
        # Chasing a moving target = steady-state lag, not accuracy; the
        # threshold below is the trapezoidal dynamics' tracking lag budget.
        turret_errs.append(abs(telem.yaw_deg - truth_az))

mean_px = sum(px_errs) / len(px_errs) if px_errs else float("inf")
mean_turret = sum(turret_errs) / len(turret_errs) if turret_errs else float("inf")
coverage = track_frames / total

print(f"frames                {total}")
print(f"track coverage        {coverage:.0%}   (tracked, non-coasting)")
print(f"mean pixel error      {mean_px:.1f} px")
print(f"telemetry received    {telem_count}")
print(f"AIM packets accepted  {link.aim_packets}")
print(f"wire errors           {link.wire_errors}")
print(f"mean turret-vs-truth  {mean_turret:.2f} deg  (closed-loop lag incl.)")

criteria = [
    ("track coverage > 75%", coverage > 0.75),
    ("mean pixel error < 30 px", mean_px < 30.0),
    ("every frame produced an accepted AIM packet", link.aim_packets == total),
    ("zero wire errors", link.wire_errors == 0),
    ("telemetry flowed", telem_count == total),
    ("turret followed target (mean err < 3 deg)", mean_turret < 3.0),
]
ok = True
for name, passed in criteria:
    print(("PASS  " if passed else "FAIL  ") + name)
    ok &= passed
sys.exit(0 if ok else 1)
