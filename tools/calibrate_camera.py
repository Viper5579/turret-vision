#!/usr/bin/env python3
"""ChArUco intrinsic calibration (SPEC 6.2) -> config/camera_intrinsics.yaml.

Workflow:
  1. python tools/calibrate_camera.py --print-board   # writes board.png; print it
     Print at 100% scale on A4/letter, tape it FLAT to something rigid, then
     measure one printed square with a ruler and pass --square-mm if it isn't
     exactly 35 mm (printers lie about scale; the solve inherits the error).
  2. python tools/calibrate_camera.py                 # live capture from the camera
     Move the board around: near/far, all four corners of the frame, tilted
     forward/back/left/right. Views auto-capture whenever the board is seen
     well and has MOVED since the last capture; ~20 views then solve.
     Works headless — progress prints to the terminal.
  3. Accept if mean reprojection error < 0.5 px (SPEC). Redo after any lens change.

Offline: --video run.mp4 or --images 'shots/*.png' instead of the live camera.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.calib import intrinsics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="/dev/video0")
ap.add_argument("--video", default=None, help="calibrate from a recorded video instead")
ap.add_argument("--images", default=None, help="glob of still images instead")
ap.add_argument("--views", type=int, default=20, help="views to collect before solving")
ap.add_argument("--square-mm", type=float, default=intrinsics.BOARD_SQUARE_M * 1000,
                help="MEASURED printed square size (printers rescale!)")
ap.add_argument("--out", default="config/camera_intrinsics.yaml")
ap.add_argument("--print-board", action="store_true",
                help="write board.png for printing and exit")
ap.add_argument("--min-move-px", type=float, default=40.0,
                help="board must move this far between auto-captured views")
args = ap.parse_args()

square_m = args.square_mm / 1000.0
marker_m = square_m * (intrinsics.BOARD_MARKER_M / intrinsics.BOARD_SQUARE_M)
board = intrinsics.make_board(square_m=square_m, marker_m=marker_m)

if args.print_board:
    n = intrinsics.BOARD_SQUARES
    img = board.generateImage((n[0] * 200, n[1] * 200), marginSize=40)
    cv2.imwrite("board.png", img)
    print(f"wrote board.png ({n[0]}x{n[1]} squares). Print at 100% scale, measure a "
          f"square, pass --square-mm <measured> when calibrating.")
    sys.exit(0)


def frames():
    if args.images:
        for p in sorted(glob.glob(args.images)):
            img = cv2.imread(p)
            if img is not None:
                yield img
        return
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    if not cap.isOpened():
        sys.exit("cannot open capture source")
    while True:
        ok, img = cap.read()
        if not ok:
            return
        yield img


views = []
last_center = None
size = None
t_last_msg = 0.0
for img in frames():
    size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    view = intrinsics.detect_view(gray, board)
    now = time.monotonic()
    if view is None:
        if now - t_last_msg > 2.0:
            t_last_msg = now
            print(f"[{len(views)}/{args.views}] board not visible / too few corners")
        continue
    # Auto-capture gate: only bank a view once the board has genuinely moved,
    # otherwise 20 near-identical views solve to a confident wrong answer.
    if last_center is not None and args.min_move_px > 0:
        dx = view.center_px[0] - last_center[0]
        dy = view.center_px[1] - last_center[1]
        if (dx * dx + dy * dy) ** 0.5 < args.min_move_px:
            continue
    views.append(view)
    last_center = view.center_px
    print(f"[{len(views)}/{args.views}] captured ({len(view.obj_points)} corners) — "
          f"move the board")
    if len(views) >= args.views:
        break

if len(views) < 4:
    sys.exit(f"only {len(views)} usable views — need at least 4 (got a printed board? "
             f"enough light? try --min-move-px 0 for image sets)")

print(f"solving with {len(views)} views…")
result = intrinsics.calibrate(views, size)
K = result.camera_matrix
print(f"fx={K[0][0]:.1f} fy={K[1][1]:.1f} cx={K[0][2]:.1f} cy={K[1][2]:.1f}")
print(f"dist={[round(float(d), 4) for d in result.dist_coeffs]}")
print(f"mean reprojection error: {result.reprojection_error_px:.3f} px "
      f"(worst view {max(result.per_view_error_px):.3f})")
out = intrinsics.save(result, args.out)
print(f"saved -> {out}")
if result.reprojection_error_px < 0.5:
    print("PASS  < 0.5 px (SPEC 6.2) — the ~5%-angle-error startup warning is gone")
else:
    print("WARN  >= 0.5 px: recapture with more varied tilts/positions, check the "
          "board is truly flat, and verify --square-mm against a ruler")
