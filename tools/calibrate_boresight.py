#!/usr/bin/env python3
"""Camera-vs-launcher boresight solver (SPEC 6.3) -> config/boresight.yaml.

Procedure: physically aim the launcher axis (laser pointer or straightedge
standing in for the future launcher) at an ArUco marker, then run this. The
marker's residual angular offset from the image center IS the camera-vs-
launcher misalignment; the saved offsets make pixel_to_angles() report angles
relative to the LAUNCHER, which is what the turret needs to aim.

Per SPEC 6.3 this only needs to be structurally present (config the geometry
consumes) until fire hardware exists — precision comes later by repeating at
two ranges (--range tags each solve so the two-range separation of angular vs
translational offset can be done when it matters).

Usage: python tools/calibrate_boresight.py --range 4.0   [--device /dev/video0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.calib.geometry import PixelAngleMapper  # noqa: E402
from turretvision.capture.base import Frame  # noqa: E402
from turretvision.detect.aruco import ArucoDetector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="/dev/video0")
ap.add_argument("--video", default=None, help="use a recording instead of the camera")
ap.add_argument("--range", type=float, required=True, dest="range_m",
                help="tape-measured distance to the marker (m); tags the solve")
ap.add_argument("--frames", type=int, default=60, help="frames to average over")
ap.add_argument("--intrinsics", default="config/camera_intrinsics.yaml")
ap.add_argument("--out", default="config/boresight.yaml")
args = ap.parse_args()

cap = (cv2.VideoCapture(args.video) if args.video
       else cv2.VideoCapture(args.device, cv2.CAP_V4L2))
if not cap.isOpened():
    sys.exit("cannot open capture source")
ok, img = cap.read()
if not ok:
    sys.exit("no frames from source")
h, w = img.shape[:2]

# NOTE boresight offsets forced to 0 here: we are MEASURING them; loading an
# old boresight.yaml would bake the previous solve into the new one.
mapper = PixelAngleMapper(w, h, intrinsics_file=args.intrinsics,
                          boresight_yaw_deg=0.0, boresight_pitch_deg=0.0)
if not mapper.calibrated:
    print("WARN: no intrinsics — offsets will carry the ~5% FOV-fallback scale error. "
          "Run tools/calibrate_camera.py first for a solve worth keeping.")
detector = ArucoDetector()

azs, els = [], []
n_read = 0
while len(azs) < args.frames and n_read < args.frames * 20:
    ok, img = cap.read()
    if not ok:
        break
    n_read += 1
    dets = detector.detect(Frame(img=img, t=0.0, idx=n_read))
    if not dets:
        continue
    az, el = mapper.pixel_to_angles(dets[0].cx, dets[0].cy)
    azs.append(az)
    els.append(el)
cap.release()

if len(azs) < 10:
    sys.exit(f"marker seen in only {len(azs)} frames — need >=10 steady views "
             f"(marker printed? in frame? enough light?)")

mean_az = sum(azs) / len(azs)
mean_el = sum(els) / len(els)
spread = max(max(azs) - min(azs), max(els) - min(els))
# The launcher points AT the marker, so the marker should read (0,0) in
# launcher frame: the boresight offset is the negation of what the camera saw.
data = {
    "boresight_yaw_deg": round(-mean_az, 4),
    "boresight_pitch_deg": round(-mean_el, 4),
    "solved_at_range_m": args.range_m,
    "n_frames": len(azs),
    "spread_deg": round(spread, 4),
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    f.write("# Camera-vs-launcher offsets from tools/calibrate_boresight.py.\n"
            "# Loaded by calib/geometry.py on top of the config's manual offsets.\n"
            "# Redo whenever the camera or launcher mounting moves. For full\n"
            "# angular/translational separation, solve again at a second range\n"
            "# and compare (SPEC 6.3).\n")
    yaml.safe_dump(data, f)
print(f"marker at az {mean_az:+.3f} el {mean_el:+.3f} deg over {len(azs)} frames "
      f"(spread {spread:.3f} deg)")
print(f"saved boresight yaw {-mean_az:+.3f} pitch {-mean_el:+.3f} -> {args.out}")
if spread > 0.2:
    print("WARN: spread > 0.2 deg — camera or marker was moving; redo on a steady mount")
