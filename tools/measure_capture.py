#!/usr/bin/env python3
"""Measure RAW camera fps with no pipeline attached.

WHY: when the pipeline reports low fps there are two suspects — the camera
(wrong pixel format, auto-exposure forcing long integration times) and the
Python loop (decode + detect + overlay + CSV). This grabs frames and does
NOTHING else, so the number it prints is the camera's actual delivery rate.
If this says 100 and the pipeline says 30, the bottleneck is processing;
if this says 30 too, fix the camera first (format fallback / exposure —
see README troubleshooting).

Usage: python tools/measure_capture.py [/dev/video0] [--seconds 5]
       (add --width/--height/--fps/--fourcc to test other modes)
"""
import argparse
import time

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("device", nargs="?", default="/dev/video0")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--fps", type=int, default=100)
ap.add_argument("--fourcc", default="MJPG")
ap.add_argument("--seconds", type=float, default=5.0)
args = ap.parse_args()

cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit(f"cannot open {args.device}")
# Same order the pipeline uses: format BEFORE resolution/fps, else V4L2 may
# renegotiate the mode out from under you.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
got_cc = fcc.to_bytes(4, "little").decode("ascii", errors="replace") if fcc else "????"
print(f"negotiated: {got_cc} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} "
      f"(requested {args.fourcc} {args.width}x{args.height} @ {args.fps})")
if got_cc != args.fourcc:
    print(f"WARNING: pixel format fell back to {got_cc} — this alone can cap fps "
          f"(YUYV@1280x800 is ~10fps on the OV9782)")

for _ in range(10):  # let exposure settle before timing
    cap.grab()

n = 0
t0 = time.monotonic()
grab_only = 0.0
while time.monotonic() - t0 < args.seconds:
    g0 = time.monotonic()
    if not cap.grab():
        continue
    grab_only += time.monotonic() - g0
    cap.retrieve()  # include the decode cost the pipeline also pays
    n += 1
dt = time.monotonic() - t0
cap.release()

print(f"{n} frames in {dt:.2f}s -> {n / dt:.1f} fps "
      f"(mean grab wait {grab_only / max(n, 1) * 1000:.1f} ms)")
if n / dt < args.fps * 0.8:
    print("camera is the bottleneck. Checklist: 1) format fallback above? "
          "2) auto_exposure still on / exposure_time_absolute too long? "
          "(v4l2-ctl -d DEV --list-ctrls)  3) mode not actually supported "
          "(tools/enumerate_camera.py)")
else:
    print("camera delivers full rate — if the pipeline is slower, the "
          "bottleneck is processing, not capture.")
