#!/usr/bin/env python3
"""Measure camera delivery rate and JPEG decode cost SEPARATELY.

WHY two phases: cv2's read() = grab (wait for the driver to hand over a
frame) + retrieve (decode MJPG -> BGR in software). A slow number from a
combined loop can't tell you which half is the problem, and the fixes are
completely different:
  - delivery slow  -> camera/driver problem: pixel format fallback or
                      auto-exposure integration time (see README)
  - delivery fine, decode slow -> CPU JPEG decode is the ceiling; on a
                      Jetson switch camera.backend to 'gstreamer' to use
                      the hardware decoder (nvv4l2decoder)

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
ap.add_argument("--seconds", type=float, default=5.0, help="duration of EACH phase")
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


def run_phase(label: str, do_retrieve: bool) -> float:
    n = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        if not cap.grab():
            continue
        if do_retrieve:
            cap.retrieve()
        n += 1
    fps = n / (time.monotonic() - t0)
    print(f"{label}: {n} frames -> {fps:.1f} fps")
    return fps


# Phase A: how fast does the camera actually hand frames over? (no decode)
delivery = run_phase("delivery (grab only)      ", do_retrieve=False)
# Phase B: what the pipeline pays per frame (grab + MJPG->BGR software decode)
effective = run_phase("effective (grab + decode) ", do_retrieve=True)
cap.release()

decode_ms = (1000.0 / effective - 1000.0 / delivery) if delivery > 0 and effective > 0 else 0.0
print(f"implied decode cost: ~{max(decode_ms, 0):.1f} ms/frame")

if delivery < args.fps * 0.8:
    print("verdict: the CAMERA/DRIVER caps delivery. Checklist: 1) format fallback "
          "above?  2) auto_exposure still on / exposure_time_absolute longer than "
          "the frame period? (v4l2-ctl -d DEV --list-ctrls)  3) mode not actually "
          "supported (tools/enumerate_camera.py)")
elif effective < delivery * 0.8:
    print("verdict: software JPEG DECODE is the bottleneck, not the camera. On a "
          "Jetson set camera.backend: gstreamer in the config to route decode "
          "through the hardware decoder (nvv4l2decoder). See README troubleshooting.")
else:
    print("verdict: capture delivers full rate — if the pipeline runs slower, the "
          "bottleneck is detect/track/overlay processing, not capture.")
