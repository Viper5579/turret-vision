#!/usr/bin/env python3
"""Measure camera delivery rate and JPEG decode cost SEPARATELY, and say why.

WHY two phases: cv2's read() = grab (wait for the driver to hand over a
frame) + retrieve (decode MJPG -> BGR in software). A slow number from a
combined loop can't tell you which half is the problem, and the fixes are
completely different. Both can be broken at once (measured on the OV9782:
default 15.7ms exposure caps delivery at ~57fps AND software decode adds
~12ms/frame), so every issue found is reported, not just the first.

The tool also reads the camera's CURRENT exposure controls and checks the
delivery rate against 1/exposure_time — if they match, the cap is exposure,
not USB/driver/mode.

Usage:
  python tools/measure_capture.py [/dev/video0]
  python tools/measure_capture.py /dev/video0 --exposure 80   # lock manual 8ms first
  python tools/measure_capture.py /dev/video0 --gst           # also test the
        # GStreamer/hardware-decode path (what camera.backend: gstreamer uses)

Nothing here persists: --exposure writes a driver control that resets on
replug/reboot, and --gst only opens a test pipeline.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.capture import uvc_ctrl  # noqa: E402
from turretvision.capture.gstreamer import SW_DECODE, build_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("device", nargs="?", default="/dev/video0")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--fps", type=int, default=100)
ap.add_argument("--fourcc", default="MJPG")
ap.add_argument("--seconds", type=float, default=5.0, help="duration of EACH phase")
ap.add_argument("--exposure", type=int, default=None, metavar="N",
                help="lock manual exposure to N (100µs units, e.g. 80 = 8ms) before measuring")
ap.add_argument("--gst", action="store_true",
                help="also measure the GStreamer pipeline (hardware decode on Jetson)")
args = ap.parse_args()

# ---- current driver state (the usual culprit) ------------------------------
cam = uvc_ctrl.probe(args.device)
exposure_ms = None
if cam:
    if args.exposure is not None:
        cam.set("auto_exposure", 1)  # manual mode first, or the next write bounces
        cam.set("exposure_time_absolute", args.exposure)
    interesting = ("auto_exposure", "exposure_time_absolute",
                   "white_balance_automatic", "gain")
    state = {k: cam.get(k) for k in interesting if k in cam.controls}
    print("driver controls:", ", ".join(f"{k}={v}" for k, v in state.items()))
    if state.get("auto_exposure") == 3:
        print("  NOTE: auto_exposure=3 means AUTO — the driver picks the exposure "
              "time; whatever you set manually is being ignored")
    if "exposure_time_absolute" in state:
        exposure_ms = state["exposure_time_absolute"] * 0.1
        print(f"  exposure_time_absolute={state['exposure_time_absolute']} "
              f"-> {exposure_ms:.1f} ms -> sensor hard cap ~{1000 / exposure_ms:.0f} fps")
else:
    print("(v4l2-ctl unavailable — skipping driver control snapshot)")

# ---- V4L2 path: delivery vs decode ------------------------------------------
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


delivery = run_phase("delivery (grab only)      ", do_retrieve=False)
effective = run_phase("effective (grab + decode) ", do_retrieve=True)
cap.release()

decode_ms = (1000.0 / effective - 1000.0 / delivery) if delivery > 0 and effective > 0 else 0.0
print(f"implied decode cost: ~{max(decode_ms, 0):.1f} ms/frame")

# ---- optional: the hardware-decode path --------------------------------------
if args.gst:
    for kind, decode in (("hw (nvv4l2decoder)", None), ("sw (jpegdec)", SW_DECODE)):
        pipe = (build_pipeline(args.device, args.width, args.height, args.fps)
                if decode is None else
                build_pipeline(args.device, args.width, args.height, args.fps, decode=decode))
        gcap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not gcap.isOpened():
            print(f"gstreamer {kind}: pipeline failed to open"
                  + (" — is this OpenCV built with GStreamer?" if decode else ""))
            gcap.release()
            continue
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            if gcap.read()[0]:
                n += 1
        print(f"gstreamer {kind}: {n} frames -> {n / (time.monotonic() - t0):.1f} fps")
        gcap.release()
        break  # hw worked; no need to test the fallback

# ---- verdicts (ALL that apply, not just the first) ---------------------------
print()
issues = 0
if delivery < args.fps * 0.9:
    issues += 1
    frame_ms = 1000.0 / delivery if delivery > 0 else float("inf")
    print(f"ISSUE {issues}: camera delivers {delivery:.0f} fps ({frame_ms:.1f} ms/frame), "
          f"below the {args.fps} the mode promises.")
    if exposure_ms and abs(frame_ms - exposure_ms) / frame_ms < 0.35:
        print(f"  -> delivery matches the current {exposure_ms:.1f} ms exposure almost "
              f"exactly: EXPOSURE is the cap. Rerun with --exposure 80, and make it "
              f"permanent via the tuning UI's Camera section + Save.")
    else:
        print("  -> exposure doesn't explain it: check format fallback above, "
              "tools/enumerate_camera.py for real modes, and the USB port/hub.")
if decode_ms > 4.0:
    issues += 1
    print(f"ISSUE {issues}: software JPEG decode costs ~{decode_ms:.1f} ms/frame "
          f"(a ~{1000 / decode_ms:.0f} fps ceiling by itself).")
    print("  -> use the hardware decoder: run the pipeline with --source gstreamer "
          "(one-off, changes nothing), or set camera.backend: gstreamer to make it "
          "the default. Verify with --gst here first if you want.")
if issues == 0:
    print("no capture issues: full rate delivered and decoded — if the pipeline is "
          "slower, the bottleneck is detect/track/overlay processing.")
