# turret-vision

Real-time vision/tracking pipeline for a small educational turret.
Jetson Orin Nano Super processes camera frames; an ESP32 (later phases) drives
a NEMA 17 yaw stepper + pitch servo from absolute angle setpoints.

**Read SPEC.md first** — every design decision in this codebase is documented
there with its rationale (decisions D1–D7). The short version: frame differencing
is the primary detector, commands are absolute idempotent setpoints, the protocol
is bidirectional from v1, and everything runs with zero hardware attached.

## Status
- [x] Phase 2 — capture (v4l2/replay/synthetic), detectors (frame_diff/aruco/color_mask),
      alpha-beta tracker, pixel→angle geometry, overlay, config, console output
- [ ] Phase 3 — binary protocol + pyserial link + MockLink + simulators
- [ ] Phase 4 — calibration tools, ranging, lead prediction
- [ ] Phase 4.5 — ego-motion compensated frame differencing
- [ ] Phase 5 — replay regression harness, docs polish

## Setup (Jetson)
```bash
pip install -e ".[dev]"       # inside your working container/venv
# NOTE: on the Jetson keep your existing numpy<2 pin — this code is agnostic,
# but your OpenCV build was compiled against numpy 1.x ABI.
```

## Run
```bash
# 0. Verify what modes the camera ACTUALLY supports (do this once, first):
python tools/enumerate_camera.py /dev/video0
#    then set camera.width/height/fps in config/default.yaml to a REAL mode.

# Live camera with debug window:
python -m turretvision.main

# No hardware at all — synthetic target, headless:
python -m turretvision.main --source synthetic --headless --max-frames 300

# Replay a recording:
python -m turretvision.main --source replay --replay logs/run1.mp4
```
`q` quits the window. Per-frame state lands in `logs/state.csv` for plotting.

## Verify
```bash
python -m pytest -q      # 14 tests: filters, geometry, tracker, end-to-end synthetic
ruff check .
```

## Tuning quick reference (config/default.yaml)
- Tracker feels **laggy** → raise `tracking.alpha` (position trust)
- Tracker feels **jittery** → lower `tracking.alpha`, then `tracking.beta`
- Detector misses dim/small targets → lower `detection.frame_diff.threshold` / `min_area_px`
- Detector fires on noise → raise the same two
- Track flaps between valid/invalid → raise `tracking.max_coast_frames` or lower
  `tracking.min_confidence_output`

## Troubleshooting
- **Black window / no camera**: check `v4l2-ctl --list-devices`; the OV9782 must be
  the device in `camera.device`.
- **fps far below requested**: the UVC firmware fell back to another mode — this is
  why `tools/enumerate_camera.py` exists; pick a listed mode exactly.
- **`[warn] no intrinsics file`**: expected until Phase 4 calibration; angles carry
  ~5% scale error from the FOV fallback, fine for bring-up.
