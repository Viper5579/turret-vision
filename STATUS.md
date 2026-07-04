# Project Status

Last updated: 2026-07-04 · Hardware: Jetson Orin Nano Super + Arducam OV9782 USB

## Where we are

**Phase 2 (vision pipeline) is complete and running on real hardware at
79–91 fps**, up from the ~30 fps it started at. Phase 3 (binary protocol +
serial link to the ESP32) has not started.

```
capture (GStreamer hw decode, 100fps) -> frame_diff -> alpha-beta tracker
    -> pixel->angle -> console link          [tuned live via browser UI]
```

## The fps investigation (why it was 30, how it got to ~90)

Live camera fps sat near 30 despite the OV9782's MJPG 1280x800@100 mode.
`tools/measure_capture.py` (measures camera delivery and JPEG decode cost
separately) plus the negotiated-format readback pinned down THREE stacked
causes:

| Suspect | Finding | Fix |
|---|---|---|
| Pixel format fallback (YUYV = 10fps) | **Cleared** — MJPG negotiated correctly | none needed (startup now warns if it ever regresses) |
| Auto-exposure / long exposure time | Real: default 15.7ms exposure + AE hunting; also caused false frame_diff detections | manual exposure 8ms (`exposure_time_absolute=80`), auto-WB off — via the tuning UI's Camera section |
| Software JPEG decode | ~12 ms/frame on the CPU — a ~60fps ceiling alone | `camera.backend: gstreamer` → Jetson hardware decoder (`nvv4l2decoder`) |

Bonus finding: cv2's V4L2 backend itself tops out at ~57 fps on this camera
even grab-only, while the GStreamer path reads **102 fps** from the same
device — one more reason the gstreamer backend is the right default on the
Jetson.

Measured end state: GStreamer hw decode 102 fps raw; full pipeline 79–91 fps
(remaining gap = detect/track/overlay + tuning-stream overhead; acceptable
for Phase 2/3, revisit only if it ever matters for intercept quality).

## What got built beyond the original Phase 2 scope

- **Browser tuning UI** (`--tune`): live MJPEG view with raw-detection boxes,
  pipeline stats, sliders for every detector/tracker knob, and a Camera
  section (exposure/WB/gain via v4l2-ctl). Works headless over the LAN.
- **Config overlay**: Save in the UI writes `config/local.yaml` (gitignored),
  auto-merged over `default.yaml` on every run. Camera driver controls saved
  there are re-applied at startup, since UVC settings reset on replug/reboot.
- **GStreamer capture backend** (`capture/gstreamer.py`) with hw→sw decoder
  fallback and custom-pipeline override (`camera.gst_pipeline`).
- **Diagnostics**: `tools/measure_capture.py` (delivery vs decode split,
  exposure-math cross-check, `--exposure N`, `--gst`), negotiated-FOURCC
  warning in the v4l2 backend.
- **Jetson install path** that can't clobber the JetPack OpenCV/numpy
  (README Setup), `[build-system]` fix for editable installs.
- Test suite grew 14 → 26 (tuning stack, UVC control layer, gstreamer
  pipeline construction).

## To make the current state permanent (2 minutes, one-time)

1. Make gstreamer the default backend — add to `config/local.yaml`:
   ```yaml
   camera:
     backend: gstreamer
   ```
   (until then, pass `--source gstreamer` on each run)
2. Lock the camera exposure permanently: run with `--tune`, open the UI,
   Camera section → Auto exposure OFF, Exposure time 80, Auto WB OFF →
   **Save to config**. Saved controls re-apply at every startup from then on.

## Next up

- [ ] **Static-target sanity test** (from the Phase 2 bring-up notes): point
      the camera at a static scene with exposure/WB locked, confirm zero
      detections fire, then a real motion-tracking test.
- [ ] Iterate `exposure_time_absolute` against room lighting (80 = 8ms is a
      starting point; raise gain, not exposure, if too dark).
- [ ] **Phase 3**: binary protocol + pyserial link + MockLink + simulators
      (`firmware/PROTOCOL.md` has the wire format).
- [ ] Phase 4: ChArUco intrinsics calibration (kills the ~5% angle error from
      the FOV fallback), boresight, ranging, lead prediction.
- [ ] Phase 4.5: ego-motion compensated differencing (replaces the
      quasi-static gate so detection works mid-slew).

## Pick-up commands

```bash
python3 -m turretvision.main --tune --headless --source gstreamer  # run + tune UI
#   -> http://<jetson-ip>:8089
python3 tools/measure_capture.py /dev/video0 --gst   # capture health check
python3 -m pytest -q && python3 -m ruff check .      # 26 tests
```
