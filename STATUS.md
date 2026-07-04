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

## Configuration: nothing to edit by hand

The hardware-proven values are now the committed defaults in
`config/default.yaml` — a plain `git pull` brings them, and
`python3 -m turretvision.main --tune --headless` (no flags) runs the full
working setup: gstreamer backend + manual exposure 8ms + WB locked, applied
to the camera at every startup.

`config/local.yaml` is only for machine/room-specific overrides (e.g. a
different exposure for your lighting). It is **gitignored — `git pull` never
touches or reverts it**. Two ways to write it: the tuning UI's **Save to
config** button, or by hand (`nano config/local.yaml`, same YAML shape as
default.yaml, only the keys you want to override).

## Phase 2 sign-off checklist (the last bits — validation, not code)

Exit criterion per SPEC §7: *"Live window tracks a thrown bright object."*

- [ ] **Static-scene test**: camera at a static scene, nothing moving →
      DETECTIONS stat in the tuning UI should sit at 0. If it fires, the
      exposure/WB locks aren't holding (check the Camera section) before
      touching any detector knob.
- [ ] **Thrown-object test**: toss something bright across the frame → the
      overlay locks on (TRACKING pill), az/el track it, conf near 1.
- [ ] If the image is too dark at 8ms exposure: raise **Gain** in the UI,
      not exposure, then Save.

Both pass → Phase 2 is formally complete.

## Phase 3 prep: what hardware you need

**None.** Per SPEC §7: *"Nothing before Phase 4.5 requires the ESP32 to
exist."* Phase 3 is the binary protocol + SerialLink + **MockLink** (a fake
turret in software) + `sim_target`; its exit criteria run entirely on the
Jetson (byte-exact packet unit tests + a serial **loopback** test). Do NOT
mount the camera/servo/stepper to the turret yet — that's Phase 4.5+
territory (ego-motion work is when the turret actually slewing matters).

Optional, only for the loopback test at the end of Phase 3: any USB-serial
adapter with TX jumpered to RX (an ESP32 dev board on USB can serve as
exactly that). Worth ordering if not on hand, but it doesn't block the
protocol/MockLink work.

## Next up

- [ ] Phase 2 sign-off checklist above (10 minutes with the tuning UI open)
- [ ] **Phase 3**: protocol.py + SerialLink + MockLink + link_monitor +
      sim_target (`firmware/PROTOCOL.md` has the wire format)
- [ ] Phase 4: ChArUco intrinsics calibration (kills the ~5% angle error from
      the FOV fallback), boresight, ranging, lead prediction — this is where
      the physical turret build starts to matter
- [ ] Phase 4.5: ego-motion compensated differencing (replaces the
      quasi-static gate so detection works mid-slew)

## Pick-up commands

```bash
python3 -m turretvision.main --tune --headless    # run + tune UI (defaults do the rest)
#   -> http://<jetson-ip>:8089
python3 tools/measure_capture.py /dev/video0 --gst   # capture health check
python3 -m pytest -q && python3 -m ruff check .      # 26 tests
```
