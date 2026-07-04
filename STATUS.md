# Project Status

Last updated: 2026-07-04 · Hardware: Jetson Orin Nano Super + Arducam OV9782 USB

## Where we are

**Phase 2 (vision) runs on real hardware at 79–91 fps. Phase 3 (protocol +
links) is complete and hardware-tested. Phase 4 (calibration, ranging, lead)
is code-complete** — its exit criterion (range error <15% on ArUco at
tape-measured 2/4/6 m) is a hardware measurement waiting on printed
targets; checklist below.

```
capture (GStreamer hw decode) -> frame_diff (telemetry-gated) -> tracker
  -> pixel->angle -> absolute setpoint (pose+error, clamped) -> link
       console: prints | mock: simulated turret | serial: ESP32, 50Hz heartbeat
                              ^--- telemetry (pose/rate/status) feeds back ---^
```

## Phase 3: what got built

- `link/protocol.py` — framing (AA 55 | LEN | TYPE | payload | CRC16),
  CRC-16/CCITT-FALSE, AIM/TELEM pack/unpack, byte-at-a-time parser with
  resync. This file is the reference for the ESP32 firmware.
- `link/serial_link.py` — pyserial + IO thread; the pipeline never blocks on
  a write. Heartbeats at `aim_rate_hz` even when silent/no-target (so
  firmware can tell "no target" from "link dead"), min-filtered clock offset
  for Phase 4.5 pose interpolation.
- `link/mock_link.py` — simulated turret that behaves like the future
  firmware: parses real AIM bytes, clamps to travel limits, honors e-stop,
  enters safe-hold on link timeout, trapezoidal yaw + rate-limited pitch,
  emits real TELEM bytes.
- `main.py` — telemetry now closes the loop: turret rate feeds the
  quasi-static detection gate, setpoints are absolute (pose + tracked error)
  and clamped Jetson-side.
- `tools/sim_target.py` (scored end-to-end run, exit code = pass/fail) and
  `tools/link_monitor.py` (live packet decoder).
- **Design finding worth keeping:** the AIM packet's rate feedforward field
  cuts closed-loop tracking lag ~10x (mean 3.35° → 0.39° in sim). Recorded in
  `firmware/PROTOCOL.md` so the firmware implements it, not just carries it.

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

## Phase 4: what got built

- `calib/intrinsics.py` + `tools/calibrate_camera.py` — ChArUco calibration:
  `--print-board` emits the printable board, live capture auto-banks ~20
  views (only when the board has actually moved), solves, reports
  reprojection error against the SPEC's <0.5 px acceptance, writes
  `config/camera_intrinsics.yaml`. The solver is unit-tested against
  synthetic views projected through a known camera (recovers fx within 1%).
- `tools/calibrate_boresight.py` — aims-at-marker residual solver, writes
  `config/boresight.yaml` (auto-loaded by geometry on top of config offsets).
- `ranging/` — `fixed` (bring-up default), `known_size` (pinhole similar
  triangles off the tracked bbox), `aruco_pose` (solvePnP exact range, the
  ground-truth instrument). All behind one interface, selected by
  `ranging.mode`.
- `lead/predictor.py` — closed-form constant-velocity intercept quadratic +
  gravity drop (D7), max-lead clamp as the bad-input sanity valve. Verified
  against the textbook crossing-shot answer (lead = asin(v_t/v_p)).
- Pipeline: range estimate + lead solution feed the AIM setpoint and
  `range_mm` on the wire; overlay draws range text and the aim point offset
  from the target.

## Phase 4 hardware validation checklist (your part — printer + tape measure)

1. `python3 tools/calibrate_camera.py --print-board`, print at 100% scale,
   tape it flat to cardboard, measure a square with a ruler.
2. `python3 tools/calibrate_camera.py --square-mm <measured>` — wave the board
   around until 20 views bank; accept < 0.5 px. This removes the ~5% angle
   error the pipeline has been carrying since day one.
3. Print an ArUco marker (DICT_4X4_50, any id, 10 cm — set
   `detection.aruco.marker_size_m` to the MEASURED size).
4. Config: `detection.mode: aruco`, `ranging.mode: aruco_pose`, run with
   `--tune`, put the marker at tape-measured 2, 4, and 6 m: overlay range vs
   tape < 15% at all three = **Phase 4 exit criterion met**.
5. Then try `ranging.mode: known_size` on a real thrown target and flip
   `lead.enabled: true` to watch the aim point lead the motion.

## Hardware build — start now, in parallel

Nothing in Phase 4 validation needs the turret built (a tripod or clamp that
holds the camera STILL is enough). But Phase 4.5 (ego-motion) and the first
real closed-loop test need the physical rig, so mechanical design is the
right thing to work on while the calibration prints are on the wall:

- **Pan-tilt turret**: NEMA 17 yaw (direct or belt), pitch servo. The camera
  mounts ON the pitch mechanism (SPEC: co-mounted) — design the camera mount
  as part of the pitch assembly from the start, not bolted on after.
- **Rigidity beats speed**: every calibration (boresight especially) assumes
  camera and launcher axis don't flex relative to each other. Overbuild the
  camera-to-launcher bracket; it is the one part where slop = permanent miss.
- **Wiring for the OV9782**: USB up through the yaw axis — leave a service
  loop or slip-ring plan so ±170° yaw doesn't strain the cable.
- **Electronics**: ESP32 DevKit V1 + A4988 (yaw stepper) + PCA9685 (pitch
  servo) per SPEC. The ESP32's USB port doubles as the serial link the
  Phase 3 protocol already speaks.
- **Homing**: the firmware status has a `homed` bit — plan a yaw end-stop or
  index mark so the step counter has an absolute reference on boot.
- When the ESP32 is wired: implement firmware against `firmware/PROTOCOL.md`
  (byte-exact reference: `link/protocol.py`, behavior model: MockLink, live
  debugging: `tools/link_monitor.py`).

## Next up

- [ ] Phase 2 sign-off checklist above (10 minutes with the tuning UI open)
- [ ] Phase 4 hardware validation checklist above (printer + tape measure)
- [ ] Turret mechanical design/build (see hardware section — parallel track)
- [ ] ESP32 firmware once wired: `firmware/PROTOCOL.md` + `link/protocol.py`
      reference, MockLink behavior model, `tools/link_monitor.py` debugging
- [ ] **Phase 4.5** (code): ego-motion compensated differencing — warp the
      previous frame by the telemetry pose delta so detection survives the
      turret's own motion; consumes Phase 3's telemetry + clock offset
- [ ] Phase 5: replay regression harness, docs polish

## Pick-up commands

```bash
python3 -m turretvision.main --tune --headless    # run + tune UI (defaults do the rest)
#   -> http://<jetson-ip>:8089
python3 tools/sim_target.py                       # Phase 3 exit criteria, scored
python3 tools/measure_capture.py /dev/video0 --gst   # capture health check
python3 -m pytest -q && python3 -m ruff check .      # 47 tests
```
