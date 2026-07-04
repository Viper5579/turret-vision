# turret-vision

Real-time vision/tracking pipeline for a small educational turret.
Jetson Orin Nano Super processes camera frames; an ESP32 (later phases) drives
a NEMA 17 yaw stepper + pitch servo from absolute angle setpoints.

**Read SPEC.md first** — every design decision in this codebase is documented
there with its rationale (decisions D1–D7). The short version: frame differencing
is the primary detector, commands are absolute idempotent setpoints, the protocol
is bidirectional from v1, and everything runs with zero hardware attached.

## Status
See **STATUS.md** for the detailed current state, the fps investigation
findings, and pick-up commands.
- [x] Phase 2 — capture (v4l2/gstreamer/replay/synthetic), detectors
      (frame_diff/aruco/color_mask), alpha-beta tracker, pixel→angle geometry,
      overlay, config, console output
- [x] Phase 2.5 — browser tuning UI (pipeline + camera controls), config
      overlay persistence, capture diagnostics; verified on hardware at
      79–91 fps via Jetson hw JPEG decode
- [x] Phase 3 — binary protocol (framing/CRC16/pack/unpack) + SerialLink +
      MockLink (simulated turret w/ firmware behaviors) + link_monitor +
      sim_target; loopback serial test passes over a pty, sim_target passes
      all exit criteria with zero wire errors
- [x] Phase 4 (code) — ChArUco intrinsics + boresight calibration tools,
      ranging (fixed / known_size / aruco_pose), constant-velocity lead
      predictor with gravity drop; lead point renders on the overlay.
      Hardware validation (range error <15% on ArUco at 2/4/6 m with a tape
      measure) pending — see STATUS.md checklist
- [ ] Phase 4.5 — ego-motion compensated frame differencing
- [ ] Phase 5 — replay regression harness, docs polish

## Setup (dev machine / CI)
```bash
pip install -e ".[dev]"
```

## Setup (Jetson)
JetPack ships an OpenCV built against numpy 1.x, with GStreamer/CUDA/CSI
support that the generic PyPI `opencv-python` wheel does NOT have. A plain
`pip install -e ".[dev]"` will pull that wheel plus numpy 2.x and silently
shadow the system build — do this instead:

```bash
# 1. JetPack's stock setuptools (59.x) is too old for editable installs
#    (PEP 660 needs >=64):
pip3 install --user --upgrade "setuptools>=64" wheel

# 2. Install WITHOUT deps so system cv2/numpy stay untouched:
pip3 install -e . --no-deps --no-build-isolation

# 3. Add the remaining deps individually (never opencv-python/numpy here):
pip3 install pyyaml pyserial pytest ruff
```

If you already clobbered the system packages, recover with
`pip3 uninstall opencv-python numpy -y` and confirm `python3 -c "import cv2;
print(cv2.__version__, cv2.__file__)"` points back at
`/usr/lib/python3.10/dist-packages`.

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

## Calibration (Phase 4)
```bash
# 1. Intrinsics (kills the ~5% FOV-fallback angle error):
python tools/calibrate_camera.py --print-board    # print board.png at 100% scale
python tools/calibrate_camera.py --square-mm 35   # pass the MEASURED square size
#    move the board around until 20 views auto-capture; accept if error < 0.5 px

# 2. Boresight (after a launcher axis stand-in exists — laser pointer works):
#    aim the launcher axis at a printed ArUco marker, then:
python tools/calibrate_boresight.py --range 4.0

# 3. Range validation (the Phase 4 exit criterion):
#    detection.mode: aruco + ranging.mode: aruco_pose in config, marker at
#    tape-measured 2/4/6 m — overlay shows "rng X.XXm (aruco_pose)"; <15% error.
```
Ranging mode and the lead predictor are config (`ranging:`, `lead:`); flip
`lead.enabled: true` once ranging is sane and the overlay draws the aim point
offset ahead of the target.

## Turret link (Phase 3)
`link.backend` in the config selects the turret connection: `console`
(default; human-readable prints, no telemetry), `mock` (simulated turret
dynamics + telemetry — the full closed loop with zero hardware), or `serial`
(real ESP32 over pyserial; heartbeats at `link.aim_rate_hz` even with no
target, so the firmware can tell "no target" from "link dead").

```bash
python tools/sim_target.py                    # full pipeline vs MockLink, scored,
                                              # exit code 0 = Phase 3 criteria pass
python tools/link_monitor.py /dev/ttyUSB0     # decode live packet traffic
```
The wire format reference implementation is `turretvision/link/protocol.py`;
`firmware/PROTOCOL.md` tells the future ESP32 firmware exactly what to match
(including MockLink's behavior model and the rate-feedforward finding).
`q` quits the window. Per-frame state lands in `logs/state.csv` for plotting.

## Tuning UI (browser)
```bash
python -m turretvision.main --tune --headless     # on the Jetson, over SSH
```
Then open `http://<jetson-ip>:8089` from any machine on the LAN. You get the
live camera view (with raw detection boxes and the track overlay), pipeline
stats, the negotiated camera mode, and sliders for every detector/tracker
knob. On a real camera there is also a **Camera (UVC driver)** section —
auto-exposure/auto-WB toggles, exposure time, gain, WB temperature — driven
through `v4l2-ctl`, so the exposure-locking dance from Troubleshooting is
two switches and a slider instead of five shell commands.

Changes apply to the running pipeline instantly; **Save to config** persists
them to `config/local.yaml`, which is auto-merged over `default.yaml` on
every subsequent run (tuned or not) and is gitignored. Saved camera controls
land under `camera.v4l2_ctrls` and are re-applied to the driver at every
startup (UVC settings don't survive replug on their own). **Revert**
restores the values the session started with. `--tune` also works alongside
the local cv2 window, and the port comes from `ui.tune_port` or `--tune-port`.

## Verify
```bash
python -m pytest -q      # 64 tests: filters/geometry/tracker, protocol (byte-exact
                         # + fuzz), mock turret, serial loopback (pty), calibration
                         # (synthetic known-camera), ranging, lead, end-to-end
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
- **Constant false detections with nothing moving (live camera)**: disable
  auto-exposure and auto-white-balance FIRST, before touching any
  `detection.frame_diff.*` knob. Continuous AE/AWB hunting shifts global
  brightness/color between frames, which frame differencing reads as motion
  everywhere — it masquerades as a detector sensitivity problem and sends you
  tuning the wrong thing. On a UVC camera (e.g. the OV9782):
  ```bash
  v4l2-ctl -d /dev/video0 -c auto_exposure=1            # manual mode
  v4l2-ctl -d /dev/video0 -c white_balance_automatic=0
  v4l2-ctl -d /dev/video0 -c white_balance_temperature=4600
  v4l2-ctl -d /dev/video0 -c backlight_compensation=0
  v4l2-ctl -d /dev/video0 -c exposure_time_absolute=80  # 100µs units; see note
  ```
  Exposure must fit the frame budget: at 100 fps the frame is 10 ms, so
  `exposure_time_absolute` above ~100 (10 ms) will blow out whites and/or drop
  the real frame rate. Start around 80 and iterate against room lighting.
  Verify the settings held with `v4l2-ctl -d /dev/video0 --list-ctrls`
  (manual controls should no longer show `flags=inactive`).
- **Black window / no camera**: check `v4l2-ctl --list-devices`; the OV9782 must be
  the device in `camera.device`.
- **fps far below requested** (e.g. ~30 or ~10 instead of 100): three suspects,
  in the order to check them:
  1. **Pixel format fallback.** The OV9782 does 100fps only in MJPG; raw YUYV
     at 1280x800 is ~10fps. The pipeline now prints the NEGOTIATED format at
     startup (`[v4l2] requested MJPG ... -> negotiated YUYV ...` plus a
     warning on mismatch) — believe that line, not the config. Pick a mode
     `tools/enumerate_camera.py` actually lists.
  2. **Exposure time capping the sensor** (auto OR a stale manual value — UVC
     controls reset on replug/reboot). The sensor physically cannot exceed
     `1/exposure_time`; the OV9782's default 15.7ms caps at ~60fps, indoors AE
     lands near 30ms → ~33fps. `measure_capture.py` prints the current
     controls and says outright when delivery matches the exposure math;
     `--exposure 80` tests the fix in one command. Make it permanent via the
     tuning UI's Camera section + **Save** (re-applied every startup).
  3. **Software JPEG decode.** Decoding 1280x800 MJPG on the CPU costs
     ~10–17ms per frame — a ~35–60fps ceiling even when the camera delivers
     100. `python tools/measure_capture.py /dev/video0` measures delivery and
     decode separately (add `--gst` to benchmark the hardware-decode path
     too) and reports every issue it finds. If decode is the bottleneck, set
     `camera.backend: gstreamer` — on the Jetson that routes decode through
     the hardware block (`nvv4l2decoder`) instead of the CPU. This needs the
     GStreamer-enabled system OpenCV (yet another reason never to let pip
     shadow it).
  4. **The Python loop itself.** If `measure_capture.py` says capture is at
     full rate but the pipeline is still slow, the bottleneck is
     detect/track/overlay processing — profile that separately instead of
     touching camera settings.
- **`[warn] no intrinsics file`**: expected until Phase 4 calibration; angles carry
  ~5% scale error from the FOV fallback, fine for bring-up.
