# SPEC.md — Turret Vision & Tracking Pipeline

**Target platform:** Jetson Orin Nano Super (JetPack 6, aarch64) → ESP32 DevKit V1 (NEMA 17 yaw via A4988, pitch servo via PCA9685)
**Camera:** Arducam OV9782, global shutter, USB UVC, ~100 fps, low-distortion M12 lens, co-mounted on the pitch mechanism
**Scope of this spec:** vision, tracking, state estimation, range, lead prediction, serial protocol. No fire hardware, no fire command.

---

## 1. Locked design decisions (and why)

These were open forks in the original brief. They are now decided. Each one is here with its rationale so the reasoning survives even if the code gets rewritten.

### D1 — Primary detector: frame differencing. ArUco is a calibration/ground-truth tool, not the tracking detector.

**Why:** The target's defining property is that it is the only fast-moving object in the scene. Frame differencing exploits exactly that property with zero training, trivially runs at 100 fps, and doesn't care what the target looks like. ArUco/color-mask detectors are still implemented behind the same `Detector` interface — ArUco because it's *required anyway* for boresight calibration and range validation (a marker of known size gives ground-truth pose for free), color mask because it's ~30 lines and useful as a static-target sanity check. This isn't scope creep; it's the calibration toolchain.

**Phasing caveat:** frame differencing while the turret itself is rotating is the hard version of the problem (everything moves in the image). The spec phases it: v1 does frame differencing with a **quasi-static gate** (suppress detection while commanded turret velocity exceeds a threshold), and ego-motion compensation (warp the previous frame by the turret's measured rotation before differencing) is a defined later phase. Why gate first: it gets a working tracking loop on real hardware weeks earlier, and the warp needs telemetry infrastructure (D3) that only exists after Phase 3 anyway.

### D2 — Jetson→ESP32 commands are **absolute angle setpoints**, not deltas.

**Why:** absolute setpoints are *idempotent* — sending the same packet twice, or losing one, causes zero accumulated error; the next packet fully corrects the state. Deltas are not: one dropped packet permanently shifts your aim, and errors compound silently over a link that will drop bytes (serial + a vibrating platform + alligator-clip history — you know this failure mode personally). This does NOT contradict the IBVS decision: the *control law* still drives pixel error to zero. The Jetson computes `setpoint = current_turret_angle (from telemetry) + angular_error (from pixel offset)` each frame and sends the result. IBVS math, idempotent transport.

Consequences:
- The ESP32 remains sole owner of the fast motion loop and step counting (as decided).
- Travel limits are enforced in **both** places: Jetson clamps setpoints (never sends an illegal command) and ESP32 clamps received setpoints (never trusts the wire). Why both: defense in depth costs two `min/max` calls and protects the hardware from bugs on either side.
- The Jetson always speaks **degrees**, never steps. Why: units are pinned at the system boundary, so a future drivetrain change (microstepping, gearing) touches only ESP32 firmware config, never the vision code.

### D3 — Bidirectional protocol from day one: ESP32→Jetson telemetry packet in v1.

**Why:** two independent consumers need the turret's actual pose: (a) the absolute-setpoint computation in D2, and (b) ego-motion compensation, which needs the pose *delta between two frame timestamps* to compute the warp. An open-loop stepper's position exists only on the ESP32 (it's the thing counting steps). Dead-reckoning from sent commands drifts on any dropped packet and is wrong whenever the stepper is mid-move at frame capture — which during tracking is *always*.

### D4 — The pipeline must run with zero hardware attached.

Every hardware boundary gets a swappable backend selected in config:

| Boundary | Real backend | Test backends |
|---|---|---|
| Camera | `V4L2Camera` | `ReplaySource` (recorded video), `SyntheticSource` (rendered moving target) |
| Turret link | `SerialLink` (pyserial) | `MockLink` (simulated turret dynamics + fake telemetry), `ConsoleLink` (prints packets) |

**Why:** you'll be testing Jetson-only first, and long-term this is what makes every bug bisectable — replay the same recorded frames through a code change and diff the outputs. `MockLink` simulates a turret with AccelStepper-like trapezoidal velocity limits and emits telemetry, so the *entire* closed loop (including ego-motion compensation later) runs on the bench with nothing plugged in.

### D5 — Undistort **points**, not frames.

Detected centroids are corrected with `cv2.undistortPoints()`; full-frame remap is off by default.

**Why:** full-frame undistortion at 100 fps burns milliseconds per frame to correct pixels you never look at. Frame differencing is insensitive to the small distortion of a low-distortion M12 lens (both frames are distorted identically, so the difference image is barely affected). The only place distortion actually matters is the pixel→angle conversion of the *one* detected point — so correct exactly that, for microseconds instead of milliseconds. Full-frame undistort remains available via config for calibration visualization.

### D6 — Filter: alpha-beta first, Kalman as a drop-in later.

**Why:** an alpha-beta filter has two gains you can reason about by hand and tune in minutes; a Kalman filter has process/measurement covariance matrices you can't populate honestly until you've *measured* your detection noise — which requires a working pipeline first. Both implement the same `StateEstimator` interface (`update(measurement, t) → TargetState`), so the swap later is one config line. Build the tool that lets you characterize the noise, then build the filter that needs the noise characterization.

### D7 — Lead model: constant-velocity intercept, no drag, **with gravity drop**.

**Why no drag:** at 3–6 m indoor ranges, drag correction on an airsoft BB is second-order; modeling it before you can measure muzzle velocity consistency (the real bottleneck, per earlier analysis) is precision theater. **Why gravity anyway:** at 50 m/s over 5 m, time of flight ≈ 0.1 s, drop = ½·g·t² ≈ **5 cm** — larger than your acceptable miss distance on a paper airplane. Gravity is one line of code and matters; drag is a config placeholder (`drag_model: none`) until Phase-3-hardware data exists.

---

## 2. Repository structure

```
turret-vision/
├── SPEC.md
├── README.md
├── pyproject.toml               # deps, ruff + pytest config
├── config/
│   ├── default.yaml             # master config, documented inline
│   ├── camera_intrinsics.yaml   # generated by tools/calibrate_camera.py
│   └── boresight.yaml           # generated by tools/calibrate_boresight.py
├── turretvision/
│   ├── main.py                  # pipeline assembly + main loop only
│   ├── capture/
│   │   ├── base.py              # FrameSource ABC: read() -> Frame(img, t_mono)
│   │   ├── v4l2.py              # threaded UVC capture, timestamp at grab
│   │   ├── replay.py            # recorded video + sidecar timestamps
│   │   ├── synthetic.py         # rendered moving target, ground truth known
│   │   └── recorder.py          # tee frames+timestamps to disk
│   ├── calib/
│   │   ├── intrinsics.py        # load/save/apply camera calibration
│   │   └── geometry.py          # pixel <-> angle math, boresight offsets
│   ├── detect/
│   │   ├── base.py              # Detector ABC: detect(frame, ctx) -> [Detection]
│   │   ├── frame_diff.py        # primary; quasi-static gate; ego-motion hook
│   │   ├── aruco.py             # calibration/ground truth; pose estimation
│   │   └── color_mask.py        # HSV mask sanity-check detector
│   ├── track/
│   │   ├── tracker.py           # association, coasting, confidence, track lifecycle
│   │   └── filters.py           # AlphaBeta + (later) Kalman behind StateEstimator ABC
│   ├── ranging/
│   │   ├── base.py              # RangeEstimator ABC
│   │   ├── known_size.py        # apparent size + intrinsics -> distance
│   │   ├── aruco_pose.py        # marker pose -> exact distance
│   │   └── fixed.py             # configured constant range (test mode)
│   ├── lead/
│   │   └── predictor.py         # intercept solver, gravity drop, confidence
│   ├── link/
│   │   ├── protocol.py          # packet pack/unpack, CRC16, framing (pure functions)
│   │   ├── base.py              # TurretLink ABC: send_aim(), poll_telemetry()
│   │   ├── serial_link.py       # pyserial, dedicated IO thread, non-blocking queue
│   │   ├── mock_link.py         # simulated turret dynamics + telemetry
│   │   └── console_link.py      # human-readable packet dump
│   ├── ui/
│   │   └── overlay.py           # debug draw: target, velocity vector, lead point, FPS, latency
│   └── util/
│       ├── config.py            # YAML load, validation, dot-access
│       ├── timing.py            # monotonic clocks, per-stage latency stats
│       └── logging.py           # structured CSV/JSONL run logs for replay analysis
├── firmware/
│   └── PROTOCOL.md              # ESP32-side implementation notes (firmware later)
├── tools/
│   ├── enumerate_camera.py      # list real V4L2 modes/fps of the OV9782
│   ├── calibrate_camera.py      # ChArUco intrinsic calibration
│   ├── calibrate_boresight.py   # camera-vs-launcher offset solver
│   ├── record.py                # capture session to disk
│   ├── replay.py                # run pipeline on a recording
│   ├── sim_target.py            # synthetic end-to-end run with ground truth scoring
│   └── link_monitor.py          # decode + display live packet traffic
└── tests/
    ├── test_protocol.py
    ├── test_filters.py
    ├── test_ranging.py
    ├── test_lead.py
    └── test_geometry.py
```

**Why this shape:** each directory is one replaceable stage; the ABCs in each `base.py` are the contracts. `main.py` contains *no logic* — only wiring — so any stage can be tested against recorded data without dragging the rest along. `protocol.py` is pure functions (bytes in, dataclass out) specifically so it's unit-testable with zero hardware and reusable as the reference when writing the ESP32 firmware.

---

## 3. Data flow

```
                          ┌────────────────────────────────────────────┐
                          │                telemetry (100 Hz)          │
                          ▼                                            │
 FrameSource ──► Detector ──► Tracker/Filter ──► Range ──► Lead ──► TurretLink ──► ESP32
 (thread,        (frame_diff:   (alpha-beta:      (mode-     (intercept   (IO thread,
  t_mono at       gate/warp      pos, vel,         select)    + gravity,   non-blocking
  grab)           via telem)     confidence)                  setpoint)    queue)
      │                                                                        
      └──► Recorder (optional tee)          Overlay ◄── every stage (debug taps)
```

Per-frame contract (dataclasses, all carrying the frame's monotonic timestamp):

```
Frame(img, t_mono)
Detection(cx, cy, area, bbox, kind, t_mono)
TargetState(az_deg, el_deg, az_rate, el_rate, confidence, coasting, t_mono)   # angles, not pixels
RangeEstimate(dist_m, sigma_m, method)
AimCommand(yaw_set_deg, pitch_set_deg, yaw_rate, pitch_rate, confidence, range_mm, flags, t_mono)
Telemetry(yaw_deg, pitch_deg, yaw_rate, status, t_esp_ms, t_rx_mono)
```

**Why angles immediately after tracking:** converting pixels→angles (via intrinsics + boresight) as early as possible means every downstream module (range, lead, protocol, logs) is in physical units, so changing camera resolution or lens changes exactly one conversion, not five modules.

**Threading model:** capture thread (grab + timestamp), main processing loop, serial IO thread. **Why timestamp at grab, in the capture thread:** V4L2/OpenCV buffer frames; timestamping when the *processing loop* gets around to a frame can be 1–3 frames late, which corrupts every velocity estimate downstream (velocity = Δposition/Δt — garbage t, garbage v). Capture thread also sets buffer size 1 and drops stale frames: for control, the newest frame is worth more than every frame.

---

## 4. Configuration schema (`config/default.yaml`)

```yaml
camera:
  backend: v4l2            # v4l2 | replay | synthetic
  device: /dev/video0
  width: 1280
  height: 800
  fps: 100                 # requested; verify with tools/enumerate_camera.py
  fourcc: MJPG             # verify actual modes before trusting datasheet fps
  replay_path: null

calibration:
  intrinsics_file: config/camera_intrinsics.yaml
  boresight_file: config/boresight.yaml
  undistort_points: true
  undistort_full_frame: false     # debug/calibration only (slow)

detection:
  mode: frame_diff         # frame_diff | aruco | color_mask
  frame_diff:
    threshold: 25          # abs-diff intensity threshold
    min_area_px: 40
    max_area_px: 20000
    morph_kernel: 3
    static_gate_dps: 5.0   # suppress detection above this commanded turret speed (v1)
    ego_motion_comp: false # phase 4.5+
  aruco:
    dictionary: DICT_4X4_50
    marker_size_m: 0.10
  color_mask:
    hsv_low: [5, 120, 120]
    hsv_high: [25, 255, 255]

tracking:
  filter: alpha_beta       # alpha_beta | kalman (later)
  alpha: 0.5
  beta: 0.2
  max_coast_frames: 8      # keep predicting through misses this long
  gate_px: 120             # max association distance, rejects spurious blobs
  min_confidence_output: 0.4   # below this -> target_valid = false on the wire

ranging:
  mode: fixed              # fixed | known_size | aruco_pose
  fixed_distance_m: 4.0
  target_size_m: 0.30      # paper airplane wingspan; see limitations §9

lead:
  enabled: true
  projectile_speed_mps: 50.0
  drag_model: none         # placeholder; none only in v1
  gravity_comp: true
  max_lead_deg: 15.0       # clamp; huge computed leads mean bad inputs, not real leads

turret:
  yaw_min_deg: -170.0
  yaw_max_deg: 170.0
  pitch_min_deg: -10.0
  pitch_max_deg: 60.0
  camera_to_launcher_offset_m: [0.0, 0.03, 0.0]   # x right, y up, z forward
  # steps-per-degree lives in ESP32 firmware; Jetson speaks degrees ONLY (see D2)

link:
  backend: mock            # mock | serial | console
  port: /dev/ttyUSB0
  baud: 115200
  aim_rate_hz: 50
  telemetry_timeout_s: 0.5  # stale telemetry -> hold fire on ego-motion features

logging:
  run_log_dir: logs/
  log_level: INFO
  record_frames: false
  csv_state_log: true      # per-frame state for offline plots
```

**Why config over constants everywhere:** every value above is one you *will* retune when the projectile, lens, or target changes. The rule applied: if changing it shouldn't require re-reading code, it's config.

---

## 5. Serial protocol (`firmware/PROTOCOL.md` mirrors this)

Binary, little-endian, framed. UART/USB-CDC @ 115200 (bandwidth analysis below shows this is ~5× headroom; bump to 460800 only if measured latency demands it).

### Framing (both directions)

```
[0xAA] [0x55] [LEN u8] [TYPE u8] [PAYLOAD ...] [CRC16 u16]
```

- `LEN` = payload byte count.
- `CRC16` = CRC-16/CCITT-FALSE over `LEN..PAYLOAD`. **Why CRC16 and not an XOR checksum:** XOR passes any two errors that cancel and all byte-order swaps — exactly the corruption a noisy line on a vibrating platform produces. CRC16 catches all single/double-bit errors and all burst errors ≤16 bits, and costs one table lookup per byte on the ESP32.
- Receiver is a byte-at-a-time state machine (hunt 0xAA → 0x55 → …). **Why:** serial has no message boundaries; assuming reads align to packets works on the bench and fails in the field. Resync on any CRC failure.

### AIM packet — Jetson → ESP32, `TYPE=0x01`, sent at `aim_rate_hz`

| Field | Type | Units / notes |
|---|---|---|
| t_ms | u32 | Jetson monotonic ms (wraps; deltas only) |
| flags | u8 | bit0 target_valid, bit1 **disable/e-stop**, bit2-3 mode (0 idle, 1 track, 2 manual) |
| yaw_set | i16 | centidegrees, absolute |
| pitch_set | i16 | centidegrees, absolute |
| yaw_rate | i16 | centideg/s (feedforward; 0 if unused) |
| pitch_rate | i16 | centideg/s |
| confidence | u8 | 0–255 |
| range_mm | u16 | 0 = unknown |

Payload 16 B → frame 22 B → at 50 Hz = 8.8 kbps. **Why i16 centidegrees instead of float32:** 0.01° resolution is ~10× finer than a 1.8° stepper can use even at 1/16 microstepping (0.1125°/step), halves packet size, and sidesteps float-marshaling/endianness bugs between Python's `struct` and C.

**No-target behavior:** packets keep flowing at `aim_rate_hz` with `target_valid=0` and setpoints frozen at last value. **Why keep sending:** a silent line is ambiguous — crashed Jetson? unplugged cable? no target? A heartbeat with an explicit flag lets the ESP32 distinguish "no target, hold position" from "link dead, enter safe state after timeout."

**E-stop semantics (bit1):** when set, ESP32 must stop motion and ignore setpoints until cleared. It's in every packet (not a separate message) so it can't be lost as a one-shot. No fire command exists in this protocol; adding one later is a new TYPE, not a change to this packet.

### TELEM packet — ESP32 → Jetson, `TYPE=0x02`, ~100 Hz

| Field | Type | Units / notes |
|---|---|---|
| t_ms | u32 | ESP32 millis() |
| yaw_pos | i16 | centidegrees (from step count) |
| pitch_pos | i16 | centidegrees (commanded servo angle) |
| yaw_rate | i16 | centideg/s |
| status | u8 | bit0 homed, bit1 yaw@limit, bit2 pitch@limit, bit3 fault, bit4 e-stopped |

**Clock domains:** ESP32 `millis()` and Jetson monotonic are unrelated. The Jetson stamps each telemetry packet on receipt (`t_rx_mono`) and maintains a rolling offset estimate `offset = t_rx_mono − t_ms` (min-filtered). **Why min-filtered:** receive jitter only ever *adds* delay, so the minimum observed offset is the best estimate of the true one. Good to ~1–2 ms, sufficient for ego-motion pose interpolation at 100 Hz.

---

## 6. Calibration workflow

1. **Camera modes** (`tools/enumerate_camera.py`): dump real V4L2 formats/fps. **Why first:** "~100 fps" is a datasheet claim; the actual UVC firmware exposes specific resolution/format/fps combos, and every latency budget downstream depends on which one is real. Verify, don't trust marketing — this project has been burned by that pattern before.
2. **Intrinsics** (`tools/calibrate_camera.py`): ChArUco board, ~20 views, saves `camera_intrinsics.yaml` (fx, fy, cx, cy, dist coeffs, reprojection error). **Why ChArUco over plain chessboard:** works with partial board views and gives corner IDs, so fewer garbage frames poison the solve. Accept < 0.5 px mean reprojection error.
3. **Boresight** (`tools/calibrate_boresight.py`): with a laser pointer or straightedge standing in for the future launcher axis, aim turret at an ArUco marker at known range; the residual pixel offset of the marker from image center = camera-vs-launcher angular offset. Saved to `boresight.yaml`, applied in `calib/geometry.py`. Repeat at two ranges to separate angular from translational offset. **Why it can wait for precision:** until fire hardware exists, boresight only needs to be *structurally present* (a config the geometry math consumes), not accurate.
4. **Actuator mapping:** lives in ESP32 firmware (steps/deg, servo µs↔deg, limits). Jetson-side verification: command +30.00°, read telemetry, confirm agreement — this catches unit mismatches at the boundary, historically the most embarrassing class of robotics bug.
5. **End-to-end check:** printed ArUco at measured distances (2/4/6 m tape measure): verify detected angle vs protractor-truth, range vs tape, packet contents via `tools/link_monitor.py`.

---

## 7. Phased implementation plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **1** | This spec | Approved by you |
| **2** | Capture (v4l2/replay/synthetic) + frame_diff & aruco detectors + alpha-beta tracker + overlay + config + ConsoleLink | Live window tracks a thrown bright object; FPS/latency on overlay; runs from replay with no camera |
| **3** | protocol.py + SerialLink + MockLink + link_monitor + sim_target | `sim_target` drives full pipeline against MockLink; packets verified byte-exact by unit tests; loopback serial test passes |
| **4** | Calibration tools (enumerate, intrinsics, boresight) + known_size/aruco ranging + lead predictor | Range error <15% on ArUco at 2/4/6 m; lead point renders sanely on overlay |
| **4.5** | Ego-motion compensation in frame_diff (warp prev frame by telemetry pose delta) | Detection survives commanded pans in MockLink/replay tests |
| **5** | Test suite, replay regression harness, README | `pytest` + `ruff` clean; documented setup/run/calibrate/troubleshoot |

Note phase 2 outputs to console, phase 3 adds the wire — matching your "test with just the Jetson first" requirement. Nothing before 4.5 requires the ESP32 to exist.

---

## 8. Performance targets & verification

| Metric | Target | Measured how |
|---|---|---|
| Capture→setpoint latency | < 20 ms | `util/timing.py` per-stage stamps, overlay + CSV |
| Pipeline throughput | ≥ 60 fps (stretch: camera-limited) | overlay + CSV |
| Detection recall (bright target, static turret, indoor) | > 90 % of frames | replay scoring vs hand labels |
| Range error (ArUco, static) | < 10 % | tape-measure ground truth |
| Range error (known-size airplane) | < 25 % (see §9) | tape-measure ground truth |
| Velocity estimate settle | < 10 frames on synthetic constant-velocity | `test_filters.py` |
| Packet integrity | 0 CRC-accepted corrupt packets in fuzz test | `test_protocol.py` |

Verification per phase: `pytest -q`, `ruff check .`, exact commands + real output pasted into the phase report. No claim of hardware success without hardware having been touched.

---

## 9. Known limitations (honest list)

- **Known-size ranging on a paper airplane is coarse.** Apparent wingspan varies ~2–3× with orientation as it banks; expect ±25 % range error, which feeds straight into lead time-of-flight error. This is why `fixed` range mode exists for controlled lead testing, and why ArUco is the ranging *validation* tool. If monocular proves insufficient, the `RangeEstimator` ABC is where a second synced OV9281 (stereo) plugs in — already the agreed fallback plan.
- **Quasi-static gate (v1) means the turret can't detect while slewing fast.** Acceptable for v1 bring-up; removed in Phase 4.5.
- **No drag model.** Fine at 3–6 m; revisit when muzzle velocity is measurable.
- **Servo pitch has no position feedback.** Telemetry reports *commanded* pitch. True for yaw too (open-loop stepper) — telemetry is commanded state, trusted only because loads are sized with margin. A missed-step detection scheme is out of scope for v1.
- **USB serial (CP2102) adds jitter** vs direct UART GPIO. Fine for bench; note for later if telemetry timing gets tight in ego-motion comp.
