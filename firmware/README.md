# ESP32 firmware (reference implementation)

Target: ESP32 DevKit V1 · NEMA 17 yaw via A4988 · pitch servo via PCA9685
(SPEC header). Wire protocol: `PROTOCOL.md` / SPEC §5.

## What's here

| File | Role |
|---|---|
| `turret/protocol.h/.cpp` | Wire protocol: CRC16, framing parser, AIM unpack, TELEM pack. **Portable C++** — the exact code that runs on the ESP32 is compiled on the host by `tests/test_firmware_parity.py` and proven byte-for-byte compatible with the Python side. |
| `turret/control.h/.cpp` | Trapezoid yaw profile (with rate feedforward) + rate-limited pitch, ported from `turretvision/link/mock_link.py` — parity-tested step-for-step against it. |
| `turret/turret.ino` | Arduino glue: serial RX → parser → clamp/e-stop/watchdog, 500 Hz motion tick (AccelStepper + PCA9685), homing on boot, 100 Hz telemetry. |
| `turret/host_harness.cpp` | Host-only CLI the parity test drives. Never flashed. |

## Before flashing

1. **Edit the config block** at the top of `turret.ino`: pins, `STEPS_PER_DEG`
   (gearing!), travel limits, servo pulse calibration.
2. Install libraries: `AccelStepper`, `Adafruit PWM Servo Driver Library`
   (Arduino IDE Library Manager, or PlatformIO `lib_deps`).
3. Board: "ESP32 Dev Module", flash over USB.

## Verify (in this order — each step needs no more hardware than listed)

1. **No hardware at all**: `python -m pytest tests/test_firmware_parity.py -v`
   compiles the protocol/control code on this machine and checks parity with
   Python. Green = the firmware speaks the wire format, guaranteed.
2. **Board only, nothing mechanical**: flash, then on the Jetson run
   `python tools/link_monitor.py /dev/ttyUSB0` — TELEM lines at ~100 Hz
   (homed=0 is fine with no home switch). Then set `link.backend: serial` and
   run the pipeline: the ESP32's parser should accept AIM frames (its counters
   never appear on the wire, but motion targets change — see step 3).
3. **Motors wired, turret NOT assembled** (motors loose on the bench):
   SPEC §6.4 actuator-mapping check — command +30.00°, read telemetry back,
   confirm the shaft actually moved 30°. This catches steps/deg and direction
   errors while they're still cheap.
4. **E-stop and watchdog**: unplug USB mid-motion → motion must stop within
   `AIM_TIMEOUT_MS` (safe-hold). That behavior is the contract MockLink and
   the tests defined.

## Design notes

- Setpoints are absolute and clamped on BOTH ends of the wire (SPEC D2).
- The rate feedforward is implemented, not just parsed — it removed ~10x of
  tracking lag in simulation (`PROTOCOL.md`). Keep it when you refactor.
- Telemetry flows from boot, unconditionally. The Jetson's ego-motion
  compensation and absolute-setpoint math starve without it.
