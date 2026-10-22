# Turret Vision

A camera turret that finds, tracks, and (eventually) shoots down a thrown paper airplane.
A Jetson Orin Nano Super sees and decides; an ESP32 owns timing and motion.

This is v2, a rebuild around one idea v1 never had: **the turret knows exactly where it
was pointing at the instant each frame was captured.** The ESP32 fires the camera shutter
and latches the turret's pose at that edge, so camera motion can be cancelled exactly and
the vision works while the turret slews. v1 is preserved in git history (tag `v1-legacy`).

**Status: milestone M0 (foundations) complete.** The full loop runs end to end against a
physics simulator, and the ESP32 firmware drives the real turret from the GUI. Camera
capture on hardware arrives in M1. The plan and schedule live in
[the rebuild plan](https://claude.ai/artifact/PLwiF1D2a6LmXesiGWB18u); locked decisions
and their reasons are in [docs/SPEC.md](docs/SPEC.md).

## What is in here

| Piece | What it does |
| --- | --- |
| `turret/vision/mosaic.py` | Live 360 degree panorama of the room. Because the camera only rotates, the expected background for any heading is one image remap away, so the plane is found by background subtraction even while the turret pans. |
| `turret/vision/motion.py` | Training-free detector: mosaic subtraction where the room is known, ego-motion-compensated three-frame differencing where it is not. |
| `turret/link/` | Jetson to ESP32 protocol: COBS framing, CRC-16, clock sync, version handshake. Python and C++ are generated from one schema (`proto/messages.yaml`). |
| `turret/runtime/frames.py` | Matches every camera frame to the trigger pulse that exposed it, surviving dropped frames and jitter. |
| `turret/sim/` | Paper airplane flight physics (glide, phugoid bobbing, banked turns), a ray-cast room, a rendered camera with motion blur, and a simulated ESP32 speaking the real wire protocol. |
| `turret/server/`, `ui/` | Mission Control: a browser GUI (React + TypeScript) with live video and overlays, a turret dial, telemetry charts, aim pad, keyboard and Xbox controller support, and a system check page. |
| `firmware/` | ESP32 firmware (PlatformIO): hardware-timed stepping, servo, camera trigger, telemetry, heartbeat safe-stop. |
| `turret/doctor.py` | Checks the machine, camera modes, ESP32 link, and compute budget, and prints the fix for each failure. |

## Measured so far (simulator, 15 s of throws, turret in Track mode)

| Metric | Result |
| --- | --- |
| Frames matched to their trigger | 1348 of 1348 |
| Frame rate | 90 fps (camera trigger rate) |
| Clock sync error vs simulated ESP32 | under 1 microsecond (sim); hardware target under 0.5 ms |
| Plane under a confirmed track while in view | 97.7% of frames |
| Bearing error, median / 95th percentile | 0.21 deg / 1.2 deg |
| Detection while panning at 60 deg/s | 197 of 199 frames with the plane in view; 0 false detections in 495 frames without it |
| False tracks | 0 |

The motion detector's bearing error was 1.3 deg before the panorama background model
(three-frame differencing leaves a ghost behind a slow target); the fix is described in
`turret/vision/mosaic.py`.

## Run it on the Jetson

One-time host setup (JetPack 6.2.1 already flashed):

```bash
sudo nvpmodel -m 2                       # MAXN_SUPER
sudo usermod -aG docker,dialout $USER    # then log out and back in
git clone https://github.com/Viper5579/turret-vision && cd turret-vision
./tv doctor                              # checks everything; fix what it flags
```

Then:

```bash
./tv flash      # once, with the ESP32 on USB: builds and flashes the firmware
./tv up         # ESP32 plugged in: drive the real turret. Not plugged in: simulator
```

Open `http://<jetson-ip>:8080` from your laptop or phone. `./tv help` lists everything else.

Nothing is installed into the Jetson's own Python. The system runs in a container that CI
builds and tests on ARM64, so what passed CI is exactly what runs on the Jetson.

## Develop on a laptop

```bash
pip install -c requirements.lock -e ".[dev]"
python -m turret serve          # simulator + GUI at http://localhost:8080
python -m pytest                # 30 tests
python -m turret sim --check    # scored simulator run (the CI gate)
cd ui && npm ci && npm run dev  # GUI with hot reload, proxied to the running server
```

After editing `proto/messages.yaml`, run `python tools/gen_proto.py`.

## Hardware

Jetson Orin Nano Super, Arducam OV9782 global shutter USB camera (90 fps triggered),
ESP32 DevKit V1, yaw stepper on an A4988, MG996R pitch servo. M1 adds an AS5600 magnetic
encoder and an ICM-42688-P gyro. Pins: STEP 16, DIR 17, servo 25, camera trigger 27.

## Safety

Nothing that fires a projectile is used on the OSU campus (university policy bans
airsoft-type guns). On campus the turret points a Class 2 laser. A hardware e-stop and a
physical arm switch are required before anything emits (milestone M5).
