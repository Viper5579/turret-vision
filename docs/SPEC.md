# Turret Vision v2: locked decisions

Every decision below has its reason next to it, so the reasoning survives even when the
code is rewritten. The milestone plan and schedule live in the rebuild plan document
(linked from the README); this file is what the code is held to.

## Conventions

- **Base frame:** X forward (yaw 0), Y left, Z up, origin on the yaw axis. Yaw is positive
  to the left (counter-clockwise from above). Azimuth = atan2(Y, X), elevation =
  atan2(Z, horizontal range).
- **Camera frame:** OpenCV (x right, y down, z forward). Pixel centers on integers, so the
  image center is ((w-1)/2, (h-1)/2).
- **Units:** degrees and degrees/second on the wire and in the GUI; radians only inside
  math. ESP32 time in microseconds since boot; Jetson time in monotonic seconds.

## Decisions

**D1. The camera rides the yaw stage and never pitches.** Camera motion is then a single
measured angle and the background moves by one exact homography, H = K R_now^T R_then K^-1.
Pitch servo jitter never shakes the image. Cost: vertical coverage of the 47 deg vertical
FOV, so the camera is tilted up 10 deg.

**D2. The ESP32 triggers every exposure.** Each frame then has an exact capture time and
pose in one clock, instead of a USB arrival time that jitters by milliseconds. Frames are
matched to triggers by sequence number once the offset is locked (`runtime/frames.py`).
Fallback if trigger mode fails on hardware: free-run plus gyro/optical-flow correlation.

**D3. Detection works while slewing, using measured motion.** Two complementary tests
(`vision/motion.py`):
- Background subtraction against a live 360 deg panorama of the room (`vision/mosaic.py`).
  Possible only because of D1. Gives the whole silhouette, so the center is unbiased, and
  still sees a target that hovers.
- Ego-motion-compensated three-frame differencing with a 3 x 3 neighborhood tolerance,
  used where the panorama has not seen the room yet.

Measured in the simulator: three-frame differencing alone put the centroid one frame of
motion behind the plane (0.96 deg), because the overlap of the two previous positions
survives the AND. The panorama cut median bearing error from 1.29 to 0.21 deg.

**D4. Learned detector (M4): an Ultralytics nano pose model on stacked grayscale frames
t, t-1, t-2,** already ego-motion warped, compiled to TensorRT FP16, run on a crop around
the prediction while tracking. Stacked frames let the network see motion; keypoints (nose,
wingtips, tail) give range by solvePnP.

**D5. Tracking.** M0 ships a constant-velocity Kalman filter on bearing
(`track/bearing.py`) so the loop closes. M3 replaces it with a 3D IMM filter (glide,
straight, maneuver models) in the base frame with a glide-speed pseudo-measurement for
range, behind the same interface.

**D6. Motion control.** The Jetson streams absolute angle + rate valid at an ESP32 time
(`Traj`); the ESP32 extrapolates between messages. Absolute setpoints are idempotent (a
lost packet causes no lasting error) and the rate term cut simulated lag about 10x in v1.
Steps come from hardware timers (FastAccelStepper), not from loop(). M1 closes the loop
on the AS5600 encoder and detects missed steps.

**D7. One protocol schema.** `proto/messages.yaml` generates the Python dataclasses, the
C++ structs, and golden byte vectors that the C++ tests decode. Framing is COBS with a
CRC-16/CCITT-FALSE. The CRC32 of the schema is the protocol hash; the Jetson refuses
firmware with a different hash and says to reflash.

**D8. Clock sync is NTP-style with minimum-delay filtering.** USB delay only adds jitter,
so the lowest round trips are trusted and a line (offset plus drift) is fitted through them.
Target error under 0.5 ms.

**D9. Runs in one container, built by CI on ARM64, pulled by the Jetson.** The Jetson's
own Python is never installed into (v1's pip replaced JetPack's OpenCV and numpy). Through
M3 the image is plain Ubuntu 22.04 (Python 3.10, matches JetPack 6), because nothing needs
CUDA yet: vision runs on the CPU and the GUI forwards the camera's own JPEGs. M4 switches
the base to `l4t-jetpack:r36.4.0` for TensorRT, once the NVMe drive has room for it.
Python versions are pinned in `requirements.lock`, resolved for aarch64 / Python 3.10.

**D10. JPEG decode at half resolution in the DCT domain** (`cv2.IMREAD_REDUCED_GRAYSCALE_2`).
The motion stage works at 640 x 400; decoding straight to that size skips most of the work.
No GStreamer or hardware decoder dependency.

**D11. GUI video costs no encoding.** The Orin Nano has no hardware video encoder. The
server forwards the camera's JPEG bytes; overlays travel separately with the frame number
and are drawn by the browser.

**D12. Firmware platform.** PlatformIO `espressif32` (Arduino core 2.x, ESP-IDF 4.4).
FastAccelStepper supports it with hardware step generation; Arduino core 3.0.x does not
compile it. FastAccelStepper is pinned to tag v0.33.9.

**D13. Keep the A4988; drop the PCA9685.** No measured reason to change the stepper driver
(the M1 gyro will show microstep vibration if it exists). The ESP32's own 16-bit PWM is
about 15x finer than the PCA9685's 12-bit at 50 Hz.

**D14. Safety.** On campus the turret emits only a Class 2 laser (OSU policy bans
airsoft-type guns). Firing will require a physical arm switch, a hardware e-stop, Engage
mode, a fresh heartbeat, and no person near the aim line. The ESP32 safe-stops when Traj
messages stop for `timeout_ms` (200 ms).

## Interfaces that must not drift

| Boundary | Source of truth | Checked by |
| --- | --- | --- |
| Jetson to ESP32 bytes | `proto/messages.yaml` | `gen_proto.py --check` in CI; golden C++ tests |
| ESP32 behavior | `firmware/src/main.cpp` | `turret/sim/esp32.py` must model it; change both together |
| Python versions | `requirements.lock` | CI resolves against it; image installs from it |
| Tracking quality | `turret sim --check` | CI gate: coverage > 0.9, median bearing error < 0.5 deg, no false tracks |
