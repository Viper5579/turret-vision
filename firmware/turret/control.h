// Motion control primitives, ported from the behavior model in
// turretvision/link/mock_link.py -- the Python side is what the whole
// pipeline was developed against, so the firmware matches IT, not the other
// way around. tests/test_firmware_parity.py checks this file's trapezoid
// against MockLink's step-for-step on the host.
//
// Portable C++ (no Arduino) so it compiles in the host parity test.
#pragma once

namespace tvctl {

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Trapezoidal profile with rate feedforward: command the velocity that could
// still stop in time (sqrt(2*a*d)) toward the target, capped at vmax,
// approached at amax, PLUS the feedforward rate from the AIM packet.
// WHY the ff term is not optional: position-error-only control chasing a
// target moving at constant v carries a permanent v^2/(2a) lag; the ff term
// removed 10x of tracking error in simulation (see firmware/PROTOCOL.md).
struct TrapezoidAxis {
  float pos = 0.0f;   // deg
  float vel = 0.0f;   // deg/s
  float vmax;         // deg/s
  float amax;         // deg/s^2

  TrapezoidAxis(float vmax_dps, float amax_dps2) : vmax(vmax_dps), amax(amax_dps2) {}

  void step(float dt, float target_deg, float ff_dps) {
    float err = target_deg - pos;
    float mag = err < 0 ? -err : err;
    // sqrtf via Newton iterations would be silly; libm sqrtf is fine on ESP32
    float stoppable = sqrtf_(2.0f * amax * mag);
    float v_des = (err < 0 ? -stoppable : stoppable) + ff_dps;
    v_des = clampf(v_des, -vmax, vmax);
    float dv = clampf(v_des - vel, -amax * dt, amax * dt);
    vel += dv;
    pos += vel * dt;
  }

  void hold() { vel = 0.0f; }

 private:
  // tiny wrapper so this header stays freestanding (no <cmath> include order
  // headaches between Arduino cores and hosts)
  static float sqrtf_(float x);
};

// Rate-limited move (servo pitch model): step toward target at a fixed rate.
inline float rate_limited_step(float pos, float target, float rate_dps, float dt) {
  float step = rate_dps * dt;
  float err = target - pos;
  return pos + clampf(err, -step, step);
}

}  // namespace tvctl
