// Turret Vision v2 firmware, milestone M0.
//
// The ESP32 owns time and motion; the Jetson owns seeing and deciding.
// This build:
//   * speaks the v2 link protocol (COBS + CRC16, generated messages, hash handshake)
//   * answers clock-sync pings with microsecond timestamps
//   * follows Traj messages on the yaw stepper with hardware step timing (FastAccelStepper)
//   * drives the pitch servo from a 16-bit LEDC channel (no PCA9685)
//   * pulses the camera trigger and reports FrameSync (pose latched at the edge)
//   * safe-stops when Traj messages stop arriving (heartbeat timeout) or on e-stop
//   * reports Telem at 200 Hz
// M1 adds: AS5600 encoder closed loop + missed-step detection, ICM-42688 gyro, trigger
// wiring to the camera. The behavior model the simulator uses is turret/sim/esp32.py;
// change both together.

#include <Arduino.h>
#include <FastAccelStepper.h>
#include <math.h>

#include "framing.h"
#include "messages_gen.h"

// ------------------------------------------------------------------ pins
static constexpr uint8_t PIN_STEP = 16;
static constexpr uint8_t PIN_DIR = 17;
static constexpr uint8_t PIN_SERVO = 25;
static constexpr uint8_t PIN_TRIGGER = 27;  // camera trigger input (pad F); wired in M1
static constexpr uint8_t SERVO_CH = 0;

static constexpr uint16_t FW_VERSION = 1;
static constexpr uint32_t BAUD = 921600;

// status bits (keep in sync with proto/messages.yaml and turret/sim/esp32.py)
static constexpr uint16_t ST_LINKED = 1, ST_CONFIGURED = 2, ST_ESTOP = 4, ST_ARMED = 8, ST_YAW_LIMIT = 16,
                          ST_TIMEOUT = 32;

// ------------------------------------------------------------------ state
static FastAccelStepperEngine engine;
static FastAccelStepper* stepper = nullptr;
static proto::FrameDecoder decoder;

static proto::Config cfg;
static bool configured = false;
static proto::Traj traj;
static bool have_traj = false;
static uint32_t traj_rx_ms = 0;
static uint16_t status = 0;
static float yaw_ref = 0.0f;
static float pitch_cmd = 0.0f;

static hw_timer_t* trig_timer = nullptr;
static volatile uint32_t trig_pending = 0;  // set in the ISR, consumed in loop()
static volatile int64_t trig_t_us = 0;
static volatile int32_t trig_steps = 0;
static uint32_t trig_seq = 0;

// ------------------------------------------------------------------ helpers
template <class M>
static void send(const M& m) {
  uint8_t buf[proto::MAX_ENCODED];
  size_t n = proto::encode_frame(m, buf);
  Serial.write(buf, n);
}

static float steps_to_deg(int32_t steps) { return configured ? steps / cfg.steps_per_deg : 0.0f; }
static int32_t deg_to_steps(float deg) { return static_cast<int32_t>(lroundf(deg * cfg.steps_per_deg)); }

static void servo_write_deg(float deg) {
  // Map the pitch range onto the servo pulse range linearly; 50 Hz frame = 20000 us.
  float lo = cfg.pitch_min_deg, hi = cfg.pitch_max_deg;
  float f = (deg - lo) / (hi - lo);
  f = f < 0 ? 0 : (f > 1 ? 1 : f);
  float us = cfg.servo_us_min + f * (cfg.servo_us_max - cfg.servo_us_min);
  ledcWrite(SERVO_CH, static_cast<uint32_t>(us / 20000.0f * 65535.0f));
}

static void IRAM_ATTR on_trigger_timer() {
  // Rising edge now; the falling edge goes out from loop() (pulse >> 2 us minimum).
  digitalWrite(PIN_TRIGGER, HIGH);
  trig_t_us = esp_timer_get_time();
  trig_steps = stepper ? stepper->getCurrentPosition() : 0;
  trig_pending++;
}

static void set_trigger_rate(float hz) {
  if (trig_timer) {
    timerAlarmDisable(trig_timer);
  }
  if (hz <= 0.0f) return;
  if (!trig_timer) {
    trig_timer = timerBegin(0, 80, true);  // 80 MHz / 80 = 1 us ticks
    timerAttachInterrupt(trig_timer, &on_trigger_timer, true);
  }
  timerAlarmWrite(trig_timer, static_cast<uint64_t>(1e6f / hz), true);
  timerAlarmEnable(trig_timer);
}

// ------------------------------------------------------------------ rx
static void handle_frame() {
  const uint8_t id = decoder.id();
  const uint8_t* p = decoder.payload();
  const size_t n = decoder.payload_len();
  switch (id) {
    case proto::Hello::ID: {
      proto::Hello h;
      if (!proto::Hello::decode(p, n, h)) return;
      status |= ST_LINKED;
      proto::HelloAck ack;
      ack.proto_hash = proto::PROTO_HASH;
      ack.fw_version = FW_VERSION;
      ack.uptime_ms = millis();
      ack.features = 0x4;  // trigger output only in M0 (bit0 encoder, bit1 imu arrive in M1)
      send(ack);
      break;
    }
    case proto::TimePing::ID: {
      proto::TimePing ping;
      int64_t t_rx = esp_timer_get_time();
      if (!proto::TimePing::decode(p, n, ping)) return;
      proto::TimePong pong;
      pong.seq = ping.seq;
      pong.t_rx_us = static_cast<uint64_t>(t_rx);
      pong.t_tx_us = static_cast<uint64_t>(esp_timer_get_time());
      send(pong);
      break;
    }
    case proto::Config::ID: {
      proto::Config c;
      if (!proto::Config::decode(p, n, c) || c.steps_per_deg <= 0.0f) return;
      cfg = c;
      configured = true;
      status |= ST_CONFIGURED;
      stepper->setSpeedInHz(static_cast<uint32_t>(cfg.max_rate_dps * cfg.steps_per_deg));
      stepper->setAcceleration(static_cast<int32_t>(cfg.max_accel_dps2 * cfg.steps_per_deg));
      set_trigger_rate(cfg.trigger_hz);
      break;
    }
    case proto::Traj::ID: {
      proto::Traj t;
      if (!proto::Traj::decode(p, n, t)) return;
      traj = t;
      have_traj = true;
      traj_rx_ms = millis();
      break;
    }
    default:
      break;  // Fire arrives with the safety hardware in M5
  }
}

// ------------------------------------------------------------------ control (1 kHz)
static void control() {
  if (!configured) return;
  const bool timed_out = !have_traj || (millis() - traj_rx_ms) > cfg.timeout_ms;
  const bool estop = have_traj && (traj.flags & 1);
  status = (status & ~(ST_TIMEOUT | ST_ESTOP)) | (timed_out ? ST_TIMEOUT : 0) | (estop ? ST_ESTOP : 0);

  if (timed_out || estop) {
    if (stepper->isRunning()) stepper->stopMove();  // decelerate, then hold
    return;
  }
  const float age_s = (static_cast<int64_t>(esp_timer_get_time()) - static_cast<int64_t>(traj.valid_at_us)) * 1e-6f;
  float ref = traj.yaw_deg + traj.yaw_rate_dps * age_s;
  bool at_limit = false;
  if (ref > cfg.yaw_max_deg) {
    ref = cfg.yaw_max_deg;
    at_limit = true;
  } else if (ref < cfg.yaw_min_deg) {
    ref = cfg.yaw_min_deg;
    at_limit = true;
  }
  status = (status & ~ST_YAW_LIMIT) | (at_limit ? ST_YAW_LIMIT : 0);
  yaw_ref = ref;

  // M0 motion: re-target the hardware ramp generator every millisecond, aiming slightly ahead
  // by the commanded rate (feedforward). The planner respects max rate and acceleration.
  // M1 replaces this with encoder-closed velocity control.
  float target = ref + traj.yaw_rate_dps * cfg.lead_s;
  if (target > cfg.yaw_max_deg) target = cfg.yaw_max_deg;
  if (target < cfg.yaw_min_deg) target = cfg.yaw_min_deg;
  stepper->moveTo(deg_to_steps(target));

  float pitch = traj.pitch_deg + traj.pitch_rate_dps * age_s;
  pitch = pitch < cfg.pitch_min_deg ? cfg.pitch_min_deg : (pitch > cfg.pitch_max_deg ? cfg.pitch_max_deg : pitch);
  if (fabsf(pitch - pitch_cmd) > 0.05f) {
    pitch_cmd = pitch;
    servo_write_deg(pitch_cmd);
  }
}

static void telemetry() {
  proto::Telem t;
  t.t_us = static_cast<uint64_t>(esp_timer_get_time());
  t.yaw_enc_deg = NAN;  // AS5600 arrives in M1
  t.yaw_step_deg = steps_to_deg(stepper->getCurrentPosition());
  t.yaw_rate_dps = configured ? stepper->getCurrentSpeedInMilliHz() / 1000.0f / cfg.steps_per_deg : 0.0f;
  t.yaw_ref_deg = yaw_ref;
  t.pitch_cmd_deg = pitch_cmd;
  t.gyro_z_dps = NAN;  // ICM-42688 arrives in M1
  t.status = status;
  t.faults = 0;
  send(t);
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  Serial.setRxBufferSize(1024);
  Serial.begin(BAUD);
  pinMode(PIN_TRIGGER, OUTPUT);
  digitalWrite(PIN_TRIGGER, LOW);

  engine.init();
  stepper = engine.stepperConnectToPin(PIN_STEP);
  if (stepper) {
    stepper->setDirectionPin(PIN_DIR);
    stepper->setSpeedInHz(500);
    stepper->setAcceleration(2000);
    stepper->setCurrentPosition(0);  // M1: encoder gives the true absolute angle at boot
  }
  ledcSetup(SERVO_CH, 50, 16);
  ledcAttachPin(PIN_SERVO, SERVO_CH);
  cfg.pitch_min_deg = -10.0f;
  cfg.pitch_max_deg = 60.0f;
  cfg.servo_us_min = 500;
  cfg.servo_us_max = 2500;
  servo_write_deg(0.0f);
}

void loop() {
  static uint32_t next_ctrl_us = 0;
  static uint32_t next_telem_us = 0;

  while (Serial.available()) {
    if (decoder.push(static_cast<uint8_t>(Serial.read()))) handle_frame();
  }

  if (trig_pending) {
    noInterrupts();
    trig_pending = 0;
    const int64_t t = trig_t_us;
    const int32_t steps = trig_steps;
    interrupts();
    digitalWrite(PIN_TRIGGER, LOW);  // ends a >= several-us pulse; the camera latches on the rising edge
    proto::FrameSync fs;
    fs.seq = ++trig_seq;
    fs.t_us = static_cast<uint64_t>(t);
    fs.yaw_deg = steps_to_deg(steps);
    fs.yaw_rate_dps = configured ? stepper->getCurrentSpeedInMilliHz() / 1000.0f / cfg.steps_per_deg : 0.0f;
    fs.gyro_yaw_deg = NAN;
    fs.pitch_deg = pitch_cmd;
    send(fs);
  }

  const uint32_t now = micros();
  if (stepper && static_cast<int32_t>(now - next_ctrl_us) >= 0) {
    next_ctrl_us = now + 1000;
    control();
  }
  if (stepper && static_cast<int32_t>(now - next_telem_us) >= 0) {
    next_telem_us = now + 5000;
    telemetry();
  }
}
