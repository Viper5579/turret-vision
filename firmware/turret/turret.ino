// turret-vision ESP32 firmware (SPEC 5 + firmware/PROTOCOL.md).
//
// Wiring assumptions (EDIT THE CONFIG BLOCK, then verify with the SPEC 6.4
// actuator-mapping check: command +30.00 deg, read telemetry back, confirm):
//   - NEMA 17 yaw stepper via A4988: STEP/DIR/EN pins below
//   - pitch servo via PCA9685 on I2C (default SDA 21 / SCL 22, addr 0x40)
//   - yaw home switch (endstop or index mark), normally-open to GND
//   - USB-CDC serial to the Jetson @ 115200 (the same port used to flash)
//
// Libraries (Arduino IDE Library Manager / platformio lib_deps):
//   - AccelStepper
//   - Adafruit PWM Servo Driver Library
//
// Behavior contract (mirrors turretvision/link/mock_link.py, which the whole
// vision pipeline was developed against):
//   - clamp every setpoint to travel limits (never trust the wire)
//   - e-stop bit: stop motion, ignore setpoints until it clears
//   - no valid AIM frame for AIM_TIMEOUT_MS -> safe-hold (stop where you are);
//     target_valid=0 heartbeats DO reset the timeout (link alive, no target)
//   - USE the rate feedforward (10x tracking-lag reduction, PROTOCOL.md)
//   - TELEM at ~100 Hz, always, from boot

#include <Wire.h>

#include <AccelStepper.h>
#include <Adafruit_PWMServoDriver.h>

#include "control.h"
#include "protocol.h"

// ---------------- configuration (the ONLY block you should need to edit) ---
static const int PIN_YAW_STEP = 26;
static const int PIN_YAW_DIR = 27;
static const int PIN_YAW_EN = 25;      // A4988 EN is active-low
static const int PIN_YAW_HOME = 33;    // NO switch to GND, INPUT_PULLUP
static const int SERVO_CHANNEL = 0;    // PCA9685 channel for pitch

static const float STEPS_PER_DEG = 8.888889f;  // 1.8deg motor, 1/16 microstep,
                                               // direct drive: 3200/360. EDIT
                                               // for your gearing/belt ratio.
static const float YAW_MIN_DEG = -170.0f, YAW_MAX_DEG = 170.0f;
static const float PITCH_MIN_DEG = -10.0f, PITCH_MAX_DEG = 60.0f;
static const float YAW_VMAX_DPS = 240.0f, YAW_AMAX_DPS2 = 900.0f;
static const float PITCH_RATE_DPS = 180.0f;
// Servo pulse calibration: microseconds at the pitch travel extremes. Measure
// on YOUR servo -- datasheet values are a starting point, not truth.
static const int SERVO_US_AT_MIN = 1000, SERVO_US_AT_MAX = 2000;
static const uint32_t AIM_TIMEOUT_MS = 500;   // link.telemetry_timeout_s
static const uint32_t TELEM_PERIOD_MS = 10;   // ~100 Hz
// ---------------------------------------------------------------------------

AccelStepper yaw_stepper(AccelStepper::DRIVER, PIN_YAW_STEP, PIN_YAW_DIR);
Adafruit_PWMServoDriver pwm;
tvproto::Parser parser;
tvctl::TrapezoidAxis yaw_model(YAW_VMAX_DPS, YAW_AMAX_DPS2);

static float yaw_target_deg = 0, pitch_target_deg = 0, pitch_pos_deg = 0;
static float yaw_ff_dps = 0;
static bool estopped = false, homed = false, safe_hold = false;
static uint32_t last_aim_ms = 0, last_telem_ms = 0, last_ctl_us = 0;

static void write_pitch_servo(float deg) {
  float a = (deg - PITCH_MIN_DEG) / (PITCH_MAX_DEG - PITCH_MIN_DEG);
  int us = SERVO_US_AT_MIN + (int)(a * (SERVO_US_AT_MAX - SERVO_US_AT_MIN));
  pwm.writeMicroseconds(SERVO_CHANNEL, us);
}

static void handle_aim(const tvproto::AimCmd& cmd) {
  last_aim_ms = millis();
  safe_hold = false;
  estopped = cmd.estop();
  if (estopped) return;               // stop NOW; setpoints ignored until clear
  if (cmd.target_valid()) {
    // clamp to travel limits: never trust the wire (SPEC D2)
    yaw_target_deg = tvctl::clampf(cmd.yaw_cdeg / 100.0f, YAW_MIN_DEG, YAW_MAX_DEG);
    pitch_target_deg =
        tvctl::clampf(cmd.pitch_cdeg / 100.0f, PITCH_MIN_DEG, PITCH_MAX_DEG);
    yaw_ff_dps = cmd.yaw_rate_cdps / 100.0f;
  } else {
    yaw_ff_dps = 0;                   // heartbeat: hold setpoints, drop the ff
  }
}

static void send_telem() {
  float yaw_deg = yaw_stepper.currentPosition() / STEPS_PER_DEG;
  float yaw_rate = yaw_stepper.speed() / STEPS_PER_DEG;
  uint8_t status = 0;
  if (homed) status |= tvproto::STATUS_HOMED;
  if (yaw_deg <= YAW_MIN_DEG + 0.05f || yaw_deg >= YAW_MAX_DEG - 0.05f)
    status |= tvproto::STATUS_YAW_AT_LIMIT;
  if (pitch_pos_deg <= PITCH_MIN_DEG + 0.05f || pitch_pos_deg >= PITCH_MAX_DEG - 0.05f)
    status |= tvproto::STATUS_PITCH_AT_LIMIT;
  if (estopped) status |= tvproto::STATUS_ESTOPPED;
  uint8_t frame[tvproto::TELEM_FRAME_LEN];
  size_t n = tvproto::pack_telem(frame, millis(), (int16_t)(yaw_deg * 100.0f),
                                 (int16_t)(pitch_pos_deg * 100.0f),
                                 (int16_t)(yaw_rate * 100.0f), status);
  Serial.write(frame, n);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_YAW_EN, OUTPUT);
  digitalWrite(PIN_YAW_EN, LOW);      // enable driver (active-low)
  pinMode(PIN_YAW_HOME, INPUT_PULLUP);
  yaw_stepper.setMaxSpeed(YAW_VMAX_DPS * STEPS_PER_DEG);
  yaw_stepper.setAcceleration(YAW_AMAX_DPS2 * STEPS_PER_DEG);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);                 // standard servo frame rate
  pitch_pos_deg = 0;
  write_pitch_servo(pitch_pos_deg);

  // Homing: slow-seek the yaw switch, zero the step counter there. Until this
  // completes, telemetry reports homed=0 and the Jetson side knows the pose
  // is untrusted. TODO: pick seek direction/offset for your mechanism.
  yaw_stepper.setSpeed(-20.0f * STEPS_PER_DEG);
  uint32_t t0 = millis();
  while (digitalRead(PIN_YAW_HOME) == HIGH && millis() - t0 < 20000)
    yaw_stepper.runSpeed();
  if (digitalRead(PIN_YAW_HOME) == LOW) {
    yaw_stepper.setCurrentPosition((long)(YAW_MIN_DEG * STEPS_PER_DEG));
    homed = true;
  }  // else: unhomed but functional; status bit tells the story

  last_aim_ms = millis();
  last_ctl_us = micros();
}

void loop() {
  // 1. RX: byte-at-a-time, never assumes alignment (SPEC 5)
  while (Serial.available()) {
    if (parser.feed((uint8_t)Serial.read()) && parser.type() == tvproto::TYPE_AIM) {
      tvproto::AimCmd cmd;
      if (tvproto::unpack_aim(parser.payload(), parser.payload_len(), cmd))
        handle_aim(cmd);
    }
  }

  // 2. link-dead watchdog: heartbeats normally keep this from ever firing
  if (!safe_hold && millis() - last_aim_ms > AIM_TIMEOUT_MS) {
    safe_hold = true;
    yaw_target_deg = yaw_stepper.currentPosition() / STEPS_PER_DEG;
    pitch_target_deg = pitch_pos_deg;
    yaw_ff_dps = 0;
  }

  // 3. motion: trapezoid model generates the commanded trajectory; the
  //    stepper follows it position-mode so step timing stays glitch-free
  uint32_t now_us = micros();
  float dt = (now_us - last_ctl_us) * 1e-6f;
  if (dt >= 0.002f) {                  // ~500 Hz control tick
    last_ctl_us = now_us;
    if (estopped) {
      yaw_model.pos = yaw_stepper.currentPosition() / STEPS_PER_DEG;
      yaw_model.hold();
    } else {
      yaw_model.step(dt, yaw_target_deg, safe_hold ? 0 : yaw_ff_dps);
      pitch_pos_deg =
          tvctl::rate_limited_step(pitch_pos_deg, pitch_target_deg, PITCH_RATE_DPS, dt);
      write_pitch_servo(pitch_pos_deg);
    }
    yaw_stepper.moveTo((long)(yaw_model.pos * STEPS_PER_DEG));
  }
  yaw_stepper.run();

  // 4. TELEM at ~100 Hz, always -- the Jetson's ego-motion comp eats this
  if (millis() - last_telem_ms >= TELEM_PERIOD_MS) {
    last_telem_ms = millis();
    send_telem();
  }
}
