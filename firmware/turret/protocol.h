// Wire protocol for the turret link (SPEC 5). Ported from the reference
// implementation turretvision/link/protocol.py -- the constants and byte
// layout MUST match it exactly; tests/test_firmware_parity.py compiles this
// file on the host and proves byte-for-byte parity against the Python side.
//
// Portable C++ (no Arduino, no STL, no dynamic allocation) so the exact code
// that runs on the ESP32 is the code the host parity test exercises.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace tvproto {

constexpr uint8_t SYNC1 = 0xAA, SYNC2 = 0x55;
constexpr uint8_t TYPE_AIM = 0x01;    // Jetson -> ESP32
constexpr uint8_t TYPE_TELEM = 0x02;  // ESP32 -> Jetson
constexpr uint8_t MAX_PAYLOAD = 32;

// AIM flags
constexpr uint8_t FLAG_TARGET_VALID = 0x01;
constexpr uint8_t FLAG_ESTOP = 0x02;  // in EVERY packet, never a one-shot
constexpr uint8_t MODE_SHIFT = 2;
constexpr uint8_t MODE_MASK = 0x03;   // 0 idle, 1 track, 2 manual

// TELEM status bits
constexpr uint8_t STATUS_HOMED = 0x01;
constexpr uint8_t STATUS_YAW_AT_LIMIT = 0x02;
constexpr uint8_t STATUS_PITCH_AT_LIMIT = 0x04;
constexpr uint8_t STATUS_FAULT = 0x08;
constexpr uint8_t STATUS_ESTOPPED = 0x10;

constexpr uint8_t AIM_PAYLOAD_LEN = 16;
constexpr uint8_t TELEM_PAYLOAD_LEN = 11;
constexpr size_t TELEM_FRAME_LEN = 2 + 2 + TELEM_PAYLOAD_LEN + 2;  // 17

// CRC-16/CCITT-FALSE over LEN..PAYLOAD. Check value: crc("123456789")=0x29B1.
uint16_t crc16_ccitt(const uint8_t* data, size_t len);

struct AimCmd {
  uint32_t t_ms;
  uint8_t flags;
  int16_t yaw_cdeg, pitch_cdeg;          // ABSOLUTE setpoints, centidegrees
  int16_t yaw_rate_cdps, pitch_rate_cdps;  // feedforward -- USE it (PROTOCOL.md)
  uint8_t confidence;                    // 0..255
  uint16_t range_mm;                     // 0 = unknown

  bool target_valid() const { return flags & FLAG_TARGET_VALID; }
  bool estop() const { return flags & FLAG_ESTOP; }
  uint8_t mode() const { return (flags >> MODE_SHIFT) & MODE_MASK; }
};

// Decode a validated AIM payload (from Parser). False if len is wrong.
bool unpack_aim(const uint8_t* payload, uint8_t len, AimCmd& out);

// Build a complete TELEM frame (sync..crc) into buf (>= TELEM_FRAME_LEN).
// Returns the frame length.
size_t pack_telem(uint8_t* buf, uint32_t t_ms, int16_t yaw_cdeg, int16_t pitch_cdeg,
                  int16_t yaw_rate_cdps, uint8_t status);

// Byte-at-a-time framing state machine (SPEC 5: never assume a read aligns
// to a packet). feed() returns true when a CRC-validated frame completed;
// type()/payload() are then valid until the next feed().
class Parser {
 public:
  bool feed(uint8_t b);
  uint8_t type() const { return type_; }
  const uint8_t* payload() const { return buf_ + 2; }
  uint8_t payload_len() const { return len_; }

  uint32_t crc_errors = 0;
  uint32_t oversize_errors = 0;
  uint32_t packets = 0;

 private:
  enum State : uint8_t { HUNT1, HUNT2, LEN, TYPE, PAYLOAD, CRC1, CRC2 };
  State state_ = HUNT1;
  uint8_t buf_[2 + MAX_PAYLOAD];  // LEN, TYPE, payload (CRC input span)
  uint8_t len_ = 0, type_ = 0, got_ = 0, crc_lo_ = 0;
};

}  // namespace tvproto
