// Link framing for the ESP32: id + payload + CRC16, COBS encoded, 0x00 delimited.
// Twin of turret/link/{crc,cobs,framing}.py. Pure C++, no Arduino headers, so the
// same code is unit-tested on a laptop (`pio test -e native`).
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace proto {

static constexpr size_t MAX_PAYLOAD = 64;
static constexpr size_t MAX_BODY = 1 + MAX_PAYLOAD + 2;               // id + payload + crc
static constexpr size_t MAX_ENCODED = MAX_BODY + MAX_BODY / 254 + 2;  // + code bytes + 0x00

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
inline uint16_t crc16(const uint8_t* data, size_t n, uint16_t crc = 0xFFFF) {
  for (size_t i = 0; i < n; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (int k = 0; k < 8; ++k) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

// COBS encode `n` bytes into `out` (no trailing delimiter). Returns encoded length.
inline size_t cobs_encode(const uint8_t* in, size_t n, uint8_t* out) {
  size_t code_idx = 0;
  size_t o = 1;
  uint8_t code = 1;
  for (size_t i = 0; i < n; ++i) {
    if (in[i] == 0) {
      out[code_idx] = code;
      code_idx = o++;
      code = 1;
    } else {
      out[o++] = in[i];
      if (++code == 0xFF) {
        out[code_idx] = code;
        code_idx = o++;
        code = 1;
      }
    }
  }
  out[code_idx] = code;
  return o;
}

// COBS decode one block (without delimiter). Returns decoded length, or 0 on error.
inline size_t cobs_decode(const uint8_t* in, size_t n, uint8_t* out, size_t cap) {
  size_t i = 0, o = 0;
  while (i < n) {
    uint8_t code = in[i++];
    if (code == 0) return 0;
    size_t end = i + code - 1;
    if (end > n || o + (code - 1) > cap) return 0;
    while (i < end) out[o++] = in[i++];
    if (code < 0xFF && i < n) {
      if (o >= cap) return 0;
      out[o++] = 0;
    }
  }
  return o;
}

// Encode a generated message struct into a complete wire frame (with 0x00).
template <class M>
size_t encode_frame(const M& m, uint8_t* out) {
  uint8_t body[MAX_BODY];
  body[0] = M::ID;
  size_t n = 1 + m.encode(body + 1);
  uint16_t crc = crc16(body, n);
  body[n++] = static_cast<uint8_t>(crc & 0xFF);
  body[n++] = static_cast<uint8_t>(crc >> 8);
  size_t e = cobs_encode(body, n, out);
  out[e++] = 0;
  return e;
}

// Byte-at-a-time receiver. push() returns true when a valid frame completed;
// then id()/payload()/payload_len() describe it until the next push().
class FrameDecoder {
 public:
  bool push(uint8_t b) {
    if (b != 0) {
      if (len_ < sizeof(raw_)) {
        raw_[len_++] = b;
      } else {
        overflow_ = true;
      }
      return false;
    }
    size_t n = len_;
    bool overflow = overflow_;
    len_ = 0;
    overflow_ = false;
    if (n == 0) return false;
    if (overflow) {
      ++oversize;
      return false;
    }
    size_t d = cobs_decode(raw_, n, body_, sizeof(body_));
    if (d < 3) {
      ++cobs_errors;
      return false;
    }
    uint16_t want = static_cast<uint16_t>(body_[d - 2]) | (static_cast<uint16_t>(body_[d - 1]) << 8);
    if (crc16(body_, d - 2) != want) {
      ++crc_errors;
      return false;
    }
    body_len_ = d - 2;
    ++frames;
    return true;
  }

  uint8_t id() const { return body_[0]; }
  const uint8_t* payload() const { return body_ + 1; }
  size_t payload_len() const { return body_len_ - 1; }

  uint32_t frames = 0;
  uint32_t crc_errors = 0;
  uint32_t cobs_errors = 0;
  uint32_t oversize = 0;

 private:
  uint8_t raw_[MAX_ENCODED];
  uint8_t body_[MAX_BODY + 2];
  size_t len_ = 0;
  size_t body_len_ = 0;
  bool overflow_ = false;
};

}  // namespace proto
