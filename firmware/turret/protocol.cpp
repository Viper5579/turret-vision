#include "protocol.h"

namespace tvproto {

// Table-based CRC-16/CCITT-FALSE: one lookup per byte. The table costs 512 B
// of flash -- cheap insurance on a link that WILL see corrupt bytes.
static uint16_t crc_table[256];
static bool crc_init_done = false;

static void crc_init() {
  for (uint16_t i = 0; i < 256; ++i) {
    uint16_t crc = i << 8;
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    crc_table[i] = crc;
  }
  crc_init_done = true;
}

uint16_t crc16_ccitt(const uint8_t* data, size_t len) {
  if (!crc_init_done) crc_init();
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i)
    crc = (uint16_t)((crc << 8) ^ crc_table[(uint8_t)((crc >> 8) ^ data[i])]);
  return crc;
}

// Explicit little-endian readers/writers: works identically on any host the
// parity test compiles on, regardless of that host's endianness.
static uint16_t rd_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static void wr_u16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
static void wr_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

bool unpack_aim(const uint8_t* payload, uint8_t len, AimCmd& out) {
  if (len != AIM_PAYLOAD_LEN) return false;
  out.t_ms = rd_u32(payload);
  out.flags = payload[4];
  out.yaw_cdeg = (int16_t)rd_u16(payload + 5);
  out.pitch_cdeg = (int16_t)rd_u16(payload + 7);
  out.yaw_rate_cdps = (int16_t)rd_u16(payload + 9);
  out.pitch_rate_cdps = (int16_t)rd_u16(payload + 11);
  out.confidence = payload[13];
  out.range_mm = rd_u16(payload + 14);
  return true;
}

size_t pack_telem(uint8_t* buf, uint32_t t_ms, int16_t yaw_cdeg, int16_t pitch_cdeg,
                  int16_t yaw_rate_cdps, uint8_t status) {
  buf[0] = SYNC1;
  buf[1] = SYNC2;
  buf[2] = TELEM_PAYLOAD_LEN;
  buf[3] = TYPE_TELEM;
  wr_u32(buf + 4, t_ms);
  wr_u16(buf + 8, (uint16_t)yaw_cdeg);
  wr_u16(buf + 10, (uint16_t)pitch_cdeg);
  wr_u16(buf + 12, (uint16_t)yaw_rate_cdps);
  buf[14] = status;
  uint16_t crc = crc16_ccitt(buf + 2, 2 + TELEM_PAYLOAD_LEN);
  wr_u16(buf + 15, crc);
  return TELEM_FRAME_LEN;
}

bool Parser::feed(uint8_t b) {
  switch (state_) {
    case HUNT1:
      if (b == SYNC1) state_ = HUNT2;
      return false;
    case HUNT2:
      // AA AA 55 must still sync: stay in HUNT2 on a repeated AA
      if (b == SYNC2) state_ = LEN;
      else if (b != SYNC1) state_ = HUNT1;
      return false;
    case LEN:
      if (b > MAX_PAYLOAD) {
        ++oversize_errors;
        state_ = HUNT1;
        return false;
      }
      len_ = b;
      buf_[0] = b;
      got_ = 0;
      state_ = TYPE;
      return false;
    case TYPE:
      type_ = b;
      buf_[1] = b;
      state_ = len_ ? PAYLOAD : CRC1;
      return false;
    case PAYLOAD:
      buf_[2 + got_++] = b;
      if (got_ == len_) state_ = CRC1;
      return false;
    case CRC1:
      crc_lo_ = b;
      state_ = CRC2;
      return false;
    case CRC2: {
      state_ = HUNT1;
      uint16_t rx = (uint16_t)(crc_lo_ | (b << 8));
      if (rx == crc16_ccitt(buf_, (size_t)2 + len_)) {
        ++packets;
        return true;
      }
      ++crc_errors;
      return false;
    }
  }
  return false;
}

}  // namespace tvproto
