// Native unit tests for the link protocol: `pio test -e native`.
// The golden tests decode frames produced by the Python encoder, which is
// what proves both sides agree byte for byte.
#include <unity.h>

#include "framing.h"
#include "golden.h"
#include "messages_gen.h"

void setUp() {}
void tearDown() {}

static void test_crc_check_value() {
  // Standard check value for CRC-16/CCITT-FALSE.
  const uint8_t s[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  TEST_ASSERT_EQUAL_HEX16(0x29B1, proto::crc16(s, sizeof(s)));
}

static void roundtrip(const uint8_t* in, size_t n) {
  uint8_t enc[600];
  uint8_t dec[600];
  size_t e = proto::cobs_encode(in, n, enc);
  for (size_t i = 0; i < e; ++i) TEST_ASSERT_NOT_EQUAL(0, enc[i]);
  size_t d = proto::cobs_decode(enc, e, dec, sizeof(dec));
  TEST_ASSERT_EQUAL_UINT32(n, d);
  if (n) TEST_ASSERT_EQUAL_MEMORY(in, dec, n);
}

static void test_cobs_edge_cases() {
  const uint8_t one_zero[] = {0};
  const uint8_t zeros[] = {0, 0, 0};
  const uint8_t mixed[] = {0x11, 0, 0x22, 0x33, 0};
  roundtrip(one_zero, sizeof(one_zero));
  roundtrip(zeros, sizeof(zeros));
  roundtrip(mixed, sizeof(mixed));
  uint8_t run[520];
  for (size_t i = 0; i < sizeof(run); ++i) run[i] = static_cast<uint8_t>(i % 255 + 1);
  roundtrip(run, 254);
  roundtrip(run, 255);
  roundtrip(run, 520);
}

static void test_decoder_resyncs_after_garbage() {
  proto::Traj t;
  t.valid_at_us = 123456789;
  t.yaw_deg = 12.5f;
  t.mode = 2;
  uint8_t frame[proto::MAX_ENCODED];
  size_t n = proto::encode_frame(t, frame);

  proto::FrameDecoder dec;
  const uint8_t junk[] = {0x55, 0x13, 0x77, 0x00, 0x09, 0x00};
  int got = 0;
  for (uint8_t b : junk) got += dec.push(b);
  for (size_t i = 0; i < n; ++i) got += dec.push(frame[i]);
  TEST_ASSERT_EQUAL_INT(1, got);
  TEST_ASSERT_EQUAL_UINT8(proto::Traj::ID, dec.id());
  proto::Traj back;
  TEST_ASSERT_TRUE(proto::Traj::decode(dec.payload(), dec.payload_len(), back));
  TEST_ASSERT_EQUAL_UINT64(t.valid_at_us, back.valid_at_us);
  TEST_ASSERT_EQUAL_FLOAT(12.5f, back.yaw_deg);
  TEST_ASSERT_TRUE(dec.cobs_errors + dec.crc_errors >= 1);
}

static void test_decoder_rejects_bit_flip() {
  proto::Telem m;
  m.t_us = 42;
  uint8_t frame[proto::MAX_ENCODED];
  size_t n = proto::encode_frame(m, frame);
  frame[3] ^= 0x04;
  proto::FrameDecoder dec;
  int got = 0;
  for (size_t i = 0; i < n; ++i) got += dec.push(frame[i]);
  TEST_ASSERT_EQUAL_INT(0, got);
}

static void test_wrong_size_payload_is_rejected() {
  uint8_t short_payload[3] = {1, 2, 3};
  proto::Traj t;
  TEST_ASSERT_FALSE(proto::Traj::decode(short_payload, sizeof(short_payload), t));
}

int main(int, char**) {
  UNITY_BEGIN();
  RUN_TEST(test_crc_check_value);
  RUN_TEST(test_cobs_edge_cases);
  RUN_TEST(test_decoder_resyncs_after_garbage);
  RUN_TEST(test_decoder_rejects_bit_flip);
  RUN_TEST(test_wrong_size_payload_is_rejected);
  RUN_GOLDEN_TESTS();
  return UNITY_END();
}
