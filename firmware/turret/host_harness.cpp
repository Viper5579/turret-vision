// Host-side test harness: compiles the EXACT firmware protocol/control code
// with g++ and exposes it to tests/test_firmware_parity.py, which drives it
// against the Python reference implementation. Never flashed to the ESP32.
//
// Build (the pytest does this automatically):
//   g++ -std=c++17 -O2 -o harness host_harness.cpp protocol.cpp control.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "control.h"
#include "protocol.h"

static std::vector<uint8_t> from_hex(const char* s) {
  std::vector<uint8_t> out;
  size_t n = strlen(s);
  for (size_t i = 0; i + 1 < n; i += 2) {
    unsigned v;
    sscanf(s + i, "%2x", &v);
    out.push_back((uint8_t)v);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::string mode = argv[1];

  if (mode == "crc" && argc == 3) {
    auto data = from_hex(argv[2]);
    printf("%04X\n", tvproto::crc16_ccitt(data.data(), data.size()));
    return 0;
  }

  if (mode == "packtelem" && argc == 7) {
    uint8_t buf[tvproto::TELEM_FRAME_LEN];
    size_t n = tvproto::pack_telem(buf, (uint32_t)strtoul(argv[2], nullptr, 10),
                                   (int16_t)atoi(argv[3]), (int16_t)atoi(argv[4]),
                                   (int16_t)atoi(argv[5]), (uint8_t)atoi(argv[6]));
    for (size_t i = 0; i < n; ++i) printf("%02x", buf[i]);
    printf("\n");
    return 0;
  }

  if (mode == "parse" && argc == 3) {
    auto data = from_hex(argv[2]);
    tvproto::Parser p;
    for (uint8_t b : data) {
      if (p.feed(b) && p.type() == tvproto::TYPE_AIM) {
        tvproto::AimCmd c;
        if (tvproto::unpack_aim(p.payload(), p.payload_len(), c))
          printf("AIM t=%u flags=%u yaw=%d pitch=%d yr=%d pr=%d conf=%u range=%u\n",
                 c.t_ms, c.flags, c.yaw_cdeg, c.pitch_cdeg, c.yaw_rate_cdps,
                 c.pitch_rate_cdps, c.confidence, c.range_mm);
      }
    }
    printf("stats packets=%u crc_errors=%u oversize=%u\n", p.packets, p.crc_errors,
           p.oversize_errors);
    return 0;
  }

  if (mode == "axis" && argc == 8) {
    tvctl::TrapezoidAxis ax((float)atof(argv[2]), (float)atof(argv[3]));
    float dt = (float)atof(argv[4]);
    int n = atoi(argv[5]);
    float target = (float)atof(argv[6]), ff = (float)atof(argv[7]);
    for (int i = 0; i < n; ++i) {
      ax.step(dt, target, ff);
      printf("%.6f %.6f\n", ax.pos, ax.vel);
    }
    return 0;
  }

  fprintf(stderr, "usage: harness crc|packtelem|parse|axis ...\n");
  return 2;
}
