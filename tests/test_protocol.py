"""Wire protocol: CRC vector, byte-exact packing (Phase 3 exit criterion),
round-trips, parser resync behavior, and the SPEC 8 fuzz requirement
(0 CRC-accepted corrupt packets)."""
import random
import struct

from turretvision.link import protocol
from turretvision.link.protocol import (
    AimPacket,
    Parser,
    TelemPacket,
    crc16_ccitt,
    pack_aim,
    pack_telem,
    unpack_aim,
)


def crc16_bitwise(data: bytes) -> int:
    """Independent CRC-16/CCITT-FALSE (bit-by-bit) to cross-check the table
    implementation — a shared bug in one algorithm can't hide in the other."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
    return crc


AIM = AimPacket(t_ms=0x01020304, target_valid=True, estop=False, mode=1,
                yaw_deg=12.34, pitch_deg=-5.0, yaw_rate_dps=100.0,
                pitch_rate_dps=-100.0, confidence=1.0, range_mm=4000)


def test_crc16_check_value():
    # The published check value for CRC-16/CCITT-FALSE
    assert crc16_ccitt(b"123456789") == 0x29B1
    assert crc16_bitwise(b"123456789") == 0x29B1


def test_aim_packs_byte_exact():
    wire = pack_aim(AIM)
    # header: sync, LEN=16, TYPE=0x01
    assert wire[:4] == bytes((0xAA, 0x55, 16, 0x01))
    # payload, field by field, little-endian, centidegrees:
    # t_ms u32 | flags(valid|mode=1)=0x05 | 1234 | -500 | 10000 | -10000 | 255 | 4000
    expected_payload = struct.pack("<IBhhhhBH",
                                   0x01020304, 0x05, 1234, -500, 10000, -10000, 255, 4000)
    assert wire[4:-2] == expected_payload
    # CRC over LEN..payload, verified against the independent implementation
    assert struct.unpack("<H", wire[-2:])[0] == crc16_bitwise(wire[2:-2])
    assert len(wire) == 22  # SPEC 5: 16B payload -> 22B frame


def test_telem_packs_byte_exact():
    wire = pack_telem(TelemPacket(t_ms=1000, yaw_deg=90.0, pitch_deg=-1.25,
                                  yaw_rate_dps=33.33, status=0x11))
    assert wire[:4] == bytes((0xAA, 0x55, 11, 0x02))
    assert wire[4:-2] == struct.pack("<IhhhB", 1000, 9000, -125, 3333, 0x11)
    assert struct.unpack("<H", wire[-2:])[0] == crc16_bitwise(wire[2:-2])


def test_aim_round_trip():
    out = unpack_aim(pack_aim(AIM)[4:-2])
    assert out.t_ms == AIM.t_ms
    assert out.target_valid and not out.estop and out.mode == 1
    assert abs(out.yaw_deg - 12.34) < 1e-9
    assert abs(out.pitch_deg - -5.0) < 1e-9
    assert out.confidence == 1.0
    assert out.range_mm == 4000


def test_setpoints_clamp_to_i16_and_u8():
    a = AimPacket(t_ms=0, target_valid=True, estop=False, mode=1,
                  yaw_deg=400.0, pitch_deg=-400.0, yaw_rate_dps=0.0,
                  pitch_rate_dps=0.0, confidence=7.5, range_mm=99999)
    out = unpack_aim(pack_aim(a)[4:-2])
    assert out.yaw_deg == 327.67       # i16 max centideg
    assert out.pitch_deg == -327.68
    assert out.confidence == 1.0
    assert out.range_mm == 0xFFFF


def test_estop_flag_survives():
    a = AimPacket(t_ms=1, target_valid=False, estop=True, mode=0,
                  yaw_deg=0, pitch_deg=0, yaw_rate_dps=0, pitch_rate_dps=0,
                  confidence=0, range_mm=0)
    assert unpack_aim(pack_aim(a)[4:-2]).estop


def test_parser_reassembles_across_arbitrary_chunks():
    wire = pack_aim(AIM) + pack_telem(TelemPacket(5, 1.0, 2.0, 3.0, 0x01))
    p = Parser()
    got = []
    for i in range(len(wire)):        # worst case: one byte at a time
        got += p.feed(wire[i:i + 1])
    assert [t for t, _ in got] == [protocol.TYPE_AIM, protocol.TYPE_TELEM]
    assert unpack_aim(got[0][1]) == AIM


def test_parser_hunts_through_garbage_and_repeated_sync():
    # 0xAA 0xAA 0x55 must still lock on (the repeated-sync trap)
    noise = bytes((0x00, 0xFF, 0xAA, 0x13, 0xAA))
    p = Parser()
    got = p.feed(noise + pack_aim(AIM) + b"\xaa\x55\x03")  # trailing partial frame
    assert len(got) == 1
    assert unpack_aim(got[0][1]) == AIM


def test_parser_resyncs_after_corrupt_frame():
    good = pack_aim(AIM)
    corrupt = bytearray(good)
    corrupt[10] ^= 0xFF
    p = Parser()
    got = p.feed(bytes(corrupt) + good)
    assert len(got) == 1 and unpack_aim(got[0][1]) == AIM
    assert p.crc_errors == 1


def test_parser_rejects_oversize_len():
    p = Parser()
    assert p.feed(bytes((0xAA, 0x55, 200, 0x01)) + b"\x00" * 40) == []
    assert p.oversize_errors == 1


def test_fuzz_no_corrupt_packet_accepted():
    """SPEC 8: 0 CRC-accepted corrupt packets. CRC16 detects all 1- and 2-bit
    errors at these frame lengths, so every flipped frame must be rejected."""
    rng = random.Random(1234)
    wire = pack_aim(AIM)
    for _ in range(2000):
        corrupt = bytearray(wire)
        for _ in range(rng.choice((1, 2))):
            bit = rng.randrange(len(wire) * 8)
            corrupt[bit // 8] ^= 1 << (bit % 8)
        if bytes(corrupt) == wire:
            continue  # two flips landed on the same bit
        assert Parser().feed(bytes(corrupt)) == []
