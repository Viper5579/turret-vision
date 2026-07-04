"""Wire protocol: framing, CRC16, pack/unpack (SPEC 5, mirrored in
firmware/PROTOCOL.md). Pure functions and dataclasses — bytes in, packets out —
so this file is unit-testable with zero hardware and doubles as the reference
implementation when writing the ESP32 firmware (port the constants, don't
retype them).

Frame (both directions, little-endian):
    [0xAA] [0x55] [LEN u8] [TYPE u8] [PAYLOAD ...] [CRC16 u16 LE]
CRC-16/CCITT-FALSE over LEN..PAYLOAD. WHY CRC16 over an XOR checksum: XOR
passes any two errors that cancel and all byte swaps — exactly what a noisy
line on a vibrating platform produces. CRC16 catches all single/double-bit
errors and burst errors <=16 bits.

WHY i16 centidegrees instead of float32: 0.01 deg resolution is ~10x finer
than a 1.8deg stepper can use even at 1/16 microstepping, halves packet size,
and sidesteps float-marshaling bugs between Python's struct and C.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

SYNC1, SYNC2 = 0xAA, 0x55
TYPE_AIM = 0x01     # Jetson -> ESP32
TYPE_TELEM = 0x02   # ESP32 -> Jetson
MAX_PAYLOAD = 32    # sanity bound for the parser; largest real payload is 16

# AIM flags (u8)
FLAG_TARGET_VALID = 0x01
FLAG_ESTOP = 0x02       # in EVERY packet, not a one-shot: can't be lost
MODE_SHIFT, MODE_MASK = 2, 0x03   # 0 idle, 1 track, 2 manual

# TELEM status bits (u8)
STATUS_HOMED = 0x01
STATUS_YAW_AT_LIMIT = 0x02
STATUS_PITCH_AT_LIMIT = 0x04
STATUS_FAULT = 0x08
STATUS_ESTOPPED = 0x10

_AIM_STRUCT = struct.Struct("<IBhhhhBH")   # t_ms flags yaw pitch yaw_rate pitch_rate conf range
_TELEM_STRUCT = struct.Struct("<IhhhB")    # t_ms yaw pitch yaw_rate status
AIM_PAYLOAD_LEN = _AIM_STRUCT.size         # 16
TELEM_PAYLOAD_LEN = _TELEM_STRUCT.size     # 11


def _crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1) & 0xFFFF
        table.append(crc)
    return table


_CRC_TABLE = _crc_table()


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no xorout.
    Check value: crc16_ccitt(b"123456789") == 0x29B1."""
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ b]
    return crc


def _clamp_i16_centideg(deg: float) -> int:
    return max(-32768, min(32767, round(deg * 100.0)))


def _clamp_u8(v: int) -> int:
    return max(0, min(255, v))


def _clamp_u16(v: int) -> int:
    return max(0, min(0xFFFF, v))


@dataclass(frozen=True)
class AimPacket:
    """Jetson -> ESP32 setpoint command. Angles are ABSOLUTE degrees (D2:
    idempotent — a lost or repeated packet causes zero accumulated error)."""
    t_ms: int                 # Jetson monotonic ms (wraps u32; deltas only)
    target_valid: bool
    estop: bool
    mode: int                 # 0 idle, 1 track, 2 manual
    yaw_deg: float
    pitch_deg: float
    yaw_rate_dps: float       # feedforward; 0 if unused
    pitch_rate_dps: float
    confidence: float         # 0..1
    range_mm: int             # 0 = unknown

    @property
    def flags(self) -> int:
        return ((FLAG_TARGET_VALID if self.target_valid else 0)
                | (FLAG_ESTOP if self.estop else 0)
                | ((self.mode & MODE_MASK) << MODE_SHIFT))


@dataclass(frozen=True)
class TelemPacket:
    """ESP32 -> Jetson turret state (the ONLY true source of pose — the step
    counter lives on the ESP32; dead-reckoning from sent commands drifts, D3)."""
    t_ms: int                 # ESP32 millis(); unrelated clock domain to Jetson
    yaw_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    status: int

    @property
    def homed(self) -> bool:
        return bool(self.status & STATUS_HOMED)

    @property
    def estopped(self) -> bool:
        return bool(self.status & STATUS_ESTOPPED)

    @property
    def fault(self) -> bool:
        return bool(self.status & STATUS_FAULT)


def _frame(pkt_type: int, payload: bytes) -> bytes:
    body = bytes((len(payload), pkt_type)) + payload
    return bytes((SYNC1, SYNC2)) + body + struct.pack("<H", crc16_ccitt(body))


def pack_aim(a: AimPacket) -> bytes:
    payload = _AIM_STRUCT.pack(
        a.t_ms & 0xFFFFFFFF, a.flags,
        _clamp_i16_centideg(a.yaw_deg), _clamp_i16_centideg(a.pitch_deg),
        _clamp_i16_centideg(a.yaw_rate_dps), _clamp_i16_centideg(a.pitch_rate_dps),
        _clamp_u8(round(a.confidence * 255.0)), _clamp_u16(a.range_mm))
    return _frame(TYPE_AIM, payload)


def unpack_aim(payload: bytes) -> AimPacket:
    t_ms, flags, yaw, pitch, yr, pr, conf, rng = _AIM_STRUCT.unpack(payload)
    return AimPacket(
        t_ms=t_ms,
        target_valid=bool(flags & FLAG_TARGET_VALID),
        estop=bool(flags & FLAG_ESTOP),
        mode=(flags >> MODE_SHIFT) & MODE_MASK,
        yaw_deg=yaw / 100.0, pitch_deg=pitch / 100.0,
        yaw_rate_dps=yr / 100.0, pitch_rate_dps=pr / 100.0,
        confidence=conf / 255.0, range_mm=rng)


def pack_telem(t: TelemPacket) -> bytes:
    payload = _TELEM_STRUCT.pack(
        t.t_ms & 0xFFFFFFFF,
        _clamp_i16_centideg(t.yaw_deg), _clamp_i16_centideg(t.pitch_deg),
        _clamp_i16_centideg(t.yaw_rate_dps), t.status & 0xFF)
    return _frame(TYPE_TELEM, payload)


def unpack_telem(payload: bytes) -> TelemPacket:
    t_ms, yaw, pitch, yr, status = _TELEM_STRUCT.unpack(payload)
    return TelemPacket(t_ms=t_ms, yaw_deg=yaw / 100.0, pitch_deg=pitch / 100.0,
                       yaw_rate_dps=yr / 100.0, status=status)


def decode(pkt_type: int, payload: bytes) -> AimPacket | TelemPacket | None:
    """Decode a validated (type, payload) from the parser; None if unknown/short."""
    if pkt_type == TYPE_AIM and len(payload) == AIM_PAYLOAD_LEN:
        return unpack_aim(payload)
    if pkt_type == TYPE_TELEM and len(payload) == TELEM_PAYLOAD_LEN:
        return unpack_telem(payload)
    return None


class Parser:
    """Byte-at-a-time framing state machine.

    WHY a state machine and not read-then-split: serial has no message
    boundaries; assuming reads align to packets works on the bench and fails
    in the field. Feed it arbitrary chunks; it returns every validated
    (type, payload) and silently resyncs (hunting the next 0xAA 0x55) on CRC
    failure or a nonsense LEN. Counters stay public so links/monitors can
    report line health.
    """
    _HUNT1, _HUNT2, _LEN, _TYPE, _PAYLOAD, _CRC1, _CRC2 = range(7)

    def __init__(self):
        self._state = self._HUNT1
        self._len = 0
        self._type = 0
        self._buf = bytearray()
        self._crc_lo = 0
        self.crc_errors = 0
        self.oversize_errors = 0
        self.packets = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        out: list[tuple[int, bytes]] = []
        for b in data:
            if self._state == self._HUNT1:
                if b == SYNC1:
                    self._state = self._HUNT2
            elif self._state == self._HUNT2:
                # AA AA 55 must still sync: stay in HUNT2 on a repeated AA
                if b == SYNC2:
                    self._state = self._LEN
                elif b != SYNC1:
                    self._state = self._HUNT1
            elif self._state == self._LEN:
                if b > MAX_PAYLOAD:
                    self.oversize_errors += 1
                    self._state = self._HUNT1
                else:
                    self._len = b
                    self._buf = bytearray((b,))
                    self._state = self._TYPE
            elif self._state == self._TYPE:
                self._type = b
                self._buf.append(b)
                self._state = self._PAYLOAD if self._len else self._CRC1
            elif self._state == self._PAYLOAD:
                self._buf.append(b)
                if len(self._buf) - 2 == self._len:
                    self._state = self._CRC1
            elif self._state == self._CRC1:
                self._crc_lo = b
                self._state = self._CRC2
            else:  # _CRC2
                rx_crc = self._crc_lo | (b << 8)
                if rx_crc == crc16_ccitt(bytes(self._buf)):
                    self.packets += 1
                    out.append((self._type, bytes(self._buf[2:])))
                else:
                    self.crc_errors += 1
                self._state = self._HUNT1
        return out
