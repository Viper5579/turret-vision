"""CRC-16/CCITT-FALSE, the checksum on every link packet.

Why CRC-16 and not a sum or XOR: it catches every single- and double-bit
error and every burst up to 16 bits, which is exactly what a noisy USB-serial
line on a vibrating turret produces. The firmware has the same table-driven
implementation (firmware/lib/proto/src/crc16.h).
"""

from __future__ import annotations


def _make_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
        table.append(crc)
    return table


_TABLE = _make_table()


def crc16(data: bytes | bytearray | memoryview, crc: int = 0xFFFF) -> int:
    """Return CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection)."""
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc
