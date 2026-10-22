"""Consistent Overhead Byte Stuffing.

COBS rewrites a packet so it contains no 0x00 bytes, then 0x00 marks the end
of every packet. A receiver that joins mid-stream resyncs at the next zero,
and there are no escape sequences to get wrong. Overhead is 1 byte per 254.
The firmware twin is firmware/lib/proto/src/cobs.h.
"""

from __future__ import annotations


def encode(data: bytes) -> bytes:
    """Encode a packet. The result contains no zero bytes and no delimiter."""
    out = bytearray([0])  # placeholder for the first code byte
    code_idx = 0
    code = 1
    for b in data:
        if b == 0:
            out[code_idx] = code
            code_idx = len(out)
            out.append(0)
            code = 1
        else:
            out.append(b)
            code += 1
            if code == 0xFF:
                out[code_idx] = code
                code_idx = len(out)
                out.append(0)
                code = 1
    out[code_idx] = code
    return bytes(out)


def decode(data: bytes) -> bytes:
    """Decode one COBS block (without the trailing 0x00). Raises ValueError on corruption."""
    out = bytearray()
    idx = 0
    n = len(data)
    while idx < n:
        code = data[idx]
        if code == 0:
            raise ValueError("zero byte inside COBS block")
        idx += 1
        end = idx + code - 1
        if end > n:
            raise ValueError("COBS block truncated")
        out += data[idx:end]
        idx = end
        if code < 0xFF and idx < n:
            out.append(0)
    return bytes(out)
