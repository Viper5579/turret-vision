"""Packet framing: id + payload + CRC16, COBS encoded, 0x00 delimited.

`encode_frame` builds wire bytes. `FrameDecoder` eats an arbitrary byte stream
(serial reads never line up with packets) and yields complete, CRC-checked
(id, payload) pairs, counting everything it had to throw away.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from . import cobs
from .crc import crc16

MAX_FRAME = 256  # encoded bytes; anything longer is garbage


def encode_frame(msg_id: int, payload: bytes) -> bytes:
    body = bytes([msg_id]) + payload
    crc = crc16(body)
    return cobs.encode(body + crc.to_bytes(2, "little")) + b"\x00"


@dataclass
class DecoderStats:
    frames: int = 0
    crc_errors: int = 0
    cobs_errors: int = 0
    oversize: int = 0


class FrameDecoder:
    def __init__(self) -> None:
        self._buf = bytearray()
        self.stats = DecoderStats()

    def feed(self, data: bytes) -> Iterator[tuple[int, bytes]]:
        for b in data:
            if b != 0:
                if len(self._buf) < MAX_FRAME:
                    self._buf.append(b)
                elif len(self._buf) == MAX_FRAME:
                    self.stats.oversize += 1
                    self._buf.append(b)  # mark as poisoned; dropped at delimiter
                continue
            raw = bytes(self._buf)
            self._buf.clear()
            if not raw:
                continue
            if len(raw) > MAX_FRAME:
                continue
            try:
                body = cobs.decode(raw)
            except ValueError:
                self.stats.cobs_errors += 1
                continue
            if len(body) < 3:
                self.stats.cobs_errors += 1
                continue
            data_part, crc_bytes = body[:-2], body[-2:]
            if crc16(data_part) != int.from_bytes(crc_bytes, "little"):
                self.stats.crc_errors += 1
                continue
            self.stats.frames += 1
            yield data_part[0], data_part[1:]
