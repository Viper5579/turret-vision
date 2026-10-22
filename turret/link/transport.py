"""Byte transports under the link: real serial port, or an in-memory pipe for the simulator.

Both look the same to `Link`: `write(bytes)` never blocks the caller, `read()`
returns whatever bytes have arrived (possibly none).
"""

from __future__ import annotations

import heapq
import itertools
import queue
import random
import threading
from collections.abc import Callable
from typing import Protocol


class Transport(Protocol):
    def write(self, data: bytes) -> None: ...

    def read(self) -> bytes: ...

    def close(self) -> None: ...


class DelayedPipe:
    """One direction of a simulated wire: bytes appear after a delay with jitter.

    Delivery order is preserved (a real UART never reorders bytes), so each
    write is scheduled no earlier than the previous one.
    """

    def __init__(
        self,
        now: Callable[[], float],
        delay_s: float = 0.001,
        jitter_s: float = 0.0003,
        seed: int = 0,
    ) -> None:
        self._now = now
        self._delay = delay_s
        self._jitter = jitter_s
        self._rng = random.Random(seed)
        self._heap: list[tuple[float, int, bytes]] = []
        self._last = 0.0
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> None:
        with self._lock:
            t = self._now() + self._delay + self._rng.random() * self._jitter
            t = max(t, self._last)
            self._last = t
            heapq.heappush(self._heap, (t, next(self._counter), data))

    def read(self) -> bytes:
        now = self._now()
        out = bytearray()
        with self._lock:
            while self._heap and self._heap[0][0] <= now:
                out += heapq.heappop(self._heap)[2]
        return bytes(out)


class PipeEnd:
    """A Transport made of two DelayedPipes (tx, rx)."""

    def __init__(self, tx: DelayedPipe, rx: DelayedPipe) -> None:
        self._tx = tx
        self._rx = rx

    def write(self, data: bytes) -> None:
        self._tx.write(data)

    def read(self) -> bytes:
        return self._rx.read()

    def close(self) -> None:
        pass


def sim_wire(now: Callable[[], float], delay_s: float, jitter_s: float, seed: int = 0) -> tuple[PipeEnd, PipeEnd]:
    """Return (jetson_end, esp_end) of a simulated USB-serial wire."""
    j2e = DelayedPipe(now, delay_s, jitter_s, seed)
    e2j = DelayedPipe(now, delay_s, jitter_s, seed + 1)
    return PipeEnd(j2e, e2j), PipeEnd(e2j, j2e)


class SerialTransport:
    """pyserial port with a reader thread, so the pipeline never waits on I/O."""

    def __init__(self, port: str, baud: int) -> None:
        import serial  # imported lazily so the simulator needs no pyserial

        self._ser = serial.Serial(port, baud, timeout=0.01, write_timeout=0.05)
        self._rx: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, name="serial-rx", daemon=True)
        self._thread.start()
        self.write_errors = 0

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
            except Exception:  # port vanished (unplugged); stop quietly, Link reports link loss
                break
            if chunk:
                self._rx.put(chunk)

    def write(self, data: bytes) -> None:
        try:
            self._ser.write(data)
        except Exception:
            self.write_errors += 1

    def read(self) -> bytes:
        out = bytearray()
        while True:
            try:
                out += self._rx.get_nowait()
            except queue.Empty:
                return bytes(out)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._ser.close()
