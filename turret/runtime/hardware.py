"""Run the pipeline against the real ESP32 over USB serial.

M0 scope: link, clock sync, and turret control from the GUI (Hold / Manual, the
aim pad, keyboard, Xbox controller). The camera capture path joins in M1, at
which point frames feed `Pipeline.on_frame()` exactly like the simulator does.
"""

from __future__ import annotations

import logging
import threading
import time

from ..config import Config
from ..doctor import find_serial_ports
from ..link.transport import SerialTransport
from .bus import Bus
from .pipeline import Mode, Pipeline

log = logging.getLogger(__name__)


class HardwareRunner:
    def __init__(self, cfg: Config, bus: Bus) -> None:
        port = cfg.link.port
        if port == "auto":
            ports = find_serial_ports()
            if not ports:
                raise RuntimeError("No ESP32 serial port found (expected /dev/ttyUSB0). Is it plugged in?")
            port = ports[0]
        log.info("ESP32 on %s at %d baud", port, cfg.link.baud)
        self.transport = SerialTransport(port, cfg.link.baud)
        self.pipeline = Pipeline(cfg, self.transport, time.monotonic, bus)
        self.pipeline.mode = Mode.IDLE
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="hardware", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        p = self.pipeline
        next_pub = 0.0
        while not self._stop.is_set():
            p.tick()
            now = time.monotonic()
            if now >= next_pub:  # no camera yet, so publish telemetry on a timer
                p.bus.publish_telemetry(p.telemetry())
                next_pub = now + 1 / 30
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.transport.close()
