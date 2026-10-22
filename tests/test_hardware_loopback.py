"""Hardware mode over a real (virtual) serial port, with the simulated ESP32 on the far end.

Exercises the exact pyserial path the Jetson uses: open the port, handshake,
push Config, sync clocks, and drive the turret from a GUI-style jog command.
"""

import os
import threading
import time

import numpy as np
import pytest

from turret.config import Config
from turret.runtime.bus import Bus
from turret.runtime.hardware import HardwareRunner
from turret.sim.esp32 import SimESP32

pytestmark = pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a POSIX pty")


class FdTransport:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        os.set_blocking(fd, False)

    def write(self, data: bytes) -> None:
        os.write(self.fd, data)

    def read(self) -> bytes:
        try:
            return os.read(self.fd, 4096)
        except BlockingIOError:
            return b""

    def close(self) -> None:
        pass


def test_hardware_mode_over_pty():
    master, slave = os.openpty()
    import tty

    tty.setraw(slave)  # no line discipline mangling: binary frames must pass untouched
    esp = SimESP32(FdTransport(master), time.monotonic, 5.0, 20.0, np.random.default_rng(0))
    stop = threading.Event()

    def esp_loop():
        while not stop.is_set():
            esp.update()
            time.sleep(0.0005)

    t = threading.Thread(target=esp_loop, daemon=True)
    t.start()
    cfg = Config()
    cfg.link.port = os.ttyname(slave)
    bus = Bus()
    runner = HardwareRunner(cfg, bus)
    runner.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not runner.pipeline.link.synced:
            time.sleep(0.02)
        assert runner.pipeline.link.synced, runner.pipeline.link.stats
        assert esp.cfg is not None and esp.cfg.trigger_hz == cfg.camera.trigger_hz
        bus.command({"t": "mode", "mode": "MANUAL"})
        bus.command({"t": "jog", "yaw": 60.0, "pitch": 0.0})
        time.sleep(1.0)
        assert esp.truth.yaw_deg > 20.0
        tel = bus.get_telemetry()
        assert tel["link"]["synced"] and tel["mode"] == "MANUAL"
    finally:
        runner.stop()
        stop.set()
        t.join(1)
        os.close(master)
