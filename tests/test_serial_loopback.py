"""SerialLink over a real serial file descriptor (pty pair): the loopback test
from the Phase 3 exit criteria, no hardware needed.

A fake 'ESP32' owns the pty master: it parses incoming AIM frames byte-at-a-
time and answers each with a TELEM frame, so this exercises the full TX->RX->
decode->Telemetry path of SerialLink, including its IO thread, heartbeat
transmission, and the min-filtered clock offset."""
import os
import pty
import struct
import threading
import time

import pytest

from turretvision.link import protocol
from turretvision.link.base import AimOutput


@pytest.fixture
def pty_pair():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    yield master, os.ttyname(slave)
    os.close(master)
    os.close(slave)


class FakeEsp32(threading.Thread):
    """Parses AIM frames from the master fd; replies TELEM echoing the setpoint."""

    def __init__(self, fd):
        super().__init__(daemon=True)
        self.fd = fd
        self.parser = protocol.Parser()
        self.aims: list[protocol.AimPacket] = []
        self.run_flag = True

    def run(self):
        while self.run_flag:
            try:
                data = os.read(self.fd, 4096)
            except BlockingIOError:
                time.sleep(0.001)
                continue
            except OSError:
                return
            for pkt_type, payload in self.parser.feed(data):
                if pkt_type != protocol.TYPE_AIM:
                    continue
                aim = protocol.unpack_aim(payload)
                self.aims.append(aim)
                telem = protocol.pack_telem(protocol.TelemPacket(
                    t_ms=len(self.aims), yaw_deg=aim.yaw_deg, pitch_deg=aim.pitch_deg,
                    yaw_rate_dps=0.0, status=protocol.STATUS_HOMED))
                os.write(self.fd, telem)


def test_serial_loopback_roundtrip(pty_pair):
    from turretvision.link.serial_link import SerialLink

    master, slave_name = pty_pair
    esp = FakeEsp32(master)
    esp.start()
    link = SerialLink(slave_name, baud=115200, aim_rate_hz=200)
    try:
        link.send_aim(AimOutput(t=1.0, valid=True, az_deg=42.5, el_deg=-3.25,
                                az_rate_dps=10.0, el_rate_dps=-5.0, confidence=0.9))
        deadline = time.monotonic() + 3.0
        telem = None
        while time.monotonic() < deadline:
            telem = link.poll_telemetry()
            if telem is not None and abs(telem.yaw_deg - 42.5) < 1e-9:
                break
            time.sleep(0.005)
        assert telem is not None, "no telemetry made it back through the loopback"
        assert telem.yaw_deg == 42.5          # centideg-exact round trip
        assert telem.pitch_deg == -3.25
        assert telem.status & protocol.STATUS_HOMED
        assert telem.t_rx > 0
        assert link.clock_offset() is not None
        assert link.rx_errors == 0
        # every AIM frame the fake firmware accepted was CRC-clean
        assert esp.parser.crc_errors == 0
        assert any(a.target_valid and a.yaw_deg == 42.5 for a in esp.aims)
    finally:
        esp.run_flag = False
        link.close()


def test_heartbeat_flows_when_pipeline_is_silent(pty_pair):
    from turretvision.link.serial_link import SerialLink

    master, slave_name = pty_pair
    esp = FakeEsp32(master)
    esp.start()
    link = SerialLink(slave_name, baud=115200, aim_rate_hz=100)
    try:
        # never call send_aim: the IO thread must still heartbeat target_valid=0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(esp.aims) < 5:
            time.sleep(0.01)
        assert len(esp.aims) >= 5, "no heartbeats on a silent pipeline"
        assert all(not a.target_valid for a in esp.aims[:5])
    finally:
        esp.run_flag = False
        link.close()


def test_heartbeat_freezes_last_setpoint(pty_pair):
    from turretvision.link.serial_link import SerialLink

    master, slave_name = pty_pair
    esp = FakeEsp32(master)
    esp.start()
    link = SerialLink(slave_name, baud=115200, aim_rate_hz=100)
    try:
        link.send_aim(AimOutput(2.0, True, 30.0, 5.0, 0.0, 0.0, 1.0))
        time.sleep(0.1)
        link.send_aim(AimOutput(2.1, False, 0.0, 0.0, 0.0, 0.0, 0.0))  # target lost
        time.sleep(0.2)
        lost = [a for a in esp.aims if not a.target_valid]
        assert lost, "no heartbeat frames observed"
        # SPEC 5: setpoints frozen at last commanded value, not zeroed
        assert struct.unpack("<h", struct.pack("<h", 3000))[0] / 100.0 == 30.0
        assert all(a.yaw_deg == 30.0 and a.pitch_deg == 5.0 for a in lost[-3:])
    finally:
        esp.run_flag = False
        link.close()
