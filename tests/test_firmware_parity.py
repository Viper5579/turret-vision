"""Firmware <-> Python parity: compile the ACTUAL ESP32 protocol/control code
with the host compiler and prove it is byte-for-byte and step-for-step
compatible with the Python reference (link/protocol.py, MockLink's trapezoid).

This is the test that lets the firmware be trusted before the board is even
wired: if these pass, the C++ that will run on the ESP32 already speaks the
exact same bytes the Jetson sends and expects.
"""
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from turretvision.link import protocol
from turretvision.link.mock_link import _step_axis

REPO = Path(__file__).resolve().parents[1]
FW = REPO / "firmware" / "turret"

pytestmark = pytest.mark.skipif(shutil.which("g++") is None,
                                reason="g++ not available for host-compiling firmware")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    out = tmp_path_factory.mktemp("fw") / "harness"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Werror", "-o", str(out),
                    str(FW / "host_harness.cpp"), str(FW / "protocol.cpp"),
                    str(FW / "control.cpp")],
                   check=True, capture_output=True, text=True)

    def run(*args) -> str:
        return subprocess.run([str(out), *args], check=True, capture_output=True,
                              text=True).stdout.strip()
    return run


def test_crc_check_value_and_random_parity(harness):
    assert harness("crc", b"123456789".hex()) == "29B1"
    rng = random.Random(99)
    for _ in range(20):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
        assert int(harness("crc", data.hex()), 16) == protocol.crc16_ccitt(data)


def test_telem_pack_byte_exact_parity(harness):
    cases = [(1000, 9000, -125, 3333, 0x11), (0, 0, 0, 0, 0),
             (0xFFFFFFFF, -17000, 6000, -24000, 0x1F)]
    for t_ms, yaw, pitch, rate, status in cases:
        py = protocol.pack_telem(protocol.TelemPacket(
            t_ms=t_ms, yaw_deg=yaw / 100.0, pitch_deg=pitch / 100.0,
            yaw_rate_dps=rate / 100.0, status=status))
        cpp = bytes.fromhex(harness("packtelem", str(t_ms), str(yaw), str(pitch),
                                    str(rate), str(status)))
        assert cpp == py


def test_firmware_parses_python_aim_frames(harness):
    aim = protocol.AimPacket(t_ms=123456, target_valid=True, estop=False, mode=1,
                             yaw_deg=42.5, pitch_deg=-3.25, yaw_rate_dps=100.0,
                             pitch_rate_dps=-55.5, confidence=0.9, range_mm=4000)
    stream = b"\x00\xaa\x13" + protocol.pack_aim(aim) + protocol.pack_aim(aim)
    out = harness("parse", stream.hex())
    lines = out.splitlines()
    assert len(lines) == 3      # two AIM lines + stats
    assert lines[0] == ("AIM t=123456 flags=5 yaw=4250 pitch=-325 yr=10000 "
                        "pr=-5550 conf=230 range=4000")   # round(0.9*255)=230
    assert "packets=2 crc_errors=0" in lines[2]


def test_firmware_rejects_corruption_like_python(harness):
    wire = bytearray(protocol.pack_aim(protocol.AimPacket(
        t_ms=1, target_valid=True, estop=False, mode=1, yaw_deg=1.0, pitch_deg=2.0,
        yaw_rate_dps=0, pitch_rate_dps=0, confidence=1.0, range_mm=0)))
    wire[9] ^= 0x40
    out = harness("parse", bytes(wire).hex())
    assert "AIM" not in out
    assert "crc_errors=1" in out


def test_trapezoid_axis_matches_mocklink_step_for_step(harness):
    vmax, amax, dt, n, target, ff = 240.0, 900.0, 0.005, 400, 90.0, 12.0
    cpp = [tuple(map(float, line.split()))
           for line in harness("axis", str(vmax), str(amax), str(dt), str(n),
                               str(target), str(ff)).splitlines()]
    pos = vel = 0.0
    for i in range(n):
        pos, vel = _step_axis(pos, vel, target, dt, vmax, amax, ff)
        assert math.isclose(cpp[i][0], pos, abs_tol=2e-3), f"pos diverged at step {i}"
        assert math.isclose(cpp[i][1], vel, abs_tol=2e-3), f"vel diverged at step {i}"
    assert abs(pos - target) < 0.5      # and it actually arrives
