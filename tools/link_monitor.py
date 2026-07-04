#!/usr/bin/env python3
"""Decode and display live packet traffic on a serial port.

Point it at whichever side of the link you can tap: the ESP32's TX gives you
TELEM packets, a loopback/tee of the Jetson's TX gives you AIM packets — the
parser handles both TYPEs on the same stream. Prints one line per packet plus
a rate/error summary line every second; corrupt frames bump the error counter
instead of crashing, which is the point (SPEC 5: resync and carry on).

Usage: python tools/link_monitor.py /dev/ttyUSB0 [--baud 115200] [--hex]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.link import protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--hex", action="store_true", help="also dump raw frame payload hex")
args = ap.parse_args()

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed: pip install pyserial")

try:
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
except serial.SerialException as e:
    sys.exit(f"cannot open {args.port}: {e}")

parser = protocol.Parser()
counts = {protocol.TYPE_AIM: 0, protocol.TYPE_TELEM: 0}
t_stats = time.monotonic()
print(f"monitoring {args.port} @ {args.baud} — ctrl-C to stop")
try:
    while True:
        data = ser.read(4096)
        for pkt_type, payload in parser.feed(data):
            counts[pkt_type] = counts.get(pkt_type, 0) + 1
            pkt = protocol.decode(pkt_type, payload)
            if isinstance(pkt, protocol.AimPacket):
                print(f"AIM   t={pkt.t_ms:>10}ms yaw={pkt.yaw_deg:+8.2f} "
                      f"pitch={pkt.pitch_deg:+7.2f} rate=({pkt.yaw_rate_dps:+7.1f},"
                      f"{pkt.pitch_rate_dps:+7.1f}) conf={pkt.confidence:.2f} "
                      f"range={pkt.range_mm}mm valid={int(pkt.target_valid)} "
                      f"estop={int(pkt.estop)} mode={pkt.mode}")
            elif isinstance(pkt, protocol.TelemPacket):
                bits = "".join((
                    "H" if pkt.homed else "-",
                    "Y" if pkt.status & protocol.STATUS_YAW_AT_LIMIT else "-",
                    "P" if pkt.status & protocol.STATUS_PITCH_AT_LIMIT else "-",
                    "F" if pkt.fault else "-",
                    "E" if pkt.estopped else "-"))
                print(f"TELEM t={pkt.t_ms:>10}ms yaw={pkt.yaw_deg:+8.2f} "
                      f"pitch={pkt.pitch_deg:+7.2f} rate={pkt.yaw_rate_dps:+7.1f} "
                      f"status=[{bits}]")
            else:
                print(f"?     TYPE=0x{pkt_type:02x} LEN={len(payload)} (unknown)")
            if args.hex:
                print(f"      {payload.hex(' ')}")
        now = time.monotonic()
        if now - t_stats >= 1.0:
            t_stats = now
            print(f"--- aim {counts.get(protocol.TYPE_AIM, 0)}  "
                  f"telem {counts.get(protocol.TYPE_TELEM, 0)}  "
                  f"crc_err {parser.crc_errors}  oversize {parser.oversize_errors}")
except KeyboardInterrupt:
    print(f"\ntotal: aim {counts.get(protocol.TYPE_AIM, 0)}  "
          f"telem {counts.get(protocol.TYPE_TELEM, 0)}  "
          f"crc_err {parser.crc_errors}")
    ser.close()
