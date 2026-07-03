#!/usr/bin/env python3
"""Dump the REAL modes the OV9782's UVC firmware exposes.

WHY run this before trusting any fps number: '~100 fps' is a datasheet claim;
the camera actually offers specific (format, resolution, fps) combos and UVC
silently falls back to the nearest one it likes. Every latency budget in the
pipeline depends on which mode is real. Verify, don't trust marketing.
Usage: python tools/enumerate_camera.py [/dev/video0]
"""
import subprocess
import sys

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
try:
    out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats-ext"],
                         capture_output=True, text=True, check=True).stdout
    print(out)
except FileNotFoundError:
    sys.exit("v4l2-ctl not found -- install with: sudo apt install v4l-utils")
except subprocess.CalledProcessError as e:
    sys.exit(f"v4l2-ctl failed: {e.stderr}")
