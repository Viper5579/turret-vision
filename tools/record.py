#!/usr/bin/env python3
"""Record raw camera frames + true grab timestamps to disk (Phase 5).

Capture ONLY — no detection running — so the recording is a clean input for
tools/replay.py regression runs and offline tuning. The sidecar
.timestamps.jsonl carries the real grab times (capture jitter and drops
included) because dt is what the tracker's filter eats; container fps
metadata lies about it.

Alternative: set logging.record_frames: true in the config to tee frames
DURING a live tracking run instead.

Usage: python tools/record.py [--seconds 10] [--out logs/rec.mp4]
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.capture.recorder import Recorder  # noqa: E402
from turretvision.main import build_source  # noqa: E402
from turretvision.util.config import Config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/default.yaml")
ap.add_argument("--source", choices=["v4l2", "gstreamer", "synthetic"], default=None)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--out", default=None)
args = ap.parse_args()
args.replay = None          # build_source expects the attribute
args.max_frames = None
args.headless = True

cfg = Config.load(args.config)
src = build_source(cfg, args)
src.start()
out = args.out or f"logs/rec_{datetime.now():%Y%m%d_%H%M%S}.mp4"
rec = Recorder(out, fps=cfg.get("camera.fps", 60), resolution=src.resolution)

n = 0
t0 = time.monotonic()
try:
    while time.monotonic() - t0 < args.seconds:
        frame = src.read()
        if frame is None:
            continue
        rec.write(frame)
        n += 1
finally:
    src.stop()
    rec.close()
dt = time.monotonic() - t0
print(f"recorded {n} frames in {dt:.1f}s ({n / dt:.1f} fps) -> {out} "
      f"(+ .timestamps.jsonl)")
