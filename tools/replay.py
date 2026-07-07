#!/usr/bin/env python3
"""Run the pipeline on a recording; optionally diff against a baseline (Phase 5).

The regression workflow:
  python tools/record.py --seconds 10                     # once: record a run
  python tools/replay.py logs/rec_*.mp4 --save-baseline golden.csv
  ... hack on the detector/tracker ...
  python tools/replay.py logs/rec_*.mp4 --baseline golden.csv   # exit 0 = no regression

Same frames + same config + no wall clock = deterministic, so any CSV drift
is YOUR change, not noise. Tolerances exist for intentional tuning changes
(--tol-px / --tol-valid).
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turretvision.util.regression import compare_csv, run_replay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("recording", help="video recorded by tools/record.py or logging.record_frames")
ap.add_argument("--config", default="config/default.yaml")
ap.add_argument("--out", default=None, help="dir for the state CSV (default: temp)")
ap.add_argument("--baseline", default=None, help="golden CSV to regression-diff against")
ap.add_argument("--save-baseline", default=None, help="save this run's CSV as the new golden")
ap.add_argument("--tol-px", type=float, default=3.0)
ap.add_argument("--tol-valid", type=float, default=0.02)
ap.add_argument("--max-frames", type=int, default=None)
args = ap.parse_args()

out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="tv-replay-"))
csv_path = run_replay(args.recording, out_dir, base_config=args.config,
                      max_frames=args.max_frames)
print(f"state CSV: {csv_path}")

if args.save_baseline:
    shutil.copy(csv_path, args.save_baseline)
    print(f"baseline saved -> {args.save_baseline}")

if args.baseline:
    report = compare_csv(args.baseline, csv_path)
    print(report.summary())
    if report.passed(tol_valid_frac=args.tol_valid, tol_px=args.tol_px):
        print(f"PASS  within tolerance (valid <= {args.tol_valid:.0%}, "
              f"px <= {args.tol_px})")
        sys.exit(0)
    print("FAIL  behavior drifted from the baseline — if the change is "
          "intentional, re-run with --save-baseline to accept it")
    sys.exit(1)
