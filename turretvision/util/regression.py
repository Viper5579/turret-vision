"""Replay regression harness (Phase 5, D4's payoff).

The workflow this enables: record a run once, replay those exact frames
through any code change, and DIFF the per-frame state CSVs. "Did my detector
tweak break tracking?" becomes a number instead of a vibe.

run_replay() drives the REAL main() (not a parallel re-implementation that
could drift from it) with a temporary config: same stages, same CSV writer,
just pointed at the recording with realtime pacing off and the window
suppressed. The temp config lives in its own directory, so no local.yaml
overlay leaks into a regression run — replays are deterministic by
construction (same frames, same config, no wall-clock dependence).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import deep_merge, set_dotted


def run_replay(recording: str | Path, out_dir: str | Path,
               base_config: str | Path = "config/default.yaml",
               overrides: dict | None = None,
               max_frames: int | None = None) -> Path:
    """Run the full pipeline over a recording; returns the state CSV path."""
    from ..main import main
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    for key, value in {
        "camera.backend": "replay",
        "camera.replay_path": str(recording),
        "camera.replay_realtime": False,
        "ui.show_window": False,
        "link.backend": "none",          # mock dynamics are wall-clock = nondeterministic
        "logging.csv_state_log": True,
        "logging.record_frames": False,  # replaying a replay helps no one
        "logging.run_log_dir": str(out_dir),
    }.items():
        set_dotted(cfg, key, value)
    if overrides:
        deep_merge(cfg, overrides)
    tmp_cfg = out_dir / "replay_config.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)
    argv = ["--config", str(tmp_cfg), "--headless"]
    if max_frames:
        argv += ["--max-frames", str(max_frames)]
    rc = main(argv)
    if rc != 0:
        raise RuntimeError(f"pipeline exited {rc} on replay {recording}")
    return out_dir / "state.csv"


@dataclass
class CompareReport:
    frames: int                  # rows compared
    valid_mismatch_frac: float   # fraction of frames where target_valid differs
    mean_px_dev: float           # mean |pos| deviation where both runs tracked
    max_px_dev: float
    mean_angle_dev: float        # mean az/el deviation (deg)
    tracked_a: int
    tracked_b: int

    def passed(self, tol_valid_frac: float = 0.02, tol_px: float = 3.0) -> bool:
        return (self.valid_mismatch_frac <= tol_valid_frac
                and self.mean_px_dev <= tol_px)

    def summary(self) -> str:
        return (f"{self.frames} frames | valid mismatch {self.valid_mismatch_frac:.1%} "
                f"| px dev mean {self.mean_px_dev:.2f} max {self.max_px_dev:.2f} "
                f"| angle dev {self.mean_angle_dev:.3f} deg "
                f"| tracked {self.tracked_a} vs {self.tracked_b}")


def _rows(path: str | Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def compare_csv(a_path: str | Path, b_path: str | Path) -> CompareReport:
    """Row-by-row comparison of two state CSVs from the same recording."""
    a, b = _rows(a_path), _rows(b_path)
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("empty state CSV — did the replay produce frames?")
    mismatch = 0
    px_devs: list[float] = []
    ang_devs: list[float] = []
    for ra, rb in zip(a[:n], b[:n], strict=True):
        va, vb = ra["valid"] == "1", rb["valid"] == "1"
        if va != vb:
            mismatch += 1
            continue
        if va and ra["px_x"] and rb["px_x"]:
            px_devs.append(math.hypot(float(ra["px_x"]) - float(rb["px_x"]),
                                      float(ra["px_y"]) - float(rb["px_y"])))
            ang_devs.append(math.hypot(float(ra["az"]) - float(rb["az"]),
                                       float(ra["el"]) - float(rb["el"])))
    return CompareReport(
        frames=n,
        valid_mismatch_frac=mismatch / n,
        mean_px_dev=sum(px_devs) / len(px_devs) if px_devs else 0.0,
        max_px_dev=max(px_devs, default=0.0),
        mean_angle_dev=sum(ang_devs) / len(ang_devs) if ang_devs else 0.0,
        tracked_a=sum(1 for r in a[:n] if r["valid"] == "1"),
        tracked_b=sum(1 for r in b[:n] if r["valid"] == "1"),
    )
