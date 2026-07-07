"""Phase 5 replay regression harness: record synthetic frames, replay them
through the REAL pipeline twice, and prove the runs are identical — then prove
the comparator actually catches drift when a run is tampered with."""
import csv
import json

import pytest

from turretvision.capture.recorder import Recorder
from turretvision.capture.replay import ReplaySource
from turretvision.capture.synthetic import SyntheticSource
from turretvision.util.regression import compare_csv, run_replay


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A short synthetic run recorded to mp4 + sidecar timestamps."""
    out = tmp_path_factory.mktemp("rec") / "run.mp4"
    src = SyntheticSource(n_frames=120, fps=60, realtime=False)
    src.start()
    rec = Recorder(str(out), fps=60, resolution=src.resolution)
    while True:
        f = src.read()
        if f is None:
            break
        rec.write(f)
    rec.close()
    return out


def test_replay_source_uses_recorded_timestamps(recording):
    sidecar = recording.with_suffix(".timestamps.jsonl")
    assert sidecar.exists()
    truth = [json.loads(line)["t"] for line in sidecar.read_text().splitlines()]
    src = ReplaySource(str(recording), realtime=False)
    src.start()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.stop()
    assert len(frames) == 120
    assert [f.t for f in frames] == truth


def test_replay_is_deterministic_and_comparator_passes(recording, tmp_path):
    csv_a = run_replay(recording, tmp_path / "a")
    csv_b = run_replay(recording, tmp_path / "b")
    report = compare_csv(csv_a, csv_b)
    assert report.frames == 120
    assert report.tracked_a > 60          # the pipeline genuinely tracked
    assert report.valid_mismatch_frac == 0.0
    assert report.mean_px_dev == 0.0      # same frames + same config = identical
    assert report.passed()


def test_comparator_catches_behavior_drift(recording, tmp_path):
    csv_a = run_replay(recording, tmp_path / "a")
    # simulate a code change that shifts tracking and drops some frames' lock
    rows = list(csv.DictReader(open(csv_a)))
    drifted = tmp_path / "drifted.csv"
    for i, r in enumerate(rows):
        if r["valid"] == "1" and r["px_x"]:
            r["px_x"] = str(float(r["px_x"]) + 8.0)      # 8px systematic shift
        if i % 10 == 0:
            r["valid"] = "0" if r["valid"] == "1" else "1"
    with open(drifted, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    report = compare_csv(csv_a, drifted)
    assert report.valid_mismatch_frac > 0.05
    assert report.mean_px_dev > 3.0
    assert not report.passed()


def test_config_override_changes_are_visible(recording, tmp_path):
    """The harness must honor overrides — that's how a tuning change gets
    evaluated against a recording before it's accepted."""
    csv_a = run_replay(recording, tmp_path / "a")
    csv_c = run_replay(recording, tmp_path / "c",
                       overrides={"detection": {"frame_diff": {"threshold": 250}}})
    report = compare_csv(csv_a, csv_c)
    # threshold 250 kills every detection -> massive valid mismatch
    assert report.tracked_b == 0
    assert not report.passed()
