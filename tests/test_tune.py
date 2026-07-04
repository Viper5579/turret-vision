"""Tuning stack: config overlay merge/save, param registry semantics, and an
HTTP round-trip against a real TuningServer (ephemeral port, no camera)."""
import json
import urllib.request

import numpy as np
import yaml

from turretvision.detect.frame_diff import FrameDiffDetector
from turretvision.track.filters import AlphaBetaFilter
from turretvision.track.tracker import SingleTargetTracker
from turretvision.tune.params import build_registry
from turretvision.tune.server import TuningServer
from turretvision.util.config import Config, save_overlay


def _pipeline():
    det = FrameDiffDetector(threshold=25, min_area_px=40)
    trk = SingleTargetTracker(AlphaBetaFilter(0.5, 0.2), gate_px=120, max_coast_frames=8)
    ref = {"value": 0.4}
    return det, trk, ref, build_registry(det, trk.estimator, trk, ref)


def test_config_overlay_merges_over_base(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "detection:\n  frame_diff:\n    threshold: 25\n    min_area_px: 40\n")
    (tmp_path / "local.yaml").write_text(
        "detection:\n  frame_diff:\n    threshold: 60\n")
    cfg = Config.load(tmp_path / "base.yaml")
    assert cfg.get("detection.frame_diff.threshold") == 60   # overridden
    assert cfg.get("detection.frame_diff.min_area_px") == 40  # preserved


def test_save_overlay_roundtrip_preserves_foreign_keys(tmp_path):
    base = tmp_path / "default.yaml"
    base.write_text("tracking:\n  alpha: 0.5\n")
    (tmp_path / "local.yaml").write_text("camera:\n  device: /dev/video9\n")
    save_overlay(base, {"tracking.alpha": 0.8, "detection.frame_diff.threshold": 42})
    data = yaml.safe_load((tmp_path / "local.yaml").read_text())
    assert data["tracking"]["alpha"] == 0.8
    assert data["detection"]["frame_diff"]["threshold"] == 42
    assert data["camera"]["device"] == "/dev/video9"  # hand-edits survive a save
    cfg = Config.load(base)
    assert cfg.get("tracking.alpha") == 0.8


def test_registry_applies_only_from_pipeline_side():
    det, trk, ref, reg = _pipeline()
    reg.queue("detection.frame_diff.threshold", 55)
    reg.queue("tracking.alpha", 0.9)
    reg.queue("tracking.min_confidence_output", 0.7)
    assert det.threshold == 25          # nothing moves until the pipeline applies
    reg.apply_pending()
    assert det.threshold == 55
    assert trk.estimator.alpha == 0.9
    assert ref["value"] == 0.7


def test_registry_coerces_type_and_clamps_bounds():
    det, _, _, reg = _pipeline()
    assert reg.queue("detection.frame_diff.threshold", "999") == 120  # clamped to max
    assert reg.queue("detection.frame_diff.morph_kernel", 5.6) == 6  # int rounding
    reg.apply_pending()
    assert det.threshold == 120
    assert det.morph_kernel == 6
    assert det._kernel.shape == (6, 6)  # derived state rebuilt by the setter


def test_http_roundtrip_state_set_save(tmp_path):
    base = tmp_path / "default.yaml"
    base.write_text("tracking:\n  alpha: 0.5\n")
    det, _, _, reg = _pipeline()
    srv = TuningServer(reg, config_path=str(base), port=0)
    srv.start()
    try:
        url = f"http://127.0.0.1:{srv.port}"
        srv.publish(np.zeros((60, 80, 3), np.uint8),
                    {"fps": 30.0, "n_dets": 0, "detect_ms": 1.0, "track_ms": 0.1,
                     "tracking": False, "coasting": False,
                     "az": 0.0, "el": 0.0, "az_rate": 0.0, "el_rate": 0.0, "conf": 0.0})

        state = json.load(urllib.request.urlopen(f"{url}/api/state", timeout=5))
        keys = {p["key"] for p in state["params"]}
        assert "detection.frame_diff.threshold" in keys
        assert "tracking.alpha" in keys
        assert state["stats"]["fps"] == 30.0

        req = urllib.request.Request(
            f"{url}/api/set", method="POST",
            data=json.dumps({"key": "detection.frame_diff.threshold", "value": 77}).encode(),
            headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req, timeout=5))["value"] == 77
        srv.apply_pending()
        assert det.threshold == 77

        req = urllib.request.Request(f"{url}/api/save", method="POST", data=b"{}")
        saved = json.load(urllib.request.urlopen(req, timeout=5))["saved"]
        data = yaml.safe_load(open(saved))
        assert data["detection"]["frame_diff"]["threshold"] == 77

        jpeg, seq = srv.wait_frame_jpeg(last_seq=0)
        assert seq == 1 and jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    finally:
        srv.stop()
