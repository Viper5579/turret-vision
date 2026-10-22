"""The whole stack against the simulator, scored against ground truth.

This is the regression gate: if a change makes tracking worse, CI fails here.
Thresholds are deliberately looser than current results so noise does not
flake the test, but a real regression trips them.
"""

from turret.config import Config
from turret.runtime.pipeline import Mode
from turret.runtime.runner import SimRunner


def test_tracking_loop_in_sim():
    cfg = Config()
    r = SimRunner(cfg, mode=Mode.TRACK)
    s = r.run_headless(8.0).summary()
    st = r.pipeline.stats()
    assert st["unmatched"] <= 2, st  # every frame tied to its trigger
    assert s["throws"] >= 3, s
    assert s["visible_frames"] > 60, s
    assert s["track_coverage"] > 0.9, s
    assert s["bearing_err_deg_p50"] < 0.5, s
    assert s["false_track_frames"] <= 2, s
    assert s["sync_residual_ms"] < 0.5, s


def test_manual_mode_moves_the_turret():
    cfg = Config()
    r = SimRunner(cfg, mode=Mode.MANUAL)
    r.pipeline.bus.command({"t": "jog", "yaw": 45.0, "pitch": 0.0})
    r.run_headless(2.0)
    assert r.world.esp.truth.yaw_deg > 30.0


def test_estop_stops_motion():
    cfg = Config()
    r = SimRunner(cfg, mode=Mode.MANUAL)
    r.pipeline.bus.command({"t": "jog", "yaw": 90.0, "pitch": 0.0})
    r.run_headless(1.0)
    r.pipeline.bus.command({"t": "estop", "on": True})
    r.run_headless(0.3)
    yaw = r.world.esp.truth.yaw_deg
    r.run_headless(0.5)
    assert abs(r.world.esp.truth.yaw_deg - yaw) < 0.5
    assert abs(r.world.esp.truth.yaw_rate_dps) < 1.0
