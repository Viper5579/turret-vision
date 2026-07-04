"""Lead predictor: hand-checkable intercept geometry (D7).

The crossing-target case has a closed-form answer (lead = asin(v_target/v_proj))
and the gravity drop is atan(g*t^2 / (2r)) — both verified against the solver.
"""
import math

from turretvision.lead.predictor import LeadPredictor


def test_stationary_target_tof_and_gravity_drop():
    p = LeadPredictor(projectile_speed_mps=50.0, gravity_comp=True)
    s = p.solve(az_deg=10.0, el_deg=5.0, az_rate_dps=0.0, el_rate_dps=0.0, range_m=5.0)
    assert abs(s.tof_s - 0.1) < 1e-9                  # 5m at 50m/s
    assert abs(s.yaw_deg - 10.0) < 1e-9               # no lateral lead
    drop = math.degrees(math.atan2(0.5 * 9.81 * 0.1 ** 2, 5.0))
    assert abs(s.pitch_deg - (5.0 + drop)) < 1e-6     # aim high by the drop
    assert not s.clamped


def test_gravity_off_means_pitch_unchanged_for_stationary():
    p = LeadPredictor(projectile_speed_mps=50.0, gravity_comp=False)
    s = p.solve(0.0, 5.0, 0.0, 0.0, 5.0)
    assert abs(s.pitch_deg - 5.0) < 1e-9


def test_crossing_target_matches_closed_form():
    # v_t = 5 m/s at r=10m -> az_rate = 5/10 rad/s; textbook lead = asin(vt/vp)
    p = LeadPredictor(projectile_speed_mps=50.0, gravity_comp=False)
    s = p.solve(0.0, 0.0, math.degrees(0.5), 0.0, 10.0)
    assert abs(s.yaw_deg - math.degrees(math.asin(5.0 / 50.0))) < 0.01
    assert s.yaw_deg > 0                             # leads INTO the motion
    s_neg = p.solve(0.0, 0.0, -math.degrees(0.5), 0.0, 10.0)
    assert abs(s_neg.yaw_deg + s.yaw_deg) < 1e-9     # symmetric


def test_lead_clamp_flags_bad_inputs():
    p = LeadPredictor(projectile_speed_mps=50.0, max_lead_deg=15.0, gravity_comp=False)
    s = p.solve(0.0, 0.0, 200.0, 0.0, 10.0)          # rate spike from a noisy frame
    assert s.clamped
    assert abs(s.lead_deg - 15.0) < 1e-9
    assert abs(s.yaw_deg) <= 15.0 + 1e-9


def test_target_outrunning_projectile_returns_none():
    p = LeadPredictor(projectile_speed_mps=50.0)
    # tangential 60 m/s > projectile 50 m/s: no intercept exists
    assert p.solve(0.0, 0.0, math.degrees(6.0), 0.0, 10.0) is None


def test_degenerate_inputs_return_none():
    assert LeadPredictor(projectile_speed_mps=0.0).solve(0, 0, 0, 0, 5.0) is None
    assert LeadPredictor().solve(0, 0, 0, 0, 0.0) is None
