from turretvision.detect.base import Detection
from turretvision.track.filters import AlphaBetaFilter
from turretvision.track.tracker import SingleTargetTracker


def det(x, y, t, area=100.0):
    return Detection(cx=x, cy=y, area=area, bbox=(0, 0, 1, 1), kind="test", t=t)


def make_tracker(max_coast=5):
    return SingleTargetTracker(AlphaBetaFilter(0.5, 0.2), gate_px=100,
                               max_coast_frames=max_coast)


def test_acquires_and_tracks():
    tr = make_tracker()
    dt = 1 / 60
    s = None
    for i in range(30):
        s = tr.step([det(100 + 5 * i, 200, i * dt)], i * dt)
    assert s is not None and s.valid and not s.coasting
    assert s.confidence > 0.8


def test_coasts_through_gap_then_reacquires():
    tr = make_tracker(max_coast=5)
    dt = 1 / 60
    for i in range(20):
        tr.step([det(100 + 5 * i, 200, i * dt)], i * dt)
    # 3-frame dropout: must coast (predict), not die
    for i in range(20, 23):
        s = tr.step([], i * dt)
        assert s is not None and s.coasting
    s = tr.step([det(100 + 5 * 23, 200, 23 * dt)], 23 * dt)
    assert s is not None and not s.coasting


def test_drops_track_after_max_coast():
    """WHY: coasting forever means confidently aiming at a ghost."""
    tr = make_tracker(max_coast=3)
    dt = 1 / 60
    for i in range(10):
        tr.step([det(100, 200, i * dt)], i * dt)
    out = None
    for i in range(10, 20):
        out = tr.step([], i * dt)
    assert out is None


def test_gate_rejects_far_clutter():
    tr = make_tracker()
    dt = 1 / 60
    for i in range(20):
        tr.step([det(100 + 5 * i, 200, i * dt)], i * dt)
    # clutter blob 500px away must not steal the track
    s = tr.step([det(600, 500, 20 * dt, area=9999)], 20 * dt)
    assert s is not None and s.coasting  # coasted instead of jumping to clutter
