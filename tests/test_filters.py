"""Filter must converge on synthetic constant-velocity motion (SPEC 8 target:
velocity estimate settles in <10 frames) and coast sanely."""
import math

from turretvision.track.filters import AlphaBetaFilter


def test_constant_velocity_convergence():
    f = AlphaBetaFilter(alpha=0.5, beta=0.2)
    vx_true, vy_true = 120.0, -40.0  # px/s
    dt = 1 / 60
    s = None
    for i in range(30):
        t = i * dt
        s = f.update(10 + vx_true * t, 300 + vy_true * t, t)
    assert s is not None
    assert math.isclose(s.vx, vx_true, rel_tol=0.05)
    assert math.isclose(s.vy, vy_true, rel_tol=0.05)


def test_velocity_settles_within_10_frames():
    f = AlphaBetaFilter(alpha=0.5, beta=0.2)
    vx_true = 200.0
    dt = 1 / 60
    for i in range(10):
        s = f.update(vx_true * i * dt, 0.0, i * dt)
    assert abs(s.vx - vx_true) / vx_true < 0.15  # within 15% by frame 10


def test_predict_extrapolates():
    f = AlphaBetaFilter(alpha=0.5, beta=0.2)
    dt = 1 / 60
    for i in range(20):
        f.update(100.0 * i * dt, 0.0, i * dt)
    p = f.predict(20 * dt)
    expected = 100.0 * 20 * dt
    assert abs(p.x - expected) < 3.0


def test_noise_rejection():
    """Filtered jitter must be smaller than measurement jitter, else the filter
    is decorative."""
    import random
    random.seed(1)
    f = AlphaBetaFilter(alpha=0.4, beta=0.15)
    dt = 1 / 60
    errs = []
    for i in range(120):
        t = i * dt
        true_x = 50.0 * t
        z = true_x + random.gauss(0, 3.0)
        s = f.update(z, 0.0, t)
        if i > 30:
            errs.append(abs(s.x - true_x))
    mean_err = sum(errs) / len(errs)
    assert mean_err < 3.0  # below the 3px measurement sigma
