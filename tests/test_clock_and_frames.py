import numpy as np

from turret.link.clock import ClockSync
from turret.link.messages import FrameSync
from turret.runtime.frames import FrameMatcher


def test_clock_sync_tracks_offset_and_drift_through_asymmetric_jitter():
    rng = np.random.default_rng(0)
    offset, drift = 1234.5, 40e-6
    cs = ClockSync()

    def esp(t):
        return t * (1 + drift) + offset

    for i in range(400):
        t1 = 100.0 + i * 0.25
        up = 0.0004 + rng.exponential(0.0008)  # USB delay: a floor plus one-sided jitter
        down = 0.0006 + rng.exponential(0.0015)
        t2 = esp(t1 + up)
        t3 = t2 + 30e-6
        t4 = t1 + up + 30e-6 / (1 + drift) + down
        cs.add(t1, int(t2 * 1e6), int(t3 * 1e6), t4)
    t = 200.0
    est = cs.esp_to_jetson(esp(t) * 1e6)
    assert abs(est - t) < 3e-4  # well under the 0.5 ms target
    # Drift only matters over the ~1 s between pings, so 10 ppm error is ~10 us: harmless.
    assert abs(cs.drift_ppm - 40.0) < 10.0
    assert abs(cs.jetson_to_esp_us(t) * 1e-6 - esp(t)) < 3e-4


def test_frame_matcher_locks_and_survives_drops():
    rng = np.random.default_rng(1)
    m = FrameMatcher(lambda t_us: t_us * 1e-6)  # identity clock for the test
    period = 1 / 90
    trig_seq0, cam_seq0 = 5000, 17
    results = {}
    for i in range(300):
        t_trig = 10.0 + i * period
        m.add_sync(FrameSync(seq=trig_seq0 + i, t_us=int(t_trig * 1e6)))
        if rng.random() < 0.05:
            continue  # frame lost on USB
        arrival = t_trig + 0.009 + rng.random() * 0.003
        res = m.match(cam_seq0 + i, arrival, 0.0)
        results[i] = res
    good = [i for i, r in results.items() if isinstance(r, FrameSync)]
    assert m.locked
    assert len(good) >= len(results) - 1
    for i in good:
        assert results[i].seq == trig_seq0 + i  # every match is the right trigger
