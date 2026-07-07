"""Ego-motion compensation (Phase 4.5): a panning camera must NOT read as
motion, while a genuinely moving target still must. Pose interpolation too."""
import numpy as np

from turretvision.capture.base import Frame
from turretvision.detect.frame_diff import FrameDiffDetector
from turretvision.link.base import Telemetry
from turretvision.link.pose import PoseHistory

PX_PER_DEG = 10.0
W, H = 400, 300


def scene(rng):
    """Static high-texture background — worst case for differencing under pan."""
    return rng.integers(0, 200, (H, W), dtype=np.uint8)


def frame_pair_with_pan(pan_px: int, blob_move: bool):
    """Two frames of the same scene: camera panned right by pan_px between
    them (scene shifts LEFT), optional genuinely-moving bright blob."""
    rng = np.random.default_rng(3)
    base = scene(rng)
    f1 = base.copy()
    f2 = np.roll(base, -pan_px, axis=1)   # camera pans right -> scene goes left
    if blob_move:
        import cv2
        cv2.circle(f1, (150, 150), 12, 255, -1)
        cv2.circle(f2, (190, 150), 12, 255, -1)   # 40px of real motion
    import cv2
    return (Frame(cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR), t=0.00, idx=0),
            Frame(cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR), t=0.01, idx=1))


def make_detector(ego: bool):
    d = FrameDiffDetector(threshold=25, min_area_px=40, max_area_px=40000,
                          ego_motion_comp=ego)
    d.px_per_deg = PX_PER_DEG
    return d


def test_pan_without_comp_floods_with_false_motion():
    f1, f2 = frame_pair_with_pan(pan_px=20, blob_move=False)
    d = make_detector(ego=False)
    d.detect(f1)
    dets = d.detect(f2)
    assert len(dets) > 0     # the whole frame reads as motion — the v1 problem


def test_pan_with_comp_is_silent():
    f1, f2 = frame_pair_with_pan(pan_px=20, blob_move=False)
    d = make_detector(ego=True)
    d.set_ego_pose(0.0, 0.0)
    d.detect(f1)
    d.set_ego_pose(20 / PX_PER_DEG, 0.0)   # camera yawed right by exactly the pan
    dets = d.detect(f2)
    assert dets == []        # ego motion fully cancelled


def test_comp_still_sees_the_real_target_mid_slew():
    f1, f2 = frame_pair_with_pan(pan_px=20, blob_move=True)
    d = make_detector(ego=True)
    d.set_ego_pose(0.0, 0.0)
    d.detect(f1)
    d.set_ego_pose(20 / PX_PER_DEG, 0.0)
    d.set_turret_rate(200.0)   # gate would have killed this; comp must bypass it
    dets = d.detect(f2)
    assert len(dets) >= 1
    # biggest detection sits on the blob's motion (old/new positions after the
    # pan shift: 150-20=130 .. 190-20=170)
    assert 110 <= dets[0].cx <= 190
    assert 130 <= dets[0].cy <= 170


def test_gate_still_applies_when_no_pose_available():
    f1, f2 = frame_pair_with_pan(pan_px=20, blob_move=True)
    d = make_detector(ego=True)          # comp enabled but no pose ever fed
    d.set_turret_rate(200.0)
    d.detect(f1)
    assert d.detect(f2) == []            # falls back to the quasi-static gate


def test_zero_delta_comp_matches_plain_differencing():
    f1, f2 = frame_pair_with_pan(pan_px=0, blob_move=True)
    d_plain, d_comp = make_detector(False), make_detector(True)
    d_comp.set_ego_pose(5.0, 2.0)
    d_plain.detect(f1)
    d_comp.detect(f1)
    d_comp.set_ego_pose(5.0, 2.0)        # pose unchanged -> zero delta
    a, b = d_plain.detect(f2), d_comp.detect(f2)
    # both paths see the same thing (two-blob ghosting incl. — see frame_diff
    # docstring), byte-identical since zero delta means no warp at all
    assert len(a) == len(b) > 0
    for da, db in zip(a, b, strict=True):
        assert abs(da.cx - db.cx) < 1e-6 and abs(da.cy - db.cy) < 1e-6


def telem(t, yaw, pitch=0.0):
    return Telemetry(t_esp_ms=int(t * 1000), yaw_deg=yaw, pitch_deg=pitch,
                     yaw_rate_dps=0.0, status=1, t_rx=t)


def test_pose_history_interpolates_and_clamps():
    h = PoseHistory()
    assert h.pose_at(1.0) is None
    h.add(telem(1.0, 10.0, 1.0))
    h.add(telem(1.1, 20.0, 2.0))
    yaw, pitch = h.pose_at(1.05)
    assert abs(yaw - 15.0) < 1e-9 and abs(pitch - 1.5) < 1e-9
    assert h.pose_at(0.5) == (10.0, 1.0)     # clamp, don't extrapolate
    assert h.pose_at(9.9) == (20.0, 2.0)
    h.add(telem(1.1, 99.0))                  # duplicate timestamp ignored
    assert len(h) == 2
