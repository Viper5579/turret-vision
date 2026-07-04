"""Range estimators: fixed, known-size pinhole math, ArUco solvePnP against a
synthetically projected marker with known ground-truth distance."""
import numpy as np

from turretvision.detect.base import Detection
from turretvision.ranging.aruco_pose import ArucoPoseRange
from turretvision.ranging.base import nearest_detection
from turretvision.ranging.fixed import FixedRange
from turretvision.ranging.known_size import KnownSizeRange


def det(cx, cy, w=60, h=20, corners=None):
    return Detection(cx=cx, cy=cy, area=float(w * h), bbox=(int(cx - w / 2), int(cy - h / 2), w, h),
                     kind="test", t=0.0, corners=corners)


def test_fixed_returns_configured_constant():
    r = FixedRange(fixed_distance_m=4.0).estimate([], None)
    assert r.dist_m == 4.0 and r.method == "fixed"
    assert r.sigma_m > 0.5  # honest: it's a guess


def test_known_size_pinhole_math():
    # f=800px, 0.30m object spanning 60px -> 800*0.30/60 = 4.0m exactly
    r = KnownSizeRange(focal_px=800.0, target_size_m=0.30)
    est = r.estimate([det(100, 100, w=60, h=20)], (100, 100))
    assert abs(est.dist_m - 4.0) < 1e-9
    # larger bbox side wins (banking airplane forshortens one axis)
    est2 = r.estimate([det(100, 100, w=20, h=60)], (100, 100))
    assert abs(est2.dist_m - 4.0) < 1e-9


def test_known_size_rejects_speck_blobs():
    r = KnownSizeRange(focal_px=800.0, target_size_m=0.30, min_px=4.0)
    assert r.estimate([det(100, 100, w=2, h=2)], (100, 100)) is None


def test_nearest_detection_binds_to_the_track():
    far, near = det(500, 500), det(110, 100)
    assert nearest_detection([far, near], (100, 100)) is near
    assert nearest_detection([far], (100, 100)) is None       # outside gate
    assert nearest_detection([], (100, 100)) is None


def test_aruco_pose_recovers_known_distance():
    fx, cx_, cy_ = 800.0, 640.0, 400.0
    K = np.array([[fx, 0, cx_], [0, fx, cy_], [0, 0, 1.0]])
    size, z = 0.10, 3.0
    s = size / 2
    # project corners (aruco order tl,tr,br,bl; +y up in marker frame) at z=3m
    world = [(-s, s), (s, s), (s, -s), (-s, -s)]
    corners = [(cx_ + fx * x / z, cy_ - fx * y / z) for x, y in world]
    cx = float(np.mean([c[0] for c in corners]))
    cy = float(np.mean([c[1] for c in corners]))
    d = det(cx, cy, corners=corners)
    est = ArucoPoseRange(K, None, marker_size_m=size).estimate([d], (cx, cy))
    assert est is not None and est.method == "aruco_pose"
    assert abs(est.dist_m - z) / z < 0.01     # within 1% on clean synthetic data


def test_aruco_pose_needs_corners():
    K = np.eye(3)
    assert ArucoPoseRange(K, None).estimate([det(10, 10)], (10, 10)) is None
