import cv2
import numpy as np
import pytest

from turret.config import Config, MotionCfg
from turret.geometry import (
    azel_to_pixel,
    cam_to_base,
    intrinsics_from_fov,
    pixel_to_azel,
    rotation_homography,
    scale_intrinsics,
)
from turret.sim.plane import Room
from turret.sim.render import Renderer
from turret.vision.mosaic import BackgroundMosaic
from turret.vision.motion import MotionDetector


@pytest.fixture(scope="module")
def renderer():
    c = Config().camera
    return Renderer(c.width // 2, c.height // 2, c.hfov_deg, c.tilt_deg, Room(), seed=3, noise=0.0)


def test_cam_to_base_is_a_rotation():
    R = cam_to_base(37.0, 12.0)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_pixel_azel_roundtrip():
    K = intrinsics_from_fov(1280, 800, 70.0)
    R = cam_to_base(-20.0, 10.0)
    for u, v in [(640, 400), (10, 20), (1270, 790), (300, 600)]:
        az, el = pixel_to_azel(u, v, K, R)
        assert np.allclose(azel_to_pixel(az, el, K, R), (u, v), atol=1e-6)
    az, el = pixel_to_azel(639.5, 399.5, K, cam_to_base(25.0, 10.0))
    assert abs(az - 25.0) < 1e-6 and abs(el - 10.0) < 1e-6


def test_rotation_homography_matches_rendered_rotation(renderer):
    """The warp the pipeline uses must line up two renders of the room at different yaws."""
    K = renderer.K
    a = cv2.cvtColor(renderer.background(0.0), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(renderer.background(1.3), cv2.COLOR_BGR2GRAY)
    H = rotation_homography(K, cam_to_base(0.0, renderer.tilt), cam_to_base(1.3, renderer.tilt))
    w = cv2.warpPerspective(a, H, (a.shape[1], a.shape[0]))
    inner = (slice(40, -40), slice(80, -80))
    assert cv2.absdiff(w, b)[inner].mean() < 2.0
    assert cv2.absdiff(a, b)[inner].mean() > 4.0  # sanity: unwarped really is different


def _frames(renderer, yaws, square_at=None):
    out = []
    for i, yaw in enumerate(yaws):
        g = cv2.GaussianBlur(cv2.cvtColor(renderer.background(yaw), cv2.COLOR_BGR2GRAY), (3, 3), 0.8)
        if square_at is not None:
            x, y = square_at(i)
            cv2.rectangle(g, (x, y), (x + 12, y + 8), 250, -1)
        out.append(g)
    return out


def test_motion_detector_silent_while_panning_and_finds_mover(renderer):
    K = renderer.K
    tilt = renderer.tilt
    det = MotionDetector(MotionCfg(), decode_scale=1)
    yaws = [0.0, 0.7, 1.4, 2.1]
    frames = _frames(renderer, yaws)
    Rs = [cam_to_base(y, tilt) for y in yaws]

    def run(fr):
        return det.detect(
            fr[3], fr[2], rotation_homography(K, Rs[2], Rs[3]), fr[1], rotation_homography(K, Rs[1], Rs[3])
        )

    assert run(frames) == []  # pure pan of a static room: nothing moves
    mover = _frames(renderer, yaws, square_at=lambda i: (200 + 25 * i, 150))
    blobs = run(mover)
    assert len(blobs) == 1
    assert abs(blobs[0].cx - (200 + 75 + 6)) < 10


def test_mosaic_predicts_the_room_at_a_new_yaw(renderer):
    K = renderer.K
    h, w = renderer.h, renderer.w
    mos = BackgroundMosaic(K, w, h, renderer.tilt)
    for yaw in np.arange(-10, 10.1, 1.0):
        g = cv2.cvtColor(renderer.background(float(yaw)), cv2.COLOR_BGR2GRAY)
        mos.update(g, float(yaw))
    truth = cv2.cvtColor(renderer.background(3.37), cv2.COLOR_BGR2GRAY)
    pred, known = mos.predict(3.37)
    inner = (slice(30, -30), slice(60, -60))
    assert (known[inner] > 0).all()
    assert cv2.absdiff(pred, truth)[inner].mean() < 4.0


def test_scale_intrinsics_maps_pixel_centers():
    K = intrinsics_from_fov(1280, 800, 70.0)
    Ks = scale_intrinsics(K, 0.5)
    # Full-res pixel (u, v) center lands at ((u + 0.5) / 2 - 0.5) in the half-res image.
    p = K @ np.array([0.1, -0.05, 1.0])
    ps = Ks @ np.array([0.1, -0.05, 1.0])
    assert np.allclose(ps[:2], (p[:2] + 0.5) / 2 - 0.5)
