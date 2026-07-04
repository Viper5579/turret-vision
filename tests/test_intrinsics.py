"""Intrinsics calibration solved against synthetic views of the ChArUco board
projected through a KNOWN camera — the solve must recover that camera. Also
covers detection on a rendered board image and the yaml round-trip into
PixelAngleMapper."""
import cv2
import numpy as np

from turretvision.calib import intrinsics
from turretvision.calib.geometry import PixelAngleMapper

K_TRUE = np.array([[900.0, 0, 640.0], [0, 900.0, 400.0], [0, 0, 1.0]])
D_TRUE = np.array([0.05, -0.10, 0.0, 0.0, 0.0])
SIZE = (1280, 800)


def _synthetic_views(n=10):
    board = intrinsics.make_board()
    obj = board.getChessboardCorners().astype(np.float64)   # Nx3 board frame, meters
    rng = np.random.default_rng(7)
    views = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                         rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.10, 0.02), rng.uniform(-0.08, 0.02),
                         rng.uniform(0.35, 0.6)])
        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, D_TRUE)
        pts = img_pts.reshape(-1, 2)
        views.append(intrinsics.BoardView(
            obj_points=obj, img_points=pts,
            center_px=(float(pts[:, 0].mean()), float(pts[:, 1].mean()))))
    return views


def test_calibration_recovers_known_camera():
    result = intrinsics.calibrate(_synthetic_views(), SIZE)
    K = result.camera_matrix
    assert abs(K[0, 0] - 900.0) / 900.0 < 0.01      # fx within 1%
    assert abs(K[0, 2] - 640.0) < 10.0              # principal point
    assert result.reprojection_error_px < 0.1       # clean synthetic data
    assert len(result.per_view_error_px) == result.n_views == 10


def test_calibrate_refuses_too_few_views():
    import pytest
    with pytest.raises(ValueError):
        intrinsics.calibrate(_synthetic_views(2), SIZE)


def test_detect_view_on_rendered_board():
    board = intrinsics.make_board()
    n = intrinsics.BOARD_SQUARES
    img = board.generateImage((n[0] * 120, n[1] * 120), marginSize=30)
    view = intrinsics.detect_view(img, board)
    assert view is not None
    assert len(view.obj_points) >= (n[0] - 1) * (n[1] - 1) - 2  # nearly all corners


def test_save_load_feeds_pixel_angle_mapper(tmp_path):
    result = intrinsics.calibrate(_synthetic_views(), SIZE)
    path = intrinsics.save(result, tmp_path / "camera_intrinsics.yaml")
    mapper = PixelAngleMapper(*SIZE, intrinsics_file=str(path))
    assert mapper.calibrated
    # optical center must map to (0,0) angles through the calibrated K
    az, el = mapper.pixel_to_angles(result.camera_matrix[0, 2], result.camera_matrix[1, 2])
    assert abs(az) < 1e-6 and abs(el) < 1e-6


def test_boresight_file_offsets_apply(tmp_path):
    bs = tmp_path / "boresight.yaml"
    bs.write_text("boresight_yaw_deg: 1.5\nboresight_pitch_deg: -0.75\n")
    m = PixelAngleMapper(1280, 800, boresight_file=str(bs))
    m0 = PixelAngleMapper(1280, 800)
    az, el = m.pixel_to_angles(640, 400)
    az0, el0 = m0.pixel_to_angles(640, 400)
    assert abs((az - az0) - 1.5) < 1e-9
    assert abs((el - el0) + 0.75) < 1e-9
