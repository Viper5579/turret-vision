"""ChArUco intrinsic calibration: detection, solve, save/load (SPEC 6.2).

WHY ChArUco over a plain chessboard: it works with partial board views and
every corner carries an ID, so a frame where half the board is cut off still
contributes good points instead of poisoning the solve with mismatched
correspondences.

WHY this lives in the library and not in tools/calibrate_camera.py: the tool
is a thin capture loop; the geometry (detect corners -> accumulate views ->
calibrate -> reprojection error -> yaml) is pure logic that gets unit-tested
against synthetically projected boards with a KNOWN camera matrix.

The saved YAML is exactly what calib/geometry.PixelAngleMapper loads
(camera_matrix + dist_coeffs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# Board geometry defaults — match the printable board from make_board().
BOARD_SQUARES = (7, 5)          # columns, rows
BOARD_SQUARE_M = 0.035          # printed square side
BOARD_MARKER_M = 0.026          # marker side (must be < square)
BOARD_DICT = "DICT_4X4_50"


def make_board(squares: tuple[int, int] = BOARD_SQUARES,
               square_m: float = BOARD_SQUARE_M,
               marker_m: float = BOARD_MARKER_M,
               dictionary: str = BOARD_DICT) -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.CharucoBoard(squares, square_m, marker_m, d)


@dataclass
class BoardView:
    """One usable observation of the board: matched 3D<->2D correspondences."""
    obj_points: np.ndarray      # Nx3 board-frame meters
    img_points: np.ndarray      # Nx2 pixels
    center_px: tuple[float, float]


def detect_view(gray: np.ndarray, board: cv2.aruco.CharucoBoard,
                min_corners: int = 12) -> BoardView | None:
    """Detect ChArUco corners in one frame; None if too few for a stable pose.
    WHY a minimum: 4 corners technically solve a homography, but near-degenerate
    views dominate the residual and drag the whole calibration sideways."""
    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_corners is None or ch_ids is None or len(ch_corners) < min_corners:
        return None
    obj, img = board.matchImagePoints(ch_corners, ch_ids)
    if obj is None or len(obj) < min_corners:
        return None
    pts = img.reshape(-1, 2)
    return BoardView(obj_points=obj.reshape(-1, 3).astype(np.float64),
                     img_points=pts.astype(np.float64),
                     center_px=(float(pts[:, 0].mean()), float(pts[:, 1].mean())))


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]       # (w, h)
    reprojection_error_px: float      # mean over all points
    per_view_error_px: list[float]
    n_views: int


def calibrate(views: list[BoardView], image_size: tuple[int, int]) -> CalibrationResult:
    if len(views) < 4:
        raise ValueError(f"need at least 4 views, got {len(views)}")
    obj = [v.obj_points.astype(np.float32) for v in views]
    img = [v.img_points.astype(np.float32).reshape(-1, 1, 2) for v in views]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None)
    per_view = []
    total_err = 0.0
    total_pts = 0
    for o, i, r, t in zip(obj, img, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1)
        per_view.append(float(err.mean()))
        total_err += float(err.sum())
        total_pts += len(err)
    return CalibrationResult(camera_matrix=K, dist_coeffs=D.reshape(-1),
                             image_size=image_size,
                             reprojection_error_px=total_err / total_pts,
                             per_view_error_px=per_view, n_views=len(views))


def save(result: CalibrationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "camera_matrix": result.camera_matrix.tolist(),
        "dist_coeffs": result.dist_coeffs.tolist(),
        "image_size": list(result.image_size),
        "reprojection_error_px": round(result.reprojection_error_px, 4),
        "n_views": result.n_views,
    }
    header = ("# Camera intrinsics from tools/calibrate_camera.py (ChArUco).\n"
              "# Loaded by calib/geometry.PixelAngleMapper. Redo after ANY change\n"
              "# to lens focus, lens position, or resolution.\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(data, f)
    return path


def load(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
