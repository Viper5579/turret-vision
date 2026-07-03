"""End-to-end (SPEC Phase 2 exit criterion, hardware-free half): synthetic
moving target -> frame_diff -> tracker must stay near ground truth.

WHY this test matters more than the unit tests: it exercises the real seams
(detector ghosting, association gating, filter lag) together. Tolerance is
loose on purpose -- two-frame differencing has known centroid wobble (see
frame_diff.py docstring); we're asserting 'locked onto the right object',
not sub-pixel accuracy.
"""
import math

from turretvision.capture.synthetic import SyntheticSource
from turretvision.detect.frame_diff import FrameDiffDetector
from turretvision.track.filters import AlphaBetaFilter
from turretvision.track.tracker import SingleTargetTracker


def test_tracks_synthetic_target():
    src = SyntheticSource(n_frames=200, fps=60, speed_px=8.0)
    src.start()
    detector = FrameDiffDetector(threshold=25, min_area_px=40, max_area_px=40000)
    tracker = SingleTargetTracker(AlphaBetaFilter(0.5, 0.2), gate_px=120, max_coast_frames=8)

    errs = []
    tracked_frames = 0
    total = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        total += 1
        s = tracker.step(detector.detect(frame), frame.t)
        if total < 15:  # ignore acquisition transient
            continue
        _, gx, gy = src.truth[frame.idx]
        # target wraps horizontally each lap; skip scoring the teleport frames
        if frame.idx > 0 and abs(src.truth[frame.idx][1] - src.truth[frame.idx - 1][1]) > 50:
            continue
        if s is not None and not s.coasting:
            tracked_frames += 1
            errs.append(math.hypot(s.x - gx, s.y - gy))

    assert total == 200
    assert tracked_frames / total > 0.75          # detects+tracks most frames
    assert sum(errs) / len(errs) < 30.0           # mean err well under 2 blob diameters
