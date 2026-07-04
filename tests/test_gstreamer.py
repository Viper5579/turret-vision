"""GStreamer backend: pipeline string construction and open-attempt ordering.
No GStreamer runtime needed — the generic OpenCV here fails to open any gst
pipeline, which is exactly the error path we assert on."""
import pytest

from turretvision.capture.gstreamer import HW_DECODE, GstCamera, build_pipeline


def test_pipeline_string_hw_decode_and_freshness_policy():
    p = build_pipeline("/dev/video0", 1280, 800, 100)
    assert p.startswith("v4l2src device=/dev/video0")
    assert "image/jpeg,width=1280,height=800,framerate=100/1" in p
    assert HW_DECODE in p                      # Jetson hardware decode block
    assert "appsink drop=true max-buffers=1" in p  # newest-frame-only, like v4l2


def test_custom_pipeline_overrides_builtin():
    cam = GstCamera("/dev/video0", 1280, 800, 100, pipeline="my custom ! appsink")
    assert cam._custom_pipeline == "my custom ! appsink"


def test_open_failure_raises_actionable_error():
    cam = GstCamera("/dev/video99", 1280, 800, 100)
    with pytest.raises(RuntimeError, match="GStreamer"):
        cam.start()
