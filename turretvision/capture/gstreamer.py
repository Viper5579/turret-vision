"""Jetson hardware-decode capture via GStreamer.

WHY this backend exists: MJPG at 1280x800@100fps means 100 JPEG decodes per
second, and cv2's V4L2 backend does them in software — measured ~16ms/frame
on an Orin Nano CPU, which caps the whole pipeline near 35fps no matter how
fast the camera delivers (tools/measure_capture.py separates the two). The
Jetson has a dedicated hardware JPEG decode block; routing capture through
GStreamer's nvv4l2decoder moves the decode there and frees the CPU for
detect/track.

Requires an OpenCV built with GStreamer — the JetPack system build has it,
which is one more reason the README forbids letting pip shadow it with the
generic PyPI wheel (that wheel has no GStreamer support at all).

Inherits the grab-thread/newest-frame-only design and v4l2_ctrls handling
from V4L2Camera; only the way the capture is opened differs. UVC controls
still work: GStreamer's v4l2src uses the same /dev/videoX node v4l2-ctl
talks to.
"""
from __future__ import annotations

import cv2

from .v4l2 import V4L2Camera

# nvv4l2decoder = Jetson hardware decode block. jpegdec = GStreamer's software
# decoder, used as fallback so the backend still runs on non-Jetson machines.
HW_DECODE = "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx"
SW_DECODE = "jpegdec ! videoconvert ! video/x-raw"


def build_pipeline(device: str, width: int, height: int, fps: int,
                   decode: str = HW_DECODE) -> str:
    # drop=true max-buffers=1: same freshness-over-completeness policy as the
    # V4L2 path — if processing hiccups, old frames are discarded, not queued.
    return (f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            f"{decode} ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false")


class GstCamera(V4L2Camera):
    def __init__(self, device: str, width: int, height: int, fps: int,
                 fourcc: str = "MJPG", ctrls: dict | None = None,
                 pipeline: str | None = None):
        super().__init__(device, width, height, fps, fourcc, ctrls)
        self._custom_pipeline = pipeline

    def _open(self) -> None:
        attempts = ([("custom", self._custom_pipeline)] if self._custom_pipeline else
                    [("hw", build_pipeline(self._device, self._w, self._h, self._fps)),
                     ("sw", build_pipeline(self._device, self._w, self._h, self._fps,
                                           decode=SW_DECODE))])
        for kind, pipe in attempts:
            print(f"[gst] trying {kind} pipeline: {pipe}")
            self._cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._w
                got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._h
                self._w, self._h = got_w, got_h
                decode_note = {"hw": "hw decode", "sw": "SW decode (no nvv4l2decoder?)",
                               "custom": "custom pipeline"}[kind]
                if kind == "sw":
                    print("[gst] WARNING: hardware decoder unavailable, using software "
                          "jpegdec — expect little speedup over the v4l2 backend")
                self.mode_desc = f"GST MJPG {got_w}×{got_h} @ {self._fps} fps ({decode_note})"
                return
            self._cap.release()
        raise RuntimeError(
            "GStreamer pipeline failed to open. Check that this OpenCV build has "
            "GStreamer (python3 -c \"import cv2; print(cv2.getBuildInformation())\" "
            "| grep -i gstreamer) — on the Jetson that means the SYSTEM OpenCV, "
            "not a pip wheel. Set camera.gst_pipeline in the config to override "
            "the pipeline, or fall back to camera.backend: v4l2.")
