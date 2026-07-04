"""Threaded UVC capture.

WHY a capture thread that always holds only the NEWEST frame:
1) Timestamp accuracy. V4L2/OpenCV buffer frames internally; if the processing
   loop stamps a frame when it finally reads it, the stamp can be 1-3 frame
   periods late. Every velocity estimate is dpos/dt -- a wrong t poisons v,
   which poisons the lead prediction. So we stamp in the grab thread,
   immediately after grab().
2) Freshness beats completeness for control. If processing hiccups, we want
   the latest view of the world, not a queue of stale ones. The thread
   overwrites a single slot; old frames are dropped on purpose.
"""
from __future__ import annotations

import threading

import cv2

from ..util.timing import now
from . import uvc_ctrl
from .base import Frame, FrameSource


class V4L2Camera(FrameSource):
    def __init__(self, device: str, width: int, height: int, fps: int, fourcc: str = "MJPG",
                 ctrls: dict | None = None):
        self._device = device
        self._w, self._h, self._fps, self._fourcc = width, height, fps, fourcc
        self._ctrls = ctrls or {}
        self.mode_desc = "v4l2 (not started)"
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._consumed = True
        self._run = False
        self._thread: threading.Thread | None = None
        self._idx = 0

    def start(self) -> None:
        self._open()
        if self._ctrls:
            # Re-apply saved driver controls (exposure/WB locks) — they do not
            # survive reboot/replug on their own.
            uvc_ctrl.apply_ctrls(self._device, self._ctrls)
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self) -> None:
        """Open the capture and negotiate the mode. Subclasses swap this out
        (GstCamera) while keeping the grab thread and ctrl handling."""
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {self._device}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # WHY buffersize 1: minimizes driver-side queuing latency where honored.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fps = self._cap.get(cv2.CAP_PROP_FPS)
        fcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        got_cc = fcc.to_bytes(4, "little").decode("ascii", errors="replace") if fcc else "????"
        # WHY log negotiated vs requested: UVC silently falls back to the nearest
        # supported mode; assuming you got what you asked for hides a 2x latency bug.
        print(f"[v4l2] requested {self._fourcc} {self._w}x{self._h}@{self._fps} "
              f"-> negotiated {got_cc} {got_w}x{got_h}@{got_fps:.0f}")
        if got_cc != self._fourcc:
            # The classic fps killer: the OV9782 does 100fps in MJPG but only
            # ~10fps in raw YUYV at 1280x800. A silent format fallback caps the
            # whole pipeline and looks like a tuning problem.
            print(f"[v4l2] WARNING: pixel format fell back to {got_cc} — raw formats are "
                  f"fps-capped (YUYV@1280x800 is ~10fps on the OV9782). Verify supported "
                  f"modes with tools/enumerate_camera.py {self._device}")
        self._w, self._h = got_w, got_h
        self.mode_desc = f"{got_cc} {got_w}×{got_h} @ {got_fps:.0f} fps"

    def _loop(self) -> None:
        assert self._cap is not None
        while self._run:
            ok = self._cap.grab()
            t = now()  # stamp at grab -- the whole point of this thread
            if not ok:
                continue
            ok, img = self._cap.retrieve()
            if not ok:
                continue
            with self._lock:
                self._latest = Frame(img=img, t=t, idx=self._idx)
                self._consumed = False
                self._idx += 1

    def read(self) -> Frame | None:
        with self._lock:
            if self._latest is None or self._consumed:
                return None
            self._consumed = True
            return self._latest

    def stop(self) -> None:
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._w, self._h

    @property
    def device(self) -> str:
        return self._device
