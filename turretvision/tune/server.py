"""Browser-based tuning server (--tune).

WHY a web page instead of cv2 trackbars or a Qt panel: the Jetson usually runs
headless over SSH, so the UI has to reach a laptop anyway; a browser needs zero
extra dependencies on either end (stdlib http.server + an MJPEG stream), and it
gets a UI that is actually pleasant to use.

Threading model: ThreadingHTTPServer handles each request on its own thread.
The pipeline thread calls publish() once per frame; stream handlers block on a
Condition until a new frame lands, then JPEG-encode at a capped rate so a
100 fps pipeline doesn't turn into a 100 fps network stream. Parameter writes
go through ParamRegistry.queue and are applied by the pipeline thread.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

import cv2

from ..util.config import save_overlay
from .params import ParamRegistry

BOUNDARY = "turretvision-frame"


class TuningServer:
    def __init__(self, registry: ParamRegistry, config_path: str, port: int = 8089,
                 jpeg_quality: int = 80, max_stream_fps: float = 20.0):
        self._registry = registry
        self._config_path = config_path
        self._quality = int(jpeg_quality)
        self._min_frame_dt = 1.0 / max_stream_fps
        self._cond = threading.Condition()
        self._frame = None            # latest BGR overlay frame (pipeline-owned copy)
        self._seq = 0
        self._stats: dict[str, Any] = {}
        self._conf_hist: deque[float] = deque(maxlen=120)
        self._initial = registry.snapshot()   # for the Revert button
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
        self.port = self._httpd.server_address[1]  # resolves port=0 (tests)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="tune-http", daemon=True)

    # -- pipeline side -----------------------------------------------------
    def start(self) -> None:
        self._thread.start()
        host = socket.gethostname()
        print(f"[tune] tuning UI on http://{host}:{self.port}  "
              f"(or http://<this-machine's-ip>:{self.port})")

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def apply_pending(self) -> None:
        self._registry.apply_pending()

    def value(self, key: str) -> Any:
        return self._registry.value(key)

    def publish(self, img, stats: dict[str, Any]) -> None:
        """Hand the latest overlay frame + stats to the server. The pipeline
        must not mutate img afterwards (it renders onto a fresh copy anyway)."""
        with self._cond:
            self._frame = img
            self._seq += 1
            self._stats = stats
            self._conf_hist.append(float(stats.get("conf", 0.0)))
            self._cond.notify_all()

    # -- server side ---------------------------------------------------------
    def wait_frame_jpeg(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            if self._frame is None or self._seq == last_seq:
                return None, last_seq
            frame, seq = self._frame, self._seq
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return (buf.tobytes() if ok else None), seq

    def state_json(self) -> bytes:
        with self._cond:
            stats = dict(self._stats)
            hist = list(self._conf_hist)
        return json.dumps({"params": self._registry.describe(), "stats": stats,
                           "conf_history": hist}).encode()

    def set_param(self, key: str, value: Any) -> Any:
        return self._registry.queue(key, value)

    def save(self) -> str:
        return str(save_overlay(self._config_path, self._registry.snapshot()))

    def revert(self) -> None:
        for key, v in self._initial.items():
            self._registry.queue(key, v)


def _make_handler(server: TuningServer):
    page = resources.files("turretvision.tune").joinpath("page.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # keep the pipeline console clean
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, server.state_json(), "application/json")
            elif self.path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            next_t = 0.0
            try:
                while True:
                    jpeg, seq = server.wait_frame_jpeg(seq)
                    if jpeg is None:
                        continue  # timeout (source stalled); keep the socket open
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t = time.monotonic() + server._min_frame_dt
                    self.wfile.write(f"--{BOUNDARY}\r\n"
                                     f"Content-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/set":
                try:
                    v = server.set_param(body["key"], body["value"])
                except KeyError:
                    self._send(404, b'{"error":"unknown key"}', "application/json")
                    return
                self._send(200, json.dumps({"key": body["key"], "value": v}).encode(),
                           "application/json")
            elif self.path == "/api/save":
                path = server.save()
                self._send(200, json.dumps({"saved": path}).encode(), "application/json")
            elif self.path == "/api/revert":
                server.revert()
                self._send(200, b'{"ok":true}', "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler
