"""Mission Control backend: serves the GUI, streams video + telemetry, takes commands.

Video costs no encoding: the camera already produces JPEG, so the server
forwards those exact bytes. Overlays travel as small separate fields tagged
with the frame number and are drawn by the browser, so they stay sharp and
the Jetson never re-encodes anything.

WebSocket messages are MessagePack maps with a "t" (type) key:
  server -> browser: hello, frame {seq, jpeg, ov}, tel {...}
  browser -> server: mode {mode}, jog {yaw, pitch}, estop {on}, video {fps}
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, doctor
from ..config import Config
from ..runtime.bus import Bus

log = logging.getLogger(__name__)


def _ui_dir() -> Path | None:
    env = os.environ.get("TURRET_UI_DIR")
    for cand in [env, Path(__file__).resolve().parents[2] / "ui" / "dist"]:
        if cand and Path(cand, "index.html").exists():
            return Path(cand)
    return None


def create_app(cfg: Config, bus: Bus, source: str) -> FastAPI:
    health: dict = {"checks": [], "running": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=run_health, daemon=True).start()
        yield

    app = FastAPI(title="Turret Mission Control", version=__version__, lifespan=lifespan)

    def run_health() -> None:
        health["running"] = True
        try:
            # Only checks that are safe while the system runs: the ESP32 check would fight
            # the live link for the serial port, and the compute benchmark would steal CPU
            # from the very pipeline it measures. `tv doctor` runs both with the system down.
            checks = doctor.check_python() + doctor.check_platform(cfg) + doctor.check_camera(cfg)
            health["checks"] = [asdict(c) for c in checks]
        finally:
            health["running"] = False

    @app.get("/api/health")
    async def get_health() -> JSONResponse:
        return JSONResponse({**health, "source": source, "version": __version__})

    @app.post("/api/health/run")
    async def rerun_health() -> JSONResponse:
        if not health["running"]:
            threading.Thread(target=run_health, daemon=True).start()
        return JSONResponse({"started": True})

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse({"config": cfg.model_dump(), "schema": cfg.model_json_schema()})

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        state = {"fps": cfg.server.video_fps}
        await sock.send_bytes(
            msgpack.packb(
                {
                    "t": "hello",
                    "version": __version__,
                    "source": source,
                    "camera": {"w": cfg.camera.width, "h": cfg.camera.height, "hfov": cfg.camera.hfov_deg},
                    "limits": cfg.turret.model_dump(),
                }
            )
        )

        async def sender() -> None:
            last_seq = -1
            loop = asyncio.get_running_loop()
            next_tel = 0.0
            next_frame = 0.0
            while True:
                now = loop.time()
                f = bus.frame
                if f is not None and f.seq != last_seq and now >= next_frame:
                    last_seq = f.seq
                    next_frame = now + 1.0 / max(state["fps"], 1.0)
                    await sock.send_bytes(msgpack.packb({"t": "frame", "seq": f.seq, "jpeg": f.jpeg, "ov": f.overlay}))
                if now >= next_tel:
                    next_tel = now + 1 / 30
                    await sock.send_bytes(msgpack.packb({"t": "tel", **bus.get_telemetry()}, default=str))
                await asyncio.sleep(0.004)

        async def receiver() -> None:
            while True:
                msg = await sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                raw = msg.get("bytes")
                data = msgpack.unpackb(raw) if raw else None
                if data is None and msg.get("text"):
                    import json

                    data = json.loads(msg["text"])
                if not isinstance(data, dict):
                    continue
                if data.get("t") == "video":
                    state["fps"] = float(min(max(data.get("fps", 15), 1), 60))
                else:
                    bus.command(data)

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()

    ui = _ui_dir()
    if ui is not None:
        app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    else:

        @app.get("/")
        async def no_ui() -> JSONResponse:
            return JSONResponse({"error": "GUI not built. Run: cd ui && npm ci && npm run build"}, status_code=503)

    return app
