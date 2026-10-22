import msgpack
from fastapi.testclient import TestClient

from turret.config import Config
from turret.runtime.bus import Bus, FramePacket
from turret.server.app import create_app


def test_websocket_library_installed():
    # uvicorn silently refuses WebSocket upgrades without it (the GUI then shows "offline").
    import websockets  # noqa: F401


def test_ws_streams_frames_and_accepts_commands():
    bus = Bus()
    bus.publish_frame(FramePacket(seq=7, jpeg=b"\xff\xd8fake", overlay={"w": 1280, "h": 800, "blobs": []}))
    bus.publish_telemetry({"time": 1.0, "mode": "IDLE"})
    app = create_app(Config(), bus, source="sim")
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        seen = set()
        for _ in range(6):
            msg = msgpack.unpackb(ws.receive_bytes())
            seen.add(msg["t"])
            if msg["t"] == "frame":
                assert msg["seq"] == 7 and msg["jpeg"].startswith(b"\xff\xd8")
            if {"hello", "frame", "tel"} <= seen:
                break
        assert {"hello", "frame", "tel"} <= seen
        ws.send_bytes(msgpack.packb({"t": "mode", "mode": "TRACK"}))
        ws.send_bytes(msgpack.packb({"t": "jog", "yaw": 10.0, "pitch": 0.0}))
        ws.receive_bytes()  # give the server a turn to read our messages
    cmds = bus.drain_commands()
    assert {"t": "mode", "mode": "TRACK"} in cmds


def test_health_and_config_endpoints():
    app = create_app(Config(), Bus(), source="sim")
    with TestClient(app) as client:
        assert client.get("/api/health").json()["source"] == "sim"
        cfg = client.get("/api/config").json()
        assert cfg["config"]["camera"]["width"] == 1280
        assert "properties" in cfg["schema"]
