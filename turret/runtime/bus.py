"""Hand-off point between the real-time pipeline thread and the web server.

The pipeline publishes the newest frame (the camera's own JPEG bytes plus
overlay data) and a telemetry snapshot; browsers read whatever is newest and
never slow the pipeline down. Commands from the GUI travel the other way
through a queue the pipeline drains once per loop.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FramePacket:
    seq: int
    jpeg: bytes
    overlay: dict[str, Any]


@dataclass
class Bus:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    frame: FramePacket | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    commands: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    frame_event: threading.Condition = field(default_factory=threading.Condition)

    def publish_frame(self, pkt: FramePacket) -> None:
        with self.frame_event:
            self.frame = pkt
            self.frame_event.notify_all()

    def publish_telemetry(self, tel: dict[str, Any]) -> None:
        with self._lock:
            self.telemetry = tel

    def get_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.telemetry)

    def command(self, cmd: dict[str, Any]) -> None:
        self.commands.put(cmd)

    def drain_commands(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out
