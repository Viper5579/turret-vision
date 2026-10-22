"""Jetson side of the ESP32 link: handshake, clock sync, heartbeat, message dispatch.

`Link.poll()` is called from the owner's loop (every few ms). It never blocks:
it drains received bytes, answers protocol housekeeping (Hello, pings), and
hands Telem / FrameSync / Log messages to the callbacks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from . import messages as m
from .clock import ClockSync
from .framing import FrameDecoder, encode_frame
from .transport import Transport

log = logging.getLogger(__name__)


@dataclass
class LinkStats:
    tx_frames: int = 0
    rx_frames: int = 0
    unknown_ids: int = 0
    bad_sizes: int = 0
    hash_mismatch: bool = False
    fw_version: int = 0
    features: int = 0
    last_rx: float = float("-inf")


@dataclass
class LinkCallbacks:
    on_telem: Callable[[m.Telem, float], None] | None = None
    on_frame_sync: Callable[[m.FrameSync], None] | None = None
    on_log: Callable[[m.Log], None] | None = None
    on_fire_ack: Callable[[m.FireAck], None] | None = None
    on_connect: Callable[[m.HelloAck], None] | None = None
    extra: dict = field(default_factory=dict)


class Link:
    HELLO_PERIOD_S = 2.0

    def __init__(
        self,
        transport: Transport,
        now: Callable[[], float],
        callbacks: LinkCallbacks | None = None,
        config_msg: m.Config | None = None,
        rx_timeout_s: float = 0.5,
    ) -> None:
        self._t = transport
        self._now = now
        self.cb = callbacks or LinkCallbacks()
        self.config_msg = config_msg
        self.rx_timeout_s = rx_timeout_s
        self.clock = ClockSync()
        self.stats = LinkStats()
        self._dec = FrameDecoder()
        self._ping_seq = 0
        self._ping_sent: dict[int, float] = {}
        self._next_ping = 0.0
        self._next_hello = 0.0
        self._connected_at: float | None = None
        self.connected = False

    # ------------------------------------------------------------------ sending
    def send(self, msg: m.Message) -> None:
        self._t.write(encode_frame(msg.ID, msg.pack()))
        self.stats.tx_frames += 1

    def send_traj(self, traj: m.Traj) -> None:
        if self.connected and not self.stats.hash_mismatch:
            self.send(traj)

    # ------------------------------------------------------------------ polling
    def poll(self) -> None:
        now = self._now()
        data = self._t.read()
        if data:
            for msg_id, payload in self._dec.feed(data):
                self._dispatch(msg_id, payload, self._now())
        if not self.connected or self.stats.hash_mismatch:
            if now >= self._next_hello:
                self.send(m.Hello(proto_hash=m.PROTO_HASH))
                self._next_hello = now + self.HELLO_PERIOD_S
        else:
            if now - self.stats.last_rx > self.rx_timeout_s:
                log.warning("ESP32 link silent for %.2f s; re-handshaking", now - self.stats.last_rx)
                self.connected = False
                self._next_hello = now
            elif now >= self._next_ping:
                self._send_ping(now)
                young = self._connected_at is not None and now - self._connected_at < 3.0
                self._next_ping = now + (0.05 if young else 0.25)

    def _send_ping(self, now: float) -> None:
        self._ping_seq = (self._ping_seq + 1) & 0xFFFF
        self._ping_sent[self._ping_seq] = now
        if len(self._ping_sent) > 64:
            self._ping_sent.pop(next(iter(self._ping_sent)))
        self.send(m.TimePing(seq=self._ping_seq))

    def _dispatch(self, msg_id: int, payload: bytes, t_rx: float) -> None:
        cls = m.BY_ID.get(msg_id)
        if cls is None:
            self.stats.unknown_ids += 1
            return
        if len(payload) != cls.SIZE:
            self.stats.bad_sizes += 1
            return
        msg = cls.unpack(payload)
        self.stats.rx_frames += 1
        self.stats.last_rx = t_rx
        if isinstance(msg, m.HelloAck):
            self._on_hello_ack(msg, t_rx)
        elif isinstance(msg, m.TimePong):
            t1 = self._ping_sent.pop(msg.seq, None)
            if t1 is not None:
                self.clock.add(t1, msg.t_rx_us, msg.t_tx_us, t_rx)
        elif isinstance(msg, m.Telem):
            if self.cb.on_telem:
                self.cb.on_telem(msg, t_rx)
        elif isinstance(msg, m.FrameSync):
            if self.cb.on_frame_sync:
                self.cb.on_frame_sync(msg)
        elif isinstance(msg, m.Log):
            log.info("esp32: %s", msg.text)
            if self.cb.on_log:
                self.cb.on_log(msg)
        elif isinstance(msg, m.FireAck) and self.cb.on_fire_ack:
            self.cb.on_fire_ack(msg)

    def _on_hello_ack(self, ack: m.HelloAck, t_rx: float) -> None:
        self.stats.fw_version = ack.fw_version
        self.stats.features = ack.features
        if ack.proto_hash != m.PROTO_HASH:
            if not self.stats.hash_mismatch:
                log.error(
                    "firmware protocol hash 0x%08X != 0x%08X: reflash the ESP32 (tv flash)",
                    ack.proto_hash,
                    m.PROTO_HASH,
                )
            self.stats.hash_mismatch = True
            self.connected = False
            return
        self.stats.hash_mismatch = False
        if not self.connected:
            self.connected = True
            self._connected_at = t_rx
            self._next_ping = t_rx
            if self.config_msg is not None:
                self.send(self.config_msg)
            if self.cb.on_connect:
                self.cb.on_connect(ack)

    @property
    def synced(self) -> bool:
        return self.connected and self.clock.valid

    def close(self) -> None:
        self._t.close()
