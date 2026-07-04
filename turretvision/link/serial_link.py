"""Real serial link to the ESP32 (pyserial, dedicated IO thread).

WHY an IO thread with the pipeline only posting the latest aim: a serial
write can block (full OS buffer, USB hiccup) and a blocked write in the
processing loop is a latency spike in the control path. The pipeline's
send_aim() just swaps a reference; the IO thread owns the port.

WHY the thread keeps transmitting at aim_rate_hz even when the pipeline goes
quiet: a silent line is ambiguous — crashed Jetson? unplugged cable? no
target? The heartbeat (target_valid=0, setpoints frozen at last value) lets
the firmware distinguish "no target, hold position" from "link dead, enter
safe state after timeout" (SPEC 5).

Clock domains: ESP32 millis() and Jetson monotonic are unrelated. Each
telemetry packet is stamped on receipt and a min-filtered offset estimate is
maintained (receive jitter only ever ADDS delay, so the minimum observed
offset is the best estimate of the true one). Phase 4.5 pose interpolation
consumes clock_offset().
"""
from __future__ import annotations

import threading

from ..util.timing import now
from . import protocol
from .base import AimOutput, Telemetry, TurretLink


class SerialLink(TurretLink):
    def __init__(self, port: str, baud: int = 115200, aim_rate_hz: float = 50.0,
                 **_unused):
        import serial  # local import: only this backend needs pyserial
        self._ser = serial.Serial(port, baud, timeout=0, write_timeout=0.2)
        self._send_dt = 1.0 / aim_rate_hz
        self._parser = protocol.Parser()
        self._lock = threading.Lock()
        self._latest_aim = AimOutput(t=0.0, valid=False, az_deg=0.0, el_deg=0.0,
                                     az_rate_dps=0.0, el_rate_dps=0.0, confidence=0.0)
        self._latest_telem: Telemetry | None = None
        self._offset: float | None = None   # min-filtered t_rx - t_esp
        self._run = True
        self.tx_frames = 0
        self._thread = threading.Thread(target=self._io_loop, name="serial-io", daemon=True)
        self._thread.start()

    # -- pipeline side -------------------------------------------------------
    def send_aim(self, aim: AimOutput) -> None:
        with self._lock:
            if not aim.valid and not aim.estop:
                # heartbeat semantics: freeze setpoints at last commanded value
                prev = self._latest_aim
                aim = AimOutput(aim.t, False, prev.az_deg, prev.el_deg, 0.0, 0.0,
                                aim.confidence, aim.range_m, aim.estop)
            self._latest_aim = aim

    def poll_telemetry(self) -> Telemetry | None:
        with self._lock:
            return self._latest_telem

    def clock_offset(self) -> float | None:
        """Best estimate of (Jetson monotonic s) - (ESP32 millis s)."""
        with self._lock:
            return self._offset

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=1.0)
        self._ser.close()

    @property
    def rx_errors(self) -> int:
        return self._parser.crc_errors + self._parser.oversize_errors

    # -- IO thread -----------------------------------------------------------
    def _io_loop(self) -> None:
        next_tx = now()
        while self._run:
            try:
                data = self._ser.read(4096)
            except OSError:
                break  # port vanished (unplug); pipeline sees stale telemetry
            if data:
                t_rx = now()
                for pkt_type, payload in self._parser.feed(data):
                    pkt = protocol.decode(pkt_type, payload)
                    if isinstance(pkt, protocol.TelemPacket):
                        with self._lock:
                            self._latest_telem = Telemetry(
                                t_esp_ms=pkt.t_ms, yaw_deg=pkt.yaw_deg,
                                pitch_deg=pkt.pitch_deg, yaw_rate_dps=pkt.yaw_rate_dps,
                                status=pkt.status, t_rx=t_rx)
                            off = t_rx - pkt.t_ms / 1000.0
                            if self._offset is None or off < self._offset:
                                self._offset = off
            t = now()
            if t >= next_tx:
                next_tx = t + self._send_dt
                with self._lock:
                    aim = self._latest_aim
                wire = protocol.pack_aim(protocol.AimPacket(
                    t_ms=int(t * 1000), target_valid=aim.valid, estop=aim.estop,
                    mode=1 if aim.valid else 0,
                    yaw_deg=aim.az_deg, pitch_deg=aim.el_deg,
                    yaw_rate_dps=aim.az_rate_dps, pitch_rate_dps=aim.el_rate_dps,
                    confidence=aim.confidence,
                    range_mm=int((aim.range_m or 0) * 1000)))
                try:
                    self._ser.write(wire)
                    self.tx_frames += 1
                except OSError:
                    break
            # short sleep: reads are non-blocking; 1ms keeps RX latency low
            # without busy-spinning a core
            self._stop_wait(0.001)

    def _stop_wait(self, dt: float) -> None:
        # time.sleep in a tight loop is fine; separated for test monkeypatching
        import time
        time.sleep(dt)
