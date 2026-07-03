"""Human-readable aim output. Phase 2 stand-in for the serial link.

WHY rate-limited: at 60-100 fps a full-rate print floods the terminal into
uselessness and, worse, stdout blocking can stall the pipeline -- the exact
latency bug the real serial link avoids with a queue+thread.
"""
from __future__ import annotations

from ..util.timing import now
from .base import AimOutput, TurretLink


class ConsoleLink(TurretLink):
    def __init__(self, console_rate_hz: float = 5.0, **_unused):
        self._min_dt = 1.0 / console_rate_hz if console_rate_hz > 0 else 0.0
        self._t_last = 0.0

    def send_aim(self, aim: AimOutput) -> None:
        t = now()
        if t - self._t_last < self._min_dt:
            return
        self._t_last = t
        if aim.valid:
            print(f"[aim] t={aim.t:8.3f} az={aim.az_deg:+7.2f} el={aim.el_deg:+7.2f} "
                  f"rate=({aim.az_rate_dps:+6.1f},{aim.el_rate_dps:+6.1f}) dps "
                  f"conf={aim.confidence:4.2f}")
        else:
            print(f"[aim] t={aim.t:8.3f} NO TARGET (conf={aim.confidence:4.2f})")
