from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AimOutput:
    """What the pipeline hands the link each frame. az/el are the ABSOLUTE
    yaw/pitch setpoints in degrees (turret pose from telemetry + tracked error;
    identical to the raw error when no telemetry exists, e.g. ConsoleLink).
    Serial/mock links pack this into the AIM wire packet (SPEC 5)."""
    t: float
    valid: bool
    az_deg: float
    el_deg: float
    az_rate_dps: float
    el_rate_dps: float
    confidence: float           # 0..1
    range_m: float | None = None
    estop: bool = False         # in every packet on purpose: can't be lost


@dataclass
class Telemetry:
    t_esp_ms: int
    yaw_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    status: int                 # TELEM status bits, see link/protocol.py
    t_rx: float

    @property
    def homed(self) -> bool:
        return bool(self.status & 0x01)

    @property
    def estopped(self) -> bool:
        return bool(self.status & 0x10)


class TurretLink(ABC):
    @abstractmethod
    def send_aim(self, aim: AimOutput) -> None: ...

    def poll_telemetry(self) -> Telemetry | None:
        """Latest turret state. Default None (console link, no hardware)."""
        return None

    def close(self) -> None:  # noqa: B027 (optional hook by design)
        pass
