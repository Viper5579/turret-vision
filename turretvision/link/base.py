from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AimOutput:
    """What Phase 2 hands the link each frame. Phase 3 packs this into the wire
    format (AIM packet, SPEC 5); until then links render it however they like."""
    t: float
    valid: bool
    az_deg: float
    el_deg: float
    az_rate_dps: float
    el_rate_dps: float
    confidence: float           # 0..1
    range_m: float | None = None


@dataclass
class Telemetry:
    t_esp_ms: int
    yaw_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    status: int
    t_rx: float


class TurretLink(ABC):
    @abstractmethod
    def send_aim(self, aim: AimOutput) -> None: ...

    def poll_telemetry(self) -> Telemetry | None:
        """Latest turret state. Default None (console link, no hardware)."""
        return None

    def close(self) -> None:  # noqa: B027 (optional hook by design)
        pass
