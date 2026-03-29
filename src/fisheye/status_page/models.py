"""Data models for status page responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HealthReport:
    """Connectivity and schema readiness report for the registry."""

    ok: bool
    registry_path: str
    checked_utc: str
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "registry_path": self.registry_path,
            "checked_utc": self.checked_utc,
            "message": self.message,
            "details": self.details,
        }
