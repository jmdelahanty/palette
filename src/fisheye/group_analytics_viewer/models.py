"""Response helpers for the group analytics viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class HealthReport:
    ok: bool
    checked_utc: str
    export_run_id: str
    export_root: str
    message: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_utc": self.checked_utc,
            "export_run_id": self.export_run_id,
            "export_root": self.export_root,
            "message": self.message,
            "details": self.details,
        }
