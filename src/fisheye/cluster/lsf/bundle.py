"""Atomic JSON persistence for LSF plan and submission bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import json_ready


def write_json_snapshot(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write one JSON-ready snapshot atomically and return its normalized form."""

    normalized = json_ready(dict(payload))
    write_json_atomic(path, normalized)
    return normalized


__all__ = ["write_json_snapshot"]
