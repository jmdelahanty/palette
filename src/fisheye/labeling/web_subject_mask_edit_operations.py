"""Provenance for resampling edits on a mask save (currently: rotation).

The mask editor's Rotate tool resamples the mask (bilinear, threshold 0.5,
about the mask centroid). Each finished rotation since the row was loaded is
sent with Save as ``edit_operations`` and recorded in the checkpoint
metadata, so rotated rows can be audited or excluded from training later.
Only the declared shape is accepted; anything else refuses the save.
"""

from __future__ import annotations

import math
from typing import Any

ROTATE_OPERATION = {"op": "rotate", "method": "bilinear_threshold_0.5", "pivot": "mask_centroid"}
MAX_EDIT_OPERATIONS = 32


def validated_edit_operations(value: Any) -> list[dict[str, object]] | None:
    """Normalize the save body's ``edit_operations``; raise ValueError if malformed."""

    if value is None:
        return None
    if not isinstance(value, list) or not value or len(value) > MAX_EDIT_OPERATIONS:
        raise ValueError(f"edit_operations must be a list of 1-{MAX_EDIT_OPERATIONS} operations.")
    operations = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"op", "angle_deg", "method", "pivot"}:
            raise ValueError("Each edit operation needs exactly op, angle_deg, method and pivot.")
        if any(item[key] != expected for key, expected in ROTATE_OPERATION.items()):
            raise ValueError("Only rotate operations (bilinear_threshold_0.5 about mask_centroid) are recorded.")
        angle = item["angle_deg"]
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 < abs(angle) <= 360:
            raise ValueError("A rotate operation's angle_deg must be a finite, nonzero number of degrees.")
        operations.append({**ROTATE_OPERATION, "angle_deg": round(float(angle), 3)})
    return operations


__all__ = ["MAX_EDIT_OPERATIONS", "ROTATE_OPERATION", "validated_edit_operations"]
