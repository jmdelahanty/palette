"""Serialization helpers for read-only report plans."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from typing import Any

from fisheye.shared.json_safety import json_attr_safe

from .models import ReportPlan


def report_plan_to_dict(plan: ReportPlan) -> dict[str, Any]:
    return json_attr_safe(asdict(plan))


def report_plan_json(plan: ReportPlan, *, pretty: bool = True) -> str:
    return json.dumps(
        report_plan_to_dict(plan),
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
    )


def report_plan_sha256(plan: ReportPlan) -> str:
    """Return a stable digest over the complete serialized report plan."""

    payload = json.dumps(
        report_plan_to_dict(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
