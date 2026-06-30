"""Notification event helpers for Palette web labeling."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .notifications import LabelingNotificationConfig


def _request_truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _notification_config_from_values(
    *,
    mode: str | None = None,
    base_url: str | None = None,
) -> LabelingNotificationConfig:
    normalized_mode = str(mode or "").strip() or None
    normalized_base_url = str(base_url or "").strip().rstrip("/") or None
    return LabelingNotificationConfig.from_env(
        mode=normalized_mode,
        base_url=normalized_base_url,
    )


def _notification_event_type(result: Mapping[str, object], *, prefix: str) -> str:
    status = str(result.get("status") or "unknown").strip().lower() or "unknown"
    return f"{prefix}_{status}"


def _notification_exception_result(
    *,
    kind: str,
    to_user: str,
    exc: Exception,
) -> dict[str, object]:
    return {
        "ok": False,
        "schema": "palette.web_labeling_notification_result.v1",
        "notification_id": None,
        "status": "failed",
        "transport": "unknown",
        "kind": kind,
        "to_user": to_user,
        "reason": "notification_exception",
        "details": str(exc),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sent": False,
        "queued": False,
    }
