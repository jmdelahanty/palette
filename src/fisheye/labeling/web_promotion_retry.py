"""Promotion retry support payload helpers for web labeling."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .assignment_store import LabelingStore
from .web_authorization_metadata import (
    _add_task_open_personalized_launch_metadata,
    _labeler_authorization_context,
)
from .web_policy import _browser_mutation_write_runtime_checklist
from .web_responses import _format_error


def _parse_clip_index(clip_id: object) -> int:
    text = str(clip_id or "").strip()
    if text.startswith("clip_"):
        text = text[len("clip_") :]
    try:
        return int(text)
    except ValueError:
        return -1


def _jsonish(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_jsonish(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonish(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact_promotion_result(result: Mapping[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {
        "status": result.get("status"),
        "target_crop_run": result.get("target_crop_run"),
        "action_counts": result.get("action_counts", {}),
    }
    items = result.get("items")
    if isinstance(items, list):
        compact["item_count"] = len(items)
        compact["items"] = [dict(item) for item in items[:3] if isinstance(item, Mapping)]
    if result.get("target_refined_run") is not None:
        compact["target_refined_run"] = result.get("target_refined_run")
    return _jsonish(compact)  # type: ignore[return-value]


def _labeler_promotion_retry_operator_support_payload(
    *,
    user: str,
    expected_user: str | None,
    event: Mapping[str, object],
    event_task: Mapping[str, object] | None,
    session: Mapping[str, object],
) -> dict[str, object]:
    event_id = str(event.get("event_id") or "")
    route_template = "/api/admin/events/{event_id}/retry-promotion"
    extra = _labeler_promotion_retry_failure_metadata(
        user=user,
        expected_user=expected_user,
        event_id=event_id,
        event=event,
        event_task=event_task,
        session=session,
    )
    extra.update(
        {
            "operator_recovery_route_template": route_template,
            "operator_recovery_route": route_template.replace("{event_id}", event_id),
        }
    )
    return _format_error(
        "operator_support_required",
        details=(
            "Failed-promotion retry can mutate server-owned training Zarr targets and is "
            "operator-only. Give the operator this event ID; operators retry from the "
            "admin recovery route after audit lookup."
        ),
        status=HTTPStatus.FORBIDDEN,
        extra=extra,
    )

def _labeler_promotion_retry_failure_metadata(
    *,
    user: str,
    expected_user: str | None,
    event_id: str = "",
    event: Mapping[str, object] | None = None,
    event_task: Mapping[str, object] | None = None,
    session: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_event_id = str(event_id or (event or {}).get("event_id") or "")
    authorization_context = _labeler_authorization_context(
        user=user,
        expected_user=expected_user,
        task=event_task,
        session=session,
    )
    metadata: dict[str, object] = {
        "failed_event_id": resolved_event_id,
        "labeler_failed_promotion_retry_action": "operator_support_only",
        "labeler_promotion_retry_mutation_enabled": False,
        "promotion_retry_attempted": False,
        "promotion_retry_claimed": False,
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "browser_label_write_target": "training_zarr",
        "handoff_csv_artifacts_are_label_write_targets": False,
        "intermediate_csv_artifacts_are_label_write_targets": False,
        "browser_writes_csv_or_handoff_files": False,
        "browser_writes_handoff_csv": False,
        "browser_writes_intermediate_csv": False,
        "browser_receives_zarr_write_authority": False,
        "browser_has_direct_zarr_write_authority": False,
        "operator_validation_mutation_gate_checked_server_side": True,
        "operator_validation_mutation_gate_required": False,
        "operator_validation_mutation_gate_ready": True,
        "operator_validation_mutation_gate_blocks_browser_mutation": False,
        "operator_validation_mutation_gate_not_ready_reason": "",
        "authorization_context": authorization_context,
        "return_expected_user": str(authorization_context.get("return_expected_user") or ""),
        "return_personal_dataset_queue_url": str(
            authorization_context.get("return_personal_dataset_queue_url") or ""
        ),
        "return_personal_dataset_queue_expected_user_guarded": bool(
            authorization_context.get("return_personal_dataset_queue_expected_user_guarded")
        ),
        "return_personal_work_url": str(
            authorization_context.get("return_personal_work_url") or ""
        ),
        "return_personal_work_expected_user_guarded": bool(
            authorization_context.get("return_personal_work_expected_user_guarded")
        ),
    }
    return _add_task_open_personalized_launch_metadata(
        metadata,
        authorization_context=authorization_context,
    )

def _promotion_success_event_for_retry(
    store: LabelingStore,
    failed_event_id: str,
    *,
    user: str,
) -> dict[str, object] | None:
    failed_event_id = str(failed_event_id or "").strip()
    if not failed_event_id:
        return None
    for event in store.list_events(
        event_type="promotion_success",
        assignee_user=str(user or "").strip() or None,
        limit=500,
    ):
        target = event.get("target")
        if not isinstance(target, Mapping):
            continue
        if str(target.get("retry_of_event_id") or "").strip() == failed_event_id:
            return event
    return None

def _admin_promotion_retry_preflight_error(
    event: Mapping[str, object],
) -> tuple[dict[str, object], HTTPStatus] | None:
    event_type = str(event.get("event_type") or "")
    if event_type == "promotion_failed":
        return None
    status = HTTPStatus.CONFLICT
    return (
        _format_error(
            "promotion_retry_not_supported",
            details="Only promotion_failed audit events can be retried by an operator.",
            status=status,
            extra={
                "event_id": str(event.get("event_id") or ""),
                "event_type": event_type,
                "required_event_type": "promotion_failed",
                "promotion_retry_attempted": False,
                "promotion_retry_claimed": False,
                "operator_action": (
                    "Use the operator audit lookup to confirm the event ID is a failed "
                    "promotion event before retrying."
                ),
            },
        ),
        status,
    )
