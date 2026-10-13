"""Lean labeler queue payload for GET /api/me/queue.

The full personal payloads (/api/me/datasets, /api/me/tasks) carry every
policy, contract and diagnostic field the operator tooling uses. The queue a
labeler sees needs only who they are, how far along they are, what blocks
them, and for each task whether and how it can be started. This module
projects that from the same `work` summary, so the enforcement and the start
decision stay identical to the full payloads; nothing here decides access.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

LABELER_QUEUE_SCHEMA = "palette.labeler_queue.v1"


def _truthy(value: object) -> bool:
    return value is True or value == "true"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _task_start(
    task: Mapping[str, Any],
    *,
    startable_states: set[str],
    validation_gate: Mapping[str, Any],
) -> dict[str, object]:
    """Whether the queue may offer Start, with the same rule the queue page applies."""

    contract = _mapping(task.get("direct_browser_start_authorization_contract"))
    endpoint = _text(task.get("direct_browser_start_endpoint"))
    gate_blocks = _truthy(validation_gate.get("blocks_task_open"))
    ready = (
        not gate_blocks
        and _truthy(task.get("direct_browser_start_authorization_contract_ready"))
        and bool(task.get("labeler_start_ready"))
        and _text(task.get("state")) in startable_states
        and bool(_text(task.get("task_id")))
        and bool(endpoint)
    )
    if gate_blocks:
        reason = _text(validation_gate.get("not_ready_reason")) or "operator_validation_start_blocked"
        action = _text(validation_gate.get("operator_action")) or (
            "Complete required operator validation evidence before browser Start/Open."
        )
    else:
        reason = _text(task.get("direct_browser_start_not_ready_reason")) or _text(
            contract.get("not_ready_reason")
        )
        action = _text(task.get("direct_browser_start_operator_action")) or _text(
            contract.get("operator_action")
        )
    return {
        "ready": ready,
        "endpoint": endpoint if ready else "",
        "method": "POST",
        "not_ready_reason": "" if ready else reason,
        "operator_action": "" if ready else action,
    }


_EMPTY_PROGRESS = {
    "row_total": None,
    "saved_row_count": 0,
    "applied_row_count": 0,
    "unapplied_row_count": 0,
    "carried_row_count": 0,
}


def _task(
    task: Mapping[str, Any],
    *,
    fallback_url: str,
    row_progress: Mapping[str, Mapping[str, object]],
    **start: Any,
) -> dict[str, object]:
    task_id = _text(task.get("task_id"))
    return {
        "task_id": _text(task.get("task_id")),
        "title": _text(task.get("title") or task.get("task_id")),
        "workflow_kind": _text(task.get("workflow_kind")),
        "component_name": _text(task.get("component_name")),
        "state": _text(task.get("state")),
        "priority": task.get("priority"),
        "notes": _text(task.get("notes")),
        "work_url": _text(task.get("expected_user_work_url") or task.get("work_url") or fallback_url),
        "start": _task_start(task, **start),
        "progress": dict(row_progress.get(task_id) or _EMPTY_PROGRESS),
    }


def queue_task_ids(work: Mapping[str, Any]) -> list[str]:
    """Every task id the queue will list, for one batched row-progress query."""

    return [
        _text(task.get("task_id"))
        for dataset in work.get("dataset_queue") or []
        for recording in dataset.get("recordings") or []
        for task in recording.get("tasks") or []
        if _text(task.get("task_id"))
    ]


def labeler_queue_payload(
    work: Mapping[str, Any],
    *,
    user: str,
    row_progress: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Project the labeler's queue from the full personal `work` summary.

    ``row_progress`` is ``LabelingStore.task_row_progress`` for the listed
    tasks; a task missing from it reports no saved rows and an unknown total.
    """

    row_progress = row_progress or {}

    policy = _mapping(work.get("dataset_queue_direct_start_policy"))
    start = {
        "startable_states": {str(s) for s in policy.get("startable_task_states") or []},
        "validation_gate": _mapping(work.get("operator_validation_start_gate")),
    }
    counts = _mapping(_mapping(work.get("dataset_queue_state")).get("counts"))
    completion = _mapping(work.get("labeler_work_completion"))
    datasets = []
    for dataset in work.get("dataset_queue") or []:
        dataset_url = _text(dataset.get("expected_user_work_url") or dataset.get("work_url"))
        recordings = []
        for recording in dataset.get("recordings") or []:
            recording_url = _text(
                recording.get("expected_user_work_url") or recording.get("work_url") or dataset_url
            )
            recordings.append({
                "recording_id": _text(recording.get("recording_id")),
                "open_task_count": int(recording.get("open_task_count") or 0),
                "task_count": int(recording.get("task_count") or 0),
                "blocked_reason": _text(recording.get("blocked_reason")),
                "work_url": recording_url,
                "tasks": [
                    _task(task, fallback_url=recording_url, row_progress=row_progress, **start)
                    for task in recording.get("tasks") or []
                ],
            })
        datasets.append({
            "dataset_id": _text(dataset.get("dataset_id")),
            "label": _text(dataset.get("dataset_label") or dataset.get("dataset_id")),
            "open_task_count": int(dataset.get("open_task_count") or 0),
            "task_count": int(dataset.get("task_count") or 0),
            "recording_count": int(dataset.get("recording_count") or 0),
            "workflow_counts": dict(_mapping(dataset.get("workflow_counts"))),
            "work_url": dataset_url,
            "recordings": recordings,
        })

    blockers = []
    gate = start["validation_gate"]
    if _truthy(gate.get("blocks_task_open")):
        blockers.append({
            "code": _text(gate.get("not_ready_reason")) or "operator_validation_start_blocked",
            "message": _text(gate.get("operator_action")),
        })
    safety = _mapping(work.get("reassignment_session_safety"))
    if _truthy(safety.get("blocks_labeler_mutation")):
        blockers.append({
            "code": "reassignment_session_safety",
            "message": _text(safety.get("operator_action")),
        })

    expected_user = _text(work.get("expected_user")) or user
    return {
        "ok": True,
        "schema": LABELER_QUEUE_SCHEMA,
        "user": user,
        "expected_user": expected_user,
        "include_completed": bool(work.get("include_completed")),
        "progress": {
            "open_task_count": int(counts.get("open_task_count") or 0),
            "complete_task_count": int(counts.get("complete_task_count") or 0),
            "total_task_count": int(counts.get("total_task_count") or 0),
            "waiting_dataset_count": int(counts.get("waiting_dataset_count") or 0),
            "waiting_recording_count": int(counts.get("waiting_recording_count") or 0),
            "blocked_recording_count": int(counts.get("blocked_recording_count") or 0),
            "completion_percent": completion.get("completion_percent"),
            "state": _text(completion.get("completion_state")),
        },
        "labeler": {
            "start_ready": bool(work.get("labeler_start_ready")),
            "status": _text(work.get("labeler_start_status")),
            "message": _text(work.get("labeler_start_message")),
            "operator_action": _text(work.get("labeler_start_operator_action")),
        },
        "blockers": blockers,
        "links": {
            "personal_dataset_queue": _text(work.get("expected_user_personal_dataset_queue_url")),
            "personal_work": _text(work.get("expected_user_personal_work_url")),
            "identity_probe": _text(work.get("expected_user_identity_probe_url")),
            # The full payload with every policy and diagnostic field, for
            # "Operator details" and support references, fetched on demand.
            "diagnostics": "/api/me/datasets?expected_user=" + quote(expected_user, safe=""),
        },
        "datasets": datasets,
    }
