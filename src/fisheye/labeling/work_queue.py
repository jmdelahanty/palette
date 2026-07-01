"""Labeler work and dataset-queue payload shaping helpers."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence
from urllib.parse import quote

from .assignment_store import LABELER_START_TASK_STATES
from .web_auth import DASHBOARD_PATH
from .web_policy import _operator_validation_visibility_fields


def _work_empty_state(work: Mapping[str, object]) -> dict[str, object]:
    recording_count = int(work.get("recording_count") or 0)
    task_count = int(work.get("task_count") or 0)
    total_task_count = int(work.get("total_task_count") or 0)
    complete_task_count = int(work.get("complete_task_count") or 0)
    incomplete_task_count = int(work.get("incomplete_task_count") or task_count)
    startable_task_count = int(work.get("startable_task_count") or 0)
    if recording_count <= 0:
        return {
            "code": "no_active_assignments",
            "is_empty": True,
            "message": "No active labeling recordings are assigned to you right now. If you expected work, ask the operator to check your recording assignment.",
            "operator_action": "Assign at least one recording to this user or confirm the user is signing in with the expected identity.",
        }
    if total_task_count > 0 and complete_task_count >= total_task_count:
        return {
            "code": "all_tasks_complete",
            "is_empty": True,
            "message": "All assigned labeling tasks are complete. Ask the operator before reopening or continuing work.",
            "operator_action": "No action is needed unless more labeling is required; reopen or generate tasks only after review.",
        }
    if incomplete_task_count <= 0 or task_count <= 0 or startable_task_count <= 0:
        return {
            "code": "no_open_tasks",
            "is_empty": True,
            "message": "Your active recordings have no startable browser-labeling tasks. If you expected work, ask the operator to generate tasks, move tasks to pending/in_progress, or inspect the batch.",
            "operator_action": "Generate, import, reopen, or move browser-labeling tasks to pending/in_progress for this user's active recordings.",
        }
    return {
        "code": "has_open_work",
        "is_empty": False,
        "message": "",
        "operator_action": "",
    }

def _work_progress_summary(work: Mapping[str, object]) -> dict[str, object]:
    recordings = [row for row in work.get("recordings", []) if isinstance(row, Mapping)]
    waiting_recordings: list[str] = []
    complete_recordings: list[str] = []
    blocked_recordings: list[str] = []
    blocked_by_reason: dict[str, int] = {}
    for recording in recordings:
        recording_id = str(recording.get("recording_id") or "")
        startable_count = int(recording.get("startable_task_count") or 0)
        total_count = int(recording.get("total_task_count") or 0)
        complete_count = int(recording.get("complete_task_count") or 0)
        if startable_count > 0:
            waiting_recordings.append(recording_id)
            continue
        if total_count > 0 and complete_count >= total_count:
            complete_recordings.append(recording_id)
            continue
        blocked_recordings.append(recording_id)
        reason = str(recording.get("no_open_task_reason") or "no_open_tasks_in_current_summary")
        blocked_by_reason[reason] = blocked_by_reason.get(reason, 0) + 1
    return {
        "recording_count": int(work.get("recording_count") or len(recordings)),
        "open_task_count": int(work.get("startable_task_count") or 0),
        "total_task_count": int(work.get("total_task_count") or 0),
        "complete_task_count": int(work.get("complete_task_count") or 0),
        "incomplete_task_count": int(work.get("incomplete_task_count") or 0),
        "startable_task_count": int(work.get("startable_task_count") or 0),
        "non_startable_task_count": int(work.get("non_startable_task_count") or 0),
        "waiting_recording_count": len(waiting_recordings),
        "complete_recording_count": len(complete_recordings),
        "blocked_recording_count": len(blocked_recordings),
        "waiting_recordings": waiting_recordings,
        "complete_recordings": complete_recordings,
        "blocked_recordings": blocked_recordings,
        "blocked_recordings_by_reason": dict(sorted(blocked_by_reason.items())),
    }

def _task_priority_value(task: Mapping[str, object]) -> float:
    try:
        return float(task.get("priority") or 0)
    except (TypeError, ValueError):
        return 0.0

def _reassignment_session_safety_operator_action() -> str:
    return (
        "Close stale previous-owner sessions with repair-reassignment-sessions or "
        "re-run assignment through assign_recording_with_session_closure before "
        "exposing labeler work."
    )

def _reassignment_session_safety_recording_ids(
    reassignment_session_safety: Mapping[str, object] | None,
) -> set[str]:
    if not isinstance(reassignment_session_safety, Mapping):
        return set()
    raw_ids = reassignment_session_safety.get("active_session_assignment_mismatch_recording_ids")
    if not isinstance(raw_ids, list):
        return set()
    return {str(recording_id).strip() for recording_id in raw_ids if str(recording_id).strip()}

def _work_recording_ids(work: Mapping[str, object]) -> set[str]:
    recordings = work.get("recordings") if isinstance(work.get("recordings"), list) else []
    return {
        str(recording.get("recording_id") or "").strip()
        for recording in recordings
        if isinstance(recording, Mapping) and str(recording.get("recording_id") or "").strip()
    }

def _reassignment_session_safety_blocks_recording(
    recording_id: str,
    reassignment_session_safety: Mapping[str, object] | None,
) -> bool:
    if not isinstance(reassignment_session_safety, Mapping):
        return False
    if bool(reassignment_session_safety.get("ok", True)):
        return False
    if not bool(reassignment_session_safety.get("blocks_labeler_mutation")):
        return False
    blocked_recording_ids = _reassignment_session_safety_recording_ids(reassignment_session_safety)
    return not blocked_recording_ids or str(recording_id).strip() in blocked_recording_ids

def _public_reassignment_session_safety_fields(
    reassignment_session_safety: Mapping[str, object] | None,
    *,
    recording_ids: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(reassignment_session_safety, Mapping):
        return {}
    store_ok = bool(reassignment_session_safety.get("ok", True))
    store_blocks = bool(reassignment_session_safety.get("blocks_labeler_mutation"))
    all_blocked_recording_ids = _reassignment_session_safety_recording_ids(
        reassignment_session_safety
    )
    if recording_ids is not None and all_blocked_recording_ids:
        visible_blocked_recording_ids = all_blocked_recording_ids & recording_ids
        affects_current_user = (not store_ok) and store_blocks and bool(visible_blocked_recording_ids)
    else:
        visible_blocked_recording_ids = all_blocked_recording_ids
        affects_current_user = (not store_ok) and store_blocks
    public_ok = store_ok or not affects_current_user
    visible_mismatch_count = (
        len(visible_blocked_recording_ids)
        if recording_ids is not None and all_blocked_recording_ids
        else int(reassignment_session_safety.get("active_session_assignment_mismatch_count") or 0)
    )
    return {
        "schema": str(
            reassignment_session_safety.get("schema")
            or "palette.web_labeling_reassignment_session_safety.v1"
        ),
        "ok": public_ok,
        "affects_current_user": affects_current_user,
        "active_session_assignment_mismatch_count": visible_mismatch_count if affects_current_user else 0,
        "active_session_assignment_mismatch_recording_ids": sorted(visible_blocked_recording_ids)
        if affects_current_user
        else [],
        "blocks_labeler_mutation": store_blocks and affects_current_user,
        "requires_operator_recovery": bool(
            reassignment_session_safety.get("requires_operator_recovery")
        )
        and affects_current_user,
        "operator_action": str(
            reassignment_session_safety.get("operator_action")
            or _reassignment_session_safety_operator_action()
        ),
    }

def _work_dataset_queue_task(
    task: Mapping[str, object],
    *,
    expected_user: str = "",
    reassignment_session_safety_blocked: bool = False,
    support_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    task_id = str(task.get("task_id") or "")
    dataset_id = str(task.get("dataset_id") or "").strip()
    recording_id = str(task.get("recording_id") or "").strip()
    workflow_kind = str(task.get("workflow_kind") or "")
    state = str(task.get("state") or "")
    zarr_use = str(task.get("zarr_use") or "")
    stage_group = str(task.get("stage_group") or "")
    run_name = str(task.get("run_name") or "")
    component_name = str(task.get("component_name") or "")
    work_url = _work_filter_url(dataset_id=dataset_id, recording_id=recording_id, task_id=task_id, workflow=workflow_kind)
    expected_user_work_url = _work_filter_url(
        expected_user=expected_user,
        dataset_id=dataset_id,
        recording_id=recording_id,
        task_id=task_id,
        workflow=workflow_kind,
    )
    open_for_labeling = state in LABELER_START_TASK_STATES
    labeler_start_ready = open_for_labeling and not reassignment_session_safety_blocked
    direct_browser_start_endpoint = (
        f"/api/tasks/{quote(task_id, safe='')}/open"
        if task_id and labeler_start_ready
        else ""
    )
    blocked_reason = "reassignment_session_safety_failed" if reassignment_session_safety_blocked else ""
    support_context_fields = dict(support_context or {})
    labeler_action = (
        "wait_for_operator"
        if reassignment_session_safety_blocked
        else "open_task"
        if open_for_labeling
        else "none"
    )
    direct_start_not_ready_reasons: list[str] = []
    if not task_id:
        direct_start_not_ready_reasons.append("missing_task_id")
    if not open_for_labeling:
        direct_start_not_ready_reasons.append("task_complete" if state == "complete" else "task_not_startable")
    if reassignment_session_safety_blocked:
        direct_start_not_ready_reasons.append("reassignment_session_safety_failed")
    if not labeler_start_ready and not direct_start_not_ready_reasons:
        direct_start_not_ready_reasons.append("labeler_start_not_ready")
    direct_start_not_ready_reason = direct_start_not_ready_reasons[0] if direct_start_not_ready_reasons else ""
    direct_start_operator_action = (
        _reassignment_session_safety_operator_action()
        if reassignment_session_safety_blocked
        else "reopen_or_move_task_to_startable_state"
        if direct_start_not_ready_reason in {"task_complete", "task_not_startable"}
        else ""
    )
    return {
        "task_id": task_id,
        "title": str(task.get("title") or task_id or task.get("workflow_kind") or "task"),
        "workflow_kind": workflow_kind,
        "state": state,
        "priority": task.get("priority"),
        "zarr_use": zarr_use,
        "stage_group": stage_group,
        "run_name": run_name,
        "component_name": component_name,
        "notes": str(task.get("notes") or ""),
        "work_url": work_url,
        "expected_user_work_url": expected_user_work_url,
        "direct_browser_start_endpoint": direct_browser_start_endpoint,
        "direct_browser_start_method": "POST" if direct_browser_start_endpoint else "",
        "direct_browser_start_uses_existing_task_open_api": bool(direct_browser_start_endpoint),
        "direct_browser_start_requires_expected_user_guard": bool(direct_browser_start_endpoint),
        "direct_browser_start_authorization_contract_ready": labeler_start_ready,
        "direct_browser_start_expected_user_guard_required": True,
        "direct_browser_start_expected_user_guard_enforced_by_api": True,
        "direct_browser_start_server_rechecks_on_post": True,
        "direct_browser_start_not_ready_reason": direct_start_not_ready_reason,
        "direct_browser_start_not_ready_reasons": list(direct_start_not_ready_reasons),
        "direct_browser_start_operator_action": direct_start_operator_action,
        "direct_browser_start_authorization_contract": {
            "schema": "palette.web_labeling_dataset_queue_direct_start_authorization_contract.v1",
            "ready": labeler_start_ready,
            "not_ready_reason": direct_start_not_ready_reason,
            "not_ready_reasons": list(direct_start_not_ready_reasons),
            "task_id": task_id,
            "recording_id": recording_id,
            "expected_user": str(expected_user or ""),
            "direct_browser_start_endpoint": direct_browser_start_endpoint,
            "method": "POST" if direct_browser_start_endpoint else "",
            "uses_existing_task_open_api": bool(direct_browser_start_endpoint),
            "same_origin_only": True,
            "exact_route_required": True,
            "endpoint_task_segment_must_match_row_task_id": True,
            "expected_user_guard_required": True,
            "expected_user_guard_enforced_by_api": True,
            "known_assignment_store_user_required": True,
            "active_assignment_required": True,
            "active_assignment_present": True,
            "task_assigned_to_expected_user": True,
            "task_open_requires_startable_task_state": True,
            "task_state_startable": open_for_labeling,
            "reassignment_session_safety_checked_for_row": True,
            "reassignment_session_safety_passed": not reassignment_session_safety_blocked,
            "operator_action": direct_start_operator_action,
            "server_rechecks_on_post": True,
            "server_creates_session": True,
            "client_authorizes_open": False,
            "server_authorizes_open": True,
            "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
            "label_mutation_target_kind": "task_scoped_training_zarr",
            "browser_label_write_target": "training_zarr",
            "csv_handoff_artifact_role": "metadata_only_control_plane",
            "csv_handoff_artifacts_are_label_write_targets": False,
            "handoff_csv_artifacts_are_label_write_targets": False,
            "intermediate_csv_artifacts_are_label_write_targets": False,
            "browser_writes_csv_or_handoff_files": False,
            "browser_writes_handoff_csv": False,
            "browser_writes_intermediate_csv": False,
            "browser_receives_zarr_write_authority": False,
            "browser_has_direct_zarr_write_authority": False,
        },
        "labeler_start_ready": labeler_start_ready,
        "labeler_action": labeler_action,
        "blocked_reason": blocked_reason,
            "labeler_start_operator_action": (
            direct_start_operator_action
            if reassignment_session_safety_blocked
            else ""
        ),
        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
        "authoritative_label_state": "assigned_task_zarr_scope",
        "mutable_label_data_plane": "task_scoped_training_zarr",
        "label_mutation_target_kind": "task_scoped_training_zarr",
        "browser_label_write_target": "training_zarr",
        "training_zarr_mutations_are_server_owned": True,
        "handoff_artifacts_are_metadata_only": True,
        "csv_handoff_artifact_role": "metadata_only_control_plane",
        "csv_handoff_artifacts_are_label_write_targets": False,
        "handoff_csv_artifacts_are_label_write_targets": False,
        "intermediate_csv_artifacts_are_label_write_targets": False,
        "browser_writes_csv_or_handoff_files": False,
        "browser_writes_handoff_csv": False,
        "browser_writes_intermediate_csv": False,
        "browser_receives_zarr_write_authority": False,
        "browser_has_direct_zarr_write_authority": False,
        "operator_support": {
            **support_context_fields,
            "user": expected_user,
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "task_id": task_id,
            "workflow_kind": workflow_kind,
            "state": state,
            "zarr_use": zarr_use,
            "stage_group": stage_group,
            "run_name": run_name,
            "component_name": component_name,
            "expected_user_work_url": expected_user_work_url,
            "direct_browser_start_endpoint": direct_browser_start_endpoint,
            "direct_browser_start_method": "POST" if direct_browser_start_endpoint else "",
            "direct_browser_start_uses_existing_task_open_api": bool(direct_browser_start_endpoint),
            "direct_browser_start_requires_expected_user_guard": bool(direct_browser_start_endpoint),
            "direct_browser_start_authorization_contract_ready": labeler_start_ready,
            "direct_browser_start_expected_user_guard_required": True,
            "direct_browser_start_expected_user_guard_enforced_by_api": True,
            "direct_browser_start_server_rechecks_on_post": True,
            "direct_browser_start_not_ready_reason": direct_start_not_ready_reason,
            "direct_browser_start_not_ready_reasons": list(direct_start_not_ready_reasons),
            "direct_browser_start_operator_action": direct_start_operator_action,
            "blocked_reason": blocked_reason,
            "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
            "label_mutation_target_kind": "task_scoped_training_zarr",
            "browser_label_write_target": "training_zarr",
            "csv_handoff_artifact_role": "metadata_only_control_plane",
            "csv_handoff_artifacts_are_label_write_targets": False,
            "handoff_csv_artifacts_are_label_write_targets": False,
            "intermediate_csv_artifacts_are_label_write_targets": False,
            "browser_writes_csv_or_handoff_files": False,
            "browser_writes_handoff_csv": False,
            "browser_writes_intermediate_csv": False,
            "browser_receives_zarr_write_authority": False,
            "browser_has_direct_zarr_write_authority": False,
        },
    }

def _recording_blocked_by_reassignment_session_safety(
    reassignment_session_safety: Mapping[str, object] | None,
    recording_id: str,
) -> bool:
    """Return whether a recording has stale reassignment sessions blocking start."""
    recording_id = str(recording_id or "").strip()
    if not recording_id or not isinstance(reassignment_session_safety, Mapping):
        return False

    recording_id_fields = (
        "active_session_assignment_mismatch_recording_ids",
        "blocked_recording_ids",
        "recording_ids",
    )
    for field_name in recording_id_fields:
        values = reassignment_session_safety.get(field_name)
        if isinstance(values, (list, tuple, set)):
            if recording_id in {str(value).strip() for value in values}:
                return True

    by_recording_fields = (
        "active_session_assignment_mismatches_by_recording",
        "blocked_recordings",
        "blocked_recordings_by_reason",
    )
    for field_name in by_recording_fields:
        values = reassignment_session_safety.get(field_name)
        if isinstance(values, Mapping) and recording_id in {
            str(value).strip() for value in values.keys()
        }:
            return True

    session_fields = (
        "active_session_assignment_mismatches",
        "active_session_assignment_mismatch_sessions",
        "mismatched_sessions",
    )
    for field_name in session_fields:
        values = reassignment_session_safety.get(field_name)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, Mapping) and str(value.get("recording_id") or "").strip() == recording_id:
                return True
    return False

def _add_direct_start_contracts_to_work_tasks(
    work: MutableMapping[str, object],
    *,
    expected_user: str,
    reassignment_session_safety: Mapping[str, object] | None = None,
) -> None:
    summary: dict[str, object] = {
        "schema": "palette.web_labeling_direct_browser_start_contract_summary.v1",
        "ready": True,
        "task_count": 0,
        "ready_task_count": 0,
        "not_ready_task_count": 0,
        "not_ready_reason_counts": {},
        "operator_action_counts": {},
        "expected_user_guard_enforced_by_api": True,
        "server_rechecks_on_post": True,
        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
        "label_mutation_target_kind": "task_scoped_training_zarr",
        "browser_label_write_target": "training_zarr",
        "csv_handoff_artifact_role": "metadata_only_control_plane",
        "csv_handoff_artifacts_are_label_write_targets": False,
        "handoff_csv_artifacts_are_label_write_targets": False,
        "intermediate_csv_artifacts_are_label_write_targets": False,
        "browser_writes_csv_or_handoff_files": False,
        "browser_writes_handoff_csv": False,
        "browser_writes_intermediate_csv": False,
        "browser_receives_zarr_write_authority": False,
        "browser_has_direct_zarr_write_authority": False,
    }
    recordings = work.get("recordings")
    if not isinstance(recordings, list):
        summary["ready"] = False
        summary["not_ready_reason_counts"] = {"missing_recordings": 1}
        work["direct_browser_start_contract_summary"] = summary
        return
    safe_field_names = {
        "direct_browser_start_endpoint",
        "direct_browser_start_method",
        "direct_browser_start_uses_existing_task_open_api",
        "direct_browser_start_requires_expected_user_guard",
        "direct_browser_start_authorization_contract_ready",
        "direct_browser_start_expected_user_guard_required",
        "direct_browser_start_expected_user_guard_enforced_by_api",
        "direct_browser_start_server_rechecks_on_post",
        "direct_browser_start_not_ready_reason",
        "direct_browser_start_not_ready_reasons",
        "direct_browser_start_authorization_contract",
        "labeler_start_ready",
        "labeler_action",
        "blocked_reason",
        "labeler_start_operator_action",
        "data_plane_write_target",
        "authoritative_label_state",
        "mutable_label_data_plane",
        "label_mutation_target_kind",
        "browser_label_write_target",
        "training_zarr_mutations_are_server_owned",
        "handoff_artifacts_are_metadata_only",
        "csv_handoff_artifact_role",
        "csv_handoff_artifacts_are_label_write_targets",
        "handoff_csv_artifacts_are_label_write_targets",
        "intermediate_csv_artifacts_are_label_write_targets",
        "browser_writes_csv_or_handoff_files",
        "browser_writes_handoff_csv",
        "browser_writes_intermediate_csv",
        "browser_receives_zarr_write_authority",
        "browser_has_direct_zarr_write_authority",
        "operator_support",
    }
    support_context: dict[str, object] = {
        "expected_user_personal_dataset_queue_url": str(
            work.get("expected_user_personal_dataset_queue_url") or ""
        ),
        "personalized_labeler_entry_url": str(
            work.get("personalized_labeler_entry_url")
            or work.get("expected_user_personal_dataset_queue_url")
            or ""
        ),
        "personal_dataset_queue_link_role": str(
            work.get("personal_dataset_queue_link_role") or "preferred_queue"
        ),
        "canonical_dataset_queue_link_role": str(
            work.get("canonical_dataset_queue_link_role") or "canonical_queue_fallback"
        ),
        "browser_label_write_target": "training_zarr",
        "browser_writes_csv_or_handoff_files": False,
        "browser_has_direct_zarr_write_authority": False,
        "csv_handoff_artifact_role": "metadata_only_control_plane",
    }
    for recording in recordings:
        if not isinstance(recording, MutableMapping):
            continue
        recording_id = str(recording.get("recording_id") or "")
        recording_blocked = _recording_blocked_by_reassignment_session_safety(
            reassignment_session_safety,
            recording_id,
        )
        tasks = recording.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, MutableMapping):
                continue
            task_for_contract = dict(task)
            task_for_contract.setdefault("recording_id", recording_id)
            contract_fields = _work_dataset_queue_task(
                task_for_contract,
                expected_user=expected_user,
                reassignment_session_safety_blocked=recording_blocked,
                support_context=support_context,
            )
            summary["task_count"] = int(summary["task_count"]) + 1
            if bool(contract_fields.get("direct_browser_start_authorization_contract_ready")):
                summary["ready_task_count"] = int(summary["ready_task_count"]) + 1
            else:
                summary["not_ready_task_count"] = int(summary["not_ready_task_count"]) + 1
                reason_counts = summary["not_ready_reason_counts"]
                if isinstance(reason_counts, MutableMapping):
                    reasons = contract_fields.get("direct_browser_start_not_ready_reasons")
                    reason_values = [
                        str(reason)
                        for reason in (reasons if isinstance(reasons, list) else [])
                        if str(reason).strip()
                    ] or [str(contract_fields.get("direct_browser_start_not_ready_reason") or "unknown")]
                    for reason in reason_values:
                        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                operator_action = str(contract_fields.get("direct_browser_start_operator_action") or "").strip()
                if operator_action:
                    action_counts = summary["operator_action_counts"]
                    if isinstance(action_counts, MutableMapping):
                        action_counts[operator_action] = int(action_counts.get(operator_action, 0)) + 1
            for field_name in safe_field_names:
                if field_name in contract_fields:
                    task[field_name] = contract_fields[field_name]
    summary["ready"] = int(summary["task_count"]) > 0 and int(summary["not_ready_task_count"]) == 0
    if isinstance(summary.get("not_ready_reason_counts"), MutableMapping):
        summary["not_ready_reason_counts"] = dict(sorted(summary["not_ready_reason_counts"].items()))
    if isinstance(summary.get("operator_action_counts"), MutableMapping):
        summary["operator_action_counts"] = dict(sorted(summary["operator_action_counts"].items()))
    work["direct_browser_start_contract_summary"] = summary

def _work_filter_url(
    *,
    expected_user: str | None = None,
    dataset_id: str | None = None,
    recording_id: str | None = None,
    task_id: str | None = None,
    workflow: str | None = None,
) -> str:
    filters = [
        ("expected_user", str(expected_user or "").strip()),
        ("dataset_id", str(dataset_id or "").strip()),
        ("recording_id", str(recording_id or "").strip()),
        ("task_id", str(task_id or "").strip()),
        ("workflow", str(workflow or "").strip()),
    ]
    query = "&".join(f"{key}={quote(value, safe='')}" for key, value in filters if value)
    return f"{DASHBOARD_PATH}?{query}" if query else DASHBOARD_PATH

def _work_dataset_queue(
    work: Mapping[str, object],
    *,
    reassignment_session_safety: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    datasets: dict[str, dict[str, object]] = {}
    recording_index: dict[str, dict[str, dict[str, object]]] = {}
    expected_user = str(work.get("user") or "").strip()
    row_support_context: dict[str, object] = {
        "expected_user_personal_dataset_queue_url": str(
            work.get("expected_user_personal_dataset_queue_url") or ""
        ),
        "personalized_labeler_entry_url": str(
            work.get("personalized_labeler_entry_url")
            or work.get("expected_user_personal_dataset_queue_url")
            or ""
        ),
        "personal_dataset_queue_link_role": str(
            work.get("personal_dataset_queue_link_role") or "preferred_queue"
        ),
        "canonical_dataset_queue_link_role": str(
            work.get("canonical_dataset_queue_link_role") or "canonical_queue_fallback"
        ),
        "browser_label_write_target": "training_zarr",
        "browser_writes_csv_or_handoff_files": False,
        "browser_has_direct_zarr_write_authority": False,
        "csv_handoff_artifact_role": "metadata_only_control_plane",
    }
    for recording in work.get("recordings", []):
        if not isinstance(recording, Mapping):
            continue
        recording_id = str(recording.get("recording_id") or "")
        assignment_notes = str(recording.get("assignment_notes") or "")
        for task in recording.get("tasks", []):
            if not isinstance(task, Mapping):
                continue
            task_state = str(task.get("state") or "")
            if task_state == "complete":
                continue
            dataset_id = str(task.get("dataset_id") or "").strip()
            dataset_key = dataset_id or "__unspecified_dataset__"
            entry = datasets.setdefault(
                dataset_key,
                {
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_id or "Unspecified dataset",
                    "work_url": _work_filter_url(dataset_id=dataset_id),
                    "expected_user_work_url": _work_filter_url(expected_user=expected_user, dataset_id=dataset_id),
                    "operator_support": {
                        **row_support_context,
                        "user": expected_user,
                        "dataset_id": dataset_id,
                        "dataset_label": dataset_id or "Unspecified dataset",
                        "expected_user_work_url": _work_filter_url(expected_user=expected_user, dataset_id=dataset_id),
                    },
                    "task_count": 0,
                    "open_task_count": 0,
                    "complete_task_count": 0,
                    "non_startable_task_count": 0,
                    "recording_count": 0,
                    "workflow_counts": {},
                    "zarr_use_counts": {},
                    "max_priority": None,
                    "recordings": [],
                },
            )
            entry["task_count"] = int(entry.get("task_count") or 0) + 1
            reassignment_blocked = _reassignment_session_safety_blocks_recording(
                recording_id,
                reassignment_session_safety,
            )
            task_startable = task_state in LABELER_START_TASK_STATES and not reassignment_blocked
            if task_state == "complete":
                entry["complete_task_count"] = int(entry.get("complete_task_count") or 0) + 1
            elif task_startable:
                entry["open_task_count"] = int(entry.get("open_task_count") or 0) + 1
            else:
                entry["non_startable_task_count"] = int(entry.get("non_startable_task_count") or 0) + 1
            priority = _task_priority_value(task)
            current_max_priority = entry.get("max_priority")
            if current_max_priority is None or priority > _task_priority_value({"priority": current_max_priority}):
                entry["max_priority"] = task.get("priority", priority)
            workflow_counts = entry["workflow_counts"]
            assert isinstance(workflow_counts, dict)
            workflow = str(task.get("workflow_kind") or "unknown")
            workflow_counts[workflow] = int(workflow_counts.get(workflow, 0)) + 1
            zarr_use_counts = entry["zarr_use_counts"]
            assert isinstance(zarr_use_counts, dict)
            zarr_use = str(task.get("zarr_use") or "").strip() or "unspecified"
            zarr_use_counts[zarr_use] = int(zarr_use_counts.get(zarr_use, 0)) + 1
            per_dataset_recordings = recording_index.setdefault(dataset_key, {})
            recording_entry = per_dataset_recordings.get(recording_id)
            if recording_entry is None:
                recording_entry = {
                    "recording_id": recording_id,
                    "assignment_notes": assignment_notes,
                    "work_url": _work_filter_url(dataset_id=dataset_id, recording_id=recording_id),
                    "expected_user_work_url": _work_filter_url(
                        expected_user=expected_user,
                        dataset_id=dataset_id,
                        recording_id=recording_id,
                    ),
                    "operator_support": {
                        **row_support_context,
                        "user": expected_user,
                        "dataset_id": dataset_id,
                        "recording_id": recording_id,
                        "expected_user_work_url": _work_filter_url(
                            expected_user=expected_user,
                            dataset_id=dataset_id,
                            recording_id=recording_id,
                        ),
                    },
                    "task_count": 0,
                    "open_task_count": 0,
                    "complete_task_count": 0,
                    "non_startable_task_count": 0,
                    "workflow_counts": {},
                    "tasks": [],
                }
                per_dataset_recordings[recording_id] = recording_entry
                recordings = entry["recordings"]
                assert isinstance(recordings, list)
                recordings.append(recording_entry)
            recording_entry["task_count"] = int(recording_entry.get("task_count") or 0) + 1
            if task_state == "complete":
                recording_entry["complete_task_count"] = int(recording_entry.get("complete_task_count") or 0) + 1
            elif task_startable:
                recording_entry["open_task_count"] = int(recording_entry.get("open_task_count") or 0) + 1
            else:
                recording_entry["non_startable_task_count"] = int(recording_entry.get("non_startable_task_count") or 0) + 1
            recording_workflows = recording_entry["workflow_counts"]
            assert isinstance(recording_workflows, dict)
            recording_workflows[workflow] = int(recording_workflows.get(workflow, 0)) + 1
            queue_tasks = recording_entry["tasks"]
            assert isinstance(queue_tasks, list)
            queue_tasks.append(
                _work_dataset_queue_task(
                    task,
                    expected_user=expected_user,
                    reassignment_session_safety_blocked=reassignment_blocked,
                    support_context=row_support_context,
                )
            )
    rows = list(datasets.values())
    for row in rows:
        recordings = row.get("recordings") if isinstance(row.get("recordings"), list) else []
        row["recording_count"] = len(recordings)
        row_open_task_count = int(row.get("open_task_count") or 0)
        row_safety_blocked = any(
            bool(task.get("blocked_reason") == "reassignment_session_safety_failed")
            for recording in recordings
            if isinstance(recording, Mapping)
            for task in (
                recording.get("tasks")
                if isinstance(recording.get("tasks"), list)
                else []
            )
            if isinstance(task, Mapping)
        )
        row["labeler_start_ready"] = row_open_task_count > 0 and not row_safety_blocked
        row["labeler_action"] = (
            "wait_for_operator"
            if row_safety_blocked
            else "open_dataset"
            if row_open_task_count > 0
            else "none"
        )
        row["blocked_reason"] = (
            "reassignment_session_safety_failed"
            if row_safety_blocked
            else ""
            if row_open_task_count > 0
            else "no_open_tasks"
        )
        row["data_plane_write_target"] = "server_owned_assigned_task_zarr_scope"
        row["authoritative_label_state"] = "assigned_task_zarr_scope"
        row["mutable_label_data_plane"] = "task_scoped_training_zarr"
        row["label_mutation_target_kind"] = "task_scoped_training_zarr"
        row["browser_label_write_target"] = "training_zarr"
        row["training_zarr_mutations_are_server_owned"] = True
        row["handoff_artifacts_are_metadata_only"] = True
        row["csv_handoff_artifact_role"] = "metadata_only_control_plane"
        row["csv_handoff_artifacts_are_label_write_targets"] = False
        row["handoff_csv_artifacts_are_label_write_targets"] = False
        row["intermediate_csv_artifacts_are_label_write_targets"] = False
        row["browser_writes_csv_or_handoff_files"] = False
        row["browser_writes_handoff_csv"] = False
        row["browser_writes_intermediate_csv"] = False
        row["browser_receives_zarr_write_authority"] = False
        row["browser_has_direct_zarr_write_authority"] = False
        row_support = row.get("operator_support") if isinstance(row.get("operator_support"), dict) else {}
        row_support.update(
            {
                "task_count": row.get("task_count", 0),
                "open_task_count": row.get("open_task_count", 0),
                "non_startable_task_count": row.get("non_startable_task_count", 0),
                "recording_count": row.get("recording_count", 0),
                "workflow_counts": row.get("workflow_counts", {}),
            }
        )
        row["operator_support"] = row_support
        for recording in recordings:
            if not isinstance(recording, dict):
                continue
            recording_open_task_count = int(recording.get("open_task_count") or 0)
            recording_safety_blocked = any(
                bool(task.get("blocked_reason") == "reassignment_session_safety_failed")
                for task in (
                    recording.get("tasks")
                    if isinstance(recording.get("tasks"), list)
                    else []
                )
                if isinstance(task, Mapping)
            )
            recording["labeler_start_ready"] = recording_open_task_count > 0 and not recording_safety_blocked
            recording["labeler_action"] = (
                "wait_for_operator"
                if recording_safety_blocked
                else "open_recording"
                if recording_open_task_count > 0
                else "none"
            )
            recording["blocked_reason"] = (
                "reassignment_session_safety_failed"
                if recording_safety_blocked
                else ""
                if recording_open_task_count > 0
                else "no_open_tasks"
            )
            recording["data_plane_write_target"] = "server_owned_assigned_task_zarr_scope"
            recording["authoritative_label_state"] = "assigned_task_zarr_scope"
            recording["mutable_label_data_plane"] = "task_scoped_training_zarr"
            recording["training_zarr_mutations_are_server_owned"] = True
            recording["handoff_artifacts_are_metadata_only"] = True
            recording["browser_writes_csv_or_handoff_files"] = False
            recording["browser_receives_zarr_write_authority"] = False
            recording["browser_has_direct_zarr_write_authority"] = False
            recording_support = recording.get("operator_support") if isinstance(recording.get("operator_support"), dict) else {}
            recording_support.update(
                {
                    "task_count": recording.get("task_count", 0),
                    "open_task_count": recording.get("open_task_count", 0),
                    "non_startable_task_count": recording.get("non_startable_task_count", 0),
                    "workflow_counts": recording.get("workflow_counts", {}),
                }
            )
            recording["operator_support"] = recording_support
            tasks = recording.get("tasks") if isinstance(recording.get("tasks"), list) else []
            tasks.sort(
                key=lambda task: (
                    -_task_priority_value(task if isinstance(task, Mapping) else {}),
                    str(task.get("title") or "") if isinstance(task, Mapping) else "",
                    str(task.get("task_id") or "") if isinstance(task, Mapping) else "",
                )
            )
        recordings.sort(key=lambda recording: str(recording.get("recording_id") or "") if isinstance(recording, Mapping) else "")
    rows.sort(
        key=lambda row: (
            -_task_priority_value({"priority": row.get("max_priority")}),
            str(row.get("dataset_label") or ""),
            str(row.get("dataset_id") or ""),
        )
    )
    return rows

def _work_dataset_queue_summary(dataset_queue: Sequence[Mapping[str, object]]) -> dict[str, object]:
    dataset_rows = [row for row in dataset_queue if isinstance(row, Mapping)]
    dataset_ids = [str(row.get("dataset_id") or "") for row in dataset_rows if str(row.get("dataset_id") or "").strip()]
    return {
        "dataset_count": len(dataset_rows),
        "waiting_dataset_count": sum(1 for row in dataset_rows if int(row.get("open_task_count") or 0) > 0),
        "open_task_count": sum(int(row.get("open_task_count") or 0) for row in dataset_rows),
        "non_startable_task_count": sum(int(row.get("non_startable_task_count") or 0) for row in dataset_rows),
        "complete_task_count": sum(int(row.get("complete_task_count") or 0) for row in dataset_rows),
        "task_count": sum(int(row.get("task_count") or 0) for row in dataset_rows),
        "dataset_ids": dataset_ids,
    }

def _service_absolute_url(base_url: str | None, path_or_url: object) -> str:
    text = str(path_or_url or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    if not str(base_url or "").strip():
        return text
    if not text.startswith("/"):
        return text
    return f"{str(base_url).rstrip('/')}{text}"

def _first_dataset_queue_url(dataset_queue: Sequence[Mapping[str, object]], *, base_url: str | None = None) -> str:
    for dataset in dataset_queue:
        if not isinstance(dataset, Mapping):
            continue
        candidate = str(dataset.get("expected_user_work_url") or dataset.get("work_url") or "").strip()
        if candidate:
            return _service_absolute_url(base_url, candidate)
        recordings = dataset.get("recordings")
        if not isinstance(recordings, list):
            continue
        for recording in recordings:
            if not isinstance(recording, Mapping):
                continue
            candidate = str(recording.get("expected_user_work_url") or recording.get("work_url") or "").strip()
            if candidate:
                return _service_absolute_url(base_url, candidate)
    return ""

def _reassignment_session_safety_flat_fields(
    reassignment_session_safety: Mapping[str, object] | None,
) -> dict[str, object]:
    safety = reassignment_session_safety if isinstance(reassignment_session_safety, Mapping) else {}
    mismatch_session_ids = (
        safety.get("active_session_assignment_mismatch_session_ids")
        if isinstance(safety.get("active_session_assignment_mismatch_session_ids"), list)
        else []
    )
    mismatch_recording_ids = (
        safety.get("active_session_assignment_mismatch_recording_ids")
        if isinstance(safety.get("active_session_assignment_mismatch_recording_ids"), list)
        else []
    )
    return {
        "reassignment_session_safety_ok": bool(safety.get("ok", True)),
        "reassignment_session_safety_blocks_labeler_mutation": bool(
            safety.get("blocks_labeler_mutation")
        ),
        "reassignment_session_safety_active_session_assignment_mismatch_count": int(
            safety.get("active_session_assignment_mismatch_count") or 0
        ),
        "reassignment_session_safety_active_session_assignment_mismatch_session_ids": [
            str(session_id)
            for session_id in mismatch_session_ids
            if str(session_id).strip()
        ],
        "reassignment_session_safety_active_session_assignment_mismatch_recording_ids": [
            str(recording_id)
            for recording_id in mismatch_recording_ids
            if str(recording_id).strip()
        ],
        "reassignment_session_safety_requires_operator_recovery": bool(
            safety.get("requires_operator_recovery")
        ),
        "reassignment_session_safety_operator_action": str(
            safety.get("operator_action") or ""
        ),
    }

def _add_work_summary_fields(
    work: dict[str, object],
    *,
    reassignment_session_safety: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(reassignment_session_safety, Mapping):
        public_reassignment_session_safety = _public_reassignment_session_safety_fields(
            reassignment_session_safety,
            recording_ids=_work_recording_ids(work),
        )
        work["reassignment_session_safety"] = public_reassignment_session_safety
        work.update(_reassignment_session_safety_flat_fields(public_reassignment_session_safety))
    work["empty_state"] = _work_empty_state(work)
    work["progress_summary"] = _work_progress_summary(work)
    work.update(_operator_validation_visibility_fields())
    dataset_queue = _work_dataset_queue(
        work,
        reassignment_session_safety=reassignment_session_safety,
    )
    work["dataset_queue"] = dataset_queue
    work["dataset_queue_summary"] = _work_dataset_queue_summary(dataset_queue)
    dataset_queue_state = _dataset_queue_state(work)
    work["dataset_queue_state"] = dataset_queue_state
    work.update(_dataset_queue_labeler_start_fields(dataset_queue_state))
    labeler_work_completion = _labeler_work_completion_contract(
        dataset_queue_state=dataset_queue_state,
        progress_summary=work["progress_summary"]
        if isinstance(work.get("progress_summary"), Mapping)
        else {},
        dataset_queue_summary=work["dataset_queue_summary"]
        if isinstance(work.get("dataset_queue_summary"), Mapping)
        else {},
    )
    work["labeler_work_completion"] = labeler_work_completion
    work.update(_labeler_work_completion_fields(labeler_work_completion))
    return work

def _dataset_queue_labeler_start_fields(dataset_queue_state: Mapping[str, object]) -> dict[str, object]:
    code = str(dataset_queue_state.get("code") or "")
    has_open_dataset_work = bool(dataset_queue_state.get("has_open_dataset_work"))
    blocks_start = bool(dataset_queue_state.get("blocks_labeler_start"))
    if has_open_dataset_work:
        action = "open_dataset_queue"
    elif code == "all_assigned_work_complete":
        action = "complete"
    elif code == "no_active_assignments":
        action = "wait_for_assignment"
    elif code == "assigned_recordings_need_operator_action":
        action = "wait_for_operator"
    elif blocks_start:
        action = "wait_for_operator"
    else:
        action = "wait_for_work"
    return {
        "labeler_start_ready": has_open_dataset_work and not blocks_start,
        "labeler_start_status": code or "unknown",
        "labeler_action": action,
        "labeler_start_message": str(dataset_queue_state.get("message") or ""),
        "labeler_start_operator_action": str(dataset_queue_state.get("operator_action") or ""),
    }

def _labeler_work_completion_contract(
    *,
    dataset_queue_state: Mapping[str, object],
    progress_summary: Mapping[str, object],
    dataset_queue_summary: Mapping[str, object],
) -> dict[str, object]:
    code = str(dataset_queue_state.get("code") or "")
    has_open_dataset_work = bool(dataset_queue_state.get("has_open_dataset_work"))
    blocks_labeler_start = bool(dataset_queue_state.get("blocks_labeler_start"))
    total_task_count = int(progress_summary.get("total_task_count") or 0)
    complete_task_count = int(progress_summary.get("complete_task_count") or 0)
    open_task_count = int(
        progress_summary.get("open_task_count")
        or dataset_queue_summary.get("open_task_count")
        or 0
    )
    waiting_dataset_count = int(dataset_queue_summary.get("waiting_dataset_count") or 0)
    waiting_recording_count = int(progress_summary.get("waiting_recording_count") or 0)
    complete_recording_count = int(progress_summary.get("complete_recording_count") or 0)
    blocked_recording_count = int(progress_summary.get("blocked_recording_count") or 0)
    if code == "all_assigned_work_complete":
        status = "complete"
        labeler_action = "complete"
    elif has_open_dataset_work and not blocks_labeler_start:
        status = "waiting"
        labeler_action = "open_dataset_queue"
    elif code == "no_active_assignments":
        status = "unassigned"
        labeler_action = "wait_for_assignment"
    elif blocks_labeler_start:
        status = "blocked"
        labeler_action = "wait_for_operator"
    else:
        status = "idle"
        labeler_action = "wait_for_work"
    operator_action = str(dataset_queue_state.get("operator_action") or "")
    return {
        "schema": "palette.web_labeling_labeler_work_completion.v1",
        "status": status,
        "dataset_queue_state_code": code,
        "completed": status == "complete",
        "has_waiting_work": status == "waiting",
        "ready_for_more_labeling": status == "waiting",
        "blocks_labeler_start": blocks_labeler_start,
        "operator_action_required": status == "blocked",
        "labeler_action": labeler_action,
        "message": str(dataset_queue_state.get("message") or ""),
        "operator_action": operator_action,
        "completion_state": _completion_state(
            complete_tasks=complete_task_count,
            total_tasks=total_task_count,
        ),
        "completion_percent": _completion_percent(complete_task_count, total_task_count),
        "total_task_count": total_task_count,
        "complete_task_count": complete_task_count,
        "open_task_count": open_task_count,
        "waiting_dataset_count": waiting_dataset_count,
        "waiting_recording_count": waiting_recording_count,
        "complete_recording_count": complete_recording_count,
        "blocked_recording_count": blocked_recording_count,
    }

def _labeler_work_completion_fields(
    completion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = completion if isinstance(completion, Mapping) else {}
    return {
        "labeler_work_completion_schema": str(source.get("schema") or ""),
        "labeler_work_completion_status": str(source.get("status") or ""),
        "labeler_work_completion_dataset_queue_state_code": str(
            source.get("dataset_queue_state_code") or ""
        ),
        "labeler_work_completion_completed": bool(source.get("completed")),
        "labeler_work_completion_has_waiting_work": bool(source.get("has_waiting_work")),
        "labeler_work_completion_ready_for_more_labeling": bool(
            source.get("ready_for_more_labeling")
        ),
        "labeler_work_completion_blocks_labeler_start": bool(
            source.get("blocks_labeler_start")
        ),
        "labeler_work_completion_operator_action_required": bool(
            source.get("operator_action_required")
        ),
        "labeler_work_completion_labeler_action": str(source.get("labeler_action") or ""),
        "labeler_work_completion_completion_state": str(source.get("completion_state") or ""),
        "labeler_work_completion_completion_percent": source.get("completion_percent"),
        "labeler_work_completion_total_task_count": int(source.get("total_task_count") or 0),
        "labeler_work_completion_complete_task_count": int(
            source.get("complete_task_count") or 0
        ),
        "labeler_work_completion_open_task_count": int(source.get("open_task_count") or 0),
        "labeler_work_completion_waiting_dataset_count": int(
            source.get("waiting_dataset_count") or 0
        ),
        "labeler_work_completion_waiting_recording_count": int(
            source.get("waiting_recording_count") or 0
        ),
        "labeler_work_completion_complete_recording_count": int(
            source.get("complete_recording_count") or 0
        ),
        "labeler_work_completion_blocked_recording_count": int(
            source.get("blocked_recording_count") or 0
        ),
    }

def _dataset_queue_state(work: Mapping[str, object]) -> dict[str, object]:
    empty_state = work.get("empty_state") if isinstance(work.get("empty_state"), Mapping) else _work_empty_state(work)
    progress = work.get("progress_summary") if isinstance(work.get("progress_summary"), Mapping) else _work_progress_summary(work)
    reassignment_session_safety = (
        work.get("reassignment_session_safety")
        if isinstance(work.get("reassignment_session_safety"), Mapping)
        else {}
    )
    dataset_queue = work.get("dataset_queue") if isinstance(work.get("dataset_queue"), Sequence) else ()
    summary = (
        work.get("dataset_queue_summary")
        if isinstance(work.get("dataset_queue_summary"), Mapping)
        else _work_dataset_queue_summary([row for row in dataset_queue if isinstance(row, Mapping)])
    )
    waiting_dataset_count = int(summary.get("waiting_dataset_count") or 0)
    open_task_count = int(summary.get("open_task_count") or 0)
    non_startable_task_count = int(summary.get("non_startable_task_count") or 0)
    complete_task_count = int(progress.get("complete_task_count") or summary.get("complete_task_count") or 0)
    total_task_count = int(progress.get("total_task_count") or summary.get("task_count") or 0)
    waiting_recording_count = int(progress.get("waiting_recording_count") or 0)
    complete_recording_count = int(progress.get("complete_recording_count") or 0)
    blocked_recording_count = int(progress.get("blocked_recording_count") or 0)
    empty_code = str(empty_state.get("code") or "")
    has_open_dataset_work = waiting_dataset_count > 0 or open_task_count > 0
    if reassignment_session_safety and not bool(reassignment_session_safety.get("ok", True)):
        code = "reassignment_session_safety_failed"
        title = "Assigned recording sessions need operator recovery."
        message = "Stale previous-owner sessions are still open, so browser labeling is temporarily blocked."
        operator_action = str(
            reassignment_session_safety.get("operator_action")
            or _reassignment_session_safety_operator_action()
        )
        has_open_dataset_work = False
    elif has_open_dataset_work:
        code = "has_open_dataset_work"
        title = "Datasets are waiting for completion."
        message = "Open a dataset, recording, or task from this queue."
        operator_action = ""
    elif empty_code == "no_active_assignments":
        code = "no_active_assignments"
        title = "No active labeling recordings are assigned to this user."
        message = str(empty_state.get("message") or "")
        operator_action = str(empty_state.get("operator_action") or "")
    elif empty_code == "all_tasks_complete" or (
        total_task_count > 0 and complete_task_count >= total_task_count and blocked_recording_count <= 0
    ):
        code = "all_assigned_work_complete"
        title = "All assigned dataset work is complete."
        message = str(
            empty_state.get("message")
            or "There is no open dataset work waiting. Reopen or assign new tasks only if more labeling is required."
        )
        operator_action = str(empty_state.get("operator_action") or "")
    elif blocked_recording_count > 0:
        code = "assigned_recordings_need_operator_action"
        title = "Assigned recordings need operator action before more labeling."
        message = str(
            empty_state.get("message")
            or "Assigned recordings currently have no startable queue tasks. The operator may need to generate tasks, reopen completed work, move blocked tasks to pending/in_progress, or inspect task visibility."
        )
        operator_action = str(empty_state.get("operator_action") or "")
    elif empty_code == "no_open_tasks":
        code = "no_open_dataset_work"
        title = "No open dataset work is currently waiting for completion."
        message = str(empty_state.get("message") or "")
        operator_action = str(empty_state.get("operator_action") or "")
    else:
        code = "no_open_dataset_work"
        title = "No open dataset work is currently waiting for completion."
        message = "Refresh the queue or ask the operator to inspect this assignment if work was expected."
        operator_action = str(empty_state.get("operator_action") or "")
    return {
        "schema": "palette.web_labeling_dataset_queue_state.v1",
        "code": code,
        "is_empty": not has_open_dataset_work,
        "has_open_dataset_work": has_open_dataset_work,
        "blocks_labeler_start": not has_open_dataset_work,
        "title": title,
        "message": message,
        "operator_action": operator_action,
        "empty_state_code": empty_code,
        "counts": {
            "waiting_dataset_count": waiting_dataset_count,
            "open_task_count": open_task_count,
            "non_startable_task_count": non_startable_task_count,
            "complete_task_count": complete_task_count,
            "total_task_count": total_task_count,
            "waiting_recording_count": waiting_recording_count,
            "complete_recording_count": complete_recording_count,
            "blocked_recording_count": blocked_recording_count,
        },
        "blocked_recordings_by_reason": dict(progress.get("blocked_recordings_by_reason") or {}),
    }

def _completion_percent(complete_tasks: int, total_tasks: int) -> float | None:
    if total_tasks <= 0:
        return None
    return round(100.0 * max(0, int(complete_tasks)) / max(1, int(total_tasks)), 1)

def _completion_state(*, complete_tasks: int, total_tasks: int) -> str:
    if total_tasks <= 0:
        return "no_tasks"
    if complete_tasks >= total_tasks:
        return "complete"
    if complete_tasks <= 0:
        return "not_started"
    return "in_progress"
