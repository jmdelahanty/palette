"""Authorization and mutation response metadata helpers for web labeling."""

from __future__ import annotations

from typing import Mapping

from .assignment_store import LABELER_START_TASK_STATES
from .admin_dashboard import (
    _dataset_queue_direct_start_policy,
    _runtime_operator_validation_mutation_gate_not_required,
    _runtime_operator_validation_start_gate_not_required,
)
from .web_auth import PERSONAL_DATASET_QUEUE_PATH, _dashboard_url_for_expected_user
from .web_diagnostics import _add_payload_contract_compact_fields
from .web_policy import (
    LABELING_HOME_PATH,
    PERSONAL_WORK_PATH,
    _browser_mutation_write_contract_policy,
    _browser_mutation_write_policy,
    _browser_mutation_write_runtime_checklist,
)


def _task_open_authorization_contract(
    *,
    user: str,
    expected_user: str | None,
    task: Mapping[str, object] | None,
    ready: bool,
    not_ready_reason: str | None = None,
    session_created_server_side: bool = False,
    server_authorizes_open: bool = False,
    reassignment_session_safety_checked_server_side: bool | None = None,
    reassignment_session_safety_passed: bool | None = None,
    operator_validation_start_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    direct_start_policy = _dataset_queue_direct_start_policy()
    task_state = str((task or {}).get("state") or "")
    task_assigned_to_resolved_user = bool(
        task is not None and str(task.get("assignee_user") or "") == resolved_user
    )
    assignment_status_active = bool(
        task is not None and str(task.get("assignment_status") or "") == "active"
    )
    active_assignment_present = bool(
        task_assigned_to_resolved_user and assignment_status_active
    )
    task_state_startable = bool(task_state in LABELER_START_TASK_STATES)
    reason = str(not_ready_reason or "").strip()
    reassignment_session_safety_checked = (
        bool(reassignment_session_safety_checked_server_side)
        if reassignment_session_safety_checked_server_side is not None
        else bool(ready or reason in {"reassignment_session_safety_failed", "session_open_failed"})
    )
    reassignment_session_safety_ok = (
        bool(reassignment_session_safety_passed)
        if reassignment_session_safety_passed is not None
        else bool(reassignment_session_safety_checked and reason != "reassignment_session_safety_failed")
    )
    operator_gate = (
        dict(operator_validation_start_gate)
        if isinstance(operator_validation_start_gate, Mapping)
        else _runtime_operator_validation_start_gate_not_required()
    )
    return {
        "schema": "palette.web_labeling_task_open_authorization_contract.v1",
        "ready": bool(ready),
        "not_ready_reason": reason,
        "expected_user_guard_checked_server_side": True,
        "expected_user_guard_present": bool(expected),
        "expected_user_guard_required_for_handoff_links": True,
        "expected_user_matches_resolved_user": not expected or expected == resolved_user,
        "known_assignment_store_user_required": True,
        "active_assignment_required": True,
        "active_assignment_present": active_assignment_present,
        "task_reloaded_server_side": True,
        "task_assigned_to_resolved_user": task_assigned_to_resolved_user,
        "assignment_status_active": assignment_status_active,
        "task_open_requires_startable_task_state": True,
        "task_state_startable": task_state_startable,
        "task_complete_rejected_before_session_create": True,
        "reassignment_session_safety_checked_server_side": reassignment_session_safety_checked,
        "reassignment_session_safety_passed": reassignment_session_safety_ok,
        "session_created_server_side": bool(session_created_server_side),
        "superseded_sessions_closed_before_session_returned": bool(session_created_server_side),
        "client_authorizes_open": False,
        "server_authorizes_open": bool(server_authorizes_open),
        "operator_validation_start_gate": operator_gate,
        "operator_validation_start_gate_checked_server_side": True,
        "operator_validation_start_gate_required": bool(
            operator_gate.get("required_for_browser_start")
        ),
        "operator_validation_start_gate_ready": bool(operator_gate.get("ready")),
        "operator_validation_start_gate_blocks_task_open": bool(
            operator_gate.get("blocks_task_open")
        ),
        "operator_validation_start_gate_not_ready_reason": str(
            operator_gate.get("not_ready_reason") or ""
        ),
        "operator_validation_status": str(operator_gate.get("operator_validation_status") or ""),
        "operator_validation_pending_gate_ids": list(
            operator_gate.get("operator_validation_pending_gate_ids")
            if isinstance(operator_gate.get("operator_validation_pending_gate_ids"), list)
            else []
        ),
        "operator_validation_required_missing_evidence_gate_ids": list(
            operator_gate.get("operator_validation_required_missing_evidence_gate_ids")
            if isinstance(
                operator_gate.get("operator_validation_required_missing_evidence_gate_ids"),
                list,
            )
            else []
        ),
        "data_plane_write_target": str(direct_start_policy.get("data_plane_write_target") or ""),
        "label_mutation_target_kind": str(direct_start_policy.get("label_mutation_target_kind") or ""),
        "browser_label_write_target": str(direct_start_policy.get("browser_label_write_target") or ""),
        "csv_handoff_artifact_role": str(direct_start_policy.get("csv_handoff_artifact_role") or ""),
        "csv_handoff_artifacts_are_label_write_targets": bool(
            direct_start_policy.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "handoff_csv_artifacts_are_label_write_targets": bool(
            direct_start_policy.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "intermediate_csv_artifacts_are_label_write_targets": bool(
            direct_start_policy.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "browser_writes_csv_or_handoff_files": bool(
            direct_start_policy.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_writes_handoff_csv": bool(direct_start_policy.get("browser_writes_handoff_csv")),
        "browser_writes_intermediate_csv": bool(
            direct_start_policy.get("browser_writes_intermediate_csv")
        ),
        "browser_receives_zarr_write_authority": bool(
            direct_start_policy.get("browser_receives_zarr_write_authority")
        ),
        "browser_has_direct_zarr_write_authority": bool(
            direct_start_policy.get("browser_has_direct_zarr_write_authority")
        ),
    }

def _task_open_response_metadata(
    *,
    user: str,
    expected_user: str | None,
    task: Mapping[str, object] | None,
    session_id: str,
    operator_validation_start_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    authorization_context = _labeler_authorization_context(
        user=resolved_user,
        expected_user=expected or None,
        task=task,
        session_id=session_id,
    )
    metadata: dict[str, object] = {
        "task_open_authorization_contract": _task_open_authorization_contract(
            user=resolved_user,
            expected_user=expected or None,
            task=task,
            ready=True,
            session_created_server_side=True,
            server_authorizes_open=True,
            operator_validation_start_gate=operator_validation_start_gate,
        ),
        "authorization_context": authorization_context,
    }
    return _add_task_open_personalized_launch_metadata(
        metadata,
        authorization_context=authorization_context,
    )

def _task_open_failure_metadata(
    *,
    user: str,
    expected_user: str | None,
    task: Mapping[str, object] | None,
    error: str,
    operator_validation_start_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    authorization_context = _labeler_authorization_context(
        user=resolved_user,
        expected_user=expected or None,
        task=task,
    )
    metadata: dict[str, object] = {
        "resolved_user": resolved_user,
        "expected_user": expected,
        "task_open_authorization_contract": _task_open_authorization_contract(
            user=resolved_user,
            expected_user=expected or None,
            task=task,
            ready=False,
            not_ready_reason=error,
            session_created_server_side=False,
            server_authorizes_open=False,
            operator_validation_start_gate=operator_validation_start_gate,
        ),
        "authorization_context": authorization_context,
    }
    return _add_task_open_personalized_launch_metadata(
        metadata,
        authorization_context=authorization_context,
    )

def _add_task_open_personalized_launch_metadata(
    payload: dict[str, object],
    *,
    authorization_context: Mapping[str, object],
) -> dict[str, object]:
    direct_start_policy = _dataset_queue_direct_start_policy()
    personal_queue_url = str(
        authorization_context.get("return_personal_dataset_queue_url") or ""
    )
    payload.update(
        {
            "expected_user_personal_dataset_queue_url": personal_queue_url,
            "personalized_labeler_entry_url": personal_queue_url,
            "preferred_labeler_entry_url": personal_queue_url,
            "personal_dataset_queue_link_role": "preferred_queue",
            "dataset_queue_link_role": "canonical_queue_fallback",
            "dataset_queue_direct_start_policy": direct_start_policy,
            "browser_label_write_target": str(
                direct_start_policy.get("browser_label_write_target") or ""
            ),
            "browser_writes_csv_or_handoff_files": bool(
                direct_start_policy.get("browser_writes_csv_or_handoff_files")
            ),
            "browser_has_direct_zarr_write_authority": bool(
                direct_start_policy.get("browser_has_direct_zarr_write_authority")
            ),
            "csv_handoff_artifact_role": str(
                direct_start_policy.get("csv_handoff_artifact_role") or ""
            ),
        }
    )
    _add_payload_contract_compact_fields(payload)
    return payload

def _task_completion_authorization_contract(
    *,
    user: str,
    expected_user: str | None,
    task: Mapping[str, object] | None,
    session: Mapping[str, object] | None,
    requested_task_id: str,
    ready: bool,
    not_ready_reason: str | None = None,
    server_authorizes_completion: bool = False,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    task_id = str((task or {}).get("task_id") or requested_task_id or "")
    session_task_id = str((session or {}).get("task_id") or "")
    task_assigned_to_resolved_user = bool(
        task is not None and str(task.get("assignee_user") or "") == resolved_user
    )
    assignment_status_active = bool(
        task is not None and str(task.get("assignment_status") or "") == "active"
    )
    return {
        "schema": "palette.web_labeling_task_completion_authorization_contract.v1",
        "ready": bool(ready),
        "not_ready_reason": str(not_ready_reason or ""),
        "expected_user_guard_checked_server_side": True,
        "expected_user_guard_present": bool(expected),
        "expected_user_matches_resolved_user": not expected or expected == resolved_user,
        "task_reloaded_server_side": task is not None,
        "active_assignment_required": True,
        "active_assignment_present": bool(
            task_assigned_to_resolved_user and assignment_status_active
        ),
        "task_assigned_to_resolved_user": task_assigned_to_resolved_user,
        "assignment_status_active": assignment_status_active,
        "current_session_required": True,
        "current_session_present": session is not None,
        "session_owned_by_resolved_user": bool(
            session is not None and str(session.get("user") or "") == resolved_user
        ),
        "session_task_matches_requested_task": bool(
            session is not None and session_task_id == task_id
        ),
        "task_completion_state_mutation_target": "labeling_task_store",
        "completion_closes_open_sessions_server_side": True,
        "client_authorizes_completion": False,
        "server_authorizes_completion": bool(server_authorizes_completion),
        "browser_writes_csv_or_handoff_files": False,
        "browser_writes_handoff_csv": False,
        "browser_writes_intermediate_csv": False,
        "browser_has_direct_zarr_write_authority": False,
    }

def _task_completion_failure_metadata(
    *,
    user: str,
    expected_user: str | None,
    task: Mapping[str, object] | None,
    session: Mapping[str, object] | None,
    requested_task_id: str,
    error: str,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    authorization_context = _labeler_authorization_context(
        user=resolved_user,
        expected_user=expected or None,
        task=task,
        session=session,
    )
    metadata: dict[str, object] = {
        "task_completion_authorization_contract": (
            _task_completion_authorization_contract(
                user=resolved_user,
                expected_user=expected or None,
                task=task,
                session=session,
                requested_task_id=requested_task_id,
                ready=False,
                not_ready_reason=error,
                server_authorizes_completion=False,
            )
        ),
        "authorization_context": authorization_context,
    }
    return _add_task_open_personalized_launch_metadata(
        metadata,
        authorization_context=authorization_context,
    )

def _labeler_authorization_context(
    *,
    user: str | None = None,
    expected_user: str | None = None,
    task: Mapping[str, object] | None = None,
    session: Mapping[str, object] | None = None,
    session_id: str | None = None,
    current_session: Mapping[str, object] | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "schema": "palette.web_labeling_authorization_context.v1",
        "resolved_user": str(user or ""),
    }
    if expected_user:
        context["expected_user"] = str(expected_user)
    effective_session_id = str(session_id or (session or {}).get("session_id") or "").strip()
    if effective_session_id:
        context["session_id"] = effective_session_id
    if session is not None:
        context.update(
            {
                "session_user": str(session.get("user") or ""),
                "session_task_id": str(session.get("task_id") or ""),
                "session_recording_id": str(session.get("recording_id") or ""),
                "session_closed": bool(session.get("closed_at_utc")),
                "session_closed_at_utc": str(session.get("closed_at_utc") or ""),
                "session_expires_at_utc": str(session.get("expires_at_utc") or ""),
            }
        )
    if task is not None:
        context.update(
            {
                "task_id": str(task.get("task_id") or ""),
                "recording_id": str(task.get("recording_id") or ""),
                "assignee_user": str(task.get("assignee_user") or ""),
                "assignment_status": str(task.get("assignment_status") or ""),
                "task_state": str(task.get("state") or ""),
                "workflow_kind": str(task.get("workflow_kind") or ""),
            }
        )
    if current_session is not None:
        context["current_session_id"] = str(current_session.get("session_id") or "")
    return_expected_user = str(
        context.get("expected_user")
        or context.get("assignee_user")
        or context.get("session_user")
        or context.get("resolved_user")
        or ""
    ).strip()
    if return_expected_user:
        context["return_expected_user"] = return_expected_user
        context["return_labeling_home_url"] = _dashboard_url_for_expected_user(
            LABELING_HOME_PATH,
            return_expected_user,
        )
        context["return_labeling_home_expected_user_guarded"] = True
        context["return_personal_dataset_queue_url"] = _dashboard_url_for_expected_user(
            PERSONAL_DATASET_QUEUE_PATH,
            return_expected_user,
        )
        context["return_personal_dataset_queue_expected_user_guarded"] = True
        context["return_personal_work_url"] = _dashboard_url_for_expected_user(
            PERSONAL_WORK_PATH,
            return_expected_user,
        )
        context["return_personal_work_expected_user_guarded"] = True
    return context

def _labeler_read_authorization_denial_metadata(
    *,
    user: str,
    expected_user: str,
    route_path: str,
    response_kind: str,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    personal_dataset_queue_url = (
        _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, expected)
        if expected
        else ""
    )
    browser_mutation_write_policy = _browser_mutation_write_policy()
    browser_mutation_write_checklist = _browser_mutation_write_runtime_checklist(
        browser_mutation_write_policy
    )
    dataset_queue_direct_start_policy = _dataset_queue_direct_start_policy()
    payload = {
        "authorization_context": _labeler_authorization_context(
            user=resolved_user,
            expected_user=expected,
        ),
        "expected_user": expected,
        "expected_user_personal_dataset_queue_url": personal_dataset_queue_url,
        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
        "preferred_labeler_entry_url": personal_dataset_queue_url,
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": personal_dataset_queue_url,
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
            personal_dataset_queue_url
        ),
        "browser_mutation_write_policy": browser_mutation_write_policy,
        "browser_mutation_write_checklist": browser_mutation_write_checklist,
        "dataset_queue_direct_start_policy": dataset_queue_direct_start_policy,
        "labeler_read_authorization_contract": {
            "schema": "palette.web_labeling_labeler_read_authorization_contract.v1",
            "ready": False,
            "not_ready_reason": "dashboard_user_mismatch",
            "route_path": str(route_path or ""),
            "response_kind": str(response_kind or ""),
            "resolved_user": resolved_user,
            "expected_user": expected,
            "expected_user_guard_checked_server_side": True,
            "expected_user_guard_present": bool(expected),
            "expected_user_matches_resolved_user": bool(
                expected and expected == resolved_user
            ),
            "known_assignment_store_user_required": True,
            "personal_work_reads_filtered_by_resolved_user": True,
            "dataset_queue_reads_filtered_by_resolved_user": True,
            "labeler_visible_scope": "assigned_recordings_for_resolved_user",
            "returns_assigned_work_payload": False,
            "returns_dataset_queue_payload": False,
            "server_authorizes_read": False,
            "server_authorizes_task_open": False,
            "server_authorizes_mutation": False,
            "browser_writes_csv_or_handoff_files": False,
            "browser_writes_handoff_csv": False,
            "browser_writes_intermediate_csv": False,
            "browser_has_direct_zarr_write_authority": False,
        },
    }
    _add_payload_contract_compact_fields(payload)
    return payload

def _browser_mutation_response_metadata(
    *,
    workflow_kind: str,
    session: Mapping[str, object],
    mutation_event: Mapping[str, object],
    promotion_event: Mapping[str, object] | None = None,
    operator_validation_mutation_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    policy = _browser_mutation_write_policy()
    contract = _browser_mutation_write_contract_policy(policy)
    mutation_gate = (
        dict(operator_validation_mutation_gate)
        if isinstance(operator_validation_mutation_gate, Mapping)
        else _runtime_operator_validation_mutation_gate_not_required()
    )
    audit_events = []
    for event in (mutation_event, promotion_event):
        if not isinstance(event, Mapping):
            continue
        audit_events.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "event_type": str(event.get("event_type") or ""),
                "task_id": str(event.get("task_id") or session.get("task_id") or ""),
                "recording_id": str(event.get("recording_id") or session.get("recording_id") or ""),
                "created_at_utc": str(event.get("created_at_utc") or ""),
            }
        )
    authorization_context = _labeler_authorization_context(
        user=str(session.get("user") or ""),
        session=session,
    )
    return {
        "schema": "palette.web_labeling_browser_mutation_response.v1",
        "workflow_kind": workflow_kind,
        "task_id": str(mutation_event.get("task_id") or session.get("task_id") or ""),
        "recording_id": str(mutation_event.get("recording_id") or session.get("recording_id") or ""),
        "assignment_authorization_checked_server_side": True,
        "assignment_authorization_result": "passed",
        "active_assignment_checked_server_side": True,
        "active_assignment_required": True,
        "active_assignment_present": True,
        "task_assigned_to_resolved_user_checked_server_side": True,
        "task_assigned_to_resolved_user": True,
        "task_state_checked_server_side": True,
        "session_checked_server_side": True,
        "session_ownership_checked_server_side": True,
        "session_user_matches_resolved_user": True,
        "current_session_checked_server_side": True,
        "current_session_required": True,
        "reassignment_session_safety_checked_server_side": True,
        "reassignment_session_safety_passed": True,
        "current_target_token_checked_server_side": True,
        "target_token_required_for_mutation": True,
        "server_authorizes_mutation": True,
        "mutation_authorization_contract": {
            "schema": "palette.web_labeling_mutation_authorization_contract.v1",
            "ready": True,
            "session_lookup": "required",
            "session_lookup_result": "passed",
            "session_owned_by_resolved_user": True,
            "task_reloaded_server_side": True,
            "task_assigned_to_resolved_user": True,
            "assignment_status_active": True,
            "task_open_for_mutation": True,
            "current_session_required": True,
            "current_session_result": "passed",
            "reassignment_session_safety_required": True,
            "reassignment_session_safety_result": "passed",
            "current_target_token_required": True,
            "current_target_token_result": "passed",
            "browser_supplied_zarr_or_csv_target_allowed": False,
            "browser_supplied_target_selectors_allowed": False,
            "client_authorizes_mutation": False,
            "operator_validation_mutation_gate": mutation_gate,
            "operator_validation_mutation_gate_checked_server_side": True,
            "operator_validation_mutation_gate_required": bool(
                mutation_gate.get("required_for_browser_mutation")
            ),
            "operator_validation_mutation_gate_ready": bool(mutation_gate.get("ready")),
            "operator_validation_mutation_gate_blocks_browser_mutation": bool(
                mutation_gate.get("blocks_browser_mutation")
            ),
            "operator_validation_mutation_gate_not_ready_reason": str(
                mutation_gate.get("not_ready_reason") or ""
            ),
            "server_authorizes_mutation": True,
        },
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
        "data_plane_write_target": str(contract.get("data_plane_write_target") or ""),
        "mutable_label_data_plane": str(contract.get("mutable_label_data_plane") or ""),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(policy),
        "label_mutation_target_kind": str(contract.get("label_mutation_target_kind") or ""),
        "browser_label_write_target": str(contract.get("browser_label_write_target") or ""),
        "server_mutates_task_scoped_zarr_targets": bool(
            contract.get("server_mutates_task_scoped_zarr_targets")
        ),
        "training_zarr_mutations_are_server_owned": bool(
            contract.get("training_zarr_mutations_are_server_owned")
        ),
        "promotion_training_zarr_requires_task_scope": bool(
            policy.get("promotion_training_zarr_mutation_requires_task_scope")
        ),
        "handoff_artifacts_are_metadata_only": bool(
            contract.get("handoff_artifacts_are_metadata_only")
        ),
        "csv_handoff_artifact_role": str(contract.get("csv_handoff_artifact_role") or ""),
        "csv_handoff_artifacts_are_label_write_targets": bool(
            contract.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "handoff_csv_artifacts_are_label_write_targets": bool(
            contract.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "intermediate_csv_artifacts_are_label_write_targets": bool(
            contract.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "browser_writes_csv_or_handoff_files": bool(
            contract.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_writes_handoff_csv": bool(contract.get("browser_writes_handoff_csv")),
        "browser_writes_intermediate_csv": bool(
            contract.get("browser_writes_intermediate_csv")
        ),
        "browser_receives_zarr_write_authority": bool(
            contract.get("browser_receives_zarr_write_authority")
        ),
        "browser_has_direct_zarr_write_authority": bool(
            contract.get("browser_has_direct_zarr_write_authority")
        ),
        "operator_validation_mutation_gate": mutation_gate,
        "operator_validation_mutation_gate_checked_server_side": True,
        "operator_validation_mutation_gate_required": bool(
            mutation_gate.get("required_for_browser_mutation")
        ),
        "operator_validation_mutation_gate_ready": bool(mutation_gate.get("ready")),
        "operator_validation_mutation_gate_blocks_browser_mutation": bool(
            mutation_gate.get("blocks_browser_mutation")
        ),
        "operator_validation_mutation_gate_not_ready_reason": str(
            mutation_gate.get("not_ready_reason") or ""
        ),
        "audit_event_store": str(policy.get("audit_event_store") or "labeling_task_events"),
        "audit_event_id": str(mutation_event.get("event_id") or ""),
        "audit_event_type": str(mutation_event.get("event_type") or ""),
        "audit_events": audit_events,
    }

def _browser_mutation_failure_metadata(
    *,
    session: Mapping[str, object],
    error: str,
    operator_validation_mutation_gate: Mapping[str, object] | None = None,
    session_lookup_result: str = "passed",
    session_owned_by_resolved_user: bool = True,
    task_reloaded_server_side: bool = True,
    task_assigned_to_resolved_user: bool = True,
    assignment_status_active: bool = True,
    task_open_for_mutation: bool = True,
    current_session_result: str = "passed",
    reassignment_session_safety_result: str = "passed",
    current_target_token_result: str = "not_checked",
    browser_supplied_target_selectors_result: str = "not_checked",
) -> dict[str, object]:
    policy = _browser_mutation_write_policy()
    contract = _browser_mutation_write_contract_policy(policy)
    mutation_gate = (
        dict(operator_validation_mutation_gate)
        if isinstance(operator_validation_mutation_gate, Mapping)
        else _runtime_operator_validation_mutation_gate_not_required()
    )
    authorization_context = _labeler_authorization_context(
        user=str(session.get("user") or ""),
        session=session,
    )
    return {
        "mutation_authorization_contract": {
            "schema": "palette.web_labeling_mutation_authorization_contract.v1",
            "ready": False,
            "not_ready_reason": str(error or ""),
            "session_lookup": "required",
            "session_lookup_result": str(session_lookup_result or "passed"),
            "session_owned_by_resolved_user": bool(session_owned_by_resolved_user),
            "task_reloaded_server_side": bool(task_reloaded_server_side),
            "task_assigned_to_resolved_user": bool(task_assigned_to_resolved_user),
            "assignment_status_active": bool(assignment_status_active),
            "task_open_for_mutation": bool(task_open_for_mutation),
            "current_session_required": True,
            "current_session_result": str(current_session_result or "passed"),
            "reassignment_session_safety_required": True,
            "reassignment_session_safety_result": str(
                reassignment_session_safety_result or "passed"
            ),
            "current_target_token_required": True,
            "current_target_token_result": str(current_target_token_result or "not_checked"),
            "browser_supplied_zarr_or_csv_target_allowed": False,
            "browser_supplied_target_selectors_allowed": False,
            "browser_supplied_target_selectors_result": str(
                browser_supplied_target_selectors_result or "not_checked"
            ),
            "client_authorizes_mutation": False,
            "operator_validation_mutation_gate": mutation_gate,
            "operator_validation_mutation_gate_checked_server_side": True,
            "operator_validation_mutation_gate_required": bool(
                mutation_gate.get("required_for_browser_mutation")
            ),
            "operator_validation_mutation_gate_ready": bool(mutation_gate.get("ready")),
            "operator_validation_mutation_gate_blocks_browser_mutation": bool(
                mutation_gate.get("blocks_browser_mutation")
            ),
            "operator_validation_mutation_gate_not_ready_reason": str(
                mutation_gate.get("not_ready_reason") or ""
            ),
            "server_authorizes_mutation": False,
        },
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
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(policy),
        "data_plane_write_target": str(contract.get("data_plane_write_target") or ""),
        "label_mutation_target_kind": str(contract.get("label_mutation_target_kind") or ""),
        "mutable_label_data_plane": str(contract.get("mutable_label_data_plane") or ""),
        "browser_label_write_target": str(contract.get("browser_label_write_target") or ""),
        "server_mutates_task_scoped_zarr_targets": bool(
            contract.get("server_mutates_task_scoped_zarr_targets")
        ),
        "training_zarr_mutations_are_server_owned": bool(
            contract.get("training_zarr_mutations_are_server_owned")
        ),
        "promotion_training_zarr_requires_task_scope": bool(
            contract.get("promotion_training_zarr_mutation_requires_task_scope")
            or contract.get("promotion_training_zarr_requires_task_scope")
            or policy.get("promotion_training_zarr_mutation_requires_task_scope")
        ),
        "handoff_artifacts_are_metadata_only": bool(
            contract.get("handoff_artifacts_are_metadata_only")
        ),
        "csv_handoff_artifact_role": str(contract.get("csv_handoff_artifact_role") or ""),
        "csv_handoff_artifacts_are_label_write_targets": bool(
            contract.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "handoff_csv_artifacts_are_label_write_targets": bool(
            contract.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "intermediate_csv_artifacts_are_label_write_targets": bool(
            contract.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "browser_writes_csv_or_handoff_files": bool(
            contract.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_writes_handoff_csv": bool(contract.get("browser_writes_handoff_csv")),
        "browser_writes_intermediate_csv": bool(
            contract.get("browser_writes_intermediate_csv")
        ),
        "browser_receives_zarr_write_authority": bool(
            contract.get("browser_receives_zarr_write_authority")
        ),
        "browser_has_direct_zarr_write_authority": bool(
            contract.get("browser_has_direct_zarr_write_authority")
        ),
        "operator_validation_mutation_gate": mutation_gate,
        "operator_validation_mutation_gate_checked_server_side": True,
        "operator_validation_mutation_gate_required": bool(
            mutation_gate.get("required_for_browser_mutation")
        ),
        "operator_validation_mutation_gate_ready": bool(mutation_gate.get("ready")),
        "operator_validation_mutation_gate_blocks_browser_mutation": bool(
            mutation_gate.get("blocks_browser_mutation")
        ),
        "operator_validation_mutation_gate_not_ready_reason": str(
            mutation_gate.get("not_ready_reason") or ""
        ),
    }
