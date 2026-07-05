"""Admin and dashboard read-only payload builders for web labeling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

from .admin_registry import (
    REGISTRY_PATH_ENV_VAR,
    _admin_compact_task,
    _admin_registry_lookup,
    _admin_registry_path_from_env,
    _admin_registry_summary,
    _admin_registry_warning,
    _admin_registry_warnings_for_recording,
)
from .assignment_store import (
    LABELER_START_TASK_STATES,
    LABELING_USER_ROLES,
    LABELING_USER_STATUSES,
)
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
    _dataset_queue_url_for_dashboard,
    _is_admin_user,
    _personal_dataset_queue_url_for_dashboard,
    _utc_timestamp,
)
from .web_policy import (
    _browser_mutation_write_contract_policy,
    _browser_mutation_write_policy,
    _browser_mutation_write_runtime_checklist,
    _browser_response_security_policy,
    _browser_signed_link_policy,
    _browser_task_state_policy,
    _browser_workflow_capabilities,
    _operator_validation_visibility_fields,
    _operator_validation_visibility_policy,
    _session_guard_policy,
)
from .web_responses import _is_loopback_host
from .work_queue import (
    _add_direct_start_contracts_to_work_tasks,
    _add_work_summary_fields,
    _completion_percent,
    _completion_state,
    _dataset_queue_state,
    _first_dataset_queue_url,
    _labeler_work_completion_fields,
    _public_reassignment_session_safety_fields,
    _reassignment_session_safety_flat_fields,
    _reassignment_session_safety_operator_action,
    _reassignment_session_safety_recording_ids,
    _work_dataset_queue_summary,
)


DATASET_QUEUE_PATH = "/datasets"

DASHBOARD_PATH = "/work"

PERSONAL_DATASET_QUEUE_PATH = "/my-datasets"

LABELING_HOME_PATH = "/labeling"

PERSONAL_WORK_PATH = "/my-work"

IDENTITY_PROBE_PATH = "/identity"

BROWSER_MUTATION_AUDIT_PROVENANCE: dict[str, object] = {
    "event_store": "labeling_task_events",
    "required_event_fields": [
        "event_id",
        "task_id",
        "recording_id",
        "user",
        "event_type",
        "created_at_utc",
        "target",
        "before",
        "after",
    ],
    "timestamp_field": "created_at_utc",
    "identity_fields": ["task_id", "recording_id", "user"],
    "mutation_summary_fields": ["target", "before", "after"],
}

BROWSER_MUTATION_RETRY_POLICY: dict[str, object] = {
    "data_write_semantics": "replace_target_payload",
    "same_payload_retry_safe": True,
    "audit_semantics": "append_only",
    "duplicate_audit_events_possible": True,
    "client_idempotency_key_supported": False,
    "retry_guidance": "If the browser loses the response after submitting, reopening the task and saving the same target payload again should leave the label data in the same state, but records another audit event.",
}

ASSIGNMENT_OWNERSHIP_POLICY: dict[str, object] = {
    "assignment_scope": "recording",
    "recording_assignment_key": "recording_id",
    "recording_id_primary_key": True,
    "schema_enforced_recording_primary_key": True,
    "one_current_assignment_row_per_recording": True,
    "one_active_owner": True,
    "multiple_labelers_per_recording_allowed": False,
    "reassignment_replaces_owner": True,
    "stale_sessions_closed_on_reassignment": True,
    "stale_sessions_closed_before_assignment_update": True,
    "reassignment_target_validated_before_session_closure": True,
    "session_closure_and_assignment_update_atomic": True,
    "raw_assignment_change_blocks_open_sessions": True,
    "assignment_manifests_are_control_plane": True,
    "duplicate_manifest_rows_do_not_create_multiple_owners": True,
    "assignment_user_match_required_for_mutation": True,
    "browser_mutation_requires_current_assignment_owner": True,
    "browser_mutation_target_resolved_server_side": True,
    "browser_mutation_target_source": "recording_assignments.active_assignment",
    "labelers_mutate_assigned_training_zarrs": True,
    "labelers_mutate_intermediate_csvs": False,
    "operator_reassignment_helper": "assign_recording_with_session_closure",
}

DEFAULT_OPERATOR_VALIDATION_GATE_IDS = (
    "mutable_zarr_backup_confirmation",
    "browser_response_security_headers",
    "identity_probe_verification",
    "browser_smoke",
    "disposable_zarr_mutation_smoke",
    "operator_recovery_contract",
)

OPERATOR_VALIDATION_GATE_STATUS_VALUES = (
    "unknown",
    "pending",
    "missing_evidence",
    "needs_review",
    "passed",
)

OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES = (
    "status",
    "pending",
    "missing_evidence",
    "needs_review",
    "passed",
)

_DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS = "row_readiness_not_safe_share_approval"

_DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA = "palette.web_labeling_ready_row_draft_bundle.v1"

_DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND = "ready_row_draft_text"

_DASHBOARD_READY_ROW_STATE_VALUES = ("ready_row_draft", "diagnostic_note")

_DASHBOARD_COPY_INTENT_VALUES = _DASHBOARD_READY_ROW_STATE_VALUES

_DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES = {
    "browser_mutation_write_ready": True,
    "browser_mutation_label_mutation_target_kind": "task_scoped_training_zarr",
    "browser_mutation_browser_label_write_target": "training_zarr",
    "browser_mutation_csv_handoff_artifact_role": "metadata_only_control_plane",
    "browser_mutation_csv_handoff_artifacts_are_label_write_targets": False,
    "browser_mutation_handoff_csv_artifacts_are_label_write_targets": False,
    "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": False,
    "browser_mutation_browser_writes_csv_or_handoff_files": False,
    "browser_mutation_browser_writes_handoff_csv": False,
    "browser_mutation_browser_writes_intermediate_csv": False,
    "browser_mutation_browser_has_direct_zarr_write_authority": False,
}

_DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS = tuple(
    _DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES.keys()
)

_DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES = {
    "dataset_queue_direct_start_policy_present": True,
    "dataset_queue_direct_start_enabled": True,
    "dataset_queue_direct_start_method": "POST",
    "dataset_queue_direct_start_same_origin_only": True,
    "dataset_queue_direct_start_exact_route_required": True,
    "dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id": True,
    "dataset_queue_direct_start_expected_user_guard_required": True,
    "dataset_queue_direct_start_post_body_expected_user_required": True,
    "dataset_queue_direct_start_post_body_expected_user_field": "expected_user",
    "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract": True,
    "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract": True,
    "dataset_queue_direct_start_denied_start_support_includes_authorization_context": True,
    "dataset_queue_direct_start_denied_start_contract_reports_no_session_created": True,
    "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false": True,
    "dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint": True,
    "dataset_queue_direct_start_label_mutation_target_kind": "task_scoped_training_zarr",
    "dataset_queue_direct_start_browser_label_write_target": "training_zarr",
    "dataset_queue_direct_start_csv_handoff_artifact_role": "metadata_only_control_plane",
    "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets": False,
    "dataset_queue_direct_start_browser_writes_csv_or_handoff_files": False,
    "dataset_queue_direct_start_browser_writes_handoff_csv": False,
    "dataset_queue_direct_start_browser_writes_intermediate_csv": False,
    "dataset_queue_direct_start_browser_receives_zarr_write_authority": False,
    "dataset_queue_direct_start_browser_has_direct_zarr_write_authority": False,
}

_DASHBOARD_DIRECT_BROWSER_START_FIELDS = tuple(
    _DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES.keys()
)

_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD = "labeler_links_safe_to_share"

_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE = True

_DASHBOARD_READY_ROW_DRAFT_SHARE_RULE = (
    "Do not share copied ready-row draft text until inspect-handoff "
    f"--require-shareable reports {_DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD}=true."
)

_VALIDATION_GATE_BLOCKS_INVITATION_LEGACY_SEMANTICS = (
    "blocks_ready_row_draft_or_launch_readiness_not_safe_share_approval"
)

_VALIDATION_GATE_BLOCKS_INVITATION_SAFE_SHARE_FIELD = "labeler_links_safe_to_share"

OPERATOR_EVIDENCE_VALIDATION_GATE_IDS = frozenset(
    {
        "identity_probe_verification",
        "operator_authorization_boundary",
        "browser_response_security_headers",
        "dashboard_visibility",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "multi_user_dry_run",
        "final_signoff",
    }
)

OPERATOR_EVIDENCE_TEMPLATE_FIELDS: dict[str, str] = {
    "identity_probe_verification": "identity_source_evidence_template",
    "browser_response_security_headers": "browser_response_security_evidence_template",
    "browser_smoke": "browser_smoke_evidence_template",
    "disposable_zarr_mutation_smoke": "disposable_zarr_mutation_smoke_evidence_template",
    "mutable_zarr_backup_confirmation": "zarr_backup_evidence_template",
}

def _session_is_expired(session: Mapping[str, object]) -> bool:
    raw = str(session.get("expires_at_utc") or "").strip()
    if not raw:
        return False
    try:
        expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= datetime.now(timezone.utc)

def _known_labeler_status(store: LabelingStore, user: str | None) -> dict[str, object]:
    normalized_user = str(user or "").strip()
    assignments = store.list_assignments(assignee_user=normalized_user, status=None) if normalized_user else []
    recording_ids: list[str] = []
    for assignment in assignments:
        recording_id = str(assignment.get("recording_id") or "").strip()
        if recording_id:
            recording_ids.append(recording_id)
    status = store.labeling_user_status(normalized_user)
    return {
        **status,
        "schema": "palette.web_labeling_known_labeler_status.v1",
        "user": normalized_user,
        "is_known_labeler": bool(status.get("is_known_labeler")),
        "is_active_labeling_user": bool(status.get("is_active_labeling_user")),
        "registry_row_present": bool(status.get("registry_row_present")),
        "assignment_count": int(status.get("assignment_count") or 0),
        "active_assignment_count": int(status.get("active_assignment_count") or 0),
        "has_active_assignment": bool(status.get("has_active_assignment")),
        "recording_ids": sorted(recording_ids),
    }

def _unresolved_failed_promotions(
    store: LabelingStore,
    *,
    assignee_user: str | None = None,
    limit: int = 200,
) -> list[dict[str, object]]:
    failed = store.list_events(
        event_type="promotion_failed",
        assignee_user=assignee_user,
        limit=limit,
    )
    successes = store.list_events(
        event_type="promotion_success",
        assignee_user=assignee_user,
        limit=max(limit * 5, 500),
    )
    resolved: set[str] = set()
    for event in successes:
        target = event.get("target")
        if isinstance(target, Mapping):
            retry_of = str(target.get("retry_of_event_id") or "").strip()
            if retry_of:
                resolved.add(retry_of)
    return [event for event in failed if str(event.get("event_id") or "") not in resolved]

def _admin_summary_payload(store: LabelingStore, *, config: ServerConfig) -> dict[str, object]:
    assignments = store.list_assignments(status=None)
    tasks = store.list_tasks(include_completed=True)
    active_sessions = store.list_sessions(include_closed=False, limit=500)
    stale_sessions = store.list_sessions(include_closed=False, expired_only=True, limit=500)
    failed_promotions = _unresolved_failed_promotions(store, limit=500)
    recent_audit_events = store.list_events(limit=500)
    assignment_operator_rows = _assignment_operator_status_rows(assignments, tasks)
    assignment_ownership_integrity = _assignment_ownership_integrity(
        assignments,
        schema_integrity=store.assignment_schema_integrity(),
    )
    dashboard_users = _dashboard_roster_rows(
        store,
        dashboard_url=DASHBOARD_PATH,
        include_inactive=False,
        include_completed=False,
        require_dashboard_url=False,
    )
    dashboard_invite_reasons = _dashboard_invite_reason_counts(dashboard_users)
    dashboard_identity_probe_counts = _dashboard_identity_probe_counts(dashboard_users)
    dashboard_dataset_queue_counts = _dashboard_dataset_queue_counts(dashboard_users)
    dataset_queue_start_readiness = _dataset_queue_start_readiness_from_counts(dashboard_dataset_queue_counts)
    dashboard_total_tasks = sum(int(row.get("total_tasks") or 0) for row in dashboard_users)
    dashboard_complete_tasks = sum(int(row.get("complete_tasks") or 0) for row in dashboard_users)
    raw_operator_validation_fields = _dashboard_operator_validation_fields_for_config(config)
    operator_validation_fields = _operator_validation_public_fields(raw_operator_validation_fields)
    operator_validation_report = {
        **operator_validation_fields,
        **_operator_validation_gate_metadata_fields(),
        **_operator_validation_gate_flat_fields(operator_validation_fields),
    }
    operator_validation_command_templates = _operator_validation_command_templates(
        operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids")
        if isinstance(
            operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids"),
            list,
        )
        else None
    )
    safe_share_gate = _safe_share_gate_policy()
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        raw_operator_validation_fields,
        safe_share_gate=safe_share_gate,
    )
    operator_validation_report.update(safe_share_checklist_fields)
    task_state_counts: dict[str, int] = {}
    workflow_counts: dict[str, int] = {}
    assignment_user_counts: dict[str, int] = {}
    for task in tasks:
        state_value = str(task.get("state") or "unknown")
        workflow = str(task.get("workflow_kind") or "unknown")
        task_state_counts[state_value] = task_state_counts.get(state_value, 0) + 1
        workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
    for assignment in assignments:
        key = f"{assignment.get('assignee_user')}:{assignment.get('status')}"
        assignment_user_counts[key] = assignment_user_counts.get(key, 0) + 1
    return {
        "assignment_count": len(assignments),
        "task_count": len(tasks),
        "active_session_count": len(active_sessions),
        "stale_session_count": len(stale_sessions),
        "failed_promotion_count": len(failed_promotions),
        "active_session_user_counts": _count_rows_by_field(active_sessions, "user"),
        "active_session_workflow_counts": _count_rows_by_field(active_sessions, "workflow_kind"),
        "stale_session_user_counts": _count_rows_by_field(stale_sessions, "user"),
        "stale_session_workflow_counts": _count_rows_by_field(stale_sessions, "workflow_kind"),
        "assignment_work_state_counts": _count_rows_by_field(assignment_operator_rows, "work_state"),
        "recent_audit_event_count": len(recent_audit_events),
        "recent_audit_event_user_counts": _count_rows_by_field(recent_audit_events, "user"),
        "recent_audit_event_type_counts": _count_rows_by_field(recent_audit_events, "event_type"),
        "recent_audit_event_workflow_counts": _count_rows_by_field(recent_audit_events, "workflow_kind"),
        "assignment_user_counts": assignment_user_counts,
        "task_state_counts": task_state_counts,
        "workflow_counts": workflow_counts,
        "labeler_landing_page_path": "/",
        "dashboard_path": DASHBOARD_PATH,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "personal_work_alias_for": DASHBOARD_PATH,
        "personal_dataset_queue_alias_for": DATASET_QUEUE_PATH,
        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
        "preferred_labeler_entry_path": PERSONAL_DATASET_QUEUE_PATH,
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_path": PERSONAL_DATASET_QUEUE_PATH,
        "single_owner_policy": _assignment_ownership_policy(),
        "assignment_ownership_integrity": assignment_ownership_integrity,
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "identity_source_policy": _identity_source_policy(config),
        "operator_authorization_policy": _operator_authorization_policy(config, include_admin_details=True),
        "operator_recovery_policy": _operator_recovery_policy(),
        "operator_recovery_contract": _operator_recovery_contract_policy(_operator_recovery_policy()),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
        "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
        "safe_share_gate": safe_share_gate,
        **_safe_share_gate_flat_fields(safe_share_gate),
        **safe_share_checklist_fields,
        "browser_response_security_policy": _browser_response_security_policy(),
        "operator_validation": operator_validation_report,
        "operator_validation_command_templates": operator_validation_command_templates,
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        "dataset_queue_start_readiness": dataset_queue_start_readiness,
        "reassignment_session_safety_blocked_users": (
            dashboard_dataset_queue_counts.get("reassignment_session_safety_blocked_users")
            if isinstance(
                dashboard_dataset_queue_counts.get("reassignment_session_safety_blocked_users"),
                list,
            )
            else []
        ),
        "reassignment_session_safety_blocked_user_count": int(
            dashboard_dataset_queue_counts.get("reassignment_session_safety_blocked_user_count") or 0
        ),
        "reassignment_session_safety_mismatch_count": int(
            dashboard_dataset_queue_counts.get("reassignment_session_safety_mismatch_count") or 0
        ),
        "reassignment_session_safety_blocked_recording_ids": (
            dashboard_dataset_queue_counts.get("reassignment_session_safety_blocked_recording_ids")
            if isinstance(
                dashboard_dataset_queue_counts.get("reassignment_session_safety_blocked_recording_ids"),
                list,
            )
            else []
        ),
        "dashboard_invite_actions": _dashboard_invite_actions(dashboard_invite_reasons),
        "dashboard_user_counts": {
            "users": len(dashboard_users),
            "ready_to_invite": sum(1 for row in dashboard_users if bool(row.get("ready_to_invite"))),
            "not_ready_to_invite": sum(1 for row in dashboard_users if not bool(row.get("ready_to_invite"))),
            "ready_to_invite_users": [
                str(row.get("user") or "")
                for row in dashboard_users
                if bool(row.get("ready_to_invite"))
            ],
            "not_ready_to_invite_users": [
                str(row.get("user") or "")
                for row in dashboard_users
                if not bool(row.get("ready_to_invite"))
            ],
            "ready_to_invite_legacy_semantics": _DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS,
            "ready_row_state_values": list(_DASHBOARD_READY_ROW_STATE_VALUES),
            "copy_intent_values": list(_DASHBOARD_COPY_INTENT_VALUES),
            "ready_row_draft_count": sum(
                1 for row in dashboard_users if str(row.get("copy_intent") or "") == "ready_row_draft"
            ),
            "diagnostic_note_count": sum(
                1 for row in dashboard_users if str(row.get("copy_intent") or "") == "diagnostic_note"
            ),
            "ready_row_draft_users": [
                str(row.get("user") or "")
                for row in dashboard_users
                if str(row.get("copy_intent") or "") == "ready_row_draft"
            ],
            "diagnostic_note_users": [
                str(row.get("user") or "")
                for row in dashboard_users
                if str(row.get("copy_intent") or "") == "diagnostic_note"
            ],
            "open_tasks": sum(int(row.get("open_tasks") or 0) for row in dashboard_users),
            "total_tasks": dashboard_total_tasks,
            "complete_tasks": dashboard_complete_tasks,
            "completion_percent": _completion_percent(dashboard_complete_tasks, dashboard_total_tasks),
            "completion_states": _dashboard_completion_state_counts(dashboard_users),
            "recordings_without_open_tasks": sum(int(row.get("recordings_without_open_tasks") or 0) for row in dashboard_users),
            "invite_reasons": dashboard_invite_reasons,
            "copy_intents": _dashboard_copy_intent_counts(dashboard_users),
            "ready_states": _dashboard_ready_state_counts(dashboard_users),
            **dashboard_identity_probe_counts,
            **dashboard_dataset_queue_counts,
        },
        "dashboard_users": dashboard_users,
        "preflight": _server_safety_payload(config, include_admin_details=True),
        "assignments": assignments,
        "assignment_operator_rows": assignment_operator_rows,
        "tasks": tasks[:500],
        "active_sessions": active_sessions,
        "stale_sessions": stale_sessions,
        "failed_promotions": failed_promotions[:500],
        "recent_audit_events": recent_audit_events[:100],
    }

def _runtime_operator_validation_gate_cli_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_runtime_operator_validation_gate_cli_policy.v1",
        "validation_checklist_flag": "--validation-checklist",
        "preferred_require_flag": "--require-operator-validation-for-browser-work",
        "legacy_require_flag": "--require-operator-validation-for-start",
        "legacy_require_flag_retained_for_compatibility": True,
        "config_field": "require_operator_validation_for_start",
        "requires_validation_checklist": True,
        "protects_browser_start_open": True,
        "protects_browser_mutations": True,
        "blocks_before_session_creation": True,
        "blocks_before_target_token_check": True,
        "blocks_before_zarr_write": True,
        "blocks_before_audit_event_creation": True,
    }

def _safe_share_gate_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_safe_share_gate.v1",
        "gate": "labeler_links_safe_to_share",
        "inspection_command_requires": "--require-shareable",
        "required_inspection_field": "labeler_links_safe_to_share",
        "required_inspection_value": True,
        "ready_to_send_is_sufficient": False,
        "launch_blocking_evidence_gate_ids": [
            "mutable_zarr_backup_confirmation",
            "browser_response_security_headers",
            "identity_probe_verification",
            "browser_smoke",
            "disposable_zarr_mutation_smoke",
            "operator_recovery_contract",
        ],
        "operator_action": "Run inspect-handoff --path PACKAGE --require-shareable and share links only when labeler_links_safe_to_share is true.",
    }

def _safe_share_gate_flat_fields(policy: Mapping[str, object] | None = None) -> dict[str, object]:
    source = policy if isinstance(policy, Mapping) else _safe_share_gate_policy()
    launch_blocking_gate_ids = (
        source.get("launch_blocking_evidence_gate_ids")
        if isinstance(source.get("launch_blocking_evidence_gate_ids"), list)
        else _safe_share_gate_policy()["launch_blocking_evidence_gate_ids"]
    )
    required_inspection_field = str(
        source.get("required_inspection_field") or "labeler_links_safe_to_share"
    )
    ready_to_send_is_sufficient = bool(source.get("ready_to_send_is_sufficient"))
    return {
        "safe_share_gate_schema": str(
            source.get("schema") or "palette.web_labeling_safe_share_gate.v1"
        ),
        "safe_share_gate_id": str(source.get("gate") or "labeler_links_safe_to_share"),
        "safe_share_requires_require_shareable_inspection": (
            str(source.get("inspection_command_requires") or "") == "--require-shareable"
        ),
        "safe_share_ready_to_send_is_sufficient": ready_to_send_is_sufficient,
        "ready_to_send_is_sufficient_for_safe_share": ready_to_send_is_sufficient,
        "safe_share_required_inspection_field": required_inspection_field,
        required_inspection_field: False,
        "safe_share_required_inspection_value": bool(
            source.get("required_inspection_value", True)
        ),
        "safe_share_launch_blocking_evidence_gate_ids": list(launch_blocking_gate_ids),
        "safe_share_operator_action": str(
            source.get("operator_action")
            or "Run inspect-handoff --path PACKAGE --require-shareable and share links only when labeler_links_safe_to_share is true."
        ),
    }

def _safe_share_next_action_command_fields() -> list[str]:
    return [
        "operator_validation_command_template_schema",
        "operator_validation_command_ids",
        "operator_validation_record_command_ids",
        "operator_validation_apply_command_id",
        "operator_validation_apply_required_after_approval",
        "operator_validation_evidence_template_field",
        "operator_validation_evidence_template_path",
    ]

def _safe_share_next_action_detail_fields() -> list[str]:
    return [
        "gate_id",
        "status",
        "operator_only",
        "blocks_share",
        "action",
        *_safe_share_next_action_command_fields(),
    ]

def _safe_share_external_launch_evidence_gap_todo_fields() -> list[str]:
    return [
        "gate_id",
        "status",
        "operator_only",
        "blocks_share",
        "action",
        "operator_validation_command_template_schema",
        "operator_validation_record_command_ids",
        "operator_validation_apply_command_id",
        "operator_validation_apply_required_after_approval",
        "operator_validation_evidence_template_field",
        "operator_validation_evidence_template_path",
    ]

def _safe_share_launch_blocking_next_action_command_fields(
    gate_id: str,
) -> dict[str, object]:
    normalized_gate_id = str(gate_id).strip()
    templates = _operator_validation_command_templates(
        [normalized_gate_id] if normalized_gate_id else []
    )
    commands_by_gate_id = (
        templates.get("commands_by_gate_id")
        if isinstance(templates.get("commands_by_gate_id"), Mapping)
        else {}
    )
    command_ids = [
        str(command_id)
        for command_id in (
            commands_by_gate_id.get(normalized_gate_id)
            if isinstance(commands_by_gate_id.get(normalized_gate_id), list)
            else []
        )
        if str(command_id).strip()
    ]
    apply_required_gate_ids = {
        str(apply_gate_id)
        for apply_gate_id in (
            templates.get("apply_required_gate_ids")
            if isinstance(templates.get("apply_required_gate_ids"), list)
            else []
        )
        if str(apply_gate_id).strip()
    }
    apply_command_id = str(templates.get("apply_command_id") or "")
    evidence_templates_by_gate_id = (
        templates.get("evidence_templates_by_gate_id")
        if isinstance(templates.get("evidence_templates_by_gate_id"), Mapping)
        else {}
    )
    evidence_template = (
        evidence_templates_by_gate_id.get(normalized_gate_id)
        if isinstance(evidence_templates_by_gate_id.get(normalized_gate_id), Mapping)
        else {}
    )
    record_command_ids = [
        command_id
        for command_id in command_ids
        if command_id != apply_command_id
    ]
    return {
        "operator_validation_command_template_schema": str(
            templates.get("schema") or ""
        ),
        "operator_validation_command_ids": command_ids,
        "operator_validation_record_command_ids": record_command_ids,
        "operator_validation_apply_command_id": (
            apply_command_id if normalized_gate_id in apply_required_gate_ids else ""
        ),
        "operator_validation_apply_required_after_approval": (
            normalized_gate_id in apply_required_gate_ids
        ),
        "operator_validation_evidence_template_field": str(
            evidence_template.get("template_field") or ""
        ),
        "operator_validation_evidence_template_path": str(
            evidence_template.get("template_path") or ""
        ),
    }

def _safe_share_launch_blocking_next_action(gate_id: str, status: str) -> dict[str, object]:
    normalized_gate_id = str(gate_id).strip()
    normalized_status = str(status or "unknown").strip() or "unknown"
    if normalized_status == "missing_gate":
        action = (
            f"Regenerate or repair the validation checklist so safe-share "
            f"launch-blocking gate {normalized_gate_id} is present before sharing links."
        )
    elif normalized_status == "pending_operator_evidence":
        action = (
            f"Record and approve operator evidence for safe-share launch-blocking "
            f"gate {normalized_gate_id}, then apply approved evidence templates "
            "and refresh checksums if required."
        )
    elif normalized_status == "needs_review":
        action = (
            f"Resolve operator review for safe-share launch-blocking gate "
            f"{normalized_gate_id}; links remain unsafe until the gate is passed "
            "or explicitly not_applicable."
        )
    elif normalized_status == "missing_evidence":
        action = (
            f"Attach approved evidence for safe-share launch-blocking gate "
            f"{normalized_gate_id}, apply evidence templates, and rerun "
            "shareability inspection before sharing links."
        )
    else:
        action = (
            f"Normalize safe-share launch-blocking gate {normalized_gate_id} "
            f"from status {normalized_status!r} to passed, not_applicable, or a "
            "known blocking status before sharing links."
        )
    return {
        "gate_id": normalized_gate_id,
        "status": normalized_status,
        "operator_only": True,
        "blocks_share": True,
        "action": action,
        **_safe_share_launch_blocking_next_action_command_fields(normalized_gate_id),
    }

def _safe_share_next_action_summary_from_fields(
    *,
    actions: Sequence[object],
    statuses: Mapping[str, object],
    count: int | None = None,
) -> str:
    parts: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        gate_id = str(action.get("gate_id") or "").strip()
        status = str(action.get("status") or "").strip()
        if gate_id:
            parts.append(f"{gate_id}={status or 'unknown'}")
    if not parts:
        for gate_id, status in statuses.items():
            normalized_status = str(status or "unknown").strip() or "unknown"
            if normalized_status not in {"passed", "not_applicable"}:
                parts.append(f"{gate_id}={normalized_status}")
    resolved_count = int(count if count is not None else len(parts))
    if resolved_count <= 0:
        return "Safe-share next actions: 0; no launch-blocking evidence blockers reported."
    detail = ", ".join(parts) if parts else "inspect machine-readable safe_share_launch_blocking_next_actions"
    return f"Safe-share next actions: {resolved_count}; {detail}"

def _safe_share_external_launch_evidence_gap_fields(
    *,
    gate_statuses: Mapping[str, object],
    unsatisfied_gate_ids: Sequence[object],
    next_actions: Sequence[object],
) -> dict[str, object]:
    gap_gate_ids = [
        str(gate_id)
        for gate_id in unsatisfied_gate_ids
        if str(gate_id).strip()
    ]
    gap_statuses = {
        gate_id: str(gate_statuses.get(gate_id) or "unknown")
        for gate_id in gap_gate_ids
    }
    template_paths_by_gate_id: dict[str, str] = {}
    record_command_ids_by_gate_id: dict[str, list[str]] = {}
    gap_gate_id_set = set(gap_gate_ids)
    actions_by_gate_id: dict[str, Mapping[str, object]] = {}
    for action in next_actions:
        if not isinstance(action, Mapping):
            continue
        gate_id = str(action.get("gate_id") or "")
        if gate_id not in gap_gate_id_set:
            continue
        actions_by_gate_id.setdefault(gate_id, action)
        template_path = str(
            action.get("operator_validation_evidence_template_path") or ""
        )
        if template_path:
            template_paths_by_gate_id[gate_id] = template_path
        record_command_ids = action.get("operator_validation_record_command_ids")
        if isinstance(record_command_ids, list):
            normalized_record_command_ids = [
                str(command_id)
                for command_id in record_command_ids
                if str(command_id).strip()
            ]
            if normalized_record_command_ids:
                record_command_ids_by_gate_id[gate_id] = normalized_record_command_ids
    todo_fields = _safe_share_external_launch_evidence_gap_todo_fields()
    todos: list[dict[str, object]] = []
    for gate_id in gap_gate_ids:
        action = actions_by_gate_id.get(gate_id, {})
        todos.append(
            {
                "gate_id": gate_id,
                "status": gap_statuses.get(gate_id, "unknown"),
                "operator_only": bool(action.get("operator_only", True)),
                "blocks_share": bool(action.get("blocks_share", True)),
                "action": str(action.get("action") or ""),
                "operator_validation_command_template_schema": str(
                    action.get("operator_validation_command_template_schema") or ""
                ),
                "operator_validation_record_command_ids": list(
                    record_command_ids_by_gate_id.get(gate_id, [])
                ),
                "operator_validation_apply_command_id": str(
                    action.get("operator_validation_apply_command_id") or ""
                ),
                "operator_validation_apply_required_after_approval": bool(
                    action.get("operator_validation_apply_required_after_approval")
                ),
                "operator_validation_evidence_template_field": str(
                    action.get("operator_validation_evidence_template_field") or ""
                ),
                "operator_validation_evidence_template_path": str(
                    template_paths_by_gate_id.get(gate_id)
                    or action.get("operator_validation_evidence_template_path")
                    or ""
                ),
            }
        )
    if gap_gate_ids:
        summary = (
            f"External launch evidence gaps: {len(gap_gate_ids)}; "
            + ", ".join(f"{gate_id}={gap_statuses[gate_id]}" for gate_id in gap_gate_ids)
        )
    else:
        summary = "External launch evidence gaps: 0; safe-share evidence gates satisfied."
    return {
        "safe_share_external_launch_evidence_gap_gate_ids": gap_gate_ids,
        "safe_share_external_launch_evidence_gap_count": len(gap_gate_ids),
        "safe_share_external_launch_evidence_gap_statuses": gap_statuses,
        "safe_share_external_launch_evidence_gap_action_required": bool(gap_gate_ids),
        "safe_share_external_launch_evidence_gap_summary": summary,
        "safe_share_external_launch_evidence_gap_todos": todos,
        "safe_share_external_launch_evidence_gap_todo_count": len(todos),
        "safe_share_external_launch_evidence_gap_todo_fields": todo_fields,
        "safe_share_external_launch_evidence_gap_template_paths_by_gate_id": (
            template_paths_by_gate_id
        ),
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id": (
            record_command_ids_by_gate_id
        ),
    }

def _safe_share_checklist_gate_status_fields(
    *,
    gates: Sequence[Mapping[str, object]] | Sequence[object],
    safe_share_gate: Mapping[str, object],
) -> dict[str, object]:
    blocking_gate_ids = [
        str(gate_id)
        for gate_id in (
            safe_share_gate.get("launch_blocking_evidence_gate_ids")
            if isinstance(safe_share_gate.get("launch_blocking_evidence_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    ]
    gates_by_id = {
        str(gate.get("id") or ""): gate
        for gate in gates
        if isinstance(gate, Mapping) and str(gate.get("id") or "").strip()
    }
    gate_statuses: dict[str, str] = {}
    missing_gate_ids: list[str] = []
    pending_gate_ids: list[str] = []
    needs_review_gate_ids: list[str] = []
    missing_evidence_gate_ids: list[str] = []
    unknown_gate_ids: list[str] = []
    satisfied_gate_ids: list[str] = []
    unsatisfied_gate_ids: list[str] = []
    next_actions: list[dict[str, object]] = []
    for gate_id in blocking_gate_ids:
        gate = gates_by_id.get(gate_id)
        if gate is None:
            status = "missing_gate"
        else:
            status = str(gate.get("status") or "unknown").strip() or "unknown"
        gate_statuses[gate_id] = status
        if gate is None:
            missing_gate_ids.append(gate_id)
            unsatisfied_gate_ids.append(gate_id)
            next_actions.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "operator_only": True,
                    "blocks_share": True,
                    "action": (
                        f"Regenerate or repair the validation checklist so safe-share "
                        f"launch-blocking gate {gate_id} is present before sharing links."
                    ),
                }
            )
            continue
        if status in {"passed", "not_applicable"}:
            satisfied_gate_ids.append(gate_id)
        elif status == "pending_operator_evidence":
            pending_gate_ids.append(gate_id)
            unsatisfied_gate_ids.append(gate_id)
            next_actions.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "operator_only": True,
                    "blocks_share": True,
                    "action": (
                        f"Record and approve operator evidence for safe-share "
                        f"launch-blocking gate {gate_id}, then apply approved evidence "
                        "templates and refresh checksums if required."
                    ),
                }
            )
        elif status == "needs_review":
            needs_review_gate_ids.append(gate_id)
            unsatisfied_gate_ids.append(gate_id)
            next_actions.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "operator_only": True,
                    "blocks_share": True,
                    "action": (
                        f"Resolve operator review for safe-share launch-blocking gate "
                        f"{gate_id}; links remain unsafe until the gate is passed or "
                        "explicitly not_applicable."
                    ),
                }
            )
        elif status == "missing_evidence":
            missing_evidence_gate_ids.append(gate_id)
            unsatisfied_gate_ids.append(gate_id)
            next_actions.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "operator_only": True,
                    "blocks_share": True,
                    "action": (
                        f"Attach approved evidence for safe-share launch-blocking gate "
                        f"{gate_id}, apply evidence templates, and rerun shareability "
                        "inspection before sharing links."
                    ),
                }
            )
        else:
            unknown_gate_ids.append(gate_id)
            unsatisfied_gate_ids.append(gate_id)
            next_actions.append(
                {
                    "gate_id": gate_id,
                    "status": status,
                    "operator_only": True,
                    "blocks_share": True,
                    "action": (
                        f"Normalize safe-share launch-blocking gate {gate_id} from "
                        f"status {status!r} to passed, not_applicable, or a known "
                        "blocking status before sharing links."
                    ),
                }
            )
    next_actions = [
        {
            **action,
            **_safe_share_launch_blocking_next_action_command_fields(
                str(action.get("gate_id") or "")
            ),
        }
        for action in next_actions
    ]
    checklist_complete = bool(blocking_gate_ids) and not unsatisfied_gate_ids
    operator_action = (
        "Checklist-side safe-share evidence blockers are satisfied; still run inspect-handoff --require-shareable and require labeler_links_safe_to_share=true before sharing links."
        if checklist_complete
        else "Complete or mark not_applicable every safe-share launch-blocking evidence gate, apply approved evidence templates, refresh checksums when required, then run inspect-handoff --require-shareable before sharing links."
    )
    external_launch_evidence_gap_fields = (
        _safe_share_external_launch_evidence_gap_fields(
            gate_statuses=gate_statuses,
            unsatisfied_gate_ids=unsatisfied_gate_ids,
            next_actions=next_actions,
        )
    )
    return {
        "safe_share_launch_blocking_gate_statuses": gate_statuses,
        "safe_share_launch_blocking_missing_gate_ids": missing_gate_ids,
        "safe_share_launch_blocking_pending_gate_ids": pending_gate_ids,
        "safe_share_launch_blocking_needs_review_gate_ids": needs_review_gate_ids,
        "safe_share_launch_blocking_missing_evidence_gate_ids": missing_evidence_gate_ids,
        "safe_share_launch_blocking_unknown_gate_ids": unknown_gate_ids,
        "safe_share_launch_blocking_satisfied_gate_ids": satisfied_gate_ids,
        "safe_share_launch_blocking_unsatisfied_gate_ids": unsatisfied_gate_ids,
        "safe_share_launch_blocking_next_actions": next_actions,
        "safe_share_launch_blocking_next_action_detail_fields": (
            _safe_share_next_action_detail_fields()
        ),
        "safe_share_launch_blocking_next_action_command_fields": (
            _safe_share_next_action_command_fields()
        ),
        "safe_share_launch_blocking_next_action_count": len(next_actions),
        "safe_share_next_action_summary": _safe_share_next_action_summary_from_fields(
            actions=next_actions,
            statuses=gate_statuses,
            count=len(next_actions),
        ),
        "safe_share_launch_blocking_gate_count": len(blocking_gate_ids),
        "safe_share_launch_blocking_missing_gate_count": len(missing_gate_ids),
        "safe_share_launch_blocking_pending_gate_count": len(pending_gate_ids),
        "safe_share_launch_blocking_needs_review_gate_count": len(needs_review_gate_ids),
        "safe_share_launch_blocking_missing_evidence_gate_count": len(missing_evidence_gate_ids),
        "safe_share_launch_blocking_unknown_gate_count": len(unknown_gate_ids),
        "safe_share_launch_blocking_satisfied_gate_count": len(satisfied_gate_ids),
        "safe_share_launch_blocking_unsatisfied_gate_count": len(unsatisfied_gate_ids),
        "safe_share_checklist_gate_evidence_complete": checklist_complete,
        "safe_share_checklist_operator_action": operator_action,
        **external_launch_evidence_gap_fields,
    }

def _safe_share_checklist_gate_status_fields_from_operator_validation(
    source: Mapping[str, object],
    *,
    safe_share_gate: Mapping[str, object],
) -> dict[str, object]:
    exact_fields = _safe_share_checklist_field_values(source)
    explicit_summary = str(source.get("safe_share_next_action_summary") or "").strip()
    has_operator_validation_projection = any(
        str(key).startswith("operator_validation_") for key in source.keys()
    )
    if (
        exact_fields["safe_share_launch_blocking_gate_statuses"]
        or exact_fields["safe_share_launch_blocking_next_actions"]
        or (
            explicit_summary
            and (
                int(exact_fields.get("safe_share_launch_blocking_next_action_count") or 0) > 0
                or not has_operator_validation_projection
            )
        )
    ):
        return exact_fields
    public = _operator_validation_public_fields(source)
    pending_gate_ids = {
        str(gate_id)
        for gate_id in (
            public.get("operator_validation_pending_gate_ids")
            if isinstance(public.get("operator_validation_pending_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    }
    needs_review_gate_ids = {
        str(gate_id)
        for gate_id in (
            public.get("operator_validation_needs_review_gate_ids")
            if isinstance(public.get("operator_validation_needs_review_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    }
    missing_evidence_gate_ids = {
        str(gate_id)
        for gate_id in (
            public.get("operator_validation_required_missing_evidence_gate_ids")
            if isinstance(public.get("operator_validation_required_missing_evidence_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    }
    missing_evidence_gate_ids.update(
        str(gate_id)
        for gate_id in (
            public.get("operator_validation_missing_evidence_gate_ids")
            if isinstance(public.get("operator_validation_missing_evidence_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    )
    blocking_gate_ids = [
        str(gate_id)
        for gate_id in (
            safe_share_gate.get("launch_blocking_evidence_gate_ids")
            if isinstance(safe_share_gate.get("launch_blocking_evidence_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    ]
    pseudo_gates: list[dict[str, object]] = []
    for gate_id in blocking_gate_ids:
        flat_status = str(public.get(f"operator_validation_gate_{gate_id}_status") or "").strip()
        if flat_status in {"passed", "not_applicable", "pending_operator_evidence", "needs_review", "missing_evidence"}:
            status = flat_status
        elif gate_id in needs_review_gate_ids:
            status = "needs_review"
        elif gate_id in missing_evidence_gate_ids:
            status = "missing_evidence"
        elif gate_id in pending_gate_ids:
            status = "pending_operator_evidence"
        else:
            status = "unknown"
        pseudo_gates.append({"id": gate_id, "status": status})
    return _safe_share_checklist_gate_status_fields(
        gates=pseudo_gates,
        safe_share_gate=safe_share_gate,
    )

def _safe_share_checklist_field_values(source: Mapping[str, object]) -> dict[str, object]:
    def _merge_safe_share_action_command_fields(
        action: Mapping[str, object],
    ) -> dict[str, object]:
        merged = dict(action)
        command_fields = _safe_share_launch_blocking_next_action_command_fields(
            str(action.get("gate_id") or "")
        )
        for key, value in command_fields.items():
            existing = merged.get(key)
            if key in {
                "operator_validation_command_ids",
                "operator_validation_record_command_ids",
            } and isinstance(existing, list) and isinstance(value, list):
                merged[key] = [
                    str(command_id)
                    for command_id in dict.fromkeys([*existing, *value])
                    if str(command_id).strip()
                ]
            elif existing in (None, "", [], {}):
                merged[key] = value
            elif (
                key == "operator_validation_apply_required_after_approval"
                and not bool(existing)
                and bool(value)
            ):
                merged[key] = value
        return merged

    list_fields = [
        "safe_share_launch_blocking_missing_gate_ids",
        "safe_share_launch_blocking_pending_gate_ids",
        "safe_share_launch_blocking_needs_review_gate_ids",
        "safe_share_launch_blocking_missing_evidence_gate_ids",
        "safe_share_launch_blocking_unknown_gate_ids",
        "safe_share_launch_blocking_satisfied_gate_ids",
        "safe_share_launch_blocking_unsatisfied_gate_ids",
    ]
    count_fields = [
        "safe_share_launch_blocking_gate_count",
        "safe_share_launch_blocking_missing_gate_count",
        "safe_share_launch_blocking_pending_gate_count",
        "safe_share_launch_blocking_needs_review_gate_count",
        "safe_share_launch_blocking_missing_evidence_gate_count",
        "safe_share_launch_blocking_unknown_gate_count",
        "safe_share_launch_blocking_satisfied_gate_count",
        "safe_share_launch_blocking_unsatisfied_gate_count",
    ]
    gate_statuses = source.get("safe_share_launch_blocking_gate_statuses")
    normalized_gate_statuses = dict(gate_statuses) if isinstance(gate_statuses, Mapping) else {}
    gap_todos = source.get("safe_share_external_launch_evidence_gap_todos")
    normalized_gap_todo_actions = [
        {
            "gate_id": str(todo.get("gate_id") or ""),
            "status": str(todo.get("status") or "unknown"),
            "operator_only": bool(todo.get("operator_only", True)),
            "blocks_share": bool(todo.get("blocks_share", True)),
            "action": str(todo.get("action") or ""),
            "operator_validation_command_template_schema": str(
                todo.get("operator_validation_command_template_schema") or ""
            ),
            "operator_validation_command_ids": list(
                todo.get("operator_validation_command_ids")
                if isinstance(todo.get("operator_validation_command_ids"), list)
                else []
            ),
            "operator_validation_record_command_ids": list(
                todo.get("operator_validation_record_command_ids")
                if isinstance(todo.get("operator_validation_record_command_ids"), list)
                else []
            ),
            "operator_validation_apply_command_id": str(
                todo.get("operator_validation_apply_command_id") or ""
            ),
            "operator_validation_apply_required_after_approval": bool(
                todo.get("operator_validation_apply_required_after_approval")
            ),
            "operator_validation_evidence_template_field": str(
                todo.get("operator_validation_evidence_template_field") or ""
            ),
            "operator_validation_evidence_template_path": str(
                todo.get("operator_validation_evidence_template_path") or ""
            ),
        }
        for todo in gap_todos
        if isinstance(todo, Mapping)
    ] if isinstance(gap_todos, list) else []
    next_actions = source.get("safe_share_launch_blocking_next_actions")
    normalized_next_actions = [
        {
            "gate_id": str(action.get("gate_id") or ""),
            "status": str(action.get("status") or ""),
            "operator_only": bool(action.get("operator_only", True)),
            "blocks_share": bool(action.get("blocks_share", True)),
            "action": str(action.get("action") or ""),
            "operator_validation_command_template_schema": str(
                action.get("operator_validation_command_template_schema") or ""
            ),
            "operator_validation_command_ids": list(
                action.get("operator_validation_command_ids")
                if isinstance(action.get("operator_validation_command_ids"), list)
                else []
            ),
            "operator_validation_record_command_ids": list(
                action.get("operator_validation_record_command_ids")
                if isinstance(action.get("operator_validation_record_command_ids"), list)
                else []
            ),
            "operator_validation_apply_command_id": str(
                action.get("operator_validation_apply_command_id") or ""
            ),
            "operator_validation_apply_required_after_approval": bool(
                action.get("operator_validation_apply_required_after_approval")
            ),
            "operator_validation_evidence_template_field": str(
                action.get("operator_validation_evidence_template_field") or ""
            ),
            "operator_validation_evidence_template_path": str(
                action.get("operator_validation_evidence_template_path") or ""
            ),
        }
        for action in next_actions
        if isinstance(action, Mapping)
    ] if isinstance(next_actions, list) else []
    normalized_next_actions = [
        _merge_safe_share_action_command_fields(action)
        for action in normalized_next_actions
    ]
    if not normalized_next_actions and normalized_gap_todo_actions:
        normalized_next_actions = [
            _merge_safe_share_action_command_fields(action)
            for action in normalized_gap_todo_actions
        ]
    if not normalized_gate_statuses and normalized_next_actions:
        normalized_gate_statuses = {
            str(action.get("gate_id") or ""): str(action.get("status") or "unknown")
            for action in normalized_next_actions
            if str(action.get("gate_id") or "").strip()
        }
    if not normalized_next_actions and normalized_gate_statuses:
        normalized_next_actions = [
            _safe_share_launch_blocking_next_action(str(gate_id), str(status or "unknown"))
            for gate_id, status in normalized_gate_statuses.items()
            if str(status or "unknown") not in {"passed", "not_applicable"}
        ]
    normalized_list_values = {
        field: (
            [str(item) for item in source.get(field)]
            if isinstance(source.get(field), list)
            else []
        )
        for field in list_fields
    }
    if normalized_gate_statuses and not any(normalized_list_values.values()):
        for gate_id, status_value in normalized_gate_statuses.items():
            status = str(status_value or "unknown")
            if status in {"passed", "not_applicable"}:
                normalized_list_values["safe_share_launch_blocking_satisfied_gate_ids"].append(
                    str(gate_id)
                )
            else:
                normalized_list_values["safe_share_launch_blocking_unsatisfied_gate_ids"].append(
                    str(gate_id)
                )
                if status == "missing_gate":
                    normalized_list_values["safe_share_launch_blocking_missing_gate_ids"].append(
                        str(gate_id)
                    )
                elif status == "pending_operator_evidence":
                    normalized_list_values["safe_share_launch_blocking_pending_gate_ids"].append(
                        str(gate_id)
                    )
                elif status == "needs_review":
                    normalized_list_values[
                        "safe_share_launch_blocking_needs_review_gate_ids"
                    ].append(str(gate_id))
                elif status == "missing_evidence":
                    normalized_list_values[
                        "safe_share_launch_blocking_missing_evidence_gate_ids"
                    ].append(str(gate_id))
                else:
                    normalized_list_values["safe_share_launch_blocking_unknown_gate_ids"].append(
                        str(gate_id)
                    )
    normalized_counts = {}
    for field in count_fields:
        if field == "safe_share_launch_blocking_gate_count" and source.get(field) is None:
            normalized_counts[field] = len(normalized_gate_statuses)
        elif source.get(field) is None and field.startswith("safe_share_launch_blocking_"):
            list_field = field.replace("_count", "_ids")
            normalized_counts[field] = len(normalized_list_values.get(list_field, []))
        else:
            normalized_counts[field] = int(source.get(field) or 0)
    normalized_counts["safe_share_launch_blocking_next_action_count"] = int(
        source.get("safe_share_launch_blocking_next_action_count")
        or len(normalized_next_actions)
    )
    safe_share_next_action_summary = str(source.get("safe_share_next_action_summary") or "")
    if not safe_share_next_action_summary:
        safe_share_next_action_summary = _safe_share_next_action_summary_from_fields(
            actions=normalized_next_actions,
            statuses=normalized_gate_statuses,
            count=normalized_counts["safe_share_launch_blocking_next_action_count"],
        )
    external_launch_evidence_gap_fields = (
        _safe_share_external_launch_evidence_gap_fields(
            gate_statuses=normalized_gate_statuses,
            unsatisfied_gate_ids=normalized_list_values[
                "safe_share_launch_blocking_unsatisfied_gate_ids"
            ],
            next_actions=normalized_next_actions,
        )
    )
    return {
        "safe_share_launch_blocking_gate_statuses": normalized_gate_statuses,
        "safe_share_launch_blocking_next_actions": normalized_next_actions,
        "safe_share_launch_blocking_next_action_detail_fields": (
            _safe_share_next_action_detail_fields()
        ),
        "safe_share_launch_blocking_next_action_command_fields": (
            _safe_share_next_action_command_fields()
        ),
        **normalized_list_values,
        **normalized_counts,
        "safe_share_next_action_summary": safe_share_next_action_summary,
        "safe_share_checklist_gate_evidence_complete": bool(
            source.get("safe_share_checklist_gate_evidence_complete")
        ),
        "safe_share_checklist_operator_action": str(
            source.get("safe_share_checklist_operator_action") or ""
        ),
        **external_launch_evidence_gap_fields,
    }

def _server_safety_payload(config: ServerConfig, *, include_admin_details: bool = False) -> dict[str, object]:
    warnings: list[str] = []
    if config.fixed_user:
        auth_mode = "fixed_user"
    elif config.trust_auth_header:
        auth_mode = f"trusted_header:{config.auth_header}"
    else:
        auth_mode = "disabled"
    if config.fixed_user and not _is_loopback_host(config.host):
        warnings.append("fixed_user_auth_on_non_loopback_host")
    if config.production and config.fixed_user:
        warnings.append("production_fixed_user_auth")
    if config.production and not config.trust_auth_header:
        warnings.append("production_auth_header_not_trusted")
    if not config.fixed_user and not config.trust_auth_header:
        warnings.append("auth_header_not_trusted")
    if config.trust_auth_header and not config.auth_header:
        warnings.append("missing_auth_header")
    if config.trust_auth_header and not _is_loopback_host(config.host):
        warnings.append("trusted_header_auth_on_non_loopback_host")
    if not _is_loopback_host(config.host) and not config.allow_non_loopback:
        warnings.append("non_loopback_bind_not_allowed")
    if not config.admin_users:
        warnings.append("no_admin_users_configured")
    if not config.link_secret:
        warnings.append("signed_links_disabled")
    if config.production and not config.require_operator_validation_for_start:
        warnings.append("production_operator_validation_start_gate_disabled")
        warnings.append("production_operator_validation_mutation_gate_disabled")
    if config.link_not_before_utc:
        try:
            _utc_timestamp(config.link_not_before_utc)
        except Exception:
            warnings.append("invalid_link_not_before_utc")
    if not config.csrf_same_origin:
        warnings.append("same_origin_post_guard_disabled")
    raw_operator_validation_fields = _dashboard_operator_validation_fields_for_config(config)
    operator_validation_fields = _operator_validation_public_fields(raw_operator_validation_fields)
    operator_validation_report = {
        **operator_validation_fields,
        **_operator_validation_gate_metadata_fields(),
        **_operator_validation_gate_flat_fields(operator_validation_fields),
    }
    operator_validation_command_templates = _operator_validation_command_templates(
        operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids")
        if isinstance(
            operator_validation_fields.get("operator_validation_required_missing_evidence_gate_ids"),
            list,
        )
        else None
    )
    safe_share_gate = _safe_share_gate_policy()
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        raw_operator_validation_fields,
        safe_share_gate=safe_share_gate,
    )
    payload: dict[str, object] = {
        "auth_mode": auth_mode,
        "identity_source_policy": _identity_source_policy(config),
        "operator_authorization_policy": _operator_authorization_policy(config, include_admin_details=include_admin_details),
        "operator_recovery_policy": _operator_recovery_policy(),
        "operator_recovery_contract": _operator_recovery_contract_policy(_operator_recovery_policy()),
        "fixed_user_enabled": bool(config.fixed_user),
        "trusted_auth_header_enabled": bool(config.trust_auth_header),
        "auth_header": str(config.auth_header),
        "host": str(config.host),
        "port": int(config.port),
        "admin_user_count": len(config.admin_users),
        "signed_links_enabled": bool(config.link_secret),
        "link_not_before_utc": str(config.link_not_before_utc or ""),
        "same_origin_post_guard_enabled": bool(config.csrf_same_origin),
        "labeler_landing_page_path": "/",
        "dashboard_path": DASHBOARD_PATH,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "personal_work_alias_for": DASHBOARD_PATH,
        "personal_dataset_queue_alias_for": DATASET_QUEUE_PATH,
        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
        "preferred_labeler_entry_path": PERSONAL_DATASET_QUEUE_PATH,
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_path": PERSONAL_DATASET_QUEUE_PATH,
        "single_owner_policy": _assignment_ownership_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
        "runtime_operator_validation_gate_cli_policy": (
            _runtime_operator_validation_gate_cli_policy()
        ),
        "safe_share_gate": safe_share_gate,
        **_safe_share_gate_flat_fields(safe_share_gate),
        **safe_share_checklist_fields,
        "operator_validation_start_gate": _runtime_operator_validation_start_gate(
            config,
            include_operator_details=include_admin_details,
        ),
        "operator_validation_mutation_gate": _runtime_operator_validation_mutation_gate(
            config,
            include_operator_details=include_admin_details,
        ),
        "browser_response_security_policy": _browser_response_security_policy(),
        "operator_validation": operator_validation_report,
        "operator_validation_command_templates": operator_validation_command_templates,
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        "access_log_enabled": bool(config.access_log),
        "allow_non_loopback": bool(config.allow_non_loopback),
        "production_enabled": bool(config.production),
        "session_ttl_seconds": int(config.session_ttl_seconds),
        "warnings": warnings,
        "warning_count": len(warnings),
    }
    if include_admin_details:
        payload["admin_users"] = list(config.admin_users)
        payload["store_path"] = str(config.store_path)
    return payload

def _admin_datasets_payload(
    store: LabelingStore,
    *,
    config: ServerConfig,
    dataset_id: str | None = None,
    recording_id: str | None = None,
    assignee_user: str | None = None,
    status: str | None = None,
    warnings_only: bool = False,
) -> dict[str, object]:
    dataset_filter = str(dataset_id or "").strip()
    recording_filter = str(recording_id or "").strip()
    user_filter = str(assignee_user or "").strip()
    status_filter = str(status or "").strip().lower()
    assignments = store.list_assignments(status=None)
    tasks = store.list_tasks(include_completed=True)
    active_sessions = store.list_sessions(include_closed=False, limit=10000)
    recent_events = store.list_events(limit=5000)
    assignments_by_recording = {
        str(row.get("recording_id") or ""): row
        for row in assignments
        if str(row.get("recording_id") or "")
    }
    sessions_by_recording: dict[str, list[Mapping[str, object]]] = {}
    stale_sessions_by_recording: dict[str, int] = {}
    for session in active_sessions:
        recording_id = str(session.get("recording_id") or "")
        if recording_id:
            sessions_by_recording.setdefault(recording_id, []).append(session)
            if _session_is_expired(session):
                stale_sessions_by_recording[recording_id] = stale_sessions_by_recording.get(recording_id, 0) + 1

    latest_event_by_recording: dict[str, Mapping[str, object]] = {}
    latest_save_by_recording: dict[str, Mapping[str, object]] = {}
    failed_promotion_counts: dict[str, int] = {}
    for event in recent_events:
        recording_id = str(event.get("recording_id") or "")
        if not recording_id:
            continue
        created = str(event.get("created_at_utc") or "")
        current_latest = latest_event_by_recording.get(recording_id)
        if current_latest is None or created >= str(current_latest.get("created_at_utc") or ""):
            latest_event_by_recording[recording_id] = event
        event_type = str(event.get("event_type") or "")
        if event_type.startswith("save_") or event_type in {"set_review_status", "complete_task"}:
            current_save = latest_save_by_recording.get(recording_id)
            if current_save is None or created >= str(current_save.get("created_at_utc") or ""):
                latest_save_by_recording[recording_id] = event
        if event_type == "promotion_failed":
            failed_promotion_counts[recording_id] = failed_promotion_counts.get(recording_id, 0) + 1

    dataset_rows: dict[str, dict[str, object]] = {}
    recording_rows_by_dataset: dict[str, dict[str, dict[str, object]]] = {}

    def dataset_row(dataset_key: str, dataset_id: str, zarr_use: str) -> dict[str, object]:
        row = dataset_rows.get(dataset_key)
        if row is None:
            row = {
                "dataset_key": dataset_key,
                "dataset_id": dataset_id,
                "dataset_label": dataset_id or "Assigned recordings without generated task dataset",
                "zarr_uses": sorted({zarr_use} if zarr_use else set()),
                "recording_count": 0,
                "assignees": [],
                "task_count": 0,
                "open_task_count": 0,
                "complete_task_count": 0,
                "non_startable_task_count": 0,
                "blocked_recording_count": 0,
                "active_session_count": 0,
                "stale_session_count": 0,
                "failed_promotion_count": 0,
                "workflow_counts": {},
                "state_counts": {},
                "recordings": [],
            }
            dataset_rows[dataset_key] = row
            recording_rows_by_dataset[dataset_key] = {}
        elif zarr_use:
            zarr_uses = set(str(item) for item in row.get("zarr_uses", []) if str(item))
            zarr_uses.add(zarr_use)
            row["zarr_uses"] = sorted(zarr_uses)
        return row

    def recording_row(
        dataset_key: str,
        *,
        dataset_id: str,
        zarr_use: str,
        recording_id: str,
    ) -> dict[str, object]:
        per_dataset = recording_rows_by_dataset.setdefault(dataset_key, {})
        row = per_dataset.get(recording_id)
        if row is None:
            assignment = assignments_by_recording.get(recording_id, {})
            assignee_user = str(assignment.get("assignee_user") or "")
            assignment_status = str(assignment.get("status") or "")
            row = {
                "recording_id": recording_id,
                "dataset_id": dataset_id,
                "zarr_uses": sorted({zarr_use} if zarr_use else set()),
                "assignee_user": assignee_user,
                "assignment_status": assignment_status,
                "assignment_notes": str(assignment.get("notes") or ""),
                "assigned_by": str(assignment.get("assigned_by") or ""),
                "assigned_at_utc": str(assignment.get("assigned_at_utc") or ""),
                "task_count": 0,
                "open_task_count": 0,
                "complete_task_count": 0,
                "non_startable_task_count": 0,
                "workflow_counts": {},
                "state_counts": {},
                "active_session_count": len(sessions_by_recording.get(recording_id, [])),
                "stale_session_count": int(stale_sessions_by_recording.get(recording_id, 0)),
                "failed_promotion_count": int(failed_promotion_counts.get(recording_id, 0)),
                "latest_event": dict(latest_event_by_recording.get(recording_id, {})),
                "latest_save_event": dict(latest_save_by_recording.get(recording_id, {})),
                "tasks": [],
                "admin_recording_url": f"/admin/recordings/{quote(recording_id, safe='')}",
                "expected_user_personal_dataset_queue_url": (
                    _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, assignee_user)
                    if assignee_user
                    else ""
                ),
                "expected_user_personal_work_url": (
                    _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, assignee_user)
                    if assignee_user
                    else ""
                ),
            }
            per_dataset[recording_id] = row
            dataset = dataset_rows[dataset_key]
            recordings = dataset["recordings"]
            assert isinstance(recordings, list)
            recordings.append(row)
        elif zarr_use:
            zarr_uses = set(str(item) for item in row.get("zarr_uses", []) if str(item))
            zarr_uses.add(zarr_use)
            row["zarr_uses"] = sorted(zarr_uses)
        return row

    for task in tasks:
        recording_id = str(task.get("recording_id") or "")
        dataset_id = str(task.get("dataset_id") or "")
        zarr_use = str(task.get("zarr_use") or "")
        assignment = assignments_by_recording.get(recording_id, {})
        task_assignee_user = str(task.get("assignee_user") or assignment.get("assignee_user") or "")
        if dataset_filter and dataset_id != dataset_filter:
            continue
        if recording_filter and recording_id != recording_filter:
            continue
        if user_filter and task_assignee_user != user_filter:
            continue
        dataset_key = dataset_id or f"recording:{recording_id or 'unscoped'}"
        dataset = dataset_row(dataset_key, dataset_id, zarr_use)
        recording = recording_row(
            dataset_key,
            dataset_id=dataset_id,
            zarr_use=zarr_use,
            recording_id=recording_id,
        )
        compact_task = _admin_compact_task(task)
        task_list = recording["tasks"]
        assert isinstance(task_list, list)
        task_list.append(compact_task)
        task_state = str(task.get("state") or "")
        task_startable = task_state in LABELER_START_TASK_STATES
        for target in (dataset, recording):
            target["task_count"] = int(target.get("task_count") or 0) + 1
            state_counts = target["state_counts"]
            assert isinstance(state_counts, dict)
            state_counts[task_state or "unknown"] = int(state_counts.get(task_state or "unknown", 0)) + 1
            workflow_counts = target["workflow_counts"]
            assert isinstance(workflow_counts, dict)
            workflow = str(task.get("workflow_kind") or "unknown")
            workflow_counts[workflow] = int(workflow_counts.get(workflow, 0)) + 1
            if task_state == "complete":
                target["complete_task_count"] = int(target.get("complete_task_count") or 0) + 1
            elif task_startable:
                target["open_task_count"] = int(target.get("open_task_count") or 0) + 1
            else:
                target["non_startable_task_count"] = int(target.get("non_startable_task_count") or 0) + 1

    no_task_dataset_key = "__assigned_recordings_without_tasks__"
    for assignment in assignments:
        recording_id = str(assignment.get("recording_id") or "")
        if not recording_id:
            continue
        if dataset_filter:
            continue
        if recording_filter and recording_id != recording_filter:
            continue
        if user_filter and str(assignment.get("assignee_user") or "") != user_filter:
            continue
        if any(
            recording_id in per_dataset
            for per_dataset in recording_rows_by_dataset.values()
        ):
            continue
        dataset_row(no_task_dataset_key, "", "")
        recording_row(
            no_task_dataset_key,
            dataset_id="",
            zarr_use="",
            recording_id=recording_id,
        )

    dataset_list = list(dataset_rows.values())
    for dataset in dataset_list:
        recordings = dataset.get("recordings") if isinstance(dataset.get("recordings"), list) else []
        assignees = sorted(
            {
                str(recording.get("assignee_user") or "")
                for recording in recordings
                if isinstance(recording, Mapping) and str(recording.get("assignee_user") or "")
            }
        )
        dataset["assignees"] = assignees
        dataset["recording_count"] = len(recordings)
        dataset["active_session_count"] = sum(int(recording.get("active_session_count") or 0) for recording in recordings if isinstance(recording, Mapping))
        dataset["stale_session_count"] = sum(int(recording.get("stale_session_count") or 0) for recording in recordings if isinstance(recording, Mapping))
        dataset["failed_promotion_count"] = sum(int(recording.get("failed_promotion_count") or 0) for recording in recordings if isinstance(recording, Mapping))
        blocked = 0
        for recording in recordings:
            if not isinstance(recording, dict):
                continue
            total = int(recording.get("task_count") or 0)
            complete = int(recording.get("complete_task_count") or 0)
            open_tasks = int(recording.get("open_task_count") or 0)
            non_startable = int(recording.get("non_startable_task_count") or 0)
            recording["progress_fraction"] = (complete / total) if total else 0.0
            recording["progress_percent"] = round(100.0 * float(recording["progress_fraction"]), 1)
            recording["has_waiting_work"] = bool(open_tasks > 0)
            recording["blocked"] = bool(total == 0 or (open_tasks == 0 and complete < total and non_startable > 0))
            recording["blocked_reason"] = (
                "tasks_not_generated"
                if total == 0
                else "no_startable_tasks"
                if bool(recording["blocked"])
                else ""
            )
            if bool(recording["blocked"]):
                blocked += 1
            task_rows = recording.get("tasks") if isinstance(recording.get("tasks"), list) else []
            task_rows.sort(
                key=lambda task: (
                    str(task.get("workflow_kind") or "") if isinstance(task, Mapping) else "",
                    str(task.get("state") or "") if isinstance(task, Mapping) else "",
                    str(task.get("title") or "") if isinstance(task, Mapping) else "",
                )
            )
        total_tasks = int(dataset.get("task_count") or 0)
        complete_tasks = int(dataset.get("complete_task_count") or 0)
        dataset["progress_fraction"] = (complete_tasks / total_tasks) if total_tasks else 0.0
        dataset["progress_percent"] = round(100.0 * float(dataset["progress_fraction"]), 1)
        dataset["blocked_recording_count"] = blocked
        dataset["has_waiting_work"] = int(dataset.get("open_task_count") or 0) > 0

    dataset_list.sort(
        key=lambda row: (
            0 if bool(row.get("has_waiting_work")) else 1,
            str(row.get("dataset_label") or ""),
        )
    )
    if status_filter:
        def _matches_admin_dataset_status(row: Mapping[str, object]) -> bool:
            task_count = int(row.get("task_count") or 0)
            open_task_count = int(row.get("open_task_count") or 0)
            complete_task_count = int(row.get("complete_task_count") or 0)
            blocked_recording_count = int(row.get("blocked_recording_count") or 0)
            if status_filter == "waiting":
                return open_task_count > 0
            if status_filter == "blocked":
                return blocked_recording_count > 0
            if status_filter == "complete":
                return task_count > 0 and complete_task_count >= task_count
            return True

        dataset_list = [row for row in dataset_list if _matches_admin_dataset_status(row)]
    registry_lookup = _admin_registry_lookup(
        registry_path=_admin_registry_path_from_env(),
        dataset_ids=[
            str(row.get("dataset_id") or "")
            for row in dataset_list
            if str(row.get("dataset_id") or "")
        ],
        recording_ids=[
            str(recording.get("recording_id") or "")
            for dataset in dataset_list
            for recording in (
                dataset.get("recordings")
                if isinstance(dataset.get("recordings"), list)
                else []
            )
            if isinstance(recording, Mapping) and str(recording.get("recording_id") or "")
        ],
    )
    rows_by_dataset_id = (
        registry_lookup.get("rows_by_dataset_id")
        if isinstance(registry_lookup.get("rows_by_dataset_id"), Mapping)
        else {}
    )
    rows_by_recording_id = (
        registry_lookup.get("rows_by_recording_id")
        if isinstance(registry_lookup.get("rows_by_recording_id"), Mapping)
        else {}
    )
    for dataset in dataset_list:
        dataset_id = str(dataset.get("dataset_id") or "")
        dataset_registry_rows: list[Mapping[str, object]] = []
        if dataset_id:
            rows = rows_by_dataset_id.get(dataset_id, []) if isinstance(rows_by_dataset_id, Mapping) else []
            if isinstance(rows, list):
                dataset_registry_rows.extend(row for row in rows if isinstance(row, Mapping))
        recordings = dataset.get("recordings") if isinstance(dataset.get("recordings"), list) else []
        for recording in recordings:
            if not isinstance(recording, dict):
                continue
            recording_id = str(recording.get("recording_id") or "")
            recording_registry_rows = []
            rows = rows_by_recording_id.get(recording_id, []) if isinstance(rows_by_recording_id, Mapping) else []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    row_dataset_id = str(row.get("dataset_id") or "")
                    if dataset_id and row_dataset_id and row_dataset_id != dataset_id:
                        continue
                    recording_registry_rows.append(row)
                    dataset_registry_rows.append(row)
            deduped_recording_rows = {
                (
                    str(row.get("table") or ""),
                    str(row.get("dataset_id") or ""),
                    str(row.get("recording_id") or ""),
                    str(row.get("zarr_path") or ""),
                    str(row.get("zarr_use") or ""),
                    str(row.get("status") or ""),
                ): dict(row)
                for row in recording_registry_rows
            }
            compact_recording_rows = list(deduped_recording_rows.values())
            recording["registry_rows"] = compact_recording_rows[:25]
            recording["registry_summary"] = _admin_registry_summary(compact_recording_rows)
            recording["registry_warnings"] = _admin_registry_warnings_for_recording(recording)
        deduped_dataset_rows = {
            (
                str(row.get("table") or ""),
                str(row.get("dataset_id") or ""),
                str(row.get("recording_id") or ""),
                str(row.get("zarr_path") or ""),
                str(row.get("zarr_use") or ""),
                str(row.get("status") or ""),
            ): dict(row)
            for row in dataset_registry_rows
        }
        compact_dataset_rows = list(deduped_dataset_rows.values())
        dataset["registry_rows"] = compact_dataset_rows[:50]
        dataset["registry_summary"] = _admin_registry_summary(compact_dataset_rows)
        dataset_warnings = []
        for recording in recordings:
            if not isinstance(recording, Mapping):
                continue
            warning_rows = (
                recording.get("registry_warnings")
                if isinstance(recording.get("registry_warnings"), list)
                else []
            )
            dataset_warnings.extend(
                dict(warning)
                for warning in warning_rows
                if isinstance(warning, Mapping)
            )
        dataset["registry_warnings"] = dataset_warnings
        dataset["registry_warning_count"] = len(dataset_warnings)
    if warnings_only:
        dataset_list = [
            row
            for row in dataset_list
            if int(row.get("registry_warning_count") or 0) > 0
        ]
    open_task_count = sum(int(row.get("open_task_count") or 0) for row in dataset_list)
    complete_task_count = sum(int(row.get("complete_task_count") or 0) for row in dataset_list)
    task_count = sum(int(row.get("task_count") or 0) for row in dataset_list)
    blocked_recording_count = sum(int(row.get("blocked_recording_count") or 0) for row in dataset_list)
    top_level_warnings: list[dict[str, object]] = []
    if bool(registry_lookup.get("enabled")) and not bool(registry_lookup.get("available")):
        top_level_warnings.append(
            _admin_registry_warning(
                "registry_unavailable",
                details=str(registry_lookup.get("error") or "Registry could not be queried."),
                operator_action=(
                    f"Check {REGISTRY_PATH_ENV_VAR} and server filesystem access, or restart without registry enrichment."
                ),
                extra={"registry_path": str(registry_lookup.get("path") or "")},
            )
        )
    top_level_warnings.extend(
        dict(warning)
        for dataset in dataset_list
        for warning in (
            dataset.get("registry_warnings")
            if isinstance(dataset.get("registry_warnings"), list)
            else []
        )
        if isinstance(warning, Mapping)
    )
    return {
        "ok": True,
        "schema": "palette.web_labeling_admin_datasets.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "store_path": str(config.store_path),
        "registry": {
            "enabled": bool(registry_lookup.get("enabled")),
            "available": bool(registry_lookup.get("available")),
            "path": str(registry_lookup.get("path") or ""),
            "error": str(registry_lookup.get("error") or ""),
            "matched_row_count": int(registry_lookup.get("matched_row_count") or 0),
            "tables_scanned": list(registry_lookup.get("tables_scanned") or []),
        },
        "warnings": top_level_warnings,
        "warning_count": len(top_level_warnings),
        "filters": {
            "dataset_id": dataset_filter,
            "recording_id": recording_filter,
            "assignee_user": user_filter,
            "status": status_filter,
            "warnings_only": bool(warnings_only),
        },
        "counts": {
            "dataset_count": len(dataset_list),
            "recording_count": sum(int(row.get("recording_count") or 0) for row in dataset_list),
            "assignment_count": len(assignments),
            "active_assignment_count": sum(1 for row in assignments if str(row.get("status") or "") == "active"),
            "task_count": task_count,
            "open_task_count": open_task_count,
            "complete_task_count": complete_task_count,
            "non_startable_task_count": sum(int(row.get("non_startable_task_count") or 0) for row in dataset_list),
            "blocked_recording_count": blocked_recording_count,
            "active_session_count": len(active_sessions),
            "stale_session_count": sum(1 for session in active_sessions if _session_is_expired(session)),
            "failed_promotion_count": sum(failed_promotion_counts.values()),
            "progress_percent": round(100.0 * (complete_task_count / task_count), 1) if task_count else 0.0,
        },
        "datasets": dataset_list,
    }

def _redact_admin_recording_task(task: Mapping[str, object]) -> dict[str, object]:
    row = dict(task)
    existing_redacted = row.get("redacted_fields")
    redacted_fields = [str(field) for field in existing_redacted] if isinstance(existing_redacted, list) else []
    if "scope" in row:
        row.pop("scope", None)
        redacted_fields.append("scope")
    if redacted_fields:
        row["redacted_fields"] = sorted({field for field in redacted_fields if field})
    return row

def _admin_recording_session_summary(session: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "session_id",
        "task_id",
        "recording_id",
        "user",
        "workflow_kind",
        "assignment_status",
        "task_state",
        "created_at_utc",
        "expires_at_utc",
        "closed_at_utc",
        "title",
        "dataset_id",
        "component_name",
    )
    return {field: session.get(field) for field in fields if field in session}

def _admin_recording_event_summary(event: Mapping[str, object]) -> dict[str, object]:
    row = {
        "event_id": event.get("event_id"),
        "task_id": event.get("task_id"),
        "recording_id": event.get("recording_id"),
        "user": event.get("user"),
        "assignee_user": event.get("assignee_user"),
        "workflow_kind": event.get("workflow_kind"),
        "event_type": event.get("event_type"),
        "created_at_utc": event.get("created_at_utc"),
    }
    for field in ("target", "before", "after"):
        value = event.get(field)
        row[f"has_{field}"] = bool(value)
        row[f"{field}_keys"] = sorted(str(key) for key in value.keys()) if isinstance(value, Mapping) else []
    return row

def _count_recording_values(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))

def _admin_recording_payload(store: LabelingStore, *, recording_id: str) -> dict[str, object]:
    assignment = store.get_assignment(recording_id)
    tasks = [_redact_admin_recording_task(task) for task in store.list_tasks(recording_id=recording_id, include_completed=True)]
    sessions = [
        _admin_recording_session_summary(session)
        for session in store.list_sessions(include_closed=True, limit=500)
        if str(session.get("recording_id") or "") == recording_id
    ]
    active_sessions = [session for session in sessions if not session.get("closed_at_utc")]
    reassignment_mismatched_sessions = store.active_assignment_mismatched_sessions_for_recording(
        recording_id,
        limit=500,
    )
    events = [_admin_recording_event_summary(event) for event in store.list_events(recording_id=recording_id, limit=100)]
    owner = str(assignment.get("assignee_user") or "") if assignment else ""
    task_states = _count_recording_values(tasks, "state")
    workflow_counts = _count_recording_values(tasks, "workflow_kind")
    open_tasks = sum(1 for task in tasks if str(task.get("state") or "") in LABELER_START_TASK_STATES)
    non_startable_tasks = sum(
        1
        for task in tasks
        if str(task.get("state") or "") != "complete"
        and str(task.get("state") or "") not in LABELER_START_TASK_STATES
    )
    complete_tasks = sum(1 for task in tasks if str(task.get("state") or "") == "complete")
    return {
        "recording_id": recording_id,
        "assignment": assignment,
        "assignee_user": owner,
        "expected_user_labeler_landing_url": _dashboard_url_for_expected_user("/", owner) if owner else "",
        "expected_user_labeling_home_url": _dashboard_url_for_expected_user(LABELING_HOME_PATH, owner) if owner else "",
        "expected_user_dashboard_url": _dashboard_url_for_expected_user(DASHBOARD_PATH, owner) if owner else "",
        "expected_user_dataset_queue_url": _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, owner) if owner else "",
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "expected_user_personal_work_url": _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, owner) if owner else "",
        "expected_user_personal_dataset_queue_url": (
            _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, owner) if owner else ""
        ),
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": (
            _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, owner) if owner else ""
        ),
        "single_owner_policy": _assignment_ownership_policy(),
        "task_counts": {
            "total_tasks": len(tasks),
            "open_tasks": open_tasks,
            "non_startable_tasks": non_startable_tasks,
            "complete_tasks": complete_tasks,
            "task_states": task_states,
            "workflow_counts": workflow_counts,
        },
        "active_session_count": len(active_sessions),
        "reassignment_session_repair_route": (
            f"/api/admin/recordings/{quote(recording_id, safe='')}/repair-reassignment-sessions"
        ),
        "reassignment_session_safety_blocks_labeler_mutation": bool(
            reassignment_mismatched_sessions
        ),
        "reassignment_session_safety_active_session_assignment_mismatch_count": len(
            reassignment_mismatched_sessions
        ),
        "reassignment_session_safety_active_session_assignment_mismatch_session_ids": [
            str(session.get("session_id") or "")
            for session in reassignment_mismatched_sessions
        ],
        "reassignment_session_safety_active_session_assignment_mismatch_recording_ids": (
            [recording_id] if reassignment_mismatched_sessions else []
        ),
        "reassignment_session_safety_operator_action": (
            _reassignment_session_safety_operator_action()
            if reassignment_mismatched_sessions
            else ""
        ),
        "reassignment_session_safety_mismatched_sessions": reassignment_mismatched_sessions,
        "recent_session_count": len(sessions),
        "recent_event_count": len(events),
        "tasks": tasks,
        "active_sessions": active_sessions,
        "recent_sessions": sessions,
        "recent_events": events,
    }

def _admin_user_payload(store: LabelingStore, *, user: str) -> dict[str, object]:
    work = store.task_summary_for_user(user, include_completed=True)
    work["include_completed"] = True
    check_report = _store_consistency_report(store)
    _add_work_summary_fields(
        work,
        reassignment_session_safety=check_report.get("reassignment_session_safety", {}),
    )
    roster_rows = _dashboard_roster_rows(
        store,
        dashboard_url=DASHBOARD_PATH,
        user=user,
        include_completed=True,
        require_dashboard_url=False,
    )
    dashboard_row = roster_rows[0] if roster_rows else None
    labeler_safety = _labeler_safety_policy()
    queue_first_entry_contract = _queue_first_entry_contract_policy(
        labeler_safety=labeler_safety,
        labeler_landing_page_path="/",
        labeler_landing_url="/",
        expected_user_labeler_landing_url=_dashboard_url_for_expected_user("/", user),
        labeling_home_page_path=LABELING_HOME_PATH,
        labeling_home_url=LABELING_HOME_PATH,
        expected_user_labeling_home_url=_dashboard_url_for_expected_user(LABELING_HOME_PATH, user),
        dataset_queue_page_path=DATASET_QUEUE_PATH,
        dataset_queue_url=DATASET_QUEUE_PATH,
        expected_user_dataset_queue_url=_dashboard_url_for_expected_user(
            DATASET_QUEUE_PATH,
            user,
        ),
        dashboard_url=DASHBOARD_PATH,
        expected_user_dashboard_url=_dashboard_url_for_expected_user(DASHBOARD_PATH, user),
        personal_dataset_queue_page_path=PERSONAL_DATASET_QUEUE_PATH,
        personal_dataset_queue_url=PERSONAL_DATASET_QUEUE_PATH,
        expected_user_personal_dataset_queue_url=_dashboard_url_for_expected_user(
            PERSONAL_DATASET_QUEUE_PATH,
            user,
        ),
        personal_work_page_path=PERSONAL_WORK_PATH,
        personal_work_url=PERSONAL_WORK_PATH,
        expected_user_personal_work_url=_dashboard_url_for_expected_user(
            PERSONAL_WORK_PATH,
            user,
        ),
    )
    work["queue_first_entry_contract"] = queue_first_entry_contract
    return {
        "user": user,
        "labeling_user": store.get_labeling_user(user),
        "labeling_user_status": _known_labeler_status(store, user),
        "labeling_user_events": store.list_labeling_user_events(user_id=user, limit=50),
        "dashboard_path": DASHBOARD_PATH,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "labeler_landing_page_path": "/",
        "labeling_home_page_path": LABELING_HOME_PATH,
        "expected_user_labeler_landing_url": _dashboard_url_for_expected_user("/", user),
        "expected_user_labeling_home_url": _dashboard_url_for_expected_user(LABELING_HOME_PATH, user),
        "expected_user_dashboard_url": _dashboard_url_for_expected_user(DASHBOARD_PATH, user),
        "expected_user_personal_work_url": _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, user),
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "expected_user_dataset_queue_url": _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, user),
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "expected_user_personal_dataset_queue_url": _dashboard_url_for_expected_user(
            PERSONAL_DATASET_QUEUE_PATH,
            user,
        ),
        "expected_user_identity_probe_url": _dashboard_url_for_expected_user(IDENTITY_PROBE_PATH, user),
        "labeler_safety": labeler_safety,
        "queue_first_entry_contract": queue_first_entry_contract,
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        "dataset_queue_state": work.get("dataset_queue_state", {}),
        "reassignment_session_safety": work.get("reassignment_session_safety", {}),
        **_reassignment_session_safety_flat_fields(
            work.get("reassignment_session_safety")
            if isinstance(work.get("reassignment_session_safety"), Mapping)
            else {}
        ),
        "work": work,
        "dashboard_user": dashboard_row,
    }

def _admin_users_payload(store: LabelingStore, *, config: ServerConfig) -> dict[str, object]:
    users = store.list_labeling_users(status=None)
    assignments = store.list_assignments(status=None)
    tasks = store.list_tasks(include_completed=True)
    assignment_counts: dict[str, dict[str, int]] = {}
    for assignment in assignments:
        user = str(assignment.get("assignee_user") or "").strip()
        if not user:
            continue
        counts = assignment_counts.setdefault(user, {"total": 0, "active": 0, "inactive": 0})
        counts["total"] += 1
        status = str(assignment.get("status") or "inactive")
        counts["active" if status == "active" else "inactive"] += 1
    task_counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        user = str(task.get("assignee_user") or "").strip()
        if not user:
            continue
        counts = task_counts.setdefault(user, {"total": 0, "open": 0, "complete": 0})
        counts["total"] += 1
        state = str(task.get("state") or "")
        if state == "complete":
            counts["complete"] += 1
        elif state in LABELER_START_TASK_STATES:
            counts["open"] += 1
    rows: list[dict[str, object]] = []
    seen_users: set[str] = set()
    for user_row in users:
        user_id = str(user_row.get("user_id") or "").strip()
        if not user_id:
            continue
        seen_users.add(user_id)
        rows.append(
            {
                **user_row,
                "is_admin": _is_admin_user(user_id, config),
                "assignment_counts": assignment_counts.get(user_id, {"total": 0, "active": 0, "inactive": 0}),
                "task_counts": task_counts.get(user_id, {"total": 0, "open": 0, "complete": 0}),
                "admin_user_url": f"/admin/users/{quote(user_id, safe='')}",
            }
        )
    for user_id in sorted(set(assignment_counts) - seen_users):
        rows.append(
            {
                "user_id": user_id,
                "display_name": "",
                "email": "",
                "role": "",
                "status": "missing_registry_row",
                "created_at_utc": "",
                "updated_at_utc": "",
                "notes": "Assignment exists but no labeling_users row is present.",
                "is_admin": _is_admin_user(user_id, config),
                "assignment_counts": assignment_counts.get(user_id, {"total": 0, "active": 0, "inactive": 0}),
                "task_counts": task_counts.get(user_id, {"total": 0, "open": 0, "complete": 0}),
                "admin_user_url": f"/admin/users/{quote(user_id, safe='')}",
            }
        )
    rows.sort(
        key=lambda row: (
            0 if str(row.get("status") or "") == "active" else 1,
            str(row.get("role") or ""),
            str(row.get("user_id") or ""),
        )
    )
    return {
        "ok": True,
        "schema": "palette.web_labeling_admin_users.v1",
        "store_path": str(config.store_path),
        "user_table": "labeling_users",
        "event_table": "labeling_user_events",
        "roles": list(LABELING_USER_ROLES),
        "statuses": list(LABELING_USER_STATUSES),
        "configured_admin_users": list(config.admin_users),
        "count": len(rows),
        "active_count": sum(1 for row in rows if str(row.get("status") or "") == "active"),
        "inactive_count": sum(1 for row in rows if str(row.get("status") or "") == "inactive"),
        "missing_registry_row_count": sum(1 for row in rows if str(row.get("status") or "") == "missing_registry_row"),
        "users": rows,
        "recent_user_events": store.list_labeling_user_events(limit=50),
    }

def _store_consistency_report(store: LabelingStore) -> dict[str, object]:
    assignments = store.list_assignments(status=None)
    tasks = store.list_tasks(include_completed=True)
    active_sessions = store.list_sessions(include_closed=False, limit=10000)
    assignment_ownership_integrity = _assignment_ownership_integrity(
        assignments,
        schema_integrity=store.assignment_schema_integrity(),
    )
    now = datetime.now(timezone.utc).isoformat()
    issues: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for duplicate in assignment_ownership_integrity["duplicate_active_owners"]:
        issues.append(
            {
                "code": "duplicate_active_recording_owners",
                "recording_id": duplicate["recording_id"],
                "assignee_users": duplicate["assignee_users"],
                "blocks_labeler_mutation": True,
                "operator_action": (
                    "Repair recording ownership so exactly one active assignment remains before labelers can mutate this recording."
                ),
                "details": "Recording has more than one active owner; browser mutation must be blocked until ownership is repaired.",
            }
        )
    if not bool(assignment_ownership_integrity.get("recording_id_primary_key")) or not bool(
        assignment_ownership_integrity.get("schema_enforced_recording_primary_key")
    ):
        issues.append(
            {
                "code": "assignment_schema_primary_key_missing",
                "assignment_table": "recording_assignments",
                "primary_key_columns": (
                    assignment_ownership_integrity.get("primary_key_columns")
                    if isinstance(assignment_ownership_integrity.get("primary_key_columns"), list)
                    else []
                ),
                "blocks_labeler_mutation": True,
                "operator_action": (
                    "Repair or migrate the labeling store so recording_assignments has recording_id as its primary key before exposing labeler work."
                ),
                "details": "The assignment table does not prove one-row-per-recording ownership from the live store schema.",
            }
        )

    for task in tasks:
        task_id = str(task.get("task_id") or "")
        recording_id = str(task.get("recording_id") or "")
        assignee_user = str(task.get("assignee_user") or "")
        assignment_status = str(task.get("assignment_status") or "")
        state = str(task.get("state") or "")
        if not assignee_user:
            issues.append(
                {
                    "code": "task_missing_assignment",
                    "task_id": task_id,
                    "recording_id": recording_id,
                    "blocks_labeler_mutation": True,
                    "operator_action": (
                        "Assign the recording with `assign --recording-id "
                        f"{recording_id} --user USER --assigned-by OPERATOR` before exposing this task."
                    ),
                    "details": "Task recording has no recording assignment.",
                }
            )
            continue
        if assignment_status != "active" and state != "complete":
            warnings.append(
                {
                    "code": "incomplete_task_in_inactive_assignment",
                    "task_id": task_id,
                    "recording_id": recording_id,
                    "assignee_user": assignee_user,
                    "assignment_status": assignment_status,
                    "blocks_labeler_start": True,
                    "operator_action": (
                        "Reactivate, reassign, complete, or remove this task before expecting it in the labeler's queue."
                    ),
                    "details": "Incomplete task is under a non-active assignment and will not be available to the labeler.",
                }
            )

    for session in active_sessions:
        session_id = str(session.get("session_id") or "")
        task_id = str(session.get("task_id") or "")
        recording_id = str(session.get("recording_id") or "")
        session_user = str(session.get("user") or "")
        assignee_user = str(session.get("assignee_user") or "")
        assignment_status = str(session.get("assignment_status") or "")
        task_state = str(session.get("task_state") or "")
        expires_at_utc = str(session.get("expires_at_utc") or "")
        if assignment_status != "active" or assignee_user != session_user:
            issues.append(
                {
                    "code": "active_session_assignment_mismatch",
                    "session_id": session_id,
                    "task_id": task_id,
                    "recording_id": recording_id,
                    "session_user": session_user,
                    "assignee_user": assignee_user,
                    "assignment_status": assignment_status,
                    "blocks_labeler_mutation": True,
                    "operator_action": (
                        "Close the stale session or re-run `assign --recording-id "
                        f"{recording_id} --user {assignee_user or 'USER'} --assigned-by OPERATOR` "
                        "so previous-owner sessions are closed before the labeler reopens work."
                    ),
                    "details": "Active session no longer matches the current active recording assignment.",
                }
            )
        if task_state == "complete":
            issues.append(
                {
                    "code": "active_session_for_completed_task",
                    "session_id": session_id,
                    "task_id": task_id,
                    "recording_id": recording_id,
                    "blocks_labeler_mutation": True,
                    "operator_action": "Keep the completed task closed or explicitly reopen it through the operator task-state workflow.",
                    "details": "Completed task still has an active browser session.",
                }
            )
        if expires_at_utc and expires_at_utc < now:
            warnings.append(
                {
                    "code": "expired_active_session",
                    "session_id": session_id,
                    "task_id": task_id,
                    "recording_id": recording_id,
                    "expires_at_utc": expires_at_utc,
                    "operator_action": "Run `cleanup-stale-sessions --user OPERATOR` to close expired browser sessions.",
                    "details": "Expired session is still open; cleanup-stale-sessions can close it.",
                }
            )

    issue_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for issue in issues:
        code = str(issue.get("code") or "unknown")
        issue_counts[code] = issue_counts.get(code, 0) + 1
    for warning in warnings:
        code = str(warning.get("code") or "unknown")
        warning_counts[code] = warning_counts.get(code, 0) + 1
    reassignment_session_mismatch_issues = [
        issue
        for issue in issues
        if str(issue.get("code") or "") == "active_session_assignment_mismatch"
    ]
    reassignment_session_safety = {
        "schema": "palette.web_labeling_reassignment_session_safety.v1",
        "ok": not reassignment_session_mismatch_issues,
        "active_session_assignment_mismatch_count": len(reassignment_session_mismatch_issues),
        "active_session_assignment_mismatch_session_ids": [
            str(issue.get("session_id") or "")
            for issue in reassignment_session_mismatch_issues
            if str(issue.get("session_id") or "")
        ],
        "active_session_assignment_mismatch_recording_ids": sorted(
            {
                str(issue.get("recording_id") or "")
                for issue in reassignment_session_mismatch_issues
                if str(issue.get("recording_id") or "")
            }
        ),
        "blocks_labeler_mutation": bool(reassignment_session_mismatch_issues),
        "requires_operator_recovery": bool(reassignment_session_mismatch_issues),
        "operator_action": (
            "Close stale previous-owner sessions or re-run assignment through assign_recording_with_session_closure before exposing labeler work."
            if reassignment_session_mismatch_issues
            else ""
        ),
    }

    return {
        "ok": not issues,
        "single_owner_policy": _assignment_ownership_policy(),
        "assignment_ownership_integrity": assignment_ownership_integrity,
        "reassignment_session_safety": reassignment_session_safety,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issue_counts": dict(sorted(issue_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "operator_actions": [
            str(item.get("operator_action") or "")
            for item in [*issues, *warnings]
            if str(item.get("operator_action") or "").strip()
        ],
        "counts": {
            "assignments": len(assignments),
            "tasks": len(tasks),
            "active_sessions": len(active_sessions),
        },
        "issues": issues,
        "warnings": warnings,
    }

def _assignment_ownership_integrity(
    assignments: Sequence[Mapping[str, object]],
    *,
    schema_integrity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    schema_report = schema_integrity if isinstance(schema_integrity, Mapping) else {}
    recording_id_primary_key = bool(schema_report.get("recording_id_primary_key", True))
    schema_enforced_recording_primary_key = bool(
        schema_report.get("schema_enforced_recording_primary_key", recording_id_primary_key)
    )
    active_by_recording: dict[str, set[str]] = {}
    active_assignment_count = 0
    for assignment in assignments:
        if str(assignment.get("status") or "") != "active":
            continue
        recording_id = str(assignment.get("recording_id") or "").strip()
        assignee_user = str(assignment.get("assignee_user") or "").strip()
        if not recording_id:
            continue
        active_assignment_count += 1
        active_by_recording.setdefault(recording_id, set()).add(assignee_user)
    duplicate_active_owners = [
        {
            "recording_id": recording_id,
            "assignee_users": sorted(user for user in users if user),
            "active_owner_count": len([user for user in users if user]),
        }
        for recording_id, users in sorted(active_by_recording.items())
        if len([user for user in users if user]) > 1
    ]
    return {
        "ok": not duplicate_active_owners and recording_id_primary_key and schema_enforced_recording_primary_key,
        "policy": _assignment_ownership_policy(),
        "assignment_table": "recording_assignments",
        "schema_integrity": dict(schema_report) if schema_report else {},
        "schema_integrity_source": "store_pragma" if schema_report else "policy_default",
        "recording_id_primary_key": recording_id_primary_key,
        "schema_enforced_recording_primary_key": schema_enforced_recording_primary_key,
        "primary_key_columns": (
            schema_report.get("primary_key_columns")
            if isinstance(schema_report.get("primary_key_columns"), list)
            else ["recording_id"]
        ),
        "one_row_per_recording_enforced": bool(
            schema_report.get("one_row_per_recording_enforced", recording_id_primary_key)
        ),
        "active_assignment_count": active_assignment_count,
        "unique_active_recording_count": len(active_by_recording),
        "duplicate_active_owner_count": len(duplicate_active_owners),
        "duplicate_active_owners": duplicate_active_owners,
    }

def _zarr_backup_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_zarr_backup_policy.v1",
        "read_only_plan": True,
        "operator_only": True,
        "copy_before_labeling": True,
        "mutable_zarr_backup_required_before_invite": True,
        "validation_gate": "mutable_zarr_backup_confirmation",
        "pause_or_unassign_recording_before_restore": True,
        "sidecar_backup_command": "execute-zarr-backup-plan",
        "sidecar_restore_command": "restore-zarr-backup",
        "backup_execution_manifest_schema": "palette.web_labeling_zarr_backup_execution_manifest.v1",
        "labelers_do_not_edit_zarrs_directly": True,
        "labelers_do_not_receive_backup_paths": True,
        "restore_requires_registry_refresh_when_registry_visible_metadata_changes": True,
        "rollback_owner": "operator",
    }

def _queue_first_entry_contract_policy(
    *,
    labeler_safety: Mapping[str, object],
    labeler_landing_page_path: str,
    labeler_landing_url: str,
    expected_user_labeler_landing_url: str,
    dataset_queue_page_path: str,
    dataset_queue_url: str,
    expected_user_dataset_queue_url: str,
    dashboard_url: str,
    expected_user_dashboard_url: str,
    labeling_home_page_path: str = LABELING_HOME_PATH,
    labeling_home_url: str = LABELING_HOME_PATH,
    expected_user_labeling_home_url: str = "",
    personal_dataset_queue_page_path: str = "",
    personal_dataset_queue_url: str = "",
    expected_user_personal_dataset_queue_url: str = "",
    personal_work_page_path: str = "",
    personal_work_url: str = "",
    expected_user_personal_work_url: str = "",
) -> dict[str, object]:
    queue_first_landing_paths = [
        str(path)
        for path in (
            labeler_safety.get("queue_first_landing_paths")
            if isinstance(labeler_safety.get("queue_first_landing_paths"), list)
            else []
        )
        if str(path).strip()
    ]
    expected_user_guards = (
        labeler_safety.get("expected_user_guards")
        if isinstance(labeler_safety.get("expected_user_guards"), Mapping)
        else {}
    )
    landing_ready = str(labeler_landing_page_path or "").strip() == "/" and bool(
        str(labeler_landing_url or expected_user_labeler_landing_url or "").strip()
    )
    labeling_home_ready = str(labeling_home_page_path or "").strip() == LABELING_HOME_PATH and bool(
        str(labeling_home_url or expected_user_labeling_home_url or LABELING_HOME_PATH).strip()
    )
    dataset_queue_ready = str(dataset_queue_page_path or "").strip() == DATASET_QUEUE_PATH and bool(
        str(dataset_queue_url or expected_user_dataset_queue_url or "").strip()
    )
    personal_dataset_queue_ready = (
        str(personal_dataset_queue_page_path or "").strip() == PERSONAL_DATASET_QUEUE_PATH
        and bool(str(personal_dataset_queue_url or expected_user_personal_dataset_queue_url or "").strip())
    )
    personal_work_ready = (
        str(personal_work_page_path or "").strip() == PERSONAL_WORK_PATH
        and bool(str(personal_work_url or expected_user_personal_work_url or "").strip())
    )
    dashboard_ready = bool(str(dashboard_url or expected_user_dashboard_url or "").strip())
    queue_first_paths_ready = all(
        path in queue_first_landing_paths
        for path in ("/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH)
    )
    datasets_waiting_alias_paths = [
        str(path)
        for path in (
            labeler_safety.get("datasets_waiting_alias_paths")
            if isinstance(labeler_safety.get("datasets_waiting_alias_paths"), list)
            else []
        )
        if str(path).strip()
    ]
    datasets_waiting_aliases_ready = all(
        path in datasets_waiting_alias_paths
        for path in ("/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH)
    )
    labeler_landing_page_kind = str(labeler_safety.get("labeler_landing_page_kind") or "")
    landing_serves_datasets_waiting_queue = bool(
        labeler_safety.get("landing_serves_datasets_waiting_queue")
    )
    dashboard_is_fallback = bool(labeler_safety.get("dashboard_is_fallback"))
    identity_check_required = bool(labeler_safety.get("dashboard_identity_check_required"))
    expected_user_landing_guard = str(expected_user_guards.get("labeler_landing_page") or "") == "dashboard_user_mismatch"
    expected_user_queue_guard = str(expected_user_guards.get("dataset_queue_page") or "") == "dashboard_user_mismatch"
    expected_user_dashboard_guard = str(expected_user_guards.get("dashboard") or "") == "dashboard_user_mismatch"
    dataset_queue_entry_url = str(expected_user_dataset_queue_url or dataset_queue_url or "").strip()
    personal_dataset_queue_entry_url = str(
        expected_user_personal_dataset_queue_url or personal_dataset_queue_url or ""
    ).strip()
    labeler_landing_entry_url = str(
        expected_user_labeler_landing_url or labeler_landing_url or ""
    ).strip()
    dashboard_entry_url = str(expected_user_dashboard_url or dashboard_url or "").strip()
    preferred_labeler_entrypoint = (
        "personal_datasets_waiting_queue"
        if personal_dataset_queue_entry_url
        else "datasets_waiting_queue"
    )
    preferred_labeler_entry_url = (
        personal_dataset_queue_entry_url
        or dataset_queue_entry_url
        or labeler_landing_entry_url
        or dashboard_entry_url
    )
    expected_user_personal_dataset_queue_url_normalized = str(
        expected_user_personal_dataset_queue_url or ""
    ).strip()
    personal_dataset_queue_url_normalized = str(personal_dataset_queue_url or "").strip()
    personalized_labeler_entrypoint = "personal_datasets_waiting_queue"
    personalized_labeler_entry_url = (
        expected_user_personal_dataset_queue_url_normalized
        or personal_dataset_queue_url_normalized
        or dataset_queue_entry_url
        or labeler_landing_entry_url
        or dashboard_entry_url
    )
    expected_user_dataset_queue_url_normalized = str(expected_user_dataset_queue_url or "").strip()
    personalized_entry_required = bool(
        expected_user_dataset_queue_url_normalized
        or expected_user_personal_dataset_queue_url_normalized
    )
    preferred_labeler_entry_url_matches_dataset_queue = bool(
        personal_dataset_queue_entry_url or dataset_queue_entry_url
    ) and preferred_labeler_entry_url in {
        personal_dataset_queue_entry_url,
        dataset_queue_entry_url,
    }
    preferred_labeler_entry_url_matches_personal_dataset_queue = bool(
        personal_dataset_queue_entry_url
    ) and preferred_labeler_entry_url == personal_dataset_queue_entry_url
    preferred_expected_user_queue_url = (
        expected_user_personal_dataset_queue_url_normalized
        or expected_user_dataset_queue_url_normalized
    )
    preferred_labeler_entry_url_is_expected_user_guarded = bool(
        preferred_expected_user_queue_url
    ) and (preferred_labeler_entry_url == preferred_expected_user_queue_url)
    personalized_labeler_entry_url_matches_personal_dataset_queue = bool(
        expected_user_personal_dataset_queue_url_normalized
        or personal_dataset_queue_url_normalized
    ) and personalized_labeler_entry_url in {
        expected_user_personal_dataset_queue_url_normalized,
        personal_dataset_queue_url_normalized,
    }
    personalized_expected_user_queue_url = (
        expected_user_personal_dataset_queue_url_normalized
        or expected_user_dataset_queue_url_normalized
    )
    personalized_labeler_entry_url_is_expected_user_guarded = bool(
        personalized_expected_user_queue_url
    ) and (
        personalized_labeler_entry_url == personalized_expected_user_queue_url
    )
    ready = (
        landing_ready
        and labeling_home_ready
        and dataset_queue_ready
        and personal_dataset_queue_ready
        and personal_work_ready
        and dashboard_ready
        and queue_first_paths_ready
        and datasets_waiting_aliases_ready
        and labeler_landing_page_kind == "datasets_waiting_queue"
        and landing_serves_datasets_waiting_queue
        and dashboard_is_fallback
        and identity_check_required
        and expected_user_landing_guard
        and expected_user_queue_guard
        and expected_user_dashboard_guard
        and preferred_labeler_entry_url_matches_dataset_queue
        and preferred_labeler_entry_url_matches_personal_dataset_queue
        and (
            not personalized_entry_required
            or preferred_labeler_entry_url_is_expected_user_guarded
        )
        and (
            not personalized_entry_required
            or (
                personal_dataset_queue_ready
                and personal_work_ready
                and personalized_labeler_entry_url_matches_personal_dataset_queue
                and personalized_labeler_entry_url_is_expected_user_guarded
            )
        )
    )
    return {
        "schema": "palette.web_labeling_queue_first_entry_contract.v1",
        "ready": ready,
        "labeler_landing_page_path": str(labeler_landing_page_path or ""),
        "labeler_landing_url": str(labeler_landing_url or ""),
        "expected_user_labeler_landing_url": str(expected_user_labeler_landing_url or ""),
        "labeling_home_page_path": str(labeling_home_page_path or ""),
        "labeling_home_url": str(labeling_home_url or ""),
        "expected_user_labeling_home_url": str(expected_user_labeling_home_url or ""),
        "dataset_queue_page_path": str(dataset_queue_page_path or ""),
        "dataset_queue_url": str(dataset_queue_url or ""),
        "expected_user_dataset_queue_url": str(expected_user_dataset_queue_url or ""),
        "personal_dataset_queue_page_path": str(personal_dataset_queue_page_path or ""),
        "personal_dataset_queue_url": str(personal_dataset_queue_url or ""),
        "expected_user_personal_dataset_queue_url": str(
            expected_user_personal_dataset_queue_url or ""
        ),
        "personal_work_page_path": str(personal_work_page_path or ""),
        "personal_work_url": str(personal_work_url or ""),
        "expected_user_personal_work_url": str(expected_user_personal_work_url or ""),
        "dashboard_url": str(dashboard_url or ""),
        "expected_user_dashboard_url": str(expected_user_dashboard_url or ""),
        "labeler_landing_page_kind": labeler_landing_page_kind,
        "landing_serves_datasets_waiting_queue": landing_serves_datasets_waiting_queue,
        "datasets_waiting_alias_paths": datasets_waiting_alias_paths,
        "datasets_waiting_aliases_ready": datasets_waiting_aliases_ready,
        "dashboard_is_fallback": dashboard_is_fallback,
        "preferred_labeler_entrypoint": preferred_labeler_entrypoint,
        "preferred_labeler_entry_url": preferred_labeler_entry_url,
        "personalized_labeler_entrypoint": personalized_labeler_entrypoint,
        "personalized_labeler_entry_url": personalized_labeler_entry_url,
        "personalized_entry_required": personalized_entry_required,
        "personal_dataset_queue_ready": personal_dataset_queue_ready,
        "personal_work_ready": personal_work_ready,
        "personalized_labeler_entry_url_matches_personal_dataset_queue": (
            personalized_labeler_entry_url_matches_personal_dataset_queue
        ),
        "personalized_labeler_entry_url_is_expected_user_guarded": (
            personalized_labeler_entry_url_is_expected_user_guarded
        ),
        "preferred_labeler_entry_url_matches_dataset_queue": (
            preferred_labeler_entry_url_matches_dataset_queue
        ),
        "preferred_labeler_entry_url_matches_personal_dataset_queue": (
            preferred_labeler_entry_url_matches_personal_dataset_queue
        ),
        "preferred_labeler_entry_url_is_expected_user_guarded": (
            preferred_labeler_entry_url_is_expected_user_guarded
        ),
        "labeler_landing_link_role": "queue_first_start",
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "dashboard_link_role": "fallback_dashboard",
        "identity_probe_link_role": "identity_check",
        "task_links_role": "convenience_entry_hints",
        "queue_first_landing_paths": queue_first_landing_paths,
        "landing_ready": landing_ready,
        "labeling_home_ready": labeling_home_ready,
        "dataset_queue_ready": dataset_queue_ready,
        "dashboard_ready": dashboard_ready,
        "queue_first_paths_ready": queue_first_paths_ready,
        "identity_check_required": identity_check_required,
        "expected_user_landing_guard": expected_user_landing_guard,
        "expected_user_queue_guard": expected_user_queue_guard,
        "expected_user_dashboard_guard": expected_user_dashboard_guard,
    }

def _labeler_route_authorization_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_labeler_route_authorization_policy.v1",
        "resolved_browser_user_required": True,
        "known_assignment_store_user_required": True,
        "expected_user_must_match_resolved_user": True,
        "personal_work_reads_filtered_by_resolved_user": True,
        "dataset_queue_reads_filtered_by_resolved_user": True,
        "personal_work_page_expected_user_guarded": True,
        "personal_dataset_queue_page_expected_user_guarded": True,
        "personal_aliases_route_to_canonical_browser_surfaces": True,
        "task_open_requires_active_assignment": True,
        "task_open_requires_task_assigned_to_resolved_user": True,
        "task_open_rejects_completed_tasks": True,
        "task_open_requires_startable_task_state": True,
        "startable_task_states": list(LABELER_START_TASK_STATES),
        "mutation_requires_current_session": True,
        "mutation_requires_session_owned_by_resolved_user": True,
        "mutation_requires_task_still_open": True,
        "mutation_requires_active_assignment": True,
        "mutation_requires_task_assigned_to_resolved_user": True,
        "mutation_requires_browser_supported_workflow": True,
        "mutation_requires_current_target_token": True,
        "mutation_rejects_client_target_selectors": True,
        "signed_links_are_entry_hints_not_authorization": True,
        "forwarded_expected_user_links_recheck_identity": True,
        "forwarded_signed_links_recheck_runtime_operator_validation_start_gate": True,
        "session_closure_events_reported_to_labeler": True,
        "single_owner_store_proof_required_for_browser_work": True,
        "single_owner_store_proof_requires_integrity_ok": True,
        "single_owner_store_proof_requires_zero_duplicate_active_owners": True,
        "single_owner_store_proof_requires_training_zarr_target": True,
        "single_owner_store_proof_rejects_intermediate_csv_mutation": True,
    }

def _labeler_route_authorization_runtime_checklist(
    *,
    policy: Mapping[str, object],
    user: str | None,
    expected_user: str | None,
    known_user_status: Mapping[str, object],
    assignment_ownership_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_user = str(user or "").strip()
    expected = str(expected_user or "").strip()
    expected_user_matches = not expected or resolved_user == expected
    known_user = bool(known_user_status.get("is_known_labeler"))
    resolved_user_required = bool(policy.get("resolved_browser_user_required"))
    known_user_required = bool(policy.get("known_assignment_store_user_required"))
    active_assignment_count = int(known_user_status.get("active_assignment_count") or 0)
    has_active_assignment = bool(known_user_status.get("has_active_assignment")) or (
        active_assignment_count > 0
    )
    active_assignment_required = bool(policy.get("task_open_requires_active_assignment"))
    expected_user_required = bool(policy.get("expected_user_must_match_resolved_user"))
    read_filters_ready = bool(policy.get("personal_work_reads_filtered_by_resolved_user")) and bool(
        policy.get("dataset_queue_reads_filtered_by_resolved_user")
    )
    task_open_ready = (
        bool(policy.get("task_open_requires_active_assignment"))
        and bool(policy.get("task_open_requires_task_assigned_to_resolved_user"))
        and bool(policy.get("task_open_rejects_completed_tasks"))
        and bool(policy.get("task_open_requires_startable_task_state"))
        and [str(state) for state in policy.get("startable_task_states") or []]
        == list(LABELER_START_TASK_STATES)
    )
    mutation_ready = (
        bool(policy.get("mutation_requires_current_session"))
        and bool(policy.get("mutation_requires_session_owned_by_resolved_user"))
        and bool(policy.get("mutation_requires_task_still_open"))
        and bool(policy.get("mutation_requires_active_assignment"))
        and bool(policy.get("mutation_requires_task_assigned_to_resolved_user"))
        and bool(policy.get("mutation_requires_browser_supported_workflow"))
        and bool(policy.get("mutation_requires_current_target_token"))
        and bool(policy.get("mutation_rejects_client_target_selectors"))
    )
    signed_link_ready = (
        bool(policy.get("signed_links_are_entry_hints_not_authorization"))
        and bool(policy.get("forwarded_expected_user_links_recheck_identity"))
        and bool(policy.get("forwarded_signed_links_recheck_runtime_operator_validation_start_gate"))
    )
    session_closure_ready = bool(policy.get("session_closure_events_reported_to_labeler"))
    ownership_contract = (
        assignment_ownership_contract
        if isinstance(assignment_ownership_contract, Mapping)
        else {}
    )
    single_owner_store_contract_required = bool(
        policy.get(
            "single_owner_store_proof_required_for_browser_work",
            bool(ownership_contract),
        )
    )
    single_owner_store_contract_present = bool(
        ownership_contract.get("store_single_owner_assignment_contract_present")
    )
    single_owner_store_contract_ready = bool(
        ownership_contract.get("store_single_owner_assignment_contract_ready")
    )
    single_owner_store_contract_met = bool(
        ownership_contract.get("store_single_owner_assignment_contract_met")
    )
    assignment_ownership_integrity_ok = bool(
        ownership_contract.get("assignment_ownership_integrity_ok", True)
    )
    duplicate_active_owner_count = int(
        ownership_contract.get("duplicate_active_owner_count") or 0
    )
    single_owner_store_proof_ready = (
        not single_owner_store_contract_required
        or (
            single_owner_store_contract_present
            and single_owner_store_contract_ready
            and single_owner_store_contract_met
            and assignment_ownership_integrity_ok
            and duplicate_active_owner_count == 0
            and bool(ownership_contract.get("browser_mutation_target_resolved_server_side"))
            and bool(ownership_contract.get("labelers_mutate_assigned_training_zarrs"))
            and ownership_contract.get("labelers_mutate_intermediate_csvs") is False
        )
    )
    ready = (
        (not resolved_user_required or bool(resolved_user))
        and (not known_user_required or known_user)
        and (not active_assignment_required or has_active_assignment)
        and (not expected_user_required or expected_user_matches)
        and read_filters_ready
        and task_open_ready
        and mutation_ready
        and signed_link_ready
        and session_closure_ready
        and single_owner_store_proof_ready
    )
    return {
        "schema": "palette.web_labeling_labeler_route_authorization_runtime_checklist.v1",
        "ready": ready,
        "resolved_user": resolved_user,
        "resolved_browser_user_required": resolved_user_required,
        "resolved_browser_user_present": bool(resolved_user),
        "expected_user": expected,
        "expected_user_present": bool(expected),
        "expected_user_must_match_resolved_user": expected_user_required,
        "expected_user_matches_resolved_user": expected_user_matches,
        "known_assignment_store_user_required": known_user_required,
        "known_assignment_store_user": known_user,
        "active_assignment_required": active_assignment_required,
        "active_assignment_count": active_assignment_count,
        "has_active_assignment": has_active_assignment,
        "personal_work_reads_filtered_by_resolved_user": bool(
            policy.get("personal_work_reads_filtered_by_resolved_user")
        ),
        "dataset_queue_reads_filtered_by_resolved_user": bool(
            policy.get("dataset_queue_reads_filtered_by_resolved_user")
        ),
        "task_open_requires_active_assignment": bool(policy.get("task_open_requires_active_assignment")),
        "task_open_requires_task_assigned_to_resolved_user": bool(
            policy.get("task_open_requires_task_assigned_to_resolved_user")
        ),
        "task_open_rejects_completed_tasks": bool(policy.get("task_open_rejects_completed_tasks")),
        "task_open_requires_startable_task_state": bool(
            policy.get("task_open_requires_startable_task_state")
        ),
        "startable_task_states": [str(state) for state in policy.get("startable_task_states") or []],
        "mutation_requires_current_session": bool(policy.get("mutation_requires_current_session")),
        "mutation_requires_session_owned_by_resolved_user": bool(
            policy.get("mutation_requires_session_owned_by_resolved_user")
        ),
        "mutation_requires_task_still_open": bool(policy.get("mutation_requires_task_still_open")),
        "mutation_requires_active_assignment": bool(policy.get("mutation_requires_active_assignment")),
        "mutation_requires_task_assigned_to_resolved_user": bool(
            policy.get("mutation_requires_task_assigned_to_resolved_user")
        ),
        "mutation_requires_browser_supported_workflow": bool(
            policy.get("mutation_requires_browser_supported_workflow")
        ),
        "mutation_requires_current_target_token": bool(policy.get("mutation_requires_current_target_token")),
        "mutation_rejects_client_target_selectors": bool(
            policy.get("mutation_rejects_client_target_selectors")
        ),
        "signed_links_are_entry_hints_not_authorization": bool(
            policy.get("signed_links_are_entry_hints_not_authorization")
        ),
        "forwarded_expected_user_links_recheck_identity": bool(
            policy.get("forwarded_expected_user_links_recheck_identity")
        ),
        "forwarded_signed_links_recheck_runtime_operator_validation_start_gate": bool(
            policy.get("forwarded_signed_links_recheck_runtime_operator_validation_start_gate")
        ),
        "session_closure_events_reported_to_labeler": session_closure_ready,
        "single_owner_store_contract_required": single_owner_store_contract_required,
        "single_owner_store_contract_present": single_owner_store_contract_present,
        "single_owner_store_contract_ready": single_owner_store_contract_ready,
        "single_owner_store_contract_met": single_owner_store_contract_met,
        "assignment_ownership_integrity_ok": assignment_ownership_integrity_ok,
        "duplicate_active_owner_count": duplicate_active_owner_count,
        "single_owner_store_proof_ready": single_owner_store_proof_ready,
        "browser_mutation_target_resolved_server_side": bool(
            ownership_contract.get("browser_mutation_target_resolved_server_side")
        ),
        "labelers_mutate_assigned_training_zarrs": bool(
            ownership_contract.get("labelers_mutate_assigned_training_zarrs")
        ),
        "labelers_mutate_intermediate_csvs": bool(
            ownership_contract.get("labelers_mutate_intermediate_csvs")
        ),
        "labeler_visible_scope": "assigned_recordings_for_resolved_user",
        "data_plane_mutation_scope": "current_guarded_session_task_target",
    }

def _assignment_ownership_policy() -> dict[str, object]:
    return dict(ASSIGNMENT_OWNERSHIP_POLICY)

def _single_owner_assignment_live_contract_fields(
    store: LabelingStore,
    *,
    integrity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    policy = _assignment_ownership_policy()
    single_owner_assignment_contract = store.single_owner_assignment_contract()
    integrity_source = (
        integrity
        if isinstance(integrity, Mapping)
        else {
            "ok": bool(single_owner_assignment_contract.get("ready")),
            "recording_id_primary_key": bool(
                single_owner_assignment_contract.get("recording_id_primary_key")
            ),
            "schema_enforced_recording_primary_key": bool(
                single_owner_assignment_contract.get(
                    "schema_enforced_recording_primary_key"
                )
            ),
            "primary_key_columns": (
                single_owner_assignment_contract.get("primary_key_columns")
                if isinstance(
                    single_owner_assignment_contract.get("primary_key_columns"),
                    list,
                )
                else []
            ),
            "schema_integrity_source": "LabelingStore.single_owner_assignment_contract",
            "duplicate_active_owner_count": 0,
        }
    )
    assignment_ownership_contract = _assignment_ownership_contract_policy(
        policy,
        integrity_source,
        store_single_owner_contract=single_owner_assignment_contract,
    )
    return {
        "assignment_ownership_integrity": dict(integrity_source),
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_contract": assignment_ownership_contract,
        **_assignment_ownership_contract_fields(assignment_ownership_contract),
        "single_owner_policy_contract_met": bool(
            assignment_ownership_contract.get("ready")
        )
        and int(integrity_source.get("duplicate_active_owner_count") or 0) == 0,
    }

def _single_owner_policy_fields(
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = policy if isinstance(policy, Mapping) else _assignment_ownership_policy()
    return {
        "single_owner_policy_assignment_scope": str(source.get("assignment_scope") or ""),
        "single_owner_policy_recording_assignment_key": str(
            source.get("recording_assignment_key") or ""
        ),
        "single_owner_policy_one_current_assignment_row_per_recording": bool(
            source.get("one_current_assignment_row_per_recording")
        ),
        "single_owner_policy_one_active_owner": bool(source.get("one_active_owner")),
        "single_owner_policy_multiple_labelers_per_recording_allowed": bool(
            source.get("multiple_labelers_per_recording_allowed")
        ),
        "single_owner_policy_assignment_manifests_are_control_plane": bool(
            source.get("assignment_manifests_are_control_plane")
        ),
        "single_owner_policy_duplicate_manifest_rows_do_not_create_multiple_owners": bool(
            source.get("duplicate_manifest_rows_do_not_create_multiple_owners")
        ),
        "single_owner_policy_assignment_user_match_required_for_mutation": bool(
            source.get("assignment_user_match_required_for_mutation")
        ),
        "single_owner_policy_browser_mutation_requires_current_assignment_owner": bool(
            source.get("browser_mutation_requires_current_assignment_owner")
        ),
        "single_owner_policy_browser_mutation_target_resolved_server_side": bool(
            source.get("browser_mutation_target_resolved_server_side", True)
        ),
        "single_owner_policy_browser_mutation_target_source": str(
            source.get("browser_mutation_target_source")
            or "recording_assignments.active_assignment"
        ),
        "single_owner_policy_labelers_mutate_assigned_training_zarrs": bool(
            source.get("labelers_mutate_assigned_training_zarrs", True)
        ),
        "single_owner_policy_labelers_mutate_intermediate_csvs": bool(
            source.get("labelers_mutate_intermediate_csvs", False)
        ),
    }

def _assignment_ownership_contract_fields(
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = contract if isinstance(contract, Mapping) else {}
    primary_key_columns = source.get("primary_key_columns")
    return {
        "assignment_ownership_contract_schema": str(source.get("schema") or ""),
        "assignment_ownership_contract_ready": bool(source.get("ready")),
        "assignment_ownership_contract_assignment_scope": str(
            source.get("assignment_scope") or ""
        ),
        "assignment_ownership_contract_recording_assignment_key": str(
            source.get("recording_assignment_key") or ""
        ),
        "assignment_ownership_contract_recording_id_primary_key": bool(
            source.get("recording_id_primary_key")
        ),
        "assignment_ownership_contract_schema_enforced_recording_primary_key": bool(
            source.get("schema_enforced_recording_primary_key")
        ),
        "assignment_ownership_contract_store_recording_id_primary_key": bool(
            source.get("store_recording_id_primary_key")
        ),
        "assignment_ownership_contract_store_schema_enforced_recording_primary_key": bool(
            source.get("store_schema_enforced_recording_primary_key")
        ),
        "assignment_ownership_contract_schema_integrity_source": str(
            source.get("schema_integrity_source") or ""
        ),
        "assignment_ownership_contract_primary_key_columns": json.dumps(
            primary_key_columns if isinstance(primary_key_columns, list) else [],
            sort_keys=True,
        ),
        "assignment_ownership_contract_one_current_assignment_row_per_recording": bool(
            source.get("one_current_assignment_row_per_recording")
        ),
        "assignment_ownership_contract_one_active_owner": bool(
            source.get("one_active_owner")
        ),
        "assignment_ownership_contract_multiple_labelers_per_recording_allowed": bool(
            source.get("multiple_labelers_per_recording_allowed")
        ),
        "assignment_ownership_contract_reassignment_replaces_owner": bool(
            source.get("reassignment_replaces_owner")
        ),
        "assignment_ownership_contract_stale_sessions_closed_on_reassignment": bool(
            source.get("stale_sessions_closed_on_reassignment")
        ),
        "assignment_ownership_contract_stale_sessions_closed_before_assignment_update": bool(
            source.get("stale_sessions_closed_before_assignment_update")
        ),
        "assignment_ownership_contract_reassignment_target_validated_before_session_closure": bool(
            source.get("reassignment_target_validated_before_session_closure")
        ),
        "assignment_ownership_contract_session_closure_and_assignment_update_atomic": bool(
            source.get("session_closure_and_assignment_update_atomic")
        ),
        "assignment_ownership_contract_raw_assignment_change_blocks_open_sessions": bool(
            source.get("raw_assignment_change_blocks_open_sessions")
        ),
        "assignment_ownership_contract_assignment_manifests_are_control_plane": bool(
            source.get("assignment_manifests_are_control_plane")
        ),
        "assignment_ownership_contract_duplicate_manifest_rows_do_not_create_multiple_owners": bool(
            source.get("duplicate_manifest_rows_do_not_create_multiple_owners")
        ),
        "assignment_ownership_contract_assignment_user_match_required_for_mutation": bool(
            source.get("assignment_user_match_required_for_mutation")
        ),
        "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner": bool(
            source.get("browser_mutation_requires_current_assignment_owner")
        ),
        "assignment_ownership_contract_browser_mutation_target_resolved_server_side": bool(
            source.get("browser_mutation_target_resolved_server_side")
        ),
        "assignment_ownership_contract_browser_mutation_target_source": str(
            source.get("browser_mutation_target_source") or ""
        ),
        "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs": bool(
            source.get("labelers_mutate_assigned_training_zarrs")
        ),
        "assignment_ownership_contract_labelers_mutate_intermediate_csvs": bool(
            source.get("labelers_mutate_intermediate_csvs")
        ),
        "assignment_ownership_contract_store_single_owner_assignment_contract_present": bool(
            source.get("store_single_owner_assignment_contract_present")
        ),
        "assignment_ownership_contract_store_single_owner_assignment_contract_ready": bool(
            source.get("store_single_owner_assignment_contract_ready")
        ),
        "assignment_ownership_contract_store_single_owner_assignment_contract_met": bool(
            source.get("store_single_owner_assignment_contract_met")
        ),
        "assignment_ownership_contract_store_single_owner_assignment_contract_schema": str(
            source.get("store_single_owner_assignment_contract_schema") or ""
        ),
        "assignment_ownership_contract_assignment_ownership_integrity_ok": bool(
            source.get("assignment_ownership_integrity_ok")
        ),
        "assignment_ownership_contract_active_assignment_count": int(
            source.get("active_assignment_count") or 0
        ),
        "assignment_ownership_contract_unique_active_recording_count": int(
            source.get("unique_active_recording_count") or 0
        ),
        "assignment_ownership_contract_duplicate_active_owner_count": int(
            source.get("duplicate_active_owner_count") or 0
        ),
    }

def _assignment_ownership_contract_policy(
    policy: Mapping[str, object],
    integrity: Mapping[str, object],
    *,
    store_single_owner_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    assignment_scope = str(policy.get("assignment_scope") or "")
    recording_assignment_key = str(policy.get("recording_assignment_key") or "")
    recording_id_primary_key = bool(policy.get("recording_id_primary_key"))
    schema_enforced_recording_primary_key = bool(policy.get("schema_enforced_recording_primary_key"))
    one_current_assignment_row_per_recording = bool(
        policy.get("one_current_assignment_row_per_recording")
    )
    one_active_owner = bool(policy.get("one_active_owner"))
    multiple_labelers_per_recording_allowed = bool(
        policy.get("multiple_labelers_per_recording_allowed")
    )
    reassignment_replaces_owner = bool(policy.get("reassignment_replaces_owner"))
    stale_sessions_closed_on_reassignment = bool(policy.get("stale_sessions_closed_on_reassignment"))
    stale_sessions_closed_before_assignment_update = bool(
        policy.get(
            "stale_sessions_closed_before_assignment_update",
            stale_sessions_closed_on_reassignment,
        )
    )
    reassignment_target_validated_before_session_closure = bool(
        policy.get("reassignment_target_validated_before_session_closure")
    )
    session_closure_and_assignment_update_atomic = bool(
        policy.get("session_closure_and_assignment_update_atomic")
    )
    raw_assignment_change_blocks_open_sessions = bool(policy.get("raw_assignment_change_blocks_open_sessions"))
    assignment_manifests_are_control_plane = bool(
        policy.get("assignment_manifests_are_control_plane")
    )
    duplicate_manifest_rows_do_not_create_multiple_owners = bool(
        policy.get("duplicate_manifest_rows_do_not_create_multiple_owners")
    )
    assignment_user_match_required_for_mutation = bool(policy.get("assignment_user_match_required_for_mutation"))
    browser_mutation_requires_current_assignment_owner = bool(
        policy.get("browser_mutation_requires_current_assignment_owner")
    )
    browser_mutation_target_resolved_server_side = bool(
        policy.get("browser_mutation_target_resolved_server_side", True)
    )
    browser_mutation_target_source = str(
        policy.get("browser_mutation_target_source")
        or "recording_assignments.active_assignment"
    )
    labelers_mutate_assigned_training_zarrs = bool(
        policy.get("labelers_mutate_assigned_training_zarrs", True)
    )
    labelers_mutate_intermediate_csvs = bool(
        policy.get("labelers_mutate_intermediate_csvs", False)
    )
    store_contract = (
        store_single_owner_contract
        if isinstance(store_single_owner_contract, Mapping)
        else {}
    )
    store_single_owner_assignment_contract_present = bool(store_contract)
    store_single_owner_assignment_contract_ready = bool(store_contract.get("ready"))
    store_single_owner_assignment_contract_met = bool(
        store_contract.get("single_owner_assignment_contract_met")
    )
    store_single_owner_assignment_contract_schema = str(store_contract.get("schema") or "")
    store_single_owner_contract_allows_ready = (
        not store_single_owner_assignment_contract_present
        or (
            store_single_owner_assignment_contract_ready
            and store_single_owner_assignment_contract_met
            and bool(store_contract.get("browser_mutation_target_resolved_server_side"))
            and str(store_contract.get("browser_mutation_target_source") or "")
            == "recording_assignments.active_assignment"
            and bool(store_contract.get("labelers_mutate_assigned_training_zarrs"))
            and store_contract.get("labelers_mutate_intermediate_csvs") is False
        )
    )
    integrity_recording_id_primary_key = bool(integrity.get("recording_id_primary_key"))
    integrity_schema_enforced_recording_primary_key = bool(
        integrity.get("schema_enforced_recording_primary_key")
    )
    integrity_ok = bool(integrity.get("ok", True))
    duplicate_active_owner_count = int(integrity.get("duplicate_active_owner_count") or 0)
    ready = (
        assignment_scope == "recording"
        and recording_assignment_key == "recording_id"
        and recording_id_primary_key
        and schema_enforced_recording_primary_key
        and one_current_assignment_row_per_recording
        and integrity_recording_id_primary_key
        and integrity_schema_enforced_recording_primary_key
        and one_active_owner
        and not multiple_labelers_per_recording_allowed
        and reassignment_replaces_owner
        and stale_sessions_closed_on_reassignment
        and stale_sessions_closed_before_assignment_update
        and reassignment_target_validated_before_session_closure
        and session_closure_and_assignment_update_atomic
        and raw_assignment_change_blocks_open_sessions
        and assignment_manifests_are_control_plane
        and duplicate_manifest_rows_do_not_create_multiple_owners
        and assignment_user_match_required_for_mutation
        and browser_mutation_requires_current_assignment_owner
        and browser_mutation_target_resolved_server_side
        and browser_mutation_target_source == "recording_assignments.active_assignment"
        and labelers_mutate_assigned_training_zarrs
        and not labelers_mutate_intermediate_csvs
        and store_single_owner_contract_allows_ready
        and integrity_ok
        and duplicate_active_owner_count == 0
    )
    return {
        "schema": "palette.web_labeling_assignment_ownership_contract.v1",
        "ready": ready,
        "assignment_scope": assignment_scope,
        "recording_assignment_key": recording_assignment_key,
        "recording_id_primary_key": recording_id_primary_key,
        "schema_enforced_recording_primary_key": schema_enforced_recording_primary_key,
        "one_current_assignment_row_per_recording": one_current_assignment_row_per_recording,
        "store_recording_id_primary_key": integrity_recording_id_primary_key,
        "store_schema_enforced_recording_primary_key": integrity_schema_enforced_recording_primary_key,
        "schema_integrity_source": str(integrity.get("schema_integrity_source") or ""),
        "primary_key_columns": (
            integrity.get("primary_key_columns")
            if isinstance(integrity.get("primary_key_columns"), list)
            else []
        ),
        "one_active_owner": one_active_owner,
        "multiple_labelers_per_recording_allowed": multiple_labelers_per_recording_allowed,
        "reassignment_replaces_owner": reassignment_replaces_owner,
        "stale_sessions_closed_on_reassignment": stale_sessions_closed_on_reassignment,
        "stale_sessions_closed_before_assignment_update": stale_sessions_closed_before_assignment_update,
        "reassignment_target_validated_before_session_closure": (
            reassignment_target_validated_before_session_closure
        ),
        "session_closure_and_assignment_update_atomic": (
            session_closure_and_assignment_update_atomic
        ),
        "raw_assignment_change_blocks_open_sessions": raw_assignment_change_blocks_open_sessions,
        "assignment_manifests_are_control_plane": assignment_manifests_are_control_plane,
        "duplicate_manifest_rows_do_not_create_multiple_owners": (
            duplicate_manifest_rows_do_not_create_multiple_owners
        ),
        "assignment_user_match_required_for_mutation": assignment_user_match_required_for_mutation,
        "browser_mutation_requires_current_assignment_owner": (
            browser_mutation_requires_current_assignment_owner
        ),
        "browser_mutation_target_resolved_server_side": (
            browser_mutation_target_resolved_server_side
        ),
        "browser_mutation_target_source": browser_mutation_target_source,
        "labelers_mutate_assigned_training_zarrs": labelers_mutate_assigned_training_zarrs,
        "labelers_mutate_intermediate_csvs": labelers_mutate_intermediate_csvs,
        "store_single_owner_assignment_contract_present": (
            store_single_owner_assignment_contract_present
        ),
        "store_single_owner_assignment_contract_ready": (
            store_single_owner_assignment_contract_ready
        ),
        "store_single_owner_assignment_contract_met": (
            store_single_owner_assignment_contract_met
        ),
        "store_single_owner_assignment_contract_schema": (
            store_single_owner_assignment_contract_schema
        ),
        "store_single_owner_assignment_contract": dict(store_contract),
        "assignment_ownership_integrity_ok": integrity_ok,
        "active_assignment_count": int(integrity.get("active_assignment_count") or 0),
        "unique_active_recording_count": int(integrity.get("unique_active_recording_count") or 0),
        "duplicate_active_owner_count": duplicate_active_owner_count,
        "duplicate_active_owners": (
            integrity.get("duplicate_active_owners")
            if isinstance(integrity.get("duplicate_active_owners"), list)
            else []
        ),
    }

def _operator_recovery_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_operator_recovery_policy.v1",
        "operator_only": True,
        "admin_routes_require_operator": True,
        "assignment_reassign_route": "/api/admin/assignments",
        "reassignment_session_repair_route": "/api/admin/recordings/{recording_id}/repair-reassignment-sessions",
        "task_detail_route": "/api/admin/tasks/{task_id}",
        "task_state_route": "/api/admin/tasks/{task_id}/state",
        "task_repair_route": "/api/admin/tasks/{task_id}/repair",
        "session_cleanup_route": "/api/admin/sessions/cleanup-stale",
        "session_closure_route": "/api/admin/sessions/{session_id}/closure",
        "audit_event_lookup_route": "/api/admin/events/{event_id}",
        "failed_promotion_retry_route": "/api/admin/events/{event_id}/retry-promotion",
        "reassignment_closes_previous_owner_sessions": True,
        "reassignment_closes_previous_owner_sessions_before_assignment_update": True,
        "reassignment_target_validated_before_session_closure": True,
        "session_closure_and_assignment_update_atomic": True,
        "task_reopen_operator_only": True,
        "completion_closes_open_sessions": True,
        "failed_promotion_retry_operator_only": True,
        "failed_promotion_retry_requires_failed_event": True,
        "failed_promotion_retry_claims_after_event_type_check": True,
        "failed_promotion_retry_claim_event_type": "promotion_retry_started",
        "labeler_failed_promotion_retry_action": "operator_support_only",
        "session_closure_events_operator_inspectable": True,
        "operator_repair_closes_or_supersedes_sessions": True,
        "operator_repair_records_audit_event": True,
        "rollback_requires_backup_plan": True,
        "bad_disposable_mutation_recovery_modes": [
            "restore_backup",
            "regenerate_known_good",
            "discard_disposable",
        ],
        "disposable_mutation_smoke_requires_recovery_path_verification": True,
        "restore_pauses_or_unassigns_recording_before_write": True,
        "labelers_receive_recovery_write_authority": False,
        "browser_recovery_mutations_direct": False,
        "validation_gate": "operator_recovery_contract",
    }

def _operator_recovery_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    required_routes = {
        "assignment_reassign_route": "/api/admin/assignments",
        "reassignment_session_repair_route": "/api/admin/recordings/{recording_id}/repair-reassignment-sessions",
        "task_detail_route": "/api/admin/tasks/{task_id}",
        "task_state_route": "/api/admin/tasks/{task_id}/state",
        "task_repair_route": "/api/admin/tasks/{task_id}/repair",
        "session_cleanup_route": "/api/admin/sessions/cleanup-stale",
        "session_closure_route": "/api/admin/sessions/{session_id}/closure",
        "audit_event_lookup_route": "/api/admin/events/{event_id}",
        "failed_promotion_retry_route": "/api/admin/events/{event_id}/retry-promotion",
    }
    route_values = {key: str(policy.get(key) or "") for key in required_routes}
    missing_or_mismatched_routes = [
        key for key, expected in required_routes.items() if route_values.get(key) != expected
    ]
    operator_only = bool(policy.get("operator_only"))
    admin_routes_require_operator = bool(policy.get("admin_routes_require_operator"))
    reassignment_closes_previous_owner_sessions = bool(
        policy.get("reassignment_closes_previous_owner_sessions")
    )
    reassignment_closes_previous_owner_sessions_before_assignment_update = bool(
        policy.get(
            "reassignment_closes_previous_owner_sessions_before_assignment_update",
            reassignment_closes_previous_owner_sessions,
        )
    )
    reassignment_target_validated_before_session_closure = bool(
        policy.get("reassignment_target_validated_before_session_closure")
    )
    session_closure_and_assignment_update_atomic = bool(
        policy.get("session_closure_and_assignment_update_atomic")
    )
    task_reopen_operator_only = bool(policy.get("task_reopen_operator_only"))
    completion_closes_open_sessions = bool(policy.get("completion_closes_open_sessions"))
    failed_promotion_retry_operator_only = bool(policy.get("failed_promotion_retry_operator_only"))
    failed_promotion_retry_requires_failed_event = bool(
        policy.get("failed_promotion_retry_requires_failed_event")
    )
    failed_promotion_retry_claims_after_event_type_check = bool(
        policy.get("failed_promotion_retry_claims_after_event_type_check")
    )
    failed_promotion_retry_claim_event_type = str(
        policy.get("failed_promotion_retry_claim_event_type") or ""
    )
    labeler_retry_action = str(policy.get("labeler_failed_promotion_retry_action") or "")
    session_closure_events_operator_inspectable = bool(
        policy.get("session_closure_events_operator_inspectable")
    )
    operator_repair_closes_or_supersedes_sessions = bool(
        policy.get("operator_repair_closes_or_supersedes_sessions")
    )
    operator_repair_records_audit_event = bool(policy.get("operator_repair_records_audit_event"))
    rollback_requires_backup_plan = bool(policy.get("rollback_requires_backup_plan"))
    recovery_modes = (
        policy.get("bad_disposable_mutation_recovery_modes")
        if isinstance(policy.get("bad_disposable_mutation_recovery_modes"), (list, tuple))
        else []
    )
    normalized_recovery_modes = sorted({str(mode) for mode in recovery_modes if str(mode)})
    bad_disposable_mutation_recovery_ready = {
        "discard_disposable",
        "regenerate_known_good",
        "restore_backup",
    }.issubset(set(normalized_recovery_modes))
    disposable_mutation_smoke_requires_recovery_path_verification = bool(
        policy.get("disposable_mutation_smoke_requires_recovery_path_verification")
    )
    restore_pauses_or_unassigns_recording_before_write = bool(
        policy.get("restore_pauses_or_unassigns_recording_before_write")
    )
    labelers_receive_recovery_write_authority = bool(
        policy.get("labelers_receive_recovery_write_authority")
    )
    browser_recovery_mutations_direct = bool(policy.get("browser_recovery_mutations_direct"))
    ready = (
        not missing_or_mismatched_routes
        and operator_only
        and admin_routes_require_operator
        and reassignment_closes_previous_owner_sessions
        and reassignment_closes_previous_owner_sessions_before_assignment_update
        and reassignment_target_validated_before_session_closure
        and session_closure_and_assignment_update_atomic
        and task_reopen_operator_only
        and completion_closes_open_sessions
        and failed_promotion_retry_operator_only
        and failed_promotion_retry_requires_failed_event
        and failed_promotion_retry_claims_after_event_type_check
        and failed_promotion_retry_claim_event_type == "promotion_retry_started"
        and labeler_retry_action == "operator_support_only"
        and session_closure_events_operator_inspectable
        and operator_repair_closes_or_supersedes_sessions
        and operator_repair_records_audit_event
        and rollback_requires_backup_plan
        and bad_disposable_mutation_recovery_ready
        and disposable_mutation_smoke_requires_recovery_path_verification
        and restore_pauses_or_unassigns_recording_before_write
        and not labelers_receive_recovery_write_authority
        and not browser_recovery_mutations_direct
    )
    return {
        "schema": "palette.web_labeling_operator_recovery_contract.v1",
        "ready": ready,
        "operator_only": operator_only,
        "admin_routes_require_operator": admin_routes_require_operator,
        "required_routes": dict(required_routes),
        "route_values": route_values,
        "missing_or_mismatched_routes": missing_or_mismatched_routes,
        "assignment_reassign_route": route_values["assignment_reassign_route"],
        "reassignment_session_repair_route": route_values["reassignment_session_repair_route"],
        "task_detail_route": route_values["task_detail_route"],
        "task_state_route": route_values["task_state_route"],
        "task_repair_route": route_values["task_repair_route"],
        "session_cleanup_route": route_values["session_cleanup_route"],
        "session_closure_route": route_values["session_closure_route"],
        "audit_event_lookup_route": route_values["audit_event_lookup_route"],
        "failed_promotion_retry_route": route_values["failed_promotion_retry_route"],
        "reassignment_closes_previous_owner_sessions": reassignment_closes_previous_owner_sessions,
        "reassignment_closes_previous_owner_sessions_before_assignment_update": (
            reassignment_closes_previous_owner_sessions_before_assignment_update
        ),
        "reassignment_target_validated_before_session_closure": (
            reassignment_target_validated_before_session_closure
        ),
        "session_closure_and_assignment_update_atomic": (
            session_closure_and_assignment_update_atomic
        ),
        "task_reopen_operator_only": task_reopen_operator_only,
        "completion_closes_open_sessions": completion_closes_open_sessions,
        "failed_promotion_retry_operator_only": failed_promotion_retry_operator_only,
        "failed_promotion_retry_requires_failed_event": failed_promotion_retry_requires_failed_event,
        "failed_promotion_retry_claims_after_event_type_check": (
            failed_promotion_retry_claims_after_event_type_check
        ),
        "failed_promotion_retry_claim_event_type": failed_promotion_retry_claim_event_type,
        "labeler_failed_promotion_retry_action": labeler_retry_action,
        "session_closure_events_operator_inspectable": session_closure_events_operator_inspectable,
        "operator_repair_closes_or_supersedes_sessions": operator_repair_closes_or_supersedes_sessions,
        "operator_repair_records_audit_event": operator_repair_records_audit_event,
        "rollback_requires_backup_plan": rollback_requires_backup_plan,
        "bad_disposable_mutation_recovery_modes": normalized_recovery_modes,
        "bad_disposable_mutation_recovery_ready": bad_disposable_mutation_recovery_ready,
        "disposable_mutation_smoke_requires_recovery_path_verification": (
            disposable_mutation_smoke_requires_recovery_path_verification
        ),
        "restore_pauses_or_unassigns_recording_before_write": (
            restore_pauses_or_unassigns_recording_before_write
        ),
        "labelers_receive_recovery_write_authority": labelers_receive_recovery_write_authority,
        "browser_recovery_mutations_direct": browser_recovery_mutations_direct,
        "validation_gate": str(policy.get("validation_gate") or ""),
    }

def _mutation_audit_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_mutation_audit_policy.v1",
        "event_store": BROWSER_MUTATION_AUDIT_PROVENANCE["event_store"],
        "append_only": True,
        "server_records_events": True,
        "browser_records_events_directly": False,
        "browser_receives_audit_store_write_credentials": False,
        "required_event_fields": list(BROWSER_MUTATION_AUDIT_PROVENANCE["required_event_fields"]),
        "timestamp_field": BROWSER_MUTATION_AUDIT_PROVENANCE["timestamp_field"],
        "identity_fields": list(BROWSER_MUTATION_AUDIT_PROVENANCE["identity_fields"]),
        "mutation_summary_fields": list(BROWSER_MUTATION_AUDIT_PROVENANCE["mutation_summary_fields"]),
        "per_workflow_write_contracts_include_audit_provenance": True,
        "same_payload_retry_safe": bool(BROWSER_MUTATION_RETRY_POLICY["same_payload_retry_safe"]),
        "duplicate_audit_events_possible": bool(BROWSER_MUTATION_RETRY_POLICY["duplicate_audit_events_possible"]),
        "validation_gate": "disposable_zarr_mutation_smoke",
    }

def _dashboard_base_url(dashboard_url: str | None) -> str:
    text = str(dashboard_url or "").strip()
    if text.endswith(DASHBOARD_PATH):
        return text[: -len(DASHBOARD_PATH)].rstrip("/")
    return ""

def _identity_source_policy(config: ServerConfig) -> dict[str, object]:
    if config.fixed_user:
        source = "fixed_user"
        source_header = ""
        production_ready = bool(_is_loopback_host(config.host) and not config.production)
    elif config.trust_auth_header:
        source = "trusted_auth_header"
        source_header = str(config.auth_header)
        production_ready = bool(config.production and config.auth_header and _is_loopback_host(config.host))
    else:
        source = "disabled"
        source_header = ""
        production_ready = False
    return {
        "assignment_user_source": source,
        "auth_header": source_header,
        "assignment_user_match_required": True,
        "labeler_landing_page_path": "/",
        "labeling_home_page_path": LABELING_HOME_PATH,
        "labeler_landing_page_kind": "datasets_waiting_queue",
        "landing_serves_datasets_waiting_queue": True,
        "datasets_waiting_alias_paths": ["/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH],
        "dashboard_is_fallback": True,
        "queue_first_landing_paths": ["/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH],
        "queue_first_landing_expected_user_guard_supported": True,
        "dashboard_expected_user_guard_supported": True,
        "dataset_queue_page_expected_user_guard_supported": True,
        "personal_work_expected_user_guard_supported": True,
        "personal_dataset_queue_expected_user_guard_supported": True,
        "dataset_queue_expected_user_guard_supported": True,
        "task_open_expected_user_guard_supported": True,
        "task_complete_expected_user_guard_supported": True,
        "promotion_retry_expected_user_guard_supported": True,
        "promotion_retry_current_session_required": True,
        "promotion_retry_labeler_mutation_enabled": False,
        "promotion_retry_labeler_rejection_error": "operator_support_required",
        "promotion_retry_dashboard_action": "operator_support_only",
        "signed_task_link_expected_user_binding_supported": True,
        "expected_user_probe_supported": True,
        "expected_user_guards": {
            "labeler_landing_page": "dashboard_user_mismatch",
            "labeler_me_page": "dashboard_user_mismatch",
            "labeling_home_page": "dashboard_user_mismatch",
            "dashboard": "dashboard_user_mismatch",
            "dataset_queue_page": "dashboard_user_mismatch",
            "personal_work_page": "dashboard_user_mismatch",
            "personal_dataset_queue_page": "dashboard_user_mismatch",
            "personal_work_api": "dashboard_user_mismatch",
            "dataset_queue_api": "dashboard_user_mismatch",
            "task_open_api": "task_open_user_mismatch",
            "task_complete_api": "task_complete_user_mismatch",
            "promotion_retry_api": "promotion_retry_user_mismatch",
            "signed_task_link": "signed_link_user_mismatch",
        },
        "identity_probe_path": IDENTITY_PROBE_PATH,
        "identity_probe_api_path": "/api/me/identity",
        "signed_links_are_not_identity": True,
        "production_ready": production_ready,
        "operator_verification_required_before_invite": True,
        "operator_confirmation": (
            "Configure the deployed reverse proxy or auth layer so the browser user resolved by Palette exactly matches assignment assignee_user values."
        ),
        "verification_instructions": (
            "Before sending real work, ask each labeler to open /identity?expected_user=<assignment user> "
            "and confirm the page reports an identity match."
        ),
    }

def _operator_authorization_policy(
    config: ServerConfig | None = None,
    *,
    include_admin_details: bool = False,
) -> dict[str, object]:
    admin_users = tuple(str(item) for item in config.admin_users) if config is not None else ()
    operator_boundary_known = config is not None
    policy: dict[str, object] = {
        "admin_routes_require_operator": True,
        "admin_route_prefixes": ["/admin", "/api/admin"],
        "operator_user_source": "server_config_admin_users",
        "configuration_flag": "--admin-user",
        "admin_required_error": "admin_required",
        "resolved_user_must_be_in_admin_users": True,
        "labelers_are_not_operators_by_default": True,
        "operator_boundary_known": operator_boundary_known,
        "runtime_preflight_required": not operator_boundary_known,
        "operator_recovery_routes": [
            "/admin",
            "/admin/users/{user}",
            "/admin/recordings/{recording_id}",
            "/admin/tasks/{task_id}",
            "/api/admin/assignments",
            "/api/admin/recordings/{recording_id}/repair-reassignment-sessions",
            "/api/admin/tasks/{task_id}/state",
            "/api/admin/tasks/{task_id}/repair",
            "/api/admin/sessions/cleanup-stale",
            "/api/admin/sessions/{session_id}/closure",
            "/api/admin/events/{event_id}",
            "/api/admin/events/{event_id}/retry-promotion",
        ],
        "operator_authorization_grants_labeler_mutation": False,
        "operator_boundary_required_for_launch": True,
        "production_requires_admin_user": True,
        "admin_user_count": len(admin_users) if operator_boundary_known else None,
        "admin_users_configured": bool(admin_users),
        "operator_boundary_ready": bool(admin_users),
    }
    if include_admin_details:
        policy["admin_users"] = list(admin_users)
    return policy

def _labeler_landing_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return str(base_url).rstrip("/")

def _labeling_home_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{LABELING_HOME_PATH}"

def _identity_probe_url_for_dashboard(dashboard_url: str | None, user: str | None) -> str:
    dashboard = str(dashboard_url or "").strip()
    if not dashboard:
        return ""
    if dashboard.endswith(DASHBOARD_PATH):
        probe_url = f"{dashboard[:-len(DASHBOARD_PATH)]}{IDENTITY_PROBE_PATH}"
    else:
        probe_url = IDENTITY_PROBE_PATH if dashboard == DASHBOARD_PATH else ""
    return _dashboard_url_for_expected_user(probe_url, user) if probe_url else ""

def _labeler_landing_url_for_dashboard(dashboard_url: str | None, user: str | None) -> str:
    dashboard = str(dashboard_url or "").strip()
    if not dashboard:
        return ""
    if dashboard.endswith(DASHBOARD_PATH):
        landing_url = dashboard[: -len(DASHBOARD_PATH)].rstrip("/") or "/"
    else:
        landing_url = "/" if dashboard == DASHBOARD_PATH else ""
    return _dashboard_url_for_expected_user(landing_url, user) if landing_url else ""

def _personal_work_url_for_dashboard(dashboard_url: str | None, user: str | None) -> str:
    dashboard = str(dashboard_url or "").strip()
    if not dashboard:
        return ""
    if dashboard.endswith(DASHBOARD_PATH):
        work_url = f"{dashboard[:-len(DASHBOARD_PATH)]}{PERSONAL_WORK_PATH}"
    else:
        work_url = PERSONAL_WORK_PATH if dashboard == DASHBOARD_PATH else ""
    return _dashboard_url_for_expected_user(work_url, user) if work_url else ""

def _dataset_queue_direct_start_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_dataset_queue_direct_start_policy.v1",
        "enabled": True,
        "button_label": "Start browser task",
        "fallback_label": "Open dashboard fallback",
        "uses_existing_task_open_api": True,
        "method": "POST",
        "endpoint_route_template": "/api/tasks/{task_id}/open",
        "same_origin_only": True,
        "exact_route_required": True,
        "endpoint_task_segment_must_match_row_task_id": True,
        "expected_user_guard_required": True,
        "post_body_expected_user_required": True,
        "post_body_expected_user_field": "expected_user",
        "opens_guarded_browser_session": True,
        "denied_start_returns_task_open_authorization_contract": True,
        "denied_start_support_preserves_task_open_authorization_contract": True,
        "denied_start_support_includes_authorization_context": True,
        "denied_start_contract_reports_no_session_created": True,
        "denied_start_contract_reports_server_authorizes_open_false": True,
        "startable_task_states": list(LABELER_START_TASK_STATES),
        "task_rows_advertise_endpoint_only_when_startable": True,
        "completed_tasks_not_startable": True,
        "non_startable_tasks_do_not_advertise_endpoint": True,
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
    }

def _labeler_safety_policy() -> dict[str, object]:
    return {
        "browser_only": True,
        "labeler_runtime_surface": "browser",
        "requires_local_palette_installation": False,
        "requires_local_crimson_installation": False,
        "requires_local_conda_environment": False,
        "requires_local_project_dependencies": False,
        "dashboard_identity_check_required": True,
        "labeler_landing_page_path": "/",
        "labeling_home_page_path": LABELING_HOME_PATH,
        "labeler_landing_page_kind": "datasets_waiting_queue",
        "landing_serves_datasets_waiting_queue": True,
        "datasets_waiting_alias_paths": ["/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH],
        "dashboard_is_fallback": True,
        "queue_first_landing_paths": ["/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH],
        "dashboard_path": DASHBOARD_PATH,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "personal_work_expected_user_guard_supported": True,
        "personal_dataset_queue_expected_user_guard_supported": True,
        "identity_probe_path": IDENTITY_PROBE_PATH,
        "identity_probe_api_path": "/api/me/identity",
        "identity_probe_expected_user_guard_required": True,
        "identity_probe_diagnostic_only": True,
        "identity_probe_does_not_authorize_work": True,
        "identity_probe_unknown_user_blocks_work_surfaces": True,
        "identity_probe_success_launch_ctas_rendered": True,
        "identity_probe_failed_launch_ctas_suppressed": True,
        "identity_probe_failed_support_urls_diagnostic_only": True,
        "work_filter_query_keys": ["expected_user", "dataset_id", "recording_id", "task_id", "workflow"],
        "expected_user_guards": {
            "labeler_landing_page": "dashboard_user_mismatch",
            "labeler_me_page": "dashboard_user_mismatch",
            "labeling_home_page": "dashboard_user_mismatch",
            "dashboard": "dashboard_user_mismatch",
            "dataset_queue_page": "dashboard_user_mismatch",
            "personal_work_page": "dashboard_user_mismatch",
            "personal_dataset_queue_page": "dashboard_user_mismatch",
            "personal_work_api": "dashboard_user_mismatch",
            "dataset_queue_api": "dashboard_user_mismatch",
            "task_open_api": "task_open_user_mismatch",
            "task_complete_api": "task_complete_user_mismatch",
            "promotion_retry_api": "promotion_retry_user_mismatch",
            "signed_task_link": "signed_link_user_mismatch",
        },
        "no_direct_zarr_edits": True,
        "no_forwarding_links_or_handoffs": True,
        "browser_receives_task_scope": False,
        "browser_receives_raw_zarr_paths": False,
        "labeler_failed_promotion_retry_action": "operator_support_only",
        "labeler_promotion_retry_requires_current_session": True,
        "promotion_retry_labeler_mutation_enabled": False,
        "promotion_retry_labeler_rejection_error": "operator_support_required",
        "operator_failed_promotion_retry_route": "/api/admin/events/{event_id}/retry-promotion",
        "operator_audit_event_lookup_route": "/api/admin/events/{event_id}",
        "browser_response_security_policy": _browser_response_security_policy(),
        "labeler_api_redaction": {
            "redacts_task_scope": True,
            "redacts_raw_zarr_paths": True,
            "redacts_runtime_state_paths": True,
            "redacts_mutation_response_paths": True,
            "redacts_error_detail_paths": True,
            "redacts_path_like_string_values": True,
            "redacts_user_summary_path_like_string_values": True,
            "redacts_direct_storage_paths": True,
            "redacted_key_names": [
                "scope",
                "scope_json",
                "zarr_path",
                "registry_path",
                "review_proxy_manifest",
                "analysis_zarr",
                "analysis_zarr_path",
                "training_zarr",
                "training_zarr_path",
                "promote_training_zarr",
                "promote_training_zarr_path",
                "path",
                "source_path",
            ],
            "redacted_key_suffixes": ["_path", "_zarr"],
            "admin_diagnostics_unredacted": True,
            "operator_support_redacted": True,
        },
        "operator_contact_on_mismatch": True,
        "instructions": (
            "Confirm the dashboard shows the expected assigned user before opening work; "
            "if it shows another user, stop and contact the operator."
        ),
    }

def _dashboard_invitation_message(
    *,
    user: str,
    dashboard_url: str,
    identity_probe_url: str,
    labeler_landing_url: str = "",
    labeling_home_url: str = "",
    dataset_queue_url: str = "",
    personalized_entry_url: str = "",
    ready_to_invite: bool,
    invite_reasons: Sequence[str],
) -> str:
    if not ready_to_invite:
        reasons = ", ".join(str(reason) for reason in invite_reasons if str(reason).strip()) or "operator_review_required"
        actions = " ".join(_dashboard_invite_actions(invite_reasons))
        action_text = f" Next action: {actions}" if actions else ""
        preview_links = [
            f"start page: {labeler_landing_url}" if labeler_landing_url else "",
            f"labeling home: {labeling_home_url}" if labeling_home_url else "",
            f"personalized work queue: {personalized_entry_url}" if personalized_entry_url else "",
            f"dataset queue: {dataset_queue_url}" if dataset_queue_url else "",
            f"dashboard: {dashboard_url}" if dashboard_url else "",
            f"identity check: {identity_probe_url}" if identity_probe_url else "",
        ]
        preview_text = (
            " Preview only: " + "; ".join(link for link in preview_links if link) + "."
            if any(preview_links)
            else ""
        )
        return (
            f"Dashboard ready-row draft is not ready: {reasons}. "
            f"The operator must resolve this before sending.{action_text}{preview_text}"
        )
    identity_text = (
        f" First open the identity check: {identity_probe_url}; confirm it reports you as {user} before labeling."
        if identity_probe_url
        else ""
    )
    preferred_entry_url = personalized_entry_url or dataset_queue_url or labeler_landing_url or dashboard_url
    preferred_text = f" Start here: {preferred_entry_url}." if preferred_entry_url else ""
    landing_text = f" Queue-first start page: {labeler_landing_url}." if labeler_landing_url else ""
    labeling_home_text = f" Human-readable labeling home alias: {labeling_home_url}." if labeling_home_url else ""
    personalized_text = (
        f" Open your personalized dataset queue: {personalized_entry_url}."
        if personalized_entry_url
        else ""
    )
    dataset_text = f" Canonical dataset queue fallback: {dataset_queue_url}." if dataset_queue_url else ""
    dashboard_text = f" Full dashboard fallback: {dashboard_url}." if dashboard_url else ""
    return (
        "Your Palette labeling work is ready. "
        f"{preferred_text} "
        f"{landing_text} "
        f"{labeling_home_text} "
        f"{personalized_text} "
        f"{dataset_text} "
        f"{dashboard_text} "
        f"{identity_text} "
        f"Sign in as {user}. "
        f"Confirm the dashboard shows you as {user} before opening work; "
        "if it shows another user, stop and contact the operator. "
        "No local Palette or Crimson installation is needed. "
        "Use the browser controls only; do not edit zarr files directly or forward this link. "
        "Browser saves are applied server-side to your assigned task/training Zarr scope; CSV, HTML, JSON, and handoff files are metadata only. "
        "Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work."
    )

def _dashboard_invite_actions(reasons: Iterable[str]) -> list[str]:
    guidance = {
        "missing_base_url": "Regenerate the roster with --base-url set to the deployed labeling service URL.",
        "no_users": "Check assignment imports or the --user filter; no assigned users matched this roster.",
        "no_active_recordings": "Assign or reactivate at least one recording for this user before inviting them.",
        "no_open_tasks": "Generate, import, or reopen browser-labeling tasks for this user's active recordings.",
        "preferred_personal_queue_mismatch": "Regenerate roster or handoff links so Start here uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of a canonical fallback.",
        "reassignment_session_safety_failed": "Close stale previous-owner sessions or re-run assignment through assign_recording_with_session_closure before copying ready-row draft text or sharing links.",
        "operator_validation_pending": "Approve required operator validation evidence before copying ready-row draft text or sharing links.",
        "operator_validation_needs_review": "Resolve validation checklist gates marked needs_review before copying ready-row draft text or sharing links.",
        "operator_review_required": "Inspect the assignment/task state before using this ready-row draft.",
    }
    actions: list[str] = []
    for reason in reasons:
        reason_text = str(reason or "").strip()
        if not reason_text:
            continue
        actions.append(guidance.get(reason_text, f"Inspect dashboard invite reason {reason_text}."))
    return actions

def _identity_personal_queue_evidence_status(
    *,
    ready_count: int = 0,
    missing_count: int = 0,
    ready_users: Sequence[str] | None = None,
    missing_users: Sequence[str] | None = None,
    missing_fields_by_user: Mapping[str, object] | None = None,
    all_users_have_personal_queue_evidence: bool = False,
) -> str:
    has_ready_personal_queue_evidence = bool(ready_count or ready_users)
    has_missing_personal_queue_evidence = bool(
        missing_count or missing_users or missing_fields_by_user
    )
    if (
        all_users_have_personal_queue_evidence
        and has_ready_personal_queue_evidence
        and not has_missing_personal_queue_evidence
    ):
        return "ready"
    if has_ready_personal_queue_evidence or has_missing_personal_queue_evidence:
        return "incomplete"
    return "missing"

def _operator_validation_pending_action(gate_ids: Sequence[str]) -> str:
    return (
        "Approve required operator validation evidence before copying ready-row draft text or sharing links: "
        + ", ".join(str(gate_id) for gate_id in gate_ids)
        + "."
    )

def _operator_validation_command_templates(
    gate_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    requested_gate_ids: list[str] = []
    seen_gate_ids: set[str] = set()
    for gate_id in (gate_ids if gate_ids is not None else DEFAULT_OPERATOR_VALIDATION_GATE_IDS):
        gate_text = str(gate_id).strip()
        if not gate_text or gate_text in seen_gate_ids:
            continue
        seen_gate_ids.add(gate_text)
        requested_gate_ids.append(gate_text)
    command_specs: dict[str, dict[str, object]] = {
        "mutable_zarr_backup_confirmation": {
            "id": "record_zarr_backup_evidence",
            "category": "operator_evidence",
            "label": "Record mutable-Zarr backup evidence",
            "evidence_template_field": "zarr_backup_evidence_template",
            "evidence_template_path": "zarr-backup-evidence-template.json",
            "command": (
                "record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json "
                "--execution-manifest BACKUP_EXECUTION_MANIFEST --target-index TARGET_INDEX "
                "--restore-test-result RESTORE_TEST_RESULT --operator OPERATOR"
            ),
            "requires_checksum_refresh_after_run": True,
        },
        "browser_response_security_headers": {
            "id": "record_browser_response_security_evidence",
            "category": "operator_evidence",
            "label": "Record deployed browser response-security evidence",
            "evidence_template_field": "browser_response_security_evidence_template",
            "evidence_template_path": "browser-response-security-evidence-template.json",
            "command": (
                "record-browser-response-security-evidence --evidence "
                "browser-response-security-evidence-template.json --header 'HEADER=VALUE' "
                "--operator OPERATOR --capture-url DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER "
                "--authenticated-test-user SAME_USER_AS_EXPECTED_USER"
            ),
            "requires_checksum_refresh_after_run": True,
        },
        "identity_probe_verification": {
            "id": "record_identity_source_evidence",
            "category": "operator_evidence",
            "label": "Record deployed identity-source and personal queue evidence",
            "evidence_template_field": "identity_source_evidence_template",
            "evidence_template_path": "identity-source-evidence-template.json",
            "command": (
                "record-identity-source-evidence --evidence identity-source-evidence-template.json "
                "--expected-user USER --resolved-user RESOLVED_USER --operator OPERATOR "
                "--authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED"
            ),
            "requires_checksum_refresh_after_run": True,
        },
        "browser_smoke": {
            "id": "record_browser_smoke_evidence",
            "category": "operator_evidence",
            "label": "Record representative browser smoke evidence",
            "evidence_template_field": "browser_smoke_evidence_template",
            "evidence_template_path": "browser-smoke-evidence-template.json",
            "command": (
                "record-browser-smoke-evidence --evidence browser-smoke-evidence-template.json "
                "--expected-user USER --resolved-user USER --operator OPERATOR "
                "--browser-only-runtime-verified --no-local-palette-install-verified "
                "--no-local-crimson-install-verified --no-local-conda-or-project-dependencies-verified "
                "--personalized-dataset-queue-verified "
                "--preferred-labeler-entry-url-matches-personal-dataset-queue "
                "--personalized-labeler-entry-url-matches-personal-dataset-queue "
                "--personalized-work-dashboard-verified "
                "--labeler-sees-only-assigned-work --support-text-redacted "
                "--expected-user-mismatch-rejected --task-opened "
                "--induced-failure-support-detail-redacted --completion-verified "
                "--completed-task-read-only-verified --stale-tab-save-rejected --operator-reopen-verified"
            ),
            "requires_checksum_refresh_after_run": True,
        },
        "disposable_zarr_mutation_smoke": {
            "id": "record_disposable_zarr_mutation_smoke_evidence",
            "category": "operator_evidence",
            "label": "Record disposable-Zarr mutation smoke evidence",
            "evidence_template_field": "disposable_zarr_mutation_smoke_evidence_template",
            "evidence_template_path": "disposable-zarr-mutation-smoke-evidence-template.json",
            "command": (
                "record-disposable-zarr-mutation-smoke-evidence --evidence "
                "disposable-zarr-mutation-smoke-evidence-template.json --workflow-kind WORKFLOW_KIND "
                "--mutation-event-id EVENT_ID --event-lookup-report EVENT_ID-lookup.json "
                "--operator OPERATOR --labeler-user LABELER_USER "
                "--task-scoped-training-zarr-write-verified "
                "--browser-no-direct-zarr-write-authority-verified "
                "--handoff-artifacts-metadata-only-verified --browser-no-csv-or-handoff-write-verified "
                "--client-target-selector-rejection-verified "
                "--operator-event-lookup-verified --bad-mutation-recovery-verified "
                "--bad-mutation-recovery-mode RECOVERY_MODE --bad-mutation-recovery-report RECOVERY_REPORT"
            ),
            "requires_checksum_refresh_after_run": True,
        },
        "operator_recovery_contract": {
            "id": "record_operator_recovery_contract_gate",
            "category": "validation_checklist",
            "label": "Record operator recovery contract gate",
            "command": (
                "update-validation-checklist --path validation-checklist.json "
                "--gate operator_recovery_contract --status passed --operator OPERATOR "
                "--append-log validation-log-template.md"
            ),
            "requires_checksum_refresh_after_run": True,
        },
    }
    commands: list[dict[str, object]] = []
    missing_command_gate_ids: list[str] = []
    template_apply_gate_ids: list[str] = []
    evidence_templates_by_gate_id: dict[str, dict[str, object]] = {}
    validation_checklist_gate_ids: list[str] = []
    for gate_id in requested_gate_ids:
        spec = command_specs.get(gate_id)
        if not spec:
            missing_command_gate_ids.append(gate_id)
            continue
        if str(spec.get("category") or "") == "operator_evidence":
            template_apply_gate_ids.append(gate_id)
            evidence_templates_by_gate_id[gate_id] = {
                "gate_id": gate_id,
                "template_field": str(spec.get("evidence_template_field") or ""),
                "template_path": str(spec.get("evidence_template_path") or ""),
                "record_command_id": str(spec.get("id") or ""),
                "apply_required_after_approval": True,
                "apply_command_id": "apply_operator_evidence_templates",
            }
        elif str(spec.get("category") or "") == "validation_checklist":
            validation_checklist_gate_ids.append(gate_id)
        commands.append(
            {
                **spec,
                "gate_ids": [gate_id],
            }
        )
    if template_apply_gate_ids:
        commands.append(
            {
                "id": "apply_operator_evidence_templates",
                "category": "validation_checklist",
                "label": "Apply approved operator evidence templates to handoff readiness",
                "command": (
                    "apply-operator-evidence-templates --path validation-checklist.json "
                    "--operator OPERATOR --append-log validation-log-template.md"
                ),
                "gate_ids": template_apply_gate_ids,
                "requires_checksum_refresh_after_run": True,
            }
        )
    launch_evidence_collection_steps: list[dict[str, object]] = []
    for gate_id in requested_gate_ids:
        spec = command_specs.get(gate_id)
        if not spec:
            continue
        category = str(spec.get("category") or "")
        template_backed = category == "operator_evidence"
        launch_evidence_collection_steps.append(
            {
                "gate_id": gate_id,
                "gate_category": category,
                "operator_only": True,
                "blocks_labeler_link_share_until_satisfied": True,
                "record_command_id": str(spec.get("id") or ""),
                "record_command": str(spec.get("command") or ""),
                "evidence_template_field": str(
                    spec.get("evidence_template_field") or ""
                ),
                "evidence_template_path": str(
                    spec.get("evidence_template_path") or ""
                ),
                "template_backed": template_backed,
                "apply_required_after_approval": template_backed,
                "apply_command_id": (
                    "apply_operator_evidence_templates" if template_backed else ""
                ),
                "requires_checksum_refresh_after_run": bool(
                    spec.get("requires_checksum_refresh_after_run")
                ),
            }
        )
    launch_evidence_collection_plan = {
        "schema": "palette.web_labeling_launch_evidence_collection_plan.v1",
        "operator_only": True,
        "commands_are_labeler_instructions": False,
        "labelers_must_not_run_commands": True,
        "safe_to_share_blocked_until_plan_complete": True,
        "gate_ids": requested_gate_ids,
        "template_backed_gate_ids": template_apply_gate_ids,
        "validation_checklist_gate_ids": validation_checklist_gate_ids,
        "missing_command_gate_ids": missing_command_gate_ids,
        "steps": launch_evidence_collection_steps,
        "steps_by_gate_id": {
            str(step.get("gate_id") or ""): step
            for step in launch_evidence_collection_steps
        },
        "step_count": len(launch_evidence_collection_steps),
        "record_command_ids": [
            str(step.get("record_command_id") or "")
            for step in launch_evidence_collection_steps
        ],
        "apply_required_gate_ids": template_apply_gate_ids,
        "apply_command_id": (
            "apply_operator_evidence_templates" if template_apply_gate_ids else ""
        ),
        "final_inspection_command": "inspect-handoff --path PACKAGE --require-shareable",
        "required_final_field": "labeler_links_safe_to_share",
        "required_final_value": True,
        "operator_action": (
            "Record or approve each launch evidence step, apply approved evidence "
            "templates when required, refresh checksums, then run inspect-handoff "
            "--require-shareable and require labeler_links_safe_to_share=true."
            if launch_evidence_collection_steps
            else ""
        ),
    }
    return {
        "schema": "palette.web_labeling_operator_validation_command_templates.v1",
        "commands_are_operator_only": True,
        "commands_are_labeler_instructions": False,
        "labelers_must_not_run_commands": True,
        "operator_authorization_required": True,
        "gate_ids": requested_gate_ids,
        "command_count": len(commands),
        "command_ids": [str(command.get("id") or "") for command in commands],
        "commands": commands,
        "template_backed_gate_ids": template_apply_gate_ids,
        "validation_checklist_gate_ids": validation_checklist_gate_ids,
        "evidence_templates_by_gate_id": evidence_templates_by_gate_id,
        "evidence_template_fields_by_gate_id": {
            gate_id: str(template.get("template_field") or "")
            for gate_id, template in evidence_templates_by_gate_id.items()
        },
        "evidence_template_paths_by_gate_id": {
            gate_id: str(template.get("template_path") or "")
            for gate_id, template in evidence_templates_by_gate_id.items()
        },
        "apply_required_gate_ids": template_apply_gate_ids,
        "apply_command_id": "apply_operator_evidence_templates" if template_apply_gate_ids else "",
        "commands_by_gate_id": {
            gate_id: [
                str(command.get("id") or "")
                for command in commands
                if gate_id in command.get("gate_ids", [])
            ]
            for gate_id in requested_gate_ids
        },
        "missing_command_gate_ids": missing_command_gate_ids,
        "launch_evidence_collection_plan": launch_evidence_collection_plan,
        "launch_evidence_collection_plan_schema": str(
            launch_evidence_collection_plan["schema"]
        ),
        "launch_evidence_collection_step_count": len(
            launch_evidence_collection_steps
        ),
        "launch_evidence_collection_gate_ids": requested_gate_ids,
        "launch_evidence_collection_record_command_ids": [
            str(step.get("record_command_id") or "")
            for step in launch_evidence_collection_steps
        ],
        "launch_evidence_collection_operator_only": True,
        "launch_evidence_collection_required_final_field": (
            "labeler_links_safe_to_share"
        ),
        "launch_evidence_collection_required_final_value": True,
        "launch_evidence_collection_final_inspection_command": (
            "inspect-handoff --path PACKAGE --require-shareable"
        ),
        "operator_action": (
            "Collect and approve the listed operator evidence before sharing labeler links."
            if commands
            else ""
        ),
    }

def _operator_validation_command_template_fields(source: Mapping[str, object]) -> dict[str, object]:
    templates = (
        source.get("operator_validation_command_templates")
        if isinstance(source.get("operator_validation_command_templates"), Mapping)
        else _operator_validation_command_templates(
            source.get("operator_validation_required_missing_evidence_gate_ids")
            if isinstance(source.get("operator_validation_required_missing_evidence_gate_ids"), list)
            else None
        )
    )
    return {
        "operator_validation_command_template_schema": str(templates.get("schema") or ""),
        "operator_validation_command_template_commands_are_operator_only": bool(
            templates.get("commands_are_operator_only", True)
        ),
        "operator_validation_command_template_commands_are_labeler_instructions": bool(
            templates.get("commands_are_labeler_instructions")
        ),
        "operator_validation_command_template_labelers_must_not_run_commands": bool(
            templates.get("labelers_must_not_run_commands", True)
        ),
        "operator_validation_command_template_operator_authorization_required": bool(
            templates.get("operator_authorization_required", True)
        ),
        "operator_validation_command_template_command_count": int(
            templates.get("command_count") or 0
        ),
        "operator_validation_command_template_gate_ids": json.dumps(
            templates.get("gate_ids") if isinstance(templates.get("gate_ids"), list) else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_command_ids": json.dumps(
            templates.get("command_ids") if isinstance(templates.get("command_ids"), list) else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_template_backed_gate_ids": json.dumps(
            templates.get("template_backed_gate_ids")
            if isinstance(templates.get("template_backed_gate_ids"), list)
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_validation_checklist_gate_ids": json.dumps(
            templates.get("validation_checklist_gate_ids")
            if isinstance(templates.get("validation_checklist_gate_ids"), list)
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_apply_required_gate_ids": json.dumps(
            templates.get("apply_required_gate_ids")
            if isinstance(templates.get("apply_required_gate_ids"), list)
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_evidence_template_fields_by_gate_id": json.dumps(
            templates.get("evidence_template_fields_by_gate_id")
            if isinstance(templates.get("evidence_template_fields_by_gate_id"), Mapping)
            else {},
            sort_keys=True,
        ),
        "operator_validation_command_template_evidence_template_paths_by_gate_id": json.dumps(
            templates.get("evidence_template_paths_by_gate_id")
            if isinstance(templates.get("evidence_template_paths_by_gate_id"), Mapping)
            else {},
            sort_keys=True,
        ),
        "operator_validation_command_template_missing_command_gate_ids": json.dumps(
            templates.get("missing_command_gate_ids")
            if isinstance(templates.get("missing_command_gate_ids"), list)
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_launch_evidence_collection_plan_schema": str(
            templates.get("launch_evidence_collection_plan_schema") or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_step_count": int(
            templates.get("launch_evidence_collection_step_count") or 0
        ),
        "operator_validation_command_template_launch_evidence_collection_gate_ids": json.dumps(
            templates.get("launch_evidence_collection_gate_ids")
            if isinstance(templates.get("launch_evidence_collection_gate_ids"), list)
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_launch_evidence_collection_record_command_ids": json.dumps(
            templates.get("launch_evidence_collection_record_command_ids")
            if isinstance(
                templates.get("launch_evidence_collection_record_command_ids"),
                list,
            )
            else [],
            sort_keys=True,
        ),
        "operator_validation_command_template_launch_evidence_collection_operator_only": bool(
            templates.get("launch_evidence_collection_operator_only", True)
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_field": str(
            templates.get("launch_evidence_collection_required_final_field") or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_value": bool(
            templates.get("launch_evidence_collection_required_final_value")
        ),
        "operator_validation_command_template_launch_evidence_collection_final_inspection_command": str(
            templates.get("launch_evidence_collection_final_inspection_command") or ""
        ),
        "operator_validation_command_template_operator_action": str(
            templates.get("operator_action") or ""
        ),
    }

def _operator_validation_approval_scope_fields() -> dict[str, object]:
    return {
        "operator_validation_required_before_invite_legacy_semantics": (
            "operator_validation_required_before_ready_row_draft_not_safe_share_approval"
        ),
        "operator_validation_required_before_invite_is_safe_share_approval": False,
        "operator_validation_required_before_invite_safe_share_field": (
            "labeler_links_safe_to_share"
        ),
        "operator_launch_approved_legacy_semantics": (
            "operator_validation_evidence_only_not_safe_share_approval"
        ),
        "operator_launch_approved_is_safe_share_approval": False,
        "operator_launch_approved_requires_safe_share_inspection": True,
        "operator_launch_approved_required_safe_share_field": (
            "labeler_links_safe_to_share"
        ),
        "operator_launch_approved_required_safe_share_value": True,
    }

def _dashboard_operator_validation_fields(
    *,
    checklist_path: str | None = None,
    operator_launch_approved: bool = False,
) -> dict[str, object]:
    approval_scope_fields = _operator_validation_approval_scope_fields()
    if checklist_path and operator_launch_approved:
        raise ValueError(
            "Use either --operator-validation-checklist or --operator-launch-approved, not both."
        )
    if checklist_path:
        path = Path(checklist_path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"Operator validation checklist must be a JSON object: {path}")
        fields = _operator_validation_invitation_fields(payload)
        fields.update(
            {
                "operator_validation_source": "validation_checklist",
                "operator_validation_checklist_path": str(path),
                **approval_scope_fields,
            }
        )
        return fields
    if operator_launch_approved:
        return {
            **approval_scope_fields,
            "operator_validation_required_before_invite": True,
            "operator_validation_all_complete": True,
            "operator_validation_declared_all_complete": True,
            "operator_validation_gate_count": 0,
            "operator_validation_ready_for_operator_validation": True,
            "operator_validation_status": "passed",
            "operator_validation_pending_gate_ids": [],
            "operator_validation_needs_review_gate_ids": [],
            "operator_validation_required_missing_evidence_gate_ids": [],
            "operator_validation_required_pending_gate_count": 0,
            "operator_validation_needs_review_gate_count": 0,
            "operator_validation_required_missing_evidence_gate_count": 0,
            "operator_validation_operator_action": "",
            "operator_validation_source": "manual_operator_assertion",
            "operator_validation_checklist_path": "",
        }
    default_required_gate_ids = list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS)
    return {
        **approval_scope_fields,
        "operator_validation_required_before_invite": True,
        "operator_validation_all_complete": False,
        "operator_validation_declared_all_complete": False,
        "operator_validation_gate_count": len(default_required_gate_ids),
        "operator_validation_ready_for_operator_validation": False,
        "operator_validation_status": "pending_operator_evidence",
        "operator_validation_pending_gate_ids": default_required_gate_ids,
        "operator_validation_needs_review_gate_ids": [],
        "operator_validation_required_missing_evidence_gate_ids": default_required_gate_ids,
        "operator_validation_required_pending_gate_count": len(default_required_gate_ids),
        "operator_validation_needs_review_gate_count": 0,
        "operator_validation_required_missing_evidence_gate_count": len(default_required_gate_ids),
        "operator_validation_operator_action": _operator_validation_pending_action(
            default_required_gate_ids
        ),
        "operator_validation_source": "none",
        "operator_validation_checklist_path": "",
    }

def _dashboard_operator_validation_fields_for_config(config: ServerConfig) -> dict[str, object]:
    if config.validation_checklist_path is None:
        return _dashboard_operator_validation_fields()
    try:
        return _dashboard_operator_validation_fields(
            checklist_path=str(config.validation_checklist_path)
        )
    except Exception as exc:
        default_required_gate_ids = list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS)
        return {
            **_operator_validation_approval_scope_fields(),
            "operator_validation_required_before_invite": True,
            "operator_validation_all_complete": False,
            "operator_validation_declared_all_complete": False,
            "operator_validation_gate_count": len(default_required_gate_ids),
            "operator_validation_ready_for_operator_validation": False,
            "operator_validation_status": "invalid_checklist",
            "operator_validation_pending_gate_ids": default_required_gate_ids,
            "operator_validation_needs_review_gate_ids": [],
            "operator_validation_required_missing_evidence_gate_ids": default_required_gate_ids,
            "operator_validation_required_pending_gate_count": len(default_required_gate_ids),
            "operator_validation_needs_review_gate_count": 0,
            "operator_validation_required_missing_evidence_gate_count": len(
                default_required_gate_ids
            ),
            "operator_validation_operator_action": (
                "Fix or regenerate the configured validation checklist before sharing "
                "labeler links or allowing browser Start/Open."
            ),
            "operator_validation_source": "invalid_validation_checklist",
            "operator_validation_checklist_path": str(config.validation_checklist_path),
            "operator_validation_checklist_error": str(exc),
        }

def _runtime_operator_validation_start_gate_not_required() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_runtime_operator_validation_start_gate.v1",
        "required_for_browser_start": False,
        "validation_checklist_configured": False,
        "ready": True,
        "blocks_task_open": False,
        "not_ready_reason": "",
        "operator_validation_status": "not_required",
        "operator_validation_all_complete": False,
        "operator_validation_pending_gate_ids": [],
        "operator_validation_needs_review_gate_ids": [],
        "operator_validation_required_missing_evidence_gate_ids": [],
        "operator_validation_required_pending_gate_count": 0,
        "operator_validation_needs_review_gate_count": 0,
        "operator_validation_required_missing_evidence_gate_count": 0,
        "operator_action": "",
    }

def _runtime_operator_validation_start_gate(
    config: ServerConfig,
    *,
    include_operator_details: bool = False,
) -> dict[str, object]:
    if not bool(config.require_operator_validation_for_start):
        gate = _runtime_operator_validation_start_gate_not_required()
        gate["validation_checklist_configured"] = config.validation_checklist_path is not None
        if include_operator_details and config.validation_checklist_path is not None:
            gate["validation_checklist_path"] = str(config.validation_checklist_path)
        return gate
    if config.validation_checklist_path is None:
        return {
            "schema": "palette.web_labeling_runtime_operator_validation_start_gate.v1",
            "required_for_browser_start": True,
            "validation_checklist_configured": False,
            "ready": False,
            "blocks_task_open": True,
            "not_ready_reason": "operator_validation_checklist_missing",
            "operator_validation_status": "missing_checklist",
            "operator_validation_all_complete": False,
            "operator_validation_pending_gate_ids": [],
            "operator_validation_needs_review_gate_ids": [],
            "operator_validation_required_missing_evidence_gate_ids": [],
            "operator_validation_required_pending_gate_count": 0,
            "operator_validation_needs_review_gate_count": 0,
            "operator_validation_required_missing_evidence_gate_count": 0,
            "operator_action": (
                "Start the labeling server with --validation-checklist pointing at the "
                "operator-approved launch validation checklist, or disable the runtime "
                "start gate only for controlled local development."
            ),
        }
    try:
        fields = _dashboard_operator_validation_fields(
            checklist_path=str(config.validation_checklist_path)
        )
        public_fields = _operator_validation_public_fields(fields)
    except Exception as exc:
        gate = {
            "schema": "palette.web_labeling_runtime_operator_validation_start_gate.v1",
            "required_for_browser_start": True,
            "validation_checklist_configured": True,
            "ready": False,
            "blocks_task_open": True,
            "not_ready_reason": "operator_validation_checklist_invalid",
            "operator_validation_status": "invalid_checklist",
            "operator_validation_all_complete": False,
            "operator_validation_pending_gate_ids": [],
            "operator_validation_needs_review_gate_ids": [],
            "operator_validation_required_missing_evidence_gate_ids": [],
            "operator_validation_required_pending_gate_count": 0,
            "operator_validation_needs_review_gate_count": 0,
            "operator_validation_required_missing_evidence_gate_count": 0,
            "operator_action": (
                "Fix or regenerate the validation checklist before allowing browser "
                "Start/Open to create labeling sessions."
            ),
        }
        if include_operator_details:
            gate["validation_checklist_path"] = str(config.validation_checklist_path)
            gate["validation_checklist_error"] = str(exc)
        return gate
    pending_gate_ids = [
        str(gate_id)
        for gate_id in public_fields.get("operator_validation_pending_gate_ids", [])
        if str(gate_id).strip()
    ]
    needs_review_gate_ids = [
        str(gate_id)
        for gate_id in public_fields.get("operator_validation_needs_review_gate_ids", [])
        if str(gate_id).strip()
    ]
    required_missing_evidence_gate_ids = [
        str(gate_id)
        for gate_id in public_fields.get(
            "operator_validation_required_missing_evidence_gate_ids",
            [],
        )
        if str(gate_id).strip()
    ]
    all_complete = bool(public_fields.get("operator_validation_all_complete"))
    ready = (
        all_complete
        and not pending_gate_ids
        and not needs_review_gate_ids
        and not required_missing_evidence_gate_ids
    )
    if ready:
        not_ready_reason = ""
        operator_action = ""
    elif needs_review_gate_ids:
        not_ready_reason = "operator_validation_needs_review"
        operator_action = (
            str(public_fields.get("operator_validation_operator_action") or "")
            or "Resolve validation checklist gates marked needs_review before allowing browser Start/Open."
        )
    else:
        not_ready_reason = "operator_validation_pending_operator_evidence"
        operator_action = (
            str(public_fields.get("operator_validation_operator_action") or "")
            or "Complete required operator validation evidence before allowing browser Start/Open."
        )
    gate = {
        "schema": "palette.web_labeling_runtime_operator_validation_start_gate.v1",
        "required_for_browser_start": True,
        "validation_checklist_configured": True,
        "ready": ready,
        "blocks_task_open": not ready,
        "not_ready_reason": not_ready_reason,
        "operator_validation_status": str(
            public_fields.get("operator_validation_status") or ""
        ),
        "operator_validation_all_complete": all_complete,
        "operator_validation_pending_gate_ids": pending_gate_ids,
        "operator_validation_needs_review_gate_ids": needs_review_gate_ids,
        "operator_validation_required_missing_evidence_gate_ids": (
            required_missing_evidence_gate_ids
        ),
        "operator_validation_required_pending_gate_count": int(
            public_fields.get("operator_validation_required_pending_gate_count") or 0
        ),
        "operator_validation_needs_review_gate_count": int(
            public_fields.get("operator_validation_needs_review_gate_count") or 0
        ),
        "operator_validation_required_missing_evidence_gate_count": int(
            public_fields.get("operator_validation_required_missing_evidence_gate_count")
            or 0
        ),
        "operator_action": operator_action,
    }
    if include_operator_details:
        gate["validation_checklist_path"] = str(config.validation_checklist_path)
    return gate

def _runtime_operator_validation_mutation_gate_not_required() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_runtime_operator_validation_mutation_gate.v1",
        "required_for_browser_mutation": False,
        "validation_checklist_configured": False,
        "ready": True,
        "blocks_browser_mutation": False,
        "not_ready_reason": "",
        "operator_validation_status": "not_required",
        "operator_validation_all_complete": False,
        "operator_validation_pending_gate_ids": [],
        "operator_validation_needs_review_gate_ids": [],
        "operator_validation_required_missing_evidence_gate_ids": [],
        "operator_validation_required_pending_gate_count": 0,
        "operator_validation_needs_review_gate_count": 0,
        "operator_validation_required_missing_evidence_gate_count": 0,
        "operator_action": "",
    }

def _runtime_operator_validation_mutation_gate(
    config: ServerConfig,
    *,
    include_operator_details: bool = False,
) -> dict[str, object]:
    start_gate = _runtime_operator_validation_start_gate(
        config,
        include_operator_details=include_operator_details,
    )
    if not bool(start_gate.get("required_for_browser_start")):
        gate = _runtime_operator_validation_mutation_gate_not_required()
        gate["validation_checklist_configured"] = bool(
            start_gate.get("validation_checklist_configured")
        )
        if include_operator_details and "validation_checklist_path" in start_gate:
            gate["validation_checklist_path"] = str(start_gate.get("validation_checklist_path") or "")
        return gate
    gate = {
        "schema": "palette.web_labeling_runtime_operator_validation_mutation_gate.v1",
        "required_for_browser_mutation": True,
        "validation_checklist_configured": bool(
            start_gate.get("validation_checklist_configured")
        ),
        "ready": bool(start_gate.get("ready")),
        "blocks_browser_mutation": bool(start_gate.get("blocks_task_open")),
        "not_ready_reason": str(start_gate.get("not_ready_reason") or ""),
        "operator_validation_status": str(start_gate.get("operator_validation_status") or ""),
        "operator_validation_all_complete": bool(
            start_gate.get("operator_validation_all_complete")
        ),
        "operator_validation_pending_gate_ids": list(
            start_gate.get("operator_validation_pending_gate_ids")
            if isinstance(start_gate.get("operator_validation_pending_gate_ids"), list)
            else []
        ),
        "operator_validation_needs_review_gate_ids": list(
            start_gate.get("operator_validation_needs_review_gate_ids")
            if isinstance(start_gate.get("operator_validation_needs_review_gate_ids"), list)
            else []
        ),
        "operator_validation_required_missing_evidence_gate_ids": list(
            start_gate.get("operator_validation_required_missing_evidence_gate_ids")
            if isinstance(
                start_gate.get("operator_validation_required_missing_evidence_gate_ids"),
                list,
            )
            else []
        ),
        "operator_validation_required_pending_gate_count": int(
            start_gate.get("operator_validation_required_pending_gate_count") or 0
        ),
        "operator_validation_needs_review_gate_count": int(
            start_gate.get("operator_validation_needs_review_gate_count") or 0
        ),
        "operator_validation_required_missing_evidence_gate_count": int(
            start_gate.get("operator_validation_required_missing_evidence_gate_count")
            or 0
        ),
        "operator_action": str(start_gate.get("operator_action") or ""),
    }
    if include_operator_details and "validation_checklist_path" in start_gate:
        gate["validation_checklist_path"] = str(start_gate.get("validation_checklist_path") or "")
    if include_operator_details and "validation_checklist_error" in start_gate:
        gate["validation_checklist_error"] = str(start_gate.get("validation_checklist_error") or "")
    return gate

def _operator_validation_public_fields(source: Mapping[str, object]) -> dict[str, object]:
    status = str(source.get("operator_validation_status") or "")
    validation_source = str(source.get("operator_validation_source") or "")
    all_complete = bool(source.get("operator_validation_all_complete"))
    pending_gate_ids = (
        source.get("operator_validation_pending_gate_ids")
        if isinstance(source.get("operator_validation_pending_gate_ids"), list)
        else []
    )
    needs_review_gate_ids = (
        source.get("operator_validation_needs_review_gate_ids")
        if isinstance(source.get("operator_validation_needs_review_gate_ids"), list)
        else []
    )
    required_missing_evidence_gate_ids = (
        source.get("operator_validation_required_missing_evidence_gate_ids")
        if isinstance(source.get("operator_validation_required_missing_evidence_gate_ids"), list)
        else []
    )
    public_pending_without_evidence_source = (
        not all_complete
        and status == "pending_operator_evidence"
        and validation_source in {"", "none"}
    )
    default_pending_required_gates = (
        public_pending_without_evidence_source
        and not pending_gate_ids
        and not required_missing_evidence_gate_ids
    )
    if default_pending_required_gates:
        pending_gate_ids = list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS)
        required_missing_evidence_gate_ids = list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS)
    elif public_pending_without_evidence_source and not pending_gate_ids:
        pending_gate_ids = list(required_missing_evidence_gate_ids)
    elif public_pending_without_evidence_source and not required_missing_evidence_gate_ids:
        required_missing_evidence_gate_ids = list(pending_gate_ids)
    if public_pending_without_evidence_source:
        if not validation_source:
            validation_source = "none"
    gate_count = int(source.get("operator_validation_gate_count") or 0)
    if public_pending_without_evidence_source and gate_count == 0:
        gate_count = max(len(pending_gate_ids), len(required_missing_evidence_gate_ids))
    required_pending_gate_count = int(
        source.get("operator_validation_required_pending_gate_count") or 0
    )
    required_missing_evidence_gate_count = int(
        source.get("operator_validation_required_missing_evidence_gate_count") or 0
    )
    if public_pending_without_evidence_source:
        if required_pending_gate_count == 0:
            required_pending_gate_count = len(pending_gate_ids)
        if required_missing_evidence_gate_count == 0:
            required_missing_evidence_gate_count = len(required_missing_evidence_gate_ids)
    operator_action = str(source.get("operator_validation_operator_action") or "")
    if public_pending_without_evidence_source and not operator_action:
        operator_action = _operator_validation_pending_action(pending_gate_ids)
    outstanding_gate_id_set = {
        str(gate_id)
        for gate_id in [
            *pending_gate_ids,
            *needs_review_gate_ids,
            *required_missing_evidence_gate_ids,
        ]
        if str(gate_id).strip()
    }
    outstanding_gate_ids = [
        gate_id
        for gate_id in DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        if gate_id in outstanding_gate_id_set
    ]
    outstanding_gate_ids.extend(
        sorted(
            gate_id
            for gate_id in outstanding_gate_id_set
            if gate_id not in DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        )
    )
    external_evidence_required_gate_ids = [
        gate_id for gate_id in outstanding_gate_ids if gate_id in OPERATOR_EVIDENCE_TEMPLATE_FIELDS
    ]
    checklist_only_required_gate_ids = [
        gate_id
        for gate_id in outstanding_gate_ids
        if gate_id not in OPERATOR_EVIDENCE_TEMPLATE_FIELDS
    ]
    external_command_templates = _operator_validation_command_templates(
        external_evidence_required_gate_ids
    )
    external_template_fields_by_gate_id = dict(
        external_command_templates.get("evidence_template_fields_by_gate_id")
        if isinstance(
            external_command_templates.get("evidence_template_fields_by_gate_id"),
            Mapping,
        )
        else {}
    )
    external_template_paths_by_gate_id = dict(
        external_command_templates.get("evidence_template_paths_by_gate_id")
        if isinstance(
            external_command_templates.get("evidence_template_paths_by_gate_id"),
            Mapping,
        )
        else {}
    )
    identity_ready_users = (
        source.get("identity_personal_queue_evidence_ready_users")
        if isinstance(source.get("identity_personal_queue_evidence_ready_users"), list)
        else []
    )
    identity_missing_users = (
        source.get("identity_personal_queue_evidence_missing_users")
        if isinstance(source.get("identity_personal_queue_evidence_missing_users"), list)
        else []
    )
    identity_missing_fields_by_user_source = source.get(
        "identity_personal_queue_evidence_missing_fields_by_user"
    )
    identity_missing_fields_by_user = (
        {
            str(user): [str(field) for field in fields] if isinstance(fields, list) else []
            for user, fields in identity_missing_fields_by_user_source.items()
        }
        if isinstance(identity_missing_fields_by_user_source, Mapping)
        else {}
    )
    identity_ready_count = int(
        source.get("identity_personal_queue_evidence_ready_count") or len(identity_ready_users)
    )
    identity_missing_count = int(
        source.get("identity_personal_queue_evidence_missing_count")
        or len(identity_missing_users)
    )
    identity_all_users_have_personal_queue_evidence = bool(
        source.get("identity_all_users_have_personal_queue_evidence")
    )
    identity_personal_queue_evidence_status = _identity_personal_queue_evidence_status(
        ready_count=identity_ready_count,
        missing_count=identity_missing_count,
        ready_users=identity_ready_users,
        missing_users=identity_missing_users,
        missing_fields_by_user=identity_missing_fields_by_user,
        all_users_have_personal_queue_evidence=identity_all_users_have_personal_queue_evidence,
    )
    return {
        "operator_validation_required_before_invite": bool(
            source.get("operator_validation_required_before_invite")
        ),
        "operator_validation_all_complete": all_complete,
        "operator_validation_declared_all_complete": bool(
            source.get("operator_validation_declared_all_complete")
        ),
        "operator_validation_ready_for_operator_validation": bool(
            source.get("operator_validation_ready_for_operator_validation")
        ),
        "operator_validation_gate_count": gate_count,
        "operator_validation_status": status,
        "operator_validation_source": validation_source,
        "operator_validation_pending_gate_ids": pending_gate_ids,
        "operator_validation_needs_review_gate_ids": needs_review_gate_ids,
        "operator_validation_required_missing_evidence_gate_ids": required_missing_evidence_gate_ids,
        "operator_validation_required_pending_gate_count": required_pending_gate_count,
        "operator_validation_needs_review_gate_count": int(
            source.get("operator_validation_needs_review_gate_count") or 0
        ),
        "operator_validation_required_missing_evidence_gate_count": (
            required_missing_evidence_gate_count
        ),
        "operator_validation_operator_action": operator_action,
        **_safe_share_checklist_field_values(source),
        "operator_validation_external_evidence_required": bool(
            external_evidence_required_gate_ids
        ),
        "operator_validation_external_evidence_required_gate_ids": (
            external_evidence_required_gate_ids
        ),
        "operator_validation_external_evidence_required_gate_count": len(
            external_evidence_required_gate_ids
        ),
        "operator_validation_external_evidence_template_fields_by_gate_id": (
            external_template_fields_by_gate_id
        ),
        "operator_validation_external_evidence_template_paths_by_gate_id": (
            external_template_paths_by_gate_id
        ),
        "operator_validation_checklist_only_required_gate_ids": (
            checklist_only_required_gate_ids
        ),
        "operator_validation_checklist_only_required_gate_count": len(
            checklist_only_required_gate_ids
        ),
        "identity_personal_queue_evidence_status": identity_personal_queue_evidence_status,
        "identity_personal_queue_evidence_ready_count": identity_ready_count,
        "identity_personal_queue_evidence_missing_count": identity_missing_count,
        "identity_personal_queue_evidence_ready_users": identity_ready_users,
        "identity_personal_queue_evidence_missing_users": identity_missing_users,
        "identity_personal_queue_evidence_missing_fields_by_user": (
            identity_missing_fields_by_user
        ),
        "identity_all_users_have_personal_queue_evidence": (
            identity_all_users_have_personal_queue_evidence
        ),
    }

def _operator_validation_gate_metadata_fields() -> dict[str, object]:
    return {
        "operator_validation_gate_status_values": list(
            OPERATOR_VALIDATION_GATE_STATUS_VALUES
        ),
        "operator_validation_gate_ids": list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS),
        "operator_validation_gate_flat_field_suffixes": list(
            OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES
        ),
    }

def _operator_validation_gate_flat_fields(source: Mapping[str, object]) -> dict[str, object]:
    public_fields = _operator_validation_public_fields(source)
    safe_share_gate_statuses = (
        source.get("safe_share_launch_blocking_gate_statuses")
        if isinstance(source.get("safe_share_launch_blocking_gate_statuses"), Mapping)
        else {}
    )
    pending_gate_ids = {
        str(gate_id)
        for gate_id in public_fields.get("operator_validation_pending_gate_ids", [])
        if str(gate_id).strip()
    }
    missing_evidence_gate_ids = {
        str(gate_id)
        for gate_id in public_fields.get(
            "operator_validation_required_missing_evidence_gate_ids", []
        )
        if str(gate_id).strip()
    }
    needs_review_gate_ids = {
        str(gate_id)
        for gate_id in public_fields.get("operator_validation_needs_review_gate_ids", [])
        if str(gate_id).strip()
    }
    for gate_id, status in safe_share_gate_statuses.items():
        gate_text = str(gate_id).strip()
        status_text = str(status or "").strip()
        if not gate_text:
            continue
        if status_text == "missing_evidence":
            missing_evidence_gate_ids.add(gate_text)
            pending_gate_ids.add(gate_text)
            needs_review_gate_ids.discard(gate_text)
        elif status_text == "needs_review":
            needs_review_gate_ids.add(gate_text)
            pending_gate_ids.discard(gate_text)
            missing_evidence_gate_ids.discard(gate_text)
        elif status_text == "pending_operator_evidence":
            pending_gate_ids.add(gate_text)
            missing_evidence_gate_ids.discard(gate_text)
            needs_review_gate_ids.discard(gate_text)
    all_complete = bool(public_fields.get("operator_validation_all_complete"))
    has_gate_evidence = bool(
        pending_gate_ids
        or missing_evidence_gate_ids
        or needs_review_gate_ids
        or int(public_fields.get("operator_validation_gate_count") or 0)
    )
    fields: dict[str, object] = {}
    for gate_id in DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        missing_evidence = gate_id in missing_evidence_gate_ids
        needs_review = gate_id in needs_review_gate_ids
        pending = gate_id in pending_gate_ids
        passed = not (missing_evidence or needs_review or pending) and (
            all_complete or has_gate_evidence
        )
        if missing_evidence:
            status = "missing_evidence"
        elif needs_review:
            status = "needs_review"
        elif pending:
            status = "pending"
        elif passed:
            status = "passed"
        else:
            status = "unknown"
        prefix = f"operator_validation_gate_{gate_id}"
        fields.update(
            {
                f"{prefix}_status": status,
                f"{prefix}_pending": pending,
                f"{prefix}_missing_evidence": missing_evidence,
                f"{prefix}_needs_review": needs_review,
                f"{prefix}_passed": passed,
            }
        )
    return fields

def _dataset_queue_direct_start_policy_fields(
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = policy if isinstance(policy, Mapping) else _dataset_queue_direct_start_policy()
    return {
        "dataset_queue_direct_start_enabled": bool(source.get("enabled")),
        "dataset_queue_direct_start_method": str(source.get("method") or ""),
        "dataset_queue_direct_start_endpoint_route_template": str(
            source.get("endpoint_route_template") or ""
        ),
        "dataset_queue_direct_start_same_origin_only": bool(source.get("same_origin_only")),
        "dataset_queue_direct_start_exact_route_required": bool(source.get("exact_route_required")),
        "dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id": bool(
            source.get("endpoint_task_segment_must_match_row_task_id")
        ),
        "dataset_queue_direct_start_expected_user_guard_required": bool(
            source.get("expected_user_guard_required")
        ),
        "dataset_queue_direct_start_post_body_expected_user_required": bool(
            source.get("post_body_expected_user_required")
        ),
        "dataset_queue_direct_start_post_body_expected_user_field": str(
            source.get("post_body_expected_user_field") or ""
        ),
        "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract": bool(
            source.get("denied_start_returns_task_open_authorization_contract")
        ),
        "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract": bool(
            source.get("denied_start_support_preserves_task_open_authorization_contract")
        ),
        "dataset_queue_direct_start_denied_start_support_includes_authorization_context": bool(
            source.get("denied_start_support_includes_authorization_context")
        ),
        "dataset_queue_direct_start_denied_start_contract_reports_no_session_created": bool(
            source.get("denied_start_contract_reports_no_session_created")
        ),
        "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false": bool(
            source.get("denied_start_contract_reports_server_authorizes_open_false")
        ),
        "dataset_queue_direct_start_startable_task_states": json.dumps(
            list(source.get("startable_task_states") or [])
        ),
        "dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint": bool(
            source.get("non_startable_tasks_do_not_advertise_endpoint")
        ),
        "dataset_queue_direct_start_label_mutation_target_kind": str(
            source.get("label_mutation_target_kind") or ""
        ),
        "dataset_queue_direct_start_browser_label_write_target": str(
            source.get("browser_label_write_target") or ""
        ),
        "dataset_queue_direct_start_csv_handoff_artifact_role": str(
            source.get("csv_handoff_artifact_role") or ""
        ),
        "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets": bool(
            source.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets": bool(
            source.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets": bool(
            source.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "dataset_queue_direct_start_browser_writes_csv_or_handoff_files": bool(
            source.get("browser_writes_csv_or_handoff_files")
        ),
        "dataset_queue_direct_start_browser_writes_handoff_csv": bool(
            source.get("browser_writes_handoff_csv")
        ),
        "dataset_queue_direct_start_browser_writes_intermediate_csv": bool(
            source.get("browser_writes_intermediate_csv")
        ),
        "dataset_queue_direct_start_browser_receives_zarr_write_authority": bool(
            source.get("browser_receives_zarr_write_authority")
        ),
        "dataset_queue_direct_start_browser_has_direct_zarr_write_authority": bool(
            source.get("browser_has_direct_zarr_write_authority")
        ),
    }

def _runtime_operator_validation_gate_cli_policy_fields(
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = (
        policy
        if isinstance(policy, Mapping)
        else _runtime_operator_validation_gate_cli_policy()
    )
    return {
        "runtime_operator_validation_gate_cli_policy_schema": str(
            source.get("schema") or ""
        ),
        "runtime_operator_validation_gate_cli_policy_validation_checklist_flag": str(
            source.get("validation_checklist_flag") or ""
        ),
        "runtime_operator_validation_gate_cli_policy_preferred_require_flag": str(
            source.get("preferred_require_flag") or ""
        ),
        "runtime_operator_validation_gate_cli_policy_legacy_require_flag": str(
            source.get("legacy_require_flag") or ""
        ),
        "runtime_operator_validation_gate_cli_policy_legacy_require_flag_retained_for_compatibility": bool(
            source.get("legacy_require_flag_retained_for_compatibility")
        ),
        "runtime_operator_validation_gate_cli_policy_config_field": str(
            source.get("config_field") or ""
        ),
        "runtime_operator_validation_gate_cli_policy_requires_validation_checklist": bool(
            source.get("requires_validation_checklist")
        ),
        "runtime_operator_validation_gate_cli_policy_protects_browser_start_open": bool(
            source.get("protects_browser_start_open")
        ),
        "runtime_operator_validation_gate_cli_policy_protects_browser_mutations": bool(
            source.get("protects_browser_mutations")
        ),
        "runtime_operator_validation_gate_cli_policy_blocks_before_session_creation": bool(
            source.get("blocks_before_session_creation")
        ),
        "runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check": bool(
            source.get("blocks_before_target_token_check")
        ),
        "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write": bool(
            source.get("blocks_before_zarr_write")
        ),
        "runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation": bool(
            source.get("blocks_before_audit_event_creation")
        ),
    }

def _direct_browser_start_contract_summary_fields(
    summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = summary if isinstance(summary, Mapping) else {}
    return {
        "direct_browser_start_contract_summary_schema": str(source.get("schema") or ""),
        "direct_browser_start_contract_summary_ready": bool(source.get("ready")),
        "direct_browser_start_contract_summary_task_count": int(source.get("task_count") or 0),
        "direct_browser_start_contract_summary_ready_task_count": int(
            source.get("ready_task_count") or 0
        ),
        "direct_browser_start_contract_summary_not_ready_task_count": int(
            source.get("not_ready_task_count") or 0
        ),
        "direct_browser_start_contract_summary_not_ready_reason_counts": json.dumps(
            source.get("not_ready_reason_counts") or {},
            sort_keys=True,
        ),
        "direct_browser_start_contract_summary_operator_action_counts": json.dumps(
            source.get("operator_action_counts") or {},
            sort_keys=True,
        ),
        "direct_browser_start_contract_summary_expected_user_guard_enforced_by_api": bool(
            source.get("expected_user_guard_enforced_by_api")
        ),
        "direct_browser_start_contract_summary_server_rechecks_on_post": bool(
            source.get("server_rechecks_on_post")
        ),
        "direct_browser_start_contract_summary_label_mutation_target_kind": str(
            source.get("label_mutation_target_kind") or ""
        ),
        "direct_browser_start_contract_summary_browser_label_write_target": str(
            source.get("browser_label_write_target") or ""
        ),
        "direct_browser_start_contract_summary_csv_handoff_artifact_role": str(
            source.get("csv_handoff_artifact_role") or ""
        ),
        "direct_browser_start_contract_summary_csv_handoff_artifacts_are_label_write_targets": bool(
            source.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "direct_browser_start_contract_summary_handoff_csv_artifacts_are_label_write_targets": bool(
            source.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "direct_browser_start_contract_summary_intermediate_csv_artifacts_are_label_write_targets": bool(
            source.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "direct_browser_start_contract_summary_browser_writes_csv_or_handoff_files": bool(
            source.get("browser_writes_csv_or_handoff_files")
        ),
        "direct_browser_start_contract_summary_browser_writes_handoff_csv": bool(
            source.get("browser_writes_handoff_csv")
        ),
        "direct_browser_start_contract_summary_browser_writes_intermediate_csv": bool(
            source.get("browser_writes_intermediate_csv")
        ),
        "direct_browser_start_contract_summary_browser_receives_zarr_write_authority": bool(
            source.get("browser_receives_zarr_write_authority")
        ),
        "direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority": bool(
            source.get("browser_has_direct_zarr_write_authority")
        ),
    }

def _dashboard_roster_rows(
    store: LabelingStore,
    *,
    dashboard_url: str,
    user: str | None = None,
    include_inactive: bool = False,
    include_completed: bool = False,
    require_dashboard_url: bool = True,
    operator_launch_approved: bool = False,
    operator_validation_fields: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    assignments = store.list_assignments(
        assignee_user=user,
        status=None if include_inactive else "active",
    )
    users = sorted(
        {
            str(assignment.get("assignee_user") or "").strip()
            for assignment in assignments
            if str(assignment.get("assignee_user") or "").strip()
        }
    )
    rows: list[dict[str, object]] = []
    operator_validation_fields = dict(
        operator_validation_fields
        if operator_validation_fields is not None
        else _dashboard_operator_validation_fields(operator_launch_approved=operator_launch_approved)
    )
    operator_validation_all_complete = bool(
        operator_validation_fields.get("operator_validation_all_complete")
    )
    operator_validation_status = str(
        operator_validation_fields.get("operator_validation_status") or "pending_operator_evidence"
    )
    operator_validation_public_fields = _operator_validation_public_fields(
        operator_validation_fields
    )
    operator_validation_command_templates = _operator_validation_command_templates(
        operator_validation_public_fields.get(
            "operator_validation_required_missing_evidence_gate_ids"
        )
        if isinstance(
            operator_validation_public_fields.get(
                "operator_validation_required_missing_evidence_gate_ids"
            ),
            list,
        )
        else None
    )
    safe_share_gate = _safe_share_gate_policy()
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        operator_validation_public_fields,
        safe_share_gate=safe_share_gate,
    )
    operator_validation_needs_review_gate_ids = (
        operator_validation_fields.get("operator_validation_needs_review_gate_ids")
        if isinstance(operator_validation_fields.get("operator_validation_needs_review_gate_ids"), list)
        else []
    )
    check_report = _store_consistency_report(store)
    reassignment_session_safety = (
        check_report.get("reassignment_session_safety")
        if isinstance(check_report.get("reassignment_session_safety"), Mapping)
        else {}
    )
    reassignment_session_safety_blocked_recording_ids = _reassignment_session_safety_recording_ids(
        reassignment_session_safety
    )
    for row_user in users:
        work = store.task_summary_for_user(row_user, include_completed=include_completed)
        work["user"] = str(work.get("user") or row_user)
        recordings = work.get("recordings") if isinstance(work.get("recordings"), list) else []
        user_recording_ids = {
            str(recording.get("recording_id") or "").strip()
            for recording in recordings
            if isinstance(recording, Mapping) and str(recording.get("recording_id") or "").strip()
        }
        reassignment_session_safety_blocks_user = (
            isinstance(reassignment_session_safety, Mapping)
            and not bool(reassignment_session_safety.get("ok", True))
            and bool(reassignment_session_safety.get("blocks_labeler_mutation"))
            and (
                not reassignment_session_safety_blocked_recording_ids
                or bool(user_recording_ids & reassignment_session_safety_blocked_recording_ids)
            )
        )
        total_tasks = sum(
            int(recording.get("total_task_count") or 0)
            for recording in recordings
            if isinstance(recording, Mapping)
        )
        complete_tasks = sum(
            int(recording.get("complete_task_count") or 0)
            for recording in recordings
            if isinstance(recording, Mapping)
        )
        workflow_counts: dict[str, int] = {}
        for recording in recordings:
            if not isinstance(recording, Mapping):
                continue
            counts = recording.get("workflow_counts")
            if not isinstance(counts, Mapping):
                continue
            for workflow_kind, count in counts.items():
                workflow_counts[str(workflow_kind)] = workflow_counts.get(str(workflow_kind), 0) + int(count or 0)
        active_recording_count = len(recordings)
        visible_task_count = int(work.get("task_count", 0))
        open_task_count = int(work.get("startable_task_count", 0))
        completion_percent = _completion_percent(complete_tasks, total_tasks)
        completion_state = _completion_state(complete_tasks=complete_tasks, total_tasks=total_tasks)
        invite_reasons: list[str] = []
        if require_dashboard_url and not dashboard_url:
            invite_reasons.append("missing_base_url")
        if active_recording_count <= 0:
            invite_reasons.append("no_active_recordings")
        elif open_task_count <= 0:
            invite_reasons.append("no_open_tasks")
        if active_recording_count > 0 and reassignment_session_safety_blocks_user:
            invite_reasons.append("reassignment_session_safety_failed")
        if not invite_reasons and not operator_validation_all_complete:
            invite_reasons.append(
                "operator_validation_needs_review"
                if operator_validation_needs_review_gate_ids or operator_validation_status == "needs_review"
                else "operator_validation_pending"
            )
        ready_to_invite = not invite_reasons
        labeler_safety = _labeler_safety_policy()
        browser_mutation_write_policy = _browser_mutation_write_policy()
        browser_mutation_write_contract = _browser_mutation_write_contract_policy(browser_mutation_write_policy)
        dataset_queue_direct_start_policy = _dataset_queue_direct_start_policy()
        runtime_gate_cli_policy = _runtime_operator_validation_gate_cli_policy()
        single_owner_policy = _assignment_ownership_policy()
        assignment_ownership_integrity = (
            check_report.get("assignment_ownership_integrity")
            if isinstance(check_report.get("assignment_ownership_integrity"), Mapping)
            else {}
        )
        single_owner_assignment_contract = store.single_owner_assignment_contract()
        assignment_ownership_contract = _assignment_ownership_contract_policy(
            single_owner_policy,
            assignment_ownership_integrity,
            store_single_owner_contract=single_owner_assignment_contract,
        )
        single_owner_policy_contract_met = bool(
            assignment_ownership_contract.get("ready")
        ) and int(assignment_ownership_integrity.get("duplicate_active_owner_count") or 0) == 0
        browser_mutation_target_contract = _browser_mutation_target_contract_summary(
            [
                {
                    "user": row_user,
                    "browser_mutation_write_ready": bool(
                        browser_mutation_write_contract.get("ready")
                    ),
                    "browser_mutation_label_mutation_target_kind": str(
                        browser_mutation_write_contract.get("label_mutation_target_kind")
                        or ""
                    ),
                    "browser_mutation_browser_label_write_target": str(
                        browser_mutation_write_contract.get("browser_label_write_target")
                        or ""
                    ),
                    "browser_mutation_csv_handoff_artifact_role": str(
                        browser_mutation_write_contract.get("csv_handoff_artifact_role")
                        or ""
                    ),
                    "browser_mutation_csv_handoff_artifacts_are_label_write_targets": bool(
                        browser_mutation_write_contract.get(
                            "csv_handoff_artifacts_are_label_write_targets"
                        )
                    ),
                    "browser_mutation_handoff_csv_artifacts_are_label_write_targets": bool(
                        browser_mutation_write_contract.get(
                            "handoff_csv_artifacts_are_label_write_targets"
                        )
                    ),
                    "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": bool(
                        browser_mutation_write_contract.get(
                            "intermediate_csv_artifacts_are_label_write_targets"
                        )
                    ),
                    "browser_mutation_browser_writes_csv_or_handoff_files": bool(
                        browser_mutation_write_policy.get(
                            "browser_writes_csv_or_handoff_files"
                        )
                    ),
                    "browser_mutation_browser_writes_handoff_csv": bool(
                        browser_mutation_write_policy.get("browser_writes_handoff_csv")
                    ),
                    "browser_mutation_browser_writes_intermediate_csv": bool(
                        browser_mutation_write_policy.get(
                            "browser_writes_intermediate_csv"
                        )
                    ),
                    "browser_mutation_browser_has_direct_zarr_write_authority": bool(
                        browser_mutation_write_contract.get(
                            "browser_has_direct_zarr_write_authority"
                        )
                    ),
                }
            ]
        )
        direct_browser_start_contract = _direct_browser_start_contract_summary(
            [
                {
                    "user": row_user,
                    "dataset_queue_direct_start_policy_present": True,
                    **_dataset_queue_direct_start_policy_fields(
                        dataset_queue_direct_start_policy
                    ),
                }
            ]
        )
        operator_recovery_policy = _operator_recovery_policy()
        operator_recovery_fields = _handoff_operator_recovery_fields(
            {"operator_recovery_policy": operator_recovery_policy}
        )
        public_reassignment_session_safety = (
            work.get("reassignment_session_safety")
            if isinstance(work.get("reassignment_session_safety"), Mapping)
            else _public_reassignment_session_safety_fields(
                reassignment_session_safety,
                recording_ids=user_recording_ids,
            )
        )
        no_open_by_reason = _count_recordings_without_open_tasks_by_reason(work)
        labeler_landing_url = _labeler_landing_url_for_base(_dashboard_base_url(dashboard_url)) or (
            "/" if dashboard_url == DASHBOARD_PATH else ""
        )
        labeling_home_url = _labeling_home_url_for_base(_dashboard_base_url(dashboard_url)) or (
            LABELING_HOME_PATH if dashboard_url == DASHBOARD_PATH else ""
        )
        expected_user_labeler_landing_url = _labeler_landing_url_for_dashboard(dashboard_url, row_user)
        expected_user_labeling_home_url = _dashboard_url_for_expected_user(labeling_home_url, row_user)
        expected_user_dashboard_url = _dashboard_url_for_expected_user(dashboard_url, row_user)
        expected_user_identity_probe_url = _identity_probe_url_for_dashboard(dashboard_url, row_user)
        expected_user_dataset_queue_url = _dataset_queue_url_for_dashboard(dashboard_url, row_user)
        expected_user_personal_work_url = _personal_work_url_for_dashboard(dashboard_url, row_user)
        expected_user_personal_dataset_queue_url = _personal_dataset_queue_url_for_dashboard(
            dashboard_url,
            row_user,
        )
        identity_probe_required = bool(labeler_safety["dashboard_identity_check_required"])
        identity_probe_available = bool(expected_user_identity_probe_url)
        _add_work_summary_fields(
            work,
            reassignment_session_safety=reassignment_session_safety,
        )
        _add_direct_start_contracts_to_work_tasks(
            work,
            expected_user=row_user,
            reassignment_session_safety=reassignment_session_safety,
        )
        dataset_queue = (
            work.get("dataset_queue")
            if isinstance(work.get("dataset_queue"), list)
            else []
        )
        dataset_queue_summary = (
            work.get("dataset_queue_summary")
            if isinstance(work.get("dataset_queue_summary"), Mapping)
            else _work_dataset_queue_summary(dataset_queue)
        )
        dataset_queue_state = (
            work.get("dataset_queue_state")
            if isinstance(work.get("dataset_queue_state"), Mapping)
            else _dataset_queue_state(work)
        )
        dataset_queue_blocks_labeler_start = bool(dataset_queue_state.get("blocks_labeler_start"))
        dataset_queue_start_status = "needs_review" if dataset_queue_blocks_labeler_start else "passed"
        dataset_queue_start_operator_action = (
            str(dataset_queue_state.get("operator_action") or "")
            if dataset_queue_blocks_labeler_start
            else ""
        )
        canonical_dataset_queue_preview_url = ""
        dataset_queue_preview_url = ""
        if dashboard_url or not require_dashboard_url:
            canonical_dataset_queue_preview_url = expected_user_dataset_queue_url or _first_dataset_queue_url(
                dataset_queue,
                base_url=_dashboard_base_url(dashboard_url),
            )
            dataset_queue_preview_url = (
                expected_user_personal_dataset_queue_url
                or canonical_dataset_queue_preview_url
            )
        preferred_labeler_entry_url = (
            expected_user_personal_dataset_queue_url
            or expected_user_dataset_queue_url
            or expected_user_labeler_landing_url
            or expected_user_dashboard_url
            or dashboard_url
        )
        personalized_labeler_entry_url = (
            expected_user_personal_dataset_queue_url or preferred_labeler_entry_url
        )
        preferred_labeler_entry_url_matches_dataset_queue = bool(
            preferred_labeler_entry_url
            and preferred_labeler_entry_url
            in {
                expected_user_personal_dataset_queue_url,
                expected_user_dataset_queue_url,
            }
        )
        preferred_labeler_entry_url_matches_personal_dataset_queue = bool(
            expected_user_personal_dataset_queue_url
            and preferred_labeler_entry_url == expected_user_personal_dataset_queue_url
        )
        personalized_labeler_entry_url_matches_personal_dataset_queue = bool(
            expected_user_personal_dataset_queue_url
            and personalized_labeler_entry_url == expected_user_personal_dataset_queue_url
        )
        if (
            not invite_reasons and
            active_recording_count > 0
            and open_task_count > 0
            and not preferred_labeler_entry_url_matches_personal_dataset_queue
        ):
            invite_reasons.append("preferred_personal_queue_mismatch")
        ready_to_invite = not invite_reasons
        known_user_status = _known_labeler_status(store, row_user)
        labeler_route_authorization_policy = _labeler_route_authorization_policy()
        labeler_route_authorization_checklist = (
            _labeler_route_authorization_runtime_checklist(
                policy=labeler_route_authorization_policy,
                user=row_user,
                expected_user=row_user,
                known_user_status=known_user_status,
                assignment_ownership_contract=assignment_ownership_contract,
            )
        )
        rows.append(
            {
                "user": row_user,
                "ready_to_invite": ready_to_invite,
                "ready_to_invite_legacy_semantics": _DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS,
                "ready_state": "ready_to_invite" if ready_to_invite else "not_ready_to_invite",
                "ready_row_state": "ready_row_draft" if ready_to_invite else "diagnostic_note",
                **_dashboard_ready_row_draft_metadata_fields(),
                "invite_reasons": invite_reasons,
                "invite_actions": _dashboard_invite_actions(invite_reasons),
                "copy_label": "Copy ready-row draft" if ready_to_invite else "Copy not-ready note",
                "copy_intent": "ready_row_draft" if ready_to_invite else "diagnostic_note",
                **operator_validation_public_fields,
                **_operator_validation_gate_flat_fields(operator_validation_public_fields),
                **_operator_validation_visibility_fields(),
                "operator_validation_command_templates": operator_validation_command_templates,
                **_operator_validation_command_template_fields(
                    {
                        "operator_validation_command_templates": (
                            operator_validation_command_templates
                        )
                    }
                ),
                "safe_share_gate": safe_share_gate,
                **safe_share_fields,
                **safe_share_checklist_fields,
                "known_user_status": known_user_status,
                **_handoff_known_user_status_fields(
                    {"known_user_status": known_user_status}
                ),
                "labeler_route_authorization_policy": (
                    labeler_route_authorization_policy
                ),
                "labeler_route_authorization_checklist": (
                    labeler_route_authorization_checklist
                ),
                **_labeler_route_authorization_runtime_checklist_compact_fields(
                    labeler_route_authorization_checklist,
                    user=row_user,
                ),
                "dashboard_path": DASHBOARD_PATH,
                "dashboard_url": dashboard_url,
                "single_owner_policy": single_owner_policy,
                **_single_owner_policy_fields(single_owner_policy),
                "assignment_ownership_integrity": assignment_ownership_integrity,
                "assignment_ownership_contract": assignment_ownership_contract,
                **_assignment_ownership_contract_fields(assignment_ownership_contract),
                "single_owner_policy_contract_met": single_owner_policy_contract_met,
                "labeler_landing_url": labeler_landing_url,
                "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
                "labeling_home_url": labeling_home_url,
                "expected_user_labeling_home_url": expected_user_labeling_home_url,
                "expected_user_dashboard_url": expected_user_dashboard_url,
                "expected_user_identity_probe_url": expected_user_identity_probe_url,
                "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
                "expected_user_personal_work_url": expected_user_personal_work_url,
                "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
                "preferred_labeler_entrypoint": (
                    "personal_datasets_waiting_queue"
                    if expected_user_personal_dataset_queue_url
                    else "datasets_waiting_queue"
                ),
                "preferred_labeler_entry_url": preferred_labeler_entry_url,
                "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                "personalized_labeler_entry_url": personalized_labeler_entry_url,
                "preferred_labeler_entry_url_matches_dataset_queue": (
                    preferred_labeler_entry_url_matches_dataset_queue
                ),
                "preferred_labeler_entry_url_matches_personal_dataset_queue": (
                    preferred_labeler_entry_url_matches_personal_dataset_queue
                ),
                "personalized_labeler_entry_url_matches_personal_dataset_queue": (
                    personalized_labeler_entry_url_matches_personal_dataset_queue
                ),
                "labeler_landing_link_role": "queue_first_start",
                "labeling_home_link_role": "human_readable_queue_alias",
                "personal_dataset_queue_link_role": "preferred_queue",
                "dataset_queue_link_role": "canonical_queue_fallback",
                "canonical_dataset_queue_link_role": "canonical_queue_fallback",
                "dashboard_link_role": "fallback_dashboard",
                "identity_probe_link_role": "identity_check",
                "task_links_role": "convenience_entry_hints",
                "identity_probe_required": identity_probe_required,
                "identity_probe_available": identity_probe_available,
                "dashboard_identity_check_required": identity_probe_required,
                "browser_only": bool(labeler_safety["browser_only"]),
                "labeler_runtime_surface": str(labeler_safety["labeler_runtime_surface"]),
                "requires_local_palette_installation": bool(
                    labeler_safety["requires_local_palette_installation"]
                ),
                "requires_local_crimson_installation": bool(
                    labeler_safety["requires_local_crimson_installation"]
                ),
                "requires_local_conda_environment": bool(
                    labeler_safety["requires_local_conda_environment"]
                ),
                "requires_local_project_dependencies": bool(
                    labeler_safety["requires_local_project_dependencies"]
                ),
                "no_direct_zarr_edits": bool(labeler_safety["no_direct_zarr_edits"]),
                "no_forwarding_links_or_handoffs": bool(labeler_safety["no_forwarding_links_or_handoffs"]),
                "browser_mutation_authoritative_label_state": str(
                    browser_mutation_write_policy.get("authoritative_label_state") or ""
                ),
                "browser_mutation_data_plane_write_target": str(
                    browser_mutation_write_contract.get("data_plane_write_target") or ""
                ),
                "browser_mutation_mutable_label_data_plane": str(
                    browser_mutation_write_contract.get("mutable_label_data_plane") or ""
                ),
                "browser_mutation_label_mutation_target_kind": str(
                    browser_mutation_write_contract.get("label_mutation_target_kind") or ""
                ),
                "browser_mutation_browser_label_write_target": str(
                    browser_mutation_write_contract.get("browser_label_write_target") or ""
                ),
                "browser_mutation_handoff_artifacts_are_metadata_only": bool(
                    browser_mutation_write_policy.get("handoff_artifacts_are_metadata_only")
                ),
                "browser_mutation_csv_handoff_artifact_role": str(
                    browser_mutation_write_contract.get("csv_handoff_artifact_role") or ""
                ),
                "browser_mutation_csv_handoff_artifacts_are_label_write_targets": bool(
                    browser_mutation_write_contract.get(
                        "csv_handoff_artifacts_are_label_write_targets"
                    )
                ),
                "browser_mutation_handoff_csv_artifacts_are_label_write_targets": bool(
                    browser_mutation_write_contract.get(
                        "handoff_csv_artifacts_are_label_write_targets"
                    )
                ),
                "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": bool(
                    browser_mutation_write_contract.get(
                        "intermediate_csv_artifacts_are_label_write_targets"
                    )
                ),
                "browser_mutation_training_zarr_mutations_are_server_owned": bool(
                    browser_mutation_write_contract.get("training_zarr_mutations_are_server_owned")
                ),
                "browser_mutation_browser_writes_csv_or_handoff_files": bool(
                    browser_mutation_write_policy.get("browser_writes_csv_or_handoff_files")
                ),
                "browser_mutation_browser_writes_handoff_csv": bool(
                    browser_mutation_write_policy.get("browser_writes_handoff_csv")
                ),
                "browser_mutation_browser_writes_intermediate_csv": bool(
                    browser_mutation_write_policy.get("browser_writes_intermediate_csv")
                ),
                "browser_mutation_browser_receives_zarr_write_authority": bool(
                    browser_mutation_write_contract.get("browser_receives_zarr_write_authority")
                ),
                "browser_mutation_browser_has_direct_zarr_write_authority": bool(
                    browser_mutation_write_contract.get("browser_has_direct_zarr_write_authority")
                ),
                **_handoff_browser_mutation_write_fields(
                    {"browser_mutation_write_policy": browser_mutation_write_policy}
                ),
                "browser_mutation_target_contract_met": bool(
                    browser_mutation_target_contract.get("met")
                ),
                "browser_mutation_target_mismatch_count": int(
                    browser_mutation_target_contract.get("mismatch_count") or 0
                ),
                "browser_mutation_target_mismatch_users": list(
                    browser_mutation_target_contract.get("mismatch_users", [])
                ),
                "dataset_queue_direct_start_policy": dataset_queue_direct_start_policy,
                **_dataset_queue_direct_start_policy_fields(dataset_queue_direct_start_policy),
                "direct_browser_start_contract_met": bool(
                    direct_browser_start_contract.get("met")
                ),
                "direct_browser_start_mismatch_count": int(
                    direct_browser_start_contract.get("mismatch_count") or 0
                ),
                "direct_browser_start_mismatch_users": list(
                    direct_browser_start_contract.get("mismatch_users", [])
                ),
                "runtime_operator_validation_gate_cli_policy": runtime_gate_cli_policy,
                **_runtime_operator_validation_gate_cli_policy_fields(
                    runtime_gate_cli_policy
                ),
                "reassignment_session_safety": _public_reassignment_session_safety_fields(
                    reassignment_session_safety,
                    recording_ids=user_recording_ids,
                ),
                "reassignment_session_safety_blocks_labeler_mutation": bool(
                    public_reassignment_session_safety.get("blocks_labeler_mutation")
                ),
                "reassignment_session_safety_active_session_assignment_mismatch_count": int(
                    public_reassignment_session_safety.get(
                        "active_session_assignment_mismatch_count"
                    )
                    or 0
                ),
                "operator_recovery_policy": operator_recovery_policy,
                **operator_recovery_fields,
                "labeler_safety": labeler_safety,
                "invitation_message": _dashboard_invitation_message(
                    user=row_user,
                    dashboard_url=expected_user_dashboard_url or dashboard_url,
                    identity_probe_url=expected_user_identity_probe_url,
                    labeler_landing_url=expected_user_labeler_landing_url,
                    labeling_home_url=expected_user_labeling_home_url,
                    dataset_queue_url=expected_user_dataset_queue_url,
                    personalized_entry_url=(
                        expected_user_personal_dataset_queue_url
                        or expected_user_dataset_queue_url
                    ),
                    ready_to_invite=ready_to_invite,
                    invite_reasons=invite_reasons,
                ),
                "recordings": active_recording_count,
                "visible_tasks": visible_task_count,
                "open_tasks": open_task_count,
                "total_tasks": total_tasks,
                "complete_tasks": complete_tasks,
                "completion_percent": completion_percent,
                "completion_state": completion_state,
                "dataset_queue_summary": dataset_queue_summary,
                "direct_browser_start_contract_summary": work.get(
                    "direct_browser_start_contract_summary", {}
                ),
                **_direct_browser_start_contract_summary_fields(
                    work.get("direct_browser_start_contract_summary")
                    if isinstance(work.get("direct_browser_start_contract_summary"), Mapping)
                    else None
                ),
                "dataset_queue_state": dataset_queue_state,
                "dataset_queue_state_code": str(dataset_queue_state.get("code") or ""),
                "dataset_queue_state_title": str(dataset_queue_state.get("title") or ""),
                "labeler_work_completion": work.get("labeler_work_completion", {}),
                **_labeler_work_completion_fields(
                    work.get("labeler_work_completion")
                    if isinstance(work.get("labeler_work_completion"), Mapping)
                    else None
                ),
                "dataset_queue_blocks_labeler_start": dataset_queue_blocks_labeler_start,
                "dataset_queue_start_ready": not dataset_queue_blocks_labeler_start,
                "dataset_queue_start_status": dataset_queue_start_status,
                "dataset_queue_start_operator_action": dataset_queue_start_operator_action,
                "dataset_queue_preview_url": dataset_queue_preview_url,
                "canonical_dataset_queue_preview_url": canonical_dataset_queue_preview_url,
                "waiting_datasets": int(dataset_queue_summary.get("waiting_dataset_count") or 0),
                "dataset_open_tasks": int(dataset_queue_summary.get("open_task_count") or 0),
                "recordings_without_open_tasks": _count_recordings_without_open_tasks(work),
                "recordings_without_open_tasks_by_reason": no_open_by_reason,
                "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(no_open_by_reason),
                "workflow_counts": workflow_counts,
            }
        )
    return rows

def _dashboard_invite_reason_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    invite_reason_counts: dict[str, int] = {}
    if not rows:
        invite_reason_counts["no_users"] = 1
    for row in rows:
        reasons = row.get("invite_reasons") if isinstance(row.get("invite_reasons"), list) else []
        for reason in reasons:
            reason_text = str(reason).strip()
            if reason_text:
                invite_reason_counts[reason_text] = invite_reason_counts.get(reason_text, 0) + 1
    return invite_reason_counts

def _dashboard_copy_intent_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        intent = str(row.get("copy_intent") or "").strip() or "unknown"
        counts[intent] = counts.get(intent, 0) + 1
    return dict(sorted(counts.items()))

def _dashboard_ready_state_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("ready_state") or "").strip() or "unknown"
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))

def _dashboard_identity_probe_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    required = 0
    available = 0
    missing_users: list[str] = []
    for row in rows:
        probe_required = bool(row.get("identity_probe_required", row.get("dashboard_identity_check_required")))
        if not probe_required:
            continue
        required += 1
        if str(row.get("expected_user_identity_probe_url") or "").strip():
            available += 1
            continue
        user = str(row.get("user") or "").strip()
        missing_users.append(user or "unknown")
    return {
        "identity_probe_required": required,
        "identity_probe_available": available,
        "identity_probe_missing": max(0, required - available),
        "identity_probe_missing_users": missing_users,
    }

def _dashboard_dataset_queue_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    state_counts: dict[str, int] = {}
    preferred_entrypoint_counts: dict[str, int] = {}
    dataset_queue_link_role_counts: dict[str, int] = {}
    reassignment_blocked_users: list[str] = []
    reassignment_blocked_recording_ids: set[str] = set()
    reassignment_mismatch_count = 0
    direct_start_ready_users: list[str] = []
    direct_start_not_ready_users: list[str] = []
    direct_start_missing_summary_users: list[str] = []
    direct_start_task_count = 0
    direct_start_ready_task_count = 0
    direct_start_not_ready_task_count = 0
    direct_start_not_ready_reason_counts: dict[str, int] = {}
    direct_start_operator_action_counts: dict[str, int] = {}
    personalized_preview_users: list[str] = []
    canonical_preview_users: list[str] = []
    missing_personalized_preview_users: list[str] = []
    preferred_personal_queue_match_users: list[str] = []
    missing_preferred_personal_queue_match_users: list[str] = []
    personalized_personal_queue_match_users: list[str] = []
    missing_personalized_personal_queue_match_users: list[str] = []
    browser_mutation_target_contract_not_met_users: list[str] = []
    direct_browser_start_contract_not_met_users: list[str] = []
    single_owner_policy_contract_not_met_users: list[str] = []
    labeler_route_authorization_runtime_checklist_not_met_users: list[str] = []
    browser_mutation_target_total_mismatch_count = 0
    direct_browser_start_total_mismatch_count = 0
    labeler_route_authorization_runtime_checklist_total_mismatch_count = 0
    def _row_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _row_int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _count_mapping(value: object) -> dict[str, int]:
        source: Mapping[object, object]
        if isinstance(value, Mapping):
            source = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            if not isinstance(parsed, Mapping):
                return {}
            source = parsed
        else:
            return {}
        counts: dict[str, int] = {}
        for key, count in source.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            counts[key_text] = counts.get(key_text, 0) + _row_int(count)
        return counts

    def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
        for key, count in source.items():
            target[key] = target.get(key, 0) + int(count)

    for row in rows:
        user = str(row.get("user") or "").strip() or "unknown"
        code = str(row.get("dataset_queue_state_code") or "")
        if code:
            state_counts[code] = state_counts.get(code, 0) + 1
        preferred_entrypoint = str(row.get("preferred_labeler_entrypoint") or "").strip() or "unknown"
        preferred_entrypoint_counts[preferred_entrypoint] = (
            preferred_entrypoint_counts.get(preferred_entrypoint, 0) + 1
        )
        queue_role = str(row.get("dataset_queue_link_role") or "").strip() or "unknown"
        dataset_queue_link_role_counts[queue_role] = (
            dataset_queue_link_role_counts.get(queue_role, 0) + 1
        )
        personalized_queue_url = str(
            row.get("expected_user_personal_dataset_queue_url")
            or row.get("personalized_labeler_entry_url")
            or ""
        ).strip()
        has_personalized_preview = (
            preferred_entrypoint == "personal_datasets_waiting_queue"
            and str(row.get("personal_dataset_queue_link_role") or "").strip() == "preferred_queue"
            and bool(personalized_queue_url)
            and str(row.get("dataset_queue_preview_url") or "").strip() == personalized_queue_url
        )
        if has_personalized_preview:
            personalized_preview_users.append(user)
        else:
            missing_personalized_preview_users.append(user)
        if str(row.get("canonical_dataset_queue_preview_url") or "").strip():
            canonical_preview_users.append(user)
        preferred_entry_url = str(row.get("preferred_labeler_entry_url") or "").strip()
        personalized_entry_url = str(row.get("personalized_labeler_entry_url") or "").strip()
        preferred_matches_personal_queue = _row_bool(
            row.get("preferred_labeler_entry_url_matches_personal_dataset_queue")
        ) or bool(preferred_entry_url and personalized_queue_url and preferred_entry_url == personalized_queue_url)
        personalized_matches_personal_queue = _row_bool(
            row.get("personalized_labeler_entry_url_matches_personal_dataset_queue")
        ) or bool(
            personalized_entry_url
            and personalized_queue_url
            and personalized_entry_url == personalized_queue_url
        )
        if preferred_matches_personal_queue:
            preferred_personal_queue_match_users.append(user)
        else:
            missing_preferred_personal_queue_match_users.append(user)
        if personalized_matches_personal_queue:
            personalized_personal_queue_match_users.append(user)
        else:
            missing_personalized_personal_queue_match_users.append(user)
        if not _row_bool(row.get("browser_mutation_target_contract_met")):
            browser_mutation_target_contract_not_met_users.append(user)
        browser_mutation_target_total_mismatch_count += _row_int(
            row.get("browser_mutation_target_mismatch_count")
        )
        if not _row_bool(row.get("direct_browser_start_contract_met")):
            direct_browser_start_contract_not_met_users.append(user)
        direct_browser_start_total_mismatch_count += _row_int(
            row.get("direct_browser_start_mismatch_count")
        )
        if not _row_bool(row.get("single_owner_policy_contract_met")):
            single_owner_policy_contract_not_met_users.append(user)
        if not _row_bool(
            row.get("labeler_route_authorization_runtime_checklist_gate_met")
        ):
            labeler_route_authorization_runtime_checklist_not_met_users.append(user)
        labeler_route_authorization_runtime_checklist_total_mismatch_count += _row_int(
            row.get("labeler_route_authorization_runtime_checklist_mismatch_count")
        )
        if bool(row.get("reassignment_session_safety_blocks_labeler_mutation")):
            reassignment_blocked_users.append(user or "unknown")
            reassignment_mismatch_count += int(
                row.get("reassignment_session_safety_active_session_assignment_mismatch_count")
                or 0
            )
            safety = row.get("reassignment_session_safety") if isinstance(row.get("reassignment_session_safety"), Mapping) else {}
            recording_ids = (
                safety.get("active_session_assignment_mismatch_recording_ids")
                if isinstance(safety.get("active_session_assignment_mismatch_recording_ids"), list)
                else []
            )
            for recording_id in recording_ids:
                recording_text = str(recording_id).strip()
                if recording_text:
                    reassignment_blocked_recording_ids.add(recording_text)
        direct_summary_value = row.get("direct_browser_start_contract_summary")
        direct_summary = (
            direct_summary_value
            if isinstance(direct_summary_value, Mapping) and direct_summary_value
            else None
        )
        has_flat_direct_summary = bool(
            str(row.get("direct_browser_start_contract_summary_schema") or "").strip()
        )
        if direct_summary is None and has_flat_direct_summary:
            direct_summary = {}
        if direct_summary is None:
            direct_start_missing_summary_users.append(user)
            direct_start_not_ready_users.append(user)
            direct_start_not_ready_reason_counts[
                "missing_direct_browser_start_contract_summary"
            ] = direct_start_not_ready_reason_counts.get(
                "missing_direct_browser_start_contract_summary", 0
            ) + 1
            continue

        def _direct_value(summary_key: str, flat_key: str) -> object:
            if direct_summary is not None and summary_key in direct_summary:
                return direct_summary.get(summary_key)
            return row.get(flat_key)

        direct_ready = _row_bool(
            _direct_value("ready", "direct_browser_start_contract_summary_ready")
        )
        if direct_ready:
            direct_start_ready_users.append(user)
        else:
            direct_start_not_ready_users.append(user)
        direct_start_task_count += _row_int(
            _direct_value("task_count", "direct_browser_start_contract_summary_task_count")
        )
        direct_start_ready_task_count += _row_int(
            _direct_value(
                "ready_task_count",
                "direct_browser_start_contract_summary_ready_task_count",
            )
        )
        direct_start_not_ready_task_count += _row_int(
            _direct_value(
                "not_ready_task_count",
                "direct_browser_start_contract_summary_not_ready_task_count",
            )
        )
        reason_counts = _count_mapping(
            _direct_value(
                "not_ready_reason_counts",
                "direct_browser_start_contract_summary_not_ready_reason_counts",
            )
        )
        if not direct_ready and not reason_counts:
            reason_counts = {"unknown_direct_browser_start_not_ready": 1}
        _merge_counts(direct_start_not_ready_reason_counts, reason_counts)
        _merge_counts(
            direct_start_operator_action_counts,
            _count_mapping(
                _direct_value(
                    "operator_action_counts",
                    "direct_browser_start_contract_summary_operator_action_counts",
                )
            ),
        )
    return {
        "waiting_datasets": sum(int(row.get("waiting_datasets") or 0) for row in rows),
        "dataset_open_tasks": sum(int(row.get("dataset_open_tasks") or 0) for row in rows),
        "users_with_waiting_datasets": sum(1 for row in rows if int(row.get("waiting_datasets") or 0) > 0),
        "dataset_queue_states": dict(sorted(state_counts.items())),
        "dataset_queue_blocked_start_users": [
            str(row.get("user") or "")
            for row in rows
            if bool(row.get("dataset_queue_blocks_labeler_start"))
        ],
        "dataset_queue_preview_users": [
            str(row.get("user") or "")
            for row in rows
            if str(row.get("dataset_queue_preview_url") or "").strip()
        ],
        "personalized_dataset_queue_preview_users": personalized_preview_users,
        "canonical_dataset_queue_preview_users": canonical_preview_users,
        "missing_personalized_dataset_queue_preview_users": missing_personalized_preview_users,
        "all_users_have_personalized_dataset_queue_preview": (
            bool(rows) and len(personalized_preview_users) == len(rows)
        ),
        "preferred_personal_queue_match_users": preferred_personal_queue_match_users,
        "missing_preferred_personal_queue_match_users": (
            missing_preferred_personal_queue_match_users
        ),
        "all_users_have_preferred_personal_queue_match": (
            bool(rows) and len(preferred_personal_queue_match_users) == len(rows)
        ),
        "personalized_personal_queue_match_users": personalized_personal_queue_match_users,
        "missing_personalized_personal_queue_match_users": (
            missing_personalized_personal_queue_match_users
        ),
        "all_users_have_personalized_personal_queue_match": (
            bool(rows) and len(personalized_personal_queue_match_users) == len(rows)
        ),
        "browser_mutation_target_contract_all_users_met": bool(rows)
        and not browser_mutation_target_contract_not_met_users,
        "browser_mutation_target_contract_not_met_users": (
            browser_mutation_target_contract_not_met_users
        ),
        "browser_mutation_target_contract_not_met_user_count": len(
            browser_mutation_target_contract_not_met_users
        ),
        "browser_mutation_target_total_mismatch_count": (
            browser_mutation_target_total_mismatch_count
        ),
        "direct_browser_start_contract_all_users_met": bool(rows)
        and not direct_browser_start_contract_not_met_users,
        "direct_browser_start_contract_not_met_users": (
            direct_browser_start_contract_not_met_users
        ),
        "direct_browser_start_contract_not_met_user_count": len(
            direct_browser_start_contract_not_met_users
        ),
        "direct_browser_start_total_mismatch_count": direct_browser_start_total_mismatch_count,
        "single_owner_policy_contract_all_users_met": bool(rows)
        and not single_owner_policy_contract_not_met_users,
        "single_owner_policy_contract_not_met_users": (
            single_owner_policy_contract_not_met_users
        ),
        "single_owner_policy_contract_not_met_user_count": len(
            single_owner_policy_contract_not_met_users
        ),
        "labeler_route_authorization_runtime_checklist_gate_all_users_met": bool(rows)
        and not labeler_route_authorization_runtime_checklist_not_met_users,
        "labeler_route_authorization_runtime_checklist_not_met_users": (
            labeler_route_authorization_runtime_checklist_not_met_users
        ),
        "labeler_route_authorization_runtime_checklist_not_met_user_count": len(
            labeler_route_authorization_runtime_checklist_not_met_users
        ),
        "labeler_route_authorization_runtime_checklist_total_mismatch_count": (
            labeler_route_authorization_runtime_checklist_total_mismatch_count
        ),
        "dataset_queue_preferred_entrypoint_counts": dict(sorted(preferred_entrypoint_counts.items())),
        "dataset_queue_link_role_counts": dict(sorted(dataset_queue_link_role_counts.items())),
        "reassignment_session_safety_blocked_users": reassignment_blocked_users,
        "reassignment_session_safety_blocked_user_count": len(reassignment_blocked_users),
        "reassignment_session_safety_mismatch_count": reassignment_mismatch_count,
        "reassignment_session_safety_blocked_recording_ids": sorted(reassignment_blocked_recording_ids),
        "direct_browser_start_contract_ready_users": direct_start_ready_users,
        "direct_browser_start_contract_ready_user_count": len(direct_start_ready_users),
        "direct_browser_start_contract_not_ready_users": direct_start_not_ready_users,
        "direct_browser_start_contract_not_ready_user_count": len(direct_start_not_ready_users),
        "direct_browser_start_contract_missing_summary_users": direct_start_missing_summary_users,
        "direct_browser_start_contract_missing_summary_user_count": len(
            direct_start_missing_summary_users
        ),
        "direct_browser_start_contract_all_users_ready": bool(rows)
        and len(direct_start_ready_users) == len(rows),
        "direct_browser_start_contract_task_count": direct_start_task_count,
        "direct_browser_start_contract_ready_task_count": direct_start_ready_task_count,
        "direct_browser_start_contract_not_ready_task_count": direct_start_not_ready_task_count,
        "direct_browser_start_contract_not_ready_reason_counts": dict(
            sorted(direct_start_not_ready_reason_counts.items())
        ),
        "direct_browser_start_contract_operator_action_counts": dict(
            sorted(direct_start_operator_action_counts.items())
        ),
    }

def _dataset_queue_start_readiness_from_counts(counts: Mapping[str, object]) -> dict[str, object]:
    blocked_users = (
        counts.get("dataset_queue_blocked_start_users")
        if isinstance(counts.get("dataset_queue_blocked_start_users"), list)
        else []
    )
    direct_start_not_ready_users = (
        counts.get("direct_browser_start_contract_not_ready_users")
        if isinstance(counts.get("direct_browser_start_contract_not_ready_users"), list)
        else []
    )
    browser_mutation_target_contract_not_met_users = (
        counts.get("browser_mutation_target_contract_not_met_users")
        if isinstance(counts.get("browser_mutation_target_contract_not_met_users"), list)
        else []
    )
    direct_browser_start_contract_not_met_users = (
        counts.get("direct_browser_start_contract_not_met_users")
        if isinstance(counts.get("direct_browser_start_contract_not_met_users"), list)
        else []
    )
    single_owner_policy_contract_not_met_users = (
        counts.get("single_owner_policy_contract_not_met_users")
        if isinstance(counts.get("single_owner_policy_contract_not_met_users"), list)
        else []
    )
    labeler_route_authorization_runtime_checklist_not_met_users = (
        counts.get("labeler_route_authorization_runtime_checklist_not_met_users")
        if isinstance(
            counts.get("labeler_route_authorization_runtime_checklist_not_met_users"),
            list,
        )
        else []
    )
    states = (
        counts.get("dataset_queue_states")
        if isinstance(counts.get("dataset_queue_states"), Mapping)
        else {}
    )
    ready = not bool(
        blocked_users
        or direct_start_not_ready_users
        or browser_mutation_target_contract_not_met_users
        or direct_browser_start_contract_not_met_users
        or single_owner_policy_contract_not_met_users
        or labeler_route_authorization_runtime_checklist_not_met_users
    )
    return {
        "schema": "palette.web_labeling_dataset_queue_start_readiness.v1",
        "gate_id": "dataset_queue_start_readiness",
        "status": "needs_review" if not ready else "passed",
        "ready": ready,
        "dataset_queue_states": dict(states),
        "dataset_queue_blocked_start_users": [str(user) for user in blocked_users],
        "blocked_start_user_count": len(blocked_users),
        "direct_browser_start_contract_not_ready_users": [
            str(user) for user in direct_start_not_ready_users
        ],
        "direct_browser_start_contract_not_ready_user_count": len(
            direct_start_not_ready_users
        ),
        "direct_browser_start_contract_not_ready_reason_counts": dict(
            counts.get("direct_browser_start_contract_not_ready_reason_counts")
            if isinstance(
                counts.get("direct_browser_start_contract_not_ready_reason_counts"),
                Mapping,
            )
            else {}
        ),
        "browser_mutation_target_contract_all_users_met": bool(
            counts.get("browser_mutation_target_contract_all_users_met")
        ),
        "browser_mutation_target_contract_not_met_users": [
            str(user) for user in browser_mutation_target_contract_not_met_users
        ],
        "browser_mutation_target_contract_not_met_user_count": len(
            browser_mutation_target_contract_not_met_users
        ),
        "browser_mutation_target_total_mismatch_count": int(
            counts.get("browser_mutation_target_total_mismatch_count") or 0
        ),
        "direct_browser_start_contract_all_users_met": bool(
            counts.get("direct_browser_start_contract_all_users_met")
        ),
        "direct_browser_start_contract_not_met_users": [
            str(user) for user in direct_browser_start_contract_not_met_users
        ],
        "direct_browser_start_contract_not_met_user_count": len(
            direct_browser_start_contract_not_met_users
        ),
        "direct_browser_start_total_mismatch_count": int(
            counts.get("direct_browser_start_total_mismatch_count") or 0
        ),
        "direct_browser_start_contract_ready_users": counts.get(
            "direct_browser_start_contract_ready_users",
            [],
        ),
        "direct_browser_start_contract_ready_user_count": int(
            counts.get("direct_browser_start_contract_ready_user_count") or 0
        ),
        "direct_browser_start_contract_not_ready_users": counts.get(
            "direct_browser_start_contract_not_ready_users",
            [],
        ),
        "direct_browser_start_contract_not_ready_user_count": int(
            counts.get("direct_browser_start_contract_not_ready_user_count") or 0
        ),
        "direct_browser_start_contract_all_users_ready": bool(
            counts.get("direct_browser_start_contract_all_users_ready")
        ),
        "direct_browser_start_contract_not_ready_reason_counts": counts.get(
            "direct_browser_start_contract_not_ready_reason_counts",
            {},
        ),
        "direct_browser_start_contract_task_count": int(
            counts.get("direct_browser_start_contract_task_count") or 0
        ),
        "direct_browser_start_contract_ready_task_count": int(
            counts.get("direct_browser_start_contract_ready_task_count") or 0
        ),
        "direct_browser_start_contract_not_ready_task_count": int(
            counts.get("direct_browser_start_contract_not_ready_task_count") or 0
        ),
        "direct_browser_start_contract_operator_action_counts": counts.get(
            "direct_browser_start_contract_operator_action_counts",
            {},
        ),
        "direct_browser_start_contract_missing_summary_users": counts.get(
            "direct_browser_start_contract_missing_summary_users",
            [],
        ),
        "direct_browser_start_contract_missing_summary_user_count": int(
            counts.get("direct_browser_start_contract_missing_summary_user_count") or 0
        ),
        "single_owner_policy_contract_all_users_met": bool(
            counts.get("single_owner_policy_contract_all_users_met")
        ),
        "single_owner_policy_contract_not_met_users": [
            str(user) for user in single_owner_policy_contract_not_met_users
        ],
        "single_owner_policy_contract_not_met_user_count": len(
            single_owner_policy_contract_not_met_users
        ),
        "labeler_route_authorization_runtime_checklist_gate_all_users_met": bool(
            counts.get(
                "labeler_route_authorization_runtime_checklist_gate_all_users_met"
            )
        ),
        "labeler_route_authorization_runtime_checklist_not_met_users": [
            str(user)
            for user in labeler_route_authorization_runtime_checklist_not_met_users
        ],
        "labeler_route_authorization_runtime_checklist_not_met_user_count": len(
            labeler_route_authorization_runtime_checklist_not_met_users
        ),
        "labeler_route_authorization_runtime_checklist_total_mismatch_count": int(
            counts.get(
                "labeler_route_authorization_runtime_checklist_total_mismatch_count"
            )
            or 0
        ),
        "reassignment_session_safety_blocked_users": list(
            counts.get("reassignment_session_safety_blocked_users")
            if isinstance(counts.get("reassignment_session_safety_blocked_users"), list)
            else []
        ),
        "reassignment_session_safety_blocked_user_count": int(
            counts.get("reassignment_session_safety_blocked_user_count") or 0
        ),
        "reassignment_session_safety_mismatch_count": int(
            counts.get("reassignment_session_safety_mismatch_count") or 0
        ),
        "reassignment_session_safety_blocked_recording_ids": list(
            counts.get("reassignment_session_safety_blocked_recording_ids")
            if isinstance(counts.get("reassignment_session_safety_blocked_recording_ids"), list)
            else []
        ),
        "operator_action": (
            "Resolve blocked dataset queue states before inviting labelers; generate or reopen work if labeling should continue, or treat completed assignments as finished."
            if blocked_users
            else "Resolve direct browser start contract blockers before inviting labelers; repair assignments, sessions, or task states until every user's Start links can be authorized server-side."
            if direct_start_not_ready_users
            else "Regenerate or repair dashboard rows so browser label mutations target server-owned task-scoped training Zarrs and CSV/handoff artifacts remain metadata-only."
            if browser_mutation_target_contract_not_met_users
            else "Regenerate or repair dashboard rows so direct browser Start/Open keeps POST-only expected-user guarded task-scoped training-Zarr semantics."
            if direct_browser_start_contract_not_met_users
            else "Repair assignment ownership evidence so each recording has one active labeler before inviting labelers."
            if single_owner_policy_contract_not_met_users
            else "Regenerate or repair dashboard rows so labeler-route authorization runtime checklist evidence proves resolved identity, active assignment, single-owner store proof, server-resolved training-Zarr targets, and no intermediate CSV mutation before inviting labelers."
            if labeler_route_authorization_runtime_checklist_not_met_users
            else "No dataset queue start blockers are visible."
        ),
    }

def _dashboard_completion_state_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("completion_state") or "").strip() or "unknown"
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))

def _dashboard_status_report(
    rows: Sequence[Mapping[str, object]],
    counts: Mapping[str, object],
    warnings: Sequence[Mapping[str, object]],
    operator_validation_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    operator_validation = _operator_validation_public_fields(
        operator_validation_fields or {}
    )
    operator_validation_required_missing_gate_ids = (
        operator_validation.get("operator_validation_required_missing_evidence_gate_ids")
        if isinstance(
            operator_validation.get("operator_validation_required_missing_evidence_gate_ids"),
            list,
        )
        else []
    )
    safe_share_gate = _safe_share_gate_policy()
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        operator_validation,
        safe_share_gate=safe_share_gate,
    )
    operator_validation_approval_scope_fields = _operator_validation_approval_scope_fields()
    return {
        "report_kind": "multi_user_labeling_status",
        "ok": not warnings,
        "warning_count": len(warnings),
        "warning_codes": [str(warning.get("code") or "") for warning in warnings],
        "dashboard_warnings": counts.get("dashboard_warnings", {}),
        "operator_validation": {
            **operator_validation,
            **_operator_validation_gate_flat_fields(operator_validation),
            **operator_validation_approval_scope_fields,
            **safe_share_fields,
            **safe_share_checklist_fields,
        },
        "operator_validation_command_templates": _operator_validation_command_templates(
            operator_validation_required_missing_gate_ids
        ),
        "operator_validation_visibility_policy": _operator_validation_visibility_policy(),
        "operator_recovery_policy": _operator_recovery_policy(),
        "operator_recovery_contract": _operator_recovery_contract_policy(_operator_recovery_policy()),
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "users": int(counts.get("users") or 0),
        "ready_to_invite": int(counts.get("ready_to_invite") or 0),
        "not_ready_to_invite": int(counts.get("not_ready_to_invite") or 0),
        "ready_to_invite_legacy_semantics": _DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS,
        "ready_row_state_values": list(_DASHBOARD_READY_ROW_STATE_VALUES),
        "copy_intent_values": list(_DASHBOARD_COPY_INTENT_VALUES),
        "ready_row_draft_count": int(counts.get("ready_row_draft_count") or 0),
        "diagnostic_note_count": int(counts.get("diagnostic_note_count") or 0),
        "ready_row_draft_users": counts.get("ready_row_draft_users", []),
        "diagnostic_note_users": counts.get("diagnostic_note_users", []),
        "open_tasks": int(counts.get("open_tasks") or 0),
        "waiting_datasets": int(counts.get("waiting_datasets") or 0),
        "dataset_open_tasks": int(counts.get("dataset_open_tasks") or 0),
        "users_with_waiting_datasets": int(counts.get("users_with_waiting_datasets") or 0),
        "dataset_queue_states": counts.get("dataset_queue_states", {}),
        "dataset_queue_preferred_entrypoint_counts": counts.get(
            "dataset_queue_preferred_entrypoint_counts",
            {},
        ),
        "dataset_queue_link_role_counts": counts.get("dataset_queue_link_role_counts", {}),
        "personalized_dataset_queue_preview_users": counts.get(
            "personalized_dataset_queue_preview_users",
            [],
        ),
        "canonical_dataset_queue_preview_users": counts.get(
            "canonical_dataset_queue_preview_users",
            [],
        ),
        "missing_personalized_dataset_queue_preview_users": counts.get(
            "missing_personalized_dataset_queue_preview_users",
            [],
        ),
        "all_users_have_personalized_dataset_queue_preview": bool(
            counts.get("all_users_have_personalized_dataset_queue_preview")
        ),
        "preferred_personal_queue_match_users": counts.get(
            "preferred_personal_queue_match_users",
            [],
        ),
        "missing_preferred_personal_queue_match_users": counts.get(
            "missing_preferred_personal_queue_match_users",
            [],
        ),
        "all_users_have_preferred_personal_queue_match": bool(
            counts.get("all_users_have_preferred_personal_queue_match")
        ),
        "personalized_personal_queue_match_users": counts.get(
            "personalized_personal_queue_match_users",
            [],
        ),
        "missing_personalized_personal_queue_match_users": counts.get(
            "missing_personalized_personal_queue_match_users",
            [],
        ),
        "all_users_have_personalized_personal_queue_match": bool(
            counts.get("all_users_have_personalized_personal_queue_match")
        ),
        "browser_mutation_target_contract_all_users_met": bool(
            counts.get("browser_mutation_target_contract_all_users_met")
        ),
        "browser_mutation_target_contract_not_met_users": counts.get(
            "browser_mutation_target_contract_not_met_users",
            [],
        ),
        "browser_mutation_target_contract_not_met_user_count": int(
            counts.get("browser_mutation_target_contract_not_met_user_count") or 0
        ),
        "browser_mutation_target_total_mismatch_count": int(
            counts.get("browser_mutation_target_total_mismatch_count") or 0
        ),
        "direct_browser_start_contract_all_users_met": bool(
            counts.get("direct_browser_start_contract_all_users_met")
        ),
        "direct_browser_start_contract_not_met_users": counts.get(
            "direct_browser_start_contract_not_met_users",
            [],
        ),
        "direct_browser_start_contract_not_met_user_count": int(
            counts.get("direct_browser_start_contract_not_met_user_count") or 0
        ),
        "direct_browser_start_total_mismatch_count": int(
            counts.get("direct_browser_start_total_mismatch_count") or 0
        ),
        "direct_browser_start_contract_ready_users": counts.get(
            "direct_browser_start_contract_ready_users",
            [],
        ),
        "direct_browser_start_contract_ready_user_count": int(
            counts.get("direct_browser_start_contract_ready_user_count") or 0
        ),
        "direct_browser_start_contract_not_ready_users": counts.get(
            "direct_browser_start_contract_not_ready_users",
            [],
        ),
        "direct_browser_start_contract_not_ready_user_count": int(
            counts.get("direct_browser_start_contract_not_ready_user_count") or 0
        ),
        "direct_browser_start_contract_all_users_ready": bool(
            counts.get("direct_browser_start_contract_all_users_ready")
        ),
        "direct_browser_start_contract_not_ready_reason_counts": counts.get(
            "direct_browser_start_contract_not_ready_reason_counts",
            {},
        ),
        "direct_browser_start_contract_task_count": int(
            counts.get("direct_browser_start_contract_task_count") or 0
        ),
        "direct_browser_start_contract_ready_task_count": int(
            counts.get("direct_browser_start_contract_ready_task_count") or 0
        ),
        "direct_browser_start_contract_not_ready_task_count": int(
            counts.get("direct_browser_start_contract_not_ready_task_count") or 0
        ),
        "direct_browser_start_contract_operator_action_counts": counts.get(
            "direct_browser_start_contract_operator_action_counts",
            {},
        ),
        "direct_browser_start_contract_missing_summary_users": counts.get(
            "direct_browser_start_contract_missing_summary_users",
            [],
        ),
        "direct_browser_start_contract_missing_summary_user_count": int(
            counts.get("direct_browser_start_contract_missing_summary_user_count") or 0
        ),
        "single_owner_policy_contract_all_users_met": bool(
            counts.get("single_owner_policy_contract_all_users_met")
        ),
        "single_owner_policy_contract_not_met_users": counts.get(
            "single_owner_policy_contract_not_met_users",
            [],
        ),
        "single_owner_policy_contract_not_met_user_count": int(
            counts.get("single_owner_policy_contract_not_met_user_count") or 0
        ),
        "labeler_route_authorization_runtime_checklist_gate_all_users_met": bool(
            counts.get(
                "labeler_route_authorization_runtime_checklist_gate_all_users_met"
            )
        ),
        "labeler_route_authorization_runtime_checklist_not_met_users": counts.get(
            "labeler_route_authorization_runtime_checklist_not_met_users",
            [],
        ),
        "labeler_route_authorization_runtime_checklist_not_met_user_count": int(
            counts.get(
                "labeler_route_authorization_runtime_checklist_not_met_user_count"
            )
            or 0
        ),
        "labeler_route_authorization_runtime_checklist_total_mismatch_count": int(
            counts.get(
                "labeler_route_authorization_runtime_checklist_total_mismatch_count"
            )
            or 0
        ),
        "dataset_queue_blocked_start_users": counts.get("dataset_queue_blocked_start_users", []),
        "dataset_queue_start_readiness": counts.get("dataset_queue_start_readiness", {}),
        "reassignment_session_safety_blocked_users": counts.get("reassignment_session_safety_blocked_users", []),
        "reassignment_session_safety_blocked_user_count": int(
            counts.get("reassignment_session_safety_blocked_user_count") or 0
        ),
        "reassignment_session_safety_mismatch_count": int(
            counts.get("reassignment_session_safety_mismatch_count") or 0
        ),
        "reassignment_session_safety_blocked_recording_ids": counts.get(
            "reassignment_session_safety_blocked_recording_ids",
            [],
        ),
        "total_tasks": int(counts.get("total_tasks") or 0),
        "complete_tasks": int(counts.get("complete_tasks") or 0),
        "completion_percent": counts.get("completion_percent"),
        "completion_states": counts.get("completion_states", {}),
        "identity_probe_required": int(counts.get("identity_probe_required") or 0),
        "identity_probe_available": int(counts.get("identity_probe_available") or 0),
        "identity_probe_missing": int(counts.get("identity_probe_missing") or 0),
        "identity_probe_missing_users": counts.get("identity_probe_missing_users", []),
        "invite_reasons": counts.get("invite_reasons", {}),
        "copy_intents": counts.get("copy_intents", {}),
        "user_statuses": [
            {
                "user": str(row.get("user") or ""),
                "ready_to_invite": bool(row.get("ready_to_invite")),
                "ready_to_invite_legacy_semantics": str(
                    row.get("ready_to_invite_legacy_semantics")
                    or _DASHBOARD_READY_TO_INVITE_LEGACY_SEMANTICS
                ),
                "ready_state": str(row.get("ready_state") or ""),
                "ready_row_state": str(row.get("ready_row_state") or ""),
                **_dashboard_ready_row_draft_metadata_fields(row),
                "copy_intent": str(row.get("copy_intent") or ""),
                **_operator_validation_public_fields(row),
                **_operator_validation_gate_flat_fields(row),
                "safe_share_gate_schema": str(row.get("safe_share_gate_schema") or ""),
                "safe_share_gate_id": str(row.get("safe_share_gate_id") or ""),
                "safe_share_requires_require_shareable_inspection": bool(
                    row.get("safe_share_requires_require_shareable_inspection")
                ),
                "safe_share_ready_to_send_is_sufficient": bool(
                    row.get("safe_share_ready_to_send_is_sufficient")
                ),
                "safe_share_required_inspection_field": str(
                    row.get("safe_share_required_inspection_field") or ""
                ),
                "safe_share_required_inspection_value": bool(
                    row.get("safe_share_required_inspection_value")
                ),
                "safe_share_launch_blocking_evidence_gate_ids": (
                    row.get("safe_share_launch_blocking_evidence_gate_ids")
                    if isinstance(row.get("safe_share_launch_blocking_evidence_gate_ids"), list)
                    else []
                ),
                "safe_share_operator_action": str(
                    row.get("safe_share_operator_action") or ""
                ),
                **_safe_share_checklist_field_values(row),
                "completion_state": str(row.get("completion_state") or ""),
                "completion_percent": row.get("completion_percent"),
                "recordings": int(row.get("recordings") or 0),
                "open_tasks": int(row.get("open_tasks") or 0),
                "waiting_datasets": int(row.get("waiting_datasets") or 0),
                "dataset_open_tasks": int(row.get("dataset_open_tasks") or 0),
                "dataset_queue_state": row.get("dataset_queue_state", {}),
                "dataset_queue_state_code": str(row.get("dataset_queue_state_code") or ""),
                "dataset_queue_state_title": str(row.get("dataset_queue_state_title") or ""),
                "labeler_work_completion": row.get("labeler_work_completion", {}),
                **_labeler_work_completion_fields(
                    row.get("labeler_work_completion")
                    if isinstance(row.get("labeler_work_completion"), Mapping)
                    else None
                ),
                "dataset_queue_blocks_labeler_start": bool(row.get("dataset_queue_blocks_labeler_start")),
                "dataset_queue_start_ready": bool(row.get("dataset_queue_start_ready")),
                "dataset_queue_start_status": str(row.get("dataset_queue_start_status") or ""),
                "dataset_queue_start_operator_action": str(row.get("dataset_queue_start_operator_action") or ""),
                "dataset_queue_preview_url": str(row.get("dataset_queue_preview_url") or ""),
                "canonical_dataset_queue_preview_url": str(
                    row.get("canonical_dataset_queue_preview_url") or ""
                ),
                "preferred_labeler_entrypoint": str(row.get("preferred_labeler_entrypoint") or ""),
                "preferred_labeler_entry_url": str(row.get("preferred_labeler_entry_url") or ""),
                "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
                    row.get("preferred_labeler_entry_url_matches_personal_dataset_queue")
                ),
                "personalized_labeler_entrypoint": str(
                    row.get("personalized_labeler_entrypoint") or ""
                ),
                "personalized_labeler_entry_url": str(
                    row.get("personalized_labeler_entry_url") or ""
                ),
                "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
                    row.get(
                        "personalized_labeler_entry_url_matches_personal_dataset_queue"
                    )
                ),
                "personal_dataset_queue_link_role": str(
                    row.get("personal_dataset_queue_link_role") or ""
                ),
                "dataset_queue_link_role": str(row.get("dataset_queue_link_role") or ""),
                "canonical_dataset_queue_link_role": str(
                    row.get("canonical_dataset_queue_link_role") or ""
                ),
                "dashboard_link_role": str(row.get("dashboard_link_role") or ""),
                "identity_probe_link_role": str(row.get("identity_probe_link_role") or ""),
                "task_links_role": str(row.get("task_links_role") or ""),
                "reassignment_session_safety": row.get("reassignment_session_safety", {}),
                "reassignment_session_safety_blocks_labeler_mutation": bool(
                    row.get("reassignment_session_safety_blocks_labeler_mutation")
                ),
                "reassignment_session_safety_active_session_assignment_mismatch_count": int(
                    row.get("reassignment_session_safety_active_session_assignment_mismatch_count")
                    or 0
                ),
                "direct_browser_start_contract_summary": row.get(
                    "direct_browser_start_contract_summary", {}
                ),
                "browser_mutation_target_contract_met": bool(
                    row.get("browser_mutation_target_contract_met")
                ),
                "browser_mutation_target_mismatch_count": int(
                    row.get("browser_mutation_target_mismatch_count") or 0
                ),
                "browser_mutation_target_mismatch_users": (
                    row.get("browser_mutation_target_mismatch_users")
                    if isinstance(row.get("browser_mutation_target_mismatch_users"), list)
                    else []
                ),
                "direct_browser_start_contract_met": bool(
                    row.get("direct_browser_start_contract_met")
                ),
                "direct_browser_start_mismatch_count": int(
                    row.get("direct_browser_start_mismatch_count") or 0
                ),
                "direct_browser_start_mismatch_users": (
                    row.get("direct_browser_start_mismatch_users")
                    if isinstance(row.get("direct_browser_start_mismatch_users"), list)
                    else []
                ),
                "single_owner_policy_contract_met": bool(
                    row.get("single_owner_policy_contract_met")
                ),
                "runtime_operator_validation_gate_cli_policy": (
                    row.get("runtime_operator_validation_gate_cli_policy")
                    if isinstance(
                        row.get("runtime_operator_validation_gate_cli_policy"),
                        Mapping,
                    )
                    else _runtime_operator_validation_gate_cli_policy()
                ),
                **_runtime_operator_validation_gate_cli_policy_fields(
                    row.get("runtime_operator_validation_gate_cli_policy")
                    if isinstance(
                        row.get("runtime_operator_validation_gate_cli_policy"),
                        Mapping,
                    )
                    else None
                ),
                "labeler_route_authorization_policy": (
                    row.get("labeler_route_authorization_policy")
                    if isinstance(row.get("labeler_route_authorization_policy"), Mapping)
                    else _labeler_route_authorization_policy()
                ),
                "labeler_route_authorization_checklist": (
                    row.get("labeler_route_authorization_checklist")
                    if isinstance(
                        row.get("labeler_route_authorization_checklist"),
                        Mapping,
                    )
                    else {}
                ),
                **_labeler_route_authorization_runtime_checklist_compact_fields(
                    row.get("labeler_route_authorization_checklist")
                    if isinstance(
                        row.get("labeler_route_authorization_checklist"),
                        Mapping,
                    )
                    else None,
                    user=str(row.get("user") or ""),
                ),
                **_direct_browser_start_contract_summary_fields(
                    row.get("direct_browser_start_contract_summary")
                    if isinstance(row.get("direct_browser_start_contract_summary"), Mapping)
                    else None
                ),
                "dataset_queue_direct_start_enabled": bool(row.get("dataset_queue_direct_start_enabled")),
                "dataset_queue_direct_start_method": str(row.get("dataset_queue_direct_start_method") or ""),
                "dataset_queue_direct_start_endpoint_route_template": str(
                    row.get("dataset_queue_direct_start_endpoint_route_template") or ""
                ),
                "dataset_queue_direct_start_same_origin_only": bool(
                    row.get("dataset_queue_direct_start_same_origin_only")
                ),
                "dataset_queue_direct_start_exact_route_required": bool(
                    row.get("dataset_queue_direct_start_exact_route_required")
                ),
                "dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id": bool(
                    row.get("dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id")
                ),
                "dataset_queue_direct_start_expected_user_guard_required": bool(
                    row.get("dataset_queue_direct_start_expected_user_guard_required")
                ),
                "dataset_queue_direct_start_post_body_expected_user_required": bool(
                    row.get("dataset_queue_direct_start_post_body_expected_user_required")
                ),
                "dataset_queue_direct_start_post_body_expected_user_field": str(
                    row.get("dataset_queue_direct_start_post_body_expected_user_field") or ""
                ),
                "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract": bool(
                    row.get(
                        "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"
                    )
                ),
                "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract": bool(
                    row.get(
                        "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"
                    )
                ),
                "dataset_queue_direct_start_denied_start_support_includes_authorization_context": bool(
                    row.get(
                        "dataset_queue_direct_start_denied_start_support_includes_authorization_context"
                    )
                ),
                "dataset_queue_direct_start_denied_start_contract_reports_no_session_created": bool(
                    row.get(
                        "dataset_queue_direct_start_denied_start_contract_reports_no_session_created"
                    )
                ),
                "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false": bool(
                    row.get(
                        "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"
                    )
                ),
                "dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint": bool(
                    row.get("dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint")
                ),
                "dataset_queue_direct_start_label_mutation_target_kind": str(
                    row.get("dataset_queue_direct_start_label_mutation_target_kind") or ""
                ),
                "dataset_queue_direct_start_browser_label_write_target": str(
                    row.get("dataset_queue_direct_start_browser_label_write_target") or ""
                ),
                "dataset_queue_direct_start_csv_handoff_artifact_role": str(
                    row.get("dataset_queue_direct_start_csv_handoff_artifact_role") or ""
                ),
                "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets": bool(
                    row.get("dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets")
                ),
                "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets": bool(
                    row.get("dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets")
                ),
                "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets": bool(
                    row.get(
                        "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets"
                    )
                ),
                "dataset_queue_direct_start_browser_writes_csv_or_handoff_files": bool(
                    row.get("dataset_queue_direct_start_browser_writes_csv_or_handoff_files")
                ),
                "dataset_queue_direct_start_browser_writes_handoff_csv": bool(
                    row.get("dataset_queue_direct_start_browser_writes_handoff_csv")
                ),
                "dataset_queue_direct_start_browser_writes_intermediate_csv": bool(
                    row.get("dataset_queue_direct_start_browser_writes_intermediate_csv")
                ),
                "dataset_queue_direct_start_browser_receives_zarr_write_authority": bool(
                    row.get("dataset_queue_direct_start_browser_receives_zarr_write_authority")
                ),
                "dataset_queue_direct_start_browser_has_direct_zarr_write_authority": bool(
                    row.get("dataset_queue_direct_start_browser_has_direct_zarr_write_authority")
                ),
                "operator_recovery_ready": bool(row.get("operator_recovery_ready")),
                "operator_recovery_reassignment_closes_previous_owner_sessions": bool(
                    row.get("operator_recovery_reassignment_closes_previous_owner_sessions")
                ),
                "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update": bool(
                    row.get("operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update")
                ),
                "operator_recovery_reassignment_target_validated_before_session_closure": bool(
                    row.get("operator_recovery_reassignment_target_validated_before_session_closure")
                ),
                "operator_recovery_session_closure_and_assignment_update_atomic": bool(
                    row.get("operator_recovery_session_closure_and_assignment_update_atomic")
                ),
                "operator_recovery_task_reopen_operator_only": bool(
                    row.get("operator_recovery_task_reopen_operator_only")
                ),
                "operator_recovery_failed_promotion_retry_operator_only": bool(
                    row.get("operator_recovery_failed_promotion_retry_operator_only")
                ),
                "operator_recovery_reassignment_session_repair_route": str(
                    row.get("operator_recovery_reassignment_session_repair_route") or ""
                ),
                "operator_recovery_task_state_route": str(row.get("operator_recovery_task_state_route") or ""),
                "operator_recovery_task_repair_route": str(row.get("operator_recovery_task_repair_route") or ""),
                "operator_recovery_audit_event_lookup_route": str(
                    row.get("operator_recovery_audit_event_lookup_route") or ""
                ),
                "operator_recovery_failed_promotion_retry_route": str(
                    row.get("operator_recovery_failed_promotion_retry_route") or ""
                ),
                "operator_recovery_validation_gate": str(row.get("operator_recovery_validation_gate") or ""),
                "total_tasks": int(row.get("total_tasks") or 0),
                "complete_tasks": int(row.get("complete_tasks") or 0),
                "invite_reasons": row.get("invite_reasons", []),
                "next_actions": [
                    *(
                        row.get("invite_actions")
                        if isinstance(row.get("invite_actions"), list)
                        else []
                    ),
                    *(
                        row.get("recordings_without_open_tasks_actions")
                        if isinstance(row.get("recordings_without_open_tasks_actions"), list)
                        else []
                    ),
                ],
            }
            for row in rows
            if isinstance(row, Mapping)
        ],
    }

def _count_rows_by_field(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

def _assignment_operator_status_rows(
    assignments: Sequence[Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tasks_by_recording: dict[str, list[Mapping[str, object]]] = {}
    for task in tasks:
        recording_id = str(task.get("recording_id") or "").strip()
        if not recording_id:
            continue
        tasks_by_recording.setdefault(recording_id, []).append(task)
    rows: list[dict[str, object]] = []
    for assignment in assignments:
        recording_id = str(assignment.get("recording_id") or "").strip()
        status = str(assignment.get("status") or "").strip() or "unknown"
        task_rows = tasks_by_recording.get(recording_id, [])
        total_tasks = len(task_rows)
        complete_tasks = sum(1 for task in task_rows if str(task.get("state") or "") == "complete")
        open_tasks = sum(1 for task in task_rows if str(task.get("state") or "") in LABELER_START_TASK_STATES)
        non_startable_tasks = sum(
            1
            for task in task_rows
            if str(task.get("state") or "") != "complete"
            and str(task.get("state") or "") not in LABELER_START_TASK_STATES
        )
        if status != "active":
            work_state = f"{status}_assignment"
        elif total_tasks <= 0:
            work_state = "blocked_no_tasks"
        elif open_tasks <= 0:
            work_state = "blocked_non_startable_tasks" if non_startable_tasks else "complete"
        else:
            work_state = "active_work"
        rows.append(
            {
                "recording_id": recording_id,
                "assignee_user": str(assignment.get("assignee_user") or ""),
                "assignment_status": status,
                "work_state": work_state,
                "total_tasks": total_tasks,
                "open_tasks": open_tasks,
                "non_startable_tasks": non_startable_tasks,
                "complete_tasks": complete_tasks,
                "completion_percent": _completion_percent(complete_tasks, total_tasks),
                "notes": str(assignment.get("notes") or ""),
            }
        )
    return rows

def _browser_mutation_target_contract_summary(
    handoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    for handoff in handoffs:
        user = str(handoff.get("user") or "").strip()
        for field, expected in _DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES.items():
            actual = handoff.get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "user": user,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                        "missing": field not in handoff,
                    }
                )
    mismatch_users = sorted(
        {
            str(mismatch.get("user") or "")
            for mismatch in mismatches
            if str(mismatch.get("user") or "").strip()
        }
    )
    met = bool(handoffs) and not mismatches
    return {
        "schema": "palette.web_labeling_browser_mutation_target_contract.v1",
        "met": met,
        "required_fields": list(_DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS),
        "required_values": dict(_DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES),
        "mismatch_count": len(mismatches),
        "mismatch_users": mismatch_users,
        "mismatches": mismatches,
        "operator_action": ""
        if met
        else (
            "Regenerate or repair the handoff so browser label mutations target "
            "server-owned task-scoped training Zarrs and handoff/intermediate CSV "
            "artifacts remain metadata-only non-write targets."
        ),
    }

def _direct_browser_start_contract_summary(
    handoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    for handoff in handoffs:
        user = str(handoff.get("user") or "").strip()
        for field, expected in _DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES.items():
            actual = handoff.get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "user": user,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                        "missing": field not in handoff,
                    }
                )
    mismatch_users = sorted(
        {
            str(mismatch.get("user") or "")
            for mismatch in mismatches
            if str(mismatch.get("user") or "").strip()
        }
    )
    met = bool(handoffs) and not mismatches
    return {
        "schema": "palette.web_labeling_direct_browser_start_contract.v1",
        "met": met,
        "required_fields": list(_DASHBOARD_DIRECT_BROWSER_START_FIELDS),
        "required_values": dict(_DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES),
        "mismatch_count": len(mismatches),
        "mismatch_users": mismatch_users,
        "mismatches": mismatches,
        "operator_action": ""
        if met
        else (
            "Regenerate or repair the handoff so direct browser Start/Open uses "
            "POST-only same-origin task-open requests, expected-user server rechecks, "
            "denied-start authorization contracts, task-scoped training-Zarr targets, "
            "and metadata-only non-write CSV/handoff artifacts."
        ),
    }

def _labeler_route_authorization_runtime_checklist_contract_summary(
    handoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required_fields = _shareability_labeler_route_authorization_runtime_checklist_fields()
    required_values = (
        _shareability_labeler_route_authorization_runtime_checklist_required_values()
    )
    mismatches: list[dict[str, object]] = []
    for handoff in handoffs:
        user = str(handoff.get("user") or "").strip()
        for field, expected in required_values.items():
            actual = handoff.get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "user": user,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                        "missing": field not in handoff,
                    }
                )
    mismatch_users = sorted(
        {
            str(mismatch.get("user") or "")
            for mismatch in mismatches
            if str(mismatch.get("user") or "").strip()
        }
    )
    met = bool(handoffs) and not mismatches
    return {
        "schema": (
            "palette.web_labeling_labeler_route_authorization_runtime_checklist_"
            "contract.v1"
        ),
        "met": met,
        "required_fields": list(required_fields),
        "required_values": dict(required_values),
        "mismatch_count": len(mismatches),
        "mismatch_users": mismatch_users,
        "mismatches": mismatches,
        "operator_action": ""
        if met
        else (
            "Regenerate or repair the handoff so labeler-route authorization "
            "runtime checklist evidence is present and ready, single-owner store "
            "proof is ready, assignment integrity is OK, duplicate active owner "
            "count is zero, training-Zarr targets are server resolved, and "
            "intermediate CSV mutation remains rejected."
        ),
    }

def _labeler_route_authorization_runtime_checklist_contract_source_from_checklist(
    checklist: Mapping[str, object] | None,
    *,
    user: str = "",
) -> dict[str, object]:
    source = checklist if isinstance(checklist, Mapping) else {}
    try:
        duplicate_active_owner_count = int(
            source.get("duplicate_active_owner_count") or 0
        )
    except (TypeError, ValueError):
        duplicate_active_owner_count = 0
    return {
        "user": str(user),
        "labeler_route_authorization_runtime_checklist_present": isinstance(
            checklist,
            Mapping,
        ),
        "labeler_route_authorization_runtime_checklist_ready": bool(
            source.get("ready")
        ),
        "labeler_route_authorization_single_owner_store_proof_ready": bool(
            source.get("single_owner_store_proof_ready")
        ),
        "labeler_route_authorization_assignment_ownership_integrity_ok": bool(
            source.get("assignment_ownership_integrity_ok")
        ),
        "labeler_route_authorization_duplicate_active_owner_count": (
            duplicate_active_owner_count
        ),
        "labeler_route_authorization_browser_mutation_target_resolved_server_side": bool(
            source.get("browser_mutation_target_resolved_server_side")
        ),
        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs": bool(
            source.get("labelers_mutate_assigned_training_zarrs")
        ),
        "labeler_route_authorization_labelers_mutate_intermediate_csvs": bool(
            source.get("labelers_mutate_intermediate_csvs")
        ),
    }

def _labeler_route_authorization_runtime_checklist_compact_fields(
    checklist: Mapping[str, object] | None,
    *,
    user: str = "",
) -> dict[str, object]:
    summary = _labeler_route_authorization_runtime_checklist_contract_summary(
        [
            _labeler_route_authorization_runtime_checklist_contract_source_from_checklist(
                checklist,
                user=user,
            )
        ]
    )
    gate = _shareability_labeler_route_authorization_runtime_checklist_gate(summary)
    return {
        "labeler_route_authorization_runtime_checklist_gate": gate,
        "labeler_route_authorization_runtime_checklist_gate_met": bool(
            gate.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": int(
            gate.get("mismatch_count") or 0
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": list(
            gate.get("mismatch_users", [])
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": list(
            gate.get("mismatches", [])
        ),
        "labeler_route_authorization_runtime_checklist_required_fields": list(
            gate.get("required_fields", [])
        ),
        "labeler_route_authorization_runtime_checklist_required_values": dict(
            gate.get("required_values", {})
        ),
    }

def _dashboard_ready_row_draft_metadata_fields(
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = source or {}
    return {
        "ready_row_draft_bundle_schema": str(
            source.get("ready_row_draft_bundle_schema")
            or _DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA
        ),
        "ready_row_draft_bundle_kind": str(
            source.get("ready_row_draft_bundle_kind")
            or _DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND
        ),
        "ready_row_state_values": list(_DASHBOARD_READY_ROW_STATE_VALUES),
        "copy_intent_values": list(_DASHBOARD_COPY_INTENT_VALUES),
        "ready_row_draft_share_rule": str(
            source.get("ready_row_draft_share_rule")
            or _DASHBOARD_READY_ROW_DRAFT_SHARE_RULE
        ),
        "ready_row_draft_requires_safe_share_inspection": bool(
            source.get("ready_row_draft_requires_safe_share_inspection", True)
        ),
        "ready_row_draft_required_safe_share_field": str(
            source.get("ready_row_draft_required_safe_share_field")
            or _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD
        ),
        "ready_row_draft_required_safe_share_value": bool(
            source.get(
                "ready_row_draft_required_safe_share_value",
                _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE,
            )
        ),
    }

def _handoff_known_user_status_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    def flag(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    status = handoff.get("known_user_status") if isinstance(handoff.get("known_user_status"), Mapping) else {}
    has_flat_status = any(
        key in handoff
        for key in (
            "known_labeler",
            "known_user_active_assignment_count",
            "known_user_assignment_count",
            "known_user_readiness",
        )
    )
    has_status = bool(status) or has_flat_status
    is_known = (
        flag(status.get("is_known_labeler"))
        if status
        else flag(handoff.get("known_labeler"))
    )
    active_count = (
        int(status.get("active_assignment_count") or 0)
        if status
        else int(handoff.get("known_user_active_assignment_count") or 0)
    )
    assignment_count = (
        int(status.get("assignment_count") or 0)
        if status
        else int(handoff.get("known_user_assignment_count") or 0)
    )
    if not has_status:
        readiness = "missing_evidence"
        action = "Regenerate the handoff so it includes known-user assignment-store evidence before sharing."
    elif not is_known:
        readiness = "unknown_user"
        action = "Assign at least one recording to this user or confirm the expected login identity before sharing."
    elif active_count <= 0:
        readiness = "no_active_assignment"
        action = "Activate or assign a recording for this user before sharing labeling links."
    else:
        readiness = "passed"
        action = ""
    return {
        "known_user_status_present": has_status,
        "known_labeler": is_known,
        "known_user_active_assignment_count": active_count,
        "known_user_assignment_count": assignment_count,
        "known_user_readiness": readiness,
        "known_user_operator_action": action,
    }

def _handoff_operator_recovery_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    policy = (
        handoff.get("operator_recovery_policy")
        if isinstance(handoff.get("operator_recovery_policy"), Mapping)
        else None
    )
    present = policy is not None
    contract = _operator_recovery_contract_policy(policy or {})
    ready = present and bool(contract.get("ready"))
    readiness = "passed" if ready else ("not_ready" if present else "missing")
    action = (
        ""
        if ready
        else (
            "Regenerate or repair the handoff so operator recovery policy metadata includes reassignment, reopen, closure inspection, retry, and backup/rollback controls."
            if present
            else "Regenerate the handoff so operator recovery policy metadata is present before broad launch."
        )
    )
    return {
        "operator_recovery_policy_present": present,
        "operator_recovery_ready": ready,
        "operator_recovery_readiness": readiness,
        "operator_recovery_operator_action": action,
        "operator_recovery_contract_ready": bool(contract.get("ready")),
        "operator_recovery_reassignment_closes_previous_owner_sessions": bool(
            contract.get("reassignment_closes_previous_owner_sessions")
        ),
        "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update": bool(
            contract.get("reassignment_closes_previous_owner_sessions_before_assignment_update")
        ),
        "operator_recovery_reassignment_target_validated_before_session_closure": bool(
            contract.get("reassignment_target_validated_before_session_closure")
        ),
        "operator_recovery_session_closure_and_assignment_update_atomic": bool(
            contract.get("session_closure_and_assignment_update_atomic")
        ),
        "operator_recovery_task_reopen_operator_only": bool(contract.get("task_reopen_operator_only")),
        "operator_recovery_completion_closes_open_sessions": bool(
            contract.get("completion_closes_open_sessions")
        ),
        "operator_recovery_failed_promotion_retry_operator_only": bool(
            contract.get("failed_promotion_retry_operator_only")
        ),
        "operator_recovery_session_closure_events_operator_inspectable": bool(
            contract.get("session_closure_events_operator_inspectable")
        ),
        "operator_recovery_operator_repair_closes_or_supersedes_sessions": bool(
            contract.get("operator_repair_closes_or_supersedes_sessions")
        ),
        "operator_recovery_rollback_requires_backup_plan": bool(
            contract.get("rollback_requires_backup_plan")
        ),
        "operator_recovery_bad_disposable_mutation_recovery_ready": bool(
            contract.get("bad_disposable_mutation_recovery_ready")
        ),
        "operator_recovery_disposable_mutation_smoke_requires_recovery_path_verification": bool(
            contract.get("disposable_mutation_smoke_requires_recovery_path_verification")
        ),
        "operator_recovery_restore_pauses_or_unassigns_recording_before_write": bool(
            contract.get("restore_pauses_or_unassigns_recording_before_write")
        ),
        "operator_recovery_labelers_receive_recovery_write_authority": bool(
            contract.get("labelers_receive_recovery_write_authority")
        ),
        "operator_recovery_browser_recovery_mutations_direct": bool(
            contract.get("browser_recovery_mutations_direct")
        ),
        "operator_recovery_reassignment_session_repair_route": str(
            contract.get("reassignment_session_repair_route") or ""
        ),
        "operator_recovery_task_state_route": str(contract.get("task_state_route") or ""),
        "operator_recovery_task_repair_route": str(contract.get("task_repair_route") or ""),
        "operator_recovery_audit_event_lookup_route": str(
            contract.get("audit_event_lookup_route") or ""
        ),
        "operator_recovery_failed_promotion_retry_route": str(
            contract.get("failed_promotion_retry_route") or ""
        ),
        "operator_recovery_validation_gate": str(contract.get("validation_gate") or ""),
    }

def _handoff_browser_mutation_write_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    policy = (
        handoff.get("browser_mutation_write_policy")
        if isinstance(handoff.get("browser_mutation_write_policy"), Mapping)
        else None
    )
    contract = _browser_mutation_write_contract_policy(policy or {})
    present = policy is not None
    ready = present and bool(contract.get("ready"))
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so browser mutation write policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so browser saves target server-owned assigned task/training Zarr scope "
            "with task-scoped training Zarrs as the mutable label data plane and label mutation target kind, "
            "CSV/HTML/JSON/handoff artifacts remain metadata-only control-plane non-label-write targets, "
            "and browsers write neither handoff CSVs nor intermediate CSVs."
        )
    return {
        "browser_mutation_write_policy_present": present,
        "browser_mutation_write_ready": ready,
        "browser_mutation_write_readiness": readiness,
        "browser_mutation_write_operator_action": action,
        "browser_mutation_authoritative_label_state": str(policy.get("authoritative_label_state") or "") if policy else "",
        "browser_mutation_data_plane_write_target": str(contract.get("data_plane_write_target") or ""),
        "browser_mutation_mutable_label_data_plane": str(contract.get("mutable_label_data_plane") or ""),
        "browser_mutation_label_mutation_target_kind": str(
            contract.get("label_mutation_target_kind") or ""
        ),
        "browser_mutation_browser_label_write_target": str(
            contract.get("browser_label_write_target") or ""
        ),
        "browser_mutation_server_mutates_task_scoped_zarr_targets": bool(
            contract.get("server_mutates_task_scoped_zarr_targets")
        ),
        "browser_mutation_training_zarr_mutations_are_server_owned": bool(
            contract.get("training_zarr_mutations_are_server_owned")
        ),
        "browser_mutation_promotion_training_zarr_requires_task_scope": bool(
            policy.get("promotion_training_zarr_mutation_requires_task_scope")
        )
        if policy
        else False,
        "browser_mutation_handoff_artifacts_are_metadata_only": bool(
            contract.get("handoff_artifacts_are_metadata_only")
        ),
        "browser_mutation_csv_handoff_artifact_role": str(
            contract.get("csv_handoff_artifact_role") or ""
        ),
        "browser_mutation_csv_handoff_artifacts_are_label_write_targets": bool(
            contract.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "browser_mutation_handoff_csv_artifacts_are_label_write_targets": bool(
            contract.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": bool(
            contract.get("intermediate_csv_artifacts_are_label_write_targets")
        ),
        "browser_mutation_browser_writes_csv_or_handoff_files": bool(
            contract.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_mutation_browser_writes_handoff_csv": bool(
            contract.get("browser_writes_handoff_csv")
        ),
        "browser_mutation_browser_writes_intermediate_csv": bool(
            contract.get("browser_writes_intermediate_csv")
        ),
        "browser_mutation_browser_receives_zarr_write_authority": bool(
            contract.get("browser_receives_zarr_write_authority")
        ),
        "browser_mutation_browser_has_direct_zarr_write_authority": bool(
            contract.get("browser_has_direct_zarr_write_authority")
        ),
    }

def _count_recordings_without_open_tasks(work: Mapping[str, object]) -> int:
    recordings = work.get("recordings") if isinstance(work.get("recordings"), list) else []
    return sum(
        1
        for recording in recordings
        if isinstance(recording, Mapping)
        and int(recording.get("startable_task_count") or 0) <= 0
    )

def _no_open_task_reason_for_recording(recording: Mapping[str, object]) -> str:
    reason = str(recording.get("no_open_task_reason") or "").strip()
    if reason:
        return reason
    total = int(recording.get("total_task_count") or 0)
    complete = int(recording.get("complete_task_count") or 0)
    incomplete = int(recording.get("incomplete_task_count") or 0)
    if total > 0 and complete >= total:
        return "all_tasks_complete"
    if incomplete > 0:
        return "non_startable_task_state"
    if total > 0:
        return "no_open_tasks_in_current_summary"
    return "tasks_not_generated"

def _count_recordings_without_open_tasks_by_reason(work: Mapping[str, object]) -> dict[str, int]:
    recordings = work.get("recordings") if isinstance(work.get("recordings"), list) else []
    counts: dict[str, int] = {}
    for recording in recordings:
        if not isinstance(recording, Mapping):
            continue
        if int(recording.get("startable_task_count") or 0) > 0:
            continue
        reason = _no_open_task_reason_for_recording(recording)
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))

def _recordings_without_open_tasks_actions(by_reason: Mapping[str, object] | Iterable[str]) -> list[str]:
    guidance = {
        "tasks_not_generated": "Generate or import browser-labeling tasks for assigned recordings before inviting the labeler.",
        "all_tasks_complete": "Reopen a completed task only if more labeling work is required; otherwise treat the recording as finished.",
        "no_open_tasks_in_current_summary": "Inspect filters, task states, and assignment status; the assignment exists but no open task is currently visible.",
        "non_startable_task_state": "Move an assigned task to pending/in_progress or reopen/regenerate work before inviting the labeler.",
        "unknown": "Inspect the assignment/task summary to determine why no open task is visible.",
    }
    reasons = by_reason.keys() if isinstance(by_reason, Mapping) else by_reason
    actions: list[str] = []
    for reason in reasons:
        reason_text = str(reason or "").strip()
        if not reason_text:
            continue
        actions.append(guidance.get(reason_text, f"Inspect no-open-task reason {reason_text}."))
    return actions

def _validation_gate_kind(gate_id: str) -> str:
    return (
        "operator_evidence"
        if str(gate_id or "").strip() in OPERATOR_EVIDENCE_VALIDATION_GATE_IDS
        else "generated_contract"
    )

def _validation_gate_classification(gates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    operator_evidence_gate_ids: list[str] = []
    generated_contract_gate_ids: list[str] = []
    operator_evidence_pending_gate_ids: list[str] = []
    operator_evidence_needs_review_gate_ids: list[str] = []
    operator_evidence_complete_gate_ids: list[str] = []
    generated_contract_failed_gate_ids: list[str] = []
    generated_contract_passed_gate_ids: list[str] = []
    for gate in gates:
        gate_id = str(gate.get("id") or "").strip()
        if not gate_id:
            continue
        status = str(gate.get("status") or "").strip() or "unknown"
        kind = _validation_gate_kind(gate_id)
        if kind == "operator_evidence":
            operator_evidence_gate_ids.append(gate_id)
            if status == "pending_operator_evidence":
                operator_evidence_pending_gate_ids.append(gate_id)
            elif status == "needs_review":
                operator_evidence_needs_review_gate_ids.append(gate_id)
            elif status in {"passed", "not_applicable"}:
                operator_evidence_complete_gate_ids.append(gate_id)
        else:
            generated_contract_gate_ids.append(gate_id)
            if status == "needs_review":
                generated_contract_failed_gate_ids.append(gate_id)
            elif status in {"passed", "not_applicable"}:
                generated_contract_passed_gate_ids.append(gate_id)
    return {
        "operator_evidence_gate_ids": operator_evidence_gate_ids,
        "generated_contract_gate_ids": generated_contract_gate_ids,
        "operator_evidence_pending_gate_ids": operator_evidence_pending_gate_ids,
        "operator_evidence_needs_review_gate_ids": operator_evidence_needs_review_gate_ids,
        "operator_evidence_complete_gate_ids": operator_evidence_complete_gate_ids,
        "generated_contract_failed_gate_ids": generated_contract_failed_gate_ids,
        "generated_contract_passed_gate_ids": generated_contract_passed_gate_ids,
        "operator_evidence_gate_count": len(operator_evidence_gate_ids),
        "generated_contract_gate_count": len(generated_contract_gate_ids),
        "operator_evidence_pending_gate_count": len(operator_evidence_pending_gate_ids),
        "operator_evidence_needs_review_gate_count": len(operator_evidence_needs_review_gate_ids),
        "operator_evidence_complete_gate_count": len(operator_evidence_complete_gate_ids),
        "generated_contract_failed_gate_count": len(generated_contract_failed_gate_ids),
        "generated_contract_passed_gate_count": len(generated_contract_passed_gate_ids),
    }

def _shareability_labeler_route_authorization_runtime_checklist_fields() -> list[str]:
    return [
        "labeler_route_authorization_runtime_checklist_present",
        "labeler_route_authorization_runtime_checklist_ready",
        "labeler_route_authorization_single_owner_store_proof_ready",
        "labeler_route_authorization_assignment_ownership_integrity_ok",
        "labeler_route_authorization_duplicate_active_owner_count",
        "labeler_route_authorization_browser_mutation_target_resolved_server_side",
        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs",
        "labeler_route_authorization_labelers_mutate_intermediate_csvs",
    ]

def _shareability_labeler_route_authorization_runtime_checklist_required_values() -> dict[str, object]:
    return {
        "labeler_route_authorization_runtime_checklist_present": True,
        "labeler_route_authorization_runtime_checklist_ready": True,
        "labeler_route_authorization_single_owner_store_proof_ready": True,
        "labeler_route_authorization_assignment_ownership_integrity_ok": True,
        "labeler_route_authorization_duplicate_active_owner_count": 0,
        "labeler_route_authorization_browser_mutation_target_resolved_server_side": True,
        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs": True,
        "labeler_route_authorization_labelers_mutate_intermediate_csvs": False,
    }

def _shareability_labeler_route_authorization_runtime_checklist_gate(
    summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = summary if isinstance(summary, Mapping) else {}
    required_fields = list(
        source.get("required_fields")
        if isinstance(source.get("required_fields"), list)
        else _shareability_labeler_route_authorization_runtime_checklist_fields()
    )
    required_values = dict(
        source.get("required_values")
        if isinstance(source.get("required_values"), Mapping)
        else _shareability_labeler_route_authorization_runtime_checklist_required_values()
    )
    mismatches = list(
        source.get("mismatches") if isinstance(source.get("mismatches"), list) else []
    )
    mismatch_users = list(
        source.get("mismatch_users")
        if isinstance(source.get("mismatch_users"), list)
        else []
    )
    return {
        "schema": (
            "palette.web_labeling_labeler_route_authorization_runtime_checklist_"
            "shareability_gate.v1"
        ),
        "gate_id": "labeler_route_authorization_runtime_checklist_gate",
        "met": bool(source.get("met")),
        "checklist_schema": (
            "palette.web_labeling_labeler_route_authorization_runtime_checklist.v1"
        ),
        "checklist_field": "labeler_route_authorization_checklist",
        "required_fields": list(required_fields),
        "required_field_count": len(required_fields),
        "required_values": dict(required_values),
        "required_value_count": len(required_values),
        "mismatch_count": int(source.get("mismatch_count") or len(mismatches)),
        "mismatch_users": mismatch_users,
        "mismatches": mismatches,
        "required_fields_source": (
            "shareability_labeler_route_authorization_runtime_checklist_fields"
        ),
        "required_values_source": (
            "shareability_labeler_route_authorization_runtime_checklist_required_values"
        ),
        "fail_closed_when_missing": True,
        "safe_share_blocking_reason_id": "labeler_route_authorization_policy_not_ready",
        "requires_runtime_checklist_present": True,
        "requires_runtime_checklist_ready": True,
        "requires_single_owner_store_proof_ready": True,
        "required_duplicate_active_owner_count": 0,
        "operator_action": str(source.get("operator_action") or ""),
    }

def _validation_checklist_gate_summary(payload: Mapping[str, object]) -> dict[str, object]:
    gates = payload.get("gates") if isinstance(payload.get("gates"), list) else []
    status_counts: dict[str, int] = {}
    gate_statuses: list[dict[str, object]] = []
    needs_review_gate_ids: list[str] = []
    pending_gate_ids: list[str] = []
    required_pending_gate_ids: list[str] = []
    evidence_recorded_gate_ids: list[str] = []
    required_missing_evidence_gate_ids: list[str] = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        gate_id = str(gate.get("id") or "").strip()
        status = str(gate.get("status") or "").strip() or "unknown"
        required = bool(gate.get("required", True))
        kind = _validation_gate_kind(gate_id)
        evidence_entries = gate.get("evidence") if isinstance(gate.get("evidence"), list) else []
        evidence_notes = gate.get("evidence_notes") if isinstance(gate.get("evidence_notes"), list) else []
        evidence_files = gate.get("evidence_files") if isinstance(gate.get("evidence_files"), list) else []
        evidence_recorded = bool(
            evidence_entries
            or evidence_notes
            or str(gate.get("evidence_recorded_at_utc") or "").strip()
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        gate_statuses.append(
            {
                "id": gate_id,
                "status": status,
                "required": required,
                "blocks_invitation": bool(gate.get("blocks_invitation", True)),
                "blocks_invitation_legacy_semantics": str(
                    gate.get("blocks_invitation_legacy_semantics")
                    or _VALIDATION_GATE_BLOCKS_INVITATION_LEGACY_SEMANTICS
                ),
                "blocks_invitation_is_safe_share_approval": bool(
                    gate.get("blocks_invitation_is_safe_share_approval", False)
                ),
                "blocks_invitation_safe_share_field": str(
                    gate.get("blocks_invitation_safe_share_field")
                    or _VALIDATION_GATE_BLOCKS_INVITATION_SAFE_SHARE_FIELD
                ),
                "gate_kind": kind,
                "operator_evidence_gate": kind == "operator_evidence",
                "generated_contract_gate": kind == "generated_contract",
                "evidence_recorded": evidence_recorded,
                "evidence_count": len(evidence_entries),
                "evidence_file_count": len(evidence_files),
                "evidence_recorded_at_utc": str(gate.get("evidence_recorded_at_utc") or ""),
                "evidence_recorded_by": str(gate.get("evidence_recorded_by") or ""),
            }
        )
        if evidence_recorded:
            evidence_recorded_gate_ids.append(gate_id)
        elif (
            required
            and kind == "operator_evidence"
            and status in {"needs_review", "pending_operator_evidence"}
        ):
            required_missing_evidence_gate_ids.append(gate_id)
        if status == "needs_review":
            needs_review_gate_ids.append(gate_id)
        if status == "pending_operator_evidence":
            pending_gate_ids.append(gate_id)
            if required:
                required_pending_gate_ids.append(gate_id)
    gate_classification = _validation_gate_classification(
        [gate for gate in gates if isinstance(gate, Mapping)]
    )
    operator_validation_command_templates = _operator_validation_command_templates(
        [
            str(gate_id)
            for gate_id in [
                *gate_classification["operator_evidence_pending_gate_ids"],
                *gate_classification["operator_evidence_needs_review_gate_ids"],
            ]
            if str(gate_id).strip()
        ]
    )
    safe_share_gate = (
        payload.get("safe_share_gate")
        if isinstance(payload.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields(
        gates=gates,
        safe_share_gate=safe_share_gate,
    )
    return {
        "schema": str(payload.get("schema") or ""),
        "bundle_label": str(payload.get("bundle_label") or ""),
        "operator_validation_visibility_policy": (
            payload.get("operator_validation_visibility_policy")
            if isinstance(payload.get("operator_validation_visibility_policy"), Mapping)
            else _operator_validation_visibility_policy()
        ),
        "operator_validation_command_templates": operator_validation_command_templates,
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "base_url": str(payload.get("base_url") or ""),
        "labeler_landing_page_path": str(payload.get("labeler_landing_page_path") or ""),
        "labeler_landing_url": str(payload.get("labeler_landing_url") or ""),
        "expected_user_labeler_landing_url": str(payload.get("expected_user_labeler_landing_url") or ""),
        "dashboard_url": str(payload.get("dashboard_url") or ""),
        "expected_user_dashboard_url": str(payload.get("expected_user_dashboard_url") or ""),
        "expected_user_identity_probe_url": str(payload.get("expected_user_identity_probe_url") or ""),
        "dataset_queue_page_path": str(payload.get("dataset_queue_page_path") or ""),
        "dataset_queue_url": str(payload.get("dataset_queue_url") or ""),
        "expected_user_dataset_queue_url": str(payload.get("expected_user_dataset_queue_url") or ""),
        "dry_run": bool(payload.get("dry_run")),
        "all_validation_complete": bool(payload.get("all_validation_complete")),
        "ready_for_operator_validation": bool(payload.get("ready_for_operator_validation")),
        "gate_count": len(gate_statuses),
        "status_counts": dict(sorted(status_counts.items())),
        "validation_gate_classification": gate_classification,
        "operator_evidence_gate_ids": gate_classification["operator_evidence_gate_ids"],
        "generated_contract_gate_ids": gate_classification["generated_contract_gate_ids"],
        "operator_evidence_pending_gate_ids": gate_classification[
            "operator_evidence_pending_gate_ids"
        ],
        "operator_evidence_needs_review_gate_ids": gate_classification[
            "operator_evidence_needs_review_gate_ids"
        ],
        "generated_contract_failed_gate_ids": gate_classification[
            "generated_contract_failed_gate_ids"
        ],
        "needs_review_gate_ids": needs_review_gate_ids,
        "pending_gate_ids": pending_gate_ids,
        "required_pending_gate_ids": required_pending_gate_ids,
        "evidence_recorded_gate_ids": evidence_recorded_gate_ids,
        "required_missing_evidence_gate_ids": required_missing_evidence_gate_ids,
        "evidence_recorded_gate_count": len(evidence_recorded_gate_ids),
        "required_missing_evidence_gate_count": len(required_missing_evidence_gate_ids),
        "gates": gate_statuses,
    }

def _operator_validation_invitation_fields(payload: Mapping[str, object]) -> dict[str, object]:
    summary = _validation_checklist_gate_summary(payload)
    needs_review_gate_ids = [
        str(gate_id)
        for gate_id in summary.get("needs_review_gate_ids", [])
        if str(gate_id).strip()
    ]
    required_pending_gate_ids = [
        str(gate_id)
        for gate_id in summary.get("required_pending_gate_ids", [])
        if str(gate_id).strip()
    ]
    required_missing_evidence_gate_ids = [
        str(gate_id)
        for gate_id in summary.get("required_missing_evidence_gate_ids", [])
        if str(gate_id).strip()
    ]
    gate_count = int(summary.get("gate_count") or 0)
    declared_all_complete = bool(summary.get("all_validation_complete"))
    all_complete = (
        declared_all_complete
        and gate_count > 0
        and not needs_review_gate_ids
        and not required_pending_gate_ids
        and not required_missing_evidence_gate_ids
    )
    if all_complete:
        status = "passed"
        action = ""
    elif needs_review_gate_ids:
        status = "needs_review"
        action = "Resolve validation checklist gates marked needs_review before inviting labelers."
    else:
        status = "pending_operator_evidence"
        action = "Complete required operator validation evidence before inviting labelers to start or save work."
    return {
        "operator_validation_required_before_invite": True,
        "operator_validation_all_complete": all_complete,
        "operator_validation_declared_all_complete": declared_all_complete,
        "operator_validation_gate_count": gate_count,
        "operator_validation_ready_for_operator_validation": bool(
            summary.get("ready_for_operator_validation")
        ),
        "operator_validation_status": status,
        "operator_validation_pending_gate_ids": required_pending_gate_ids,
        "operator_validation_needs_review_gate_ids": needs_review_gate_ids,
        "operator_validation_required_missing_evidence_gate_ids": required_missing_evidence_gate_ids,
        "operator_validation_required_pending_gate_count": len(required_pending_gate_ids),
        "operator_validation_needs_review_gate_count": len(needs_review_gate_ids),
        "operator_validation_required_missing_evidence_gate_count": len(
            required_missing_evidence_gate_ids
        ),
        "operator_validation_operator_action": action,
        **_safe_share_checklist_field_values(summary),
    }
