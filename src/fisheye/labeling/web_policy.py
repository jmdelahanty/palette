"""Pure browser policy and contract payload builders for web labeling."""

from __future__ import annotations

from typing import Mapping, Sequence

from .assignment_store import LABELER_START_TASK_STATES
from .web_auth import DASHBOARD_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH


BROWSER_MUTATION_TARGET_SELECTOR_KEYS: tuple[str, ...] = (
    "position",
    "roi_idx",
    "row_idx",
    "frame_idx",
    "parent_frame_index",
    "source_frame_index",
    "component_idx",
    "component_name",
    "target_zarr",
    "zarr_target",
    "zarr_path",
    "training_zarr",
    "analysis_zarr",
    "target_csv",
    "csv_target",
    "csv_path",
    "handoff_csv",
    "intermediate_csv",
    "output_csv",
    "write_target",
    "data_plane_write_target",
    "label_write_target",
    "browser_label_write_target",
    "target_store",
    "target_uri",
)

LABELING_HOME_PATH = "/labeling"

PERSONAL_WORK_PATH = "/my-work"

IDENTITY_PROBE_PATH = "/identity"

BROWSER_TASK_STATE_POLICY: dict[str, object] = {
    "startable_task_states": list(LABELER_START_TASK_STATES),
    "completed_tasks_read_only": True,
    "completed_tasks_open_new_sessions": False,
    "completed_task_open_requests": "reject_task_complete",
    "completed_task_save_requests": "reject_task_complete",
    "non_startable_task_open_requests": "reject_task_not_startable",
    "non_startable_task_save_requests": "reject_task_not_startable",
    "absolute_navigation_out_of_scope": "reject_nav_error",
    "browser_mutation_target_selectors": "server_owned_reject_client_fields",
    "browser_mutation_target_token": "required_current_target_token",
    "task_completion_requires_current_session": True,
    "labeler_promotion_retry_requires_open_task": True,
    "labeler_promotion_retry_requires_current_session": True,
    "labeler_promotion_retry_mutation_enabled": False,
    "labeler_promotion_retry_rejection_error": "operator_support_required",
    "completion_closes_open_sessions": True,
    "reopen_authority": "operator",
    "reopen_required_for_more_labeling": True,
}

BROWSER_RESPONSE_SECURITY_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

BROWSER_SIGNED_LINK_POLICY: dict[str, object] = {
    "canonical_entrypoint": DASHBOARD_PATH,
    "task_specific_links": "short_lived_convenience_links",
    "default_ttl_seconds": 24 * 60 * 60,
    "authorization_grant": False,
    "requires_authenticated_user": True,
    "requires_active_assignment": True,
    "requires_open_task": True,
    "binds_expected_user_in_new_links": True,
    "expected_user_mismatch_error": "signed_link_user_mismatch",
    "opens_guarded_session": True,
    "session_bound_after_open": True,
    "runtime_operator_validation_start_gate_enforced": True,
    "dashboard_preferred_for_multi_task_work": True,
}

BROWSER_CLIENT_AUTHORITY: dict[str, object] = {
    "mutation_executor": "server",
    "browser_can_submit_edits": True,
    "browser_can_write_zarr": False,
    "browser_can_write_filesystem": False,
    "browser_receives_write_credentials": False,
    "browser_receives_direct_zarr_handles": False,
}

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

BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT: dict[str, object] = {
    "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
    "server_owned_write_target": True,
    "payload_role": "browser_command_payload",
    "training_zarr_mutation_target_kind": "task_scoped_training_zarr",
    "browser_label_write_target": "training_zarr",
    "csv_handoff_artifact_role": "metadata_only_control_plane",
    "csv_handoff_artifacts_are_label_write_targets": False,
    "handoff_csv_artifacts_are_label_write_targets": False,
    "intermediate_csv_artifacts_are_label_write_targets": False,
    "handoff_artifacts_are_metadata_only": True,
    "browser_writes_csv_or_handoff_files": False,
    "browser_writes_handoff_csv": False,
    "browser_writes_intermediate_csv": False,
    "browser_receives_zarr_write_authority": False,
    "browser_has_direct_zarr_write_authority": False,
    "requires_active_assignment": True,
    "requires_open_task": True,
    "requires_current_session": True,
    "requires_current_target_token": True,
}

BROWSER_WORKFLOW_CAPABILITIES: tuple[dict[str, object], ...] = (
    {
        "workflow_kind": "keypoints",
        "label": "Keypoint correction",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Correct failed or reviewed keypoints through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "direct",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/keypoints/save",
            "payload_fields": ["points", "advance", "target_token"],
            "required_fields": ["points", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "save_keypoints",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "The browser submits keypoint edits; the server applies them through Palette review/write tooling.",
    },
    {
        "workflow_kind": "detect_training",
        "label": "Detection training boxes",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Edit training bounding boxes through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "direct",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/detect/save",
            "payload_fields": ["bbox_norm", "advance", "target_token"],
            "required_fields": ["bbox_norm", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "save_detect_bbox",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "The browser never receives direct zarr write authority.",
    },
    {
        "workflow_kind": "detect_analysis",
        "label": "Analysis detection boxes",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Reviewable by default; editable only when task scope enables analysis-box edits.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_analysis_zarr",
            "source_mutation_target_kind": "task_scoped_analysis_zarr",
            "promotion_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "promotion_when_configured",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/detect-analysis/save",
            "payload_fields": ["bbox_norm", "advance", "target_token"],
            "required_fields": ["bbox_norm", "target_token"],
            "response_fields": ["ok", "result", "state", "promotion"],
            "audit_event": "save_detect_analysis_bbox",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": {
                **dict(BROWSER_MUTATION_RETRY_POLICY),
                "secondary_side_effects": ["promotion_success", "promotion_failed"],
                "retry_guidance": "Saving the same editable analysis box again should leave the analysis label data in the same state, but may enqueue or record another promotion attempt when promotion is enabled.",
            },
            "secondary_events": ["promotion_success", "promotion_failed"],
            "scope_required": {"editable": True},
            "registry_refresh": True,
            "guard": "session_for_user",
        },
        "notes": "Use task scope to decide whether a detection-analysis task is review-only or mutable.",
    },
    {
        "workflow_kind": "subject_mask_component",
        "label": "Subject mask component masks",
        "browser_editor": True,
        "server_mutation": True,
        "completion_supported": True,
        "client_authority": dict(BROWSER_CLIENT_AUTHORITY),
        "write_scope": "Edit assigned subject-mask components through the guarded session editor.",
        "write_contract": {
            **dict(BROWSER_WORKFLOW_SERVER_WRITE_CONTRACT),
            "primary_mutation_target_kind": "task_scoped_training_zarr",
            "training_zarr_write_mode": "session_checkpoint_then_apply",
            "save_method": "POST",
            "save_endpoint": "/api/sessions/{session_id}/subject-mask/save",
            "save_semantics": "checkpoint_only_no_canonical_zarr_write",
            "apply_method": "POST",
            "apply_endpoint": "/api/sessions/{session_id}/subject-mask/apply",
            "apply_semantics": "coalesce_saved_session_checkpoints_and_write_canonical_zarr_before_assignment_completion",
            "payload_fields": ["mask", "advance", "target_token"],
            "required_fields": ["mask", "target_token"],
            "response_fields": ["ok", "result", "state"],
            "audit_event": "checkpoint_subject_mask_roi",
            "canonical_apply_audit_event": "apply_subject_mask_session_checkpoints",
            "audit_provenance": dict(BROWSER_MUTATION_AUDIT_PROVENANCE),
            "retry_policy": dict(BROWSER_MUTATION_RETRY_POLICY),
            "registry_refresh": "apply_only",
            "guard": "session_for_user",
        },
        "notes": "Subject-mask browser saves checkpoint to the labeling sidecar; explicit apply writes the unified refined subject-mask path while the assignment remains open.",
    },
)

def _browser_mutation_write_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_browser_mutation_write_policy.v1",
        "authoritative_label_state": "assigned_task_zarr_scope",
        "mutable_label_data_plane": "task_scoped_training_zarr",
        "label_mutation_target_kind": "task_scoped_training_zarr",
        "browser_label_write_target": "training_zarr",
        "server_mutates_task_scoped_zarr_targets": True,
        "training_zarr_mutations_are_server_owned": True,
        "promotion_training_zarr_mutation_requires_task_scope": True,
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
        "requires_active_assignment": True,
        "requires_open_task": True,
        "requires_current_session": True,
        "requires_current_target_token": True,
        "audit_event_store": "labeling_task_events",
        "operator_backup_plan_gate": "mutable_zarr_backup_confirmation",
        "write_target_roles_reported_by_zarr_backup_plan": True,
    }

def _browser_mutation_write_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    authoritative_label_state = str(policy.get("authoritative_label_state") or "")
    mutable_label_data_plane = str(policy.get("mutable_label_data_plane") or "")
    label_mutation_target_kind = str(policy.get("label_mutation_target_kind") or "")
    browser_label_write_target = str(policy.get("browser_label_write_target") or "")
    server_mutates_task_zarr = bool(policy.get("server_mutates_task_scoped_zarr_targets"))
    training_zarr_mutations_are_server_owned = bool(
        policy.get("training_zarr_mutations_are_server_owned")
    )
    handoffs_metadata_only = bool(policy.get("handoff_artifacts_are_metadata_only"))
    csv_handoff_artifact_role = str(policy.get("csv_handoff_artifact_role") or "")
    csv_handoff_artifacts_are_label_write_targets = bool(
        policy.get("csv_handoff_artifacts_are_label_write_targets")
    )
    handoff_csv_artifacts_are_label_write_targets = policy.get(
        "handoff_csv_artifacts_are_label_write_targets"
    )
    intermediate_csv_artifacts_are_label_write_targets = policy.get(
        "intermediate_csv_artifacts_are_label_write_targets"
    )
    browser_writes_control_plane = bool(policy.get("browser_writes_csv_or_handoff_files"))
    browser_writes_handoff_csv = policy.get("browser_writes_handoff_csv")
    browser_writes_intermediate_csv = policy.get("browser_writes_intermediate_csv")
    browser_receives_zarr_write_authority = bool(policy.get("browser_receives_zarr_write_authority"))
    browser_has_direct_zarr_write_authority = bool(
        policy.get("browser_has_direct_zarr_write_authority")
    )
    requires_active_assignment = bool(policy.get("requires_active_assignment"))
    requires_open_task = bool(policy.get("requires_open_task"))
    requires_current_session = bool(policy.get("requires_current_session"))
    requires_current_target_token = bool(policy.get("requires_current_target_token"))
    ready = (
        authoritative_label_state == "assigned_task_zarr_scope"
        and mutable_label_data_plane == "task_scoped_training_zarr"
        and label_mutation_target_kind == "task_scoped_training_zarr"
        and browser_label_write_target == "training_zarr"
        and server_mutates_task_zarr
        and training_zarr_mutations_are_server_owned
        and handoffs_metadata_only
        and csv_handoff_artifact_role == "metadata_only_control_plane"
        and not csv_handoff_artifacts_are_label_write_targets
        and handoff_csv_artifacts_are_label_write_targets is False
        and intermediate_csv_artifacts_are_label_write_targets is False
        and not browser_writes_control_plane
        and browser_writes_handoff_csv is False
        and browser_writes_intermediate_csv is False
        and not browser_receives_zarr_write_authority
        and not browser_has_direct_zarr_write_authority
        and requires_active_assignment
        and requires_open_task
        and requires_current_session
        and requires_current_target_token
    )
    return {
        "schema": "palette.web_labeling_browser_mutation_write_contract.v1",
        "ready": ready,
        "authoritative_label_state": authoritative_label_state,
        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
        "mutable_label_data_plane": mutable_label_data_plane,
        "label_mutation_target_kind": label_mutation_target_kind,
        "browser_label_write_target": browser_label_write_target,
        "control_plane_artifacts": ["handoff_csv", "handoff_html", "handoff_json"],
        "server_mutates_task_scoped_zarr_targets": server_mutates_task_zarr,
        "training_zarr_mutations_are_server_owned": training_zarr_mutations_are_server_owned,
        "handoff_artifacts_are_metadata_only": handoffs_metadata_only,
        "csv_handoff_artifact_role": csv_handoff_artifact_role,
        "csv_handoff_artifacts_are_label_write_targets": (
            csv_handoff_artifacts_are_label_write_targets
        ),
        "handoff_csv_artifacts_are_label_write_targets": bool(
            handoff_csv_artifacts_are_label_write_targets
        ),
        "intermediate_csv_artifacts_are_label_write_targets": bool(
            intermediate_csv_artifacts_are_label_write_targets
        ),
        "browser_writes_csv_or_handoff_files": browser_writes_control_plane,
        "browser_writes_handoff_csv": bool(browser_writes_handoff_csv),
        "browser_writes_intermediate_csv": bool(browser_writes_intermediate_csv),
        "browser_receives_zarr_write_authority": browser_receives_zarr_write_authority,
        "browser_has_direct_zarr_write_authority": browser_has_direct_zarr_write_authority,
        "requires_active_assignment": requires_active_assignment,
        "requires_open_task": requires_open_task,
        "requires_current_session": requires_current_session,
        "requires_current_target_token": requires_current_target_token,
        "operator_backup_plan_gate": str(policy.get("operator_backup_plan_gate") or ""),
    }

def _browser_mutation_write_runtime_checklist(
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = policy if isinstance(policy, Mapping) else _browser_mutation_write_policy()
    contract = _browser_mutation_write_contract_policy(source)
    return {
        "schema": "palette.web_labeling_browser_mutation_write_checklist.v1",
        "ready": bool(contract.get("ready")),
        "authoritative_label_state": str(contract.get("authoritative_label_state") or ""),
        "data_plane_write_target": str(contract.get("data_plane_write_target") or ""),
        "mutable_label_data_plane": str(contract.get("mutable_label_data_plane") or ""),
        "label_mutation_target_kind": str(contract.get("label_mutation_target_kind") or ""),
        "browser_label_write_target": str(contract.get("browser_label_write_target") or ""),
        "server_mutates_task_scoped_zarr_targets": bool(
            contract.get("server_mutates_task_scoped_zarr_targets")
        ),
        "training_zarr_mutations_are_server_owned": bool(
            contract.get("training_zarr_mutations_are_server_owned")
        ),
        "promotion_training_zarr_mutation_requires_task_scope": bool(
            source.get("promotion_training_zarr_mutation_requires_task_scope")
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
        "requires_active_assignment": bool(contract.get("requires_active_assignment")),
        "requires_open_task": bool(contract.get("requires_open_task")),
        "requires_current_session": bool(contract.get("requires_current_session")),
        "requires_current_target_token": bool(contract.get("requires_current_target_token")),
    }

def _browser_workflow_capabilities() -> list[dict[str, object]]:
    return [dict(row) for row in BROWSER_WORKFLOW_CAPABILITIES]

def _browser_workflow_kinds() -> list[str]:
    return [str(row["workflow_kind"]) for row in BROWSER_WORKFLOW_CAPABILITIES]

def _browser_task_state_policy() -> dict[str, object]:
    return dict(BROWSER_TASK_STATE_POLICY)

def _browser_task_state_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    startable_task_states = [str(state) for state in policy.get("startable_task_states") or []]
    completed_tasks_read_only = bool(policy.get("completed_tasks_read_only"))
    completed_tasks_open_new_sessions = bool(policy.get("completed_tasks_open_new_sessions"))
    completed_task_open_requests = str(policy.get("completed_task_open_requests") or "")
    completed_task_save_requests = str(policy.get("completed_task_save_requests") or "")
    non_startable_task_open_requests = str(policy.get("non_startable_task_open_requests") or "")
    non_startable_task_save_requests = str(policy.get("non_startable_task_save_requests") or "")
    completion_closes_open_sessions = bool(policy.get("completion_closes_open_sessions"))
    reopen_authority = str(policy.get("reopen_authority") or "")
    reopen_required_for_more_labeling = bool(policy.get("reopen_required_for_more_labeling"))
    labeler_promotion_retry_requires_open_task = bool(policy.get("labeler_promotion_retry_requires_open_task"))
    labeler_promotion_retry_mutation_enabled = bool(
        policy.get("labeler_promotion_retry_mutation_enabled")
    )
    labeler_promotion_retry_rejection_error = str(
        policy.get("labeler_promotion_retry_rejection_error") or ""
    )
    task_completion_requires_current_session = bool(policy.get("task_completion_requires_current_session"))
    ready = (
        startable_task_states == list(LABELER_START_TASK_STATES)
        and completed_tasks_read_only
        and not completed_tasks_open_new_sessions
        and completed_task_open_requests == "reject_task_complete"
        and completed_task_save_requests == "reject_task_complete"
        and non_startable_task_open_requests == "reject_task_not_startable"
        and non_startable_task_save_requests == "reject_task_not_startable"
        and completion_closes_open_sessions
        and reopen_authority == "operator"
        and reopen_required_for_more_labeling
        and labeler_promotion_retry_requires_open_task
        and not labeler_promotion_retry_mutation_enabled
        and labeler_promotion_retry_rejection_error == "operator_support_required"
        and task_completion_requires_current_session
    )
    return {
        "schema": "palette.web_labeling_task_state_contract.v1",
        "ready": ready,
        "startable_task_states": startable_task_states,
        "completed_tasks_read_only": completed_tasks_read_only,
        "completed_tasks_open_new_sessions": completed_tasks_open_new_sessions,
        "completed_task_open_requests": completed_task_open_requests,
        "completed_task_save_requests": completed_task_save_requests,
        "non_startable_task_open_requests": non_startable_task_open_requests,
        "non_startable_task_save_requests": non_startable_task_save_requests,
        "completion_closes_open_sessions": completion_closes_open_sessions,
        "reopen_authority": reopen_authority,
        "reopen_required_for_more_labeling": reopen_required_for_more_labeling,
        "labeler_promotion_retry_requires_open_task": labeler_promotion_retry_requires_open_task,
        "labeler_promotion_retry_mutation_enabled": labeler_promotion_retry_mutation_enabled,
        "labeler_promotion_retry_rejection_error": labeler_promotion_retry_rejection_error,
        "task_completion_requires_current_session": task_completion_requires_current_session,
        "ordinary_labeler_mutation_after_completion": "reject_task_complete",
        "ordinary_labeler_promotion_retry_mutation": "operator_support_required",
        "operator_reopen_required_before_more_labeling": ready,
    }

def _browser_mutation_target_contract_policy(
    *,
    task_state_policy: Mapping[str, object] | None = None,
    browser_workflows: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    policy = dict(task_state_policy or _browser_task_state_policy())
    workflows = list(browser_workflows or _browser_workflow_capabilities())
    mutable_workflow_count = 0
    workflows_missing_target_token: list[str] = []
    for workflow in workflows:
        if not bool(workflow.get("server_mutation")):
            continue
        mutable_workflow_count += 1
        workflow_kind = str(workflow.get("workflow_kind") or "unknown")
        write_contract = workflow.get("write_contract") if isinstance(workflow.get("write_contract"), Mapping) else {}
        raw_payload_fields = write_contract.get("payload_fields") if isinstance(write_contract.get("payload_fields"), list) else []
        raw_required_fields = write_contract.get("required_fields") if isinstance(write_contract.get("required_fields"), list) else []
        payload_fields = {
            str(field)
            for field in raw_payload_fields
            if str(field or "").strip()
        }
        required_fields = {
            str(field)
            for field in raw_required_fields
            if str(field or "").strip()
        }
        if "target_token" not in payload_fields or "target_token" not in required_fields:
            workflows_missing_target_token.append(workflow_kind)
    rejects_client_target_selectors = (
        str(policy.get("browser_mutation_target_selectors") or "") == "server_owned_reject_client_fields"
    )
    requires_current_target_token = (
        str(policy.get("browser_mutation_target_token") or "") == "required_current_target_token"
    )
    workflow_contracts_require_target_token = not workflows_missing_target_token
    ready = (
        rejects_client_target_selectors
        and requires_current_target_token
        and workflow_contracts_require_target_token
        and mutable_workflow_count > 0
    )
    return {
        "schema": "palette.web_labeling_browser_mutation_target_contract.v1",
        "server_owns_mutation_target": True,
        "target_token_field": "target_token",
        "target_selector_fields_rejected": list(BROWSER_MUTATION_TARGET_SELECTOR_KEYS),
        "rejects_client_target_selectors": rejects_client_target_selectors,
        "requires_current_target_token": requires_current_target_token,
        "stale_same_session_tab_guard": requires_current_target_token,
        "mutable_workflow_count": mutable_workflow_count,
        "workflow_contracts_require_target_token": workflow_contracts_require_target_token,
        "workflows_missing_target_token": workflows_missing_target_token,
        "ready": ready,
    }

def _browser_workflow_scope_contract_policy(
    *,
    task_state_policy: Mapping[str, object],
    browser_workflows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    supported_workflow_kinds = [
        str(workflow.get("workflow_kind") or "")
        for workflow in browser_workflows
        if str(workflow.get("workflow_kind") or "").strip()
    ]
    mutable_workflow_kinds = [
        str(workflow.get("workflow_kind") or "")
        for workflow in browser_workflows
        if bool(workflow.get("server_mutation")) and str(workflow.get("workflow_kind") or "").strip()
    ]
    workflow_training_zarr_write_modes: dict[str, str] = {}
    workflows_missing_server_owned_zarr_target: list[str] = []
    workflows_missing_csv_metadata_contract: list[str] = []
    workflows_with_csv_handoff_label_targets: list[str] = []
    workflows_with_browser_csv_handoff_writes: list[str] = []
    workflows_with_browser_zarr_authority: list[str] = []
    workflows_with_browser_filesystem_authority: list[str] = []
    for workflow in browser_workflows:
        workflow_kind = str(workflow.get("workflow_kind") or "unknown")
        authority = workflow.get("client_authority") if isinstance(workflow.get("client_authority"), Mapping) else {}
        if bool(authority.get("browser_can_write_zarr")):
            workflows_with_browser_zarr_authority.append(workflow_kind)
        if bool(authority.get("browser_can_write_filesystem")):
            workflows_with_browser_filesystem_authority.append(workflow_kind)
        if not bool(workflow.get("server_mutation")):
            continue
        write_contract = workflow.get("write_contract") if isinstance(workflow.get("write_contract"), Mapping) else {}
        primary_target_kind = str(write_contract.get("primary_mutation_target_kind") or "")
        training_target_kind = str(write_contract.get("training_zarr_mutation_target_kind") or "")
        training_write_mode = str(write_contract.get("training_zarr_write_mode") or "")
        workflow_training_zarr_write_modes[workflow_kind] = training_write_mode
        if not (
            bool(write_contract.get("server_owned_write_target"))
            and str(write_contract.get("data_plane_write_target") or "") == "server_owned_assigned_task_zarr_scope"
            and primary_target_kind.startswith("task_scoped_")
            and training_target_kind == "task_scoped_training_zarr"
            and str(write_contract.get("browser_label_write_target") or "") == "training_zarr"
            and training_write_mode in {"direct", "promotion_when_configured", "session_checkpoint_then_apply"}
            and not bool(write_contract.get("browser_receives_zarr_write_authority"))
            and not bool(write_contract.get("browser_has_direct_zarr_write_authority"))
        ):
            workflows_missing_server_owned_zarr_target.append(workflow_kind)
        if not (
            bool(write_contract.get("handoff_artifacts_are_metadata_only"))
            and str(write_contract.get("csv_handoff_artifact_role") or "") == "metadata_only_control_plane"
            and write_contract.get("handoff_csv_artifacts_are_label_write_targets") is False
            and write_contract.get("intermediate_csv_artifacts_are_label_write_targets") is False
            and write_contract.get("browser_writes_handoff_csv") is False
            and write_contract.get("browser_writes_intermediate_csv") is False
        ):
            workflows_missing_csv_metadata_contract.append(workflow_kind)
        if (
            bool(write_contract.get("csv_handoff_artifacts_are_label_write_targets"))
            or bool(write_contract.get("handoff_csv_artifacts_are_label_write_targets"))
            or bool(write_contract.get("intermediate_csv_artifacts_are_label_write_targets"))
        ):
            workflows_with_csv_handoff_label_targets.append(workflow_kind)
        if (
            write_contract.get("browser_writes_csv_or_handoff_files") is not False
            or write_contract.get("browser_writes_handoff_csv") is not False
            or write_contract.get("browser_writes_intermediate_csv") is not False
        ):
            workflows_with_browser_csv_handoff_writes.append(workflow_kind)
    absolute_navigation_out_of_scope = str(task_state_policy.get("absolute_navigation_out_of_scope") or "")
    browser_mutation_target_selectors = str(task_state_policy.get("browser_mutation_target_selectors") or "")
    browser_mutation_target_token = str(task_state_policy.get("browser_mutation_target_token") or "")
    ready = (
        bool(supported_workflow_kinds)
        and absolute_navigation_out_of_scope == "reject_nav_error"
        and browser_mutation_target_selectors == "server_owned_reject_client_fields"
        and browser_mutation_target_token == "required_current_target_token"
        and not workflows_missing_server_owned_zarr_target
        and not workflows_missing_csv_metadata_contract
        and not workflows_with_csv_handoff_label_targets
        and not workflows_with_browser_csv_handoff_writes
        and not workflows_with_browser_zarr_authority
        and not workflows_with_browser_filesystem_authority
    )
    return {
        "schema": "palette.web_labeling_browser_workflow_scope_contract.v1",
        "ready": ready,
        "supported_workflow_kinds": supported_workflow_kinds,
        "supported_workflow_count": len(supported_workflow_kinds),
        "mutable_workflow_kinds": mutable_workflow_kinds,
        "mutable_workflow_count": len(mutable_workflow_kinds),
        "absolute_navigation_out_of_scope": absolute_navigation_out_of_scope,
        "browser_mutation_target_selectors": browser_mutation_target_selectors,
        "target_selector_fields_rejected": list(BROWSER_MUTATION_TARGET_SELECTOR_KEYS),
        "browser_mutation_target_token": browser_mutation_target_token,
        "workflow_contracts_server_owned_zarr_targets": not workflows_missing_server_owned_zarr_target,
        "workflow_contracts_training_zarr_target_kind": "task_scoped_training_zarr",
        "workflow_training_zarr_write_modes": workflow_training_zarr_write_modes,
        "workflow_contracts_csv_handoff_metadata_only": (
            not workflows_missing_csv_metadata_contract
            and not workflows_with_csv_handoff_label_targets
            and not workflows_with_browser_csv_handoff_writes
        ),
        "workflows_missing_server_owned_zarr_target": workflows_missing_server_owned_zarr_target,
        "workflows_missing_csv_metadata_contract": workflows_missing_csv_metadata_contract,
        "workflows_with_csv_handoff_label_targets": workflows_with_csv_handoff_label_targets,
        "workflows_with_browser_csv_handoff_writes": workflows_with_browser_csv_handoff_writes,
        "target_indices_components_labels_frames_server_owned": (
            browser_mutation_target_selectors == "server_owned_reject_client_fields"
            and browser_mutation_target_token == "required_current_target_token"
        ),
        "absolute_navigation_out_of_scope_rejects": absolute_navigation_out_of_scope == "reject_nav_error",
        "workflows_with_browser_zarr_authority": workflows_with_browser_zarr_authority,
        "workflows_with_browser_filesystem_authority": workflows_with_browser_filesystem_authority,
    }

def _browser_signed_link_policy() -> dict[str, object]:
    return dict(BROWSER_SIGNED_LINK_POLICY)

def _signed_link_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    task_specific_links = bool(policy.get("task_specific_links"))
    authorization_grant = bool(policy.get("authorization_grant"))
    binds_expected_user_in_new_links = bool(policy.get("binds_expected_user_in_new_links"))
    expected_user_required_on_open = bool(
        policy.get("expected_user_required_on_open", binds_expected_user_in_new_links)
    )
    signed_links_are_not_identity = bool(policy.get("signed_links_are_not_identity", not authorization_grant))
    runtime_operator_validation_start_gate_enforced = bool(
        policy.get("runtime_operator_validation_start_gate_enforced", True)
    )
    ready = (
        task_specific_links
        and not authorization_grant
        and binds_expected_user_in_new_links
        and expected_user_required_on_open
        and signed_links_are_not_identity
        and runtime_operator_validation_start_gate_enforced
    )
    return {
        "schema": "palette.web_labeling_signed_link_contract.v1",
        "ready": ready,
        "task_specific_links": task_specific_links,
        "authorization_grant": authorization_grant,
        "binds_expected_user_in_new_links": binds_expected_user_in_new_links,
        "expected_user_required_on_open": expected_user_required_on_open,
        "signed_links_are_not_identity": signed_links_are_not_identity,
        "signed_links_are_entry_hints_not_authorization": not authorization_grant,
        "forwarded_signed_links_recheck_identity": expected_user_required_on_open and signed_links_are_not_identity,
        "requires_server_side_task_authorization": not authorization_grant,
        "runtime_operator_validation_start_gate_enforced": runtime_operator_validation_start_gate_enforced,
        "operator_validation_start_gate_checked_before_session_create": runtime_operator_validation_start_gate_enforced,
    }

def _browser_workflow_scope_runtime_checklist(
    *,
    task_state_policy: Mapping[str, object],
    browser_workflows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    workflow_rows = [dict(row) for row in browser_workflows]
    workflow_kinds = [
        str(row.get("workflow_kind") or "").strip()
        for row in workflow_rows
        if str(row.get("workflow_kind") or "").strip()
    ]
    mutable_workflow_kinds: list[str] = []
    workflows_missing_target_token: list[str] = []
    workflows_missing_write_scope: list[str] = []
    workflows_missing_session_guard: list[str] = []
    for row in workflow_rows:
        workflow_kind = str(row.get("workflow_kind") or "").strip() or "unknown"
        if not bool(row.get("server_mutation")):
            continue
        mutable_workflow_kinds.append(workflow_kind)
        write_scope = str(row.get("write_scope") or "").strip()
        if not write_scope:
            workflows_missing_write_scope.append(workflow_kind)
        write_contract = row.get("write_contract") if isinstance(row.get("write_contract"), Mapping) else {}
        if str(write_contract.get("guard") or "") != "session_for_user":
            workflows_missing_session_guard.append(workflow_kind)
        payload_fields = {
            str(field)
            for field in (
                write_contract.get("payload_fields")
                if isinstance(write_contract.get("payload_fields"), list)
                else []
            )
            if str(field or "").strip()
        }
        required_fields = {
            str(field)
            for field in (
                write_contract.get("required_fields")
                if isinstance(write_contract.get("required_fields"), list)
                else []
            )
            if str(field or "").strip()
        }
        if "target_token" not in payload_fields or "target_token" not in required_fields:
            workflows_missing_target_token.append(workflow_kind)
    absolute_navigation_out_of_scope = str(
        task_state_policy.get("absolute_navigation_out_of_scope") or ""
    )
    browser_mutation_target_selectors = str(
        task_state_policy.get("browser_mutation_target_selectors") or ""
    )
    browser_mutation_target_token = str(task_state_policy.get("browser_mutation_target_token") or "")
    absolute_navigation_out_of_scope_rejects = absolute_navigation_out_of_scope == "reject_nav_error"
    browser_mutation_targets_server_owned = (
        browser_mutation_target_selectors == "server_owned_reject_client_fields"
    )
    current_target_token_required = browser_mutation_target_token == "required_current_target_token"
    mutable_workflows_require_target_token = not workflows_missing_target_token
    mutable_workflows_session_guarded = not workflows_missing_session_guard
    mutable_workflows_have_write_scope = not workflows_missing_write_scope
    target_scope_enforced = (
        absolute_navigation_out_of_scope_rejects
        and browser_mutation_targets_server_owned
        and current_target_token_required
        and mutable_workflows_require_target_token
        and mutable_workflows_session_guarded
        and mutable_workflows_have_write_scope
        and bool(workflow_kinds)
    )
    return {
        "schema": "palette.web_labeling_browser_workflow_scope_runtime_checklist.v1",
        "ready": target_scope_enforced,
        "supported_browser_workflow_kinds": workflow_kinds,
        "browser_workflow_count": len(workflow_kinds),
        "mutable_browser_workflow_kinds": mutable_workflow_kinds,
        "mutable_browser_workflow_count": len(mutable_workflow_kinds),
        "absolute_navigation_out_of_scope": absolute_navigation_out_of_scope,
        "absolute_navigation_out_of_scope_rejects": absolute_navigation_out_of_scope_rejects,
        "browser_mutation_target_selectors": browser_mutation_target_selectors,
        "browser_mutation_targets_server_owned": browser_mutation_targets_server_owned,
        "browser_mutation_target_token": browser_mutation_target_token,
        "current_target_token_required": current_target_token_required,
        "mutable_workflows_require_target_token": mutable_workflows_require_target_token,
        "workflows_missing_target_token": workflows_missing_target_token,
        "mutable_workflows_session_guarded": mutable_workflows_session_guarded,
        "workflows_missing_session_guard": workflows_missing_session_guard,
        "mutable_workflows_have_write_scope": mutable_workflows_have_write_scope,
        "workflows_missing_write_scope": workflows_missing_write_scope,
        "browser_direct_target_selection_rejected": browser_mutation_targets_server_owned,
        "target_selector_fields_rejected": list(BROWSER_MUTATION_TARGET_SELECTOR_KEYS),
        "target_indices_components_labels_frames_inside_task_scope": target_scope_enforced,
        "labeler_visible_scope": "assigned_recordings_for_resolved_user",
        "data_plane_mutation_scope": "current_guarded_session_task_target",
    }

def _browser_response_security_contract_policy(policy: Mapping[str, object]) -> dict[str, object]:
    headers = policy.get("headers") if isinstance(policy.get("headers"), Mapping) else {}
    cache_control = str(headers.get("Cache-Control") or "")
    pragma = str(headers.get("Pragma") or "")
    expires = str(headers.get("Expires") or "")
    x_frame_options = str(headers.get("X-Frame-Options") or "")
    x_content_type_options = str(headers.get("X-Content-Type-Options") or "")
    referrer_policy = str(headers.get("Referrer-Policy") or "")
    content_security_policy = str(headers.get("Content-Security-Policy") or "")
    permissions_policy = str(headers.get("Permissions-Policy") or "")
    no_store_cache = bool(policy.get("no_store_cache")) and "no-store" in cache_control
    clickjacking_protection = bool(policy.get("clickjacking_protection")) and x_frame_options == "DENY"
    mime_sniffing_protection = bool(policy.get("mime_sniffing_protection")) and x_content_type_options == "nosniff"
    referrer_leakage_protection = bool(policy.get("referrer_leakage_protection")) and referrer_policy == "no-referrer"
    csp_scope_ready = (
        bool(policy.get("content_security_policy_scope"))
        and "frame-ancestors 'none'" in content_security_policy
        and "base-uri 'self'" in content_security_policy
        and "form-action 'self'" in content_security_policy
        and "object-src 'none'" in content_security_policy
    )
    permissions_ready = all(
        token in permissions_policy
        for token in ("camera=()", "microphone=()", "geolocation=()")
    )
    protected_labeler_paths = [
        str(path)
        for path in policy.get("protected_labeler_paths", [])
        if str(path)
    ]
    personalized_alias_paths = [
        str(path)
        for path in policy.get("personalized_alias_paths", [])
        if str(path)
    ]
    canonical_fallback_paths = [
        str(path)
        for path in policy.get("canonical_fallback_paths", [])
        if str(path)
    ]
    personal_api_paths = [
        str(path)
        for path in policy.get("personal_api_paths", [])
        if str(path)
    ]
    protected_labeler_paths_ready = all(
        path in protected_labeler_paths
        for path in (
            "/",
            "/me",
            LABELING_HOME_PATH,
            PERSONAL_DATASET_QUEUE_PATH,
            DATASET_QUEUE_PATH,
            PERSONAL_WORK_PATH,
            DASHBOARD_PATH,
            IDENTITY_PROBE_PATH,
            "/api/me/identity",
            "/api/me/tasks",
            "/api/me/datasets",
        )
    )
    personalized_alias_paths_ready = all(
        path in personalized_alias_paths
        for path in (PERSONAL_DATASET_QUEUE_PATH, PERSONAL_WORK_PATH)
    )
    canonical_fallback_paths_ready = all(
        path in canonical_fallback_paths
        for path in (DATASET_QUEUE_PATH, DASHBOARD_PATH)
    )
    personal_api_paths_ready = all(
        path in personal_api_paths
        for path in ("/api/me/identity", "/api/me/tasks", "/api/me/datasets")
    )
    personalized_alias_header_parity_ready = bool(
        policy.get("personalized_alias_headers_must_match_canonical")
    )
    ready = (
        no_store_cache
        and pragma == "no-cache"
        and expires == "0"
        and clickjacking_protection
        and mime_sniffing_protection
        and referrer_leakage_protection
        and csp_scope_ready
        and permissions_ready
        and bool(policy.get("proxy_must_preserve_headers"))
        and protected_labeler_paths_ready
        and personalized_alias_paths_ready
        and canonical_fallback_paths_ready
        and personal_api_paths_ready
        and personalized_alias_header_parity_ready
    )
    return {
        "schema": "palette.web_labeling_browser_response_security_contract.v1",
        "ready": ready,
        "headers": dict(headers),
        "protected_labeler_paths": protected_labeler_paths,
        "protected_labeler_paths_ready": protected_labeler_paths_ready,
        "personalized_alias_paths": personalized_alias_paths,
        "personalized_alias_paths_ready": personalized_alias_paths_ready,
        "canonical_fallback_paths": canonical_fallback_paths,
        "canonical_fallback_paths_ready": canonical_fallback_paths_ready,
        "personal_api_paths": personal_api_paths,
        "personal_api_paths_ready": personal_api_paths_ready,
        "personalized_alias_headers_must_match_canonical": personalized_alias_header_parity_ready,
        "no_store_cache": no_store_cache,
        "pragma_no_cache": pragma == "no-cache",
        "expires_zero": expires == "0",
        "clickjacking_protection": clickjacking_protection,
        "mime_sniffing_protection": mime_sniffing_protection,
        "referrer_leakage_protection": referrer_leakage_protection,
        "content_security_policy_scope_ready": csp_scope_ready,
        "permissions_policy_ready": permissions_ready,
        "proxy_must_preserve_headers": bool(policy.get("proxy_must_preserve_headers")),
    }

def _browser_response_security_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_browser_response_security_policy.v1",
        "headers": dict(BROWSER_RESPONSE_SECURITY_HEADERS),
        "protected_labeler_paths": [
            "/",
            "/me",
            LABELING_HOME_PATH,
            PERSONAL_DATASET_QUEUE_PATH,
            DATASET_QUEUE_PATH,
            PERSONAL_WORK_PATH,
            DASHBOARD_PATH,
            IDENTITY_PROBE_PATH,
            "/api/me/identity",
            "/api/me/tasks",
            "/api/me/datasets",
        ],
        "personalized_alias_paths": [
            PERSONAL_DATASET_QUEUE_PATH,
            PERSONAL_WORK_PATH,
        ],
        "canonical_fallback_paths": [
            DATASET_QUEUE_PATH,
            DASHBOARD_PATH,
        ],
        "personal_api_paths": [
            "/api/me/identity",
            "/api/me/tasks",
            "/api/me/datasets",
        ],
        "personalized_alias_headers_must_match_canonical": True,
        "no_store_cache": True,
        "proxy_must_preserve_headers": True,
        "clickjacking_protection": True,
        "mime_sniffing_protection": True,
        "referrer_leakage_protection": True,
        "content_security_policy_scope": "frame_ancestors_base_uri_form_action_object_src",
        "permissions_disabled": ["camera", "microphone", "geolocation"],
    }

def _session_guard_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_session_guard_policy.v1",
        "opens_guarded_session": True,
        "session_bound_after_open": True,
        "requires_authenticated_user": True,
        "requires_active_assignment": True,
        "requires_open_task": True,
        "requires_current_session": True,
        "requires_unexpired_session": True,
        "closed_sessions_rejected": True,
        "superseded_sessions_rejected": True,
        "expired_sessions_rejected": True,
        "completed_task_sessions_rejected": True,
        "non_startable_task_sessions_rejected": True,
        "reassigned_sessions_rejected": True,
        "stale_tab_save_rejected": True,
        "target_token_required_for_mutation": True,
        "labeler_promotion_retry_requires_current_session": True,
        "session_closure_event_support": True,
        "closure_event_field": "session_closure_event",
        "reopen_authority": "operator",
        "recovery_entrypoints": [
            "/",
            PERSONAL_DATASET_QUEUE_PATH,
            DATASET_QUEUE_PATH,
            PERSONAL_WORK_PATH,
            DASHBOARD_PATH,
        ],
        "rejection_errors": {
            "assignment_mismatch": "not_assigned",
            "task_complete": "task_complete",
            "closed_session": "session_closed",
            "expired_session": "session_expired",
            "superseded_session": "session_superseded",
            "user_mismatch": "session_user_mismatch",
            "task_not_startable": "task_not_startable",
        },
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

IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES = (
    "missing",
    "incomplete",
    "ready",
)

def _operator_validation_visibility_policy() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_operator_validation_visibility_policy.v1",
        "public_fields": [
            "operator_validation_required_before_invite",
            "operator_validation_required_before_invite_legacy_semantics",
            "operator_validation_required_before_invite_is_safe_share_approval",
            "operator_validation_required_before_invite_safe_share_field",
            "operator_validation_all_complete",
            "operator_validation_declared_all_complete",
            "operator_validation_ready_for_operator_validation",
            "operator_validation_gate_count",
            "operator_validation_status",
            "operator_validation_source",
            "operator_launch_approved_legacy_semantics",
            "operator_launch_approved_is_safe_share_approval",
            "operator_launch_approved_requires_safe_share_inspection",
            "operator_launch_approved_required_safe_share_field",
            "operator_launch_approved_required_safe_share_value",
            "operator_validation_pending_gate_ids",
            "operator_validation_needs_review_gate_ids",
            "operator_validation_required_missing_evidence_gate_ids",
            "operator_validation_required_pending_gate_count",
            "operator_validation_needs_review_gate_count",
            "operator_validation_required_missing_evidence_gate_count",
            "operator_validation_operator_action",
            "operator_validation_external_evidence_required",
            "operator_validation_external_evidence_required_gate_ids",
            "operator_validation_external_evidence_required_gate_count",
            "operator_validation_external_evidence_template_fields_by_gate_id",
            "operator_validation_external_evidence_template_paths_by_gate_id",
            "operator_validation_checklist_only_required_gate_ids",
            "operator_validation_checklist_only_required_gate_count",
            "identity_personal_queue_evidence_status",
            "identity_personal_queue_evidence_ready_count",
            "identity_personal_queue_evidence_missing_count",
            "identity_personal_queue_evidence_ready_users",
            "identity_personal_queue_evidence_missing_users",
            "identity_personal_queue_evidence_missing_fields_by_user",
            "identity_all_users_have_personal_queue_evidence",
        ],
        "identity_personal_queue_evidence_status_values": list(
            IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES
        ),
        "operator_validation_gate_status_values": list(
            OPERATOR_VALIDATION_GATE_STATUS_VALUES
        ),
        "operator_validation_gate_ids": list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS),
        "operator_validation_gate_flat_field_suffixes": list(
            OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES
        ),
        "operator_only_fields": ["operator_validation_checklist_path"],
        "operator_action_fields": ["operator_validation_command_templates"],
        "operator_action_fields_are_labeler_instructions": False,
        "labeler_visible_payloads_may_include_operator_action_fields_for_support": True,
        "labeler_visible_payloads_include_operator_only_fields": False,
        "per_user_payloads_use_public_fields_only": True,
        "top_level_operator_reports_may_include_operator_only_fields": True,
    }

def _operator_validation_visibility_fields(
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source = policy if isinstance(policy, Mapping) else _operator_validation_visibility_policy()
    public_fields = (
        source.get("public_fields") if isinstance(source.get("public_fields"), list) else []
    )
    operator_only_fields = (
        source.get("operator_only_fields")
        if isinstance(source.get("operator_only_fields"), list)
        else []
    )
    operator_action_fields = (
        source.get("operator_action_fields")
        if isinstance(source.get("operator_action_fields"), list)
        else []
    )
    identity_personal_queue_evidence_status_values = (
        source.get("identity_personal_queue_evidence_status_values")
        if isinstance(source.get("identity_personal_queue_evidence_status_values"), list)
        else []
    )
    operator_validation_gate_status_values = (
        source.get("operator_validation_gate_status_values")
        if isinstance(source.get("operator_validation_gate_status_values"), list)
        else []
    )
    operator_validation_gate_ids = (
        source.get("operator_validation_gate_ids")
        if isinstance(source.get("operator_validation_gate_ids"), list)
        else []
    )
    operator_validation_gate_flat_field_suffixes = (
        source.get("operator_validation_gate_flat_field_suffixes")
        if isinstance(source.get("operator_validation_gate_flat_field_suffixes"), list)
        else []
    )
    return {
        "operator_validation_public_fields": [str(field) for field in public_fields],
        "operator_validation_identity_personal_queue_evidence_status_values": [
            str(value) for value in identity_personal_queue_evidence_status_values
        ],
        "operator_validation_gate_status_values": [
            str(value) for value in operator_validation_gate_status_values
        ],
        "operator_validation_gate_ids": [
            str(gate_id) for gate_id in operator_validation_gate_ids
        ],
        "operator_validation_gate_flat_field_suffixes": [
            str(suffix) for suffix in operator_validation_gate_flat_field_suffixes
        ],
        "operator_validation_operator_only_fields": [
            str(field) for field in operator_only_fields
        ],
        "operator_validation_operator_action_fields": [
            str(field) for field in operator_action_fields
        ],
        "operator_validation_operator_action_fields_are_labeler_instructions": bool(
            source.get("operator_action_fields_are_labeler_instructions")
        ),
        "operator_validation_labeler_visible_payloads_may_include_operator_action_fields_for_support": bool(
            source.get("labeler_visible_payloads_may_include_operator_action_fields_for_support")
        ),
        "operator_validation_labeler_visible_payloads_include_operator_only_fields": bool(
            source.get("labeler_visible_payloads_include_operator_only_fields")
        ),
        "operator_validation_per_user_payloads_use_public_fields_only": bool(
            source.get("per_user_payloads_use_public_fields_only")
        ),
        "operator_validation_top_level_operator_reports_may_include_operator_only_fields": bool(
            source.get("top_level_operator_reports_may_include_operator_only_fields")
        ),
    }
