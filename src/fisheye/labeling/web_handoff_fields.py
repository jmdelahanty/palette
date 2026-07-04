from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .web_policy import (
    DEFAULT_OPERATOR_VALIDATION_GATE_IDS,
    OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES,
)


def _require_helpers(
    helpers: Mapping[str, Callable[..., Any]],
    names: tuple[str, ...],
) -> dict[str, Callable[..., Any]]:
    missing = [name for name in names if name not in helpers]
    if missing:
        raise KeyError(f"missing handoff field helper(s): {', '.join(missing)}")
    return {name: helpers[name] for name in names}


def _safe_share_external_launch_evidence_gap_field_names() -> list[str]:
    return [
        "safe_share_external_launch_evidence_gap_gate_ids",
        "safe_share_external_launch_evidence_gap_count",
        "safe_share_external_launch_evidence_gap_statuses",
        "safe_share_external_launch_evidence_gap_action_required",
        "safe_share_external_launch_evidence_gap_summary",
        "safe_share_external_launch_evidence_gap_todos",
        "safe_share_external_launch_evidence_gap_todo_count",
        "safe_share_external_launch_evidence_gap_todo_fields",
        "safe_share_external_launch_evidence_gap_template_paths_by_gate_id",
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id",
    ]



def _operator_validation_gate_flat_fieldnames() -> list[str]:
    fieldnames: list[str] = []
    for gate_id in DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        prefix = f"operator_validation_gate_{gate_id}"
        fieldnames.extend(f"{prefix}_{suffix}" for suffix in OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES)
    return fieldnames



def _handoff_ready_to_send(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> bool:
    field_helpers = _require_helpers(helpers, ('_handoff_assignment_ownership_fields', '_handoff_browser_mutation_write_fields', '_handoff_browser_response_security_fields', '_handoff_dataset_queue_blocks_labeler_start', '_handoff_entry_artifact_fields', '_handoff_known_user_status_fields', '_handoff_labeler_route_authorization_fields', '_handoff_labeler_safety_fields', '_handoff_mutation_audit_fields', '_handoff_session_guard_fields', '_handoff_signed_link_policy_fields', '_handoff_task_state_policy_fields', '_handoff_zarr_backup_fields'))
    _handoff_assignment_ownership_fields = field_helpers['_handoff_assignment_ownership_fields']
    _handoff_browser_mutation_write_fields = field_helpers['_handoff_browser_mutation_write_fields']
    _handoff_browser_response_security_fields = field_helpers['_handoff_browser_response_security_fields']
    _handoff_dataset_queue_blocks_labeler_start = field_helpers['_handoff_dataset_queue_blocks_labeler_start']
    _handoff_entry_artifact_fields = field_helpers['_handoff_entry_artifact_fields']
    _handoff_known_user_status_fields = field_helpers['_handoff_known_user_status_fields']
    _handoff_labeler_route_authorization_fields = field_helpers['_handoff_labeler_route_authorization_fields']
    _handoff_labeler_safety_fields = field_helpers['_handoff_labeler_safety_fields']
    _handoff_mutation_audit_fields = field_helpers['_handoff_mutation_audit_fields']
    _handoff_session_guard_fields = field_helpers['_handoff_session_guard_fields']
    _handoff_signed_link_policy_fields = field_helpers['_handoff_signed_link_policy_fields']
    _handoff_task_state_policy_fields = field_helpers['_handoff_task_state_policy_fields']
    _handoff_zarr_backup_fields = field_helpers['_handoff_zarr_backup_fields']

    counts = handoff.get("counts") if isinstance(handoff.get("counts"), Mapping) else {}
    ready_to_share_links = int(counts.get("ready_to_share_links") or 0)
    operator_validation_required = bool(handoff.get("operator_validation_required_before_invite"))
    operator_validation_complete = bool(handoff.get("operator_validation_all_complete"))
    known_user_fields = _handoff_known_user_status_fields(handoff)
    ownership_fields = _handoff_assignment_ownership_fields(handoff)
    entry_fields = _handoff_entry_artifact_fields(handoff)
    labeler_safety_fields = _handoff_labeler_safety_fields(handoff)
    route_authorization_fields = _handoff_labeler_route_authorization_fields(handoff)
    signed_link_fields = _handoff_signed_link_policy_fields(handoff)
    session_guard_fields = _handoff_session_guard_fields(handoff)
    task_state_fields = _handoff_task_state_policy_fields(handoff)
    zarr_backup_fields = _handoff_zarr_backup_fields(handoff)
    mutation_audit_fields = _handoff_mutation_audit_fields(handoff)
    response_security_fields = _handoff_browser_response_security_fields(handoff)
    mutation_write_fields = _handoff_browser_mutation_write_fields(handoff)
    return (
        bool(handoff.get("ok"))
        and bool(known_user_fields["known_labeler"])
        and int(known_user_fields["known_user_active_assignment_count"]) > 0
        and bool(ownership_fields["assignment_ownership_ok"])
        and bool(ownership_fields["assignment_ownership_contract_ready"])
        and bool(entry_fields["guarded_links_ready"])
        and bool(entry_fields["handoff_artifacts_ready"])
        and bool(entry_fields["preferred_labeler_entry_url_matches_personal_dataset_queue"])
        and bool(labeler_safety_fields["labeler_safety_ready"])
        and bool(route_authorization_fields["labeler_route_authorization_ready"])
        and bool(signed_link_fields["signed_link_policy_ready"])
        and bool(session_guard_fields["session_guard_policy_ready"])
        and bool(task_state_fields["task_state_policy_ready"])
        and bool(zarr_backup_fields["zarr_backup_ready"])
        and bool(mutation_audit_fields["mutation_audit_ready"])
        and bool(response_security_fields["browser_response_security_ready"])
        and bool(mutation_write_fields["browser_mutation_write_ready"])
        and int(counts.get("tasks") or 0) > 0
        and int(counts.get("signed_links") or 0) > 0
        and ready_to_share_links > 0
        and bool(str(handoff.get("base_url") or "").strip())
        and not _handoff_dataset_queue_blocks_labeler_start(handoff)
        and (not operator_validation_required or operator_validation_complete)
    )



def _handoff_browser_response_security_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_browser_response_security_contract_policy',))
    _browser_response_security_contract_policy = field_helpers['_browser_response_security_contract_policy']

    policy = (
        handoff.get("browser_response_security_policy")
        if isinstance(handoff.get("browser_response_security_policy"), Mapping)
        else None
    )
    contract = _browser_response_security_contract_policy(policy or {})
    headers = contract.get("headers") if isinstance(contract.get("headers"), Mapping) else {}
    present = policy is not None
    ready = present and bool(contract.get("ready"))
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = (
            "Regenerate the handoff so browser response-security policy metadata is present "
            "before sharing links."
        )
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so browser responses require no-store caching, "
            "anti-framing, MIME-sniffing protection, no-referrer behavior, narrow CSP, "
            "permissions restrictions, and proxy header preservation."
        )
    return {
        "browser_response_security_policy_present": present,
        "browser_response_security_ready": ready,
        "browser_response_security_readiness": readiness,
        "browser_response_security_operator_action": action,
        "browser_response_security_contract_ready": bool(contract.get("ready")),
        "browser_response_security_protected_labeler_paths": list(
            contract.get("protected_labeler_paths") or []
        ),
        "browser_response_security_protected_labeler_paths_ready": bool(
            contract.get("protected_labeler_paths_ready")
        ),
        "browser_response_security_personalized_alias_paths": list(
            contract.get("personalized_alias_paths") or []
        ),
        "browser_response_security_personalized_alias_paths_ready": bool(
            contract.get("personalized_alias_paths_ready")
        ),
        "browser_response_security_canonical_fallback_paths": list(
            contract.get("canonical_fallback_paths") or []
        ),
        "browser_response_security_canonical_fallback_paths_ready": bool(
            contract.get("canonical_fallback_paths_ready")
        ),
        "browser_response_security_personal_api_paths": list(
            contract.get("personal_api_paths") or []
        ),
        "browser_response_security_personal_api_paths_ready": bool(
            contract.get("personal_api_paths_ready")
        ),
        "browser_response_security_personalized_alias_headers_must_match_canonical": bool(
            contract.get("personalized_alias_headers_must_match_canonical")
        ),
        "browser_response_security_no_store_cache": bool(contract.get("no_store_cache")),
        "browser_response_security_pragma_no_cache": bool(contract.get("pragma_no_cache")),
        "browser_response_security_expires_zero": bool(contract.get("expires_zero")),
        "browser_response_security_clickjacking_protection": bool(
            contract.get("clickjacking_protection")
        ),
        "browser_response_security_mime_sniffing_protection": bool(
            contract.get("mime_sniffing_protection")
        ),
        "browser_response_security_referrer_leakage_protection": bool(
            contract.get("referrer_leakage_protection")
        ),
        "browser_response_security_csp_scope_ready": bool(
            contract.get("content_security_policy_scope_ready")
        ),
        "browser_response_security_permissions_policy_ready": bool(
            contract.get("permissions_policy_ready")
        ),
        "browser_response_security_proxy_must_preserve_headers": bool(
            contract.get("proxy_must_preserve_headers")
        ),
        "browser_response_security_cache_control": str(headers.get("Cache-Control") or ""),
        "browser_response_security_x_frame_options": str(headers.get("X-Frame-Options") or ""),
        "browser_response_security_x_content_type_options": str(
            headers.get("X-Content-Type-Options") or ""
        ),
        "browser_response_security_referrer_policy": str(headers.get("Referrer-Policy") or ""),
        "browser_response_security_content_security_policy": str(
            headers.get("Content-Security-Policy") or ""
        ),
        "browser_response_security_permissions_policy": str(headers.get("Permissions-Policy") or ""),
    }



def _handoff_mutation_audit_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_mutation_audit_contract_policy',))
    _mutation_audit_contract_policy = field_helpers['_mutation_audit_contract_policy']

    policy = (
        handoff.get("mutation_audit_policy")
        if isinstance(handoff.get("mutation_audit_policy"), Mapping)
        else None
    )
    contract = _mutation_audit_contract_policy(policy or {})
    present = policy is not None
    ready = present and bool(contract.get("ready"))
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so mutation-audit policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so browser mutations are recorded by the server "
            "in the append-only labeling task event store, browsers cannot write audit records "
            "directly, and required task/user/event fields are present."
        )
    return {
        "mutation_audit_policy_present": present,
        "mutation_audit_ready": ready,
        "mutation_audit_readiness": readiness,
        "mutation_audit_operator_action": action,
        "mutation_audit_contract_ready": bool(contract.get("ready")),
        "mutation_audit_event_store": str(contract.get("event_store") or ""),
        "mutation_audit_append_only": bool(contract.get("append_only")),
        "mutation_audit_server_records_events": bool(contract.get("server_records_events")),
        "mutation_audit_browser_records_events_directly": bool(
            contract.get("browser_records_events_directly")
        ),
        "mutation_audit_browser_receives_write_credentials": bool(
            contract.get("browser_receives_audit_store_write_credentials")
        ),
        "mutation_audit_per_workflow_contracts_include_provenance": bool(
            contract.get("per_workflow_write_contracts_include_audit_provenance")
        ),
        "mutation_audit_required_event_fields_present": bool(
            contract.get("required_event_fields_present")
        ),
        "mutation_audit_required_event_fields": (
            contract.get("required_event_fields")
            if isinstance(contract.get("required_event_fields"), list)
            else []
        ),
        "mutation_audit_timestamp_field": str(contract.get("timestamp_field") or ""),
        "mutation_audit_same_payload_retry_safe": bool(contract.get("same_payload_retry_safe")),
        "mutation_audit_duplicate_events_possible": bool(
            contract.get("duplicate_audit_events_possible")
        ),
        "mutation_audit_validation_gate": str(contract.get("validation_gate") or ""),
    }



def _handoff_zarr_backup_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_zarr_backup_contract_policy',))
    _zarr_backup_contract_policy = field_helpers['_zarr_backup_contract_policy']

    policy = (
        handoff.get("zarr_backup_policy")
        if isinstance(handoff.get("zarr_backup_policy"), Mapping)
        else None
    )
    counts = handoff.get("counts") if isinstance(handoff.get("counts"), Mapping) else {}
    files = handoff.get("files") if isinstance(handoff.get("files"), Mapping) else {}
    contract = _zarr_backup_contract_policy(policy or {}, counts, files)
    present = policy is not None
    contract_ready = bool(contract.get("ready"))
    backup_required_targets = int(contract.get("backup_required_targets") or 0)
    tasks_missing_zarr_path = int(contract.get("tasks_missing_zarr_path") or 0)
    backup_plan = str(contract.get("zarr_backup_plan") or "").strip()
    backup_plan_required = backup_required_targets > 0
    backup_plan_present = bool(backup_plan)
    ready = present and contract_ready and (not backup_plan_required or backup_plan_present)
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so Zarr backup policy metadata is present before sharing links."
    elif not contract_ready:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so mutable Zarr backup policy is operator-only, "
            "read-only for labelers, requires copy-before-labeling, hides backup paths from labelers, "
            "and assigns rollback ownership to the operator."
        )
    else:
        readiness = "missing_backup_plan"
        action = (
            "Regenerate the handoff or launch bundle so mutable Zarr backup targets have a "
            "backup-plan artifact before labeler links are shared."
        )
    return {
        "zarr_backup_policy_present": present,
        "zarr_backup_ready": ready,
        "zarr_backup_readiness": readiness,
        "zarr_backup_operator_action": action,
        "zarr_backup_contract_ready": contract_ready,
        "zarr_backup_read_only_plan": bool(contract.get("read_only_plan")),
        "zarr_backup_operator_only": bool(contract.get("operator_only")),
        "zarr_backup_copy_before_labeling": bool(contract.get("copy_before_labeling")),
        "zarr_backup_required_before_invite": bool(
            contract.get("mutable_zarr_backup_required_before_invite")
        ),
        "zarr_backup_labelers_do_not_edit_zarrs_directly": bool(
            contract.get("labelers_do_not_edit_zarrs_directly")
        ),
        "zarr_backup_labelers_do_not_receive_backup_paths": bool(
            contract.get("labelers_do_not_receive_backup_paths")
        ),
        "zarr_backup_pause_or_unassign_before_restore": bool(
            contract.get("pause_or_unassign_recording_before_restore")
        ),
        "zarr_backup_rollback_owner": str(contract.get("rollback_owner") or ""),
        "zarr_backup_validation_gate": str(contract.get("validation_gate") or ""),
        "zarr_backup_plan": backup_plan,
        "zarr_backup_plan_present": backup_plan_present,
        "zarr_backup_plan_required": backup_plan_required,
        "zarr_backup_required_targets": backup_required_targets,
        "zarr_backup_missing_path_tasks": tasks_missing_zarr_path,
        "zarr_backup_required_targets_by_role": (
            contract.get("backup_required_targets_by_role")
            if isinstance(contract.get("backup_required_targets_by_role"), Mapping)
            else {}
        ),
    }



def _handoff_task_state_policy_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_browser_mutation_target_contract_policy', '_browser_task_state_contract_policy'))
    _browser_mutation_target_contract_policy = field_helpers['_browser_mutation_target_contract_policy']
    _browser_task_state_contract_policy = field_helpers['_browser_task_state_contract_policy']

    policy = (
        handoff.get("task_state_policy")
        if isinstance(handoff.get("task_state_policy"), Mapping)
        else None
    )
    task_contract = _browser_task_state_contract_policy(policy or {})
    mutation_target_contract = _browser_mutation_target_contract_policy(task_state_policy=policy or {})
    present = policy is not None
    task_contract_ready = bool(task_contract.get("ready"))
    mutation_target_ready = bool(mutation_target_contract.get("ready"))
    ready = present and task_contract_ready and mutation_target_ready
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so task-state policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so completed tasks are read-only to labelers, "
            "non-startable task states reject labeler opens, operator reopen is required before "
            "more labeling, and browser mutations reject client-selected targets unless they "
            "carry the current server target token."
        )
    return {
        "task_state_policy_present": present,
        "task_state_policy_ready": ready,
        "task_state_policy_readiness": readiness,
        "task_state_policy_operator_action": action,
        "task_state_startable_task_states": json.dumps(
            task_contract.get("startable_task_states") or []
        ),
        "task_state_completed_tasks_read_only": bool(task_contract.get("completed_tasks_read_only")),
        "task_state_completed_tasks_open_new_sessions": bool(
            task_contract.get("completed_tasks_open_new_sessions")
        ),
        "task_state_completed_task_open_requests": str(
            task_contract.get("completed_task_open_requests") or ""
        ),
        "task_state_completed_task_save_requests": str(
            task_contract.get("completed_task_save_requests") or ""
        ),
        "task_state_non_startable_task_open_requests": str(
            task_contract.get("non_startable_task_open_requests") or ""
        ),
        "task_state_non_startable_task_save_requests": str(
            task_contract.get("non_startable_task_save_requests") or ""
        ),
        "task_state_completion_closes_open_sessions": bool(
            task_contract.get("completion_closes_open_sessions")
        ),
        "task_state_reopen_authority": str(task_contract.get("reopen_authority") or ""),
        "task_state_reopen_required_for_more_labeling": bool(
            task_contract.get("reopen_required_for_more_labeling")
        ),
        "task_state_operator_reopen_required_before_more_labeling": bool(
            task_contract.get("operator_reopen_required_before_more_labeling")
        ),
        "task_state_labeler_promotion_retry_requires_open_task": bool(
            task_contract.get("labeler_promotion_retry_requires_open_task")
        ),
        "task_state_labeler_promotion_retry_mutation_enabled": bool(
            task_contract.get("labeler_promotion_retry_mutation_enabled")
        ),
        "task_state_labeler_promotion_retry_rejection_error": str(
            task_contract.get("labeler_promotion_retry_rejection_error") or ""
        ),
        "task_state_ordinary_labeler_promotion_retry_mutation": str(
            task_contract.get("ordinary_labeler_promotion_retry_mutation") or ""
        ),
        "task_state_completion_requires_current_session": bool(
            task_contract.get("task_completion_requires_current_session")
        ),
        "task_state_browser_mutation_target_selectors": (
            str(policy.get("browser_mutation_target_selectors") or "") if policy else ""
        ),
        "task_state_browser_mutation_target_token": (
            str(policy.get("browser_mutation_target_token") or "") if policy else ""
        ),
        "task_state_rejects_client_target_selectors": bool(
            mutation_target_contract.get("rejects_client_target_selectors")
        ),
        "task_state_requires_current_target_token": bool(
            mutation_target_contract.get("requires_current_target_token")
        ),
        "task_state_workflow_contracts_require_target_token": bool(
            mutation_target_contract.get("workflow_contracts_require_target_token")
        ),
    }



def _handoff_session_guard_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_session_guard_contract_policy',))
    _session_guard_contract_policy = field_helpers['_session_guard_contract_policy']

    policy = (
        handoff.get("session_guard_policy")
        if isinstance(handoff.get("session_guard_policy"), Mapping)
        else None
    )
    contract = _session_guard_contract_policy(policy or {})
    present = policy is not None
    ready = present and bool(contract.get("ready"))
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so session-guard policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so browser mutations require a current unexpired session, "
            "reject stale, superseded, or non-startable task sessions, require the current target token, "
            "and expose closure-event support."
        )
    return {
        "session_guard_policy_present": present,
        "session_guard_policy_ready": ready,
        "session_guard_policy_readiness": readiness,
        "session_guard_policy_operator_action": action,
        "session_guard_requires_current_session": bool(contract.get("requires_current_session")),
        "session_guard_requires_unexpired_session": bool(contract.get("requires_unexpired_session")),
        "session_guard_stale_tab_save_rejected": bool(contract.get("stale_tab_save_rejected")),
        "session_guard_superseded_sessions_rejected": bool(contract.get("superseded_sessions_rejected")),
        "session_guard_non_startable_task_sessions_rejected": bool(
            contract.get("non_startable_task_sessions_rejected")
        ),
        "session_guard_target_token_required_for_mutation": bool(
            contract.get("target_token_required_for_mutation")
        ),
        "session_guard_labeler_promotion_retry_requires_current_session": bool(
            contract.get("labeler_promotion_retry_requires_current_session")
        ),
        "session_guard_closure_event_support": bool(contract.get("session_closure_event_support")),
        "session_guard_rejects_after_reassignment": bool(contract.get("rejects_after_reassignment")),
        "session_guard_rejects_after_completion_or_reopen": bool(
            contract.get("rejects_after_completion_or_reopen")
        ),
        "session_guard_rejects_after_expiration": bool(contract.get("rejects_after_expiration")),
        "session_guard_rejects_after_target_navigation": bool(
            contract.get("rejects_after_target_navigation")
        ),
    }



def _handoff_labeler_safety_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_browser_payload_redaction_contract_policy', '_browser_signed_link_policy', '_expected_user_guard_contract_policy', '_signed_link_contract_policy'))
    _browser_payload_redaction_contract_policy = field_helpers['_browser_payload_redaction_contract_policy']
    _browser_signed_link_policy = field_helpers['_browser_signed_link_policy']
    _expected_user_guard_contract_policy = field_helpers['_expected_user_guard_contract_policy']
    _signed_link_contract_policy = field_helpers['_signed_link_contract_policy']

    policy = (
        handoff.get("labeler_safety")
        if isinstance(handoff.get("labeler_safety"), Mapping)
        else None
    )
    redaction_contract = _browser_payload_redaction_contract_policy(policy or {})
    signed_link_contract = _signed_link_contract_policy(
        handoff.get("signed_link_policy")
        if isinstance(handoff.get("signed_link_policy"), Mapping)
        else _browser_signed_link_policy()
    )
    expected_user_guard_contract = _expected_user_guard_contract_policy(
        policy or {},
        signed_link_contract,
    )
    present = policy is not None
    browser_only = bool(policy.get("browser_only")) if policy else False
    labeler_runtime_surface = str(policy.get("labeler_runtime_surface") or "") if policy else ""
    requires_local_palette_installation = (
        bool(policy.get("requires_local_palette_installation")) if policy else True
    )
    requires_local_crimson_installation = (
        bool(policy.get("requires_local_crimson_installation")) if policy else True
    )
    requires_local_conda_environment = (
        bool(policy.get("requires_local_conda_environment")) if policy else True
    )
    requires_local_project_dependencies = (
        bool(policy.get("requires_local_project_dependencies")) if policy else True
    )
    no_local_install_required = (
        labeler_runtime_surface == "browser"
        and not requires_local_palette_installation
        and not requires_local_crimson_installation
        and not requires_local_conda_environment
        and not requires_local_project_dependencies
    )
    identity_check_required = bool(policy.get("dashboard_identity_check_required")) if policy else False
    identity_probe_expected_user_guard_required = bool(
        policy.get("identity_probe_expected_user_guard_required")
    ) if policy else False
    identity_probe_diagnostic_only = bool(policy.get("identity_probe_diagnostic_only")) if policy else False
    identity_probe_does_not_authorize_work = bool(policy.get("identity_probe_does_not_authorize_work")) if policy else False
    identity_probe_unknown_user_blocks_work_surfaces = bool(
        policy.get("identity_probe_unknown_user_blocks_work_surfaces")
    ) if policy else False
    identity_probe_success_launch_ctas_rendered = bool(
        policy.get("identity_probe_success_launch_ctas_rendered")
    ) if policy else False
    identity_probe_failed_launch_ctas_suppressed = bool(
        policy.get("identity_probe_failed_launch_ctas_suppressed")
    ) if policy else False
    identity_probe_failed_support_urls_diagnostic_only = bool(
        policy.get("identity_probe_failed_support_urls_diagnostic_only")
    ) if policy else False
    no_direct_zarr_edits = bool(policy.get("no_direct_zarr_edits")) if policy else False
    no_forwarding = bool(policy.get("no_forwarding_links_or_handoffs")) if policy else False
    redaction_ready = bool(redaction_contract.get("ready"))
    expected_user_guards_ready = bool(expected_user_guard_contract.get("ready"))
    ready = (
        present
        and browser_only
        and no_local_install_required
        and identity_check_required
        and identity_probe_expected_user_guard_required
        and identity_probe_diagnostic_only
        and identity_probe_does_not_authorize_work
        and identity_probe_unknown_user_blocks_work_surfaces
        and identity_probe_success_launch_ctas_rendered
        and identity_probe_failed_launch_ctas_suppressed
        and identity_probe_failed_support_urls_diagnostic_only
        and no_direct_zarr_edits
        and no_forwarding
        and redaction_ready
        and expected_user_guards_ready
    )
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so labeler safety metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so labelers are browser-only with no local Palette, Crimson, Conda, "
            "or project dependency installation, identity checks and expected-user guarded identity probes are configured, raw Zarr "
            "paths/task scope are redacted, identity probe remains diagnostic-only/non-authorizing, failed identity "
            "probes suppress launch CTAs, support URLs remain diagnostic-only, and direct "
            "Zarr edits/link forwarding are prohibited."
        )
    return {
        "labeler_safety_policy_present": present,
        "labeler_safety_ready": ready,
        "labeler_safety_readiness": readiness,
        "labeler_safety_operator_action": action,
        "labeler_safety_browser_only": browser_only,
        "labeler_safety_labeler_runtime_surface": labeler_runtime_surface,
        "labeler_safety_requires_local_palette_installation": requires_local_palette_installation,
        "labeler_safety_requires_local_crimson_installation": requires_local_crimson_installation,
        "labeler_safety_requires_local_conda_environment": requires_local_conda_environment,
        "labeler_safety_requires_local_project_dependencies": requires_local_project_dependencies,
        "labeler_safety_no_local_install_required": no_local_install_required,
        "labeler_safety_identity_check_required": identity_check_required,
        "labeler_safety_identity_probe_expected_user_guard_required": (
            identity_probe_expected_user_guard_required
        ),
        "labeler_safety_identity_probe_diagnostic_only": identity_probe_diagnostic_only,
        "labeler_safety_identity_probe_does_not_authorize_work": identity_probe_does_not_authorize_work,
        "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces": (
            identity_probe_unknown_user_blocks_work_surfaces
        ),
        "labeler_safety_identity_probe_success_launch_ctas_rendered": (
            identity_probe_success_launch_ctas_rendered
        ),
        "labeler_safety_identity_probe_failed_launch_ctas_suppressed": (
            identity_probe_failed_launch_ctas_suppressed
        ),
        "labeler_safety_identity_probe_failed_support_urls_diagnostic_only": (
            identity_probe_failed_support_urls_diagnostic_only
        ),
        "labeler_safety_no_direct_zarr_edits": no_direct_zarr_edits,
        "labeler_safety_no_forwarding_links_or_handoffs": no_forwarding,
        "labeler_safety_expected_user_guards_ready": expected_user_guards_ready,
        "labeler_safety_browser_payload_redaction_ready": redaction_ready,
        "labeler_safety_browser_receives_task_scope": bool(
            redaction_contract.get("browser_receives_task_scope")
        ),
        "labeler_safety_browser_receives_raw_zarr_paths": bool(
            redaction_contract.get("browser_receives_raw_zarr_paths")
        ),
        "labeler_safety_missing_or_mismatched_expected_user_guards": (
            expected_user_guard_contract.get("missing_or_mismatched_guards")
            if isinstance(expected_user_guard_contract.get("missing_or_mismatched_guards"), list)
            else []
        ),
    }



def _handoff_signed_link_policy_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_signed_link_contract_policy',))
    _signed_link_contract_policy = field_helpers['_signed_link_contract_policy']

    policy = (
        handoff.get("signed_link_policy")
        if isinstance(handoff.get("signed_link_policy"), Mapping)
        else None
    )
    contract = _signed_link_contract_policy(policy or {})
    present = policy is not None
    ready = present and bool(contract.get("ready"))
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so signed-link policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so signed links are task-specific expected-user-bound "
            "entry hints, not authorization grants or identity proofs, and enforce runtime "
            "operator-validation start gates before session creation."
        )
    return {
        "signed_link_policy_present": present,
        "signed_link_policy_ready": ready,
        "signed_link_policy_readiness": readiness,
        "signed_link_policy_operator_action": action,
        "signed_link_task_specific": bool(contract.get("task_specific_links")),
        "signed_link_authorization_grant": bool(contract.get("authorization_grant")),
        "signed_link_binds_expected_user": bool(contract.get("binds_expected_user_in_new_links")),
        "signed_link_expected_user_required_on_open": bool(contract.get("expected_user_required_on_open")),
        "signed_link_forwarded_links_recheck_identity": bool(
            contract.get("forwarded_signed_links_recheck_identity")
        ),
        "signed_link_requires_server_side_task_authorization": bool(
            contract.get("requires_server_side_task_authorization")
        ),
        "signed_link_runtime_operator_validation_start_gate_enforced": bool(
            contract.get("runtime_operator_validation_start_gate_enforced")
        ),
        "signed_link_operator_validation_start_gate_checked_before_session_create": bool(
            contract.get("operator_validation_start_gate_checked_before_session_create")
        ),
    }



def _handoff_labeler_route_authorization_fields(
    handoff: Mapping[str, object],
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, object]:
    field_helpers = _require_helpers(helpers, ('_labeler_route_authorization_contract_policy',))
    _labeler_route_authorization_contract_policy = field_helpers['_labeler_route_authorization_contract_policy']

    policy = (
        handoff.get("labeler_route_authorization_policy")
        if isinstance(handoff.get("labeler_route_authorization_policy"), Mapping)
        else None
    )
    checklist = (
        handoff.get("labeler_route_authorization_checklist")
        if isinstance(handoff.get("labeler_route_authorization_checklist"), Mapping)
        else None
    )
    contract = _labeler_route_authorization_contract_policy(policy or {})
    present = policy is not None
    runtime_checklist_present = checklist is not None
    runtime_checklist_ready = bool((checklist or {}).get("ready"))
    single_owner_store_proof_required = bool(
        contract.get("single_owner_store_proof_required_for_browser_work")
    )
    single_owner_store_proof_ready = bool(
        (checklist or {}).get("single_owner_store_proof_ready")
    )
    assignment_ownership_integrity_ok = bool(
        (checklist or {}).get("assignment_ownership_integrity_ok")
    )
    duplicate_active_owner_count = int(
        (checklist or {}).get("duplicate_active_owner_count") or 0
    )
    browser_mutation_target_resolved_server_side = bool(
        (checklist or {}).get("browser_mutation_target_resolved_server_side")
    )
    labelers_mutate_assigned_training_zarrs = bool(
        (checklist or {}).get("labelers_mutate_assigned_training_zarrs")
    )
    labelers_mutate_intermediate_csvs = bool(
        (checklist or {}).get("labelers_mutate_intermediate_csvs")
    )
    runtime_store_proof_ready = (
        not single_owner_store_proof_required
        or (
            runtime_checklist_present
            and single_owner_store_proof_ready
            and assignment_ownership_integrity_ok
            and duplicate_active_owner_count == 0
            and browser_mutation_target_resolved_server_side
            and labelers_mutate_assigned_training_zarrs
            and not labelers_mutate_intermediate_csvs
        )
    )
    ready = present and bool(contract.get("ready")) and runtime_store_proof_ready
    if ready:
        readiness = "passed"
        action = ""
    elif not present:
        readiness = "missing_policy"
        action = "Regenerate the handoff so labeler-route authorization policy metadata is present before sharing links."
    else:
        readiness = "needs_review"
        action = (
            "Regenerate or repair the handoff so copied queue, dashboard, and signed task links require "
            "resolved identity, expected-user match, known assignment-store user, active assignment, "
            "startable task-state checks, single-owner store proof, assignment-integrity OK, zero duplicate "
            "active owners, runtime operator-validation start-gate rechecks for signed links, "
            "server-resolved training-Zarr targets, and no intermediate CSV mutation; "
            "the runtime route authorization checklist must include single_owner_store_proof_ready=true, "
            "assignment_ownership_integrity_ok=true, browser_mutation_target_resolved_server_side=true, "
            "labelers_mutate_assigned_training_zarrs=true, "
            "forwarded_signed_links_recheck_runtime_operator_validation_start_gate=true, and "
            "labelers_mutate_intermediate_csvs=false."
        )
    fields = {
        "labeler_route_authorization_policy_present": present,
        "labeler_route_authorization_ready": ready,
        "labeler_route_authorization_readiness": readiness,
        "labeler_route_authorization_operator_action": action,
        "labeler_route_authorization_expected_user_match_required": bool(
            contract.get("expected_user_must_match_resolved_user")
        ),
        "labeler_route_authorization_known_user_required": bool(
            contract.get("known_assignment_store_user_required")
        ),
        "labeler_route_authorization_active_assignment_required": bool(
            contract.get("task_open_requires_active_assignment")
        ),
        "labeler_route_authorization_task_open_requires_active_assignment": bool(
            contract.get("task_open_requires_active_assignment")
        ),
        "labeler_route_authorization_task_open_requires_startable_task_state": bool(
            contract.get("task_open_requires_startable_task_state")
        ),
        "labeler_route_authorization_startable_task_states": json.dumps(
            contract.get("startable_task_states") or []
        ),
        "labeler_route_authorization_signed_links_are_entry_hints": bool(
            contract.get("signed_links_are_entry_hints_not_authorization")
        ),
        "labeler_route_authorization_signed_links_recheck_runtime_start_gate": bool(
            contract.get("forwarded_signed_links_recheck_runtime_operator_validation_start_gate")
        ),
        "labeler_route_authorization_forwarded_links_rechecked": bool(
            contract.get("copied_links_rechecked_server_side")
        ),
        "labeler_route_authorization_runtime_checklist_present": runtime_checklist_present,
        "labeler_route_authorization_runtime_checklist_ready": runtime_checklist_ready,
        "labeler_route_authorization_single_owner_store_proof_ready": (
            single_owner_store_proof_ready
        ),
        "labeler_route_authorization_assignment_ownership_integrity_ok": (
            assignment_ownership_integrity_ok
        ),
        "labeler_route_authorization_duplicate_active_owner_count": (
            duplicate_active_owner_count
        ),
        "labeler_route_authorization_browser_mutation_target_resolved_server_side": (
            browser_mutation_target_resolved_server_side
        ),
        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs": (
            labelers_mutate_assigned_training_zarrs
        ),
        "labeler_route_authorization_labelers_mutate_intermediate_csvs": (
            labelers_mutate_intermediate_csvs
        ),
        "labeler_route_authorization_single_owner_store_proof_required_for_browser_work": bool(
            contract.get("single_owner_store_proof_required_for_browser_work")
        ),
        "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok": bool(
            contract.get("single_owner_store_proof_requires_integrity_ok")
        ),
        "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners": bool(
            contract.get("single_owner_store_proof_requires_zero_duplicate_active_owners")
        ),
        "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target": bool(
            contract.get("single_owner_store_proof_requires_training_zarr_target")
        ),
        "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation": bool(
            contract.get("single_owner_store_proof_rejects_intermediate_csv_mutation")
        ),
    }
    if not runtime_checklist_present:
        for key in list(fields):
            if key in handoff:
                fields[key] = handoff[key]
    return fields


