from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


_ROSTER_HELPER_NAMES = (
    "_browser_mutation_target_contract_summary",
    "_browser_mutation_write_policy",
    "_dataset_queue_direct_start_policy_fields",
    "_direct_browser_start_contract_summary",
    "_direct_browser_start_contract_summary_fields",
    "_handoff_assignment_ownership_fields",
    "_handoff_browser_mutation_write_fields",
    "_handoff_browser_response_security_fields",
    "_handoff_dataset_queue_start_fields",
    "_handoff_entry_artifact_fields",
    "_handoff_known_user_status_fields",
    "_handoff_labeler_route_authorization_fields",
    "_handoff_labeler_safety_fields",
    "_handoff_mutation_audit_fields",
    "_handoff_operator_recovery_fields",
    "_handoff_ready_to_send",
    "_handoff_session_guard_fields",
    "_handoff_signed_link_policy_fields",
    "_handoff_task_state_policy_fields",
    "_handoff_zarr_backup_fields",
    "_labeler_work_completion_fields",
    "_operator_recovery_policy",
    "_operator_validation_command_template_fields",
    "_operator_validation_gate_flat_fieldnames",
    "_operator_validation_gate_flat_fields",
    "_operator_validation_public_fields",
    "_operator_validation_visibility_fields",
    "_personalized_launch_readiness_field_names",
    "_personalized_launch_readiness_summary",
    "_public_reassignment_session_safety_fields",
    "_queue_first_entry_contract_flat_fields",
    "_runtime_operator_validation_gate_cli_policy_fields",
    "_safe_share_checklist_gate_status_fields_from_operator_validation",
    "_safe_share_external_launch_evidence_gap_field_names",
    "_safe_share_gate_flat_fields",
    "_safe_share_gate_policy"
)


def _require_roster_helpers(
    helpers: Mapping[str, Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    missing = [name for name in _ROSTER_HELPER_NAMES if name not in helpers]
    if missing:
        raise KeyError(f"missing user handoff roster helper(s): {', '.join(missing)}")
    return {name: helpers[name] for name in _ROSTER_HELPER_NAMES}


def _write_user_handoffs_roster_csv(
    index: dict[str, object],
    output_path: Path,
    *,
    helpers: Mapping[str, Callable[..., Any]],
) -> None:
    import csv

    roster_helpers = _require_roster_helpers(helpers)
    _browser_mutation_target_contract_summary = roster_helpers["_browser_mutation_target_contract_summary"]
    _browser_mutation_write_policy = roster_helpers["_browser_mutation_write_policy"]
    _dataset_queue_direct_start_policy_fields = roster_helpers["_dataset_queue_direct_start_policy_fields"]
    _direct_browser_start_contract_summary = roster_helpers["_direct_browser_start_contract_summary"]
    _direct_browser_start_contract_summary_fields = roster_helpers["_direct_browser_start_contract_summary_fields"]
    _handoff_assignment_ownership_fields = roster_helpers["_handoff_assignment_ownership_fields"]
    _handoff_browser_mutation_write_fields = roster_helpers["_handoff_browser_mutation_write_fields"]
    _handoff_browser_response_security_fields = roster_helpers["_handoff_browser_response_security_fields"]
    _handoff_dataset_queue_start_fields = roster_helpers["_handoff_dataset_queue_start_fields"]
    _handoff_entry_artifact_fields = roster_helpers["_handoff_entry_artifact_fields"]
    _handoff_known_user_status_fields = roster_helpers["_handoff_known_user_status_fields"]
    _handoff_labeler_route_authorization_fields = roster_helpers["_handoff_labeler_route_authorization_fields"]
    _handoff_labeler_safety_fields = roster_helpers["_handoff_labeler_safety_fields"]
    _handoff_mutation_audit_fields = roster_helpers["_handoff_mutation_audit_fields"]
    _handoff_operator_recovery_fields = roster_helpers["_handoff_operator_recovery_fields"]
    _handoff_ready_to_send = roster_helpers["_handoff_ready_to_send"]
    _handoff_session_guard_fields = roster_helpers["_handoff_session_guard_fields"]
    _handoff_signed_link_policy_fields = roster_helpers["_handoff_signed_link_policy_fields"]
    _handoff_task_state_policy_fields = roster_helpers["_handoff_task_state_policy_fields"]
    _handoff_zarr_backup_fields = roster_helpers["_handoff_zarr_backup_fields"]
    _labeler_work_completion_fields = roster_helpers["_labeler_work_completion_fields"]
    _operator_recovery_policy = roster_helpers["_operator_recovery_policy"]
    _operator_validation_command_template_fields = roster_helpers["_operator_validation_command_template_fields"]
    _operator_validation_gate_flat_fieldnames = roster_helpers["_operator_validation_gate_flat_fieldnames"]
    _operator_validation_gate_flat_fields = roster_helpers["_operator_validation_gate_flat_fields"]
    _operator_validation_public_fields = roster_helpers["_operator_validation_public_fields"]
    _operator_validation_visibility_fields = roster_helpers["_operator_validation_visibility_fields"]
    _personalized_launch_readiness_field_names = roster_helpers["_personalized_launch_readiness_field_names"]
    _personalized_launch_readiness_summary = roster_helpers["_personalized_launch_readiness_summary"]
    _public_reassignment_session_safety_fields = roster_helpers["_public_reassignment_session_safety_fields"]
    _queue_first_entry_contract_flat_fields = roster_helpers["_queue_first_entry_contract_flat_fields"]
    _runtime_operator_validation_gate_cli_policy_fields = roster_helpers["_runtime_operator_validation_gate_cli_policy_fields"]
    _safe_share_checklist_gate_status_fields_from_operator_validation = roster_helpers["_safe_share_checklist_gate_status_fields_from_operator_validation"]
    _safe_share_external_launch_evidence_gap_field_names = roster_helpers["_safe_share_external_launch_evidence_gap_field_names"]
    _safe_share_gate_flat_fields = roster_helpers["_safe_share_gate_flat_fields"]
    _safe_share_gate_policy = roster_helpers["_safe_share_gate_policy"]

    handoffs = index.get("handoffs", []) if isinstance(index.get("handoffs"), list) else []
    browser_mutation_write_policy = (
        index.get("browser_mutation_write_policy")
        if isinstance(index.get("browser_mutation_write_policy"), Mapping)
        else _browser_mutation_write_policy()
    )
    operator_recovery_policy = (
        index.get("operator_recovery_policy")
        if isinstance(index.get("operator_recovery_policy"), Mapping)
        else _operator_recovery_policy()
    )
    safe_share_gate = (
        index.get("safe_share_gate")
        if isinstance(index.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "user",
                "ok",
                "known_labeler",
                "known_user_active_assignment_count",
                "known_user_assignment_count",
                "known_user_readiness",
                "known_user_operator_action",
                "assignment_ownership_ok",
                "assignment_active_assignment_count",
                "assignment_duplicate_active_owner_count",
                "assignment_ownership_readiness",
                "assignment_ownership_operator_action",
                "assignment_ownership_contract_schema",
                "assignment_ownership_contract_ready",
                "assignment_ownership_contract_assignment_scope",
                "assignment_ownership_contract_recording_assignment_key",
                "assignment_ownership_contract_recording_id_primary_key",
                "assignment_ownership_contract_schema_enforced_recording_primary_key",
                "assignment_ownership_contract_store_recording_id_primary_key",
                "assignment_ownership_contract_store_schema_enforced_recording_primary_key",
                "assignment_ownership_contract_schema_integrity_source",
                "assignment_ownership_contract_primary_key_columns",
                "assignment_ownership_contract_one_current_assignment_row_per_recording",
                "assignment_ownership_contract_one_active_owner",
                "assignment_ownership_contract_multiple_labelers_per_recording_allowed",
                "assignment_ownership_contract_reassignment_replaces_owner",
                "assignment_ownership_contract_stale_sessions_closed_on_reassignment",
                "assignment_ownership_contract_stale_sessions_closed_before_assignment_update",
                "assignment_ownership_contract_reassignment_target_validated_before_session_closure",
                "assignment_ownership_contract_session_closure_and_assignment_update_atomic",
                "assignment_ownership_contract_raw_assignment_change_blocks_open_sessions",
                "assignment_ownership_contract_assignment_manifests_are_control_plane",
                "assignment_ownership_contract_duplicate_manifest_rows_do_not_create_multiple_owners",
                "assignment_ownership_contract_assignment_user_match_required_for_mutation",
                "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner",
                "assignment_ownership_contract_browser_mutation_target_resolved_server_side",
                "assignment_ownership_contract_browser_mutation_target_source",
                "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs",
                "assignment_ownership_contract_labelers_mutate_intermediate_csvs",
                "assignment_ownership_contract_store_single_owner_assignment_contract_present",
                "assignment_ownership_contract_store_single_owner_assignment_contract_ready",
                "assignment_ownership_contract_store_single_owner_assignment_contract_met",
                "assignment_ownership_contract_store_single_owner_assignment_contract_schema",
                "assignment_ownership_contract_assignment_ownership_integrity_ok",
                "assignment_ownership_contract_active_assignment_count",
                "assignment_ownership_contract_unique_active_recording_count",
                "assignment_ownership_contract_duplicate_active_owner_count",
                "single_owner_policy_assignment_scope",
                "single_owner_policy_recording_assignment_key",
                "single_owner_policy_one_current_assignment_row_per_recording",
                "single_owner_policy_one_active_owner",
                "single_owner_policy_multiple_labelers_per_recording_allowed",
                "single_owner_policy_assignment_manifests_are_control_plane",
                "single_owner_policy_duplicate_manifest_rows_do_not_create_multiple_owners",
                "single_owner_policy_assignment_user_match_required_for_mutation",
                "single_owner_policy_browser_mutation_requires_current_assignment_owner",
                "single_owner_policy_browser_mutation_target_resolved_server_side",
                "single_owner_policy_browser_mutation_target_source",
                "single_owner_policy_labelers_mutate_assigned_training_zarrs",
                "single_owner_policy_labelers_mutate_intermediate_csvs",
                "single_owner_policy_contract_met",
                "guarded_links_ready",
                "missing_guarded_links",
                "handoff_artifacts_ready",
                "missing_handoff_artifacts",
                "handoff_entry_readiness",
                "handoff_entry_operator_action",
                "preferred_labeler_entrypoint",
                "preferred_labeler_entry_url",
                "personalized_labeler_entrypoint",
                "personalized_labeler_entry_url",
                "personalized_launch_readiness_schema",
                "personalized_launch_readiness_fields",
                "personalized_launch_readiness_field_count",
                "personalized_launch_readiness_personalized_labeler_entry_url",
                "personalized_launch_readiness_labeler_start_ready",
                "personalized_launch_readiness_labeler_start_status",
                "personalized_launch_readiness_labeler_work_completion_status",
                "personalized_launch_readiness_safe_share_gate_id",
                "personalized_launch_readiness_safe_share_checklist_gate_evidence_complete",
                "personalized_launch_readiness_external_launch_evidence_gap_action_required",
                "personalized_launch_readiness_external_launch_evidence_gap_count",
                "personalized_launch_readiness_external_launch_evidence_gap_gate_ids",
                "personalized_launch_readiness_external_launch_evidence_gap_statuses",
                "personalized_launch_readiness_external_launch_evidence_gap_summary",
                "personalized_launch_readiness_external_launch_evidence_gap_todos",
                "personalized_launch_readiness_external_launch_evidence_gap_todo_count",
                "personalized_launch_readiness_external_launch_evidence_gap_todo_fields",
                "personalized_launch_readiness_external_launch_evidence_gap_template_paths_by_gate_id",
                "personalized_launch_readiness_external_launch_evidence_gap_record_command_ids_by_gate_id",
                "personalized_launch_readiness_browser_label_write_target",
                "personalized_launch_readiness_browser_writes_csv_or_handoff_files",
                "personalized_launch_readiness_browser_has_direct_zarr_write_authority",
                "queue_first_entry_contract_schema",
                "queue_first_entry_contract_ready",
                "queue_first_entry_contract_preferred_labeler_entrypoint",
                "queue_first_entry_contract_preferred_labeler_entry_url",
                "queue_first_entry_contract_personalized_labeler_entrypoint",
                "queue_first_entry_contract_personalized_labeler_entry_url",
                "queue_first_entry_contract_personalized_entry_required",
                "queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue",
                "queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue",
                "queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded",
                "queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded",
                "queue_first_entry_contract_landing_ready",
                "queue_first_entry_contract_labeling_home_ready",
                "queue_first_entry_contract_dataset_queue_ready",
                "queue_first_entry_contract_personal_dataset_queue_ready",
                "queue_first_entry_contract_personal_work_ready",
                "queue_first_entry_contract_queue_first_paths_ready",
                "queue_first_entry_contract_datasets_waiting_aliases_ready",
                "queue_first_entry_contract_expected_user_landing_guard",
                "queue_first_entry_contract_expected_user_queue_guard",
                "queue_first_entry_contract_expected_user_dashboard_guard",
                "labeler_landing_link_role",
                "personal_dataset_queue_link_role",
                "dataset_queue_link_role",
                "canonical_dataset_queue_link_role",
                "dashboard_link_role",
                "identity_probe_link_role",
                "task_links_role",
                "preferred_labeler_entry_url_matches_dataset_queue",
                "preferred_labeler_entry_url_matches_personal_dataset_queue",
                "personalized_labeler_entry_url_matches_personal_dataset_queue",
                "identity_check_url",
                "labeler_safety_policy_present",
                "labeler_safety_ready",
                "labeler_safety_readiness",
                "labeler_safety_operator_action",
                "labeler_safety_browser_only",
                "labeler_safety_labeler_runtime_surface",
                "labeler_safety_requires_local_palette_installation",
                "labeler_safety_requires_local_crimson_installation",
                "labeler_safety_requires_local_conda_environment",
                "labeler_safety_requires_local_project_dependencies",
                "labeler_safety_no_local_install_required",
                "labeler_safety_identity_check_required",
                "labeler_safety_identity_probe_expected_user_guard_required",
                "labeler_safety_identity_probe_diagnostic_only",
                "labeler_safety_identity_probe_does_not_authorize_work",
                "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces",
                "labeler_safety_identity_probe_success_launch_ctas_rendered",
                "labeler_safety_identity_probe_failed_launch_ctas_suppressed",
                "labeler_safety_identity_probe_failed_support_urls_diagnostic_only",
                "labeler_safety_no_direct_zarr_edits",
                "labeler_safety_no_forwarding_links_or_handoffs",
                "labeler_safety_expected_user_guards_ready",
                "labeler_safety_browser_payload_redaction_ready",
                "labeler_safety_browser_receives_task_scope",
                "labeler_safety_browser_receives_raw_zarr_paths",
                "labeler_safety_missing_or_mismatched_expected_user_guards",
                "labeler_route_authorization_policy_present",
                "labeler_route_authorization_ready",
                "labeler_route_authorization_readiness",
                "labeler_route_authorization_operator_action",
                "labeler_route_authorization_expected_user_match_required",
                "labeler_route_authorization_known_user_required",
                "labeler_route_authorization_active_assignment_required",
                "labeler_route_authorization_task_open_requires_active_assignment",
                "labeler_route_authorization_task_open_requires_startable_task_state",
                "labeler_route_authorization_startable_task_states",
                "labeler_route_authorization_signed_links_are_entry_hints",
                "labeler_route_authorization_signed_links_recheck_runtime_start_gate",
                "labeler_route_authorization_forwarded_links_rechecked",
                "labeler_route_authorization_runtime_checklist_present",
                "labeler_route_authorization_runtime_checklist_ready",
                "labeler_route_authorization_single_owner_store_proof_ready",
                "labeler_route_authorization_assignment_ownership_integrity_ok",
                "labeler_route_authorization_duplicate_active_owner_count",
                "labeler_route_authorization_browser_mutation_target_resolved_server_side",
                "labeler_route_authorization_labelers_mutate_assigned_training_zarrs",
                "labeler_route_authorization_labelers_mutate_intermediate_csvs",
                "labeler_route_authorization_single_owner_store_proof_required_for_browser_work",
                "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok",
                "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners",
                "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target",
                "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation",
                "signed_link_policy_present",
                "signed_link_policy_ready",
                "signed_link_policy_readiness",
                "signed_link_policy_operator_action",
                "signed_link_task_specific",
                "signed_link_authorization_grant",
                "signed_link_binds_expected_user",
                "signed_link_expected_user_required_on_open",
                "signed_link_forwarded_links_recheck_identity",
                "signed_link_requires_server_side_task_authorization",
                "signed_link_runtime_operator_validation_start_gate_enforced",
                "signed_link_operator_validation_start_gate_checked_before_session_create",
                "session_guard_policy_present",
                "session_guard_policy_ready",
                "session_guard_policy_readiness",
                "session_guard_policy_operator_action",
                "session_guard_requires_current_session",
                "session_guard_requires_unexpired_session",
                "session_guard_stale_tab_save_rejected",
                "session_guard_superseded_sessions_rejected",
                "session_guard_non_startable_task_sessions_rejected",
                "session_guard_target_token_required_for_mutation",
                "session_guard_labeler_promotion_retry_requires_current_session",
                "session_guard_closure_event_support",
                "session_guard_rejects_after_reassignment",
                "session_guard_rejects_after_completion_or_reopen",
                "session_guard_rejects_after_expiration",
                "session_guard_rejects_after_target_navigation",
                "task_state_policy_present",
                "task_state_policy_ready",
                "task_state_policy_readiness",
                "task_state_policy_operator_action",
                "task_state_startable_task_states",
                "task_state_completed_tasks_read_only",
                "task_state_completed_tasks_open_new_sessions",
                "task_state_completed_task_open_requests",
                "task_state_completed_task_save_requests",
                "task_state_non_startable_task_open_requests",
                "task_state_non_startable_task_save_requests",
                "task_state_completion_closes_open_sessions",
                "task_state_reopen_authority",
                "task_state_reopen_required_for_more_labeling",
                "task_state_operator_reopen_required_before_more_labeling",
                "task_state_labeler_promotion_retry_requires_open_task",
                "task_state_labeler_promotion_retry_mutation_enabled",
                "task_state_labeler_promotion_retry_rejection_error",
                "task_state_ordinary_labeler_promotion_retry_mutation",
                "task_state_completion_requires_current_session",
                "task_state_browser_mutation_target_selectors",
                "task_state_browser_mutation_target_token",
                "task_state_rejects_client_target_selectors",
                "task_state_requires_current_target_token",
                "task_state_workflow_contracts_require_target_token",
                "operator_recovery_policy_present",
                "operator_recovery_ready",
                "operator_recovery_readiness",
                "operator_recovery_operator_action",
                "operator_recovery_contract_ready",
                "operator_recovery_reassignment_closes_previous_owner_sessions",
                "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update",
                "operator_recovery_reassignment_target_validated_before_session_closure",
                "operator_recovery_session_closure_and_assignment_update_atomic",
                "operator_recovery_task_reopen_operator_only",
                "operator_recovery_completion_closes_open_sessions",
                "operator_recovery_failed_promotion_retry_operator_only",
                "operator_recovery_session_closure_events_operator_inspectable",
                "operator_recovery_operator_repair_closes_or_supersedes_sessions",
                "operator_recovery_rollback_requires_backup_plan",
                "operator_recovery_bad_disposable_mutation_recovery_ready",
                "operator_recovery_disposable_mutation_smoke_requires_recovery_path_verification",
                "operator_recovery_restore_pauses_or_unassigns_recording_before_write",
                "operator_recovery_labelers_receive_recovery_write_authority",
                "operator_recovery_browser_recovery_mutations_direct",
                "operator_recovery_reassignment_session_repair_route",
                "operator_recovery_task_state_route",
                "operator_recovery_task_repair_route",
                "operator_recovery_audit_event_lookup_route",
                "operator_recovery_failed_promotion_retry_route",
                "operator_recovery_validation_gate",
                "zarr_backup_policy_present",
                "zarr_backup_ready",
                "zarr_backup_readiness",
                "zarr_backup_operator_action",
                "zarr_backup_contract_ready",
                "zarr_backup_read_only_plan",
                "zarr_backup_operator_only",
                "zarr_backup_copy_before_labeling",
                "zarr_backup_required_before_invite",
                "zarr_backup_labelers_do_not_edit_zarrs_directly",
                "zarr_backup_labelers_do_not_receive_backup_paths",
                "zarr_backup_pause_or_unassign_before_restore",
                "zarr_backup_rollback_owner",
                "zarr_backup_validation_gate",
                "zarr_backup_plan",
                "zarr_backup_plan_present",
                "zarr_backup_plan_required",
                "zarr_backup_required_targets",
                "zarr_backup_missing_path_tasks",
                "zarr_backup_required_targets_by_role",
                "mutation_audit_policy_present",
                "mutation_audit_ready",
                "mutation_audit_readiness",
                "mutation_audit_operator_action",
                "mutation_audit_contract_ready",
                "mutation_audit_event_store",
                "mutation_audit_append_only",
                "mutation_audit_server_records_events",
                "mutation_audit_browser_records_events_directly",
                "mutation_audit_browser_receives_write_credentials",
                "mutation_audit_per_workflow_contracts_include_provenance",
                "mutation_audit_required_event_fields_present",
                "mutation_audit_required_event_fields",
                "mutation_audit_timestamp_field",
                "mutation_audit_same_payload_retry_safe",
                "mutation_audit_duplicate_events_possible",
                "mutation_audit_validation_gate",
                "browser_response_security_policy_present",
                "browser_response_security_ready",
                "browser_response_security_readiness",
                "browser_response_security_operator_action",
                "browser_response_security_contract_ready",
                "browser_response_security_no_store_cache",
                "browser_response_security_pragma_no_cache",
                "browser_response_security_expires_zero",
                "browser_response_security_clickjacking_protection",
                "browser_response_security_mime_sniffing_protection",
                "browser_response_security_referrer_leakage_protection",
                "browser_response_security_csp_scope_ready",
                "browser_response_security_permissions_policy_ready",
                "browser_response_security_proxy_must_preserve_headers",
                "browser_response_security_cache_control",
                "browser_response_security_x_frame_options",
                "browser_response_security_x_content_type_options",
                "browser_response_security_referrer_policy",
                "browser_response_security_content_security_policy",
                "browser_response_security_permissions_policy",
                "ready_to_send",
                "safe_share_gate_schema",
                "safe_share_gate_id",
                "safe_share_requires_require_shareable_inspection",
                "safe_share_ready_to_send_is_sufficient",
                "safe_share_required_inspection_field",
                "safe_share_required_inspection_value",
                "safe_share_launch_blocking_evidence_gate_ids",
                "safe_share_launch_blocking_gate_statuses",
                "safe_share_launch_blocking_missing_gate_ids",
                "safe_share_launch_blocking_pending_gate_ids",
                "safe_share_launch_blocking_needs_review_gate_ids",
                "safe_share_launch_blocking_missing_evidence_gate_ids",
                "safe_share_launch_blocking_unknown_gate_ids",
                "safe_share_launch_blocking_satisfied_gate_ids",
                "safe_share_launch_blocking_unsatisfied_gate_ids",
                "safe_share_launch_blocking_next_actions",
                "safe_share_launch_blocking_next_action_detail_fields",
                "safe_share_launch_blocking_next_action_command_fields",
                "safe_share_launch_blocking_next_action_count",
                "safe_share_next_action_summary",
                *_safe_share_external_launch_evidence_gap_field_names(),
                "safe_share_launch_blocking_gate_count",
                "safe_share_launch_blocking_missing_gate_count",
                "safe_share_launch_blocking_pending_gate_count",
                "safe_share_launch_blocking_needs_review_gate_count",
                "safe_share_launch_blocking_missing_evidence_gate_count",
                "safe_share_launch_blocking_unknown_gate_count",
                "safe_share_launch_blocking_satisfied_gate_count",
                "safe_share_launch_blocking_unsatisfied_gate_count",
                "safe_share_checklist_gate_evidence_complete",
                "safe_share_checklist_operator_action",
                "safe_share_operator_action",
                "reassignment_session_safety",
                "reassignment_session_safety_ok",
                "reassignment_session_safety_blocks_labeler_mutation",
                "reassignment_session_safety_active_session_assignment_mismatch_count",
                "reassignment_session_safety_active_session_assignment_mismatch_session_ids",
                "reassignment_session_safety_active_session_assignment_mismatch_recording_ids",
                "reassignment_session_safety_requires_operator_recovery",
                "reassignment_session_safety_operator_action",
                "operator_validation_required_before_invite",
                "operator_validation_all_complete",
                "operator_validation_declared_all_complete",
                "operator_validation_ready_for_operator_validation",
                "operator_validation_gate_count",
                "operator_validation_status",
                "operator_validation_source",
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
                *_operator_validation_gate_flat_fieldnames(),
                "identity_personal_queue_evidence_status",
                "identity_personal_queue_evidence_ready_count",
                "identity_personal_queue_evidence_missing_count",
                "identity_personal_queue_evidence_ready_users",
                "identity_personal_queue_evidence_missing_users",
                "identity_personal_queue_evidence_missing_fields_by_user",
                "identity_all_users_have_personal_queue_evidence",
                "operator_validation_command_template_schema",
                "operator_validation_command_template_commands_are_operator_only",
                "operator_validation_command_template_commands_are_labeler_instructions",
                "operator_validation_command_template_labelers_must_not_run_commands",
                "operator_validation_command_template_operator_authorization_required",
                "operator_validation_command_template_command_count",
                "operator_validation_command_template_gate_ids",
                "operator_validation_command_template_command_ids",
                "operator_validation_command_template_template_backed_gate_ids",
                "operator_validation_command_template_validation_checklist_gate_ids",
                "operator_validation_command_template_apply_required_gate_ids",
                "operator_validation_command_template_evidence_template_fields_by_gate_id",
                "operator_validation_command_template_evidence_template_paths_by_gate_id",
                "operator_validation_command_template_missing_command_gate_ids",
                "operator_validation_command_template_launch_evidence_collection_plan_schema",
                "operator_validation_command_template_launch_evidence_collection_step_count",
                "operator_validation_command_template_launch_evidence_collection_gate_ids",
                "operator_validation_command_template_launch_evidence_collection_record_command_ids",
                "operator_validation_command_template_launch_evidence_collection_operator_only",
                "operator_validation_command_template_launch_evidence_collection_required_final_field",
                "operator_validation_command_template_launch_evidence_collection_required_final_value",
                "operator_validation_command_template_launch_evidence_collection_final_inspection_command",
                "operator_validation_command_template_operator_action",
                "operator_validation_public_fields",
                "operator_validation_identity_personal_queue_evidence_status_values",
                "operator_validation_gate_status_values",
                "operator_validation_gate_ids",
                "operator_validation_gate_flat_field_suffixes",
                "operator_validation_operator_only_fields",
                "operator_validation_labeler_visible_payloads_include_operator_only_fields",
                "operator_validation_per_user_payloads_use_public_fields_only",
                "operator_validation_top_level_operator_reports_may_include_operator_only_fields",
                "sendability_reasons",
                "sendability_actions",
                "labeler_landing_url",
                "expected_user_labeler_landing_url",
                "labeling_home_url",
                "expected_user_labeling_home_url",
                "dashboard_url",
                "expected_user_dashboard_url",
                "expected_user_dataset_queue_url",
                "expected_user_personal_work_url",
                "expected_user_personal_dataset_queue_url",
                "preferred_labeler_entrypoint",
                "preferred_labeler_entry_url",
                "personalized_labeler_entrypoint",
                "personalized_labeler_entry_url",
                "labeling_home_link_role",
                "personal_dataset_queue_link_role",
                "dataset_queue_link_role",
                "canonical_dataset_queue_link_role",
                "preferred_labeler_entry_url_matches_dataset_queue",
                "preferred_labeler_entry_url_matches_personal_dataset_queue",
                "personalized_labeler_entry_url_matches_personal_dataset_queue",
                "expected_user_identity_probe_url",
                "links_expire_at_utc",
                "browser_mutation_write_policy_present",
                "browser_mutation_write_ready",
                "browser_mutation_write_readiness",
                "browser_mutation_write_operator_action",
                "browser_mutation_authoritative_label_state",
                "browser_mutation_data_plane_write_target",
                "browser_mutation_mutable_label_data_plane",
                "browser_mutation_label_mutation_target_kind",
                "browser_mutation_browser_label_write_target",
                "browser_mutation_server_mutates_task_scoped_zarr_targets",
                "browser_mutation_training_zarr_mutations_are_server_owned",
                "browser_mutation_promotion_training_zarr_requires_task_scope",
                "browser_mutation_handoff_artifacts_are_metadata_only",
                "browser_mutation_csv_handoff_artifact_role",
                "browser_mutation_csv_handoff_artifacts_are_label_write_targets",
                "browser_mutation_handoff_csv_artifacts_are_label_write_targets",
                "browser_mutation_intermediate_csv_artifacts_are_label_write_targets",
                "browser_mutation_browser_writes_csv_or_handoff_files",
                "browser_mutation_browser_writes_handoff_csv",
                "browser_mutation_browser_writes_intermediate_csv",
                "browser_mutation_browser_receives_zarr_write_authority",
                "browser_mutation_browser_has_direct_zarr_write_authority",
                "browser_mutation_target_contract_met",
                "browser_mutation_target_mismatch_count",
                "browser_mutation_target_mismatch_users",
                "dataset_queue_direct_start_policy_present",
                "dataset_queue_direct_start_enabled",
                "dataset_queue_direct_start_method",
                "dataset_queue_direct_start_endpoint_route_template",
                "dataset_queue_direct_start_same_origin_only",
                "dataset_queue_direct_start_exact_route_required",
                "dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id",
                "dataset_queue_direct_start_expected_user_guard_required",
                "dataset_queue_direct_start_post_body_expected_user_required",
                "dataset_queue_direct_start_post_body_expected_user_field",
                "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract",
                "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract",
                "dataset_queue_direct_start_denied_start_support_includes_authorization_context",
                "dataset_queue_direct_start_denied_start_contract_reports_no_session_created",
                "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false",
                "dataset_queue_direct_start_startable_task_states",
                "dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint",
                "dataset_queue_direct_start_label_mutation_target_kind",
                "dataset_queue_direct_start_browser_label_write_target",
                "dataset_queue_direct_start_csv_handoff_artifact_role",
                "dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets",
                "dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets",
                "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets",
                "dataset_queue_direct_start_browser_writes_csv_or_handoff_files",
                "dataset_queue_direct_start_browser_writes_handoff_csv",
                "dataset_queue_direct_start_browser_writes_intermediate_csv",
                "dataset_queue_direct_start_browser_receives_zarr_write_authority",
                "dataset_queue_direct_start_browser_has_direct_zarr_write_authority",
                "direct_browser_start_contract_met",
                "direct_browser_start_mismatch_count",
                "direct_browser_start_mismatch_users",
                "runtime_operator_validation_gate_cli_policy_schema",
                "runtime_operator_validation_gate_cli_policy_validation_checklist_flag",
                "runtime_operator_validation_gate_cli_policy_preferred_require_flag",
                "runtime_operator_validation_gate_cli_policy_legacy_require_flag",
                "runtime_operator_validation_gate_cli_policy_legacy_require_flag_retained_for_compatibility",
                "runtime_operator_validation_gate_cli_policy_config_field",
                "runtime_operator_validation_gate_cli_policy_requires_validation_checklist",
                "runtime_operator_validation_gate_cli_policy_protects_browser_start_open",
                "runtime_operator_validation_gate_cli_policy_protects_browser_mutations",
                "runtime_operator_validation_gate_cli_policy_blocks_before_session_creation",
                "runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check",
                "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write",
                "runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation",
                "direct_browser_start_contract_summary_schema",
                "direct_browser_start_contract_summary_ready",
                "direct_browser_start_contract_summary_task_count",
                "direct_browser_start_contract_summary_ready_task_count",
                "direct_browser_start_contract_summary_not_ready_task_count",
                "direct_browser_start_contract_summary_not_ready_reason_counts",
                "direct_browser_start_contract_summary_operator_action_counts",
                "direct_browser_start_contract_summary_expected_user_guard_enforced_by_api",
                "direct_browser_start_contract_summary_server_rechecks_on_post",
                "direct_browser_start_contract_summary_label_mutation_target_kind",
                "direct_browser_start_contract_summary_browser_label_write_target",
                "direct_browser_start_contract_summary_csv_handoff_artifact_role",
                "direct_browser_start_contract_summary_csv_handoff_artifacts_are_label_write_targets",
                "direct_browser_start_contract_summary_handoff_csv_artifacts_are_label_write_targets",
                "direct_browser_start_contract_summary_intermediate_csv_artifacts_are_label_write_targets",
                "direct_browser_start_contract_summary_browser_writes_csv_or_handoff_files",
                "direct_browser_start_contract_summary_browser_writes_handoff_csv",
                "direct_browser_start_contract_summary_browser_writes_intermediate_csv",
                "direct_browser_start_contract_summary_browser_receives_zarr_write_authority",
                "direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority",
                "recordings",
                "tasks",
                "waiting_datasets",
                "dataset_open_tasks",
                "dataset_queue_state_code",
                "dataset_queue_state_title",
                "labeler_work_completion_schema",
                "labeler_work_completion_status",
                "labeler_work_completion_dataset_queue_state_code",
                "labeler_work_completion_completed",
                "labeler_work_completion_has_waiting_work",
                "labeler_work_completion_ready_for_more_labeling",
                "labeler_work_completion_blocks_labeler_start",
                "labeler_work_completion_operator_action_required",
                "labeler_work_completion_labeler_action",
                "labeler_work_completion_completion_state",
                "labeler_work_completion_completion_percent",
                "labeler_work_completion_total_task_count",
                "labeler_work_completion_complete_task_count",
                "labeler_work_completion_open_task_count",
                "labeler_work_completion_waiting_dataset_count",
                "labeler_work_completion_waiting_recording_count",
                "labeler_work_completion_complete_recording_count",
                "labeler_work_completion_blocked_recording_count",
                "dataset_queue_blocks_labeler_start",
                "dataset_queue_start_ready",
                "dataset_queue_start_status",
                "dataset_queue_start_operator_action",
                "dataset_queue_preview_url",
                "canonical_dataset_queue_preview_url",
                "signed_links",
                "recordings_without_open_tasks",
                "recordings_without_open_tasks_by_reason",
                "recordings_without_open_tasks_actions",
                "redacted_summary_fields",
                "store_check_issues",
                "store_check_warnings",
                "index_html",
                "message",
                "quickstart",
                "dataset_queue",
                "manifest",
            ],
        )
        writer.writeheader()
        for handoff in handoffs:
            if not isinstance(handoff, dict):
                continue
            counts = handoff.get("counts") if isinstance(handoff.get("counts"), dict) else {}
            files = handoff.get("files") if isinstance(handoff.get("files"), dict) else {}
            dataset_queue_state = handoff.get("dataset_queue_state") if isinstance(handoff.get("dataset_queue_state"), Mapping) else {}
            dataset_queue_blocks_start = handoff.get("dataset_queue_blocks_labeler_start")
            if dataset_queue_blocks_start is None:
                dataset_queue_blocks_start = dataset_queue_state.get("blocks_labeler_start")
            queue_start_fields = _handoff_dataset_queue_start_fields(handoff)
            known_user_fields = _handoff_known_user_status_fields(handoff)
            ownership_fields = _handoff_assignment_ownership_fields(handoff)
            entry_fields = _handoff_entry_artifact_fields(handoff)
            personalized_launch_readiness_source = handoff.get(
                "personalized_launch_readiness"
            )
            personalized_launch_readiness = (
                dict(personalized_launch_readiness_source)
                if isinstance(personalized_launch_readiness_source, Mapping)
                else _personalized_launch_readiness_summary(handoff)
            )
            personalized_launch_readiness.setdefault(
                "fields",
                _personalized_launch_readiness_field_names(),
            )
            personalized_launch_readiness.setdefault(
                "field_count",
                len(_personalized_launch_readiness_field_names()),
            )
            safety_handoff = dict(handoff)
            if "labeler_safety" not in safety_handoff and isinstance(index.get("labeler_safety"), Mapping):
                safety_handoff["labeler_safety"] = index["labeler_safety"]
            labeler_safety_fields = _handoff_labeler_safety_fields(safety_handoff)
            route_handoff = dict(handoff)
            if "labeler_route_authorization_policy" not in route_handoff and isinstance(
                index.get("labeler_route_authorization_policy"), Mapping
            ):
                route_handoff["labeler_route_authorization_policy"] = index["labeler_route_authorization_policy"]
            route_authorization_fields = _handoff_labeler_route_authorization_fields(route_handoff)
            signed_link_handoff = dict(handoff)
            if "signed_link_policy" not in signed_link_handoff and isinstance(
                index.get("signed_link_policy"), Mapping
            ):
                signed_link_handoff["signed_link_policy"] = index["signed_link_policy"]
            signed_link_fields = _handoff_signed_link_policy_fields(signed_link_handoff)
            session_handoff = dict(handoff)
            if "session_guard_policy" not in session_handoff and isinstance(
                index.get("session_guard_policy"), Mapping
            ):
                session_handoff["session_guard_policy"] = index["session_guard_policy"]
            session_guard_fields = _handoff_session_guard_fields(session_handoff)
            task_state_handoff = dict(handoff)
            if "task_state_policy" not in task_state_handoff and isinstance(
                index.get("task_state_policy"), Mapping
            ):
                task_state_handoff["task_state_policy"] = index["task_state_policy"]
            task_state_fields = _handoff_task_state_policy_fields(task_state_handoff)
            zarr_backup_handoff = dict(handoff)
            if "zarr_backup_policy" not in zarr_backup_handoff and isinstance(
                index.get("zarr_backup_policy"), Mapping
            ):
                zarr_backup_handoff["zarr_backup_policy"] = index["zarr_backup_policy"]
            if isinstance(index.get("files"), Mapping):
                merged_backup_files = dict(index["files"])
                if isinstance(zarr_backup_handoff.get("files"), Mapping):
                    merged_backup_files.update(zarr_backup_handoff["files"])
                zarr_backup_handoff["files"] = merged_backup_files
            zarr_backup_fields = _handoff_zarr_backup_fields(zarr_backup_handoff)
            mutation_audit_handoff = dict(handoff)
            if "mutation_audit_policy" not in mutation_audit_handoff and isinstance(
                index.get("mutation_audit_policy"), Mapping
            ):
                mutation_audit_handoff["mutation_audit_policy"] = index["mutation_audit_policy"]
            mutation_audit_fields = _handoff_mutation_audit_fields(mutation_audit_handoff)
            response_security_handoff = dict(handoff)
            if "browser_response_security_policy" not in response_security_handoff and isinstance(
                index.get("browser_response_security_policy"), Mapping
            ):
                response_security_handoff["browser_response_security_policy"] = index[
                    "browser_response_security_policy"
                ]
            response_security_fields = _handoff_browser_response_security_fields(response_security_handoff)
            mutation_handoff = dict(handoff)
            if "browser_mutation_write_policy" not in mutation_handoff:
                mutation_handoff["browser_mutation_write_policy"] = browser_mutation_write_policy
            mutation_write_fields = _handoff_browser_mutation_write_fields(mutation_handoff)
            browser_mutation_target_summary = _browser_mutation_target_contract_summary(
                [mutation_write_fields]
            )
            direct_start_policy_present = isinstance(
                handoff.get("dataset_queue_direct_start_policy"), Mapping
            ) or isinstance(index.get("dataset_queue_direct_start_policy"), Mapping)
            direct_start_policy_fields = _dataset_queue_direct_start_policy_fields(
                handoff.get("dataset_queue_direct_start_policy")
                if isinstance(handoff.get("dataset_queue_direct_start_policy"), Mapping)
                else index.get("dataset_queue_direct_start_policy")
                if isinstance(index.get("dataset_queue_direct_start_policy"), Mapping)
                else None
            )
            direct_browser_start_summary = _direct_browser_start_contract_summary(
                [
                    {
                        "user": str(handoff.get("user") or ""),
                        "dataset_queue_direct_start_policy_present": direct_start_policy_present,
                        **direct_start_policy_fields,
                    }
                ]
            )
            single_owner_policy_contract_met = (
                bool(handoff.get("single_owner_policy_contract_met"))
                if "single_owner_policy_contract_met" in handoff
                else bool(ownership_fields.get("assignment_ownership_contract_ready"))
                and int(
                    ownership_fields.get(
                        "assignment_ownership_contract_duplicate_active_owner_count"
                    )
                    or 0
                )
                == 0
            )
            runtime_gate_cli_policy_fields = (
                _runtime_operator_validation_gate_cli_policy_fields(
                    handoff.get("runtime_operator_validation_gate_cli_policy")
                    if isinstance(
                        handoff.get("runtime_operator_validation_gate_cli_policy"),
                        Mapping,
                    )
                    else index.get("runtime_operator_validation_gate_cli_policy")
                    if isinstance(
                        index.get("runtime_operator_validation_gate_cli_policy"),
                        Mapping,
                    )
                    else None
                )
            )
            direct_start_summary_fields = _direct_browser_start_contract_summary_fields(
                handoff.get("direct_browser_start_contract_summary")
                if isinstance(handoff.get("direct_browser_start_contract_summary"), Mapping)
                else None
            )
            operator_recovery_handoff = dict(handoff)
            if "operator_recovery_policy" not in operator_recovery_handoff:
                operator_recovery_handoff["operator_recovery_policy"] = operator_recovery_policy
            operator_recovery_fields = _handoff_operator_recovery_fields(operator_recovery_handoff)
            operator_validation_command_template_fields = (
                _operator_validation_command_template_fields(handoff)
            )
            reassignment_session_safety = handoff.get("reassignment_session_safety")
            if not isinstance(reassignment_session_safety, Mapping):
                store_consistency = (
                    handoff.get("store_consistency")
                    if isinstance(handoff.get("store_consistency"), Mapping)
                    else {}
                )
                reassignment_session_safety = (
                    store_consistency.get("reassignment_session_safety")
                    if isinstance(store_consistency.get("reassignment_session_safety"), Mapping)
                    else {}
                )
            if not reassignment_session_safety and (
                "reassignment_session_safety_ok" in handoff
                or "reassignment_session_safety_blocks_labeler_mutation" in handoff
                or "reassignment_session_safety_active_session_assignment_mismatch_count" in handoff
            ):
                reassignment_session_safety = {
                    "ok": bool(handoff.get("reassignment_session_safety_ok", True)),
                    "blocks_labeler_mutation": bool(
                        handoff.get("reassignment_session_safety_blocks_labeler_mutation")
                    ),
                    "active_session_assignment_mismatch_count": int(
                        handoff.get(
                            "reassignment_session_safety_active_session_assignment_mismatch_count"
                        )
                        or 0
                    ),
                    "active_session_assignment_mismatch_session_ids": (
                        handoff.get(
                            "reassignment_session_safety_active_session_assignment_mismatch_session_ids"
                        )
                        if isinstance(
                            handoff.get(
                                "reassignment_session_safety_active_session_assignment_mismatch_session_ids"
                            ),
                            list,
                        )
                        else []
                    ),
                    "active_session_assignment_mismatch_recording_ids": (
                        handoff.get(
                            "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
                        )
                        if isinstance(
                            handoff.get(
                                "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
                            ),
                            list,
                        )
                        else []
                    ),
                    "requires_operator_recovery": bool(
                        handoff.get("reassignment_session_safety_requires_operator_recovery")
                    ),
                    "operator_action": str(
                        handoff.get("reassignment_session_safety_operator_action") or ""
                    ),
                }
            reassignment_session_safety_fields = _public_reassignment_session_safety_fields(
                reassignment_session_safety
                if isinstance(reassignment_session_safety, Mapping)
                else {}
            )
            safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
                handoff,
                safe_share_gate=safe_share_gate,
            )
            writer.writerow(
                {
                    "user": handoff.get("user", ""),
                    "ok": bool(handoff.get("ok")),
                    "known_labeler": known_user_fields["known_labeler"],
                    "known_user_active_assignment_count": known_user_fields["known_user_active_assignment_count"],
                    "known_user_assignment_count": known_user_fields["known_user_assignment_count"],
                    "known_user_readiness": known_user_fields["known_user_readiness"],
                    "known_user_operator_action": known_user_fields["known_user_operator_action"],
                    "assignment_ownership_ok": ownership_fields["assignment_ownership_ok"],
                    "assignment_active_assignment_count": ownership_fields["assignment_active_assignment_count"],
                    "assignment_duplicate_active_owner_count": ownership_fields["assignment_duplicate_active_owner_count"],
                    "assignment_ownership_readiness": ownership_fields["assignment_ownership_readiness"],
                    "assignment_ownership_operator_action": ownership_fields["assignment_ownership_operator_action"],
                    "assignment_ownership_contract_schema": ownership_fields[
                        "assignment_ownership_contract_schema"
                    ],
                    "assignment_ownership_contract_ready": ownership_fields[
                        "assignment_ownership_contract_ready"
                    ],
                    "assignment_ownership_contract_assignment_scope": ownership_fields[
                        "assignment_ownership_contract_assignment_scope"
                    ],
                    "assignment_ownership_contract_recording_assignment_key": ownership_fields[
                        "assignment_ownership_contract_recording_assignment_key"
                    ],
                    "assignment_ownership_contract_recording_id_primary_key": ownership_fields[
                        "assignment_ownership_contract_recording_id_primary_key"
                    ],
                    "assignment_ownership_contract_schema_enforced_recording_primary_key": ownership_fields[
                        "assignment_ownership_contract_schema_enforced_recording_primary_key"
                    ],
                    "assignment_ownership_contract_store_recording_id_primary_key": ownership_fields[
                        "assignment_ownership_contract_store_recording_id_primary_key"
                    ],
                    "assignment_ownership_contract_store_schema_enforced_recording_primary_key": ownership_fields[
                        "assignment_ownership_contract_store_schema_enforced_recording_primary_key"
                    ],
                    "assignment_ownership_contract_schema_integrity_source": ownership_fields[
                        "assignment_ownership_contract_schema_integrity_source"
                    ],
                    "assignment_ownership_contract_primary_key_columns": ownership_fields[
                        "assignment_ownership_contract_primary_key_columns"
                    ],
                    "assignment_ownership_contract_one_current_assignment_row_per_recording": ownership_fields[
                        "assignment_ownership_contract_one_current_assignment_row_per_recording"
                    ],
                    "assignment_ownership_contract_one_active_owner": ownership_fields[
                        "assignment_ownership_contract_one_active_owner"
                    ],
                    "assignment_ownership_contract_multiple_labelers_per_recording_allowed": ownership_fields[
                        "assignment_ownership_contract_multiple_labelers_per_recording_allowed"
                    ],
                    "assignment_ownership_contract_reassignment_replaces_owner": ownership_fields[
                        "assignment_ownership_contract_reassignment_replaces_owner"
                    ],
                    "assignment_ownership_contract_stale_sessions_closed_on_reassignment": ownership_fields[
                        "assignment_ownership_contract_stale_sessions_closed_on_reassignment"
                    ],
                    "assignment_ownership_contract_stale_sessions_closed_before_assignment_update": ownership_fields[
                        "assignment_ownership_contract_stale_sessions_closed_before_assignment_update"
                    ],
                    "assignment_ownership_contract_reassignment_target_validated_before_session_closure": ownership_fields[
                        "assignment_ownership_contract_reassignment_target_validated_before_session_closure"
                    ],
                    "assignment_ownership_contract_session_closure_and_assignment_update_atomic": ownership_fields[
                        "assignment_ownership_contract_session_closure_and_assignment_update_atomic"
                    ],
                    "assignment_ownership_contract_raw_assignment_change_blocks_open_sessions": ownership_fields[
                        "assignment_ownership_contract_raw_assignment_change_blocks_open_sessions"
                    ],
                    "assignment_ownership_contract_assignment_manifests_are_control_plane": ownership_fields[
                        "assignment_ownership_contract_assignment_manifests_are_control_plane"
                    ],
                    "assignment_ownership_contract_duplicate_manifest_rows_do_not_create_multiple_owners": ownership_fields[
                        "assignment_ownership_contract_duplicate_manifest_rows_do_not_create_multiple_owners"
                    ],
                    "assignment_ownership_contract_assignment_user_match_required_for_mutation": ownership_fields[
                        "assignment_ownership_contract_assignment_user_match_required_for_mutation"
                    ],
                    "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner": ownership_fields[
                        "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner"
                    ],
                    "assignment_ownership_contract_browser_mutation_target_resolved_server_side": ownership_fields[
                        "assignment_ownership_contract_browser_mutation_target_resolved_server_side"
                    ],
                    "assignment_ownership_contract_browser_mutation_target_source": ownership_fields[
                        "assignment_ownership_contract_browser_mutation_target_source"
                    ],
                    "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs": ownership_fields[
                        "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs"
                    ],
                    "assignment_ownership_contract_labelers_mutate_intermediate_csvs": ownership_fields[
                        "assignment_ownership_contract_labelers_mutate_intermediate_csvs"
                    ],
                    "assignment_ownership_contract_store_single_owner_assignment_contract_present": ownership_fields[
                        "assignment_ownership_contract_store_single_owner_assignment_contract_present"
                    ],
                    "assignment_ownership_contract_store_single_owner_assignment_contract_ready": ownership_fields[
                        "assignment_ownership_contract_store_single_owner_assignment_contract_ready"
                    ],
                    "assignment_ownership_contract_store_single_owner_assignment_contract_met": ownership_fields[
                        "assignment_ownership_contract_store_single_owner_assignment_contract_met"
                    ],
                    "assignment_ownership_contract_store_single_owner_assignment_contract_schema": ownership_fields[
                        "assignment_ownership_contract_store_single_owner_assignment_contract_schema"
                    ],
                    "assignment_ownership_contract_assignment_ownership_integrity_ok": ownership_fields[
                        "assignment_ownership_contract_assignment_ownership_integrity_ok"
                    ],
                    "assignment_ownership_contract_active_assignment_count": ownership_fields[
                        "assignment_ownership_contract_active_assignment_count"
                    ],
                    "assignment_ownership_contract_unique_active_recording_count": ownership_fields[
                        "assignment_ownership_contract_unique_active_recording_count"
                    ],
                    "assignment_ownership_contract_duplicate_active_owner_count": ownership_fields[
                        "assignment_ownership_contract_duplicate_active_owner_count"
                    ],
                    "single_owner_policy_assignment_scope": ownership_fields[
                        "single_owner_policy_assignment_scope"
                    ],
                    "single_owner_policy_recording_assignment_key": ownership_fields[
                        "single_owner_policy_recording_assignment_key"
                    ],
                    "single_owner_policy_one_current_assignment_row_per_recording": ownership_fields[
                        "single_owner_policy_one_current_assignment_row_per_recording"
                    ],
                    "single_owner_policy_one_active_owner": ownership_fields[
                        "single_owner_policy_one_active_owner"
                    ],
                    "single_owner_policy_multiple_labelers_per_recording_allowed": ownership_fields[
                        "single_owner_policy_multiple_labelers_per_recording_allowed"
                    ],
                    "single_owner_policy_assignment_manifests_are_control_plane": ownership_fields[
                        "single_owner_policy_assignment_manifests_are_control_plane"
                    ],
                    "single_owner_policy_duplicate_manifest_rows_do_not_create_multiple_owners": ownership_fields[
                        "single_owner_policy_duplicate_manifest_rows_do_not_create_multiple_owners"
                    ],
                    "single_owner_policy_assignment_user_match_required_for_mutation": ownership_fields[
                        "single_owner_policy_assignment_user_match_required_for_mutation"
                    ],
                    "single_owner_policy_browser_mutation_requires_current_assignment_owner": ownership_fields[
                        "single_owner_policy_browser_mutation_requires_current_assignment_owner"
                    ],
                    "single_owner_policy_browser_mutation_target_resolved_server_side": ownership_fields[
                        "single_owner_policy_browser_mutation_target_resolved_server_side"
                    ],
                    "single_owner_policy_browser_mutation_target_source": ownership_fields[
                        "single_owner_policy_browser_mutation_target_source"
                    ],
                    "single_owner_policy_labelers_mutate_assigned_training_zarrs": ownership_fields[
                        "single_owner_policy_labelers_mutate_assigned_training_zarrs"
                    ],
                    "single_owner_policy_labelers_mutate_intermediate_csvs": ownership_fields[
                        "single_owner_policy_labelers_mutate_intermediate_csvs"
                    ],
                    "single_owner_policy_contract_met": single_owner_policy_contract_met,
                    "guarded_links_ready": entry_fields["guarded_links_ready"],
                    "missing_guarded_links": json.dumps(entry_fields["missing_guarded_links"], sort_keys=True),
                    "handoff_artifacts_ready": entry_fields["handoff_artifacts_ready"],
                    "missing_handoff_artifacts": json.dumps(entry_fields["missing_handoff_artifacts"], sort_keys=True),
                    "handoff_entry_readiness": entry_fields["handoff_entry_readiness"],
                    "handoff_entry_operator_action": entry_fields["handoff_entry_operator_action"],
                    "preferred_labeler_entrypoint": entry_fields["preferred_labeler_entrypoint"],
                    "preferred_labeler_entry_url": entry_fields["preferred_labeler_entry_url"],
                    "personalized_labeler_entrypoint": entry_fields[
                        "personalized_labeler_entrypoint"
                    ],
                    "personalized_labeler_entry_url": entry_fields[
                        "personalized_labeler_entry_url"
                    ],
                    "personalized_launch_readiness_schema": personalized_launch_readiness[
                        "schema"
                    ],
                    "personalized_launch_readiness_fields": json.dumps(
                        personalized_launch_readiness["fields"],
                        sort_keys=True,
                    ),
                    "personalized_launch_readiness_field_count": personalized_launch_readiness[
                        "field_count"
                    ],
                    "personalized_launch_readiness_personalized_labeler_entry_url": (
                        personalized_launch_readiness[
                            "personalized_labeler_entry_url"
                        ]
                    ),
                    "personalized_launch_readiness_labeler_start_ready": (
                        personalized_launch_readiness["labeler_start_ready"]
                    ),
                    "personalized_launch_readiness_labeler_start_status": (
                        personalized_launch_readiness["labeler_start_status"]
                    ),
                    "personalized_launch_readiness_labeler_work_completion_status": (
                        personalized_launch_readiness[
                            "labeler_work_completion_status"
                        ]
                    ),
                    "personalized_launch_readiness_safe_share_gate_id": (
                        personalized_launch_readiness["safe_share_gate_id"]
                    ),
                    "personalized_launch_readiness_safe_share_checklist_gate_evidence_complete": (
                        personalized_launch_readiness[
                            "safe_share_checklist_gate_evidence_complete"
                        ]
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_action_required": (
                        personalized_launch_readiness[
                            "external_launch_evidence_gap_action_required"
                        ]
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_count": (
                        personalized_launch_readiness[
                            "external_launch_evidence_gap_count"
                        ]
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_gate_ids": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_gate_ids"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_statuses": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_statuses"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_summary": (
                        personalized_launch_readiness[
                            "external_launch_evidence_gap_summary"
                        ]
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_todos": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_todos"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_todo_count": (
                        personalized_launch_readiness[
                            "external_launch_evidence_gap_todo_count"
                        ]
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_todo_fields": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_todo_fields"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_template_paths_by_gate_id": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_template_paths_by_gate_id"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_external_launch_evidence_gap_record_command_ids_by_gate_id": (
                        json.dumps(
                            personalized_launch_readiness[
                                "external_launch_evidence_gap_record_command_ids_by_gate_id"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "personalized_launch_readiness_browser_label_write_target": (
                        personalized_launch_readiness["browser_label_write_target"]
                    ),
                    "personalized_launch_readiness_browser_writes_csv_or_handoff_files": (
                        personalized_launch_readiness[
                            "browser_writes_csv_or_handoff_files"
                        ]
                    ),
                    "personalized_launch_readiness_browser_has_direct_zarr_write_authority": (
                        personalized_launch_readiness[
                            "browser_has_direct_zarr_write_authority"
                        ]
                    ),
                    "labeler_landing_link_role": entry_fields["labeler_landing_link_role"],
                    "personal_dataset_queue_link_role": entry_fields[
                        "personal_dataset_queue_link_role"
                    ],
                    "dataset_queue_link_role": entry_fields["dataset_queue_link_role"],
                    "canonical_dataset_queue_link_role": entry_fields[
                        "canonical_dataset_queue_link_role"
                    ],
                    "dashboard_link_role": entry_fields["dashboard_link_role"],
                    "identity_probe_link_role": entry_fields["identity_probe_link_role"],
                    "task_links_role": entry_fields["task_links_role"],
                    "preferred_labeler_entry_url_matches_dataset_queue": entry_fields[
                        "preferred_labeler_entry_url_matches_dataset_queue"
                    ],
                    "preferred_labeler_entry_url_matches_personal_dataset_queue": entry_fields[
                        "preferred_labeler_entry_url_matches_personal_dataset_queue"
                    ],
                    "personalized_labeler_entry_url_matches_personal_dataset_queue": entry_fields[
                        "personalized_labeler_entry_url_matches_personal_dataset_queue"
                    ],
                    "identity_check_url": entry_fields["identity_check_url"],
                    "labeler_safety_policy_present": labeler_safety_fields[
                        "labeler_safety_policy_present"
                    ],
                    "labeler_safety_ready": labeler_safety_fields["labeler_safety_ready"],
                    "labeler_safety_readiness": labeler_safety_fields["labeler_safety_readiness"],
                    "labeler_safety_operator_action": labeler_safety_fields[
                        "labeler_safety_operator_action"
                    ],
                    "labeler_safety_browser_only": labeler_safety_fields["labeler_safety_browser_only"],
                    "labeler_safety_labeler_runtime_surface": labeler_safety_fields[
                        "labeler_safety_labeler_runtime_surface"
                    ],
                    "labeler_safety_requires_local_palette_installation": labeler_safety_fields[
                        "labeler_safety_requires_local_palette_installation"
                    ],
                    "labeler_safety_requires_local_crimson_installation": labeler_safety_fields[
                        "labeler_safety_requires_local_crimson_installation"
                    ],
                    "labeler_safety_requires_local_conda_environment": labeler_safety_fields[
                        "labeler_safety_requires_local_conda_environment"
                    ],
                    "labeler_safety_requires_local_project_dependencies": labeler_safety_fields[
                        "labeler_safety_requires_local_project_dependencies"
                    ],
                    "labeler_safety_no_local_install_required": labeler_safety_fields[
                        "labeler_safety_no_local_install_required"
                    ],
                    "labeler_safety_identity_check_required": labeler_safety_fields[
                        "labeler_safety_identity_check_required"
                    ],
                    "labeler_safety_identity_probe_expected_user_guard_required": labeler_safety_fields[
                        "labeler_safety_identity_probe_expected_user_guard_required"
                    ],
                    "labeler_safety_identity_probe_diagnostic_only": labeler_safety_fields[
                        "labeler_safety_identity_probe_diagnostic_only"
                    ],
                    "labeler_safety_identity_probe_does_not_authorize_work": labeler_safety_fields[
                        "labeler_safety_identity_probe_does_not_authorize_work"
                    ],
                    "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces": labeler_safety_fields[
                        "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces"
                    ],
                    "labeler_safety_identity_probe_success_launch_ctas_rendered": labeler_safety_fields[
                        "labeler_safety_identity_probe_success_launch_ctas_rendered"
                    ],
                    "labeler_safety_identity_probe_failed_launch_ctas_suppressed": labeler_safety_fields[
                        "labeler_safety_identity_probe_failed_launch_ctas_suppressed"
                    ],
                    "labeler_safety_identity_probe_failed_support_urls_diagnostic_only": labeler_safety_fields[
                        "labeler_safety_identity_probe_failed_support_urls_diagnostic_only"
                    ],
                    "labeler_safety_no_direct_zarr_edits": labeler_safety_fields[
                        "labeler_safety_no_direct_zarr_edits"
                    ],
                    "labeler_safety_no_forwarding_links_or_handoffs": labeler_safety_fields[
                        "labeler_safety_no_forwarding_links_or_handoffs"
                    ],
                    "labeler_safety_expected_user_guards_ready": labeler_safety_fields[
                        "labeler_safety_expected_user_guards_ready"
                    ],
                    "labeler_safety_browser_payload_redaction_ready": labeler_safety_fields[
                        "labeler_safety_browser_payload_redaction_ready"
                    ],
                    "labeler_safety_browser_receives_task_scope": labeler_safety_fields[
                        "labeler_safety_browser_receives_task_scope"
                    ],
                    "labeler_safety_browser_receives_raw_zarr_paths": labeler_safety_fields[
                        "labeler_safety_browser_receives_raw_zarr_paths"
                    ],
                    "labeler_safety_missing_or_mismatched_expected_user_guards": json.dumps(
                        labeler_safety_fields["labeler_safety_missing_or_mismatched_expected_user_guards"],
                        sort_keys=True,
                    ),
                    "labeler_route_authorization_policy_present": route_authorization_fields[
                        "labeler_route_authorization_policy_present"
                    ],
                    "labeler_route_authorization_ready": route_authorization_fields[
                        "labeler_route_authorization_ready"
                    ],
                    "labeler_route_authorization_readiness": route_authorization_fields[
                        "labeler_route_authorization_readiness"
                    ],
                    "labeler_route_authorization_operator_action": route_authorization_fields[
                        "labeler_route_authorization_operator_action"
                    ],
                    "labeler_route_authorization_expected_user_match_required": route_authorization_fields[
                        "labeler_route_authorization_expected_user_match_required"
                    ],
                    "labeler_route_authorization_known_user_required": route_authorization_fields[
                        "labeler_route_authorization_known_user_required"
                    ],
                    "labeler_route_authorization_active_assignment_required": route_authorization_fields[
                        "labeler_route_authorization_active_assignment_required"
                    ],
                    "labeler_route_authorization_task_open_requires_active_assignment": route_authorization_fields[
                        "labeler_route_authorization_task_open_requires_active_assignment"
                    ],
                    "labeler_route_authorization_task_open_requires_startable_task_state": route_authorization_fields[
                        "labeler_route_authorization_task_open_requires_startable_task_state"
                    ],
                    "labeler_route_authorization_startable_task_states": route_authorization_fields[
                        "labeler_route_authorization_startable_task_states"
                    ],
                    "labeler_route_authorization_signed_links_are_entry_hints": route_authorization_fields[
                        "labeler_route_authorization_signed_links_are_entry_hints"
                    ],
                    "labeler_route_authorization_signed_links_recheck_runtime_start_gate": route_authorization_fields[
                        "labeler_route_authorization_signed_links_recheck_runtime_start_gate"
                    ],
                    "labeler_route_authorization_forwarded_links_rechecked": route_authorization_fields[
                        "labeler_route_authorization_forwarded_links_rechecked"
                    ],
                    "labeler_route_authorization_runtime_checklist_present": route_authorization_fields[
                        "labeler_route_authorization_runtime_checklist_present"
                    ],
                    "labeler_route_authorization_runtime_checklist_ready": route_authorization_fields[
                        "labeler_route_authorization_runtime_checklist_ready"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_ready": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_ready"
                    ],
                    "labeler_route_authorization_assignment_ownership_integrity_ok": route_authorization_fields[
                        "labeler_route_authorization_assignment_ownership_integrity_ok"
                    ],
                    "labeler_route_authorization_duplicate_active_owner_count": route_authorization_fields[
                        "labeler_route_authorization_duplicate_active_owner_count"
                    ],
                    "labeler_route_authorization_browser_mutation_target_resolved_server_side": route_authorization_fields[
                        "labeler_route_authorization_browser_mutation_target_resolved_server_side"
                    ],
                    "labeler_route_authorization_labelers_mutate_assigned_training_zarrs": route_authorization_fields[
                        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs"
                    ],
                    "labeler_route_authorization_labelers_mutate_intermediate_csvs": route_authorization_fields[
                        "labeler_route_authorization_labelers_mutate_intermediate_csvs"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_required_for_browser_work": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_required_for_browser_work"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target"
                    ],
                    "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation": route_authorization_fields[
                        "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation"
                    ],
                    "signed_link_policy_present": signed_link_fields["signed_link_policy_present"],
                    "signed_link_policy_ready": signed_link_fields["signed_link_policy_ready"],
                    "signed_link_policy_readiness": signed_link_fields["signed_link_policy_readiness"],
                    "signed_link_policy_operator_action": signed_link_fields[
                        "signed_link_policy_operator_action"
                    ],
                    "signed_link_task_specific": signed_link_fields["signed_link_task_specific"],
                    "signed_link_authorization_grant": signed_link_fields[
                        "signed_link_authorization_grant"
                    ],
                    "signed_link_binds_expected_user": signed_link_fields[
                        "signed_link_binds_expected_user"
                    ],
                    "signed_link_expected_user_required_on_open": signed_link_fields[
                        "signed_link_expected_user_required_on_open"
                    ],
                    "signed_link_forwarded_links_recheck_identity": signed_link_fields[
                        "signed_link_forwarded_links_recheck_identity"
                    ],
                    "signed_link_requires_server_side_task_authorization": signed_link_fields[
                        "signed_link_requires_server_side_task_authorization"
                    ],
                    "signed_link_runtime_operator_validation_start_gate_enforced": signed_link_fields[
                        "signed_link_runtime_operator_validation_start_gate_enforced"
                    ],
                    "signed_link_operator_validation_start_gate_checked_before_session_create": signed_link_fields[
                        "signed_link_operator_validation_start_gate_checked_before_session_create"
                    ],
                    "session_guard_policy_present": session_guard_fields["session_guard_policy_present"],
                    "session_guard_policy_ready": session_guard_fields["session_guard_policy_ready"],
                    "session_guard_policy_readiness": session_guard_fields[
                        "session_guard_policy_readiness"
                    ],
                    "session_guard_policy_operator_action": session_guard_fields[
                        "session_guard_policy_operator_action"
                    ],
                    "session_guard_requires_current_session": session_guard_fields[
                        "session_guard_requires_current_session"
                    ],
                    "session_guard_requires_unexpired_session": session_guard_fields[
                        "session_guard_requires_unexpired_session"
                    ],
                    "session_guard_stale_tab_save_rejected": session_guard_fields[
                        "session_guard_stale_tab_save_rejected"
                    ],
                    "session_guard_superseded_sessions_rejected": session_guard_fields[
                        "session_guard_superseded_sessions_rejected"
                    ],
                    "session_guard_non_startable_task_sessions_rejected": session_guard_fields[
                        "session_guard_non_startable_task_sessions_rejected"
                    ],
                    "session_guard_target_token_required_for_mutation": session_guard_fields[
                        "session_guard_target_token_required_for_mutation"
                    ],
                    "session_guard_labeler_promotion_retry_requires_current_session": session_guard_fields[
                        "session_guard_labeler_promotion_retry_requires_current_session"
                    ],
                    "session_guard_closure_event_support": session_guard_fields[
                        "session_guard_closure_event_support"
                    ],
                    "session_guard_rejects_after_reassignment": session_guard_fields[
                        "session_guard_rejects_after_reassignment"
                    ],
                    "session_guard_rejects_after_completion_or_reopen": session_guard_fields[
                        "session_guard_rejects_after_completion_or_reopen"
                    ],
                    "session_guard_rejects_after_expiration": session_guard_fields[
                        "session_guard_rejects_after_expiration"
                    ],
                    "session_guard_rejects_after_target_navigation": session_guard_fields[
                        "session_guard_rejects_after_target_navigation"
                    ],
                    "task_state_policy_present": task_state_fields["task_state_policy_present"],
                    "task_state_policy_ready": task_state_fields["task_state_policy_ready"],
                    "task_state_policy_readiness": task_state_fields[
                        "task_state_policy_readiness"
                    ],
                    "task_state_policy_operator_action": task_state_fields[
                        "task_state_policy_operator_action"
                    ],
                    "task_state_startable_task_states": task_state_fields[
                        "task_state_startable_task_states"
                    ],
                    "task_state_completed_tasks_read_only": task_state_fields[
                        "task_state_completed_tasks_read_only"
                    ],
                    "task_state_completed_tasks_open_new_sessions": task_state_fields[
                        "task_state_completed_tasks_open_new_sessions"
                    ],
                    "task_state_completed_task_open_requests": task_state_fields[
                        "task_state_completed_task_open_requests"
                    ],
                    "task_state_completed_task_save_requests": task_state_fields[
                        "task_state_completed_task_save_requests"
                    ],
                    "task_state_non_startable_task_open_requests": task_state_fields[
                        "task_state_non_startable_task_open_requests"
                    ],
                    "task_state_non_startable_task_save_requests": task_state_fields[
                        "task_state_non_startable_task_save_requests"
                    ],
                    "task_state_completion_closes_open_sessions": task_state_fields[
                        "task_state_completion_closes_open_sessions"
                    ],
                    "task_state_reopen_authority": task_state_fields["task_state_reopen_authority"],
                    "task_state_reopen_required_for_more_labeling": task_state_fields[
                        "task_state_reopen_required_for_more_labeling"
                    ],
                    "task_state_operator_reopen_required_before_more_labeling": task_state_fields[
                        "task_state_operator_reopen_required_before_more_labeling"
                    ],
                    "task_state_labeler_promotion_retry_requires_open_task": task_state_fields[
                        "task_state_labeler_promotion_retry_requires_open_task"
                    ],
                    "task_state_labeler_promotion_retry_mutation_enabled": task_state_fields[
                        "task_state_labeler_promotion_retry_mutation_enabled"
                    ],
                    "task_state_labeler_promotion_retry_rejection_error": task_state_fields[
                        "task_state_labeler_promotion_retry_rejection_error"
                    ],
                    "task_state_ordinary_labeler_promotion_retry_mutation": task_state_fields[
                        "task_state_ordinary_labeler_promotion_retry_mutation"
                    ],
                    "task_state_completion_requires_current_session": task_state_fields[
                        "task_state_completion_requires_current_session"
                    ],
                    "task_state_browser_mutation_target_selectors": task_state_fields[
                        "task_state_browser_mutation_target_selectors"
                    ],
                    "task_state_browser_mutation_target_token": task_state_fields[
                        "task_state_browser_mutation_target_token"
                    ],
                    "task_state_rejects_client_target_selectors": task_state_fields[
                        "task_state_rejects_client_target_selectors"
                    ],
                    "task_state_requires_current_target_token": task_state_fields[
                        "task_state_requires_current_target_token"
                    ],
                    "task_state_workflow_contracts_require_target_token": task_state_fields[
                        "task_state_workflow_contracts_require_target_token"
                    ],
                    "operator_recovery_policy_present": operator_recovery_fields[
                        "operator_recovery_policy_present"
                    ],
                    "operator_recovery_ready": operator_recovery_fields["operator_recovery_ready"],
                    "operator_recovery_readiness": operator_recovery_fields[
                        "operator_recovery_readiness"
                    ],
                    "operator_recovery_operator_action": operator_recovery_fields[
                        "operator_recovery_operator_action"
                    ],
                    "operator_recovery_contract_ready": operator_recovery_fields[
                        "operator_recovery_contract_ready"
                    ],
                    "operator_recovery_reassignment_closes_previous_owner_sessions": operator_recovery_fields[
                        "operator_recovery_reassignment_closes_previous_owner_sessions"
                    ],
                    "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update": operator_recovery_fields[
                        "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"
                    ],
                    "operator_recovery_reassignment_target_validated_before_session_closure": operator_recovery_fields[
                        "operator_recovery_reassignment_target_validated_before_session_closure"
                    ],
                    "operator_recovery_session_closure_and_assignment_update_atomic": operator_recovery_fields[
                        "operator_recovery_session_closure_and_assignment_update_atomic"
                    ],
                    "operator_recovery_task_reopen_operator_only": operator_recovery_fields[
                        "operator_recovery_task_reopen_operator_only"
                    ],
                    "operator_recovery_completion_closes_open_sessions": operator_recovery_fields[
                        "operator_recovery_completion_closes_open_sessions"
                    ],
                    "operator_recovery_failed_promotion_retry_operator_only": operator_recovery_fields[
                        "operator_recovery_failed_promotion_retry_operator_only"
                    ],
                    "operator_recovery_session_closure_events_operator_inspectable": operator_recovery_fields[
                        "operator_recovery_session_closure_events_operator_inspectable"
                    ],
                    "operator_recovery_operator_repair_closes_or_supersedes_sessions": operator_recovery_fields[
                        "operator_recovery_operator_repair_closes_or_supersedes_sessions"
                    ],
                    "operator_recovery_rollback_requires_backup_plan": operator_recovery_fields[
                        "operator_recovery_rollback_requires_backup_plan"
                    ],
                    "operator_recovery_bad_disposable_mutation_recovery_ready": operator_recovery_fields[
                        "operator_recovery_bad_disposable_mutation_recovery_ready"
                    ],
                    "operator_recovery_disposable_mutation_smoke_requires_recovery_path_verification": operator_recovery_fields[
                        "operator_recovery_disposable_mutation_smoke_requires_recovery_path_verification"
                    ],
                    "operator_recovery_restore_pauses_or_unassigns_recording_before_write": operator_recovery_fields[
                        "operator_recovery_restore_pauses_or_unassigns_recording_before_write"
                    ],
                    "operator_recovery_labelers_receive_recovery_write_authority": operator_recovery_fields[
                        "operator_recovery_labelers_receive_recovery_write_authority"
                    ],
                    "operator_recovery_browser_recovery_mutations_direct": operator_recovery_fields[
                        "operator_recovery_browser_recovery_mutations_direct"
                    ],
                    "operator_recovery_reassignment_session_repair_route": operator_recovery_fields[
                        "operator_recovery_reassignment_session_repair_route"
                    ],
                    "operator_recovery_task_state_route": operator_recovery_fields[
                        "operator_recovery_task_state_route"
                    ],
                    "operator_recovery_task_repair_route": operator_recovery_fields[
                        "operator_recovery_task_repair_route"
                    ],
                    "operator_recovery_audit_event_lookup_route": operator_recovery_fields[
                        "operator_recovery_audit_event_lookup_route"
                    ],
                    "operator_recovery_failed_promotion_retry_route": operator_recovery_fields[
                        "operator_recovery_failed_promotion_retry_route"
                    ],
                    "operator_recovery_validation_gate": operator_recovery_fields[
                        "operator_recovery_validation_gate"
                    ],
                    "zarr_backup_policy_present": zarr_backup_fields[
                        "zarr_backup_policy_present"
                    ],
                    "zarr_backup_ready": zarr_backup_fields["zarr_backup_ready"],
                    "zarr_backup_readiness": zarr_backup_fields["zarr_backup_readiness"],
                    "zarr_backup_operator_action": zarr_backup_fields[
                        "zarr_backup_operator_action"
                    ],
                    "zarr_backup_contract_ready": zarr_backup_fields[
                        "zarr_backup_contract_ready"
                    ],
                    "zarr_backup_read_only_plan": zarr_backup_fields["zarr_backup_read_only_plan"],
                    "zarr_backup_operator_only": zarr_backup_fields["zarr_backup_operator_only"],
                    "zarr_backup_copy_before_labeling": zarr_backup_fields[
                        "zarr_backup_copy_before_labeling"
                    ],
                    "zarr_backup_required_before_invite": zarr_backup_fields[
                        "zarr_backup_required_before_invite"
                    ],
                    "zarr_backup_labelers_do_not_edit_zarrs_directly": zarr_backup_fields[
                        "zarr_backup_labelers_do_not_edit_zarrs_directly"
                    ],
                    "zarr_backup_labelers_do_not_receive_backup_paths": zarr_backup_fields[
                        "zarr_backup_labelers_do_not_receive_backup_paths"
                    ],
                    "zarr_backup_pause_or_unassign_before_restore": zarr_backup_fields[
                        "zarr_backup_pause_or_unassign_before_restore"
                    ],
                    "zarr_backup_rollback_owner": zarr_backup_fields["zarr_backup_rollback_owner"],
                    "zarr_backup_validation_gate": zarr_backup_fields["zarr_backup_validation_gate"],
                    "zarr_backup_plan": zarr_backup_fields["zarr_backup_plan"],
                    "zarr_backup_plan_present": zarr_backup_fields["zarr_backup_plan_present"],
                    "zarr_backup_plan_required": zarr_backup_fields["zarr_backup_plan_required"],
                    "zarr_backup_required_targets": zarr_backup_fields[
                        "zarr_backup_required_targets"
                    ],
                    "zarr_backup_missing_path_tasks": zarr_backup_fields[
                        "zarr_backup_missing_path_tasks"
                    ],
                    "zarr_backup_required_targets_by_role": json.dumps(
                        zarr_backup_fields["zarr_backup_required_targets_by_role"],
                        sort_keys=True,
                    ),
                    "mutation_audit_policy_present": mutation_audit_fields[
                        "mutation_audit_policy_present"
                    ],
                    "mutation_audit_ready": mutation_audit_fields["mutation_audit_ready"],
                    "mutation_audit_readiness": mutation_audit_fields[
                        "mutation_audit_readiness"
                    ],
                    "mutation_audit_operator_action": mutation_audit_fields[
                        "mutation_audit_operator_action"
                    ],
                    "mutation_audit_contract_ready": mutation_audit_fields[
                        "mutation_audit_contract_ready"
                    ],
                    "mutation_audit_event_store": mutation_audit_fields[
                        "mutation_audit_event_store"
                    ],
                    "mutation_audit_append_only": mutation_audit_fields[
                        "mutation_audit_append_only"
                    ],
                    "mutation_audit_server_records_events": mutation_audit_fields[
                        "mutation_audit_server_records_events"
                    ],
                    "mutation_audit_browser_records_events_directly": mutation_audit_fields[
                        "mutation_audit_browser_records_events_directly"
                    ],
                    "mutation_audit_browser_receives_write_credentials": mutation_audit_fields[
                        "mutation_audit_browser_receives_write_credentials"
                    ],
                    "mutation_audit_per_workflow_contracts_include_provenance": mutation_audit_fields[
                        "mutation_audit_per_workflow_contracts_include_provenance"
                    ],
                    "mutation_audit_required_event_fields_present": mutation_audit_fields[
                        "mutation_audit_required_event_fields_present"
                    ],
                    "mutation_audit_required_event_fields": json.dumps(
                        mutation_audit_fields["mutation_audit_required_event_fields"],
                        sort_keys=True,
                    ),
                    "mutation_audit_timestamp_field": mutation_audit_fields[
                        "mutation_audit_timestamp_field"
                    ],
                    "mutation_audit_same_payload_retry_safe": mutation_audit_fields[
                        "mutation_audit_same_payload_retry_safe"
                    ],
                    "mutation_audit_duplicate_events_possible": mutation_audit_fields[
                        "mutation_audit_duplicate_events_possible"
                    ],
                    "mutation_audit_validation_gate": mutation_audit_fields[
                        "mutation_audit_validation_gate"
                    ],
                    "browser_response_security_policy_present": response_security_fields[
                        "browser_response_security_policy_present"
                    ],
                    "browser_response_security_ready": response_security_fields[
                        "browser_response_security_ready"
                    ],
                    "browser_response_security_readiness": response_security_fields[
                        "browser_response_security_readiness"
                    ],
                    "browser_response_security_operator_action": response_security_fields[
                        "browser_response_security_operator_action"
                    ],
                    "browser_response_security_contract_ready": response_security_fields[
                        "browser_response_security_contract_ready"
                    ],
                    "browser_response_security_no_store_cache": response_security_fields[
                        "browser_response_security_no_store_cache"
                    ],
                    "browser_response_security_pragma_no_cache": response_security_fields[
                        "browser_response_security_pragma_no_cache"
                    ],
                    "browser_response_security_expires_zero": response_security_fields[
                        "browser_response_security_expires_zero"
                    ],
                    "browser_response_security_clickjacking_protection": response_security_fields[
                        "browser_response_security_clickjacking_protection"
                    ],
                    "browser_response_security_mime_sniffing_protection": response_security_fields[
                        "browser_response_security_mime_sniffing_protection"
                    ],
                    "browser_response_security_referrer_leakage_protection": response_security_fields[
                        "browser_response_security_referrer_leakage_protection"
                    ],
                    "browser_response_security_csp_scope_ready": response_security_fields[
                        "browser_response_security_csp_scope_ready"
                    ],
                    "browser_response_security_permissions_policy_ready": response_security_fields[
                        "browser_response_security_permissions_policy_ready"
                    ],
                    "browser_response_security_proxy_must_preserve_headers": response_security_fields[
                        "browser_response_security_proxy_must_preserve_headers"
                    ],
                    "browser_response_security_cache_control": response_security_fields[
                        "browser_response_security_cache_control"
                    ],
                    "browser_response_security_x_frame_options": response_security_fields[
                        "browser_response_security_x_frame_options"
                    ],
                    "browser_response_security_x_content_type_options": response_security_fields[
                        "browser_response_security_x_content_type_options"
                    ],
                    "browser_response_security_referrer_policy": response_security_fields[
                        "browser_response_security_referrer_policy"
                    ],
                    "browser_response_security_content_security_policy": response_security_fields[
                        "browser_response_security_content_security_policy"
                    ],
                    "browser_response_security_permissions_policy": response_security_fields[
                        "browser_response_security_permissions_policy"
                    ],
                    "ready_to_send": bool(handoff.get("ready_to_send"))
                    if "ready_to_send" in handoff
                    else _handoff_ready_to_send(handoff),
                    "safe_share_gate_schema": str(
                        safe_share_fields["safe_share_gate_schema"]
                    ),
                    "safe_share_gate_id": str(safe_share_fields["safe_share_gate_id"]),
                    "safe_share_requires_require_shareable_inspection": bool(
                        safe_share_fields[
                            "safe_share_requires_require_shareable_inspection"
                        ]
                    ),
                    "safe_share_ready_to_send_is_sufficient": bool(
                        safe_share_fields["safe_share_ready_to_send_is_sufficient"]
                    ),
                    "safe_share_required_inspection_field": str(
                        safe_share_fields["safe_share_required_inspection_field"]
                    ),
                    "safe_share_required_inspection_value": bool(
                        safe_share_fields["safe_share_required_inspection_value"]
                    ),
                    "safe_share_launch_blocking_evidence_gate_ids": json.dumps(
                        safe_share_fields[
                            "safe_share_launch_blocking_evidence_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_gate_statuses": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_gate_statuses"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_missing_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_missing_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_pending_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_pending_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_needs_review_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_needs_review_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_missing_evidence_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_missing_evidence_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_unknown_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_unknown_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_satisfied_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_satisfied_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_unsatisfied_gate_ids": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_unsatisfied_gate_ids"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_next_actions": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_next_actions"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_next_action_detail_fields": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_next_action_detail_fields"
                        ],
                        sort_keys=True,
                    ),
                    "safe_share_launch_blocking_next_action_command_fields": json.dumps(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_next_action_command_fields"
                        ],
                        sort_keys=True,
                    ),
                    **{
                        field_name: (
                            json.dumps(
                                safe_share_checklist_fields[field_name],
                                sort_keys=True,
                            )
                            if isinstance(
                                safe_share_checklist_fields[field_name],
                                (dict, list),
                            )
                            else str(safe_share_checklist_fields[field_name])
                        )
                        for field_name in _safe_share_external_launch_evidence_gap_field_names()
                    },
                    "safe_share_launch_blocking_next_action_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_next_action_count"
                        ]
                    ),
                    "safe_share_next_action_summary": str(
                        safe_share_checklist_fields["safe_share_next_action_summary"]
                    ),
                    "safe_share_launch_blocking_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_missing_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_missing_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_pending_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_pending_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_needs_review_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_needs_review_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_missing_evidence_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_missing_evidence_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_unknown_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_unknown_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_satisfied_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_satisfied_gate_count"
                        ]
                    ),
                    "safe_share_launch_blocking_unsatisfied_gate_count": int(
                        safe_share_checklist_fields[
                            "safe_share_launch_blocking_unsatisfied_gate_count"
                        ]
                    ),
                    "safe_share_checklist_gate_evidence_complete": bool(
                        safe_share_checklist_fields[
                            "safe_share_checklist_gate_evidence_complete"
                        ]
                    ),
                    "safe_share_checklist_operator_action": str(
                        safe_share_checklist_fields[
                            "safe_share_checklist_operator_action"
                        ]
                    ),
                    "safe_share_operator_action": str(
                        safe_share_fields["safe_share_operator_action"]
                    ),
                    "reassignment_session_safety": json.dumps(
                        reassignment_session_safety_fields,
                        sort_keys=True,
                    ),
                    "reassignment_session_safety_ok": bool(
                        reassignment_session_safety_fields.get("ok", True)
                    ),
                    "reassignment_session_safety_blocks_labeler_mutation": bool(
                        reassignment_session_safety_fields.get("blocks_labeler_mutation")
                    ),
                    "reassignment_session_safety_active_session_assignment_mismatch_count": int(
                        reassignment_session_safety_fields.get(
                            "active_session_assignment_mismatch_count"
                        )
                        or 0
                    ),
                    "reassignment_session_safety_active_session_assignment_mismatch_session_ids": json.dumps(
                        reassignment_session_safety_fields.get(
                            "active_session_assignment_mismatch_session_ids",
                            [],
                        ),
                        sort_keys=True,
                    ),
                    "reassignment_session_safety_active_session_assignment_mismatch_recording_ids": json.dumps(
                        reassignment_session_safety_fields.get(
                            "active_session_assignment_mismatch_recording_ids",
                            [],
                        ),
                        sort_keys=True,
                    ),
                    "reassignment_session_safety_requires_operator_recovery": bool(
                        reassignment_session_safety_fields.get("requires_operator_recovery")
                    ),
                    "reassignment_session_safety_operator_action": str(
                        reassignment_session_safety_fields.get("operator_action") or ""
                    ),
                    "sendability_reasons": ", ".join(
                        str(reason)
                        for reason in (
                            handoff.get("sendability_reasons")
                            if isinstance(handoff.get("sendability_reasons"), list)
                            else []
                        )
                        if str(reason).strip()
                    ),
                    "sendability_actions": " ".join(
                        str(action)
                        for action in (
                            handoff.get("sendability_actions")
                            if isinstance(handoff.get("sendability_actions"), list)
                            else []
                        )
                        if str(action).strip()
                    ),
                    "labeler_landing_url": handoff.get("labeler_landing_url", ""),
                    "expected_user_labeler_landing_url": handoff.get("expected_user_labeler_landing_url", ""),
                    "labeling_home_url": handoff.get("labeling_home_url", ""),
                    "expected_user_labeling_home_url": handoff.get("expected_user_labeling_home_url", ""),
                    "labeling_home_link_role": handoff.get(
                        "labeling_home_link_role",
                        entry_fields["labeling_home_link_role"],
                    ),
                    "dashboard_url": handoff.get("dashboard_url", ""),
                    "expected_user_dashboard_url": handoff.get("expected_user_dashboard_url", ""),
                    "expected_user_dataset_queue_url": handoff.get("expected_user_dataset_queue_url", ""),
                    "expected_user_personal_work_url": handoff.get("expected_user_personal_work_url", ""),
                    "expected_user_personal_dataset_queue_url": handoff.get(
                        "expected_user_personal_dataset_queue_url",
                        "",
                    ),
                    "preferred_labeler_entrypoint": handoff.get(
                        "preferred_labeler_entrypoint",
                        entry_fields["preferred_labeler_entrypoint"],
                    ),
                    "preferred_labeler_entry_url": handoff.get(
                        "preferred_labeler_entry_url",
                        entry_fields["preferred_labeler_entry_url"],
                    ),
                    "personalized_labeler_entrypoint": handoff.get(
                        "personalized_labeler_entrypoint",
                        "",
                    ),
                    "personalized_labeler_entry_url": handoff.get(
                        "personalized_labeler_entry_url",
                        "",
                    ),
                    **_queue_first_entry_contract_flat_fields(handoff),
                    "personal_dataset_queue_link_role": handoff.get(
                        "personal_dataset_queue_link_role",
                        entry_fields["personal_dataset_queue_link_role"],
                    ),
                    "dataset_queue_link_role": handoff.get(
                        "dataset_queue_link_role",
                        entry_fields["dataset_queue_link_role"],
                    ),
                    "canonical_dataset_queue_link_role": handoff.get(
                        "canonical_dataset_queue_link_role",
                        entry_fields["canonical_dataset_queue_link_role"],
                    ),
                    "preferred_labeler_entry_url_matches_dataset_queue": handoff.get(
                        "preferred_labeler_entry_url_matches_dataset_queue",
                        entry_fields["preferred_labeler_entry_url_matches_dataset_queue"],
                    ),
                    "preferred_labeler_entry_url_matches_personal_dataset_queue": handoff.get(
                        "preferred_labeler_entry_url_matches_personal_dataset_queue",
                        entry_fields[
                            "preferred_labeler_entry_url_matches_personal_dataset_queue"
                        ],
                    ),
                    "personalized_labeler_entry_url_matches_personal_dataset_queue": handoff.get(
                        "personalized_labeler_entry_url_matches_personal_dataset_queue",
                        entry_fields[
                            "personalized_labeler_entry_url_matches_personal_dataset_queue"
                        ],
                    ),
                    "expected_user_identity_probe_url": handoff.get("expected_user_identity_probe_url", ""),
                    "links_expire_at_utc": handoff.get("links_expire_at_utc", ""),
                    "browser_mutation_write_policy_present": mutation_write_fields[
                        "browser_mutation_write_policy_present"
                    ],
                    "browser_mutation_write_ready": mutation_write_fields["browser_mutation_write_ready"],
                    "browser_mutation_write_readiness": mutation_write_fields[
                        "browser_mutation_write_readiness"
                    ],
                    "browser_mutation_write_operator_action": mutation_write_fields[
                        "browser_mutation_write_operator_action"
                    ],
                    "browser_mutation_authoritative_label_state": mutation_write_fields[
                        "browser_mutation_authoritative_label_state"
                    ],
                    "browser_mutation_data_plane_write_target": mutation_write_fields[
                        "browser_mutation_data_plane_write_target"
                    ],
                    "browser_mutation_mutable_label_data_plane": mutation_write_fields[
                        "browser_mutation_mutable_label_data_plane"
                    ],
                    "browser_mutation_label_mutation_target_kind": mutation_write_fields[
                        "browser_mutation_label_mutation_target_kind"
                    ],
                    "browser_mutation_browser_label_write_target": mutation_write_fields[
                        "browser_mutation_browser_label_write_target"
                    ],
                    "browser_mutation_server_mutates_task_scoped_zarr_targets": mutation_write_fields[
                        "browser_mutation_server_mutates_task_scoped_zarr_targets"
                    ],
                    "browser_mutation_training_zarr_mutations_are_server_owned": mutation_write_fields[
                        "browser_mutation_training_zarr_mutations_are_server_owned"
                    ],
                    "browser_mutation_promotion_training_zarr_requires_task_scope": mutation_write_fields[
                        "browser_mutation_promotion_training_zarr_requires_task_scope"
                    ],
                    "browser_mutation_handoff_artifacts_are_metadata_only": mutation_write_fields[
                        "browser_mutation_handoff_artifacts_are_metadata_only"
                    ],
                    "browser_mutation_csv_handoff_artifact_role": mutation_write_fields[
                        "browser_mutation_csv_handoff_artifact_role"
                    ],
                    "browser_mutation_csv_handoff_artifacts_are_label_write_targets": (
                        mutation_write_fields[
                            "browser_mutation_csv_handoff_artifacts_are_label_write_targets"
                        ]
                    ),
                    "browser_mutation_handoff_csv_artifacts_are_label_write_targets": (
                        mutation_write_fields[
                            "browser_mutation_handoff_csv_artifacts_are_label_write_targets"
                        ]
                    ),
                    "browser_mutation_intermediate_csv_artifacts_are_label_write_targets": (
                        mutation_write_fields[
                            "browser_mutation_intermediate_csv_artifacts_are_label_write_targets"
                        ]
                    ),
                    "browser_mutation_browser_writes_csv_or_handoff_files": mutation_write_fields[
                        "browser_mutation_browser_writes_csv_or_handoff_files"
                    ],
                    "browser_mutation_browser_writes_handoff_csv": mutation_write_fields[
                        "browser_mutation_browser_writes_handoff_csv"
                    ],
                    "browser_mutation_browser_writes_intermediate_csv": mutation_write_fields[
                        "browser_mutation_browser_writes_intermediate_csv"
                    ],
                    "browser_mutation_browser_receives_zarr_write_authority": mutation_write_fields[
                        "browser_mutation_browser_receives_zarr_write_authority"
                    ],
                    "browser_mutation_browser_has_direct_zarr_write_authority": mutation_write_fields[
                        "browser_mutation_browser_has_direct_zarr_write_authority"
                    ],
                    "browser_mutation_target_contract_met": bool(
                        browser_mutation_target_summary.get("met")
                    ),
                    "browser_mutation_target_mismatch_count": int(
                        browser_mutation_target_summary.get("mismatch_count") or 0
                    ),
                    "browser_mutation_target_mismatch_users": json.dumps(
                        browser_mutation_target_summary.get("mismatch_users", []),
                        sort_keys=True,
                    ),
                    "dataset_queue_direct_start_policy_present": direct_start_policy_present,
                    **direct_start_policy_fields,
                    "direct_browser_start_contract_met": bool(
                        direct_browser_start_summary.get("met")
                    ),
                    "direct_browser_start_mismatch_count": int(
                        direct_browser_start_summary.get("mismatch_count") or 0
                    ),
                    "direct_browser_start_mismatch_users": json.dumps(
                        direct_browser_start_summary.get("mismatch_users", []),
                        sort_keys=True,
                    ),
                    **runtime_gate_cli_policy_fields,
                    **direct_start_summary_fields,
                    "operator_validation_required_before_invite": bool(
                        handoff.get("operator_validation_required_before_invite")
                    ),
                    "operator_validation_all_complete": bool(
                        handoff.get("operator_validation_all_complete")
                    ),
                    "operator_validation_declared_all_complete": bool(
                        handoff.get("operator_validation_declared_all_complete")
                    ),
                    "operator_validation_ready_for_operator_validation": bool(
                        handoff.get("operator_validation_ready_for_operator_validation")
                    ),
                    "operator_validation_gate_count": int(
                        handoff.get("operator_validation_gate_count") or 0
                    ),
                    "operator_validation_status": str(handoff.get("operator_validation_status") or ""),
                    "operator_validation_source": str(handoff.get("operator_validation_source") or ""),
                    "operator_validation_pending_gate_ids": json.dumps(
                        handoff.get("operator_validation_pending_gate_ids", []),
                        sort_keys=True,
                    ),
                    "operator_validation_needs_review_gate_ids": json.dumps(
                        handoff.get("operator_validation_needs_review_gate_ids", []),
                        sort_keys=True,
                    ),
                    "operator_validation_required_missing_evidence_gate_ids": json.dumps(
                        handoff.get("operator_validation_required_missing_evidence_gate_ids", []),
                        sort_keys=True,
                    ),
                    "operator_validation_required_pending_gate_count": int(
                        handoff.get("operator_validation_required_pending_gate_count") or 0
                    ),
                    "operator_validation_needs_review_gate_count": int(
                        handoff.get("operator_validation_needs_review_gate_count") or 0
                    ),
                    "operator_validation_required_missing_evidence_gate_count": int(
                        handoff.get("operator_validation_required_missing_evidence_gate_count") or 0
                    ),
                    "operator_validation_operator_action": str(
                        handoff.get("operator_validation_operator_action") or ""
                    ),
                    "operator_validation_external_evidence_required": bool(
                        handoff.get("operator_validation_external_evidence_required")
                    ),
                    "operator_validation_external_evidence_required_gate_ids": json.dumps(
                        handoff.get("operator_validation_external_evidence_required_gate_ids")
                        if isinstance(
                            handoff.get(
                                "operator_validation_external_evidence_required_gate_ids"
                            ),
                            list,
                        )
                        else [],
                        sort_keys=True,
                    ),
                    "operator_validation_external_evidence_required_gate_count": int(
                        handoff.get(
                            "operator_validation_external_evidence_required_gate_count"
                        )
                        or 0
                    ),
                    "operator_validation_external_evidence_template_fields_by_gate_id": json.dumps(
                        handoff.get(
                            "operator_validation_external_evidence_template_fields_by_gate_id"
                        )
                        if isinstance(
                            handoff.get(
                                "operator_validation_external_evidence_template_fields_by_gate_id"
                            ),
                            Mapping,
                        )
                        else {},
                        sort_keys=True,
                    ),
                    "operator_validation_external_evidence_template_paths_by_gate_id": json.dumps(
                        handoff.get(
                            "operator_validation_external_evidence_template_paths_by_gate_id"
                        )
                        if isinstance(
                            handoff.get(
                                "operator_validation_external_evidence_template_paths_by_gate_id"
                            ),
                            Mapping,
                        )
                        else {},
                        sort_keys=True,
                    ),
                    "operator_validation_checklist_only_required_gate_ids": json.dumps(
                        handoff.get("operator_validation_checklist_only_required_gate_ids")
                        if isinstance(
                            handoff.get(
                                "operator_validation_checklist_only_required_gate_ids"
                            ),
                            list,
                        )
                        else [],
                        sort_keys=True,
                    ),
                    "operator_validation_checklist_only_required_gate_count": int(
                        handoff.get("operator_validation_checklist_only_required_gate_count")
                        or 0
                    ),
                    **_operator_validation_gate_flat_fields(handoff),
                    "identity_personal_queue_evidence_status": str(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_status"
                        ]
                    ),
                    "identity_personal_queue_evidence_ready_count": int(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_ready_count"
                        ]
                    ),
                    "identity_personal_queue_evidence_missing_count": int(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_missing_count"
                        ]
                    ),
                    "identity_personal_queue_evidence_ready_users": json.dumps(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_ready_users"
                        ],
                        sort_keys=True,
                    ),
                    "identity_personal_queue_evidence_missing_users": json.dumps(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_missing_users"
                        ],
                        sort_keys=True,
                    ),
                    "identity_personal_queue_evidence_missing_fields_by_user": json.dumps(
                        _operator_validation_public_fields(handoff)[
                            "identity_personal_queue_evidence_missing_fields_by_user"
                        ],
                        sort_keys=True,
                    ),
                    "identity_all_users_have_personal_queue_evidence": bool(
                        _operator_validation_public_fields(handoff)[
                            "identity_all_users_have_personal_queue_evidence"
                        ]
                    ),
                    "operator_validation_command_template_schema": operator_validation_command_template_fields[
                        "operator_validation_command_template_schema"
                    ],
                    "operator_validation_command_template_commands_are_operator_only": operator_validation_command_template_fields[
                        "operator_validation_command_template_commands_are_operator_only"
                    ],
                    "operator_validation_command_template_commands_are_labeler_instructions": operator_validation_command_template_fields[
                        "operator_validation_command_template_commands_are_labeler_instructions"
                    ],
                    "operator_validation_command_template_labelers_must_not_run_commands": operator_validation_command_template_fields[
                        "operator_validation_command_template_labelers_must_not_run_commands"
                    ],
                    "operator_validation_command_template_operator_authorization_required": operator_validation_command_template_fields[
                        "operator_validation_command_template_operator_authorization_required"
                    ],
                    "operator_validation_command_template_command_count": operator_validation_command_template_fields[
                        "operator_validation_command_template_command_count"
                    ],
                    "operator_validation_command_template_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_gate_ids"
                    ],
                    "operator_validation_command_template_command_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_command_ids"
                    ],
                    "operator_validation_command_template_template_backed_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_template_backed_gate_ids"
                    ],
                    "operator_validation_command_template_validation_checklist_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_validation_checklist_gate_ids"
                    ],
                    "operator_validation_command_template_apply_required_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_apply_required_gate_ids"
                    ],
                    "operator_validation_command_template_evidence_template_fields_by_gate_id": operator_validation_command_template_fields[
                        "operator_validation_command_template_evidence_template_fields_by_gate_id"
                    ],
                    "operator_validation_command_template_evidence_template_paths_by_gate_id": operator_validation_command_template_fields[
                        "operator_validation_command_template_evidence_template_paths_by_gate_id"
                    ],
                    "operator_validation_command_template_missing_command_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_missing_command_gate_ids"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_plan_schema": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_plan_schema"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_step_count": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_step_count"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_gate_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_gate_ids"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_record_command_ids": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_record_command_ids"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_operator_only": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_operator_only"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_required_final_field": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_required_final_field"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_required_final_value": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_required_final_value"
                    ],
                    "operator_validation_command_template_launch_evidence_collection_final_inspection_command": operator_validation_command_template_fields[
                        "operator_validation_command_template_launch_evidence_collection_final_inspection_command"
                    ],
                    "operator_validation_command_template_operator_action": operator_validation_command_template_fields[
                        "operator_validation_command_template_operator_action"
                    ],
                    "operator_validation_public_fields": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_public_fields"],
                        sort_keys=True,
                    ),
                    "operator_validation_identity_personal_queue_evidence_status_values": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_identity_personal_queue_evidence_status_values"],
                        sort_keys=True,
                    ),
                    "operator_validation_gate_status_values": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_gate_status_values"],
                        sort_keys=True,
                    ),
                    "operator_validation_gate_ids": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_gate_ids"],
                        sort_keys=True,
                    ),
                    "operator_validation_gate_flat_field_suffixes": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_gate_flat_field_suffixes"],
                        sort_keys=True,
                    ),
                    "operator_validation_operator_only_fields": json.dumps(
                        _operator_validation_visibility_fields(
                            handoff.get("operator_validation_visibility_policy")
                            if isinstance(
                                handoff.get("operator_validation_visibility_policy"),
                                Mapping,
                            )
                            else None
                        )["operator_validation_operator_only_fields"],
                        sort_keys=True,
                    ),
                    "operator_validation_labeler_visible_payloads_include_operator_only_fields": _operator_validation_visibility_fields(
                        handoff.get("operator_validation_visibility_policy")
                        if isinstance(
                            handoff.get("operator_validation_visibility_policy"),
                            Mapping,
                        )
                        else None
                    )[
                        "operator_validation_labeler_visible_payloads_include_operator_only_fields"
                    ],
                    "operator_validation_per_user_payloads_use_public_fields_only": _operator_validation_visibility_fields(
                        handoff.get("operator_validation_visibility_policy")
                        if isinstance(
                            handoff.get("operator_validation_visibility_policy"),
                            Mapping,
                        )
                        else None
                    )[
                        "operator_validation_per_user_payloads_use_public_fields_only"
                    ],
                    "operator_validation_top_level_operator_reports_may_include_operator_only_fields": _operator_validation_visibility_fields(
                        handoff.get("operator_validation_visibility_policy")
                        if isinstance(
                            handoff.get("operator_validation_visibility_policy"),
                            Mapping,
                        )
                        else None
                    )[
                        "operator_validation_top_level_operator_reports_may_include_operator_only_fields"
                    ],
                    "recordings": counts.get("recordings", 0),
                    "tasks": counts.get("tasks", 0),
                    "waiting_datasets": counts.get("waiting_datasets", 0),
                    "dataset_open_tasks": counts.get("dataset_open_tasks", 0),
                    "dataset_queue_state_code": handoff.get("dataset_queue_state_code")
                    or dataset_queue_state.get("code", ""),
                    "dataset_queue_state_title": handoff.get("dataset_queue_state_title")
                    or dataset_queue_state.get("title", ""),
                    **_labeler_work_completion_fields(
                        handoff.get("labeler_work_completion")
                        if isinstance(handoff.get("labeler_work_completion"), Mapping)
                        else None
                    ),
                    "dataset_queue_blocks_labeler_start": (
                        "" if dataset_queue_blocks_start is None else bool(dataset_queue_blocks_start)
                    ),
                    "dataset_queue_start_ready": handoff.get(
                        "dataset_queue_start_ready", queue_start_fields["dataset_queue_start_ready"]
                    ),
                    "dataset_queue_start_status": handoff.get(
                        "dataset_queue_start_status", queue_start_fields["dataset_queue_start_status"]
                    ),
                    "dataset_queue_start_operator_action": handoff.get(
                        "dataset_queue_start_operator_action",
                        queue_start_fields["dataset_queue_start_operator_action"],
                    ),
                    "dataset_queue_preview_url": handoff.get("dataset_queue_preview_url", ""),
                    "canonical_dataset_queue_preview_url": handoff.get(
                        "canonical_dataset_queue_preview_url",
                        handoff.get("expected_user_dataset_queue_url", ""),
                    ),
                    "signed_links": counts.get("signed_links", 0),
                    "recordings_without_open_tasks": counts.get("recordings_without_open_tasks", 0),
                    "recordings_without_open_tasks_by_reason": json.dumps(
                        counts.get("recordings_without_open_tasks_by_reason", {}),
                        sort_keys=True,
                    ),
                    "recordings_without_open_tasks_actions": " ".join(
                        str(action)
                        for action in (
                            handoff.get("recordings_without_open_tasks_actions")
                            if isinstance(handoff.get("recordings_without_open_tasks_actions"), list)
                            else counts.get("recordings_without_open_tasks_actions")
                            if isinstance(counts.get("recordings_without_open_tasks_actions"), list)
                            else []
                        )
                        if str(action).strip()
                    ),
                    "redacted_summary_fields": counts.get("redacted_summary_fields", 0),
                    "store_check_issues": counts.get("store_check_issues", 0),
                    "store_check_warnings": counts.get("store_check_warnings", 0),
                    "index_html": handoff.get("index_html", ""),
                    "message": files.get("message", ""),
                    "quickstart": files.get("quickstart", ""),
                    "dataset_queue": files.get("dataset_queue", ""),
                    "manifest": handoff.get("manifest", ""),
                }
            )
