"""Handoff shareability and inspection-report helpers for web labeling."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


def configure_handoff_shareability_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

def _single_owner_package_contract_summary_impl(
    handoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    users_by_recording: dict[str, set[str]] = {}
    for handoff in handoffs:
        handoff_user = str(handoff.get("user") or "").strip()
        assignment_snapshot = (
            handoff.get("assignment_snapshot")
            if isinstance(handoff.get("assignment_snapshot"), Mapping)
            else {}
        )
        assignments = (
            assignment_snapshot.get("assignments")
            if isinstance(assignment_snapshot.get("assignments"), list)
            else []
        )
        if assignments:
            for assignment in assignments:
                if not isinstance(assignment, Mapping):
                    continue
                status = str(assignment.get("status") or "active").strip() or "active"
                if status != "active":
                    continue
                recording_id = str(assignment.get("recording_id") or "").strip()
                user = str(assignment.get("assignee_user") or handoff_user).strip()
                if recording_id and user:
                    users_by_recording.setdefault(recording_id, set()).add(user)
            continue
        recording_ids = (
            assignment_snapshot.get("recording_ids")
            if isinstance(assignment_snapshot.get("recording_ids"), list)
            else []
        )
        for recording_id_value in recording_ids:
            recording_id = str(recording_id_value or "").strip()
            if recording_id and handoff_user:
                users_by_recording.setdefault(recording_id, set()).add(handoff_user)
    duplicate_owners_by_recording = {
        recording_id: sorted(users)
        for recording_id, users in sorted(users_by_recording.items())
        if len(users) > 1
    }
    mismatch_recording_ids = sorted(duplicate_owners_by_recording)
    met = bool(handoffs) and not mismatch_recording_ids
    return {
        "schema": "palette.web_labeling_single_owner_package_contract.v1",
        "met": met,
        "assignment_scope": "recording",
        "recording_assignment_key": "recording_id",
        "one_active_owner_per_recording": True,
        "multiple_labelers_per_recording_allowed": False,
        "recording_count": len(users_by_recording),
        "mismatch_count": len(mismatch_recording_ids),
        "mismatch_recording_ids": mismatch_recording_ids,
        "users_by_recording": {
            recording_id: sorted(users)
            for recording_id, users in sorted(users_by_recording.items())
        },
        "duplicate_owners_by_recording": duplicate_owners_by_recording,
        "operator_action": ""
        if met
        else (
            "Regenerate or repair the handoff from the current assignment store so "
            "each recording appears under exactly one active labeler before links are shared."
        ),
    }


def _handoff_sendability_reasons_impl(manifest: Mapping[str, object]) -> list[str]:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    base_url_present = bool(str(manifest.get("base_url") or "").strip())
    reasons: list[str] = []
    known_user_fields = _handoff_known_user_status_fields(manifest)
    if not bool(known_user_fields["known_user_status_present"]):
        reasons.append("known_user_status_missing")
    elif not bool(known_user_fields["known_labeler"]):
        reasons.append("unknown_labeling_user")
    elif int(known_user_fields["known_user_active_assignment_count"]) <= 0:
        reasons.append("no_active_assignment")
    ownership_fields = _handoff_assignment_ownership_fields(manifest)
    if not bool(ownership_fields["assignment_ownership_evidence_present"]):
        reasons.append("assignment_ownership_missing")
    elif not bool(ownership_fields["assignment_ownership_ok"]):
        reasons.append("assignment_ownership_conflict")
    elif not bool(ownership_fields["assignment_ownership_contract_ready"]):
        reasons.append("assignment_ownership_contract_not_ready")
    entry_fields = _handoff_entry_artifact_fields(manifest)
    if base_url_present and not bool(entry_fields["preferred_labeler_entry_url_matches_personal_dataset_queue"]):
        reasons.append("preferred_personal_queue_mismatch")
    labeler_safety_fields = _handoff_labeler_safety_fields(manifest)
    if not bool(labeler_safety_fields["labeler_safety_policy_present"]):
        reasons.append("labeler_safety_policy_missing")
    elif not bool(labeler_safety_fields["labeler_safety_ready"]):
        reasons.append("labeler_safety_policy_not_ready")
    route_authorization_fields = _handoff_labeler_route_authorization_fields(manifest)
    if not bool(route_authorization_fields["labeler_route_authorization_policy_present"]):
        reasons.append("labeler_route_authorization_policy_missing")
    elif not bool(route_authorization_fields["labeler_route_authorization_ready"]):
        reasons.append("labeler_route_authorization_policy_not_ready")
    signed_link_fields = _handoff_signed_link_policy_fields(manifest)
    if not bool(signed_link_fields["signed_link_policy_present"]):
        reasons.append("signed_link_policy_missing")
    elif not bool(signed_link_fields["signed_link_policy_ready"]):
        reasons.append("signed_link_policy_not_ready")
    session_guard_fields = _handoff_session_guard_fields(manifest)
    if not bool(session_guard_fields["session_guard_policy_present"]):
        reasons.append("session_guard_policy_missing")
    elif not bool(session_guard_fields["session_guard_policy_ready"]):
        reasons.append("session_guard_policy_not_ready")
    task_state_fields = _handoff_task_state_policy_fields(manifest)
    if not bool(task_state_fields["task_state_policy_present"]):
        reasons.append("task_state_policy_missing")
    elif not bool(task_state_fields["task_state_policy_ready"]):
        reasons.append("task_state_policy_not_ready")
    zarr_backup_fields = _handoff_zarr_backup_fields(manifest)
    if not bool(zarr_backup_fields["zarr_backup_policy_present"]):
        reasons.append("zarr_backup_policy_missing")
    elif not bool(zarr_backup_fields["zarr_backup_ready"]):
        reasons.append("zarr_backup_policy_not_ready")
    mutation_audit_fields = _handoff_mutation_audit_fields(manifest)
    if not bool(mutation_audit_fields["mutation_audit_policy_present"]):
        reasons.append("mutation_audit_policy_missing")
    elif not bool(mutation_audit_fields["mutation_audit_ready"]):
        reasons.append("mutation_audit_policy_not_ready")
    response_security_fields = _handoff_browser_response_security_fields(manifest)
    if not bool(response_security_fields["browser_response_security_policy_present"]):
        reasons.append("browser_response_security_policy_missing")
    elif not bool(response_security_fields["browser_response_security_ready"]):
        reasons.append("browser_response_security_policy_not_ready")
    mutation_write_fields = _handoff_browser_mutation_write_fields(manifest)
    if not bool(mutation_write_fields["browser_mutation_write_policy_present"]):
        reasons.append("browser_mutation_write_policy_missing")
    elif not bool(mutation_write_fields["browser_mutation_write_ready"]):
        reasons.append("browser_mutation_write_policy_not_ready")
    store_consistency = (
        manifest.get("store_consistency")
        if isinstance(manifest.get("store_consistency"), Mapping)
        else {}
    )
    reassignment_session_safety = manifest.get("reassignment_session_safety")
    if not isinstance(reassignment_session_safety, Mapping):
        reassignment_session_safety = (
            store_consistency.get("reassignment_session_safety")
            if isinstance(store_consistency.get("reassignment_session_safety"), Mapping)
            else {}
        )
    if reassignment_session_safety and not bool(reassignment_session_safety.get("ok")):
        reasons.append("reassignment_session_safety_failed")
    if not bool(manifest.get("ok")):
        reasons.append("store_check_failed")
    if int(counts.get("tasks") or 0) <= 0:
        reasons.append("no_tasks")
    if int(counts.get("signed_links") or 0) <= 0:
        reasons.append("no_signed_links")
    if not base_url_present:
        reasons.append("missing_base_url")
    elif int(counts.get("signed_links") or 0) > 0 and int(counts.get("ready_to_share_links") or 0) <= 0:
        reasons.append("no_ready_to_share_links")
    if base_url_present:
        guarded_link_reasons = {
            "expected_user_labeler_landing_url": "missing_guarded_landing_url",
            "expected_user_dataset_queue_url": "missing_guarded_dataset_queue_url",
            "expected_user_dashboard_url": "missing_guarded_dashboard_url",
            "expected_user_identity_probe_url": "missing_guarded_identity_probe_url",
        }
        for key in entry_fields["missing_guarded_links"]:
            reason = guarded_link_reasons.get(str(key))
            if reason:
                reasons.append(reason)
    artifact_reasons = {
        "html_index": "missing_handoff_index_artifact",
        "message": "missing_message_artifact",
        "quickstart": "missing_quickstart_artifact",
        "dataset_queue": "missing_dataset_queue_artifact",
        "manifest": "missing_handoff_manifest_artifact",
    }
    for key in entry_fields["missing_handoff_artifacts"]:
        reason = artifact_reasons.get(str(key))
        if reason:
            reasons.append(reason)
    if _handoff_dataset_queue_blocks_labeler_start(manifest):
        reasons.append("dataset_queue_blocks_labeler_start")
    if bool(manifest.get("operator_validation_required_before_invite")) and not bool(
        manifest.get("operator_validation_all_complete")
    ):
        needs_review_gate_ids = (
            manifest.get("operator_validation_needs_review_gate_ids")
            if isinstance(manifest.get("operator_validation_needs_review_gate_ids"), list)
            else []
        )
        reasons.append("operator_validation_needs_review" if needs_review_gate_ids else "operator_validation_pending")
    return reasons


def _handoff_sendability_actions_impl(reasons: Iterable[str]) -> list[str]:
    guidance = {
        "store_check_failed": "Run check-store and repair hard store issues before sending this handoff.",
        "reassignment_session_safety_failed": "Close stale previous-owner sessions or re-run assignment through assign_recording_with_session_closure before sending this handoff.",
        "no_tasks": "Generate, import, or reopen browser-labeling tasks for this user's assigned recordings.",
        "no_signed_links": "Regenerate the handoff with a link secret after startable tasks exist.",
        "missing_base_url": "Regenerate the handoff with --base-url set to the deployed labeling service URL.",
        "preferred_personal_queue_mismatch": "Regenerate the handoff so Start here uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of a canonical /datasets fallback.",
        "known_user_status_missing": "Regenerate the handoff so it includes current known-user assignment-store evidence.",
        "unknown_labeling_user": "Assign at least one recording to this user or correct the expected login identity before sharing links.",
        "no_active_assignment": "Activate or assign a recording for this user before sharing labeling links.",
        "assignment_ownership_missing": "Regenerate the handoff so assignment ownership integrity can be checked before sharing.",
        "assignment_ownership_conflict": "Repair duplicate active recording ownership before sharing labeler links.",
        "assignment_ownership_contract_not_ready": "Regenerate or repair assignment ownership evidence so the recording_id primary key, one-active-owner policy, reassignment session closure, and current-owner-only mutation contract are ready before sharing.",
        "no_ready_to_share_links": "Inspect signed-link shareability warnings; ensure tasks are active, incomplete, assigned, and generated with an absolute service URL.",
        "missing_guarded_landing_url": "Regenerate the handoff so it includes the expected-user guarded landing page link.",
        "missing_guarded_dataset_queue_url": "Regenerate the handoff so it includes the expected-user guarded dataset queue link.",
        "missing_guarded_dashboard_url": "Regenerate the handoff so it includes the expected-user guarded dashboard link.",
        "missing_guarded_identity_probe_url": "Regenerate the handoff so it includes the expected-user identity-probe link.",
        "missing_handoff_index_artifact": "Regenerate the handoff so the per-user index.html artifact is present.",
        "missing_message_artifact": "Regenerate the handoff so the per-user message.txt invitation artifact is present.",
        "missing_quickstart_artifact": "Regenerate the handoff so the per-user quickstart artifact is present.",
        "missing_dataset_queue_artifact": "Regenerate the handoff so the per-user dataset-queue.json artifact is present.",
        "missing_handoff_manifest_artifact": "Regenerate the handoff so the per-user manifest.json artifact is present.",
        "dataset_queue_blocks_labeler_start": "Resolve the user's dataset queue state before sending a start link; generate or reopen work if labeling should continue, or mark the assignment complete if no more labeling is required.",
        "labeler_safety_policy_missing": "Regenerate the handoff so labeler safety metadata is present before sharing links.",
        "labeler_safety_policy_not_ready": "Regenerate or repair the handoff so labelers are browser-only with no local Palette, Crimson, Conda, or project dependency installation, expected-user guarded, redacted from raw Zarr paths/task scope, failed identity probes suppress launch CTAs with diagnostic-only support URLs, and labelers are told not to edit Zarrs directly or forward links.",
        "labeler_route_authorization_policy_missing": "Regenerate the handoff so labeler-route authorization policy metadata is present before sharing links.",
        "labeler_route_authorization_policy_not_ready": "Regenerate or repair the handoff so copied queue, dashboard, and signed task links recheck resolved identity, expected user, known user, active assignment, startable task state, single-owner store proof, assignment integrity, zero duplicate active owners, server-resolved training-Zarr targets, and no intermediate CSV mutation before returning or mutating work; runtime route-checklist evidence must include single_owner_store_proof_ready=true, assignment_ownership_integrity_ok=true, browser_mutation_target_resolved_server_side=true, labelers_mutate_assigned_training_zarrs=true, and labelers_mutate_intermediate_csvs=false.",
        "signed_link_policy_missing": "Regenerate the handoff so signed-link policy metadata is present before sharing links.",
        "signed_link_policy_not_ready": "Regenerate or repair the handoff so signed links are task-specific expected-user-bound entry hints, not authorization grants or identity proofs, and enforce runtime operator-validation start gates before session creation.",
        "session_guard_policy_missing": "Regenerate the handoff so session-guard policy metadata is present before sharing links.",
        "session_guard_policy_not_ready": "Regenerate or repair the handoff so browser mutations require a current unexpired session, reject stale or superseded tabs, require the current target token, and expose closure-event support.",
        "task_state_policy_missing": "Regenerate the handoff so task-state policy metadata is present before sharing links.",
        "task_state_policy_not_ready": "Regenerate or repair the handoff so only pending or in-progress tasks can open labeler sessions, completed tasks and other non-startable tasks reject ordinary labeler open/save, operator reopen is required before more labeling, and browser mutations require the current server target token.",
        "zarr_backup_policy_missing": "Regenerate the handoff so Zarr backup policy metadata is present before sharing links.",
        "zarr_backup_policy_not_ready": "Regenerate or repair the handoff so mutable Zarr backups are operator-owned, copy-before-labeling is required, backup paths are hidden from labelers, and any mutable backup targets have a backup-plan artifact.",
        "mutation_audit_policy_missing": "Regenerate the handoff so mutation-audit policy metadata is present before sharing links.",
        "mutation_audit_policy_not_ready": "Regenerate or repair the handoff so browser mutations are server-recorded in append-only labeling task events, browsers cannot write audit records directly, and required task/user/event fields are present.",
        "browser_response_security_policy_missing": "Regenerate the handoff so browser response-security policy metadata is present before sharing links.",
        "browser_response_security_policy_not_ready": "Regenerate or repair the handoff so browser responses require no-store caching, anti-framing, MIME-sniffing protection, no-referrer behavior, narrow CSP, permissions restrictions, and proxy header preservation.",
        "browser_mutation_write_policy_missing": "Regenerate the handoff so browser mutation write policy metadata is present before sharing links.",
        "browser_mutation_write_policy_not_ready": "Regenerate or repair the handoff so browser saves target server-owned assigned task/training Zarr scope, task-scoped training Zarrs are the mutable label data plane and label mutation target kind, browser_label_write_target is training_zarr, and handoff CSV/HTML/JSON files are metadata-only control-plane non-label-write targets with no browser handoff/intermediate CSV writes.",
        "operator_validation_pending": "Complete required operator validation evidence before inviting labelers to start or save work.",
        "operator_validation_needs_review": "Resolve validation checklist gates marked needs_review before inviting labelers.",
        "operator_review_required": "Inspect the handoff manifest and store check before sending.",
    }
    actions: list[str] = []
    for reason in reasons:
        reason_text = str(reason or "").strip()
        if not reason_text:
            continue
        actions.append(guidance.get(reason_text, f"Inspect handoff sendability reason {reason_text}."))
    return actions


def _handoff_sendability_summary_impl(manifests: Sequence[Mapping[str, object]]) -> dict[str, object]:
    warnings: list[dict[str, object]] = []
    for manifest in manifests:
        counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
        reasons = _handoff_sendability_reasons(manifest)
        if reasons:
            warnings.append(
                {
                    "user": manifest.get("user"),
                    "output_dir": manifest.get("output_dir"),
                    "reasons": reasons,
                    "actions": _handoff_sendability_actions(reasons),
                    "recordings": counts.get("recordings", 0),
                    "tasks": counts.get("tasks", 0),
                    "signed_links": counts.get("signed_links", 0),
                    "ready_to_share_links": counts.get("ready_to_share_links", 0),
                    "details": "Handoff is not ready to send without operator review.",
                }
            )
    return {
        "ready_to_send_count": len(manifests) - len(warnings),
        "not_ready_to_send_count": len(warnings),
        "warnings": warnings,
    }


def _count_handoff_sendability_reasons_impl(warnings: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(warnings, list):
        return counts
    for warning in warnings:
        if not isinstance(warning, Mapping):
            continue
        reasons = warning.get("reasons")
        if not isinstance(reasons, list):
            reasons = warning.get("sendability_reasons")
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            reason_text = str(reason or "").strip()
            if not reason_text:
                continue
            counts[reason_text] = counts.get(reason_text, 0) + 1
    return dict(sorted(counts.items()))


def _shareability_safe_to_share_requires_impl() -> list[str]:
    return [
        "inspection_ok",
        "no_operator_action_required_before_share",
        "implementation_status_checklist_artifact_complete",
        "implementation_status_checklist_artifact_complete_matches_required_value",
        "browser_mutation_target_contract_met",
        "direct_browser_start_contract_met",
        "single_owner_package_contract_met",
        "labeler_route_authorization_runtime_checklist_gate_met",
    ]


def _shareability_labeler_route_authorization_runtime_checklist_gate_contract_impl() -> dict[str, object]:
    return {
        "schema": (
            "palette.web_labeling_labeler_route_authorization_runtime_checklist_"
            "shareability_gate_contract.v1"
        ),
        "gate_field": "labeler_route_authorization_runtime_checklist_gate",
        "compact_contract_field": (
            "shareability_contract.labeler_route_authorization_runtime_checklist_gate"
        ),
        "nested_shareability_field": (
            "shareability.labeler_route_authorization_runtime_checklist_gate"
        ),
        "observed_value_field": (
            "labeler_route_authorization_runtime_checklist_gate_met"
        ),
        "required_value": True,
        "safe_share_requires_value": (
            "labeler_route_authorization_runtime_checklist_gate_met"
        ),
        "required_fields_field": (
            "labeler_route_authorization_runtime_checklist_required_fields"
        ),
        "required_values_field": (
            "labeler_route_authorization_runtime_checklist_required_values"
        ),
        "mismatch_count_field": (
            "labeler_route_authorization_runtime_checklist_mismatch_count"
        ),
        "mismatch_users_field": (
            "labeler_route_authorization_runtime_checklist_mismatch_users"
        ),
        "mismatches_field": (
            "labeler_route_authorization_runtime_checklist_mismatches"
        ),
        "required_fields": (
            _shareability_labeler_route_authorization_runtime_checklist_fields()
        ),
        "required_values": (
            _shareability_labeler_route_authorization_runtime_checklist_required_values()
        ),
        "fail_closed_reason": (
            "labeler_route_authorization_runtime_checklist_mismatch"
        ),
        "repair_command_id": (
            "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
        ),
    }


def _shareability_compact_contract_fields_impl() -> list[str]:
    return [
        "schema",
        "decision_source",
        "top_level_field",
        "nested_field",
        "fields",
        "field_count",
        "source_fields",
        "source_field_count",
        "safe_to_share",
        "safe_to_share_required_value",
        "safe_to_share_matches_required_value",
        "status",
        "operator_action",
        "operator_action_required",
        "requirement_met",
        "blocking_reason_ids",
        "blocking_gate_ids",
        "repair_command_ids",
        "repair_command_count",
        "safe_share_gate",
        "safe_share_required_field",
        "safe_share_required_value",
        "safe_to_share_requires",
        "safe_share_launch_blocking_next_action_detail_fields",
        "safe_share_launch_blocking_next_action_command_fields",
        "safe_share_external_launch_evidence_gap_fields",
        *_safe_share_external_launch_evidence_gap_field_names(),
        "implementation_status_checklist_artifact_gate",
        "launch_evidence_execution_checklist_summary",
        "launch_evidence_execution_checklist_present",
        "launch_evidence_execution_checklist_valid",
        "launch_evidence_execution_checklist_contract_present",
        "launch_evidence_execution_checklist_blocking_reason_id",
        "browser_mutation_target_contract",
        "direct_browser_start_contract",
        "single_owner_package_contract",
        "labeler_route_authorization_runtime_checklist_gate",
        "labeler_route_authorization_runtime_checklist_gate_met",
        "labeler_route_authorization_runtime_checklist_mismatch_count",
        "labeler_route_authorization_runtime_checklist_mismatch_users",
        "labeler_route_authorization_runtime_checklist_mismatches",
        "repair_command_detail_fields",
        "repair_command_detail_fields_by_id",
        "repair_command_contracts",
    ]


def _shareability_compact_contract_source_fields_impl() -> dict[str, str]:
    return {
        "safe_to_share": "labeler_links_safe_to_share",
        "safe_to_share_required_value": "constant:true",
        "safe_to_share_matches_required_value": (
            "shareability_contract.safe_to_share == "
            "shareability_contract.safe_to_share_required_value"
        ),
        "status": "shareability_status",
        "operator_action": "shareability_operator_action",
        "operator_action_required": "operator_action_required_before_share",
        "requirement_met": "shareability_requirement_met",
        "blocking_reason_ids": "shareability_blocking_reason_ids",
        "blocking_gate_ids": "shareability_blocking_gate_ids",
        "repair_command_ids": "operator_repair_commands[].id",
        "repair_command_count": "operator_repair_command_count",
        "safe_to_share_requires": "shareability.safe_to_share_requires",
        "safe_share_launch_blocking_next_action_detail_fields": (
            "safe_share_launch_blocking_next_action_detail_fields"
        ),
        "safe_share_launch_blocking_next_action_command_fields": (
            "safe_share_launch_blocking_next_action_command_fields"
        ),
        "safe_share_external_launch_evidence_gap_fields": (
            "shareability_contract.safe_share_external_launch_evidence_gap_fields"
        ),
        **{
            field_name: field_name
            for field_name in _safe_share_external_launch_evidence_gap_field_names()
        },
        "implementation_status_checklist_artifact_gate": (
            "implementation_status_checklist_artifact_gate"
        ),
        "launch_evidence_execution_checklist_summary": (
            "launch_evidence_execution_checklist_summary"
        ),
        "launch_evidence_execution_checklist_present": (
            "launch_evidence_execution_checklist_summary.present"
        ),
        "launch_evidence_execution_checklist_valid": (
            "launch_evidence_execution_checklist_summary.valid"
        ),
        "launch_evidence_execution_checklist_contract_present": (
            "launch_evidence_execution_checklist_summary.checklist_contract_present"
        ),
        "launch_evidence_execution_checklist_blocking_reason_id": (
            "launch_evidence_execution_checklist_summary.blocking_reason_id"
        ),
        "labeler_route_authorization_runtime_checklist_gate": (
            "shareability_labeler_route_authorization_runtime_checklist_fields + "
            "shareability_labeler_route_authorization_runtime_checklist_required_values"
        ),
        "labeler_route_authorization_runtime_checklist_gate_met": (
            "labeler_route_authorization_runtime_checklist_gate.met"
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": (
            "labeler_route_authorization_runtime_checklist_gate.mismatch_count"
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": (
            "labeler_route_authorization_runtime_checklist_gate.mismatch_users"
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": (
            "labeler_route_authorization_runtime_checklist_gate.mismatches"
        ),
        "repair_command_contracts": "operator_repair_command_contracts",
    }


def _shareability_compact_contract_safe_to_share_target_impl() -> dict[str, object]:
    return {
        "shareability_compact_contract_safe_to_share_field": (
            "shareability_contract.safe_to_share"
        ),
        "shareability_compact_contract_safe_to_share_required_value": True,
        "shareability_compact_contract_safe_to_share_matches_required_value_field": (
            "shareability_contract.safe_to_share_matches_required_value"
        ),
    }


def _shareability_compact_contract_next_action_target_impl() -> dict[str, object]:
    return {
        "shareability_compact_contract_next_action_detail_fields_field": (
            "shareability_contract.safe_share_launch_blocking_next_action_detail_fields"
        ),
        "shareability_compact_contract_next_action_command_fields_field": (
            "shareability_contract.safe_share_launch_blocking_next_action_command_fields"
        ),
        "shareability_nested_next_action_detail_fields_field": (
            "shareability.safe_share_launch_blocking_next_action_detail_fields"
        ),
        "shareability_nested_next_action_command_fields_field": (
            "shareability.safe_share_launch_blocking_next_action_command_fields"
        ),
        "shareability_next_action_detail_fields": _safe_share_next_action_detail_fields(),
        "shareability_next_action_command_fields": _safe_share_next_action_command_fields(),
    }


def _shareability_external_launch_evidence_gap_target_impl() -> dict[str, object]:
    return {
        "shareability_safe_share_external_launch_evidence_gap_fields": (
            _safe_share_external_launch_evidence_gap_field_names()
        ),
        "shareability_compact_contract_external_launch_evidence_gap_fields_field": (
            "shareability_contract.safe_share_external_launch_evidence_gap_fields"
        ),
        "shareability_nested_external_launch_evidence_gap_fields_field": (
            "shareability.safe_share_external_launch_evidence_gap_fields"
        ),
        "shareability_external_launch_evidence_gap_top_level_field_prefix": (
            "safe_share_external_launch_evidence_gap_"
        ),
        "shareability_external_launch_evidence_gap_nested_field_prefix": (
            "shareability.safe_share_external_launch_evidence_gap_"
        ),
    }


def _shareability_repair_command_detail_fields_impl() -> list[str]:
    return [
        "id",
        "category",
        "contract",
        "repair_mode",
        "safe_share_blocker",
        "safe_share_blockers",
        "required_values",
        "mismatches",
        "mismatch_count",
        "mismatch_users",
        "mismatch_recording_ids",
        "duplicate_owners_by_recording",
        "missing_fields",
        "missing_field_count",
        "missing_phrases",
        "missing_phrase_count",
        "required_file",
        "required_phrase_contract",
        "required_phrases",
        "artifact_contract",
    ]


def _shareability_repair_command_detail_fields_by_id_impl() -> dict[str, list[str]]:
    return {
        "regenerate_handoffs_with_browser_mutation_target_contract": [
            "contract",
            "repair_mode",
            "safe_share_blocker",
            "required_values",
            "mismatches",
            "mismatch_count",
            "mismatch_users",
        ],
        "regenerate_handoffs_with_direct_browser_start_contract": [
            "contract",
            "repair_mode",
            "safe_share_blocker",
            "required_values",
            "mismatches",
            "mismatch_count",
            "mismatch_users",
        ],
        "regenerate_handoffs_with_single_owner_package_contract": [
            "contract",
            "repair_mode",
            "safe_share_blocker",
            "mismatch_count",
            "mismatch_recording_ids",
            "duplicate_owners_by_recording",
        ],
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist": [
            "contract",
            "repair_mode",
            "safe_share_blocker",
            "required_values",
            "mismatches",
            "mismatch_count",
            "mismatch_users",
        ],
        "regenerate_package_with_implementation_status_artifact": [
            "missing_fields",
            "missing_field_count",
            "repair_mode",
            "artifact_contract",
            "safe_share_blocker",
            "safe_share_blockers",
        ],
        "regenerate_package_with_launch_evidence_execution_checklist": [
            "required_file",
            "required_phrase_contract",
            "required_phrases",
            "missing_phrases",
            "missing_phrase_count",
            "repair_mode",
            "artifact_contract",
            "safe_share_blocker",
        ],
    }


def _shareability_repair_command_contracts_impl() -> dict[str, dict[str, object]]:
    return {
        "regenerate_handoffs_with_browser_mutation_target_contract": {
            "contract": "browser_mutation_target_contract",
            "repair_mode": "regenerate_handoff_package",
            "safe_share_blocker": "browser_mutation_target_contract_mismatch",
            "required_target": "training_zarr",
            "metadata_only_artifacts": "handoff_csv_and_intermediate_csv",
        },
        "regenerate_handoffs_with_direct_browser_start_contract": {
            "contract": "direct_browser_start_contract",
            "repair_mode": "regenerate_handoff_package",
            "safe_share_blocker": "direct_browser_start_contract_mismatch",
            "required_target": "training_zarr",
            "metadata_only_artifacts": "handoff_csv_and_intermediate_csv",
        },
        "regenerate_handoffs_with_single_owner_package_contract": {
            "contract": "single_owner_package_contract",
            "repair_mode": "regenerate_handoff_package",
            "safe_share_blocker": "single_owner_package_contract_mismatch",
            "multiple_active_owners_allowed": False,
        },
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist": {
            "contract": "labeler_route_authorization_runtime_checklist_gate",
            "repair_mode": "regenerate_handoff_package",
            "safe_share_blocker": (
                "labeler_route_authorization_runtime_checklist_mismatch"
            ),
            "requires_runtime_checklist_present": True,
            "requires_runtime_checklist_ready": True,
            "requires_single_owner_store_proof_ready": True,
            "required_duplicate_active_owner_count": 0,
            "required_target": "training_zarr",
            "rejects_intermediate_csv_mutation": True,
        },
        "regenerate_package_with_implementation_status_artifact": {
            "artifact_contract": "implementation_status_artifact",
            "repair_mode": "regenerate_package",
            "safe_share_blocker": (
                "implementation_status_checklist_artifact_complete_required_value_mismatch"
            ),
        },
        "regenerate_package_with_launch_evidence_execution_checklist": {
            "artifact_contract": "launch_evidence_execution_checklist",
            "repair_mode": "regenerate_package",
            "safe_share_blocker": "launch_evidence_execution_checklist_incomplete",
            "safe_share_blockers": [
                "launch_evidence_execution_checklist_missing",
                "launch_evidence_execution_checklist_incomplete",
                "launch_evidence_execution_checklist_invalid",
            ],
            "required_file": "launch-evidence-execution-checklist.txt",
            "required_phrase_contract": (
                "shareability_launch_evidence_execution_checklist_required_phrases"
            ),
            "required_phrases": [
                "Palette web-labeling launch evidence execution checklist",
                "Operator-only checklist",
                "record-zarr-backup-evidence",
                "record-browser-response-security-evidence",
                "record-identity-source-evidence",
                "record-browser-smoke-evidence",
                "record-disposable-zarr-mutation-smoke-evidence",
                "apply-operator-evidence-templates",
                "inspect-handoff --path PACKAGE_PATH --require-shareable",
                "labeler_links_safe_to_share=true",
            ],
        },
    }


def _write_launch_bundle_inspection_targets_impl(
    *,
    store_path: Path,
    output_dir: Path,
    zip_output: Path | None,
    output_path: Path,
) -> None:
    response_security_policy = _browser_response_security_policy()
    response_security_inspection_fields = {
        "shareability_response_security_protected_labeler_paths": list(
            response_security_policy.get("protected_labeler_paths", [])
        ),
        "shareability_response_security_personal_api_paths": list(
            response_security_policy.get("personal_api_paths", [])
        ),
        "shareability_response_security_preferred_capture_path": PERSONAL_DATASET_QUEUE_PATH,
        "shareability_response_security_labeling_home_path": LABELING_HOME_PATH,
        "shareability_response_security_personal_work_path": PERSONAL_WORK_PATH,
        "shareability_response_security_identity_probe_path": IDENTITY_PROBE_PATH,
        "shareability_response_security_identity_probe_api_path": "/api/me/identity",
        "shareability_response_security_expected_user_query_required": True,
        "shareability_response_security_route_header_parity_required": True,
    }
    operator_validation_command_templates = _operator_validation_command_templates()
    launch_evidence_collection_plan = (
        operator_validation_command_templates.get("launch_evidence_collection_plan")
        if isinstance(
            operator_validation_command_templates.get("launch_evidence_collection_plan"),
            Mapping,
        )
        else {}
    )
    launch_evidence_collection_inspection_fields = {
        "operator_validation_launch_evidence_collection_plan_contract": {
            "schema": "palette.web_labeling_launch_evidence_collection_plan_contract.v1",
            "plan_field": "operator_validation_command_templates.launch_evidence_collection_plan",
            "plan_schema": "palette.web_labeling_launch_evidence_collection_plan.v1",
            "flat_field_prefix": (
                "operator_validation_command_template_launch_evidence_collection"
            ),
            "flat_fields": [
                "operator_validation_command_template_launch_evidence_collection_plan_schema",
                "operator_validation_command_template_launch_evidence_collection_step_count",
                "operator_validation_command_template_launch_evidence_collection_gate_ids",
                "operator_validation_command_template_launch_evidence_collection_record_command_ids",
                "operator_validation_command_template_launch_evidence_collection_operator_only",
                "operator_validation_command_template_launch_evidence_collection_required_final_field",
                "operator_validation_command_template_launch_evidence_collection_required_final_value",
                "operator_validation_command_template_launch_evidence_collection_final_inspection_command",
            ],
            "step_fields": [
                "gate_id",
                "gate_category",
                "operator_only",
                "blocks_labeler_link_share_until_satisfied",
                "record_command_id",
                "record_command",
                "evidence_template_field",
                "evidence_template_path",
                "template_backed",
                "apply_required_after_approval",
                "apply_command_id",
                "requires_checksum_refresh_after_run",
            ],
            "operator_only_required": True,
            "commands_are_labeler_instructions_required_value": False,
            "labelers_must_not_run_commands_required_value": True,
            "safe_to_share_blocked_until_plan_complete_required_value": True,
            "required_final_field": "labeler_links_safe_to_share",
            "required_final_value": True,
            "final_inspection_command": "inspect-handoff --path PACKAGE --require-shareable",
            "gate_ids": list(
                launch_evidence_collection_plan.get("gate_ids")
                if isinstance(launch_evidence_collection_plan.get("gate_ids"), list)
                else operator_validation_command_templates.get("gate_ids")
                if isinstance(operator_validation_command_templates.get("gate_ids"), list)
                else []
            ),
            "record_command_ids": list(
                launch_evidence_collection_plan.get("record_command_ids")
                if isinstance(
                    launch_evidence_collection_plan.get("record_command_ids"),
                    list,
                )
                else operator_validation_command_templates.get(
                    "launch_evidence_collection_record_command_ids"
                )
                if isinstance(
                    operator_validation_command_templates.get(
                        "launch_evidence_collection_record_command_ids"
                    ),
                    list,
                )
                else []
            ),
        },
        "operator_validation_launch_evidence_collection_plan_field": (
            "operator_validation_command_templates.launch_evidence_collection_plan"
        ),
        "operator_validation_launch_evidence_collection_plan_schema": (
            "palette.web_labeling_launch_evidence_collection_plan.v1"
        ),
        "operator_validation_launch_evidence_collection_flat_fields": [
            "operator_validation_command_template_launch_evidence_collection_plan_schema",
            "operator_validation_command_template_launch_evidence_collection_step_count",
            "operator_validation_command_template_launch_evidence_collection_gate_ids",
            "operator_validation_command_template_launch_evidence_collection_record_command_ids",
            "operator_validation_command_template_launch_evidence_collection_operator_only",
            "operator_validation_command_template_launch_evidence_collection_required_final_field",
            "operator_validation_command_template_launch_evidence_collection_required_final_value",
            "operator_validation_command_template_launch_evidence_collection_final_inspection_command",
        ],
    }
    safe_share_next_action_detail_fields = _safe_share_next_action_detail_fields()
    safe_share_next_action_command_fields = _safe_share_next_action_command_fields()
    ready_row_draft_inspection_fields = {
        "shareability_ready_row_draft_bundle_schema": (
            _DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA
        ),
        "shareability_ready_row_draft_bundle_kind": (
            _DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND
        ),
        "shareability_ready_row_draft_top_level_fields": [
            "ready_row_draft_bundle_schema",
            "ready_row_draft_bundle_kind",
            "ready_row_drafts",
            "ready_row_draft_text",
            "ready_row_draft_share_rule",
            "ready_invitations_legacy_semantics",
            "ready_invitations_legacy_field_names",
        ],
        "shareability_ready_row_draft_legacy_field_names": list(
            _DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES
        ),
        "shareability_browser_mutation_target_fields": list(
            _DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS
        ),
        "shareability_browser_mutation_target_required_values": dict(
            _DASHBOARD_BROWSER_MUTATION_TARGET_REQUIRED_VALUES
        ),
        "shareability_browser_mutation_target_selector_policy": (
            "server_owned_reject_client_fields"
        ),
        "shareability_browser_mutation_target_selector_fields_rejected": list(
            BROWSER_MUTATION_TARGET_SELECTOR_KEYS
        ),
        "shareability_direct_browser_start_fields": list(
            _DASHBOARD_DIRECT_BROWSER_START_FIELDS
        ),
        "shareability_direct_browser_start_required_values": dict(
            _DASHBOARD_DIRECT_BROWSER_START_REQUIRED_VALUES
        ),
        "shareability_single_owner_package_fields": [
            "single_owner_package_contract_met",
            "single_owner_package_mismatch_count",
            "single_owner_package_mismatch_recording_ids",
            "single_owner_package_duplicate_owners_by_recording",
        ],
        "shareability_single_owner_store_contract_fields": [
            "single_owner_policy_browser_mutation_target_resolved_server_side",
            "single_owner_policy_browser_mutation_target_source",
            "single_owner_policy_labelers_mutate_assigned_training_zarrs",
            "single_owner_policy_labelers_mutate_intermediate_csvs",
            "assignment_ownership_contract_browser_mutation_target_resolved_server_side",
            "assignment_ownership_contract_browser_mutation_target_source",
            "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs",
            "assignment_ownership_contract_labelers_mutate_intermediate_csvs",
            "assignment_ownership_contract_store_single_owner_assignment_contract_present",
            "assignment_ownership_contract_store_single_owner_assignment_contract_ready",
            "assignment_ownership_contract_store_single_owner_assignment_contract_met",
            "assignment_ownership_contract_store_single_owner_assignment_contract_schema",
        ],
        "shareability_single_owner_store_contract_required_values": {
            "single_owner_policy_browser_mutation_target_resolved_server_side": True,
            "single_owner_policy_browser_mutation_target_source": (
                "recording_assignments.active_assignment"
            ),
            "single_owner_policy_labelers_mutate_assigned_training_zarrs": True,
            "single_owner_policy_labelers_mutate_intermediate_csvs": False,
            "assignment_ownership_contract_browser_mutation_target_resolved_server_side": True,
            "assignment_ownership_contract_browser_mutation_target_source": (
                "recording_assignments.active_assignment"
            ),
            "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs": True,
            "assignment_ownership_contract_labelers_mutate_intermediate_csvs": False,
            "assignment_ownership_contract_store_single_owner_assignment_contract_present": True,
            "assignment_ownership_contract_store_single_owner_assignment_contract_ready": True,
            "assignment_ownership_contract_store_single_owner_assignment_contract_met": True,
            "assignment_ownership_contract_store_single_owner_assignment_contract_schema": (
                "palette.web_labeling_assignment_single_owner_contract.v1"
            ),
        },
        "shareability_labeler_route_authorization_store_proof_fields": [
            "labeler_route_authorization_single_owner_store_proof_required_for_browser_work",
            "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok",
            "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners",
            "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target",
            "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation",
        ],
        "shareability_labeler_route_authorization_store_proof_required_values": {
            "labeler_route_authorization_single_owner_store_proof_required_for_browser_work": True,
            "labeler_route_authorization_single_owner_store_proof_requires_integrity_ok": True,
            "labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners": True,
            "labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target": True,
            "labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation": True,
        },
        "shareability_labeler_route_authorization_runtime_checklist_fields": list(
            _shareability_labeler_route_authorization_runtime_checklist_fields()
        ),
        "shareability_labeler_route_authorization_runtime_checklist_required_values": dict(
            _shareability_labeler_route_authorization_runtime_checklist_required_values()
        ),
        "shareability_labeler_route_authorization_runtime_checklist_gate_contract": (
            _shareability_labeler_route_authorization_runtime_checklist_gate_contract()
        ),
        "shareability_labeler_safety_policy_fields": [
            "labeler_safety_policy_present",
            "labeler_safety_ready",
            "labeler_safety_browser_only",
            "labeler_safety_no_local_install_required",
            "labeler_safety_identity_probe_expected_user_guard_required",
            "labeler_safety_identity_probe_diagnostic_only",
            "labeler_safety_identity_probe_does_not_authorize_work",
            "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces",
            "labeler_safety_identity_probe_success_launch_ctas_rendered",
            "labeler_safety_identity_probe_failed_launch_ctas_suppressed",
            "labeler_safety_identity_probe_failed_support_urls_diagnostic_only",
            "labeler_safety_browser_receives_task_scope",
            "labeler_safety_browser_receives_raw_zarr_paths",
        ],
        "shareability_labeler_safety_policy_required_values": {
            "labeler_safety_policy_present": True,
            "labeler_safety_ready": True,
            "labeler_safety_browser_only": True,
            "labeler_safety_no_local_install_required": True,
            "labeler_safety_identity_probe_expected_user_guard_required": True,
            "labeler_safety_identity_probe_diagnostic_only": True,
            "labeler_safety_identity_probe_does_not_authorize_work": True,
            "labeler_safety_identity_probe_unknown_user_blocks_work_surfaces": True,
            "labeler_safety_identity_probe_success_launch_ctas_rendered": True,
            "labeler_safety_identity_probe_failed_launch_ctas_suppressed": True,
            "labeler_safety_identity_probe_failed_support_urls_diagnostic_only": True,
            "labeler_safety_browser_receives_task_scope": False,
            "labeler_safety_browser_receives_raw_zarr_paths": False,
        },
        "shareability_compact_contract_gate_fields": [
            "browser_mutation_target_contract_met",
            "browser_mutation_target_mismatch_count",
            "direct_browser_start_contract_met",
            "direct_browser_start_mismatch_count",
            "single_owner_policy_contract_met",
            "labeler_route_authorization_runtime_checklist_gate_met",
            "labeler_route_authorization_runtime_checklist_mismatch_count",
        ],
        "shareability_ready_row_draft_row_fields": [
            "ready_row_state",
            "ready_row_draft_bundle_schema",
            "ready_row_draft_bundle_kind",
            "ready_row_draft_share_rule",
            "ready_row_draft_requires_safe_share_inspection",
            "ready_row_draft_required_safe_share_field",
            "ready_row_draft_required_safe_share_value",
        ],
        "shareability_ready_row_state_values": list(_DASHBOARD_READY_ROW_STATE_VALUES),
        "shareability_copy_intent_values": list(_DASHBOARD_COPY_INTENT_VALUES),
        "shareability_ready_row_draft_required_safe_share_field": (
            _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_FIELD
        ),
        "shareability_ready_row_draft_required_safe_share_value": (
            _DASHBOARD_READY_ROW_DRAFT_REQUIRED_SAFE_SHARE_VALUE
        ),
        **_implementation_status_inspection_target_fields(),
    }
    shareability_safe_to_share_requires = _shareability_safe_to_share_requires()
    shareability_repair_command_detail_fields = (
        _shareability_repair_command_detail_fields()
    )
    shareability_repair_command_detail_fields_by_id = (
        _shareability_repair_command_detail_fields_by_id()
    )
    shareability_repair_command_contracts = (
        _shareability_repair_command_contracts()
    )
    targets: list[dict[str, object]] = [
        {
            "schema": "palette.web_labeling_inspection_target.v1",
            "kind": "directory",
            "path": str(output_dir),
            "shareability_required": True,
            "shareability_gate": "labeler_links_safe_to_share",
            "shareability_contract_schema": "palette.web_labeling_handoff_shareability.v1",
            "shareability_decision_source": "inspect_handoff_package",
            "shareability_contract_field": "shareability_contract",
            "shareability_nested_contract_field": "shareability.contract",
            "shareability_compact_contract_schema": (
                "palette.web_labeling_handoff_shareability_contract.v1"
            ),
            "shareability_compact_contract_fields": list(_shareability_compact_contract_fields()),
            "shareability_compact_contract_field_count": len(_shareability_compact_contract_fields()),
            "shareability_compact_contract_source_fields": dict(_shareability_compact_contract_source_fields()),
            "shareability_compact_contract_source_field_count": len(_shareability_compact_contract_source_fields()),
            **_shareability_compact_contract_safe_to_share_target(),
            **_shareability_compact_contract_next_action_target(),
            **_shareability_external_launch_evidence_gap_target(),
            **_personalized_launch_readiness_target(),
            "shareability_safe_to_share_requires": list(shareability_safe_to_share_requires),
            "shareability_repair_commands_field": "repair_commands",
            "shareability_repair_command_detail_fields": list(shareability_repair_command_detail_fields),
            "shareability_repair_command_detail_fields_by_id": dict(shareability_repair_command_detail_fields_by_id),
            "shareability_repair_command_contracts": dict(shareability_repair_command_contracts),
            "safe_share_gate": _safe_share_gate_policy(),
            **_safe_share_gate_flat_fields(),
            "shareability_safe_share_next_action_fields": [
                "safe_share_launch_blocking_next_actions",
                "safe_share_launch_blocking_next_action_count",
                "safe_share_next_action_summary",
            ],
            "shareability_safe_share_next_action_detail_fields": list(
                safe_share_next_action_detail_fields
            ),
            "shareability_safe_share_next_action_command_fields": list(
                safe_share_next_action_command_fields
            ),
            **ready_row_draft_inspection_fields,
            **response_security_inspection_fields,
            **launch_evidence_collection_inspection_fields,
            **_launch_evidence_execution_checklist_inspection_target(),
            "shareability_identity_personal_queue_evidence_required": True,
            "shareability_identity_personal_queue_evidence_field": (
                "shareability.identity_personal_queue_evidence"
            ),
            "shareability_identity_personal_queue_evidence_status_values": list(
                IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES
            ),
            "shareability_operator_validation_gate_status_values": list(
                OPERATOR_VALIDATION_GATE_STATUS_VALUES
            ),
            "shareability_operator_validation_gate_ids": list(
                DEFAULT_OPERATOR_VALIDATION_GATE_IDS
            ),
            "shareability_operator_validation_gate_flat_field_suffixes": [
                "status",
                "pending",
                "missing_evidence",
                "needs_review",
                "passed",
            ],
            "shareability_identity_personal_queue_evidence_top_level_fields": [
                "identity_personal_queue_evidence_status",
                "identity_personal_queue_evidence_ready_count",
                "identity_personal_queue_evidence_missing_count",
                "identity_personal_queue_evidence_ready_users",
                "identity_personal_queue_evidence_missing_users",
                "identity_personal_queue_evidence_missing_fields_by_user",
                "identity_all_users_have_personal_queue_evidence",
            ],
            "command": "scripts/py -m fisheye.utils.labeling_work "
            f"--store {store_path} inspect-handoff --path {output_dir} --require-shareable",
        }
    ]
    if zip_output is not None:
        targets.append(
            {
                "schema": "palette.web_labeling_inspection_target.v1",
                "kind": "zip",
                "path": str(zip_output),
                "shareability_required": True,
                "shareability_gate": "labeler_links_safe_to_share",
                "shareability_contract_schema": "palette.web_labeling_handoff_shareability.v1",
                "shareability_decision_source": "inspect_handoff_package",
                "shareability_contract_field": "shareability_contract",
                "shareability_nested_contract_field": "shareability.contract",
                "shareability_compact_contract_schema": (
                    "palette.web_labeling_handoff_shareability_contract.v1"
                ),
            "shareability_compact_contract_fields": list(_shareability_compact_contract_fields()),
            "shareability_compact_contract_field_count": len(_shareability_compact_contract_fields()),
            "shareability_compact_contract_source_fields": dict(_shareability_compact_contract_source_fields()),
            "shareability_compact_contract_source_field_count": len(_shareability_compact_contract_source_fields()),
            **_shareability_compact_contract_safe_to_share_target(),
            **_shareability_compact_contract_next_action_target(),
            **_shareability_external_launch_evidence_gap_target(),
                **_personalized_launch_readiness_target(),
                "shareability_safe_to_share_requires": list(shareability_safe_to_share_requires),
                "shareability_repair_commands_field": "repair_commands",
            "shareability_repair_command_detail_fields": list(shareability_repair_command_detail_fields),
            "shareability_repair_command_detail_fields_by_id": dict(shareability_repair_command_detail_fields_by_id),
            "shareability_repair_command_contracts": dict(shareability_repair_command_contracts),
                "safe_share_gate": _safe_share_gate_policy(),
                **_safe_share_gate_flat_fields(),
                "shareability_safe_share_next_action_fields": [
                    "safe_share_launch_blocking_next_actions",
                    "safe_share_launch_blocking_next_action_count",
                    "safe_share_next_action_summary",
                ],
                "shareability_safe_share_next_action_detail_fields": list(
                    safe_share_next_action_detail_fields
                ),
                "shareability_safe_share_next_action_command_fields": list(
                    safe_share_next_action_command_fields
                ),
                **ready_row_draft_inspection_fields,
                **response_security_inspection_fields,
                **launch_evidence_collection_inspection_fields,
                **_launch_evidence_execution_checklist_inspection_target(),
                "shareability_identity_personal_queue_evidence_required": True,
                "shareability_identity_personal_queue_evidence_field": (
                    "shareability.identity_personal_queue_evidence"
                ),
                "shareability_identity_personal_queue_evidence_status_values": list(
                    IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES
                ),
                "shareability_operator_validation_gate_status_values": list(
                    OPERATOR_VALIDATION_GATE_STATUS_VALUES
                ),
                "shareability_operator_validation_gate_ids": list(
                    DEFAULT_OPERATOR_VALIDATION_GATE_IDS
                ),
                "shareability_operator_validation_gate_flat_field_suffixes": [
                    "status",
                    "pending",
                    "missing_evidence",
                    "needs_review",
                    "passed",
                ],
                "shareability_identity_personal_queue_evidence_top_level_fields": [
                    "identity_personal_queue_evidence_status",
                    "identity_personal_queue_evidence_ready_count",
                    "identity_personal_queue_evidence_missing_count",
                    "identity_personal_queue_evidence_ready_users",
                    "identity_personal_queue_evidence_missing_users",
                    "identity_personal_queue_evidence_missing_fields_by_user",
                    "identity_all_users_have_personal_queue_evidence",
                ],
                "command": "scripts/py -m fisheye.utils.labeling_work "
                f"--store {store_path} inspect-handoff --path {zip_output} --require-shareable",
            }
        )
    payload = {
        "schema": "palette.web_labeling_inspection_targets.v1",
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "store_path": str(store_path),
        "shareability_required": True,
        "shareability_gate": "labeler_links_safe_to_share",
        "shareability_contract_schema": "palette.web_labeling_handoff_shareability.v1",
        "shareability_decision_source": "inspect_handoff_package",
        "shareability_contract_field": "shareability_contract",
        "shareability_nested_contract_field": "shareability.contract",
        "shareability_compact_contract_schema": (
            "palette.web_labeling_handoff_shareability_contract.v1"
        ),
            "shareability_compact_contract_fields": list(_shareability_compact_contract_fields()),
            "shareability_compact_contract_field_count": len(_shareability_compact_contract_fields()),
            "shareability_compact_contract_source_fields": dict(_shareability_compact_contract_source_fields()),
            "shareability_compact_contract_source_field_count": len(_shareability_compact_contract_source_fields()),
            **_shareability_compact_contract_safe_to_share_target(),
            **_shareability_compact_contract_next_action_target(),
            **_shareability_external_launch_evidence_gap_target(),
            **_personalized_launch_readiness_target(),
        "shareability_safe_to_share_requires": list(shareability_safe_to_share_requires),
        "shareability_repair_commands_field": "repair_commands",
            "shareability_repair_command_detail_fields": list(shareability_repair_command_detail_fields),
            "shareability_repair_command_detail_fields_by_id": dict(shareability_repair_command_detail_fields_by_id),
            "shareability_repair_command_contracts": dict(shareability_repair_command_contracts),
        "safe_share_gate": _safe_share_gate_policy(),
        **_safe_share_gate_flat_fields(),
        "shareability_safe_share_next_action_fields": [
            "safe_share_launch_blocking_next_actions",
            "safe_share_launch_blocking_next_action_count",
            "safe_share_next_action_summary",
        ],
        "shareability_safe_share_next_action_detail_fields": list(
            safe_share_next_action_detail_fields
        ),
        "shareability_safe_share_next_action_command_fields": list(
            safe_share_next_action_command_fields
        ),
        **ready_row_draft_inspection_fields,
        **response_security_inspection_fields,
        **launch_evidence_collection_inspection_fields,
        **_launch_evidence_execution_checklist_inspection_target(),
        "shareability_identity_personal_queue_evidence_required": True,
        "shareability_identity_personal_queue_evidence_field": (
            "shareability.identity_personal_queue_evidence"
        ),
        "shareability_identity_personal_queue_evidence_status_values": list(
            IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES
        ),
        "shareability_operator_validation_gate_status_values": list(
            OPERATOR_VALIDATION_GATE_STATUS_VALUES
        ),
        "shareability_operator_validation_gate_ids": list(
            DEFAULT_OPERATOR_VALIDATION_GATE_IDS
        ),
        "shareability_operator_validation_gate_flat_field_suffixes": [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ],
        "shareability_identity_personal_queue_evidence_top_level_fields": [
            "identity_personal_queue_evidence_status",
            "identity_personal_queue_evidence_ready_count",
            "identity_personal_queue_evidence_missing_count",
            "identity_personal_queue_evidence_ready_users",
            "identity_personal_queue_evidence_missing_users",
            "identity_personal_queue_evidence_missing_fields_by_user",
            "identity_all_users_have_personal_queue_evidence",
        ],
        "targets": targets,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inspection_failure_actions_impl(
    failure_reasons: list[str],
    *,
    validation_checklist: Mapping[str, object],
) -> list[str]:
    action_by_reason = {
        "checksum_failed": "Regenerate the package or restore missing/modified files from the original bundle before re-sharing.",
        "dataset_queue_blocks_labeler_start": "Resolve blocked dataset queue states before re-sharing; generate or reopen work if labeling should continue, or treat completed assignments as finished.",
        "expired": "Regenerate the handoff package because the current links have expired.",
        "handoff_not_ready": "Do not share not-ready labeler handoffs; inspect per-handoff sendability reasons and repair the assignment or link state.",
        "preferred_personal_queue_mismatch": "Regenerate the handoff package so every labeler Start here link uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of canonical /datasets fallback.",
        "browser_mutation_target_contract_mismatch": "Regenerate or repair the handoff package so browser label mutations target server-owned task-scoped training Zarrs and handoff/intermediate CSV artifacts are metadata-only non-write targets.",
        "direct_browser_start_contract_mismatch": "Regenerate or repair the handoff package so direct browser Start/Open uses POST-only same-origin task-open requests with expected-user server rechecks, denied-start authorization contracts, task-scoped training-Zarr targets, and metadata-only non-write CSV/handoff artifacts.",
        "single_owner_package_contract_mismatch": "Regenerate or repair the handoff package from the current assignment store so each recording appears under exactly one active labeler before links are shared.",
        "labeler_route_authorization_runtime_checklist_mismatch": "Regenerate or repair the handoff package so labeler-route authorization runtime checklist evidence is present/ready, single-owner store proof is ready, assignment integrity is OK, duplicate active owner count is zero, browser work targets server-resolved training Zarrs, and intermediate CSV mutation is rejected.",
        "handoff_store_checks_failed": "Resolve per-user handoff store-check failures, then regenerate the handoff package.",
        "handoffs_failed": "Resolve failed per-user handoffs, then regenerate the launch bundle.",
        "implementation_status_artifact_incomplete": "Regenerate the package so validation-checklist.json includes the complete implementation_status_artifact contract and required fields, then rerun inspect-handoff before sharing labeler links.",
        "reassignment_session_safety_failed": "Close stale previous-owner sessions or re-run assignment through assign_recording_with_session_closure, then regenerate or refresh the handoff package before sharing.",
        "assignment_freshness_incomplete": "Regenerate the handoff package because current assignments include active work omitted from the archived package.",
        "assignment_freshness_mismatch": "Regenerate the handoff package because current assignment ownership no longer matches the archived package.",
        "assignment_freshness_unverified": "Regenerate the handoff package so assignment freshness can be checked against the current store.",
        "needs_review": "Inspect per-handoff status and repair notes before sharing this package.",
        "no_handoffs": "Regenerate the package because no labeler handoff manifests were found.",
        "operator_evidence_commands_missing": "Regenerate the launch bundle so operator-evidence-commands.txt is present before operator validation.",
        "operator_evidence_commands_boundary_missing": "Regenerate the launch bundle so operator-evidence-commands.txt starts with the operator-only boundary warning before operator validation.",
        "operator_evidence_commands_invalid": "Regenerate the launch bundle so operator-evidence-commands.txt is valid and starts with the operator-only boundary warning before operator validation.",
        "launch_evidence_execution_checklist_missing": "Regenerate the launch bundle so launch-evidence-execution-checklist.txt is present before operator validation.",
        "launch_evidence_execution_checklist_incomplete": "Regenerate the launch bundle so launch-evidence-execution-checklist.txt contains the complete operator-only launch evidence runbook.",
        "launch_evidence_execution_checklist_invalid": "Regenerate the launch bundle so launch-evidence-execution-checklist.txt is valid and contains the operator-only launch evidence runbook.",
        "readiness_failed": "Resolve batch-readiness issues, then regenerate the launch bundle.",
        "unknown_expiration": "Regenerate the handoff package so link expiration metadata is present.",
        "validation_checklist_invalid": "Repair validation-checklist.json so it is valid JSON, then rerun inspect-handoff.",
        "validation_checklist_missing": "Regenerate the package or copy validation-checklist.json into the top-level package before re-sharing.",
        "validation_checklist_needs_review": "Resolve validation checklist gates marked needs_review before sharing this package.",
        "validation_evidence_pending": "Record evidence for required pending validation gates with update-validation-checklist, then rerun inspect-handoff.",
        "validation_evidence_template_unapproved": "Approve or repair operator evidence template files, run apply-operator-evidence-templates to mark approved gates passed and refresh handoff readiness, then refresh checksums and rerun inspect-handoff.",
        "validation_log_missing": "Regenerate the package or copy validation-log-template.md into the top-level package before re-sharing.",
    }
    actions: list[str] = []
    for reason in failure_reasons:
        action = action_by_reason.get(reason)
        if action and action not in actions:
            actions.append(action)
    required_pending_gate_ids = (
        validation_checklist.get("required_pending_gate_ids")
        if isinstance(validation_checklist.get("required_pending_gate_ids"), list)
        else []
    )
    pending_gate_text = ", ".join(str(gate_id) for gate_id in required_pending_gate_ids if str(gate_id))
    if pending_gate_text:
        actions.append(f"Pending required validation gates: {pending_gate_text}.")
    needs_review_gate_ids = (
        validation_checklist.get("needs_review_gate_ids")
        if isinstance(validation_checklist.get("needs_review_gate_ids"), list)
        else []
    )
    needs_review_gate_text = ", ".join(str(gate_id) for gate_id in needs_review_gate_ids if str(gate_id))
    if needs_review_gate_text:
        actions.append(f"Validation checklist gates needing review: {needs_review_gate_text}.")
    unapproved_template_gate_ids = (
        validation_checklist.get("passed_gates_with_unapproved_evidence_templates")
        if isinstance(validation_checklist.get("passed_gates_with_unapproved_evidence_templates"), list)
        else []
    )
    unapproved_template_text = ", ".join(
        str(gate_id) for gate_id in unapproved_template_gate_ids if str(gate_id)
    )
    if unapproved_template_text:
        actions.append(f"Passed gates with unapproved evidence templates: {unapproved_template_text}.")
    implementation_status_missing_fields = (
        validation_checklist.get("implementation_status_artifact_missing_fields")
        if isinstance(
            validation_checklist.get("implementation_status_artifact_missing_fields"),
            list,
        )
        else []
    )
    implementation_status_missing_field_text = ", ".join(
        str(field) for field in implementation_status_missing_fields if str(field)
    )
    if implementation_status_missing_field_text:
        actions.append(
            "Implementation status artifact missing required fields: "
            f"{implementation_status_missing_field_text}. Regenerate the package with the "
            "current implementation_status_artifact contract before sharing labeler links."
        )
    template_statuses = (
        validation_checklist.get("operator_evidence_template_statuses")
        if isinstance(validation_checklist.get("operator_evidence_template_statuses"), Mapping)
        else {}
    )
    identity_status = (
        template_statuses.get("identity_probe_verification")
        if isinstance(template_statuses.get("identity_probe_verification"), Mapping)
        else {}
    )
    missing_identity_users = (
        identity_status.get("users_missing_required_fields")
        if isinstance(identity_status.get("users_missing_required_fields"), list)
        else []
    )
    for user_status in missing_identity_users:
        if not isinstance(user_status, Mapping):
            continue
        missing_fields = [
            str(field)
            for field in (
                user_status.get("missing_fields")
                if isinstance(user_status.get("missing_fields"), list)
                else []
            )
            if str(field)
        ]
        if not missing_fields:
            continue
        expected_user = str(user_status.get("expected_user") or "expected_labeler")
        resolved_user = str(user_status.get("resolved_user") or "")
        resolved_text = f" resolved_user={resolved_user}." if resolved_user else ""
        actions.append(
            "Identity-source evidence incomplete for "
            f"{expected_user}: missing {', '.join(missing_fields)}.{resolved_text} "
            "Open /identity?expected_user=<user> in the deployed authenticated browser context, "
            "confirm the resolved Palette user exactly matches the expected assignee, and confirm "
            "preferred_labeler_entry_url plus personalized_labeler_entry_url both equal the guarded "
            "/my-datasets?expected_user=<user> personal queue URL. Then record "
            "record-identity-source-evidence --expected-user "
            f"{expected_user} --resolved-user RESOLVED_USER --operator OPERATOR."
        )
    response_security_status = (
        template_statuses.get("browser_response_security_headers")
        if isinstance(template_statuses.get("browser_response_security_headers"), Mapping)
        else {}
    )
    missing_response_headers = [
        str(header)
        for header in (
            response_security_status.get("missing_headers")
            if isinstance(response_security_status.get("missing_headers"), list)
            else []
        )
        if str(header)
    ]
    missing_response_checks = [
        str(check)
        for check in (
            response_security_status.get("missing_checks")
            if isinstance(response_security_status.get("missing_checks"), list)
            else []
        )
        if str(check)
    ]
    response_security_approval_missing = bool(
        response_security_status.get("operator_approval_missing")
    )
    if missing_response_headers or missing_response_checks or response_security_approval_missing:
        header_args = " ".join(
            f"--header '{header}=VALUE'" for header in missing_response_headers
        )
        header_args_text = header_args or "--header 'HEADER=VALUE'"
        actions.append(
            "Browser response-security evidence incomplete: "
            f"missing headers {', '.join(missing_response_headers) or 'none'}; "
            f"missing checks {', '.join(missing_response_checks) or 'none'}; "
            f"operator approval missing={response_security_approval_missing}. "
            "Capture deployed labeler-facing response headers from /datasets?expected_user=<user> "
            "or /api/me/tasks?expected_user=<user>, then record "
            "record-browser-response-security-evidence "
            f"{header_args_text} --operator OPERATOR "
            "--capture-url URL --authenticated-test-user USER."
        )
    backup_status = (
        template_statuses.get("mutable_zarr_backup_confirmation")
        if isinstance(template_statuses.get("mutable_zarr_backup_confirmation"), Mapping)
        else {}
    )
    missing_backup_targets = (
        backup_status.get("targets_missing_required_fields")
        if isinstance(backup_status.get("targets_missing_required_fields"), list)
        else []
    )
    for target_status in missing_backup_targets:
        if not isinstance(target_status, Mapping):
            continue
        missing_fields = [
            str(field)
            for field in (
                target_status.get("missing_fields")
                if isinstance(target_status.get("missing_fields"), list)
                else []
            )
            if str(field)
        ]
        if not missing_fields:
            continue
        target_index = target_status.get("target_index")
        target_label = f"target_index={target_index}" if target_index is not None else "target"
        role = str(target_status.get("role") or "")
        role_text = f" role={role}." if role else ""
        actions.append(
            "Mutable-Zarr backup evidence incomplete for "
            f"{target_label}: missing {', '.join(missing_fields)}.{role_text} "
            "Run execute-zarr-backup-plan for the backup plan, then record "
            "record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json "
            "--execution-manifest BACKUP_EXECUTION_MANIFEST --target-index "
            f"{target_index if target_index is not None else 'TARGET_INDEX'} "
            "--restore-test-result RESTORE_TEST_RESULT --operator OPERATOR."
        )
    browser_smoke_status = (
        template_statuses.get("browser_smoke")
        if isinstance(template_statuses.get("browser_smoke"), Mapping)
        else {}
    )
    browser_smoke_route_contract_action = str(
        browser_smoke_status.get("personalized_route_smoke_contract_operator_action")
        or ""
    ).strip()
    if browser_smoke_route_contract_action:
        missing_contract_fields = [
            str(field)
            for field in (
                browser_smoke_status.get("personalized_route_smoke_contract_missing_fields")
                if isinstance(
                    browser_smoke_status.get("personalized_route_smoke_contract_missing_fields"),
                    list,
                )
                else []
            )
            if str(field)
        ]
        actions.append(
            "Browser smoke personalized route contract is stale"
            + (f": missing {', '.join(missing_contract_fields)}. " if missing_contract_fields else ". ")
            + browser_smoke_route_contract_action
            + " Then re-run record-browser-smoke-evidence with --personalized-dataset-queue-verified, "
            "--preferred-labeler-entry-url-matches-personal-dataset-queue, "
            "--personalized-labeler-entry-url-matches-personal-dataset-queue, and "
            "--personalized-work-dashboard-verified."
        )
    missing_browser_smoke_users = (
        browser_smoke_status.get("users_missing_required_fields")
        if isinstance(browser_smoke_status.get("users_missing_required_fields"), list)
        else []
    )
    for user_status in missing_browser_smoke_users:
        if not isinstance(user_status, Mapping):
            continue
        missing_fields = [
            str(field)
            for field in (
                user_status.get("missing_fields")
                if isinstance(user_status.get("missing_fields"), list)
                else []
            )
            if str(field)
        ]
        if not missing_fields:
            continue
        expected_user = str(user_status.get("expected_user") or "representative_labeler")
        actions.append(
            "Browser smoke evidence incomplete for "
            f"{expected_user}: missing {', '.join(missing_fields)}. "
            "Re-run or update record-browser-smoke-evidence after confirming browser-only/no-local-install "
            "runtime, personalized dataset queue entry, personalized work dashboard fallback, assigned-only visibility, expected-user mismatch rejection, support redaction, "
            "completion/read-only behavior, stale-tab rejection, and operator reopen before recording "
            "--browser-only-runtime-verified, "
            "--no-local-palette-install-verified, "
            "--no-local-crimson-install-verified, "
            "--no-local-conda-or-project-dependencies-verified, "
            "--personalized-dataset-queue-verified, "
            "--preferred-labeler-entry-url-matches-personal-dataset-queue, "
            "--personalized-labeler-entry-url-matches-personal-dataset-queue, "
            "--personalized-work-dashboard-verified, "
            "--labeler-sees-only-assigned-work, "
            "--support-text-redacted, "
            "--expected-user-mismatch-rejected, "
            "--task-opened, "
            "--induced-failure-support-detail-redacted, "
            "--completion-verified, "
            "--completed-task-read-only-verified, "
            "--stale-tab-save-rejected, and --operator-reopen-verified."
        )
    disposable_status = (
        template_statuses.get("disposable_zarr_mutation_smoke")
        if isinstance(template_statuses.get("disposable_zarr_mutation_smoke"), Mapping)
        else {}
    )
    missing_workflows = (
        disposable_status.get("workflows_missing_required_fields")
        if isinstance(disposable_status.get("workflows_missing_required_fields"), list)
        else []
    )
    for workflow_status in missing_workflows:
        if not isinstance(workflow_status, Mapping):
            continue
        missing_fields = [
            str(field)
            for field in (
                workflow_status.get("missing_fields")
                if isinstance(workflow_status.get("missing_fields"), list)
                else []
            )
            if str(field)
        ]
        if not missing_fields:
            continue
        workflow_kind = str(workflow_status.get("workflow_kind") or "unknown_workflow")
        actions.append(
            "Disposable-Zarr mutation smoke evidence incomplete for "
            f"{workflow_kind}: missing {', '.join(missing_fields)}. "
            "Verify labeler-reported event IDs through /admin Audit event lookup, archive lookup reports, "
            "confirm task-scoped training-Zarr writes, browser_label_write_target=training_zarr, "
            "no direct browser Zarr authority, metadata-only handoff/CSV artifacts, no browser "
            "handoff/intermediate CSV writes, and rejected browser-supplied CSV/Zarr target selectors, then record bad-mutation recovery mode/report before recording "
            "--task-scoped-training-zarr-write-verified, "
            "--browser-no-direct-zarr-write-authority-verified, "
            "--handoff-artifacts-metadata-only-verified, "
            "--browser-no-csv-or-handoff-write-verified, "
            "--client-target-selector-rejection-verified, "
            "--operator-event-lookup-verified, and --bad-mutation-recovery-verified."
        )
    return actions


def _inspection_operator_repair_commands_impl(
    failure_actions: Sequence[str],
    *,
    validation_checklist: Mapping[str, object] | None = None,
    launch_evidence_execution_checklist_summary: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    action_lines = [str(action) for action in failure_actions]
    action_text = "\n".join(action_lines)
    validation_checklist_source = (
        validation_checklist if isinstance(validation_checklist, Mapping) else {}
    )
    launch_evidence_execution_checklist_summary_source = (
        launch_evidence_execution_checklist_summary
        if isinstance(launch_evidence_execution_checklist_summary, Mapping)
        else {}
    )
    template_statuses = (
        validation_checklist_source.get("operator_evidence_template_statuses")
        if isinstance(
            validation_checklist_source.get("operator_evidence_template_statuses"),
            Mapping,
        )
        else {}
    )
    identity_status = (
        template_statuses.get("identity_probe_verification")
        if isinstance(template_statuses.get("identity_probe_verification"), Mapping)
        else {}
    )
    identity_personal_queue_evidence_missing_count = str(
        identity_status.get("personal_queue_evidence_missing_count") or ""
    ).strip()
    identity_personal_queue_evidence_incomplete = bool(
        identity_status.get("personal_queue_evidence_missing_users")
    ) or identity_personal_queue_evidence_missing_count not in {"", "0", "0.0", "false", "False"}
    pending_validation_gate_ids: list[str] = []
    unapproved_template_gate_ids: list[str] = []
    for action in action_lines:
        if not action.startswith("Pending required validation gates:"):
            continue
        pending_text = action.split(":", 1)[1].strip().rstrip(".")
        pending_validation_gate_ids = [
            gate_id.strip()
            for gate_id in pending_text.split(",")
            if gate_id.strip()
        ]
        break
    for action in action_lines:
        if not action.startswith("Passed gates with unapproved evidence templates:"):
            continue
        unapproved_text = action.split(":", 1)[1].strip().rstrip(".")
        unapproved_template_gate_ids = [
            gate_id.strip()
            for gate_id in unapproved_text.split(",")
            if gate_id.strip()
        ]
        break
    commands: list[dict[str, object]] = []
    commands_by_id: dict[str, dict[str, object]] = {}

    def add(
        command_id: str,
        label: str,
        command: str,
        *,
        category: str = "operator_evidence",
        gate_ids: Sequence[str] = (),
        reason_ids: Sequence[str] = (),
        requires_checksum_refresh_after_run: bool = True,
    ) -> None:
        clean_command_id = str(command_id or "")
        if not clean_command_id:
            return
        gate_id_values = [str(gate_id) for gate_id in gate_ids if str(gate_id)]
        reason_id_values = [str(reason_id) for reason_id in reason_ids if str(reason_id)]
        if clean_command_id in commands_by_id:
            existing = commands_by_id[clean_command_id]
            existing_gate_ids = (
                existing.get("gate_ids") if isinstance(existing.get("gate_ids"), list) else []
            )
            existing_reason_ids = (
                existing.get("reason_ids") if isinstance(existing.get("reason_ids"), list) else []
            )
            existing["gate_ids"] = list(
                dict.fromkeys([*existing_gate_ids, *gate_id_values])
            )
            existing["reason_ids"] = list(
                dict.fromkeys([*existing_reason_ids, *reason_id_values])
            )
            existing["requires_checksum_refresh_after_run"] = bool(
                existing.get("requires_checksum_refresh_after_run")
                or requires_checksum_refresh_after_run
            )
            return
        row: dict[str, object] = {
            "id": clean_command_id,
            "category": category,
            "label": label,
            "command": command,
            "gate_ids": gate_id_values,
            "reason_ids": reason_id_values,
            "requires_checksum_refresh_after_run": requires_checksum_refresh_after_run,
        }
        commands_by_id[clean_command_id] = row
        commands.append(row)

    validation_command_templates = (
        validation_checklist_source.get("operator_validation_command_templates")
        if isinstance(
            validation_checklist_source.get("operator_validation_command_templates"),
            Mapping,
        )
        else {}
    )
    validation_command_gate_ids = (
        validation_command_templates.get("gate_ids")
        if isinstance(validation_command_templates.get("gate_ids"), list)
        else []
    )
    locally_generated_validation_commands = _operator_validation_command_templates(
        validation_command_gate_ids
    )
    for command_template in locally_generated_validation_commands.get("commands", []):
        if not isinstance(command_template, Mapping):
            continue
        command_id = str(command_template.get("id") or "")
        command_reason_ids: tuple[str, ...] = ()
        if command_id == "record_identity_source_evidence" and identity_personal_queue_evidence_incomplete:
            command_reason_ids = ("personal_dataset_queue_link_evidence_incomplete",)
        add(
            command_id,
            str(command_template.get("label") or ""),
            str(command_template.get("command") or ""),
            category=str(command_template.get("category") or "operator_evidence"),
            gate_ids=(
                command_template.get("gate_ids")
                if isinstance(command_template.get("gate_ids"), list)
                else []
            ),
            reason_ids=command_reason_ids,
            requires_checksum_refresh_after_run=bool(
                command_template.get("requires_checksum_refresh_after_run", True)
            ),
        )

    if "record-zarr-backup-evidence" in action_text:
        add(
            "record_zarr_backup_evidence",
            "Record mutable-Zarr backup evidence",
            "record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json "
            "--execution-manifest BACKUP_EXECUTION_MANIFEST --target-index TARGET_INDEX "
            "--restore-test-result RESTORE_TEST_RESULT --operator OPERATOR",
            gate_ids=("mutable_zarr_backup_confirmation",),
        )
    if "record-browser-response-security-evidence" in action_text:
        add(
            "record_browser_response_security_evidence",
            "Record deployed browser response-security evidence",
            "record-browser-response-security-evidence --evidence "
            "browser-response-security-evidence-template.json --header 'HEADER=VALUE' "
            "--operator OPERATOR --capture-url DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER "
            "--authenticated-test-user SAME_USER_AS_EXPECTED_USER",
            gate_ids=("browser_response_security_headers",),
        )
    if "record-identity-source-evidence" in action_text:
        identity_reason_ids = (
            ("personal_dataset_queue_link_evidence_incomplete",)
            if identity_personal_queue_evidence_incomplete
            or "preferred_labeler_entry_url" in action_text
            or "personalized_labeler_entry_url" in action_text
            or "/my-datasets?expected_user=<user>" in action_text
            else ()
        )
        add(
            "record_identity_source_evidence",
            "Record deployed identity-source and personal queue evidence",
            "record-identity-source-evidence --evidence identity-source-evidence-template.json "
            "--expected-user USER --resolved-user RESOLVED_USER --operator OPERATOR "
            "--authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED",
            gate_ids=("identity_probe_verification",),
            reason_ids=identity_reason_ids,
        )
    if (
        "record-browser-smoke-evidence" in action_text
        or "Browser smoke personalized route contract is stale" in action_text
    ):
        add(
            "record_browser_smoke_evidence",
            "Record representative browser smoke evidence",
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
            "--completed-task-read-only-verified --stale-tab-save-rejected --operator-reopen-verified",
            gate_ids=("browser_smoke",),
            reason_ids=(
                ("browser_smoke_personalized_route_contract_stale",)
                if "Browser smoke personalized route contract is stale" in action_text
                else ()
            ),
        )
    if "Disposable-Zarr mutation smoke evidence incomplete" in action_text:
        add(
            "record_disposable_zarr_mutation_smoke_evidence",
            "Record disposable-Zarr mutation smoke evidence",
            "record-disposable-zarr-mutation-smoke-evidence --evidence "
            "disposable-zarr-mutation-smoke-evidence-template.json --workflow-kind WORKFLOW_KIND "
            "--mutation-event-id EVENT_ID --event-lookup-report EVENT_ID-lookup.json "
            "--operator OPERATOR --labeler-user LABELER_USER "
            "--task-scoped-training-zarr-write-verified "
            "--browser-no-direct-zarr-write-authority-verified "
            "--handoff-artifacts-metadata-only-verified --browser-no-csv-or-handoff-write-verified "
            "--client-target-selector-rejection-verified "
            "--operator-event-lookup-verified --bad-mutation-recovery-verified "
            "--bad-mutation-recovery-mode RECOVERY_MODE --bad-mutation-recovery-report RECOVERY_REPORT",
            gate_ids=("disposable_zarr_mutation_smoke",),
        )
    if (
        "repair-reassignment-sessions" in action_text
        or "assign_recording_with_session_closure" in action_text
    ):
        add(
            "repair_reassignment_sessions",
            "Close stale previous-owner sessions",
            "repair-reassignment-sessions --user OPERATOR --recording-id RECORDING_ID",
            category="session_repair",
            requires_checksum_refresh_after_run=False,
        )
    if (
        "preferred queue URL instead of canonical /datasets fallback" in action_text
        or "preferred_personal_queue_mismatch" in action_text
    ):
        add(
            "regenerate_handoffs_with_personal_dataset_queue",
            "Regenerate handoffs with preferred personalized dataset queue links",
            "export-user-handoffs --store STORE_PATH --output OUTPUT_DIR "
            "--base-url DEPLOYED_LABELING_URL --link-secret LINK_SECRET",
            category="handoff_regeneration",
            reason_ids=("preferred_personal_queue_mismatch",),
            requires_checksum_refresh_after_run=False,
        )
    if (
        "browser label mutations target server-owned task-scoped training Zarrs" in action_text
        or "browser_mutation_target_contract_mismatch" in action_text
    ):
        add(
            "regenerate_handoffs_with_browser_mutation_target_contract",
            "Regenerate handoffs with browser mutation target contract",
            "export-user-handoffs --store STORE_PATH --output OUTPUT_DIR "
            "--base-url DEPLOYED_LABELING_URL --link-secret LINK_SECRET",
            category="handoff_regeneration",
            reason_ids=("browser_mutation_target_contract_mismatch",),
            requires_checksum_refresh_after_run=False,
        )
    if (
        "direct browser Start/Open uses POST-only same-origin task-open requests" in action_text
        or "direct_browser_start_contract_mismatch" in action_text
    ):
        add(
            "regenerate_handoffs_with_direct_browser_start_contract",
            "Regenerate handoffs with direct browser Start/Open contract",
            "export-user-handoffs --store STORE_PATH --output OUTPUT_DIR "
            "--base-url DEPLOYED_LABELING_URL --link-secret LINK_SECRET",
            category="handoff_regeneration",
            reason_ids=("direct_browser_start_contract_mismatch",),
            requires_checksum_refresh_after_run=False,
        )
    if (
        "each recording appears under exactly one active labeler" in action_text
        or "single_owner_package_contract_mismatch" in action_text
    ):
        add(
            "regenerate_handoffs_with_single_owner_package_contract",
            "Regenerate handoffs with one active owner per recording",
            "export-user-handoffs --store STORE_PATH --output OUTPUT_DIR "
            "--base-url DEPLOYED_LABELING_URL --link-secret LINK_SECRET",
            category="handoff_regeneration",
            reason_ids=("single_owner_package_contract_mismatch",),
            requires_checksum_refresh_after_run=False,
        )
    if (
        "labeler-route authorization runtime checklist evidence is present/ready"
        in action_text
        or "labeler_route_authorization_runtime_checklist_mismatch" in action_text
    ):
        add(
            "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist",
            "Regenerate handoffs with labeler-route runtime checklist proof",
            "export-user-handoffs --store STORE_PATH --output OUTPUT_DIR "
            "--base-url DEPLOYED_LABELING_URL --link-secret LINK_SECRET",
            category="handoff_regeneration",
            reason_ids=("labeler_route_authorization_runtime_checklist_mismatch",),
            requires_checksum_refresh_after_run=False,
        )
    if (
        "complete implementation_status_artifact contract" in action_text
        or "implementation_status_artifact_incomplete" in action_text
    ):
        implementation_status_missing_fields = [
            str(field)
            for field in (
                validation_checklist_source.get(
                    "implementation_status_artifact_missing_fields"
                )
                if isinstance(
                    validation_checklist_source.get(
                        "implementation_status_artifact_missing_fields"
                    ),
                    list,
                )
                else []
            )
            if str(field)
        ]
        add(
            "regenerate_package_with_implementation_status_artifact",
            "Regenerate package with implementation-status artifact contract",
            "regenerate the handoff or launch bundle from the current assignment store "
            "with the current Palette labeling_work command, then rerun inspect-handoff "
            "--require-shareable before sharing labeler links",
            category="handoff_regeneration",
            reason_ids=("implementation_status_artifact_incomplete",),
            requires_checksum_refresh_after_run=False,
        )
        implementation_status_command = commands_by_id.get(
            "regenerate_package_with_implementation_status_artifact"
        )
        if implementation_status_command is not None:
            implementation_status_command["missing_fields"] = implementation_status_missing_fields
            implementation_status_command["missing_field_count"] = len(
                implementation_status_missing_fields
            )
            implementation_status_command["repair_mode"] = "regenerate_package"
            implementation_status_command["artifact_contract"] = (
                "implementation_status_artifact"
            )
            implementation_status_command["safe_share_blocker"] = (
                "implementation_status_checklist_artifact_complete_required_value_mismatch"
            )
    if (
        "launch-evidence-execution-checklist.txt" in action_text
        or "launch_evidence_execution_checklist" in action_text
    ):
        launch_checklist_missing_phrases = [
            str(phrase)
            for phrase in (
                launch_evidence_execution_checklist_summary_source.get(
                    "checklist_missing_phrases"
                )
                if isinstance(
                    launch_evidence_execution_checklist_summary_source.get(
                        "checklist_missing_phrases"
                    ),
                    list,
                )
                else []
            )
            if str(phrase)
        ]
        launch_checklist_blocker = str(
            launch_evidence_execution_checklist_summary_source.get("blocking_reason_id")
            or "launch_evidence_execution_checklist_incomplete"
        )
        add(
            "regenerate_package_with_launch_evidence_execution_checklist",
            "Regenerate package with launch-evidence execution checklist",
            "regenerate the launch bundle from the current assignment store with the "
            "current Palette labeling_work command so launch-evidence-execution-checklist.txt "
            "is present and complete, then rerun inspect-handoff --require-shareable "
            "before sharing labeler links",
            category="handoff_regeneration",
            reason_ids=(launch_checklist_blocker,),
            requires_checksum_refresh_after_run=False,
        )
        launch_checklist_command = commands_by_id.get(
            "regenerate_package_with_launch_evidence_execution_checklist"
        )
        if launch_checklist_command is not None:
            launch_checklist_command["required_file"] = (
                "launch-evidence-execution-checklist.txt"
            )
            launch_checklist_command["required_phrase_contract"] = (
                "shareability_launch_evidence_execution_checklist_required_phrases"
            )
            launch_checklist_command["required_phrases"] = list(
                _launch_evidence_execution_checklist_status("")[
                    "checklist_required_phrases"
                ]
            )
            launch_checklist_command["missing_phrases"] = launch_checklist_missing_phrases
            launch_checklist_command["missing_phrase_count"] = len(
                launch_checklist_missing_phrases
            )
            launch_checklist_command["repair_mode"] = "regenerate_package"
            launch_checklist_command["artifact_contract"] = (
                "launch_evidence_execution_checklist"
            )
            launch_checklist_command["safe_share_blocker"] = launch_checklist_blocker
            launch_checklist_command["safe_share_blockers"] = [
                "launch_evidence_execution_checklist_missing",
                "launch_evidence_execution_checklist_incomplete",
                "launch_evidence_execution_checklist_invalid",
            ]
    if (
        "operator evidence template" in action_text
        or "record-zarr-backup-evidence" in action_text
        or "record-browser-response-security-evidence" in action_text
        or "record-identity-source-evidence" in action_text
        or "record-browser-smoke-evidence" in action_text
        or "Browser smoke personalized route contract is stale" in action_text
        or "Disposable-Zarr mutation smoke evidence incomplete" in action_text
    ):
        apply_gate_ids = unapproved_template_gate_ids
        if not apply_gate_ids:
            apply_gate_ids = [
                gate_id
                for gate_id, marker in (
                    ("mutable_zarr_backup_confirmation", "record-zarr-backup-evidence"),
                    ("browser_response_security_headers", "record-browser-response-security-evidence"),
                    ("identity_probe_verification", "record-identity-source-evidence"),
                    ("browser_smoke", "record-browser-smoke-evidence"),
                    ("browser_smoke", "Browser smoke personalized route contract is stale"),
                    (
                        "disposable_zarr_mutation_smoke",
                        "Disposable-Zarr mutation smoke evidence incomplete",
                    ),
                )
                if marker in action_text
            ]
        add(
            "apply_operator_evidence_templates",
            "Apply approved operator evidence templates to handoff readiness",
            "apply-operator-evidence-templates --path validation-checklist.json "
            "--operator OPERATOR --append-log validation-log-template.md",
            category="validation_checklist",
            gate_ids=tuple(apply_gate_ids or ["OPERATOR_EVIDENCE_GATE_ID"]),
        )
    if "update-validation-checklist" in action_text or "Pending required validation gates:" in action_text:
        add(
            "update_validation_checklist",
            "Record validation checklist gate evidence",
            "update-validation-checklist --path validation-checklist.json --gate GATE_ID "
            "--status passed --operator OPERATOR --append-log validation-log-template.md",
            category="validation_checklist",
            gate_ids=tuple(pending_validation_gate_ids or ["GATE_ID"]),
        )
    if commands:
        add(
            "refresh_handoff_checksums",
            "Refresh handoff checksums after evidence updates",
            "refresh-handoff-checksums --path PACKAGE_DIR --operator OPERATOR "
            "--reason 'operator evidence update'",
            category="checksum_refresh",
            requires_checksum_refresh_after_run=False,
        )
    return commands


def _inspection_labeler_entrypoint_summary_impl(
    index: Mapping[str, object] | None,
    handoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    source = index if isinstance(index, Mapping) else {}
    personalized_urls_by_user: dict[str, str] = {}
    personal_work_urls_by_user: dict[str, str] = {}
    canonical_dataset_queue_urls_by_user: dict[str, str] = {}
    users_with_handoffs: list[str] = []
    for handoff in handoffs:
        user = str(handoff.get("user") or "").strip()
        if not user:
            continue
        users_with_handoffs.append(user)
        personalized_url = str(
            handoff.get("personalized_labeler_entry_url")
            or handoff.get("expected_user_personal_dataset_queue_url")
            or ""
        ).strip()
        if personalized_url:
            personalized_urls_by_user[user] = personalized_url
        personal_work_url = str(handoff.get("expected_user_personal_work_url") or "").strip()
        if personal_work_url:
            personal_work_urls_by_user[user] = personal_work_url
        canonical_dataset_queue_url = str(handoff.get("expected_user_dataset_queue_url") or "").strip()
        if canonical_dataset_queue_url:
            canonical_dataset_queue_urls_by_user[user] = canonical_dataset_queue_url
    user_count = len(users_with_handoffs)
    personalized_entry_url_count = len(personalized_urls_by_user)
    personal_dataset_queue_page_path = str(
        source.get("personal_dataset_queue_page_path") or PERSONAL_DATASET_QUEUE_PATH
    )
    personal_work_page_path = str(source.get("personal_work_page_path") or PERSONAL_WORK_PATH)
    dataset_queue_page_path = str(source.get("dataset_queue_page_path") or DATASET_QUEUE_PATH)
    return {
        "schema": "palette.web_labeling_inspection_labeler_entrypoints.v1",
        "preferred_operator_link_kind": "personalized_dataset_queue",
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "canonical_fallback_entrypoint": "datasets_waiting_queue",
        "personal_dataset_queue_page_path": personal_dataset_queue_page_path,
        "personal_dataset_queue_url": str(source.get("personal_dataset_queue_url") or ""),
        "personal_work_page_path": personal_work_page_path,
        "personal_work_url": str(source.get("personal_work_url") or ""),
        "dataset_queue_page_path": dataset_queue_page_path,
        "dataset_queue_url": str(source.get("dataset_queue_url") or ""),
        "dashboard_path": str(source.get("dashboard_path") or DASHBOARD_PATH),
        "dashboard_url": str(source.get("dashboard_url") or ""),
        "user_count": user_count,
        "personalized_labeler_entry_url_count": personalized_entry_url_count,
        "all_handoffs_have_personalized_entry_url": (
            user_count > 0 and personalized_entry_url_count == user_count
        ),
        "personalized_labeler_entry_url_by_user": dict(sorted(personalized_urls_by_user.items())),
        "expected_user_personal_work_url_by_user": dict(sorted(personal_work_urls_by_user.items())),
        "canonical_dataset_queue_url_by_user": dict(
            sorted(canonical_dataset_queue_urls_by_user.items())
        ),
        "personalized_labeler_entry_urls": sorted(set(personalized_urls_by_user.values())),
        "canonical_dataset_queue_urls": sorted(set(canonical_dataset_queue_urls_by_user.values())),
    }


# Preserve original helper names inside this module so moved helpers can
# continue to call each other exactly as they did in web.py.
_single_owner_package_contract_summary = _single_owner_package_contract_summary_impl
_handoff_sendability_reasons = _handoff_sendability_reasons_impl
_handoff_sendability_actions = _handoff_sendability_actions_impl
_handoff_sendability_summary = _handoff_sendability_summary_impl
_count_handoff_sendability_reasons = _count_handoff_sendability_reasons_impl
_shareability_safe_to_share_requires = _shareability_safe_to_share_requires_impl
_shareability_labeler_route_authorization_runtime_checklist_gate_contract = _shareability_labeler_route_authorization_runtime_checklist_gate_contract_impl
_shareability_compact_contract_fields = _shareability_compact_contract_fields_impl
_shareability_compact_contract_source_fields = _shareability_compact_contract_source_fields_impl
_shareability_compact_contract_safe_to_share_target = _shareability_compact_contract_safe_to_share_target_impl
_shareability_compact_contract_next_action_target = _shareability_compact_contract_next_action_target_impl
_shareability_external_launch_evidence_gap_target = _shareability_external_launch_evidence_gap_target_impl
_shareability_repair_command_detail_fields = _shareability_repair_command_detail_fields_impl
_shareability_repair_command_detail_fields_by_id = _shareability_repair_command_detail_fields_by_id_impl
_shareability_repair_command_contracts = _shareability_repair_command_contracts_impl
_write_launch_bundle_inspection_targets = _write_launch_bundle_inspection_targets_impl
_inspection_failure_actions = _inspection_failure_actions_impl
_inspection_operator_repair_commands = _inspection_operator_repair_commands_impl
_inspection_labeler_entrypoint_summary = _inspection_labeler_entrypoint_summary_impl
