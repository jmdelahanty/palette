"""User-handoff inspection and status helpers for web labeling."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _require_inspection_dependencies(
    dependencies: Mapping[str, object],
    names: tuple[str, ...],
) -> dict[str, Any]:
    missing = [name for name in names if name not in dependencies]
    if missing:
        raise KeyError(f"missing user handoff inspection dependencies: {', '.join(missing)}")
    return {name: dependencies[name] for name in names}


def _handoff_status_from_manifest(
    manifest: dict[str, object],
    now: datetime,
    *,
    dependencies: Mapping[str, object],
) -> dict[str, object]:
    inspection_dependencies = _require_inspection_dependencies(dependencies, ('PERSONAL_DATASET_QUEUE_PATH', 'PERSONAL_WORK_PATH', '_browser_mutation_target_contract_compact_fields', '_browser_mutation_write_policy', '_browser_mutation_write_runtime_checklist', '_browser_response_security_policy', '_browser_signed_link_policy', '_browser_task_state_policy', '_browser_workflow_capabilities', '_dataset_queue_direct_start_policy', '_dataset_queue_direct_start_policy_fields', '_direct_browser_start_contract_compact_fields', '_handoff_assignment_ownership_fields', '_handoff_browser_mutation_write_fields', '_handoff_browser_response_security_fields', '_handoff_dataset_queue_blocks_labeler_start', '_handoff_dataset_queue_start_fields', '_handoff_dataset_queue_state', '_handoff_entry_artifact_fields', '_handoff_known_user_status_fields', '_handoff_labeler_route_authorization_fields', '_handoff_labeler_safety_fields', '_handoff_mutation_audit_fields', '_handoff_operator_recovery_fields', '_handoff_ready_to_send', '_handoff_sendability_actions', '_handoff_sendability_reasons', '_handoff_session_guard_fields', '_handoff_signed_link_policy_fields', '_handoff_task_state_policy_fields', '_handoff_zarr_backup_fields', '_labeler_route_authorization_policy', '_labeler_route_authorization_runtime_checklist', '_labeler_safety_policy', '_labeler_work_completion_fields', '_mutation_audit_policy', '_parse_handoff_utc', '_public_reassignment_session_safety_fields', '_reassignment_session_safety_flat_fields', '_recordings_without_open_tasks_actions', '_runtime_operator_validation_gate_cli_policy', '_runtime_operator_validation_gate_cli_policy_fields', '_session_guard_policy', '_zarr_backup_policy'))
    PERSONAL_DATASET_QUEUE_PATH = inspection_dependencies['PERSONAL_DATASET_QUEUE_PATH']
    PERSONAL_WORK_PATH = inspection_dependencies['PERSONAL_WORK_PATH']
    _browser_mutation_target_contract_compact_fields = inspection_dependencies['_browser_mutation_target_contract_compact_fields']
    _browser_mutation_write_policy = inspection_dependencies['_browser_mutation_write_policy']
    _browser_mutation_write_runtime_checklist = inspection_dependencies['_browser_mutation_write_runtime_checklist']
    _browser_response_security_policy = inspection_dependencies['_browser_response_security_policy']
    _browser_signed_link_policy = inspection_dependencies['_browser_signed_link_policy']
    _browser_task_state_policy = inspection_dependencies['_browser_task_state_policy']
    _browser_workflow_capabilities = inspection_dependencies['_browser_workflow_capabilities']
    _dataset_queue_direct_start_policy = inspection_dependencies['_dataset_queue_direct_start_policy']
    _dataset_queue_direct_start_policy_fields = inspection_dependencies['_dataset_queue_direct_start_policy_fields']
    _direct_browser_start_contract_compact_fields = inspection_dependencies['_direct_browser_start_contract_compact_fields']
    _handoff_assignment_ownership_fields = inspection_dependencies['_handoff_assignment_ownership_fields']
    _handoff_browser_mutation_write_fields = inspection_dependencies['_handoff_browser_mutation_write_fields']
    _handoff_browser_response_security_fields = inspection_dependencies['_handoff_browser_response_security_fields']
    _handoff_dataset_queue_blocks_labeler_start = inspection_dependencies['_handoff_dataset_queue_blocks_labeler_start']
    _handoff_dataset_queue_start_fields = inspection_dependencies['_handoff_dataset_queue_start_fields']
    _handoff_dataset_queue_state = inspection_dependencies['_handoff_dataset_queue_state']
    _handoff_entry_artifact_fields = inspection_dependencies['_handoff_entry_artifact_fields']
    _handoff_known_user_status_fields = inspection_dependencies['_handoff_known_user_status_fields']
    _handoff_labeler_route_authorization_fields = inspection_dependencies['_handoff_labeler_route_authorization_fields']
    _handoff_labeler_safety_fields = inspection_dependencies['_handoff_labeler_safety_fields']
    _handoff_mutation_audit_fields = inspection_dependencies['_handoff_mutation_audit_fields']
    _handoff_operator_recovery_fields = inspection_dependencies['_handoff_operator_recovery_fields']
    _handoff_ready_to_send = inspection_dependencies['_handoff_ready_to_send']
    _handoff_sendability_actions = inspection_dependencies['_handoff_sendability_actions']
    _handoff_sendability_reasons = inspection_dependencies['_handoff_sendability_reasons']
    _handoff_session_guard_fields = inspection_dependencies['_handoff_session_guard_fields']
    _handoff_signed_link_policy_fields = inspection_dependencies['_handoff_signed_link_policy_fields']
    _handoff_task_state_policy_fields = inspection_dependencies['_handoff_task_state_policy_fields']
    _handoff_zarr_backup_fields = inspection_dependencies['_handoff_zarr_backup_fields']
    _labeler_route_authorization_policy = inspection_dependencies['_labeler_route_authorization_policy']
    _labeler_route_authorization_runtime_checklist = inspection_dependencies['_labeler_route_authorization_runtime_checklist']
    _labeler_safety_policy = inspection_dependencies['_labeler_safety_policy']
    _labeler_work_completion_fields = inspection_dependencies['_labeler_work_completion_fields']
    _mutation_audit_policy = inspection_dependencies['_mutation_audit_policy']
    _parse_handoff_utc = inspection_dependencies['_parse_handoff_utc']
    _public_reassignment_session_safety_fields = inspection_dependencies['_public_reassignment_session_safety_fields']
    _reassignment_session_safety_flat_fields = inspection_dependencies['_reassignment_session_safety_flat_fields']
    _recordings_without_open_tasks_actions = inspection_dependencies['_recordings_without_open_tasks_actions']
    _runtime_operator_validation_gate_cli_policy = inspection_dependencies['_runtime_operator_validation_gate_cli_policy']
    _runtime_operator_validation_gate_cli_policy_fields = inspection_dependencies['_runtime_operator_validation_gate_cli_policy_fields']
    _session_guard_policy = inspection_dependencies['_session_guard_policy']
    _zarr_backup_policy = inspection_dependencies['_zarr_backup_policy']

    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    signed_links = int(counts.get("signed_links") or 0)
    expires_at = _parse_handoff_utc(manifest.get("links_expire_at_utc"))
    expired = bool(expires_at is not None and expires_at <= now)
    unknown_expiration = bool(signed_links and expires_at is None)
    seconds_until_expiration = None
    if expires_at is not None:
        seconds_until_expiration = int((expires_at - now).total_seconds())
    ok = bool(manifest.get("ok")) and not expired and not unknown_expiration
    if unknown_expiration:
        status = "unknown_expiration"
    elif expired:
        status = "expired"
    elif not bool(manifest.get("ok")):
        status = "needs_review"
    else:
        status = "fresh"
    ready_to_send = _handoff_ready_to_send(manifest)
    sendability_reasons = _handoff_sendability_reasons(manifest)
    sendability_actions = _handoff_sendability_actions(sendability_reasons)
    dataset_queue_state = _handoff_dataset_queue_state(manifest)
    raw_reassignment_session_safety = manifest.get("reassignment_session_safety")
    if not isinstance(raw_reassignment_session_safety, Mapping):
        store_consistency = (
            manifest.get("store_consistency")
            if isinstance(manifest.get("store_consistency"), Mapping)
            else {}
        )
        raw_reassignment_session_safety = (
            store_consistency.get("reassignment_session_safety")
            if isinstance(store_consistency.get("reassignment_session_safety"), Mapping)
            else {}
        )
    assignment_snapshot = (
        manifest.get("assignment_snapshot")
        if isinstance(manifest.get("assignment_snapshot"), Mapping)
        else {}
    )
    snapshot_recording_ids = (
        {
            str(recording_id).strip()
            for recording_id in assignment_snapshot.get("recording_ids", [])
            if str(recording_id).strip()
        }
        if isinstance(assignment_snapshot.get("recording_ids"), list)
        else None
    )
    reassignment_session_safety = _public_reassignment_session_safety_fields(
        raw_reassignment_session_safety,
        recording_ids=snapshot_recording_ids,
    )
    labeler_route_authorization_policy = (
        manifest.get("labeler_route_authorization_policy")
        if isinstance(manifest.get("labeler_route_authorization_policy"), Mapping)
        else _labeler_route_authorization_policy()
    )
    known_user_status = (
        manifest.get("known_user_status")
        if isinstance(manifest.get("known_user_status"), Mapping)
        else {}
    )
    labeler_route_authorization_checklist = _labeler_route_authorization_runtime_checklist(
        policy=labeler_route_authorization_policy,
        user=str(manifest.get("user") or ""),
        expected_user=str(manifest.get("expected_user") or manifest.get("user") or ""),
        known_user_status=known_user_status,
        assignment_ownership_contract=(
            manifest.get("assignment_ownership_contract")
            if isinstance(manifest.get("assignment_ownership_contract"), Mapping)
            else None
        ),
    )
    return {
        "ok": ok,
        "status": status,
        "user": manifest.get("user"),
        "output_dir": manifest.get("output_dir"),
        "labeler_landing_url": str(manifest.get("labeler_landing_url") or ""),
        "expected_user_labeler_landing_url": str(manifest.get("expected_user_labeler_landing_url") or ""),
        "labeling_home_url": str(manifest.get("labeling_home_url") or ""),
        "expected_user_labeling_home_url": str(manifest.get("expected_user_labeling_home_url") or ""),
        "dashboard_url": str(manifest.get("dashboard_url") or ""),
        "expected_user_dashboard_url": str(manifest.get("expected_user_dashboard_url") or ""),
        "expected_user_dataset_queue_url": str(manifest.get("expected_user_dataset_queue_url") or ""),
        "personal_work_page_path": str(manifest.get("personal_work_page_path") or PERSONAL_WORK_PATH),
        "personal_dataset_queue_page_path": str(
            manifest.get("personal_dataset_queue_page_path") or PERSONAL_DATASET_QUEUE_PATH
        ),
        "expected_user_personal_work_url": str(manifest.get("expected_user_personal_work_url") or ""),
        "expected_user_personal_dataset_queue_url": str(
            manifest.get("expected_user_personal_dataset_queue_url") or ""
        ),
        "personalized_labeler_entrypoint": str(
            manifest.get("personalized_labeler_entrypoint") or "personal_datasets_waiting_queue"
        ),
        "personalized_labeler_entry_url": str(manifest.get("personalized_labeler_entry_url") or ""),
        "expected_user_identity_probe_url": str(manifest.get("expected_user_identity_probe_url") or ""),
        "dataset_queue_summary": manifest.get("dataset_queue_summary", {}),
        "dataset_queue_state": dataset_queue_state,
        "dataset_queue_state_code": str(dataset_queue_state.get("code") or ""),
        "dataset_queue_state_title": str(dataset_queue_state.get("title") or ""),
        "labeler_work_completion": manifest.get("labeler_work_completion", {}),
        **_labeler_work_completion_fields(
            manifest.get("labeler_work_completion")
            if isinstance(manifest.get("labeler_work_completion"), Mapping)
            else None
        ),
        "dataset_queue_blocks_labeler_start": _handoff_dataset_queue_blocks_labeler_start(manifest),
        "reassignment_session_safety": reassignment_session_safety,
        **_reassignment_session_safety_flat_fields(reassignment_session_safety),
        **_handoff_known_user_status_fields(manifest),
        **_handoff_assignment_ownership_fields(manifest),
        **_handoff_entry_artifact_fields(manifest),
        **_handoff_dataset_queue_start_fields(manifest),
        **_handoff_labeler_safety_fields(manifest),
        **_handoff_labeler_route_authorization_fields(manifest),
        "browser_mutation_write_policy": manifest.get(
            "browser_mutation_write_policy", _browser_mutation_write_policy()
        ),
        "browser_mutation_write_checklist": manifest.get(
            "browser_mutation_write_checklist",
            _browser_mutation_write_runtime_checklist(
                manifest.get("browser_mutation_write_policy")
                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                else None
            ),
        ),
        **_browser_mutation_target_contract_compact_fields(
            manifest.get("browser_mutation_write_checklist")
            if isinstance(manifest.get("browser_mutation_write_checklist"), Mapping)
            else _browser_mutation_write_runtime_checklist(
                manifest.get("browser_mutation_write_policy")
                if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
                else None
            ),
            user=str(manifest.get("user") or ""),
        ),
        "dataset_queue_direct_start_policy": manifest.get(
            "dataset_queue_direct_start_policy", _dataset_queue_direct_start_policy()
        ),
        "dataset_queue_direct_start_policy_present": isinstance(
            manifest.get("dataset_queue_direct_start_policy"), Mapping
        ),
        **_direct_browser_start_contract_compact_fields(
            manifest.get("dataset_queue_direct_start_policy")
            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
            else None,
            user=str(manifest.get("user") or ""),
        ),
        "single_owner_policy_contract_met": bool(
            manifest.get("single_owner_policy_contract_met")
        ),
        **_dataset_queue_direct_start_policy_fields(
            manifest.get("dataset_queue_direct_start_policy")
            if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
            else None
        ),
        "runtime_operator_validation_gate_cli_policy": manifest.get(
            "runtime_operator_validation_gate_cli_policy",
            _runtime_operator_validation_gate_cli_policy(),
        ),
        **_runtime_operator_validation_gate_cli_policy_fields(
            manifest.get("runtime_operator_validation_gate_cli_policy")
            if isinstance(
                manifest.get("runtime_operator_validation_gate_cli_policy"),
                Mapping,
            )
            else None
        ),
        **_handoff_browser_mutation_write_fields(manifest),
        **_handoff_operator_recovery_fields(manifest),
        "zarr_backup_policy": manifest.get("zarr_backup_policy", _zarr_backup_policy()),
        **_handoff_zarr_backup_fields(manifest),
        "mutation_audit_policy": manifest.get("mutation_audit_policy", _mutation_audit_policy()),
        **_handoff_mutation_audit_fields(manifest),
        "browser_response_security_policy": manifest.get(
            "browser_response_security_policy", _browser_response_security_policy()
        ),
        **_handoff_browser_response_security_fields(manifest),
        "dataset_queue_preview_url": str(manifest.get("dataset_queue_preview_url") or ""),
        "canonical_dataset_queue_preview_url": str(
            manifest.get("canonical_dataset_queue_preview_url")
            or manifest.get("expected_user_dataset_queue_url")
            or ""
        ),
        "labeler_safety": manifest.get("labeler_safety", _labeler_safety_policy()),
        "labeler_route_authorization_policy": labeler_route_authorization_policy,
        "labeler_route_authorization_checklist": labeler_route_authorization_checklist,
        "task_state_policy": manifest.get("task_state_policy", _browser_task_state_policy()),
        **_handoff_task_state_policy_fields(manifest),
        "signed_link_policy": manifest.get("signed_link_policy", _browser_signed_link_policy()),
        **_handoff_signed_link_policy_fields(manifest),
        "session_guard_policy": manifest.get("session_guard_policy", _session_guard_policy()),
        **_handoff_session_guard_fields(manifest),
        "browser_workflows": manifest.get("browser_workflows", _browser_workflow_capabilities()),
        "assignment_snapshot": manifest.get("assignment_snapshot", {}),
        "ready_to_send": ready_to_send,
        "sendability_reasons": sendability_reasons,
        "sendability_actions": sendability_actions,
        "recordings_without_open_tasks_actions": counts.get("recordings_without_open_tasks_actions", [])
        if isinstance(counts.get("recordings_without_open_tasks_actions"), list)
        else _recordings_without_open_tasks_actions(
            counts.get("recordings_without_open_tasks_by_reason", {})
            if isinstance(counts.get("recordings_without_open_tasks_by_reason"), Mapping)
            else []
        ),
        "generated_at_utc": manifest.get("generated_at_utc"),
        "links_expire_at_utc": manifest.get("links_expire_at_utc"),
        "seconds_until_expiration": seconds_until_expiration,
        "counts": counts,
    }


def _inspect_handoff_package(
    path: Path,
    *,
    store: LabelingStore | None = None,
    require_shareable: bool = False,
    dependencies: Mapping[str, object],
) -> dict[str, object]:
    inspection_dependencies = _require_inspection_dependencies(dependencies, ('_handoff_status_from_manifest', 'DEFAULT_OPERATOR_VALIDATION_GATE_IDS', 'OPERATOR_VALIDATION_GATE_STATUS_VALUES', '_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS', '_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT', '_IMPLEMENTATION_STATUS_FLAT_FIELDS', '_IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT', '_browser_mutation_target_contract_summary', '_count_handoff_sendability_reasons', '_direct_browser_start_contract_summary', '_handoff_dataset_queue_state_counts', '_identity_personal_queue_evidence_status', '_implementation_status_artifact', '_inspect_handoff_assignment_freshness', '_inspect_handoff_launch_evidence_execution_checklist', '_inspect_handoff_operator_evidence_commands', '_inspect_handoff_validation_checklist', '_inspect_handoff_validation_log', '_inspection_failure_actions', '_inspection_labeler_entrypoint_summary', '_inspection_operator_repair_commands', '_labeler_route_authorization_runtime_checklist_contract_summary', '_launch_evidence_execution_checklist_public_summary', '_load_handoff_documents', '_operator_evidence_commands_public_summary', '_parse_handoff_utc', '_recordings_without_open_tasks_actions', '_safe_share_checklist_gate_status_fields', '_safe_share_external_launch_evidence_gap_field_names', '_safe_share_gate_flat_fields', '_safe_share_gate_policy', '_safe_share_next_action_command_fields', '_safe_share_next_action_detail_fields', '_shareability_compact_contract_fields', '_shareability_compact_contract_source_fields', '_shareability_labeler_route_authorization_runtime_checklist_gate', '_shareability_repair_command_contracts', '_shareability_repair_command_detail_fields', '_shareability_repair_command_detail_fields_by_id', '_shareability_safe_to_share_requires', '_single_owner_package_contract_summary', '_sum_recordings_without_open_tasks_by_reason', '_verify_handoff_checksums'))
    _handoff_status_from_manifest = inspection_dependencies['_handoff_status_from_manifest']
    DEFAULT_OPERATOR_VALIDATION_GATE_IDS = inspection_dependencies['DEFAULT_OPERATOR_VALIDATION_GATE_IDS']
    OPERATOR_VALIDATION_GATE_STATUS_VALUES = inspection_dependencies['OPERATOR_VALIDATION_GATE_STATUS_VALUES']
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS = inspection_dependencies['_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS']
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT = inspection_dependencies['_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT']
    _IMPLEMENTATION_STATUS_FLAT_FIELDS = inspection_dependencies['_IMPLEMENTATION_STATUS_FLAT_FIELDS']
    _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT = inspection_dependencies['_IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT']
    _browser_mutation_target_contract_summary = inspection_dependencies['_browser_mutation_target_contract_summary']
    _count_handoff_sendability_reasons = inspection_dependencies['_count_handoff_sendability_reasons']
    _direct_browser_start_contract_summary = inspection_dependencies['_direct_browser_start_contract_summary']
    _handoff_dataset_queue_state_counts = inspection_dependencies['_handoff_dataset_queue_state_counts']
    _identity_personal_queue_evidence_status = inspection_dependencies['_identity_personal_queue_evidence_status']
    _implementation_status_artifact = inspection_dependencies['_implementation_status_artifact']
    _inspect_handoff_assignment_freshness = inspection_dependencies['_inspect_handoff_assignment_freshness']
    _inspect_handoff_launch_evidence_execution_checklist = inspection_dependencies['_inspect_handoff_launch_evidence_execution_checklist']
    _inspect_handoff_operator_evidence_commands = inspection_dependencies['_inspect_handoff_operator_evidence_commands']
    _inspect_handoff_validation_checklist = inspection_dependencies['_inspect_handoff_validation_checklist']
    _inspect_handoff_validation_log = inspection_dependencies['_inspect_handoff_validation_log']
    _inspection_failure_actions = inspection_dependencies['_inspection_failure_actions']
    _inspection_labeler_entrypoint_summary = inspection_dependencies['_inspection_labeler_entrypoint_summary']
    _inspection_operator_repair_commands = inspection_dependencies['_inspection_operator_repair_commands']
    _labeler_route_authorization_runtime_checklist_contract_summary = inspection_dependencies['_labeler_route_authorization_runtime_checklist_contract_summary']
    _launch_evidence_execution_checklist_public_summary = inspection_dependencies['_launch_evidence_execution_checklist_public_summary']
    _load_handoff_documents = inspection_dependencies['_load_handoff_documents']
    _operator_evidence_commands_public_summary = inspection_dependencies['_operator_evidence_commands_public_summary']
    _parse_handoff_utc = inspection_dependencies['_parse_handoff_utc']
    _recordings_without_open_tasks_actions = inspection_dependencies['_recordings_without_open_tasks_actions']
    _safe_share_checklist_gate_status_fields = inspection_dependencies['_safe_share_checklist_gate_status_fields']
    _safe_share_external_launch_evidence_gap_field_names = inspection_dependencies['_safe_share_external_launch_evidence_gap_field_names']
    _safe_share_gate_flat_fields = inspection_dependencies['_safe_share_gate_flat_fields']
    _safe_share_gate_policy = inspection_dependencies['_safe_share_gate_policy']
    _safe_share_next_action_command_fields = inspection_dependencies['_safe_share_next_action_command_fields']
    _safe_share_next_action_detail_fields = inspection_dependencies['_safe_share_next_action_detail_fields']
    _shareability_compact_contract_fields = inspection_dependencies['_shareability_compact_contract_fields']
    _shareability_compact_contract_source_fields = inspection_dependencies['_shareability_compact_contract_source_fields']
    _shareability_labeler_route_authorization_runtime_checklist_gate = inspection_dependencies['_shareability_labeler_route_authorization_runtime_checklist_gate']
    _shareability_repair_command_contracts = inspection_dependencies['_shareability_repair_command_contracts']
    _shareability_repair_command_detail_fields = inspection_dependencies['_shareability_repair_command_detail_fields']
    _shareability_repair_command_detail_fields_by_id = inspection_dependencies['_shareability_repair_command_detail_fields_by_id']
    _shareability_safe_to_share_requires = inspection_dependencies['_shareability_safe_to_share_requires']
    _single_owner_package_contract_summary = inspection_dependencies['_single_owner_package_contract_summary']
    _sum_recordings_without_open_tasks_by_reason = inspection_dependencies['_sum_recordings_without_open_tasks_by_reason']
    _verify_handoff_checksums = inspection_dependencies['_verify_handoff_checksums']

    now = datetime.now(timezone.utc)
    kind, index, manifests = _load_handoff_documents(path)
    checksum_verification = _verify_handoff_checksums(path)
    validation_log = _inspect_handoff_validation_log(path)
    validation_checklist = _inspect_handoff_validation_checklist(path)
    is_launch_bundle = index is not None and str(kind).startswith("launch")
    operator_evidence_commands = _inspect_handoff_operator_evidence_commands(path, required=is_launch_bundle)
    launch_evidence_execution_checklist = (
        _inspect_handoff_launch_evidence_execution_checklist(
            path,
            required=is_launch_bundle,
        )
    )
    handoffs = [_handoff_status_from_manifest(manifest, now) for manifest in manifests]
    labeler_entrypoint_summary = _inspection_labeler_entrypoint_summary(index, handoffs)
    assignment_freshness = _inspect_handoff_assignment_freshness(manifests, store=store, package_kind=kind)
    earliest_expiration: datetime | None = None
    for handoff in handoffs:
        expires_at = _parse_handoff_utc(handoff.get("links_expire_at_utc"))
        if expires_at is not None and (earliest_expiration is None or expires_at < earliest_expiration):
            earliest_expiration = expires_at
    aggregate_counts = {
        "users": len(handoffs),
        "recordings": sum(int((handoff.get("counts") or {}).get("recordings") or 0) for handoff in handoffs),
        "tasks": sum(int((handoff.get("counts") or {}).get("tasks") or 0) for handoff in handoffs),
        "signed_links": sum(int((handoff.get("counts") or {}).get("signed_links") or 0) for handoff in handoffs),
        "waiting_datasets": sum(int((handoff.get("counts") or {}).get("waiting_datasets") or 0) for handoff in handoffs),
        "dataset_open_tasks": sum(int((handoff.get("counts") or {}).get("dataset_open_tasks") or 0) for handoff in handoffs),
        "dataset_queue_states": _handoff_dataset_queue_state_counts(handoffs),
        "dataset_queue_blocked_start_users": [
            str(handoff.get("user") or "")
            for handoff in handoffs
            if bool(handoff.get("dataset_queue_blocks_labeler_start"))
        ],
        "reassignment_session_safety_blocked_users": [
            str(handoff.get("user") or "")
            for handoff in handoffs
            if bool(handoff.get("reassignment_session_safety_blocks_labeler_mutation"))
        ],
        "reassignment_session_safety_blocked_user_count": sum(
            1
            for handoff in handoffs
            if bool(handoff.get("reassignment_session_safety_blocks_labeler_mutation"))
        ),
        "reassignment_session_safety_mismatch_count": sum(
            int(handoff.get("reassignment_session_safety_active_session_assignment_mismatch_count") or 0)
            for handoff in handoffs
        ),
        "reassignment_session_safety_blocked_recording_ids": sorted(
            {
                str(recording_id)
                for handoff in handoffs
                for recording_id in (
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
                )
                if str(recording_id).strip()
            }
        ),
        "recordings_without_open_tasks": sum(
            int((handoff.get("counts") or {}).get("recordings_without_open_tasks") or 0)
            for handoff in handoffs
        ),
        "recordings_without_open_tasks_by_reason": _sum_recordings_without_open_tasks_by_reason(handoffs),
        "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(
            _sum_recordings_without_open_tasks_by_reason(handoffs)
        ),
        "sendability_reasons": _count_handoff_sendability_reasons(handoffs),
        "redacted_summary_fields": sum(
            int((handoff.get("counts") or {}).get("redacted_summary_fields") or 0)
            for handoff in handoffs
        ),
        "needs_review": sum(1 for handoff in handoffs if str(handoff.get("status")) == "needs_review"),
        "expired": sum(1 for handoff in handoffs if str(handoff.get("status")) == "expired"),
        "unknown_expiration": sum(1 for handoff in handoffs if str(handoff.get("status")) == "unknown_expiration"),
    }
    ok = bool(handoffs) and all(
        bool(handoff.get("ok")) and bool(handoff.get("ready_to_send"))
        for handoff in handoffs
    )
    readiness_ok = None
    handoffs_ok = None
    handoff_store_checks_ok = None
    failure_reasons: list[str] = []
    if not handoffs:
        failure_reasons.append("no_handoffs")
    for handoff in handoffs:
        status = str(handoff.get("status") or "")
        if status != "fresh":
            failure_reasons.append(status)
        if not bool(handoff.get("ready_to_send")):
            failure_reasons.append("handoff_not_ready")
        sendability_reasons = (
            handoff.get("sendability_reasons")
            if isinstance(handoff.get("sendability_reasons"), list)
            else []
        )
        if "preferred_personal_queue_mismatch" in {
            str(reason) for reason in sendability_reasons
        }:
            failure_reasons.append("preferred_personal_queue_mismatch")
        if bool(handoff.get("dataset_queue_blocks_labeler_start")):
            failure_reasons.append("dataset_queue_blocks_labeler_start")
        if bool(handoff.get("reassignment_session_safety_blocks_labeler_mutation")):
            failure_reasons.append("reassignment_session_safety_failed")
    browser_mutation_target_contract = _browser_mutation_target_contract_summary(handoffs)
    ok = ok and bool(browser_mutation_target_contract.get("met"))
    if handoffs and not bool(browser_mutation_target_contract.get("met")):
        failure_reasons.append("browser_mutation_target_contract_mismatch")
    direct_browser_start_contract = _direct_browser_start_contract_summary(handoffs)
    ok = ok and bool(direct_browser_start_contract.get("met"))
    if handoffs and not bool(direct_browser_start_contract.get("met")):
        failure_reasons.append("direct_browser_start_contract_mismatch")
    single_owner_package_contract = _single_owner_package_contract_summary(handoffs)
    ok = ok and bool(single_owner_package_contract.get("met"))
    if handoffs and not bool(single_owner_package_contract.get("met")):
        failure_reasons.append("single_owner_package_contract_mismatch")
    labeler_route_authorization_runtime_checklist_contract = (
        _labeler_route_authorization_runtime_checklist_contract_summary(handoffs)
    )
    ok = ok and bool(labeler_route_authorization_runtime_checklist_contract.get("met"))
    if handoffs and not bool(
        labeler_route_authorization_runtime_checklist_contract.get("met")
    ):
        failure_reasons.append(
            "labeler_route_authorization_runtime_checklist_mismatch"
        )
    if is_launch_bundle:
        readiness_ok = bool(index.get("readiness_ok"))
        handoffs_ok = bool(index.get("handoffs_ok"))
        handoff_store_checks_ok = bool(
            index.get("handoff_store_checks_ok", all(bool(handoff.get("ok")) for handoff in handoffs))
        )
        ok = ok and readiness_ok and handoffs_ok and handoff_store_checks_ok
        if not readiness_ok:
            failure_reasons.append("readiness_failed")
        if not handoffs_ok:
            failure_reasons.append("handoffs_failed")
        if not handoff_store_checks_ok:
            failure_reasons.append("handoff_store_checks_failed")
        operator_evidence_commands_present = bool(operator_evidence_commands.get("present"))
        operator_evidence_commands_valid = bool(operator_evidence_commands.get("valid"))
        ok = ok and operator_evidence_commands_present and operator_evidence_commands_valid
        if not bool(operator_evidence_commands.get("present")):
            failure_reasons.append("operator_evidence_commands_missing")
        elif not operator_evidence_commands_valid:
            if not bool(operator_evidence_commands.get("operator_only_boundary_present")):
                failure_reasons.append("operator_evidence_commands_boundary_missing")
            else:
                failure_reasons.append("operator_evidence_commands_invalid")
        launch_evidence_execution_checklist_present = bool(
            launch_evidence_execution_checklist.get("present")
        )
        launch_evidence_execution_checklist_valid = bool(
            launch_evidence_execution_checklist.get("valid")
        )
        ok = ok and launch_evidence_execution_checklist_present and launch_evidence_execution_checklist_valid
        if not launch_evidence_execution_checklist_present:
            failure_reasons.append("launch_evidence_execution_checklist_missing")
        elif not launch_evidence_execution_checklist_valid:
            if not bool(
                launch_evidence_execution_checklist.get("checklist_contract_present")
            ):
                failure_reasons.append("launch_evidence_execution_checklist_incomplete")
            else:
                failure_reasons.append("launch_evidence_execution_checklist_invalid")
    if bool(assignment_freshness.get("checked_against_current_store")):
        ok = ok and bool(assignment_freshness.get("ok"))
        if int(assignment_freshness.get("snapshot_missing_count") or 0):
            failure_reasons.append("assignment_freshness_unverified")
        if int(assignment_freshness.get("stale_recording_count") or 0):
            failure_reasons.append("assignment_freshness_mismatch")
        if int(assignment_freshness.get("extra_current_assignment_count") or 0):
            failure_reasons.append("assignment_freshness_incomplete")
    ok = ok and bool(checksum_verification.get("ok", True))
    if not bool(checksum_verification.get("ok", True)):
        failure_reasons.append("checksum_failed")
    ok = ok and bool(validation_log.get("present"))
    if not bool(validation_log.get("present")):
        failure_reasons.append("validation_log_missing")
    ok = ok and bool(validation_checklist.get("present"))
    if not bool(validation_checklist.get("present")):
        failure_reasons.append("validation_checklist_missing")
    ok = ok and bool(validation_checklist.get("valid", True))
    if validation_checklist.get("present") and not bool(validation_checklist.get("valid", True)):
        failure_reasons.append("validation_checklist_invalid")
    ok = ok and bool(validation_checklist.get("ready_for_operator_validation", True))
    if validation_checklist.get("present") and not bool(validation_checklist.get("ready_for_operator_validation", True)):
        failure_reasons.append("validation_checklist_needs_review")
    implementation_status_checklist_artifact_missing_fields = [
        str(field)
        for field in (
            validation_checklist.get("implementation_status_artifact_missing_fields")
            if isinstance(
                validation_checklist.get("implementation_status_artifact_missing_fields"),
                list,
            )
            else []
        )
        if str(field)
    ]
    implementation_status_checklist_artifact_complete = bool(
        validation_checklist.get("implementation_status_artifact_complete")
    )
    if validation_checklist.get("present"):
        ok = ok and implementation_status_checklist_artifact_complete
        if not implementation_status_checklist_artifact_complete:
            failure_reasons.append("implementation_status_artifact_incomplete")
    required_pending_gate_ids = (
        validation_checklist.get("required_pending_gate_ids")
        if isinstance(validation_checklist.get("required_pending_gate_ids"), list)
        else []
    )
    ok = ok and not required_pending_gate_ids
    if required_pending_gate_ids:
        failure_reasons.append("validation_evidence_pending")
    passed_unapproved_template_gate_ids = (
        validation_checklist.get("passed_gates_with_unapproved_evidence_templates")
        if isinstance(validation_checklist.get("passed_gates_with_unapproved_evidence_templates"), list)
        else []
    )
    ok = ok and not passed_unapproved_template_gate_ids
    if passed_unapproved_template_gate_ids:
        failure_reasons.append("validation_evidence_template_unapproved")
    failure_reasons = sorted(set(reason for reason in failure_reasons if reason))
    if ok:
        status = "fresh"
    elif "checksum_failed" in failure_reasons:
        status = "checksum_failed"
    elif "expired" in failure_reasons:
        status = "expired"
    elif "unknown_expiration" in failure_reasons:
        status = "unknown_expiration"
    elif "readiness_failed" in failure_reasons:
        status = "readiness_failed"
    elif "assignment_freshness_mismatch" in failure_reasons or "assignment_freshness_incomplete" in failure_reasons:
        status = "stale_assignment"
    elif (
        "handoffs_failed" in failure_reasons
        or "needs_review" in failure_reasons
        or "handoff_not_ready" in failure_reasons
        or "browser_mutation_target_contract_mismatch" in failure_reasons
        or "direct_browser_start_contract_mismatch" in failure_reasons
        or "single_owner_package_contract_mismatch" in failure_reasons
        or "labeler_route_authorization_runtime_checklist_mismatch" in failure_reasons
        or "handoff_store_checks_failed" in failure_reasons
        or "assignment_freshness_unverified" in failure_reasons
        or "operator_evidence_commands_missing" in failure_reasons
        or "operator_evidence_commands_boundary_missing" in failure_reasons
        or "operator_evidence_commands_invalid" in failure_reasons
        or "launch_evidence_execution_checklist_missing" in failure_reasons
        or "launch_evidence_execution_checklist_incomplete" in failure_reasons
        or "launch_evidence_execution_checklist_invalid" in failure_reasons
        or "validation_log_missing" in failure_reasons
        or "validation_checklist_missing" in failure_reasons
        or "validation_checklist_invalid" in failure_reasons
        or "validation_checklist_needs_review" in failure_reasons
        or "implementation_status_artifact_incomplete" in failure_reasons
        or "validation_evidence_pending" in failure_reasons
        or "validation_evidence_template_unapproved" in failure_reasons
    ):
        status = "needs_review"
    else:
        status = "failed"
    failure_actions = _inspection_failure_actions(
        failure_reasons,
        validation_checklist=validation_checklist,
    )
    launch_evidence_execution_checklist_summary = (
        _launch_evidence_execution_checklist_public_summary(
            launch_evidence_execution_checklist
        )
    )
    operator_repair_commands = _inspection_operator_repair_commands(
        failure_actions,
        validation_checklist=validation_checklist,
        launch_evidence_execution_checklist_summary=(
            launch_evidence_execution_checklist_summary
        ),
    )
    for repair_command in operator_repair_commands:
        repair_command_id = str(repair_command.get("id") or "")
        if repair_command_id == "regenerate_handoffs_with_browser_mutation_target_contract":
            repair_command["contract"] = "browser_mutation_target_contract"
            repair_command["repair_mode"] = "regenerate_handoff_package"
            repair_command["safe_share_blocker"] = "browser_mutation_target_contract_mismatch"
            repair_command["required_values"] = dict(
                browser_mutation_target_contract.get("required_values", {})
            )
            repair_command["mismatches"] = list(
                browser_mutation_target_contract.get("mismatches", [])
            )
            repair_command["mismatch_count"] = int(
                browser_mutation_target_contract.get("mismatch_count") or 0
            )
            repair_command["mismatch_users"] = list(
                browser_mutation_target_contract.get("mismatch_users", [])
            )
        elif repair_command_id == "regenerate_handoffs_with_direct_browser_start_contract":
            repair_command["contract"] = "direct_browser_start_contract"
            repair_command["repair_mode"] = "regenerate_handoff_package"
            repair_command["safe_share_blocker"] = "direct_browser_start_contract_mismatch"
            repair_command["required_values"] = dict(
                direct_browser_start_contract.get("required_values", {})
            )
            repair_command["mismatches"] = list(
                direct_browser_start_contract.get("mismatches", [])
            )
            repair_command["mismatch_count"] = int(
                direct_browser_start_contract.get("mismatch_count") or 0
            )
            repair_command["mismatch_users"] = list(
                direct_browser_start_contract.get("mismatch_users", [])
            )
        elif repair_command_id == "regenerate_handoffs_with_single_owner_package_contract":
            repair_command["contract"] = "single_owner_package_contract"
            repair_command["repair_mode"] = "regenerate_handoff_package"
            repair_command["safe_share_blocker"] = "single_owner_package_contract_mismatch"
            repair_command["mismatch_count"] = int(
                single_owner_package_contract.get("mismatch_count") or 0
            )
            repair_command["mismatch_recording_ids"] = list(
                single_owner_package_contract.get("mismatch_recording_ids", [])
            )
            repair_command["duplicate_owners_by_recording"] = dict(
                single_owner_package_contract.get("duplicate_owners_by_recording", {})
            )
        elif (
            repair_command_id
            == "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
        ):
            repair_command["contract"] = (
                "labeler_route_authorization_runtime_checklist_gate"
            )
            repair_command["repair_mode"] = "regenerate_handoff_package"
            repair_command["safe_share_blocker"] = (
                "labeler_route_authorization_runtime_checklist_mismatch"
            )
            repair_command["required_values"] = dict(
                labeler_route_authorization_runtime_checklist_contract.get(
                    "required_values", {}
                )
            )
            repair_command["mismatches"] = list(
                labeler_route_authorization_runtime_checklist_contract.get(
                    "mismatches", []
                )
            )
            repair_command["mismatch_count"] = int(
                labeler_route_authorization_runtime_checklist_contract.get(
                    "mismatch_count"
                )
                or 0
            )
            repair_command["mismatch_users"] = list(
                labeler_route_authorization_runtime_checklist_contract.get(
                    "mismatch_users", []
                )
            )
    operator_repair_command_categories: dict[str, int] = {}
    operator_repair_command_gate_ids: set[str] = set()
    operator_repair_command_reason_ids: set[str] = set()
    for command in operator_repair_commands:
        category = str(command.get("category") or "")
        if category:
            operator_repair_command_categories[category] = (
                operator_repair_command_categories.get(category, 0) + 1
            )
        gate_ids = command.get("gate_ids") if isinstance(command.get("gate_ids"), list) else []
        for gate_id in gate_ids:
            gate_id_text = str(gate_id)
            if gate_id_text and gate_id_text != "GATE_ID":
                operator_repair_command_gate_ids.add(gate_id_text)
        reason_ids = (
            command.get("reason_ids") if isinstance(command.get("reason_ids"), list) else []
        )
        for reason_id in reason_ids:
            reason_id_text = str(reason_id)
            if reason_id_text:
                operator_repair_command_reason_ids.add(reason_id_text)
    operator_repair_command_gate_id_list = sorted(operator_repair_command_gate_ids)
    operator_repair_command_reason_id_list = sorted(operator_repair_command_reason_ids)
    operator_repair_commands_requiring_checksum_refresh = sum(
        1 for command in operator_repair_commands if bool(command.get("requires_checksum_refresh_after_run"))
    )
    operator_validation_command_templates = (
        validation_checklist.get("operator_validation_command_templates")
        if isinstance(validation_checklist.get("operator_validation_command_templates"), Mapping)
        else {}
    )
    operator_validation_command_template_command_ids = [
        str(command_id)
        for command_id in (
            operator_validation_command_templates.get("command_ids")
            if isinstance(operator_validation_command_templates.get("command_ids"), list)
            else []
        )
        if str(command_id)
    ]
    operator_validation_command_template_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get("gate_ids")
            if isinstance(operator_validation_command_templates.get("gate_ids"), list)
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_missing_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get("missing_command_gate_ids")
            if isinstance(
                operator_validation_command_templates.get("missing_command_gate_ids"),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_template_backed_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get("template_backed_gate_ids")
            if isinstance(
                operator_validation_command_templates.get("template_backed_gate_ids"),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_validation_checklist_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get("validation_checklist_gate_ids")
            if isinstance(
                operator_validation_command_templates.get("validation_checklist_gate_ids"),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_apply_required_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get("apply_required_gate_ids")
            if isinstance(
                operator_validation_command_templates.get("apply_required_gate_ids"),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_evidence_template_fields_by_gate_id = dict(
        operator_validation_command_templates.get("evidence_template_fields_by_gate_id")
        if isinstance(
            operator_validation_command_templates.get(
                "evidence_template_fields_by_gate_id"
            ),
            Mapping,
        )
        else {}
    )
    operator_validation_command_template_evidence_template_paths_by_gate_id = dict(
        operator_validation_command_templates.get("evidence_template_paths_by_gate_id")
        if isinstance(
            operator_validation_command_templates.get(
                "evidence_template_paths_by_gate_id"
            ),
            Mapping,
        )
        else {}
    )
    operator_validation_command_template_launch_evidence_collection_plan = dict(
        operator_validation_command_templates.get("launch_evidence_collection_plan")
        if isinstance(
            operator_validation_command_templates.get("launch_evidence_collection_plan"),
            Mapping,
        )
        else {}
    )
    operator_validation_command_template_launch_evidence_collection_gate_ids = [
        str(gate_id)
        for gate_id in (
            operator_validation_command_templates.get(
                "launch_evidence_collection_gate_ids"
            )
            if isinstance(
                operator_validation_command_templates.get(
                    "launch_evidence_collection_gate_ids"
                ),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_command_template_launch_evidence_collection_record_command_ids = [
        str(command_id)
        for command_id in (
            operator_validation_command_templates.get(
                "launch_evidence_collection_record_command_ids"
            )
            if isinstance(
                operator_validation_command_templates.get(
                    "launch_evidence_collection_record_command_ids"
                ),
                list,
            )
            else []
        )
        if str(command_id)
    ]
    operator_validation_command_template_summary = {
        "schema": str(operator_validation_command_templates.get("schema") or ""),
        "commands_are_operator_only": bool(
            operator_validation_command_templates.get("commands_are_operator_only", True)
        ),
        "commands_are_labeler_instructions": bool(
            operator_validation_command_templates.get("commands_are_labeler_instructions")
        ),
        "labelers_must_not_run_commands": bool(
            operator_validation_command_templates.get("labelers_must_not_run_commands", True)
        ),
        "operator_authorization_required": bool(
            operator_validation_command_templates.get("operator_authorization_required", True)
        ),
        "command_count": int(operator_validation_command_templates.get("command_count") or 0),
        "command_ids": operator_validation_command_template_command_ids,
        "gate_ids": operator_validation_command_template_gate_ids,
        "missing_command_gate_ids": operator_validation_command_template_missing_gate_ids,
        "missing_command_gate_count": len(operator_validation_command_template_missing_gate_ids),
        "template_backed_gate_ids": (
            operator_validation_command_template_template_backed_gate_ids
        ),
        "validation_checklist_gate_ids": (
            operator_validation_command_template_validation_checklist_gate_ids
        ),
        "apply_required_gate_ids": (
            operator_validation_command_template_apply_required_gate_ids
        ),
        "evidence_template_fields_by_gate_id": (
            operator_validation_command_template_evidence_template_fields_by_gate_id
        ),
        "evidence_template_paths_by_gate_id": (
            operator_validation_command_template_evidence_template_paths_by_gate_id
        ),
        "launch_evidence_collection_plan_schema": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_plan_schema"
            )
            or ""
        ),
        "launch_evidence_collection_step_count": int(
            operator_validation_command_templates.get(
                "launch_evidence_collection_step_count"
            )
            or 0
        ),
        "launch_evidence_collection_gate_ids": (
            operator_validation_command_template_launch_evidence_collection_gate_ids
        ),
        "launch_evidence_collection_record_command_ids": (
            operator_validation_command_template_launch_evidence_collection_record_command_ids
        ),
        "launch_evidence_collection_operator_only": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_operator_only",
                True,
            )
        ),
        "launch_evidence_collection_required_final_field": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_field"
            )
            or ""
        ),
        "launch_evidence_collection_required_final_value": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_value"
            )
        ),
        "launch_evidence_collection_final_inspection_command": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_final_inspection_command"
            )
            or ""
        ),
        "operator_action": str(operator_validation_command_templates.get("operator_action") or ""),
    }
    operator_evidence_commands_summary = _operator_evidence_commands_public_summary(
        operator_evidence_commands
    )
    required_pending_gate_ids = [
        str(gate_id)
        for gate_id in (
            validation_checklist.get("required_pending_gate_ids")
            if isinstance(validation_checklist.get("required_pending_gate_ids"), list)
            else []
        )
        if str(gate_id)
    ]
    needs_review_gate_ids = [
        str(gate_id)
        for gate_id in (
            validation_checklist.get("needs_review_gate_ids")
            if isinstance(validation_checklist.get("needs_review_gate_ids"), list)
            else []
        )
        if str(gate_id)
    ]
    unapproved_template_gate_ids = [
        str(gate_id)
        for gate_id in (
            validation_checklist.get("passed_gates_with_unapproved_evidence_templates")
            if isinstance(validation_checklist.get("passed_gates_with_unapproved_evidence_templates"), list)
            else []
        )
        if str(gate_id)
    ]
    shareability_blocking_gate_ids = sorted(
        set(
            required_pending_gate_ids
            + needs_review_gate_ids
            + unapproved_template_gate_ids
            + operator_repair_command_gate_id_list
        )
    )
    operator_action_required_before_share = bool(failure_actions or operator_repair_commands)
    shareability_blocking_reason_ids = set(str(reason) for reason in failure_reasons if str(reason))
    if not implementation_status_checklist_artifact_complete:
        shareability_blocking_reason_ids.add(
            "implementation_status_checklist_artifact_complete_required_value_mismatch"
        )
    if shareability_blocking_gate_ids:
        shareability_blocking_reason_ids.add("validation_gates_blocking_share")
    if operator_repair_commands:
        shareability_blocking_reason_ids.add("operator_repair_commands_required")
    shareability_blocking_reason_ids.update(operator_repair_command_reason_id_list)
    if operator_repair_commands_requiring_checksum_refresh:
        shareability_blocking_reason_ids.add("checksum_refresh_required_after_repair")
    if status == "needs_review":
        shareability_blocking_reason_ids.add("operator_review_required")
    elif not ok:
        shareability_blocking_reason_ids.add("inspection_failed")
    shareability_blocking_reason_id_list = sorted(shareability_blocking_reason_ids)
    if ok and not operator_action_required_before_share:
        shareability_status = "ready_to_share"
        shareability_operator_action = ""
    elif status == "needs_review":
        shareability_status = "needs_operator_review"
        shareability_operator_action = "Complete required operator review, evidence, repair, and checksum steps before sharing labeler links."
    else:
        shareability_status = "blocked"
        shareability_operator_action = "Repair failed handoff inspection checks before sharing labeler links."
    labeler_links_safe_to_share = shareability_status == "ready_to_share"
    shareability_requirement_met = (
        not bool(require_shareable) or bool(labeler_links_safe_to_share)
    )
    safe_share_gate = _safe_share_gate_policy()
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields(
        gates=validation_checklist.get("gates")
        if isinstance(validation_checklist.get("gates"), list)
        else [],
        safe_share_gate=safe_share_gate,
    )
    identity_personal_queue_evidence_ready_users = list(
        validation_checklist.get("identity_personal_queue_evidence_ready_users")
        if isinstance(validation_checklist.get("identity_personal_queue_evidence_ready_users"), list)
        else []
    )
    identity_personal_queue_evidence_missing_users = list(
        validation_checklist.get("identity_personal_queue_evidence_missing_users")
        if isinstance(validation_checklist.get("identity_personal_queue_evidence_missing_users"), list)
        else []
    )
    identity_personal_queue_evidence_missing_fields_by_user = dict(
        validation_checklist.get("identity_personal_queue_evidence_missing_fields_by_user")
        if isinstance(
            validation_checklist.get("identity_personal_queue_evidence_missing_fields_by_user"),
            Mapping,
        )
        else {}
    )
    identity_personal_queue_ready_count_text = str(
        validation_checklist.get("identity_personal_queue_evidence_ready_count") or "0"
    ).strip()
    identity_personal_queue_missing_count_text = str(
        validation_checklist.get("identity_personal_queue_evidence_missing_count") or "0"
    ).strip()
    identity_personal_queue_evidence_summary = {
        "ready_count": int(identity_personal_queue_ready_count_text)
        if identity_personal_queue_ready_count_text.isdigit()
        else 0,
        "missing_count": int(identity_personal_queue_missing_count_text)
        if identity_personal_queue_missing_count_text.isdigit()
        else 0,
        "ready_users": identity_personal_queue_evidence_ready_users,
        "missing_users": identity_personal_queue_evidence_missing_users,
        "missing_fields_by_user": identity_personal_queue_evidence_missing_fields_by_user,
        "all_users_have_personal_queue_evidence": bool(
            validation_checklist.get("identity_all_users_have_personal_queue_evidence")
        ),
    }
    identity_personal_queue_evidence_status = _identity_personal_queue_evidence_status(
        ready_count=int(identity_personal_queue_evidence_summary["ready_count"] or 0),
        missing_count=int(identity_personal_queue_evidence_summary["missing_count"] or 0),
        ready_users=identity_personal_queue_evidence_ready_users,
        missing_users=identity_personal_queue_evidence_missing_users,
        missing_fields_by_user=identity_personal_queue_evidence_missing_fields_by_user,
        all_users_have_personal_queue_evidence=bool(
            identity_personal_queue_evidence_summary[
                "all_users_have_personal_queue_evidence"
            ]
        ),
    )
    identity_personal_queue_evidence_summary["status"] = (
        identity_personal_queue_evidence_status
    )
    operator_validation_external_evidence_required_gate_ids = [
        str(gate_id)
        for gate_id in (
            validation_checklist.get(
                "operator_validation_external_evidence_required_gate_ids"
            )
            if isinstance(
                validation_checklist.get(
                    "operator_validation_external_evidence_required_gate_ids"
                ),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    operator_validation_external_evidence_template_fields_by_gate_id = dict(
        validation_checklist.get(
            "operator_validation_external_evidence_template_fields_by_gate_id"
        )
        if isinstance(
            validation_checklist.get(
                "operator_validation_external_evidence_template_fields_by_gate_id"
            ),
            Mapping,
        )
        else {}
    )
    operator_validation_external_evidence_template_paths_by_gate_id = dict(
        validation_checklist.get(
            "operator_validation_external_evidence_template_paths_by_gate_id"
        )
        if isinstance(
            validation_checklist.get(
                "operator_validation_external_evidence_template_paths_by_gate_id"
            ),
            Mapping,
        )
        else {}
    )
    operator_validation_checklist_only_required_gate_ids = [
        str(gate_id)
        for gate_id in (
            validation_checklist.get("operator_validation_checklist_only_required_gate_ids")
            if isinstance(
                validation_checklist.get(
                    "operator_validation_checklist_only_required_gate_ids"
                ),
                list,
            )
            else []
        )
        if str(gate_id)
    ]
    implementation_status_matched_paths = [
        str(path)
        for path in (
            validation_checklist.get("implementation_status_matched_paths")
            if isinstance(validation_checklist.get("implementation_status_matched_paths"), list)
            else []
        )
        if str(path)
    ]
    implementation_status_related_paths = [
        str(path)
        for path in (
            validation_checklist.get("implementation_status_related_paths")
            if isinstance(validation_checklist.get("implementation_status_related_paths"), list)
            else []
        )
        if str(path)
    ]
    implementation_status_declared_count_text = str(
        validation_checklist.get("implementation_status_declared_present_count") or "0"
    ).strip()
    implementation_status_declared_present_count = (
        int(implementation_status_declared_count_text)
        if implementation_status_declared_count_text.isdigit()
        else 0
    )
    implementation_status_artifact = {
        **_implementation_status_artifact(
            checklist_declared_path=str(validation_checklist.get("implementation_status") or ""),
            file=str(validation_checklist.get("implementation_status_file") or ""),
        ),
        "safe_share_gate": str(safe_share_fields.get("safe_share_gate_id") or "labeler_links_safe_to_share"),
        "safe_share_required_inspection_field": str(
            safe_share_fields.get("safe_share_required_inspection_field")
            or "labeler_links_safe_to_share"
        ),
        "safe_share_required_inspection_value": bool(
            safe_share_fields.get("safe_share_required_inspection_value", True)
        ),
        "required_path": str(validation_checklist.get("implementation_status_required_path") or ""),
        "declared_present": bool(
            validation_checklist.get("implementation_status_declared_present")
        )
        or implementation_status_declared_present_count > 0,
        "declared_present_count": implementation_status_declared_present_count,
        "present": bool(validation_checklist.get("implementation_status_present")),
        "matched_paths": implementation_status_matched_paths,
        "related_paths": implementation_status_related_paths,
        "matched_path_count": len(implementation_status_matched_paths),
        "related_path_count": len(implementation_status_related_paths),
    }
    implementation_status_checklist_artifact_gate = {
        "schema": "palette.web_labeling_implementation_status_checklist_artifact_gate.v1",
        "field": "implementation_status_checklist_artifact_complete",
        "observed_value": implementation_status_checklist_artifact_complete,
        "required_value": True,
        "matches_required_value": implementation_status_checklist_artifact_complete is True,
        "artifact_present": bool(
            validation_checklist.get("implementation_status_artifact_present")
        ),
        "missing_fields": implementation_status_checklist_artifact_missing_fields,
        "missing_field_count": len(
            implementation_status_checklist_artifact_missing_fields
        ),
        "fail_closed_reason": "implementation_status_artifact_incomplete",
        "required_value_mismatch_blocking_reason": (
            "implementation_status_checklist_artifact_complete_required_value_mismatch"
        ),
        "repair_command_id": "regenerate_package_with_implementation_status_artifact",
    }
    shareability_repair_command_detail_fields = (
        _shareability_repair_command_detail_fields()
    )
    shareability_repair_command_detail_fields_by_id = (
        _shareability_repair_command_detail_fields_by_id()
    )
    shareability_repair_command_contracts = (
        _shareability_repair_command_contracts()
    )
    labeler_route_authorization_runtime_checklist_gate = (
        _shareability_labeler_route_authorization_runtime_checklist_gate(
            labeler_route_authorization_runtime_checklist_contract
        )
    )
    shareability_contract_source_fields = (
        _shareability_compact_contract_source_fields()
    )
    shareability_contract = {
        "schema": "palette.web_labeling_handoff_shareability_contract.v1",
        "decision_source": "inspect_handoff_package",
        "top_level_field": "shareability_contract",
        "nested_field": "shareability.contract",
        "fields": list(_shareability_compact_contract_fields()),
        "field_count": len(_shareability_compact_contract_fields()),
        "source_fields": dict(shareability_contract_source_fields),
        "source_field_count": len(shareability_contract_source_fields),
        "safe_to_share": labeler_links_safe_to_share,
        "safe_to_share_required_value": True,
        "safe_to_share_matches_required_value": labeler_links_safe_to_share is True,
        "status": shareability_status,
        "operator_action": shareability_operator_action,
        "operator_action_required": operator_action_required_before_share,
        "requirement_met": shareability_requirement_met,
        "blocking_reason_ids": shareability_blocking_reason_id_list,
        "blocking_gate_ids": shareability_blocking_gate_ids,
        "repair_command_ids": [
            str(command.get("id") or "")
            for command in operator_repair_commands
            if str(command.get("id") or "")
        ],
        "repair_command_count": len(operator_repair_commands),
        "safe_share_gate": safe_share_gate,
        "safe_share_required_field": str(
            safe_share_fields.get("safe_share_required_inspection_field")
            or "labeler_links_safe_to_share"
        ),
        "safe_share_required_value": bool(
            safe_share_fields.get("safe_share_required_inspection_value", True)
        ),
        "safe_to_share_requires": _shareability_safe_to_share_requires(),
        "safe_share_launch_blocking_next_action_detail_fields": (
            _safe_share_next_action_detail_fields()
        ),
        "safe_share_launch_blocking_next_action_command_fields": (
            _safe_share_next_action_command_fields()
        ),
        "safe_share_external_launch_evidence_gap_fields": (
            _safe_share_external_launch_evidence_gap_field_names()
        ),
        **{
            field_name: safe_share_fields.get(field_name)
            for field_name in _safe_share_external_launch_evidence_gap_field_names()
        },
        "implementation_status_checklist_artifact_gate": (
            implementation_status_checklist_artifact_gate
        ),
        "launch_evidence_execution_checklist_summary": (
            launch_evidence_execution_checklist_summary
        ),
        "launch_evidence_execution_checklist_present": (
            launch_evidence_execution_checklist_summary["present"]
        ),
        "launch_evidence_execution_checklist_valid": (
            launch_evidence_execution_checklist_summary["valid"]
        ),
        "launch_evidence_execution_checklist_contract_present": (
            launch_evidence_execution_checklist_summary["checklist_contract_present"]
        ),
        "launch_evidence_execution_checklist_blocking_reason_id": (
            launch_evidence_execution_checklist_summary["blocking_reason_id"]
        ),
        "browser_mutation_target_contract": browser_mutation_target_contract,
        "direct_browser_start_contract": direct_browser_start_contract,
        "single_owner_package_contract": single_owner_package_contract,
        "labeler_route_authorization_runtime_checklist_gate": (
            labeler_route_authorization_runtime_checklist_gate
        ),
        "labeler_route_authorization_runtime_checklist_gate_met": bool(
            labeler_route_authorization_runtime_checklist_contract.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_required_fields": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_fields", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_required_values": dict(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_values", {}
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": int(
            labeler_route_authorization_runtime_checklist_contract.get("mismatch_count")
            or 0
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatch_users", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatches", []
            )
        ),
        "repair_command_detail_fields": list(shareability_repair_command_detail_fields),
        "repair_command_detail_fields_by_id": dict(
            shareability_repair_command_detail_fields_by_id
        ),
        "repair_command_contracts": dict(shareability_repair_command_contracts),
    }
    shareability_summary = {
        "schema": "palette.web_labeling_handoff_shareability.v1",
        "decision_source": "inspect_handoff_package",
        "contract": shareability_contract,
        "safe_to_share_requires": _shareability_safe_to_share_requires(),
        "safe_share_launch_blocking_next_action_detail_fields": (
            _safe_share_next_action_detail_fields()
        ),
        "safe_share_launch_blocking_next_action_command_fields": (
            _safe_share_next_action_command_fields()
        ),
        "safe_share_external_launch_evidence_gap_fields": (
            _safe_share_external_launch_evidence_gap_field_names()
        ),
        **{
            field_name: safe_share_fields.get(field_name)
            for field_name in _safe_share_external_launch_evidence_gap_field_names()
        },
        "safe_to_share": labeler_links_safe_to_share,
        "status": shareability_status,
        "operator_action": shareability_operator_action,
        "operator_action_required": operator_action_required_before_share,
        "required": bool(require_shareable),
        "requirement_met": shareability_requirement_met,
        "browser_mutation_target_contract": browser_mutation_target_contract,
        "browser_mutation_target_contract_met": bool(
            browser_mutation_target_contract.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_gate": (
            labeler_route_authorization_runtime_checklist_gate
        ),
        "labeler_route_authorization_runtime_checklist_gate_met": bool(
            labeler_route_authorization_runtime_checklist_contract.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_required_fields": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_fields", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_required_values": dict(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_values", {}
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": int(
            labeler_route_authorization_runtime_checklist_contract.get("mismatch_count")
            or 0
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatch_users", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatches", []
            )
        ),
        "browser_mutation_target_required_fields": list(
            browser_mutation_target_contract.get("required_fields", [])
        ),
        "browser_mutation_target_required_values": dict(
            browser_mutation_target_contract.get("required_values", {})
        ),
        "browser_mutation_target_mismatch_count": int(
            browser_mutation_target_contract.get("mismatch_count") or 0
        ),
        "browser_mutation_target_mismatch_users": list(
            browser_mutation_target_contract.get("mismatch_users", [])
        ),
        "browser_mutation_target_mismatches": list(
            browser_mutation_target_contract.get("mismatches", [])
        ),
        "direct_browser_start_contract": direct_browser_start_contract,
        "direct_browser_start_contract_met": bool(
            direct_browser_start_contract.get("met")
        ),
        "direct_browser_start_required_fields": list(
            direct_browser_start_contract.get("required_fields", [])
        ),
        "direct_browser_start_required_values": dict(
            direct_browser_start_contract.get("required_values", {})
        ),
        "direct_browser_start_mismatch_count": int(
            direct_browser_start_contract.get("mismatch_count") or 0
        ),
        "direct_browser_start_mismatch_users": list(
            direct_browser_start_contract.get("mismatch_users", [])
        ),
        "direct_browser_start_mismatches": list(
            direct_browser_start_contract.get("mismatches", [])
        ),
        "single_owner_package_contract": single_owner_package_contract,
        "single_owner_package_contract_met": bool(
            single_owner_package_contract.get("met")
        ),
        "single_owner_package_mismatch_count": int(
            single_owner_package_contract.get("mismatch_count") or 0
        ),
        "single_owner_package_mismatch_recording_ids": list(
            single_owner_package_contract.get("mismatch_recording_ids", [])
        ),
        "single_owner_package_duplicate_owners_by_recording": dict(
            single_owner_package_contract.get("duplicate_owners_by_recording", {})
        ),
        "labeler_route_authorization_runtime_checklist_gate": (
            labeler_route_authorization_runtime_checklist_gate
        ),
        "labeler_route_authorization_runtime_checklist_gate_met": bool(
            labeler_route_authorization_runtime_checklist_contract.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_required_fields": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_fields", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_required_values": dict(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_values", {}
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": int(
            labeler_route_authorization_runtime_checklist_contract.get("mismatch_count")
            or 0
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatch_users", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatches", []
            )
        ),
        "safe_share_gate": safe_share_gate,
        "shareability_contract": shareability_contract,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "failure_reasons": list(failure_reasons),
        "blocking_reason_ids": shareability_blocking_reason_id_list,
        "blocking_gate_ids": shareability_blocking_gate_ids,
        "operator_validation_gate_status_values": list(
            OPERATOR_VALIDATION_GATE_STATUS_VALUES
        ),
        "operator_validation_gate_ids": list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS),
        "operator_validation_gate_flat_field_suffixes": [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ],
        "required_pending_gate_ids": required_pending_gate_ids,
        "needs_review_gate_ids": needs_review_gate_ids,
        "unapproved_template_gate_ids": unapproved_template_gate_ids,
        "repair_command_ids": [
            str(command.get("id") or "")
            for command in operator_repair_commands
            if str(command.get("id") or "")
        ],
        "repair_commands": list(operator_repair_commands),
        "repair_command_detail_fields": list(shareability_repair_command_detail_fields),
        "repair_command_detail_fields_by_id": dict(
            shareability_repair_command_detail_fields_by_id
        ),
        "repair_command_contracts": dict(shareability_repair_command_contracts),
        "repair_command_gate_ids": operator_repair_command_gate_id_list,
        "repair_command_reason_ids": operator_repair_command_reason_id_list,
        "repair_command_count": len(operator_repair_commands),
        "repair_command_categories": dict(operator_repair_command_categories),
        "repair_command_categories_required": sorted(operator_repair_command_categories),
        "repair_commands_require_checksum_refresh": (
            operator_repair_commands_requiring_checksum_refresh > 0
        ),
        "repair_commands_requiring_checksum_refresh": (
            operator_repair_commands_requiring_checksum_refresh
        ),
        "operator_validation_command_template_summary": operator_validation_command_template_summary,
        "operator_validation_command_template_command_ids": (
            operator_validation_command_template_command_ids
        ),
        "operator_validation_command_template_gate_ids": (
            operator_validation_command_template_gate_ids
        ),
        "operator_validation_command_template_missing_command_gate_ids": (
            operator_validation_command_template_missing_gate_ids
        ),
        "operator_validation_command_template_template_backed_gate_ids": (
            operator_validation_command_template_template_backed_gate_ids
        ),
        "operator_validation_command_template_validation_checklist_gate_ids": (
            operator_validation_command_template_validation_checklist_gate_ids
        ),
        "operator_validation_command_template_apply_required_gate_ids": (
            operator_validation_command_template_apply_required_gate_ids
        ),
        "operator_validation_command_template_evidence_template_fields_by_gate_id": (
            operator_validation_command_template_evidence_template_fields_by_gate_id
        ),
        "operator_validation_command_template_evidence_template_paths_by_gate_id": (
            operator_validation_command_template_evidence_template_paths_by_gate_id
        ),
        "operator_validation_command_template_launch_evidence_collection_plan": (
            operator_validation_command_template_launch_evidence_collection_plan
        ),
        "operator_validation_command_template_launch_evidence_collection_plan_schema": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_plan_schema"
            )
            or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_step_count": int(
            operator_validation_command_templates.get(
                "launch_evidence_collection_step_count"
            )
            or 0
        ),
        "operator_validation_command_template_launch_evidence_collection_gate_ids": (
            operator_validation_command_template_launch_evidence_collection_gate_ids
        ),
        "operator_validation_command_template_launch_evidence_collection_record_command_ids": (
            operator_validation_command_template_launch_evidence_collection_record_command_ids
        ),
        "operator_validation_command_template_launch_evidence_collection_operator_only": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_operator_only",
                True,
            )
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_field": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_field"
            )
            or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_value": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_value"
            )
        ),
        "operator_validation_command_template_launch_evidence_collection_final_inspection_command": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_final_inspection_command"
            )
            or ""
        ),
        "operator_evidence_commands_summary": operator_evidence_commands_summary,
        "operator_evidence_commands_required": operator_evidence_commands_summary["required"],
        "operator_evidence_commands_present": operator_evidence_commands_summary["present"],
        "operator_evidence_commands_valid": operator_evidence_commands_summary["valid"],
        "operator_evidence_commands_operator_only_boundary_present": (
            operator_evidence_commands_summary["operator_only_boundary_present"]
        ),
        "operator_evidence_commands_operator_only_boundary_missing_phrases": (
            operator_evidence_commands_summary["operator_only_boundary_missing_phrases"]
        ),
        "operator_evidence_commands_blocking_reason_id": (
            operator_evidence_commands_summary["blocking_reason_id"]
        ),
        "launch_evidence_execution_checklist_summary": (
            launch_evidence_execution_checklist_summary
        ),
        "launch_evidence_execution_checklist_required": (
            launch_evidence_execution_checklist_summary["required"]
        ),
        "launch_evidence_execution_checklist_present": (
            launch_evidence_execution_checklist_summary["present"]
        ),
        "launch_evidence_execution_checklist_valid": (
            launch_evidence_execution_checklist_summary["valid"]
        ),
        "launch_evidence_execution_checklist_contract_present": (
            launch_evidence_execution_checklist_summary["checklist_contract_present"]
        ),
        "launch_evidence_execution_checklist_missing_phrases": (
            launch_evidence_execution_checklist_summary["checklist_missing_phrases"]
        ),
        "launch_evidence_execution_checklist_blocking_reason_id": (
            launch_evidence_execution_checklist_summary["blocking_reason_id"]
        ),
        "identity_personal_queue_evidence": identity_personal_queue_evidence_summary,
        "operator_validation_external_evidence_required": bool(
            validation_checklist.get("operator_validation_external_evidence_required")
        ),
        "operator_validation_external_evidence_required_gate_ids": (
            operator_validation_external_evidence_required_gate_ids
        ),
        "operator_validation_external_evidence_required_gate_count": int(
            validation_checklist.get(
                "operator_validation_external_evidence_required_gate_count"
            )
            or len(operator_validation_external_evidence_required_gate_ids)
        ),
        "operator_validation_external_evidence_template_fields_by_gate_id": (
            operator_validation_external_evidence_template_fields_by_gate_id
        ),
        "operator_validation_external_evidence_template_paths_by_gate_id": (
            operator_validation_external_evidence_template_paths_by_gate_id
        ),
        "operator_validation_checklist_only_required_gate_ids": (
            operator_validation_checklist_only_required_gate_ids
        ),
        "operator_validation_checklist_only_required_gate_count": int(
            validation_checklist.get(
                "operator_validation_checklist_only_required_gate_count"
            )
            or len(operator_validation_checklist_only_required_gate_ids)
        ),
        "implementation_status_checklist_artifact_present": bool(
            validation_checklist.get("implementation_status_artifact_present")
        ),
        "implementation_status_checklist_artifact_complete": bool(
            validation_checklist.get("implementation_status_artifact_complete")
        ),
        "implementation_status_checklist_artifact_complete_required_value": True,
        "implementation_status_checklist_artifact_complete_matches_required_value": (
            bool(validation_checklist.get("implementation_status_artifact_complete")) is True
        ),
        "implementation_status_checklist_artifact_gate": (
            implementation_status_checklist_artifact_gate
        ),
        "implementation_status_checklist_artifact_missing_fields": (
            implementation_status_checklist_artifact_missing_fields
        ),
        "implementation_status_checklist_artifact_missing_field_count": len(
            implementation_status_checklist_artifact_missing_fields
        ),
        "implementation_status_artifact": implementation_status_artifact,
        "implementation_status_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        "implementation_status_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        "implementation_status_flat_fields": list(_IMPLEMENTATION_STATUS_FLAT_FIELDS),
        "implementation_status_flat_field_count": _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
        "implementation_status": implementation_status_artifact["checklist_declared_path"],
        "implementation_status_file": implementation_status_artifact["file"],
        "implementation_status_required_path": implementation_status_artifact["required_path"],
        "implementation_status_declared_present": implementation_status_artifact[
            "declared_present"
        ],
        "implementation_status_declared_present_count": implementation_status_artifact[
            "declared_present_count"
        ],
        "implementation_status_present": implementation_status_artifact["present"],
        "implementation_status_is_launch_approval": implementation_status_artifact[
            "is_launch_approval"
        ],
        "implementation_status_operator_evidence_required_before_share": (
            implementation_status_artifact["operator_evidence_required_before_share"]
        ),
        "implementation_status_safe_share_gate": implementation_status_artifact[
            "safe_share_gate"
        ],
        "implementation_status_safe_share_required_inspection_field": (
            implementation_status_artifact["safe_share_required_inspection_field"]
        ),
        "implementation_status_safe_share_required_inspection_value": (
            implementation_status_artifact["safe_share_required_inspection_value"]
        ),
        "implementation_status_require_shareable_inspection_before_share": (
            implementation_status_artifact[
                "require_shareable_inspection_before_share"
            ]
        ),
        "implementation_status_matched_paths": implementation_status_artifact[
            "matched_paths"
        ],
        "labeler_entrypoint_summary": labeler_entrypoint_summary,
        "personalized_labeler_entrypoint": labeler_entrypoint_summary[
            "personalized_labeler_entrypoint"
        ],
        "personalized_labeler_entry_url_count": labeler_entrypoint_summary[
            "personalized_labeler_entry_url_count"
        ],
        "all_handoffs_have_personalized_entry_url": labeler_entrypoint_summary[
            "all_handoffs_have_personalized_entry_url"
        ],
    }
    seconds_until_earliest_expiration = None
    if earliest_expiration is not None:
        seconds_until_earliest_expiration = int((earliest_expiration - now).total_seconds())
    return {
        "ok": ok,
        "status": status,
        "failure_reasons": failure_reasons,
        "failure_actions": failure_actions,
        "operator_repair_commands": operator_repair_commands,
        "operator_repair_command_detail_fields": list(
            shareability_repair_command_detail_fields
        ),
        "operator_repair_command_detail_fields_by_id": dict(
            shareability_repair_command_detail_fields_by_id
        ),
        "operator_repair_command_contracts": dict(
            shareability_repair_command_contracts
        ),
        "operator_repair_command_count": len(operator_repair_commands),
        "operator_repair_command_categories": operator_repair_command_categories,
        "operator_repair_command_gate_ids": operator_repair_command_gate_id_list,
        "operator_repair_command_reason_ids": operator_repair_command_reason_id_list,
        "operator_repair_commands_requiring_checksum_refresh": (
            operator_repair_commands_requiring_checksum_refresh
        ),
        "operator_validation_command_template_summary": operator_validation_command_template_summary,
        "operator_validation_command_template_command_ids": (
            operator_validation_command_template_command_ids
        ),
        "operator_validation_command_template_gate_ids": (
            operator_validation_command_template_gate_ids
        ),
        "operator_validation_command_template_missing_command_gate_ids": (
            operator_validation_command_template_missing_gate_ids
        ),
        "operator_validation_command_template_template_backed_gate_ids": (
            operator_validation_command_template_template_backed_gate_ids
        ),
        "operator_validation_command_template_validation_checklist_gate_ids": (
            operator_validation_command_template_validation_checklist_gate_ids
        ),
        "operator_validation_command_template_apply_required_gate_ids": (
            operator_validation_command_template_apply_required_gate_ids
        ),
        "operator_validation_command_template_evidence_template_fields_by_gate_id": (
            operator_validation_command_template_evidence_template_fields_by_gate_id
        ),
        "operator_validation_command_template_evidence_template_paths_by_gate_id": (
            operator_validation_command_template_evidence_template_paths_by_gate_id
        ),
        "operator_validation_command_template_launch_evidence_collection_plan": (
            operator_validation_command_template_launch_evidence_collection_plan
        ),
        "operator_validation_command_template_launch_evidence_collection_plan_schema": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_plan_schema"
            )
            or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_step_count": int(
            operator_validation_command_templates.get(
                "launch_evidence_collection_step_count"
            )
            or 0
        ),
        "operator_validation_command_template_launch_evidence_collection_gate_ids": (
            operator_validation_command_template_launch_evidence_collection_gate_ids
        ),
        "operator_validation_command_template_launch_evidence_collection_record_command_ids": (
            operator_validation_command_template_launch_evidence_collection_record_command_ids
        ),
        "operator_validation_command_template_launch_evidence_collection_operator_only": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_operator_only",
                True,
            )
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_field": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_field"
            )
            or ""
        ),
        "operator_validation_command_template_launch_evidence_collection_required_final_value": bool(
            operator_validation_command_templates.get(
                "launch_evidence_collection_required_final_value"
            )
        ),
        "operator_validation_command_template_launch_evidence_collection_final_inspection_command": str(
            operator_validation_command_templates.get(
                "launch_evidence_collection_final_inspection_command"
            )
            or ""
        ),
        "operator_evidence_commands_summary": operator_evidence_commands_summary,
        "operator_evidence_commands_required": operator_evidence_commands_summary["required"],
        "operator_evidence_commands_present": operator_evidence_commands_summary["present"],
        "operator_evidence_commands_valid": operator_evidence_commands_summary["valid"],
        "operator_evidence_commands_operator_only_boundary_present": (
            operator_evidence_commands_summary["operator_only_boundary_present"]
        ),
        "operator_evidence_commands_operator_only_boundary_missing_phrases": (
            operator_evidence_commands_summary["operator_only_boundary_missing_phrases"]
        ),
        "operator_evidence_commands_blocking_reason_id": (
            operator_evidence_commands_summary["blocking_reason_id"]
        ),
        "launch_evidence_execution_checklist_summary": (
            launch_evidence_execution_checklist_summary
        ),
        "launch_evidence_execution_checklist_required": (
            launch_evidence_execution_checklist_summary["required"]
        ),
        "launch_evidence_execution_checklist_present": (
            launch_evidence_execution_checklist_summary["present"]
        ),
        "launch_evidence_execution_checklist_valid": (
            launch_evidence_execution_checklist_summary["valid"]
        ),
        "launch_evidence_execution_checklist_contract_present": (
            launch_evidence_execution_checklist_summary["checklist_contract_present"]
        ),
        "launch_evidence_execution_checklist_missing_phrases": (
            launch_evidence_execution_checklist_summary["checklist_missing_phrases"]
        ),
        "launch_evidence_execution_checklist_blocking_reason_id": (
            launch_evidence_execution_checklist_summary["blocking_reason_id"]
        ),
        "labeler_entrypoint_summary": labeler_entrypoint_summary,
        "personalized_labeler_entrypoint": labeler_entrypoint_summary[
            "personalized_labeler_entrypoint"
        ],
        "personal_dataset_queue_page_path": labeler_entrypoint_summary[
            "personal_dataset_queue_page_path"
        ],
        "personal_dataset_queue_url": labeler_entrypoint_summary["personal_dataset_queue_url"],
        "personal_work_page_path": labeler_entrypoint_summary["personal_work_page_path"],
        "personal_work_url": labeler_entrypoint_summary["personal_work_url"],
        "personalized_labeler_entry_url_count": labeler_entrypoint_summary[
            "personalized_labeler_entry_url_count"
        ],
        "all_handoffs_have_personalized_entry_url": labeler_entrypoint_summary[
            "all_handoffs_have_personalized_entry_url"
        ],
        "personalized_labeler_entry_url_by_user": labeler_entrypoint_summary[
            "personalized_labeler_entry_url_by_user"
        ],
        "canonical_dataset_queue_url_by_user": labeler_entrypoint_summary[
            "canonical_dataset_queue_url_by_user"
        ],
        "labeler_links_safe_to_share": labeler_links_safe_to_share,
        "shareability_status": shareability_status,
        "shareability_operator_action": shareability_operator_action,
        "operator_action_required_before_share": operator_action_required_before_share,
        "browser_mutation_target_contract": browser_mutation_target_contract,
        "browser_mutation_target_contract_met": bool(
            browser_mutation_target_contract.get("met")
        ),
        "browser_mutation_target_required_fields": list(
            browser_mutation_target_contract.get("required_fields", [])
        ),
        "browser_mutation_target_required_values": dict(
            browser_mutation_target_contract.get("required_values", {})
        ),
        "browser_mutation_target_mismatch_count": int(
            browser_mutation_target_contract.get("mismatch_count") or 0
        ),
        "browser_mutation_target_mismatch_users": list(
            browser_mutation_target_contract.get("mismatch_users", [])
        ),
        "browser_mutation_target_mismatches": list(
            browser_mutation_target_contract.get("mismatches", [])
        ),
        "direct_browser_start_contract": direct_browser_start_contract,
        "direct_browser_start_contract_met": bool(
            direct_browser_start_contract.get("met")
        ),
        "direct_browser_start_required_fields": list(
            direct_browser_start_contract.get("required_fields", [])
        ),
        "direct_browser_start_required_values": dict(
            direct_browser_start_contract.get("required_values", {})
        ),
        "direct_browser_start_mismatch_count": int(
            direct_browser_start_contract.get("mismatch_count") or 0
        ),
        "direct_browser_start_mismatch_users": list(
            direct_browser_start_contract.get("mismatch_users", [])
        ),
        "direct_browser_start_mismatches": list(
            direct_browser_start_contract.get("mismatches", [])
        ),
        "single_owner_package_contract": single_owner_package_contract,
        "single_owner_package_contract_met": bool(
            single_owner_package_contract.get("met")
        ),
        "single_owner_package_mismatch_count": int(
            single_owner_package_contract.get("mismatch_count") or 0
        ),
        "single_owner_package_mismatch_recording_ids": list(
            single_owner_package_contract.get("mismatch_recording_ids", [])
        ),
        "single_owner_package_duplicate_owners_by_recording": dict(
            single_owner_package_contract.get("duplicate_owners_by_recording", {})
        ),
        "labeler_route_authorization_runtime_checklist_gate": (
            labeler_route_authorization_runtime_checklist_gate
        ),
        "labeler_route_authorization_runtime_checklist_gate_met": bool(
            labeler_route_authorization_runtime_checklist_contract.get("met")
        ),
        "labeler_route_authorization_runtime_checklist_required_fields": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_fields", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_required_values": dict(
            labeler_route_authorization_runtime_checklist_contract.get(
                "required_values", {}
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_count": int(
            labeler_route_authorization_runtime_checklist_contract.get("mismatch_count")
            or 0
        ),
        "labeler_route_authorization_runtime_checklist_mismatch_users": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatch_users", []
            )
        ),
        "labeler_route_authorization_runtime_checklist_mismatches": list(
            labeler_route_authorization_runtime_checklist_contract.get(
                "mismatches", []
            )
        ),
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "shareability_blocking_gate_ids": shareability_blocking_gate_ids,
        "shareability_blocking_reason_ids": shareability_blocking_reason_id_list,
        "operator_validation_gate_status_values": list(
            OPERATOR_VALIDATION_GATE_STATUS_VALUES
        ),
        "operator_validation_gate_ids": list(DEFAULT_OPERATOR_VALIDATION_GATE_IDS),
        "operator_validation_gate_flat_field_suffixes": [
            "status",
            "pending",
            "missing_evidence",
            "needs_review",
            "passed",
        ],
        "shareability_required": bool(require_shareable),
        "shareability_requirement_met": shareability_requirement_met,
        "identity_personal_queue_evidence_status": (
            identity_personal_queue_evidence_summary["status"]
        ),
        "identity_personal_queue_evidence_ready_count": identity_personal_queue_evidence_summary[
            "ready_count"
        ],
        "identity_personal_queue_evidence_missing_count": identity_personal_queue_evidence_summary[
            "missing_count"
        ],
        "identity_personal_queue_evidence_ready_users": identity_personal_queue_evidence_ready_users,
        "identity_personal_queue_evidence_missing_users": identity_personal_queue_evidence_missing_users,
        "identity_personal_queue_evidence_missing_fields_by_user": (
            identity_personal_queue_evidence_missing_fields_by_user
        ),
        "identity_all_users_have_personal_queue_evidence": identity_personal_queue_evidence_summary[
            "all_users_have_personal_queue_evidence"
        ],
        "operator_validation_external_evidence_required": bool(
            validation_checklist.get("operator_validation_external_evidence_required")
        ),
        "operator_validation_external_evidence_required_gate_ids": (
            operator_validation_external_evidence_required_gate_ids
        ),
        "operator_validation_external_evidence_required_gate_count": int(
            validation_checklist.get(
                "operator_validation_external_evidence_required_gate_count"
            )
            or len(operator_validation_external_evidence_required_gate_ids)
        ),
        "operator_validation_external_evidence_template_fields_by_gate_id": (
            operator_validation_external_evidence_template_fields_by_gate_id
        ),
        "operator_validation_external_evidence_template_paths_by_gate_id": (
            operator_validation_external_evidence_template_paths_by_gate_id
        ),
        "operator_validation_checklist_only_required_gate_ids": (
            operator_validation_checklist_only_required_gate_ids
        ),
        "operator_validation_checklist_only_required_gate_count": int(
            validation_checklist.get(
                "operator_validation_checklist_only_required_gate_count"
            )
            or len(operator_validation_checklist_only_required_gate_ids)
        ),
        "implementation_status_checklist_artifact_present": bool(
            validation_checklist.get("implementation_status_artifact_present")
        ),
        "implementation_status_checklist_artifact_complete": bool(
            validation_checklist.get("implementation_status_artifact_complete")
        ),
        "implementation_status_checklist_artifact_complete_required_value": True,
        "implementation_status_checklist_artifact_complete_matches_required_value": (
            bool(validation_checklist.get("implementation_status_artifact_complete")) is True
        ),
        "implementation_status_checklist_artifact_gate": (
            implementation_status_checklist_artifact_gate
        ),
        "implementation_status_checklist_artifact_missing_fields": (
            implementation_status_checklist_artifact_missing_fields
        ),
        "implementation_status_checklist_artifact_missing_field_count": len(
            implementation_status_checklist_artifact_missing_fields
        ),
        "implementation_status_artifact": implementation_status_artifact,
        "implementation_status_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        "implementation_status_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        "implementation_status_flat_fields": list(_IMPLEMENTATION_STATUS_FLAT_FIELDS),
        "implementation_status_flat_field_count": _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
        "implementation_status": implementation_status_artifact["checklist_declared_path"],
        "implementation_status_file": implementation_status_artifact["file"],
        "implementation_status_required_path": implementation_status_artifact["required_path"],
        "implementation_status_declared_present": implementation_status_artifact[
            "declared_present"
        ],
        "implementation_status_declared_present_count": implementation_status_artifact[
            "declared_present_count"
        ],
        "implementation_status_present": implementation_status_artifact["present"],
        "implementation_status_is_launch_approval": implementation_status_artifact[
            "is_launch_approval"
        ],
        "implementation_status_operator_evidence_required_before_share": (
            implementation_status_artifact["operator_evidence_required_before_share"]
        ),
        "implementation_status_safe_share_gate": implementation_status_artifact[
            "safe_share_gate"
        ],
        "implementation_status_safe_share_required_inspection_field": (
            implementation_status_artifact["safe_share_required_inspection_field"]
        ),
        "implementation_status_safe_share_required_inspection_value": (
            implementation_status_artifact["safe_share_required_inspection_value"]
        ),
        "implementation_status_require_shareable_inspection_before_share": (
            implementation_status_artifact[
                "require_shareable_inspection_before_share"
            ]
        ),
        "implementation_status_matched_paths": implementation_status_artifact[
            "matched_paths"
        ],
        "shareability_contract": shareability_contract,
        "shareability": shareability_summary,
        "path": str(path),
        "kind": kind,
        "checked_at_utc": now.isoformat(),
        "batch_generated_at_utc": (index or {}).get("generated_at_utc") if index else None,
        "readiness_ok": readiness_ok,
        "handoffs_ok": handoffs_ok,
        "handoff_store_checks_ok": handoff_store_checks_ok,
        "assignment_freshness": assignment_freshness,
        "earliest_links_expire_at_utc": earliest_expiration.isoformat() if earliest_expiration else None,
        "seconds_until_earliest_expiration": seconds_until_earliest_expiration,
        "counts": aggregate_counts,
        "checksum_verification": checksum_verification,
        "validation_log": validation_log,
        "validation_checklist": validation_checklist,
        "operator_evidence_commands": operator_evidence_commands,
        "launch_evidence_execution_checklist": launch_evidence_execution_checklist,
        "handoffs": handoffs,
    }
