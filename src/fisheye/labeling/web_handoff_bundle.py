"""User-handoff bundle orchestration for web labeling launch artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BUNDLE_DEPENDENCY_NAMES = (
    'DASHBOARD_PATH',
    'DATASET_QUEUE_PATH',
    'LABELER_START_TASK_STATES',
    'LABELING_HOME_PATH',
    'PERSONAL_DATASET_QUEUE_PATH',
    'PERSONAL_WORK_PATH',
    '_add_direct_start_contracts_to_work_tasks',
    '_add_payload_contract_compact_fields',
    '_add_work_summary_fields',
    '_assignment_ownership_contract_fields',
    '_assignment_ownership_contract_policy',
    '_assignment_ownership_integrity',
    '_assignment_ownership_policy',
    '_browser_mutation_write_policy',
    '_browser_mutation_write_runtime_checklist',
    '_browser_response_security_policy',
    '_browser_signed_link_policy',
    '_browser_task_state_policy',
    '_browser_workflow_capabilities',
    '_check_directory_zip_output',
    '_count_recordings_without_open_tasks',
    '_count_recordings_without_open_tasks_by_reason',
    '_count_redacted_summary_fields',
    '_dashboard_operator_validation_fields',
    '_dashboard_url_for_base',
    '_dashboard_url_for_expected_user',
    '_dataset_queue_direct_start_policy',
    '_dataset_queue_state',
    '_dataset_queue_url_for_base',
    '_effective_signed_link_ttl_seconds',
    '_first_dataset_queue_url',
    '_handoff_assignment_snapshot_from_work',
    '_handoff_browser_mutation_write_fields',
    '_handoff_browser_response_security_fields',
    '_handoff_entry_artifact_fields',
    '_handoff_labeler_route_authorization_fields',
    '_handoff_labeler_safety_fields',
    '_handoff_mutation_audit_fields',
    '_handoff_operator_recovery_fields',
    '_handoff_ready_to_send',
    '_handoff_sendability_actions',
    '_handoff_sendability_reasons',
    '_handoff_sendability_summary',
    '_handoff_session_guard_fields',
    '_handoff_signed_link_policy_fields',
    '_handoff_store_checks_ok_for_user',
    '_handoff_task_state_policy_fields',
    '_handoff_zarr_backup_fields',
    '_identity_probe_url_for_base',
    '_known_labeler_status',
    '_labeler_landing_url_for_base',
    '_labeler_route_authorization_policy',
    '_labeler_route_authorization_runtime_checklist',
    '_labeler_safety_policy',
    '_labeler_work_completion_fields',
    '_labeling_home_url_for_base',
    '_mutation_audit_policy',
    '_operator_authorization_policy',
    '_operator_recovery_policy',
    '_operator_validation_command_templates',
    '_operator_validation_gate_flat_fields',
    '_operator_validation_invitation_fields',
    '_operator_validation_public_fields',
    '_operator_validation_visibility_policy',
    '_personalized_launch_readiness_summary',
    '_public_reassignment_session_safety_fields',
    '_queue_first_entry_contract_policy',
    '_reassignment_session_safety_flat_fields',
    '_recordings_without_open_tasks_actions',
    '_runtime_operator_validation_gate_cli_policy',
    '_runtime_operator_validation_gate_cli_policy_fields',
    '_safe_share_checklist_gate_status_fields_from_operator_validation',
    '_safe_share_gate_flat_fields',
    '_safe_share_gate_policy',
    '_session_guard_policy',
    '_signed_task_link_token_info',
    '_store_consistency_report',
    '_user_handoff_paths',
    '_web_labeling_validation_checklist_payload',
    '_work_dataset_queue_summary',
    '_work_progress_summary',
    '_work_recording_ids',
    '_write_directory_zip',
    '_write_user_handoff_html_index',
    '_write_user_handoff_message',
    '_write_user_handoff_quickstart',
    '_write_user_handoff_validation_checklist',
    '_write_user_handoff_validation_log',
    '_zarr_backup_policy'
)


def _require_bundle_dependencies(dependencies: Mapping[str, object]) -> dict[str, Any]:
    missing = [name for name in _BUNDLE_DEPENDENCY_NAMES if name not in dependencies]
    if missing:
        raise KeyError(f"missing user handoff bundle dependencies: {', '.join(missing)}")
    return {name: dependencies[name] for name in _BUNDLE_DEPENDENCY_NAMES}


def _write_user_handoff_bundle(
    *,
    store: LabelingStore,
    store_path: Path,
    user: str,
    output_dir: Path,
    secret: str,
    base_url: str | None,
    ttl_seconds: int,
    include_completed: bool,
    overwrite: bool,
    zip_output: Path | None = None,
    dependencies: Mapping[str, object],
) -> dict[str, object]:
    bundle_dependencies = _require_bundle_dependencies(dependencies)
    DASHBOARD_PATH = bundle_dependencies['DASHBOARD_PATH']
    DATASET_QUEUE_PATH = bundle_dependencies['DATASET_QUEUE_PATH']
    LABELER_START_TASK_STATES = bundle_dependencies['LABELER_START_TASK_STATES']
    LABELING_HOME_PATH = bundle_dependencies['LABELING_HOME_PATH']
    PERSONAL_DATASET_QUEUE_PATH = bundle_dependencies['PERSONAL_DATASET_QUEUE_PATH']
    PERSONAL_WORK_PATH = bundle_dependencies['PERSONAL_WORK_PATH']
    _add_direct_start_contracts_to_work_tasks = bundle_dependencies['_add_direct_start_contracts_to_work_tasks']
    _add_payload_contract_compact_fields = bundle_dependencies['_add_payload_contract_compact_fields']
    _add_work_summary_fields = bundle_dependencies['_add_work_summary_fields']
    _assignment_ownership_contract_fields = bundle_dependencies['_assignment_ownership_contract_fields']
    _assignment_ownership_contract_policy = bundle_dependencies['_assignment_ownership_contract_policy']
    _assignment_ownership_integrity = bundle_dependencies['_assignment_ownership_integrity']
    _assignment_ownership_policy = bundle_dependencies['_assignment_ownership_policy']
    _browser_mutation_write_policy = bundle_dependencies['_browser_mutation_write_policy']
    _browser_mutation_write_runtime_checklist = bundle_dependencies['_browser_mutation_write_runtime_checklist']
    _browser_response_security_policy = bundle_dependencies['_browser_response_security_policy']
    _browser_signed_link_policy = bundle_dependencies['_browser_signed_link_policy']
    _browser_task_state_policy = bundle_dependencies['_browser_task_state_policy']
    _browser_workflow_capabilities = bundle_dependencies['_browser_workflow_capabilities']
    _check_directory_zip_output = bundle_dependencies['_check_directory_zip_output']
    _count_recordings_without_open_tasks = bundle_dependencies['_count_recordings_without_open_tasks']
    _count_recordings_without_open_tasks_by_reason = bundle_dependencies['_count_recordings_without_open_tasks_by_reason']
    _count_redacted_summary_fields = bundle_dependencies['_count_redacted_summary_fields']
    _dashboard_operator_validation_fields = bundle_dependencies['_dashboard_operator_validation_fields']
    _dashboard_url_for_base = bundle_dependencies['_dashboard_url_for_base']
    _dashboard_url_for_expected_user = bundle_dependencies['_dashboard_url_for_expected_user']
    _dataset_queue_direct_start_policy = bundle_dependencies['_dataset_queue_direct_start_policy']
    _dataset_queue_state = bundle_dependencies['_dataset_queue_state']
    _dataset_queue_url_for_base = bundle_dependencies['_dataset_queue_url_for_base']
    _effective_signed_link_ttl_seconds = bundle_dependencies['_effective_signed_link_ttl_seconds']
    _first_dataset_queue_url = bundle_dependencies['_first_dataset_queue_url']
    _handoff_assignment_snapshot_from_work = bundle_dependencies['_handoff_assignment_snapshot_from_work']
    _handoff_browser_mutation_write_fields = bundle_dependencies['_handoff_browser_mutation_write_fields']
    _handoff_browser_response_security_fields = bundle_dependencies['_handoff_browser_response_security_fields']
    _handoff_entry_artifact_fields = bundle_dependencies['_handoff_entry_artifact_fields']
    _handoff_labeler_route_authorization_fields = bundle_dependencies['_handoff_labeler_route_authorization_fields']
    _handoff_labeler_safety_fields = bundle_dependencies['_handoff_labeler_safety_fields']
    _handoff_mutation_audit_fields = bundle_dependencies['_handoff_mutation_audit_fields']
    _handoff_operator_recovery_fields = bundle_dependencies['_handoff_operator_recovery_fields']
    _handoff_ready_to_send = bundle_dependencies['_handoff_ready_to_send']
    _handoff_sendability_actions = bundle_dependencies['_handoff_sendability_actions']
    _handoff_sendability_reasons = bundle_dependencies['_handoff_sendability_reasons']
    _handoff_sendability_summary = bundle_dependencies['_handoff_sendability_summary']
    _handoff_session_guard_fields = bundle_dependencies['_handoff_session_guard_fields']
    _handoff_signed_link_policy_fields = bundle_dependencies['_handoff_signed_link_policy_fields']
    _handoff_store_checks_ok_for_user = bundle_dependencies['_handoff_store_checks_ok_for_user']
    _handoff_task_state_policy_fields = bundle_dependencies['_handoff_task_state_policy_fields']
    _handoff_zarr_backup_fields = bundle_dependencies['_handoff_zarr_backup_fields']
    _identity_probe_url_for_base = bundle_dependencies['_identity_probe_url_for_base']
    _known_labeler_status = bundle_dependencies['_known_labeler_status']
    _labeler_landing_url_for_base = bundle_dependencies['_labeler_landing_url_for_base']
    _labeler_route_authorization_policy = bundle_dependencies['_labeler_route_authorization_policy']
    _labeler_route_authorization_runtime_checklist = bundle_dependencies['_labeler_route_authorization_runtime_checklist']
    _labeler_safety_policy = bundle_dependencies['_labeler_safety_policy']
    _labeler_work_completion_fields = bundle_dependencies['_labeler_work_completion_fields']
    _labeling_home_url_for_base = bundle_dependencies['_labeling_home_url_for_base']
    _mutation_audit_policy = bundle_dependencies['_mutation_audit_policy']
    _operator_authorization_policy = bundle_dependencies['_operator_authorization_policy']
    _operator_recovery_policy = bundle_dependencies['_operator_recovery_policy']
    _operator_validation_command_templates = bundle_dependencies['_operator_validation_command_templates']
    _operator_validation_gate_flat_fields = bundle_dependencies['_operator_validation_gate_flat_fields']
    _operator_validation_invitation_fields = bundle_dependencies['_operator_validation_invitation_fields']
    _operator_validation_public_fields = bundle_dependencies['_operator_validation_public_fields']
    _operator_validation_visibility_policy = bundle_dependencies['_operator_validation_visibility_policy']
    _personalized_launch_readiness_summary = bundle_dependencies['_personalized_launch_readiness_summary']
    _public_reassignment_session_safety_fields = bundle_dependencies['_public_reassignment_session_safety_fields']
    _queue_first_entry_contract_policy = bundle_dependencies['_queue_first_entry_contract_policy']
    _reassignment_session_safety_flat_fields = bundle_dependencies['_reassignment_session_safety_flat_fields']
    _recordings_without_open_tasks_actions = bundle_dependencies['_recordings_without_open_tasks_actions']
    _runtime_operator_validation_gate_cli_policy = bundle_dependencies['_runtime_operator_validation_gate_cli_policy']
    _runtime_operator_validation_gate_cli_policy_fields = bundle_dependencies['_runtime_operator_validation_gate_cli_policy_fields']
    _safe_share_checklist_gate_status_fields_from_operator_validation = bundle_dependencies['_safe_share_checklist_gate_status_fields_from_operator_validation']
    _safe_share_gate_flat_fields = bundle_dependencies['_safe_share_gate_flat_fields']
    _safe_share_gate_policy = bundle_dependencies['_safe_share_gate_policy']
    _session_guard_policy = bundle_dependencies['_session_guard_policy']
    _signed_task_link_token_info = bundle_dependencies['_signed_task_link_token_info']
    _store_consistency_report = bundle_dependencies['_store_consistency_report']
    _user_handoff_paths = bundle_dependencies['_user_handoff_paths']
    _web_labeling_validation_checklist_payload = bundle_dependencies['_web_labeling_validation_checklist_payload']
    _work_dataset_queue_summary = bundle_dependencies['_work_dataset_queue_summary']
    _work_progress_summary = bundle_dependencies['_work_progress_summary']
    _work_recording_ids = bundle_dependencies['_work_recording_ids']
    _write_directory_zip = bundle_dependencies['_write_directory_zip']
    _write_user_handoff_html_index = bundle_dependencies['_write_user_handoff_html_index']
    _write_user_handoff_message = bundle_dependencies['_write_user_handoff_message']
    _write_user_handoff_quickstart = bundle_dependencies['_write_user_handoff_quickstart']
    _write_user_handoff_validation_checklist = bundle_dependencies['_write_user_handoff_validation_checklist']
    _write_user_handoff_validation_log = bundle_dependencies['_write_user_handoff_validation_log']
    _zarr_backup_policy = bundle_dependencies['_zarr_backup_policy']

    if zip_output is not None:
        _check_directory_zip_output(output_dir, zip_output, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_paths = _user_handoff_paths(output_dir)
    existing_paths = [path for path in handoff_paths.values() if path.exists()]
    if zip_output is not None and zip_output.exists():
        existing_paths.append(zip_output)
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing user handoff files: "
            + ", ".join(str(path) for path in existing_paths)
        )

    normalized_user = str(user)
    normalized_base_url = str(base_url or "").rstrip("/")
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    link_issued_at_utc: list[str] = []
    link_expires_at_utc: list[str] = []
    check_report = _store_consistency_report(store)
    work = store.task_summary_for_user(normalized_user, include_completed=include_completed)
    work["include_completed"] = include_completed
    _add_work_summary_fields(
        work,
        reassignment_session_safety=check_report.get("reassignment_session_safety", {}),
    )
    work["browser_mutation_write_policy"] = _browser_mutation_write_policy()
    work["browser_mutation_write_checklist"] = _browser_mutation_write_runtime_checklist()
    work["dataset_queue_direct_start_policy"] = _dataset_queue_direct_start_policy()
    work["runtime_operator_validation_gate_cli_policy"] = _runtime_operator_validation_gate_cli_policy()
    work["single_owner_policy"] = _assignment_ownership_policy()
    work["assignment_ownership_integrity"] = check_report.get(
        "assignment_ownership_integrity",
        {},
    )
    single_owner_assignment_contract = store.single_owner_assignment_contract()
    work["single_owner_assignment_contract"] = single_owner_assignment_contract
    _add_payload_contract_compact_fields(work)
    _add_direct_start_contracts_to_work_tasks(
        work,
        expected_user=normalized_user,
        reassignment_session_safety=check_report.get("reassignment_session_safety", {}),
    )
    tasks = [
        task
        for task in store.list_tasks(
            assignee_user=normalized_user,
            include_completed=include_completed,
        )
        if str(task.get("assignment_status") or "") == "active"
    ]
    links: list[dict[str, object]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        row_warnings: list[dict[str, object]] = []
        if not normalized_base_url:
            row_warnings.append(
                {
                    "code": "missing_base_url",
                    "details": "Generated URL is service-relative. Regenerate the handoff with --base-url before sharing with a labeler.",
                }
            )
        if str(task.get("state") or "") == "complete":
            row_warnings.append(
                {
                    "code": "task_completed",
                    "task_id": task_id,
                    "recording_id": task.get("recording_id"),
                    "details": "Task is complete, so the signed link will not open a new labeling session unless the task is reopened.",
                }
            )
        elif str(task.get("state") or "") not in LABELER_START_TASK_STATES:
            row_warnings.append(
                {
                    "code": "task_not_startable",
                    "task_id": task_id,
                    "recording_id": task.get("recording_id"),
                    "state": task.get("state"),
                    "startable_task_states": list(LABELER_START_TASK_STATES),
                    "details": "Task is not in a startable labeling state, so the signed link will not open a new labeling session.",
                }
            )
        token_info = _signed_task_link_token_info(
            task_id=task_id,
            secret=secret,
            ttl_seconds=ttl_seconds,
            expected_user=str(task.get("assignee_user") or ""),
        )
        token = str(token_info["token"])
        link_issued_at_utc.append(str(token_info["issued_at_utc"]))
        link_expires_at_utc.append(str(token_info["expires_at_utc"]))
        path = f"/t/{token}"
        links.append(
            {
                "task_id": task_id,
                "recording_id": task.get("recording_id"),
                "assignee_user": task.get("assignee_user"),
                "expected_user": token_info.get("expected_user") or task.get("assignee_user"),
                "workflow_kind": task.get("workflow_kind"),
                "state": task.get("state"),
                "startable_task_states": list(LABELER_START_TASK_STATES),
                "title": task.get("title"),
                "issued_at_utc": token_info["issued_at_utc"],
                "expires_in_seconds": token_info["ttl_seconds"],
                "expires_at_utc": token_info["expires_at_utc"],
                "path": path,
                "url": f"{normalized_base_url}{path}" if normalized_base_url else path,
                "url_is_absolute": bool(normalized_base_url),
                "task_launchable": not any(
                    str(warning.get("code") or "").startswith("task_")
                    for warning in row_warnings
                ),
                "ready_to_share": bool(normalized_base_url) and not row_warnings,
                "shareability_warnings": row_warnings,
            }
        )
    if link_issued_at_utc:
        generated_at_utc = min(link_issued_at_utc)
    links_expire_at_utc = min(link_expires_at_utc) if link_expires_at_utc else None
    known_user_status = _known_labeler_status(store, normalized_user)
    labeler_route_authorization_policy = _labeler_route_authorization_policy()
    assignment_ownership_integrity = _assignment_ownership_integrity(
        store.list_assignments(status=None),
        schema_integrity=store.assignment_schema_integrity(),
    )
    assignment_ownership_contract = _assignment_ownership_contract_policy(
        _assignment_ownership_policy(),
        assignment_ownership_integrity,
        store_single_owner_contract=single_owner_assignment_contract,
    )
    work["single_owner_assignment_contract"] = single_owner_assignment_contract
    work["assignment_ownership_contract"] = assignment_ownership_contract
    work.update(_assignment_ownership_contract_fields(assignment_ownership_contract))
    work["single_owner_policy_contract_met"] = bool(
        assignment_ownership_contract.get("ready")
    ) and int(assignment_ownership_integrity.get("duplicate_active_owner_count") or 0) == 0
    labeler_route_authorization_checklist = _labeler_route_authorization_runtime_checklist(
        policy=labeler_route_authorization_policy,
        user=normalized_user,
        expected_user=normalized_user,
        known_user_status=known_user_status,
        assignment_ownership_contract=assignment_ownership_contract,
    )
    work["labeler_route_authorization_policy"] = labeler_route_authorization_policy
    work["labeler_route_authorization_checklist"] = labeler_route_authorization_checklist
    operator_validation_visibility_policy = _operator_validation_visibility_policy()
    initial_operator_validation_public_fields = _operator_validation_public_fields(
        _dashboard_operator_validation_fields()
    )
    initial_safe_share_gate = _safe_share_gate_policy()
    initial_safe_share_fields = _safe_share_gate_flat_fields(initial_safe_share_gate)
    initial_safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        initial_operator_validation_public_fields,
        safe_share_gate=initial_safe_share_gate,
    )
    initial_operator_validation_command_templates = _operator_validation_command_templates(
        initial_operator_validation_public_fields.get(
            "operator_validation_required_missing_evidence_gate_ids"
        )
        if isinstance(
            initial_operator_validation_public_fields.get(
                "operator_validation_required_missing_evidence_gate_ids"
            ),
            list,
        )
        else None
    )
    work.update(initial_operator_validation_public_fields)
    work["safe_share_gate"] = initial_safe_share_gate
    work.update(initial_safe_share_fields)
    work.update(initial_safe_share_checklist_fields)
    work["operator_validation_command_templates"] = initial_operator_validation_command_templates
    work["operator_validation_visibility_policy"] = operator_validation_visibility_policy
    public_reassignment_session_safety = (
        work.get("reassignment_session_safety")
        if isinstance(work.get("reassignment_session_safety"), Mapping)
        else _public_reassignment_session_safety_fields(
            check_report.get("reassignment_session_safety", {}),
            recording_ids=_work_recording_ids(work),
        )
    )
    work["reassignment_session_safety"] = public_reassignment_session_safety
    work_payload = {
        "ok": True,
        "store_path": str(store_path),
        "include_completed": include_completed,
        "known_user_status": known_user_status,
        "labeler_route_authorization_policy": labeler_route_authorization_policy,
        "labeler_route_authorization_checklist": labeler_route_authorization_checklist,
        "operator_validation_visibility_policy": operator_validation_visibility_policy,
        "operator_validation_command_templates": initial_operator_validation_command_templates,
        "labeler_work_completion": work.get("labeler_work_completion", {}),
        **_labeler_work_completion_fields(
            work.get("labeler_work_completion")
            if isinstance(work.get("labeler_work_completion"), Mapping)
            else None
        ),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
        "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
        "single_owner_policy": _assignment_ownership_policy(),
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_integrity": check_report.get(
            "assignment_ownership_integrity",
            {},
        ),
        "safe_share_gate": initial_safe_share_gate,
        **initial_safe_share_fields,
        **initial_safe_share_checklist_fields,
        "reassignment_session_safety": public_reassignment_session_safety,
        **_reassignment_session_safety_flat_fields(public_reassignment_session_safety),
        "browser_response_security_policy": _browser_response_security_policy(),
        "session_guard_policy": _session_guard_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        **initial_operator_validation_public_fields,
        "work": work,
    }
    _add_payload_contract_compact_fields(work_payload)
    handoff_paths["work_summary"].write_text(
        json.dumps(work_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff_paths["signed_links"].write_text(
        "".join(json.dumps(link, sort_keys=True) + "\n" for link in links),
        encoding="utf-8",
    )
    handoff_paths["store_check"].write_text(
        json.dumps({"store_path": str(store_path), **check_report}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_files = {key: str(path) for key, path in handoff_paths.items()}
    if zip_output is not None:
        manifest_files["bundle_zip"] = str(zip_output)
    no_open_by_reason = _count_recordings_without_open_tasks_by_reason(work)
    labeler_landing_url = _labeler_landing_url_for_base(normalized_base_url)
    labeling_home_url = _labeling_home_url_for_base(normalized_base_url)
    dashboard_url = _dashboard_url_for_base(normalized_base_url)
    dataset_queue_url = _dataset_queue_url_for_base(normalized_base_url)
    personal_work_url = (
        f"{str(normalized_base_url).rstrip('/')}{PERSONAL_WORK_PATH}"
        if normalized_base_url
        else ""
    )
    personal_dataset_queue_url = (
        f"{str(normalized_base_url).rstrip('/')}{PERSONAL_DATASET_QUEUE_PATH}"
        if normalized_base_url
        else ""
    )
    expected_user_labeler_landing_url = _dashboard_url_for_expected_user(labeler_landing_url, normalized_user)
    expected_user_labeling_home_url = _dashboard_url_for_expected_user(labeling_home_url, normalized_user)
    expected_user_dashboard_url = _dashboard_url_for_expected_user(dashboard_url, normalized_user)
    expected_user_dataset_queue_url = _dashboard_url_for_expected_user(dataset_queue_url, normalized_user)
    expected_user_personal_work_url = _dashboard_url_for_expected_user(
        personal_work_url,
        normalized_user,
    )
    expected_user_personal_dataset_queue_url = _dashboard_url_for_expected_user(
        personal_dataset_queue_url,
        normalized_user,
    )
    expected_user_identity_probe_url = _dashboard_url_for_expected_user(
        _identity_probe_url_for_base(normalized_base_url),
        normalized_user,
    )
    work["personal_work_page_path"] = PERSONAL_WORK_PATH
    work["labeling_home_page_path"] = LABELING_HOME_PATH
    work["personal_dataset_queue_page_path"] = PERSONAL_DATASET_QUEUE_PATH
    work["expected_user_labeling_home_url"] = expected_user_labeling_home_url
    work["expected_user_personal_work_url"] = expected_user_personal_work_url
    work["expected_user_personal_dataset_queue_url"] = expected_user_personal_dataset_queue_url
    work["preferred_labeler_entrypoint"] = (
        "personal_datasets_waiting_queue"
        if expected_user_personal_dataset_queue_url
        else "datasets_waiting_queue"
    )
    work["preferred_labeler_entry_url"] = (
        expected_user_personal_dataset_queue_url or expected_user_dataset_queue_url
    )
    work["personalized_labeler_entrypoint"] = "personal_datasets_waiting_queue"
    work["personalized_labeler_entry_url"] = (
        expected_user_personal_dataset_queue_url or expected_user_dataset_queue_url
    )
    work["personal_dataset_queue_link_role"] = "preferred_queue"
    work["dataset_queue_link_role"] = "canonical_queue_fallback"
    work["canonical_dataset_queue_link_role"] = "canonical_queue_fallback"
    dataset_queue = work.get("dataset_queue") if isinstance(work.get("dataset_queue"), list) else []
    dataset_queue_summary = (
        work.get("dataset_queue_summary")
        if isinstance(work.get("dataset_queue_summary"), dict)
        else _work_dataset_queue_summary([row for row in dataset_queue if isinstance(row, Mapping)])
    )
    progress_summary = work["progress_summary"] if isinstance(work.get("progress_summary"), dict) else _work_progress_summary(work)
    dataset_queue_state = (
        work.get("dataset_queue_state")
        if isinstance(work.get("dataset_queue_state"), dict)
        else _dataset_queue_state(work)
    )
    canonical_dataset_queue_preview_url = expected_user_dataset_queue_url or _first_dataset_queue_url(
        dataset_queue,
        base_url=normalized_base_url,
    )
    dataset_queue_preview_url = (
        expected_user_personal_dataset_queue_url or canonical_dataset_queue_preview_url
    )
    labeler_safety = _labeler_safety_policy()
    queue_first_entry_contract = _queue_first_entry_contract_policy(
        labeler_safety=labeler_safety,
        labeler_landing_page_path="/",
        labeler_landing_url=labeler_landing_url,
        expected_user_labeler_landing_url=expected_user_labeler_landing_url,
        labeling_home_page_path=LABELING_HOME_PATH,
        labeling_home_url=labeling_home_url,
        expected_user_labeling_home_url=expected_user_labeling_home_url,
        dataset_queue_page_path=DATASET_QUEUE_PATH,
        dataset_queue_url=dataset_queue_url,
        expected_user_dataset_queue_url=expected_user_dataset_queue_url,
        dashboard_url=dashboard_url,
        expected_user_dashboard_url=expected_user_dashboard_url,
        personal_dataset_queue_page_path=PERSONAL_DATASET_QUEUE_PATH,
        personal_dataset_queue_url=personal_dataset_queue_url,
        expected_user_personal_dataset_queue_url=expected_user_personal_dataset_queue_url,
        personal_work_page_path=PERSONAL_WORK_PATH,
        personal_work_url=personal_work_url,
        expected_user_personal_work_url=expected_user_personal_work_url,
    )
    work["labeler_safety"] = labeler_safety
    work["queue_first_entry_contract"] = queue_first_entry_contract
    work_payload["labeler_safety"] = labeler_safety
    work_payload["queue_first_entry_contract"] = queue_first_entry_contract
    assignment_snapshot = _handoff_assignment_snapshot_from_work(work, normalized_user)
    public_reassignment_session_safety = (
        work.get("reassignment_session_safety")
        if isinstance(work.get("reassignment_session_safety"), Mapping)
        else _public_reassignment_session_safety_fields(
            check_report.get("reassignment_session_safety", {}),
            recording_ids=_work_recording_ids(work),
        )
    )
    handoff_store_checks_ok = _handoff_store_checks_ok_for_user(
        check_report,
        public_reassignment_session_safety,
    )
    dataset_queue_payload = {
        "ok": True,
        "schema": "palette.web_labeling_dataset_queue.v1",
        "store_path": str(store_path),
        "user": normalized_user,
        "include_completed": include_completed,
        "base_url": normalized_base_url or None,
        "labeler_landing_page_path": "/",
        "labeler_landing_url": labeler_landing_url,
        "labeling_home_page_path": LABELING_HOME_PATH,
        "labeling_home_url": labeling_home_url,
        "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
        "expected_user_labeling_home_url": expected_user_labeling_home_url,
        "dashboard_path": DASHBOARD_PATH,
        "dashboard_url": dashboard_url,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "dataset_queue_url": dataset_queue_url,
        "expected_user_dashboard_url": expected_user_dashboard_url,
        "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "expected_user_personal_work_url": expected_user_personal_work_url,
        "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
        "preferred_labeler_entrypoint": work["preferred_labeler_entrypoint"],
        "preferred_labeler_entry_url": work["preferred_labeler_entry_url"],
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": (
            expected_user_personal_dataset_queue_url or expected_user_dataset_queue_url
        ),
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "expected_user_identity_probe_url": expected_user_identity_probe_url,
        "empty_state": work.get("empty_state", {}),
        "progress_summary": progress_summary,
        "dataset_queue_summary": dataset_queue_summary,
        "direct_browser_start_contract_summary": work.get(
            "direct_browser_start_contract_summary", {}
        ),
        "dataset_queue_state": dataset_queue_state,
        "labeler_work_completion": work.get("labeler_work_completion", {}),
        **_labeler_work_completion_fields(
            work.get("labeler_work_completion")
            if isinstance(work.get("labeler_work_completion"), Mapping)
            else None
        ),
        "reassignment_session_safety": public_reassignment_session_safety,
        **_reassignment_session_safety_flat_fields(public_reassignment_session_safety),
        "labeler_start_ready": bool(work.get("labeler_start_ready")),
        "labeler_start_status": str(work.get("labeler_start_status") or ""),
        "labeler_action": str(work.get("labeler_action") or ""),
        "labeler_start_message": str(work.get("labeler_start_message") or ""),
        "labeler_start_operator_action": str(work.get("labeler_start_operator_action") or ""),
        "dataset_queue_preview_url": dataset_queue_preview_url,
        "canonical_dataset_queue_preview_url": canonical_dataset_queue_preview_url,
        "dataset_queue": dataset_queue,
        "datasets": dataset_queue,
        "assignment_snapshot": assignment_snapshot,
        "known_user_status": known_user_status,
        "labeler_safety": labeler_safety,
        "queue_first_entry_contract": queue_first_entry_contract,
        "labeler_route_authorization_policy": labeler_route_authorization_policy,
        "labeler_route_authorization_checklist": labeler_route_authorization_checklist,
        "operator_authorization_policy": _operator_authorization_policy(),
        "operator_recovery_policy": _operator_recovery_policy(),
        "operator_validation_visibility_policy": operator_validation_visibility_policy,
        "operator_validation_command_templates": initial_operator_validation_command_templates,
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
        "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
        "single_owner_policy": _assignment_ownership_policy(),
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_integrity": check_report.get(
            "assignment_ownership_integrity",
            {},
        ),
        "safe_share_gate": initial_safe_share_gate,
        **initial_safe_share_fields,
        **initial_safe_share_checklist_fields,
        "browser_response_security_policy": _browser_response_security_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        **initial_operator_validation_public_fields,
    }
    _add_payload_contract_compact_fields(dataset_queue_payload)
    handoff_paths["dataset_queue"].write_text(
        json.dumps(dataset_queue_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "ok": handoff_store_checks_ok,
        "store_path": str(store_path),
        "output_dir": str(output_dir),
        "user": normalized_user,
        "include_completed": include_completed,
        "base_url": normalized_base_url or None,
        "labeler_landing_page_path": "/",
        "labeler_landing_url": labeler_landing_url,
        "labeling_home_page_path": LABELING_HOME_PATH,
        "labeling_home_url": labeling_home_url,
        "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
        "expected_user_labeling_home_url": expected_user_labeling_home_url,
        "dashboard_path": DASHBOARD_PATH,
        "dashboard_url": dashboard_url,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "dataset_queue_url": dataset_queue_url,
        "expected_user_dashboard_url": expected_user_dashboard_url,
        "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "expected_user_personal_work_url": expected_user_personal_work_url,
        "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
        "preferred_labeler_entrypoint": work["preferred_labeler_entrypoint"],
        "preferred_labeler_entry_url": work["preferred_labeler_entry_url"],
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": (
            expected_user_personal_dataset_queue_url or expected_user_dataset_queue_url
        ),
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "expected_user_identity_probe_url": expected_user_identity_probe_url,
        "known_user_status": known_user_status,
        "labeler_safety": labeler_safety,
        "queue_first_entry_contract": queue_first_entry_contract,
        "labeler_route_authorization_policy": labeler_route_authorization_policy,
        "labeler_route_authorization_checklist": labeler_route_authorization_checklist,
        "operator_authorization_policy": _operator_authorization_policy(),
        "operator_recovery_policy": _operator_recovery_policy(),
        "operator_validation_visibility_policy": operator_validation_visibility_policy,
        "operator_validation_command_templates": initial_operator_validation_command_templates,
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_mutation_write_checklist": _browser_mutation_write_runtime_checklist(),
        "dataset_queue_direct_start_policy": _dataset_queue_direct_start_policy(),
        "runtime_operator_validation_gate_cli_policy": _runtime_operator_validation_gate_cli_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_workflows": _browser_workflow_capabilities(),
        "single_owner_policy": _assignment_ownership_policy(),
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_integrity": check_report.get("assignment_ownership_integrity", {}),
        "store_consistency": check_report,
        "reassignment_session_safety": public_reassignment_session_safety,
        "assignment_snapshot": assignment_snapshot,
        "ttl_seconds": _effective_signed_link_ttl_seconds(ttl_seconds),
        "generated_at_utc": generated_at_utc,
        "links_expire_at_utc": links_expire_at_utc,
        "files": manifest_files,
        "progress_summary": progress_summary,
        "dataset_queue_summary": dataset_queue_summary,
        "direct_browser_start_contract_summary": work.get(
            "direct_browser_start_contract_summary", {}
        ),
        "dataset_queue_state": dataset_queue_state,
        "labeler_work_completion": work.get("labeler_work_completion", {}),
        **_labeler_work_completion_fields(
            work.get("labeler_work_completion")
            if isinstance(work.get("labeler_work_completion"), Mapping)
            else None
        ),
        "labeler_start_ready": bool(work.get("labeler_start_ready")),
        "labeler_start_status": str(work.get("labeler_start_status") or ""),
        "labeler_action": str(work.get("labeler_action") or ""),
        "labeler_start_message": str(work.get("labeler_start_message") or ""),
        "labeler_start_operator_action": str(work.get("labeler_start_operator_action") or ""),
        "dataset_queue_preview_url": dataset_queue_preview_url,
        "canonical_dataset_queue_preview_url": canonical_dataset_queue_preview_url,
        "counts": {
            "recordings": len(work.get("recordings", [])),
            "tasks": int(work.get("task_count", 0)),
            "signed_links": len(links),
            "ready_to_share_links": sum(1 for link in links if bool(link.get("ready_to_share"))),
            "dataset_queue": int(dataset_queue_summary.get("dataset_count") or 0),
            "waiting_datasets": int(dataset_queue_summary.get("waiting_dataset_count") or 0),
            "dataset_open_tasks": int(dataset_queue_summary.get("open_task_count") or 0),
            "recordings_without_open_tasks": _count_recordings_without_open_tasks(work),
            "recordings_without_open_tasks_by_reason": no_open_by_reason,
            "recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(no_open_by_reason),
            "redacted_summary_fields": _count_redacted_summary_fields(work),
            "store_check_issues": int(check_report["issue_count"]),
            "store_check_warnings": int(check_report["warning_count"]),
            "reassignment_session_safety_ok": bool(
                public_reassignment_session_safety.get("ok", True)
            ),
            "reassignment_session_safety_mismatch_count": int(
                public_reassignment_session_safety.get("active_session_assignment_mismatch_count")
                or 0
            ),
            "reassignment_session_safety_blocks_labeler_mutation": bool(
                public_reassignment_session_safety.get("blocks_labeler_mutation")
            ),
            "assignment_ownership_duplicate_active_owners": int(
                (check_report.get("assignment_ownership_integrity") or {}).get("duplicate_active_owner_count") or 0
            )
            if isinstance(check_report.get("assignment_ownership_integrity"), Mapping)
            else 0,
            "assignment_ownership_unique_active_recordings": int(
                (check_report.get("assignment_ownership_integrity") or {}).get("unique_active_recording_count") or 0
            )
            if isinstance(check_report.get("assignment_ownership_integrity"), Mapping)
            else 0,
        },
    }
    manifest.update(_handoff_labeler_safety_fields(manifest))
    manifest.update(_handoff_labeler_route_authorization_fields(manifest))
    manifest.update(_handoff_signed_link_policy_fields(manifest))
    manifest.update(_handoff_session_guard_fields(manifest))
    manifest.update(_handoff_task_state_policy_fields(manifest))
    manifest.update(_handoff_zarr_backup_fields(manifest))
    manifest.update(_handoff_mutation_audit_fields(manifest))
    manifest.update(_handoff_browser_response_security_fields(manifest))
    manifest.update(_handoff_browser_mutation_write_fields(manifest))
    manifest.setdefault("single_owner_policy", _assignment_ownership_policy())
    manifest.setdefault(
        "assignment_ownership_integrity",
        check_report.get("assignment_ownership_integrity", {}),
    )
    _add_payload_contract_compact_fields(manifest)
    if not isinstance(
        manifest.get("runtime_operator_validation_gate_cli_policy"),
        Mapping,
    ):
        manifest["runtime_operator_validation_gate_cli_policy"] = (
            _runtime_operator_validation_gate_cli_policy()
        )
    manifest.update(
        _runtime_operator_validation_gate_cli_policy_fields(
            manifest.get("runtime_operator_validation_gate_cli_policy")
            if isinstance(
                manifest.get("runtime_operator_validation_gate_cli_policy"),
                Mapping,
            )
            else None
        )
    )
    manifest.update(_handoff_operator_recovery_fields(manifest))
    manifest.update(
        _operator_validation_invitation_fields(
            _web_labeling_validation_checklist_payload(
                manifest,
                bundle_label="single-user handoff bundle",
            )
        )
    )
    sendability = _handoff_sendability_summary([manifest])
    manifest.update(_handoff_entry_artifact_fields(manifest))
    manifest["safe_share_gate"] = initial_safe_share_gate
    manifest.update(initial_safe_share_fields)
    manifest.update(initial_safe_share_checklist_fields)
    manifest["ready_to_send"] = _handoff_ready_to_send(manifest)
    manifest["sendability_reasons"] = _handoff_sendability_reasons(manifest)
    manifest["sendability_actions"] = _handoff_sendability_actions(manifest["sendability_reasons"])
    manifest["sendability_warnings"] = sendability["warnings"]
    final_operator_validation_public_fields = _operator_validation_public_fields(manifest)
    final_operator_validation_command_templates = _operator_validation_command_templates()
    manifest["operator_validation_command_templates"] = final_operator_validation_command_templates
    final_operator_validation_gate_fields = _operator_validation_gate_flat_fields(manifest)
    manifest.update(final_operator_validation_gate_fields)
    final_safe_share_gate = _safe_share_gate_policy()
    final_safe_share_fields = _safe_share_gate_flat_fields(final_safe_share_gate)
    final_safe_share_checklist_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
        final_operator_validation_public_fields,
        safe_share_gate=final_safe_share_gate,
    )
    manifest["safe_share_gate"] = final_safe_share_gate
    manifest.update(final_safe_share_fields)
    manifest.update(final_safe_share_checklist_fields)
    final_operator_validation_gate_fields = _operator_validation_gate_flat_fields(
        {
            **final_operator_validation_public_fields,
            **final_safe_share_checklist_fields,
        }
    )
    manifest.update(final_operator_validation_gate_fields)
    manifest["personalized_launch_readiness"] = _personalized_launch_readiness_summary(
        manifest
    )
    work.update(final_operator_validation_public_fields)
    work.update(final_operator_validation_gate_fields)
    work["safe_share_gate"] = final_safe_share_gate
    work.update(final_safe_share_fields)
    work.update(final_safe_share_checklist_fields)
    work["operator_validation_command_templates"] = final_operator_validation_command_templates
    work["operator_validation_visibility_policy"] = operator_validation_visibility_policy
    work["personalized_launch_readiness"] = _personalized_launch_readiness_summary(
        work,
        fallback=manifest,
    )
    work_payload.update(final_operator_validation_public_fields)
    work_payload.update(final_operator_validation_gate_fields)
    work_payload["safe_share_gate"] = final_safe_share_gate
    work_payload.update(final_safe_share_fields)
    work_payload.update(final_safe_share_checklist_fields)
    work_payload["operator_validation_command_templates"] = final_operator_validation_command_templates
    work_payload["operator_validation_visibility_policy"] = operator_validation_visibility_policy
    work_payload["work"] = work
    _add_payload_contract_compact_fields(work_payload)
    dataset_queue_payload.update(final_operator_validation_public_fields)
    dataset_queue_payload.update(final_operator_validation_gate_fields)
    dataset_queue_payload["safe_share_gate"] = final_safe_share_gate
    dataset_queue_payload.update(final_safe_share_fields)
    dataset_queue_payload.update(final_safe_share_checklist_fields)
    dataset_queue_payload["operator_validation_command_templates"] = final_operator_validation_command_templates
    dataset_queue_payload["operator_validation_visibility_policy"] = operator_validation_visibility_policy
    _add_payload_contract_compact_fields(dataset_queue_payload)
    handoff_paths["work_summary"].write_text(
        json.dumps(work_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff_paths["dataset_queue"].write_text(
        json.dumps(dataset_queue_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    handoff_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_user_handoff_html_index(
        user=normalized_user,
        work=work,
        links=links,
        manifest=manifest,
        output_path=handoff_paths["html_index"],
    )
    _write_user_handoff_message(
        user=normalized_user,
        manifest=manifest,
        output_path=handoff_paths["message"],
    )
    _write_user_handoff_quickstart(
        user=normalized_user,
        manifest=manifest,
        output_path=handoff_paths["quickstart"],
    )
    _write_user_handoff_validation_log(manifest, handoff_paths["validation_log"])
    _write_user_handoff_validation_checklist(manifest, handoff_paths["validation_checklist"])
    if zip_output is not None:
        _write_directory_zip(output_dir, zip_output, overwrite=overwrite)
    return manifest
