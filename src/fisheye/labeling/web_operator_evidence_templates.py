"""Operator evidence template builders for web-labeling launch checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone


def configure_operator_evidence_template_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

def _identity_source_evidence_template_impl(
    *,
    base_url: str | None = None,
    users: Sequence[str] = (),
) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    normalized_base_url = str(base_url or "").rstrip("/")
    identity_probe_base = (
        f"{normalized_base_url}{IDENTITY_PROBE_PATH}" if normalized_base_url else IDENTITY_PROBE_PATH
    )
    dataset_queue_base = (
        f"{normalized_base_url}{DATASET_QUEUE_PATH}" if normalized_base_url else DATASET_QUEUE_PATH
    )
    labeling_home_base = (
        f"{normalized_base_url}{LABELING_HOME_PATH}" if normalized_base_url else LABELING_HOME_PATH
    )
    personal_dataset_queue_base = (
        f"{normalized_base_url}{PERSONAL_DATASET_QUEUE_PATH}"
        if normalized_base_url
        else PERSONAL_DATASET_QUEUE_PATH
    )
    evidence_users: list[dict[str, object]] = []
    for user in sorted({str(item).strip() for item in users if str(item).strip()}):
        expected_user_dataset_queue_url = _dashboard_url_for_expected_user(
            dataset_queue_base,
            user,
        )
        expected_user_personal_dataset_queue_url = _dashboard_url_for_expected_user(
            personal_dataset_queue_base,
            user,
        )
        expected_user_labeling_home_url = _dashboard_url_for_expected_user(
            labeling_home_base,
            user,
        )
        evidence_users.append(
            {
                "expected_user": user,
                "expected_user_identity_probe_url": _dashboard_url_for_expected_user(
                    identity_probe_base,
                    user,
                ),
                "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
                "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
                "expected_user_labeling_home_url": expected_user_labeling_home_url,
                "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                "preferred_labeler_entry_url": expected_user_personal_dataset_queue_url,
                "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                "personalized_labeler_entry_url": expected_user_personal_dataset_queue_url,
                "personal_dataset_queue_link_role": "preferred_queue",
                "labeling_home_link_role": "human_readable_queue_alias",
                "labeling_home_is_preferred_entrypoint": False,
                "dataset_queue_link_role": "canonical_queue_fallback",
                "canonical_dataset_queue_link_role": "canonical_queue_fallback",
                "preferred_labeler_entry_url_matches_dataset_queue": True,
                "preferred_labeler_entry_url_matches_personal_dataset_queue": True,
                "personalized_labeler_entry_url_matches_personal_dataset_queue": True,
                "resolved_user": "",
                "identity_matches_expected_user": False,
                "captured_at_utc": "",
                "authenticated_session_context": "",
                "operator": "",
                "operator_approved_at_utc": "",
                "notes": "",
            }
        )
    return {
        "ok": True,
        "schema": "palette.web_labeling_identity_source_evidence_template.v1",
        "generated_at_utc": generated_at_utc,
        "operator_only": True,
        "labelers_do_not_receive": True,
        "base_url": normalized_base_url or None,
        "validation_gate": "identity_probe_verification",
        "identity_probe_path": IDENTITY_PROBE_PATH,
        "identity_probe_api_path": "/api/me/identity",
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "labeling_home_page_path": LABELING_HOME_PATH,
        "labeling_home_url": labeling_home_base,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
        "assignment_user_source": "deployment_resolved_browser_user",
        "expected_user_guard": "dashboard_user_mismatch",
        "instructions": [
            "Open each expected_user_identity_probe_url in the deployed authentication context before sharing labeler links.",
            "Confirm the probe reports preferred_labeler_entry_url_matches_personal_dataset_queue and personalized_labeler_entry_url_matches_personal_dataset_queue as true for the guarded /my-datasets URL.",
            "Use expected_user_labeling_home_url only as the guarded human-readable /labeling alias; /my-datasets remains the preferred approval gate.",
            "Record the resolved user exactly as Palette reports it; identity evidence is approved only on an exact identity match and a guarded personalized dataset queue match.",
            "Do not share this evidence file with labelers; it may include deployment and authenticated-session details.",
            "Only send guarded personalized dataset queue links for users whose identity evidence is operator-approved.",
        ],
        "counts": {
            "users": len(evidence_users),
            "pending_operator_confirmation": len(evidence_users),
            "operator_approved": 0,
        },
        "users": evidence_users,
    }


def _browser_smoke_evidence_template_impl(
    *,
    base_url: str | None = None,
    users: Sequence[str] = (),
) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    normalized_base_url = str(base_url or "").rstrip("/")
    identity_probe_base = (
        f"{normalized_base_url}{IDENTITY_PROBE_PATH}" if normalized_base_url else IDENTITY_PROBE_PATH
    )
    landing_base = normalized_base_url or "/"
    dataset_queue_base = (
        f"{normalized_base_url}{DATASET_QUEUE_PATH}" if normalized_base_url else DATASET_QUEUE_PATH
    )
    personal_dataset_queue_base = (
        f"{normalized_base_url}{PERSONAL_DATASET_QUEUE_PATH}"
        if normalized_base_url
        else PERSONAL_DATASET_QUEUE_PATH
    )
    personal_work_base = (
        f"{normalized_base_url}{PERSONAL_WORK_PATH}" if normalized_base_url else PERSONAL_WORK_PATH
    )
    labeling_home_base = (
        f"{normalized_base_url}{LABELING_HOME_PATH}" if normalized_base_url else LABELING_HOME_PATH
    )
    dashboard_base = f"{normalized_base_url}{DASHBOARD_PATH}" if normalized_base_url else DASHBOARD_PATH
    sorted_users = sorted({str(item).strip() for item in users if str(item).strip()})
    smoke_users: list[dict[str, object]] = []
    for idx, user in enumerate(sorted_users):
        wrong_user = next((candidate for candidate in sorted_users if candidate != user), "")
        smoke_users.append(
            {
                "expected_user": user,
                "wrong_expected_user": wrong_user,
                "identity_probe_url": _dashboard_url_for_expected_user(identity_probe_base, user),
                "landing_url": _dashboard_url_for_expected_user(landing_base, user),
                "labeling_home_url": _dashboard_url_for_expected_user(labeling_home_base, user),
                "dataset_queue_url": _dashboard_url_for_expected_user(dataset_queue_base, user),
                "personalized_dataset_queue_url": _dashboard_url_for_expected_user(
                    personal_dataset_queue_base,
                    user,
                ),
                "personalized_work_url": _dashboard_url_for_expected_user(
                    personal_work_base,
                    user,
                ),
                "dashboard_url": _dashboard_url_for_expected_user(dashboard_base, user),
                "wrong_expected_user_dataset_queue_url": _dashboard_url_for_expected_user(
                    dataset_queue_base,
                    wrong_user,
                )
                if wrong_user
                else "",
                "wrong_expected_user_labeling_home_url": _dashboard_url_for_expected_user(
                    labeling_home_base,
                    wrong_user,
                )
                if wrong_user
                else "",
                "wrong_expected_user_personalized_dataset_queue_url": _dashboard_url_for_expected_user(
                    personal_dataset_queue_base,
                    wrong_user,
                )
                if wrong_user
                else "",
                "wrong_expected_user_personalized_work_url": _dashboard_url_for_expected_user(
                    personal_work_base,
                    wrong_user,
                )
                if wrong_user
                else "",
                "run_status": "pending_operator_confirmation" if idx == 0 else "available_candidate",
                "resolved_user": "",
                "identity_matches_expected_user": False,
                "browser_only_runtime_verified": False,
                "no_local_palette_install_verified": False,
                "no_local_crimson_install_verified": False,
                "no_local_conda_or_project_dependencies_verified": False,
                "personalized_dataset_queue_verified": False,
                "preferred_labeler_entry_url_matches_personal_dataset_queue": False,
                "personalized_labeler_entry_url_matches_personal_dataset_queue": False,
                "personalized_work_dashboard_verified": False,
                "labeler_sees_only_assigned_work": False,
                "support_text_redacted": False,
                "expected_user_mismatch_rejected": False,
                "task_opened": False,
                "induced_failure_support_detail_redacted": False,
                "completion_verified": False,
                "completed_task_read_only_verified": False,
                "stale_tab_save_rejected": False,
                "operator_reopen_verified": False,
                "captured_at_utc": "",
                "operator": "",
                "operator_approved_at_utc": "",
                "notes": "",
            }
        )
    return {
        "ok": True,
        "schema": "palette.web_labeling_browser_smoke_evidence_template.v1",
        "generated_at_utc": generated_at_utc,
        "operator_only": True,
        "labelers_do_not_receive": True,
        "base_url": normalized_base_url or None,
        "validation_gate": "browser_smoke",
        "recommended_representative_user": sorted_users[0] if sorted_users else "",
        "required_checks": list(BROWSER_SMOKE_REQUIRED_FIELDS),
        "personalized_route_smoke_contract": _browser_smoke_personalized_route_contract(),
        "instructions": [
            "Pick one representative expected_user with open assigned work and run the queue-first browser smoke procedure.",
            "Open the identity probe, personalized /my-datasets queue, human-readable /labeling alias, personalized /my-work dashboard fallback, canonical fallbacks, and landing page from this template in the deployed authentication context.",
            "Record only redacted support details and event IDs; do not add raw Zarr paths or credentials to this evidence file.",
            "Set run_status to operator_approved only after identity, browser-only/no-local-install runtime, personalized queue visibility, preferred/personalized entry URL match diagnostics, personalized work-dashboard visibility, mismatch rejection, completion/read-only behavior, stale-tab rejection, and operator reopen evidence are recorded.",
        ],
        "counts": {
            "candidate_users": len(smoke_users),
            "pending_operator_confirmation": 1 if smoke_users else 0,
            "operator_approved": 0,
        },
        "users": smoke_users,
    }


def _disposable_zarr_mutation_smoke_evidence_template_impl(
    plan: Mapping[str, object],
) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    workflows = [
        workflow
        for workflow in _browser_workflow_capabilities()
        if isinstance(workflow, Mapping) and isinstance(workflow.get("write_contract"), Mapping)
    ]
    workflow_rows: list[dict[str, object]] = []
    for workflow in workflows:
        write_contract = workflow.get("write_contract") if isinstance(workflow.get("write_contract"), Mapping) else {}
        workflow_rows.append(
            {
                "workflow_kind": str(workflow.get("workflow_kind") or ""),
                "save_endpoint": str(write_contract.get("save_endpoint") or ""),
                "audit_event": str(write_contract.get("audit_event") or ""),
                "data_plane_write_target": str(write_contract.get("data_plane_write_target") or ""),
                "primary_mutation_target_kind": str(
                    write_contract.get("primary_mutation_target_kind") or ""
                ),
                "source_mutation_target_kind": str(
                    write_contract.get("source_mutation_target_kind") or ""
                ),
                "promotion_mutation_target_kind": str(
                    write_contract.get("promotion_mutation_target_kind") or ""
                ),
                "training_zarr_mutation_target_kind": str(
                    write_contract.get("training_zarr_mutation_target_kind") or ""
                ),
                "browser_label_write_target": str(
                    write_contract.get("browser_label_write_target") or ""
                ),
                "training_zarr_write_mode": str(write_contract.get("training_zarr_write_mode") or ""),
                "csv_handoff_artifact_role": str(write_contract.get("csv_handoff_artifact_role") or ""),
                "csv_handoff_artifacts_are_label_write_targets": bool(
                    write_contract.get("csv_handoff_artifacts_are_label_write_targets")
                ),
                "handoff_csv_artifacts_are_label_write_targets": bool(
                    write_contract.get("handoff_csv_artifacts_are_label_write_targets")
                ),
                "intermediate_csv_artifacts_are_label_write_targets": bool(
                    write_contract.get("intermediate_csv_artifacts_are_label_write_targets")
                ),
                "handoff_artifacts_are_metadata_only": bool(
                    write_contract.get("handoff_artifacts_are_metadata_only")
                ),
                "browser_writes_csv_or_handoff_files": bool(
                    write_contract.get("browser_writes_csv_or_handoff_files")
                ),
                "browser_writes_handoff_csv": bool(write_contract.get("browser_writes_handoff_csv")),
                "browser_writes_intermediate_csv": bool(
                    write_contract.get("browser_writes_intermediate_csv")
                ),
                "browser_receives_zarr_write_authority": bool(
                    write_contract.get("browser_receives_zarr_write_authority")
                ),
                "browser_has_direct_zarr_write_authority": bool(
                    write_contract.get("browser_has_direct_zarr_write_authority")
                ),
                "status": "pending_operator_confirmation",
                "disposable_recording_id": "",
                "disposable_task_id": "",
                "labeler_user": "",
                "disposable_zarr_or_known_good_source": "",
                "backup_or_regeneration_verified": False,
                "server_write_scope_verified": False,
                "task_scoped_training_zarr_write_verified": False,
                "browser_no_direct_zarr_write_authority_verified": False,
                "handoff_artifacts_metadata_only_verified": False,
                "browser_no_csv_or_handoff_write_verified": False,
                "client_target_selector_rejection_verified": False,
                "mutation_event_ids": [],
                "audit_event_verified": False,
                "operator_event_lookup_verified": False,
                "operator_event_lookup_report_paths": [],
                "operator_event_lookup_event_ids": [],
                "registry_refresh_event_ids": [],
                "completion_verified": False,
                "stale_tab_save_rejected": False,
                "bad_mutation_recovery_verified": False,
                "bad_mutation_recovery_mode": "",
                "bad_mutation_recovery_report": "",
                "restored_or_discarded": False,
                "operator": "",
                "operator_approved_at_utc": "",
                "notes": "",
            }
        )
    counts = plan.get("counts") if isinstance(plan.get("counts"), Mapping) else {}
    return {
        "ok": True,
        "schema": "palette.web_labeling_disposable_zarr_mutation_smoke_evidence_template.v1",
        "generated_at_utc": generated_at_utc,
        "operator_only": True,
        "labelers_do_not_receive": True,
        "validation_gate": "disposable_zarr_mutation_smoke",
        "source_backup_plan_schema": str(plan.get("schema") or ""),
        "source_backup_plan_generated_at_utc": str(plan.get("generated_at_utc") or ""),
        "backup_required_targets": int(counts.get("backup_required_targets") or 0),
        "required_checks": list(DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS),
        "instructions": [
            "Run one representative browser save per launched workflow kind against disposable or fully restorable Zarr data.",
            "Confirm the server writes only the intended assigned task/training Zarr scope and records audit/provenance events.",
            "Use each row's data_plane_write_target, primary_mutation_target_kind, and training_zarr_write_mode to confirm the expected Zarr write path.",
            "Confirm the browser receives no direct Zarr write authority and does not write CSV, HTML, JSON, or handoff files.",
            "Confirm browser-supplied CSV/Zarr/write-target selector payloads are rejected before mutation.",
            "Confirm stale browser tabs reject before mutation after completion, reassignment, reopen, or target navigation.",
            "Restore or discard the disposable data before approving this evidence.",
        ],
        "counts": {
            "workflow_kinds": len(workflow_rows),
            "pending_operator_confirmation": len(workflow_rows),
            "operator_approved": 0,
        },
        "workflows": workflow_rows,
    }


def _zarr_backup_evidence_template_impl(plan: Mapping[str, object]) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    targets = plan.get("zarr_targets") if isinstance(plan.get("zarr_targets"), list) else []
    evidence_targets: list[dict[str, object]] = []
    for idx, target in enumerate(targets):
        if not isinstance(target, Mapping) or not bool(target.get("backup_required")):
            continue
        evidence_targets.append(
            {
                "target_index": idx,
                "status": "pending_operator_confirmation",
                "zarr_path": str(target.get("zarr_path") or ""),
                "zarr_role": str(target.get("zarr_role") or ""),
                "recording_ids": list(target.get("recording_ids") or [])
                if isinstance(target.get("recording_ids"), list)
                else [],
                "dataset_ids": list(target.get("dataset_ids") or [])
                if isinstance(target.get("dataset_ids"), list)
                else [],
                "task_ids": list(target.get("task_ids") or [])
                if isinstance(target.get("task_ids"), list)
                else [],
                "registry_paths": list(target.get("registry_paths") or [])
                if isinstance(target.get("registry_paths"), list)
                else [],
                "backup_required": True,
                "copy_before_labeling": bool(target.get("copy_before_labeling")),
                "restore_requires_paused_assignment": bool(
                    target.get("restore_requires_paused_assignment")
                ),
                "backup_execution_manifest_path": "",
                "backup_manifest_path": "",
                "backup_destination": "",
                "backup_created_at_utc": "",
                "backup_verified_at_utc": "",
                "restore_test_result": "",
                "operator": "",
                "operator_approved_at_utc": "",
                "notes": "",
            }
        )
    return {
        "ok": True,
        "schema": "palette.web_labeling_zarr_backup_evidence_template.v1",
        "generated_at_utc": generated_at_utc,
        "operator_only": True,
        "labelers_do_not_receive": True,
        "source_plan_schema": str(plan.get("schema") or ""),
        "source_plan_generated_at_utc": str(plan.get("generated_at_utc") or ""),
        "store_path": str(plan.get("store_path") or ""),
        "policy": _zarr_backup_policy(),
        "instructions": [
            "Fill one row per backup-required mutable Zarr target before broad labeler link sharing.",
            "Record backup_manifest_path or backup_destination only in operator-controlled storage; do not send this file to labelers.",
            "Set status to operator_approved only after the backup or known-good regeneration path is verified.",
            "Pause or unassign affected recordings before any restore from this backup evidence.",
        ],
        "counts": {
            "backup_required_targets": len(evidence_targets),
            "pending_operator_confirmation": len(evidence_targets),
            "operator_approved": 0,
        },
        "targets": evidence_targets,
    }


def _browser_response_security_evidence_template_impl(
    *,
    base_url: str | None = None,
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    response_policy = policy if isinstance(policy, Mapping) else _browser_response_security_policy()
    contract = _browser_response_security_contract_policy(response_policy)
    expected_headers = (
        response_policy.get("headers")
        if isinstance(response_policy.get("headers"), Mapping)
        else {}
    )
    normalized_base_url = str(base_url or "").rstrip("/")
    sample_capture_paths = [
        PERSONAL_DATASET_QUEUE_PATH,
        LABELING_HOME_PATH,
        DATASET_QUEUE_PATH,
        PERSONAL_WORK_PATH,
        DASHBOARD_PATH,
        "/api/me/tasks",
        "/api/me/datasets",
    ]

    def _sample_url(path: str) -> str:
        return f"{normalized_base_url}{path}" if normalized_base_url else path

    sample_urls = [_sample_url(path) for path in sample_capture_paths]
    return {
        "ok": True,
        "schema": "palette.web_labeling_browser_response_security_evidence_template.v1",
        "generated_at_utc": generated_at_utc,
        "operator_only": True,
        "labelers_do_not_receive": True,
        "base_url": normalized_base_url or None,
        "validation_gate": "browser_response_security_headers",
        "policy_schema": str(response_policy.get("schema") or ""),
        "static_contract_ready": bool(contract.get("ready")),
        "proxy_must_preserve_headers": bool(contract.get("proxy_must_preserve_headers")),
        "preferred_capture_path": PERSONAL_DATASET_QUEUE_PATH,
        "preferred_capture_url": _sample_url(PERSONAL_DATASET_QUEUE_PATH),
        "expected_user_capture_query_required": True,
        "required_capture_contract": {
            "schema": "palette.web_labeling_browser_response_security_capture_contract.v1",
            "preferred_capture_path": PERSONAL_DATASET_QUEUE_PATH,
            "preferred_capture_url": _sample_url(PERSONAL_DATASET_QUEUE_PATH),
            "expected_user_query_required": True,
            "authenticated_test_user_required": True,
            "authenticated_test_user_must_match_expected_user": True,
            "capture_path_must_match_declared_sample_path": True,
            "personalized_alias_headers_must_match_canonical": True,
            "labeler_entrypoint": "personal_datasets_waiting_queue",
        },
        "sample_capture_paths": sample_capture_paths,
        "sample_capture_urls": sample_urls,
        "sample_expected_user_capture_urls": [
            f"{url}?expected_user=USER" for url in sample_urls
        ],
        "personalized_alias_capture_paths": [
            PERSONAL_DATASET_QUEUE_PATH,
            PERSONAL_WORK_PATH,
        ],
        "human_readable_queue_alias_capture_paths": [
            LABELING_HOME_PATH,
        ],
        "canonical_fallback_capture_paths": [
            DATASET_QUEUE_PATH,
            DASHBOARD_PATH,
        ],
        "api_capture_paths": [
            "/api/me/tasks",
            "/api/me/datasets",
        ],
        "personalized_alias_headers_must_match_canonical": True,
        "expected_headers": dict(expected_headers),
        "captured_headers": {str(key): "" for key in expected_headers},
        "capture": {
            "url": "",
            "authenticated_test_user": "",
            "captured_at_utc": "",
            "capture_command_or_browser_note": "",
            "proxy_or_deployment": "",
        },
        "checks": {
            "cache_control_preserved": False,
            "pragma_preserved": False,
            "expires_preserved": False,
            "x_frame_options_preserved": False,
            "x_content_type_options_preserved": False,
            "referrer_policy_preserved": False,
            "content_security_policy_preserved": False,
            "permissions_policy_preserved": False,
            "proxy_strips_or_weakens_no_headers": False,
            "expected_user_capture_query_present": False,
            "authenticated_test_user_present": False,
            "authenticated_test_user_matches_expected_user": False,
            "capture_url_matches_preferred_path": False,
            "capture_url_matches_sample_path": False,
            "capture_url_contract_ready": False,
            "authenticated_test_user_contract_ready": False,
        },
        "operator_approval": {
            "status": "pending_operator_confirmation",
            "operator": "",
            "approved_at_utc": "",
            "notes": "",
        },
        "instructions": [
            "Capture deployed response headers from /my-datasets?expected_user=<user> first as an authenticated test labeler.",
            "Record authenticated_test_user as the same user named by the expected_user query; mismatches are not approved.",
            "Use /labeling, /datasets, /my-work, /work, /api/me/tasks, or /api/me/datasets as fallback or route-specific spot checks; personalized alias headers must match canonical route headers.",
            "Copy the observed values into captured_headers and mark each check true only when the proxy preserved the expected value.",
            "Do not share this evidence file with labelers; it may include deployment and authenticated-test-user details.",
            "Set operator_approval.status to operator_approved only after all required response-security headers are preserved.",
        ],
    }


# Preserve original helper names inside this module so moved helpers can
# continue to call each other if future template builders share logic.
_identity_source_evidence_template = _identity_source_evidence_template_impl
_browser_smoke_evidence_template = _browser_smoke_evidence_template_impl
_disposable_zarr_mutation_smoke_evidence_template = _disposable_zarr_mutation_smoke_evidence_template_impl
_zarr_backup_evidence_template = _zarr_backup_evidence_template_impl
_browser_response_security_evidence_template = _browser_response_security_evidence_template_impl
