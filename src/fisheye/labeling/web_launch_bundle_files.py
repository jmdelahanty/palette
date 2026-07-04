"""Launch-bundle file writers for web labeling operator artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

from .admin_dashboard import (
    _DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS,
    _DASHBOARD_COPY_INTENT_VALUES,
    _DASHBOARD_DIRECT_BROWSER_START_FIELDS,
    _DASHBOARD_READY_ROW_STATE_VALUES,
)
from .web_implementation_status import (
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS,
    _IMPLEMENTATION_STATUS_DOES_NOT_REPLACE_SENTENCE,
    _IMPLEMENTATION_STATUS_FILE_ADVISORY_SENTENCE,
    _IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_FLAT_FIELDS,
    _IMPLEMENTATION_STATUS_INSPECT_FIELDS_ARE_SENTENCE,
    _IMPLEMENTATION_STATUS_MACHINE_READABLE_FIELDS_SENTENCE,
    _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT,
    _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS,
)
from .web_report_renderers import (
    _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
    _IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE,
    _IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE,
    _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
    _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
    _RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE,
    _SHAREABILITY_COMPACT_CONTRACT_SENTENCE,
    _SHAREABILITY_COMPACT_GATE_SENTENCE,
    _SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE,
    _SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE,
)

_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES = (
    "ready_invitations",
    "ready_invitations_text",
)

def _write_launch_bundle_readme(manifest: dict[str, object], output_path: Path) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    lines = [
        "Palette labeling launch bundle",
        "",
        "This directory is a complete operator package for announcing a labeling batch.",
        "",
        "Recommended order:",
        "1. Review batch-readiness.json.",
        "2. Open handoffs/index.html for the operator overview.",
        "3. Share each labeler's message.txt, labeler-quickstart.txt, or index.html only after readiness is acceptable.",
        "4. Keep assignments.json and tasks.json with the batch archive as the current-state plan.",
        "5. Review zarr-backup-plan.json and copy listed mutable Zarrs before inviting labelers.",
        "6. Use inspect-command.txt after copying the bundle or ZIP to verify freshness and checksums.",
        "7. Review validation-checklist.json for required validation gates.",
        "8. Fill validation-log-template.md during static, browser, real-zarr, and deployed dry-run validation.",
        "9. Run server preflight and serve with --validation-checklist plus --require-operator-validation-for-browser-work before giving labelers Start/Open and mutation access.",
        "",
        "Important:",
        "- This bundle is a snapshot; it does not update itself after assignments or tasks change.",
        "- Implementation status: browser labeling contracts are generated into this bundle, but link sharing is not approved until operator evidence gates pass.",
        "- Code contracts cover guarded personalized queues, server-owned task/training Zarr writes, metadata-only CSV/handoff artifacts, one-owner assignment policy, and signed links as entry hints that enforce runtime start gates before session creation.",
        "- Launch evidence still requires approved mutable-Zarr backup, response-security, identity-source, browser-smoke, disposable-Zarr mutation, and operator-recovery evidence.",
        "- If launch-evidence-execution-checklist.txt is missing or stale, inspect-handoff emits regenerate_package_with_launch_evidence_execution_checklist; regenerate the launch bundle before sharing.",
        "- Repository operator status reference: docs/web_labeling_implementation_status.md.",
        "- Signed links are convenience entry points; the service still requires authenticated access.",
        "- With --require-operator-validation-for-browser-work enabled, browser Start/Open and browser mutations fail closed until validation-checklist.json has every required gate passed or not_applicable.",
        "- Regenerate the bundle if links expire, assignments change, or new tasks are added.",
        "",
        "Runtime Start/Open and mutation enforcement:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {manifest.get('store_path', '/path/to/labeling_work.sqlite')} preflight "
        "--trust-auth-header --admin-user OPERATOR_USER "
        f"--validation-checklist {files.get('validation_checklist', 'validation-checklist.json')} "
        "--require-operator-validation-for-browser-work",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {manifest.get('store_path', '/path/to/labeling_work.sqlite')} serve "
        "--trust-auth-header --admin-user OPERATOR_USER "
        f"--validation-checklist {files.get('validation_checklist', 'validation-checklist.json')} "
        "--require-operator-validation-for-browser-work",
        "",
        f"Store: {manifest.get('store_path', '')}",
        f"Base URL: {manifest.get('base_url') or '(relative links)'}",
        f"Landing URL: {manifest.get('labeler_landing_url') or '(missing --base-url)'}",
        f"Labeling home URL: {manifest.get('labeling_home_url') or '(missing --base-url)'}",
        f"Dashboard URL: {manifest.get('dashboard_url') or '(missing --base-url)'}",
        f"Personalized dataset queue URL: {manifest.get('personal_dataset_queue_url') or '(missing --base-url)'}",
        f"Personalized work URL: {manifest.get('personal_work_url') or '(missing --base-url)'}",
        f"Dataset queue URL: {manifest.get('dataset_queue_url') or '(missing --base-url)'}",
        f"Validation log: {files.get('validation_log', '')}",
        f"Validation checklist: {files.get('validation_checklist', '')}",
        f"Operator evidence commands: {files.get('operator_evidence_commands', '')}",
        f"Launch evidence execution checklist: {files.get('launch_evidence_execution_checklist', 'launch-evidence-execution-checklist.txt')}",
        f"Implementation status: {files.get('implementation_status', 'implementation-status.txt')}",
        _IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE,
        _IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE,
        _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
        _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
        _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
        _RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE,
        _SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE,
        _SHAREABILITY_COMPACT_CONTRACT_SENTENCE,
        _SHAREABILITY_COMPACT_GATE_SENTENCE,
        _SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE,
        f"Generated at UTC: {manifest.get('generated_at_utc', '')}",
        f"Assignment ownership ok: {(manifest.get('assignment_ownership_integrity') or {}).get('ok') if isinstance(manifest.get('assignment_ownership_integrity'), Mapping) else ''}",
        f"Assignment ownership duplicate active owners: {counts.get('assignment_ownership_duplicate_active_owners', 0)}",
        f"Assignment ownership unique active recordings: {counts.get('assignment_ownership_unique_active_recordings', 0)}",
        f"Users: {counts.get('users', 0)}",
        f"Assignments: {counts.get('assignments', 0)}",
        f"Tasks: {counts.get('tasks', 0)}",
        f"Zarr backup targets: {counts.get('zarr_backup_targets', 0)}",
        f"Zarr backup targets by role: {json.dumps(counts.get('zarr_backup_targets_by_role', {}), sort_keys=True)}",
        f"Zarr backup required targets: {counts.get('zarr_backup_required_targets', 0)}",
        f"Zarr backup required targets by role: {json.dumps(counts.get('zarr_backup_required_targets_by_role', {}), sort_keys=True)}",
        f"Tasks missing zarr paths for backup plan: {counts.get('zarr_backup_missing_path_tasks', 0)}",
        f"Handoff waiting datasets: {counts.get('handoff_waiting_datasets', 0)}",
        f"Handoff dataset startable tasks: {counts.get('handoff_dataset_open_tasks', 0)}",
        f"Handoff assigned recordings without startable tasks: {counts.get('handoff_recordings_without_open_tasks', 0)}",
        f"Handoff assigned recordings without startable tasks by reason: {json.dumps(counts.get('handoff_recordings_without_open_tasks_by_reason', {}), sort_keys=True)}",
        f"Handoff assigned recordings without startable task actions: {' '.join(str(action) for action in (counts.get('handoff_recordings_without_open_tasks_actions') if isinstance(counts.get('handoff_recordings_without_open_tasks_actions'), list) else []))}",
        f"Handoff redacted user-summary fields: {counts.get('handoff_redacted_summary_fields', 0)}",
        f"Handoff store checks ok: {counts.get('handoff_store_checks_ok', manifest.get('handoff_store_checks_ok'))}",
        f"Handoff not-ready reasons: {json.dumps(counts.get('handoff_sendability_reasons', {}), sort_keys=True)}",
        f"Readiness ok: {manifest.get('readiness_ok')}",
        f"Readiness assigned recordings without startable tasks by reason: {json.dumps(counts.get('readiness_active_recordings_without_open_tasks_by_reason', {}), sort_keys=True)}",
        f"Readiness assigned recordings without startable task actions: {' '.join(str(action) for action in (counts.get('readiness_active_recordings_without_open_tasks_actions') if isinstance(counts.get('readiness_active_recordings_without_open_tasks_actions'), list) else []))}",
        "",
    ]
    if manifest.get("include_audit_events"):
        lines.extend(
            [
                "Audit files:",
                f"- Task events: {files.get('audit_task_events', '')}",
                f"- Assignment events: {files.get('audit_assignment_events', '')}",
                f"- Task definition events: {files.get('audit_task_definition_events', '')}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_launch_bundle_implementation_status(manifest: dict[str, object], output_path: Path) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    lines = [
        "Palette browser labeling implementation status",
        "",
        "Implemented code contracts in this launch bundle:",
        "- Labelers use guarded browser entrypoints, preferably /my-datasets?expected_user=<user>.",
        "- Labelers do not need local Palette, Crimson, Conda, or project dependency installs.",
        "- Browser saves are server-side mutations against assigned task/training Zarr scope.",
        "- CSV, HTML, JSON, handoff, roster, and intermediate files are metadata/control-plane artifacts, not label-write targets.",
        "- Browser mutation payloads reject client-selected CSV, Zarr, row/frame/component, and generic write-target selectors.",
        "- Recording assignment is single-owner: one active current assignee per recording.",
        "- Signed task links are short-lived entry hints, not authorization grants.",
        "",
        "Remaining launch blockers before sharing links:",
        "- Mutable Zarr backup-plan evidence exists and is operator-approved.",
        "- Browser/proxy response-security-header evidence exists and is operator-approved.",
        "- Required validation gates have approved evidence.",
        "- Deployment identity source has been checked in the target environment.",
        "- Representative browser smoke evidence has been recorded for at least one assigned labeler.",
        "- Disposable-Zarr mutation smoke evidence has been recorded before broad launch.",
        "",
        "Safe-share rule:",
        "Do not share labeler links because a row says ready_to_send or ready_to_invite.",
        "Share only after inspect-handoff --require-shareable reports labeler_links_safe_to_share=true.",
        "If launch-evidence-execution-checklist.txt is missing or stale, inspect-handoff emits regenerate_package_with_launch_evidence_execution_checklist; regenerate the launch bundle before sharing.",
        _IMPLEMENTATION_STATUS_FILE_ADVISORY_SENTENCE,
        _IMPLEMENTATION_STATUS_MACHINE_READABLE_FIELDS_SENTENCE,
        "Required implementation_status_artifact fields: "
        + ", ".join(_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS),
        f"Required implementation_status_artifact field count: {_IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT}",
        "Payload implementation_status_* flat fields: "
        + ", ".join(_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS),
        f"Payload implementation_status_* flat field count: {_IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT}",
        "Inspect-handoff implementation_status_* flat fields: "
        + ", ".join(_IMPLEMENTATION_STATUS_FLAT_FIELDS),
        f"Inspect-handoff implementation_status_* flat field count: {_IMPLEMENTATION_STATUS_FLAT_FIELD_COUNT}",
        "",
        "Preferred labeler link:",
        "/my-datasets?expected_user=<user>",
        "",
        "Bundle evidence files:",
        f"- Validation checklist: {files.get('validation_checklist', 'validation-checklist.json')}",
        f"- Operator evidence commands: {files.get('operator_evidence_commands', 'operator-evidence-commands.txt')}",
        f"- Launch evidence execution checklist: {files.get('launch_evidence_execution_checklist', 'launch-evidence-execution-checklist.txt')}",
        f"- Inspection command: {files.get('inspect_command', 'inspect-command.txt')}",
        f"- Inspection targets: {files.get('inspection_targets', 'inspection-targets.json')}",
        f"- Checksums: {files.get('checksums', 'checksums.json')}",
        "",
        "Current generated counts:",
        f"- Users: {counts.get('users', 0)}",
        f"- Handoffs ready_to_send: {counts.get('handoffs_ready_to_send', counts.get('ready_to_send', 0))}",
        f"- Handoffs not ready_to_send: {counts.get('handoffs_not_ready_to_send', counts.get('not_ready_to_send', 0))}",
        f"- Zarr backup required targets: {counts.get('zarr_backup_required_targets', 0)}",
        f"- Assignment ownership duplicate active owners: {counts.get('assignment_ownership_duplicate_active_owners', 0)}",
        "",
        "Validation status:",
        "This file is generated status text only. It does not prove tests, syntax checks, deployment smoke tests, or operator evidence approval.",
        _IMPLEMENTATION_STATUS_DOES_NOT_REPLACE_SENTENCE,
        _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
        _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
        _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
        _RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE,
        _SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE,
        _SHAREABILITY_COMPACT_CONTRACT_SENTENCE,
        _SHAREABILITY_COMPACT_GATE_SENTENCE,
        _SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE,
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_launch_bundle_launch_evidence_execution_checklist(
    manifest: Mapping[str, object],
    output_path: Path,
) -> None:
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    output_dir = str(manifest.get("output_dir") or output_path.parent)
    checklist = str(files.get("validation_checklist") or Path(output_dir) / "validation-checklist.json")
    lines = [
        "Palette web-labeling launch evidence execution checklist",
        "",
        "Operator-only checklist. Do not send this file or these commands to labelers.",
        "",
        "Current implementation boundary:",
        "- Browser labeling writes are server-side mutations against assigned task/training Zarr scope.",
        "- CSV, HTML, JSON, handoff, roster, and intermediate files are metadata/control-plane artifacts only.",
        "- Each recording must have exactly one active current assignee.",
        "- Labeler links are not broadly shareable until inspect-handoff --require-shareable reports labeler_links_safe_to_share=true.",
        "",
        "Inspection and repair contract:",
        "- Artifact file: launch-evidence-execution-checklist.txt.",
        "- Inspection field: launch_evidence_execution_checklist.",
        "- Summary field: launch_evidence_execution_checklist_summary.",
        "- Required phrase contract: shareability_launch_evidence_execution_checklist_required_phrases.",
        "- Blocking reasons: launch_evidence_execution_checklist_missing, launch_evidence_execution_checklist_incomplete, launch_evidence_execution_checklist_invalid.",
        "- Repair command: regenerate_package_with_launch_evidence_execution_checklist.",
        "- Repair metadata fields: required_file, required_phrase_contract, required_phrases, missing_phrases, missing_phrase_count.",
        "",
        "Inputs to fill in:",
        "- OPERATOR: operator name or ID.",
        "- USER: representative expected labeler user.",
        "- DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER: deployed /my-datasets?expected_user=USER URL.",
        "- EVENT_ID: mutation event ID from disposable-Zarr browser smoke.",
        "- EVENT_ID-lookup.json: operator event lookup report for EVENT_ID.",
        "",
        "1. Mutable Zarr backup evidence",
        "Required proof: backup targets identified, backup execution manifest exists, restore test result recorded, and operator approval recorded.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-zarr-backup-evidence --evidence {files.get('zarr_backup_evidence_template', 'zarr-backup-evidence-template.json')} "
        "--execution-manifest OPERATOR_BACKUP_DIR/zarr-backup-execution-manifest.json "
        "--target-index TARGET_INDEX --restore-test-result RESULT --operator OPERATOR",
        "",
        "2. Browser/proxy response-security evidence",
        "Required proof: deployed guarded /my-datasets?expected_user=USER capture, authenticated user equals expected user, and protected-route headers satisfy policy.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-browser-response-security-evidence --evidence {files.get('browser_response_security_evidence_template', 'browser-response-security-evidence-template.json')} "
        "--header 'Cache-Control=VALUE' --header 'X-Frame-Options=VALUE' "
        "--operator OPERATOR --capture-url DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER "
        "--authenticated-test-user USER",
        "",
        "3. Deployment identity-source evidence",
        "Required proof: deployed identity probe resolves exactly USER and preferred/personalized entries point at guarded /my-datasets?expected_user=USER.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-identity-source-evidence --evidence {files.get('identity_source_evidence_template', 'identity-source-evidence-template.json')} "
        "--expected-user USER --resolved-user USER --operator OPERATOR "
        "--authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED",
        "",
        "4. Representative browser smoke evidence",
        "Required proof: browser-only/no-local-install runtime, personalized queue/work aliases, assigned-only visibility, mismatch rejection, stale-tab/completion/reopen checks.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-browser-smoke-evidence --evidence {files.get('browser_smoke_evidence_template', 'browser-smoke-evidence-template.json')} "
        "--expected-user USER --resolved-user USER --operator OPERATOR "
        "--browser-only-runtime-verified --no-local-palette-install-verified "
        "--no-local-crimson-install-verified --no-local-conda-or-project-dependencies-verified "
        "--personalized-dataset-queue-verified "
        "--preferred-labeler-entry-url-matches-personal-dataset-queue "
        "--personalized-labeler-entry-url-matches-personal-dataset-queue "
        "--personalized-work-dashboard-verified",
        "",
        "5. Disposable-Zarr mutation smoke evidence",
        "Required proof: task-scoped training-Zarr write, no direct browser Zarr authority, metadata-only CSV/handoff artifacts, rejected client target selectors, operator event lookup, and recovery path.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-disposable-zarr-mutation-smoke-evidence --evidence {files.get('disposable_zarr_mutation_smoke_evidence_template', 'disposable-zarr-mutation-smoke-evidence-template.json')} "
        "--workflow-kind WORKFLOW_KIND --mutation-event-id EVENT_ID "
        "--event-lookup-report EVENT_ID-lookup.json --operator OPERATOR --labeler-user USER "
        "--task-scoped-training-zarr-write-verified "
        "--browser-no-direct-zarr-write-authority-verified "
        "--handoff-artifacts-metadata-only-verified "
        "--browser-no-csv-or-handoff-write-verified "
        "--client-target-selector-rejection-verified "
        "--operator-event-lookup-verified --bad-mutation-recovery-verified "
        "--bad-mutation-recovery-mode RECOVERY_MODE --bad-mutation-recovery-report RECOVERY_REPORT",
        "",
        "6. Apply approved evidence templates",
        "scripts/py -m fisheye.utils.labeling_work "
        f"apply-operator-evidence-templates --path {checklist} --operator OPERATOR",
        "",
        "7. Final shareability inspection",
        "scripts/py -m fisheye.utils.labeling_work "
        "inspect-handoff --path PACKAGE_PATH --require-shareable",
        "",
        "Share labeler links only when inspection succeeds and reports labeler_links_safe_to_share=true.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_launch_bundle_inspect_command(
    *,
    store_path: Path,
    output_dir: Path,
    zip_output: Path | None,
    output_path: Path,
) -> None:
    lines = [
        "Inspect this launch bundle before re-sharing it.",
        "",
        "Directory inspection:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {store_path} inspect-handoff --path {output_dir} --require-shareable",
        "",
    ]
    if zip_output is not None:
        lines.extend(
            [
                "ZIP inspection:",
                "scripts/py -m fisheye.utils.labeling_work "
                f"--store {store_path} inspect-handoff --path {zip_output} --require-shareable",
                "",
            ]
        )
    lines.extend(
        [
            "Inspection is read-only. It checks readiness status, handoff freshness, link expiration, and checksums when present.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_launch_bundle_operator_evidence_commands(
    manifest: Mapping[str, object],
    output_path: Path,
) -> None:
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    store_path = str(manifest.get("store_path") or "/path/to/labeling_work.sqlite")
    output_dir = str(manifest.get("output_dir") or output_path.parent)
    base_url = str(manifest.get("base_url") or "BASE_URL").rstrip("/") or "BASE_URL"
    checklist = str(files.get("validation_checklist") or Path(output_dir) / "validation-checklist.json")
    validation_log = str(files.get("validation_log") or Path(output_dir) / "validation-log-template.md")
    lines = [
        "Palette web-labeling operator evidence command sheet",
        "",
        "Boundary: operator-only. These commands require operator authorization and are not labeler instructions.",
        "Do not send this command sheet, operator evidence templates, backup manifests, or runnable operator commands to labelers.",
        "Labelers should use only their guarded browser links and should not run Palette/Crimson commands or edit Zarrs directly.",
        f"Companion execution checklist: {files.get('launch_evidence_execution_checklist', 'launch-evidence-execution-checklist.txt')}",
        "",
        "Run these from an operator shell after generating a launch bundle. Replace UPPERCASE placeholders with observed values.",
        "",
        "1. Record deployed identity evidence for each expected user:",
        "Confirm the identity probe reports the expected resolved user and that preferred_labeler_entry_url plus personalized_labeler_entry_url both equal DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER before recording approval.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-identity-source-evidence --evidence {files.get('identity_source_evidence_template', 'identity-source-evidence-template.json')} "
        "--expected-user USER --resolved-user RESOLVED_USER --operator OPERATOR "
        "--authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED",
        "",
        "2. Record deployed browser/proxy response-security headers:",
        "Capture the preferred guarded /my-datasets?expected_user=<user> entry first, then spot-check protected route/header parity on /me, /labeling?expected_user=<user>, /identity?expected_user=<user>, /api/me/identity?expected_user=<user>, /my-work?expected_user=<user>, canonical /datasets and /work fallbacks, and personal APIs /api/me/tasks and /api/me/datasets.",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-browser-response-security-evidence --evidence {files.get('browser_response_security_evidence_template', 'browser-response-security-evidence-template.json')} "
        "--header 'Cache-Control=VALUE' --header 'Pragma=VALUE' --header 'Expires=VALUE' "
        "--header 'X-Frame-Options=VALUE' --header 'X-Content-Type-Options=VALUE' "
        "--header 'Referrer-Policy=VALUE' --header 'Content-Security-Policy=VALUE' "
        "--header 'Permissions-Policy=VALUE' --operator OPERATOR "
        "--capture-url DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER "
        "--authenticated-test-user SAME_USER_AS_EXPECTED_USER",
        "",
        "Implementation status summary: implementation-status.txt",
        _IMPLEMENTATION_STATUS_INSPECT_FIELDS_ARE_SENTENCE,
        _IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE,
        _IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE,
        "Checklist artifact repair command: regenerate_package_with_launch_evidence_execution_checklist is emitted when launch-evidence-execution-checklist.txt is missing or stale.",
        _IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE,
        _IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE,
        _RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE,
        _SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE,
        _SHAREABILITY_COMPACT_CONTRACT_SENTENCE,
        _SHAREABILITY_COMPACT_GATE_SENTENCE,
        _SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE,
        "",
        "3. Record one representative browser smoke run:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-browser-smoke-evidence --evidence {files.get('browser_smoke_evidence_template', 'browser-smoke-evidence-template.json')} "
        "--expected-user USER --resolved-user USER --operator OPERATOR "
        "--browser-only-runtime-verified --no-local-palette-install-verified "
        "--no-local-crimson-install-verified --no-local-conda-or-project-dependencies-verified "
        "--personalized-dataset-queue-verified "
        "--preferred-labeler-entry-url-matches-personal-dataset-queue "
        "--personalized-labeler-entry-url-matches-personal-dataset-queue "
        "--personalized-work-dashboard-verified "
        "--labeler-sees-only-assigned-work "
        "--support-text-redacted --expected-user-mismatch-rejected "
        "--task-opened --induced-failure-support-detail-redacted --completion-verified "
        "--completed-task-read-only-verified --stale-tab-save-rejected --operator-reopen-verified",
        "",
        "4. Execute and record mutable-Zarr backups:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"execute-zarr-backup-plan --plan {files.get('zarr_backup_plan', 'zarr-backup-plan.json')} "
        "--backup-dir OPERATOR_BACKUP_DIR --operator OPERATOR",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-zarr-backup-evidence --evidence {files.get('zarr_backup_evidence_template', 'zarr-backup-evidence-template.json')} "
        "--execution-manifest OPERATOR_BACKUP_DIR/zarr-backup-execution-manifest.json "
        "--target-index TARGET_INDEX --restore-test-result RESULT --operator OPERATOR",
        "",
        "5. Verify labeler-reported mutation event IDs through the operator-only admin lookup:",
        f"Open {base_url}/admin, paste EVENT_ID into Audit event lookup, "
        "call GET /api/admin/events/EVENT_ID as an operator, or run:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {store_path} lookup-event --event-id EVENT_ID --output EVENT_ID-lookup.json",
        "Confirm the event task, recording, user, workflow, target, and mutation outcome match the disposable-Zarr smoke notes, then archive the lookup report before approving audit evidence.",
        "",
        "6. Record disposable-Zarr mutation smoke per launched workflow kind after lookup verification:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"record-disposable-zarr-mutation-smoke-evidence --evidence {files.get('disposable_zarr_mutation_smoke_evidence_template', 'disposable-zarr-mutation-smoke-evidence-template.json')} "
        "--workflow-kind WORKFLOW_KIND --mutation-event-id EVENT_ID --operator OPERATOR --labeler-user LABELER_USER "
        "--backup-or-regeneration-verified --server-write-scope-verified --audit-event-verified "
        "--task-scoped-training-zarr-write-verified --browser-no-direct-zarr-write-authority-verified "
        "--handoff-artifacts-metadata-only-verified --browser-no-csv-or-handoff-write-verified "
        "--client-target-selector-rejection-verified "
        "--event-lookup-report EVENT_ID-lookup.json --operator-event-lookup-verified "
        "--completion-verified --stale-tab-save-rejected --bad-mutation-recovery-verified "
        "--bad-mutation-recovery-mode RECOVERY_MODE --bad-mutation-recovery-report RECOVERY_REPORT "
        "--restored-or-discarded",
        "",
        "7. Apply approved evidence templates to validation-checklist.json:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"apply-operator-evidence-templates --path {checklist} --operator OPERATOR --append-log {validation_log}",
        "",
        "8. Export copyable dashboard draft text only after approved validation and safe-share inspection:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {store_path} dashboard-roster --base-url {base_url} "
        f"--operator-validation-checklist {checklist} --output dashboard-roster-approved.json",
        "Confirm this roster reports ok=true, safe_share_next_action_summary has no blockers, and inspect-handoff --require-shareable reports labeler_links_safe_to_share=true before sharing with labelers.",
        "",
        "9. Refresh directory bundle checksums after evidence/checklist/log changes:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"refresh-handoff-checksums --path {output_dir} --operator OPERATOR --reason 'operator evidence update'",
        "",
        "10. Inspect before sharing:",
        "scripts/py -m fisheye.utils.labeling_work "
        f"--store {store_path} inspect-handoff --path {output_dir} --require-shareable",
        "Wrapper-safe share diagnostics: require labeler_links_safe_to_share=true, then read "
        "safe_share_launch_blocking_next_actions, safe_share_launch_blocking_next_action_count, "
        "and safe_share_next_action_summary for per-gate operator evidence todos when the "
        "package is not shareable.",
        "Each safe_share_launch_blocking_next_actions row exposes gate_id, status, "
        "operator_only, blocks_share, action, operator_validation_command_ids, "
        "operator_validation_record_command_ids, operator_validation_apply_command_id, "
        "operator_validation_apply_required_after_approval, "
        "operator_validation_evidence_template_field, and "
        "operator_validation_evidence_template_path; wrappers can also discover this row "
        "schema through inspection-targets.json shareability_safe_share_next_action_detail_fields.",
        "Ready-row draft discovery fields: ready_row_draft_bundle_schema, ready_row_draft_bundle_kind, "
        "ready_row_drafts, ready_row_draft_text, ready_row_draft_share_rule, ready_row_state, "
        "ready_row_draft_requires_safe_share_inspection, ready_row_draft_required_safe_share_field, "
        "and ready_row_draft_required_safe_share_value; ready_row_state values are "
        f"{', '.join(_DASHBOARD_READY_ROW_STATE_VALUES)}; copy_intent values are "
        f"{', '.join(_DASHBOARD_COPY_INTENT_VALUES)}; legacy ready-row draft fields are "
        f"{', '.join(_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES)}; wrappers must still require labeler_links_safe_to_share=true.",
        "Browser mutation target CSV fields are "
        f"{', '.join(_DASHBOARD_BROWSER_MUTATION_TARGET_FIELDS)}; required values include "
        "browser_mutation_label_mutation_target_kind=task_scoped_training_zarr, "
        "browser_mutation_browser_label_write_target=training_zarr, "
        "browser_mutation_browser_writes_handoff_csv=false, and "
        "browser_mutation_browser_writes_intermediate_csv=false.",
        "Direct browser Start/Open inspection fields are "
        f"{', '.join(_DASHBOARD_DIRECT_BROWSER_START_FIELDS)}; required values include "
        "dataset_queue_direct_start_policy_present=true, "
        "dataset_queue_direct_start_method=POST, "
        "dataset_queue_direct_start_post_body_expected_user_required=true, "
        "dataset_queue_direct_start_browser_label_write_target=training_zarr, and "
        "dataset_queue_direct_start_browser_writes_intermediate_csv=false.",
        "Single-owner package inspection fields are single_owner_package_contract_met, "
        "single_owner_package_mismatch_count, single_owner_package_mismatch_recording_ids, "
        "and single_owner_package_duplicate_owners_by_recording; each recording must appear "
        "under exactly one active labeler before links are shared.",
        "Compact contract gates are browser_mutation_target_contract_met, "
        "direct_browser_start_contract_met, single_owner_policy_contract_met, and "
        "labeler_route_authorization_runtime_checklist_gate_met; wrappers should require true values "
        "plus zero mismatch counts on per-user payloads and checklist required values on roster rows.",
        "Single-owner store-proof roster fields are advertised in inspection-targets.json as "
        "shareability_single_owner_store_contract_fields with required values in "
        "shareability_single_owner_store_contract_required_values; require the store contract "
        "present/ready/met fields plus server-resolved training-Zarr and no-intermediate-CSV assertions.",
        "Route-authorization store-proof policy fields are advertised as "
        "shareability_labeler_route_authorization_store_proof_fields with required values in "
        "shareability_labeler_route_authorization_store_proof_required_values; require those fields "
        "on roster rows so route policy, route checklist, and single-owner store proof stay aligned.",
        "Observed runtime route-checklist evidence fields are advertised as "
        "shareability_labeler_route_authorization_runtime_checklist_fields with required values in "
        "shareability_labeler_route_authorization_runtime_checklist_required_values; require checklist "
        "present/ready, single_owner_store_proof_ready=true, "
        "assignment_ownership_integrity_ok=true, duplicate_active_owner_count=0, "
        "browser_mutation_target_resolved_server_side=true, "
        "labelers_mutate_assigned_training_zarrs=true, and "
        "labelers_mutate_intermediate_csvs=false.",
        "Operator evidence collection is advertised in inspection-targets.json as "
        "operator_validation_launch_evidence_collection_plan_contract; wrappers can use "
        "operator_validation_command_templates.launch_evidence_collection_plan plus the flat "
        "operator_validation_command_template_launch_evidence_collection_* fields to guide "
        "operator-only evidence collection before requiring labeler_links_safe_to_share=true.",
        "External launch evidence gaps are advertised as "
        "shareability_safe_share_external_launch_evidence_gap_fields in inspection-targets.json "
        "and as safe_share_external_launch_evidence_gap_* fields in inspect-handoff and "
        "shareability output; safe_share_external_launch_evidence_gap_todos gives one "
        "operator-only row per remaining gate with status, template path, record command IDs, "
        "apply command ID, and action text.",
        "Personalized launch readiness is advertised as personalized_launch_readiness in "
        "live APIs, static manifests, handoff index rows, inspection-targets.json, and "
        "personalized_launch_readiness_* roster CSV columns; its required browser write "
        "target values are training_zarr, no CSV/handoff writes, and no direct browser "
        "Zarr write authority.",
        "",
        "Do not send operator evidence templates, backup manifests, or this command sheet to labelers.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_directory_zip(source_dir: Path, zip_path: Path, *, overwrite: bool) -> None:
    import zipfile

    _check_directory_zip_output(source_dir, zip_path, overwrite=overwrite)
    source_dir = source_dir.resolve()
    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, Path(source_dir.name) / path.relative_to(source_dir))


def _check_directory_zip_output(source_dir: Path, zip_path: Path, *, overwrite: bool) -> None:
    source_dir = source_dir.resolve()
    zip_path = zip_path.resolve()
    try:
        zip_path.relative_to(source_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"ZIP output must be outside the handoff directory: {zip_path}")
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing handoff ZIP: {zip_path}")


def _write_directory_checksums(source_dir: Path, output_path: Path) -> dict[str, object]:
    import hashlib

    source_dir = source_dir.resolve()
    output_path = output_path.resolve()
    files: list[dict[str, object]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == output_path:
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(source_dir).as_posix(),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    payload = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(source_dir),
        "count": len(files),
        "files": files,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

