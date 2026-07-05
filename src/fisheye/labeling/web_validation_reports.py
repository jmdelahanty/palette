"""Validation log and checklist report builders for web-labeling handoffs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def configure_validation_report_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

def _write_web_labeling_validation_log_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    bundle_label: str,
) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    lines = [
        "# Web Labeling Validation Log",
        "",
        f"Use this file to record validation evidence for this {bundle_label} before inviting labelers.",
        "",
        "## Batch",
        "",
        f"- Store path: {manifest.get('store_path', '')}",
        f"- Base URL: {manifest.get('base_url') or '(missing --base-url)'}",
        f"- Landing URL: {manifest.get('labeler_landing_url') or manifest.get('expected_user_labeler_landing_url') or '(missing --base-url)'}",
        f"- Dashboard URL: {manifest.get('dashboard_url') or '(missing --base-url)'}",
        f"- Dataset queue URL: {manifest.get('dataset_queue_url') or '(missing --base-url)'}",
        f"- Generated at UTC: {manifest.get('generated_at_utc', '')}",
        f"- User: {manifest.get('user', '')}",
        f"- Users: {counts.get('users', '')}",
        f"- Assignments: {counts.get('assignments', counts.get('handoffs', ''))}",
        f"- Tasks: {counts.get('tasks', '')}",
        f"- Handoffs ready to send: {counts.get('handoffs_ready_to_send', counts.get('ready_to_send', ''))}",
        f"- Handoffs not ready to send: {counts.get('handoffs_not_ready_to_send', counts.get('not_ready_to_send', ''))}",
        "",
        "## Required Files To Review",
        "",
        f"- Manifest: {files.get('manifest', files.get('index', ''))}",
        f"- Batch readiness: {files.get('readiness', '')}",
        f"- Handoffs index: {files.get('handoffs_index', files.get('index', ''))}",
        f"- Handoff roster: {files.get('handoffs_roster', files.get('labeler_roster', ''))}",
        f"- HTML index: {files.get('html_index', '')}",
        f"- README: {files.get('readme', '')}",
        f"- Checksums: {files.get('checksums', '')}",
        f"- Validation checklist: {files.get('validation_checklist', '')}",
        f"- Launch evidence execution checklist: {files.get('launch_evidence_execution_checklist', 'launch-evidence-execution-checklist.txt')}",
        "",
        "## Recording Evidence",
        "",
        "After each validation step, update the machine-readable checklist and append evidence to this log:",
        "",
        "```bash",
        "scripts/py -m fisheye.utils.labeling_work \\",
        f"  --store {manifest.get('store_path', '/path/to/labeling_work.sqlite')} \\",
        "  update-validation-checklist \\",
        f"  --path {files.get('validation_checklist', '/path/to/validation-checklist.json')} \\",
        "  --gate GATE_ID --status passed \\",
        "  --evidence \"Describe the validation result.\" \\",
        f"  --append-log {output_path} \\",
        "  --operator OPERATOR",
        "```",
        "",
        "After operator evidence template files are approved, run `apply-operator-evidence-templates --path validation-checklist.json --operator OPERATOR` to mark approved template-backed gates passed.",
        "After modifying evidence, checklist, or validation-log files inside a directory bundle, run `refresh-handoff-checksums --path PACKAGE_DIR --operator OPERATOR` before re-running inspection.",
        "Use `inspect-handoff --path PACKAGE_PATH` after updating evidence to confirm remaining pending gates.",
        "If `inspect-handoff` reports `launch_evidence_execution_checklist_missing`, `launch_evidence_execution_checklist_incomplete`, or `launch_evidence_execution_checklist_invalid`, follow `regenerate_package_with_launch_evidence_execution_checklist` and regenerate the launch bundle before sharing links.",
        "",
        "## Static And Unit Validation",
        "",
        "- Command: `scripts/check_labeling_web_readiness.sh`",
        "- Exit code:",
        "- Warnings:",
        "- Failures:",
        "- Follow-up:",
        "",
        "## Browser Smoke Validation",
        "",
        "- `/work` shows the expected authenticated user:",
        "- No-assignment empty state is clear:",
        "- `/admin` preflight and policy cards load:",
        "- Task open creates a guarded session:",
        "- Failed API state shows copyable support details:",
        "- Browser smoke evidence template is filled and archived by the operator:",
        "- `record-browser-smoke-evidence` update report is archived:",
        "- Result:",
        "",
        "## Queue-First Entry",
        "",
        "- Guarded landing page is present:",
        "- Guarded dataset queue is present:",
        "- Dashboard fallback is present:",
        "- Identity check is required before labeling:",
        "- Queue-first paths include `/`, `/me`, and `/datasets`:",
        "- Result:",
        "",
        "## Identity Probe Links",
        "",
        "- Expected-user identity probe URL is present for single-user handoffs:",
        "- Roster or handoff index carries identity probe evidence for batch handoffs:",
        "- Operator verification in deployed auth context is still required:",
        "- Preferred entry URL matches guarded personal dataset queue:",
        "- Personalized entry URL matches guarded personal dataset queue:",
        "- Identity source evidence template is filled and archived by the operator:",
        "- `record-identity-source-evidence` update report is archived:",
        "- Result:",
        "",
        "## Browser Payload Redaction",
        "",
        "- Labeler APIs do not expose raw Zarr paths:",
        "- Labeler APIs do not expose direct task scope:",
        "- Runtime state paths are redacted:",
        "- Mutation response paths are redacted:",
        "- Error/support details are redacted:",
        "- Admin diagnostics remain operator-only:",
        "- Result:",
        "",
        "## Assignment Ownership Contract",
        "",
        "- One active owner per recording is enforced:",
        "- Assignment integrity has no duplicate active owners:",
        "- Reassignment replaces the previous owner:",
        "- Reassignment closes previous-owner stale sessions:",
        "- Raw assignment changes block open sessions until repaired:",
        "- Result:",
        "",
        "## Labeler Route Authorization",
        "",
        "- Resolved browser user is required:",
        "- Known assignment-store user is required:",
        "- `expected_user` mismatch rejects:",
        "- Copied queue/dashboard/task links are rechecked server-side:",
        "- Signed links are entry hints, not authorization grants:",
        "- Task open requires active assignment and task ownership:",
        "- Result:",
        "",
        "## Signed Link Contract",
        "",
        "- Signed links are task-specific:",
        "- Signed links are not authorization grants:",
        "- New signed links bind expected user:",
        "- Opening a signed link rechecks expected user and identity:",
        "- Opening a signed link enforces runtime operator-validation start gate before session creation:",
        "- Forwarded signed links do not bypass server-side authorization:",
        "- Result:",
        "",
        "## Expected-User Guard Coverage",
        "",
        "- Landing, `/me`, dashboard, and dataset queue pages reject expected-user mismatch:",
        "- Personal work and dataset queue APIs reject expected-user mismatch:",
        "- Task open, task complete, and promotion retry APIs reject expected-user mismatch:",
        "- Signed task links are expected-user bound:",
        "- Forwarded links stop on expected-user mismatch:",
        "- Result:",
        "",
        "## Session Guard and Stale Tab Rejection",
        "",
        "- Current session is required:",
        "- Unexpired session is required:",
        "- Stale tab save is rejected:",
        "- Superseded session is rejected after reassignment, completion, reopen, or operator repair:",
        "- Target token changes reject stale same-session saves:",
        "- Session closure event is recorded:",
        "- Result:",
        "",
        "## Task Completion and Reopen Guard",
        "",
        "- Completed tasks are read-only to ordinary labelers:",
        "- Completed task open requests reject with `task_complete`:",
        "- Completed task save requests reject with `task_complete`:",
        "- Completing a task closes open sessions:",
        "- Only an operator can reopen work for more labeling:",
        "- Result:",
        "",
        "## Operator Boundary Static Contract",
        "",
        "- Admin routes require operator authorization:",
        "- Non-admin labelers receive `admin_required`:",
        "- Resolved user must be in configured admin users:",
        "- Labelers are not operators by default:",
        "- Operator authorization does not grant labeler mutation authority:",
        "- Result:",
        "",
        "## Operator Recovery Static Contract",
        "",
        "- Recovery routes require operator authorization:",
        "- Reassignment closes previous-owner sessions:",
        "- Completed task reopen is operator-only:",
        "- Task repair closes open sessions and records `task_operator_repaired`:",
        "- Failed promotion retry is operator-only:",
        "- Session closure events are operator-inspectable:",
        "- Audit event lookup route `/api/admin/events/{event_id}` resolves labeler-provided save event IDs:",
        "- Backup/rollback recovery pauses or unassigns recordings before restore:",
        "- Result:",
        "",
        "## Browser Response Security Static Contract",
        "",
        "- Application response policy uses no-store cache headers:",
        "- Application response policy denies framing:",
        "- Application response policy disables MIME sniffing:",
        "- Application response policy uses no-referrer:",
        "- Application response policy includes CSP frame/base/form/object restrictions:",
        "- Application response policy disables camera, microphone, and geolocation permissions:",
        "- Result:",
        "",
        "## Browser Workflow Scope",
        "",
        "- Requested workflow kind is browser-supported:",
        "- Target indices, components, labels, and frames are server-owned:",
        "- Browser mutation payloads cannot select arbitrary targets:",
        "- Absolute navigation outside task scope rejects with `nav_error`:",
        "- Browser workflows do not grant direct Zarr or filesystem write authority:",
        "- Result:",
        "",
        "## Operator Authorization Boundary",
        "",
        "- Preflight command or `/api/admin/preflight` captured:",
        "- `admin_routes_require_operator=true`:",
        "- `admin_users_configured=true`:",
        "- `operator_boundary_ready=true`:",
        "- Non-admin labeler receives `admin_required` for `/admin`:",
        "- Result:",
        "",
        "## Browser Response Security Headers",
        "",
        "- Captured URL (`/datasets` or `/api/me/tasks`):",
        "- Authenticated test labeler:",
        "- Capture command or browser devtools note:",
        "- `Cache-Control` preserved:",
        "- `Pragma` preserved:",
        "- `Expires` preserved:",
        "- `X-Frame-Options` preserved:",
        "- `X-Content-Type-Options` preserved:",
        "- `Referrer-Policy` preserved:",
        "- `Content-Security-Policy` preserved:",
        "- `Permissions-Policy` preserved:",
        "- Proxy strips or weakens none of these headers:",
        "- Browser response security evidence template is filled and archived by the operator:",
        "- `record-browser-response-security-evidence` update report is archived:",
        "- Result:",
        "",
        "## Browser Mutation Write Target Policy",
        "",
        "- Authoritative label state is assigned task Zarr scope:",
        "- Label mutation target kind is task_scoped_training_zarr:",
        "- CSV/handoff artifact role is metadata_only_control_plane:",
        "- CSV/handoff artifacts are not label write targets:",
        "- Handoff CSV/HTML/JSON artifacts confirmed metadata-only:",
        "- Browser cannot write CSV/handoff files directly:",
        "- Browser cannot write handoff CSV files:",
        "- Browser cannot write intermediate CSV files:",
        "- Browser cannot receive raw Zarr write authority:",
        "- Save route requires active assignment, open task, current session, and current target token:",
        "- Result:",
        "",
        "## Mutation Audit and Provenance",
        "",
        "- Server records mutation events:",
        "- Audit store is append-only:",
        "- Browser does not write audit records directly:",
        "- Browser does not receive audit-store write credentials:",
        "- Required event fields are present:",
        "- Workflow write contracts include audit provenance:",
        "- `/admin` audit event lookup confirms labeler-reported event ID, task, recording, user, workflow, target, and mutation outcome:",
        "- Result:",
        "",
        "## Zarr Backup and Rollback Contract",
        "",
        "- Backup plan is read-only and operator-owned:",
        "- Backup evidence template is filled and archived by the operator:",
        "- `execute-zarr-backup-plan` execution manifest is archived:",
        "- `record-zarr-backup-evidence` update report is archived:",
        "- `restore-zarr-backup` drill or restore report is archived when rollback is needed:",
        "- Bad disposable-mutation recovery path is verified and archived:",
        "- Copy before labeling is required:",
        "- Labelers do not edit Zarrs directly:",
        "- Labelers do not receive backup paths:",
        "- Restore pauses or unassigns affected recordings first:",
        "- Rollback owner is operator:",
        "- Result:",
        "",
        "## Dataset Queue Start Readiness",
        "",
        "- `dataset_queue_start_readiness` gate status:",
        "- `dataset_queue_blocked_start_users` is empty:",
        "- Every invite-ready labeler has `dataset_queue_state.blocks_labeler_start=false`:",
        "- If any queue is blocked, operator repair or stop-labeling decision:",
        "- Result:",
        "",
        "## Real-Zarr Smoke Validation",
        "",
        "- Smoke spec:",
        "- Exit code:",
        "- Mutation event IDs:",
        "- `/admin` audit event lookup result for each mutation event ID:",
        "- Archived `lookup-event --output` JSON report paths:",
        "- Event lookup confirms task, recording, user, workflow, target, and mutation outcome:",
        "- Registry refresh events:",
        "- Zarr backup used:",
        "- Disposable-Zarr mutation smoke evidence template is filled and archived by the operator:",
        "- `record-disposable-zarr-mutation-smoke-evidence` update report is archived:",
        "- Result:",
        "",
        "## One-Operator / One-Labeler Dry Run",
        "",
        "- Operator:",
        "- Labeler:",
        "- Recording:",
        "- Task:",
        "- Browser smoke evidence template is filled and archived by the operator:",
        "- `record-browser-smoke-evidence` update report is archived:",
        "- Labeler sees only assigned recording:",
        "- Save event ID:",
        "- Completion verified:",
        "- Result:",
        "",
        "## Multi-User Dry Run",
        "",
        "- Users and recordings:",
        "- One active owner per recording verified:",
        "- Expected-user dashboard URLs verified:",
        "- Reassignment stale-tab guard verified:",
        "- Completed-task read-only behavior verified:",
        "- Result:",
        "",
        "## Assignment Transition Evidence",
        "",
        "- Assignment API response archived:",
        "- Recording:",
        "- Previous assignee:",
        "- Previous status:",
        "- New assignee:",
        "- New status:",
        "- Closed session IDs:",
        "- Old browser tab rejected after transition:",
        "- Follow-up:",
        "",
        "## Rollback Drill",
        "",
        "- Incorrect assignment drill:",
        "- Bad mutation drill:",
        "- Backup/restore path verified:",
        "- Result:",
        "",
        "## Final Sign-Off",
        "",
        "- Identity source matches assignment users:",
        "- Status report has no launch-blocking warnings:",
        "- Rollback path and backups exist:",
        "- Safe-share approved to contact labelers (`labeler_links_safe_to_share=true`):",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _web_labeling_validation_checklist_payload_impl(
    manifest: dict[str, object],
    *,
    bundle_label: str,
) -> dict[str, object]:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    base_url = str(manifest.get("base_url") or "").strip()
    labeler_landing_page_path = str(manifest.get("labeler_landing_page_path") or "/").strip()
    labeler_landing_url = str(manifest.get("labeler_landing_url") or "").strip()
    expected_user_labeler_landing_url = str(manifest.get("expected_user_labeler_landing_url") or "").strip()
    labeling_home_page_path = str(manifest.get("labeling_home_page_path") or LABELING_HOME_PATH).strip()
    labeling_home_url = str(manifest.get("labeling_home_url") or "").strip()
    expected_user_labeling_home_url = str(manifest.get("expected_user_labeling_home_url") or "").strip()
    dashboard_url = str(manifest.get("dashboard_url") or manifest.get("expected_user_dashboard_url") or "").strip()
    expected_user_dashboard_url = str(manifest.get("expected_user_dashboard_url") or "").strip()
    expected_user_identity_probe_url = str(manifest.get("expected_user_identity_probe_url") or "").strip()
    dataset_queue_page_path = str(manifest.get("dataset_queue_page_path") or DATASET_QUEUE_PATH).strip()
    dataset_queue_url = str(manifest.get("dataset_queue_url") or "").strip()
    expected_user_dataset_queue_url = str(manifest.get("expected_user_dataset_queue_url") or "").strip()
    personal_dataset_queue_page_path = str(
        manifest.get("personal_dataset_queue_page_path") or PERSONAL_DATASET_QUEUE_PATH
    ).strip()
    manifest_base_url = str(manifest.get("base_url") or "").rstrip("/")
    personal_dataset_queue_url = str(
        manifest.get("personal_dataset_queue_url")
        or _personal_dataset_queue_url_for_base(manifest_base_url)
    ).strip()
    expected_user_personal_dataset_queue_url = str(
        manifest.get("expected_user_personal_dataset_queue_url") or ""
    ).strip()
    personal_work_page_path = str(manifest.get("personal_work_page_path") or PERSONAL_WORK_PATH).strip()
    personal_work_url = str(
        manifest.get("personal_work_url")
        or _personal_work_url_for_base(manifest_base_url)
    ).strip()
    expected_user_personal_work_url = str(
        manifest.get("expected_user_personal_work_url") or ""
    ).strip()
    single_owner_policy = (
        manifest.get("single_owner_policy")
        if isinstance(manifest.get("single_owner_policy"), Mapping)
        else _assignment_ownership_policy()
    )
    operator_authorization_policy = (
        manifest.get("operator_authorization_policy")
        if isinstance(manifest.get("operator_authorization_policy"), Mapping)
        else _operator_authorization_policy()
    )
    operator_authorization_contract = _operator_authorization_contract_policy(operator_authorization_policy)
    operator_recovery_policy = (
        manifest.get("operator_recovery_policy")
        if isinstance(manifest.get("operator_recovery_policy"), Mapping)
        else _operator_recovery_policy()
    )
    operator_recovery_contract = _operator_recovery_contract_policy(operator_recovery_policy)
    labeler_route_authorization_policy = (
        manifest.get("labeler_route_authorization_policy")
        if isinstance(manifest.get("labeler_route_authorization_policy"), Mapping)
        else _labeler_route_authorization_policy()
    )
    labeler_safety = (
        manifest.get("labeler_safety")
        if isinstance(manifest.get("labeler_safety"), Mapping)
        else _labeler_safety_policy()
    )
    zarr_backup_policy = (
        manifest.get("zarr_backup_policy")
        if isinstance(manifest.get("zarr_backup_policy"), Mapping)
        else _zarr_backup_policy()
    )
    mutation_audit_policy = (
        manifest.get("mutation_audit_policy")
        if isinstance(manifest.get("mutation_audit_policy"), Mapping)
        else _mutation_audit_policy()
    )
    browser_mutation_write_policy = (
        manifest.get("browser_mutation_write_policy")
        if isinstance(manifest.get("browser_mutation_write_policy"), Mapping)
        else _browser_mutation_write_policy()
    )
    dataset_queue_direct_start_policy = (
        manifest.get("dataset_queue_direct_start_policy")
        if isinstance(manifest.get("dataset_queue_direct_start_policy"), Mapping)
        else _dataset_queue_direct_start_policy()
    )
    runtime_gate_cli_policy = (
        manifest.get("runtime_operator_validation_gate_cli_policy")
        if isinstance(manifest.get("runtime_operator_validation_gate_cli_policy"), Mapping)
        else _runtime_operator_validation_gate_cli_policy()
    )
    safe_share_gate = (
        manifest.get("safe_share_gate")
        if isinstance(manifest.get("safe_share_gate"), Mapping)
        else _safe_share_gate_policy()
    )
    safe_share_fields = _safe_share_gate_flat_fields(safe_share_gate)
    browser_response_security_policy = (
        manifest.get("browser_response_security_policy")
        if isinstance(manifest.get("browser_response_security_policy"), Mapping)
        else _browser_response_security_policy()
    )
    browser_response_security_contract = _browser_response_security_contract_policy(
        browser_response_security_policy
    )
    session_guard_policy = (
        manifest.get("session_guard_policy")
        if isinstance(manifest.get("session_guard_policy"), Mapping)
        else _session_guard_policy()
    )
    task_state_policy = (
        manifest.get("task_state_policy")
        if isinstance(manifest.get("task_state_policy"), Mapping)
        else _browser_task_state_policy()
    )
    signed_link_policy = (
        manifest.get("signed_link_policy")
        if isinstance(manifest.get("signed_link_policy"), Mapping)
        else _browser_signed_link_policy()
    )
    browser_workflows = (
        manifest.get("browser_workflows")
        if isinstance(manifest.get("browser_workflows"), list)
        else _browser_workflow_capabilities()
    )
    browser_workflow_mappings = [
        workflow
        for workflow in browser_workflows
        if isinstance(workflow, Mapping)
    ]
    browser_mutation_target_contract = _browser_mutation_target_contract_policy(
        task_state_policy=task_state_policy,
        browser_workflows=browser_workflow_mappings,
    )
    task_state_contract = _browser_task_state_contract_policy(task_state_policy)
    browser_workflow_scope_contract = _browser_workflow_scope_contract_policy(
        task_state_policy=task_state_policy,
        browser_workflows=browser_workflow_mappings,
    )
    mutation_audit_contract = _mutation_audit_contract_policy(mutation_audit_policy)
    browser_mutation_write_contract = _browser_mutation_write_contract_policy(browser_mutation_write_policy)
    labeler_route_authorization_contract = _labeler_route_authorization_contract_policy(
        labeler_route_authorization_policy
    )
    signed_link_contract = _signed_link_contract_policy(signed_link_policy)
    expected_user_guard_contract = _expected_user_guard_contract_policy(
        labeler_safety,
        signed_link_contract,
    )
    session_guard_contract = _session_guard_contract_policy(session_guard_policy)
    assignment_ownership_integrity = (
        manifest.get("assignment_ownership_integrity")
        if isinstance(manifest.get("assignment_ownership_integrity"), Mapping)
        else {}
    )
    assignment_ownership_contract = _assignment_ownership_contract_policy(
        single_owner_policy,
        assignment_ownership_integrity,
    )
    browser_payload_redaction_contract = _browser_payload_redaction_contract_policy(labeler_safety)
    queue_first_entry_contract = _queue_first_entry_contract_policy(
        labeler_safety=labeler_safety,
        labeler_landing_page_path=labeler_landing_page_path,
        labeler_landing_url=labeler_landing_url,
        expected_user_labeler_landing_url=expected_user_labeler_landing_url,
        labeling_home_page_path=labeling_home_page_path,
        labeling_home_url=labeling_home_url,
        expected_user_labeling_home_url=expected_user_labeling_home_url,
        dataset_queue_page_path=dataset_queue_page_path,
        dataset_queue_url=dataset_queue_url,
        expected_user_dataset_queue_url=expected_user_dataset_queue_url,
        dashboard_url=dashboard_url,
        expected_user_dashboard_url=expected_user_dashboard_url,
        personal_dataset_queue_page_path=personal_dataset_queue_page_path,
        personal_dataset_queue_url=personal_dataset_queue_url,
        expected_user_personal_dataset_queue_url=expected_user_personal_dataset_queue_url,
        personal_work_page_path=personal_work_page_path,
        personal_work_url=personal_work_url,
        expected_user_personal_work_url=expected_user_personal_work_url,
    )
    identity_probe_link_contract = _identity_probe_link_contract_policy(
        labeler_safety=labeler_safety,
        expected_user_identity_probe_url=expected_user_identity_probe_url,
        files=files,
    )
    dataset_queue_state = (
        manifest.get("dataset_queue_state")
        if isinstance(manifest.get("dataset_queue_state"), Mapping)
        else {}
    )
    dataset_queue_states = (
        counts.get("dataset_queue_states")
        if isinstance(counts.get("dataset_queue_states"), Mapping)
        else {}
    )
    dataset_queue_blocked_start_users = (
        counts.get("dataset_queue_blocked_start_users")
        if isinstance(counts.get("dataset_queue_blocked_start_users"), list)
        else []
    )
    dataset_queue_start_known = bool(
        dataset_queue_state
        or dataset_queue_states
        or "dataset_queue_blocks_labeler_start" in manifest
        or "dataset_queue_blocked_start_users" in counts
    )
    single_queue_start_blocked = (
        _handoff_dataset_queue_blocks_labeler_start(manifest)
        if dataset_queue_state or "dataset_queue_blocks_labeler_start" in manifest
        else False
    )
    dataset_queue_start_blocked = bool(single_queue_start_blocked or dataset_queue_blocked_start_users)
    if dataset_queue_start_blocked:
        dataset_queue_start_status = "needs_review"
        dataset_queue_start_details = (
            "One or more labeler dataset queues block start. Resolve the queue state before labeler links are shared."
        )
    elif dataset_queue_start_known:
        dataset_queue_start_status = "passed"
        dataset_queue_start_details = "Generated dataset queue state reports no start-blocking labeler queues."
    else:
        dataset_queue_start_status = "pending_operator_evidence"
        dataset_queue_start_details = (
            "Dataset queue start state was not present in this artifact. Inspect the generated queue or live /datasets page before inviting labelers."
        )
    ownership_ready = bool(assignment_ownership_integrity.get("ok", True))
    has_browser_entry = bool(
        dashboard_url
        or expected_user_dashboard_url
        or labeler_landing_url
        or expected_user_labeler_landing_url
        or dataset_queue_url
        or expected_user_dataset_queue_url
        or base_url
    )
    dry_run = bool(manifest.get("dry_run"))
    readiness_ready = bool(
        manifest.get(
            "readiness_ok",
            manifest.get("store_checks_ok", manifest.get("ok", False)),
        )
    )
    handoff_store_checks_ready = bool(
        manifest.get(
            "handoff_store_checks_ok",
            manifest.get("store_checks_ok", manifest.get("ok", False)),
        )
    )
    static_ready = (
        readiness_ready
        and handoff_store_checks_ready
        and ownership_ready
    )
    if dry_run:
        static_status = "pending_operator_evidence" if readiness_ready else "needs_review"
        static_details = (
            "Dry-run preview only. Non-dry-run export produces handoff sendability, "
            "store-check, and checksum evidence for this gate."
        )
    else:
        static_status = "passed" if static_ready else "needs_review"
        static_details = "Generated from bundle readiness, store-check status, and one-owner assignment integrity. Operator-validation-only sendability blockers are handled by operator evidence gates."
    backup_required_targets = int(counts.get("zarr_backup_required_targets") or 0)
    backup_missing_path_tasks = int(counts.get("zarr_backup_missing_path_tasks") or 0)
    backup_plan_path = str(files.get("zarr_backup_plan") or "").strip()
    backup_evidence_template_path = str(files.get("zarr_backup_evidence_template") or "").strip()
    response_security_evidence_template_path = str(
        files.get("browser_response_security_evidence_template") or ""
    ).strip()
    identity_source_evidence_template_path = str(
        files.get("identity_source_evidence_template") or ""
    ).strip()
    browser_smoke_evidence_template_path = str(files.get("browser_smoke_evidence_template") or "").strip()
    disposable_zarr_mutation_smoke_evidence_template_path = str(
        files.get("disposable_zarr_mutation_smoke_evidence_template") or ""
    ).strip()
    implementation_status_path = str(files.get("implementation_status") or "").strip()
    implementation_status_file = (
        Path(implementation_status_path).name if implementation_status_path else "implementation-status.txt"
    )
    implementation_status_artifact = _implementation_status_artifact(
        checklist_declared_path=implementation_status_path,
        file=implementation_status_file,
    )
    handoff_only_bundle = bundle_label in {"single-user handoff bundle", "multi-user handoff bundle"}
    if backup_missing_path_tasks > 0:
        backup_status = "needs_review"
        backup_required = True
        backup_details = "One or more assigned tasks did not expose a zarr path for backup planning."
    elif backup_required_targets > 0:
        backup_status = "pending_operator_evidence"
        backup_required = True
        backup_details = "Confirm each mutable zarr target was copied or otherwise protected before inviting labelers."
    elif handoff_only_bundle and not backup_plan_path:
        backup_status = "pending_operator_evidence"
        backup_required = True
        backup_details = (
            "This handoff bundle does not include a top-level zarr backup plan. "
            "Reference the launch bundle or a separately archived zarr-backup-plan export before inviting labelers."
        )
    else:
        backup_status = "not_applicable"
        backup_required = False
        backup_details = "No mutable zarr backup-required targets were reported in this bundle."
    zarr_backup_contract = _zarr_backup_contract_policy(zarr_backup_policy, counts, files)
    gates = [
        _validation_gate(
            "static_readiness",
            "Static readiness and store checks",
            static_status,
            evidence_files=[
                files.get("readiness"),
                files.get("handoffs_index", files.get("index")),
                files.get("manifest", files.get("index")),
            ],
            operator_evidence=[
                "Confirm readiness/store-check JSON has no launch-blocking issues.",
                "Confirm assignment ownership integrity is ok and duplicate active owner count is zero.",
                "Confirm not-ready handoffs are not shared.",
            ],
            details=static_details,
        ),
        _validation_gate(
            "identity_probe_verification",
            "Expected-user identity probe verification",
            "pending_operator_evidence" if has_browser_entry else "needs_review",
            evidence_files=[
                files.get("identity_source_evidence_template"),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("manifest", files.get("index")),
            ],
            operator_evidence=[
                "Open `/identity?expected_user=<user>` as each invite-ready test user.",
                "Fill and archive identity-source-evidence-template.json with resolved user, exact match result, session context, and operator approval.",
                "Record the resolved authenticated user and mismatch behavior.",
            ],
            details="This must be verified in the deployed authentication context; generated links alone do not prove identity mapping.",
        ),
        _validation_gate(
            "queue_first_entry_contract",
            "Queue-first guarded entry contract",
            "passed" if bool(queue_first_entry_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("dataset_queue"),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm generated handoffs include a guarded datasets-waiting landing page, guarded dataset queue, dashboard fallback, and identity check.",
                "Confirm `/`, `/me`, and `/datasets` are datasets-waiting aliases for the labeler and do not require local Palette or Crimson installation.",
                "Confirm packages missing a deployed base URL are not shared as ready-row drafts or labeler links.",
            ],
            details=(
                "Labeler entry is queue-first: open the guarded datasets-waiting landing or dataset queue, confirm identity, then use the dashboard only as a fallback."
            ),
        ),
        _validation_gate(
            "identity_probe_link_contract",
            "Identity probe link contract",
            "passed" if bool(identity_probe_link_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm generated handoffs include expected-user identity probe links.",
                "Confirm each identity probe is opened in the deployed auth context before labeler link sharing.",
            ],
            details=(
                "Generated artifacts expose identity probe entrypoints, but only deployed runtime verification proves the identity source maps browsers to assignment users."
            ),
        ),
        _validation_gate(
            "assignment_ownership_contract",
            "Assignment ownership and one-active-owner contract",
            "passed" if bool(assignment_ownership_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("assignments"),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm one active assignment owner per recording.",
                "Confirm assignment_ownership_integrity.duplicate_active_owner_count is zero.",
                "Confirm reassignment replaces the previous owner and closes previous-owner sessions.",
            ],
            details=(
                "Each recording is expected to have one active owner. Duplicate active owners or missing reassignment/session-closure protections block labeler link sharing."
            ),
        ),
        _validation_gate(
            "browser_payload_redaction_contract",
            "Browser payload redaction contract",
            "passed" if bool(browser_payload_redaction_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("work_summary"),
                files.get("dataset_queue"),
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm labeler runtime, mutation, error, and support payloads do not expose raw Zarr paths or direct task scope.",
                "Confirm copied support text contains redacted IDs sufficient for operator lookup.",
                "Confirm unredacted diagnostics remain on operator/admin surfaces only.",
            ],
            details=(
                "Labeler-facing browser payloads are expected to redact storage paths, task scopes, and direct write authority; operators retain separate diagnostics."
            ),
        ),
        _validation_gate(
            "labeler_route_authorization",
            "Labeler route authorization contract",
            "passed" if bool(labeler_route_authorization_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm labeler routes resolve an authenticated browser user before returning data or accepting mutation.",
                "Confirm expected_user mismatches reject before work or session access.",
                "Confirm copied or signed links are rechecked against known user, active assignment, task ownership, task state, runtime operator-validation start gate, and session state.",
            ],
            details=(
                "Forwarded queue, dashboard, and signed task links are entry hints only; "
                "authorization remains server-side and is tied to resolved user, known assignment-store user, expected-user guard, and active assignment."
            ),
        ),
        _validation_gate(
            "signed_link_contract",
            "Signed link entry-hint contract",
            "passed" if bool(signed_link_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("signed_links"),
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm generated task links are task-specific and expected-user bound.",
                "Confirm signed links are not authentication or authorization grants.",
                "Confirm signed links enforce the runtime operator-validation start gate before creating a browser session.",
                "Confirm forwarded signed links recheck browser identity, active assignment, task state, and session state.",
            ],
            details=(
                "Signed task links are short-lived convenience entry hints only. Server-side identity, expected-user, assignment, task, and session checks still authorize access."
            ),
        ),
        _validation_gate(
            "expected_user_guard_contract",
            "Expected-user guard coverage contract",
            "passed" if bool(expected_user_guard_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm generated guarded links include expected_user for the intended labeler.",
                "Confirm entry pages, queue/dashboard APIs, task open/complete APIs, promotion retry, and signed links reject expected-user mismatch.",
                "Confirm forwarded links stop instead of showing or mutating another user's work.",
            ],
            details=(
                "Expected-user guards are copied-link guard rails. Authentication still comes from the deployment, and authorization remains server-side."
            ),
        ),
        _validation_gate(
            "dataset_queue_start_readiness",
            "Dataset queue start readiness",
            dataset_queue_start_status,
            evidence_files=[
                files.get("dataset_queue"),
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("manifest", files.get("index")),
            ],
            operator_evidence=[
                "Confirm dataset_queue_state.blocks_labeler_start is false for every invite-ready labeler.",
                "Confirm dataset_queue_blocked_start_users is empty for batch handoffs.",
                "If any queue blocks start, generate/reopen work if labeling should continue or mark the assignment finished before inviting the labeler.",
            ],
            details=dataset_queue_start_details,
        ),
        _validation_gate(
            "browser_mutation_target_contract",
            "Browser mutation target-token contract",
            "passed" if bool(browser_mutation_target_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm task_state_policy.browser_mutation_target_token is required_current_target_token.",
                "Confirm each mutable browser_workflows write_contract requires target_token.",
                "Confirm stale same-session tab saves reject before mutation during browser smoke.",
            ],
            details=(
                "Server-owned browser mutation targets require an opaque target_token and reject client target selectors. "
                f"Missing target_token workflows: {browser_mutation_target_contract.get('workflows_missing_target_token') or []}."
            ),
        ),
        _validation_gate(
            "browser_mutation_write_policy",
            "Browser mutation write target policy",
            "passed" if bool(browser_mutation_write_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("dataset_queue"),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm authoritative label state is assigned_task_zarr_scope.",
                "Confirm task-scoped training Zarrs are the mutable label data plane.",
                "Confirm task_scoped_training_zarr is the label mutation target kind.",
                "Confirm browser_label_write_target is training_zarr.",
                "Confirm handoff CSV/HTML/JSON artifacts are metadata-only control-plane outputs and not label write targets.",
                "Confirm browser_writes_handoff_csv and browser_writes_intermediate_csv are false.",
                "Confirm browser save routes mutate only server-owned task-scoped Zarr targets after assignment, task-state, session, and target-token checks.",
            ],
            details=(
                "Browser saves are a data-plane operation against server-owned assigned task Zarr scope; "
                "task_scoped_training_zarr is the label mutation target kind; browser_label_write_target is "
                "training_zarr; handoff CSV/HTML/JSON artifacts are metadata_only_control_plane outputs and "
                "are not browser mutation targets; browsers write neither handoff CSVs nor intermediate CSVs."
            ),
        ),
        _validation_gate(
            "mutation_audit_contract",
            "Mutation audit and provenance contract",
            "passed" if bool(mutation_audit_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm successful representative browser mutations append labeling_task_events records.",
                "Confirm browser clients do not receive audit-store write credentials.",
                "Confirm per-workflow write contracts include audit provenance.",
            ],
            details=(
                "Successful mutations are expected to append server-side audit/provenance events; browser clients do not write audit records directly."
            ),
        ),
        _validation_gate(
            "zarr_backup_contract",
            "Zarr backup and rollback ownership contract",
            "passed" if bool(zarr_backup_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("zarr_backup_plan"),
                files.get("zarr_backup_evidence_template"),
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm backup/rollback remains operator-owned.",
                "Confirm labelers do not receive backup paths or direct Zarr edit instructions.",
                "Confirm restore pauses or unassigns affected recordings before rollback.",
            ],
            details=(
                "This static contract records backup ownership and redaction expectations. Actual backup copy/restore evidence is still tracked by mutable_zarr_backup_confirmation."
            ),
        ),
        _validation_gate(
            "session_guard_contract",
            "Session guard and stale-tab rejection contract",
            "passed" if bool(session_guard_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm a current, unexpired session is required before mutation.",
                "Confirm stale tabs reject after reassignment, completion, reopen, expiration, session supersession, or target navigation.",
                "Confirm session closure events are visible to operators during reassignment or repair.",
            ],
            details=(
                "Mutation routes require current session state and current target token before write; stale or superseded browser tabs are rejected before Zarr mutation."
            ),
        ),
        _validation_gate(
            "task_state_contract",
            "Completed-task read-only and operator reopen contract",
            "passed" if bool(task_state_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm completed task open requests reject with task_complete.",
                "Confirm completed task save requests reject with task_complete until operator reopen.",
                "Confirm completing a task closes open sessions and records operator-visible state.",
            ],
            details=(
                "Ordinary labeler mutation is read-only after task completion; more labeling requires operator reopen before a new active session can mutate Zarr."
            ),
        ),
        _validation_gate(
            "operator_authorization_contract",
            "Operator authorization static contract",
            "passed" if bool(operator_authorization_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm admin routes require operator authorization.",
                "Confirm labelers are not operators by default.",
                "Confirm operator authorization does not grant labeler mutation authority.",
            ],
            details=(
                "This static contract records the intended operator/labeler boundary. Runtime admin-user configuration is still tracked by operator_authorization_boundary."
            ),
        ),
        _validation_gate(
            "operator_recovery_contract",
            "Operator recovery static contract",
            "passed" if bool(operator_recovery_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm operator recovery routes are admin-only and require operator authorization.",
                "Confirm reassignment closes previous-owner sessions, completed tasks require operator reopen, and failed promotion retry is operator-only.",
                "Confirm backup/rollback recovery pauses or unassigns affected recordings before restore.",
            ],
            details=(
                "This static contract records the available operator recovery surfaces: reassignment with session closure, task reopen, stale-session closure inspection, failed-promotion retry, and backup/rollback ownership."
            ),
        ),
        _validation_gate(
            "browser_response_security_contract",
            "Browser response security static contract",
            "passed" if bool(browser_response_security_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm the application policy includes no-store, anti-framing, no-sniff, no-referrer, CSP, and permissions headers.",
                "Confirm deployed proxy evidence is recorded separately in browser_response_security_headers.",
            ],
            details=(
                "This static contract records the application's intended browser response security headers. Deployed proxy/header preservation is still tracked by browser_response_security_headers."
            ),
        ),
        _validation_gate(
            "browser_workflow_scope_contract",
            "Browser workflow and task-scope contract",
            "passed" if bool(browser_workflow_scope_contract.get("ready")) else "needs_review",
            evidence_files=[
                files.get("manifest", files.get("index")),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Confirm every requested workflow kind is in browser_workflows.",
                "Confirm browser mutation targets are server-owned and cannot be selected by client indices/components/labels/frames.",
                "Confirm absolute navigation outside task scope rejects with nav_error.",
            ],
            details=(
                "Browser workflows expose only supported workflow kinds. Targets remain server-held and out-of-scope navigation rejects instead of resolving to a different target."
            ),
        ),
        _validation_gate(
            "operator_authorization_boundary",
            "Operator authorization boundary",
            "pending_operator_evidence",
            evidence_files=[
                files.get("validation_log"),
                files.get("manifest", files.get("index")),
            ],
            operator_evidence=[
                "Run `preflight` with the deployment's real auth and --admin-user configuration, or capture `/api/admin/preflight` from the deployed service.",
                "Confirm operator_authorization_policy.admin_routes_require_operator, admin_users_configured, and operator_boundary_ready are true.",
                "Open `/admin` as a non-admin labeler and confirm it returns admin_required.",
            ],
            details=(
                "Generated artifacts can describe the operator boundary, but only runtime preflight proves that the deployed "
                "service has admin users configured and rejects non-admin labelers from operator routes."
            ),
        ),
        _validation_gate(
            "browser_response_security_headers",
            "Browser/proxy response security headers",
            "pending_operator_evidence" if has_browser_entry else "needs_review",
            evidence_files=[
                files.get("validation_log"),
                files.get("manifest", files.get("index")),
                files.get("browser_response_security_evidence_template"),
            ],
            operator_evidence=[
                "Capture response headers from deployed `/datasets` or `/api/me/tasks` as an authenticated test labeler.",
                "Fill and archive browser-response-security-evidence-template.json with the captured headers and operator approval.",
                "Confirm Cache-Control, Pragma, Expires, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Content-Security-Policy, and Permissions-Policy match browser_response_security_policy.headers.",
                "Confirm the proxy does not strip or weaken no-store, anti-framing, MIME-sniffing, referrer, CSP, or permissions headers.",
            ],
            details=(
                "The application emits browser response security headers, but the deployed proxy/browser path must prove it "
                "preserves them before labeler links are shared."
            ),
        ),
        _validation_gate(
            "dashboard_visibility",
            "Personalized dashboard visibility",
            "pending_operator_evidence" if has_browser_entry else "needs_review",
            evidence_files=[
                files.get("handoffs_roster", files.get("labeler_roster")),
                files.get("html_index"),
            ],
            operator_evidence=[
                "Open `/datasets` as a test labeler and confirm only assigned waiting work is visible.",
                "Open `/work` as a test labeler and confirm only assigned recordings are visible.",
                "Confirm the dashboard stops on expected-user mismatch.",
            ],
        ),
        _validation_gate(
            "browser_smoke",
            "Browser smoke for dashboard, admin, session load, save failure, completion, and reopen",
            "pending_operator_evidence",
            evidence_files=[files.get("browser_smoke_evidence_template"), files.get("validation_log")],
            operator_evidence=[
                "Run the browser smoke path and copy results into validation-log-template.md.",
                "Fill and archive browser-smoke-evidence-template.json with identity, browser-only/no-local-install runtime, personalized /my-datasets queue entry, human-readable /labeling alias, personalized /my-work dashboard fallback, queue visibility, expected-user mismatch, support-redaction, completion/read-only, stale-tab, and operator-reopen evidence.",
                "Include support-detail text for at least one induced failure path.",
            ],
        ),
        _validation_gate(
            "disposable_zarr_mutation_smoke",
            "Disposable-zarr mutation smoke",
            "pending_operator_evidence",
            evidence_files=[
                files.get("disposable_zarr_mutation_smoke_evidence_template"),
                files.get("zarr_backup_plan"),
                files.get("zarr_backup_evidence_template"),
                files.get("validation_log"),
            ],
            operator_evidence=[
                "Save through each browser workflow against disposable zarr data.",
                "Fill and archive disposable-zarr-mutation-smoke-evidence-template.json with mutation event IDs, task-scoped training-Zarr write proof, browser_label_write_target=training_zarr, no direct browser Zarr authority, metadata-only handoff/CSV proof, no browser handoff/intermediate CSV writes, rejected browser-supplied CSV/Zarr target selector proof, audit verification, stale-tab rejection, and restore/discard evidence.",
                "Record mutation event IDs and any registry refresh events.",
            ],
        ),
        _validation_gate(
            "mutable_zarr_backup_confirmation",
            "Mutable zarr backup and rollback confirmation",
            backup_status,
            required=backup_required,
            evidence_files=[backup_plan_path, backup_evidence_template_path, files.get("validation_log")],
            operator_evidence=[
                "Reference the launch bundle backup plan or a separately archived zarr-backup-plan export when this artifact does not include one.",
                "Fill and archive zarr-backup-evidence-template.json with backup destination, manifest, verification, restore-test result, and operator approval for each backup-required target.",
                "Record backup destination, owner, and restore test result.",
                "Confirm rollback is operator-owned and not labeler-facing.",
            ],
            details=backup_details,
        ),
        _validation_gate(
            "one_labeler_dry_run",
            "One-operator / one-labeler dry run",
            "pending_operator_evidence",
            evidence_files=[files.get("validation_log")],
            operator_evidence=[
                "Record labeler, recording, task, save event ID, and completion evidence.",
            ],
        ),
        _validation_gate(
            "multi_user_dry_run",
            "Multi-user dry run with one active owner per recording",
            "pending_operator_evidence",
            evidence_files=[files.get("validation_log")],
            operator_evidence=[
                "Record at least two users, recording ownership, guarded URLs, stale-tab rejection, and completed-task read-only behavior.",
                "Confirm assignment_ownership_integrity.duplicate_active_owner_count is zero.",
            ],
        ),
        _validation_gate(
            "final_signoff",
            "Final safe-share sign-off before labeler link sharing",
            "pending_operator_evidence",
            evidence_files=[files.get("validation_log")],
            operator_evidence=[
                "Confirm identity source, readiness report, backup/rollback path, and safe-share approval (`labeler_links_safe_to_share=true`) before contacting labelers.",
            ],
        ),
    ]
    gate_counts: dict[str, int] = {}
    for gate in gates:
        status = str(gate.get("status") or "unknown")
        gate_counts[status] = gate_counts.get(status, 0) + 1
    gate_classification = _validation_gate_classification(gates)
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
    operator_validation_visibility_policy = (
        manifest.get("operator_validation_visibility_policy")
        if isinstance(manifest.get("operator_validation_visibility_policy"), Mapping)
        else _operator_validation_visibility_policy()
    )
    safe_share_checklist_fields = _safe_share_checklist_gate_status_fields(
        gates=gates,
        safe_share_gate=safe_share_gate,
    )
    return {
        "schema": "palette.web_labeling_validation_checklist.v1",
        "bundle_label": bundle_label,
        "operator_validation_visibility_policy": operator_validation_visibility_policy,
        "operator_validation_command_templates": operator_validation_command_templates,
        "store_path": str(manifest.get("store_path") or ""),
        "base_url": base_url or None,
        "labeler_landing_page_path": labeler_landing_page_path or "/",
        "labeler_landing_url": labeler_landing_url,
        "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
        "labeling_home_page_path": labeling_home_page_path or LABELING_HOME_PATH,
        "labeling_home_url": labeling_home_url,
        "expected_user_labeling_home_url": expected_user_labeling_home_url,
        "dashboard_url": dashboard_url,
        "expected_user_dashboard_url": expected_user_dashboard_url,
        "expected_user_identity_probe_url": expected_user_identity_probe_url,
        "dataset_queue_page_path": dataset_queue_page_path or DATASET_QUEUE_PATH,
        "dataset_queue_url": dataset_queue_url,
        "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
        "personal_dataset_queue_page_path": personal_dataset_queue_page_path
        or PERSONAL_DATASET_QUEUE_PATH,
        "personal_dataset_queue_url": personal_dataset_queue_url,
        "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
        "personal_work_page_path": personal_work_page_path or PERSONAL_WORK_PATH,
        "personal_work_url": personal_work_url,
        "expected_user_personal_work_url": expected_user_personal_work_url,
        "personalized_labeler_entrypoint": str(
            queue_first_entry_contract.get("personalized_labeler_entrypoint") or ""
        ),
        "personalized_labeler_entry_url": str(
            queue_first_entry_contract.get("personalized_labeler_entry_url") or ""
        ),
        "single_owner_policy": single_owner_policy,
        "labeler_safety": labeler_safety,
        "queue_first_entry_contract": queue_first_entry_contract,
        "identity_probe_link_contract": identity_probe_link_contract,
        "identity_source_evidence_template": identity_source_evidence_template_path,
        "browser_payload_redaction_contract": browser_payload_redaction_contract,
        "operator_authorization_policy": operator_authorization_policy,
        "operator_authorization_contract": operator_authorization_contract,
        "operator_recovery_policy": operator_recovery_policy,
        "operator_recovery_contract": operator_recovery_contract,
        "labeler_route_authorization_policy": labeler_route_authorization_policy,
        "labeler_route_authorization_contract": labeler_route_authorization_contract,
        "zarr_backup_policy": zarr_backup_policy,
        "zarr_backup_contract": zarr_backup_contract,
        "zarr_backup_evidence_template": backup_evidence_template_path,
        "mutation_audit_policy": mutation_audit_policy,
        "mutation_audit_contract": mutation_audit_contract,
        "browser_mutation_write_policy": browser_mutation_write_policy,
        "browser_mutation_write_contract": browser_mutation_write_contract,
        "dataset_queue_direct_start_policy": dataset_queue_direct_start_policy,
        "runtime_operator_validation_gate_cli_policy": runtime_gate_cli_policy,
        **_runtime_operator_validation_gate_cli_policy_fields(runtime_gate_cli_policy),
        "safe_share_gate": safe_share_gate,
        **safe_share_fields,
        **safe_share_checklist_fields,
        "browser_response_security_policy": browser_response_security_policy,
        "browser_response_security_contract": browser_response_security_contract,
        "browser_response_security_evidence_template": response_security_evidence_template_path,
        "browser_smoke_evidence_template": browser_smoke_evidence_template_path,
        "disposable_zarr_mutation_smoke_evidence_template": (
            disposable_zarr_mutation_smoke_evidence_template_path
        ),
        "implementation_status": implementation_status_path,
        "implementation_status_artifact": implementation_status_artifact,
        "implementation_status_artifact_required_fields": list(
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS
        ),
        "implementation_status_artifact_required_field_count": (
            _IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELD_COUNT
        ),
        "implementation_status_flat_fields": list(
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELDS
        ),
        "implementation_status_flat_field_count": (
            _IMPLEMENTATION_STATUS_PAYLOAD_FLAT_FIELD_COUNT
        ),
        **_implementation_status_flat_fields_from_artifact(implementation_status_artifact),
        "session_guard_policy": session_guard_policy,
        "session_guard_contract": session_guard_contract,
        "task_state_policy": task_state_policy,
        "task_state_contract": task_state_contract,
        "browser_workflow_scope_contract": browser_workflow_scope_contract,
        "signed_link_policy": signed_link_policy,
        "signed_link_contract": signed_link_contract,
        "expected_user_guard_contract": expected_user_guard_contract,
        "browser_workflows": browser_workflow_mappings,
        "browser_mutation_target_contract": browser_mutation_target_contract,
        "dataset_queue_state": dataset_queue_state,
        "dataset_queue_states": dataset_queue_states,
        "dataset_queue_blocked_start_users": dataset_queue_blocked_start_users,
        "assignment_ownership_integrity": assignment_ownership_integrity,
        "assignment_ownership_contract": assignment_ownership_contract,
        "generated_at_utc": manifest.get("generated_at_utc"),
        "dry_run": dry_run,
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
        "validation_log": str(files.get("validation_log") or ""),
        "all_validation_complete": not any(
            str(gate.get("status") or "") in {"needs_review", "pending_operator_evidence"}
            for gate in gates
            if bool(gate.get("required"))
        ),
        "ready_for_operator_validation": not any(str(gate.get("status") or "") == "needs_review" for gate in gates),
        "counts": {
            "gates": len(gates),
            "assignment_ownership_duplicate_active_owners": int(
                assignment_ownership_integrity.get("duplicate_active_owner_count") or 0
            ),
            "assignment_ownership_unique_active_recordings": int(
                assignment_ownership_integrity.get("unique_active_recording_count") or 0
            ),
            "queue_first_entry_contract_ready": int(bool(queue_first_entry_contract.get("ready"))),
            "identity_probe_link_contract_ready": int(bool(identity_probe_link_contract.get("ready"))),
            "identity_source_evidence_template_present": int(
                bool(identity_source_evidence_template_path)
            ),
            "assignment_ownership_contract_ready": int(bool(assignment_ownership_contract.get("ready"))),
            "browser_payload_redaction_contract_ready": int(
                bool(browser_payload_redaction_contract.get("ready"))
            ),
            "operator_authorization_contract_ready": int(bool(operator_authorization_contract.get("ready"))),
            "operator_recovery_contract_ready": int(bool(operator_recovery_contract.get("ready"))),
            "browser_response_security_contract_ready": int(
                bool(browser_response_security_contract.get("ready"))
            ),
            "browser_response_security_evidence_template_present": int(
                bool(response_security_evidence_template_path)
            ),
            "browser_smoke_evidence_template_present": int(bool(browser_smoke_evidence_template_path)),
            "disposable_zarr_mutation_smoke_evidence_template_present": int(
                bool(disposable_zarr_mutation_smoke_evidence_template_path)
            ),
            "implementation_status_present": int(bool(implementation_status_path)),
            "zarr_backup_contract_ready": int(bool(zarr_backup_contract.get("ready"))),
            "zarr_backup_evidence_template_present": int(bool(backup_evidence_template_path)),
            "browser_mutation_target_contract_ready": int(bool(browser_mutation_target_contract.get("ready"))),
            "browser_mutation_write_contract_ready": int(bool(browser_mutation_write_contract.get("ready"))),
            "labeler_route_authorization_contract_ready": int(
                bool(labeler_route_authorization_contract.get("ready"))
            ),
            "signed_link_contract_ready": int(bool(signed_link_contract.get("ready"))),
            "expected_user_guard_contract_ready": int(bool(expected_user_guard_contract.get("ready"))),
            "mutation_audit_contract_ready": int(bool(mutation_audit_contract.get("ready"))),
            "session_guard_contract_ready": int(bool(session_guard_contract.get("ready"))),
            "task_state_contract_ready": int(bool(task_state_contract.get("ready"))),
            "browser_workflow_scope_contract_ready": int(bool(browser_workflow_scope_contract.get("ready"))),
            "browser_workflow_contracts_missing_target_token": len(
                browser_mutation_target_contract.get("workflows_missing_target_token")
                if isinstance(browser_mutation_target_contract.get("workflows_missing_target_token"), list)
                else []
            ),
            "operator_evidence_gates": int(gate_classification["operator_evidence_gate_count"]),
            "generated_contract_gates": int(gate_classification["generated_contract_gate_count"]),
            "operator_evidence_pending_gates": int(
                gate_classification["operator_evidence_pending_gate_count"]
            ),
            "operator_evidence_needs_review_gates": int(
                gate_classification["operator_evidence_needs_review_gate_count"]
            ),
            "generated_contract_failed_gates": int(
                gate_classification["generated_contract_failed_gate_count"]
            ),
            **dict(sorted(gate_counts.items())),
        },
        "gates": gates,
    }


def _write_web_labeling_validation_checklist_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    bundle_label: str,
) -> None:
    payload = _web_labeling_validation_checklist_payload(manifest, bundle_label=bundle_label)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# Preserve original helper names inside this module so moved helpers can
# continue to call each other exactly as they did in web.py.
_write_web_labeling_validation_log = _write_web_labeling_validation_log_impl
_web_labeling_validation_checklist_payload = _web_labeling_validation_checklist_payload_impl
_write_web_labeling_validation_checklist = _write_web_labeling_validation_checklist_impl
