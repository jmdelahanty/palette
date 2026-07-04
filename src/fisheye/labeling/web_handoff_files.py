"""User-handoff file writers for web labeling launch artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .web_report_renderers import (
    _dashboard_url_for_base,
    _safe_share_next_action_summary_text,
)

def _user_handoff_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "work_summary": output_dir / "work-summary.json",
        "dataset_queue": output_dir / "dataset-queue.json",
        "signed_links": output_dir / "signed-links.jsonl",
        "store_check": output_dir / "check-store.json",
        "manifest": output_dir / "manifest.json",
        "html_index": output_dir / "index.html",
        "message": output_dir / "message.txt",
        "quickstart": output_dir / "labeler-quickstart.txt",
        "validation_log": output_dir / "validation-log-template.md",
        "validation_checklist": output_dir / "validation-checklist.json",
    }


def _safe_user_handoff_dir_name(user: str, used_names: set[str]) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._-"
    base = "".join(char if char in allowed else "_" for char in str(user).strip()).strip("._-")
    if not base:
        base = "user"
    candidate = base
    index = 2
    while candidate in used_names:
        candidate = f"{base}-{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _write_user_handoffs_readme(index: dict[str, object], output_path: Path) -> None:
    counts = index.get("counts") if isinstance(index.get("counts"), dict) else {}
    files = index.get("files") if isinstance(index.get("files"), dict) else {}
    progress_summary = index.get("progress_summary") if isinstance(index.get("progress_summary"), dict) else {}
    safe_share_readme_gap_statuses: dict[str, str] = {}
    for handoff in index.get("handoffs", []):
        if not isinstance(handoff, Mapping):
            continue
        handoff_gap_statuses = handoff.get("safe_share_external_launch_evidence_gap_statuses")
        if not isinstance(handoff_gap_statuses, Mapping):
            continue
        safe_share_readme_gap_statuses.update(
            {
                str(gate_id): str(status)
                for gate_id, status in handoff_gap_statuses.items()
                if str(gate_id).strip()
            }
        )
    if not safe_share_readme_gap_statuses and isinstance(
        index.get("safe_share_external_launch_evidence_gap_statuses"),
        Mapping,
    ):
        safe_share_readme_gap_statuses.update(
            {
                str(gate_id): str(status)
                for gate_id, status in index["safe_share_external_launch_evidence_gap_statuses"].items()
                if str(gate_id).strip()
            }
        )
    safe_share_readme_status_text = ", ".join(
        f"{gate_id}={status}"
        for gate_id, status in sorted(safe_share_readme_gap_statuses.items())
    )
    safe_share_readme_next_action_count = (
        len(safe_share_readme_gap_statuses)
        if safe_share_readme_gap_statuses
        else int(index.get("safe_share_launch_blocking_next_action_count") or 0)
    )
    lines = [
        "Palette labeling handoff bundle",
        "",
        "Open index.html for an operator overview of all generated labeler handoffs.",
        "Each labeler subdirectory contains an index.html, message.txt, labeler-quickstart.txt, assigned tasks, and signed task links.",
        "Fill validation-log-template.md during static, browser, real-zarr, and deployed dry-run validation.",
        "",
        "Important:",
        "- Signed links are convenience entry points; the service still requires authenticated access.",
        "- Forwarded queue, dashboard, and task links are rechecked against the resolved browser user, expected-user guard, assignment, task, runtime operator-validation start gate, and session state.",
        "- Links expire after the configured TTL and can be revoked with the service not-before floor.",
        "- If any handoff status is not ok, inspect that labeler's check-store.json before sharing.",
        "- Use the guarded landing page or personalized dataset queue as the queue-first entry point; use the full dashboard after that.",
        "- Handoff CSV, HTML, and JSON files are metadata only; browser saves mutate server-owned assigned task Zarr targets and append audit events.",
        "- Implementation status: browser labeling contracts are generated into this bundle, but link sharing is not approved until operator evidence gates pass.",
        "- Implementation status checklist: docs/web_labeling_implementation_status.md.",
        "- Code contracts cover guarded personalized queues, server-owned task/training Zarr writes, metadata-only CSV/handoff artifacts, one-owner assignment policy, and signed links as entry hints that enforce runtime start gates before session creation.",
        "- Launch evidence still requires approved mutable-Zarr backup, response-security, identity-source, browser-smoke, disposable-Zarr mutation, and operator-recovery evidence.",
        "- inspect-handoff blocks sharing if browser mutation or direct Start/Open metadata stops proving task-scoped training-Zarr targets, no handoff/intermediate CSV writes, no direct browser Zarr authority, and expected-user server rechecks.",
        "- inspect-handoff also blocks sharing if validation-checklist.json lacks a complete implementation_status_artifact contract or if one recording appears under multiple active labelers in the package; wrapper fields include implementation_status_checklist_artifact_complete, implementation_status_checklist_artifact_complete_matches_required_value, browser_mutation_target_contract_met, direct_browser_start_contract_met, single_owner_package_contract_met, and labeler_route_authorization_runtime_checklist_gate_met.",
        "- If Base URL is '(relative links)', handoff pages preview work but labelers need the service dashboard URL before opening tasks.",
        "- Do not send links solely because per-user handoffs say ready_to_send; first run inspect-handoff --path PACKAGE --require-shareable and require labeler_links_safe_to_share=true.",
        "- Unapproved mutable-Zarr backup, browser response-security, identity-source, browser-smoke, disposable-Zarr mutation, or operator-recovery evidence gates are launch blockers.",
        f"- Safe-share next actions: {safe_share_readme_next_action_count}",
        f"- Safe-share gate statuses: {safe_share_readme_status_text}",
        f"- {_safe_share_next_action_summary_text(index)}",
        "",
        f"Store: {index.get('store_path', '')}",
        f"Base URL: {index.get('base_url') or '(relative links)'}",
        f"Landing URL: {index.get('labeler_landing_url') or '(missing --base-url)'}",
        f"Labeling home URL: {index.get('labeling_home_url') or '(missing --base-url)'}",
        f"Dashboard URL: {index.get('dashboard_url') or '(missing --base-url)'}",
        f"Personalized dataset queue URL: {index.get('personal_dataset_queue_url') or '(missing --base-url)'}",
        f"Personalized work URL: {index.get('personal_work_url') or '(missing --base-url)'}",
        f"Dataset queue URL: {index.get('dataset_queue_url') or '(missing --base-url)'}",
        f"Validation log: {files.get('validation_log', '')}",
        f"Link TTL seconds: {index.get('ttl_seconds', '')}",
        f"Generated at UTC: {index.get('generated_at_utc', '')}",
        f"Users exported: {counts.get('users', 0)}",
        f"Ready to send: {counts.get('ready_to_send', 0)}",
        f"Not ready to send: {counts.get('not_ready_to_send', 0)}",
        f"Waiting recordings: {progress_summary.get('waiting_recording_count', 0)}",
        f"Complete recordings: {progress_summary.get('complete_recording_count', 0)}",
        f"Blocked/no-open recordings: {progress_summary.get('blocked_recording_count', 0)}",
        f"Waiting datasets: {(index.get('dataset_queue_summary') or {}).get('waiting_dataset_count', 0) if isinstance(index.get('dataset_queue_summary'), dict) else 0}",
        f"Blocked/no-open recordings by reason: {json.dumps(progress_summary.get('blocked_recordings_by_reason', {}), sort_keys=True)}",
        f"Store checks ok: {index.get('store_checks_ok', '')}",
        f"Not-ready reasons: {json.dumps(counts.get('sendability_reasons', {}), sort_keys=True)}",
        f"Not-ready actions: {' '.join(str(action) for action in (index.get('sendability_actions') if isinstance(index.get('sendability_actions'), list) else []))}",
        f"Failed store checks: {counts.get('failed_store_checks', 0)}",
        f"Assigned recordings without startable tasks: {counts.get('recordings_without_open_tasks', 0)}",
        f"Assigned recordings without startable tasks by reason: {json.dumps(counts.get('recordings_without_open_tasks_by_reason', {}), sort_keys=True)}",
        f"Assigned recordings without startable task actions: {' '.join(str(action) for action in (counts.get('recordings_without_open_tasks_actions') if isinstance(counts.get('recordings_without_open_tasks_actions'), list) else []))}",
        f"Redacted user-summary fields: {counts.get('redacted_summary_fields', 0)}",
        "",
        "Top-level files:",
        "- index.html: operator overview",
        "- index.json: machine-readable batch manifest",
        "- handoff-readme.txt: this file",
        "- validation-log-template.md: operator validation and sign-off log",
        "- validation-checklist.json: machine-readable launch validation gates and pending operator evidence",
        "- zarr-backup-evidence-template.json: operator-only mutable-Zarr backup confirmation template",
        "- browser-response-security-evidence-template.json: operator-only deployed header capture template",
        "- identity-source-evidence-template.json: operator-only deployed identity probe capture template",
        "- browser-smoke-evidence-template.json: operator-only queue-first browser smoke capture template",
        "- disposable-zarr-mutation-smoke-evidence-template.json: operator-only disposable-Zarr mutation smoke capture template",
        "- USER/dataset-queue.json: per-labeler assigned open-work dataset queue with guarded dataset queue page links and dashboard filter links",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_user_handoff_quickstart(
    *,
    user: str,
    manifest: dict[str, object],
    output_path: Path,
) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    base_url = str(manifest.get("base_url") or "").rstrip("/")
    landing_url = str(manifest.get("expected_user_labeler_landing_url") or manifest.get("labeler_landing_url") or base_url).strip()
    dashboard_url = str(manifest.get("expected_user_dashboard_url") or _dashboard_url_for_base(base_url))
    dataset_queue_url = str(manifest.get("expected_user_dataset_queue_url") or manifest.get("dataset_queue_url") or "").strip()
    labeling_home_url = str(
        manifest.get("expected_user_labeling_home_url")
        or manifest.get("labeling_home_url")
        or ""
    ).strip()
    personalized_entry_url = str(
        manifest.get("personalized_labeler_entry_url")
        or manifest.get("expected_user_personal_dataset_queue_url")
        or ""
    ).strip()
    identity_probe_url = str(manifest.get("expected_user_identity_probe_url") or "").strip()
    ready_to_send = bool(manifest.get("ready_to_send"))
    dataset_queue_state = (
        manifest.get("dataset_queue_state")
        if isinstance(manifest.get("dataset_queue_state"), Mapping)
        else {}
    )
    dataset_queue_state_code = str(dataset_queue_state.get("code") or "")
    dataset_queue_state_title = str(dataset_queue_state.get("title") or "")
    dataset_queue_state_message = str(dataset_queue_state.get("message") or "")
    dataset_queue_operator_action = str(dataset_queue_state.get("operator_action") or "")
    dataset_queue_blocks_start = bool(dataset_queue_state.get("blocks_labeler_start"))
    warning_reasons: list[str] = []
    warnings = manifest.get("sendability_warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if not isinstance(warning, Mapping):
                continue
            reasons = warning.get("reasons")
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                reason_text = str(reason or "").strip()
                if reason_text and reason_text not in warning_reasons:
                    warning_reasons.append(reason_text)
    sendability_actions = [
        str(action).strip()
        for action in (manifest.get("sendability_actions") if isinstance(manifest.get("sendability_actions"), list) else [])
        if str(action).strip()
    ]
    lines = [
        "Palette labeling quickstart",
        "",
        f"User: {user}",
        f"Recordings: {counts.get('recordings', 0)}",
        f"Tasks: {counts.get('tasks', 0)}",
        f"Signed links: {counts.get('signed_links', 0)}",
        f"Generated at UTC: {manifest.get('generated_at_utc', '')}",
        f"Links expire at UTC: {manifest.get('links_expire_at_utc', '')}",
        "",
        "No Palette or Crimson installation is needed for this browser workflow.",
        "",
        "Readiness:",
        "This handoff is ready to use." if ready_to_send else "Wait for operator review before starting.",
        f"Review reasons: {', '.join(warning_reasons) if warning_reasons else 'none' if ready_to_send else 'operator_review_required'}",
        f"Dataset queue state: {dataset_queue_state_code or 'unknown'}{(' - ' + dataset_queue_state_title) if dataset_queue_state_title else ''}",
        f"Dataset queue start: {'blocked' if dataset_queue_blocks_start else 'allowed'}",
    ]
    if dataset_queue_state_message:
        lines.append(f"Dataset queue message: {dataset_queue_state_message}")
    if dataset_queue_operator_action:
        lines.append(f"Dataset queue operator action: {dataset_queue_operator_action}")
    if dataset_queue_blocks_start:
        lines.append("Do not start new labeling from the dataset queue until the operator resolves this state.")
    if sendability_actions:
        lines.extend(["Repair actions:", *[f"- {action}" for action in sendability_actions]])
    lines.extend(
        [
            "",
            "How to start:" if ready_to_send else "How to preview while waiting:",
        ]
    )
    if dashboard_url:
        queue_start_url = personalized_entry_url or dataset_queue_url or dashboard_url
        if ready_to_send:
            lines.extend(
                [
                    f"1. Open the identity check and confirm it reports you as {user}: {identity_probe_url or '(operator did not provide an identity probe URL)'}",
                    f"2. Open your datasets-waiting landing page: {landing_url or dataset_queue_url or dashboard_url}",
                    f"3. Human-readable labeling home alias: {labeling_home_url or landing_url or dataset_queue_url or dashboard_url}",
                    f"4. Open your personalized dataset queue as the preferred queue-first view: {queue_start_url}",
                    f"5. Canonical dataset queue fallback: {dataset_queue_url or dashboard_url}",
                    f"6. Open your full dashboard when you need the fallback view: {dashboard_url}",
                    f"7. Confirm the dashboard shows you as {user}; if it shows another user, stop and contact the operator.",
                    "8. Open one assigned task at a time and save through the browser controls.",
                    "9. Browser saves are applied server-side to your assigned task/training Zarr scope; CSV, HTML, JSON, and handoff files are metadata only.",
                    "10. Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.",
                    "11. Labelers should not run operator evidence, repair, checksum, or validation commands; those commands are operator-only launch controls.",
                ]
            )
        else:
            lines.extend(
                [
                    "1. Open index.html from this handoff directory to preview your assigned work.",
                    "2. Do not open or save task work until the operator confirms this handoff is ready.",
                    f"3. Open the identity check and confirm it reports you as {user}: {identity_probe_url or '(operator did not provide an identity probe URL)'}",
                    f"4. Open your datasets-waiting landing page: {landing_url or dataset_queue_url or dashboard_url}",
                    f"5. Human-readable labeling home alias: {labeling_home_url or landing_url or dataset_queue_url or dashboard_url}",
                    f"6. Then use your personalized dataset queue as the preferred queue-first view or dashboard. Confirm the dashboard shows you as {user}: {queue_start_url}",
                    f"7. Open your full dashboard preview when needed: {dashboard_url}",
                    f"8. Canonical dataset queue fallback: {dataset_queue_url or dashboard_url}",
                ]
            )
    else:
        lines.extend(
            [
                "1. Open index.html from this handoff directory to preview your assigned work.",
                "2. Ask the operator for the service dashboard URL before opening tasks.",
                f"3. If the operator gives you a service URL, confirm the dashboard shows you as {user} before opening work.",
            ]
        )
    lines.extend(
        [
            "",
            "Safety rules:",
            "- Do not edit zarr files directly.",
            "- Browser saves are applied server-side to your assigned task/training Zarr scope.",
            "- CSV, HTML, JSON, and handoff files are metadata only and are not label write targets.",
            "- Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.",
            "- Labelers should not run operator evidence, repair, checksum, or validation commands; those commands are operator-only launch controls.",
            "- Do not forward this handoff or signed links to another user.",
            "- Do not keep working after the operator says a recording was reassigned or paused.",
            "- Stop before opening work if the dashboard shows a different signed-in user.",
            "- Use the task completion controls only after the assigned review/edit is finished.",
            "",
            "Ask the operator for help if:",
            "- the dashboard does not show the expected recording",
            "- a signed link is expired",
            "- the handoff was generated without a service URL",
            "- a task is missing",
            "- saving fails",
            "- the browser shows another user's work",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_user_handoff_message(
    *,
    user: str,
    manifest: dict[str, object],
    output_path: Path,
) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    base_url = str(manifest.get("base_url") or "").rstrip("/")
    landing_url = str(manifest.get("expected_user_labeler_landing_url") or manifest.get("labeler_landing_url") or base_url).strip() or None
    dashboard_url = str(manifest.get("expected_user_dashboard_url") or _dashboard_url_for_base(base_url)).strip() or None
    dataset_queue_url = str(manifest.get("expected_user_dataset_queue_url") or manifest.get("dataset_queue_url") or "").strip() or None
    labeling_home_url = (
        str(
            manifest.get("expected_user_labeling_home_url")
            or manifest.get("labeling_home_url")
            or ""
        ).strip()
        or None
    )
    personalized_entry_url = (
        str(
            manifest.get("personalized_labeler_entry_url")
            or manifest.get("expected_user_personal_dataset_queue_url")
            or ""
        ).strip()
        or None
    )
    identity_probe_url = str(manifest.get("expected_user_identity_probe_url") or "").strip() or None
    ready_to_send = bool(manifest.get("ready_to_send"))
    dataset_queue_state = (
        manifest.get("dataset_queue_state")
        if isinstance(manifest.get("dataset_queue_state"), Mapping)
        else {}
    )
    dataset_queue_state_code = str(dataset_queue_state.get("code") or "")
    dataset_queue_state_title = str(dataset_queue_state.get("title") or "")
    dataset_queue_state_message = str(dataset_queue_state.get("message") or "")
    dataset_queue_operator_action = str(dataset_queue_state.get("operator_action") or "")
    dataset_queue_blocks_start = bool(dataset_queue_state.get("blocks_labeler_start"))
    warning_reasons: list[str] = []
    warnings = manifest.get("sendability_warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if not isinstance(warning, Mapping):
                continue
            reasons = warning.get("reasons")
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                reason_text = str(reason or "").strip()
                if reason_text and reason_text not in warning_reasons:
                    warning_reasons.append(reason_text)
    sendability_actions = [
        str(action).strip()
        for action in (manifest.get("sendability_actions") if isinstance(manifest.get("sendability_actions"), list) else [])
        if str(action).strip()
    ]
    lines = [
        "Your Palette labeling work is ready."
        if ready_to_send
        else "Your Palette labeling handoff needs operator review before starting.",
        "",
        f"User: {user}",
        f"Recordings: {counts.get('recordings', 0)}",
        f"Tasks: {counts.get('tasks', 0)}",
        f"Signed links: {counts.get('signed_links', 0)}",
        f"Link TTL seconds: {manifest.get('ttl_seconds', '')}",
        f"Generated at UTC: {manifest.get('generated_at_utc', '')}",
        f"Links expire at UTC: {manifest.get('links_expire_at_utc', '')}",
        "",
        "No Palette or Crimson installation is needed for this browser workflow.",
        "Read labeler-quickstart.txt before starting.",
        f"Dataset queue state: {dataset_queue_state_code or 'unknown'}{(' - ' + dataset_queue_state_title) if dataset_queue_state_title else ''}",
        f"Dataset queue start: {'blocked' if dataset_queue_blocks_start else 'allowed'}",
        "",
    ]
    if dataset_queue_state_message:
        lines.extend(["Dataset queue message:", dataset_queue_state_message, ""])
    if dataset_queue_operator_action:
        lines.extend(["Dataset queue operator action:", dataset_queue_operator_action, ""])
    if dataset_queue_blocks_start:
        lines.extend(
            [
                "Do not start new labeling from the dataset queue until the operator resolves this state.",
                "",
            ]
        )
    if not ready_to_send:
        lines.extend(
            [
                "Do not start labeling from this handoff until the operator confirms it is ready.",
                f"Review reasons: {', '.join(warning_reasons) if warning_reasons else 'operator_review_required'}",
                *(
                    ["Repair actions:", *[f"- {action}" for action in sendability_actions]]
                    if sendability_actions
                    else []
                ),
                "",
            ]
        )
    if dashboard_url:
        queue_start_url = personalized_entry_url or dataset_queue_url or landing_url or dashboard_url
        lines.extend(
            [
                "Identity check:",
                identity_probe_url or "(operator did not provide an identity probe URL)",
                f"Open this first and confirm it reports you as {user} before labeling.",
                "",
                "Preferred queue-first entry point:" if ready_to_send else "Preview queue-first entry point:",
                queue_start_url or "(operator did not provide a dataset queue URL)",
                "",
                "Start here:",
                queue_start_url or "(operator did not provide a dataset queue URL)",
                "",
                "Queue-first start page:",
                landing_url or dataset_queue_url or dashboard_url or "(operator did not provide a landing URL)",
                "",
                "Human-readable labeling home alias:",
                labeling_home_url or landing_url or dataset_queue_url or dashboard_url or "(operator did not provide a labeling home URL)",
                "",
                "Personalized dataset queue:",
                personalized_entry_url or dataset_queue_url or dashboard_url or "(operator did not provide a dataset queue URL)",
                "",
                "Canonical dataset queue fallback:",
                dataset_queue_url or dashboard_url or "(operator did not provide a dataset queue URL)",
                "",
                "Full dashboard fallback:" if ready_to_send else "Preview dashboard:",
                dashboard_url,
                f"Before opening work, confirm the dashboard shows you as {user}. If it shows another user, stop and contact the operator.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "This handoff was generated without a service base URL.",
                "Open index.html to preview your work, then ask the operator for the service dashboard URL before opening tasks.",
                f"Before opening work from any service URL, confirm the dashboard shows you as {user}. If it shows another user, stop and contact the operator.",
                "",
            ]
        )
    lines.extend(
        [
            "If you were given this handoff bundle, open index.html in this directory.",
            "The included task links are convenience entry points and still require authenticated access.",
            "Do not edit zarr files directly or forward this handoff to another user.",
            "Browser saves are applied server-side to your assigned task/training Zarr scope; CSV, HTML, JSON, and handoff files are metadata only. "
            "Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.",
            "Labelers should not run operator evidence, repair, checksum, or validation commands; those commands are operator-only launch controls.",
            "If a link is expired or a task is missing, ask the operator to regenerate your handoff.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")

