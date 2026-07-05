"""Static batch and handoff HTML report renderers for labeling workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .admin_dashboard import (
    _assignment_ownership_contract_fields,
    _assignment_ownership_contract_policy,
    _assignment_ownership_policy,
    _handoff_browser_mutation_write_fields,
    _handoff_known_user_status_fields,
    _safe_share_checklist_gate_status_fields_from_operator_validation,
    _safe_share_gate_policy,
    _safe_share_next_action_summary_from_fields,
    _single_owner_policy_fields,
)
from .web_auth import DASHBOARD_PATH
from .work_queue import _dataset_queue_state

__all__ = [
    "_IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE",
    "_IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT",
    "_IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE",
    "_IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE",
    "_IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE",
    "_IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE",
    "_RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE",
    "_SHAREABILITY_COMPACT_CONTRACT_SENTENCE",
    "_SHAREABILITY_COMPACT_GATE_SENTENCE",
    "_SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE",
    "_SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE",
    "_dashboard_url_for_base",
    "_dashboard_roster_html",
    "_dashboard_ready_invitation_bundle",
    "_handoff_assignment_ownership_fields",
    "_handoff_dataset_queue_blocks_labeler_start",
    "_handoff_dataset_queue_start_fields",
    "_handoff_dataset_queue_state",
    "_handoff_entry_artifact_fields",
    "_handoff_no_open_task_message",
    "_handoff_relative_href",
    "_html_escape",
    "_safe_share_next_action_summary_text",
    "_write_batch_plan_html_report",
    "_write_launch_bundle_html_index",
    "_write_user_handoff_html_index",
    "_write_user_handoffs_html_index",
]

def _safe_share_next_action_summary_text(source: Mapping[str, object]) -> str:
    explicit_summary = str(source.get("safe_share_next_action_summary") or "").strip()
    if explicit_summary:
        return explicit_summary
    actions = (
        source.get("safe_share_launch_blocking_next_actions")
        if isinstance(source.get("safe_share_launch_blocking_next_actions"), list)
        else []
    )
    statuses = (
        source.get("safe_share_launch_blocking_gate_statuses")
        if isinstance(source.get("safe_share_launch_blocking_gate_statuses"), Mapping)
        else {}
    )
    summary = _safe_share_next_action_summary_from_fields(
        actions=actions,
        statuses=statuses,
        count=int(source.get("safe_share_launch_blocking_next_action_count") or 0) or None,
    )
    if "Safe-share next actions: 0;" not in summary:
        return summary
    if (
        not actions
        and not statuses
        and not source.get("safe_share_launch_blocking_next_action_count")
    ):
        derived_fields = _safe_share_checklist_gate_status_fields_from_operator_validation(
            source,
            safe_share_gate=_safe_share_gate_policy(),
        )
        if derived_fields.get("safe_share_launch_blocking_next_actions"):
            return _safe_share_next_action_summary_text(derived_fields)
        fallback_gate_ids = _safe_share_gate_policy().get("launch_blocking_evidence_gate_ids")
        if isinstance(fallback_gate_ids, list):
            fallback_statuses = {
                str(gate_id): "unknown"
                for gate_id in fallback_gate_ids
                if str(gate_id).strip()
            }
            return _safe_share_next_action_summary_from_fields(
                actions=[],
                statuses=fallback_statuses,
                count=len(fallback_statuses),
            )
    return summary

def _html_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def _dashboard_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{DASHBOARD_PATH}"

def _handoff_relative_href(index_dir: Path, value: object) -> str:
    path = Path(str(value))
    try:
        return path.relative_to(index_dir).as_posix()
    except ValueError:
        return path.as_posix()

def _handoff_dataset_queue_state(manifest: Mapping[str, object]) -> dict[str, object]:
    state = manifest.get("dataset_queue_state")
    if isinstance(state, Mapping):
        return dict(state)
    return _dataset_queue_state(
        {
            "empty_state": manifest.get("empty_state", {}),
            "progress_summary": manifest.get("progress_summary", {}),
            "dataset_queue_summary": manifest.get("dataset_queue_summary", {}),
            "dataset_queue": manifest.get("dataset_queue", []),
        }
    )

def _handoff_dataset_queue_blocks_labeler_start(handoff: Mapping[str, object]) -> bool:
    def flag(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    explicit_value = handoff.get("dataset_queue_blocks_labeler_start")
    if explicit_value is not None:
        return flag(explicit_value)
    dataset_queue_state = _handoff_dataset_queue_state(handoff)
    return flag(dataset_queue_state.get("blocks_labeler_start"))

def _handoff_dataset_queue_start_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    state = _handoff_dataset_queue_state(handoff)
    blocked = _handoff_dataset_queue_blocks_labeler_start(handoff)
    return {
        "dataset_queue_start_ready": not blocked,
        "dataset_queue_start_status": "needs_review" if blocked else "passed",
        "dataset_queue_start_operator_action": str(state.get("operator_action") or "") if blocked else "",
    }

def _handoff_assignment_ownership_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    def flag(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    integrity = (
        handoff.get("assignment_ownership_integrity")
        if isinstance(handoff.get("assignment_ownership_integrity"), Mapping)
        else {}
    )
    has_flat_integrity = any(
        key in handoff
        for key in (
            "assignment_ownership_ok",
            "assignment_active_assignment_count",
            "assignment_duplicate_active_owner_count",
            "assignment_ownership_readiness",
        )
    )
    has_integrity = bool(integrity) or has_flat_integrity
    ok = flag(integrity.get("ok")) if integrity else flag(handoff.get("assignment_ownership_ok"))
    duplicate_count = (
        int(integrity.get("duplicate_active_owner_count") or 0)
        if integrity
        else int(handoff.get("assignment_duplicate_active_owner_count") or 0)
    )
    active_count = (
        int(integrity.get("active_assignment_count") or 0)
        if integrity
        else int(handoff.get("assignment_active_assignment_count") or 0)
    )
    policy = (
        handoff.get("single_owner_policy")
        if isinstance(handoff.get("single_owner_policy"), Mapping)
        else _assignment_ownership_policy()
    )
    raw_contract = handoff.get("assignment_ownership_contract")
    contract = (
        raw_contract
        if isinstance(raw_contract, Mapping)
        else _assignment_ownership_contract_policy(policy, integrity)
        if has_integrity
        else {}
    )
    if not has_integrity:
        readiness = "missing_evidence"
        action = "Regenerate the handoff so it includes assignment ownership integrity evidence before sharing."
    elif not ok or duplicate_count > 0:
        readiness = "needs_review"
        action = "Repair duplicate active recording ownership before sharing labeler links."
    else:
        readiness = "passed"
        action = ""
    contract_fields = _assignment_ownership_contract_fields(contract)
    for key in list(contract_fields):
        if key in handoff:
            contract_fields[key] = handoff[key]
    return {
        "assignment_ownership_evidence_present": has_integrity,
        "assignment_ownership_ok": ok,
        "assignment_active_assignment_count": active_count,
        "assignment_duplicate_active_owner_count": duplicate_count,
        "assignment_ownership_readiness": readiness,
        "assignment_ownership_operator_action": action,
        **contract_fields,
        **_single_owner_policy_fields(
            policy
        ),
    }

def _handoff_entry_artifact_fields(handoff: Mapping[str, object]) -> dict[str, object]:
    def flag(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def list_value(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [part.strip() for part in text.split(",") if part.strip()]
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [str(parsed).strip()] if str(parsed).strip() else []
        return []

    files = handoff.get("files") if isinstance(handoff.get("files"), Mapping) else {}
    base_url_present = bool(str(handoff.get("base_url") or "").strip())
    required_links = (
        "expected_user_labeler_landing_url",
        "expected_user_dataset_queue_url",
        "expected_user_dashboard_url",
        "expected_user_identity_probe_url",
    )
    required_artifacts = (
        "html_index",
        "message",
        "quickstart",
        "dataset_queue",
        "manifest",
    )
    if "guarded_links_ready" in handoff:
        guarded_links_ready = flag(handoff.get("guarded_links_ready"))
        missing_guarded_links = list_value(handoff.get("missing_guarded_links"))
    else:
        missing_guarded_links = [
            key
            for key in required_links
            if not str(handoff.get(key) or "").strip()
        ]
        if not base_url_present:
            missing_guarded_links = ["absolute_base_url", *missing_guarded_links]
        guarded_links_ready = base_url_present and not missing_guarded_links

    if "handoff_artifacts_ready" in handoff:
        handoff_artifacts_ready = flag(handoff.get("handoff_artifacts_ready"))
        missing_handoff_artifacts = list_value(handoff.get("missing_handoff_artifacts"))
    else:
        missing_handoff_artifacts = []
        for key in required_artifacts:
            row_key = "index_html" if key == "html_index" else key
            if not str(files.get(key) or handoff.get(row_key) or "").strip():
                missing_handoff_artifacts.append(key)
        handoff_artifacts_ready = not missing_handoff_artifacts

    if handoff.get("handoff_entry_readiness"):
        readiness = str(handoff.get("handoff_entry_readiness") or "")
    elif guarded_links_ready and handoff_artifacts_ready:
        readiness = "passed"
    elif not base_url_present:
        readiness = "missing_base_url"
    elif not guarded_links_ready and not handoff_artifacts_ready:
        readiness = "missing_guarded_links_and_handoff_artifacts"
    elif not guarded_links_ready:
        readiness = "missing_guarded_links"
    else:
        readiness = "missing_handoff_artifacts"

    if handoff.get("handoff_entry_operator_action"):
        action = str(handoff.get("handoff_entry_operator_action") or "")
    elif readiness == "passed":
        action = ""
    elif readiness == "missing_base_url":
        action = "Regenerate the handoff with --base-url set to the deployed labeling service URL."
    elif readiness == "missing_guarded_links":
        action = "Regenerate the handoff so guarded landing, dataset queue, dashboard, and identity-probe links are present."
    elif readiness == "missing_handoff_artifacts":
        action = "Regenerate the handoff so per-user index, message, quickstart, dataset queue, and manifest artifacts are present."
    else:
        action = (
            "Regenerate the handoff with a deployed base URL and the required per-user browser-entry artifacts."
        )
    expected_user_labeler_landing_url = str(handoff.get("expected_user_labeler_landing_url") or "").strip()
    expected_user_labeling_home_url = str(handoff.get("expected_user_labeling_home_url") or "").strip()
    expected_user_dataset_queue_url = str(handoff.get("expected_user_dataset_queue_url") or "").strip()
    expected_user_personal_dataset_queue_url = str(
        handoff.get("expected_user_personal_dataset_queue_url") or ""
    ).strip()
    expected_user_dashboard_url = str(handoff.get("expected_user_dashboard_url") or "").strip()
    expected_user_identity_probe_url = str(handoff.get("expected_user_identity_probe_url") or "").strip()
    preferred_labeler_entry_url = (
        expected_user_personal_dataset_queue_url
        or expected_user_dataset_queue_url
        or expected_user_labeler_landing_url
        or expected_user_dashboard_url
    )
    personalized_labeler_entry_url = (
        expected_user_personal_dataset_queue_url or preferred_labeler_entry_url
    )
    return {
        "guarded_links_ready": guarded_links_ready,
        "missing_guarded_links": missing_guarded_links,
        "handoff_artifacts_ready": handoff_artifacts_ready,
        "missing_handoff_artifacts": missing_handoff_artifacts,
        "handoff_entry_readiness": readiness,
        "handoff_entry_operator_action": action,
        "preferred_labeler_entrypoint": (
            "personal_datasets_waiting_queue"
            if expected_user_personal_dataset_queue_url
            else "datasets_waiting_queue"
        ),
        "preferred_labeler_entry_url": preferred_labeler_entry_url,
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": personalized_labeler_entry_url,
        "expected_user_labeling_home_url": expected_user_labeling_home_url,
        "labeling_home_link_role": "human_readable_queue_alias",
        "labeler_landing_link_role": "queue_first_start",
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "dashboard_link_role": "fallback_dashboard",
        "identity_probe_link_role": "identity_check",
        "task_links_role": "convenience_entry_hints",
        "preferred_labeler_entry_url_matches_dataset_queue": bool(
            expected_user_personal_dataset_queue_url or expected_user_dataset_queue_url
        )
        and preferred_labeler_entry_url in {
            expected_user_personal_dataset_queue_url,
            expected_user_dataset_queue_url,
        },
        "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
            expected_user_personal_dataset_queue_url
            and preferred_labeler_entry_url == expected_user_personal_dataset_queue_url
        ),
        "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
            expected_user_personal_dataset_queue_url
            and personalized_labeler_entry_url == expected_user_personal_dataset_queue_url
        ),
        "identity_check_url": expected_user_identity_probe_url,
    }

def _handoff_no_open_task_message(recording: Mapping[str, object]) -> str:
    message = str(recording.get("no_open_task_message") or "").strip()
    if message:
        return message
    total = int(recording.get("total_task_count") or 0)
    complete = int(recording.get("complete_task_count") or 0)
    if total > 0 and complete >= total:
        return "All tasks for this recording are complete. Ask the operator before reopening or continuing work."
    if total > 0:
        return "This recording is assigned to you, but no startable tasks are included in this handoff. Ask the operator to inspect the batch if you expected work here."
    return "This recording is assigned to you, but no browser-labeling tasks have been generated yet. Ask the operator to generate tasks or inspect the batch if you expected work here."

_IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT = (
    "implementation_status_artifact, shareability.implementation_status_artifact, "
    "and flat implementation_status_* fields"
)

_IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE = (
    f"Implementation status inspect fields: {_IMPLEMENTATION_STATUS_INSPECT_FIELDS_TEXT}."
)

_IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE = (
    "Implementation status is not launch approval; require inspect-handoff --require-shareable "
    "and labeler_links_safe_to_share=true before sharing links."
)

_IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE = (
    "If validation-checklist.json is missing the complete implementation_status_artifact "
    "contract, inspect-handoff fails closed with implementation_status_artifact_incomplete "
    "and emits regenerate_package_with_implementation_status_artifact."
)

_IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE = (
    "Safe sharing also requires implementation_status_checklist_artifact_complete=true "
    "in shareability.safe_to_share_requires."
)

_IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE = (
    "inspection-targets.json advertises implementation_status_checklist_artifact_gate_contract "
    "for wrapper interpretation of observed value, required value, match status, "
    "fail-closed reason, mismatch reason, and repair command ID."
)

_RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE = (
    "inspection-targets.json advertises "
    "shareability_labeler_route_authorization_runtime_checklist_gate_contract "
    "for wrapper interpretation of the runtime route-checklist gate field, required "
    "value, mismatch fields, fail-closed reason, and repair command ID."
)

_SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE = (
    "inspection-targets.json and inspect-handoff shareability output advertise "
    "shareability_repair_command_contracts for wrapper repair UI over browser "
    "mutation target, direct Start/Open, single-owner, runtime route-checklist, "
    "and implementation-status regeneration failures."
)

_SHAREABILITY_COMPACT_CONTRACT_SENTENCE = (
    "inspect-handoff exposes compact shareability_contract and mirrored "
    "shareability.contract for one-object wrapper gating; inspection-targets.json "
    "advertises the field names and schema."
)

_SHAREABILITY_COMPACT_GATE_SENTENCE = (
    "Wrappers using the compact contract must require "
    "shareability_contract.safe_to_share=true before sharing labeler links."
)

_SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE = (
    "The compact shareability_contract includes fields, field_count, source_fields, "
    "and source_field_count so wrappers can detect malformed or truncated contract payloads."
)

def _write_batch_plan_html_report(payload: dict[str, object], output_path: Path) -> None:
    issues = payload.get("issues", []) if isinstance(payload.get("issues"), list) else []
    warnings = payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []
    assignments = payload.get("assignments", []) if isinstance(payload.get("assignments"), list) else []
    tasks = payload.get("tasks", []) if isinstance(payload.get("tasks"), list) else []
    changed_assignment_count = sum(
        1 for row in assignments if isinstance(row, dict) and bool(row.get("would_change"))
    )
    changed_task_count = sum(
        1 for row in tasks if isinstance(row, dict) and bool(row.get("would_change"))
    )
    closed_session_count = sum(
        int(row.get("closed_session_count") or 0)
        for row in assignments
        if isinstance(row, dict)
    )
    skipped_duplicate_assignment_apply_count = int(
        payload.get(
            "skipped_duplicate_assignment_apply_count",
            payload.get("skipped_duplicate_apply_count", 0),
        )
        or 0
    )
    issue_code_chips = " ".join(
        f"<span>{_html_escape(code)}</span>"
        for code in payload.get("issue_codes", [])
    ) or "<span>none</span>"
    warning_code_chips = " ".join(
        f"<span>{_html_escape(code)}</span>"
        for code in payload.get("warning_codes", [])
    ) or "<span>none</span>"
    blocking_warning_code_chips = " ".join(
        f"<span>{_html_escape(code)}</span>"
        for code in payload.get("blocking_warning_codes", [])
    ) or "<span>none</span>"

    issue_rows = "\n".join(
        "      <tr>"
        f"<td>{_html_escape(row.get('code', ''))}</td>"
        f"<td>{_html_escape(row.get('recording_id', ''))}</td>"
        f"<td>{_html_escape(row.get('task_id', ''))}</td>"
        f"<td>{_html_escape(row.get('source_line', ''))}</td>"
        f"<td>{_html_escape(row.get('details', ''))}</td>"
        "</tr>"
        for row in issues
        if isinstance(row, dict)
    ) or '      <tr><td colspan="5">No cross-file issues.</td></tr>'
    warning_rows = "\n".join(
        "      <tr>"
        f"<td>{_html_escape(row.get('code', ''))}</td>"
        f"<td>{_html_escape(row.get('recording_id', ''))}</td>"
        f"<td>{_html_escape(row.get('task_id', ''))}</td>"
        f"<td>{_html_escape(row.get('task_ids', ''))}</td>"
        f"<td>{_html_escape(row.get('workflow_kind', ''))}</td>"
        f"<td>{_html_escape(row.get('workflow_kinds', ''))}</td>"
        f"<td>{_html_escape(row.get('source_line', ''))}</td>"
        f"<td>{_html_escape(row.get('source_lines', ''))}</td>"
        f"<td>{_html_escape(row.get('details', ''))}</td>"
        "</tr>"
        for row in warnings
        if isinstance(row, dict)
    ) or '      <tr><td colspan="9">No warnings.</td></tr>'
    assignment_rows = "\n".join(
        "      <tr>"
        f"<td>{_html_escape(row.get('recording_id', ''))}</td>"
        f"<td>{_html_escape((row.get('assignment') or {}).get('assignee_user', '') if isinstance(row.get('assignment'), dict) else '')}</td>"
        f"<td>{_html_escape((row.get('assignment') or {}).get('status', '') if isinstance(row.get('assignment'), dict) else '')}</td>"
        f"<td>{_html_escape(row.get('would_change', ''))}</td>"
        f"<td>{_html_escape(row.get('applied', ''))}</td>"
        f"<td>{_html_escape(row.get('skipped_by_duplicate_apply', ''))}</td>"
        f"<td>{_html_escape([warning.get('code') for warning in row.get('warnings', []) if isinstance(warning, dict)] if isinstance(row.get('warnings'), list) else [])}</td>"
        f"<td>{_html_escape(row.get('closed_session_count', ''))}</td>"
        f"<td>{_html_escape(row.get('source_line', ''))}</td>"
        "</tr>"
        for row in assignments
        if isinstance(row, dict)
    ) or '      <tr><td colspan="9">No assignment rows.</td></tr>'
    task_rows = "\n".join(
        "      <tr>"
        f"<td>{_html_escape(row.get('task_id', ''))}</td>"
        f"<td>{_html_escape((row.get('task') or {}).get('recording_id', '') if isinstance(row.get('task'), dict) else '')}</td>"
        f"<td>{_html_escape((row.get('task') or {}).get('workflow_kind', '') if isinstance(row.get('task'), dict) else '')}</td>"
        f"<td>{_html_escape(row.get('would_change', ''))}</td>"
        f"<td>{_html_escape(row.get('source_line', ''))}</td>"
        "</tr>"
        for row in tasks
        if isinstance(row, dict)
    ) or '      <tr><td colspan="5">No task rows.</td></tr>'
    status = "ready" if bool(payload.get("ok")) else "needs review"
    status_class = "ok" if bool(payload.get("ok")) else "warn"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette batch plan report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211d;
      --muted: #5f6d65;
      --paper: #fffdf6;
      --line: #d8cfbc;
      --accent: #125f55;
      --warn: #a84b18;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 10% 0%, rgba(18, 95, 85, .13), transparent 30rem), var(--paper);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 76rem;
      margin: 0 auto;
      padding: 3rem 1.25rem;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 6vw, 4.5rem);
      line-height: .95;
      letter-spacing: -.045em;
    }}
    .meta {{
      color: var(--muted);
      margin: 1rem 0 2rem;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: .25rem .75rem;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .status.ok {{
      background: #dcefe9;
      color: #0d5c50;
    }}
    .status.warn {{
      background: #f8ddcb;
      color: var(--warn);
    }}
    section {{
      margin: 1.5rem 0;
      background: rgba(255, 255, 255, .76);
      border: 1px solid var(--line);
      border-radius: 1.25rem;
      padding: 1rem;
      box-shadow: 0 1rem 3rem rgba(23, 33, 29, .07);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
      gap: .8rem;
      margin: 2rem 0;
    }}
    .summary div {{
      background: rgba(255, 255, 255, .82);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1rem;
    }}
    .summary b {{
      display: block;
      font-size: 1.8rem;
      line-height: 1;
    }}
    .summary span {{
      color: var(--muted);
      font-size: .82rem;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .codes {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
      gap: .8rem;
      margin: 0 0 2rem;
    }}
    .codes div {{
      background: rgba(255, 255, 255, .82);
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 1rem;
    }}
    .codes b {{
      display: block;
      margin-bottom: .5rem;
      color: var(--muted);
      font-size: .82rem;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .codes span {{
      display: inline-block;
      margin: .15rem .2rem .15rem 0;
      border-radius: 999px;
      background: #edf3ef;
      padding: .25rem .55rem;
      font-size: .85rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: .65rem .7rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: .78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    @media (max-width: 760px) {{
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette batch plan report</h1>
    <p class="meta">
      <span class="status {status_class}">{_html_escape(status)}</span><br>
      Dry run: {_html_escape(payload.get('dry_run', ''))}<br>
      Assignments: {_html_escape(payload.get('assignment_count', 0))}
      - Tasks: {_html_escape(payload.get('task_count', 0))}
      - Issues: {_html_escape(payload.get('issue_count', 0))}
      - Warnings: {_html_escape(payload.get('warning_count', 0))}
    </p>
    <div class="summary">
      <div><b>{_html_escape(payload.get('assignment_count', 0))}</b><span>Assignments</span></div>
      <div><b>{_html_escape(changed_assignment_count)}</b><span>Assignment changes</span></div>
      <div><b>{_html_escape(payload.get('task_count', 0))}</b><span>Tasks</span></div>
      <div><b>{_html_escape(changed_task_count)}</b><span>Task changes</span></div>
      <div><b>{_html_escape(payload.get('issue_count', 0))}</b><span>Issues</span></div>
      <div><b>{_html_escape(payload.get('warning_count', 0))}</b><span>Warnings</span></div>
      <div><b>{_html_escape(closed_session_count)}</b><span>Closed sessions</span></div>
      <div><b>{_html_escape(skipped_duplicate_assignment_apply_count)}</b><span>Skipped duplicate assignment rows</span></div>
    </div>
    <div class="codes">
      <div><b>Issue codes</b>{issue_code_chips}</div>
      <div><b>Warning codes</b>{warning_code_chips}</div>
      <div><b>Blocking warning codes</b>{blocking_warning_code_chips}</div>
    </div>
    <section>
      <h2>Cross-file issues</h2>
      <table>
        <thead><tr><th>Code</th><th>Recording</th><th>Task</th><th>Source line</th><th>Details</th></tr></thead>
        <tbody>
{issue_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Warnings</h2>
      <table>
        <thead><tr><th>Code</th><th>Recording</th><th>Task</th><th>Tasks</th><th>Workflow</th><th>Workflows</th><th>Source line</th><th>Source lines</th><th>Details</th></tr></thead>
        <tbody>
{warning_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Assignments</h2>
      <table>
        <thead><tr><th>Recording</th><th>User</th><th>Status</th><th>Would change</th><th>Applied</th><th>Skipped duplicate</th><th>Row warnings</th><th>Closed sessions</th><th>Source line</th></tr></thead>
        <tbody>
{assignment_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Tasks</h2>
      <table>
        <thead><tr><th>Task</th><th>Recording</th><th>Workflow</th><th>Would change</th><th>Source line</th></tr></thead>
        <tbody>
{task_rows}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

def _write_user_handoffs_html_index(
    index: dict[str, object],
    output_path: Path,
    *,
    handoff_ready_to_send: Callable[[Mapping[str, object]], bool] | None = None,
) -> None:
    rows: list[str] = []
    index_dir = output_path.parent
    files = index.get("files") if isinstance(index.get("files"), Mapping) else {}
    progress_summary = index.get("progress_summary") if isinstance(index.get("progress_summary"), Mapping) else {}
    validation_log_href = _html_escape(_handoff_relative_href(index_dir, files.get("validation_log", "")))
    validation_checklist_href = _html_escape(_handoff_relative_href(index_dir, files.get("validation_checklist", "")))
    safe_share_next_action_gate_ids = (
        index.get("safe_share_launch_blocking_evidence_gate_ids")
        if isinstance(index.get("safe_share_launch_blocking_evidence_gate_ids"), list)
        else (
            index.get("safe_share_gate", {}).get("launch_blocking_evidence_gate_ids", [])
            if isinstance(index.get("safe_share_gate"), Mapping)
            else []
        )
    )
    safe_share_next_actions = (
        index.get("safe_share_launch_blocking_next_actions")
        if isinstance(index.get("safe_share_launch_blocking_next_actions"), list)
        else []
    )
    if not safe_share_next_action_gate_ids:
        safe_share_next_action_gate_ids = [
            action.get("gate_id")
            for action in safe_share_next_actions
            if isinstance(action, Mapping)
        ]
    safe_share_next_action_statuses = {
        str(action.get("gate_id") or ""): str(action.get("status") or "")
        for action in safe_share_next_actions
        if isinstance(action, Mapping) and str(action.get("gate_id") or "").strip()
    }
    safe_share_external_gap_statuses: dict[str, str] = {
        str(gate_id): str(status)
        for gate_id, status in (
            index.get("safe_share_external_launch_evidence_gap_statuses", {}).items()
            if isinstance(index.get("safe_share_external_launch_evidence_gap_statuses"), Mapping)
            else []
        )
        if str(gate_id).strip()
    }
    for handoff in index.get("handoffs", []):
        if not isinstance(handoff, Mapping):
            continue
        handoff_external_gap_statuses = handoff.get(
            "safe_share_external_launch_evidence_gap_statuses"
        )
        if not isinstance(handoff_external_gap_statuses, Mapping):
            continue
        safe_share_external_gap_statuses.update(
            {
                str(gate_id): str(status)
                for gate_id, status in handoff_external_gap_statuses.items()
                if str(gate_id).strip()
            }
        )
    safe_share_next_action_count = len(
        [gate_id for gate_id in safe_share_next_action_gate_ids if str(gate_id).strip()]
    )
    safe_share_gate_statuses = (
        index.get("safe_share_launch_blocking_gate_statuses")
        if isinstance(index.get("safe_share_launch_blocking_gate_statuses"), Mapping)
        else _safe_share_checklist_gate_status_fields_from_operator_validation(index).get(
            "safe_share_launch_blocking_gate_statuses",
            {},
        )
    )
    safe_share_missing_evidence_gate_ids = {
        str(gate_id)
        for gate_id in (
            index.get("safe_share_launch_blocking_missing_evidence_gate_ids")
            if isinstance(index.get("safe_share_launch_blocking_missing_evidence_gate_ids"), list)
            else index.get("operator_validation_required_missing_evidence_gate_ids")
            if isinstance(index.get("operator_validation_required_missing_evidence_gate_ids"), list)
            else []
        )
        if str(gate_id).strip()
    }
    safe_share_status_pairs: list[tuple[str, str]] = []
    for gate_id in safe_share_next_action_gate_ids:
        gate_id_text = str(gate_id).strip()
        if not gate_id_text:
            continue
        status_text = (
            str(safe_share_gate_statuses.get(gate_id_text) or "").strip()
            if isinstance(safe_share_gate_statuses, Mapping)
            else ""
        )
        if not status_text:
            status_text = safe_share_next_action_statuses.get(gate_id_text, "")
        if safe_share_external_gap_statuses.get(gate_id_text):
            status_text = safe_share_external_gap_statuses[gate_id_text]
        if (
            gate_id_text in safe_share_missing_evidence_gate_ids
            or status_text in {"", "unknown", "missing_gate"}
        ):
            status_text = "missing_evidence"
        safe_share_status_pairs.append((gate_id_text, status_text))
    safe_share_gate_status_text = " ".join(
        f"{gate_id}={status}"
        for gate_id, status in sorted(safe_share_status_pairs)
    )
    for handoff in index.get("handoffs", []):
        if not isinstance(handoff, dict):
            continue
        counts = handoff.get("counts") if isinstance(handoff.get("counts"), dict) else {}
        user_index_href = _html_escape(_handoff_relative_href(index_dir, handoff.get("index_html", "")))
        message_href = _html_escape(_handoff_relative_href(index_dir, handoff.get("message", "")))
        files = handoff.get("files") if isinstance(handoff.get("files"), dict) else {}
        quickstart_href = _html_escape(_handoff_relative_href(index_dir, files.get("quickstart", "")))
        manifest_href = _html_escape(_handoff_relative_href(index_dir, handoff.get("manifest", "")))
        dataset_queue_href = _html_escape(_handoff_relative_href(index_dir, files.get("dataset_queue", "")))
        ready = bool(handoff.get("ready_to_send")) if "ready_to_send" in handoff else (bool(handoff_ready_to_send(handoff)) if handoff_ready_to_send is not None else False)
        status = "ready" if ready else ("not ready" if bool(handoff.get("ok")) else "needs review")
        status_class = "ok" if ready else "warn"
        sendability_reasons = handoff.get("sendability_reasons") if isinstance(handoff.get("sendability_reasons"), list) else []
        sendability_reasons_text = ", ".join(str(reason) for reason in sendability_reasons if str(reason).strip())
        sendability_actions = handoff.get("sendability_actions") if isinstance(handoff.get("sendability_actions"), list) else []
        sendability_actions_text = " ".join(str(action) for action in sendability_actions if str(action).strip())
        no_open_actions = (
            handoff.get("recordings_without_open_tasks_actions")
            if isinstance(handoff.get("recordings_without_open_tasks_actions"), list)
            else counts.get("recordings_without_open_tasks_actions")
            if isinstance(counts.get("recordings_without_open_tasks_actions"), list)
            else []
        )
        no_open_actions_text = " ".join(str(action) for action in no_open_actions if str(action).strip())
        dataset_queue_state = handoff.get("dataset_queue_state") if isinstance(handoff.get("dataset_queue_state"), Mapping) else {}
        dataset_queue_state_code = str(handoff.get("dataset_queue_state_code") or dataset_queue_state.get("code") or "")
        dataset_queue_blocks_start = handoff.get("dataset_queue_blocks_labeler_start")
        if dataset_queue_blocks_start is None:
            dataset_queue_blocks_start = dataset_queue_state.get("blocks_labeler_start")
        queue_start_status = str(handoff.get("dataset_queue_start_status") or "")
        queue_start_action = str(handoff.get("dataset_queue_start_operator_action") or "")
        if not queue_start_status:
            queue_start_fields = _handoff_dataset_queue_start_fields(handoff)
            queue_start_status = str(queue_start_fields.get("dataset_queue_start_status") or "")
            queue_start_action = str(queue_start_fields.get("dataset_queue_start_operator_action") or "")
        known_user_fields = _handoff_known_user_status_fields(handoff)
        ownership_fields = _handoff_assignment_ownership_fields(handoff)
        entry_fields = _handoff_entry_artifact_fields(handoff)
        rows.append(
            "      <tr>"
            f"<td>{_html_escape(handoff.get('user', ''))}</td>"
            f"<td>{_html_escape(known_user_fields.get('known_labeler'))}</td>"
            f"<td>{_html_escape(known_user_fields.get('known_user_active_assignment_count'))}</td>"
            f"<td>{_html_escape(ownership_fields.get('assignment_ownership_ok'))}</td>"
            f"<td>{_html_escape(ownership_fields.get('assignment_duplicate_active_owner_count'))}</td>"
            f"<td>{_html_escape(entry_fields.get('guarded_links_ready'))}</td>"
            f"<td>{_html_escape(entry_fields.get('handoff_artifacts_ready'))}</td>"
            f"<td>{_html_escape(entry_fields.get('handoff_entry_readiness'))}</td>"
            f"<td><span class=\"status {status_class}\">{_html_escape(status)}</span></td>"
            f"<td>{_html_escape(sendability_reasons_text)}</td>"
            f"<td>{_html_escape(sendability_actions_text)}</td>"
            f"<td>{_html_escape(handoff.get('expected_user_labeler_landing_url') or handoff.get('labeler_landing_url') or '')}</td>"
            f"<td>{_html_escape(handoff.get('expected_user_dashboard_url') or handoff.get('dashboard_url') or '')}</td>"
            f"<td>{_html_escape(handoff.get('personalized_labeler_entry_url') or handoff.get('expected_user_personal_dataset_queue_url') or '')}</td>"
            f"<td>{_html_escape(handoff.get('expected_user_dataset_queue_url') or '')}</td>"
            f"<td>{_html_escape(handoff.get('expected_user_labeling_home_url') or '')}</td>"
            f"<td>{_html_escape(handoff.get('expected_user_identity_probe_url') or '')}</td>"
            f"<td>{_html_escape(counts.get('recordings', 0))}</td>"
            f"<td>{_html_escape(counts.get('tasks', 0))}</td>"
            f"<td>{_html_escape(counts.get('waiting_datasets', 0))}</td>"
            f"<td>{_html_escape(dataset_queue_state_code)}</td>"
            f"<td>{_html_escape(dataset_queue_blocks_start if dataset_queue_blocks_start is not None else '')}</td>"
            f"<td>{_html_escape(queue_start_status)}</td>"
            f"<td>{_html_escape(queue_start_action)}</td>"
            f"<td>{_html_escape(handoff.get('dataset_queue_preview_url') or '')}</td>"
            f"<td>{_html_escape(counts.get('signed_links', 0))}</td>"
            f"<td>{_html_escape(counts.get('recordings_without_open_tasks', 0))}</td>"
            f"<td>{_html_escape(json.dumps(counts.get('recordings_without_open_tasks_by_reason', {}), sort_keys=True))}</td>"
            f"<td>{_html_escape(no_open_actions_text)}</td>"
            f"<td>{_html_escape(counts.get('redacted_summary_fields', 0))}</td>"
            f"<td>{_html_escape(counts.get('store_check_issues', 0))}</td>"
            f"<td>{_html_escape(counts.get('store_check_warnings', 0))}</td>"
            f"<td>{_html_escape(handoff.get('links_expire_at_utc') or '')}</td>"
            f"<td><a href=\"{user_index_href}\">index.html</a></td>"
            f"<td><a href=\"{message_href}\">message.txt</a></td>"
            f"<td><a href=\"{quickstart_href}\">quickstart</a></td>"
            f"<td><a href=\"{dataset_queue_href}\">dataset-queue.json</a></td>"
            f"<td><a href=\"{manifest_href}\">manifest.json</a></td>"
            "</tr>"
        )
    generated_rows = "\n".join(rows) or "      <tr><td colspan=\"38\">No assigned users were exported.</td></tr>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette Labeling Handoffs</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211c;
      --muted: #5d6d64;
      --paper: #fbfaf5;
      --line: #d8d1bf;
      --accent: #0f6b5f;
      --warn: #a74818;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #e8f1dc 0, transparent 34rem), var(--paper);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 72rem;
      margin: 0 auto;
      padding: 3rem 1.5rem;
    }}
    h1 {{
      margin: 0 0 .5rem;
      font-size: clamp(2rem, 5vw, 4rem);
      line-height: .95;
      letter-spacing: -.04em;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 2rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, .72);
      border: 1px solid var(--line);
      box-shadow: 0 1rem 3rem rgba(23, 33, 28, .08);
    }}
    th, td {{
      padding: .8rem .9rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: .78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    a {{
      color: var(--accent);
      font-weight: 700;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: .2rem .65rem;
      font-size: .8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .status.ok {{
      background: #d9efe7;
      color: #0d5d4f;
    }}
    .status.warn {{
      background: #f8ddcb;
      color: var(--warn);
    }}
    @media (max-width: 760px) {{
      main {{
        padding: 2rem 1rem;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette labeling handoffs</h1>
    <p class="meta">
      Store: {_html_escape(index.get('store_path', ''))}<br>
      Landing URL: {_html_escape(index.get('labeler_landing_url') or '(missing --base-url)')}<br>
      Labeling home URL: {_html_escape(index.get('labeling_home_url') or '(missing --base-url)')}<br>
      Dashboard URL: {_html_escape(index.get('dashboard_url') or '(missing --base-url)')}<br>
      Personalized dataset queue URL: {_html_escape(index.get('personal_dataset_queue_url') or '(missing --base-url)')}<br>
      Personalized work URL: {_html_escape(index.get('personal_work_url') or '(missing --base-url)')}<br>
      Dataset queue URL: {_html_escape(index.get('dataset_queue_url') or '(missing --base-url)')}<br>
      <p><b>Safe-share gate:</b> Do not share labeler links solely because per-user handoffs say ready_to_send. Run <code>inspect-handoff --path PACKAGE --require-shareable</code> and require <code>labeler_links_safe_to_share=true</code>. Unapproved mutable-Zarr backup, browser response-security, identity-source, browser-smoke, disposable-Zarr mutation, or operator-recovery evidence gates are launch blockers.</p>
      <p>Safe-share next actions: {_html_escape(safe_share_next_action_count)}</p>
      <p>Safe-share gate statuses: {_html_escape(safe_share_gate_status_text)}</p>
      <p>{_html_escape(_safe_share_next_action_summary_text(index))}</p>
      Users: {_html_escape(index.get('counts', {}).get('users', 0) if isinstance(index.get('counts'), dict) else 0)}
      - Ready: {_html_escape(index.get('counts', {}).get('ready_to_send', 0) if isinstance(index.get('counts'), dict) else 0)}
      - Not ready: {_html_escape(index.get('counts', {}).get('not_ready_to_send', 0) if isinstance(index.get('counts'), dict) else 0)}
      - Store checks ok: {_html_escape(index.get('store_checks_ok', ''))}
      - TTL: {_html_escape(index.get('ttl_seconds', ''))} seconds
      - JSON index: <a href="index.json">index.json</a><br>
      Waiting recordings: {_html_escape(progress_summary.get('waiting_recording_count', 0))}
      - Complete recordings: {_html_escape(progress_summary.get('complete_recording_count', 0))}
      - Blocked/no-open recordings: {_html_escape(progress_summary.get('blocked_recording_count', 0))}
      - Waiting datasets: {_html_escape(index.get('dataset_queue_summary', {}).get('waiting_dataset_count', 0) if isinstance(index.get('dataset_queue_summary'), dict) else 0)}
      - Validation log: <a href="{validation_log_href}">validation-log-template.md</a>
      - Validation checklist: <a href="{validation_checklist_href}">validation-checklist.json</a>
    </p>
    <table>
      <thead>
        <tr>
          <th>User</th>
          <th>Known user</th>
          <th>Active assignments</th>
          <th>Ownership ok</th>
          <th>Duplicate owners</th>
          <th>Guarded links</th>
          <th>Artifacts</th>
          <th>Entry readiness</th>
          <th>Status</th>
          <th>Reasons</th>
          <th>Next action</th>
          <th>Landing</th>
          <th>Dashboard</th>
          <th>Personal queue</th>
          <th>Queue page</th>
          <th>Labeling home</th>
          <th>Identity probe</th>
          <th>Recordings</th>
          <th>Tasks</th>
          <th>Waiting datasets</th>
          <th>Queue state</th>
          <th>Blocks start</th>
          <th>Start status</th>
          <th>Start action</th>
          <th>Dataset queue link</th>
          <th>Links</th>
          <th>No startable tasks</th>
          <th>No open reasons</th>
          <th>No open actions</th>
          <th>Redacted fields</th>
          <th>Issues</th>
          <th>Warnings</th>
          <th>Links expire</th>
          <th>User page</th>
          <th>Message</th>
          <th>Quickstart</th>
          <th>Dataset queue</th>
          <th>Manifest</th>
        </tr>
      </thead>
      <tbody>
{generated_rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

def _write_launch_bundle_html_index(manifest: dict[str, object], output_path: Path) -> None:
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    audit_rows = ""
    if manifest.get("include_audit_events"):
        audit_rows = f"""
        <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('audit_task_events', '')))}">Task audit events</a></li>
        <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('audit_assignment_events', '')))}">Assignment audit events</a></li>
        <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('audit_task_definition_events', '')))}">Task definition audit events</a></li>"""
    status = "ready" if bool(manifest.get("ok")) else "needs review"
    status_class = "ok" if bool(manifest.get("ok")) else "warn"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette labeling launch bundle</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211d;
      --muted: #5f6d65;
      --paper: #fffdf6;
      --line: #d8cfbc;
      --accent: #125f55;
      --warn: #a84b18;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 10%, rgba(18, 95, 85, .14), transparent 28rem),
        linear-gradient(145deg, rgba(214, 138, 72, .12), transparent 34rem),
        var(--paper);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 70rem;
      margin: 0 auto;
      padding: 3rem 1.25rem;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2.2rem, 6vw, 4.8rem);
      line-height: .92;
      letter-spacing: -.05em;
    }}
    .meta {{
      margin: 1rem 0 2rem;
      color: var(--muted);
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: .25rem .75rem;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .status.ok {{
      background: #dcefe9;
      color: #0d5c50;
    }}
    .status.warn {{
      background: #f8ddcb;
      color: var(--warn);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
      gap: 1rem;
      margin: 2rem 0;
    }}
    section {{
      background: rgba(255, 255, 255, .76);
      border: 1px solid var(--line);
      border-radius: 1.25rem;
      padding: 1rem 1.1rem;
      box-shadow: 0 1rem 3rem rgba(23, 33, 29, .07);
    }}
    h2 {{
      margin: 0 0 .6rem;
      font-size: 1rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    ul {{
      margin: 0;
      padding-left: 1.1rem;
    }}
    li {{
      margin: .4rem 0;
    }}
    a {{
      color: var(--accent);
      font-weight: 800;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette labeling launch bundle</h1>
    <p class="meta">
      <span class="status {status_class}">{_html_escape(status)}</span><br>
      Generated: {_html_escape(manifest.get('generated_at_utc', ''))}<br>
      Store: {_html_escape(manifest.get('store_path', ''))}<br>
      Landing URL: {_html_escape(manifest.get('labeler_landing_url') or '(missing --base-url)')}<br>
      Dashboard URL: {_html_escape(manifest.get('dashboard_url') or '(missing --base-url)')}<br>
      Labeling home URL: {_html_escape(manifest.get('labeling_home_url') or '(missing --base-url)')}<br>
      Personalized dataset queue URL: {_html_escape(manifest.get('personal_dataset_queue_url') or '(missing --base-url)')}<br>
      Personalized work URL: {_html_escape(manifest.get('personal_work_url') or '(missing --base-url)')}<br>
      Dataset queue URL: {_html_escape(manifest.get('dataset_queue_url') or '(missing --base-url)')}<br>
      Users: {_html_escape(counts.get('users', 0))}
      - Assignments: {_html_escape(counts.get('assignments', 0))}
      - Tasks: {_html_escape(counts.get('tasks', 0))}
      - Assignment ownership ok: {_html_escape((manifest.get('assignment_ownership_integrity') or {}).get('ok') if isinstance(manifest.get('assignment_ownership_integrity'), Mapping) else '')}
      - Duplicate active owners: {_html_escape(counts.get('assignment_ownership_duplicate_active_owners', 0))}
      - Zarr backup targets: {_html_escape(counts.get('zarr_backup_targets', 0))}
      - Zarr backup targets by role: {_html_escape(json.dumps(counts.get('zarr_backup_targets_by_role', {}), sort_keys=True))}
      - Zarr backup required targets by role: {_html_escape(json.dumps(counts.get('zarr_backup_required_targets_by_role', {}), sort_keys=True))}
      - Waiting datasets: {_html_escape(counts.get('handoff_waiting_datasets', 0))}
      - No-open-task recordings: {_html_escape(counts.get('handoff_recordings_without_open_tasks', 0))}
      - No-open reasons: {_html_escape(json.dumps(counts.get('handoff_recordings_without_open_tasks_by_reason', {}), sort_keys=True))}
      - No-open actions: {_html_escape(' '.join(str(action) for action in (counts.get('handoff_recordings_without_open_tasks_actions') if isinstance(counts.get('handoff_recordings_without_open_tasks_actions'), list) else [])))}
      - Redacted fields: {_html_escape(counts.get('handoff_redacted_summary_fields', 0))}
      - Handoff store checks ok: {_html_escape(counts.get('handoff_store_checks_ok', manifest.get('handoff_store_checks_ok')))}
      - Handoff not-ready reasons: {_html_escape(json.dumps(counts.get('handoff_sendability_reasons', {}), sort_keys=True))}
      - Readiness issues: {_html_escape(counts.get('readiness_issues', 0))}
      - Readiness warnings: {_html_escape(counts.get('readiness_warnings', 0))}
      - Readiness no-open reasons: {_html_escape(json.dumps(counts.get('readiness_active_recordings_without_open_tasks_by_reason', {}), sort_keys=True))}
    </p>
    <div class="grid">
      <section>
        <h2>Review first</h2>
        <ul>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('readiness', '')))}">Batch readiness report</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('zarr_backup_plan', '')))}">Mutable Zarr backup plan</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('validation_log', '')))}">Validation log template</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('validation_checklist', '')))}">Validation checklist JSON</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('operator_evidence_commands', '')))}">Operator evidence command sheet</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('launch_evidence_execution_checklist', '')))}">Launch evidence execution checklist</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('readme', '')))}">Launch README</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('implementation_status', '')))}">Implementation status summary</a></li>
          <li>{_html_escape(_IMPLEMENTATION_STATUS_INSPECT_FIELDS_SENTENCE)}</li>
          <li>{_html_escape(_IMPLEMENTATION_STATUS_NOT_LAUNCH_APPROVAL_SENTENCE)}</li>
          <li>{_html_escape(_IMPLEMENTATION_STATUS_STALE_PACKAGE_FAIL_CLOSED_SENTENCE)}</li>
          <li>If launch-evidence-execution-checklist.txt is missing or stale, inspect-handoff emits regenerate_package_with_launch_evidence_execution_checklist; regenerate the launch bundle before sharing.</li>
          <li>{_html_escape(_IMPLEMENTATION_STATUS_SAFE_SHARE_REQUIREMENT_SENTENCE)}</li>
          <li>{_html_escape(_IMPLEMENTATION_STATUS_GATE_CONTRACT_SENTENCE)}</li>
          <li>{_html_escape(_RUNTIME_ROUTE_CHECKLIST_GATE_CONTRACT_SENTENCE)}</li>
          <li>{_html_escape(_SHAREABILITY_REPAIR_COMMAND_CONTRACTS_SENTENCE)}</li>
          <li>{_html_escape(_SHAREABILITY_COMPACT_CONTRACT_SENTENCE)}</li>
          <li>{_html_escape(_SHAREABILITY_COMPACT_GATE_SENTENCE)}</li>
          <li>{_html_escape(_SHAREABILITY_COMPACT_SELF_CHECK_SENTENCE)}</li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('inspect_command', '')))}">Inspection command</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('inspection_targets', '')))}">Inspection targets</a></li>
          <li><a href="manifest.json">Machine-readable manifest</a></li>
        </ul>
      </section>
      <section>
        <h2>Handoffs</h2>
        <ul>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('handoffs_html_index', '')))}">All labeler handoffs</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('handoffs_index', '')))}">Handoff JSON index</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('handoffs_roster', '')))}">Labeler roster CSV</a></li>
        </ul>
      </section>
      <section>
        <h2>Plan snapshot</h2>
        <ul>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('assignments', '')))}">Assignments snapshot</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('tasks', '')))}">Tasks snapshot</a></li>
          <li><a href="{_html_escape(_handoff_relative_href(output_path.parent, files.get('zarr_backup_plan', '')))}">Zarr backup plan JSON</a></li>
        </ul>
      </section>
      <section>
        <h2>Audit</h2>
        <ul>{audit_rows}
        </ul>
      </section>
    </div>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

def _write_user_handoff_html_index(
    *,
    user: str,
    work: dict[str, object],
    links: list[dict[str, object]],
    manifest: dict[str, object],
    output_path: Path,
) -> None:
    links_by_task_id = {str(link.get("task_id") or ""): link for link in links}
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
    has_absolute_base_url = bool(base_url)
    ready_to_send = bool(manifest.get("ready_to_send"))
    progress = (
        manifest.get("progress_summary")
        if isinstance(manifest.get("progress_summary"), Mapping)
        else work.get("progress_summary")
        if isinstance(work.get("progress_summary"), Mapping)
        else {}
    )
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
    warning_reason_text = ", ".join(warning_reasons) if warning_reasons else "operator_review_required"
    sendability_actions = [
        str(action).strip()
        for action in (manifest.get("sendability_actions") if isinstance(manifest.get("sendability_actions"), list) else [])
        if str(action).strip()
    ]
    sendability_action_text = " ".join(sendability_actions)
    dashboard_entry = (
        f'<a href="{_html_escape(dashboard_url)}">{"Open" if ready_to_send else "Preview"} your personalized dashboard</a>'
        if dashboard_url
        else "This handoff was generated without a service URL. Use this page to preview assigned work, then ask the operator for the dashboard URL before opening tasks."
    )
    dataset_queue_entry = (
        f'<a href="{_html_escape(personalized_entry_url or dataset_queue_url)}">{"Open" if ready_to_send else "Preview"} your personalized dataset queue</a>'
        if personalized_entry_url or dataset_queue_url
        else ""
    )
    canonical_dataset_queue_entry = (
        f'<a href="{_html_escape(dataset_queue_url)}">Canonical dataset queue fallback</a>'
        if dataset_queue_url
        else ""
    )
    labeling_home_entry = (
        f'<a href="{_html_escape(labeling_home_url)}">Human-readable labeling home alias</a>'
        if labeling_home_url
        else ""
    )
    landing_entry = (
        f'<a href="{_html_escape(landing_url)}">{"Open" if ready_to_send else "Preview"} your datasets-waiting landing page</a>'
        if landing_url
        else ""
    )
    quickstart_class = "quickstart" if ready_to_send else "quickstart warn"
    quickstart_heading = "Before you start" if ready_to_send else "Wait for operator review"
    start_guidance = (
        "No Palette or Crimson installation is needed. Work from this browser page or the personalized dashboard."
        if ready_to_send
        else "No Palette or Crimson installation is needed. Use this page only to preview assigned work until the operator confirms this handoff is ready."
    )
    readiness_notice = (
        ""
        if ready_to_send
        else (
            f"<p><b>This handoff is not ready to start.</b> Do not open or save task work until the operator clears it. Review reasons: {_html_escape(warning_reason_text)}.</p>"
            + (
                f"<p><b>Operator repair action:</b> {_html_escape(sendability_action_text)}</p>"
                if sendability_action_text
                else ""
            )
        )
    )
    dataset_queue_state_notice = (
        f"<p><b>Dataset queue state:</b> {_html_escape(dataset_queue_state_code or 'unknown')}"
        f"{' - ' + _html_escape(dataset_queue_state_title) if dataset_queue_state_title else ''}. "
        f"Labeler start {'blocked' if dataset_queue_blocks_start else 'allowed'}.</p>"
        + (
            f"<p>{_html_escape(dataset_queue_state_message)}</p>"
            if dataset_queue_state_message
            else ""
        )
        + (
            "<p><b>Do not start new labeling from this queue until the operator resolves this state.</b></p>"
            if dataset_queue_blocks_start
            else ""
        )
        + (
            f"<p><b>Operator action:</b> {_html_escape(dataset_queue_operator_action)}</p>"
            if dataset_queue_operator_action
            else ""
        )
    )
    def task_priority(task: Mapping[str, object]) -> float:
        try:
            return float(task.get("priority") or 0)
        except (TypeError, ValueError):
            return 0.0

    rows: list[str] = []
    for recording in work.get("recordings", []):
        if not isinstance(recording, dict):
            continue
        recording_id = recording.get("recording_id", "")
        assignment_notes = recording.get("assignment_notes") or ""
        tasks = [task for task in recording.get("tasks", []) if isinstance(task, dict)]
        tasks.sort(
            key=lambda task: (
                -task_priority(task),
                str(task.get("title") or task.get("task_id") or ""),
                str(task.get("task_id") or ""),
            )
        )
        if not tasks:
            rows.append(
                "      <tr>"
                f"<td>{_html_escape(recording_id)}</td>"
                "<td></td>"
                "<td></td>"
                f"<td><span class=\"muted\">{_html_escape(_handoff_no_open_task_message(recording))}</span></td>"
                "<td></td>"
                f"<td>{_html_escape(assignment_notes)}</td>"
                "<td><span class=\"muted\">No open signed task link</span></td>"
                "</tr>"
            )
            continue
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            link = links_by_task_id.get(task_id)
            url = str((link or {}).get("url") or "")
            path = str((link or {}).get("path") or "")
            title = str(task.get("title") or task_id)
            priority = task.get("priority")
            task_notes = str(task.get("notes") or "")
            task_cell = _html_escape(title)
            if task_notes:
                task_cell += f"<div class=\"task-note\"><b>Task note:</b> {_html_escape(task_notes)}</div>"
            link_ready = bool((link or {}).get("ready_to_share"))
            link_warnings = [
                str(warning.get("code") or "").strip()
                for warning in ((link or {}).get("shareability_warnings") or [])
                if isinstance(warning, Mapping) and str(warning.get("code") or "").strip()
            ]
            link_warning_text = ", ".join(link_warnings) if link_warnings else "not_ready_to_share"
            link_cell = (
                f"<a class=\"open-link\" href=\"{_html_escape(url)}\">Open task</a>"
                if url and has_absolute_base_url and link_ready
                else (
                    f"<span class=\"muted\">Not ready to open: {_html_escape(link_warning_text)}</span>"
                    if url and has_absolute_base_url
                    else (
                        f"<span class=\"muted\">Needs service URL: {_html_escape(path or url)}</span>"
                        if url
                        else "<span class=\"muted\">No active signed link</span>"
                    )
                )
            )
            rows.append(
                "      <tr>"
                f"<td>{_html_escape(recording_id)}</td>"
                f"<td>{_html_escape(task.get('workflow_kind', ''))}</td>"
                f"<td>{_html_escape(priority if priority is not None else '')}</td>"
                f"<td>{task_cell}</td>"
                f"<td>{_html_escape(task.get('state', ''))}</td>"
                f"<td>{_html_escape(assignment_notes)}</td>"
                f"<td>{link_cell}</td>"
                "</tr>"
            )
    generated_rows = "\n".join(rows) or "      <tr><td colspan=\"7\">No tasks are currently assigned.</td></tr>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette labeling work for {_html_escape(user)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f251d;
      --muted: #647063;
      --paper: #fffdf7;
      --line: #ddd4be;
      --accent: #115c55;
      --accent-bg: #dceee8;
      --warn: #a74818;
    }}
    body {{
      margin: 0;
      background:
        linear-gradient(135deg, rgba(17, 92, 85, .10), transparent 30rem),
        radial-gradient(circle at 85% 10%, rgba(210, 128, 62, .16), transparent 22rem),
        var(--paper);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 76rem;
      margin: 0 auto;
      padding: 3rem 1.25rem;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 6vw, 4.5rem);
      line-height: .95;
      letter-spacing: -.045em;
    }}
    .meta {{
      margin: .9rem 0 2rem;
      color: var(--muted);
      max-width: 46rem;
    }}
    .quickstart {{
      margin: 0 0 1.5rem;
      padding: 1rem 1.1rem;
      border: 1px solid var(--line);
      border-left: .45rem solid var(--accent);
      background: rgba(255, 255, 255, .72);
      box-shadow: 0 1rem 2.5rem rgba(31, 37, 29, .06);
    }}
    .quickstart h2 {{
      margin: 0 0 .55rem;
      font-size: 1.05rem;
    }}
    .quickstart.warn {{
      border-left-color: var(--warn);
      background: #fff3ec;
    }}
    .quickstart.warn h2 {{
      color: var(--warn);
    }}
    .quickstart p {{
      margin: .4rem 0;
    }}
    .quickstart a {{
      color: var(--accent);
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, .76);
      border: 1px solid var(--line);
      box-shadow: 0 1.5rem 4rem rgba(31, 37, 29, .08);
    }}
    th, td {{
      padding: .85rem .9rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: .78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .open-link {{
      display: inline-block;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: .45rem .8rem;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }}
    .muted {{
      color: var(--muted);
    }}
    .task-note {{
      margin-top: .35rem;
      color: var(--ink);
      font-size: .9rem;
      overflow-wrap: anywhere;
    }}
    .files {{
      margin-top: 1rem;
      color: var(--muted);
      font-size: .92rem;
    }}
    .files a {{
      color: var(--accent);
      font-weight: 700;
    }}
    @media (max-width: 760px) {{
      main {{
        padding: 2rem 1rem;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Your Palette labeling work</h1>
    <p class="meta">
      User: {_html_escape(user)}<br>
      Recordings: {_html_escape(manifest.get('counts', {}).get('recordings', 0) if isinstance(manifest.get('counts'), dict) else 0)}
      - Tasks: {_html_escape(manifest.get('counts', {}).get('tasks', 0) if isinstance(manifest.get('counts'), dict) else 0)}
      - Links: {_html_escape(manifest.get('counts', {}).get('signed_links', 0) if isinstance(manifest.get('counts'), dict) else 0)}
      - Link TTL: {_html_escape(manifest.get('ttl_seconds', ''))} seconds<br>
      Waiting recordings: {_html_escape(progress.get('waiting_recording_count', 0))}
      - Complete recordings: {_html_escape(progress.get('complete_recording_count', 0))}
      - Blocked/no-open recordings: {_html_escape(progress.get('blocked_recording_count', 0))}<br>
      Generated: {_html_escape(manifest.get('generated_at_utc', ''))}
      - Links expire: {_html_escape(manifest.get('links_expire_at_utc', ''))}
    </p>
    <section class="{quickstart_class}">
      <h2>{_html_escape(quickstart_heading)}</h2>
      {readiness_notice}
      <p>{_html_escape(start_guidance)}</p>
      {dataset_queue_state_notice}
      <p>Higher-priority tasks are listed first within each recording. Recording instructions appear in the Instructions column; task-specific notes appear under the relevant task.</p>
      {f'<p>{landing_entry}</p>' if landing_entry else ''}
      {f'<p>{labeling_home_entry}</p>' if labeling_home_entry else ''}
      {f'<p>{dataset_queue_entry}</p>' if dataset_queue_entry else ''}
      {f'<p>{canonical_dataset_queue_entry}</p>' if canonical_dataset_queue_entry else ''}
      <p>{dashboard_entry}</p>
      {f'<p>First open the identity check and confirm it reports you as <b>{_html_escape(user)}</b>: <a href="{_html_escape(identity_probe_url)}">{_html_escape(identity_probe_url)}</a></p>' if identity_probe_url else ''}
      <p>Before opening work, confirm the dashboard shows you as <b>{_html_escape(user)}</b>. If it shows another user, stop and contact the operator.</p>
      <p>Do not edit zarr files directly, forward this handoff, or share signed links. If access fails, the link expires, a task is missing, or this page only shows service-relative paths, ask the operator to regenerate your handoff with the service URL.</p>
      <p>Browser saves are applied server-side to your assigned task/training Zarr scope. CSV, HTML, JSON, and handoff files are metadata only and are not label write targets. Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work. Labelers should not run operator evidence, repair, checksum, or validation commands; those commands are operator-only launch controls.</p>
      <p><a href="labeler-quickstart.txt">Read the text quickstart</a></p>
    </section>
    <table>
      <thead>
        <tr>
          <th>Recording</th>
          <th>Workflow</th>
          <th>Priority</th>
          <th>Task</th>
          <th>State</th>
          <th>Instructions</th>
          <th>Link</th>
        </tr>
      </thead>
      <tbody>
{generated_rows}
      </tbody>
    </table>
    <p class="files">
      Bundle files:
      <a href="work-summary.json">work-summary.json</a>,
      <a href="signed-links.jsonl">signed-links.jsonl</a>,
      <a href="check-store.json">check-store.json</a>,
      <a href="manifest.json">manifest.json</a>,
      <a href="labeler-quickstart.txt">labeler-quickstart.txt</a>,
      <a href="validation-log-template.md">validation-log-template.md</a>,
      <a href="validation-checklist.json">validation-checklist.json</a>
    </p>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def configure_dashboard_roster_renderer_dependencies(dependencies: Mapping[str, object]) -> None:
    globals().update(dependencies)

def _dashboard_ready_invitation_bundle_impl(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    invitations = [
        {
            "user": str(row.get("user") or ""),
            "message": str(row.get("invitation_message") or ""),
        }
        for row in rows
        if str(row.get("copy_intent") or "") == "ready_row_draft"
        and str(row.get("invitation_message") or "").strip()
    ]
    return {
        "schema": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_SCHEMA,
        "kind": _DASHBOARD_READY_ROW_DRAFT_BUNDLE_KIND,
        "messages": invitations,
        "text": "\n\n".join(str(row["message"]) for row in invitations),
        "legacy_semantics": _DASHBOARD_READY_ROW_DRAFT_LEGACY_SEMANTICS,
        "legacy_field_names": list(_DASHBOARD_READY_ROW_DRAFT_LEGACY_FIELD_NAMES),
        "share_rule": _DASHBOARD_READY_ROW_DRAFT_SHARE_RULE,
    }

def _dashboard_roster_html_impl(payload: Mapping[str, object]) -> str:
    counts = payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {}
    status_report = payload.get("status_report") if isinstance(payload.get("status_report"), Mapping) else {}
    dataset_queue_start_readiness = (
        payload.get("dataset_queue_start_readiness")
        if isinstance(payload.get("dataset_queue_start_readiness"), Mapping)
        else {}
    )
    operator_recovery_contract = (
        payload.get("operator_recovery_contract")
        if isinstance(payload.get("operator_recovery_contract"), Mapping)
        else (
            status_report.get("operator_recovery_contract")
            if isinstance(status_report.get("operator_recovery_contract"), Mapping)
            else {}
        )
    )
    runtime_gate_cli_policy = (
        payload.get("runtime_operator_validation_gate_cli_policy")
        if isinstance(payload.get("runtime_operator_validation_gate_cli_policy"), Mapping)
        else (
            status_report.get("runtime_operator_validation_gate_cli_policy")
            if isinstance(
                status_report.get("runtime_operator_validation_gate_cli_policy"),
                Mapping,
            )
            else _runtime_operator_validation_gate_cli_policy()
        )
    )
    rows = payload.get("users") if isinstance(payload.get("users"), list) else []
    status = "ready" if bool(payload.get("ok")) else "needs review"
    status_class = "ok" if bool(payload.get("ok")) else "warn"
    ready_row_draft_text = str(
        payload.get("ready_row_draft_text") or payload.get("ready_invitations_text") or ""
    )
    if not ready_row_draft_text:
        ready_row_draft_text = str(_dashboard_ready_invitation_bundle([row for row in rows if isinstance(row, Mapping)])["text"])
    ready_invitation_block = (
        f"""    <section class="copy-box">
      <h2>Ready-row drafts only; safe-share review required</h2>
      <p>This block excludes not-ready diagnostic notes. Do not share until inspect-handoff --require-shareable reports labeler_links_safe_to_share=true.</p>
      <button type="button" onclick="copyInvitation(this)">Copy ready-row draft text</button>
      <textarea readonly>{_html_escape(ready_row_draft_text)}</textarea>
    </section>
"""
        if ready_row_draft_text
        else """    <section class="copy-box">
      <h2>Ready-row drafts only; safe-share review required</h2>
      <p class="muted">No ready-row draft text is available to copy.</p>
    </section>
"""
    )
    operator_validation_command_templates = (
        payload.get("operator_validation_command_templates")
        if isinstance(payload.get("operator_validation_command_templates"), Mapping)
        else (
            status_report.get("operator_validation_command_templates")
            if isinstance(status_report.get("operator_validation_command_templates"), Mapping)
            else _operator_validation_command_templates()
        )
    )
    operator_validation_command_rows = (
        operator_validation_command_templates.get("commands")
        if isinstance(operator_validation_command_templates.get("commands"), list)
        else []
    )
    safe_share_next_action_summary = _safe_share_next_action_summary_text(
        status_report if isinstance(status_report, Mapping) else payload
    )
    safe_share_external_launch_evidence_gap_summary = str(
        (
            status_report.get("safe_share_external_launch_evidence_gap_summary")
            if isinstance(status_report, Mapping)
            else ""
        )
        or payload.get("safe_share_external_launch_evidence_gap_summary")
        or ""
    )
    safe_share_next_action_detail_fields = (
        status_report.get("safe_share_launch_blocking_next_action_detail_fields")
        if isinstance(
            status_report.get("safe_share_launch_blocking_next_action_detail_fields"),
            list,
        )
        else payload.get("safe_share_launch_blocking_next_action_detail_fields")
        if isinstance(
            payload.get("safe_share_launch_blocking_next_action_detail_fields"),
            list,
        )
        else _safe_share_next_action_detail_fields()
    )
    safe_share_next_action_command_fields = (
        status_report.get("safe_share_launch_blocking_next_action_command_fields")
        if isinstance(
            status_report.get("safe_share_launch_blocking_next_action_command_fields"),
            list,
        )
        else payload.get("safe_share_launch_blocking_next_action_command_fields")
        if isinstance(
            payload.get("safe_share_launch_blocking_next_action_command_fields"),
            list,
        )
        else _safe_share_next_action_command_fields()
    )
    operator_validation_commands_html = "\n".join(
        (
            "      <li>"
            f"<b>{_html_escape(command.get('label') or command.get('id') or 'Operator command')}</b><br>"
            f"<code>{_html_escape(command.get('command') or '')}</code><br>"
            f"<span class=\"muted\">gates {_html_escape(json.dumps(command.get('gate_ids') if isinstance(command.get('gate_ids'), list) else []))}; "
            f"checksum refresh {_html_escape(command.get('requires_checksum_refresh_after_run'))}</span>"
            "</li>"
        )
        for command in operator_validation_command_rows
        if isinstance(command, Mapping)
    )
    operator_validation_commands_block = f"""    <section class="copy-box">
      <h2>Operator validation commands</h2>
      <p>Generated command templates for pending operator-evidence gates. These are operator-only next steps, not labeler instructions.</p>
      <p>Command count: {_html_escape(operator_validation_command_templates.get('command_count') or 0)}; missing-command gates: {_html_escape(json.dumps(operator_validation_command_templates.get('missing_command_gate_ids') if isinstance(operator_validation_command_templates.get('missing_command_gate_ids'), list) else []))}</p>
      <p>Safe-share next actions: {_html_escape(safe_share_next_action_summary)}</p>
      <p>External launch evidence gaps: {_html_escape(safe_share_external_launch_evidence_gap_summary or 'inspect machine-readable safe_share_external_launch_evidence_gap_* fields')}</p>
      <p>Safe-share blocker action detail fields: {_html_escape(json.dumps(safe_share_next_action_detail_fields))}</p>
      <p>Safe-share blocker action command fields: {_html_escape(json.dumps(safe_share_next_action_command_fields))}</p>
      <ul>
{operator_validation_commands_html or '        <li>No generated operator-validation commands are required.</li>'}
      </ul>
    </section>
"""
    row_html: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ready = bool(row.get("ready_to_invite"))
        reasons = row.get("invite_reasons") if isinstance(row.get("invite_reasons"), list) else []
        reason_text = ", ".join(str(reason) for reason in reasons if str(reason).strip())
        actions = row.get("invite_actions") if isinstance(row.get("invite_actions"), list) else []
        no_open_actions = (
            row.get("recordings_without_open_tasks_actions")
            if isinstance(row.get("recordings_without_open_tasks_actions"), list)
            else []
        )
        action_text = " ".join(str(action) for action in [*actions, *no_open_actions] if str(action).strip())
        dataset_queue_preview_url = str(row.get("dataset_queue_preview_url") or "")
        expected_user_labeler_landing_url = str(row.get("expected_user_labeler_landing_url") or "")
        expected_user_dataset_queue_url = str(row.get("expected_user_dataset_queue_url") or "")
        canonical_dataset_queue_url = str(
            row.get("canonical_dataset_queue_preview_url")
            or expected_user_dataset_queue_url
            or ""
        )
        preferred_labeler_entry_url = str(row.get("preferred_labeler_entry_url") or "")
        preferred_labeler_entrypoint = str(row.get("preferred_labeler_entrypoint") or "")
        personalized_labeler_entry_url = str(row.get("personalized_labeler_entry_url") or "")
        displayed_labeler_entry_url = personalized_labeler_entry_url or preferred_labeler_entry_url
        link_role_text = "; ".join(
            role
            for role in (
                f"entry={preferred_labeler_entrypoint}" if preferred_labeler_entrypoint else "",
                f"personal_queue={row.get('personal_dataset_queue_link_role')}" if row.get("personal_dataset_queue_link_role") else "",
                f"queue={row.get('dataset_queue_link_role')}" if row.get("dataset_queue_link_role") else "",
                f"canonical_queue={row.get('canonical_dataset_queue_link_role')}" if row.get("canonical_dataset_queue_link_role") else "",
                f"dashboard={row.get('dashboard_link_role')}" if row.get("dashboard_link_role") else "",
                f"identity={row.get('identity_probe_link_role')}" if row.get("identity_probe_link_role") else "",
                f"tasks={row.get('task_links_role')}" if row.get("task_links_role") else "",
            )
            if role
        )
        dataset_queue_state_code = str(row.get("dataset_queue_state_code") or "")
        dataset_queue_state_title = str(row.get("dataset_queue_state_title") or "")
        dataset_queue_blocks_start = bool(row.get("dataset_queue_blocks_labeler_start"))
        safety_bits = [
            "confirm dashboard user" if bool(row.get("dashboard_identity_check_required")) else "",
            "browser only" if bool(row.get("browser_only")) else "",
            "no local install" if not bool(row.get("requires_local_palette_installation")) and not bool(row.get("requires_local_crimson_installation")) and not bool(row.get("requires_local_conda_environment")) and not bool(row.get("requires_local_project_dependencies")) else "",
            "no direct zarr edits" if bool(row.get("no_direct_zarr_edits")) else "",
            (
                "compact contract gates: "
                f"browser_mutation_target_contract_met={row.get('browser_mutation_target_contract_met')}; "
                f"browser_mutation_target_mismatch_count={row.get('browser_mutation_target_mismatch_count')}; "
                f"direct_browser_start_contract_met={row.get('direct_browser_start_contract_met')}; "
                f"direct_browser_start_mismatch_count={row.get('direct_browser_start_mismatch_count')}; "
                f"single_owner_policy_contract_met={row.get('single_owner_policy_contract_met')}; "
                f"labeler_route_authorization_runtime_checklist_gate_met={row.get('labeler_route_authorization_runtime_checklist_gate_met')}; "
                f"labeler_route_authorization_runtime_checklist_mismatch_count={row.get('labeler_route_authorization_runtime_checklist_mismatch_count')}"
            ),
            "validates reassignment target before closing sessions" if bool(row.get("operator_recovery_reassignment_target_validated_before_session_closure")) else "",
            "session closure and assignment update are atomic" if bool(row.get("operator_recovery_session_closure_and_assignment_update_atomic")) else "",
            "reassignment closes previous-owner sessions before owner update" if bool(row.get("operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update")) else "",
        ]
        safety_text = "; ".join(bit for bit in safety_bits if bit)
        row_html.append(
            "      <tr>"
            f"<td>{_html_escape(row.get('user', ''))}</td>"
            f"<td><span class=\"status {'ok' if ready else 'warn'}\">{_html_escape('ready row draft; safe-share review required' if ready else 'not-ready diagnostic row')}</span></td>"
            f"<td>{_html_escape(reason_text)}</td>"
            f"<td>{_html_escape(row.get('open_tasks', 0))}</td>"
            f"<td>{_html_escape(row.get('waiting_datasets', 0))}</td>"
            f"<td>{_html_escape(dataset_queue_state_code)}<br>{_html_escape(dataset_queue_state_title)}<br>blocks start: {_html_escape(dataset_queue_blocks_start)}</td>"
            f"<td>{_html_escape(displayed_labeler_entry_url)}"
            + (
                f"<br><span class=\"muted\">Canonical fallback: {_html_escape(canonical_dataset_queue_url)}</span>"
                if personalized_labeler_entry_url and canonical_dataset_queue_url
                else ""
            )
            + f"<br>{_html_escape(link_role_text)}</td>"
            f"<td>{_html_escape(expected_user_labeler_landing_url)}</td>"
            f"<td>{_html_escape(dataset_queue_preview_url)}</td>"
            f"<td>{_html_escape(expected_user_dataset_queue_url)}</td>"
            f"<td>{_html_escape(row.get('complete_tasks', 0))} / {_html_escape(row.get('total_tasks', 0))} ({_html_escape(row.get('completion_percent', 'n/a'))}%)<br>{_html_escape(row.get('completion_state', 'unknown'))}</td>"
            f"<td>{_html_escape(row.get('recordings', 0))}</td>"
            f"<td>{_html_escape(row.get('recordings_without_open_tasks', 0))}</td>"
            f"<td>{_html_escape(action_text)}</td>"
            f"<td>{_html_escape(safety_text)}</td>"
            f"<td>{_html_escape(row.get('expected_user_identity_probe_url', ''))}</td>"
        f"<td><button type=\"button\" onclick=\"copyInvitation(this)\">{_html_escape(row.get('copy_label') or ('Copy ready-row draft' if ready else 'Copy not-ready note'))}</button><br><textarea readonly>{_html_escape(row.get('invitation_message', ''))}</textarea></td>"
            "</tr>"
        )
    generated_rows = "\n".join(row_html) or "      <tr><td colspan=\"17\">No users matched this dashboard roster.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette dashboard ready-row draft roster</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211c;
      --muted: #5d6d64;
      --paper: #fbfaf5;
      --line: #d8d1bf;
      --accent: #0f6b5f;
      --warn: #a74818;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top right, #e9f0de 0, transparent 32rem), var(--paper);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 78rem;
      margin: 0 auto;
      padding: 3rem 1.5rem;
    }}
    h1 {{
      margin: 0 0 .5rem;
      font-size: clamp(2rem, 5vw, 4rem);
      line-height: .95;
      letter-spacing: -.04em;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 2rem;
    }}
    .copy-box {{
      border: 1px solid var(--line);
      border-radius: 1rem;
      background: rgba(255, 255, 255, .76);
      margin: 0 0 1.5rem;
      padding: 1rem;
      box-shadow: 0 .75rem 2rem rgba(23, 33, 28, .06);
    }}
    .copy-box h2 {{
      margin: 0 0 .4rem;
    }}
    .copy-box p {{
      margin: .25rem 0 .8rem;
    }}
    .muted {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, .76);
      border: 1px solid var(--line);
      box-shadow: 0 1rem 3rem rgba(23, 33, 28, .08);
    }}
    th, td {{
      padding: .8rem .9rem;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: .78rem;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    textarea {{
      width: min(34rem, 72vw);
      min-height: 5rem;
      border: 1px solid var(--line);
      border-radius: .65rem;
      padding: .65rem;
      color: var(--ink);
      background: #fffdf6;
      font: inherit;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      margin: 0 0 .5rem;
      padding: .45rem .8rem;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: .2rem .65rem;
      font-size: .8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .status.ok {{
      background: #d9efe7;
      color: #0d5d4f;
    }}
    .status.warn {{
      background: #f8ddcb;
      color: var(--warn);
    }}
    @media (max-width: 760px) {{
      main {{
        padding: 2rem 1rem;
      }}
      table {{
        display: block;
        overflow-x: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette dashboard ready-row draft roster</h1>
    <p class="meta">
      <span class="status {status_class}">{_html_escape(status)}</span><br>
      Status report: {_html_escape(status_report.get('report_kind') or 'multi_user_labeling_status')}<br>
      Landing URL: {_html_escape(payload.get('labeler_landing_url') or '(missing --base-url)')}<br>
      Labeling home URL: {_html_escape(payload.get('labeling_home_url') or '(missing --base-url)')}<br>
      Dashboard URL: {_html_escape(payload.get('dashboard_url') or '(missing --base-url)')}<br>
      Dataset queue URL: {_html_escape(payload.get('dataset_queue_url') or '(missing --base-url)')}<br>
      Users: {_html_escape(counts.get('users', 0))}
      - Ready rows: {_html_escape(counts.get('ready_to_invite', 0))}
      - Not-ready rows: {_html_escape(counts.get('not_ready_to_invite', 0))}
      - Ready rows are draft text only until safe-share inspection passes
      - Ready states: {_html_escape(json.dumps(counts.get('ready_states', {}), sort_keys=True))}<br>
      Completion: {_html_escape(counts.get('complete_tasks', 0))} / {_html_escape(counts.get('total_tasks', 0))} tasks complete ({_html_escape(counts.get('completion_percent', 'n/a'))}%)<br>
      Completion states: {_html_escape(json.dumps(counts.get('completion_states', {}), sort_keys=True))}<br>
      Waiting datasets: {_html_escape(counts.get('waiting_datasets', 0))} - Dataset startable tasks: {_html_escape(counts.get('dataset_open_tasks', 0))} - Users with waiting datasets: {_html_escape(counts.get('users_with_waiting_datasets', 0))}<br>
          Dataset queue states: {_html_escape(json.dumps(counts.get('dataset_queue_states', {}), sort_keys=True))} - Blocked-start users: {_html_escape(', '.join(str(user) for user in (counts.get('dataset_queue_blocked_start_users') if isinstance(counts.get('dataset_queue_blocked_start_users'), list) else [])) or 'none')}<br>
          Queue start readiness: {_html_escape(dataset_queue_start_readiness.get('status') or 'unknown')} - Blocked users: {_html_escape(', '.join(str(user) for user in (dataset_queue_start_readiness.get('dataset_queue_blocked_start_users') if isinstance(dataset_queue_start_readiness.get('dataset_queue_blocked_start_users'), list) else [])) or 'none')}<br>
          Compact contract gates: browser_mutation_target_contract_all_users_met={_html_escape(counts.get('browser_mutation_target_contract_all_users_met', ''))}; browser_mutation_target_total_mismatch_count={_html_escape(counts.get('browser_mutation_target_total_mismatch_count', 0))}; browser_mutation_target_contract_not_met_users={_html_escape(', '.join(str(user) for user in (counts.get('browser_mutation_target_contract_not_met_users') if isinstance(counts.get('browser_mutation_target_contract_not_met_users'), list) else [])) or 'none')}; direct_browser_start_contract_all_users_met={_html_escape(counts.get('direct_browser_start_contract_all_users_met', ''))}; direct_browser_start_total_mismatch_count={_html_escape(counts.get('direct_browser_start_total_mismatch_count', 0))}; direct_browser_start_contract_not_met_users={_html_escape(', '.join(str(user) for user in (counts.get('direct_browser_start_contract_not_met_users') if isinstance(counts.get('direct_browser_start_contract_not_met_users'), list) else [])) or 'none')}; single_owner_policy_contract_all_users_met={_html_escape(counts.get('single_owner_policy_contract_all_users_met', ''))}; single_owner_policy_contract_not_met_users={_html_escape(', '.join(str(user) for user in (counts.get('single_owner_policy_contract_not_met_users') if isinstance(counts.get('single_owner_policy_contract_not_met_users'), list) else [])) or 'none')}; labeler_route_authorization_runtime_checklist_gate_all_users_met={_html_escape(counts.get('labeler_route_authorization_runtime_checklist_gate_all_users_met', ''))}; labeler_route_authorization_runtime_checklist_total_mismatch_count={_html_escape(counts.get('labeler_route_authorization_runtime_checklist_total_mismatch_count', 0))}; labeler_route_authorization_runtime_checklist_not_met_users={_html_escape(', '.join(str(user) for user in (counts.get('labeler_route_authorization_runtime_checklist_not_met_users') if isinstance(counts.get('labeler_route_authorization_runtime_checklist_not_met_users'), list) else [])) or 'none')}<br>
          Runtime validation gate CLI: preferred={_html_escape(runtime_gate_cli_policy.get('preferred_require_flag') or '')}; legacy={_html_escape(runtime_gate_cli_policy.get('legacy_require_flag') or '')}; checklist={_html_escape(runtime_gate_cli_policy.get('validation_checklist_flag') or '')}; protects Start/Open={_html_escape(runtime_gate_cli_policy.get('protects_browser_start_open', ''))}; protects mutations={_html_escape(runtime_gate_cli_policy.get('protects_browser_mutations', ''))}; blocks before zarr write={_html_escape(runtime_gate_cli_policy.get('blocks_before_zarr_write', ''))}<br>
          Operator validation start gate: blocks task open={_html_escape(runtime_gate_cli_policy.get('protects_browser_start_open', ''))}; blocks before session creation={_html_escape(runtime_gate_cli_policy.get('blocks_before_session_creation', ''))}; blocks before target token check={_html_escape(runtime_gate_cli_policy.get('blocks_before_target_token_check', ''))}<br>
          Operator validation mutation gate: blocks browser mutation={_html_escape(runtime_gate_cli_policy.get('protects_browser_mutations', ''))}; blocks before zarr write={_html_escape(runtime_gate_cli_policy.get('blocks_before_zarr_write', ''))}; blocks before audit event creation={_html_escape(runtime_gate_cli_policy.get('blocks_before_audit_event_creation', ''))}<br>
          Operator recovery: ready={_html_escape(operator_recovery_contract.get('ready', ''))}; validates target before session closure={_html_escape(operator_recovery_contract.get('reassignment_target_validated_before_session_closure', ''))}; session closure/update atomic={_html_escape(operator_recovery_contract.get('session_closure_and_assignment_update_atomic', ''))}; reassignment closes previous-owner sessions={_html_escape(operator_recovery_contract.get('reassignment_closes_previous_owner_sessions', ''))}; before owner update={_html_escape(operator_recovery_contract.get('reassignment_closes_previous_owner_sessions_before_assignment_update', ''))}<br>
      Ready-row draft users: {_html_escape(', '.join(str(user) for user in (counts.get('ready_row_draft_users') if isinstance(counts.get('ready_row_draft_users'), list) else counts.get('ready_to_invite_users') if isinstance(counts.get('ready_to_invite_users'), list) else [])) or 'none')}<br>
      Diagnostic-note users: {_html_escape(', '.join(str(user) for user in (counts.get('diagnostic_note_users') if isinstance(counts.get('diagnostic_note_users'), list) else counts.get('not_ready_to_invite_users') if isinstance(counts.get('not_ready_to_invite_users'), list) else [])) or 'none')}<br>
      Identity probes: {_html_escape(counts.get('identity_probe_available', 0))} / {_html_escape(counts.get('identity_probe_required', 0))} available; missing {_html_escape(counts.get('identity_probe_missing', 0))} ({_html_escape(', '.join(str(user) for user in (counts.get('identity_probe_missing_users') if isinstance(counts.get('identity_probe_missing_users'), list) else [])) or 'none')})<br>
      - Invite reasons: {_html_escape(json.dumps(counts.get('invite_reasons', {}), sort_keys=True))}<br>
      Copy intents: {_html_escape(json.dumps(counts.get('copy_intents', {}), sort_keys=True))}<br>
      Invite actions: {_html_escape(' '.join(str(action) for action in (payload.get('invite_actions') if isinstance(payload.get('invite_actions'), list) else [])))}
    </p>
{ready_invitation_block}
{operator_validation_commands_block}
    <table>
      <thead>
        <tr>
          <th>User</th>
          <th>Status</th>
          <th>Reasons</th>
          <th>Startable tasks</th>
          <th>Waiting datasets</th>
          <th>Queue state</th>
          <th>Preferred entry</th>
          <th>Landing</th>
          <th>Dataset queue</th>
          <th>Guarded queue page</th>
          <th>Completion</th>
          <th>Recordings</th>
          <th>No-open recordings</th>
          <th>Next action</th>
          <th>Safety</th>
          <th>Identity probe</th>
          <th>Invitation message</th>
        </tr>
      </thead>
      <tbody>
{generated_rows}
      </tbody>
    </table>
  </main>
  <script>
    async function copyInvitation(button) {{
      const container = button.closest("td") || button.closest(".copy-box");
      const textarea = container ? container.querySelector("textarea") : null;
      if (!textarea) return;
      try {{
        await navigator.clipboard.writeText(textarea.value);
        button.textContent = "Copied";
      }} catch (_error) {{
        textarea.focus();
        textarea.select();
        document.execCommand("copy");
        button.textContent = "Copied";
      }}
    }}
  </script>
</body>
</html>
"""

# Preserve original helper names inside this module so moved helpers can
# continue to call each other exactly as they did in web.py.
_dashboard_ready_invitation_bundle = _dashboard_ready_invitation_bundle_impl
_dashboard_roster_html = _dashboard_roster_html_impl
