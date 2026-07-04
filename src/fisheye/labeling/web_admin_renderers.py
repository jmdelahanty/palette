"""Read-only admin HTML renderers for labeling web surfaces."""

from __future__ import annotations

import html
import json
from typing import Mapping, Sequence
from urllib.parse import quote

from .assignment_store import LABELING_USER_ROLES, LABELING_USER_STATUSES
from .admin_registry import _task_title
from .template_assets import read_labeling_asset, render_labeling_template
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
)
from .web_policy import IDENTITY_PROBE_PATH, PERSONAL_WORK_PATH

__all__ = [
    "_admin_datasets_html",
    "_admin_html",
    "_admin_recording_html",
    "_admin_task_html",
    "_admin_user_html",
    "_admin_users_html",
]

def _admin_datasets_html() -> bytes:
    return read_labeling_asset("templates/admin/datasets.html.j2").encode("utf-8")



def _admin_html() -> bytes:
    return read_labeling_asset("templates/admin/index.html.j2").encode("utf-8")













def _admin_recording_html(payload: Mapping[str, object]) -> bytes:
    recording_id = str(payload.get("recording_id") or "")
    assignment = payload.get("assignment") if isinstance(payload.get("assignment"), Mapping) else {}
    task_counts = payload.get("task_counts") if isinstance(payload.get("task_counts"), Mapping) else {}
    tasks = [task for task in payload.get("tasks", []) if isinstance(task, Mapping)]
    active_sessions = [session for session in payload.get("active_sessions", []) if isinstance(session, Mapping)]
    repair_route = str(payload.get("reassignment_session_repair_route") or "")
    reassignment_repair_html = ""
    if bool(payload.get("reassignment_session_safety_blocks_labeler_mutation")) and repair_route:
        mismatch_ids = (
            payload.get("reassignment_session_safety_active_session_assignment_mismatch_session_ids")
            if isinstance(
                payload.get("reassignment_session_safety_active_session_assignment_mismatch_session_ids"),
                list,
            )
            else []
        )
        reassignment_repair_html = (
            '<section class="card">'
            "<h2>Reassignment session safety</h2>"
            "<p>Stale previous-owner sessions are blocking this recording from safe browser labeling.</p>"
            f"<p class=\"muted\">Mismatched sessions: {html.escape(', '.join(str(session_id) for session_id in mismatch_ids) or 'unknown')}</p>"
            f"<p class=\"muted\">Operator action: {html.escape(str(payload.get('reassignment_session_safety_operator_action') or ''))}</p>"
            f"<button type=\"button\" data-repair-route=\"{html.escape(repair_route)}\" onclick=\"repairReassignmentSessions(this)\">Repair reassignment sessions</button>"
            '<pre id="reassignment-repair-result" class="muted"></pre>'
            "</section>"
        )
    recent_events = [event for event in payload.get("recent_events", []) if isinstance(event, Mapping)]
    owner = str(payload.get("assignee_user") or "")
    owner_html = (
        f'<a href="/admin/users/{quote(owner, safe="")}">{html.escape(owner)}</a>'
        if owner
        else '<span class="muted">unassigned</span>'
    )
    focused_dataset_progress_url = f"/admin/datasets?recording_id={quote(recording_id, safe='')}"
    landing_url = str(payload.get("expected_user_labeler_landing_url") or "")
    labeling_home_url = str(payload.get("expected_user_labeling_home_url") or "")
    landing_html = (
        f'<a href="{html.escape(landing_url)}">{html.escape(landing_url)}</a>'
        if landing_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    labeling_home_html = (
        f'<a href="{html.escape(labeling_home_url)}">{html.escape(labeling_home_url)}</a>'
        if labeling_home_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    dashboard_url = str(payload.get("expected_user_dashboard_url") or "")
    dashboard_html = (
        f'<a href="{html.escape(dashboard_url)}">{html.escape(dashboard_url)}</a>'
        if dashboard_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    dataset_queue_url = str(payload.get("expected_user_dataset_queue_url") or "")
    dataset_queue_html = (
        f'<a href="{html.escape(dataset_queue_url)}">{html.escape(dataset_queue_url)}</a>'
        if dataset_queue_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    personal_work_url = str(payload.get("expected_user_personal_work_url") or "")
    personal_work_html = (
        f'<a href="{html.escape(personal_work_url)}">{html.escape(personal_work_url)}</a>'
        if personal_work_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    personal_dataset_queue_url = str(payload.get("expected_user_personal_dataset_queue_url") or "")
    personal_dataset_queue_html = (
        f'<a href="{html.escape(personal_dataset_queue_url)}">{html.escape(personal_dataset_queue_url)}</a>'
        if personal_dataset_queue_url
        else '<span class="muted">unavailable until assigned</span>'
    )
    assignment_rows = [
        ("Recording", recording_id),
        ("Assignee", assignment.get("assignee_user") if assignment else ""),
        ("Status", assignment.get("status") if assignment else ""),
        ("Notes", assignment.get("notes") if assignment else ""),
        ("Assigned by", assignment.get("assigned_by") if assignment else ""),
        ("Assigned at", assignment.get("assigned_at_utc") if assignment else ""),
        ("Updated at", assignment.get("updated_at_utc") if assignment else ""),
    ]
    assignment_table = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value or ''))}</td></tr>"
        for label, value in assignment_rows
    )
    task_rows = "\n".join(
        "<tr>"
        f"<td><a href=\"/admin/tasks/{quote(str(task.get('task_id') or ''), safe='')}\">{html.escape(str(task.get('task_id') or ''))}</a></td>"
        f"<td>{html.escape(str(task.get('workflow_kind') or ''))}</td>"
        f"<td>{html.escape(str(task.get('state') or ''))}</td>"
        f"<td>{html.escape(str(task.get('dataset_id') or ''))}</td>"
        f"<td>{html.escape(', '.join(str(field) for field in (task.get('redacted_fields') if isinstance(task.get('redacted_fields'), list) else [])) or 'none')}</td>"
        "</tr>"
        for task in tasks
    ) or '<tr><td colspan="5" class="muted">No tasks recorded for this recording.</td></tr>'
    session_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(session.get('session_id') or ''))}</td>"
        f"<td><a href=\"/admin/tasks/{quote(str(session.get('task_id') or ''), safe='')}\">{html.escape(str(session.get('task_id') or ''))}</a></td>"
        f"<td>{html.escape(str(session.get('user') or ''))}</td>"
        f"<td>{html.escape(str(session.get('workflow_kind') or ''))}</td>"
        f"<td>{html.escape(str(session.get('expires_at_utc') or ''))}</td>"
        "</tr>"
        for session in active_sessions
    ) or '<tr><td colspan="5" class="muted">No active browser sessions for this recording.</td></tr>'
    event_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(event.get('created_at_utc') or ''))}</td>"
        f"<td>{html.escape(str(event.get('event_type') or ''))}</td>"
        f"<td>{html.escape(str(event.get('user') or ''))}</td>"
        f"<td><a href=\"/admin/tasks/{quote(str(event.get('task_id') or ''), safe='')}\">{html.escape(str(event.get('task_id') or ''))}</a></td>"
        f"<td>{html.escape(', '.join(str(key) for key in (event.get('target_keys') if isinstance(event.get('target_keys'), list) else [])) or 'none')}</td>"
        "</tr>"
        for event in recent_events
    ) or '<tr><td colspan="5" class="muted">No recent audit events for this recording.</td></tr>'
    body = render_labeling_template(
        "admin/recording.html.j2",
        {
            "slot_001": html.escape(recording_id),
            "slot_002": html.escape(recording_id),
            "slot_003": owner_html,
            "slot_004": html.escape(focused_dataset_progress_url),
            "slot_005": html.escape(focused_dataset_progress_url),
            "slot_006": personal_dataset_queue_html,
            "slot_007": personal_work_html,
            "slot_008": landing_html,
            "slot_009": labeling_home_html,
            "slot_010": dashboard_html,
            "slot_011": dataset_queue_html,
            "slot_012": assignment_table,
            "slot_013": reassignment_repair_html,
            "slot_014": html.escape(str(task_counts.get('open_tasks', 0))),
            "slot_015": html.escape(str(task_counts.get('complete_tasks', 0))),
            "slot_016": html.escape(str(task_counts.get('total_tasks', 0))),
            "slot_017": html.escape(str(payload.get('active_session_count', 0))),
            "slot_018": html.escape(str(payload.get('recent_event_count', 0))),
            "slot_019": task_rows,
            "slot_020": session_rows,
            "slot_021": event_rows,
        },
    )
    return body.encode("utf-8")


def _admin_task_html(task: Mapping[str, object], *, events: Sequence[Mapping[str, object]] = ()) -> bytes:
    safe_title = html.escape(_task_title(task))
    task_id = str(task.get("task_id") or "")
    task_id_js = json.dumps(task_id)
    current_state = str(task.get("state") or "")
    state_options = "\n".join(
        f'<option value="{html.escape(state)}"{" selected" if state == current_state else ""}>{html.escape(state)}</option>'
        for state in ("pending", "in_progress", "complete")
    )
    scope = task.get("scope")
    scope_text = json.dumps(scope if isinstance(scope, (dict, list)) else {}, indent=2, sort_keys=True)
    rows = [
        ("Task id", task.get("task_id")),
        ("Recording", task.get("recording_id")),
        ("Assignee", task.get("assignee_user")),
        ("Assignment status", task.get("assignment_status")),
        ("Workflow", task.get("workflow_kind")),
        ("State", task.get("state")),
        ("Dataset", task.get("dataset_id")),
        ("Zarr use", task.get("zarr_use")),
        ("Stage", task.get("stage_group")),
        ("Run", task.get("run_name")),
        ("Component", task.get("component_name")),
        ("Priority", task.get("priority")),
        ("Created", task.get("created_at_utc")),
        ("Updated", task.get("updated_at_utc")),
        ("Completed", task.get("completed_at_utc")),
        ("Notes", task.get("notes")),
    ]
    row_html = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value or ''))}</td></tr>"
        for label, value in rows
    )
    if events:
        event_html = "\n".join(
            "<details>"
            f"<summary><b>{html.escape(str(event.get('event_type') or 'event'))}</b> "
            f"{html.escape(str(event.get('created_at_utc') or ''))} "
            f"<code>{html.escape(str(event.get('event_id') or ''))}</code></summary>"
            f"<pre>{html.escape(json.dumps({key: event.get(key) for key in ('target', 'before', 'after')}, indent=2, sort_keys=True))}</pre>"
            "</details>"
            for event in events
        )
    else:
        event_html = '<p class="muted">No audit events recorded for this task.</p>'
    body = render_labeling_template(
        "admin/task.html.j2",
        {
            "slot_001": safe_title,
            "slot_002": safe_title,
            "slot_003": row_html,
            "slot_004": state_options,
            "slot_005": html.escape(scope_text),
            "slot_006": event_html,
            "slot_007": task_id_js,
        },
    )
    return body.encode("utf-8")






def _admin_users_html(payload: Mapping[str, object]) -> bytes:
    users = payload.get("users") if isinstance(payload.get("users"), list) else []
    role_options = "".join(
        f'<option value="{html.escape(role)}">{html.escape(role)}</option>'
        for role in LABELING_USER_ROLES
    )
    status_options = "".join(
        f'<option value="{html.escape(status)}">{html.escape(status)}</option>'
        for status in LABELING_USER_STATUSES
    )
    rows_html = ""
    for row in users:
        if not isinstance(row, Mapping):
            continue
        user_id = str(row.get("user_id") or "")
        assignment_counts = row.get("assignment_counts") if isinstance(row.get("assignment_counts"), Mapping) else {}
        task_counts = row.get("task_counts") if isinstance(row.get("task_counts"), Mapping) else {}
        action = (
            f'<button type="button" data-user="{html.escape(user_id)}" data-action="deactivate" onclick="setUserStatus(this)">Deactivate</button>'
            if str(row.get("status") or "") == "active"
            else f'<button type="button" data-user="{html.escape(user_id)}" data-action="activate" onclick="setUserStatus(this)">Activate</button>'
            if str(row.get("status") or "") == "inactive"
            else ""
        )
        rows_html += (
            "<tr>"
            f'<td><a href="{html.escape(str(row.get("admin_user_url") or ""))}">{html.escape(user_id)}</a></td>'
            f"<td>{html.escape(str(row.get('display_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('email') or ''))}</td>"
            f"<td>{html.escape(str(row.get('role') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(assignment_counts.get('active') or 0))} active / {html.escape(str(assignment_counts.get('total') or 0))} total</td>"
            f"<td>{html.escape(str(task_counts.get('open') or 0))} open / {html.escape(str(task_counts.get('complete') or 0))} complete</td>"
            f"<td>{action}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html = '<tr><td colspan="8" class="muted">No labeling users are registered.</td></tr>'
    body = render_labeling_template(
        "admin/users.html.j2",
        {
            "slot_001": html.escape(str(payload.get('count') or 0)),
            "slot_002": html.escape(str(payload.get('active_count') or 0)),
            "slot_003": html.escape(str(payload.get('inactive_count') or 0)),
            "slot_004": html.escape(str(payload.get('missing_registry_row_count') or 0)),
            "slot_005": role_options,
            "slot_006": status_options,
            "slot_007": rows_html,
        },
    )
    return body.encode("utf-8")


def _admin_user_html(*, user: str, work: Mapping[str, object], dashboard_row: Mapping[str, object] | None) -> bytes:
    progress = work.get("progress_summary") if isinstance(work.get("progress_summary"), Mapping) else {}
    empty_state = work.get("empty_state") if isinstance(work.get("empty_state"), Mapping) else {}
    dataset_queue_state = work.get("dataset_queue_state") if isinstance(work.get("dataset_queue_state"), Mapping) else {}
    row = dashboard_row or {}
    landing_url = str(row.get("expected_user_labeler_landing_url") or _dashboard_url_for_expected_user("/", user))
    dashboard_url = str(row.get("expected_user_dashboard_url") or _dashboard_url_for_expected_user(DASHBOARD_PATH, user))
    dataset_queue_url = str(row.get("expected_user_dataset_queue_url") or _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, user))
    personal_work_url = str(row.get("expected_user_personal_work_url") or _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, user))
    personal_dataset_queue_url = str(
        row.get("expected_user_personal_dataset_queue_url")
        or _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, user)
    )
    identity_probe_url = _dashboard_url_for_expected_user(IDENTITY_PROBE_PATH, user)
    ready_text = (
        "ready row draft; safe-share review required"
        if bool(row.get("ready_to_invite"))
        else "not-ready diagnostic row"
    )
    copy_label = str(row.get("copy_label") or ("Copy ready-row draft" if bool(row.get("ready_to_invite")) else "Copy not-ready note"))
    copy_intent = str(row.get("copy_intent") or ("ready_row_draft" if bool(row.get("ready_to_invite")) else "diagnostic_note"))
    invite_reasons = row.get("invite_reasons") if isinstance(row.get("invite_reasons"), list) else []
    invite_actions = row.get("invite_actions") if isinstance(row.get("invite_actions"), list) else []
    reassignment_session_safety = (
        work.get("reassignment_session_safety")
        if isinstance(work.get("reassignment_session_safety"), Mapping)
        else {}
    )
    reassignment_safety_html = ""
    if bool(work.get("reassignment_session_safety_blocks_labeler_mutation")):
        reassignment_recordings = (
            work.get("reassignment_session_safety_active_session_assignment_mismatch_recording_ids")
            if isinstance(
                work.get("reassignment_session_safety_active_session_assignment_mismatch_recording_ids"),
                list,
            )
            else reassignment_session_safety.get("active_session_assignment_mismatch_recording_ids")
            if isinstance(
                reassignment_session_safety.get("active_session_assignment_mismatch_recording_ids"),
                list,
            )
            else []
        )
        reassignment_repair_buttons = "".join(
            "<button type=\"button\" "
            f"data-repair-route=\"/api/admin/recordings/{quote(str(recording_id), safe='')}/repair-reassignment-sessions\" "
            f"onclick=\"repairReassignmentSessions(this)\">Repair sessions for {html.escape(str(recording_id))}</button>"
            for recording_id in reassignment_recordings
        )
        reassignment_safety_html = (
            "<p><b>Reassignment session safety:</b> blocked by stale previous-owner sessions. "
            f"Affected recordings: {html.escape(', '.join(str(recording_id) for recording_id in reassignment_recordings) or 'unknown')}. "
            f"Operator action: {html.escape(str(work.get('reassignment_session_safety_operator_action') or reassignment_session_safety.get('operator_action') or ''))}</p>"
            f"{reassignment_repair_buttons}"
            '<pre id="reassignment-repair-result" class="muted"></pre>'
        )
    blocked_actions = (
        row.get("recordings_without_open_tasks_actions")
        if isinstance(row.get("recordings_without_open_tasks_actions"), list)
        else []
    )
    recording_rows: list[str] = []
    for recording in work.get("recordings", []):
        if not isinstance(recording, Mapping):
            continue
        recording_id = str(recording.get("recording_id") or "")
        tasks = [task for task in recording.get("tasks", []) if isinstance(task, Mapping)]
        task_rows = "".join(
            "<li>"
            f"<a href=\"/admin/tasks/{quote(str(task.get('task_id') or ''), safe='')}\">{html.escape(str(task.get('task_id') or ''))}</a>"
            f" {html.escape(str(task.get('workflow_kind') or ''))}"
            f" | state {html.escape(str(task.get('state') or ''))}"
            f" | priority {html.escape(str(task.get('priority') if task.get('priority') is not None else ''))}"
            f"{' | ' + html.escape(str(task.get('title') or '')) if task.get('title') else ''}"
            "</li>"
            for task in tasks
        ) or f"<li class=\"muted\">{html.escape(str(recording.get('no_open_task_message') or 'No startable tasks.'))}</li>"
        recording_rows.append(
            "<section class=\"recording\">"
            f"<h2><a href=\"/admin/recordings/{quote(recording_id, safe='')}\">{html.escape(recording_id)}</a></h2>"
            f"<p class=\"muted\">{html.escape(str(recording.get('startable_task_count') or 0))} startable / "
            f"{html.escape(str(recording.get('total_task_count') or 0))} total; "
            f"{html.escape(str(recording.get('non_startable_task_count') or 0))} non-startable; "
            f"{html.escape(str(recording.get('complete_task_count') or 0))} complete</p>"
            + (
                f"<p><b>Instructions:</b> {html.escape(str(recording.get('assignment_notes') or ''))}</p>"
                if recording.get("assignment_notes")
                else ""
            )
            + f"<ul>{task_rows}</ul>"
            "</section>"
        )
    recordings_html = "\n".join(recording_rows) or '<p class="muted">No active assigned recordings for this user.</p>'
    body = render_labeling_template(
        "admin/user.html.j2",
        {
            "slot_001": html.escape(user),
            "slot_002": html.escape(user),
            "slot_003": html.escape(ready_text),
            "slot_004": html.escape(landing_url),
            "slot_005": html.escape(landing_url),
            "slot_006": html.escape(dashboard_url),
            "slot_007": html.escape(dashboard_url),
            "slot_008": html.escape(personal_work_url),
            "slot_009": html.escape(personal_work_url),
            "slot_010": html.escape(personal_dataset_queue_url),
            "slot_011": html.escape(personal_dataset_queue_url),
            "slot_012": html.escape(dataset_queue_url),
            "slot_013": html.escape(dataset_queue_url),
            "slot_014": html.escape(identity_probe_url),
            "slot_015": html.escape(identity_probe_url),
            "slot_016": html.escape(str(empty_state.get('code') or '')),
            "slot_017": html.escape(str(empty_state.get('message') or '')),
            "slot_018": html.escape(str(dataset_queue_state.get('code') or '')),
            "slot_019": html.escape(str(dataset_queue_state.get('title') or '')),
            "slot_020": reassignment_safety_html,
            "slot_021": html.escape(', '.join(str(reason) for reason in invite_reasons) or 'none'),
            "slot_022": html.escape(copy_intent),
            "slot_023": html.escape(' '.join(str(action) for action in [*invite_actions, *blocked_actions]) or 'none'),
            "slot_024": html.escape(copy_label),
            "slot_025": html.escape(str(row.get('invitation_message') or '')),
            "slot_026": html.escape(str(progress.get('waiting_recording_count', 0))),
            "slot_027": html.escape(str(progress.get('complete_recording_count', 0))),
            "slot_028": html.escape(str(progress.get('blocked_recording_count', 0))),
            "slot_029": html.escape(str(progress.get('open_task_count', work.get('startable_task_count', 0)))),
            "slot_030": html.escape(str(progress.get('complete_task_count', work.get('complete_task_count', 0)))),
            "slot_031": recordings_html,
        },
    )
    return body.encode("utf-8")

