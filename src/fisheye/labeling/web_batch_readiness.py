"""Batch readiness report helpers for web-labeling handoffs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .assignment_store import LABELER_START_TASK_STATES, LabelingStore


def configure_batch_readiness_dependencies(dependencies: dict[str, object]) -> None:
    globals().update(dependencies)

def _batch_readiness_report_impl(store: LabelingStore) -> dict[str, object]:
    assignments = store.list_assignments(status=None)
    tasks = store.list_tasks(include_completed=True)
    active_sessions = store.list_sessions(include_closed=False, limit=10000)
    consistency = _store_consistency_report(store)
    active_assignments = [
        assignment
        for assignment in assignments
        if str(assignment.get("status") or "") == "active"
    ]
    active_recording_ids = {
        str(assignment.get("recording_id") or "")
        for assignment in active_assignments
        if str(assignment.get("recording_id") or "")
    }
    tasks_by_recording: dict[str, list[dict[str, object]]] = {}
    users: dict[str, dict[str, object]] = {}
    open_task_count = 0
    completed_task_count = 0
    active_open_task_count = 0
    active_completed_task_count = 0
    active_recordings_without_tasks_count = 0
    active_recordings_without_open_tasks_count = 0
    active_recordings_without_open_tasks_by_reason: dict[str, int] = {}
    active_users_without_open_tasks_count = 0

    for assignment in active_assignments:
        user = str(assignment.get("assignee_user") or "").strip()
        if not user:
            continue
        users.setdefault(
            user,
            {
                "user": user,
                "active_recordings": 0,
                "open_tasks": 0,
                "completed_tasks": 0,
                "total_tasks": 0,
            },
        )
        users[user]["active_recordings"] = int(users[user]["active_recordings"]) + 1

    for task in tasks:
        recording_id = str(task.get("recording_id") or "")
        tasks_by_recording.setdefault(recording_id, []).append(task)
        state = str(task.get("state") or "")
        startable = state in LABELER_START_TASK_STATES
        assignment_status = str(task.get("assignment_status") or "")
        assignee_user = str(task.get("assignee_user") or "").strip()
        if state == "complete":
            completed_task_count += 1
        elif startable:
            open_task_count += 1
        if assignment_status != "active" or not assignee_user:
            continue
        if state == "complete":
            active_completed_task_count += 1
        elif startable:
            active_open_task_count += 1
        row = users.setdefault(
            assignee_user,
            {
                "user": assignee_user,
                "active_recordings": 0,
                "open_tasks": 0,
                "completed_tasks": 0,
                "total_tasks": 0,
            },
        )
        row["total_tasks"] = int(row["total_tasks"]) + 1
        if state == "complete":
            row["completed_tasks"] = int(row["completed_tasks"]) + 1
        elif startable:
            row["open_tasks"] = int(row["open_tasks"]) + 1

    readiness_issues: list[dict[str, object]] = []
    readiness_warnings: list[dict[str, object]] = []
    if not active_assignments:
        readiness_issues.append(
            {
                "code": "no_active_assignments",
                "details": "No active recording assignments are available for labelers.",
            }
        )
    if active_open_task_count == 0:
        readiness_issues.append(
            {
                "code": "no_open_tasks",
                "details": "No startable tasks under active assignments are available for labelers.",
            }
        )

    for assignment in active_assignments:
        recording_id = str(assignment.get("recording_id") or "")
        recording_tasks = tasks_by_recording.get(recording_id, [])
        open_recording_tasks = [
            task
            for task in recording_tasks
            if str(task.get("state") or "") in LABELER_START_TASK_STATES
        ]
        if not recording_tasks:
            active_recordings_without_tasks_count += 1
            active_recordings_without_open_tasks_count += 1
            reason = "tasks_not_generated"
            active_recordings_without_open_tasks_by_reason[reason] = (
                active_recordings_without_open_tasks_by_reason.get(reason, 0) + 1
            )
            readiness_warnings.append(
                {
                    "code": "active_assignment_without_tasks",
                    "no_open_task_reason": reason,
                    "no_open_task_actions": _recordings_without_open_tasks_actions([reason]),
                    "recording_id": recording_id,
                    "assignee_user": assignment.get("assignee_user"),
                    "details": "Recording is actively assigned but has no labeling tasks.",
                }
            )
        elif not open_recording_tasks:
            active_recordings_without_open_tasks_count += 1
            reason = (
                "all_tasks_complete"
                if all(str(task.get("state") or "") == "complete" for task in recording_tasks)
                else "non_startable_task_state"
            )
            active_recordings_without_open_tasks_by_reason[reason] = (
                active_recordings_without_open_tasks_by_reason.get(reason, 0) + 1
            )
            readiness_warnings.append(
                {
                    "code": "active_assignment_without_open_tasks",
                    "no_open_task_reason": reason,
                    "no_open_task_actions": _recordings_without_open_tasks_actions([reason]),
                    "recording_id": recording_id,
                    "assignee_user": assignment.get("assignee_user"),
                    "details": "Recording is actively assigned but no tasks are in a startable labeling state.",
                }
            )

    for row in users.values():
        if int(row.get("open_tasks") or 0) == 0:
            active_users_without_open_tasks_count += 1
            readiness_warnings.append(
                {
                    "code": "active_user_without_open_tasks",
                    "assignee_user": row.get("user"),
                    "details": "User has active recording assignments but no startable tasks.",
                }
            )

    return {
        "ok": bool(consistency["ok"]) and not readiness_issues,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_owner_policy": _assignment_ownership_policy(),
        "assignment_ownership_integrity": consistency.get("assignment_ownership_integrity", {}),
        "counts": {
            "assignments": len(assignments),
            "active_assignments": len(active_assignments),
            "inactive_assignments": len(assignments) - len(active_assignments),
            "active_recordings": len(active_recording_ids),
            "active_users": len(users),
            "active_recordings_without_tasks": active_recordings_without_tasks_count,
            "active_recordings_without_open_tasks": active_recordings_without_open_tasks_count,
            "active_recordings_without_open_tasks_by_reason": dict(
                sorted(active_recordings_without_open_tasks_by_reason.items())
            ),
            "active_recordings_without_open_tasks_actions": _recordings_without_open_tasks_actions(
                active_recordings_without_open_tasks_by_reason
            ),
            "active_users_without_open_tasks": active_users_without_open_tasks_count,
            "tasks": len(tasks),
            "open_tasks": open_task_count,
            "completed_tasks": completed_task_count,
            "active_open_tasks": active_open_task_count,
            "active_completed_tasks": active_completed_task_count,
            "active_sessions": len(active_sessions),
        },
        "readiness_issue_count": len(readiness_issues),
        "readiness_warning_count": len(readiness_warnings),
        "readiness_issues": readiness_issues,
        "readiness_warnings": readiness_warnings,
        "store_consistency": consistency,
        "users": sorted(users.values(), key=lambda row: str(row.get("user") or "")),
    }
