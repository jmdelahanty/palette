"""Admin registry and export helpers for Palette web labeling."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

REGISTRY_PATH_ENV_VAR = "PALETTE_REGISTRY_PATH"

def _task_title(task: Mapping[str, object]) -> str:
    title = str(task.get("title") or "").strip()
    if title:
        return title
    bits = [str(task.get("workflow_kind") or "task")]
    component = task.get("component_name")
    if component:
        bits.append(str(component))
    run_name = task.get("run_name")
    if run_name:
        bits.append(str(run_name))
    return " / ".join(bits)

def _admin_task_state_counts(tasks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        state = str(task.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts

def _admin_workflow_counts(tasks: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        workflow = str(task.get("workflow_kind") or "unknown")
        counts[workflow] = counts.get(workflow, 0) + 1
    return counts

def _admin_compact_task(task: Mapping[str, object]) -> dict[str, object]:
    scope = task.get("scope")
    scope_keys = sorted(str(key) for key in scope.keys()) if isinstance(scope, Mapping) else []
    return {
        "task_id": str(task.get("task_id") or ""),
        "recording_id": str(task.get("recording_id") or ""),
        "assignee_user": str(task.get("assignee_user") or ""),
        "assignment_status": str(task.get("assignment_status") or ""),
        "workflow_kind": str(task.get("workflow_kind") or ""),
        "state": str(task.get("state") or ""),
        "title": _task_title(task),
        "dataset_id": str(task.get("dataset_id") or ""),
        "zarr_use": str(task.get("zarr_use") or ""),
        "stage_group": str(task.get("stage_group") or ""),
        "run_name": str(task.get("run_name") or ""),
        "component_name": str(task.get("component_name") or ""),
        "priority": task.get("priority"),
        "notes": str(task.get("notes") or ""),
        "created_at_utc": str(task.get("created_at_utc") or ""),
        "updated_at_utc": str(task.get("updated_at_utc") or ""),
        "completed_at_utc": str(task.get("completed_at_utc") or ""),
        "scope_keys": scope_keys,
        "admin_task_url": f"/admin/tasks/{quote(str(task.get('task_id') or ''), safe='')}",
    }

def _admin_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _admin_event_type_counts(events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts

def _admin_is_save_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip()
    return normalized in {
        "save_keypoints",
        "save_detect_instances",
        "save_detection",
        "checkpoint_subject_mask_roi",
        "apply_subject_mask_session_checkpoints",
        "apply_subject_mask_session_checkpoint",
    } or normalized.startswith(("save_", "admin_save_"))

def _admin_is_issue_event(event_type: str) -> bool:
    normalized = str(event_type or "").strip()
    return (
        normalized.startswith("keypoint_mark_")
        or normalized.startswith("detect_mark_")
        or normalized.startswith("subject_mask_mark_")
        or normalized.endswith("_issue")
        or "issue" in normalized
    )

def _admin_latest_event(events: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    return dict(events[0]) if events else None

def _admin_latest_matching_event(
    events: Sequence[Mapping[str, object]],
    predicate,
) -> dict[str, object] | None:
    for event in events:
        if predicate(str(event.get("event_type") or "")):
            return dict(event)
    return None

def _admin_optional_review_rows_by_task(
    store,
    task_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Return the persisted admin-review row for each requested task.

    The table-existence check keeps read-only inspection compatible with an older
    sidecar opened before :meth:`LabelingStore.initialize` can migrate it.
    """

    normalized_task_ids = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    if not normalized_task_ids:
        return {}
    store.initialize()
    table_row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='labeling_admin_reviews';"
    ).fetchone()
    if table_row is None:
        return {}
    placeholders = ",".join("?" for _ in normalized_task_ids)
    rows = store.conn.execute(
        f"""
        SELECT *
        FROM labeling_admin_reviews
        WHERE task_id IN ({placeholders})
        ORDER BY task_id, updated_at_utc DESC, created_at_utc DESC;
        """,
        normalized_task_ids,
    ).fetchall()
    reviews: dict[str, dict[str, object]] = {}
    for row in rows:
        row_dict = dict(row)
        task_id = str(row_dict.get("task_id") or "")
        if task_id and task_id not in reviews:
            reviews[task_id] = row_dict
    return reviews

def _admin_completed_work_payload(
    store,
    *,
    assignee_user: str | None = None,
    workflow_kind: str | None = None,
    recording_id: str | None = None,
    admin_review_state: str | None = "pending",
    event_limit_per_task: int = 1000,
) -> dict[str, object]:
    """Build the admin queue for completed work awaiting review."""

    tasks = [
        task
        for task in store.list_tasks(
            recording_id=recording_id or None,
            assignee_user=assignee_user or None,
            include_completed=True,
        )
        if str(task.get("state") or "") == "complete"
    ]
    if workflow_kind:
        tasks = [
            task
            for task in tasks
            if str(task.get("workflow_kind") or "") == str(workflow_kind)
        ]
    review_rows_by_task = _admin_optional_review_rows_by_task(
        store,
        [str(task.get("task_id") or "") for task in tasks],
    )
    requested_review_state = str(admin_review_state or "pending").strip() or "pending"
    include_all_review_states = requested_review_state in {"all", "*"}
    rows: list[dict[str, object]] = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        events = store.list_events(task_id=task_id, limit=max(1, int(event_limit_per_task)))
        event_counts = _admin_event_type_counts(events)
        save_events = [
            event
            for event in events
            if _admin_is_save_event(str(event.get("event_type") or ""))
        ]
        issue_events = [
            event
            for event in events
            if _admin_is_issue_event(str(event.get("event_type") or ""))
        ]
        review = review_rows_by_task.get(task_id, {})
        review_state = str(review.get("state") or "pending")
        if not include_all_review_states and review_state != requested_review_state:
            continue
        compact = _admin_compact_task(task)
        latest_event = _admin_latest_event(events)
        latest_save_event = _admin_latest_matching_event(events, _admin_is_save_event)
        rows.append(
            {
                **compact,
                "labeler_user": str(compact.get("assignee_user") or ""),
                "admin_review_state": review_state,
                "admin_review_id": str(review.get("review_id") or ""),
                "admin_review_reviewer_user": str(review.get("reviewer_user") or ""),
                "admin_review_updated_at_utc": str(review.get("updated_at_utc") or ""),
                "admin_review_notes": str(review.get("notes") or ""),
                "admin_review_correction_event_count": int(review.get("correction_event_count") or 0),
                "admin_review_required": review_state in {"", "pending"},
                "open_admin_review_url": str(compact.get("admin_task_url") or ""),
                "event_count": len(events),
                "event_counts": event_counts,
                "save_event_count": len(save_events),
                "issue_event_count": len(issue_events),
                "issue_event_types": sorted(
                    {
                        str(event.get("event_type") or "")
                        for event in issue_events
                        if str(event.get("event_type") or "")
                    }
                ),
                "latest_event": latest_event or {},
                "latest_event_type": str((latest_event or {}).get("event_type") or ""),
                "latest_event_at_utc": str((latest_event or {}).get("created_at_utc") or ""),
                "latest_save_event": latest_save_event or {},
                "latest_save_event_id": str((latest_save_event or {}).get("event_id") or ""),
                "latest_save_event_at_utc": str((latest_save_event or {}).get("created_at_utc") or ""),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("admin_review_state") or ""),
            str(row.get("completed_at_utc") or ""),
            str(row.get("labeler_user") or ""),
            str(row.get("recording_id") or ""),
        )
    )
    state_counts: dict[str, int] = {}
    workflow_counts: dict[str, int] = {}
    labeler_counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get("admin_review_state") or "pending")
        workflow = str(row.get("workflow_kind") or "unknown")
        labeler = str(row.get("labeler_user") or "unassigned")
        state_counts[state] = state_counts.get(state, 0) + 1
        workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
        labeler_counts[labeler] = labeler_counts.get(labeler, 0) + 1
    return {
        "ok": True,
        "schema": "palette.web_labeling_admin_completed_work.v1",
        "generated_at_utc": _admin_utc_now(),
        "store_path": str(store.path),
        "filters": {
            "assignee_user": str(assignee_user or ""),
            "workflow_kind": str(workflow_kind or ""),
            "recording_id": str(recording_id or ""),
            "admin_review_state": requested_review_state,
        },
        "counts": {
            "completed_work_count": len(rows),
            "pending_admin_review_count": state_counts.get("pending", 0),
            "accepted_as_is_count": state_counts.get("accepted_as_is", 0),
            "accepted_with_corrections_count": state_counts.get("accepted_with_corrections", 0),
            "needs_revision_count": state_counts.get("needs_revision", 0),
            "rejected_count": state_counts.get("rejected", 0),
        },
        "admin_review_state_counts": state_counts,
        "workflow_counts": workflow_counts,
        "labeler_counts": labeler_counts,
        "completed_work": rows,
        "mutation_routes_enabled": True,
        "operator_action": (
            "Inspect completed tasks from this queue. Use each task detail page to record an admin review decision."
        ),
    }

def _admin_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}

def _admin_event_snapshot(event: Mapping[str, object]) -> dict[str, object]:
    return {
        "event_id": str(event.get("event_id") or ""),
        "event_type": str(event.get("event_type") or ""),
        "created_at_utc": str(event.get("created_at_utc") or ""),
        "user": str(event.get("user") or ""),
        "recording_id": str(event.get("recording_id") or ""),
        "task_id": str(event.get("task_id") or ""),
        "target": dict(_admin_mapping(event.get("target"))),
        "before": dict(_admin_mapping(event.get("before"))),
        "after": dict(_admin_mapping(event.get("after"))),
    }

def _admin_points_summary(points: object) -> dict[str, object]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        return {"count": 0, "finite_count": 0, "missing_count": 0}
    count = 0
    finite_count = 0
    missing_count = 0
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            continue
        values = list(point)
        if len(values) < 2:
            continue
        count += 1
        try:
            x = float(values[0])  # type: ignore[arg-type]
            y = float(values[1])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            missing_count += 1
            continue
        if x == x and y == y:
            finite_count += 1
        else:
            missing_count += 1
    return {"count": count, "finite_count": finite_count, "missing_count": missing_count}

def _admin_keypoint_review_rows(
    events: Sequence[Mapping[str, object]],
    *,
    max_rows: int = 40,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_roi: set[str] = set()
    for event in events:
        if str(event.get("event_type") or "") != "save_keypoints":
            continue
        target = _admin_mapping(event.get("target"))
        before = _admin_mapping(event.get("before"))
        after = _admin_mapping(event.get("after"))
        readback = _admin_mapping(after.get("readback"))
        status = _admin_mapping(readback.get("status"))
        roi_idx = (
            target.get("roi_idx")
            if target.get("roi_idx") is not None
            else readback.get("roi_idx")
            if readback.get("roi_idx") is not None
            else before.get("roi_idx")
        )
        roi_key = str(roi_idx)
        if not roi_key or roi_key in seen_roi:
            continue
        seen_roi.add(roi_key)
        before_points = before.get("points")
        rows.append(
            {
                "roi_idx": roi_idx,
                "frame_idx": (
                    target.get("frame_idx")
                    if target.get("frame_idx") is not None
                    else readback.get("frame_idx")
                    if readback.get("frame_idx") is not None
                    else before.get("frame_idx")
                ),
                "event_id": str(event.get("event_id") or ""),
                "saved_at_utc": str(event.get("created_at_utc") or ""),
                "saved_by": str(event.get("user") or ""),
                "changed": bool(after.get("changed")),
                "reason_before": str(before.get("reason") or ""),
                "reason_after": str(readback.get("reason") or before.get("reason") or ""),
                "status_after": dict(status),
                "usable_keypoints": bool(status.get("usable_keypoints")),
                "refined_success": bool(status.get("refined_success")),
                "heading": status.get("heading"),
                "before_points_summary": _admin_points_summary(before_points),
            }
        )
        if len(rows) >= max_rows:
            break
    return rows

def _admin_subject_mask_review_rows(
    events: Sequence[Mapping[str, object]],
    *,
    max_rows: int = 40,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event_type not in {
            "checkpoint_subject_mask_roi",
            "apply_subject_mask_session_checkpoints",
            "apply_subject_mask_session_checkpoint",
        }:
            continue
        target = _admin_mapping(event.get("target"))
        before = _admin_mapping(event.get("before"))
        after = _admin_mapping(event.get("after"))
        rows.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "event_type": event_type,
                "saved_at_utc": str(event.get("created_at_utc") or ""),
                "saved_by": str(event.get("user") or ""),
                "roi_idx": target.get("roi_idx") if target.get("roi_idx") is not None else after.get("roi_idx"),
                "component_name": str(target.get("component_name") or after.get("component_name") or ""),
                "edit_revision_before": before.get("edit_revision"),
                "edit_revision_after": after.get("edit_revision"),
                "area_px_before": before.get("area_px_total"),
                "area_px_after": after.get("area_px_total"),
                "applied_checkpoint_count": after.get("applied_checkpoint_count"),
            }
        )
        if len(rows) >= max_rows:
            break
    return rows

def _admin_task_review_payload(
    store,
    task_id: str,
    *,
    event_limit: int = 500,
) -> dict[str, object] | None:
    """Build an admin review payload for a single task."""

    resolved_task_id = str(task_id or "").strip()
    if not resolved_task_id:
        return None
    task = store.get_task(resolved_task_id)
    if task is None:
        return None
    events = store.list_events(task_id=resolved_task_id, limit=max(1, int(event_limit)))
    review = _admin_optional_review_rows_by_task(store, [resolved_task_id]).get(resolved_task_id, {})
    event_counts = _admin_event_type_counts(events)
    save_events = [
        event
        for event in events
        if _admin_is_save_event(str(event.get("event_type") or ""))
    ]
    issue_events = [
        event
        for event in events
        if _admin_is_issue_event(str(event.get("event_type") or ""))
    ]
    latest_event = _admin_latest_event(events)
    latest_save_event = _admin_latest_matching_event(events, _admin_is_save_event)
    workflow_kind = str(task.get("workflow_kind") or "")
    return {
        "ok": True,
        "schema": "palette.web_labeling_admin_task_review.v1",
        "generated_at_utc": _admin_utc_now(),
        "store_path": str(store.path),
        "task": _admin_compact_task(task),
        "task_scope": task.get("scope") if isinstance(task.get("scope"), Mapping) else {},
        "task_raw": dict(task),
        "admin_review": {
            "state": str(review.get("state") or "pending"),
            "review_id": str(review.get("review_id") or ""),
            "reviewer_user": str(review.get("reviewer_user") or ""),
            "updated_at_utc": str(review.get("updated_at_utc") or ""),
            "notes": str(review.get("notes") or ""),
            "correction_event_count": int(review.get("correction_event_count") or 0),
            "mutation_routes_enabled": True,
        },
        "event_count": len(events),
        "event_counts": event_counts,
        "save_event_count": len(save_events),
        "issue_event_count": len(issue_events),
        "issue_event_types": sorted(
            {
                str(event.get("event_type") or "")
                for event in issue_events
                if str(event.get("event_type") or "")
            }
        ),
        "latest_event": _admin_event_snapshot(latest_event) if latest_event else {},
        "latest_save_event": _admin_event_snapshot(latest_save_event) if latest_save_event else {},
        "recent_events": [_admin_event_snapshot(event) for event in events[:100]],
        "keypoint_review": {
            "available": workflow_kind == "keypoints",
            "rows": _admin_keypoint_review_rows(events) if workflow_kind == "keypoints" else [],
            "operator_action": (
                "Inspect the latest saved ROI rows. The preview is read-only until an admin explicitly enables correction mode."
                if workflow_kind == "keypoints"
                else ""
            ),
        },
        "subject_mask_review": {
            "available": workflow_kind == "subject_mask_component",
            "rows": _admin_subject_mask_review_rows(events) if workflow_kind == "subject_mask_component" else [],
            "operator_action": (
                "Inspect checkpoint/apply events. Visual mask overlays and admin correction writes are a follow-up slice."
                if workflow_kind == "subject_mask_component"
                else ""
            ),
        },
        "task_state_mutation_route_enabled": True,
        "admin_review_mutation_routes_enabled": True,
        "operator_action": (
            "Inspect the completed task and record an admin review decision. Keypoint tasks also support explicit audited corrections."
        ),
    }

def _admin_registry_path_from_env() -> Path | None:
    value = str(os.environ.get(REGISTRY_PATH_ENV_VAR) or "").strip()
    return Path(value).expanduser() if value else None

def _admin_sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'

def _admin_registry_public_row(row: Mapping[str, object], *, table_name: str) -> dict[str, object]:
    return {
        "table": table_name,
        "dataset_id": str(row.get("dataset_id") or ""),
        "recording_id": str(row.get("recording_id") or ""),
        "artifact_kind": str(row.get("artifact_kind") or ""),
        "zarr_origin": str(row.get("zarr_origin") or ""),
        "zarr_use": str(row.get("zarr_use") or ""),
        "status": str(row.get("status") or ""),
        "zarr_path": str(row.get("zarr_path") or ""),
        "session_uuid": str(row.get("session_uuid") or ""),
    }

def _admin_registry_lookup(
    *,
    registry_path: Path | None,
    dataset_ids: Sequence[str],
    recording_ids: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "enabled": bool(registry_path),
        "path": str(registry_path) if registry_path else "",
        "available": False,
        "error": "",
        "matched_row_count": 0,
        "tables_scanned": [],
        "rows_by_dataset_id": {},
        "rows_by_recording_id": {},
    }
    if registry_path is None:
        result["error"] = f"{REGISTRY_PATH_ENV_VAR} is not set."
        return result
    if not registry_path.exists():
        result["error"] = f"Registry path does not exist: {registry_path}"
        return result
    dataset_filter = sorted({str(item).strip() for item in dataset_ids if str(item).strip()})
    recording_filter = sorted({str(item).strip() for item in recording_ids if str(item).strip()})
    if not dataset_filter and not recording_filter:
        result["available"] = True
        return result
    rows_by_dataset_id: dict[str, list[dict[str, object]]] = {}
    rows_by_recording_id: dict[str, list[dict[str, object]]] = {}
    try:
        connection = sqlite3.connect(str(registry_path))
        connection.row_factory = sqlite3.Row
        try:
            table_names = [
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            for table_name in table_names:
                columns = [
                    str(row["name"])
                    for row in connection.execute(
                        f"PRAGMA table_info({_admin_sql_identifier(table_name)})"
                    ).fetchall()
                ]
                column_set = set(columns)
                if not {"dataset_id", "recording_id"} & column_set:
                    continue
                selected_columns = [
                    column
                    for column in (
                        "dataset_id",
                        "recording_id",
                        "artifact_kind",
                        "zarr_origin",
                        "zarr_use",
                        "status",
                        "zarr_path",
                        "session_uuid",
                    )
                    if column in column_set
                ]
                if not selected_columns:
                    continue
                where_parts: list[str] = []
                params: list[object] = []
                if dataset_filter and "dataset_id" in column_set:
                    where_parts.append(
                        f"{_admin_sql_identifier('dataset_id')} IN ({','.join('?' for _ in dataset_filter)})"
                    )
                    params.extend(dataset_filter)
                if recording_filter and "recording_id" in column_set:
                    where_parts.append(
                        f"{_admin_sql_identifier('recording_id')} IN ({','.join('?' for _ in recording_filter)})"
                    )
                    params.extend(recording_filter)
                if not where_parts:
                    continue
                query = (
                    "SELECT "
                    + ", ".join(_admin_sql_identifier(column) for column in selected_columns)
                    + f" FROM {_admin_sql_identifier(table_name)} WHERE "
                    + " OR ".join(f"({part})" for part in where_parts)
                    + " LIMIT 5000"
                )
                result["tables_scanned"] = [
                    *list(result.get("tables_scanned", [])),
                    table_name,
                ]
                for sqlite_row in connection.execute(query, params).fetchall():
                    row = {column: sqlite_row[column] if column in sqlite_row.keys() else "" for column in selected_columns}
                    public_row = _admin_registry_public_row(row, table_name=table_name)
                    dataset_id = str(public_row.get("dataset_id") or "")
                    recording_id = str(public_row.get("recording_id") or "")
                    if dataset_id:
                        rows_by_dataset_id.setdefault(dataset_id, []).append(public_row)
                    if recording_id:
                        rows_by_recording_id.setdefault(recording_id, []).append(public_row)
        finally:
            connection.close()
    except Exception as exc:
        result["error"] = str(exc)
        return result
    result["available"] = True
    result["matched_row_count"] = sum(len(rows) for rows in rows_by_dataset_id.values())
    result["rows_by_dataset_id"] = rows_by_dataset_id
    result["rows_by_recording_id"] = rows_by_recording_id
    return result

def _admin_registry_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "match_count": len(rows),
        "dataset_ids": sorted({str(row.get("dataset_id") or "") for row in rows if str(row.get("dataset_id") or "")}),
        "recording_ids": sorted({str(row.get("recording_id") or "") for row in rows if str(row.get("recording_id") or "")}),
        "statuses": sorted({str(row.get("status") or "") for row in rows if str(row.get("status") or "")}),
        "zarr_uses": sorted({str(row.get("zarr_use") or "") for row in rows if str(row.get("zarr_use") or "")}),
        "artifact_kinds": sorted({str(row.get("artifact_kind") or "") for row in rows if str(row.get("artifact_kind") or "")}),
        "zarr_origins": sorted({str(row.get("zarr_origin") or "") for row in rows if str(row.get("zarr_origin") or "")}),
        "zarr_paths": sorted({str(row.get("zarr_path") or "") for row in rows if str(row.get("zarr_path") or "")}),
    }

def _admin_registry_warning(
    code: str,
    *,
    severity: str = "warning",
    dataset_id: str = "",
    recording_id: str = "",
    task_id: str = "",
    details: str = "",
    operator_action: str = "",
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "task_id": task_id,
        "details": details,
        "operator_action": operator_action,
        **(dict(extra) if isinstance(extra, Mapping) else {}),
    }

def _admin_registry_warnings_for_recording(recording: Mapping[str, object]) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    registry_rows = [
        row
        for row in (
            recording.get("registry_rows")
            if isinstance(recording.get("registry_rows"), list)
            else []
        )
        if isinstance(row, Mapping)
    ]
    tasks = [
        task
        for task in (
            recording.get("tasks")
            if isinstance(recording.get("tasks"), list)
            else []
        )
        if isinstance(task, Mapping)
    ]
    recording_id = str(recording.get("recording_id") or "")
    active_training_rows = [
        row
        for row in registry_rows
        if str(row.get("status") or "") == "active"
        and str(row.get("zarr_use") or "") == "training"
    ]
    if not active_training_rows:
        warnings.append(
            _admin_registry_warning(
                "recording_missing_active_training_registry_row",
                recording_id=recording_id,
                details=(
                    "This assigned recording has no active training registry row in the configured registry."
                ),
                operator_action=(
                    "Confirm the assigned recording has a registered active training Zarr, or set PALETTE_REGISTRY_PATH to the registry that contains it."
                ),
            )
        )
    seen_inactive_keys: set[tuple[str, str, str]] = set()
    for row in registry_rows:
        status = str(row.get("status") or "")
        if status and status != "active":
            key = (
                str(row.get("table") or ""),
                str(row.get("dataset_id") or ""),
                status,
            )
            if key in seen_inactive_keys:
                continue
            seen_inactive_keys.add(key)
            warnings.append(
                _admin_registry_warning(
                    "registry_status_not_active",
                    dataset_id=str(row.get("dataset_id") or ""),
                    recording_id=recording_id,
                    details=f"Registry row is status={status}, not active.",
                    operator_action="Inspect whether this assignment points at the current active registry dataset.",
                    extra={
                        "registry_table": str(row.get("table") or ""),
                        "registry_status": status,
                        "registry_zarr_use": str(row.get("zarr_use") or ""),
                        "registry_zarr_path": str(row.get("zarr_path") or ""),
                    },
                )
            )
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        task_dataset_id = str(task.get("dataset_id") or "")
        task_zarr_use = str(task.get("zarr_use") or "")
        if task_dataset_id:
            dataset_rows = [
                row
                for row in registry_rows
                if str(row.get("dataset_id") or "") == task_dataset_id
            ]
            if not dataset_rows:
                warnings.append(
                    _admin_registry_warning(
                        "task_dataset_missing_registry_match",
                        dataset_id=task_dataset_id,
                        recording_id=recording_id,
                        task_id=task_id,
                        details="Task dataset_id has no matching registry row.",
                        operator_action="Confirm the task dataset_id or point PALETTE_REGISTRY_PATH at the registry that contains this dataset.",
                    )
                )
            elif task_zarr_use:
                registry_zarr_uses = {
                    str(row.get("zarr_use") or "")
                    for row in dataset_rows
                    if str(row.get("zarr_use") or "")
                }
                if registry_zarr_uses and task_zarr_use not in registry_zarr_uses:
                    warnings.append(
                        _admin_registry_warning(
                            "task_zarr_use_registry_mismatch",
                            dataset_id=task_dataset_id,
                            recording_id=recording_id,
                            task_id=task_id,
                            details=(
                                f"Task zarr_use={task_zarr_use} does not match registry zarr_use values "
                                f"{', '.join(sorted(registry_zarr_uses))}."
                            ),
                            operator_action="Inspect task generation inputs and registry identity before sharing or continuing work.",
                            extra={"registry_zarr_uses": sorted(registry_zarr_uses)},
                        )
                    )
    return warnings

def _admin_export_join(values: object) -> str:
    if isinstance(values, Mapping):
        return json.dumps(values, sort_keys=True)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return ", ".join(str(value) for value in values if str(value))
    return str(values or "")

def _admin_dataset_export_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        dataset_registry_summary = (
            dataset.get("registry_summary")
            if isinstance(dataset.get("registry_summary"), Mapping)
            else {}
        )
        recordings = dataset.get("recordings") if isinstance(dataset.get("recordings"), list) else []
        if not recordings:
            dataset_warnings = (
                dataset.get("registry_warnings")
                if isinstance(dataset.get("registry_warnings"), list)
                else []
            )
            rows.append(
                {
                    "dataset_id": str(dataset.get("dataset_id") or ""),
                    "dataset_label": str(dataset.get("dataset_label") or ""),
                    "recording_id": "",
                    "assignee_user": "",
                    "assignment_status": "",
                    "task_count": int(dataset.get("task_count") or 0),
                    "open_task_count": int(dataset.get("open_task_count") or 0),
                    "complete_task_count": int(dataset.get("complete_task_count") or 0),
                    "non_startable_task_count": int(dataset.get("non_startable_task_count") or 0),
                    "progress_percent": float(dataset.get("progress_percent") or 0.0),
                    "blocked": bool(dataset.get("blocked_recording_count")),
                    "blocked_reason": "",
                    "workflow_counts": _admin_export_join(dataset.get("workflow_counts", {})),
                    "state_counts": _admin_export_join(dataset.get("state_counts", {})),
                    "active_session_count": int(dataset.get("active_session_count") or 0),
                    "stale_session_count": int(dataset.get("stale_session_count") or 0),
                    "failed_promotion_count": int(dataset.get("failed_promotion_count") or 0),
                    "latest_event_type": "",
                    "latest_event_at_utc": "",
                    "latest_save_event_id": "",
                    "registry_match_count": int(dataset_registry_summary.get("match_count") or 0),
                    "registry_statuses": _admin_export_join(dataset_registry_summary.get("statuses", [])),
                    "registry_zarr_uses": _admin_export_join(dataset_registry_summary.get("zarr_uses", [])),
                    "registry_artifact_kinds": _admin_export_join(dataset_registry_summary.get("artifact_kinds", [])),
                    "registry_zarr_paths": _admin_export_join(dataset_registry_summary.get("zarr_paths", [])),
                    "registry_warning_count": len(dataset_warnings),
                    "registry_warning_codes": _admin_export_join(
                        sorted(
                            {
                                str(warning.get("code") or "")
                                for warning in dataset_warnings
                                if isinstance(warning, Mapping) and str(warning.get("code") or "")
                            }
                        )
                    ),
                    "admin_recording_url": "",
                    "labeler_queue_url": "",
                    "labeler_work_url": "",
                }
            )
            continue
        for recording in recordings:
            if not isinstance(recording, Mapping):
                continue
            latest_event = recording.get("latest_event") if isinstance(recording.get("latest_event"), Mapping) else {}
            latest_save = (
                recording.get("latest_save_event")
                if isinstance(recording.get("latest_save_event"), Mapping)
                else {}
            )
            registry_summary = (
                recording.get("registry_summary")
                if isinstance(recording.get("registry_summary"), Mapping)
                else {}
            )
            registry_warnings = (
                recording.get("registry_warnings")
                if isinstance(recording.get("registry_warnings"), list)
                else []
            )
            rows.append(
                {
                    "dataset_id": str(dataset.get("dataset_id") or ""),
                    "dataset_label": str(dataset.get("dataset_label") or ""),
                    "recording_id": str(recording.get("recording_id") or ""),
                    "assignee_user": str(recording.get("assignee_user") or ""),
                    "assignment_status": str(recording.get("assignment_status") or ""),
                    "task_count": int(recording.get("task_count") or 0),
                    "open_task_count": int(recording.get("open_task_count") or 0),
                    "complete_task_count": int(recording.get("complete_task_count") or 0),
                    "non_startable_task_count": int(recording.get("non_startable_task_count") or 0),
                    "progress_percent": float(recording.get("progress_percent") or 0.0),
                    "blocked": bool(recording.get("blocked")),
                    "blocked_reason": str(recording.get("blocked_reason") or ""),
                    "workflow_counts": _admin_export_join(recording.get("workflow_counts", {})),
                    "state_counts": _admin_export_join(recording.get("state_counts", {})),
                    "active_session_count": int(recording.get("active_session_count") or 0),
                    "stale_session_count": int(recording.get("stale_session_count") or 0),
                    "failed_promotion_count": int(recording.get("failed_promotion_count") or 0),
                    "latest_event_type": str(latest_event.get("event_type") or ""),
                    "latest_event_at_utc": str(latest_event.get("created_at_utc") or ""),
                    "latest_save_event_id": str(latest_save.get("event_id") or ""),
                    "registry_match_count": int(registry_summary.get("match_count") or 0),
                    "registry_statuses": _admin_export_join(registry_summary.get("statuses", [])),
                    "registry_zarr_uses": _admin_export_join(registry_summary.get("zarr_uses", [])),
                    "registry_artifact_kinds": _admin_export_join(registry_summary.get("artifact_kinds", [])),
                    "registry_zarr_paths": _admin_export_join(registry_summary.get("zarr_paths", [])),
                    "registry_warning_count": len(registry_warnings),
                    "registry_warning_codes": _admin_export_join(
                        sorted(
                            {
                                str(warning.get("code") or "")
                                for warning in registry_warnings
                                if isinstance(warning, Mapping) and str(warning.get("code") or "")
                            }
                        )
                    ),
                    "admin_recording_url": str(recording.get("admin_recording_url") or ""),
                    "labeler_queue_url": str(recording.get("expected_user_personal_dataset_queue_url") or ""),
                    "labeler_work_url": str(recording.get("expected_user_personal_work_url") or ""),
                }
            )
    return rows

def _admin_dataset_export_csv(rows: Sequence[Mapping[str, object]]) -> str:
    fieldnames = [
        "dataset_id",
        "dataset_label",
        "recording_id",
        "assignee_user",
        "assignment_status",
        "task_count",
        "open_task_count",
        "complete_task_count",
        "non_startable_task_count",
        "progress_percent",
        "blocked",
        "blocked_reason",
        "workflow_counts",
        "state_counts",
        "active_session_count",
        "stale_session_count",
        "failed_promotion_count",
        "latest_event_type",
        "latest_event_at_utc",
        "latest_save_event_id",
        "registry_match_count",
        "registry_statuses",
        "registry_zarr_uses",
        "registry_artifact_kinds",
        "registry_zarr_paths",
        "registry_warning_count",
        "registry_warning_codes",
        "admin_recording_url",
        "labeler_queue_url",
        "labeler_work_url",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue()
