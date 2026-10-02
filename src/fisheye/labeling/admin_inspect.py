"""Read-only admin inspection of any labeler's work, applied and saved.

The editors open archives for writing and their runtimes may create editable
runs, so inspection never uses them. Here every Zarr open is ``mode="r"``,
the store is read through a read-only SQLite connection, and no session,
event or checkpoint is created: an operator can look at a task while its
labeler is working on it.

A row is ``applied`` when its latest checkpoint was applied, ``saved`` when it
has a checkpoint not yet applied (active or applying), and ``untouched`` when
no labeler checkpoint exists in this store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SAVED_STATES = ("active", "applying")


class InspectError(ValueError):
    """The task or row cannot be inspected."""


def _connect(store_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(store_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT t.*, a.assignee_user FROM labeling_tasks t
        LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
        WHERE t.task_id = ?;
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise InspectError(f"Unknown task_id: {task_id}")
    task = dict(row)
    task["scope"] = json.loads(task.get("scope_json") or "{}")
    return task


def _open_root(zarr_path: str):
    import zarr

    if not zarr_path:
        raise InspectError("Task scope has no zarr_path.")
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)


def _mask_sources(task: Mapping[str, Any]):
    """Resolve, read-only, the refined mask run, component index and crop images."""

    scope = task["scope"]
    root = _open_root(str(scope.get("zarr_path") or ""))
    run_name = str(scope.get("refined_run") or "").strip()
    runs = root.get("refined_subject_masks_runs")
    if not run_name or runs is None or run_name not in runs:
        raise InspectError(f"refined_subject_masks_runs/{run_name or '<unset>'} not found.")
    run = runs[run_name]
    labels = [str(v) for v in (run.attrs.get("mask_labels") or [])]
    component = str(task.get("component_name") or scope.get("component_name") or "")
    if component not in labels:
        raise InspectError(f"Component {component!r} is not in {labels}.")
    candidates = []
    if scope.get("crop_run"):
        candidates.append(("task scope", str(scope["crop_run"])))
    subject_runs = root.get("subject_mask_runs")
    subject_run = str(scope.get("subject_run") or "")
    if subject_runs is not None and subject_run in subject_runs:
        crop = subject_runs[subject_run].attrs.get("crop_run")
        if crop:
            candidates.append(("source subject run", str(crop)))
    if run.attrs.get("source_crop_run"):
        candidates.append(("refined run", str(run.attrs["source_crop_run"])))
    if not candidates:
        raise InspectError("No crop run is declared for this mask run.")
    names = {name for _where, name in candidates}
    if len(names) > 1:
        raise InspectError(f"Crop run sources disagree: {candidates}.")
    crop_name = candidates[0][1]
    crops = root.get("crop_runs")
    if crops is None or crop_name not in crops or "roi_images" not in crops[crop_name]:
        raise InspectError(f"crop_runs/{crop_name}/roi_images not found.")
    images = crops[crop_name]["roi_images"]
    masks = run["masks_roi"]
    if int(images.shape[0]) != int(masks.shape[0]) or tuple(images.shape[1:3]) != tuple(masks.shape[2:4]):
        raise InspectError(f"Crop images {tuple(images.shape)} do not match masks {tuple(masks.shape)}.")
    return run, labels.index(component), images, run_name, crop_name


def _keypoint_sources(task: Mapping[str, Any]):
    from fisheye.tune import keypoint_review_backend as backend

    scope = task["scope"]
    _root, refined, crop, refined_run, crop_run = backend.resolve_latest_refined_and_crop(
        str(scope.get("zarr_path") or ""),
        refined_run=str(scope.get("refined_run") or "").strip() or None,
        crop_run=str(scope.get("crop_run") or "").strip() or None,
        mode="r",
    )
    return refined, crop, str(refined_run), str(crop_run)


def _row_count(task: Mapping[str, Any]) -> int:
    if task["workflow_kind"] == "subject_mask_component":
        run, _idx, _images, _run, _crop = _mask_sources(task)
        return int(run["masks_roi"].shape[0])
    if task["workflow_kind"] == "keypoints":
        refined, _crop, _run, _crop_run = _keypoint_sources(task)
        return int(refined["keypoints_roi"].shape[0])
    raise InspectError(f"Inspection supports mask and keypoint tasks, not {task['workflow_kind']!r}.")


def _latest_checkpoints(conn: sqlite3.Connection, task_id: str, roi_idx: int | None = None):
    sql = """
        SELECT checkpoint_id, roi_idx, state, user, updated_at_utc, applied_at_utc, apply_id
        FROM labeling_session_checkpoints
        WHERE task_id = ? AND state != 'discarded'
    """
    params: list[object] = [task_id]
    if roi_idx is not None:
        sql += " AND roi_idx = ?"
        params.append(int(roi_idx))
    latest: dict[int, dict[str, Any]] = {}
    for row in conn.execute(sql + " ORDER BY updated_at_utc;", params):
        latest[int(row["roi_idx"])] = dict(row)
    return latest


def _status(checkpoint: Mapping[str, Any] | None) -> str:
    if checkpoint is None:
        return "untouched"
    return "saved" if checkpoint["state"] in SAVED_STATES else "applied"


def inspect_task(store_path: Path, task_id: str) -> dict[str, Any]:
    """Every row of one task with its applied/saved/untouched status."""

    conn = _connect(store_path)
    try:
        task = _task(conn, task_id)
        targets = task["scope"].get("target_roi_indices")
        rows = [int(v) for v in targets] if isinstance(targets, list) and targets else list(range(_row_count(task)))
        latest = _latest_checkpoints(conn, task_id)
        listed = []
        for roi in rows:
            checkpoint = latest.get(roi)
            listed.append({
                "roi_idx": roi,
                "status": _status(checkpoint),
                "labeler": checkpoint["user"] if checkpoint else None,
                "saved_at_utc": checkpoint["updated_at_utc"] if checkpoint else None,
                "applied_at_utc": checkpoint["applied_at_utc"] if checkpoint else None,
            })
        counts = {name: sum(1 for r in listed if r["status"] == name) for name in ("applied", "saved", "untouched")}
        return {
            "ok": True,
            "task_id": task_id,
            "title": task.get("title") or task_id,
            "recording_id": task.get("recording_id"),
            "workflow_kind": task.get("workflow_kind"),
            "component_name": task.get("component_name"),
            "state": task.get("state"),
            "assignee": task.get("assignee_user"),
            "counts": counts,
            "rows": listed,
        }
    finally:
        conn.close()


def inspect_row(store_path: Path, task_id: str, roi_idx: int) -> dict[str, Any]:
    """One row's crop image with its applied label and any saved, unapplied label."""

    from .web_responses import _raw_array_payload
    from .web_runtimes import _subject_mask_checkpoint_mask

    conn = _connect(store_path)
    try:
        task = _task(conn, task_id)
        roi_idx = int(roi_idx)
        payload_row = conn.execute(
            """
            SELECT state, user, updated_at_utc, payload_json FROM labeling_session_checkpoints
            WHERE task_id = ? AND roi_idx = ? AND state != 'discarded'
            ORDER BY updated_at_utc DESC LIMIT 1;
            """,
            (task_id, roi_idx),
        ).fetchone()
        checkpoint = dict(payload_row) if payload_row else None
        status = _status(checkpoint)
        saved_payload = json.loads(checkpoint["payload_json"]) if checkpoint and status == "saved" else None
        result: dict[str, Any] = {
            "ok": True, "task_id": task_id, "roi_idx": roi_idx, "status": status,
            "workflow_kind": task["workflow_kind"],
            "labeler": checkpoint["user"] if checkpoint else None,
            "saved_at_utc": checkpoint["updated_at_utc"] if checkpoint else None,
            "read_only": True,
        }
        if task["workflow_kind"] == "subject_mask_component":
            run, comp_idx, images, run_name, crop_name = _mask_sources(task)
            if not 0 <= roi_idx < int(run["masks_roi"].shape[0]):
                raise InspectError("roi_idx is out of range.")
            result.update({
                "component_name": task.get("component_name"),
                "refined_run": run_name, "crop_run": crop_name,
                "image": _raw_array_payload(np.asarray(images[roi_idx], dtype=np.uint8)),
                "applied_mask": _raw_array_payload((np.asarray(run["masks_roi"][roi_idx, comp_idx]) > 0).astype(np.uint8)),
                "saved_mask": (
                    _raw_array_payload(_subject_mask_checkpoint_mask({"payload": saved_payload}))
                    if saved_payload is not None else None
                ),
            })
        elif task["workflow_kind"] == "keypoints":
            refined, crop, run_name, crop_name = _keypoint_sources(task)
            if not 0 <= roi_idx < int(refined["keypoints_roi"].shape[0]):
                raise InspectError("roi_idx is out of range.")
            points = np.asarray(refined["keypoints_roi"][roi_idx], dtype=float)
            labels = [str(v) for v in (refined.attrs.get("keypoint_labels") or [])] or [
                str(i + 1) for i in range(int(points.shape[0]))
            ]
            result.update({
                "refined_run": run_name, "crop_run": crop_name, "labels": labels,
                "image": _raw_array_payload(np.asarray(crop["roi_images"][roi_idx], dtype=np.uint8)),
                "applied_points": [[float(x) if np.isfinite(x) else None, float(y) if np.isfinite(y) else None]
                                   for x, y in points],
                "saved_operation": (saved_payload or {}).get("operation") if saved_payload else None,
                "saved_points": (saved_payload or {}).get("points") if saved_payload else None,
            })
        else:
            raise InspectError(f"Inspection supports mask and keypoint tasks, not {task['workflow_kind']!r}.")
        return result
    finally:
        conn.close()


def inspect_tasks(store_path: Path) -> dict[str, Any]:
    """Mask and keypoint tasks with labeler and applied/saved row counts."""

    conn = _connect(store_path)
    try:
        tasks = []
        for row in conn.execute(
            """
            SELECT t.task_id, t.title, t.recording_id, t.workflow_kind, t.component_name, t.state,
                   t.scope_json, a.assignee_user
            FROM labeling_tasks t LEFT JOIN recording_assignments a ON a.recording_id = t.recording_id
            WHERE t.workflow_kind IN ('subject_mask_component', 'keypoints')
            ORDER BY a.assignee_user, t.recording_id, t.component_name;
            """
        ):
            latest = _latest_checkpoints(conn, row["task_id"])
            statuses = [_status(c) for c in latest.values()]
            scope = json.loads(row["scope_json"] or "{}")
            targets = scope.get("target_roi_indices")
            tasks.append({
                "task_id": row["task_id"],
                "title": row["title"] or row["task_id"],
                "recording_id": row["recording_id"],
                "workflow_kind": row["workflow_kind"],
                "component_name": row["component_name"],
                "state": row["state"],
                "assignee": row["assignee_user"],
                "applied_rows": statuses.count("applied"),
                "saved_rows": statuses.count("saved"),
                "row_total": len(targets) if isinstance(targets, list) and targets else None,
                "labelers": sorted({c["user"] for c in latest.values()}),
            })
        return {"ok": True, "tasks": tasks}
    finally:
        conn.close()
