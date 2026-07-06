"""Registry-backed task generation helpers for Palette web labeling."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .assignment_store import LabelingStore


REGISTRY_PATH_ENV_VAR = "PALETTE_REGISTRY_PATH"


def _safe_task_id(*parts: object) -> str:
    raw = ":".join(str(part) for part in parts if part is not None)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"palette-labeling:{raw}"))

def _zarr_child_group_exists(zarr_path: object, child: str) -> bool:
    if not zarr_path:
        return False
    child_path = Path(str(zarr_path)).expanduser() / child
    if not child_path.is_dir():
        return False
    if (child_path / "zarr.json").exists() or (child_path / ".zgroup").exists():
        return True
    try:
        next(child_path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return False
    return True

def _zarr_child_path_exists(zarr_path: object, child: str) -> bool:
    if not zarr_path:
        return False
    child_path = Path(str(zarr_path)).expanduser() / child
    return child_path.exists()

def _read_zarr_attrs(path: Path) -> dict[str, object]:
    try:
        data = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = data.get("attributes")
    return dict(attrs) if isinstance(attrs, Mapping) else {}

def _keypoint_review_status_for_zarr(zarr_path: object) -> dict[str, object]:
    if not zarr_path:
        return {"approved": False, "reason": "missing_zarr_path"}
    refined_parent = Path(str(zarr_path)).expanduser() / "refined_keypoints_runs"
    parent_attrs = _read_zarr_attrs(refined_parent)
    candidate_names: list[str] = []
    for key in ("keypoint_review_status_latest", "latest"):
        value = parent_attrs.get(key)
        if value is not None and str(value).strip():
            candidate_names.append(str(value).strip())
    if not candidate_names:
        try:
            candidate_names = sorted(path.name for path in refined_parent.iterdir() if path.is_dir())
        except OSError:
            candidate_names = []
    for run_name in dict.fromkeys(candidate_names):
        run_attrs = _read_zarr_attrs(refined_parent / run_name)
        status = run_attrs.get("keypoint_review_status")
        if not isinstance(status, Mapping):
            continue
        state = str(status.get("state") or "").strip().lower()
        intended_use = str(status.get("intended_use") or "").strip().lower()
        approved = state == "approved" and (not intended_use or intended_use == "training")
        return {
            "approved": approved,
            "review_state": state or None,
            "review_intended_use": intended_use or None,
            "review_run": run_name,
            "review_status": dict(status),
            "reason": "approved" if approved else "not_approved",
        }
    return {
        "approved": False,
        "review_state": None,
        "review_intended_use": None,
        "review_run": candidate_names[0] if candidate_names else None,
        "review_status": None,
        "reason": "no_keypoint_review_status",
    }

def _detect_review_status_for_zarr(zarr_path: object) -> dict[str, object]:
    if not zarr_path:
        return {"approved": False, "reason": "missing_zarr_path"}
    refined_parent = Path(str(zarr_path)).expanduser() / "refined_detect_runs"
    parent_attrs = _read_zarr_attrs(refined_parent)
    candidate_names: list[str] = []
    for key in ("detect_review_status_latest", "latest"):
        value = parent_attrs.get(key)
        if value is not None and str(value).strip():
            candidate_names.append(str(value).strip())
    if not candidate_names:
        try:
            candidate_names = sorted(path.name for path in refined_parent.iterdir() if path.is_dir())
        except OSError:
            candidate_names = []
    for run_name in dict.fromkeys(candidate_names):
        run_attrs = _read_zarr_attrs(refined_parent / run_name)
        status = run_attrs.get("detect_review_status")
        if not isinstance(status, Mapping):
            continue
        state = str(status.get("state") or "").strip().lower()
        intended_use = str(status.get("intended_use") or "").strip().lower()
        approved = state == "approved" and (not intended_use or intended_use == "training")
        return {
            "approved": approved,
            "review_state": state or None,
            "review_intended_use": intended_use or None,
            "review_run": run_name,
            "review_status": dict(status),
            "reason": "approved" if approved else "not_approved",
        }
    return {
        "approved": False,
        "review_state": None,
        "review_intended_use": None,
        "review_run": candidate_names[0] if candidate_names else None,
        "review_status": None,
        "reason": "no_detect_review_status",
    }

def _registry_path_from_arg(raw: object | None) -> Path:
    if raw is not None and str(raw).strip() and str(raw).strip() != "auto":
        return Path(str(raw)).expanduser()
    from fisheye.registry.db import RegistryPaths

    return RegistryPaths.from_env(Path.cwd()).path.expanduser()

def _query_training_datasets_for_recordings(
    registry_path: Path,
    recording_ids: Sequence[str],
) -> list[dict[str, object]]:
    recording_ids = [str(item) for item in recording_ids if str(item).strip()]
    if not recording_ids:
        return []
    if not registry_path.expanduser().is_file():
        raise FileNotFoundError(f"Registry SQLite path not found: {registry_path}")
    conn = sqlite3.connect(str(registry_path.expanduser()))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in recording_ids)
        rows = conn.execute(
            f"""
            SELECT dataset_id, session_uuid, zarr_path, recording_id, zarr_origin,
                   zarr_use, dataset_status AS status, source_layout, dish_design,
                   COALESCE(subject_id, legacy_fish_id) AS fish_id,
                   subject_id, camera_serial, fps
            FROM dataset_context_current
            WHERE recording_id IN ({placeholders})
              AND zarr_use = 'training'
              AND (dataset_status IS NULL OR dataset_status != 'missing')
            ORDER BY recording_id, dataset_id
            """,
            recording_ids,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]

def _query_analysis_datasets_for_recordings(
    registry_path: Path,
    recording_ids: Sequence[str],
) -> list[dict[str, object]]:
    recording_ids = [str(item) for item in recording_ids if str(item).strip()]
    if not recording_ids:
        return []
    if not registry_path.expanduser().is_file():
        raise FileNotFoundError(f"Registry SQLite path not found: {registry_path}")
    conn = sqlite3.connect(str(registry_path.expanduser()))
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in recording_ids)
        rows = conn.execute(
            f"""
            SELECT dataset_id, session_uuid, zarr_path, recording_id, zarr_origin,
                   zarr_use, dataset_status AS status, source_layout, dish_design,
                   COALESCE(subject_id, legacy_fish_id) AS fish_id,
                   subject_id, camera_serial, fps
            FROM dataset_context_current
            WHERE recording_id IN ({placeholders})
              AND zarr_use = 'analysis'
              AND (dataset_status IS NULL OR dataset_status != 'missing')
            ORDER BY recording_id, dataset_id
            """,
            recording_ids,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]

def _query_subject_mask_components_for_recordings(
    registry_path: Path,
    recording_ids: Sequence[str],
    *,
    component_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    recording_ids = [str(item) for item in recording_ids if str(item).strip()]
    if not recording_ids:
        return []
    if not registry_path.expanduser().is_file():
        raise FileNotFoundError(f"Registry SQLite path not found: {registry_path}")
    component_names = [str(item).strip() for item in (component_names or ()) if str(item).strip()]
    conn = sqlite3.connect(str(registry_path.expanduser()))
    conn.row_factory = sqlite3.Row
    try:
        recording_placeholders = ",".join("?" for _ in recording_ids)
        component_filter = ""
        params: list[object] = list(recording_ids)
        if component_names:
            component_placeholders = ",".join("?" for _ in component_names)
            component_filter = f" AND component_name IN ({component_placeholders})"
            params.extend(component_names)
        rows = conn.execute(
            f"""
            SELECT dataset_id, zarr_path, zarr_origin, zarr_use, dataset_status AS status,
                   stage_group, run_name, component_name, component_family,
                   run_created_utc, recording_id, subject_mask_method,
                   label_schema_id, eye_component_mode, source_subject_mask_run,
                   source_subject_mask_stale_state, source_subject_mask_stale_reason,
                   available, review_state, review_method, review_intended_use,
                   review_reviewer, review_timestamp_utc, total_rois,
                   rows_with_component_mask, rows_with_component_mask_rate,
                   lifecycle_state, lifecycle_reason, quality_updated_utc,
                   quality_stale
            FROM subject_mask_component_quality_latest
            WHERE recording_id IN ({recording_placeholders})
              AND zarr_use = 'training'
              AND stage_group = 'refined_subject_masks_runs'
              AND COALESCE(available, 0) = 1
              AND (dataset_status IS NULL OR dataset_status != 'missing')
              {component_filter}
            ORDER BY recording_id, dataset_id, component_name, run_name
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]

def generate_keypoint_tasks_from_registry(
    *,
    store: LabelingStore,
    registry_path: Path,
    assignee_user: str | None = None,
    recording_id: str | None = None,
    review_filter: str = "needs_review",
    priority: int = 0,
    include_all: bool = False,
    auto_advance_on_save: bool = True,
) -> dict[str, object]:
    assignments = store.list_assignments(assignee_user=assignee_user, status="active")
    if recording_id:
        assignments = [row for row in assignments if str(row.get("recording_id") or "") == str(recording_id)]
    recording_ids = [str(row["recording_id"]) for row in assignments]
    datasets = _query_training_datasets_for_recordings(registry_path, recording_ids)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    mode = str(review_filter or "needs_review").strip().lower()
    assigned_by_recording = {str(row["recording_id"]): row for row in assignments}
    for dataset in datasets:
        dataset_recording_id = str(dataset.get("recording_id") or "")
        zarr_path = dataset.get("zarr_path")
        dataset_id = str(dataset.get("dataset_id") or "")
        if dataset_recording_id not in assigned_by_recording:
            skipped.append({"dataset_id": dataset_id, "reason": "recording_not_assigned"})
            continue
        if not _zarr_child_group_exists(zarr_path, "refined_keypoints_runs") or not _zarr_child_group_exists(zarr_path, "crop_runs"):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_keypoint_reviewable"})
            continue
        review_status = _keypoint_review_status_for_zarr(zarr_path)
        if mode in {"needs_review", "unapproved", "not_approved"} and bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "already_approved"})
            continue
        if mode == "approved" and not bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_approved"})
            continue
        scope = {
            "zarr_path": str(zarr_path),
            "dataset_id": dataset_id,
            "registry_path": str(registry_path),
            "include_all": bool(include_all),
            "filter_mode": "all" if include_all else "failed",
            "review_method": "manual",
            "review_intended_use": "training",
            "auto_advance_on_save": bool(auto_advance_on_save),
        }
        if review_status.get("review_run"):
            scope["refined_run"] = str(review_status["review_run"])
        task = store.upsert_task(
            task_id=_safe_task_id("keypoints", dataset_recording_id, dataset_id),
            recording_id=dataset_recording_id,
            workflow_kind="keypoints",
            dataset_id=dataset_id,
            zarr_use="training",
            stage_group="refined_keypoints_runs",
            run_name=str(review_status.get("review_run") or "") or None,
            title=f"Review keypoints: {dataset_id}",
            scope=scope,
            state="pending",
            priority=int(priority),
            notes=f"Generated from registry {registry_path}",
        )
        generated.append({"task": task, "dataset": dataset, "review_status": review_status})
    return {
        "registry_path": str(registry_path),
        "assignment_count": len(assignments),
        "dataset_count": len(datasets),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }

def generate_detect_training_tasks_from_registry(
    *,
    store: LabelingStore,
    registry_path: Path,
    assignee_user: str | None = None,
    recording_id: str | None = None,
    review_filter: str = "needs_review",
    priority: int = 0,
    include_all: bool = False,
    auto_advance_on_save: bool = True,
) -> dict[str, object]:
    assignments = store.list_assignments(assignee_user=assignee_user, status="active")
    if recording_id:
        assignments = [row for row in assignments if str(row.get("recording_id") or "") == str(recording_id)]
    recording_ids = [str(row["recording_id"]) for row in assignments]
    datasets = _query_training_datasets_for_recordings(registry_path, recording_ids)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    mode = str(review_filter or "needs_review").strip().lower()
    assigned_by_recording = {str(row["recording_id"]): row for row in assignments}
    for dataset in datasets:
        dataset_recording_id = str(dataset.get("recording_id") or "")
        zarr_path = dataset.get("zarr_path")
        dataset_id = str(dataset.get("dataset_id") or "")
        if dataset_recording_id not in assigned_by_recording:
            skipped.append({"dataset_id": dataset_id, "reason": "recording_not_assigned"})
            continue
        if not _zarr_child_group_exists(zarr_path, "refined_detect_runs") or not _zarr_child_path_exists(zarr_path, "raw_video/images_ds"):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_detect_training_reviewable"})
            continue
        review_status = _detect_review_status_for_zarr(zarr_path)
        if mode in {"needs_review", "unapproved", "not_approved"} and bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "already_approved"})
            continue
        if mode == "approved" and not bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_approved"})
            continue
        scope = {
            "zarr_path": str(zarr_path),
            "dataset_id": dataset_id,
            "registry_path": str(registry_path),
            "include_all": bool(include_all),
            "manual_score": 1.0,
            "manual_class_id": 0,
            "auto_advance_on_save": bool(auto_advance_on_save),
        }
        if review_status.get("review_run"):
            scope["refined_run"] = str(review_status["review_run"])
        task = store.upsert_task(
            task_id=_safe_task_id("detect_training", dataset_recording_id, dataset_id),
            recording_id=dataset_recording_id,
            workflow_kind="detect_training",
            dataset_id=dataset_id,
            zarr_use="training",
            stage_group="refined_detect_runs",
            run_name=str(review_status.get("review_run") or "") or None,
            title=f"Review detection boxes: {dataset_id}",
            scope=scope,
            state="pending",
            priority=int(priority),
            notes=f"Generated from registry {registry_path}",
        )
        generated.append({"task": task, "dataset": dataset, "review_status": review_status})
    return {
        "registry_path": str(registry_path),
        "assignment_count": len(assignments),
        "dataset_count": len(datasets),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }

def generate_detect_analysis_tasks_from_registry(
    *,
    store: LabelingStore,
    registry_path: Path,
    assignee_user: str | None = None,
    recording_id: str | None = None,
    review_filter: str = "needs_review",
    priority: int = 0,
    editable: bool = False,
    promote_training_zarr: str | None = None,
    promote_target_crop_run: str | None = None,
    promote_target_refined_run: str | None = None,
    promote_label_origin: str = "palette_labeling_work",
    promote_include_negative: bool = True,
    promote_allow_unreviewed_negative: bool = False,
    promote_target_size: tuple[int, int] | None = None,
    auto_advance_on_save: bool = True,
) -> dict[str, object]:
    assignments = store.list_assignments(assignee_user=assignee_user, status="active")
    if recording_id:
        assignments = [row for row in assignments if str(row.get("recording_id") or "") == str(recording_id)]
    recording_ids = [str(row["recording_id"]) for row in assignments]
    datasets = _query_analysis_datasets_for_recordings(registry_path, recording_ids)
    promote_mode = str(promote_training_zarr or "").strip()
    training_dataset_by_recording: dict[str, dict[str, object]] = {}
    if promote_mode == "auto":
        training_datasets = _query_training_datasets_for_recordings(registry_path, recording_ids)
        for training_dataset in training_datasets:
            rid = str(training_dataset.get("recording_id") or "")
            zarr_path = str(training_dataset.get("zarr_path") or "").strip()
            if rid and zarr_path and rid not in training_dataset_by_recording:
                training_dataset_by_recording[rid] = training_dataset
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    mode = str(review_filter or "needs_review").strip().lower()
    assigned_by_recording = {str(row["recording_id"]): row for row in assignments}
    for dataset in datasets:
        dataset_recording_id = str(dataset.get("recording_id") or "")
        zarr_path = dataset.get("zarr_path")
        dataset_id = str(dataset.get("dataset_id") or "")
        if dataset_recording_id not in assigned_by_recording:
            skipped.append({"dataset_id": dataset_id, "reason": "recording_not_assigned"})
            continue
        if not _zarr_child_group_exists(zarr_path, "refined_detect_runs"):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_detect_analysis_reviewable"})
            continue
        review_status = _detect_review_status_for_zarr(zarr_path)
        if mode in {"needs_review", "unapproved", "not_approved"} and bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "already_approved"})
            continue
        if mode == "approved" and not bool(review_status.get("approved")):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "not_approved"})
            continue
        resolved_promotion_zarr: str | None = None
        resolved_promotion_dataset_id: str | None = None
        if promote_mode:
            if promote_mode == "auto":
                training_dataset = training_dataset_by_recording.get(dataset_recording_id)
                resolved_promotion_zarr = str(training_dataset.get("zarr_path") or "").strip() if training_dataset else None
                resolved_promotion_dataset_id = str(training_dataset.get("dataset_id") or "").strip() if training_dataset else None
                if not resolved_promotion_zarr:
                    skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "reason": "missing_training_zarr_for_promotion"})
                    continue
            else:
                resolved_promotion_zarr = promote_mode
        refined_parent_attrs = _read_zarr_attrs(Path(str(zarr_path)).expanduser() / "refined_detect_runs")
        collection_id = str(refined_parent_attrs.get("latest_collection") or "").strip()
        scope = {
            "zarr_path": str(zarr_path),
            "dataset_id": dataset_id,
            "registry_path": str(registry_path),
            "editable": bool(editable or resolved_promotion_zarr),
            "manual_score": 1.0,
            "manual_class_id": 0,
            "auto_advance_on_save": bool(auto_advance_on_save),
        }
        if resolved_promotion_zarr:
            scope["promote_training_zarr"] = resolved_promotion_zarr
            if resolved_promotion_dataset_id:
                scope["promote_training_dataset_id"] = resolved_promotion_dataset_id
            scope["promote_label_origin"] = str(promote_label_origin or "palette_labeling_work")
            scope["promote_include_negative"] = bool(promote_include_negative)
            scope["promote_allow_unreviewed_negative"] = bool(promote_allow_unreviewed_negative)
            if promote_target_crop_run:
                scope["promote_target_crop_run"] = str(promote_target_crop_run)
            if promote_target_refined_run:
                scope["promote_target_refined_run"] = str(promote_target_refined_run)
            if promote_target_size:
                scope["promote_target_size"] = [int(promote_target_size[0]), int(promote_target_size[1])]
        if collection_id:
            scope["collection_id"] = collection_id
        if review_status.get("review_run"):
            scope["refined_run"] = str(review_status["review_run"])
        task = store.upsert_task(
            task_id=_safe_task_id("detect_analysis", dataset_recording_id, dataset_id),
            recording_id=dataset_recording_id,
            workflow_kind="detect_analysis",
            dataset_id=dataset_id,
            zarr_use="analysis",
            stage_group="refined_detect_runs",
            run_name=str(review_status.get("review_run") or collection_id or "") or None,
            title=f"Review analysis detection video: {dataset_id}",
            scope=scope,
            state="pending",
            priority=int(priority),
            notes=f"Generated from registry {registry_path}",
        )
        generated.append({"task": task, "dataset": dataset, "review_status": review_status})
    return {
        "registry_path": str(registry_path),
        "assignment_count": len(assignments),
        "dataset_count": len(datasets),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }

def generate_subject_mask_component_tasks_from_registry(
    *,
    store: LabelingStore,
    registry_path: Path,
    assignee_user: str | None = None,
    recording_id: str | None = None,
    review_filter: str = "needs_review",
    priority: int = 0,
    component_names: Sequence[str] | None = None,
    auto_advance_on_save: bool = True,
) -> dict[str, object]:
    assignments = store.list_assignments(assignee_user=assignee_user, status="active")
    if recording_id:
        assignments = [row for row in assignments if str(row.get("recording_id") or "") == str(recording_id)]
    recording_ids = [str(row["recording_id"]) for row in assignments]
    components = _query_subject_mask_components_for_recordings(
        registry_path,
        recording_ids,
        component_names=component_names,
    )
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    mode = str(review_filter or "needs_review").strip().lower()
    assigned_by_recording = {str(row["recording_id"]): row for row in assignments}
    for component in components:
        dataset_recording_id = str(component.get("recording_id") or "")
        zarr_path = component.get("zarr_path")
        dataset_id = str(component.get("dataset_id") or "")
        run_name = str(component.get("run_name") or "").strip()
        component_name = str(component.get("component_name") or "").strip()
        if dataset_recording_id not in assigned_by_recording:
            skipped.append({"dataset_id": dataset_id, "component_name": component_name, "reason": "recording_not_assigned"})
            continue
        if not zarr_path or not run_name or not component_name:
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "component_name": component_name, "reason": "missing_required_scope"})
            continue
        if not _zarr_child_group_exists(zarr_path, "refined_subject_masks_runs") or not _zarr_child_group_exists(zarr_path, "crop_runs"):
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "component_name": component_name, "reason": "not_subject_mask_reviewable"})
            continue
        review_state = str(component.get("review_state") or "").strip().lower()
        review_intended_use = str(component.get("review_intended_use") or "").strip().lower()
        approved = review_state == "approved" and (not review_intended_use or review_intended_use == "training")
        if mode in {"needs_review", "unapproved", "not_approved"} and approved:
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "component_name": component_name, "reason": "already_approved"})
            continue
        if mode == "approved" and not approved:
            skipped.append({"dataset_id": dataset_id, "recording_id": dataset_recording_id, "component_name": component_name, "reason": "not_approved"})
            continue
        scope = {
            "zarr_path": str(zarr_path),
            "dataset_id": dataset_id,
            "registry_path": str(registry_path),
            "refined_run": run_name,
            "component_name": component_name,
            "review_method": "manual",
            "review_intended_use": "training",
            "auto_advance_on_save": bool(auto_advance_on_save),
        }
        source_subject_mask_run = str(component.get("source_subject_mask_run") or "").strip()
        if source_subject_mask_run:
            scope["subject_run"] = source_subject_mask_run
        task = store.upsert_task(
            task_id=_safe_task_id("subject_mask_component", dataset_recording_id, dataset_id, run_name, component_name),
            recording_id=dataset_recording_id,
            workflow_kind="subject_mask_component",
            dataset_id=dataset_id,
            zarr_use="training",
            stage_group="refined_subject_masks_runs",
            run_name=run_name,
            component_name=component_name,
            title=f"Review {component_name} mask: {dataset_id}",
            scope=scope,
            state="pending",
            priority=int(priority),
            notes=f"Generated from registry {registry_path}",
        )
        generated.append({"task": task, "component": component, "approved": approved})
    return {
        "registry_path": str(registry_path),
        "assignment_count": len(assignments),
        "component_count": len(components),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }

def _task_generation_cli_payload(payload: Mapping[str, object], *, warnings_as_errors: bool = False) -> dict[str, object]:
    skipped = payload.get("skipped") if isinstance(payload.get("skipped"), list) else []
    warnings: list[dict[str, object]] = []
    for row in skipped:
        if not isinstance(row, Mapping):
            continue
        reason = str(row.get("reason") or "skipped")
        warning = {
            "code": f"generation_skipped_{reason}",
            "reason": reason,
            "dataset_id": row.get("dataset_id"),
            "recording_id": row.get("recording_id"),
            "component_name": row.get("component_name"),
            "details": "Registry row was skipped during task generation.",
        }
        warnings.append({key: value for key, value in warning.items() if value is not None})
    warning_codes = sorted(
        {
            str(warning.get("code") or "")
            for warning in warnings
            if str(warning.get("code") or "")
        }
    )
    failed_by_warnings = bool(warnings) and bool(warnings_as_errors)
    return {
        "ok": not failed_by_warnings,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
        "warning_count": len(warnings),
        "warning_codes": warning_codes,
        "warnings_as_errors": bool(warnings_as_errors),
        "failed_by_warnings": failed_by_warnings,
        "blocking_warning_count": len(warnings) if failed_by_warnings else 0,
        "blocking_warning_codes": warning_codes if failed_by_warnings else [],
        "warnings": warnings,
    }
