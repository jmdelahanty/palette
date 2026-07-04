#!/usr/bin/env python3
"""Plan or apply a sparse-first refined-detect migration for one archive.

Default mode is dry-run. Use ``--apply`` to materialize a successor refined
run from raw detect rows plus detect_quality labels, then promote it to latest
unless explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.refinement.refine_detect import (
    _build_sparse_refined_inputs_from_filtered,
    _get_sampled_frame_count,
    _read_sampled_import_meta,
    _resolve_detection_quality_labels,
    filter_detections,
)
from fisheye.shared.refined_detect_curation import (
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    has_curated_refined_detect_surface,
    has_curated_refined_source_detections_projection,
    has_sparse_curated_refined_detect_instances_arrays,
    write_curated_refined_detect_surfaces,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.metadata import get_total_frames


REFINED_PARENT_NAMES = ("refined_detect_runs", "refined_runs")
_RUN_NAME_NULLS = {"", "n/a", "na", "none", "null", "unknown"}
_REVIEW_PREFERENCE_CHAIN = ["refined", "manual", "interpolated", "filtered", "raw"]


@dataclass
class SparseMigrationPlan:
    source_refined_run_name: str
    output_refined_run_name: str
    parent_latest_refined_run_name: Optional[str]
    source_is_parent_latest: bool
    source_detect_run: str
    source_quality_run: Optional[str]
    total_frames: int
    coverage_frame_source: str
    coverage_frames_full: Optional[int]
    sampled_import: bool
    sampled_import_meta: Dict[str, Any]
    legacy_sparse_groups: list[str]
    legacy_group_policy: str
    current_summary: Dict[str, Any]
    planned_summary: Dict[str, Any]
    coverage_comparison: Dict[str, Any]
    write_payload: Dict[str, np.ndarray]


class SparseMigrationConflictError(RuntimeError):
    """Raised when migration requires explicit operator acknowledgement."""

    def __init__(
        self,
        message: str,
        *,
        legacy_sparse_groups: Optional[Sequence[str]] = None,
        output_refined_run_name: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.legacy_sparse_groups = [str(name) for name in (legacy_sparse_groups or [])]
        self.output_refined_run_name = output_refined_run_name


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode=mode)


def _normalize_text(value: Any) -> Optional[str]:
    text = normalize_attr(value)
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in _RUN_NAME_NULLS:
        return None
    return text


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pick_refined_parent(root: zarr.Group) -> zarr.Group:
    for parent_name in REFINED_PARENT_NAMES:
        if parent_name in root:
            return root[parent_name]
    raise RuntimeError("No refined_detect_runs found in archive.")


def _open_child_group(parent: zarr.Group, name: str, *, mode: str) -> zarr.Group:
    if not (hasattr(parent, "store_path") and hasattr(parent, "path")):
        existing = parent.get(name)
        if existing is not None:
            return existing
        if mode == "a":
            return parent.require_group(name)
        raise KeyError(name)
    child_path = f"{parent.path}/{name}" if getattr(parent, "path", "") else name
    try:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
            use_consolidated=False,
        )
    except TypeError:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
        )


def _select_refined_run(parent: zarr.Group, requested: Optional[str]) -> str:
    normalized_requested = _normalize_text(requested)
    if normalized_requested:
        if normalized_requested in parent:
            return normalized_requested
        raise RuntimeError(f"Refined run '{normalized_requested}' not found.")
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        raise RuntimeError("No refined runs available.")
    return sorted(str(name) for name in names)[-1]


def _default_output_run_name(parent: zarr.Group, *, source_run_name: str) -> str:
    base = f"{source_run_name}_sparse"
    candidate = base
    suffix = 2
    while candidate in parent:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _promotion_blocked_reason(
    plan: SparseMigrationPlan,
    *,
    promote_latest_requested: bool,
    force_promote_nonlatest: bool,
) -> Optional[str]:
    if not promote_latest_requested:
        return None
    if plan.source_is_parent_latest or force_promote_nonlatest:
        return None
    parent_latest = plan.parent_latest_refined_run_name or "unset"
    return (
        f"Source refined run '{plan.source_refined_run_name}' is not the current parent latest "
        f"('{parent_latest}'). Pass --no-promote-latest to materialize the successor run without "
        "promotion, or --force-promote-nonlatest to override."
    )


def _resolve_detect_group(
    root: zarr.Group,
    refined_run: zarr.Group,
    *,
    detect_run: Optional[str],
) -> tuple[str, zarr.Group]:
    detect_parent = root.get("detect_runs")
    if detect_parent is None:
        raise RuntimeError("Archive is missing detect_runs.")
    resolved_detect_run = _normalize_text(detect_run)
    if resolved_detect_run is None:
        resolved_detect_run = _normalize_text(refined_run.attrs.get("source_detect_run"))
    if resolved_detect_run is None:
        resolved_detect_run = _normalize_text(detect_parent.attrs.get("latest"))
    if resolved_detect_run is None or resolved_detect_run not in detect_parent:
        raise RuntimeError("Could not resolve a valid source detect run for migration.")
    return resolved_detect_run, detect_parent[resolved_detect_run]


def _raw_scores(detect_group: zarr.Group, row_count: int) -> np.ndarray:
    if "scores" in detect_group:
        return np.asarray(detect_group["scores"][:], dtype=np.float32).reshape(-1)
    return np.ones(int(row_count), dtype=np.float32)


def _raw_class_ids(detect_group: zarr.Group, row_count: int) -> np.ndarray:
    if "class_ids" in detect_group:
        return np.asarray(detect_group["class_ids"][:], dtype=np.int32).reshape(-1)
    return np.zeros(int(row_count), dtype=np.int32)


def _legacy_sparse_groups(refined_run: zarr.Group) -> list[str]:
    try:
        group_names = list(refined_run.group_keys())
    except Exception:
        group_names = list(refined_run.keys())
    legacy: list[str] = []
    for name in sorted(str(group_name) for group_name in group_names):
        if name in {"instances", "source_detections"}:
            continue
        if name in {"filtered", "interpolated", "manual"} or name.startswith("manual"):
            legacy.append(name)
    return legacy


def _current_summary(refined_run: zarr.Group) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "has_curated_surface": bool(has_curated_refined_detect_surface(refined_run)),
        "has_sparse_instances": bool(has_sparse_curated_refined_detect_instances_arrays(refined_run)),
        "has_source_detections": bool(has_curated_refined_source_detections_projection(refined_run)),
        "legacy_sparse_groups": _legacy_sparse_groups(refined_run),
        "review_status": None,
        "instances": {
            "total_instances": 0,
            "frame_count": 0,
            "manual_edited_instances": 0,
        },
        "source_detection_decision_counts": {},
    }

    review_status = refined_run.attrs.get("detect_review_status")
    if isinstance(review_status, Mapping):
        summary["review_status"] = _normalize_json(dict(review_status))

    if has_curated_refined_detect_surface(refined_run):
        payload = extract_present_curated_rows(refined_run)
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
        manual_edit_flags = np.asarray(
            payload.get("manual_edit_flags", np.zeros(frame_indices.shape[0], dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        summary["instances"] = {
            "total_instances": int(frame_indices.shape[0]),
            "frame_count": int(np.unique(frame_indices).shape[0]) if frame_indices.size else 0,
            "manual_edited_instances": int(np.count_nonzero(manual_edit_flags)),
        }

    if has_curated_refined_source_detections_projection(refined_run):
        summary["source_detection_decision_counts"] = _normalize_json(
            build_source_detection_decision_summary(refined_run)
        )

    return summary


def _coverage_summary(
    *,
    raw_frame_indices: np.ndarray,
    instance_frame_indices: np.ndarray,
    total_frames: int,
) -> Dict[str, Any]:
    raw_counts = np.bincount(raw_frame_indices, minlength=total_frames).astype(np.int32, copy=False)
    instance_counts = np.bincount(instance_frame_indices, minlength=total_frames).astype(np.int32, copy=False)
    original_coverage = (float(np.sum(raw_counts > 0)) / float(total_frames) * 100.0) if total_frames > 0 else 0.0
    refined_coverage = (
        float(np.sum(instance_counts > 0)) / float(total_frames) * 100.0
        if total_frames > 0
        else 0.0
    )
    return {
        "original": {
            "total_detections": int(raw_frame_indices.shape[0]),
            "frames_with_detections": int(np.sum(raw_counts > 0)),
            "coverage_percent": float(original_coverage),
        },
        "refined": {
            "total_detections": int(instance_frame_indices.shape[0]),
            "frames_with_detections": int(np.sum(instance_counts > 0)),
            "coverage_percent": float(refined_coverage),
            "detections_removed": int(raw_frame_indices.shape[0] - instance_frame_indices.shape[0]),
            "coverage_delta": float(refined_coverage - original_coverage),
        },
    }


def _planned_summary(
    *,
    source_refined_run_name: str,
    output_refined_run_name: str,
    parent_latest_refined_run_name: Optional[str],
    source_is_parent_latest: bool,
    source_detect_run: str,
    source_quality_run: Optional[str],
    total_frames: int,
    sampled_import: bool,
    legacy_group_policy: str,
    ignored_legacy_groups: Sequence[str],
    sparse_payload: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    instance_frame_indices = np.asarray(sparse_payload["instance_frame_indices"], dtype=np.int32).reshape(-1)
    frame_counts = np.bincount(instance_frame_indices, minlength=total_frames).astype(np.int32, copy=False)
    source_decisions = Counter(
        str(label)
        for label in np.asarray(sparse_payload["source_detection_decision_labels"], dtype=object).reshape(-1).tolist()
    )
    return {
        "source_refined_run": source_refined_run_name,
        "output_refined_run": output_refined_run_name,
        "parent_latest_refined_run": parent_latest_refined_run_name,
        "source_is_parent_latest": bool(source_is_parent_latest),
        "source_detect_run": source_detect_run,
        "source_quality_run": source_quality_run,
        "sampled_import": bool(sampled_import),
        "selection_policy": (
            "sampled_import_sparse_instances_passthrough"
            if sampled_import
            else "recompute_from_raw_detect_quality_sparse_instances"
        ),
        "total_instances": int(instance_frame_indices.shape[0]),
        "frame_count": int(np.unique(instance_frame_indices).shape[0]) if instance_frame_indices.size else 0,
        "multi_instance_frames": int(np.count_nonzero(frame_counts > 1)),
        "max_instances_per_frame": int(frame_counts.max()) if frame_counts.size else 0,
        "total_source_detections": int(
            np.asarray(sparse_payload["source_detection_frame_indices"], dtype=np.int32).reshape(-1).shape[0]
        ),
        "source_detection_decision_counts": dict(source_decisions),
        "legacy_group_policy": legacy_group_policy,
        "ignored_legacy_groups": [str(name) for name in ignored_legacy_groups],
        "review_state_after_apply": "pending",
    }


def build_sparse_migration_plan(
    root: zarr.Group,
    *,
    refined_run_name: Optional[str] = None,
    output_run_name: Optional[str] = None,
    detect_run: Optional[str] = None,
    quality_run: Optional[str] = None,
    require_detect_quality: bool = True,
    ignore_legacy_groups: bool = False,
) -> SparseMigrationPlan:
    refined_parent = _pick_refined_parent(root)
    resolved_source_refined_run_name = _select_refined_run(refined_parent, refined_run_name)
    source_refined_run = _open_child_group(refined_parent, resolved_source_refined_run_name, mode="r")
    parent_latest_refined_run_name = _normalize_text(refined_parent.attrs.get("latest"))
    source_is_parent_latest = bool(
        parent_latest_refined_run_name and parent_latest_refined_run_name == resolved_source_refined_run_name
    )

    resolved_output_run_name = _normalize_text(output_run_name)
    if resolved_output_run_name is None:
        resolved_output_run_name = _default_output_run_name(
            refined_parent,
            source_run_name=resolved_source_refined_run_name,
        )
    if resolved_output_run_name == resolved_source_refined_run_name:
        raise SparseMigrationConflictError(
            "Output refined run must differ from the source refined run.",
            output_refined_run_name=resolved_output_run_name,
        )
    if resolved_output_run_name in refined_parent:
        raise SparseMigrationConflictError(
            f"Output refined run '{resolved_output_run_name}' already exists.",
            output_refined_run_name=resolved_output_run_name,
        )

    legacy_sparse_groups = _legacy_sparse_groups(source_refined_run)
    if legacy_sparse_groups and not ignore_legacy_groups:
        raise SparseMigrationConflictError(
            (
                f"Source refined run '{resolved_source_refined_run_name}' has legacy sparse groups "
                f"{legacy_sparse_groups}. Pass --ignore-legacy-groups to materialize a successor sparse run "
                "without replaying them."
            ),
            legacy_sparse_groups=legacy_sparse_groups,
            output_refined_run_name=resolved_output_run_name,
        )

    source_detect_run, detect_group = _resolve_detect_group(root, source_refined_run, detect_run=detect_run)

    raw_bbox = np.asarray(detect_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    raw_frame_indices = np.asarray(detect_group["frame_indices"][:], dtype=np.int32).reshape(-1)
    raw_scores = _raw_scores(detect_group, raw_bbox.shape[0])
    raw_class_ids = _raw_class_ids(detect_group, raw_bbox.shape[0])

    sampled_import, sampled_meta = _read_sampled_import_meta(root)
    require_quality_for_run = bool(require_detect_quality and not sampled_import)
    requested_quality_run = _normalize_text(quality_run) or _normalize_text(
        source_refined_run.attrs.get("source_quality_run")
    )
    quality_labels, resolved_quality_run, _quality_group = _resolve_detection_quality_labels(
        detect_group,
        detect_run=source_detect_run,
        quality_run=requested_quality_run,
        total_detections=int(raw_bbox.shape[0]),
        require_quality=require_quality_for_run,
        allow_missing_reason="explicit opt-out (--allow-missing-quality)",
        console=None,
    )
    if sampled_import:
        quality_labels = np.zeros_like(quality_labels)

    total_frames = get_total_frames(root, detect_group)
    full_frame_count = total_frames
    coverage_frame_source = "full"
    if sampled_import:
        sampled_frames = _get_sampled_frame_count(root, detect_group)
        if sampled_frames is not None:
            total_frames = sampled_frames
            coverage_frame_source = "sampled"
    if total_frames is None:
        total_frames = int(raw_frame_indices.max() + 1) if raw_frame_indices.size else 0

    filtered_bbox, filtered_scores, _filtered_counts, filtered_frame_indices, filtered_class_ids, _drop_stats = (
        filter_detections(
            raw_bbox,
            raw_scores,
            raw_frame_indices,
            raw_class_ids,
            quality_labels,
            int(total_frames),
            [],
        )
    )
    sparse_payload = _build_sparse_refined_inputs_from_filtered(
        raw_bboxes=raw_bbox,
        raw_scores=raw_scores,
        raw_frame_indices=raw_frame_indices,
        raw_class_ids=raw_class_ids,
        detection_quality_labels=quality_labels,
        interp_bboxes=filtered_bbox,
        interp_scores=filtered_scores,
        interp_frame_indices=filtered_frame_indices,
        interp_class_ids=filtered_class_ids,
    )

    return SparseMigrationPlan(
        source_refined_run_name=resolved_source_refined_run_name,
        output_refined_run_name=resolved_output_run_name,
        parent_latest_refined_run_name=parent_latest_refined_run_name,
        source_is_parent_latest=source_is_parent_latest,
        source_detect_run=source_detect_run,
        source_quality_run=resolved_quality_run,
        total_frames=int(total_frames),
        coverage_frame_source=coverage_frame_source,
        coverage_frames_full=int(full_frame_count) if full_frame_count is not None else None,
        sampled_import=bool(sampled_import),
        sampled_import_meta=_normalize_json(sampled_meta),
        legacy_sparse_groups=list(legacy_sparse_groups),
        legacy_group_policy="ignored_by_operator" if legacy_sparse_groups else "none_present",
        current_summary=_current_summary(source_refined_run),
        planned_summary=_planned_summary(
            source_refined_run_name=resolved_source_refined_run_name,
            output_refined_run_name=resolved_output_run_name,
            parent_latest_refined_run_name=parent_latest_refined_run_name,
            source_is_parent_latest=source_is_parent_latest,
            source_detect_run=source_detect_run,
            source_quality_run=resolved_quality_run,
            total_frames=int(total_frames),
            sampled_import=bool(sampled_import),
            legacy_group_policy="ignored_by_operator" if legacy_sparse_groups else "none_present",
            ignored_legacy_groups=legacy_sparse_groups,
            sparse_payload=sparse_payload,
        ),
        coverage_comparison=_coverage_summary(
            raw_frame_indices=raw_frame_indices,
            instance_frame_indices=np.asarray(sparse_payload["instance_frame_indices"], dtype=np.int32).reshape(-1),
            total_frames=int(total_frames),
        ),
        write_payload={key: np.asarray(value) for key, value in sparse_payload.items()},
    )


def _review_status_for_migration(
    source_refined_run: zarr.Group,
    *,
    source_refined_run_name: str,
    output_refined_run_name: str,
    source_detect_run: str,
    source_quality_run: Optional[str],
    legacy_sparse_groups: Sequence[str],
) -> Dict[str, Any]:
    existing = source_refined_run.attrs.get("detect_review_status")
    intended_use = "training"
    if isinstance(existing, Mapping):
        existing_use = _normalize_text(existing.get("intended_use"))
        if existing_use:
            intended_use = existing_use
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    notes = [
        (
            "Sparse successor migration materialized a new canonical refined detect run "
            "from raw detect + detect_quality."
        ),
        f"Source refined run '{source_refined_run_name}' was preserved.",
    ]
    if legacy_sparse_groups:
        notes.append(
            "Legacy manual/interpolated subgroup edits were not replayed into the successor run."
        )
    return {
        "state": "pending",
        "method": "algorithmic",
        "intended_use": intended_use,
        "timestamp_utc": timestamp_utc,
        "timestamp": timestamp_utc,
        "resolved_group": "refined",
        "target_group": "refined",
        "preference_chain": list(_REVIEW_PREFERENCE_CHAIN),
        "notes": " ".join(notes),
        "migration_source_refined_run": source_refined_run_name,
        "migration_output_refined_run": output_refined_run_name,
        "migration_source_detect_run": source_detect_run,
        "migration_source_quality_run": source_quality_run or "N/A",
    }


def _migration_metadata(
    source_refined_run: zarr.Group,
    *,
    plan: SparseMigrationPlan,
    promoted_to_latest: bool,
) -> Dict[str, Any]:
    previous_review_status = source_refined_run.attrs.get("detect_review_status")
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "recompute_from_raw_detect_quality",
        "migrated_from_refined_run": plan.source_refined_run_name,
        "output_refined_run": plan.output_refined_run_name,
        "parent_latest_refined_run": plan.parent_latest_refined_run_name,
        "source_is_parent_latest": bool(plan.source_is_parent_latest),
        "legacy_group_policy": plan.legacy_group_policy,
        "ignored_legacy_groups": list(plan.legacy_sparse_groups),
        "source_detect_run": plan.source_detect_run,
        "source_quality_run": plan.source_quality_run or "N/A",
        "sampled_import": bool(plan.sampled_import),
        "instances_after": int(plan.planned_summary.get("total_instances", 0)),
        "source_detections_after": int(plan.planned_summary.get("total_source_detections", 0)),
        "review_status_reset_to": "pending",
        "promoted_to_latest": bool(promoted_to_latest),
    }
    if isinstance(previous_review_status, Mapping):
        payload["previous_review_status"] = _normalize_json(dict(previous_review_status))
    return payload


def apply_sparse_migration(
    root: zarr.Group,
    *,
    zarr_path: Path,
    plan: SparseMigrationPlan,
    command: Optional[str] = None,
    promote_latest: bool = True,
    force_promote_nonlatest: bool = False,
) -> Dict[str, Any]:
    blocked_reason = _promotion_blocked_reason(
        plan,
        promote_latest_requested=bool(promote_latest),
        force_promote_nonlatest=bool(force_promote_nonlatest),
    )
    if blocked_reason:
        raise SparseMigrationConflictError(
            blocked_reason,
            output_refined_run_name=plan.output_refined_run_name,
        )
    refined_parent = _pick_refined_parent(root)
    if plan.output_refined_run_name in refined_parent:
        raise SparseMigrationConflictError(
            f"Output refined run '{plan.output_refined_run_name}' already exists.",
            output_refined_run_name=plan.output_refined_run_name,
        )
    source_refined_run = _open_child_group(refined_parent, plan.source_refined_run_name, mode="r")
    _open_child_group(refined_parent, plan.output_refined_run_name, mode="a")
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name=plan.output_refined_run_name,
        instance_frame_indices=np.asarray(plan.write_payload["instance_frame_indices"], dtype=np.int32),
        instance_bbox_norm_coords=np.asarray(plan.write_payload["instance_bbox_norm_coords"], dtype=np.float64),
        instance_source_kind_labels=np.asarray(plan.write_payload["instance_source_kind_labels"], dtype=object),
        instance_reason_labels=np.asarray(plan.write_payload["instance_reason_labels"], dtype=object),
        instance_source_detect_row_index=np.asarray(
            plan.write_payload["instance_source_detect_row_index"], dtype=np.int32
        ),
        instance_manual_edit_flags=np.asarray(plan.write_payload["instance_manual_edit_flags"], dtype=bool),
        instance_confidence_scores=np.asarray(plan.write_payload["instance_confidence_scores"], dtype=np.float32),
        instance_class_ids=np.asarray(plan.write_payload["instance_class_ids"], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray(
            plan.write_payload["source_detection_source_detect_row_index"], dtype=np.int32
        ),
        source_detection_frame_indices=np.asarray(
            plan.write_payload["source_detection_frame_indices"], dtype=np.int32
        ),
        source_detection_bbox_norm_coords=np.asarray(
            plan.write_payload["source_detection_bbox_norm_coords"], dtype=np.float64
        ),
        source_detection_decision_labels=np.asarray(
            plan.write_payload["source_detection_decision_labels"], dtype=object
        ),
        source_detection_reason_labels=np.asarray(
            plan.write_payload["source_detection_reason_labels"], dtype=object
        ),
        source_detection_confidence_scores=np.asarray(
            plan.write_payload["source_detection_confidence_scores"], dtype=np.float32
        ),
        source_detection_class_ids=np.asarray(plan.write_payload["source_detection_class_ids"], dtype=np.int32),
        command=command,
        source_context={
            "source_detect_run": plan.source_detect_run,
            "quality_run": plan.source_quality_run or "N/A",
            "selection_policy": plan.planned_summary["selection_policy"],
            "migration_mode": "recompute_from_raw_detect_quality",
        },
    )

    # Re-resolve the group after the sparse writer runs. On real Zarr stores,
    # mutating attrs through a stale group handle can clobber metadata that the
    # writer just persisted.
    refined_parent = _pick_refined_parent(root)
    output_refined_run = _open_child_group(refined_parent, plan.output_refined_run_name, mode="a")

    output_refined_run.attrs["source_detect_run"] = plan.source_detect_run
    output_refined_run.attrs["source_quality_run"] = plan.source_quality_run or "N/A"
    output_refined_run.attrs["migrated_from_refined_run"] = plan.source_refined_run_name
    output_refined_run.attrs["coverage_comparison"] = _normalize_json(plan.coverage_comparison)
    output_refined_run.attrs["coverage_frames_total"] = int(plan.total_frames)
    output_refined_run.attrs["coverage_frame_source"] = plan.coverage_frame_source
    if plan.coverage_frames_full is not None:
        output_refined_run.attrs["coverage_frames_full"] = int(plan.coverage_frames_full)
    output_refined_run.attrs["sparse_migration"] = _migration_metadata(
        source_refined_run,
        plan=plan,
        promoted_to_latest=bool(promote_latest),
    )

    review_status = _review_status_for_migration(
        source_refined_run,
        source_refined_run_name=plan.source_refined_run_name,
        output_refined_run_name=plan.output_refined_run_name,
        source_detect_run=plan.source_detect_run,
        source_quality_run=plan.source_quality_run,
        legacy_sparse_groups=plan.legacy_sparse_groups,
    )
    output_refined_run.attrs["detect_review_status"] = review_status
    if promote_latest:
        refined_parent.attrs["latest"] = plan.output_refined_run_name
        refined_parent.attrs["detect_review_status_latest"] = plan.output_refined_run_name

    return {
        "applied": True,
        "source_refined_run": plan.source_refined_run_name,
        "output_refined_run": plan.output_refined_run_name,
        "promoted_to_latest": bool(promote_latest),
        "force_promote_nonlatest": bool(force_promote_nonlatest),
        "review_status": _normalize_json(review_status),
    }


def _plan_payload(
    plan: SparseMigrationPlan,
    *,
    zarr_path: Path,
    mode: str,
    promote_latest_requested: bool,
    force_promote_nonlatest: bool,
) -> Dict[str, Any]:
    promotion_blocked_reason = _promotion_blocked_reason(
        plan,
        promote_latest_requested=bool(promote_latest_requested),
        force_promote_nonlatest=bool(force_promote_nonlatest),
    )
    return {
        "zarr_path": str(zarr_path),
        "mode": mode,
        "source_refined_run": plan.source_refined_run_name,
        "output_refined_run": plan.output_refined_run_name,
        "parent_latest_refined_run": plan.parent_latest_refined_run_name,
        "source_is_parent_latest": bool(plan.source_is_parent_latest),
        "source_detect_run": plan.source_detect_run,
        "source_quality_run": plan.source_quality_run,
        "sampled_import": bool(plan.sampled_import),
        "sampled_import_meta": _normalize_json(plan.sampled_import_meta),
        "promotion_requested": bool(promote_latest_requested),
        "force_promote_nonlatest": bool(force_promote_nonlatest),
        "promotion_allowed": bool(promote_latest_requested and promotion_blocked_reason is None),
        "promotion_blocked_reason": promotion_blocked_reason,
        "current": _normalize_json(plan.current_summary),
        "planned": _normalize_json(plan.planned_summary),
        "coverage_comparison": _normalize_json(plan.coverage_comparison),
    }


def _emit(payload: Dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Archive: {payload['zarr_path']}")
    print(f"Mode: {payload['mode']}")
    print(f"Source refined run: {payload['source_refined_run']}")
    print(f"Output refined run: {payload['output_refined_run']}")
    print(f"Parent latest refined run: {payload.get('parent_latest_refined_run') or '-'}")
    print(f"Source is parent latest: {payload.get('source_is_parent_latest')}")
    print(f"Source detect run: {payload['source_detect_run']}")
    print(f"Source quality run: {payload.get('source_quality_run') or '-'}")
    print(f"Sampled import: {payload['sampled_import']}")
    print(f"Promotion requested: {payload.get('promotion_requested')}")
    print(f"Promotion allowed: {payload.get('promotion_allowed')}")
    if payload.get("promotion_blocked_reason"):
        print(f"Promotion blocked reason: {payload['promotion_blocked_reason']}")
    print()

    current = payload.get("current", {})
    print("Current:")
    print(f"- has_curated_surface: {current.get('has_curated_surface')}")
    print(f"- has_sparse_instances: {current.get('has_sparse_instances')}")
    print(f"- has_source_detections: {current.get('has_source_detections')}")
    print(f"- legacy_sparse_groups: {current.get('legacy_sparse_groups') or []}")
    print(f"- review_status: {json.dumps(current.get('review_status') or {}, sort_keys=True)}")
    instances = current.get("instances", {})
    print(f"- current_instances: {instances.get('total_instances', 0)}")
    print()

    planned = payload.get("planned", {})
    print("Planned:")
    print(f"- selection_policy: {planned.get('selection_policy')}")
    print(f"- total_instances: {planned.get('total_instances', 0)}")
    print(f"- frame_count: {planned.get('frame_count', 0)}")
    print(f"- multi_instance_frames: {planned.get('multi_instance_frames', 0)}")
    print(f"- max_instances_per_frame: {planned.get('max_instances_per_frame', 0)}")
    print(f"- total_source_detections: {planned.get('total_source_detections', 0)}")
    print(
        f"- source_detection_decision_counts: "
        f"{json.dumps(planned.get('source_detection_decision_counts') or {}, sort_keys=True)}"
    )
    print(f"- legacy_group_policy: {planned.get('legacy_group_policy')}")
    print(f"- ignored_legacy_groups: {planned.get('ignored_legacy_groups') or []}")
    print(f"- review_state_after_apply: {planned.get('review_state_after_apply')}")
    if "applied" in payload:
        print()
        print(f"Applied: {payload['applied']}")
        print(f"Promoted latest: {payload.get('promoted_to_latest')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--refined-run", help="Source refined detect run to migrate (default: latest).")
    parser.add_argument(
        "--output-run",
        help="Name for the successor sparse refined run (default: <source>_sparse with collision suffixes).",
    )
    parser.add_argument("--detect-run", help="Override source detect run (default: refined run attr/latest detect).")
    parser.add_argument("--quality-run", help="Override source quality run (default: refined run attr/latest quality).")
    parser.add_argument(
        "--allow-missing-quality",
        action="store_true",
        help="Allow migration without a usable detect_quality report.",
    )
    parser.add_argument(
        "--ignore-legacy-groups",
        action="store_true",
        help="Allow migration to proceed when the source refined run has legacy manual/interpolated groups.",
    )
    parser.add_argument(
        "--no-promote-latest",
        action="store_true",
        help="Do not update refined_detect_runs.attrs['latest'] or detect_review_status_latest.",
    )
    parser.add_argument(
        "--force-promote-nonlatest",
        action="store_true",
        help="Allow promotion even when the source refined run is not the current parent latest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write migrated sparse surfaces into a new successor refined run.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        mode = "a" if args.apply else "r"
        zarr_path = args.zarr_path.expanduser().resolve()
        root = _open_root(zarr_path, mode=mode)
        promote_latest_requested = not bool(args.no_promote_latest)
        plan = build_sparse_migration_plan(
            root,
            refined_run_name=args.refined_run,
            output_run_name=args.output_run,
            detect_run=args.detect_run,
            quality_run=args.quality_run,
            require_detect_quality=not bool(args.allow_missing_quality),
            ignore_legacy_groups=bool(args.ignore_legacy_groups),
        )
        payload = _plan_payload(
            plan,
            zarr_path=zarr_path,
            mode="apply" if args.apply else "dry-run",
            promote_latest_requested=promote_latest_requested,
            force_promote_nonlatest=bool(args.force_promote_nonlatest),
        )
        if args.apply:
            payload.update(
                apply_sparse_migration(
                    root,
                    zarr_path=zarr_path,
                    plan=plan,
                    command=" ".join(sys.argv),
                    promote_latest=promote_latest_requested,
                    force_promote_nonlatest=bool(args.force_promote_nonlatest),
                )
            )
        _emit(payload, as_json=bool(args.json))
        return 0
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
