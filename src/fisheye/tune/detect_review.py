"""
Manual review tool for detection runs.

Current refined detect review edits the canonical sparse
refined_detect_runs/<run>/instances surface through a single-instance-per-frame
compatibility view. Legacy subgroup reads remain fallback-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from rich.console import Console
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, erosion

from fisheye.detection.detect_traditional import create_dish_mask, get_detection_parameters
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.refined_detect_curation import (
    REFINED_DETECT_STATUS_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
    build_curated_detection_source_array,
    has_curated_refined_detect_arrays,
    has_curated_refined_source_detections_projection,
    has_sparse_curated_refined_detect_instances_arrays,
    materialize_refined_detect_curation,
    resolve_bound_source_detect_group,
    resolve_curated_instance_keys,
    update_curated_refined_detect_rows,
    write_curated_refined_detect_surfaces,
)
from fisheye.shared.refined_detect_resolution import resolve_detect_review_target
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name, set_authoritative_run
from fisheye.utils.accept_detect_review import (
    _pick_refined_parent_name,
    _write_profile_and_sync_registry,
)
from fisheye.shared.system_metadata import get_environment_info


def _get_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")
    latest = resolve_latest_complete_run_name(refined_parent)
    if not latest:
        raise RuntimeError("No refined detect runs recorded.")
    return latest


def _pick_variant(refined_run: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in refined_run:
            return requested
        if requested == "refined" and (
            has_sparse_curated_refined_detect_instances_arrays(refined_run)
            or has_curated_refined_detect_arrays(refined_run)
        ):
            return requested
        raise RuntimeError(f"Requested variant '{requested}' not found in refined run.")
    if has_sparse_curated_refined_detect_instances_arrays(refined_run) or has_curated_refined_detect_arrays(refined_run):
        return "refined"
    if "interpolated" in refined_run:
        return "interpolated"
    if "filtered" in refined_run:
        return "filtered"
    raise RuntimeError("Refined run has no canonical curated surface or legacy variants to review.")


_STATUS_LABEL_BY_CODE = {
    value: key for key, value in REFINED_DETECT_STATUS_CODE_MAP.items()
}
_SOURCE_KIND_LABEL_BY_CODE = {
    value: key for key, value in REFINED_SOURCE_KIND_CODE_MAP.items()
}


def _normalize_arena_definitions(raw_masks: object) -> List[Dict[str, Any]]:
    if not isinstance(raw_masks, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw in raw_masks:
        if not isinstance(raw, Mapping):
            continue
        raw_id = raw.get("id")
        roi = raw.get("roi_pixels")
        if raw_id is None or not isinstance(roi, (list, tuple)) or len(roi) != 4:
            continue
        try:
            arena_id = int(raw_id)
            x, y, w, h = [int(value) for value in roi]
        except Exception:
            continue
        if arena_id in seen_ids:
            continue
        seen_ids.add(arena_id)
        out.append(
            {
                "id": arena_id,
                "roi_pixels": [x, y, w, h],
                "source": raw.get("source"),
            }
        )
    out.sort(key=lambda item: int(item["id"]))
    return out


def _resolve_review_arena_definitions(root: zarr.Group) -> List[Dict[str, Any]]:
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is not None:
        subdish = analysis_meta.attrs.get("subdish_mask_tuning")
        if isinstance(subdish, Mapping):
            masks = _normalize_arena_definitions(subdish.get("masks"))
            if masks:
                return masks

    arena_parent = root.get("arena_assignment_runs")
    if arena_parent is not None:
        latest = resolve_latest_complete_run_name(arena_parent)
        if latest and latest in arena_parent:
            arena_group = arena_parent[latest]
            masks = _normalize_arena_definitions(arena_group.attrs.get("arena_definitions"))
            if masks:
                return masks

    return []


def _bbox_norm_to_arena_id(
    bbox_norm: np.ndarray,
    *,
    arena_definitions: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
) -> int:
    bbox = np.asarray(bbox_norm, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        return -1
    center_x = float(bbox[0]) * float(width)
    center_y = float(bbox[1]) * float(height)
    for arena in arena_definitions:
        arena_id = int(arena["id"])
        x, y, w, h = [int(value) for value in arena["roi_pixels"]]
        if x <= center_x < (x + w) and y <= center_y < (y + h):
            return arena_id
    return -1


def _pick_slot_source_surface_row(
    row_indices: np.ndarray,
    *,
    preferred_source_detect_row_index: int,
    source_surface_source_detect_row_index: np.ndarray,
    source_surface_decision_labels: np.ndarray,
    source_surface_confidence_scores: np.ndarray,
) -> Optional[int]:
    rows = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    if rows.size == 0:
        return None
    if preferred_source_detect_row_index >= 0:
        for row_idx in rows.tolist():
            if int(source_surface_source_detect_row_index[row_idx]) == int(preferred_source_detect_row_index):
                return int(row_idx)

    for preferred_decision in ("accepted", "manual_clear"):
        matching = [
            int(row_idx)
            for row_idx in rows.tolist()
            if str(source_surface_decision_labels[int(row_idx)]) == preferred_decision
        ]
        if matching:
            return matching[0]

    confidence_rows = sorted(
        rows.tolist(),
        key=lambda row_idx: (
            float(source_surface_confidence_scores[int(row_idx)]),
            -int(source_surface_source_detect_row_index[int(row_idx)]),
        ),
        reverse=True,
    )
    return int(confidence_rows[0]) if confidence_rows else None


def _resolve_source_surface_row_for_slot(
    payload: Dict[str, np.ndarray],
    *,
    frame: int,
    arena_id: int,
    preferred_source_detect_row_index: int,
) -> Optional[int]:
    lookup = payload.get("source_row_lookup")
    if isinstance(lookup, dict) and preferred_source_detect_row_index >= 0:
        row_idx = lookup.get(int(preferred_source_detect_row_index))
        if row_idx is not None:
            return int(row_idx)

    rows_by_slot = payload.get("source_rows_by_slot")
    if not isinstance(rows_by_slot, dict):
        return None
    row_indices = rows_by_slot.get((int(frame), int(arena_id)))
    if row_indices is None:
        return None
    return _pick_slot_source_surface_row(
        np.asarray(row_indices, dtype=np.int32),
        preferred_source_detect_row_index=preferred_source_detect_row_index,
        source_surface_source_detect_row_index=np.asarray(
            payload["source_surface_source_detect_row_index"],
            dtype=np.int32,
        ).reshape(-1),
        source_surface_decision_labels=np.asarray(
            payload["source_surface_decision_labels"],
            dtype=object,
        ).reshape(-1),
        source_surface_confidence_scores=np.asarray(
            payload["source_surface_confidence_scores"],
            dtype=np.float32,
        ).reshape(-1),
    )


def _load_arena_slot_curated_edit_payload(
    refined_run: zarr.Group,
    *,
    arena_definitions: Sequence[Mapping[str, Any]],
    total_frames: int,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    if not has_sparse_curated_refined_detect_instances_arrays(refined_run):
        raise RuntimeError(
            "Arena-aware detect review currently requires sparse refined detect instances."
        )

    arena_defs = [dict(arena) for arena in arena_definitions]
    if not arena_defs:
        raise RuntimeError("Arena-aware detect review requires at least one arena definition.")

    ordered_arena_ids = np.asarray([int(arena["id"]) for arena in arena_defs], dtype=np.int32)
    slot_frame_indices = np.repeat(np.arange(int(total_frames), dtype=np.int32), ordered_arena_ids.shape[0])
    slot_arena_ids = np.tile(ordered_arena_ids, int(total_frames))
    slot_count = int(slot_frame_indices.shape[0])
    slot_to_row = {
        (int(frame), int(arena_id)): idx
        for idx, (frame, arena_id) in enumerate(zip(slot_frame_indices.tolist(), slot_arena_ids.tolist()))
    }

    bbox_norm = np.full((slot_count, 4), np.nan, dtype=np.float64)
    confidence_scores = np.full(slot_count, np.nan, dtype=np.float32)
    class_ids = np.full(slot_count, -1, dtype=np.int32)
    status_labels = np.full(slot_count, "missing", dtype=object)
    source_kind_labels = np.full(slot_count, "none", dtype=object)
    manual_edit_flags = np.zeros(slot_count, dtype=bool)
    reason_labels = np.full(slot_count, "missing_detection", dtype=object)
    source_detect_row_index = np.full(slot_count, -1, dtype=np.int32)
    detection_source = np.zeros(slot_count, dtype=np.int8)
    refined_row_ids = np.full(slot_count, -1, dtype=np.int64)

    source_surface_source_detect_row_index = np.empty((0,), dtype=np.int32)
    source_surface_frame_indices = np.empty((0,), dtype=np.int32)
    source_surface_bbox_norm_coords = np.empty((0, 4), dtype=np.float64)
    source_surface_decision_labels = np.empty((0,), dtype=object)
    source_surface_reason_labels = np.empty((0,), dtype=object)
    source_surface_confidence_scores = np.empty((0,), dtype=np.float32)
    source_surface_class_ids = np.empty((0,), dtype=np.int32)
    source_surface_review_notes = np.empty((0,), dtype=object)
    source_row_lookup: Dict[int, int] = {}
    source_rows_by_slot: Dict[Tuple[int, int], np.ndarray] = {}

    if has_curated_refined_source_detections_projection(refined_run):
        source_group = refined_run["source_detections"]
        source_surface_source_detect_row_index = np.asarray(
            source_group["source_detect_row_index"][:],
            dtype=np.int32,
        ).reshape(-1)
        source_surface_frame_indices = np.asarray(
            source_group["frame_indices"][:],
            dtype=np.int32,
        ).reshape(-1)
        source_surface_bbox_norm_coords = np.asarray(
            source_group["bbox_norm_coords"][:],
            dtype=np.float64,
        ).reshape(-1, 4)
        source_surface_decision_codes = np.asarray(
            source_group["decision_codes"][:],
            dtype=np.int8,
        ).reshape(-1)
        source_surface_decision_labels = np.asarray(
            [
                {
                    0: "accepted",
                    1: "filtered",
                    2: "duplicate",
                    3: "manual_clear",
                }.get(int(code), "filtered")
                for code in source_surface_decision_codes.tolist()
            ],
            dtype=object,
        )
        source_surface_reason = read_reason_labels(source_group)
        if source_surface_reason is None:
            source_surface_reason = np.asarray(source_surface_decision_labels, dtype=object)
        source_surface_reason_labels = np.asarray(source_surface_reason, dtype=object).reshape(-1)
        source_surface_confidence_scores = (
            np.asarray(source_group["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in source_group
            else np.full(source_surface_frame_indices.shape[0], np.nan, dtype=np.float32)
        )
        source_surface_class_ids = (
            np.asarray(source_group["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in source_group
            else np.full(source_surface_frame_indices.shape[0], -1, dtype=np.int32)
        )
        source_surface_review_notes = (
            np.asarray(source_group["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in source_group
            else np.full(source_surface_frame_indices.shape[0], "", dtype=object)
        )

        source_slot_rows: Dict[Tuple[int, int], List[int]] = {}
        for idx, (frame, raw_row_index, bbox) in enumerate(
            zip(
                source_surface_frame_indices.tolist(),
                source_surface_source_detect_row_index.tolist(),
                source_surface_bbox_norm_coords.tolist(),
            )
        ):
            source_row_lookup[int(raw_row_index)] = int(idx)
            arena_id = _bbox_norm_to_arena_id(
                np.asarray(bbox, dtype=np.float64),
                arena_definitions=arena_defs,
                width=width,
                height=height,
            )
            if arena_id < 0:
                continue
            source_slot_rows.setdefault((int(frame), int(arena_id)), []).append(int(idx))

        source_rows_by_slot = {
            key: np.asarray(value, dtype=np.int32)
            for key, value in source_slot_rows.items()
        }

        for (frame, arena_id), row_indices in source_rows_by_slot.items():
            slot_row = slot_to_row.get((int(frame), int(arena_id)))
            if slot_row is None:
                continue
            chosen_row_idx = _pick_slot_source_surface_row(
                row_indices,
                preferred_source_detect_row_index=-1,
                source_surface_source_detect_row_index=source_surface_source_detect_row_index,
                source_surface_decision_labels=source_surface_decision_labels,
                source_surface_confidence_scores=source_surface_confidence_scores,
            )
            if chosen_row_idx is None:
                continue
            decision_label = str(source_surface_decision_labels[chosen_row_idx])
            bbox_norm[slot_row] = np.asarray(
                source_surface_bbox_norm_coords[chosen_row_idx],
                dtype=np.float64,
            )
            source_detect_row_index[slot_row] = int(
                source_surface_source_detect_row_index[chosen_row_idx]
            )
            confidence_scores[slot_row] = np.float32(
                source_surface_confidence_scores[chosen_row_idx]
            )
            class_ids[slot_row] = np.int32(source_surface_class_ids[chosen_row_idx])
            if decision_label == "manual_clear":
                status_labels[slot_row] = "filtered_out"
                source_kind_labels[slot_row] = "none"
                manual_edit_flags[slot_row] = True
                reason_labels[slot_row] = (
                    str(source_surface_reason_labels[chosen_row_idx]) or "manual_clear"
                )
            elif decision_label in {"filtered", "duplicate"}:
                status_labels[slot_row] = "filtered_out"
                source_kind_labels[slot_row] = "raw_detect"
                reason_labels[slot_row] = (
                    str(source_surface_reason_labels[chosen_row_idx]) or decision_label
                )
            elif decision_label == "accepted":
                status_labels[slot_row] = "present"
                source_kind_labels[slot_row] = "raw_detect"
                reason_labels[slot_row] = (
                    str(source_surface_reason_labels[chosen_row_idx]) or "clean"
                )

    instances = refined_run["instances"]
    inst_frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
    inst_refined_row_ids = np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1)
    inst_bbox_norm = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    inst_source_kind_codes = np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
    inst_source_kind_labels = np.asarray(
        [_SOURCE_KIND_LABEL_BY_CODE.get(int(code), "unknown") for code in inst_source_kind_codes.tolist()],
        dtype=object,
    )
    inst_source_detect_row_index = np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
    inst_manual_edit_flags = np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1)
    inst_reason_labels = read_reason_labels(instances)
    if inst_reason_labels is None:
        inst_reason_labels = np.full(inst_frame_indices.shape[0], "", dtype=object)
    inst_confidence_scores = (
        np.asarray(instances["confidence_scores"][:], dtype=np.float32).reshape(-1)
        if "confidence_scores" in instances
        else np.full(inst_frame_indices.shape[0], np.nan, dtype=np.float32)
    )
    inst_class_ids = (
        np.asarray(instances["class_ids"][:], dtype=np.int32).reshape(-1)
        if "class_ids" in instances
        else np.full(inst_frame_indices.shape[0], -1, dtype=np.int32)
    )

    seen_slots: set[Tuple[int, int]] = set()
    for frame, row_id, row_bbox, row_source_kind, row_source_detect_idx, row_manual_flag, row_reason, row_score, row_class in zip(
        inst_frame_indices.tolist(),
        inst_refined_row_ids.tolist(),
        inst_bbox_norm.tolist(),
        inst_source_kind_labels.tolist(),
        inst_source_detect_row_index.tolist(),
        inst_manual_edit_flags.tolist(),
        np.asarray(inst_reason_labels, dtype=object).tolist(),
        inst_confidence_scores.tolist(),
        inst_class_ids.tolist(),
    ):
        arena_id = _bbox_norm_to_arena_id(
            np.asarray(row_bbox, dtype=np.float64),
            arena_definitions=arena_defs,
            width=width,
            height=height,
        )
        if arena_id < 0:
            raise RuntimeError(
                "Arena-aware detect review requires curated instance boxes to fall inside "
                "a configured arena ROI."
            )
        key = (int(frame), int(arena_id))
        if key in seen_slots:
            raise RuntimeError(
                "Arena-aware detect review found multiple curated instances for the same "
                f"(frame, arena_id)=({frame}, {arena_id})."
            )
        seen_slots.add(key)
        slot_row = slot_to_row.get(key)
        if slot_row is None:
            continue
        bbox_norm[slot_row] = np.asarray(row_bbox, dtype=np.float64)
        status_labels[slot_row] = "present"
        source_kind_labels[slot_row] = str(row_source_kind)
        source_detect_row_index[slot_row] = int(row_source_detect_idx)
        manual_edit_flags[slot_row] = bool(row_manual_flag)
        reason_labels[slot_row] = str(row_reason) or "present"
        confidence_scores[slot_row] = np.float32(row_score)
        class_ids[slot_row] = np.int32(row_class)
        refined_row_ids[slot_row] = np.int64(row_id)
        detection_source[slot_row] = 1 if str(row_source_kind) == "interpolated" else 0

    roi_lookup = {
        int(arena["id"]): tuple(int(value) for value in arena["roi_pixels"])
        for arena in arena_defs
    }
    return {
        "review_axis": np.asarray(["frame_arena"], dtype=object),
        "frame_indices": slot_frame_indices,
        "arena_ids": slot_arena_ids,
        "bbox_norm_coords": bbox_norm,
        "confidence_scores": confidence_scores,
        "class_ids": class_ids,
        "status_labels": status_labels,
        "source_kind_labels": source_kind_labels,
        "manual_edit_flags": manual_edit_flags,
        "reason_labels": reason_labels,
        "source_detect_row_index": source_detect_row_index,
        "detection_source": detection_source,
        "refined_row_ids": refined_row_ids,
        "slot_to_row": slot_to_row,
        "arena_definitions": np.asarray(arena_defs, dtype=object),
        "arena_roi_lookup": roi_lookup,
        "source_surface_source_detect_row_index": source_surface_source_detect_row_index,
        "source_surface_frame_indices": source_surface_frame_indices,
        "source_surface_bbox_norm_coords": source_surface_bbox_norm_coords,
        "source_surface_decision_labels": source_surface_decision_labels,
        "source_surface_reason_labels": source_surface_reason_labels,
        "source_surface_confidence_scores": source_surface_confidence_scores,
        "source_surface_class_ids": source_surface_class_ids,
        "source_surface_review_notes": source_surface_review_notes,
        "source_rows_by_slot": source_rows_by_slot,
        "source_row_lookup": source_row_lookup,
    }


def _apply_manual_changes_to_arena_payload(
    payload: Dict[str, np.ndarray],
    *,
    manual_changes: Mapping[Tuple[int, int], Optional[np.ndarray]],
    manual_score: float,
    manual_class_id: int,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    updated = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in payload.items()
        if key != "slot_to_row"
    }
    slot_to_row = payload["slot_to_row"]
    removed_slots = 0
    added_slots = 0
    for (frame, arena_id), rect_norm in manual_changes.items():
        row_idx = slot_to_row.get((int(frame), int(arena_id)))
        if row_idx is None:
            raise RuntimeError(f"Arena-aware refined payload is missing slot {(frame, arena_id)}.")
        current_source_row_index = int(updated["source_detect_row_index"][row_idx])
        source_surface_row_idx = _resolve_source_surface_row_for_slot(
            updated,
            frame=int(frame),
            arena_id=int(arena_id),
            preferred_source_detect_row_index=current_source_row_index,
        )
        chosen_source_row_index = (
            int(updated["source_surface_source_detect_row_index"][source_surface_row_idx])
            if source_surface_row_idx is not None
            else -1
        )
        updated["source_detect_row_index"][row_idx] = chosen_source_row_index
        updated["detection_source"][row_idx] = 0
        if rect_norm is None:
            removed_slots += 1
            updated["bbox_norm_coords"][row_idx] = np.full((4,), np.nan, dtype=np.float64)
            updated["confidence_scores"][row_idx] = np.float32(np.nan)
            updated["class_ids"][row_idx] = np.int32(-1)
            updated["status_labels"][row_idx] = "filtered_out"
            updated["source_kind_labels"][row_idx] = "none"
            updated["manual_edit_flags"][row_idx] = True
            updated["reason_labels"][row_idx] = "manual_clear"
            if source_surface_row_idx is not None:
                updated["source_surface_decision_labels"][source_surface_row_idx] = "manual_clear"
                updated["source_surface_reason_labels"][source_surface_row_idx] = "manual_clear"
        else:
            added_slots += 1
            updated["bbox_norm_coords"][row_idx] = np.asarray(rect_norm, dtype=np.float64)
            updated["confidence_scores"][row_idx] = np.float32(manual_score)
            updated["class_ids"][row_idx] = np.int32(manual_class_id)
            updated["status_labels"][row_idx] = "present"
            updated["source_kind_labels"][row_idx] = "manual"
            updated["manual_edit_flags"][row_idx] = True
            updated["reason_labels"][row_idx] = "manual_correction"
            if source_surface_row_idx is not None:
                updated["source_surface_decision_labels"][source_surface_row_idx] = "accepted"
                updated["source_surface_reason_labels"][source_surface_row_idx] = "manual_correction"
    return updated, added_slots, removed_slots


def _payload_review_axis(payload: Mapping[str, object]) -> str:
    raw = payload.get("review_axis")
    if raw is None:
        return "frame"
    axis = np.asarray(raw, dtype=object).reshape(-1)
    if axis.size == 0:
        return "frame"
    return str(axis[0] or "frame")


def _load_refined_review_payload(
    root: zarr.Group,
    refined_run: zarr.Group,
    *,
    total_frames: int,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    arena_definitions = _resolve_review_arena_definitions(root)
    if len(arena_definitions) > 1 and has_sparse_curated_refined_detect_instances_arrays(refined_run):
        return _load_arena_slot_curated_edit_payload(
            refined_run,
            arena_definitions=arena_definitions,
            total_frames=total_frames,
            width=width,
            height=height,
        )
    return _load_dense_curated_edit_payload(refined_run, total_frames=total_frames)


def _select_refined_review_rows(
    payload: Mapping[str, np.ndarray],
    *,
    review_all: bool,
    target_frames: Optional[np.ndarray],
    max_items: Optional[int],
) -> np.ndarray:
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    review_default = np.asarray(payload["status_labels"], dtype=object).reshape(-1) != "present"
    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        review_rows = np.where(np.isin(frame_indices, unique_frames))[0].astype(np.int32, copy=False)
    elif review_all:
        review_rows = np.arange(frame_indices.shape[0], dtype=np.int32)
    else:
        review_rows = np.where(review_default)[0].astype(np.int32, copy=False)

    if max_items is not None:
        review_rows = review_rows[:max_items]
    return review_rows.astype(np.int32, copy=False)


def _load_dense_curated_edit_payload(
    refined_run: zarr.Group,
    *,
    total_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if has_sparse_curated_refined_detect_instances_arrays(refined_run):
        instances = refined_run["instances"]
        if total_frames is None:
            total_frames = int(instances["frame_counts"].shape[0]) if "frame_counts" in instances else 0
            if total_frames <= 0 and has_curated_refined_source_detections_projection(refined_run):
                src_frames = np.asarray(
                    refined_run["source_detections"]["frame_indices"][:],
                    dtype=np.int32,
                ).reshape(-1)
                total_frames = int(np.max(src_frames)) + 1 if src_frames.size else 0
            if total_frames <= 0 and has_curated_refined_detect_arrays(refined_run):
                total_frames = int(refined_run["frame_indices"].shape[0])

        frame_indices = np.arange(int(total_frames), dtype=np.int32)
        entity_ids = np.zeros(int(total_frames), dtype=np.int32)
        bbox_norm = np.full((int(total_frames), 4), np.nan, dtype=np.float64)
        confidence_scores = np.full(int(total_frames), np.nan, dtype=np.float32)
        class_ids = np.full(int(total_frames), -1, dtype=np.int32)
        status_labels = np.full(int(total_frames), "missing", dtype=object)
        source_kind_labels = np.full(int(total_frames), "none", dtype=object)
        manual_edit_flags = np.zeros(int(total_frames), dtype=bool)
        reason_labels = np.full(int(total_frames), "missing_detection", dtype=object)
        source_detect_row_index = np.full(int(total_frames), -1, dtype=np.int32)
        detection_source = np.zeros(int(total_frames), dtype=np.int8)
        refined_row_ids = np.full(int(total_frames), -1, dtype=np.int64)

        source_surface_frame_indices = np.empty((0,), dtype=np.int32)
        source_surface_source_detect_row_index = np.empty((0,), dtype=np.int32)
        source_surface_bbox_norm_coords = np.empty((0, 4), dtype=np.float64)
        source_surface_decision_labels = np.empty((0,), dtype=object)
        source_surface_reason_labels = np.empty((0,), dtype=object)
        source_surface_confidence_scores = np.empty((0,), dtype=np.float32)
        source_surface_class_ids = np.empty((0,), dtype=np.int32)
        source_surface_review_notes = np.empty((0,), dtype=object)
        source_rows_by_frame: Dict[int, np.ndarray] = {}
        source_row_lookup: Dict[int, int] = {}

        if has_curated_refined_source_detections_projection(refined_run):
            source_group = refined_run["source_detections"]
            source_surface_frame_indices = np.asarray(source_group["frame_indices"][:], dtype=np.int32).reshape(-1)
            source_surface_source_detect_row_index = np.asarray(
                source_group["source_detect_row_index"][:],
                dtype=np.int32,
            ).reshape(-1)
            source_surface_bbox_norm_coords = np.asarray(
                source_group["bbox_norm_coords"][:],
                dtype=np.float64,
            ).reshape(-1, 4)
            source_surface_decision_codes = np.asarray(source_group["decision_codes"][:], dtype=np.int8).reshape(-1)
            source_surface_decision_labels = np.asarray(
                [
                    {
                        0: "accepted",
                        1: "filtered",
                        2: "duplicate",
                        3: "manual_clear",
                    }.get(int(code), "filtered")
                    for code in source_surface_decision_codes.tolist()
                ],
                dtype=object,
            )
            source_surface_reason = read_reason_labels(source_group)
            if source_surface_reason is None:
                source_surface_reason = np.asarray(source_surface_decision_labels, dtype=object)
            source_surface_reason_labels = np.asarray(source_surface_reason, dtype=object).reshape(-1)
            source_surface_confidence_scores = (
                np.asarray(source_group["confidence_scores"][:], dtype=np.float32).reshape(-1)
                if "confidence_scores" in source_group
                else np.full(source_surface_frame_indices.shape[0], np.nan, dtype=np.float32)
            )
            source_surface_class_ids = (
                np.asarray(source_group["class_ids"][:], dtype=np.int32).reshape(-1)
                if "class_ids" in source_group
                else np.full(source_surface_frame_indices.shape[0], -1, dtype=np.int32)
            )
            source_surface_review_notes = (
                np.asarray(source_group["review_notes"][:], dtype=object).reshape(-1)
                if "review_notes" in source_group
                else np.full(source_surface_frame_indices.shape[0], "", dtype=object)
            )
            for idx, (frame, source_row_index_value) in enumerate(
                zip(source_surface_frame_indices.tolist(), source_surface_source_detect_row_index.tolist())
            ):
                source_rows_by_frame.setdefault(int(frame), []).append(int(idx))
                source_row_lookup[int(source_row_index_value)] = int(idx)

            for frame, row_indices in source_rows_by_frame.items():
                if frame < 0 or frame >= int(total_frames):
                    continue
                if len(row_indices) > 1:
                    status_labels[frame] = "ambiguous"
                    reason_labels[frame] = "multiple_candidates"
                    manual_edit_flags[frame] = any(
                        str(source_surface_decision_labels[row_idx]) == "manual_clear"
                        for row_idx in row_indices
                    )
                    continue
                row_idx = int(row_indices[0])
                decision_label = str(source_surface_decision_labels[row_idx])
                source_row_index_value = int(source_surface_source_detect_row_index[row_idx])
                source_detect_row_index[frame] = source_row_index_value
                confidence_scores[frame] = np.float32(source_surface_confidence_scores[row_idx])
                class_ids[frame] = np.int32(source_surface_class_ids[row_idx])
                if decision_label == "manual_clear":
                    status_labels[frame] = "filtered_out"
                    source_kind_labels[frame] = "none"
                    manual_edit_flags[frame] = True
                    reason_labels[frame] = str(source_surface_reason_labels[row_idx]) or "manual_clear"
                elif decision_label in {"filtered", "duplicate"}:
                    status_labels[frame] = "filtered_out"
                    source_kind_labels[frame] = "raw_detect"
                    reason_labels[frame] = str(source_surface_reason_labels[row_idx]) or decision_label

        inst_frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
        if inst_frame_indices.size and np.any(np.diff(np.sort(inst_frame_indices)) == 0):
            raise RuntimeError(
                "Refined detect review does not yet support multi-instance refined runs. "
                "The canonical sparse surface may contain more than one instance per frame, "
                "but the current review UI still assumes at most one curated instance per frame."
            )
        inst_refined_row_ids = np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1)
        inst_bbox_norm = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        inst_source_kind_codes = np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        inst_source_kind_labels = np.asarray(
            [_SOURCE_KIND_LABEL_BY_CODE.get(int(code), "unknown") for code in inst_source_kind_codes.tolist()],
            dtype=object,
        )
        inst_source_detect_row_index = np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        inst_manual_edit_flags = np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1)
        inst_reason_labels = read_reason_labels(instances)
        if inst_reason_labels is None:
            inst_reason_labels = np.full(inst_frame_indices.shape[0], "", dtype=object)
        inst_confidence_scores = (
            np.asarray(instances["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in instances
            else np.full(inst_frame_indices.shape[0], np.nan, dtype=np.float32)
        )
        inst_class_ids = (
            np.asarray(instances["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in instances
            else np.full(inst_frame_indices.shape[0], -1, dtype=np.int32)
        )
        for frame, row_id, row_bbox, row_source_kind, row_source_detect_idx, row_manual_flag, row_reason, row_score, row_class in zip(
            inst_frame_indices.tolist(),
            inst_refined_row_ids.tolist(),
            inst_bbox_norm.tolist(),
            inst_source_kind_labels.tolist(),
            inst_source_detect_row_index.tolist(),
            inst_manual_edit_flags.tolist(),
            np.asarray(inst_reason_labels, dtype=object).tolist(),
            inst_confidence_scores.tolist(),
            inst_class_ids.tolist(),
        ):
            frame_idx = int(frame)
            if frame_idx < 0 or frame_idx >= int(total_frames):
                continue
            bbox_norm[frame_idx] = np.asarray(row_bbox, dtype=np.float64)
            status_labels[frame_idx] = "present"
            source_kind_labels[frame_idx] = str(row_source_kind)
            source_detect_row_index[frame_idx] = int(row_source_detect_idx)
            manual_edit_flags[frame_idx] = bool(row_manual_flag)
            reason_labels[frame_idx] = str(row_reason) or "present"
            confidence_scores[frame_idx] = np.float32(row_score)
            class_ids[frame_idx] = np.int32(row_class)
            refined_row_ids[frame_idx] = np.int64(row_id)

        detection_source = np.where(
            np.asarray(source_kind_labels, dtype=object) == "interpolated",
            1,
            0,
        ).astype(np.int8, copy=False)
        frame_to_row = {int(frame): int(frame) for frame in frame_indices.tolist()}

        return {
            "review_axis": np.asarray(["frame"], dtype=object),
            "frame_indices": frame_indices,
            "storage_row_indices": np.asarray(frame_indices, dtype=np.int32),
            "entity_ids": entity_ids,
            "bbox_norm_coords": bbox_norm,
            "confidence_scores": confidence_scores,
            "class_ids": class_ids,
            "status_labels": status_labels,
            "source_kind_labels": source_kind_labels,
            "manual_edit_flags": manual_edit_flags,
            "reason_labels": reason_labels,
            "source_detect_row_index": source_detect_row_index,
            "detection_source": detection_source,
            "refined_row_ids": refined_row_ids,
            "frame_to_row": frame_to_row,
            "source_surface_source_detect_row_index": source_surface_source_detect_row_index,
            "source_surface_frame_indices": source_surface_frame_indices,
            "source_surface_bbox_norm_coords": source_surface_bbox_norm_coords,
            "source_surface_decision_labels": source_surface_decision_labels,
            "source_surface_reason_labels": source_surface_reason_labels,
            "source_surface_confidence_scores": source_surface_confidence_scores,
            "source_surface_class_ids": source_surface_class_ids,
            "source_surface_review_notes": source_surface_review_notes,
            "source_rows_by_frame": {key: np.asarray(value, dtype=np.int32) for key, value in source_rows_by_frame.items()},
            "source_row_lookup": source_row_lookup,
        }

    if not has_curated_refined_detect_arrays(refined_run):
        raise RuntimeError("Refined run does not have a readable canonical curated detect surface.")

    frame_indices = np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)
    entity_ids = np.asarray(refined_run["entity_ids"][:], dtype=np.int32).reshape(-1)
    order = np.argsort(frame_indices, kind="stable")
    frame_indices = frame_indices[order]
    entity_ids = entity_ids[order]

    if frame_indices.size and np.any(np.diff(frame_indices) == 0):
        raise RuntimeError(
            "Refined detect review does not yet support multi-instance dense compatibility rows."
        )
    if np.any(entity_ids != 0):
        raise RuntimeError(
            "Refined detect review currently supports only the single-instance dense compatibility projection."
        )

    bbox_norm = np.asarray(refined_run["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)[order]
    status_codes = np.asarray(refined_run["status_codes"][:], dtype=np.int8).reshape(-1)[order]
    source_kind_codes = np.asarray(refined_run["source_kind_codes"][:], dtype=np.int8).reshape(-1)[order]
    confidence_scores = (
        np.asarray(refined_run["confidence_scores"][:], dtype=np.float32).reshape(-1)[order]
        if "confidence_scores" in refined_run
        else np.full(frame_indices.shape[0], np.nan, dtype=np.float32)
    )
    class_ids = (
        np.asarray(refined_run["class_ids"][:], dtype=np.int32).reshape(-1)[order]
        if "class_ids" in refined_run
        else np.full(frame_indices.shape[0], -1, dtype=np.int32)
    )
    source_detect_row_index = (
        np.asarray(refined_run["source_detect_row_index"][:], dtype=np.int32).reshape(-1)[order]
        if "source_detect_row_index" in refined_run
        else np.full(frame_indices.shape[0], -1, dtype=np.int32)
    )
    manual_edit_flags = (
        np.asarray(refined_run["manual_edit_flags"][:], dtype=bool).reshape(-1)[order]
        if "manual_edit_flags" in refined_run
        else np.zeros(frame_indices.shape[0], dtype=bool)
    )
    reason_labels = read_reason_labels(refined_run)
    if reason_labels is None:
        reason_labels = np.full(frame_indices.shape[0], "unknown", dtype=object)
    else:
        reason_labels = np.asarray(reason_labels, dtype=object)[order]

    detection_source = build_curated_detection_source_array(refined_run)[order]
    status_labels = np.asarray(
        [_STATUS_LABEL_BY_CODE.get(int(code), "unknown") for code in status_codes.tolist()],
        dtype=object,
    )
    source_kind_labels = np.asarray(
        [_SOURCE_KIND_LABEL_BY_CODE.get(int(code), "unknown") for code in source_kind_codes.tolist()],
        dtype=object,
    )
    frame_to_row = {int(frame): idx for idx, frame in enumerate(frame_indices.tolist())}

    return {
        "review_axis": np.asarray(["frame"], dtype=object),
        "frame_indices": frame_indices,
        "storage_row_indices": np.asarray(order, dtype=np.int32),
        "entity_ids": entity_ids,
        "bbox_norm_coords": bbox_norm,
        "confidence_scores": confidence_scores,
        "class_ids": class_ids,
        "status_labels": status_labels,
        "source_kind_labels": source_kind_labels,
        "manual_edit_flags": manual_edit_flags,
        "reason_labels": np.asarray(reason_labels, dtype=object),
        "source_detect_row_index": source_detect_row_index,
        "detection_source": detection_source,
        "frame_to_row": frame_to_row,
    }


def _load_bound_detect_instance_keys(
    root: zarr.Group,
    *,
    refined_run_name: str,
) -> Optional[np.ndarray]:
    """Return detect-row-aligned instance keys from the bound source detect run."""

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        refined_parent = root.get("refined_runs")
    if refined_parent is None or refined_run_name not in refined_parent:
        return None
    detect_group, _ = resolve_bound_source_detect_group(root, refined_parent[refined_run_name])
    if detect_group is None or "instance_key" not in detect_group:
        return None
    return np.asarray(detect_group["instance_key"][:], dtype=np.uint64).reshape(-1)


def _write_dense_curated_edit_payload(
    root: zarr.Group,
    *,
    zarr_path: str,
    refined_run_name: str,
    payload: Dict[str, np.ndarray],
    row_indices: np.ndarray,
    command_label: str,
    source_context: Dict[str, object],
) -> None:
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=zarr_path,
        collect_ip=False,
    )
    payload_row_indices = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    if "source_surface_source_detect_row_index" in payload:
        status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
        bbox_norm = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
        present_mask = (status_labels == "present") & np.all(np.isfinite(bbox_norm), axis=1)
        instance_frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)[present_mask]
        instance_bbox_norm_coords = bbox_norm[present_mask]
        instance_source_detect_row_index = np.asarray(
            payload["source_detect_row_index"],
            dtype=np.int32,
        ).reshape(-1)[present_mask]
        instance_class_ids = np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)[present_mask]
        source_surface_source_detect_row_index = np.asarray(
            payload["source_surface_source_detect_row_index"],
            dtype=np.int32,
        ).reshape(-1)

        detect_instance_keys = _load_bound_detect_instance_keys(root, refined_run_name=refined_run_name)
        instance_key = None
        instance_key_origin_codes = None
        source_detection_instance_key = None
        if detect_instance_keys is not None:
            source_rows = source_surface_source_detect_row_index.astype(np.int64, copy=False)
            if source_rows.size and (
                int(source_rows.min()) < 0 or int(source_rows.max()) >= int(detect_instance_keys.shape[0])
            ):
                raise ValueError(
                    "Curated source_detections rows reference detect rows outside the bound "
                    f"detect run ({int(detect_instance_keys.shape[0])} rows); refusing to rewrite "
                    "instance keys against a mismatched detect binding."
                )
            source_detection_instance_key = detect_instance_keys[source_rows]
            instance_key, instance_key_origin_codes = resolve_curated_instance_keys(
                root,
                zarr_path=Path(zarr_path),
                instance_frame_indices=instance_frame_indices,
                instance_bbox_norm_coords=instance_bbox_norm_coords,
                instance_class_ids=instance_class_ids,
                instance_source_detect_row_index=instance_source_detect_row_index,
                source_detection_instance_key=detect_instance_keys,
            )

        write_curated_refined_detect_surfaces(
            root,
            zarr_path=Path(zarr_path),
            refined_run_name=refined_run_name,
            instance_frame_indices=instance_frame_indices,
            instance_bbox_norm_coords=instance_bbox_norm_coords,
            instance_source_kind_labels=np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1)[present_mask],
            instance_reason_labels=np.asarray(payload["reason_labels"], dtype=object).reshape(-1)[present_mask],
            instance_source_detect_row_index=instance_source_detect_row_index,
            instance_manual_edit_flags=np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)[present_mask],
            instance_confidence_scores=np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)[present_mask],
            instance_class_ids=instance_class_ids,
            instance_key=instance_key,
            instance_key_origin_codes=instance_key_origin_codes,
            instance_refined_row_ids=np.asarray(payload["refined_row_ids"], dtype=np.int64).reshape(-1)[present_mask],
            source_detection_source_detect_row_index=source_surface_source_detect_row_index,
            source_detection_frame_indices=np.asarray(
                payload["source_surface_frame_indices"],
                dtype=np.int32,
            ).reshape(-1),
            source_detection_bbox_norm_coords=np.asarray(
                payload["source_surface_bbox_norm_coords"],
                dtype=np.float64,
            ).reshape(-1, 4),
            source_detection_decision_labels=np.asarray(
                payload["source_surface_decision_labels"],
                dtype=object,
            ).reshape(-1),
            source_detection_reason_labels=np.asarray(
                payload["source_surface_reason_labels"],
                dtype=object,
            ).reshape(-1),
            source_detection_confidence_scores=np.asarray(
                payload["source_surface_confidence_scores"],
                dtype=np.float32,
            ).reshape(-1),
            source_detection_class_ids=np.asarray(
                payload["source_surface_class_ids"],
                dtype=np.int32,
            ).reshape(-1),
            source_detection_instance_key=source_detection_instance_key,
            source_detection_review_notes=np.asarray(
                payload["source_surface_review_notes"],
                dtype=object,
            ).reshape(-1),
            command=command_label,
            env_info=env_info,
            source_context=source_context,
        )
        return

    storage_row_indices = np.asarray(payload["storage_row_indices"], dtype=np.int32).reshape(-1)
    row_indices_arr = storage_row_indices[payload_row_indices] if payload_row_indices.size else np.empty((0,), dtype=np.int32)
    update_kwargs: Dict[str, object] = {
        "zarr_path": Path(zarr_path),
        "refined_run_name": refined_run_name,
        "row_indices": row_indices_arr,
        "command": command_label,
        "env_info": env_info,
        "source_context": source_context,
    }
    if row_indices_arr.size:
        update_kwargs.update(
            {
                "bbox_norm_coords": payload["bbox_norm_coords"][payload_row_indices],
                "status_labels": payload["status_labels"][payload_row_indices],
                "source_kind_labels": payload["source_kind_labels"][payload_row_indices],
                "reason_labels": payload["reason_labels"][payload_row_indices],
                "source_detect_row_index": payload["source_detect_row_index"][payload_row_indices],
                "manual_edit_flags": payload["manual_edit_flags"][payload_row_indices],
                "detection_source": payload["detection_source"][payload_row_indices],
                "confidence_scores": payload["confidence_scores"][payload_row_indices],
                "class_ids": payload["class_ids"][payload_row_indices],
            }
        )
    update_curated_refined_detect_rows(root, **update_kwargs)


def _norm_to_rect(norm: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    cx, cy, w, h = norm.tolist()
    x1 = (cx - w * 0.5) * width
    y1 = (cy - h * 0.5) * height
    x2 = (cx + w * 0.5) * width
    y2 = (cy + h * 0.5) * height
    return x1, y1, x2, y2


def _rect_to_norm(rect: Tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    cx = (x_min + x_max) * 0.5 / width
    cy = (y_min + y_max) * 0.5 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return np.array([cx, cy, w, h], dtype=np.float64)


def _resolve_source_surface_row_for_frame(
    payload: Dict[str, np.ndarray],
    *,
    frame: int,
    preferred_source_detect_row_index: int,
) -> Optional[int]:
    lookup = payload.get("source_row_lookup")
    if isinstance(lookup, dict) and preferred_source_detect_row_index >= 0:
        row_idx = lookup.get(int(preferred_source_detect_row_index))
        if row_idx is not None:
            return int(row_idx)
    rows_by_frame = payload.get("source_rows_by_frame")
    if isinstance(rows_by_frame, dict):
        row_indices = rows_by_frame.get(int(frame))
        if row_indices is not None and int(np.asarray(row_indices).shape[0]) == 1:
            return int(np.asarray(row_indices, dtype=np.int32).reshape(-1)[0])
    return None


def _group_by_frame(frame_indices: np.ndarray) -> Dict[int, np.ndarray]:
    frame_map: Dict[int, List[int]] = {}
    for idx, frame in enumerate(frame_indices.astype(int)):
        frame_map.setdefault(int(frame), []).append(idx)
    return {frame: np.asarray(indices, dtype=np.int32) for frame, indices in frame_map.items()}


def _compute_frame_counts(frame_indices: np.ndarray, n_frames: int) -> np.ndarray:
    if frame_indices.size == 0:
        return np.zeros(n_frames, dtype=np.int32)
    return np.bincount(frame_indices.astype(np.int32, copy=False), minlength=n_frames).astype(np.int32, copy=False)


def _parse_frames_arg(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    items: List[object]
    path = Path(text)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, list):
            items = data
        else:
            items = re.split(r"[,\s]+", raw.strip())
    else:
        items = re.split(r"[,\s]+", text)
    frames: List[int] = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            frames.append(int(item))
            continue
        token = str(item).strip()
        if not token:
            continue
        try:
            frames.append(int(token))
        except ValueError:
            continue
    if not frames:
        return None
    return np.asarray(sorted(set(frames)), dtype=np.int32)


def _get_or_create_retune_id(refined_run: zarr.Group, params: Dict[str, object]) -> int:
    existing = refined_run.attrs.get("retune_params")
    retune_params = existing if isinstance(existing, dict) else {}

    def signature(values: Dict[str, object]) -> tuple:
        return tuple(sorted(values.items()))

    target = signature(params)
    for key, value in retune_params.items():
        if isinstance(value, dict) and signature(value) == target:
            try:
                return int(key)
            except ValueError:
                continue

    existing_ids = [int(k) for k in retune_params.keys() if str(k).isdigit()]
    next_id = max(existing_ids, default=0) + 1
    retune_params[str(next_id)] = params
    refined_run.attrs["retune_params"] = retune_params
    return next_id


def _select_retune_base(
    refined_run: zarr.Group,
    variant: str,
    output_group: str,
) -> Tuple[zarr.Group, str, bool]:
    if variant == "refined":
        return refined_run, "refined", False
    if output_group and output_group in refined_run:
        return refined_run[output_group], output_group, True
    if output_group == "manual":
        manual_latest = refined_run.attrs.get("manual_review_latest")
        if manual_latest and manual_latest in refined_run:
            return refined_run[manual_latest], str(manual_latest), True
    return refined_run[variant], variant, False


def _write_manual_group(
    refined_run: zarr.Group,
    output_group: str,
    frame_indices: np.ndarray,
    bbox_norm: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    retune_id: Optional[np.ndarray],
    frame_counts: np.ndarray,
    detection_source: Optional[np.ndarray],
    reason: Optional[np.ndarray],
    metadata: Dict[str, object],
    overwrite: bool,
) -> None:
    if output_group in refined_run:
        if not overwrite:
            raise RuntimeError(f"Output group '{output_group}' already exists. Use --overwrite to replace it.")
        del refined_run[output_group]

    group = refined_run.create_group(output_group)

    det_chunk = max(1, min(max(1, frame_indices.size), 4096))
    counts_chunk = max(1, min(frame_counts.size, 16384))

    group.create_array("frame_indices", data=frame_indices, chunks=(det_chunk,), overwrite=True)
    group.create_array("bbox_norm_coords", data=bbox_norm, chunks=(det_chunk, 4), overwrite=True)
    group.create_array("scores", data=scores, chunks=(det_chunk,), overwrite=True)
    group.create_array("class_ids", data=class_ids, chunks=(det_chunk,), overwrite=True)
    if retune_id is not None:
        group.create_array("retune_id", data=retune_id, chunks=(det_chunk,), overwrite=True)
    group.create_array("frame_counts", data=frame_counts, chunks=(counts_chunk,), overwrite=True)
    group.create_array("n_detections", data=frame_counts, chunks=(counts_chunk,), overwrite=True)
    group.create_array("frame_mapping", data=frame_indices, chunks=(det_chunk,), overwrite=True)

    column_fields = ["frame_indices", "bbox_norm_coords", "scores", "class_ids"]
    if retune_id is not None:
        column_fields.append("retune_id")
    if detection_source is None:
        detection_source_arr = np.zeros(frame_indices.shape[0], dtype=np.int8)
    else:
        detection_source_arr = np.asarray(detection_source, dtype=np.int8)
    if detection_source_arr.shape[0] != frame_indices.shape[0]:
        raise RuntimeError("detection_source length does not match frame_indices length.")
    group.create_array("detection_source", data=detection_source_arr, chunks=(det_chunk,), overwrite=True)
    column_fields.append("detection_source")

    if reason is None:
        reason_arr = np.where(detection_source_arr == 1, "interpolated", "clean").astype(object)
    else:
        reason_arr = np.asarray(reason, dtype=object)
    if reason_arr.shape[0] != frame_indices.shape[0]:
        raise RuntimeError("reason length does not match frame_indices length.")
    written_reason_fields = write_reason_columns(
        group,
        reason_arr,
        det_chunk,
        overwrite=True,
    )
    column_fields.extend(written_reason_fields)

    group.attrs["storage_layout"] = "columnar"
    group.attrs["column_fields"] = column_fields
    group.attrs["field_names"] = column_fields
    group.attrs.update(metadata)


def _save_detection_tuning(
    root: zarr.Group,
    params: Dict[str, object],
    tuned_frame: Optional[int] = None,
) -> None:
    if "analysis_metadata" not in root:
        analysis_meta = root.create_group("analysis_metadata")
    else:
        analysis_meta = root["analysis_metadata"]

    metadata = dict(analysis_meta.attrs) if analysis_meta.attrs else {}
    metadata["detection_tuning"] = {
        "method": "blob_detection",
        "version": "1.0",
        "tuned_timestamp": datetime.now(timezone.utc).isoformat(),
        "tuned_parameters": {
            "ds_thresh": int(params["ds_thresh"]),
            "se1_radius": int(params["se1_radius"]),
            "se4_radius": int(params["se4_radius"]),
            "min_area": int(params["min_area"]),
            "max_area": int(params["max_area"]),
            "max_fish": int(params.get("max_fish", 20)),
        },
        "tuned_on_frame": int(tuned_frame) if tuned_frame is not None else None,
    }
    analysis_meta.attrs.update(metadata)


def _update_curated_refined_after_manual_write(
    root: zarr.Group,
    *,
    zarr_path: str,
    refined_run_name: str,
    source_group: str,
) -> None:
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=zarr_path,
        collect_ip=False,
    )
    try:
        payload = materialize_refined_detect_curation(
            root,
            zarr_path=Path(zarr_path),
            refined_run_name=refined_run_name,
            source_group=source_group,
            command=" ".join(sys.argv),
            env_info=env_info,
        )
    except Exception as exc:
        print(
            "Warning: refined detect curated surface update failed for "
            f"refined_detect_runs/{refined_run_name}/{source_group}: {exc}"
        )
        return

    print(
        "Updated refined detect curated surface: "
        f"refined_detect_runs/{payload['refined_detect_run']} "
        f"(source={payload['source_group']}, rows={payload['rows_materialized']})"
    )


def run_manual_review(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    review_all: bool = False,
    max_frames: Optional[int] = None,
    target_frames: Optional[np.ndarray] = None,
    manual_score: float = 1.0,
    manual_class_id: int = 0,
    use_full_res: bool = False,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: str = "training",
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
    update_curated: bool = True,
    profile_run: Optional[str] = None,
    overwrite_profile: bool = False,
    skip_detection_profile: bool = False,
    registry: Optional[Path] = None,
    sync_registry: bool = True,
) -> None:
    root = open_zarr_group_direct(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or resolve_latest_complete_run_name(
        refined_parent,
    )
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, _, _ = _select_retune_base(refined_run, variant, output_group)

    if "raw_video" not in root:
        raise RuntimeError("Zarr archive is missing raw_video group.")

    if use_full_res:
        raise RuntimeError("Retune uses downsampled frames; omit --use-full-res.")

    images = root["raw_video"]["images_ds"]

    n_frames = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    dense_payload = (
        _load_refined_review_payload(
            root,
            refined_run,
            total_frames=n_frames,
            width=width,
            height=height,
        )
        if variant == "refined"
        else None
    )
    review_axis = _payload_review_axis(dense_payload or {})
    if (
        dense_payload is not None
        and review_axis == "frame"
        and dense_payload["frame_indices"].shape[0] != n_frames
    ):
        raise RuntimeError(
            "Dense refined detect review expects one canonical row per frame. "
            f"Found {dense_payload['frame_indices'].shape[0]} rows for {n_frames} video frames."
        )

    if dense_payload is not None:
        frame_indices = dense_payload["frame_indices"]
        bbox_norm = dense_payload["bbox_norm_coords"]
        scores = dense_payload["confidence_scores"]
        class_ids = dense_payload["class_ids"]
        detection_source_arr = dense_payload["detection_source"]
        review_statuses = dense_payload["status_labels"]
        frame_counts_arr = (review_statuses == "present").astype(np.int32, copy=False)
        review_rows = _select_refined_review_rows(
            dense_payload,
            review_all=review_all,
            target_frames=target_frames,
            max_items=max_frames,
        )
    else:
        frame_indices = np.asarray(base_group["frame_indices"][:], dtype=np.int32)
        bbox_norm = np.asarray(base_group["bbox_norm_coords"][:], dtype=np.float64)
        scores = np.asarray(base_group["scores"][:], dtype=np.float32)
        class_ids = np.asarray(base_group["class_ids"][:], dtype=np.int32)
        detection_source = base_group.get("detection_source")
        detection_source_arr = np.asarray(detection_source[:], dtype=np.int8) if detection_source is not None else None

        frame_counts = base_group.get("frame_counts")
        if frame_counts is None:
            frame_counts_arr = _compute_frame_counts(frame_indices, n_frames)
        else:
            frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
            if frame_counts_arr.shape[0] != n_frames:
                n_frames = min(n_frames, frame_counts_arr.shape[0])
                frame_counts_arr = frame_counts_arr[:n_frames]
        review_frames_default = frame_counts_arr == 0

        if target_frames is not None:
            unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
            review_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
        elif review_all:
            review_frames = np.arange(n_frames, dtype=np.int32)
        else:
            review_frames = np.where(review_frames_default)[0].astype(np.int32, copy=False)

        if max_frames is not None:
            review_frames = review_frames[:max_frames]
        review_rows = review_frames.astype(np.int32, copy=False)

    review_label = "slots" if review_axis == "frame_arena" else "frames"
    if review_rows.size == 0:
        print(f"No {review_label} to review.")
        return

    if dense_payload is not None and review_axis == "frame":
        detections_by_frame: Dict[int, np.ndarray] = {
            int(frame): np.asarray([idx], dtype=np.int32)
            for idx, (frame, status_label, row_bbox) in enumerate(
                zip(frame_indices.tolist(), review_statuses.tolist(), bbox_norm)
            )
            if status_label == "present" and np.all(np.isfinite(row_bbox))
        }
    elif dense_payload is not None:
        detections_by_frame = {}
    else:
        detections_by_frame = _group_by_frame(frame_indices)
    manual_changes: Dict[object, Optional[np.ndarray]] = {}

    if review_axis == "frame_arena":
        arena_ids = np.asarray(dense_payload["arena_ids"], dtype=np.int32).reshape(-1)
        arena_definitions = [
            dict(item)
            for item in np.asarray(dense_payload["arena_definitions"], dtype=object).reshape(-1).tolist()
        ]
        arena_roi_lookup = {
            int(key): tuple(int(value) for value in values)
            for key, values in dict(dense_payload["arena_roi_lookup"]).items()
        }
        print(
            "Arena-aware refined detect review: "
            f"{len(arena_definitions)} arenas, {int(review_rows.shape[0])} review slots"
        )
    else:
        arena_ids = np.empty((0,), dtype=np.int32)
        arena_definitions = []
        arena_roi_lookup: Dict[int, Tuple[int, int, int, int]] = {}

    def _approve_authoritative_refined_detect() -> Dict[str, object]:
        if str(review_state).strip().lower() != "approved":
            return {"attempted": False, "reason": "review_state_not_approved"}
        resolved_zarr_path = Path(zarr_path).expanduser()
        if not resolved_zarr_path.exists():
            return {
                "attempted": False,
                "reason": "zarr_path_unavailable",
                "zarr_path": str(zarr_path),
            }

        from fisheye.cli.palette import ApproveRequest, approve

        envelope = approve(
            ApproveRequest(
                recording=resolved_zarr_path,
                stage="refined_detect",
                run=refined_run_name,
                approved_by=reviewer,
                note=review_notes or "detect review sign-off",
                apply=True,
            )
        )
        return {
            "attempted": True,
            "status": envelope.get("status"),
            "reason_code": envelope.get("reason_code"),
            "run": envelope.get("run"),
            "envelope": envelope,
        }

    def _apply_review_status() -> Dict[str, object]:
        authoritative_approval = _approve_authoritative_refined_detect()
        approval_ok = bool(authoritative_approval.get("attempted")) and (
            str(authoritative_approval.get("status") or "").strip().lower() == "ok"
        )
        if str(review_state).strip().lower() == "approved" and not approval_ok:
            reason = (
                authoritative_approval.get("reason_code")
                or authoritative_approval.get("reason")
                or "unknown"
            )
            print(
                "Refused to set detect_review_status on "
                f"refined_detect_runs/{refined_run_name}: authoritative approval failed ({reason})."
            )
            return {
                "state": "approval_failed",
                "authoritative_approval": authoritative_approval,
            }
        if approval_ok:
            envelope = authoritative_approval.get("envelope")
            approval = envelope.get("approval") if isinstance(envelope, Mapping) else None
            if not isinstance(approval, Mapping):
                approval = {}
            set_authoritative_run(
                refined_parent,
                refined_run_name,
                approved_by=str(approval.get("approved_by") or "unknown"),
                approved_at=str(approval.get("approved_at") or ""),
                git_sha=str(approval.get("git_sha") or ""),
                note=str(approval.get("note") or ""),
            )
        resolved = resolve_detect_review_target(
            root,
            refined_run_name=refined_run_name,
            refined_run=refined_run,
            override_group=variant,
        )
        reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
        payload: Dict[str, object] = {
            "state": review_state,
            "method": review_method,
            "intended_use": review_intended_use,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved_group": resolved.resolved_group,
            "preference_chain": list(resolved.preference_chain),
            "authoritative_approval": authoritative_approval,
        }
        if reviewer_name:
            payload["reviewer"] = reviewer_name
        if review_notes:
            payload["notes"] = review_notes
        refined_run.attrs["detect_review_status"] = payload
        print(f"Set detect_review_status on refined_detect_runs/{refined_run_name}")
        return payload

    def _finalize_detection_profile_if_approved(
        status_payload: Mapping[str, object],
        *,
        profile_target_group: Optional[str],
    ) -> None:
        if skip_detection_profile:
            print("Detection profile: skipped by --skip-detection-profile")
            return
        if str(status_payload.get("state") or "") != "approved" or review_intended_use != "training":
            return
        refined_parent_name = _pick_refined_parent_name(root)
        if refined_parent_name is None:
            print("Detection profile: skipped; refined detect parent not found")
            return
        result = _write_profile_and_sync_registry(
            root=root,
            zarr_path=Path(zarr_path),
            refined_parent_name=refined_parent_name,
            refined_run_name=refined_run_name,
            refined_run=refined_run,
            target_group=profile_target_group,
            resolved_group=str(status_payload.get("resolved_group") or ""),
            profile_run=profile_run,
            overwrite_profile=overwrite_profile,
            registry_path=registry,
            sync_registry=sync_registry,
            dry_run=False,
        )
        print(f"Detection profile: {result.get('status')}")
        print(f"Detection profile run: {result.get('profile_run') or '—'}")
        registry_sync = result.get("registry_sync")
        if isinstance(registry_sync, Mapping):
            print(f"Detection profile registry sync: {registry_sync.get('status')}")

    def _current_review_key() -> object:
        current_row = int(review_rows[idx_pos])
        frame = int(frame_indices[current_row])
        if review_axis == "frame_arena":
            return (frame, int(arena_ids[current_row]))
        return frame

    def _current_frame_and_status() -> Tuple[int, str]:
        current_row = int(review_rows[idx_pos])
        frame = int(frame_indices[current_row])
        if dense_payload is not None:
            status = str(review_statuses[current_row])
        else:
            base_indices = detections_by_frame.get(frame, np.array([], dtype=np.int32))
            status = "missing" if base_indices.size == 0 else "has detection"
        key = _current_review_key()
        if key in manual_changes:
            status = "manual" if manual_changes[key] is not None else "cleared"
        return frame, status

    idx_pos = 0
    current_rect: Optional[Tuple[float, float, float, float]] = None
    selector: Optional[RectangleSelector] = None
    approve_on_exit = False

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)
    base_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--")
    manual_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2)

    base_patch.set_visible(False)
    manual_patch.set_visible(False)

    def load_frame() -> None:
        nonlocal current_rect
        current_row = int(review_rows[idx_pos])
        frame = int(frame_indices[current_row])
        img = images[frame]
        ax.clear()
        ax.imshow(img, cmap="gray")
        ax.set_axis_off()
        ax.add_patch(base_patch)
        ax.add_patch(manual_patch)

        base_patch.set_visible(False)
        manual_patch.set_visible(False)
        current_rect = None

        if review_axis == "frame_arena":
            active_arena_id = int(arena_ids[current_row])
            for arena in arena_definitions:
                arena_id = int(arena["id"])
                x, y, w, h = [int(value) for value in arena["roi_pixels"]]
                arena_patch = Rectangle(
                    (x, y),
                    w,
                    h,
                    fill=False,
                    edgecolor="orange" if arena_id == active_arena_id else "white",
                    linewidth=2.0 if arena_id == active_arena_id else 1.0,
                    linestyle="-" if arena_id == active_arena_id else ":",
                    alpha=0.9 if arena_id == active_arena_id else 0.6,
                )
                ax.add_patch(arena_patch)

            same_frame_rows = np.where(frame_indices == frame)[0].astype(np.int32, copy=False)
            for row_idx in same_frame_rows.tolist():
                if int(row_idx) == current_row:
                    continue
                row_bbox = np.asarray(bbox_norm[int(row_idx)], dtype=np.float64)
                if not np.all(np.isfinite(row_bbox)):
                    continue
                row_status = str(review_statuses[int(row_idx)])
                color = "lime" if row_status == "present" else "gold"
                rect = _norm_to_rect(row_bbox, width, height)
                ax.add_patch(
                    Rectangle(
                        (rect[0], rect[1]),
                        rect[2] - rect[0],
                        rect[3] - rect[1],
                        fill=False,
                        edgecolor=color,
                        linewidth=1.0,
                        linestyle=":",
                        alpha=0.8,
                    )
                )

            row_bbox = np.asarray(bbox_norm[current_row], dtype=np.float64)
            if np.all(np.isfinite(row_bbox)):
                rect = _norm_to_rect(row_bbox, width, height)
                row_status = str(review_statuses[current_row])
                base_patch.set_bounds(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                base_patch.set_edgecolor("cyan" if row_status == "present" else "gold")
                base_patch.set_linestyle("-" if row_status == "present" else "--")
                base_patch.set_visible(True)
        elif dense_payload is not None:
            row_bbox = np.asarray(bbox_norm[current_row], dtype=np.float64)
            if np.all(np.isfinite(row_bbox)):
                rect = _norm_to_rect(row_bbox, width, height)
                row_status = str(review_statuses[current_row])
                base_patch.set_bounds(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                base_patch.set_edgecolor("cyan" if row_status == "present" else "gold")
                base_patch.set_linestyle("-" if row_status == "present" else "--")
                base_patch.set_visible(True)
        else:
            base_indices = detections_by_frame.get(frame, np.array([], dtype=np.int32))
            if base_indices.size > 0:
                best_idx = base_indices[np.argmax(scores[base_indices])]
                rect = _norm_to_rect(bbox_norm[best_idx], width, height)
                base_patch.set_bounds(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                base_patch.set_edgecolor("cyan")
                base_patch.set_linestyle("--")
                base_patch.set_visible(True)

        review_key = _current_review_key()
        if review_key in manual_changes and manual_changes[review_key] is not None:
            manual_rect = _norm_to_rect(np.asarray(manual_changes[review_key], dtype=np.float64), width, height)
            manual_patch.set_bounds(
                manual_rect[0],
                manual_rect[1],
                manual_rect[2] - manual_rect[0],
                manual_rect[3] - manual_rect[1],
            )
            manual_patch.set_visible(True)
            current_rect = manual_rect

        _, status = _current_frame_and_status()

        if selector is not None:
            selector.clear()
            selector.set_visible(False)

        if review_axis == "frame_arena":
            arena_id = int(arena_ids[current_row])
            ax.set_title(
                f"Frame {frame} | Arena {arena_id} | {idx_pos + 1}/{len(review_rows)} | {status}",
                fontsize=10,
            )
        else:
            ax.set_title(
                f"Frame {frame} | {idx_pos + 1}/{len(review_rows)} | {status}",
                fontsize=10,
            )
        fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        nonlocal current_rect
        if eclick.xdata is None or erelease.xdata is None:
            return
        current_rect = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        rect_norm = _rect_to_norm(current_rect, width, height)
        if rect_norm[2] <= 0 or rect_norm[3] <= 0:
            return
        rect_norm[:2] = np.clip(rect_norm[:2], 0.0, 1.0)
        rect_norm[2:] = np.clip(rect_norm[2:], 0.0, 1.0)

        if review_axis == "frame_arena":
            current_row = int(review_rows[idx_pos])
            active_arena_id = int(arena_ids[current_row])
            assigned_arena_id = _bbox_norm_to_arena_id(
                rect_norm,
                arena_definitions=arena_definitions,
                width=width,
                height=height,
            )
            if assigned_arena_id != active_arena_id:
                roi = arena_roi_lookup.get(active_arena_id)
                if roi is None:
                    print(
                        f"Manual box center must fall inside arena {active_arena_id}; selection ignored."
                    )
                else:
                    print(
                        "Manual box center must fall inside the active arena ROI "
                        f"{active_arena_id} ({roi}); selection ignored."
                    )
                return

        manual_patch.set_bounds(
            min(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
            abs(erelease.xdata - eclick.xdata),
            abs(erelease.ydata - eclick.ydata),
        )
        manual_patch.set_visible(True)
        fig.canvas.draw_idle()
        manual_changes[_current_review_key()] = rect_norm
        if selector is not None:
            selector.set_visible(False)

    def on_key(event):
        nonlocal idx_pos, approve_on_exit
        key = (event.key or "").lower()
        if key == "n":
            idx_pos = min(idx_pos + 1, len(review_rows) - 1)
            load_frame()
        elif key == "p":
            idx_pos = max(idx_pos - 1, 0)
            load_frame()
        elif key == "c":
            manual_changes[_current_review_key()] = None
            load_frame()
        elif key == "r":
            review_key = _current_review_key()
            if review_key in manual_changes:
                del manual_changes[review_key]
            load_frame()
        elif key == "a":
            approve_on_exit = True
            plt.close(fig)
        elif key == "q":
            plt.close(fig)
        elif key == "h":
            target_word = "slot" if review_axis == "frame_arena" else "frame"
            print(f"Keys: n=next, p=prev, c=clear detection, r=reset, a=approve, q=quit ({target_word}-aware)")

    # Keep a reference; otherwise the selector can be garbage-collected.
    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        interactive=True,
    )
    selector.set_visible(False)

    fig.canvas.mpl_connect("key_press_event", on_key)

    load_frame()
    print("\nManual review controls:")
    if review_axis == "frame_arena":
        print("  - Drag to draw a bounding box inside the active arena ROI")
        print("  - n / p: next / previous frame-arena slot")
        print("  - c: clear detection for this slot")
        print("  - r: reset changes for this slot")
    else:
        print("  - Drag to draw a bounding box on the current frame")
        print("  - n / p: next / previous frame")
        print("  - c: clear detection for this frame")
        print("  - r: reset changes for this frame")
    print("  - a: approve recording (sets detect_review_status)")
    print("  - q: save changes and quit")
    print("  - h: print this help again")
    plt.show()

    if not manual_changes:
        print("No manual changes recorded.")
        if approve_on_exit:
            status_payload = _apply_review_status()
            _finalize_detection_profile_if_approved(
                status_payload,
                profile_target_group=variant,
            )
        return

    if dense_payload is not None:
        if not update_curated:
            print("Refined runs update the canonical curated surface in place; ignoring --no-update-curated.")

        if output_group != "manual":
            print(
                f"Note: output_group='{output_group}' is ignored for refined runs; "
                "manual edits are written to the canonical curated surface."
            )

        if review_axis == "frame_arena":
            arena_manual_changes = {
                (int(frame), int(arena_id)): rect_norm
                for (frame, arena_id), rect_norm in manual_changes.items()
            }
            updated, added_slots, removed_slots = _apply_manual_changes_to_arena_payload(
                dense_payload,
                manual_changes=arena_manual_changes,
                manual_score=manual_score,
                manual_class_id=manual_class_id,
            )
            slot_to_row = dense_payload["slot_to_row"]
            row_indices = np.asarray(
                sorted(
                    {
                        int(slot_to_row[(int(frame), int(arena_id))])
                        for frame, arena_id in arena_manual_changes.keys()
                    }
                ),
                dtype=np.int32,
            )
            source_context = {
                "editor": "detect_review",
                "edit_mode": "manual",
                "review_axis": "frame_arena",
                "manual_review_slots": int(len(arena_manual_changes)),
                "manual_review_added": int(added_slots),
                "manual_review_removed": int(removed_slots),
            }
        else:
            updated = {
                key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                for key, value in dense_payload.items()
                if key != "frame_to_row"
            }
            frame_to_row = dense_payload["frame_to_row"]
            removed_frames = 0
            added_frames = 0
            for frame, rect_norm in manual_changes.items():
                row_idx = frame_to_row.get(int(frame))
                if row_idx is None:
                    raise RuntimeError(f"Dense refined root is missing frame {frame}.")
                current_source_row_index = int(updated["source_detect_row_index"][row_idx])
                source_surface_row_idx = _resolve_source_surface_row_for_frame(
                    updated,
                    frame=int(frame),
                    preferred_source_detect_row_index=current_source_row_index,
                )
                chosen_source_row_index = (
                    int(updated["source_surface_source_detect_row_index"][source_surface_row_idx])
                    if source_surface_row_idx is not None
                    else -1
                )
                updated["source_detect_row_index"][row_idx] = chosen_source_row_index
                updated["detection_source"][row_idx] = 0
                if rect_norm is None:
                    removed_frames += 1
                    updated["bbox_norm_coords"][row_idx] = np.full((4,), np.nan, dtype=np.float64)
                    updated["confidence_scores"][row_idx] = np.float32(np.nan)
                    updated["class_ids"][row_idx] = np.int32(-1)
                    updated["status_labels"][row_idx] = "filtered_out"
                    updated["source_kind_labels"][row_idx] = "none"
                    updated["manual_edit_flags"][row_idx] = True
                    updated["reason_labels"][row_idx] = "manual_clear"
                    if source_surface_row_idx is not None:
                        updated["source_surface_decision_labels"][source_surface_row_idx] = "manual_clear"
                        updated["source_surface_reason_labels"][source_surface_row_idx] = "manual_clear"
                else:
                    added_frames += 1
                    updated["bbox_norm_coords"][row_idx] = np.asarray(rect_norm, dtype=np.float64)
                    updated["confidence_scores"][row_idx] = np.float32(manual_score)
                    updated["class_ids"][row_idx] = np.int32(manual_class_id)
                    updated["status_labels"][row_idx] = "present"
                    updated["source_kind_labels"][row_idx] = "manual"
                    updated["manual_edit_flags"][row_idx] = True
                    updated["reason_labels"][row_idx] = "manual_correction"
                    if source_surface_row_idx is not None:
                        updated["source_surface_decision_labels"][source_surface_row_idx] = "accepted"
                        updated["source_surface_reason_labels"][source_surface_row_idx] = "manual_correction"

            row_indices = np.asarray(
                sorted({int(frame_to_row[int(frame)]) for frame in manual_changes.keys()}),
                dtype=np.int32,
            )
            source_context = {
                "editor": "detect_review",
                "edit_mode": "manual",
                "manual_review_frames": int(len(manual_changes)),
                "manual_review_added": int(added_frames),
                "manual_review_removed": int(removed_frames),
            }

        _write_dense_curated_edit_payload(
            root,
            zarr_path=zarr_path,
            refined_run_name=refined_run_name,
            payload=updated,
            row_indices=row_indices,
            command_label=" ".join(sys.argv),
            source_context=source_context,
        )
        print(f"Saved manual detections to refined_detect_runs/{refined_run_name} (canonical curated surface)")
        if approve_on_exit:
            status_payload = _apply_review_status()
            print(f"Applied detect_review_status to refined_detect_runs/{refined_run_name}")
            _finalize_detection_profile_if_approved(
                status_payload,
                profile_target_group=variant,
            )
        return

    # Build updated detection arrays
    frames_to_replace = np.array(sorted(manual_changes.keys()), dtype=np.int32)
    keep_mask = ~np.isin(frame_indices, frames_to_replace)

    kept_frames = frame_indices[keep_mask]
    kept_bboxes = bbox_norm[keep_mask]
    kept_scores = scores[keep_mask]
    kept_class_ids = class_ids[keep_mask]
    kept_source = detection_source_arr[keep_mask] if detection_source_arr is not None else None

    manual_frames: List[int] = []
    manual_bboxes: List[np.ndarray] = []
    manual_scores: List[float] = []
    manual_class_ids: List[int] = []
    manual_source: List[int] = []

    removed_frames = 0
    for frame, rect_norm in manual_changes.items():
        if rect_norm is None:
            removed_frames += 1
            continue
        manual_frames.append(int(frame))
        manual_bboxes.append(rect_norm)
        manual_scores.append(float(manual_score))
        manual_class_ids.append(int(manual_class_id))
        manual_source.append(0)

    if manual_frames:
        manual_frames_arr = np.asarray(manual_frames, dtype=np.int32)
        manual_bboxes_arr = np.stack(manual_bboxes, axis=0).astype(np.float64, copy=False)
        manual_scores_arr = np.asarray(manual_scores, dtype=np.float32)
        manual_class_ids_arr = np.asarray(manual_class_ids, dtype=np.int32)
        manual_source_arr = np.asarray(manual_source, dtype=np.int8)
    else:
        manual_frames_arr = np.empty((0,), dtype=np.int32)
        manual_bboxes_arr = np.empty((0, 4), dtype=np.float64)
        manual_scores_arr = np.empty((0,), dtype=np.float32)
        manual_class_ids_arr = np.empty((0,), dtype=np.int32)
        manual_source_arr = np.empty((0,), dtype=np.int8)

    out_frames = np.concatenate([kept_frames, manual_frames_arr])
    out_bboxes = np.concatenate([kept_bboxes, manual_bboxes_arr])
    out_scores = np.concatenate([kept_scores, manual_scores_arr])
    out_class_ids = np.concatenate([kept_class_ids, manual_class_ids_arr])

    if detection_source_arr is not None:
        out_source = np.concatenate([kept_source, manual_source_arr])
    else:
        out_source = None

    reason = None
    if manual_frames_arr.size > 0:
        kept_reason = np.full(kept_frames.shape[0], "kept", dtype=object)
        manual_reason = np.full(manual_frames_arr.shape[0], "manual_correction", dtype=object)
        reason = np.concatenate([kept_reason, manual_reason])

    if out_frames.size > 0:
        order = np.argsort(out_frames)
        out_frames = out_frames[order]
        out_bboxes = out_bboxes[order]
        out_scores = out_scores[order]
        out_class_ids = out_class_ids[order]
        if out_source is not None:
            out_source = out_source[order]
        if reason is not None:
            reason = reason[order]

    out_counts = _compute_frame_counts(out_frames, n_frames)
    out_retune_id = np.full(out_frames.shape[0], -1, dtype=np.int32)

    metadata = {
        "manual_review_timestamp": datetime.now(timezone.utc).isoformat(),
        "manual_review_frames": int(len(manual_changes)),
        "manual_review_added": int(len(manual_frames)),
        "manual_review_removed": int(removed_frames),
        "source_variant": variant,
        "source_refined_run": refined_run_name,
        "source_detect_run": refined_run.attrs.get("source_detect_run"),
        "manual_score": float(manual_score),
        "manual_class_id": int(manual_class_id),
        "detection_source_type": "manual",
        "detection_source_path": f"{refined_run.path}/{output_group}",
    }

    _write_manual_group(
        refined_run,
        output_group,
        out_frames,
        out_bboxes,
        out_scores,
        out_class_ids,
        out_retune_id,
        out_counts,
        out_source,
        reason,
        metadata,
        overwrite,
    )

    refined_run.attrs["manual_review_latest"] = output_group
    print(f"Saved manual detections to refined_detect_runs/{refined_run_name}/{output_group}")
    if update_curated:
        _update_curated_refined_after_manual_write(
            root,
            zarr_path=zarr_path,
            refined_run_name=refined_run_name,
            source_group=output_group,
        )
    if approve_on_exit:
        status_payload = _apply_review_status()
        _finalize_detection_profile_if_approved(
            status_payload,
            profile_target_group=output_group,
        )


def _detect_frames_with_params(
    images_ds: np.ndarray,
    background_ds: np.ndarray,
    frame_indices: np.ndarray,
    detect_params: Dict[str, object],
    dish_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    se1 = disk(int(detect_params["se1_radius"]))
    se4 = disk(int(detect_params["se4_radius"]))
    min_area = int(detect_params["min_area"])
    max_area = int(detect_params["max_area"])
    max_fish = int(detect_params.get("max_fish", 20))
    base_thresh = int(detect_params["ds_thresh"])

    all_frames: List[int] = []
    all_bboxes: List[List[float]] = []

    ds_height = int(images_ds.shape[1])
    ds_width = int(images_ds.shape[2])

    for frame_idx in frame_indices:
        img = images_ds[frame_idx]
        diff_ds = np.clip(
            background_ds.astype(np.int16) - img.astype(np.int16),
            0,
            255,
        ).astype(np.uint8)
        if dish_mask is not None:
            diff_ds = np.where(dish_mask > 0, diff_ds, 0)

        current_thresh = base_thresh
        valid_blobs: List[object] = []
        for _ in range(5):
            im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
            all_blobs = regionprops(label(im_ds))
            valid_blobs = [r for r in all_blobs if min_area <= r.area <= max_area]
            if valid_blobs:
                break
            current_thresh -= 5

        if not valid_blobs:
            continue

        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:max_fish]
        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            center_y, center_x = (min_r + max_r) / 2.0, (min_c + max_c) / 2.0
            height, width = max_r - min_r, max_c - min_c
            center_norm = np.array([center_x / ds_width, center_y / ds_height])
            size_norm = np.array([width / ds_width, height / ds_height])
            all_frames.append(int(frame_idx))
            all_bboxes.append([*center_norm, *size_norm])

    if not all_frames:
        return np.empty((0,), dtype=np.int32), np.empty((0, 4), dtype=np.float64)

    return (
        np.asarray(all_frames, dtype=np.int32),
        np.asarray(all_bboxes, dtype=np.float64),
    )


def _build_retune_dashboard(
    image: np.ndarray,
    background: np.ndarray,
    dish_mask: Optional[np.ndarray],
    detect_params: Dict[str, object],
) -> Tuple[np.ndarray, int, int]:
    ds_thresh = int(detect_params["ds_thresh"])
    se1 = disk(int(detect_params["se1_radius"]))
    se4 = disk(int(detect_params["se4_radius"]))
    min_area = int(detect_params["min_area"])
    max_area = int(detect_params["max_area"])
    max_fish = int(detect_params.get("max_fish", 20))

    diff = np.clip(
        background.astype(np.int16) - image.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)
    if dish_mask is not None:
        diff = np.where(dish_mask > 0, diff, 0)

    used_thresh = ds_thresh
    valid_blobs: List[object] = []
    processed_mask = None
    for _ in range(5):
        processed_mask = erosion(dilation(erosion(diff >= used_thresh, se1), se4), se1)
        all_blobs = regionprops(label(processed_mask))
        valid_blobs = [r for r in all_blobs if min_area <= r.area <= max_area]
        if valid_blobs:
            break
        used_thresh -= 5

    panel_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel_diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    panel_mask = cv2.cvtColor((processed_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    panel_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if valid_blobs:
        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:max_fish]
        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            cv2.rectangle(panel_overlay, (min_c, min_r), (max_c, max_r), (0, 255, 0), 2)

    display_size = (480, 480)
    top_row = np.hstack((cv2.resize(panel_image, display_size), cv2.resize(panel_diff, display_size)))
    bottom_row = np.hstack((cv2.resize(panel_mask, display_size), cv2.resize(panel_overlay, display_size)))
    dashboard = np.vstack((top_row, bottom_row))

    return dashboard, used_thresh, len(valid_blobs)


def run_retune_interactive(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = "pipeline_config.yaml",
    retune_score: float = 1.0,
    retune_class_id: int = 0,
    target_frames: Optional[np.ndarray] = None,
) -> None:
    console = Console()
    root = open_zarr_group_direct(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or resolve_latest_complete_run_name(
        refined_parent,
    )
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, base_name, base_is_manual = _select_retune_base(refined_run, variant, output_group)

    source_detect_run = refined_run.attrs.get("source_detect_run")
    if source_detect_run:
        detect_group = root.get(f"detect_runs/{source_detect_run}")
        if detect_group is not None:
            detection_method = str(detect_group.attrs.get("detection_method", "")).lower()
            pipeline_type = str(detect_group.attrs.get("pipeline_type", "")).lower()
            model_type = str(detect_group.attrs.get("model_type", "")).lower()
            if "yolo" in detection_method or "yolo" in pipeline_type or "yolo" in model_type:
                raise RuntimeError(
                    "Retune is not supported for YOLO detection runs. "
                    "Rerun YOLO detection with updated thresholds/model instead."
                )

    if "raw_video" not in root or "images_ds" not in root["raw_video"]:
        raise RuntimeError("Zarr archive is missing raw_video/images_ds.")

    images = root["raw_video"]["images_ds"]
    n_frames = int(images.shape[0])
    if variant == "refined":
        dense_payload = _load_dense_curated_edit_payload(refined_run, total_frames=n_frames)
        if dense_payload["frame_indices"].shape[0] != n_frames:
            raise RuntimeError(
                "Dense refined detect retune expects one canonical row per frame. "
                f"Found {dense_payload['frame_indices'].shape[0]} rows for {n_frames} video frames."
            )
        frame_counts_arr = (dense_payload["status_labels"] == "present").astype(np.int32, copy=False)
    else:
        frame_counts = base_group.get("frame_counts")
        if frame_counts is None:
            frame_counts_arr = _compute_frame_counts(base_group["frame_indices"][:], n_frames)
        else:
            frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
            if frame_counts_arr.shape[0] != n_frames:
                n_frames = min(n_frames, frame_counts_arr.shape[0])
                frame_counts_arr = frame_counts_arr[:n_frames]

    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        retune_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
    else:
        retune_frames = np.where(frame_counts_arr == 0)[0].astype(np.int32, copy=False)
    if max_frames is not None:
        retune_frames = retune_frames[:max_frames]

    if retune_frames.size == 0:
        print("No frames to retune.")
        return

    import yaml

    config: Dict[str, object] = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    detect_params, param_source = get_detection_parameters(root, config, console)

    if "background_runs" not in root:
        raise RuntimeError("Background stage not run. Run background computation first.")
    latest_bg_run = resolve_latest_complete_run_name(root["background_runs"])
    if not latest_bg_run:
        raise RuntimeError("No background runs found (missing latest background run).")

    background_ds = root[f"background_runs/{latest_bg_run}/background_ds"][:]
    dish_mask = create_dish_mask(detect_params.get("dish_mask", {}), background_ds.shape, console)
    if dish_mask is None:
        console.print("[yellow]No dish mask loaded; using full frame.[/yellow]")
    else:
        console.print("[green]✓ Dish mask loaded for retune UI.[/green]")

    window_name = "Detection Retune (Interactive)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 900)

    current_idx = 0

    def update_idx(val):
        nonlocal current_idx
        current_idx = int(np.clip(val, 0, max(0, len(retune_frames) - 1)))

    def update_param(key: str):
        def _inner(val):
            detect_params[key] = val
        return _inner

    max_failure = len(retune_frames) - 1
    if max_failure >= 1:
        cv2.createTrackbar("Failure", window_name, current_idx, max_failure, update_idx)
    else:
        console.print("[yellow]Single missing frame; failure slider disabled.[/yellow]")
    cv2.createTrackbar("ds_thresh", window_name, int(detect_params["ds_thresh"]), 255, update_param("ds_thresh"))
    cv2.createTrackbar("se1_radius", window_name, int(detect_params["se1_radius"]), 10, update_param("se1_radius"))
    cv2.createTrackbar("se4_radius", window_name, int(detect_params["se4_radius"]), 10, update_param("se4_radius"))
    cv2.createTrackbar("min_area", window_name, int(detect_params["min_area"]), 5000, update_param("min_area"))
    cv2.createTrackbar("max_area", window_name, int(detect_params["max_area"]), 50000, update_param("max_area"))
    cv2.createTrackbar("max_fish", window_name, int(detect_params.get("max_fish", 20)), 50, update_param("max_fish"))

    print("\nDetection Retune (Interactive)")
    print(f"  Zarr: {zarr_path}")
    print(f"  Refined run: {refined_run_name}")
    if base_is_manual:
        print(f"  Base group: {base_name} (incremental)")
    else:
        print(f"  Base group: {base_name}")
    if target_frames is not None:
        print(f"  Targeted frames: {len(retune_frames)}")
    else:
        print(f"  Missing frames: {len(retune_frames)}")
    print("Controls:")
    frame_label = "targeted frames" if target_frames is not None else "missing frames"
    print(f"  - Adjust sliders to test parameters on {frame_label}")
    print("  - Left/Right arrows to step through failures")
    print("  - Press 's' to save tuning to zarr (analysis_metadata)")
    print(f"  - Press 'a' to APPLY retune to remaining {frame_label} (does not update analysis_metadata tuning)")
    print("  - Press 'q' or Esc to quit")

    while True:
        frame_idx = int(retune_frames[current_idx])
        dashboard, used_thresh, blob_count = _build_retune_dashboard(
            images[frame_idx], background_ds, dish_mask, detect_params
        )
        cv2.putText(
            dashboard,
            f"Frame {frame_idx} | Failure {current_idx+1}/{len(retune_frames)} | Used thresh {used_thresh} | blobs {blob_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, dashboard)
        cv2.setTrackbarPos("Failure", window_name, current_idx)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == 83:  # Right arrow
            current_idx = min(len(retune_frames) - 1, current_idx + 1)
        elif key == 81:  # Left arrow
            current_idx = max(0, current_idx - 1)
        elif key == ord("s"):
            _save_detection_tuning(root, detect_params, tuned_frame=frame_idx)
            print("✓ Saved detection tuning to analysis_metadata.")
        elif key == ord("a"):
            cv2.destroyAllWindows()
            run_retune_review(
                zarr_path,
                refined_run_name=refined_run_name,
                variant=variant,
                output_group=output_group,
                overwrite=overwrite,
                max_frames=max_frames,
                config_path=config_path,
                retune_score=retune_score,
                retune_class_id=retune_class_id,
                target_frames=retune_frames,
                use_full_res=False,
                detect_params_override=dict(detect_params),
                param_source_override="retune_ui",
            )
            return

    cv2.destroyAllWindows()

def run_retune_review(
    zarr_path: str,
    refined_run_name: Optional[str] = None,
    variant: Optional[str] = None,
    output_group: str = "manual",
    overwrite: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[str] = "pipeline_config.yaml",
    retune_score: float = 1.0,
    retune_class_id: int = 0,
    target_frames: Optional[np.ndarray] = None,
    use_full_res: bool = False,
    detect_params_override: Optional[Dict[str, object]] = None,
    param_source_override: Optional[str] = None,
) -> None:
    console = Console()
    root = open_zarr_group_direct(zarr_path, mode="a")

    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = refined_run_name or resolve_latest_complete_run_name(
        refined_parent,
    )
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")

    refined_run = refined_parent[refined_run_name]
    variant = _pick_variant(refined_run, variant)
    base_group, _, _ = _select_retune_base(refined_run, variant, output_group)

    source_detect_run = refined_run.attrs.get("source_detect_run")
    if source_detect_run:
        detect_group = root.get(f"detect_runs/{source_detect_run}")
        if detect_group is not None:
            detection_method = str(detect_group.attrs.get("detection_method", "")).lower()
            pipeline_type = str(detect_group.attrs.get("pipeline_type", "")).lower()
            model_type = str(detect_group.attrs.get("model_type", "")).lower()
            if "yolo" in detection_method or "yolo" in pipeline_type or "yolo" in model_type:
                raise RuntimeError(
                    "Retune is not supported for YOLO detection runs. "
                    "Rerun YOLO detection with updated thresholds/model instead."
                )

    if "raw_video" not in root:
        raise RuntimeError("Zarr archive is missing raw_video group.")

    if use_full_res or "images_ds" not in root["raw_video"]:
        images = root["raw_video"]["images_full"]
    else:
        images = root["raw_video"]["images_ds"]

    n_frames = int(images.shape[0])
    dense_payload = (
        _load_dense_curated_edit_payload(refined_run, total_frames=n_frames)
        if variant == "refined"
        else None
    )
    if dense_payload is not None:
        if dense_payload["frame_indices"].shape[0] != n_frames:
            raise RuntimeError(
                "Dense refined detect retune expects one canonical row per frame. "
                f"Found {dense_payload['frame_indices'].shape[0]} rows for {n_frames} video frames."
            )
        frame_counts_arr = (dense_payload["status_labels"] == "present").astype(np.int32, copy=False)
    else:
        frame_counts = base_group.get("frame_counts")
        if frame_counts is None:
            frame_counts_arr = _compute_frame_counts(base_group["frame_indices"][:], n_frames)
        else:
            frame_counts_arr = np.asarray(frame_counts[:], dtype=np.int32)
            if frame_counts_arr.shape[0] != n_frames:
                n_frames = min(n_frames, frame_counts_arr.shape[0])
                frame_counts_arr = frame_counts_arr[:n_frames]

    missing_before = int(np.sum(frame_counts_arr == 0))
    if target_frames is not None:
        unique_frames = np.unique(target_frames.astype(np.int32, copy=False))
        retune_frames = unique_frames[(unique_frames >= 0) & (unique_frames < n_frames)]
    else:
        retune_frames = np.where(frame_counts_arr == 0)[0].astype(np.int32, copy=False)
    if max_frames is not None:
        retune_frames = retune_frames[:max_frames]

    if retune_frames.size == 0:
        print("No frames to retune.")
        return

    # Resolve detection parameters (config defaults + zarr tuning), unless overridden.
    if detect_params_override is None:
        import yaml

        config: Dict[str, object] = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

        detect_params, param_source = get_detection_parameters(root, config, console)
    else:
        detect_params = detect_params_override
        param_source = param_source_override or "override"

    if "background_runs" not in root:
        raise RuntimeError("Background stage not run. Run background computation first.")
    latest_bg_run = resolve_latest_complete_run_name(root["background_runs"])
    if not latest_bg_run:
        raise RuntimeError("No background runs found (missing latest background run).")

    background_ds = root[f"background_runs/{latest_bg_run}/background_ds"][:]
    dish_mask = create_dish_mask(detect_params.get("dish_mask", {}), background_ds.shape, console)

    retune_id = _get_or_create_retune_id(refined_run, detect_params)

    if target_frames is not None:
        console.print(f"[cyan]Retuning frames:[/cyan] {len(retune_frames)} targeted")
    else:
        console.print(f"[cyan]Retuning frames:[/cyan] {len(retune_frames)} missing detections")
    console.print(f"[cyan]Retune base:[/cyan] {base_group.path}")
    console.print(f"[cyan]Using parameters from:[/cyan] {param_source}")

    new_frames, new_bboxes = _detect_frames_with_params(
        images,
        background_ds,
        retune_frames,
        detect_params,
        dish_mask,
    )

    if dense_payload is not None:
        if output_group != "manual":
            print(
                f"Note: output_group='{output_group}' is ignored for refined runs; "
                "retune updates are written to the canonical curated surface."
            )

        updated = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in dense_payload.items()
            if key != "frame_to_row"
        }
        frame_to_row = dense_payload["frame_to_row"]
        new_frame_to_bbox = {
            int(frame): np.asarray(bbox, dtype=np.float64)
            for frame, bbox in zip(new_frames.tolist(), new_bboxes)
        }

        for frame in retune_frames.tolist():
            row_idx = frame_to_row.get(int(frame))
            if row_idx is None:
                raise RuntimeError(f"Dense refined root is missing frame {frame}.")
            current_source_row_index = int(updated["source_detect_row_index"][row_idx])
            source_surface_row_idx = _resolve_source_surface_row_for_frame(
                updated,
                frame=int(frame),
                preferred_source_detect_row_index=current_source_row_index,
            )
            chosen_source_row_index = (
                int(updated["source_surface_source_detect_row_index"][source_surface_row_idx])
                if source_surface_row_idx is not None
                else -1
            )
            bbox = new_frame_to_bbox.get(int(frame))
            updated["source_detect_row_index"][row_idx] = chosen_source_row_index
            updated["detection_source"][row_idx] = 0
            if bbox is None:
                continue
            updated["bbox_norm_coords"][row_idx] = bbox
            updated["confidence_scores"][row_idx] = np.float32(retune_score)
            updated["class_ids"][row_idx] = np.int32(retune_class_id)
            updated["status_labels"][row_idx] = "present"
            updated["source_kind_labels"][row_idx] = "manual"
            updated["manual_edit_flags"][row_idx] = True
            updated["reason_labels"][row_idx] = "retune"
            if source_surface_row_idx is not None:
                updated["source_surface_decision_labels"][source_surface_row_idx] = "accepted"
                updated["source_surface_reason_labels"][source_surface_row_idx] = "retune"

        missing_after = int(np.sum(updated["status_labels"] != "present"))
        remaining_targeted = int(
            np.sum(updated["status_labels"][retune_frames.astype(np.int32, copy=False)] != "present")
        ) if retune_frames.size else 0
        corrected = missing_before - missing_after

        _write_dense_curated_edit_payload(
            root,
            zarr_path=zarr_path,
            refined_run_name=refined_run_name,
            payload=updated,
            row_indices=np.asarray(
                sorted({int(frame_to_row[int(frame)]) for frame in retune_frames.tolist()}),
                dtype=np.int32,
            ),
            command_label=" ".join(sys.argv),
            source_context={
                "editor": "detect_review",
                "edit_mode": "retune",
                "retune_frames_requested": int(len(retune_frames)),
                "retune_detections_added": int(new_frames.shape[0]),
                "retune_parameter_source": param_source,
            },
        )

        print(f"Saved retuned detections to refined_detect_runs/{refined_run_name} (canonical curated surface)")
        console.print("[green]Retune summary:[/green]")
        console.print(f"  Missing before: {missing_before}")
        console.print(f"  Retune detections added: {int(new_frames.shape[0])}")
        console.print(f"  Targeted this run: {len(retune_frames)}")
        console.print(f"  Corrected this run: {corrected}")
        console.print(f"  Still missing (targeted): {remaining_targeted}")
        console.print(f"  Missing after (overall): {missing_after}")
        return

    # Build updated detection arrays
    frame_indices = np.asarray(base_group["frame_indices"][:], dtype=np.int32)
    bbox_norm = np.asarray(base_group["bbox_norm_coords"][:], dtype=np.float64)
    scores = np.asarray(base_group["scores"][:], dtype=np.float32)
    class_ids = np.asarray(base_group["class_ids"][:], dtype=np.int32)
    detection_source = base_group.get("detection_source")
    detection_source_arr = np.asarray(detection_source[:], dtype=np.int8) if detection_source is not None else None
    base_retune_id = base_group.get("retune_id")
    base_retune_id_arr = np.asarray(base_retune_id[:], dtype=np.int32) if base_retune_id is not None else None
    base_reason_arr = read_reason_labels(base_group)

    frames_to_replace = retune_frames
    keep_mask = ~np.isin(frame_indices, frames_to_replace)

    kept_frames = frame_indices[keep_mask]
    kept_bboxes = bbox_norm[keep_mask]
    kept_scores = scores[keep_mask]
    kept_class_ids = class_ids[keep_mask]
    kept_source = detection_source_arr[keep_mask] if detection_source_arr is not None else None

    if new_frames.size > 0:
        retune_scores = np.full((new_frames.shape[0],), float(retune_score), dtype=np.float32)
        retune_class_ids = np.full((new_frames.shape[0],), int(retune_class_id), dtype=np.int32)
        retune_source = np.zeros((new_frames.shape[0],), dtype=np.int8)
        retune_ids = np.full((new_frames.shape[0],), int(retune_id), dtype=np.int32)
    else:
        retune_scores = np.empty((0,), dtype=np.float32)
        retune_class_ids = np.empty((0,), dtype=np.int32)
        retune_source = np.empty((0,), dtype=np.int8)
        retune_ids = np.empty((0,), dtype=np.int32)

    out_frames = np.concatenate([kept_frames, new_frames])
    out_bboxes = np.concatenate([kept_bboxes, new_bboxes])
    out_scores = np.concatenate([kept_scores, retune_scores])
    out_class_ids = np.concatenate([kept_class_ids, retune_class_ids])
    kept_retune_id = (
        base_retune_id_arr[keep_mask]
        if base_retune_id_arr is not None
        else np.full(kept_frames.shape[0], -1, dtype=np.int32)
    )
    out_retune_id = np.concatenate([kept_retune_id, retune_ids])

    if detection_source_arr is not None:
        out_source = np.concatenate([kept_source, retune_source])
    else:
        out_source = None

    reason = None
    if out_frames.size > 0:
        if base_reason_arr is not None:
            kept_reason = base_reason_arr[keep_mask]
        else:
            kept_reason = np.full(kept_frames.shape[0], "kept", dtype=object)
        retune_reason = np.full(new_frames.shape[0], "retune", dtype=object)
        reason = np.concatenate([kept_reason, retune_reason])

    if out_frames.size > 0:
        order = np.argsort(out_frames)
        out_frames = out_frames[order]
        out_bboxes = out_bboxes[order]
        out_scores = out_scores[order]
        out_class_ids = out_class_ids[order]
        out_retune_id = out_retune_id[order]
        if out_source is not None:
            out_source = out_source[order]
        if reason is not None:
            reason = reason[order]

    out_counts = _compute_frame_counts(out_frames, n_frames)
    missing_after = int(np.sum(out_counts == 0))
    remaining_targeted = int(np.sum(out_counts[retune_frames] == 0)) if retune_frames.size else 0
    corrected = missing_before - missing_after

    metadata = {
        "retune_timestamp": datetime.now(timezone.utc).isoformat(),
        "retune_frames_requested": int(len(retune_frames)),
        "retune_detections_added": int(new_frames.shape[0]),
        "source_variant": variant,
        "source_refined_run": refined_run_name,
        "source_detect_run": refined_run.attrs.get("source_detect_run"),
        "retune_score": float(retune_score),
        "retune_class_id": int(retune_class_id),
        "retune_parameters": detect_params,
        "retune_parameter_source": param_source,
        "retune_base_group": base_group.path,
        "detection_source_type": "retune",
        "detection_source_path": f"{refined_run.path}/{output_group}",
    }

    _write_manual_group(
        refined_run,
        output_group,
        out_frames,
        out_bboxes,
        out_scores,
        out_class_ids,
        out_retune_id,
        out_counts,
        out_source,
        reason,
        metadata,
        overwrite,
    )

    refined_run.attrs["manual_review_latest"] = output_group
    print(f"Saved retuned detections to refined_detect_runs/{refined_run_name}/{output_group}")
    console.print("[green]Retune summary:[/green]")
    console.print(f"  Missing before: {missing_before}")
    console.print(f"  Retune detections added: {int(new_frames.shape[0])}")
    console.print(f"  Targeted this run: {len(retune_frames)}")
    console.print(f"  Corrected this run: {corrected}")
    console.print(f"  Still missing (targeted): {remaining_targeted}")
    console.print(f"  Missing after (overall): {missing_after}")

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manual review for refined detection runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr directory.")
    parser.add_argument(
        "--refined-run",
        help="Refined detect run to review (defaults to latest).",
    )
    parser.add_argument(
        "--variant",
        choices=["filtered", "interpolated", "refined"],
        help="Legacy sparse variant to review, or 'refined' for the canonical curated refined surface (default: refined when available).",
    )
    parser.add_argument(
        "--output-group",
        default="manual",
        help="Legacy sparse output subgroup name (ignored for canonical refined-surface edits).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output group.")
    parser.add_argument("--all", action="store_true", help="Review all frames (not just missing detections).")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit number of frames to review (useful for quick passes).",
    )
    parser.add_argument(
        "--manual-score",
        type=float,
        default=1.0,
        help="Score value to assign to manual detections.",
    )
    parser.add_argument(
        "--manual-class-id",
        type=int,
        default=0,
        help="Class id to assign to manual detections.",
    )
    parser.add_argument(
        "--use-full-res",
        action="store_true",
        help="Use full-resolution frames instead of downsampled.",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Retune missing detections using tuned parameters.",
    )
    parser.add_argument(
        "--retune-ui",
        action="store_true",
        help="Interactive retune (slider UI) for missing detections.",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.yaml",
        help="Config file to source defaults for retune (default: pipeline_config.yaml).",
    )
    parser.add_argument(
        "--retune-score",
        type=float,
        default=1.0,
        help="Score to assign to retuned detections (default: 1.0).",
    )
    parser.add_argument(
        "--retune-class-id",
        type=int,
        default=0,
        help="Class id to assign to retuned detections (default: 0).",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when approving in manual review (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label for review status (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    parser.add_argument(
        "--profile-run",
        help=(
            "Optional detection-profile run name to write when approving training labels. "
            "By default a timestamped run name is generated."
        ),
    )
    parser.add_argument(
        "--overwrite-profile",
        action="store_true",
        help="Allow replacing an existing detection-profile run named by --profile-run.",
    )
    parser.add_argument(
        "--skip-detection-profile",
        action="store_true",
        help="Do not materialize analysis/detection_profile_runs when approving for training.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional registry SQLite path for automatic detection-profile sync. "
            "Defaults to PALETTE_REGISTRY_PATH/config only when that path already exists."
        ),
    )
    parser.add_argument(
        "--no-registry-sync",
        action="store_true",
        help="Write the Zarr detection profile but do not sync registry projection rows.",
    )
    parser.add_argument(
        "--no-update-curated",
        action="store_true",
        help="Skip updating the canonical refined-detect root arrays after manual saves.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Comma/space-separated frame indices (or path to a text/JSON list) to target.",
    )

    args = parser.parse_args(argv)
    target_frames = _parse_frames_arg(args.frames)
    if args.frames and args.all:
        print("Note: --frames provided; ignoring --all.")

    if args.retune_ui:
        run_retune_interactive(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            max_frames=args.max_frames,
            config_path=args.config,
            retune_score=args.retune_score,
            retune_class_id=args.retune_class_id,
            target_frames=target_frames,
        )
    elif args.retune:
        run_retune_review(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            max_frames=args.max_frames,
            config_path=args.config,
            retune_score=args.retune_score,
            retune_class_id=args.retune_class_id,
            target_frames=target_frames,
            use_full_res=args.use_full_res,
        )
    else:
        run_manual_review(
            str(args.zarr_path),
            refined_run_name=args.refined_run,
            variant=args.variant,
            output_group=args.output_group,
            overwrite=args.overwrite,
            review_all=args.all,
            max_frames=args.max_frames,
            target_frames=target_frames,
            manual_score=args.manual_score,
            manual_class_id=args.manual_class_id,
            use_full_res=args.use_full_res,
            review_state=args.review_state,
            review_method=args.review_method,
            review_intended_use=args.review_intended_use,
            reviewer=args.reviewer,
            review_notes=args.review_notes,
            update_curated=not args.no_update_curated,
            profile_run=args.profile_run,
            overwrite_profile=args.overwrite_profile,
            skip_detection_profile=args.skip_detection_profile,
            registry=args.registry,
            sync_registry=not args.no_registry_sync,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
