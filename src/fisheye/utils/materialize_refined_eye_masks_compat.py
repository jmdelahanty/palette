"""Materialize refined-eye compatibility runs from canonical refined subject masks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..pose.schema import resolve_required_keypoint_indices_from_attrs
from ..refinement.refine_eye_masks import _compute_roi_metrics, _measure_mask, _write_contours_from_masks
from ..shared.detect_reason_codec import read_reason_labels, write_reason_columns
from ..shared.provenance_attrs import build_source_keypoints_attrs, resolve_source_keypoints_run
from ..shared.row_lineage import copy_row_lineage_arrays
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr_run_completion import mark_run_complete, mark_run_started, note_pending_latest, require_runs_parent
from ..utils.refined_eye_masks_compat import (
    DERIVED_REFINED_EYE_MASKS_COMPAT_ROLE as DERIVED_COMPAT_ROLE,
)
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root


MATERIALIZE_REFINED_EYE_MASKS_COMPAT_METHOD = "materialize_refined_eye_masks_compat"
MATERIALIZE_REFINED_EYE_MASKS_STAGE = "materialize_refined_eye_masks_compat"
DERIVED_REFINED_EYE_MASKS_COMPAT_ROLE = DERIVED_COMPAT_ROLE
EYE_COMPONENTS = ("eye_left", "eye_right")
_EYE_KEYPOINT_LABELS = ("eye_left", "eye_right")
_TRIVIAL_REASON_TAGS = {"clean", "copied_from_source"}
_CARRYOVER_ATTR_KEYS = (
    "source_eye_masks_run",
    "source_refined_eye_masks_run",
    "source_eye_masks_method",
    "refine_stats",
    "summary_statistics",
    "smoothing",
    "mask_probabilities_available",
    "mask_probability_threshold",
    "mask_probability_threshold_source",
    "mask_probability_source",
    "refined_probabilities_written",
    "mask_probability_policy",
    "mask_binary_source",
    "mask_binary_identity",
    "mask_probability_identity",
    "mask_bundle_provenance",
    "ellipse_fit_backend",
    "ellipse_fit_method",
    "ellipse_contour_mode",
    "success_min_eye_area_px",
    "min_eye_area_success_px",
    "traditional_fast_path",
    "force_refine_traditional",
    "dask_scheduler",
    "dask_num_workers",
    "dask_chunk_size",
    "dask_version",
    "source_eye_masks_provenance",
    "source_keypoint_stale",
    "source_eye_labels",
    "hostname",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_latest_run(parent: Optional[zarr.Group], label: str) -> str:
    if parent is None:
        raise RuntimeError(f"No {label} found in archive.")
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    keys = sorted(str(key) for key in parent.keys())
    if not keys:
        raise RuntimeError(f"No runs found under {label}.")
    return keys[-1]


def _open_root_from_group(group: zarr.Group, *, mode: str = "a") -> zarr.Group:
    store = getattr(group, "store", None)
    if store is None:
        raise RuntimeError("Unable to resolve root group from nested zarr group without a store.")
    try:
        return zarr.open_group(store=store, mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(store, mode=mode)


def _open_root(source: str | Path | zarr.Group, *, mode: str = "a") -> zarr.Group:
    if isinstance(source, zarr.Group):
        return _open_root_from_group(source, mode=mode)
    return open_zarr_root(source, mode=mode)


def _get_component_provenance(group: zarr.Group, component_name: str) -> Mapping[str, object]:
    components_parent = group.get("components")
    if components_parent is None:
        return {}
    component_group = components_parent.get(component_name)
    if component_group is None:
        return {}
    provenance_group = component_group.get("provenance")
    if provenance_group is None or not hasattr(provenance_group, "attrs"):
        return {}
    return dict(provenance_group.attrs)


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        return {}
    return {str(name): idx for idx, name in enumerate(labels_raw)}


def _available_channel_mask(group: zarr.Group, *, expected_count: int) -> np.ndarray:
    available = group.get("available_channels")
    if available is None:
        return np.ones((expected_count,), dtype=bool)
    arr = np.asarray(available[:], dtype=bool).reshape(-1)
    if int(arr.shape[0]) < expected_count:
        out = np.zeros((expected_count,), dtype=bool)
        out[: int(arr.shape[0])] = arr
        return out
    return arr[:expected_count]


def _resolve_eye_component_source_subject_runs(
    root: zarr.Group,
    refined_group: zarr.Group,
) -> tuple[dict[str, Optional[zarr.Group]], Optional[str]]:
    subject_parent = root.get("subject_mask_runs")
    coarse_run = _normalize_text(refined_group.attrs.get("source_subject_mask_run"))
    coarse_group = (
        subject_parent[coarse_run]
        if subject_parent is not None and coarse_run is not None and coarse_run in subject_parent
        else None
    )

    resolved: dict[str, Optional[zarr.Group]] = {}
    for component_name in EYE_COMPONENTS:
        provenance = _get_component_provenance(refined_group, component_name)
        source_stage = _normalize_text(provenance.get("source_stage"))
        source_run = _normalize_text(provenance.get("source_run"))
        if (
            source_stage == "subject_mask_runs"
            and source_run is not None
            and subject_parent is not None
            and source_run in subject_parent
        ):
            resolved[component_name] = subject_parent[source_run]
        else:
            resolved[component_name] = coarse_group
    return resolved, coarse_run


def _resolve_source_eye_runs(
    refined_group: zarr.Group,
    component_source_groups: Mapping[str, Optional[zarr.Group]],
    carryover_attrs: Mapping[str, object],
) -> tuple[Optional[str], Optional[str]]:
    raw_eye_run = None
    refined_eye_run = None
    for component_name in EYE_COMPONENTS:
        provenance = _get_component_provenance(refined_group, component_name)
        source_stage = _normalize_text(provenance.get("source_stage"))
        source_run = _normalize_text(provenance.get("source_run"))
        if source_stage == "eye_masks_runs" and source_run is not None and raw_eye_run is None:
            raw_eye_run = source_run
        if source_stage == "refined_eye_masks_runs" and source_run is not None and refined_eye_run is None:
            refined_eye_run = source_run

    for group in component_source_groups.values():
        if group is None:
            continue
        if raw_eye_run is None:
            raw_eye_run = _normalize_text(group.attrs.get("source_eye_masks_run"))
        if refined_eye_run is None:
            refined_eye_run = _normalize_text(group.attrs.get("source_refined_eye_masks_run"))

    if raw_eye_run is None:
        raw_eye_run = _normalize_text(carryover_attrs.get("source_eye_masks_run"))
    if refined_eye_run is None:
        refined_eye_run = _normalize_text(carryover_attrs.get("source_refined_eye_masks_run"))
    return raw_eye_run, refined_eye_run


def _derive_refined_eye_run_from_raw(raw_eye_run: str) -> str:
    if raw_eye_run.startswith("eye_masks_"):
        return f"refined_{raw_eye_run}"
    return f"refined_eye_masks_from_{raw_eye_run}"


def _derive_refined_eye_run_from_refined_subject(refined_subject_run: str) -> str:
    if refined_subject_run.startswith("refined_subject_masks_from_refined_eye_masks_"):
        return refined_subject_run[len("refined_subject_masks_from_") :]
    if refined_subject_run.startswith("refined_subject_masks_from_eye_masks_"):
        suffix = refined_subject_run[len("refined_subject_masks_from_") :]
        return f"refined_{suffix}"
    if refined_subject_run.startswith("refined_subject_masks_"):
        suffix = refined_subject_run[len("refined_subject_masks_") :]
        return f"refined_eye_masks_{suffix}"
    return f"refined_eye_masks_from_{refined_subject_run}"


def _resolve_target_run_name(
    refined_subject_run: str,
    refined_group: zarr.Group,
    *,
    raw_eye_run: Optional[str],
    source_refined_eye_run: Optional[str],
    explicit_target: Optional[str],
) -> str:
    if explicit_target is not None:
        return str(explicit_target)
    compat_run = _normalize_text(refined_group.attrs.get("compat_refined_eye_masks_run"))
    if compat_run is not None:
        return compat_run
    if source_refined_eye_run is not None:
        return source_refined_eye_run
    if raw_eye_run is not None:
        return _derive_refined_eye_run_from_raw(raw_eye_run)
    return _derive_refined_eye_run_from_refined_subject(refined_subject_run)


def _load_subject_component_masks(
    group: Optional[zarr.Group],
    component_name: str,
    *,
    total_rois: int,
    height: int,
    width: int,
) -> np.ndarray:
    empty = np.zeros((total_rois, height, width), dtype=np.uint8)
    if group is None or "masks_roi" not in group:
        return empty
    label_map = _label_index_map(group)
    if component_name not in label_map:
        return empty
    component_idx = int(label_map[component_name])
    available = _available_channel_mask(group, expected_count=max(len(label_map), component_idx + 1))
    if component_idx >= int(available.shape[0]) or not bool(available[component_idx]):
        return empty
    return np.asarray(group["masks_roi"][:, component_idx], dtype=np.uint8)


def _load_keypoints_roi(
    root: zarr.Group,
    refined_group: zarr.Group,
    *,
    total_rois: int,
) -> tuple[Optional[np.ndarray], Optional[dict[str, int]]]:
    keypoint_run = _normalize_text(resolve_source_keypoints_run(refined_group.attrs))
    keypoint_group = _normalize_text(refined_group.attrs.get("source_keypoint_group"))
    candidates = [keypoint_group] if keypoint_group else []
    for fallback in ("refined_keypoints_runs", "keypoints_runs"):
        if fallback not in candidates:
            candidates.append(fallback)

    if keypoint_run is None:
        return None, None
    for parent_name in candidates:
        parent = root.get(parent_name)
        if parent is None or keypoint_run not in parent:
            continue
        keypoints_group = parent[keypoint_run]
        keypoints_roi = keypoints_group.get("keypoints_roi")
        if keypoints_roi is None:
            continue
        keypoints = np.asarray(keypoints_roi[:], dtype=np.float32)
        if keypoints.ndim < 3 or int(keypoints.shape[0]) != total_rois:
            continue
        try:
            indices = resolve_required_keypoint_indices_from_attrs(
                keypoints_group.attrs,
                _EYE_KEYPOINT_LABELS,
                keypoint_count=int(keypoints.shape[1]),
            )
        except ValueError:
            continue
        return keypoints, {name: int(value) for name, value in indices.items()}
    return None, None


def _read_component_reason_labels(
    refined_group: zarr.Group,
    component_name: str,
    *,
    total_rois: int,
) -> np.ndarray:
    components_parent = refined_group.get("components")
    if components_parent is None:
        return np.asarray([""] * total_rois, dtype=object)
    component_group = components_parent.get(component_name)
    if component_group is None:
        return np.asarray([""] * total_rois, dtype=object)
    labels = read_reason_labels(component_group)
    if labels is None:
        return np.asarray([""] * total_rois, dtype=object)
    arr = np.asarray(labels, dtype=object).reshape(-1)
    if int(arr.shape[0]) != total_rois:
        return np.asarray([""] * total_rois, dtype=object)
    return np.asarray([str(value) if value is not None else "" for value in arr], dtype=object)


def _reason_tags(value: object) -> list[str]:
    text = _normalize_text(value)
    if text is None:
        return []
    return [tag.strip() for tag in text.split("|") if tag.strip()]


def _aggregate_pair_reason_label(left_label: object, right_label: object) -> str:
    ordered_tags: list[str] = []
    seen: set[str] = set()
    for value in (left_label, right_label):
        for tag in _reason_tags(value):
            if tag in seen:
                continue
            seen.add(tag)
            ordered_tags.append(tag)
    nontrivial = [tag for tag in ordered_tags if tag not in _TRIVIAL_REASON_TAGS]
    if nontrivial:
        return "|".join(nontrivial)
    if "copied_from_source" in seen:
        return "copied_from_source"
    if "clean" in seen:
        return "clean"
    return ""


def _nan_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def _load_eye_masks_and_edit_state(refined_group: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    label_map = _label_index_map(refined_group)
    missing = [name for name in EYE_COMPONENTS if name not in label_map]
    if missing:
        raise RuntimeError(
            "refined_subject_masks run does not expose canonical left/right eye components: "
            f"missing {', '.join(missing)}."
        )
    masks_roi = np.asarray(refined_group["masks_roi"][:], dtype=np.uint8)
    eye_masks = np.stack(
        [np.asarray(masks_roi[:, label_map[name]], dtype=np.uint8) for name in EYE_COMPONENTS],
        axis=1,
    )

    edit_applied_arr = refined_group.get("edit_applied")
    if edit_applied_arr is None:
        edit_applied = np.zeros((int(eye_masks.shape[0]), len(EYE_COMPONENTS)), dtype=bool)
    else:
        edit_full = np.asarray(edit_applied_arr[:], dtype=bool)
        edit_applied = np.stack(
            [np.asarray(edit_full[:, label_map[name]], dtype=bool) for name in EYE_COMPONENTS],
            axis=1,
        )
    return eye_masks, edit_applied


def _empty_summary_statistics(carryover_attrs: Mapping[str, object]) -> dict[str, object]:
    existing = carryover_attrs.get("summary_statistics")
    if isinstance(existing, dict):
        return dict(existing)
    refine_stats = carryover_attrs.get("refine_stats")
    if isinstance(refine_stats, dict):
        return {"refine": dict(refine_stats)}
    return {}


def _build_metrics_summary(
    *,
    area_refined: np.ndarray,
    area_union_refined: np.ndarray,
    centroid_error: np.ndarray,
    symmetry_sum: np.ndarray,
    separation_delta: np.ndarray,
    ellipse_success: np.ndarray,
    pair_reason_labels: np.ndarray,
    success_min_eye_area_px: Optional[float],
) -> tuple[dict[str, object], dict[str, int]]:
    reason_strings = np.asarray(pair_reason_labels, dtype=object).astype(str).tolist()
    reason_tag_counts = dict(
        Counter(
            tag
            for item in reason_strings
            for tag in (str(item).split("|") if item else ["<empty>"])
            if tag
        )
    )
    centroid_mean = (
        np.nanmean(centroid_error, axis=0)
        if np.isfinite(centroid_error).any()
        else np.full(2, np.nan, dtype=np.float32)
    )
    metrics_summary: dict[str, object] = {
        "num_rois": int(area_refined.shape[0]),
        "area_refined_mean": np.nanmean(area_refined, axis=0).astype(np.float32).tolist() if area_refined.size else [0.0, 0.0],
        "area_refined_std": np.nanstd(area_refined, axis=0).astype(np.float32).tolist() if area_refined.size else [0.0, 0.0],
        "area_union_mean": float(np.nanmean(area_union_refined)) if np.isfinite(area_union_refined).any() else float("nan"),
        "area_union_std": float(np.nanstd(area_union_refined)) if np.isfinite(area_union_refined).any() else float("nan"),
        "centroid_error_mean": centroid_mean.astype(np.float32).tolist(),
        "symmetry_sum_mean": float(np.nanmean(symmetry_sum)) if np.isfinite(symmetry_sum).any() else float("nan"),
        "separation_delta_mean": float(np.nanmean(separation_delta)) if np.isfinite(separation_delta).any() else float("nan"),
        "probability_mean_mean": [float("nan"), float("nan")],
        "probability_threshold": None,
        "reason_counts": dict(Counter(reason_strings)),
        "reason_tag_counts": reason_tag_counts,
        "probability_usage_rate": 0.0,
        "filtered_left": 0,
        "filtered_right": 0,
        "filtered_roi_pairs": 0,
        "ellipse_fail_left": int((~ellipse_success[:, 0]).sum()),
        "ellipse_fail_right": int((~ellipse_success[:, 1]).sum()),
        "ellipse_fail_pairs": int(np.any(~ellipse_success, axis=1).sum()),
        "small_area_left": 0,
        "small_area_right": 0,
        "small_area_roi_pairs": 0,
        "area_filter_threshold": None,
        "area_filter_mode": None,
        "success_min_eye_area_px": float(success_min_eye_area_px) if success_min_eye_area_px is not None else None,
    }
    return metrics_summary, reason_tag_counts


def _parse_timestamp(value: object) -> float:
    text = _normalize_text(value)
    if text is None:
        return float("-inf")
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return float("-inf")


def _aggregate_eye_review_status(
    refined_subject_run: str,
    refined_group: zarr.Group,
    *,
    target_run: str,
    source_subject_run: Optional[str],
    source_eye_masks_run: Optional[str],
    source_refined_eye_run: Optional[str],
    source_keypoints_run: Optional[str],
    source_keypoint_group: Optional[str],
) -> dict[str, object]:
    component_reviews = refined_group.attrs.get("component_review_statuses")
    if not isinstance(component_reviews, dict):
        component_reviews = {}

    eye_payloads = [
        dict(component_reviews.get(component_name) or {})
        for component_name in EYE_COMPONENTS
        if isinstance(component_reviews.get(component_name), dict)
    ]
    states = [str(payload.get("state") or "pending") for payload in eye_payloads]
    if not states:
        state = "pending"
    elif any(item == "rejected" for item in states):
        state = "rejected"
    elif any(item == "needs_review" for item in states):
        state = "needs_review"
    elif all(item == "approved" for item in states):
        state = "approved"
    elif any(item == "pending" for item in states):
        state = "pending"
    else:
        state = states[0]

    reference = max(
        eye_payloads or [{}],
        key=lambda payload: _parse_timestamp(payload.get("timestamp_utc") or payload.get("timestamp")),
    )
    timestamp_utc = _utc_now()
    payload: dict[str, object] = {
        "state": state,
        "method": str(reference.get("method") or "manual"),
        "intended_use": str(reference.get("intended_use") or "training"),
        "timestamp_utc": timestamp_utc,
        "timestamp": timestamp_utc,
        "notes": "auto_materialized_from_refined_subject_component_review_statuses",
        "source_refined_subject_masks_run": refined_subject_run,
        "compat_refined_eye_masks_run": target_run,
    }
    reviewer = _normalize_text(reference.get("reviewer"))
    if reviewer is not None:
        payload["reviewer"] = reviewer
    if source_subject_run is not None:
        payload["source_subject_mask_run"] = source_subject_run
    if source_eye_masks_run is not None:
        payload["source_eye_masks_run"] = source_eye_masks_run
    if source_refined_eye_run is not None:
        payload["source_refined_eye_masks_run"] = source_refined_eye_run
    if source_keypoints_run is not None:
        payload["source_keypoints_run"] = source_keypoints_run
        payload["source_keypoint_run"] = source_keypoints_run
    if source_keypoint_group is not None:
        payload["source_keypoint_group"] = source_keypoint_group
    return payload


def materialize_refined_eye_masks_compat(
    source: str | Path | zarr.Group,
    *,
    refined_subject_run: Optional[str] = None,
    refined_eye_run: Optional[str] = None,
    overwrite: bool = True,
    command: Optional[str] = None,
) -> dict[str, object]:
    stage_start = time.perf_counter()
    root = _open_root(source, mode="a")
    refined_parent = root.get("refined_subject_masks_runs")
    resolved_subject_run = (
        str(refined_subject_run)
        if refined_subject_run is not None
        else _resolve_latest_run(refined_parent, "refined_subject_masks_runs")
    )
    if refined_parent is None or resolved_subject_run not in refined_parent:
        raise RuntimeError(f"refined_subject_masks_runs/{resolved_subject_run} not found.")

    refined_group = refined_parent[resolved_subject_run]
    label_map = _label_index_map(refined_group)
    missing = [component_name for component_name in EYE_COMPONENTS if component_name not in label_map]
    if missing:
        return {
            "status": "skipped",
            "reason": "missing_lr_eye_components",
            "refined_subject_run": resolved_subject_run,
            "missing_components": missing,
        }

    eye_masks, edit_applied = _load_eye_masks_and_edit_state(refined_group)
    total_rois = int(eye_masks.shape[0])
    height = int(eye_masks.shape[2])
    width = int(eye_masks.shape[3])

    component_source_groups, coarse_source_run = _resolve_eye_component_source_subject_runs(root, refined_group)
    compat_parent = require_runs_parent(root, "refined_eye_masks_runs")

    carryover_attrs: dict[str, object] = {}
    raw_eye_run, source_refined_eye_run = _resolve_source_eye_runs(
        refined_group,
        component_source_groups,
        carryover_attrs={},
    )
    target_run = _resolve_target_run_name(
        resolved_subject_run,
        refined_group,
        raw_eye_run=raw_eye_run,
        source_refined_eye_run=source_refined_eye_run,
        explicit_target=refined_eye_run,
    )
    if target_run in compat_parent:
        for key in _CARRYOVER_ATTR_KEYS:
            value = compat_parent[target_run].attrs.get(key)
            if value is not None:
                carryover_attrs[key] = value
        raw_eye_run, source_refined_eye_run = _resolve_source_eye_runs(
            refined_group,
            component_source_groups,
            carryover_attrs=carryover_attrs,
        )
    else:
        raw_eye_run, source_refined_eye_run = _resolve_source_eye_runs(
            refined_group,
            component_source_groups,
            carryover_attrs=carryover_attrs,
        )

    if target_run in compat_parent and not overwrite:
        return {
            "status": "exists",
            "refined_subject_run": resolved_subject_run,
            "target_run": target_run,
        }

    source_left = _load_subject_component_masks(
        component_source_groups.get("eye_left"),
        "eye_left",
        total_rois=total_rois,
        height=height,
        width=width,
    )
    source_right = _load_subject_component_masks(
        component_source_groups.get("eye_right"),
        "eye_right",
        total_rois=total_rois,
        height=height,
        width=width,
    )
    source_union = np.logical_or(source_left > 0, source_right > 0).astype(np.uint8)

    keypoints_roi, keypoint_indices = _load_keypoints_roi(root, refined_group, total_rois=total_rois)
    keypoint_run = _normalize_text(resolve_source_keypoints_run(refined_group.attrs))
    keypoint_group = _normalize_text(refined_group.attrs.get("source_keypoint_group"))
    left_reason_labels = _read_component_reason_labels(refined_group, "eye_left", total_rois=total_rois)
    right_reason_labels = _read_component_reason_labels(refined_group, "eye_right", total_rois=total_rois)
    pair_reason_labels = np.asarray(
        [
            _aggregate_pair_reason_label(left_reason_labels[idx], right_reason_labels[idx])
            for idx in range(total_rois)
        ],
        dtype=object,
    )

    ellipse_params = _nan_array((total_rois, 2, 5))
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)
    area_refined = np.zeros((total_rois, 2), dtype=np.float32)
    area_source = np.zeros((total_rois, 2), dtype=np.float32)
    area_union_refined = np.zeros((total_rois,), dtype=np.float32)
    area_union_source = np.zeros((total_rois,), dtype=np.float32)
    area_ratio_left_right = np.full((total_rois,), np.nan, dtype=np.float32)
    area_diff_left_right = np.full((total_rois,), np.nan, dtype=np.float32)
    area_delta_vs_source = np.full((total_rois, 2), np.nan, dtype=np.float32)
    area_ratio_vs_source = np.full((total_rois, 2), np.nan, dtype=np.float32)
    area_union_delta = np.full((total_rois,), np.nan, dtype=np.float32)
    area_union_ratio = np.full((total_rois,), np.nan, dtype=np.float32)
    centroid_error = _nan_array((total_rois, 2))
    symmetry_offsets = _nan_array((total_rois, 2))
    symmetry_sum = np.full((total_rois,), np.nan, dtype=np.float32)
    symmetry_abs_diff = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_refined = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_keypoint = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_delta = np.full((total_rois,), np.nan, dtype=np.float32)
    axis_ratio = _nan_array((total_rois, 2))
    circularity = _nan_array((total_rois, 2))
    area_zscore = _nan_array((total_rois, 2))
    filter_flags = np.zeros((total_rois, 2), dtype=bool)
    connectivity_flags = np.zeros((total_rois,), dtype=np.uint8)
    smoothing_flags = np.zeros((total_rois, 2), dtype=np.uint8)
    pixels_reassigned = np.zeros((total_rois,), dtype=np.int32)
    probabilities_used = np.zeros((total_rois,), dtype=bool)
    probability_mean = _nan_array((total_rois, 2))
    probability_max = _nan_array((total_rois, 2))
    probability_var = _nan_array((total_rois, 2))
    probability_high_fraction = _nan_array((total_rois, 2))

    for roi_idx in range(total_rois):
        row_masks = np.asarray(eye_masks[roi_idx], dtype=np.uint8)
        centroids = _nan_array((2, 2))
        contours: tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        for eye_idx in range(2):
            success, ellipse, centroid, contour, _failure = _measure_mask(row_masks[eye_idx])
            ellipse_params[roi_idx, eye_idx] = ellipse
            ellipse_success[roi_idx, eye_idx] = bool(success)
            centroids[eye_idx] = centroid
            contours = (
                contour if eye_idx == 0 else contours[0],
                contour if eye_idx == 1 else contours[1],
            )

        if np.all(ellipse_success[roi_idx]) and np.all(np.isfinite(centroids)):
            eye_separation[roi_idx] = np.float32(np.linalg.norm(centroids[0] - centroids[1]))

        if keypoints_roi is not None and keypoint_indices is not None:
            eye_left_kp = np.asarray(
                keypoints_roi[roi_idx, int(keypoint_indices["eye_left"]), :2],
                dtype=np.float32,
            )
            eye_right_kp = np.asarray(
                keypoints_roi[roi_idx, int(keypoint_indices["eye_right"]), :2],
                dtype=np.float32,
            )
            keypoints_valid = (
                eye_left_kp.size >= 2
                and eye_right_kp.size >= 2
                and np.all(np.isfinite(eye_left_kp))
                and np.all(np.isfinite(eye_right_kp))
                and not np.allclose(eye_left_kp, eye_right_kp, atol=1e-3)
            )
        else:
            eye_left_kp = np.full((2,), np.nan, dtype=np.float32)
            eye_right_kp = np.full((2,), np.nan, dtype=np.float32)
            keypoints_valid = False

        metrics = _compute_roi_metrics(
            row_masks[0],
            row_masks[1],
            np.asarray(source_left[roi_idx], dtype=np.uint8),
            np.asarray(source_right[roi_idx], dtype=np.uint8),
            np.asarray(source_union[roi_idx], dtype=np.uint8),
            centroids,
            eye_left_kp,
            eye_right_kp,
            bool(keypoints_valid),
            float(eye_separation[roi_idx]),
            np.asarray(ellipse_params[roi_idx], dtype=np.float32),
            contours,
            None,
        )
        area_refined[roi_idx] = np.asarray(metrics["refined_areas"], dtype=np.float32)
        area_source[roi_idx] = np.asarray(metrics["source_areas"], dtype=np.float32)
        area_union_refined[roi_idx] = np.float32(metrics["refined_union_area"])
        area_union_source[roi_idx] = np.float32(metrics["source_union_area"])
        centroid_error[roi_idx] = np.asarray(metrics["centroid_errors"], dtype=np.float32)
        symmetry_offsets[roi_idx] = np.asarray(metrics["symmetry_offsets"], dtype=np.float32)
        separation_keypoint[roi_idx] = np.float32(metrics["keypoint_separation"])
        separation_delta[roi_idx] = np.float32(metrics["separation_delta"])
        axis_ratio[roi_idx] = np.asarray(metrics["axis_ratio"], dtype=np.float32)
        circularity[roi_idx] = np.asarray(metrics["circularity"], dtype=np.float32)

    valid_ratio = area_refined[:, 1] > 0
    area_ratio_left_right[valid_ratio] = area_refined[valid_ratio, 0] / np.maximum(area_refined[valid_ratio, 1], 1e-6)
    area_diff_left_right = (area_refined[:, 0] - area_refined[:, 1]).astype(np.float32, copy=False)
    area_delta_vs_source = (area_refined - area_source).astype(np.float32, copy=False)
    area_ratio_vs_source = (area_refined / np.maximum(area_source, 1e-6)).astype(np.float32, copy=False)
    area_union_delta = (area_union_refined - area_union_source).astype(np.float32, copy=False)
    area_union_ratio = (area_union_refined / np.maximum(area_union_source, 1e-6)).astype(np.float32, copy=False)
    separation_refined[:] = eye_separation
    symmetry_mask = np.all(np.isfinite(symmetry_offsets), axis=1)
    symmetry_sum[symmetry_mask] = symmetry_offsets[symmetry_mask].sum(axis=1)
    symmetry_abs_diff[symmetry_mask] = np.abs(
        np.abs(symmetry_offsets[symmetry_mask, 0]) - np.abs(symmetry_offsets[symmetry_mask, 1])
    )

    if target_run in compat_parent:
        del compat_parent[target_run]
    run_group = compat_parent.create_group(target_run)
    mark_run_started(run_group, run_name=target_run, stage="refined_eye_masks")
    note_pending_latest(compat_parent, target_run)
    chunk_rois = max(1, min(256, total_rois))
    run_group.create_array("masks_roi", data=eye_masks, chunks=(chunk_rois, 2, height, width), overwrite=True)
    run_group.create_array("ellipse_params", data=ellipse_params, chunks=(chunk_rois, 2, 5), overwrite=True)
    run_group.create_array("ellipse_success", data=ellipse_success, chunks=(chunk_rois, 2), overwrite=True)
    run_group.create_array("eye_separation", data=eye_separation, chunks=(chunk_rois,), overwrite=True)
    run_group.create_array("edit_applied", data=edit_applied, chunks=(chunk_rois, 2), overwrite=True)
    if "detection_source" in refined_group:
        run_group.create_array(
            "detection_source",
            data=np.asarray(refined_group["detection_source"][:], dtype=np.int8),
            chunks=(chunk_rois,),
            overwrite=True,
        )
    else:
        run_group.create_array(
            "detection_source",
            data=np.zeros((total_rois,), dtype=np.int8),
            chunks=(chunk_rois,),
            overwrite=True,
        )

    copy_row_lineage_arrays(run_group, refined_group, total_rois=total_rois)

    metrics_group = run_group.require_group("metrics")
    metrics_group.create_array("area_refined", data=area_refined, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("area_source", data=area_source, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("area_zscore", data=area_zscore, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("area_delta_vs_source", data=area_delta_vs_source, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("area_ratio_vs_source", data=area_ratio_vs_source, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("centroid_error", data=centroid_error, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("symmetry_offsets", data=symmetry_offsets, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("axis_ratio", data=axis_ratio, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("circularity", data=circularity, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("probability_mean", data=probability_mean, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("probability_max", data=probability_max, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("probability_var", data=probability_var, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array(
        "probability_high_fraction",
        data=probability_high_fraction,
        chunks=(chunk_rois, 2),
        overwrite=True,
    )
    metrics_group.create_array("area_union_refined", data=area_union_refined, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("area_union_source", data=area_union_source, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array(
        "area_ratio_left_right",
        data=area_ratio_left_right,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    metrics_group.create_array(
        "area_diff_left_right",
        data=area_diff_left_right,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    metrics_group.create_array("area_union_delta", data=area_union_delta, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("area_union_ratio", data=area_union_ratio, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("symmetry_sum", data=symmetry_sum, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("symmetry_abs_diff", data=symmetry_abs_diff, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("separation_refined", data=separation_refined, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("separation_keypoint", data=separation_keypoint, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("separation_delta", data=separation_delta, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("connectivity_flags", data=connectivity_flags, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("smoothing_flags", data=smoothing_flags, chunks=(chunk_rois, 2), overwrite=True)
    metrics_group.create_array("pixels_reassigned", data=pixels_reassigned, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("probabilities_used", data=probabilities_used, chunks=(chunk_rois,), overwrite=True)
    metrics_group.create_array("filter_flags", data=filter_flags, chunks=(chunk_rois, 2), overwrite=True)
    write_reason_columns(
        metrics_group,
        pair_reason_labels,
        chunk_size=chunk_rois,
        include_reason_text=True,
        overwrite=True,
    )

    _write_contours_from_masks(run_group, total_rois=total_rois, chunk_rois=chunk_rois)

    source_eye_group = None
    if raw_eye_run is not None:
        eye_parent = root.get("eye_masks_runs")
        if eye_parent is not None and raw_eye_run in eye_parent:
            source_eye_group = eye_parent[raw_eye_run]
    source_eye_method = (
        _normalize_text(source_eye_group.attrs.get("method"))
        if source_eye_group is not None
        else _normalize_text(carryover_attrs.get("source_eye_masks_method"))
    )
    source_eye_labels = (
        list(source_eye_group.attrs.get("eye_labels"))
        if source_eye_group is not None and isinstance(source_eye_group.attrs.get("eye_labels"), (list, tuple))
        else (
            list(carryover_attrs.get("source_eye_labels"))
            if isinstance(carryover_attrs.get("source_eye_labels"), (list, tuple))
            else ["eye_left", "eye_right"]
        )
    )
    success_min_eye_area_px = None
    for key in ("success_min_eye_area_px", "min_eye_area_success_px"):
        value = carryover_attrs.get(key)
        if value is not None:
            success_min_eye_area_px = float(value)
            break

    metrics_summary, reason_tag_counts = _build_metrics_summary(
        area_refined=area_refined,
        area_union_refined=area_union_refined,
        centroid_error=centroid_error,
        symmetry_sum=symmetry_sum,
        separation_delta=separation_delta,
        ellipse_success=ellipse_success,
        pair_reason_labels=pair_reason_labels,
        success_min_eye_area_px=success_min_eye_area_px,
    )

    created_at_utc = _utc_now()
    review_payload = _aggregate_eye_review_status(
        resolved_subject_run,
        refined_group,
        target_run=target_run,
        source_subject_run=coarse_source_run,
        source_eye_masks_run=raw_eye_run,
        source_refined_eye_run=source_refined_eye_run,
        source_keypoints_run=keypoint_run,
        source_keypoint_group=keypoint_group,
    )

    summary_statistics = _empty_summary_statistics(carryover_attrs)
    run_group.attrs.update(carryover_attrs)
    run_group.attrs.update(
        {
            "method": MATERIALIZE_REFINED_EYE_MASKS_COMPAT_METHOD,
            "compatibility_role": "derived_from_refined_subject_masks",
            "source_refined_subject_masks_run": resolved_subject_run,
            "source_subject_mask_run": coarse_source_run,
            "source_subject_mask_method": _normalize_text(refined_group.attrs.get("source_subject_mask_method")),
            "source_eye_masks_run": raw_eye_run,
            "source_refined_eye_masks_run": source_refined_eye_run,
            "source_eye_masks_method": source_eye_method,
            "source_crop_run": _normalize_text(refined_group.attrs.get("source_crop_run")),
            "total_rois": total_rois,
            "eye_labels": list(EYE_COMPONENTS),
            "source_eye_labels": source_eye_labels,
            "created_at_utc": created_at_utc,
            "created_utc": created_at_utc,
            "updated_at_utc": created_at_utc,
            "duration_seconds": float(time.perf_counter() - stage_start),
            "metrics_summary": metrics_summary,
            "summary_statistics": summary_statistics,
            "reason_tag_counts": reason_tag_counts,
            "eye_mask_review_status": review_payload,
        }
    )
    if keypoint_run is not None:
        run_group.attrs.update(build_source_keypoints_attrs(keypoint_run, include_legacy_alias=True))
    if keypoint_group is not None:
        run_group.attrs["source_keypoint_group"] = keypoint_group

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage=MATERIALIZE_REFINED_EYE_MASKS_STAGE,
        command=command or (" ".join(sys.argv) if sys.argv else "unknown"),
        created_at_utc=created_at_utc,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters={
            "method": MATERIALIZE_REFINED_EYE_MASKS_COMPAT_METHOD,
            "components": list(EYE_COMPONENTS),
            "overwrite": bool(overwrite),
            "compatibility_role": "derived_from_refined_subject_masks",
        },
        inputs={
            "refined_subject_masks_run": resolved_subject_run,
            "subject_mask_run": coarse_source_run,
            "source_eye_masks_run": raw_eye_run,
            "source_refined_eye_masks_run": source_refined_eye_run,
            "source_keypoints_run": keypoint_run,
            "source_keypoint_group": keypoint_group,
        },
    )
    write_stage_provenance(run_group, provenance)
    refined_group.attrs["compat_refined_eye_masks_run"] = target_run
    mark_run_complete(run_group, parent_group=compat_parent, run_name=target_run)
    compat_parent.attrs["eye_mask_review_status_latest"] = target_run

    from ..tune.eye_mask_review import _update_postprocess_summary

    _update_postprocess_summary(root, run_group, print_summary=False)

    return {
        "status": "updated",
        "refined_subject_run": resolved_subject_run,
        "target_run": target_run,
        "source_subject_mask_run": coarse_source_run,
        "source_eye_masks_run": raw_eye_run,
        "source_refined_eye_masks_run": source_refined_eye_run,
        "review_state": str(review_payload.get("state") or ""),
        "total_rois": total_rois,
        "duration_seconds": float(time.perf_counter() - stage_start),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--refined-subject-run",
        help="Canonical refined_subject_masks_runs/<run> to materialize from (default: latest).",
    )
    parser.add_argument(
        "--refined-eye-run",
        help="Target refined_eye_masks_runs/<run> override.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite an existing target refined_eye_masks_runs/<run>.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the materialization summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = materialize_refined_eye_masks_compat(
        args.zarr_path,
        refined_subject_run=args.refined_subject_run,
        refined_eye_run=args.refined_eye_run,
        overwrite=not bool(args.no_overwrite),
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
