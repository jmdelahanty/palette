#!/usr/bin/env python3
"""Inspect/apply utility for SAM-based subject-mask generation from Palette zarrs."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.pose.schema import schema_from_metadata
from fisheye.shared.crop_image_source import resolve_materialized_crop_run
from fisheye.shared.provenance_attrs import build_source_keypoints_attrs
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root

KEYPOINT_GROUP_CHOICES = ("auto", "refined_keypoints_runs", "keypoints_runs")
SAM3_ENV_VARS = ("PALETTE_SAM3_ROOT", "SAM3_ROOT")
DEFAULT_PREPARE_COUNT = 8
SUBJECT_MASK_LABEL_SCHEMA = "subject_v1_union"
SUBJECT_MASK_LABELS: tuple[str, ...] = ("subject_body", "eyes_union", "swim_bladder")
SUBJECT_MASK_AVAILABLE_CHANNELS = (True, False, False)
DEFAULT_INFER_BATCH_SIZE = 8
BOX_PROMPT_SOURCE_CHOICES = ("detect", "roi_inset", "pose_roi")
DEFAULT_BOX_PROMPT_SOURCE = "detect"
DEFAULT_ROI_INSET_FRACTION = 0.05
DEFAULT_POSE_BOX_EXPAND_FRACTION = 0.10
NEGATIVE_POINT_POLICY_CHOICES = ("none", "corners", "border8")
DEFAULT_NEGATIVE_POINT_POLICY = "none"
DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION = 0.05
DEFAULT_KEYPOINT_LABELS: tuple[str, ...] = ("swim_bladder", "eye_left", "eye_right")
KEYPOINT_LABEL_ALIASES: dict[str, str] = {
    "bladder": "swim_bladder",
    "swimbladder": "swim_bladder",
    "swim_bladder": "swim_bladder",
    "left": "eye_left",
    "left_eye": "eye_left",
    "eye_left": "eye_left",
    "right": "eye_right",
    "right_eye": "eye_right",
    "eye_right": "eye_right",
}


@dataclass(frozen=True)
class ResolvedKeypointRun:
    group_name: str
    run_name: str
    group: zarr.Group


@dataclass(frozen=True)
class ResolvedSamInputs:
    crop_run: str
    keypoint_group: str
    keypoint_run: str
    roi_images: zarr.Array
    roi_coordinates_full: np.ndarray
    bbox_norm_coords: np.ndarray
    frame_indices: np.ndarray
    detection_indices: np.ndarray
    detection_source: np.ndarray
    keypoints_roi: np.ndarray
    keypoint_labels: tuple[str, ...]
    success_flags: np.ndarray | None
    geometry_valid: np.ndarray | None
    usable_keypoints: np.ndarray | None
    frame_height: int
    frame_width: int
    warnings: tuple[str, ...]
    pose_bbox_xyxy_roi: np.ndarray | None = None

    @property
    def row_count(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def roi_height(self) -> int:
        return int(self.roi_images.shape[1])

    @property
    def roi_width(self) -> int:
        return int(self.roi_images.shape[2])

    @property
    def roi_channels(self) -> int:
        if int(self.roi_images.ndim) == 4:
            return int(self.roi_images.shape[3])
        return 1


@dataclass(frozen=True)
class RowEligibility:
    eligible: np.ndarray
    prompt_point_finite: np.ndarray
    prompt_point_in_bounds: np.ndarray
    prompt_point_count: np.ndarray
    skipped_interpolated: np.ndarray
    success_ok: np.ndarray
    geometry_ok: np.ndarray
    usable_ok: np.ndarray


@dataclass(frozen=True)
class PreparedSamBatch:
    row_indices: np.ndarray
    images_rgb: list[np.ndarray]
    point_coords_batch: list[np.ndarray]
    point_labels_batch: list[np.ndarray]
    box_batch: list[np.ndarray] | None


@dataclass(frozen=True)
class PromptKeypointSelection:
    labels: tuple[str, ...]
    indices: np.ndarray


@dataclass(frozen=True)
class SelectedMaskBatch:
    row_indices: np.ndarray
    binary: np.ndarray
    probs: np.ndarray
    scores: np.ndarray


def _require_array(group: zarr.Group, name: str, *, context: str) -> zarr.Array:
    arr = group.get(name)
    if arr is None or not hasattr(arr, "shape"):
        raise ValueError(f"{context} missing required array '{name}'.")
    return arr


def _load_1d_int(group: zarr.Group, name: str, *, context: str) -> np.ndarray:
    arr = np.asarray(_require_array(group, name, context=context)[:], dtype=np.int64).reshape(-1)
    return arr


def _load_2d(group: zarr.Group, name: str, *, context: str, width: int, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.asarray(_require_array(group, name, context=context)[:], dtype=dtype)
    if arr.ndim != 2 or int(arr.shape[1]) != int(width):
        raise ValueError(f"{context}/{name} expected shape (N,{width}), got {tuple(int(v) for v in arr.shape)}.")
    return arr


def _load_optional_bool(
    group: zarr.Group,
    name: str,
    *,
    expected_len: int,
) -> np.ndarray | None:
    arr = group.get(name)
    if arr is None:
        return None
    values = np.asarray(arr[:], dtype=bool).reshape(-1)
    if int(values.shape[0]) != int(expected_len):
        raise ValueError(
            f"{name} length mismatch ({values.shape[0]} != expected {expected_len})."
        )
    return values


def _normalize_sam3_root(explicit: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())

    for env_name in SAM3_ENV_VARS:
        env_value = os.environ.get(env_name)  # type: ignore[name-defined]
        if env_value:
            candidates.append(Path(env_value).expanduser())

    palette_repo = Path(__file__).resolve().parents[3]
    candidates.append(palette_repo.parent / "sam3")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if (resolved / "sam3").is_dir():
            return resolved
        if resolved.name == "sam3" and (resolved / "model").is_dir():
            return resolved.parent
    return None


def _resolve_frame_shape(root: zarr.Group, crop_group: zarr.Group) -> tuple[int, int]:
    width = (
        root.attrs.get("video_width")
        or root.attrs.get("width")
        or root.attrs.get("source_video_width")
        or crop_group.attrs.get("width")
    )
    height = (
        root.attrs.get("video_height")
        or root.attrs.get("height")
        or root.attrs.get("source_video_height")
        or crop_group.attrs.get("height")
    )
    if width is None or height is None:
        raise ValueError(
            "Could not resolve full-frame width/height from root or crop attrs; "
            "required to project detection boxes into ROI-local SAM prompts."
        )
    return int(height), int(width)


def _inspect_sam3_runtime(explicit_root: str | Path | None) -> dict[str, Any]:
    root = _normalize_sam3_root(explicit_root)
    if root is None:
        return {
            "root": None,
            "available": False,
            "predictor_available": False,
            "builder_available": False,
            "error": "Could not resolve a SAM3 checkout. Set --sam3-root or PALETTE_SAM3_ROOT.",
        }

    added_path = False
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        added_path = True

    try:
        predictor_mod = importlib.import_module("sam3.model.sam1_task_predictor")
        builder_mod = importlib.import_module("sam3.model_builder")
        return {
            "root": root_text,
            "available": True,
            "predictor_available": hasattr(predictor_mod, "SAM3InteractiveImagePredictor"),
            "builder_available": hasattr(builder_mod, "build_sam3_image_model"),
            "error": None,
        }
    except Exception as exc:
        return {
            "root": root_text,
            "available": False,
            "predictor_available": False,
            "builder_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if added_path and root_text in sys.path:
            sys.path.remove(root_text)


def _normalize_keypoint_label_name(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    return KEYPOINT_LABEL_ALIASES.get(lowered, lowered)


def _normalize_keypoint_labels_from_attrs(
    labels_raw: object,
    *,
    expected_count: int,
) -> tuple[str, ...] | None:
    if not isinstance(labels_raw, (list, tuple)):
        return None
    labels = [str(item).strip() for item in labels_raw if str(item).strip()]
    if len(labels) != int(expected_count):
        return None
    return tuple(labels)


def _resolve_keypoint_labels(group: zarr.Group, *, n_keypoints: int) -> tuple[str, ...]:
    labels = _normalize_keypoint_labels_from_attrs(
        group.attrs.get("keypoint_labels"),
        expected_count=n_keypoints,
    )
    if labels is not None:
        return labels

    pose_schema_meta = group.attrs.get("pose_schema")
    if isinstance(pose_schema_meta, dict):
        try:
            pose_schema = schema_from_metadata(pose_schema_meta)
        except Exception:
            pose_schema = None
        if pose_schema is not None and int(pose_schema.num_keypoints) == int(n_keypoints):
            return tuple(str(name) for name in pose_schema.node_names)

    if n_keypoints == len(DEFAULT_KEYPOINT_LABELS):
        return DEFAULT_KEYPOINT_LABELS
    return tuple(f"kp_{idx}" for idx in range(int(n_keypoints)))


def _normalize_requested_positive_labels(
    labels: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if labels is None:
        return None
    normalized: list[str] = []
    for item in labels:
        for part in str(item).split(","):
            text = part.strip()
            if not text:
                continue
            normalized.append(text)
    return tuple(normalized) if normalized else None


def resolve_prompt_keypoint_selection(
    inputs: ResolvedSamInputs,
    *,
    positive_keypoint_labels: Sequence[str] | None = None,
) -> PromptKeypointSelection:
    labels = list(inputs.keypoint_labels)
    if not labels:
        raise ValueError("Resolved SAM inputs do not contain any keypoint labels.")

    normalized_requested = _normalize_requested_positive_labels(positive_keypoint_labels)
    if normalized_requested is None:
        return PromptKeypointSelection(
            labels=tuple(labels),
            indices=np.arange(len(labels), dtype=np.int32),
        )

    canonical_to_index: dict[str, int] = {}
    for idx, label in enumerate(labels):
        canonical_to_index.setdefault(_normalize_keypoint_label_name(label), idx)

    resolved_labels: list[str] = []
    resolved_indices: list[int] = []
    missing: list[str] = []
    seen_indices: set[int] = set()
    for requested in normalized_requested:
        canonical = _normalize_keypoint_label_name(requested)
        idx = canonical_to_index.get(canonical)
        if idx is None:
            missing.append(requested)
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        resolved_indices.append(int(idx))
        resolved_labels.append(str(labels[idx]))
    if missing:
        raise ValueError(
            f"Requested positive keypoint labels not present on {inputs.keypoint_group}/{inputs.keypoint_run}: "
            f"{missing!r}. Available labels: {labels!r}."
        )
    if not resolved_indices:
        raise ValueError("Resolved positive_keypoint_labels is empty after normalization.")
    return PromptKeypointSelection(
        labels=tuple(resolved_labels),
        indices=np.asarray(resolved_indices, dtype=np.int32),
    )


def _sam_prompt_policy(
    use_box_prompt: bool,
    box_prompt_source: str,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
) -> str:
    parts = ["keypoint_points"]
    if use_box_prompt:
        if box_prompt_source == "roi_inset":
            parts.append("plus_roi_inset_box")
        elif box_prompt_source == "pose_roi":
            parts.append("plus_pose_roi_box")
        else:
            parts.append("plus_detect_box")
    if negative_point_policy == "corners":
        parts.append("plus_corner_negatives")
    elif negative_point_policy == "border8":
        parts.append("plus_border8_negatives")
    elif negative_point_policy != "none":
        raise ValueError(f"Unknown negative_point_policy {negative_point_policy!r}.")
    return "_".join(parts)


def _sam_method_name(
    use_box_prompt: bool,
    box_prompt_source: str,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
) -> str:
    if negative_point_policy == "none":
        if not use_box_prompt:
            return "sam3_interactive_keypoints_body_v1"
        if box_prompt_source == "roi_inset":
            return "sam3_interactive_keypoints_roi_inset_box_body_v1"
        if box_prompt_source == "pose_roi":
            return "sam3_interactive_keypoints_pose_roi_box_body_v1"
        return "sam3_interactive_keypoints_detect_box_body_v1"

    if negative_point_policy == "corners":
        suffix = "corner_negatives"
    elif negative_point_policy == "border8":
        suffix = "border8_negatives"
    else:
        raise ValueError(f"Unknown negative_point_policy {negative_point_policy!r}.")

    if not use_box_prompt:
        return f"sam3_interactive_keypoints_{suffix}_body_v1"
    if box_prompt_source == "roi_inset":
        return f"sam3_interactive_keypoints_roi_inset_box_{suffix}_body_v1"
    if box_prompt_source == "pose_roi":
        return f"sam3_interactive_keypoints_pose_roi_box_{suffix}_body_v1"
    return f"sam3_interactive_keypoints_detect_box_{suffix}_body_v1"


def _resolve_keypoint_run(
    root: zarr.Group,
    *,
    keypoint_run: str | None,
    keypoint_group: str,
    zarr_path: str | Path | None = None,
) -> ResolvedKeypointRun:
    archive_path = Path(zarr_path).expanduser() if zarr_path is not None else None

    def _direct_run_names(group_name: str) -> list[str]:
        if archive_path is None:
            return []
        parent_path = archive_path / group_name
        if not parent_path.is_dir():
            return []
        names: list[str] = []
        for candidate in sorted(parent_path.iterdir()):
            if not candidate.is_dir():
                continue
            if (candidate / "zarr.json").is_file() or (candidate / ".zgroup").is_file():
                names.append(candidate.name)
        return names

    def _direct_open(group_name: str, run_name: str) -> zarr.Group:
        if archive_path is None:
            raise ValueError("archive_path is required for direct keypoint run fallback")
        run_path = archive_path / group_name / run_name
        try:
            return zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        except TypeError:
            return zarr.open_group(str(run_path), mode="r", consolidated=False)

    group_candidates: Sequence[str]
    if keypoint_group == "auto":
        group_candidates = ("refined_keypoints_runs", "keypoints_runs")
    else:
        group_candidates = (keypoint_group,)

    errors: list[str] = []
    for group_name in group_candidates:
        try:
            run_group, run_name = resolve_zarr_run(
                root,
                group_name,
                keypoint_run,
                fallback_to_latest=True,
                run_label="Keypoint run",
            )
            return ResolvedKeypointRun(group_name=group_name, run_name=run_name, group=run_group)
        except Exception as exc:
            direct_names = _direct_run_names(group_name)
            requested = str(keypoint_run).strip() if keypoint_run is not None and str(keypoint_run).strip() else None
            latest_name = None
            try:
                parent_group = root.get(group_name)
            except Exception:
                parent_group = None
            if parent_group is not None:
                latest_name = str(parent_group.attrs.get("latest")).strip() if parent_group.attrs.get("latest") else None
            fallback_name = requested or latest_name
            if fallback_name and fallback_name in direct_names:
                return ResolvedKeypointRun(
                    group_name=group_name,
                    run_name=fallback_name,
                    group=_direct_open(group_name, fallback_name),
                )
            errors.append(f"{group_name}: {exc}")
    raise ValueError("; ".join(errors) if errors else "Could not resolve keypoint run.")


def _raise_alignment_error(
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_path: str,
    right_path: str,
) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{label} length mismatch between {left_path} ({left.shape[0]}) and "
            f"{right_path} ({right.shape[0]})."
        )
    mismatch_idx = np.flatnonzero(left != right)
    if mismatch_idx.size:
        idx = int(mismatch_idx[0])
        raise ValueError(
            f"{label} mismatch at row {idx}: {left_path}={left[idx]!r}, {right_path}={right[idx]!r}."
        )


def resolve_sam_subject_inputs(
    root: zarr.Group,
    *,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "auto",
    zarr_path: str | Path | None = None,
) -> ResolvedSamInputs:
    _, crop_group, resolved_crop_run = resolve_materialized_crop_run(root, crop_run=crop_run)
    keypoints = _resolve_keypoint_run(
        root,
        keypoint_run=keypoint_run,
        keypoint_group=keypoint_group,
        zarr_path=zarr_path,
    )

    roi_images = _require_array(crop_group, "roi_images", context=f"crop_runs/{resolved_crop_run}")
    roi_coordinates_full = _load_2d(
        crop_group,
        "roi_coordinates_full",
        context=f"crop_runs/{resolved_crop_run}",
        width=2,
        dtype=np.int32,
    )
    bbox_norm_coords = _load_2d(
        crop_group,
        "bbox_norm_coords",
        context=f"crop_runs/{resolved_crop_run}",
        width=4,
        dtype=np.float32,
    )
    frame_indices = _load_1d_int(crop_group, "frame_indices", context=f"crop_runs/{resolved_crop_run}")
    detection_indices = _load_1d_int(
        crop_group, "detection_indices", context=f"crop_runs/{resolved_crop_run}"
    )
    detection_source = np.asarray(
        _require_array(crop_group, "detection_source", context=f"crop_runs/{resolved_crop_run}")[:],
        dtype=np.int8,
    ).reshape(-1)

    row_count = int(roi_images.shape[0])
    for name, arr in (
        ("roi_coordinates_full", roi_coordinates_full),
        ("bbox_norm_coords", bbox_norm_coords),
        ("frame_indices", frame_indices),
        ("detection_indices", detection_indices),
        ("detection_source", detection_source),
    ):
        if int(arr.shape[0]) != row_count:
            raise ValueError(
                f"crop_runs/{resolved_crop_run}/{name} length mismatch ({arr.shape[0]} != roi_images rows {row_count})."
            )

    keypoints_roi = np.asarray(
        _require_array(
            keypoints.group,
            "keypoints_roi",
            context=f"{keypoints.group_name}/{keypoints.run_name}",
        )[:],
        dtype=np.float32,
    )
    if keypoints_roi.ndim != 3 or int(keypoints_roi.shape[2]) != 2:
        raise ValueError(
            f"{keypoints.group_name}/{keypoints.run_name}/keypoints_roi expected shape (n_rois, K, 2), "
            f"got {tuple(int(v) for v in keypoints_roi.shape)}."
        )

    kp_frame_indices = _load_1d_int(
        keypoints.group, "frame_indices", context=f"{keypoints.group_name}/{keypoints.run_name}"
    )
    kp_detection_indices = _load_1d_int(
        keypoints.group, "detection_indices", context=f"{keypoints.group_name}/{keypoints.run_name}"
    )
    if int(keypoints_roi.shape[0]) != row_count:
        raise ValueError(
            f"Row mismatch between crop_runs/{resolved_crop_run}/roi_images ({row_count}) and "
            f"{keypoints.group_name}/{keypoints.run_name}/keypoints_roi ({keypoints_roi.shape[0]})."
        )
    pose_bbox_xyxy_roi = None
    pose_bbox_arr = keypoints.group.get("pose_bbox_xyxy_roi")
    if pose_bbox_arr is not None:
        pose_bbox_xyxy_roi = np.asarray(pose_bbox_arr[:], dtype=np.float32)
        if pose_bbox_xyxy_roi.ndim != 2 or int(pose_bbox_xyxy_roi.shape[1]) != 4:
            raise ValueError(
                f"{keypoints.group_name}/{keypoints.run_name}/pose_bbox_xyxy_roi expected shape (n_rois, 4), "
                f"got {tuple(int(v) for v in pose_bbox_xyxy_roi.shape)}."
            )
        if int(pose_bbox_xyxy_roi.shape[0]) != row_count:
            raise ValueError(
                f"Row mismatch between crop_runs/{resolved_crop_run}/roi_images ({row_count}) and "
                f"{keypoints.group_name}/{keypoints.run_name}/pose_bbox_xyxy_roi ({pose_bbox_xyxy_roi.shape[0]})."
            )

    _raise_alignment_error(
        "frame_indices",
        frame_indices,
        kp_frame_indices,
        left_path=f"crop_runs/{resolved_crop_run}/frame_indices",
        right_path=f"{keypoints.group_name}/{keypoints.run_name}/frame_indices",
    )
    _raise_alignment_error(
        "detection_indices",
        detection_indices,
        kp_detection_indices,
        left_path=f"crop_runs/{resolved_crop_run}/detection_indices",
        right_path=f"{keypoints.group_name}/{keypoints.run_name}/detection_indices",
    )

    warnings: list[str] = []
    kp_detection_source_arr = keypoints.group.get("detection_source")
    if kp_detection_source_arr is not None:
        kp_detection_source = np.asarray(kp_detection_source_arr[:], dtype=np.int8).reshape(-1)
        if int(kp_detection_source.shape[0]) != row_count:
            raise ValueError(
                f"{keypoints.group_name}/{keypoints.run_name}/detection_source length mismatch "
                f"({kp_detection_source.shape[0]} != expected {row_count})."
            )
        mismatch = np.flatnonzero(kp_detection_source != detection_source)
        if mismatch.size:
            warnings.append(
                f"detection_source differs between crop and keypoint runs in {int(mismatch.size)} rows; "
                "using crop detection_source as the SAM eligibility source."
            )

    success_flags = None
    for success_name in ("refined_success", "detection_success", "source_success"):
        success_flags = _load_optional_bool(keypoints.group, success_name, expected_len=row_count)
        if success_flags is not None:
            break

    geometry_valid = _load_optional_bool(keypoints.group, "geometry_valid", expected_len=row_count)
    usable_keypoints = _load_optional_bool(keypoints.group, "usable_keypoints", expected_len=row_count)
    frame_height, frame_width = _resolve_frame_shape(root, crop_group)
    keypoint_labels = _resolve_keypoint_labels(keypoints.group, n_keypoints=int(keypoints_roi.shape[1]))

    return ResolvedSamInputs(
        crop_run=resolved_crop_run,
        keypoint_group=keypoints.group_name,
        keypoint_run=keypoints.run_name,
        roi_images=roi_images,
        roi_coordinates_full=roi_coordinates_full.astype(np.int32, copy=False),
        bbox_norm_coords=bbox_norm_coords.astype(np.float32, copy=False),
        frame_indices=frame_indices.astype(np.int32, copy=False),
        detection_indices=detection_indices.astype(np.int32, copy=False),
        detection_source=detection_source.astype(np.int8, copy=False),
        keypoints_roi=keypoints_roi.astype(np.float32, copy=False),
        keypoint_labels=keypoint_labels,
        success_flags=success_flags,
        geometry_valid=geometry_valid,
        usable_keypoints=usable_keypoints,
        frame_height=int(frame_height),
        frame_width=int(frame_width),
        warnings=tuple(warnings),
        pose_bbox_xyxy_roi=None if pose_bbox_xyxy_roi is None else pose_bbox_xyxy_roi.astype(np.float32, copy=False),
    )


def compute_row_eligibility(
    inputs: ResolvedSamInputs,
    *,
    prompt_selection: PromptKeypointSelection,
    skip_interpolated: bool = True,
) -> RowEligibility:
    selected_points = np.asarray(
        inputs.keypoints_roi[:, prompt_selection.indices, :],
        dtype=np.float32,
    )
    finite_per_point = np.isfinite(selected_points).all(axis=2)
    in_bounds_per_point = finite_per_point.copy()
    in_bounds_per_point &= selected_points[:, :, 0] >= 0.0
    in_bounds_per_point &= selected_points[:, :, 1] >= 0.0
    in_bounds_per_point &= selected_points[:, :, 0] < float(inputs.roi_width)
    in_bounds_per_point &= selected_points[:, :, 1] < float(inputs.roi_height)
    finite = np.any(finite_per_point, axis=1)
    in_bounds = np.any(in_bounds_per_point, axis=1)
    prompt_point_count = np.sum(in_bounds_per_point, axis=1, dtype=np.int32)

    success_ok = (
        inputs.success_flags.astype(bool, copy=False)
        if inputs.success_flags is not None
        else np.ones((inputs.row_count,), dtype=bool)
    )
    geometry_ok = (
        inputs.geometry_valid.astype(bool, copy=False)
        if inputs.geometry_valid is not None
        else np.ones((inputs.row_count,), dtype=bool)
    )
    usable_ok = (
        inputs.usable_keypoints.astype(bool, copy=False)
        if inputs.usable_keypoints is not None
        else np.ones((inputs.row_count,), dtype=bool)
    )
    skipped_interpolated = (
        inputs.detection_source != 0 if skip_interpolated else np.zeros((inputs.row_count,), dtype=bool)
    )

    eligible = in_bounds & success_ok & geometry_ok & usable_ok & ~skipped_interpolated
    return RowEligibility(
        eligible=eligible,
        prompt_point_finite=finite,
        prompt_point_in_bounds=in_bounds,
        prompt_point_count=prompt_point_count.astype(np.int32, copy=False),
        skipped_interpolated=skipped_interpolated,
        success_ok=success_ok,
        geometry_ok=geometry_ok,
        usable_ok=usable_ok,
    )


def convert_roi_batch_to_rgb(roi_batch: np.ndarray) -> np.ndarray:
    arr = np.asarray(roi_batch)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Expected ROI batch with shape (N,H,W) or (N,H,W,C), got {arr.shape}.")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {arr.shape[-1]}.")

    if arr.dtype == np.bool_:
        return (arr.astype(np.uint8, copy=False) * np.uint8(255)).astype(np.uint8, copy=False)

    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, 0.0, 255.0)
        max_value = float(np.max(clipped)) if clipped.size else 0.0
        if max_value <= 1.0:
            clipped = clipped * 255.0
        return clipped.astype(np.uint8)

    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def project_bbox_norm_to_roi_xyxy(
    bbox_norm_xywh: np.ndarray,
    roi_xy_full: np.ndarray,
    *,
    frame_height: int,
    frame_width: int,
    roi_height: int,
    roi_width: int,
) -> np.ndarray:
    bbox = np.asarray(bbox_norm_xywh, dtype=np.float32).reshape(4)
    roi_xy = np.asarray(roi_xy_full, dtype=np.float32).reshape(2)
    if not np.isfinite(bbox).all():
        raise ValueError(f"bbox_norm_xywh must be finite, got {bbox.tolist()}.")
    if not np.isfinite(roi_xy).all():
        raise ValueError(f"roi_xy_full must be finite, got {roi_xy.tolist()}.")
    if frame_height <= 0 or frame_width <= 0 or roi_height <= 0 or roi_width <= 0:
        raise ValueError("frame and ROI dimensions must be positive.")

    cx = float(bbox[0]) * float(frame_width)
    cy = float(bbox[1]) * float(frame_height)
    bw = float(bbox[2]) * float(frame_width)
    bh = float(bbox[3]) * float(frame_height)
    x0 = cx - (bw / 2.0) - float(roi_xy[0])
    y0 = cy - (bh / 2.0) - float(roi_xy[1])
    x1 = cx + (bw / 2.0) - float(roi_xy[0])
    y1 = cy + (bh / 2.0) - float(roi_xy[1])

    max_x = float(roi_width - 1)
    max_y = float(roi_height - 1)
    x0 = float(np.clip(x0, 0.0, max_x))
    y0 = float(np.clip(y0, 0.0, max_y))
    x1 = float(np.clip(x1, 0.0, max_x))
    y1 = float(np.clip(y1, 0.0, max_y))

    if roi_width > 1 and x1 <= x0:
        x1 = min(max_x, x0 + 1.0)
    if roi_height > 1 and y1 <= y0:
        y1 = min(max_y, y0 + 1.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Projected ROI-local detection box collapsed after clipping: "
            f"bbox_norm={bbox.tolist()} roi_xy={roi_xy.tolist()} roi_shape=({roi_height},{roi_width})."
        )

    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def build_roi_inset_box_xyxy(
    *,
    roi_height: int,
    roi_width: int,
    inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
) -> np.ndarray:
    if roi_height <= 0 or roi_width <= 0:
        raise ValueError("roi_height and roi_width must be positive.")
    if not np.isfinite(inset_fraction):
        raise ValueError("inset_fraction must be finite.")
    if inset_fraction < 0.0 or inset_fraction >= 0.5:
        raise ValueError("inset_fraction must be in [0.0, 0.5).")

    max_x = float(roi_width - 1)
    max_y = float(roi_height - 1)
    inset_x = float(np.floor(float(roi_width) * float(inset_fraction)))
    inset_y = float(np.floor(float(roi_height) * float(inset_fraction)))
    x0 = float(np.clip(inset_x, 0.0, max_x))
    y0 = float(np.clip(inset_y, 0.0, max_y))
    x1 = float(np.clip(max_x - inset_x, 0.0, max_x))
    y1 = float(np.clip(max_y - inset_y, 0.0, max_y))

    if roi_width > 1 and x1 <= x0:
        x0 = 0.0
        x1 = max_x
    if roi_height > 1 and y1 <= y0:
        y0 = 0.0
        y1 = max_y

    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def expand_roi_box_xyxy(
    box_xyxy: np.ndarray,
    *,
    roi_height: int,
    roi_width: int,
    expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
) -> np.ndarray:
    if roi_height <= 0 or roi_width <= 0:
        raise ValueError("roi_height and roi_width must be positive.")
    if not np.isfinite(expand_fraction):
        raise ValueError("expand_fraction must be finite.")
    if expand_fraction < 0.0:
        raise ValueError("expand_fraction must be non-negative.")

    box = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
    if not np.isfinite(box).all():
        raise ValueError(f"box_xyxy must be finite, got {box.tolist()}.")

    max_x = float(roi_width - 1)
    max_y = float(roi_height - 1)
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    width = max(float(box[2]) - float(box[0]), 1.0)
    height = max(float(box[3]) - float(box[1]), 1.0)
    width *= 1.0 + float(expand_fraction)
    height *= 1.0 + float(expand_fraction)
    x0 = float(np.clip(cx - (width / 2.0), 0.0, max_x))
    y0 = float(np.clip(cy - (height / 2.0), 0.0, max_y))
    x1 = float(np.clip(cx + (width / 2.0), 0.0, max_x))
    y1 = float(np.clip(cy + (height / 2.0), 0.0, max_y))

    if roi_width > 1 and x1 <= x0:
        x1 = min(max_x, x0 + 1.0)
    if roi_height > 1 and y1 <= y0:
        y1 = min(max_y, y0 + 1.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Expanded ROI-local box collapsed after clipping: "
            f"box_xyxy={box.tolist()} roi_shape=({roi_height},{roi_width}) expand_fraction={expand_fraction}."
        )
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def build_negative_prompt_points(
    *,
    roi_height: int,
    roi_width: int,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
) -> np.ndarray:
    if negative_point_policy == "none":
        return np.zeros((0, 2), dtype=np.float32)
    if negative_point_policy not in NEGATIVE_POINT_POLICY_CHOICES:
        raise ValueError(
            f"negative_point_policy must be one of {NEGATIVE_POINT_POLICY_CHOICES}, got {negative_point_policy!r}."
        )
    if roi_height <= 0 or roi_width <= 0:
        raise ValueError("roi_height and roi_width must be positive.")
    if not np.isfinite(margin_fraction):
        raise ValueError("margin_fraction must be finite.")
    if margin_fraction < 0.0 or margin_fraction >= 0.5:
        raise ValueError("margin_fraction must be in [0.0, 0.5).")

    max_x = float(roi_width - 1)
    max_y = float(roi_height - 1)
    margin_x = float(np.floor(float(roi_width) * float(margin_fraction)))
    margin_y = float(np.floor(float(roi_height) * float(margin_fraction)))
    left = float(np.clip(margin_x, 0.0, max_x))
    right = float(np.clip(max_x - margin_x, 0.0, max_x))
    top = float(np.clip(margin_y, 0.0, max_y))
    bottom = float(np.clip(max_y - margin_y, 0.0, max_y))
    mid_x = float(np.clip(np.floor(max_x / 2.0), 0.0, max_x))
    mid_y = float(np.clip(np.floor(max_y / 2.0), 0.0, max_y))

    points: list[tuple[float, float]] = [
        (left, top),
        (right, top),
        (left, bottom),
        (right, bottom),
    ]
    if negative_point_policy == "border8":
        points.extend(
            [
                (mid_x, top),
                (mid_x, bottom),
                (left, mid_y),
                (right, mid_y),
            ]
        )

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        key = (int(round(x)), int(round(y)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((float(key[0]), float(key[1])))
    return np.asarray(deduped, dtype=np.float32)


def build_point_prompt_coords_labels(
    positive_point_xy: np.ndarray,
    *,
    roi_height: int,
    roi_width: int,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(positive_point_xy, dtype=np.float32)
    if positive.ndim == 1:
        positive = positive.reshape(1, 2)
    elif positive.ndim != 2 or int(positive.shape[1]) != 2:
        raise ValueError(
            f"positive_point_xy must have shape (2,) or (K,2), got {tuple(int(v) for v in positive.shape)}."
        )
    if int(positive.shape[0]) <= 0:
        raise ValueError("At least one positive prompt point is required.")
    negative = build_negative_prompt_points(
        roi_height=roi_height,
        roi_width=roi_width,
        negative_point_policy=negative_point_policy,
        margin_fraction=negative_point_margin_fraction,
    )
    if negative.size:
        coords = np.concatenate([positive, negative], axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(
            [
                np.ones((positive.shape[0],), dtype=np.int32),
                np.zeros((negative.shape[0],), dtype=np.int32),
            ],
            axis=0,
        )
        return coords, labels
    return positive.astype(np.float32, copy=False), np.ones((positive.shape[0],), dtype=np.int32)


def _resolve_positive_prompt_points_for_row(
    inputs: ResolvedSamInputs,
    row_idx: int,
    *,
    prompt_selection: PromptKeypointSelection,
) -> np.ndarray:
    points = np.asarray(inputs.keypoints_roi[int(row_idx), prompt_selection.indices, :], dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    in_bounds = finite.copy()
    in_bounds &= points[:, 0] >= 0.0
    in_bounds &= points[:, 1] >= 0.0
    in_bounds &= points[:, 0] < float(inputs.roi_width)
    in_bounds &= points[:, 1] < float(inputs.roi_height)
    return points[in_bounds].astype(np.float32, copy=False)


def _build_detect_box_batch(
    inputs: ResolvedSamInputs,
    row_indices: np.ndarray,
) -> list[np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    boxes: list[np.ndarray] = []
    for idx in rows.tolist():
        box_xyxy = project_bbox_norm_to_roi_xyxy(
            inputs.bbox_norm_coords[int(idx)],
            inputs.roi_coordinates_full[int(idx)],
            frame_height=int(inputs.frame_height),
            frame_width=int(inputs.frame_width),
            roi_height=int(inputs.roi_height),
            roi_width=int(inputs.roi_width),
        )
        boxes.append(box_xyxy.reshape(1, 4))
    return boxes


def _build_pose_roi_box_batch(
    inputs: ResolvedSamInputs,
    row_indices: np.ndarray,
    *,
    expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
) -> list[np.ndarray]:
    if inputs.pose_bbox_xyxy_roi is None:
        raise ValueError(
            f"{inputs.keypoint_group}/{inputs.keypoint_run} does not contain pose_bbox_xyxy_roi, "
            "so box_prompt_source='pose_roi' is unavailable."
        )
    rows = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    boxes: list[np.ndarray] = []
    for idx in rows.tolist():
        box_xyxy = expand_roi_box_xyxy(
            inputs.pose_bbox_xyxy_roi[int(idx)],
            roi_height=int(inputs.roi_height),
            roi_width=int(inputs.roi_width),
            expand_fraction=float(expand_fraction),
        )
        boxes.append(box_xyxy.reshape(1, 4))
    return boxes


def _build_roi_inset_box_batch(
    inputs: ResolvedSamInputs,
    row_indices: np.ndarray,
    *,
    inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
) -> list[np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    box_xyxy = build_roi_inset_box_xyxy(
        roi_height=int(inputs.roi_height),
        roi_width=int(inputs.roi_width),
        inset_fraction=float(inset_fraction),
    ).reshape(1, 4)
    return [box_xyxy.copy() for _ in rows.tolist()]


def build_sam_preview_batch(
    inputs: ResolvedSamInputs,
    eligibility: RowEligibility,
    *,
    prompt_selection: PromptKeypointSelection,
    max_rows: int = DEFAULT_PREPARE_COUNT,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    pose_box_expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
) -> PreparedSamBatch:
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")

    eligible_indices = np.flatnonzero(eligibility.eligible)
    if max_rows:
        eligible_indices = eligible_indices[:max_rows]
    else:
        eligible_indices = eligible_indices[:0]

    if eligible_indices.size == 0:
        return PreparedSamBatch(
            row_indices=np.asarray([], dtype=np.int32),
            images_rgb=[],
            point_coords_batch=[],
            point_labels_batch=[],
            box_batch=[] if use_box_prompt else None,
        )

    roi_batch = np.asarray(inputs.roi_images[eligible_indices])
    images_rgb = convert_roi_batch_to_rgb(roi_batch)
    point_prompts = [
        build_point_prompt_coords_labels(
            _resolve_positive_prompt_points_for_row(
                inputs,
                int(idx),
                prompt_selection=prompt_selection,
            ),
            roi_height=int(inputs.roi_height),
            roi_width=int(inputs.roi_width),
            negative_point_policy=negative_point_policy,
            negative_point_margin_fraction=negative_point_margin_fraction,
        )
        for idx in eligible_indices.tolist()
    ]
    point_coords_batch = [coords for coords, _ in point_prompts]
    point_labels_batch = [labels for _, labels in point_prompts]
    image_list = [np.asarray(image, dtype=np.uint8) for image in images_rgb]
    if use_box_prompt and box_prompt_source == "roi_inset":
        box_batch = _build_roi_inset_box_batch(
            inputs,
            eligible_indices,
            inset_fraction=roi_inset_fraction,
        )
    elif use_box_prompt and box_prompt_source == "pose_roi":
        box_batch = _build_pose_roi_box_batch(
            inputs,
            eligible_indices,
            expand_fraction=pose_box_expand_fraction,
        )
    elif use_box_prompt:
        box_batch = _build_detect_box_batch(inputs, eligible_indices)
    else:
        box_batch = None

    return PreparedSamBatch(
        row_indices=eligible_indices.astype(np.int32, copy=False),
        images_rgb=image_list,
        point_coords_batch=point_coords_batch,
        point_labels_batch=point_labels_batch,
        box_batch=box_batch,
    )


def build_sam_batch_for_rows(
    inputs: ResolvedSamInputs,
    row_indices: np.ndarray,
    *,
    prompt_selection: PromptKeypointSelection,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    pose_box_expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
) -> PreparedSamBatch:
    rows = np.asarray(row_indices, dtype=np.int32).reshape(-1)
    if rows.size == 0:
        return PreparedSamBatch(
            row_indices=np.asarray([], dtype=np.int32),
            images_rgb=[],
            point_coords_batch=[],
            point_labels_batch=[],
            box_batch=[] if use_box_prompt else None,
        )

    roi_batch = np.asarray(inputs.roi_images[rows])
    images_rgb = convert_roi_batch_to_rgb(roi_batch)
    point_prompts = [
        build_point_prompt_coords_labels(
            _resolve_positive_prompt_points_for_row(
                inputs,
                int(idx),
                prompt_selection=prompt_selection,
            ),
            roi_height=int(inputs.roi_height),
            roi_width=int(inputs.roi_width),
            negative_point_policy=negative_point_policy,
            negative_point_margin_fraction=negative_point_margin_fraction,
        )
        for idx in rows.tolist()
    ]
    return PreparedSamBatch(
        row_indices=rows.astype(np.int32, copy=False),
        images_rgb=[np.asarray(image, dtype=np.uint8) for image in images_rgb],
        point_coords_batch=[coords for coords, _ in point_prompts],
        point_labels_batch=[labels for _, labels in point_prompts],
        box_batch=(
            _build_roi_inset_box_batch(inputs, rows, inset_fraction=roi_inset_fraction)
            if use_box_prompt and box_prompt_source == "roi_inset"
            else _build_pose_roi_box_batch(inputs, rows, expand_fraction=pose_box_expand_fraction)
            if use_box_prompt and box_prompt_source == "pose_roi"
            else _build_detect_box_batch(inputs, rows)
            if use_box_prompt
            else None
        ),
    )


def _histogram_int(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    uniq, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(uniq.tolist(), counts.tolist())}


def _preview_batch_summary(batch: PreparedSamBatch) -> dict[str, Any]:
    if not batch.images_rgb:
        return {
            "prepared_rows": 0,
            "row_indices": [],
            "image_shape": None,
            "image_dtype": None,
            "point_coords_shape": None,
            "point_labels_shape": None,
            "box_shape": None,
            "positive_point_count": None,
            "negative_point_count": None,
        }

    first_image = batch.images_rgb[0]
    first_points = batch.point_coords_batch[0]
    first_labels = batch.point_labels_batch[0]
    return {
        "prepared_rows": len(batch.images_rgb),
        "row_indices": [int(idx) for idx in batch.row_indices.tolist()],
        "image_shape": [int(v) for v in first_image.shape],
        "image_dtype": str(first_image.dtype),
        "point_coords_shape": [int(v) for v in first_points.shape],
        "point_labels_shape": [int(v) for v in first_labels.shape],
        "box_shape": [int(v) for v in batch.box_batch[0].shape] if batch.box_batch else None,
        "positive_point_count": int(np.sum(first_labels == 1)),
        "negative_point_count": int(np.sum(first_labels == 0)),
    }


def _default_output_run(inputs: ResolvedSamInputs) -> str:
    return f"sam_subject_masks_from_{inputs.keypoint_run}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-logits64))).astype(np.float32, copy=False)


def _select_best_masks(
    row_indices: np.ndarray,
    masks_list: Sequence[np.ndarray],
    ious_list: Sequence[np.ndarray],
) -> SelectedMaskBatch:
    if len(masks_list) != len(ious_list) or len(masks_list) != int(row_indices.shape[0]):
        raise ValueError("SAM output length mismatch between row indices, masks, and IoU scores.")
    if len(masks_list) == 0:
        return SelectedMaskBatch(
            row_indices=np.asarray([], dtype=np.int32),
            binary=np.zeros((0, 0, 0), dtype=np.uint8),
            probs=np.zeros((0, 0, 0), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
        )

    first_mask = np.asarray(masks_list[0], dtype=np.float32)
    if first_mask.ndim != 3:
        raise ValueError(f"Expected SAM masks in (C,H,W) format, got {first_mask.shape}.")
    height = int(first_mask.shape[1])
    width = int(first_mask.shape[2])
    selected_binary = np.zeros((len(masks_list), height, width), dtype=np.uint8)
    selected_probs = np.zeros((len(masks_list), height, width), dtype=np.float32)
    selected_scores = np.zeros((len(masks_list),), dtype=np.float32)

    for idx, (mask_candidates, iou_candidates) in enumerate(zip(masks_list, ious_list)):
        mask_arr = np.asarray(mask_candidates, dtype=np.float32)
        iou_arr = np.asarray(iou_candidates, dtype=np.float32).reshape(-1)
        if mask_arr.ndim != 3 or mask_arr.shape[1:] != (height, width):
            raise ValueError(
                f"Row {idx} SAM mask shape mismatch: expected (*,{height},{width}), got {mask_arr.shape}."
            )
        if iou_arr.size == 0:
            raise ValueError(f"Row {idx} returned no IoU scores from SAM.")
        candidate_index = int(np.argmax(iou_arr))
        candidate_logits = mask_arr[candidate_index]
        selected_probs[idx] = _safe_sigmoid(candidate_logits)
        selected_binary[idx] = (candidate_logits > 0.0).astype(np.uint8, copy=False)
        selected_scores[idx] = float(iou_arr[candidate_index])

    return SelectedMaskBatch(
        row_indices=row_indices.astype(np.int32, copy=False),
        binary=selected_binary,
        probs=selected_probs,
        scores=selected_scores,
    )


def _compute_channel_metrics(
    binary_masks: np.ndarray,
    probs: np.ndarray,
) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    prob_arr = np.asarray(probs, dtype=np.float32)
    if binary.ndim != 3 or prob_arr.shape != binary.shape:
        raise ValueError("binary_masks and probs must both have shape (N,H,W).")

    n_rows = int(binary.shape[0])
    area_px = binary.sum(axis=(1, 2), dtype=np.int64).astype(np.float32)
    mask_present = area_px > 0.0
    prob_max = prob_arr.max(axis=(1, 2)).astype(np.float32, copy=False) if n_rows else np.zeros((0,), dtype=np.float32)

    centroid_xy = np.zeros((n_rows, 2), dtype=np.float32)
    centroid_valid = np.zeros((n_rows,), dtype=bool)
    bbox_xyxy = np.zeros((n_rows, 4), dtype=np.float32)
    bbox_valid = np.zeros((n_rows,), dtype=bool)

    for idx in range(n_rows):
        ys, xs = np.nonzero(binary[idx] > 0)
        if ys.size == 0:
            continue
        centroid_xy[idx, 0] = float(xs.mean())
        centroid_xy[idx, 1] = float(ys.mean())
        centroid_valid[idx] = True
        bbox_xyxy[idx] = np.asarray(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32,
        )
        bbox_valid[idx] = True

    return {
        "prob_max": prob_max,
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _copy_lineage_array(
    run_group: zarr.Group,
    crop_group: zarr.Group,
    name: str,
) -> None:
    source_arr = crop_group.get(name)
    if source_arr is None:
        return
    data = np.asarray(source_arr[:])
    run_group.create_array(name, data=data, overwrite=True)


def _build_subject_mask_summary(
    inputs: ResolvedSamInputs,
    eligibility: RowEligibility,
    prompt_selection: PromptKeypointSelection,
    selected: SelectedMaskBatch,
    *,
    output_run: str,
    checkpoint_path: str | None,
    sam3_root: str | None,
    multimask_output: bool,
    use_box_prompt: bool,
    box_prompt_source: str,
    roi_inset_fraction: float | None,
    pose_box_expand_fraction: float | None,
    negative_point_policy: str,
    negative_point_margin_fraction: float | None,
    device: str,
    duration_seconds: float,
) -> dict[str, Any]:
    mask_present = np.any(selected.binary > 0, axis=(1, 2)) if selected.binary.size else np.zeros((0,), dtype=bool)
    area_px = selected.binary.sum(axis=(1, 2), dtype=np.int64).astype(np.float32) if selected.binary.size else np.zeros((0,), dtype=np.float32)
    summary = {
        "rows_total": int(inputs.row_count),
        "rows_eligible": int(np.sum(eligibility.eligible)),
        "rows_segmented": int(selected.row_indices.shape[0]),
        "rows_with_nonempty_masks": int(np.sum(mask_present)),
        "rows_skipped_interpolated": int(np.sum(eligibility.skipped_interpolated)),
        "rows_with_any_prompt_point": int(np.sum(eligibility.prompt_point_finite)),
        "rows_with_any_in_bounds_prompt_point": int(np.sum(eligibility.prompt_point_in_bounds)),
        "rows_failed_prompt_finite": int(np.sum(~eligibility.prompt_point_finite)),
        "rows_failed_prompt_bounds": int(
            np.sum(eligibility.prompt_point_finite & ~eligibility.prompt_point_in_bounds)
        ),
        "prompt_keypoint_labels": list(prompt_selection.labels),
        "prompt_keypoint_count": int(len(prompt_selection.labels)),
        "prompt_point_count_min": int(eligibility.prompt_point_count.min()) if eligibility.prompt_point_count.size else 0,
        "prompt_point_count_mean": float(eligibility.prompt_point_count.mean()) if eligibility.prompt_point_count.size else 0.0,
        "prompt_point_count_max": int(eligibility.prompt_point_count.max()) if eligibility.prompt_point_count.size else 0,
        "rows_failed_success_policy": int(np.sum(~eligibility.success_ok)),
        "rows_failed_geometry_policy": int(np.sum(~eligibility.geometry_ok)),
        "rows_failed_usable_policy": int(np.sum(~eligibility.usable_ok)),
        "score_min": float(selected.scores.min()) if selected.scores.size else None,
        "score_mean": float(selected.scores.mean()) if selected.scores.size else None,
        "score_max": float(selected.scores.max()) if selected.scores.size else None,
        "area_px_min": float(area_px.min()) if area_px.size else None,
        "area_px_mean": float(area_px.mean()) if area_px.size else None,
        "area_px_max": float(area_px.max()) if area_px.size else None,
        "output_run": output_run,
        "crop_run": inputs.crop_run,
        "keypoint_group": inputs.keypoint_group,
        "keypoint_run": inputs.keypoint_run,
        "checkpoint_path": checkpoint_path,
        "sam3_root": sam3_root,
        "multimask_output": bool(multimask_output),
        "box_prompt_enabled": bool(use_box_prompt),
        "box_prompt_source": box_prompt_source if use_box_prompt else None,
        "roi_inset_fraction": float(roi_inset_fraction) if use_box_prompt and box_prompt_source == "roi_inset" else None,
        "pose_box_expand_fraction": float(pose_box_expand_fraction)
        if use_box_prompt and box_prompt_source == "pose_roi"
        else None,
        "negative_point_policy": negative_point_policy,
        "negative_point_margin_fraction": float(negative_point_margin_fraction)
        if negative_point_policy != "none"
        else None,
        "device": device,
        "duration_seconds": float(duration_seconds),
        "created_at_utc": _utc_now(),
    }
    return summary


def _load_sam3_builder(
    sam3_root: str | Path | None,
):
    root = _normalize_sam3_root(sam3_root)
    if root is None:
        raise RuntimeError("Could not resolve a SAM3 checkout. Set --sam3-root or PALETTE_SAM3_ROOT.")

    root_text = str(root)
    added_path = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        added_path = True

    try:
        builder_mod = importlib.import_module("sam3.model_builder")
        processor_mod = importlib.import_module("sam3.model.sam3_image_processor")
        pil_image_mod = importlib.import_module("PIL.Image")
        build_fn = getattr(builder_mod, "build_sam3_image_model", None)
        if build_fn is None:
            raise RuntimeError("sam3.model_builder does not expose build_sam3_image_model.")
        processor_cls = getattr(processor_mod, "Sam3Processor", None)
        if processor_cls is None:
            raise RuntimeError("sam3.model.sam3_image_processor does not expose Sam3Processor.")
        return root_text, build_fn, processor_cls, pil_image_mod, added_path
    except Exception:
        if added_path and root_text in sys.path:
            sys.path.remove(root_text)
        raise


def _resolve_runtime_device(build_model_fn, explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device
    torch_module = build_model_fn.__globals__.get("torch")
    try:
        if torch_module is not None and bool(torch_module.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _release_runtime_memory(build_model_fn, *, device: str | None) -> None:
    gc.collect()
    if device != "cuda":
        return
    torch_module = build_model_fn.__globals__.get("torch")
    if torch_module is None or not hasattr(torch_module, "cuda"):
        return
    cuda_mod = torch_module.cuda
    try:
        empty_cache = getattr(cuda_mod, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        pass
    try:
        ipc_collect = getattr(cuda_mod, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        pass


def _subject_mask_storage_chunks(n_rows: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(1, min(64, n_rows)),
        1,
        max(1, min(128, height)),
        max(1, min(128, width)),
    )


def _write_sparse_mask_channel(
    dest: zarr.Array,
    row_indices: np.ndarray,
    channel_index: int,
    values: np.ndarray,
) -> None:
    rows = np.asarray(row_indices, dtype=np.int64)
    data = np.asarray(values)
    if rows.size == 0:
        return
    if data.ndim == 3:
        data = data[:, None, :, :]
    if data.ndim != 4 or int(data.shape[0]) != int(rows.shape[0]):
        raise ValueError(
            f"Sparse mask write expects (N,H,W) or (N,C,H,W) values matching row indices, got rows={rows.shape} "
            f"values={data.shape}."
        )
    if rows.size > 1:
        order = np.argsort(rows, kind="stable")
        if not np.array_equal(order, np.arange(rows.size)):
            rows = rows[order]
            data = data[order]
    start = 0
    while start < rows.size:
        stop = start + 1
        while stop < rows.size and int(rows[stop]) == int(rows[stop - 1]) + 1:
            stop += 1
        row_start = int(rows[start])
        row_stop = int(rows[stop - 1]) + 1
        dest[row_start:row_stop, channel_index : channel_index + int(data.shape[1]), :, :] = data[start:stop]
        start = stop


def write_sam_subject_mask_run(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None,
    inputs: ResolvedSamInputs,
    eligibility: RowEligibility,
    selected: SelectedMaskBatch,
    crop_group: zarr.Group,
    prompt_selection: PromptKeypointSelection,
    output_run: str,
    overwrite: bool,
    checkpoint_path: str | None,
    sam3_root: str | None,
    multimask_output: bool,
    use_box_prompt: bool,
    box_prompt_source: str,
    roi_inset_fraction: float,
    pose_box_expand_fraction: float,
    negative_point_policy: str,
    negative_point_margin_fraction: float,
    device: str,
    duration_seconds: float,
) -> dict[str, Any]:
    parent = root.require_group("subject_mask_runs")
    if output_run in parent and not overwrite:
        raise ValueError(
            f"subject_mask_runs/{output_run} already exists. Pass --overwrite to replace it."
        )
    if output_run in parent and overwrite:
        del parent[output_run]

    n_rows = int(inputs.row_count)
    height = int(inputs.roi_height)
    width = int(inputs.roi_width)
    n_channels = len(SUBJECT_MASK_LABELS)
    created_at = _utc_now()

    run_group = parent.create_group(output_run)
    parent.attrs["latest"] = output_run

    run_group.attrs.update(
        {
            "source_crop_run": inputs.crop_run,
            **build_source_keypoints_attrs(inputs.keypoint_run, include_legacy_alias=True),
            "source_keypoint_group": inputs.keypoint_group,
            "label_schema_id": SUBJECT_MASK_LABEL_SCHEMA,
            "mask_labels": list(SUBJECT_MASK_LABELS),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "method": _sam_method_name(use_box_prompt, box_prompt_source, negative_point_policy),
            "run_semantics": "sam_body_mask_inference",
            "sam_prompt_policy": _sam_prompt_policy(use_box_prompt, box_prompt_source, negative_point_policy),
            "sam_multimask_output": bool(multimask_output),
            "sam_positive_keypoint_labels": list(prompt_selection.labels),
            "sam_positive_keypoint_count": int(len(prompt_selection.labels)),
            "sam_box_prompt_enabled": bool(use_box_prompt),
            "sam_box_prompt_source": box_prompt_source if use_box_prompt else None,
            "sam_roi_inset_fraction": float(roi_inset_fraction) if use_box_prompt and box_prompt_source == "roi_inset" else None,
            "sam_pose_box_expand_fraction": float(pose_box_expand_fraction)
            if use_box_prompt and box_prompt_source == "pose_roi"
            else None,
            "sam_negative_point_policy": negative_point_policy,
            "sam_negative_point_margin_fraction": float(negative_point_margin_fraction)
            if negative_point_policy != "none"
            else None,
            "sam_checkpoint_path": checkpoint_path,
            "sam3_root": sam3_root,
            "input_format": "gray" if inputs.roi_channels == 1 else "rgb",
            "probability_semantics": "sigmoid_selected_mask_logits",
            "sam_quality_score_semantics": "predicted_mask_quality",
            "probabilities_dtype": "float16",
            "probabilities_encoding": "unit_float",
            "created_at_utc": created_at,
        }
    )

    _copy_lineage_array(run_group, crop_group, "frame_indices")
    _copy_lineage_array(run_group, crop_group, "frame_counts")
    _copy_lineage_array(run_group, crop_group, "detection_indices")
    _copy_lineage_array(run_group, crop_group, "detection_source")

    storage_chunks = _subject_mask_storage_chunks(n_rows, height, width)
    masks_arr = run_group.create_array(
        "masks_roi",
        shape=(n_rows, n_channels, height, width),
        dtype=np.uint8,
        chunks=storage_chunks,
        fill_value=0,
        overwrite=True,
    )
    probs_arr = run_group.create_array(
        "mask_probs_roi",
        shape=(n_rows, n_channels, height, width),
        dtype=np.float16,
        chunks=storage_chunks,
        fill_value=np.float16(0.0),
        overwrite=True,
    )
    if selected.row_indices.size:
        _write_sparse_mask_channel(masks_arr, selected.row_indices, 0, selected.binary)
        _write_sparse_mask_channel(probs_arr, selected.row_indices, 0, selected.probs.astype(np.float16, copy=False))
    run_group.create_array(
        "available_channels",
        data=np.asarray(SUBJECT_MASK_AVAILABLE_CHANNELS, dtype=bool),
        overwrite=True,
    )
    write_subject_mask_component_provenance(
        run_group,
        component_name="subject_body",
        source_stage="subject_mask_runs",
        source_run=output_run,
        source_method=str(run_group.attrs["method"]),
        source_channels=["subject_body"],
        source_label_schema_id=SUBJECT_MASK_LABEL_SCHEMA,
        source_created_at_utc=created_at,
    )

    metrics_group = run_group.require_group("metrics")
    prob_max = np.zeros((n_rows, n_channels), dtype=np.float32)
    sam_quality_score = np.zeros((n_rows, n_channels), dtype=np.float32)
    mask_present = np.zeros((n_rows, n_channels), dtype=bool)
    area_px = np.zeros((n_rows, n_channels), dtype=np.float32)
    centroid_xy = np.zeros((n_rows, n_channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((n_rows, n_channels), dtype=bool)
    bbox_xyxy = np.zeros((n_rows, n_channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((n_rows, n_channels), dtype=bool)

    if selected.row_indices.size:
        channel_metrics = _compute_channel_metrics(selected.binary, selected.probs)
        prob_max[selected.row_indices, 0] = channel_metrics["prob_max"]
        sam_quality_score[selected.row_indices, 0] = selected.scores
        mask_present[selected.row_indices, 0] = channel_metrics["mask_present"]
        area_px[selected.row_indices, 0] = channel_metrics["area_px"]
        centroid_xy[selected.row_indices, 0, :] = channel_metrics["centroid_xy"]
        centroid_valid[selected.row_indices, 0] = channel_metrics["centroid_valid"]
        bbox_xyxy[selected.row_indices, 0, :] = channel_metrics["bbox_xyxy"]
        bbox_valid[selected.row_indices, 0] = channel_metrics["bbox_valid"]

    metrics_group.create_array("prob_max", data=prob_max, overwrite=True)
    metrics_group.create_array("sam_quality_score", data=sam_quality_score, overwrite=True)
    metrics_group.create_array("mask_present", data=mask_present, overwrite=True)
    metrics_group.create_array("area_px", data=area_px, overwrite=True)
    metrics_group.create_array("centroid_xy", data=centroid_xy, overwrite=True)
    metrics_group.create_array("centroid_valid", data=centroid_valid, overwrite=True)
    metrics_group.create_array("bbox_xyxy", data=bbox_xyxy, overwrite=True)
    metrics_group.create_array("bbox_valid", data=bbox_valid, overwrite=True)

    summary_statistics = _build_subject_mask_summary(
        inputs,
        eligibility,
        prompt_selection,
        selected,
        output_run=output_run,
        checkpoint_path=checkpoint_path,
        sam3_root=sam3_root,
        multimask_output=multimask_output,
        use_box_prompt=use_box_prompt,
        box_prompt_source=box_prompt_source,
        roi_inset_fraction=roi_inset_fraction,
        pose_box_expand_fraction=pose_box_expand_fraction,
        negative_point_policy=negative_point_policy,
        negative_point_margin_fraction=negative_point_margin_fraction,
        device=device,
        duration_seconds=duration_seconds,
    )
    run_group.attrs["duration_seconds"] = float(duration_seconds)
    run_group.attrs["summary_statistics"] = summary_statistics

    zarr_path_text = str(Path(zarr_path).expanduser().resolve()) if zarr_path is not None else None
    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=zarr_path_text,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage="subject_masks",
        command=" ".join(sys.argv) if sys.argv else "unknown",
        created_at_utc=created_at,
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
            "method": _sam_method_name(use_box_prompt, box_prompt_source, negative_point_policy),
            "run_semantics": "sam_body_mask_inference",
            "prompt_policy": _sam_prompt_policy(use_box_prompt, box_prompt_source, negative_point_policy),
            "positive_keypoint_labels": list(prompt_selection.labels),
            "positive_keypoint_count": int(len(prompt_selection.labels)),
            "multimask_output": bool(multimask_output),
            "box_prompt_enabled": bool(use_box_prompt),
            "box_prompt_source": box_prompt_source if use_box_prompt else None,
            "roi_inset_fraction": float(roi_inset_fraction) if use_box_prompt and box_prompt_source == "roi_inset" else None,
            "negative_point_policy": negative_point_policy,
            "negative_point_margin_fraction": float(negative_point_margin_fraction)
            if negative_point_policy != "none"
            else None,
            "device": device,
            "input_format": "gray" if inputs.roi_channels == 1 else "rgb",
            "probabilities_dtype": "float16",
            "probabilities_encoding": "unit_float",
        },
        inputs={
            "source_crop_run": inputs.crop_run,
            "source_keypoints_run": inputs.keypoint_run,
            "source_keypoint_group": inputs.keypoint_group,
            "sam_checkpoint_path": checkpoint_path,
            "sam3_root": sam3_root,
            "frame_source": crop_group.attrs.get("video_source_type", "zarr"),
            "source_video_path": crop_group.attrs.get("video_source_path"),
        },
    )
    write_stage_provenance(run_group, provenance)
    return summary_statistics


def run_sam_subject_mask_inference(
    zarr_path: str | Path,
    *,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "auto",
    output_run: str | None = None,
    sam3_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    skip_interpolated: bool = True,
    multimask_output: bool = True,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    pose_box_expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    overwrite: bool = False,
    device: str | None = None,
    no_hf_download: bool = False,
    positive_keypoint_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if box_prompt_source not in BOX_PROMPT_SOURCE_CHOICES:
        raise ValueError(
            f"box_prompt_source must be one of {BOX_PROMPT_SOURCE_CHOICES}, got {box_prompt_source!r}."
        )
    if negative_point_policy not in NEGATIVE_POINT_POLICY_CHOICES:
        raise ValueError(
            f"negative_point_policy must be one of {NEGATIVE_POINT_POLICY_CHOICES}, got {negative_point_policy!r}."
        )
    if use_box_prompt and box_prompt_source == "roi_inset":
        build_roi_inset_box_xyxy(
            roi_height=2,
            roi_width=2,
            inset_fraction=float(roi_inset_fraction),
        )
    if use_box_prompt and box_prompt_source == "pose_roi":
        expand_roi_box_xyxy(
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            roi_height=2,
            roi_width=2,
            expand_fraction=float(pose_box_expand_fraction),
        )
    if negative_point_policy != "none":
        build_negative_prompt_points(
            roi_height=2,
            roi_width=2,
            negative_point_policy=negative_point_policy,
            margin_fraction=float(negative_point_margin_fraction),
        )

    root = open_zarr_root(zarr_path, mode="r+")
    _, crop_group, resolved_crop_run = resolve_materialized_crop_run(root, crop_run=crop_run)
    inputs = resolve_sam_subject_inputs(
        root,
        crop_run=resolved_crop_run,
        keypoint_run=keypoint_run,
        keypoint_group=keypoint_group,
        zarr_path=zarr_path,
    )
    prompt_selection = resolve_prompt_keypoint_selection(
        inputs,
        positive_keypoint_labels=positive_keypoint_labels,
    )
    eligibility = compute_row_eligibility(
        inputs,
        prompt_selection=prompt_selection,
        skip_interpolated=skip_interpolated,
    )
    eligible_indices = np.flatnonzero(eligibility.eligible).astype(np.int32, copy=False)
    resolved_output_run = output_run or _default_output_run(inputs)

    if no_hf_download and checkpoint_path is None:
        raise ValueError("--no-hf-download requires --checkpoint /path/to/sam3.pt.")

    sam3_root_text, build_model_fn, processor_cls, pil_image_mod, added_path = _load_sam3_builder(sam3_root)
    resolved_device = _resolve_runtime_device(build_model_fn, device)
    checkpoint_text = str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None

    model = None
    processor = None
    try:
        start = time.perf_counter()
        collected_binary: list[np.ndarray] = []
        collected_probs: list[np.ndarray] = []
        collected_scores: list[np.ndarray] = []
        collected_rows: list[np.ndarray] = []

        if eligible_indices.size:
            model = build_model_fn(
                checkpoint_path=checkpoint_text,
                load_from_HF=not no_hf_download,
                device=resolved_device,
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
            )
            processor = processor_cls(model)
            image_fromarray = getattr(pil_image_mod, "fromarray", None)
            if image_fromarray is None:
                raise RuntimeError("PIL.Image does not expose fromarray.")

            for batch_start in range(0, int(eligible_indices.shape[0]), int(batch_size)):
                batch_rows = eligible_indices[batch_start : batch_start + int(batch_size)]
                preview = build_sam_batch_for_rows(
                    inputs,
                    batch_rows,
                    prompt_selection=prompt_selection,
                    use_box_prompt=use_box_prompt,
                    box_prompt_source=box_prompt_source,
                    roi_inset_fraction=roi_inset_fraction,
                    pose_box_expand_fraction=pose_box_expand_fraction,
                    negative_point_policy=negative_point_policy,
                    negative_point_margin_fraction=negative_point_margin_fraction,
                )
                pil_images = [image_fromarray(image) for image in preview.images_rgb]
                inference_state = processor.set_image_batch(pil_images)
                masks_list, ious_list, _ = model.predict_inst_batch(
                    inference_state,
                    preview.point_coords_batch,
                    preview.point_labels_batch,
                    box_batch=preview.box_batch,
                    multimask_output=bool(multimask_output),
                    return_logits=True,
                    normalize_coords=True,
                )
                selected = _select_best_masks(batch_rows, masks_list, ious_list)
                collected_rows.append(selected.row_indices)
                collected_binary.append(selected.binary)
                collected_probs.append(selected.probs)
                collected_scores.append(selected.scores)
                del preview
                del pil_images
                del inference_state
                del masks_list
                del ious_list
                del selected

        duration = float(time.perf_counter() - start)
        if collected_rows:
            selected_batch = SelectedMaskBatch(
                row_indices=np.concatenate(collected_rows, axis=0).astype(np.int32, copy=False),
                binary=np.concatenate(collected_binary, axis=0).astype(np.uint8, copy=False),
                probs=np.concatenate(collected_probs, axis=0).astype(np.float32, copy=False),
                scores=np.concatenate(collected_scores, axis=0).astype(np.float32, copy=False),
            )
        else:
            selected_batch = SelectedMaskBatch(
                row_indices=np.asarray([], dtype=np.int32),
                binary=np.zeros((0, inputs.roi_height, inputs.roi_width), dtype=np.uint8),
                probs=np.zeros((0, inputs.roi_height, inputs.roi_width), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
            )

        summary_statistics = write_sam_subject_mask_run(
            root,
            zarr_path=zarr_path,
            inputs=inputs,
            eligibility=eligibility,
            selected=selected_batch,
            crop_group=crop_group,
            prompt_selection=prompt_selection,
            output_run=resolved_output_run,
            overwrite=overwrite,
            checkpoint_path=checkpoint_text,
            sam3_root=sam3_root_text,
            multimask_output=multimask_output,
            use_box_prompt=use_box_prompt,
            box_prompt_source=box_prompt_source,
            roi_inset_fraction=roi_inset_fraction,
            pose_box_expand_fraction=pose_box_expand_fraction,
            negative_point_policy=negative_point_policy,
            negative_point_margin_fraction=negative_point_margin_fraction,
            device=resolved_device,
            duration_seconds=duration,
        )
    finally:
        del processor
        del model
        _release_runtime_memory(build_model_fn, device=resolved_device)
        if added_path and sam3_root_text in sys.path:
            sys.path.remove(sam3_root_text)

    return {
        "status": "updated",
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "crop_run": inputs.crop_run,
        "keypoint_group": inputs.keypoint_group,
        "keypoint_run": inputs.keypoint_run,
        "positive_keypoint_labels": list(prompt_selection.labels),
        "output_run": resolved_output_run,
        "rows_eligible": int(np.sum(eligibility.eligible)),
        "rows_segmented": int(summary_statistics["rows_segmented"]),
        "rows_with_nonempty_masks": int(summary_statistics["rows_with_nonempty_masks"]),
        "duration_seconds": duration,
        "device": resolved_device,
        "checkpoint_path": checkpoint_text,
        "sam3_root": sam3_root_text,
    }


def inspect_sam_subject_archive(
    root: zarr.Group,
    *,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "auto",
    zarr_path: str | Path | None = None,
    skip_interpolated: bool = True,
    prepare_count: int = DEFAULT_PREPARE_COUNT,
    output_run: str | None = None,
    sam3_root: str | Path | None = None,
    inspect_runtime: bool = True,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    pose_box_expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    positive_keypoint_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    inputs = resolve_sam_subject_inputs(
        root,
        crop_run=crop_run,
        keypoint_run=keypoint_run,
        keypoint_group=keypoint_group,
        zarr_path=zarr_path,
    )
    prompt_selection = resolve_prompt_keypoint_selection(
        inputs,
        positive_keypoint_labels=positive_keypoint_labels,
    )
    eligibility = compute_row_eligibility(
        inputs,
        prompt_selection=prompt_selection,
        skip_interpolated=skip_interpolated,
    )
    preview_batch = build_sam_preview_batch(
        inputs,
        eligibility,
        prompt_selection=prompt_selection,
        max_rows=prepare_count,
        use_box_prompt=use_box_prompt,
        box_prompt_source=box_prompt_source,
        roi_inset_fraction=roi_inset_fraction,
        pose_box_expand_fraction=pose_box_expand_fraction,
        negative_point_policy=negative_point_policy,
        negative_point_margin_fraction=negative_point_margin_fraction,
    )

    summary: dict[str, Any] = {
        "crop_run": inputs.crop_run,
        "keypoint_group": inputs.keypoint_group,
        "keypoint_run": inputs.keypoint_run,
        "keypoint_labels": list(inputs.keypoint_labels),
        "positive_keypoint_labels": list(prompt_selection.labels),
        "row_count": int(inputs.row_count),
        "roi_shape": [int(inputs.roi_height), int(inputs.roi_width)],
        "roi_channels": int(inputs.roi_channels),
        "roi_dtype": str(inputs.roi_images.dtype),
        "alignment": {
            "status": "ok",
            "warnings": list(inputs.warnings),
        },
        "detection_source_histogram": _histogram_int(inputs.detection_source),
        "eligibility": {
            "eligible_rows": int(np.sum(eligibility.eligible)),
            "ineligible_rows": int(inputs.row_count - np.sum(eligibility.eligible)),
            "rows_with_any_prompt_point": int(np.sum(eligibility.prompt_point_finite)),
            "rows_with_any_in_bounds_prompt_point": int(np.sum(eligibility.prompt_point_in_bounds)),
            "failed_prompt_rows": int(np.sum(~eligibility.prompt_point_finite)),
            "off_image_prompt_rows": int(
                np.sum(eligibility.prompt_point_finite & ~eligibility.prompt_point_in_bounds)
            ),
            "prompt_point_count_min": int(eligibility.prompt_point_count.min()) if eligibility.prompt_point_count.size else 0,
            "prompt_point_count_mean": float(eligibility.prompt_point_count.mean()) if eligibility.prompt_point_count.size else 0.0,
            "prompt_point_count_max": int(eligibility.prompt_point_count.max()) if eligibility.prompt_point_count.size else 0,
            "success_flag_failed_rows": int(np.sum(~eligibility.success_ok)),
            "geometry_invalid_rows": int(np.sum(~eligibility.geometry_ok)),
            "usable_keypoints_failed_rows": int(np.sum(~eligibility.usable_ok)),
            "skipped_interpolated_rows": int(np.sum(eligibility.skipped_interpolated)),
            "eligible_row_indices_preview": [
                int(idx) for idx in np.flatnonzero(eligibility.eligible)[: min(16, int(np.sum(eligibility.eligible)))].tolist()
            ],
        },
        "prepared_batch": _preview_batch_summary(preview_batch),
        "planned_output": {
            "output_run": output_run or _default_output_run(inputs),
            "label_schema_id": SUBJECT_MASK_LABEL_SCHEMA,
            "mask_labels": list(SUBJECT_MASK_LABELS),
            "available_channels": list(SUBJECT_MASK_AVAILABLE_CHANNELS),
            "mode": "body_only",
            "sam_prompt_policy": _sam_prompt_policy(use_box_prompt, box_prompt_source, negative_point_policy),
            "sam_box_prompt_source": box_prompt_source if use_box_prompt else None,
            "sam_roi_inset_fraction": float(roi_inset_fraction) if use_box_prompt and box_prompt_source == "roi_inset" else None,
            "sam_pose_box_expand_fraction": float(pose_box_expand_fraction)
            if use_box_prompt and box_prompt_source == "pose_roi"
            else None,
            "sam_negative_point_policy": negative_point_policy,
            "sam_negative_point_margin_fraction": float(negative_point_margin_fraction)
            if negative_point_policy != "none"
            else None,
            "sam_positive_keypoint_labels": list(prompt_selection.labels),
        },
    }
    if inspect_runtime:
        summary["sam3_runtime"] = _inspect_sam3_runtime(sam3_root)
    return summary


def inspect_sam_subject_archive_path(
    zarr_path: str | Path,
    *,
    crop_run: str | None = None,
    keypoint_run: str | None = None,
    keypoint_group: str = "auto",
    skip_interpolated: bool = True,
    prepare_count: int = DEFAULT_PREPARE_COUNT,
    output_run: str | None = None,
    sam3_root: str | Path | None = None,
    inspect_runtime: bool = True,
    use_box_prompt: bool = True,
    box_prompt_source: str = DEFAULT_BOX_PROMPT_SOURCE,
    roi_inset_fraction: float = DEFAULT_ROI_INSET_FRACTION,
    pose_box_expand_fraction: float = DEFAULT_POSE_BOX_EXPAND_FRACTION,
    negative_point_policy: str = DEFAULT_NEGATIVE_POINT_POLICY,
    negative_point_margin_fraction: float = DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
    positive_keypoint_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = open_zarr_root(zarr_path, mode="r")
    summary = inspect_sam_subject_archive(
        root,
        crop_run=crop_run,
        keypoint_run=keypoint_run,
        keypoint_group=keypoint_group,
        skip_interpolated=skip_interpolated,
        prepare_count=prepare_count,
        output_run=output_run,
        sam3_root=sam3_root,
        inspect_runtime=inspect_runtime,
        use_box_prompt=use_box_prompt,
        box_prompt_source=box_prompt_source,
        roi_inset_fraction=roi_inset_fraction,
        pose_box_expand_fraction=pose_box_expand_fraction,
        negative_point_policy=negative_point_policy,
        negative_point_margin_fraction=negative_point_margin_fraction,
        positive_keypoint_labels=positive_keypoint_labels,
        zarr_path=zarr_path,
    )
    summary["zarr_path"] = str(Path(zarr_path).expanduser().resolve())
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("SAM Subject-Mask Canary")
    print(f"  zarr={summary.get('zarr_path', '(in-memory)')}")
    print(f"  crop_run={summary['crop_run']}")
    print(f"  keypoint_run={summary['keypoint_group']}/{summary['keypoint_run']}")
    print(f"  keypoint_labels={summary.get('keypoint_labels', [])}")
    print(f"  positive_keypoint_labels={summary.get('positive_keypoint_labels', [])}")
    print(
        f"  rows={summary['row_count']} roi_shape={tuple(summary['roi_shape'])} "
        f"channels={summary['roi_channels']} dtype={summary['roi_dtype']}"
    )
    print(f"  detection_source_histogram={summary['detection_source_histogram']}")

    alignment = summary["alignment"]
    print(f"  alignment={alignment['status']}")
    for warning in alignment.get("warnings", []):
        print(f"    warning: {warning}")

    eligibility = summary["eligibility"]
    print(
        "  eligibility="
        f"{eligibility['eligible_rows']}/{summary['row_count']} "
        f"(failed_prompt={eligibility['failed_prompt_rows']}, "
        f"off_image={eligibility['off_image_prompt_rows']}, "
        f"success_fail={eligibility['success_flag_failed_rows']}, "
        f"geometry_fail={eligibility['geometry_invalid_rows']}, "
        f"usable_fail={eligibility['usable_keypoints_failed_rows']}, "
        f"interpolated_skipped={eligibility['skipped_interpolated_rows']})"
    )

    preview = summary["prepared_batch"]
    print(
        "  prepared_batch="
        f"{preview['prepared_rows']} rows "
        f"row_indices={preview['row_indices'][:8]}"
    )
    if preview["image_shape"] is not None:
        print(
            f"    first_image_shape={tuple(preview['image_shape'])} "
            f"dtype={preview['image_dtype']} "
            f"point_coords_shape={tuple(preview['point_coords_shape'])}"
        )
        if preview["box_shape"] is not None:
            print(f"    first_box_shape={tuple(preview['box_shape'])}")
        if preview["positive_point_count"] is not None:
            print(f"    first_positive_point_count={preview['positive_point_count']}")
        if preview["negative_point_count"] is not None:
            print(f"    first_negative_point_count={preview['negative_point_count']}")

    planned = summary["planned_output"]
    print(
        f"  planned_output={planned['output_run']} "
        f"schema={planned['label_schema_id']} "
        f"available={planned['available_channels']} "
        f"prompt_policy={planned['sam_prompt_policy']}"
    )

    runtime = summary.get("sam3_runtime")
    if isinstance(runtime, dict):
        print(
            "  sam3_runtime="
            f"available={runtime['available']} "
            f"predictor={runtime['predictor_available']} "
            f"builder={runtime['builder_available']}"
        )
        if runtime.get("root"):
            print(f"    root={runtime['root']}")
        if runtime.get("error"):
            print(f"    error={runtime['error']}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette training or analysis zarr to inspect.")
    parser.add_argument("--crop-run", type=str, help="Materialized crop_runs/<run> to read (default: latest materialized).")
    parser.add_argument(
        "--keypoint-run",
        type=str,
        help="Keypoint run name (default: latest refined_keypoints_runs, else keypoints_runs).",
    )
    parser.add_argument(
        "--keypoint-group",
        choices=list(KEYPOINT_GROUP_CHOICES),
        default="auto",
        help="Keypoint parent group preference (default: auto).",
    )
    parser.add_argument(
        "--prepare-count",
        type=int,
        default=DEFAULT_PREPARE_COUNT,
        help=f"Maximum eligible rows to convert into SAM-ready RGB/prompt batches (default: {DEFAULT_PREPARE_COUNT}).",
    )
    parser.add_argument(
        "--output-run",
        type=str,
        help="Planned subject_mask_runs/<run> name for the future write-back step.",
    )
    parser.add_argument(
        "--sam3-root",
        type=Path,
        help="Optional SAM3 checkout root to probe for predictor/runtime availability.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local path to sam3.pt. If omitted, SAM3 may try Hugging Face unless --no-hf-download is set.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Inference device for --apply (default: auto-detect).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFER_BATCH_SIZE,
        help=f"Eligible ROI batch size for SAM inference when --apply is used (default: {DEFAULT_INFER_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="Use multimask_output=False during SAM inference.",
    )
    parser.add_argument(
        "--box-prompt-source",
        choices=list(BOX_PROMPT_SOURCE_CHOICES),
        default=DEFAULT_BOX_PROMPT_SOURCE,
        help=f"Box prompt source when box prompting is enabled (default: {DEFAULT_BOX_PROMPT_SOURCE}).",
    )
    parser.add_argument(
        "--roi-inset-fraction",
        type=float,
        default=DEFAULT_ROI_INSET_FRACTION,
        help=(
            "Inset fraction for --box-prompt-source roi_inset, as a fraction of ROI width/height "
            f"(default: {DEFAULT_ROI_INSET_FRACTION})."
        ),
    )
    parser.add_argument(
        "--pose-box-expand-fraction",
        type=float,
        default=DEFAULT_POSE_BOX_EXPAND_FRACTION,
        help=(
            "Expansion fraction for --box-prompt-source pose_roi, applied to the persisted ROI-local pose box "
            f"(default: {DEFAULT_POSE_BOX_EXPAND_FRACTION})."
        ),
    )
    parser.add_argument(
        "--negative-point-policy",
        choices=list(NEGATIVE_POINT_POLICY_CHOICES),
        default=DEFAULT_NEGATIVE_POINT_POLICY,
        help=f"Add fixed background negative points to the SAM prompt (default: {DEFAULT_NEGATIVE_POINT_POLICY}).",
    )
    parser.add_argument(
        "--negative-point-margin-fraction",
        type=float,
        default=DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION,
        help=(
            "Border inset fraction for negative points, as a fraction of ROI width/height "
            f"(default: {DEFAULT_NEGATIVE_POINT_MARGIN_FRACTION})."
        ),
    )
    parser.add_argument(
        "--positive-keypoint-labels",
        nargs="+",
        help=(
            "Optional keypoint labels to use as positive SAM prompts. "
            "Defaults to all labels available on the resolved keypoint run."
        ),
    )
    parser.add_argument(
        "--no-box-prompt",
        action="store_true",
        help="Disable the ROI-local box prompt and use only positive keypoint prompts.",
    )
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Do not allow SAM3 to download checkpoints from Hugging Face when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing subject_mask_runs/<run> when --apply is used.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run SAM inference and write a new body-only subject_mask_runs/<run>.",
    )
    parser.add_argument(
        "--include-interpolated",
        action="store_true",
        help="Allow rows with detection_source != 0 to remain eligible.",
    )
    parser.add_argument("--json", action="store_true", help="Print the inspection summary as JSON.")
    args = parser.parse_args(argv)

    if args.apply:
        summary = run_sam_subject_mask_inference(
            args.zarr_path,
            crop_run=args.crop_run,
            keypoint_run=args.keypoint_run,
            keypoint_group=args.keypoint_group,
            output_run=args.output_run,
            sam3_root=args.sam3_root,
            checkpoint_path=args.checkpoint,
            batch_size=int(args.batch_size),
            skip_interpolated=not args.include_interpolated,
            multimask_output=not args.single_mask,
            use_box_prompt=not args.no_box_prompt,
            box_prompt_source=args.box_prompt_source,
            roi_inset_fraction=float(args.roi_inset_fraction),
            pose_box_expand_fraction=float(args.pose_box_expand_fraction),
            negative_point_policy=args.negative_point_policy,
            negative_point_margin_fraction=float(args.negative_point_margin_fraction),
            overwrite=bool(args.overwrite),
            device=args.device,
            no_hf_download=bool(args.no_hf_download),
            positive_keypoint_labels=args.positive_keypoint_labels,
        )
    else:
        summary = inspect_sam_subject_archive_path(
            args.zarr_path,
            crop_run=args.crop_run,
            keypoint_run=args.keypoint_run,
            keypoint_group=args.keypoint_group,
            skip_interpolated=not args.include_interpolated,
            prepare_count=int(args.prepare_count),
            output_run=args.output_run,
            sam3_root=args.sam3_root,
            inspect_runtime=True,
            use_box_prompt=not args.no_box_prompt,
            box_prompt_source=args.box_prompt_source,
            roi_inset_fraction=float(args.roi_inset_fraction),
            pose_box_expand_fraction=float(args.pose_box_expand_fraction),
            negative_point_policy=args.negative_point_policy,
            negative_point_margin_fraction=float(args.negative_point_margin_fraction),
            positive_keypoint_labels=args.positive_keypoint_labels,
        )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        if args.apply:
            print("SAM Subject-Mask Apply")
            print(f"  zarr={summary['zarr_path']}")
            print(f"  crop_run={summary['crop_run']}")
            print(f"  keypoint_run={summary['keypoint_group']}/{summary['keypoint_run']}")
            print(f"  output_run=subject_mask_runs/{summary['output_run']}")
            print(
                f"  segmented={summary['rows_segmented']}/{summary['rows_eligible']} "
                f"nonempty={summary['rows_with_nonempty_masks']} "
                f"duration_seconds={summary['duration_seconds']:.2f}"
            )
            print(f"  device={summary['device']}")
            if summary.get("checkpoint_path"):
                print(f"  checkpoint={summary['checkpoint_path']}")
            if summary.get("sam3_root"):
                print(f"  sam3_root={summary['sam3_root']}")
        else:
            _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
