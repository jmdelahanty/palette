#!/usr/bin/env python3
"""Export and validate merged eye-mask-training Zarr artifacts.

This module provides:
- ``export_merged_eye_mask_training_zarr``: scaffold exporter for a single source Zarr.
- ``validate_merged_eye_mask_training_zarr``: contract validator for merged artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)
    text = text.strip()
    return text or None


def _normalize_input_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    return None


def _normalize_label_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"lr", "left_right", "left-right"}:
        return "lr"
    if text in {"union", "merged"}:
        return "union"
    return None


def _iter_chunk_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterable[Tuple[slice, ...]]:
    if not chunks:
        yield tuple(slice(0, int(dim)) for dim in shape)
        return
    chunk_dims = []
    for axis, dim in enumerate(shape):
        chunk = chunks[axis] if axis < len(chunks) else chunks[-1]
        if int(chunk) <= 0:
            chunk = dim
        chunk_dims.append(int(chunk))
    grid = [int(math.ceil(int(dim) / int(chunk))) for dim, chunk in zip(shape, chunk_dims)]
    for idx in np.ndindex(*grid):
        slices: List[slice] = []
        for axis, chunk_idx in enumerate(idx):
            start = int(chunk_idx) * int(chunk_dims[axis])
            stop = min(start + int(chunk_dims[axis]), int(shape[axis]))
            slices.append(slice(start, stop))
        yield tuple(slices)


def _copy_array(src: zarr.Array, dest_group: zarr.Group, name: str) -> zarr.Array:
    chunks = src.chunks if src.chunks is not None else None
    dest = dest_group.create_array(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=chunks,
        overwrite=True,
    )
    if chunks is None:
        dest[...] = src[...]
        return dest
    for slc in _iter_chunk_slices(tuple(int(v) for v in src.shape), tuple(int(v) for v in chunks)):
        dest[slc] = src[slc]
    return dest


def _write_string_array(group: zarr.Group, name: str, values: Sequence[str]) -> zarr.Array:
    arr = group.create_array(
        name,
        shape=(int(len(values)),),
        dtype=VariableLengthUTF8(),
        chunks=(max(1, min(65536, int(len(values)) or 1)),),
        overwrite=True,
    )
    arr[:] = np.asarray([str(v) for v in values], dtype=object)
    return arr


def _resolve_run_name(root: zarr.Group, parent_name: str, explicit: Optional[str]) -> str:
    parent = root.get(parent_name)
    if not isinstance(parent, zarr.Group):
        raise ValueError(f"Missing required group '{parent_name}'.")
    if explicit:
        if explicit not in parent:
            raise ValueError(f"Run '{explicit}' not found under {parent_name}.")
        return str(explicit)
    latest = parent.attrs.get("latest")
    latest_text = _as_text(latest)
    if latest_text and latest_text in parent:
        return latest_text
    names = sorted(str(name) for name in parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
    if not names:
        raise ValueError(f"No runs found under {parent_name}.")
    return str(names[-1])


def _resolve_eye_source(
    root: zarr.Group,
    *,
    eye_stage: str,
    eye_run: Optional[str],
) -> Tuple[str, str, zarr.Group]:
    if eye_stage not in {"auto", "eye_masks_runs", "refined_eye_masks_runs"}:
        raise ValueError(f"Unsupported eye_stage '{eye_stage}'.")

    stage_order = (
        ["refined_eye_masks_runs", "eye_masks_runs"]
        if eye_stage == "auto"
        else [eye_stage]
    )

    if eye_run:
        for stage in stage_order:
            parent = root.get(stage)
            if isinstance(parent, zarr.Group) and eye_run in parent:
                return stage, str(eye_run), parent[str(eye_run)]
        raise ValueError(f"Eye run '{eye_run}' not found in selected stage(s): {stage_order}.")

    for stage in stage_order:
        parent = root.get(stage)
        if not isinstance(parent, zarr.Group):
            continue
        latest = _as_text(parent.attrs.get("latest"))
        if latest and latest in parent:
            return stage, latest, parent[latest]
        names = sorted(str(name) for name in parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
        if names:
            return stage, str(names[-1]), parent[str(names[-1])]
    raise ValueError(f"No eye-mask runs found in selected stage(s): {stage_order}.")


def _resolve_reason_source(run_group: zarr.Group) -> Optional[zarr.Array]:
    metrics_group = run_group.get("metrics")
    if isinstance(metrics_group, zarr.Group):
        reason_arr = metrics_group.get("reason")
        if isinstance(reason_arr, zarr.Array):
            return reason_arr
    reason_arr = run_group.get("reason")
    if isinstance(reason_arr, zarr.Array):
        return reason_arr
    return None


def _resolve_mask_probs_name(run_group: zarr.Group) -> Optional[str]:
    for candidate in ("mask_probs_roi_refined", "mask_probs_roi"):
        if candidate in run_group:
            return candidate
    return None


def _normalized_split_ratios(
    train: float,
    val: float,
    test: float,
) -> Tuple[float, float, float]:
    train_v = max(0.0, float(train))
    val_v = max(0.0, float(val))
    test_v = max(0.0, float(test))
    total = train_v + val_v + test_v
    if total <= 0.0:
        raise ValueError("At least one split ratio must be > 0.")
    return train_v / total, val_v / total, test_v / total


def _make_split_indices(
    total_samples: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(total_samples)
    if total <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    tr, vr, ter = _normalized_split_ratios(train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(int(seed)).permutation(total).astype(np.int64, copy=False)
    train_count = int(round(float(total) * tr))
    val_count = int(round(float(total) * vr))
    train_count = max(0, min(train_count, total))
    val_count = max(0, min(val_count, total - train_count))
    test_count = total - train_count - val_count
    if ter <= 0.0:
        val_count = total - train_count
        test_count = 0

    train_idx = order[:train_count]
    val_idx = order[train_count: train_count + val_count]
    test_idx = order[train_count + val_count: train_count + val_count + test_count]
    return train_idx, val_idx, test_idx


@dataclass
class EyeExportSelection:
    crop_run: str
    eye_stage: str
    eye_run: str
    total_samples: int
    channels: int
    mask_probs_name: Optional[str]


def _select_source_runs(
    source_root: zarr.Group,
    *,
    crop_run: Optional[str],
    eye_stage: str,
    eye_run: Optional[str],
) -> Tuple[EyeExportSelection, zarr.Group, zarr.Group]:
    stage_name, run_name, eye_group = _resolve_eye_source(
        source_root,
        eye_stage=eye_stage,
        eye_run=eye_run,
    )

    selected_crop = _as_text(crop_run)
    if selected_crop is None:
        selected_crop = _as_text(eye_group.attrs.get("source_crop_run"))
    if selected_crop is None:
        selected_crop = _resolve_run_name(source_root, "crop_runs", explicit=None)

    crop_parent = source_root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group) or selected_crop not in crop_parent:
        raise ValueError(f"Crop run '{selected_crop}' not found under crop_runs.")
    crop_group = crop_parent[str(selected_crop)]

    if "roi_images" not in crop_group:
        raise ValueError(f"crop_runs/{selected_crop} missing roi_images.")
    if "bbox_norm_coords" not in crop_group:
        raise ValueError(f"crop_runs/{selected_crop} missing bbox_norm_coords.")
    if "masks_roi" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing masks_roi.")
    if "ellipse_params" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing ellipse_params.")
    if "ellipse_success" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing ellipse_success.")

    roi_images = crop_group["roi_images"]
    masks_roi = eye_group["masks_roi"]
    if int(roi_images.shape[0]) != int(masks_roi.shape[0]):
        raise ValueError(
            f"Row mismatch: crop_runs/{selected_crop}/roi_images has {roi_images.shape[0]} rows "
            f"but {stage_name}/{run_name}/masks_roi has {masks_roi.shape[0]} rows."
        )

    selection = EyeExportSelection(
        crop_run=str(selected_crop),
        eye_stage=stage_name,
        eye_run=run_name,
        total_samples=int(masks_roi.shape[0]),
        channels=int(masks_roi.shape[1]) if masks_roi.ndim >= 2 else 0,
        mask_probs_name=_resolve_mask_probs_name(eye_group),
    )
    return selection, crop_group, eye_group


def export_merged_eye_mask_training_zarr(
    source_zarr: Path,
    out_zarr: Path,
    *,
    crop_run: Optional[str] = None,
    eye_stage: str = "auto",
    eye_run: Optional[str] = None,
    run_name: str = "merged_export_smoke",
    input_format: str = "gray",
    label_mode: str = "lr",
    split_train: float = 0.8,
    split_val: float = 0.2,
    split_test: float = 0.0,
    split_seed: int = 42,
    overwrite: bool = False,
    validate: bool = True,
) -> Dict[str, Any]:
    source_path = Path(source_zarr).expanduser().resolve()
    out_path = Path(out_zarr).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source zarr does not exist: {source_path}")

    normalized_input_format = _normalize_input_format(input_format)
    if normalized_input_format is None:
        raise ValueError(f"Unsupported input_format '{input_format}'. Expected gray or rgb.")
    normalized_label_mode = _normalize_label_mode(label_mode)
    if normalized_label_mode is None:
        raise ValueError(f"Unsupported label_mode '{label_mode}'. Expected lr or union.")

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {out_path}")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_root = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    selection, crop_group, eye_group = _select_source_runs(
        src_root,
        crop_run=crop_run,
        eye_stage=eye_stage,
        eye_run=eye_run,
    )

    roi_images = crop_group["roi_images"]
    bbox_norm = crop_group["bbox_norm_coords"]
    crop_bbox = crop_group.get("crop_bbox_norm_coords")
    crop_frame_indices = crop_group.get("frame_indices")
    crop_detection_source = crop_group.get("detection_source")

    masks_roi = eye_group["masks_roi"]
    ellipse_params = eye_group["ellipse_params"]
    ellipse_success = eye_group["ellipse_success"]
    eye_separation = eye_group.get("eye_separation")
    reason_src = _resolve_reason_source(eye_group)
    mask_probs_name = selection.mask_probs_name
    mask_probs_src = eye_group[mask_probs_name] if mask_probs_name else None

    total_samples = int(selection.total_samples)
    local_frame_indices = np.arange(total_samples, dtype=np.int64)
    source_frame_idx = (
        np.asarray(crop_frame_indices[:], dtype=np.int64)
        if isinstance(crop_frame_indices, zarr.Array)
        else local_frame_indices.copy()
    )
    if int(source_frame_idx.shape[0]) != total_samples:
        raise ValueError(
            f"crop_runs/{selection.crop_run}/frame_indices has {source_frame_idx.shape[0]} rows, expected {total_samples}."
        )

    detection_source = (
        np.asarray(crop_detection_source[:], dtype=np.int8)
        if isinstance(crop_detection_source, zarr.Array)
        else np.zeros((total_samples,), dtype=np.int8)
    )
    if int(detection_source.shape[0]) != total_samples:
        raise ValueError(
            f"crop_runs/{selection.crop_run}/detection_source has {detection_source.shape[0]} rows, expected {total_samples}."
        )

    if eye_separation is None:
        eye_separation_data = np.full((total_samples,), np.nan, dtype=np.float32)
    else:
        eye_separation_data = np.asarray(eye_separation[:], dtype=np.float32)
    if int(eye_separation_data.shape[0]) != total_samples:
        raise ValueError(
            f"{selection.eye_stage}/{selection.eye_run}/eye_separation has {eye_separation_data.shape[0]} rows, expected {total_samples}."
        )

    if reason_src is not None:
        reason_values = np.asarray(reason_src[:], dtype=object)
        if int(reason_values.shape[0]) != total_samples:
            raise ValueError(
                f"Reason array in {selection.eye_stage}/{selection.eye_run} has {reason_values.shape[0]} rows, expected {total_samples}."
            )
        reason_values = np.asarray([str(v) if v is not None else "" for v in reason_values], dtype=object)
    else:
        reason_values = np.asarray([""] * total_samples, dtype=object)

    dst_root = zarr.open_group(str(out_path), mode="w")
    training_export_payload = {
        "tool": "fisheye.utils.export_eye_mask_training_zarr",
        "created_at_utc": _utc_now(),
        "input_format": normalized_input_format,
        "label_mode": normalized_label_mode,
        "source_stage": selection.eye_stage,
        "source_eye_run": selection.eye_run,
        "source_crop_run": selection.crop_run,
        "split_seed": int(split_seed),
    }
    dst_root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_task": "eye_masks",
            "training_export": training_export_payload,
        }
    )

    dst_crop_parent = dst_root.create_group("crop_runs")
    dst_crop_parent.attrs["latest"] = run_name
    dst_crop = dst_crop_parent.create_group(run_name)
    _copy_array(roi_images, dst_crop, "roi_images")
    _copy_array(bbox_norm, dst_crop, "bbox_norm_coords")
    if isinstance(crop_bbox, zarr.Array):
        _copy_array(crop_bbox, dst_crop, "crop_bbox_norm_coords")
    else:
        bbox_copy = np.asarray(bbox_norm[:], dtype=np.float32)
        dst_crop.create_array(
            "crop_bbox_norm_coords",
            data=bbox_copy,
            chunks=getattr(bbox_norm, "chunks", None),
            overwrite=True,
        )
    dst_crop.create_array(
        "frame_indices",
        data=local_frame_indices,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_crop.create_array(
        "detection_source",
        data=detection_source,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_crop.attrs.update(
        {
            "source_crop_run": selection.crop_run,
            "source_zarr_path": str(source_path),
        }
    )

    dst_eye_parent = dst_root.create_group("eye_masks_runs")
    dst_eye_parent.attrs["latest"] = run_name
    dst_eye = dst_eye_parent.create_group(run_name)
    _copy_array(masks_roi, dst_eye, "masks_roi")
    _copy_array(ellipse_params, dst_eye, "ellipse_params")
    _copy_array(ellipse_success, dst_eye, "ellipse_success")
    dst_eye.create_array(
        "eye_separation",
        data=eye_separation_data,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_eye.create_array(
        "frame_indices",
        data=local_frame_indices,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_eye.create_array(
        "detection_source",
        data=detection_source,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    reason_dst = dst_eye.create_array(
        "reason",
        shape=(total_samples,),
        dtype=VariableLengthUTF8(),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    reason_dst[:] = reason_values
    if isinstance(mask_probs_src, zarr.Array) and mask_probs_name:
        _copy_array(mask_probs_src, dst_eye, mask_probs_name)

    for attr_name in (
        "method",
        "eye_labels",
        "min_eye_separation",
        "max_eye_separation",
        "config",
        "source_keypoints_run",
        "source_keypoint_run",
        "source_keypoint_group",
        "source_eye_masks_run",
        "source_eye_masks_method",
        "reason_counts",
    ):
        if attr_name in eye_group.attrs:
            dst_eye.attrs[attr_name] = eye_group.attrs[attr_name]
    dst_eye.attrs.update(
        {
            "source_eye_stage": selection.eye_stage,
            "source_eye_run": selection.eye_run,
            "source_crop_run": selection.crop_run,
            "source_zarr_path": str(source_path),
            "label_mode": normalized_label_mode,
        }
    )

    train_idx, val_idx, test_idx = _make_split_indices(
        total_samples,
        train_ratio=float(split_train),
        val_ratio=float(split_val),
        test_ratio=float(split_test),
        seed=int(split_seed),
    )
    split_group = dst_root.create_group("splits")
    split_group.create_array("train_indices", data=train_idx.astype(np.int64, copy=False), chunks=(max(1, min(train_idx.size or 1, 65536)),))
    split_group.create_array("val_indices", data=val_idx.astype(np.int64, copy=False), chunks=(max(1, min(val_idx.size or 1, 65536)),))
    split_group.create_array("test_indices", data=test_idx.astype(np.int64, copy=False), chunks=(max(1, min(test_idx.size or 1, 65536)),))
    split_group.attrs.update(
        {
            "split_seed": int(split_seed),
            "split_ratios": {
                "train": float(split_train),
                "val": float(split_val),
                "test": float(split_test),
            },
        }
    )

    source_index = dst_root.create_group("source_index")
    source_index.create_array(
        "source_dataset_idx",
        data=np.zeros((total_samples,), dtype=np.int32),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    source_index.create_array(
        "source_frame_idx",
        data=source_frame_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    source_index.create_array(
        "source_roi_idx",
        data=np.arange(total_samples, dtype=np.int64),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dataset_id = source_path.stem
    _write_string_array(source_index, "source_dataset_id", [dataset_id])
    _write_string_array(source_index, "source_zarr_path", [str(source_path)])
    source_index.attrs.update(
        {
            "mapping_version": 1,
            "source_count": 1,
        }
    )

    summary: Dict[str, Any]
    if validate:
        summary = validate_merged_eye_mask_training_zarr(
            out_path,
            expected_input_format=normalized_input_format,
            expected_total_samples=total_samples,
            expected_label_mode=normalized_label_mode,
        )
    else:
        summary = {
            "zarr_path": str(out_path),
            "run_name": str(run_name),
            "total_samples": int(total_samples),
            "split_counts": {
                "train": int(train_idx.shape[0]),
                "val": int(val_idx.shape[0]),
                "test": int(test_idx.shape[0]),
            },
        }

    summary.update(
        {
            "source_zarr": str(source_path),
            "source_eye_stage": selection.eye_stage,
            "source_eye_run": selection.eye_run,
            "source_crop_run": selection.crop_run,
        }
    )
    return summary


def validate_merged_eye_mask_training_zarr(
    zarr_path: Path,
    *,
    expected_input_format: Optional[str] = None,
    expected_total_samples: Optional[int] = None,
    expected_label_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate merged eye-mask-training Zarr layout and trainer-facing invariants."""
    root = zarr.open_group(str(zarr_path), mode="r")
    errors: List[str] = []

    if str(root.attrs.get("zarr_purpose", "")).strip().lower() != "training":
        errors.append("root attr zarr_purpose must be 'training'.")
    training_task = str(root.attrs.get("training_task", "")).strip().lower()
    if training_task and training_task != "eye_masks":
        errors.append("root attr training_task must be 'eye_masks' when present.")

    for group_name in ("crop_runs", "eye_masks_runs", "splits", "source_index"):
        if group_name not in root:
            errors.append(f"missing group {group_name}.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop_parent = root["crop_runs"]
    eye_parent = root["eye_masks_runs"]
    crop_latest = _as_text(crop_parent.attrs.get("latest"))
    eye_latest = _as_text(eye_parent.attrs.get("latest"))
    if not crop_latest or crop_latest not in crop_parent:
        errors.append("crop_runs/latest missing or points to a non-existent run.")
    if not eye_latest or eye_latest not in eye_parent:
        errors.append("eye_masks_runs/latest missing or points to a non-existent run.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop = crop_parent[str(crop_latest)]
    eye = eye_parent[str(eye_latest)]

    required_crop_arrays = (
        "roi_images",
        "bbox_norm_coords",
        "crop_bbox_norm_coords",
        "frame_indices",
        "detection_source",
    )
    for name in required_crop_arrays:
        if name not in crop:
            errors.append(f"missing required array crop_runs/{crop_latest}/{name}.")

    required_eye_arrays = (
        "masks_roi",
        "ellipse_params",
        "ellipse_success",
        "eye_separation",
        "frame_indices",
        "detection_source",
    )
    for name in required_eye_arrays:
        if name not in eye:
            errors.append(f"missing required array eye_masks_runs/{eye_latest}/{name}.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    roi_images = np.asarray(crop["roi_images"][:])
    bbox_norm = np.asarray(crop["bbox_norm_coords"][:])
    crop_bbox = np.asarray(crop["crop_bbox_norm_coords"][:])
    crop_frame_indices = np.asarray(crop["frame_indices"][:])
    crop_detection_source = np.asarray(crop["detection_source"][:])

    masks_roi = np.asarray(eye["masks_roi"][:])
    ellipse_params = np.asarray(eye["ellipse_params"][:], dtype=np.float32)
    ellipse_success = np.asarray(eye["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(eye["eye_separation"][:], dtype=np.float32)
    eye_frame_indices = np.asarray(eye["frame_indices"][:])
    eye_detection_source = np.asarray(eye["detection_source"][:])

    if roi_images.ndim < 3:
        errors.append(f"roi_images must have shape (N,H,W) or (N,H,W,C), got {tuple(roi_images.shape)}.")
    total_samples = int(roi_images.shape[0]) if roi_images.ndim >= 1 else 0
    if expected_total_samples is not None and int(expected_total_samples) != total_samples:
        errors.append(f"total sample mismatch ({total_samples} != expected {int(expected_total_samples)}).")

    input_format = _normalize_input_format(expected_input_format)
    if input_format is None:
        export_meta = root.attrs.get("training_export")
        if isinstance(export_meta, dict):
            input_format = _normalize_input_format(export_meta.get("input_format"))
    if input_format == "rgb":
        if roi_images.ndim != 4 or int(roi_images.shape[-1]) != 3:
            errors.append("roi_images must be (N,H,W,3) for rgb input format.")
    if input_format == "gray":
        if roi_images.ndim == 4 and int(roi_images.shape[-1]) == 3:
            errors.append("roi_images appears rgb but expected gray input format.")

    label_mode = _normalize_label_mode(expected_label_mode)
    if label_mode is None:
        export_meta = root.attrs.get("training_export")
        if isinstance(export_meta, dict):
            label_mode = _normalize_label_mode(export_meta.get("label_mode"))

    if bbox_norm.ndim != 2 or int(bbox_norm.shape[1]) != 4:
        errors.append(f"bbox_norm_coords must have shape (N,4), got {tuple(bbox_norm.shape)}.")
    if crop_bbox.ndim != 2 or int(crop_bbox.shape[1]) != 4:
        errors.append(f"crop_bbox_norm_coords must have shape (N,4), got {tuple(crop_bbox.shape)}.")
    if bbox_norm.ndim == 2 and int(bbox_norm.shape[0]) != total_samples:
        errors.append(f"bbox_norm_coords length mismatch ({bbox_norm.shape[0]} != {total_samples}).")
    if crop_bbox.ndim == 2 and int(crop_bbox.shape[0]) != total_samples:
        errors.append(f"crop_bbox_norm_coords length mismatch ({crop_bbox.shape[0]} != {total_samples}).")

    if masks_roi.ndim != 4:
        errors.append(f"masks_roi must have shape (N,C,H,W), got {tuple(masks_roi.shape)}.")
    else:
        if int(masks_roi.shape[0]) != total_samples:
            errors.append(f"masks_roi length mismatch ({masks_roi.shape[0]} != {total_samples}).")
        channels = int(masks_roi.shape[1])
        if channels < 1:
            errors.append("masks_roi must have at least 1 channel.")
        if label_mode == "lr" and channels != 2:
            errors.append(f"label_mode=lr requires masks_roi channel count 2, got {channels}.")
        if label_mode == "union" and channels != 1:
            errors.append(f"label_mode=union requires masks_roi channel count 1, got {channels}.")
        if roi_images.ndim >= 3:
            roi_h = int(roi_images.shape[1])
            roi_w = int(roi_images.shape[2])
            if int(masks_roi.shape[2]) != roi_h or int(masks_roi.shape[3]) != roi_w:
                errors.append(
                    "masks_roi spatial dims do not match roi_images "
                    f"(({masks_roi.shape[2]}, {masks_roi.shape[3]}) != ({roi_h}, {roi_w}))."
                )
        unique_vals = np.unique(masks_roi.astype(np.int8, copy=False))
        invalid_vals = [int(v) for v in unique_vals.tolist() if int(v) not in (0, 1)]
        if invalid_vals:
            errors.append(f"masks_roi contains non-binary values: {sorted(set(invalid_vals))}.")

    if ellipse_params.ndim != 3 or int(ellipse_params.shape[-1]) != 5:
        errors.append(f"ellipse_params must have shape (N,C,5), got {tuple(ellipse_params.shape)}.")
    if ellipse_success.ndim != 2:
        errors.append(f"ellipse_success must have shape (N,C), got {tuple(ellipse_success.shape)}.")
    if ellipse_params.ndim == 3 and int(ellipse_params.shape[0]) != total_samples:
        errors.append(f"ellipse_params length mismatch ({ellipse_params.shape[0]} != {total_samples}).")
    if ellipse_success.ndim == 2 and int(ellipse_success.shape[0]) != total_samples:
        errors.append(f"ellipse_success length mismatch ({ellipse_success.shape[0]} != {total_samples}).")
    if (
        ellipse_params.ndim == 3
        and ellipse_success.ndim == 2
        and tuple(ellipse_params.shape[:2]) != tuple(ellipse_success.shape[:2])
    ):
        errors.append(
            "ellipse_params and ellipse_success channel shapes differ "
            f"({tuple(ellipse_params.shape[:2])} != {tuple(ellipse_success.shape[:2])})."
        )

    if eye_separation.ndim != 1 or int(eye_separation.shape[0]) != total_samples:
        errors.append(f"eye_separation must be 1D length N ({total_samples}), got {tuple(eye_separation.shape)}.")

    for name, arr in (("crop frame_indices", crop_frame_indices), ("eye frame_indices", eye_frame_indices)):
        if arr.ndim != 1:
            errors.append(f"{name} must be 1D, got ndim={arr.ndim}.")
        elif int(arr.shape[0]) != total_samples:
            errors.append(f"{name} length mismatch ({arr.shape[0]} != {total_samples}).")
        elif not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{name} must be integer dtype, got {arr.dtype}.")
        else:
            expected_local = np.arange(total_samples, dtype=np.int64)
            if not np.array_equal(arr.astype(np.int64, copy=False), expected_local):
                errors.append(f"{name} must be local 0..N-1 indexing.")

    for name, arr in (("crop detection_source", crop_detection_source), ("eye detection_source", eye_detection_source)):
        if arr.ndim != 1:
            errors.append(f"{name} must be 1D, got ndim={arr.ndim}.")
        elif int(arr.shape[0]) != total_samples:
            errors.append(f"{name} length mismatch ({arr.shape[0]} != {total_samples}).")
        elif not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{name} must be integer dtype, got {arr.dtype}.")
        else:
            unique_codes = np.unique(arr.astype(np.int64, copy=False))
            invalid_codes = [int(code) for code in unique_codes.tolist() if int(code) not in (0, 1)]
            if invalid_codes:
                errors.append(f"{name} contains invalid codes: {sorted(set(invalid_codes))} (expected 0 or 1).")

    if crop_detection_source.ndim == 1 and eye_detection_source.ndim == 1:
        if crop_detection_source.shape == eye_detection_source.shape:
            if not np.array_equal(
                crop_detection_source.astype(np.int64, copy=False),
                eye_detection_source.astype(np.int64, copy=False),
            ):
                errors.append("eye detection_source must match crop detection_source.")

    if "reason" in eye:
        reason_arr = np.asarray(eye["reason"][:], dtype=object)
        if reason_arr.ndim != 1 or int(reason_arr.shape[0]) != total_samples:
            errors.append(f"eye reason must be 1D length N ({total_samples}), got {tuple(reason_arr.shape)}.")

    probs_name_present = None
    for probs_name in ("mask_probs_roi_refined", "mask_probs_roi"):
        if probs_name in eye:
            probs_name_present = probs_name
            probs = np.asarray(eye[probs_name][:], dtype=np.float32)
            if probs.shape != masks_roi.shape:
                errors.append(
                    f"{probs_name} shape must match masks_roi "
                    f"({tuple(probs.shape)} != {tuple(masks_roi.shape)})."
                )
            elif not np.all(np.isfinite(probs)):
                errors.append(f"{probs_name} contains non-finite values.")
            else:
                min_val = float(np.min(probs)) if probs.size else 0.0
                max_val = float(np.max(probs)) if probs.size else 0.0
                if min_val < -1e-6 or max_val > 1.0 + 1e-6:
                    errors.append(f"{probs_name} values must be in [0,1], got min={min_val:.4f}, max={max_val:.4f}.")
            break

    if ellipse_params.ndim == 3 and ellipse_success.ndim == 2 and tuple(ellipse_params.shape[:2]) == tuple(ellipse_success.shape[:2]):
        success_mask = ellipse_success.astype(bool, copy=False)
        major = ellipse_params[:, :, 2]
        minor = ellipse_params[:, :, 3]
        success_major = major[success_mask]
        success_minor = minor[success_mask]
        if success_major.size > 0:
            if not np.all(np.isfinite(success_major)) or not np.all(np.isfinite(success_minor)):
                errors.append("Successful ellipse rows contain non-finite major/minor axes.")
            if np.any(success_major <= 0.0) or np.any(success_minor <= 0.0):
                errors.append("Successful ellipse rows must have positive major/minor axes.")
            if np.any(success_major < success_minor):
                errors.append("Successful ellipse rows must satisfy major >= minor.")

    split_arrays: Dict[str, np.ndarray] = {}
    for name in ("train_indices", "val_indices", "test_indices"):
        path = f"splits/{name}"
        if name == "test_indices" and path not in root:
            split_arrays[name] = np.empty(0, dtype=np.int64)
            continue
        if path not in root:
            errors.append(f"missing required array {path}.")
            continue
        arr = np.asarray(root[path][:])
        if arr.ndim != 1:
            errors.append(f"{path} must be 1D, got ndim={arr.ndim}.")
            continue
        if not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{path} must be integer dtype, got {arr.dtype}.")
            continue
        arr_i64 = arr.astype(np.int64, copy=False)
        if arr_i64.size > 0:
            min_idx = int(arr_i64.min())
            max_idx = int(arr_i64.max())
            if min_idx < 0 or max_idx >= total_samples:
                errors.append(
                    f"{path} indices out of bounds (min={min_idx}, max={max_idx}, total_samples={total_samples})."
                )
            if np.unique(arr_i64).size != arr_i64.size:
                errors.append(f"{path} contains duplicate indices.")
        split_arrays[name] = arr_i64

    train_idx = split_arrays.get("train_indices", np.empty(0, dtype=np.int64))
    val_idx = split_arrays.get("val_indices", np.empty(0, dtype=np.int64))
    test_idx = split_arrays.get("test_indices", np.empty(0, dtype=np.int64))
    if np.intersect1d(train_idx, val_idx).size > 0:
        errors.append("splits/train_indices overlaps with splits/val_indices.")
    if np.intersect1d(train_idx, test_idx).size > 0:
        errors.append("splits/train_indices overlaps with splits/test_indices.")
    if np.intersect1d(val_idx, test_idx).size > 0:
        errors.append("splits/val_indices overlaps with splits/test_indices.")
    combined = np.concatenate([train_idx, val_idx, test_idx]) if total_samples > 0 else np.empty(0, dtype=np.int64)
    if total_samples > 0:
        if combined.size != total_samples:
            errors.append(
                "split coverage mismatch "
                f"(train+val+test={combined.size} but total_samples={total_samples})."
            )
        elif np.unique(combined).size != total_samples:
            errors.append("split coverage must be exact and non-duplicated across split arrays.")

    src_dataset_idx_path = "source_index/source_dataset_idx"
    src_frame_idx_path = "source_index/source_frame_idx"
    src_dataset_id_path = "source_index/source_dataset_id"
    src_zarr_path_path = "source_index/source_zarr_path"
    for path in (src_dataset_idx_path, src_frame_idx_path, src_dataset_id_path, src_zarr_path_path):
        if path not in root:
            errors.append(f"missing required array {path}.")

    source_count = 0
    if not errors:
        source_dataset_idx = np.asarray(root[src_dataset_idx_path][:])
        source_frame_idx = np.asarray(root[src_frame_idx_path][:])
        source_dataset_id = np.asarray(root[src_dataset_id_path][:])
        source_zarr_path = np.asarray(root[src_zarr_path_path][:])
        source_roi_idx = np.asarray(root["source_index/source_roi_idx"][:]) if "source_index/source_roi_idx" in root else None

        if source_dataset_idx.ndim != 1 or source_dataset_idx.shape[0] != total_samples:
            errors.append(
                f"{src_dataset_idx_path} must be 1D length N ({total_samples}), got {source_dataset_idx.shape}."
            )
        if source_frame_idx.ndim != 1 or source_frame_idx.shape[0] != total_samples:
            errors.append(
                f"{src_frame_idx_path} must be 1D length N ({total_samples}), got {source_frame_idx.shape}."
            )
        if source_dataset_id.ndim != 1 or source_zarr_path.ndim != 1:
            errors.append("source_index/source_dataset_id and source_index/source_zarr_path must be 1D arrays.")
        elif source_dataset_id.shape[0] != source_zarr_path.shape[0]:
            errors.append(
                "source_index/source_dataset_id and source_index/source_zarr_path length mismatch "
                f"({source_dataset_id.shape[0]} != {source_zarr_path.shape[0]})."
            )
        elif total_samples > 0 and source_dataset_id.shape[0] == 0:
            errors.append("source index mapping arrays are empty but dataset has samples.")
        source_count = int(source_dataset_id.shape[0]) if source_dataset_id.ndim == 1 else 0

        if source_dataset_idx.ndim == 1 and np.issubdtype(source_dataset_idx.dtype, np.integer):
            source_dataset_idx_i64 = source_dataset_idx.astype(np.int64, copy=False)
            if source_dataset_idx_i64.size > 0 and int(source_dataset_idx_i64.min()) < 0:
                errors.append(f"{src_dataset_idx_path} contains negative indices.")
            if source_count > 0 and source_dataset_idx_i64.size > 0:
                max_idx = int(source_dataset_idx_i64.max())
                if max_idx >= source_count:
                    errors.append(
                        f"{src_dataset_idx_path} has value {max_idx} outside mapping length {source_count}."
                    )
        else:
            errors.append(f"{src_dataset_idx_path} must be integer dtype.")

        if source_frame_idx.ndim == 1 and np.issubdtype(source_frame_idx.dtype, np.integer):
            source_frame_idx_i64 = source_frame_idx.astype(np.int64, copy=False)
            if source_frame_idx_i64.size > 0 and int(source_frame_idx_i64.min()) < 0:
                errors.append(f"{src_frame_idx_path} contains negative indices.")
        else:
            errors.append(f"{src_frame_idx_path} must be integer dtype.")

        if source_roi_idx is not None:
            if source_roi_idx.ndim != 1 or source_roi_idx.shape[0] != total_samples:
                errors.append(
                    f"source_index/source_roi_idx must be 1D length N ({total_samples}), got {source_roi_idx.shape}."
                )
            elif not np.issubdtype(source_roi_idx.dtype, np.integer):
                errors.append(f"source_index/source_roi_idx must be integer dtype, got {source_roi_idx.dtype}.")
            else:
                source_roi_idx_i64 = source_roi_idx.astype(np.int64, copy=False)
                if source_roi_idx_i64.size > 0:
                    min_idx = int(source_roi_idx_i64.min())
                    max_idx = int(source_roi_idx_i64.max())
                    if min_idx < 0 or max_idx >= total_samples:
                        errors.append(
                            "source_index/source_roi_idx out of bounds "
                            f"(min={min_idx}, max={max_idx}, total_samples={total_samples})."
                        )

    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    success_eyes = int(ellipse_success.sum())
    successful_roi_pairs = int(np.all(ellipse_success, axis=1).sum()) if ellipse_success.ndim == 2 else 0
    return {
        "zarr_path": str(zarr_path),
        "crop_run": str(crop_latest),
        "eye_run": str(eye_latest),
        "input_format": input_format,
        "label_mode": label_mode,
        "total_samples": int(total_samples),
        "channels": int(masks_roi.shape[1]),
        "success_eyes": success_eyes,
        "successful_roi_pairs": successful_roi_pairs,
        "split_counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "source_count": int(source_count),
        "mask_probs_array": probs_name_present,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path, help="Source training/analysis zarr path.")
    parser.add_argument("out_zarr", type=Path, help="Output merged eye-mask-training zarr path.")
    parser.add_argument("--crop-run", help="Optional crop run override.")
    parser.add_argument(
        "--eye-stage",
        choices=["auto", "eye_masks_runs", "refined_eye_masks_runs"],
        default="auto",
        help="Eye-mask stage selector (default: auto prefers refined).",
    )
    parser.add_argument("--eye-run", help="Optional explicit eye-mask run name.")
    parser.add_argument("--run-name", default="merged_export_smoke", help="Merged run name inside output zarr.")
    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--label-mode", choices=["lr", "union"], default="lr")
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing out_zarr.")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-export validation.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = export_merged_eye_mask_training_zarr(
        source_zarr=args.source_zarr,
        out_zarr=args.out_zarr,
        crop_run=args.crop_run,
        eye_stage=args.eye_stage,
        eye_run=args.eye_run,
        run_name=args.run_name,
        input_format=args.input_format,
        label_mode=args.label_mode,
        split_train=float(args.split_train),
        split_val=float(args.split_val),
        split_test=float(args.split_test),
        split_seed=int(args.split_seed),
        overwrite=bool(args.overwrite),
        validate=not bool(args.no_validate),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
