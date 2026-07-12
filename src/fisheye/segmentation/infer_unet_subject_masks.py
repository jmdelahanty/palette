"""Run a trained U-Net segmenter to produce unified subject-mask probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ..pose.schema import resolve_required_keypoint_indices_from_attrs
from ..shared.crop_image_source import CropImageSource
from ..shared.artifact_fingerprint import fingerprint_artifact
from ..shared.inference_timing import InferenceTimingProfiler
from ..shared.model_input_transform import MODEL_INPUT_TRANSFORM_CHOICES, ModelInputTransform, resolve_model_input_transform
from ..shared.provenance_attrs import (
    build_assignment_keypoint_attrs,
    build_source_crop_snapshot_attrs,
    build_source_roi_pixel_attrs,
)
from ..shared.row_alignment import assert_row_alignment
from ..shared.row_lineage import copy_row_lineage_arrays, write_direct_source_crop_row_ids
from ..shared.run_provenance import append_input_artifacts, build_run_provenance_from_stage_record
from ..shared.subject_mask_registry_status import emit_subject_mask_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import subject_mask_metric_row_chunk, subject_mask_storage_chunks
from ..shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..registry.db import Registry, RegistryPaths
from ..registry.model_resolution import (
    build_resolution_payload,
    resolve_best_subject_mask_model,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from .unet import UNetSmall

SUBJECT_MASK_SCHEMAS: dict[str, tuple[str, ...]] = {
    "subject_v1_union": ("subject_body", "eyes_union", "swim_bladder"),
    "subject_v1_lr": ("subject_body", "eye_left", "eye_right", "swim_bladder"),
}
SUBJECT_MASK_LABEL_SCHEMA = "subject_v1_union"
SUBJECT_MASK_LABELS: tuple[str, ...] = SUBJECT_MASK_SCHEMAS[SUBJECT_MASK_LABEL_SCHEMA]
_SUBJECT_MASKS_STATUS_SOURCE = "runtime_infer_unet_subject_masks"
KEYPOINT_GROUP_CHOICES = ("refined_keypoints_runs", "keypoints_runs")
KEYPOINT_SUCCESS_DATASET_CANDIDATES = (
    "usable_keypoints",
    "detection_success",
    "refined_success",
    "source_success",
)
EYE_KEYPOINT_LABELS = ("eye_left", "eye_right")
SUBJECT_MASK_OUTPUT_PARENTS = ("subject_mask_runs", "subject_mask_shard_runs")
SUBJECT_MASK_CANONICAL_OUTPUT_PARENT = "subject_mask_runs"
SUBJECT_MASK_SHARD_OUTPUT_PARENT = "subject_mask_shard_runs"
MASK_PROBS_WORKING_ARRAY = "_mask_probs_roi_working"
MASK_PROBS_CANONICAL_ARRAY = "mask_probs_roi"
MASK_PROBS_SHARDING_SCHEMA = "palette.subject_mask_probability_postpack.v1"


@dataclass(frozen=True)
class _SubjectMaskOutputBatch:
    start: int
    stop: int
    probs_out: np.ndarray
    binary: Optional[np.ndarray]
    metrics: dict[str, np.ndarray]


def _compression_kwargs(array: zarr.Array) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    sentinel = object()
    compressors = getattr(array, "compressors", sentinel)
    if compressors is not sentinel:
        if compressors:
            kwargs["compressors"] = compressors
    else:
        try:
            compressor = array.compressor
        except (TypeError, AttributeError):
            compressor = None
        if compressor is not None:
            kwargs["compressor"] = compressor

    chunk_codecs = getattr(array, "chunk_codecs", None)
    if chunk_codecs:
        kwargs.setdefault("chunk_codecs", chunk_codecs)
    filters = getattr(array, "filters", None)
    if filters:
        kwargs.setdefault("filters", filters)
    return kwargs


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_str)
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _normalise_roi_tensor(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)
    if batch.ndim != 4:
        raise ValueError(f"Unexpected ROI batch shape {tuple(batch.shape)}")
    if batch.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported ROI channel count: {int(batch.shape[1])}")

    if not batch.is_floating_point():
        max_value = float(torch.iinfo(batch.dtype).max)
        batch = batch.to(dtype=torch.float32)
        if max_value > 0:
            batch = batch / max_value
    else:
        batch = batch.to(dtype=torch.float32)
        if batch.numel():
            max_val = torch.nan_to_num(batch, nan=0.0, posinf=float("inf"), neginf=float("-inf")).amax()
            batch = batch / torch.maximum(max_val, torch.tensor(1.0, device=batch.device, dtype=torch.float32))

    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
    batch = torch.clamp(batch, 0.0, 1.0)
    if batch.device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch


def _probabilities_from_logits(logits: torch.Tensor, *, mask_probs_dtype: str) -> np.ndarray:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    if mask_probs_dtype == "uint8":
        probs = torch.round(probs * 255.0).to(dtype=torch.uint8)
    else:
        probs = probs.to(dtype=torch.float16)
    return probs.cpu().numpy()


def _postprocess_logits_on_device(
    logits: torch.Tensor,
    *,
    mask_probs_dtype: str,
    return_binary: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, np.ndarray]]:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)

    if mask_probs_dtype == "uint8":
        probs_out = torch.round(probs * 255.0).to(dtype=torch.uint8)
        binary = (probs_out >= 128).to(dtype=torch.uint8)
        probs_for_metrics = probs_out.to(dtype=torch.float32) / 255.0
    else:
        probs_out = probs.to(dtype=torch.float16)
        binary = (probs_out >= 0.5).to(dtype=torch.uint8)
        probs_for_metrics = probs_out.to(dtype=torch.float32)

    area_px = binary.sum(dim=(2, 3), dtype=torch.float32)
    prob_max = probs_for_metrics.amax(dim=(2, 3))
    spatial_metrics = _compute_spatial_metrics_from_binary_tensor(binary, area_px=area_px)
    metrics = {
        "prob_max": prob_max.cpu().numpy().astype(np.float32, copy=False),
        "mask_present": (area_px > 0.0).cpu().numpy().astype(bool, copy=False),
        "area_px": area_px.cpu().numpy().astype(np.float32, copy=False),
        "centroid_xy": spatial_metrics["centroid_xy"].cpu().numpy().astype(np.float32, copy=False),
        "centroid_valid": spatial_metrics["centroid_valid"].cpu().numpy().astype(bool, copy=False),
        "bbox_xyxy": spatial_metrics["bbox_xyxy"].cpu().numpy().astype(np.float32, copy=False),
        "bbox_valid": spatial_metrics["bbox_valid"].cpu().numpy().astype(bool, copy=False),
    }
    binary_out = binary.cpu().numpy() if return_binary else None
    return probs_out.cpu().numpy(), binary_out, metrics


def _compute_spatial_metrics_from_binary_tensor(
    binary: torch.Tensor,
    *,
    area_px: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    if binary.ndim != 4:
        raise ValueError("binary must have shape (N,C,H,W).")

    mask = binary > 0
    _row_count, _channel_count, height, width = mask.shape
    if area_px is None:
        area_px = mask.sum(dim=(2, 3), dtype=torch.float32)
    else:
        area_px = area_px.to(dtype=torch.float32)
    valid = area_px > 0.0
    denominator = torch.clamp(area_px, min=1.0)

    y_counts = mask.sum(dim=3, dtype=torch.float32)
    x_counts = mask.sum(dim=2, dtype=torch.float32)
    y_float = torch.arange(height, device=mask.device, dtype=torch.float32).view(1, 1, height)
    x_float = torch.arange(width, device=mask.device, dtype=torch.float32).view(1, 1, width)
    y_sum = (y_counts * y_float).sum(dim=2)
    x_sum = (x_counts * x_float).sum(dim=2)

    centroid_xy = torch.stack((x_sum / denominator, y_sum / denominator), dim=2)
    centroid_xy = torch.where(valid.unsqueeze(2), centroid_xy, torch.zeros_like(centroid_xy))

    row_has_mask = mask.any(dim=3)
    col_has_mask = mask.any(dim=2)
    y_int = torch.arange(height, device=mask.device, dtype=torch.int32).view(1, 1, height)
    x_int = torch.arange(width, device=mask.device, dtype=torch.int32).view(1, 1, width)
    y_min = torch.where(
        row_has_mask,
        y_int,
        torch.full((1, 1, height), height, device=mask.device, dtype=torch.int32),
    ).amin(dim=2)
    y_max = torch.where(
        row_has_mask,
        y_int,
        torch.full((1, 1, height), -1, device=mask.device, dtype=torch.int32),
    ).amax(dim=2)
    x_min = torch.where(
        col_has_mask,
        x_int,
        torch.full((1, 1, width), width, device=mask.device, dtype=torch.int32),
    ).amin(dim=2)
    x_max = torch.where(
        col_has_mask,
        x_int,
        torch.full((1, 1, width), -1, device=mask.device, dtype=torch.int32),
    ).amax(dim=2)

    bbox_xyxy = torch.stack((x_min, y_min, x_max, y_max), dim=2).to(dtype=torch.float32)
    bbox_xyxy = torch.where(valid.unsqueeze(2), bbox_xyxy, torch.zeros_like(bbox_xyxy))
    return {
        "centroid_xy": centroid_xy,
        "centroid_valid": valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": valid,
    }


def _compute_channel_metrics(binary_masks: np.ndarray, probs: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    prob_arr = np.asarray(probs, dtype=np.float32)
    if binary.ndim != 3 or prob_arr.shape != binary.shape:
        raise ValueError("binary_masks and probs must both have shape (N,H,W).")

    row_count = int(binary.shape[0])
    area_px = binary.sum(axis=(1, 2), dtype=np.int64).astype(np.float32)
    mask_present = area_px > 0.0
    prob_max = prob_arr.max(axis=(1, 2)).astype(np.float32, copy=False) if row_count else np.zeros((0,), dtype=np.float32)
    spatial_metrics = _compute_channel_spatial_metrics(binary)

    return {
        "prob_max": prob_max,
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        **spatial_metrics,
    }


def _compute_channel_spatial_metrics(binary_masks: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError("binary_masks must have shape (N,H,W).")

    row_count = int(binary.shape[0])

    centroid_xy = np.zeros((row_count, 2), dtype=np.float32)
    centroid_valid = np.zeros((row_count,), dtype=bool)
    bbox_xyxy = np.zeros((row_count, 4), dtype=np.float32)
    bbox_valid = np.zeros((row_count,), dtype=bool)

    for row_idx in range(row_count):
        ys, xs = np.nonzero(binary[row_idx] > 0)
        if ys.size == 0:
            continue
        centroid_xy[row_idx, 0] = float(xs.mean())
        centroid_xy[row_idx, 1] = float(ys.mean())
        centroid_valid[row_idx] = True
        bbox_xyxy[row_idx] = np.asarray(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32,
        )
        bbox_valid[row_idx] = True

    return {
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[UNetSmall, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint.get("model_config")
    if not model_cfg:
        raise ValueError("Checkpoint missing 'model_config'; retrain with updated trainer.")
    model = UNetSmall(**model_cfg)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model_state'.")
    state_dict = _normalize_checkpoint_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def _normalize_checkpoint_state_dict(state_dict: object) -> object:
    """Normalize checkpoint keys saved from torch.compile-wrapped modules."""

    if not isinstance(state_dict, dict):
        return state_dict
    prefix = "_orig_mod."
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    normalized: dict[object, object] = {}
    for key, value in state_dict.items():
        key_text = str(key)
        normalized_key = key_text[len(prefix) :] if key_text.startswith(prefix) else key
        if normalized_key in normalized:
            raise ValueError(f"Checkpoint state_dict key collision after removing {prefix!r}.")
        normalized[normalized_key] = value
    return normalized


def _resolve_checkpoint_schema(checkpoint: Dict[str, object]) -> Tuple[str, Tuple[str, ...]]:
    label_schema_id = str(checkpoint.get("label_schema_id") or "").strip() or SUBJECT_MASK_LABEL_SCHEMA
    expected_labels = SUBJECT_MASK_SCHEMAS.get(label_schema_id)
    if expected_labels is None:
        raise ValueError(
            f"Unsupported subject-mask checkpoint label_schema_id {label_schema_id!r}. "
            f"Expected one of {sorted(SUBJECT_MASK_SCHEMAS)}."
        )

    mask_labels_raw = checkpoint.get("mask_labels")
    if isinstance(mask_labels_raw, (list, tuple)) and mask_labels_raw:
        mask_labels = tuple(str(item) for item in mask_labels_raw)
    else:
        mask_labels = expected_labels
    if mask_labels != expected_labels:
        raise ValueError(
            f"Checkpoint mask_labels {mask_labels!r} do not match schema "
            f"{label_schema_id!r}: {expected_labels!r}."
        )
    return label_schema_id, mask_labels


def _resolve_assignment_keypoint_attrs(
    root: zarr.Group,
    *,
    assignment_keypoint_group: Optional[str],
    assignment_keypoints_run: Optional[str],
    total_rois: int,
    mask_labels: Sequence[str],
) -> dict[str, object]:
    has_group = bool(assignment_keypoint_group)
    has_run = bool(assignment_keypoints_run)
    if has_group != has_run:
        raise ValueError(
            "Pass both --assignment-keypoint-group and --assignment-keypoint-run, or neither."
        )
    if not has_group or not has_run:
        return {}
    if "eyes_union" not in {str(label) for label in mask_labels}:
        raise ValueError(
            "Assignment keypoints are only valid for subject-mask schemas that expose eyes_union."
        )

    group_name = str(assignment_keypoint_group)
    parent = root.get(group_name)
    run_name = str(assignment_keypoints_run)
    if parent is None or run_name not in parent:
        raise ValueError(f"Assignment keypoint source not found: {group_name}/{run_name}.")
    kp_group = parent[run_name]
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Assignment keypoint source {group_name}/{run_name} missing keypoints_roi.")
    success_name = next((name for name in KEYPOINT_SUCCESS_DATASET_CANDIDATES if name in kp_group), None)
    if success_name is None:
        raise ValueError(
            f"Assignment keypoint source {group_name}/{run_name} missing success dataset "
            f"({', '.join(KEYPOINT_SUCCESS_DATASET_CANDIDATES)})."
        )
    success_arr = kp_group.get(success_name)
    assert_row_alignment(
        int(total_rois),
        (
            (f"{group_name}/{run_name}/keypoints_roi", keypoints_roi),
            (f"{group_name}/{run_name}/{success_name}", success_arr),
        ),
        stage="subject-mask assignment keypoint source",
    )
    keypoint_shape = getattr(keypoints_roi, "shape", ())
    keypoint_count = int(keypoint_shape[1]) if len(keypoint_shape) >= 2 else None
    eye_indices = resolve_required_keypoint_indices_from_attrs(
        kp_group.attrs,
        EYE_KEYPOINT_LABELS,
        keypoint_count=keypoint_count,
    )
    payload = build_assignment_keypoint_attrs(
        run_name,
        assignment_keypoint_group=group_name,
        selection="cli_explicit",
    )
    payload["assignment_keypoint_success_dataset"] = str(success_name)
    payload["assignment_keypoint_eye_indices"] = {key: int(value) for key, value in eye_indices.items()}
    return payload


def _prepare_run_group(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    overwrite: bool,
    output_parent: str = SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
) -> Tuple[zarr.Group, str]:
    if output_parent not in SUBJECT_MASK_OUTPUT_PARENTS:
        raise ValueError(f"output_parent must be one of {SUBJECT_MASK_OUTPUT_PARENTS}; got {output_parent!r}.")
    parent = require_runs_parent(root, output_parent)
    resolved_name = run_name
    if resolved_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"subject_masks_{timestamp}"
        resolved_name = base_name
        suffix = 1
        while resolved_name in parent:
            resolved_name = f"{base_name}_{suffix:03d}"
            suffix += 1
    if resolved_name in parent:
        if not overwrite:
            raise ValueError(
                f"{output_parent}/{resolved_name} already exists. Pass --overwrite to replace it."
            )
        del parent[resolved_name]
    run_group = parent.create_group(resolved_name)
    mark_run_started(run_group, run_name=str(resolved_name), stage="subject_masks")
    note_pending_latest(parent, str(resolved_name))
    return run_group, str(resolved_name)


def _output_parent_from_args(args: argparse.Namespace) -> str:
    output_parent = str(getattr(args, "output_parent", SUBJECT_MASK_CANONICAL_OUTPUT_PARENT))
    if output_parent not in SUBJECT_MASK_OUTPUT_PARENTS:
        raise ValueError(f"--output-parent must be one of {SUBJECT_MASK_OUTPUT_PARENTS}; got {output_parent!r}.")
    return output_parent


def _is_shard_output_parent(output_parent: str) -> bool:
    return str(output_parent) == SUBJECT_MASK_SHARD_OUTPUT_PARENT


def _subject_mask_stage_path(output_parent: str, run_name: str, artifact: str = "mask_probs_roi") -> str:
    return f"{output_parent}/{run_name}/{artifact}"


def _shard_attrs_from_args(args: argparse.Namespace, *, output_parent: str) -> dict[str, object]:
    if not _is_shard_output_parent(output_parent):
        return {}
    alias_manifest = args.source_roi_cache_alias_manifest or args.roi_cache_manifest
    attrs: dict[str, object] = {
        "is_collection_shard": True,
        "stage_selector_eligible": False,
        "subject_mask_output_parent": output_parent,
        "canonical_stage_parent": SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
        "canonical_selector_publication": "suppressed_for_collection_shard",
        "registry_status_publication": "suppressed_for_collection_shard",
    }
    optional_values = {
        "source_collection_id": args.source_collection_id,
        "source_collection_path": args.source_collection_path,
        "source_clip_id": args.source_clip_id,
        "source_clip_index": args.source_clip_index,
        "source_work_unit_id": args.source_work_unit_id,
        "source_roi_cache_alias_manifest": alias_manifest,
        "source_roi_cache_row_index_path": args.source_roi_cache_row_index_path,
        "source_shard_id": args.source_shard_id,
    }
    for key, value in optional_values.items():
        if value is None:
            continue
        if isinstance(value, Path):
            attrs[key] = str(value)
        else:
            attrs[key] = int(value) if key == "source_clip_index" else str(value)
    return attrs


def _copy_detection_source_array(run_group: zarr.Group, crop_group: zarr.Group) -> None:
    array_name = "detection_source"
    if array_name not in crop_group:
        return
    src = crop_group[array_name]
    data = src[:]
    chunks = getattr(src, "chunks", None)
    if not chunks:
        chunks = tuple(max(1, min(dim, 1024)) for dim in data.shape)
    run_group.create_array(array_name, data=data, chunks=chunks, overwrite=True)


def _write_subject_mask_output_batch(
    batch: _SubjectMaskOutputBatch,
    *,
    probs_arr: zarr.Array,
    masks_arr: Optional[zarr.Array],
    prob_max: np.ndarray,
    mask_present: np.ndarray,
    area_px: np.ndarray,
    centroid_xy: np.ndarray,
    centroid_valid: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_valid: np.ndarray,
    progress: Progress,
    task: int,
    profiler: InferenceTimingProfiler,
) -> None:
    start = int(batch.start)
    stop = int(batch.stop)
    batch_count = max(0, stop - start)

    with profiler.time("output_write_probs", items=batch_count):
        probs_arr[start:stop] = batch.probs_out
    if masks_arr is not None:
        if batch.binary is None:
            raise ValueError("masks_roi materialization requested but binary masks were not returned.")
        with profiler.time("output_write_binary", items=batch_count):
            masks_arr[start:stop] = batch.binary

    with profiler.time("metric_compute", items=batch_count):
        prob_max[start:stop, :] = batch.metrics["prob_max"]
        mask_present[start:stop, :] = batch.metrics["mask_present"]
        area_px[start:stop, :] = batch.metrics["area_px"]
        centroid_xy[start:stop, :, :] = batch.metrics["centroid_xy"]
        centroid_valid[start:stop, :] = batch.metrics["centroid_valid"]
        bbox_xyxy[start:stop, :, :] = batch.metrics["bbox_xyxy"]
        bbox_valid[start:stop, :] = batch.metrics["bbox_valid"]

    with profiler.time("progress_update", items=batch_count):
        progress.advance(task, batch_count)


def _probability_array_digest(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    total_rows = int(array.shape[0])
    channels = int(array.shape[1])
    for channel in range(channels):
        for start in range(0, total_rows, int(row_step)):
            stop = min(total_rows, start + int(row_step))
            values = np.asarray(array[start:stop, channel, :, :])
            digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _postpack_probability_shards(
    run_group: zarr.Group,
    *,
    source_name: str,
    shard_rows: int,
    profiler: InferenceTimingProfiler,
) -> dict[str, object]:
    source = run_group[source_name]
    if int(source.ndim) != 4:
        raise ValueError(f"{source_name} must have shape (N,C,H,W); got {source.shape}.")
    chunks = tuple(int(value) for value in source.chunks)
    inner_rows = int(chunks[0])
    resolved_shard_rows = int(shard_rows)
    if resolved_shard_rows <= inner_rows:
        raise ValueError(
            f"--mask-probs-shard-rois must exceed inner chunk rows {inner_rows}; "
            f"got {resolved_shard_rows}."
        )
    if resolved_shard_rows % inner_rows != 0:
        raise ValueError(
            f"--mask-probs-shard-rois must be an integer multiple of "
            f"--mask-probs-chunk-rois ({inner_rows}); got {resolved_shard_rows}."
        )

    shape = tuple(int(value) for value in source.shape)
    shards = (resolved_shard_rows, 1, shape[2], shape[3])
    if MASK_PROBS_CANONICAL_ARRAY in run_group:
        del run_group[MASK_PROBS_CANONICAL_ARRAY]
    destination = run_group.create_array(
        MASK_PROBS_CANONICAL_ARRAY,
        shape=shape,
        dtype=source.dtype,
        chunks=chunks,
        shards=shards,
        fill_value=source.fill_value,
        overwrite=True,
        **_compression_kwargs(source),
    )
    destination.attrs.update(
        {
            "storage_layout": "indexed_sharding_v1",
            "inner_chunk_shape": list(chunks),
            "outer_shard_shape": list(shards),
            "postpack_source_array": source_name,
        }
    )

    write_started = time.perf_counter()
    with profiler.time("output_postpack_shards", items=shape[0]):
        for channel in range(shape[1]):
            for start in range(0, shape[0], resolved_shard_rows):
                stop = min(shape[0], start + resolved_shard_rows)
                values = np.asarray(source[start:stop, channel : channel + 1, :, :])
                destination[start:stop, channel : channel + 1, :, :] = values
    write_seconds = float(time.perf_counter() - write_started)

    validation_started = time.perf_counter()
    with profiler.time("output_postpack_validate", items=shape[0]):
        source_digest = _probability_array_digest(source, row_step=inner_rows)
        destination_digest = _probability_array_digest(destination, row_step=inner_rows)
    validation_seconds = float(time.perf_counter() - validation_started)
    if source_digest != destination_digest:
        raise RuntimeError(
            "Post-packed probability digest mismatch: "
            f"source={source_digest} destination={destination_digest}."
        )

    del run_group[source_name]
    return {
        "schema_id": MASK_PROBS_SHARDING_SCHEMA,
        "status": "complete",
        "source_working_array_removed": True,
        "canonical_array": MASK_PROBS_CANONICAL_ARRAY,
        "row_count": shape[0],
        "channel_count": shape[1],
        "inner_chunk_shape": list(chunks),
        "outer_shard_shape": list(shards),
        "inner_chunks_per_shard": int(resolved_shard_rows // inner_rows),
        "write_seconds": write_seconds,
        "validation_seconds": validation_seconds,
        "source_sha256": source_digest,
        "destination_sha256": destination_digest,
        "exact_match": True,
    }


def _raise_async_writer_error(errors: Sequence[BaseException]) -> None:
    if errors:
        raise RuntimeError("Async subject-mask output writer failed.") from errors[0]


def _write_subject_mask_outputs(
    run_group: zarr.Group,
    model: UNetSmall,
    roi_source: CropImageSource,
    *,
    batch_size: int,
    device: torch.device,
    mask_labels: Sequence[str],
    mask_probs_chunk_rois: Optional[int],
    mask_probs_shard_rois: Optional[int],
    mask_probs_dtype: str,
    write_masks_roi: bool,
    async_output: bool,
    output_queue_size: int,
    model_input_transform: ModelInputTransform,
    show_progress: bool,
    console: Console,
    timing_profiler: Optional[InferenceTimingProfiler],
) -> float:
    total_rois = int(roi_source.total_rois)
    height, width = map(int, roi_source.roi_shape)
    n_channels = int(len(mask_labels))
    storage_chunks = subject_mask_storage_chunks(total_rois, height, width)
    if mask_probs_chunk_rois is not None:
        storage_chunks = (max(1, min(int(mask_probs_chunk_rois), max(1, total_rois))), storage_chunks[1], storage_chunks[2], storage_chunks[3])
    if mask_probs_shard_rois is not None:
        shard_rows = int(mask_probs_shard_rois)
        if shard_rows <= int(storage_chunks[0]) or shard_rows % int(storage_chunks[0]) != 0:
            raise ValueError(
                "--mask-probs-shard-rois must exceed and be an integer multiple of "
                f"the effective inner chunk rows ({int(storage_chunks[0])}); got {shard_rows}."
            )
    metric_row_chunk = subject_mask_metric_row_chunk(total_rois)

    roi_array = roi_source.roi_array
    compression_kwargs = _compression_kwargs(roi_array) if roi_array is not None else {}
    stored_prob_dtype = np.dtype(np.uint8 if mask_probs_dtype == "uint8" else np.float16)

    masks_arr: Optional[zarr.Array] = None
    if write_masks_roi:
        masks_arr = run_group.create_array(
            "masks_roi",
            shape=(total_rois, n_channels, height, width),
            dtype=np.uint8,
            chunks=storage_chunks,
            fill_value=0,
            overwrite=True,
            **compression_kwargs,
        )
    probability_array_name = (
        MASK_PROBS_WORKING_ARRAY if mask_probs_shard_rois is not None else MASK_PROBS_CANONICAL_ARRAY
    )
    probs_arr = run_group.create_array(
        probability_array_name,
        shape=(total_rois, n_channels, height, width),
        dtype=stored_prob_dtype,
        chunks=storage_chunks,
        fill_value=np.uint8(0) if mask_probs_dtype == "uint8" else np.float16(0.0),
        overwrite=True,
        **compression_kwargs,
    )
    run_group.create_array(
        "available_channels",
        data=np.ones((n_channels,), dtype=bool),
        overwrite=True,
    )

    metrics_group = run_group.require_group("metrics")
    prob_max = np.zeros((total_rois, n_channels), dtype=np.float32)
    mask_present = np.zeros((total_rois, n_channels), dtype=bool)
    area_px = np.zeros((total_rois, n_channels), dtype=np.float32)
    centroid_xy = np.zeros((total_rois, n_channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((total_rois, n_channels), dtype=bool)
    bbox_xyxy = np.zeros((total_rois, n_channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((total_rois, n_channels), dtype=bool)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not show_progress,
    )
    task = progress.add_task("[cyan]Running inference[/cyan]", total=total_rois)

    def _sync_cuda(stage: str, *, items: int) -> None:
        profiler = timing_profiler or InferenceTimingProfiler(enabled=False)
        if profiler.enabled and device.type == "cuda":
            with profiler.time(stage, items=items):
                torch.cuda.synchronize(device)

    output_queue: Optional[Queue[object]] = None
    output_worker: Optional[Thread] = None
    output_sentinel = object()
    output_errors: list[BaseException] = []
    if async_output:
        output_queue = Queue(maxsize=max(1, int(output_queue_size)))

        def _output_worker() -> None:
            failed = False
            while True:
                item = output_queue.get()
                try:
                    if item is output_sentinel:
                        return
                    if failed:
                        continue
                    assert isinstance(item, _SubjectMaskOutputBatch)
                    _write_subject_mask_output_batch(
                        item,
                        probs_arr=probs_arr,
                        masks_arr=masks_arr,
                        prob_max=prob_max,
                        mask_present=mask_present,
                        area_px=area_px,
                        centroid_xy=centroid_xy,
                        centroid_valid=centroid_valid,
                        bbox_xyxy=bbox_xyxy,
                        bbox_valid=bbox_valid,
                        progress=progress,
                        task=task,
                        profiler=profiler,
                    )
                except BaseException as exc:  # pragma: no cover - exercised through caller failure
                    output_errors.append(exc)
                    failed = True
                finally:
                    output_queue.task_done()

        output_worker = Thread(
            target=_output_worker,
            name="subject-mask-output-writer",
            daemon=True,
        )

    stage_start = time.perf_counter()
    profiler = timing_profiler or InferenceTimingProfiler(enabled=False)
    with progress, torch.no_grad():
        if output_worker is not None:
            output_worker.start()
        try:
            for start in range(0, total_rois, batch_size):
                stop = min(start + batch_size, total_rois)
                batch_count = stop - start

                _raise_async_writer_error(output_errors)
                with profiler.time("roi_read", items=batch_count):
                    roi_np = roi_source.read_slice(start, stop)

                _sync_cuda("sync_before_h2d", items=batch_count)
                with profiler.time("h2d_copy", items=batch_count):
                    imgs = torch.from_numpy(roi_np).to(device, non_blocking=True)
                _sync_cuda("sync_after_h2d", items=batch_count)
                with profiler.time("input_normalize", items=batch_count):
                    imgs = _normalise_roi_tensor(imgs)
                with profiler.time("input_transform", items=batch_count):
                    imgs = model_input_transform.apply_torch_image_batch(imgs)
                _sync_cuda("sync_after_normalize", items=batch_count)

                amp_module = getattr(torch, "amp", None)
                if device.type == "cuda" and amp_module is not None and hasattr(amp_module, "autocast"):
                    autocast_cm = amp_module.autocast("cuda")
                elif device.type == "cuda" and hasattr(torch.cuda, "amp"):
                    autocast_cm = torch.cuda.amp.autocast()
                else:
                    autocast_cm = nullcontext()

                _sync_cuda("sync_before_forward", items=batch_count)
                with profiler.time("model_forward", items=batch_count):
                    with autocast_cm:
                        logits = model(imgs)
                    logits = model_input_transform.crop_torch_output(logits)
                _sync_cuda("sync_after_forward", items=batch_count)

                _sync_cuda("sync_before_d2h", items=batch_count)
                with profiler.time("d2h_copy", items=batch_count):
                    probs_out, binary, output_metrics = _postprocess_logits_on_device(
                        logits,
                        mask_probs_dtype=mask_probs_dtype,
                        return_binary=write_masks_roi,
                    )
                _sync_cuda("sync_after_d2h", items=batch_count)

                if probs_out.ndim == 3:
                    probs_out = probs_out[:, None, :, :]
                if binary is not None and binary.ndim == 3:
                    binary = binary[:, None, :, :]
                if probs_out.shape[1] != n_channels:
                    raise ValueError(
                        f"Checkpoint/model produced {probs_out.shape[1]} channels but expected {n_channels}."
                    )

                with profiler.time("output_postprocess", items=batch_count):
                    if mask_probs_dtype == "uint8":
                        probs_out = probs_out.astype(np.uint8, copy=False)
                    else:
                        probs_out = probs_out.astype(np.float16, copy=False)
                    if binary is not None:
                        binary = binary.astype(np.uint8, copy=False)

                output_batch = _SubjectMaskOutputBatch(
                    start=start,
                    stop=stop,
                    probs_out=probs_out,
                    binary=binary,
                    metrics=output_metrics,
                )
                if output_queue is None:
                    _write_subject_mask_output_batch(
                        output_batch,
                        probs_arr=probs_arr,
                        masks_arr=masks_arr,
                        prob_max=prob_max,
                        mask_present=mask_present,
                        area_px=area_px,
                        centroid_xy=centroid_xy,
                        centroid_valid=centroid_valid,
                        bbox_xyxy=bbox_xyxy,
                        bbox_valid=bbox_valid,
                        progress=progress,
                        task=task,
                        profiler=profiler,
                    )
                else:
                    _raise_async_writer_error(output_errors)
                    with profiler.time("output_queue_put", items=batch_count):
                        output_queue.put(output_batch)
        finally:
            if output_queue is not None:
                with profiler.time("output_queue_drain", items=total_rois):
                    output_queue.put(output_sentinel)
                    output_queue.join()
            if output_worker is not None:
                output_worker.join()
        _raise_async_writer_error(output_errors)

    metrics_group.create_array("prob_max", data=prob_max, chunks=(metric_row_chunk, n_channels), overwrite=True)
    metrics_group.create_array("mask_present", data=mask_present, chunks=(metric_row_chunk, n_channels), overwrite=True)
    metrics_group.create_array("area_px", data=area_px, chunks=(metric_row_chunk, n_channels), overwrite=True)
    metrics_group.create_array("centroid_xy", data=centroid_xy, chunks=(metric_row_chunk, n_channels, 2), overwrite=True)
    metrics_group.create_array("centroid_valid", data=centroid_valid, chunks=(metric_row_chunk, n_channels), overwrite=True)
    metrics_group.create_array("bbox_xyxy", data=bbox_xyxy, chunks=(metric_row_chunk, n_channels, 4), overwrite=True)
    metrics_group.create_array("bbox_valid", data=bbox_valid, chunks=(metric_row_chunk, n_channels), overwrite=True)
    if mask_probs_shard_rois is not None:
        run_group.attrs["mask_probs_postpack"] = _postpack_probability_shards(
            run_group,
            source_name=probability_array_name,
            shard_rows=int(mask_probs_shard_rois),
            profiler=profiler,
        )
    return float(time.perf_counter() - stage_start)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer unified subject-mask probabilities using a trained U-Net segmenter."
    )
    parser.add_argument("zarr_path", help="Path to Palette Zarr archive.")
    parser.add_argument("checkpoint", nargs="?", help="Path to trained U-Net checkpoint (.pt).")
    parser.add_argument("--checkpoint", dest="checkpoint_option", help="Path to trained U-Net checkpoint (.pt).")
    parser.add_argument(
        "--resolve-model-from-registry",
        action="store_true",
        help="Resolve the checkpoint from subject_mask_training_models instead of passing a path.",
    )
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for --resolve-model-from-registry.")
    parser.add_argument(
        "--model-coverage-class",
        default="dense_all_components",
        help="Required subject-mask model coverage_class when resolving from registry.",
    )
    parser.add_argument(
        "--model-component-coverage-key",
        help="Optional component_coverage_key filter when resolving from registry.",
    )
    parser.add_argument(
        "--model-label-schema-id",
        help="Optional label_schema_id filter when resolving from registry.",
    )
    parser.add_argument(
        "--model-include-non-success",
        action="store_true",
        help="Allow non-success registry model rows during model resolution.",
    )
    parser.add_argument(
        "--model-allow-missing-path",
        action="store_true",
        help="Do not filter resolved registry candidates whose model_path is missing on disk.",
    )
    parser.add_argument(
        "--model-require-unique",
        action="store_true",
        help="Fail registry model resolution when the top metric is tied.",
    )
    parser.add_argument(
        "--model-top-k",
        type=int,
        default=5,
        help="Number of registry candidates to retain in model-resolution provenance.",
    )
    parser.add_argument("--crop-run", help="Explicit crop run providing ROI images (default: latest/auto).")
    parser.add_argument("--run-name", help="Optional name for the output run.")
    parser.add_argument(
        "--output-parent",
        choices=SUBJECT_MASK_OUTPUT_PARENTS,
        default=SUBJECT_MASK_CANONICAL_OUTPUT_PARENT,
        help=(
            "Output parent group. Use subject_mask_shard_runs for clipped-collection "
            "shards that must not publish ordinary subject_mask_runs selectors."
        ),
    )
    parser.add_argument("--source-collection-id", help="Optional clipped collection id for shard outputs.")
    parser.add_argument("--source-collection-path", help="Optional clipped collection path for shard outputs.")
    parser.add_argument("--source-clip-id", help="Optional clip id for shard outputs.")
    parser.add_argument("--source-clip-index", type=int, help="Optional clip index for shard outputs.")
    parser.add_argument("--source-work-unit-id", help="Optional work-unit id for shard outputs.")
    parser.add_argument("--source-shard-id", help="Optional shard id for non-clip collection shards.")
    parser.add_argument(
        "--source-roi-cache-alias-manifest",
        type=Path,
        help=(
            "Optional cache alias manifest used for this shard. Defaults to "
            "--roi-cache-manifest when omitted."
        ),
    )
    parser.add_argument(
        "--source-roi-cache-row-index-path",
        type=Path,
        help="Optional row-index parquet path for the clipped collection flat ROI cache shard.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size used during inference (default: 256).")
    parser.add_argument("--device", help="Torch device to use (e.g. 'cuda:0', 'cpu').")
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=None,
        help="Optional square model input size. When larger than native ROI size, pair with --model-input-transform auto/pad_to_size.",
    )
    parser.add_argument(
        "--model-input-transform",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        default="auto",
        help=(
            "Reversible transform from native ROI crops to model input size. "
            "'auto' is identity when sizes match and centered zero-padding when --model-input-size is larger."
        ),
    )
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument("--roi-cache-dir", type=Path, default=None, help="Optional scratch directory for temporary ROI caches.")
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        default=None,
        help="Optional flat_bin_v1 ROI cache manifest to read instead of materializing/re-decoding ROIs.",
    )
    parser.add_argument(
        "--roi-cache-expected-archive-path",
        type=Path,
        default=None,
        help=(
            "Canonical analysis zarr path expected by --roi-cache-manifest. "
            "Use when writing into a staged zarr overlay whose logical source "
            "archive is the original recording zarr."
        ),
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--mask-probs-chunk-rois",
        type=int,
        default=32,
        help="ROI chunk length override for mask_probs_roi and masks_roi outputs (default: 32).",
    )
    parser.add_argument(
        "--mask-probs-shard-rois",
        type=int,
        default=None,
        help=(
            "Optional outer storage-shard row count for mask_probs_roi. Inference first writes "
            "ordinary inner chunks, then packs and exact-validates whole indexed shards before "
            "run completion (production candidate: 2048)."
        ),
    )
    parser.add_argument(
        "--mask-probs-dtype",
        choices=("float16", "uint8"),
        default="uint8",
        help="Storage dtype for mask_probs_roi (default: uint8 for analysis runs).",
    )
    parser.add_argument(
        "--write-masks-roi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Materialize thresholded binary masks_roi alongside mask_probs_roi "
            "(default: false for probability-first raw runs; use --write-masks-roi for compatibility output)."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the Rich progress bar during inference (default: false for log-friendly runs).",
    )
    parser.add_argument(
        "--async-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap model inference with a bounded background writer for Zarr output and spatial metrics (default: true).",
    )
    parser.add_argument(
        "--output-queue-size",
        type=int,
        default=2,
        help="Maximum number of dense output batches buffered by --async-output (default: 2).",
    )
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=KEYPOINT_GROUP_CHOICES,
        help="Keypoint group to use later when splitting eyes_union into anatomical LR eyes.",
    )
    parser.add_argument(
        "--assignment-keypoint-run",
        help="Keypoint run to use later when splitting eyes_union into anatomical LR eyes.",
    )
    parser.add_argument("--profile-timings", action="store_true", help="Collect per-stage timing diagnostics.")
    parser.add_argument(
        "--defer-registry-status",
        action="store_true",
        help=(
            "Write and complete the zarr run group without emitting registry "
            "step status. Use for local staged output that will be published to "
            "the canonical archive before status is emitted."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the requested output run if it exists.")
    return parser


def _resolve_registry_checkpoint(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        selected, candidates = resolve_best_subject_mask_model(
            registry,
            coverage_class=args.model_coverage_class,
            component_coverage_key=args.model_component_coverage_key,
            label_schema_id=args.model_label_schema_id,
            include_non_success=bool(args.model_include_non_success),
            require_existing_path=not bool(args.model_allow_missing_path),
            require_unique=bool(args.model_require_unique),
        )
    finally:
        registry.close()

    payload = build_resolution_payload(
        registry_path=registry_path,
        selected=selected,
        candidates=candidates,
        top_k=int(args.model_top_k),
        parameters={
            "coverage_class": args.model_coverage_class,
            "component_coverage_key": args.model_component_coverage_key,
            "label_schema_id": args.model_label_schema_id,
            "include_non_success": bool(args.model_include_non_success),
            "require_existing_path": not bool(args.model_allow_missing_path),
            "require_unique": bool(args.model_require_unique),
            "top_k": int(args.model_top_k),
        },
    )
    return Path(selected.model_path).expanduser().resolve(), payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    console.print("[bold cyan]Running U-Net subject-mask inference[/bold cyan]\n")

    checkpoint_value = args.checkpoint_option or args.checkpoint
    if checkpoint_value and args.resolve_model_from_registry:
        raise ValueError("Pass either a checkpoint path or --resolve-model-from-registry, not both.")
    model_resolution_payload: Optional[dict[str, Any]] = None
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    elif args.resolve_model_from_registry:
        checkpoint_path, model_resolution_payload = _resolve_registry_checkpoint(args)
        selected = model_resolution_payload.get("selected", {})
        console.print(
            "[dim]Resolved subject-mask model from registry: "
            f"{selected.get('run_id')} -> {checkpoint_path}[/dim]"
        )
    else:
        raise ValueError(
            "U-Net subject-mask inference requires a checkpoint path or --resolve-model-from-registry."
        )
    device = _resolve_device(args.device)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)

    label_schema_id, mask_labels = _resolve_checkpoint_schema(checkpoint)

    zarr_path = Path(args.zarr_path).expanduser().resolve()
    root = zarr.open(str(zarr_path), mode="a", use_consolidated=False)

    crop_source = CropImageSource.open(
        root,
        crop_run=args.crop_run,
        zarr_path=zarr_path,
        roi_cache_policy=args.roi_cache_policy,
        roi_live_acceleration=args.roi_live_acceleration,
        roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
        roi_cache_dir=args.roi_cache_dir,
        roi_cache_manifest=args.roi_cache_manifest,
        roi_cache_expected_archive_path=args.roi_cache_expected_archive_path,
        console=console,
    )
    crop_group = crop_source.crop_group
    crop_run_name = crop_source.crop_run_name
    total_rois = int(crop_source.total_rois)
    if total_rois == 0:
        crop_source.close()
        raise ValueError("ROI image array is empty; nothing to segment.")
    roi_height, roi_width = map(int, crop_source.roi_shape)
    model_input_size = int(args.model_input_size) if args.model_input_size is not None else max(roi_height, roi_width)
    model_input_transform = resolve_model_input_transform(
        (roi_height, roi_width),
        mode=str(args.model_input_transform),
        model_hw=(model_input_size, model_input_size),
    )
    try:
        assignment_keypoint_attrs = _resolve_assignment_keypoint_attrs(
            root,
            assignment_keypoint_group=args.assignment_keypoint_group,
            assignment_keypoints_run=args.assignment_keypoint_run,
            total_rois=total_rois,
            mask_labels=mask_labels,
        )
    except Exception:
        crop_source.close()
        raise

    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    git_info = get_git_info()
    output_parent = _output_parent_from_args(args)
    shard_output = _is_shard_output_parent(output_parent)
    run_group, resolved_run_name = _prepare_run_group(
        root,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        output_parent=output_parent,
    )
    shard_attrs = _shard_attrs_from_args(args, output_parent=output_parent)
    timing_profiler = InferenceTimingProfiler(enabled=bool(args.profile_timings))

    try:
        copy_row_lineage_arrays(run_group, crop_group, total_rois=total_rois)
        write_direct_source_crop_row_ids(run_group, total_rois=total_rois)
        _copy_detection_source_array(run_group, crop_group)
        if "detection_source" not in run_group:
            run_group.create_array(
                "detection_source",
                data=np.zeros((total_rois,), dtype=np.int8),
                overwrite=True,
            )

        duration = _write_subject_mask_outputs(
            run_group,
            model,
            crop_source,
            batch_size=int(args.batch_size),
            device=device,
            mask_labels=mask_labels,
            mask_probs_chunk_rois=args.mask_probs_chunk_rois,
            mask_probs_shard_rois=args.mask_probs_shard_rois,
            mask_probs_dtype=str(args.mask_probs_dtype),
            write_masks_roi=bool(args.write_masks_roi),
            async_output=bool(args.async_output),
            output_queue_size=int(args.output_queue_size),
            model_input_transform=model_input_transform,
            show_progress=bool(args.progress),
            console=console,
            timing_profiler=timing_profiler,
        )
    finally:
        crop_source.close()

    created_at = datetime.now(timezone.utc).isoformat()
    crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=crop_source.storage_mode,
    )
    crop_pixel_attrs = build_source_roi_pixel_attrs(crop_source)
    run_group.attrs.update(
        {
            "method": "unet_subject_mask_segmenter",
            "run_semantics": "unet_subject_mask_inference",
            "subject_mask_output_parent": output_parent,
            "source_crop_run": crop_run_name,
            **crop_snapshot_attrs,
            **crop_pixel_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "source_roi_cache_used": bool(crop_source.roi_cache_used),
            "source_roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
            "source_roi_cache_canonical_path": getattr(crop_source, "roi_cache_canonical_path", None),
            "source_roi_cache_expected_archive_path": (
                str(args.roi_cache_expected_archive_path)
                if args.roi_cache_expected_archive_path is not None
                else None
            ),
            "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "label_schema_id": label_schema_id,
            "mask_labels": list(mask_labels),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "probability_semantics": "sigmoid_multilabel_logits",
            "probabilities_dtype": str(args.mask_probs_dtype),
            "probabilities_encoding": "linear_uint8_0_255" if args.mask_probs_dtype == "uint8" else "unit_float",
            "mask_probability_threshold": 0.5,
            "masks_roi_materialized": bool(args.write_masks_roi),
            "binary_masks_materialized": bool(args.write_masks_roi),
            "binary_masks_source": (
                "threshold(mask_probs_roi, threshold=0.5)" if args.write_masks_roi else "not_materialized"
            ),
            "input_format": "gray",
            "model_input_transform": model_input_transform.to_attrs(),
            "model_input_transform_name": model_input_transform.name,
            "model_input_shape_hw": list(model_input_transform.model_shape),
            "native_roi_shape_hw": list(model_input_transform.native_shape),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_best_val_dice": float(checkpoint.get("best_val_dice", float("nan"))),
            "inference_device": str(device),
            "inference_batch_size": int(args.batch_size),
            "async_output": bool(args.async_output),
            "output_queue_size": int(args.output_queue_size),
            "duration_seconds": float(duration),
            "inference_duration_seconds": float(duration),
            "profile_timings_enabled": bool(args.profile_timings),
            "created_at_utc": created_at,
            **shard_attrs,
        }
    )
    if assignment_keypoint_attrs:
        run_group.attrs.update(assignment_keypoint_attrs)
    if crop_source.roi_cache_key is not None:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path is not None:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    if args.mask_probs_chunk_rois is not None:
        run_group.attrs["mask_probs_chunk_rois"] = int(args.mask_probs_chunk_rois)
    if args.mask_probs_shard_rois is not None:
        run_group.attrs["mask_probs_shard_rois"] = int(args.mask_probs_shard_rois)
        run_group.attrs["mask_probs_storage_layout"] = "indexed_sharding_v1"
    else:
        run_group.attrs["mask_probs_storage_layout"] = "regular_chunks_v1"
    if model_resolution_payload is not None:
        selected = model_resolution_payload.get("selected", {})
        if not isinstance(selected, dict):
            selected = {}
        run_group.attrs["model_resolution_mode"] = "registry"
        run_group.attrs["model_resolution_task"] = "subject_masks"
        run_group.attrs["model_resolution_registry_path"] = model_resolution_payload.get("registry_path")
        run_group.attrs["model_resolution_resolved_at_utc"] = model_resolution_payload.get("resolved_at_utc")
        run_group.attrs["model_resolution_selected_run_id"] = selected.get("run_id")
        run_group.attrs["model_resolution_selected_set_id"] = selected.get("set_id")
        run_group.attrs["model_resolution_selected_model_path"] = selected.get("model_path")
        run_group.attrs["model_resolution_selected_coverage_class"] = selected.get("coverage_class")
        run_group.attrs["model_resolution_selected_component_coverage_key"] = selected.get("component_coverage_key")
        run_group.attrs["model_resolution_selected_metric_name"] = selected.get("best_metric_name")
        run_group.attrs["model_resolution_selected_metric_value"] = selected.get("best_metric_value")
        run_group.attrs["model_resolution_candidates_json"] = json.dumps(
            model_resolution_payload.get("candidates", []),
            sort_keys=True,
        )
    else:
        selected = {}

    metrics_group = run_group.get("metrics")
    mask_present_array = metrics_group.get("mask_present") if metrics_group is not None else None
    if mask_present_array is not None:
        mask_present_values = np.asarray(mask_present_array[:], dtype=bool)
        nonempty_rows = np.any(mask_present_values, axis=1) if mask_present_values.ndim == 2 else np.zeros((total_rois,), dtype=bool)
    elif "masks_roi" in run_group:
        masks_array = np.asarray(run_group["masks_roi"][:], dtype=np.uint8)
        nonempty_rows = np.any(masks_array > 0, axis=(1, 2, 3))
    else:
        nonempty_rows = np.zeros((total_rois,), dtype=bool)
    run_group.attrs["summary_statistics"] = {
        "rows_total": int(total_rois),
        "rows_with_nonempty_masks": int(np.sum(nonempty_rows)),
        "rows_empty_masks": int(total_rois - np.sum(nonempty_rows)),
        "output_run": resolved_run_name,
        "crop_run": str(crop_run_name),
        "duration_seconds": float(duration),
        "created_at_utc": created_at,
        "masks_roi_materialized": bool(args.write_masks_roi),
    }

    for label in mask_labels:
        write_subject_mask_component_provenance(
            run_group,
            component_name=label,
            source_stage=output_parent,
            source_run=resolved_run_name,
            source_method=str(run_group.attrs["method"]),
            source_channels=[label],
            source_label_schema_id=label_schema_id,
            source_created_at_utc=created_at,
        )

    if timing_profiler.enabled:
        run_group.attrs["timing_profile"] = timing_profiler.summary(
            total_items=int(total_rois),
            wall_seconds=float(duration),
            notes=[
                "roi_read measures ROI slice fetch from the active crop image source.",
                "sync_before_* and sync_after_* measure explicit CUDA synchronize calls used to attribute queued GPU work deterministically.",
                "input_normalize runs after the device transfer so dtype conversion, scaling, and clipping can execute on GPU.",
                "h2d_copy and model_forward are measured separately for the U-Net loop.",
                "d2h_copy includes sigmoid + clamp + dtype conversion, on-device spatial metrics, probability transfer, and optional binary transfer when masks_roi is materialized.",
                "output_write_probs measures dense probability Zarr writes; output_write_binary is present only when masks_roi is materialized.",
                "output_postpack_shards writes complete immutable probability storage shards after inference; output_postpack_validate exact-checks decoded values before removing the working array.",
                "metric_compute covers copying precomputed per-batch metrics into full-run metric arrays.",
                "output_queue_put and output_queue_drain appear when --async-output overlaps inference with background output writes.",
            ],
        )

    platform_info = env_info.get("platform", {})
    provenance_inputs = {
        "output_parent": output_parent,
        "source_crop_run": str(crop_run_name),
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "roi_cache_used": bool(crop_source.roi_cache_used),
        "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "roi_cache_key": crop_source.roi_cache_key,
        "roi_cache_path": crop_source.roi_cache_path,
        "roi_cache_canonical_path": getattr(crop_source, "roi_cache_canonical_path", None),
        "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "frame_source": crop_source.frame_source_kind,
        "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
    }
    provenance_inputs.update(shard_attrs)
    provenance_inputs.update(assignment_keypoint_attrs)
    if model_resolution_payload is not None:
        provenance_inputs["model_resolution"] = model_resolution_payload
    if crop_source.roi_cache_path is not None:
        provenance_inputs["roi_cache_path"] = crop_source.roi_cache_path
    if getattr(crop_source, "roi_cache_canonical_path", None) is not None:
        provenance_inputs["roi_cache_canonical_path"] = crop_source.roi_cache_canonical_path
    provenance = build_stage_provenance(
        stage="subject_masks",
        command=" ".join(sys.argv),
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
            "batch_size": int(args.batch_size),
            "device": str(device),
            "output_parent": output_parent,
            "model_input_size": int(model_input_size),
            "model_input_transform": model_input_transform.to_attrs(),
            "label_schema_id": label_schema_id,
            "mask_labels": list(mask_labels),
            "mask_probs_chunk_rois": int(args.mask_probs_chunk_rois),
            "mask_probs_shard_rois": (
                int(args.mask_probs_shard_rois) if args.mask_probs_shard_rois is not None else None
            ),
            "mask_probs_storage_layout": run_group.attrs["mask_probs_storage_layout"],
            "mask_probs_dtype": str(args.mask_probs_dtype),
            "write_masks_roi": bool(args.write_masks_roi),
            "async_output": bool(args.async_output),
            "output_queue_size": int(args.output_queue_size),
            "progress": bool(args.progress),
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_manifest": str(args.roi_cache_manifest) if args.roi_cache_manifest else None,
            "source_roi_cache_alias_manifest": (
                str(args.source_roi_cache_alias_manifest)
                if args.source_roi_cache_alias_manifest
                else None
            ),
            "source_roi_cache_row_index_path": (
                str(args.source_roi_cache_row_index_path)
                if args.source_roi_cache_row_index_path
                else None
            ),
            "roi_live_acceleration": crop_source.roi_live_acceleration_requested,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "mask_probs_postpack": run_group.attrs.get("mask_probs_postpack"),
        },
        inputs=provenance_inputs,
        artifacts={
            "checkpoint_path": str(checkpoint_path),
            "segmenter": "unet",
            "label_schema_id": label_schema_id,
            "masks_roi_materialized": bool(args.write_masks_roi),
            "mask_probs_storage_layout": run_group.attrs["mask_probs_storage_layout"],
            "mask_probs_postpack": run_group.attrs.get("mask_probs_postpack"),
            "model_resolution": model_resolution_payload,
        },
    )
    write_stage_provenance(run_group, provenance)
    checkpoint_hash = selected.get("model_sha256") if isinstance(selected.get("model_sha256"), str) else None
    run_provenance = append_input_artifacts(
        build_run_provenance_from_stage_record(provenance),
        [
            fingerprint_artifact(
                checkpoint_path,
                role="subject_mask_unet_checkpoint",
                registry_hash=checkpoint_hash,
            )
        ],
    )
    mark_run_complete(
        run_group,
        parent_group=root[output_parent],
        run_name=resolved_run_name,
        run_provenance=run_provenance,
    )
    if shard_output:
        run_group.attrs["registry_status_deferred_reason"] = "collection_shard_not_canonical_stage_output"
    if not args.defer_registry_status and not shard_output:
        emit_subject_mask_stage_completion(
            root,
            zarr_path,
            run_group=run_group,
            run_name=resolved_run_name,
            source=_SUBJECT_MASKS_STATUS_SOURCE,
            console=console,
            invalidate_on_ok=True,
        )

    console.print(
        f"\n[green]✓[/green] U-Net subject masks written to "
        f"[cyan]{_subject_mask_stage_path(output_parent, resolved_run_name)}[/cyan] "
        f"({total_rois:,} ROIs processed in {duration:.1f}s)."
    )
    if timing_profiler.enabled:
        console.print("[bold]Timing Profile:[/bold]")
        for line in timing_profiler.render_lines(total_items=total_rois, wall_seconds=duration, limit=8):
            console.print(f"[dim]{line}[/dim]")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
