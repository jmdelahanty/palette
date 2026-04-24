#!/usr/bin/env python3
"""Merge compatible subject_mask_runs into one canonical raw subject-mask run."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr

from fisheye.shared.provenance_attrs import (
    CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR,
    SOURCE_CROP_REVISION_ATTR,
    SOURCE_CROP_SIGNATURE_ATTR,
    SOURCE_CROP_STORAGE_MODE_ATTR,
    build_source_keypoints_attrs,
    extract_source_crop_snapshot_attrs,
    resolve_source_keypoints_run,
)
from fisheye.shared.row_lineage import assert_row_lineage_sources_equal, copy_row_lineage_arrays
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.subject_mask_chunks import subject_mask_metric_row_chunk, subject_mask_storage_chunks
from fisheye.shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from fisheye.shared.type_conversions import normalize_attr
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root

TARGET_LABEL_SCHEMA = "subject_v1_lr"
TARGET_LABELS: tuple[str, ...] = ("subject_body", "eye_left", "eye_right", "swim_bladder")
TARGET_AVAILABLE_CHANNELS = np.asarray([True, True, True, False], dtype=bool)
_REQUIRED_CROP_SNAPSHOT_FIELDS = (
    SOURCE_CROP_STORAGE_MODE_ATTR,
    SOURCE_CROP_SIGNATURE_ATTR,
    SOURCE_CROP_REVISION_ATTR,
)
_OPTIONAL_CROP_SNAPSHOT_FIELDS = (
    CANONICAL_SOURCE_DETECT_REVIEW_STATUS_REF_ATTR,
)


@dataclass(frozen=True)
class ResolvedSubjectRun:
    run_name: str
    run_group: zarr.Group
    masks_roi: zarr.Array
    mask_probs_roi: Optional[zarr.Array]
    crop_run: str
    mask_labels: tuple[str, ...]
    available_channels: np.ndarray
    detection_source: zarr.Array
    frame_indices: Optional[zarr.Array]
    frame_counts: Optional[zarr.Array]
    detection_indices: Optional[zarr.Array]
    source_refined_row_ids: Optional[zarr.Array]
    source_detect_row_index: Optional[zarr.Array]
    source_keypoints_run: Optional[str]
    source_keypoint_group: Optional[str]
    source_crop_snapshot: dict[str, Any]
    probabilities_encoding: str
    probability_source_path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_encoding(value: Any) -> Optional[str]:
    text = normalize_attr(value)
    if text in {"unit_float", "linear_uint8_0_255"}:
        return text
    return None


def _default_encoding_for_dtype(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "linear_uint8_0_255"
    return "unit_float"


def _target_mask_chunks(total_rows: int, height: int, width: int) -> tuple[int, int, int, int]:
    return subject_mask_storage_chunks(total_rows, height, width)


def _target_metric_chunks(total_rows: int) -> tuple[int, int]:
    return (subject_mask_metric_row_chunk(total_rows), 1)


def _decode_probabilities(batch: np.ndarray, *, encoding: str) -> np.ndarray:
    decoded = batch.astype(np.float32, copy=False)
    if np.issubdtype(batch.dtype, np.integer) and encoding == "linear_uint8_0_255":
        max_value = float(np.iinfo(batch.dtype).max)
        if max_value > 1.0:
            decoded /= max_value
    decoded = np.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(decoded, 0.0, 1.0, out=decoded)


def _component_index(source: ResolvedSubjectRun, component_name: str) -> int:
    if component_name not in source.mask_labels:
        raise ValueError(f"subject_mask_runs/{source.run_name} does not define component '{component_name}'.")
    idx = source.mask_labels.index(component_name)
    if idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[idx]):
        raise ValueError(f"subject_mask_runs/{source.run_name} has component '{component_name}' marked unavailable.")
    return idx


def _resolve_subject_run(root: zarr.Group, run_name: str) -> ResolvedSubjectRun:
    parent = root.get("subject_mask_runs")
    if parent is None:
        raise ValueError("Missing subject_mask_runs group.")
    if run_name not in parent:
        raise ValueError(f"subject_mask_runs/{run_name} not found.")
    group = parent[run_name]
    masks_roi = group.get("masks_roi")
    if masks_roi is None:
        raise ValueError(f"subject_mask_runs/{run_name} missing masks_roi.")
    if len(masks_roi.shape) != 4:
        raise ValueError(f"subject_mask_runs/{run_name} masks_roi must be 4D, got {masks_roi.shape}.")
    detection_source = group.get("detection_source")
    if detection_source is None:
        raise ValueError(f"subject_mask_runs/{run_name} missing detection_source.")
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise ValueError(f"subject_mask_runs/{run_name} missing usable mask_labels attr.")
    available = group.get("available_channels")
    if available is None:
        raise ValueError(f"subject_mask_runs/{run_name} missing available_channels.")
    crop_run = normalize_attr(group.attrs.get("source_crop_run"))
    if not crop_run:
        raise ValueError(f"subject_mask_runs/{run_name} missing source_crop_run attr.")
    crop_parent = root.get("crop_runs")
    crop_group = crop_parent.get(crop_run) if crop_parent is not None else None

    def _lineage_array(name: str) -> Optional[zarr.Array]:
        if name in group:
            return group[name]
        if crop_group is not None:
            return crop_group.get(name)
        return None

    mask_probs_roi = group.get("mask_probs_roi")
    if mask_probs_roi is not None and tuple(mask_probs_roi.shape) != tuple(masks_roi.shape):
        raise ValueError(
            f"subject_mask_runs/{run_name} mask_probs_roi shape {mask_probs_roi.shape} "
            f"does not match masks_roi {masks_roi.shape}."
        )
    probabilities_encoding = _normalize_encoding(group.attrs.get("probabilities_encoding"))
    if probabilities_encoding is None:
        probs_dtype = mask_probs_roi.dtype if mask_probs_roi is not None else masks_roi.dtype
        probabilities_encoding = _default_encoding_for_dtype(np.dtype(probs_dtype))
    probability_source_path = (
        f"subject_mask_runs/{run_name}/mask_probs_roi" if mask_probs_roi is not None else f"subject_mask_runs/{run_name}/masks_roi"
    )
    return ResolvedSubjectRun(
        run_name=run_name,
        run_group=group,
        masks_roi=masks_roi,
        mask_probs_roi=mask_probs_roi,
        crop_run=str(crop_run),
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        detection_source=detection_source,
        frame_indices=_lineage_array("frame_indices"),
        frame_counts=_lineage_array("frame_counts"),
        detection_indices=_lineage_array("detection_indices"),
        source_refined_row_ids=_lineage_array("source_refined_row_ids"),
        source_detect_row_index=_lineage_array("source_detect_row_index"),
        source_keypoints_run=normalize_attr(resolve_source_keypoints_run(group.attrs)),
        source_keypoint_group=normalize_attr(group.attrs.get("source_keypoint_group")),
        source_crop_snapshot=extract_source_crop_snapshot_attrs(group.attrs),
        probabilities_encoding=str(probabilities_encoding),
        probability_source_path=probability_source_path,
    )


def _required_array_equal(name: str, left: zarr.Array, right: zarr.Array) -> None:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Alignment mismatch for {name}: {left.shape} != {right.shape}.")
    if not np.array_equal(np.asarray(left[:]), np.asarray(right[:])):
        raise ValueError(f"Alignment mismatch for {name}.")


def _source_lineage_arrays(source: ResolvedSubjectRun) -> dict[str, object | None]:
    return {
        "frame_indices": source.frame_indices,
        "frame_counts": source.frame_counts,
        "detection_indices": source.detection_indices,
        "source_refined_row_ids": source.source_refined_row_ids,
        "source_detect_row_index": source.source_detect_row_index,
    }


def _semantic_probabilities(
    source: ResolvedSubjectRun,
    row_slice: slice,
    component_idx: int,
) -> np.ndarray:
    if source.mask_probs_roi is None:
        return np.asarray(source.masks_roi[row_slice, component_idx : component_idx + 1], dtype=np.float32)
    batch = np.asarray(source.mask_probs_roi[row_slice, component_idx : component_idx + 1])
    return _decode_probabilities(batch, encoding=source.probabilities_encoding)


def _resolve_shared_crop_snapshot(
    body_source: ResolvedSubjectRun,
    eye_source: ResolvedSubjectRun,
) -> dict[str, Any]:
    missing_fields = [
        field
        for field in _REQUIRED_CROP_SNAPSHOT_FIELDS
        if field not in body_source.source_crop_snapshot or field not in eye_source.source_crop_snapshot
    ]
    if missing_fields:
        raise ValueError(
            "Missing required crop snapshot fields for merge: " + ", ".join(sorted(missing_fields))
        )

    mismatches: list[str] = []
    shared: dict[str, Any] = {}
    for field in (*_REQUIRED_CROP_SNAPSHOT_FIELDS, *_OPTIONAL_CROP_SNAPSHOT_FIELDS):
        body_value = body_source.source_crop_snapshot.get(field)
        eye_value = eye_source.source_crop_snapshot.get(field)
        if body_value != eye_value:
            mismatches.append(f"{field}: {body_value!r} != {eye_value!r}")
            continue
        if body_value is not None:
            shared[field] = body_value

    if mismatches:
        raise ValueError("Alignment mismatch for crop snapshot fields: " + "; ".join(mismatches))

    return shared


def _build_component_provenance_entry(
    *,
    source_stage: Optional[str],
    source_run: Optional[str],
    source_channel: str,
    source_probability_path: Optional[str],
    source_crop_run: str,
    source_crop_snapshot: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_stage": source_stage,
        "source_run": source_run,
        "source_channel": source_channel,
        "source_probability_path": source_probability_path,
        "source_crop_run": source_crop_run,
    }
    payload.update(source_crop_snapshot)
    return payload


def _compute_metrics(masks_roi: np.ndarray, probs_roi: np.ndarray) -> dict[str, np.ndarray]:
    if masks_roi.shape != probs_roi.shape:
        raise ValueError(f"Metric inputs must share shape, got {masks_roi.shape} vs {probs_roi.shape}.")
    n_rows, n_channels = int(masks_roi.shape[0]), int(masks_roi.shape[1])
    prob_max = probs_roi.max(axis=(2, 3), initial=0.0).astype(np.float32, copy=False)
    mask_present = masks_roi.any(axis=(2, 3))
    area_px = masks_roi.sum(axis=(2, 3), dtype=np.int64).astype(np.float32, copy=False)
    centroid_xy = np.zeros((n_rows, n_channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((n_rows, n_channels), dtype=bool)
    bbox_xyxy = np.zeros((n_rows, n_channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((n_rows, n_channels), dtype=bool)

    for row_idx in range(n_rows):
        for channel_idx in range(n_channels):
            mask = np.asarray(masks_roi[row_idx, channel_idx], dtype=bool)
            if not bool(mask_present[row_idx, channel_idx]):
                continue
            ys, xs = np.nonzero(mask)
            centroid_xy[row_idx, channel_idx] = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
            centroid_valid[row_idx, channel_idx] = True
            bbox_xyxy[row_idx, channel_idx] = np.asarray(
                [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                dtype=np.float32,
            )
            bbox_valid[row_idx, channel_idx] = True

    return {
        "prob_max": prob_max,
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def merge_subject_mask_runs(
    zarr_path: Path | str,
    *,
    body_run: str,
    eye_run: str,
    run_name: str,
    batch_size: int = 256,
    overwrite: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r+" if apply else "r")
    body_source = _resolve_subject_run(root, body_run)
    eye_source = _resolve_subject_run(root, eye_run)

    body_idx = _component_index(body_source, "subject_body")
    eye_left_idx = _component_index(eye_source, "eye_left")
    eye_right_idx = _component_index(eye_source, "eye_right")

    if body_source.crop_run != eye_source.crop_run:
        raise ValueError(
            f"Alignment mismatch for source_crop_run: {body_source.crop_run!r} != {eye_source.crop_run!r}."
        )
    shared_crop_snapshot = _resolve_shared_crop_snapshot(body_source, eye_source)
    if int(body_source.masks_roi.shape[0]) != int(eye_source.masks_roi.shape[0]):
        raise ValueError(
            f"Row-count mismatch: {body_source.masks_roi.shape[0]} != {eye_source.masks_roi.shape[0]}."
        )
    if tuple(body_source.masks_roi.shape[2:]) != tuple(eye_source.masks_roi.shape[2:]):
        raise ValueError(
            f"ROI shape mismatch: {body_source.masks_roi.shape[2:]} != {eye_source.masks_roi.shape[2:]}."
        )
    _required_array_equal("detection_source", body_source.detection_source, eye_source.detection_source)
    assert_row_lineage_sources_equal(_source_lineage_arrays(body_source), _source_lineage_arrays(eye_source))

    total_rows = int(body_source.masks_roi.shape[0])
    height = int(body_source.masks_roi.shape[2])
    width = int(body_source.masks_roi.shape[3])
    created_at = _utc_now()
    shared_crop_inputs = {"source_crop_run": body_source.crop_run, **shared_crop_snapshot}
    component_provenance = {
        "components": {
            "subject_body": _build_component_provenance_entry(
                source_stage="subject_mask_runs",
                source_run=body_source.run_name,
                source_channel="subject_body",
                source_probability_path=body_source.probability_source_path,
                source_crop_run=body_source.crop_run,
                source_crop_snapshot=shared_crop_snapshot,
            ),
            "eye_left": _build_component_provenance_entry(
                source_stage="subject_mask_runs",
                source_run=eye_source.run_name,
                source_channel="eye_left",
                source_probability_path=eye_source.probability_source_path,
                source_crop_run=body_source.crop_run,
                source_crop_snapshot=shared_crop_snapshot,
            ),
            "eye_right": _build_component_provenance_entry(
                source_stage="subject_mask_runs",
                source_run=eye_source.run_name,
                source_channel="eye_right",
                source_probability_path=eye_source.probability_source_path,
                source_crop_run=body_source.crop_run,
                source_crop_snapshot=shared_crop_snapshot,
            ),
            "swim_bladder": _build_component_provenance_entry(
                source_stage=None,
                source_run=None,
                source_channel="swim_bladder",
                source_probability_path=None,
                source_crop_run=body_source.crop_run,
                source_crop_snapshot=shared_crop_snapshot,
            ),
        }
    }
    summary = {
        "zarr_path": str(zarr_path),
        "status": "would_update" if not apply else "updated",
        "target_run": run_name,
        "body_run": body_source.run_name,
        "eye_run": eye_source.run_name,
        "label_schema_id": TARGET_LABEL_SCHEMA,
        "available_channels": [bool(v) for v in TARGET_AVAILABLE_CHANNELS.tolist()],
        "source_crop_run": body_source.crop_run,
        "source_crop_snapshot": dict(shared_crop_snapshot),
        "component_provenance": component_provenance,
    }
    if not apply:
        return summary

    start = time.perf_counter()
    parent = root.require_group("subject_mask_runs")
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"subject_mask_runs/{run_name} already exists. Pass overwrite=True to replace it.")
        del parent[run_name]
    run_group = parent.create_group(run_name)
    parent.attrs["latest"] = run_name

    run_group.attrs.update(
        {
            "source_crop_run": body_source.crop_run,
            **shared_crop_snapshot,
            "label_schema_id": TARGET_LABEL_SCHEMA,
            "mask_labels": list(TARGET_LABELS),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "method": "fisheye.utils.merge_subject_mask_runs",
            "run_semantics": "merged_subject_components",
            "probabilities_dtype": "float32",
            "probabilities_encoding": "unit_float",
            "created_at_utc": created_at,
            "component_provenance": component_provenance,
            "source_body_subject_mask_run": body_source.run_name,
            "source_eye_subject_mask_run": eye_source.run_name,
        }
    )
    write_subject_mask_component_provenance(
        run_group,
        component_name="subject_body",
        source_stage="subject_mask_runs",
        source_run=body_source.run_name,
        source_method=str(body_source.run_group.attrs.get("method") or "unknown"),
        source_channels=["subject_body"],
        source_crop_run=body_source.crop_run,
        source_crop_snapshot=shared_crop_snapshot,
        source_label_schema_id=normalize_attr(body_source.run_group.attrs.get("label_schema_id")),
        source_created_at_utc=normalize_attr(body_source.run_group.attrs.get("created_at_utc")),
    )
    write_subject_mask_component_provenance(
        run_group,
        component_name="eye_left",
        source_stage="subject_mask_runs",
        source_run=eye_source.run_name,
        source_method=str(eye_source.run_group.attrs.get("method") or "unknown"),
        source_channels=["eye_left"],
        source_crop_run=body_source.crop_run,
        source_crop_snapshot=shared_crop_snapshot,
        source_label_schema_id=normalize_attr(eye_source.run_group.attrs.get("label_schema_id")),
        source_created_at_utc=normalize_attr(eye_source.run_group.attrs.get("created_at_utc")),
    )
    write_subject_mask_component_provenance(
        run_group,
        component_name="eye_right",
        source_stage="subject_mask_runs",
        source_run=eye_source.run_name,
        source_method=str(eye_source.run_group.attrs.get("method") or "unknown"),
        source_channels=["eye_right"],
        source_crop_run=body_source.crop_run,
        source_crop_snapshot=shared_crop_snapshot,
        source_label_schema_id=normalize_attr(eye_source.run_group.attrs.get("label_schema_id")),
        source_created_at_utc=normalize_attr(eye_source.run_group.attrs.get("created_at_utc")),
    )

    if body_source.source_keypoints_run and body_source.source_keypoints_run == eye_source.source_keypoints_run:
        run_group.attrs.update(build_source_keypoints_attrs(body_source.source_keypoints_run, include_legacy_alias=True))
    if body_source.source_keypoint_group and body_source.source_keypoint_group == eye_source.source_keypoint_group:
        run_group.attrs["source_keypoint_group"] = body_source.source_keypoint_group

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})

    copy_row_lineage_arrays(run_group, body_source.run_group, total_rois=total_rows)
    run_group.create_array("detection_source", data=np.asarray(body_source.detection_source[:], dtype=np.int8), overwrite=True)

    storage_chunks = _target_mask_chunks(total_rows, height, width)
    masks_out = run_group.create_array(
        "masks_roi",
        shape=(total_rows, len(TARGET_LABELS), height, width),
        dtype=np.uint8,
        chunks=storage_chunks,
        fill_value=0,
        overwrite=True,
    )
    probs_out = run_group.create_array(
        "mask_probs_roi",
        shape=(total_rows, len(TARGET_LABELS), height, width),
        dtype=np.float32,
        chunks=storage_chunks,
        fill_value=0.0,
        overwrite=True,
    )
    run_group.create_array("available_channels", data=TARGET_AVAILABLE_CHANNELS, overwrite=True)

    metric_chunks = _target_metric_chunks(total_rows)
    metrics_group = run_group.require_group("metrics")
    metric_arrays = {
        "prob_max": metrics_group.create_array(
            "prob_max",
            shape=(total_rows, len(TARGET_LABELS)),
            dtype=np.float32,
            chunks=metric_chunks,
            fill_value=0.0,
            overwrite=True,
        ),
        "mask_present": metrics_group.create_array(
            "mask_present",
            shape=(total_rows, len(TARGET_LABELS)),
            dtype=bool,
            chunks=metric_chunks,
            fill_value=False,
            overwrite=True,
        ),
        "area_px": metrics_group.create_array(
            "area_px",
            shape=(total_rows, len(TARGET_LABELS)),
            dtype=np.float32,
            chunks=metric_chunks,
            fill_value=0.0,
            overwrite=True,
        ),
        "centroid_xy": metrics_group.create_array(
            "centroid_xy",
            shape=(total_rows, len(TARGET_LABELS), 2),
            dtype=np.float32,
            chunks=(metric_chunks[0], 1, 2),
            fill_value=0.0,
            overwrite=True,
        ),
        "centroid_valid": metrics_group.create_array(
            "centroid_valid",
            shape=(total_rows, len(TARGET_LABELS)),
            dtype=bool,
            chunks=metric_chunks,
            fill_value=False,
            overwrite=True,
        ),
        "bbox_xyxy": metrics_group.create_array(
            "bbox_xyxy",
            shape=(total_rows, len(TARGET_LABELS), 4),
            dtype=np.float32,
            chunks=(metric_chunks[0], 1, 4),
            fill_value=0.0,
            overwrite=True,
        ),
        "bbox_valid": metrics_group.create_array(
            "bbox_valid",
            shape=(total_rows, len(TARGET_LABELS)),
            dtype=bool,
            chunks=metric_chunks,
            fill_value=False,
            overwrite=True,
        ),
    }

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(total_rows, start_idx + batch_size)
        row_slice = slice(start_idx, end_idx)
        merged_masks = np.zeros((end_idx - start_idx, len(TARGET_LABELS), height, width), dtype=np.uint8)
        merged_probs = np.zeros((end_idx - start_idx, len(TARGET_LABELS), height, width), dtype=np.float32)

        merged_masks[:, 0:1] = np.asarray(body_source.masks_roi[row_slice, body_idx : body_idx + 1], dtype=np.uint8)
        merged_masks[:, 1:2] = np.asarray(eye_source.masks_roi[row_slice, eye_left_idx : eye_left_idx + 1], dtype=np.uint8)
        merged_masks[:, 2:3] = np.asarray(eye_source.masks_roi[row_slice, eye_right_idx : eye_right_idx + 1], dtype=np.uint8)

        merged_probs[:, 0:1] = _semantic_probabilities(body_source, row_slice, body_idx)
        merged_probs[:, 1:2] = _semantic_probabilities(eye_source, row_slice, eye_left_idx)
        merged_probs[:, 2:3] = _semantic_probabilities(eye_source, row_slice, eye_right_idx)

        masks_out[row_slice] = merged_masks
        probs_out[row_slice] = merged_probs

        metrics = _compute_metrics(merged_masks, merged_probs)
        for name, arr in metric_arrays.items():
            arr[row_slice] = metrics[name]

    duration = float(time.perf_counter() - start)
    mask_present_all = np.asarray(metric_arrays["mask_present"][:], dtype=bool)
    area_all = np.asarray(metric_arrays["area_px"][:], dtype=np.float32)
    run_group.attrs["duration_seconds"] = duration
    run_group.attrs["summary_statistics"] = {
        "rows_total": int(total_rows),
        "rows_with_nonempty_masks": int(np.sum(mask_present_all.any(axis=1))),
        "rows_with_subject_body_masks": int(np.sum(mask_present_all[:, 0])),
        "rows_with_eye_left_masks": int(np.sum(mask_present_all[:, 1])),
        "rows_with_eye_right_masks": int(np.sum(mask_present_all[:, 2])),
        "rows_with_swim_bladder_masks": int(np.sum(mask_present_all[:, 3])),
        "area_px_subject_body_mean": float(area_all[:, 0].mean()) if total_rows else 0.0,
        "area_px_eye_left_mean": float(area_all[:, 1].mean()) if total_rows else 0.0,
        "area_px_eye_right_mean": float(area_all[:, 2].mean()) if total_rows else 0.0,
        "available_channels": [bool(v) for v in TARGET_AVAILABLE_CHANNELS.tolist()],
        "label_schema_id": TARGET_LABEL_SCHEMA,
        "body_run": body_source.run_name,
        "eye_run": eye_source.run_name,
        "crop_run": body_source.crop_run,
        "created_at_utc": created_at,
        "output_run": run_name,
    }
    provenance_inputs = {
        **shared_crop_inputs,
        "source_body_subject_mask_run": body_source.run_name,
        "source_eye_subject_mask_run": eye_source.run_name,
    }
    if body_source.source_keypoints_run and body_source.source_keypoints_run == eye_source.source_keypoints_run:
        provenance_inputs["source_keypoints_run"] = body_source.source_keypoints_run
    if body_source.source_keypoint_group and body_source.source_keypoint_group == eye_source.source_keypoint_group:
        provenance_inputs["source_keypoint_group"] = body_source.source_keypoint_group
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
            "method": "fisheye.utils.merge_subject_mask_runs",
            "run_semantics": "merged_subject_components",
            "body_run": body_source.run_name,
            "eye_run": eye_source.run_name,
            "batch_size": int(batch_size),
            "overwrite": bool(overwrite),
        },
        inputs=provenance_inputs,
        artifacts={
            "body_probability_source_path": body_source.probability_source_path,
            "eye_probability_source_path": eye_source.probability_source_path,
        },
    )
    write_stage_provenance(run_group, provenance)
    summary["duration_seconds"] = duration
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to a training zarr containing subject_mask_runs.")
    parser.add_argument("--body-run", required=True, help="Body source subject_mask_runs/<run>.")
    parser.add_argument("--eye-run", required=True, help="Eye source subject_mask_runs/<run>.")
    parser.add_argument("--run-name", required=True, help="Merged output subject_mask_runs/<run>.")
    parser.add_argument("--batch-size", type=int, default=256, help="Rows per write batch (default: 256).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target run when --apply is set.")
    parser.add_argument("--apply", action="store_true", help="Write the merged run. Default is dry-run.")
    args = parser.parse_args(argv)

    result = merge_subject_mask_runs(
        args.zarr_path,
        body_run=str(args.body_run),
        eye_run=str(args.eye_run),
        run_name=str(args.run_name),
        batch_size=int(args.batch_size),
        overwrite=bool(args.overwrite),
        apply=bool(args.apply),
    )
    print(
        f"{result['status'].upper()} {args.zarr_path} "
        f"body=subject_mask_runs/{result['body_run']} "
        f"eyes=subject_mask_runs/{result['eye_run']} "
        f"target=subject_mask_runs/{result['target_run']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
