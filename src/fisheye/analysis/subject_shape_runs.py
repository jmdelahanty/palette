"""Build analysis/subject_shape_runs from refined subject masks."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import dask
from dask import delayed
import numpy as np
from skimage.measure import find_contours, label as label_components
from skimage.morphology import skeletonize
import zarr

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from ..refinement.refine_eye_masks import _measure_mask
from ..shared.row_lineage import copy_row_lineage_arrays
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..tune.refined_subject_mask_review import _compute_geometry_metrics, _compute_mask_metrics
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root

SUBJECT_SHAPE_SCHEMA_ID = "analysis.subject_shape_runs"
SUBJECT_SHAPE_SCHEMA_VERSION = 1
SUBJECT_SHAPE_METHOD = "subject_shape_from_refined_masks_v2"
SUBJECT_SHAPE_METHOD_VERSION = 2
SUBJECT_SHAPE_STAGE_NAME = "analysis.subject_shape_runs"
COMPONENT_ORDER = ("subject_body", "swim_bladder", "eye_left", "eye_right")
ELLIPSE_COMPONENTS = ("swim_bladder", "eye_left", "eye_right")
EYE_COMPONENTS = ("eye_left", "eye_right")
BODY_FRAME_COMPONENTS = ("swim_bladder", "eye_left", "eye_right")
BODY_FRAME_SCHEMA_ID = "fish_anatomical_body_frame"
BODY_FRAME_SCHEMA_VERSION = 1
BODY_FRAME_ESTIMATOR = "mask_component_axis"
TAIL_GEOMETRY_SCHEMA_ID = "analysis.subject_shape.tail_geometry"
TAIL_GEOMETRY_SCHEMA_VERSION = 1
TAIL_ANCHOR_METHOD = "caudal_swim_bladder_contour_min_forward_projection_v1"
CENTERLINE_METHOD = "skeleton_longest_endpoint_path_v1"
CENTERLINE_SAMPLE_COUNT = 64
REASON_BYTES_WIDTH = 64
SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks")
SERIAL_EXECUTION_BACKEND = "serial_driver"
DASK_WORKER_EXECUTION_BACKEND = "dask_worker_chunks"


@dataclass(frozen=True)
class ComponentBatch:
    mask_present: np.ndarray
    area_px: np.ndarray
    centroid_xy: np.ndarray
    centroid_valid: np.ndarray
    bbox_xyxy: np.ndarray
    bbox_valid: np.ndarray
    principal_axis_xy: np.ndarray
    principal_axis_valid: np.ndarray
    principal_axis_length_px: np.ndarray
    secondary_axis_length_px: np.ndarray
    ellipse_params: np.ndarray
    ellipse_success: np.ndarray


@dataclass(frozen=True)
class BodyFrameBatch:
    origin_xy: np.ndarray
    forward_axis_xy: np.ndarray
    left_axis_xy: np.ndarray
    heading_deg: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class CaudalAnchorBatch:
    point_xy: np.ndarray
    projection_px: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class CenterlineBatch:
    centerline_xy: np.ndarray
    centerline_valid: np.ndarray
    centerline_failure_reason_bytes: np.ndarray
    head_endpoint_xy: np.ndarray
    tail_tip_xy: np.ndarray
    tail_base_xy: np.ndarray
    tail_base_valid: np.ndarray
    tail_base_arclength_px: np.ndarray
    tail_base_failure_reason_bytes: np.ndarray
    tail_segment_arclength_px: np.ndarray
    body_arclength_px: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"subject_shape_{stamp}"


def _normalize_scheduler(value: str) -> str:
    scheduler = str(value).strip().lower().replace("_", "-")
    aliases = {
        "single": "single-threaded",
        "single_threaded": "single-threaded",
        "thread": "threads",
        "process": "processes",
        "local-cluster": "distributed",
        "local_cluster": "distributed",
    }
    scheduler = aliases.get(scheduler, scheduler)
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise argparse.ArgumentTypeError(
            f"scheduler must be one of {', '.join(SUPPORTED_SCHEDULERS)}; got {value!r}."
        )
    return scheduler


def _normalize_execution_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("-", "_")
    aliases = {
        "serial": SERIAL_EXECUTION_BACKEND,
        "driver": SERIAL_EXECUTION_BACKEND,
        "dask": DASK_WORKER_EXECUTION_BACKEND,
        "dask_chunks": DASK_WORKER_EXECUTION_BACKEND,
    }
    backend = aliases.get(backend, backend)
    if backend not in EXECUTION_BACKENDS:
        raise ValueError(f"execution_backend must be one of {EXECUTION_BACKENDS}; got {value!r}.")
    return backend


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    chunk = max(1, int(chunk_size))
    return [(start, min(total, start + chunk)) for start in range(0, total, chunk)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _encode_reason(reason: object, *, width: int = REASON_BYTES_WIDTH) -> np.ndarray:
    data = str(reason or "").encode("utf-8", errors="replace")[: max(0, int(width) - 1)]
    out = np.zeros((int(width),), dtype=np.uint8)
    out[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    return out


def _encode_reasons(reasons: Sequence[object], *, width: int = REASON_BYTES_WIDTH) -> np.ndarray:
    out = np.zeros((len(reasons), int(width)), dtype=np.uint8)
    for idx, reason in enumerate(reasons):
        out[int(idx), :] = _encode_reason(reason, width=width)
    return out


def _set_reason_bytes_attrs(group: zarr.Group, *, width: int = REASON_BYTES_WIDTH) -> None:
    group.attrs["reason_encoding"] = "utf8-null-terminated"
    group.attrs["reason_bytes_width"] = int(width)
    group.attrs["reason_bytes_null_terminated"] = True


def _label_index_map(refined_group: zarr.Group) -> dict[str, int]:
    labels = refined_group.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)):
        raise ValueError("refined subject-mask run is missing mask_labels attrs.")
    return {str(label): int(idx) for idx, label in enumerate(labels)}


def _component_available(refined_group: zarr.Group, component_idx: int) -> bool:
    available = refined_group.get("available_channels")
    if available is None:
        return True
    values = np.asarray(available[:], dtype=bool).reshape(-1)
    return int(component_idx) < int(values.shape[0]) and bool(values[int(component_idx)])


def _resolve_refined_run(root: zarr.Group, refined_run: Optional[str]) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    if refined_run:
        run_name = str(refined_run)
    else:
        latest = parent.attrs.get("latest")
        if latest is None:
            keys = sorted(str(key) for key in parent.keys())
            if not keys:
                raise ValueError("refined_subject_masks_runs has no runs.")
            run_name = keys[-1]
        else:
            run_name = str(latest)
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _resolve_components(refined_group: zarr.Group, components: Optional[Sequence[str]]) -> tuple[tuple[str, int], ...]:
    label_map = _label_index_map(refined_group)
    requested = [str(value) for value in components] if components else [name for name in COMPONENT_ORDER if name in label_map]
    resolved: list[tuple[str, int]] = []
    seen: set[str] = set()
    for component in requested:
        name = str(component)
        if name in seen:
            continue
        if name not in label_map:
            raise ValueError(f"Component {name!r} not present in refined run mask_labels.")
        idx = int(label_map[name])
        if not _component_available(refined_group, idx):
            continue
        resolved.append((name, idx))
        seen.add(name)
    if not resolved:
        raise ValueError("No available refined subject-mask components selected for subject-shape analysis.")
    return tuple(resolved)


def _create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> None:
    if name in group:
        del group[name]
    group.create_array(
        name,
        shape=tuple(int(dim) for dim in shape),
        dtype=dtype,
        chunks=tuple(int(dim) for dim in chunks),
        fill_value=fill_value,
        overwrite=True,
    )


def _metric_chunks(total_rows: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(width))


def _metric_chunks_3d(total_rows: int, middle: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(middle), int(width))


def _component_review_states(refined_group: zarr.Group, components: Sequence[str]) -> dict[str, str]:
    statuses = refined_group.attrs.get("component_review_statuses")
    if not isinstance(statuses, Mapping):
        return {str(component): "unknown" for component in components}
    result: dict[str, str] = {}
    for component in components:
        payload = statuses.get(str(component))
        if isinstance(payload, Mapping):
            state = payload.get("state") or payload.get("review_state")
            result[str(component)] = str(state or "unknown")
        else:
            result[str(component)] = "unknown"
    return result


def _prepare_component_group(run_group: zarr.Group, component_name: str, *, total_rows: int) -> None:
    chunks_1d = _metric_chunks(total_rows)
    component_group = run_group.require_group("components").require_group(component_name)
    component_group.attrs["component_name"] = component_name
    component_group.attrs["source_component"] = component_name
    component_group.attrs["component_schema_id"] = "analysis.subject_shape_component_v1"
    _create_array(component_group, "mask_present", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(component_group, "area_px", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
    _create_array(
        component_group,
        "centroid_xy",
        shape=(total_rows, 2),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 2),
        fill_value=np.nan,
    )
    _create_array(component_group, "centroid_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(
        component_group,
        "bbox_xyxy",
        shape=(total_rows, 4),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 4),
        fill_value=np.nan,
    )
    _create_array(component_group, "bbox_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)

    if component_name == "subject_body":
        component_group.attrs["principal_axis_method"] = "pca_mask_pixels_v1"
        component_group.attrs["principal_axis_semantics"] = "unoriented_principal_axis_in_roi_xy"
        component_group.attrs["centerline_method"] = CENTERLINE_METHOD
        component_group.attrs["centerline_sample_count"] = CENTERLINE_SAMPLE_COUNT
        component_group.attrs["tail_tip_semantic_label"] = "tail_tip"
        component_group.attrs["tail_tip_estimator"] = "subject_body_centerline_posterior_endpoint"
        component_group.attrs["tail_base_definition"] = (
            "body_centerline_projection_of_caudal_swim_bladder_contour_point"
        )
        _set_reason_bytes_attrs(component_group)
        _create_array(
            component_group,
            "principal_axis_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "principal_axis_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "principal_axis_length_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "secondary_axis_length_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "centerline_xy",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "centerline_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "centerline_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        for name in ("head_endpoint_xy", "tail_tip_xy", "tail_base_xy"):
            _create_array(
                component_group,
                name,
                shape=(total_rows, 2),
                dtype=np.float32,
                chunks=_metric_chunks_lastdim(total_rows, 2),
                fill_value=np.nan,
            )
        _create_array(component_group, "tail_base_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "tail_base_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "tail_base_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "tail_segment_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "body_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )

    if component_name in ELLIPSE_COMPONENTS:
        component_group.attrs["ellipse_method"] = "cv2.fitEllipse_component_contour_v1"
        _create_array(
            component_group,
            "ellipse_params",
            shape=(total_rows, 5),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 5),
            fill_value=np.nan,
        )
        _create_array(component_group, "ellipse_success", shape=(total_rows,), dtype=bool, chunks=chunks_1d)

    if component_name == "swim_bladder":
        component_group.attrs["caudal_anchor_method"] = TAIL_ANCHOR_METHOD
        component_group.attrs["caudal_anchor_definition"] = "min_projection_on_body_forward_axis"
        _set_reason_bytes_attrs(component_group)
        _create_array(
            component_group,
            "caudal_contour_point_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "caudal_contour_projection_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(component_group, "caudal_contour_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "caudal_contour_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )


def _prepare_body_frame_group(run_group: zarr.Group, *, total_rows: int) -> None:
    chunks_1d = _metric_chunks(total_rows)
    group = run_group.require_group("body_frame")
    _set_reason_bytes_attrs(group)
    group.attrs.update(
        {
            "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
            "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
            "body_frame_estimator": BODY_FRAME_ESTIMATOR,
            "body_frame_coordinate_space": "roi_pixels",
            "body_frame_angle_convention": "math_ccw_degrees_after_y_flip",
            "origin_definition": "midpoint_eye_left_eye_right",
            "forward_axis_definition": "swim_bladder_centroid_to_eye_pair_midpoint",
            "left_axis_definition": "eye_right_to_eye_left_projected_perpendicular_to_forward",
        }
    )
    for name in ("origin_xy", "forward_axis_xy", "left_axis_xy"):
        _create_array(
            group,
            name,
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
    _create_array(group, "heading_deg", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
    _create_array(group, "valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(
        group,
        "failure_reason_bytes",
        shape=(total_rows, REASON_BYTES_WIDTH),
        dtype=np.uint8,
        chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
    )


def _prepare_relation_groups(run_group: zarr.Group, components: Sequence[str], *, total_rows: int) -> tuple[str, ...]:
    chunks_1d = _metric_chunks(total_rows)
    relation_names: list[str] = []
    relations = run_group.require_group("relations")
    component_set = set(components)
    if set(EYE_COMPONENTS).issubset(component_set):
        relation_names.append("eye_pair")
        group = relations.require_group("eye_pair")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.eye_pair_v1"
        group.attrs["relation_components"] = list(EYE_COMPONENTS)
        _create_array(group, "separation_px", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
        _create_array(group, "separation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            group,
            "midpoint_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(group, "midpoint_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    if {"subject_body", "swim_bladder"}.issubset(component_set):
        relation_names.append("swim_bladder_to_body")
        group = relations.require_group("swim_bladder_to_body")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.swim_bladder_to_body_v1"
        group.attrs["relation_components"] = ["swim_bladder", "subject_body"]
        group.attrs["axis_semantics"] = "unoriented_body_principal_axis"
        _create_array(group, "relation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            group,
            "distance_to_body_centroid_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            group,
            "longitudinal_offset_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            group,
            "lateral_offset_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
    if {"subject_body", *EYE_COMPONENTS}.issubset(component_set):
        relation_names.append("eyes_to_body")
        group = relations.require_group("eyes_to_body")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.eyes_to_body_v1"
        group.attrs["relation_components"] = ["eye_left", "eye_right", "subject_body"]
        group.attrs["angle_semantics"] = "unoriented_eye_major_axis_relative_to_unoriented_body_principal_axis"
        for eye in EYE_COMPONENTS:
            prefix = "left" if eye == "eye_left" else "right"
            _create_array(group, f"{prefix}_eye_relation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
            _create_array(
                group,
                f"{prefix}_eye_offset_xy",
                shape=(total_rows, 2),
                dtype=np.float32,
                chunks=_metric_chunks_lastdim(total_rows, 2),
                fill_value=np.nan,
            )
            _create_array(
                group,
                f"{prefix}_eye_distance_to_body_centroid_px",
                shape=(total_rows,),
                dtype=np.float32,
                chunks=chunks_1d,
                fill_value=np.nan,
            )
            _create_array(
                group,
                f"{prefix}_eye_axis_angle_to_body_rad",
                shape=(total_rows,),
                dtype=np.float32,
                chunks=chunks_1d,
                fill_value=np.nan,
            )
    return tuple(relation_names)


def _prepare_subject_shape_run(
    root: zarr.Group,
    *,
    target_run: str,
    refined_run_name: str,
    refined_group: zarr.Group,
    component_indices: Sequence[tuple[str, int]],
    chunk_size: int,
    execution_backend: str,
    scheduler: str,
    num_workers: Optional[int],
    stage_command: str,
    overwrite: bool,
) -> zarr.Group:
    if "masks_roi" not in refined_group:
        raise ValueError(f"refined_subject_masks_runs/{refined_run_name} missing masks_roi.")
    masks = refined_group["masks_roi"]
    total_rows = int(masks.shape[0])
    components = tuple(name for name, _idx in component_indices)
    analysis_group = root.require_group("analysis")
    parent = analysis_group.require_group("subject_shape_runs")
    if target_run in parent:
        if not overwrite:
            raise ValueError(
                f"analysis/subject_shape_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del parent[target_run]
    run_group = parent.create_group(target_run)

    row_index = run_group.require_group("row_index")
    copy_result = copy_row_lineage_arrays(
        row_index,
        refined_group,
        names=("frame_indices", "detection_indices", "source_refined_row_ids", "source_detect_row_index"),
        total_rois=total_rows,
        overwrite=True,
    )
    run_group.attrs["row_lineage_copied"] = list(copy_result.copied)
    run_group.attrs["row_lineage_missing"] = list(copy_result.missing)

    for component_name in components:
        _prepare_component_group(run_group, component_name, total_rows=total_rows)
    relation_names = _prepare_relation_groups(run_group, components, total_rows=total_rows)
    if set(BODY_FRAME_COMPONENTS).issubset(set(components)):
        _prepare_body_frame_group(run_group, total_rows=total_rows)

    created = _utc_now()
    dask_metadata = {
        "execution_backend": execution_backend,
        "dask_execution_enabled": execution_backend == DASK_WORKER_EXECUTION_BACKEND,
        "dask_scheduler": scheduler,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_chunk_size": max(1, int(chunk_size)),
        "dask_version": getattr(dask, "__version__", "unknown"),
    }
    source_labels = list(refined_group.attrs.get("mask_labels") or [])
    source_refs = {
        "refined_subject_masks": f"refined_subject_masks_runs/{refined_run_name}",
        "refined_subject_masks_masks_roi": f"refined_subject_masks_runs/{refined_run_name}/masks_roi",
    }
    if "source_subject_mask_run" in refined_group.attrs:
        source_refs["source_subject_mask_run"] = str(refined_group.attrs["source_subject_mask_run"])

    run_group.attrs.update(
        {
            "schema_id": SUBJECT_SHAPE_SCHEMA_ID,
            "schema_version": SUBJECT_SHAPE_SCHEMA_VERSION,
            "method": SUBJECT_SHAPE_METHOD,
            "method_version": SUBJECT_SHAPE_METHOD_VERSION,
            "created_at_utc": created,
            "created_utc": created,
            "row_axis": "refined_subject_mask_rows",
            "source_refined_subject_masks_run": refined_run_name,
            "source_refined_subject_masks_stage": "refined_subject_masks_runs",
            "source_mask_labels": source_labels,
            "source_mask_label_schema_id": refined_group.attrs.get("label_schema_id"),
            "source_mask_geometry_schema_id": refined_group.attrs.get("component_metrics_schema_id"),
            "source_component_review_states": _component_review_states(refined_group, components),
            "source_refs": source_refs,
            "component_names": list(components),
            "relation_names": list(relation_names),
            "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
            "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
            "body_frame_estimator": (
                BODY_FRAME_ESTIMATOR if set(BODY_FRAME_COMPONENTS).issubset(set(components)) else None
            ),
            "body_frame_source_refs": {
                "refined_subject_masks_run": f"refined_subject_masks_runs/{refined_run_name}",
                "swim_bladder_component": "refined_subject_masks_runs/{}/components/swim_bladder".format(
                    refined_run_name
                ),
                "eye_left_component": f"refined_subject_masks_runs/{refined_run_name}/components/eye_left",
                "eye_right_component": f"refined_subject_masks_runs/{refined_run_name}/components/eye_right",
            }
            if set(BODY_FRAME_COMPONENTS).issubset(set(components))
            else None,
            "tail_geometry_schema_id": TAIL_GEOMETRY_SCHEMA_ID,
            "tail_geometry_schema_version": TAIL_GEOMETRY_SCHEMA_VERSION,
            "tail_anchor_method": TAIL_ANCHOR_METHOD,
            "centerline_method": CENTERLINE_METHOD,
            "centerline_sample_count": CENTERLINE_SAMPLE_COUNT,
            "chunk_size": max(1, int(chunk_size)),
            "chunk_count": len(_row_chunks(total_rows, max(1, int(chunk_size)))),
            **dask_metadata,
        }
    )

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage=SUBJECT_SHAPE_STAGE_NAME,
        command=stage_command,
        created_at_utc=created,
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
        scheduler=dask_metadata,
        parameters={
            "method": SUBJECT_SHAPE_METHOD,
            "components": list(components),
            "relations": list(relation_names),
            "body_frame_estimator": (
                BODY_FRAME_ESTIMATOR if set(BODY_FRAME_COMPONENTS).issubset(set(components)) else None
            ),
            "tail_anchor_method": TAIL_ANCHOR_METHOD,
            "centerline_method": CENTERLINE_METHOD,
            "centerline_sample_count": CENTERLINE_SAMPLE_COUNT,
            "chunk_size": max(1, int(chunk_size)),
        },
        inputs={
            "source_refined_subject_masks_run": refined_run_name,
            "source_refined_subject_masks_stage": "refined_subject_masks_runs",
            "source_refs": source_refs,
        },
    )
    write_stage_provenance(run_group, provenance)
    return run_group


def _compute_principal_axis_metrics(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks_bool = np.asarray(masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    axis_valid = np.zeros((row_count,), dtype=bool)
    major_length = np.full((row_count,), np.nan, dtype=np.float32)
    minor_length = np.full((row_count,), np.nan, dtype=np.float32)
    for row_idx in range(row_count):
        ys, xs = np.nonzero(masks_bool[row_idx])
        if int(xs.size) < 2:
            continue
        coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        centered = coords - coords.mean(axis=0, keepdims=True)
        try:
            cov = centered.T @ centered / max(1, int(coords.shape[0]) - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigvals)
        major_vec = np.asarray(eigvecs[:, int(order[-1])], dtype=np.float64)
        if not np.all(np.isfinite(major_vec)):
            continue
        norm = float(np.linalg.norm(major_vec))
        if norm <= 0.0:
            continue
        major_vec = major_vec / norm
        if major_vec[0] < 0.0 or (major_vec[0] == 0.0 and major_vec[1] < 0.0):
            major_vec *= -1.0
        minor_vec = np.asarray([-major_vec[1], major_vec[0]], dtype=np.float64)
        major_projection = centered @ major_vec
        minor_projection = centered @ minor_vec
        axis_xy[row_idx] = major_vec.astype(np.float32)
        axis_valid[row_idx] = True
        major_length[row_idx] = np.float32(float(major_projection.max() - major_projection.min() + 1.0))
        minor_length[row_idx] = np.float32(float(minor_projection.max() - minor_projection.min() + 1.0))
    return axis_xy, axis_valid, major_length, minor_length


def _compute_ellipse_metrics(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    row_count = int(masks_u8.shape[0])
    ellipse_params = np.full((row_count, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((row_count,), dtype=bool)
    for row_idx in range(row_count):
        success, ellipse, _centroid, _contour, _reason = _measure_mask(masks_u8[row_idx])
        ellipse_params[row_idx] = np.asarray(ellipse, dtype=np.float32)
        ellipse_success[row_idx] = bool(success)
    return ellipse_params, ellipse_success


def _compute_component_batch(masks: np.ndarray, component_name: str) -> ComponentBatch:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    mask_present, area_px = _compute_mask_metrics(masks_u8[:, None, :, :])
    geometry = _compute_geometry_metrics(masks_u8[:, None, :, :])
    if component_name == "subject_body":
        axis_xy, axis_valid, major_length, minor_length = _compute_principal_axis_metrics(masks_u8)
    else:
        row_count = int(masks_u8.shape[0])
        axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
        axis_valid = np.zeros((row_count,), dtype=bool)
        major_length = np.full((row_count,), np.nan, dtype=np.float32)
        minor_length = np.full((row_count,), np.nan, dtype=np.float32)
    if component_name in ELLIPSE_COMPONENTS:
        ellipse_params, ellipse_success = _compute_ellipse_metrics(masks_u8)
    else:
        row_count = int(masks_u8.shape[0])
        ellipse_params = np.full((row_count, 5), np.nan, dtype=np.float32)
        ellipse_success = np.zeros((row_count,), dtype=bool)
    return ComponentBatch(
        mask_present=np.asarray(mask_present[:, 0], dtype=bool),
        area_px=np.asarray(area_px[:, 0], dtype=np.float32),
        centroid_xy=np.asarray(geometry["centroid_xy"][:, 0, :], dtype=np.float32),
        centroid_valid=np.asarray(geometry["centroid_valid"][:, 0], dtype=bool),
        bbox_xyxy=np.asarray(geometry["bbox_xyxy"][:, 0, :], dtype=np.float32),
        bbox_valid=np.asarray(geometry["bbox_valid"][:, 0], dtype=bool),
        principal_axis_xy=axis_xy,
        principal_axis_valid=axis_valid,
        principal_axis_length_px=major_length,
        secondary_axis_length_px=minor_length,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
    )


def _write_component_batch(
    run_group: zarr.Group,
    component_name: str,
    row_slice: slice,
    batch: ComponentBatch,
) -> None:
    group = run_group["components"][component_name]
    group["mask_present"][row_slice] = batch.mask_present
    group["area_px"][row_slice] = batch.area_px
    group["centroid_xy"][row_slice, :] = batch.centroid_xy
    group["centroid_valid"][row_slice] = batch.centroid_valid
    group["bbox_xyxy"][row_slice, :] = batch.bbox_xyxy
    group["bbox_valid"][row_slice] = batch.bbox_valid
    if component_name == "subject_body":
        group["principal_axis_xy"][row_slice, :] = batch.principal_axis_xy
        group["principal_axis_valid"][row_slice] = batch.principal_axis_valid
        group["principal_axis_length_px"][row_slice] = batch.principal_axis_length_px
        group["secondary_axis_length_px"][row_slice] = batch.secondary_axis_length_px
    if component_name in ELLIPSE_COMPONENTS:
        group["ellipse_params"][row_slice, :] = batch.ellipse_params
        group["ellipse_success"][row_slice] = batch.ellipse_success


def _unit_vector_xy(vector: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(vector, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


def _compute_body_frame_batch(batches: Mapping[str, ComponentBatch]) -> BodyFrameBatch:
    any_batch = next(iter(batches.values()))
    row_count = int(any_batch.mask_present.shape[0])
    origin_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    forward_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    left_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    heading_deg = np.full((row_count,), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_source_component"] * row_count

    if not set(BODY_FRAME_COMPONENTS).issubset(batches):
        return BodyFrameBatch(
            origin_xy=origin_xy,
            forward_axis_xy=forward_axis_xy,
            left_axis_xy=left_axis_xy,
            heading_deg=heading_deg,
            valid=valid,
            failure_reason_bytes=_encode_reasons(reasons),
        )

    swim = batches["swim_bladder"]
    left = batches["eye_left"]
    right = batches["eye_right"]
    for row_idx in range(row_count):
        if not (swim.centroid_valid[row_idx] and left.centroid_valid[row_idx] and right.centroid_valid[row_idx]):
            reasons[row_idx] = "missing_source_anchor"
            continue
        swim_xy = swim.centroid_xy[row_idx].astype(np.float64)
        left_xy = left.centroid_xy[row_idx].astype(np.float64)
        right_xy = right.centroid_xy[row_idx].astype(np.float64)
        eye_midpoint = (left_xy + right_xy) * 0.5
        forward = _unit_vector_xy(eye_midpoint - swim_xy)
        if forward is None:
            reasons[row_idx] = "degenerate_forward_axis"
            continue
        left_candidate = left_xy - right_xy
        left_candidate = left_candidate - float(np.dot(left_candidate, forward)) * forward
        left_axis = _unit_vector_xy(left_candidate)
        if left_axis is None:
            reasons[row_idx] = "left_right_unresolved"
            continue
        origin_xy[row_idx] = eye_midpoint.astype(np.float32)
        forward_axis_xy[row_idx] = forward.astype(np.float32)
        left_axis_xy[row_idx] = left_axis.astype(np.float32)
        heading_deg[row_idx] = np.float32(math.degrees(math.atan2(-float(forward[1]), float(forward[0]))))
        valid[row_idx] = True
        reasons[row_idx] = "ok"
    return BodyFrameBatch(
        origin_xy=origin_xy,
        forward_axis_xy=forward_axis_xy,
        left_axis_xy=left_axis_xy,
        heading_deg=heading_deg,
        valid=valid,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _write_body_frame_batch(run_group: zarr.Group, row_slice: slice, batch: BodyFrameBatch) -> None:
    if "body_frame" not in run_group:
        return
    group = run_group["body_frame"]
    group["origin_xy"][row_slice, :] = batch.origin_xy
    group["forward_axis_xy"][row_slice, :] = batch.forward_axis_xy
    group["left_axis_xy"][row_slice, :] = batch.left_axis_xy
    group["heading_deg"][row_slice] = batch.heading_deg
    group["valid"][row_slice] = batch.valid
    group["failure_reason_bytes"][row_slice, :] = batch.failure_reason_bytes


def _single_component_count(mask: np.ndarray) -> int:
    labels = label_components(np.asarray(mask, dtype=bool), connectivity=2)
    return int(labels.max()) if labels.size else 0


def _contour_points_xy(mask: np.ndarray) -> Optional[np.ndarray]:
    contours = find_contours(np.asarray(mask, dtype=bool).astype(np.float32), level=0.5)
    if not contours:
        return None
    points: list[np.ndarray] = []
    for contour in contours:
        if int(contour.shape[0]) < 2:
            continue
        xy = np.stack([contour[:, 1], contour[:, 0]], axis=1).astype(np.float64, copy=False)
        points.append(xy)
    if not points:
        return None
    return np.concatenate(points, axis=0)


def _compute_caudal_anchor_batch(
    swim_masks: np.ndarray,
    body_frame: BodyFrameBatch,
) -> CaudalAnchorBatch:
    masks_bool = np.asarray(swim_masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    point_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    projection_px = np.full((row_count,), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_swim_bladder_mask"] * row_count
    for row_idx in range(row_count):
        if not bool(body_frame.valid[row_idx]):
            reasons[row_idx] = "missing_body_frame"
            continue
        mask = masks_bool[row_idx]
        if int(np.count_nonzero(mask)) == 0:
            reasons[row_idx] = "missing_swim_bladder_mask"
            continue
        if _single_component_count(mask) != 1:
            reasons[row_idx] = "fragmented_swim_bladder_mask"
            continue
        contour_xy = _contour_points_xy(mask)
        if contour_xy is None or int(contour_xy.shape[0]) == 0:
            reasons[row_idx] = "empty_swim_bladder_contour"
            continue
        origin = body_frame.origin_xy[row_idx].astype(np.float64)
        forward = body_frame.forward_axis_xy[row_idx].astype(np.float64)
        projections = (contour_xy - origin[None, :]) @ forward
        if not np.any(np.isfinite(projections)):
            reasons[row_idx] = "caudal_projection_failed"
            continue
        caudal_idx = int(np.nanargmin(projections))
        point_xy[row_idx] = contour_xy[caudal_idx].astype(np.float32)
        projection_px[row_idx] = np.float32(float(projections[caudal_idx]))
        valid[row_idx] = True
        reasons[row_idx] = "ok"
    return CaudalAnchorBatch(
        point_xy=point_xy,
        projection_px=projection_px,
        valid=valid,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _write_caudal_anchor_batch(run_group: zarr.Group, row_slice: slice, batch: CaudalAnchorBatch) -> None:
    components = run_group.get("components")
    if components is None or "swim_bladder" not in components:
        return
    group = components["swim_bladder"]
    if "caudal_contour_point_xy" not in group:
        return
    group["caudal_contour_point_xy"][row_slice, :] = batch.point_xy
    group["caudal_contour_projection_px"][row_slice] = batch.projection_px
    group["caudal_contour_valid"][row_slice] = batch.valid
    group["caudal_contour_failure_reason_bytes"][row_slice, :] = batch.failure_reason_bytes


def _skeleton_neighbors(coords_yx: np.ndarray) -> list[list[tuple[int, float]]]:
    index_by_coord = {(int(y), int(x)): int(idx) for idx, (y, x) in enumerate(coords_yx)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(int(coords_yx.shape[0]))]
    for idx, (y, x) in enumerate(coords_yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = index_by_coord.get((int(y) + dy, int(x) + dx))
                if other is None:
                    continue
                weight = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
                neighbors[int(idx)].append((int(other), float(weight)))
    return neighbors


def _dijkstra_path_tree(
    neighbors: Sequence[Sequence[tuple[int, float]]],
    start_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(neighbors)
    dist = np.full((node_count,), np.inf, dtype=np.float64)
    prev = np.full((node_count,), -1, dtype=np.int64)
    dist[int(start_idx)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > float(dist[current]):
            continue
        for other, weight in neighbors[current]:
            next_dist = current_dist + float(weight)
            if next_dist < float(dist[other]):
                dist[other] = next_dist
                prev[other] = current
                heapq.heappush(heap, (next_dist, int(other)))
    return dist, prev


def _farthest_endpoint(dist: np.ndarray, endpoints: Sequence[int]) -> Optional[int]:
    best_idx: Optional[int] = None
    best_dist = -np.inf
    for endpoint in endpoints:
        value = float(dist[int(endpoint)])
        if np.isfinite(value) and value > best_dist:
            best_dist = value
            best_idx = int(endpoint)
    return best_idx


def _reconstruct_path(prev: np.ndarray, start_idx: int, end_idx: int) -> Optional[list[int]]:
    path = [int(end_idx)]
    current = int(end_idx)
    seen = {current}
    while current != int(start_idx):
        current = int(prev[current])
        if current < 0 or current in seen:
            return None
        path.append(current)
        seen.add(current)
    path.reverse()
    return path


def _longest_skeleton_endpoint_path_xy(mask: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    mask_bool = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask_bool)) == 0:
        return None, "missing_subject_body_mask"
    if _single_component_count(mask_bool) != 1:
        return None, "fragmented_subject_body_mask"
    skeleton = skeletonize(mask_bool)
    coords_yx = np.argwhere(skeleton)
    if int(coords_yx.shape[0]) < 2:
        return None, "skeleton_empty"
    neighbors = _skeleton_neighbors(coords_yx)
    endpoints = [idx for idx, items in enumerate(neighbors) if len(items) == 1]
    if len(endpoints) < 2:
        return None, "skeleton_endpoint_ambiguous"
    dist0, _prev0 = _dijkstra_path_tree(neighbors, endpoints[0])
    start = _farthest_endpoint(dist0, endpoints)
    if start is None:
        return None, "centerline_order_failed"
    dist1, prev1 = _dijkstra_path_tree(neighbors, start)
    end = _farthest_endpoint(dist1, endpoints)
    if end is None or end == start:
        return None, "centerline_order_failed"
    path_indices = _reconstruct_path(prev1, start, end)
    if path_indices is None or len(path_indices) < 2:
        return None, "centerline_order_failed"
    path_yx = coords_yx[np.asarray(path_indices, dtype=np.int64)]
    path_xy = np.stack([path_yx[:, 1], path_yx[:, 0]], axis=1).astype(np.float64)
    return path_xy, "ok"


def _polyline_arclength(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if int(points.shape[0]) < 2:
        return np.zeros((int(points.shape[0]),), dtype=np.float64), 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])
    return cumulative, float(cumulative[-1])


def _resample_polyline(points_xy: np.ndarray, sample_count: int) -> tuple[Optional[np.ndarray], float]:
    points = np.asarray(points_xy, dtype=np.float64)
    cumulative, total = _polyline_arclength(points)
    if total <= 1e-6:
        return None, total
    targets = np.linspace(0.0, total, int(sample_count), dtype=np.float64)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1), total


def _project_point_to_polyline(point_xy: np.ndarray, polyline_xy: np.ndarray) -> tuple[Optional[np.ndarray], float]:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    polyline = np.asarray(polyline_xy, dtype=np.float64)
    if int(polyline.shape[0]) < 2:
        return None, float("nan")
    cumulative, _total = _polyline_arclength(polyline)
    best_point: Optional[np.ndarray] = None
    best_arclength = float("nan")
    best_dist = np.inf
    for idx in range(int(polyline.shape[0]) - 1):
        a = polyline[idx]
        b = polyline[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        t = min(1.0, max(0.0, float(np.dot(point - a, ab) / denom)))
        candidate = a + t * ab
        dist = float(np.linalg.norm(point - candidate))
        if dist < best_dist:
            best_dist = dist
            best_point = candidate
            best_arclength = float(cumulative[idx] + t * math.sqrt(denom))
    return best_point, best_arclength


def _compute_centerline_batch(
    body_masks: np.ndarray,
    body_frame: BodyFrameBatch,
    caudal_anchor: CaudalAnchorBatch,
    *,
    sample_count: int = CENTERLINE_SAMPLE_COUNT,
) -> CenterlineBatch:
    masks_bool = np.asarray(body_masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    centerline_xy = np.full((row_count, int(sample_count), 2), np.nan, dtype=np.float32)
    centerline_valid = np.zeros((row_count,), dtype=bool)
    centerline_reasons = ["missing_subject_body_mask"] * row_count
    head_endpoint_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_tip_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_base_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_base_valid = np.zeros((row_count,), dtype=bool)
    tail_base_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)
    tail_base_reasons = ["missing_centerline"] * row_count
    tail_segment_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)
    body_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)

    for row_idx in range(row_count):
        if not bool(body_frame.valid[row_idx]):
            centerline_reasons[row_idx] = "missing_body_frame"
            tail_base_reasons[row_idx] = "missing_body_frame"
            continue
        path_xy, reason = _longest_skeleton_endpoint_path_xy(masks_bool[row_idx])
        if path_xy is None:
            centerline_reasons[row_idx] = reason
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        origin = body_frame.origin_xy[row_idx].astype(np.float64)
        forward = body_frame.forward_axis_xy[row_idx].astype(np.float64)
        first_projection = float(np.dot(path_xy[0] - origin, forward))
        last_projection = float(np.dot(path_xy[-1] - origin, forward))
        if not (np.isfinite(first_projection) and np.isfinite(last_projection)):
            centerline_reasons[row_idx] = "endpoint_orientation_failed"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        if abs(first_projection - last_projection) <= 1e-6:
            centerline_reasons[row_idx] = "ambiguous_polarity"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        if first_projection < last_projection:
            path_xy = path_xy[::-1]
        sampled, total_length = _resample_polyline(path_xy, int(sample_count))
        if sampled is None or total_length <= 1e-6:
            centerline_reasons[row_idx] = "centerline_order_failed"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        centerline_xy[row_idx] = sampled.astype(np.float32)
        centerline_valid[row_idx] = True
        centerline_reasons[row_idx] = "ok"
        head_endpoint_xy[row_idx] = path_xy[0].astype(np.float32)
        tail_tip_xy[row_idx] = path_xy[-1].astype(np.float32)
        body_arclength_px[row_idx] = np.float32(total_length)

        if not bool(caudal_anchor.valid[row_idx]):
            tail_base_reasons[row_idx] = "missing_tail_anchor"
            continue
        projected, arclength = _project_point_to_polyline(caudal_anchor.point_xy[row_idx], path_xy)
        if projected is None or not np.isfinite(arclength):
            tail_base_reasons[row_idx] = "tail_base_projection_failed"
            continue
        tail_length = float(total_length - arclength)
        if tail_length < 0.0:
            tail_base_reasons[row_idx] = "tail_base_projection_failed"
            continue
        tail_base_xy[row_idx] = projected.astype(np.float32)
        tail_base_valid[row_idx] = True
        tail_base_arclength_px[row_idx] = np.float32(arclength)
        tail_segment_arclength_px[row_idx] = np.float32(tail_length)
        tail_base_reasons[row_idx] = "ok"

    return CenterlineBatch(
        centerline_xy=centerline_xy,
        centerline_valid=centerline_valid,
        centerline_failure_reason_bytes=_encode_reasons(centerline_reasons),
        head_endpoint_xy=head_endpoint_xy,
        tail_tip_xy=tail_tip_xy,
        tail_base_xy=tail_base_xy,
        tail_base_valid=tail_base_valid,
        tail_base_arclength_px=tail_base_arclength_px,
        tail_base_failure_reason_bytes=_encode_reasons(tail_base_reasons),
        tail_segment_arclength_px=tail_segment_arclength_px,
        body_arclength_px=body_arclength_px,
    )


def _write_centerline_batch(run_group: zarr.Group, row_slice: slice, batch: CenterlineBatch) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "centerline_xy" not in group:
        return
    group["centerline_xy"][row_slice, :, :] = batch.centerline_xy
    group["centerline_valid"][row_slice] = batch.centerline_valid
    group["centerline_failure_reason_bytes"][row_slice, :] = batch.centerline_failure_reason_bytes
    group["head_endpoint_xy"][row_slice, :] = batch.head_endpoint_xy
    group["tail_tip_xy"][row_slice, :] = batch.tail_tip_xy
    group["tail_base_xy"][row_slice, :] = batch.tail_base_xy
    group["tail_base_valid"][row_slice] = batch.tail_base_valid
    group["tail_base_arclength_px"][row_slice] = batch.tail_base_arclength_px
    group["tail_base_failure_reason_bytes"][row_slice, :] = batch.tail_base_failure_reason_bytes
    group["tail_segment_arclength_px"][row_slice] = batch.tail_segment_arclength_px
    group["body_arclength_px"][row_slice] = batch.body_arclength_px


def _angle_between_unoriented_axes(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    dot = float(np.dot(axis_a, axis_b))
    cross = float(axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0])
    angle = math.atan2(cross, dot)
    if angle > math.pi / 2.0:
        angle -= math.pi
    elif angle < -math.pi / 2.0:
        angle += math.pi
    return float(angle)


def _eye_axis_from_ellipse_params(params: np.ndarray) -> np.ndarray:
    theta = math.radians(float(params[4]))
    return np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)


def _write_relations(run_group: zarr.Group, row_slice: slice, batches: Mapping[str, ComponentBatch]) -> None:
    relations = run_group.get("relations")
    if relations is None:
        return
    if "eye_pair" in relations and set(EYE_COMPONENTS).issubset(batches):
        left = batches["eye_left"]
        right = batches["eye_right"]
        valid = left.centroid_valid & right.centroid_valid
        separation = np.full(valid.shape, np.nan, dtype=np.float32)
        midpoint = np.full((valid.shape[0], 2), np.nan, dtype=np.float32)
        if np.any(valid):
            delta = left.centroid_xy[valid] - right.centroid_xy[valid]
            separation[valid] = np.linalg.norm(delta, axis=1).astype(np.float32, copy=False)
            midpoint[valid] = ((left.centroid_xy[valid] + right.centroid_xy[valid]) * 0.5).astype(np.float32)
        group = relations["eye_pair"]
        group["separation_px"][row_slice] = separation
        group["separation_valid"][row_slice] = valid
        group["midpoint_xy"][row_slice, :] = midpoint
        group["midpoint_valid"][row_slice] = valid

    if "swim_bladder_to_body" in relations and {"subject_body", "swim_bladder"}.issubset(batches):
        body = batches["subject_body"]
        swim = batches["swim_bladder"]
        valid = body.centroid_valid & body.principal_axis_valid & swim.centroid_valid
        distance = np.full(valid.shape, np.nan, dtype=np.float32)
        longitudinal = np.full(valid.shape, np.nan, dtype=np.float32)
        lateral = np.full(valid.shape, np.nan, dtype=np.float32)
        for row_idx in np.flatnonzero(valid):
            axis = body.principal_axis_xy[int(row_idx)].astype(np.float64)
            delta = (swim.centroid_xy[int(row_idx)] - body.centroid_xy[int(row_idx)]).astype(np.float64)
            perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            distance[int(row_idx)] = np.float32(np.linalg.norm(delta))
            longitudinal[int(row_idx)] = np.float32(np.dot(delta, axis))
            lateral[int(row_idx)] = np.float32(np.dot(delta, perp))
        group = relations["swim_bladder_to_body"]
        group["relation_valid"][row_slice] = valid
        group["distance_to_body_centroid_px"][row_slice] = distance
        group["longitudinal_offset_px"][row_slice] = longitudinal
        group["lateral_offset_px"][row_slice] = lateral

    if "eyes_to_body" in relations and {"subject_body", *EYE_COMPONENTS}.issubset(batches):
        body = batches["subject_body"]
        group = relations["eyes_to_body"]
        for eye in EYE_COMPONENTS:
            prefix = "left" if eye == "eye_left" else "right"
            eye_batch = batches[eye]
            valid = body.centroid_valid & body.principal_axis_valid & eye_batch.centroid_valid
            offset = np.full((valid.shape[0], 2), np.nan, dtype=np.float32)
            distance = np.full(valid.shape, np.nan, dtype=np.float32)
            angle = np.full(valid.shape, np.nan, dtype=np.float32)
            if np.any(valid):
                offset[valid] = (eye_batch.centroid_xy[valid] - body.centroid_xy[valid]).astype(np.float32)
                distance[valid] = np.linalg.norm(offset[valid], axis=1).astype(np.float32, copy=False)
            angle_valid = valid & eye_batch.ellipse_success
            for row_idx in np.flatnonzero(angle_valid):
                body_axis = body.principal_axis_xy[int(row_idx)]
                eye_axis = _eye_axis_from_ellipse_params(eye_batch.ellipse_params[int(row_idx)])
                angle[int(row_idx)] = np.float32(_angle_between_unoriented_axes(body_axis, eye_axis))
            group[f"{prefix}_eye_relation_valid"][row_slice] = valid
            group[f"{prefix}_eye_offset_xy"][row_slice, :] = offset
            group[f"{prefix}_eye_distance_to_body_centroid_px"][row_slice] = distance
            group[f"{prefix}_eye_axis_angle_to_body_rad"][row_slice] = angle


def _process_and_write_subject_shape_chunk_groups(
    refined_group: zarr.Group,
    run_group: zarr.Group,
    *,
    component_indices: Sequence[tuple[str, int]],
    start_row: int,
    stop_row: int,
    chunk_index: int,
    execution_backend: str,
) -> dict[str, object]:
    masks = refined_group["masks_roi"]
    row_slice = slice(int(start_row), int(stop_row))
    chunk_start = time.perf_counter()
    chunk_timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row) - int(start_row),
        "execution_backend": execution_backend,
    }
    batches: dict[str, ComponentBatch] = {}
    component_masks_by_name: dict[str, np.ndarray] = {}
    rows_with_component: dict[str, int] = {}
    for component_name, component_idx in component_indices:
        phase_start = time.perf_counter()
        component_masks = np.asarray(masks[row_slice, int(component_idx)], dtype=np.uint8)
        batch = _compute_component_batch(component_masks, str(component_name))
        _write_component_batch(run_group, str(component_name), row_slice, batch)
        batches[str(component_name)] = batch
        if str(component_name) in {"subject_body", "swim_bladder"}:
            component_masks_by_name[str(component_name)] = component_masks
        rows_with_component[str(component_name)] = int(np.count_nonzero(batch.mask_present))
        chunk_timing[f"write_{component_name}_seconds"] = float(time.perf_counter() - phase_start)

    body_frame: Optional[BodyFrameBatch] = None
    if set(BODY_FRAME_COMPONENTS).issubset(batches):
        phase_start = time.perf_counter()
        body_frame = _compute_body_frame_batch(batches)
        _write_body_frame_batch(run_group, row_slice, body_frame)
        chunk_timing["write_body_frame_seconds"] = float(time.perf_counter() - phase_start)
        rows_with_component["body_frame_valid"] = int(np.count_nonzero(body_frame.valid))

    caudal_anchor: Optional[CaudalAnchorBatch] = None
    if body_frame is not None and "swim_bladder" in component_masks_by_name:
        phase_start = time.perf_counter()
        caudal_anchor = _compute_caudal_anchor_batch(component_masks_by_name["swim_bladder"], body_frame)
        _write_caudal_anchor_batch(run_group, row_slice, caudal_anchor)
        chunk_timing["write_caudal_anchor_seconds"] = float(time.perf_counter() - phase_start)
        rows_with_component["caudal_contour_valid"] = int(np.count_nonzero(caudal_anchor.valid))

    if body_frame is not None and caudal_anchor is not None and "subject_body" in component_masks_by_name:
        phase_start = time.perf_counter()
        centerline = _compute_centerline_batch(
            component_masks_by_name["subject_body"],
            body_frame,
            caudal_anchor,
            sample_count=CENTERLINE_SAMPLE_COUNT,
        )
        _write_centerline_batch(run_group, row_slice, centerline)
        chunk_timing["write_centerline_seconds"] = float(time.perf_counter() - phase_start)
        rows_with_component["centerline_valid"] = int(np.count_nonzero(centerline.centerline_valid))
        rows_with_component["tail_base_valid"] = int(np.count_nonzero(centerline.tail_base_valid))

    phase_start = time.perf_counter()
    _write_relations(run_group, row_slice, batches)
    chunk_timing["write_relations_seconds"] = float(time.perf_counter() - phase_start)
    chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {
        "chunk_timing": chunk_timing,
        "rows_with_component": rows_with_component,
    }


def _process_and_write_subject_shape_chunk(
    zarr_path: str,
    *,
    refined_run: str,
    shape_run: str,
    component_indices: Sequence[tuple[str, int]],
    start_row: int,
    stop_row: int,
    chunk_index: int,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return _process_and_write_subject_shape_chunk_groups(
        root["refined_subject_masks_runs"][refined_run],
        root["analysis"]["subject_shape_runs"][shape_run],
        component_indices=component_indices,
        start_row=start_row,
        stop_row=stop_row,
        chunk_index=chunk_index,
        execution_backend=DASK_WORKER_EXECUTION_BACKEND,
    )


def _compute_dask_tasks(
    tasks: Sequence[object],
    *,
    scheduler_key: str,
    num_workers: Optional[int],
) -> list[dict[str, object]]:
    if not tasks:
        return []
    cluster = None
    client = None
    try:
        if scheduler_key == "distributed":
            if not HAVE_DISTRIBUTED:
                raise RuntimeError(
                    "Dask distributed is not available. Install dask[distributed] or choose a different scheduler."
                )
            cluster_kwargs: dict[str, object] = {}
            if num_workers is not None:
                cluster_kwargs["n_workers"] = int(num_workers)
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            results = list(client.gather(client.compute(list(tasks))))
        else:
            compute_kwargs: dict[str, object] = {"scheduler": scheduler_key}
            if num_workers is not None and scheduler_key != "single-threaded":
                compute_kwargs["num_workers"] = int(num_workers)
            results = list(dask.compute(*tasks, **compute_kwargs))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    return [dict(result) for result in results]


def write_subject_shape_run_group(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    refined_run: Optional[str] = None,
    run_name: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    execution_backend: str = SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    include_chunk_timings: bool = False,
    stage_command: Optional[str] = None,
) -> dict[str, object]:
    """Write one row-aligned subject-shape analysis run."""

    scheduler_key = _normalize_scheduler(scheduler)
    backend = _normalize_execution_backend(execution_backend)
    if backend == DASK_WORKER_EXECUTION_BACKEND and zarr_path is None:
        raise ValueError("execution_backend='dask_worker_chunks' requires a filesystem zarr_path.")
    refined_run_name, refined_group = _resolve_refined_run(root, refined_run)
    component_indices = _resolve_components(refined_group, components)
    masks = refined_group["masks_roi"]
    total_rows = int(masks.shape[0])
    target_run = str(run_name or _default_run_name())
    chunks = _row_chunks(total_rows, max(1, int(chunk_size)))
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "source_refined_subject_masks_run": refined_run_name,
        "subject_shape_run": target_run,
        "component_names": [name for name, _idx in component_indices],
        "roi_count": total_rows,
        "chunk_size": max(1, int(chunk_size)),
        "chunk_count": len(chunks),
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "mutates_archive": not bool(dry_run),
    }
    if dry_run:
        return summary

    stage_start = time.perf_counter()
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    run_group = _prepare_subject_shape_run(
        root,
        target_run=target_run,
        refined_run_name=refined_run_name,
        refined_group=refined_group,
        component_indices=component_indices,
        chunk_size=max(1, int(chunk_size)),
        execution_backend=backend,
        scheduler=scheduler_key,
        num_workers=num_workers,
        stage_command=command,
        overwrite=overwrite,
    )

    chunk_timings: list[dict[str, object]] = []
    rows_with_component: dict[str, int] = {name: 0 for name, _idx in component_indices}
    if backend == DASK_WORKER_EXECUTION_BACKEND:
        assert zarr_path is not None
        tasks = [
            delayed(_process_and_write_subject_shape_chunk)(
                str(zarr_path),
                refined_run=refined_run_name,
                shape_run=target_run,
                component_indices=tuple(component_indices),
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
        results = _compute_dask_tasks(tasks, scheduler_key=scheduler_key, num_workers=num_workers)
        for result in sorted(results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
            chunk_timings.append(dict(result["chunk_timing"]))
            for component_name, count in dict(result.get("rows_with_component") or {}).items():
                rows_with_component[str(component_name)] = int(rows_with_component.get(str(component_name), 0)) + int(count)
    else:
        for chunk_index, (start_row, stop_row) in enumerate(chunks):
            result = _process_and_write_subject_shape_chunk_groups(
                refined_group,
                run_group,
                component_indices=tuple(component_indices),
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                execution_backend=SERIAL_EXECUTION_BACKEND,
            )
            chunk_timing = dict(result["chunk_timing"])
            chunk_timings.append(chunk_timing)
            for component_name, count in dict(result.get("rows_with_component") or {}).items():
                rows_with_component[str(component_name)] = int(rows_with_component.get(str(component_name), 0)) + int(count)

    duration_seconds = float(time.perf_counter() - stage_start)
    rows_per_second = float(total_rows / duration_seconds) if duration_seconds > 0.0 else float("inf")
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["rows_per_second"] = rows_per_second
    run_group.attrs["rows_with_component"] = rows_with_component
    run_group.attrs["subject_shape_timing_summary"] = {
        "total_rows": total_rows,
        "duration_seconds": duration_seconds,
        "rows_per_second": rows_per_second,
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_chunk_size": max(1, int(chunk_size)),
        "dask_version": getattr(dask, "__version__", "unknown"),
    }
    run_group.attrs["subject_shape_chunk_timings"] = list(_json_safe(chunk_timings))
    parent = root["analysis"]["subject_shape_runs"]
    parent.attrs["latest"] = target_run
    summary.update(
        {
            "status": "updated",
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "rows_with_component": rows_with_component,
            "chunk_timing_count": len(chunk_timings),
        }
    )
    if include_chunk_timings:
        summary["chunk_timings"] = list(_json_safe(chunk_timings))
    return dict(_json_safe(summary))


def write_subject_shape_run(
    zarr_path: str | Path,
    *,
    refined_run: Optional[str] = None,
    run_name: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    execution_backend: str = SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    include_chunk_timings: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run=refined_run,
        run_name=run_name,
        components=components,
        chunk_size=chunk_size,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        overwrite=overwrite,
        dry_run=dry_run,
        include_chunk_timings=include_chunk_timings,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write analysis/subject_shape_runs from refined subject-mask components."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--refined-run", help="refined_subject_masks_runs/<run> to consume; defaults to latest.")
    parser.add_argument("--run-name", help="Target analysis/subject_shape_runs/<run>; defaults to timestamped name.")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENT_ORDER,
        help="Optional component subset. Defaults to all available known subject-shape components.",
    )
    parser.add_argument("--component", action="append", dest="component_values", choices=COMPONENT_ORDER)
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of refined rows per worker chunk.")
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default=SERIAL_EXECUTION_BACKEND,
        help="Use dask_worker_chunks to let Dask workers write disjoint row chunks.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help="Dask scheduler used when --execution-backend=dask_worker_chunks.",
    )
    parser.add_argument("--num-workers", type=int, help="Dask worker count.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target subject-shape run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without mutating the archive.")
    parser.add_argument(
        "--include-chunk-timings",
        action="store_true",
        help="Include full per-chunk timing records in stdout. They are always stored in zarr attrs.",
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def _parse_components(values: Optional[Sequence[str]], repeated: Optional[Sequence[str]]) -> Optional[list[str]]:
    result: list[str] = []
    if values:
        result.extend(str(value) for value in values)
    if repeated:
        result.extend(str(value) for value in repeated)
    return result or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = write_subject_shape_run(
        args.zarr_path,
        refined_run=args.refined_run,
        run_name=args.run_name,
        components=_parse_components(args.components, args.component_values),
        chunk_size=int(args.chunk_size),
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        include_chunk_timings=bool(args.include_chunk_timings),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
