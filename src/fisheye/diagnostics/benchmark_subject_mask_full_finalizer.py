"""Benchmark the full subject-mask finalizer on a temporary zarr slice.

The existing scheduler diagnostic profiles only the pure spatial mask
finalization path. This diagnostic copies a contiguous subject-mask row window
into a temporary mini archive, runs the production finalizer there, and reports
the same phase timings written by normal finalized runs. It includes optional
eye-geometry and component-contour writes, so it can profile the expensive
postcompute phases without mutating canonical recording zarrs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from ..refinement.assemble_refined_subject_masks import (
    _resolve_keypoint_success_array,
    _resolve_subject_keypoint_group,
)
from ..refinement.finalize_subject_masks import _MASK_STORAGE_CHOICES, finalize_subject_masks
from ..shared.refined_subject_component_contours import (
    COMPONENT_CONTOUR_SCHEMA_ID,
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    extract_largest_external_contour,
)
from ..shared.refined_subject_eye_geometry import (
    EYE_COMPONENTS,
    EYE_GEOMETRY_SCHEMA_ID,
    EYE_PAIR_RELATION_SCHEMA_ID,
)
from ..shared.workflow_profile import WorkflowProfiler
from ..shared.workflow_profile import json_safe
from ..shared.zarr_run_completion import require_runs_parent
from ..tune.refined_subject_mask_review import _load_source_subject_mask_run
from ..utils.zarr_io import open_zarr_root

_ROW_ARRAY_CANDIDATES = (
    "mask_probs_roi",
    "masks_roi",
    "detection_source",
    "frame_indices",
    "frame_counts",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
)
_FULL_ARRAY_CANDIDATES = ("available_channels",)
_KEYPOINT_ROW_ARRAYS = ("keypoints_roi", "keypoints_img", "keypoint_scores")
_EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks", "process_shards")
_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
_POSTCOMPUTE_MODES = ("production", "sharded")
_CONTOUR_COMPONENTS = ("subject_body", "swim_bladder")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_attrs(source: zarr.Group, target: zarr.Group) -> None:
    target.attrs.update(dict(source.attrs))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return json_safe(value)


def _array_chunks_for_data(source: Any, data: np.ndarray) -> tuple[int, ...] | None:
    chunks = getattr(source, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(dim) for dim in chunks)
    if not chunks_tuple or data.ndim == 0:
        return chunks_tuple
    adjusted = list(chunks_tuple)
    adjusted[0] = max(1, min(int(data.shape[0]), int(adjusted[0])))
    for axis in range(1, min(data.ndim, len(adjusted))):
        adjusted[axis] = max(1, min(int(data.shape[axis]), int(adjusted[axis])))
    return tuple(adjusted)


def _create_array_from_data(target: zarr.Group, name: str, source_array: Any, data: np.ndarray) -> None:
    chunks = _array_chunks_for_data(source_array, data)
    kwargs: dict[str, object] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    target.create_array(name, **kwargs)


def _copy_row_array(
    source: zarr.Group,
    target: zarr.Group,
    name: str,
    *,
    start_row: int,
    stop_row: int,
) -> bool:
    if name not in source:
        return False
    source_array = source[name]
    if len(source_array.shape) == 0:
        data = np.asarray(source_array[:])
    else:
        if int(source_array.shape[0]) < int(stop_row):
            raise ValueError(
                f"{source.path}/{name} has {source_array.shape[0]} rows; cannot copy rows "
                f"{start_row}:{stop_row}."
            )
        data = np.asarray(source_array[int(start_row) : int(stop_row)])
    _create_array_from_data(target, name, source_array, data)
    return True


def _copy_full_array(source: zarr.Group, target: zarr.Group, name: str) -> bool:
    if name not in source:
        return False
    source_array = source[name]
    data = np.asarray(source_array[:])
    _create_array_from_data(target, name, source_array, data)
    return True


def _recompute_frame_counts_from_frame_indices(group: zarr.Group) -> dict[str, object] | None:
    """Repair sliced temp archives where copied frame_counts would cut a frame window."""

    frame_indices_array = group.get("frame_indices")
    if frame_indices_array is None:
        return None
    frame_indices = np.asarray(frame_indices_array[:], dtype=np.int64).reshape(-1)
    valid = frame_indices >= 0
    max_frame = int(frame_indices[valid].max()) if bool(np.any(valid)) else -1
    counts = np.bincount(frame_indices[valid], minlength=max_frame + 1).astype(np.int32, copy=False)
    source_counts = group.get("frame_counts")
    chunks = None
    if source_counts is not None:
        chunks = _array_chunks_for_data(source_counts, counts)
    kwargs: dict[str, object] = {"data": counts, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    group.create_array("frame_counts", **kwargs)
    return {
        "status": "recomputed_from_frame_indices",
        "frame_count_rows": int(counts.shape[0]),
        "roi_count": int(frame_indices.shape[0]),
        "count_sum": int(np.sum(counts, dtype=np.int64)),
    }


def _copy_group_attrs_only(
    source_parent: zarr.Group,
    target_root: zarr.Group,
    parent_name: str,
    run_name: str,
) -> zarr.Group | None:
    if run_name not in source_parent:
        return None
    target_parent = require_runs_parent(target_root, parent_name)
    target_parent.attrs["latest"] = str(run_name)
    source_run = source_parent[run_name]
    target_run = target_parent.require_group(run_name)
    _copy_attrs(source_run, target_run)
    return target_run


def _copy_subject_mask_slice(
    source_root: zarr.Group,
    target_root: zarr.Group,
    *,
    source_run_name: str,
    start_row: int,
    stop_row: int,
) -> dict[str, object]:
    source_parent = source_root.get("subject_mask_runs")
    if source_parent is None or source_run_name not in source_parent:
        raise RuntimeError(f"subject_mask_runs/{source_run_name} not found.")
    source_run = source_parent[source_run_name]
    target_parent = require_runs_parent(target_root, "subject_mask_runs")
    target_parent.attrs["latest"] = str(source_run_name)
    target_run = target_parent.require_group(source_run_name)
    _copy_attrs(source_run, target_run)
    target_run.attrs["benchmark_source_start_row"] = int(start_row)
    target_run.attrs["benchmark_source_stop_row"] = int(stop_row)

    copied_rows = []
    copied_full = []
    for name in _ROW_ARRAY_CANDIDATES:
        if _copy_row_array(source_run, target_run, name, start_row=start_row, stop_row=stop_row):
            copied_rows.append(name)
    for name in _FULL_ARRAY_CANDIDATES:
        if _copy_full_array(source_run, target_run, name):
            copied_full.append(name)
    repaired_frame_counts = _recompute_frame_counts_from_frame_indices(target_run)
    return {
        "subject_run": str(source_run_name),
        "row_arrays": copied_rows,
        "full_arrays": copied_full,
        "frame_counts_repair": repaired_frame_counts,
    }


def _copy_crop_context(
    source_root: zarr.Group,
    target_root: zarr.Group,
    *,
    source_crop_run: str,
    start_row: int,
    stop_row: int,
) -> dict[str, object] | None:
    crop_parent = source_root.get("crop_runs")
    if crop_parent is None or not source_crop_run or source_crop_run not in crop_parent:
        return None
    target_crop = _copy_group_attrs_only(crop_parent, target_root, "crop_runs", source_crop_run)
    if target_crop is None:
        return None
    source_crop = crop_parent[source_crop_run]
    copied_rows = []
    for name in (
        "frame_indices",
        "frame_counts",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
    ):
        if _copy_row_array(source_crop, target_crop, name, start_row=start_row, stop_row=stop_row):
            copied_rows.append(name)
    repaired_frame_counts = _recompute_frame_counts_from_frame_indices(target_crop)
    return {
        "crop_run": str(source_crop_run),
        "row_arrays": copied_rows,
        "frame_counts_repair": repaired_frame_counts,
    }


def _copy_keypoint_context(
    source_root: zarr.Group,
    target_root: zarr.Group,
    *,
    source: Any,
    start_row: int,
    stop_row: int,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    kp_group, keypoint_run_name, keypoint_group_name, source_kind = _resolve_subject_keypoint_group(
        source_root,
        source,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    source_parent = source_root.get(keypoint_group_name)
    if source_parent is None:
        raise RuntimeError(f"{keypoint_group_name} not found.")
    target_parent = target_root.require_group(keypoint_group_name)
    target_parent.attrs["latest"] = str(keypoint_run_name)
    target_run = target_parent.require_group(keypoint_run_name)
    _copy_attrs(kp_group, target_run)

    copied_rows = []
    for name in _KEYPOINT_ROW_ARRAYS:
        if _copy_row_array(kp_group, target_run, name, start_row=start_row, stop_row=stop_row):
            copied_rows.append(name)
    _success_values, success_dataset = _resolve_keypoint_success_array(kp_group, keypoint_run_name)
    if _copy_row_array(kp_group, target_run, success_dataset, start_row=start_row, stop_row=stop_row):
        copied_rows.append(success_dataset)
    return {
        "keypoint_group": str(keypoint_group_name),
        "keypoint_run": str(keypoint_run_name),
        "keypoint_source_kind": str(source_kind),
        "keypoint_success_dataset": str(success_dataset),
        "row_arrays": copied_rows,
    }


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _make_temp_zarr_path(temp_dir: Optional[str | Path]) -> tuple[Path, Path]:
    parent = Path(temp_dir).expanduser().resolve() if temp_dir else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="palette_subject_mask_full_finalizer_", dir=str(parent)))
    return work_dir, work_dir / "benchmark_subject_mask_full_finalizer.zarr"


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): int(idx) for idx, label in enumerate(labels_raw)}


def _available_channels(group: zarr.Group, channel_count: int) -> np.ndarray:
    available_arr = group.get("available_channels")
    if available_arr is None:
        return np.ones((int(channel_count),), dtype=bool)
    available = np.asarray(available_arr[:], dtype=bool).reshape(-1)
    if int(available.shape[0]) < int(channel_count):
        padded = np.zeros((int(channel_count),), dtype=bool)
        padded[: int(available.shape[0])] = available
        return padded
    return available[: int(channel_count)]


def _empty_local_contour_pack(row_count: int) -> dict[str, object]:
    return {
        "ptr": np.full((int(row_count),), -1, dtype=np.int64),
        "len": np.zeros((int(row_count),), dtype=np.int32),
        "points": [],
        "point_count": 0,
        "contour_count": 0,
    }


def _add_local_contour(pack: dict[str, object], row_index: int, contour: np.ndarray | None, *, min_points: int) -> None:
    if contour is None:
        return
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if int(points.shape[0]) < int(min_points):
        return
    local_offset = int(pack["point_count"])
    ptr = pack["ptr"]
    length = pack["len"]
    assert isinstance(ptr, np.ndarray)
    assert isinstance(length, np.ndarray)
    ptr[int(row_index)] = np.int64(local_offset)
    length[int(row_index)] = np.int32(points.shape[0])
    points_list = pack["points"]
    assert isinstance(points_list, list)
    points_list.append(points)
    pack["point_count"] = local_offset + int(points.shape[0])
    pack["contour_count"] = int(pack["contour_count"]) + 1


def _finalize_local_contour_pack(pack: dict[str, object]) -> dict[str, object]:
    points_list = pack["points"]
    assert isinstance(points_list, list)
    if points_list:
        points_xy = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
    else:
        points_xy = np.zeros((0, 2), dtype=np.float32)
    return {
        "ptr": pack["ptr"],
        "len": pack["len"],
        "points_xy": points_xy,
        "point_count": int(pack["point_count"]),
        "contour_count": int(pack["contour_count"]),
    }


def _compute_subject_mask_postcompute_shard(
    zarr_path: str,
    refined_run: str,
    start_row: int,
    stop_row: int,
    *,
    write_eye_geometry: bool,
    write_component_contours: bool,
) -> dict[str, object]:
    from ..refinement.refine_eye_masks import _measure_mask

    started = time.perf_counter()
    root = open_zarr_root(zarr_path, mode="r")
    run_group = root["refined_subject_masks_runs"][refined_run]
    masks_roi = run_group["masks_roi"]
    label_map = _label_index_map(run_group)
    channel_count = int(masks_roi.shape[1])
    available = _available_channels(run_group, channel_count)
    start = int(start_row)
    stop = int(stop_row)
    row_count = max(0, stop - start)
    masks = np.asarray(masks_roi[start:stop], dtype=np.uint8)

    eye_payload: dict[str, object] | None = None
    if write_eye_geometry and set(EYE_COMPONENTS).issubset(label_map):
        ellipse_params = np.full((row_count, 2, 5), np.nan, dtype=np.float32)
        ellipse_success = np.zeros((row_count, 2), dtype=bool)
        separation_px = np.full((row_count,), np.nan, dtype=np.float32)
        separation_valid = np.zeros((row_count,), dtype=bool)
        centroids = np.full((row_count, 2, 2), np.nan, dtype=np.float32)
        eye_contours = {name: _empty_local_contour_pack(row_count) for name in EYE_COMPONENTS}

        for local_idx in range(row_count):
            for eye_idx, component_name in enumerate(EYE_COMPONENTS):
                comp_idx = int(label_map[component_name])
                if comp_idx >= int(available.shape[0]) or not bool(available[comp_idx]):
                    continue
                success, ellipse, centroid, contour, _failure = _measure_mask(masks[local_idx, comp_idx])
                ellipse_params[local_idx, eye_idx] = np.asarray(ellipse, dtype=np.float32)
                ellipse_success[local_idx, eye_idx] = bool(success)
                centroids[local_idx, eye_idx] = np.asarray(centroid, dtype=np.float32)
                _add_local_contour(eye_contours[component_name], local_idx, contour, min_points=1)
            if bool(np.all(ellipse_success[local_idx])) and bool(np.all(np.isfinite(centroids[local_idx]))):
                separation_px[local_idx] = np.float32(np.linalg.norm(centroids[local_idx, 0] - centroids[local_idx, 1]))
                separation_valid[local_idx] = True

        eye_payload = {
            "ellipse_params": ellipse_params,
            "ellipse_success": ellipse_success,
            "separation_px": separation_px,
            "separation_valid": separation_valid,
            "contours": {name: _finalize_local_contour_pack(pack) for name, pack in eye_contours.items()},
            "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
            "pair_success_count": int(np.count_nonzero(separation_valid)),
        }

    contour_payload: dict[str, object] = {}
    if write_component_contours:
        for component_name in _CONTOUR_COMPONENTS:
            if component_name not in label_map:
                continue
            comp_idx = int(label_map[component_name])
            if comp_idx >= int(available.shape[0]) or not bool(available[comp_idx]):
                continue
            pack = _empty_local_contour_pack(row_count)
            for local_idx in range(row_count):
                contour = extract_largest_external_contour(masks[local_idx, comp_idx], min_points=2)
                _add_local_contour(pack, local_idx, contour, min_points=2)
            contour_payload[component_name] = _finalize_local_contour_pack(pack)

    return {
        "start_row": start,
        "stop_row": stop,
        "row_count": int(row_count),
        "duration_seconds": float(time.perf_counter() - started),
        "eye_geometry": eye_payload,
        "component_contours": contour_payload,
    }


def _merge_contour_packs(
    shards: Sequence[dict[str, object]],
    component: str,
    *,
    total_rois: int,
    source_key: str,
) -> dict[str, object]:
    ptr = np.full((int(total_rois),), -1, dtype=np.int64)
    length = np.zeros((int(total_rois),), dtype=np.int32)
    point_chunks: list[np.ndarray] = []
    global_offset = 0
    contour_count = 0

    for shard in shards:
        source_payload = shard.get(source_key)
        if not isinstance(source_payload, dict):
            continue
        if source_key == "eye_geometry":
            contours_by_component = source_payload.get("contours")
            if not isinstance(contours_by_component, dict):
                continue
            pack = contours_by_component.get(component)
        else:
            pack = source_payload.get(component)
        if not isinstance(pack, dict):
            continue
        start = int(shard["start_row"])
        local_ptr = np.asarray(pack["ptr"], dtype=np.int64)
        local_len = np.asarray(pack["len"], dtype=np.int32)
        local_points = np.asarray(pack["points_xy"], dtype=np.float32).reshape(-1, 2)
        valid = local_len > 0
        local_rows = np.nonzero(valid)[0]
        for local_idx in local_rows:
            ptr[start + int(local_idx)] = np.int64(global_offset + int(local_ptr[int(local_idx)]))
            length[start + int(local_idx)] = np.int32(local_len[int(local_idx)])
        if int(local_points.shape[0]) > 0:
            point_chunks.append(local_points)
            global_offset += int(local_points.shape[0])
        contour_count += int(np.count_nonzero(valid))

    points_xy = (
        np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        if point_chunks
        else np.zeros((1, 2), dtype=np.float32)
    )
    return {
        "ptr": ptr,
        "len": length,
        "points_xy": points_xy,
        "point_count": int(points_xy.shape[0]) if point_chunks else 0,
        "contour_count": int(contour_count),
        "points_placeholder_when_empty": bool(not point_chunks),
    }


def _write_packed_component_contours(
    component_group: zarr.Group,
    pack: dict[str, object],
    *,
    chunk_rois: int,
    component: str,
    source_mask_run: str,
    source_mask_label_schema_id: str,
    min_points: int,
) -> dict[str, object]:
    ptr = np.asarray(pack["ptr"], dtype=np.int64)
    length = np.asarray(pack["len"], dtype=np.int32)
    points_xy = np.asarray(pack["points_xy"], dtype=np.float32).reshape(-1, 2)
    contours_group = component_group.require_group("contours")
    contours_group.attrs.update(
        {
            "schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "contour_schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
            "point_order": "xy",
            "source_component": str(component),
            "source_mask_run": str(source_mask_run),
            "source_mask_label_schema_id": str(source_mask_label_schema_id or ""),
            "method": DEFAULT_CONTOUR_METHOD,
            "method_version": DEFAULT_CONTOUR_METHOD_VERSION,
            "boundary_policy": DEFAULT_BOUNDARY_POLICY,
            "min_points": int(min_points),
            "generated_at_utc": _utc_now(),
            "points_placeholder_when_empty": bool(pack.get("points_placeholder_when_empty")),
            "cache_coverage": "full_indexed_rows",
            "benchmark_postcompute_mode": "sharded_parent_merge",
        }
    )
    contours_group.create_array("ptr", data=ptr, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array("len", data=length, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array(
        "points_xy",
        data=points_xy,
        chunks=(max(1, min(4096, int(points_xy.shape[0]))), 2),
        overwrite=True,
    )
    return {
        "component": str(component),
        "status": "written",
        "roi_count": int(ptr.shape[0]),
        "contour_count": int(pack.get("contour_count", 0)),
        "point_count": int(pack.get("point_count", 0)),
    }


def _write_sharded_eye_geometry(
    run_group: zarr.Group,
    shards: Sequence[dict[str, object]],
    *,
    chunk_rois: int,
    refined_run: str,
) -> dict[str, object]:
    label_map = _label_index_map(run_group)
    if not set(EYE_COMPONENTS).issubset(label_map):
        return {"status": "skipped", "reason": "missing_eye_components"}
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return {"status": "skipped", "reason": "missing_masks_roi"}

    total_rois = int(masks_roi.shape[0])
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    separation_px = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_valid = np.zeros((total_rois,), dtype=bool)

    for shard in shards:
        eye_payload = shard.get("eye_geometry")
        if not isinstance(eye_payload, dict):
            continue
        start = int(shard["start_row"])
        stop = int(shard["stop_row"])
        ellipse_params[start:stop] = np.asarray(eye_payload["ellipse_params"], dtype=np.float32)
        ellipse_success[start:stop] = np.asarray(eye_payload["ellipse_success"], dtype=bool)
        separation_px[start:stop] = np.asarray(eye_payload["separation_px"], dtype=np.float32)
        separation_valid[start:stop] = np.asarray(eye_payload["separation_valid"], dtype=bool)

    components_parent = run_group.require_group("components")
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    for eye_idx, component_name in enumerate(EYE_COMPONENTS):
        component_group = components_parent.require_group(component_name)
        geometry_group = component_group.require_group("geometry")
        geometry_group.attrs["geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
        geometry_group.attrs["geometry_method"] = "fit_ellipse_from_refined_subject_component_mask"
        geometry_group.attrs["source_mask_component"] = component_name
        geometry_group.attrs["updated_at_utc"] = _utc_now()
        geometry_group.attrs["benchmark_postcompute_mode"] = "sharded_parent_merge"
        geometry_group.create_array(
            "ellipse_params",
            data=ellipse_params[:, eye_idx, :],
            chunks=(max(1, int(chunk_rois)), 5),
            overwrite=True,
        )
        geometry_group.create_array(
            "ellipse_success",
            data=ellipse_success[:, eye_idx],
            chunks=(max(1, int(chunk_rois)),),
            overwrite=True,
        )
        pack = _merge_contour_packs(shards, component_name, total_rois=total_rois, source_key="eye_geometry")
        _write_packed_component_contours(
            component_group,
            pack,
            chunk_rois=chunk_rois,
            component=component_name,
            source_mask_run=refined_run,
            source_mask_label_schema_id=source_label_schema,
            min_points=1,
        )

    relation_metrics = run_group.require_group("relations").require_group("eye_pair").require_group("metrics")
    relation_metrics.attrs["relation_schema_id"] = EYE_PAIR_RELATION_SCHEMA_ID
    relation_metrics.attrs["relation_components"] = list(EYE_COMPONENTS)
    relation_metrics.attrs["relation_method"] = "ellipse_centroid_distance"
    relation_metrics.attrs["updated_at_utc"] = _utc_now()
    relation_metrics.attrs["benchmark_postcompute_mode"] = "sharded_parent_merge"
    relation_metrics.create_array("separation_px", data=separation_px, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    relation_metrics.create_array(
        "separation_valid",
        data=separation_valid,
        chunks=(max(1, int(chunk_rois)),),
        overwrite=True,
    )

    run_group.attrs["eye_geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
    run_group.attrs["eye_geometry_updated_at_utc"] = _utc_now()
    run_group.attrs["eye_geometry_status"] = "computed"
    run_group.attrs["eye_geometry_benchmark_postcompute_mode"] = "sharded_parent_merge"
    if "eye_geometry_deferred_reason" in run_group.attrs:
        del run_group.attrs["eye_geometry_deferred_reason"]
    return {
        "status": "updated",
        "roi_count": total_rois,
        "components": list(EYE_COMPONENTS),
        "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
        "pair_success_count": int(np.count_nonzero(separation_valid)),
    }


def _write_sharded_component_contours(
    run_group: zarr.Group,
    shards: Sequence[dict[str, object]],
    *,
    chunk_rois: int,
    refined_run: str,
) -> list[dict[str, object]]:
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return []
    total_rois = int(masks_roi.shape[0])
    label_map = _label_index_map(run_group)
    components_parent = run_group.require_group("components")
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    summaries: list[dict[str, object]] = []
    for component_name in _CONTOUR_COMPONENTS:
        if component_name not in label_map:
            continue
        pack = _merge_contour_packs(shards, component_name, total_rois=total_rois, source_key="component_contours")
        component_group = components_parent.require_group(component_name)
        summaries.append(
            _write_packed_component_contours(
                component_group,
                pack,
                chunk_rois=chunk_rois,
                component=component_name,
                source_mask_run=refined_run,
                source_mask_label_schema_id=source_label_schema,
                min_points=2,
            )
        )
    if summaries:
        run_group.attrs["component_contours_status"] = "computed"
        run_group.attrs["component_contours_components"] = [item["component"] for item in summaries]
        run_group.attrs["component_contours_updated_at_utc"] = _utc_now()
        run_group.attrs["component_contours_summary"] = list(_json_safe(summaries))
        run_group.attrs["component_contours_benchmark_postcompute_mode"] = "sharded_parent_merge"
    return summaries


def _run_sharded_subject_mask_postcompute(
    temp_zarr_path: Path,
    *,
    refined_run: str,
    chunk_size: int,
    num_workers: Optional[int],
    write_eye_geometry: bool,
    write_component_contours: bool,
) -> dict[str, object]:
    if not write_eye_geometry and not write_component_contours:
        return {"status": "skipped", "reason": "no_postcompute_requested"}

    root = open_zarr_root(temp_zarr_path, mode="r")
    run_group = root["refined_subject_masks_runs"][refined_run]
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return {"status": "skipped", "reason": "missing_masks_roi"}
    total_rois = int(masks_roi.shape[0])
    shard_size = max(1, min(int(chunk_size), total_rois if total_rois > 0 else 1))
    ranges = [(start, min(total_rois, start + shard_size)) for start in range(0, total_rois, shard_size)]
    worker_count = max(1, int(num_workers or 1))
    started = time.perf_counter()
    if worker_count == 1 or len(ranges) <= 1:
        shards = [
            _compute_subject_mask_postcompute_shard(
                str(temp_zarr_path),
                refined_run,
                start,
                stop,
                write_eye_geometry=write_eye_geometry,
                write_component_contours=write_component_contours,
            )
            for start, stop in ranges
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    _compute_subject_mask_postcompute_shard,
                    str(temp_zarr_path),
                    refined_run,
                    start,
                    stop,
                    write_eye_geometry=write_eye_geometry,
                    write_component_contours=write_component_contours,
                )
                for start, stop in ranges
            ]
            shards = [future.result() for future in futures]
    shards = sorted(shards, key=lambda item: int(item["start_row"]))

    root = open_zarr_root(temp_zarr_path, mode="a")
    run_group = root["refined_subject_masks_runs"][refined_run]
    chunk_rois = max(1, min(256, total_rois if total_rois > 0 else 1))
    eye_summary = (
        _write_sharded_eye_geometry(run_group, shards, chunk_rois=chunk_rois, refined_run=refined_run)
        if write_eye_geometry
        else {"status": "skipped", "reason": "write_eye_geometry=false"}
    )
    contour_summaries = (
        _write_sharded_component_contours(run_group, shards, chunk_rois=chunk_rois, refined_run=refined_run)
        if write_component_contours
        else []
    )
    duration_seconds = float(time.perf_counter() - started)
    run_group.attrs["benchmark_sharded_postcompute_summary"] = dict(
        _json_safe(
            {
                "status": "updated",
                "duration_seconds": duration_seconds,
                "rows_per_second": float(total_rois / duration_seconds) if duration_seconds > 0 else None,
                "roi_count": total_rois,
                "shard_count": len(shards),
                "shard_size": int(shard_size),
                "num_workers": int(worker_count),
                "write_eye_geometry": bool(write_eye_geometry),
                "write_component_contours": bool(write_component_contours),
                "eye_geometry": eye_summary,
                "component_contours": contour_summaries,
                "worker_durations_seconds": [float(item.get("duration_seconds") or 0.0) for item in shards],
            }
        )
    )
    return dict(run_group.attrs["benchmark_sharded_postcompute_summary"])


def _copy_benchmark_slice(
    zarr_path: str | Path,
    *,
    source_run: str,
    start_row: int,
    roi_count: int,
    temp_zarr_path: Path,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    source_root = open_zarr_root(zarr_path, mode="r")
    source = _load_source_subject_mask_run(source_root, source_run)
    total_rows = int(source.masks_roi.shape[0])
    start = max(0, int(start_row))
    count = max(1, int(roi_count))
    stop = min(total_rows, start + count)
    if stop <= start:
        raise ValueError(f"Empty benchmark window for total_rows={total_rows}, start_row={start_row}, roi_count={roi_count}.")

    target_root = zarr.open_group(str(temp_zarr_path), mode="w")
    target_root.attrs.update(
        {
            "benchmark_kind": "subject_mask_full_finalizer_slice",
            "benchmark_source_zarr_path": str(Path(zarr_path)),
            "benchmark_source_subject_run": str(source_run),
            "benchmark_source_start_row": int(start),
            "benchmark_source_stop_row": int(stop),
            "benchmark_source_roi_count": int(stop - start),
        }
    )

    copy_summary: dict[str, object] = {
        "source_total_rows": int(total_rows),
        "start_row": int(start),
        "stop_row": int(stop),
        "roi_count": int(stop - start),
    }
    copy_summary["subject_mask"] = _copy_subject_mask_slice(
        source_root,
        target_root,
        source_run_name=source.run_name,
        start_row=start,
        stop_row=stop,
    )
    crop_context = _copy_crop_context(
        source_root,
        target_root,
        source_crop_run=str(source.crop_run or ""),
        start_row=start,
        stop_row=stop,
    )
    if crop_context is not None:
        copy_summary["crop"] = crop_context
    copy_summary["keypoints"] = _copy_keypoint_context(
        source_root,
        target_root,
        source=source,
        start_row=start,
        stop_row=stop,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    return copy_summary


def benchmark_subject_mask_full_finalizer(
    zarr_path: str | Path,
    *,
    source_run: str,
    start_row: int = 0,
    roi_count: int = 256,
    chunk_size: int = 128,
    metric_level: str = "cheap",
    components: Optional[Sequence[str]] = None,
    write_eye_geometry: bool = True,
    write_component_contours: bool = True,
    retain_source_seeds: bool = False,
    mask_storage: str = "dense_uint8",
    postcompute_mode: str = "production",
    postcompute_chunk_size: Optional[int] = None,
    postcompute_num_workers: Optional[int] = None,
    execution_backend: str = "serial_driver",
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
    temp_dir: Optional[str | Path] = None,
    keep_temp: bool = False,
    progress_jsonl: Optional[str | Path] = None,
    workflow_profile_jsonl: Optional[str | Path] = None,
) -> dict[str, object]:
    postcompute_mode_key = str(postcompute_mode)
    if postcompute_mode_key not in _POSTCOMPUTE_MODES:
        raise ValueError(f"postcompute_mode must be one of {_POSTCOMPUTE_MODES}, got {postcompute_mode!r}.")
    work_dir, temp_zarr_path = _make_temp_zarr_path(temp_dir)
    refined_run = "refined_subject_masks_full_finalizer_benchmark"
    workflow_profile_explicit = workflow_profile_jsonl is not None
    workflow_profile_path = (
        Path(workflow_profile_jsonl).expanduser().resolve()
        if workflow_profile_jsonl is not None
        else work_dir / "full_finalizer_workflow.profile.jsonl"
    )
    profiler_path = workflow_profile_path if bool(workflow_profile_explicit or keep_temp) else None
    profiler = WorkflowProfiler(
        profiler_path,
        schema_prefix="palette_subject_mask_full_finalizer_benchmark_workflow",
    )
    archive_started = time.perf_counter()
    profiler.emit(
        "start",
        "archive_total",
        source_zarr_path=str(Path(zarr_path)),
        source_run=str(source_run),
        start_row=int(start_row),
        roi_count=int(roi_count),
        keep_temp=bool(keep_temp),
        postcompute_mode=postcompute_mode_key,
        retain_source_seeds=bool(retain_source_seeds),
        mask_storage=str(mask_storage),
    )
    payload: dict[str, object] | None = None
    caught: BaseException | None = None
    try:
        with profiler.phase(
            "copy_benchmark_slice",
            temp_zarr_path=str(temp_zarr_path),
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        ) as phase:
            copy_summary = _copy_benchmark_slice(
                zarr_path,
                source_run=source_run,
                start_row=start_row,
                roi_count=roi_count,
                temp_zarr_path=temp_zarr_path,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
            )
            phase["copied_roi_count"] = copy_summary.get("roi_count")
        copy_seconds = float(profiler.phase_seconds.get("copy_benchmark_slice", 0.0))
        progress_jsonl_explicit = progress_jsonl is not None
        if progress_jsonl is None:
            progress_path = work_dir / "full_finalizer_progress.jsonl"
        else:
            progress_path = Path(progress_jsonl).expanduser().resolve()
            progress_path.parent.mkdir(parents=True, exist_ok=True)

        with profiler.phase(
            "finalizer_run",
            temp_zarr_path=str(temp_zarr_path),
            refined_run=refined_run,
            execution_backend=execution_backend,
            scheduler=scheduler,
            num_workers=num_workers,
            chunk_size=int(chunk_size),
            write_eye_geometry=bool(write_eye_geometry) and postcompute_mode_key == "production",
            write_component_contours=bool(write_component_contours) and postcompute_mode_key == "production",
            retain_source_seeds=bool(retain_source_seeds),
            mask_storage=str(mask_storage),
            postcompute_mode=postcompute_mode_key,
        ) as phase:
            finalizer_summary = finalize_subject_masks(
                temp_zarr_path,
                subject_run=source_run,
                refined_run=refined_run,
                components=components,
                chunk_size=chunk_size,
                metric_level=metric_level,
                write_eye_geometry=bool(write_eye_geometry) and postcompute_mode_key == "production",
                write_component_contours=bool(write_component_contours) and postcompute_mode_key == "production",
                retain_source_seeds=bool(retain_source_seeds),
                mask_storage=str(mask_storage),
                execution_backend=execution_backend,
                scheduler=scheduler,
                num_workers=num_workers,
                overwrite=True,
                dry_run=False,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
                defer_registry_status=True,
                progress_jsonl=progress_path,
            )
            phase["rows_per_second"] = dict(finalizer_summary.get("timing_summary") or {}).get("rows_per_second")
            phase["duration_seconds_reported"] = finalizer_summary.get("duration_seconds")
        finalizer_wall_seconds = float(profiler.phase_seconds.get("finalizer_run", 0.0))
        sharded_postcompute_summary: dict[str, object] | None = None
        if postcompute_mode_key == "sharded":
            resolved_postcompute_chunk_size = int(postcompute_chunk_size or chunk_size)
            resolved_postcompute_num_workers = int(postcompute_num_workers or num_workers or 1)
            with profiler.phase(
                "sharded_postcompute",
                temp_zarr_path=str(temp_zarr_path),
                refined_run=refined_run,
                chunk_size=resolved_postcompute_chunk_size,
                num_workers=resolved_postcompute_num_workers,
                write_eye_geometry=bool(write_eye_geometry),
                write_component_contours=bool(write_component_contours),
            ) as phase:
                sharded_postcompute_summary = _run_sharded_subject_mask_postcompute(
                    temp_zarr_path,
                    refined_run=refined_run,
                    chunk_size=resolved_postcompute_chunk_size,
                    num_workers=resolved_postcompute_num_workers,
                    write_eye_geometry=bool(write_eye_geometry),
                    write_component_contours=bool(write_component_contours),
                )
                phase["rows_per_second"] = sharded_postcompute_summary.get("rows_per_second")
                phase["shard_count"] = sharded_postcompute_summary.get("shard_count")
                phase["shard_size"] = sharded_postcompute_summary.get("shard_size")

        temp_root = open_zarr_root(temp_zarr_path, mode="r")
        run_group = temp_root["refined_subject_masks_runs"][refined_run]
        timing_summary = dict(run_group.attrs.get("smart_finalizer_timing_summary") or {})
        summary_statistics = dict(run_group.attrs.get("summary_statistics") or {})
        output_group_path = temp_zarr_path / "refined_subject_masks_runs" / refined_run
        payload = {
            "status": "ok",
            "source_zarr_path": str(Path(zarr_path)),
            "source_run": str(source_run),
            "temp_work_dir": str(work_dir),
            "temp_zarr_path": str(temp_zarr_path),
            "keep_temp": bool(keep_temp),
            "copy_seconds": copy_seconds,
            "copy_summary": copy_summary,
            "finalizer_wall_seconds": finalizer_wall_seconds,
            "postcompute_mode": postcompute_mode_key,
            "requested_write_eye_geometry": bool(write_eye_geometry),
            "requested_write_component_contours": bool(write_component_contours),
            "retain_source_seeds": bool(retain_source_seeds),
            "mask_storage": str(mask_storage),
            "mask_storage_encoding": run_group.attrs.get("mask_storage_encoding"),
            "mask_store_encodings": list(run_group.attrs.get("mask_store_encodings") or []),
            "masks_roi_materialized": run_group.attrs.get("masks_roi_materialized"),
            "finalizer_write_eye_geometry": bool(write_eye_geometry) and postcompute_mode_key == "production",
            "finalizer_write_component_contours": (
                bool(write_component_contours) and postcompute_mode_key == "production"
            ),
            "sharded_postcompute_summary": sharded_postcompute_summary,
            "finalizer_summary": finalizer_summary,
            "timing_summary": timing_summary,
            "summary_statistics": summary_statistics,
            "phase_seconds": dict(timing_summary.get("phase_seconds") or {}),
            "progress_jsonl": str(progress_path),
            "progress_jsonl_retained": bool(progress_jsonl_explicit or keep_temp),
            "workflow_profile_jsonl": str(workflow_profile_path),
            "workflow_profile_jsonl_retained": bool(workflow_profile_explicit or keep_temp),
            "temp_zarr_size_bytes": _dir_size_bytes(temp_zarr_path),
            "refined_output_size_bytes": _dir_size_bytes(output_group_path),
            "temp_removed_after_run": not bool(keep_temp),
        }
    except BaseException as exc:
        caught = exc
        raise
    finally:
        if not keep_temp:
            with profiler.phase("cleanup_temp_work_dir", temp_work_dir=str(work_dir)):
                shutil.rmtree(work_dir, ignore_errors=True)
        profiler.record_finish(
            "archive_total",
            {
                "duration_seconds": float(time.perf_counter() - archive_started),
                "status": "error" if caught is not None else "ok",
                "error": repr(caught) if caught is not None else "",
            },
        )
    if payload is None:
        raise RuntimeError("Benchmark did not produce a payload.")
    payload["workflow_profile"] = profiler.summary()
    return _json_safe(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the source Palette zarr archive.")
    parser.add_argument("--source-run", required=True, help="subject_mask_runs/<run> to benchmark.")
    parser.add_argument("--start-row", type=int, default=0, help="First ROI row to copy into the temp archive.")
    parser.add_argument("--roi-count", type=int, default=256, help="Number of contiguous ROI rows to benchmark.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Rows per production finalizer chunk.")
    parser.add_argument(
        "--metric-level",
        choices=("cheap", "full"),
        default="cheap",
        help="Metric depth passed to the production finalizer.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help="Optional output component selector. Repeat to add components. Defaults to production auto-selection.",
    )
    parser.add_argument(
        "--write-eye-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include refined eye-geometry writes in the full workflow benchmark.",
    )
    parser.add_argument(
        "--write-component-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include body/swim component-contour writes in the full workflow benchmark.",
    )
    parser.add_argument(
        "--retain-source-seeds",
        action="store_true",
        help="Retain source_seed_masks_roi arrays in the benchmarked refined run.",
    )
    parser.add_argument(
        "--mask-storage",
        choices=_MASK_STORAGE_CHOICES,
        default="dense_uint8",
        help="Physical mask storage mode passed to the production finalizer.",
    )
    parser.add_argument(
        "--postcompute-mode",
        choices=_POSTCOMPUTE_MODES,
        default="production",
        help=(
            "Postcompute strategy for eye geometry and component contours. "
            "'production' runs the canonical finalizer path; 'sharded' benchmarks an experimental "
            "worker-sharded compute plus parent-merged write path."
        ),
    )
    parser.add_argument(
        "--postcompute-chunk-size",
        type=int,
        help="Rows per sharded postcompute worker task. Defaults to --chunk-size.",
    )
    parser.add_argument(
        "--postcompute-num-workers",
        type=int,
        help="Worker count for sharded postcompute. Defaults to --num-workers, then 1.",
    )
    parser.add_argument("--execution-backend", choices=_EXECUTION_BACKENDS, default="serial_driver")
    parser.add_argument("--scheduler", choices=_SCHEDULERS, default="single-threaded")
    parser.add_argument("--num-workers", type=int, help="Worker count for process_shards or dask_worker_chunks.")
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=("refined_keypoints_runs", "keypoints_runs"),
        help="Explicit keypoint group for eye assignment.",
    )
    parser.add_argument("--assignment-keypoints-run", help="Explicit keypoint run for eye assignment.")
    parser.add_argument("--temp-dir", help="Directory for the temporary benchmark archive. Defaults to system temp.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary archive for inspection.")
    parser.add_argument("--progress-jsonl", help="Optional path for finalizer progress events.")
    parser.add_argument("--workflow-profile-jsonl", help="Optional path for benchmark workflow profile events.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = benchmark_subject_mask_full_finalizer(
        args.zarr_path,
        source_run=args.source_run,
        start_row=int(args.start_row),
        roi_count=int(args.roi_count),
        chunk_size=int(args.chunk_size),
        metric_level=args.metric_level,
        components=args.components,
        write_eye_geometry=bool(args.write_eye_geometry),
        write_component_contours=bool(args.write_component_contours),
        retain_source_seeds=bool(args.retain_source_seeds),
        mask_storage=args.mask_storage,
        postcompute_mode=args.postcompute_mode,
        postcompute_chunk_size=args.postcompute_chunk_size,
        postcompute_num_workers=args.postcompute_num_workers,
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
        temp_dir=args.temp_dir,
        keep_temp=bool(args.keep_temp),
        progress_jsonl=args.progress_jsonl,
        workflow_profile_jsonl=args.workflow_profile_jsonl,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
