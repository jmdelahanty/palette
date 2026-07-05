"""Write tool-compatible tail posture views from Palette subject-shape geometry."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.detect_reason_codec import decode_reason_bytes
from ..shared.json_safety import json_attr_safe
from ..shared.row_lineage import copy_row_lineage_arrays
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root
from .megabouts_convention_audit import resample_tail_keypoints
from .subject_shape_io import SubjectShapeRunTables, load_subject_shape_run_tables, resolve_subject_shape_run

TAIL_POSTURE_VIEW_SCHEMA_ID = "analysis.tail_posture_view_runs"
TAIL_POSTURE_VIEW_SCHEMA_VERSION = 1
TAIL_POSTURE_VIEW_STAGE_NAME = "analysis.tail_posture_view_runs"
TAIL_POSTURE_VIEW_METHOD = "tail_posture_view_from_subject_shape"
TAIL_POSTURE_VIEW_METHOD_VERSION = 1
DEFAULT_VIEW_FAMILY = "megabouts_compatible"
DEFAULT_KEYPOINT_COUNT = 11
REASON_BYTES_WIDTH = 64


@dataclass(frozen=True)
class TailPostureViewBatch:
    """Computed tool-compatible posture-view arrays."""

    head_xy: np.ndarray
    head_yaw_rad: np.ndarray
    tail_keypoints_xy: np.ndarray
    tail_angle_rad: np.ndarray
    tail_angle_deg: np.ndarray
    valid: np.ndarray
    failure_reason: np.ndarray
    failure_reason_bytes: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name(view_family: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"tail_posture_view_{view_family}_{stamp}"


_json_safe = json_attr_safe


def _encode_reasons(reasons: Sequence[object], *, width: int = REASON_BYTES_WIDTH) -> np.ndarray:
    out = np.zeros((len(reasons), int(width)), dtype=np.uint8)
    for idx, reason in enumerate(reasons):
        payload = str(reason or "").encode("utf-8", errors="replace")[: max(0, int(width) - 1)]
        if payload:
            out[int(idx), : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return out


def _set_reason_bytes_attrs(group: zarr.Group, *, width: int = REASON_BYTES_WIDTH) -> None:
    group.attrs["reason_encoding"] = "utf8-null-terminated"
    group.attrs["reason_bytes_width"] = int(width)
    group.attrs["reason_bytes_null_terminated"] = True


def _metric_chunks(total_rows: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(width))


def _metric_chunks_3d(total_rows: int, middle: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(middle), int(width))


def _write_array(
    group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    chunks: Optional[Sequence[int]] = None,
) -> None:
    if name in group:
        del group[name]
    kwargs: dict[str, object] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = tuple(int(value) for value in chunks)
    group.create_array(name, **kwargs)


def _resolve_subject_shape_tables(
    root: zarr.Group,
    shape_run: Optional[str],
    *,
    head_source: str,
) -> tuple[str, zarr.Group, SubjectShapeRunTables]:
    shape_group, run_name, _run_path = resolve_subject_shape_run(root, shape_run)
    tables = load_subject_shape_run_tables(
        root,
        run_name=run_name,
        component_names=("subject_body",),
        relation_names=(),
        component_array_names={
            "subject_body": (
                "tail_sample_s",
                "tail_sample_xy",
                str(head_source),
                "tail_sample_valid",
                "bspline_valid",
                "tail_sample_failure_reason_bytes",
                "bspline_failure_reason_bytes",
            )
        },
        include_body_frame=False,
        include_row_index=True,
        include_source_refined_subject_masks=True,
    )
    return run_name, shape_group, tables


def _read_optional_reason_labels(group: zarr.Group | Mapping[str, np.ndarray], name: str, row_count: int) -> np.ndarray:
    arr = group.get(name)
    if arr is None:
        return np.full((int(row_count),), "", dtype=object)
    data = np.asarray(arr[:] if hasattr(arr, "shape") and not isinstance(arr, np.ndarray) else arr)
    if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
        return decode_reason_bytes(data)
    return np.asarray(data, dtype=object).reshape(-1)


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    valid: Optional[np.ndarray] = None
    for arr in arrays:
        data = np.asarray(arr)
        row_valid = np.all(np.isfinite(data.reshape((data.shape[0], -1))), axis=1)
        valid = row_valid if valid is None else (valid & row_valid)
    if valid is None:
        raise ValueError("At least one array is required.")
    return valid


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return signed angle from v1 to v2 in image xy coordinates."""

    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or int(a.shape[1]) != 2:
        raise ValueError("v1 and v2 must both have shape (N, 2).")
    dot = np.einsum("ij,ij->i", a, b)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.arctan2(cross, dot)


def compute_cumulative_segment_angles_from_keypoints(
    *,
    head_xy: np.ndarray,
    tail_keypoints_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Megabouts-compatible cumulative segment angles.

    This is a small compatibility implementation of the public Megabouts
    keypoint contract: K ordered tail keypoints produce K-1 cumulative segment
    angle channels. It does not call or copy Megabouts preprocessing/classifier
    code and should be treated as a data-contract adapter, not Megabouts output.
    """

    head = np.asarray(head_xy, dtype=np.float64)
    tail = np.asarray(tail_keypoints_xy, dtype=np.float64)
    if head.ndim != 2 or int(head.shape[1]) != 2:
        raise ValueError("head_xy must have shape (N, 2).")
    if tail.ndim != 3 or int(tail.shape[2]) != 2:
        raise ValueError("tail_keypoints_xy must have shape (N, K, 2).")
    if int(tail.shape[0]) != int(head.shape[0]):
        raise ValueError("head_xy and tail_keypoints_xy must have the same row count.")
    if int(tail.shape[1]) < 2:
        raise ValueError("At least two tail keypoints are required.")

    row_count = int(tail.shape[0])
    segment_count = int(tail.shape[1] - 1)
    start_vector = tail[:, 0, :] - head
    segments = np.diff(tail, axis=1)
    start_norm = np.linalg.norm(start_vector, axis=1)
    segment_norm = np.linalg.norm(segments, axis=2)
    valid = (
        _finite_rows(head, tail)
        & np.isfinite(start_norm)
        & (start_norm > 1e-12)
        & np.all(np.isfinite(segment_norm) & (segment_norm > 1e-12), axis=1)
    )

    angles = np.full((row_count, segment_count), np.nan, dtype=np.float32)
    head_yaw = np.full((row_count,), np.nan, dtype=np.float32)
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        start_valid = start_vector[valid]
        segments_valid = segments[valid]
        relative = np.zeros((valid_idx.shape[0], segment_count), dtype=np.float64)
        relative[:, 0] = _angle_between_vectors(start_valid, segments_valid[:, 0, :])
        for idx in range(segment_count - 1):
            relative[:, idx + 1] = _angle_between_vectors(
                segments_valid[:, idx, :],
                segments_valid[:, idx + 1, :],
            )
        angles[valid, :] = np.cumsum(relative, axis=1).astype(np.float32)
        head_yaw[valid] = np.arctan2(-start_valid[:, 1], -start_valid[:, 0]).astype(np.float32)
    return angles, head_yaw, valid


def compute_tail_posture_view_from_subject_shape_arrays(
    *,
    source_tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    head_xy: np.ndarray,
    tail_sample_valid: np.ndarray,
    bspline_valid: np.ndarray,
    tail_sample_failure_reason: Optional[Sequence[object]] = None,
    bspline_failure_reason: Optional[Sequence[object]] = None,
    keypoint_count: int = DEFAULT_KEYPOINT_COUNT,
) -> TailPostureViewBatch:
    """Compute a Megabouts-compatible posture view from subject-shape tail geometry."""

    tail_xy = np.asarray(tail_sample_xy, dtype=np.float64)
    head = np.asarray(head_xy, dtype=np.float64)
    if tail_xy.ndim != 3 or int(tail_xy.shape[2]) != 2:
        raise ValueError("tail_sample_xy must have shape (N, S, 2).")
    row_count = int(tail_xy.shape[0])
    if head.shape != (row_count, 2):
        raise ValueError("head_xy must have shape (N, 2).")
    sample_valid = np.asarray(tail_sample_valid, dtype=bool).reshape(-1)
    spline_valid = np.asarray(bspline_valid, dtype=bool).reshape(-1)
    if int(sample_valid.shape[0]) != row_count or int(spline_valid.shape[0]) != row_count:
        raise ValueError("validity arrays must have the same row count as tail arrays.")

    source_valid = sample_valid & spline_valid & _finite_rows(head)
    tail_keypoints = resample_tail_keypoints(
        source_tail_sample_s=source_tail_sample_s,
        tail_sample_xy=tail_xy,
        target_count=int(keypoint_count),
        valid=source_valid,
    )
    tail_angle, head_yaw, geometry_valid = compute_cumulative_segment_angles_from_keypoints(
        head_xy=head,
        tail_keypoints_xy=tail_keypoints,
    )

    valid = source_valid & geometry_valid
    reasons = np.full((row_count,), "ok", dtype=object)
    tail_reasons = (
        np.asarray(tail_sample_failure_reason, dtype=object).reshape(-1)
        if tail_sample_failure_reason is not None
        else np.full((row_count,), "tail_sample_invalid", dtype=object)
    )
    spline_reasons = (
        np.asarray(bspline_failure_reason, dtype=object).reshape(-1)
        if bspline_failure_reason is not None
        else np.full((row_count,), "bspline_invalid", dtype=object)
    )
    for row_idx in range(row_count):
        if bool(valid[row_idx]):
            continue
        if not bool(sample_valid[row_idx]):
            reason = str(tail_reasons[row_idx] or "tail_sample_invalid")
            reasons[row_idx] = reason if reason != "ok" else "tail_sample_invalid"
        elif not bool(spline_valid[row_idx]):
            reason = str(spline_reasons[row_idx] or "bspline_invalid")
            reasons[row_idx] = reason if reason != "ok" else "bspline_invalid"
        elif not np.all(np.isfinite(head[row_idx])):
            reasons[row_idx] = "head_nonfinite"
        else:
            reasons[row_idx] = "tail_geometry_nonfinite"

    invalid = ~valid
    tail_keypoints[invalid, :, :] = np.nan
    tail_angle[invalid, :] = np.nan
    head_yaw[invalid] = np.nan

    return TailPostureViewBatch(
        head_xy=head.astype(np.float32),
        head_yaw_rad=head_yaw.astype(np.float32),
        tail_keypoints_xy=tail_keypoints.astype(np.float32),
        tail_angle_rad=tail_angle.astype(np.float32),
        tail_angle_deg=np.rad2deg(tail_angle).astype(np.float32),
        valid=valid.astype(bool),
        failure_reason=reasons,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _read_sources(
    shape_tables: SubjectShapeRunTables,
    *,
    head_source: str,
) -> tuple[dict[str, np.ndarray], int]:
    body = shape_tables.require_component("subject_body")
    tail_xy = np.asarray(body.require_array("tail_sample_xy"), dtype=np.float32)
    row_count = int(tail_xy.shape[0])
    sources = {
        "source_tail_sample_s": np.asarray(body.require_array("tail_sample_s"), dtype=np.float32),
        "tail_sample_xy": tail_xy,
        "head_xy": np.asarray(body.require_array(head_source), dtype=np.float32),
        "tail_sample_valid": np.asarray(body.require_array("tail_sample_valid"), dtype=bool),
        "bspline_valid": np.asarray(body.require_array("bspline_valid"), dtype=bool),
        "tail_sample_failure_reason": _read_optional_reason_labels(
            body.arrays, "tail_sample_failure_reason_bytes", row_count
        ),
        "bspline_failure_reason": _read_optional_reason_labels(body.arrays, "bspline_failure_reason_bytes", row_count),
    }
    return sources, row_count


def _prepare_run_group(
    root: zarr.Group,
    *,
    target_run: str,
    shape_run_name: str,
    shape_group: zarr.Group,
    row_count: int,
    view_family: str,
    head_source: str,
    keypoint_count: int,
    source_tail_kinematics_run: Optional[str],
    stage_command: str,
    overwrite: bool,
) -> zarr.Group:
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "tail_posture_view_runs")
    if target_run in parent:
        if not overwrite:
            raise ValueError(
                f"analysis/tail_posture_view_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del parent[target_run]
    run_group = parent.create_group(target_run)
    mark_run_started(run_group, run_name=target_run, stage="tail_posture_view")
    _set_reason_bytes_attrs(run_group)

    row_index = run_group.require_group("row_index")
    shape_row_index = shape_group.get("row_index")
    if isinstance(shape_row_index, zarr.Group):
        copy_result = copy_row_lineage_arrays(
            row_index,
            shape_row_index,
            names=(
                "frame_indices",
                "detection_indices",
                "source_refined_row_ids",
                "source_detect_row_index",
                "source_crop_row_ids",
                "instance_key",
            ),
            total_rois=row_count,
            overwrite=True,
        )
    else:
        copy_result = None
    copied = list(copy_result.copied) if copy_result is not None else []
    missing = list(copy_result.missing) if copy_result is not None else [
        "frame_indices",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "source_crop_row_ids",
        "instance_key",
    ]
    if "frame_indices" in row_index:
        frame_index = np.asarray(row_index["frame_indices"][:])
        frame_index_source = "row_index/frame_indices"
    else:
        frame_index = np.arange(int(row_count), dtype=np.int64)
        frame_index_source = "row_number_fallback"
    _write_array(run_group, "frame_index", np.asarray(frame_index), chunks=_metric_chunks(row_count))

    created = _utc_now()
    angle_count = int(keypoint_count) - 1
    source_refined_run = shape_group.attrs.get("source_refined_subject_masks_run")
    source_refs = {
        "subject_shape_run": f"analysis/subject_shape_runs/{shape_run_name}",
        "subject_shape_body_component": f"analysis/subject_shape_runs/{shape_run_name}/components/subject_body",
    }
    if source_tail_kinematics_run:
        source_refs["tail_kinematics_run"] = f"analysis/tail_kinematics_runs/{source_tail_kinematics_run}"

    run_group.attrs.update(
        {
            "schema_id": TAIL_POSTURE_VIEW_SCHEMA_ID,
            "schema_version": TAIL_POSTURE_VIEW_SCHEMA_VERSION,
            "method": TAIL_POSTURE_VIEW_METHOD,
            "method_version": TAIL_POSTURE_VIEW_METHOD_VERSION,
            "created_at_utc": created,
            "created_utc": created,
            "row_axis": "roi_rows",
            "view_family": str(view_family),
            "compatible_tool": "megabouts" if str(view_family) == DEFAULT_VIEW_FAMILY else None,
            "dependency_policy": "no_megabouts_dependency_required",
            "source_subject_shape_run": str(shape_run_name),
            "source_subject_shape_path": f"analysis/subject_shape_runs/{shape_run_name}",
            "source_refined_subject_masks_run": str(source_refined_run) if source_refined_run is not None else None,
            "source_tail_kinematics_run": str(source_tail_kinematics_run) if source_tail_kinematics_run else None,
            "source_tail_geometry_kind": "subject_shape_tail_curve_resample",
            "head_source": str(head_source),
            "keypoint_count": int(keypoint_count),
            "angle_count": angle_count,
            "angle_units_primary": "rad",
            "angle_convention": "megabouts_cumulative_segment_angle",
            "keypoint_order": "tail_base_to_tail_tip",
            "tail_base_definition": "subject_shape.components.subject_body.tail_sample_xy[:,0]",
            "tail_tip_definition": "subject_shape.components.subject_body.tail_sample_xy[:,-1]",
            "frame_index_source": frame_index_source,
            "row_lineage_copied": copied,
            "row_lineage_missing": missing,
            "source_refs": source_refs,
            "algorithm_provenance": {
                "implementation": "independent_palette_compatible",
                "compatible_with": "megabouts.tracking_data.convert_tracking.compute_angles_from_keypoints",
                "copies_megabouts_code": False,
                "requires_megabouts_install": False,
            },
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
        stage=TAIL_POSTURE_VIEW_STAGE_NAME,
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
        parameters={
            "method": TAIL_POSTURE_VIEW_METHOD,
            "method_version": TAIL_POSTURE_VIEW_METHOD_VERSION,
            "view_family": str(view_family),
            "head_source": str(head_source),
            "keypoint_count": int(keypoint_count),
            "angle_convention": "megabouts_cumulative_segment_angle",
        },
        inputs={
            "source_subject_shape_run": shape_run_name,
            "source_refined_subject_masks_run": source_refined_run,
            "source_tail_kinematics_run": source_tail_kinematics_run,
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="tail_posture_view_run")
    return run_group


def _write_batch(run_group: zarr.Group, batch: TailPostureViewBatch) -> None:
    row_count = int(batch.valid.shape[0])
    keypoint_count = int(batch.tail_keypoints_xy.shape[1])
    angle_count = int(batch.tail_angle_rad.shape[1])
    _write_array(run_group, "valid", batch.valid.astype(bool), chunks=_metric_chunks(row_count))
    _write_array(
        run_group,
        "failure_reason_bytes",
        batch.failure_reason_bytes,
        chunks=_metric_chunks_lastdim(row_count, int(batch.failure_reason_bytes.shape[1])),
    )
    _write_array(run_group, "head_xy", batch.head_xy.astype(np.float32), chunks=_metric_chunks_lastdim(row_count, 2))
    _write_array(run_group, "head_yaw_rad", batch.head_yaw_rad.astype(np.float32), chunks=_metric_chunks(row_count))
    _write_array(
        run_group,
        "tail_keypoints_xy",
        batch.tail_keypoints_xy.astype(np.float32),
        chunks=_metric_chunks_3d(row_count, keypoint_count, 2),
    )
    _write_array(
        run_group,
        "tail_angle_rad",
        batch.tail_angle_rad.astype(np.float32),
        chunks=_metric_chunks_lastdim(row_count, angle_count),
    )
    _write_array(
        run_group,
        "tail_angle_deg",
        batch.tail_angle_deg.astype(np.float32),
        chunks=_metric_chunks_lastdim(row_count, angle_count),
    )


def write_tail_posture_view_run_group(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    view_family: str = DEFAULT_VIEW_FAMILY,
    head_source: str = "head_endpoint_xy",
    keypoint_count: int = DEFAULT_KEYPOINT_COUNT,
    source_tail_kinematics_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    stage_command: Optional[str] = None,
) -> dict[str, object]:
    """Write one tool-compatible tail posture view from subject-shape geometry."""

    if int(keypoint_count) < 2:
        raise ValueError("keypoint_count must be >= 2.")
    shape_run_name, shape_group, shape_tables = _resolve_subject_shape_tables(
        root,
        subject_shape_run,
        head_source=str(head_source),
    )
    sources, row_count = _read_sources(shape_tables, head_source=str(head_source))
    target_run = str(run_name or _default_run_name(str(view_family)))
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "tail_posture_view_run": target_run,
        "view_family": str(view_family),
        "source_subject_shape_run": shape_run_name,
        "source_tail_kinematics_run": source_tail_kinematics_run,
        "roi_count": int(row_count),
        "keypoint_count": int(keypoint_count),
        "angle_count": int(keypoint_count) - 1,
        "mutates_archive": not bool(dry_run),
    }
    if dry_run:
        return dict(_json_safe(summary))

    started = time.perf_counter()
    batch = compute_tail_posture_view_from_subject_shape_arrays(
        **sources,
        keypoint_count=int(keypoint_count),
    )
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    run_group = _prepare_run_group(
        root,
        target_run=target_run,
        shape_run_name=shape_run_name,
        shape_group=shape_group,
        row_count=row_count,
        view_family=str(view_family),
        head_source=str(head_source),
        keypoint_count=int(keypoint_count),
        source_tail_kinematics_run=source_tail_kinematics_run,
        stage_command=command,
        overwrite=overwrite,
    )
    _write_batch(run_group, batch)

    duration_seconds = float(time.perf_counter() - started)
    valid_count = int(np.count_nonzero(batch.valid))
    invalid_count = int(row_count - valid_count)
    reason_counts: dict[str, int] = {}
    for reason in np.asarray(batch.failure_reason, dtype=object).tolist():
        key = str(reason or "")
        reason_counts[key] = int(reason_counts.get(key, 0) + 1)

    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["rows_per_second"] = float(row_count / duration_seconds) if duration_seconds > 0.0 else math.inf
    run_group.attrs["valid_row_count"] = valid_count
    run_group.attrs["invalid_row_count"] = invalid_count
    run_group.attrs["failure_reason_counts"] = reason_counts
    parent = root["analysis"]["tail_posture_view_runs"]
    mark_run_complete(
        run_group,
        parent_group=parent,
        run_name=target_run,
        run_provenance=build_run_provenance_from_stage_record(
            run_group.attrs.get("provenance", {}),
            fallback_command=command,
        ),
    )
    if str(view_family):
        parent.attrs[f"latest_{view_family}"] = target_run

    summary.update(
        {
            "status": "updated",
            "valid_row_count": valid_count,
            "invalid_row_count": invalid_count,
            "failure_reason_counts": reason_counts,
            "duration_seconds": duration_seconds,
            "rows_per_second": run_group.attrs["rows_per_second"],
        }
    )
    return dict(_json_safe(summary))


def write_tail_posture_view_run(
    zarr_path: str | Path,
    *,
    subject_shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    view_family: str = DEFAULT_VIEW_FAMILY,
    head_source: str = "head_endpoint_xy",
    keypoint_count: int = DEFAULT_KEYPOINT_COUNT,
    source_tail_kinematics_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_tail_posture_view_run_group(
        root,
        subject_shape_run=subject_shape_run,
        run_name=run_name,
        view_family=view_family,
        head_source=head_source,
        keypoint_count=keypoint_count,
        source_tail_kinematics_run=source_tail_kinematics_run,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write analysis/tail_posture_view_runs from subject-shape tail geometry."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--subject-shape-run", help="analysis/subject_shape_runs/<run> to consume; defaults to latest.")
    parser.add_argument("--run-name", help="Target analysis/tail_posture_view_runs/<run>; defaults to timestamped.")
    parser.add_argument("--view-family", default=DEFAULT_VIEW_FAMILY, help="Tool/view family label.")
    parser.add_argument(
        "--head-source",
        default="head_endpoint_xy",
        choices=("head_endpoint_xy", "snout_tip_xy"),
        help="Subject-shape head point used to compute compatible cumulative segment angles.",
    )
    parser.add_argument(
        "--keypoint-count",
        type=int,
        default=DEFAULT_KEYPOINT_COUNT,
        help="Ordered tail keypoints from tail base to tail tip; 11 gives 10 Megabouts-compatible channels.",
    )
    parser.add_argument(
        "--source-tail-kinematics-run",
        help="Optional analysis/tail_kinematics_runs/<run> comparison source to record in attrs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target tail posture view run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without mutating the archive.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = write_tail_posture_view_run(
        args.zarr_path,
        subject_shape_run=args.subject_shape_run,
        run_name=args.run_name,
        view_family=str(args.view_family),
        head_source=str(args.head_source),
        keypoint_count=int(args.keypoint_count),
        source_tail_kinematics_run=args.source_tail_kinematics_run,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
