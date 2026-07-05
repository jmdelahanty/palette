"""Build frame-level tail kinematics from subject-shape tail geometry."""

from __future__ import annotations

import argparse
import json
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
from ..shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root
from .subject_shape_io import SubjectShapeRunTables, load_subject_shape_run_tables, resolve_subject_shape_run
from .subject_shape_spline import tail_sample_positions

TAIL_KINEMATICS_SCHEMA_ID = "analysis.tail_kinematics_runs"
TAIL_KINEMATICS_SCHEMA_VERSION = 1
TAIL_KINEMATICS_METHOD = "tail_metrics_from_subject_shape"
TAIL_KINEMATICS_METHOD_VERSION = 1
TAIL_KINEMATICS_STAGE_NAME = "analysis.tail_kinematics_runs"
SOURCE_TAIL_GEOMETRY_KIND = "subject_shape_bspline_tail_resample"
DEFAULT_TAIL_ANGLE_SAMPLE_COUNT = 10
REASON_BYTES_WIDTH = 64


@dataclass(frozen=True)
class TailKinematicsBatch:
    """Computed tail-kinematics arrays for one row-aligned batch."""

    tail_angle_sample_s: np.ndarray
    tail_angle_sample_xy: np.ndarray
    tail_angle_rad: np.ndarray
    tail_angle_deg: np.ndarray
    tail_tip_angle_rad: np.ndarray
    tail_tip_angle_deg: np.ndarray
    tail_lateral_deflection_px: np.ndarray
    tail_tip_lateral_deflection_px: np.ndarray
    max_abs_tail_angle_rad: np.ndarray
    max_abs_tail_angle_deg: np.ndarray
    tail_angle_rms_rad: np.ndarray
    tail_angle_rms_deg: np.ndarray
    integrated_abs_tail_angle_rad: np.ndarray
    tail_curvature_px_inv: np.ndarray
    max_abs_tail_curvature_px_inv: np.ndarray
    integrated_abs_tail_curvature: np.ndarray
    valid: np.ndarray
    failure_reason: np.ndarray
    failure_reason_bytes: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"tail_kinematics_{stamp}"


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


def _copy_optional_source_revision_snapshot(
    target_run: zarr.Group,
    shape_group: zarr.Group,
    *,
    shape_run_name: str,
) -> bool:
    source = shape_group.get("source_refined_subject_masks")
    if not isinstance(source, zarr.Group):
        target_run.attrs["source_refined_subject_masks_revision_snapshot"] = False
        return False

    target = target_run.require_group("source_refined_subject_masks")
    for key, value in source.attrs.items():
        target.attrs[str(key)] = _json_safe(value)
    target.attrs["copied_from_subject_shape_run"] = str(shape_run_name)
    target.attrs["snapshot_semantics"] = (
        "refined mask row revisions copied from the source subject-shape run used by this tail-kinematics run"
    )
    copied = []
    for name in ("row_revision", "row_revision_available"):
        arr = source.get(name)
        if arr is None:
            continue
        data = np.asarray(arr[:])
        _write_array(target, name, data, chunks=getattr(arr, "chunks", None))
        copied.append(name)
    target.attrs["copied_arrays"] = copied
    target_run.attrs["source_refined_subject_masks_revision_snapshot"] = bool(copied)
    return bool(copied)


def _read_optional_reason_labels(group: zarr.Group | Mapping[str, np.ndarray], name: str, row_count: int) -> np.ndarray:
    arr = group.get(name)
    if arr is None:
        return np.full((int(row_count),), "", dtype=object)
    data = np.asarray(arr[:] if hasattr(arr, "shape") and not isinstance(arr, np.ndarray) else arr)
    if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
        return decode_reason_bytes(data)
    return np.asarray(data, dtype=object).reshape(-1)


def _normalize_vectors(vectors_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors_xy, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1)
    normalized = np.full(vectors.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms > 1e-12)
    normalized[valid] = vectors[valid] / norms[valid, None]
    return normalized, valid


def _interp_rows_2d(
    source_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    row_valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    out = np.full((int(data.shape[0]), int(target.shape[0]), int(data.shape[2])), np.nan, dtype=np.float32)
    for row_idx in range(int(data.shape[0])):
        if not bool(row_valid[row_idx]):
            continue
        row = data[row_idx]
        if not np.all(np.isfinite(row)):
            continue
        for dim_idx in range(int(data.shape[2])):
            out[row_idx, :, dim_idx] = np.interp(target, source, row[:, dim_idx]).astype(np.float32)
    return out


def _interp_rows_1d(
    source_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    row_valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    out = np.full((int(data.shape[0]), int(target.shape[0])), np.nan, dtype=np.float32)
    for row_idx in range(int(data.shape[0])):
        if not bool(row_valid[row_idx]):
            continue
        row = data[row_idx]
        if not np.all(np.isfinite(row)):
            continue
        out[row_idx, :] = np.interp(target, source, row).astype(np.float32)
    return out


def compute_tail_kinematics_from_subject_shape_arrays(
    *,
    source_tail_sample_s: np.ndarray,
    tail_sample_xy: np.ndarray,
    tail_tangent_xy: np.ndarray,
    tail_curvature_px_inv: np.ndarray,
    tail_sample_valid: np.ndarray,
    bspline_valid: np.ndarray,
    tail_base_xy: np.ndarray,
    body_forward_axis_xy: np.ndarray,
    body_left_axis_xy: np.ndarray,
    body_frame_valid: np.ndarray,
    tail_sample_failure_reason: Optional[Sequence[object]] = None,
    bspline_failure_reason: Optional[Sequence[object]] = None,
    body_frame_failure_reason: Optional[Sequence[object]] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
) -> TailKinematicsBatch:
    """Compute signed body-frame tail angles from subject-shape tail samples.

    The source tangents are expected to point from tail base toward tail tip.
    Angles are measured relative to the caudal body axis (-forward), positive
    toward anatomical left.
    """

    source_s = np.asarray(source_tail_sample_s, dtype=np.float64).reshape(-1)
    if source_s.ndim != 1 or int(source_s.shape[0]) < 2:
        raise ValueError("source_tail_sample_s must contain at least two positions.")
    if np.any(~np.isfinite(source_s)) or np.any(np.diff(source_s) <= 0.0):
        raise ValueError("source_tail_sample_s must be finite and strictly increasing.")

    xy = np.asarray(tail_sample_xy, dtype=np.float64)
    tangent = np.asarray(tail_tangent_xy, dtype=np.float64)
    curvature = np.asarray(tail_curvature_px_inv, dtype=np.float64)
    if xy.ndim != 3 or int(xy.shape[2]) != 2:
        raise ValueError("tail_sample_xy must have shape (N, S, 2).")
    if tangent.shape != xy.shape:
        raise ValueError("tail_tangent_xy must have the same shape as tail_sample_xy.")
    if curvature.shape != xy.shape[:2]:
        raise ValueError("tail_curvature_px_inv must have shape (N, S).")
    if int(xy.shape[1]) != int(source_s.shape[0]):
        raise ValueError("source_tail_sample_s length must match tail sample arrays.")

    row_count = int(xy.shape[0])
    source_valid = (
        np.asarray(tail_sample_valid, dtype=bool).reshape(-1)
        & np.asarray(bspline_valid, dtype=bool).reshape(-1)
        & np.asarray(body_frame_valid, dtype=bool).reshape(-1)
    )
    if source_valid.shape[0] != row_count:
        raise ValueError("validity arrays must have the same row count as tail arrays.")

    target_s = tail_sample_positions(int(tail_angle_sample_count)).astype(np.float32)
    sampled_xy = _interp_rows_2d(source_s, xy, target_s, source_valid)
    sampled_tangent = _interp_rows_2d(source_s, tangent, target_s, source_valid)
    sampled_curvature = _interp_rows_1d(source_s, curvature, target_s, source_valid)
    sampled_tangent64, tangent_norm_valid = _normalize_vectors(sampled_tangent)

    forward, forward_valid = _normalize_vectors(np.asarray(body_forward_axis_xy, dtype=np.float64))
    left, left_valid = _normalize_vectors(np.asarray(body_left_axis_xy, dtype=np.float64))
    tail_base = np.asarray(tail_base_xy, dtype=np.float64)
    if forward.shape != (row_count, 2) or left.shape != (row_count, 2) or tail_base.shape != (row_count, 2):
        raise ValueError("body-frame and tail-base arrays must have shape (N, 2).")

    angle_rad = np.full((row_count, int(target_s.shape[0])), np.nan, dtype=np.float32)
    lateral_px = np.full_like(angle_rad, np.nan)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = np.full((row_count,), "ok", dtype=object)

    tail_reasons = (
        np.asarray(tail_sample_failure_reason, dtype=object).reshape(-1)
        if tail_sample_failure_reason is not None
        else np.full((row_count,), "tail_sample_invalid", dtype=object)
    )
    bspline_reasons = (
        np.asarray(bspline_failure_reason, dtype=object).reshape(-1)
        if bspline_failure_reason is not None
        else np.full((row_count,), "bspline_invalid", dtype=object)
    )
    body_reasons = (
        np.asarray(body_frame_failure_reason, dtype=object).reshape(-1)
        if body_frame_failure_reason is not None
        else np.full((row_count,), "body_frame_invalid", dtype=object)
    )

    caudal = -forward
    for row_idx in range(row_count):
        if not bool(body_frame_valid[row_idx]) or not bool(forward_valid[row_idx]) or not bool(left_valid[row_idx]):
            reason = str(body_reasons[row_idx] or "body_frame_invalid")
            reasons[row_idx] = reason if reason != "ok" else "body_frame_invalid"
            continue
        if not bool(bspline_valid[row_idx]):
            reason = str(bspline_reasons[row_idx] or "bspline_invalid")
            reasons[row_idx] = reason if reason != "ok" else "bspline_invalid"
            continue
        if not bool(tail_sample_valid[row_idx]):
            reason = str(tail_reasons[row_idx] or "tail_sample_invalid")
            reasons[row_idx] = reason if reason != "ok" else "tail_sample_invalid"
            continue
        if not np.all(tangent_norm_valid[row_idx]) or not np.all(np.isfinite(sampled_xy[row_idx])):
            reasons[row_idx] = "tail_geometry_nonfinite"
            continue
        if not np.all(np.isfinite(sampled_curvature[row_idx])) or not np.all(np.isfinite(tail_base[row_idx])):
            reasons[row_idx] = "tail_geometry_nonfinite"
            continue

        tangents = sampled_tangent64[row_idx]
        dot_left = tangents @ left[row_idx]
        dot_caudal = tangents @ caudal[row_idx]
        angles = np.arctan2(dot_left, dot_caudal)
        offsets = np.asarray(sampled_xy[row_idx], dtype=np.float64) - tail_base[row_idx][None, :]
        lateral = offsets @ left[row_idx]
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(lateral)):
            reasons[row_idx] = "tail_geometry_nonfinite"
            continue

        angle_rad[row_idx, :] = angles.astype(np.float32)
        lateral_px[row_idx, :] = lateral.astype(np.float32)
        valid[row_idx] = True

    angle_deg = np.rad2deg(angle_rad).astype(np.float32)
    tail_tip_angle_rad = angle_rad[:, -1].astype(np.float32)
    tail_tip_angle_deg = angle_deg[:, -1].astype(np.float32)
    tail_tip_lateral_px = lateral_px[:, -1].astype(np.float32)

    max_abs_angle_rad = np.full((row_count,), np.nan, dtype=np.float32)
    angle_rms_rad = np.full((row_count,), np.nan, dtype=np.float32)
    max_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_angle = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    for row_idx in np.flatnonzero(valid):
        max_abs_angle_rad[row_idx] = np.float32(float(np.max(np.abs(angle_rad[row_idx]))))
        angle_rms_rad[row_idx] = np.float32(float(np.sqrt(np.mean(np.square(angle_rad[row_idx])))))
        max_abs_curvature[row_idx] = np.float32(float(np.max(np.abs(sampled_curvature[row_idx]))))
        integrated_abs_angle[row_idx] = np.float32(
            np.trapezoid(np.abs(angle_rad[row_idx]).astype(np.float64), target_s.astype(np.float64))
        )
        integrated_abs_curvature[row_idx] = np.float32(
            np.trapezoid(np.abs(sampled_curvature[row_idx]).astype(np.float64), target_s.astype(np.float64))
        )

    return TailKinematicsBatch(
        tail_angle_sample_s=target_s,
        tail_angle_sample_xy=sampled_xy.astype(np.float32),
        tail_angle_rad=angle_rad,
        tail_angle_deg=angle_deg,
        tail_tip_angle_rad=tail_tip_angle_rad,
        tail_tip_angle_deg=tail_tip_angle_deg,
        tail_lateral_deflection_px=lateral_px.astype(np.float32),
        tail_tip_lateral_deflection_px=tail_tip_lateral_px,
        max_abs_tail_angle_rad=max_abs_angle_rad,
        max_abs_tail_angle_deg=np.rad2deg(max_abs_angle_rad).astype(np.float32),
        tail_angle_rms_rad=angle_rms_rad,
        tail_angle_rms_deg=np.rad2deg(angle_rms_rad).astype(np.float32),
        integrated_abs_tail_angle_rad=integrated_abs_angle,
        tail_curvature_px_inv=sampled_curvature.astype(np.float32),
        max_abs_tail_curvature_px_inv=max_abs_curvature,
        integrated_abs_tail_curvature=integrated_abs_curvature,
        valid=valid,
        failure_reason=reasons,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _resolve_subject_shape_tables(
    root: zarr.Group,
    shape_run: Optional[str],
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
                "tail_tangent_xy",
                "tail_curvature_px_inv",
                "tail_sample_valid",
                "bspline_valid",
                "tail_base_xy",
                "tail_sample_failure_reason_bytes",
                "bspline_failure_reason_bytes",
            )
        },
        include_body_frame=True,
        include_row_index=True,
        include_source_refined_subject_masks=True,
    )
    return run_name, shape_group, tables


def _read_tail_kinematics_sources(shape_tables: SubjectShapeRunTables) -> tuple[dict[str, np.ndarray], int]:
    body = shape_tables.require_component("subject_body")

    tail_xy = np.asarray(body.require_array("tail_sample_xy"), dtype=np.float32)
    row_count = int(tail_xy.shape[0])
    sources = {
        "source_tail_sample_s": np.asarray(body.require_array("tail_sample_s"), dtype=np.float32),
        "tail_sample_xy": tail_xy,
        "tail_tangent_xy": np.asarray(body.require_array("tail_tangent_xy"), dtype=np.float32),
        "tail_curvature_px_inv": np.asarray(body.require_array("tail_curvature_px_inv"), dtype=np.float32),
        "tail_sample_valid": np.asarray(body.require_array("tail_sample_valid"), dtype=bool),
        "bspline_valid": np.asarray(body.require_array("bspline_valid"), dtype=bool),
        "tail_base_xy": np.asarray(body.require_array("tail_base_xy"), dtype=np.float32),
        "body_forward_axis_xy": np.asarray(
            shape_tables.require_body_frame_array("forward_axis_xy"),
            dtype=np.float32,
        ),
        "body_left_axis_xy": np.asarray(
            shape_tables.require_body_frame_array("left_axis_xy"),
            dtype=np.float32,
        ),
        "body_frame_valid": np.asarray(shape_tables.require_body_frame_array("valid"), dtype=bool),
        "tail_sample_failure_reason": _read_optional_reason_labels(
            body.arrays, "tail_sample_failure_reason_bytes", row_count
        ),
        "bspline_failure_reason": _read_optional_reason_labels(body.arrays, "bspline_failure_reason_bytes", row_count),
        "body_frame_failure_reason": _read_optional_reason_labels(
            shape_tables.body_frame,
            "failure_reason_bytes",
            row_count,
        ),
    }
    return sources, row_count


def _prepare_tail_kinematics_run(
    root: zarr.Group,
    *,
    target_run: str,
    shape_run_name: str,
    shape_group: zarr.Group,
    row_count: int,
    tail_angle_sample_count: int,
    stage_command: str,
    overwrite: bool,
) -> zarr.Group:
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "tail_kinematics_runs")
    if target_run in parent:
        if not overwrite:
            raise ValueError(
                f"analysis/tail_kinematics_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del parent[target_run]
    run_group = parent.create_group(target_run)
    mark_run_started(run_group, run_name=target_run, stage="tail_kinematics")
    _set_reason_bytes_attrs(run_group)

    source_refined_run = shape_group.attrs.get("source_refined_subject_masks_run")
    body = shape_group["components"]["subject_body"]
    created = _utc_now()

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

    run_group.attrs.update(
        {
            "schema_id": TAIL_KINEMATICS_SCHEMA_ID,
            "schema_version": TAIL_KINEMATICS_SCHEMA_VERSION,
            "method": TAIL_KINEMATICS_METHOD,
            "method_version": TAIL_KINEMATICS_METHOD_VERSION,
            "created_at_utc": created,
            "created_utc": created,
            "row_axis": "roi_rows",
            "source_subject_shape_run": str(shape_run_name),
            "source_subject_shape_path": f"analysis/subject_shape_runs/{shape_run_name}",
            "source_refined_subject_masks_run": str(source_refined_run) if source_refined_run is not None else None,
            "source_tail_geometry_kind": SOURCE_TAIL_GEOMETRY_KIND,
            "body_frame_convention": shape_group.attrs.get("body_frame_schema_id", "fish_anatomical_body_frame"),
            "body_frame_source": f"analysis/subject_shape_runs/{shape_run_name}/body_frame",
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
            "tail_angle_units_primary": "rad",
            "tail_sample_domain": "tail_segment_normalized_arclength",
            "tail_angle_sample_count": int(tail_angle_sample_count),
            "source_geometry_tail_sample_count": int(body.attrs.get("tail_sample_count", body["tail_sample_xy"].shape[1])),
            "curvature_source": "subject_shape.tail_curvature_px_inv",
            "frame_index_source": frame_index_source,
            "row_lineage_copied": copied,
            "row_lineage_missing": missing,
            "source_refs": {
                "subject_shape_run": f"analysis/subject_shape_runs/{shape_run_name}",
                "subject_shape_body_component": f"analysis/subject_shape_runs/{shape_run_name}/components/subject_body",
                "subject_shape_body_frame": f"analysis/subject_shape_runs/{shape_run_name}/body_frame",
            },
        }
    )
    _copy_optional_source_revision_snapshot(
        run_group,
        shape_group,
        shape_run_name=shape_run_name,
    )

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage=TAIL_KINEMATICS_STAGE_NAME,
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
            "method": TAIL_KINEMATICS_METHOD,
            "method_version": TAIL_KINEMATICS_METHOD_VERSION,
            "tail_angle_sample_count": int(tail_angle_sample_count),
            "tail_angle_reference_axis": "caudal_axis=-forward_axis",
            "tail_angle_positive_direction": "anatomical_left",
            "source_tail_geometry_kind": SOURCE_TAIL_GEOMETRY_KIND,
        },
        inputs={
            "source_subject_shape_run": shape_run_name,
            "source_refined_subject_masks_run": source_refined_run,
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="tail_kinematics_run")
    return run_group


def _write_tail_kinematics_batch(run_group: zarr.Group, batch: TailKinematicsBatch) -> None:
    row_count = int(batch.valid.shape[0])
    sample_count = int(batch.tail_angle_sample_s.shape[0])
    _write_array(run_group, "valid", batch.valid.astype(bool), chunks=_metric_chunks(row_count))
    _write_array(
        run_group,
        "failure_reason_bytes",
        batch.failure_reason_bytes,
        chunks=_metric_chunks_lastdim(row_count, int(batch.failure_reason_bytes.shape[1])),
    )
    _write_array(run_group, "tail_angle_sample_s", batch.tail_angle_sample_s, chunks=(sample_count,))
    _write_array(
        run_group,
        "tail_angle_sample_xy",
        batch.tail_angle_sample_xy,
        chunks=_metric_chunks_3d(row_count, sample_count, 2),
    )
    _write_array(run_group, "tail_angle_rad", batch.tail_angle_rad, chunks=_metric_chunks_lastdim(row_count, sample_count))
    _write_array(run_group, "tail_angle_deg", batch.tail_angle_deg, chunks=_metric_chunks_lastdim(row_count, sample_count))
    _write_array(run_group, "tail_tip_angle_rad", batch.tail_tip_angle_rad, chunks=_metric_chunks(row_count))
    _write_array(run_group, "tail_tip_angle_deg", batch.tail_tip_angle_deg, chunks=_metric_chunks(row_count))
    _write_array(
        run_group,
        "tail_lateral_deflection_px",
        batch.tail_lateral_deflection_px,
        chunks=_metric_chunks_lastdim(row_count, sample_count),
    )
    _write_array(
        run_group,
        "tail_tip_lateral_deflection_px",
        batch.tail_tip_lateral_deflection_px,
        chunks=_metric_chunks(row_count),
    )
    _write_array(run_group, "max_abs_tail_angle_rad", batch.max_abs_tail_angle_rad, chunks=_metric_chunks(row_count))
    _write_array(run_group, "max_abs_tail_angle_deg", batch.max_abs_tail_angle_deg, chunks=_metric_chunks(row_count))
    _write_array(run_group, "tail_angle_rms_rad", batch.tail_angle_rms_rad, chunks=_metric_chunks(row_count))
    _write_array(run_group, "tail_angle_rms_deg", batch.tail_angle_rms_deg, chunks=_metric_chunks(row_count))
    _write_array(
        run_group,
        "integrated_abs_tail_angle_rad",
        batch.integrated_abs_tail_angle_rad,
        chunks=_metric_chunks(row_count),
    )
    _write_array(
        run_group,
        "tail_curvature_px_inv",
        batch.tail_curvature_px_inv,
        chunks=_metric_chunks_lastdim(row_count, sample_count),
    )
    _write_array(
        run_group,
        "max_abs_tail_curvature_px_inv",
        batch.max_abs_tail_curvature_px_inv,
        chunks=_metric_chunks(row_count),
    )
    _write_array(
        run_group,
        "integrated_abs_tail_curvature",
        batch.integrated_abs_tail_curvature,
        chunks=_metric_chunks(row_count),
    )


def write_tail_kinematics_run_group(
    root: zarr.Group,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    overwrite: bool = False,
    dry_run: bool = False,
    stage_command: Optional[str] = None,
) -> dict[str, object]:
    """Write one tail-kinematics run from an existing subject-shape run."""

    if int(tail_angle_sample_count) < 2:
        raise ValueError("tail_angle_sample_count must be >= 2.")
    shape_run_name, shape_group, shape_tables = _resolve_subject_shape_tables(root, shape_run)
    sources, row_count = _read_tail_kinematics_sources(shape_tables)
    target_run = str(run_name or _default_run_name())
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "tail_kinematics_run": target_run,
        "source_subject_shape_run": shape_run_name,
        "source_refined_subject_masks_run": shape_group.attrs.get("source_refined_subject_masks_run"),
        "roi_count": int(row_count),
        "tail_angle_sample_count": int(tail_angle_sample_count),
        "mutates_archive": not bool(dry_run),
    }
    if dry_run:
        return dict(_json_safe(summary))

    started = time.perf_counter()
    batch = compute_tail_kinematics_from_subject_shape_arrays(
        **sources,
        tail_angle_sample_count=int(tail_angle_sample_count),
    )
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    run_group = _prepare_tail_kinematics_run(
        root,
        target_run=target_run,
        shape_run_name=shape_run_name,
        shape_group=shape_group,
        row_count=row_count,
        tail_angle_sample_count=int(tail_angle_sample_count),
        stage_command=command,
        overwrite=overwrite,
    )
    _write_tail_kinematics_batch(run_group, batch)

    duration_seconds = float(time.perf_counter() - started)
    valid_count = int(np.count_nonzero(batch.valid))
    invalid_count = int(row_count - valid_count)
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["rows_per_second"] = float(row_count / duration_seconds) if duration_seconds > 0.0 else float("inf")
    run_group.attrs["valid_row_count"] = valid_count
    run_group.attrs["invalid_row_count"] = invalid_count
    mark_run_complete(
        run_group,
        parent_group=root["analysis"]["tail_kinematics_runs"],
        run_name=target_run,
    )

    reason_counts: dict[str, int] = {}
    for reason in np.asarray(batch.failure_reason, dtype=object).tolist():
        key = str(reason or "")
        reason_counts[key] = int(reason_counts.get(key, 0) + 1)
    run_group.attrs["failure_reason_counts"] = reason_counts

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


def write_tail_kinematics_run(
    zarr_path: str | Path,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_tail_kinematics_run_group(
        root,
        shape_run=shape_run,
        run_name=run_name,
        tail_angle_sample_count=tail_angle_sample_count,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write analysis/tail_kinematics_runs from subject-shape tail geometry."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--shape-run", help="analysis/subject_shape_runs/<run> to consume; defaults to latest.")
    parser.add_argument("--run-name", help="Target analysis/tail_kinematics_runs/<run>; defaults to timestamped.")
    parser.add_argument(
        "--tail-angle-sample-count",
        type=int,
        default=DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
        help="Low-dimensional behavior-facing tail samples from base to tip.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target tail-kinematics run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without mutating the archive.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = write_tail_kinematics_run(
        args.zarr_path,
        shape_run=args.shape_run,
        run_name=args.run_name,
        tail_angle_sample_count=int(args.tail_angle_sample_count),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
