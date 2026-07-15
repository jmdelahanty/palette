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
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.zarr_run_completion import mark_run_complete, mark_run_failed, mark_run_started, require_runs_parent
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root
from .subject_shape_io import SubjectShapeIOError, resolve_subject_shape_run
from .subject_shape_spline import tail_sample_positions

TAIL_KINEMATICS_SCHEMA_ID = "analysis.tail_kinematics_runs"
TAIL_KINEMATICS_SCHEMA_VERSION = 1
TAIL_KINEMATICS_METHOD = "tail_metrics_from_subject_shape"
TAIL_KINEMATICS_METHOD_VERSION = 1
TAIL_KINEMATICS_COMPUTE_KERNEL = "vectorized_shared_grid_v1"
TAIL_KINEMATICS_STAGE_NAME = "analysis.tail_kinematics_runs"
SOURCE_TAIL_GEOMETRY_KIND = "subject_shape_bspline_tail_resample"
DEFAULT_TAIL_ANGLE_SAMPLE_COUNT = 10
DEFAULT_BLOCK_ROWS = 16_384
REASON_BYTES_WIDTH = 64
ROW_LINEAGE_NAMES = (
    "frame_indices",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "source_crop_row_ids",
    "instance_key",
)
SUBJECT_SHAPE_BODY_ARRAY_NAMES = (
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
SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES = (
    "forward_axis_xy",
    "left_axis_xy",
    "valid",
    "failure_reason_bytes",
)
SOURCE_REVISION_ARRAY_NAMES = ("row_revision", "row_revision_available")


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


@dataclass(frozen=True)
class TailKinematicsSources:
    """Lazy source-array handles for one subject-shape run."""

    source_tail_sample_s: np.ndarray
    arrays: Mapping[str, Any]
    reason_arrays: Mapping[str, Any | None]
    row_count: int
    source_sample_count: int


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


def _create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    shards: Sequence[int] | None = None,
    fill_value: object | None = None,
) -> Any:
    if name in group:
        del group[name]
    kwargs: dict[str, object] = {
        "shape": tuple(int(value) for value in shape),
        "dtype": dtype,
        "chunks": tuple(int(value) for value in chunks),
        "overwrite": True,
    }
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    if shards is not None:
        kwargs["shards"] = tuple(int(value) for value in shards)
    return group.create_array(name, **kwargs)


def _source_chunks(source: Any, shape: Sequence[int]) -> tuple[int, ...]:
    raw_chunks = getattr(source, "chunks", None)
    dims = tuple(int(value) for value in shape)
    if raw_chunks is None:
        if not dims:
            return ()
        return tuple(max(1, min(int(dim), 256)) for dim in dims)
    if isinstance(raw_chunks, int):
        chunks = (int(raw_chunks),)
    else:
        chunks = tuple(int(value) for value in raw_chunks)
    if len(chunks) != len(dims):
        raise ValueError(f"Source chunks {chunks!r} do not match array shape {dims!r}.")
    return tuple(max(1, min(int(chunk), int(dim))) if int(dim) > 0 else 1 for chunk, dim in zip(chunks, dims))


def _effective_block_rows(*, row_count: int, requested_block_rows: int) -> int:
    requested = int(requested_block_rows)
    if requested <= 0:
        raise ValueError("block_rows must be positive.")
    output_row_chunk = int(_metric_chunks(int(row_count))[0])
    aligned = max(output_row_chunk, ((requested + output_row_chunk - 1) // output_row_chunk) * output_row_chunk)
    return min(aligned, int(row_count)) if int(row_count) > 0 else aligned


def _output_shards(
    chunks: Sequence[int],
    *,
    shard_rows: int | None,
    dtype: object,
) -> tuple[int, ...] | None:
    if shard_rows is None:
        return None
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        return None
    if kind in {"O", "S", "U"}:
        return None
    chunk_shape = tuple(int(value) for value in chunks)
    if not chunk_shape:
        return None
    outer_rows = max(
        int(chunk_shape[0]),
        ((int(shard_rows) + int(chunk_shape[0]) - 1) // int(chunk_shape[0])) * int(chunk_shape[0]),
    )
    return (outer_rows, *chunk_shape[1:])


def _iter_row_slices(row_count: int, block_rows: int) -> Sequence[slice]:
    return tuple(
        slice(start, min(int(row_count), start + int(block_rows)))
        for start in range(0, int(row_count), int(block_rows))
    )


def _copy_array_bounded(
    target_group: zarr.Group,
    name: str,
    source: Any,
    *,
    block_rows: int,
    row_aligned_count: int | None = None,
    shard_rows: int | None = None,
) -> Any:
    shape = tuple(int(value) for value in source.shape)
    if row_aligned_count is not None:
        if not shape or int(shape[0]) != int(row_aligned_count):
            raise ValueError(
                f"Source array {name!r} has shape {shape!r}; expected first axis {int(row_aligned_count)}."
            )
    chunks = _source_chunks(source, shape)
    target = _create_array(
        target_group,
        name,
        shape=shape,
        dtype=source.dtype,
        chunks=chunks,
        shards=_output_shards(chunks, shard_rows=shard_rows, dtype=source.dtype),
    )
    if not shape:
        target[...] = np.asarray(source[...])
    elif row_aligned_count is not None:
        for row_slice in _iter_row_slices(int(row_aligned_count), int(block_rows)):
            target[row_slice] = np.asarray(source[row_slice])
    else:
        target[:] = np.asarray(source[:])
    return target


def _copy_optional_source_revision_snapshot(
    target_run: zarr.Group,
    shape_group: zarr.Group,
    *,
    shape_run_name: str,
    row_count: int,
    block_rows: int,
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
        _copy_array_bounded(
            target,
            name,
            arr,
            block_rows=int(block_rows),
            row_aligned_count=int(row_count) if name == "row_revision" else None,
            shard_rows=int(block_rows) if name == "row_revision" else None,
        )
        copied.append(name)
    target.attrs["copied_arrays"] = copied
    target_run.attrs["source_refined_subject_masks_revision_snapshot"] = bool(copied)
    return bool(copied)


def _read_optional_reason_labels(arr: Any | None, row_slice: slice) -> np.ndarray:
    row_count = int((row_slice.stop or 0) - (row_slice.start or 0))
    if arr is None:
        return np.full((int(row_count),), "", dtype=object)
    data = np.asarray(arr[row_slice])
    if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):
        decoded = decode_reason_bytes(data)
    else:
        decoded = np.asarray(data, dtype=object).reshape(-1)
    if int(decoded.shape[0]) != int(row_count):
        raise ValueError(f"Failure-reason slice has {decoded.shape[0]} rows; expected {row_count}.")
    return decoded


def _normalize_vectors(vectors_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors_xy, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1)
    normalized = np.full(vectors.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(norms) & (norms > 1e-12)
    normalized[valid] = vectors[valid] / norms[valid, None]
    return normalized, valid


def _interpolation_plan(source_s: np.ndarray, target_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared-grid interpolation indices and weights matching ``np.interp`` endpoints."""

    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    insertion = np.searchsorted(source, target, side="left")
    right = np.clip(insertion, 0, int(source.shape[0]) - 1).astype(np.intp)
    left = np.clip(right - 1, 0, int(source.shape[0]) - 1).astype(np.intp)

    exact = (insertion < int(source.shape[0])) & (source[right] == target)
    left[exact] = right[exact]
    below = target <= source[0]
    above = target >= source[-1]
    left[below] = 0
    right[below] = 0
    left[above] = int(source.shape[0]) - 1
    right[above] = int(source.shape[0]) - 1

    weights = np.zeros(target.shape, dtype=np.float64)
    interpolated = left != right
    weights[interpolated] = (
        (target[interpolated] - source[left[interpolated]])
        / (source[right[interpolated]] - source[left[interpolated]])
    )
    return left, right, weights


def _interp_rows_2d(
    source_s: np.ndarray,
    values: np.ndarray,
    target_s: np.ndarray,
    row_valid: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("2D row interpolation values must have shape (N, S, D).")
    valid_rows = np.asarray(row_valid, dtype=bool).reshape(-1)
    if int(valid_rows.shape[0]) != int(data.shape[0]):
        raise ValueError("row_valid must have the same row count as interpolation values.")
    left, right, weights = _interpolation_plan(source, target)
    lower = data[:, left, :]
    upper = data[:, right, :]
    values_interp = lower + (upper - lower) * weights[None, :, None]
    eligible = valid_rows & np.all(np.isfinite(data), axis=(1, 2))
    out = np.full((int(data.shape[0]), int(target.shape[0]), int(data.shape[2])), np.nan, dtype=np.float32)
    out[eligible] = values_interp[eligible].astype(np.float32)
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
    if data.ndim != 2:
        raise ValueError("1D row interpolation values must have shape (N, S).")
    valid_rows = np.asarray(row_valid, dtype=bool).reshape(-1)
    if int(valid_rows.shape[0]) != int(data.shape[0]):
        raise ValueError("row_valid must have the same row count as interpolation values.")
    left, right, weights = _interpolation_plan(source, target)
    lower = data[:, left]
    upper = data[:, right]
    values_interp = lower + (upper - lower) * weights[None, :]
    eligible = valid_rows & np.all(np.isfinite(data), axis=1)
    out = np.full((int(data.shape[0]), int(target.shape[0])), np.nan, dtype=np.float32)
    out[eligible] = values_interp[eligible].astype(np.float32)
    return out


def _reason_values(
    values: Optional[Sequence[object]],
    *,
    row_count: int,
    fallback: str,
) -> np.ndarray:
    if values is None:
        return np.full((int(row_count),), str(fallback), dtype=object)
    reasons = np.asarray(values, dtype=object).reshape(-1)
    if int(reasons.shape[0]) != int(row_count):
        raise ValueError(f"Failure-reason array has {reasons.shape[0]} rows; expected {row_count}.")
    return reasons


def _assign_failure_reasons(
    target: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray,
    *,
    fallback: str,
) -> None:
    for row_idx in np.flatnonzero(mask):
        reason = str(source[row_idx] or fallback)
        target[row_idx] = reason if reason != "ok" else fallback


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
    toward anatomical left. All dense numerical operations are vectorized over
    the bounded input batch; failure-label normalization remains sparse.
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
    tail_valid = np.asarray(tail_sample_valid, dtype=bool).reshape(-1)
    spline_valid = np.asarray(bspline_valid, dtype=bool).reshape(-1)
    body_valid = np.asarray(body_frame_valid, dtype=bool).reshape(-1)
    if any(int(values.shape[0]) != row_count for values in (tail_valid, spline_valid, body_valid)):
        raise ValueError("validity arrays must have the same row count as tail arrays.")
    source_valid = tail_valid & spline_valid & body_valid

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

    tail_reasons = _reason_values(
        tail_sample_failure_reason,
        row_count=row_count,
        fallback="tail_sample_invalid",
    )
    bspline_reasons = _reason_values(
        bspline_failure_reason,
        row_count=row_count,
        fallback="bspline_invalid",
    )
    body_reasons = _reason_values(
        body_frame_failure_reason,
        row_count=row_count,
        fallback="body_frame_invalid",
    )

    caudal = -forward
    remaining = np.ones((row_count,), dtype=bool)
    body_invalid = remaining & ~(body_valid & forward_valid & left_valid)
    _assign_failure_reasons(
        reasons,
        body_invalid,
        body_reasons,
        fallback="body_frame_invalid",
    )
    remaining &= ~body_invalid

    spline_invalid = remaining & ~spline_valid
    _assign_failure_reasons(
        reasons,
        spline_invalid,
        bspline_reasons,
        fallback="bspline_invalid",
    )
    remaining &= ~spline_invalid

    tail_invalid = remaining & ~tail_valid
    _assign_failure_reasons(
        reasons,
        tail_invalid,
        tail_reasons,
        fallback="tail_sample_invalid",
    )
    remaining &= ~tail_invalid

    geometry_finite = (
        np.all(tangent_norm_valid, axis=1)
        & np.all(np.isfinite(sampled_xy), axis=(1, 2))
        & np.all(np.isfinite(sampled_curvature), axis=1)
        & np.all(np.isfinite(tail_base), axis=1)
    )
    geometry_invalid = remaining & ~geometry_finite
    reasons[geometry_invalid] = "tail_geometry_nonfinite"
    remaining &= ~geometry_invalid

    dot_left = np.sum(sampled_tangent64 * left[:, None, :], axis=-1)
    dot_caudal = np.sum(sampled_tangent64 * caudal[:, None, :], axis=-1)
    angles = np.arctan2(dot_left, dot_caudal)
    offsets = np.asarray(sampled_xy, dtype=np.float64) - tail_base[:, None, :]
    lateral = np.sum(offsets * left[:, None, :], axis=-1)
    calculation_finite = np.all(np.isfinite(angles), axis=1) & np.all(np.isfinite(lateral), axis=1)
    calculation_invalid = remaining & ~calculation_finite
    reasons[calculation_invalid] = "tail_geometry_nonfinite"
    valid = remaining & calculation_finite
    angle_rad[valid] = angles[valid].astype(np.float32)
    lateral_px[valid] = lateral[valid].astype(np.float32)

    angle_deg = np.rad2deg(angle_rad).astype(np.float32)
    tail_tip_angle_rad = angle_rad[:, -1].astype(np.float32)
    tail_tip_angle_deg = angle_deg[:, -1].astype(np.float32)
    tail_tip_lateral_px = lateral_px[:, -1].astype(np.float32)

    max_abs_angle_rad = np.full((row_count,), np.nan, dtype=np.float32)
    angle_rms_rad = np.full((row_count,), np.nan, dtype=np.float32)
    max_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_angle = np.full((row_count,), np.nan, dtype=np.float32)
    integrated_abs_curvature = np.full((row_count,), np.nan, dtype=np.float32)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size:
        valid_angles = angle_rad[valid_rows]
        valid_curvature = sampled_curvature[valid_rows]
        max_abs_angle_rad[valid_rows] = np.max(np.abs(valid_angles), axis=1).astype(np.float32)
        angle_rms_rad[valid_rows] = np.sqrt(np.mean(np.square(valid_angles), axis=1)).astype(np.float32)
        max_abs_curvature[valid_rows] = np.max(np.abs(valid_curvature), axis=1).astype(np.float32)
        integrated_abs_angle[valid_rows] = np.trapezoid(
            np.abs(valid_angles).astype(np.float64),
            target_s.astype(np.float64),
            axis=1,
        ).astype(np.float32)
        integrated_abs_curvature[valid_rows] = np.trapezoid(
            np.abs(valid_curvature).astype(np.float64),
            target_s.astype(np.float64),
            axis=1,
        ).astype(np.float32)

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


def _require_group(parent: zarr.Group, name: str, *, path: str) -> zarr.Group:
    value = parent.get(name)
    if not isinstance(value, zarr.Group):
        raise SubjectShapeIOError(f"{path}/{name} is missing or is not a Zarr group.")
    return value


def _require_array_handle(parent: zarr.Group, name: str, *, path: str) -> Any:
    value = parent.get(name)
    if value is None or not hasattr(value, "shape") or not hasattr(value, "__getitem__"):
        raise SubjectShapeIOError(f"{path}/{name} is missing or is not a readable Zarr array.")
    return value


def _validate_source_shape(name: str, source: Any, expected: Sequence[int]) -> None:
    actual = tuple(int(value) for value in source.shape)
    wanted = tuple(int(value) for value in expected)
    if actual != wanted:
        raise SubjectShapeIOError(f"{name} has shape {actual!r}; expected {wanted!r}.")


def _resolve_tail_kinematics_sources(
    root: zarr.Group,
    shape_run: Optional[str],
) -> tuple[str, zarr.Group, TailKinematicsSources]:
    """Resolve lazy array handles without materializing framewise source tables."""

    shape_group, run_name, run_path = resolve_subject_shape_run(root, shape_run)
    components = _require_group(shape_group, "components", path=run_path)
    body_path = f"{run_path}/components"
    body = _require_group(components, "subject_body", path=body_path)
    body_frame = _require_group(shape_group, "body_frame", path=run_path)

    source_s_array = _require_array_handle(body, "tail_sample_s", path=f"{body_path}/subject_body")
    source_s = np.asarray(source_s_array[:], dtype=np.float32)
    if source_s.ndim != 1 or int(source_s.shape[0]) < 2:
        raise SubjectShapeIOError("tail_sample_s must be one-dimensional with at least two positions.")
    if np.any(~np.isfinite(source_s)) or np.any(np.diff(source_s.astype(np.float64)) <= 0.0):
        raise SubjectShapeIOError("tail_sample_s must be finite and strictly increasing.")

    arrays = {
        "tail_sample_xy": _require_array_handle(body, "tail_sample_xy", path=f"{body_path}/subject_body"),
        "tail_tangent_xy": _require_array_handle(body, "tail_tangent_xy", path=f"{body_path}/subject_body"),
        "tail_curvature_px_inv": _require_array_handle(
            body, "tail_curvature_px_inv", path=f"{body_path}/subject_body"
        ),
        "tail_sample_valid": _require_array_handle(body, "tail_sample_valid", path=f"{body_path}/subject_body"),
        "bspline_valid": _require_array_handle(body, "bspline_valid", path=f"{body_path}/subject_body"),
        "tail_base_xy": _require_array_handle(body, "tail_base_xy", path=f"{body_path}/subject_body"),
        "body_forward_axis_xy": _require_array_handle(body_frame, "forward_axis_xy", path=f"{run_path}/body_frame"),
        "body_left_axis_xy": _require_array_handle(body_frame, "left_axis_xy", path=f"{run_path}/body_frame"),
        "body_frame_valid": _require_array_handle(body_frame, "valid", path=f"{run_path}/body_frame"),
    }
    tail_xy_shape = tuple(int(value) for value in arrays["tail_sample_xy"].shape)
    if len(tail_xy_shape) != 3 or int(tail_xy_shape[2]) != 2:
        raise SubjectShapeIOError(f"tail_sample_xy has shape {tail_xy_shape!r}; expected (N, S, 2).")
    row_count, source_sample_count, _xy = tail_xy_shape
    if int(source_sample_count) != int(source_s.shape[0]):
        raise SubjectShapeIOError(
            "tail_sample_s length does not match the second axis of tail_sample_xy "
            f"({source_s.shape[0]} != {source_sample_count})."
        )
    _validate_source_shape("tail_tangent_xy", arrays["tail_tangent_xy"], tail_xy_shape)
    _validate_source_shape(
        "tail_curvature_px_inv", arrays["tail_curvature_px_inv"], (row_count, source_sample_count)
    )
    for name in ("tail_sample_valid", "bspline_valid", "body_frame_valid"):
        _validate_source_shape(name, arrays[name], (row_count,))
    for name in ("tail_base_xy", "body_forward_axis_xy", "body_left_axis_xy"):
        _validate_source_shape(name, arrays[name], (row_count, 2))

    reason_arrays = {
        "tail_sample_failure_reason": body.get("tail_sample_failure_reason_bytes"),
        "bspline_failure_reason": body.get("bspline_failure_reason_bytes"),
        "body_frame_failure_reason": body_frame.get("failure_reason_bytes"),
    }
    for name, source in reason_arrays.items():
        if source is not None and (not source.shape or int(source.shape[0]) != int(row_count)):
            raise SubjectShapeIOError(
                f"{name} has shape {tuple(int(value) for value in source.shape)!r}; expected first axis {row_count}."
            )

    return (
        run_name,
        shape_group,
        TailKinematicsSources(
            source_tail_sample_s=source_s,
            arrays=arrays,
            reason_arrays=reason_arrays,
            row_count=int(row_count),
            source_sample_count=int(source_sample_count),
        ),
    )


def _read_tail_kinematics_source_block(
    sources: TailKinematicsSources,
    row_slice: slice,
) -> dict[str, np.ndarray]:
    """Read one bounded row block from the lazy subject-shape source handles."""

    arrays = sources.arrays
    return {
        "source_tail_sample_s": sources.source_tail_sample_s,
        "tail_sample_xy": np.asarray(arrays["tail_sample_xy"][row_slice], dtype=np.float32),
        "tail_tangent_xy": np.asarray(arrays["tail_tangent_xy"][row_slice], dtype=np.float32),
        "tail_curvature_px_inv": np.asarray(arrays["tail_curvature_px_inv"][row_slice], dtype=np.float32),
        "tail_sample_valid": np.asarray(arrays["tail_sample_valid"][row_slice], dtype=bool),
        "bspline_valid": np.asarray(arrays["bspline_valid"][row_slice], dtype=bool),
        "tail_base_xy": np.asarray(arrays["tail_base_xy"][row_slice], dtype=np.float32),
        "body_forward_axis_xy": np.asarray(arrays["body_forward_axis_xy"][row_slice], dtype=np.float32),
        "body_left_axis_xy": np.asarray(arrays["body_left_axis_xy"][row_slice], dtype=np.float32),
        "body_frame_valid": np.asarray(arrays["body_frame_valid"][row_slice], dtype=bool),
        "tail_sample_failure_reason": _read_optional_reason_labels(
            sources.reason_arrays.get("tail_sample_failure_reason"), row_slice
        ),
        "bspline_failure_reason": _read_optional_reason_labels(
            sources.reason_arrays.get("bspline_failure_reason"), row_slice
        ),
        "body_frame_failure_reason": _read_optional_reason_labels(
            sources.reason_arrays.get("body_frame_failure_reason"), row_slice
        ),
    }


def _copy_row_lineage_bounded(
    run_group: zarr.Group,
    shape_group: zarr.Group,
    *,
    row_count: int,
    block_rows: int,
) -> tuple[list[str], list[str], str]:
    target = run_group.require_group("row_index")
    source = shape_group.get("row_index")
    copied: list[str] = []
    missing: list[str] = []
    frame_source: Any | None = None
    if isinstance(source, zarr.Group):
        for name in ROW_LINEAGE_NAMES:
            source_array = source.get(name)
            if source_array is None:
                missing.append(name)
                continue
            _copy_array_bounded(
                target,
                name,
                source_array,
                block_rows=int(block_rows),
                row_aligned_count=int(row_count),
                shard_rows=int(block_rows),
            )
            copied.append(name)
            if name == "frame_indices":
                frame_source = source_array
    else:
        missing.extend(ROW_LINEAGE_NAMES)

    frame_dtype = frame_source.dtype if frame_source is not None else np.dtype(np.int64)
    frame_chunks = _metric_chunks(int(row_count))
    frame_index = _create_array(
        run_group,
        "frame_index",
        shape=(int(row_count),),
        dtype=frame_dtype,
        chunks=frame_chunks,
        shards=_output_shards(frame_chunks, shard_rows=int(block_rows), dtype=frame_dtype),
    )
    for row_slice in _iter_row_slices(int(row_count), int(block_rows)):
        if frame_source is not None:
            frame_index[row_slice] = np.asarray(frame_source[row_slice])
        else:
            start = int(row_slice.start or 0)
            stop = int(row_slice.stop or start)
            frame_index[row_slice] = np.arange(start, stop, dtype=np.int64)
    frame_index_source = "row_index/frame_indices" if frame_source is not None else "row_number_fallback"
    return copied, missing, frame_index_source


def _prepare_tail_kinematics_run(
    root: zarr.Group,
    *,
    target_run: str,
    shape_run_name: str,
    shape_group: zarr.Group,
    row_count: int,
    tail_angle_sample_count: int,
    source_geometry_tail_sample_count: int,
    requested_block_rows: int,
    effective_block_rows: int,
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
    created = _utc_now()

    copied, missing, frame_index_source = _copy_row_lineage_bounded(
        run_group,
        shape_group,
        row_count=int(row_count),
        block_rows=int(effective_block_rows),
    )
    output_row_chunk = int(_metric_chunks(int(row_count))[0])
    block_count = len(_iter_row_slices(int(row_count), int(effective_block_rows)))

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
            "source_geometry_tail_sample_count": int(source_geometry_tail_sample_count),
            "curvature_source": "subject_shape.tail_curvature_px_inv",
            "frame_index_source": frame_index_source,
            "row_lineage_copied": copied,
            "row_lineage_missing": missing,
            "materialization_mode": "bounded_streaming_single_writer",
            "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
            "requested_block_rows": int(requested_block_rows),
            "effective_block_rows": int(effective_block_rows),
            "output_row_chunk": output_row_chunk,
            "output_shard_rows": int(effective_block_rows),
            "block_count": int(block_count),
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
        row_count=int(row_count),
        block_rows=int(effective_block_rows),
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
            "materialization_mode": "bounded_streaming_single_writer",
            "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
            "requested_block_rows": int(requested_block_rows),
            "effective_block_rows": int(effective_block_rows),
            "output_row_chunk": output_row_chunk,
            "output_shard_rows": int(effective_block_rows),
            "block_count": int(block_count),
        },
        inputs={
            "source_subject_shape_run": shape_run_name,
            "source_refined_subject_masks_run": source_refined_run,
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="tail_kinematics_run")
    return run_group


def _prepare_tail_kinematics_output_arrays(
    run_group: zarr.Group,
    *,
    row_count: int,
    tail_angle_sample_s: np.ndarray,
    shard_rows: int | None = None,
) -> None:
    sample_s = np.asarray(tail_angle_sample_s, dtype=np.float32).reshape(-1)
    sample_count = int(sample_s.shape[0])
    chunks_1d = _metric_chunks(row_count)
    shards_1d = _output_shards(chunks_1d, shard_rows=shard_rows, dtype=bool)
    _create_array(
        run_group,
        "valid",
        shape=(row_count,),
        dtype=bool,
        chunks=chunks_1d,
        shards=shards_1d,
    )
    reason_chunks = _metric_chunks_lastdim(row_count, REASON_BYTES_WIDTH)
    _create_array(
        run_group,
        "failure_reason_bytes",
        shape=(row_count, REASON_BYTES_WIDTH),
        dtype=np.uint8,
        chunks=reason_chunks,
        shards=_output_shards(reason_chunks, shard_rows=shard_rows, dtype=np.uint8),
    )
    _write_array(run_group, "tail_angle_sample_s", sample_s, chunks=(sample_count,))
    sample_xy_chunks = _metric_chunks_3d(row_count, sample_count, 2)
    _create_array(
        run_group,
        "tail_angle_sample_xy",
        shape=(row_count, sample_count, 2),
        dtype=np.float32,
        chunks=sample_xy_chunks,
        shards=_output_shards(sample_xy_chunks, shard_rows=shard_rows, dtype=np.float32),
        fill_value=np.nan,
    )
    for name in ("tail_angle_rad", "tail_angle_deg", "tail_lateral_deflection_px", "tail_curvature_px_inv"):
        chunks_2d = _metric_chunks_lastdim(row_count, sample_count)
        _create_array(
            run_group,
            name,
            shape=(row_count, sample_count),
            dtype=np.float32,
            chunks=chunks_2d,
            shards=_output_shards(chunks_2d, shard_rows=shard_rows, dtype=np.float32),
            fill_value=np.nan,
        )
    for name in (
        "tail_tip_angle_rad",
        "tail_tip_angle_deg",
        "tail_tip_lateral_deflection_px",
        "max_abs_tail_angle_rad",
        "max_abs_tail_angle_deg",
        "tail_angle_rms_rad",
        "tail_angle_rms_deg",
        "integrated_abs_tail_angle_rad",
        "max_abs_tail_curvature_px_inv",
        "integrated_abs_tail_curvature",
    ):
        _create_array(
            run_group,
            name,
            shape=(row_count,),
            dtype=np.float32,
            chunks=chunks_1d,
            shards=_output_shards(chunks_1d, shard_rows=shard_rows, dtype=np.float32),
            fill_value=np.nan,
        )


def _write_tail_kinematics_batch_slice(
    run_group: zarr.Group,
    row_slice: slice,
    batch: TailKinematicsBatch,
) -> None:
    run_group["valid"][row_slice] = batch.valid.astype(bool)
    run_group["failure_reason_bytes"][row_slice] = batch.failure_reason_bytes
    run_group["tail_angle_sample_xy"][row_slice] = batch.tail_angle_sample_xy
    run_group["tail_angle_rad"][row_slice] = batch.tail_angle_rad
    run_group["tail_angle_deg"][row_slice] = batch.tail_angle_deg
    run_group["tail_tip_angle_rad"][row_slice] = batch.tail_tip_angle_rad
    run_group["tail_tip_angle_deg"][row_slice] = batch.tail_tip_angle_deg
    run_group["tail_lateral_deflection_px"][row_slice] = batch.tail_lateral_deflection_px
    run_group["tail_tip_lateral_deflection_px"][row_slice] = batch.tail_tip_lateral_deflection_px
    run_group["max_abs_tail_angle_rad"][row_slice] = batch.max_abs_tail_angle_rad
    run_group["max_abs_tail_angle_deg"][row_slice] = batch.max_abs_tail_angle_deg
    run_group["tail_angle_rms_rad"][row_slice] = batch.tail_angle_rms_rad
    run_group["tail_angle_rms_deg"][row_slice] = batch.tail_angle_rms_deg
    run_group["integrated_abs_tail_angle_rad"][row_slice] = batch.integrated_abs_tail_angle_rad
    run_group["tail_curvature_px_inv"][row_slice] = batch.tail_curvature_px_inv
    run_group["max_abs_tail_curvature_px_inv"][row_slice] = batch.max_abs_tail_curvature_px_inv
    run_group["integrated_abs_tail_curvature"][row_slice] = batch.integrated_abs_tail_curvature


def _write_tail_kinematics_batch(run_group: zarr.Group, batch: TailKinematicsBatch) -> None:
    """Compatibility helper for writing an already-materialized batch."""

    row_count = int(batch.valid.shape[0])
    _prepare_tail_kinematics_output_arrays(
        run_group,
        row_count=row_count,
        tail_angle_sample_s=batch.tail_angle_sample_s,
        shard_rows=None,
    )
    _write_tail_kinematics_batch_slice(run_group, slice(0, row_count), batch)


def write_tail_kinematics_run_group(
    root: zarr.Group,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    overwrite: bool = False,
    dry_run: bool = False,
    stage_command: Optional[str] = None,
) -> dict[str, object]:
    """Write one tail-kinematics run from an existing subject-shape run."""

    if int(tail_angle_sample_count) < 2:
        raise ValueError("tail_angle_sample_count must be >= 2.")
    if int(block_rows) <= 0:
        raise ValueError("block_rows must be positive.")
    shape_run_name, shape_group, sources = _resolve_tail_kinematics_sources(root, shape_run)
    row_count = int(sources.row_count)
    effective_block_rows = _effective_block_rows(
        row_count=row_count,
        requested_block_rows=int(block_rows),
    )
    row_slices = _iter_row_slices(row_count, effective_block_rows)
    target_run = str(run_name or _default_run_name())
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "tail_kinematics_run": target_run,
        "source_subject_shape_run": shape_run_name,
        "source_refined_subject_masks_run": shape_group.attrs.get("source_refined_subject_masks_run"),
        "roi_count": int(row_count),
        "tail_angle_sample_count": int(tail_angle_sample_count),
        "materialization_mode": "bounded_streaming_single_writer",
        "compute_kernel": TAIL_KINEMATICS_COMPUTE_KERNEL,
        "requested_block_rows": int(block_rows),
        "effective_block_rows": int(effective_block_rows),
        "output_row_chunk": int(_metric_chunks(row_count)[0]),
        "output_shard_rows": int(effective_block_rows),
        "block_count": int(len(row_slices)),
        "mutates_archive": not bool(dry_run),
    }
    if dry_run:
        return dict(_json_safe(summary))

    started = time.perf_counter()
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    run_group: zarr.Group | None = None
    completed_block_count = 0
    valid_count = 0
    reason_counts: dict[str, int] = {}
    try:
        run_group = _prepare_tail_kinematics_run(
            root,
            target_run=target_run,
            shape_run_name=shape_run_name,
            shape_group=shape_group,
            row_count=row_count,
            tail_angle_sample_count=int(tail_angle_sample_count),
            source_geometry_tail_sample_count=int(sources.source_sample_count),
            requested_block_rows=int(block_rows),
            effective_block_rows=int(effective_block_rows),
            stage_command=command,
            overwrite=overwrite,
        )
        target_s = tail_sample_positions(int(tail_angle_sample_count)).astype(np.float32)
        _prepare_tail_kinematics_output_arrays(
            run_group,
            row_count=row_count,
            tail_angle_sample_s=target_s,
            shard_rows=int(effective_block_rows),
        )
        for row_slice in row_slices:
            block_sources = _read_tail_kinematics_source_block(sources, row_slice)
            batch = compute_tail_kinematics_from_subject_shape_arrays(
                **block_sources,
                tail_angle_sample_count=int(tail_angle_sample_count),
            )
            _write_tail_kinematics_batch_slice(run_group, row_slice, batch)
            valid_count += int(np.count_nonzero(batch.valid))
            for reason in np.asarray(batch.failure_reason, dtype=object).tolist():
                key = str(reason or "")
                reason_counts[key] = int(reason_counts.get(key, 0) + 1)
            completed_block_count += 1

        duration_seconds = float(time.perf_counter() - started)
        invalid_count = int(row_count - valid_count)
        rows_per_second = float(row_count / duration_seconds) if duration_seconds > 0.0 else float("inf")
        run_group.attrs["duration_seconds"] = duration_seconds
        run_group.attrs["rows_per_second"] = rows_per_second
        run_group.attrs["valid_row_count"] = int(valid_count)
        run_group.attrs["invalid_row_count"] = invalid_count
        run_group.attrs["completed_block_count"] = int(completed_block_count)
        run_group.attrs["failure_reason_counts"] = reason_counts
        mark_run_complete(
            run_group,
            parent_group=root["analysis"]["tail_kinematics_runs"],
            run_name=target_run,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command=command,
            ),
        )
    except Exception as exc:
        if run_group is None:
            candidate = root.get(f"analysis/tail_kinematics_runs/{target_run}")
            if isinstance(candidate, zarr.Group):
                run_group = candidate
        if run_group is not None:
            run_group.attrs["completed_block_count"] = int(completed_block_count)
            mark_run_failed(
                run_group,
                parent_group=root.get("analysis/tail_kinematics_runs"),
                run_name=target_run,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    summary.update(
        {
            "status": "updated",
            "valid_row_count": valid_count,
            "invalid_row_count": invalid_count,
            "failure_reason_counts": reason_counts,
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "completed_block_count": int(completed_block_count),
        }
    )
    return dict(_json_safe(summary))


def write_tail_kinematics_run(
    zarr_path: str | Path,
    *,
    shape_run: Optional[str] = None,
    run_name: Optional[str] = None,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_tail_kinematics_run_group(
        root,
        shape_run=shape_run,
        run_name=run_name,
        tail_angle_sample_count=tail_angle_sample_count,
        block_rows=block_rows,
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
    parser.add_argument(
        "--block-rows",
        type=int,
        default=DEFAULT_BLOCK_ROWS,
        help=(
            "Requested rows per bounded compute block. The effective value is rounded up "
            "to the output row-chunk grid."
        ),
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
        block_rows=int(args.block_rows),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
