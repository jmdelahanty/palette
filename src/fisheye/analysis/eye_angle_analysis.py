#!/usr/bin/env python3
"""
Frame-wise eye angle computation for Palette archives.

This module derives head-relative eye angles, per-eye kinematics, and quality
flags from canonical refined-subject eye geometry, legacy refined-eye geometry
fallbacks, and their source keypoint headings. The results are stored under
``analysis/eye_angle_runs/<run>`` with full provenance metadata so downstream
tools can consume clean, frame-aligned measurements.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import dask
from dask import delayed
import numpy as np
import zarr
from rich.console import Console

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from fisheye.shared.provenance_attrs import (
    build_source_keypoints_attrs,
    resolve_source_keypoints_run,
)
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_REFINED_EYE,
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    resolve_eye_geometry_source,
)
from fisheye.pose.schema import resolve_required_keypoint_indices_from_attrs
from fisheye.utils.metadata import get_fps
from fisheye.utils.system import get_git_info
from fisheye.utils.zarr_io import open_zarr_root

# Reason-code bitmask values (shared across detection- and frame-level QA)
REASON_NONE = np.uint16(0)
REASON_DETECTION_FAILURE = np.uint16(1 << 0)
REASON_HEADING_INVALID = np.uint16(1 << 1)
REASON_LEFT_ELLIPSE_INVALID = np.uint16(1 << 2)
REASON_RIGHT_ELLIPSE_INVALID = np.uint16(1 << 3)
REASON_MULTI_DETECTION = np.uint16(1 << 4)
REASON_NO_DETECTION = np.uint16(1 << 5)

REASON_CODE_MAP = {
    int(REASON_DETECTION_FAILURE): "detection_failure",
    int(REASON_HEADING_INVALID): "heading_invalid",
    int(REASON_LEFT_ELLIPSE_INVALID): "left_ellipse_invalid",
    int(REASON_RIGHT_ELLIPSE_INVALID): "right_ellipse_invalid",
    int(REASON_MULTI_DETECTION): "multiple_detections",
    int(REASON_NO_DETECTION): "no_detection",
}

ELLIPSE_CIRCULARITY_THRESHOLD = 0.95  # reject nearly circular fits that lack a stable major axis
DERIVATIVE_MAX_DT = 0.25  # seconds; ignore large gaps when computing discrete derivatives
ANGLE_SMOOTHING_WINDOW = 7  # frames; moving-average window for smoothed angle outputs
_HEAD_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")
SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks")
SERIAL_EXECUTION_BACKEND = "serial_driver"
DASK_WORKER_EXECUTION_BACKEND = "dask_worker_chunks"

_BASE_ROI_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("left_deg", "left_deg"),
    ("right_deg", "right_deg"),
    ("left_signed_deg", "left_signed_deg"),
    ("right_signed_deg", "right_signed_deg"),
    ("vergence_deg", "vergence_deg"),
    ("vergence_signed_deg", "vergence_signed_deg"),
    ("version_deg", "version_deg"),
    ("left_minor_signed_deg", "left_minor_signed_deg"),
    ("right_minor_signed_deg", "right_minor_signed_deg"),
    ("vergence_minor_signed_deg", "vergence_minor_signed_deg"),
    ("version_minor_deg", "version_minor_deg"),
    ("heading_deg", "heading_deg"),
    ("left_centroid_deg", "left_centroid_deg"),
    ("right_centroid_deg", "right_centroid_deg"),
    ("vergence_centroid_deg", "vergence_centroid_deg"),
)

_BASE_QA_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("valid_left", "valid_left"),
    ("valid_right", "valid_right"),
    ("valid_frame", "valid_frame"),
    ("reason_codes", "reason_codes"),
)

_BASE_SUPPORT_RESULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("ellipse_major", "ellipse_major"),
    ("ellipse_minor", "ellipse_minor"),
    ("ellipse_ratio", "ellipse_ratio"),
)


def _eye_angle_definition_attrs() -> Dict[str, object]:
    """Return stable metadata definitions for eye-angle output arrays."""
    return {
        "signed_angles": True,
        "signed_angle_convention": "per-eye signed angles are temporal-positive",
        "vergence_definition": "abs(vergence_signed_deg)",
        "vergence_signed_definition": "-(left_signed_deg + right_signed_deg)",
        "version_definition": "0.5*(-left_signed_deg + right_signed_deg)",
        "minor_signed_angles": True,
        "minor_signed_angle_convention": "per-eye minor signed angles are temporal-positive",
        "minor_vergence_definition": "abs(vergence_minor_signed_deg)",
        "minor_vergence_signed_definition": "-(left_minor_signed_deg + right_minor_signed_deg)",
        "minor_version_definition": "0.5*(-left_minor_signed_deg + right_minor_signed_deg)",
    }


@dataclass
class EyeAngleResults:
    """Container for detection-level outputs."""

    left_deg: np.ndarray
    right_deg: np.ndarray
    left_signed_deg: np.ndarray
    right_signed_deg: np.ndarray
    left_minor_signed_deg: np.ndarray
    right_minor_signed_deg: np.ndarray
    vergence_deg: np.ndarray
    vergence_signed_deg: np.ndarray
    vergence_minor_signed_deg: np.ndarray
    version_deg: np.ndarray
    version_minor_deg: np.ndarray
    ellipse_major: np.ndarray
    ellipse_minor: np.ndarray
    ellipse_ratio: np.ndarray
    valid_left: np.ndarray
    valid_right: np.ndarray
    valid_frame: np.ndarray
    reason_codes: np.ndarray
    heading_deg: np.ndarray
    # Centroid-based angles (paper-comparable)
    left_centroid_deg: np.ndarray
    right_centroid_deg: np.ndarray
    vergence_centroid_deg: np.ndarray


@dataclass
class EyeAngleInputContext:
    """Resolved zarr inputs for one eye-angle run."""

    eye_geometry: Any
    kp_group: zarr.Group
    detection_success_source: zarr.Group
    detection_success_key: str
    frame_indices_source: zarr.Group
    keypoint_run_name: str
    keypoint_indices: Dict[str, int]


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
        raise argparse.ArgumentTypeError(f"execution_backend must be one of {EXECUTION_BACKENDS}; got {value!r}.")
    return backend


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    chunk = max(1, int(chunk_size))
    return [(start, min(total, start + chunk)) for start in range(0, total, chunk)]


def _to_half_turn(angle_rad: np.ndarray) -> np.ndarray:
    """Map angles into [0, π) so 180° flips of the major axis are treated identically."""
    return np.mod(angle_rad, np.pi)


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit-length vectors, protecting against zero magnitude."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return v / norm


def _resolve_smoothing_window(length: int, desired: int) -> int:
    """Return an odd window length that fits within the sequence, else 0 for no smoothing."""
    if length <= 0:
        return 0
    window = min(desired, length)
    if window < 3:
        return 0
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 0
    return window


def _smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a NaN-aware moving average to 1D data."""
    if window <= 2:
        return np.array(values, copy=True)
    kernel = np.ones(window, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.full_like(values, np.nan)
    sums = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="same")
    counts = np.convolve(finite_mask.astype(np.float32), kernel, mode="same")
    smoothed = np.full_like(values, np.nan)
    valid = counts > 0
    smoothed[valid] = sums[valid] / counts[valid]
    return smoothed


def _compute_delta(values: np.ndarray) -> np.ndarray:
    """Compute absolute frame-to-frame differences, preserving NaNs."""
    delta = np.full_like(values, np.nan)
    if values.size > 1:
        prev = values[:-1]
        curr = values[1:]
        mask = np.isfinite(prev) & np.isfinite(curr)
        diffs = np.abs(curr - prev)
        out_slice = delta[1:]
        out_slice[mask] = diffs[mask]
        delta[1:] = out_slice
    return delta


def _process_chunk(
    ellipse_params: np.ndarray,
    ellipse_success: np.ndarray,
    keypoints_roi: np.ndarray,
    heading_deg: np.ndarray,
    detection_success: np.ndarray,
    *,
    keypoint_indices: Dict[str, int],
) -> EyeAngleResults:
    """Process a chunk of detections into eye angles and QA flags."""
    chunk_len = ellipse_params.shape[0]

    left_angles = np.full(chunk_len, np.nan, dtype=np.float32)
    right_angles = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence = np.full(chunk_len, np.nan, dtype=np.float32)
    left_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    right_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    version = np.full(chunk_len, np.nan, dtype=np.float32)
    left_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    right_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_minor_signed = np.full(chunk_len, np.nan, dtype=np.float32)
    version_minor = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_major = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_minor = np.full(chunk_len, np.nan, dtype=np.float32)
    ellipse_ratio = np.full(chunk_len, np.nan, dtype=np.float32)
    valid_left = np.zeros(chunk_len, dtype=bool)
    valid_right = np.zeros(chunk_len, dtype=bool)
    valid_frame = np.zeros(chunk_len, dtype=bool)
    reason_codes = np.zeros(chunk_len, dtype=np.uint16)

    heading_out = heading_deg.astype(np.float64, copy=True)
    heading_valid = np.isfinite(heading_out)

    bladder = keypoints_roi[:, int(keypoint_indices["swim_bladder"]), :]
    eye_left_kp = keypoints_roi[:, int(keypoint_indices["eye_left"]), :]
    eye_right_kp = keypoints_roi[:, int(keypoint_indices["eye_right"]), :]

    reason_codes[~detection_success] |= REASON_DETECTION_FAILURE
    reason_codes[~heading_valid] |= REASON_HEADING_INVALID

    heading_rad = np.full(chunk_len, np.nan, dtype=np.float64)
    heading_rad[heading_valid] = np.deg2rad(heading_out[heading_valid])
    heading_rad = _to_half_turn(heading_rad)

    # ---------- Centroid-based angles (paper-comparable) ----------
    # Measures eye position angle in fish-frame coordinates.
    # Paper method: vergence = |theta_L| + |theta_R|
    left_centroid = np.full(chunk_len, np.nan, dtype=np.float32)
    right_centroid = np.full(chunk_len, np.nan, dtype=np.float32)
    vergence_centroid = np.full(chunk_len, np.nan, dtype=np.float32)

    # Paper head center: mean of the 3 ROI keypoints
    head_center = (bladder + eye_left_kp + eye_right_kp) / 3.0

    centroid_mask = (
        detection_success
        & heading_valid
        & np.all(np.isfinite(head_center), axis=1)
        & np.all(np.isfinite(eye_left_kp), axis=1)
        & np.all(np.isfinite(eye_right_kp), axis=1)
    )

    if np.any(centroid_mask):
        cidxs = np.where(centroid_mask)[0]

        # Vectors from head center to each eye in image coords
        vL = eye_left_kp[cidxs] - head_center[cidxs]
        vR = eye_right_kp[cidxs] - head_center[cidxs]

        # Convert to math coords (y up) to match heading computation: (x, -y)
        vLx, vLy = vL[:, 0], -vL[:, 1]
        vRx, vRy = vR[:, 0], -vR[:, 1]

        # Rotate by -heading into fish frame (heading aligned to +x)
        ang = np.deg2rad(heading_out[cidxs]).astype(np.float64)
        c, s = np.cos(-ang), np.sin(-ang)

        Lx = c * vLx - s * vLy
        Ly = s * vLx + c * vLy
        Rx = c * vRx - s * vRy
        Ry = s * vRx + c * vRy

        theta_L = np.degrees(np.arctan2(Ly, Lx)).astype(np.float32)
        theta_R = np.degrees(np.arctan2(Ry, Rx)).astype(np.float32)

        left_centroid[cidxs] = theta_L
        right_centroid[cidxs] = theta_R
        vergence_centroid[cidxs] = np.abs(theta_L) + np.abs(theta_R)

    for eye_idx, target_array, valid_array, signed_array, fail_bit in (
        (0, left_angles, valid_left, left_signed, REASON_LEFT_ELLIPSE_INVALID),
        (1, right_angles, valid_right, right_signed, REASON_RIGHT_ELLIPSE_INVALID),
    ):
        ellipse_ok = ellipse_success[:, eye_idx]
        eye_params = ellipse_params[:, eye_idx, :]
        angle_deg = eye_params[:, 4]
        major = eye_params[:, 2]
        minor = eye_params[:, 3]

        finite_mask = (
            ellipse_ok
            & np.isfinite(angle_deg)
            & np.isfinite(major)
            & np.isfinite(minor)
            & (major > 0)
            & (minor > 0)
        )

        ratio = np.zeros_like(major, dtype=np.float64)
        ratio_mask = finite_mask & (major > 0)
        ratio[ratio_mask] = minor[ratio_mask] / major[ratio_mask]
        circular_mask = ratio_mask & (ratio > ELLIPSE_CIRCULARITY_THRESHOLD)
        finite_mask &= ~circular_mask
        if np.any(circular_mask):
            reason_codes[circular_mask] |= fail_bit

        combined_mask = finite_mask & detection_success
        reason_codes[~finite_mask] |= fail_bit

        if np.any(combined_mask):
            idxs = np.where(combined_mask)[0]
            # alpha_eye: major-axis angle (radians) in image coordinates; 0 rad along +x, CCW positive
            alpha_eye = np.deg2rad(angle_deg[idxs]).astype(np.float64)
            alpha_eye = _to_half_turn(alpha_eye)

            centers = 0.5 * (eye_left_kp[idxs] + eye_right_kp[idxs])
            # u_head: fish head axis from swim bladder to head centre
            head_axis = _unit(centers - bladder[idxs])
            if eye_idx == 0:
                nasal_axis = _unit(centers - eye_left_kp[idxs])
            else:
                nasal_axis = _unit(centers - eye_right_kp[idxs])
            temporal_axis = -nasal_axis

            axis_major = np.stack([np.cos(alpha_eye), np.sin(alpha_eye)], axis=1)
            dot_temporal_major = np.einsum("ij,ij->i", axis_major, temporal_axis)
            sign_major = np.where(dot_temporal_major >= 0.0, 1.0, -1.0)
            axis_major_aligned = axis_major * sign_major[:, None]

            dot_head_major = np.clip(
                np.einsum("ij,ij->i", axis_major_aligned, head_axis),
                -1.0,
                1.0,
            )
            theta_major_rad = np.arccos(dot_head_major)
            theta_major_deg = np.degrees(theta_major_rad).astype(np.float32)

            target_array[idxs] = theta_major_deg
            signed_array[idxs] = sign_major.astype(np.float32) * theta_major_deg

            axis_minor = np.stack([-np.sin(alpha_eye), np.cos(alpha_eye)], axis=1)
            dot_temporal = np.einsum("ij,ij->i", axis_minor, temporal_axis)
            sign_minor = np.where(dot_temporal >= 0.0, 1.0, -1.0)
            axis_minor_aligned = axis_minor * sign_minor[:, None]

            dot_head_minor = np.clip(
                np.einsum("ij,ij->i", axis_minor_aligned, head_axis),
                -1.0,
                1.0,
            )
            theta_minor_rad = np.arccos(dot_head_minor)
            theta_minor_deg = np.degrees(theta_minor_rad).astype(np.float32)
            theta_minor_clipped = np.clip(theta_minor_deg, 0.0, 90.0)

            if eye_idx == 0:
                left_minor_signed[idxs] = sign_minor.astype(np.float32) * theta_minor_clipped
            else:
                right_minor_signed[idxs] = sign_minor.astype(np.float32) * theta_minor_clipped

            ellipse_major[idxs] = major[idxs].astype(np.float32, copy=False)
            ellipse_minor[idxs] = minor[idxs].astype(np.float32, copy=False)
            # ratio stored post-thresholding to allow post-hoc tuning
            ellipse_ratio[idxs] = (minor[idxs] / major[idxs]).astype(np.float32, copy=False)
            valid_array[combined_mask] = True

        # Ensure invalid entries remain NaN
        target_array[~valid_array] = np.nan
        signed_array[~valid_array] = np.nan

    left_signed[~valid_left] = np.nan
    right_signed[~valid_right] = np.nan
    left_minor_signed[~valid_left] = np.nan
    right_minor_signed[~valid_right] = np.nan

    valid_frame[:] = valid_left & valid_right & detection_success

    mask = valid_frame
    if np.any(mask):
        # Adopt binocular movement conventions:
        #   vergence  = θ_L(nasal) + θ_R(nasal)
        #             = -(left_temporal + right_temporal)
        #   version   = 0.5 * (θ_L(nasal) - θ_R(nasal))
        # Minor variants follow the same relationships.
        left_temporal = left_signed[mask]
        right_temporal = right_signed[mask]
        left_minor_temporal = left_minor_signed[mask]
        right_minor_temporal = right_minor_signed[mask]

        left_nasal = -left_temporal
        right_nasal = -right_temporal
        left_minor_nasal = -left_minor_temporal
        right_minor_nasal = -right_minor_temporal

        vergence_signed_vals = left_nasal + right_nasal
        vergence[mask] = np.abs(vergence_signed_vals)
        vergence_signed[mask] = vergence_signed_vals
        version[mask] = 0.5 * (left_nasal - right_nasal)

        vergence_minor_signed_vals = left_minor_nasal + right_minor_nasal
        vergence_minor_signed[mask] = vergence_minor_signed_vals
        version_minor[mask] = 0.5 * (left_minor_nasal - right_minor_nasal)

    return EyeAngleResults(
        left_deg=left_angles,
        right_deg=right_angles,
        left_signed_deg=left_signed,
        right_signed_deg=right_signed,
        left_minor_signed_deg=left_minor_signed,
        right_minor_signed_deg=right_minor_signed,
        vergence_deg=vergence,
        vergence_signed_deg=vergence_signed,
        vergence_minor_signed_deg=vergence_minor_signed,
        version_deg=version,
        version_minor_deg=version_minor,
        ellipse_major=ellipse_major,
        ellipse_minor=ellipse_minor,
        ellipse_ratio=ellipse_ratio,
        valid_left=valid_left,
        valid_right=valid_right,
        valid_frame=valid_frame,
        reason_codes=reason_codes,
        heading_deg=heading_out.astype(np.float32, copy=False),
        left_centroid_deg=left_centroid,
        right_centroid_deg=right_centroid,
        vergence_centroid_deg=vergence_centroid,
    )


def _compute_derivative(
    values: np.ndarray,
    time_seconds: np.ndarray,
    valid_mask: np.ndarray,
    max_dt: Optional[float] = None,
) -> np.ndarray:
    """Backward difference using the previous valid sample."""
    derivative = np.full(values.shape, np.nan, dtype=np.float32)
    valid_indices = np.where(valid_mask & np.isfinite(values) & np.isfinite(time_seconds))[0]
    if valid_indices.size < 2:
        return derivative

    prev_idx = valid_indices[0]
    for idx in valid_indices[1:]:
        dt = time_seconds[idx] - time_seconds[prev_idx]
        if dt > 0 and (max_dt is None or dt <= max_dt):
            derivative[idx] = (values[idx] - values[prev_idx]) / dt
        prev_idx = idx
    return derivative


def _to_serializable(value):
    """Convert numpy/python types to plain JSON-serialisable values."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def _count_reason_bits(reason_codes: np.ndarray) -> Dict[str, int]:
    """Aggregate counts for each reason bit."""
    counts: Dict[str, int] = {}
    for code, name in REASON_CODE_MAP.items():
        mask = (reason_codes & code) > 0
        counts[name] = int(mask.sum())
    return counts


def _prepare_output_arrays(
    group: zarr.Group,
    dataset_specs: List[Tuple[str, Tuple[int, ...], Tuple[int, ...], str]],
    fill_value: Optional[float] = None,
) -> None:
    """Create (or overwrite) output arrays according to specs."""
    for name, shape, chunks, dtype in dataset_specs:
        if name in group:
            existing = group[name]
            if tuple(existing.shape) == tuple(shape) and np.dtype(existing.dtype) == np.dtype(dtype):
                continue
            del group[name]
        kwargs = {"dtype": dtype, "chunks": chunks, "overwrite": True}
        if fill_value is not None:
            kwargs["fill_value"] = fill_value
        group.create_array(name, shape=shape, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute head-relative eye angles and QA flags from subject-shape or "
            "refined-subject eye geometry, with refined-eye compatibility fallback."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--subject-shape-run",
        type=str,
        help=(
            "analysis/subject_shape_runs/<run> providing preferred eye geometry "
            "(default: latest subject-shape run with LR eye geometry when available)."
        ),
    )
    parser.add_argument(
        "--refined-eye-run",
        type=str,
        help=(
            "Compatibility refined eye mask run under refined_eye_masks_runs. "
            "When it maps to refined_subject_masks_runs, canonical subject-eye geometry is used."
        ),
    )
    parser.add_argument(
        "--refined-subject-run",
        type=str,
        help="Canonical refined_subject_masks_runs/<run> providing eye geometry (default: latest with LR eyes).",
    )
    parser.add_argument(
        "--keypoint-run",
        type=str,
        help="Refined keypoint run providing heading and ROI coordinates (default: inferred from refined eye run or latest).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional name for the output run (default: timestamp-based).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Number of detections to process per chunk (default: 8192).",
    )
    parser.add_argument(
        "--execution-backend",
        type=_normalize_execution_backend,
        choices=EXECUTION_BACKENDS,
        default=SERIAL_EXECUTION_BACKEND,
        help="Use dask_worker_chunks to process and write independent ROI chunks from workers.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help="Dask scheduler used when --execution-backend=dask_worker_chunks.",
    )
    parser.add_argument("--num-workers", type=int, help="Dask worker count for --execution-backend=dask_worker_chunks.")
    parser.add_argument(
        "--include-chunk-timings",
        action="store_true",
        help="Store per-chunk timing metadata and include detailed timings in run attributes.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Override frames-per-second when computing derivatives (default: infer from archive).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help=f"Override the moving-average window for angle smoothing (default: {ANGLE_SMOOTHING_WINDOW}).",
    )
    return parser


def _resolve_keypoint_run_name(
    *,
    explicit_keypoint_run: Optional[str],
    refined_attrs: Dict[str, object],
    parent_latest: Optional[str],
) -> Optional[str]:
    """Resolve refined-keypoints run from explicit, canonical, legacy, then latest."""
    return (
        explicit_keypoint_run
        or resolve_source_keypoints_run(refined_attrs)
        or parent_latest
    )


def _open_archive_for_eye_angle(zarr_path: Path) -> zarr.Group:
    """Open mutable Palette zarrs with the repository's non-consolidated fallback policy."""
    return open_zarr_root(zarr_path, mode="a")


def _resolve_head_keypoint_indices(kp_group: zarr.Group) -> Dict[str, int]:
    keypoint_count = int(kp_group["keypoints_roi"].shape[1])
    try:
        return resolve_required_keypoint_indices_from_attrs(
            kp_group.attrs,
            _HEAD_KEYPOINT_LABELS,
            keypoint_count=keypoint_count,
        )
    except ValueError as exc:
        raise ValueError(
            "Keypoint run is missing canonical head labels required for eye-angle analysis "
            f"({_HEAD_KEYPOINT_LABELS}): {exc}"
        ) from exc


def _resolve_eye_angle_inputs(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    refined_eye_run: Optional[str],
    keypoint_run: Optional[str],
) -> EyeAngleInputContext:
    eye_geometry = resolve_eye_geometry_source(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        refined_eye_run=refined_eye_run,
        prefer_subject_shape=True,
        prefer_subject=True,
    )

    kp_parent = root.require_group("refined_keypoints_runs")
    keypoint_run_name = _resolve_keypoint_run_name(
        explicit_keypoint_run=keypoint_run,
        refined_attrs=dict(eye_geometry.lineage_attrs),
        parent_latest=kp_parent.attrs.get("latest"),
    )
    if not keypoint_run_name or keypoint_run_name not in kp_parent:
        raise ValueError("Refined keypoint run not found; specify --keypoint-run.")
    kp_group = kp_parent[keypoint_run_name]

    source_kp_run_name = resolve_source_keypoints_run(kp_group.attrs)
    source_kp_group = None
    if source_kp_run_name:
        source_kp_parent = root.get("keypoints_runs")
        if source_kp_parent and source_kp_run_name in source_kp_parent:
            source_kp_group = source_kp_parent[source_kp_run_name]

    required_kp = ["keypoints_roi", "heading"]
    for dataset in required_kp:
        if dataset not in kp_group:
            raise ValueError(f"Keypoint run '{keypoint_run_name}' missing dataset '{dataset}'.")

    if "refined_success" in kp_group:
        detection_success_key = "refined_success"
        detection_success_source = kp_group
    elif "detection_success" in kp_group:
        detection_success_key = "detection_success"
        detection_success_source = kp_group
    elif source_kp_group is not None and "detection_success" in source_kp_group:
        detection_success_key = "detection_success"
        detection_success_source = source_kp_group
    else:
        raise ValueError(
            f"Keypoint run '{keypoint_run_name}' missing detection success data "
            "(no 'refined_success' or 'detection_success' in refined or source keypoints run)."
        )

    frame_indices_source = kp_group if "frame_indices" in kp_group else source_kp_group
    if frame_indices_source is None or "frame_indices" not in frame_indices_source:
        raise ValueError(
            f"Keypoint run '{keypoint_run_name}' missing 'frame_indices' "
            "(not in refined or source keypoints run)."
        )

    total_detections = eye_geometry.ellipse_params.shape[0]
    if kp_group["keypoints_roi"].shape[0] != total_detections:
        raise ValueError("Mismatch between eye geometry source and keypoint detections.")

    return EyeAngleInputContext(
        eye_geometry=eye_geometry,
        kp_group=kp_group,
        detection_success_source=detection_success_source,
        detection_success_key=detection_success_key,
        frame_indices_source=frame_indices_source,
        keypoint_run_name=keypoint_run_name,
        keypoint_indices=_resolve_head_keypoint_indices(kp_group),
    )


def _prepare_base_output_arrays(
    run_group: zarr.Group,
    *,
    total_detections: int,
    chunk_len: int,
) -> None:
    angles_group = run_group.require_group("angles")
    roi_group = angles_group.require_group("roi")
    qa_group = run_group.require_group("qa")
    qa_roi = qa_group.require_group("roi")
    support_group = run_group.require_group("support")

    _prepare_output_arrays(
        roi_group,
        [(name, (total_detections,), (chunk_len,), "f4") for name, _field in _BASE_ROI_RESULT_FIELDS],
    )
    _prepare_output_arrays(
        qa_roi,
        [
            ("valid_left", (total_detections,), (chunk_len,), "bool"),
            ("valid_right", (total_detections,), (chunk_len,), "bool"),
            ("valid_frame", (total_detections,), (chunk_len,), "bool"),
            ("reason_codes", (total_detections,), (chunk_len,), "u2"),
        ],
    )
    _prepare_output_arrays(
        support_group,
        [
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
    )


def _write_base_eye_angle_result(
    run_group: zarr.Group,
    row_slice: slice,
    result: EyeAngleResults,
    *,
    frame_indices: np.ndarray,
    time_seconds: np.ndarray,
) -> None:
    roi_group = run_group["angles"]["roi"]
    qa_roi = run_group["qa"]["roi"]
    support_group = run_group["support"]

    for dataset_name, field_name in _BASE_ROI_RESULT_FIELDS:
        roi_group[dataset_name][row_slice] = getattr(result, field_name)
    for dataset_name, field_name in _BASE_QA_RESULT_FIELDS:
        qa_roi[dataset_name][row_slice] = getattr(result, field_name)
    support_group["frame_indices"][row_slice] = frame_indices
    support_group["time_seconds"][row_slice] = time_seconds
    for dataset_name, field_name in _BASE_SUPPORT_RESULT_FIELDS:
        support_group[dataset_name][row_slice] = getattr(result, field_name)


def _process_and_write_eye_angle_chunk_groups(
    context: EyeAngleInputContext,
    run_group: zarr.Group,
    *,
    start_row: int,
    stop_row: int,
    chunk_index: int,
    fps: Optional[float],
    execution_backend: str,
) -> dict[str, object]:
    chunk_start = time.perf_counter()
    row_slice = slice(int(start_row), int(stop_row))
    timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row - start_row),
        "execution_backend": execution_backend,
    }

    phase_start = time.perf_counter()
    ellipse_params = context.eye_geometry.ellipse_params[row_slice]
    ellipse_success = context.eye_geometry.ellipse_success[row_slice]
    keypoints_roi = context.kp_group["keypoints_roi"][row_slice]
    heading_deg = context.kp_group["heading"][row_slice]
    detection_success = context.detection_success_source[context.detection_success_key][row_slice].astype(bool, copy=False)
    frame_indices = context.frame_indices_source["frame_indices"][row_slice].astype(np.int64, copy=False)
    timing["read_seconds"] = float(time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    chunk_result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=heading_deg,
        detection_success=detection_success,
        keypoint_indices=context.keypoint_indices,
    )
    timing["compute_seconds"] = float(time.perf_counter() - phase_start)

    if fps:
        chunk_time_seconds = (frame_indices.astype(np.float64) / float(fps)).astype(np.float32, copy=False)
    else:
        chunk_time_seconds = np.full(frame_indices.shape, np.nan, dtype=np.float32)

    phase_start = time.perf_counter()
    _write_base_eye_angle_result(
        run_group,
        row_slice,
        chunk_result,
        frame_indices=frame_indices,
        time_seconds=chunk_time_seconds,
    )
    timing["write_seconds"] = float(time.perf_counter() - phase_start)
    timing["valid_frame_count"] = int(chunk_result.valid_frame.sum())
    timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {"chunk_timing": timing, "valid_frame_count": int(chunk_result.valid_frame.sum())}


def _process_and_write_eye_angle_chunk(
    zarr_path: str,
    *,
    subject_shape_run: Optional[str],
    refined_subject_run: Optional[str],
    refined_eye_run: Optional[str],
    keypoint_run: Optional[str],
    eye_angle_run: str,
    start_row: int,
    stop_row: int,
    chunk_index: int,
    fps: Optional[float],
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=refined_subject_run,
        refined_eye_run=refined_eye_run,
        keypoint_run=keypoint_run,
    )
    run_group = root["analysis"]["eye_angle_runs"][eye_angle_run]
    return _process_and_write_eye_angle_chunk_groups(
        context,
        run_group,
        start_row=start_row,
        stop_row=stop_row,
        chunk_index=chunk_index,
        fps=fps,
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


def _project_detection_arrays_to_frames(
    frame_indices: np.ndarray,
    *,
    num_frames: int,
    valid_frame: np.ndarray,
    reason_codes: np.ndarray,
    arrays: Dict[str, np.ndarray],
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    frame_arrays = {name: np.full(num_frames, np.nan, dtype=np.float32) for name in arrays}
    frame_valid = np.zeros(num_frames, dtype=bool)
    frame_reason = np.zeros(num_frames, dtype=np.uint16)
    if num_frames <= 0:
        return frame_arrays, frame_valid, frame_reason

    valid_index_mask = (frame_indices >= 0) & (frame_indices < num_frames)
    valid_indices = frame_indices[valid_index_mask]
    counts = np.bincount(valid_indices, minlength=num_frames) if valid_indices.size else np.zeros(num_frames, dtype=np.int64)
    frame_reason[counts == 0] |= REASON_NO_DETECTION
    frame_reason[counts > 1] |= REASON_MULTI_DETECTION

    detection_indices = np.nonzero(valid_index_mask)[0]
    unique_detection_indices = detection_indices[counts[frame_indices[detection_indices]] == 1]
    unique_frames = frame_indices[unique_detection_indices]
    if unique_detection_indices.size:
        for name, values in arrays.items():
            frame_arrays[name][unique_frames] = values[unique_detection_indices]
        frame_valid[unique_frames] = valid_frame[unique_detection_indices]
        frame_reason[unique_frames] |= reason_codes[unique_detection_indices]
    return frame_arrays, frame_valid, frame_reason


def run(args: argparse.Namespace) -> None:
    console = Console()
    root = _open_archive_for_eye_angle(args.zarr_path)

    analysis_group = root.require_group("analysis")
    parent_group = analysis_group.require_group("eye_angle_runs")

    backend = _normalize_execution_backend(args.execution_backend)
    scheduler_key = _normalize_scheduler(args.scheduler)
    context = _resolve_eye_angle_inputs(
        root,
        subject_shape_run=args.subject_shape_run,
        refined_subject_run=args.refined_subject_run,
        refined_eye_run=args.refined_eye_run,
        keypoint_run=args.keypoint_run,
    )
    eye_geometry = context.eye_geometry
    keypoint_run_name = context.keypoint_run_name
    total_detections = int(eye_geometry.ellipse_params.shape[0])
    chunk_size = max(1, int(args.chunk_size))
    if total_detections and chunk_size > total_detections:
        chunk_size = total_detections

    frame_indices = context.frame_indices_source["frame_indices"][:].astype(np.int64, copy=False)
    if frame_indices.shape[0] != total_detections:
        raise ValueError("Mismatch between frame_indices and detection count.")

    fps = args.fps or get_fps(root)
    if fps is None or fps <= 0:
        fps = None
    smoothing_window_param = args.smoothing_window
    valid_frame_index_mask = frame_indices >= 0
    num_frames = int(frame_indices[valid_frame_index_mask].max() + 1) if np.any(valid_frame_index_mask) else 0
    chunk_len = min(chunk_size, total_detections) if total_detections else 1
    frame_chunk = min(chunk_size, num_frames) if num_frames else 1

    if args.run_name:
        resolved_run_name = args.run_name
    else:
        resolved_run_name = datetime.now(timezone.utc).strftime("eye_angle_%Y%m%d_%H%M%S")

    if resolved_run_name in parent_group:
        raise ValueError(f"Run '{resolved_run_name}' already exists in analysis/eye_angle_runs.")

    run_group = parent_group.create_group(resolved_run_name)
    run_group.attrs["status"] = "running"
    run_group.attrs["execution_backend"] = backend
    run_group.attrs["dask_scheduler"] = scheduler_key
    run_group.attrs["dask_num_workers"] = int(args.num_workers) if args.num_workers is not None else None
    if not args.quiet:
        console.print(f"Created run group: [cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]")

    _prepare_base_output_arrays(run_group, total_detections=total_detections, chunk_len=chunk_len)
    chunks = _row_chunks(total_detections, chunk_size)
    chunk_timings: list[dict[str, object]] = []
    stage_start = time.perf_counter()

    if backend == DASK_WORKER_EXECUTION_BACKEND:
        worker_zarr_path = str(args.zarr_path.expanduser().resolve())
        worker_refined_subject_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT else None
        )
        worker_refined_eye_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_REFINED_EYE else None
        )
        worker_subject_shape_run = (
            eye_geometry.run_name if eye_geometry.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE else None
        )
        tasks = [
            delayed(_process_and_write_eye_angle_chunk)(
                worker_zarr_path,
                subject_shape_run=worker_subject_shape_run,
                refined_subject_run=worker_refined_subject_run,
                refined_eye_run=worker_refined_eye_run,
                keypoint_run=keypoint_run_name,
                eye_angle_run=resolved_run_name,
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                fps=fps,
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
        results = _compute_dask_tasks(tasks, scheduler_key=scheduler_key, num_workers=args.num_workers)
    else:
        results = [
            _process_and_write_eye_angle_chunk_groups(
                context,
                run_group,
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                fps=fps,
                execution_backend=SERIAL_EXECUTION_BACKEND,
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
    for result in sorted(results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
        chunk_timings.append(dict(result["chunk_timing"]))

    roi_group = run_group["angles"]["roi"]
    qa_roi = run_group["qa"]["roi"]
    support_group = run_group["support"]
    left_angles = roi_group["left_deg"][:]
    right_angles = roi_group["right_deg"][:]
    left_signed = roi_group["left_signed_deg"][:]
    right_signed = roi_group["right_signed_deg"][:]
    left_minor_signed = roi_group["left_minor_signed_deg"][:]
    right_minor_signed = roi_group["right_minor_signed_deg"][:]
    vergence = roi_group["vergence_deg"][:]
    vergence_signed = roi_group["vergence_signed_deg"][:]
    vergence_minor_signed = roi_group["vergence_minor_signed_deg"][:]
    version = roi_group["version_deg"][:]
    version_minor = roi_group["version_minor_deg"][:]
    heading_deg_out = roi_group["heading_deg"][:]
    left_centroid = roi_group["left_centroid_deg"][:]
    right_centroid = roi_group["right_centroid_deg"][:]
    vergence_centroid = roi_group["vergence_centroid_deg"][:]
    valid_left = qa_roi["valid_left"][:]
    valid_right = qa_roi["valid_right"][:]
    valid_frame = qa_roi["valid_frame"][:]
    reason_codes = qa_roi["reason_codes"][:]
    time_seconds = support_group["time_seconds"][:]
    ellipse_major = support_group["ellipse_major"][:]
    ellipse_minor = support_group["ellipse_minor"][:]
    ellipse_ratio = support_group["ellipse_ratio"][:]

    left_speed = (
        _compute_derivative(left_angles, time_seconds, valid_left, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_angles, np.nan)
    )
    right_speed = (
        _compute_derivative(right_angles, time_seconds, valid_right, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_angles, np.nan)
    )
    vergence_speed = (
        _compute_derivative(vergence, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence, np.nan)
    )
    vergence_signed_speed = (
        _compute_derivative(vergence_signed, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_signed, np.nan)
    )
    version_speed = (
        _compute_derivative(version, time_seconds, valid_frame, max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version, np.nan)
    )

    left_accel = (
        _compute_derivative(left_speed, time_seconds, np.isfinite(left_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(left_angles, np.nan)
    )
    right_accel = (
        _compute_derivative(right_speed, time_seconds, np.isfinite(right_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(right_angles, np.nan)
    )
    vergence_accel = (
        _compute_derivative(vergence_speed, time_seconds, np.isfinite(vergence_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence, np.nan)
    )
    vergence_signed_accel = (
        _compute_derivative(vergence_signed_speed, time_seconds, np.isfinite(vergence_signed_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(vergence_signed, np.nan)
    )
    version_accel = (
        _compute_derivative(version_speed, time_seconds, np.isfinite(version_speed), max_dt=DERIVATIVE_MAX_DT)
        if fps
        else np.full_like(version, np.nan)
    )

    window_setting = smoothing_window_param if smoothing_window_param is not None else ANGLE_SMOOTHING_WINDOW
    detection_smooth_window = _resolve_smoothing_window(total_detections, window_setting)
    if detection_smooth_window:
        left_smoothed = _smooth_signal(left_angles, detection_smooth_window).astype(np.float32, copy=False)
        right_smoothed = _smooth_signal(right_angles, detection_smooth_window).astype(np.float32, copy=False)
        vergence_smoothed = _smooth_signal(vergence, detection_smooth_window).astype(np.float32, copy=False)
        left_signed_smoothed = _smooth_signal(left_signed, detection_smooth_window).astype(np.float32, copy=False)
        right_signed_smoothed = _smooth_signal(right_signed, detection_smooth_window).astype(np.float32, copy=False)
        vergence_signed_smoothed = _smooth_signal(vergence_signed, detection_smooth_window).astype(np.float32, copy=False)
        version_smoothed = _smooth_signal(version, detection_smooth_window).astype(np.float32, copy=False)
        left_minor_signed_smoothed = _smooth_signal(left_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        right_minor_signed_smoothed = _smooth_signal(right_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        vergence_minor_signed_smoothed = _smooth_signal(vergence_minor_signed, detection_smooth_window).astype(np.float32, copy=False)
        version_minor_smoothed = _smooth_signal(version_minor, detection_smooth_window).astype(np.float32, copy=False)
        left_centroid_smoothed = _smooth_signal(left_centroid, detection_smooth_window).astype(np.float32, copy=False)
        right_centroid_smoothed = _smooth_signal(right_centroid, detection_smooth_window).astype(np.float32, copy=False)
        vergence_centroid_smoothed = _smooth_signal(vergence_centroid, detection_smooth_window).astype(np.float32, copy=False)
    else:
        left_smoothed = np.array(left_angles, copy=True)
        right_smoothed = np.array(right_angles, copy=True)
        vergence_smoothed = np.array(vergence, copy=True)
        left_signed_smoothed = np.array(left_signed, copy=True)
        right_signed_smoothed = np.array(right_signed, copy=True)
        vergence_signed_smoothed = np.array(vergence_signed, copy=True)
        version_smoothed = np.array(version, copy=True)
        left_minor_signed_smoothed = np.array(left_minor_signed, copy=True)
        right_minor_signed_smoothed = np.array(right_minor_signed, copy=True)
        vergence_minor_signed_smoothed = np.array(vergence_minor_signed, copy=True)
        version_minor_smoothed = np.array(version_minor, copy=True)
        left_centroid_smoothed = np.array(left_centroid, copy=True)
        right_centroid_smoothed = np.array(right_centroid, copy=True)
        vergence_centroid_smoothed = np.array(vergence_centroid, copy=True)

    left_delta = _compute_delta(left_angles)
    right_delta = _compute_delta(right_angles)
    vergence_delta = _compute_delta(vergence)
    left_signed_delta = _compute_delta(left_signed)
    right_signed_delta = _compute_delta(right_signed)
    vergence_signed_delta = _compute_delta(vergence_signed)
    version_delta = _compute_delta(version)
    left_minor_delta = _compute_delta(left_minor_signed)
    right_minor_delta = _compute_delta(right_minor_signed)
    vergence_minor_delta = _compute_delta(vergence_minor_signed)
    version_minor_delta = _compute_delta(version_minor)
    left_centroid_delta = _compute_delta(left_centroid)
    right_centroid_delta = _compute_delta(right_centroid)
    vergence_centroid_delta = _compute_delta(vergence_centroid)

    left_delta_smoothed = _compute_delta(left_smoothed)
    right_delta_smoothed = _compute_delta(right_smoothed)
    vergence_delta_smoothed = _compute_delta(vergence_smoothed)
    left_signed_delta_smoothed = _compute_delta(left_signed_smoothed)
    right_signed_delta_smoothed = _compute_delta(right_signed_smoothed)
    vergence_signed_delta_smoothed = _compute_delta(vergence_signed_smoothed)
    version_delta_smoothed = _compute_delta(version_smoothed)
    left_minor_delta_smoothed = _compute_delta(left_minor_signed_smoothed)
    right_minor_delta_smoothed = _compute_delta(right_minor_signed_smoothed)
    vergence_minor_delta_smoothed = _compute_delta(vergence_minor_signed_smoothed)
    version_minor_delta_smoothed = _compute_delta(version_minor_smoothed)
    left_centroid_delta_smoothed = _compute_delta(left_centroid_smoothed)
    right_centroid_delta_smoothed = _compute_delta(right_centroid_smoothed)
    vergence_centroid_delta_smoothed = _compute_delta(vergence_centroid_smoothed)

    frame_arrays, frame_valid, frame_reason = _project_detection_arrays_to_frames(
        frame_indices,
        num_frames=num_frames,
        valid_frame=valid_frame,
        reason_codes=reason_codes,
        arrays={
            "left": left_angles,
            "right": right_angles,
            "vergence": vergence,
            "vergence_signed": vergence_signed,
            "vergence_signed_minor": vergence_minor_signed,
            "version": version,
            "version_minor": version_minor,
            "left_centroid": left_centroid,
            "right_centroid": right_centroid,
            "vergence_centroid": vergence_centroid,
        },
    )
    frame_left = frame_arrays["left"]
    frame_right = frame_arrays["right"]
    frame_vergence = frame_arrays["vergence"]
    frame_vergence_signed = frame_arrays["vergence_signed"]
    frame_vergence_signed_minor = frame_arrays["vergence_signed_minor"]
    frame_version = frame_arrays["version"]
    frame_version_minor = frame_arrays["version_minor"]
    frame_left_centroid = frame_arrays["left_centroid"]
    frame_right_centroid = frame_arrays["right_centroid"]
    frame_vergence_centroid = frame_arrays["vergence_centroid"]

    frame_smooth_window = _resolve_smoothing_window(num_frames, window_setting)
    if frame_smooth_window:
        frame_left_smoothed = _smooth_signal(frame_left, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_smoothed = _smooth_signal(frame_right, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_smoothed = _smooth_signal(frame_vergence, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_signed_smoothed = _smooth_signal(frame_vergence_signed, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_smoothed = _smooth_signal(frame_version, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_minor_signed_smoothed = _smooth_signal(frame_vergence_signed_minor, frame_smooth_window).astype(np.float32, copy=False)
        frame_version_minor_smoothed = _smooth_signal(frame_version_minor, frame_smooth_window).astype(np.float32, copy=False)
        frame_left_centroid_smoothed = _smooth_signal(frame_left_centroid, frame_smooth_window).astype(np.float32, copy=False)
        frame_right_centroid_smoothed = _smooth_signal(frame_right_centroid, frame_smooth_window).astype(np.float32, copy=False)
        frame_vergence_centroid_smoothed = _smooth_signal(frame_vergence_centroid, frame_smooth_window).astype(np.float32, copy=False)
    else:
        frame_left_smoothed = np.array(frame_left, copy=True)
        frame_right_smoothed = np.array(frame_right, copy=True)
        frame_vergence_smoothed = np.array(frame_vergence, copy=True)
        frame_vergence_signed_smoothed = np.array(frame_vergence_signed, copy=True)
        frame_version_smoothed = np.array(frame_version, copy=True)
        frame_vergence_minor_signed_smoothed = np.array(frame_vergence_signed_minor, copy=True)
        frame_version_minor_smoothed = np.array(frame_version_minor, copy=True)
        frame_left_centroid_smoothed = np.array(frame_left_centroid, copy=True)
        frame_right_centroid_smoothed = np.array(frame_right_centroid, copy=True)
        frame_vergence_centroid_smoothed = np.array(frame_vergence_centroid, copy=True)

    frame_left_delta = _compute_delta(frame_left)
    frame_right_delta = _compute_delta(frame_right)
    frame_vergence_delta = _compute_delta(frame_vergence)
    frame_vergence_signed_delta = _compute_delta(frame_vergence_signed)
    frame_vergence_minor_delta = _compute_delta(frame_vergence_signed_minor)
    frame_version_delta = _compute_delta(frame_version)
    frame_version_minor_delta = _compute_delta(frame_version_minor)
    frame_left_centroid_delta = _compute_delta(frame_left_centroid)
    frame_right_centroid_delta = _compute_delta(frame_right_centroid)
    frame_vergence_centroid_delta = _compute_delta(frame_vergence_centroid)

    frame_left_delta_smoothed = _compute_delta(frame_left_smoothed)
    frame_right_delta_smoothed = _compute_delta(frame_right_smoothed)
    frame_vergence_delta_smoothed = _compute_delta(frame_vergence_smoothed)
    frame_vergence_signed_delta_smoothed = _compute_delta(frame_vergence_signed_smoothed)
    frame_vergence_minor_delta_smoothed = _compute_delta(frame_vergence_minor_signed_smoothed)
    frame_version_delta_smoothed = _compute_delta(frame_version_smoothed)
    frame_version_minor_delta_smoothed = _compute_delta(frame_version_minor_smoothed)
    frame_left_centroid_delta_smoothed = _compute_delta(frame_left_centroid_smoothed)
    frame_right_centroid_delta_smoothed = _compute_delta(frame_right_centroid_smoothed)
    frame_vergence_centroid_delta_smoothed = _compute_delta(frame_vergence_centroid_smoothed)

    angles_group = run_group.require_group("angles")
    roi_group = angles_group.require_group("roi")
    frame_group = angles_group.require_group("frame")

    _prepare_output_arrays(
        roi_group,
        [
            ("left_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_minor_signed_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("version_minor_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("right_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("version_speed_deg_s", (total_detections,), (chunk_len,), "f4"),
            ("left_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("right_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("vergence_signed_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("version_accel_deg_s2", (total_detections,), (chunk_len,), "f4"),
            ("heading_deg", (total_detections,), (chunk_len,), "f4"),
            # Centroid-based angles (paper-comparable)
            ("left_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("left_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("right_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_delta_deg", (total_detections,), (chunk_len,), "f4"),
            ("vergence_centroid_delta_deg_smoothed", (total_detections,), (chunk_len,), "f4"),
        ],
    )
    roi_group["left_deg_smoothed"][:] = left_smoothed
    roi_group["left_delta_deg"][:] = left_delta
    roi_group["left_delta_deg_smoothed"][:] = left_delta_smoothed
    roi_group["right_deg_smoothed"][:] = right_smoothed
    roi_group["right_delta_deg"][:] = right_delta
    roi_group["right_delta_deg_smoothed"][:] = right_delta_smoothed
    roi_group["vergence_deg_smoothed"][:] = vergence_smoothed
    roi_group["vergence_delta_deg"][:] = vergence_delta
    roi_group["vergence_delta_deg_smoothed"][:] = vergence_delta_smoothed
    roi_group["left_signed_deg_smoothed"][:] = left_signed_smoothed
    roi_group["left_signed_delta_deg"][:] = left_signed_delta
    roi_group["left_signed_delta_deg_smoothed"][:] = left_signed_delta_smoothed
    roi_group["right_signed_deg_smoothed"][:] = right_signed_smoothed
    roi_group["right_signed_delta_deg"][:] = right_signed_delta
    roi_group["right_signed_delta_deg_smoothed"][:] = right_signed_delta_smoothed
    roi_group["vergence_signed_deg_smoothed"][:] = vergence_signed_smoothed
    roi_group["vergence_signed_delta_deg"][:] = vergence_signed_delta
    roi_group["vergence_signed_delta_deg_smoothed"][:] = vergence_signed_delta_smoothed
    roi_group["version_deg_smoothed"][:] = version_smoothed
    roi_group["version_delta_deg"][:] = version_delta
    roi_group["version_delta_deg_smoothed"][:] = version_delta_smoothed
    roi_group["left_minor_signed_deg_smoothed"][:] = left_minor_signed_smoothed
    roi_group["left_minor_signed_delta_deg"][:] = left_minor_delta
    roi_group["left_minor_signed_delta_deg_smoothed"][:] = left_minor_delta_smoothed
    roi_group["right_minor_signed_deg_smoothed"][:] = right_minor_signed_smoothed
    roi_group["right_minor_signed_delta_deg"][:] = right_minor_delta
    roi_group["right_minor_signed_delta_deg_smoothed"][:] = right_minor_delta_smoothed
    roi_group["vergence_minor_signed_deg_smoothed"][:] = vergence_minor_signed_smoothed
    roi_group["vergence_minor_signed_delta_deg"][:] = vergence_minor_delta
    roi_group["vergence_minor_signed_delta_deg_smoothed"][:] = vergence_minor_delta_smoothed
    roi_group["version_minor_deg_smoothed"][:] = version_minor_smoothed
    roi_group["version_minor_delta_deg"][:] = version_minor_delta
    roi_group["version_minor_delta_deg_smoothed"][:] = version_minor_delta_smoothed
    roi_group["left_speed_deg_s"][:] = left_speed
    roi_group["right_speed_deg_s"][:] = right_speed
    roi_group["vergence_speed_deg_s"][:] = vergence_speed
    roi_group["vergence_signed_speed_deg_s"][:] = vergence_signed_speed
    roi_group["version_speed_deg_s"][:] = version_speed
    roi_group["left_accel_deg_s2"][:] = left_accel
    roi_group["right_accel_deg_s2"][:] = right_accel
    roi_group["vergence_accel_deg_s2"][:] = vergence_accel
    roi_group["vergence_signed_accel_deg_s2"][:] = vergence_signed_accel
    roi_group["version_accel_deg_s2"][:] = version_accel
    # Centroid-based angles
    roi_group["left_centroid_deg_smoothed"][:] = left_centroid_smoothed
    roi_group["left_centroid_delta_deg"][:] = left_centroid_delta
    roi_group["left_centroid_delta_deg_smoothed"][:] = left_centroid_delta_smoothed
    roi_group["right_centroid_deg_smoothed"][:] = right_centroid_smoothed
    roi_group["right_centroid_delta_deg"][:] = right_centroid_delta
    roi_group["right_centroid_delta_deg_smoothed"][:] = right_centroid_delta_smoothed
    roi_group["vergence_centroid_deg_smoothed"][:] = vergence_centroid_smoothed
    roi_group["vergence_centroid_delta_deg"][:] = vergence_centroid_delta
    roi_group["vergence_centroid_delta_deg_smoothed"][:] = vergence_centroid_delta_smoothed

    _prepare_output_arrays(
        frame_group,
        [
            ("left_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_minor_signed_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("version_minor_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            # Centroid-based angles (paper-comparable)
            ("left_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("left_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("right_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_delta_deg", (num_frames,), (frame_chunk,), "f4"),
            ("vergence_centroid_delta_deg_smoothed", (num_frames,), (frame_chunk,), "f4"),
        ],
    )
    if num_frames > 0:
        frame_group["left_deg"][:] = frame_left
        frame_group["left_deg_smoothed"][:] = frame_left_smoothed
        frame_group["left_delta_deg"][:] = frame_left_delta
        frame_group["left_delta_deg_smoothed"][:] = frame_left_delta_smoothed
        frame_group["right_deg"][:] = frame_right
        frame_group["right_deg_smoothed"][:] = frame_right_smoothed
        frame_group["right_delta_deg"][:] = frame_right_delta
        frame_group["right_delta_deg_smoothed"][:] = frame_right_delta_smoothed
        frame_group["vergence_deg"][:] = frame_vergence
        frame_group["vergence_deg_smoothed"][:] = frame_vergence_smoothed
        frame_group["vergence_delta_deg"][:] = frame_vergence_delta
        frame_group["vergence_delta_deg_smoothed"][:] = frame_vergence_delta_smoothed
        frame_group["vergence_signed_deg"][:] = frame_vergence_signed
        frame_group["vergence_signed_deg_smoothed"][:] = frame_vergence_signed_smoothed
        frame_group["vergence_signed_delta_deg"][:] = frame_vergence_signed_delta
        frame_group["vergence_signed_delta_deg_smoothed"][:] = frame_vergence_signed_delta_smoothed
        frame_group["vergence_minor_signed_deg"][:] = frame_vergence_signed_minor
        frame_group["vergence_minor_signed_deg_smoothed"][:] = frame_vergence_minor_signed_smoothed
        frame_group["vergence_minor_signed_delta_deg"][:] = frame_vergence_minor_delta
        frame_group["vergence_minor_signed_delta_deg_smoothed"][:] = frame_vergence_minor_delta_smoothed
        frame_group["version_deg"][:] = frame_version
        frame_group["version_deg_smoothed"][:] = frame_version_smoothed
        frame_group["version_delta_deg"][:] = frame_version_delta
        frame_group["version_delta_deg_smoothed"][:] = frame_version_delta_smoothed
        frame_group["version_minor_deg"][:] = frame_version_minor
        frame_group["version_minor_deg_smoothed"][:] = frame_version_minor_smoothed
        frame_group["version_minor_delta_deg"][:] = frame_version_minor_delta
        frame_group["version_minor_delta_deg_smoothed"][:] = frame_version_minor_delta_smoothed
        # Centroid-based angles
        frame_group["left_centroid_deg"][:] = frame_left_centroid
        frame_group["left_centroid_deg_smoothed"][:] = frame_left_centroid_smoothed
        frame_group["left_centroid_delta_deg"][:] = frame_left_centroid_delta
        frame_group["left_centroid_delta_deg_smoothed"][:] = frame_left_centroid_delta_smoothed
        frame_group["right_centroid_deg"][:] = frame_right_centroid
        frame_group["right_centroid_deg_smoothed"][:] = frame_right_centroid_smoothed
        frame_group["right_centroid_delta_deg"][:] = frame_right_centroid_delta
        frame_group["right_centroid_delta_deg_smoothed"][:] = frame_right_centroid_delta_smoothed
        frame_group["vergence_centroid_deg"][:] = frame_vergence_centroid
        frame_group["vergence_centroid_deg_smoothed"][:] = frame_vergence_centroid_smoothed
        frame_group["vergence_centroid_delta_deg"][:] = frame_vergence_centroid_delta
        frame_group["vergence_centroid_delta_deg_smoothed"][:] = frame_vergence_centroid_delta_smoothed

    qa_group = run_group.require_group("qa")
    qa_roi = qa_group.require_group("roi")
    qa_frame = qa_group.require_group("frame")

    _prepare_output_arrays(
        qa_roi,
        [
            ("valid_left", (total_detections,), (chunk_len,), "bool"),
            ("valid_right", (total_detections,), (chunk_len,), "bool"),
            ("valid_frame", (total_detections,), (chunk_len,), "bool"),
            ("reason_codes", (total_detections,), (chunk_len,), "u2"),
        ],
    )

    _prepare_output_arrays(
        qa_frame,
        [
            ("valid_frame", (num_frames,), (frame_chunk,), "bool"),
            ("reason_codes", (num_frames,), (frame_chunk,), "u2"),
        ],
    )
    if num_frames > 0:
        qa_frame["valid_frame"][:] = frame_valid
        qa_frame["reason_codes"][:] = frame_reason

    support_group = run_group.require_group("support")
    _prepare_output_arrays(
        support_group,
        [
            ("frame_indices", (total_detections,), (chunk_len,), "i8"),
            ("time_seconds", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_major", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_minor", (total_detections,), (chunk_len,), "f4"),
            ("ellipse_ratio", (total_detections,), (chunk_len,), "f4"),
        ],
    )

    if num_frames > 0 and fps:
        frame_time = np.arange(num_frames, dtype=np.float32) / float(fps)
        if "frame_time_seconds" in support_group:
            del support_group["frame_time_seconds"]
        support_group.create_array(
            "frame_time_seconds",
            data=frame_time,
            chunks=(frame_chunk,),
            overwrite=True,
        )

    duration_seconds = float(time.perf_counter() - stage_start)
    rows_per_second = float(total_detections / duration_seconds) if duration_seconds > 0.0 else float("inf")
    timing_summary = {
        "total_detections": int(total_detections),
        "duration_seconds": duration_seconds,
        "rows_per_second": rows_per_second,
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
        "dask_chunk_size": int(chunk_size),
        "dask_version": getattr(dask, "__version__", "unknown"),
        "chunk_count": len(chunks),
        "chunk_timing_count": len(chunk_timings),
    }
    run_group.attrs.update(
        {
            "status": "complete",
            "report_version": "1.4",
            "reason_code_map": REASON_CODE_MAP,
            "source_eye_geometry_stage": eye_geometry.stage_group,
            "source_eye_geometry_run": eye_geometry.run_name,
            "source_subject_shape_run": eye_geometry.source_subject_shape_run,
            "source_refined_eye_run": eye_geometry.source_refined_eye_run,
            "source_refined_subject_masks_run": eye_geometry.source_refined_subject_run,
            **build_source_keypoints_attrs(keypoint_run_name, include_legacy_alias=True),
            "fps": float(fps) if fps else None,
            "num_detections": int(total_detections),
            "num_frames": int(num_frames),
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "execution_backend": backend,
            "dask_scheduler": scheduler_key,
            "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
            "dask_chunk_size": int(chunk_size),
            "dask_version": getattr(dask, "__version__", "unknown"),
            "eye_angle_timing_summary": json.loads(json.dumps(timing_summary, default=_to_serializable)),
            "valid_detection_fraction": float(valid_frame.sum() / total_detections) if total_detections else 0.0,
            "valid_frame_fraction": float(frame_valid.sum() / num_frames) if num_frames else 0.0,
            "circularity_reject_ratio": float(ELLIPSE_CIRCULARITY_THRESHOLD),
            **_eye_angle_definition_attrs(),
            "angle_smoothing_method": "moving_average",
            "angle_smoothing_window_detections": int(detection_smooth_window) if detection_smooth_window else None,
            "angle_smoothing_window_frames": int(frame_smooth_window) if frame_smooth_window else None,
            "angle_smoothing_window_requested": int(smoothing_window_param) if smoothing_window_param else None,
            # Centroid-based angles (paper-comparable)
            "centroid_angles": True,
            "centroid_angle_definition": "atan2(rotated_eye_vector_y, rotated_eye_vector_x) in fish frame",
            "centroid_vergence_definition": "abs(left_centroid_deg) + abs(right_centroid_deg)",
        }
    )
    if args.include_chunk_timings:
        run_group.attrs["eye_angle_chunk_timings"] = json.loads(json.dumps(chunk_timings, default=_to_serializable))
    parent_group.attrs["latest"] = resolved_run_name

    provenance = {
        "script": "fisheye.analysis.eye_angle_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "arguments": {
            "zarr_path": str(args.zarr_path),
            "eye_geometry_stage": eye_geometry.stage_group,
            "eye_geometry_run": eye_geometry.run_name,
            "subject_shape_run": eye_geometry.source_subject_shape_run,
            "refined_eye_run": eye_geometry.source_refined_eye_run,
            "refined_subject_run": eye_geometry.source_refined_subject_run,
            "keypoint_run": keypoint_run_name,
            "run_name": args.run_name,
            "chunk_size": chunk_size,
            "execution_backend": backend,
            "dask_scheduler": scheduler_key,
            "dask_num_workers": int(args.num_workers) if args.num_workers is not None else None,
            "fps_override": args.fps,
            "smoothing_window": smoothing_window_param,
        },
        "outputs": {
            "left_signed_deg": True,
            "right_signed_deg": True,
            "vergence_signed_deg": True,
            "version_deg": True,
            "left_deg_smoothed": True,
            "right_deg_smoothed": True,
            "vergence_deg_smoothed": True,
            "left_signed_deg_smoothed": True,
            "right_signed_deg_smoothed": True,
            "vergence_signed_deg_smoothed": True,
            "version_deg_smoothed": True,
            "vergence_signed_speed_deg_s": bool(fps),
            "version_speed_deg_s": bool(fps),
            "vergence_signed_accel_deg_s2": bool(fps),
            "version_accel_deg_s2": bool(fps),
            "ellipse_major": True,
            "ellipse_minor": True,
            "ellipse_ratio": True,
            "left_minor_signed_deg": True,
            "right_minor_signed_deg": True,
            "vergence_minor_signed_deg": True,
            "version_minor_deg": True,
            "left_minor_signed_deg_smoothed": True,
            "right_minor_signed_deg_smoothed": True,
            "vergence_minor_signed_deg_smoothed": True,
            "version_minor_deg_smoothed": True,
            "left_delta_deg": True,
            "right_delta_deg": True,
            "vergence_delta_deg": True,
            "left_signed_delta_deg": True,
            "right_signed_delta_deg": True,
            "vergence_signed_delta_deg": True,
            "version_delta_deg": True,
            "left_delta_deg_smoothed": True,
            "right_delta_deg_smoothed": True,
            "vergence_delta_deg_smoothed": True,
            "left_signed_delta_deg_smoothed": True,
            "right_signed_delta_deg_smoothed": True,
            "vergence_signed_delta_deg_smoothed": True,
            "version_delta_deg_smoothed": True,
            "left_minor_signed_delta_deg": True,
            "right_minor_signed_delta_deg": True,
            "vergence_minor_signed_delta_deg": True,
            "version_minor_delta_deg": True,
            "left_minor_signed_delta_deg_smoothed": True,
            "right_minor_signed_delta_deg_smoothed": True,
            "vergence_minor_signed_delta_deg_smoothed": True,
            "version_minor_delta_deg_smoothed": True,
            "frame_left_deg_smoothed": True,
            "frame_right_deg_smoothed": True,
            "frame_vergence_deg_smoothed": True,
            "frame_vergence_signed_deg_smoothed": True,
            "frame_version_deg_smoothed": True,
            "frame_vergence_minor_signed_deg": True,
            "frame_vergence_minor_signed_deg_smoothed": True,
            "frame_left_delta_deg": True,
            "frame_right_delta_deg": True,
            "frame_vergence_delta_deg": True,
            "frame_vergence_signed_delta_deg": True,
            "frame_version_delta_deg": True,
            "frame_left_delta_deg_smoothed": True,
            "frame_right_delta_deg_smoothed": True,
            "frame_vergence_delta_deg_smoothed": True,
            "frame_vergence_signed_delta_deg_smoothed": True,
            "frame_version_delta_deg_smoothed": True,
            "frame_vergence_minor_signed_delta_deg": True,
            "frame_vergence_minor_signed_delta_deg_smoothed": True,
            "frame_version_minor_delta_deg": True,
            "frame_version_minor_delta_deg_smoothed": True,
            "frame_version_minor_deg": True,
            "frame_version_minor_deg_smoothed": True,
            # Centroid-based angles (paper-comparable)
            "left_centroid_deg": True,
            "right_centroid_deg": True,
            "vergence_centroid_deg": True,
            "left_centroid_deg_smoothed": True,
            "right_centroid_deg_smoothed": True,
            "vergence_centroid_deg_smoothed": True,
            "left_centroid_delta_deg": True,
            "right_centroid_delta_deg": True,
            "vergence_centroid_delta_deg": True,
            "left_centroid_delta_deg_smoothed": True,
            "right_centroid_delta_deg_smoothed": True,
            "vergence_centroid_delta_deg_smoothed": True,
            "frame_left_centroid_deg": True,
            "frame_right_centroid_deg": True,
            "frame_vergence_centroid_deg": True,
            "frame_left_centroid_deg_smoothed": True,
            "frame_right_centroid_deg_smoothed": True,
            "frame_vergence_centroid_deg_smoothed": True,
            "frame_left_centroid_delta_deg": True,
            "frame_right_centroid_delta_deg": True,
            "frame_vergence_centroid_delta_deg": True,
            "frame_left_centroid_delta_deg_smoothed": True,
            "frame_right_centroid_delta_deg_smoothed": True,
            "frame_vergence_centroid_delta_deg_smoothed": True,
        },
        "valid_reason_counts": _count_reason_bits(reason_codes),
        "frame_reason_counts": _count_reason_bits(frame_reason) if num_frames else {},
    }
    run_group.attrs["provenance"] = json.loads(json.dumps(provenance, default=_to_serializable))

    if not args.quiet:
        console.print(
            f"[green]✓[/green] Eye angle analysis saved to "
            f"[cyan]analysis/eye_angle_runs/{resolved_run_name}[/cyan]"
        )
        console.print(
            f"Valid detections: {valid_frame.sum()} / {total_detections} "
            f"({(valid_frame.sum() / total_detections * 100.0) if total_detections else 0:.1f}%)"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
