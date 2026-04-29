"""Pure helpers for subject-body centerline B-spline fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised indirectly when scipy is installed
    from scipy import interpolate
except Exception:  # pragma: no cover - optional dependency guard
    interpolate = None  # type: ignore[assignment]


BSPLINE_METHOD = "centerline_scipy_splprep_v1"
DEFAULT_BSPLINE_DEGREE = 3
DEFAULT_BSPLINE_SMOOTHING = 0.0
DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT = 256
DEFAULT_TAIL_SAMPLE_COUNT = 32


@dataclass(frozen=True)
class SubjectBodySplineBatch:
    bspline_control_points_xy: np.ndarray
    bspline_knots: np.ndarray
    bspline_degree_used: np.ndarray
    bspline_sample_xy: np.ndarray
    bspline_valid: np.ndarray
    bspline_failure_reasons: np.ndarray
    bspline_arc_length_px: np.ndarray
    tail_sample_s: np.ndarray
    tail_sample_xy: np.ndarray
    tail_tangent_xy: np.ndarray
    tail_normal_xy: np.ndarray
    tail_curvature_px_inv: np.ndarray
    tail_sample_valid: np.ndarray
    tail_sample_failure_reasons: np.ndarray


def tail_sample_positions(sample_count: int = DEFAULT_TAIL_SAMPLE_COUNT) -> np.ndarray:
    """Return normalized tail-sample positions from tail base to tail tip."""
    return np.linspace(0.0, 1.0, int(sample_count), dtype=np.float32)


def _polyline_arclength(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or int(points.shape[0]) < 2:
        return np.zeros((0,), dtype=np.float64), 0.0
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(segment_lengths)])
    return cumulative, float(cumulative[-1])


def _clean_centerline(points_xy: np.ndarray) -> Optional[np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if int(points.shape[0]) < 2:
        return None
    keep = np.ones((int(points.shape[0]),), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6
    points = points[keep]
    if int(points.shape[0]) < 2:
        return None
    return points


def _resample_polyline(points_xy: np.ndarray, sample_count: int) -> tuple[Optional[np.ndarray], float]:
    points = _clean_centerline(points_xy)
    if points is None:
        return None, 0.0
    cumulative, total = _polyline_arclength(points)
    if total <= 1e-6:
        return None, total
    targets = np.linspace(0.0, total, int(sample_count), dtype=np.float64)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1), total


def _unit_vectors(vectors_xy: np.ndarray) -> Optional[np.ndarray]:
    vectors = np.asarray(vectors_xy, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        return None
    return vectors / norms[:, None]


def _fit_one(
    centerline_xy: np.ndarray,
    *,
    centerline_valid: bool,
    tail_base_valid: bool,
    tail_base_arclength_px: float,
    centerline_invalid_reason: str,
    tail_base_invalid_reason: str,
    centerline_sample_count: int,
    tail_sample_count: int,
    degree: int,
    smoothing: float,
    arclength_sample_count: int,
) -> dict[str, object]:
    control = np.full((int(centerline_sample_count), 2), np.nan, dtype=np.float32)
    knots = np.full((int(centerline_sample_count) + int(degree) + 1,), np.nan, dtype=np.float32)
    sample = np.full((int(centerline_sample_count), 2), np.nan, dtype=np.float32)
    tail_xy = np.full((int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_tangent = np.full((int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_normal = np.full((int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_curvature = np.full((int(tail_sample_count),), np.nan, dtype=np.float32)

    result: dict[str, object] = {
        "control": control,
        "knots": knots,
        "degree_used": np.int16(-1),
        "sample": sample,
        "bspline_valid": False,
        "bspline_reason": str(centerline_invalid_reason or "missing_centerline"),
        "arc_length": np.float32(np.nan),
        "tail_xy": tail_xy,
        "tail_tangent": tail_tangent,
        "tail_normal": tail_normal,
        "tail_curvature": tail_curvature,
        "tail_valid": False,
        "tail_reason": str(centerline_invalid_reason or "missing_centerline"),
    }

    if not bool(centerline_valid):
        return result
    if interpolate is None:
        result["bspline_reason"] = "scipy_unavailable"
        result["tail_reason"] = "scipy_unavailable"
        return result

    fitted_points, source_total = _resample_polyline(centerline_xy, int(centerline_sample_count))
    if fitted_points is None or source_total <= 1e-6:
        result["bspline_reason"] = "insufficient_centerline_points"
        result["tail_reason"] = "insufficient_centerline_points"
        return result
    if int(fitted_points.shape[0]) <= int(degree):
        result["bspline_reason"] = "insufficient_centerline_points"
        result["tail_reason"] = "insufficient_centerline_points"
        return result

    u = np.linspace(0.0, 1.0, int(centerline_sample_count), dtype=np.float64)
    try:
        tck, _u_out = interpolate.splprep(
            [fitted_points[:, 0], fitted_points[:, 1]],
            u=u,
            k=int(degree),
            s=float(smoothing),
        )
        knot_values, coeff_values, degree_used = tck
        coeff_xy = np.stack([np.asarray(coeff_values[0]), np.asarray(coeff_values[1])], axis=1)
        if int(coeff_xy.shape[0]) > int(control.shape[0]) or int(len(knot_values)) > int(knots.shape[0]):
            result["bspline_reason"] = "unexpected_spline_shape"
            result["tail_reason"] = "unexpected_spline_shape"
            return result
        eval_u = np.linspace(0.0, 1.0, int(centerline_sample_count), dtype=np.float64)
        sample_xy = np.stack(interpolate.splev(eval_u, tck, der=0), axis=1)
        dense_u = np.linspace(0.0, 1.0, max(8, int(arclength_sample_count)), dtype=np.float64)
        dense_xy = np.stack(interpolate.splev(dense_u, tck, der=0), axis=1)
    except Exception:
        result["bspline_reason"] = "spline_fit_failed"
        result["tail_reason"] = "spline_fit_failed"
        return result

    control[: int(coeff_xy.shape[0]), :] = coeff_xy.astype(np.float32)
    knots[: int(len(knot_values))] = np.asarray(knot_values, dtype=np.float32)
    sample[:, :] = sample_xy.astype(np.float32)
    _dense_cumulative, dense_total = _polyline_arclength(dense_xy)

    result["degree_used"] = np.int16(int(degree_used))
    result["bspline_valid"] = True
    result["bspline_reason"] = "ok"
    result["arc_length"] = np.float32(dense_total)

    if not bool(tail_base_valid):
        result["tail_reason"] = str(tail_base_invalid_reason or "missing_tail_base")
        return result
    if not np.isfinite(float(tail_base_arclength_px)):
        result["tail_reason"] = "missing_tail_base"
        return result
    tail_base_u = float(np.clip(float(tail_base_arclength_px) / max(source_total, 1e-6), 0.0, 1.0))
    if 1.0 - tail_base_u <= 1e-6:
        result["tail_reason"] = "tail_segment_too_short"
        return result

    tail_s = tail_sample_positions(int(tail_sample_count)).astype(np.float64)
    tail_u = tail_base_u + tail_s * (1.0 - tail_base_u)
    try:
        tail_points = np.stack(interpolate.splev(tail_u, tck, der=0), axis=1)
        first_derivative = np.stack(interpolate.splev(tail_u, tck, der=1), axis=1)
        second_derivative = np.stack(interpolate.splev(tail_u, tck, der=2), axis=1)
    except Exception:
        result["tail_reason"] = "spline_eval_failed"
        return result

    tangents = _unit_vectors(first_derivative)
    if tangents is None:
        result["tail_reason"] = "tail_spline_derivative_failed"
        return result
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    numerator = first_derivative[:, 0] * second_derivative[:, 1] - first_derivative[:, 1] * second_derivative[:, 0]
    denominator = np.power(np.sum(first_derivative * first_derivative, axis=1), 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = numerator / denominator
    if np.any(~np.isfinite(curvature)):
        result["tail_reason"] = "tail_curvature_failed"
        return result

    tail_xy[:, :] = tail_points.astype(np.float32)
    tail_tangent[:, :] = tangents.astype(np.float32)
    tail_normal[:, :] = normals.astype(np.float32)
    tail_curvature[:] = curvature.astype(np.float32)
    result["tail_valid"] = True
    result["tail_reason"] = "ok"
    return result


def fit_subject_body_spline_batch(
    centerline_xy: np.ndarray,
    centerline_valid: np.ndarray,
    tail_base_valid: np.ndarray,
    tail_base_arclength_px: np.ndarray,
    *,
    centerline_failure_reasons: Optional[Sequence[object]] = None,
    tail_base_failure_reasons: Optional[Sequence[object]] = None,
    centerline_sample_count: int,
    tail_sample_count: int = DEFAULT_TAIL_SAMPLE_COUNT,
    degree: int = DEFAULT_BSPLINE_DEGREE,
    smoothing: float = DEFAULT_BSPLINE_SMOOTHING,
    arclength_sample_count: int = DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT,
) -> SubjectBodySplineBatch:
    """Fit per-row B-splines from ordered body centerlines.

    The helper is intentionally Zarr-free: callers own persistence, provenance,
    and source-QC policy. Invalid source rows return NaNs plus reason strings.
    """
    centerlines = np.asarray(centerline_xy, dtype=np.float64)
    valid = np.asarray(centerline_valid, dtype=bool).reshape(-1)
    tail_valid = np.asarray(tail_base_valid, dtype=bool).reshape(-1)
    tail_base = np.asarray(tail_base_arclength_px, dtype=np.float64).reshape(-1)
    row_count = int(centerlines.shape[0])
    if valid.shape[0] != row_count or tail_valid.shape[0] != row_count or tail_base.shape[0] != row_count:
        raise ValueError("centerline and tail-base arrays must have the same row count")
    if int(centerlines.shape[1]) != int(centerline_sample_count) or int(centerlines.shape[2]) != 2:
        raise ValueError("centerline_xy shape must be (N, centerline_sample_count, 2)")

    centerline_reasons = (
        np.asarray(centerline_failure_reasons, dtype=object)
        if centerline_failure_reasons is not None
        else np.full((row_count,), "missing_centerline", dtype=object)
    )
    tail_reasons = (
        np.asarray(tail_base_failure_reasons, dtype=object)
        if tail_base_failure_reasons is not None
        else np.full((row_count,), "missing_tail_base", dtype=object)
    )

    control = np.full((row_count, int(centerline_sample_count), 2), np.nan, dtype=np.float32)
    knots = np.full((row_count, int(centerline_sample_count) + int(degree) + 1), np.nan, dtype=np.float32)
    degree_used = np.full((row_count,), -1, dtype=np.int16)
    sample = np.full((row_count, int(centerline_sample_count), 2), np.nan, dtype=np.float32)
    bspline_valid = np.zeros((row_count,), dtype=bool)
    bspline_reasons = np.full((row_count,), "missing_centerline", dtype=object)
    arc_length = np.full((row_count,), np.nan, dtype=np.float32)
    tail_s = tail_sample_positions(int(tail_sample_count))
    tail_xy = np.full((row_count, int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_tangent = np.full((row_count, int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_normal = np.full((row_count, int(tail_sample_count), 2), np.nan, dtype=np.float32)
    tail_curvature = np.full((row_count, int(tail_sample_count)), np.nan, dtype=np.float32)
    tail_sample_valid = np.zeros((row_count,), dtype=bool)
    tail_sample_reasons = np.full((row_count,), "missing_centerline", dtype=object)

    for row_idx in range(row_count):
        fitted = _fit_one(
            centerlines[row_idx],
            centerline_valid=bool(valid[row_idx]),
            tail_base_valid=bool(tail_valid[row_idx]),
            tail_base_arclength_px=float(tail_base[row_idx]),
            centerline_invalid_reason=str(centerline_reasons[row_idx] or "missing_centerline"),
            tail_base_invalid_reason=str(tail_reasons[row_idx] or "missing_tail_base"),
            centerline_sample_count=int(centerline_sample_count),
            tail_sample_count=int(tail_sample_count),
            degree=int(degree),
            smoothing=float(smoothing),
            arclength_sample_count=int(arclength_sample_count),
        )
        control[row_idx] = fitted["control"]  # type: ignore[assignment]
        knots[row_idx] = fitted["knots"]  # type: ignore[assignment]
        degree_used[row_idx] = fitted["degree_used"]  # type: ignore[assignment]
        sample[row_idx] = fitted["sample"]  # type: ignore[assignment]
        bspline_valid[row_idx] = bool(fitted["bspline_valid"])
        bspline_reasons[row_idx] = str(fitted["bspline_reason"])
        arc_length[row_idx] = fitted["arc_length"]  # type: ignore[assignment]
        tail_xy[row_idx] = fitted["tail_xy"]  # type: ignore[assignment]
        tail_tangent[row_idx] = fitted["tail_tangent"]  # type: ignore[assignment]
        tail_normal[row_idx] = fitted["tail_normal"]  # type: ignore[assignment]
        tail_curvature[row_idx] = fitted["tail_curvature"]  # type: ignore[assignment]
        tail_sample_valid[row_idx] = bool(fitted["tail_valid"])
        tail_sample_reasons[row_idx] = str(fitted["tail_reason"])

    return SubjectBodySplineBatch(
        bspline_control_points_xy=control,
        bspline_knots=knots,
        bspline_degree_used=degree_used,
        bspline_sample_xy=sample,
        bspline_valid=bspline_valid,
        bspline_failure_reasons=bspline_reasons,
        bspline_arc_length_px=arc_length,
        tail_sample_s=tail_s,
        tail_sample_xy=tail_xy,
        tail_tangent_xy=tail_tangent,
        tail_normal_xy=tail_normal,
        tail_curvature_px_inv=tail_curvature,
        tail_sample_valid=tail_sample_valid,
        tail_sample_failure_reasons=tail_sample_reasons,
    )
