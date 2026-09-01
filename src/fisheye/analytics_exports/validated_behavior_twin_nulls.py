"""Rotated virtual-twin null summaries for validated-behavior chaser exports.

The legacy assay suite separated "fish avoids the chaser" from "fish hugs the
wall" by rotating the chaser position about the arena center (60/120/180/240/
300 degrees; never 90 for symmetry reasons) and recomputing fish-to-chaser
distance summaries against each distance-from-center-matched virtual twin.
Observed minus twin is the object-specific effect.  The phase-b validated
behavior export carries no twin surface, so this module derives one from
``chaser_relative_samples`` frames.

Honest-scope notes (recorded in the CLI manifest as ``policy_parity:
"approximate_v1"``):

- Distance quantiles, valid counts, and ``near_zone_fraction_valid`` follow
  the same definitions as ``radial_near_field_summary`` (valid = selection
  member AND chaser occurrence AND behavior-role valid AND physical-relative
  valid, near zone uses ``distance <= near_zone_radius_mm``); rotation 0 must
  reproduce the published summary and the CLI reports the deviations.
- The hysteresis entry count reimplements the entry/gap-censoring idea of the
  exact-session-time visit policy (entry counted only after an observed sample
  beyond the entry radius within the same contiguous valid segment; segments
  break on missing/invalid frames or non-increasing timestamps).  Dwell and
  censor bookkeeping fields of the source policy are not reproduced here, so
  exact policy parity is not claimed even though entry counts are expected to
  match at rotation 0.
- The export publishes no explicit arena-center coordinates (the
  ``reviewed_arena_and_scale`` binding carries only digests and zarr refs), so
  the center is recovered per recording x provider by solving for the point
  whose per-epoch mean fish distance reproduces the summary's
  ``fish_arena_radius_mean_mm`` values.  The fit residual is persisted per row
  so a bad recovery is visible, and the CLI refuses recordings whose residual
  exceeds a strict tolerance.

This module is pure compute (numpy only); all export I/O lives in
``fisheye.utils.compute_validated_behavior_twin_nulls``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ROTATION_DEGREES: tuple[int, ...] = (0, 60, 120, 180, 240, 300)
"""Legacy twin rotation set: 0 is the observed chaser, never 90 degrees."""

POLICY_PARITY = "approximate_v1"

DISTANCE_QUANTILES: tuple[tuple[str, float], ...] = (
    ("distance_p05_mm", 5.0),
    ("distance_p25_mm", 25.0),
    ("distance_p50_mm", 50.0),
    ("distance_p75_mm", 75.0),
    ("distance_p95_mm", 95.0),
)


class TwinNullError(ValueError):
    """Raised when twin-null inputs violate their contract."""


def _fail(message: str) -> None:
    raise TwinNullError(message)


def rotate_points_about_center(
    xy_mm: np.ndarray, center_mm: np.ndarray, rotation_deg: float
) -> np.ndarray:
    """Rotate ``xy_mm`` (N x 2) about ``center_mm`` by ``rotation_deg`` CCW.

    NaN coordinates propagate; distance from the center is preserved exactly
    up to floating point for every finite point.
    """

    points = np.asarray(xy_mm, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64).reshape(2)
    if points.ndim != 2 or points.shape[1] != 2:
        _fail("rotate_points_about_center expects an (N, 2) array.")
    if not np.all(np.isfinite(center)):
        _fail("Arena center must be finite to rotate about it.")
    theta = math.radians(float(rotation_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    shifted = points - center
    rotated = np.empty_like(shifted)
    rotated[:, 0] = cos_t * shifted[:, 0] - sin_t * shifted[:, 1]
    rotated[:, 1] = sin_t * shifted[:, 0] + cos_t * shifted[:, 1]
    return rotated + center


@dataclass(frozen=True)
class CenterFitResult:
    """Recovered arena center with its constraint residual diagnostics."""

    center_mm: tuple[float, float]
    max_abs_residual_mm: float
    constraint_count: int
    iterations: int


def fit_arena_center_mm(
    fish_xy_mm: np.ndarray,
    constraints: Sequence[tuple[np.ndarray, float]],
    *,
    max_iterations: int = 100,
    tolerance_mm: float = 1e-9,
) -> CenterFitResult:
    """Solve for the arena center from mean-radius constraints.

    Each constraint is ``(valid_mask, target_mean_radius_mm)`` asserting that
    the mean of ``|fish_xy_mm[valid_mask] - center|`` equals the published
    ``fish_arena_radius_mean_mm``.  With three or more constraints over
    different frame subsets the two-parameter Gauss-Newton solve is strongly
    over-determined; residuals stay near machine precision when the frames
    match the frames the summary actually consumed.
    """

    points = np.asarray(fish_xy_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        _fail("fit_arena_center_mm expects an (N, 2) fish position array.")
    usable: list[tuple[np.ndarray, float]] = []
    for mask, target in constraints:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.shape[0] != points.shape[0]:
            _fail("Center-fit constraint mask length differs from positions.")
        if not math.isfinite(float(target)):
            continue
        if int(np.count_nonzero(mask_arr)) == 0:
            continue
        usable.append((mask_arr, float(target)))
    if len(usable) < 3:
        _fail(
            "Arena-center recovery needs at least three finite mean-radius "
            f"constraints; got {len(usable)}."
        )
    all_mask = np.zeros(points.shape[0], dtype=bool)
    for mask_arr, _ in usable:
        all_mask |= mask_arr
    center = points[all_mask].mean(axis=0)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        residual = np.empty(len(usable), dtype=np.float64)
        jacobian = np.empty((len(usable), 2), dtype=np.float64)
        for row, (mask_arr, target) in enumerate(usable):
            delta = center[None, :] - points[mask_arr]
            norms = np.hypot(delta[:, 0], delta[:, 1])
            safe = np.maximum(norms, 1e-12)
            residual[row] = float(norms.mean()) - target
            jacobian[row] = (delta / safe[:, None]).mean(axis=0)
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        center = center + step
        if float(np.hypot(step[0], step[1])) < tolerance_mm:
            break
    final = np.empty(len(usable), dtype=np.float64)
    for row, (mask_arr, target) in enumerate(usable):
        delta = center[None, :] - points[mask_arr]
        final[row] = float(np.hypot(delta[:, 0], delta[:, 1]).mean()) - target
    return CenterFitResult(
        center_mm=(float(center[0]), float(center[1])),
        max_abs_residual_mm=float(np.max(np.abs(final))),
        constraint_count=len(usable),
        iterations=iterations,
    )


@dataclass(frozen=True)
class HysteresisEntryStats:
    """Entry-count summary under the approximate gap-censored visit policy."""

    entry_count: int
    valid_tracked_duration_s: float
    entry_rate_per_min_valid_time: float
    invalid_gap_count: int


def hysteresis_entry_stats(
    *,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    distance_mm: np.ndarray,
    distance_valid: np.ndarray,
    enter_mm: float,
    exit_mm: float,
) -> HysteresisEntryStats:
    """Count hysteretic near-zone entries with valid-gap censoring.

    An entry is counted only when an observed sample crosses below
    ``enter_mm`` after an observed sample beyond ``exit_mm`` within the same
    contiguous segment.  Segments break whenever consecutive observed samples
    are not frame-adjacent with strictly increasing valid timestamps, so a
    presence inside the zone at a segment start is censored, never counted.
    Tracked duration integrates exact timestamp deltas over contiguous
    observed pairs (matching the exact-session-time policy's denominator).
    """

    frame = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    timestamp = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    ts_ok = np.asarray(timestamp_valid, dtype=bool).reshape(-1)
    distance = np.asarray(distance_mm, dtype=np.float64).reshape(-1)
    observed = (
        np.asarray(distance_valid, dtype=bool).reshape(-1)
        & ts_ok
        & np.isfinite(distance)
    )
    if not (frame.size == timestamp.size == distance.size == observed.size):
        _fail("Hysteresis vectors have inconsistent lengths.")
    if frame.size > 1 and np.any(np.diff(frame) <= 0):
        _fail("Hysteresis frame IDs must be strictly increasing.")
    if not (0.0 < float(enter_mm) <= float(exit_mm)):
        _fail("Hysteresis radii require 0 < enter_mm <= exit_mm.")

    index = np.flatnonzero(observed)
    if index.size == 0:
        return HysteresisEntryStats(0, 0.0, math.nan, 0)
    values = distance[index]
    obs_frames = frame[index]
    obs_ts = timestamp[index]
    contiguous = np.zeros(index.size, dtype=bool)
    if index.size > 1:
        contiguous[1:] = (np.diff(obs_frames) == 1) & (np.diff(obs_ts) > 0)
    segment = np.cumsum(~contiguous)
    tracked_s = (
        float(np.sum(np.diff(obs_ts)[contiguous[1:]])) / 1e9
        if index.size > 1
        else 0.0
    )
    gap_count = int(np.count_nonzero(~contiguous[1:]))

    below_enter = values < float(enter_mm)
    above_exit = values > float(exit_mm)
    trigger = below_enter | above_exit
    trig_idx = np.flatnonzero(trigger)
    entries = 0
    if trig_idx.size:
        trig_seg = segment[trig_idx]
        trig_inside = below_enter[trig_idx]
        prev_same_segment = np.zeros(trig_idx.size, dtype=bool)
        prev_was_outside = np.zeros(trig_idx.size, dtype=bool)
        if trig_idx.size > 1:
            prev_same_segment[1:] = trig_seg[1:] == trig_seg[:-1]
            prev_was_outside[1:] = ~trig_inside[:-1]
        entries = int(
            np.count_nonzero(trig_inside & prev_same_segment & prev_was_outside)
        )
    rate = float(entries) / (tracked_s / 60.0) if tracked_s > 0 else math.nan
    return HysteresisEntryStats(entries, tracked_s, rate, gap_count)


def summarize_distances(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    *,
    near_zone_radius_mm: float,
) -> dict[str, object]:
    """Distance quantiles and near-zone fraction over the valid frames.

    Mirrors ``provider_chaser_position_suite``: quantiles use linear
    ``np.percentile`` over valid finite distances, the near zone is
    ``distance <= near_zone_radius_mm``, and fractions are None when no valid
    frame exists.
    """

    distance = np.asarray(distance_mm, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if distance.shape != mask.shape:
        _fail("summarize_distances requires aligned distance/valid vectors.")
    with np.errstate(invalid="ignore"):
        near = mask & (distance <= float(near_zone_radius_mm))
    values = distance[mask]
    finite = values[np.isfinite(values)]
    valid_count = int(finite.size)
    near_count = int(np.count_nonzero(near))
    row: dict[str, object] = {
        "valid_distance_frame_count": valid_count,
        "near_zone_frame_count": near_count,
        "near_zone_fraction_valid": (
            float(near_count) / valid_count if valid_count else None
        ),
        "distance_mean_mm": float(np.mean(finite)) if valid_count else None,
    }
    for name, percentile in DISTANCE_QUANTILES:
        row[name] = float(np.percentile(finite, percentile)) if valid_count else None
    return row


@dataclass(frozen=True)
class TwinEpochWindow:
    """One half-open semantic epoch window on the acquisition frame axis."""

    epoch_window_id: int
    epoch_role: str
    start_frame: int
    end_frame_exclusive: int


@dataclass(frozen=True)
class TwinChaserTrack:
    """Frame-axis arrays for one chaser under one provider role."""

    chaser_identity_code: int
    chaser_identity: str
    behavior_role: str
    chaser_xy_mm: np.ndarray
    valid: np.ndarray


def compute_twin_rows_for_provider(
    *,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    fish_xy_mm: np.ndarray,
    chasers: Sequence[TwinChaserTrack],
    epochs: Sequence[TwinEpochWindow],
    center_mm: np.ndarray,
    near_zone_radius_mm: float,
    near_entry_radius_mm: float,
    near_exit_radius_mm: float,
    rotations_deg: Sequence[int] = ROTATION_DEGREES,
) -> list[dict[str, object]]:
    """Compute one summary row per epoch x chaser x rotation.

    ``valid`` per chaser must already encode the summary's candidate policy
    (selection member AND occurrence AND role valid AND physical-relative
    valid); the epoch restriction is applied here.
    """

    frame = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    fish = np.asarray(fish_xy_mm, dtype=np.float64)
    if fish.shape != (frame.size, 2):
        _fail("fish_xy_mm must align with the frame axis as (N, 2).")
    rows: list[dict[str, object]] = []
    for chaser in chasers:
        chaser_xy = np.asarray(chaser.chaser_xy_mm, dtype=np.float64)
        valid = np.asarray(chaser.valid, dtype=bool).reshape(-1)
        if chaser_xy.shape != (frame.size, 2) or valid.size != frame.size:
            _fail("Chaser arrays must align with the frame axis.")
        for rotation in rotations_deg:
            rotated = (
                chaser_xy
                if int(rotation) == 0
                else rotate_points_about_center(chaser_xy, center_mm, rotation)
            )
            delta = fish - rotated
            distance = np.hypot(delta[:, 0], delta[:, 1])
            for epoch in epochs:
                in_epoch = (frame >= epoch.start_frame) & (
                    frame < epoch.end_frame_exclusive
                )
                epoch_valid = valid & in_epoch
                row: dict[str, object] = {
                    "epoch_window_id": int(epoch.epoch_window_id),
                    "epoch_role": epoch.epoch_role,
                    "chaser_identity_code": int(chaser.chaser_identity_code),
                    "chaser_identity": chaser.chaser_identity,
                    "behavior_role": chaser.behavior_role,
                    "rotation_deg": int(rotation),
                }
                row.update(
                    summarize_distances(
                        distance,
                        epoch_valid,
                        near_zone_radius_mm=near_zone_radius_mm,
                    )
                )
                visits = hysteresis_entry_stats(
                    frame_id=frame[in_epoch],
                    timestamp_ns=np.asarray(timestamp_ns, dtype=np.int64)[in_epoch],
                    timestamp_valid=np.asarray(timestamp_valid, dtype=bool)[in_epoch],
                    distance_mm=distance[in_epoch],
                    distance_valid=epoch_valid[in_epoch],
                    enter_mm=near_entry_radius_mm,
                    exit_mm=near_exit_radius_mm,
                )
                row["near_zone_entry_count"] = visits.entry_count
                row["near_zone_valid_tracked_duration_s"] = (
                    visits.valid_tracked_duration_s
                )
                row["near_zone_entry_rate_per_min_valid_time"] = (
                    visits.entry_rate_per_min_valid_time
                )
                row["near_zone_invalid_gap_count"] = visits.invalid_gap_count
                rows.append(row)
    return rows


EXCESS_METRICS: tuple[str, ...] = ("near_zone_fraction_valid", "distance_p50_mm")


def compute_twin_excess(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str] = EXCESS_METRICS,
    key_fields: Sequence[str] = (
        "recording_id",
        "provider_role",
        "epoch_role",
        "epoch_window_id",
        "chaser_identity_code",
        "chaser_identity",
        "behavior_role",
    ),
) -> list[dict[str, object]]:
    """Observed (rotation 0) minus mean-over-twins per grouping key.

    A group contributes a metric only when the observed row and every twin
    row carry a finite value for it; otherwise the excess is None.
    """

    groups: dict[tuple, dict[int, Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, {})[int(row["rotation_deg"])] = row  # type: ignore[arg-type]
    twin_set = [deg for deg in ROTATION_DEGREES if deg != 0]
    result: list[dict[str, object]] = []
    for key in sorted(groups, key=repr):
        by_rotation = groups[key]
        if sorted(by_rotation) != sorted(ROTATION_DEGREES):
            _fail(f"Twin group {key!r} lacks the full rotation set.")
        out: dict[str, object] = dict(zip(key_fields, key))
        for metric in metrics:
            observed = by_rotation[0].get(metric)
            twins = [by_rotation[deg].get(metric) for deg in twin_set]
            usable = (
                observed is not None
                and math.isfinite(float(observed))
                and all(
                    value is not None and math.isfinite(float(value))
                    for value in twins
                )
            )
            if usable:
                twin_mean = float(np.mean([float(value) for value in twins]))
                out[f"{metric}_observed"] = float(observed)  # type: ignore[arg-type]
                out[f"{metric}_twin_mean"] = twin_mean
                out[f"{metric}_excess"] = float(observed) - twin_mean  # type: ignore[arg-type]
            else:
                out[f"{metric}_observed"] = None
                out[f"{metric}_twin_mean"] = None
                out[f"{metric}_excess"] = None
        result.append(out)
    return result


__all__ = [
    "ROTATION_DEGREES",
    "POLICY_PARITY",
    "EXCESS_METRICS",
    "TwinNullError",
    "CenterFitResult",
    "HysteresisEntryStats",
    "TwinEpochWindow",
    "TwinChaserTrack",
    "rotate_points_about_center",
    "fit_arena_center_mm",
    "hysteresis_entry_stats",
    "summarize_distances",
    "compute_twin_rows_for_provider",
    "compute_twin_excess",
]
