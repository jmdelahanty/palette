"""Pure provider-aware, position-only chaser analytics.

This module computes spatial relationships from one already-sealed position
provider.  It deliberately has no provider discovery, Zarr selection, protocol
label inference, motion, heading, bout, gaze, or trial logic.  Callers must bind
exact half-open acquisition-frame epochs, one reviewed circular arena boundary,
one physical source-camera scale, and explicit behavior-role identities.

The moving-chaser radial null is a geometric correction only.  During closed-
loop pursuit it must not be interpreted as a behavioral independence null.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.chaser_near_field_occupancy import (
    DEFAULT_R_IN_MM,
    DEFAULT_R_OUT_MM,
    DEFAULT_R_ZONE_MM,
    VALID_TRACKED_VISIT_POLICY,
    compute_hysteresis_visits,
)
from fisheye.analysis.chaser_radial_occupancy import (
    DEFAULT_CDF_THRESHOLDS_MM,
    DEFAULT_MIN_EXPECTED_COUNT,
    DEFAULT_PERIMETER_BAND_MM,
    DEFAULT_RADIAL_BIN_WIDTH_MM,
    _ring_areas_mm2,
    _selection_index,
    _wall_mask,
)
from fisheye.shared.arena_geometry import ArenaGeometry


SCHEMA_ID = "palette.provider_chaser_position_suite"
SCHEMA_VERSION = 1
METHOD_ID = "sealed_provider_position_chaser_spatial_suite_v1"
QUADRANT_POLICY_ID = "selected_circle_center_native_xy_y_down_v1"
RADIAL_POLICY_ID = "moving_chaser_circle_clipped_geometric_null_v1"
NEAR_FIELD_POLICY_ID = "valid_tracked_hysteresis_5mm_6mm_v1"
ROLE_CONTRAST_POLICY_ID = "explicit_treatment_minus_baseline_same_epoch_v1"
QUADRANT_LABELS = ("top_left", "top_right", "bottom_left", "bottom_right")


class ProviderChaserPositionSuiteError(ValueError):
    """Raised when exact position-suite inputs are incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise ProviderChaserPositionSuiteError(message)


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        _fail(f"{field} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderChaserPositionSuiteError(
            f"{field} must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0:
        _fail(f"{field} must be finite and positive.")
    return result


def _exact_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one exact nonempty string.")
    return value


@dataclass(frozen=True)
class PositionSuiteEpoch:
    """One caller-role-bound exact half-open acquisition-frame interval."""

    analysis_role: str
    window_id: int
    source_label: str
    start_frame: int
    end_frame: int
    source_interval_sha256: str

    def __post_init__(self) -> None:
        _exact_text(self.analysis_role, field="epoch.analysis_role")
        _exact_text(self.source_label, field="epoch.source_label")
        _exact_text(self.source_interval_sha256, field="epoch.source_interval_sha256")
        if type(self.window_id) is not int or self.window_id < 0:
            _fail("epoch.window_id must be one non-negative integer.")
        if (
            type(self.start_frame) is not int
            or type(self.end_frame) is not int
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
        ):
            _fail("Epoch bounds must be a nonempty half-open frame interval.")


@dataclass(frozen=True)
class CircularArena:
    """Reviewed circular boundary in native continuous source-camera pixels."""

    center_x_px: float
    center_y_px: float
    radius_px: float
    boundary_role: str
    observed_feature: str

    def __post_init__(self) -> None:
        for name in ("center_x_px", "center_y_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                _fail(f"arena.{name} must be finite.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "radius_px",
            _finite_positive(self.radius_px, field="arena.radius_px"),
        )
        _exact_text(self.boundary_role, field="arena.boundary_role")
        _exact_text(self.observed_feature, field="arena.observed_feature")


@dataclass(frozen=True)
class PositionSuiteConfig:
    radial_bin_width_mm: float = DEFAULT_RADIAL_BIN_WIDTH_MM
    cdf_thresholds_mm: tuple[float, ...] = DEFAULT_CDF_THRESHOLDS_MM
    near_zone_radius_mm: float = DEFAULT_R_ZONE_MM
    near_entry_radius_mm: float = DEFAULT_R_IN_MM
    near_exit_radius_mm: float = DEFAULT_R_OUT_MM
    perimeter_band_mm: float = DEFAULT_PERIMETER_BAND_MM
    min_expected_count: float = DEFAULT_MIN_EXPECTED_COUNT
    treatment_role: str = "aggressive"
    baseline_role: str = "inert"

    def __post_init__(self) -> None:
        for field in (
            "radial_bin_width_mm",
            "near_zone_radius_mm",
            "near_entry_radius_mm",
            "near_exit_radius_mm",
            "min_expected_count",
        ):
            object.__setattr__(
                self,
                field,
                _finite_positive(getattr(self, field), field=f"config.{field}"),
            )
        perimeter = float(self.perimeter_band_mm)
        if not math.isfinite(perimeter) or perimeter < 0:
            _fail("config.perimeter_band_mm must be finite and non-negative.")
        object.__setattr__(self, "perimeter_band_mm", perimeter)
        if self.near_exit_radius_mm <= self.near_entry_radius_mm:
            _fail("Near-field exit radius must exceed the entry radius.")
        thresholds = np.asarray(self.cdf_thresholds_mm, dtype=np.float64)
        if (
            thresholds.ndim != 1
            or thresholds.size == 0
            or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)
            or np.any(np.diff(thresholds) <= 0)
        ):
            _fail("CDF thresholds must be finite, non-negative, and increasing.")
        object.__setattr__(
            self, "cdf_thresholds_mm", tuple(float(v) for v in thresholds)
        )
        treatment = _exact_text(self.treatment_role, field="config.treatment_role")
        baseline = _exact_text(self.baseline_role, field="config.baseline_role")
        if treatment == baseline:
            _fail("Treatment and baseline roles must differ.")


def _shape(
    value: Any, expected: tuple[int, ...], *, field: str, dtype: Any
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != expected:
        _fail(f"{field} shape {result.shape!r} differs from {expected!r}.")
    return result


def _quadrant(xy: np.ndarray, arena: CircularArena) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    right = values[..., 0] >= arena.center_x_px
    bottom = values[..., 1] >= arena.center_y_px
    return right.astype(np.int8) + 2 * bottom.astype(np.int8)


def _optional_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _optional_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def _optional_fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _difference(treatment: object, baseline: object) -> float | None:
    if treatment is None or baseline is None:
        return None
    left = float(treatment)
    right = float(baseline)
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return left - right


def _arena_geometry(arena: CircularArena) -> ArenaGeometry:
    diameter = 2.0 * arena.radius_px
    return ArenaGeometry(
        status="exact_reviewed_selection",
        source="provider_chaser_position_suite",
        width_px=diameter,
        height_px=diameter,
        shape="circle",
        center_x_px=arena.center_x_px,
        center_y_px=arena.center_y_px,
        radius_px=arena.radius_px,
    )


def _radial_edges(maximum_distance_mm: float, width_mm: float) -> np.ndarray:
    count = max(1, int(math.ceil(maximum_distance_mm / width_mm)))
    return np.arange(count + 1, dtype=np.float64) * width_mm


def compute_provider_chaser_position_suite(
    *,
    frame_ids: Any,
    fish_xy_px: Any,
    fish_valid: Any,
    chaser_xy_px: Any,
    chaser_valid: Any,
    distance_px: Any,
    distance_px_valid: Any,
    distance_mm: Any,
    distance_mm_valid: Any,
    selection_member: Any,
    chaser_occurrence_member: Any,
    chaser_role_codes: Any,
    chaser_role_valid: Any,
    chaser_identity_codes: Any,
    role_registry: Mapping[str, str],
    chaser_registry: Mapping[str, str],
    epochs: Sequence[PositionSuiteEpoch],
    arena: CircularArena,
    mm_per_pixel: float,
    fps: float,
    config: PositionSuiteConfig = PositionSuiteConfig(),
) -> dict[str, Any]:
    """Compute exact epoch/chaser spatial products without performing I/O."""

    frame_ids_array = np.asarray(frame_ids, dtype=np.int64)
    if frame_ids_array.ndim != 1 or frame_ids_array.size == 0:
        _fail("frame_ids must be one nonempty one-dimensional frame axis.")
    if np.any(np.diff(frame_ids_array) <= 0):
        _fail("frame_ids must be strictly increasing and unique.")
    n_frames = int(frame_ids_array.size)
    chaser_xy_array = np.asarray(chaser_xy_px, dtype=np.float64)
    if (
        chaser_xy_array.ndim != 3
        or chaser_xy_array.shape[:1] != (n_frames,)
        or chaser_xy_array.shape[2] != 2
    ):
        _fail("chaser_xy_px must have shape (frame, chaser, xy).")
    n_chasers = int(chaser_xy_array.shape[1])
    if n_chasers <= 0:
        _fail("At least one chaser column is required.")

    fish_xy = _shape(fish_xy_px, (n_frames, 2), field="fish_xy_px", dtype=np.float64)
    fish_ok = _shape(fish_valid, (n_frames,), field="fish_valid", dtype=bool)
    chaser_ok = _shape(
        chaser_valid, (n_frames, n_chasers), field="chaser_valid", dtype=bool
    )
    distance_px_array = _shape(
        distance_px, (n_frames, n_chasers), field="distance_px", dtype=np.float64
    )
    distance_px_ok = _shape(
        distance_px_valid, (n_frames, n_chasers), field="distance_px_valid", dtype=bool
    )
    distance_mm_array = _shape(
        distance_mm, (n_frames, n_chasers), field="distance_mm", dtype=np.float64
    )
    distance_mm_ok = _shape(
        distance_mm_valid, (n_frames, n_chasers), field="distance_mm_valid", dtype=bool
    )
    selected = _shape(
        selection_member, (n_frames,), field="selection_member", dtype=bool
    )
    occurrence = _shape(
        chaser_occurrence_member,
        (n_frames, n_chasers),
        field="chaser_occurrence_member",
        dtype=bool,
    )
    role_codes = _shape(
        chaser_role_codes,
        (n_frames, n_chasers),
        field="chaser_role_codes",
        dtype=np.uint8,
    )
    role_ok = _shape(
        chaser_role_valid,
        (n_frames, n_chasers),
        field="chaser_role_valid",
        dtype=bool,
    )
    identity_codes = _shape(
        chaser_identity_codes,
        (n_frames, n_chasers),
        field="chaser_identity_codes",
        dtype=np.uint16,
    )
    scale = _finite_positive(mm_per_pixel, field="mm_per_pixel")
    fps_hz = _finite_positive(fps, field="fps")
    if not isinstance(role_registry, Mapping) or not isinstance(
        chaser_registry, Mapping
    ):
        _fail("Chaser role and identity registries must be exact mappings.")
    if not epochs:
        _fail("At least one exact caller-role-bound epoch is required.")
    if len({epoch.analysis_role for epoch in epochs}) != len(epochs):
        _fail("Epoch analysis roles must be unique.")
    if len({epoch.window_id for epoch in epochs}) != len(epochs):
        _fail("Epoch window IDs must be unique.")
    covered = np.zeros(n_frames, dtype=bool)
    for epoch in epochs:
        mask = (frame_ids_array >= epoch.start_frame) & (
            frame_ids_array < epoch.end_frame
        )
        if np.any(covered & mask):
            _fail(
                "Exact epoch intervals overlap on the provider acquisition-frame axis."
            )
        covered |= mask

    finite_fish = np.isfinite(fish_xy).all(axis=1)
    finite_chaser = np.isfinite(chaser_xy_array).all(axis=2)
    if np.any(fish_ok & ~finite_fish) or np.any(chaser_ok & ~finite_chaser):
        _fail("A declared-valid position contains non-finite coordinates.")
    derived_px = np.linalg.norm(chaser_xy_array - fish_xy[:, None, :], axis=2)
    px_comparable = distance_px_ok & fish_ok[:, None] & chaser_ok
    if np.any(px_comparable) and not np.allclose(
        distance_px_array[px_comparable],
        derived_px[px_comparable],
        rtol=1e-5,
        atol=1e-4,
    ):
        _fail("Persisted pixel distances disagree with the bound provider coordinates.")
    mm_comparable = distance_mm_ok & distance_px_ok
    if np.any(mm_comparable) and not np.allclose(
        distance_mm_array[mm_comparable],
        distance_px_array[mm_comparable] * scale,
        rtol=1e-5,
        atol=1e-4,
    ):
        _fail("Persisted millimetre distances disagree with the exact physical scale.")
    if np.any(distance_mm_ok & ~(distance_px_ok & fish_ok[:, None] & chaser_ok)):
        _fail(
            "A valid millimetre distance lacks valid coordinate/pixel-distance evidence."
        )

    stable_roles: list[tuple[int, str]] = []
    stable_chasers: list[tuple[int, str]] = []
    for column in range(n_chasers):
        role_values = np.unique(role_codes[:, column][role_ok[:, column]])
        identity_values = np.unique(identity_codes[:, column])
        if role_values.size != 1 or identity_values.size != 1:
            _fail("Chaser role and identity must each be stable per chaser column.")
        role_code = int(role_values[0])
        identity_code = int(identity_values[0])
        role_label = role_registry.get(str(role_code))
        identity_label = chaser_registry.get(str(identity_code))
        if type(role_label) is not str or type(identity_label) is not str:
            _fail("Stable chaser role or identity code is absent from its registry.")
        stable_roles.append((role_code, role_label))
        stable_chasers.append((identity_code, identity_label))

    treatment_columns = [
        i for i, value in enumerate(stable_roles) if value[1] == config.treatment_role
    ]
    baseline_columns = [
        i for i, value in enumerate(stable_roles) if value[1] == config.baseline_role
    ]
    if len(treatment_columns) != 1 or len(baseline_columns) != 1:
        _fail(
            "Role contrast requires exactly one chaser column for each explicitly "
            "declared treatment and baseline role."
        )

    geometry = _arena_geometry(arena)
    arena_radius_mm = arena.radius_px * scale
    if config.perimeter_band_mm >= arena_radius_mm:
        _fail(
            "Perimeter exclusion band must be smaller than the selected arena radius."
        )
    bound_chaser_rows = selected[:, None] & occurrence & role_ok & chaser_ok
    if not np.any(bound_chaser_rows):
        _fail("No bound valid chaser positions are available for radial geometry.")
    chaser_center_radius_mm = (
        np.linalg.norm(
            chaser_xy_array
            - np.asarray([arena.center_x_px, arena.center_y_px], dtype=np.float64),
            axis=2,
        )
        * scale
    )
    maximum_reference_radius_mm = float(
        np.max(chaser_center_radius_mm[bound_chaser_rows])
    )
    maximum_radial_distance_mm = arena_radius_mm + maximum_reference_radius_mm
    radial_edges = _radial_edges(
        maximum_radial_distance_mm,
        config.radial_bin_width_mm,
    )
    cdf_thresholds = np.asarray(config.cdf_thresholds_mm, dtype=np.float64)
    fish_quadrant = _quadrant(fish_xy, arena)
    chaser_quadrant = _quadrant(chaser_xy_array, arena)
    fish_radius_mm = (
        np.linalg.norm(
            fish_xy - np.asarray([arena.center_x_px, arena.center_y_px]), axis=1
        )
        * scale
    )
    fish_wall_distance_mm = arena_radius_mm - fish_radius_mm
    wall = _wall_mask(
        fish_xy,
        geometry=geometry,
        perimeter_band_mm=config.perimeter_band_mm,
        pixels_per_mm=1.0 / scale,
    )

    metrics: list[dict[str, Any]] = []
    cdf_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    quadrant_rows: list[dict[str, Any]] = []
    by_epoch_role: dict[tuple[str, str], dict[str, Any]] = {}
    for epoch in epochs:
        epoch_mask = (frame_ids_array >= epoch.start_frame) & (
            frame_ids_array < epoch.end_frame
        )
        epoch_frame_count = int(np.count_nonzero(epoch_mask))
        source_interval_frame_count = epoch.end_frame - epoch.start_frame
        for column in range(n_chasers):
            role_code, role_label = stable_roles[column]
            identity_code, identity_label = stable_chasers[column]
            candidate = (
                epoch_mask & selected & occurrence[:, column] & role_ok[:, column]
            )
            valid = candidate & distance_mm_ok[:, column]
            valid_count = int(np.count_nonzero(valid))
            candidate_count = int(np.count_nonzero(candidate))
            distances = distance_mm_array[:, column][valid]
            same_quadrant = fish_quadrant == chaser_quadrant[:, column]
            same_count = int(np.count_nonzero(valid & same_quadrant))
            near_count = int(
                np.count_nonzero(
                    valid & (distance_mm_array[:, column] <= config.near_zone_radius_mm)
                )
            )
            visit = compute_hysteresis_visits(
                distance_mm_array[epoch_mask, column],
                valid[epoch_mask],
                fps=fps_hz,
                r_in_mm=config.near_entry_radius_mm,
                r_out_mm=config.near_exit_radius_mm,
                policy_version=VALID_TRACKED_VISIT_POLICY,
            )
            values = distance_mm_array[:, column][valid]
            chaser_values = chaser_xy_array[:, column, :][valid]
            if valid_count:
                ring_area = _ring_areas_mm2(
                    bin_edges_mm=radial_edges,
                    chaser_xy_px=chaser_values,
                    geometry=geometry,
                    pixels_per_mm=1.0 / scale,
                    exclude_perimeter_band_mm=0.0,
                    cache_step_mm=1.0,
                )
                counts, _ = np.histogram(values, bins=radial_edges)
                observed = counts.astype(np.float64) / valid_count
                expected = (
                    ring_area / ring_area.sum()
                    if ring_area.sum() > 0
                    else np.full(ring_area.shape, np.nan)
                )
                selection_index, _ = _selection_index(
                    observed,
                    expected,
                    sample_count=valid_count,
                    min_expected_count=config.min_expected_count,
                )
                valid_wall_excluded = valid & ~wall
                wall_values = distance_mm_array[:, column][valid_wall_excluded]
                wall_chaser_values = chaser_xy_array[:, column, :][valid_wall_excluded]
                wall_count = int(wall_values.size)
                ring_area_wall = _ring_areas_mm2(
                    bin_edges_mm=radial_edges,
                    chaser_xy_px=wall_chaser_values,
                    geometry=geometry,
                    pixels_per_mm=1.0 / scale,
                    exclude_perimeter_band_mm=config.perimeter_band_mm,
                    cache_step_mm=1.0,
                )
                counts_wall, _ = np.histogram(wall_values, bins=radial_edges)
                observed_wall = (
                    counts_wall.astype(np.float64) / wall_count
                    if wall_count
                    else np.full(counts_wall.shape, np.nan)
                )
                expected_wall = (
                    ring_area_wall / ring_area_wall.sum()
                    if ring_area_wall.sum() > 0
                    else np.full(ring_area_wall.shape, np.nan)
                )
                selection_wall, _ = _selection_index(
                    observed_wall,
                    expected_wall,
                    sample_count=wall_count,
                    min_expected_count=config.min_expected_count,
                )
            else:
                counts = np.zeros(radial_edges.size - 1, dtype=np.int64)
                ring_area = np.full(counts.shape, np.nan)
                observed = expected = selection_index = np.full(counts.shape, np.nan)
                counts_wall = np.zeros(counts.shape, dtype=np.int64)
                ring_area_wall = observed_wall = expected_wall = selection_wall = (
                    np.full(counts.shape, np.nan)
                )
                wall_count = 0

            near_ring_area = (
                _ring_areas_mm2(
                    bin_edges_mm=np.asarray([0.0, config.near_zone_radius_mm]),
                    chaser_xy_px=chaser_values,
                    geometry=geometry,
                    pixels_per_mm=1.0 / scale,
                    exclude_perimeter_band_mm=0.0,
                    cache_step_mm=1.0,
                )
                if valid_count
                else np.asarray([np.nan])
            )
            near_expected = (
                float(near_ring_area[0] / ring_area.sum())
                if valid_count and ring_area.sum() > 0
                else None
            )
            near_observed = _optional_fraction(near_count, valid_count)
            record = {
                "analysis_role": epoch.analysis_role,
                "epoch_window_id": epoch.window_id,
                "epoch_source_label": epoch.source_label,
                "epoch_start_frame": epoch.start_frame,
                "epoch_end_frame_exclusive": epoch.end_frame,
                "source_interval_sha256": epoch.source_interval_sha256,
                "source_interval_frame_count": source_interval_frame_count,
                "epoch_provider_frame_count": epoch_frame_count,
                "epoch_provider_frame_coverage_fraction": _optional_fraction(
                    epoch_frame_count, source_interval_frame_count
                ),
                "chaser_column": column,
                "chaser_identity_code": identity_code,
                "chaser_identity": identity_label,
                "behavior_role_code": role_code,
                "behavior_role": role_label,
                "candidate_frame_count": candidate_count,
                "valid_distance_frame_count": valid_count,
                "valid_distance_fraction": _optional_fraction(
                    valid_count, candidate_count
                ),
                "distance_mean_mm": _optional_mean(distances),
                "distance_p05_mm": _optional_percentile(distances, 5),
                "distance_p25_mm": _optional_percentile(distances, 25),
                "distance_p50_mm": _optional_percentile(distances, 50),
                "distance_p75_mm": _optional_percentile(distances, 75),
                "distance_p95_mm": _optional_percentile(distances, 95),
                "same_quadrant_valid_frame_count": same_count,
                "same_quadrant_fraction_valid": _optional_fraction(
                    same_count, valid_count
                ),
                "same_quadrant_fraction_candidate": _optional_fraction(
                    same_count, candidate_count
                ),
                "near_zone_frame_count": near_count,
                "near_zone_fraction_valid": near_observed,
                "near_zone_fraction_candidate": _optional_fraction(
                    near_count, candidate_count
                ),
                "near_zone_dwell_s": near_count / fps_hz,
                "near_zone_expected_fraction_geometric": near_expected,
                "near_zone_enrichment_geometric": (
                    float(near_observed / near_expected)
                    if near_observed is not None
                    and near_expected is not None
                    and near_expected > 0
                    else None
                ),
                "near_zone_entry_count": visit.entry_count,
                "near_zone_entry_rate_per_min_valid_time": visit.entry_rate_per_min,
                "near_zone_valid_tracked_duration_s": visit.valid_tracked_duration_s,
                "near_zone_complete_visit_median_dwell_s": visit.complete_visit_median_dwell_s,
                "near_zone_complete_visit_total_dwell_s": visit.complete_visit_total_dwell_s,
                "near_zone_invalid_gap_count": visit.invalid_gap_count,
                "near_zone_censor_event_count": visit.censor_event_count,
                "near_zone_boundary_censor_event_count": visit.boundary_censor_event_count,
                "near_zone_invalid_gap_censor_event_count": visit.invalid_gap_censor_event_count,
                "wall_excluded_valid_frame_count": wall_count,
                "fish_arena_radius_mean_mm": _optional_mean(fish_radius_mm[valid]),
                "fish_arena_radius_p50_mm": _optional_percentile(
                    fish_radius_mm[valid], 50
                ),
                "fish_wall_distance_mean_mm": _optional_mean(
                    fish_wall_distance_mm[valid]
                ),
                "fish_wall_distance_p50_mm": _optional_percentile(
                    fish_wall_distance_mm[valid], 50
                ),
            }
            metrics.append(record)
            by_epoch_role[(epoch.analysis_role, role_label)] = record

            for threshold in cdf_thresholds:
                cdf_rows.append(
                    {
                        "analysis_role": epoch.analysis_role,
                        "epoch_window_id": epoch.window_id,
                        "behavior_role": role_label,
                        "chaser_identity": identity_label,
                        "threshold_mm": float(threshold),
                        "fraction_at_or_below": (
                            float(np.mean(distances <= threshold))
                            if distances.size
                            else None
                        ),
                    }
                )
            for index in range(radial_edges.size - 1):
                radial_rows.append(
                    {
                        "analysis_role": epoch.analysis_role,
                        "epoch_window_id": epoch.window_id,
                        "behavior_role": role_label,
                        "chaser_identity": identity_label,
                        "bin_start_mm": float(radial_edges[index]),
                        "bin_end_mm": float(radial_edges[index + 1]),
                        "observed_count": int(counts[index]),
                        "observed_fraction": float(observed[index])
                        if math.isfinite(float(observed[index]))
                        else None,
                        "expected_available_area_mm2_frames": float(ring_area[index])
                        if math.isfinite(float(ring_area[index]))
                        else None,
                        "expected_fraction_geometric": float(expected[index])
                        if math.isfinite(float(expected[index]))
                        else None,
                        "selection_index_geometric": float(selection_index[index])
                        if math.isfinite(float(selection_index[index]))
                        else None,
                        "wall_excluded_observed_count": int(counts_wall[index]),
                        "wall_excluded_observed_fraction": float(observed_wall[index])
                        if math.isfinite(float(observed_wall[index]))
                        else None,
                        "wall_excluded_expected_available_area_mm2_frames": float(
                            ring_area_wall[index]
                        )
                        if math.isfinite(float(ring_area_wall[index]))
                        else None,
                        "wall_excluded_expected_fraction_geometric": float(
                            expected_wall[index]
                        )
                        if math.isfinite(float(expected_wall[index]))
                        else None,
                        "wall_excluded_selection_index_geometric": float(
                            selection_wall[index]
                        )
                        if math.isfinite(float(selection_wall[index]))
                        else None,
                    }
                )
            for fish_code, fish_label in enumerate(QUADRANT_LABELS):
                for chaser_code, chaser_label in enumerate(QUADRANT_LABELS):
                    count = int(
                        np.count_nonzero(
                            valid
                            & (fish_quadrant == fish_code)
                            & (chaser_quadrant[:, column] == chaser_code)
                        )
                    )
                    quadrant_rows.append(
                        {
                            "analysis_role": epoch.analysis_role,
                            "epoch_window_id": epoch.window_id,
                            "behavior_role": role_label,
                            "chaser_identity": identity_label,
                            "fish_quadrant": fish_label,
                            "chaser_quadrant": chaser_label,
                            "valid_joint_frame_count": count,
                            "valid_joint_fraction": _optional_fraction(
                                count, valid_count
                            ),
                        }
                    )

    scalar_contrast_fields = (
        "valid_distance_fraction",
        "distance_mean_mm",
        "distance_p50_mm",
        "same_quadrant_fraction_valid",
        "near_zone_fraction_valid",
        "near_zone_dwell_s",
        "near_zone_expected_fraction_geometric",
        "near_zone_enrichment_geometric",
        "near_zone_entry_rate_per_min_valid_time",
        "fish_arena_radius_mean_mm",
        "fish_wall_distance_mean_mm",
    )
    contrasts: list[dict[str, Any]] = []
    radial_contrasts: list[dict[str, Any]] = []
    for epoch in epochs:
        treatment = by_epoch_role[(epoch.analysis_role, config.treatment_role)]
        baseline = by_epoch_role[(epoch.analysis_role, config.baseline_role)]
        for metric in scalar_contrast_fields:
            contrasts.append(
                {
                    "analysis_role": epoch.analysis_role,
                    "epoch_window_id": epoch.window_id,
                    "metric": metric,
                    "treatment_role": config.treatment_role,
                    "baseline_role": config.baseline_role,
                    "treatment_value": treatment[metric],
                    "baseline_value": baseline[metric],
                    "treatment_minus_baseline": _difference(
                        treatment[metric], baseline[metric]
                    ),
                }
            )
        treatment_radial = [
            row
            for row in radial_rows
            if row["analysis_role"] == epoch.analysis_role
            and row["behavior_role"] == config.treatment_role
        ]
        baseline_radial = [
            row
            for row in radial_rows
            if row["analysis_role"] == epoch.analysis_role
            and row["behavior_role"] == config.baseline_role
        ]
        if len(treatment_radial) != len(baseline_radial):
            _fail("Role radial contrasts have inconsistent bin coverage.")
        for left, right in zip(treatment_radial, baseline_radial, strict=True):
            if (left["bin_start_mm"], left["bin_end_mm"]) != (
                right["bin_start_mm"],
                right["bin_end_mm"],
            ):
                _fail("Role radial contrasts have inconsistent bin identities.")
            radial_contrasts.append(
                {
                    "analysis_role": epoch.analysis_role,
                    "epoch_window_id": epoch.window_id,
                    "bin_start_mm": left["bin_start_mm"],
                    "bin_end_mm": left["bin_end_mm"],
                    "treatment_role": config.treatment_role,
                    "baseline_role": config.baseline_role,
                    "observed_fraction_treatment_minus_baseline": _difference(
                        left["observed_fraction"], right["observed_fraction"]
                    ),
                    "selection_index_treatment_minus_baseline": _difference(
                        left["selection_index_geometric"],
                        right["selection_index_geometric"],
                    ),
                }
            )

    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "scientific_scope": "position_only",
        "policies": {
            "epoch_membership": "acquisition_frame_id_in_exact_half_open_interval_v1",
            "quadrant": QUADRANT_POLICY_ID,
            "radial": RADIAL_POLICY_ID,
            "near_field": NEAR_FIELD_POLICY_ID,
            "visit": VALID_TRACKED_VISIT_POLICY,
            "role_contrast": ROLE_CONTRAST_POLICY_ID,
            "distance_boundary": "near_zone_inclusive_le;hysteresis_enter_lt_exit_gt",
            "validity_denominator": "valid_provider_fish_and_chaser_rows_only",
            "candidate_denominator": "exact_epoch_selection_and_chaser_occurrence_rows",
        },
        "config": {
            "radial_bin_width_mm": config.radial_bin_width_mm,
            "radial_bin_edges_mm": radial_edges.tolist(),
            "radial_maximum_distance_policy": (
                "selected_arena_radius_plus_max_bound_chaser_center_distance_v1"
            ),
            "radial_maximum_distance_mm": maximum_radial_distance_mm,
            "cdf_thresholds_mm": list(config.cdf_thresholds_mm),
            "near_zone_radius_mm": config.near_zone_radius_mm,
            "near_entry_radius_mm": config.near_entry_radius_mm,
            "near_exit_radius_mm": config.near_exit_radius_mm,
            "perimeter_band_mm": config.perimeter_band_mm,
            "min_expected_count": config.min_expected_count,
            "treatment_role": config.treatment_role,
            "baseline_role": config.baseline_role,
        },
        "arena": {
            "center_x_px": arena.center_x_px,
            "center_y_px": arena.center_y_px,
            "radius_px": arena.radius_px,
            "radius_mm": arena_radius_mm,
            "boundary_role": arena.boundary_role,
            "observed_feature": arena.observed_feature,
            "coordinate_space": "source_camera_continuous_pixel_xy_top_left_y_down",
        },
        "fps_hz": fps_hz,
        "mm_per_pixel": scale,
        "frame_count": n_frames,
        "chaser_count": n_chasers,
        "epoch_roles": [
            {
                "analysis_role": epoch.analysis_role,
                "window_id": epoch.window_id,
                "source_label": epoch.source_label,
                "start_frame": epoch.start_frame,
                "end_frame_exclusive": epoch.end_frame,
                "source_interval_sha256": epoch.source_interval_sha256,
            }
            for epoch in epochs
        ],
        "per_epoch_chaser_metrics": metrics,
        "distance_cdf": cdf_rows,
        "radial_occupancy": radial_rows,
        "quadrant_joint_occupancy": quadrant_rows,
        "role_contrasts": contrasts,
        "role_radial_contrasts": radial_contrasts,
        "interpretation_caveats": [
            "The moving-chaser geometric null corrects available area only; during closed-loop pursuit it is not a behavioral independence null.",
            "This product contains no heading, body-frame, speed, bout, gaze, trial, or escape inference.",
        ],
    }


__all__ = [
    "CircularArena",
    "METHOD_ID",
    "NEAR_FIELD_POLICY_ID",
    "PositionSuiteConfig",
    "PositionSuiteEpoch",
    "ProviderChaserPositionSuiteError",
    "QUADRANT_LABELS",
    "RADIAL_POLICY_ID",
    "ROLE_CONTRAST_POLICY_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "compute_provider_chaser_position_suite",
]
