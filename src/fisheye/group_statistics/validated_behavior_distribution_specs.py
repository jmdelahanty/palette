"""Declarative recipes for receipt-bound validated-behavior distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCOPE_ORDER = (
    "whole_session",
    "chaser_pre",
    "chaser_training",
    "chaser_post",
)
SCOPE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "whole_session": "Whole session",
        "chaser_pre": "Pre",
        "chaser_training": "Training",
        "chaser_post": "Post",
    }
)
WEIGHTING_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "event": "Event weighted",
        "frame": "Frame weighted",
        "time": "Time weighted",
    }
)


@dataclass(frozen=True, slots=True)
class DistributionMetricSpec:
    """One source value, validity policy, and immutable histogram-axis recipe."""

    metric_id: str
    metric_family: str
    source_surface: str
    value_column: str
    unit: str
    bin_width: float
    lower_bound: float
    upper_bound: float | None
    coverage_policy: str
    weighting_ids: tuple[str, ...]
    group_columns: tuple[str, ...]
    validity_policy_id: str
    scope_binding_id: str
    interpretation: str

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_id", self.metric_id),
            ("metric_family", self.metric_family),
            ("source_surface", self.source_surface),
            ("value_column", self.value_column),
            ("unit", self.unit),
            ("validity_policy_id", self.validity_policy_id),
            ("scope_binding_id", self.scope_binding_id),
            ("interpretation", self.interpretation),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be one nonempty stripped string")
        if self.source_surface not in {
            "bout_observations",
            "inter_bout_interval_observations",
            "provider_motion_samples",
            "chaser_relative_samples",
        }:
            raise ValueError(f"{self.metric_id}: unsupported source surface")
        width = float(self.bin_width)
        lower = float(self.lower_bound)
        if not math.isfinite(width) or width <= 0 or not math.isfinite(lower):
            raise ValueError(
                f"{self.metric_id}: invalid histogram width or lower bound"
            )
        if self.coverage_policy == "fixed_closed_terminal":
            if self.upper_bound is None:
                raise ValueError(
                    f"{self.metric_id}: fixed range requires an upper bound"
                )
            upper = float(self.upper_bound)
            bins = (upper - lower) / width
            if (
                not math.isfinite(upper)
                or upper <= lower
                or abs(bins - round(bins)) > 1e-9
            ):
                raise ValueError(f"{self.metric_id}: fixed range is not bin aligned")
        elif self.coverage_policy == "zero_anchored_cover_valid_max":
            if lower != 0.0 or self.upper_bound is not None:
                raise ValueError(
                    f"{self.metric_id}: zero-cover range must start at zero"
                )
        elif self.coverage_policy == "symmetric_cover_valid_abs_max":
            if lower != 0.0 or self.upper_bound is not None:
                raise ValueError(
                    f"{self.metric_id}: symmetric range uses an absolute seed"
                )
        elif self.coverage_policy == "log10_cover_valid_positive_range":
            if lower != 0.0 or self.upper_bound is not None:
                raise ValueError(
                    f"{self.metric_id}: logarithmic range is data resolved"
                )
        else:
            raise ValueError(f"{self.metric_id}: unknown histogram coverage policy")
        if (
            not self.weighting_ids
            or len(set(self.weighting_ids)) != len(self.weighting_ids)
            or not set(self.weighting_ids).issubset({"event", "frame", "time"})
        ):
            raise ValueError(f"{self.metric_id}: invalid weighting registry")
        event_source = self.source_surface in {
            "bout_observations",
            "inter_bout_interval_observations",
        }
        if event_source != (self.weighting_ids == ("event",)):
            raise ValueError(
                f"{self.metric_id}: event surfaces must be event weighted only"
            )
        if len(set(self.group_columns)) != len(self.group_columns):
            raise ValueError(f"{self.metric_id}: group columns must be unique")
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in self.group_columns
        ):
            raise ValueError(f"{self.metric_id}: invalid group column")

    @property
    def spec_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_family": self.metric_family,
            "source_surface": self.source_surface,
            "value_column": self.value_column,
            "unit": self.unit,
            "bin_width": float(self.bin_width),
            "lower_bound": float(self.lower_bound),
            "upper_bound": (
                None if self.upper_bound is None else float(self.upper_bound)
            ),
            "coverage_policy": self.coverage_policy,
            "terminal_bin_policy": "right_closed_only_for_final_bin",
            "axis_scale": (
                "log10"
                if self.coverage_policy == "log10_cover_valid_positive_range"
                else "linear"
            ),
            "weighting_ids": list(self.weighting_ids),
            "group_columns": list(self.group_columns),
            "validity_policy_id": self.validity_policy_id,
            "scope_binding_id": self.scope_binding_id,
            "interpretation": self.interpretation,
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
        }


def _metric(
    metric_id: str,
    family: str,
    surface: str,
    column: str,
    unit: str,
    width: float,
    interpretation: str,
    *,
    coverage: str = "zero_anchored_cover_valid_max",
    lower: float = 0.0,
    upper: float | None = None,
    weightings: Sequence[str] = ("event",),
    groups: Sequence[str] = (),
    validity: str,
    scope: str,
) -> DistributionMetricSpec:
    return DistributionMetricSpec(
        metric_id=metric_id,
        metric_family=family,
        source_surface=surface,
        value_column=column,
        unit=unit,
        bin_width=width,
        lower_bound=lower,
        upper_bound=upper,
        coverage_policy=coverage,
        weighting_ids=tuple(weightings),
        group_columns=tuple(groups),
        validity_policy_id=validity,
        scope_binding_id=scope,
        interpretation=interpretation,
    )


DEFAULT_DISTRIBUTION_METRICS: tuple[DistributionMetricSpec, ...] = (
    _metric(
        "bout.duration_s",
        "bout_kinematics",
        "bout_observations",
        "duration_s",
        "s",
        0.02,
        "Canonical swim-bout duration",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.path_length_mm",
        "bout_kinematics",
        "bout_observations",
        "path_length_mm",
        "mm",
        0.25,
        "Canonical swim-bout path length",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.net_displacement_mm",
        "bout_kinematics",
        "bout_observations",
        "net_displacement_mm",
        "mm",
        0.25,
        "Canonical swim-bout net displacement",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.mean_speed_mm_s",
        "bout_kinematics",
        "bout_observations",
        "mean_speed_mm_s",
        "mm/s",
        1.0,
        "Canonical mean physical speed within a swim bout",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.peak_speed_mm_s",
        "bout_kinematics",
        "bout_observations",
        "peak_speed_mm_s",
        "mm/s",
        1.0,
        "Canonical peak physical speed within a swim bout",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.tortuosity",
        "bout_kinematics",
        "bout_observations",
        "tortuosity",
        "dimensionless",
        0.1,
        "Canonical swim-bout path tortuosity on shared log-spaced bins",
        coverage="log10_cover_valid_positive_range",
        validity="finite_nonnegative_canonical_bout_value_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.net_heading_change_deg",
        "bout_heading",
        "bout_observations",
        "net_heading_change_deg",
        "deg",
        10.0,
        "Signed first-to-last smoothed heading change within a canonical bout",
        coverage="fixed_closed_terminal",
        lower=-180.0,
        upper=180.0,
        validity="derived_angular_valid_and_epoch_crosschecked_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.abs_net_heading_change_deg",
        "bout_heading",
        "bout_observations",
        "abs_net_heading_change_deg",
        "deg",
        10.0,
        "Absolute first-to-last smoothed heading change within a canonical bout",
        coverage="fixed_closed_terminal",
        upper=180.0,
        validity="derived_angular_valid_and_epoch_crosschecked_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.heading_path_deg",
        "bout_heading",
        "bout_observations",
        "heading_path_deg",
        "deg",
        10.0,
        "Accumulated absolute smoothed heading change within a canonical bout",
        validity="derived_angular_valid_and_epoch_crosschecked_v1",
        scope="sealed_bout_source_row_epoch_membership_v1",
    ),
    _metric(
        "bout.inter_bout_interval_s",
        "inter_bout_interval",
        "inter_bout_interval_observations",
        "interval_s",
        "s",
        0.1,
        "Gap between consecutive canonical swim bouts",
        validity="producer_interval_canonical_axis_epoch_crosschecked_v1",
        scope="both_interval_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.filtered_speed_mm_s",
        "motion_speed",
        "provider_motion_samples",
        "speed_filtered_mm_s",
        "mm/s",
        1.0,
        "Filtered physical swimming speed",
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="linear_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.smoothed_speed_mm_s",
        "motion_speed",
        "provider_motion_samples",
        "speed_smoothed_mm_s",
        "mm/s",
        1.0,
        "Smoothed physical swimming speed",
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="linear_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.frame_path_distance_smoothed_mm",
        "motion_displacement",
        "provider_motion_samples",
        "frame_path_distance_smoothed_mm",
        "mm",
        0.05,
        "Smoothed path displacement across one valid motion transition",
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="linear_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.delta_heading_smoothed_deg",
        "motion_heading",
        "provider_motion_samples",
        "delta_heading_smoothed_deg",
        "deg",
        5.0,
        "Signed smoothed frame-to-frame heading change",
        coverage="fixed_closed_terminal",
        lower=-180.0,
        upper=180.0,
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="angular_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.angular_velocity_smoothed_deg_s",
        "motion_heading",
        "provider_motion_samples",
        "angular_velocity_smoothed_deg_s",
        "deg/s",
        25.0,
        "Signed smoothed angular velocity",
        coverage="symmetric_cover_valid_abs_max",
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="angular_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "motion.angular_speed_smoothed_deg_s",
        "motion_heading",
        "provider_motion_samples",
        "angular_speed_smoothed_deg_s",
        "deg/s",
        25.0,
        "Absolute smoothed angular speed",
        weightings=("frame", "time"),
        groups=("provider_role",),
        validity="angular_sample_valid_and_transition_valid_positive_time_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
    _metric(
        "chaser.relative_distance_mm",
        "chaser_distance",
        "chaser_relative_samples",
        "relative_distance_mm",
        "mm",
        2.5,
        "Physical fish-to-chaser distance",
        weightings=("frame", "time"),
        groups=("provider_role", "behavior_role"),
        validity="exact_occurrence_relative_physical_and_time_transition_valid_v1",
        scope="frame_row_or_both_transition_endpoints_inside_exact_epoch_v1",
    ),
)


def validate_distribution_metric_specs(
    specs: Sequence[DistributionMetricSpec],
) -> tuple[DistributionMetricSpec, ...]:
    result = tuple(specs)
    if not result:
        raise ValueError("At least one distribution metric is required")
    ids = [spec.metric_id for spec in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Distribution metric IDs must be unique")
    return tuple(sorted(result, key=lambda item: item.metric_id))


def distribution_metric_specs_for_families(
    families: Sequence[str] = (),
) -> tuple[DistributionMetricSpec, ...]:
    requested = tuple(str(value).strip() for value in families if str(value).strip())
    available = {spec.metric_family for spec in DEFAULT_DISTRIBUTION_METRICS}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise KeyError(f"Unknown distribution metric families: {unknown}")
    selected = (
        DEFAULT_DISTRIBUTION_METRICS
        if not requested
        else tuple(
            spec
            for spec in DEFAULT_DISTRIBUTION_METRICS
            if spec.metric_family in requested
        )
    )
    return validate_distribution_metric_specs(selected)


def distribution_metric_family_ids() -> tuple[str, ...]:
    return tuple(sorted({spec.metric_family for spec in DEFAULT_DISTRIBUTION_METRICS}))


__all__ = [
    "DEFAULT_DISTRIBUTION_METRICS",
    "DistributionMetricSpec",
    "SCOPE_LABELS",
    "SCOPE_ORDER",
    "WEIGHTING_LABELS",
    "distribution_metric_family_ids",
    "distribution_metric_specs_for_families",
    "validate_distribution_metric_specs",
]
