"""Declarative metric registry for validated-behavior cohort statistics.

The registry names exact columns already sealed by a validated-behavior
export.  It does not reconstruct scientific metrics from raw arrays.  Each
metric is reduced to one value per recording, condition, and declared group
before any cohort summary or paired contrast is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

CHASER_EPOCH_CONDITIONS = (
    "chaser_pre",
    "chaser_training",
    "chaser_post",
)


@dataclass(frozen=True, slots=True)
class ConditionContrastSpec:
    """One paired condition contrast; the reported difference is ``b - a``."""

    contrast_id: str
    condition_a: str
    condition_b: str

    def __post_init__(self) -> None:
        for name, value in (
            ("contrast_id", self.contrast_id),
            ("condition_a", self.condition_a),
            ("condition_b", self.condition_b),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be one nonempty stripped string")
        if self.condition_a == self.condition_b:
            raise ValueError("A condition contrast requires two distinct conditions")

    def to_dict(self) -> dict[str, str]:
        return {
            "contrast_id": self.contrast_id,
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "difference_direction": "condition_b_minus_condition_a",
        }


CHASER_EPOCH_CONTRASTS = (
    ConditionContrastSpec("training-pre", "chaser_pre", "chaser_training"),
    ConditionContrastSpec("post-pre", "chaser_pre", "chaser_post"),
    ConditionContrastSpec("post-training", "chaser_training", "chaser_post"),
)

CONTRAST_SETS: Mapping[str, tuple[ConditionContrastSpec, ...]] = MappingProxyType(
    {"chaser_epoch_v1": CHASER_EPOCH_CONTRASTS}
)


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorMetricSpec:
    """One exact source column and its recording-level statistical grain."""

    metric_id: str
    metric_family: str
    source_table: str
    value_column: str
    unit: str
    condition_column: str | None
    expected_conditions: tuple[str, ...]
    group_columns: tuple[str, ...]
    contrast_set_id: str | None
    multiplicity_family: str | None
    retain_recording_values: bool
    interpretation: str
    analysis_status: str = "exploratory"
    recording_reducer: str = "unique_exact_row"
    source_identity_columns: tuple[str, ...] = ()
    reducer_order_column: str | None = None

    def __post_init__(self) -> None:
        text_fields = {
            "metric_id": self.metric_id,
            "metric_family": self.metric_family,
            "source_table": self.source_table,
            "value_column": self.value_column,
            "unit": self.unit,
            "interpretation": self.interpretation,
        }
        for name, value in text_fields.items():
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be one nonempty stripped string")
        if self.analysis_status != "exploratory":
            raise ValueError(
                "Validated-behavior v1 statistics are exploratory while "
                "authoritative acquisition-batch identity is unavailable"
            )
        if self.recording_reducer not in {
            "unique_exact_row",
            "terminal_at_max_order_v1",
        }:
            raise ValueError("Unsupported recording reducer")
        if len(set(self.group_columns)) != len(self.group_columns):
            raise ValueError(f"{self.metric_id}: group columns must be unique")
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in self.group_columns
        ):
            raise ValueError(f"{self.metric_id}: invalid group column")
        if (
            len(set(self.source_identity_columns)) != len(self.source_identity_columns)
            or any(
                type(value) is not str
                or not value
                or value != value.strip()
                for value in self.source_identity_columns
            )
        ):
            raise ValueError(f"{self.metric_id}: invalid source identity columns")
        if self.reducer_order_column is not None and (
            type(self.reducer_order_column) is not str
            or not self.reducer_order_column
            or self.reducer_order_column != self.reducer_order_column.strip()
        ):
            raise ValueError(f"{self.metric_id}: invalid reducer order column")
        source_roles = (
            self.group_columns
            + self.source_identity_columns
            + ((self.condition_column,) if self.condition_column else ())
            + ((self.reducer_order_column,) if self.reducer_order_column else ())
            + (self.value_column,)
        )
        if len(set(source_roles)) != len(source_roles):
            raise ValueError(f"{self.metric_id}: source-column roles must be disjoint")
        if self.recording_reducer == "unique_exact_row":
            if self.source_identity_columns or self.reducer_order_column is not None:
                raise ValueError(
                    f"{self.metric_id}: unique-row reducer cannot declare terminal-row fields"
                )
        elif not self.source_identity_columns or self.reducer_order_column is None:
            raise ValueError(
                f"{self.metric_id}: terminal reducer requires identity and order columns"
            )
        if len(set(self.expected_conditions)) != len(self.expected_conditions):
            raise ValueError(f"{self.metric_id}: expected conditions must be unique")
        if self.condition_column is None:
            if self.expected_conditions or self.contrast_set_id is not None:
                raise ValueError(
                    f"{self.metric_id}: unconditional metrics cannot declare contrasts"
                )
        else:
            if (
                type(self.condition_column) is not str
                or not self.condition_column
                or self.condition_column != self.condition_column.strip()
                or not self.expected_conditions
            ):
                raise ValueError(
                    f"{self.metric_id}: conditional metrics need an exact condition axis"
                )
            if self.condition_column in self.group_columns:
                raise ValueError(
                    f"{self.metric_id}: condition cannot also be a group column"
                )
        if self.contrast_set_id is not None:
            if self.contrast_set_id not in CONTRAST_SETS:
                raise ValueError(f"{self.metric_id}: unknown contrast set")
            if self.multiplicity_family is None:
                raise ValueError(
                    f"{self.metric_id}: contrasted metrics need a multiplicity family"
                )
            allowed = set(self.expected_conditions)
            for contrast in CONTRAST_SETS[self.contrast_set_id]:
                if not {contrast.condition_a, contrast.condition_b}.issubset(allowed):
                    raise ValueError(
                        f"{self.metric_id}: contrast conditions are not declared"
                    )
        elif self.multiplicity_family is not None:
            raise ValueError(
                f"{self.metric_id}: descriptive-only metrics cannot claim a test family"
            )

    @property
    def contrast_eligible(self) -> bool:
        return self.contrast_set_id is not None

    @property
    def spec_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_family": self.metric_family,
            "source_table": self.source_table,
            "value_column": self.value_column,
            "unit": self.unit,
            "condition_column": self.condition_column,
            "expected_conditions": list(self.expected_conditions),
            "group_columns": list(self.group_columns),
            "contrast_set_id": self.contrast_set_id,
            "multiplicity_family": self.multiplicity_family,
            "retain_recording_values": self.retain_recording_values,
            "interpretation": self.interpretation,
            "analysis_status": self.analysis_status,
            "recording_reducer": self.recording_reducer,
            "source_identity_columns": list(self.source_identity_columns),
            "reducer_order_column": self.reducer_order_column,
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
        }


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorHistogramAxisSpec:
    """One fixed-width axis in a recording-scoped histogram recipe."""

    axis_id: str
    source_column: str
    unit: str
    bin_width: float
    lower_bound: float
    upper_bound: float | None
    coverage_policy: str

    def __post_init__(self) -> None:
        for name, value in (
            ("axis_id", self.axis_id),
            ("source_column", self.source_column),
            ("unit", self.unit),
            ("coverage_policy", self.coverage_policy),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be one nonempty stripped string")
        width = float(self.bin_width)
        lower = float(self.lower_bound)
        if not (width > 0.0) or not all(
            value == value and abs(value) != float("inf") for value in (width, lower)
        ):
            raise ValueError("Histogram bin width and lower bound must be finite")
        if self.coverage_policy == "fixed_closed_terminal":
            if self.upper_bound is None:
                raise ValueError("A fixed histogram axis requires an upper bound")
            upper = float(self.upper_bound)
            span_bins = (upper - lower) / width
            if (
                not upper > lower
                or not upper == upper
                or abs(upper) == float("inf")
                or abs(span_bins - round(span_bins)) > 1e-9
            ):
                raise ValueError(
                    "A fixed histogram range must contain a whole number of bins"
                )
        elif self.coverage_policy == "zero_anchored_cover_valid_max":
            if lower != 0.0 or self.upper_bound is not None:
                raise ValueError(
                    "A data-covering histogram axis must be zero anchored and open"
                )
        else:
            raise ValueError("Unknown histogram-axis coverage policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis_id": self.axis_id,
            "source_column": self.source_column,
            "unit": self.unit,
            "bin_width": float(self.bin_width),
            "lower_bound": float(self.lower_bound),
            "upper_bound": (
                None if self.upper_bound is None else float(self.upper_bound)
            ),
            "coverage_policy": self.coverage_policy,
            "terminal_bin_policy": "right_closed_only_for_final_bin",
        }


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorHistogramSpec:
    """One exact sample query reduced to normalized bins per recording."""

    metric_id: str
    metric_family: str
    source_table: str
    condition_column: str
    expected_conditions: tuple[str, ...]
    group_columns: tuple[str, ...]
    identity_columns: tuple[str, ...]
    membership_columns: tuple[str, ...]
    validity_columns: tuple[str, ...]
    axes: tuple[ValidatedBehaviorHistogramAxisSpec, ...]
    interpretation: str
    analysis_status: str = "exploratory"
    recording_reducer: str = "fixed_bin_count_and_fraction_v1"
    cohort_reducer: str = "equal_weight_recording_fraction_v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_id", self.metric_id),
            ("metric_family", self.metric_family),
            ("source_table", self.source_table),
            ("condition_column", self.condition_column),
            ("interpretation", self.interpretation),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be one nonempty stripped string")
        if self.analysis_status != "exploratory":
            raise ValueError("Validated-behavior histograms remain exploratory")
        if self.recording_reducer != "fixed_bin_count_and_fraction_v1":
            raise ValueError("Unsupported recording histogram reducer")
        if self.cohort_reducer != "equal_weight_recording_fraction_v1":
            raise ValueError("Unsupported cohort histogram reducer")
        if len(self.axes) not in (1, 2):
            raise ValueError("Validated-behavior histograms require one or two axes")
        if len({axis.axis_id for axis in self.axes}) != len(self.axes):
            raise ValueError("Histogram axis IDs must be unique")
        for label, columns in (
            ("expected conditions", self.expected_conditions),
            ("group columns", self.group_columns),
            ("identity columns", self.identity_columns),
            ("membership columns", self.membership_columns),
            ("validity columns", self.validity_columns),
        ):
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(f"Histogram {label} must be nonempty and unique")
            if any(
                type(value) is not str or not value or value != value.strip()
                for value in columns
            ):
                raise ValueError(f"Histogram {label} contain an invalid value")
        selected = (
            (self.condition_column,)
            + self.group_columns
            + self.identity_columns
            + self.membership_columns
            + self.validity_columns
            + tuple(axis.source_column for axis in self.axes)
        )
        if len(set(selected)) != len(selected):
            raise ValueError(
                f"{self.metric_id}: histogram source-column roles must be disjoint"
            )

    @property
    def spec_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "metric_family": self.metric_family,
            "source_table": self.source_table,
            "condition_column": self.condition_column,
            "expected_conditions": list(self.expected_conditions),
            "group_columns": list(self.group_columns),
            "identity_columns": list(self.identity_columns),
            "membership_columns": list(self.membership_columns),
            "validity_columns": list(self.validity_columns),
            "axes": [axis.to_dict() for axis in self.axes],
            "interpretation": self.interpretation,
            "analysis_status": self.analysis_status,
            "recording_reducer": self.recording_reducer,
            "cohort_reducer": self.cohort_reducer,
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "empty_denominator_policy": "explicit_null_exclusion",
            "interpolation": "prohibited",
        }


def _metric(
    family: str,
    table: str,
    column: str,
    unit: str,
    interpretation: str,
    *,
    condition: str | None = None,
    groups: Sequence[str] = (),
    contrasted: bool = False,
    retain: bool = True,
    metric_id: str | None = None,
    recording_reducer: str = "unique_exact_row",
    source_identity_columns: Sequence[str] = (),
    reducer_order_column: str | None = None,
) -> ValidatedBehaviorMetricSpec:
    return ValidatedBehaviorMetricSpec(
        metric_id=metric_id or f"{family}.{column}",
        metric_family=family,
        source_table=table,
        value_column=column,
        unit=unit,
        condition_column=condition,
        expected_conditions=(CHASER_EPOCH_CONDITIONS if condition else ()),
        group_columns=tuple(groups),
        contrast_set_id="chaser_epoch_v1" if contrasted else None,
        multiplicity_family=f"{family}.epoch_contrasts" if contrasted else None,
        retain_recording_values=bool(retain),
        interpretation=interpretation,
        recording_reducer=recording_reducer,
        source_identity_columns=tuple(source_identity_columns),
        reducer_order_column=reducer_order_column,
    )


def _many(
    family: str,
    table: str,
    values: Iterable[tuple[str, str, str]],
    *,
    condition: str | None = None,
    groups: Sequence[str] = (),
    contrasted: bool = False,
    retain: bool = True,
) -> tuple[ValidatedBehaviorMetricSpec, ...]:
    return tuple(
        _metric(
            family,
            table,
            column,
            unit,
            interpretation,
            condition=condition,
            groups=groups,
            contrasted=contrasted,
            retain=retain,
        )
        for column, unit, interpretation in values
    )


DEFAULT_VALIDATED_BEHAVIOR_METRICS: tuple[ValidatedBehaviorMetricSpec, ...] = (
    *_many(
        "core_behavior",
        "epoch_behavior_summary",
        (
            ("mean_speed_mm_s", "mm/s", "Mean filtered physical speed"),
            ("bout_rate_per_min", "1/min", "Canonical swim-bout rate"),
            ("mean_bout_duration_s", "s", "Mean canonical bout duration"),
            (
                "mean_bout_path_length_mm",
                "mm",
                "Mean canonical bout path length",
            ),
            (
                "mean_abs_bout_net_heading_change_deg",
                "deg",
                "Mean absolute net heading change per canonical bout",
            ),
            (
                "mean_bout_heading_path_deg",
                "deg",
                "Mean accumulated heading path per canonical bout",
            ),
            (
                "median_inter_bout_interval_s",
                "s",
                "Median interval between canonical bouts",
            ),
            (
                "tracking_dropout_fraction",
                "fraction",
                "Fraction of exact epoch rows lacking valid tracking",
            ),
        ),
        condition="analysis_role",
        contrasted=True,
    ),
    _metric(
        "distance_traveled",
        "provider_motion_samples",
        "cumulative_path_distance_mm",
        "mm",
        (
            "Whole-session observed cumulative smoothed path distance at the "
            "exact terminal provider-motion sample"
        ),
        groups=("provider_role",),
        metric_id="distance_traveled.session_total_path_mm",
        recording_reducer="terminal_at_max_order_v1",
        source_identity_columns=(
            "membership_member_sha256",
            "bundle_set_member_sha256",
            "bundle_record_sha256",
            "source_binding_key",
            "source_run_path",
            "source_manifest_sha256",
            "source_verification_digest",
            "position_provider_id",
            "position_provider_digest",
            "track_id",
        ),
        reducer_order_column="track_sample_row_id",
    ),
    _metric(
        "distance_traveled",
        "epoch_behavior_summary",
        "total_path_mm",
        "mm",
        "Observed path distance within the exact semantic epoch",
        condition="analysis_role",
        contrasted=True,
        metric_id="distance_traveled.epoch_total_path_mm",
    ),
    _metric(
        "distance_traveled",
        "epoch_behavior_summary",
        "mean_speed_mm_s",
        "mm/s",
        "Mean observed speed within the exact semantic epoch",
        condition="analysis_role",
        contrasted=True,
        metric_id="distance_traveled.epoch_mean_speed_mm_s",
    ),
    _metric(
        "distance_traveled",
        "epoch_behavior_summary",
        "tracking_dropout_fraction",
        "fraction",
        "Tracking dropout fraction within the exact semantic epoch",
        condition="analysis_role",
        contrasted=True,
        metric_id="distance_traveled.epoch_tracking_dropout_fraction",
    ),
    *_many(
        "near_field",
        "radial_near_field_summary",
        (
            ("distance_mean_mm", "mm", "Mean fish-to-chaser distance"),
            ("distance_p05_mm", "mm", "Fifth percentile fish-to-chaser distance"),
            ("distance_p50_mm", "mm", "Median fish-to-chaser distance"),
            (
                "near_zone_fraction_valid",
                "fraction",
                "Near-zone occupancy over valid distance rows",
            ),
            (
                "near_zone_entry_rate_per_min_valid_time",
                "1/min",
                "Near-zone entry rate over valid tracked time",
            ),
            (
                "near_zone_complete_visit_median_dwell_s",
                "s",
                "Median dwell of complete near-zone visits",
            ),
            (
                "near_zone_enrichment_geometric",
                "dimensionless",
                "Near-zone occupancy relative to the moving-reference geometric null",
            ),
            (
                "valid_distance_fraction",
                "fraction",
                "Fraction of candidate rows with valid fish-to-chaser distance",
            ),
        ),
        condition="epoch_role",
        groups=("provider_role", "behavior_role"),
        contrasted=True,
    ),
    *_many(
        "same_quadrant",
        "same_quadrant_occupancy",
        (
            (
                "same_quadrant_fraction_valid",
                "fraction",
                "Fish and chaser occupancy in the same arena quadrant over valid rows",
            ),
        ),
        condition="epoch_role",
        groups=("provider_role", "behavior_role"),
        contrasted=True,
    ),
    *_many(
        "occupancy_support",
        "spatial_occupancy_support",
        (
            (
                "in_arena_coverage_fraction_candidate",
                "fraction",
                "In-arena finite positions over all candidate epoch rows",
            ),
            (
                "in_arena_fraction_finite_valid",
                "fraction",
                "In-arena positions over finite valid positions",
            ),
        ),
        condition="epoch_role",
        groups=("provider_role",),
        contrasted=True,
    ),
    *_many(
        "bout_response_by_distance",
        "bout_response_distance_bins",
        (
            ("bout_rate_per_min", "1/min", "Bout rate within a persisted distance bin"),
            ("median_duration_s", "s", "Median bout duration within a distance bin"),
            (
                "median_path_length_mm",
                "mm",
                "Median bout path length within a distance bin",
            ),
            (
                "median_peak_speed_mm_s",
                "mm/s",
                "Median bout peak speed within a distance bin",
            ),
        ),
        condition="semantic_role",
        groups=(
            "behavior_role",
            "distance_bin_index",
            "distance_bin_start_mm",
            "distance_bin_end_mm",
        ),
        retain=False,
    ),
    *_many(
        "body_alignment_by_distance",
        "body_alignment_distance_bins",
        (
            (
                "mean_alignment_cos",
                "dimensionless",
                "Mean body-forward alignment with the chaser direction",
            ),
            (
                "mean_abs_bearing_deg",
                "deg",
                "Mean absolute anatomical body-frame bearing",
            ),
            (
                "circular_resultant_length",
                "dimensionless",
                "Concentration of signed body-frame bearings",
            ),
        ),
        condition="epoch_role",
        groups=(
            "behavior_role",
            "distance_bin_index",
            "distance_bin_start_mm",
            "distance_bin_end_mm",
            "distance_bin_center_mm",
        ),
        retain=False,
    ),
    *_many(
        "trial_response",
        "trial_escape_freeze_summaries",
        (
            ("trigger_distance_mm", "mm", "Fish-to-chaser distance at trial trigger"),
            (
                "escape_event_rate_per_min",
                "1/min",
                "Escape-event rate over exact valid trial time",
            ),
            (
                "first_escape_latency_s",
                "s",
                "Latency from trial trigger to the first escape event",
            ),
            (
                "mean_separation_gain_mm",
                "mm",
                "Mean chaser-separation gain across escape events",
            ),
            (
                "recapture_fraction",
                "fraction",
                "Fraction of escape events followed by recapture",
            ),
            (
                "freeze_low_speed_fraction",
                "fraction",
                "Low-speed fraction in the trial freeze window",
            ),
            (
                "freeze_valid_fraction",
                "fraction",
                "Valid-trace fraction in the trial freeze window",
            ),
            (
                "escape_speed_class",
                "fraction",
                "Recording fraction classified as a speed escape at this trial ordinal",
            ),
            (
                "freeze_candidate",
                "fraction",
                "Recording fraction classified as a freeze candidate at this trial ordinal",
            ),
        ),
        groups=("behavior_role", "trial_ordinal"),
    ),
    *_many(
        "spatial_occupancy",
        "spatial_occupancy_bins",
        (
            (
                "occupancy_density_valid_in_arena",
                "fraction",
                "Per-bin occupancy normalized over valid in-arena rows",
            ),
            (
                "occupancy_fraction_candidate_epoch",
                "fraction",
                "Per-bin occupancy normalized over all exact candidate epoch rows",
            ),
        ),
        condition="epoch_role",
        groups=(
            "provider_role",
            "x_bin_index",
            "y_bin_index",
            "x_bin_start_mm",
            "x_bin_end_mm",
            "y_bin_start_mm",
            "y_bin_end_mm",
            "arena_bin_center_member",
        ),
        retain=False,
    ),
    *_many(
        "radial_distribution",
        "radial_near_field_density_bins",
        (
            (
                "observed_fraction",
                "fraction",
                "Observed fish-to-chaser radial-bin fraction",
            ),
            (
                "wall_excluded_observed_fraction",
                "fraction",
                "Observed radial-bin fraction after wall exclusion",
            ),
            (
                "selection_index_geometric",
                "dimensionless",
                "Radial occupancy relative to available geometric area",
            ),
        ),
        condition="epoch_role",
        groups=(
            "provider_role",
            "behavior_role",
            "radial_bin_index",
            "radial_bin_start_mm",
            "radial_bin_end_mm",
        ),
        retain=False,
    ),
    *_many(
        "distance_cdf",
        "radial_near_field_distance_cdf",
        (
            (
                "fraction_at_or_below",
                "fraction",
                "Fish-to-chaser distance CDF at the persisted threshold",
            ),
        ),
        condition="epoch_role",
        groups=(
            "provider_role",
            "behavior_role",
            "threshold_index",
            "threshold_mm",
        ),
        retain=False,
    ),
)


_BODY_BEARING_AXIS_10_DEG = ValidatedBehaviorHistogramAxisSpec(
    axis_id="bearing",
    source_column="body_bearing_deg",
    unit="deg",
    bin_width=10.0,
    lower_bound=-180.0,
    upper_bound=180.0,
    coverage_policy="fixed_closed_terminal",
)

_BODY_BEARING_AXIS_30_DEG = ValidatedBehaviorHistogramAxisSpec(
    axis_id="bearing",
    source_column="body_bearing_deg",
    unit="deg",
    bin_width=30.0,
    lower_bound=-180.0,
    upper_bound=180.0,
    coverage_policy="fixed_closed_terminal",
)

_CHASER_DISTANCE_AXIS_5_MM = ValidatedBehaviorHistogramAxisSpec(
    axis_id="distance",
    source_column="relative_distance_mm",
    unit="mm",
    bin_width=5.0,
    lower_bound=0.0,
    upper_bound=None,
    coverage_policy="zero_anchored_cover_valid_max",
)


DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS: tuple[ValidatedBehaviorHistogramSpec, ...] = (
    ValidatedBehaviorHistogramSpec(
        metric_id="body_bearing_polar.recording_fraction",
        metric_family="body_bearing_polar",
        source_table="body_relative_samples",
        condition_column="epoch_role",
        expected_conditions=CHASER_EPOCH_CONDITIONS,
        group_columns=("behavior_role",),
        identity_columns=("chaser_identity",),
        membership_columns=(
            "selection_member",
            "chaser_occurrence_member",
            "chaser_behavior_role_valid",
        ),
        validity_columns=("body_bearing_valid",),
        axes=(_BODY_BEARING_AXIS_10_DEG,),
        interpretation=(
            "Recording-normalized signed anatomical body-bearing distribution"
        ),
    ),
    ValidatedBehaviorHistogramSpec(
        metric_id="body_bearing_distance.recording_joint_fraction",
        metric_family="body_bearing_distance",
        source_table="body_relative_samples",
        condition_column="epoch_role",
        expected_conditions=CHASER_EPOCH_CONDITIONS,
        group_columns=("behavior_role",),
        identity_columns=("chaser_identity",),
        membership_columns=(
            "selection_member",
            "chaser_occurrence_member",
            "chaser_behavior_role_valid",
        ),
        validity_columns=("body_bearing_valid", "relative_physical_valid"),
        axes=(_BODY_BEARING_AXIS_30_DEG, _CHASER_DISTANCE_AXIS_5_MM),
        interpretation=(
            "Recording-normalized joint signed body-bearing and fish--chaser "
            "distance distribution"
        ),
    ),
)


def validate_metric_specs(
    specs: Sequence[ValidatedBehaviorMetricSpec],
    *,
    allow_empty: bool = False,
) -> tuple[ValidatedBehaviorMetricSpec, ...]:
    resolved = tuple(specs)
    if not resolved and not allow_empty:
        raise ValueError("At least one validated-behavior metric is required")
    ids = tuple(spec.metric_id for spec in resolved)
    if len(set(ids)) != len(ids):
        raise ValueError("Validated-behavior metric IDs must be unique")
    return resolved


def validate_histogram_specs(
    specs: Sequence[ValidatedBehaviorHistogramSpec],
    *,
    allow_empty: bool = False,
) -> tuple[ValidatedBehaviorHistogramSpec, ...]:
    resolved = tuple(specs)
    if not resolved and not allow_empty:
        raise ValueError("At least one validated-behavior histogram is required")
    ids = tuple(spec.metric_id for spec in resolved)
    if len(set(ids)) != len(ids):
        raise ValueError("Validated-behavior histogram IDs must be unique")
    return resolved


def metric_specs_for_families(
    families: Sequence[str] | None = None,
) -> tuple[ValidatedBehaviorMetricSpec, ...]:
    specs = validate_metric_specs(DEFAULT_VALIDATED_BEHAVIOR_METRICS)
    if not families:
        return specs
    requested = tuple(families)
    if len(set(requested)) != len(requested):
        raise ValueError("Metric families must be unique")
    known = set(validated_behavior_family_ids())
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown validated-behavior metric families: {unknown}")
    selected = tuple(spec for spec in specs if spec.metric_family in requested)
    return validate_metric_specs(selected, allow_empty=True)


def histogram_specs_for_families(
    families: Sequence[str] | None = None,
) -> tuple[ValidatedBehaviorHistogramSpec, ...]:
    specs = validate_histogram_specs(DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS)
    if not families:
        return specs
    requested = tuple(families)
    if len(set(requested)) != len(requested):
        raise ValueError("Metric families must be unique")
    return tuple(spec for spec in specs if spec.metric_family in requested)


def validated_behavior_family_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            {spec.metric_family for spec in DEFAULT_VALIDATED_BEHAVIOR_METRICS}
            | {spec.metric_family for spec in DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS}
        )
    )


__all__ = [
    "CHASER_EPOCH_CONDITIONS",
    "CHASER_EPOCH_CONTRASTS",
    "CONTRAST_SETS",
    "ConditionContrastSpec",
    "DEFAULT_VALIDATED_BEHAVIOR_HISTOGRAMS",
    "DEFAULT_VALIDATED_BEHAVIOR_METRICS",
    "ValidatedBehaviorHistogramAxisSpec",
    "ValidatedBehaviorHistogramSpec",
    "ValidatedBehaviorMetricSpec",
    "histogram_specs_for_families",
    "metric_specs_for_families",
    "validate_histogram_specs",
    "validate_metric_specs",
    "validated_behavior_family_ids",
]
