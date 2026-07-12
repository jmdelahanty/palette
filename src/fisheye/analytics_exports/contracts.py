"""Version-2 table contracts for Palette analytics exports.

Table names describe analysis semantics and grain. Protocol/cohort identity is
metadata and never part of a table name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXPORT_SCHEMA_ID = "palette.analytics_export"
EXPORT_SCHEMA_VERSION = 2
TABLE_CONTRACT_VERSION = 1

RECORDING_SUMMARY_TABLE = "recording_summary"
STIMULUS_STEPS_TABLE = "stimulus_steps"
STIMULUS_STEP_SUMMARY_TABLE = "stimulus_step_summary"
STIMULUS_RESPONSE_TABLE = "stimulus_response_per_fish_step"
SWIM_BOUT_METRICS_TABLE = "swim_bout_metrics"
BOUT_KINEMATICS_METRICS_TABLE = "bout_kinematics_metrics"

CHASER_SPATIAL_TABLE = "chaser_epoch_spatial_occupancy_zones"
CHASER_DISTANCE_SUMMARY_TABLE = "chaser_epoch_distance_summary"
CHASER_EPOCH_BEHAVIOR_TABLE = "chaser_epoch_behavior_summary"
CHASER_BOUT_EVENTS_TABLE = "chaser_epoch_bout_events"
CHASER_BOUT_HISTOGRAM_TABLE = "chaser_epoch_bout_histogram"
CHASER_IBI_HISTOGRAM_TABLE = "chaser_epoch_inter_bout_interval_histogram"
CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE = "chaser_epoch_center_distance_histogram"
CHASER_SPEED_DISTANCE_TABLE = "chaser_speed_distance_bins"
CHASER_DISTANCE_HISTOGRAM_TABLE = "chaser_epoch_distance_histogram"
CHASER_CRA_SUMMARY_TABLE = "chaser_cra_primary_endpoint_summary"
CHASER_CRA_OBJECT_PHASE_TABLE = "chaser_cra_primary_endpoint_object_phase"
CHASER_CRA_QUADRANT_TABLE = "chaser_cra_quadrant_occupancy"
CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE = "chaser_cra_near_field_summary"
CHASER_CRA_NEAR_FIELD_OBJECT_PHASE_TABLE = "chaser_cra_near_field_object_phase"
CHASER_CRA_NEAR_FIELD_RADIAL_TABLE = "chaser_cra_near_field_radial_density"
CHASER_CRA_NEAR_FIELD_CDF_TABLE = "chaser_cra_near_field_distance_cdf"
CHASER_EGOCENTRIC_SUMMARY_TABLE = "chaser_egocentric_epoch_summary"
CHASER_EGOCENTRIC_HISTOGRAM_TABLE = "chaser_egocentric_distance_bearing_histogram"

STATISTICS_TABLE = "group_statistical_summary"
DESCRIPTIVE_TABLE = "group_descriptive_summary"


@dataclass(frozen=True)
class TableContract:
    """Minimum machine-checkable semantics for one exported table."""

    table_name: str
    grain: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]
    units: Mapping[str, str]
    contract_version: int = TABLE_CONTRACT_VERSION

    @property
    def contract_id(self) -> str:
        return f"palette.analytics.table.{self.table_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "table_name": self.table_name,
            "grain": self.grain,
            "primary_key": list(self.primary_key),
            "required_columns": list(self.required_columns),
            "units": dict(self.units),
        }


def _contract(
    table_name: str,
    grain: str,
    primary_key: Sequence[str],
    required_columns: Sequence[str],
    units: Mapping[str, str] | None = None,
) -> TableContract:
    base = ("export_schema_version", "table_name", *primary_key)
    return TableContract(
        table_name=table_name,
        grain=grain,
        primary_key=tuple(primary_key),
        required_columns=tuple(dict.fromkeys((*base, *required_columns))),
        units=dict(units or {}),
    )


TABLE_CONTRACTS: dict[str, TableContract] = {
    RECORDING_SUMMARY_TABLE: _contract(
        RECORDING_SUMMARY_TABLE,
        "recording",
        ("recording_id",),
        ("recording_id",),
    ),
    STIMULUS_STEPS_TABLE: _contract(
        STIMULUS_STEPS_TABLE,
        "recording_x_stimulus_step",
        ("recording_id", "step_index"),
        ("step_index",),
    ),
    STIMULUS_STEP_SUMMARY_TABLE: _contract(
        STIMULUS_STEP_SUMMARY_TABLE,
        "recording_x_stimulus_step_summary",
        ("recording_id", "step_index"),
        ("step_index",),
    ),
    STIMULUS_RESPONSE_TABLE: _contract(
        STIMULUS_RESPONSE_TABLE,
        "recording_x_fish_x_stimulus_step",
        ("recording_id", "fish_id", "step_index"),
        ("fish_id", "step_index"),
    ),
    SWIM_BOUT_METRICS_TABLE: _contract(
        SWIM_BOUT_METRICS_TABLE,
        "recording_x_swim_bout",
        ("recording_id", "bout_id"),
        ("bout_id",),
    ),
    BOUT_KINEMATICS_METRICS_TABLE: _contract(
        BOUT_KINEMATICS_METRICS_TABLE,
        "recording_x_swim_bout_kinematics",
        ("recording_id", "bout_id"),
        ("bout_id",),
    ),
    CHASER_SPATIAL_TABLE: _contract(
        CHASER_SPATIAL_TABLE,
        "recording_x_chaser_epoch_x_spatial_zone",
        ("recording_id", "window_id", "zone_id"),
        ("window_id", "window_label", "zone_id", "frame_count"),
        {"time_s": "s"},
    ),
    CHASER_DISTANCE_SUMMARY_TABLE: _contract(
        CHASER_DISTANCE_SUMMARY_TABLE,
        "recording_x_chaser_epoch_x_chaser",
        ("recording_id", "window_id", "chaser_index"),
        (
            "window_id",
            "window_label",
            "chaser_index",
            "behavior_class",
            "mean_distance_mm",
            "p50_distance_mm",
        ),
        {
            "mean_distance_mm": "mm",
            "min_distance_mm": "mm",
            "p05_distance_mm": "mm",
            "p50_distance_mm": "mm",
            "p95_distance_mm": "mm",
        },
    ),
    CHASER_EPOCH_BEHAVIOR_TABLE: _contract(
        CHASER_EPOCH_BEHAVIOR_TABLE,
        "recording_x_chaser_epoch",
        ("recording_id", "window_id"),
        (
            "window_id",
            "window_label",
            "mean_speed_mm_s",
            "total_path_mm",
            "bout_count",
            "bout_rate_per_min",
        ),
        {
            "mean_speed_mm_s": "mm/s",
            "total_path_mm": "mm",
            "mean_bout_duration_s": "s",
            "mean_bout_path_length_mm": "mm",
            "mean_inter_bout_interval_s": "s",
        },
    ),
    CHASER_BOUT_EVENTS_TABLE: _contract(
        CHASER_BOUT_EVENTS_TABLE,
        "recording_x_chaser_epoch_x_swim_bout",
        ("recording_id", "window_id", "bout_source_row"),
        (
            "window_id",
            "window_label",
            "bout_source_row",
            "bout_id",
            "bout_start_frame",
            "bout_end_frame",
            "bout_duration_s",
        ),
        {
            "bout_duration_s": "s",
            "bout_path_length_mm": "mm",
            "bout_net_heading_change_deg": "deg",
            "bout_heading_path_deg": "deg",
        },
    ),
    CHASER_BOUT_HISTOGRAM_TABLE: _contract(
        CHASER_BOUT_HISTOGRAM_TABLE,
        "recording_x_chaser_epoch_x_bout_metric_x_bin",
        ("recording_id", "window_id", "metric_name", "bin_index"),
        ("window_id", "metric_name", "bin_index", "hist_count", "units"),
    ),
    CHASER_IBI_HISTOGRAM_TABLE: _contract(
        CHASER_IBI_HISTOGRAM_TABLE,
        "recording_x_chaser_epoch_x_inter_bout_interval_bin",
        ("recording_id", "window_id", "metric_name", "bin_index"),
        ("window_id", "metric_name", "bin_index", "hist_count", "units"),
        {"bin_center": "s", "bin_left": "s", "bin_right": "s"},
    ),
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE: _contract(
        CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
        "recording_x_chaser_epoch_x_center_distance_bin",
        ("recording_id", "window_id", "bin_index"),
        ("window_id", "bin_index", "hist_count", "bin_center_mm"),
        {"bin_center_mm": "mm", "bin_left_mm": "mm", "bin_right_mm": "mm"},
    ),
    CHASER_SPEED_DISTANCE_TABLE: _contract(
        CHASER_SPEED_DISTANCE_TABLE,
        "recording_x_chaser_epoch_x_chaser_x_distance_bin",
        ("recording_id", "window_id", "chaser_index", "distance_bin_index"),
        (
            "window_id",
            "chaser_index",
            "distance_bin_index",
            "mean_speed_mm_s",
            "speed_sample_count",
        ),
        {"distance_bin_center_mm": "mm", "mean_speed_mm_s": "mm/s"},
    ),
    CHASER_DISTANCE_HISTOGRAM_TABLE: _contract(
        CHASER_DISTANCE_HISTOGRAM_TABLE,
        "recording_x_chaser_epoch_x_chaser_x_distance_bin",
        ("recording_id", "window_id", "chaser_index", "distance_bin_index"),
        ("window_id", "chaser_index", "distance_bin_index", "hist_count"),
        {"bin_center_mm": "mm", "bin_left_mm": "mm", "bin_right_mm": "mm"},
    ),
    CHASER_CRA_SUMMARY_TABLE: _contract(
        CHASER_CRA_SUMMARY_TABLE,
        "recording_x_cra_primary_endpoint",
        ("recording_id",),
        ("delta_agg", "delta_inert", "specificity_distance"),
        {"delta_agg": "mm", "delta_inert": "mm", "specificity_distance": "mm"},
    ),
    CHASER_CRA_OBJECT_PHASE_TABLE: _contract(
        CHASER_CRA_OBJECT_PHASE_TABLE,
        "recording_x_cra_phase_x_chaser_object",
        ("recording_id", "phase_axis_index", "object_index"),
        ("phase_label", "object_index", "object_role", "median_distance_mm"),
        {"median_distance_mm": "mm", "mean_distance_mm": "mm"},
    ),
    CHASER_CRA_QUADRANT_TABLE: _contract(
        CHASER_CRA_QUADRANT_TABLE,
        "recording_x_cra_phase_x_quadrant",
        ("recording_id", "phase_axis_index", "quadrant_id"),
        ("phase_label", "quadrant_id", "occupancy_fraction"),
    ),
    CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE: _contract(
        CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE,
        "recording_x_cra_near_field_endpoint",
        ("recording_id",),
        (
            "approach_p05_delta_agg",
            "approach_p05_delta_inert",
            "nearzone_occ_specificity",
        ),
        {"approach_p05_delta_agg": "mm", "approach_p05_delta_inert": "mm"},
    ),
    CHASER_CRA_NEAR_FIELD_OBJECT_PHASE_TABLE: _contract(
        CHASER_CRA_NEAR_FIELD_OBJECT_PHASE_TABLE,
        "recording_x_cra_phase_x_chaser_object_near_field",
        ("recording_id", "phase_axis_index", "object_index"),
        ("phase_label", "object_index", "object_role", "approach_p05_mm"),
        {"approach_p05_mm": "mm", "approach_p10_mm": "mm"},
    ),
    CHASER_CRA_NEAR_FIELD_RADIAL_TABLE: _contract(
        CHASER_CRA_NEAR_FIELD_RADIAL_TABLE,
        "recording_x_cra_phase_x_chaser_object_x_radial_bin",
        ("recording_id", "phase_axis_index", "object_index", "radial_bin_index"),
        ("phase_label", "object_role", "radial_bin_index", "radial_count"),
        {"radial_bin_center_mm": "mm", "radial_density_per_mm2": "1/mm^2"},
    ),
    CHASER_CRA_NEAR_FIELD_CDF_TABLE: _contract(
        CHASER_CRA_NEAR_FIELD_CDF_TABLE,
        "recording_x_cra_phase_x_chaser_object_x_distance_threshold",
        ("recording_id", "phase_axis_index", "object_index", "cdf_threshold_index"),
        ("phase_label", "object_role", "cdf_threshold_index", "cdf_fraction"),
        {"distance_threshold_mm": "mm"},
    ),
    CHASER_EGOCENTRIC_SUMMARY_TABLE: _contract(
        CHASER_EGOCENTRIC_SUMMARY_TABLE,
        "recording_x_chaser_epoch_x_chaser_egocentric_summary",
        ("recording_id", "window_id", "chaser_index"),
        ("window_id", "chaser_index", "mean_alignment_cos", "circular_mean_bearing_deg"),
        {"circular_mean_bearing_deg": "deg"},
    ),
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE: _contract(
        CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
        "recording_x_chaser_epoch_x_chaser_x_distance_x_bearing_bin",
        (
            "recording_id",
            "window_id",
            "chaser_index",
            "distance_bin_index",
            "bearing_bin_index",
        ),
        (
            "window_id",
            "chaser_index",
            "distance_bin_index",
            "bearing_bin_index",
            "hist_count",
        ),
        {"distance_bin_center_mm": "mm", "bearing_bin_center_deg": "deg"},
    ),
    STATISTICS_TABLE: _contract(
        STATISTICS_TABLE,
        "statistical_result",
        ("stat_result_id",),
        ("stat_result_id", "source_export_run_id", "source_table", "metric_name", "status"),
    ),
    DESCRIPTIVE_TABLE: _contract(
        DESCRIPTIVE_TABLE,
        "descriptive_result",
        ("descriptive_result_id",),
        (
            "descriptive_result_id",
            "source_export_run_id",
            "source_table",
            "metric_name",
            "unit_count",
        ),
    ),
}

DEFAULT_TABLES = (
    RECORDING_SUMMARY_TABLE,
    STIMULUS_STEPS_TABLE,
    STIMULUS_STEP_SUMMARY_TABLE,
    STIMULUS_RESPONSE_TABLE,
    SWIM_BOUT_METRICS_TABLE,
    BOUT_KINEMATICS_METRICS_TABLE,
)

CHASER_TABLES = (
    CHASER_SPATIAL_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_CRA_SUMMARY_TABLE,
    CHASER_CRA_OBJECT_PHASE_TABLE,
    CHASER_CRA_QUADRANT_TABLE,
    CHASER_CRA_NEAR_FIELD_SUMMARY_TABLE,
    CHASER_CRA_NEAR_FIELD_OBJECT_PHASE_TABLE,
    CHASER_CRA_NEAR_FIELD_RADIAL_TABLE,
    CHASER_CRA_NEAR_FIELD_CDF_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
)

ALL_TABLES = DEFAULT_TABLES + CHASER_TABLES


def contract_snapshot(table_names: Sequence[str]) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(table_names) - set(TABLE_CONTRACTS))
    if unknown:
        raise ValueError(f"No version-2 table contract is registered for: {', '.join(unknown)}")
    return {table_name: TABLE_CONTRACTS[table_name].to_dict() for table_name in table_names}


def _canonical_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "benign":
        return "inert"
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonical_value(item) for item in value)
    if isinstance(value, Mapping):
        return {
            str(key).replace("benign", "inert"): _canonical_value(item)
            for key, item in value.items()
        }
    return value


def canonicalize_export_row(table_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert source-analysis vocabulary into the strict V2 export schema."""

    if table_name not in TABLE_CONTRACTS:
        raise ValueError(f"Unknown version-2 table: {table_name}")
    out: dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        key = str(raw_key).replace("benign", "inert")
        value = _canonical_value(raw_value)
        if key in out and out[key] != value:
            raise ValueError(f"Conflicting values while canonicalizing {raw_key!r} to {key!r}")
        out[key] = value
    out["export_schema_version"] = EXPORT_SCHEMA_VERSION
    out["table_name"] = table_name
    forbidden = sorted(key for key in out if "benign" in key.lower())
    if forbidden:
        raise ValueError(f"Version-2 row contains forbidden legacy columns: {forbidden}")
    return out


def validate_table_columns(table_name: str, columns: Sequence[str]) -> tuple[str, ...]:
    contract = TABLE_CONTRACTS[table_name]
    return tuple(sorted(set(contract.required_columns) - set(columns)))


__all__ = [name for name in globals() if name.isupper()] + [
    "TableContract",
    "canonicalize_export_row",
    "contract_snapshot",
    "validate_table_columns",
]
