"""Exact physical Arrow contracts for immutable analytics exports.

The logical V2 table contracts intentionally began as minimum-column
contracts.  This module versions the independent physical Arrow layer.  A
table is either governed by one installed exact schema or is named explicitly
as an inferred-V2 compatibility table in the closed manifest envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from .contracts import (
    ACTIVITY_SPATIAL_TIME_BINS_TABLE,
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    BOUT_KINEMATICS_METRICS_TABLE,
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_SPATIAL_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    DESCRIPTIVE_TABLE,
    EYE_TRACE_SAMPLES_TABLE,
    KINEMATICS_SAMPLES_TABLE,
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
    STATISTICS_TABLE,
    STIMULUS_RESPONSE_TABLE,
    STIMULUS_STEP_SUMMARY_TABLE,
    STIMULUS_STEPS_TABLE,
    SWIM_BOUT_METRICS_TABLE,
    TAIL_TRACE_SAMPLES_TABLE,
    TABLE_CONTRACTS,
)


ARROW_CONTRACT_ENVELOPE_SCHEMA_ID = "palette.analytics_export.arrow_contracts"
ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION = 1
ARROW_TABLE_SCHEMA_VERSION = 1
EXACT_ARROW_SCHEMA_TABLES = (
    POSITION_OCCUPANCY_HISTOGRAM_TABLE,
    RECORDING_SUMMARY_TABLE,
    STIMULUS_STEPS_TABLE,
    STIMULUS_STEP_SUMMARY_TABLE,
    STIMULUS_RESPONSE_TABLE,
    SWIM_BOUT_METRICS_TABLE,
    BOUT_KINEMATICS_METRICS_TABLE,
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
    STATISTICS_TABLE,
    DESCRIPTIVE_TABLE,
    KINEMATICS_SAMPLES_TABLE,
    ACTIVITY_SPATIAL_TIME_BINS_TABLE,
    EYE_TRACE_SAMPLES_TABLE,
    TAIL_TRACE_SAMPLES_TABLE,
    CHASER_SPATIAL_TABLE,
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_BOUT_EVENTS_TABLE,
    CHASER_BOUT_HISTOGRAM_TABLE,
    CHASER_IBI_HISTOGRAM_TABLE,
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
    CHASER_DISTANCE_HISTOGRAM_TABLE,
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
)

_ENVELOPE_FIELDS = {
    "schema_id",
    "schema_version",
    "exact_tables",
    "inferred_v2_compatibility_tables",
    "payload_sha256",
}
_TABLE_FIELDS = {
    "schema_id",
    "schema_version",
    "table_name",
    "fields",
    "payload_sha256",
}
_FIELD_FIELDS = {"name", "arrow_type", "nullable"}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ArrowFieldContract:
    """One exact Arrow field declaration."""

    name: str
    arrow_type: str
    nullable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arrow_type": self.arrow_type,
            "nullable": self.nullable,
        }


@dataclass(frozen=True)
class ArrowTableContract:
    """Closed ordered Arrow schema for one maintained export table."""

    table_name: str
    fields: tuple[ArrowFieldContract, ...]
    schema_version: int = ARROW_TABLE_SCHEMA_VERSION

    @property
    def schema_id(self) -> str:
        return f"palette.analytics_export.arrow_table.{self.table_name}"

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "table_name": self.table_name,
            "fields": [field.to_dict() for field in self.fields],
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "payload_sha256": self.payload_sha256}


def _field(name: str, arrow_type: str, *, nullable: bool = False) -> ArrowFieldContract:
    return ArrowFieldContract(name=name, arrow_type=arrow_type, nullable=nullable)


_EXPORT_IDENTITY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
)

_COLLECTION_FIELDS = (
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)

# These fields describe the independently selected chaser-distance authority.
# Nullable provenance reflects the current compatibility reader. The dormant
# derivative tables remain fail closed until their own component manifests are
# sealed; exact Arrow schemas do not reactivate those readers.
_CHASER_RUN_FIELDS = (
    _field("chaser_distance_run", "string"),
    _field("chaser_distance_path", "string"),
    _field("chaser_distance_schema_id", "string", nullable=True),
    _field("chaser_distance_schema_version", "int64", nullable=True),
    _field("chaser_distance_method", "string", nullable=True),
    _field("chaser_distance_method_version", "string", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_stimulus_run", "string", nullable=True),
    _field("source_stimulus_path", "string", nullable=True),
    _field("source_stimulus_epoch_run", "string", nullable=True),
    _field("source_stimulus_epoch_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("coordinate_frame", "string", nullable=True),
    _field("coordinate_origin", "string", nullable=True),
    _field("fps", "float64", nullable=True),
    _field("total_frames", "int64", nullable=True),
    _field("pixels_per_mm_projector", "float64", nullable=True),
)

_CHASER_OPTIONAL_WINDOW_FIELDS = (
    _field("window_index", "int64"),
    _field("window_id", "int64"),
    _field("window_label", "string", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("start_time_s", "float64", nullable=True),
    _field("end_time_s", "float64", nullable=True),
    _field("duration_s", "float64", nullable=True),
)

_CHASER_REQUIRED_WINDOW_FIELDS = (
    _field("window_id", "int64"),
    _field("window_index", "int64"),
    _field("window_label", "string"),
    _field("start_frame", "int64"),
    _field("end_frame", "int64"),
    _field("start_time_s", "float64"),
    _field("end_time_s", "float64"),
    _field("duration_s", "float64"),
)

_EPOCH_BEHAVIOR_COMPONENT_FIELDS = (
    _field("epoch_behavior_component", "string"),
    _field("epoch_behavior_path", "string"),
    _field("epoch_behavior_schema_id", "string", nullable=True),
    _field("epoch_behavior_schema_version", "int64", nullable=True),
    _field("epoch_behavior_method", "string", nullable=True),
    _field("epoch_behavior_method_version", "string", nullable=True),
    _field("epoch_behavior_status", "string", nullable=True),
    _field("epoch_behavior_created_at_utc", "string", nullable=True),
    _field("epoch_behavior_source_refs_json", "string"),
    _field("epoch_behavior_parameters_json", "string"),
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("source_track_kinematics_scope", "string", nullable=True),
    _field("source_track_kinematics_track_id", "int64", nullable=True),
    _field("source_track_kinematics_track_path", "string", nullable=True),
    _field("source_swim_bout_run", "string", nullable=True),
    _field("source_swim_bout_path", "string", nullable=True),
    _field("source_swim_bout_level_path", "string", nullable=True),
    _field("source_speed_level", "string", nullable=True),
    _field("swim_bout_signal_level", "string", nullable=True),
)

_CHASER_SPATIAL_FIELDS = _EXPORT_IDENTITY_FIELDS + (
    _field("detection_occupancy_run", "string"),
    _field("detection_occupancy_path", "string"),
    _field("detection_occupancy_schema_id", "string", nullable=True),
    _field("detection_occupancy_schema_version", "int64", nullable=True),
    _field("detection_occupancy_method", "string", nullable=True),
    _field("detection_occupancy_method_version", "string", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_stimulus_epoch_run", "string", nullable=True),
    _field("source_stimulus_epoch_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("zone_schema_id", "string", nullable=True),
    _field("zone_schema_version", "int64", nullable=True),
    _field("zone_set_id", "string"),
    _field("zone_set_source", "string", nullable=True),
    _field("zone_set_source_ref", "string", nullable=True),
    _field("coordinate_frame", "string", nullable=True),
    _field("coordinate_origin", "string", nullable=True),
    _field("x_axis_direction", "string", nullable=True),
    _field("y_axis_direction", "string", nullable=True),
    _field("width_px", "int64", nullable=True),
    _field("height_px", "int64", nullable=True),
    _field("fps", "float64", nullable=True),
    _field("detection_selection_policy", "string", nullable=True),
    _field("zone_overlap_policy", "string", nullable=True),
    _field("time_basis", "string", nullable=True),
    _field("window_index", "int64"),
    _field("window_id", "int64"),
    _field("window_label", "string", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("start_time_s", "float64", nullable=True),
    _field("end_time_s", "float64", nullable=True),
    _field("duration_s", "float64", nullable=True),
    _field("zone_index", "int64"),
    _field("zone_id", "string"),
    _field("zone_label", "string"),
    _field("display_order", "int64", nullable=True),
    _field("geometry_type", "string", nullable=True),
    _field("x_min", "float64", nullable=True),
    _field("y_min", "float64", nullable=True),
    _field("x_max", "float64", nullable=True),
    _field("y_max", "float64", nullable=True),
    _field("frame_count", "int64"),
    _field("time_s", "float64", nullable=True),
    _field("fraction_of_epoch", "float64", nullable=True),
    _field("fraction_of_detected", "float64", nullable=True),
    _field("detected_frame_count", "int64", nullable=True),
    _field("missing_frame_count", "int64", nullable=True),
    _field("total_span_frames", "int64", nullable=True),
    _field("coverage_pct", "float64", nullable=True),
) + _COLLECTION_FIELDS

_CHASER_DISTANCE_SUMMARY_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _CHASER_OPTIONAL_WINDOW_FIELDS
    + (
        _field("chaser_column_index", "int64"),
        _field("chaser_index", "int64"),
        _field("behavior_class_id", "int64"),
        _field("behavior_class", "string"),
        _field("threshold_mm", "float64", nullable=True),
        _field("valid_frame_count", "int64"),
        _field("mean_distance_mm", "float64", nullable=True),
        _field("min_distance_mm", "float64", nullable=True),
        _field("p05_distance_mm", "float64", nullable=True),
        _field("p50_distance_mm", "float64", nullable=True),
        _field("p95_distance_mm", "float64", nullable=True),
        _field("fraction_within_threshold", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_EPOCH_BEHAVIOR_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _EPOCH_BEHAVIOR_COMPONENT_FIELDS
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("total_span_frames", "int64"),
        _field("valid_frame_count", "int64"),
        _field("missing_frame_count", "int64"),
        _field("tracking_dropout_fraction", "float64", nullable=True),
        _field("center_distance_sample_count", "int64"),
        _field("mean_distance_from_arena_center_mm", "float64", nullable=True),
        _field("median_distance_from_arena_center_mm", "float64", nullable=True),
        _field("p05_distance_from_arena_center_mm", "float64", nullable=True),
        _field("p95_distance_from_arena_center_mm", "float64", nullable=True),
        _field("max_distance_from_arena_center_mm", "float64", nullable=True),
        _field("arena_radius_mm", "float64", nullable=True),
        _field("wall_band_mm", "float64"),
        _field("wall_frame_count", "int64"),
        _field("wall_fraction", "float64", nullable=True),
        _field("wall_time_s", "float64"),
        _field("speed_sample_count", "int64"),
        _field("mean_speed_mm_s", "float64", nullable=True),
        _field("median_speed_mm_s", "float64", nullable=True),
        _field("p05_speed_mm_s", "float64", nullable=True),
        _field("p95_speed_mm_s", "float64", nullable=True),
        _field("max_speed_mm_s", "float64", nullable=True),
        _field("total_path_mm", "float64", nullable=True),
        _field("bout_count", "int64"),
        _field("bout_rate_per_min", "float64", nullable=True),
        _field("median_bout_duration_s", "float64", nullable=True),
        _field("mean_bout_duration_s", "float64", nullable=True),
        _field("median_bout_path_length_mm", "float64", nullable=True),
        _field("mean_bout_path_length_mm", "float64", nullable=True),
        _field("bout_heading_sample_count", "int64"),
        _field("mean_bout_net_heading_change_deg", "float64", nullable=True),
        _field("median_bout_net_heading_change_deg", "float64", nullable=True),
        _field("mean_abs_bout_net_heading_change_deg", "float64", nullable=True),
        _field("median_abs_bout_net_heading_change_deg", "float64", nullable=True),
        _field("mean_bout_heading_path_deg", "float64", nullable=True),
        _field("median_bout_heading_path_deg", "float64", nullable=True),
        _field("inter_bout_interval_count", "int64"),
        _field("mean_inter_bout_interval_s", "float64", nullable=True),
        _field("median_inter_bout_interval_s", "float64", nullable=True),
        _field("p05_inter_bout_interval_s", "float64", nullable=True),
        _field("p95_inter_bout_interval_s", "float64", nullable=True),
        _field("inter_bout_interval_rate_per_min", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_BOUT_EVENT_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _EPOCH_BEHAVIOR_COMPONENT_FIELDS
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("bout_source_row", "int64"),
        _field("bout_id", "int64"),
        _field("bout_event_frame", "int64"),
        _field("bout_event_time_s", "float64", nullable=True),
        _field("bout_start_frame", "int64"),
        _field("bout_end_frame", "int64"),
        _field("bout_start_time_s", "float64", nullable=True),
        _field("bout_end_time_s", "float64", nullable=True),
        _field("bout_duration_s", "float64", nullable=True),
        _field("bout_path_length_mm", "float64", nullable=True),
        _field("bout_net_heading_change_deg", "float64", nullable=True),
        _field("abs_bout_net_heading_change_deg", "float64", nullable=True),
        _field("bout_heading_path_deg", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_EPOCH_HISTOGRAM_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _EPOCH_BEHAVIOR_COMPONENT_FIELDS
    + (
        _field("histogram_dataset", "string"),
        _field("histogram_bin_contract_json", "string"),
        _field("metric_name", "string"),
        _field("units", "string"),
    )
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("bin_index", "int64"),
        _field("bin_left", "float64"),
        _field("bin_right", "float64"),
        _field("bin_center", "float64"),
        _field("bin_width", "float64"),
        _field("hist_count", "int64"),
        _field("hist_fraction", "float64", nullable=True),
        _field("hist_density", "float64", nullable=True),
        _field("source_sample_count", "int64"),
        _field("finite_sample_count", "int64"),
        _field("bin_policy", "string"),
    )
    + _COLLECTION_FIELDS
)

_CENTER_DISTANCE_COMPONENT_FIELDS = _EPOCH_BEHAVIOR_COMPONENT_FIELDS[:10] + (
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("source_swim_bout_run", "string", nullable=True),
)

_CHASER_CENTER_DISTANCE_HISTOGRAM_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _CENTER_DISTANCE_COMPONENT_FIELDS
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("bin_index", "int64"),
        _field("bin_left_mm", "float64"),
        _field("bin_right_mm", "float64"),
        _field("bin_center_mm", "float64"),
        _field("bin_width_mm", "float64"),
        _field("hist_count", "int64"),
        _field("hist_fraction", "float64", nullable=True),
        _field("hist_density_per_mm", "float64", nullable=True),
        _field("valid_frame_count", "int64"),
        _field("arena_radius_mm", "float64"),
        _field("wall_band_mm", "float64"),
        _field("geometry_status", "string"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_SPEED_DISTANCE_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _CHASER_OPTIONAL_WINDOW_FIELDS
    + (
        _field("source_position_path", "string"),
        _field("source_distance_path", "string"),
        _field("speed_distance_definition", "string"),
        _field("chaser_column_index", "int64"),
        _field("chaser_index", "int64"),
        _field("distance_bin_index", "int64"),
        _field("distance_bin_left_mm", "float64"),
        _field("distance_bin_right_mm", "float64"),
        _field("distance_bin_center_mm", "float64"),
        _field("distance_bin_width_mm", "float64"),
        _field("speed_sample_count", "int64"),
        _field("speed_sum_mm_s", "float64"),
        _field("mean_speed_mm_s", "float64", nullable=True),
        _field("median_speed_mm_s", "float64", nullable=True),
        _field("p05_speed_mm_s", "float64", nullable=True),
        _field("p95_speed_mm_s", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_DISTANCE_HISTOGRAM_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _CHASER_OPTIONAL_WINDOW_FIELDS
    + (
        _field("chaser_column_index", "int64"),
        _field("chaser_index", "int64"),
        _field("behavior_class_id", "int64"),
        _field("behavior_class", "string"),
        _field("distance_bin_index", "int64"),
        _field("bin_left_mm", "float64", nullable=True),
        _field("bin_right_mm", "float64", nullable=True),
        _field("bin_center_mm", "float64", nullable=True),
        _field("bin_width_mm", "float64", nullable=True),
        _field("hist_count", "int64"),
        _field("hist_density", "float64", nullable=True),
        _field("valid_sample_count", "int64", nullable=True),
        _field("density_normalization", "string", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_QUADRANT_COMPONENT_FIELDS = (
    _field("cra_primary_endpoint_component", "string"),
    _field("cra_primary_endpoint_path", "string"),
    _field("source_cra_primary_endpoint_component", "string"),
    _field("source_cra_primary_endpoint_path", "string"),
    _field("source_component_schema_id", "string"),
    _field("source_component_schema_version", "int64"),
    _field("source_component_fingerprint", "string", nullable=True),
    _field("source_component_fingerprint_status", "string", nullable=True),
    _field("cra_primary_endpoint_schema_id", "string"),
    _field("cra_primary_endpoint_schema_version", "int64"),
    _field("cra_primary_endpoint_method", "string"),
    _field("cra_primary_endpoint_method_version", "string"),
    _field("cra_primary_endpoint_created_at_utc", "string"),
    _field("endpoint_status", "string"),
    _field("cra_primary_endpoint_source_refs_json", "string"),
    _field("cra_primary_endpoint_parameters_json", "string"),
    _field("qc_warnings_json", "string"),
    _field("diagnostics_json", "string"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("x_axis_direction", "string", nullable=True),
    _field("y_axis_direction", "string", nullable=True),
    _field("quadrant_bounds_source", "string"),
    _field("quadrant_width_px", "float64"),
    _field("quadrant_height_px", "float64"),
)

_NEAR_FIELD_COMPONENT_FIELDS = (
    _field("cra_near_field_component", "string"),
    _field("cra_near_field_path", "string"),
    _field("source_cra_near_field_component", "string"),
    _field("source_cra_near_field_path", "string"),
    _field("source_component_schema_id", "string"),
    _field("source_component_schema_version", "int64"),
    _field("source_component_fingerprint", "string", nullable=True),
    _field("source_component_fingerprint_status", "string", nullable=True),
    _field("cra_near_field_schema_id", "string"),
    _field("cra_near_field_schema_version", "int64"),
    _field("cra_near_field_method", "string"),
    _field("cra_near_field_method_version", "string"),
    _field("cra_near_field_created_at_utc", "string"),
    _field("endpoint_status", "string"),
    _field("cra_near_field_source_refs_json", "string"),
    _field("cra_near_field_parameters_json", "string"),
    _field("qc_warnings_json", "string"),
    _field("diagnostics_json", "string"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("source_quadrant_occupancy_component", "string"),
    _field("source_quadrant_occupancy_path", "string"),
    _field("x_axis_direction", "string", nullable=True),
    _field("y_axis_direction", "string", nullable=True),
    _field("geometry_status", "string"),
    _field("arena_shape", "string"),
    _field("arena_geometry_source", "string"),
    _field("arena_center_x_px", "float64", nullable=True),
    _field("arena_center_y_px", "float64", nullable=True),
    _field("arena_radius_px", "float64", nullable=True),
    _field("arena_width_px", "float64", nullable=True),
    _field("arena_height_px", "float64", nullable=True),
    _field("r_zone_mm", "float64"),
    _field("r_in_mm", "float64"),
    _field("r_out_mm", "float64"),
    _field("perimeter_band_mm", "float64"),
)

_EGOCENTRIC_COMPONENT_FIELDS = (
    _field("egocentric_component_name", "string"),
    _field("egocentric_component_path", "string"),
    _field("egocentric_schema_id", "string"),
    _field("egocentric_schema_version", "int64"),
    _field("egocentric_method", "string"),
    _field("egocentric_method_version", "string"),
    _field("egocentric_created_at_utc", "string"),
    _field("egocentric_source_refs_json", "string"),
    _field("egocentric_parameters_json", "string"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_track_id", "int64"),
    _field("source_track_kinematics_track_path", "string"),
    _field("source_heading_array", "string"),
    _field("heading_level", "string"),
    _field("angle_convention", "string"),
    _field("distance_bin_width_mm", "float64"),
    _field("bearing_bin_width_deg", "float64"),
)

_CHASER_QUADRANT_SUMMARY_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _QUADRANT_COMPONENT_FIELDS
    + (
        _field("fish_id", "string"),
        _field("dpf", "int64", nullable=True),
        _field("chaser_count", "int64"),
        _field("phase_labels", "string"),
        _field("valid_frame_count_by_phase", "string"),
        _field("per_chaser", "string"),
        _field("per_role", "string"),
        _field("pairwise_role_contrast_policy", "string"),
        _field("cra_summary_recording_id", "string"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_QUADRANT_PHASE_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _QUADRANT_COMPONENT_FIELDS
    + (
        _field("phase_axis_index", "int64"),
        _field("phase_index", "int64", nullable=True),
        _field("phase_label", "string"),
        _field("source_window_label", "string", nullable=True),
        _field("source_start_frame", "int64", nullable=True),
        _field("source_end_frame", "int64", nullable=True),
        _field("effective_start_frame", "int64", nullable=True),
        _field("effective_end_frame", "int64", nullable=True),
        _field("settle_excluded_frame_count", "int64", nullable=True),
        _field("object_column_index", "int64"),
        _field("object_index", "int64"),
        _field("object_role", "string"),
        _field("behavior_class", "string"),
        _field("raw_color_hex", "string", nullable=True),
        _field("enable_chase", "bool", nullable=True),
        _field("behavior_mode", "int64", nullable=True),
        _field("start_position_preset", "string", nullable=True),
        _field("end_position_preset", "string", nullable=True),
        _field("object_x_px", "float64", nullable=True),
        _field("object_y_px", "float64", nullable=True),
        _field("object_x_mm", "float64", nullable=True),
        _field("object_y_mm", "float64", nullable=True),
        _field("object_quadrant_code", "int64", nullable=True),
        _field("object_quadrant_label", "string", nullable=True),
        _field("object_position_sample_count", "int64", nullable=True),
        _field("object_max_drift_mm", "float64", nullable=True),
        _field("object_median_drift_mm", "float64", nullable=True),
        _field("median_distance_mm", "float64", nullable=True),
        _field("mean_distance_mm", "float64", nullable=True),
        _field("occupancy_fraction", "float64", nullable=True),
        _field("occupancy_fraction_of_epoch", "float64", nullable=True),
        _field("valid_frame_count", "int64", nullable=True),
        _field("distance_valid_frame_count", "int64", nullable=True),
        _field("total_frame_count", "int64", nullable=True),
        _field("missing_frame_count", "int64", nullable=True),
        _field("tracking_dropout_fraction", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_QUADRANT_DENSITY_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _QUADRANT_COMPONENT_FIELDS
    + (
        _field("fish_id", "string"),
        _field("phase_axis_index", "int64"),
        _field("phase_index", "int64", nullable=True),
        _field("phase_label", "string"),
        _field("source_window_label", "string", nullable=True),
        _field("source_start_frame", "int64", nullable=True),
        _field("source_end_frame", "int64", nullable=True),
        _field("effective_start_frame", "int64"),
        _field("effective_end_frame", "int64"),
        _field("settle_excluded_frame_count", "int64", nullable=True),
        _field("quadrant_code", "int64"),
        _field("quadrant_id", "string"),
        _field("quadrant_label", "string"),
        _field("display_order", "int64"),
        _field("frame_count", "int64"),
        _field("occupancy_fraction", "float64", nullable=True),
        _field("fraction_of_detected", "float64", nullable=True),
        _field("occupancy_fraction_of_epoch", "float64", nullable=True),
        _field("fraction_of_epoch", "float64", nullable=True),
        _field("total_frame_count", "int64"),
        _field("valid_frame_count", "int64"),
        _field("quadrant_valid_frame_count", "int64"),
        _field("missing_frame_count", "int64"),
        _field("out_of_bounds_frame_count", "int64"),
        _field("tracking_dropout_fraction", "float64", nullable=True),
        _field("chaser_object_index", "int64", nullable=True),
        _field("chaser_object_role", "string"),
        _field("chaser_raw_color_hex", "string", nullable=True),
        _field("chaser_x_px", "float64", nullable=True),
        _field("chaser_y_px", "float64", nullable=True),
        _field("chaser_quadrant_code", "int64", nullable=True),
        _field("chaser_quadrant_label", "string", nullable=True),
        _field("chaser_quadrant_occ", "float64", nullable=True),
        _field("is_chaser_quadrant", "bool"),
        _field("series_role", "string"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_NEAR_FIELD_SUMMARY_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _NEAR_FIELD_COMPONENT_FIELDS
    + (
        _field("fish_id", "string"),
        _field("dpf", "int64", nullable=True),
        _field("chaser_count", "int64"),
        _field("phase_labels", "string"),
        _field("approach_percentile_cdf_max_abs_error", "float64", nullable=True),
        _field("per_chaser", "string"),
        _field("per_role", "string"),
        _field("fish_phase_values", "string"),
        _field("pairwise_role_contrast_policy", "string"),
        _field("cra_near_field_summary_recording_id", "string"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_NEAR_FIELD_PHASE_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _NEAR_FIELD_COMPONENT_FIELDS
    + (
        _field("phase_axis_index", "int64"),
        _field("phase_index", "int64", nullable=True),
        _field("phase_label", "string"),
        _field("effective_start_frame", "int64", nullable=True),
        _field("effective_end_frame", "int64", nullable=True),
        _field("total_frame_count", "int64", nullable=True),
        _field("object_column_index", "int64"),
        _field("object_index", "int64"),
        _field("object_role", "string"),
        _field("behavior_class", "string"),
        _field("object_role_code", "int64", nullable=True),
        _field("raw_color_hex", "string", nullable=True),
        _field("near_zone_occupancy_fraction", "float64", nullable=True),
        _field("near_zone_occupancy_fraction_of_epoch", "float64", nullable=True),
        _field("near_zone_dwell_s", "float64", nullable=True),
        _field("near_zone_density_per_mm2", "float64", nullable=True),
        _field("near_zone_available_area_mm2", "float64", nullable=True),
        _field("near_zone_entry_count", "int64", nullable=True),
        _field("near_zone_entry_rate_per_min", "float64", nullable=True),
        _field("near_zone_visit_median_dwell_s", "float64", nullable=True),
        _field("near_zone_visit_total_dwell_s", "float64", nullable=True),
        _field("valid_distance_count", "int64", nullable=True),
        _field("missing_frame_count", "int64", nullable=True),
        _field("tracking_dropout_fraction", "float64", nullable=True),
        # Physical v1 deliberately freezes the maintained percentile axis.
        # Future arbitrary percentile sets need a row axis, not new columns.
        _field("approach_p05_mm", "float64", nullable=True),
        _field("approach_p05_mm_percentile", "float64"),
        _field("approach_p10_mm", "float64", nullable=True),
        _field("approach_p10_mm_percentile", "float64"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_NEAR_FIELD_RADIAL_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _NEAR_FIELD_COMPONENT_FIELDS
    + (
        _field("phase_axis_index", "int64"),
        _field("phase_index", "int64", nullable=True),
        _field("phase_label", "string"),
        _field("effective_start_frame", "int64", nullable=True),
        _field("effective_end_frame", "int64", nullable=True),
        _field("total_frame_count", "int64", nullable=True),
        _field("object_column_index", "int64"),
        _field("object_index", "int64"),
        _field("object_role", "string"),
        _field("behavior_class", "string"),
        _field("raw_color_hex", "string", nullable=True),
        _field("radial_bin_index", "int64"),
        _field("radial_bin_left_mm", "float64"),
        _field("radial_bin_right_mm", "float64"),
        _field("radial_bin_center_mm", "float64"),
        _field("radial_bin_width_mm", "float64"),
        _field("radial_count", "int64", nullable=True),
        _field("radial_fraction", "float64", nullable=True),
        _field("radial_density_per_mm2", "float64", nullable=True),
        _field("radial_available_area_mm2", "float64", nullable=True),
        _field("radial_count_wall_excluded", "int64", nullable=True),
        _field("radial_fraction_wall_excluded", "float64", nullable=True),
        _field("radial_density_wall_excluded_per_mm2", "float64", nullable=True),
        _field("radial_available_area_wall_excluded_mm2", "float64", nullable=True),
        _field("radial_wall_excluded_valid_count", "int64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_NEAR_FIELD_CDF_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _NEAR_FIELD_COMPONENT_FIELDS
    + (
        _field("phase_axis_index", "int64"),
        _field("phase_index", "int64", nullable=True),
        _field("phase_label", "string"),
        _field("effective_start_frame", "int64", nullable=True),
        _field("effective_end_frame", "int64", nullable=True),
        _field("total_frame_count", "int64", nullable=True),
        _field("object_column_index", "int64"),
        _field("object_index", "int64"),
        _field("object_role", "string"),
        _field("behavior_class", "string"),
        _field("raw_color_hex", "string", nullable=True),
        _field("cdf_threshold_index", "int64"),
        _field("distance_threshold_mm", "float64"),
        _field("cdf_fraction", "float64", nullable=True),
    )
    + _COLLECTION_FIELDS
)

_CHASER_EGOCENTRIC_SUMMARY_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _EGOCENTRIC_COMPONENT_FIELDS
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("chaser_column_index", "int64"),
        _field("chaser_index", "int64"),
        _field("behavior_class_id", "int64"),
        _field("behavior_class", "string"),
        _field("valid_frame_count", "int64"),
        _field("circular_mean_bearing_deg", "float64", nullable=True),
        _field("circular_resultant_length", "float64", nullable=True),
        _field("mean_alignment_cos", "float64", nullable=True),
        _field("mean_lateral_sin", "float64", nullable=True),
        _field("fraction_front_45", "float64", nullable=True),
        _field("fraction_lateral_45", "float64", nullable=True),
        _field("fraction_behind_45", "float64", nullable=True),
        _field("front_definition", "string"),
        _field("lateral_definition", "string"),
        _field("behind_definition", "string"),
    )
    + _COLLECTION_FIELDS
)

_CHASER_EGOCENTRIC_HISTOGRAM_FIELDS = (
    _EXPORT_IDENTITY_FIELDS
    + _CHASER_RUN_FIELDS
    + _EGOCENTRIC_COMPONENT_FIELDS
    + _CHASER_REQUIRED_WINDOW_FIELDS
    + (
        _field("chaser_column_index", "int64"),
        _field("chaser_index", "int64"),
        _field("behavior_class_id", "int64"),
        _field("behavior_class", "string"),
        _field("distance_bin_index", "int64"),
        _field("distance_bin_left_mm", "float64"),
        _field("distance_bin_right_mm", "float64"),
        _field("distance_bin_center_mm", "float64"),
        _field("bearing_bin_index", "int64"),
        _field("bearing_bin_left_deg", "float64"),
        _field("bearing_bin_right_deg", "float64"),
        _field("bearing_bin_center_deg", "float64"),
        _field("hist_count", "int64"),
        _field("hist_probability", "float64"),
        _field("valid_sample_count", "int64"),
        _field("probability_normalization", "string"),
    )
    + _COLLECTION_FIELDS
)

# Detection-occupancy is the first exact table because its writer has one
# closed row shape, its physical units are already frozen, and it exercises
# strings, booleans, integers, floats, nullable lineage, and a list field.
_POSITION_OCCUPANCY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("position_occupancy_run", "string"),
    _field("position_occupancy_path", "string"),
    _field("position_occupancy_schema_id", "string", nullable=True),
    _field("position_occupancy_schema_version", "int64", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_segment_kind", "string", nullable=True),
    _field("source_segment_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("window_index", "int64"),
    # ``window_id`` participates in the logical primary key.  Missing IDs must
    # fail publication rather than create an unaddressable nullable key.
    _field("window_id", "int64"),
    _field("window_label", "string", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("start_time_s", "float64", nullable=True),
    _field("end_time_s", "float64", nullable=True),
    _field("duration_s", "float64", nullable=True),
    _field("coordinate_frame", "string"),
    _field("source_coordinate_frame", "string"),
    _field("coordinate_origin", "string"),
    _field("x_axis_direction", "string"),
    _field("y_axis_direction", "string"),
    _field("image_width_px", "float64"),
    _field("image_height_px", "float64"),
    _field("normalized_grid_id", "string"),
    _field("normalized_grid_uniform", "bool"),
    _field("sparse_zero_bins_omitted", "bool"),
    _field("x_bin_count", "int64"),
    _field("y_bin_count", "int64"),
    _field("x_bin_index", "int64"),
    _field("x_bin_left_px", "float64"),
    _field("x_bin_right_px", "float64"),
    _field("x_bin_center_px", "float64"),
    _field("x_bin_width_px", "float64"),
    _field("x_bin_left_fraction", "float64"),
    _field("x_bin_right_fraction", "float64"),
    _field("x_bin_center_fraction", "float64"),
    _field("x_bin_width_fraction", "float64"),
    _field("y_bin_index", "int64"),
    _field("y_bin_left_px", "float64"),
    _field("y_bin_right_px", "float64"),
    _field("y_bin_center_px", "float64"),
    _field("y_bin_width_px", "float64"),
    _field("y_bin_left_fraction", "float64"),
    _field("y_bin_right_fraction", "float64"),
    _field("y_bin_center_fraction", "float64"),
    _field("y_bin_width_fraction", "float64"),
    _field("hist_count", "int64"),
    _field("window_detection_count", "int64", nullable=True),
    _field("covered_frame_count", "int64", nullable=True),
    _field("total_span_frames", "int64", nullable=True),
    _field("coverage_pct", "float64", nullable=True),
    _field("axis_order", "list<string>"),
    _field("source_bin_size_px", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Recording-summary has one closed producer row shape even when its source
# capabilities are absent.  The five shared identity fields and the always
# computed stimulus-step count are required.  Every other field depends on an
# optional stimulus, response, swim-bout, or collection source.  The derived
# protocol hash is a deprecated alias retained in physical v1 so this exact
# contract does not silently change the producer's current row vocabulary.
_RECORDING_SUMMARY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_run", "string", nullable=True),
    _field("stimulus_response_run", "string", nullable=True),
    _field("swim_bout_run", "string", nullable=True),
    _field("stimulus_step_count", "int64"),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("source_track_kinematics_type", "string", nullable=True),
    _field("source_bout_run", "string", nullable=True),
    _field("n_fish", "int64", nullable=True),
    _field("n_steps", "int64", nullable=True),
    _field("global_fish_count", "int64", nullable=True),
    _field("total_distance_mm_sum", "float64", nullable=True),
    _field("mean_speed_mm_s_mean", "float64", nullable=True),
    _field("fraction_moving_mean", "float64", nullable=True),
    _field("total_active_s_sum", "float64", nullable=True),
    _field("swim_bout_default_level", "string", nullable=True),
    _field("swim_bout_default_n_bouts", "int64", nullable=True),
    _field("swim_bout_default_mean_duration_s", "float64", nullable=True),
    _field("swim_bout_default_total_path_length_mm", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Stimulus steps are one row per source step.  The fixed row prefix is followed
# by the exact maintained metadata vocabularies written for moving and
# concentric gratings, then by the protocol-signature and optional collection
# fields.  Child metadata is mode-specific, so every prefixed child field is
# nullable even when its owning source writer always emits a value.  The
# exporter's historical free-form child-attribute flattening is deliberately
# not an open extension point in physical v1: undeclared legacy, future, and
# looming-dot attributes fail before publication instead of changing Parquet
# schemas by observation.
_STIMULUS_STEPS_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_run", "string"),
    _field("step_index", "int64"),
    _field("step_group", "string"),
    _field("step_name", "string", nullable=True),
    _field("stimulus_mode", "string", nullable=True),
    _field("stimulus_mode_id", "int64", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("start_camera_frame", "int64", nullable=True),
    _field("end_camera_frame", "int64", nullable=True),
    _field("duration_s", "float64", nullable=True),
    _field("stimulus_params_json", "string", nullable=True),
    _field("moving_grating_metadata_schema_version", "int64", nullable=True),
    _field("moving_grating_source", "string", nullable=True),
    _field("moving_grating_orientation_degrees_authored", "float64", nullable=True),
    _field("moving_grating_grating_direction_camera_deg", "float64", nullable=True),
    _field("moving_grating_camera_to_projector_offset_deg", "float64", nullable=True),
    _field("moving_grating_direction_mapping_source", "string", nullable=True),
    _field("moving_grating_direction_mapping_status", "string", nullable=True),
    _field("moving_grating_direction_mapping_validated", "bool", nullable=True),
    _field("moving_grating_speed_mm_s", "float64", nullable=True),
    _field("moving_grating_speed_pps", "float64", nullable=True),
    _field("moving_grating_spatial_freq_cycles_per_mm", "float64", nullable=True),
    _field("moving_grating_spatial_freq_rpp", "float64", nullable=True),
    _field("moving_grating_temporal_frequency_hz", "float64", nullable=True),
    _field(
        "moving_grating_actual_rendered_temporal_frequency_hz",
        "float64",
        nullable=True,
    ),
    _field("moving_grating_duty_cycle", "float64", nullable=True),
    _field("concentric_grating_metadata_schema_version", "int64", nullable=True),
    _field("concentric_grating_source", "string", nullable=True),
    _field("concentric_grating_stimulus_role", "string", nullable=True),
    _field("concentric_grating_radial_polarity_authored", "string", nullable=True),
    _field("concentric_grating_radial_sign_authored", "int64", nullable=True),
    _field("concentric_grating_radial_polarity_source", "string", nullable=True),
    _field("concentric_grating_radial_polarity_validated", "bool", nullable=True),
    _field("concentric_grating_speed_mm_s", "float64", nullable=True),
    _field("concentric_grating_speed_pps", "float64", nullable=True),
    _field("concentric_grating_spatial_freq_cycles_per_mm", "float64", nullable=True),
    _field("concentric_grating_spatial_freq_rpp", "float64", nullable=True),
    _field("concentric_grating_temporal_frequency_hz", "float64", nullable=True),
    _field(
        "concentric_grating_actual_rendered_temporal_frequency_hz",
        "float64",
        nullable=True,
    ),
    _field("concentric_grating_duty_cycle", "float64", nullable=True),
    _field("concentric_grating_target_radius_min_mm", "float64", nullable=True),
    _field("concentric_grating_target_radius_max_mm", "float64", nullable=True),
    _field("concentric_grating_target_radius_source", "string", nullable=True),
    _field(
        "concentric_grating_centering_success_fraction_threshold",
        "float64",
        nullable=True,
    ),
    _field("concentric_grating_coordinate_geometry_status", "string", nullable=True),
    _field("protocol_signature_schema", "string"),
    _field("protocol_signature_hash", "string"),
    _field("derived_protocol_hash", "string"),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64"),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Stimulus step summary is one row per fish per stimulus step.  Its closed
# physical vocabulary is the fixed export/lineage prefix, the maintained
# ``step_per_fish`` base bundle, and the optional step-bout summary bundle.
# Source float32 scientific values are represented as nullable Arrow float64
# values because the exporter normalizes non-finite source values to null.
# Optional bout fields remain null when that source bundle is absent; in
# particular, ``num_bouts == 0`` is a real measured value, not a sentinel.
# The source stimulus-response resolver remains a compatibility boundary; this
# physical contract does not promote one recording-local run-selection policy.
_STIMULUS_STEP_SUMMARY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_response_run", "string"),
    _field("source_stimulus_run", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_type", "string"),
    _field("source_bout_run", "string", nullable=True),
    _field("step_index", "int64"),
    _field("step_name", "string"),
    _field("stimulus_mode", "string"),
    _field("stimulus_mode_id", "int64"),
    _field("start_frame", "int64"),
    _field("end_frame", "int64"),
    _field("start_camera_frame", "int64"),
    _field("end_camera_frame", "int64"),
    _field("duration_s", "float64"),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("fish_id", "int64"),
    _field("total_distance_mm", "float64", nullable=True),
    _field("mean_speed_mm_s", "float64", nullable=True),
    _field("median_speed_mm_s", "float64", nullable=True),
    _field("max_speed_mm_s", "float64", nullable=True),
    _field("fraction_moving", "float64", nullable=True),
    _field("coverage", "float64", nullable=True),
    _field("num_bouts", "int64", nullable=True),
    _field("mean_bout_duration_s", "float64", nullable=True),
    _field("mean_interbout_interval_s", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# The response table is a closed union of the maintained step-per-fish,
# moving-grating, moving-OMR, concentric, and radial-OMR logical tables.  Each
# mode-specific value is nullable because a row belongs to only one stimulus
# family.  OMR group attributes are deliberately not flattened: only the
# method version is projected, so adding provenance attributes cannot silently
# change the Parquet schema.
_STIMULUS_RESPONSE_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_response_run", "string"),
    _field("source_stimulus_run", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_type", "string"),
    _field("source_bout_run", "string", nullable=True),
    _field("step_index", "int64"),
    _field("step_name", "string"),
    _field("stimulus_mode", "string"),
    _field("stimulus_mode_id", "int64"),
    _field("start_frame", "int64"),
    _field("end_frame", "int64"),
    _field("start_camera_frame", "int64"),
    _field("end_camera_frame", "int64"),
    _field("duration_s", "float64"),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("fish_id", "int64"),
    _field("total_distance_mm", "float64", nullable=True),
    _field("mean_speed_mm_s", "float64", nullable=True),
    _field("median_speed_mm_s", "float64", nullable=True),
    _field("max_speed_mm_s", "float64", nullable=True),
    _field("fraction_moving", "float64", nullable=True),
    _field("coverage", "float64", nullable=True),
    _field("num_bouts", "int64", nullable=True),
    _field("mean_bout_duration_s", "float64", nullable=True),
    _field("mean_interbout_interval_s", "float64", nullable=True),
    _field("omr_family", "string", nullable=True),
    _field("omr_attr_method_version", "string", nullable=True),
    _field("radial_omr_attr_method_version", "string", nullable=True),
    _field("grating_mean_alignment_cos", "float64", nullable=True),
    _field("grating_resultant_vector_length", "float64", nullable=True),
    _field("grating_fraction_following", "float64", nullable=True),
    _field("grating_fraction_opposing", "float64", nullable=True),
    _field("grating_fraction_perpendicular", "float64", nullable=True),
    _field("grating_speed_weighted_alignment", "float64", nullable=True),
    _field("grating_optomotor_gain", "float64", nullable=True),
    _field("grating_drift_along_grating_mm", "float64", nullable=True),
    _field("grating_drift_perp_grating_mm", "float64", nullable=True),
    _field("grating_latency_to_follow_s", "float64", nullable=True),
    _field("omr_path_index", "float64", nullable=True),
    _field("omr_net_direction_index", "float64", nullable=True),
    _field("parallel_displacement_mm", "float64", nullable=True),
    _field("net_displacement_mm", "float64", nullable=True),
    _field("path_length_mm", "float64", nullable=True),
    _field("valid_transition_count", "int64", nullable=True),
    _field("coverage_fraction", "float64", nullable=True),
    _field("bout_fraction_correct_classified", "float64", nullable=True),
    _field("bout_fraction_correct_all", "float64", nullable=True),
    _field("bout_choice_index", "float64", nullable=True),
    _field("bout_path_index", "float64", nullable=True),
    _field("bout_fraction_correct_weighted_by_path", "float64", nullable=True),
    _field(
        "bout_fraction_correct_weighted_by_displacement",
        "float64",
        nullable=True,
    ),
    _field("bout_parallel_displacement_sum_mm", "float64", nullable=True),
    _field("bout_path_length_sum_mm", "float64", nullable=True),
    _field("bout_displacement_sum_mm", "float64", nullable=True),
    _field("bout_classified_path_length_sum_mm", "float64", nullable=True),
    _field("bout_classified_displacement_sum_mm", "float64", nullable=True),
    _field("bout_classifiable_path_fraction", "float64", nullable=True),
    _field("bout_classifiable_displacement_fraction", "float64", nullable=True),
    _field("bout_count_total", "int64", nullable=True),
    _field("bout_count_correct", "int64", nullable=True),
    _field("bout_count_opposing", "int64", nullable=True),
    _field("bout_count_ambiguous", "int64", nullable=True),
    _field("time_fraction_correct_classified", "float64", nullable=True),
    _field("time_choice_index", "float64", nullable=True),
    _field("time_correct_s", "float64", nullable=True),
    _field("time_opposing_s", "float64", nullable=True),
    _field("time_classified_s", "float64", nullable=True),
    _field("start_position_axis_mm", "float64", nullable=True),
    _field("end_position_axis_mm", "float64", nullable=True),
    _field("mean_position_axis_mm", "float64", nullable=True),
    _field("start_position_axis_norm", "float64", nullable=True),
    _field("end_position_axis_norm", "float64", nullable=True),
    _field("mean_position_axis_norm", "float64", nullable=True),
    _field("fraction_time_correct_side", "float64", nullable=True),
    _field("available_forward_space_at_start_mm", "float64", nullable=True),
    _field("available_backward_space_at_start_mm", "float64", nullable=True),
    _field("available_forward_space_at_start_norm", "float64", nullable=True),
    _field("available_backward_space_at_start_norm", "float64", nullable=True),
    _field(
        "opportunity_normalized_parallel_displacement",
        "float64",
        nullable=True,
    ),
    _field("first_aligned_bout_id", "int64", nullable=True),
    _field("first_aligned_bout_start_frame", "int64", nullable=True),
    _field("first_aligned_bout_latency_s", "float64", nullable=True),
    _field("first_aligned_bout_score", "float64", nullable=True),
    _field("first_opposing_bout_id", "int64", nullable=True),
    _field("first_opposing_bout_start_frame", "int64", nullable=True),
    _field("first_opposing_bout_latency_s", "float64", nullable=True),
    _field("first_opposing_bout_score", "float64", nullable=True),
    _field("first_classified_bout_id", "int64", nullable=True),
    _field("first_classified_bout_start_frame", "int64", nullable=True),
    _field("first_classified_bout_latency_s", "float64", nullable=True),
    _field("first_classified_bout_score", "float64", nullable=True),
    _field("quality_flag", "int64", nullable=True),
    _field("concentric_mean_distance_to_center_mm", "float64", nullable=True),
    _field(
        "concentric_initial_distance_to_center_mm", "float64", nullable=True
    ),
    _field("concentric_final_distance_to_center_mm", "float64", nullable=True),
    _field("concentric_min_distance_to_center_mm", "float64", nullable=True),
    _field("concentric_net_radial_displacement_mm", "float64", nullable=True),
    _field("concentric_fraction_approaching", "float64", nullable=True),
    _field("concentric_mean_radial_heading_cos", "float64", nullable=True),
    _field("concentric_time_to_center_s", "float64", nullable=True),
    _field("concentric_fraction_near_center", "float64", nullable=True),
    _field("concentric_mean_radial_speed_mm_s", "float64", nullable=True),
    _field("concentric_mean_tangential_speed_mm_s", "float64", nullable=True),
    _field("radial_path_index", "float64", nullable=True),
    _field("tangential_bias_index", "float64", nullable=True),
    _field(
        "stimulus_aligned_radial_displacement_mm", "float64", nullable=True
    ),
    _field("radial_displacement_integrated_mm", "float64", nullable=True),
    _field("tangential_displacement_mm", "float64", nullable=True),
    _field("start_radius_mm", "float64", nullable=True),
    _field("end_radius_mm", "float64", nullable=True),
    _field("mean_radius_mm", "float64", nullable=True),
    _field("start_radius_norm", "float64", nullable=True),
    _field("end_radius_norm", "float64", nullable=True),
    _field("mean_radius_norm", "float64", nullable=True),
    _field("available_outward_space_at_start_mm", "float64", nullable=True),
    _field("available_inward_space_at_start_mm", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# The selected default candidate/signal is exported at one row per bout.  The
# complete maintained compact-v2 bout payload is represented, while fields
# absent from historical compatibility sources remain null.  Candidate and
# signal identity are required even though the exporter currently selects one
# of each; this prevents those semantics from being lost if selection expands.
_SWIM_BOUT_METRICS_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_run", "string", nullable=True),
    _field("swim_bout_run", "string"),
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("source_track_kinematics_type", "string", nullable=True),
    _field("track_id", "int64", nullable=True),
    _field("speed_level", "string"),
    _field("candidate_id", "int64"),
    _field("signal_id", "int64"),
    _field("signal_role", "string"),
    _field("signal_source_level", "string", nullable=True),
    _field("detection_method", "string", nullable=True),
    _field("detection_signal_transform_type", "string", nullable=True),
    _field("detection_signal_source_level", "string", nullable=True),
    _field("movement_metric_source_level", "string", nullable=True),
    _field("threshold_mm_s", "float64", nullable=True),
    _field("peak_prominence_mm_s", "float64", nullable=True),
    _field("step_index", "int64", nullable=True),
    _field("step_name", "string", nullable=True),
    _field("stimulus_mode", "string", nullable=True),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("bout_id", "int64"),
    _field("estimator_signal_id", "int64", nullable=True),
    _field("mean_speed_px_s", "float64", nullable=True),
    _field("peak_detection_signal_px_s", "float64", nullable=True),
    _field("peak_frame", "int64", nullable=True),
    _field("peak_time_s", "float64", nullable=True),
    _field("threshold_crossing_valid", "bool", nullable=True),
    _field("start_frame", "int64", nullable=True),
    _field("end_frame", "int64", nullable=True),
    _field("core_start_frame", "int64", nullable=True),
    _field("core_end_frame", "int64", nullable=True),
    _field("duration_frames", "int64", nullable=True),
    _field("duration_s", "float64", nullable=True),
    _field("elapsed_duration_s", "float64", nullable=True),
    _field("observed_duration_s", "float64", nullable=True),
    _field("core_duration_frames", "int64", nullable=True),
    _field("core_duration_s", "float64", nullable=True),
    _field("path_length_mm", "float64", nullable=True),
    _field("path_length_px", "float64", nullable=True),
    _field("net_displacement_mm", "float64", nullable=True),
    _field("net_displacement_px", "float64", nullable=True),
    _field("mean_speed_mm_s", "float64", nullable=True),
    _field("peak_detection_signal_mm_s", "float64", nullable=True),
    _field("peak_physical_speed_mm_s", "float64", nullable=True),
    _field("n_valid_transitions", "int64", nullable=True),
    _field("n_invalid_transitions", "int64", nullable=True),
    _field("valid_transition_fraction", "float64", nullable=True),
    _field("gap_censored", "bool", nullable=True),
    _field("start_time_s", "float64", nullable=True),
    _field("end_time_s", "float64", nullable=True),
    _field("core_start_time_s", "float64", nullable=True),
    _field("core_end_time_s", "float64", nullable=True),
    _field("core_start_time_s_interpolated", "float64", nullable=True),
    _field("core_end_time_s_interpolated", "float64", nullable=True),
    _field("core_duration_s_interpolated", "float64", nullable=True),
    _field("core_start_time_interpolated_valid", "bool", nullable=True),
    _field("core_end_time_interpolated_valid", "bool", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Bout kinematics has one row per bout *and measurement level*.  The exact
# schema is the nullable union of movement, heading, and optional eye-gaze
# metric families.  Fixed-width storage strings are exposed without the
# physical ``_bytes`` suffix, keeping the Parquet contract semantic.
_BOUT_KINEMATICS_METRICS_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("stimulus_run", "string", nullable=True),
    _field("bout_kinematics_run", "string"),
    _field("measurement_level", "string"),
    _field("measurement_family", "string"),
    _field("is_default_heading_level", "bool"),
    _field("source_swim_bout_run", "string", nullable=True),
    _field("source_swim_bout_speed_level", "string", nullable=True),
    _field("source_track_kinematics_run", "string", nullable=True),
    _field("track_id", "int64", nullable=True),
    _field("schema_version", "int64", nullable=True),
    _field("method", "string", nullable=True),
    _field("method_version", "string", nullable=True),
    _field("step_index", "int64", nullable=True),
    _field("step_name", "string", nullable=True),
    _field("stimulus_mode", "string", nullable=True),
    _field("protocol_signature_schema", "string", nullable=True),
    _field("protocol_signature_hash", "string", nullable=True),
    _field("derived_protocol_hash", "string", nullable=True),
    _field("protocol_mode_sequence", "string", nullable=True),
    _field("protocol_duration_sequence_s", "string", nullable=True),
    _field("protocol_step_count", "int64", nullable=True),
    _field("bout_id", "int64"),
    _field("source_start_frame", "int64", nullable=True),
    _field("source_end_frame", "int64", nullable=True),
    _field("source_core_start_frame", "int64", nullable=True),
    _field("source_core_end_frame", "int64", nullable=True),
    _field("detector_duration_s", "float64", nullable=True),
    _field("detector_observed_duration_s", "float64", nullable=True),
    _field("detector_core_duration_s", "float64", nullable=True),
    _field("physical_active_start_frame", "int64", nullable=True),
    _field("physical_active_end_frame", "int64", nullable=True),
    _field("physical_active_start_time_s", "float64", nullable=True),
    _field("physical_active_end_time_s", "float64", nullable=True),
    _field("physical_active_duration_s", "float64", nullable=True),
    _field("physical_active_observed_duration_s", "float64", nullable=True),
    _field(
        "physical_active_start_time_s_interpolated", "float64", nullable=True
    ),
    _field("physical_active_end_time_s_interpolated", "float64", nullable=True),
    _field("physical_active_duration_s_interpolated", "float64", nullable=True),
    _field(
        "physical_active_start_time_interpolated_valid", "bool", nullable=True
    ),
    _field("physical_active_end_time_interpolated_valid", "bool", nullable=True),
    _field("physical_active_sample_count", "int64", nullable=True),
    _field("physical_active_valid_transition_count", "int64", nullable=True),
    _field(
        "physical_active_valid_transition_fraction", "float64", nullable=True
    ),
    _field("physical_active_path_length_mm", "float64", nullable=True),
    _field("physical_active_path_length_px", "float64", nullable=True),
    _field("physical_active_mean_speed_mm_s", "float64", nullable=True),
    _field("physical_active_peak_speed_mm_s", "float64", nullable=True),
    _field("physical_active_threshold_mm_s", "float64", nullable=True),
    _field("physical_active_boundary_margin_s", "float64", nullable=True),
    _field("physical_active_boundary_policy", "string", nullable=True),
    _field("physical_active_boundary_constraint", "string", nullable=True),
    _field("physical_active_valid", "bool", nullable=True),
    _field("failure_reason", "string", nullable=True),
    _field("source_core_start_time_s_interpolated", "float64", nullable=True),
    _field("source_core_end_time_s_interpolated", "float64", nullable=True),
    _field("source_core_duration_s_interpolated", "float64", nullable=True),
    _field("source_core_start_time_interpolated_valid", "bool", nullable=True),
    _field("source_core_end_time_interpolated_valid", "bool", nullable=True),
    _field("source_peak_frame", "int64", nullable=True),
    _field("source_peak_time_s", "float64", nullable=True),
    _field("source_peak_signal_value_mm_s", "float64", nullable=True),
    _field("source_peak_prominence_mm_s", "float64", nullable=True),
    _field("source_peak_width_s", "float64", nullable=True),
    _field("source_peak_width_height_mm_s", "float64", nullable=True),
    _field(
        "source_peak_left_width_frame_interpolated", "float64", nullable=True
    ),
    _field(
        "source_peak_right_width_frame_interpolated", "float64", nullable=True
    ),
    _field("source_peak_left_width_time_s", "float64", nullable=True),
    _field("source_peak_right_width_time_s", "float64", nullable=True),
    _field("source_peak_boundary_mode", "string", nullable=True),
    _field("source_peak_shape_split_policy", "string", nullable=True),
    _field("pre_epoch_start_frame", "int64", nullable=True),
    _field("pre_epoch_end_frame", "int64", nullable=True),
    _field("post_epoch_start_frame", "int64", nullable=True),
    _field("post_epoch_end_frame", "int64", nullable=True),
    _field("pre_heading_mean_deg", "float64", nullable=True),
    _field("post_heading_mean_deg", "float64", nullable=True),
    _field("net_delta_heading_deg", "float64", nullable=True),
    _field("abs_net_delta_heading_deg", "float64", nullable=True),
    _field("pre_position_mean_x_mm", "float64", nullable=True),
    _field("pre_position_mean_y_mm", "float64", nullable=True),
    _field("post_position_mean_x_mm", "float64", nullable=True),
    _field("post_position_mean_y_mm", "float64", nullable=True),
    _field("interbout_epoch_displacement_mm", "float64", nullable=True),
    _field("pre_position_mean_x_px", "float64", nullable=True),
    _field("pre_position_mean_y_px", "float64", nullable=True),
    _field("post_position_mean_x_px", "float64", nullable=True),
    _field("post_position_mean_y_px", "float64", nullable=True),
    _field("interbout_epoch_displacement_px", "float64", nullable=True),
    _field("within_heading_range_deg", "float64", nullable=True),
    _field("within_heading_peak_to_peak_deg", "float64", nullable=True),
    _field("within_heading_path_deg", "float64", nullable=True),
    _field("within_heading_std_deg", "float64", nullable=True),
    _field("within_heading_zero_crossings", "int64", nullable=True),
    _field("within_heading_dominant_frequency_hz", "float64", nullable=True),
    _field("within_angular_velocity_mean_deg_s", "float64", nullable=True),
    _field("within_angular_speed_mean_deg_s", "float64", nullable=True),
    _field("within_angular_speed_max_deg_s", "float64", nullable=True),
    _field("within_angular_velocity_std_deg_s", "float64", nullable=True),
    _field("pre_window_valid", "bool", nullable=True),
    _field("post_window_valid", "bool", nullable=True),
    _field("pre_position_valid", "bool", nullable=True),
    _field("post_position_valid", "bool", nullable=True),
    _field("within_window_valid", "bool", nullable=True),
    _field("within_angular_velocity_valid", "bool", nullable=True),
    _field("dominant_frequency_valid", "bool", nullable=True),
    _field("pre_window_sample_count", "int64", nullable=True),
    _field("post_window_sample_count", "int64", nullable=True),
    _field("pre_position_sample_count", "int64", nullable=True),
    _field("post_position_sample_count", "int64", nullable=True),
    _field("within_window_sample_count", "int64", nullable=True),
    _field("within_angular_velocity_transition_count", "int64", nullable=True),
    _field("within_epoch_start_frame", "int64", nullable=True),
    _field("within_epoch_end_frame", "int64", nullable=True),
    _field("pre_left_gaze_mean_deg", "float64", nullable=True),
    _field("pre_right_gaze_mean_deg", "float64", nullable=True),
    _field("pre_vergence_gaze_mean_deg", "float64", nullable=True),
    _field("pre_vergence_gaze_signed_mean_deg", "float64", nullable=True),
    _field("pre_vergence_gaze_std_deg", "float64", nullable=True),
    _field("pre_vergence_gaze_valid_fraction", "float64", nullable=True),
    _field("pre_converged_fraction", "float64", nullable=True),
    _field("post_left_gaze_mean_deg", "float64", nullable=True),
    _field("post_right_gaze_mean_deg", "float64", nullable=True),
    _field("post_vergence_gaze_mean_deg", "float64", nullable=True),
    _field("post_vergence_gaze_signed_mean_deg", "float64", nullable=True),
    _field("post_vergence_gaze_std_deg", "float64", nullable=True),
    _field("post_vergence_gaze_valid_fraction", "float64", nullable=True),
    _field("post_converged_fraction", "float64", nullable=True),
    _field("within_bout_left_gaze_mean_deg", "float64", nullable=True),
    _field("within_bout_right_gaze_mean_deg", "float64", nullable=True),
    _field("within_bout_vergence_gaze_mean_deg", "float64", nullable=True),
    _field(
        "within_bout_vergence_gaze_signed_mean_deg", "float64", nullable=True
    ),
    _field("within_bout_vergence_gaze_max_deg", "float64", nullable=True),
    _field("within_bout_vergence_gaze_range_deg", "float64", nullable=True),
    _field("within_bout_vergence_gaze_std_deg", "float64", nullable=True),
    _field(
        "within_bout_vergence_gaze_valid_fraction", "float64", nullable=True
    ),
    _field("within_bout_converged_fraction", "float64", nullable=True),
    _field("pre_eye_window_valid", "bool", nullable=True),
    _field("post_eye_window_valid", "bool", nullable=True),
    _field("within_eye_window_valid", "bool", nullable=True),
    _field("pre_eye_sample_count", "int64", nullable=True),
    _field("post_eye_sample_count", "int64", nullable=True),
    _field("within_eye_sample_count", "int64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Baseline behavior summary has a closed producer vocabulary assembled from
# four fixed surfaces: the shared export identity, the fixed chaser/source
# lineage literals, ``build_summary_metrics``, and the three optional
# collection fields.  The source epoch row is deliberately not merged: the
# metrics builder projects exactly eight named bout/IBI summary values.  The
# physical schema therefore stays closed if that source structured dtype later
# grows.  ``fps`` remains nullable in v1 because computation may fall back to
# track FPS while the exported lineage column currently reads only the chaser
# run attribute.  Freezing that discrepancy is representation, not authority
# repair or source promotion.
_BASELINE_BEHAVIOR_SUMMARY_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("chaser_distance_run", "string"),
    _field("chaser_distance_path", "string"),
    _field("chaser_distance_schema_id", "string", nullable=True),
    _field("chaser_distance_schema_version", "int64", nullable=True),
    _field("chaser_distance_method", "string", nullable=True),
    _field("chaser_distance_method_version", "string", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_stimulus_run", "string", nullable=True),
    _field("source_stimulus_path", "string", nullable=True),
    _field("source_stimulus_epoch_run", "string", nullable=True),
    _field("source_stimulus_epoch_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("coordinate_frame", "string"),
    _field("coordinate_origin", "string"),
    _field("fps", "float64", nullable=True),
    _field("total_frames", "int64", nullable=True),
    _field("pixels_per_mm_projector", "float64"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("source_epoch_behavior_component", "string"),
    _field("source_epoch_behavior_path", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_kinematics_track_path", "string"),
    _field("source_speed_level", "string"),
    _field("source_swim_bout_run", "string", nullable=True),
    _field("source_swim_bout_path", "string", nullable=True),
    _field("track_id", "int64"),
    _field("arena_center_x_px", "float64"),
    _field("arena_center_y_px", "float64"),
    _field("arena_radius_px", "float64"),
    _field("baseline_method", "string"),
    _field("baseline_method_version", "string"),
    _field("baseline_window_id", "int64"),
    _field("baseline_window_label", "string"),
    _field("start_frame", "int64"),
    _field("end_frame", "int64"),
    _field("start_time_s", "float64"),
    _field("end_time_s", "float64"),
    _field("duration_s", "float64"),
    _field("total_frame_count", "int64"),
    _field("valid_frame_count", "int64"),
    _field("missing_frame_count", "int64"),
    _field("tracking_dropout_fraction", "float64", nullable=True),
    _field("speed_sample_count", "int64"),
    _field("mean_speed_mm_s", "float64", nullable=True),
    _field("median_speed_mm_s", "float64", nullable=True),
    _field("p95_speed_mm_s", "float64", nullable=True),
    _field("max_speed_mm_s", "float64", nullable=True),
    _field("total_path_mm", "float64", nullable=True),
    _field("bout_count", "int64"),
    _field("bout_rate_per_min", "float64", nullable=True),
    _field("arena_radius_mm", "float64"),
    _field("wall_band_mm", "float64"),
    _field("expected_uniform_wall_fraction", "float64"),
    _field("experimental_area_geometry_type", "string"),
    _field("boundary_distance_method", "string"),
    _field("wall_fraction_denominator", "string"),
    _field("wall_frame_count", "int64"),
    _field("wall_fraction", "float64", nullable=True),
    _field("mean_distance_from_arena_center_mm", "float64", nullable=True),
    _field("median_distance_from_arena_center_mm", "float64", nullable=True),
    _field("p95_distance_from_arena_center_mm", "float64", nullable=True),
    _field("mean_distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("median_distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("p95_distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("mean_center_distance_norm", "float64", nullable=True),
    _field("median_center_distance_norm", "float64", nullable=True),
    _field("x_axis_direction", "string"),
    _field("y_axis_direction", "string"),
    _field("spatial_grid_size", "int64"),
    _field("spatial_valid_sample_count", "int64"),
    _field("spatial_visited_cell_count", "int64"),
    _field("spatial_entropy_normalized", "float64", nullable=True),
    _field("spatial_max_cell_fraction", "float64", nullable=True),
    _field("quadrant_entropy_normalized", "float64", nullable=True),
    _field("quadrant_max_fraction", "float64", nullable=True),
    _field("median_bout_duration_s", "float64", nullable=True),
    _field("mean_bout_duration_s", "float64", nullable=True),
    _field("median_bout_path_length_mm", "float64", nullable=True),
    _field("mean_bout_path_length_mm", "float64", nullable=True),
    _field("median_abs_bout_net_heading_change_deg", "float64", nullable=True),
    _field("mean_abs_bout_net_heading_change_deg", "float64", nullable=True),
    _field("median_inter_bout_interval_s", "float64", nullable=True),
    _field("mean_inter_bout_interval_s", "float64", nullable=True),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Baseline time bins use the same fixed identity/source prefix as the summary
# table, followed by the closed dictionary emitted by ``build_time_bin_metrics``
# and three optional collection fields.  The source epoch row only supplies
# the already-normalized window bounds; it is never merged into an output row,
# so later source columns cannot expand this physical vocabulary.  As in the
# summary contract, ``fps`` is nullable because the current computation may use
# verified track FPS while the exported lineage value reads the chaser attr.
_BASELINE_BEHAVIOR_TIME_BINS_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("chaser_distance_run", "string"),
    _field("chaser_distance_path", "string"),
    _field("chaser_distance_schema_id", "string", nullable=True),
    _field("chaser_distance_schema_version", "int64", nullable=True),
    _field("chaser_distance_method", "string", nullable=True),
    _field("chaser_distance_method_version", "string", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_stimulus_run", "string", nullable=True),
    _field("source_stimulus_path", "string", nullable=True),
    _field("source_stimulus_epoch_run", "string", nullable=True),
    _field("source_stimulus_epoch_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("coordinate_frame", "string"),
    _field("coordinate_origin", "string"),
    _field("fps", "float64", nullable=True),
    _field("total_frames", "int64", nullable=True),
    _field("pixels_per_mm_projector", "float64"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("source_epoch_behavior_component", "string"),
    _field("source_epoch_behavior_path", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_kinematics_track_path", "string"),
    _field("source_speed_level", "string"),
    _field("source_swim_bout_run", "string", nullable=True),
    _field("source_swim_bout_path", "string", nullable=True),
    _field("track_id", "int64"),
    _field("arena_center_x_px", "float64"),
    _field("arena_center_y_px", "float64"),
    _field("arena_radius_px", "float64"),
    _field("baseline_method", "string"),
    _field("baseline_method_version", "string"),
    _field("baseline_window_id", "int64"),
    _field("baseline_window_label", "string"),
    _field("time_bin_index", "int64"),
    _field("relative_start_s", "float64"),
    _field("relative_end_s", "float64"),
    _field("time_bin_duration_s", "float64"),
    _field("source_start_frame", "int64"),
    _field("source_end_frame", "int64"),
    _field("expected_frame_count", "int64"),
    _field("valid_position_count", "int64"),
    _field("valid_position_fraction", "float64", nullable=True),
    _field("speed_sample_count", "int64"),
    _field("mean_speed_mm_s", "float64", nullable=True),
    _field("median_speed_mm_s", "float64", nullable=True),
    _field("p95_speed_mm_s", "float64", nullable=True),
    _field("distance_travelled_mm", "float64", nullable=True),
    _field("mean_center_distance_mm", "float64", nullable=True),
    _field("median_center_distance_mm", "float64", nullable=True),
    _field("mean_distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("median_distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("experimental_area_geometry_type", "string"),
    _field("boundary_distance_method", "string"),
    _field("wall_fraction_denominator", "string"),
    _field("wall_frame_count", "int64"),
    _field("wall_fraction", "float64", nullable=True),
    _field("representative_position_method", "string"),
    _field("representative_x_mm", "float64", nullable=True),
    _field("representative_y_mm", "float64", nullable=True),
    _field("mean_heading_deg", "float64", nullable=True),
    _field("heading_resultant", "float64", nullable=True),
    _field("bout_count", "int64"),
    _field("x_axis_direction", "string"),
    _field("y_axis_direction", "string"),
    _field("time_bin_policy", "string"),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# Baseline kinematic samples retain the same closed identity/source prefix as
# the summary and time-bin tables.  ``build_sample_metrics`` contributes one
# fixed row dictionary for every selected source sample; the source epoch row
# is not merged, and collection publication adds only three named fields.
# ``fps`` retains the companion-table nullable discrepancy.  Sample position,
# motion, wall, and requested-rate values are nullable when their source value
# is unavailable or the full-resolution policy has no requested target rate.
_BASELINE_KINEMATIC_SAMPLES_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("chaser_distance_run", "string"),
    _field("chaser_distance_path", "string"),
    _field("chaser_distance_schema_id", "string", nullable=True),
    _field("chaser_distance_schema_version", "int64", nullable=True),
    _field("chaser_distance_method", "string", nullable=True),
    _field("chaser_distance_method_version", "string", nullable=True),
    _field("source_detection_path", "string", nullable=True),
    _field("source_detection_kind", "string", nullable=True),
    _field("source_stimulus_run", "string", nullable=True),
    _field("source_stimulus_path", "string", nullable=True),
    _field("source_stimulus_epoch_run", "string", nullable=True),
    _field("source_stimulus_epoch_path", "string", nullable=True),
    _field("source_refs_json", "string"),
    _field("coordinate_frame", "string"),
    _field("coordinate_origin", "string"),
    _field("fps", "float64", nullable=True),
    _field("total_frames", "int64", nullable=True),
    _field("pixels_per_mm_projector", "float64"),
    _field("source_chaser_distance_run", "string"),
    _field("source_chaser_distance_path", "string"),
    _field("source_epoch_behavior_component", "string"),
    _field("source_epoch_behavior_path", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_kinematics_track_path", "string"),
    _field("source_speed_level", "string"),
    _field("source_swim_bout_run", "string", nullable=True),
    _field("source_swim_bout_path", "string", nullable=True),
    _field("track_id", "int64"),
    _field("arena_center_x_px", "float64"),
    _field("arena_center_y_px", "float64"),
    _field("arena_radius_px", "float64"),
    _field("baseline_method", "string"),
    _field("baseline_method_version", "string"),
    _field("baseline_window_id", "int64"),
    _field("baseline_window_label", "string"),
    _field("source_sample_index", "int64"),
    _field("source_frame", "int64"),
    _field("source_time_s", "float64"),
    _field("relative_time_s", "float64"),
    _field("x_arena_mm", "float64", nullable=True),
    _field("y_arena_mm", "float64", nullable=True),
    _field("x_arena_fraction", "float64", nullable=True),
    _field("y_arena_fraction", "float64", nullable=True),
    _field("speed_mm_s", "float64", nullable=True),
    _field("heading_deg", "float64", nullable=True),
    _field("frame_path_distance_mm", "float64", nullable=True),
    _field("center_distance_mm", "float64", nullable=True),
    _field("distance_to_arena_boundary_mm", "float64", nullable=True),
    _field("wall", "bool", nullable=True),
    _field("experimental_area_geometry_type", "string"),
    _field("boundary_distance_method", "string"),
    _field("position_valid", "bool"),
    _field("sample_valid", "bool"),
    _field("sampling_policy", "string"),
    _field("sampling_stride_frames", "int64"),
    _field("requested_sample_rate_hz", "float64", nullable=True),
    _field("source_sample_rate_hz", "float64"),
    _field("nominal_sample_rate_hz", "float64"),
    _field("effective_sample_rate_hz", "float64"),
    _field("x_axis_direction", "string"),
    _field("y_axis_direction", "string"),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("collection_manifest_path", "string", nullable=True),
)


# One row per acquisition-frame-aligned sample of one track. Source scientific
# representations remain exact: float32 motion values, float64 physical
# positions, integer lineage/QC codes, and explicit booleans. Invalid float
# values remain IEEE NaN and are interpreted through the validity columns.
_KINEMATICS_SAMPLES_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_motion_manifest_schema_id", "string"),
    _field("source_track_motion_manifest_schema_version", "int64"),
    _field("source_track_motion_manifest_sha256", "string"),
    _field("source_binding_sha256", "string"),
    _field("projection_contract_sha256", "string"),
    _field("source_speed_level", "string"),
    _field("source_sample_rate_hz", "float64"),
    _field("requested_sample_rate_hz", "float64"),
    _field("sampling_stride_frames", "int64"),
    _field("nominal_sample_rate_hz", "float64"),
    _field("sampling_policy", "string"),
    _field("position_coordinate_space", "string"),
    _field("position_coordinate_descriptor_sha256", "string"),
    _field("physical_authority_sha256", "string"),
    _field("track_id", "int64"),
    _field("track_sample_index", "int64"),
    _field("source_acquisition_frame_index", "int64"),
    _field("time_seconds", "float32"),
    _field("source_row_index", "int64"),
    _field("source_instance_key_valid", "bool"),
    _field("source_instance_key", "uint64"),
    _field("detection_source", "int8"),
    _field("position_x_mm", "float32"),
    _field("position_y_mm", "float32"),
    _field("speed_mm_s", "float32"),
    _field("frame_path_distance_mm", "float32"),
    _field("motion_heading_degrees", "float32"),
    _field("smoothed_motion_heading_degrees", "float32"),
    _field("smoothed_angular_velocity_deg_s", "float32"),
    _field("source_observed", "bool"),
    _field("sample_observed", "bool"),
    _field("position_finite", "bool"),
    _field("heading_usable", "bool"),
    _field("sample_valid", "bool"),
    _field("transition_valid", "bool"),
    _field("sample_reason_code", "int16"),
    _field("transition_reason_code", "int16"),
)


# One row per track and global acquisition-frame-aligned time bin. This table
# reports physical position distributions and motion/bout summaries only. It
# deliberately does not claim arena-normalized occupancy without a bound
# experimental-area geometry authority.
_ACTIVITY_SPATIAL_TIME_BIN_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_motion_manifest_schema_id", "string"),
    _field("source_track_motion_manifest_schema_version", "int64"),
    _field("source_track_motion_manifest_sha256", "string"),
    _field("source_track_binding_sha256", "string"),
    _field("source_swim_bout_run", "string"),
    _field("source_swim_bout_path", "string"),
    _field("source_swim_bout_schema_id", "string"),
    _field("source_swim_bout_schema_version", "int64"),
    _field("source_swim_bout_manifest_sha256", "string"),
    _field("source_swim_bout_binding_sha256", "string"),
    _field("source_swim_bout_candidate_id", "int32"),
    _field("source_swim_bout_signal_id", "int32"),
    _field("source_speed_level", "string"),
    _field("source_sample_rate_hz", "float64"),
    _field("requested_bin_size_s", "float64"),
    _field("bin_size_frames", "int64"),
    _field("effective_bin_size_s", "float64"),
    _field("binning_policy", "string"),
    _field("position_coordinate_space", "string"),
    _field("position_coordinate_descriptor_sha256", "string"),
    _field("physical_authority_sha256", "string"),
    _field("track_id", "int64"),
    _field("time_bin_index", "int64"),
    _field("start_acquisition_frame_index", "int64"),
    _field("end_acquisition_frame_index_exclusive", "int64"),
    _field("start_time_seconds", "float64"),
    _field("end_time_seconds", "float64"),
    _field("bin_duration_seconds", "float64"),
    _field("expected_track_frame_count", "int64"),
    _field("source_sample_count", "int64"),
    _field("source_observed_count", "int64"),
    _field("source_observed_fraction", "float64"),
    _field("sample_valid_count", "int64"),
    _field("sample_valid_fraction", "float64"),
    _field("position_valid_count", "int64"),
    _field("position_valid_fraction", "float64"),
    _field("transition_valid_count", "int64"),
    _field("transition_valid_fraction", "float64"),
    _field("mean_position_x_mm", "float64"),
    _field("mean_position_y_mm", "float64"),
    _field("std_position_x_mm", "float64"),
    _field("std_position_y_mm", "float64"),
    _field("covariance_xy_mm2", "float64"),
    _field("min_position_x_mm", "float64"),
    _field("max_position_x_mm", "float64"),
    _field("min_position_y_mm", "float64"),
    _field("max_position_y_mm", "float64"),
    _field("net_displacement_mm", "float64"),
    _field("mean_speed_mm_s", "float64"),
    _field("median_speed_mm_s", "float64"),
    _field("p95_speed_mm_s", "float64"),
    _field("path_distance_mm_sum", "float64"),
    _field("bout_count_started", "int64"),
    _field("bout_duration_s_started_sum", "float64"),
    _field("bout_path_length_mm_started_sum", "float64"),
    _field("bout_occupied_frame_count", "int64"),
    _field("bout_occupancy_fraction", "float64"),
    _field("position_metrics_valid", "bool"),
    _field("speed_metrics_valid", "bool"),
    _field("bout_metrics_valid", "bool"),
    _field("bin_valid", "bool"),
    _field("bin_reason_code", "int16"),
)


# One row per camera frame from the exact compact-v7 frame axis.  Floating
# values deliberately remain float32 so the query product preserves the
# recording-local authority's decoded representation instead of silently
# widening it.  Invalid scientific values remain IEEE NaN under the source
# contract; they are not Arrow nulls.
_EYE_TRACE_SAMPLES_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("source_eye_angle_run", "string"),
    _field("source_eye_angle_path", "string"),
    _field("source_eye_angle_schema_id", "string"),
    _field("source_eye_angle_schema_version", "int64"),
    _field("source_eye_angle_layout", "string"),
    _field("source_eye_angle_method", "string", nullable=True),
    _field("source_eye_angle_method_version", "string", nullable=True),
    _field("source_binding_sha256", "string"),
    _field("projection_contract_sha256", "string"),
    _field("source_acquisition_frame_index", "int64"),
    _field("time_seconds", "float32"),
    _field("left_eye_angle_deg", "float32"),
    _field("right_eye_angle_deg", "float32"),
    _field("vergence_eye_angle_deg", "float32"),
    _field("left_eye_angle_deg_smoothed", "float32"),
    _field("right_eye_angle_deg_smoothed", "float32"),
    _field("vergence_eye_angle_deg_smoothed", "float32"),
    _field("left_gaze_signed_deg", "float32"),
    _field("right_gaze_signed_deg", "float32"),
    _field("left_gaze_signed_deg_smoothed", "float32"),
    _field("right_gaze_signed_deg_smoothed", "float32"),
    _field("mean_eye_vergence_gaze_deg", "float32"),
    _field("mean_eye_vergence_gaze_deg_smoothed", "float32"),
    _field("valid_frame", "bool"),
    _field("major_axis_marginal", "bool"),
    _field("reason_codes", "uint16"),
)


# One row per observation and normalized tail-axis sample. Long form is
# deliberate: the source sample cardinality is run-specific, while the exact
# normalized ``s`` coordinate makes cross-run projections and predicate
# pushdown explicit without embedding variable-length lists in each frame row.
_TAIL_TRACE_SAMPLES_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("recording_id", "string"),
    _field("zarr_path", "string"),
    _field("source_lineage_hash", "string"),
    _field("source_tail_kinematics_run", "string"),
    _field("source_tail_kinematics_path", "string"),
    _field("source_tail_kinematics_schema_id", "string"),
    _field("source_tail_kinematics_schema_version", "int64"),
    _field("source_tail_publication_manifest_sha256", "string"),
    _field("source_subject_shape_run", "string"),
    _field("source_subject_shape_path", "string"),
    _field("source_subject_shape_schema_id", "string"),
    _field("source_subject_shape_schema_version", "int64"),
    _field("source_subject_shape_publication_manifest_sha256", "string"),
    _field("source_track_kinematics_scope", "string"),
    _field("source_track_kinematics_run", "string"),
    _field("source_track_kinematics_path", "string"),
    _field("source_track_motion_manifest_sha256", "string"),
    _field("source_binding_sha256", "string"),
    _field("projection_contract_sha256", "string"),
    _field("source_sample_rate_hz", "float64"),
    _field("source_tail_sample_count", "int32"),
    _field("source_tail_sample_axis_sha256", "string"),
    _field("body_frame_record_sha256", "string"),
    _field("reference_length_kind", "string"),
    _field("longitudinal_axis_convention", "string"),
    _field("lateral_axis_convention", "string"),
    _field("angle_convention", "string"),
    _field("curvature_convention", "string"),
    _field("source_tail_row_index", "int64"),
    _field("track_id", "int64"),
    _field("instance_key", "uint64"),
    _field("source_crop_row_id", "int64"),
    _field("source_acquisition_frame_index", "int64"),
    _field("time_seconds", "float64"),
    _field("tail_sample_index", "int32"),
    _field("normalized_tail_position", "float32"),
    _field("reference_length_px", "float32"),
    _field("body_longitudinal_fraction", "float32"),
    _field("body_lateral_fraction", "float32"),
    _field("tangent_angle_rad", "float32"),
    _field("body_curvature_dimensionless", "float32"),
    _field("source_camera_x_px", "float32"),
    _field("source_camera_y_px", "float32"),
    _field("source_camera_curvature_px_inv", "float32"),
    _field("source_lateral_deflection_px", "float32"),
    _field("source_tail_row_valid", "bool"),
    _field("reference_length_valid", "bool"),
    _field("sample_valid", "bool"),
    _field("sample_reason_code", "uint16"),
    _field("source_failure_reason", "string"),
)


# Group statistics are published by one closed producer.  Statistical values
# are nullable because insufficient complete recordings, disabled bootstrap
# work, or an unavailable test legitimately produce nulls; status and
# skip_reason explain that state.  Identity, source binding, grouping, method,
# and iteration counts are always present.  Timestamps remain strings in
# physical v1 to preserve the producer's current RFC-3339 representation.
_GROUP_STATISTICS_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("stat_result_id", "string"),
    _field("stats_run_id", "string"),
    _field("source_export_run_id", "string"),
    _field("source_export_manifest_path", "string"),
    _field("source_export_manifest_sha256", "string"),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("source_table", "string"),
    _field("source_row_count", "int64"),
    _field("metric_family", "string"),
    _field("metric_name", "string"),
    _field("metric_unit", "string"),
    _field("contrast_name", "string"),
    _field("condition_a", "string"),
    _field("condition_b", "string"),
    _field("group_key_json", "string"),
    _field("primary", "bool"),
    _field("exploratory", "bool"),
    _field("unit", "string"),
    _field("unit_count", "int64"),
    _field("paired_unit_count", "int64"),
    _field("excluded_unit_count", "int64"),
    _field("missing_policy", "string"),
    _field("mean_a", "float64", nullable=True),
    _field("mean_b", "float64", nullable=True),
    _field("mean_difference", "float64", nullable=True),
    _field("median_difference", "float64", nullable=True),
    _field("std_difference", "float64", nullable=True),
    _field("effect_size", "float64", nullable=True),
    _field("effect_size_kind", "string"),
    _field("ci_estimand", "string"),
    _field("ci_low", "float64", nullable=True),
    _field("ci_high", "float64", nullable=True),
    _field("p_value", "float64", nullable=True),
    _field("q_value", "float64", nullable=True),
    _field("multiple_comparison_family", "string"),
    _field("test_method", "string"),
    _field("bootstrap_iterations", "int64"),
    _field("permutation_iterations", "int64"),
    _field("status", "string"),
    _field("skip_reason", "string", nullable=True),
    _field("parameters_json", "string"),
    _field("created_at_utc", "string"),
)


# Descriptive rows use the same immutable source and grouping identity but do
# not carry a contrast.  All seven descriptive statistics are nullable: an
# empty finite-value set has no numerical summary, and sample standard
# deviation/SEM are unavailable for a singleton.
_GROUP_DESCRIPTIVE_FIELDS = (
    _field("export_schema_version", "int32"),
    _field("table_name", "string"),
    _field("descriptive_result_id", "string"),
    _field("stats_run_id", "string"),
    _field("source_export_run_id", "string"),
    _field("source_export_manifest_path", "string"),
    _field("source_export_manifest_sha256", "string"),
    _field("collection_id", "string", nullable=True),
    _field("collection_manifest_sha256", "string", nullable=True),
    _field("source_table", "string"),
    _field("source_row_count", "int64"),
    _field("metric_family", "string"),
    _field("metric_name", "string"),
    _field("metric_unit", "string"),
    _field("condition_name", "string"),
    _field("group_key_json", "string"),
    _field("primary", "bool"),
    _field("exploratory", "bool"),
    _field("unit", "string"),
    _field("unit_count", "int64"),
    _field("sum", "float64", nullable=True),
    _field("mean", "float64", nullable=True),
    _field("median", "float64", nullable=True),
    _field("std_dev", "float64", nullable=True),
    _field("sem", "float64", nullable=True),
    _field("min", "float64", nullable=True),
    _field("max", "float64", nullable=True),
    _field("missing_policy", "string"),
    _field("parameters_json", "string"),
    _field("created_at_utc", "string"),
)


ARROW_TABLE_CONTRACTS: dict[str, ArrowTableContract] = {
    POSITION_OCCUPANCY_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        fields=_POSITION_OCCUPANCY_FIELDS,
    ),
    RECORDING_SUMMARY_TABLE: ArrowTableContract(
        table_name=RECORDING_SUMMARY_TABLE,
        fields=_RECORDING_SUMMARY_FIELDS,
    ),
    STIMULUS_STEPS_TABLE: ArrowTableContract(
        table_name=STIMULUS_STEPS_TABLE,
        fields=_STIMULUS_STEPS_FIELDS,
    ),
    STIMULUS_STEP_SUMMARY_TABLE: ArrowTableContract(
        table_name=STIMULUS_STEP_SUMMARY_TABLE,
        fields=_STIMULUS_STEP_SUMMARY_FIELDS,
    ),
    STIMULUS_RESPONSE_TABLE: ArrowTableContract(
        table_name=STIMULUS_RESPONSE_TABLE,
        fields=_STIMULUS_RESPONSE_FIELDS,
    ),
    SWIM_BOUT_METRICS_TABLE: ArrowTableContract(
        table_name=SWIM_BOUT_METRICS_TABLE,
        fields=_SWIM_BOUT_METRICS_FIELDS,
    ),
    BOUT_KINEMATICS_METRICS_TABLE: ArrowTableContract(
        table_name=BOUT_KINEMATICS_METRICS_TABLE,
        fields=_BOUT_KINEMATICS_METRICS_FIELDS,
    ),
    BASELINE_BEHAVIOR_SUMMARY_TABLE: ArrowTableContract(
        table_name=BASELINE_BEHAVIOR_SUMMARY_TABLE,
        fields=_BASELINE_BEHAVIOR_SUMMARY_FIELDS,
    ),
    BASELINE_BEHAVIOR_TIME_BINS_TABLE: ArrowTableContract(
        table_name=BASELINE_BEHAVIOR_TIME_BINS_TABLE,
        fields=_BASELINE_BEHAVIOR_TIME_BINS_FIELDS,
    ),
    BASELINE_KINEMATIC_SAMPLES_TABLE: ArrowTableContract(
        table_name=BASELINE_KINEMATIC_SAMPLES_TABLE,
        fields=_BASELINE_KINEMATIC_SAMPLES_FIELDS,
    ),
    KINEMATICS_SAMPLES_TABLE: ArrowTableContract(
        table_name=KINEMATICS_SAMPLES_TABLE,
        fields=_KINEMATICS_SAMPLES_FIELDS,
    ),
    ACTIVITY_SPATIAL_TIME_BINS_TABLE: ArrowTableContract(
        table_name=ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        fields=_ACTIVITY_SPATIAL_TIME_BIN_FIELDS,
    ),
    STATISTICS_TABLE: ArrowTableContract(
        table_name=STATISTICS_TABLE,
        fields=_GROUP_STATISTICS_FIELDS,
    ),
    DESCRIPTIVE_TABLE: ArrowTableContract(
        table_name=DESCRIPTIVE_TABLE,
        fields=_GROUP_DESCRIPTIVE_FIELDS,
    ),
    EYE_TRACE_SAMPLES_TABLE: ArrowTableContract(
        table_name=EYE_TRACE_SAMPLES_TABLE,
        fields=_EYE_TRACE_SAMPLES_FIELDS,
    ),
    TAIL_TRACE_SAMPLES_TABLE: ArrowTableContract(
        table_name=TAIL_TRACE_SAMPLES_TABLE,
        fields=_TAIL_TRACE_SAMPLES_FIELDS,
    ),
    CHASER_SPATIAL_TABLE: ArrowTableContract(
        table_name=CHASER_SPATIAL_TABLE,
        fields=_CHASER_SPATIAL_FIELDS,
    ),
    CHASER_DISTANCE_SUMMARY_TABLE: ArrowTableContract(
        table_name=CHASER_DISTANCE_SUMMARY_TABLE,
        fields=_CHASER_DISTANCE_SUMMARY_FIELDS,
    ),
    CHASER_EPOCH_BEHAVIOR_TABLE: ArrowTableContract(
        table_name=CHASER_EPOCH_BEHAVIOR_TABLE,
        fields=_CHASER_EPOCH_BEHAVIOR_FIELDS,
    ),
    CHASER_BOUT_EVENTS_TABLE: ArrowTableContract(
        table_name=CHASER_BOUT_EVENTS_TABLE,
        fields=_CHASER_BOUT_EVENT_FIELDS,
    ),
    CHASER_BOUT_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=CHASER_BOUT_HISTOGRAM_TABLE,
        fields=_CHASER_EPOCH_HISTOGRAM_FIELDS,
    ),
    CHASER_IBI_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=CHASER_IBI_HISTOGRAM_TABLE,
        fields=_CHASER_EPOCH_HISTOGRAM_FIELDS,
    ),
    CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE,
        fields=_CHASER_CENTER_DISTANCE_HISTOGRAM_FIELDS,
    ),
    CHASER_SPEED_DISTANCE_TABLE: ArrowTableContract(
        table_name=CHASER_SPEED_DISTANCE_TABLE,
        fields=_CHASER_SPEED_DISTANCE_FIELDS,
    ),
    CHASER_DISTANCE_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=CHASER_DISTANCE_HISTOGRAM_TABLE,
        fields=_CHASER_DISTANCE_HISTOGRAM_FIELDS,
    ),
    CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE: ArrowTableContract(
        table_name=CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE,
        fields=_CHASER_QUADRANT_SUMMARY_FIELDS,
    ),
    CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE: ArrowTableContract(
        table_name=CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE,
        fields=_CHASER_QUADRANT_PHASE_FIELDS,
    ),
    CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE: ArrowTableContract(
        table_name=CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE,
        fields=_CHASER_QUADRANT_DENSITY_FIELDS,
    ),
    CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE: ArrowTableContract(
        table_name=CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE,
        fields=_CHASER_NEAR_FIELD_SUMMARY_FIELDS,
    ),
    CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE: ArrowTableContract(
        table_name=CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE,
        fields=_CHASER_NEAR_FIELD_PHASE_FIELDS,
    ),
    CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE: ArrowTableContract(
        table_name=CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE,
        fields=_CHASER_NEAR_FIELD_RADIAL_FIELDS,
    ),
    CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE: ArrowTableContract(
        table_name=CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE,
        fields=_CHASER_NEAR_FIELD_CDF_FIELDS,
    ),
    CHASER_EGOCENTRIC_SUMMARY_TABLE: ArrowTableContract(
        table_name=CHASER_EGOCENTRIC_SUMMARY_TABLE,
        fields=_CHASER_EGOCENTRIC_SUMMARY_FIELDS,
    ),
    CHASER_EGOCENTRIC_HISTOGRAM_TABLE: ArrowTableContract(
        table_name=CHASER_EGOCENTRIC_HISTOGRAM_TABLE,
        fields=_CHASER_EGOCENTRIC_HISTOGRAM_FIELDS,
    ),
}


def arrow_contract_envelope(table_names: Sequence[str]) -> dict[str, object]:
    """Build the closed exact/inferred partition for one export manifest."""

    names = tuple(table_names)
    if len(set(names)) != len(names):
        raise ValueError("Arrow contract table names must be unique")
    unknown = sorted(set(names) - set(TABLE_CONTRACTS))
    if unknown:
        raise ValueError(f"Unknown analytics tables in Arrow contract: {unknown}")
    exact = {
        name: ARROW_TABLE_CONTRACTS[name].to_dict()
        for name in names
        if name in ARROW_TABLE_CONTRACTS
    }
    inferred = sorted(set(names) - set(exact))
    payload: dict[str, object] = {
        "schema_id": ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        "schema_version": ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        "exact_tables": exact,
        "inferred_v2_compatibility_tables": inferred,
    }
    return {**payload, "payload_sha256": _sha256(payload)}


def validate_arrow_contract_envelope(
    value: object,
    table_names: Sequence[str],
) -> dict[str, object]:
    """Validate an envelope against installed contracts, not merely its hash."""

    if not isinstance(value, Mapping) or set(value) != _ENVELOPE_FIELDS:
        raise ValueError("Arrow contract envelope has an unexpected field set")
    if value.get("schema_id") != ARROW_CONTRACT_ENVELOPE_SCHEMA_ID:
        raise ValueError("Arrow contract envelope schema ID is invalid")
    if type(value.get("schema_version")) is not int or value.get(
        "schema_version"
    ) != ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("Arrow contract envelope schema version is invalid")
    payload = {key: value[key] for key in _ENVELOPE_FIELDS - {"payload_sha256"}}
    if value.get("payload_sha256") != _sha256(payload):
        raise ValueError("Arrow contract envelope payload digest is invalid")
    expected = arrow_contract_envelope(table_names)
    if dict(value) != expected:
        raise ValueError("Arrow contract envelope differs from installed contracts")
    exact = value.get("exact_tables")
    assert isinstance(exact, Mapping)
    for table_name, raw_contract in exact.items():
        if not isinstance(raw_contract, Mapping) or set(raw_contract) != _TABLE_FIELDS:
            raise ValueError(f"{table_name}: Arrow table contract field set is invalid")
        raw_fields = raw_contract.get("fields")
        if not isinstance(raw_fields, list) or any(
            not isinstance(item, Mapping) or set(item) != _FIELD_FIELDS
            for item in raw_fields
        ):
            raise ValueError(f"{table_name}: Arrow field declarations are invalid")
    return expected


def _arrow_type(type_id: str) -> Any:
    import pyarrow as pa

    types = {
        "bool": pa.bool_(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "list<string>": pa.list_(pa.string()),
        "string": pa.string(),
        "uint16": pa.uint16(),
        "uint64": pa.uint64(),
    }
    try:
        return types[type_id]
    except KeyError as exc:  # pragma: no cover - installed declarations are tested.
        raise ValueError(f"Unsupported Arrow contract type: {type_id}") from exc


def exact_arrow_schema(table_name: str, *, metadata: Mapping[bytes, bytes]) -> Any:
    """Return the installed exact PyArrow schema with contract metadata."""

    import pyarrow as pa

    contract = ARROW_TABLE_CONTRACTS[table_name]
    fields = [
        pa.field(field.name, _arrow_type(field.arrow_type), nullable=field.nullable)
        for field in contract.fields
    ]
    contract_metadata = {
        **metadata,
        b"palette.arrow_schema_mode": b"exact",
        b"palette.arrow_schema_id": contract.schema_id.encode("utf-8"),
        b"palette.arrow_schema_version": str(contract.schema_version).encode("ascii"),
        b"palette.arrow_schema_sha256": contract.payload_sha256.encode("ascii"),
    }
    return pa.schema(fields, metadata=contract_metadata)


def validate_arrow_schema(table_name: str, schema: Any) -> None:
    """Validate one physical schema against its exact or compatibility mode."""

    metadata = schema.metadata or {}
    contract = ARROW_TABLE_CONTRACTS.get(table_name)
    if contract is None:
        if metadata.get(b"palette.arrow_schema_mode") != b"inferred_v2_compatibility":
            raise ValueError(f"{table_name}: Arrow schema compatibility mode is missing")
        return
    expected = exact_arrow_schema(table_name, metadata={})
    if schema.remove_metadata() != expected.remove_metadata():
        raise ValueError(f"{table_name}: physical Arrow fields differ from the exact contract")
    expected_metadata = expected.metadata or {}
    for key in (
        b"palette.arrow_schema_mode",
        b"palette.arrow_schema_id",
        b"palette.arrow_schema_version",
        b"palette.arrow_schema_sha256",
    ):
        if metadata.get(key) != expected_metadata[key]:
            raise ValueError(f"{table_name}: Arrow footer contract metadata is invalid")


__all__ = [
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_ID",
    "ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION",
    "ARROW_TABLE_CONTRACTS",
    "EXACT_ARROW_SCHEMA_TABLES",
    "ArrowFieldContract",
    "ArrowTableContract",
    "arrow_contract_envelope",
    "exact_arrow_schema",
    "validate_arrow_contract_envelope",
    "validate_arrow_schema",
]
