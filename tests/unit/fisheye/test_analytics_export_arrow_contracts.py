from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    EXACT_ARROW_SCHEMA_TABLES,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_contract_envelope,
    validate_arrow_schema,
)
from fisheye.analytics_exports.contracts import (
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
    EXPORT_SCHEMA_VERSION,
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
from fisheye.analytics_exports.publication import (
    manifest_selected_part_files,
    sha256_file,
)
from fisheye.analytics_exports.validation import ExportValidationError, validate_export_run
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.utils.export_cross_recording_analytics import (
    SourceExportResult,
    _write_table_parts,
    export_sources,
)
from tests.unit.fisheye.test_goodcopbadcop_interactive import (
    _make_archive_with_detection_occupancy,
)
from tests.unit.fisheye.test_export_cross_recording_analytics import (
    _make_source_zarr,
    _write_collection_manifest,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rehash(envelope: dict[str, Any]) -> None:
    for contract in envelope["exact_tables"].values():
        contract["payload_sha256"] = _canonical_sha256(
            {key: value for key, value in contract.items() if key != "payload_sha256"}
        )
    envelope["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in envelope.items() if key != "payload_sha256"}
    )


def _valid_position_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[POSITION_OCCUPANCY_HISTOGRAM_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        elif field.arrow_type == "bool":
            row[field.name] = True
        elif field.arrow_type == "list<string>":
            row[field.name] = ["window", "y_bin", "x_bin"]
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            "recording_id": "recording-1",
            "position_occupancy_path": "analysis/detection_occupancy_runs/run-1",
            "source_refs_json": "{}",
        }
    )
    return row


def _valid_recording_summary_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[RECORDING_SUMMARY_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": RECORDING_SUMMARY_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "a" * 64,
            "stimulus_step_count": 0,
        }
    )
    return row


def _valid_core_analytics_row(table_name: str) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in ARROW_TABLE_CONTRACTS[table_name].fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        elif field.arrow_type == "bool":
            row[field.name] = True
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": table_name,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "f" * 64,
        }
    )
    return row


CHASER_ARROW_TABLES = (
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


def _valid_chaser_row(table_name: str) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in ARROW_TABLE_CONTRACTS[table_name].fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        elif field.arrow_type == "bool":
            row[field.name] = True
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": table_name,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "7" * 64,
        }
    )
    return row


def _valid_stimulus_step_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[STIMULUS_STEPS_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": STIMULUS_STEPS_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "e" * 64,
            "stimulus_run": "stimulus-1",
            "step_index": 0,
            "step_group": "step_0",
            "protocol_signature_schema": "palette_protocol_signature_v1",
            "protocol_signature_hash": "f" * 64,
            "derived_protocol_hash": "f" * 64,
            "protocol_step_count": 1,
        }
    )
    return row


def _valid_stimulus_step_summary_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": STIMULUS_STEP_SUMMARY_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "9" * 64,
            "stimulus_response_run": "response-1",
            "source_stimulus_run": "stimulus-1",
            "source_track_kinematics_run": "kinematics-1",
            "source_track_kinematics_type": "offline",
            "step_index": 0,
            "step_name": "moving",
            "stimulus_mode": "MOVING_GRATING",
            "stimulus_mode_id": 5,
            "start_frame": 10,
            "end_frame": 20,
            "start_camera_frame": 10,
            "end_camera_frame": 20,
            "duration_s": 1.0,
            "fish_id": 7,
        }
    )
    return row


def _valid_baseline_summary_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": BASELINE_BEHAVIOR_SUMMARY_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "b" * 64,
            "source_refs_json": "{}",
        }
    )
    return row


def _valid_baseline_time_bin_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_TIME_BINS_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": BASELINE_BEHAVIOR_TIME_BINS_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "c" * 64,
            "source_refs_json": "{}",
        }
    )
    return row


def _valid_baseline_sample_row() -> dict[str, object]:
    contract = ARROW_TABLE_CONTRACTS[BASELINE_KINEMATIC_SAMPLES_TABLE]
    row: dict[str, object] = {}
    for field in contract.fields:
        if field.nullable:
            row[field.name] = None
        elif field.arrow_type in {"int32", "int64"}:
            row[field.name] = 1
        elif field.arrow_type == "float64":
            row[field.name] = 1.5
        elif field.arrow_type == "bool":
            row[field.name] = True
        else:
            row[field.name] = "value"
    row.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "table_name": BASELINE_KINEMATIC_SAMPLES_TABLE,
            "recording_id": "recording-1",
            "zarr_path": "/recordings/recording-1_analysis.zarr",
            "source_lineage_hash": "d" * 64,
            "source_refs_json": "{}",
        }
    )
    return row


def test_arrow_contract_envelope_partitions_exact_and_compatibility_tables() -> None:
    envelope = arrow_contract_envelope(
        (
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
        )
    )

    assert tuple(envelope["exact_tables"]) == (
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
    )
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert (
        validate_arrow_contract_envelope(
            envelope,
            (
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
            ),
        )
        == envelope
    )


def test_recording_summary_contract_freezes_exact_field_order_and_nullability() -> None:
    assert EXACT_ARROW_SCHEMA_TABLES == (
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
        *CHASER_ARROW_TABLES,
    )
    fields = ARROW_TABLE_CONTRACTS[RECORDING_SUMMARY_TABLE].fields
    assert tuple(field.name for field in fields) == (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_run",
        "stimulus_response_run",
        "swim_bout_run",
        "stimulus_step_count",
        "protocol_signature_schema",
        "protocol_signature_hash",
        "derived_protocol_hash",
        "protocol_mode_sequence",
        "protocol_duration_sequence_s",
        "protocol_step_count",
        "source_track_kinematics_run",
        "source_track_kinematics_type",
        "source_bout_run",
        "n_fish",
        "n_steps",
        "global_fish_count",
        "total_distance_mm_sum",
        "mean_speed_mm_s_mean",
        "fraction_moving_mean",
        "total_active_s_sum",
        "swim_bout_default_level",
        "swim_bout_default_n_bouts",
        "swim_bout_default_mean_duration_s",
        "swim_bout_default_total_path_length_mm",
        "collection_id",
        "collection_manifest_sha256",
        "collection_manifest_path",
    )
    assert len(fields) == 32
    assert {field.name for field in fields if not field.nullable} == {
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_step_count",
    }
    assert next(field for field in fields if field.name == "derived_protocol_hash").nullable


def test_generic_kinematics_samples_contract_freezes_exact_multi_track_schema() -> None:
    fields = ARROW_TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_motion_manifest_schema_id", "string", False),
        ("source_track_motion_manifest_schema_version", "int64", False),
        ("source_track_motion_manifest_sha256", "string", False),
        ("source_binding_sha256", "string", False),
        ("projection_contract_sha256", "string", False),
        ("source_speed_level", "string", False),
        ("source_sample_rate_hz", "float64", False),
        ("requested_sample_rate_hz", "float64", False),
        ("sampling_stride_frames", "int64", False),
        ("nominal_sample_rate_hz", "float64", False),
        ("sampling_policy", "string", False),
        ("position_coordinate_space", "string", False),
        ("position_coordinate_descriptor_sha256", "string", False),
        ("physical_authority_sha256", "string", False),
        ("track_id", "int64", False),
        ("track_sample_index", "int64", False),
        ("source_acquisition_frame_index", "int64", False),
        ("time_seconds", "float32", False),
        ("source_row_index", "int64", False),
        ("source_instance_key_valid", "bool", False),
        ("source_instance_key", "uint64", False),
        ("detection_source", "int8", False),
        ("position_x_mm", "float32", False),
        ("position_y_mm", "float32", False),
        ("speed_mm_s", "float32", False),
        ("frame_path_distance_mm", "float32", False),
        ("motion_heading_degrees", "float32", False),
        ("smoothed_motion_heading_degrees", "float32", False),
        ("smoothed_angular_velocity_deg_s", "float32", False),
        ("source_observed", "bool", False),
        ("sample_observed", "bool", False),
        ("position_finite", "bool", False),
        ("heading_usable", "bool", False),
        ("sample_valid", "bool", False),
        ("transition_valid", "bool", False),
        ("sample_reason_code", "int16", False),
        ("transition_reason_code", "int16", False),
    )
    assert TABLE_CONTRACTS[KINEMATICS_SAMPLES_TABLE].primary_key == (
        "recording_id",
        "source_track_kinematics_scope",
        "source_track_kinematics_run",
        "track_id",
        "source_acquisition_frame_index",
    )
    field_names = {field.name for field in fields}
    assert "baseline_window_id" not in field_names
    assert "chaser_distance_run" not in field_names


def test_activity_spatial_time_bins_freeze_exact_geometry_honest_schema() -> None:
    fields = ARROW_TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_motion_manifest_schema_id", "string", False),
        ("source_track_motion_manifest_schema_version", "int64", False),
        ("source_track_motion_manifest_sha256", "string", False),
        ("source_track_binding_sha256", "string", False),
        ("source_swim_bout_run", "string", False),
        ("source_swim_bout_path", "string", False),
        ("source_swim_bout_schema_id", "string", False),
        ("source_swim_bout_schema_version", "int64", False),
        ("source_swim_bout_manifest_sha256", "string", False),
        ("source_swim_bout_binding_sha256", "string", False),
        ("source_swim_bout_candidate_id", "int32", False),
        ("source_swim_bout_signal_id", "int32", False),
        ("source_speed_level", "string", False),
        ("source_sample_rate_hz", "float64", False),
        ("requested_bin_size_s", "float64", False),
        ("bin_size_frames", "int64", False),
        ("effective_bin_size_s", "float64", False),
        ("binning_policy", "string", False),
        ("position_coordinate_space", "string", False),
        ("position_coordinate_descriptor_sha256", "string", False),
        ("physical_authority_sha256", "string", False),
        ("track_id", "int64", False),
        ("time_bin_index", "int64", False),
        ("start_acquisition_frame_index", "int64", False),
        ("end_acquisition_frame_index_exclusive", "int64", False),
        ("start_time_seconds", "float64", False),
        ("end_time_seconds", "float64", False),
        ("bin_duration_seconds", "float64", False),
        ("expected_track_frame_count", "int64", False),
        ("source_sample_count", "int64", False),
        ("source_observed_count", "int64", False),
        ("source_observed_fraction", "float64", False),
        ("sample_valid_count", "int64", False),
        ("sample_valid_fraction", "float64", False),
        ("position_valid_count", "int64", False),
        ("position_valid_fraction", "float64", False),
        ("transition_valid_count", "int64", False),
        ("transition_valid_fraction", "float64", False),
        ("mean_position_x_mm", "float64", False),
        ("mean_position_y_mm", "float64", False),
        ("std_position_x_mm", "float64", False),
        ("std_position_y_mm", "float64", False),
        ("covariance_xy_mm2", "float64", False),
        ("min_position_x_mm", "float64", False),
        ("max_position_x_mm", "float64", False),
        ("min_position_y_mm", "float64", False),
        ("max_position_y_mm", "float64", False),
        ("net_displacement_mm", "float64", False),
        ("mean_speed_mm_s", "float64", False),
        ("median_speed_mm_s", "float64", False),
        ("p95_speed_mm_s", "float64", False),
        ("path_distance_mm_sum", "float64", False),
        ("bout_count_started", "int64", False),
        ("bout_duration_s_started_sum", "float64", False),
        ("bout_path_length_mm_started_sum", "float64", False),
        ("bout_occupied_frame_count", "int64", False),
        ("bout_occupancy_fraction", "float64", False),
        ("position_metrics_valid", "bool", False),
        ("speed_metrics_valid", "bool", False),
        ("bout_metrics_valid", "bool", False),
        ("bin_valid", "bool", False),
        ("bin_reason_code", "int16", False),
    )
    contract = TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE]
    assert contract.primary_key == (
        "recording_id",
        "source_track_kinematics_scope",
        "source_track_kinematics_run",
        "source_swim_bout_run",
        "track_id",
        "time_bin_index",
    )
    names = {field.name for field in fields}
    assert "wall_fraction" not in names
    assert "arena_radius_mm" not in names
    assert "occupancy_grid" not in names


def test_tail_trace_samples_freeze_exact_long_form_body_frame_schema() -> None:
    fields = ARROW_TABLE_CONTRACTS[TAIL_TRACE_SAMPLES_TABLE].fields
    assert len(fields) == 52
    assert tuple((field.name, field.arrow_type, field.nullable) for field in fields) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("source_tail_kinematics_run", "string", False),
        ("source_tail_kinematics_path", "string", False),
        ("source_tail_kinematics_schema_id", "string", False),
        ("source_tail_kinematics_schema_version", "int64", False),
        ("source_tail_publication_manifest_sha256", "string", False),
        ("source_subject_shape_run", "string", False),
        ("source_subject_shape_path", "string", False),
        ("source_subject_shape_schema_id", "string", False),
        ("source_subject_shape_schema_version", "int64", False),
        ("source_subject_shape_publication_manifest_sha256", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_motion_manifest_sha256", "string", False),
        ("source_binding_sha256", "string", False),
        ("projection_contract_sha256", "string", False),
        ("source_sample_rate_hz", "float64", False),
        ("source_tail_sample_count", "int32", False),
        ("source_tail_sample_axis_sha256", "string", False),
        ("body_frame_record_sha256", "string", False),
        ("reference_length_kind", "string", False),
        ("longitudinal_axis_convention", "string", False),
        ("lateral_axis_convention", "string", False),
        ("angle_convention", "string", False),
        ("curvature_convention", "string", False),
        ("source_tail_row_index", "int64", False),
        ("track_id", "int64", False),
        ("instance_key", "uint64", False),
        ("source_crop_row_id", "int64", False),
        ("source_acquisition_frame_index", "int64", False),
        ("time_seconds", "float64", False),
        ("tail_sample_index", "int32", False),
        ("normalized_tail_position", "float32", False),
        ("reference_length_px", "float32", False),
        ("body_longitudinal_fraction", "float32", False),
        ("body_lateral_fraction", "float32", False),
        ("tangent_angle_rad", "float32", False),
        ("body_curvature_dimensionless", "float32", False),
        ("source_camera_x_px", "float32", False),
        ("source_camera_y_px", "float32", False),
        ("source_camera_curvature_px_inv", "float32", False),
        ("source_lateral_deflection_px", "float32", False),
        ("source_tail_row_valid", "bool", False),
        ("reference_length_valid", "bool", False),
        ("sample_valid", "bool", False),
        ("sample_reason_code", "uint16", False),
        ("source_failure_reason", "string", False),
    )
    assert TABLE_CONTRACTS[TAIL_TRACE_SAMPLES_TABLE].primary_key == (
        "recording_id",
        "source_tail_kinematics_run",
        "source_tail_row_index",
        "tail_sample_index",
    )
    names = {field.name for field in fields}
    assert "camera_space_points" not in names
    assert "variable_length_tail_samples" not in names


def test_group_statistics_contract_freezes_all_45_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[STATISTICS_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("stat_result_id", "string", False),
        ("stats_run_id", "string", False),
        ("source_export_run_id", "string", False),
        ("source_export_manifest_path", "string", False),
        ("source_export_manifest_sha256", "string", False),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("source_table", "string", False),
        ("source_row_count", "int64", False),
        ("metric_family", "string", False),
        ("metric_name", "string", False),
        ("metric_unit", "string", False),
        ("contrast_name", "string", False),
        ("condition_a", "string", False),
        ("condition_b", "string", False),
        ("group_key_json", "string", False),
        ("primary", "bool", False),
        ("exploratory", "bool", False),
        ("unit", "string", False),
        ("unit_count", "int64", False),
        ("paired_unit_count", "int64", False),
        ("excluded_unit_count", "int64", False),
        ("missing_policy", "string", False),
        ("mean_a", "float64", True),
        ("mean_b", "float64", True),
        ("mean_difference", "float64", True),
        ("median_difference", "float64", True),
        ("std_difference", "float64", True),
        ("effect_size", "float64", True),
        ("effect_size_kind", "string", False),
        ("ci_estimand", "string", False),
        ("ci_low", "float64", True),
        ("ci_high", "float64", True),
        ("p_value", "float64", True),
        ("q_value", "float64", True),
        ("multiple_comparison_family", "string", False),
        ("test_method", "string", False),
        ("bootstrap_iterations", "int64", False),
        ("permutation_iterations", "int64", False),
        ("status", "string", False),
        ("skip_reason", "string", True),
        ("parameters_json", "string", False),
        ("created_at_utc", "string", False),
    )


def test_group_descriptive_contract_freezes_all_30_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[DESCRIPTIVE_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("descriptive_result_id", "string", False),
        ("stats_run_id", "string", False),
        ("source_export_run_id", "string", False),
        ("source_export_manifest_path", "string", False),
        ("source_export_manifest_sha256", "string", False),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("source_table", "string", False),
        ("source_row_count", "int64", False),
        ("metric_family", "string", False),
        ("metric_name", "string", False),
        ("metric_unit", "string", False),
        ("condition_name", "string", False),
        ("group_key_json", "string", False),
        ("primary", "bool", False),
        ("exploratory", "bool", False),
        ("unit", "string", False),
        ("unit_count", "int64", False),
        ("sum", "float64", True),
        ("mean", "float64", True),
        ("median", "float64", True),
        ("std_dev", "float64", True),
        ("sem", "float64", True),
        ("min", "float64", True),
        ("max", "float64", True),
        ("missing_policy", "string", False),
        ("parameters_json", "string", False),
        ("created_at_utc", "string", False),
    )


def test_stimulus_steps_contract_freezes_all_60_maintained_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[STIMULUS_STEPS_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("stimulus_run", "string", False),
        ("step_index", "int64", False),
        ("step_group", "string", False),
        ("step_name", "string", True),
        ("stimulus_mode", "string", True),
        ("stimulus_mode_id", "int64", True),
        ("start_frame", "int64", True),
        ("end_frame", "int64", True),
        ("start_camera_frame", "int64", True),
        ("end_camera_frame", "int64", True),
        ("duration_s", "float64", True),
        ("stimulus_params_json", "string", True),
        ("moving_grating_metadata_schema_version", "int64", True),
        ("moving_grating_source", "string", True),
        ("moving_grating_orientation_degrees_authored", "float64", True),
        ("moving_grating_grating_direction_camera_deg", "float64", True),
        ("moving_grating_camera_to_projector_offset_deg", "float64", True),
        ("moving_grating_direction_mapping_source", "string", True),
        ("moving_grating_direction_mapping_status", "string", True),
        ("moving_grating_direction_mapping_validated", "bool", True),
        ("moving_grating_speed_mm_s", "float64", True),
        ("moving_grating_speed_pps", "float64", True),
        ("moving_grating_spatial_freq_cycles_per_mm", "float64", True),
        ("moving_grating_spatial_freq_rpp", "float64", True),
        ("moving_grating_temporal_frequency_hz", "float64", True),
        (
            "moving_grating_actual_rendered_temporal_frequency_hz",
            "float64",
            True,
        ),
        ("moving_grating_duty_cycle", "float64", True),
        ("concentric_grating_metadata_schema_version", "int64", True),
        ("concentric_grating_source", "string", True),
        ("concentric_grating_stimulus_role", "string", True),
        ("concentric_grating_radial_polarity_authored", "string", True),
        ("concentric_grating_radial_sign_authored", "int64", True),
        ("concentric_grating_radial_polarity_source", "string", True),
        ("concentric_grating_radial_polarity_validated", "bool", True),
        ("concentric_grating_speed_mm_s", "float64", True),
        ("concentric_grating_speed_pps", "float64", True),
        ("concentric_grating_spatial_freq_cycles_per_mm", "float64", True),
        ("concentric_grating_spatial_freq_rpp", "float64", True),
        ("concentric_grating_temporal_frequency_hz", "float64", True),
        (
            "concentric_grating_actual_rendered_temporal_frequency_hz",
            "float64",
            True,
        ),
        ("concentric_grating_duty_cycle", "float64", True),
        ("concentric_grating_target_radius_min_mm", "float64", True),
        ("concentric_grating_target_radius_max_mm", "float64", True),
        ("concentric_grating_target_radius_source", "string", True),
        (
            "concentric_grating_centering_success_fraction_threshold",
            "float64",
            True,
        ),
        ("concentric_grating_coordinate_geometry_status", "string", True),
        ("protocol_signature_schema", "string", False),
        ("protocol_signature_hash", "string", False),
        ("derived_protocol_hash", "string", False),
        ("protocol_mode_sequence", "string", True),
        ("protocol_duration_sequence_s", "string", True),
        ("protocol_step_count", "int64", False),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


def test_stimulus_step_summary_contract_freezes_all_38_maintained_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("stimulus_response_run", "string", False),
        ("source_stimulus_run", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_type", "string", False),
        ("source_bout_run", "string", True),
        ("step_index", "int64", False),
        ("step_name", "string", False),
        ("stimulus_mode", "string", False),
        ("stimulus_mode_id", "int64", False),
        ("start_frame", "int64", False),
        ("end_frame", "int64", False),
        ("start_camera_frame", "int64", False),
        ("end_camera_frame", "int64", False),
        ("duration_s", "float64", False),
        ("protocol_signature_schema", "string", True),
        ("protocol_signature_hash", "string", True),
        ("derived_protocol_hash", "string", True),
        ("protocol_mode_sequence", "string", True),
        ("protocol_duration_sequence_s", "string", True),
        ("protocol_step_count", "int64", True),
        ("fish_id", "int64", False),
        ("total_distance_mm", "float64", True),
        ("mean_speed_mm_s", "float64", True),
        ("median_speed_mm_s", "float64", True),
        ("max_speed_mm_s", "float64", True),
        ("fraction_moving", "float64", True),
        ("coverage", "float64", True),
        ("num_bouts", "int64", True),
        ("mean_bout_duration_s", "float64", True),
        ("mean_interbout_interval_s", "float64", True),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


@pytest.mark.parametrize(
    ("table_name", "field_count"),
    (
        (STIMULUS_RESPONSE_TABLE, 129),
        (SWIM_BOUT_METRICS_TABLE, 70),
        (BOUT_KINEMATICS_METRICS_TABLE, 150),
    ),
)
def test_core_analytics_contracts_have_closed_unique_field_sets(
    table_name: str,
    field_count: int,
) -> None:
    fields = ARROW_TABLE_CONTRACTS[table_name].fields
    names = tuple(field.name for field in fields)

    assert len(fields) == field_count
    assert len(set(names)) == field_count
    assert names[:5] == (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
    )
    assert names[-3:] == (
        "collection_id",
        "collection_manifest_sha256",
        "collection_manifest_path",
    )


def test_every_analytics_table_has_an_exact_arrow_contract() -> None:
    assert set(ARROW_TABLE_CONTRACTS) == set(TABLE_CONTRACTS)
    assert set(EXACT_ARROW_SCHEMA_TABLES) == set(TABLE_CONTRACTS)
    envelope = arrow_contract_envelope(tuple(TABLE_CONTRACTS))
    assert envelope["inferred_v2_compatibility_tables"] == []
    assert set(envelope["exact_tables"]) == set(TABLE_CONTRACTS)


@pytest.mark.parametrize(
    ("table_name", "field_count"),
    (
        (CHASER_SPATIAL_TABLE, 59),
        (CHASER_DISTANCE_SUMMARY_TABLE, 46),
        (CHASER_EPOCH_BEHAVIOR_TABLE, 94),
        (CHASER_BOUT_EVENTS_TABLE, 66),
        (CHASER_BOUT_HISTOGRAM_TABLE, 68),
        (CHASER_IBI_HISTOGRAM_TABLE, 68),
        (CHASER_CENTER_DISTANCE_HISTOGRAM_TABLE, 58),
        (CHASER_SPEED_DISTANCE_TABLE, 50),
        (CHASER_DISTANCE_HISTOGRAM_TABLE, 47),
        (CHASER_QUADRANT_OCCUPANCY_SUMMARY_TABLE, 60),
        (CHASER_QUADRANT_OCCUPANCY_CHASER_PHASE_TABLE, 87),
        (CHASER_QUADRANT_OCCUPANCY_DENSITY_TABLE, 86),
        (CHASER_NEAR_FIELD_OCCUPANCY_SUMMARY_TABLE, 72),
        (CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE, 90),
        (CHASER_NEAR_FIELD_OCCUPANCY_RADIAL_DENSITY_TABLE, 87),
        (CHASER_NEAR_FIELD_OCCUPANCY_DISTANCE_CDF_TABLE, 76),
        (CHASER_EGOCENTRIC_SUMMARY_TABLE, 69),
        (CHASER_EGOCENTRIC_HISTOGRAM_TABLE, 70),
    ),
)
def test_chaser_arrow_contracts_are_closed_unique_and_keyed(
    table_name: str,
    field_count: int,
) -> None:
    fields = ARROW_TABLE_CONTRACTS[table_name].fields
    names = tuple(field.name for field in fields)
    by_name = {field.name: field for field in fields}

    assert len(fields) == field_count
    assert len(set(names)) == field_count
    assert names[:5] == (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
    )
    assert names[-3:] == (
        "collection_id",
        "collection_manifest_sha256",
        "collection_manifest_path",
    )
    for key in TABLE_CONTRACTS[table_name].primary_key:
        assert key in by_name
        assert not by_name[key].nullable


def test_near_field_v1_freezes_percentiles_without_dynamic_columns() -> None:
    names = tuple(
        field.name
        for field in ARROW_TABLE_CONTRACTS[
            CHASER_NEAR_FIELD_OCCUPANCY_CHASER_PHASE_TABLE
        ].fields
    )
    percentile_names = tuple(name for name in names if name.startswith("approach_p"))
    assert percentile_names == (
        "approach_p05_mm",
        "approach_p05_mm_percentile",
        "approach_p10_mm",
        "approach_p10_mm_percentile",
    )


@pytest.mark.parametrize("table_name", CHASER_ARROW_TABLES)
def test_chaser_exact_writer_uses_declared_schema(
    tmp_path: Path,
    table_name: str,
) -> None:
    count, parts = _write_table_parts(
        generation_root=tmp_path / table_name,
        table=table_name,
        rows_by_source=(("source-1", [_valid_chaser_row(table_name)]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()


@pytest.mark.parametrize("table_name", CHASER_ARROW_TABLES)
def test_chaser_exact_writer_rejects_unexpected_missing_and_duplicate_rows(
    tmp_path: Path,
    table_name: str,
) -> None:
    unexpected = _valid_chaser_row(table_name)
    unexpected["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / table_name / "unexpected",
            table=table_name,
            rows_by_source=(("source", [unexpected]),),
        )

    missing = _valid_chaser_row(table_name)
    required_name = next(
        field.name
        for field in ARROW_TABLE_CONTRACTS[table_name].fields
        if not field.nullable and field.name not in TABLE_CONTRACTS[table_name].primary_key
    )
    del missing[required_name]
    with pytest.raises(ValueError, match="null/missing non-nullable"):
        _write_table_parts(
            generation_root=tmp_path / table_name / "missing",
            table=table_name,
            rows_by_source=(("source", [missing]),),
        )

    duplicate = _valid_chaser_row(table_name)
    with pytest.raises(ValueError, match="duplicate primary key"):
        _write_table_parts(
            generation_root=tmp_path / table_name / "duplicate",
            table=table_name,
            rows_by_source=(("source", [duplicate, dict(duplicate)]),),
        )


def test_bout_kinematics_contract_keys_every_measurement_level() -> None:
    contract = TABLE_CONTRACTS[BOUT_KINEMATICS_METRICS_TABLE]

    assert contract.contract_version == 2
    assert contract.grain == "recording_x_swim_bout_x_measurement_level"
    assert contract.primary_key == ("recording_id", "bout_id", "measurement_level")


def test_core_analytics_exact_types_preserve_semantic_text_and_nullable_unions() -> None:
    response = {
        field.name: (field.arrow_type, field.nullable)
        for field in ARROW_TABLE_CONTRACTS[STIMULUS_RESPONSE_TABLE].fields
    }
    swim = {
        field.name: (field.arrow_type, field.nullable)
        for field in ARROW_TABLE_CONTRACTS[SWIM_BOUT_METRICS_TABLE].fields
    }
    bout = {
        field.name: (field.arrow_type, field.nullable)
        for field in ARROW_TABLE_CONTRACTS[BOUT_KINEMATICS_METRICS_TABLE].fields
    }

    assert response["fish_id"] == ("int64", False)
    assert response["omr_path_index"] == ("float64", True)
    assert response["radial_path_index"] == ("float64", True)
    assert swim["bout_id"] == ("int64", False)
    assert swim["threshold_crossing_valid"] == ("bool", True)
    assert bout["measurement_level"] == ("string", False)
    assert bout["physical_active_boundary_policy"] == ("string", True)
    assert bout["failure_reason"] == ("string", True)
    assert "failure_reason_bytes" not in bout


def test_baseline_summary_contract_freezes_all_95_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("chaser_distance_run", "string", False),
        ("chaser_distance_path", "string", False),
        ("chaser_distance_schema_id", "string", True),
        ("chaser_distance_schema_version", "int64", True),
        ("chaser_distance_method", "string", True),
        ("chaser_distance_method_version", "string", True),
        ("source_detection_path", "string", True),
        ("source_detection_kind", "string", True),
        ("source_stimulus_run", "string", True),
        ("source_stimulus_path", "string", True),
        ("source_stimulus_epoch_run", "string", True),
        ("source_stimulus_epoch_path", "string", True),
        ("source_refs_json", "string", False),
        ("coordinate_frame", "string", False),
        ("coordinate_origin", "string", False),
        ("fps", "float64", True),
        ("total_frames", "int64", True),
        ("pixels_per_mm_projector", "float64", False),
        ("source_chaser_distance_run", "string", False),
        ("source_chaser_distance_path", "string", False),
        ("source_epoch_behavior_component", "string", False),
        ("source_epoch_behavior_path", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_kinematics_track_path", "string", False),
        ("source_speed_level", "string", False),
        ("source_swim_bout_run", "string", True),
        ("source_swim_bout_path", "string", True),
        ("track_id", "int64", False),
        ("arena_center_x_px", "float64", False),
        ("arena_center_y_px", "float64", False),
        ("arena_radius_px", "float64", False),
        ("baseline_method", "string", False),
        ("baseline_method_version", "string", False),
        ("baseline_window_id", "int64", False),
        ("baseline_window_label", "string", False),
        ("start_frame", "int64", False),
        ("end_frame", "int64", False),
        ("start_time_s", "float64", False),
        ("end_time_s", "float64", False),
        ("duration_s", "float64", False),
        ("total_frame_count", "int64", False),
        ("valid_frame_count", "int64", False),
        ("missing_frame_count", "int64", False),
        ("tracking_dropout_fraction", "float64", True),
        ("speed_sample_count", "int64", False),
        ("mean_speed_mm_s", "float64", True),
        ("median_speed_mm_s", "float64", True),
        ("p95_speed_mm_s", "float64", True),
        ("max_speed_mm_s", "float64", True),
        ("total_path_mm", "float64", True),
        ("bout_count", "int64", False),
        ("bout_rate_per_min", "float64", True),
        ("arena_radius_mm", "float64", False),
        ("wall_band_mm", "float64", False),
        ("expected_uniform_wall_fraction", "float64", False),
        ("experimental_area_geometry_type", "string", False),
        ("boundary_distance_method", "string", False),
        ("wall_fraction_denominator", "string", False),
        ("wall_frame_count", "int64", False),
        ("wall_fraction", "float64", True),
        ("mean_distance_from_arena_center_mm", "float64", True),
        ("median_distance_from_arena_center_mm", "float64", True),
        ("p95_distance_from_arena_center_mm", "float64", True),
        ("mean_distance_to_arena_boundary_mm", "float64", True),
        ("median_distance_to_arena_boundary_mm", "float64", True),
        ("p95_distance_to_arena_boundary_mm", "float64", True),
        ("mean_center_distance_norm", "float64", True),
        ("median_center_distance_norm", "float64", True),
        ("x_axis_direction", "string", False),
        ("y_axis_direction", "string", False),
        ("spatial_grid_size", "int64", False),
        ("spatial_valid_sample_count", "int64", False),
        ("spatial_visited_cell_count", "int64", False),
        ("spatial_entropy_normalized", "float64", True),
        ("spatial_max_cell_fraction", "float64", True),
        ("quadrant_entropy_normalized", "float64", True),
        ("quadrant_max_fraction", "float64", True),
        ("median_bout_duration_s", "float64", True),
        ("mean_bout_duration_s", "float64", True),
        ("median_bout_path_length_mm", "float64", True),
        ("mean_bout_path_length_mm", "float64", True),
        ("median_abs_bout_net_heading_change_deg", "float64", True),
        ("mean_abs_bout_net_heading_change_deg", "float64", True),
        ("median_inter_bout_interval_s", "float64", True),
        ("mean_inter_bout_interval_s", "float64", True),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


def test_baseline_time_bins_contract_freezes_all_77_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_TIME_BINS_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("chaser_distance_run", "string", False),
        ("chaser_distance_path", "string", False),
        ("chaser_distance_schema_id", "string", True),
        ("chaser_distance_schema_version", "int64", True),
        ("chaser_distance_method", "string", True),
        ("chaser_distance_method_version", "string", True),
        ("source_detection_path", "string", True),
        ("source_detection_kind", "string", True),
        ("source_stimulus_run", "string", True),
        ("source_stimulus_path", "string", True),
        ("source_stimulus_epoch_run", "string", True),
        ("source_stimulus_epoch_path", "string", True),
        ("source_refs_json", "string", False),
        ("coordinate_frame", "string", False),
        ("coordinate_origin", "string", False),
        ("fps", "float64", True),
        ("total_frames", "int64", True),
        ("pixels_per_mm_projector", "float64", False),
        ("source_chaser_distance_run", "string", False),
        ("source_chaser_distance_path", "string", False),
        ("source_epoch_behavior_component", "string", False),
        ("source_epoch_behavior_path", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_kinematics_track_path", "string", False),
        ("source_speed_level", "string", False),
        ("source_swim_bout_run", "string", True),
        ("source_swim_bout_path", "string", True),
        ("track_id", "int64", False),
        ("arena_center_x_px", "float64", False),
        ("arena_center_y_px", "float64", False),
        ("arena_radius_px", "float64", False),
        ("baseline_method", "string", False),
        ("baseline_method_version", "string", False),
        ("baseline_window_id", "int64", False),
        ("baseline_window_label", "string", False),
        ("time_bin_index", "int64", False),
        ("relative_start_s", "float64", False),
        ("relative_end_s", "float64", False),
        ("time_bin_duration_s", "float64", False),
        ("source_start_frame", "int64", False),
        ("source_end_frame", "int64", False),
        ("expected_frame_count", "int64", False),
        ("valid_position_count", "int64", False),
        ("valid_position_fraction", "float64", True),
        ("speed_sample_count", "int64", False),
        ("mean_speed_mm_s", "float64", True),
        ("median_speed_mm_s", "float64", True),
        ("p95_speed_mm_s", "float64", True),
        ("distance_travelled_mm", "float64", True),
        ("mean_center_distance_mm", "float64", True),
        ("median_center_distance_mm", "float64", True),
        ("mean_distance_to_arena_boundary_mm", "float64", True),
        ("median_distance_to_arena_boundary_mm", "float64", True),
        ("experimental_area_geometry_type", "string", False),
        ("boundary_distance_method", "string", False),
        ("wall_fraction_denominator", "string", False),
        ("wall_frame_count", "int64", False),
        ("wall_fraction", "float64", True),
        ("representative_position_method", "string", False),
        ("representative_x_mm", "float64", True),
        ("representative_y_mm", "float64", True),
        ("mean_heading_deg", "float64", True),
        ("heading_resultant", "float64", True),
        ("bout_count", "int64", False),
        ("x_axis_direction", "string", False),
        ("y_axis_direction", "string", False),
        ("time_bin_policy", "string", False),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


def test_baseline_samples_contract_freezes_all_71_fields_in_order() -> None:
    fields = ARROW_TABLE_CONTRACTS[BASELINE_KINEMATIC_SAMPLES_TABLE].fields
    assert tuple(
        (field.name, field.arrow_type, field.nullable) for field in fields
    ) == (
        ("export_schema_version", "int32", False),
        ("table_name", "string", False),
        ("recording_id", "string", False),
        ("zarr_path", "string", False),
        ("source_lineage_hash", "string", False),
        ("chaser_distance_run", "string", False),
        ("chaser_distance_path", "string", False),
        ("chaser_distance_schema_id", "string", True),
        ("chaser_distance_schema_version", "int64", True),
        ("chaser_distance_method", "string", True),
        ("chaser_distance_method_version", "string", True),
        ("source_detection_path", "string", True),
        ("source_detection_kind", "string", True),
        ("source_stimulus_run", "string", True),
        ("source_stimulus_path", "string", True),
        ("source_stimulus_epoch_run", "string", True),
        ("source_stimulus_epoch_path", "string", True),
        ("source_refs_json", "string", False),
        ("coordinate_frame", "string", False),
        ("coordinate_origin", "string", False),
        ("fps", "float64", True),
        ("total_frames", "int64", True),
        ("pixels_per_mm_projector", "float64", False),
        ("source_chaser_distance_run", "string", False),
        ("source_chaser_distance_path", "string", False),
        ("source_epoch_behavior_component", "string", False),
        ("source_epoch_behavior_path", "string", False),
        ("source_track_kinematics_run", "string", False),
        ("source_track_kinematics_scope", "string", False),
        ("source_track_kinematics_path", "string", False),
        ("source_track_kinematics_track_path", "string", False),
        ("source_speed_level", "string", False),
        ("source_swim_bout_run", "string", True),
        ("source_swim_bout_path", "string", True),
        ("track_id", "int64", False),
        ("arena_center_x_px", "float64", False),
        ("arena_center_y_px", "float64", False),
        ("arena_radius_px", "float64", False),
        ("baseline_method", "string", False),
        ("baseline_method_version", "string", False),
        ("baseline_window_id", "int64", False),
        ("baseline_window_label", "string", False),
        ("source_sample_index", "int64", False),
        ("source_frame", "int64", False),
        ("source_time_s", "float64", False),
        ("relative_time_s", "float64", False),
        ("x_arena_mm", "float64", True),
        ("y_arena_mm", "float64", True),
        ("x_arena_fraction", "float64", True),
        ("y_arena_fraction", "float64", True),
        ("speed_mm_s", "float64", True),
        ("heading_deg", "float64", True),
        ("frame_path_distance_mm", "float64", True),
        ("center_distance_mm", "float64", True),
        ("distance_to_arena_boundary_mm", "float64", True),
        ("wall", "bool", True),
        ("experimental_area_geometry_type", "string", False),
        ("boundary_distance_method", "string", False),
        ("position_valid", "bool", False),
        ("sample_valid", "bool", False),
        ("sampling_policy", "string", False),
        ("sampling_stride_frames", "int64", False),
        ("requested_sample_rate_hz", "float64", True),
        ("source_sample_rate_hz", "float64", False),
        ("nominal_sample_rate_hz", "float64", False),
        ("effective_sample_rate_hz", "float64", False),
        ("x_axis_direction", "string", False),
        ("y_axis_direction", "string", False),
        ("collection_id", "string", True),
        ("collection_manifest_sha256", "string", True),
        ("collection_manifest_path", "string", True),
    )


@pytest.mark.parametrize(
    "table_name",
    (
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
    ),
)
@pytest.mark.parametrize(
    "mutation",
    (
        lambda fields: fields.reverse(),
        lambda fields: fields[0].update({"arrow_type": "int64"}),
        lambda fields: fields[0].update({"nullable": True}),
        lambda fields: fields.append(
            {"name": "unexpected", "arrow_type": "string", "nullable": True}
        ),
        lambda fields: fields.pop(),
    ),
    ids=("reordered", "wrong-type", "wrong-nullability", "unexpected", "missing"),
)
def test_rehashed_arrow_contract_tampering_fails_closed(
    table_name: str,
    mutation: Any,
) -> None:
    envelope = arrow_contract_envelope((table_name,))
    fields = envelope["exact_tables"][table_name]["fields"]
    mutation(fields)
    _rehash(envelope)

    with pytest.raises(ValueError, match="differs from installed contracts"):
        validate_arrow_contract_envelope(
            envelope,
            (table_name,),
        )


def test_exact_writer_uses_declared_order_types_nullability_and_digest(tmp_path: Path) -> None:
    table_name = POSITION_OCCUPANCY_HISTOGRAM_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_position_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    expected = exact_arrow_schema(table_name, metadata={})
    assert schema.remove_metadata() == expected.remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_recording_summary_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = RECORDING_SUMMARY_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_recording_summary_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_stimulus_steps_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = STIMULUS_STEPS_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_stimulus_step_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_stimulus_step_summary_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = STIMULUS_STEP_SUMMARY_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_stimulus_step_summary_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


@pytest.mark.parametrize(
    "table_name",
    (
        STIMULUS_RESPONSE_TABLE,
        SWIM_BOUT_METRICS_TABLE,
        BOUT_KINEMATICS_METRICS_TABLE,
    ),
)
def test_core_analytics_exact_writer_uses_declared_schema(
    tmp_path: Path,
    table_name: str,
) -> None:
    count, parts = _write_table_parts(
        generation_root=tmp_path / table_name,
        table=table_name,
        rows_by_source=(("source-1", [_valid_core_analytics_row(table_name)]),),
    )

    assert count == 1
    validate_arrow_schema(table_name, pq.ParquetFile(parts[0]).schema_arrow)


@pytest.mark.parametrize(
    "table_name",
    (
        STIMULUS_RESPONSE_TABLE,
        SWIM_BOUT_METRICS_TABLE,
        BOUT_KINEMATICS_METRICS_TABLE,
    ),
)
def test_core_analytics_exact_writer_rejects_duplicate_primary_keys(
    tmp_path: Path,
    table_name: str,
) -> None:
    row = _valid_core_analytics_row(table_name)

    with pytest.raises(ValueError, match="duplicate primary key"):
        _write_table_parts(
            generation_root=tmp_path / table_name,
            table=table_name,
            rows_by_source=(("source-1", [row, dict(row)]),),
        )


@pytest.mark.parametrize(
    "table_name",
    (
        STIMULUS_RESPONSE_TABLE,
        SWIM_BOUT_METRICS_TABLE,
        BOUT_KINEMATICS_METRICS_TABLE,
    ),
)
def test_core_analytics_exact_writer_rejects_null_primary_keys(
    tmp_path: Path,
    table_name: str,
) -> None:
    row = _valid_core_analytics_row(table_name)
    row[TABLE_CONTRACTS[table_name].primary_key[-1]] = None

    with pytest.raises(ValueError, match="null/missing primary key"):
        _write_table_parts(
            generation_root=tmp_path / table_name,
            table=table_name,
            rows_by_source=(("source-1", [row]),),
        )


@pytest.mark.parametrize(
    ("table_name", "row_factory"),
    (
        (BASELINE_BEHAVIOR_SUMMARY_TABLE, _valid_baseline_summary_row),
        (BASELINE_BEHAVIOR_TIME_BINS_TABLE, _valid_baseline_time_bin_row),
        (BASELINE_KINEMATIC_SAMPLES_TABLE, _valid_baseline_sample_row),
    ),
)
def test_baseline_exact_writers_reject_duplicate_primary_keys(
    tmp_path: Path,
    table_name: str,
    row_factory: Any,
) -> None:
    row = row_factory()
    with pytest.raises(ValueError, match="duplicate primary key"):
        _write_table_parts(
            generation_root=tmp_path / table_name,
            table=table_name,
            rows_by_source=(("source-1", [row, dict(row)]),),
        )


def test_baseline_summary_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = BASELINE_BEHAVIOR_SUMMARY_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_baseline_summary_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert (
        schema.remove_metadata()
        == exact_arrow_schema(
            table_name,
            metadata={},
        ).remove_metadata()
    )
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_baseline_time_bins_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = BASELINE_BEHAVIOR_TIME_BINS_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_baseline_time_bin_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_baseline_samples_exact_writer_uses_declared_schema(tmp_path: Path) -> None:
    table_name = BASELINE_KINEMATIC_SAMPLES_TABLE
    count, parts = _write_table_parts(
        generation_root=tmp_path / "generation",
        table=table_name,
        rows_by_source=(("source-1", [_valid_baseline_sample_row()]),),
    )

    assert count == 1
    schema = pq.ParquetFile(parts[0]).schema_arrow
    validate_arrow_schema(table_name, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        table_name,
        metadata={},
    ).remove_metadata()
    assert schema.metadata[b"palette.arrow_schema_sha256"].decode() == (
        ARROW_TABLE_CONTRACTS[table_name].payload_sha256
    )


def test_exact_writer_rejects_unexpected_and_missing_nonnullable_fields(
    tmp_path: Path,
) -> None:
    row = _valid_position_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            rows_by_source=(("source", [row]),),
        )

    row = _valid_position_row()
    del row["hist_count"]
    with pytest.raises(ValueError, match="null/missing non-nullable"):
        _write_table_parts(
            generation_root=tmp_path / "missing",
            table=POSITION_OCCUPANCY_HISTOGRAM_TABLE,
            rows_by_source=(("source", [row]),),
        )


def test_recording_summary_exact_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_recording_summary_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=RECORDING_SUMMARY_TABLE,
            rows_by_source=(("source", [row]),),
        )

    for field_name in (
        "export_schema_version",
        "table_name",
        "recording_id",
        "zarr_path",
        "source_lineage_hash",
        "stimulus_step_count",
    ):
        row = _valid_recording_summary_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=RECORDING_SUMMARY_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_stimulus_steps_writer_rejects_unexpected_and_every_missing_required_field(
    tmp_path: Path,
) -> None:
    row = _valid_stimulus_step_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=STIMULUS_STEPS_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[STIMULUS_STEPS_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_stimulus_step_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=STIMULUS_STEPS_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_stimulus_step_summary_writer_rejects_unexpected_and_every_missing_required_field(
    tmp_path: Path,
) -> None:
    row = _valid_stimulus_step_summary_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=STIMULUS_STEP_SUMMARY_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_stimulus_step_summary_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=STIMULUS_STEP_SUMMARY_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_baseline_summary_exact_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_baseline_summary_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=BASELINE_BEHAVIOR_SUMMARY_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_baseline_summary_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=BASELINE_BEHAVIOR_SUMMARY_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_baseline_time_bins_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_baseline_time_bin_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=BASELINE_BEHAVIOR_TIME_BINS_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_TIME_BINS_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_baseline_time_bin_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=BASELINE_BEHAVIOR_TIME_BINS_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_baseline_samples_writer_rejects_unexpected_and_missing_required_fields(
    tmp_path: Path,
) -> None:
    row = _valid_baseline_sample_row()
    row["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected fields"):
        _write_table_parts(
            generation_root=tmp_path / "unexpected",
            table=BASELINE_KINEMATIC_SAMPLES_TABLE,
            rows_by_source=(("source", [row]),),
        )

    required = {
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_KINEMATIC_SAMPLES_TABLE].fields
        if not field.nullable
    }
    for field_name in sorted(required):
        row = _valid_baseline_sample_row()
        del row[field_name]
        with pytest.raises(ValueError, match="null/missing non-nullable"):
            _write_table_parts(
                generation_root=tmp_path / f"missing-{field_name}",
                table=BASELINE_KINEMATIC_SAMPLES_TABLE,
                rows_by_source=(("source", [row]),),
            )


def test_recording_summary_zero_rows_publish_no_parts_but_retain_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={RECORDING_SUMMARY_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-recording-summary",
        tables=(RECORDING_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {RECORDING_SUMMARY_TABLE: 0}
    assert manifest["part_files_by_table"] == {RECORDING_SUMMARY_TABLE: []}
    assert manifest["publication"]["parts_by_table"] == {
        RECORDING_SUMMARY_TABLE: []
    }
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        RECORDING_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-recording-summary")["status"] == "valid"


def test_stimulus_steps_zero_rows_retain_exact_contract_without_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={STIMULUS_STEPS_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-stimulus-steps",
        tables=(STIMULUS_STEPS_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {STIMULUS_STEPS_TABLE: 0}
    assert manifest["part_files_by_table"] == {STIMULUS_STEPS_TABLE: []}
    assert manifest["publication"]["parts_by_table"] == {STIMULUS_STEPS_TABLE: []}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        STIMULUS_STEPS_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-stimulus-steps")["status"] == "valid"


def test_stimulus_step_summary_zero_rows_retain_exact_contract_without_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={STIMULUS_STEP_SUMMARY_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-stimulus-step-summary",
        tables=(STIMULUS_STEP_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {STIMULUS_STEP_SUMMARY_TABLE: 0}
    assert manifest["part_files_by_table"] == {STIMULUS_STEP_SUMMARY_TABLE: []}
    assert manifest["publication"]["parts_by_table"] == {
        STIMULUS_STEP_SUMMARY_TABLE: []
    }
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        STIMULUS_STEP_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert (
        validate_export_run(root, "empty-stimulus-step-summary")["status"]
        == "valid"
    )


def test_baseline_summary_zero_rows_publish_no_parts_but_retain_exact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={BASELINE_BEHAVIOR_SUMMARY_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-baseline-summary",
        tables=(BASELINE_BEHAVIOR_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: 0}
    assert manifest["part_files_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: []}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        BASELINE_BEHAVIOR_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-baseline-summary")["status"] == "valid"


def test_baseline_time_bins_zero_rows_retain_exact_contract_without_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={BASELINE_BEHAVIOR_TIME_BINS_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-baseline-time-bins",
        tables=(BASELINE_BEHAVIOR_TIME_BINS_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_TIME_BINS_TABLE: 0}
    assert manifest["part_files_by_table"] == {BASELINE_BEHAVIOR_TIME_BINS_TABLE: []}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-baseline-time-bins")["status"] == "valid"


def test_baseline_samples_zero_rows_retain_exact_contract_without_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={BASELINE_KINEMATIC_SAMPLES_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="empty-baseline-samples",
        tables=(BASELINE_KINEMATIC_SAMPLES_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_KINEMATIC_SAMPLES_TABLE: 0}
    assert manifest["part_files_by_table"] == {BASELINE_KINEMATIC_SAMPLES_TABLE: []}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        BASELINE_KINEMATIC_SAMPLES_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    assert validate_export_run(root, "empty-baseline-samples")["status"] == "valid"


def test_real_detection_occupancy_export_uses_exact_arrow_contract(
    tmp_path: Path,
) -> None:
    source = _make_archive_with_detection_occupancy(tmp_path)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="occupancy-arrow",
        tables=(POSITION_OCCUPANCY_HISTOGRAM_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"][POSITION_OCCUPANCY_HISTOGRAM_TABLE] > 0
    part = root / manifest["part_files_by_table"][POSITION_OCCUPANCY_HISTOGRAM_TABLE][0]
    validate_arrow_schema(
        POSITION_OCCUPANCY_HISTOGRAM_TABLE,
        pq.ParquetFile(part).schema_arrow,
    )
    assert validate_export_run(root, "occupancy-arrow")["status"] == "valid"


def test_real_recording_summary_export_uses_exact_schema_and_collection_fields(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path, source)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="recording-summary-arrow",
        tables=(RECORDING_SUMMARY_TABLE,),
        jobs=1,
        collection_manifest_path=collection_path,
    )

    assert manifest["row_counts_by_table"] == {RECORDING_SUMMARY_TABLE: 1}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        RECORDING_SUMMARY_TABLE,
    )
    assert manifest["arrow_schema_contracts"]["inferred_v2_compatibility_tables"] == []
    part = root / manifest["part_files_by_table"][RECORDING_SUMMARY_TABLE][0]
    schema = pq.ParquetFile(part).schema_arrow
    validate_arrow_schema(RECORDING_SUMMARY_TABLE, schema)
    assert schema.remove_metadata() == exact_arrow_schema(
        RECORDING_SUMMARY_TABLE,
        metadata={},
    ).remove_metadata()
    row = pq.read_table(part).to_pylist()[0]
    assert row["recording_id"] == "recording_a"
    assert row["stimulus_step_count"] == 2
    assert row["collection_id"] == collection["collection_id"]
    assert row["collection_manifest_sha256"] == collection["manifest_sha256"]
    assert validate_export_run(root, "recording-summary-arrow")["status"] == "valid"


def test_real_stimulus_steps_export_uses_maintained_exact_selected_representation(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path, source)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="stimulus-steps-arrow",
        tables=(STIMULUS_STEPS_TABLE,),
        jobs=1,
        collection_manifest_path=collection_path,
    )

    assert manifest["row_counts_by_table"] == {STIMULUS_STEPS_TABLE: 2}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        STIMULUS_STEPS_TABLE,
    )
    selected = manifest_selected_part_files(
        root,
        "stimulus-steps-arrow",
        STIMULUS_STEPS_TABLE,
    )
    assert len(selected) == 1
    parquet_file = pq.ParquetFile(selected[0])
    validate_arrow_schema(STIMULUS_STEPS_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name for field in ARROW_TABLE_CONTRACTS[STIMULUS_STEPS_TABLE].fields
    ]
    rows = parquet_file.read().to_pylist()
    moving, concentric = rows
    assert moving["step_index"] == 0
    assert moving["moving_grating_orientation_degrees_authored"] == 0.0
    assert moving["moving_grating_direction_mapping_validated"] is False
    assert moving["concentric_grating_radial_sign_authored"] is None
    assert concentric["step_index"] == 1
    assert concentric["moving_grating_metadata_schema_version"] is None
    assert concentric["concentric_grating_radial_sign_authored"] == -1
    assert concentric["concentric_grating_radial_polarity_validated"] is False
    assert {row["collection_id"] for row in rows} == {collection["collection_id"]}
    assert all(
        row["protocol_signature_hash"] == row["derived_protocol_hash"]
        for row in rows
    )
    assert not any(
        "direction_degrees" == name or "stimulus_radial_polarity_authored" in name
        for name in parquet_file.schema_arrow.names
    )
    assert not any(name.startswith("looming_dot_") for name in parquet_file.schema_arrow.names)
    assert validate_export_run(root, "stimulus-steps-arrow")["status"] == "valid"


def test_real_stimulus_step_summary_export_uses_exact_selected_representation(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    collection_path = tmp_path / "collection.manifest.json"
    collection = _write_collection_manifest(collection_path, source)
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="stimulus-step-summary-arrow",
        tables=(STIMULUS_STEP_SUMMARY_TABLE,),
        jobs=1,
        collection_manifest_path=collection_path,
    )

    assert manifest["row_counts_by_table"] == {STIMULUS_STEP_SUMMARY_TABLE: 2}
    assert tuple(manifest["arrow_schema_contracts"]["exact_tables"]) == (
        STIMULUS_STEP_SUMMARY_TABLE,
    )
    selected = manifest_selected_part_files(
        root,
        "stimulus-step-summary-arrow",
        STIMULUS_STEP_SUMMARY_TABLE,
    )
    assert len(selected) == 1
    parquet_file = pq.ParquetFile(selected[0])
    validate_arrow_schema(STIMULUS_STEP_SUMMARY_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name
        for field in ARROW_TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE].fields
    ]
    rows = parquet_file.read().to_pylist()
    moving, concentric = rows
    assert (moving["fish_id"], moving["step_index"]) == (0, 0)
    assert moving["total_distance_mm"] == 12.5
    assert moving["mean_speed_mm_s"] == 5.0
    assert moving["median_speed_mm_s"] is None
    assert moving["num_bouts"] == 2
    assert moving["mean_bout_duration_s"] is None
    assert (concentric["fish_id"], concentric["step_index"]) == (0, 1)
    assert {row["collection_id"] for row in rows} == {collection["collection_id"]}
    assert all(
        row["protocol_signature_hash"] == row["derived_protocol_hash"]
        for row in rows
    )
    assert validate_export_run(root, "stimulus-step-summary-arrow")["status"] == "valid"


def test_real_stimulus_step_summary_primary_key_distinguishes_two_fish_in_one_step(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "two_fish_analysis.zarr")
    archive = zarr.open_group(str(source), mode="a", use_consolidated=False)
    per_fish = archive[
        "analysis/stimulus_response_runs/stimulus_response_test/steps/step_0/per_fish"
    ]
    for name in tuple(per_fish.array_keys()):
        array = per_fish[name]
        first = array[0]
        array.resize((2,))
        array[1] = 1 if name == "fish_id" else first
    root = tmp_path / "exports"

    manifest = export_sources(
        [source],
        output_root=root,
        export_run_id="stimulus-step-summary-two-fish",
        tables=(STIMULUS_STEP_SUMMARY_TABLE,),
        jobs=1,
    )

    rows = pq.read_table(
        [
            str(path)
            for path in manifest_selected_part_files(
            root,
            "stimulus-step-summary-two-fish",
            STIMULUS_STEP_SUMMARY_TABLE,
            )
        ]
    ).to_pylist()
    same_step = [row for row in rows if row["step_index"] == 0]
    contract = TABLE_CONTRACTS[STIMULUS_STEP_SUMMARY_TABLE]
    keys = {
        tuple(row[field] for field in contract.primary_key)
        for row in same_step
    }

    assert manifest["row_counts_by_table"] == {STIMULUS_STEP_SUMMARY_TABLE: 3}
    assert {row["fish_id"] for row in same_step} == {0, 1}
    assert keys == {("two_fish", 0, 0), ("two_fish", 1, 0)}
    assert validate_export_run(root, "stimulus-step-summary-two-fish")["status"] == "valid"


def test_selected_stimulus_step_summary_rejects_old_two_field_primary_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_name = STIMULUS_STEP_SUMMARY_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        first = _valid_stimulus_step_summary_row()
        second = {**first, "fish_id": 8, "source_lineage_hash": "8" * 64}
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [first, second]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    run_id = "stimulus-step-summary-old-key"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=run_id,
        tables=(table_name,),
        jobs=1,
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["table_contracts"][table_name]["primary_key"] = [
        "recording_id",
        "step_index",
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="manifest table contract does not match installed V2",
    ):
        validate_export_run(root, run_id)


def test_stimulus_step_summary_rejects_unowned_dynamic_metric_before_publication(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "dynamic_metric_analysis.zarr")
    archive = zarr.open_group(str(source), mode="a", use_consolidated=False)
    per_fish = archive[
        "analysis/stimulus_response_runs/stimulus_response_test/steps/step_0/per_fish"
    ]
    per_fish.create_array("future_metric", data=np.asarray([1.0], dtype=np.float32))
    root = tmp_path / "exports"
    run_id = "stimulus-step-summary-dynamic"

    with pytest.raises(ValueError, match="future_metric"):
        export_sources(
            [source],
            output_root=root,
            export_run_id=run_id,
            tables=(STIMULUS_STEP_SUMMARY_TABLE,),
            jobs=1,
        )

    assert not (root / "v1" / "manifests" / f"export_run_id={run_id}.json").exists()
    assert not list((root / "v1" / ".generations").glob(f"export_run_id={run_id}/**/*"))


@pytest.mark.parametrize(
    ("case", "unexpected_field"),
    (
        ("legacy-moving", "moving_grating_direction_degrees"),
        (
            "legacy-concentric",
            "concentric_grating_stimulus_radial_polarity_authored",
        ),
        ("looming", "looming_dot_radius_px"),
    ),
)
def test_stimulus_steps_rejects_legacy_and_undeclared_child_metadata_before_publication(
    tmp_path: Path,
    case: str,
    unexpected_field: str,
) -> None:
    source = _make_source_zarr(tmp_path / f"{case}_analysis.zarr")
    archive = zarr.open_group(str(source), mode="a", use_consolidated=False)
    steps = archive["analysis/stimulus_runs/stimulus_test/steps"]
    if case == "legacy-moving":
        steps["step_0/moving_grating"].attrs["direction_degrees"] = 0.0
    elif case == "legacy-concentric":
        steps["step_1/concentric_grating"].attrs[
            "stimulus_radial_polarity_authored"
        ] = "contracting"
    elif case == "looming":
        looming = steps["step_0"].create_group("looming_dot")
        looming.attrs["radius_px"] = 12.0
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    root = tmp_path / "exports"
    run_id = f"stimulus-steps-{case}"
    with pytest.raises(ValueError, match=unexpected_field):
        export_sources(
            [source],
            output_root=root,
            export_run_id=run_id,
            tables=(STIMULUS_STEPS_TABLE,),
            jobs=1,
        )

    assert not (root / "v1" / "manifests" / f"export_run_id={run_id}.json").exists()
    assert not list((root / "v1" / ".generations").glob(f"export_run_id={run_id}/**/*"))


def test_stimulus_steps_empty_looming_group_has_no_current_row_representation(
    tmp_path: Path,
) -> None:
    source = _make_source_zarr(tmp_path / "empty_looming_analysis.zarr")
    archive = zarr.open_group(str(source), mode="a", use_consolidated=False)
    archive[
        "analysis/stimulus_runs/stimulus_test/steps/step_0"
    ].create_group("looming_dot")
    root = tmp_path / "exports"

    export_sources(
        [source],
        output_root=root,
        export_run_id="stimulus-steps-empty-looming",
        tables=(STIMULUS_STEPS_TABLE,),
        jobs=1,
    )

    selected = manifest_selected_part_files(
        root,
        "stimulus-steps-empty-looming",
        STIMULUS_STEPS_TABLE,
    )
    assert selected
    assert not any(
        name.startswith("looming_dot_")
        for name in pq.ParquetFile(selected[0]).schema_arrow.names
    )
    assert (
        validate_export_run(root, "stimulus-steps-empty-looming")["status"]
        == "valid"
    )


def _make_baseline_representation_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    source = tmp_path / "baseline_recording_analysis.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    calibration = analysis.create_group("calibration")
    calibration.attrs.update(
        {
            "experimental_area_shape": "circle",
            "experimental_area_center_x_px": 10.0,
            "experimental_area_center_y_px": 10.0,
            "experimental_area_radius_px": 10.0,
        }
    )
    chaser_parent = analysis.create_group("chaser_distance_runs")
    chaser = chaser_parent.create_group("run-1")
    chaser.attrs.update(
        {
            "schema_id": "palette.chaser.distance.v1",
            "schema_version": 1,
            "method": "fixture",
            "method_version": "1",
            "coordinate_frame": "projector_canvas_px",
            "coordinate_origin": "top_left",
            "total_frames": 20,
            "pixels_per_mm_projector": 1.0,
            "source_refs": {},
        }
    )
    positions = chaser.create_group("positions")
    positions.create_array(
        "fish_centroid_arena_xy",
        data=np.column_stack([np.linspace(10.0, 19.0, 20), np.full(20, 10.0)]),
    )
    positions.create_array("fish_valid", data=np.ones(20, dtype=bool))

    components = chaser.create_group("epoch_behavior_summary")
    components.attrs.update({"latest": "component-1", "latest_complete": "component-1"})
    component = components.create_group("component-1")
    component.attrs.update(
        {
            "status": "complete",
            "schema_id": "palette.chaser.epoch_behavior_summary.v1",
            "source_refs": {
                "source_track_kinematics_run": "track-1",
                "source_track_kinematics_scope": "offline",
                "source_track_kinematics_track_id": 0,
            },
            "parameters": {"speed_level": "filtered", "wall_band_mm": 2.0},
        }
    )
    summary = np.asarray(
        [
            (
                b"pre_event",
                0,
                0,
                9,
                0.0,
                1.0,
                1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                999.0,
            )
        ],
        dtype=[
            ("window_label", "S32"),
            ("window_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("start_time_s", "f8"),
            ("end_time_s", "f8"),
            ("duration_s", "f8"),
            ("median_bout_duration_s", "f8"),
            ("mean_bout_duration_s", "f8"),
            ("median_bout_path_length_mm", "f8"),
            ("mean_bout_path_length_mm", "f8"),
            ("median_abs_bout_net_heading_change_deg", "f8"),
            ("mean_abs_bout_net_heading_change_deg", "f8"),
            ("median_inter_bout_interval_s", "f8"),
            ("mean_inter_bout_interval_s", "f8"),
            ("future_source_metric", "f8"),
        ],
    )
    write_columnar_dataset(component, "per_epoch_fish", summary, shard_rows=None)

    def latest_run(
        opened_root: Any,
        parent_path: str,
        requested: str | None = None,
    ) -> tuple[Any | None, str | None, str | None]:
        if parent_path == "analysis/chaser_distance_runs":
            assert requested is None
            return opened_root[parent_path]["run-1"], "run-1", None
        return None, None, "fixture has no requested run"

    track = SimpleNamespace(
        run_name="track-1",
        scope="offline",
        run_path="analysis/track_kinematics_runs/track-1",
        track_path="analysis/track_kinematics_runs/track-1/tracks/id_0",
        run_attrs={"fps": 10.0},
        frame_indices=np.arange(20, dtype=np.int64),
        time_seconds=np.arange(20, dtype=np.float64) / 10.0,
        speed_mm_by_level={"filtered": np.arange(20, dtype=np.float64)},
        frame_path_distance_mm_by_level={"filtered": np.ones(20, dtype=np.float64)},
        smoothed_heading_degrees=np.linspace(-45.0, 45.0, 20),
        heading_degrees=np.linspace(-45.0, 45.0, 20),
        sample_valid=np.ones(20, dtype=bool),
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics._latest_run",
        latest_run,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.load_track_kinematics_track",
        lambda *_args, **_kwargs: track,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )

    return source


def test_real_baseline_summary_export_uses_exact_schema_without_promoting_source_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_baseline_representation_source(tmp_path, monkeypatch)

    export_root = tmp_path / "exports"
    manifest = export_sources(
        [source],
        output_root=export_root,
        export_run_id="baseline-summary-arrow",
        tables=(BASELINE_BEHAVIOR_SUMMARY_TABLE,),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_SUMMARY_TABLE: 1}
    part = (
        export_root
        / manifest["part_files_by_table"][BASELINE_BEHAVIOR_SUMMARY_TABLE][0]
    )
    parquet_file = pq.ParquetFile(part)
    table = parquet_file.read()
    validate_arrow_schema(BASELINE_BEHAVIOR_SUMMARY_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_SUMMARY_TABLE].fields
    ]
    row = table.to_pylist()[0]
    assert row["recording_id"] == "baseline_recording"
    assert row["fps"] is None
    assert row["median_bout_duration_s"] == 1.0
    assert "future_source_metric" not in row
    assert (
        validate_export_run(export_root, "baseline-summary-arrow")["status"] == "valid"
    )


def test_real_baseline_time_bins_export_uses_exact_closed_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_baseline_representation_source(tmp_path, monkeypatch)
    export_root = tmp_path / "time-bin-exports"
    manifest = export_sources(
        [source],
        output_root=export_root,
        export_run_id="baseline-time-bins-arrow",
        tables=(BASELINE_BEHAVIOR_TIME_BINS_TABLE,),
        jobs=1,
        baseline_time_bin_s=0.5,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_BEHAVIOR_TIME_BINS_TABLE: 2}
    part = (
        export_root
        / manifest["part_files_by_table"][BASELINE_BEHAVIOR_TIME_BINS_TABLE][0]
    )
    parquet_file = pq.ParquetFile(part)
    table = parquet_file.read()
    validate_arrow_schema(BASELINE_BEHAVIOR_TIME_BINS_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_BEHAVIOR_TIME_BINS_TABLE].fields
    ]
    rows = table.to_pylist()
    assert [row["time_bin_index"] for row in rows] == [0, 1]
    assert [row["source_start_frame"] for row in rows] == [0, 5]
    assert all(row["fps"] is None for row in rows)
    assert all("future_source_metric" not in row for row in rows)
    assert (
        validate_export_run(export_root, "baseline-time-bins-arrow")["status"]
        == "valid"
    )


def test_real_baseline_samples_export_uses_exact_closed_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_baseline_representation_source(tmp_path, monkeypatch)
    export_root = tmp_path / "sample-exports"
    manifest = export_sources(
        [source],
        output_root=export_root,
        export_run_id="baseline-samples-arrow",
        tables=(BASELINE_KINEMATIC_SAMPLES_TABLE,),
        jobs=1,
        baseline_full_resolution_samples=True,
    )

    assert manifest["row_counts_by_table"] == {BASELINE_KINEMATIC_SAMPLES_TABLE: 10}
    part = (
        export_root
        / manifest["part_files_by_table"][BASELINE_KINEMATIC_SAMPLES_TABLE][0]
    )
    parquet_file = pq.ParquetFile(part)
    table = parquet_file.read()
    validate_arrow_schema(BASELINE_KINEMATIC_SAMPLES_TABLE, parquet_file.schema_arrow)
    assert parquet_file.schema_arrow.names == [
        field.name
        for field in ARROW_TABLE_CONTRACTS[BASELINE_KINEMATIC_SAMPLES_TABLE].fields
    ]
    rows = table.to_pylist()
    assert [row["source_sample_index"] for row in rows] == list(range(10))
    assert [row["source_frame"] for row in rows] == list(range(10))
    assert all(row["requested_sample_rate_hz"] is None for row in rows)
    assert all(row["fps"] is None for row in rows)
    assert all("future_source_metric" not in row for row in rows)
    assert validate_export_run(export_root, "baseline-samples-arrow")["status"] == "valid"


def test_manifest_selected_reader_rejects_rehashed_wrong_physical_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_name = POSITION_OCCUPANCY_HISTOGRAM_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_position_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr", source
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id="exact-arrow",
        tables=(table_name,),
        jobs=1,
    )
    assert validate_export_run(root, "exact-arrow")["status"] == "valid"

    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.read_table(part)
    column_index = original.schema.get_field_index("hist_count")
    columns = list(original.columns)
    columns[column_index] = pa.array([1.0], type=pa.float64())
    wrong_schema = original.schema.set(
        column_index,
        pa.field("hist_count", pa.float64(), nullable=False),
    )
    pq.write_table(pa.Table.from_arrays(columns, schema=wrong_schema), part)

    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="physical Arrow fields"):
        validate_export_run(root, "exact-arrow")


@pytest.mark.parametrize(
    "mutation",
    ("reordered", "wrong_type", "wrong_nullability", "unexpected", "missing", "metadata"),
)
def test_recording_summary_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    table_name = RECORDING_SUMMARY_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_recording_summary_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=f"recording-summary-{mutation}",
        tables=(table_name,),
        jobs=1,
    )
    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.ParquetFile(part).read()
    arrays = list(original.columns)
    fields = list(original.schema)
    metadata = dict(original.schema.metadata or {})

    if mutation == "reordered":
        arrays[0], arrays[1] = arrays[1], arrays[0]
        fields[0], fields[1] = fields[1], fields[0]
    elif mutation == "wrong_type":
        index = original.schema.get_field_index("stimulus_step_count")
        arrays[index] = pa.array([0.0], type=pa.float64())
        fields[index] = pa.field("stimulus_step_count", pa.float64(), nullable=False)
    elif mutation == "wrong_nullability":
        index = original.schema.get_field_index("recording_id")
        fields[index] = pa.field("recording_id", pa.string(), nullable=True)
    elif mutation == "unexpected":
        arrays.append(pa.array(["surprise"], type=pa.string()))
        fields.append(pa.field("unexpected", pa.string(), nullable=False))
    elif mutation == "missing":
        index = original.schema.get_field_index("source_lineage_hash")
        del arrays[index]
        del fields[index]
    elif mutation == "metadata":
        metadata[b"palette.arrow_schema_sha256"] = b"0" * 64
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    wrong_schema = pa.schema(fields, metadata=metadata)
    pq.write_table(pa.Table.from_arrays(arrays, schema=wrong_schema), part)

    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="physical Arrow fields|footer contract metadata",
    ):
        validate_export_run(root, f"recording-summary-{mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "reordered",
        "wrong_type",
        "wrong_nullability",
        "unexpected",
        "missing",
        "metadata",
    ),
)
def test_stimulus_steps_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    table_name = STIMULUS_STEPS_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_stimulus_step_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    run_id = f"stimulus-steps-{mutation}"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=run_id,
        tables=(table_name,),
        jobs=1,
    )
    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.ParquetFile(part).read()
    arrays = list(original.columns)
    fields = list(original.schema)
    metadata = dict(original.schema.metadata or {})

    if mutation == "reordered":
        arrays[0], arrays[1] = arrays[1], arrays[0]
        fields[0], fields[1] = fields[1], fields[0]
    elif mutation == "wrong_type":
        index = original.schema.get_field_index("step_index")
        arrays[index] = pa.array([0.0], type=pa.float64())
        fields[index] = pa.field("step_index", pa.float64(), nullable=False)
    elif mutation == "wrong_nullability":
        index = original.schema.get_field_index("recording_id")
        fields[index] = pa.field("recording_id", pa.string(), nullable=True)
    elif mutation == "unexpected":
        arrays.append(pa.array(["surprise"], type=pa.string()))
        fields.append(pa.field("unexpected", pa.string(), nullable=False))
    elif mutation == "missing":
        index = original.schema.get_field_index("source_lineage_hash")
        del arrays[index]
        del fields[index]
    elif mutation == "metadata":
        metadata[b"palette.arrow_schema_sha256"] = b"0" * 64
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)

    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata)),
        part,
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="physical Arrow fields|footer contract metadata",
    ):
        validate_export_run(root, run_id)


@pytest.mark.parametrize(
    "mutation",
    (
        "reordered",
        "wrong_type",
        "wrong_nullability",
        "unexpected",
        "missing",
        "metadata",
    ),
)
def test_stimulus_step_summary_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    table_name = STIMULUS_STEP_SUMMARY_TABLE

    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [_valid_stimulus_step_summary_row()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    run_id = f"stimulus-step-summary-{mutation}"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=run_id,
        tables=(table_name,),
        jobs=1,
    )
    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.ParquetFile(part).read()
    arrays = list(original.columns)
    fields = list(original.schema)
    metadata = dict(original.schema.metadata or {})

    if mutation == "reordered":
        arrays[0], arrays[1] = arrays[1], arrays[0]
        fields[0], fields[1] = fields[1], fields[0]
    elif mutation == "wrong_type":
        index = original.schema.get_field_index("fish_id")
        arrays[index] = pa.array([7.0], type=pa.float64())
        fields[index] = pa.field("fish_id", pa.float64(), nullable=False)
    elif mutation == "wrong_nullability":
        index = original.schema.get_field_index("recording_id")
        fields[index] = pa.field("recording_id", pa.string(), nullable=True)
    elif mutation == "unexpected":
        arrays.append(pa.array(["surprise"], type=pa.string()))
        fields.append(pa.field("unexpected", pa.string(), nullable=False))
    elif mutation == "missing":
        index = original.schema.get_field_index("source_lineage_hash")
        del arrays[index]
        del fields[index]
    elif mutation == "metadata":
        metadata[b"palette.arrow_schema_sha256"] = b"0" * 64
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata)),
        part,
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="physical Arrow fields|footer contract metadata",
    ):
        validate_export_run(root, run_id)


@pytest.mark.parametrize(
    "mutation",
    (
        "reordered",
        "wrong_type",
        "wrong_nullability",
        "unexpected",
        "missing",
        "metadata",
    ),
)
@pytest.mark.parametrize(
    ("table_name", "row_factory", "integer_field"),
    (
        (
            BASELINE_BEHAVIOR_SUMMARY_TABLE,
            _valid_baseline_summary_row,
            "total_frame_count",
        ),
        (
            BASELINE_BEHAVIOR_TIME_BINS_TABLE,
            _valid_baseline_time_bin_row,
            "expected_frame_count",
        ),
        (
            BASELINE_KINEMATIC_SAMPLES_TABLE,
            _valid_baseline_sample_row,
            "source_sample_index",
        ),
    ),
)
def test_baseline_tables_manifest_reader_rejects_rehashed_physical_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    table_name: str,
    row_factory: Any,
    integer_field: str,
) -> None:
    def source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id="recording-1",
            rows_by_table={table_name: [row_factory()]},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "source.zarr"],
        output_root=root,
        export_run_id=f"baseline-{table_name}-{mutation}",
        tables=(table_name,),
        jobs=1,
    )
    part = root / manifest["part_files_by_table"][table_name][0]
    original = pq.ParquetFile(part).read()
    arrays = list(original.columns)
    fields = list(original.schema)
    metadata = dict(original.schema.metadata or {})

    if mutation == "reordered":
        arrays[0], arrays[1] = arrays[1], arrays[0]
        fields[0], fields[1] = fields[1], fields[0]
    elif mutation == "wrong_type":
        index = original.schema.get_field_index(integer_field)
        arrays[index] = pa.array([1.0], type=pa.float64())
        fields[index] = pa.field(integer_field, pa.float64(), nullable=False)
    elif mutation == "wrong_nullability":
        index = original.schema.get_field_index("recording_id")
        fields[index] = pa.field("recording_id", pa.string(), nullable=True)
    elif mutation == "unexpected":
        arrays.append(pa.array(["surprise"], type=pa.string()))
        fields.append(pa.field("unexpected", pa.string(), nullable=False))
    elif mutation == "missing":
        index = original.schema.get_field_index("source_lineage_hash")
        del arrays[index]
        del fields[index]
    elif mutation == "metadata":
        metadata[b"palette.arrow_schema_sha256"] = b"0" * 64
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    pq.write_table(
        pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata)),
        part,
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["publication"]["parts_by_table"][table_name][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExportValidationError,
        match="physical Arrow fields|footer contract metadata",
    ):
        validate_export_run(root, f"baseline-{table_name}-{mutation}")
