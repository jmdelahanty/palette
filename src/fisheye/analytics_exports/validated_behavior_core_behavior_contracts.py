"""Five normalized core-behavior grains for ``validated_behavior/v1``.

This module installs table contracts only.  It does not define a publisher,
manifest, generation layout, selector, or command-line surface.  The generic
validated-behavior engine owns all of those mechanics, exactly as it does for
the existing recording-behavior profiles.

The v1 motion, eye, and tail fields preserve the established standalone export
projections.  The v2 core-motion projection additionally exposes persisted
physical derivatives and integrals with explicit semantics.  Cohort provenance
and one cross-grain join-authority digest are added without recomputing their
scientific values.  Subject body frames are a source-native grain because the
existing chaser ``body_frame_samples`` table carries session-timestamp and
provider-role semantics that these recordings do not possess.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from .arrow_contract_core import ArrowFieldContract, ArrowTableContract, field
from .arrow_contracts import ARROW_TABLE_CONTRACTS
from .contracts import (
    EYE_TRACE_SAMPLES_TABLE,
    KINEMATICS_SAMPLES_TABLE,
    TAIL_TRACE_SAMPLES_TABLE,
)
from .validated_behavior_contracts import (
    CORE_TABLE_SPECS,
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)
from .validated_behavior_phase_a_contracts import CANONICAL_SWIM_BOUTS

SUBJECT_BODY_FRAME_SAMPLES_TABLE = "subject_body_frame_samples"
CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1 = "core_behavior_five_grain_sources_v1"
CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1 = "validated_core_behavior_five_grain_v1"
CORE_BEHAVIOR_CAPABILITY_PROFILE_ID = "core_behavior_five_grain_sources_v2"
CORE_BEHAVIOR_EXPORT_PROFILE_ID = "validated_core_behavior_five_grain_v2"

CROSS_GRAIN_JOIN_AUTHORITY = "cross_grain_join_authority"
KINEMATICS_SAMPLES_CAPABILITY = "kinematics_samples"
SUBJECT_BODY_FRAME_CAPABILITY = "subject_body_frame_samples"
SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID = "subject_body_frame_samples_v1"
EYE_TRACE_CAPABILITY = "eye_trace_samples"
TAIL_TRACE_CAPABILITY = "tail_trace_samples"
CANONICAL_SWIM_BOUTS_CAPABILITY = "canonical_swim_bouts"

CORE_BEHAVIOR_CAPABILITY_KEYS = (
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
    EYE_TRACE_CAPABILITY,
    TAIL_TRACE_CAPABILITY,
    CANONICAL_SWIM_BOUTS_CAPABILITY,
)

_BUNDLE_PROVENANCE = (
    field("export_run_id", "string"),
    field("recording_id", "string"),
    field("membership_member_sha256", "string"),
    field("bundle_set_member_sha256", "string"),
    field("bundle_record_sha256", "string"),
    field("cross_grain_join_authority_sha256", "string"),
)
_STANDALONE_IDENTITY_FIELDS = frozenset(
    {"export_schema_version", "table_name", "recording_id"}
)
_RECORDING_FK = (
    (
        ("export_run_id", "recording_id"),
        "cohort_recordings",
        ("export_run_id", "recording_id"),
    ),
)


def _contract(
    table_name: str,
    fields: Sequence[ArrowFieldContract],
    *,
    primary_key: Sequence[str],
    schema_version: int = 1,
) -> ArrowTableContract:
    return ArrowTableContract(
        table_name=table_name,
        fields=tuple(fields),
        primary_key=tuple(primary_key),
        schema_namespace=TABLE_SCHEMA_NAMESPACE,
        schema_version=schema_version,
    )


def _standalone_projection_fields(table_name: str) -> tuple[ArrowFieldContract, ...]:
    return tuple(
        item
        for item in ARROW_TABLE_CONTRACTS[table_name].fields
        if item.name not in _STANDALONE_IDENTITY_FIELDS
    )


def _spec(
    contract: ArrowTableContract,
    *,
    grain: str,
    capability: str,
    projection_id: str,
) -> ValidatedBehaviorTableSpec:
    return ValidatedBehaviorTableSpec(
        contract=contract,
        grain=grain,
        capability_policy="required_all_admitted",
        required_capability=capability,
        foreign_keys=_RECORDING_FK,
        zero_rows_allowed=True,
        primary_key_validation="strictly_increasing_v1",
        semantic_metadata=(
            ("publication_surface", "validated_behavior/v1"),
            ("source_projection", projection_id),
            ("cross_grain_join_policy", "bundle_bound_join_authority_v1"),
        ),
    )


KINEMATICS_SAMPLES_V1 = _contract(
    KINEMATICS_SAMPLES_TABLE,
    _BUNDLE_PROVENANCE + _standalone_projection_fields(KINEMATICS_SAMPLES_TABLE),
    primary_key=(
        "export_run_id",
        "recording_id",
        "track_id",
        "track_sample_index",
    ),
)

CORE_MOTION_KINEMATICS_PROJECTION_FIELDS = (
    field("export_schema_version", "int32"),
    field("table_name", "string"),
    field("recording_id", "string"),
    field("zarr_path", "string"),
    field("source_lineage_hash", "string"),
    field("source_track_kinematics_scope", "string"),
    field("source_track_kinematics_run", "string"),
    field("source_track_kinematics_path", "string"),
    field("source_track_motion_manifest_schema_id", "string"),
    field("source_track_motion_manifest_schema_version", "int64"),
    field("source_track_motion_manifest_sha256", "string"),
    field("source_binding_sha256", "string"),
    field("projection_contract_sha256", "string"),
    field("motion_projection_profile_id", "string"),
    field("acceleration_source_speed_level", "string"),
    field("cumulative_path_distance_source_level", "string"),
    field("source_sample_rate_hz", "float64"),
    field("requested_sample_rate_hz", "float64"),
    field("sampling_stride_frames", "int64"),
    field("nominal_sample_rate_hz", "float64"),
    field("sampling_policy", "string"),
    field("position_coordinate_space", "string"),
    field("position_coordinate_descriptor_sha256", "string"),
    field("physical_authority_sha256", "string"),
    field("track_id", "int64"),
    field("track_sample_index", "int64"),
    field("source_acquisition_frame_index", "int64"),
    field("time_seconds", "float32"),
    field("source_row_index", "int64"),
    field("source_instance_key_valid", "bool"),
    field("source_instance_key", "uint64"),
    field("detection_source", "int8"),
    field("position_x_mm", "float32"),
    field("position_y_mm", "float32"),
    field("delta_frames", "int32"),
    field("delta_seconds", "float32"),
    field("speed_filtered_mm_s", "float32"),
    field("frame_path_distance_filtered_mm", "float32"),
    field("speed_smoothed_mm_s", "float32"),
    field("frame_path_distance_smoothed_mm", "float32"),
    field("signed_tangential_acceleration_mm_s2", "float32"),
    field("smoothed_signed_tangential_acceleration_mm_s2", "float32"),
    field("cumulative_smoothed_path_distance_mm", "float32"),
    field("motion_heading_degrees", "float32"),
    field("smoothed_motion_heading_degrees", "float32"),
    field("smoothed_angular_velocity_deg_s", "float32"),
    field("source_observed", "bool"),
    field("sample_observed", "bool"),
    field("position_finite", "bool"),
    field("heading_usable", "bool"),
    field("sample_valid", "bool"),
    field("transition_valid", "bool"),
    field("sample_reason_code", "int16"),
    field("transition_reason_code", "int16"),
)

KINEMATICS_SAMPLES = _contract(
    KINEMATICS_SAMPLES_TABLE,
    _BUNDLE_PROVENANCE
    + tuple(
        item
        for item in CORE_MOTION_KINEMATICS_PROJECTION_FIELDS
        if item.name not in _STANDALONE_IDENTITY_FIELDS
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "track_id",
        "track_sample_index",
    ),
    schema_version=2,
)

EYE_TRACE_SAMPLES = _contract(
    EYE_TRACE_SAMPLES_TABLE,
    _BUNDLE_PROVENANCE + _standalone_projection_fields(EYE_TRACE_SAMPLES_TABLE),
    primary_key=(
        "export_run_id",
        "recording_id",
        "source_acquisition_frame_index",
    ),
)

TAIL_TRACE_SAMPLES = _contract(
    TAIL_TRACE_SAMPLES_TABLE,
    _BUNDLE_PROVENANCE + _standalone_projection_fields(TAIL_TRACE_SAMPLES_TABLE),
    primary_key=(
        "export_run_id",
        "recording_id",
        "source_tail_row_index",
        "tail_sample_index",
    ),
)

SUBJECT_BODY_FRAME_SAMPLES = _contract(
    SUBJECT_BODY_FRAME_SAMPLES_TABLE,
    _BUNDLE_PROVENANCE
    + (
        field("zarr_path", "string"),
        field("source_lineage_hash", "string"),
        field("source_subject_shape_run", "string"),
        field("source_subject_shape_path", "string"),
        field("source_subject_shape_schema_id", "string"),
        field("source_subject_shape_schema_version", "int64"),
        field("source_subject_shape_publication_manifest_sha256", "string"),
        field("source_binding_sha256", "string"),
        field("projection_contract_sha256", "string"),
        field("row_identity_sha256", "string"),
        field("temporal_authority_sha256", "string"),
        field("acquisition_camera_frame_sha256", "string"),
        field("camera_id", "string"),
        field("source_sample_rate_hz", "float64"),
        field("body_frame_record_sha256", "string"),
        field("heading_semantics_sha256", "string"),
        field("origin_coordinate_descriptor_sha256", "string"),
        field("forward_coordinate_descriptor_sha256", "string"),
        field("left_coordinate_descriptor_sha256", "string"),
        field("subject_shape_row_index", "int64"),
        field("instance_key", "uint64"),
        field("source_crop_row_id", "int64"),
        field("source_acquisition_frame_index", "int64"),
        field("time_seconds", "float64"),
        field("origin_x_px", "float32"),
        field("origin_y_px", "float32"),
        field("forward_x", "float32"),
        field("forward_y", "float32"),
        field("left_x", "float32"),
        field("left_y", "float32"),
        field("heading_deg", "float32"),
        field("body_frame_valid", "bool"),
        field("failure_reason", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "subject_shape_row_index",
    ),
)


def _core_behavior_table_specs(
    kinematics_contract: ArrowTableContract,
    *,
    kinematics_projection_id: str,
) -> Mapping[str, ValidatedBehaviorTableSpec]:
    return MappingProxyType(
        {
            **CORE_TABLE_SPECS,
            KINEMATICS_SAMPLES_TABLE: _spec(
                kinematics_contract,
                grain="one row per selected track kinematics sample",
                capability=KINEMATICS_SAMPLES_CAPABILITY,
                projection_id=kinematics_projection_id,
            ),
            SUBJECT_BODY_FRAME_SAMPLES_TABLE: _spec(
                SUBJECT_BODY_FRAME_SAMPLES,
                grain="one row per canonical subject-shape body-frame observation",
                capability=SUBJECT_BODY_FRAME_CAPABILITY,
                projection_id="palette.subject_body_frame_samples.projection",
            ),
            EYE_TRACE_SAMPLES_TABLE: _spec(
                EYE_TRACE_SAMPLES,
                grain="one row per acquisition frame with both eyes, vergence, and gaze",
                capability=EYE_TRACE_CAPABILITY,
                projection_id="palette.eye_trace.projection",
            ),
            TAIL_TRACE_SAMPLES_TABLE: _spec(
                TAIL_TRACE_SAMPLES,
                grain="one row per subject observation and normalized tail sample",
                capability=TAIL_TRACE_CAPABILITY,
                projection_id="palette.tail_trace_samples.projection",
            ),
            "canonical_swim_bouts": _spec(
                CANONICAL_SWIM_BOUTS,
                grain="one row per exact selected-track canonical swim bout",
                capability=CANONICAL_SWIM_BOUTS_CAPABILITY,
                projection_id="palette.canonical_swim_bouts.projection",
            ),
        }
    )


CORE_BEHAVIOR_TABLE_SPECS_V1 = _core_behavior_table_specs(
    KINEMATICS_SAMPLES_V1,
    kinematics_projection_id="palette.kinematics_samples.projection",
)
CORE_BEHAVIOR_TABLE_SPECS = _core_behavior_table_specs(
    KINEMATICS_SAMPLES,
    kinematics_projection_id="palette.kinematics_samples.projection.v2",
)
CORE_BEHAVIOR_TABLE_NAMES = tuple(sorted(CORE_BEHAVIOR_TABLE_SPECS))


__all__ = [
    "CANONICAL_SWIM_BOUTS_CAPABILITY",
    "CORE_BEHAVIOR_CAPABILITY_KEYS",
    "CORE_BEHAVIOR_CAPABILITY_PROFILE_ID",
    "CORE_BEHAVIOR_CAPABILITY_PROFILE_ID_V1",
    "CORE_BEHAVIOR_EXPORT_PROFILE_ID",
    "CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1",
    "CORE_BEHAVIOR_TABLE_NAMES",
    "CORE_BEHAVIOR_TABLE_SPECS",
    "CORE_BEHAVIOR_TABLE_SPECS_V1",
    "CORE_MOTION_KINEMATICS_PROJECTION_FIELDS",
    "EYE_TRACE_SAMPLES",
    "EYE_TRACE_CAPABILITY",
    "KINEMATICS_SAMPLES_CAPABILITY",
    "KINEMATICS_SAMPLES",
    "KINEMATICS_SAMPLES_V1",
    "CROSS_GRAIN_JOIN_AUTHORITY",
    "SUBJECT_BODY_FRAME_CAPABILITY",
    "SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID",
    "SUBJECT_BODY_FRAME_SAMPLES",
    "SUBJECT_BODY_FRAME_SAMPLES_TABLE",
    "TAIL_TRACE_SAMPLES",
    "TAIL_TRACE_CAPABILITY",
]
