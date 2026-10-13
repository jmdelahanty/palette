"""Session-aware acquisition clocks for the full behavior export.

This profile extends the immutable bout-kinematics v1 profile.  Clock samples
stay normalized in their own table, while one recording-level metadata row
declares the source, clock domains, and the limits of cross-recording joins.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .arrow_contract_core import ArrowTableContract, field
from .validated_behavior_bout_kinematics_contracts import (
    BOUT_KINEMATICS_CAPABILITY_KEYS,
    BOUT_KINEMATICS_EXPORT_TABLE_SPECS,
)
from .validated_behavior_contracts import (
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)

FRAME_CLOCK_EXPORT_PROFILE_ID = "validated_core_behavior_bout_kinematics_frame_clock_v1"
FRAME_CLOCK_CAPABILITY_PROFILE_ID = (
    "core_behavior_bout_kinematics_frame_clock_sources_v1"
)
ACQUISITION_FRAME_CLOCK_CAPABILITY = "acquisition_frame_clock"
FRAME_CLOCK_CAPABILITY_KEYS = (
    *BOUT_KINEMATICS_CAPABILITY_KEYS,
    ACQUISITION_FRAME_CLOCK_CAPABILITY,
)

RECORDING_CLOCK_METADATA_TABLE = "recording_clock_metadata"
ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE = "acquisition_frame_clock_samples"

_PROVENANCE_FIELDS = (
    field("export_run_id", "string"),
    field("recording_id", "string"),
    field("membership_member_sha256", "string"),
    field("bundle_set_member_sha256", "string"),
    field("bundle_record_sha256", "string"),
    field("cross_grain_join_authority_sha256", "string"),
    field("source_binding_sha256", "string"),
    field("projection_contract_sha256", "string"),
)
_RECORDING_KEY = ("export_run_id", "recording_id")
_RECORDING_FK = ((_RECORDING_KEY, "cohort_recordings", _RECORDING_KEY),)

RECORDING_CLOCK_METADATA = ArrowTableContract(
    table_name=RECORDING_CLOCK_METADATA_TABLE,
    fields=_PROVENANCE_FIELDS
    + (
        field("session_id", "string"),
        field("session_start_iso8601_utc", "string"),
        field("camera_id", "string"),
        field("source_frame_count", "int64"),
        field("raw_recording_path", "string"),
        field("frame_clock_source_path", "string"),
        field("frame_clock_source_file_sha256", "string"),
        field("recording_manifest_path", "string"),
        field("recording_manifest_file_sha256", "string"),
        field("ptp_sync_summary_path", "string", nullable=True),
        field("ptp_sync_summary_file_sha256", "string", nullable=True),
        field("acquisition_camera_frame_sha256", "string"),
        field("source_video_metadata_sha256", "string"),
        field("acquisition_frame_clock_source_sha256", "string"),
        field("clock_semantics_sha256", "string"),
        field("camera_clock_domain", "string"),
        field("camera_time_reference_kind", "string"),
        field("camera_time_origin", "string"),
        field("camera_timescale", "string"),
        field("camera_semantic_status", "string"),
        field("system_clock_domain", "string"),
        field("system_time_reference_kind", "string"),
        field("system_time_origin", "string"),
        field("system_timescale", "string"),
        field("system_semantic_status", "string"),
        field("within_session_alignment_status", "string"),
        field("cross_session_alignment_status", "string"),
        field("equal_frame_rate_alignment_valid", "bool"),
    ),
    primary_key=_RECORDING_KEY,
    schema_namespace=TABLE_SCHEMA_NAMESPACE,
)

ACQUISITION_FRAME_CLOCK_SAMPLES = ArrowTableContract(
    table_name=ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE,
    fields=_PROVENANCE_FIELDS
    + (
        field("session_id", "string"),
        field("camera_id", "string"),
        field("source_acquisition_frame_index", "int64"),
        field("recording_frame_id", "int64"),
        field("camera_timestamp_ns", "int64"),
        field("camera_timestamp_valid", "bool"),
        field("system_timestamp_ns", "int64"),
        field("system_timestamp_valid", "bool"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "source_acquisition_frame_index",
    ),
    schema_namespace=TABLE_SCHEMA_NAMESPACE,
)

_METADATA_SPEC = ValidatedBehaviorTableSpec(
    contract=RECORDING_CLOCK_METADATA,
    grain="one clock and session semantics record per admitted recording",
    capability_policy="required_all_admitted",
    required_capability=ACQUISITION_FRAME_CLOCK_CAPABILITY,
    foreign_keys=_RECORDING_FK,
    zero_rows_allowed=True,
    primary_key_validation="strictly_increasing_v1",
    semantic_metadata=(
        ("publication_surface", "validated_behavior/v1"),
        ("source_projection", "palette.acquisition_frame_clock.export_metadata.v1"),
        ("session_identity", "orange_session_id"),
        (
            "within_recording_join",
            "recording_id_and_source_acquisition_frame_index",
        ),
        (
            "within_session_cross_camera_join",
            "equal_session_id_and_valid_camera_timestamp_ns_with_declared_tolerance",
        ),
        (
            "cross_session_join",
            "requires_separately_validated_traceable_absolute_clock_or_external_anchor",
        ),
        ("equal_frame_rate_alignment", "invalid_without_clock_or_anchor"),
    ),
)

_SAMPLES_SPEC = ValidatedBehaviorTableSpec(
    contract=ACQUISITION_FRAME_CLOCK_SAMPLES,
    grain="one acquisition clock observation per recording camera frame",
    capability_policy="required_all_admitted",
    required_capability=ACQUISITION_FRAME_CLOCK_CAPABILITY,
    foreign_keys=(
        *_RECORDING_FK,
        (
            _RECORDING_KEY,
            RECORDING_CLOCK_METADATA_TABLE,
            _RECORDING_KEY,
        ),
    ),
    zero_rows_allowed=True,
    primary_key_validation="strictly_increasing_v1",
    semantic_metadata=(
        ("publication_surface", "validated_behavior/v1"),
        ("source_projection", "palette.acquisition_frame_clock.samples.v1"),
        ("frame_index_source", "parent_frame_index"),
        ("camera_timestamp_unit", "nanosecond"),
        ("system_timestamp_unit", "nanosecond"),
        (
            "missing_timestamp_policy",
            "int64_sentinel_is_invalid_unless_corresponding_valid_flag_is_true",
        ),
    ),
)

FRAME_CLOCK_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {
        RECORDING_CLOCK_METADATA_TABLE: _METADATA_SPEC,
        ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE: _SAMPLES_SPEC,
    }
)
FRAME_CLOCK_EXPORT_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = (
    MappingProxyType({**BOUT_KINEMATICS_EXPORT_TABLE_SPECS, **FRAME_CLOCK_TABLE_SPECS})
)


__all__ = [
    "ACQUISITION_FRAME_CLOCK_CAPABILITY",
    "ACQUISITION_FRAME_CLOCK_SAMPLES",
    "ACQUISITION_FRAME_CLOCK_SAMPLES_TABLE",
    "FRAME_CLOCK_CAPABILITY_KEYS",
    "FRAME_CLOCK_CAPABILITY_PROFILE_ID",
    "FRAME_CLOCK_EXPORT_PROFILE_ID",
    "FRAME_CLOCK_EXPORT_TABLE_SPECS",
    "FRAME_CLOCK_TABLE_SPECS",
    "RECORDING_CLOCK_METADATA",
    "RECORDING_CLOCK_METADATA_TABLE",
]
