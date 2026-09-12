"""Versioned bout-metric grains added to the full-rate core behavior export.

The native source dtype fingerprints freeze these three projections. A later
source schema change requires a new export profile, rather than silently
changing the meaning or Arrow bytes of this profile.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .arrow_contract_core import ArrowFieldContract, ArrowTableContract, field
from .validated_behavior_contracts import (
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)
from .validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_TABLE_SPECS,
)


BOUT_KINEMATICS_EXPORT_PROFILE_ID = "validated_core_behavior_bout_kinematics_v1"
BOUT_KINEMATICS_CAPABILITY_PROFILE_ID = "core_behavior_bout_kinematics_sources_v1"
BOUT_KINEMATICS_CAPABILITY = "bout_kinematics_metrics"
BOUT_KINEMATICS_CAPABILITY_KEYS = (
    *CORE_BEHAVIOR_CAPABILITY_KEYS,
    BOUT_KINEMATICS_CAPABILITY,
)
BOUT_MOVEMENT_TABLE = "bout_movement_metrics"
BOUT_HEADING_TABLE = "bout_heading_metrics"
BOUT_EYE_GAZE_TABLE = "bout_eye_gaze_metrics"

SOURCE_DTYPE_SHA256 = MappingProxyType(
    {
        "movement": "9c66a7802862e7ba946eaf58da148fb3143637f22cd1958bca6848ffb1f6cef6",
        "heading": "b7cff9e1cff25fbd31674545536358a921e52430ea6d9e95afbc29723d973d29",
        "eye_gaze": "5b3d08f5b990a1ae3e9fc1732425754ed8a72196fc583d1718a514a7e22997c1",
    }
)
_SOURCE_DTYPES = MappingProxyType(
    {
        "movement": np.dtype(
            [
                ("bout_id", "<i4"),
                ("source_start_frame", "<i8"),
                ("source_end_frame", "<i8"),
                ("source_core_start_frame", "<i8"),
                ("source_core_end_frame", "<i8"),
                ("detector_duration_s", "<f8"),
                ("detector_observed_duration_s", "<f8"),
                ("detector_core_duration_s", "<f8"),
                ("physical_active_start_frame", "<i8"),
                ("physical_active_end_frame", "<i8"),
                ("physical_active_start_time_s", "<f8"),
                ("physical_active_end_time_s", "<f8"),
                ("physical_active_duration_s", "<f8"),
                ("physical_active_observed_duration_s", "<f8"),
                ("physical_active_start_time_s_interpolated", "<f8"),
                ("physical_active_end_time_s_interpolated", "<f8"),
                ("physical_active_duration_s_interpolated", "<f8"),
                ("physical_active_start_time_interpolated_valid", "|b1"),
                ("physical_active_end_time_interpolated_valid", "|b1"),
                ("physical_active_sample_count", "<i4"),
                ("physical_active_valid_transition_count", "<i4"),
                ("physical_active_valid_transition_fraction", "<f8"),
                ("physical_active_path_length_mm", "<f8"),
                ("physical_active_path_length_px", "<f8"),
                ("physical_active_mean_speed_mm_s", "<f8"),
                ("physical_active_peak_speed_mm_s", "<f8"),
                ("physical_active_threshold_mm_s", "<f8"),
                ("physical_active_boundary_margin_s", "<f8"),
                ("physical_active_boundary_policy_bytes", "|S64"),
                ("physical_active_boundary_constraint_bytes", "|S64"),
                ("physical_active_valid", "|b1"),
                ("failure_reason_bytes", "|S256"),
            ]
        ),
        "heading": np.dtype(
            [
                ("bout_id", "<i4"),
                ("source_start_frame", "<i8"),
                ("source_end_frame", "<i8"),
                ("source_core_start_frame", "<i8"),
                ("source_core_end_frame", "<i8"),
                ("source_core_start_time_s_interpolated", "<f8"),
                ("source_core_end_time_s_interpolated", "<f8"),
                ("source_core_duration_s_interpolated", "<f8"),
                ("source_core_start_time_interpolated_valid", "|b1"),
                ("source_core_end_time_interpolated_valid", "|b1"),
                ("source_peak_frame", "<i8"),
                ("source_peak_time_s", "<f8"),
                ("source_peak_signal_value_mm_s", "<f8"),
                ("source_peak_prominence_mm_s", "<f8"),
                ("source_peak_width_s", "<f8"),
                ("source_peak_width_height_mm_s", "<f8"),
                ("source_peak_left_width_frame_interpolated", "<f8"),
                ("source_peak_right_width_frame_interpolated", "<f8"),
                ("source_peak_left_width_time_s", "<f8"),
                ("source_peak_right_width_time_s", "<f8"),
                ("source_peak_boundary_mode_bytes", "|S64"),
                ("source_peak_shape_split_policy_bytes", "|S64"),
                ("pre_epoch_start_frame", "<i8"),
                ("pre_epoch_end_frame", "<i8"),
                ("post_epoch_start_frame", "<i8"),
                ("post_epoch_end_frame", "<i8"),
                ("pre_heading_mean_deg", "<f8"),
                ("post_heading_mean_deg", "<f8"),
                ("net_delta_heading_deg", "<f8"),
                ("abs_net_delta_heading_deg", "<f8"),
                ("pre_position_mean_x_mm", "<f8"),
                ("pre_position_mean_y_mm", "<f8"),
                ("post_position_mean_x_mm", "<f8"),
                ("post_position_mean_y_mm", "<f8"),
                ("interbout_epoch_displacement_mm", "<f8"),
                ("pre_position_mean_x_px", "<f8"),
                ("pre_position_mean_y_px", "<f8"),
                ("post_position_mean_x_px", "<f8"),
                ("post_position_mean_y_px", "<f8"),
                ("interbout_epoch_displacement_px", "<f8"),
                ("within_heading_range_deg", "<f8"),
                ("within_heading_peak_to_peak_deg", "<f8"),
                ("within_heading_path_deg", "<f8"),
                ("within_heading_std_deg", "<f8"),
                ("within_heading_zero_crossings", "<i4"),
                ("within_heading_dominant_frequency_hz", "<f8"),
                ("within_angular_velocity_mean_deg_s", "<f8"),
                ("within_angular_speed_mean_deg_s", "<f8"),
                ("within_angular_speed_max_deg_s", "<f8"),
                ("within_angular_velocity_std_deg_s", "<f8"),
                ("pre_window_valid", "|b1"),
                ("post_window_valid", "|b1"),
                ("pre_position_valid", "|b1"),
                ("post_position_valid", "|b1"),
                ("within_window_valid", "|b1"),
                ("within_angular_velocity_valid", "|b1"),
                ("dominant_frequency_valid", "|b1"),
                ("pre_window_sample_count", "<i4"),
                ("post_window_sample_count", "<i4"),
                ("pre_position_sample_count", "<i4"),
                ("post_position_sample_count", "<i4"),
                ("within_window_sample_count", "<i4"),
                ("within_angular_velocity_transition_count", "<i4"),
                ("failure_reason_bytes", "|S256"),
            ]
        ),
        "eye_gaze": np.dtype(
            [
                ("bout_id", "<i4"),
                ("source_start_frame", "<i8"),
                ("source_end_frame", "<i8"),
                ("source_core_start_frame", "<i8"),
                ("source_core_end_frame", "<i8"),
                ("pre_epoch_start_frame", "<i8"),
                ("pre_epoch_end_frame", "<i8"),
                ("post_epoch_start_frame", "<i8"),
                ("post_epoch_end_frame", "<i8"),
                ("within_epoch_start_frame", "<i8"),
                ("within_epoch_end_frame", "<i8"),
                ("pre_left_gaze_mean_deg", "<f8"),
                ("pre_right_gaze_mean_deg", "<f8"),
                ("pre_vergence_gaze_mean_deg", "<f8"),
                ("pre_vergence_gaze_signed_mean_deg", "<f8"),
                ("pre_vergence_gaze_std_deg", "<f8"),
                ("pre_vergence_gaze_valid_fraction", "<f8"),
                ("pre_converged_fraction", "<f8"),
                ("post_left_gaze_mean_deg", "<f8"),
                ("post_right_gaze_mean_deg", "<f8"),
                ("post_vergence_gaze_mean_deg", "<f8"),
                ("post_vergence_gaze_signed_mean_deg", "<f8"),
                ("post_vergence_gaze_std_deg", "<f8"),
                ("post_vergence_gaze_valid_fraction", "<f8"),
                ("post_converged_fraction", "<f8"),
                ("within_bout_left_gaze_mean_deg", "<f8"),
                ("within_bout_right_gaze_mean_deg", "<f8"),
                ("within_bout_vergence_gaze_mean_deg", "<f8"),
                ("within_bout_vergence_gaze_signed_mean_deg", "<f8"),
                ("within_bout_vergence_gaze_max_deg", "<f8"),
                ("within_bout_vergence_gaze_range_deg", "<f8"),
                ("within_bout_vergence_gaze_std_deg", "<f8"),
                ("within_bout_vergence_gaze_valid_fraction", "<f8"),
                ("within_bout_converged_fraction", "<f8"),
                ("pre_eye_window_valid", "|b1"),
                ("post_eye_window_valid", "|b1"),
                ("within_eye_window_valid", "|b1"),
                ("pre_eye_sample_count", "<i4"),
                ("post_eye_sample_count", "<i4"),
                ("within_eye_sample_count", "<i4"),
                ("failure_reason_bytes", "|S256"),
            ]
        ),
    }
)
for _level, _dtype in _SOURCE_DTYPES.items():
    if canonical_json_sha256(_dtype.descr) != SOURCE_DTYPE_SHA256[_level]:
        raise ValueError(
            f"Native bout-kinematics {_level} dtype changed; install a new "
            "validated export profile instead of reinterpreting v1."
        )


def source_field_name(name: str) -> str:
    """Decode native fixed UTF-8 byte columns into plainly named strings."""

    return name.removesuffix("_bytes")


def source_dtype(level: str) -> np.dtype:
    return _SOURCE_DTYPES[level]


def _source_fields(level: str) -> tuple[ArrowFieldContract, ...]:
    dtype = source_dtype(level)
    result: list[ArrowFieldContract] = []
    for name in dtype.names or ():
        native_field = dtype[name]
        if native_field.kind == "S":
            arrow_type = "string"
        elif native_field.kind == "b":
            arrow_type = "bool"
        elif native_field.kind == "f" and native_field.itemsize == 8:
            arrow_type = "float64"
        elif native_field.kind == "i" and native_field.itemsize in (4, 8):
            arrow_type = f"int{native_field.itemsize * 8}"
        else:  # pragma: no cover - frozen dtype fingerprints guard this
            raise ValueError(f"Unsupported native bout metric dtype: {name}")
        result.append(field(source_field_name(name), arrow_type))
    return tuple(result)


_PROVENANCE_FIELDS = (
    field("export_run_id", "string"),
    field("recording_id", "string"),
    field("membership_member_sha256", "string"),
    field("bundle_set_member_sha256", "string"),
    field("bundle_record_sha256", "string"),
    field("cross_grain_join_authority_sha256", "string"),
    field("source_binding_sha256", "string"),
    field("projection_contract_sha256", "string"),
    field("source_bout_kinematics_run", "string"),
    field("source_bout_kinematics_path", "string"),
    field("source_array_manifest_sha256", "string"),
    field("source_metric_content_sha256", "string"),
    field("track_id", "int64"),
    field("source_signal_id", "int64"),
)
_BOUT_KEY = (
    "export_run_id",
    "recording_id",
    "track_id",
    "source_signal_id",
    "bout_id",
)
_BOUT_FK = (
    (
        ("export_run_id", "recording_id"),
        "cohort_recordings",
        ("export_run_id", "recording_id"),
    ),
    (_BOUT_KEY, "canonical_swim_bouts", _BOUT_KEY),
)


def _spec(
    table_name: str,
    *,
    level: str,
    heading_levels: bool = False,
) -> ValidatedBehaviorTableSpec:
    contract = ArrowTableContract(
        table_name=table_name,
        fields=(
            _PROVENANCE_FIELDS
            + ((field("heading_level", "string"),) if heading_levels else ())
            + _source_fields(level)
        ),
        primary_key=(
            _BOUT_KEY[:-1] + ("heading_level", "bout_id")
            if heading_levels
            else _BOUT_KEY
        ),
        schema_namespace=TABLE_SCHEMA_NAMESPACE,
    )
    return ValidatedBehaviorTableSpec(
        contract=contract,
        grain=(
            "one selected canonical bout and heading level"
            if heading_levels
            else "one selected canonical bout"
        ),
        capability_policy="required_all_admitted",
        required_capability=BOUT_KINEMATICS_CAPABILITY,
        foreign_keys=_BOUT_FK,
        zero_rows_allowed=True,
        primary_key_validation="strictly_increasing_v1",
        semantic_metadata=(
            ("publication_surface", "validated_behavior/v1"),
            ("source_projection", "palette.bout_kinematics_metrics.projection.v1"),
            ("measurement_family", level),
            ("source_layout", "compact_tabular_v2"),
            ("source_dtype_sha256", SOURCE_DTYPE_SHA256[level]),
            ("fixed_text_policy", "decode_nul_padded_utf8_strip_bytes_suffix"),
            ("invalid_float_semantics", "source_ieee_nan_not_arrow_null"),
            ("cross_grain_join_policy", "bundle_bound_join_authority_v1"),
        ),
    )


BOUT_KINEMATICS_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = (
    MappingProxyType(
        {
            BOUT_MOVEMENT_TABLE: _spec(BOUT_MOVEMENT_TABLE, level="movement"),
            BOUT_HEADING_TABLE: _spec(
                BOUT_HEADING_TABLE, level="heading", heading_levels=True
            ),
            BOUT_EYE_GAZE_TABLE: _spec(BOUT_EYE_GAZE_TABLE, level="eye_gaze"),
        }
    )
)
BOUT_KINEMATICS_EXPORT_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = (
    MappingProxyType({**CORE_BEHAVIOR_TABLE_SPECS, **BOUT_KINEMATICS_TABLE_SPECS})
)


__all__ = [
    "BOUT_EYE_GAZE_TABLE",
    "BOUT_HEADING_TABLE",
    "BOUT_KINEMATICS_CAPABILITY",
    "BOUT_KINEMATICS_CAPABILITY_KEYS",
    "BOUT_KINEMATICS_CAPABILITY_PROFILE_ID",
    "BOUT_KINEMATICS_EXPORT_PROFILE_ID",
    "BOUT_KINEMATICS_EXPORT_TABLE_SPECS",
    "BOUT_KINEMATICS_TABLE_SPECS",
    "BOUT_MOVEMENT_TABLE",
    "SOURCE_DTYPE_SHA256",
    "source_dtype",
    "source_field_name",
]
