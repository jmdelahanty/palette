"""Dense exact-sample contracts for validated recording-behavior exports.

Phase B extends the compact Phase-A publication without changing any Phase-A
grain.  Every table is a lossless projection of one bundle-bound persisted
authority.  Physical motion estimators and the swim-bout detector response are
deliberately separate tables because they have different scientific meanings.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from .arrow_contract_core import ArrowFieldContract, ArrowTableContract, field
from .validated_behavior_contracts import (
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)
from .validated_behavior_phase_a_contracts import PHASE_A_TABLE_SPECS


PHASE_B_PROFILE_ID = "validated_recording_behavior_phase_b_v1"


def _contract(
    table_name: str,
    fields: Sequence[ArrowFieldContract],
    *,
    primary_key: Sequence[str],
) -> ArrowTableContract:
    return ArrowTableContract(
        table_name=table_name,
        fields=tuple(fields),
        primary_key=tuple(primary_key),
        schema_namespace=TABLE_SCHEMA_NAMESPACE,
    )


_BUNDLE_PROVENANCE = (
    field("export_run_id", "string"),
    field("recording_id", "string"),
    field("membership_member_sha256", "string"),
    field("bundle_set_member_sha256", "string"),
    field("bundle_record_sha256", "string"),
)

_CHILD_PROVENANCE = _BUNDLE_PROVENANCE + (
    field("source_child_key", "string"),
    field("source_run_path", "string"),
    field("source_manifest_sha256", "string"),
    field("source_payload_sha256", "string"),
    field("source_receipt_sha256", "string"),
)

_RECORDING_FK = (
    (
        ("export_run_id", "recording_id"),
        "cohort_recordings",
        ("export_run_id", "recording_id"),
    ),
)
_CHASER_FK = (
    (
        ("export_run_id", "recording_id", "chaser_identity_code"),
        "chaser_occurrences",
        ("export_run_id", "recording_id", "chaser_identity_code"),
    ),
)
_PROVIDER_FK = (
    (
        ("export_run_id", "recording_id", "provider_role"),
        "position_providers",
        ("export_run_id", "recording_id", "provider_role"),
    ),
)


PROVIDER_MOTION_SAMPLES = _contract(
    "provider_motion_samples",
    _BUNDLE_PROVENANCE
    + (
        field("source_binding_key", "string"),
        field("source_run_path", "string"),
        field("source_manifest_sha256", "string"),
        field("source_verification_digest", "string"),
        field("provider_role", "string"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
        field("track_id", "int64"),
        field("track_sample_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("source_observation_instance_key", "uint64"),
        field("source_provider_row_id", "int64"),
        field("source_position_row_id", "int64"),
        field("source_body_frame_row_id", "int64"),
        field("source_tracking_row_id", "int64"),
        field("time_s", "float32"),
        field("position_x_px", "float32"),
        field("position_y_px", "float32"),
        field("position_x_mm", "float32"),
        field("position_y_mm", "float32"),
        field("position_source_valid", "bool"),
        field("body_frame_source_valid", "bool"),
        field("linear_sample_valid", "bool"),
        field("linear_sample_reason_code", "int32"),
        field("angular_sample_valid", "bool"),
        field("angular_sample_reason_code", "int32"),
        field("heading_deg", "float32"),
        field("heading_rad", "float32"),
        field("smoothed_heading_deg", "float32"),
        field("smoothed_heading_rad", "float32"),
        field("delta_frames", "int32"),
        field("delta_s", "float32"),
        field("transition_valid", "bool"),
        field("transition_reason_code", "int32"),
        field("speed_raw_px_s", "float32"),
        field("speed_filtered_px_s", "float32"),
        field("speed_smoothed_px_s", "float32"),
        field("speed_averaged_px_s", "float32"),
        field("speed_raw_mm_s", "float32"),
        field("speed_filtered_mm_s", "float32"),
        field("speed_smoothed_mm_s", "float32"),
        field("speed_averaged_mm_s", "float32"),
        field("acceleration_px_s2", "float32"),
        field("smoothed_acceleration_px_s2", "float32"),
        field("acceleration_mm_s2", "float32"),
        field("smoothed_acceleration_mm_s2", "float32"),
        field("frame_path_distance_raw_px", "float32"),
        field("frame_path_distance_filtered_px", "float32"),
        field("frame_path_distance_smoothed_px", "float32"),
        field("cumulative_path_distance_px", "float32"),
        field("frame_path_distance_raw_mm", "float32"),
        field("frame_path_distance_filtered_mm", "float32"),
        field("frame_path_distance_smoothed_mm", "float32"),
        field("cumulative_path_distance_mm", "float32"),
        field("delta_heading_deg", "float32"),
        field("angular_velocity_raw_deg_s", "float32"),
        field("angular_speed_raw_deg_s", "float32"),
        field("delta_heading_smoothed_deg", "float32"),
        field("angular_velocity_smoothed_deg_s", "float32"),
        field("angular_speed_smoothed_deg_s", "float32"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "track_id",
        "track_sample_row_id",
    ),
)


BOUT_DETECTOR_SIGNAL_SAMPLES = _contract(
    "bout_detector_signal_samples",
    _BUNDLE_PROVENANCE
    + (
        field("source_binding_key", "string"),
        field("swim_bout_run_path", "string"),
        field("swim_bout_lineage_sha256", "string"),
        field("source_track_motion_manifest_sha256", "string"),
        field("source_track_motion_verification_digest", "string"),
        field("track_id", "int64"),
        field("candidate_id", "int32"),
        field("signal_id", "int32"),
        field("signal_level", "string"),
        field("signal_name", "string"),
        field("signal_role", "string"),
        field("source_level", "string", nullable=True),
        field("signal_sample_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("detection_signal_mm_s", "float32"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "track_id",
        "candidate_id",
        "signal_id",
        "signal_sample_row_id",
    ),
)


STIMULUS_NATIVE_STATE_SUPPORT = _contract(
    "stimulus_native_state_support",
    _CHILD_PROVENANCE
    + (
        field("temporal_proxy_role", "string"),
        field("temporal_proxy_run_path", "string"),
        field("temporal_proxy_manifest_sha256", "string"),
        field("temporal_proxy_verification_digest", "string"),
        field("acquisition_projection_record_sha256", "string"),
        field("source_stimulus_run_path", "string"),
        field("source_stimulus_manifest_sha256", "string"),
        field("source_stimulus_verification_digest", "string"),
        field("projection_policy_id", "string"),
        field("scientific_use_class", "string"),
        field("physical_presentation_verified", "bool"),
        field("acquisition_frame_id", "int64"),
        field("candidate_ordinal_within_frame", "int32"),
        field("native_sample_row_id", "int64"),
        field("stimulus_frame_num", "int64"),
        field("timestamp_ns_session", "int64"),
        field("source_acquisition_frame_id", "int64"),
        field("candidate_complete", "bool"),
        field("candidate_reason_code", "int32"),
        field("candidate_reason", "string"),
        field("selected_candidate", "bool"),
        field("chaser_identity_code", "int32"),
        field("chaser_index", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("source_stimulus_run_row_id", "int64"),
        field("source_stimulus_source_row_id", "int64"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "temporal_proxy_role",
        "acquisition_frame_id",
        "native_sample_row_id",
        "chaser_identity_code",
    ),
)


CHASER_RELATIVE_SAMPLES = _contract(
    "chaser_relative_samples",
    _CHILD_PROVENANCE
    + (
        field("provider_role", "string"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
        field("relative_frame_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("track_sample_id", "int64"),
        field("timestamp_ns_session", "int64"),
        field("timestamp_valid", "bool"),
        field("timestamp_reason_code", "int32"),
        field("fish_source_row_id", "int64"),
        field("fish_source_row_valid", "bool"),
        field("fish_source_row_reason_code", "int32"),
        field("chaser_source_row_id", "int64"),
        field("chaser_source_row_valid", "bool"),
        field("chaser_source_row_reason_code", "int32"),
        field("fish_position_x_px", "float32"),
        field("fish_position_y_px", "float32"),
        field("fish_position_valid", "bool"),
        field("fish_position_reason_code", "int32"),
        field("chaser_position_x_px", "float32"),
        field("chaser_position_y_px", "float32"),
        field("chaser_position_valid", "bool"),
        field("chaser_position_reason_code", "int32"),
        field("fish_identity_code", "int32"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("chaser_behavior_role_code", "int32"),
        field("behavior_role", "string"),
        field("chaser_behavior_role_valid", "bool"),
        field("chaser_behavior_role_reason_code", "int32"),
        field("selection_member", "bool"),
        field("chaser_occurrence_member", "bool"),
        field("trial_id", "int64", nullable=True),
        field("trial_valid", "bool"),
        field("trial_reason_code", "int32"),
        field("active_state", "string", nullable=True),
        field("active_state_valid", "bool"),
        field("active_state_reason_code", "int32"),
        field("row_valid", "bool"),
        field("row_reason_code", "int32"),
        field("acquisition_frame_delta", "int64"),
        field("timestamp_delta_ns", "int64"),
        field("fish_transition_valid", "bool"),
        field("fish_transition_reason_code", "int32"),
        field("relative_transition_valid", "bool"),
        field("relative_transition_reason_code", "int32"),
        field("relative_vector_x_px", "float32"),
        field("relative_vector_y_px", "float32"),
        field("relative_distance_px", "float32"),
        field("relative_px_valid", "bool"),
        field("relative_px_reason_code", "int32"),
        field("relative_vector_x_mm", "float32"),
        field("relative_vector_y_mm", "float32"),
        field("relative_distance_mm", "float32"),
        field("relative_physical_valid", "bool"),
        field("relative_physical_reason_code", "int32"),
        field("nearest_chaser_member", "bool"),
        field("nearest_chaser_identity_code", "int32"),
        field("nearest_chaser_source_row_id", "int64"),
        field("nearest_chaser_distance_px", "float32"),
        field("nearest_chaser_distance_mm", "float32"),
        field("nearest_chaser_valid", "bool"),
        field("nearest_chaser_reason_code", "int32"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role",
        "relative_frame_row_id",
    ),
)


BODY_FRAME_SAMPLES = _contract(
    "body_frame_samples",
    _CHILD_PROVENANCE
    + (
        field("provider_role", "string"),
        field("body_frame_provider_id", "string"),
        field("body_frame_provider_digest", "string"),
        field("acquisition_frame_id", "int64"),
        field("track_sample_id", "int64"),
        field("timestamp_ns_session", "int64"),
        field("timestamp_valid", "bool"),
        field("body_source_row_id", "int64"),
        field("body_source_row_valid", "bool"),
        field("body_source_row_reason_code", "int32"),
        field("body_origin_x_px", "float32"),
        field("body_origin_y_px", "float32"),
        field("body_forward_x", "float32"),
        field("body_forward_y", "float32"),
        field("body_left_x", "float32"),
        field("body_left_y", "float32"),
        field("body_origin_valid", "bool"),
        field("body_origin_reason_code", "int32"),
        field("body_axes_valid", "bool"),
        field("body_axes_reason_code", "int32"),
        field("body_heading_deg", "float32"),
        field("body_heading_valid", "bool"),
        field("body_heading_reason_code", "int32"),
        field("body_heading_transition_valid", "bool"),
        field("body_heading_transition_reason_code", "int32"),
        field("body_valid", "bool"),
        field("body_reason_code", "int32"),
    ),
    primary_key=("export_run_id", "recording_id", "acquisition_frame_id"),
)


BODY_RELATIVE_SAMPLES = _contract(
    "body_relative_samples",
    _CHILD_PROVENANCE
    + (
        field("body_alignment_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("epoch_window_id", "int64", nullable=True),
        field("epoch_role", "string", nullable=True),
        field("selection_member", "bool"),
        field("chaser_occurrence_member", "bool"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("chaser_behavior_role_code", "int32"),
        field("behavior_role", "string", nullable=True),
        field("chaser_behavior_role_valid", "bool"),
        field("relative_distance_mm", "float64"),
        field("relative_physical_valid", "bool"),
        field("relative_physical_reason_code", "int32"),
        field("body_source_row_id", "int64"),
        field("body_source_row_valid", "bool"),
        field("body_heading_deg", "float64"),
        field("body_heading_valid", "bool"),
        field("body_heading_reason_code", "int32"),
        field("body_bearing_deg", "float64"),
        field("body_bearing_valid", "bool"),
        field("body_bearing_reason_code", "int32"),
        field("alignment_cos", "float64"),
        field("lateral_sin", "float64"),
        field("alignment_valid", "bool"),
        field("alignment_reason_code", "int32"),
    ),
    primary_key=("export_run_id", "recording_id", "body_alignment_row_id"),
)


CONTROLLER_TRIAL_MEMBERSHIP = _contract(
    "controller_trial_membership",
    _CHILD_PROVENANCE
    + (
        field("source_relative_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("trial_row_id", "int64"),
        field("logged_trial_id", "int64"),
        field("logged_active_trial_id_unavailable", "bool"),
    ),
    primary_key=("export_run_id", "recording_id", "source_relative_row_id"),
)


CONTROLLER_TRIAL_GAP_EVIDENCE = _contract(
    "controller_trial_gap_evidence",
    _CHILD_PROVENANCE
    + (
        field("source_relative_row_id", "int64"),
        field("acquisition_frame_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("trial_envelope_row_id", "int64"),
        field("logged_trial_id", "int64"),
        field("trial_gap_reason_code", "int32"),
        field("trial_gap_reason", "string"),
        field("logged_active_trial_id_unavailable", "bool"),
    ),
    primary_key=("export_run_id", "recording_id", "source_relative_row_id"),
)


def _dense_spec(
    contract: ArrowTableContract,
    *,
    grain: str,
    capability: str,
    foreign_keys: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...],
    semantic_metadata: tuple[tuple[str, str], ...],
) -> ValidatedBehaviorTableSpec:
    return ValidatedBehaviorTableSpec(
        contract=contract,
        grain=grain,
        capability_policy="optional_explicit_coverage",
        required_capability=capability,
        foreign_keys=foreign_keys,
        zero_rows_allowed=True,
        primary_key_validation="strictly_increasing_v1",
        semantic_metadata=semantic_metadata,
    )


_DENSE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {
        "provider_motion_samples": _dense_spec(
            PROVIDER_MOTION_SAMPLES,
            grain="recording x provider x exact track-sample row",
            capability="provider_motion",
            foreign_keys=_RECORDING_FK + _PROVIDER_FK,
            semantic_metadata=(
                ("row_axis", "bundle_selected_provider_motion_track_partition"),
                ("position_units", "explicit_px_and_mm_columns"),
                ("speed_units", "explicit_px_per_s_and_mm_per_s_columns"),
                ("validity", "persisted_source_validity_and_reason_codes"),
            ),
        ),
        "bout_detector_signal_samples": _dense_spec(
            BOUT_DETECTOR_SIGNAL_SAMPLES,
            grain="recording x exact selected bout-detector signal x sample row",
            capability="canonical_swim_bouts",
            foreign_keys=_RECORDING_FK,
            semantic_metadata=(
                ("row_axis", "exact_bound_swim_bout_frame_axis"),
                ("signal_role", "detector_response_not_physical_speed_estimator"),
                ("value_units", "source_field_detection_signal_mm_s"),
            ),
        ),
        "stimulus_native_state_support": _dense_spec(
            STIMULUS_NATIVE_STATE_SUPPORT,
            grain=(
                "recording x temporal proxy x acquisition frame x native sample x chaser"
            ),
            capability="chaser_relative_keypoint",
            foreign_keys=_RECORDING_FK + _CHASER_FK,
            semantic_metadata=(
                ("row_axis", "all_native_candidates_preserved_per_proxy_frame"),
                ("timing", "logged_session_timestamp_not_camera_presentation_time"),
                ("presentation_claim", "physical_presentation_unverified"),
                ("multiplicity", "native_samples_are_not_collapsed"),
            ),
        ),
        "chaser_relative_samples": _dense_spec(
            CHASER_RELATIVE_SAMPLES,
            grain="recording x position provider x acquisition frame x chaser",
            capability="chaser_relative_keypoint",
            foreign_keys=_RECORDING_FK + _PROVIDER_FK + _CHASER_FK,
            semantic_metadata=(
                ("row_axis", "frame_major_chaser_minor_per_provider"),
                ("coordinate_space", "source_camera_continuous_pixel_xy"),
                ("physical_space", "calibrated_length_xy_mm"),
                ("interpolation", "prohibited"),
            ),
        ),
        "body_frame_samples": _dense_spec(
            BODY_FRAME_SAMPLES,
            grain="recording x acquisition frame anatomical body frame",
            capability="anatomical_body_frame",
            foreign_keys=_RECORDING_FK + _PROVIDER_FK,
            semantic_metadata=(
                ("row_axis", "one_deduplicated_body_frame_per_acquisition_frame"),
                ("coordinate_space", "source_camera_continuous_pixel_xy"),
                ("heading_units", "degrees"),
                ("fallback", "motion_heading_substitution_prohibited"),
            ),
        ),
        "body_relative_samples": _dense_spec(
            BODY_RELATIVE_SAMPLES,
            grain="recording x acquisition frame x chaser anatomical alignment row",
            capability="body_alignment_by_distance",
            foreign_keys=(
                *_RECORDING_FK,
                *_CHASER_FK,
                (
                    ("export_run_id", "recording_id", "acquisition_frame_id"),
                    "body_frame_samples",
                    ("export_run_id", "recording_id", "acquisition_frame_id"),
                ),
            ),
            semantic_metadata=(
                ("bearing_convention", "positive_toward_anatomical_left_degrees"),
                ("alignment", "cos_body_bearing_and_sin_body_bearing"),
                ("distance_units", "mm"),
                ("interpolation", "prohibited"),
            ),
        ),
        "controller_trial_membership": _dense_spec(
            CONTROLLER_TRIAL_MEMBERSHIP,
            grain="recording x exact logged active relative-frame row",
            capability="controller_trials",
            foreign_keys=(
                *_RECORDING_FK,
                *_CHASER_FK,
                (
                    ("export_run_id", "recording_id", "trial_row_id"),
                    "controller_trials",
                    ("export_run_id", "recording_id", "trial_row_id"),
                ),
            ),
            semantic_metadata=(
                ("membership", "exact_logged_active_rows_only"),
                ("legacy_reconstruction", "prohibited"),
                ("gaps", "excluded_here_and_preserved_in_gap_evidence"),
            ),
        ),
        "controller_trial_gap_evidence": _dense_spec(
            CONTROLLER_TRIAL_GAP_EVIDENCE,
            grain="recording x nonmember row inside exact controller-trial envelope",
            capability="controller_trials",
            foreign_keys=(
                *_RECORDING_FK,
                *_CHASER_FK,
                (
                    ("export_run_id", "recording_id", "trial_envelope_row_id"),
                    "controller_trials",
                    ("export_run_id", "recording_id", "trial_row_id"),
                ),
            ),
            semantic_metadata=(
                ("membership", "evidence_rows_are_not_trial_members"),
                ("gap_identity", "exact_persisted_trial_envelope_and_reason"),
                ("legacy_reconstruction", "prohibited"),
            ),
        ),
    }
)

PHASE_B_DENSE_TABLE_SPECS = _DENSE_SPECS
PHASE_B_DENSE_TABLE_NAMES = tuple(_DENSE_SPECS)
PHASE_B_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {**PHASE_A_TABLE_SPECS, **_DENSE_SPECS}
)


__all__ = [
    "PHASE_B_DENSE_TABLE_NAMES",
    "PHASE_B_DENSE_TABLE_SPECS",
    "PHASE_B_PROFILE_ID",
    "PHASE_B_TABLE_SPECS",
]
