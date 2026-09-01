"""Exact compact-table contracts for validated recording-behavior bundles.

This profile is deliberately bundle-shaped rather than protocol-shaped.  It
copies only facts already sealed by a validated recording bundle and its exact
scientific children.  No table in this module reconstructs frame membership,
re-bins a persisted distribution, or chooses a position provider.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .arrow_contract_core import ArrowFieldContract, ArrowTableContract, field
from .validated_behavior_contracts import (
    CORE_TABLE_SPECS,
    TABLE_SCHEMA_NAMESPACE,
    ValidatedBehaviorTableSpec,
)

PHASE_A_PROFILE_ID = "validated_recording_behavior_phase_a_v1"


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


def _typed(
    names: Iterable[str], arrow_type: str, *, nullable: bool = False
) -> tuple[ArrowFieldContract, ...]:
    return tuple(field(name, arrow_type, nullable=nullable) for name in names)


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


RECORDING_SOURCE_BINDINGS = _contract(
    "recording_source_bindings",
    _BUNDLE_PROVENANCE
    + (
        field("source_binding_key", "string"),
        field("binding_type", "string"),
        field("binding_record_sha256", "string"),
        field("binding_json", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "source_binding_key"),
)

POSITION_PROVIDERS = _contract(
    "position_providers",
    _BUNDLE_PROVENANCE
    + _typed(
        (
            "provider_role",
            "position_provider_id",
            "position_provider_digest",
            "source_authority_id",
            "source_authority_digest",
            "coordinate_authority_id",
            "row_axis_authority_id",
            "row_axis_authority_digest",
            "timing_authority_id",
            "scale_authority_id",
        ),
        "string",
    ),
    primary_key=("export_run_id", "recording_id", "provider_role"),
)

CHASER_OCCURRENCES = _contract(
    "chaser_occurrences",
    _BUNDLE_PROVENANCE
    + (
        field("chaser_identity_code", "int32"),
        field("chaser_index", "int32"),
    )
    + _typed(
        (
            "chaser_identity",
            "behavior_role",
            "stimulus_run_path",
            "source_protocol_sha256",
            "chaser_identity_policy_id",
            "occurrence_policy_id",
            "occurrence_semantics",
        ),
        "string",
    ),
    primary_key=("export_run_id", "recording_id", "chaser_identity_code"),
)

SEMANTIC_EPOCHS = _contract(
    "semantic_epochs",
    _CHILD_PROVENANCE
    + (
        field("epoch_window_id", "int64"),
        field("analysis_role", "string"),
        field("source_label", "string"),
        field("start_frame", "int64"),
        field("end_frame_exclusive", "int64"),
        field("source_interval_sha256", "string"),
        field("protocol_semantic_hash", "string"),
        field("protocol_semantic_step_index", "int32"),
        field("protocol_semantic_step_ref", "string"),
        field("terminal_frame_excluded_pending_step_end_contract", "bool"),
        field("selection_identity_sha256", "string"),
        field("source_epoch_selection_sha256", "string"),
        field("step_end_interval_semantics", "string"),
        field("trial_index_integrity_status", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "epoch_window_id"),
)

CONTROLLER_TRIALS = _contract(
    "controller_trials",
    _CHILD_PROVENANCE
    + (
        field("trial_row_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("logged_trial_id", "int64"),
        field("trial_ordinal", "int32"),
        field("start_source_frame_row", "int64"),
        field("end_source_frame_row_exclusive", "int64"),
        field("start_acquisition_frame_id", "int64"),
        field("end_acquisition_frame_id_inclusive", "int64"),
        field("trigger_acquisition_frame_id", "int64"),
        field("trigger_timestamp_ns", "int64"),
        field("trigger_timestamp_valid", "bool"),
        field("trigger_source_code", "int32"),
        field("trigger_source", "string"),
        field("active_member_count", "int64"),
        field("envelope_frame_count", "int64"),
        field("gap_frame_count", "int64"),
        field("gap_fraction", "float64"),
        field("fallback_used", "bool"),
    ),
    primary_key=("export_run_id", "recording_id", "trial_row_id"),
)

CANONICAL_SWIM_BOUTS = _contract(
    "canonical_swim_bouts",
    _CHILD_PROVENANCE
    + (
        field("swim_bout_run_path", "string"),
        field("swim_bout_lineage_sha256", "string"),
        field("track_id", "int64"),
        field("source_signal_id", "int32"),
        field("bout_id", "int64"),
        field("bout_row_id", "int64"),
        field("start_acquisition_frame_id", "int64"),
        field("end_acquisition_frame_id", "int64"),
        field("duration_s", "float64"),
        field("path_length_mm", "float64"),
        field("net_displacement_mm", "float64"),
        field("mean_speed_mm_s", "float64"),
        field("peak_speed_mm_s", "float64"),
        field("tortuosity", "float64"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "track_id",
        "source_signal_id",
        "bout_id",
    ),
)

BOUT_CHASER_ASSOCIATIONS = _contract(
    "bout_chaser_associations",
    _CHILD_PROVENANCE
    + (
        field("bout_chaser_row_id", "int64"),
        field("track_id", "int64"),
        field("source_signal_id", "int32"),
        field("bout_id", "int64"),
        field("bout_row_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("semantic_role_code", "int32"),
        field("semantic_role", "string", nullable=True),
        field("start_acquisition_frame_id", "int64"),
        field("end_acquisition_frame_id", "int64"),
        field("base_valid", "bool"),
        field("attachment_reason_code", "int32"),
        field("attachment_reason", "string"),
        field("distance_at_onset_mm", "float64"),
        field("distance_at_end_mm", "float64"),
        field("delta_distance_mm", "float64"),
        field("directed_valid", "bool"),
        field("bearing_at_onset_deg", "float64"),
        field("turn_deg", "float64"),
        field("turn_toward_chaser", "bool"),
        field("controller_trial_row_id", "int64"),
        field("controller_trial_envelope_row_id", "int64"),
        field("controller_trial_gap_reason_code", "int32"),
        field("controller_trial_gap_reason", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "bout_chaser_row_id"),
)

BOUT_RESPONSE_DISTANCE_BINS = _contract(
    "bout_response_distance_bins",
    _CHILD_PROVENANCE
    + (
        field("semantic_role_code", "int32"),
        field("semantic_role", "string"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("distance_bin_index", "int32"),
        field("distance_bin_start_mm", "float64"),
        field("distance_bin_end_mm", "float64"),
        field("bout_count", "int64"),
        field("valid_time_s", "float64"),
        field("bout_rate_per_min", "float64"),
        field("median_duration_s", "float64"),
        field("median_path_length_mm", "float64"),
        field("median_net_displacement_mm", "float64"),
        field("median_peak_speed_mm_s", "float64"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "semantic_role_code",
        "chaser_identity_code",
        "distance_bin_index",
    ),
)

TRIAL_ESCAPE_FREEZE_EVENTS = _contract(
    "trial_escape_freeze_events",
    _CHILD_PROVENANCE
    + (
        field("event_row_id", "int64"),
        field("source_bout_chaser_row_id", "int64"),
        field("bout_row_id", "int64"),
        field("bout_id", "int64"),
        field("controller_trial_row_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("onset_acquisition_frame_id", "int64"),
        field("peak_speed_mm_s", "float64"),
        field("distance_at_onset_mm", "float64"),
        field("trigger_distance_mm", "float64"),
        field("latency_from_trigger_s", "float64"),
        field("separation_gain_mm", "float64"),
        field("turn_deg", "float64"),
        field("directed_valid", "bool"),
        field("high_turn", "bool"),
        field("recaptured", "bool"),
        field("recapture_latency_s", "float64"),
        field("trace_valid", "bool"),
        field("trace_exclusion_reason_code", "int32"),
        field("trace_exclusion_reason", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "event_row_id"),
)

TRIAL_ESCAPE_FREEZE_SUMMARIES = _contract(
    "trial_escape_freeze_summaries",
    _CHILD_PROVENANCE
    + (
        field("trial_row_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("logged_trial_id", "int64"),
        field("trial_ordinal", "int32"),
        field("trigger_acquisition_frame_id", "int64"),
        field("trigger_distance_mm", "float64"),
        field("valid_time_s", "float64"),
        field("envelope_frame_count", "int64"),
        field("gap_frame_count", "int64"),
        field("gap_fraction", "float64"),
        field("logged_active_id_unavailable_count", "int64"),
        field("bout_count", "int64"),
        field("escape_event_count", "int64"),
        field("high_turn_escape_count", "int64"),
        field("escape_event_rate_per_min", "float64"),
        field("first_escape_latency_s", "float64"),
        field("mean_separation_gain_mm", "float64"),
        field("recapture_fraction", "float64"),
        field("freeze_low_speed_fraction", "float64"),
        field("freeze_valid_fraction", "float64"),
        field("escape_speed_class", "bool"),
        field("freeze_candidate", "bool"),
        field("response_class_code", "int32"),
        field("response_class", "string"),
        field("speed_level", "string"),
        field("freeze_window_s", "float64"),
        field("freeze_speed_threshold_mm_s", "float64"),
        field("escape_speed_threshold_mm_s", "float64"),
        field("signal_provenance_status", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "trial_row_id"),
)

TRIAL_ESCAPE_FREEZE_THRESHOLD_SWEEPS = _contract(
    "trial_escape_freeze_threshold_sweeps",
    _CHILD_PROVENANCE
    + (
        field("sweep_row_id", "int64"),
        field("trial_row_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role", "string"),
        field("speed_threshold_mm_s", "float64"),
        field("escape_event_count", "int64"),
        field("escape_event_rate_per_min", "float64"),
    ),
    primary_key=("export_run_id", "recording_id", "sweep_row_id"),
)

EPOCH_BEHAVIOR_SUMMARY = _contract(
    "epoch_behavior_summary",
    _CHILD_PROVENANCE
    + (
        field("track_id", "int64"),
        field("epoch_window_id", "int64"),
        field("window_index", "int32"),
        field("window_label", "string"),
        field("analysis_role", "string"),
        field("start_frame", "int64"),
        field("end_frame", "int64"),
        field("start_time_s", "float64"),
        field("end_time_s", "float64"),
        field("duration_s", "float64"),
        field("total_span_frames", "int64"),
        field("provider_sample_count", "int64"),
        field("valid_tracked_frame_count", "int64"),
        field("missing_frame_count", "int64"),
        field("tracking_dropout_fraction", "float64"),
        field("valid_tracked_duration_s", "float64"),
        field("motion_valid_sample_count", "int64"),
        field("speed_sample_count", "int64"),
        field("mean_speed_mm_s", "float64"),
        field("median_speed_mm_s", "float64"),
        field("p05_speed_mm_s", "float64"),
        field("p95_speed_mm_s", "float64"),
        field("max_speed_mm_s", "float64"),
        field("total_path_mm", "float64"),
        field("bout_count", "int64"),
        field("bout_rate_per_min", "float64"),
        field("median_bout_duration_s", "float64"),
        field("mean_bout_duration_s", "float64"),
        field("median_bout_path_length_mm", "float64"),
        field("mean_bout_path_length_mm", "float64"),
        field("bout_heading_sample_count", "int64"),
        field("mean_bout_net_heading_change_deg", "float64"),
        field("median_bout_net_heading_change_deg", "float64"),
        field("mean_abs_bout_net_heading_change_deg", "float64"),
        field("median_abs_bout_net_heading_change_deg", "float64"),
        field("mean_bout_heading_path_deg", "float64"),
        field("median_bout_heading_path_deg", "float64"),
        field("inter_bout_interval_count", "int64"),
        field("mean_inter_bout_interval_s", "float64"),
        field("median_inter_bout_interval_s", "float64"),
        field("p05_inter_bout_interval_s", "float64"),
        field("p95_inter_bout_interval_s", "float64"),
        field("inter_bout_interval_rate_per_min", "float64"),
        field("rate_denominator", "string"),
        field("motion_validity_rule", "string"),
        field("source_interval_sha256", "string"),
        field("protocol_semantic_hash", "string"),
        field("protocol_semantic_step_index", "int32"),
        field("protocol_semantic_step_ref", "string"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
    ),
    primary_key=("export_run_id", "recording_id", "track_id", "epoch_window_id"),
)

SPATIAL_OCCUPANCY_SUPPORT = _contract(
    "spatial_occupancy_support",
    _CHILD_PROVENANCE
    + (
        field("provider_role_code", "int32"),
        field("provider_role", "string"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
        field("epoch_role_code", "int32"),
        field("epoch_role", "string"),
        field("epoch_window_id", "int64"),
        field("epoch_start_frame", "int64"),
        field("epoch_end_frame_exclusive", "int64"),
        field("candidate_frame_count", "int64"),
        field("declared_valid_position_frame_count", "int64"),
        field("finite_valid_position_frame_count", "int64"),
        field("invalid_position_frame_count", "int64"),
        field("in_arena_position_frame_count", "int64"),
        field("out_of_arena_position_frame_count", "int64"),
        field("in_arena_coverage_fraction_candidate", "float64"),
        field("in_arena_fraction_finite_valid", "float64"),
        field("grid_recipe_sha256", "string"),
        field("arena_authority_sha256", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role_code",
        "epoch_window_id",
    ),
)

SPATIAL_OCCUPANCY_BINS = _contract(
    "spatial_occupancy_bins",
    _CHILD_PROVENANCE
    + (
        field("provider_role_code", "int32"),
        field("provider_role", "string"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
        field("epoch_role_code", "int32"),
        field("epoch_role", "string"),
        field("epoch_window_id", "int64"),
        field("x_bin_index", "int32"),
        field("y_bin_index", "int32"),
        field("x_bin_start_mm", "float64"),
        field("x_bin_end_mm", "float64"),
        field("y_bin_start_mm", "float64"),
        field("y_bin_end_mm", "float64"),
        field("arena_bin_center_member", "bool"),
        field("occupancy_count", "int64"),
        field("occupancy_density_valid_in_arena", "float64"),
        field("occupancy_fraction_candidate_epoch", "float64"),
        field("grid_recipe_sha256", "string"),
        field("arena_authority_sha256", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role_code",
        "epoch_window_id",
        "x_bin_index",
        "y_bin_index",
    ),
)

_RADIAL_IDENTITY_FIELDS = (
    field("provider_role", "string"),
    field("position_provider_id", "string"),
    field("position_provider_digest", "string"),
    field("epoch_role_code", "int32"),
    field("epoch_role", "string"),
    field("epoch_window_id", "int64"),
    field("behavior_role_code", "int32"),
    field("behavior_role", "string"),
    field("chaser_identity_code", "int32"),
    field("chaser_identity", "string"),
)

RADIAL_NEAR_FIELD_SUMMARY = _contract(
    "radial_near_field_summary",
    _CHILD_PROVENANCE
    + _RADIAL_IDENTITY_FIELDS
    + _typed(
        (
            "candidate_frame_count",
            "valid_distance_frame_count",
            "wall_excluded_valid_frame_count",
            "near_zone_frame_count",
            "near_zone_entry_count",
            "near_zone_censor_event_count",
            "near_zone_boundary_censor_event_count",
            "near_zone_invalid_gap_count",
            "near_zone_invalid_gap_censor_event_count",
        ),
        "int64",
    )
    + _typed(
        (
            "valid_distance_fraction",
            "distance_mean_mm",
            "distance_p05_mm",
            "distance_p25_mm",
            "distance_p50_mm",
            "distance_p75_mm",
            "distance_p95_mm",
            "fish_arena_radius_mean_mm",
            "fish_arena_radius_p50_mm",
            "fish_wall_distance_mean_mm",
            "fish_wall_distance_p50_mm",
            "near_zone_fraction_candidate",
            "near_zone_fraction_valid",
            "near_zone_dwell_s",
            "near_zone_valid_tracked_duration_s",
            "near_zone_entry_rate_per_min_valid_time",
            "near_zone_complete_visit_total_dwell_s",
            "near_zone_complete_visit_median_dwell_s",
            "near_zone_expected_fraction_geometric",
            "near_zone_enrichment_geometric",
            "near_zone_radius_mm",
            "near_entry_radius_mm",
            "near_exit_radius_mm",
            "perimeter_band_mm",
            "min_expected_count",
        ),
        "float64",
    )
    + (
        field("radial_policy_sha256", "string"),
        field("arena_authority_sha256", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role",
        "epoch_window_id",
        "chaser_identity_code",
    ),
)

SAME_QUADRANT_OCCUPANCY = _contract(
    "same_quadrant_occupancy",
    _CHILD_PROVENANCE
    + _RADIAL_IDENTITY_FIELDS
    + (
        field("candidate_frame_count", "int64"),
        field("valid_distance_frame_count", "int64"),
        field("same_quadrant_valid_frame_count", "int64"),
        field("same_quadrant_fraction_candidate", "float64"),
        field("same_quadrant_fraction_valid", "float64"),
        field("quadrant_policy", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role",
        "epoch_window_id",
        "chaser_identity_code",
    ),
)

RADIAL_NEAR_FIELD_DENSITY_BINS = _contract(
    "radial_near_field_density_bins",
    _CHILD_PROVENANCE
    + _RADIAL_IDENTITY_FIELDS
    + (
        field("radial_bin_index", "int32"),
        field("radial_bin_start_mm", "float64"),
        field("radial_bin_end_mm", "float64"),
        field("observed_count", "int64"),
        field("observed_fraction", "float64"),
        field("expected_available_area_mm2_frames", "float64"),
        field("expected_fraction_geometric", "float64"),
        field("selection_index_geometric", "float64"),
        field("wall_excluded_observed_count", "int64"),
        field("wall_excluded_observed_fraction", "float64"),
        field("wall_excluded_expected_available_area_mm2_frames", "float64"),
        field("wall_excluded_expected_fraction_geometric", "float64"),
        field("wall_excluded_selection_index_geometric", "float64"),
        field("radial_policy_sha256", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role",
        "epoch_window_id",
        "chaser_identity_code",
        "radial_bin_index",
    ),
)

RADIAL_NEAR_FIELD_DISTANCE_CDF = _contract(
    "radial_near_field_distance_cdf",
    _CHILD_PROVENANCE
    + _RADIAL_IDENTITY_FIELDS
    + (
        field("threshold_index", "int32"),
        field("threshold_mm", "float64"),
        field("fraction_at_or_below", "float64"),
        field("cdf_policy", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "provider_role",
        "epoch_window_id",
        "chaser_identity_code",
        "threshold_index",
    ),
)

BODY_ALIGNMENT_DISTANCE_BINS = _contract(
    "body_alignment_distance_bins",
    _CHILD_PROVENANCE
    + (
        field("epoch_role_code", "int32"),
        field("epoch_role", "string"),
        field("epoch_window_id", "int64"),
        field("chaser_identity_code", "int32"),
        field("chaser_identity", "string"),
        field("behavior_role_code", "int32"),
        field("behavior_role", "string"),
        field("distance_bin_index", "int32"),
        field("distance_bin_start_mm", "float64"),
        field("distance_bin_end_mm", "float64"),
        field("distance_bin_center_mm", "float64"),
        field("candidate_row_count", "int64"),
        field("joint_valid_row_count", "int64"),
        field("epoch_occurrence_row_count", "int64"),
        field("epoch_distance_valid_row_count", "int64"),
        field("epoch_distance_invalid_row_count", "int64"),
        field("epoch_distance_invalid_body_valid_row_count", "int64"),
        field("epoch_chaser_absent_row_count", "int64"),
        field("body_source_missing_row_count", "int64"),
        field("body_heading_invalid_row_count", "int64"),
        field("body_bearing_invalid_row_count", "int64"),
        field("other_alignment_invalid_row_count", "int64"),
        field("mean_alignment_cos", "float64"),
        field("alignment_cos_p25", "float64"),
        field("alignment_cos_p50", "float64"),
        field("alignment_cos_p75", "float64"),
        field("mean_abs_bearing_deg", "float64"),
        field("abs_bearing_p25_deg", "float64"),
        field("abs_bearing_p50_deg", "float64"),
        field("abs_bearing_p75_deg", "float64"),
        field("circular_mean_bearing_deg", "float64"),
        field("circular_resultant_length", "float64"),
        field("position_provider_id", "string"),
        field("position_provider_digest", "string"),
        field("body_frame_provider_id", "string"),
        field("body_frame_provider_digest", "string"),
        field("angle_convention_id", "string"),
        field("distance_bin_recipe_sha256", "string"),
    ),
    primary_key=(
        "export_run_id",
        "recording_id",
        "epoch_window_id",
        "chaser_identity_code",
        "distance_bin_index",
    ),
)


def _spec(
    contract: ArrowTableContract,
    *,
    grain: str,
    capability: str,
    policy: str = "required_all_admitted",
    foreign_keys: tuple[
        tuple[tuple[str, ...], str, tuple[str, ...]], ...
    ] = _RECORDING_FK,
) -> ValidatedBehaviorTableSpec:
    return ValidatedBehaviorTableSpec(
        contract=contract,
        grain=grain,
        capability_policy=policy,
        required_capability=capability,
        foreign_keys=foreign_keys,
        zero_rows_allowed=True,
    )


_EPOCH_FK = (
    (
        ("export_run_id", "recording_id", "epoch_window_id"),
        "semantic_epochs",
        ("export_run_id", "recording_id", "epoch_window_id"),
    ),
)

_CHASER_FK = (
    (
        ("export_run_id", "recording_id", "chaser_identity_code"),
        "chaser_occurrences",
        ("export_run_id", "recording_id", "chaser_identity_code"),
    ),
)

_TRIAL_FK = (
    (
        ("export_run_id", "recording_id", "trial_row_id"),
        "controller_trials",
        ("export_run_id", "recording_id", "trial_row_id"),
    ),
)


PHASE_A_TABLE_SPECS: Mapping[str, ValidatedBehaviorTableSpec] = MappingProxyType(
    {
        **CORE_TABLE_SPECS,
        "recording_source_bindings": _spec(
            RECORDING_SOURCE_BINDINGS,
            grain="one row per exact source binding in a complete recording bundle",
            capability="semantic_epochs",
        ),
        "position_providers": _spec(
            POSITION_PROVIDERS,
            grain="one row per exact first-class fish-position provider",
            capability="reviewed_arena_and_scale",
        ),
        "chaser_occurrences": _spec(
            CHASER_OCCURRENCES,
            grain="one row per stimulus-run-scoped chaser occurrence",
            capability="chaser_relative_keypoint",
        ),
        "semantic_epochs": _spec(
            SEMANTIC_EPOCHS,
            grain="one row per exact half-open semantic window",
            capability="semantic_epochs",
        ),
        "controller_trials": _spec(
            CONTROLLER_TRIALS,
            grain="one row per exact logged controller trial and chaser",
            capability="controller_trials",
            foreign_keys=_RECORDING_FK + _CHASER_FK,
        ),
        "canonical_swim_bouts": _spec(
            CANONICAL_SWIM_BOUTS,
            grain="one row per exact selected-track canonical swim bout",
            capability="canonical_swim_bouts",
        ),
        "bout_chaser_associations": _spec(
            BOUT_CHASER_ASSOCIATIONS,
            grain="one row per canonical swim bout and chaser occurrence",
            capability="generalized_bout_response",
            foreign_keys=_RECORDING_FK
            + _CHASER_FK
            + (
                (
                    (
                        "export_run_id",
                        "recording_id",
                        "track_id",
                        "source_signal_id",
                        "bout_id",
                    ),
                    "canonical_swim_bouts",
                    (
                        "export_run_id",
                        "recording_id",
                        "track_id",
                        "source_signal_id",
                        "bout_id",
                    ),
                ),
            ),
        ),
        "bout_response_distance_bins": _spec(
            BOUT_RESPONSE_DISTANCE_BINS,
            grain="one row per semantic role, chaser, and persisted distance bin",
            capability="generalized_bout_response",
            foreign_keys=_RECORDING_FK + _CHASER_FK,
        ),
        "trial_escape_freeze_events": _spec(
            TRIAL_ESCAPE_FREEZE_EVENTS,
            grain="one row per persisted trial-locked escape event",
            capability="escape_freeze",
            foreign_keys=_RECORDING_FK
            + _CHASER_FK
            + (
                (
                    (
                        "export_run_id",
                        "recording_id",
                        "source_bout_chaser_row_id",
                    ),
                    "bout_chaser_associations",
                    ("export_run_id", "recording_id", "bout_chaser_row_id"),
                ),
            ),
        ),
        "trial_escape_freeze_summaries": _spec(
            TRIAL_ESCAPE_FREEZE_SUMMARIES,
            grain="one row per exact controller trial and chaser response summary",
            capability="escape_freeze",
            foreign_keys=_RECORDING_FK + _CHASER_FK + _TRIAL_FK,
        ),
        "trial_escape_freeze_threshold_sweeps": _spec(
            TRIAL_ESCAPE_FREEZE_THRESHOLD_SWEEPS,
            grain="one row per exact trial and persisted speed-threshold sweep value",
            capability="escape_freeze",
            foreign_keys=_RECORDING_FK + _CHASER_FK + _TRIAL_FK,
        ),
        "epoch_behavior_summary": _spec(
            EPOCH_BEHAVIOR_SUMMARY,
            grain="one row per selected track and exact semantic epoch",
            capability="epoch_behavior",
            foreign_keys=_RECORDING_FK + _EPOCH_FK,
        ),
        "spatial_occupancy_support": _spec(
            SPATIAL_OCCUPANCY_SUPPORT,
            grain="one row per position provider and semantic epoch support",
            capability="spatial_occupancy",
            foreign_keys=_RECORDING_FK + _EPOCH_FK,
        ),
        "spatial_occupancy_bins": _spec(
            SPATIAL_OCCUPANCY_BINS,
            grain="one row per position provider, semantic epoch, and persisted xy bin",
            capability="spatial_occupancy",
            foreign_keys=_RECORDING_FK + _EPOCH_FK,
        ),
        "radial_near_field_summary": _spec(
            RADIAL_NEAR_FIELD_SUMMARY,
            grain="one row per position provider, semantic epoch, and chaser",
            capability="reviewed_arena_and_scale",
            foreign_keys=_RECORDING_FK + _EPOCH_FK + _CHASER_FK,
        ),
        "same_quadrant_occupancy": _spec(
            SAME_QUADRANT_OCCUPANCY,
            grain="one row per position provider, semantic epoch, and chaser",
            capability="reviewed_arena_and_scale",
            foreign_keys=_RECORDING_FK + _EPOCH_FK + _CHASER_FK,
        ),
        "radial_near_field_density_bins": _spec(
            RADIAL_NEAR_FIELD_DENSITY_BINS,
            grain="one row per provider, epoch, chaser, and persisted radial bin",
            capability="reviewed_arena_and_scale",
            foreign_keys=_RECORDING_FK + _EPOCH_FK + _CHASER_FK,
        ),
        "radial_near_field_distance_cdf": _spec(
            RADIAL_NEAR_FIELD_DISTANCE_CDF,
            grain="one row per provider, epoch, chaser, and persisted CDF threshold",
            capability="reviewed_arena_and_scale",
            foreign_keys=_RECORDING_FK + _EPOCH_FK + _CHASER_FK,
        ),
        "body_alignment_distance_bins": _spec(
            BODY_ALIGNMENT_DISTANCE_BINS,
            grain="one row per epoch, chaser, and persisted anatomical distance bin",
            capability="body_alignment_by_distance",
            policy="optional_explicit_coverage",
            foreign_keys=_RECORDING_FK + _EPOCH_FK + _CHASER_FK,
        ),
    }
)

PHASE_A_TABLE_NAMES = tuple(sorted(PHASE_A_TABLE_SPECS))


__all__ = [
    "PHASE_A_PROFILE_ID",
    "PHASE_A_TABLE_NAMES",
    "PHASE_A_TABLE_SPECS",
]
