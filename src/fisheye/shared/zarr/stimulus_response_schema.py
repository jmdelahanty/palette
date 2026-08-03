"""Exact logical and physical array contract for stimulus-response v3 runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT8,
    INT32,
    INT64,
    UINT8,
    ArrayContract,
    DTypeContract,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode

STIMULUS_RESPONSE_SCHEMA_ID = "palette.stimulus_response"
STIMULUS_RESPONSE_SCHEMA_VERSION = 3
STIMULUS_RESPONSE_LAYOUT = "compact_tabular_v3"
STIMULUS_RESPONSE_ARRAY_SCHEMA_ID = "palette.stimulus_response.compact_tabular_arrays"
STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION = 1
STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION = 2
STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR = "stimulus_response_array_schema"
STIMULUS_RESPONSE_BUNDLES_ATTR = "stimulus_response_v3_bundles"
STIMULUS_RESPONSE_PHYSICAL_POLICY_OWNER = "stimulus_response_compact_tabular_v3"
STIMULUS_RESPONSE_BYTE_PLANNER_POLICY_OWNER = "analysis_storage_planning_v1"
STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR = "analysis_storage_profile_role"
STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE = "explicit_unpromoted_candidate"

_BYTE_PLANNED_ROW_AXES = {
    "grating_per_fish": "grating_fish_rows",
    "moving_grating_omr_per_fish": "moving_grating_omr_fish_rows",
    "moving_grating_omr_per_bout": "moving_grating_omr_bout_rows",
    "moving_grating_omr_windows": "moving_grating_omr_window_rows",
    "moving_grating_omr_early_windows": "moving_grating_omr_early_window_rows",
    "concentric_per_fish": "concentric_fish_rows",
    "concentric_radial_omr_per_fish": "concentric_radial_omr_fish_rows",
    "concentric_radial_omr_per_bout": "concentric_radial_omr_bout_rows",
    "concentric_radial_omr_windows": "concentric_radial_omr_window_rows",
    "concentric_radial_omr_early_windows": "concentric_radial_omr_early_window_rows",
    "looming_trials": "looming_trial_rows",
    "looming_per_trial_per_fish": "looming_trial_fish_rows",
    "looming_per_fish": "looming_fish_rows",
}

_CANDIDATE_REQUIRED_ATTRS = frozenset(
    {
        STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR,
        "analysis_storage_profile_id",
        "analysis_storage_plan_receipt",
        "analysis_storage_plan_payload_sha256",
        "stimulus_response_storage_candidate",
    }
)
_CANDIDATE_EQUIVALENCE_ATTR = "analysis_storage_direct_consolidated_equivalence"


@dataclass(frozen=True)
class StimulusResponseField:
    dtype: DTypeContract
    units: str | None = None
    string_width: int | None = None
    description: str = "Stimulus-response table field."

    def __post_init__(self) -> None:
        if self.string_width is not None:
            if (
                self.dtype is not UINT8
                or type(self.string_width) is not int
                or self.string_width <= 0
            ):
                raise ValueError(
                    "Fixed text fields require uint8 and a positive exact width."
                )

    @property
    def logical_dtype(self) -> np.dtype:
        if self.string_width is not None:
            return np.dtype(f"S{self.string_width}")
        return np.dtype(self.dtype.numpy_dtype)


@dataclass(frozen=True)
class StimulusResponseTable:
    name: str
    row_axis: str
    fields: tuple[tuple[str, StimulusResponseField], ...]
    required: bool = False
    bundle: str | None = None

    @property
    def field_map(self) -> dict[str, StimulusResponseField]:
        return dict(self.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(name for name, _field in self.fields)


def _f(dtype: DTypeContract, units: str | None = None) -> StimulusResponseField:
    return StimulusResponseField(dtype=dtype, units=units)


def _s(width: int) -> StimulusResponseField:
    return StimulusResponseField(dtype=UINT8, string_width=width, units="utf8")


STEP_INDEX = (
    ("step_index", _f(INT32)),
    ("step_name", _s(128)),
    ("stimulus_mode", _s(64)),
    ("stimulus_mode_id", _f(INT32)),
    ("start_frame", _f(INT64, "camera_frame")),
    ("end_frame", _f(INT64, "camera_frame")),
    ("duration_s", _f(FLOAT32, "s")),
)
STEP_JOIN = (("step_index", _f(INT32)),)
GLOBAL_FIELDS = (
    ("fish_id", _f(INT32)),
    ("total_distance_mm", _f(FLOAT32, "mm")),
    ("mean_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("total_active_s", _f(FLOAT32, "s")),
    ("fraction_moving", _f(FLOAT32, "fraction")),
)
STEP_PER_FISH_BASE = (
    ("fish_id", _f(INT32)),
    ("total_distance_mm", _f(FLOAT32, "mm")),
    ("mean_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("median_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("max_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("fraction_moving", _f(FLOAT32, "fraction")),
    ("coverage", _f(FLOAT32, "fraction")),
)
STEP_BOUT_SUMMARY = (
    ("num_bouts", _f(INT32)),
    ("mean_bout_duration_s", _f(FLOAT32, "s")),
    ("mean_interbout_interval_s", _f(FLOAT32, "s")),
)
STEP_PER_BOUT = (
    ("fish_id", _f(INT32)),
    ("bout_id", _f(INT32)),
    ("start_frame", _f(INT64, "camera_frame")),
    ("end_frame", _f(INT64, "camera_frame")),
    ("duration_s", _f(FLOAT32, "s")),
    ("mean_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("peak_physical_speed_mm_s", _f(FLOAT32, "mm/s")),
)
GRATING_PER_FISH = (
    ("fish_id", _f(INT32)),
    ("mean_alignment_cos", _f(FLOAT32)),
    ("resultant_vector_length", _f(FLOAT32)),
    ("fraction_following", _f(FLOAT32, "fraction")),
    ("fraction_opposing", _f(FLOAT32, "fraction")),
    ("fraction_perpendicular", _f(FLOAT32, "fraction")),
    ("speed_weighted_alignment", _f(FLOAT32)),
    ("optomotor_gain", _f(FLOAT32)),
    ("drift_along_grating_mm", _f(FLOAT32, "mm")),
    ("drift_perp_grating_mm", _f(FLOAT32, "mm")),
    ("latency_to_follow_s", _f(FLOAT32, "s")),
)
CONCENTRIC_PER_FISH = (
    ("fish_id", _f(INT32)),
    ("mean_distance_to_center_mm", _f(FLOAT32, "mm")),
    ("initial_distance_to_center_mm", _f(FLOAT32, "mm")),
    ("final_distance_to_center_mm", _f(FLOAT32, "mm")),
    ("min_distance_to_center_mm", _f(FLOAT32, "mm")),
    ("net_radial_displacement_mm", _f(FLOAT32, "mm")),
    ("fraction_approaching", _f(FLOAT32, "fraction")),
    ("mean_radial_heading_cos", _f(FLOAT32)),
    ("time_to_center_s", _f(FLOAT32, "s")),
    ("fraction_near_center", _f(FLOAT32, "fraction")),
    ("mean_radial_speed_mm_s", _f(FLOAT32, "mm/s")),
    ("mean_tangential_speed_mm_s", _f(FLOAT32, "mm/s")),
)


def _typed(
    names: Sequence[str],
    *,
    int32: Sequence[str] = (),
    int64: Sequence[str] = (),
    int8: Sequence[str] = (),
    boolean: Sequence[str] = (),
) -> tuple[tuple[str, StimulusResponseField], ...]:
    i32, i64, i8, bo = set(int32), set(int64), set(int8), set(boolean)
    out = []
    for name in names:
        dtype = (
            INT32
            if name in i32
            else INT64
            if name in i64
            else INT8
            if name in i8
            else BOOL
            if name in bo
            else FLOAT32
        )
        out.append((name, _f(dtype)))
    return tuple(out)


MOVING_OMR_PER_FISH_NAMES = (
    "fish_id",
    "omr_path_index",
    "omr_net_direction_index",
    "parallel_displacement_mm",
    "net_displacement_mm",
    "path_length_mm",
    "valid_transition_count",
    "coverage_fraction",
    "bout_fraction_correct_classified",
    "bout_fraction_correct_all",
    "bout_choice_index",
    "bout_path_index",
    "bout_fraction_correct_weighted_by_path",
    "bout_fraction_correct_weighted_by_displacement",
    "bout_parallel_displacement_sum_mm",
    "bout_path_length_sum_mm",
    "bout_displacement_sum_mm",
    "bout_classified_path_length_sum_mm",
    "bout_classified_displacement_sum_mm",
    "bout_classifiable_path_fraction",
    "bout_classifiable_displacement_fraction",
    "bout_count_total",
    "bout_count_correct",
    "bout_count_opposing",
    "bout_count_ambiguous",
    "time_fraction_correct_classified",
    "time_choice_index",
    "time_correct_s",
    "time_opposing_s",
    "time_classified_s",
    "start_position_axis_mm",
    "end_position_axis_mm",
    "mean_position_axis_mm",
    "start_position_axis_norm",
    "end_position_axis_norm",
    "mean_position_axis_norm",
    "fraction_time_correct_side",
    "available_forward_space_at_start_mm",
    "available_backward_space_at_start_mm",
    "available_forward_space_at_start_norm",
    "available_backward_space_at_start_norm",
    "opportunity_normalized_parallel_displacement",
    "first_aligned_bout_id",
    "first_aligned_bout_start_frame",
    "first_aligned_bout_latency_s",
    "first_aligned_bout_score",
    "first_opposing_bout_id",
    "first_opposing_bout_start_frame",
    "first_opposing_bout_latency_s",
    "first_opposing_bout_score",
    "first_classified_bout_id",
    "first_classified_bout_start_frame",
    "first_classified_bout_latency_s",
    "first_classified_bout_score",
    "quality_flag",
)
MOVING_OMR_PER_FISH = _typed(
    MOVING_OMR_PER_FISH_NAMES,
    int32=(
        "fish_id",
        "valid_transition_count",
        "bout_count_total",
        "bout_count_correct",
        "bout_count_opposing",
        "bout_count_ambiguous",
        "first_aligned_bout_id",
        "first_opposing_bout_id",
        "first_classified_bout_id",
    ),
    int64=(
        "first_aligned_bout_start_frame",
        "first_opposing_bout_start_frame",
        "first_classified_bout_start_frame",
    ),
    int8=("quality_flag",),
)
MOVING_OMR_PER_BOUT = _typed(
    (
        "fish_id",
        "bout_id",
        "start_frame",
        "end_frame",
        "per_bout_omr_score",
        "parallel_displacement_mm",
        "bout_displacement_mm",
        "bout_path_length_mm",
        "correct_label",
        "quality_flag",
    ),
    int32=("fish_id", "bout_id"),
    int64=("start_frame", "end_frame"),
    int8=("correct_label", "quality_flag"),
)
MOVING_OMR_WINDOWS = _typed(
    (
        "window_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "window_length_s",
        "omr_path_index",
        "time_choice_index",
        "coverage_fraction",
        "mean_position_axis_norm",
        "fraction_time_correct_side",
        "n_bouts",
        "quality_flag",
    ),
    int32=("window_id", "fish_id", "n_bouts"),
    int64=("start_frame", "end_frame"),
    int8=("quality_flag",),
)
MOVING_OMR_EARLY = _typed(
    (
        "window_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "window_length_s",
        "actual_window_length_s",
        "omr_path_index",
        "omr_net_direction_index",
        "parallel_displacement_mm",
        "net_displacement_mm",
        "path_length_mm",
        "time_fraction_correct_classified",
        "time_choice_index",
        "coverage_fraction",
        "start_position_axis_norm",
        "end_position_axis_norm",
        "mean_position_axis_norm",
        "fraction_time_correct_side",
        "n_bouts",
        "n_aligned_bouts",
        "n_opposing_bouts",
        "n_ambiguous_bouts",
        "bout_path_index",
        "bout_fraction_correct_weighted_by_path",
        "bout_fraction_correct_weighted_by_displacement",
        "quality_flag",
    ),
    int32=(
        "window_id",
        "fish_id",
        "n_bouts",
        "n_aligned_bouts",
        "n_opposing_bouts",
        "n_ambiguous_bouts",
    ),
    int64=("start_frame", "end_frame"),
    int8=("quality_flag",),
)
GLOBAL_OMR = _typed(
    (
        "fish_id",
        "eligible_step_count",
        "eligible_window_count",
        "omr_path_index_mean",
        "omr_path_index_weighted_by_path",
        "bout_fraction_correct_classified",
        "bout_choice_index",
        "bout_path_index",
        "bout_fraction_correct_weighted_by_path",
        "bout_fraction_correct_weighted_by_displacement",
        "time_choice_index",
        "mean_fraction_time_correct_side",
        "mean_start_position_axis_norm",
        "mean_end_position_axis_norm",
        "mean_mean_position_axis_norm",
        "first_aligned_bout_latency_s_min",
        "total_path_length_mm",
        "total_parallel_displacement_mm",
        "total_bouts",
        "total_bout_correct",
        "total_bout_opposing",
        "total_bout_ambiguous",
        "total_bout_parallel_displacement_mm",
        "total_bout_path_length_mm",
        "total_bout_displacement_mm",
        "coverage_fraction",
        "quality_flag",
    ),
    int32=(
        "fish_id",
        "eligible_step_count",
        "eligible_window_count",
        "total_bouts",
        "total_bout_correct",
        "total_bout_opposing",
        "total_bout_ambiguous",
    ),
    int8=("quality_flag",),
)

RADIAL_PER_FISH_NAMES = (
    "fish_id",
    "omr_path_index",
    "radial_path_index",
    "omr_net_direction_index",
    "tangential_bias_index",
    "stimulus_aligned_radial_displacement_mm",
    "radial_displacement_integrated_mm",
    "tangential_displacement_mm",
    "path_length_mm",
    "net_displacement_mm",
    "valid_transition_count",
    "coverage_fraction",
    "time_fraction_correct_classified",
    "time_choice_index",
    "time_correct_s",
    "time_opposing_s",
    "time_classified_s",
    "start_radius_mm",
    "end_radius_mm",
    "mean_radius_mm",
    "start_radius_norm",
    "end_radius_norm",
    "mean_radius_norm",
    "available_outward_space_at_start_mm",
    "available_inward_space_at_start_mm",
    "bout_fraction_correct_classified",
    "bout_fraction_correct_all",
    "bout_choice_index",
    "bout_count_total",
    "bout_count_correct",
    "bout_count_opposing",
    "bout_count_ambiguous",
    "first_aligned_bout_id",
    "first_aligned_bout_start_frame",
    "first_aligned_bout_latency_s",
    "first_aligned_bout_score",
    "first_opposing_bout_id",
    "first_opposing_bout_start_frame",
    "first_opposing_bout_latency_s",
    "first_opposing_bout_score",
    "quality_flag",
)
RADIAL_PER_FISH = _typed(
    RADIAL_PER_FISH_NAMES,
    int32=(
        "fish_id",
        "valid_transition_count",
        "bout_count_total",
        "bout_count_correct",
        "bout_count_opposing",
        "bout_count_ambiguous",
        "first_aligned_bout_id",
        "first_opposing_bout_id",
    ),
    int64=("first_aligned_bout_start_frame", "first_opposing_bout_start_frame"),
    int8=("quality_flag",),
)
RADIAL_PER_BOUT = _typed(
    (
        "fish_id",
        "bout_id",
        "start_frame",
        "end_frame",
        "start_radius_mm",
        "end_radius_mm",
        "mean_radius_mm",
        "radial_displacement_endpoint_mm",
        "radial_displacement_integrated_mm",
        "stimulus_aligned_radial_displacement_mm",
        "tangential_displacement_mm",
        "path_length_mm",
        "net_displacement_mm",
        "radial_omr_score",
        "radial_net_direction_score",
        "tangential_bias_score",
        "omr_label",
        "valid_radial_basis",
        "quality_flag",
    ),
    int32=("fish_id", "bout_id"),
    int64=("start_frame", "end_frame"),
    int8=("omr_label", "quality_flag"),
    boolean=("valid_radial_basis",),
)
RADIAL_WINDOWS = _typed(
    (
        "window_id",
        "fish_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "window_length_s",
        "omr_path_index",
        "time_choice_index",
        "coverage_fraction",
        "mean_radius_norm",
        "n_bouts",
        "quality_flag",
    ),
    int32=("window_id", "fish_id", "n_bouts"),
    int64=("start_frame", "end_frame"),
    int8=("quality_flag",),
)
# Radial early windows are the onset-filtered subset of radial windows.
RADIAL_EARLY = RADIAL_WINDOWS

LOOM_TRIALS = (
    ("trial_index", _f(INT32)),
    ("onset_frame", _f(INT64)),
    ("offset_frame", _f(INT64)),
    ("trial_duration_s", _f(FLOAT32)),
)
LOOM_PER_TRIAL = (
    ("fish_id", _f(INT32)),
    ("trial_index", _f(INT32)),
    ("escaped", _f(BOOL)),
    ("escape_latency_s", _f(FLOAT32)),
    ("escape_latency_frames", _f(INT32)),
    ("peak_escape_speed_mm_s", _f(FLOAT32)),
    ("distance_at_escape_mm", _f(FLOAT32)),
    ("visual_angle_at_escape_deg", _f(FLOAT32)),
    ("escape_heading_deg", _f(FLOAT32)),
)
LOOM_PER_FISH = (
    ("fish_id", _f(INT32)),
    ("n_escape_responses", _f(INT32)),
    ("escape_probability", _f(FLOAT32)),
    ("mean_escape_latency_s", _f(FLOAT32)),
    ("median_escape_latency_s", _f(FLOAT32)),
    ("mean_peak_escape_speed_mm_s", _f(FLOAT32)),
    ("mean_distance_at_escape_mm", _f(FLOAT32)),
    ("mean_visual_angle_at_escape_deg", _f(FLOAT32)),
    ("habituation_index", _f(FLOAT32)),
)


def _table(
    name: str,
    row_axis: str,
    fields: tuple[tuple[str, StimulusResponseField], ...],
    *,
    required: bool = False,
    bundle: str | None = None,
) -> StimulusResponseTable:
    return StimulusResponseTable(
        name, row_axis, fields, required=required, bundle=bundle
    )


STIMULUS_RESPONSE_TABLES = (
    _table(
        "step_index",
        "stimulus_step_rows",
        STEP_INDEX + (("stimulus_params_json", _s(16384)),),
        required=True,
    ),
    _table("global_per_fish", "fish_rows", GLOBAL_FIELDS, required=True),
    _table(
        "global_omr_per_fish",
        "fish_rows",
        GLOBAL_OMR,
        bundle="moving_grating_omr",
    ),
    _table(
        "frame_annotations",
        "camera_frames",
        (("step_index", _f(INT32)), ("stimulus_mode_id", _f(INT32))),
        bundle="frame_annotations",
    ),
    _table(
        "step_per_fish", "step_fish_rows", STEP_JOIN + STEP_PER_FISH_BASE, required=True
    ),
    _table(
        "step_per_bout",
        "step_bout_rows",
        STEP_JOIN + STEP_PER_BOUT,
        bundle="step_bouts",
    ),
    _table(
        "grating_per_fish",
        "step_fish_rows",
        STEP_JOIN + GRATING_PER_FISH,
        bundle="moving_grating",
    ),
    _table(
        "moving_grating_omr_per_fish",
        "step_fish_rows",
        STEP_JOIN + MOVING_OMR_PER_FISH,
        bundle="moving_grating_omr",
    ),
    _table(
        "moving_grating_omr_per_bout",
        "step_bout_rows",
        STEP_JOIN + MOVING_OMR_PER_BOUT,
        bundle="moving_grating_omr",
    ),
    _table(
        "moving_grating_omr_windows",
        "step_window_rows",
        STEP_JOIN + MOVING_OMR_WINDOWS,
        bundle="moving_grating_omr",
    ),
    _table(
        "moving_grating_omr_early_windows",
        "step_window_rows",
        STEP_JOIN + MOVING_OMR_EARLY,
        bundle="moving_grating_omr",
    ),
    _table(
        "concentric_per_fish",
        "step_fish_rows",
        STEP_JOIN + CONCENTRIC_PER_FISH,
        bundle="concentric_grating",
    ),
    _table(
        "concentric_radial_omr_per_fish",
        "step_fish_rows",
        STEP_JOIN + RADIAL_PER_FISH,
        bundle="concentric_radial_omr",
    ),
    _table(
        "concentric_radial_omr_per_bout",
        "step_bout_rows",
        STEP_JOIN + RADIAL_PER_BOUT,
        bundle="concentric_radial_omr",
    ),
    _table(
        "concentric_radial_omr_windows",
        "step_window_rows",
        STEP_JOIN + RADIAL_WINDOWS,
        bundle="concentric_radial_omr",
    ),
    _table(
        "concentric_radial_omr_early_windows",
        "step_window_rows",
        STEP_JOIN + RADIAL_EARLY,
        bundle="concentric_radial_omr",
    ),
    _table(
        "looming_trials",
        "step_trial_rows",
        STEP_JOIN + LOOM_TRIALS,
        bundle="looming",
    ),
    _table(
        "looming_per_trial_per_fish",
        "step_trial_fish_rows",
        STEP_JOIN + LOOM_PER_TRIAL,
        bundle="looming",
    ),
    _table(
        "looming_per_fish",
        "step_fish_rows",
        STEP_JOIN + LOOM_PER_FISH,
        bundle="looming",
    ),
)
TABLE_BY_NAME = {table.name: table for table in STIMULUS_RESPONSE_TABLES}
KNOWN_BUNDLES = frozenset(
    table.bundle for table in STIMULUS_RESPONSE_TABLES if table.bundle
)

_SEMANTIC_METADATA_FIELDS = frozenset(
    {"step_name", "stimulus_mode", "stimulus_mode_id", "stimulus_params_json"}
)
_LINEAGE_INDEX_FIELDS = frozenset(
    {
        "bout_id",
        "end_frame",
        "fish_id",
        "offset_frame",
        "onset_frame",
        "start_frame",
        "step_index",
        "trial_index",
        "window_id",
    }
)


def table_contract(name: str, *, bundles: Sequence[str] = ()) -> StimulusResponseTable:
    table = TABLE_BY_NAME[name]
    if name == "step_per_fish" and "step_bouts" in set(bundles):
        return StimulusResponseTable(
            table.name, table.row_axis, table.fields + STEP_BOUT_SUMMARY, required=True
        )
    return table


def expected_table_names(bundles: Sequence[str]) -> tuple[str, ...]:
    selected = set(bundles)
    unknown = selected - KNOWN_BUNDLES
    if unknown:
        raise ValueError(f"Unknown stimulus-response v3 bundles: {sorted(unknown)!r}.")
    return tuple(
        table.name
        for table in STIMULUS_RESPONSE_TABLES
        if table.required or table.bundle in selected
    )


def stimulus_response_candidate_fill_value(
    field_name: str,
    field: StimulusResponseField,
) -> object:
    """Return the exact v2 byte-planned physical fill for one field."""

    if field.string_width is not None:
        return 0
    if field.dtype is FLOAT32:
        return float("nan")
    if field.dtype is BOOL:
        return False
    if field_name == "quality_flag":
        return 1
    if field.dtype is INT8:
        return 0
    if (
        field_name.startswith("n_")
        or field_name.startswith("num_")
        or "_count" in field_name
        or field_name == "total_bouts"
        or field_name.startswith("total_bout_")
    ):
        return 0
    return -1


def _candidate_fill_semantics(
    field_name: str,
    field: StimulusResponseField,
) -> str:
    fill = stimulus_response_candidate_fill_value(field_name, field)
    if field.string_width is not None:
        return (
            "Physical fill is uint8 zero. Text is zero-padded, null-terminated "
            "UTF-8; empty text is an all-zero row."
        )
    if field.dtype is FLOAT32:
        return "Physical fill is float32 NaN, the unavailable-metric sentinel."
    if field.dtype is BOOL:
        return "Physical fill is false."
    if field_name == "quality_flag":
        return "Physical fill is int8 1: no valid transitions or invalid bout."
    if field.dtype is INT8:
        return "Physical fill is int8 zero, the ambiguous/unclassified label."
    if fill == 0:
        return "Physical fill is integer zero for an empty count."
    return (
        "Physical fill is integer -1 for an unavailable identity, index, "
        "or frame coordinate."
    )


def _legacy_fill_semantics(field_name: str, field: StimulusResponseField) -> str:
    if field.string_width is not None:
        return (
            "Zero-padded, null-terminated UTF-8; empty text is represented "
            "by an all-zero row."
        )
    if field.dtype is FLOAT32:
        return (
            "Float32 metric; NaN represents an unavailable metric only where "
            "the named computation defines that state."
        )
    if field.dtype is BOOL:
        return "Exact boolean value; no implicit fill state."
    if field_name == "quality_flag":
        return "Exact stage-defined int8 quality code."
    return (
        "Exact integer value; negative sentinels are valid only where the "
        "named field computation explicitly defines them."
    )


def stimulus_response_array_declarations(
    *,
    bundles: Sequence[str],
    byte_planner_adopted: bool = False,
) -> tuple[AnalysisArrayDeclaration, ...]:
    if type(byte_planner_adopted) is not bool:
        raise TypeError("byte_planner_adopted must be an exact bool.")
    declarations: list[AnalysisArrayDeclaration] = []
    for table_name in expected_table_names(bundles):
        table = table_contract(table_name, bundles=bundles)
        for field_name, field in table.fields:
            if field_name == "quality_flag":
                authority_role = AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
            elif field_name in _SEMANTIC_METADATA_FIELDS:
                authority_role = AnalysisAuthorityRole.SEMANTIC_METADATA
            elif field_name in _LINEAGE_INDEX_FIELDS:
                authority_role = AnalysisAuthorityRole.LINEAGE_INDEX
            else:
                authority_role = AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
            fill_semantics = (
                _candidate_fill_semantics(field_name, field)
                if byte_planner_adopted
                else _legacy_fill_semantics(field_name, field)
            )
            row_axis = (
                _BYTE_PLANNED_ROW_AXES.get(table.name, table.row_axis)
                if byte_planner_adopted
                else table.row_axis
            )
            shape = (
                (row_axis, field.string_width)
                if field.string_width is not None
                else (row_axis,)
            )
            axes = (
                (row_axis, "utf8_byte")
                if field.string_width is not None
                else (row_axis,)
            )
            declarations.append(
                AnalysisArrayDeclaration(
                    path=f"{table.name}/{field_name}",
                    contract=ArrayContract(
                        schema_id=f"{STIMULUS_RESPONSE_ARRAY_SCHEMA_ID}.{table.name}.{field_name}",
                        schema_version=(
                            STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION
                            if byte_planner_adopted
                            else STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION
                        ),
                        dtype=field.dtype,
                        shape_template=shape,
                        axis_names=axes,
                        description=field.description,
                        units=field.units,
                    ),
                    required=True,
                    access_pattern=(
                        AccessPattern.WINDOWED
                        if row_axis == "camera_frames"
                        else AccessPattern.EAGER
                    ),
                    write_mode=WriteMode.IMMUTABLE,
                    authority_role=authority_role,
                    fill_semantics=fill_semantics,
                    null_semantics="No nullable fields; fixed text is null-terminated UTF-8.",
                    physical_policy_owner=(
                        STIMULUS_RESPONSE_BYTE_PLANNER_POLICY_OWNER
                        if byte_planner_adopted
                        else STIMULUS_RESPONSE_PHYSICAL_POLICY_OWNER
                    ),
                    byte_planner_adopted=byte_planner_adopted,
                )
            )
    return tuple(declarations)


def stimulus_response_array_manifest(
    *,
    bundles: Sequence[str],
    byte_planner_adopted: bool = False,
) -> dict[str, Any]:
    manifest = {
        "schema_id": STIMULUS_RESPONSE_ARRAY_SCHEMA_ID,
        "schema_version": (
            STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION
            if byte_planner_adopted
            else STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION
        ),
        "run_schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
        "run_schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
        "layout": STIMULUS_RESPONSE_LAYOUT,
        "bundles": sorted(set(bundles)),
        "arrays": [
            item.as_manifest()
            for item in stimulus_response_array_declarations(
                bundles=bundles,
                byte_planner_adopted=byte_planner_adopted,
            )
        ],
    }
    if byte_planner_adopted:
        manifest["byte_planner_adopted"] = True
    return manifest


def validate_mapping(
    table_name: str,
    mapping: Mapping[str, Any],
    *,
    bundles: Sequence[str],
    excluded_fields: Sequence[str] = (),
) -> int:
    table = table_contract(table_name, bundles=bundles)
    excluded = set(excluded_fields)
    expected = {name for name in table.field_names if name not in excluded}
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{table_name} fields must be exact; missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}."
        )
    n_rows: int | None = None
    for name in table.field_names:
        if name in excluded:
            continue
        field = table.field_map[name]
        arr = np.asarray(mapping[name])
        if arr.ndim != 1:
            raise ValueError(
                f"{table_name}/{name} must be one-dimensional; got {arr.shape!r}."
            )
        if arr.dtype != field.logical_dtype:
            raise ValueError(
                f"{table_name}/{name} dtype must be {field.logical_dtype}; got {arr.dtype}."
            )
        if n_rows is None:
            n_rows = int(arr.shape[0])
        elif int(arr.shape[0]) != n_rows:
            raise ValueError(f"{table_name} fields have inconsistent row counts.")
        if field.string_width is not None:
            for value in arr:
                if len(bytes(value).rstrip(b"\x00")) >= field.string_width:
                    raise ValueError(
                        f"{table_name}/{name} exceeds null-terminated UTF-8 payload "
                        f"width {field.string_width - 1}."
                    )
    return 0 if n_rows is None else n_rows


def validate_stimulus_response_v3_run(run_group: Any) -> tuple[str, ...]:
    errors: list[str] = []
    attrs = dict(run_group.attrs)
    if attrs.get("schema_id") != STIMULUS_RESPONSE_SCHEMA_ID:
        errors.append("invalid schema_id")
    if (
        type(attrs.get("schema_version")) is not int
        or attrs.get("schema_version") != STIMULUS_RESPONSE_SCHEMA_VERSION
    ):
        errors.append("invalid schema_version")
    if attrs.get("layout") != STIMULUS_RESPONSE_LAYOUT:
        errors.append("invalid layout")
    bundles = attrs.get(STIMULUS_RESPONSE_BUNDLES_ATTR)
    if (
        not isinstance(bundles, list)
        or any(type(value) is not str for value in bundles)
        or bundles != sorted(set(bundles))
    ):
        errors.append("invalid stimulus-response bundle declaration")
        return tuple(errors)
    try:
        expected_tables = expected_table_names(bundles)
    except ValueError as exc:
        errors.append(str(exc))
        return tuple(errors)
    present_candidate_markers = {
        name for name in _CANDIDATE_REQUIRED_ATTRS if name in attrs
    }
    candidate_markers_complete = (
        present_candidate_markers == _CANDIDATE_REQUIRED_ATTRS
    )
    if present_candidate_markers and not candidate_markers_complete:
        errors.append(
            "stimulus-response storage candidate marker set is incomplete"
        )
    profile_role = attrs.get(STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR)
    if profile_role not in {None, STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE}:
        errors.append("invalid stimulus-response storage profile role")
    byte_planner_adopted = (
        candidate_markers_complete
        and profile_role == STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE
    )
    if _CANDIDATE_EQUIVALENCE_ATTR in attrs and not byte_planner_adopted:
        errors.append(
            "metadata-equivalence evidence requires the exact candidate marker set"
        )
    ignored_groups = set() if byte_planner_adopted else {"visualizations"}
    observed_tables = {
        str(name) for name in run_group.group_keys() if str(name) not in ignored_groups
    }
    if observed_tables != set(expected_tables):
        errors.append(
            f"table set mismatch: expected {sorted(expected_tables)!r}, got {sorted(observed_tables)!r}"
        )
    root_arrays = {str(name) for name in run_group.array_keys()}
    if root_arrays:
        errors.append(f"unexpected root arrays: {sorted(root_arrays)!r}")
    expected_manifest = stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=byte_planner_adopted,
    )
    if attrs.get(STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR) != expected_manifest:
        errors.append("stimulus-response array manifest is not exact")
    for table_name in expected_tables:
        if table_name not in run_group:
            continue
        table = table_contract(table_name, bundles=bundles)
        group = run_group[table_name]
        observed_fields = tuple(group.attrs.get("field_names", ()))
        if observed_fields != table.field_names:
            errors.append(f"{table_name} field_names mismatch")
        expected_field_dtypes = {
            field_name: str(field.logical_dtype) for field_name, field in table.fields
        }
        if group.attrs.get("field_dtypes") != expected_field_dtypes:
            errors.append(f"{table_name} field_dtypes mismatch")
        if group.attrs.get("storage_layout") != "columnar":
            errors.append(f"{table_name} storage_layout mismatch")
        if set(group.array_keys()) != set(table.field_names):
            errors.append(f"{table_name} array set mismatch")
        row_count: int | None = None
        for field_name, field in table.fields:
            if field_name not in group:
                continue
            arr = group[field_name]
            if (
                "storage_profile_id" in dict(arr.attrs)
                and not byte_planner_adopted
            ):
                errors.append(
                    f"{table_name}/{field_name} has byte-planner metadata "
                    "without the exact candidate marker set"
                )
            expected_dtype = (
                np.dtype("uint8")
                if field.string_width is not None
                else field.logical_dtype
            )
            expected_rank = 2 if field.string_width is not None else 1
            if np.dtype(arr.dtype) != expected_dtype or len(arr.shape) != expected_rank:
                errors.append(f"{table_name}/{field_name} dtype/rank mismatch")
                continue
            if (
                field.string_width is not None
                and int(arr.shape[1]) != field.string_width
            ):
                errors.append(f"{table_name}/{field_name} text width mismatch")
            elif field.string_width is not None and int(arr.shape[0]) > 0:
                if np.any(np.asarray(arr[:, -1]) != 0):
                    errors.append(
                        f"{table_name}/{field_name} lacks null-terminated UTF-8 rows"
                    )
            if row_count is None:
                row_count = int(arr.shape[0])
            elif int(arr.shape[0]) != row_count:
                errors.append(f"{table_name} row counts mismatch")
    if errors:
        return tuple(errors)
    try:
        step_ids = np.asarray(run_group["step_index/step_index"][:], dtype=np.int32)
        fish_ids = np.asarray(run_group["global_per_fish/fish_id"][:], dtype=np.int32)
        if step_ids.size == 0 or tuple(step_ids.tolist()) != tuple(
            sorted(set(int(value) for value in step_ids.tolist()))
        ):
            errors.append("step_index identities must be nonempty, unique, and sorted")
        if fish_ids.size == 0 or tuple(fish_ids.tolist()) != tuple(
            sorted(set(int(value) for value in fish_ids.tolist()))
        ):
            errors.append(
                "global fish_id identities must be nonempty, unique, and sorted"
            )
        step_set = set(int(value) for value in step_ids.tolist())
        fish_set = set(int(value) for value in fish_ids.tolist())
        for table_name in expected_tables:
            group = run_group[table_name]
            if table_name != "step_index" and "step_index" in group:
                observed_steps = set(
                    int(value) for value in np.asarray(group["step_index"][:]).tolist()
                )
                if not observed_steps <= step_set:
                    errors.append(f"{table_name} contains unknown step_index values")
            if "fish_id" in group:
                observed_fish = set(
                    int(value) for value in np.asarray(group["fish_id"][:]).tolist()
                )
                if not observed_fish <= fish_set:
                    errors.append(f"{table_name} contains unknown fish_id values")
        step_fish = run_group["step_per_fish"]
        base_pairs = list(
            zip(
                np.asarray(step_fish["step_index"][:]).tolist(),
                np.asarray(step_fish["fish_id"][:]).tolist(),
            )
        )
        expected_pairs = [
            (step_id, fish_id)
            for step_id in step_ids.tolist()
            for fish_id in fish_ids.tolist()
        ]
        if base_pairs != expected_pairs:
            errors.append(
                "step_per_fish must be the ordered step-by-fish Cartesian product"
            )
        if "looming" in set(bundles):
            trials = run_group["looming_trials"]
            trial_keys = list(
                zip(
                    np.asarray(trials["step_index"][:]).tolist(),
                    np.asarray(trials["trial_index"][:]).tolist(),
                )
            )
            if len(trial_keys) != len(set(trial_keys)):
                errors.append("looming trial keys must be unique")
            trial_fish = run_group["looming_per_trial_per_fish"]
            observed = list(
                zip(
                    np.asarray(trial_fish["step_index"][:]).tolist(),
                    np.asarray(trial_fish["trial_index"][:]).tolist(),
                    np.asarray(trial_fish["fish_id"][:]).tolist(),
                )
            )
            expected = [
                (step_id, trial_id, fish_id)
                for step_id in step_ids.tolist()
                for fish_id in fish_ids.tolist()
                for trial_step, trial_id in trial_keys
                if trial_step == step_id
            ]
            if observed != expected:
                errors.append(
                    "looming_per_trial_per_fish must exactly cover every trial-by-fish key"
                )
    except Exception as exc:
        errors.append(f"semantic identity validation failed: {exc}")
    return tuple(errors)


__all__ = [
    "STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR",
    "STIMULUS_RESPONSE_ARRAY_SCHEMA_ID",
    "STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION",
    "STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION",
    "STIMULUS_RESPONSE_BYTE_PLANNER_POLICY_OWNER",
    "STIMULUS_RESPONSE_BUNDLES_ATTR",
    "STIMULUS_RESPONSE_LAYOUT",
    "STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE",
    "STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR",
    "STIMULUS_RESPONSE_SCHEMA_ID",
    "STIMULUS_RESPONSE_SCHEMA_VERSION",
    "STIMULUS_RESPONSE_TABLES",
    "KNOWN_BUNDLES",
    "StimulusResponseTable",
    "expected_table_names",
    "stimulus_response_array_declarations",
    "stimulus_response_array_manifest",
    "stimulus_response_candidate_fill_value",
    "table_contract",
    "validate_mapping",
    "validate_stimulus_response_v3_run",
]
