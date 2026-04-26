"""
ArraySpec definitions for analysis modules.

Input read contracts and output specs for analysis stages that live under
the ``analysis/`` zarr group.  Pipeline-stage specs (raw_video through
tracking) remain in ``stage_arrays.py``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fisheye.shared.zarr.stage_arrays import (
    ArraySpec,
    ValidationResult,
    _validate_specs,
)

# ---------------------------------------------------------------------------
# Stimulus Response — input read contract
# ---------------------------------------------------------------------------
# Arrays that stimulus_response expects in each
# ``analysis/track_kinematics_runs/<type>/<run>/tracks/id_<N>/`` subgroup.
# Validated per-track at load time; fail fast if the upstream output changed.

STIMULUS_RESPONSE_TRACK_INPUTS: Tuple[ArraySpec, ...] = (
    ArraySpec("frame_indices", "int64", ("n_samples",),
              description="Absolute camera frame numbers for each sample."),
    ArraySpec("time_seconds", "float32", ("n_samples",),
              description="Time in seconds from recording start."),
    ArraySpec("positions_mm", "float32", ("n_samples", 2),
              description="XY position in millimetres."),
    ArraySpec("heading_degrees", "float32", ("n_samples",),
              description="Fish heading in degrees (0=right, CCW positive)."),
    ArraySpec("speed_smoothed_mm", "float32", ("n_samples",),
              description="Temporally smoothed swimming speed in mm/s."),
    ArraySpec("frame_path_distance_smoothed_mm", "float32", ("n_samples",), required=False,
              description="Gap-aware frame-to-frame path-distance increment in mm."),
    ArraySpec("cumulative_path_distance_mm", "float32", ("n_samples",), required=False,
              description="Cumulative gap-aware path distance in mm."),
    ArraySpec("angular_velocity_deg_s", "float32", ("n_samples",),
              description="Heading change rate in degrees/second."),
    ArraySpec("detection_source", "int8", ("n_samples",),
              description="0=real detection, 1=interpolated (gap-filled)."),
)

# ---------------------------------------------------------------------------
# Stimulus Response — output specs
# ---------------------------------------------------------------------------

STIMULUS_RESPONSE_FRAMES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("step_index", "int32", ("n_frames",),
              description="Protocol step active at each frame (-1 = between steps / no step)."),
    ArraySpec("stimulus_mode_id", "int32", ("n_frames",),
              description="Stimulus mode enum ID at each frame (-1 = no stimulus)."),
)

STIMULUS_RESPONSE_GLOBAL_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("fish_id", "int32", ("n_fish",)),
    ArraySpec("total_distance_mm", "float32", ("n_fish",)),
    ArraySpec("mean_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("total_active_s", "float32", ("n_fish",)),
    ArraySpec("fraction_moving", "float32", ("n_fish",)),
)

STIMULUS_RESPONSE_PER_FISH_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("fish_id", "int32", ("n_fish",)),
    ArraySpec("total_distance_mm", "float32", ("n_fish",)),
    ArraySpec("mean_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("median_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("max_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("fraction_moving", "float32", ("n_fish",)),
    ArraySpec("coverage", "float32", ("n_fish",),
              description="Fraction of step frames with valid detection data."),
    # Bout metrics (present when bout data is available).
    ArraySpec("num_bouts", "int32", ("n_fish",), required=False,
              description="Number of swim bouts during this step."),
    ArraySpec("mean_bout_duration_s", "float32", ("n_fish",), required=False,
              description="Mean swim bout duration in seconds."),
    ArraySpec("mean_interbout_interval_s", "float32", ("n_fish",), required=False,
              description="Mean time between consecutive bouts in seconds."),
)

STIMULUS_RESPONSE_PER_BOUT_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("fish_id", "int32", ("n_bouts",)),
    ArraySpec("bout_id", "int32", ("n_bouts",)),
    ArraySpec("start_frame", "int64", ("n_bouts",)),
    ArraySpec("end_frame", "int64", ("n_bouts",)),
    ArraySpec("duration_s", "float32", ("n_bouts",)),
    ArraySpec("mean_speed_mm_s", "float32", ("n_bouts",)),
    ArraySpec("peak_speed_mm_s", "float32", ("n_bouts",)),
)

# ---------------------------------------------------------------------------
# Stimulus Response — grating-specific output specs
# ---------------------------------------------------------------------------

STIMULUS_RESPONSE_GRATING_PER_FRAME_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("frame_indices", "int64", ("n_step_frames",)),
    ArraySpec("valid", "bool", ("n_fish", "n_step_frames"),
              description="True where any detection existed (real or interpolated)."),
    ArraySpec("detection_source", "int8", ("n_fish", "n_step_frames"),
              description="0=real detection, 1=interpolated, -1=no detection (gap)."),
    ArraySpec("alignment_angle_deg", "float32", ("n_fish", "n_step_frames")),
    ArraySpec("alignment_cos", "float32", ("n_fish", "n_step_frames")),
    ArraySpec("speed_along_grating_mm_s", "float32", ("n_fish", "n_step_frames")),
    ArraySpec("angular_velocity_deg_s", "float32", ("n_fish", "n_step_frames")),
)

STIMULUS_RESPONSE_GRATING_PER_FISH_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("mean_alignment_cos", "float32", ("n_fish",)),
    ArraySpec("resultant_vector_length", "float32", ("n_fish",)),
    ArraySpec("fraction_following", "float32", ("n_fish",)),
    ArraySpec("fraction_opposing", "float32", ("n_fish",)),
    ArraySpec("fraction_perpendicular", "float32", ("n_fish",)),
    ArraySpec("speed_weighted_alignment", "float32", ("n_fish",)),
    ArraySpec("optomotor_gain", "float32", ("n_fish",)),
    ArraySpec("drift_along_grating_mm", "float32", ("n_fish",)),
    ArraySpec("drift_perp_grating_mm", "float32", ("n_fish",)),
    ArraySpec("latency_to_follow_s", "float32", ("n_fish",)),
)

STIMULUS_RESPONSE_GRATING_TIME_SERIES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("bin_center_s", "float32", ("n_bins",)),
    ArraySpec("alignment_cos", "float32", ("n_fish", "n_bins")),
    ArraySpec("speed_mm_s", "float32", ("n_fish", "n_bins")),
    ArraySpec("fraction_following", "float32", ("n_fish", "n_bins")),
    ArraySpec("optomotor_gain", "float32", ("n_fish", "n_bins")),
)

# ---------------------------------------------------------------------------
# Stimulus Response — concentric grating output specs
# ---------------------------------------------------------------------------

STIMULUS_RESPONSE_CONCENTRIC_PER_FRAME_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("frame_indices", "int64", ("n_step_frames",)),
    ArraySpec("valid", "bool", ("n_fish", "n_step_frames")),
    ArraySpec("detection_source", "int8", ("n_fish", "n_step_frames")),
    ArraySpec("distance_to_center_mm", "float32", ("n_fish", "n_step_frames")),
    ArraySpec("radial_heading_angle_deg", "float32", ("n_fish", "n_step_frames"),
              description="0=toward center, +-180=away from center."),
    ArraySpec("radial_speed_mm_s", "float32", ("n_fish", "n_step_frames"),
              description="Negative=approaching center."),
    ArraySpec("tangential_speed_mm_s", "float32", ("n_fish", "n_step_frames"),
              description="Speed perpendicular to radius (orbiting)."),
)

STIMULUS_RESPONSE_CONCENTRIC_PER_FISH_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("mean_distance_to_center_mm", "float32", ("n_fish",)),
    ArraySpec("initial_distance_to_center_mm", "float32", ("n_fish",)),
    ArraySpec("final_distance_to_center_mm", "float32", ("n_fish",)),
    ArraySpec("min_distance_to_center_mm", "float32", ("n_fish",)),
    ArraySpec("net_radial_displacement_mm", "float32", ("n_fish",),
              description="Negative = moved toward center."),
    ArraySpec("fraction_approaching", "float32", ("n_fish",),
              description="Fraction of frames with radial_speed < 0."),
    ArraySpec("mean_radial_heading_cos", "float32", ("n_fish",),
              description="+1=toward center, -1=away."),
    ArraySpec("time_to_center_s", "float32", ("n_fish",),
              description="Time to first reach center_threshold_mm (NaN if never)."),
    ArraySpec("fraction_near_center", "float32", ("n_fish",),
              description="Fraction of time within center_threshold_mm."),
    ArraySpec("mean_radial_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("mean_tangential_speed_mm_s", "float32", ("n_fish",)),
)

STIMULUS_RESPONSE_CONCENTRIC_TIME_SERIES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("bin_center_s", "float32", ("n_bins",)),
    ArraySpec("distance_to_center_mm", "float32", ("n_fish", "n_bins")),
    ArraySpec("radial_speed_mm_s", "float32", ("n_fish", "n_bins")),
    ArraySpec("radial_heading_cos", "float32", ("n_fish", "n_bins")),
    ArraySpec("fraction_approaching", "float32", ("n_fish", "n_bins")),
)

# ---------------------------------------------------------------------------
# Stimulus Response — looming dot output specs
# ---------------------------------------------------------------------------

STIMULUS_RESPONSE_LOOM_TRIALS_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("onset_frame", "int64", ("n_trials",)),
    ArraySpec("offset_frame", "int64", ("n_trials",)),
    ArraySpec("duration_s", "float32", ("n_trials",)),
)

STIMULUS_RESPONSE_LOOM_PER_FRAME_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("frame_indices", "int64", ("n_step_frames",)),
    ArraySpec("valid", "bool", ("n_fish", "n_step_frames")),
    ArraySpec("detection_source", "int8", ("n_fish", "n_step_frames")),
    ArraySpec("loom_active", "bool", ("n_step_frames",),
              description="True during loom expansion."),
    ArraySpec("trial_index", "int32", ("n_step_frames",),
              description="-1 between trials."),
    ArraySpec("loom_radius_px", "float32", ("n_step_frames",),
              description="Reconstructed loom radius in projector pixels."),
    ArraySpec("visual_angle_deg", "float32", ("n_step_frames",),
              description="Loom visual angle in degrees (requires z_eff_mm calibration)."),
    ArraySpec("distance_to_loom_mm", "float32", ("n_fish", "n_step_frames")),
)

STIMULUS_RESPONSE_LOOM_PER_TRIAL_PER_FISH_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("escaped", "bool", ("n_fish", "n_trials")),
    ArraySpec("escape_latency_s", "float32", ("n_fish", "n_trials"),
              description="NaN if no escape."),
    ArraySpec("escape_latency_frames", "int32", ("n_fish", "n_trials"),
              description="-1 if no escape."),
    ArraySpec("peak_escape_speed_mm_s", "float32", ("n_fish", "n_trials")),
    ArraySpec("distance_at_escape_mm", "float32", ("n_fish", "n_trials")),
    ArraySpec("visual_angle_at_escape_deg", "float32", ("n_fish", "n_trials")),
    ArraySpec("escape_heading_deg", "float32", ("n_fish", "n_trials")),
)

STIMULUS_RESPONSE_LOOM_PER_FISH_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("n_escape_responses", "int32", ("n_fish",)),
    ArraySpec("escape_probability", "float32", ("n_fish",)),
    ArraySpec("mean_escape_latency_s", "float32", ("n_fish",)),
    ArraySpec("median_escape_latency_s", "float32", ("n_fish",)),
    ArraySpec("mean_peak_escape_speed_mm_s", "float32", ("n_fish",)),
    ArraySpec("mean_distance_at_escape_mm", "float32", ("n_fish",)),
    ArraySpec("mean_visual_angle_at_escape_deg", "float32", ("n_fish",)),
    ArraySpec("habituation_index", "float32", ("n_fish",),
              description="Linear slope of escape latency vs trial index (positive=habituating)."),
)

STIMULUS_RESPONSE_LOOM_TIME_SERIES_ARRAYS: Tuple[ArraySpec, ...] = (
    ArraySpec("trial_time_s", "float32", ("n_time_points",),
              description="Time relative to loom onset (negative=pre-stimulus)."),
    ArraySpec("mean_speed_mm_s", "float32", ("n_fish", "n_time_points"),
              description="Trial-averaged speed."),
    ArraySpec("mean_distance_to_loom_mm", "float32", ("n_fish", "n_time_points"),
              description="Trial-averaged distance to loom center."),
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_track_inputs(
    track_group,
    *,
    label: str = "track_kinematics",
) -> ValidationResult:
    """Check that a single track subgroup has the arrays stimulus_response needs."""
    errors: List[str] = []
    warnings: List[str] = []
    _validate_specs(
        track_group,
        STIMULUS_RESPONSE_TRACK_INPUTS,
        group_label=label,
        errors=errors,
        warnings=warnings,
    )
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_global_output(group) -> ValidationResult:
    """Check that a global/ output group matches the spec."""
    errors: List[str] = []
    warnings: List[str] = []
    _validate_specs(
        group,
        STIMULUS_RESPONSE_GLOBAL_ARRAYS,
        group_label="stimulus_response/global",
        errors=errors,
        warnings=warnings,
    )
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_per_fish_output(group, *, step_label: str = "step") -> ValidationResult:
    """Check that a per_fish/ output group matches the spec."""
    errors: List[str] = []
    warnings: List[str] = []
    _validate_specs(
        group,
        STIMULUS_RESPONSE_PER_FISH_ARRAYS,
        group_label=f"stimulus_response/{step_label}/per_fish",
        errors=errors,
        warnings=warnings,
    )
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_per_bout_output(group) -> ValidationResult:
    """Check that a per_bout/ output group matches the spec."""
    errors: List[str] = []
    warnings: List[str] = []
    _validate_specs(
        group,
        STIMULUS_RESPONSE_PER_BOUT_ARRAYS,
        group_label="stimulus_response/per_bout",
        errors=errors,
        warnings=warnings,
    )
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


__all__ = [
    "STIMULUS_RESPONSE_TRACK_INPUTS",
    "STIMULUS_RESPONSE_FRAMES_ARRAYS",
    "STIMULUS_RESPONSE_GLOBAL_ARRAYS",
    "STIMULUS_RESPONSE_PER_FISH_ARRAYS",
    "STIMULUS_RESPONSE_PER_BOUT_ARRAYS",
    "STIMULUS_RESPONSE_GRATING_PER_FRAME_ARRAYS",
    "STIMULUS_RESPONSE_GRATING_PER_FISH_ARRAYS",
    "STIMULUS_RESPONSE_GRATING_TIME_SERIES_ARRAYS",
    "STIMULUS_RESPONSE_CONCENTRIC_PER_FRAME_ARRAYS",
    "STIMULUS_RESPONSE_CONCENTRIC_PER_FISH_ARRAYS",
    "STIMULUS_RESPONSE_CONCENTRIC_TIME_SERIES_ARRAYS",
    "STIMULUS_RESPONSE_LOOM_TRIALS_ARRAYS",
    "STIMULUS_RESPONSE_LOOM_PER_FRAME_ARRAYS",
    "STIMULUS_RESPONSE_LOOM_PER_TRIAL_PER_FISH_ARRAYS",
    "STIMULUS_RESPONSE_LOOM_PER_FISH_ARRAYS",
    "STIMULUS_RESPONSE_LOOM_TIME_SERIES_ARRAYS",
    "validate_track_inputs",
    "validate_global_output",
    "validate_per_fish_output",
    "validate_per_bout_output",
]
