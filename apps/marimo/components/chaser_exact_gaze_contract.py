"""Closed scientific contract for exact body-frame gaze tracking."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .chaser_exact_gaze_arrays import GAZE_TRACKING_ARRAYS

GAZE_TRACKING_PARENT = "analysis/chaser_gaze_tracking_runs"
EYE_ORIENTATION_PARENT = "analysis/eye_angle_runs"
FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "current_run",
        "selected",
        "authoritative",
        "authoritative_run",
        "default",
    }
)
EXPECTED_POLICY = {
    "gaze_field": "directed_left_right_gaze_signed_deg_in_fish_body_frame",
    "bearing_field": "exact_chaser_body_bearing_deg_anatomical_left_positive",
    "world_frame_gaze": "prohibited",
    "nasal_positive_eye_angle": "prohibited_for_object_bearing_comparison",
    "orientation_fallback": "prohibited",
    "invalid_rows": "retained_and_excluded_from_summaries",
    "cohort_inference_unit": "recording_fish",
    "virtual_control_geometry": (
        "rotate_each_exact_real_chaser_trajectory_about_reviewed_arena_center"
    ),
    "virtual_collision_denominator": (
        "frames_where_parent_virtual_and_compared_real_positions_are_valid_and_present"
    ),
    "virtual_collision_exclusion": (
        "exclude_candidate_when_max_real_chaser_collision_fraction_exceeds_threshold"
    ),
    "virtual_null_denominator": (
        "finite_accepted_virtual_metric_values_with_count_persisted_per_real_summary"
    ),
    "control_direction": (
        "gain_and_lock_real_minus_virtual_mean;error_virtual_mean_minus_real"
    ),
    "dynamic_tracking": (
        "wrapped_contiguous_frame_deltas_zero_and_causal_nonnegative_lags"
    ),
    "dynamic_lag_selection": "maximum_correlation_within_exact_maximum_lag",
}
EXPECTED_REGISTRIES = {
    "eye": {"1": "left", "2": "right"},
    "semantic_role": {
        "1": "chaser_pre",
        "2": "chaser_training",
        "3": "chaser_post",
    },
}


class ExactGazeTrackingContractError(ValueError):
    """One exact gaze successor identity, policy, or lineage is invalid."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactGazeTrackingContractError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactGazeTrackingContractError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_child_path(value: Any, *, parent: str, label: str) -> str:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactGazeTrackingContractError(f"{label} must be one exact child path.")
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", ".."}
        or name.casefold() in FORBIDDEN_SELECTORS
    ):
        raise ExactGazeTrackingContractError(
            f"{label} must be one exact child below {parent!r}."
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _positive_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactGazeTrackingContractError(f"{label} must be positive and finite.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ExactGazeTrackingContractError(f"{label} must be positive and finite.")
    return result


def validate_gaze_scientific_manifest(
    value: Any,
    *,
    expected_scientific_payload_sha256: str,
    expected_n_frames: int,
    expected_n_chasers: int,
    expected_relative_binding: Mapping[str, Any],
    expected_semantic_manifest_sha256: str,
    expected_radial_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the complete version-3 gaze successor and its exact joins."""

    scientific = _mapping(value, label="gaze scientific manifest")
    expected_payload = _digest(
        expected_scientific_payload_sha256,
        label="gaze scientific payload digest",
    )
    body = dict(scientific)
    observed_payload = body.pop("payload_digest", None)
    if (
        observed_payload != expected_payload
        or canonical_json_sha256(_plain(body)) != observed_payload
    ):
        raise ExactGazeTrackingContractError("Gaze scientific payload digest is stale.")
    schema = _mapping(scientific.get("scientific_schema"), label="gaze schema")
    if dict(schema) != {
        "schema_id": "palette.analysis.chaser_gaze_tracking",
        "schema_version": 3,
        "method_id": "exact_eye_body_frame_gaze_real_rotated_controls_dynamic_v2",
        "row_unit": "acquisition_frame_x_eye_x_chaser",
        "summary_unit": "semantic_role_x_eye_x_chaser",
        "event_unit": "contiguous_lock_on_interval",
        "virtual_candidate_unit": "chaser_x_rotation",
        "virtual_summary_unit": ("semantic_role_x_eye_x_accepted_virtual_reference"),
        "control_summary_unit": "semantic_role_x_eye_x_real_chaser",
    }:
        raise ExactGazeTrackingContractError("Gaze scientific schema is incompatible.")
    if (
        scientific.get("schema_id")
        != "palette.analysis.chaser_gaze_tracking.prepared_successor"
        or scientific.get("schema_version") != 2
        or scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        raise ExactGazeTrackingContractError(
            "Gaze successor lifecycle identity is incompatible."
        )
    try:
        require_same_exact_relative_frame_child(
            expected_relative_binding,
            _mapping(
                _mapping(scientific.get("sources"), label="gaze sources").get(
                    "relative_frame"
                ),
                label="gaze relative-frame source",
            ),
            expected_label="spatial keypoint relative-frame binding",
            observed_label="gaze relative-frame binding",
        )
    except ExactRelativeFrameBindingError as exc:
        raise ExactGazeTrackingContractError(str(exc)) from exc
    sources = _mapping(scientific.get("sources"), label="gaze sources")
    if set(sources) != {
        "relative_frame",
        "eye_orientation",
        "semantic_selection_manifest_sha256",
        "radial_near_field_geometry_authority",
    }:
        raise ExactGazeTrackingContractError("Gaze source roster is incompatible.")
    semantic_digest = _digest(
        sources.get("semantic_selection_manifest_sha256"),
        label="gaze semantic-selection digest",
    )
    if semantic_digest != expected_semantic_manifest_sha256:
        raise ExactGazeTrackingContractError(
            "Gaze successor uses another semantic selection."
        )
    eye = _mapping(sources.get("eye_orientation"), label="eye-orientation source")
    if set(eye) != {
        "run_path",
        "manifest_sha256",
        "convention_receipt_sha256",
        "channel_policy",
    }:
        raise ExactGazeTrackingContractError(
            "Eye-orientation source field roster is incompatible."
        )
    eye_binding = {
        "run_path": _exact_child_path(
            eye.get("run_path"), parent=EYE_ORIENTATION_PARENT, label="eye run"
        ),
        "manifest_sha256": _digest(
            eye.get("manifest_sha256"), label="eye logical manifest digest"
        ),
        "convention_receipt_sha256": _digest(
            eye.get("convention_receipt_sha256"),
            label="eye convention receipt digest",
        ),
        "channel_policy": eye.get("channel_policy"),
    }
    if (
        type(eye_binding["channel_policy"]) is not str
        or not eye_binding["channel_policy"]
        or eye_binding["channel_policy"] != eye_binding["channel_policy"].strip()
    ):
        raise ExactGazeTrackingContractError(
            "Eye-orientation channel policy must be one exact non-empty string."
        )
    radial = _mapping(
        sources.get("radial_near_field_geometry_authority"),
        label="gaze radial geometry authority",
    )
    if set(radial) != {
        "run_path",
        "manifest_sha256",
        "scientific_payload_sha256",
        "arena_geometry_and_scale",
    }:
        raise ExactGazeTrackingContractError(
            "Gaze radial geometry authority field roster is incompatible."
        )
    expected_radial = _mapping(expected_radial_binding, label="expected radial binding")
    if (
        radial.get("run_path") != expected_radial.get("run_path")
        or radial.get("manifest_sha256") != expected_radial.get("manifest_sha256")
        or radial.get("scientific_payload_sha256")
        != expected_radial.get("scientific_payload_sha256")
        or _plain(radial.get("arena_geometry_and_scale"))
        != _plain(expected_radial.get("arena_geometry_and_scale"))
    ):
        raise ExactGazeTrackingContractError(
            "Gaze successor uses another radial geometry authority."
        )
    radial_binding = {
        "run_path": _exact_child_path(
            radial.get("run_path"),
            parent="analysis/chaser_radial_near_field_runs",
            label="gaze radial run",
        ),
        "manifest_sha256": _digest(
            radial.get("manifest_sha256"), label="gaze radial manifest digest"
        ),
        "scientific_payload_sha256": _digest(
            radial.get("scientific_payload_sha256"),
            label="gaze radial scientific payload digest",
        ),
        "arena_geometry_and_scale": dict(
            _mapping(
                radial.get("arena_geometry_and_scale"),
                label="gaze arena geometry-and-scale binding",
            )
        ),
    }
    dimensions = _mapping(scientific.get("dimensions"), label="gaze dimensions")
    try:
        n_frames = int(dimensions.get("n_frames", -1))
        n_chasers = int(dimensions.get("n_chasers", -1))
        n_gaze_rows = int(dimensions.get("n_gaze_rows", -1))
        n_summary_rows = int(dimensions.get("n_summary_rows", -1))
        n_lock_events = int(dimensions.get("n_lock_events", -1))
        n_virtual_candidates = int(dimensions.get("n_virtual_candidates", -1))
        n_virtual_references = int(dimensions.get("n_virtual_references", -1))
        n_virtual_summary_rows = int(dimensions.get("n_virtual_summary_rows", -1))
        n_control_summary_rows = int(dimensions.get("n_control_summary_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ExactGazeTrackingContractError("Gaze dimensions are invalid.") from exc
    if (
        set(dimensions)
        != {
            "n_frames",
            "n_chasers",
            "n_gaze_rows",
            "n_summary_rows",
            "n_lock_events",
            "n_virtual_candidates",
            "n_virtual_references",
            "n_virtual_summary_rows",
            "n_control_summary_rows",
        }
        or n_frames != expected_n_frames
        or n_chasers != expected_n_chasers
        or n_gaze_rows != n_frames * 2 * n_chasers
        or n_summary_rows != 3 * 2 * n_chasers
        or n_lock_events < 0
        or n_virtual_candidates < n_virtual_references
        or n_virtual_references < 0
        or n_virtual_summary_rows != 3 * 2 * n_virtual_references
        or n_control_summary_rows != n_summary_rows
    ):
        raise ExactGazeTrackingContractError(
            "Gaze dimensions differ from the keypoint relative frame."
        )
    parameters = _mapping(scientific.get("parameters"), label="gaze parameters")
    if set(parameters) != {
        "lock_threshold_deg",
        "minimum_lock_duration_s",
        "maximum_tracking_distance_mm",
        "accessible_quantiles",
        "empirical_eye_range_deg",
        "virtual_rotations_deg",
        "minimum_virtual_separation_mm",
        "maximum_virtual_collision_fraction",
        "maximum_dynamic_lag_s",
        "minimum_regression_samples",
        "minimum_regression_span_deg",
    }:
        raise ExactGazeTrackingContractError("Gaze parameter roster is incompatible.")
    for name in (
        "lock_threshold_deg",
        "minimum_lock_duration_s",
        "maximum_tracking_distance_mm",
        "minimum_virtual_separation_mm",
        "maximum_dynamic_lag_s",
        "minimum_regression_span_deg",
    ):
        _positive_finite(parameters.get(name), label=f"gaze parameter {name}")
    quantiles = np.asarray(parameters.get("accessible_quantiles"), dtype=np.float64)
    eye_range = np.asarray(parameters.get("empirical_eye_range_deg"), dtype=np.float64)
    if (
        quantiles.shape != (2,)
        or np.any(~np.isfinite(quantiles))
        or not (0 <= quantiles[0] < quantiles[1] <= 1)
        or eye_range.shape != (2, 2)
    ):
        raise ExactGazeTrackingContractError(
            "Gaze accessible-range parameters are incompatible."
        )
    rotations = np.asarray(parameters.get("virtual_rotations_deg"), dtype=np.float64)
    collision = parameters.get("maximum_virtual_collision_fraction")
    minimum_samples = parameters.get("minimum_regression_samples")
    if (
        rotations.ndim != 1
        or rotations.size == 0
        or np.any(~np.isfinite(rotations))
        or np.any((rotations <= 0) | (rotations >= 360))
        or np.unique(rotations).size != rotations.size
        or n_virtual_candidates != n_chasers * rotations.size
        or isinstance(collision, bool)
        or not isinstance(collision, (int, float))
        or not math.isfinite(float(collision))
        or not 0 <= float(collision) <= 1
        or type(minimum_samples) is not int
        or minimum_samples < 3
    ):
        raise ExactGazeTrackingContractError(
            "Gaze rotated-control or regression parameters are incompatible."
        )
    arena = _mapping(scientific.get("arena"), label="gaze arena")
    if set(arena) != {"center_xy_px", "radius_px", "radius_mm", "pixels_per_mm"}:
        raise ExactGazeTrackingContractError(
            "Gaze reviewed arena record is incompatible."
        )
    center = np.asarray(arena.get("center_xy_px"), dtype=np.float64)
    try:
        radius_px = float(arena.get("radius_px"))
        radius_mm = float(arena.get("radius_mm"))
        pixels_per_mm = float(arena.get("pixels_per_mm"))
    except (TypeError, ValueError) as exc:
        raise ExactGazeTrackingContractError("Gaze arena values are invalid.") from exc
    if (
        center.shape != (2,)
        or np.any(~np.isfinite(center))
        or min(radius_px, radius_mm, pixels_per_mm) <= 0
        or not math.isclose(
            radius_px / pixels_per_mm,
            radius_mm,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
    ):
        raise ExactGazeTrackingContractError("Gaze arena scale is inconsistent.")
    radial_arena = _mapping(expected_radial.get("arena"), label="expected radial arena")
    try:
        expected_center = np.asarray(
            [radial_arena.get("center_x_px"), radial_arena.get("center_y_px")],
            dtype=np.float64,
        )
        expected_radius_px = float(radial_arena.get("radius_px"))
        expected_radius_mm = float(radial_arena.get("radius_mm"))
    except (TypeError, ValueError) as exc:
        raise ExactGazeTrackingContractError(
            "Expected radial arena values are invalid."
        ) from exc
    if (
        expected_center.shape != (2,)
        or np.any(~np.isfinite(expected_center))
        or not np.allclose(center, expected_center, rtol=0.0, atol=1e-9)
        or not math.isclose(radius_px, expected_radius_px, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(radius_mm, expected_radius_mm, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ExactGazeTrackingContractError(
            "Gaze arena values differ from the exact radial geometry authority."
        )
    if dict(_mapping(scientific.get("policy"), label="gaze policy")) != EXPECTED_POLICY:
        raise ExactGazeTrackingContractError("Gaze policy is incompatible.")
    if (
        dict(_mapping(scientific.get("identity_registries"), label="gaze registries"))
        != EXPECTED_REGISTRIES
    ):
        raise ExactGazeTrackingContractError(
            "Gaze identity registries are incompatible."
        )
    declarations = scientific.get("array_declarations")
    if not isinstance(declarations, (list, tuple)):
        raise ExactGazeTrackingContractError("Gaze array declarations are absent.")
    declaration_paths = tuple(
        item.get("path") if isinstance(item, Mapping) else None for item in declarations
    )
    if set(declaration_paths) != set(GAZE_TRACKING_ARRAYS) or len(
        declaration_paths
    ) != len(GAZE_TRACKING_ARRAYS):
        raise ExactGazeTrackingContractError("Gaze array roster is incompatible.")
    return {
        "source_relative_frame": dict(
            _mapping(sources.get("relative_frame"), label="gaze relative source")
        ),
        "source_eye_orientation": eye_binding,
        "source_radial_geometry": radial_binding,
        "semantic_selection_manifest_sha256": semantic_digest,
        "parameters": dict(parameters),
        "n_frames": n_frames,
        "n_chasers": n_chasers,
        "n_gaze_rows": n_gaze_rows,
        "n_summary_rows": n_summary_rows,
        "n_lock_events": n_lock_events,
    }


__all__ = [
    "EYE_ORIENTATION_PARENT",
    "EXPECTED_POLICY",
    "EXPECTED_REGISTRIES",
    "GAZE_TRACKING_PARENT",
    "ExactGazeTrackingContractError",
    "validate_gaze_scientific_manifest",
]
