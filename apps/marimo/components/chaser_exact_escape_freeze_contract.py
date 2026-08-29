"""Closed scientific contract shared by escape/freeze discovery and loading."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .chaser_exact_bout_response_contract import (
    ExactBoutResponseContractError,
    normalize_motion_binding,
    require_digest as _require_digest,
    require_mapping as _require_mapping,
)

ESCAPE_FREEZE_PARENT = "analysis/chaser_escape_freeze_runs"

EXPECTED_POLICY = {
    "speed_escape": "bout_peak_speed_greater_equal_threshold",
    "high_turn_tier": "optional_directed_annotation_separate_from_speed_class",
    "freeze": "no_speed_escape_and_low_speed_fraction_with_coverage_gate",
    "trial_attachment": "exactly_one_controller_trial_row_at_bout_onset",
    "event_counts": "retained_even_when_recapture_trace_unusable",
    "recapture": "first_post_event_exact_trial_member_at_or_below_onset_distance",
    "fallback_trial_segmentation": "prohibited",
    "trial_gaps": (
        "excluded_from_membership_time_and_event_attachment;"
        "retained_as_coverage_evidence"
    ),
}
EXPECTED_REGISTRIES = {
    "response_class": {
        "0": "insufficient_valid_freeze_window",
        "1": "speed_escape",
        "2": "freeze_candidate",
        "3": "other_response",
    },
    "trace_exclusion_reason": {
        "0": "valid",
        "1": "no_post_event_valid_distance_in_trial",
        "2": "event_frame_unavailable",
    },
}


class ExactEscapeFreezeContractError(ValueError):
    """One closed escape/freeze identity, classifier, or policy is invalid."""


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    try:
        return _require_mapping(value, label=label)
    except ExactBoutResponseContractError as exc:
        raise ExactEscapeFreezeContractError(str(exc)) from exc


def require_digest(value: Any, *, label: str) -> str:
    try:
        return _require_digest(value, label=label)
    except ExactBoutResponseContractError as exc:
        raise ExactEscapeFreezeContractError(str(exc)) from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number(
    value: Any,
    *,
    label: str,
    positive: bool = False,
    fraction: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactEscapeFreezeContractError(f"{label} must be one finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ExactEscapeFreezeContractError(f"{label} must be one finite number.")
    if positive and result <= 0:
        raise ExactEscapeFreezeContractError(f"{label} must be positive.")
    if not positive and result < 0:
        raise ExactEscapeFreezeContractError(f"{label} must be non-negative.")
    if fraction and result > 1:
        raise ExactEscapeFreezeContractError(f"{label} must be in [0, 1].")
    return result


def normalize_escape_motion_binding(
    value: Any,
    *,
    expected_bout_motion: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the escape speed source against the exact bout motion source."""

    motion = require_mapping(value, label="escape/freeze motion source")
    if set(motion) != {
        "run_path",
        "manifest_sha256",
        "speed_level",
        "relative_frame_projection",
    }:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze motion source has an unsupported field set."
        )
    speed_level = motion.get("speed_level")
    if (
        type(speed_level) is not str
        or not speed_level
        or speed_level != speed_level.strip()
    ):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze speed level must be one exact non-empty string."
        )
    core = {
        "run_path": motion.get("run_path"),
        "manifest_sha256": motion.get("manifest_sha256"),
        "relative_frame_projection": motion.get("relative_frame_projection"),
    }
    try:
        normalized = normalize_motion_binding(core)
        expected = normalize_motion_binding(expected_bout_motion)
    except ExactBoutResponseContractError as exc:
        raise ExactEscapeFreezeContractError(str(exc)) from exc
    if _plain(normalized) != _plain(expected):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze motion source differs from the bout-response source."
        )
    return {**normalized, "speed_level": speed_level}


def normalize_parameters(value: Any) -> Mapping[str, Any]:
    """Validate the complete persisted classifier and source-window parameters."""

    parameters = require_mapping(value, label="escape/freeze parameters")
    if set(parameters) != {
        "escape_speed_threshold_mm_s",
        "high_turn_threshold_deg",
        "freeze_speed_threshold_mm_s",
        "freeze_window_s",
        "freeze_fraction_threshold",
        "minimum_freeze_valid_fraction",
        "threshold_sweep_mm_s",
    }:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze parameters have an unsupported field set."
        )
    sweep_value = parameters.get("threshold_sweep_mm_s")
    if not isinstance(sweep_value, (list, tuple)) or not 1 <= len(sweep_value) <= 64:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze threshold sweep is invalid."
        )
    sweep = tuple(
        _number(item, label="escape/freeze sweep threshold", positive=True)
        for item in sweep_value
    )
    if np.any(np.diff(np.asarray(sweep, dtype=np.float64)) <= 0):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze threshold sweep is not strictly increasing."
        )
    return {
        "escape_speed_threshold_mm_s": _number(
            parameters.get("escape_speed_threshold_mm_s"),
            label="escape speed threshold",
            positive=True,
        ),
        "high_turn_threshold_deg": _number(
            parameters.get("high_turn_threshold_deg"),
            label="high-turn threshold",
            positive=True,
        ),
        "freeze_speed_threshold_mm_s": _number(
            parameters.get("freeze_speed_threshold_mm_s"),
            label="freeze speed threshold",
        ),
        "freeze_window_s": _number(
            parameters.get("freeze_window_s"),
            label="freeze window",
            positive=True,
        ),
        "freeze_fraction_threshold": _number(
            parameters.get("freeze_fraction_threshold"),
            label="freeze fraction threshold",
            fraction=True,
        ),
        "minimum_freeze_valid_fraction": _number(
            parameters.get("minimum_freeze_valid_fraction"),
            label="minimum freeze valid fraction",
            fraction=True,
        ),
        "threshold_sweep_mm_s": sweep,
    }


def validate_scientific_manifest(
    value: Any,
    *,
    expected_scientific_payload_sha256: str,
    expected_controller_payload_sha256: str,
    expected_bout_response_payload_sha256: str,
    expected_bout_motion: Mapping[str, Any],
    expected_n_trials: int | None = None,
) -> Mapping[str, Any]:
    """Validate one complete version-2 escape/freeze scientific manifest."""

    scientific = require_mapping(value, label="escape/freeze scientific manifest")
    expected_payload = require_digest(
        expected_scientific_payload_sha256,
        label="escape/freeze scientific payload digest",
    )
    expected_controller = require_digest(
        expected_controller_payload_sha256,
        label="escape/freeze controller-trial payload digest",
    )
    expected_bout = require_digest(
        expected_bout_response_payload_sha256,
        label="escape/freeze bout-response payload digest",
    )
    body = dict(scientific)
    observed_payload = body.pop("payload_digest", None)
    if (
        observed_payload != expected_payload
        or canonical_json_sha256(_plain(body)) != observed_payload
    ):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze scientific payload digest is stale."
        )
    schema = require_mapping(
        scientific.get("scientific_schema"), label="escape/freeze scientific schema"
    )
    if (
        set(schema)
        != {"schema_id", "schema_version", "method_id", "event_unit", "trial_unit"}
        or schema.get("schema_id") != "palette.analysis.chaser_escape_freeze"
        or schema.get("schema_version") != 2
        or schema.get("method_id")
        != "exact_trial_speed_escape_optional_high_turn_freeze_v1"
        or schema.get("event_unit") != "speed_thresholded_exact_swim_bout_x_chaser"
        or schema.get("trial_unit") != "exact_logged_controller_trial"
    ):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze scientific schema is incompatible."
        )
    policy = require_mapping(scientific.get("policy"), label="escape/freeze policy")
    registries = require_mapping(
        scientific.get("identity_registries"),
        label="escape/freeze identity registries",
    )
    if dict(policy) != EXPECTED_POLICY or dict(registries) != EXPECTED_REGISTRIES:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze policy or identity registries are incompatible."
        )
    parameters = normalize_parameters(scientific.get("parameters"))
    dimensions = require_mapping(
        scientific.get("dimensions"), label="escape/freeze dimensions"
    )
    try:
        n_trials = int(dimensions.get("n_trials", -1))
        n_events = int(dimensions.get("n_events", -1))
        n_sweep_rows = int(dimensions.get("n_sweep_rows", -1))
    except (TypeError, ValueError) as exc:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze dimensions are invalid."
        ) from exc
    if (
        not 0 < n_trials <= 32
        or n_events < 0
        or n_sweep_rows != n_trials * len(parameters["threshold_sweep_mm_s"])
        or (expected_n_trials is not None and n_trials != expected_n_trials)
    ):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze dimensions are incompatible."
        )
    sources = require_mapping(scientific.get("sources"), label="escape/freeze sources")
    if set(sources) != {
        "motion",
        "controller_trial_payload_sha256",
        "bout_response_payload_sha256",
    }:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze sources have an unsupported field set."
        )
    controller_payload = require_digest(
        sources.get("controller_trial_payload_sha256"),
        label="escape/freeze controller payload digest",
    )
    bout_payload = require_digest(
        sources.get("bout_response_payload_sha256"),
        label="escape/freeze bout-response payload digest",
    )
    if controller_payload != expected_controller or bout_payload != expected_bout:
        raise ExactEscapeFreezeContractError(
            "Escape/freeze controller or bout-response source differs from its bundle."
        )
    motion = normalize_escape_motion_binding(
        sources.get("motion"), expected_bout_motion=expected_bout_motion
    )
    if (
        scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        raise ExactEscapeFreezeContractError(
            "Escape/freeze scientific product is not selector-ineligible."
        )
    return {
        "source_motion": motion,
        "controller_trial_payload_sha256": controller_payload,
        "bout_response_payload_sha256": bout_payload,
        "classifier_parameters": parameters,
        "n_trials": n_trials,
        "n_events": n_events,
        "n_sweep_rows": n_sweep_rows,
    }


__all__ = [
    "ESCAPE_FREEZE_PARENT",
    "EXPECTED_POLICY",
    "EXPECTED_REGISTRIES",
    "ExactEscapeFreezeContractError",
    "normalize_escape_motion_binding",
    "normalize_parameters",
    "validate_scientific_manifest",
]
