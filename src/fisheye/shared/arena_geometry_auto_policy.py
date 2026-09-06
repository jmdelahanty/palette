"""Versioned numerical geometry policy evaluation, never acceptance or activation.

This scientific-policy addition is deliberately shadow-only. Thresholds have no
enabled defaults: a caller must supply every limit and its frozen derivation
binding. A pass measures a candidate policy, not an accepted production gate.
Existing geometry candidate/comparison/selection grammars remain unchanged.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)


POLICY_ID = "operationally_corroborated_acquisition_shadow_v1"
POLICY_SCHEMA_ID = "palette.arena_geometry_shadow_policy"
EVIDENCE_SCHEMA_ID = "palette.arena_geometry_shadow_evidence"
EVALUATION_SCHEMA_ID = "palette.arena_geometry_shadow_evaluation"
SOURCE_BINDING_FIELDS = frozenset(
    {
        "acquisition_candidate_record_sha256",
        "fit_report_sha256",
        "detection_source_signature",
        "acquisition_observation_sha256",
        "coordinate_binding_sha256",
        "rim_metrics_sha256",
        "scientific_recipe_sha256",
    }
)

# threshold -> (measurement, comparison direction, dimensional domain)
THRESHOLD_CONTRACT = {
    "maximum_center_displacement_px": ("center_displacement_px", "maximum", "px"),
    "minimum_circle_iou": ("circle_iou", "minimum", "fraction"),
    "maximum_gate_disagreement_fraction": (
        "gate_disagreement_fraction",
        "maximum",
        "fraction",
    ),
    "maximum_acquisition_only_fraction": (
        "acquisition_only_fraction",
        "maximum",
        "fraction",
    ),
    "maximum_palette_only_fraction": ("palette_only_fraction", "maximum", "fraction"),
    "minimum_detection_row_count": ("detection_row_count", "minimum", "count"),
    "minimum_angular_support_fraction": (
        "angular_support_fraction",
        "minimum",
        "fraction",
    ),
    "minimum_visible_angular_fraction": (
        "visible_angular_fraction",
        "minimum",
        "fraction",
    ),
    "maximum_unsupported_arc_degrees": (
        "unsupported_arc_degrees",
        "maximum",
        "degrees",
    ),
    "maximum_radial_residual_p95_px": ("radial_residual_p95_px", "maximum", "px"),
    "minimum_quadrant_support_fraction": (
        "quadrant_support_fraction",
        "minimum",
        "fraction",
    ),
    "maximum_between_window_center_displacement_px": (
        "between_window_center_displacement_px",
        "maximum",
        "px",
    ),
    "maximum_between_window_radius_range_px": (
        "between_window_radius_range_px",
        "maximum",
        "px",
    ),
    "maximum_rim_family_center_spread_px": (
        "rim_family_center_spread_px",
        "maximum",
        "px",
    ),
    "maximum_rim_family_radius_hausdorff_distance_px": (
        "rim_family_radius_hausdorff_distance_px",
        "maximum",
        "px",
    ),
    "minimum_rim_family_candidate_count": (
        "rim_family_candidate_count",
        "minimum",
        "count",
    ),
    "minimum_acquisition_boundary_support_fraction": (
        "acquisition_boundary_support_fraction",
        "minimum",
        "fraction",
    ),
    "maximum_acquisition_boundary_radial_offset_px": (
        "acquisition_boundary_radial_offset_px",
        "maximum",
        "px",
    ),
}
_PREREQUISITES = (
    "producer_geometry_valid",
    "independent_fit_valid",
    "coordinate_bindings_match",
    "fit_frozen_before_acquisition_reveal",
)
_EVIDENCE_FIELDS = {
    "schema_id",
    "schema_version",
    "source_bindings",
    "semantic_compatibility",
    *_PREREQUISITES,
    "acquisition_gate",
    "additional_palette_tolerance_px",
    "boundary_inclusion",
    "same_feature_physical_boundary_metrics",
    "metrics",
    "applicability",
}
_CALIBRATION_FIELDS = {
    "derivation_manifest_sha256",
    "threshold_derivation_sha256",
    "frozen_at_utc",
    "validation_manifest_sha256",
    "validation_result_sha256",
    "scientific_recipe_sha256",
    "applicability",
}
_APPLICABILITY_FIELDS = {
    "rig_id",
    "canvas_name",
    "arena_id",
    "camera_serial",
    "coordinate_profile_id",
    "native_width_px",
    "native_height_px",
}


def _applicability(value: Any) -> None:
    scope = _exact(value, _APPLICABILITY_FIELDS, "applicability")
    for key in (
        "rig_id",
        "canvas_name",
        "arena_id",
        "camera_serial",
        "coordinate_profile_id",
    ):
        if not isinstance(scope[key], str) or not scope[key].strip():
            raise ValueError(
                "applicability requires exact rig/canvas/arena/camera/coordinate identities."
            )
    for key in ("native_width_px", "native_height_px"):
        if type(scope[key]) is not int or scope[key] <= 0:
            raise ValueError("applicability requires positive exact native dimensions.")


def _copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _exact(
    value: Any, keys: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} has missing or unsupported fields.")
    return value


def _digest(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")


def _number(value: Any, *, unit: str, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number.")
    if unit == "count" and (not isinstance(value, int)):
        raise ValueError(f"{label} must be an integer count.")
    if (unit == "fraction" and value > 1) or (unit == "degrees" and value > 360):
        raise ValueError(f"{label} is outside its {unit} domain.")


def build_geometry_auto_policy(
    *,
    thresholds: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inactive policy; explicit thresholds require frozen evidence.

    Calibration bindings are evidence references, not proof of promotion. Even
    a supplied validation result cannot enable selection in this schema.
    """
    payload = {
        "schema_id": POLICY_SCHEMA_ID,
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "policy_version": 1,
        "mode": "shadow",
        "thresholds": _copy(thresholds),
        "calibration": _copy(calibration),
    }
    payload["digest"] = canonical_json_sha256(payload)
    validate_geometry_auto_policy(payload)
    return payload


def validate_geometry_auto_policy(policy: Mapping[str, Any]) -> None:
    _exact(
        policy,
        {
            "schema_id",
            "schema_version",
            "policy_id",
            "policy_version",
            "mode",
            "thresholds",
            "calibration",
            "digest",
        },
        "policy",
    )
    if (
        policy["schema_id"] != POLICY_SCHEMA_ID
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["policy_id"] != POLICY_ID
        or type(policy["policy_version"]) is not int
        or policy["policy_version"] != 1
        or policy["mode"] != "shadow"
    ):
        raise ValueError("Unsupported shadow policy; activation is not implemented.")
    _digest(policy["digest"], "policy.digest")
    if (
        canonical_json_sha256({k: v for k, v in policy.items() if k != "digest"})
        != policy["digest"]
    ):
        raise ValueError("Policy digest does not match its frozen configuration.")
    thresholds = policy["thresholds"]
    calibration = policy["calibration"]
    if thresholds is None:
        if calibration is not None:
            raise ValueError("calibration cannot exist without thresholds.")
        return
    _exact(thresholds, set(THRESHOLD_CONTRACT), "thresholds")
    for name, (_metric, _direction, unit) in THRESHOLD_CONTRACT.items():
        _number(thresholds[name], unit=unit, label=name)
        if unit == "count" and thresholds[name] < 1:
            raise ValueError(f"{name} cannot permit absent evidence.")
    _exact(calibration, _CALIBRATION_FIELDS, "calibration")
    for name in ("derivation_manifest_sha256", "threshold_derivation_sha256"):
        _digest(calibration[name], "calibration." + name)
    _digest(
        calibration["scientific_recipe_sha256"], "calibration.scientific_recipe_sha256"
    )
    _applicability(calibration["applicability"])
    validation = (
        calibration["validation_manifest_sha256"],
        calibration["validation_result_sha256"],
    )
    if any(value is None for value in validation) != all(
        value is None for value in validation
    ):
        raise ValueError(
            "calibration validation manifest and result must be supplied together."
        )
    for value in validation:
        if value is not None:
            _digest(value, "calibration.validation")
    try:
        frozen = datetime.fromisoformat(
            calibration["frozen_at_utc"].replace("Z", "+00:00")
        )
        if frozen.utcoffset() is None or frozen.utcoffset().total_seconds() != 0:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "calibration.frozen_at_utc must be an explicit UTC timestamp."
        ) from exc


def validate_geometry_shadow_evidence(evidence: Mapping[str, Any]) -> None:
    _exact(evidence, _EVIDENCE_FIELDS, "shadow evidence")
    if (
        evidence["schema_id"] != EVIDENCE_SCHEMA_ID
        or type(evidence["schema_version"]) is not int
        or evidence["schema_version"] != 1
    ):
        raise ValueError("Unsupported shadow evidence schema.")
    bindings = _exact(
        evidence["source_bindings"], SOURCE_BINDING_FIELDS, "source_bindings"
    )
    for key, value in bindings.items():
        _digest(value, "source_bindings." + key)
    _applicability(evidence["applicability"])
    for key in _PREREQUISITES:
        if type(evidence[key]) is not bool:
            raise ValueError(f"{key} must be an explicit boolean.")
    semantics = evidence["semantic_compatibility"]
    if semantics not in {
        "same_feature_confirmed",
        "different_feature_confirmed",
        "projected_edges_unresolved",
    }:
        raise ValueError("Unsupported feature semantics.")
    physical = evidence["same_feature_physical_boundary_metrics"]
    if (semantics == "same_feature_confirmed") != isinstance(physical, Mapping):
        raise ValueError(
            "Same-feature metrics require independently confirmed same-feature evidence."
        )
    if (
        evidence["additional_palette_tolerance_px"] != 0.0
        or isinstance(evidence["additional_palette_tolerance_px"], bool)
        or evidence["boundary_inclusion"] != "inclusive"
    ):
        raise ValueError(
            "The acquisition gate and inclusive boundary must remain unchanged."
        )
    circle = _exact(
        evidence["acquisition_gate"],
        {"type", "center_px", "radius_px"},
        "acquisition_gate",
    )
    center = _exact(circle["center_px"], {"x", "y"}, "acquisition_gate.center_px")
    if circle["type"] != "circle":
        raise ValueError("Only unchanged circular acquisition gates are supported.")
    for name, value in {**center, "radius_px": circle["radius_px"]}.items():
        _number(value, unit="px", label="acquisition_gate." + name)
    if circle["radius_px"] <= 0:
        raise ValueError("Acquisition gate radius must be positive.")
    metrics = _exact(
        evidence["metrics"],
        {spec[0] for spec in THRESHOLD_CONTRACT.values()},
        "metrics",
    )
    for metric, _direction, unit in THRESHOLD_CONTRACT.values():
        _number(metrics[metric], unit=unit, label="metrics." + metric)
    # Strict JSON identity prevents silently coercing tuples/non-JSON values.
    if _copy(evidence) != dict(evidence):
        raise ValueError("Shadow evidence must be strict JSON data.")


def evaluate_geometry_auto_policy(
    *,
    policy: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate fully bound numerical evidence without any reviewer or writes."""
    validate_geometry_auto_policy(policy)
    validate_geometry_shadow_evidence(evidence)
    reasons = [name + "_required" for name in _PREREQUISITES if not evidence[name]]
    if evidence["semantic_compatibility"] == "different_feature_confirmed":
        reasons.append("different_physical_features_require_review")
    thresholds = policy["thresholds"]
    if thresholds is None:
        reasons.append("thresholds_not_calibrated")
    else:
        if policy["calibration"]["applicability"] != evidence["applicability"]:
            reasons.append("calibration_applicability_mismatch")
        if (
            policy["calibration"]["scientific_recipe_sha256"]
            != evidence["source_bindings"]["scientific_recipe_sha256"]
        ):
            reasons.append("calibration_scientific_recipe_mismatch")
        for name, (metric, direction, _unit) in THRESHOLD_CONTRACT.items():
            measured = evidence["metrics"][metric]
            passed = (
                measured >= thresholds[name]
                if direction == "minimum"
                else measured <= thresholds[name]
            )
            if not passed:
                reasons.append("threshold_failed:" + name)
    passed = not reasons
    result = {
        "schema_id": EVALUATION_SCHEMA_ID,
        "schema_version": 1,
        "mode": "shadow",
        "policy_id": POLICY_ID,
        "policy_digest": policy["digest"],
        "evidence_sha256": canonical_json_sha256(evidence),
        "source_bindings": _copy(evidence["source_bindings"]),
        "thresholds_satisfied": passed
        if thresholds is not None
        else (False if len(reasons) > 1 else None),
        "evidence_outcome": "shadow_operational_corroboration_pass"
        if passed
        else "shadow_review_required",
        "reason_codes": reasons,
        "semantic_compatibility": evidence["semantic_compatibility"],
        "same_feature_claim": False,
        "automatic_selection_promoted": False,
        "selection_performed": False,
        "scientific_acceptance_created": False,
        "acquisition_gate_modified": False,
        "acquisition_gate": _copy(evidence["acquisition_gate"]),
        "activation_blockers": [
            "shadow_schema_never_authorizes_selection",
            "policy_promotion_and_use_scoped_acceptance_not_implemented",
        ],
    }
    result["digest"] = canonical_json_sha256(result)
    return result
