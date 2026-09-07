from __future__ import annotations

from copy import deepcopy

import pytest

from fisheye.shared import arena_geometry_auto_policy as policy


def _applicability():
    return {
        "rig_id": "omnifin0",
        "canvas_name": "shadow",
        "arena_id": "arena_1",
        "camera_serial": "2010093",
        "coordinate_profile_id": "source_camera_image_px.top_left_y_down.v1",
        "native_width_px": 640,
        "native_height_px": 480,
    }


def _policy():
    thresholds = {
        name: (0.8 if direction == "minimum" else 10.0)
        for name, (_metric, direction, _unit) in policy.THRESHOLD_CONTRACT.items()
    }
    for name, (_metric, direction, unit) in policy.THRESHOLD_CONTRACT.items():
        if unit == "fraction":
            thresholds[name] = 0.8 if direction == "minimum" else 0.05
        if unit == "count":
            thresholds[name] = 2
    return policy.build_geometry_auto_policy(
        thresholds=thresholds,
        calibration={
            "derivation_manifest_sha256": "a" * 64,
            "threshold_derivation_sha256": "b" * 64,
            "frozen_at_utc": "2026-09-06T12:00:00Z",
            "validation_manifest_sha256": None,
            "validation_result_sha256": None,
            "scientific_recipe_sha256": "c" * 64,
            "applicability": _applicability(),
        },
    )


def _evidence():
    metrics = {}
    for _name, (metric, direction, unit) in policy.THRESHOLD_CONTRACT.items():
        metrics[metric] = 0.95 if direction == "minimum" else 0.01
        if unit == "count":
            metrics[metric] = 3
    return {
        "schema_id": policy.EVIDENCE_SCHEMA_ID,
        "schema_version": 1,
        "source_bindings": {key: "c" * 64 for key in policy.SOURCE_BINDING_FIELDS},
        "applicability": _applicability(),
        "semantic_compatibility": "projected_edges_unresolved",
        "producer_geometry_valid": True,
        "independent_fit_valid": True,
        "coordinate_bindings_match": True,
        "fit_frozen_before_acquisition_reveal": True,
        "acquisition_gate": {
            "type": "circle",
            "center_px": {"x": 100.0, "y": 100.0},
            "radius_px": 80.0,
        },
        "additional_palette_tolerance_px": 0.0,
        "boundary_inclusion": "inclusive",
        "same_feature_physical_boundary_metrics": None,
        "metrics": metrics,
    }


def test_unconfigured_policy_is_inactive_and_preserves_inputs():
    evidence = _evidence()
    original = deepcopy(evidence)
    result = policy.evaluate_geometry_auto_policy(
        policy=policy.build_geometry_auto_policy(), evidence=evidence
    )
    assert result["thresholds_satisfied"] is None
    assert result["reason_codes"] == ["thresholds_not_calibrated"]
    assert result["automatic_selection_promoted"] is False
    assert result["selection_performed"] is False
    assert evidence == original


def test_unresolved_operational_shadow_pass_never_claims_acceptance_or_same_edge():
    result = policy.evaluate_geometry_auto_policy(
        policy=_policy(), evidence=_evidence()
    )
    assert result["thresholds_satisfied"] is True
    assert result["evidence_outcome"] == "shadow_operational_corroboration_pass"
    assert result["semantic_compatibility"] == "projected_edges_unresolved"
    assert result["same_feature_claim"] is False
    assert result["selection_performed"] is False
    assert result["scientific_acceptance_created"] is False
    assert result["acquisition_gate_modified"] is False
    assert result["acquisition_gate"] == _evidence()["acquisition_gate"]
    assert result == policy.evaluate_geometry_auto_policy(
        policy=_policy(), evidence=_evidence()
    )


def test_shadow_v1_golden_digests_use_the_existing_canonical_manifest_serializer():
    assert (
        _policy()["digest"]
        == "fd96b0349dc3100ca79167831e44b10ae33b4c01bd91e1c280a2f2a4e724573f"
    )
    result = policy.evaluate_geometry_auto_policy(
        policy=_policy(), evidence=_evidence()
    )
    assert (
        result["digest"]
        == "a2f90b87e13cb0cd806d282852333b518d84acaf82acc7f2c93d8928a48f083d"
    )


@pytest.mark.parametrize(
    "field",
    [
        "producer_geometry_valid",
        "independent_fit_valid",
        "coordinate_bindings_match",
        "fit_frozen_before_acquisition_reveal",
    ],
)
def test_failed_prerequisite_never_passes(field):
    evidence = _evidence()
    evidence[field] = False
    result = policy.evaluate_geometry_auto_policy(policy=_policy(), evidence=evidence)
    assert result["thresholds_satisfied"] is False
    assert field + "_required" in result["reason_codes"]


def test_confirmed_different_feature_remains_review_required():
    evidence = _evidence()
    evidence["semantic_compatibility"] = "different_feature_confirmed"
    result = policy.evaluate_geometry_auto_policy(policy=_policy(), evidence=evidence)
    assert result["thresholds_satisfied"] is False
    assert "different_physical_features_require_review" in result["reason_codes"]


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e["metrics"].pop("circle_iou"),
        lambda e: e["metrics"].update(circle_iou=float("nan")),
        lambda e: e["metrics"].update(circle_iou=True),
        lambda e: e["metrics"].update(circle_iou=1.1),
        lambda e: e["source_bindings"].update(fit_report_sha256=""),
        lambda e: e.update(additional_palette_tolerance_px=1.0),
        lambda e: e.update(
            same_feature_physical_boundary_metrics={"radius_error_px": 0.0}
        ),
        lambda e: e.update(reviewer="automated"),
    ],
)
def test_malformed_wrong_use_or_semantic_evidence_rejected(change):
    evidence = _evidence()
    change(evidence)
    with pytest.raises(ValueError):
        policy.evaluate_geometry_auto_policy(policy=_policy(), evidence=evidence)


def test_each_threshold_is_enforced_independently():
    for name, (metric, direction, unit) in policy.THRESHOLD_CONTRACT.items():
        evidence = _evidence()
        evidence["metrics"][metric] = (
            0 if direction == "minimum" else (0.1 if unit == "fraction" else 11.0)
        )
        result = policy.evaluate_geometry_auto_policy(
            policy=_policy(), evidence=evidence
        )
        assert result["thresholds_satisfied"] is False, name
        assert "threshold_failed:" + name in result["reason_codes"]


def test_explicit_thresholds_require_frozen_derivation_binding():
    with pytest.raises(ValueError, match="calibration"):
        policy.build_geometry_auto_policy(thresholds=_policy()["thresholds"])


def test_wrong_camera_or_scientific_recipe_cannot_reuse_calibrated_thresholds():
    for label in ("camera_serial", "canvas_name", "arena_id", "recipe"):
        evidence = _evidence()
        if label != "recipe":
            evidence["applicability"][label] = "different_" + label
        else:
            evidence["source_bindings"]["scientific_recipe_sha256"] = "d" * 64
        result = policy.evaluate_geometry_auto_policy(
            policy=_policy(), evidence=evidence
        )
        assert result["thresholds_satisfied"] is False
        assert any(
            reason.startswith("calibration_") for reason in result["reason_codes"]
        )


def test_tampered_or_enabled_policy_rejected():
    for field, value in (
        ("mode", "active"),
        ("policy_version", 2),
        ("digest", "d" * 64),
    ):
        config = _policy()
        config[field] = value
        with pytest.raises(ValueError):
            policy.evaluate_geometry_auto_policy(policy=config, evidence=_evidence())
