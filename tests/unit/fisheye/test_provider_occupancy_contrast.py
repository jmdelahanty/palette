from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fisheye.analysis.provider_occupancy_contrast import (
    CONTRAST_FIELD,
    OccupancyContrastError,
    build_provider_occupancy_contrast,
    compute_occupancy_contrast,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _identity(identity: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "id": identity,
        "sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def _summary(
    role: str,
    *,
    provider_id: str = "detection_bbox_centroid.v1",
    estimator: object | None = None,
    source_suffix: str | None = None,
    fraction: object | None = None,
) -> dict[str, object]:
    suffix = source_suffix or role
    estimator = estimator or _identity(
        provider_id,
        {"estimator_id": provider_id, "version": 1},
    )
    payload = {"run": f"occupancy-{suffix}", "source": suffix}
    selection = {
        "selection_id": f"selection-{suffix}",
        "sha256": canonical_json_sha256({"selection": suffix}),
        "payload": {"selection": suffix},
    }
    occurrence_payload = {"occurrence": suffix}
    occurrence = _identity(f"occurrence-{suffix}", occurrence_payload)
    grid_payload = {
        "id": "goodbatbadbat_arena_mm_grid.v1",
        "x_edges": [0.0, 1.0, 2.0],
        "y_edges": [0.0, 1.0, 2.0],
    }
    return {
        "arm_role": role,
        "schema_family": "palette.provider_occupancy.v2",
        "schema_version": 2,
        "provider_id": provider_id,
        "estimator": estimator,
        "position_track_policy": _identity(
            "one_subject_track_sample_per_frame.v1",
            {"policy_id": "one_subject_track_sample_per_frame.v1"},
        ),
        "coordinate_frame": _identity(
            "arena_mm.v1",
            {"frame_id": "arena_mm.v1", "axis_order": "xy"},
        ),
        "transform": _identity(
            "camera_to_arena_mm.v1",
            {"transform_id": "camera_to_arena_mm.v1"},
        ),
        "geometry": _identity(
            "registered_dish.v1",
            {"geometry_id": "registered_dish.v1"},
        ),
        "grid": {
            **grid_payload,
            "sha256": canonical_json_sha256(grid_payload),
        },
        "sample_unit": "one_valid_subject_track_sample_per_acquisition_frame",
        "denominator": "valid_in_grid_sample_count",
        "normalization": "occupancy_fraction_of_valid_in_grid_samples",
        "recording_id": "recording-001",
        "subject_id": "fish-001",
        "timing_authority": _identity(
            "camera_acquisition_clock.v1",
            {"timing_id": "camera_acquisition_clock.v1"},
        ),
        "grid_policy": "fixed_declared_arena_mm_grid.v1",
        "edge_policy": "left_closed_right_open_final_outer_edge_inclusive_v1",
        "occupancy_fraction": (
            np.asarray(fraction, dtype=np.float64)
            if fraction is not None
            else np.asarray([[0.5, 0.25], [0.125, 0.125]], dtype=np.float64)
        ),
        "valid_sample_count": 8,
        "source_manifest": _identity(f"manifest-{suffix}", payload),
        "source_selections": [selection],
        "source_occurrences": [occurrence],
        # This must not leak into a difference-only result.
        "counts": np.asarray([[4, 2], [1, 1]], dtype=np.int64),
    }


def test_difference_preserves_roles_lineage_and_exact_values_without_display_normalization() -> None:
    baseline = _summary("baseline", fraction=[[0.5, 0.25], [0.125, 0.125]])
    treatment = _summary("treatment", fraction=[[0.25, 0.5], [0.125, 0.125]])

    result = compute_occupancy_contrast(
        baseline,
        treatment,
        config={"grid_profile": "goodbatbadbat_arena_mm_v1"},
    )

    np.testing.assert_array_equal(
        result[CONTRAST_FIELD],
        np.asarray([[-0.25, 0.25], [0.0, 0.0]], dtype=np.float64),
    )
    assert result["formula"] == (
        "treatment.occupancy_fraction - baseline.occupancy_fraction"
    )
    assert result["baseline_role"] == "baseline"
    assert result["treatment_role"] == "treatment"
    assert result["valid_sample_counts"] == {"baseline": 8, "treatment": 8}
    assert result["source_arms"]["baseline"]["role"] == "baseline"
    assert result["source_arms"]["treatment"]["role"] == "treatment"
    assert result["source_arms"]["baseline"]["source_selections"]
    assert result["source_arms"]["treatment"]["source_occurrences"]
    assert "counts" not in result
    assert len(result["policy_digest"]) == 64
    assert result["policy_digest"] == compute_occupancy_contrast(
        baseline,
        treatment,
        config={"grid_profile": "goodbatbadbat_arena_mm_v1"},
    )["policy_digest"]


def test_result_and_source_arrays_are_detached_and_sources_are_not_mutated() -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment", fraction=[[0.25, 0.5], [0.125, 0.125]])
    baseline_before = baseline["occupancy_fraction"].copy()
    treatment_before = treatment["occupancy_fraction"].copy()
    baseline_metadata = deepcopy(baseline["source_manifest"])

    result = build_provider_occupancy_contrast(baseline, treatment)
    result[CONTRAST_FIELD][0, 0] = 999.0
    result["source_arms"]["baseline"]["source_manifest"]["id"] = "changed"

    np.testing.assert_array_equal(baseline["occupancy_fraction"], baseline_before)
    np.testing.assert_array_equal(treatment["occupancy_fraction"], treatment_before)
    assert baseline["source_manifest"] == baseline_metadata
    assert "counts" not in result["source_arms"]["baseline"]


@pytest.mark.parametrize(
    "field",
    [
        "schema_family",
        "schema_version",
        "provider_id",
        "estimator",
        "position_track_policy",
        "coordinate_frame",
        "transform",
        "geometry",
        "sample_unit",
        "denominator",
        "normalization",
        "recording_id",
        "subject_id",
        "timing_authority",
        "grid_policy",
        "edge_policy",
    ],
)
def test_every_declared_compatibility_identity_must_match(field: str) -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    if field == "schema_version":
        treatment[field] = 3
    elif isinstance(treatment[field], dict):
        treatment[field] = {**treatment[field], "mismatch": True}
    else:
        treatment[field] = f"mismatch-{field}"

    with pytest.raises(OccupancyContrastError, match=field):
        compute_occupancy_contrast(baseline, treatment)


@pytest.mark.parametrize(
    "axis, replacement",
    [
        ("x_edges", [0.0, 1.0, 2.0000000001]),
        ("y_edges", [0.0, 1.0, 2.0000000001]),
    ],
)
def test_exact_grid_edges_are_required(axis: str, replacement: list[float]) -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    treatment["grid"] = {**treatment["grid"], axis: replacement}

    with pytest.raises(OccupancyContrastError, match="grid"):
        compute_occupancy_contrast(baseline, treatment)


def test_detection_and_keypoint_are_not_an_ordinary_contrast() -> None:
    baseline = _summary("baseline", provider_id="detection_bbox_centroid.v1")
    treatment = _summary("treatment", provider_id="keypoint_anatomical_triad_mean.v1")

    with pytest.raises(OccupancyContrastError, match="provider_id"):
        compute_occupancy_contrast(baseline, treatment)


@pytest.mark.parametrize(
    "arm, change",
    [
        ("baseline", lambda summary: summary.pop("source_manifest")),
        ("treatment", lambda summary: summary["source_manifest"].update({"sha256": ""})),
        (
            "baseline",
            lambda summary: summary["source_manifest"].update({"payload": {"changed": True}}),
        ),
        ("treatment", lambda summary: summary.update({"source_selections": []})),
        ("baseline", lambda summary: summary.update({"source_occurrences": []})),
    ],
)
def test_empty_or_stale_source_identities_fail_closed(arm: str, change) -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    change(baseline if arm == "baseline" else treatment)

    with pytest.raises(OccupancyContrastError, match="source"):
        compute_occupancy_contrast(baseline, treatment)


def test_conflicting_source_digest_aliases_fail_closed() -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    treatment["source_manifest"] = {
        **treatment["source_manifest"],
        "manifest_sha256": "a" * 64,
    }

    with pytest.raises(OccupancyContrastError, match="digest"):
        compute_occupancy_contrast(baseline, treatment)


def test_explicit_arm_roles_are_required() -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    del baseline["arm_role"]

    with pytest.raises(OccupancyContrastError, match="arm_role"):
        compute_occupancy_contrast(baseline, treatment)

    baseline = _summary("baseline")
    baseline["arm_role"] = "treatment"
    with pytest.raises(OccupancyContrastError, match="arm_role"):
        compute_occupancy_contrast(baseline, treatment)


@pytest.mark.parametrize(
    "field, value",
    [
        ("occupancy_fraction", np.asarray([[0.5]], dtype=np.float64)),
        ("occupancy_fraction", np.asarray([[np.nan, 0.0], [0.0, 1.0]])),
        ("valid_sample_count", 0),
    ],
)
def test_empty_or_shape_incompatible_summaries_fail_closed(field: str, value: object) -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    treatment[field] = value

    with pytest.raises(OccupancyContrastError):
        compute_occupancy_contrast(baseline, treatment)


def test_config_cannot_turn_v1_into_an_arbitrary_formula_or_cohort_operation() -> None:
    with pytest.raises(OccupancyContrastError, match="config"):
        compute_occupancy_contrast(
            _summary("baseline"),
            _summary("treatment"),
            config={"formula": "baseline - treatment"},
        )


def test_equivalent_mapping_order_has_the_same_policy_digest() -> None:
    baseline = _summary("baseline")
    treatment = _summary("treatment")
    first = compute_occupancy_contrast(
        baseline, treatment, config={"z": 2, "a": {"b": 1, "a": 0}}
    )
    second = compute_occupancy_contrast(
        baseline, treatment, config={"a": {"a": 0, "b": 1}, "z": 2}
    )
    assert first["config_digest"] == second["config_digest"]
    assert first["policy_digest"] == second["policy_digest"]
