from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.provider_chaser_position_suite import (
    CircularArena,
    PositionSuiteConfig,
    PositionSuiteEpoch,
    ProviderChaserPositionSuiteError,
    compute_provider_chaser_position_suite,
)
from fisheye.utils.materialize_provider_chaser_position_suite_canary import (
    OUTPUT_FILES,
    ProviderChaserPositionSuiteCanaryError,
    _validate_physical_frame_semantic_equivalence,
    publish_operational_canary,
)


def _inputs() -> dict[str, object]:
    frame_ids = np.arange(6, dtype=np.int64)
    fish_xy = np.asarray(
        [[2, 2], [4, 4], [6, 6], [2, 8], [8, 2], [8, 8]],
        dtype=np.float32,
    )
    chaser_xy = np.empty((6, 2, 2), dtype=np.float32)
    chaser_xy[:, 0, :] = (4, 4)
    chaser_xy[:, 1, :] = (8, 8)
    distance_px = np.linalg.norm(chaser_xy - fish_xy[:, None, :], axis=2).astype(
        np.float32
    )
    return {
        "frame_ids": frame_ids,
        "fish_xy_px": fish_xy,
        "fish_valid": np.ones(6, dtype=bool),
        "chaser_xy_px": chaser_xy,
        "chaser_valid": np.ones((6, 2), dtype=bool),
        "distance_px": distance_px,
        "distance_px_valid": np.ones((6, 2), dtype=bool),
        "distance_mm": distance_px.copy(),
        "distance_mm_valid": np.ones((6, 2), dtype=bool),
        "selection_member": np.ones(6, dtype=bool),
        "chaser_occurrence_member": np.ones((6, 2), dtype=bool),
        "chaser_role_codes": np.tile(np.asarray([[1, 2]], dtype=np.uint8), (6, 1)),
        "chaser_role_valid": np.ones((6, 2), dtype=bool),
        "chaser_identity_codes": np.tile(
            np.asarray([[10, 20]], dtype=np.uint16), (6, 1)
        ),
        "role_registry": {"1": "aggressive", "2": "inert"},
        "chaser_registry": {"10": "chaser_a", "20": "chaser_b"},
        "epochs": (
            PositionSuiteEpoch(
                analysis_role="pre",
                window_id=0,
                source_label="black_before",
                start_frame=0,
                end_frame=3,
                source_interval_sha256="a" * 64,
            ),
            PositionSuiteEpoch(
                analysis_role="training",
                window_id=1,
                source_label="chaser",
                start_frame=3,
                end_frame=6,
                source_interval_sha256="b" * 64,
            ),
        ),
        "arena": CircularArena(
            center_x_px=5,
            center_y_px=5,
            radius_px=5,
            boundary_role="visible_dish_top_rim_edge",
            observed_feature="visible_dish_top_rim_edge",
        ),
        "mm_per_pixel": 1.0,
        "fps": 2.0,
        "config": PositionSuiteConfig(
            radial_bin_width_mm=2,
            cdf_thresholds_mm=(1, 3, 5, 8),
            near_zone_radius_mm=3,
            near_entry_radius_mm=3,
            near_exit_radius_mm=4,
            perimeter_band_mm=1,
            min_expected_count=0.1,
        ),
    }


def _result(**overrides: object) -> dict[str, object]:
    values = _inputs()
    values.update(overrides)
    return compute_provider_chaser_position_suite(**values)


def _metric(result: dict[str, object], epoch: str, role: str) -> dict[str, object]:
    rows = result["per_epoch_chaser_metrics"]
    present = [
        row
        for row in rows
        if row["analysis_role"] == epoch and row["behavior_role"] == role
    ]
    assert len(present) == 1
    return present[0]


def test_exact_half_open_epochs_and_native_y_down_quadrants() -> None:
    result = _result()
    aggressive = _metric(result, "pre", "aggressive")
    inert = _metric(result, "pre", "inert")

    assert aggressive["epoch_provider_frame_count"] == 3
    assert aggressive["candidate_frame_count"] == 3
    assert aggressive["same_quadrant_fraction_valid"] == pytest.approx(2 / 3)
    assert inert["same_quadrant_fraction_valid"] == pytest.approx(1 / 3)
    assert result["policies"]["quadrant"] == (
        "selected_circle_center_native_xy_y_down_v1"
    )
    assert result["epoch_roles"][0]["end_frame_exclusive"] == 3


def test_near_field_uses_valid_denominator_and_invalid_gap_censoring() -> None:
    selected = np.ones(6, dtype=bool)
    selected[1] = False
    result = _result(selection_member=selected)
    aggressive = _metric(result, "pre", "aggressive")

    assert aggressive["candidate_frame_count"] == 2
    assert aggressive["valid_distance_frame_count"] == 2
    assert aggressive["near_zone_valid_tracked_duration_s"] == pytest.approx(1.0)
    assert aggressive["near_zone_invalid_gap_count"] >= 1
    assert result["policies"]["visit"] == "valid_tracked_observed_transitions_v2"


def test_radial_geometric_null_and_explicit_role_contrast_are_materialized() -> None:
    result = _result()
    rows = [
        row
        for row in result["radial_occupancy"]
        if row["analysis_role"] == "pre" and row["behavior_role"] == "aggressive"
    ]
    expected = [
        row["expected_fraction_geometric"]
        for row in rows
        if row["expected_fraction_geometric"] is not None
    ]
    assert sum(expected) == pytest.approx(1.0)
    contrast = [
        row
        for row in result["role_contrasts"]
        if row["analysis_role"] == "pre"
        and row["metric"] == "same_quadrant_fraction_valid"
    ]
    assert len(contrast) == 1
    assert contrast[0]["treatment_role"] == "aggressive"
    assert contrast[0]["baseline_role"] == "inert"
    assert contrast[0]["treatment_minus_baseline"] == pytest.approx(1 / 3)


def test_persisted_distance_must_match_coordinates_and_exact_scale() -> None:
    values = _inputs()
    wrong = np.asarray(values["distance_mm"]).copy()
    wrong[0, 0] += 1
    with pytest.raises(
        ProviderChaserPositionSuiteError,
        match="exact physical scale",
    ):
        _result(distance_mm=wrong)


def test_role_contrast_fails_closed_when_role_is_ambiguous() -> None:
    roles = np.ones((6, 2), dtype=np.uint8)
    with pytest.raises(
        ProviderChaserPositionSuiteError,
        match="exactly one chaser column",
    ):
        _result(chaser_role_codes=roles)


def test_operational_publication_is_atomic_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    report = {
        "schema_id": "palette.provider_chaser_position_suite_canary",
        "schema_version": 1,
        "recording_id": "recording-fixture",
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "suite": _result(),
    }
    target = tmp_path / "canary"
    published = publish_operational_canary(report, output_dir=target)

    assert published["status"] == "published_selector_ineligible_operational_canary"
    assert published["artifact_manifest"]["selector_eligible"] is False
    for name in (*OUTPUT_FILES, "artifact_manifest.json"):
        assert (target / name).is_file()
    with pytest.raises(FileExistsError):
        publish_operational_canary(report, output_dir=target)


def _physical_frame() -> dict[str, object]:
    return {
        "schema_id": "palette.physical_frame_calibration",
        "schema_version": 1,
        "kind": "physical_frame_calibration",
        "frame_id": "recording_camera_physical_mm",
        "coordinate_units": "mm",
        "origin": "physical_frame_origin",
        "source_origin_relation": "coincident_with_source_camera_top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "compatible_profile_ids": ["physical_mm.source_camera_y_down.v1"],
        "source_space_id": "source_camera_image_px",
        "source_camera_pixels": {
            "record_ref": "/analysis/coordinate_frames/source_camera/camera/continuous@pixel_frame_authority",
            "record_sha256": "a" * 64,
        },
        "selected_camera_evidence": {
            "record_ref": "/analysis/calibration/selected@evidence",
            "record_sha256": "b" * 64,
        },
        "camera_id": "camera",
        "scale": {
            "quantity": "pixels_per_mm_camera",
            "pixels_per_mm_camera": 50.0,
            "mm_per_pixel": 0.02,
            "derivation": "exact_binary64_reciprocal_of_selected_pixels_per_mm_camera_v1",
        },
        "physical_extent": {
            "mode": "finite",
            "width": 90.0,
            "height": 90.0,
            "units": "mm",
        },
    }


def test_physical_scale_equivalence_allows_only_path_scoped_identity_fields() -> None:
    recording = _physical_frame()
    provider = _physical_frame()
    provider["frame_id"] = "stimulus_scoped_camera_physical_mm"
    provider["selected_camera_evidence"] = {
        "record_ref": "/analysis/stimulus_runs/run/calibration/selected@evidence",
        "record_sha256": "b" * 64,
    }

    result = _validate_physical_frame_semantic_equivalence(provider, recording)
    assert result["policy_id"] == "physical_frame_path_scoped_identity_equivalence_v1"

    provider["scale"] = {**provider["scale"], "mm_per_pixel": 0.03}
    with pytest.raises(
        ProviderChaserPositionSuiteCanaryError,
        match="differ beyond",
    ):
        _validate_physical_frame_semantic_equivalence(provider, recording)
