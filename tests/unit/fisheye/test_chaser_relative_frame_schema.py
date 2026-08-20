from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.zarr.chaser_relative_frame_schema import (
    CHASER_RELATIVE_FRAME_REASON_CODES,
    CHASER_RELATIVE_FRAME_SCHEMA_V1,
    ChaserRelativeFrameDimensions,
    ChaserRelativeFrameSchemaError,
)
from fisheye.analysis_workflows.chaser_relative_frame import (
    BODY_REASON_CODES,
    EGOCENTRIC_REASON_CODES,
    NEAREST_REASON_CODES,
    RELATIVE_REASON_CODES,
)


def _base_fixture() -> tuple[
    ChaserRelativeFrameDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions = ChaserRelativeFrameDimensions(n_rows=3)
    base = {
        "acquisition_frame_id": np.asarray([0, 1, 2], dtype=np.int64),
        "track_sample_id": np.asarray([10, 11, 12], dtype=np.int64),
        "timestamp_ns": np.asarray([100, 200, 300], dtype=np.int64),
        "timestamp_valid": np.asarray([True, True, True], dtype=bool),
        "timestamp_reason_code": np.zeros(3, dtype=np.uint16),
        "fish_source_row_id": np.asarray([0, 1, 2], dtype=np.int64),
        "fish_source_row_valid": np.asarray([True, True, True], dtype=bool),
        "fish_source_row_reason_code": np.zeros(3, dtype=np.uint16),
        "chaser_source_row_id": np.asarray([20, 21, 22], dtype=np.int64),
        "chaser_source_row_valid": np.asarray([True, True, True], dtype=bool),
        "chaser_source_row_reason_code": np.zeros(3, dtype=np.uint16),
        "fish_position_xy_px": np.asarray(
            [[100, 100], [100, 100], [100, 100]], dtype=np.float32
        ),
        "fish_position_valid": np.asarray([True, True, True], dtype=bool),
        "fish_position_reason_code": np.zeros(3, dtype=np.uint16),
        "chaser_position_xy_px": np.asarray(
            [[103, 104], [100, 105], [97, 104]], dtype=np.float32
        ),
        "chaser_position_valid": np.asarray([True, True, True], dtype=bool),
        "chaser_position_reason_code": np.zeros(3, dtype=np.uint16),
        "fish_identity_code": np.asarray([1, 1, 1], dtype=np.uint16),
        "chaser_identity_code": np.asarray([10, 10, 11], dtype=np.uint16),
        "chaser_behavior_role_code": np.asarray([1, 1, 2], dtype=np.uint8),
        "chaser_behavior_role_valid": np.asarray([True, True, True], dtype=bool),
        "chaser_behavior_role_reason_code": np.zeros(3, dtype=np.uint16),
        "selection_member": np.asarray([True, True, False], dtype=bool),
        "chaser_occurrence_member": np.asarray([True, True, True], dtype=bool),
        "trial_id": np.asarray([4, 4, 5], dtype=np.int64),
        "trial_valid": np.asarray([True, True, True], dtype=bool),
        "trial_reason_code": np.zeros(3, dtype=np.uint16),
        "active_state_code": np.asarray([1, 1, 0], dtype=np.uint8),
        "active_state_valid": np.asarray([True, True, True], dtype=bool),
        "active_state_reason_code": np.zeros(3, dtype=np.uint16),
        "row_valid": np.asarray([True, True, True], dtype=bool),
        "row_reason_code": np.zeros(3, dtype=np.uint16),
        "acquisition_frame_delta": np.asarray([-1, 1, 1], dtype=np.int64),
        "timestamp_delta_ns": np.asarray([-1, 100, 100], dtype=np.int64),
        "fish_transition_valid": np.asarray([False, True, True], dtype=bool),
        "fish_transition_reason_code": np.asarray([13, 0, 0], dtype=np.uint16),
        "relative_transition_valid": np.asarray([False, True, True], dtype=bool),
        "relative_transition_reason_code": np.asarray([13, 0, 0], dtype=np.uint16),
        "relative_vector_px_xy": np.asarray(
            [[3, 4], [0, 5], [-3, 4]], dtype=np.float32
        ),
        "relative_distance_px": np.asarray([5, 5, 5], dtype=np.float32),
        "relative_px_valid": np.asarray([True, True, True], dtype=bool),
        "relative_px_reason_code": np.zeros(3, dtype=np.uint16),
        "relative_vector_physical_xy": np.asarray(
            [[0.3, 0.4], [0, 0.5], [-0.3, 0.4]], dtype=np.float32
        ),
        "relative_distance_physical": np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        "relative_physical_valid": np.asarray([True, True, True], dtype=bool),
        "relative_physical_reason_code": np.zeros(3, dtype=np.uint16),
        "nearest_chaser_member": np.asarray([True, False, True], dtype=bool),
        "nearest_chaser_identity_code": np.asarray([10, 10, 11], dtype=np.uint16),
        "nearest_chaser_source_row_id": np.asarray([20, 20, 22], dtype=np.int64),
        "nearest_chaser_distance_px": np.asarray([5, 5, 5], dtype=np.float32),
        "nearest_chaser_distance_physical": np.asarray(
            [0.5, 0.5, 0.5], dtype=np.float32
        ),
        "nearest_chaser_valid": np.asarray([True, True, True], dtype=bool),
        "nearest_chaser_reason_code": np.zeros(3, dtype=np.uint16),
    }

    forward = np.asarray([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    left = np.asarray([[0, -1], [0, -1], [0, -1]], dtype=np.float32)
    body = {
        "body_source_row_id": np.asarray([100, 101, 102], dtype=np.int64),
        "body_source_row_valid": np.asarray([True, True, True], dtype=bool),
        "body_source_row_reason_code": np.zeros(3, dtype=np.uint16),
        "body_origin_xy_px": np.asarray(
            [[102, 100], [100, 100], [96, 100]], dtype=np.float32
        ),
        "body_forward_axis_xy": forward,
        "body_left_axis_xy": left,
        "body_origin_valid": np.asarray([True, True, True], dtype=bool),
        "body_origin_reason_code": np.zeros(3, dtype=np.uint16),
        "body_axes_valid": np.asarray([True, True, True], dtype=bool),
        "body_axes_reason_code": np.zeros(3, dtype=np.uint16),
        "body_relative_vector_px_xy": np.asarray(
            [[1, 4], [0, 5], [1, 4]], dtype=np.float32
        ),
        "body_relative_px_valid": np.asarray([True, True, True], dtype=bool),
        "body_relative_px_reason_code": np.zeros(3, dtype=np.uint16),
        "body_relative_vector_physical_xy": np.asarray(
            [[0.1, 0.4], [0, 0.5], [0.1, 0.4]], dtype=np.float32
        ),
        "body_relative_physical_valid": np.asarray([True, True, True], dtype=bool),
        "body_relative_physical_reason_code": np.zeros(3, dtype=np.uint16),
        "body_heading_deg": np.zeros(3, dtype=np.float32),
        "body_heading_valid": np.asarray([True, True, True], dtype=bool),
        "body_heading_reason_code": np.zeros(3, dtype=np.uint16),
        "body_heading_transition_valid": np.asarray([False, True, True], dtype=bool),
        "body_heading_transition_reason_code": np.asarray(
            [13, 0, 0], dtype=np.uint16
        ),
        "body_forward_coordinate_px": np.asarray([1, 0, 1], dtype=np.float32),
        "body_left_coordinate_px": np.asarray([-4, -5, -4], dtype=np.float32),
        "body_coordinates_px_valid": np.asarray([True, True, True], dtype=bool),
        "body_coordinates_px_reason_code": np.zeros(3, dtype=np.uint16),
        "body_forward_coordinate_physical": np.asarray(
            [0.1, 0, 0.1], dtype=np.float32
        ),
        "body_left_coordinate_physical": np.asarray(
            [-0.4, -0.5, -0.4], dtype=np.float32
        ),
        "body_coordinates_physical_valid": np.asarray(
            [True, True, True], dtype=bool
        ),
        "body_coordinates_physical_reason_code": np.zeros(3, dtype=np.uint16),
        "body_bearing_deg": np.rad2deg(
            np.arctan2(np.asarray([-4, -5, -4]), np.asarray([1, 0, 1]))
        ).astype(np.float32),
        "body_bearing_valid": np.asarray([True, True, True], dtype=bool),
        "body_bearing_reason_code": np.zeros(3, dtype=np.uint16),
        "body_valid": np.asarray([True, True, True], dtype=bool),
        "body_reason_code": np.zeros(3, dtype=np.uint16),
    }
    return dimensions, base, body


def test_base_schema_accepts_without_body_extension() -> None:
    dimensions, base, _ = _base_fixture()

    CHASER_RELATIVE_FRAME_SCHEMA_V1.require(base, dimensions=dimensions)

    manifest = CHASER_RELATIVE_FRAME_SCHEMA_V1.as_manifest(dimensions=dimensions)
    assert manifest["layout"] == "frame_x_chaser_sparse_rows_v1"
    assert manifest["invariants"]["row_axis"] == "frame_x_chaser"
    assert manifest["body_extension"]["schema_version"] == 1
    paths = {binding["path"] for binding in manifest["bindings"]}
    assert "selection_member" in paths
    assert "selected_chaser_member" not in paths
    assert {
        "acquisition_frame_delta",
        "timestamp_delta_ns",
        "fish_transition_valid",
        "relative_transition_valid",
    } <= paths
    assert "repeated for each chaser row" in manifest["invariants"]["frame_only_evidence"]


def test_reason_registry_matches_pure_computation_vocabularies() -> None:
    expected = set(RELATIVE_REASON_CODES)
    expected.update(NEAREST_REASON_CODES)
    expected.update(BODY_REASON_CODES)
    expected.update(EGOCENTRIC_REASON_CODES)
    expected.update(
        {
            "no_predecessor",
            "nonconsecutive_acquisition_frame",
            "timestamp_unavailable",
            "nonpositive_timestamp_delta",
            "selection_boundary",
            "occurrence_boundary",
            "trial_boundary",
            "invalid_current_or_previous_position",
            "invalid_current_or_previous_body_frame",
        }
    )

    assert expected < set(CHASER_RELATIVE_FRAME_REASON_CODES.values())
    assert {
        "source_row_unavailable",
        "trial_unavailable",
        "active_state_unavailable",
        "behavior_role_unavailable",
    } <= set(CHASER_RELATIVE_FRAME_REASON_CODES.values())


def test_schema_accepts_body_extension_and_validates_projections() -> None:
    dimensions, base, body = _base_fixture()

    CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
        base, dimensions=dimensions, body_arrays=body
    )


def test_body_projection_uses_body_origin_relative_vector() -> None:
    dimensions, base, body = _base_fixture()
    body["body_relative_vector_px_xy"][0] = np.asarray([3, 4], dtype=np.float32)

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(
        base, dimensions=dimensions, body_arrays=body
    )

    assert "body_pixel_projection_mismatch" in {issue.code for issue in issues}


def test_relative_projection_is_checked_against_absolute_positions() -> None:
    dimensions, base, _ = _base_fixture()
    base["chaser_position_xy_px"][0] = np.asarray([104, 104], dtype=np.float32)

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(base, dimensions=dimensions)

    assert "relative_pixel_derivation_mismatch" in {issue.code for issue in issues}


def test_missing_base_array_fails() -> None:
    dimensions, base, _ = _base_fixture()
    del base["chaser_source_row_id"]

    with pytest.raises(ChaserRelativeFrameSchemaError, match="missing_required_array"):
        CHASER_RELATIVE_FRAME_SCHEMA_V1.require(base, dimensions=dimensions)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("timestamp_ns", np.asarray([1, 2, 3], dtype=np.float32)),
        ("relative_vector_px_xy", np.zeros((3, 3), dtype=np.float32)),
        ("chaser_identity_code", np.asarray(["a", "b", "c"], dtype=object)),
    ],
)
def test_wrong_dtype_or_shape_fails(path: str, replacement: np.ndarray) -> None:
    dimensions, base, _ = _base_fixture()
    base[path] = replacement

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(base, dimensions=dimensions)

    assert any(issue.path == path for issue in issues)
    assert any(issue.code == "array_contract_violation" for issue in issues)


def test_optional_trial_and_active_state_require_validity_pairs() -> None:
    dimensions, base, _ = _base_fixture()
    del base["trial_valid"]
    del base["active_state_code"]

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(base, dimensions=dimensions)
    codes = {issue.code for issue in issues}

    assert "missing_optional_validity_pair" in codes


def test_invalid_float_values_require_nan_and_reason_code() -> None:
    dimensions, base, _ = _base_fixture()
    base["relative_px_valid"][1] = False
    base["relative_px_reason_code"][1] = 8

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(base, dimensions=dimensions)

    assert "invalid_value_not_nan" in {issue.code for issue in issues}


def test_body_extension_rejects_non_determinant_minus_one_axes() -> None:
    dimensions, base, body = _base_fixture()
    body["body_left_axis_xy"][0] = np.asarray([0, 1], dtype=np.float32)

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(
        base, dimensions=dimensions, body_arrays=body
    )

    assert "invalid_body_axes" in {issue.code for issue in issues}


def test_body_extension_rejects_heading_not_derived_from_forward_axis() -> None:
    dimensions, base, body = _base_fixture()
    body["body_heading_deg"][0] = np.float32(90)

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(
        base, dimensions=dimensions, body_arrays=body
    )

    assert "heading_derivation_mismatch" in {issue.code for issue in issues}


def test_invalid_body_rows_use_nan_float_values() -> None:
    dimensions, base, body = _base_fixture()
    body["body_axes_valid"][2] = False
    body["body_axes_reason_code"][2] = 12

    issues = CHASER_RELATIVE_FRAME_SCHEMA_V1.validate(
        base, dimensions=dimensions, body_arrays=body
    )

    assert "invalid_value_not_nan" in {issue.code for issue in issues}
