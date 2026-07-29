from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    MAX_CANONICAL_CLASS_ID,
    CanonicalDetectionDimensions,
    CanonicalDetectionSchemaError,
)


def _path(name: str) -> str:
    return f"instances/{name}"


def _project_geometry(
    boxes: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    half = np.float32(0.5)
    width_px = np.float32(width)
    height_px = np.float32(height)
    bbox_img = np.column_stack(
        (
            (boxes[:, 0] - boxes[:, 2] * half) * width_px,
            (boxes[:, 1] - boxes[:, 3] * half) * height_px,
            (boxes[:, 0] + boxes[:, 2] * half) * width_px,
            (boxes[:, 1] + boxes[:, 3] * half) * height_px,
        )
    ).astype(np.float32, copy=False)
    centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * half,
            (bbox_img[:, 1] + bbox_img[:, 3]) * half,
        )
    ).astype(np.float32, copy=False)
    return bbox_img, centers


def _valid_payload() -> tuple[CanonicalDetectionDimensions, dict[str, np.ndarray]]:
    dimensions = CanonicalDetectionDimensions(
        n_frames=4,
        n_instances=6,
        source_width=640,
        source_height=480,
    )
    frame_indices = np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int32)
    bbox_norm = np.asarray(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.25, 0.30, 0.10, 0.20],
            [0.75, 0.60, 0.20, 0.10],
            [0.20, 0.20, 0.10, 0.10],
            [0.50, 0.75, 0.30, 0.20],
            [0.80, 0.25, 0.10, 0.20],
        ],
        dtype=np.float32,
    )
    bbox_img, centers = _project_geometry(
        bbox_norm,
        width=dimensions.source_width,
        height=dimensions.source_height,
    )
    arrays = {
        _path("frame_indices"): frame_indices,
        _path("source_acquisition_frame_index"): frame_indices.astype(np.int64),
        _path("instance_key"): np.arange(100, 106, dtype=np.uint64),
        _path("bbox_norm_coords"): bbox_norm,
        _path("bbox_img_xyxy"): bbox_img,
        _path("centers_img_xy"): centers,
        _path("scores"): np.asarray(
            [0.95, 0.80, 0.70, 1.0, 0.50, 0.0],
            dtype=np.float32,
        ),
        _path("class_ids"): np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        _path("frame_row_offsets"): np.asarray([0, 2, 2, 3, 6], dtype=np.int64),
    }
    return dimensions, arrays


def _issue_codes(
    arrays: dict[str, np.ndarray],
    dimensions: CanonicalDetectionDimensions,
) -> set[str]:
    return {
        issue.code
        for issue in CANONICAL_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
    }


def test_sparse_instances_allow_zero_one_or_many_rows_per_frame() -> None:
    dimensions, arrays = _valid_payload()

    assert (
        CANONICAL_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
        == ()
    )
    CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)

    offsets = arrays[_path("frame_row_offsets")]
    assert tuple(np.diff(offsets)) == (2, 0, 1, 3)


def test_completely_empty_instance_table_is_valid() -> None:
    dimensions = CanonicalDetectionDimensions(
        n_frames=3,
        n_instances=0,
        source_width=640,
        source_height=480,
    )
    arrays = {
        _path("frame_indices"): np.empty((0,), dtype=np.int32),
        _path("source_acquisition_frame_index"): np.empty((0,), dtype=np.int64),
        _path("instance_key"): np.empty((0,), dtype=np.uint64),
        _path("bbox_norm_coords"): np.empty((0, 4), dtype=np.float32),
        _path("bbox_img_xyxy"): np.empty((0, 4), dtype=np.float32),
        _path("centers_img_xy"): np.empty((0, 2), dtype=np.float32),
        _path("scores"): np.empty((0,), dtype=np.float32),
        _path("class_ids"): np.empty((0,), dtype=np.int32),
        _path("frame_row_offsets"): np.zeros((4,), dtype=np.int64),
    }

    assert (
        CANONICAL_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
        == ()
    )


@pytest.mark.parametrize("name", ("frame_counts", "n_detections"))
@pytest.mark.parametrize("nested", (False, True))
def test_count_vectors_are_forbidden_canonical_bindings(
    name: str,
    nested: bool,
) -> None:
    dimensions, arrays = _valid_payload()
    path = _path(name) if nested else name
    arrays[path] = np.asarray([2, 0, 1, 3], dtype=np.int32)

    assert "forbidden_count_binding" in _issue_codes(arrays, dimensions)


def test_exact_dtype_and_shape_contracts_reject_alternatives() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("bbox_norm_coords")] = arrays[_path("bbox_norm_coords")].astype(
        np.float64
    )
    arrays[_path("source_acquisition_frame_index")] = arrays[
        _path("source_acquisition_frame_index")
    ].astype(np.int32)
    arrays[_path("frame_row_offsets")] = arrays[_path("frame_row_offsets")][:-1]

    issues = CANONICAL_DETECTION_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
    )
    violations = {
        issue.path: issue.message
        for issue in issues
        if issue.code == "array_contract_violation"
    }

    assert "expected float32" in violations[_path("bbox_norm_coords")]
    assert "expected int64" in violations[_path("source_acquisition_frame_index")]
    assert "n_frame_boundaries=5" in violations[_path("frame_row_offsets")]


def test_frame_order_and_offsets_are_independent_required_invariants() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("frame_indices")] = np.asarray(
        [0, 2, 0, 3, 3, 3],
        dtype=np.int32,
    )
    arrays[_path("source_acquisition_frame_index")] = arrays[
        _path("frame_indices")
    ].astype(np.int64)
    arrays[_path("frame_row_offsets")] = np.asarray(
        [0, 1, 2, 3, 6],
        dtype=np.int64,
    )

    codes = _issue_codes(arrays, dimensions)

    assert "frame_indices_not_sorted" in codes
    assert "frame_row_offsets_mismatch" in codes


def test_source_frame_identity_and_instance_key_uniqueness_are_required() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("source_acquisition_frame_index")][2] = np.int64(1)
    arrays[_path("instance_key")][2] = arrays[_path("instance_key")][1]

    codes = _issue_codes(arrays, dimensions)

    assert "source_frame_identity_mismatch" in codes
    assert "duplicate_instance_key" in codes


def test_geometry_scores_and_class_ids_reject_sentinels_or_drift() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("bbox_img_xyxy")][0, 0] += np.float32(1.0)
    arrays[_path("centers_img_xy")][1, 1] += np.float32(1.0)
    arrays[_path("scores")][2] = np.float32(np.nan)
    arrays[_path("class_ids")][3] = np.int32(-1)

    codes = _issue_codes(arrays, dimensions)

    assert "bbox_img_projection_mismatch" in codes
    assert "center_projection_mismatch" in codes
    assert "invalid_score" in codes
    assert "invalid_class_id" in codes


@pytest.mark.parametrize(
    "invalid_box",
    (
        [0.50, 0.50, 0.00, 0.20],
        [0.05, 0.50, 0.20, 0.20],
        [0.50, 0.50, np.nan, 0.20],
    ),
)
def test_normalized_geometry_must_be_finite_positive_and_contained(
    invalid_box: list[float],
) -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("bbox_norm_coords")][0] = np.asarray(
        invalid_box,
        dtype=np.float32,
    )

    assert "invalid_bbox_norm_coords" in _issue_codes(arrays, dimensions)


def test_offsets_require_zero_start_monotonicity_and_final_cardinality() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("frame_row_offsets")] = np.asarray(
        [1, 2, 1, 3, 5],
        dtype=np.int64,
    )

    codes = _issue_codes(arrays, dimensions)

    assert "offset_start_mismatch" in codes
    assert "offsets_not_monotonic" in codes
    assert "offset_end_mismatch" in codes
    assert "frame_row_offsets_mismatch" in codes


def test_class_id_upper_bound_matches_current_crimson_public_type() -> None:
    dimensions, arrays = _valid_payload()
    arrays[_path("class_ids")][0] = np.int32(MAX_CANONICAL_CLASS_ID)
    assert (
        CANONICAL_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
        == ()
    )

    arrays[_path("class_ids")][0] = np.int32(MAX_CANONICAL_CLASS_ID + 1)
    assert "invalid_class_id" in _issue_codes(arrays, dimensions)


def test_missing_array_produces_structured_error_and_require_raises() -> None:
    dimensions, arrays = _valid_payload()
    del arrays[_path("scores")]

    issues = CANONICAL_DETECTION_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
    )
    assert any(
        issue.code == "missing_required_array" and issue.path == _path("scores")
        for issue in issues
    )
    with pytest.raises(CanonicalDetectionSchemaError, match="missing_required_array"):
        CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)


def test_manifest_is_json_safe_exact_and_count_free() -> None:
    dimensions, _arrays = _valid_payload()

    manifest = CANONICAL_DETECTION_SCHEMA_V1.as_manifest(dimensions=dimensions)
    round_tripped = json.loads(json.dumps(manifest))

    assert round_tripped == manifest
    assert manifest["schema_id"] == "palette.stage.canonical_detection"
    assert manifest["schema_version"] == 1
    assert manifest["layout"] == "sparse_instances_with_frame_row_offsets_v1"
    assert manifest["dimensions"] == {
        "n_frames": 4,
        "n_instances": 6,
        "n_frame_boundaries": 5,
        "source_width": 640,
        "source_height": 480,
    }
    binding_paths = {binding["path"] for binding in manifest["bindings"]}
    assert binding_paths == {
        _path("frame_indices"),
        _path("source_acquisition_frame_index"),
        _path("instance_key"),
        _path("bbox_norm_coords"),
        _path("bbox_img_xyxy"),
        _path("centers_img_xy"),
        _path("scores"),
        _path("class_ids"),
        _path("frame_row_offsets"),
    }
    assert _path("frame_counts") not in binding_paths
    assert _path("n_detections") not in binding_paths
    assert manifest["invariants"]["nullability"] == "forbidden"
    assert (
        manifest["invariants"]["physical_fill_semantics"]
        == "initialization_only_not_missing"
    )
    assert (
        manifest["invariants"]["source_frame_authority"]
        == "source_acquisition_frame_index"
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    (
        (
            {"n_frames": -1, "n_instances": 0, "source_width": 1, "source_height": 1},
            ValueError,
            "n_frames cannot be negative",
        ),
        (
            {"n_frames": 1, "n_instances": 0, "source_width": 0, "source_height": 1},
            ValueError,
            "width and height must be positive",
        ),
        (
            {"n_frames": True, "n_instances": 0, "source_width": 1, "source_height": 1},
            TypeError,
            "n_frames must be an exact integer",
        ),
    ),
)
def test_dimensions_fail_closed(
    kwargs: dict[str, int],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        CanonicalDetectionDimensions(**kwargs)
