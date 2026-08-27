from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_HEADING_VALIDATION_ATOL_DEG,
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
    BodyFrameSchemaError,
)
from fisheye.shared.zarr.keypoint_schema import derive_frame_row_offsets


def _fixture() -> tuple[
    BodyFrameDimensions,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions = BodyFrameDimensions(n_frames=4, n_instances=4)
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    keys = np.asarray([101, 102, 201, 301], dtype=np.uint64)
    signatures = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    source_keypoints = {
        "instance_key": keys.copy(),
        "frame_indices": frames.copy(),
        "keypoint_row_signature": signatures.copy(),
    }
    valid = np.asarray([True, False, True, True], dtype=bool)
    origin = np.asarray(
        [[25, 30], [np.nan, np.nan], [220, 130], [340, 250]],
        dtype=np.float32,
    )
    forward = np.asarray(
        [[1, 0], [np.nan, np.nan], [0, -1], [0.6, 0.8]],
        dtype=np.float32,
    )
    left = np.asarray(
        [[0, -1], [np.nan, np.nan], [-1, 0], [0.8, -0.6]],
        dtype=np.float32,
    )
    heading = np.rad2deg(np.arctan2(-forward[:, 1], forward[:, 0])).astype(np.float32)
    heading[~valid] = np.nan
    arrays = {
        "instance_key": keys,
        "source_keypoint_row_ids": np.arange(4, dtype=np.int64),
        "source_keypoint_row_signature": signatures,
        "frame_indices": frames,
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "origin_xy": origin,
        "forward_axis_xy": forward,
        "left_axis_xy": left,
        "axis_valid": valid,
        "heading_deg": heading,
    }
    return dimensions, arrays, source_keypoints


def test_body_frame_v1_accepts_exact_source_bound_geometry() -> None:
    dimensions, arrays, source_keypoints = _fixture()

    BODY_FRAME_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    manifest = BODY_FRAME_SCHEMA_V1.as_manifest(dimensions=dimensions)
    assert len(manifest["bindings"]) == 10
    assert manifest["invariants"]["coordinate_space"] == "source_camera_pixels"
    assert manifest["invariants"]["heading_derivation"] == "atan2_negative_y_x_degrees"


def test_body_frame_v1_rejects_heading_that_does_not_match_forward_axis() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["heading_deg"][0] = np.float32(45.0)

    issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    assert "heading_derivation_mismatch" in {issue.code for issue in issues}


def test_body_frame_v1_accepts_float32_heading_reproduction_noise() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["heading_deg"][3] += np.float32(
        BODY_FRAME_HEADING_VALIDATION_ATOL_DEG / 2
    )

    BODY_FRAME_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )


def test_body_frame_v1_rejects_heading_outside_reproduction_tolerance() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["heading_deg"][3] += np.float32(
        BODY_FRAME_HEADING_VALIDATION_ATOL_DEG * 2
    )

    issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    assert "heading_derivation_mismatch" in {issue.code for issue in issues}


def test_body_frame_v1_rejects_nonorthogonal_axes() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["left_axis_xy"][0] = np.asarray([1, 0], dtype=np.float32)

    issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    assert "invalid_body_axes" in {issue.code for issue in issues}


def test_body_frame_v1_rejects_tampered_source_binding() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["source_keypoint_row_signature"][1, 0] ^= np.uint8(1)

    issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    assert "source_keypoint_binding_mismatch" in {issue.code for issue in issues}


def test_body_frame_v1_requires_source_snapshot_evidence() -> None:
    dimensions, arrays, _ = _fixture()

    with pytest.raises(BodyFrameSchemaError, match="missing_source_keypoint_evidence"):
        BODY_FRAME_SCHEMA_V1.require(
            arrays,
            dimensions=dimensions,
            source_keypoint_arrays=None,
        )


def test_body_frame_v1_rejects_finite_geometry_on_invalid_row() -> None:
    dimensions, arrays, source_keypoints = _fixture()
    arrays["origin_xy"][1] = np.asarray([0, 0], dtype=np.float32)

    issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_keypoints,
    )

    codes = {issue.code for issue in issues}
    assert "invalid_axis_not_nan" in codes
