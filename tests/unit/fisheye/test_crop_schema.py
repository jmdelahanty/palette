from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    CropGeometrySchemaError,
    CropPaddingMode,
    CropSizeMode,
    derive_crop_placement_geometry,
    derive_frame_row_offsets,
)
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)


def _dimensions() -> CropDimensions:
    return CropDimensions(
        n_frames=4,
        n_instances=6,
        source_width=640,
        source_height=480,
    )


def _policy(
    *,
    padding: CropPaddingMode = CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
) -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(64, 48),
        padding_mode=padding,
    )


def _valid_arrays(
    *,
    policy: CropGeometryPolicy | None = None,
) -> dict[str, np.ndarray]:
    policy = policy or _policy()
    frames = np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int64)
    bbox_norm = np.asarray(
        [
            [0.02, 0.03, 0.02, 0.02],
            [0.25, 0.25, 0.08, 0.06],
            [0.50, 0.50, 0.10, 0.08],
            [0.65, 0.70, 0.08, 0.06],
            [0.80, 0.80, 0.08, 0.06],
            [0.95, 0.95, 0.04, 0.04],
        ],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=640,
        source_height=480,
    )
    if policy.size_mode is CropSizeMode.FIXED_PER_RUN:
        sizes = np.repeat(
            np.asarray(policy.fixed_size_wh, dtype=np.int32).reshape(1, 2),
            frames.shape[0],
            axis=0,
        )
    else:
        sizes = np.asarray(
            [[64, 48], [80, 64], [96, 72], [48, 48], [72, 96], [40, 32]],
            dtype=np.int32,
        )
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    return {
        "instance_key": np.asarray([11, 12, 21, 31, 32, 33], dtype=np.uint64),
        "source_refined_row_ids": np.asarray(
            [101, 102, 201, 301, 302, 303], dtype=np.int64
        ),
        "frame_indices": frames,
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "bbox_norm_coords": bbox_norm,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "roi_coordinates_full": coordinates,
        "roi_sizes_full": sizes,
        "source_crop_xywh": source_crop,
        "bbox_roi_xyxy": bbox_roi,
        "source_row_signature": np.arange(6 * 32, dtype=np.uint8).reshape(6, 32),
    }


def _codes(issues: object) -> set[str]:
    return {issue.code for issue in issues}


def test_exact_geometry_only_bindings_exclude_pixels_and_count_aliases() -> None:
    assert CROP_GEOMETRY_SCHEMA_V1.binding_paths == (
        "instance_key",
        "source_refined_row_ids",
        "frame_indices",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
        "roi_coordinates_full",
        "roi_sizes_full",
        "source_crop_xywh",
        "bbox_roi_xyxy",
        "source_row_signature",
    )

    arrays = _valid_arrays()
    arrays["roi_images"] = np.zeros((6, 48, 64), dtype=np.uint8)
    arrays["frame_counts"] = np.asarray([2, 0, 1, 3], dtype=np.int32)
    issues = CROP_GEOMETRY_SCHEMA_V1.validate(
        arrays,
        dimensions=_dimensions(),
        policy=_policy(),
    )
    assert _codes(issues) == {"forbidden_pixel_or_compatibility_array"}


def test_multi_instance_empty_frame_fixture_and_exact_offsets_pass() -> None:
    arrays = _valid_arrays()

    CROP_GEOMETRY_SCHEMA_V1.require(
        arrays,
        dimensions=_dimensions(),
        policy=_policy(),
    )
    np.testing.assert_array_equal(
        arrays["frame_row_offsets"],
        [0, 2, 2, 3, 6],
    )
    assert arrays["instance_key"][0:2].tolist() == [11, 12]
    assert arrays["instance_key"][2:2].tolist() == []
    assert arrays["instance_key"][3:6].tolist() == [31, 32, 33]


def test_all_empty_frames_are_a_valid_positive_duration_snapshot() -> None:
    arrays = _valid_arrays()
    empty = {
        path: (np.zeros(5, dtype=np.int64) if path == "frame_row_offsets" else value[:0])
        for path, value in arrays.items()
    }

    CROP_GEOMETRY_SCHEMA_V1.require(
        empty,
        dimensions=CropDimensions(
            n_frames=4,
            n_instances=0,
            source_width=640,
            source_height=480,
        ),
        policy=_policy(),
    )


def test_variable_per_row_sizes_are_data_and_change_geometry() -> None:
    policy = CropGeometryPolicy(
        purpose="inspection",
        size_mode=CropSizeMode.VARIABLE_PER_ROW,
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    arrays = _valid_arrays(policy=policy)

    CROP_GEOMETRY_SCHEMA_V1.require(
        arrays,
        dimensions=_dimensions(),
        policy=policy,
    )
    assert np.unique(arrays["roi_sizes_full"], axis=0).shape[0] == 6
    np.testing.assert_array_equal(
        arrays["source_crop_xywh"][:, 2:].astype(np.int32),
        arrays["roi_sizes_full"],
    )


def test_policy_identity_changes_without_changing_detection_identity() -> None:
    first = _policy()
    second = CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(96, 80),
    )

    assert first.payload_digest != second.payload_digest
    assert first.payload["purpose"] == second.payload["purpose"]
    manifest = first.as_manifest()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["payload_digest_algorithm"] == "sha256_canonical_json_v1"
    assert "instance_key" not in manifest["payload"]


def test_contained_padding_mode_rejects_edge_crossing_crops() -> None:
    policy = _policy(padding=CropPaddingMode.REQUIRE_FULLY_CONTAINED)
    issues = CROP_GEOMETRY_SCHEMA_V1.validate(
        _valid_arrays(policy=policy),
        dimensions=_dimensions(),
        policy=policy,
    )

    assert "crop_not_fully_contained" in _codes(issues)


def test_schema_rejects_dtype_offsets_and_derived_geometry_drift() -> None:
    arrays = _valid_arrays()
    arrays["frame_indices"] = arrays["frame_indices"].astype(np.int32)
    arrays["frame_row_offsets"] = np.asarray([0, 1, 2, 3, 6], dtype=np.int64)
    arrays["bbox_roi_xyxy"] = arrays["bbox_roi_xyxy"].copy()
    arrays["bbox_roi_xyxy"][0, 0] += np.float32(1.0)

    issues = CROP_GEOMETRY_SCHEMA_V1.validate(
        arrays,
        dimensions=_dimensions(),
        policy=_policy(),
    )
    assert "array_contract_violation" in _codes(issues)
    assert "frame_row_offsets_mismatch" not in _codes(issues)
    assert "bbox_roi_projection_mismatch" in _codes(issues)

    arrays = _valid_arrays()
    arrays["frame_row_offsets"] = np.asarray([0, 1, 2, 3, 6], dtype=np.int64)
    with pytest.raises(CropGeometrySchemaError) as error:
        CROP_GEOMETRY_SCHEMA_V1.require(
            arrays,
            dimensions=_dimensions(),
            policy=_policy(),
        )
    assert "frame_row_offsets_mismatch" in _codes(error.value.issues)


def test_schema_manifest_freezes_size_padding_and_observation_identity_semantics() -> None:
    manifest = CROP_GEOMETRY_SCHEMA_V1.as_manifest(
        dimensions=_dimensions(),
        policy=_policy(),
    )

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["artifact_profile"] == "geometry_only_analysis"
    assert manifest["invariants"]["crop_size"] == "fixed_per_run"
    assert manifest["invariants"]["instances_per_frame"] == "zero_one_or_many"
    assert manifest["invariants"]["instance_key_semantics"] == (
        "observation_identity_not_subject_identity"
    )
    assert manifest["invariants"]["pixel_payload"] == "absent"


def test_policy_rejects_ambiguous_size_declarations() -> None:
    with pytest.raises(ValueError, match="fixed_size_wh"):
        CropGeometryPolicy(
            purpose="analysis",
            size_mode=CropSizeMode.FIXED_PER_RUN,
        )
    with pytest.raises(ValueError, match="cannot declare fixed_size_wh"):
        CropGeometryPolicy(
            purpose="analysis",
            size_mode=CropSizeMode.VARIABLE_PER_ROW,
            fixed_size_wh=(64, 64),
        )
