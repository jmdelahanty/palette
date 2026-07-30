from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1,
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    SubjectMaskSchemaError,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)


def _fixture() -> tuple[
    SubjectMaskDimensions,
    SubjectMaskComponentRegistry,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    dimensions = SubjectMaskDimensions(
        n_frames=4,
        n_rois=6,
        n_channels=2,
        roi_height=4,
        roi_width=5,
    )
    components = SubjectMaskComponentRegistry(("body", "left_eye"))
    frames = np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int64)
    instance_key = np.asarray([10, 11, 20, 30, 31, 32], dtype=np.uint64)
    placement = np.asarray(
        [
            [0, 0, 5, 4],
            [10, 20, 5, 4],
            [20, 30, 5, 4],
            [30, 40, 5, 4],
            [40, 50, 5, 4],
            [50, 60, 5, 4],
        ],
        dtype=np.float32,
    )
    source_crop = {
        "instance_key": instance_key.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "source_crop_xywh": placement.copy(),
    }
    probabilities = np.zeros((6, 2, 4, 5), dtype=np.uint8)
    probabilities[0, 0, 1, 1:3] = 255
    probabilities[0, 1, 0, 0] = 192
    probabilities[2, 0, 2:4, 2:5] = 128
    probabilities[3, 1, :, 4] = 255
    probabilities[5, 0, 1:3, 0:2] = 200
    dense = (probabilities.astype(np.float32) / np.float32(255.0) >= 0.5).astype(
        np.uint8
    )
    metrics = derive_subject_mask_metrics(dense)
    arrays = {
        "source_crop_row_ids": np.arange(6, dtype=np.int64),
        "instance_key": instance_key.copy(),
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(
            frames,
            n_frames=4,
        ),
        "source_crop_xywh": placement.copy(),
        "mask_probs_roi": probabilities,
        "masks_roi": dense,
        "available_channels": np.ones(2, dtype=bool),
        "metrics/prob_max": np.max(
            probabilities.astype(np.float32) / np.float32(255.0),
            axis=(2, 3),
        ).astype(np.float32),
        **{f"metrics/{name}": values for name, values in metrics.items()},
    }
    return dimensions, components, arrays, source_crop


def _refined_arrays(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = {
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "source_crop_xywh",
        "masks_roi",
        "available_channels",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_xy",
        "metrics/centroid_valid",
        "metrics/bbox_xyxy",
        "metrics/bbox_valid",
    }
    return {name: values.copy() for name, values in raw.items() if name in keep}


def test_raw_uint8_schema_accepts_empty_and_multi_observation_frames() -> None:
    dimensions, components, arrays, source_crop = _fixture()

    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=source_crop,
    )

    assert arrays["frame_row_offsets"].tolist() == [0, 2, 2, 3, 6]
    manifest = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.as_manifest(
        dimensions=dimensions,
        components=components,
        threshold=0.5,
    )
    assert manifest["probability_encoding"] == "linear_uint8_0_255"
    assert manifest["invariants"]["instances_per_frame"] == "zero_one_or_many"
    assert (
        next(
            binding
            for binding in manifest["bindings"]
            if binding["path"] == "source_crop_xywh"
        )["contract_id"]
        == "palette.array.crop.source_crop_xywh"
    )


def test_raw_float16_is_an_exact_separate_schema() -> None:
    dimensions, components, arrays, source_crop = _fixture()
    probabilities = arrays["mask_probs_roi"].astype(np.float32) / np.float32(255.0)
    arrays["mask_probs_roi"] = probabilities.astype(np.float16)
    arrays["metrics/prob_max"] = np.max(
        arrays["mask_probs_roi"].astype(np.float32),
        axis=(2, 3),
    ).astype(np.float32)

    RAW_SUBJECT_MASK_FLOAT16_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=source_crop,
    )
    issues = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=source_crop,
    )
    assert any(
        issue.code == "array_contract_violation" and issue.path == "mask_probs_roi"
        for issue in issues
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        (
            lambda arrays: arrays["instance_key"].__setitem__(
                1, arrays["instance_key"][0]
            ),
            "duplicate_instance_key",
        ),
        (
            lambda arrays: arrays["frame_row_offsets"].__setitem__(2, 1),
            "frame_row_offsets_mismatch",
        ),
        (
            lambda arrays: arrays["metrics/area_px"].__setitem__((0, 0), 99),
            "derived_metric_mismatch",
        ),
        (
            lambda arrays: arrays["masks_roi"].__setitem__((0, 0, 0, 0), 1),
            "threshold_cache_mismatch",
        ),
    ),
)
def test_raw_schema_rejects_identity_and_payload_tampering(
    mutate: object,
    expected_code: str,
) -> None:
    dimensions, components, arrays, source_crop = _fixture()
    mutate(arrays)  # type: ignore[operator]

    issues = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=source_crop,
    )

    assert expected_code in {issue.code for issue in issues}


def test_raw_schema_requires_crop_v2_float32_placement() -> None:
    dimensions, components, arrays, source_crop = _fixture()
    arrays["source_crop_xywh"] = arrays["source_crop_xywh"].astype(np.float64)
    source_crop["source_crop_xywh"] = source_crop["source_crop_xywh"].astype(np.float64)

    issues = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=source_crop,
    )

    codes = {issue.code for issue in issues}
    assert "array_contract_violation" in codes
    assert "noncanonical_crop_placement_dtype" not in codes


def test_raw_schema_rejects_legacy_alias_and_missing_crop_evidence() -> None:
    dimensions, components, arrays, _source_crop = _fixture()
    arrays["frame_counts"] = np.zeros(4, dtype=np.int32)

    issues = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        components=components,
        threshold=0.5,
        source_crop_arrays=None,
    )

    codes = {issue.code for issue in issues}
    assert "forbidden_legacy_array" in codes
    assert "missing_source_crop_evidence" in codes


def test_refined_core_requires_binary_dense_authority_and_exact_metrics() -> None:
    dimensions, components, raw, source_crop = _fixture()
    arrays = _refined_arrays(raw)

    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
        source_crop_arrays=source_crop,
    )

    bad = copy.deepcopy(arrays)
    bad["masks_roi"][0, 0, 0, 0] = np.uint8(2)
    with pytest.raises(SubjectMaskSchemaError, match="invalid_dense_authority"):
        REFINED_SUBJECT_MASK_CORE_SCHEMA_V1.require(
            bad,
            dimensions=dimensions,
            components=components,
            source_crop_arrays=source_crop,
        )


def test_dimensions_reject_zero_row_canonical_snapshot() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        SubjectMaskDimensions(
            n_frames=4,
            n_rois=0,
            n_channels=2,
            roi_height=4,
            roi_width=5,
        )
