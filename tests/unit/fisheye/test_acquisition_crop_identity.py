from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.acquisition_crop_identity import (
    ACQUISITION_CROP_INSTANCE_KEY_POLICY,
    build_acquisition_crop_identity,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "bbox_norm_coords": np.asarray(
            [
                [0.25, 0.25, 0.10, 0.20],
                [0.75, 0.25, 0.10, 0.20],
                [0.50, 0.75, 0.20, 0.10],
            ],
            dtype=np.float32,
        ),
        "crop_state_codes": np.zeros(3, dtype=np.int8),
        "frame_indices": np.asarray([0, 0, 2], dtype=np.int64),
        "roi_coordinates_full": np.asarray(
            [[10, 20], [30, 40], [50, 60]], dtype=np.int32
        ),
        "roi_sizes_full": np.full((3, 2), 384, dtype=np.int32),
        "source_crop_local_frame_ids": np.asarray([100, 101, 102], dtype=np.int64),
        "source_crop_meta_row_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "source_crop_video_frame_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "source_pixel_kind_codes": np.zeros(3, dtype=np.int8),
    }


def _build(arrays: dict[str, np.ndarray] | None = None):
    return build_acquisition_crop_identity(
        arrays or _arrays(),
        recording_identity="recording-a",
        source_video_descriptor={
            "path": "/recording/crop.mp4",
            "size_bytes": 1234,
            "mtime_ns": 5678,
        },
        source_crop_meta_path="/recording/crop.csv",
        source_width=4512,
        source_height=4512,
    )


def test_acquisition_crop_identity_is_stable_unique_and_signed() -> None:
    first = _build()
    second = _build()

    np.testing.assert_array_equal(first.instance_keys, second.instance_keys)
    np.testing.assert_array_equal(
        first.row_signatures.signatures,
        second.row_signatures.signatures,
    )
    assert np.unique(first.instance_keys).shape == (3,)
    assert first.row_signatures.signatures.shape == (3, 32)
    assert first.crop_signature == second.crop_signature
    attrs = first.attrs()
    assert attrs["instance_key_policy"] == ACQUISITION_CROP_INSTANCE_KEY_POLICY
    assert attrs["crop_revision"] == 1
    assert attrs["source_row_signature_spec_digest"] == (
        first.row_signatures.spec.spec_digest
    )


def test_acquisition_crop_signature_changes_only_edited_row_for_row_content() -> None:
    baseline = _build()
    changed_arrays = _arrays()
    changed_arrays["source_crop_video_frame_indices"] = np.asarray(
        [0, 99, 2], dtype=np.int64
    )
    changed = _build(changed_arrays)

    np.testing.assert_array_equal(baseline.instance_keys, changed.instance_keys)
    np.testing.assert_array_equal(
        baseline.row_signatures.signatures[[0, 2]],
        changed.row_signatures.signatures[[0, 2]],
    )
    assert not np.array_equal(
        baseline.row_signatures.signatures[1],
        changed.row_signatures.signatures[1],
    )


def test_acquisition_crop_identity_rejects_ambiguous_pixel_contract() -> None:
    with pytest.raises(ValueError, match="exact Orange PyNvVC"):
        build_acquisition_crop_identity(
            _arrays(),
            recording_identity="recording-a",
            source_video_descriptor={"path": "/recording/crop.mp4"},
            source_crop_meta_path="/recording/crop.csv",
            source_width=4512,
            source_height=4512,
            pixel_contract={"schema": "wrong", "name": "ambiguous"},
        )
