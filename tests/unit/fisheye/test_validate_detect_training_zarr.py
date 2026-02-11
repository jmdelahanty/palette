"""Tests for merged detection-training Zarr validation."""

import warnings
from pathlib import Path
import sys

import numpy as np
import pytest
import zarr
from zarr.errors import UnstableSpecificationWarning

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.export_detect_training_zarr import validate_merged_training_zarr


def _write_valid_merged_zarr(path: Path, *, suppress_legacy_string_warning: bool = True) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["training_export"] = {"input_format": "gray"}

    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((4, 8, 8), dtype=np.uint8),
        chunks=(2, 8, 8),
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_smoke"
    crop = crop_parent.create_group("merged_export_smoke")
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )

    splits = root.create_group("splits")
    splits.create_array("train_indices", data=np.array([0, 1], dtype=np.int64), chunks=(2,))
    splits.create_array("val_indices", data=np.array([2], dtype=np.int64), chunks=(1,))
    splits.create_array("test_indices", data=np.array([3], dtype=np.int64), chunks=(1,))

    source = root.create_group("source_index")
    source.create_array(
        "source_dataset_idx",
        data=np.array([0, 0, 1, 1], dtype=np.int32),
        chunks=(4,),
    )
    source.create_array(
        "source_frame_idx",
        data=np.array([10, 11, 20, 21], dtype=np.int64),
        chunks=(4,),
    )
    if suppress_legacy_string_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            source.create_array(
                "source_dataset_id",
                data=np.array(["dataset_a", "dataset_b"], dtype="<U16"),
                chunks=(2,),
            )
            source.create_array(
                "source_zarr_path",
                data=np.array(["/a.zarr", "/b.zarr"], dtype="<U16"),
                chunks=(2,),
            )
    else:
        source.create_array(
            "source_dataset_id",
            data=np.array(["dataset_a", "dataset_b"], dtype="<U16"),
            chunks=(2,),
        )
        source.create_array(
            "source_zarr_path",
            data=np.array(["/a.zarr", "/b.zarr"], dtype="<U16"),
            chunks=(2,),
        )


def test_validate_merged_training_zarr_passes(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_ok.zarr"
    _write_valid_merged_zarr(zarr_path)

    summary = validate_merged_training_zarr(
        zarr_path,
        expected_input_format="gray",
        expected_total_samples=4,
    )

    assert summary["total_samples"] == 4
    assert summary["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert summary["source_count"] == 2


def test_validate_merged_training_zarr_rejects_invalid_frame_indices(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_bad.zarr"
    _write_valid_merged_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    latest = root["crop_runs"].attrs["latest"]
    root[f"crop_runs/{latest}/frame_indices"][:] = np.array([1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="frame_indices"):
        validate_merged_training_zarr(
            zarr_path,
            expected_input_format="gray",
            expected_total_samples=4,
        )


def test_validate_legacy_fixed_unicode_source_index_without_backfill(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_legacy_strings_ok.zarr"

    # Legacy fixed-width strings are still accepted by validators.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UnstableSpecificationWarning)
        _write_valid_merged_zarr(zarr_path, suppress_legacy_string_warning=False)
    assert any("FixedLengthUTF32" in str(item.message) for item in caught)

    summary = validate_merged_training_zarr(
        zarr_path,
        expected_input_format="gray",
        expected_total_samples=4,
    )
    assert summary["source_count"] == 2
