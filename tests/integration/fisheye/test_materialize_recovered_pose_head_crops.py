"""Real-Zarr publication checks for the recovered-source crop version."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.training.materialize_recovered_pose_head_crops import (
    RUN_ID,
    materialize_recovered_pose_head_crop,
    validate_published_recovered_crop,
)
from fisheye.training.recover_merged_training_recording import (
    SOURCE_ONLY_SCHEMA_ID,
    _sha256_array,
)


def _recovered_archive(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3, use_consolidated=False)
    pose_pixels = np.broadcast_to(
        np.arange(512, dtype=np.uint8)[None, None, :], (2, 512, 512)
    ).copy()
    pose_points = np.array(
        [
            [[200, 200], [210, 195], [190, 195]],
            [[200, 370], [210, 195], [190, 195]],
        ],
        dtype=np.float32,
    )
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]], dtype=np.float32)
    arrays = {
        "recovered_sources/pose/roi_images": pose_pixels,
        "recovered_sources/pose/keypoints_roi": pose_points,
        "recovered_sources/pose/source_merged_row": np.array(
            [100, 101], dtype=np.int64
        ),
        "recovered_sources/pose/source_frame_idx": np.array([4, 8], dtype=np.int64),
        "recovered_sources/pose/detect_local_row": np.array([0, 1], dtype=np.int32),
        "recovered_sources/pose/crop_bbox_norm_coords": boxes,
        "recovered_sources/detect/images_ds": np.zeros((2, 640, 640), dtype=np.uint8),
        "recovered_sources/detect/bbox_norm_coords": boxes,
        "recovered_sources/detect/source_merged_row": np.array(
            [200, 201], dtype=np.int64
        ),
        "recovered_sources/detect/source_frame_idx": np.array([4, 8], dtype=np.int64),
        "recovered_sources/detect/detect_only_local_row": np.empty(0, dtype=np.int32),
    }
    for name, values in arrays.items():
        group_name, array_name = name.rsplit("/", 1)
        root.require_group(group_name).create_array(
            array_name,
            data=values,
            chunks=(max(1, min(2, len(values))), *values.shape[1:]),
        )
    root.attrs.update(
        {
            "schema_id": SOURCE_ONLY_SCHEMA_ID,
            "zarr_purpose": "training",
            "training_artifact_status": "complete",
            "stage_selector_eligible": False,
            "recording_id": "rec_a",
            "recovery_mode": "source_only",
            "source_pose": {"run_id": "pose_run", "dataset_id": "rec_a:pose"},
            "source_detect": {"run_id": "detect_run", "dataset_id": "rec_a"},
            "review_snapshot": {
                "species": "Danio rerio",
                "pose_review_state": "approved",
                "pose_review_intended_use": "training",
                "pose_review_method": "manual",
            },
            "pose_source_row_count": 2,
            "detect_source_row_count": 2,
            "detect_only_row_count": 0,
            "array_sha256": {
                name: _sha256_array(values) for name, values in arrays.items()
            },
        }
    )
    zarr.consolidate_metadata(str(path))
    return path


def test_materialized_crop_preserves_rows_and_marks_outside_point(
    tmp_path: Path,
) -> None:
    archive = _recovered_archive(tmp_path / "rec_a_recovered_training.zarr")
    planned = materialize_recovered_pose_head_crop(archive=archive)
    assert planned["status"] == "planned"
    assert planned["row_count"] == 2
    assert planned["invisible_point_count"] == 1
    assert not (archive / "crop_runs").exists()

    result = materialize_recovered_pose_head_crop(
        archive=archive, scratch_root=tmp_path, apply=True
    )
    assert result["status"] == "materialized"
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    run = root[f"crop_runs/{RUN_ID}"]
    assert tuple(run["roi_images"].shape) == (2, 192, 192)
    np.testing.assert_array_equal(
        np.asarray(run["keypoint_visibility"][:]),
        np.array([[2, 2, 2], [0, 2, 2]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["roi_origin_xy_in_pose_512"][:]),
        np.array([[160, 160], [160, 160]], dtype=np.int32),
    )
    assert run.attrs["sensor_pixel_origin_available"] is False
    assert not any(
        root["crop_runs"].attrs.get(name) == RUN_ID
        for name in ("latest", "latest_complete", "authoritative_run")
    )
    assert validate_published_recovered_crop(archive)["row_count"] == 2
    assert (
        materialize_recovered_pose_head_crop(archive=archive, resume=True)["status"]
        == "already_materialized"
    )


def test_materialization_refuses_tampered_source(tmp_path: Path) -> None:
    archive = _recovered_archive(tmp_path / "rec_a_recovered_training.zarr")
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root["recovered_sources/pose/keypoints_roi"][0, 0, 0] = 100
    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_recovered_pose_head_crop(archive=archive)
    assert not (archive / "crop_runs").exists()
