from __future__ import annotations

import numpy as np
import zarr

from fisheye.utils.patch_crops_from_refined import _patch_crop_run


def test_patch_crop_run_bumps_revision_and_signature(tmp_path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 6
    root.attrs["height"] = 6

    raw_video = root.create_group("raw_video")
    frame = np.arange(36, dtype=np.uint8).reshape(1, 6, 6)
    raw_video.create_array("images_full", data=frame, overwrite=True)

    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["roi_size"] = [4, 4]
    crop.attrs["detection_source_path"] = "detect_runs/detect_old"
    crop.attrs["detection_source_type"] = "detect"
    crop.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8), overwrite=True)
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )

    detect = root.create_group("manual_detect")
    detect.create_array("frame_indices", data=np.array([0], dtype=np.int64), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.25, 0.25, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )

    result = _patch_crop_run(
        root,
        crop,
        detect,
        [0],
        apply=True,
        patch_context={"reason": "manual_bbox_move"},
        detection_source_path="refined_detect_runs/refined_001/manual",
        detection_source_type="manual",
        source_refined_run="refined_001",
    )

    assert result["patched"] == 1
    assert crop.attrs["crop_revision"] == 1
    assert crop.attrs["crop_revision_reason"] == "manual_bbox_patch"
    assert crop.attrs["detection_source_path"] == "refined_detect_runs/refined_001/manual"
    assert crop.attrs["detection_source_type"] == "manual"
    assert crop.attrs["source_refined_run"] == "refined_001"

    signature = crop.attrs["crop_signature"]
    assert signature["signature_version"] == 2
    assert signature["crop_revision"] == 1
    assert signature["detection_source_path"] == "refined_detect_runs/refined_001/manual"
    assert signature["detection_source_type"] == "manual"
