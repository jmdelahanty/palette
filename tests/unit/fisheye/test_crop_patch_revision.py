from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.utils.patch_crops_from_refined import _build_patch_audit_entry, _patch_crop_run


def test_build_patch_audit_entry_groups_multi_instance_rows_by_frame() -> None:
    entry = _build_patch_audit_entry(
        timestamp_utc="2026-04-14T12:00:00+00:00",
        frame_indices=np.array([100, 100, 101, 102], dtype=np.int64),
        target_indices=np.array([0, 1, 3], dtype=np.int64),
        refined_row_ids=np.array([2000, 2001, 2002, 2003], dtype=np.int64),
        patch_context={"reason": "keypoint_detection_issue"},
    )

    assert entry["patched_detections"] == 3
    assert entry["patched_frames"] == 2
    assert entry["patched_detection_indices"] == [0, 1, 3]
    assert entry["patched_frame_indices"] == [100, 102]
    assert entry["patched_detection_indices_by_frame"] == {"100": [0, 1], "102": [3]}
    assert entry["patched_refined_row_ids"] == [2000, 2001, 2003]
    assert entry["patched_refined_row_ids_by_frame"] == {"100": [2000, 2001], "102": [2003]}
    assert entry["reason"] == "keypoint_detection_issue"


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
    history = crop.attrs["crop_patch_history"]
    assert isinstance(history, list)
    assert history[0]["patched_detection_indices"] == [0]
    assert history[0]["patched_frame_indices"] == [0]
    assert history[0]["patched_detection_indices_by_frame"] == {"0": [0]}

    signature = crop.attrs["crop_signature"]
    assert signature["signature_version"] == 2
    assert signature["crop_revision"] == 1
    assert signature["detection_source_path"] == "refined_detect_runs/refined_001/manual"
    assert signature["detection_source_type"] == "manual"


@pytest.mark.parametrize(
    "claim_location",
    ["run", "array"],
)
@pytest.mark.parametrize("apply", [False, True])
def test_patch_crop_run_refuses_canonical_in_place_mutation(
    tmp_path,
    claim_location: str,
    apply: bool,
) -> None:
    root = zarr.open_group(str(tmp_path / "canonical.zarr"), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.arange(36, dtype=np.uint8).reshape(1, 6, 6),
        overwrite=True,
    )
    crop = root.create_group("crop_runs").create_group("canonical")
    crop.attrs["roi_size"] = [4, 4]
    crop.create_array(
        "roi_images",
        data=np.zeros((1, 4, 4), dtype=np.uint8),
        overwrite=True,
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray([[0, 0]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )
    if claim_location == "run":
        crop.attrs["coordinate_contract"] = "canonical_v2"
    else:
        crop["bbox_norm_coords"].attrs["coordinate_descriptor"] = {
            "schema_id": "palette.coordinate_descriptor"
        }

    detect = root.create_group("manual_detect")
    detect.create_array(
        "frame_indices",
        data=np.asarray([0], dtype=np.int64),
        overwrite=True,
    )
    detect.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.25, 0.25, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )
    original_pixels = np.asarray(crop["roi_images"][:]).copy()
    original_box = np.asarray(crop["bbox_norm_coords"][:]).copy()

    with pytest.raises(RuntimeError, match="new derived crop run"):
        _patch_crop_run(root, crop, detect, [0], apply=apply)

    np.testing.assert_array_equal(crop["roi_images"][:], original_pixels)
    np.testing.assert_array_equal(crop["bbox_norm_coords"][:], original_box)


def test_patch_crop_run_prefers_refined_row_identity_over_stale_roi_index(tmp_path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    raw_video = root.create_group("raw_video")
    raw_video.create_array(
        "images_full",
        data=np.arange(64, dtype=np.uint8).reshape(1, 8, 8),
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["roi_size"] = [2, 2]
    crop.create_array("roi_images", data=np.zeros((2, 2, 2), dtype=np.uint8), overwrite=True)
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [0, 0]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.array([[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.array([0, 0], dtype=np.int32), overwrite=True)
    crop.create_array("detection_indices", data=np.array([1, 0], dtype=np.int32), overwrite=True)
    crop.create_array(
        "source_refined_row_ids",
        data=np.array([20, 10], dtype=np.int64),
        overwrite=True,
    )

    detect = root.create_group("manual_detect")
    detect.create_array("frame_indices", data=np.array([0, 0], dtype=np.int64), overwrite=True)
    detect.create_array("refined_row_ids", data=np.array([10, 20], dtype=np.int64), overwrite=True)
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], dtype=np.float32),
        overwrite=True,
    )

    result = _patch_crop_run(
        root,
        crop,
        detect,
        [],
        apply=True,
        flag_entries=[
            {
                "frame_idx": 0,
                "roi_idx": 1,
                "source_refined_row_id": 20,
            }
        ],
    )

    assert result["patched"] == 1
    np.testing.assert_allclose(crop["bbox_norm_coords"][0], np.array([0.75, 0.75, 0.2, 0.2]))
    np.testing.assert_allclose(crop["bbox_norm_coords"][1], np.array([0.2, 0.2, 0.2, 0.2]))
    history = crop.attrs["crop_patch_history"]
    assert history[0]["patched_detection_indices"] == [0]
    assert history[0]["patched_refined_row_ids"] == [20]
    assert history[0]["patched_source_detection_indices"] == [1]
