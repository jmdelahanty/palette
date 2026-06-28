from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.repair_acquisition_crop_bbox_contract import repair_acquisition_crop_bbox_contract


def _write_affected_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((2, 240, 200), dtype=np.uint8), chunks=(2, 240, 200))
    parent = root.create_group("crop_runs")
    crop = parent.create_group("crop_acq")
    crop.attrs["source_type"] = "acquisition_crop_video"
    crop.attrs["source_pixels"] = "acquisition_crop_video"
    crop.attrs["roi_size"] = [20, 20]
    crop.attrs["bbox_norm_coords_semantics"] = "realtime_detection_bbox_xywh_normalized_to_crop_video_frame"
    crop.create_array("roi_images", data=np.zeros((2, 20, 20), dtype=np.uint8), chunks=(2, 20, 20))
    crop.create_array("source_crop_xywh", data=np.asarray([[100, 200, 20, 20], [110, 210, 20, 20]], dtype=np.float32), chunks=(2, 4))
    crop.create_array("roi_coordinates_full", data=np.asarray([[100, 200], [110, 210]], dtype=np.int32), chunks=(2, 2))
    crop.create_array("bbox_roi_xyxy", data=np.asarray([[4, 4, 12, 12], [6, 8, 10, 14]], dtype=np.float32), chunks=(2, 4))
    crop.create_array("bbox_norm_coords", data=np.asarray([[0.4, 0.4, 0.4, 0.4], [0.4, 0.55, 0.2, 0.3]], dtype=np.float32), chunks=(2, 4))


def test_repair_acquisition_crop_bbox_contract_dry_run_and_apply(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _write_affected_zarr(zarr_path)

    dry = repair_acquisition_crop_bbox_contract(zarr_path, apply=False)
    assert dry["status"] == "ok"
    assert dry["affected_crop_run_count"] == 1
    assert dry["changed_crop_run_count"] == 1
    assert dry["crop_runs"][0]["status"] == "would_update"

    applied = repair_acquisition_crop_bbox_contract(zarr_path, apply=True)
    assert applied["status"] == "ok"
    assert applied["changed_crop_run_count"] == 1

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    crop = root["crop_runs"]["crop_acq"]
    np.testing.assert_allclose(crop["bbox_crop_norm_coords"][:], [[0.4, 0.4, 0.4, 0.4], [0.4, 0.55, 0.2, 0.3]])
    np.testing.assert_allclose(crop["bbox_img_xyxy"][:], [[104, 204, 112, 212], [116, 218, 120, 224]])
    np.testing.assert_allclose(
        crop["bbox_norm_coords"][:],
        [[0.54, 208.0 / 240.0, 0.04, 8.0 / 240.0], [0.59, 221.0 / 240.0, 0.02, 6.0 / 240.0]],
    )
    assert crop.attrs["bbox_norm_coords_semantics"] == "bbox_xywh_normalized_to_full_frame"
    assert crop.attrs["bbox_crop_norm_coords_semantics"] == "bbox_xywh_normalized_to_crop_roi_frame"

    second = repair_acquisition_crop_bbox_contract(zarr_path, apply=False)
    assert second["crop_runs"][0]["status"] == "skipped"
