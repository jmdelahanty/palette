from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.build_analysis_acquisition_crop_run import build_analysis_acquisition_crop_run


def _write_crop_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,crop_video_frame_index,has_detection,blank_frame,"
                "crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,101,0,1,0,100,200,384,384,150,260,20,40",
                "2,102,1,0,1,0,0,0,0,0,0,0,0",
                "3,103,2,0,0,110,210,384,384,0,0,0,0",
                "4,104,3,1,0,120,220,384,384,140,250,30,50",
                "5,105,4,1,0,nan,220,384,384,140,250,30,50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_zarr(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "source_video_width": 4512,
            "source_video_height": 4512,
            "total_frames": 8,
        }
    )
    return zarr_path


def test_build_analysis_acquisition_crop_run_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    crop_meta = tmp_path / "recording" / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = crop_meta.with_name("Cam2010093_crop_external.mp4")
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.touch()
    _write_crop_meta(crop_meta)

    result = build_analysis_acquisition_crop_run(
        zarr_path,
        recording_dir=tmp_path / "recording",
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        run_name="crop_test",
        apply=False,
    )

    assert result.applied is False
    assert result.selected_rows == 2
    assert result.rejected_blank_crop_frame == 1
    assert result.rejected_crop_has_no_detection == 1
    assert result.rejected_nonfinite_crop_geometry == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "crop_runs" not in root


def test_build_analysis_acquisition_crop_run_writes_geometry_only_contract(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path)
    crop_meta = tmp_path / "recording" / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = crop_meta.with_name("Cam2010093_crop_external.mp4")
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.touch()
    _write_crop_meta(crop_meta)

    result = build_analysis_acquisition_crop_run(
        zarr_path,
        recording_dir=tmp_path / "recording",
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        run_name="crop_test",
        apply=True,
    )

    assert result.applied is True
    assert result.selected_rows == 2
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    crop_parent = root["crop_runs"]
    assert crop_parent.attrs["latest_any"] == "crop_test"
    assert crop_parent.attrs.get("latest") is None
    crop = crop_parent["crop_test"]
    assert crop.attrs["schema_id"] == "palette.analysis_acquisition_crop_run.v1"
    assert crop.attrs["crop_storage_mode"] == "geometry_only"
    assert crop.attrs["source_pixels"] == "acquisition_crop_video"
    assert crop.attrs["bbox_norm_coords_semantics"] == "bbox_xywh_normalized_to_full_frame"
    assert crop.attrs["selected_live_detection_bbox_semantics"] == "selected_postprocessed_model_detection_used_to_center_crop"
    assert "roi_images" not in crop

    assert crop["frame_indices"][:].tolist() == [0, 3]
    assert crop["source_recording_frame_ids"][:].tolist() == [1, 4]
    assert crop["source_crop_meta_row_indices"][:].tolist() == [0, 3]
    assert crop["source_crop_video_frame_indices"][:].tolist() == [0, 3]
    assert crop["source_crop_local_frame_ids"][:].tolist() == [101, 104]
    np.testing.assert_allclose(crop["source_crop_xywh"][:], [[100, 200, 384, 384], [120, 220, 384, 384]])
    np.testing.assert_array_equal(crop["roi_coordinates_full"][:], [[100, 200], [120, 220]])
    np.testing.assert_array_equal(crop["roi_sizes_full"][:], [[384, 384], [384, 384]])

    np.testing.assert_allclose(
        crop["selected_live_detection_bbox_img_xyxy"][:],
        [[150, 260, 170, 300], [140, 250, 170, 300]],
    )
    np.testing.assert_allclose(
        crop["selected_live_detection_bbox_roi_xyxy"][:],
        [[50, 60, 70, 100], [20, 30, 50, 80]],
    )
    np.testing.assert_allclose(crop["bbox_img_xyxy"][:], crop["selected_live_detection_bbox_img_xyxy"][:])
    np.testing.assert_allclose(crop["bbox_roi_xyxy"][:], crop["selected_live_detection_bbox_roi_xyxy"][:])

    expected_norm = np.asarray(
        [
            [160 / 4512, 280 / 4512, 20 / 4512, 40 / 4512],
            [155 / 4512, 275 / 4512, 30 / 4512, 50 / 4512],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(crop["bbox_norm_coords"][:], expected_norm, rtol=1e-6)
    expected_crop_norm = np.asarray(
        [
            [60 / 384, 80 / 384, 20 / 384, 40 / 384],
            [35 / 384, 55 / 384, 30 / 384, 50 / 384],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(crop["bbox_crop_norm_coords"][:], expected_crop_norm, rtol=1e-6)
    assert crop["frame_counts"][:].tolist() == [1, 0, 0, 1, 0, 0, 0, 0]
    assert crop["detection_indices"][:].tolist() == [0, 1]
    assert crop["detection_success"][:].tolist() == [True, True]
    assert crop["source_pixel_kind_codes"][:].tolist() == [0, 0]
    assert crop["crop_state_codes"][:].tolist() == [0, 0]
