from __future__ import annotations

import json

import numpy as np
import zarr

from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR, RUN_STATUS_COMPLETE
from fisheye.utils.import_acquisition_detections_to_detect_run import import_acquisition_detections_to_detect_run


def _write_crop_meta(path) -> None:
    path.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,10,100,0,0,1,0,0.50,0,0,100,100,10,20,30,40",
                "2,11,101,0,0,1,1,0.60,0,0,100,100,20,30,30,40",
                "3,12,102,0,0,0,0,0.00,0,0,100,100,,,,",
                "4,13,103,0,0,1,0,0.90,5,6,100,100,30,40,20,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_import_acquisition_detections_writes_standard_detect_run(tmp_path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    crop_meta = crop_dir / "Cam2010093_session_crop_meta.csv"
    _write_crop_meta(crop_meta)
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "full": {"width": 1000, "height": 500},
                        "crop": {
                            "metadata": "derived/external_crop_recorder/Cam2010093_session_crop_meta.csv",
                            "video": "derived/external_crop_recorder/Cam2010093_session_crop_external.mp4",
                            "selection_policy": "largest_detection_by_confidence",
                            "blank_frame_policy": "encode_black_frame_when_no_detection",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["total_frames"] = 4

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="detect_acquisition_test",
        apply=True,
    )

    assert result.applied is True
    assert result.total_detections == 2
    assert result.blank_frame_count == 1
    assert result.no_detection_frame_count == 1
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["detect_runs"]
    run = parent["detect_acquisition_test"]
    assert parent.attrs["latest"] == "detect_acquisition_test"
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert run.attrs["detection_method"] == "acquisition_runtime_import"
    assert run.attrs["detection_source"] == "external_crop_recorder_crop_meta"
    assert run["frame_indices"][:].tolist() == [0, 3]
    assert run["scores"][:].tolist() == [np.float32(0.5), np.float32(0.9)]
    assert run["class_ids"][:].tolist() == [0, 0]
    assert run["frame_counts"][:].tolist() == [1, 0, 0, 1]
    assert run["n_detections"][:].tolist() == [1, 0, 0, 1]
    np.testing.assert_allclose(
        run["bbox_norm_coords"][:],
        np.asarray(
            [
                [0.025, 0.08, 0.03, 0.08],
                [0.04, 0.09, 0.02, 0.02],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        run["bbox_img_xyxy"][:],
        np.asarray([[10.0, 20.0, 40.0, 60.0], [30.0, 40.0, 50.0, 50.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        run["source_crop_xywh"][:],
        np.asarray([[0.0, 0.0, 100.0, 100.0], [5.0, 6.0, 100.0, 100.0]], dtype=np.float32),
    )
    assert run["source_recording_frame_ids"][:].tolist() == [1, 4]
    assert run["source_crop_meta_row_indices"][:].tolist() == [0, 3]


def test_import_acquisition_detections_dry_run_does_not_write(tmp_path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    _write_crop_meta(crop_dir / "Cam2010093_session_crop_meta.csv")
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["source_video_width"] = 1000
    root.attrs["source_video_height"] = 500
    root.attrs["total_frames"] = 4

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="detect_acquisition_test",
        apply=False,
    )

    assert result.applied is False
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in root
