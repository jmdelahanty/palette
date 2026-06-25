from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from fisheye.utils import export_acquisition_crop_pose_training_zarr as mod


def _write_crop_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,5,100,0,0,1,0,0.95,100,200,20,20,104,204,8,8",
                "2,6,101,0,0,1,1,0.90,100,200,20,20,104,204,8,8",
                "3,7,102,0,0,0,0,0.00,100,200,20,20,,,,",
                "4,8,103,0,0,1,0,0.80,100,200,20,20,104,204,8,8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_source_zarr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    run = parent.create_group("refined_kp")
    mark_run_started(run, run_name="refined_kp", stage="refined_keypoints")
    run.attrs["source_crop_run"] = "crop_source"
    run.attrs["keypoint_labels"] = ["head", "tail"]
    run.attrs["skeleton_id"] = "test_pose"
    run.attrs["pose_schema"] = {
        "skeleton_id": "test_pose",
        "keypoint_labels": ["head", "tail"],
        "kpt_shape": [2, 2],
    }
    run.create_array("frame_indices", data=np.asarray([0, 1, 2, 3, 9], dtype=np.int64), chunks=(5,))
    run.create_array(
        "keypoints_img",
        data=np.asarray(
            [
                [[105.0, 205.0], [115.0, 215.0]],
                [[105.0, 205.0], [115.0, 215.0]],
                [[105.0, 205.0], [115.0, 215.0]],
                [[105.0, 205.0], [115.0, 215.0]],
                [[105.0, 205.0], [115.0, 215.0]],
            ],
            dtype=np.float64,
        ),
        chunks=(5, 2, 2),
    )
    run.create_array("usable_keypoints", data=np.asarray([True, True, True, False, True]), chunks=(5,))
    run.create_array("source_refined_row_ids", data=np.asarray([10, 11, 12, 13, 14], dtype=np.int64), chunks=(5,))
    mark_run_complete(run, parent_group=parent, run_name="refined_kp")


def _patch_ffprobe(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "_ffprobe_stream",
        lambda _path: {
            "width": 20,
            "height": 20,
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "color_range": "tv",
            "color_space": "bt709",
            "nb_frames": "10",
        },
    )


def test_inspect_acquisition_crop_pose_training_selects_sufficient_rows(tmp_path, monkeypatch) -> None:
    _patch_ffprobe(monkeypatch)
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_meta = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_external.mp4"
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.write_bytes(b"placeholder")
    _write_crop_meta(crop_meta)
    _write_source_zarr(zarr_path)

    report, selection, keypoints, _crop_meta, _resolved_meta, _resolved_video = mod.inspect_acquisition_crop_pose_training(
        zarr_path,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        margin_px=1.0,
    )

    assert report.selected_rows == 1
    assert report.usable_keypoint_rows == 4
    assert report.crop_video_dim_matches_crop_meta is True
    assert report.reject_counts == {
        "source_not_usable": 1,
        "missing_crop_meta_frame": 1,
        "blank_crop_frame": 1,
        "crop_has_no_detection": 1,
        "nonfinite_crop_geometry": 0,
        "nonfinite_keypoints": 0,
        "keypoints_outside_crop_margin": 0,
    }
    assert keypoints.run_name == "refined_kp"
    assert selection.crop_local_frame_ids.tolist() == [5]
    assert selection.source_recording_frame_ids.tolist() == [1]
    np.testing.assert_allclose(selection.source_crop_xywh, [[100.0, 200.0, 20.0, 20.0]])
    np.testing.assert_allclose(selection.keypoints_roi, [[[5.0, 5.0], [15.0, 15.0]]])
    np.testing.assert_allclose(selection.keypoints_norm, [[[0.25, 0.25], [0.75, 0.75]]])
    np.testing.assert_allclose(selection.bbox_roi_xyxy, [[5.0, 5.0, 15.0, 15.0]])
    np.testing.assert_allclose(selection.bbox_norm_xywh, [[0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(selection.realtime_detection_bbox_roi_xyxy, [[4.0, 4.0, 12.0, 12.0]])


class _FakeReader:
    source_height = 20
    source_width = 20

    def __init__(self, _path: Path, *, start_frame: int = 0, gpu_id: int = 0) -> None:
        assert start_frame == 0
        assert gpu_id == 0

    def iter_frames(self):
        for frame_idx in range(10):
            yield torch.full((20, 20), frame_idx, dtype=torch.uint8)

    def close(self) -> None:
        return None


def test_export_acquisition_crop_pose_training_writes_training_zarr(tmp_path, monkeypatch) -> None:
    _patch_ffprobe(monkeypatch)
    monkeypatch.setattr(mod, "get_git_info", lambda _repo: {"short_hash": "abc123", "commit_hash": "abc123"})
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {"environment": {"python": "test"}, "platform": {"system": "test"}},
    )
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    out_zarr = tmp_path / "training.zarr"
    crop_meta = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_external.mp4"
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.write_bytes(b"placeholder")
    _write_crop_meta(crop_meta)
    _write_source_zarr(zarr_path)

    result = mod.export_acquisition_crop_pose_training_zarr(
        zarr_path,
        out_zarr=out_zarr,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        crop_run_name="crop_acq_test",
        keypoint_export_run_name="keypoints_acq_test",
        margin_px=1.0,
        apply=True,
        require_cuda=False,
        reader_factory=_FakeReader,
    )

    assert result.applied is True
    assert result.crop_run == "crop_acq_test"
    assert result.keypoint_export_run == "keypoints_acq_test"
    assert result.crop_video.crop_video_path == str(crop_video)
    root = zarr.open_group(str(out_zarr), mode="r", use_consolidated=False)
    crop = root["crop_runs"]["crop_acq_test"]
    keypoints = root["keypoints_runs"]["keypoints_acq_test"]
    assert crop.attrs["source_type"] == "acquisition_crop_video"
    assert crop.attrs["frame_format_confirmation_status"] == "pending_orange_confirmation"
    assert keypoints.attrs["source_crop_run"] == "crop_acq_test"
    assert keypoints.attrs["keypoint_coordinate_space"] == "crop_video_frame_px"
    assert root["crop_runs"].attrs["latest"] == "crop_acq_test"
    assert root["keypoints_runs"].attrs["latest"] == "keypoints_acq_test"
    assert crop["roi_images"].shape == (1, 20, 20)
    assert int(crop["roi_images"][0, 0, 0]) == 5
    np.testing.assert_allclose(crop["source_crop_xywh"][:], [[100.0, 200.0, 20.0, 20.0]])
    np.testing.assert_allclose(crop["bbox_roi_xyxy"][:], [[5.0, 5.0, 15.0, 15.0]])
    np.testing.assert_allclose(crop["bbox_norm_coords"][:], [[0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(crop["realtime_detection_bbox_roi_xyxy"][:], [[4.0, 4.0, 12.0, 12.0]])
    assert crop["source_crop_local_frame_ids"][:].tolist() == [5]
    assert keypoints["source_refined_row_ids"][:].tolist() == [10]
    np.testing.assert_allclose(keypoints["keypoints_roi"][:], [[[5.0, 5.0], [15.0, 15.0]]])
    np.testing.assert_allclose(keypoints["keypoints_img"][:], [[[5.0, 5.0], [15.0, 15.0]]])
    np.testing.assert_allclose(keypoints["source_keypoints_img"][:], [[[105.0, 205.0], [115.0, 215.0]]])
