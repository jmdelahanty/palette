from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from fisheye.utils import append_acquisition_crop_video_training as mod


def _write_training_zarr(path: Path, recording_dir: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.create_group("raw_video")
    raw.attrs["recording_dir"] = str(recording_dir)
    raw.create_array("original_frame_indices", data=np.asarray([0, 10, 20, 30], dtype=np.int64), chunks=(4,))
    raw.create_array("images_full", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(4, 8, 8))
    raw.create_array("images_ds", data=np.zeros((4, 4, 4), dtype=np.uint8), chunks=(4, 4, 4))


def _write_crop_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,2,100,0,0,1,0,0.95,100,200,20,20,104,204,8,8",
                "11,3,101,0,0,1,1,0.90,100,200,20,20,104,204,8,8",
                "21,4,102,0,0,0,0,0.00,100,200,20,20,,,,",
                "31,5,103,0,0,1,0,0.80,110,210,20,20,116,218,4,6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_ffprobe(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "inspect_crop_video_stream",
        lambda _recording_dir, crop_meta_path, crop_video_path: mod.CropVideoStreamInfo(
            crop_video_path=str(crop_video_path),
            crop_meta_path=str(crop_meta_path),
            width=20,
            height=20,
            codec_name="h264",
            pix_fmt="yuv420p",
            color_range="tv",
            color_space="bt709",
            nb_frames="10",
            frame_format_confirmation_status="pending_orange_confirmation",
        ),
    )


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


def test_append_acquisition_crop_video_training_dry_run_selects_matching_samples(tmp_path, monkeypatch) -> None:
    _patch_ffprobe(monkeypatch)
    recording_dir = tmp_path / "recording"
    training_zarr = recording_dir / "zarr" / "recording_training.zarr"
    crop_meta = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_external.mp4"
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.write_bytes(b"placeholder")
    _write_training_zarr(training_zarr, recording_dir)
    _write_crop_meta(crop_meta)

    result = mod.append_acquisition_crop_video_training(
        training_zarr,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        apply=False,
    )

    assert result.applied is False
    assert result.source_sample_count == 4
    assert result.selected_rows == 2
    assert result.reject_counts == {
        "missing_crop_meta_frame": 0,
        "blank_crop_frame": 1,
        "crop_has_no_detection": 1,
        "nonfinite_crop_geometry": 0,
    }


def test_append_acquisition_crop_video_training_writes_crop_run(tmp_path, monkeypatch) -> None:
    _patch_ffprobe(monkeypatch)
    monkeypatch.setattr(mod, "get_git_info", lambda _repo: {"short_hash": "abc123", "commit_hash": "abc123"})
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {"environment": {"python": "test"}, "platform": {"system": "test"}},
    )
    recording_dir = tmp_path / "recording"
    training_zarr = recording_dir / "zarr" / "recording_training.zarr"
    crop_meta = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_meta.csv"
    crop_video = recording_dir / "derived" / "external_crop_recorder" / "Cam2010093_crop_external.mp4"
    crop_video.parent.mkdir(parents=True, exist_ok=True)
    crop_video.write_bytes(b"placeholder")
    _write_training_zarr(training_zarr, recording_dir)
    _write_crop_meta(crop_meta)

    result = mod.append_acquisition_crop_video_training(
        training_zarr,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta,
        crop_video_path=crop_video,
        run_name="crop_red_scare_acq",
        apply=True,
        require_cuda=False,
        reader_factory=_FakeReader,
    )

    assert result.applied is True
    assert result.run_name == "crop_red_scare_acq"
    root = zarr.open_group(str(training_zarr), mode="r", use_consolidated=False)
    crop = root["crop_runs"]["crop_red_scare_acq"]
    assert root["crop_runs"].attrs["latest"] == "crop_red_scare_acq"
    assert crop.attrs["source_pixels"] == "acquisition_crop_video"
    assert crop.attrs["frame_format_confirmation_status"] == "pending_orange_confirmation"
    assert crop.attrs["crop_detection_required"] is True
    assert crop.attrs["blank_crop_frames_excluded"] is True
    assert crop.attrs["source_sample_count"] == 4
    assert crop.attrs["selected_sample_count"] == 2
    assert crop.attrs["rejected_blank_crop_frame"] == 1
    assert crop.attrs["rejected_crop_has_no_detection"] == 1
    assert crop["roi_images"].shape == (2, 20, 20)
    assert crop["roi_images"][0, 0, 0] == 2
    assert crop["roi_images"][1, 0, 0] == 5
    assert crop["frame_indices"][:].tolist() == [0, 30]
    assert crop["source_training_row_indices"][:].tolist() == [0, 3]
    assert crop["source_recording_frame_ids"][:].tolist() == [1, 31]
    assert crop["source_crop_local_frame_ids"][:].tolist() == [2, 5]
    np.testing.assert_allclose(crop["source_crop_xywh"][:], [[100.0, 200.0, 20.0, 20.0], [110.0, 210.0, 20.0, 20.0]])
    np.testing.assert_allclose(crop["bbox_roi_xyxy"][:], [[4.0, 4.0, 12.0, 12.0], [6.0, 8.0, 10.0, 14.0]])
    np.testing.assert_allclose(crop["bbox_norm_coords"][:], [[0.4, 0.4, 0.4, 0.4], [0.4, 0.55, 0.2, 0.3]])
