from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics import check_training_crop_pynvvc_pixel_parity as parity_mod
from fisheye.diagnostics.check_training_crop_pynvvc_pixel_parity import (
    check_training_crop_pynvvc_pixel_parity,
)


class _FakePynvvcReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        import torch

        self.source_height = int(frames[0].shape[0])
        self.source_width = int(frames[0].shape[1])
        self._frames = [
            torch.from_numpy(
                np.vstack(
                    [
                        frame,
                        np.zeros((max(1, frame.shape[0] // 2), frame.shape[1]), dtype=np.uint8),
                    ]
                )
            )
            for frame in frames
        ]
        self._offset = 0

    def decode_next(self, count: int):
        result = self._frames[self._offset : self._offset + int(count)]
        self._offset += len(result)
        return result

    def close(self) -> None:
        pass


class _FakePynvvcNv12Reader:
    def __init__(self, frames: list[np.ndarray], *, source_height: int, source_width: int) -> None:
        import torch

        self.source_height = int(source_height)
        self.source_width = int(source_width)
        self._frames = [torch.from_numpy(frame) for frame in frames]
        self._offset = 0

    def decode_next(self, count: int):
        result = self._frames[self._offset : self._offset + int(count)]
        self._offset += len(result)
        return result

    def close(self) -> None:
        pass


def _make_training_archive(tmp_path: Path, *, mismatch: bool = False) -> tuple[Path, list[np.ndarray]]:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["source_video_path"] = str(tmp_path / "source.mp4")
    root.attrs["width"] = 5
    root.attrs["height"] = 4
    root.attrs["source_video_total_frames"] = 5

    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.array([0, 2, 4], dtype=np.int32), overwrite=True)

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["video_source_type"] = "zarr"
    crop.attrs["roi_size"] = [2, 2]
    crop.attrs["height"] = 0
    crop.attrs["width"] = 0
    crop.attrs["source_video_path"] = "unknown"

    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        for frame_idx in range(5)
    ]
    frame_indices = np.array([0, 1, 2], dtype=np.int64)
    roi_coordinates = np.array([[0, 0], [1, 1], [3, 2]], dtype=np.int32)
    roi_images = np.stack(
        [
            frames[0][0:2, 0:2],
            frames[2][1:3, 1:3],
            frames[4][2:4, 3:5],
        ],
        axis=0,
    )
    if mismatch:
        roi_images = roi_images.copy()
        roi_images[1, 0, 0] = np.uint8((int(roi_images[1, 0, 0]) + 3) % 256)

    crop.create_array("frame_indices", data=frame_indices, overwrite=True)
    crop.create_array("roi_coordinates_full", data=roi_coordinates, overwrite=True)
    crop.create_array("roi_images", data=roi_images, overwrite=True)
    return zarr_path, frames


def test_training_crop_pynvvc_pixel_parity_maps_original_frame_indices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames = _make_training_archive(tmp_path)
    monkeypatch.setattr(
        parity_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )

    report = check_training_crop_pynvvc_pixel_parity(
        zarr_path=zarr_path,
        rows=[0, 1, 2],
        decode_chunk_frames=2,
    )

    assert report["status"] == "ok"
    assert report["frame_index_mapping"]["mode"] == "original_frame_indices"
    assert report["frame_index_mapping"]["selected_local_frame_max"] == 2
    assert report["frame_index_mapping"]["selected_source_frame_max"] == 4
    assert report["diff"]["byte_equal"] is True
    assert report["diff"]["max_abs_diff"] == 0
    assert report["source"]["stored_crop_pixel_contract"]["name"] == (
        "raw_video_images_full_to_uint8_grayscale"
    )
    assert report["source"]["candidate_pixel_contract"]["name"] == "nv12_luma_plane_uint8"


def test_training_crop_pynvvc_pixel_parity_reports_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames = _make_training_archive(tmp_path, mismatch=True)
    monkeypatch.setattr(
        parity_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )

    report = check_training_crop_pynvvc_pixel_parity(
        zarr_path=zarr_path,
        rows=[0, 1, 2],
    )

    assert report["status"] == "fail"
    assert report["diff"]["byte_equal"] is False
    assert report["diff"]["max_abs_diff"] == 3
    assert report["diff"]["mismatched_rows"] == 1


def test_training_crop_pynvvc_pixel_parity_can_test_limited_to_full_range_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames = _make_training_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    roi_images = root["crop_runs/crop_001/roi_images"]
    raw = np.asarray(roi_images[:], dtype=np.float32)
    roi_images[:] = np.clip((raw - 16.0) * (255.0 / 219.0), 0.0, 255.0).round().astype(np.uint8)
    monkeypatch.setattr(
        parity_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )

    report = check_training_crop_pynvvc_pixel_parity(
        zarr_path=zarr_path,
        rows=[0, 1, 2],
        candidate_pixel_mode="luma_limited_to_full_range",
    )

    assert report["status"] == "ok"
    assert report["inputs"]["candidate_pixel_mode"] == "luma_limited_to_full_range"
    assert report["source"]["candidate_pixel_contract"]["name"] == (
        "nv12_luma_limited_to_full_range_uint8"
    )


def test_training_crop_pynvvc_pixel_parity_can_test_nv12_rgb_weighted_gray_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import torch

    zarr_path, _frames = _make_training_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["width"] = 6
    root.attrs["height"] = 4
    crop = root["crop_runs/crop_001"]
    crop.attrs["width"] = 6
    crop.attrs["height"] = 4
    crop.attrs["roi_size"] = [2, 2]
    root["raw_video"].attrs["gpu_fp16"] = False
    crop["roi_coordinates_full"][:] = np.array([[0, 0], [2, 1], [4, 2]], dtype=np.int32)

    source_height = 4
    source_width = 6
    y = np.arange(source_height * source_width, dtype=np.uint8).reshape(source_height, source_width) + 40
    uv = np.array(
        [
            [90, 140, 100, 150, 110, 160],
            [120, 130, 125, 135, 130, 140],
        ],
        dtype=np.uint8,
    )
    base_nv12 = np.vstack([y, uv])
    nv12_frames = [np.clip(base_nv12.astype(np.int16) + frame_idx * 3, 0, 255).astype(np.uint8) for frame_idx in range(5)]
    frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
    source_frames = np.asarray(root["raw_video/original_frame_indices"][:], dtype=np.int64)[frame_indices]
    roi_coordinates = np.asarray(crop["roi_coordinates_full"][:], dtype=np.int32)
    expected = []
    for row, source_frame in enumerate(source_frames):
        crops = parity_mod._crop_pynvvc_nv12_rgb_weighted_gray_frame(
            torch.from_numpy(nv12_frames[int(source_frame)]),
            roi_ids=[row],
            roi_coordinates_full=roi_coordinates,
            roi_shape=(2, 2),
            video_shape=(source_height, source_width),
            matrix="bt601",
            grayscale_fp16=False,
        )
        expected.append(crops.numpy()[0])
    crop["roi_images"][:] = np.stack(expected, axis=0)

    monkeypatch.setattr(
        parity_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcNv12Reader(
            nv12_frames,
            source_height=source_height,
            source_width=source_width,
        ),
    )

    report = check_training_crop_pynvvc_pixel_parity(
        zarr_path=zarr_path,
        rows=[0, 1, 2],
        candidate_pixel_mode="nv12_bt601_limited_rgb_weighted_gray",
    )

    assert report["status"] == "ok"
    assert report["inputs"]["candidate_pixel_mode"] == "nv12_bt601_limited_rgb_weighted_gray"
    assert report["inputs"]["candidate_weighted_gray_fp16"] is False
    assert report["source"]["candidate_pixel_contract"]["name"] == (
        "nv12_bt601_limited_rgb_weighted_gray_uint8"
    )
