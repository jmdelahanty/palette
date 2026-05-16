from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.roi_pixel_contract import ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
from fisheye.utils import regenerate_training_crops_pynvvc as mod
from fisheye.utils.regenerate_training_crops_pynvvc import regenerate_training_crops_pynvvc


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

    def iter_frames(self):
        while self._offset < len(self._frames):
            frame = self._frames[self._offset]
            self._offset += 1
            yield frame

    def close(self) -> None:
        pass


class _FakeStreamMetadata:
    def __init__(self, *, height: int, width: int, num_frames: int) -> None:
        self.height = height
        self.width = width
        self.num_frames = num_frames


class _FakeIndexedPynvvcDecoder:
    def __init__(self, frames: list[np.ndarray]) -> None:
        import torch

        self._height = int(frames[0].shape[0])
        self._width = int(frames[0].shape[1])
        self.requested: list[int] = []
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

    def get_stream_metadata(self) -> _FakeStreamMetadata:
        return _FakeStreamMetadata(height=self._height, width=self._width, num_frames=len(self._frames))

    def get_batch_frames_by_index(self, indices: list[int]):
        self.requested.extend(int(index) for index in indices)
        return [self._frames[int(index)] for index in indices]


def _make_training_archive(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
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
    crop.attrs["source_video_path"] = str(tmp_path / "source.mp4")

    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        for frame_idx in range(5)
    ]
    frame_indices = np.array([0, 1, 2], dtype=np.int64)
    roi_coordinates = np.array([[0, 0], [1, 1], [3, 2]], dtype=np.int32)
    stale_roi_images = np.zeros((3, 2, 2), dtype=np.uint8)

    crop.create_array("frame_indices", data=frame_indices, overwrite=True)
    crop.create_array("roi_coordinates_full", data=roi_coordinates, overwrite=True)
    crop.create_array("bbox_norm_coords", data=np.zeros((3, 4), dtype=np.float32), overwrite=True)
    crop.create_array("roi_images", data=stale_roi_images, overwrite=True)
    return zarr_path, frames


def test_regenerate_training_crops_pynvvc_writes_new_luma_crop_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames = _make_training_archive(tmp_path)
    monkeypatch.setattr(mod, "_open_pynvvc_luma_reader", lambda _video_path: _FakePynvvcReader(frames))

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_001_pynvvc_luma",
        decode_chunk_frames=2,
    )

    assert report["status"] == "ok"
    assert report["source_frame_index_mapping"]["mode"] == "original_frame_indices"
    root = zarr.open_group(str(zarr_path), mode="r")
    crop_parent = root["crop_runs"]
    assert crop_parent.attrs["latest"] == "crop_001"
    target = crop_parent["crop_001_pynvvc_luma"]
    assert target.attrs["status"] == "completed"
    assert target.attrs["source_crop_run"] == "crop_001"
    assert target.attrs["roi_pixel_contract"]["name"] == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    assert np.array_equal(target["source_frame_indices"][:], np.array([0, 2, 4], dtype=np.int64))
    expected = np.stack(
        [
            frames[0][0:2, 0:2],
            frames[2][1:3, 1:3],
            frames[4][2:4, 3:5],
        ],
        axis=0,
    )
    assert np.array_equal(target["roi_images"][:], expected)
    assert np.array_equal(target["frame_indices"][:], np.array([0, 1, 2], dtype=np.int64))
    assert "bbox_norm_coords" in target


def test_regenerate_training_crops_pynvvc_dry_run_does_not_write(
    tmp_path: Path,
) -> None:
    zarr_path, _frames = _make_training_archive(tmp_path)

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_preview",
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "crop_preview" not in root["crop_runs"]


def test_regenerate_training_crops_pynvvc_auto_uses_indexed_for_sparse_frames(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, _frames = _make_training_archive(tmp_path)
    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx % 200)
        for frame_idx in range(201)
    ]
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["source_video_total_frames"] = 201
    root["raw_video/original_frame_indices"][:] = np.array([0, 100, 200], dtype=np.int32)
    indexed_decoder = _FakeIndexedPynvvcDecoder(frames)
    monkeypatch.setattr(
        mod,
        "_open_pynvvc_luma_indexed_decoder",
        lambda _video_path: indexed_decoder,
    )
    monkeypatch.setattr(
        mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: (_ for _ in ()).throw(AssertionError("sequential reader should not be used")),
    )

    report = regenerate_training_crops_pynvvc(
        zarr_path=zarr_path,
        source_crop_run="crop_001",
        target_crop_run="crop_001_pynvvc_luma_indexed",
        decode_mode="auto",
        decode_chunk_frames=2,
    )

    assert report["status"] == "ok"
    assert report["decode_mode_effective"] == "indexed"
    assert indexed_decoder.requested == [0, 100, 200]
    target = zarr.open_group(str(zarr_path), mode="r")["crop_runs/crop_001_pynvvc_luma_indexed"]
    assert target.attrs["decode_mode_effective"] == "indexed"
    expected = np.stack(
        [
            frames[0][0:2, 0:2],
            frames[100][1:3, 1:3],
            frames[200][2:4, 3:5],
        ],
        axis=0,
    )
    assert np.array_equal(target["roi_images"][:], expected)
