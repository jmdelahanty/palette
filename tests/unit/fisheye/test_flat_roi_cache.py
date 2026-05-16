from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.flat_roi_cache import (
    FLAT_ROI_CACHE_LAYOUT,
    FLAT_ROI_CACHE_SCHEMA,
    build_flat_roi_cache,
    open_flat_roi_cache,
)


def _make_materialized_crop_archive(tmp_path: Path) -> tuple[Path, np.ndarray]:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"
    crop_parent.attrs["latest_materialized"] = "crop_001"

    roi_images = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["roi_size"] = [4, 3]
    crop.attrs["crop_signature"] = "sig-flat-cache-test"
    crop.attrs["crop_revision"] = "rev-001"
    crop.create_array("roi_images", data=roi_images, overwrite=True)
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.arange(5, dtype=np.int64), overwrite=True)
    return zarr_path, roi_images


def _make_geometry_only_crop_archive(tmp_path: Path) -> tuple[Path, np.ndarray]:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geom"

    crop = crop_parent.create_group("crop_geom")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [2, 2]
    crop.attrs["width"] = 5
    crop.attrs["height"] = 4
    crop.attrs["source_video_path"] = str(tmp_path / "fake.mp4")
    crop.attrs["crop_signature"] = "sig-flat-cache-geom-test"
    crop.attrs["crop_revision"] = "rev-geom-001"
    frame_indices = np.array([2, 0, 2], dtype=np.int64)
    roi_coordinates = np.array([[1, 1], [0, 0], [3, 2]], dtype=np.int32)
    crop.create_array("frame_indices", data=frame_indices, overwrite=True)
    crop.create_array("roi_coordinates_full", data=roi_coordinates, overwrite=True)

    frames = []
    for frame_idx in range(3):
        luma = np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        frames.append(luma)
    expected = np.stack(
        [
            frames[2][1:3, 1:3],
            frames[0][0:2, 0:2],
            frames[2][2:4, 3:5],
        ],
        axis=0,
    )
    return zarr_path, expected


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


def test_build_flat_roi_cache_roundtrips_through_manifest(tmp_path: Path) -> None:
    zarr_path, roi_images = _make_materialized_crop_archive(tmp_path)
    cache_dir = tmp_path / "cache"
    progress_events: list[dict] = []

    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=cache_dir,
        batch_size=2,
        compute_sha256=True,
        progress_callback=progress_events.append,
        progress_every_batches=1,
        progress_interval_seconds=999,
    )

    assert manifest["schema"] == FLAT_ROI_CACHE_SCHEMA
    assert manifest["layout"] == FLAT_ROI_CACHE_LAYOUT
    assert manifest["cache_complete"] is True
    assert manifest["source"]["crop_run_name"] == "crop_001"
    assert manifest["array"]["shape"] == [5, 4, 3]
    assert manifest["array"]["dtype"] == "uint8"
    assert manifest["array"]["sha256"]
    assert manifest["builder"]["timing"]["batches"] == 3
    assert manifest["builder"]["timing"]["rows"] == 5
    assert manifest["builder"]["timing"]["bytes"] == int(roi_images.size)
    assert manifest["builder"]["timing"]["read_seconds_total"] >= 0
    assert manifest["builder"]["timing"]["write_seconds_total"] >= 0

    assert progress_events[0]["event"] == "start"
    assert progress_events[-1]["event"] == "complete"
    batch_events = [event for event in progress_events if event["event"] == "batch"]
    assert [event["batch"]["index"] for event in batch_events] == [1, 2, 3]
    assert batch_events[-1]["progress"]["rows_written"] == 5
    assert batch_events[-1]["progress"]["bytes_written"] == int(roi_images.size)
    assert batch_events[-1]["batch"]["read_seconds"] >= 0
    assert batch_events[-1]["batch"]["write_seconds"] >= 0

    manifest_path = Path(str(manifest["manifest_path"]))
    cache = open_flat_roi_cache(
        manifest_path,
        expected_archive_path=zarr_path,
        expected_crop_run="crop_001",
        expected_shape=roi_images.shape,
    )
    try:
        np.testing.assert_array_equal(cache[1:4], roi_images[1:4])
    finally:
        cache.close()


def test_crop_image_source_reads_flat_roi_cache_manifest(tmp_path: Path) -> None:
    zarr_path, roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=3,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    source = CropImageSource.open(
        root,
        zarr_path=zarr_path,
        roi_cache_manifest=manifest["manifest_path"],
    )
    try:
        assert source.crop_run_name == "crop_001"
        assert source.roi_read_mode == "flat_bin_roi_cache"
        assert source.roi_cache_used is True
        assert source.roi_cache_backend == "flat_bin_v1"
        np.testing.assert_array_equal(source.read_indices([4, 0, 2]), roi_images[[4, 0, 2]])
    finally:
        source.close()


def test_flat_roi_cache_rejects_wrong_shape(tmp_path: Path) -> None:
    zarr_path, _roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=3,
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        open_flat_roi_cache(manifest["manifest_path"], expected_shape=(5, 4, 4))


def test_build_flat_roi_cache_pynvvc_luma_streams_rows_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path, expected = _make_geometry_only_crop_archive(tmp_path)
    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        for frame_idx in range(3)
    ]

    monkeypatch.setattr(
        flat_cache_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )
    progress_events: list[dict] = []

    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=2,
        roi_decode_backend="pynvvc_luma",
        progress_callback=progress_events.append,
        progress_every_batches=1,
    )

    assert manifest["builder"]["decode_backend_effective"] == "pynvvc_luma"
    assert manifest["builder"]["timing"]["decoded_frames"] == 3
    assert manifest["builder"]["timing"]["rows"] == 3
    assert any(
        event.get("event") == "batch"
        and event.get("batch", {}).get("decode_backend") == "pynvvc_luma"
        for event in progress_events
    )
    cache = open_flat_roi_cache(manifest["manifest_path"], expected_shape=expected.shape)
    try:
        np.testing.assert_array_equal(cache[:], expected)
    finally:
        cache.close()
