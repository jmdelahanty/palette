from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr
from rich.console import Console

import fisheye.shared.crop_image_source as crop_mod
from fisheye.shared.crop_image_source import CropImageSource, resolve_crop_run


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children: dict[str, Any] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(self, name: str, data, **_kwargs) -> _FakeArray:
        array = _FakeArray(np.asarray(data))
        self._children[name] = array
        return array

    def get(self, name: str) -> Any:
        return self._children.get(name)

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        if "/" not in key:
            return self._children[key]
        current: Any = self
        for token in key.split("/"):
            if not isinstance(current, _FakeGroup):
                raise KeyError(key)
            current = current._children[token]
        return current


def _make_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["source_video_path"] = "/tmp/fallback.mp4"
    return root


def _make_geometry_only_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 6
    root.attrs["height"] = 6

    raw_video = root.create_group("raw_video")
    frames = np.stack(
        [
            np.arange(36, dtype=np.uint8).reshape(6, 6),
            (np.arange(36, dtype=np.uint8).reshape(6, 6) + 50).astype(np.uint8),
        ],
        axis=0,
    )
    raw_video.create_array("images_full", data=frames, overwrite=True)

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.create_group("crop_materialized")

    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [4, 4]
    crop.attrs["crop_signature"] = "sig-001"
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[-1, 1], [3, 2]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    return zarr_path


def _make_external_geometry_only_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_external_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 6
    root.attrs["height"] = 6
    root.attrs["source_video_path"] = str(tmp_path / "source.mp4")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [4, 4]
    crop.attrs["crop_signature"] = "sig-external-001"
    crop.attrs["source_video_path"] = str(tmp_path / "source.mp4")
    crop.attrs["roi_storage"] = "uncompressed"
    crop.attrs["roi_chunk_len"] = 128
    crop.attrs["roi_use_sharding"] = False
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[-1, 1], [3, 2]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    return zarr_path


def test_resolve_crop_run_prefers_latest_any_for_mixed_mode_reader() -> None:
    root = _make_root()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"

    crop_parent.create_group("crop_materialized")
    crop_parent.create_group("crop_geometry")

    _parent, _group, run_name = resolve_crop_run(root)

    assert run_name == "crop_geometry"


def test_crop_image_source_reads_materialized_roi_batches() -> None:
    root = _make_root()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"

    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.create_array(
        "roi_images",
        data=np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5),
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32),
    )
    crop.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))

    source = CropImageSource.open(root)

    batch = source.read_slice(1, 3)

    assert source.crop_run_name == "crop_001"
    assert source.storage_mode == "materialized"
    np.testing.assert_array_equal(
        batch,
        np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)[1:3],
    )


def test_crop_image_source_reconstructs_geometry_only_rois_from_raw_video() -> None:
    root = _make_root()
    raw_video = root.create_group("raw_video")
    frames = np.stack(
        [
            np.arange(36, dtype=np.uint8).reshape(6, 6),
            (np.arange(36, dtype=np.uint8).reshape(6, 6) + 50).astype(np.uint8),
        ],
        axis=0,
    )
    raw_video.create_array("images_full", data=frames)

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"

    crop_parent.create_group("crop_materialized")

    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [4, 4]
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[-1, 1], [3, 2]], dtype=np.int32),
    )
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))

    source = CropImageSource.open(root)

    batch = source.read_slice(0, 2)

    expected_first = np.array(
        [
            [0, 6, 7, 8],
            [0, 12, 13, 14],
            [0, 18, 19, 20],
            [0, 24, 25, 26],
        ],
        dtype=np.uint8,
    )
    expected_second = np.array(
        [
            [65, 66, 67, 0],
            [71, 72, 73, 0],
            [77, 78, 79, 0],
            [83, 84, 85, 0],
        ],
        dtype=np.uint8,
    )

    assert source.crop_run_name == "crop_geometry"
    assert source.storage_mode == "geometry_only"
    assert source.frame_source_kind == "raw_video/images_full"
    np.testing.assert_array_equal(batch[0], expected_first)
    np.testing.assert_array_equal(batch[1], expected_second)


def test_crop_image_source_builds_and_reuses_temporary_roi_cache(tmp_path: Path) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    cache_dir = tmp_path / "roi-cache"

    source = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
    )
    first_batch = source.read_slice(0, 2)

    assert source.roi_read_mode == "temporary_cache"
    assert source.roi_cache_used is True
    assert source.roi_cache_created is True
    assert source.roi_cache_path is not None
    assert Path(source.roi_cache_path).exists()
    source.close()

    root_rw = zarr.open_group(str(zarr_path), mode="a")
    root_rw["raw_video"]["images_full"][:] = 0

    source_reused = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
    )
    reused_batch = source_reused.read_slice(0, 2)

    assert source_reused.roi_cache_used is True
    assert source_reused.roi_cache_created is False
    np.testing.assert_array_equal(reused_batch, first_batch)
    source_reused.close()


def test_crop_image_source_cache_console_messages_include_run_and_runtime_summary(tmp_path: Path) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    cache_dir = tmp_path / "roi-cache"

    build_console = Console(record=True, width=200)
    source = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
        console=build_console,
    )
    source.close()
    build_text = build_console.export_text()

    assert "Building temporary ROI cache" in build_text
    assert "crop_run=crop_geometry" in build_text
    assert "Temporary ROI cache ready" in build_text
    assert "acceleration=cpu, backend=standard_zarr" in build_text

    reuse_console = Console(record=True, width=200)
    source_reused = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
        console=reuse_console,
    )
    source_reused.close()
    reuse_text = reuse_console.export_text()

    assert "Reusing temporary ROI cache" in reuse_text
    assert "crop_run=crop_geometry" in reuse_text
    assert "acceleration=cpu, backend=standard_zarr" in reuse_text


def test_crop_image_source_auto_policy_can_promote_geometry_only_run_to_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    monkeypatch.setattr(crop_mod, "_ROI_CACHE_AUTO_MIN_SOURCE_PIXELS", 1)

    source = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="auto",
        roi_cache_dir=tmp_path / "roi-cache",
    )

    assert source.roi_cache_used is True
    assert source.roi_read_mode == "temporary_cache"
    source.close()


def test_crop_image_source_cache_key_changes_when_crop_signature_changes(tmp_path: Path) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    cache_dir = tmp_path / "roi-cache"

    first = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
    )
    first_cache_path = first.roi_cache_path
    assert first_cache_path is not None
    first.close()

    root = zarr.open_group(str(zarr_path), mode="a")
    crop = root["crop_runs"]["crop_geometry"]
    crop.attrs["crop_signature"] = {
        "signature_version": 2,
        "crop_revision": 1,
        "roi_size": [4, 4],
        "detection_source_path": "detect_runs/detect_001",
    }

    second = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
    )
    assert second.roi_cache_path is not None
    assert second.roi_cache_path != first_cache_path
    assert second.roi_cache_created is True
    second.close()


def test_crop_image_source_uses_accelerated_external_cache_builder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_external_geometry_only_archive(tmp_path)
    cache_dir = tmp_path / "roi-cache"
    calls: list[dict[str, object]] = []

    import fisheye.tracking.crop as tracking_crop

    def _fake_materialize_external_roi_cache_for_crop_run(**kwargs):
        calls.append(kwargs)
        cache_root = zarr.open_group(str(kwargs["cache_path"]), mode="a")
        cache_root.create_array(
            "roi_images",
            data=np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4),
            overwrite=True,
        )
        return {
            "write_backend_requested": "kvikio",
            "write_backend_effective": "kvikio_gds",
            "acceleration": "gpu",
            "fallback_reason": None,
            "decode_seconds": 1.0,
            "compute_seconds": 2.0,
            "write_seconds": 3.0,
            "duration_seconds": 6.0,
            "roi_chunk_len": 128,
            "roi_shard_len": 128,
            "roi_storage": "uncompressed",
            "roi_use_sharding": False,
            "gpu_chunk_frames": 96,
        }

    monkeypatch.setattr(
        tracking_crop,
        "materialize_external_roi_cache_for_crop_run",
        _fake_materialize_external_roi_cache_for_crop_run,
    )

    source = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
    )

    assert source.roi_cache_used is True
    assert source.roi_cache_created is True
    assert source.roi_read_mode == "temporary_cache"
    assert len(calls) == 1
    assert calls[0]["write_backend"] == "kvikio"
    assert calls[0]["source_zarr_path"] == zarr_path.resolve()
    assert calls[0]["crop_run_name"] == "crop_geometry"
    assert calls[0]["roi_storage"] == "uncompressed"
    assert calls[0]["roi_chunk_size"] == 128
    assert calls[0]["use_sharding"] is False
    assert calls[0]["roi_shard_size"] is None
    source.close()
