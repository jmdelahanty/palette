from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from rich.console import Console

import fisheye.shared.crop_image_source as crop_mod
from fisheye.shared.crop_image_source import CropImageSource, resolve_crop_run
from fisheye.shared.crop_roi_layout import (
    DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
)
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    orange_mono_pynvvc_luma_hybrid_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)


@pytest.mark.parametrize(
    ("top_left", "expected"),
    (
        ((-1, 1), [[0, 4, 5], [0, 8, 9], [0, 0, 0]]),
        ((2, 1), [[6, 7, 0], [10, 11, 0], [0, 0, 0]]),
        ((1, -1), [[0, 0, 0], [1, 2, 3], [5, 6, 7]]),
        ((1, 2), [[9, 10, 11], [0, 0, 0], [0, 0, 0]]),
        ((-1, -1), [[0, 0, 0], [0, 0, 1], [0, 4, 5]]),
    ),
)
def test_crop_from_top_left_zero_pads_every_source_frame_edge(
    top_left: tuple[int, int],
    expected: list[list[int]],
) -> None:
    frame = np.arange(12, dtype=np.uint8).reshape(3, 4)

    result = crop_mod._crop_from_top_left(frame, top_left, (3, 3))

    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.uint8))


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
        return [
            key
            for key, value in self._children.items()
            if isinstance(value, _FakeGroup)
        ]

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
    crop.create_array(
        "frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True
    )
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
    crop.create_array(
        "frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True
    )
    return zarr_path


def _make_external_geometry_only_root(source_video_path: str) -> _FakeGroup:
    root = _make_root()
    root.attrs["width"] = 6
    root.attrs["height"] = 6
    root.attrs["source_video_path"] = source_video_path

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [4, 4]
    crop.attrs["crop_signature"] = "sig-external-001"
    crop.attrs["source_video_path"] = source_video_path
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[-1, 1], [3, 2]], dtype=np.int32),
    )
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    return root


def test_crop_image_source_uses_v2_root_locator_when_crop_has_no_video_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording = tmp_path / "recording"
    source_video_path = recording / "cams" / "source.mp4"
    root = _make_external_geometry_only_root(str(source_video_path))
    crop = root["crop_runs"]["crop_geometry"]
    del crop.attrs["source_video_path"]
    root.attrs.update(
        {
            "recording_path": str(recording),
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/source.mp4",
                },
                "source_path": str(source_video_path),
            },
        }
    )
    captured: dict[str, Path] = {}

    class _FakeExternalReader:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "_ExternalFrameReader", _FakeExternalReader)

    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="cpu",
    )

    assert source.frame_source_path == str(source_video_path.resolve())
    assert captured["path"] == source_video_path.resolve()
    source.close()


def test_crop_image_source_uses_validated_source_video_relocation_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    declared = tmp_path / "recording" / "cams" / "source.mp4"
    relocated = tmp_path / "scratch" / "source.mp4"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"relocated-video")
    root = _make_external_geometry_only_root(str(declared))
    root.attrs["source_video_metadata"] = {
        "file_fingerprint": {"size_bytes": relocated.stat().st_size}
    }
    captured: dict[str, Path] = {}

    class _FakeExternalReader:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "_ExternalFrameReader", _FakeExternalReader)

    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="cpu",
        source_video_path_override=relocated,
    )

    assert source.frame_source_path == str(relocated.resolve())
    assert source.frame_source_declared_path == str(declared)
    assert source.frame_source_path_override_used is True
    assert captured["path"] == relocated.resolve()
    identity = source._build_frame_source_identity()
    assert identity["frame_source_declared_path"] == str(declared)
    assert identity["frame_source_path_override_used"] is True
    source.close()


def test_crop_image_source_rejects_wrong_sized_source_video_override(
    tmp_path: Path,
) -> None:
    declared = tmp_path / "recording" / "cams" / "source.mp4"
    relocated = tmp_path / "scratch" / "source.mp4"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(b"wrong-size")
    root = _make_external_geometry_only_root(str(declared))
    root.attrs["source_video_metadata"] = {
        "file_fingerprint": {"size_bytes": relocated.stat().st_size + 1}
    }

    with pytest.raises(ValueError, match="override size differs"):
        CropImageSource.open(
            root,
            roi_cache_policy="never",
            roi_live_acceleration="cpu",
            source_video_path_override=relocated,
        )


def test_crop_image_source_maps_acquisition_frames_into_clip_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    declared = tmp_path / "recording" / "cams" / "source.mp4"
    clip = tmp_path / "scratch" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"derived-stream-copy-window")
    root = _make_external_geometry_only_root(str(declared))
    root["crop_runs/crop_geometry"]._children["frame_indices"] = _FakeArray(
        np.asarray([100, 101], dtype=np.int64)
    )
    root.attrs["source_video_metadata"] = {
        "file_fingerprint": {"size_bytes": clip.stat().st_size + 1000}
    }
    reads: list[int] = []

    class _FakeExternalReader:
        def __init__(self, path: Path) -> None:
            assert path == clip.resolve()

        def read_frame(self, frame_index: int) -> np.ndarray:
            reads.append(int(frame_index))
            return np.full((6, 6), int(frame_index) + 10, dtype=np.uint8)

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "_ExternalFrameReader", _FakeExternalReader)

    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="cpu",
        source_video_path_override=clip,
        source_video_frame_offset=100,
        source_video_frame_count=2,
    )
    try:
        pixels = source.read_slice(0, 2)
        assert reads == [0, 1]
        assert 10 in pixels[0]
        assert 11 in pixels[1]
        identity = source._build_frame_source_identity()
        assert identity["override_semantics"] == (
            "acquisition_frame_window_relocation_v1"
        )
        assert identity["source_video_frame_offset"] == 100
        assert identity["source_video_frame_count"] == 2
    finally:
        source.close()


def test_crop_image_source_rejects_rows_outside_clip_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    declared = tmp_path / "recording" / "cams" / "source.mp4"
    clip = tmp_path / "scratch" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"window")
    root = _make_external_geometry_only_root(str(declared))
    root["crop_runs/crop_geometry"]._children["frame_indices"] = _FakeArray(
        np.asarray([99, 102], dtype=np.int64)
    )

    class _FakeExternalReader:
        def __init__(self, _path: Path) -> None:
            return None

        def read_frame(self, _frame_index: int) -> np.ndarray:
            raise AssertionError("Out-of-window video reads must fail before decode.")

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "_ExternalFrameReader", _FakeExternalReader)
    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="cpu",
        source_video_path_override=clip,
        source_video_frame_offset=100,
        source_video_frame_count=2,
    )
    try:
        with pytest.raises(IndexError, match="precedes source-video window"):
            source.read_indices([0])
        with pytest.raises(IndexError, match="exceeds source-video window"):
            source.read_indices([1])
    finally:
        source.close()


def test_crop_image_source_requires_complete_clip_window_contract(
    tmp_path: Path,
) -> None:
    declared = tmp_path / "recording" / "cams" / "source.mp4"
    clip = tmp_path / "scratch" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"window")
    root = _make_external_geometry_only_root(str(declared))

    with pytest.raises(ValueError, match="nonzero source_video_frame_offset"):
        CropImageSource.open(
            root,
            roi_cache_policy="never",
            roi_live_acceleration="cpu",
            source_video_path_override=clip,
            source_video_frame_offset=100,
        )


def _make_acquisition_crop_video_root(crop_video_path: str) -> _FakeGroup:
    root = _make_root()

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_acquisition"

    crop = crop_parent.create_group("crop_acquisition")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["source_pixels"] = "acquisition_crop_video"
    crop.attrs["roi_pixel_provider"] = "acquisition_crop_video"
    crop.attrs["source_crop_video_path"] = crop_video_path
    crop.create_array("frame_indices", data=np.array([10, 11], dtype=np.int64))
    crop.create_array(
        "source_crop_video_frame_indices", data=np.array([2, 4], dtype=np.int64)
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[100, 200], [110, 210]], dtype=np.int32),
    )
    crop.create_array(
        "roi_sizes_full",
        data=np.array([[5, 3], [5, 3]], dtype=np.int32),
    )
    crop.create_array(
        "source_crop_xywh",
        data=np.array([[100, 200, 5, 3], [110, 210, 5, 3]], dtype=np.float32),
    )
    return root


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


def test_implicit_crop_resolution_skips_selector_ineligible_pointer_target() -> None:
    root = _make_root()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs.update(
        {
            "latest_any": "crop_staged",
            "latest": "crop_previous",
            "latest_materialized": "crop_previous",
        }
    )
    staged = crop_parent.create_group("crop_staged")
    staged.attrs["stage_selector_eligible"] = False
    crop_parent.create_group("crop_previous")

    _parent, _group, run_name = resolve_crop_run(root)

    assert run_name == "crop_previous"


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


def test_crop_image_source_reads_acquisition_crop_video_with_pynvvc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_video_path = tmp_path / "crop.mp4"
    crop_video_path.touch()
    root = _make_acquisition_crop_video_root(str(crop_video_path))

    class _FakePynvvcReader:
        source_height = 3
        source_width = 5

        def __init__(
            self, video_path: Path, *, start_frame: int = 0, gpu_id: int = 0
        ) -> None:
            assert Path(video_path) == crop_video_path
            assert start_frame == 0
            assert gpu_id == 0

        def iter_frames(self):
            for frame_idx in range(6):
                yield np.full((3, 5), frame_idx, dtype=np.uint8)

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "PynvvcLumaRgbReader", _FakePynvvcReader)

    source = CropImageSource.open(
        root, crop_run="crop_acquisition", roi_cache_policy="always"
    )
    batch = source.read_slice(0, 2)

    assert source.frame_source_kind == "acquisition_crop_video"
    assert source.frame_source_path == str(crop_video_path)
    assert source.roi_read_mode == "acquisition_crop_video"
    assert source.roi_shape == (3, 5)
    assert source.roi_cache_used is False
    assert source.roi_pixel_contract is not None
    assert source.roi_pixel_contract["name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert source.roi_pixel_contract["source_pixels"] == "acquisition_crop_video"
    assert source.roi_pixel_contract["decode_backend"] == "pynvvc_luma"
    assert (
        source.roi_pixel_contract["applied_range_semantics"]
        == "orange_mono8_full_range_0_255"
    )
    np.testing.assert_array_equal(batch[0], np.full((3, 5), 2, dtype=np.uint8))
    np.testing.assert_array_equal(batch[1], np.full((3, 5), 4, dtype=np.uint8))
    source.close()


def test_acquisition_crop_video_rejects_mismatched_flat_cache_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_video_path = tmp_path / "crop.mp4"
    crop_video_path.touch()
    root = _make_acquisition_crop_video_root(str(crop_video_path))
    crop = root["crop_runs/crop_acquisition"]

    class _FakeFlatCache:
        manifest = {
            "cache_key": "wrong-contract",
            "builder": {
                "pixel_contract": {
                    "name": "decoder_default",
                }
            },
        }
        manifest_path = tmp_path / "wrong.flat_roi_cache.json"
        closed = False

        def close(self) -> None:
            self.closed = True

    cache = _FakeFlatCache()
    monkeypatch.setattr(
        crop_mod,
        "open_flat_roi_cache",
        lambda *_args, **_kwargs: cache,
    )
    source = CropImageSource(
        root=root,
        crop_group=crop,
        crop_run_name="crop_acquisition",
        storage_mode="geometry_only",
        roi_shape=(3, 5),
        roi_coordinates_full=np.asarray(
            crop["roi_coordinates_full"][:],
            dtype=np.int32,
        ),
        frame_indices=np.asarray(crop["frame_indices"][:], dtype=np.int64),
        frame_source_kind="acquisition_crop_video",
        frame_source_path=str(crop_video_path),
    )

    with pytest.raises(ValueError, match="pixel contract"):
        source._activate_flat_bin_cache(
            manifest_path=cache.manifest_path,
            zarr_path=None,
        )
    assert cache.closed is True


def test_crop_image_source_reads_hybrid_acquisition_video_and_supplemental_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop_video_path = tmp_path / "crop.mp4"
    crop_video_path.touch()
    manifest_path = tmp_path / "supplemental.flat_roi_cache.json"
    root = _make_acquisition_crop_video_root(str(crop_video_path))
    crop = root["crop_runs/crop_acquisition"]
    crop.create_array("frame_indices", data=np.array([10, 11, 12], dtype=np.int64))
    crop.create_array(
        "source_crop_video_frame_indices",
        data=np.array([2, -1, 4], dtype=np.int64),
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[100, 200], [105, 205], [110, 210]], dtype=np.int32),
    )
    crop.create_array(
        "roi_sizes_full",
        data=np.array([[5, 3], [5, 3], [5, 3]], dtype=np.int32),
    )
    crop.create_array(
        "source_pixel_kind_codes",
        data=np.array([0, 1, 0], dtype=np.int8),
    )
    crop.create_array(
        "supplemental_cache_row_indices",
        data=np.array([-1, 0, -1], dtype=np.int64),
    )
    crop.attrs["supplemental_roi_cache_manifest"] = str(manifest_path)
    crop.attrs.update(
        {
            "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
            "roi_pixel_provider": ("hybrid_acquisition_crop_video_offline_supplement"),
            "roi_pixel_contract": orange_mono_pynvvc_luma_hybrid_pixel_contract(),
        }
    )

    class _FakePynvvcReader:
        source_height = 3
        source_width = 5

        def __init__(
            self, video_path: Path, *, start_frame: int = 0, gpu_id: int = 0
        ) -> None:
            assert Path(video_path) == crop_video_path
            assert start_frame == 0
            assert gpu_id == 0

        def iter_frames(self):
            for frame_idx in range(6):
                yield np.full((3, 5), frame_idx, dtype=np.uint8)

        def close(self) -> None:
            return None

    class _FakeFlatCache:
        manifest = {
            "cache_key": "supplemental",
            "builder": {
                "pixel_contract": orange_mono_pynvvc_luma_pixel_contract(
                    source_pixels=SOURCE_PIXELS_RAW_CAMERA_VIDEO,
                )
            },
        }
        shape = (1, 3, 5)
        dtype = np.dtype(np.uint8)

        def __init__(self) -> None:
            self.manifest_path = manifest_path

        def __getitem__(self, key):
            return np.full(self.shape, 99, dtype=np.uint8)[key]

        def close(self) -> None:
            return None

    monkeypatch.setattr(crop_mod, "PynvvcLumaRgbReader", _FakePynvvcReader)
    monkeypatch.setattr(
        crop_mod, "open_flat_roi_cache", lambda *_args, **_kwargs: _FakeFlatCache()
    )

    source = CropImageSource.open(root, crop_run="crop_acquisition")
    batch = source.read_slice(0, 3)

    assert source.frame_source_kind == "acquisition_crop_video"
    assert source.roi_read_mode == "acquisition_crop_video"
    assert source.roi_pixel_contract == orange_mono_pynvvc_luma_hybrid_pixel_contract()
    np.testing.assert_array_equal(batch[0], np.full((3, 5), 2, dtype=np.uint8))
    np.testing.assert_array_equal(batch[1], np.full((3, 5), 99, dtype=np.uint8))
    np.testing.assert_array_equal(batch[2], np.full((3, 5), 4, dtype=np.uint8))
    source.close()


def test_crop_image_source_builds_and_reuses_temporary_roi_cache(
    tmp_path: Path,
) -> None:
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
    cache_root = zarr.open_group(str(source.roi_cache_path), mode="r")
    assert cache_root.attrs["cache_layout_profile"] == SCRATCH_ROI_CACHE_LAYOUT_PROFILE
    assert cache_root.attrs["cache_roi_storage"] == "uncompressed"
    assert cache_root.attrs["cache_roi_chunk_len"] == min(
        DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
        int(cache_root["roi_images"].shape[0]),
    )
    assert cache_root.attrs["cache_roi_use_sharding"] is False
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


def test_crop_image_source_records_canonical_path_for_staged_flat_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    scratch_manifest = tmp_path / "scratch" / "cache.flat_roi_cache.json"
    canonical_manifest = tmp_path / "canonical" / "cache.flat_roi_cache.json"

    class _FakeFlatCache:
        def __init__(self) -> None:
            self.manifest_path = scratch_manifest.resolve()
            self.manifest = {
                "cache_key": "cache-key-001",
                "staging": {"requested_manifest_path": str(canonical_manifest)},
                "builder": {"pixel_contract": {"name": "nv12_luma_plane_uint8"}},
            }
            self.shape = (2, 4, 4)
            self.dtype = np.dtype(np.uint8)

        def __getitem__(self, key):
            return np.zeros(self.shape, dtype=np.uint8)[key]

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        crop_mod, "open_flat_roi_cache", lambda *_args, **_kwargs: _FakeFlatCache()
    )

    source = CropImageSource.open(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        crop_run="crop_geometry",
        roi_cache_manifest=scratch_manifest,
    )

    assert source.roi_cache_path == str(scratch_manifest.resolve())
    assert source.roi_cache_canonical_path == str(canonical_manifest)
    assert source.roi_cache_backend == "flat_bin_v1"
    assert source.roi_read_mode == "flat_bin_roi_cache"
    source.close()


def test_crop_image_source_cache_console_messages_include_run_and_runtime_summary(
    tmp_path: Path,
) -> None:
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


def test_crop_image_source_cache_key_changes_when_crop_signature_changes(
    tmp_path: Path,
) -> None:
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

    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (True, "available"),
    )

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
            "gpu_chunk_frames": DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
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
    assert calls[0]["roi_chunk_size"] == DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN
    assert calls[0]["gpu_chunk_frames"] == DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES
    assert calls[0]["use_sharding"] is False
    assert calls[0]["roi_shard_size"] is None
    source.close()


def test_crop_image_source_external_cache_does_not_inherit_canonical_crop_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = _make_external_geometry_only_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    crop = root["crop_runs"]["crop_geometry"]
    crop.attrs["roi_storage"] = "compressed"
    crop.attrs["roi_chunk_len"] = 17
    crop.attrs["roi_use_sharding"] = True
    crop.attrs["roi_shard_len"] = 68

    calls: list[dict[str, object]] = []

    import fisheye.tracking.crop as tracking_crop

    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (True, "available"),
    )

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
            "roi_chunk_len": DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
            "roi_shard_len": DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
            "roi_storage": "uncompressed",
            "roi_use_sharding": False,
            "roi_layout_profile": SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
            "gpu_chunk_frames": DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
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
        roi_cache_dir=tmp_path / "roi-cache",
    )

    assert len(calls) == 1
    assert calls[0]["roi_storage"] == "uncompressed"
    assert calls[0]["roi_chunk_size"] == DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN
    assert calls[0]["gpu_chunk_frames"] == DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES
    assert calls[0]["use_sharding"] is False
    assert calls[0]["roi_shard_size"] is None
    source.close()


def test_crop_image_source_external_geometry_live_gpu_uses_gpu_helper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video_path = str((tmp_path / "source.mp4").resolve())
    root = _make_external_geometry_only_root(source_video_path)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (True, "available"),
    )

    def _fake_gpu_batch_reader(
        *,
        video_path: Path,
        frame_indices: np.ndarray,
        roi_coordinates_full: np.ndarray,
        roi_shape: tuple[int, int],
        video_shape: tuple[int, int],
        gpu_chunk_frames: int,
    ) -> np.ndarray:
        seen["video_path"] = video_path
        seen["frame_indices"] = frame_indices.copy()
        seen["roi_coordinates_full"] = roi_coordinates_full.copy()
        seen["roi_shape"] = roi_shape
        seen["video_shape"] = video_shape
        seen["gpu_chunk_frames"] = gpu_chunk_frames
        return np.full(
            (frame_indices.shape[0], roi_shape[0], roi_shape[1]), 17, dtype=np.uint8
        )

    monkeypatch.setattr(
        crop_mod, "_read_external_video_live_gpu_batch", _fake_gpu_batch_reader
    )

    def _unexpected_cpu_read(
        self, roi_indices: np.ndarray
    ) -> np.ndarray:  # noqa: ANN001
        raise AssertionError(f"CPU live read should not be used: {roi_indices}")

    monkeypatch.setattr(CropImageSource, "_read_live_indices_cpu", _unexpected_cpu_read)

    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="gpu",
        roi_live_gpu_chunk_frames=17,
    )
    batch = source.read_slice(0, 2)

    assert source.roi_live_acceleration_requested == "gpu"
    assert source.roi_live_acceleration_effective == "gpu"
    assert source.roi_live_acceleration_fallback_reason is None
    np.testing.assert_array_equal(batch, np.full((2, 4, 4), 17, dtype=np.uint8))
    assert seen["video_path"] == Path(source_video_path)
    np.testing.assert_array_equal(
        seen["frame_indices"], np.array([0, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        seen["roi_coordinates_full"],
        np.array([[-1, 1], [3, 2]], dtype=np.int32),
    )
    assert seen["roi_shape"] == (4, 4)
    assert seen["video_shape"] == (6, 6)
    assert seen["gpu_chunk_frames"] == 17
    source.close()


def test_crop_image_source_external_geometry_live_gpu_auto_rejects_cpu_fallback_when_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _make_external_geometry_only_root(str((tmp_path / "source.mp4").resolve()))

    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (False, "cuda_unavailable"),
    )

    with pytest.raises(
        RuntimeError,
        match="GPU decode unavailable; refusing CPU fallback.*cuda_unavailable",
    ):
        CropImageSource.open(
            root,
            roi_cache_policy="never",
            roi_live_acceleration="auto",
        )


def test_crop_image_source_external_geometry_live_gpu_explicit_requires_available_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _make_external_geometry_only_root(str((tmp_path / "source.mp4").resolve()))
    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (False, "cuda_unavailable"),
    )

    with pytest.raises(
        RuntimeError,
        match="GPU decode unavailable; refusing CPU fallback.*cuda_unavailable",
    ):
        CropImageSource.open(
            root,
            roi_cache_policy="never",
            roi_live_acceleration="gpu",
        )


def test_crop_image_source_external_geometry_live_gpu_auto_runtime_failure_rejects_cpu_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _make_external_geometry_only_root(str((tmp_path / "source.mp4").resolve()))

    monkeypatch.setattr(
        crop_mod,
        "_check_external_video_live_gpu_available",
        lambda: (True, "available"),
    )

    def _failing_gpu_batch_reader(**_kwargs) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        crop_mod, "_read_external_video_live_gpu_batch", _failing_gpu_batch_reader
    )

    source = CropImageSource.open(
        root,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
    )

    assert source.roi_live_acceleration_requested == "auto"
    assert source.roi_live_acceleration_effective == "gpu"
    with pytest.raises(
        RuntimeError,
        match="GPU decode unavailable; refusing CPU fallback.*RuntimeError: boom",
    ):
        source.read_slice(0, 2)
    source.close()
