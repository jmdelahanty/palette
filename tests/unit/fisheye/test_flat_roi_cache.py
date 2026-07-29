from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
import fisheye.shared.crop_image_source as crop_source_mod
from fisheye.diagnostics.check_flat_roi_cache_pixel_parity import check_flat_roi_cache_pixel_parity
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


def _make_acquisition_crop_video_archive(
    tmp_path: Path,
) -> tuple[Path, Path, np.ndarray]:
    zarr_path = tmp_path / "recording_acquisition_crop_analysis.zarr"
    crop_video = tmp_path / "acquisition_crop.mp4"
    crop_video.touch()
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_acquisition"
    crop = crop_parent.create_group("crop_acquisition")
    crop.attrs.update(
        {
            "crop_storage_mode": "geometry_only",
            "roi_size": [3, 5],
            "source_pixels": "acquisition_crop_video",
            "roi_pixel_provider": "acquisition_crop_video",
            "source_crop_video_path": str(crop_video),
            "crop_signature": "sig-acquisition-crop-video",
            "crop_revision": "rev-acquisition-crop-video-001",
        }
    )
    crop.create_array(
        "frame_indices",
        data=np.asarray([10, 11], dtype=np.int64),
        overwrite=True,
    )
    crop.create_array(
        "source_crop_video_frame_indices",
        data=np.asarray([0, 1], dtype=np.int64),
        overwrite=True,
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.asarray([[100, 200], [110, 210]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array(
        "roi_sizes_full",
        data=np.asarray([[5, 3], [5, 3]], dtype=np.int32),
        overwrite=True,
    )
    expected = np.stack(
        [
            np.arange(15, dtype=np.uint8).reshape(3, 5),
            np.arange(15, dtype=np.uint8).reshape(3, 5) + np.uint8(40),
        ]
    )
    return zarr_path, crop_video, expected


def _manifest_payload_path(manifest: dict) -> Path:
    manifest_path = Path(str(manifest["manifest_path"]))
    payload = Path(str(manifest["array"]["bin_path"]))
    if payload.is_absolute():
        return payload
    return manifest_path.parent / payload


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
        raise AssertionError("flat ROI cache writers must not retain decode_next() decoder surfaces")

    def iter_frames(self):
        while self._offset < len(self._frames):
            frame = self._frames[self._offset]
            self._offset += 1
            yield frame

    def close(self) -> None:
        pass


class _FakeOrangeMonoPynvvcReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        import torch

        self.source_height = int(frames[0].shape[0])
        self.source_width = int(frames[0].shape[1])
        uv_height = max(1, (self.source_height + 1) // 2)
        self._frames = [
            torch.from_numpy(
                np.vstack(
                    [
                        np.asarray(frame, dtype=np.uint8),
                        np.full((uv_height, self.source_width), 128, dtype=np.uint8),
                    ]
                )
            )
            for frame in frames
        ]
        self._offset = 0

    def iter_frames(self):
        while self._offset < len(self._frames):
            frame = self._frames[self._offset]
            self._offset += 1
            yield frame

    def close(self) -> None:
        pass


def _make_orange_full_range_contract_crop_archive(tmp_path: Path) -> tuple[Path, list[np.ndarray], np.ndarray]:
    zarr_path = tmp_path / "recording_orange_contract_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 8
    root.attrs["height"] = 4
    root.attrs["source_video_path"] = str(tmp_path / "orange_full_range_tv_flag.mp4")
    root.attrs["format_comment"] = (
        "source_pixel_contract=orange.camera.mono8.full_frame.v1; "
        "source_pixel_range=0_255"
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geom"
    crop = crop_parent.create_group("crop_geom")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["roi_size"] = [2, 4]
    crop.attrs["width"] = 8
    crop.attrs["height"] = 4
    crop.attrs["source_video_path"] = str(tmp_path / "orange_full_range_tv_flag.mp4")
    crop.attrs["source_pixels"] = "raw_camera_video"
    crop.attrs["source_pixel_contract"] = "orange.camera.mono8.full_frame.v1"
    crop.attrs["source_pixel_range"] = "0_255"
    crop.attrs["container_color_range_observed"] = "tv"
    crop.attrs["crop_signature"] = "sig-orange-full-range-contract"
    crop.attrs["crop_revision"] = "rev-orange-full-range-contract-001"
    crop.create_array("frame_indices", data=np.array([0, 0, 1, 1], dtype=np.int64), overwrite=True)
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [4, 0], [0, 2], [4, 2]], dtype=np.int32),
        overwrite=True,
    )

    frame0 = np.array(
        [
            [0, 1, 15, 16, 17, 127, 235, 236],
            [237, 238, 254, 255, 0, 16, 235, 255],
            [5, 16, 32, 64, 96, 128, 224, 235],
            [255, 240, 235, 16, 15, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    frame1 = np.array(
        [
            [255, 254, 240, 239, 238, 128, 20, 19],
            [18, 17, 1, 0, 255, 239, 20, 0],
            [250, 239, 223, 191, 159, 127, 31, 20],
            [0, 15, 20, 239, 240, 253, 254, 255],
        ],
        dtype=np.uint8,
    )
    frames = [frame0, frame1]
    expected = np.stack(
        [
            frame0[0:2, 0:4],
            frame0[0:2, 4:8],
            frame1[2:4, 0:4],
            frame1[2:4, 4:8],
        ],
        axis=0,
    )
    return zarr_path, frames, expected


def _write_tv_flagged_full_range_yuv420_video(tmp_path: Path, frames: list[np.ndarray]) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to synthesize the tv-flagged full-range video")
    if not frames:
        raise ValueError("frames must not be empty")
    height, width = frames[0].shape
    if height % 2 or width % 2:
        raise ValueError("yuv420p synthesis requires even frame dimensions")

    yuv_path = tmp_path / "orange_full_range_tv_flag.yuv"
    with yuv_path.open("wb") as handle:
        for frame in frames:
            y = np.asarray(frame, dtype=np.uint8)
            if y.shape != (height, width):
                raise ValueError("all frames must share shape")
            u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
            v = np.full((height // 2, width // 2), 128, dtype=np.uint8)
            handle.write(y.tobytes(order="C"))
            handle.write(u.tobytes(order="C"))
            handle.write(v.tobytes(order="C"))

    video_path = tmp_path / "orange_full_range_tv_flag.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s:v",
        f"{width}x{height}",
        "-r",
        "1",
        "-i",
        str(yuv_path),
        "-frames:v",
        str(len(frames)),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        str(video_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg could not synthesize lossless h264/yuv420p test video: {completed.stderr.strip()}")
    return video_path


def test_write_owned_roi_payload_batch_fast_path_writes_contiguous_rows(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.bin"
    row_stride = 4
    crops = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)

    with payload_path.open("w+b") as handle:
        handle.truncate(6 * row_stride)
        elapsed = flat_cache_mod._write_owned_roi_payload_batch(
            handle,
            row_stride,
            np.array([2, 3, 4], dtype=np.int64),
            crops,
        )

    payload = np.fromfile(payload_path, dtype=np.uint8).reshape(6, 2, 2)
    expected = np.zeros((6, 2, 2), dtype=np.uint8)
    expected[2:5] = crops
    assert elapsed >= 0
    np.testing.assert_array_equal(payload, expected)


def test_write_owned_roi_payload_batch_sorts_sparse_rows(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.bin"
    row_stride = 4
    crops = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)

    with payload_path.open("w+b") as handle:
        handle.truncate(6 * row_stride)
        flat_cache_mod._write_owned_roi_payload_batch(
            handle,
            row_stride,
            np.array([4, 2, 3], dtype=np.int64),
            crops,
        )

    payload = np.fromfile(payload_path, dtype=np.uint8).reshape(6, 2, 2)
    expected = np.zeros((6, 2, 2), dtype=np.uint8)
    expected[4] = crops[0]
    expected[2] = crops[1]
    expected[3] = crops[2]
    np.testing.assert_array_equal(payload, expected)


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
    assert manifest["builder"]["pixel_contract"]["name"] == "crop_image_source_uint8_grayscale"
    assert manifest["builder"]["pixel_contract_name"] == "crop_image_source_uint8_grayscale"

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
        assert source.roi_image_representation == "uint8_grayscale_roi_v1"
        assert source.roi_pixel_contract is not None
        assert source.roi_pixel_contract["name"] == "crop_image_source_uint8_grayscale"
        np.testing.assert_array_equal(source.read_indices([4, 0, 2]), roi_images[[4, 0, 2]])
    finally:
        source.close()


def test_flat_roi_cache_pixel_parity_passes_for_matching_cache(tmp_path: Path) -> None:
    zarr_path, _roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=2,
    )

    report = check_flat_roi_cache_pixel_parity(
        zarr_path=zarr_path,
        roi_cache_manifest=manifest["manifest_path"],
        rows=[0, 2, 4],
    )

    assert report["status"] == "ok"
    assert report["diff"]["byte_equal"] is True
    assert report["diff"]["max_abs_diff"] == 0
    assert report["source"]["cache_manifest_builder"]["pixel_contract"]["name"] == (
        "crop_image_source_uint8_grayscale"
    )


def test_flat_roi_cache_pixel_parity_fails_for_mismatched_cache(tmp_path: Path) -> None:
    zarr_path, _roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=2,
    )
    payload_path = _manifest_payload_path(manifest)
    with payload_path.open("r+b") as handle:
        handle.seek(0)
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([(first[0] + 1) % 256]))

    report = check_flat_roi_cache_pixel_parity(
        zarr_path=zarr_path,
        roi_cache_manifest=manifest["manifest_path"],
        rows=[0, 2, 4],
    )

    assert report["status"] == "fail"
    assert report["diff"]["byte_equal"] is False
    assert report["diff"]["max_abs_diff"] == 1
    assert report["diff"]["mismatched_rows"] == 1


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
    assert manifest["builder"]["pixel_contract"]["name"] == "nv12_luma_plane_uint8"
    assert manifest["builder"]["pixel_contract_name"] == "nv12_luma_plane_uint8"
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


def test_build_flat_roi_cache_accepts_current_acquisition_crop_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path, crop_video, expected = _make_acquisition_crop_video_archive(tmp_path)

    class _FakeAcquisitionReader:
        source_height = 3
        source_width = 5

        def __init__(
            self,
            video_path: Path,
            *,
            start_frame: int = 0,
            gpu_id: int = 0,
        ) -> None:
            assert Path(video_path) == crop_video
            assert start_frame == 0
            assert gpu_id == 0

        def iter_frames(self):
            yield from expected

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        crop_source_mod,
        "PynvvcLumaRgbReader",
        _FakeAcquisitionReader,
    )

    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache-acquisition",
        batch_size=1,
        roi_decode_backend="pynvvc_luma",
    )

    assert manifest["builder"]["decode_backend_effective"] == "pynvvc_luma"
    assert manifest["builder"]["pixel_contract"]["name"] == (
        "orange_mono_pynvvc_luma_uint8_v1"
    )
    assert manifest["builder"]["pixel_contract"]["source_pixels"] == (
        "acquisition_crop_video"
    )
    cache = open_flat_roi_cache(
        manifest["manifest_path"],
        expected_shape=expected.shape,
    )
    try:
        np.testing.assert_array_equal(cache[:], expected)
    finally:
        cache.close()


def test_build_flat_roi_cache_pynvvc_luma_preserves_orange_full_range_y_plane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path, frames, expected = _make_orange_full_range_contract_crop_archive(tmp_path)

    monkeypatch.setattr(
        flat_cache_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakeOrangeMonoPynvvcReader(frames),
    )

    manifest_a = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache-a",
        batch_size=2,
        roi_decode_backend="pynvvc_luma",
        compute_sha256=True,
    )
    manifest_b = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache-b",
        batch_size=3,
        roi_decode_backend="pynvvc_luma",
        compute_sha256=True,
    )

    cache_a = open_flat_roi_cache(manifest_a["manifest_path"], expected_shape=expected.shape)
    cache_b = open_flat_roi_cache(manifest_b["manifest_path"], expected_shape=expected.shape)
    try:
        payload_a = np.asarray(cache_a[:], dtype=np.uint8).copy()
        payload_b = np.asarray(cache_b[:], dtype=np.uint8).copy()
    finally:
        cache_a.close()
        cache_b.close()

    np.testing.assert_array_equal(payload_a, expected)
    np.testing.assert_array_equal(payload_b, expected)
    np.testing.assert_array_equal(payload_b, payload_a)
    assert set(np.unique(payload_a).tolist()) >= {0, 16, 235, 255}
    limited_expanded = np.clip((expected.astype(np.float32) - 16.0) * (255.0 / 219.0), 0.0, 255.0)
    limited_expanded = limited_expanded.round().astype(np.uint8)
    assert not np.array_equal(payload_a, limited_expanded)
    assert manifest_a["array"]["sha256"] == manifest_b["array"]["sha256"]
    assert manifest_a["builder"]["decode_backend_effective"] == "pynvvc_luma"
    assert manifest_a["builder"]["pixel_contract"]["name"] == "nv12_luma_plane_uint8"


@pytest.mark.gpu
def test_pynvvc_luma_reader_decodes_tv_flagged_full_range_y_plane_exactly(tmp_path: Path) -> None:
    _zarr_path, frames, _expected = _make_orange_full_range_contract_crop_archive(tmp_path)
    video_path = _write_tv_flagged_full_range_yuv420_video(tmp_path, frames)
    height, width = frames[0].shape

    try:
        from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

        reader = PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)
    except Exception as exc:
        pytest.skip(f"PyNvVideoCodec GPU reader is unavailable: {exc}")

    decoded: list[np.ndarray] = []
    try:
        for frame in reader.iter_frames():
            decoded.append(np.asarray(frame[:height, :width].cpu().numpy(), dtype=np.uint8).copy())
            if len(decoded) == len(frames):
                break
    finally:
        reader.close()

    assert len(decoded) == len(frames)
    for decoded_frame, expected_frame in zip(decoded, frames):
        np.testing.assert_array_equal(decoded_frame, expected_frame)


def test_build_flat_roi_cache_auto_prefers_pynvvc_luma_for_geometry_only(
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

    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=2,
        roi_decode_backend="auto",
        roi_live_acceleration="cpu",
    )

    assert manifest["builder"]["decode_backend_effective"] == "pynvvc_luma"
    assert manifest["builder"]["pixel_contract"]["name"] == "nv12_luma_plane_uint8"
    assert manifest["builder"]["pixel_contract_name"] == "nv12_luma_plane_uint8"
    cache = open_flat_roi_cache(manifest["manifest_path"], expected_shape=expected.shape)
    try:
        np.testing.assert_array_equal(cache[:], expected)
    finally:
        cache.close()


def test_build_flat_roi_cache_auto_refuses_read_slice_fallback_for_geometry_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path, _expected = _make_geometry_only_crop_archive(tmp_path)

    def _raise_no_cuda(_video_path: Path) -> object:
        raise RuntimeError("CUDA not available")

    monkeypatch.setattr(flat_cache_mod, "_open_pynvvc_luma_reader", _raise_no_cuda)

    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        build_flat_roi_cache(
            zarr_path=zarr_path,
            output_dir=tmp_path / "cache",
            batch_size=2,
            roi_decode_backend="auto",
        )
