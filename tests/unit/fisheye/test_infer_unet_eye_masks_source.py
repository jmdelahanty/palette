from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from fisheye.segmentation import infer_unet_eye_masks as mod


class _CompressionCompatArray:
    def __init__(self) -> None:
        self.compressors = ()
        self.chunk_codecs = None
        self.filters = None

    @property
    def compressor(self):
        raise AssertionError("legacy compressor property should not be accessed when compressors is present")


def test_compression_kwargs_avoids_legacy_compressor_when_zarr_v3_compressors_present() -> None:
    kwargs = mod._compression_kwargs(_CompressionCompatArray())

    assert kwargs == {}


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
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop.attrs["video_source_path"] = "/tmp/source-video.mp4"
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[1, 1], [2, 2]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    crop.create_array("detection_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    crop.create_array("detection_source", data=np.array([0, 0], dtype=np.int8), overwrite=True)
    return zarr_path


def test_infer_unet_eye_masks_supports_geometry_only_crop_runs_with_temporary_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    checkpoint_path = tmp_path / "eye_unet.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {"label_mode": "union", "dataset_meta": [], "best_val_dice": 0.91}

    def _fake_write_mask_probs(
        run_group,
        model,
        roi_source,
        batch_size,
        device,
        label_mode,
        console,
        write_binary,
        mask_probs_chunk_rois=None,
        mask_probs_dtype="uint8",
        timing_profiler=None,
    ) -> tuple[int, bool]:
        del model, batch_size, device, label_mode, console, write_binary
        seen["mask_probs_chunk_rois"] = mask_probs_chunk_rois
        seen["mask_probs_dtype"] = mask_probs_dtype
        batch = roi_source.read_slice(0, roi_source.total_rois)
        seen["batch_shape"] = batch.shape
        seen["roi_read_mode"] = roi_source.roi_read_mode
        seen["roi_cache_used"] = roi_source.roi_cache_used
        seen["roi_cache_path"] = roi_source.roi_cache_path
        if timing_profiler is not None:
            timing_profiler.record("roi_read", 0.01, items=roi_source.total_rois)
        run_group.create_array(
            "mask_probs_roi",
            data=np.zeros((roi_source.total_rois, 1, roi_source.roi_shape[0], roi_source.roi_shape[1]), dtype=np.float16),
            overwrite=True,
        )
        return 1, False

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(mod, "_write_mask_probs", _fake_write_mask_probs)
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.0",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {"name": "test-env"},
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )

    mod.main(
        [
            str(zarr_path),
            "--checkpoint",
            str(checkpoint_path),
            "--roi-cache-policy",
            "always",
            "--roi-cache-dir",
            str(tmp_path / "roi-cache"),
            "--mask-probs-chunk-rois",
            "2",
            "--profile-timings",
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    eye_parent = root["eye_masks_runs"]
    latest = str(eye_parent.attrs["latest"])
    run_group = eye_parent[latest]

    assert seen["batch_shape"] == (2, 4, 4)
    assert seen["roi_read_mode"] == "temporary_cache"
    assert seen["roi_cache_used"] is True
    assert Path(str(seen["roi_cache_path"])).exists()
    assert seen["mask_probs_chunk_rois"] == 2
    assert seen["mask_probs_dtype"] == "uint8"

    assert run_group.attrs["source_crop_run"] == "crop_geometry"
    assert run_group.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run_group.attrs["source_roi_read_mode"] == "temporary_cache"
    assert run_group.attrs["roi_cache_policy"] == "always"
    assert run_group.attrs["source_roi_cache_used"] is True
    assert run_group.attrs["source_roi_cache_path"]
    assert run_group.attrs["mask_probs_chunk_rois"] == 2
    assert run_group.attrs["profile_timings_enabled"] is True
    timing_profile = run_group.attrs["timing_profile"]
    assert timing_profile["enabled"] is True
    assert "roi_read" in timing_profile["stages"]
    assert run_group["mask_probs_roi"].shape == (2, 1, 4, 4)


def test_infer_unet_eye_masks_defaults_mask_probs_chunk_rois_to_32(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_geometry_only_archive(tmp_path)
    checkpoint_path = tmp_path / "eye_unet.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {"label_mode": "union", "dataset_meta": [], "best_val_dice": 0.91}

    def _fake_write_mask_probs(
        run_group,
        model,
        roi_source,
        batch_size,
        device,
        label_mode,
        console,
        write_binary,
        mask_probs_chunk_rois=None,
        mask_probs_dtype="uint8",
        timing_profiler=None,
    ) -> tuple[int, bool]:
        del model, roi_source, batch_size, device, label_mode, console, write_binary, timing_profiler
        seen["mask_probs_chunk_rois"] = mask_probs_chunk_rois
        seen["mask_probs_dtype"] = mask_probs_dtype
        run_group.create_array(
            "mask_probs_roi",
            data=np.zeros((2, 1, 4, 4), dtype=np.float16),
            overwrite=True,
        )
        return 1, False

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(mod, "_write_mask_probs", _fake_write_mask_probs)
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.0",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {"name": "test-env"},
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )

    mod.main(
        [
            str(zarr_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    eye_parent = root["eye_masks_runs"]
    latest = str(eye_parent.attrs["latest"])
    run_group = eye_parent[latest]

    assert seen["mask_probs_chunk_rois"] == 32
    assert seen["mask_probs_dtype"] == "uint8"
    assert run_group.attrs["mask_probs_chunk_rois"] == 2


def test_probabilities_from_logits_returns_float16_cpu_numpy() -> None:
    logits = torch.tensor(
        [[[[0.0, 100.0], [-100.0, float("inf")]]]],
        dtype=torch.float32,
    )

    probs = mod._probabilities_from_logits(logits)

    assert probs.dtype == np.float16
    assert probs.shape == (1, 1, 2, 2)
    assert probs[0, 0, 0, 0] == np.float16(0.5)
    assert probs[0, 0, 0, 1] == np.float16(1.0)
    assert probs[0, 0, 1, 0] == np.float16(0.0)
    assert probs[0, 0, 1, 1] == np.float16(1.0)


def test_probabilities_from_logits_uint8_quantizes_on_gpu_path() -> None:
    logits = torch.tensor(
        [[[[0.0, 100.0], [-100.0, float("inf")]]]],
        dtype=torch.float32,
    )

    probs = mod._probabilities_from_logits(logits, mask_probs_dtype="uint8")

    assert probs.dtype == np.uint8
    assert probs.shape == (1, 1, 2, 2)
    assert probs[0, 0, 0, 0] == np.uint8(128)
    assert probs[0, 0, 0, 1] == np.uint8(255)
    assert probs[0, 0, 1, 0] == np.uint8(0)
    assert probs[0, 0, 1, 1] == np.uint8(255)


def test_normalise_roi_tensor_scales_uint8_batch_to_unit_interval() -> None:
    batch = torch.tensor(
        [
            [[0, 255], [128, 64]],
            [[255, 0], [32, 16]],
        ],
        dtype=torch.uint8,
    )

    norm = mod._normalise_roi_tensor(batch)

    assert norm.dtype == torch.float32
    assert tuple(norm.shape) == (2, 1, 2, 2)
    assert torch.isclose(norm[0, 0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(norm[0, 0, 0, 1], torch.tensor(1.0))
    assert torch.isclose(norm[0, 0, 1, 0], torch.tensor(128.0 / 255.0))


def test_normalise_roi_tensor_rescales_float_batch_above_one() -> None:
    batch = torch.tensor([[[[0.0, 2.0], [4.0, float("nan")]]]], dtype=torch.float32)

    norm = mod._normalise_roi_tensor(batch)

    assert norm.dtype == torch.float32
    assert tuple(norm.shape) == (1, 1, 2, 2)
    assert torch.isclose(norm[0, 0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(norm[0, 0, 0, 1], torch.tensor(0.5))
    assert torch.isclose(norm[0, 0, 1, 0], torch.tensor(1.0))
    assert torch.isclose(norm[0, 0, 1, 1], torch.tensor(0.0))


def test_serialize_probabilities_uint8_quantizes_unit_interval() -> None:
    probs = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float16)

    stored = mod._serialize_probabilities(probs, mask_probs_dtype="uint8")

    assert stored.dtype == np.uint8
    assert stored.tolist() == [[[[0, 128, 255]]]]
