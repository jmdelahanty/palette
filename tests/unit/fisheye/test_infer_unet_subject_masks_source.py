from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.segmentation import infer_unet_subject_masks as mod


class _FakeArray:
    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks or ((self.shape[0],) if self.shape else (1,))
        self.compressors = ()
        self.chunk_codecs = None
        self.filters = None

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def require_group(self, name: str) -> "_FakeGroup":
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(
        self,
        name: str,
        *,
        data: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype=None,
        chunks: tuple[int, ...] | None = None,
        fill_value=None,
        overwrite: bool = False,
        **_kwargs,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape required when data is omitted")
            arr = np.full(shape, 0 if fill_value is None else fill_value, dtype=dtype or np.float32)
        else:
            arr = np.asarray(data, dtype=dtype)
        fake = _FakeArray(arr, chunks=chunks)
        self._children[name] = fake
        return fake

    def get(self, name: str, default=None):
        return self._children.get(name, default)

    def __getitem__(self, name: str):
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]


def _build_fake_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["width"] = 6
    root.attrs["height"] = 6
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_geometry")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = "sig-001"
    crop.attrs["crop_revision"] = 4
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_001/review_status"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop.attrs["video_source_path"] = "/tmp/source-video.mp4"
    crop.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    crop.create_array("detection_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    crop.create_array("detection_source", data=np.array([0, 0], dtype=np.int8), overwrite=True)
    return root


def test_infer_unet_subject_masks_supports_geometry_only_crop_runs_with_temporary_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "subject_unet.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}
    fake_root = _build_fake_root()
    cache_dir = tmp_path / "roi-cache" / "fake-cache.zarr"
    cache_dir.mkdir(parents=True, exist_ok=True)

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = None
            self.storage_mode = "geometry_only"
            self.roi_read_mode = "temporary_cache"
            self.roi_cache_policy = "always"
            self.roi_cache_used = True
            self.roi_cache_key = "cache-key-001"
            self.roi_cache_path = str(cache_dir)
            self.roi_live_acceleration_requested = "cpu"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 11
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "best_val_dice": 0.88,
        }

    def _fake_write_subject_mask_outputs(
        run_group,
        model,
        roi_source,
        *,
        batch_size,
        device,
        mask_labels,
        mask_probs_chunk_rois,
        mask_probs_dtype,
        console,
        timing_profiler,
    ) -> float:
        del model, batch_size, device, console
        seen["mask_labels"] = tuple(mask_labels)
        seen["mask_probs_chunk_rois"] = mask_probs_chunk_rois
        seen["mask_probs_dtype"] = mask_probs_dtype
        batch = roi_source.read_slice(0, roi_source.total_rois)
        seen["batch_shape"] = batch.shape
        seen["roi_read_mode"] = roi_source.roi_read_mode
        seen["roi_cache_used"] = roi_source.roi_cache_used
        seen["roi_cache_path"] = roi_source.roi_cache_path
        run_group.create_array(
            "masks_roi",
            data=np.zeros((roi_source.total_rois, 3, roi_source.roi_shape[0], roi_source.roi_shape[1]), dtype=np.uint8),
            overwrite=True,
        )
        run_group.create_array(
            "mask_probs_roi",
            data=np.zeros((roi_source.total_rois, 3, roi_source.roi_shape[0], roi_source.roi_shape[1]), dtype=np.float16),
            overwrite=True,
        )
        run_group.create_array("available_channels", data=np.array([True, True, True], dtype=np.bool_), overwrite=True)
        metrics = run_group.require_group("metrics")
        metrics.create_array("prob_max", data=np.zeros((roi_source.total_rois, 3), dtype=np.float32), overwrite=True)
        metrics.create_array("mask_present", data=np.zeros((roi_source.total_rois, 3), dtype=np.bool_), overwrite=True)
        metrics.create_array("area_px", data=np.zeros((roi_source.total_rois, 3), dtype=np.float32), overwrite=True)
        metrics.create_array("centroid_xy", data=np.zeros((roi_source.total_rois, 3, 2), dtype=np.float32), overwrite=True)
        metrics.create_array("centroid_valid", data=np.zeros((roi_source.total_rois, 3), dtype=np.bool_), overwrite=True)
        metrics.create_array("bbox_xyxy", data=np.zeros((roi_source.total_rois, 3, 4), dtype=np.float32), overwrite=True)
        metrics.create_array("bbox_valid", data=np.zeros((roi_source.total_rois, 3), dtype=np.bool_), overwrite=True)
        if timing_profiler is not None:
            timing_profiler.record("roi_read", 0.01, items=roi_source.total_rois)
        return 0.25

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(mod, "_write_subject_mask_outputs", _fake_write_subject_mask_outputs)
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda root, **_kwargs: _FakeCropSource(root["crop_runs"]["crop_geometry"]),
    )
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
            "--crop-run",
            "crop_geometry",
            "--roi-cache-policy",
            "always",
            "--roi-cache-dir",
            str(tmp_path / "roi-cache"),
            "--roi-live-acceleration",
            "cpu",
            "--roi-live-gpu-chunk-frames",
            "11",
            "--mask-probs-chunk-rois",
            "2",
            "--profile-timings",
        ]
    )

    subject_parent = fake_root["subject_mask_runs"]
    latest = str(subject_parent.attrs["latest"])
    run_group = subject_parent[latest]

    assert seen["batch_shape"] == (2, 4, 4)
    assert seen["mask_labels"] == ("subject_body", "eyes_union", "swim_bladder")
    assert seen["roi_read_mode"] == "temporary_cache"
    assert seen["roi_cache_used"] is True
    assert Path(str(seen["roi_cache_path"])).exists()
    assert seen["mask_probs_chunk_rois"] == 2
    assert seen["mask_probs_dtype"] == "uint8"

    assert run_group.attrs["source_crop_run"] == "crop_geometry"
    assert run_group.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run_group.attrs["source_crop_signature"] == "sig-001"
    assert run_group.attrs["source_crop_revision"] == 4
    assert run_group.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_001/review_status"
    assert run_group.attrs["source_roi_read_mode"] == "temporary_cache"
    assert run_group.attrs["roi_cache_policy"] == "always"
    assert run_group.attrs["source_roi_cache_used"] is True
    assert run_group.attrs["source_roi_cache_path"]
    assert run_group.attrs["source_roi_live_acceleration_requested"] == "cpu"
    assert run_group.attrs["source_roi_live_acceleration_effective"] == "cpu"
    assert run_group.attrs["source_roi_live_gpu_chunk_frames"] == 11
    assert run_group.attrs["label_schema_id"] == "subject_v1_union"
    assert list(run_group.attrs["mask_labels"]) == ["subject_body", "eyes_union", "swim_bladder"]
    assert run_group.attrs["method"] == "unet_subject_mask_segmenter"
    assert run_group.attrs["run_semantics"] == "unet_subject_mask_inference"
    assert run_group.attrs["profile_timings_enabled"] is True
    provenance_inputs = run_group.attrs["provenance"]["inputs"]
    assert provenance_inputs["source_crop_signature"] == "sig-001"
    assert provenance_inputs["source_crop_revision"] == 4
    assert provenance_inputs["source_detect_review_status_ref"] == "refined_detect_runs/refined_001/review_status"
    assert run_group["mask_probs_roi"].shape == (2, 3, 4, 4)
    assert run_group["masks_roi"].shape == (2, 3, 4, 4)


def test_infer_unet_subject_masks_writes_lr_checkpoint_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    checkpoint_path = tmp_path / "subject_unet_lr.pt"
    checkpoint_path.write_text("", encoding="utf-8")

    seen: dict[str, object] = {}
    fake_root = _build_fake_root()

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.crop_run_name = "crop_geometry"
            self.total_rois = 2
            self.roi_shape = (4, 4)
            self.roi_array = None
            self.storage_mode = "geometry_only"
            self.roi_read_mode = "live"
            self.roi_cache_policy = "never"
            self.roi_cache_used = False
            self.roi_cache_key = None
            self.roi_cache_path = None
            self.roi_live_acceleration_requested = "cpu"
            self.roi_live_acceleration_effective = "cpu"
            self.roi_live_acceleration_fallback_reason = None
            self.roi_live_gpu_chunk_frames = 32
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((stop - start, 4, 4), dtype=np.uint8)

        def close(self) -> None:
            return None

    def _fake_load_checkpoint(_path: Path, _device) -> tuple[object, dict[str, object]]:
        return object(), {
            "label_schema_id": "subject_v1_lr",
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "best_val_dice": 0.91,
        }

    def _fake_write_subject_mask_outputs(
        run_group,
        model,
        roi_source,
        *,
        batch_size,
        device,
        mask_labels,
        mask_probs_chunk_rois,
        mask_probs_dtype,
        console,
        timing_profiler,
    ) -> float:
        del model, batch_size, device, mask_probs_chunk_rois, mask_probs_dtype, console, timing_profiler
        seen["mask_labels"] = tuple(mask_labels)
        channel_count = len(mask_labels)
        run_group.create_array(
            "masks_roi",
            data=np.zeros(
                (roi_source.total_rois, channel_count, roi_source.roi_shape[0], roi_source.roi_shape[1]),
                dtype=np.uint8,
            ),
            overwrite=True,
        )
        run_group.create_array(
            "mask_probs_roi",
            data=np.zeros(
                (roi_source.total_rois, channel_count, roi_source.roi_shape[0], roi_source.roi_shape[1]),
                dtype=np.float16,
            ),
            overwrite=True,
        )
        run_group.create_array(
            "available_channels",
            data=np.ones((channel_count,), dtype=np.bool_),
            overwrite=True,
        )
        return 0.25

    monkeypatch.setattr(mod, "_load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(mod, "_write_subject_mask_outputs", _fake_write_subject_mask_outputs)
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(
        mod.CropImageSource,
        "open",
        lambda root, **_kwargs: _FakeCropSource(root["crop_runs"]["crop_geometry"]),
    )
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

    mod.main([str(zarr_path), "--checkpoint", str(checkpoint_path), "--crop-run", "crop_geometry"])

    subject_parent = fake_root["subject_mask_runs"]
    latest = str(subject_parent.attrs["latest"])
    run_group = subject_parent[latest]

    assert seen["mask_labels"] == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert run_group.attrs["label_schema_id"] == "subject_v1_lr"
    assert list(run_group.attrs["mask_labels"]) == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    assert run_group["mask_probs_roi"].shape == (2, 4, 4, 4)
    assert run_group["masks_roi"].shape == (2, 4, 4, 4)
    for label in ("subject_body", "eye_left", "eye_right", "swim_bladder"):
        provenance = run_group["components"][label]["provenance"].attrs
        assert provenance["source_label_schema_id"] == "subject_v1_lr"
        assert provenance["source_channels"] == [label]
