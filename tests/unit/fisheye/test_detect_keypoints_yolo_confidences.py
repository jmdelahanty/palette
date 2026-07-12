from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import torch
import zarr

from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.detection.detect_keypoints_yolo import (
    _create_output_arrays,
    _extract_keypoint_confidences,
    _extract_pose_bbox_xyxy_roi,
    _prepare_model_inputs,
    detect_keypoints_yolo,
)
from fisheye.shared.run_provenance import RUN_PROVENANCE_ATTR, build_writer_run_provenance
from fisheye.shared.model_input_transform import resolve_model_input_transform


class _KeypointsWithConf:
    def __init__(self, conf: torch.Tensor | None) -> None:
        self.conf = conf


class _BoxesWithXyxy:
    def __init__(self, xyxy: torch.Tensor | None) -> None:
        self.xyxy = xyxy


class _FakeCropSource:
    def __init__(self, crop_group: zarr.Group) -> None:
        self.crop_group = crop_group
        self.crop_run_name = "crop_001"
        self.frame_indices = np.asarray(crop_group["frame_indices"][:], dtype=np.int64)
        self.total_rois = int(self.frame_indices.shape[0])
        if "roi_coordinates_full" in crop_group:
            self.roi_coordinates_full = np.asarray(crop_group["roi_coordinates_full"][:], dtype=np.int32)
        else:
            self.roi_coordinates_full = np.zeros((self.total_rois, 2), dtype=np.int32)
        self.roi_shape = (8, 8)
        self.storage_mode = "zarr"
        self.roi_read_mode = "unit_test"
        self.roi_cache_policy = "never"
        self.roi_cache_used = False
        self.roi_cache_backend = None
        self.roi_cache_key = None
        self.roi_cache_path = None
        self.roi_live_acceleration_requested = "none"
        self.roi_live_acceleration_effective = "none"
        self.roi_live_acceleration_fallback_reason = None
        self.roi_live_gpu_chunk_frames = 1
        self.frame_source_kind = "unit_test"
        self.frame_source_path = None
        self.roi_pixel_contract = None
        self.roi_image_representation = "grayscale_uint8"
        self._images = np.zeros((self.total_rois, 8, 8), dtype=np.uint8)

    def read_slice(self, start: int, end: int) -> np.ndarray:
        return self._images[start:end]

    def close(self) -> None:
        return None


class _FakeBoxes:
    def __init__(self, *, success: bool) -> None:
        if success:
            self.conf = torch.tensor([0.9], dtype=torch.float32)
            self.xyxy = torch.tensor([[1.0, 1.0, 6.0, 6.0]], dtype=torch.float32)
        else:
            self.conf = torch.empty((0,), dtype=torch.float32)
            self.xyxy = torch.empty((0, 4), dtype=torch.float32)


class _FakeKeypoints:
    def __init__(self) -> None:
        points = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
        self.xy = points
        self.conf = torch.full((1, 10), 0.8, dtype=torch.float32)


class _FakeResult:
    def __init__(self, *, success: bool) -> None:
        self.boxes = _FakeBoxes(success=success)
        self.keypoints = _FakeKeypoints() if success else None


class _FakeYOLO:
    def __init__(self, _path: str) -> None:
        self.model = SimpleNamespace(
            names={0: "fish"},
            parameters=lambda: iter([torch.nn.Parameter(torch.zeros((), dtype=torch.float32))]),
        )
        self._results = [
            _FakeResult(success=True),
            _FakeResult(success=True),
            _FakeResult(success=False),
        ]

    def to(self, _device: str) -> "_FakeYOLO":
        return self

    def predict(self, inputs, **_kwargs):
        batch_count = len(inputs) if isinstance(inputs, list) else int(inputs.shape[0])
        out = self._results[:batch_count]
        self._results = self._results[batch_count:]
        return iter(out)


def test_extract_keypoint_confidences_returns_values_when_present() -> None:
    keypoints = _KeypointsWithConf(
        torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.6, 0.7, 0.8],
            ],
            dtype=torch.float32,
        )
    )

    actual = _extract_keypoint_confidences(keypoints, 1, n_keypoints=3)

    np.testing.assert_allclose(actual, np.array([0.6, 0.7, 0.8], dtype=np.float64))


def test_extract_keypoint_confidences_returns_nan_when_missing() -> None:
    keypoints = _KeypointsWithConf(None)

    actual = _extract_keypoint_confidences(keypoints, 0, n_keypoints=3)

    assert actual.shape == (3,)
    assert np.isnan(actual).all()


def test_create_output_arrays_includes_keypoint_confidences(tmp_path) -> None:
    root = zarr.open_group(store=str(tmp_path / "test.zarr"), mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")

    arrays = _create_output_arrays(run, total_rois=10, chunk_hint=4, n_keypoints=5)

    assert "keypoint_confidences" in arrays
    assert arrays["keypoints_roi"].shape == (10, 5, 2)
    assert arrays["keypoint_confidences"].shape == (10, 5)
    assert arrays["keypoint_confidences"].dtype.name == "float64"
    assert "pose_bbox_xyxy_roi" in arrays
    assert arrays["pose_bbox_xyxy_roi"].shape == (10, 4)
    assert arrays["pose_bbox_xyxy_roi"].dtype.name == "float32"


def test_create_output_arrays_can_use_aligned_indexed_shards(tmp_path) -> None:
    root = zarr.open_group(store=str(tmp_path / "sharded.zarr"), mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")

    arrays = _create_output_arrays(
        run,
        total_rois=10,
        chunk_hint=4,
        n_keypoints=5,
        shard_rows=8,
    )

    assert arrays["keypoints_roi"].chunks == (4, 5, 2)
    assert arrays["keypoints_roi"].shards == (8, 5, 2)
    assert arrays["confidence"].chunks == (4,)
    assert arrays["confidence"].shards == (8,)


def test_detect_keypoints_yolo_sizes_n_keypoints_to_run_frame_counts(monkeypatch, tmp_path) -> None:
    zarr_path = tmp_path / "training.zarr"
    root = zarr.open_group(store=str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((12, 16, 16), dtype=np.uint8))
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["source_detect_run"] = "detect_001"
    crop.create_array("frame_counts", data=np.array([1, *([0] * 9), 1, *([0] * 8), 1], dtype=np.int32))
    crop.create_array("frame_indices", data=np.array([0, 10, 19], dtype=np.int32))
    crop.create_array("detection_indices", data=np.arange(3, dtype=np.int32))
    crop_parent.attrs["latest"] = "crop_001"

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"fake")

    def _fake_open(root_group, **_kwargs):
        return _FakeCropSource(root_group["crop_runs"]["crop_001"])

    def _fake_copy_row_lineage(dst, src, *, total_rois, **_kwargs):
        dst.create_array("frame_counts", data=src["frame_counts"][:], overwrite=True)
        dst.create_array("frame_indices", data=src["frame_indices"][:], overwrite=True)
        dst.create_array("detection_indices", data=src["detection_indices"][:], overwrite=True)
        return SimpleNamespace(copied={"frame_counts", "frame_indices", "detection_indices"})

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(yolo_mod.CropImageSource, "open", staticmethod(_fake_open))
    monkeypatch.setattr(yolo_mod, "copy_row_lineage_arrays", _fake_copy_row_lineage)
    monkeypatch.setattr(yolo_mod, "_prepare_refined_roi_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        yolo_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "test",
            "short_hash": "test",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        yolo_mod,
        "get_environment_info",
        lambda **_kwargs: {"platform": {"hostname": "unit-test"}, "environment": {}, "gpu": {}},
    )
    monkeypatch.setattr(yolo_mod, "_emit_keypoint_step_status", lambda **_kwargs: None)

    run_name = detect_keypoints_yolo(
        zarr_path,
        model_path,
        run_provenance=build_writer_run_provenance(
            command="unit-keypoint-writer",
            params={"model_path": model_path},
        ),
        run_name="keypoints_001",
        pose_schema="traditional_v3",
        batch_size=8,
        imgsz=8,
        input_mode="numpy-list",
        keypoint_roi_shard_rows=8,
        keypoint_frame_shard_rows=8,
        registry=None,
    )

    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoints_runs"][run_name]
    assert run.attrs["keypoint_storage_layout"] == "indexed_sharding_v1"
    assert run.attrs["keypoint_storage_policy"] == "default_indexed_sharding_v1"
    actual = run["n_keypoints"][:]
    assert actual.shape == run["frame_counts"].shape == (20,)
    assert actual[0] == 10
    assert actual[10] == 10
    assert actual[19] == 0
    assert int(np.count_nonzero(actual)) == 2
    run_provenance = run.attrs[RUN_PROVENANCE_ATTR]
    assert run_provenance["input_artifacts"] == [
        {
            "role": "keypoint_model",
            "path": str(model_path.resolve()),
            "fingerprint_scheme": "content_v1",
            "sha256": hashlib.sha256(b"fake").hexdigest(),
            "size_bytes": 4,
            "mtime_ns": model_path.stat().st_mtime_ns,
            "source": "computed",
        }
    ]


def _make_keypoint_count_fixture(tmp_path, name: str):
    zarr_path = tmp_path / f"{name}.zarr"
    root = zarr.open_group(store=str(zarr_path), mode="w")
    root.attrs["video_width"] = 16
    root.attrs["video_height"] = 16
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["source_detect_run"] = "detect_001"
    crop.create_array("frame_counts", data=np.array([1, 1, 1], dtype=np.int32))
    crop.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    crop.create_array("detection_indices", data=np.arange(3, dtype=np.int32))
    crop.create_array("roi_coordinates_full", data=np.zeros((3, 2), dtype=np.int32))
    crop_parent.attrs["latest"] = "crop_001"
    return zarr_path


def _patch_keypoint_writer_dependencies(monkeypatch, model_path) -> None:
    def _fake_open(root_group, **_kwargs):
        return _FakeCropSource(root_group["crop_runs"]["crop_001"])

    def _fake_copy_row_lineage(dst, src, *, total_rois, **_kwargs):
        dst.create_array("frame_indices", data=src["frame_indices"][:], overwrite=True)
        dst.create_array("detection_indices", data=src["detection_indices"][:], overwrite=True)
        return SimpleNamespace(copied={"frame_indices", "detection_indices"})

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(yolo_mod.CropImageSource, "open", staticmethod(_fake_open))
    monkeypatch.setattr(yolo_mod, "copy_row_lineage_arrays", _fake_copy_row_lineage)
    monkeypatch.setattr(yolo_mod, "_prepare_refined_roi_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        yolo_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "test",
            "short_hash": "test",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        yolo_mod,
        "get_environment_info",
        lambda **_kwargs: {"platform": {"hostname": "unit-test"}, "environment": {}, "gpu": {}},
    )
    monkeypatch.setattr(yolo_mod, "_emit_keypoint_step_status", lambda **_kwargs: None)
    model_path.write_bytes(b"fake")


def _run_keypoint_count_writer(monkeypatch, tmp_path, *, name: str, legacy_count: bool):
    zarr_path = _make_keypoint_count_fixture(tmp_path, name)
    model_path = tmp_path / f"{name}.pt"
    with monkeypatch.context() as patch:
        _patch_keypoint_writer_dependencies(patch, model_path)
        if legacy_count:
            patch.setattr(yolo_mod, "_resolve_crop_run_frame_count_from_domains", lambda *_args, **_kwargs: None)
        run_name = detect_keypoints_yolo(
            zarr_path,
            model_path,
            run_provenance=build_writer_run_provenance(
                command="unit-keypoint-writer",
                params={"model_path": model_path},
            ),
            run_name="keypoints_001",
            pose_schema="traditional_v3",
            batch_size=8,
            imgsz=8,
            input_mode="numpy-list",
            keypoint_roi_shard_rows=8,
            keypoint_frame_shard_rows=8,
            registry=None,
        )
    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoints_runs"][run_name]
    return {
        "n_rois": np.asarray(run["n_rois"][:]),
        "frame_counts": np.asarray(run["frame_counts"][:]),
    }


def test_detect_keypoints_yolo_can_write_collection_shard_without_canonical_pointer(monkeypatch, tmp_path) -> None:
    zarr_path = _make_keypoint_count_fixture(tmp_path, "shard")
    model_path = tmp_path / "shard.pt"
    emit_calls: list[dict[str, object]] = []
    with monkeypatch.context() as patch:
        _patch_keypoint_writer_dependencies(patch, model_path)
        patch.setattr(yolo_mod, "_emit_keypoint_step_status", lambda **kwargs: emit_calls.append(kwargs))
        run_name = detect_keypoints_yolo(
            zarr_path,
            model_path,
            run_provenance=build_writer_run_provenance(
                command="unit-keypoint-shard-writer",
                params={"model_path": model_path},
            ),
            run_name="keypoint_shard_001",
            output_parent="keypoint_shard_runs",
            pose_schema="traditional_v3",
            batch_size=8,
            imgsz=8,
            input_mode="numpy-list",
            keypoint_roi_shard_rows=8,
            keypoint_frame_shard_rows=8,
            registry=None,
        )

    assert emit_calls == []
    root = zarr.open_group(store=str(zarr_path), mode="r")
    assert run_name == "keypoint_shard_001"
    assert "current_keypoint_group_path" not in root.attrs
    assert "keypoints_runs" not in root or "latest" not in root["keypoints_runs"].attrs

    shard_parent = root["keypoint_shard_runs"]
    assert shard_parent.attrs["latest"] == "keypoint_shard_001"
    assert shard_parent.attrs["latest_complete"] == "keypoint_shard_001"
    run = shard_parent[run_name]
    assert run.attrs["output_parent"] == "keypoint_shard_runs"
    assert run.attrs["run_group_parent"] == "keypoint_shard_runs"
    assert run.attrs["is_collection_shard"] is True
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["source_crop_run"] == "crop_001"
    assert np.asarray(run["keypoints_roi"]).shape == (3, 10, 2)
    assert run["keypoints_roi"].shards == (9, 10, 2)
    assert run.attrs["keypoint_storage_layout"] == "indexed_sharding_v1"
    assert run.attrs["keypoint_storage_policy"] == "default_indexed_sharding_v1"
    assert run.attrs["keypoint_shard_write"]["exact_match"] is True
    assert run.attrs["keypoint_shard_write"]["buffer_count"] == 2


def test_detect_keypoints_yolo_can_opt_out_to_regular_chunks(monkeypatch, tmp_path) -> None:
    zarr_path = _make_keypoint_count_fixture(tmp_path, "regular")
    model_path = tmp_path / "regular.pt"
    with monkeypatch.context() as patch:
        _patch_keypoint_writer_dependencies(patch, model_path)
        run_name = detect_keypoints_yolo(
            zarr_path,
            model_path,
            run_provenance=build_writer_run_provenance(
                command="unit-keypoint-regular-writer",
                params={"model_path": model_path},
            ),
            run_name="keypoints_regular",
            pose_schema="traditional_v3",
            batch_size=8,
            imgsz=8,
            input_mode="numpy-list",
            keypoint_roi_shard_rows=None,
            registry=None,
        )

    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoints_runs"][run_name]
    assert run["keypoints_roi"].shards is None
    assert run.attrs["keypoint_storage_layout"] == "regular_chunks_v1"
    assert run.attrs["keypoint_storage_policy"] == "explicit_regular_chunks_override"


def test_detect_keypoints_yolo_frame_count_writer_matches_legacy_arrays(monkeypatch, tmp_path) -> None:
    resolved = _run_keypoint_count_writer(
        monkeypatch,
        tmp_path,
        name="resolved",
        legacy_count=False,
    )
    legacy = _run_keypoint_count_writer(
        monkeypatch,
        tmp_path,
        name="legacy",
        legacy_count=True,
    )

    for name in ("n_rois", "frame_counts"):
        assert resolved[name].dtype == np.dtype("int32")
        assert legacy[name].dtype == np.dtype("int32")
        assert np.array_equal(resolved[name], legacy[name])
        assert resolved[name].tobytes() == legacy[name].tobytes()


def test_extract_pose_bbox_xyxy_roi_clips_to_roi_bounds() -> None:
    boxes = _BoxesWithXyxy(
        torch.tensor(
            [
                [-2.0, 1.5, 8.2, 12.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=torch.float32,
        )
    )

    actual = _extract_pose_bbox_xyxy_roi(boxes, 0, roi_height=6, roi_width=8)

    np.testing.assert_allclose(actual, np.array([0.0, 1.5, 7.0, 5.0], dtype=np.float32))


def test_extract_pose_bbox_xyxy_roi_returns_nan_when_missing() -> None:
    boxes = _BoxesWithXyxy(None)

    actual = _extract_pose_bbox_xyxy_roi(boxes, 0, roi_height=6, roi_width=8)

    assert actual.shape == (4,)
    assert np.isnan(actual).all()


def test_prepare_model_inputs_tensor_mode_returns_normalized_bchw_tensor() -> None:
    batch = np.full((2, 32, 32), 255, dtype=np.uint8)
    transform = resolve_model_input_transform((32, 32), model_hw=(32, 32))

    actual, mode = _prepare_model_inputs(batch, input_mode="tensor", model_input_transform=transform, device=None)

    assert mode == "tensor"
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2, 3, 32, 32)
    assert actual.dtype == torch.float32
    assert float(actual.max()) == 1.0


def test_prepare_model_inputs_numpy_list_preserves_legacy_rgb_arrays() -> None:
    batch = np.zeros((2, 32, 32), dtype=np.uint8)
    transform = resolve_model_input_transform((32, 32), model_hw=(32, 32))

    actual, mode = _prepare_model_inputs(batch, input_mode="numpy-list", model_input_transform=transform, device=None)

    assert mode == "numpy-list"
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert actual[0].shape == (32, 32, 3)


def test_prepare_model_inputs_tensor_mode_supports_explicit_padding() -> None:
    batch = np.full((2, 32, 32), 255, dtype=np.uint8)
    transform = resolve_model_input_transform((32, 32), mode="pad_to_size", model_hw=(64, 64))

    actual, mode = _prepare_model_inputs(batch, input_mode="tensor", model_input_transform=transform, device=None)

    assert mode == "tensor"
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2, 3, 64, 64)
    assert float(actual[:, :, 16:48, 16:48].min()) == 1.0
    assert float(actual[:, :, :16, :].max()) == 0.0


def test_prepare_model_inputs_auto_uses_tensor_for_padded_model_input() -> None:
    batch = np.zeros((2, 32, 32), dtype=np.uint8)
    transform = resolve_model_input_transform((32, 32), mode="auto", model_hw=(64, 64))

    actual, mode = _prepare_model_inputs(batch, input_mode="auto", model_input_transform=transform, device=None)

    assert mode == "tensor"
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (2, 3, 64, 64)
