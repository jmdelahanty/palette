from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.detection.detect_keypoints_yolo import (
    _AlignedKeypointShardWriter,
    _create_output_arrays,
    _extract_keypoint_confidences,
    _extract_pose_bbox_xyxy_roi,
    _prepare_model_inputs,
    _require_model_result_coordinate_contract,
    _require_prepared_model_input_contract,
    detect_keypoints_yolo,
)
from fisheye.shared.run_provenance import RUN_PROVENANCE_ATTR, build_writer_run_provenance
from fisheye.shared.immutable_yolo_storage import IMMUTABLE_YOLO_STORAGE_ATTR
from fisheye.shared.keypoint_coordinate_publication import (
    load_persisted_keypoint_coordinate_surfaces,
)
from fisheye.shared.model_input_transform import resolve_model_input_transform
from tests.unit.fisheye.test_keypoint_coordinate_publication import (
    _real_canonical_archive,
)


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
        self.orig_shape = (8, 8)


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


class _CanonicalFakeYOLO(_FakeYOLO):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        for result in self._results:
            result.orig_shape = (40, 40)


class _SecondBatchWrongShapeCanonicalFakeYOLO(_CanonicalFakeYOLO):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._results[1].orig_shape = (39, 40)


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


def test_aligned_keypoint_shard_writer_finish_and_abort_are_idempotent(tmp_path) -> None:
    root = zarr.open_group(store=str(tmp_path / "writer.zarr"), mode="w")
    destination = root.create_array(
        "values",
        shape=(2,),
        chunks=(1,),
        dtype="i4",
        fill_value=-1,
    )
    writer = _AlignedKeypointShardWriter(
        {"values": destination},
        shard_rows=1,
    )
    writer.write(0, {"values": np.asarray([10, 20], dtype="i4")})

    first = writer.finish()
    second = writer.finish()
    writer.abort()

    assert first == second
    assert writer.is_quiescent
    assert not writer._worker.is_alive()
    np.testing.assert_array_equal(destination[:], np.asarray([10, 20], dtype="i4"))


def test_aligned_keypoint_shard_writer_joins_with_preexisting_worker_error(
    tmp_path,
) -> None:
    root = zarr.open_group(store=str(tmp_path / "writer-error.zarr"), mode="w")
    destination = root.create_array(
        "values",
        shape=(1,),
        chunks=(1,),
        dtype="i4",
        fill_value=-1,
    )
    writer = _AlignedKeypointShardWriter(
        {"values": destination},
        shard_rows=1,
    )
    writer._errors.append(RuntimeError("synthetic worker error"))

    with pytest.raises(RuntimeError, match="shard writer failed"):
        writer.finish()
    writer.abort()

    assert writer.is_quiescent
    assert not writer._worker.is_alive()


def test_failure_boundary_does_not_publish_failed_until_writer_is_quiescent(
    tmp_path,
) -> None:
    root = zarr.open_group(store=str(tmp_path / "nonquiescent.zarr"), mode="w")
    parent = root.create_group("keypoints_runs")
    parent.attrs.update(
        {
            "latest": "prior",
            "latest_complete": "prior",
            "latest_pending": "prior_pending",
        }
    )
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/prior"
    boundary = yolo_mod._KeypointAttemptFailureBoundary()
    boundary.prepare(root=root, parent=parent)
    run = parent.create_group("attempt")
    boundary.bind_run(run, "attempt")
    yolo_mod.mark_run_started(run, run_name="attempt", stage="keypoints")
    yolo_mod.note_pending_latest(parent, "attempt")

    class _NonQuiescentWriter:
        is_quiescent = False

        @staticmethod
        def abort() -> None:
            raise RuntimeError("synthetic inability to join")

    boundary.bind_shard_writer(_NonQuiescentWriter())

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        boundary.fail(ValueError("synthetic inference failure"))

    assert run.attrs["palette_run_completion_status"] == "running"
    assert "palette_run_failed_at_utc" not in run.attrs
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest_pending"] == "prior_pending"
    assert root.attrs["current_keypoint_group_path"] == "keypoints_runs/prior"


def test_keypoint_attempt_keyboard_interrupt_quiesces_and_restores_selectors(
    tmp_path,
) -> None:
    root = zarr.open_group(store=str(tmp_path / "interrupt.zarr"), mode="w")
    parent = root.create_group("keypoints_runs")
    parent.attrs.update(
        {
            "latest": "prior",
            "latest_complete": "prior",
            "latest_pending": "prior_pending",
        }
    )
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/prior"
    writer_state = {"aborted": False}
    source_state = {"closed": False}

    class _InterruptWriter:
        @property
        def is_quiescent(self) -> bool:
            return writer_state["aborted"]

        @staticmethod
        def abort() -> None:
            writer_state["aborted"] = True

    class _InterruptSource:
        @staticmethod
        def close() -> None:
            source_state["closed"] = True

    @yolo_mod._fail_closed_keypoint_attempt
    def _interrupted_attempt() -> None:
        boundary = yolo_mod._ACTIVE_KEYPOINT_ATTEMPT.get()
        assert boundary is not None
        boundary.prepare(root=root, parent=parent)
        run = parent.create_group("attempt")
        boundary.bind_run(run, "attempt")
        yolo_mod.mark_run_started(run, run_name="attempt", stage="keypoints")
        yolo_mod.note_pending_latest(parent, "attempt")
        root.attrs["current_keypoint_group_path"] = "keypoints_runs/attempt"
        boundary.bind_shard_writer(_InterruptWriter())
        boundary.bind_crop_source(_InterruptSource())
        raise KeyboardInterrupt("synthetic interrupt")

    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        _interrupted_attempt()

    run = parent["attempt"]
    assert writer_state["aborted"] is True
    assert source_state["closed"] is True
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert parent.attrs["latest"] == "prior"
    assert parent.attrs["latest_complete"] == "prior"
    assert parent.attrs["latest_pending"] == "prior_pending"
    assert root.attrs["current_keypoint_group_path"] == "keypoints_runs/prior"


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
    crop.create_array("instance_key", data=np.arange(101, 104, dtype=np.uint64))
    crop_parent.attrs["latest"] = "crop_001"

    model_path = tmp_path / "pose.pt"
    model_path.write_bytes(b"fake")

    def _fake_open(root_group, **_kwargs):
        return _FakeCropSource(root_group["crop_runs"]["crop_001"])

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(yolo_mod.CropImageSource, "open", staticmethod(_fake_open))
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
        run_name="keypoint_shard_001",
        output_parent="keypoint_shard_runs",
        pose_schema="traditional_v3",
        batch_size=8,
        imgsz=8,
        input_mode="numpy-list",
        coordinate_contract_mode="legacy_noncanonical",
        keypoint_roi_shard_rows=8,
        keypoint_frame_shard_rows=8,
        registry=None,
    )

    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoint_shard_runs"][run_name]
    assert run.attrs["keypoint_storage_layout"] == "indexed_sharding_v1"
    assert run.attrs["keypoint_storage_policy"] == "default_indexed_sharding_v1"
    assert run.attrs["coordinate_contract"] == "legacy_noncanonical_explicit_v1"
    assert run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]["status"] == "ok"
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
    crop.create_array("instance_key", data=np.arange(101, 104, dtype=np.uint64))
    crop.create_array("roi_coordinates_full", data=np.zeros((3, 2), dtype=np.int32))
    crop_parent.attrs["latest"] = "crop_001"
    return zarr_path


def _patch_keypoint_writer_dependencies(monkeypatch, model_path) -> None:
    def _fake_open(root_group, **_kwargs):
        return _FakeCropSource(root_group["crop_runs"]["crop_001"])

    monkeypatch.setattr(yolo_mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(yolo_mod.CropImageSource, "open", staticmethod(_fake_open))
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


def _patch_canonical_writer_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    model_path,
    *,
    model_class: type[_FakeYOLO],
) -> None:
    monkeypatch.setattr(yolo_mod, "YOLO", model_class)
    monkeypatch.setattr(
        yolo_mod,
        "_prepare_refined_roi_overrides",
        lambda *_args, **_kwargs: None,
    )
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
        lambda **_kwargs: {
            "platform": {"hostname": "unit-test"},
            "environment": {},
            "gpu": {},
        },
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
            run_name="keypoint_shard_001",
            output_parent="keypoint_shard_runs",
            pose_schema="traditional_v3",
            batch_size=8,
            imgsz=8,
            input_mode="numpy-list",
            coordinate_contract_mode="legacy_noncanonical",
            keypoint_roi_shard_rows=8,
            keypoint_frame_shard_rows=8,
            registry=None,
        )
    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoint_shard_runs"][run_name]
    return {
        "n_rois": np.asarray(run["n_rois"][:]),
        "frame_counts": np.asarray(run["frame_counts"][:]),
    }


def test_detect_keypoints_yolo_can_write_collection_shard_without_stage_selectors_or_canonical_pointer(
    monkeypatch, tmp_path
) -> None:
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
            coordinate_contract_mode="legacy_noncanonical",
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
    assert "latest" not in shard_parent.attrs
    assert "latest_complete" not in shard_parent.attrs
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
    assert run.attrs["coordinate_contract"] == "legacy_noncanonical_explicit_v1"


def test_detect_keypoints_yolo_default_writer_publishes_only_freshly_validated_canonical_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, _prior_run = _real_canonical_archive(tmp_path)
    parent = root["keypoints_runs"]
    parent.attrs["latest"] = "k1"
    parent.attrs["latest_complete"] = "k1"
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/k1"
    model_path = tmp_path / "canonical.pt"
    telemetry_attempts: list[str] = []

    with monkeypatch.context() as patch:
        _patch_canonical_writer_dependencies(
            patch,
            model_path,
            model_class=_CanonicalFakeYOLO,
        )
        patch.setattr(
            yolo_mod,
            "_emit_keypoint_step_status",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic post-commit registry failure")
            ),
        )

        def _progress_telemetry(_path, event, **_kwargs):
            telemetry_attempts.append(str(event))
            if event == "complete":
                raise OSError("synthetic post-commit progress failure")

        patch.setattr(yolo_mod, "_write_keypoint_progress_jsonl", _progress_telemetry)
        run_name = detect_keypoints_yolo(
            tmp_path / "canonical.zarr",
            model_path,
            run_provenance=build_writer_run_provenance(
                command="unit-canonical-keypoint-writer",
                params={"model_path": model_path},
            ),
            run_name="canonical_writer",
            crop_run="c1",
            pose_schema="traditional_v3",
            batch_size=2,
            imgsz=40,
            input_mode="numpy-list",
            keypoint_roi_shard_rows=None,
            registry=None,
        )

    reopened = zarr.open_group(
        store=str(tmp_path / "canonical.zarr"),
        mode="r",
        use_consolidated=False,
    )
    run = reopened["keypoints_runs"][run_name]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["coordinate_contract"] == "canonical_v2"
    assert run.attrs["stage_selector_eligible"] is True
    assert telemetry_attempts[-1] == "complete"
    assert reopened["keypoints_runs"].attrs["latest"] == run_name
    assert reopened["keypoints_runs"].attrs["latest_complete"] == run_name
    assert reopened.attrs["current_keypoint_group_path"] == (
        f"keypoints_runs/{run_name}"
    )
    surfaces = load_persisted_keypoint_coordinate_surfaces(
        reopened,
        f"keypoints_runs/{run_name}",
    )
    assert surfaces.keypoints_img.descriptor.space_id == "source_camera_image_px"


def test_detect_keypoints_yolo_failed_canonical_attempt_preserves_prior_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root, _prior_run = _real_canonical_archive(tmp_path)
    parent = root["keypoints_runs"]
    parent.attrs["latest"] = "k1"
    parent.attrs["latest_complete"] = "k1"
    parent.attrs["latest_pending"] = "prior_pending_attempt"
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/k1"
    model_path = tmp_path / "wrong-shape.pt"
    writers = []

    with monkeypatch.context() as patch:
        _patch_canonical_writer_dependencies(
            patch,
            model_path,
            model_class=_SecondBatchWrongShapeCanonicalFakeYOLO,
        )
        writer_class = yolo_mod._AlignedKeypointShardWriter

        class _CapturedShardWriter(writer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                writers.append(self)

        patch.setattr(yolo_mod, "_AlignedKeypointShardWriter", _CapturedShardWriter)
        with pytest.raises(ValueError, match="orig_shape"):
            detect_keypoints_yolo(
                tmp_path / "canonical.zarr",
                model_path,
                run_provenance=build_writer_run_provenance(
                    command="unit-failed-canonical-keypoint-writer",
                    params={"model_path": model_path},
                ),
                run_name="failed_canonical_writer",
                crop_run="c1",
                pose_schema="traditional_v3",
                batch_size=1,
                imgsz=40,
                input_mode="numpy-list",
                keypoint_roi_shard_rows=1,
                keypoint_frame_shard_rows=1,
                registry=None,
            )

    assert len(writers) == 1
    assert writers[0].is_quiescent
    assert not writers[0]._worker.is_alive()
    assert writers[0]._flush.unfinished_tasks == 0
    reopened = zarr.open_group(
        store=str(tmp_path / "canonical.zarr"),
        mode="r",
        use_consolidated=False,
    )
    parent = reopened["keypoints_runs"]
    failed = parent["failed_canonical_writer"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert "coordinate_contract" not in failed.attrs
    assert parent.attrs["latest"] == "k1"
    assert parent.attrs["latest_complete"] == "k1"
    assert parent.attrs["latest_pending"] == "prior_pending_attempt"
    assert reopened.attrs["current_keypoint_group_path"] == "keypoints_runs/k1"


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
            run_name="keypoint_shard_regular",
            output_parent="keypoint_shard_runs",
            pose_schema="traditional_v3",
            batch_size=8,
            imgsz=8,
            input_mode="numpy-list",
            coordinate_contract_mode="legacy_noncanonical",
            keypoint_roi_shard_rows=None,
            registry=None,
        )

    run = zarr.open_group(store=str(zarr_path), mode="r")["keypoint_shard_runs"][run_name]
    assert run["keypoints_roi"].shards is None
    assert run.attrs["keypoint_storage_layout"] == "regular_chunks_v1"
    assert run.attrs["keypoint_storage_policy"] == "explicit_regular_chunks_override"
    assert run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]["status"] == "ok"


def test_detect_keypoints_yolo_fails_closed_without_instance_key(monkeypatch, tmp_path) -> None:
    zarr_path = _make_keypoint_count_fixture(tmp_path, "keyless")
    model_path = tmp_path / "keyless.pt"
    root = zarr.open_group(store=str(zarr_path), mode="a", use_consolidated=False)
    del root["crop_runs/crop_001/instance_key"]

    with monkeypatch.context() as patch:
        _patch_keypoint_writer_dependencies(patch, model_path)
        with pytest.raises(RuntimeError, match="missing required arrays.*instance_key"):
            detect_keypoints_yolo(
                zarr_path,
                model_path,
                run_provenance=build_writer_run_provenance(
                    command="unit-keypoint-keyless",
                    params={"model_path": model_path},
                ),
                run_name="keypoint_shard_keyless",
                output_parent="keypoint_shard_runs",
                pose_schema="traditional_v3",
                batch_size=8,
                imgsz=8,
                input_mode="numpy-list",
                coordinate_contract_mode="legacy_noncanonical",
                keypoint_roi_shard_rows=8,
                keypoint_frame_shard_rows=8,
                registry=None,
            )

    root = zarr.open_group(store=str(zarr_path), mode="r", use_consolidated=False)
    parent = root["keypoint_shard_runs"]
    failed = parent["keypoint_shard_keyless"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]["status"] == "error"
    assert parent.attrs.get("latest_pending") != "keypoint_shard_keyless"
    assert parent.attrs.get("latest_complete") != "keypoint_shard_keyless"
    assert parent.attrs.get("latest") != "keypoint_shard_keyless"
    assert "current_keypoint_group_path" not in root.attrs


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

    np.testing.assert_allclose(actual, np.array([0.0, 1.5, 8.0, 6.0], dtype=np.float32))


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


def test_prepared_model_input_contract_rejects_dtype_channels_and_range() -> None:
    transform = resolve_model_input_transform((8, 8), model_hw=(8, 8))
    with pytest.raises(ValueError, match="uint8 submitted extent"):
        _require_prepared_model_input_contract(
            [np.zeros((8, 8, 3), dtype=np.float32)],
            effective_mode="numpy-list",
            expected_mode="numpy-list",
            batch_count=1,
            model_input_transform=transform,
        )

    unequal_channels = np.zeros((8, 8, 3), dtype=np.uint8)
    unequal_channels[..., 1] = 1
    with pytest.raises(ValueError, match="channel semantics"):
        _require_prepared_model_input_contract(
            [unequal_channels],
            effective_mode="numpy-list",
            expected_mode="numpy-list",
            batch_count=1,
            model_input_transform=transform,
        )

    with pytest.raises(ValueError, match=r"closed \[0,1\] range"):
        _require_prepared_model_input_contract(
            torch.full((1, 3, 8, 8), 2.0, dtype=torch.float32),
            effective_mode="tensor",
            expected_mode="tensor",
            batch_count=1,
            model_input_transform=transform,
        )


def test_model_result_coordinate_contract_requires_exact_count_and_orig_shape() -> None:
    transform = resolve_model_input_transform((8, 8), model_hw=(8, 8))
    valid = SimpleNamespace(orig_shape=(8, 8))
    _require_model_result_coordinate_contract(
        (valid,),
        batch_count=1,
        model_input_transform=transform,
    )
    with pytest.raises(ValueError, match="result cardinality"):
        _require_model_result_coordinate_contract(
            (),
            batch_count=1,
            model_input_transform=transform,
        )
    with pytest.raises(ValueError, match="orig_shape"):
        _require_model_result_coordinate_contract(
            (SimpleNamespace(orig_shape=(7, 8)),),
            batch_count=1,
            model_input_transform=transform,
        )
