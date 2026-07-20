from __future__ import annotations

import inspect
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.detection import detect_yolo as mod
from fisheye.diagnostics.audit_yolo_detection_sharding import (
    audit_detection_runs,
    replay_detection_run_as_sharded,
)
from fisheye.inference import predict_detections
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PENDING_REASON,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    build_acquisition_authority_publication_status,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.observation_coordinate_publication import (
    derive_detection_source_camera_geometry,
    load_persisted_detection_observation_geometry,
    load_persisted_source_camera_position_surface,
    publish_crop_observation_geometry,
    publish_detection_observation_geometry,
)
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.shared.coordinate_identity import (
    resolve_source_acquisition_frame_indices,
)
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.immutable_yolo_storage import (
    IMMUTABLE_YOLO_STORAGE_ATTR,
    IMMUTABLE_YOLO_STORAGE_SCHEMA,
)
from fisheye.shared.instance_keys import (
    instance_key_attrs,
    mint_detection_instance_keys,
)


def test_yolo_detection_sharding_is_default_with_cli_opt_out() -> None:
    signature = inspect.signature(mod.detect_yolo)
    assert signature.parameters["detect_row_shard_rows"].default == 131_072
    assert signature.parameters["detect_frame_shard_rows"].default == 131_072
    assert signature.parameters["coordinate_contract_mode"].default == "canonical"
    assert signature.parameters["output_run_family"].default == "detect_runs"

    parser = predict_detections._build_parser()  # noqa: SLF001
    defaults = parser.parse_args(["--video", "sample.mp4"])
    assert defaults.detect_row_shard_rows == 131_072
    assert defaults.detect_frame_shard_rows == 131_072
    assert not hasattr(defaults, "coordinate_contract_mode")

    regular = parser.parse_args(["--video", "sample.mp4", "--no-detect-sharding"])
    assert regular.detect_row_shard_rows is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--video",
                "sample.mp4",
                "--coordinate-contract-mode",
                "legacy_noncanonical",
            ]
        )


def test_detection_writer_rejects_legacy_and_cross_namespace_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported coordinate_contract_mode"):
        mod.detect_yolo("missing.mp4", coordinate_contract_mode="legacy_noncanonical")
    with pytest.raises(ValueError, match="must persist under detect_runs"):
        mod.detect_yolo(
            "missing.mp4",
            coordinate_contract_mode="canonical",
            output_run_family="detection_artifact_runs",
        )
    with pytest.raises(ValueError, match="dedicated"):
        mod.detect_yolo(
            "missing.mp4",
            coordinate_contract_mode="artifact_unbound",
            output_run_family="detect_runs",
        )
    with pytest.raises(ValueError, match="detached"):
        mod.detect_yolo(
            "missing.mp4",
            coordinate_frame_evidence=object(),
        )
    with pytest.raises(ValueError, match="cannot accept canonical instance-key"):
        mod.detect_yolo(
            "missing.mp4",
            coordinate_contract_mode="artifact_unbound",
            output_run_family="detection_artifact_runs",
            instance_key_recording_identity="not-authoritative-here",
        )


def _detection_values() -> dict[str, np.ndarray]:
    frame_indices = np.asarray([0, 1, 1, 3, 5, 8, 13, 21, 34, 35], dtype=np.int32)
    bbox_coords = np.arange(40, dtype=np.float64).reshape(10, 4) / 100.0
    scores = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    class_ids = np.arange(10, dtype=np.int32) % 2
    instance_keys = np.arange(100, 110, dtype=np.uint64)
    frame_counts = np.bincount(frame_indices, minlength=40_000).astype(np.int32)
    return {
        "frame_indices": frame_indices,
        "bbox_coords": bbox_coords,
        "scores": scores,
        "class_ids": class_ids,
        "instance_keys": instance_keys,
        "frame_counts": frame_counts,
    }


def _canonical_acquisition_root(tmp_path):
    recording = tmp_path / "recording"
    camera_dir = recording / "cams"
    camera_dir.mkdir(parents=True)
    video = camera_dir / "camera-a.mp4"
    video.write_bytes(b"exact-video-source")
    source = {
        "source_path": str(video),
        "camera_id": "camera-a",
        "width": 100,
        "height": 80,
        "total_frames": 2,
        "fps": 10.0,
        "codec": "h264",
        "pix_fmt": "yuv420p",
    }
    fingerprint = source_stat_fingerprint_attrs(
        video,
        attr_prefix="source_video",
        extra={
            "codec": source["codec"],
            "pix_fmt": source["pix_fmt"],
            "width": source["width"],
            "height": source["height"],
            "fps": source["fps"],
            "frame_count": source["total_frames"],
        },
    )
    metadata = build_source_video_metadata_v2(
        source,
        recording_path=recording,
        fingerprint_attrs=fingerprint,
    )
    zarr_path = recording / "zarr" / "analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    root.attrs.update(
        {
            "recording_id": "recording-1",
            "camera_id": "camera-a",
            "recording_path": str(recording),
            "source_video_path": str(video),
            "source_path": str(video),
            "source_video_metadata": metadata,
        }
    )
    publish_external_video_acquisition_authority(root)
    return root, zarr_path, video


_TEST_RUN_PROVENANCE = {
    "schema": "palette.run_provenance.v1",
    "git_sha": "0" * 40,
    "config_hash": "1" * 64,
    "params": {"test": True},
    "input_run_ids": {"source": "test"},
    "input_artifacts": [],
    "command": "pytest detect_yolo publication",
    "fisheye_version": None,
}


class _FakeVideoCapture:
    def __init__(self) -> None:
        self.release_count = 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
        return True

    def get(self, prop: int) -> float:
        values = {
            mod.cv2.CAP_PROP_FRAME_COUNT: 1.0,
            mod.cv2.CAP_PROP_FPS: 10.0,
            mod.cv2.CAP_PROP_FRAME_WIDTH: 6.0,
            mod.cv2.CAP_PROP_FRAME_HEIGHT: 4.0,
            mod.cv2.CAP_PROP_FOURCC: 0.0,
        }
        return values.get(prop, 0.0)

    def release(self) -> None:
        self.release_count += 1


class _FakeYoloModel:
    def __init__(self, predict) -> None:
        self._predict = predict

    def fuse(self) -> None:
        return None

    def to(self, _device):
        return self

    def predict(self, frames, **_kwargs):
        return self._predict(frames)


class _FakeBoxes:
    def __init__(self, xyxy: list[list[float]]) -> None:
        self.xyxy = mod.torch.asarray(xyxy, dtype=mod.torch.float32)
        self.conf = mod.torch.ones(len(xyxy), dtype=mod.torch.float32)
        self.cls = mod.torch.zeros(len(xyxy), dtype=mod.torch.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


def _install_detect_yolo_producer_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    predict,
    decode_error: BaseException | None = None,
) -> tuple[object, _FakeVideoCapture, object, object]:
    video_path = tmp_path / "source.mp4"
    model_path = tmp_path / "model.pt"
    output_path = tmp_path / "analysis.zarr"
    video_path.write_bytes(b"fake-video")
    model_path.write_bytes(b"fake-model")

    root = zarr.open_group(output_path, mode="w")
    parent = root.create_group("detect_runs")
    parent.create_group("previous")
    selector_attrs = {
        "latest": "previous",
        "latest_complete": "previous",
        "latest_pending": "older-pending",
        "authoritative_run": "previous",
        "authoritative_run_provenance": {"approved_by": "pytest"},
    }
    parent.attrs.update(selector_attrs)

    cap = _FakeVideoCapture()
    model = _FakeYoloModel(predict)
    acquisition = SimpleNamespace(
        record=SimpleNamespace(
            recording_id="recording-test",
            width_px=6,
            height_px=4,
            source_total_frames=1,
        ),
        record_ref="/analysis/acquisition_camera_frames/camera-a#acquisition_camera_frame",
        record_sha256="a" * 64,
    )
    monkeypatch.setattr(mod, "load_config", lambda _path=None: {})
    monkeypatch.setattr(mod, "YOLO", lambda _path: model)
    monkeypatch.setattr(mod.cv2, "VideoCapture", lambda _path: cap)
    monkeypatch.setattr(
        mod,
        "get_video_metadata",
        lambda path, _cap, width, height, n_frames, fps: {
            "source_video": path.name,
            "source_path": str(path),
            "width": width,
            "height": height,
            "total_frames": n_frames,
            "fps": fps,
            "duration_seconds": n_frames / fps,
            "codec": "fake",
            "pix_fmt": "gray",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda *_args, **_kwargs: {
            "commit_hash": "0" * 40,
            "short_hash": "0" * 8,
            "branch": "pytest",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {"hostname": "pytest", "fqdn": "pytest"},
            "environment": {},
            "gpu": {"available": False, "devices": []},
        },
    )
    monkeypatch.setattr(
        mod,
        "_load_full_acquisition_video_source",
        lambda *_args, **_kwargs: acquisition,
    )

    def iter_batches(**_kwargs):
        if decode_error is not None:
            raise decode_error
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        yield [0], [frame], 0.0, 0.0, 1

    monkeypatch.setattr(mod, "_iter_opencv_rgb_batches", iter_batches)
    monkeypatch.setattr(
        mod,
        "_publish_detection_frame_evidence",
        lambda *_args, **_kwargs: (object(), ()),
    )
    monkeypatch.setattr(
        mod,
        "derive_detection_source_camera_geometry",
        lambda bbox, **_kwargs: (
            np.zeros((int(bbox.shape[0]), 4), dtype=np.float64),
            np.zeros((int(bbox.shape[0]), 2), dtype=np.float64),
        ),
    )

    def publish_mapping(run_group, **_kwargs):
        run_group.attrs[mod.DETECTION_ACQUISITION_MAPPING_ATTR] = {"test": True}
        return object()

    def publish_geometry(run_group, *nodes, **_kwargs):
        run_group.attrs.update(
            {
                "row_identity_contract": {"test": True},
                "detection_bbox_projection": {"test": True},
                "bbox_center_derivation": {"test": True},
            }
        )
        for node in nodes:
            node.attrs["coordinate_descriptor"] = {"test": True}
        return object()

    monkeypatch.setattr(mod, "_publish_detection_acquisition_mapping", publish_mapping)
    monkeypatch.setattr(
        mod,
        "publish_detection_instance_key_derivation",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        mod,
        "publish_detection_observation_cardinality",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(mod, "publish_detection_observation_geometry", publish_geometry)

    def validate_storage(run_group, **_kwargs):
        report = {
            "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
            "status": "ok",
            "stage": "detect",
            "row_count": int(run_group["instance_key"].shape[0]),
            "frame_count": int(run_group["frame_counts"].shape[0]),
            "instance_key_present": True,
            "instance_key_unique": True,
        }
        run_group.attrs[IMMUTABLE_YOLO_STORAGE_ATTR] = report
        return report

    monkeypatch.setattr(mod, "validate_immutable_yolo_storage", validate_storage)
    monkeypatch.setattr(
        mod,
        "_load_persisted_detection_observation_geometry",
        lambda *_args, **_kwargs: object(),
    )
    call_args = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "output_zarr": str(output_path),
        "decode_backend": "opencv",
        "use_gpu": False,
        "run_name": "candidate",
        "detect_row_shard_rows": None,
        "instance_key_recording_identity": "recording-test",
        "run_provenance": _TEST_RUN_PROVENANCE,
    }
    return call_args, cap, output_path, selector_attrs


def _assert_failed_producer_attempt_rolled_back(
    output_path,
    selector_attrs: dict[str, object],
) -> None:
    root = zarr.open_group(output_path, mode="r", use_consolidated=False)
    parent = root["detect_runs"]
    assert "candidate" not in parent
    for name, value in selector_attrs.items():
        assert parent.attrs[name] == value


def test_write_detection_output_arrays_uses_complete_indexed_shards(tmp_path) -> None:
    group = zarr.open_group(tmp_path / "detect_sharded.zarr", mode="w")
    values = _detection_values()

    summary = mod._write_detection_output_arrays(  # noqa: SLF001
        group,
        **values,
        det_chunk=4,
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )

    assert summary is not None
    assert summary["schema_id"] == mod.DETECT_SHARD_WRITE_SCHEMA
    assert summary["write_mode"] == "materialized_complete_shards"
    assert summary["detect_row_shard_rows_effective"] == 8
    assert summary["detect_frame_shard_rows_effective"] == 32_768
    assert summary["exact_match"] is True
    assert summary["source_sha256_by_array"] == summary["destination_sha256_by_array"]

    expected = {
        "frame_indices": values["frame_indices"],
        "bbox_norm_coords": values["bbox_coords"],
        "scores": values["scores"],
        "class_ids": values["class_ids"],
        "instance_key": values["instance_keys"],
        "n_detections": values["frame_counts"],
        "frame_counts": values["frame_counts"],
    }
    for name, source in expected.items():
        np.testing.assert_array_equal(group[name][:], source)

    assert group["bbox_norm_coords"].chunks == (4, 4)
    assert group["bbox_norm_coords"].shards == (8, 4)
    assert group["frame_counts"].chunks == (16_384,)
    assert group["frame_counts"].shards == (32_768,)
    assert group["frame_counts"].attrs["storage_profile_id"] == "geometry_preload_v1"


def test_write_detection_output_arrays_preserves_regular_layout(tmp_path) -> None:
    group = zarr.open_group(tmp_path / "detect_regular.zarr", mode="w")
    values = _detection_values()

    summary = mod._write_detection_output_arrays(  # noqa: SLF001
        group,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )

    assert summary is None
    assert group["bbox_norm_coords"].chunks == (4, 4)
    assert group["bbox_norm_coords"].shards is None
    np.testing.assert_array_equal(group["bbox_norm_coords"][:], values["bbox_coords"])


def test_write_detection_output_arrays_keeps_canonical_surfaces_atomic(
    tmp_path,
) -> None:
    group = zarr.open_group(tmp_path / "detect_canonical.zarr", mode="w")
    values = _detection_values()
    source_frames = values["frame_indices"].astype(np.int64)
    bbox_img = values["bbox_coords"] * np.asarray(
        [100.0, 80.0, 100.0, 80.0], dtype=np.float64
    )
    centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * 0.5,
            (bbox_img[:, 1] + bbox_img[:, 3]) * 0.5,
        )
    )

    summary = mod._write_detection_output_arrays(  # noqa: SLF001
        group,
        **values,
        det_chunk=4,
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
        source_acquisition_frame_indices=source_frames,
        bbox_img_xyxy=bbox_img,
        centers_img_xy=centers,
    )

    assert summary is not None
    for name, expected in (
        ("source_acquisition_frame_index", source_frames),
        ("bbox_img_xyxy", bbox_img),
        ("centers_img_xy", centers),
    ):
        np.testing.assert_array_equal(group[name][:], expected)
        assert name in summary["source_sha256_by_array"]
        assert name in summary["destination_sha256_by_array"]
    assert group["source_acquisition_frame_index"].dtype == np.dtype("<i8")
    assert group["bbox_img_xyxy"].shards == (8, 4)
    assert group["centers_img_xy"].shards == (8, 2)

    incomplete = zarr.open_group(tmp_path / "detect_incomplete.zarr", mode="w")
    with pytest.raises(ValueError, match="require source acquisition frames"):
        mod._write_detection_output_arrays(  # noqa: SLF001
            incomplete,
            **values,
            det_chunk=4,
            detect_row_shard_rows=None,
            detect_frame_shard_rows=32_768,
            source_acquisition_frame_indices=source_frames,
        )
    assert not tuple(incomplete.array_keys())


def test_organized_archive_auto_builds_full_acquisition_detection_evidence(
    tmp_path,
) -> None:
    root, zarr_path, video = _canonical_acquisition_root(tmp_path)
    acquisition = mod._load_full_acquisition_video_source(  # noqa: SLF001
        root,
        video_path=video,
        output_zarr=zarr_path,
        decoded_frame_count=2,
        decoded_width=100,
        decoded_height=80,
    )
    run = root.require_group("detect_runs").create_group("canonical")
    evidence, _frame_checkpoints = mod._publish_detection_frame_evidence(  # noqa: SLF001
        root,
        run,
        acquisition_frame=acquisition,
    )
    frame_indices = np.asarray([0, 1], dtype=np.int32)
    run.create_array(
        "frame_indices",
        data=frame_indices,
    )
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([0, 1], dtype=np.int64),
    )
    mapping = mod._publish_detection_acquisition_mapping(  # noqa: SLF001
        run,
        acquisition_frame=acquisition,
    )
    normalized = np.asarray(
        [[0.25, 0.50, 0.20, 0.25], [0.75, 0.25, 0.10, 0.20]],
        dtype=np.float64,
    )
    bbox_img, centers = derive_detection_source_camera_geometry(
        normalized,
        frame_evidence=evidence,
    )
    class_ids = np.zeros((2,), dtype=np.int32)
    instance_keys = mint_detection_instance_keys(
        recording_identity=acquisition.record.recording_id,
        frame_indices=frame_indices.astype(np.int64),
        bbox_norm_coords=normalized,
        class_ids=class_ids,
    )
    run.create_array("instance_key", data=instance_keys)
    run.create_array("bbox_norm_coords", data=normalized)
    run.create_array("bbox_img_xyxy", data=bbox_img)
    run.create_array("centers_img_xy", data=centers)
    run.create_array("class_ids", data=class_ids)
    run.create_array("scores", data=np.ones((2,), dtype=np.float32))
    run.create_array("frame_counts", data=np.ones((2,), dtype=np.int32))
    run.create_array("n_detections", data=np.ones((2,), dtype=np.int32))
    dense_mapping = np.arange(2, dtype=np.int64)
    run.attrs.update(
        {
            **instance_key_attrs(
                acquisition.record.recording_id,
                frame_domain="recording_parent_frame_index",
                frame_mapping_source=(
                    f"{acquisition.record_ref}#"
                    "full_untrimmed_video_decode_identity_v1"
                ),
                frame_mapping_sha256=hashlib.sha256(
                    np.ascontiguousarray(dense_mapping).view(np.uint8)
                ).hexdigest(),
            ),
            mod.OBSERVATION_ROW_COUNT_ATTR: 2,
            "summary_statistics": {
                "total_detections": 2,
                "frames_with_detections": 2,
                "frames_with_zero_detections": 0,
                "frames_with_multiple_detections": 0,
            },
            "detect_row_shard_rows": None,
            "detect_shard_write": None,
            IMMUTABLE_YOLO_STORAGE_ATTR: {
                "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
                "status": "ok",
                "stage": "detect",
                "row_count": 2,
                "frame_count": 2,
                "instance_key_present": True,
                "instance_key_unique": True,
            },
        }
    )
    instance_key_derivation = mod.publish_detection_instance_key_derivation(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["class_ids"],
        acquisition_frame=acquisition,
        acquisition_mapping=mapping,
    )
    mod.publish_detection_observation_cardinality(
        run,
        acquisition_frame=acquisition,
    )
    detection = publish_detection_observation_geometry(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["bbox_img_xyxy"],
        run["centers_img_xy"],
        frame_evidence=evidence,
        source_lineage_records=(mapping, instance_key_derivation),
    )
    run.attrs["coordinate_contract"] = "canonical_v2"
    run.attrs["palette_run_completion_contract"] = "palette.zarr_run_completion.v1"
    run.attrs["palette_run_completion_status"] = "complete"
    run.attrs["stage_selector_eligible"] = True
    position = load_persisted_source_camera_position_surface(
        root,
        "detect_runs/canonical",
    )

    assert evidence.acquisition_frame.record_sha256 == acquisition.record_sha256
    assert position.coordinates.descriptor.geometry_type == "point_xy"
    assert position.temporal_authority.record.source_total_frames == 2

    crop = root.require_group("crop_runs").create_group("canonical")
    source_rows = np.asarray([1, 0], dtype=np.int64)
    crop.create_array("detection_indices", data=source_rows)
    crop.create_array(
        "instance_key",
        data=np.asarray(run["instance_key"][:])[source_rows],
    )
    crop.create_array(
        "source_acquisition_frame_index",
        data=np.asarray(run["source_acquisition_frame_index"][:])[source_rows],
    )
    for name in ("bbox_norm_coords", "bbox_img_xyxy", "centers_img_xy"):
        crop.create_array(name, data=np.asarray(run[name][:])[source_rows])
    publish_crop_observation_geometry(
        crop,
        crop["instance_key"],
        crop["detection_indices"],
        crop["source_acquisition_frame_index"],
        crop["bbox_norm_coords"],
        crop["bbox_img_xyxy"],
        crop["centers_img_xy"],
        source_geometry=detection,
    )
    crop.attrs["coordinate_contract"] = "canonical_v2"
    crop.attrs["palette_run_completion_contract"] = "palette.zarr_run_completion.v1"
    crop.attrs["palette_run_completion_status"] = "complete"
    crop.attrs["stage_selector_eligible"] = True
    crop_position = load_persisted_source_camera_position_surface(
        root,
        "crop_runs/canonical",
    )
    assert np.array_equal(
        resolve_source_acquisition_frame_indices(
            crop_position.temporal_authority,
            np.asarray([0, 1], dtype=np.int64),
        ),
        np.asarray([1, 0], dtype=np.int64),
    )
    assert mapping.record["operation"] == (
        "full_untrimmed_video_decode_identity_to_acquisition_v1"
    )
    assert (
        mapping.record["source_video_fingerprint"]
        == (acquisition.record.source_video_metadata["file_fingerprint"])
    )

    run["source_acquisition_frame_index"][:] = np.asarray([1, 0], dtype=np.int64)
    with pytest.raises(Exception, match="identity|exact"):
        load_persisted_source_camera_position_surface(
            root,
            "detect_runs/canonical",
        )
    with pytest.raises(ValueError, match="identity mapping"):
        mod._publish_detection_acquisition_mapping(  # noqa: SLF001
            run,
            acquisition_frame=acquisition,
        )

    video.write_bytes(b"different-video-source")
    with pytest.raises(ValueError, match="differs from the exact acquisition"):
        mod._load_full_acquisition_video_source(  # noqa: SLF001
            root,
            video_path=video,
            output_zarr=zarr_path,
            decoded_frame_count=2,
            decoded_width=100,
            decoded_height=80,
        )


def _complete_canonical_detection_observation(
    tmp_path,
    *,
    row_count: int,
):
    root, _zarr_path, _video = _canonical_acquisition_root(tmp_path)
    acquisition = mod._load_published_detection_acquisition_frame(root)  # noqa: SLF001
    parent = mod.require_runs_parent(root, "detect_runs")
    run_name = "empty" if row_count == 0 else "nonempty"
    run = parent.create_group(run_name)
    run.attrs["stage_selector_eligible"] = True
    mod.mark_run_started(run, run_name=run_name, stage="detect")
    evidence, _checkpoints = mod._publish_detection_frame_evidence(  # noqa: SLF001
        root,
        run,
        acquisition_frame=acquisition,
    )
    frame_indices = np.arange(row_count, dtype=np.int32)
    normalized = np.tile(
        np.asarray([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64),
        (row_count, 1),
    )
    bbox_img, centers = derive_detection_source_camera_geometry(
        normalized,
        frame_evidence=evidence,
    )
    class_ids = np.zeros((row_count,), dtype=np.int32)
    instance_keys = mint_detection_instance_keys(
        recording_identity=acquisition.record.recording_id,
        frame_indices=frame_indices.astype(np.int64),
        bbox_norm_coords=normalized,
        class_ids=class_ids,
    )
    run.create_array("frame_indices", data=frame_indices)
    run.create_array(
        "source_acquisition_frame_index",
        data=frame_indices.astype(np.int64),
    )
    run.create_array("bbox_norm_coords", data=normalized)
    run.create_array("bbox_img_xyxy", data=bbox_img)
    run.create_array("centers_img_xy", data=centers)
    run.create_array("scores", data=np.ones((row_count,), dtype=np.float32))
    run.create_array("class_ids", data=class_ids)
    run.create_array(
        "instance_key",
        data=instance_keys,
    )
    run.create_array(
        "frame_counts",
        data=np.bincount(frame_indices, minlength=2).astype(np.int32),
    )
    run.create_array(
        "n_detections",
        data=np.bincount(frame_indices, minlength=2).astype(np.int32),
    )
    dense_mapping = np.arange(2, dtype=np.int64)
    run.attrs.update(
        {
            **instance_key_attrs(
                acquisition.record.recording_id,
                frame_domain="recording_parent_frame_index",
                frame_mapping_source=(
                    f"{acquisition.record_ref}#"
                    "full_untrimmed_video_decode_identity_v1"
                ),
                frame_mapping_sha256=hashlib.sha256(
                    np.ascontiguousarray(dense_mapping).view(np.uint8)
                ).hexdigest(),
            ),
            mod.OBSERVATION_ROW_COUNT_ATTR: row_count,
            "summary_statistics": {
                "total_detections": row_count,
                "frames_with_detections": row_count,
                "frames_with_zero_detections": 2 - row_count,
                "frames_with_multiple_detections": 0,
            },
            "detect_row_shard_rows": None,
            "detect_shard_write": None,
            "decode_backend_effective": "opencv",
            "decode_domain_proof": "opencv_stream_eof_and_exact_count_v1",
            "timing_summary": {
                "decode_backend_effective": "opencv",
                "decode_domain_proof": "opencv_stream_eof_and_exact_count_v1",
                "frames_processed": 2,
            },
            IMMUTABLE_YOLO_STORAGE_ATTR: {
                "schema_id": IMMUTABLE_YOLO_STORAGE_SCHEMA,
                "status": "ok",
                "stage": "detect",
                "row_count": row_count,
                "frame_count": 2,
                "instance_key_present": True,
                "instance_key_unique": True,
            },
        }
    )
    mapping = mod._publish_detection_acquisition_mapping(  # noqa: SLF001
        run,
        acquisition_frame=acquisition,
    )
    instance_key_derivation = mod.publish_detection_instance_key_derivation(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["class_ids"],
        acquisition_frame=acquisition,
        acquisition_mapping=mapping,
    )
    mod.publish_detection_observation_cardinality(
        run,
        acquisition_frame=acquisition,
    )
    publish_detection_observation_geometry(
        run,
        run["instance_key"],
        run["source_acquisition_frame_index"],
        run["bbox_norm_coords"],
        run["bbox_img_xyxy"],
        run["centers_img_xy"],
        frame_evidence=evidence,
        source_lineage_records=(mapping, instance_key_derivation),
    )
    declaration = None
    if row_count == 0:
        declaration = mod._publish_empty_detection_observation_declaration(  # noqa: SLF001
            run,
            acquisition_frame=acquisition,
            decoded_frame_count=2,
            decode_domain_proof="opencv_stream_eof_and_exact_count_v1",
        )
    run.attrs["coordinate_contract"] = "canonical_v2"
    mod.mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=_TEST_RUN_PROVENANCE,
    )
    return root, run, declaration


def test_real_coordinate_publication_accepts_only_declared_empty_rowset(
    tmp_path,
) -> None:
    root, run, declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=0,
    )

    geometry = load_persisted_detection_observation_geometry(
        root,
        "detect_runs/empty",
    )

    assert geometry.row_identity.leading_dimension == 0
    assert declaration is not None
    assert declaration.record["is_empty"] is True
    assert declaration.record["observation_row_count"] == 0
    assert set(declaration.record["row_arrays"]) == {
        "frame_indices",
        "source_acquisition_frame_index",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
        "scores",
        "class_ids",
        "instance_key",
    }
    assert set(declaration.record["frame_count_arrays"]) == {
        "frame_counts",
        "n_detections",
    }
    assert run["instance_key"].shape == (0,)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_declaration", "requires an exact persisted"),
        ("tampered_declaration", "stale"),
        ("acquisition_count_mismatch", "exact acquisition domain"),
        ("row_count_mismatch", "disagrees with the exact"),
        ("unsupported_decode_proof", "supported decode-domain proof"),
        ("missing_n_detections", "arrays are incomplete"),
        ("nonzero_n_detections", "exact frame_indices cardinality"),
        ("summary_count_mismatch", "summary count authorities"),
        ("decode_backend_mismatch", "does not match its exact decoder backend"),
    ],
)
def test_public_detection_loader_rejects_invalid_empty_publication(
    tmp_path,
    corruption: str,
    message: str,
) -> None:
    root, run, _declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=0,
    )
    declaration_attr = mod.EMPTY_OBSERVATION_DECLARATION_ATTR
    digest_attr = f"{declaration_attr}_sha256"

    if corruption == "missing_declaration":
        del run.attrs[declaration_attr]
    elif corruption == "tampered_declaration":
        record = dict(run.attrs[declaration_attr])
        record["source_total_frames"] = 3
        run.attrs[declaration_attr] = record
    elif corruption == "row_count_mismatch":
        run.attrs[mod.OBSERVATION_ROW_COUNT_ATTR] = 1
    elif corruption == "acquisition_count_mismatch":
        record = dict(run.attrs[declaration_attr])
        record["source_total_frames"] = 3
        run.attrs[declaration_attr] = record
        run.attrs[digest_attr] = coordinate_record_sha256(record)
    elif corruption == "unsupported_decode_proof":
        record = dict(run.attrs[declaration_attr])
        record["decode_domain_proof"] = "decoder_claim_without_domain_proof"
        run.attrs[declaration_attr] = record
        run.attrs[digest_attr] = coordinate_record_sha256(record)
    elif corruption == "missing_n_detections":
        del run["n_detections"]
    elif corruption == "nonzero_n_detections":
        run["n_detections"][0] = 1
    elif corruption == "summary_count_mismatch":
        summary = dict(run.attrs["summary_statistics"])
        summary["total_detections"] = 1
        run.attrs["summary_statistics"] = summary
    elif corruption == "decode_backend_mismatch":
        run.attrs["decode_backend_effective"] = "decord_cpu"
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)

    with pytest.raises(ValueError, match=message):
        load_persisted_detection_observation_geometry(
            root,
            "detect_runs/empty",
        )


def test_public_detection_loader_forbids_empty_declaration_on_nonempty_output(
    tmp_path,
) -> None:
    root, run, declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=1,
    )
    assert declaration is None
    record = {
        "schema_id": mod.EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID,
        "schema_version": mod.EMPTY_OBSERVATION_DECLARATION_SCHEMA_VERSION,
    }
    declaration_attr = mod.EMPTY_OBSERVATION_DECLARATION_ATTR
    run.attrs[declaration_attr] = record
    run.attrs[f"{declaration_attr}_sha256"] = coordinate_record_sha256(record)

    with pytest.raises(ValueError, match="must not declare"):
        load_persisted_detection_observation_geometry(
            root,
            "detect_runs/nonempty",
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "n_detections",
        "frame_counts",
        "scores_payload",
        "missing_n_detections",
        "missing_scores",
        "missing_cardinality_seal",
        "out_of_domain_frame",
    ],
)
def test_public_detection_loader_rejects_nonempty_cardinality_drift(
    tmp_path,
    corruption: str,
) -> None:
    root, run, _declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=1,
    )
    if corruption == "n_detections":
        run["n_detections"][0] = 0
    elif corruption == "frame_counts":
        run["frame_counts"][0] = 0
    elif corruption == "scores_payload":
        run["scores"][0] = np.float32(0.25)
    elif corruption == "missing_n_detections":
        del run["n_detections"]
    elif corruption == "missing_scores":
        del run["scores"]
    elif corruption == "missing_cardinality_seal":
        del run.attrs[mod.DETECTION_OBSERVATION_CARDINALITY_ATTR]
    elif corruption == "out_of_domain_frame":
        run["frame_indices"][0] = 2
        run["source_acquisition_frame_index"][0] = 2
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)

    with pytest.raises(ValueError):
        load_persisted_detection_observation_geometry(
            root,
            "detect_runs/nonempty",
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("recording_identity", "sealed acquisition identity"),
        ("class_id", "exact detect-origin derivation"),
        ("instance_key", "exact detect-origin derivation"),
    ],
)
def test_public_detection_loader_rejects_unbound_instance_identity(
    tmp_path,
    corruption: str,
    message: str,
) -> None:
    root, run, _declaration = _complete_canonical_detection_observation(
        tmp_path,
        row_count=1,
    )
    if corruption == "recording_identity":
        run.attrs["instance_key_recording_identity"] = "different-recording"
    elif corruption == "class_id":
        run["class_ids"][0] = 7
    elif corruption == "instance_key":
        run["instance_key"][0] = np.uint64(int(run["instance_key"][0]) + 1)
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(corruption)

    with pytest.raises(ValueError, match=message):
        load_persisted_detection_observation_geometry(
            root,
            "detect_runs/nonempty",
        )


def test_detection_regime_uses_typed_publication_and_never_auto_falls_back(
    tmp_path,
) -> None:
    historical = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    historical.create_group("raw_video")
    with pytest.raises(ValueError, match="publication status is missing"):
        mod._load_published_detection_acquisition_frame(historical)  # noqa: SLF001
    root, _zarr_path, _video = _canonical_acquisition_root(tmp_path)

    mode_mismatch = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
        authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-a",
    ).to_dict()
    root.attrs["acquisition_authority_publication_status"] = mode_mismatch
    root["raw_video"].attrs["acquisition_authority_publication_status"] = (
        mode_mismatch
    )
    with pytest.raises(ValueError, match="mode/path disagrees"):
        mod._load_published_detection_acquisition_frame(root)  # noqa: SLF001

    pending = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PENDING,
        reason_code=EXTERNAL_ACQUISITION_PENDING_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-a",
    ).to_dict()
    root.attrs["acquisition_authority_publication_status"] = pending
    root["raw_video"].attrs["acquisition_authority_publication_status"] = pending
    with pytest.raises(ValueError, match="requires a published acquisition"):
        mod._load_published_detection_acquisition_frame(root)  # noqa: SLF001


@pytest.mark.parametrize(
    "failure_stage",
    ["storage", "coordinate_publication", "completion", "pointer_publication"],
)
def test_detection_failure_rolls_back_coordinate_attrs_and_selectors(
    failure_stage: str,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    parent = root.create_group("detect_runs")
    parent.attrs.update(
        {
            "latest": "previous",
            "latest_complete": "previous",
            "authoritative_run": "previous",
        }
    )
    selector_snapshot = mod._snapshot_parent_selectors(parent)  # noqa: SLF001
    run = parent.create_group("candidate")
    mod.mark_run_started(run, run_name="candidate", stage="detect")
    mod.note_pending_latest(parent, "candidate")
    array = run.create_array(
        "bbox_norm_coords",
        data=np.zeros((1, 4), dtype=np.float64),
    )
    run.attrs.update(
        {
            "coordinate_contract": "canonical_v2",
            "detection_bbox_projection": {"partial": failure_stage},
            "detection_bbox_projection_sha256": "a" * 64,
            "row_identity_key": {"partial": failure_stage},
            "palette_run_completed_at_utc": "2026-07-19T00:00:00+00:00",
        }
    )
    array.attrs["coordinate_descriptor"] = {"partial": failure_stage}
    array.attrs["coordinate_descriptor_sha256"] = "b" * 64
    parent.attrs.update(
        {
            "latest": "candidate",
            "latest_complete": "candidate",
            "authoritative_run": "candidate",
        }
    )

    cause = RuntimeError(f"injected {failure_stage} failure")
    mod._rollback_failed_detection_run(  # noqa: SLF001
        run_group=run,
        parent_group=parent,
        run_name="candidate",
        selector_snapshot=selector_snapshot,
        coordinate_checkpoints=(),
        cause=cause,
        error=str(cause),
    )

    assert run.attrs["palette_run_completion_status"] == "failed"
    assert "coordinate_contract" not in run.attrs
    assert "detection_bbox_projection" not in run.attrs
    assert "detection_bbox_projection_sha256" not in run.attrs
    assert "row_identity_key" not in run.attrs
    assert "palette_run_completed_at_utc" not in run.attrs
    assert run.attrs["stage_selector_eligible"] is False
    # Zarr attributes are cached per array handle; reload the persisted node
    # instead of trusting the pre-rollback handle's stale metadata cache.
    reloaded_array = root["detect_runs/candidate/bbox_norm_coords"]
    assert "coordinate_descriptor" not in reloaded_array.attrs
    assert "coordinate_descriptor_sha256" not in reloaded_array.attrs
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["latest_complete"] == "previous"
    assert parent.attrs["authoritative_run"] == "previous"
    assert "latest_pending" not in parent.attrs


def test_real_detect_yolo_attempt_normalizes_boxes_in_validated_result_shape(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        orig_shape=(4, 6),
        boxes=_FakeBoxes([[0.0, 0.0, 6.0, 4.0]]),
    )
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [result],
    )
    call_args["resize_dims"] = [2, 3]
    staged_validation_seen: list[bool] = []

    def validate_staged(root, path, *, require_selector_eligible):
        run = root[path]
        parent = root["detect_runs"]
        assert require_selector_eligible is False
        assert run.attrs["palette_run_completion_status"] == "complete"
        assert run.attrs["stage_selector_eligible"] is False
        assert parent.attrs["latest"] == selectors["latest"]
        assert parent.attrs["latest_complete"] == selectors["latest_complete"]
        assert "latest_pending" not in parent.attrs
        staged_validation_seen.append(True)
        return object()

    original_publish_selection = mod._publish_validated_detection_selection  # noqa: SLF001

    def publish_selection(parent, run, *, run_name):
        assert staged_validation_seen == [True]
        assert cap.release_count == 1
        assert run.attrs["stage_selector_eligible"] is False
        original_publish_selection(parent, run, run_name=run_name)

    monkeypatch.setattr(
        mod,
        "_load_persisted_detection_observation_geometry",
        validate_staged,
    )
    monkeypatch.setattr(
        mod,
        "_publish_validated_detection_selection",
        publish_selection,
    )

    run_name = mod.detect_yolo(**call_args)

    assert run_name == "candidate"
    assert cap.release_count == 1
    root = zarr.open_group(output_path, mode="r", use_consolidated=False)
    run = root["detect_runs/candidate"]
    np.testing.assert_allclose(
        run["bbox_norm_coords"][:],
        np.asarray([[0.5, 0.5, 1.0, 1.0]], dtype=np.float64),
    )
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is True
    assert root["detect_runs"].attrs["latest"] == "candidate"
    assert root["detect_runs"].attrs["latest_complete"] == "candidate"


def test_real_detect_yolo_artifact_uses_only_run_local_row_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        orig_shape=(4, 6),
        boxes=_FakeBoxes([[0.0, 0.0, 6.0, 4.0]]),
    )
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [result],
    )
    del call_args["instance_key_recording_identity"]
    call_args.update(
        {
            "coordinate_contract_mode": "artifact_unbound",
            "output_run_family": "detection_artifact_runs",
        }
    )

    assert mod.detect_yolo(**call_args) == "candidate"

    assert cap.release_count == 1
    root = zarr.open_group(output_path, mode="r", use_consolidated=False)
    run = root["detection_artifact_runs/candidate"]
    assert "instance_key" not in run
    assert not any(str(name).startswith("instance_key_") for name in run.attrs)
    np.testing.assert_array_equal(
        run[mod.DETECTION_ARTIFACT_ROW_ID_ARRAY][:],
        np.asarray([0], dtype=np.uint64),
    )
    assert run.attrs["artifact_row_id_scope"] == "run_local_noncanonical"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs[mod.DETECTION_ARTIFACT_STORAGE_ATTR][
        "canonical_instance_key_present"
    ] is False
    for name, value in selectors.items():
        assert root["detect_runs"].attrs[name] == value


def test_detection_publication_attempt_closes_every_video_backend_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Closable:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    cap = _FakeVideoCapture()
    pynvvc = Closable()
    decord = Closable()
    attempt = mod._DetectionPublicationAttempt()  # noqa: SLF001
    attempt.track_cap(cap)
    attempt.track_pynvvc_reader(pynvvc)
    attempt.track_decord_reader(decord)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    assert attempt.close_video_resources() == []
    assert attempt.close_video_resources() == []
    assert cap.release_count == 1
    assert pynvvc.close_count == 1
    assert decord.close_count == 1


@pytest.mark.parametrize(
    "provided_identity",
    ["wrong-recording", " recording-test "],
    ids=["different", "whitespace_normalization_forbidden"],
)
def test_real_detect_yolo_rejects_caller_instance_identity_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    provided_identity: str,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )
    call_args["instance_key_recording_identity"] = provided_identity

    with pytest.raises(ValueError, match="sealed acquisition recording_id"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_detection_failure_closes_backend_before_failed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    parent = root.create_group("detect_runs")
    original_mark_failed = mod.mark_run_failed

    class Reader:
        def close(self) -> None:
            events.append("close")

    def ordered_mark_failed(*args, **kwargs) -> None:
        events.append("failed")
        original_mark_failed(*args, **kwargs)

    monkeypatch.setattr(mod, "mark_run_failed", ordered_mark_failed)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    @mod._guard_detection_publication_attempt  # noqa: SLF001
    def fail_after_create() -> str:
        attempt = mod._active_detection_publication_attempt()  # noqa: SLF001
        snapshot = mod._snapshot_parent_selectors(parent)  # noqa: SLF001
        attempt.prepare(
            root=root,
            parent_group=parent,
            run_name="candidate",
            selector_snapshot=snapshot,
        )
        attempt.claim_named_child()
        run = parent.create_group("candidate")
        attempt.bind_run(run)
        attempt.track_decord_reader(Reader())
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        fail_after_create()

    assert events == ["close", "failed"]
    assert "candidate" not in parent


@pytest.mark.parametrize("failure_stage", ["decode", "predict"])
def test_real_detect_yolo_attempt_rolls_back_early_failures_and_closes_capture(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    predict_error = RuntimeError("injected predict failure")

    def predict(_frames):
        if failure_stage == "predict":
            raise predict_error
        return [SimpleNamespace(orig_shape=(4, 6), boxes=None)]

    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=predict,
        decode_error=(
            RuntimeError("injected decode failure")
            if failure_stage == "decode"
            else None
        ),
    )

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([], "result cardinality mismatch"),
        (
            [SimpleNamespace(orig_shape=(5, 6), boxes=None)],
            "result orig_shape mismatch",
        ),
    ],
)
def test_real_detect_yolo_attempt_rejects_result_cardinality_and_orig_shape(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    results: list[object],
    message: str,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: results,
    )

    with pytest.raises(ValueError, match=message):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_real_detect_yolo_attempt_rolls_back_completion_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )
    original_mark_complete = mod.mark_run_complete

    def fail_after_completion(*args, **kwargs):
        original_mark_complete(*args, **kwargs)
        raise RuntimeError("injected completion failure")

    monkeypatch.setattr(mod, "mark_run_complete", fail_after_completion)

    with pytest.raises(RuntimeError, match="injected completion failure"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("injected fresh reload failure"),
        KeyboardInterrupt("injected fresh reload interrupt"),
        SystemExit("injected fresh reload exit"),
    ],
    ids=["runtime", "keyboard_interrupt", "system_exit"],
)
def test_real_detect_yolo_attempt_rolls_back_fresh_reload_baseexception(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )

    def fail_fresh_reload(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        mod,
        "_load_persisted_detection_observation_geometry",
        fail_fresh_reload,
    )

    with pytest.raises(type(failure), match="injected fresh reload"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_real_detect_yolo_rolls_back_partial_selector_commit_on_system_exit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )

    def interrupt_selector_commit(parent, run, *, run_name):
        assert cap.release_count == 1
        assert run.attrs["stage_selector_eligible"] is False
        parent.attrs["latest_complete"] = run_name
        raise SystemExit("injected selector commit exit")

    monkeypatch.setattr(
        mod,
        "_publish_validated_detection_selection",
        interrupt_selector_commit,
    )

    with pytest.raises(SystemExit, match="injected selector commit exit"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_real_detect_yolo_publishes_verified_empty_observation_declaration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args, cap, output_path, _selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )

    assert mod.detect_yolo(**call_args) == "candidate"

    assert cap.release_count == 1
    run = zarr.open_group(
        output_path, mode="r", use_consolidated=False
    )["detect_runs/candidate"]
    assert run.attrs[mod.OBSERVATION_ROW_COUNT_ATTR] == 0
    declaration = run.attrs[mod.EMPTY_OBSERVATION_DECLARATION_ATTR]
    assert declaration["schema_id"] == mod.EMPTY_OBSERVATION_DECLARATION_SCHEMA_ID
    assert declaration["is_empty"] is True
    assert declaration["observation_row_count"] == 0
    assert declaration["decoded_frame_count"] == 1
    assert declaration["source_total_frames"] == 1
    assert declaration["decode_domain_proof"] == (
        "opencv_stream_eof_and_exact_count_v1"
    )
    assert declaration["proof"] == (
        "full_acquisition_domain_processed_with_zero_observations_v1"
    )
    assert set(declaration["frame_count_arrays"]) == {
        "frame_counts",
        "n_detections",
    }
    assert declaration["count_authorities"]["observation_row_count"] == 0
    assert len(run.attrs[f"{mod.EMPTY_OBSERVATION_DECLARATION_ATTR}_sha256"]) == 64
    assert run["instance_key"].shape == (0,)
    assert run["frame_indices"].shape == (0,)
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is True


def test_real_detect_yolo_keyboard_interrupt_closes_and_rolls_back(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_frames):
        raise KeyboardInterrupt("injected interrupt")

    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_detection_frame_evidence_keyboard_interrupt_restores_shared_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    run = root.require_group("detect_runs").create_group("candidate")
    acquisition = SimpleNamespace(record=SimpleNamespace(camera_id="camera-a"))

    def interrupt_source_camera(node, **_kwargs):
        node.attrs["pixel_frame_authority"] = {"partial": True}
        node.attrs["pixel_frame_authority_sha256"] = "a" * 64
        raise KeyboardInterrupt("coordinate interrupt")

    monkeypatch.setattr(
        mod,
        "stamp_source_camera_pixel_frame_authority",
        interrupt_source_camera,
    )

    with pytest.raises(KeyboardInterrupt, match="coordinate interrupt"):
        mod._publish_detection_frame_evidence(  # noqa: SLF001
            root,
            run,
            acquisition_frame=acquisition,
        )

    camera = root["analysis/coordinate_frames/source_camera/camera-a/continuous"]
    assert dict(camera.attrs) == {}


def test_decord_short_batch_fails_transaction_and_closes_reader(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShortDecordReader:
        def __init__(self) -> None:
            self.close_count = 0

        def __len__(self) -> int:
            return 1

        def get_batch(self, _indices):
            return np.zeros((0, 4, 6, 3), dtype=np.uint8)

        def close(self) -> None:
            self.close_count += 1

    reader = ShortDecordReader()
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [],
    )
    call_args["decode_backend"] = "decord_cpu"
    monkeypatch.setattr(
        mod,
        "_init_decord_reader",
        lambda *_args, **_kwargs: {
            "reader": reader,
            "type": "decord_cpu",
            "on_gpu": False,
            "width": 6,
            "height": 4,
            "fps": 10.0,
        },
    )

    with pytest.raises(ValueError, match="decoded batch cardinality mismatch"):
        mod.detect_yolo(**call_args)

    assert reader.close_count == 1
    assert cap.release_count == 0
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_full_decode_proof_rejects_trailing_pynvvc_output() -> None:
    with pytest.raises(ValueError, match="beyond the declared acquisition domain"):
        mod._finalize_full_decode_domain_proof(  # noqa: SLF001
            backend=mod.BACKEND_PYNVVC_LUMA_RGB,
            processed_frame_count=1,
            expected_frame_count=1,
            pynvvc_frame_iter=iter([object()]),
        )
    assert mod._finalize_full_decode_domain_proof(  # noqa: SLF001
        backend=mod.BACKEND_PYNVVC_LUMA_RGB,
        processed_frame_count=1,
        expected_frame_count=1,
        pynvvc_frame_iter=iter(()),
    ) == "pynvvc_exact_count_and_eof_probe_v1"


def test_partial_create_group_failure_reopens_and_deletes_named_child(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )
    root = zarr.open_group(output_path, mode="a", use_consolidated=False)
    persisted_parent = root["detect_runs"]

    class PartialCreateParent:
        def __init__(self, parent) -> None:
            self.parent = parent
            self.attrs = parent.attrs

        def __contains__(self, name) -> bool:
            return name in self.parent

        def __getitem__(self, name):
            return self.parent[name]

        def __delitem__(self, name) -> None:
            del self.parent[name]

        def keys(self):
            return self.parent.keys()

        def create_group(self, name):
            self.parent.create_group(name)
            raise RuntimeError("injected partial create failure")

    proxy = PartialCreateParent(persisted_parent)
    monkeypatch.setattr(mod, "require_runs_parent", lambda *_args, **_kwargs: proxy)

    with pytest.raises(RuntimeError, match="injected partial create failure"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_decord_initialization_failure_closes_partial_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenReader:
        def __init__(self) -> None:
            self.close_count = 0

        def __getitem__(self, _index):
            raise RuntimeError("injected first-frame failure")

        def close(self) -> None:
            self.close_count += 1

    reader = BrokenReader()
    monkeypatch.setattr(mod, "_decord_available", lambda: True)
    monkeypatch.setattr(
        mod,
        "decord",
        SimpleNamespace(bridge=SimpleNamespace(set_bridge=lambda _name: None)),
    )
    monkeypatch.setattr(mod, "VideoReader", lambda *_args, **_kwargs: reader)
    monkeypatch.setattr(mod, "cpu", lambda: object())

    with pytest.raises(RuntimeError, match="injected first-frame failure"):
        mod._init_decord_reader(  # noqa: SLF001
            "source.mp4",
            prefer_gpu=False,
            console=SimpleNamespace(print=lambda *_args, **_kwargs: None),
        )

    assert reader.close_count == 1


def test_unopened_opencv_metadata_capture_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedCapture:
        def __init__(self) -> None:
            self.release_count = 0

        def isOpened(self) -> bool:  # noqa: N802 - OpenCV API
            return False

        def release(self) -> None:
            self.release_count += 1

    cap = ClosedCapture()
    monkeypatch.setattr(mod.cv2, "VideoCapture", lambda _path: cap)

    with pytest.raises(ValueError, match="Could not open video metadata"):
        mod._read_cv2_video_properties("source.mp4")  # noqa: SLF001

    assert cap.release_count == 1


def test_failed_run_deletion_fallback_disarms_stale_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    parent = root.create_group("detect_runs")
    parent.attrs.update(
        {
            "latest": "candidate",
            "latest_complete": "candidate",
            "latest_pending": "candidate",
            "authoritative_run": "candidate",
            "authoritative_run_provenance": {"approved_by": "stale"},
        }
    )
    snapshot = mod._snapshot_parent_selectors(parent)  # noqa: SLF001
    run = parent.create_group("candidate")
    run.attrs["coordinate_contract"] = "canonical_v2"
    run.attrs["stage_selector_eligible"] = True
    run.create_array("bbox_norm_coords", data=np.zeros((0, 4), dtype=np.float64))
    monkeypatch.setattr(mod, "_delete_new_failed_run", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="failed run deletion"):
        mod._rollback_failed_detection_run(  # noqa: SLF001
            run_group=run,
            parent_group=parent,
            run_name="candidate",
            selector_snapshot=snapshot,
            coordinate_checkpoints=(),
            cause=RuntimeError("injected failure"),
            error="injected failure",
            delete_run_when_safe=True,
        )

    assert "candidate" in parent
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["stage_selector_eligible"] is False
    assert "coordinate_contract" not in run.attrs
    for name in ("latest", "latest_complete", "latest_pending", "authoritative_run"):
        assert parent.attrs.get(name) != "candidate"
    assert "authoritative_run_provenance" not in parent.attrs


def test_stale_selector_collision_is_rejected_before_run_creation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_args, cap, output_path, selectors = _install_detect_yolo_producer_harness(
        monkeypatch,
        tmp_path,
        predict=lambda _frames: [SimpleNamespace(orig_shape=(4, 6), boxes=None)],
    )
    root = zarr.open_group(output_path, mode="a", use_consolidated=False)
    root["detect_runs"].attrs["authoritative_run"] = "candidate"
    selectors["authoritative_run"] = "candidate"

    with pytest.raises(ValueError, match="already referenced by parent selectors"):
        mod.detect_yolo(**call_args)

    assert cap.release_count == 1
    _assert_failed_producer_attempt_rolled_back(output_path, selectors)


def test_audit_detection_runs_reports_exact_parity_and_physical_counts(
    tmp_path,
) -> None:
    zarr_path = tmp_path / "detect_ab.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    parent = root.create_group("detect_runs")
    regular = parent.create_group("regular")
    sharded = parent.create_group("sharded")
    values = _detection_values()

    mod._write_detection_output_arrays(  # noqa: SLF001
        regular,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )
    mod._write_detection_output_arrays(  # noqa: SLF001
        sharded,
        **values,
        det_chunk=4,
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )

    report = audit_detection_runs(
        zarr_path, regular_run="regular", sharded_run="sharded"
    )

    assert report["all_arrays_exact"] is True
    assert all(item["exact"] for item in report["arrays"].values())
    assert (
        report["regular_physical"]["payload_files"]
        > report["sharded_physical"]["payload_files"]
    )


def test_replay_detection_run_as_sharded_uses_production_writer(tmp_path) -> None:
    zarr_path = tmp_path / "detect_replay.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    parent = root.create_group("detect_runs")
    regular = parent.create_group("regular")
    values = _detection_values()
    mod._write_detection_output_arrays(  # noqa: SLF001
        regular,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )

    summary = replay_detection_run_as_sharded(
        zarr_path,
        source_run="regular",
        destination_run="replay",
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )
    report = audit_detection_runs(
        zarr_path, regular_run="regular", sharded_run="replay"
    )

    assert summary["exact_match"] is True
    assert report["all_arrays_exact"] is True
    assert root["detect_runs/replay"].attrs["benchmark_only"] is True
    assert root["detect_runs/replay/bbox_norm_coords"].shards == (8, 4)
