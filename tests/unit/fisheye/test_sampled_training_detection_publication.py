from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import zarr

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.sampled_training_detection_publication import (
    build_sampled_training_detection_candidate,
    publish_detection_artifact_run,
    publish_sampled_training_detection_candidate,
    validate_sampled_training_detection_run,
)
from fisheye.shared.zarr.training_image_storage import (
    SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID,
    sampled_training_downsample_transform,
)
from fisheye.utils import predict_training_detections as prediction


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _Boxes:
    def __init__(self, index: int) -> None:
        count = 2 if index == 2 else 1
        rows = np.asarray(
            [[10.0, 20.0, 30.0, 40.0], [35.0, 25.0, 50.0, 45.0]],
            dtype=np.float32,
        )[:count]
        self.xyxy = _Tensor(rows)
        self.conf = _Tensor(np.asarray([0.9, 0.7], dtype=np.float32)[:count])
        self.cls = _Tensor(np.zeros((count,), dtype=np.float32))

    def __len__(self) -> int:
        return int(self.xyxy.values.shape[0])


class _FakeYOLO:
    def __init__(self, _path: str) -> None:
        self.next_index = 0

    def to(self, _device: str) -> None:
        return None

    def predict(self, images, **_kwargs):
        out = []
        for _image in images:
            out.append(SimpleNamespace(boxes=_Boxes(self.next_index)))
            self.next_index += 1
        return out


def _write_base(path: Path, *, ds_hw: tuple[int, int] = (60, 80)) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "zarr_use": "training",
            "recording_id": "recording-a",
            "source_video_width": 120,
            "source_video_height": 90,
        }
    )
    raw = root.create_group("raw_video")
    raw.attrs.update(
        {
            "source_frame_count": 20,
            "source_video_width": 120,
            "source_video_height": 90,
            "import_mode": "sampled",
            "import_purpose": "training_data",
            "frame_step": 5,
        }
    )
    transform = sampled_training_downsample_transform(
        source_hw=(90, 120),
        target_hw=ds_hw,
        method="area",
        preserve_aspect=True,
    )
    raw.attrs["downsample_transform"] = transform
    raw.create_array(
        "images_full",
        data=np.zeros((3, 90, 120), dtype=np.uint8),
        chunks=(1, 90, 120),
    )
    images_ds = raw.create_array(
        "images_ds",
        data=np.zeros((3, *ds_hw), dtype=np.uint8),
        chunks=(1, *ds_hw),
    )
    images_ds.attrs.update(
        {
            "storage_contract_schema_id": SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID,
            "access_unit": "complete_sampled_frame",
            "source_to_stored_transform": transform,
        }
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 5, 10], dtype=np.int32),
        chunks=(3,),
    )


def _spec(
    tmp_path: Path,
    *,
    img_h: int = 64,
    img_w: int = 64,
) -> prediction.ModelInputSpec:
    model = tmp_path / "model.pt"
    model.write_text("fake", encoding="utf-8")
    return prediction.ModelInputSpec(
        artifact_kind="training",
        run_id="detect-model-v1",
        set_id="detect-set-v1",
        task_type="detect",
        artifact_path=str(model),
        input_shape=[1, 3, img_h, img_w],
        input_layout="NCHW",
        input_channels=3,
        img_h=img_h,
        img_w=img_w,
        max_batch=1,
        dynamic_shapes=False,
        input_dtype="float32",
        input_color_space="RGB",
        input_shape_source="test",
        input_shape_status="explicit",
    )


def _run_artifact(
    monkeypatch,
    path: Path,
    tmp_path: Path,
    *,
    run_id: str,
    spec: prediction.ModelInputSpec | None = None,
    expected_frame_source: str = "raw_video/images_full",
) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr(
        prediction,
        "get_git_info",
        lambda: {
            "commit_hash": "abc",
            "short_hash": "abc",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        prediction,
        "get_environment_info",
        lambda **_kwargs: {
            "platform": {
                "hostname": "host",
                "system": "Linux",
                "release": "test",
                "python_version": "3.11",
                "machine": "x86_64",
            },
            "environment": {},
        },
    )
    result = prediction.run_training_zarr_prediction(
        zarr_path=path,
        spec=spec or _spec(tmp_path),
        run_name=run_id,
        batch_size=2,
        conf=0.4,
        iou=0.45,
        max_det=20,
        cpu=True,
        overwrite=False,
        argv=["predict_training_detections"],
    )
    assert result["frame_source"]["path"] == expected_frame_source


def test_zero_padding_images_ds_artifact_binds_to_source_normalized_coordinates(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "training-ds.zarr"
    _write_base(archive)
    _run_artifact(
        monkeypatch,
        archive,
        tmp_path,
        run_id="artifact-ds",
        spec=_spec(tmp_path, img_h=60, img_w=80),
        expected_frame_source="raw_video/images_ds",
    )

    artifact = zarr.open_group(
        str(archive / "detection_artifact_runs" / "artifact-ds"),
        mode="r",
        use_consolidated=False,
    )
    artifact_boxes = np.asarray(artifact["bbox_norm_coords"][:])
    candidate = build_sampled_training_detection_candidate(
        archive=archive,
        artifact_run_id="artifact-ds",
        destination=tmp_path / "candidate-ds.zarr",
        run_id="detect-review-ds",
    )

    np.testing.assert_array_equal(
        candidate.arrays["instances/bbox_norm_coords"][:], artifact_boxes
    )
    binding = candidate.manifest["payload"]["source_artifact"]["frame_source_binding"]
    assert binding["artifact_frame_source_path"] == "raw_video/images_ds"
    assert binding["coordinate_mapping"] == (
        "source_camera_normalized_identity_via_zero_padding_resize"
    )
    assert binding["source_shape_hw"] == [90, 120]
    assert binding["stored_shape_hw"] == [60, 80]


def test_images_ds_binding_rejects_missing_or_padded_transform(
    tmp_path: Path, monkeypatch
) -> None:
    missing = tmp_path / "missing-transform.zarr"
    _write_base(missing)
    _run_artifact(
        monkeypatch,
        missing,
        tmp_path,
        run_id="artifact-missing",
        spec=_spec(tmp_path, img_h=60, img_w=80),
        expected_frame_source="raw_video/images_ds",
    )
    mutable = zarr.open_group(str(missing), mode="a", use_consolidated=False)
    del mutable["raw_video/images_ds"].attrs["source_to_stored_transform"]
    try:
        build_sampled_training_detection_candidate(
            archive=missing,
            artifact_run_id="artifact-missing",
            destination=tmp_path / "candidate-missing.zarr",
            run_id="detect-review-missing",
        )
    except ValueError as exc:
        assert "source_to_stored_transform" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Missing images_ds transform unexpectedly bound.")

    padded = tmp_path / "padded-transform.zarr"
    _write_base(padded, ds_hw=(64, 64))
    _run_artifact(
        monkeypatch,
        padded,
        tmp_path,
        run_id="artifact-padded",
        spec=_spec(tmp_path, img_h=64, img_w=64),
        expected_frame_source="raw_video/images_ds",
    )
    try:
        build_sampled_training_detection_candidate(
            archive=padded,
            artifact_run_id="artifact-padded",
            destination=tmp_path / "candidate-padded.zarr",
            run_id="detect-review-padded",
        )
    except ValueError as exc:
        assert "zero-padding resize" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Padded images_ds transform unexpectedly bound.")


def test_images_ds_binding_rejects_tampered_artifact_lineage_digest(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "tampered-lineage.zarr"
    _write_base(archive)
    _run_artifact(
        monkeypatch,
        archive,
        tmp_path,
        run_id="artifact-tampered",
        spec=_spec(tmp_path, img_h=60, img_w=80),
        expected_frame_source="raw_video/images_ds",
    )
    artifact = zarr.open_group(
        str(archive / "detection_artifact_runs" / "artifact-tampered"),
        mode="a",
        use_consolidated=False,
    )
    lineage = dict(artifact.attrs["artifact_frame_source_lineage"])
    lineage["selected_array_path"] = "raw_video/images_full"
    artifact.attrs["artifact_frame_source_lineage"] = lineage
    try:
        build_sampled_training_detection_candidate(
            archive=archive,
            artifact_run_id="artifact-tampered",
            destination=tmp_path / "candidate-tampered.zarr",
            run_id="detect-review-tampered",
        )
    except ValueError as exc:
        assert "digest" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Tampered artifact lineage unexpectedly bound.")


def test_binding_preserves_local_axis_and_mints_acquisition_keys(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "training.zarr"
    _write_base(archive)
    _run_artifact(monkeypatch, archive, tmp_path, run_id="artifact-a")

    candidate = build_sampled_training_detection_candidate(
        archive=archive,
        artifact_run_id="artifact-a",
        destination=tmp_path / "candidate.zarr",
        run_id="detect-review-a",
    )

    frames = np.asarray(candidate.arrays["instances/frame_indices"][:])
    source_frames = np.asarray(
        candidate.arrays["instances/source_acquisition_frame_index"][:]
    )
    boxes = np.asarray(candidate.arrays["instances/bbox_norm_coords"][:])
    classes = np.asarray(candidate.arrays["instances/class_ids"][:])
    keys = np.asarray(candidate.arrays["instances/instance_key"][:])
    np.testing.assert_array_equal(frames, [0, 1, 2, 2])
    np.testing.assert_array_equal(source_frames, [0, 5, 10, 10])
    np.testing.assert_array_equal(
        keys,
        mint_detection_instance_keys(
            recording_identity="recording-a",
            frame_indices=source_frames,
            bbox_norm_coords=boxes,
            class_ids=classes,
        ),
    )
    assert candidate.statistics == {
        "sampled_frames": 3,
        "detection_rows": 4,
        "frames_with_zero_detections": 0,
        "frames_with_one_detection": 2,
        "frames_with_multiple_detections": 1,
        "one_fish_per_frame_expectation_pass": False,
        "cardinality_policy": (
            "report_exact_detector_evidence_do_not_force_top_one_at_inference"
        ),
    }
    assert validate_sampled_training_detection_run(
        candidate.run_path,
        archive=archive,
    )["valid"]

    mutable = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    mutable["raw_video/original_frame_indices"][1] = np.int32(6)
    invalid = validate_sampled_training_detection_run(
        candidate.run_path,
        archive=archive,
    )
    assert not invalid["valid"]


def test_artifact_and_bound_seed_publish_without_selectors_or_consolidation(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "target.zarr"
    local = tmp_path / "local.zarr"
    _write_base(target)
    shutil.copytree(target, local)
    _run_artifact(monkeypatch, local, tmp_path, run_id="artifact-b")

    artifact_result = publish_detection_artifact_run(
        local_archive=local,
        target_archive=target,
        artifact_run_id="artifact-b",
    )
    assert artifact_result["schema_id"] == (
        "palette.sampled_training_detection.publication"
    )
    candidate = build_sampled_training_detection_candidate(
        archive=target,
        artifact_run_id="artifact-b",
        destination=tmp_path / "candidate-b.zarr",
        run_id="detect-review-b",
    )
    result = publish_sampled_training_detection_candidate(
        candidate,
        archive=target,
    )

    root = zarr.open_group(str(target), mode="r", use_consolidated=False)
    assert result["stage_selector_eligible"] is False
    assert result["root_consolidation"].startswith("deferred")
    assert "artifact-b" in root["detection_artifact_runs"]
    assert "detect-review-b" in root["detect_runs"]
    for parent_name in ("detection_artifact_runs", "detect_runs"):
        parent = root[parent_name]
        for selector in ("authoritative_run", "latest_complete", "latest"):
            assert selector not in parent.attrs
    root_metadata = json.loads((target / "zarr.json").read_text(encoding="utf-8"))
    assert root_metadata.get("consolidated_metadata") is None
    assert validate_sampled_training_detection_run(
        target / "detect_runs" / "detect-review-b",
        archive=target,
    )["valid"]
