from __future__ import annotations

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


def _write_base(path: Path) -> None:
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
    raw.create_array(
        "images_full",
        data=np.zeros((3, 90, 120), dtype=np.uint8),
        chunks=(1, 90, 120),
    )
    raw.create_array(
        "images_ds",
        data=np.zeros((3, 30, 40), dtype=np.uint8),
        chunks=(1, 30, 40),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 5, 10], dtype=np.int32),
        chunks=(3,),
    )


def _spec(tmp_path: Path) -> prediction.ModelInputSpec:
    model = tmp_path / "model.pt"
    model.write_text("fake", encoding="utf-8")
    return prediction.ModelInputSpec(
        artifact_kind="training",
        run_id="detect-model-v1",
        set_id="detect-set-v1",
        task_type="detect",
        artifact_path=str(model),
        input_shape=[1, 3, 64, 64],
        input_layout="NCHW",
        input_channels=3,
        img_h=64,
        img_w=64,
        max_batch=1,
        dynamic_shapes=False,
        input_dtype="float32",
        input_color_space="RGB",
        input_shape_source="test",
        input_shape_status="explicit",
    )


def _run_artifact(monkeypatch, path: Path, tmp_path: Path, *, run_id: str) -> None:
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
        spec=_spec(tmp_path),
        run_name=run_id,
        batch_size=2,
        conf=0.4,
        iou=0.45,
        max_det=20,
        cpu=True,
        overwrite=False,
        argv=["predict_training_detections"],
    )
    assert result["frame_source"]["path"] == "raw_video/images_full"


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
    root_metadata = (target / "zarr.json").read_text(encoding="utf-8")
    assert '"consolidated_metadata":null' in root_metadata.replace(" ", "")
    assert validate_sampled_training_detection_run(
        target / "detect_runs" / "detect-review-b",
        archive=target,
    )["valid"]
