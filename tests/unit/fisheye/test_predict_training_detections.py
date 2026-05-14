import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.utils import predict_training_detections as mod


def _write_training_zarr(path: Path, *, ds_shape: tuple[int, int] = (640, 640)) -> None:
    root = zarr.open_group(str(path), mode="w")
    raw = root.require_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.zeros((3, 900, 1200), dtype=np.uint8),
        chunks=(1, 900, 1200),
    )
    raw.create_array(
        "images_ds",
        data=np.zeros((3, ds_shape[0], ds_shape[1]), dtype=np.uint8),
        chunks=(1, ds_shape[0], ds_shape[1]),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.array([0, 5000, 10000], dtype=np.int64),
        chunks=(3,),
    )


def _write_registry(path: Path, model_path: Path, *, imgsz: int = 640) -> None:
    model_path.write_text("fake model", encoding="utf-8")
    registry = Registry(path)
    try:
        registry.record_training_run(
            run_id="detect_run",
            set_id="detect_set",
            task_type="detect",
            config_path=None,
            manifest_path=None,
            model_path=model_path,
            metrics_path=None,
            status="success",
            final_metrics={"imgsz_h": imgsz, "imgsz_w": imgsz},
        )
    finally:
        registry.close()


def test_select_frame_source_prefers_downsampled_match(tmp_path: Path) -> None:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=(640, 640))
    _write_registry(registry_path, model_path, imgsz=640)

    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    selection = mod.select_frame_source(root, spec)

    assert selection.path == "raw_video/images_ds"
    assert selection.matches_model_shape is True
    assert selection.needs_gray_to_rgb is True
    assert selection.reason == "sampled_array_matches_model_shape"


def test_select_frame_source_falls_back_to_full_when_sampled_mismatches(tmp_path: Path) -> None:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=(320, 320))
    _write_registry(registry_path, model_path, imgsz=640)

    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )
    root = zarr.open_group(str(zarr_path), mode="r")
    selection = mod.select_frame_source(root, spec)

    assert selection.path == "raw_video/images_full"
    assert selection.matches_model_shape is False
    assert selection.reason == "fallback_to_available_frame_array"


class _Tensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor(np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32))
        self.conf = _Tensor(np.array([0.9], dtype=np.float32))
        self.cls = _Tensor(np.array([1], dtype=np.float32))

    def __len__(self) -> int:
        return 1


class _FakeYOLO:
    calls: list[dict[str, object]] = []

    def __init__(self, path: str) -> None:
        self.path = path

    def to(self, _device: str) -> None:
        return None

    def predict(self, images, **kwargs):
        self.calls.append({"n_images": len(images), "shape": np.asarray(images[0]).shape, "kwargs": kwargs})
        return [SimpleNamespace(boxes=_Boxes()) for _image in images]


def test_run_training_zarr_prediction_writes_detect_run(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "training.zarr"
    registry_path = tmp_path / "registry.sqlite"
    model_path = tmp_path / "best.pt"
    _write_training_zarr(zarr_path, ds_shape=(640, 640))
    _write_registry(registry_path, model_path, imgsz=640)
    spec = mod.resolve_model_input_spec(
        registry_path,
        model_run_id="detect_run",
        model_path=None,
        set_id=None,
        artifact_kind="training",
    )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr(
        mod,
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
        mod,
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

    result = mod.run_training_zarr_prediction(
        zarr_path=zarr_path,
        spec=spec,
        run_name="detect_seed_test",
        batch_size=2,
        conf=0.4,
        iou=0.45,
        max_det=20,
        cpu=True,
        overwrite=False,
        argv=["predict_training_detections"],
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["detect_runs"]["detect_seed_test"]
    assert root["detect_runs"].attrs["latest"] == "detect_seed_test"
    assert run["frame_indices"][:].tolist() == [0, 1, 2]
    assert run["source_frame_indices"][:].tolist() == [0, 5000, 10000]
    assert run["class_ids"][:].tolist() == [1, 1, 1]
    assert run.attrs["detection_source"] == "training_zarr_raw_video"
    assert run.attrs["frame_source_path"] == "raw_video/images_ds"
    assert run.attrs["model_registry_run_id"] == "detect_run"
    assert run.attrs["model_input_shape_status"] == "inferred_from_imgsz"
    assert run.attrs["summary_statistics"]["total_detections"] == 3
    assert result["summary_statistics"]["frames_with_detections"] == 3
    assert _FakeYOLO.calls[0]["shape"] == (640, 640, 3)
