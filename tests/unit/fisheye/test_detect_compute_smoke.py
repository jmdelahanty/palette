from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.diagnostics import detect_compute_smoke as mod


class _FakeBoxes:
    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count


class _FakePrediction:
    def __init__(self, box_count: int) -> None:
        self.boxes = _FakeBoxes(box_count)


class _FakeYOLO:
    instances: list["_FakeYOLO"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.device = None
        self.predict_calls = []
        _FakeYOLO.instances.append(self)

    def fuse(self) -> None:
        return None

    def to(self, device: str) -> None:
        self.device = device

    def half(self) -> None:
        raise AssertionError("CPU smoke should not request fp16")

    def predict(self, batch, **kwargs):  # noqa: ANN001, ANN003
        self.predict_calls.append((batch, kwargs))
        return [_FakePrediction(2) for _ in range(int(batch.shape[0]))]


def test_compute_smoke_runs_without_canonical_writes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_QUEUE", "gpu_l4")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PALETTE_JOB_CACHE", str(tmp_path / "cache"))
    video_path = tmp_path / "input.mp4"
    model_path = tmp_path / "best.pt"
    output_json = tmp_path / "smoke.json"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")

    _FakeYOLO.instances = []

    released = []

    monkeypatch.setattr(mod.stage, "YOLO", _FakeYOLO)
    monkeypatch.setattr(mod.stage, "_load_config", lambda _path: {})
    monkeypatch.setattr(mod.stage, "_collect_environment", lambda: {"test": True})
    monkeypatch.setattr(
        mod.stage,
        "_resolve_backend_reader",
        lambda video, backend, start: {"backend": backend, "reader": object()},
    )
    monkeypatch.setattr(mod.stage, "_release_reader", lambda reader_info: released.append(reader_info))

    def _fake_decode(_reader_info, indices):  # noqa: ANN001
        return np.ones((len(indices), 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(mod.stage, "_decode_batch", _fake_decode)

    rc = mod.main(
        [
            str(video_path),
            "--model",
            str(model_path),
            "--decode-backend",
            "decord_cpu",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-batches",
            "1",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    assert released
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["canonical_outputs_written"] is False
    assert payload["canonical_zarr_write_policy"] == "compute_only_no_detect_runs_or_zarr_chunks"
    assert payload["cluster"]["LSB_JOBID"] == "12345"
    assert payload["cluster"]["LSB_QUEUE"] == "gpu_l4"
    assert payload["cluster"]["LSB_DJOB_NUMPROC"] == "8"
    assert payload["cluster"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert payload["stage_spans"]["total"]["start_utc"]
    assert payload["stage_spans"]["total"]["end_utc"]
    assert payload["stage_spans"]["model_load"]["seconds"] >= 0
    assert payload["stage_spans"]["video_open"]["seconds"] >= 0
    assert payload["inputs"]["frames_requested"] == 2
    assert payload["inputs"]["resize_source"] == "none"
    assert payload["inputs"]["imgsz_applied"] is None
    assert payload["model_optimization"]["cudnn_benchmark_enabled"] is None
    assert payload["model_optimization"]["model_channels_last"] is False
    assert payload["summary"]["frames_processed"] == 2
    assert payload["summary"]["detections_total"] == 4
    assert payload["summary"]["first_batch"]["inference_seconds"] >= 0
    assert payload["summary"]["first_batch"]["predict_return_seconds"] >= 0
    assert payload["summary"]["first_batch"]["inference_cuda_sync_seconds"] >= 0
    assert payload["summary"]["steady_state_excluding_first_batch"]["batches_processed"] == 0
    assert payload["batches"][0]["tensor_shape"] == [2, 3, 8, 8]
    assert not (tmp_path / "detect_runs").exists()

    assert len(_FakeYOLO.instances) == 1
    fake_model = _FakeYOLO.instances[0]
    assert fake_model.model_path == str(model_path.resolve())
    assert fake_model.device == "cpu"
    assert fake_model.predict_calls[0][1]["device"] == "cpu"
    assert fake_model.predict_calls[0][1]["half"] is False


def test_compute_smoke_requires_bounded_frame_selection(tmp_path: Path) -> None:
    args = SimpleNamespace(
        video_path=tmp_path / "input.mp4",
        model=tmp_path / "best.pt",
        config=None,
        decode_backend="decord_cpu",
        start_frame=0,
        max_frames=0,
        max_batches=0,
        batch_size=4,
        resize=None,
        conf=None,
        iou=None,
        max_det=None,
        device="cpu",
        force_fp32=True,
    )
    args.video_path.write_bytes(b"video")
    args.model.write_bytes(b"model")

    with pytest.raises(ValueError, match="must be bounded"):
        mod.run_smoke(args)


def test_compute_smoke_uses_detection_resize_dims_when_video_resize_missing(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "input.mp4"
    model_path = tmp_path / "best.pt"
    output_json = tmp_path / "smoke.json"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")

    _FakeYOLO.instances = []

    monkeypatch.setattr(mod.stage, "YOLO", _FakeYOLO)
    monkeypatch.setattr(
        mod.stage,
        "_load_config",
        lambda _path: {
            "detection": {
                "resize_dims": [16, 32],
                "conf_threshold": 0.4,
                "iou_threshold": 0.8,
                "max_det": 1,
            },
            "video": {"resize": None},
        },
    )
    monkeypatch.setattr(mod.stage, "_collect_environment", lambda: {"test": True})
    monkeypatch.setattr(
        mod.stage,
        "_resolve_backend_reader",
        lambda video, backend, start: {"backend": backend, "reader": object()},
    )
    monkeypatch.setattr(mod.stage, "_release_reader", lambda _reader_info: None)
    monkeypatch.setattr(
        mod.stage,
        "_decode_batch",
        lambda _reader_info, indices: np.ones((len(indices), 8, 8, 3), dtype=np.uint8),
    )

    rc = mod.main(
        [
            str(video_path),
            "--model",
            str(model_path),
            "--decode-backend",
            "decord_cpu",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-batches",
            "1",
            "--output-json",
            str(output_json),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["inputs"]["resize"] == [32, 16]
    assert payload["inputs"]["resize_source"] == "config_detection_resize_dims"
    assert payload["inputs"]["imgsz_applied"] == [16, 32]
    assert payload["batches"][0]["tensor_shape"] == [2, 3, 16, 32]
