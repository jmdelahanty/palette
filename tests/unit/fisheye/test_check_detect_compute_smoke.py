from __future__ import annotations

import json
from pathlib import Path

from scripts import check_detect_compute_smoke as mod


def _write_smoke_json(path: Path, **overrides) -> None:
    payload = {
        "status": "ok",
        "canonical_outputs_written": False,
        "inputs": {
            "decode_backend": "decord_gpu",
            "device": "cuda",
            "fp16": True,
            "resize": [640, 640],
            "resize_source": "config_detection_resize_dims",
            "video_path": "/data/video.mp4",
            "model_path": "/models/best.pt",
        },
        "environment": {"cuda_device_name": "NVIDIA L4"},
        "stages": {
            "video_open_seconds": 1.0,
            "model_load_seconds": 2.0,
        },
        "summary": {
            "frames_processed": 4,
            "batches_processed": 1,
            "detections_total": 2,
            "decode_seconds_total": 0.5,
            "preprocess_seconds_total": 0.2,
            "inference_seconds_total": 0.1,
            "total_seconds": 4.0,
            "end_to_end_fps": 1.0,
            "inference_fps": 40.0,
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_detect_compute_smoke_accepts_valid_report(tmp_path: Path, capsys) -> None:
    report = tmp_path / "smoke.json"
    _write_smoke_json(report)

    rc = mod.main([str(report)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "validation: ok" in out
    assert "canonical_outputs_written: False" in out
    assert "backend: decord_gpu" in out


def test_check_detect_compute_smoke_rejects_canonical_write(tmp_path: Path, capsys) -> None:
    report = tmp_path / "smoke.json"
    _write_smoke_json(report, canonical_outputs_written=True)

    rc = mod.main([str(report)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "validation: failed" in out
    assert "canonical_outputs_written is not false" in out


def test_check_detect_compute_smoke_can_emit_json_summary(tmp_path: Path, capsys) -> None:
    report = tmp_path / "smoke.json"
    _write_smoke_json(report)

    rc = mod.main([str(report), "--json"])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["frames_processed"] == 4
    assert summary["inference_fps"] == 40.0
