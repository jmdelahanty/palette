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
            "imgsz_applied": 640,
            "video_path": "/data/video.mp4",
            "model_path": "/models/best.pt",
        },
        "model_optimization": {
            "model_channels_last": True,
            "cudnn_benchmark_enabled": True,
        },
        "environment": {"cuda_device_name": "NVIDIA L4"},
        "cluster": {
            "LSB_JOBID": "149842613",
            "HOSTNAME": "e11u08",
            "LSB_QUEUE": "gpu_l4",
            "LSB_DJOB_NUMPROC": "8",
            "CUDA_VISIBLE_DEVICES": "0",
            "PALETTE_JOB_CACHE": "/scratch/delahantyj/149842613/palette_cache",
        },
        "stage_spans": {
            "total": {
                "start_utc": "2026-05-14T20:00:00+00:00",
                "end_utc": "2026-05-14T20:01:00+00:00",
                "seconds": 60.0,
            },
        },
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
            "predict_return_seconds_total": 0.08,
            "inference_cuda_sync_seconds_total": 0.02,
            "total_seconds": 4.0,
            "end_to_end_fps": 1.0,
            "inference_fps": 40.0,
            "first_batch": {
                "decode_seconds": 0.5,
                "preprocess_seconds": 0.2,
                "inference_seconds": 0.1,
                "predict_return_seconds": 0.08,
                "inference_cuda_sync_seconds": 0.02,
            },
            "steady_state_excluding_first_batch": {
                "batches_processed": 0,
                "frames_processed": 0,
                "inference_fps": None,
                "predict_return_seconds_mean": None,
                "inference_cuda_sync_seconds_mean": None,
            },
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
    assert "job_id: 149842613" in out
    assert "imgsz_applied: 640" in out
    assert "model_optimization: channels_last=True cudnn_benchmark=True" in out
    assert "steady_state_excluding_first:" in out
    assert "backend: decord_gpu" in out


def test_check_detect_compute_smoke_accepts_pynvvc_luma_backend(
    tmp_path: Path, capsys
) -> None:
    report = tmp_path / "smoke.json"
    _write_smoke_json(
        report,
        inputs={
            "decode_backend": "pynvvc_luma_rgb",
            "device": "cuda",
            "fp16": True,
        },
    )

    rc = mod.main([str(report)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "validation: ok" in out
    assert "backend: pynvvc_luma_rgb" in out


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
    assert summary["job_id"] == "149842613"
    assert summary["first_batch_inference_seconds"] == 0.1
    assert summary["first_batch_predict_return_seconds"] == 0.08
    assert summary["first_batch_cuda_sync_seconds"] == 0.02
