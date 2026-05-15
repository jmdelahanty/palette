from __future__ import annotations

import json
from pathlib import Path

from scripts import report_detect_compute_smokes as mod


def _write_report(path: Path, *, backend: str, fps: float, total_s: float) -> None:
    payload = {
        "status": "ok",
        "canonical_outputs_written": False,
        "inputs": {
            "decode_backend": backend,
            "device": "cuda",
            "pipeline_mode": "sequential",
        },
        "cluster": {
            "HOSTNAME": "h08u04",
            "LSB_JOBID": path.stem,
        },
        "stages": {"video_open_seconds": 1.0},
        "summary": {
            "frames_processed": 160,
            "batches_processed": 10,
            "detections_total": 160,
            "total_seconds": total_s,
            "end_to_end_fps": fps,
            "inference_fps": 500.0,
            "decode_seconds_total": 2.0,
            "preprocess_seconds_total": 0.5,
            "predict_return_seconds_total": 1.0,
            "first_batch": {"predict_return_seconds": 0.2},
            "steady_state_excluding_first_batch": {"inference_fps": 700.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_detect_compute_smokes_sorts_by_end_to_end_fps(
    tmp_path: Path, capsys
) -> None:
    slow = tmp_path / "decord.json"
    fast = tmp_path / "pynvvc.json"
    _write_report(slow, backend="decord_gpu", fps=5.0, total_s=32.0)
    _write_report(fast, backend="pynvvc_nv12_rgb", fps=50.0, total_s=3.2)

    rc = mod.main([str(slow), str(fast)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "| ok | backend | pipeline | frames |" in out
    assert out.index("pynvvc_nv12_rgb") < out.index("decord_gpu")
    assert "| yes | pynvvc_nv12_rgb | sequential | 160 | 3.20 | 50.00 |" in out


def test_report_detect_compute_smokes_json_marks_failed_rows(
    tmp_path: Path, capsys
) -> None:
    report = tmp_path / "bad.json"
    _write_report(report, backend="opencv", fps=5.0, total_s=32.0)

    rc = mod.main([str(report), "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"][0]["ok"] is False
    assert "decode_backend is 'opencv'" in payload["rows"][0]["failures"][0]
