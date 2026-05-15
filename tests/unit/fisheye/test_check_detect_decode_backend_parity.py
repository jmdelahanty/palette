from __future__ import annotations

import json
from pathlib import Path

import scripts.check_detect_decode_backend_parity as mod


def _payload(**comparison_overrides: object) -> dict[str, object]:
    comparison = {
        "frames_compared": 2,
        "detections_a": 2,
        "detections_b": 2,
        "count_mismatch_frames": 0,
        "count_exact_match_fraction": 1.0,
        "bbox_abs_diff_max": 0.0,
        "bbox_abs_diff_mean": 0.0,
        "score_abs_diff_max": 0.0,
        "score_abs_diff_mean": 0.0,
        "class_mismatches": 0,
    }
    comparison.update(comparison_overrides)
    return {
        "status": "ok",
        "canonical_outputs_written": False,
        "backend_a": "decord_gpu",
        "backend_b": "pynvvc_luma_rgb",
        "device": "cuda",
        "fp16": True,
        "frames": [0, 100],
        "comparison": comparison,
        "backend_results": {
            "a": {"decode": {"decode_seconds": 1.0, "preprocess_seconds": 0.1}, "inference_seconds": 0.2},
            "b": {"decode": {"decode_seconds": 0.5, "preprocess_seconds": 0.1}, "inference_seconds": 0.2},
        },
    }


def test_check_detect_decode_backend_parity_accepts_valid_report(tmp_path: Path, capsys) -> None:
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    rc = mod.main([str(path)])

    assert rc == 0
    assert "validation: ok" in capsys.readouterr().out


def test_check_detect_decode_backend_parity_fails_count_mismatch(tmp_path: Path, capsys) -> None:
    path = tmp_path / "parity.json"
    path.write_text(json.dumps(_payload(count_mismatch_frames=1)), encoding="utf-8")

    rc = mod.main([str(path)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "validation: failed" in out
    assert "count_mismatch_frames is 1" in out
