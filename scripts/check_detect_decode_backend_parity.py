#!/usr/bin/env python3
"""Validate and summarize a fixed-frame detect decode-backend parity JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_MAX_BBOX_DIFF = 0.01
DEFAULT_MAX_SCORE_DIFF = 0.05


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_float(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "missing"


def _format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "missing"


def _validate_payload(
    payload: dict[str, Any],
    *,
    max_bbox_diff: Optional[float],
    max_score_diff: Optional[float],
    allow_count_mismatch: bool,
) -> list[str]:
    failures: list[str] = []
    if payload.get("status") != "ok":
        failures.append(f"status is {payload.get('status')!r}, expected 'ok'")
    if payload.get("canonical_outputs_written") is not False:
        failures.append("canonical_outputs_written is not false")

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        failures.append("comparison block missing")
        return failures

    frames_compared = int(comparison.get("frames_compared") or 0)
    if frames_compared <= 0:
        failures.append("frames_compared is missing or zero")

    count_mismatch_frames = int(comparison.get("count_mismatch_frames") or 0)
    if count_mismatch_frames and not allow_count_mismatch:
        failures.append(f"count_mismatch_frames is {count_mismatch_frames}, expected 0")

    bbox_max = comparison.get("bbox_abs_diff_max")
    if max_bbox_diff is not None and isinstance(bbox_max, (int, float)) and float(bbox_max) > max_bbox_diff:
        failures.append(f"bbox_abs_diff_max {bbox_max} exceeds {max_bbox_diff}")

    score_max = comparison.get("score_abs_diff_max")
    if max_score_diff is not None and isinstance(score_max, (int, float)) and float(score_max) > max_score_diff:
        failures.append(f"score_abs_diff_max {score_max} exceeds {max_score_diff}")

    class_mismatches = int(comparison.get("class_mismatches") or 0)
    if class_mismatches:
        failures.append(f"class_mismatches is {class_mismatches}, expected 0")

    return failures


def _print_summary(path: Path, payload: dict[str, Any]) -> None:
    comparison = payload.get("comparison") or {}
    backend_results = payload.get("backend_results") or {}
    result_a = backend_results.get("a") or {}
    result_b = backend_results.get("b") or {}

    print(f"path: {path}")
    print(f"status: {payload.get('status')}")
    print(f"canonical_outputs_written: {payload.get('canonical_outputs_written')}")
    print(f"backend_a: {payload.get('backend_a')}")
    print(f"backend_b: {payload.get('backend_b')}")
    print(f"device: {payload.get('device')} fp16={payload.get('fp16')}")
    print(f"resize: {payload.get('resize')} source={payload.get('resize_source')}")
    print(f"imgsz_applied: {payload.get('imgsz_applied')}")
    print(f"frames: {payload.get('frames')}")
    print(f"video: {payload.get('video_path')}")
    print(f"model: {payload.get('model_path')}")
    print(f"frames_compared: {comparison.get('frames_compared')}")
    print(f"detections_a: {comparison.get('detections_a')}")
    print(f"detections_b: {comparison.get('detections_b')}")
    print(f"count_mismatch_frames: {comparison.get('count_mismatch_frames')}")
    print(f"count_exact_match_fraction: {_format_float(comparison.get('count_exact_match_fraction'))}")
    print(f"bbox_abs_diff_max: {_format_float(comparison.get('bbox_abs_diff_max'), digits=6)}")
    print(f"bbox_abs_diff_mean: {_format_float(comparison.get('bbox_abs_diff_mean'), digits=6)}")
    print(f"score_abs_diff_max: {_format_float(comparison.get('score_abs_diff_max'), digits=6)}")
    print(f"score_abs_diff_mean: {_format_float(comparison.get('score_abs_diff_mean'), digits=6)}")
    print(f"class_mismatches: {comparison.get('class_mismatches')}")
    if comparison.get("first_count_mismatch_frames"):
        print(f"first_count_mismatch_frames: {comparison.get('first_count_mismatch_frames')}")
    print(
        "backend_a_timing: "
        f"decode={_format_seconds(_get_nested(result_a, 'decode', 'decode_seconds'))}s "
        f"preprocess={_format_seconds(_get_nested(result_a, 'decode', 'preprocess_seconds'))}s "
        f"inference={_format_seconds(result_a.get('inference_seconds'))}s"
    )
    print(
        "backend_b_timing: "
        f"decode={_format_seconds(_get_nested(result_b, 'decode', 'decode_seconds'))}s "
        f"preprocess={_format_seconds(_get_nested(result_b, 'decode', 'preprocess_seconds'))}s "
        f"inference={_format_seconds(result_b.get('inference_seconds'))}s"
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a fisheye.diagnostics.compare_detect_decode_backend_predictions JSON report."
    )
    parser.add_argument("json_path", type=Path, help="Path to backend parity JSON report.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    parser.add_argument(
        "--max-bbox-diff",
        type=float,
        default=DEFAULT_MAX_BBOX_DIFF,
        help=f"Maximum normalized bbox coordinate drift (default: {DEFAULT_MAX_BBOX_DIFF}).",
    )
    parser.add_argument(
        "--max-score-diff",
        type=float,
        default=DEFAULT_MAX_SCORE_DIFF,
        help=f"Maximum confidence-score drift (default: {DEFAULT_MAX_SCORE_DIFF}).",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Do not fail when detection counts differ across selected frames.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    path = args.json_path.expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed reading JSON: {path}: {exc}", file=sys.stderr)
        return 2

    failures = _validate_payload(
        payload,
        max_bbox_diff=args.max_bbox_diff,
        max_score_diff=args.max_score_diff,
        allow_count_mismatch=bool(args.allow_count_mismatch),
    )
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    summary = {
        "path": str(path),
        "ok": not failures,
        "failures": failures,
        "status": payload.get("status"),
        "canonical_outputs_written": payload.get("canonical_outputs_written"),
        "backend_a": payload.get("backend_a"),
        "backend_b": payload.get("backend_b"),
        "device": payload.get("device"),
        "fp16": payload.get("fp16"),
        "frames": payload.get("frames"),
        "frames_compared": comparison.get("frames_compared"),
        "detections_a": comparison.get("detections_a"),
        "detections_b": comparison.get("detections_b"),
        "count_mismatch_frames": comparison.get("count_mismatch_frames"),
        "count_exact_match_fraction": comparison.get("count_exact_match_fraction"),
        "bbox_abs_diff_max": comparison.get("bbox_abs_diff_max"),
        "score_abs_diff_max": comparison.get("score_abs_diff_max"),
        "class_mismatches": comparison.get("class_mismatches"),
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(path, payload)
        if failures:
            print("\nvalidation: failed")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("\nvalidation: ok")

    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
