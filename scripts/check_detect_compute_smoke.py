#!/usr/bin/env python3
"""Validate and summarize a detection compute-smoke JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "missing"


def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "missing"


def _validate_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("status") != "ok":
        failures.append(f"status is {payload.get('status')!r}, expected 'ok'")
    if payload.get("canonical_outputs_written") is not False:
        failures.append("canonical_outputs_written is not false")
    if _get_nested(payload, "inputs", "decode_backend") != "decord_gpu":
        failures.append(
            "decode_backend is "
            f"{_get_nested(payload, 'inputs', 'decode_backend')!r}, expected 'decord_gpu'"
        )
    if _get_nested(payload, "inputs", "device") != "cuda":
        failures.append(f"device is {_get_nested(payload, 'inputs', 'device')!r}, expected 'cuda'")
    if int(_get_nested(payload, "summary", "frames_processed") or 0) <= 0:
        failures.append("frames_processed is missing or zero")
    if int(_get_nested(payload, "summary", "batches_processed") or 0) <= 0:
        failures.append("batches_processed is missing or zero")
    return failures


def _print_summary(path: Path, payload: dict[str, Any]) -> None:
    inputs = payload.get("inputs") or {}
    summary = payload.get("summary") or {}
    stages = payload.get("stages") or {}
    env = payload.get("environment") or {}

    print(f"path: {path}")
    print(f"status: {payload.get('status')}")
    print(f"canonical_outputs_written: {payload.get('canonical_outputs_written')}")
    print(f"backend: {inputs.get('decode_backend')}")
    print(f"device: {inputs.get('device')} fp16={inputs.get('fp16')}")
    print(f"cuda_device: {env.get('cuda_device_name')}")
    print(f"resize: {inputs.get('resize')} source={inputs.get('resize_source')}")
    print(f"video: {inputs.get('video_path')}")
    print(f"model: {inputs.get('model_path')}")
    print(f"frames_processed: {summary.get('frames_processed')}")
    print(f"batches_processed: {summary.get('batches_processed')}")
    print(f"detections_total: {summary.get('detections_total')}")
    print(f"video_open_s: {_format_seconds(stages.get('video_open_seconds'))}")
    print(f"model_load_s: {_format_seconds(stages.get('model_load_seconds'))}")
    print(f"decode_total_s: {_format_seconds(summary.get('decode_seconds_total'))}")
    print(f"preprocess_total_s: {_format_seconds(summary.get('preprocess_seconds_total'))}")
    print(f"inference_total_s: {_format_seconds(summary.get('inference_seconds_total'))}")
    print(f"total_s: {_format_seconds(summary.get('total_seconds'))}")
    print(f"end_to_end_fps: {_format_float(summary.get('end_to_end_fps'))}")
    print(f"inference_fps: {_format_float(summary.get('inference_fps'))}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a fisheye.diagnostics.detect_compute_smoke JSON report."
    )
    parser.add_argument("json_path", type=Path, help="Path to compute-smoke JSON report.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable summary instead of human-readable lines.",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Do not fail when the smoke used a non-CUDA device/backend.",
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

    failures = _validate_payload(payload)
    if args.allow_non_cuda:
        failures = [
            failure
            for failure in failures
            if not failure.startswith("decode_backend is ") and not failure.startswith("device is ")
        ]

    if args.json:
        summary = {
            "path": str(path),
            "ok": not failures,
            "failures": failures,
            "status": payload.get("status"),
            "canonical_outputs_written": payload.get("canonical_outputs_written"),
            "decode_backend": _get_nested(payload, "inputs", "decode_backend"),
            "device": _get_nested(payload, "inputs", "device"),
            "frames_processed": _get_nested(payload, "summary", "frames_processed"),
            "detections_total": _get_nested(payload, "summary", "detections_total"),
            "total_seconds": _get_nested(payload, "summary", "total_seconds"),
            "end_to_end_fps": _get_nested(payload, "summary", "end_to_end_fps"),
            "inference_fps": _get_nested(payload, "summary", "inference_fps"),
        }
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
