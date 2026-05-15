#!/usr/bin/env python3
"""Validate and summarize a detection compute-smoke JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

CUDA_DECODE_BACKENDS = {"decord_gpu", "pynvvc_luma_rgb", "pynvvc_nv12_rgb"}


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
    decode_backend = _get_nested(payload, "inputs", "decode_backend")
    if decode_backend not in CUDA_DECODE_BACKENDS:
        failures.append(
            "decode_backend is "
            f"{decode_backend!r}, expected one of {sorted(CUDA_DECODE_BACKENDS)!r}"
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
    stage_spans = payload.get("stage_spans") or {}
    env = payload.get("environment") or {}
    cluster = payload.get("cluster") or {}
    model_optimization = payload.get("model_optimization") or {}
    steady = summary.get("steady_state_excluding_first_batch") or {}
    first_batch = summary.get("first_batch") or {}
    pipeline = summary.get("pipeline") or {}

    print(f"path: {path}")
    print(f"status: {payload.get('status')}")
    print(f"canonical_outputs_written: {payload.get('canonical_outputs_written')}")
    print(f"job_id: {cluster.get('LSB_JOBID')}")
    print(f"host: {cluster.get('HOSTNAME')}")
    print(f"queue: {cluster.get('LSB_QUEUE')} slots={cluster.get('LSB_DJOB_NUMPROC')}")
    print(f"cuda_visible_devices: {cluster.get('CUDA_VISIBLE_DEVICES')}")
    print(f"palette_job_cache: {cluster.get('PALETTE_JOB_CACHE')}")
    print(f"backend: {inputs.get('decode_backend')}")
    if (
        inputs.get("decode_backend_requested")
        and inputs.get("decode_backend_requested") != inputs.get("decode_backend")
    ):
        print(f"backend_requested: {inputs.get('decode_backend_requested')}")
    pipeline_mode = inputs.get("pipeline_mode") or "legacy_sequential"
    pipeline_depth = inputs.get("pipeline_depth") or "n/a"
    timing_policy = inputs.get("timing_policy") or "legacy_per_batch_sync"
    print(
        "pipeline: "
        f"mode={pipeline_mode} depth={pipeline_depth} "
        f"timing={timing_policy}"
    )
    print(f"device: {inputs.get('device')} fp16={inputs.get('fp16')}")
    print(f"cuda_device: {env.get('cuda_device_name')}")
    print(f"resize: {inputs.get('resize')} source={inputs.get('resize_source')}")
    print(f"imgsz_applied: {inputs.get('imgsz_applied')}")
    print(
        "model_optimization: "
        f"channels_last={model_optimization.get('model_channels_last')} "
        f"cudnn_benchmark={model_optimization.get('cudnn_benchmark_enabled')}"
    )
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
    print(f"predict_return_total_s: {_format_seconds(summary.get('predict_return_seconds_total'))}")
    print(f"inference_cuda_sync_total_s: {_format_seconds(summary.get('inference_cuda_sync_seconds_total'))}")
    print(f"total_s: {_format_seconds(summary.get('total_seconds'))}")
    print(f"end_to_end_fps: {_format_float(summary.get('end_to_end_fps'))}")
    print(f"inference_fps: {_format_float(summary.get('inference_fps'))}")
    print(f"first_batch_inference_s: {_format_seconds(first_batch.get('inference_seconds'))}")
    print(f"first_batch_predict_return_s: {_format_seconds(first_batch.get('predict_return_seconds'))}")
    print(f"first_batch_cuda_sync_s: {_format_seconds(first_batch.get('inference_cuda_sync_seconds'))}")
    print(
        "steady_state_excluding_first: "
        f"batches={steady.get('batches_processed')} "
        f"frames={steady.get('frames_processed')} "
        f"inference_fps={_format_float(steady.get('inference_fps'))} "
        f"predict_return_mean_s={_format_seconds(steady.get('predict_return_seconds_mean'))} "
        f"cuda_sync_mean_s={_format_seconds(steady.get('inference_cuda_sync_seconds_mean'))}"
    )
    if pipeline:
        print(
            "pipeline_metrics: "
            f"queue_wait_s={_format_seconds(pipeline.get('queue_wait_seconds_total'))} "
            f"consumer_s={_format_seconds(pipeline.get('consumer_seconds_total'))} "
            f"final_cuda_sync_s={_format_seconds(pipeline.get('final_cuda_sync_seconds'))}"
        )
    total_span = stage_spans.get("total") if isinstance(stage_spans, dict) else None
    if isinstance(total_span, dict):
        print(f"total_start_utc: {total_span.get('start_utc')}")
        print(f"total_end_utc: {total_span.get('end_utc')}")


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
            "decode_backend_requested": _get_nested(
                payload, "inputs", "decode_backend_requested"
            ),
            "pipeline_mode": _get_nested(payload, "inputs", "pipeline_mode"),
            "pipeline_depth": _get_nested(payload, "inputs", "pipeline_depth"),
            "timing_policy": _get_nested(payload, "inputs", "timing_policy"),
            "device": _get_nested(payload, "inputs", "device"),
            "imgsz_applied": _get_nested(payload, "inputs", "imgsz_applied"),
            "job_id": _get_nested(payload, "cluster", "LSB_JOBID"),
            "host": _get_nested(payload, "cluster", "HOSTNAME"),
            "frames_processed": _get_nested(payload, "summary", "frames_processed"),
            "detections_total": _get_nested(payload, "summary", "detections_total"),
            "total_seconds": _get_nested(payload, "summary", "total_seconds"),
            "end_to_end_fps": _get_nested(payload, "summary", "end_to_end_fps"),
            "inference_fps": _get_nested(payload, "summary", "inference_fps"),
            "predict_return_seconds_total": _get_nested(
                payload, "summary", "predict_return_seconds_total"
            ),
            "inference_cuda_sync_seconds_total": _get_nested(
                payload, "summary", "inference_cuda_sync_seconds_total"
            ),
            "first_batch_inference_seconds": _get_nested(
                payload, "summary", "first_batch", "inference_seconds"
            ),
            "first_batch_predict_return_seconds": _get_nested(
                payload, "summary", "first_batch", "predict_return_seconds"
            ),
            "first_batch_cuda_sync_seconds": _get_nested(
                payload, "summary", "first_batch", "inference_cuda_sync_seconds"
            ),
            "steady_state_inference_fps": _get_nested(
                payload,
                "summary",
                "steady_state_excluding_first_batch",
                "inference_fps",
            ),
            "steady_state_predict_return_seconds_mean": _get_nested(
                payload,
                "summary",
                "steady_state_excluding_first_batch",
                "predict_return_seconds_mean",
            ),
            "steady_state_cuda_sync_seconds_mean": _get_nested(
                payload,
                "summary",
                "steady_state_excluding_first_batch",
                "inference_cuda_sync_seconds_mean",
            ),
            "pipeline_queue_wait_seconds_total": _get_nested(
                payload, "summary", "pipeline", "queue_wait_seconds_total"
            ),
            "pipeline_consumer_seconds_total": _get_nested(
                payload, "summary", "pipeline", "consumer_seconds_total"
            ),
            "pipeline_final_cuda_sync_seconds": _get_nested(
                payload, "summary", "pipeline", "final_cuda_sync_seconds"
            ),
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
