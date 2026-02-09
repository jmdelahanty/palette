#!/usr/bin/env python3
"""Backward-compatible alias for fisheye.training.train_pose."""

from fisheye.training.train_pose import *  # noqa: F401,F403
from fisheye.training.train_pose import (  # noqa: F401
    _build_default_run_name,
    _infer_set_slug,
    _load_manifest_set_id,
    _record_registry_training_run,
    _snapshot_training_inputs,
    _strip_manifest_suffixes,
)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from fisheye.training.train_pose import main

    parser = argparse.ArgumentParser(
        description="Compatibility alias for fisheye.training.train_pose.",
    )
    parser.add_argument("config_path", type=str, help="Path to the pose training configuration YAML")
    parser.add_argument("--run-name", type=str, help="Optional name for the training run directory")
    parser.add_argument(
        "--project",
        type=str,
        help="Optional output project directory for Ultralytics runs (overrides config/default).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Optional manifest JSON path to record in the registry.",
    )
    parser.add_argument(
        "--set-id",
        type=str,
        help="Optional training set ID to associate with this run. Defaults to manifest set_id when available.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path.",
    )
    parser.add_argument(
        "--log-registry",
        dest="log_registry",
        action="store_true",
        default=True,
        help="Record this training run in the registry (default: enabled).",
    )
    parser.add_argument(
        "--no-log-registry",
        dest="log_registry",
        action="store_false",
        help="Disable registry logging for this training run.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the trained pose model to ONNX.",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=13,
        help="ONNX opset to use for pose export.",
    )
    parser.add_argument(
        "--onnx-simplify",
        action="store_true",
        help="Run ONNX simplification after export.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        help="Optional existing ONNX path to reuse (skips ONNX export).",
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Export the trained pose model to TensorRT (implies ONNX).",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode (default: fp16).",
    )
    parser.add_argument("--trtexec", type=str, help="Optional path to trtexec binary.")
    parser.add_argument(
        "--trt-cuda-graph",
        action="store_true",
        help="Enable CUDA graph in TensorRT build.",
    )
    parser.add_argument(
        "--trt-profiling",
        action="store_true",
        help="Enable TensorRT profiling verbosity.",
    )
    parser.add_argument(
        "--trt-verbose",
        action="store_true",
        help="Enable verbose TensorRT builder logging.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
