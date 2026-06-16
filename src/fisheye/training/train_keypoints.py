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
    parser.add_argument(
        "--onnx-dynamic",
        action="store_true",
        help="Export ONNX with a dynamic batch dimension for TensorRT profile builds.",
    )
    parser.add_argument(
        "--onnx-batch",
        type=int,
        default=1,
        help="Batch dimension to use during ONNX export. For dynamic export this is the nominal export batch.",
    )
    parser.add_argument("--trt-input-name", default="images", help="TensorRT input tensor name for generated shape profiles.")
    parser.add_argument("--trt-min-batch", type=int, help="TensorRT generated profile minimum batch.")
    parser.add_argument("--trt-opt-batch", type=int, help="TensorRT generated profile optimum batch.")
    parser.add_argument("--trt-max-batch", type=int, help="TensorRT generated profile maximum batch.")
    parser.add_argument("--trt-min-shapes", help="Explicit TensorRT minShapes profile string.")
    parser.add_argument("--trt-opt-shapes", help="Explicit TensorRT optShapes profile string.")
    parser.add_argument("--trt-max-shapes", help="Explicit TensorRT maxShapes profile string.")
    parser.add_argument(
        "--trt-builder-optimization-level",
        type=int,
        choices=range(0, 6),
        metavar="{0..5}",
        help="TensorRT builder effort level. trtexec defaults to 3; 5 spends more build time searching tactics.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
