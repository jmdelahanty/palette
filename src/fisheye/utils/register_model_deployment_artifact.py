#!/usr/bin/env python3
"""Register a hardware/runtime-specific model deployment artifact."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.registry.db import Registry, RegistryPaths


def _load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_flag_value(command: Sequence[Any], flag: str) -> Optional[str]:
    prefix = f"{flag}="
    items = [str(item) for item in command]
    for index, item in enumerate(items):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == flag and index + 1 < len(items):
            return items[index + 1]
    return None


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def build_registration_payload(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None
    engine_path = Path(args.engine_path).expanduser().resolve() if args.engine_path else None
    source_onnx_path = (
        Path(args.source_onnx_path).expanduser().resolve()
        if args.source_onnx_path
        else None
    )
    manifest = _load_json(manifest_path)
    export = _as_dict(manifest.get("export"))
    trt = _as_dict(manifest.get("trt"))
    engine = _as_dict(manifest.get("engine"))
    onnx = _as_dict(manifest.get("onnx"))
    build_env = _as_dict(manifest.get("build_env"))
    trtexec_runtime = _as_dict(build_env.get("trtexec_runtime"))
    nms = _as_dict(export.get("nms"))
    command = trt.get("command") if isinstance(trt.get("command"), list) else []

    if engine_path is None and engine.get("path"):
        engine_path = Path(str(engine["path"])).expanduser()
    if source_onnx_path is None and onnx.get("path"):
        source_onnx_path = Path(str(onnx["path"])).expanduser()

    builder_optimization_level = args.builder_optimization_level
    if builder_optimization_level is None:
        builder_optimization_level = _as_int(
            _extract_flag_value(command, "--builderOptimizationLevel")
        )
    avg_timing = args.avg_timing
    if avg_timing is None:
        avg_timing = _as_int(_extract_flag_value(command, "--avgTiming"))
    profiling_verbosity = args.profiling_verbosity
    if profiling_verbosity is None:
        profiling_verbosity = _extract_flag_value(command, "--profilingVerbosity")

    metadata = {
        "schema_version": "palette.model_deployment_artifact.v1",
        "manifest": manifest,
        "registration": {
            "apply": bool(args.apply),
            "registry": str(args.registry) if args.registry else None,
            "target_hardware_class": args.target_hardware_class,
            "deployment_runtime": args.deployment_runtime,
        },
    }

    return {
        "artifact_id": args.artifact_id,
        "run_id": args.run_id or manifest.get("run_id"),
        "source_onnx_run_id": args.source_onnx_run_id or manifest.get("run_id"),
        "source_onnx_path": str(source_onnx_path) if source_onnx_path else None,
        "source_onnx_sha256": args.source_onnx_sha256
        or onnx.get("sha256")
        or _sha256_file(source_onnx_path),
        "artifact_kind": args.artifact_kind,
        "deployment_runtime": args.deployment_runtime,
        "target_hardware_class": args.target_hardware_class,
        "target_gpu_name": _first_text(
            args.target_gpu_name,
            trtexec_runtime.get("selected_device_name"),
            build_env.get("gpu_name"),
        ),
        "target_compute_capability": _first_text(
            args.target_compute_capability,
            trtexec_runtime.get("compute_capability"),
            _as_dict(build_env.get("torch_device")).get("compute_capability"),
        ),
        "precision": _first_text(args.precision, trt.get("precision"), export.get("precision")),
        "engine_path": str(engine_path) if engine_path else None,
        "engine_sha256": args.engine_sha256 or engine.get("sha256") or _sha256_file(engine_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": args.manifest_sha256 or _sha256_file(manifest_path),
        "status": args.status,
        "validation_summary": _load_json(args.validation_summary_json),
        "trtexec_path": _first_text(args.trtexec_path, trt.get("trtexec_path"), build_env.get("trtexec_path")),
        "trt_version": _first_text(
            args.trt_version,
            build_env.get("tensorrt_version"),
            trtexec_runtime.get("trtexec_reported_version"),
        ),
        "cuda_version": _first_text(args.cuda_version, build_env.get("cuda_version")),
        "builder_optimization_level": builder_optimization_level,
        "avg_timing": avg_timing,
        "profiling_verbosity": profiling_verbosity,
        "cuda_graph": args.cuda_graph if args.cuda_graph is not None else trt.get("cuda_graph"),
        "nms_conf": args.nms_conf if args.nms_conf is not None else _as_float(nms.get("conf")),
        "nms_iou": args.nms_iou if args.nms_iou is not None else _as_float(nms.get("iou")),
        "nms_topk": args.nms_topk if args.nms_topk is not None else _as_int(nms.get("topk")),
        "metadata": metadata,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Registry SQLite path.")
    parser.add_argument("--run-id", help="Training/model run_id that owns this deployment artifact.")
    parser.add_argument("--artifact-id", help="Stable deployment artifact ID. Inferred when omitted.")
    parser.add_argument("--source-onnx-run-id", help="Source ONNX row/run ID. Defaults to manifest run_id.")
    parser.add_argument("--source-onnx-path", type=Path, help="Source ONNX path.")
    parser.add_argument("--source-onnx-sha256", help="Source ONNX SHA-256.")
    parser.add_argument("--artifact-kind", default="tensorrt_engine")
    parser.add_argument("--deployment-runtime", required=True, help="Runtime target, e.g. orange.")
    parser.add_argument("--target-hardware-class", required=True, help="Hardware class, e.g. A16.")
    parser.add_argument("--target-gpu-name", help="GPU model/name override.")
    parser.add_argument("--target-compute-capability", help="Compute capability override.")
    parser.add_argument("--precision", help="Engine precision, e.g. fp16.")
    parser.add_argument("--engine-path", type=Path, help="TensorRT engine path.")
    parser.add_argument("--engine-sha256", help="TensorRT engine SHA-256.")
    parser.add_argument("--manifest-path", type=Path, help="TensorRT manifest path.")
    parser.add_argument("--manifest-sha256", help="Manifest SHA-256.")
    parser.add_argument(
        "--status",
        default="candidate",
        choices=["candidate", "validated", "preferred", "deprecated"],
    )
    parser.add_argument("--validation-summary-json", type=Path)
    parser.add_argument("--trtexec-path", type=Path)
    parser.add_argument("--trt-version")
    parser.add_argument("--cuda-version")
    parser.add_argument("--builder-optimization-level", type=int)
    parser.add_argument("--avg-timing", type=int)
    parser.add_argument("--profiling-verbosity")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nms-conf", type=float)
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--nms-topk", type=int)
    parser.add_argument("--apply", action="store_true", help="Write to registry.")
    parser.add_argument("--output-json", type=Path, help="Optional path for dry-run/apply report.")
    args = parser.parse_args(argv)

    payload = build_registration_payload(args)
    if not payload.get("run_id"):
        raise SystemExit("--run-id is required when the manifest does not contain run_id")

    report = {
        "status": "planned",
        "apply": bool(args.apply),
        "registry": str(args.registry or RegistryPaths.from_env(Path.cwd()).path),
        "deployment_artifact": {
            key: value
            for key, value in payload.items()
            if key not in {"metadata", "validation_summary"}
        },
    }
    if args.apply:
        registry = Registry(args.registry or RegistryPaths.from_env(Path.cwd()).path)
        try:
            artifact_id = registry.record_model_deployment_artifact(
                run_id=str(payload["run_id"]),
                artifact_id=payload.get("artifact_id"),
                source_onnx_run_id=payload.get("source_onnx_run_id"),
                source_onnx_path=Path(payload["source_onnx_path"])
                if payload.get("source_onnx_path")
                else None,
                source_onnx_sha256=payload.get("source_onnx_sha256"),
                artifact_kind=str(payload["artifact_kind"]),
                deployment_runtime=str(payload["deployment_runtime"]),
                target_hardware_class=payload.get("target_hardware_class"),
                target_gpu_name=payload.get("target_gpu_name"),
                target_compute_capability=payload.get("target_compute_capability"),
                precision=payload.get("precision"),
                engine_path=Path(payload["engine_path"]) if payload.get("engine_path") else None,
                engine_sha256=payload.get("engine_sha256"),
                manifest_path=Path(payload["manifest_path"])
                if payload.get("manifest_path")
                else None,
                manifest_sha256=payload.get("manifest_sha256"),
                status=str(payload["status"]),
                validation_summary=payload.get("validation_summary"),
                trtexec_path=Path(payload["trtexec_path"])
                if payload.get("trtexec_path")
                else None,
                trt_version=payload.get("trt_version"),
                cuda_version=payload.get("cuda_version"),
                builder_optimization_level=payload.get("builder_optimization_level"),
                avg_timing=payload.get("avg_timing"),
                profiling_verbosity=payload.get("profiling_verbosity"),
                cuda_graph=payload.get("cuda_graph"),
                nms_conf=payload.get("nms_conf"),
                nms_iou=payload.get("nms_iou"),
                nms_topk=payload.get("nms_topk"),
                metadata=payload.get("metadata"),
            )
        finally:
            registry.close()
        report["status"] = "ok"
        report["artifact_id"] = artifact_id

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
