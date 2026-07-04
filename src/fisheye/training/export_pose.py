"""Export an existing YOLO pose run to ONNX/TensorRT without retraining."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import yaml
from rich.console import Console

from .export_shared import write_registry_model_exports
from ..registry.db import Registry, RegistryPaths
from ..shared.system_metadata import build_invocation_record


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _infer_config_path(run_dir: Path) -> Optional[Path]:
    inputs = run_dir / "inputs"
    candidates = sorted(inputs.glob("*trainable*.yaml"))
    if candidates:
        return candidates[0]
    candidates = sorted(inputs.glob("*.yaml"))
    return candidates[0] if candidates else None


def _infer_manifest_path(run_dir: Path) -> Optional[Path]:
    inputs = run_dir / "inputs"
    candidates = sorted(inputs.glob("*.manifest.json"))
    return candidates[0] if candidates else None


def _load_yaml(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists() or not path.is_file():
        return None
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_imgsz(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        size = int(text)
        return size, size
    return 640, 640


def _resolve_weights_path(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Could not find pose weights under {run_dir / 'weights'}")
    return resolved.resolve()


def _resolve_imgsz(value: Optional[list[int]], config: dict[str, Any], args_yaml: dict[str, Any]) -> list[int]:
    if value:
        if len(value) == 1:
            return [int(value[0]), int(value[0])]
        return [int(value[0]), int(value[1])]
    training_params = config.get("training_params")
    if isinstance(training_params, dict) and training_params.get("imgsz") is not None:
        h, w = _normalize_imgsz(training_params.get("imgsz"))
        return [int(h), int(w)]
    if args_yaml.get("imgsz") is not None:
        h, w = _normalize_imgsz(args_yaml.get("imgsz"))
        return [int(h), int(w)]
    return [640, 640]


def _resolve_device(value: Optional[str], config: dict[str, Any], args_yaml: dict[str, Any]) -> str:
    if value is not None:
        return str(value)
    training_params = config.get("training_params")
    if isinstance(training_params, dict) and training_params.get("device") is not None:
        return str(training_params.get("device"))
    if args_yaml.get("device") is not None:
        return str(args_yaml.get("device"))
    return "cpu"


def _resolve_pose_schema(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_schema = manifest.get("pose_schema")
    if isinstance(manifest_schema, dict):
        return dict(manifest_schema)
    raw_shape = config.get("kpt_shape")
    kpt_shape = list(raw_shape) if isinstance(raw_shape, (list, tuple)) else None
    return {
        "kpt_shape": kpt_shape,
        "keypoint_labels": None,
        "skeleton": None,
    }


def _register_exports(
    *,
    registry_path: Path,
    run_id: str,
    pose_schema: dict[str, Any],
    export_artifacts: dict[str, Any],
    console: Console,
) -> None:
    registry = Registry(registry_path)
    try:
        skeleton_id = registry.upsert_pose_skeleton_spec(
            kpt_shape=pose_schema.get("kpt_shape"),
            keypoint_labels=pose_schema.get("keypoint_labels"),
            edges=pose_schema.get("skeleton"),
            name=str(pose_schema.get("skeleton_id") or "pose_schema"),
        )
        if skeleton_id:
            export_artifacts["skeleton_id"] = skeleton_id
        export_artifacts["pose_schema"] = dict(pose_schema)
        write_registry_model_exports(
            registry=registry,
            run_id=run_id,
            export_artifacts=export_artifacts,
        )
        console.print(f"[green]✓ Registry export rows updated:[/green] {registry_path}")
    finally:
        registry.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export an existing YOLO pose training run to ONNX/TensorRT without retraining.",
    )
    parser.add_argument("run_dir", type=Path, help="Existing pose training run directory.")
    parser.add_argument("--weights", help="Optional weights path. Defaults to run_dir/weights/best.pt.")
    parser.add_argument("--run-id", help="Registry run_id. Defaults to run_dir name.")
    parser.add_argument("--set-id", help="Training set id. Defaults to manifest set_id when available.")
    parser.add_argument("--config", type=Path, help="Training config YAML. Defaults to run_dir/inputs/*trainable*.yaml.")
    parser.add_argument("--manifest", type=Path, help="Training manifest JSON. Defaults to run_dir/inputs/*.manifest.json.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for export rows.")
    parser.add_argument("--no-log-registry", dest="log_registry", action="store_false", default=True)
    parser.add_argument("--imgsz", nargs="+", type=int, help="Export image size override, either S or H W.")
    parser.add_argument("--device", help="Export device, e.g. 0, cuda:0, or cpu.")
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX.")
    parser.add_argument("--export-trt", action="store_true", help="Build TensorRT engine. Implies ONNX export.")
    parser.add_argument("--onnx-opset", type=int, default=13)
    parser.add_argument("--onnx-simplify", action="store_true")
    parser.add_argument("--onnx-path", help="Existing ONNX path to reuse instead of exporting a new ONNX.")
    parser.add_argument("--onnx-dynamic", action="store_true", help="Export ONNX with dynamic batch dimension.")
    parser.add_argument("--onnx-batch", type=int, default=1)
    parser.add_argument("--trt-precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--trtexec", help="Optional trtexec binary path.")
    parser.add_argument("--trt-cuda-graph", action="store_true")
    parser.add_argument("--trt-profiling", action="store_true")
    parser.add_argument("--trt-verbose", action="store_true")
    parser.add_argument("--trt-input-name", default="images")
    parser.add_argument("--trt-min-batch", type=int)
    parser.add_argument("--trt-opt-batch", type=int)
    parser.add_argument("--trt-max-batch", type=int)
    parser.add_argument("--trt-min-shapes")
    parser.add_argument("--trt-opt-shapes")
    parser.add_argument("--trt-max-shapes")
    parser.add_argument(
        "--trt-builder-optimization-level",
        type=int,
        choices=range(0, 6),
        metavar="{0..5}",
        help="TensorRT builder effort level. trtexec defaults to 3; 5 spends more build time searching tactics.",
    )
    args = parser.parse_args(argv)

    console = Console()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] run_dir does not exist: {run_dir}")
        return 2
    if not (args.export_onnx or args.export_trt):
        console.print("[bold red]Error:[/bold red] pass --export-onnx or --export-trt")
        return 2

    weights_path = _resolve_weights_path(run_dir, args.weights)
    run_id = args.run_id or run_dir.name
    config_path = args.config.expanduser().resolve() if args.config else _infer_config_path(run_dir)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else _infer_manifest_path(run_dir)
    config_payload = _load_yaml(config_path)
    manifest_payload = _load_json(manifest_path)
    args_yaml = _load_yaml(run_dir / "args.yaml")

    from .train_pose import _export_pose_onnx_artifacts, _load_manifest_run_hints
    from ultralytics import YOLO

    manifest_hints = _load_manifest_run_hints(str(manifest_path) if manifest_path else None)
    set_id = args.set_id or manifest_hints.get("set_id")
    pose_schema = _resolve_pose_schema(config_payload, manifest_payload)
    imgsz = _resolve_imgsz(args.imgsz, config_payload, args_yaml)
    device = _resolve_device(args.device, config_payload, args_yaml)

    console.print(f"[dim]run_dir:[/dim] {run_dir}")
    console.print(f"[dim]weights:[/dim] {weights_path}")
    console.print(f"[dim]run_id:[/dim] {run_id}")
    console.print(f"[dim]set_id:[/dim] {set_id or '<unknown>'}")
    console.print(f"[dim]imgsz:[/dim] {imgsz}")
    console.print(f"[dim]device:[/dim] {device}")

    model = YOLO(str(weights_path))
    training_params = {"imgsz": imgsz, "device": device}
    export_args = Namespace(**vars(args))
    export_args.export_onnx = bool(args.export_onnx or args.export_trt)
    export_args.registry = args.registry

    export_artifacts = _export_pose_onnx_artifacts(
        run_dir=run_dir,
        run_id=run_id,
        model=model,
        weights_path=weights_path,
        training_params=training_params,
        args=export_args,
        manifest_path=manifest_path,
        manifest_hints={**manifest_hints, "set_id": set_id},
        console=console,
    )
    if export_artifacts.get("errors"):
        console.print(f"[yellow]Export completed with issues:[/yellow] {export_artifacts.get('errors')}")
        return 1

    export_artifacts["pose_schema"] = dict(pose_schema)
    export_artifacts["source_training_run"] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "set_id": set_id,
        "weights_path": str(weights_path),
        "weights_sha256": _safe_sha256_file(weights_path),
        "config_path": str(config_path) if config_path else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "invocation": build_invocation_record(tool="fisheye.training.export_pose", args=args),
    }

    if args.log_registry:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        _register_exports(
            registry_path=registry_path,
            run_id=run_id,
            pose_schema=pose_schema,
            export_artifacts=export_artifacts,
            console=console,
        )

    if export_artifacts.get("onnx_path"):
        console.print(f"[green]✓ ONNX:[/green] {export_artifacts['onnx_path']}")
    if export_artifacts.get("engine_path"):
        console.print(f"[green]✓ TensorRT:[/green] {export_artifacts['engine_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
