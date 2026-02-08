#!/usr/bin/env python3
"""Preferred run-aware detection export CLI.

This command is the recommended export path for trained detection runs:
- resolves weights/report/manifest from a run directory,
- applies imgsz/device overrides when requested,
- exports ONNX and/or TensorRT artifacts,
- optionally logs exported artifacts to the training registry.

For low-level direct ONNX export from a weights file, use:
    python -m fisheye.training.export_onnx ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console

from .train_detection import _export_detection_artifacts, _load_manifest_summary
from ..registry.db import Registry, RegistryPaths


def _format_onnx_outputs_line(output_contract: Any) -> str | None:
    if not isinstance(output_contract, list) or not output_contract:
        return None
    parts: list[str] = []
    for item in output_contract:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "?")
        dtype = str(item.get("dtype") or "?")
        shape = item.get("shape")
        shape_text = "?"
        if isinstance(shape, list):
            shape_text = "(" + ",".join(str(v) for v in shape) + ")"
        parts.append(f"{name}[{dtype},{shape_text}]")
    if not parts:
        return None
    return ", ".join(parts)


def _load_training_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted(
        run_dir.glob("*_detection_training_report.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    text = candidates[0].read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or None
    except Exception:
        # Legacy reports may include python/object tags emitted by yaml.dump.
        try:
            return yaml.load(text, Loader=yaml.BaseLoader) or None
        except Exception:
            return None


def _resolve_weights_path(run_dir: Optional[Path], weights_arg: Optional[str]) -> Optional[Path]:
    if weights_arg:
        return Path(weights_arg)
    if not run_dir:
        return None
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return last
    return None


def _resolve_manifest_path(
    args_manifest: Optional[str],
    report: Optional[Dict[str, Any]],
) -> Optional[str]:
    if args_manifest:
        return args_manifest
    if not report:
        return None
    history = report.get("training_history") or {}
    return history.get("manifest_path")


def _resolve_training_params(report: Optional[Dict[str, Any]], args) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if report:
        history = report.get("training_history") or {}
        if isinstance(history, dict):
            effective = history.get("effective_training_params")
            if isinstance(effective, dict):
                params = dict(effective)
            else:
                params = report.get("training_params") or {}
        else:
            params = report.get("training_params") or {}
    if args.imgsz:
        if len(args.imgsz) == 1:
            params["imgsz"] = args.imgsz[0]
        else:
            params["imgsz"] = args.imgsz[:2]
    if args.device:
        params["device"] = args.device
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ONNX/TensorRT artifacts from an existing detection run.",
        epilog=(
            "Recommended for production use. "
            "Use --log-registry to record export artifacts in sqlite."
        ),
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to the training run directory (contains weights/).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to weights (overrides run_dir/weights/best.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest JSON path (overrides report manifest).",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        help="Override export image size (single value or H W).",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override export device (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export ONNX artifact.",
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Export TensorRT engine (implies ONNX).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="ONNX opset to use for export.",
    )
    parser.add_argument(
        "--onnx-simplify",
        action="store_true",
        help="Run ONNX simplification after export.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Optional existing ONNX path to reuse (skips ONNX export).",
    )
    parser.add_argument(
        "--nms-conf",
        type=float,
        default=0.8,
        help="NMS confidence threshold baked into the ONNX export.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.65,
        help="NMS IoU threshold baked into the ONNX export.",
    )
    parser.add_argument(
        "--nms-topk",
        type=int,
        default=1,
        help="Max detections for the ONNX NMS export.",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        default="fp16",
        help="Precision to use for TensorRT export.",
    )
    parser.add_argument(
        "--trtexec",
        type=str,
        default=None,
        help="Optional path to trtexec for TensorRT export.",
    )
    parser.add_argument(
        "--trt-cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture during TensorRT export.",
    )
    parser.add_argument(
        "--trt-profiling",
        action="store_true",
        help="Enable TensorRT profiling outputs (timing/output/profile JSON).",
    )
    parser.add_argument(
        "--trt-verbose",
        action="store_true",
        help="Enable verbose TensorRT build logs.",
    )
    parser.add_argument(
        "--log-registry",
        action="store_true",
        help="Record exports in the registry.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path.",
    )
    args = parser.parse_args()

    console = Console()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir and not run_dir.exists():
        console.print(f"[red]Run directory not found:[/red] {run_dir}")
        return

    weights_path = _resolve_weights_path(run_dir, args.weights)
    if not weights_path or not weights_path.exists():
        console.print("[red]Could not resolve weights path.[/red]")
        return

    if not (args.export_onnx or args.export_trt):
        console.print("[yellow]No export flags set. Use --export-onnx and/or --export-trt.[/yellow]")
        return

    report = _load_training_report(run_dir) if run_dir else None
    manifest_path = _resolve_manifest_path(args.manifest, report)
    manifest_summary = _load_manifest_summary(manifest_path)
    training_params = _resolve_training_params(report, args)

    if args.manifest:
        console.print(f"[dim]Manifest override:[/dim] {args.manifest}")
    elif report is None:
        console.print("[dim]No training report found; manifest not resolved.[/dim]")
    elif not manifest_path:
        console.print("[dim]Training report missing manifest_path; source_manifest will be empty.[/dim]")

    if manifest_summary.get("manifest_error"):
        console.print(f"[yellow]Manifest issue:[/yellow] {manifest_summary['manifest_error']}")
    elif manifest_summary.get("manifest_path"):
        console.print(
            f"[green]Source manifest:[/green] {manifest_summary['manifest_path']} "
            f"(datasets={manifest_summary.get('manifest_dataset_count')})"
        )

    run_id = run_dir.name if run_dir else weights_path.stem
    run_root = run_dir if run_dir else weights_path.parent

    export_artifacts = _export_detection_artifacts(
        run_dir=run_root,
        run_id=run_id,
        weights_path=weights_path,
        training_params=training_params,
        export_imgsz=None,
        args=args,
        manifest_summary=manifest_summary,
        console=console,
    )
    outputs_line = _format_onnx_outputs_line(export_artifacts.get("onnx_output_contract"))
    if outputs_line:
        console.print(f"[green]ONNX outputs:[/green] {outputs_line}")

    if args.log_registry:
        try:
            registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
            registry = Registry(registry_path)
            onnx_path = export_artifacts.get("onnx_path")
            if onnx_path:
                registry.record_model_export(
                    run_id=run_id,
                    export_type="onnx",
                    path=Path(onnx_path),
                    manifest_path=Path(export_artifacts.get("onnx_manifest_path"))
                    if export_artifacts.get("onnx_manifest_path")
                    else None,
                    metadata={
                        "sha256": export_artifacts.get("onnx_sha256"),
                        "manifest_sha256": export_artifacts.get("onnx_manifest_sha256"),
                        "manifest_path": export_artifacts.get("onnx_manifest_path"),
                        "output_contract": export_artifacts.get("onnx_output_contract"),
                        "build_env": export_artifacts.get("onnx_build_env"),
                        "metadata_props": export_artifacts.get("onnx_metadata_props"),
                        "requires_plugins": export_artifacts.get("onnx_requires_plugins"),
                        "plugin_ops": export_artifacts.get("onnx_plugin_ops"),
                        "plugin_versions": export_artifacts.get("onnx_plugin_versions"),
                        "errors": export_artifacts.get("errors"),
                    },
                )
            engine_path = export_artifacts.get("engine_path")
            if engine_path:
                registry.record_model_export(
                    run_id=run_id,
                    export_type="tensorrt",
                    path=Path(engine_path),
                    manifest_path=Path(export_artifacts.get("engine_manifest_path"))
                    if export_artifacts.get("engine_manifest_path")
                    else None,
                    metadata={
                        "sha256": export_artifacts.get("engine_sha256"),
                        "manifest_sha256": export_artifacts.get("engine_manifest_sha256"),
                        "precision": export_artifacts.get("engine_precision"),
                        "output_contract": export_artifacts.get("onnx_output_contract"),
                        "requires_plugins": export_artifacts.get("onnx_requires_plugins"),
                        "plugin_ops": export_artifacts.get("onnx_plugin_ops"),
                        "plugin_versions": export_artifacts.get("onnx_plugin_versions"),
                        "build_env": export_artifacts.get("build_env"),
                        "trt_device_info": export_artifacts.get("trt_device_info"),
                        "errors": export_artifacts.get("errors"),
                    },
                )
            registry.close()
            console.print(f"[green]✓ Registry updated:[/green] {registry_path}")
        except Exception as exc:
            console.print(f"[yellow]Registry update skipped:[/yellow] {exc}")


if __name__ == "__main__":
    main()
