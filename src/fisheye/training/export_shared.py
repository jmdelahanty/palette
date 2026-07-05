"""Shared helpers for ONNX/TensorRT export and export registry writes."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import torch
from rich.console import Console


def run_subprocess_streaming(
    command: list[str],
    console: Console,
    label: str,
    log_path: Path | None = None,
) -> bool:
    console.print(f"[dim]Running {label}:[/dim] {' '.join(command)}")
    log_handle = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if process.stdout:
            for line in process.stdout:
                if log_handle:
                    log_handle.write(line)
                    log_handle.flush()
                console.print(line.rstrip(), markup=False)
        process.wait()
        if process.returncode != 0:
            console.print(f"[red]✗ {label} failed with code {process.returncode}[/red]")
            return False
        return True
    except Exception as exc:
        console.print(f"[red]✗ {label} failed:[/red] {exc}")
        return False
    finally:
        if log_handle:
            log_handle.close()


def _read_trtexec_version(trtexec_path: Path | None) -> tuple[str | None, str | None, str | None]:
    if not trtexec_path:
        return None, None, None
    raw_output = None
    try:
        result = subprocess.run(
            [str(trtexec_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        raw_output = "\n".join(
            [part for part in [result.stdout.strip(), result.stderr.strip()] if part]
        ).strip()
        if raw_output:
            dotted = re.search(r"TensorRT\s+Version[:\s]+(\d+\.\d+\.\d+\.\d+)", raw_output)
            if dotted:
                return dotted.group(1), "trtexec", raw_output
            dotted = re.search(r"TensorRT\s*v?(\d+\.\d+\.\d+\.\d+)", raw_output)
            if dotted:
                return dotted.group(1), "trtexec", raw_output
    except Exception:
        raw_output = None
    path_match = re.search(r"TensorRT-(\d+\.\d+\.\d+\.\d+)", str(trtexec_path))
    if path_match:
        return path_match.group(1), "path", raw_output
    return None, None, raw_output


def resolve_trtexec_path(explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    else:
        try:
            from .onnx_to_tensorrt import TRTEXEC_PATH as default_trtexec_path  # type: ignore
        except Exception:
            default_trtexec_path = None
        if default_trtexec_path:
            candidates.append(Path(str(default_trtexec_path)).expanduser())
        which_path = shutil.which("trtexec")
        if which_path:
            candidates.append(Path(which_path))
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.resolve()
            except Exception:
                return candidate
    return None


def _parse_trtexec_device_info_text(raw_text: str) -> dict:
    if not raw_text:
        return {}
    info: dict = {}
    patterns: list[tuple[str, str, str | None]] = [
        ("selected_device_name", r"Selected Device:\s*(.+)$", None),
        ("selected_device_id", r"Selected Device ID:\s*(\d+)$", "int"),
        ("selected_device_uuid", r"Selected Device UUID:\s*(\S+)$", None),
        ("compute_capability", r"Compute Capability:\s*([0-9.]+)$", None),
        ("sm_count", r"SMs:\s*(\d+)$", "int"),
        ("device_global_memory_mib", r"Device Global Memory:\s*(\d+)\s*MiB", "int"),
        ("memory_bus_width_bits", r"Memory Bus Width:\s*(\d+)\s*bits", "int"),
        ("trtexec_reported_version", r"TensorRT version:\s*([0-9.]+)", None),
    ]
    for line in raw_text.splitlines():
        clean = re.sub(r"^\[[^\]]+\]\s+\[I\]\s*", "", line).strip()
        if not clean:
            continue
        for key, pattern, cast in patterns:
            match = re.search(pattern, clean)
            if not match:
                continue
            value = match.group(1).strip()
            if cast == "int":
                try:
                    info[key] = int(value)
                except ValueError:
                    info[key] = value
            else:
                info[key] = value
    return info


def _parse_trtexec_device_info(log_path: Path | None) -> dict:
    if not log_path or not log_path.exists():
        return {}
    try:
        raw_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    return _parse_trtexec_device_info_text(raw_text)


def collect_export_env(trtexec_path: Path | None, trtexec_log_path: Path | None = None) -> dict:
    env = {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "trtexec_path": str(trtexec_path) if trtexec_path else None,
        "system_hostname": platform.node() or None,
    }
    if torch.cuda.is_available():
        try:
            device_id = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(device_id)
            env["gpu_name"] = torch.cuda.get_device_name(device_id)
            env["torch_device"] = {
                "selected_device_id": device_id,
                "selected_device_name": str(getattr(props, "name", env["gpu_name"])),
                "compute_capability": f"{int(props.major)}.{int(props.minor)}",
                "sm_count": int(getattr(props, "multi_processor_count", 0)),
                "device_global_memory_mib": int(getattr(props, "total_memory", 0) // (1024 * 1024)),
            }
        except Exception:
            env["gpu_name"] = None
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        version, source, _raw_output = _read_trtexec_version(trtexec_path)
        env["tensorrt_version"] = version
        if source:
            env["tensorrt_version_source"] = source
    else:
        env["tensorrt_version"] = trt.__version__
        env["tensorrt_version_source"] = "python"
    trtexec_runtime = _parse_trtexec_device_info(trtexec_log_path)
    if trtexec_runtime:
        env["trtexec_runtime"] = trtexec_runtime
        if not env.get("tensorrt_version") and trtexec_runtime.get("trtexec_reported_version"):
            env["tensorrt_version"] = trtexec_runtime.get("trtexec_reported_version")
            env["tensorrt_version_source"] = "trtexec_log"
    return env


def write_registry_model_exports(
    *,
    registry: Any,
    run_id: str,
    export_artifacts: dict[str, Any],
) -> None:
    onnx_path = export_artifacts.get("onnx_path")
    if onnx_path:
        onnx_metadata = {
            "skeleton_id": export_artifacts.get("skeleton_id"),
            "pose_schema": export_artifacts.get("pose_schema"),
            "sha256": export_artifacts.get("onnx_sha256"),
            "manifest_sha256": export_artifacts.get("onnx_manifest_sha256"),
            "manifest_path": export_artifacts.get("onnx_manifest_path"),
            "opset": export_artifacts.get("onnx_opset"),
            "input_shape": export_artifacts.get("input_shape"),
            "imgsz": export_artifacts.get("imgsz"),
            "nms": {
                "conf": export_artifacts.get("nms_conf"),
                "iou": export_artifacts.get("nms_iou"),
                "topk": export_artifacts.get("nms_topk"),
            },
            "nms_conf": export_artifacts.get("nms_conf"),
            "nms_iou": export_artifacts.get("nms_iou"),
            "nms_topk": export_artifacts.get("nms_topk"),
            "build_env": export_artifacts.get("onnx_build_env") or export_artifacts.get("build_env"),
            "output_contract": export_artifacts.get("onnx_output_contract"),
            "metadata_props": export_artifacts.get("onnx_metadata_props"),
            "requires_plugins": export_artifacts.get("onnx_requires_plugins"),
            "plugin_ops": export_artifacts.get("onnx_plugin_ops"),
            "plugin_versions": export_artifacts.get("onnx_plugin_versions"),
            "errors": export_artifacts.get("errors"),
        }
        registry.record_model_export(
            run_id=run_id,
            export_type="onnx",
            path=Path(onnx_path),
            manifest_path=Path(export_artifacts.get("onnx_manifest_path"))
            if export_artifacts.get("onnx_manifest_path")
            else None,
            metadata=onnx_metadata,
        )

    engine_path = export_artifacts.get("engine_path")
    if engine_path:
        trt_metadata = {
            "skeleton_id": export_artifacts.get("skeleton_id"),
            "pose_schema": export_artifacts.get("pose_schema"),
            "sha256": export_artifacts.get("engine_sha256"),
            "manifest_sha256": export_artifacts.get("engine_manifest_sha256"),
            "precision": export_artifacts.get("engine_precision"),
            "input_shape": export_artifacts.get("input_shape"),
            "imgsz": export_artifacts.get("imgsz"),
            "nms": {
                "conf": export_artifacts.get("nms_conf"),
                "iou": export_artifacts.get("nms_iou"),
                "topk": export_artifacts.get("nms_topk"),
            },
            "nms_conf": export_artifacts.get("nms_conf"),
            "nms_iou": export_artifacts.get("nms_iou"),
            "nms_topk": export_artifacts.get("nms_topk"),
            "build_env": export_artifacts.get("build_env"),
            "trt_device_info": export_artifacts.get("trt_device_info"),
            "output_contract": export_artifacts.get("onnx_output_contract"),
            "requires_plugins": export_artifacts.get("onnx_requires_plugins"),
            "plugin_ops": export_artifacts.get("onnx_plugin_ops"),
            "plugin_versions": export_artifacts.get("onnx_plugin_versions"),
            "errors": export_artifacts.get("errors"),
        }
        registry.record_model_export(
            run_id=run_id,
            export_type="tensorrt",
            path=Path(engine_path),
            manifest_path=Path(export_artifacts.get("engine_manifest_path"))
            if export_artifacts.get("engine_manifest_path")
            else None,
            metadata=trt_metadata,
        )
