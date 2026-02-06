# src/fisheye/training/train_detection.py

"""
Detection YOLO Trainer from zarrs with Enhanced Metadata Logging

Features:
- Tracks crop source (detect/filtered/interpolated)
- Option to filter out interpolated data
- Complete provenance tracking
- Enhanced training reports
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import re
import torch
import yaml
from pathlib import Path
import time
import platform
import traceback
import pandas as pd
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json
import zarr

from .config import DetectConfig, DatasetConfig
from .zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
from ..utils.system import get_git_info, build_invocation_record
from ..registry.db import Registry, RegistryPaths

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"


# Custom DataLoader to ensure compatibility with Ultralytics YOLO's expected interface
class YoloCompatibleDataLoader(DataLoader):
    def reset(self):
        pass


def det_collate_fn(batch):
    """Collate function for detection data."""
    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
    im_files = [s['im_file'] for s in batch]
    ori_shapes = [s['ori_shape'] for s in batch]
    ratio_pads = [s['ratio_pad'] for s in batch]
    cls_list, bboxes_list, batch_idx_list = [], [], []
    
    for i, sample in enumerate(batch):
        cls_labels = np.atleast_1d(sample['cls'])
        if cls_labels.size > 0 and cls_labels[0] is not None:
            cls_list.append(torch.from_numpy(cls_labels).reshape(-1, 1).float())
            bboxes_list.append(torch.from_numpy(sample['bboxes']))
            batch_idx_list.append(torch.full((len(cls_labels),), i, dtype=torch.long))
    
    if not batch_idx_list:
        return {
            'img': images,
            'batch_idx': torch.empty(0, dtype=torch.long),
            'cls': torch.empty(0, 1, dtype=torch.float32),
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }
    
    return {
        'img': images,
        'batch_idx': torch.cat(batch_idx_list, 0),
        'cls': torch.cat(cls_list, 0),
        'bboxes': torch.cat(bboxes_list, 0),
        'im_file': im_files,
        'ori_shape': ori_shapes,
        'ratio_pad': ratio_pads
    }


class DetValidator(DetectionValidator):
    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        
        # Handle cls shape issues
        if 'cls' in pbatch and hasattr(pbatch['cls'], 'shape'):
            cls = pbatch['cls']
            
            # Convert to torch tensor if needed
            if not isinstance(cls, torch.Tensor):
                cls = torch.from_numpy(cls) if hasattr(cls, '__array__') else torch.tensor(cls)
            
            # Handle different shapes
            if cls.ndim == 0:  # Scalar
                pbatch['cls'] = cls.unsqueeze(0)
            elif cls.ndim == 2 and cls.shape[1] == 1:  # (N, 1) -> squeeze to (N,)
                pbatch['cls'] = cls.squeeze(1)
            elif cls.shape[0] == 0:  # Empty array - this might be the issue!
                # For empty batches, create proper empty tensor
                pbatch['cls'] = torch.tensor([], dtype=torch.long)
        
        return pbatch


class DetTrainer(DetectionTrainer):
    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks
        )

def get_zarr_metadata(zarr_paths, console=None):
    """
    Extract comprehensive metadata from zarr files including crop source info.
    
    Args:
        zarr_paths: List of paths to zarr files
        console: Optional Rich console for output
        
    Returns:
        Dictionary of metadata per zarr file
    """
    metadata = {}
    
    for path in zarr_paths:
        try:
            root = zarr.open(path, mode='r')
            path_name = Path(path).name
            
            zarr_meta = {
                'path': str(path),
                'video_frames': 0,
                'crop_info': {},
                'detection_info': {},
                'data_quality': {}
            }
            
            # Get video info
            if 'raw_video' in root:
                if 'images_full' in root['raw_video']:
                    zarr_meta['video_frames'] = root['raw_video/images_full'].shape[0]
                zarr_meta['fps'] = root['raw_video'].attrs.get('fps', 'N/A')
            
            # Get detection info
            if 'detect_runs' in root:
                latest_detect = root['detect_runs'].attrs.get('latest')
                if latest_detect:
                    detect_group = root[f'detect_runs/{latest_detect}']
                    if 'summary_statistics' in detect_group.attrs:
                        stats = detect_group.attrs['summary_statistics']
                        zarr_meta['detection_info'] = {
                            'run_name': latest_detect,
                            'total_detections': stats.get('total_detections', 0),
                            'frames_with_detections': stats.get('frames_with_detections', 0),
                            'detection_rate': stats.get('frames_with_detections', 0) / max(stats.get('total_frames', 1), 1) * 100
                        }
            
            # Get crop info with source tracking
            if 'crop_runs' in root:
                latest_crop = root['crop_runs'].attrs.get('latest')
                if latest_crop:
                    crop_group = root[f'crop_runs/{latest_crop}']
                    
                    # Get crop source information
                    crop_source_type = crop_group.attrs.get('detection_source_type', 'detect')
                    crop_source_path = crop_group.attrs.get('detection_source_path', 'unknown')
                    includes_interpolated = crop_group.attrs.get('includes_interpolated', False)
                    
                    zarr_meta['crop_info'] = {
                        'run_name': latest_crop,
                        'source_type': crop_source_type,
                        'source_path': crop_source_path,
                        'includes_interpolated': includes_interpolated
                    }
                    
                    # Get statistics
                    if 'summary_statistics' in crop_group.attrs:
                        stats = crop_group.attrs['summary_statistics']
                        zarr_meta['crop_info'].update({
                            'total_rois': stats.get('total_rois_cropped', 0),
                            'frames_with_crops': stats.get('frames_with_crops', 0),
                            'roi_size': stats.get('roi_size', [256, 256])
                        })
                    
                    # If interpolated, get breakdown
                    if includes_interpolated:
                        zarr_meta['crop_info']['n_real'] = crop_group.attrs.get('n_real_detections', 0)
                        zarr_meta['crop_info']['n_interpolated'] = crop_group.attrs.get('n_interpolated_detections', 0)
            
            # Get refinement info if available
            refined_root = None
            if REFINED_DETECT_GROUP in root:
                refined_root = root[REFINED_DETECT_GROUP]
            elif LEGACY_REFINED_DETECT_GROUP in root:
                refined_root = root[LEGACY_REFINED_DETECT_GROUP]
            if refined_root is not None:
                latest_refined = refined_root.attrs.get('latest')
                if latest_refined and latest_refined in refined_root:
                    refined_group = refined_root[latest_refined]
                    
                    zarr_meta['data_quality']['has_refinement'] = True
                    zarr_meta['data_quality']['refined_run'] = latest_refined
                    
                    # Check what stages exist
                    if 'filtered' in refined_group:
                        filtered_grp = refined_group['filtered']
                        zarr_meta['data_quality']['filtered_detections'] = filtered_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['jumps_removed'] = filtered_grp.attrs.get('dropped_detections', 0)
                    
                    if 'interpolated' in refined_group:
                        interp_grp = refined_group['interpolated']
                        zarr_meta['data_quality']['interpolated_detections'] = interp_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['gaps_filled'] = interp_grp.attrs.get('gaps_filled', 0)
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
            if console:
                console.print(f"[yellow]Warning: Could not read metadata from {path_name}: {e}[/yellow]")
    
    return metadata


def display_zarr_metadata(metadata, console):
    """Display zarr metadata in a nice table."""
    printed_tables = 0
    for zarr_name, meta in metadata.items():
        if 'error' in meta:
            console.print(f"[red]✗ {zarr_name}: {meta['error']}[/red]")
            continue
        
        # Create info table
        table = Table(title=f"📦 {zarr_name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Video info
        table.add_row("Video Frames", str(meta.get('video_frames', 'N/A')))
        table.add_row("FPS", str(meta.get('fps', 'N/A')))
        
        # Detection info
        if meta.get('detection_info'):
            det_info = meta['detection_info']
            table.add_row("Total Detections", f"{det_info.get('total_detections', 0):,}")
            table.add_row("Detection Rate", f"{det_info.get('detection_rate', 0):.1f}%")
        
        # Crop info with source
        if meta.get('crop_info'):
            crop_info = meta['crop_info']
            source_type = crop_info.get('source_type', 'unknown')
            
            # Color code the source
            if source_type == 'detect':
                source_display = f"[cyan]{source_type}[/cyan] (original)"
            elif source_type == 'filtered':
                source_display = f"[yellow]{source_type}[/yellow] (jumps removed)"
            elif source_type == 'interpolated':
                source_display = f"[magenta]{source_type}[/magenta] (gaps filled)"
            elif source_type == 'manual':
                source_display = f"[green]{source_type}[/green] (manual review)"
            else:
                source_display = source_type
            
            table.add_row("Crop Source", source_display)
            table.add_row("Total ROIs", f"{crop_info.get('total_rois', 0):,}")
            
            # If interpolated, show breakdown
            if crop_info.get('includes_interpolated'):
                n_real = crop_info.get('n_real', 0)
                n_interp = crop_info.get('n_interpolated', 0)
                table.add_row("  └─ Real ROIs", f"{n_real:,}")
                table.add_row("  └─ Interpolated ROIs", f"{n_interp:,}")
        
        # Data quality info
        if meta.get('data_quality', {}).get('has_refinement'):
            quality = meta['data_quality']
            if 'jumps_removed' in quality:
                table.add_row("Jumps Removed", str(quality['jumps_removed']))
            if 'gaps_filled' in quality:
                table.add_row("Gaps Filled", str(quality['gaps_filled']))

        console.print(table)
        console.print()
        printed_tables += 1

    if printed_tables == 0 and metadata:
        console.print("[yellow]No valid dataset metadata tables to display.[/yellow]")


def _normalize_source_type(value, default: str = "detect") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip().lower()
    return text if text else default


def _collect_source_mismatches(full_config: DetectConfig, zarr_metadata: dict) -> list[dict]:
    mismatches: list[dict] = []
    for dataset_name, dataset_cfg in (full_config.datasets or {}).items():
        zarr_path = getattr(dataset_cfg, "zarr_path", None)
        if zarr_path is None:
            continue

        zarr_key = Path(zarr_path).name
        dataset_meta = zarr_metadata.get(zarr_key)
        if not isinstance(dataset_meta, dict) or "error" in dataset_meta:
            continue

        requested_source = _normalize_source_type(getattr(dataset_cfg, "source_type", None))
        crop_info = dataset_meta.get("crop_info") if isinstance(dataset_meta.get("crop_info"), dict) else {}
        detection_info = (
            dataset_meta.get("detection_info") if isinstance(dataset_meta.get("detection_info"), dict) else {}
        )

        available_source = None
        available_source_path = None
        if crop_info:
            available_source = _normalize_source_type(crop_info.get("source_type"))
            available_source_path = crop_info.get("source_path")
        elif detection_info:
            available_source = "detect"
            run_name = detection_info.get("run_name")
            if run_name:
                available_source_path = f"detect_runs/{run_name}"

        if available_source and requested_source != available_source:
            mismatches.append(
                {
                    "dataset_name": dataset_name,
                    "zarr_path": str(zarr_path),
                    "requested_source_type": requested_source,
                    "available_source_type": available_source,
                    "available_source_path": available_source_path,
                }
            )

    return mismatches


def _load_manifest_summary(manifest_path: str | None) -> dict:
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {"manifest_error": f"Manifest not found: {path}"}
    try:
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = json.loads(text)
        datasets = payload.get("datasets") or []
        dataset_ids = [
            ds.get("dataset_id") for ds in datasets if isinstance(ds, dict) and ds.get("dataset_id")
        ]
        manifest_set_id = payload.get("set_id")
        query_filter = payload.get("query_filter") if isinstance(payload, dict) else None
        if not isinstance(query_filter, dict):
            query_filter = {}
        first_dataset = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
        provenance = first_dataset.get("provenance") if isinstance(first_dataset, dict) else {}
        if not isinstance(provenance, dict):
            provenance = {}
        arena = provenance.get("arena") if isinstance(provenance, dict) else {}
        if not isinstance(arena, dict):
            arena = {}
        rig_info = provenance.get("rig_info") if isinstance(provenance, dict) else {}
        if not isinstance(rig_info, dict):
            rig_info = {}
        dataset_name = first_dataset.get("name") if isinstance(first_dataset, dict) else None
        dataset_zarr = first_dataset.get("zarr_path") if isinstance(first_dataset, dict) else None
        manifest_canvas = (
            first_dataset.get("canvas_name")
            or rig_info.get("canvas_name")
            or _infer_canvas_from_dataset_label(dataset_name)
            or _infer_canvas_from_dataset_label(Path(str(dataset_zarr)).stem if dataset_zarr else None)
        )
        manifest_dish = (
            query_filter.get("dish_design")
            or first_dataset.get("dish_design")
            or arena.get("dish_design")
        )
        return {
            "manifest_path": str(path),
            "manifest_sha256": digest,
            "manifest_dataset_ids": dataset_ids,
            "manifest_dataset_count": len(dataset_ids),
            "manifest_set_id": manifest_set_id,
            "manifest_task": payload.get("task") if isinstance(payload, dict) else None,
            "manifest_dish_design": manifest_dish,
            "manifest_canvas_name": manifest_canvas,
        }
    except Exception as exc:
        return {"manifest_error": str(exc), "manifest_path": str(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        return _sha256_file(path)
    except Exception:
        return None


def _strip_manifest_suffixes(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _infer_canvas_from_dataset_label(value: str | None) -> str | None:
    if not value:
        return None
    stem = Path(str(value)).stem
    tokens = [token for token in stem.split("_") if token]
    if not tokens:
        return None
    for token in reversed(tokens):
        if token and not token.isdigit() and token.lower() != "arena":
            return token
    return tokens[-1]


def _sanitize_run_component(value: str | None, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


def _build_default_run_name(
    *,
    manifest_summary: dict,
    task_fallback: str,
    timestamp: str | None = None,
    pid: int | None = None,
) -> str:
    dish = _sanitize_run_component(manifest_summary.get("manifest_dish_design"), "unknown_dish")
    canvas = _sanitize_run_component(manifest_summary.get("manifest_canvas_name"), "unknown_canvas")
    task = _sanitize_run_component(manifest_summary.get("manifest_task") or task_fallback, task_fallback)
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    process_id = int(os.getpid() if pid is None else pid)
    return f"{dish}_{canvas}_{task}_{stamp}_{process_id}"


def _infer_set_slug(set_id: str | None, config_path: Path | None) -> str:
    if set_id:
        slug = _strip_manifest_suffixes(set_id)
        return slug or "detect_training"
    if config_path is not None:
        stem = _strip_manifest_suffixes(config_path.stem)
        return stem or "detect_training"
    return "detect_training"


def _resolve_project_dir(
    *,
    args,
    training_params: dict,
    set_id: str | None,
    config_path: Path | None,
    console: Console,
) -> None:
    if args.project:
        training_params["project"] = str(Path(args.project).expanduser().resolve())
        return

    configured_project = training_params.get("project")
    if isinstance(configured_project, str) and configured_project.strip():
        training_params["project"] = str(Path(configured_project).expanduser().resolve())
        return

    nvme_root = Path("/nvme1")
    if not nvme_root.exists():
        return

    slug = _infer_set_slug(set_id, config_path)
    project_dir = (nvme_root / "models" / "detect" / slug).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    training_params["project"] = str(project_dir)
    console.print(f"[cyan]Using default model output directory:[/cyan] {project_dir}")


def _snapshot_training_inputs(
    *,
    run_dir: Path,
    config_path: Path | None,
    manifest_path: Path | None,
    invocation_payload: dict | None,
) -> list[Path]:
    """Copy immutable run inputs into run_dir/inputs for reproducibility."""
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if config_path is not None and config_path.exists():
        dest = inputs_dir / config_path.name
        shutil.copy2(config_path, dest)
        written.append(dest)
    if manifest_path is not None and manifest_path.exists():
        dest = inputs_dir / manifest_path.name
        shutil.copy2(manifest_path, dest)
        written.append(dest)
    if invocation_payload:
        dest = inputs_dir / "train_invocation.json"
        dest.write_text(json.dumps(invocation_payload, indent=2), encoding="utf-8")
        written.append(dest)

    return written


def _record_registry_training_run(
    *,
    args,
    console: Console,
    invocation_payload: dict | None,
    run_id: str,
    set_id: str | None,
    config_path: Path | None,
    manifest_path: Path | None,
    model_path: Path | None,
    metrics_path: Path | None,
    status: str,
    final_metrics: dict | None,
    export_artifacts: dict | None = None,
) -> None:
    registry = None
    try:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        registry.record_training_run(
            run_id=run_id,
            set_id=set_id,
            config_path=config_path,
            manifest_path=manifest_path,
            model_path=model_path,
            metrics_path=metrics_path,
            config_sha256=_safe_sha256_file(config_path),
            manifest_sha256=_safe_sha256_file(manifest_path),
            model_sha256=_safe_sha256_file(model_path),
            metrics_sha256=_safe_sha256_file(metrics_path),
            status=status,
            final_metrics=final_metrics,
            invocation=invocation_payload,
        )
        if status == "success" and export_artifacts:
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
                        "errors": export_artifacts.get("errors"),
                    },
                )
        console.print(f"[green]✓ Registry updated:[/green] {registry_path}")
    except Exception as exc:
        console.print(f"[yellow]Registry update skipped:[/yellow] {exc}")
    finally:
        if registry is not None:
            try:
                registry.close()
            except Exception:
                pass


def _normalize_imgsz(value) -> tuple[int, int]:
    if value is None:
        return 640, 640
    if isinstance(value, (list, tuple)):
        if not value:
            return 640, 640
        if len(value) == 1:
            size = int(value[0])
            return size, size
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _resolve_export_device(value) -> str:
    if isinstance(value, str) and value:
        if value.isdigit():
            return f"cuda:{value}"
        if value.lower().startswith("cuda") or value.lower() == "cpu":
            return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _run_subprocess(
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
        )
        if process.stdout:
            for line in process.stdout:
                if log_handle:
                    log_handle.write(line)
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
            dotted = re.search(r"TensorRT\\s+Version[:\\s]+(\\d+\\.\\d+\\.\\d+\\.\\d+)", raw_output)
            if dotted:
                return dotted.group(1), "trtexec", raw_output
            dotted = re.search(r"TensorRT\\s*v?(\\d+\\.\\d+\\.\\d+\\.\\d+)", raw_output)
            if dotted:
                return dotted.group(1), "trtexec", raw_output
    except Exception:
        raw_output = None
    path_match = re.search(r"TensorRT-(\\d+\\.\\d+\\.\\d+\\.\\d+)", str(trtexec_path))
    if path_match:
        return path_match.group(1), "path", raw_output
    return None, None, raw_output


def _collect_export_env(trtexec_path: Path | None) -> dict:
    env = {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "trtexec_path": str(trtexec_path) if trtexec_path else None,
    }
    if torch.cuda.is_available():
        try:
            env["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            env["gpu_name"] = None
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        version, source, raw_output = _read_trtexec_version(trtexec_path)
        env["tensorrt_version"] = version
        if source:
            env["tensorrt_version_source"] = source
        if raw_output:
            env["trtexec_version_output"] = raw_output
    else:
        env["tensorrt_version"] = trt.__version__
        env["tensorrt_version_source"] = "python"
    return env


def _export_detection_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    weights_path: Path,
    training_params: dict,
    args,
    manifest_summary: dict,
    console: Console,
) -> dict:
    export_info: dict = {"enabled": True, "errors": []}
    export_onnx = args.export_onnx or args.export_trt
    if not export_onnx:
        return export_info

    exports_root = run_dir / "exports"
    onnx_dir = exports_root / "onnx"
    trt_dir = exports_root / "tensorrt"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)

    existing_onnx_path = None
    if getattr(args, "onnx_path", None):
        existing_onnx_path = Path(args.onnx_path).expanduser().resolve()
        if not existing_onnx_path.exists():
            export_info["errors"].append(f"onnx_not_found:{existing_onnx_path}")
            return export_info

    img_h, img_w = _normalize_imgsz(training_params.get("imgsz"))
    input_shape = [1, 3, img_h, img_w]
    export_device = _resolve_export_device(training_params.get("device"))

    onnx_path = existing_onnx_path or (onnx_dir / f"{run_id}.onnx")
    onnx_log_path = onnx_dir / f"{run_id}_onnx_export.log"
    onnx_manifest_path = onnx_dir / f"{run_id}.onnx.manifest.json"
    export_info["onnx_path"] = str(onnx_path)
    export_info["onnx_log_path"] = str(onnx_log_path) if existing_onnx_path is None else None
    export_info["onnx_manifest_path"] = str(onnx_manifest_path)

    export_script = Path(__file__).resolve().parent / "export_onnx.py"
    onnx_cmd = [
        sys.executable,
        str(export_script),
        "-w",
        str(weights_path),
        "--input-shape",
        *[str(v) for v in input_shape],
        "--device",
        export_device,
        "--opset",
        str(args.onnx_opset),
        "--conf-thres",
        str(args.nms_conf),
        "--iou-thres",
        str(args.nms_iou),
        "--topk",
        str(args.nms_topk),
        "--output-path",
        str(onnx_path),
    ]
    if args.onnx_simplify:
        onnx_cmd.append("--sim")
    export_info["onnx_command"] = onnx_cmd

    if existing_onnx_path is None:
        if export_script.exists():
            console.print("[bold cyan]Exporting ONNX...[/bold cyan]")
            ok = _run_subprocess(onnx_cmd, console, "ONNX export", log_path=onnx_log_path)
            if not ok:
                export_info["errors"].append("onnx_export_failed")
                return export_info
        else:
            export_info["errors"].append(f"export_script_missing:{export_script}")
            return export_info
    else:
        console.print(f"[cyan]Using existing ONNX:[/cyan] {onnx_path}")

    weights_sha = _sha256_file(weights_path)
    onnx_sha = _sha256_file(onnx_path)
    export_info["weights_sha256"] = weights_sha
    export_info["onnx_sha256"] = onnx_sha
    export_info["onnx_source"] = "existing" if existing_onnx_path else "exported"

    onnx_manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "weights": {
            "path": str(weights_path),
            "sha256": weights_sha,
        },
        "onnx": {
            "path": str(onnx_path),
            "sha256": onnx_sha,
        },
        "export": {
            "source": "existing" if existing_onnx_path else "exported",
            "input_shape": input_shape,
            "imgsz": [img_h, img_w],
            "opset": args.onnx_opset,
            "simplify": bool(args.onnx_simplify),
            "nms": {
                "conf": args.nms_conf,
                "iou": args.nms_iou,
                "topk": args.nms_topk,
            },
            "device": export_device,
            "command": onnx_cmd if existing_onnx_path is None else None,
        },
        "logs": {
            "onnx_export": str(onnx_log_path) if existing_onnx_path is None else None,
        },
        "source_manifest": {
            "manifest_path": manifest_summary.get("manifest_path"),
            "manifest_sha256": manifest_summary.get("manifest_sha256"),
            "manifest_dataset_ids": manifest_summary.get("manifest_dataset_ids"),
        },
    }
    onnx_manifest_path.write_text(json.dumps(onnx_manifest, indent=2))

    if not args.export_trt:
        return export_info

    engine_name = f"{run_id}_{args.trt_precision}"
    engine_path = trt_dir / f"{engine_name}.engine"
    manifest_path = trt_dir / f"{engine_name}.manifest.json"
    trt_log_path = trt_dir / f"{engine_name}_trtexec.log"
    export_info["trt_log_path"] = str(trt_log_path)

    trtexec_path = Path(args.trtexec) if args.trtexec else None
    trt_script = Path(__file__).resolve().parent / "onnx_to_tensorrt.py"
    trt_cmd = [
        sys.executable,
        str(trt_script),
        "--onnx",
        str(onnx_path),
        "--engine",
        str(engine_path),
        "--precision",
        args.trt_precision,
    ]
    if args.trtexec:
        trt_cmd.extend(["--trtexec", args.trtexec])
    if args.trt_cuda_graph:
        trt_cmd.append("--cuda-graph")
    if args.trt_profiling:
        trt_cmd.append("--profiling")
    if args.trt_verbose:
        trt_cmd.append("--verbose")
    export_info["trt_command"] = trt_cmd

    if trt_script.exists():
        console.print("[bold cyan]Building TensorRT engine...[/bold cyan]")
        ok = _run_subprocess(trt_cmd, console, "TensorRT export", log_path=trt_log_path)
        if not ok:
            export_info["errors"].append("tensorrt_export_failed")
            return export_info
    else:
        export_info["errors"].append(f"tensorrt_script_missing:{trt_script}")
        return export_info

    if engine_path.exists():
        engine_sha = _sha256_file(engine_path)
        engine_manifest = {
            "schema_version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "weights": {
                "path": str(weights_path),
                "sha256": weights_sha,
            },
            "onnx": {
                "path": str(onnx_path),
                "sha256": onnx_sha,
            },
            "engine": {
                "path": str(engine_path),
                "sha256": engine_sha,
            },
            "onnx_manifest_path": str(onnx_manifest_path),
            "export": {
                "precision": args.trt_precision,
                "input_shape": input_shape,
                "imgsz": [img_h, img_w],
                "opset": args.onnx_opset,
                "nms": {
                    "conf": args.nms_conf,
                    "iou": args.nms_iou,
                    "topk": args.nms_topk,
                },
                "device": export_device,
            },
            "trt": {
                "precision": args.trt_precision,
                "cuda_graph": bool(args.trt_cuda_graph),
                "profiling": bool(args.trt_profiling),
                "verbose": bool(args.trt_verbose),
                "trtexec_path": str(trtexec_path) if trtexec_path else None,
                "command": trt_cmd,
            },
            "logs": {
                "onnx_export": str(onnx_log_path),
                "tensorrt_export": str(trt_log_path),
            },
            "build_env": _collect_export_env(trtexec_path),
            "source_manifest": {
                "manifest_path": manifest_summary.get("manifest_path"),
                "manifest_sha256": manifest_summary.get("manifest_sha256"),
                "manifest_dataset_ids": manifest_summary.get("manifest_dataset_ids"),
            },
        }
        manifest_path.write_text(json.dumps(engine_manifest, indent=2))

    export_info.update(
        {
            "engine_path": str(engine_path),
            "engine_manifest_path": str(manifest_path),
            "engine_sha256": _safe_sha256_file(engine_path),
            "engine_manifest_sha256": _safe_sha256_file(manifest_path),
            "onnx_manifest_sha256": _safe_sha256_file(onnx_manifest_path),
        }
    )
    return export_info


def main(args):
    console = Console()
    console.print("[bold cyan] Starting YOLO Detection Training[/bold cyan]\n")
    invocation_payload = build_invocation_record(
        tool="fisheye.training.train_detection",
        args=args,
    ) if args.log_registry else None
    config_path = Path(args.config_path) if args.config_path else None
    manifest_path = Path(args.manifest) if args.manifest else None
    manifest_summary = _load_manifest_summary(args.manifest)
    effective_set_id = args.set_id or manifest_summary.get("manifest_set_id")
    autogenerated_run_name = _build_default_run_name(
        manifest_summary=manifest_summary,
        task_fallback="detect",
    )
    effective_run_name = args.run_name or autogenerated_run_name
    registry_run_id = effective_run_name

    try:
        # Load and validate config
        full_config = DetectConfig.from_yaml(args.config_path)
        allow_source_mismatch = bool(full_config.allow_source_mismatch or args.allow_source_mismatch)
        
        # Extract dataset config fields from flat structure
        zarr_config_dict = {
            'datasets': full_config.datasets,
            'task': full_config.task,
            'random_seed': full_config.random_seed,
            'sampling_strategy': full_config.sampling_strategy,
            'dataset_weights': full_config.dataset_weights,
            'allow_source_mismatch': allow_source_mismatch,
        }
        config = ZarrDatasetConfig(**zarr_config_dict)
        console.print(f"[bold green]✓ Loaded config:[/bold green] {args.config_path}\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error loading config:[/bold red] {e}")
        if args.log_registry:
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=config_path,
                manifest_path=manifest_path,
                model_path=None,
                metrics_path=None,
                status="failed",
                final_metrics={
                    "stage": "config_load",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        return

    if args.log_registry:
        _record_registry_training_run(
            args=args,
            console=console,
            invocation_payload=invocation_payload,
            run_id=registry_run_id,
            set_id=effective_set_id,
            config_path=config_path,
            manifest_path=manifest_path,
            model_path=None,
            metrics_path=None,
            status="in_progress",
            final_metrics={
                "stage": "preflight_and_training",
                "status_detail": "training_started",
            },
        )

    # Get comprehensive zarr metadata
    console.print("[bold cyan] Analyzing Zarr Files...[/bold cyan]\n")
    zarr_metadata = get_zarr_metadata(config.get_zarr_paths(), console)
    display_zarr_metadata(zarr_metadata, console)

    source_mismatches = _collect_source_mismatches(full_config, zarr_metadata)
    if source_mismatches:
        if allow_source_mismatch:
            console.print(
                "[yellow]⚠ Source-type mismatches detected; proceeding because allow_source_mismatch is enabled.[/yellow]"
            )
            for mismatch in source_mismatches:
                console.print(
                    "[yellow]  - {name}: requested={requested}, available={available} ({path})[/yellow]".format(
                        name=mismatch["dataset_name"],
                        requested=mismatch["requested_source_type"],
                        available=mismatch["available_source_type"],
                        path=mismatch.get("available_source_path") or "unknown path",
                    )
                )
            console.print()
        else:
            details = "; ".join(
                f"{item['dataset_name']}: requested={item['requested_source_type']} available={item['available_source_type']}"
                for item in source_mismatches
            )
            if args.log_registry:
                _record_registry_training_run(
                    args=args,
                    console=console,
                    invocation_payload=invocation_payload,
                    run_id=registry_run_id,
                    set_id=effective_set_id,
                    config_path=config_path,
                    manifest_path=manifest_path,
                    model_path=None,
                    metrics_path=None,
                    status="failed",
                    final_metrics={
                        "stage": "source_type_validation",
                        "error_type": "ValueError",
                        "error_message": details,
                    },
                )
            raise ValueError(
                "Dataset source_type mismatch detected: "
                f"{details}. Re-run crop/curation to match source_type, "
                "or pass --allow-source-mismatch."
            )
    
    # Check for interpolated data
    has_interpolated = any(
        meta.get('crop_info', {}).get('includes_interpolated', False) 
        for meta in zarr_metadata.values() 
        if 'error' not in meta
    )
    
    if has_interpolated:
        console.print("[yellow] Warning: Some datasets include interpolated (synthetic) data[/yellow]")
        console.print("[dim]  To exclude synthetic rows, use source_type=filtered/detect/manual in dataset config[/dim]\n")

    # Setup dataloader
    def get_zarr_dataloader(self, dataset_path, batch_size=16, **kwargs):
        mode = kwargs.get('mode', 'train')
        dataset = create_zarr_dataset(config=config, mode=mode)
        return YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            collate_fn=det_collate_fn,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False
        )

    DetTrainer.get_dataloader = get_zarr_dataloader

    # Get training params
    training_params = full_config.training_params.model_dump(exclude_none=True)
    _resolve_project_dir(
        args=args,
        training_params=training_params,
        set_id=effective_set_id,
        config_path=config_path,
        console=console,
    )
    model_name = training_params.get('model', 'yolov8n.pt')
    try:
        model = YOLO(model_name)
    except Exception as exc:
        if args.log_registry:
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=config_path,
                manifest_path=manifest_path,
                model_path=None,
                metrics_path=None,
                status="failed",
                final_metrics={
                    "stage": "model_init",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise

    # Display hyperparameters
    params_str = json.dumps(training_params, indent=2)
    console.print(Panel(
        params_str,
        title="[bold yellow]Training Hyperparameters[/bold yellow]",
        expand=False
    ))
    console.print()

    # Start training
    console.print("[bold green] Starting Training...[/bold green]\n")
    training_start_time = time.time()

    snapshot_state = {"done": False}

    def _on_train_start(trainer) -> None:
        if snapshot_state["done"]:
            return
        snapshot_state["done"] = True
        try:
            run_dir = Path(trainer.save_dir)
            written = _snapshot_training_inputs(
                run_dir=run_dir,
                config_path=config_path,
                manifest_path=manifest_path,
                invocation_payload=invocation_payload,
            )
            if written:
                console.print(f"[cyan]Snapshotted run inputs:[/cyan] {run_dir / 'inputs'}")
        except Exception as exc:
            console.print(f"[yellow]Warning: failed to snapshot run inputs: {exc}[/yellow]")

    model.add_callback("on_train_start", _on_train_start)
    
    try:
        results = model.train(
            trainer=DetTrainer,
            data=args.config_path,
            name=effective_run_name,
            **training_params
        )
    except Exception as exc:
        console.print(f"[bold red]✗ Training failed:[/bold red] {exc}")
        if args.log_registry:
            _record_registry_training_run(
                args=args,
                console=console,
                invocation_payload=invocation_payload,
                run_id=registry_run_id,
                set_id=effective_set_id,
                config_path=config_path,
                manifest_path=manifest_path,
                model_path=None,
                metrics_path=None,
                status="failed",
                final_metrics={
                    "stage": "model_train",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise
    
    training_duration_seconds = time.time() - training_start_time

    export_artifacts = {}
    if args.export_onnx or args.export_trt:
        weights_path = results.save_dir / "weights" / "best.pt"
        export_artifacts = _export_detection_artifacts(
            run_dir=results.save_dir,
            run_id=results.save_dir.name,
            weights_path=weights_path,
            training_params=training_params,
            args=args,
            manifest_summary=manifest_summary,
            console=console,
        )

    # Log training metadata
    console.print("\n[bold cyan] Logging Training Report...[/bold cyan]")
    final_validation_metrics = None
    try:
        git_info = get_git_info()
        results_df = pd.read_csv(results.save_dir / 'results.csv')
        results_df.columns = results_df.columns.str.strip()
        last_epoch_metrics = results_df.iloc[-1]
        final_validation_metrics = {
            'precision': float(last_epoch_metrics.get('metrics/precision(B)', 0)),
            'recall': float(last_epoch_metrics.get('metrics/recall(B)', 0)),
            'mAP50': float(last_epoch_metrics.get('metrics/mAP50(B)', 0)),
            'mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(B)', 0))
        }

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
        final_config_filename = f"{timestamp}_detection_training_report.yaml"
        final_config_path = results.save_dir / final_config_filename
        
        # Build comprehensive training report
        final_report = full_config.model_dump()
        final_report['training_history'] = {
            'source_zarr_metadata': zarr_metadata,
            'source_type_resolution': {
                'allow_source_mismatch': bool(allow_source_mismatch),
                'mismatch_count': len(source_mismatches),
                'mismatches': source_mismatches,
            },
            **manifest_summary,
            'export_artifacts': export_artifacts,
            'training_run_name': results.save_dir.name,
            'output_directory': str(results.save_dir),
            'final_model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'training_start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time)),
            'training_duration_hours': round(training_duration_seconds / 3600, 2),
            'git_commit_hash': git_info.get('commit_hash', 'N/A'),
            'git_branch': git_info.get('branch', 'N/A'),
            'python_version': platform.python_version(),
            'torch_version': str(torch.__version__),
            'ultralytics_version': str(ultralytics_version),
            'cuda_available': torch.cuda.is_available(),
            'final_training_losses': {
                'box_loss': float(last_epoch_metrics.get('train/box_loss', 0)),
                'cls_loss': float(last_epoch_metrics.get('train/cls_loss', 0)),
                'dfl_loss': float(last_epoch_metrics.get('train/dfl_loss', 0)),
            },
            'final_validation_metrics': final_validation_metrics
        }
        
        # Save report
        with open(final_config_path, 'w') as f:
            yaml.dump(final_report, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[bold green]✓ Training report saved:[/bold green] {final_config_path}")
        
        # Display final metrics
        metrics_table = Table(title=" Final Training Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics_table.add_row("Precision", f"{final_report['training_history']['final_validation_metrics']['precision']:.3f}")
        metrics_table.add_row("Recall", f"{final_report['training_history']['final_validation_metrics']['recall']:.3f}")
        metrics_table.add_row("mAP50", f"{final_report['training_history']['final_validation_metrics']['mAP50']:.3f}")
        metrics_table.add_row("mAP50-95", f"{final_report['training_history']['final_validation_metrics']['mAP50_95']:.3f}")
        metrics_table.add_row("Training Time", f"{final_report['training_history']['training_duration_hours']:.2f} hours")
        
        console.print(metrics_table)

    except Exception as e:
        console.print(f"[bold red]✗ Could not save training report:[/bold red] {e}")
        traceback.print_exc()
    
    if args.log_registry:
        model_path = results.save_dir / "weights" / "best.pt"
        metrics_path = results.save_dir / "results.csv"
        final_metrics_payload = dict(final_validation_metrics or {})
        final_metrics_payload.setdefault("stage", "completed")
        final_metrics_payload.setdefault("status_detail", "training_complete")
        _record_registry_training_run(
            args=args,
            console=console,
            invocation_payload=invocation_payload,
            run_id=registry_run_id,
            set_id=effective_set_id,
            config_path=config_path,
            manifest_path=manifest_path,
            model_path=model_path if model_path.exists() else None,
            metrics_path=metrics_path if metrics_path.exists() else None,
            status="success",
            final_metrics=final_metrics_payload,
            export_artifacts=export_artifacts,
        )

    console.print("\n[bold green]✓ Training Complete![/bold green]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-Zarr YOLO Detection Trainer with Enhanced Metadata Tracking"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training configuration YAML"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional name for the training run directory"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Optional output project directory for Ultralytics runs (overrides config/default).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Optional manifest JSON path to record in the registry."
    )
    parser.add_argument(
        "--set-id",
        type=str,
        help="Optional training set ID to associate with this run. Defaults to manifest set_id when available."
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help=(
            "Allow fallback to available crop source type when it differs from "
            "requested dataset source_type; mismatches are recorded in training report."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path."
    )
    parser.add_argument(
        "--log-registry",
        dest="log_registry",
        action="store_true",
        default=True,
        help="Record this training run in the registry (default: enabled)."
    )
    parser.add_argument(
        "--no-log-registry",
        dest="log_registry",
        action="store_false",
        help="Disable registry logging for this training run."
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the trained model to ONNX."
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Export the trained model to a TensorRT engine (implies --export-onnx)."
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="ONNX opset to use for export."
    )
    parser.add_argument(
        "--onnx-simplify",
        action="store_true",
        help="Run ONNX simplification after export."
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Optional existing ONNX path to reuse (skips ONNX export)."
    )
    parser.add_argument(
        "--nms-conf",
        type=float,
        default=0.8,
        help="NMS confidence threshold baked into the ONNX export."
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.65,
        help="NMS IoU threshold baked into the ONNX export."
    )
    parser.add_argument(
        "--nms-topk",
        type=int,
        default=1,
        help="Max detections for the ONNX NMS export."
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        default="fp16",
        help="Precision to use for TensorRT export."
    )
    parser.add_argument(
        "--trtexec",
        type=str,
        default=None,
        help="Optional path to trtexec for TensorRT export."
    )
    parser.add_argument(
        "--trt-cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture during TensorRT export."
    )
    parser.add_argument(
        "--trt-profiling",
        action="store_true",
        help="Enable TensorRT profiling outputs (timing/output/profile JSON)."
    )
    parser.add_argument(
        "--trt-verbose",
        action="store_true",
        help="Enable verbose TensorRT build logs."
    )
    args = parser.parse_args()
    main(args)
