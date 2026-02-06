# src/fisheye/training/train_keypoints.py

"""
Keypoint/Pose YOLO Trainer from Zarr files with Enhanced Metadata Logging

Features:
- Trains pose estimation models on ROI images with keypoint annotations
- Requires tracking_runs with keypoint data (bladder, eye_l, eye_r)
- Tracks crop source (detect/filtered/interpolated)
- Complete provenance tracking
- Enhanced training reports with tracking success rates

Usage:
    python -m fisheye.training.train_keypoints path/to/pose_config.yaml --run-name my_pose_model
"""

import argparse
import os
import re
import shutil
import numpy as np
# Import NumPy before Torch to avoid MKL/libgomp threading-layer conflicts in some conda envs.
import torch
import yaml
from pathlib import Path
import time
import platform
import traceback
import pandas as pd
from hashlib import sha256
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json
import zarr
from typing import Any, Dict, Optional

from .config import PoseConfig
from .zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
from ..registry.db import Registry, RegistryPaths
from ..utils.system import build_invocation_record, get_git_info


# Custom DataLoader to ensure compatibility with Ultralytics YOLO's expected interface
class YoloCompatibleDataLoader(DataLoader):
    """Minimal wrapper to add reset() method required by YOLO."""
    def reset(self):
        pass


def pose_collate_fn(batch):
    """
    Custom collate function to handle batches of pose data, including keypoints.
    This safely handles cases where some samples may not have labels and reshapes keypoints.
    
    Args:
        batch: List of sample dicts with keys: 'img', 'cls', 'bboxes', 'keypoints', etc.
    
    Returns:
        Dict with batched tensors for training
    """
    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
    im_files = [s.get('im_file', f'image_{i}.jpg') for i, s in enumerate(batch)]
    ori_shapes = [s.get('ori_shape', (images.shape[2], images.shape[3])) for s in batch]
    # Provide safe default ratio_pad if missing: ((h_ratio, w_ratio), (pad_w, pad_h))
    ratio_pads = [s.get('ratio_pad', ((1.0, 1.0), (0.0, 0.0))) for s in batch]
    
    cls_list, bboxes_list, keypoints_list, batch_idx_list = [], [], [], []
    
    for i, sample in enumerate(batch):
        cls_labels = np.atleast_1d(sample['cls'])
        if cls_labels.size > 0 and cls_labels[0] is not None:
            num_instances = len(cls_labels)
            # Ensure 2D (N,1) to satisfy Ultralytics validator which does .squeeze(-1)
            cls_list.append(torch.from_numpy(cls_labels).reshape(-1, 1))
            bboxes_list.append(torch.from_numpy(sample['bboxes']))
            
            # Handle keypoints - reshape to (num_instances, num_keypoints, 3)
            # Expected format: [x, y, visibility] for each keypoint
            if 'keypoints' in sample and sample['keypoints'].size > 0:
                kpts = torch.from_numpy(sample['keypoints'])
                kpts = kpts.view(num_instances, -1, 3)  # Reshape to (N, K, 3)
                keypoints_list.append(kpts)
            else:
                # Create empty keypoints if none provided (shouldn't happen with pose task)
                num_kpts = 3  # Default: bladder, eye_l, eye_r
                keypoints_list.append(torch.zeros((num_instances, num_kpts, 3)))

            # Use long dtype for batch indices (required for indexing)
            batch_idx_list.append(torch.full((num_instances,), i, dtype=torch.long))

    # Handle empty batch (all samples had no valid labels)
    if not batch_idx_list:
        return {
            'img': images, 
            'batch_idx': torch.empty(0, dtype=torch.long), 
            'cls': torch.empty(0, 1, dtype=torch.float32), 
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'keypoints': torch.empty(0, 3, 3, dtype=torch.float32),  # (0, 3 keypoints, 3 coords)
            'im_file': im_files, 
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }
        
    return {
        'img': images, 
        'batch_idx': torch.cat(batch_idx_list, 0), 
        'cls': torch.cat(cls_list, 0).float(),
        'bboxes': torch.cat(bboxes_list, 0).float(),
        'keypoints': torch.cat(keypoints_list, 0).float(),
        'im_file': im_files, 
        'ori_shape': ori_shapes,
        'ratio_pad': ratio_pads
    }


class ZarrPoseValidator(PoseValidator):
    """Custom validator that handles edge cases in pose data."""
    
    def _debug_batch_shapes(self, batch):
        try:
            def shape(x):
                import torch as _t
                return tuple(x.shape) if isinstance(x, _t.Tensor) else None
            imgs = batch.get('img', None)
            cls = batch.get('cls', None)
            bxs = batch.get('bboxes', None)
            kpts = batch.get('keypoints', None)
            bidx = batch.get('batch_idx', None)
            rp_list = batch.get('ratio_pad', None)
            rp0 = None
            if isinstance(rp_list, (list, tuple)) and len(rp_list) > 0:
                rp0 = rp_list[0]
            # Try to read kpt_shape from validator's data if present
            kpt_shape_cfg = None
            try:
                if hasattr(self, 'data'):
                    if isinstance(self.data, dict):
                        kpt_shape_cfg = self.data.get('kpt_shape', None)
                    else:
                        kpt_shape_cfg = getattr(self.data, 'kpt_shape', None)
            except Exception:
                pass
            msg1 = "[DEBUG] batch shapes img={} cls={} bboxes={} keypoints={} batch_idx={}\n".format(
                shape(imgs), shape(cls), shape(bxs), shape(kpts), shape(bidx))
            msg2 = f"[DEBUG] ratio_pad[0]={rp0} kpt_shape_cfg={kpt_shape_cfg}\n"

            # Prefer writing to validator save_dir to avoid console flood
            log_path = None
            try:
                from pathlib import Path as _P
                sd = getattr(self, 'save_dir', None)
                if sd is not None:
                    log_path = _P(sd) / 'prepare_batch_debug.log'
            except Exception:
                log_path = None

            if log_path is not None:
                try:
                    with open(log_path, 'a') as f:
                        f.write(msg1)
                        f.write(msg2)
                except Exception:
                    # Fallback to printing
                    print(msg1, end='')
                    print(msg2, end='')
            else:
                print(msg1, end='')
                print(msg2, end='')
        except Exception:
            # Swallow any debug errors to avoid masking real failure
            pass

    def _prepare_batch(self, si, batch):
        """Prepare batch for validation, handling shape issues."""
        # Deep copy to avoid modifying original
        import copy
        batch = copy.copy(batch)
        
        # Fix cls BEFORE calling super() to prevent shape errors in parent
        if 'cls' in batch:
            cls = batch['cls']
            
            # Handle various input types
            if cls is None:
                batch['cls'] = torch.empty(0, dtype=torch.float32)
            elif not isinstance(cls, torch.Tensor):
                # Convert to tensor
                if hasattr(cls, '__array__'):
                    cls = torch.from_numpy(np.asarray(cls))
                elif isinstance(cls, (list, tuple)):
                    cls = torch.tensor(cls)
                else:
                    cls = torch.tensor([cls])
                batch['cls'] = cls
            
            cls = batch['cls']
            
            # Now fix shape issues: ensure 2D (N,1) because parent does .squeeze(-1)
            if isinstance(cls, torch.Tensor):
                if cls.numel() == 0:
                    batch['cls'] = torch.empty(0, 1, dtype=torch.float32)
                elif cls.ndim == 0:  # scalar -> (1,1)
                    batch['cls'] = cls.view(1, 1).float()
                elif cls.ndim == 1:  # (N,) -> (N,1)
                    batch['cls'] = cls.unsqueeze(1).float()
                elif cls.ndim >= 2:  # keep at least last dim as singleton
                    if cls.shape[-1] != 1:
                        batch['cls'] = cls.view(-1, 1).float()
                    else:
                        batch['cls'] = cls.float()
        
        # Now call parent with fixed batch
        try:
            pbatch = super()._prepare_batch(si, batch)
        except Exception as e:
            # If parent still fails, create minimal valid batch
            print(f"Warning: _prepare_batch failed with {e}, creating minimal batch")
            # Optional debug logging gated by env var to avoid noisy output
            import os as _os
            if _os.getenv('FISHEYE_PREP_DEBUG', '0') == '1':
                if not hasattr(self, '_prep_debugged'):
                    self._prep_debugged = False
                if not self._prep_debugged:
                    # Write traceback to the same debug log for clarity
                    import traceback as _tb
                    try:
                        self._debug_batch_shapes(batch)
                        # Append the traceback
                        log_path = None
                        try:
                            from pathlib import Path as _P
                            sd = getattr(self, 'save_dir', None)
                            if sd is not None:
                                log_path = _P(sd) / 'prepare_batch_debug.log'
                        except Exception:
                            log_path = None
                        tb_str = ''.join(_tb.format_exception(type(e), e, e.__traceback__))
                        if log_path is not None:
                            with open(log_path, 'a') as f:
                                f.write("[DEBUG] exception traceback follows\n")
                                f.write(tb_str)
                        else:
                            print(tb_str)
                    except Exception:
                        pass
                    self._prep_debugged = True
            pbatch = batch
            if 'cls' not in pbatch or pbatch['cls'] is None:
                pbatch['cls'] = torch.empty(0, dtype=torch.float32)
        
        return pbatch


class ZarrPoseTrainer(PoseTrainer):
    """Custom pose trainer that uses Zarr dataset loader."""
    
    def get_validator(self):
        """Return custom validator with proper loss names."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return ZarrPoseValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=self.args, 
            _callbacks=self.callbacks
        )

    def get_dataloader(self, dataset_path, batch_size=16, mode='train', rank=0):
        """
        Create DataLoader for Zarr dataset.
        
        This method is called by YOLO during training setup.
        The config is accessed through the global variable set in main().
        """
        try:
            global config
            dataset = create_zarr_dataset(config=config, mode=mode)
            return YoloCompatibleDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(mode == 'train'),
                collate_fn=pose_collate_fn,
                num_workers=8,
                pin_memory=True,
                persistent_workers=False
            )
        except Exception as e:
            print(f"Error creating dataloader: {e}")
            raise


def get_zarr_metadata(zarr_paths, console=None):
    """
    Extract comprehensive metadata from zarr files including crop and tracking info.
    
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
                'tracking_info': {},
                'data_quality': {}
            }
            
            # Get video info
            if 'raw_video' in root:
                if 'images_full' in root['raw_video']:
                    zarr_meta['video_frames'] = root['raw_video/images_full'].shape[0]
                zarr_meta['fps'] = root['raw_video'].attrs.get('fps', 'N/A')
            
            # Get crop info
            if 'crop_runs' in root and 'latest' in root['crop_runs'].attrs:
                latest_crop = root['crop_runs'].attrs['latest']
                crop_group = root[f'crop_runs/{latest_crop}']
                
                zarr_meta['crop_info'] = {
                    'run_name': latest_crop,
                    'source_type': crop_group.attrs.get('detection_source_type', 'unknown'),
                    'n_rois': crop_group['roi_images'].shape[0] if 'roi_images' in crop_group else 0,
                    'roi_size': tuple(crop_group['roi_images'].shape[1:3]) if 'roi_images' in crop_group else (0, 0),
                    'includes_interpolated': crop_group.attrs.get('includes_interpolated', False),
                    'n_real_detections': crop_group.attrs.get('n_real_detections', 0),
                    'n_interpolated_detections': crop_group.attrs.get('n_interpolated_detections', 0)
                }
            
            # Get keypoint detection info if available
            if 'keypoints_runs' in root and 'latest' in root['keypoints_runs'].attrs:
                latest_kp = root['keypoints_runs'].attrs['latest']
                kp_group = root[f'keypoints_runs/{latest_kp}']
                zarr_meta['tracking_info'] = {
                    'run_name': latest_kp,
                    'keypoints_processed': int(kp_group.attrs.get('keypoints_processed', 0)),
                    'success_rate': float(kp_group.attrs.get('success_rate', 0.0))
                }
            else:
                zarr_meta['tracking_info'] = {'warning': 'keypoints_runs not found; proceeding without precomputed keypoints metadata'}
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
    
    return metadata


def _safe_sha256_file(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest_set_id(manifest_path: Optional[str]) -> Optional[str]:
    return _load_manifest_run_hints(manifest_path).get("set_id")


def _load_manifest_run_hints(manifest_path: Optional[str]) -> Dict[str, Optional[str]]:
    hints: Dict[str, Optional[str]] = {
        "set_id": None,
        "dish_design": None,
        "canvas_name": None,
        "task": None,
    }
    if not manifest_path:
        return hints
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return hints

    if not isinstance(payload, dict):
        return hints

    set_id = payload.get("set_id")
    if isinstance(set_id, str) and set_id.strip():
        hints["set_id"] = set_id.strip()

    query_filter = payload.get("query_filter")
    if not isinstance(query_filter, dict):
        query_filter = {}
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
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
    hints["canvas_name"] = (
        first_dataset.get("canvas_name")
        or rig_info.get("canvas_name")
        or _infer_canvas_from_dataset_label(dataset_name)
        or _infer_canvas_from_dataset_label(Path(str(dataset_zarr)).stem if dataset_zarr else None)
    )
    hints["dish_design"] = (
        query_filter.get("dish_design")
        or first_dataset.get("dish_design")
        or arena.get("dish_design")
    )
    task = payload.get("task")
    if isinstance(task, str) and task.strip():
        hints["task"] = task.strip()
    return hints


def _strip_manifest_suffixes(value: str) -> str:
    text = value
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _infer_canvas_from_dataset_label(value: Optional[str]) -> Optional[str]:
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


def _sanitize_run_component(value: Optional[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


def _build_default_run_name(
    *,
    manifest_hints: Dict[str, Optional[str]],
    task_fallback: str,
    timestamp: Optional[str] = None,
    pid: Optional[int] = None,
) -> str:
    dish = _sanitize_run_component(manifest_hints.get("dish_design"), "unknown_dish")
    canvas = _sanitize_run_component(manifest_hints.get("canvas_name"), "unknown_canvas")
    task = _sanitize_run_component(manifest_hints.get("task") or task_fallback, task_fallback)
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    process_id = int(os.getpid() if pid is None else pid)
    return f"{dish}_{canvas}_{task}_{stamp}_{process_id}"


def _infer_set_slug(set_id: Optional[str], config_path: Optional[Path]) -> str:
    if set_id:
        slug = _strip_manifest_suffixes(set_id)
        return slug or "pose_training"
    if config_path is not None:
        stem = _strip_manifest_suffixes(config_path.stem)
        return stem or "pose_training"
    return "pose_training"


def _resolve_project_dir(
    *,
    args,
    training_params: dict,
    set_id: Optional[str],
    config_path: Optional[Path],
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
    project_dir = (nvme_root / "models" / "pose" / slug).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    training_params["project"] = str(project_dir)
    console.print(f"[cyan]Using default model output directory:[/cyan] {project_dir}")


def _snapshot_training_inputs(
    *,
    run_dir: Path,
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    invocation_payload: Optional[Dict[str, Any]],
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
    set_id: Optional[str],
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    model_path: Optional[Path],
    metrics_path: Optional[Path],
    status: str,
    final_metrics: Optional[Dict[str, Any]],
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
        console.print(f"[green]✓ Registry updated:[/green] {registry_path}")
    except Exception as exc:
        console.print(f"[yellow]Registry update skipped:[/yellow] {exc}")
    finally:
        if registry is not None:
            try:
                registry.close()
            except Exception:
                pass


def main(args) -> int:
    """Main training function."""
    console = Console()
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]      Zarr YOLO Pose Training - Fisheye Module        [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    config_path = Path(args.config_path) if args.config_path else None
    manifest_path = Path(args.manifest) if args.manifest else None
    manifest_hints = _load_manifest_run_hints(args.manifest)
    manifest_set_id = manifest_hints.get("set_id")
    effective_set_id = args.set_id or manifest_set_id
    autogenerated_run_name = _build_default_run_name(
        manifest_hints=manifest_hints,
        task_fallback="pose",
    )
    effective_run_name = args.run_name or autogenerated_run_name
    registry_run_id = effective_run_name
    invocation_payload = (
        build_invocation_record(
            tool="fisheye.training.train_keypoints",
            args=args,
        )
        if args.log_registry
        else None
    )

    # Load and validate config
    global config
    try:
        full_config = PoseConfig.from_yaml(args.config_path)

        datasets_dict = {}
        for name, ds_cfg in full_config.datasets.items():
            split_dict = None
            if ds_cfg.split is not None:
                split_dict = {
                    'train': ds_cfg.split.train,
                    'val': ds_cfg.split.val
                }
            datasets_dict[name] = {
                'zarr_path': str(ds_cfg.zarr_path),
                'source_type': ds_cfg.source_type.value if hasattr(ds_cfg.source_type, 'value') else ds_cfg.source_type,
                'input_format': ds_cfg.input_format,
                'keypoint_run': ds_cfg.keypoint_run,
                'split': split_dict
            }

        default_split = 0.8
        if full_config.datasets:
            first_ds = next(iter(full_config.datasets.values()))
            if first_ds.split is not None:
                default_split = first_ds.split.train

        config = ZarrDatasetConfig(
            datasets=datasets_dict,
            task=full_config.task,
            sampling_strategy=full_config.sampling_strategy.value if hasattr(full_config.sampling_strategy, 'value') else full_config.sampling_strategy,
            random_seed=full_config.random_seed,
            dataset_weights=full_config.dataset_weights,
            split_ratio=default_split
        )
        console.print(f"[bold green]✓ Loaded configuration:[/bold green] {args.config_path}\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error loading config:[/bold red] {e}")
        traceback.print_exc()
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
        return 1
    
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

    # Get zarr metadata
    zarr_paths = config.get_zarr_paths()
    zarr_metadata = get_zarr_metadata(zarr_paths, console)
    
    # Display dataset info
    console.print("[bold cyan]Dataset Information[/bold cyan]")
    console.rule()
    
    for name, meta in zarr_metadata.items():
        if 'error' in meta:
            console.print(f"  [red]✗ {name}[/red]: {meta['error']}")
            continue
        
        console.print(f"\n  [green]{name}[/green]")
        console.print(f"    • Video: {meta.get('video_frames', 0)} frames @ {meta.get('fps', 'N/A')} fps")
        
        crop_info = meta.get('crop_info', {})
        console.print(f"    • Crops: {crop_info.get('n_rois', 0)} ROIs ({crop_info.get('roi_size', 'N/A')})")
        console.print(f"      - Source: {crop_info.get('source_type', 'unknown')}")
        if crop_info.get('includes_interpolated'):
            console.print(f"      - Real: {crop_info.get('n_real_detections', 0)} | "
                         f"Interpolated: {crop_info.get('n_interpolated_detections', 0)}")
        
        track_info = meta.get('tracking_info', {})
        if 'warning' in track_info:
            console.print(f"    • Keypoints: {track_info['warning']}")
        else:
            console.print(f"    • Keypoints: {track_info.get('run_name', 'N/A')} "
                         f"(processed={track_info.get('keypoints_processed', 0)}, "
                         f"success={track_info.get('success_rate', 0.0):.2f})")
    
    console.print()
    
    # Verify all datasets have tracking data
    missing_tracking = [name for name, meta in zarr_metadata.items() 
                       if 'warning' in meta.get('tracking_info', {})]
    if missing_tracking:
        console.print(f"[bold yellow]⚠ The following datasets are missing precomputed keypoint metadata:[/bold yellow]")
        for name in missing_tracking:
            console.print(f"  - {name}")
   
    # Get training params
    training_params = full_config.training_params.model_dump(exclude_none=True)
    _resolve_project_dir(
        args=args,
        training_params=training_params,
        set_id=effective_set_id,
        config_path=config_path,
        console=console,
    )
    model_name = training_params.get('model', 'yolov8n-pose.pt')
    
    # Display hyperparameters
    console.print("[bold yellow]Training Hyperparameters[/bold yellow]")
    console.rule()
    params_table = Table(show_header=False, box=None, padding=(0, 2))
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="yellow")
    
    for key, value in training_params.items():
        params_table.add_row(key, str(value))
    
    console.print(params_table)
    console.print()
    
    # Initialize model
    console.print(f"[bold]Loading model:[/bold] {model_name}")
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
    
    # Monkey-patch the trainer's get_dataloader method
    def get_zarr_dataloader(trainer_self, dataset_path, batch_size, mode, rank=0):
        dataset = create_zarr_dataset(config=config, mode=mode)
        return YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            collate_fn=pose_collate_fn,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False
        )
    
    ZarrPoseTrainer.get_dataloader = get_zarr_dataloader
    
    # Start training
    console.print("\n[bold green]⚡ Starting Training...[/bold green]\n")
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
            trainer=ZarrPoseTrainer,
            data=args.config_path,
            name=effective_run_name,
            **training_params
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed:[/bold red] {e}")
        traceback.print_exc()
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
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
        return 1
    
    training_duration_seconds = time.time() - training_start_time
    
    # Log training metadata
    console.print("\n[bold cyan]Generating Training Report...[/bold cyan]")
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
            'mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(B)', 0)),
            'pose_mAP50': float(last_epoch_metrics.get('metrics/mAP50(P)', 0)),
            'pose_mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(P)', 0))
        }
        
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
        final_config_filename = f"{timestamp}_pose_training_report.yaml"
        final_config_path = results.save_dir / final_config_filename
        
        # Build comprehensive training report
        final_report = full_config.model_dump()
        final_report['training_history'] = {
            'source_zarr_metadata': zarr_metadata,
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
                'pose_loss': float(last_epoch_metrics.get('train/pose_loss', 0)),
                'kobj_loss': float(last_epoch_metrics.get('train/kobj_loss', 0)),
                'cls_loss': float(last_epoch_metrics.get('train/cls_loss', 0)),
                'dfl_loss': float(last_epoch_metrics.get('train/dfl_loss', 0)),
            },
            'final_validation_metrics': final_validation_metrics
        }
        
        # Save report
        with open(final_config_path, 'w') as f:
            yaml.dump(final_report, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[bold green]✓ Training report saved:[/bold green] {final_config_path}\n")
        
        # Display final metrics
        metrics_table = Table(title="Final Training Metrics", title_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="yellow", justify="right")
        
        # Detection metrics
        metrics_table.add_row("Box Precision", f"{final_report['training_history']['final_validation_metrics']['precision']:.3f}")
        metrics_table.add_row("Box Recall", f"{final_report['training_history']['final_validation_metrics']['recall']:.3f}")
        metrics_table.add_row("Box mAP50", f"{final_report['training_history']['final_validation_metrics']['mAP50']:.3f}")
        metrics_table.add_row("Box mAP50-95", f"{final_report['training_history']['final_validation_metrics']['mAP50_95']:.3f}")
        
        # Pose metrics
        metrics_table.add_row("─" * 20, "─" * 10)
        metrics_table.add_row("Pose mAP50", f"{final_report['training_history']['final_validation_metrics']['pose_mAP50']:.3f}")
        metrics_table.add_row("Pose mAP50-95", f"{final_report['training_history']['final_validation_metrics']['pose_mAP50_95']:.3f}")
        
        # Training info
        metrics_table.add_row("─" * 20, "─" * 10)
        metrics_table.add_row("Training Time", f"{final_report['training_history']['training_duration_hours']:.2f}h")
        metrics_table.add_row("Model Path", Path(final_report['training_history']['final_model_path']).name)
        
        console.print(metrics_table)
        console.print()
        
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
        )

    console.print("[bold green]✓ Training Complete![/bold green]")
    console.print(f"[dim]Results saved to: {results.save_dir}[/dim]\n")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-Zarr YOLO Pose Trainer with Enhanced Metadata Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python -m fisheye.training.train_keypoints configs/pose_config.yaml
  
  # With custom run name
  python -m fisheye.training.train_keypoints configs/pose_config.yaml --run-name fish_pose_v1
  
  # After training, run pose inference
  python -m fisheye.inference.predict_pose --model runs/pose/fish_pose_v1/weights/best.pt --zarr video.zarr
        """
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the pose training configuration YAML"
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
    args = parser.parse_args()
    raise SystemExit(main(args))
