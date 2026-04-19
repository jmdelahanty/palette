# src/fisheye/training/train_pose.py

"""
Keypoint/Pose YOLO Trainer from Zarr files with Enhanced Metadata Logging

Features:
- Trains pose estimation models on ROI images with keypoint annotations
- Consumes ROI images with keypoint annotations from exported training datasets
- Tracks crop source (detect/filtered/interpolated)
- Complete provenance tracking
- Enhanced training reports with tracking success rates

Usage:
    python -m fisheye.training.train_pose path/to/pose_config.yaml --run-name my_pose_model
"""

import argparse
import hashlib
import shutil
import sys
import numpy as np
# Import NumPy before Torch to avoid MKL/libgomp threading-layer conflicts in some conda envs.
import torch
import yaml
from pathlib import Path
import time
import platform
import traceback
import pandas as pd
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
import json
import zarr
from typing import Any, Callable, Dict, Mapping, Optional

from ..pose.schema import resolve_keypoint_labels_from_attrs
from .config import PoseConfig
from .export_shared import (
    collect_export_env as _shared_collect_export_env,
    resolve_trtexec_path as _shared_resolve_trtexec_path,
    run_subprocess_streaming as _shared_run_subprocess_streaming,
)
from .training_run_shared import (
    record_registry_training_run as _shared_record_registry_training_run,
    safe_sha256_file as _shared_safe_sha256_file,
    snapshot_training_inputs as _shared_snapshot_training_inputs,
)
from .training_naming_shared import (
    build_default_pose_run_name as _shared_build_default_pose_run_name,
    infer_set_slug as _shared_infer_set_slug,
    resolve_project_dir as _shared_resolve_project_dir,
    sanitize_run_component as _shared_sanitize_run_component,
    strip_manifest_suffixes as _shared_strip_manifest_suffixes,
)
from .training_console import (
    print_dataset_details,
    print_section_header,
    print_training_banner,
    print_training_hyperparameters,
    print_training_start,
)
from .zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
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
    fallback_num_keypoints = 0
    for sample in batch:
        try:
            sample_count = int(sample.get("num_keypoints") or 0)
        except Exception:
            sample_count = 0
        if sample_count > 0:
            fallback_num_keypoints = sample_count
            break
        sample_keypoints = sample.get("keypoints")
        if hasattr(sample_keypoints, "size") and sample_keypoints.size > 0:
            flat_width = int(np.asarray(sample_keypoints).reshape(-1).shape[0])
            if flat_width > 0 and flat_width % 3 == 0:
                fallback_num_keypoints = flat_width // 3
                break
    
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
                # Create empty keypoints if none provided (shouldn't happen with pose task).
                num_kpts = int(fallback_num_keypoints)
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
            'keypoints': torch.empty(0, int(fallback_num_keypoints), 3, dtype=torch.float32),
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

    def _resolve_tracking_labels(kp_group: zarr.Group) -> Optional[list[str]]:
        if "keypoints_roi" not in kp_group:
            return None
        labels = resolve_keypoint_labels_from_attrs(
            dict(kp_group.attrs),
            keypoint_count=int(kp_group["keypoints_roi"].shape[1]),
        )
        return list(labels) if labels else None

    def _resolve_tracking_skeleton(kp_group: zarr.Group) -> Optional[list[list[int]]]:
        raw_skeleton = kp_group.attrs.get("keypoint_skeleton")
        if not isinstance(raw_skeleton, (list, tuple)):
            pose_schema = kp_group.attrs.get("pose_schema")
            if isinstance(pose_schema, Mapping):
                raw_skeleton = pose_schema.get("edges")
                if raw_skeleton is None:
                    raw_skeleton = pose_schema.get("skeleton")
        if not isinstance(raw_skeleton, (list, tuple)):
            return None
        edges: list[list[int]] = []
        for edge in raw_skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            try:
                edges.append([int(edge[0]), int(edge[1])])
            except Exception:
                continue
        return edges or None

    def _summarize_source_video_metadata(source_paths: list[str]) -> tuple[int, Any]:
        total_frames = 0
        fps_values: list[float] = []
        any_frame_count = False
        for src in source_paths:
            try:
                src_root = zarr.open(src, mode='r')
            except Exception:
                continue
            raw_src = src_root.get("raw_video")
            if raw_src is None:
                continue

            frame_count = None
            for array_name in ("images_full", "images_ds", "images_ds_rgb"):
                if array_name in raw_src:
                    try:
                        frame_count = int(raw_src[array_name].shape[0])
                    except Exception:
                        frame_count = None
                    if frame_count is not None:
                        break
            if frame_count is not None:
                any_frame_count = True
                total_frames += frame_count

            fps = raw_src.attrs.get("fps")
            if isinstance(fps, (int, float)):
                fps_values.append(float(fps))

        fps_value: Any = "N/A"
        if fps_values:
            if np.isclose(max(fps_values), min(fps_values), atol=1e-6):
                fps_value = float(fps_values[0])
            else:
                fps_value = f"mixed ({min(fps_values):.3f}-{max(fps_values):.3f})"
        return (total_frames if any_frame_count else 0), fps_value
    
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
            else:
                # Merged training zarrs typically omit raw_video; fallback to source zarr metadata.
                training_export = root.attrs.get("training_export")
                if isinstance(training_export, Mapping):
                    source_paths = training_export.get("source_zarr_paths")
                    if isinstance(source_paths, list):
                        normalized_paths = [str(p) for p in source_paths if p]
                        if normalized_paths:
                            fallback_frames, fallback_fps = _summarize_source_video_metadata(normalized_paths)
                            zarr_meta['video_frames'] = fallback_frames
                            zarr_meta['fps'] = fallback_fps
            
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
                usable_keypoints = None
                total_keypoints = None
                usable_keypoints_rate = None
                refined_run_name = None
                refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
                if refined_parent is not None:
                    candidates = []
                    for refined_run in refined_parent.group_keys():
                        refined_group = refined_parent[refined_run]
                        source_keypoint_run = (
                            refined_group.attrs.get("source_keypoints_run")
                            or refined_group.attrs.get("source_keypoint_run")
                        )
                        if source_keypoint_run is None or str(source_keypoint_run) != str(latest_kp):
                            continue
                        candidate_usable = None
                        candidate_total = None
                        if "usable_keypoints" in refined_group:
                            usable_arr = refined_group["usable_keypoints"]
                            candidate_total = int(usable_arr.shape[0])
                            candidate_usable = int(np.asarray(usable_arr[:]).sum())
                        summary_stats = refined_group.attrs.get("summary_statistics")
                        if isinstance(summary_stats, Mapping):
                            postprocess = summary_stats.get("postprocess")
                            for payload in (postprocess, summary_stats):
                                if not isinstance(payload, Mapping):
                                    continue
                                if candidate_usable is None:
                                    try:
                                        candidate_usable = int(payload.get("usable_keypoints"))
                                    except Exception:
                                        candidate_usable = None
                                if candidate_total is None:
                                    try:
                                        candidate_total = int(payload.get("total_rois"))
                                    except Exception:
                                        candidate_total = None
                        candidate_rate = (
                            float(candidate_usable) / float(candidate_total)
                            if candidate_usable is not None and candidate_total is not None and candidate_total > 0
                            else None
                        )
                        candidate_ts = (
                            str(refined_group.attrs.get("created_utc") or refined_group.attrs.get("timestamp_utc") or "")
                        )
                        candidates.append((candidate_ts, str(refined_run), candidate_usable, candidate_total, candidate_rate))
                    if candidates:
                        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                        _ts, refined_run_name, usable_keypoints, total_keypoints, usable_keypoints_rate = candidates[0]

                keypoint_labels = _resolve_tracking_labels(kp_group)
                keypoint_skeleton = _resolve_tracking_skeleton(kp_group)
                zarr_meta['tracking_info'] = {
                    'run_name': latest_kp,
                    'refined_run': refined_run_name,
                    'keypoints_processed': int(kp_group.attrs.get('keypoints_processed', 0)),
                    'success_rate': float(kp_group.attrs.get('success_rate', 0.0)),
                    'usable_keypoints': usable_keypoints,
                    'total_keypoints': total_keypoints,
                    'usable_keypoints_rate': usable_keypoints_rate,
                    'keypoint_labels': keypoint_labels,
                    'keypoint_skeleton': keypoint_skeleton,
                }
            else:
                zarr_meta['tracking_info'] = {'warning': 'keypoints_runs not found; proceeding without precomputed keypoints metadata'}
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
    
    return metadata


def _infer_pose_schema(kpt_shape: Optional[tuple], zarr_metadata: dict) -> dict:
    labels = None
    skeleton = None
    for meta in zarr_metadata.values():
        if not isinstance(meta, dict) or "error" in meta:
            continue
        tracking_info = meta.get("tracking_info") if isinstance(meta.get("tracking_info"), dict) else {}
        kp_labels = tracking_info.get("keypoint_labels")
        kp_skeleton = tracking_info.get("keypoint_skeleton")
        if not labels and isinstance(kp_labels, list) and kp_labels:
            labels = [str(v) for v in kp_labels]
        if not skeleton and isinstance(kp_skeleton, list) and kp_skeleton:
            skeleton = kp_skeleton
        if labels and skeleton:
            break

    kpt_dims = list(kpt_shape) if isinstance(kpt_shape, tuple) else None
    n_keypoints = int(kpt_dims[0]) if kpt_dims and len(kpt_dims) >= 1 else None

    return {
        "kpt_shape": kpt_dims,
        "keypoint_labels": labels,
        "skeleton": skeleton,
    }


def _safe_sha256_file(path: Optional[Path]) -> Optional[str]:
    return _shared_safe_sha256_file(path)


def _normalize_imgsz(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        size = int(text)
        return size, size
    return 640, 640


def _resolve_export_device(device_value: Any) -> str:
    text = str(device_value or "").strip().lower()
    if not text:
        return "cpu"
    if text.startswith("cuda") or text == "cpu":
        return text
    if text.isdigit():
        return f"cuda:{text}"
    if "," in text:
        first = text.split(",", 1)[0].strip()
        if first.isdigit():
            return f"cuda:{first}"
    return "cpu"


def _run_subprocess(
    command: list[str],
    console: Console,
    label: str,
    log_path: Optional[Path] = None,
) -> bool:
    return _shared_run_subprocess_streaming(
        command=command,
        console=console,
        label=label,
        log_path=log_path,
    )


def _resolve_trtexec_path(explicit_path: Optional[str]) -> Optional[Path]:
    return _shared_resolve_trtexec_path(explicit_path)


def _collect_export_env(trtexec_path: Optional[Path], trtexec_log_path: Optional[Path] = None) -> dict:
    return _shared_collect_export_env(trtexec_path, trtexec_log_path=trtexec_log_path)


def _coerce_export_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    return Path(text) if text else None


def _export_pose_onnx_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    model: YOLO,
    weights_path: Path,
    training_params: Dict[str, Any],
    args,
    manifest_path: Optional[Path],
    manifest_hints: Dict[str, Optional[str]],
    console: Console,
    checkpoint_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    export_onnx = bool(args.export_onnx or args.export_trt)
    export_info: Dict[str, Any] = {"enabled": export_onnx, "errors": []}
    if not export_onnx:
        return export_info

    exports_root = run_dir / "exports"
    onnx_dir = exports_root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    canonical_onnx_path = onnx_dir / f"{run_id}.onnx"
    existing_onnx_path: Optional[Path] = None
    if getattr(args, "onnx_path", None):
        existing_onnx_path = Path(args.onnx_path).expanduser().resolve()
        if not existing_onnx_path.exists():
            export_info["errors"].append(f"onnx_not_found:{existing_onnx_path}")
            return export_info

    img_h, img_w = _normalize_imgsz(training_params.get("imgsz"))
    export_device = _resolve_export_device(training_params.get("device"))
    onnx_path = existing_onnx_path or canonical_onnx_path
    onnx_manifest_path = onnx_dir / f"{run_id}.onnx.manifest.json"
    export_info["onnx_path"] = str(onnx_path)
    export_info["onnx_manifest_path"] = str(onnx_manifest_path)

    if existing_onnx_path is None:
        try:
            console.print("[bold cyan]Exporting ONNX...[/bold cyan]")
            exported = model.export(
                format="onnx",
                imgsz=[img_h, img_w],
                opset=int(args.onnx_opset),
                simplify=bool(args.onnx_simplify),
                device=export_device,
            )
            exported_path = _coerce_export_path(exported)
            if exported_path is None or not exported_path.exists():
                export_info["errors"].append("onnx_export_failed")
                return export_info
            exported_resolved = exported_path.expanduser().resolve()
            canonical_resolved = canonical_onnx_path.expanduser().resolve()
            if exported_resolved != canonical_resolved:
                shutil.copy2(exported_resolved, canonical_resolved)
            onnx_path = canonical_resolved
            export_info["onnx_path"] = str(onnx_path)
        except Exception as exc:
            export_info["errors"].append(f"onnx_export_exception:{type(exc).__name__}:{exc}")
            return export_info
    else:
        console.print(f"[cyan]Using existing ONNX:[/cyan] {existing_onnx_path}")

    weights_sha = _safe_sha256_file(weights_path)
    onnx_sha = _safe_sha256_file(onnx_path)
    export_info["weights_sha256"] = weights_sha
    export_info["onnx_sha256"] = onnx_sha
    export_info["onnx_source"] = "existing" if existing_onnx_path else "exported"
    export_info["onnx_opset"] = int(args.onnx_opset)
    export_info["input_shape"] = [1, 3, int(img_h), int(img_w)]
    export_info["imgsz"] = [int(img_h), int(img_w)]
    export_info["build_env"] = {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if getattr(torch.version, "cuda", None) else None,
        "system_hostname": str(platform.node()) if platform.node() else None,
    }

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "task": "pose",
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
            "input_shape": [1, 3, int(img_h), int(img_w)],
            "imgsz": [img_h, img_w],
            "opset": int(args.onnx_opset),
            "simplify": bool(args.onnx_simplify),
            "device": export_device,
        },
        "build_env": export_info.get("build_env"),
        "source_manifest": {
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": _safe_sha256_file(manifest_path),
            "set_id": manifest_hints.get("set_id"),
        },
    }
    onnx_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    export_info["onnx_manifest_sha256"] = _safe_sha256_file(onnx_manifest_path)
    if checkpoint_cb is not None and export_info.get("onnx_path"):
        checkpoint_cb("onnx", dict(export_info))
    if not args.export_trt:
        return export_info

    trt_dir = exports_root / "tensorrt"
    trt_dir.mkdir(parents=True, exist_ok=True)
    engine_name = f"{run_id}_{args.trt_precision}"
    engine_path = trt_dir / f"{engine_name}.engine"
    engine_manifest_path = trt_dir / f"{engine_name}.tensorrt.manifest.json"
    trt_log_path = trt_dir / f"{engine_name}_trtexec.log"
    trtexec_path = _resolve_trtexec_path(getattr(args, "trtexec", None))
    trt_script = Path(__file__).resolve().parent / "onnx_to_tensorrt.py"
    trt_cmd = [
        str(Path(sys.executable)),
        "-u",
        str(trt_script),
        "--onnx",
        str(onnx_path),
        "--engine",
        str(engine_path),
        "--precision",
        str(args.trt_precision),
    ]
    if trtexec_path is not None:
        trt_cmd.extend(["--trtexec", str(trtexec_path)])
    if args.trt_cuda_graph:
        trt_cmd.append("--cuda-graph")
    if args.trt_profiling:
        trt_cmd.append("--profiling")
    if args.trt_verbose:
        trt_cmd.append("--verbose")
    export_info["trt_command"] = trt_cmd
    export_info["trt_log_path"] = str(trt_log_path)

    if not trt_script.exists():
        export_info["errors"].append(f"tensorrt_script_missing:{trt_script}")
        return export_info

    console.print("[bold cyan]Building TensorRT engine...[/bold cyan]")
    ok = _run_subprocess(trt_cmd, console, "TensorRT export", log_path=trt_log_path)
    if not ok:
        export_info["errors"].append("tensorrt_export_failed")
        return export_info

    if not engine_path.exists():
        export_info["errors"].append("tensorrt_engine_missing")
        return export_info

    build_env = _collect_export_env(trtexec_path, trtexec_log_path=trt_log_path)
    trt_device_info = build_env.get("trtexec_runtime") if isinstance(build_env, dict) else None
    engine_sha = _safe_sha256_file(engine_path)
    engine_manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "task": "pose",
        "weights": {"path": str(weights_path), "sha256": weights_sha},
        "onnx": {"path": str(onnx_path), "sha256": onnx_sha},
        "engine": {"path": str(engine_path), "sha256": engine_sha},
        "onnx_manifest_path": str(onnx_manifest_path),
        "export": {
            "precision": str(args.trt_precision),
            "input_shape": [1, 3, int(img_h), int(img_w)],
            "imgsz": [int(img_h), int(img_w)],
            "opset": int(args.onnx_opset),
            "device": export_device,
        },
        "trt": {
            "precision": str(args.trt_precision),
            "cuda_graph": bool(args.trt_cuda_graph),
            "profiling": bool(args.trt_profiling),
            "verbose": bool(args.trt_verbose),
            "trtexec_path": str(trtexec_path) if trtexec_path else None,
            "command": trt_cmd,
        },
        "logs": {"tensorrt_export": str(trt_log_path)},
        "build_env": build_env,
        "source_manifest": {
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": _safe_sha256_file(manifest_path),
            "set_id": manifest_hints.get("set_id"),
        },
    }
    engine_manifest_path.write_text(json.dumps(engine_manifest, indent=2), encoding="utf-8")

    export_info.update(
        {
            "engine_path": str(engine_path),
            "engine_manifest_path": str(engine_manifest_path),
            "engine_precision": str(args.trt_precision),
            "engine_sha256": engine_sha,
            "engine_manifest_sha256": _safe_sha256_file(engine_manifest_path),
            "build_env": build_env,
            "trt_device_info": trt_device_info if isinstance(trt_device_info, dict) else None,
        }
    )
    if checkpoint_cb is not None and export_info.get("engine_path"):
        checkpoint_cb("tensorrt", dict(export_info))
    return export_info


def _load_manifest_set_id(manifest_path: Optional[str]) -> Optional[str]:
    return _load_manifest_run_hints(manifest_path).get("set_id")


def _load_manifest_run_hints(manifest_path: Optional[str]) -> Dict[str, Optional[str]]:
    hints: Dict[str, Optional[str]] = {
        "set_id": None,
        "set_slug": None,
        "manifest_sha256": None,
        "rig_name": None,
        "dish_design": None,
        "canvas_name": None,
        "task": None,
    }
    if not manifest_path:
        return hints
    manifest_raw = None
    try:
        manifest_raw = Path(manifest_path).read_text(encoding="utf-8")
        payload = json.loads(manifest_raw)
    except Exception:
        return hints

    if not isinstance(payload, dict):
        return hints

    if manifest_raw is not None:
        try:
            hints["manifest_sha256"] = hashlib.sha256(manifest_raw.encode("utf-8")).hexdigest()
        except Exception:
            pass

    set_id = payload.get("set_id")
    if isinstance(set_id, str) and set_id.strip():
        cleaned_set_id = set_id.strip()
        hints["set_id"] = cleaned_set_id
        hints["set_slug"] = _strip_manifest_suffixes(cleaned_set_id)

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
    hints["rig_name"] = (
        query_filter.get("rig_id")
        or first_dataset.get("rig_id")
        or rig_info.get("rig_id")
    )
    task = payload.get("task")
    if isinstance(task, str) and task.strip():
        hints["task"] = task.strip()
    return hints


def _strip_manifest_suffixes(value: str) -> str:
    return _shared_strip_manifest_suffixes(value)


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
    return _shared_sanitize_run_component(value, fallback)


def _build_default_run_name(
    *,
    manifest_hints: Dict[str, Optional[str]],
    task_fallback: str,
    timestamp: Optional[str] = None,
    pid: Optional[int] = None,
) -> str:
    return _shared_build_default_pose_run_name(
        manifest_hints=manifest_hints,
        task_fallback=task_fallback,
        timestamp=timestamp,
        pid=pid,
    )


def _infer_set_slug(set_id: Optional[str], config_path: Optional[Path]) -> str:
    return _shared_infer_set_slug(set_id, config_path, "pose_training")


def _resolve_project_dir(
    *,
    args,
    training_params: dict,
    set_id: Optional[str],
    config_path: Optional[Path],
    console: Console,
) -> None:
    _shared_resolve_project_dir(
        args=args,
        training_params=training_params,
        set_id=set_id,
        config_path=config_path,
        task_subdir="pose",
        default_slug="pose_training",
        console=console,
    )


def _apply_pose_loader_training_param_overrides(training_params: dict) -> tuple[dict, dict]:
    """Normalize train() params and return custom loader settings."""
    params = dict(training_params)
    loader_cfg: dict[str, Any] = {}
    loader_cfg["num_workers"] = max(0, int(params.pop("num_workers", 8) or 0))
    loader_cfg["persistent_workers"] = bool(params.pop("persistent_workers", False))
    loader_cfg["prefetch_factor"] = params.pop("prefetch_factor", None)
    loader_cfg["deterministic_val"] = bool(params.pop("deterministic_val", True))
    val_workers_raw = params.pop("val_num_workers", None)
    loader_cfg["val_num_workers"] = (
        None if val_workers_raw is None else max(0, int(val_workers_raw or 0))
    )
    # Prevent Ultralytics defaults from fighting custom loader behavior.
    params.pop("workers", None)
    params.pop("deterministic", None)
    return params, loader_cfg


def _snapshot_training_inputs(
    *,
    run_dir: Path,
    config_path: Optional[Path],
    manifest_path: Optional[Path],
    invocation_payload: Optional[Dict[str, Any]],
) -> list[Path]:
    return _shared_snapshot_training_inputs(
        run_dir=run_dir,
        config_path=config_path,
        manifest_path=manifest_path,
        invocation_payload=invocation_payload,
    )


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
    pose_schema: Optional[Dict[str, Any]] = None,
    export_artifacts: Optional[Dict[str, Any]] = None,
) -> None:
    _shared_record_registry_training_run(
        args=args,
        console=console,
        invocation_payload=invocation_payload,
        run_id=run_id,
        set_id=set_id,
        config_path=config_path,
        manifest_path=manifest_path,
        model_path=model_path,
        metrics_path=metrics_path,
        status=status,
        final_metrics=final_metrics,
        pose_schema=pose_schema,
        export_artifacts=export_artifacts,
    )


def main(args) -> int:
    """Main training function."""
    console = Console()
    print_training_banner(console, "Pose")

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
            tool="fisheye.training.train_pose",
            args=args,
        )
        if args.log_registry
        else None
    )
    pose_schema: Optional[Dict[str, Any]] = None

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
        # Seed schema from config so registry in_progress rows can carry skeleton_id
        # before dataset metadata is loaded.
        pose_schema = _infer_pose_schema(tuple(full_config.kpt_shape), {})
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
                pose_schema=pose_schema,
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
            pose_schema=pose_schema,
        )

    # Get zarr metadata
    zarr_paths = config.get_zarr_paths()
    zarr_metadata = get_zarr_metadata(zarr_paths, console)
    
    # Display dataset info
    print_section_header(console, "Dataset Information")
    
    pose_schema = _infer_pose_schema(tuple(full_config.kpt_shape), zarr_metadata)
    print_dataset_details(console, zarr_metadata, task="pose", pose_schema=pose_schema)
    
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
    training_params, loader_cfg = _apply_pose_loader_training_param_overrides(training_params)
    model_name = training_params.get('model', 'yolov8n-pose.pt')
    
    # Display hyperparameters
    print_training_hyperparameters(
        console,
        training_params=training_params,
        loader_overrides=loader_cfg,
        include_loader_note=True,
    )
    
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
                pose_schema=pose_schema,
            )
        raise
    
    # Monkey-patch the trainer's get_dataloader method
    def get_zarr_dataloader(trainer_self, dataset_path, batch_size, mode, rank=0):
        dataset = create_zarr_dataset(config=config, mode=mode)
        workers = int(loader_cfg.get("num_workers", 8) or 0)
        if mode != "train" and bool(loader_cfg.get("deterministic_val", True)):
            workers = 0
        elif mode != "train" and loader_cfg.get("val_num_workers") is not None:
            workers = int(loader_cfg["val_num_workers"] or 0)
        persistent_workers = bool(loader_cfg.get("persistent_workers", False) and mode == "train" and workers > 0)
        prefetch_factor = loader_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            try:
                prefetch_factor = max(1, int(prefetch_factor))
            except Exception:
                prefetch_factor = None
        return YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            collate_fn=pose_collate_fn,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=(prefetch_factor if workers > 0 else None),
        )
    
    ZarrPoseTrainer.get_dataloader = get_zarr_dataloader
    
    # Start training
    print_training_start(console, lightning=True)
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
    except KeyboardInterrupt as e:
        console.print("\n[bold yellow]Training interrupted by user (Ctrl-C).[/bold yellow]")
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
                    "error_message": "training_interrupted_by_user",
                },
                pose_schema=pose_schema,
            )
        return 130
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
                pose_schema=pose_schema,
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

    export_artifacts: Optional[Dict[str, Any]] = None
    model_path = results.save_dir / "weights" / "best.pt"
    if args.log_registry:
        metrics_path = results.save_dir / "results.csv"
        trained_metrics_payload = dict(final_validation_metrics or {})
        trained_metrics_payload["stage"] = "trained"
        trained_metrics_payload["status_detail"] = "model_checkpoint_ready"
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
            status="in_progress",
            final_metrics=trained_metrics_payload,
            pose_schema=pose_schema,
        )

    def _export_checkpoint(stage_name: str, artifacts: Dict[str, Any]) -> None:
        if not args.log_registry:
            return
        metrics_path = results.save_dir / "results.csv"
        checkpoint_metrics_payload = dict(final_validation_metrics or {})
        checkpoint_metrics_payload["stage"] = stage_name
        checkpoint_metrics_payload["status_detail"] = f"{stage_name}_complete"
        if artifacts.get("errors"):
            checkpoint_metrics_payload["export_errors"] = artifacts.get("errors")
        if artifacts.get("onnx_path"):
            checkpoint_metrics_payload["onnx_path"] = artifacts.get("onnx_path")
        if artifacts.get("engine_path"):
            checkpoint_metrics_payload["engine_path"] = artifacts.get("engine_path")
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
            status="in_progress",
            final_metrics=checkpoint_metrics_payload,
            pose_schema=pose_schema,
            export_artifacts=artifacts,
        )

    if (args.export_onnx or args.export_trt) and model_path.exists():
        export_artifacts = _export_pose_onnx_artifacts(
            run_dir=results.save_dir,
            run_id=registry_run_id,
            model=model,
            weights_path=model_path,
            training_params=training_params,
            args=args,
            manifest_path=manifest_path,
            manifest_hints=manifest_hints,
            console=console,
            checkpoint_cb=_export_checkpoint,
        )
        export_errors = export_artifacts.get("errors") if isinstance(export_artifacts, dict) else None
        if export_errors:
            console.print(f"[yellow]ONNX export completed with issues:[/yellow] {export_errors}")
        elif export_artifacts and export_artifacts.get("onnx_path"):
            console.print(f"[green]✓ ONNX exported:[/green] {export_artifacts.get('onnx_path')}")

    if args.log_registry:
        metrics_path = results.save_dir / "results.csv"
        final_metrics_payload = dict(final_validation_metrics or {})
        final_metrics_payload.setdefault("stage", "completed")
        final_metrics_payload.setdefault("status_detail", "training_complete")
        if export_artifacts:
            final_metrics_payload["export_onnx"] = bool(export_artifacts.get("onnx_path"))
            final_metrics_payload["export_trt"] = bool(export_artifacts.get("engine_path"))
            if export_artifacts.get("errors"):
                final_metrics_payload["export_errors"] = export_artifacts.get("errors")
            if export_artifacts.get("onnx_path"):
                final_metrics_payload["onnx_path"] = export_artifacts.get("onnx_path")
            if export_artifacts.get("engine_path"):
                final_metrics_payload["engine_path"] = export_artifacts.get("engine_path")
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
            pose_schema=pose_schema,
            export_artifacts=export_artifacts,
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
  python -m fisheye.training.train_pose configs/pose_config.yaml
  
  # With custom run name
  python -m fisheye.training.train_pose configs/pose_config.yaml --run-name fish_pose_v1
  
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
        help="Export TensorRT engine after training (implies ONNX export).",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        default="fp16",
        help="TensorRT precision for engine export.",
    )
    parser.add_argument("--trtexec", type=str, help="Optional path to trtexec binary.")
    parser.add_argument("--trt-cuda-graph", action="store_true", help="Enable TensorRT CUDA graph.")
    parser.add_argument("--trt-profiling", action="store_true", help="Enable TensorRT profiling outputs.")
    parser.add_argument("--trt-verbose", action="store_true", help="Enable verbose TensorRT build logs.")
    args = parser.parse_args()
    raise SystemExit(main(args))
