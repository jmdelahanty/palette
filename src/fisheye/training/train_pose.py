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

from ..shared.zarr_run_completion import resolve_authoritative_run_name
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
from .zarr_yolo_dataset_loader import (
    build_pose_zarr_dataset_config,
    create_zarr_dataset,
)
from ..shared.system_metadata import build_invocation_record, get_git_info
from ..shared.zarr.manifest_digest import canonical_json_sha256


POSE_EFFECTIVE_ARGUMENT_KEYS = (
    "imgsz",
    "pose",
    "kobj",
    "box",
    "cls",
    "dfl",
    "lr0",
    "momentum",
    "weight_decay",
    "rect",
    "augment",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "erasing",
    "mosaic",
    "mixup",
    "copy_paste",
    "cutmix",
    "auto_augment",
    "multi_scale",
    "workers",
    "seed",
)


def _jsonable_runtime_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_runtime_value(item) for key, item in value.items()
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _runtime_arg_value(args: Any, key: str) -> Any:
    if isinstance(args, Mapping):
        return args.get(key)
    return getattr(args, key, None)


def _runtime_values_equal(key: str, requested: Any, effective: Any) -> bool:
    if key == "imgsz":
        return _normalize_imgsz(requested) == _normalize_imgsz(effective)
    if isinstance(requested, (int, float)) and not isinstance(requested, bool):
        try:
            return bool(np.isclose(float(requested), float(effective), rtol=0.0, atol=1e-12))
        except Exception:
            return False
    return _jsonable_runtime_value(requested) == _jsonable_runtime_value(effective)


def _validate_pose_effective_arguments(
    requested: Mapping[str, Any],
    effective_args: Any,
) -> dict[str, Any]:
    effective = {
        key: _jsonable_runtime_value(_runtime_arg_value(effective_args, key))
        for key in POSE_EFFECTIVE_ARGUMENT_KEYS
    }
    normalized_requested = {
        key: _jsonable_runtime_value(requested.get(key))
        for key in POSE_EFFECTIVE_ARGUMENT_KEYS
    }
    mismatches = {
        key: {
            "requested": normalized_requested[key],
            "effective": effective[key],
        }
        for key in POSE_EFFECTIVE_ARGUMENT_KEYS
        if not _runtime_values_equal(
            key,
            requested.get(key),
            _runtime_arg_value(effective_args, key),
        )
    }
    if mismatches:
        raise ValueError(
            "Effective Ultralytics pose arguments disagree with the requested "
            f"contract: {json.dumps(mismatches, sort_keys=True)}"
        )
    return {
        "requested": normalized_requested,
        "effective": effective,
        "status": "exact_match",
    }


def _write_pose_runtime_receipt(state: dict[str, Any]) -> Optional[Path]:
    run_dir_raw = state.get("run_dir")
    if run_dir_raw is None:
        return None
    run_dir = Path(run_dir_raw)
    payload = {
        key: _jsonable_runtime_value(value)
        for key, value in state.items()
        if key not in {"run_dir", "receipt_path", "receipt_sha256"}
    }
    document = {
        "schema_id": "palette.pose_training_runtime_receipt.v1",
        "payload": payload,
        "payload_sha256": canonical_json_sha256(payload),
    }
    path = run_dir / "pose_training_runtime_receipt.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state["receipt_path"] = str(path)
    state["receipt_sha256"] = _shared_safe_sha256_file(path)
    return path


def _observe_pose_runtime_batch(
    state: dict[str, Any],
    *,
    raw_batch: torch.Tensor,
    normalized_batch: torch.Tensor,
) -> None:
    expected_hw = tuple(int(value) for value in state["model_input_shape_hw"])
    if raw_batch.ndim != 4 or tuple(raw_batch.shape[-2:]) != expected_hw:
        raise ValueError(
            "Pose loader emitted a batch that disagrees with the model-input contract: "
            f"expected NCHW with HW={expected_hw}, got {tuple(raw_batch.shape)}."
        )
    if int(raw_batch.shape[1]) != 3 or raw_batch.dtype != torch.uint8:
        raise ValueError(
            "Pose loader must emit three-channel uint8 tensors before normalization; "
            f"got shape={tuple(raw_batch.shape)} dtype={raw_batch.dtype}."
        )
    if normalized_batch.dtype != torch.float32:
        raise ValueError(
            "Pose trainer must normalize input to float32; "
            f"got {normalized_batch.dtype}."
        )
    min_value = float(normalized_batch.min().detach().cpu())
    max_value = float(normalized_batch.max().detach().cpu())
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(
            "Pose normalized tensor values must remain in [0, 1]; "
            f"observed [{min_value}, {max_value}]."
        )
    if "first_batch" not in state:
        state["first_batch"] = {
            "raw_shape_nchw": [int(value) for value in raw_batch.shape],
            "raw_dtype": str(raw_batch.dtype).removeprefix("torch."),
            "normalized_shape_nchw": [int(value) for value in normalized_batch.shape],
            "normalized_dtype": str(normalized_batch.dtype).removeprefix("torch."),
            "normalized_min": min_value,
            "normalized_max": max_value,
            "status": "verified",
        }
        state["status"] = "runtime_batch_verified"
        _write_pose_runtime_receipt(state)


def _pose_starting_model_receipt(model: YOLO, requested: str) -> dict[str, Any]:
    raw_path = getattr(model, "ckpt_path", None) or requested
    path = Path(str(raw_path)).expanduser()
    digest = _shared_safe_sha256_file(path)
    model_object = getattr(model, "model", None)
    yaml_payload = getattr(model_object, "yaml", None)
    parameter_count = None
    if model_object is not None:
        try:
            parameter_count = int(sum(parameter.numel() for parameter in model_object.parameters()))
        except Exception:
            parameter_count = None
    return {
        "requested": str(requested),
        "resolved_path": str(path.resolve()) if path.exists() else str(path),
        "sha256": digest,
        "status": "content_verified" if digest is not None else "unresolved_reference",
        "model_class": (
            f"{type(model_object).__module__}.{type(model_object).__name__}"
            if model_object is not None
            else None
        ),
        "parameter_count": parameter_count,
        "architecture": _jsonable_runtime_value(yaml_payload),
    }


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

    runtime_contract_state: Optional[dict[str, Any]] = None

    def preprocess_batch(self, batch: dict) -> dict:
        raw_images = batch.get("img")
        if not isinstance(raw_images, torch.Tensor):
            raise ValueError("Pose training batch is missing its image tensor.")
        processed = super().preprocess_batch(batch)
        state = self.runtime_contract_state
        if state is not None:
            _observe_pose_runtime_batch(
                state,
                raw_batch=raw_images,
                normalized_batch=processed["img"],
            )
        return processed
    
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


def get_zarr_metadata(
    zarr_paths,
    console=None,
    *,
    keypoint_run_resolver: Optional[Callable[[str], Optional[str]]] = None,
):
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
        keypoint_count = int(kp_group["keypoints_roi"].shape[1])
        candidates: list[list[str]] = []
        raw_candidates: list[Any] = [kp_group.attrs.get("keypoint_labels")]
        pose_schema = kp_group.attrs.get("pose_schema")
        if isinstance(pose_schema, Mapping):
            raw_candidates.append(pose_schema.get("keypoint_labels"))
            nodes = pose_schema.get("nodes")
            if isinstance(nodes, (list, tuple)):
                raw_candidates.append(
                    [
                        node.get("name") if isinstance(node, Mapping) else node
                        for node in nodes
                    ]
                )
        for raw in raw_candidates:
            if raw is None:
                continue
            if not isinstance(raw, (list, tuple)) or len(raw) != keypoint_count:
                raise ValueError(
                    "Populated live keypoint labels do not match runtime cardinality."
                )
            labels = []
            for item in raw:
                if type(item) is not str or not item or item.strip() != item:
                    raise ValueError(
                        "Populated live keypoint labels contain a noncanonical label."
                    )
                labels.append(item)
            if len(set(labels)) != len(labels):
                raise ValueError("Populated live keypoint labels are not unique.")
            candidates.append(labels)
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError(
                "Populated live ordered keypoint labels disagree within the selected run."
            )
        return candidates[0] if candidates else None

    def _resolve_tracking_skeleton(kp_group: zarr.Group) -> Optional[list[list[int]]]:
        keypoint_count = (
            int(kp_group["keypoints_roi"].shape[1])
            if "keypoints_roi" in kp_group
            else None
        )
        raw_candidates: list[Any] = [kp_group.attrs.get("keypoint_skeleton")]
        pose_schema = kp_group.attrs.get("pose_schema")
        if isinstance(pose_schema, Mapping):
            raw_candidates.extend(
                [pose_schema.get("edges"), pose_schema.get("skeleton")]
            )
        candidates: list[list[list[int]]] = []
        for raw_skeleton in raw_candidates:
            if raw_skeleton is None:
                continue
            if not isinstance(raw_skeleton, (list, tuple)):
                raise ValueError("Populated live keypoint skeleton is not an edge list.")
            edges: list[list[int]] = []
            for edge in raw_skeleton:
                if (
                    not isinstance(edge, (list, tuple))
                    or len(edge) != 2
                    or any(type(item) is not int for item in edge)
                ):
                    raise ValueError("Populated live keypoint skeleton contains an invalid edge.")
                src, dst = int(edge[0]), int(edge[1])
                if (
                    src == dst
                    or src < 0
                    or dst < 0
                    or (
                        keypoint_count is not None
                        and (src >= keypoint_count or dst >= keypoint_count)
                    )
                ):
                    raise ValueError("Populated live keypoint skeleton contains an invalid edge.")
                edges.append([src, dst])
            candidates.append(edges)
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError(
                "Populated live keypoint skeletons disagree within the selected run."
            )
        return candidates[0] if candidates else None

    def _resolve_tracking_model_kpt_shape(kp_group: zarr.Group) -> Optional[list[int]]:
        candidates: list[list[int]] = []
        raw_candidates: list[Any] = [kp_group.attrs.get("model_kpt_shape")]
        pose_schema = kp_group.attrs.get("pose_schema")
        if isinstance(pose_schema, Mapping):
            metadata = pose_schema.get("metadata")
            if isinstance(metadata, Mapping):
                raw_candidates.append(metadata.get("model_kpt_shape"))
        for raw in raw_candidates:
            if raw is None:
                continue
            candidates.append(
                _strict_pose_shape(raw, field="live model_kpt_shape")
            )
        if any(candidate != candidates[0] for candidate in candidates[1:]):
            raise ValueError(
                "Populated live model_kpt_shape values disagree within the selected run."
            )
        return candidates[0] if candidates else None

    def _resolve_tracking_skeleton_id(kp_group: zarr.Group) -> Optional[str]:
        values: list[str] = []
        top_level = kp_group.attrs.get("skeleton_id")
        if isinstance(top_level, str) and top_level.strip():
            values.append(top_level.strip())
        pose_schema = kp_group.attrs.get("pose_schema")
        if isinstance(pose_schema, Mapping):
            nested = pose_schema.get("skeleton_id")
            if isinstance(nested, str) and nested.strip():
                values.append(nested.strip())
        if len(set(values)) > 1:
            raise ValueError(
                "Populated live keypoint skeleton identities disagree within the selected run."
            )
        return values[0] if values else None

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
            if 'crop_runs' in root:
                latest_crop = resolve_authoritative_run_name(root['crop_runs'])
                if latest_crop:
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
            if 'keypoints_runs' in root:
                requested_keypoint_run = (
                    keypoint_run_resolver(str(path))
                    if keypoint_run_resolver is not None
                    else None
                )
                requested_text = (
                    str(requested_keypoint_run).strip()
                    if requested_keypoint_run is not None
                    else ""
                )
                if requested_text.lower() in {
                    "latest_traditional",
                    "latest:traditional",
                    "traditional",
                    "latest_yolo",
                    "latest:yolo",
                    "yolo",
                }:
                    raise ValueError(
                        "Hash-bound pose-schema preflight requires the exact selected "
                        "keypoint run name, not a method-relative selector."
                    )
                if requested_text and requested_text.lower() != "latest":
                    if requested_text not in root['keypoints_runs']:
                        raise ValueError(
                            f"Selected keypoint run {requested_text!r} is absent."
                        )
                    latest_kp = requested_text
                else:
                    latest_kp = resolve_authoritative_run_name(root['keypoints_runs'])
                if latest_kp:
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
                    keypoint_count = (
                        int(kp_group["keypoints_roi"].shape[1])
                        if "keypoints_roi" in kp_group
                        else None
                    )
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
                        'keypoint_count': keypoint_count,
                        'model_kpt_shape': _resolve_tracking_model_kpt_shape(kp_group),
                        'skeleton_id': _resolve_tracking_skeleton_id(kp_group),
                    }
            else:
                zarr_meta['tracking_info'] = {'warning': 'keypoints_runs not found; proceeding without precomputed keypoints metadata'}
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
    
    return metadata


def _strict_pose_shape(value: Any, *, field: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"{field} must be an exact positive [keypoints, dimensions] pair.")
    return [int(value[0]), int(value[1])]


def _strict_pose_labels(value: Any, *, keypoint_count: int, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) != keypoint_count:
        raise ValueError(f"{field} must contain exactly {keypoint_count} ordered labels.")
    labels: list[str] = []
    for item in value:
        if type(item) is not str or not item or item.strip() != item:
            raise ValueError(f"{field} must contain canonical nonempty strings.")
        labels.append(item)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field} must contain unique ordered labels.")
    return labels


def _strict_pose_skeleton(value: Any, *, keypoint_count: int, field: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an exact edge list.")
    edges: list[list[int]] = []
    for edge in value:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(item) is not int for item in edge)
            or edge[0] == edge[1]
            or any(item < 0 or item >= keypoint_count for item in edge)
        ):
            raise ValueError(f"{field} contains an invalid keypoint edge.")
        edges.append([int(edge[0]), int(edge[1])])
    return edges


def _normalize_manifest_pose_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Hash-verified pose training manifest lacks pose_schema.")
    shape = _strict_pose_shape(
        value.get("kpt_shape"),
        field="manifest pose_schema.kpt_shape",
    )
    keypoint_count = shape[0]
    labels = _strict_pose_labels(
        value.get("keypoint_labels"),
        keypoint_count=keypoint_count,
        field="manifest pose_schema.keypoint_labels",
    )
    skeleton = _strict_pose_skeleton(
        value.get("skeleton"),
        keypoint_count=keypoint_count,
        field="manifest pose_schema.skeleton",
    )
    skeleton_id = value.get("skeleton_id")
    if type(skeleton_id) is not str or not skeleton_id or skeleton_id.strip() != skeleton_id:
        raise ValueError(
            "manifest pose_schema.skeleton_id must be one nonempty canonical string."
        )
    return {
        "skeleton_id": skeleton_id,
        "kpt_shape": shape,
        "keypoint_labels": labels,
        "skeleton": skeleton,
    }


def _load_pose_manifest_authority(manifest_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if manifest_path is None:
        return None
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read exact pose training manifest {manifest_path}: {exc}.") from exc
    if not isinstance(payload, Mapping) or payload.get("task") != "pose":
        raise ValueError("Hash-verified training manifest is not a pose manifest.")
    set_id = payload.get("set_id")
    if set_id is not None and (
        type(set_id) is not str or not set_id or set_id.strip() != set_id
    ):
        raise ValueError("Pose training manifest set_id must be null or a canonical string.")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "set_id": set_id,
        "pose_schema": _normalize_manifest_pose_schema(payload.get("pose_schema")),
    }


def _infer_pose_schema(
    kpt_shape: Optional[tuple],
    zarr_metadata: dict,
    *,
    manifest_pose_schema: Optional[Mapping[str, Any]] = None,
) -> dict:
    if manifest_pose_schema is not None:
        authority = _normalize_manifest_pose_schema(manifest_pose_schema)
        config_shape = _strict_pose_shape(kpt_shape, field="configured kpt_shape")
        if config_shape != authority["kpt_shape"]:
            raise ValueError(
                "Configured kpt_shape disagrees with the hash-verified training manifest."
            )

        for source_name, meta in zarr_metadata.items():
            if not isinstance(meta, dict):
                raise ValueError(f"{source_name}: live source metadata is not a mapping.")
            if "error" in meta:
                raise ValueError(
                    f"{source_name}: live source pose-schema inspection failed: {meta['error']}"
                )
            tracking_info = (
                meta.get("tracking_info")
                if isinstance(meta.get("tracking_info"), dict)
                else {}
            )
            source_labels = tracking_info.get("keypoint_labels")
            if source_labels:
                normalized_labels = _strict_pose_labels(
                    source_labels,
                    keypoint_count=authority["kpt_shape"][0],
                    field=f"{source_name} live keypoint labels",
                )
                if normalized_labels != authority["keypoint_labels"]:
                    raise ValueError(
                        f"{source_name}: populated live ordered keypoint labels disagree "
                        "with the hash-verified training manifest."
                    )
            source_skeleton = tracking_info.get("keypoint_skeleton")
            if source_skeleton is not None:
                normalized_skeleton = _strict_pose_skeleton(
                    source_skeleton,
                    keypoint_count=authority["kpt_shape"][0],
                    field=f"{source_name} live keypoint skeleton",
                )
                if normalized_skeleton != authority["skeleton"]:
                    raise ValueError(
                        f"{source_name}: populated live keypoint skeleton disagrees "
                        "with the hash-verified training manifest."
                    )
            source_skeleton_id = tracking_info.get("skeleton_id")
            if source_skeleton_id is not None:
                if (
                    type(source_skeleton_id) is not str
                    or not source_skeleton_id
                    or source_skeleton_id.strip() != source_skeleton_id
                ):
                    raise ValueError(
                        f"{source_name}: populated live skeleton_id is not canonical."
                    )
                if source_skeleton_id != authority["skeleton_id"]:
                    raise ValueError(
                        f"{source_name}: populated live skeleton_id disagrees with "
                        "the hash-verified training manifest."
                    )
            source_count = tracking_info.get("keypoint_count")
            if source_count is not None and (
                type(source_count) is not int
                or source_count != authority["kpt_shape"][0]
            ):
                raise ValueError(
                    f"{source_name}: live keypoint cardinality disagrees with the "
                    "hash-verified training manifest."
                )
            source_model_shape = tracking_info.get("model_kpt_shape")
            if source_model_shape is not None:
                normalized_model_shape = _strict_pose_shape(
                    source_model_shape,
                    field=f"{source_name} live model_kpt_shape",
                )
                if normalized_model_shape != authority["kpt_shape"]:
                    raise ValueError(
                        f"{source_name}: populated live model_kpt_shape disagrees "
                        "with the hash-verified training manifest."
                    )
        return authority

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


def _positive_int_attr(args: Any, name: str, default: Optional[int] = None) -> Optional[int]:
    value = getattr(args, name, default)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


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
    onnx_dynamic = bool(getattr(args, "onnx_dynamic", False))
    onnx_batch = _positive_int_attr(args, "onnx_batch", 1) or 1
    trt_min_batch = _positive_int_attr(args, "trt_min_batch", None)
    trt_opt_batch = _positive_int_attr(args, "trt_opt_batch", None)
    trt_max_batch = _positive_int_attr(args, "trt_max_batch", None)
    if onnx_dynamic:
        trt_min_batch = trt_min_batch or 1
        trt_opt_batch = trt_opt_batch or onnx_batch
        trt_max_batch = trt_max_batch or max(trt_opt_batch, onnx_batch)
    effective_max_batch = trt_max_batch if onnx_dynamic else onnx_batch
    input_shape: list[Any] = (
        ["dynamic", 3, int(img_h), int(img_w)]
        if onnx_dynamic
        else [int(onnx_batch), 3, int(img_h), int(img_w)]
    )
    onnx_path = existing_onnx_path or canonical_onnx_path
    onnx_manifest_path = onnx_dir / f"{run_id}.onnx.manifest.json"
    export_info["onnx_path"] = str(onnx_path)
    export_info["onnx_manifest_path"] = str(onnx_manifest_path)

    if existing_onnx_path is None:
        try:
            console.print("[bold cyan]Exporting ONNX...[/bold cyan]")
            export_kwargs = {
                "format": "onnx",
                "imgsz": [img_h, img_w],
                "opset": int(args.onnx_opset),
                "simplify": bool(args.onnx_simplify),
                "device": export_device,
            }
            if onnx_dynamic:
                export_kwargs["dynamic"] = True
            if onnx_dynamic or onnx_batch != 1:
                export_kwargs["batch"] = int(onnx_batch)
            exported = model.export(**export_kwargs)
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
    export_info["input_shape"] = list(input_shape)
    export_info["imgsz"] = [int(img_h), int(img_w)]
    export_info["dynamic_shapes"] = bool(onnx_dynamic)
    export_info["max_batch"] = int(effective_max_batch) if effective_max_batch else None
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
            "input_shape": list(input_shape),
            "imgsz": [img_h, img_w],
            "opset": int(args.onnx_opset),
            "simplify": bool(args.onnx_simplify),
            "device": export_device,
            "dynamic": bool(onnx_dynamic),
            "batch": int(onnx_batch),
            "max_batch": int(effective_max_batch) if effective_max_batch else None,
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
    builder_optimization_level = getattr(args, "trt_builder_optimization_level", None)
    if builder_optimization_level is not None:
        trt_cmd.extend(["--builder-optimization-level", str(int(builder_optimization_level))])
    trt_profile: dict[str, Any] = {}
    explicit_min_shapes = getattr(args, "trt_min_shapes", None)
    explicit_opt_shapes = getattr(args, "trt_opt_shapes", None)
    explicit_max_shapes = getattr(args, "trt_max_shapes", None)
    input_name = str(getattr(args, "trt_input_name", "images") or "images")
    if explicit_min_shapes:
        trt_cmd.extend(["--min-shapes", str(explicit_min_shapes)])
        trt_profile["min_shapes"] = str(explicit_min_shapes)
    if explicit_opt_shapes:
        trt_cmd.extend(["--opt-shapes", str(explicit_opt_shapes)])
        trt_profile["opt_shapes"] = str(explicit_opt_shapes)
    if explicit_max_shapes:
        trt_cmd.extend(["--max-shapes", str(explicit_max_shapes)])
        trt_profile["max_shapes"] = str(explicit_max_shapes)
    if onnx_dynamic and not (explicit_min_shapes or explicit_opt_shapes or explicit_max_shapes):
        min_shape = f"{input_name}:{int(trt_min_batch)}x3x{int(img_h)}x{int(img_w)}"
        opt_shape = f"{input_name}:{int(trt_opt_batch)}x3x{int(img_h)}x{int(img_w)}"
        max_shape = f"{input_name}:{int(trt_max_batch)}x3x{int(img_h)}x{int(img_w)}"
        trt_cmd.extend(
            [
                "--min-shapes",
                min_shape,
                "--opt-shapes",
                opt_shape,
                "--max-shapes",
                max_shape,
            ]
        )
        trt_profile.update(
            {
                "input_name": input_name,
                "min_batch": int(trt_min_batch),
                "opt_batch": int(trt_opt_batch),
                "max_batch": int(trt_max_batch),
                "min_shapes": min_shape,
                "opt_shapes": opt_shape,
                "max_shapes": max_shape,
            }
        )
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
            "input_shape": list(input_shape),
            "imgsz": [int(img_h), int(img_w)],
            "opset": int(args.onnx_opset),
            "device": export_device,
            "dynamic": bool(onnx_dynamic),
            "batch": int(onnx_batch),
            "max_batch": int(effective_max_batch) if effective_max_batch else None,
        },
        "trt": {
            "precision": str(args.trt_precision),
            "cuda_graph": bool(args.trt_cuda_graph),
            "profiling": bool(args.trt_profiling),
            "verbose": bool(args.trt_verbose),
            "builder_optimization_level": (
                int(builder_optimization_level)
                if builder_optimization_level is not None
                else None
            ),
            "trtexec_path": str(trtexec_path) if trtexec_path else None,
            "command": trt_cmd,
            "profile": trt_profile,
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
    # Record the custom loader's effective train worker count in Ultralytics'
    # args.yaml even though Palette owns DataLoader construction.
    params["workers"] = int(loader_cfg["num_workers"])
    # These Ultralytics transforms cannot run through the custom Zarr loader.
    # Set them explicitly so checkpoint args never imply that they were active.
    params["mosaic"] = 0.0
    params["mixup"] = 0.0
    params["copy_paste"] = 0.0
    params["cutmix"] = 0.0
    params["auto_augment"] = None
    params["multi_scale"] = False
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
    expected_manifest_sha256: Optional[str] = None,
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
        expected_manifest_sha256=expected_manifest_sha256,
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
    manifest_authority: Optional[dict[str, Any]] = None
    manifest_authority_sha256: Optional[str] = None

    # Load and validate config
    global config
    try:
        if args.log_registry and manifest_path is None:
            raise ValueError(
                "Registry-backed pose training requires a hash-verifiable training manifest."
            )
        full_config = PoseConfig.from_yaml(args.config_path)
        manifest_authority = _load_pose_manifest_authority(manifest_path)
        if manifest_authority is not None:
            manifest_authority_sha256 = manifest_authority["manifest_sha256"]
            authority_set_id = manifest_authority.get("set_id")
            if authority_set_id is not None and authority_set_id != effective_set_id:
                raise ValueError(
                    "Pose training manifest set_id disagrees with the selected training set."
                )

        config = build_pose_zarr_dataset_config(full_config)
        # Seed from the exact manifest so every registry lifecycle row binds the
        # same ordered schema before live-source consistency checks run.
        pose_schema = _infer_pose_schema(
            tuple(full_config.kpt_shape),
            {},
            manifest_pose_schema=(
                manifest_authority["pose_schema"]
                if manifest_authority is not None
                else None
            ),
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
                pose_schema=pose_schema,
                expected_manifest_sha256=manifest_authority_sha256,
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
            expected_manifest_sha256=manifest_authority_sha256,
        )

    # The hash-bound manifest is authoritative; selected live sources are
    # independent consistency evidence and cannot replace it.
    try:
        zarr_paths = config.get_zarr_paths()
        zarr_metadata = get_zarr_metadata(
            zarr_paths,
            console,
            keypoint_run_resolver=config.get_keypoint_run,
        )
        pose_schema = _infer_pose_schema(
            tuple(full_config.kpt_shape),
            zarr_metadata,
            manifest_pose_schema=(
                manifest_authority["pose_schema"]
                if manifest_authority is not None
                else None
            ),
        )
    except Exception as exc:
        console.print(f"[bold red]✗ Pose-schema source preflight failed:[/bold red] {exc}")
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
                    "stage": "pose_schema_source_preflight",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                pose_schema=pose_schema,
                expected_manifest_sha256=manifest_authority_sha256,
            )
        return 1

    # Display dataset info
    print_section_header(console, "Dataset Information")
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
    training_params["seed"] = int(full_config.random_seed)
    _resolve_project_dir(
        args=args,
        training_params=training_params,
        set_id=effective_set_id,
        config_path=config_path,
        console=console,
    )
    training_params, loader_cfg = _apply_pose_loader_training_param_overrides(training_params)
    model_name = training_params.get('model', 'yolov8n-pose.pt')
    pose_runtime_state: dict[str, Any] = {
        "status": "declared",
        "model_input_shape_hw": list(_normalize_imgsz(training_params.get("imgsz"))),
        "requested_training_params": _jsonable_runtime_value(training_params),
        "preprocessing_contract": full_config.preprocessing.model_dump(),
        "augmentation_contract": {
            "enabled": bool(full_config.training_params.augment),
            "parameters": {
                key: _jsonable_runtime_value(training_params.get(key))
                for key in (
                    "hsv_h",
                    "hsv_s",
                    "hsv_v",
                    "degrees",
                    "translate",
                    "scale",
                    "shear",
                    "perspective",
                    "fliplr",
                    "flipud",
                    "erasing",
                )
            },
            "semantic_pairs": full_config.augmentation.model_dump(),
            "implementation": {
                "algorithm": "palette_single_sample_pose_augmentation_v1",
                "geometry_order": "model_input_transform_then_augmentation",
                "affine_border_value_uint8": 114,
                "erasing_fill_value_uint8": 114,
                "keypoint_out_of_bounds_policy": "visibility_zero_coordinates_clipped",
            },
            "unsupported_ultralytics_transforms": {
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "cutmix": 0.0,
                "auto_augment": None,
                "multi_scale": False,
            },
        },
        "loader": {
            "requested_train_workers": int(loader_cfg["num_workers"]),
            "deterministic_validation": bool(loader_cfg["deterministic_val"]),
            "requested_val_workers": loader_cfg["val_num_workers"],
            "modes": {},
        },
        "datasets": {},
        "pose_schema": _jsonable_runtime_value(pose_schema),
        "training_manifest": {
            "path": str(manifest_path) if manifest_path is not None else None,
            "sha256": manifest_authority_sha256,
        },
    }
    ZarrPoseTrainer.runtime_contract_state = pose_runtime_state
    
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
        pose_runtime_state["starting_model"] = _pose_starting_model_receipt(
            model, str(model_name)
        )
        if args.log_registry and pose_runtime_state["starting_model"]["sha256"] is None:
            raise ValueError(
                "Registry-backed pose training requires a content-addressed starting "
                "model artifact after model initialization."
            )
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
                expected_manifest_sha256=manifest_authority_sha256,
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
        pose_runtime_state["datasets"][mode] = dataset.pose_preprocessing_receipt()
        pose_runtime_state["loader"]["modes"][mode] = {
            "workers": int(workers),
            "batch_size": int(batch_size),
            "shuffle": bool(mode == "train"),
            "persistent_workers": bool(persistent_workers),
            "prefetch_factor": prefetch_factor if workers > 0 else None,
        }
        _write_pose_runtime_receipt(pose_runtime_state)
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
            pose_runtime_state["run_dir"] = str(run_dir)
            pose_runtime_state["effective_arguments"] = (
                _validate_pose_effective_arguments(training_params, trainer.args)
            )
            pose_runtime_state["status"] = "effective_arguments_verified"
            written = _snapshot_training_inputs(
                run_dir=run_dir,
                config_path=config_path,
                manifest_path=manifest_path,
                invocation_payload=invocation_payload,
            )
            if written:
                console.print(f"[cyan]Snapshotted run inputs:[/cyan] {run_dir / 'inputs'}")
            receipt = _write_pose_runtime_receipt(pose_runtime_state)
            if receipt is not None:
                console.print(f"[cyan]Pose runtime receipt:[/cyan] {receipt}")
        except Exception as exc:
            raise RuntimeError(
                "Pose runtime contract validation failed before training."
            ) from exc

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
                expected_manifest_sha256=manifest_authority_sha256,
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
                expected_manifest_sha256=manifest_authority_sha256,
            )
        return 1
    
    if "first_batch" not in pose_runtime_state:
        raise RuntimeError(
            "Pose training finished without verifying an effective runtime batch."
        )
    pose_runtime_state["status"] = "verified"
    runtime_receipt_path = _write_pose_runtime_receipt(pose_runtime_state)
    runtime_receipt_sha256 = _shared_safe_sha256_file(runtime_receipt_path)
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
            'pose_runtime_receipt_path': str(runtime_receipt_path),
            'pose_runtime_receipt_sha256': runtime_receipt_sha256,
            'effective_pose_training_contract': {
                'preprocessing': pose_runtime_state.get('preprocessing_contract'),
                'augmentation': pose_runtime_state.get('augmentation_contract'),
                'arguments': pose_runtime_state.get('effective_arguments'),
                'first_batch': pose_runtime_state.get('first_batch'),
                'loader': pose_runtime_state.get('loader'),
            },
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
        trained_metrics_payload["pose_runtime_receipt_path"] = str(
            runtime_receipt_path
        )
        trained_metrics_payload["pose_runtime_receipt_sha256"] = (
            runtime_receipt_sha256
        )
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
            expected_manifest_sha256=manifest_authority_sha256,
        )

    def _export_checkpoint(stage_name: str, artifacts: Dict[str, Any]) -> None:
        if not args.log_registry:
            return
        metrics_path = results.save_dir / "results.csv"
        checkpoint_metrics_payload = dict(final_validation_metrics or {})
        checkpoint_metrics_payload["stage"] = stage_name
        checkpoint_metrics_payload["status_detail"] = f"{stage_name}_complete"
        checkpoint_metrics_payload["pose_runtime_receipt_path"] = str(
            runtime_receipt_path
        )
        checkpoint_metrics_payload["pose_runtime_receipt_sha256"] = (
            runtime_receipt_sha256
        )
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
            expected_manifest_sha256=manifest_authority_sha256,
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
        effective_img_h, effective_img_w = _normalize_imgsz(training_params.get("imgsz"))
        final_metrics_payload = dict(final_validation_metrics or {})
        final_metrics_payload.setdefault("stage", "completed")
        final_metrics_payload.setdefault("status_detail", "training_complete")
        final_metrics_payload.setdefault("imgsz_h", int(effective_img_h))
        final_metrics_payload.setdefault("imgsz_w", int(effective_img_w))
        final_metrics_payload.setdefault("effective_imgsz", [int(effective_img_h), int(effective_img_w)])
        final_metrics_payload["pose_runtime_receipt_path"] = str(
            runtime_receipt_path
        )
        final_metrics_payload["pose_runtime_receipt_sha256"] = (
            runtime_receipt_sha256
        )
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
            expected_manifest_sha256=manifest_authority_sha256,
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
        help=(
            "Pose training manifest JSON. Required when registry logging is enabled; "
            "its ordered pose schema is authoritative."
        ),
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
