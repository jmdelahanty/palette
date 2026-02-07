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
from collections import defaultdict
import hashlib
import math
import os
import random
import shutil
import subprocess
import sys
import re
import numpy as np
# Import NumPy before Torch to avoid MKL/libgomp threading-layer conflicts in some conda envs.
import torch
import yaml
from pathlib import Path
import time
import platform
import traceback
from typing import Optional
import pandas as pd
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from torch.utils.data import DataLoader
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
    profile_collector = None

    def reset(self):
        pass

    def __iter__(self):
        iterator = super().__iter__()
        profiler = self.profile_collector
        if profiler is None:
            return iterator
        return _ProfilingIterator(iterator, profiler)


class ChunkAwareBatchSampler:
    """Batch sampler that groups detect samples by frame chunk for better read locality."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        self._buckets = self._build_buckets()
        self._total_samples = sum(len(bucket) for bucket in self._buckets)

    def _build_buckets(self):
        buckets = defaultdict(list)
        frame_index_cache = getattr(self.dataset, "frame_index_cache", {})
        chunk_len_map = getattr(self.dataset, "detect_frame_chunk_len", {})

        for sample_index, (zarr_path, det_idx) in enumerate(getattr(self.dataset, "indices", [])):
            frame_indices = frame_index_cache.get(zarr_path)
            chunk_len = max(1, int(chunk_len_map.get(zarr_path, 1) or 1))

            frame_idx = int(det_idx)
            if frame_indices is not None and int(det_idx) < len(frame_indices):
                frame_idx = int(frame_indices[int(det_idx)])

            chunk_id = frame_idx // chunk_len
            buckets[(zarr_path, chunk_id)].append(sample_index)

        return list(buckets.values())

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        bucket_indices = list(range(len(self._buckets)))
        if self.shuffle:
            rng.shuffle(bucket_indices)

        ordered_sample_indices = []
        for bucket_idx in bucket_indices:
            bucket = list(self._buckets[bucket_idx])
            if self.shuffle:
                rng.shuffle(bucket)
            ordered_sample_indices.extend(bucket)

        for start in range(0, len(ordered_sample_indices), self.batch_size):
            batch = ordered_sample_indices[start:start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch

    def __len__(self):
        if self.batch_size <= 0:
            return 0
        if self.drop_last:
            return self._total_samples // self.batch_size
        return math.ceil(self._total_samples / self.batch_size)


class _ProfilingIterator:
    def __init__(self, iterator, profiler):
        self._iterator = iterator
        self._profiler = profiler

    def __iter__(self):
        return self

    def __next__(self):
        wait_start = time.perf_counter()
        batch = next(self._iterator)
        self._profiler.record_batch_wait(time.perf_counter() - wait_start, batch)
        return batch


class InputPipelineProfiler:
    """Collect coarse timing diagnostics for the detect training input pipeline."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._seconds = defaultdict(float)
        self._counts = defaultdict(int)

    def record(self, key: str, seconds: float, count: int = 1) -> None:
        if not self.enabled:
            return
        self._seconds[key] += float(max(0.0, seconds))
        self._counts[key] += int(max(0, count))

    def record_dataset_sample(self, payload: dict) -> None:
        if not self.enabled:
            return
        self.record("dataset_zarr_read_s", payload.get("zarr_read_s", 0.0), payload.get("samples", 1))
        self.record(
            "dataset_augment_preprocess_s",
            payload.get("augment_preprocess_s", 0.0),
            payload.get("samples", 1),
        )
        self.record("dataset_getitem_total_s", payload.get("getitem_total_s", 0.0), payload.get("samples", 1))

    def record_collate(self, seconds: float, batch_size: int) -> None:
        if not self.enabled:
            return
        self.record("collate_s", seconds, 1)
        self.record("collate_samples", 0.0, batch_size)

    def record_batch_wait(self, seconds: float, batch) -> None:
        if not self.enabled:
            return
        batch_size = 0
        if isinstance(batch, dict) and isinstance(batch.get("img"), torch.Tensor):
            batch_size = int(batch["img"].shape[0])
        self.record("dataloader_wait_s", seconds, 1)
        self.record("dataloader_samples", 0.0, batch_size)

    def record_preprocess_to_device(self, seconds: float, batch) -> None:
        if not self.enabled:
            return
        batch_size = 0
        if isinstance(batch, dict) and isinstance(batch.get("img"), torch.Tensor):
            batch_size = int(batch["img"].shape[0])
        self.record("preprocess_to_device_s", seconds, 1)
        self.record("preprocess_samples", 0.0, batch_size)

    @staticmethod
    def _avg_ms(total_s: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return (float(total_s) * 1000.0) / float(count)

    def summary(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        sections = {
            "dataset_zarr_read": (
                self._seconds["dataset_zarr_read_s"],
                self._counts["dataset_zarr_read_s"],
                self._counts["dataset_getitem_total_s"],
            ),
            "dataset_augment_preprocess": (
                self._seconds["dataset_augment_preprocess_s"],
                self._counts["dataset_augment_preprocess_s"],
                self._counts["dataset_getitem_total_s"],
            ),
            "dataset_getitem_total": (
                self._seconds["dataset_getitem_total_s"],
                self._counts["dataset_getitem_total_s"],
                self._counts["dataset_getitem_total_s"],
            ),
            "collate": (
                self._seconds["collate_s"],
                self._counts["collate_s"],
                self._counts["collate_samples"],
            ),
            "dataloader_wait": (
                self._seconds["dataloader_wait_s"],
                self._counts["dataloader_wait_s"],
                self._counts["dataloader_samples"],
            ),
            "preprocess_to_device": (
                self._seconds["preprocess_to_device_s"],
                self._counts["preprocess_to_device_s"],
                self._counts["preprocess_samples"],
            ),
        }
        payload = {
            "enabled": True,
            "stages": {},
            "notes": [
                "Dataset stages are measured inside __getitem__.",
                "dataloader_wait measures batch fetch latency seen by trainer iteration.",
                "preprocess_to_device includes Ultralytics preprocess + host-to-device transfer.",
            ],
        }
        for key, (seconds_total, calls, sample_count) in sections.items():
            payload["stages"][key] = {
                "total_seconds": float(seconds_total),
                "calls": int(calls),
                "samples": int(sample_count),
                "avg_ms_per_call": self._avg_ms(seconds_total, calls),
                "avg_ms_per_sample": self._avg_ms(seconds_total, sample_count),
            }
        return payload

    def render(self, console: Console) -> None:
        if not self.enabled:
            return
        summary = self.summary()
        table = Table(title=" Input Pipeline Profile")
        table.add_column("Stage", style="cyan")
        table.add_column("Total (s)", style="yellow")
        table.add_column("Calls", style="yellow")
        table.add_column("Samples", style="yellow")
        table.add_column("Avg ms/call", style="yellow")
        table.add_column("Avg ms/sample", style="yellow")
        for stage, stats in summary.get("stages", {}).items():
            table.add_row(
                stage,
                f"{stats['total_seconds']:.3f}",
                str(stats["calls"]),
                str(stats["samples"]),
                f"{stats['avg_ms_per_call']:.3f}",
                f"{stats['avg_ms_per_sample']:.3f}",
            )
        console.print(table)
        for note in summary.get("notes", []):
            console.print(f"[dim]- {note}[/dim]")


_ACTIVE_INPUT_PIPELINE_PROFILER: InputPipelineProfiler | None = None


def det_collate_fn(batch):
    """Collate function for detection data."""
    profiler = _ACTIVE_INPUT_PIPELINE_PROFILER
    collate_start = time.perf_counter() if profiler is not None and profiler.enabled else 0.0
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
        result = {
            'img': images,
            'batch_idx': torch.empty(0, dtype=torch.long),
            'cls': torch.empty(0, 1, dtype=torch.float32),
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }
    else:
        result = {
            'img': images,
            'batch_idx': torch.cat(batch_idx_list, 0),
            'cls': torch.cat(cls_list, 0),
            'bboxes': torch.cat(bboxes_list, 0),
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }

    if profiler is not None and profiler.enabled:
        profiler.record_collate(time.perf_counter() - collate_start, batch_size=len(batch))

    return result


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
    profile_collector = None

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        profiler = self.profile_collector
        if profiler is None or not profiler.enabled:
            return super().preprocess_batch(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_start = time.perf_counter()
        out = super().preprocess_batch(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.record_preprocess_to_device(time.perf_counter() - preprocess_start, batch)
        return out

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
                raw_video = root['raw_video']
                if 'images_full' in raw_video:
                    arr = raw_video['images_full']
                    zarr_meta['video_frames'] = int(arr.shape[0])
                    if len(arr.shape) >= 3:
                        zarr_meta['frame_height'] = int(arr.shape[1])
                        zarr_meta['frame_width'] = int(arr.shape[2])
                elif 'images_ds' in raw_video:
                    arr = raw_video['images_ds']
                    zarr_meta['video_frames'] = int(arr.shape[0])
                    if len(arr.shape) >= 3:
                        zarr_meta['frame_height'] = int(arr.shape[1])
                        zarr_meta['frame_width'] = int(arr.shape[2])
                elif 'images_ds_rgb' in raw_video:
                    arr = raw_video['images_ds_rgb']
                    zarr_meta['video_frames'] = int(arr.shape[0])
                    if len(arr.shape) >= 3:
                        zarr_meta['frame_height'] = int(arr.shape[1])
                        zarr_meta['frame_width'] = int(arr.shape[2])
                zarr_meta['fps'] = raw_video.attrs.get('fps', 'N/A')
            
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
                if latest_crop and latest_crop in root['crop_runs']:
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
                    elif 'bbox_norm_coords' in crop_group:
                        # Merged training Zarrs store labels directly in bbox_norm_coords without summary_statistics.
                        total_rois = int(crop_group['bbox_norm_coords'].shape[0])
                        zarr_meta['crop_info']['total_rois'] = total_rois
                        if 'frame_indices' in crop_group:
                            zarr_meta['crop_info']['frames_with_crops'] = int(crop_group['frame_indices'].shape[0])

                    # If interpolated, get breakdown
                    if includes_interpolated:
                        if 'n_real_detections' in crop_group.attrs and 'n_interpolated_detections' in crop_group.attrs:
                            zarr_meta['crop_info']['n_real'] = crop_group.attrs.get('n_real_detections', 0)
                            zarr_meta['crop_info']['n_interpolated'] = crop_group.attrs.get('n_interpolated_detections', 0)
                        elif 'detection_source' in crop_group:
                            det_src = np.asarray(crop_group['detection_source'][:], dtype=np.int64)
                            n_interpolated = int(np.count_nonzero(det_src != 0))
                            zarr_meta['crop_info']['n_interpolated'] = n_interpolated
                            zarr_meta['crop_info']['n_real'] = int(det_src.shape[0] - n_interpolated)
            
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


def _should_enable_rect_for_non_square_inputs(zarr_metadata: dict) -> bool:
    """Return True when at least one dataset has non-square frame dimensions."""
    for meta in zarr_metadata.values():
        if not isinstance(meta, dict) or 'error' in meta:
            continue
        h = meta.get('frame_height')
        w = meta.get('frame_width')
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0 and h != w:
            return True
    return False


def _apply_zarr_loader_training_param_overrides(training_params: dict) -> tuple[dict, dict]:
    """Normalize args passed to Ultralytics and return custom-loader augmentation summary."""
    params = dict(training_params)
    custom_loader_aug_keys = (
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
    custom_loader_aug = {
        key: float(params.get(key, 0.0) or 0.0)
        for key in custom_loader_aug_keys
    }
    # Loader-only setting: not a valid Ultralytics train() argument.
    custom_loader_aug["chunk_cache_size"] = int(params.pop("chunk_cache_size", 0) or 0)
    custom_loader_aug["persistent_workers"] = bool(params.pop("persistent_workers", False))
    custom_loader_aug["chunk_locality_sampling"] = bool(params.pop("chunk_locality_sampling", False))
    custom_loader_aug["num_workers"] = max(0, int(params.pop("num_workers", 16) or 0))
    prefetch_raw = params.pop("prefetch_factor", None)
    custom_loader_aug["prefetch_factor"] = None if prefetch_raw is None else max(1, int(prefetch_raw))
    custom_loader_aug["deterministic_val"] = bool(params.pop("deterministic_val", True))
    val_num_workers_raw = params.pop("val_num_workers", None)
    custom_loader_aug["val_num_workers"] = (
        None if val_num_workers_raw is None else max(0, int(val_num_workers_raw or 0))
    )

    # We use a custom Zarr loader, so keep Ultralytics multi-sample augmentation knobs neutral.
    params.setdefault("augment", False)
    params.setdefault("mosaic", 0.0)
    params.setdefault("mixup", 0.0)
    params.setdefault("cutmix", 0.0)
    params.setdefault("copy_paste", 0.0)
    params.setdefault("close_mosaic", 0)
    params.setdefault("auto_augment", None)
    return params, custom_loader_aug


def _shutdown_dataloader_workers(loader) -> bool:
    """Best-effort shutdown for DataLoader worker processes (persistent workers)."""
    if loader is None:
        return False
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return False
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if not callable(shutdown):
        return False
    shutdown()
    try:
        loader._iterator = None  # type: ignore[attr-defined]
    except Exception:
        pass
    return True


def _cleanup_trainer_dataloaders(model, console: Optional[Console] = None) -> int:
    """Best-effort shutdown of trainer/validator DataLoader workers before process exit."""
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return 0

    cleaned = 0
    seen = set()

    def _try(loader) -> None:
        nonlocal cleaned
        if loader is None:
            return
        identity = id(loader)
        if identity in seen:
            return
        seen.add(identity)
        try:
            if _shutdown_dataloader_workers(loader):
                cleaned += 1
        except Exception as exc:
            if console is not None:
                console.print(f"[yellow]Warning: DataLoader shutdown failed: {exc}[/yellow]")

    _try(getattr(trainer, "train_loader", None))
    _try(getattr(trainer, "test_loader", None))

    validator = getattr(trainer, "validator", None)
    if validator is not None:
        _try(getattr(validator, "dataloader", None))
        _try(getattr(validator, "test_loader", None))

    if cleaned and console is not None:
        console.print(f"[dim]Shut down {cleaned} DataLoader worker pool(s).[/dim]")

    return cleaned


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
        set_name = payload.get("set_name") if isinstance(payload, dict) else None
        set_slug = _strip_manifest_suffixes(str(manifest_set_id or set_name or "")).strip() or None
        manifest_canvas = (
            payload.get("canvas_name")
            if isinstance(payload, dict)
            else None
        ) or (
            first_dataset.get("canvas_name")
            or rig_info.get("canvas_name")
        )
        manifest_dish = (
            (payload.get("dish_design") if isinstance(payload, dict) else None)
            or query_filter.get("dish_design")
            or query_filter.get("dish_design_like")
            or first_dataset.get("dish_design")
            or arena.get("dish_design")
        )
        manifest_rig = (
            (payload.get("rig_name") if isinstance(payload, dict) else None)
            or query_filter.get("rig_id")
            or first_dataset.get("rig_id")
            or rig_info.get("rig_id")
        )
        return {
            "manifest_path": str(path),
            "manifest_sha256": digest,
            "manifest_dataset_ids": dataset_ids,
            "manifest_dataset_count": len(dataset_ids),
            "manifest_set_id": manifest_set_id,
            "manifest_set_slug": set_slug,
            "manifest_task": payload.get("task") if isinstance(payload, dict) else None,
            "manifest_rig_name": manifest_rig,
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


def _sanitize_run_component(value: str | None, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


def _resolve_manifest_version_token(manifest_summary: dict) -> str:
    for key in ("manifest_set_id", "manifest_set_slug"):
        raw = str(manifest_summary.get(key) or "").strip().lower()
        if not raw:
            continue
        match = re.search(r"_v(?P<num>\d+)$", raw)
        if not match:
            continue
        try:
            version_num = int(match.group("num"))
        except Exception:
            continue
        if version_num >= 0:
            return f"v{version_num:03d}"
    return "v001"


def _resolve_run_hash(
    *,
    manifest_summary: dict,
    task: str,
    stamp: str,
    pid: int,
) -> str:
    manifest_sha = str(manifest_summary.get("manifest_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{8,}", manifest_sha):
        return manifest_sha[:8]

    seed = "|".join(
        [
            str(manifest_summary.get("manifest_set_id") or ""),
            str(manifest_summary.get("manifest_set_slug") or ""),
            str(manifest_summary.get("manifest_rig_name") or ""),
            str(manifest_summary.get("manifest_dish_design") or ""),
            str(manifest_summary.get("manifest_canvas_name") or ""),
            task,
            stamp,
            str(pid),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


def _infer_dish_canvas_from_set_slug(manifest_summary: dict) -> tuple[str | None, str | None]:
    raw = str(manifest_summary.get("manifest_set_slug") or manifest_summary.get("manifest_set_id") or "").strip()
    if not raw:
        return None, None
    slug = _sanitize_run_component(raw, "")
    if not slug:
        return None, None
    slug = re.sub(r"_v\d+$", "", slug)
    if slug.startswith("detect_"):
        slug = slug[len("detect_") :]
    parts = [part for part in slug.split("_") if part]
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def _build_default_run_name(
    *,
    manifest_summary: dict,
    task_fallback: str,
    timestamp: str | None = None,
    pid: int | None = None,
) -> str:
    fallback_dish, fallback_canvas = _infer_dish_canvas_from_set_slug(manifest_summary)
    rig = _sanitize_run_component(manifest_summary.get("manifest_rig_name"), "unknown_rig")
    dish = _sanitize_run_component(manifest_summary.get("manifest_dish_design") or fallback_dish, "unknown_dish")
    canvas = _sanitize_run_component(
        manifest_summary.get("manifest_canvas_name") or fallback_canvas,
        "unknown_canvas",
    )
    version = _resolve_manifest_version_token(manifest_summary)
    task = _sanitize_run_component(manifest_summary.get("manifest_task") or task_fallback, task_fallback)
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    process_id = int(os.getpid() if pid is None else pid)
    short_hash = _resolve_run_hash(
        manifest_summary=manifest_summary,
        task=task,
        stamp=stamp,
        pid=process_id,
    )
    return f"{rig}_{dish}_{canvas}_{version}_{task}_{stamp}_{short_hash}"


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


def _write_input_pipeline_profile(
    *,
    profiler: InputPipelineProfiler | None,
    run_dir: Path | None,
    console: Console,
) -> dict | None:
    if profiler is None or not profiler.enabled:
        return None

    payload = profiler.summary()
    profiler.render(console)
    if run_dir is None:
        return payload

    try:
        out_path = run_dir / "input_pipeline_profile.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[cyan]Wrote input pipeline profile:[/cyan] {out_path}")
    except Exception as exc:
        console.print(f"[yellow]Warning: failed to write input pipeline profile: {exc}[/yellow]")
    return payload


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
                        "build_env": export_artifacts.get("onnx_build_env"),
                        "metadata_props": export_artifacts.get("onnx_metadata_props"),
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
                        "build_env": export_artifacts.get("build_env"),
                        "trt_device_info": export_artifacts.get("trt_device_info"),
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


def _imgsz_to_config_value(img_h: int, img_w: int) -> int | list[int]:
    h = int(img_h)
    w = int(img_w)
    return h if h == w else [h, w]


def _extract_runtime_imgsz(model, fallback_imgsz) -> tuple[int, int]:
    """Read imgsz from Ultralytics trainer args after training; fallback when unavailable."""
    fallback = _normalize_imgsz(fallback_imgsz)
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return fallback
    trainer_args = getattr(trainer, "args", None)
    if trainer_args is None:
        return fallback
    return _normalize_imgsz(getattr(trainer_args, "imgsz", fallback))


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


def _resolve_trtexec_path(explicit_path: str | None) -> Path | None:
    """Resolve trtexec path from explicit CLI value, module default, or PATH."""
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


def _collect_export_env(trtexec_path: Path | None, trtexec_log_path: Path | None = None) -> dict:
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
        version, source, raw_output = _read_trtexec_version(trtexec_path)
        env["tensorrt_version"] = version
        if source:
            env["tensorrt_version_source"] = source
        if raw_output:
            env["trtexec_version_output"] = raw_output
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


def _collect_onnx_output_contract(onnx_path: Path) -> list[dict]:
    """Collect ONNX output tensor names/shapes/dtypes for export manifests."""
    try:
        import onnx  # type: ignore
        from onnx import TensorProto  # type: ignore
    except Exception:
        return []

    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return []

    outputs: list[dict] = []
    for value_info in model.graph.output:
        tensor_type = value_info.type.tensor_type
        elem_type = int(getattr(tensor_type, "elem_type", 0) or 0)
        dtype_name = TensorProto.DataType.Name(elem_type) if elem_type > 0 else "UNDEFINED"
        dims: list = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
        outputs.append(
            {
                "name": str(value_info.name),
                "shape": dims,
                "dtype": dtype_name,
            }
        )
    return outputs


def _collect_onnx_metadata_props(onnx_path: Path) -> dict[str, str]:
    """Collect ONNX metadata_props as a flat string map for auditability."""
    try:
        import onnx  # type: ignore
    except Exception:
        return {}

    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return {}

    props: dict[str, str] = {}
    for item in getattr(model, "metadata_props", []) or []:
        key = str(getattr(item, "key", "") or "").strip()
        if not key:
            continue
        props[key] = str(getattr(item, "value", "") or "")
    return props


def _export_detection_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    weights_path: Path,
    training_params: dict,
    export_imgsz: tuple[int, int] | None,
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

    canonical_onnx_path = onnx_dir / f"{run_id}.onnx"
    existing_onnx_path = None
    if getattr(args, "onnx_path", None):
        existing_onnx_path = Path(args.onnx_path).expanduser().resolve()
        if not existing_onnx_path.exists():
            export_info["errors"].append(f"onnx_not_found:{existing_onnx_path}")
            return export_info
    elif args.export_trt and not args.export_onnx and canonical_onnx_path.exists():
        # TRT-only flow: reuse canonical ONNX artifact when already present.
        existing_onnx_path = canonical_onnx_path

    if export_imgsz is None:
        img_h, img_w = _normalize_imgsz(training_params.get("imgsz"))
    else:
        img_h, img_w = int(export_imgsz[0]), int(export_imgsz[1])
    input_shape = [1, 3, img_h, img_w]
    export_device = _resolve_export_device(training_params.get("device"))

    onnx_path = existing_onnx_path or canonical_onnx_path
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
    onnx_cmd.extend(["--meta-run-id", str(run_id)])
    manifest_set_id = str(manifest_summary.get("manifest_set_id") or "").strip()
    if manifest_set_id:
        onnx_cmd.extend(["--meta-set-id", manifest_set_id])
    manifest_sha256 = str(manifest_summary.get("manifest_sha256") or "").strip().lower()
    if manifest_sha256:
        onnx_cmd.extend(["--meta-manifest-sha256", manifest_sha256])
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
    onnx_output_contract = _collect_onnx_output_contract(onnx_path)
    onnx_metadata_props = _collect_onnx_metadata_props(onnx_path)
    onnx_build_env = _collect_export_env(None, trtexec_log_path=None)
    export_info["weights_sha256"] = weights_sha
    export_info["onnx_sha256"] = onnx_sha
    export_info["onnx_source"] = "existing" if existing_onnx_path else "exported"
    export_info["onnx_output_contract"] = onnx_output_contract
    export_info["onnx_metadata_props"] = onnx_metadata_props
    export_info["onnx_build_env"] = onnx_build_env

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
            "outputs": onnx_output_contract,
            "metadata_props": onnx_metadata_props,
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
        "build_env": onnx_build_env,
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
    manifest_path = trt_dir / f"{engine_name}.tensorrt.manifest.json"
    trt_log_path = trt_dir / f"{engine_name}_trtexec.log"
    export_info["trt_log_path"] = str(trt_log_path)

    trtexec_path = _resolve_trtexec_path(getattr(args, "trtexec", None))
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
    if trtexec_path is not None:
        trt_cmd.extend(["--trtexec", str(trtexec_path)])
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
                "outputs": onnx_output_contract,
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
            "build_env": _collect_export_env(trtexec_path, trtexec_log_path=trt_log_path),
            "source_manifest": {
                "manifest_path": manifest_summary.get("manifest_path"),
                "manifest_sha256": manifest_summary.get("manifest_sha256"),
                "manifest_dataset_ids": manifest_summary.get("manifest_dataset_ids"),
            },
        }
        manifest_text = json.dumps(engine_manifest, indent=2)
        manifest_path.write_text(manifest_text)

    export_info.update(
        {
            "engine_path": str(engine_path),
            "engine_manifest_path": str(manifest_path),
            "build_env": engine_manifest.get("build_env") if engine_path.exists() else None,
            "trt_device_info": (
                engine_manifest.get("build_env", {}).get("trtexec_runtime")
                if engine_path.exists()
                else None
            ),
            "engine_sha256": _safe_sha256_file(engine_path),
            "engine_manifest_sha256": _safe_sha256_file(manifest_path),
            "onnx_manifest_sha256": _safe_sha256_file(onnx_manifest_path),
        }
    )
    return export_info


def main(args):
    global _ACTIVE_INPUT_PIPELINE_PROFILER
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
    input_profiler = InputPipelineProfiler(enabled=bool(getattr(args, "profile", False)))
    input_profile_payload = None

    try:
        # Load and validate config
        full_config = DetectConfig.from_yaml(args.config_path)
        allow_source_mismatch = bool(full_config.allow_source_mismatch or args.allow_source_mismatch)
        
        # Extract dataset config fields from flat structure
        tp = full_config.training_params
        zarr_config_dict = {
            'datasets': full_config.datasets,
            'task': full_config.task,
            'random_seed': full_config.random_seed,
            'sampling_strategy': full_config.sampling_strategy,
            'dataset_weights': full_config.dataset_weights,
            'allow_source_mismatch': allow_source_mismatch,
            'chunk_cache_size': int(tp.chunk_cache_size or 0),
            'augmentation': {
                'hsv_h': float(tp.hsv_h or 0.0),
                'hsv_s': float(tp.hsv_s or 0.0),
                'hsv_v': float(tp.hsv_v or 0.0),
                'degrees': float(tp.degrees or 0.0),
                'translate': float(tp.translate or 0.0),
                'scale': float(tp.scale or 0.0),
                'shear': float(tp.shear or 0.0),
                'perspective': float(tp.perspective or 0.0),
                'fliplr': float(tp.fliplr or 0.0),
                'flipud': float(tp.flipud or 0.0),
                'erasing': float(tp.erasing or 0.0),
            },
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
    persistent_workers_flag = False
    chunk_locality_sampling_flag = False
    num_workers_value = 16
    prefetch_factor_value = None
    profile_mode = bool(getattr(args, "profile", False))
    if profile_mode:
        num_workers_value = 0
        console.print(
            "[yellow]Profile mode enabled: forcing num_workers=0 for deterministic stage timing.[/yellow]"
        )
        _ACTIVE_INPUT_PIPELINE_PROFILER = input_profiler
        DetTrainer.profile_collector = input_profiler
        YoloCompatibleDataLoader.profile_collector = input_profiler
    else:
        _ACTIVE_INPUT_PIPELINE_PROFILER = None
        DetTrainer.profile_collector = None
        YoloCompatibleDataLoader.profile_collector = None

    deterministic_val_flag = True
    val_num_workers_value = None
    val_seed_base = int(getattr(full_config, "random_seed", 42))

    def _seed_val_worker(worker_id: int) -> None:
        worker_seed = (val_seed_base + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def get_zarr_dataloader(self, dataset_path, batch_size=16, mode="train", rank=0):
        mode = str(mode or "train").lower()
        is_train = mode == "train"
        dataset = create_zarr_dataset(config=config, mode=mode)
        if profile_mode:
            dataset._profile_callback = input_profiler.record_dataset_sample
        if is_train:
            active_num_workers = num_workers_value
        elif val_num_workers_value is not None:
            active_num_workers = val_num_workers_value
        else:
            active_num_workers = num_workers_value

        loader_persistent_workers = bool(is_train and persistent_workers_flag and active_num_workers > 0)
        loader_kwargs = {
            "num_workers": active_num_workers,
            "pin_memory": True,
            "persistent_workers": loader_persistent_workers,
        }
        if not is_train and deterministic_val_flag:
            loader_kwargs["persistent_workers"] = False
            if active_num_workers > 0:
                val_generator = torch.Generator()
                val_generator.manual_seed(val_seed_base)
                loader_kwargs["worker_init_fn"] = _seed_val_worker
                loader_kwargs["generator"] = val_generator

        if active_num_workers > 0 and prefetch_factor_value is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor_value)
        use_chunk_locality_sampling = bool(
            is_train
            and chunk_locality_sampling_flag
            and getattr(config, "task", "detect") == "detect"
        )
        if use_chunk_locality_sampling:
            batch_sampler = ChunkAwareBatchSampler(
                dataset=dataset,
                batch_size=int(batch_size),
                seed=int(getattr(full_config, "random_seed", 42)),
                drop_last=False,
                shuffle=True,
            )
            return YoloCompatibleDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=det_collate_fn,
                **loader_kwargs,
            )

        return YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=det_collate_fn,
            **loader_kwargs,
        )

    DetTrainer.get_dataloader = get_zarr_dataloader

    # Get training params
    training_params = full_config.training_params.model_dump(exclude_none=True)
    training_params, custom_loader_augment = _apply_zarr_loader_training_param_overrides(training_params)
    persistent_workers_flag = bool(custom_loader_augment.get("persistent_workers", False))
    chunk_locality_sampling_flag = bool(custom_loader_augment.get("chunk_locality_sampling", False))
    if not profile_mode:
        num_workers_value = max(0, int(custom_loader_augment.get("num_workers", 16) or 0))
        prefetch_factor_value = custom_loader_augment.get("prefetch_factor")
        deterministic_val_flag = bool(custom_loader_augment.get("deterministic_val", True))
        val_num_workers_raw = custom_loader_augment.get("val_num_workers", None)
        val_num_workers_value = (
            None if val_num_workers_raw is None else max(0, int(val_num_workers_raw or 0))
        )
    else:
        requested_prefetch = custom_loader_augment.get("prefetch_factor")
        custom_loader_augment["num_workers"] = 0
        custom_loader_augment["prefetch_factor"] = None
        custom_loader_augment["val_num_workers"] = 0
        if requested_prefetch is not None:
            console.print(
                "[dim]Ignoring prefetch_factor in profile mode because num_workers is forced to 0.[/dim]"
            )
    if not bool(training_params.get("rect", False)) and _should_enable_rect_for_non_square_inputs(zarr_metadata):
        training_params["rect"] = True
        console.print(
            "[yellow]Auto-enabled rect=True because non-square input frames were detected.[/yellow]"
        )
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
        _ACTIVE_INPUT_PIPELINE_PROFILER = None
        DetTrainer.profile_collector = None
        YoloCompatibleDataLoader.profile_collector = None
        raise

    # Display hyperparameters
    params_str = json.dumps(training_params, indent=2)
    custom_loader_aug_str = json.dumps(custom_loader_augment, indent=2)
    console.print(
        Panel(
            custom_loader_aug_str,
            title="[bold cyan]Custom Loader Augmentations[/bold cyan]",
            expand=False,
        )
    )
    console.print(
        "[dim]Ultralytics built-in mosaic/mixup/cutmix/copy_paste/auto_augment are held neutral for Zarr loader runs.[/dim]"
    )
    console.print()
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
        _cleanup_trainer_dataloaders(model, console=console)
        trainer_obj = getattr(model, "trainer", None)
        run_dir = None
        if trainer_obj is not None and getattr(trainer_obj, "save_dir", None):
            run_dir = Path(trainer_obj.save_dir)
        input_profile_payload = _write_input_pipeline_profile(
            profiler=input_profiler,
            run_dir=run_dir,
            console=console,
        )
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
                    "input_pipeline_profile": input_profile_payload,
                },
            )
        _ACTIVE_INPUT_PIPELINE_PROFILER = None
        DetTrainer.profile_collector = None
        YoloCompatibleDataLoader.profile_collector = None
        raise

    training_duration_seconds = time.time() - training_start_time
    effective_img_h, effective_img_w = _extract_runtime_imgsz(model, training_params.get("imgsz"))
    training_params["imgsz"] = _imgsz_to_config_value(effective_img_h, effective_img_w)

    export_artifacts = {}
    if args.export_onnx or args.export_trt:
        weights_path = results.save_dir / "weights" / "best.pt"
        export_artifacts = _export_detection_artifacts(
            run_dir=results.save_dir,
            run_id=results.save_dir.name,
            weights_path=weights_path,
            training_params=training_params,
            export_imgsz=(effective_img_h, effective_img_w),
            args=args,
            manifest_summary=manifest_summary,
            console=console,
        )

    input_profile_payload = _write_input_pipeline_profile(
        profiler=input_profiler,
        run_dir=Path(results.save_dir),
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
        if isinstance(final_report.get("training_params"), dict):
            final_report["training_params"]["imgsz"] = _imgsz_to_config_value(effective_img_h, effective_img_w)
        final_report['training_history'] = {
            'source_zarr_metadata': zarr_metadata,
            'custom_loader_augmentation': custom_loader_augment,
            'effective_imgsz': [int(effective_img_h), int(effective_img_w)],
            'effective_training_params': dict(training_params),
            'input_pipeline_profile': input_profile_payload,
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
        final_metrics_payload.setdefault("imgsz_h", int(effective_img_h))
        final_metrics_payload.setdefault("imgsz_w", int(effective_img_w))
        if input_profile_payload is not None:
            final_metrics_payload.setdefault("input_pipeline_profile", input_profile_payload)
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

    _cleanup_trainer_dataloaders(model, console=console)
    _ACTIVE_INPUT_PIPELINE_PROFILER = None
    DetTrainer.profile_collector = None
    YoloCompatibleDataLoader.profile_collector = None
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
        "--profile",
        action="store_true",
        help=(
            "Enable input-pipeline profiling for this run. "
            "Collects per-stage timing breakdown and writes input_pipeline_profile.json."
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
