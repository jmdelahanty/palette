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
import random
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
from typing import Any, Optional
import pandas as pd
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
import json
import zarr

from .config import DetectConfig, DatasetConfig
from .export_shared import (
    _parse_trtexec_device_info_text as _shared_parse_trtexec_device_info_text,
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
    build_default_detect_run_name as _shared_build_default_detect_run_name,
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
from ..shared.refined_detect_curation import (
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    has_curated_refined_source_detections_projection,
    has_curated_refined_detect_surface,
    has_sparse_curated_refined_detect_instances_arrays,
)
from ..shared.refined_detect_resolution import (
    resolve_detection_read_source,
)
from ..shared.zarr_run_completion import resolve_latest_complete_run_name
from ..utils.system import get_git_info, build_invocation_record

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"


def _parse_trtexec_device_info_text(raw_text: str) -> dict:
    """Backward-compatible shim; use shared parser implementation."""
    return _shared_parse_trtexec_device_info_text(raw_text)


# Custom DataLoader to ensure compatibility with Ultralytics YOLO's expected interface
class YoloCompatibleDataLoader(DataLoader):
    profile_collector = None
    shape_collector = None
    shape_mode = "unknown"

    def reset(self):
        pass

    def __iter__(self):
        iterator = super().__iter__()
        profiler = self.profile_collector
        shape_collector = self.shape_collector
        if profiler is None and shape_collector is None:
            return iterator
        return _ObservingIterator(iterator, profiler, shape_collector, mode=self.shape_mode)


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


class _ObservingIterator:
    def __init__(self, iterator, profiler, shape_collector=None, *, mode: str = "unknown"):
        self._iterator = iterator
        self._profiler = profiler
        self._shape_collector = shape_collector
        self._mode = str(mode or "unknown")

    def __iter__(self):
        return self

    def __next__(self):
        wait_start = time.perf_counter()
        batch = next(self._iterator)
        if self._profiler is not None:
            self._profiler.record_batch_wait(time.perf_counter() - wait_start, batch)
        if self._shape_collector is not None:
            self._shape_collector.record_batch(self._mode, batch)
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


class TrainingShapeObserver:
    """Collect observed dataloader tensor shapes in the main process."""

    def __init__(self) -> None:
        self._mode_counts: dict[str, int] = defaultdict(int)
        self._shape_counts: dict[str, dict[tuple[int, int, int], int]] = defaultdict(lambda: defaultdict(int))

    def record_batch(self, mode: str, batch) -> None:
        if not isinstance(batch, dict):
            return
        images = batch.get("img")
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            return
        mode_key = str(mode or "unknown")
        batch_size, channels, height, width = [int(v) for v in images.shape]
        self._mode_counts[mode_key] += int(batch_size)
        self._shape_counts[mode_key][(channels, height, width)] += int(batch_size)

    def summary(self) -> dict:
        modes = {}
        for mode, shape_counts in sorted(self._shape_counts.items()):
            shapes = [
                {
                    "channels": int(channels),
                    "height": int(height),
                    "width": int(width),
                    "samples": int(samples),
                }
                for (channels, height, width), samples in sorted(
                    shape_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            modes[mode] = {
                "samples": int(self._mode_counts.get(mode, 0)),
                "unique_shape_count": len(shapes),
                "shapes": shapes,
            }
        return {"modes": modes}


def _source_frame_shape_summary(zarr_metadata: dict) -> list[dict]:
    rows = []
    for dataset_name, meta in sorted(zarr_metadata.items()):
        if not isinstance(meta, dict) or "error" in meta:
            continue
        frame_h = meta.get("frame_height")
        frame_w = meta.get("frame_width")
        if frame_h is None or frame_w is None:
            continue
        rows.append(
            {
                "dataset": str(dataset_name),
                "frame_height": int(frame_h),
                "frame_width": int(frame_w),
            }
        )
    return rows


def _build_model_input_shape_accounting(
    *,
    requested_imgsz,
    rect: bool,
    ultralytics_train_imgsz: tuple[int, int],
    zarr_metadata: dict,
    shape_observer: TrainingShapeObserver | None,
) -> dict:
    requested_h, requested_w = _normalize_imgsz(requested_imgsz)
    payload = {
        "requested_imgsz": _imgsz_to_config_value(requested_h, requested_w),
        "requested_imgsz_hw": [int(requested_h), int(requested_w)],
        "rect": bool(rect),
        "ultralytics_train_imgsz": _imgsz_to_config_value(*ultralytics_train_imgsz),
        "ultralytics_train_imgsz_hw": [int(ultralytics_train_imgsz[0]), int(ultralytics_train_imgsz[1])],
        "source_frame_shapes": _source_frame_shape_summary(zarr_metadata),
        "observed_dataloader_batch_shapes": (
            shape_observer.summary() if shape_observer is not None else {"modes": {}}
        ),
        "notes": [
            "Ultralytics train mode normalizes rectangular imgsz requests to a scalar max-side value.",
            "Observed dataloader shapes are batch tensors yielded to the trainer before device preprocessing.",
        ],
    }
    return payload


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
                'data_quality': {},
                'detect_source_options': {},
                'detect_source_default': {},
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
                latest_detect = resolve_latest_complete_run_name(root['detect_runs'], legacy_default=True)
                if latest_detect:
                    detect_group = root[f'detect_runs/{latest_detect}']
                    zarr_meta['detect_source_options']['detect'] = f"detect_runs/{latest_detect}"
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
                latest_crop = resolve_latest_complete_run_name(root['crop_runs'], legacy_default=True)
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
                latest_refined = resolve_latest_complete_run_name(refined_root, legacy_default=True)
                if latest_refined and latest_refined in refined_root:
                    refined_group = refined_root[latest_refined]
                    parent_name = REFINED_DETECT_GROUP if REFINED_DETECT_GROUP in root else LEGACY_REFINED_DETECT_GROUP
                    
                    zarr_meta['data_quality']['has_refinement'] = True
                    zarr_meta['data_quality']['refined_run'] = latest_refined
                    
                    if has_curated_refined_detect_surface(refined_group):
                        if has_sparse_curated_refined_detect_instances_arrays(refined_group):
                            refined_path = f"{parent_name}/{latest_refined}/instances"
                        else:
                            refined_path = f"{parent_name}/{latest_refined}"
                        zarr_meta['detect_source_options']['refined'] = refined_path
                        summary_stats = refined_group.attrs.get("summary_statistics")
                        if isinstance(summary_stats, dict):
                            zarr_meta['data_quality']['refined_detections'] = summary_stats.get(
                                "sparse_instance_rows",
                                summary_stats.get("rows_present"),
                            )
                            zarr_meta['data_quality']['refined_filtered_rows'] = summary_stats.get(
                                "rows_filtered_out",
                                summary_stats.get("source_detection_filtered"),
                            )
                            zarr_meta['data_quality']['refined_missing_rows'] = summary_stats.get("rows_missing")
                            zarr_meta['data_quality']['manual_edited_rows'] = summary_stats.get("rows_manual_edited")
                        try:
                            present_rows = extract_present_curated_rows(refined_group)
                            detection_source = np.asarray(present_rows["detection_source"], dtype=np.int8)
                            zarr_meta['data_quality']['interpolated_detections'] = int(
                                np.count_nonzero(detection_source != 0)
                            )
                            manual_flags = present_rows.get("manual_edit_flags")
                            if manual_flags is not None:
                                zarr_meta['data_quality']['manual_edited_detections'] = int(
                                    np.count_nonzero(np.asarray(manual_flags, dtype=bool))
                                )
                        except Exception:
                            pass
                        if has_curated_refined_source_detections_projection(refined_group):
                            try:
                                source_summary = build_source_detection_decision_summary(refined_group)
                                if source_summary:
                                    zarr_meta['data_quality']['source_detection_candidates'] = int(
                                        source_summary.get('total_candidates', 0) or 0
                                    )
                                    for label in ("accepted", "filtered", "duplicate", "manual_clear"):
                                        zarr_meta['data_quality'][f'source_detection_{label}'] = int(
                                            source_summary.get(f'decision_{label}', 0) or 0
                                        )
                            except Exception:
                                pass

                    # Check what sparse legacy stages exist
                    if 'filtered' in refined_group:
                        filtered_grp = refined_group['filtered']
                        zarr_meta['data_quality']['filtered_detections'] = filtered_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['jumps_removed'] = filtered_grp.attrs.get('dropped_detections', 0)
                        zarr_meta['detect_source_options']['filtered'] = f"{parent_name}/{latest_refined}/filtered"
                    
                    if 'interpolated' in refined_group:
                        interp_grp = refined_group['interpolated']
                        zarr_meta['data_quality']['interpolated_detections'] = interp_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['gaps_filled'] = interp_grp.attrs.get('gaps_filled', 0)
                        zarr_meta['detect_source_options']['interpolated'] = f"{parent_name}/{latest_refined}/interpolated"
                    manual_latest = refined_group.attrs.get("manual_review_latest")
                    if manual_latest and manual_latest in refined_group:
                        zarr_meta['detect_source_options']['manual'] = f"{parent_name}/{latest_refined}/{manual_latest}"
                    elif "manual" in refined_group:
                        zarr_meta['detect_source_options']['manual'] = f"{parent_name}/{latest_refined}/manual"

                    resolution = resolve_detection_read_source(
                        root,
                        prefer_curated=True,
                        refined_run=str(latest_refined),
                        allow_sparse_fallback=True,
                    )
                    if resolution.detection_path:
                        label = "detect" if resolution.detection_kind == "raw" else str(
                            resolution.detection_kind or "detect"
                        ).lower()
                        zarr_meta['detect_source_default'] = {
                            "source_type": label,
                            "source_path": resolution.detection_path,
                        }

            if not zarr_meta['detect_source_default'] and 'detect' in zarr_meta['detect_source_options']:
                zarr_meta['detect_source_default'] = {
                    "source_type": "detect",
                    "source_path": zarr_meta['detect_source_options']['detect'],
                }
            
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
    """Display zarr metadata using shared training console renderer."""
    print_dataset_details(console, metadata, task="detect")


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
        options = dataset_meta.get("detect_source_options")
        options = options if isinstance(options, dict) else {}
        if requested_source in options:
            continue

        default_source = dataset_meta.get("detect_source_default")
        default_source = default_source if isinstance(default_source, dict) else {}
        available_source = _normalize_source_type(default_source.get("source_type"), default="")
        available_source_path = default_source.get("source_path")

        if not available_source and options:
            first_key = next(iter(options))
            available_source = _normalize_source_type(first_key, default="")
            available_source_path = options.get(first_key)

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
    return _shared_safe_sha256_file(path)


def _strip_manifest_suffixes(value: str) -> str:
    return _shared_strip_manifest_suffixes(value)


def _sanitize_run_component(value: str | None, fallback: str) -> str:
    return _shared_sanitize_run_component(value, fallback)


def _build_default_run_name(
    *,
    manifest_summary: dict,
    task_fallback: str,
    timestamp: str | None = None,
    pid: int | None = None,
) -> str:
    return _shared_build_default_detect_run_name(
        manifest_summary=manifest_summary,
        task_fallback=task_fallback,
        timestamp=timestamp,
        pid=pid,
    )


def _infer_set_slug(set_id: str | None, config_path: Path | None) -> str:
    return _shared_infer_set_slug(set_id, config_path, "detect_training")


def _resolve_project_dir(
    *,
    args,
    training_params: dict,
    set_id: str | None,
    config_path: Path | None,
    console: Console,
) -> None:
    _shared_resolve_project_dir(
        args=args,
        training_params=training_params,
        set_id=set_id,
        config_path=config_path,
        task_subdir="detect",
        default_slug="detect_training",
        console=console,
    )


def _snapshot_training_inputs(
    *,
    run_dir: Path,
    config_path: Path | None,
    manifest_path: Path | None,
    invocation_payload: dict | None,
) -> list[Path]:
    return _shared_snapshot_training_inputs(
        run_dir=run_dir,
        config_path=config_path,
        manifest_path=manifest_path,
        invocation_payload=invocation_payload,
    )


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
        export_artifacts=export_artifacts,
    )


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
    return _shared_run_subprocess_streaming(
        command=command,
        console=console,
        label=label,
        log_path=log_path,
    )


def _resolve_trtexec_path(explicit_path: str | None) -> Path | None:
    return _shared_resolve_trtexec_path(explicit_path)


def _collect_export_env(trtexec_path: Path | None, trtexec_log_path: Path | None = None) -> dict:
    return _shared_collect_export_env(trtexec_path, trtexec_log_path=trtexec_log_path)


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


def _collect_onnx_plugin_contract(onnx_path: Path) -> dict[str, Any]:
    """Collect ONNX custom/plugin op contract for deployment compatibility checks."""
    try:
        import onnx  # type: ignore
    except Exception:
        return {
            "requires_plugins": False,
            "plugin_ops": [],
            "plugin_versions": {},
        }

    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return {
            "requires_plugins": False,
            "plugin_ops": [],
            "plugin_versions": {},
        }

    standard_domains = {"", "ai.onnx", "ai.onnx.ml", "ai.onnx.training", "ai.onnx.preview.training"}
    plugin_ops: list[str] = []
    plugin_versions: dict[str, str] = {}

    for node in getattr(model.graph, "node", []) or []:
        domain = str(getattr(node, "domain", "") or "")
        op_type = str(getattr(node, "op_type", "") or "")
        if not op_type:
            continue
        if domain in standard_domains:
            continue

        op_key = f"{domain}::{op_type}" if domain else op_type
        plugin_ops.append(op_key)

        for attr in getattr(node, "attribute", []) or []:
            if str(getattr(attr, "name", "") or "") != "plugin_version":
                continue
            version_value = None
            attr_type = int(getattr(attr, "type", 0) or 0)
            if attr_type == 3:  # STRING
                try:
                    version_value = bytes(getattr(attr, "s", b"")).decode("utf-8")
                except Exception:
                    version_value = str(getattr(attr, "s", b""))
            elif attr_type == 2:  # INT
                version_value = str(int(getattr(attr, "i", 0)))

            if version_value:
                plugin_versions[op_key] = str(version_value)
            break

    plugin_ops_sorted = sorted(set(plugin_ops))
    return {
        "requires_plugins": bool(plugin_ops_sorted),
        "plugin_ops": plugin_ops_sorted,
        "plugin_versions": plugin_versions,
    }


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
    export_info["onnx_opset"] = args.onnx_opset
    export_info["nms_conf"] = float(args.nms_conf)
    export_info["nms_iou"] = float(args.nms_iou)
    export_info["nms_topk"] = int(args.nms_topk)
    export_info["input_shape"] = list(input_shape)
    export_info["imgsz"] = [int(img_h), int(img_w)]

    export_script = Path(__file__).resolve().parent / "export_onnx.py"
    onnx_cmd = [
        sys.executable,
        "-u",
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
    onnx_plugin_contract = _collect_onnx_plugin_contract(onnx_path)
    onnx_build_env = _collect_export_env(None, trtexec_log_path=None)
    export_info["weights_sha256"] = weights_sha
    export_info["onnx_sha256"] = onnx_sha
    export_info["onnx_source"] = "existing" if existing_onnx_path else "exported"
    export_info["onnx_output_contract"] = onnx_output_contract
    export_info["onnx_metadata_props"] = onnx_metadata_props
    export_info["onnx_requires_plugins"] = bool(onnx_plugin_contract.get("requires_plugins"))
    export_info["onnx_plugin_ops"] = onnx_plugin_contract.get("plugin_ops") or []
    export_info["onnx_plugin_versions"] = onnx_plugin_contract.get("plugin_versions") or {}
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
            "plugin_contract": onnx_plugin_contract,
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
    export_info["onnx_manifest_sha256"] = _safe_sha256_file(onnx_manifest_path)

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
        "-u",
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
                "plugin_contract": onnx_plugin_contract,
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
            "engine_precision": str(args.trt_precision),
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
    print_training_banner(console, "Detection")
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
    print_section_header(console, "Dataset Information")
    console.print("[dim]Analyzing Zarr files...[/dim]\n")
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
    shape_observer = TrainingShapeObserver()
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
            loader = YoloCompatibleDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=det_collate_fn,
                **loader_kwargs,
            )
            loader.shape_collector = shape_observer
            loader.shape_mode = mode
            return loader

        loader = YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=det_collate_fn,
            **loader_kwargs,
        )
        loader.shape_collector = shape_observer
        loader.shape_mode = mode
        return loader

    DetTrainer.get_dataloader = get_zarr_dataloader

    # Get training params
    training_params = full_config.training_params.model_dump(exclude_none=True)
    requested_training_imgsz = training_params.get("imgsz")
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
    print_training_hyperparameters(
        console,
        training_params=training_params,
        loader_overrides=custom_loader_augment,
        include_loader_note=True,
    )

    # Start training
    print_training_start(console)
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
    except KeyboardInterrupt as exc:
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
        console.print("[bold yellow]Training interrupted by user (Ctrl-C).[/bold yellow]")
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
                    "error_message": "training_interrupted_by_user",
                    "input_pipeline_profile": input_profile_payload,
                },
            )
        _ACTIVE_INPUT_PIPELINE_PROFILER = None
        DetTrainer.profile_collector = None
        YoloCompatibleDataLoader.profile_collector = None
        raise SystemExit(130)
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
    model_input_shape_accounting = _build_model_input_shape_accounting(
        requested_imgsz=requested_training_imgsz,
        rect=bool(training_params.get("rect", False)),
        ultralytics_train_imgsz=(effective_img_h, effective_img_w),
        zarr_metadata=zarr_metadata,
        shape_observer=shape_observer,
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
            'model_input_shape_accounting': model_input_shape_accounting,
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
        final_metrics_payload.setdefault("requested_imgsz", model_input_shape_accounting["requested_imgsz_hw"])
        final_metrics_payload.setdefault(
            "ultralytics_train_imgsz",
            model_input_shape_accounting["ultralytics_train_imgsz_hw"],
        )
        final_metrics_payload.setdefault(
            "model_input_shape_accounting",
            model_input_shape_accounting,
        )
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
        default=0.3,
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
