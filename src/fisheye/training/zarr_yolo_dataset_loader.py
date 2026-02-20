# zarr_yolo_dataset_loader.py

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import Dataset

from ..utils.zarr_metadata import get_downsample_array_path, get_downsample_formats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
class SamplingStrategy(Enum):
    """Sampling strategies for combining multiple datasets."""
    BALANCED = "balanced"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"

@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset."""
    zarr_path: str
    source_type: str = 'filtered'  # 'detect', 'filtered', 'interpolated', or 'manual'
    input_format: str = 'gray'  # 'gray' or 'rgb'
    split: Optional[Dict[str, float]] = None  # {'train': 0.8, 'val': 0.2}
    keypoint_run: Optional[str] = None  # Optional specific keypoints run


@dataclass
class DetectAugmentConfig:
    """Single-sample augmentations applied in train mode for detection."""

    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    fliplr: float = 0.0
    flipud: float = 0.0
    erasing: float = 0.0

    def uses_affine(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.degrees,
                self.translate,
                self.scale,
                self.shear,
                self.perspective,
            )
        )


@dataclass
class ZarrDatasetConfig:
    """Configuration for the Zarr dataset loader - supports both old and new formats."""
    # NEW: Support for per-dataset configuration
    datasets: Optional[Dict[str, SingleDatasetConfig]] = None
    
    # LEGACY: Old format (kept for backwards compatibility)
    zarr_paths: Optional[List[str]] = None
    
    # Common settings
    task: str = 'detect'
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED
    split_ratio: float = 0.8  # Used as default if datasets don't specify splits
    random_seed: int = 42
    dataset_weights: Optional[Dict[str, float]] = None
    allow_source_mismatch: bool = False
    target_size: Optional[int] = None
    min_confidence: float = 0.0
    filter_interpolated: bool = False  # DEPRECATED: Use source_type instead
    chunk_cache_size: int = 0
    augmentation: DetectAugmentConfig = field(default_factory=DetectAugmentConfig)

    def __post_init__(self):
        """Convert enum strings and validate configuration."""
        # Convert SamplingStrategy string to enum if needed
        if isinstance(self.sampling_strategy, str):
            try:
                self.sampling_strategy = SamplingStrategy(self.sampling_strategy)
            except ValueError:
                logger.warning(f"Unknown sampling strategy '{self.sampling_strategy}', defaulting to 'balanced'.")
                self.sampling_strategy = SamplingStrategy.BALANCED
        
        # Validate that we have either datasets or zarr_paths
        if self.datasets is None and self.zarr_paths is None:
            raise ValueError("Must provide either 'datasets' or 'zarr_paths'")
        
        # Convert old format to new format if needed
        if self.datasets is None and self.zarr_paths is not None:
            logger.info("Converting legacy zarr_paths format to new datasets format")
            self.datasets = {}
            for zarr_path in self.zarr_paths:
                dataset_name = Path(zarr_path).stem
                # Determine source_type based on filter_interpolated flag
                source_type = 'detect' if self.filter_interpolated else 'filtered'
                self.datasets[dataset_name] = SingleDatasetConfig(
                    zarr_path=zarr_path,
                    source_type=source_type,
                    split=None  # Use global split_ratio
                )
        
        # Convert dict datasets to SingleDatasetConfig objects if needed
        if self.datasets is not None:
            for name, config in self.datasets.items():
                if isinstance(config, dict):
                    self.datasets[name] = SingleDatasetConfig(**config)

        if isinstance(self.augmentation, dict):
            self.augmentation = DetectAugmentConfig(**self.augmentation)
        elif self.augmentation is None:
            self.augmentation = DetectAugmentConfig()
        self.chunk_cache_size = max(0, int(self.chunk_cache_size or 0))
    
    def get_zarr_paths(self) -> List[str]:
        """Get list of all zarr paths from datasets."""
        return [config.zarr_path for config in self.datasets.values()]

    def get_dataset_name(self, zarr_path: str) -> str:
        """Return configured dataset key for a zarr path."""
        for name, config in self.datasets.items():
            if config.zarr_path == zarr_path:
                return name
        return Path(zarr_path).stem
    
    def get_source_type(self, zarr_path: str) -> str:
        """Get source_type for a specific zarr path."""
        for config in self.datasets.values():
            if config.zarr_path == zarr_path:
                value = getattr(config, "source_type", "filtered")
                if hasattr(value, "value"):
                    value = value.value
                text = str(value).strip().lower()
                return text if text else "filtered"
        return 'filtered'  # Default

    def get_keypoint_run(self, zarr_path: str) -> Optional[str]:
        """Get configured keypoint run for a specific zarr path, if provided."""
        for config in self.datasets.values():
            if config.zarr_path == zarr_path:
                if isinstance(config, dict):
                    return config.get("keypoint_run")
                return getattr(config, "keypoint_run", None)
        return None

    def get_input_format(self, zarr_path: str) -> str:
        """Return the requested input format ('gray' or 'rgb') for this dataset."""
        for config in self.datasets.values():
            if config.zarr_path == zarr_path:
                value = getattr(config, "input_format", "gray")
                fmt = str(value).strip().lower()
                return "rgb" if fmt == "rgb" else "gray"
        return "gray"

    def get_split_config(self, zarr_path: str) -> Optional[Tuple[float, float]]:
        """Return per-dataset split ratios (train, val) for a zarr path, if configured."""
        for config in self.datasets.values():
            if config.zarr_path != zarr_path:
                continue

            split_cfg = getattr(config, "split", None)
            if split_cfg is None:
                return None

            if isinstance(split_cfg, dict):
                train_raw = split_cfg.get("train")
                val_raw = split_cfg.get("val")
            else:
                train_raw = getattr(split_cfg, "train", None)
                val_raw = getattr(split_cfg, "val", None)

            if train_raw is None or val_raw is None:
                raise ValueError(
                    f"Dataset '{self.get_dataset_name(zarr_path)}' split must define both 'train' and 'val'."
                )

            train_ratio = float(train_raw)
            val_ratio = float(val_raw)
            if train_ratio < 0.0 or val_ratio < 0.0:
                raise ValueError(
                    f"Dataset '{self.get_dataset_name(zarr_path)}' split ratios must be non-negative."
                )
            if not np.isclose(train_ratio + val_ratio, 1.0, atol=1e-6):
                raise ValueError(
                    f"Dataset '{self.get_dataset_name(zarr_path)}' split must sum to 1.0 "
                    f"(got train={train_ratio}, val={val_ratio})."
                )
            return train_ratio, val_ratio

        return None

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested data_config if present
        if 'data_config' in config_dict:
            config_dict = config_dict['data_config']
        
        strategy_str = config_dict.get('sampling_strategy', 'balanced')
        try:
            config_dict['sampling_strategy'] = SamplingStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Unknown sampling strategy '{strategy_str}', defaulting to 'balanced'.")
            config_dict['sampling_strategy'] = SamplingStrategy.BALANCED
            
        known_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}
        
        return cls(**filtered_dict)


# Core Dataset and Helper Classes

@dataclass
class DatasetMetadata:
    """Holds metadata extracted from a single Zarr file."""
    path: str
    name: str
    total_frames: int
    valid_frames: int
    column_names: List[str] = field(default_factory=list)
    tracking_success_rate: float = 0.0
    crop_source_type: str = 'unknown'
    has_interpolated: bool = False
    n_real_rois: int = 0
    n_interpolated_rois: int = 0
    requested_source_type: str = 'filtered'  # NEW: What user requested
    roi_shape: Tuple[int, int] = (0, 0)
    keypoint_run: Optional[str] = None
    requested_keypoint_run: Optional[str] = None
    bbox_array_path: str = ""
    frame_indices_path: Optional[str] = None
    detection_source_path: Optional[str] = None
    uses_crop_data: bool = True
    input_format: str = "gray"
    frame_array_path: str = "raw_video/images_ds"


class GlobalIndexManager:
    """Builds and manages a global index across all specified Zarr files."""
    def __init__(self, config: ZarrDatasetConfig):
        self.config = config
        self.metadata_list = self._validate_and_get_metadata()
        self.selected_indices_by_path: Dict[str, np.ndarray] = {}
        self.global_indices = self._build_global_index()

    @staticmethod
    def _resolve_latest_by_method(parent: zarr.Group, method: Optional[str]) -> Optional[str]:
        candidates: List[Tuple[datetime, str]] = []
        for run_name in parent.group_keys():
            run_group = parent[run_name]
            run_method = run_group.attrs.get('method')
            if method and run_method != method:
                continue
            ts_raw = run_group.attrs.get('keypoints_timestamp_utc') or run_group.attrs.get('timestamp_utc')
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.min
            except Exception:
                ts = datetime.min
            candidates.append((ts, run_name))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _normalize_requested_keypoint_run(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        val = value.strip().lower()
        if val in {"", "latest"}:
            return None
        if val in {"latest_traditional", "latest:traditional", "traditional"}:
            return "latest_traditional"
        if val in {"latest_yolo", "latest:yolo", "yolo"}:
            return "latest_yolo"
        return value

    def _validate_and_get_metadata(self) -> List[DatasetMetadata]:
        zarr_paths = self.config.get_zarr_paths()
        logger.info(f"Validating {len(zarr_paths)} Zarr files...")
        metadata_list = []
        
        for path_str in zarr_paths:
            try:
                root = zarr.open(path_str, mode='r')
                
                # Get requested source type and input format for this dataset
                requested_source = self.config.get_source_type(path_str)
                requested_input_format = self.config.get_input_format(path_str)
                if requested_input_format not in {"gray", "rgb"}:
                    requested_input_format = "gray"
                
                crop_parent = root.get('crop_runs')
                latest_crop = crop_parent.attrs.get('latest') if crop_parent is not None else None
                crop_group = None
                uses_crop_data = False
                if crop_parent is not None and latest_crop and latest_crop in crop_parent:
                    crop_group = crop_parent[latest_crop]
                    uses_crop_data = True

                detect_parent = root.get('detect_runs')
                frame_array_path: Optional[str] = None
                if self.config.task == 'detect':
                    frame_array_rel = get_downsample_array_path(root, format_hint=requested_input_format)
                    if frame_array_rel is None:
                        available_formats = get_downsample_formats(root)
                        raise KeyError(
                            f"Downsampled frames for format '{requested_input_format}' not found in {Path(path_str).name}. "
                            f"Available formats: {available_formats or 'none'}."
                        )
                    frame_array_path = frame_array_rel

                # Tracking setup
                column_names: List[str] = []
                total_frames = 0
                tracking_success_rate = 0.0
                requested_kp_run_raw = self.config.get_keypoint_run(path_str)
                requested_kp_run = self._normalize_requested_keypoint_run(requested_kp_run_raw)
                keypoint_run_name: Optional[str] = None
                success_arr: Optional[np.ndarray] = None

                roi_shape: Tuple[int, int] = (0, 0)
                bbox_array_path: Optional[str] = None
                frame_indices_path: Optional[str] = None
                detection_source_path: Optional[str] = None
                actual_source_type = requested_source
                has_interpolated = False
                n_real_rois = 0
                n_interpolated_rois = 0

                if uses_crop_data:
                    actual_source_type = crop_group.attrs.get('detection_source_type', 'detect')
                    has_interpolated = crop_group.attrs.get('includes_interpolated', False)
                    n_real_rois = crop_group.attrs.get('n_real_detections', crop_group['bbox_norm_coords'].shape[0])
                    n_interpolated_rois = crop_group.attrs.get('n_interpolated_detections', 0)
                    roi_shape = crop_group['roi_images'].shape[1:3] if 'roi_images' in crop_group else (0, 0)
                    total_frames = crop_group['roi_images'].shape[0] if 'roi_images' in crop_group else crop_group['bbox_norm_coords'].shape[0]
                    bbox_array_path = f"crop_runs/{latest_crop}/bbox_norm_coords"
                    frame_array_path = f"crop_runs/{latest_crop}/roi_images"
                    if 'frame_indices' in crop_group:
                        frame_indices_path = f"crop_runs/{latest_crop}/frame_indices"
                    detection_source_path = f"crop_runs/{latest_crop}/detection_source" if 'detection_source' in crop_group else None
                elif self.config.task == 'detect':
                    if detect_parent is None or 'latest' not in detect_parent.attrs:
                        raise KeyError(
                            f"Could not find 'crop_runs' in {Path(path_str).name} and no detect_runs available for fallback."
                        )
                    detect_run_name = detect_parent.attrs['latest']
                    if detect_run_name not in detect_parent:
                        raise KeyError(f"Detect run '{detect_run_name}' not found in {Path(path_str).name}.")
                    detect_group = detect_parent[detect_run_name]
                    actual_source_type = 'detect'
                    has_interpolated = False
                    n_interpolated_rois = 0
                    n_real_rois = int(detect_group['bbox_norm_coords'].shape[0])
                    roi_shape = (0, 0)
                    total_frames = n_real_rois
                    bbox_array_path = f"detect_runs/{detect_run_name}/bbox_norm_coords"
                    if 'frame_indices' in detect_group:
                        frame_indices_path = f"detect_runs/{detect_run_name}/frame_indices"
                    detection_source_path = None
                    logger.warning(
                        f"  ⚠ {Path(path_str).name}: crop_runs missing; training will read boxes directly from detect_runs/{detect_run_name}."
                    )
                else:
                    raise KeyError(f"Could not find 'crop_runs' in {Path(path_str).name}")

                # Warn if requested != available
                if requested_source != actual_source_type:
                    message = (
                        f"{Path(path_str).name}: requested source_type '{requested_source}' "
                        f"but available source is '{actual_source_type}'."
                    )
                    if self.config.allow_source_mismatch:
                        logger.warning(
                            f"  ⚠ {message} Using available '{actual_source_type}' data "
                            "(allow_source_mismatch=True)."
                        )
                    else:
                        raise ValueError(
                            f"{message} Re-run crop/curation to match source_type, "
                            "or set allow_source_mismatch=true (or pass --allow-source-mismatch)."
                        )

                if self.config.task == 'pose':
                    if 'keypoints_runs' not in root:
                        raise KeyError("Pose task requires 'keypoints_runs'.")

                    kp_parent = root['keypoints_runs']
                    if requested_kp_run in (None, 'latest'):
                        latest_attr = kp_parent.attrs.get('latest')
                        if not latest_attr:
                            raise KeyError(
                                "Keypoints run group missing 'latest' attribute; specify 'keypoint_run' in config."
                            )
                        keypoint_run_name = latest_attr
                    elif requested_kp_run == 'latest_traditional':
                        keypoint_run_name = self._resolve_latest_by_method(kp_parent, 'traditional_pose')
                        if keypoint_run_name is None:
                            raise KeyError(
                                f"No keypoint run with method 'traditional_pose' found in {Path(path_str).name}."
                            )
                    elif requested_kp_run == 'latest_yolo':
                        keypoint_run_name = self._resolve_latest_by_method(kp_parent, 'yolo_pose')
                        if keypoint_run_name is None:
                            raise KeyError(
                                f"No keypoint run with method 'yolo_pose' found in {Path(path_str).name}."
                            )
                    else:
                        if requested_kp_run not in kp_parent:
                            available = ', '.join(list(kp_parent.keys()))
                            raise KeyError(
                                f"Keypoint run '{requested_kp_run}' not found in {Path(path_str).name}. "
                                f"Available runs: {available}"
                            )
                        keypoint_run_name = requested_kp_run

                    kp_group = kp_parent[f'{keypoint_run_name}']
                    if 'keypoints_roi' not in kp_group:
                        raise KeyError(f"Keypoint run '{keypoint_run_name}' missing 'keypoints_roi' array.")

                    total_frames = kp_group['keypoints_roi'].shape[0]
                    column_names = list(kp_group.attrs.get('keypoint_labels', ['swim_bladder', 'eye_left', 'eye_right']))
                    success_arr = kp_group['detection_success'][:]
                    if total_frames > 0:
                        tracking_success_rate = float(np.mean(success_arr) * 100.0)

                if not bbox_array_path:
                    raise KeyError(f"Unable to determine bbox source for {Path(path_str).name}.")
                if not frame_array_path:
                    raise KeyError(f"Unable to determine frame image source for {Path(path_str).name}.")

                source_coords = root[bbox_array_path][:]
                valid_mask = np.zeros(source_coords.shape[0], dtype=bool) if source_coords.size == 0 else ~np.isnan(source_coords[:, 0])

                if requested_source in ['filtered', 'detect', 'manual'] and has_interpolated and detection_source_path:
                    detection_source = root[detection_source_path][:]
                    real_mask = (detection_source == 0)
                    valid_mask = valid_mask & real_mask

                if self.config.task == 'pose' and success_arr is not None:
                    kp_row_gate_applied = bool(kp_group.attrs.get("row_gate_applied", False)) if self.config.task == 'pose' else False
                    if kp_row_gate_applied and str(kp_group.attrs.get("method", "")).strip().lower() == "merged_export":
                        success_arr = None
                if self.config.task == 'pose' and success_arr is not None:
                    valid_mask = valid_mask & success_arr

                valid_frames = int(np.sum(valid_mask))

                metadata = DatasetMetadata(
                    path=path_str,
                    name=Path(path_str).stem,
                    total_frames=total_frames,
                    valid_frames=valid_frames,
                    column_names=column_names,
                    tracking_success_rate=tracking_success_rate,
                    crop_source_type=actual_source_type,
                    has_interpolated=has_interpolated,
                    n_real_rois=n_real_rois,
                    n_interpolated_rois=n_interpolated_rois,
                    requested_source_type=requested_source,
                    roi_shape=roi_shape,
                    keypoint_run=keypoint_run_name,
                    requested_keypoint_run=requested_kp_run,
                    bbox_array_path=bbox_array_path,
                    frame_indices_path=frame_indices_path,
                    detection_source_path=detection_source_path,
                    uses_crop_data=uses_crop_data,
                    input_format=requested_input_format,
                    frame_array_path=frame_array_path,
                )
                
                # Log crop source info
                if has_interpolated:
                    logger.info(f"  ✓ {Path(path_str).name}: {actual_source_type} source ({n_real_rois} real + {n_interpolated_rois} interpolated)")
                else:
                    logger.info(f"  ✓ {Path(path_str).name}: {actual_source_type} source (all real)")
                
                metadata_list.append(metadata)
                
            except Exception as e:
                raise IOError(f"Failed to process Zarr file at '{path_str}': {e}")
        
        logger.info("All Zarr files are compatible!")
        return metadata_list

    def _get_valid_indices(self, metadata: DatasetMetadata) -> np.ndarray:
        """Get valid frame indices, optionally filtering out interpolated data."""
        root = zarr.open(metadata.path, mode='r')
        
        if not metadata.bbox_array_path:
            raise KeyError(f"No bbox path recorded for dataset '{metadata.name}'.")

        coords = root[metadata.bbox_array_path][:]
        if coords.size == 0:
            return np.empty(0, dtype=int)
        valid_mask = ~np.isnan(coords[:, 0])
        
        # Filter out interpolated data if source_type is 'filtered', 'detect', or 'manual'
        if (
            metadata.requested_source_type in ['filtered', 'detect', 'manual']
            and metadata.has_interpolated
            and metadata.detection_source_path
        ):
            detection_source = root[metadata.detection_source_path][:]
            real_mask = (detection_source == 0)
            n_filtered = np.sum(~real_mask)
            valid_mask = valid_mask & real_mask
            logger.info(f"    Filtered out {n_filtered} interpolated ROIs from {metadata.name}")

        # For pose task, also check keypoints validity
        if self.config.task == 'pose':
            kp_run = metadata.keypoint_run or root['keypoints_runs'].attrs.get('latest')
            if kp_run is None:
                raise KeyError("Unable to determine keypoint run for pose dataset.")
            kp_group = root[f'keypoints_runs/{kp_run}']
            if bool(kp_group.attrs.get("row_gate_applied", False)) and str(
                kp_group.attrs.get("method", "")
            ).strip().lower() == "merged_export":
                return np.where(valid_mask)[0]
            if 'detection_success' not in kp_group:
                raise KeyError(f"Keypoint run '{kp_run}' missing 'detection_success' array.")
            success_mask = kp_group['detection_success'][:]
            valid_mask &= success_mask
        
        return np.where(valid_mask)[0]

    @staticmethod
    def _sample_without_replacement(indices: np.ndarray, sample_count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample deterministic subset without replacement."""
        if sample_count <= 0 or indices.size == 0:
            return np.empty(0, dtype=int)
        if sample_count >= indices.size:
            return indices.copy()
        return rng.choice(indices, size=sample_count, replace=False)

    def _resolve_weight_map(self) -> Dict[str, float]:
        """Resolve dataset weights keyed by zarr path."""
        raw_weights = self.config.dataset_weights or {}
        if not raw_weights:
            raise ValueError("dataset_weights is required when sampling_strategy='weighted'.")

        path_to_dataset_name = {
            metadata.path: self.config.get_dataset_name(metadata.path)
            for metadata in self.metadata_list
        }
        matched_keys = set()
        path_weights: Dict[str, float] = {}

        for metadata in self.metadata_list:
            dataset_name = path_to_dataset_name[metadata.path]
            candidates = [dataset_name, metadata.name, metadata.path]
            selected_key = next((key for key in candidates if key in raw_weights), None)
            if selected_key is None:
                raise ValueError(
                    "Missing dataset weight for "
                    f"'{dataset_name}' ({Path(metadata.path).name}). "
                    "Provide dataset_weights for all configured datasets."
                )
            matched_keys.add(selected_key)
            weight_value = float(raw_weights[selected_key])
            if weight_value <= 0:
                raise ValueError(
                    f"dataset_weights['{selected_key}'] must be > 0 (got {weight_value})."
                )
            path_weights[metadata.path] = weight_value

        unknown_keys = sorted(set(raw_weights.keys()) - matched_keys)
        if unknown_keys:
            raise ValueError(
                "dataset_weights has unknown keys that do not match configured datasets: "
                + ", ".join(unknown_keys)
            )

        return path_weights

    def _build_global_index(self) -> List[Tuple[str, int]]:
        """Build global index based on source_type and sampling strategy."""
        logger.info("Building global sample index...")
        
        # Check if any dataset is using filtered/detect source types
        any_filtering = any(m.requested_source_type in ['filtered', 'detect'] for m in self.metadata_list)
        if any_filtering:
            logger.info("  ℹ Using only real detections (filtering interpolated data)")
        
        all_valid_indices = {m.path: self._get_valid_indices(m) for m in self.metadata_list}
        rng = np.random.default_rng(self.config.random_seed)

        strategy = self.config.sampling_strategy
        selected_by_path: Dict[str, np.ndarray] = {path: np.empty(0, dtype=int) for path in all_valid_indices}
        path_weights_for_logging: Dict[str, float] = {}

        if strategy == SamplingStrategy.PROPORTIONAL:
            for path, indices in all_valid_indices.items():
                selected_by_path[path] = indices.copy()
            logger.info("  ✓ sampling_strategy=proportional (all valid samples).")

        elif strategy == SamplingStrategy.BALANCED:
            non_empty_counts = [indices.size for indices in all_valid_indices.values() if indices.size > 0]
            min_count = min(non_empty_counts) if non_empty_counts else 0
            for path, indices in all_valid_indices.items():
                selected_by_path[path] = self._sample_without_replacement(indices, min_count, rng)
            logger.info(
                "  ✓ sampling_strategy=balanced "
                f"(target {min_count} samples per non-empty dataset)."
            )

        elif strategy == SamplingStrategy.WEIGHTED:
            path_weights = self._resolve_weight_map()
            path_weights_for_logging = path_weights
            available_counts = {
                path: indices.size
                for path, indices in all_valid_indices.items()
                if indices.size > 0
            }
            if not available_counts:
                logger.warning("  ⚠ No valid samples available after filtering.")
                self.selected_indices_by_path = {
                    path: indices.copy() for path, indices in selected_by_path.items()
                }
                return []

            limiting_scale = min(
                available_counts[path] / path_weights[path] for path in available_counts
            )
            target_counts = {
                path: int(np.floor(path_weights[path] * limiting_scale))
                for path in available_counts
            }

            # Guard against all-zero rounding for highly skewed weights.
            if all(count == 0 for count in target_counts.values()):
                max_path = max(available_counts, key=lambda p: path_weights[p])
                target_counts[max_path] = 1

            for path, indices in all_valid_indices.items():
                target_count = target_counts.get(path, 0)
                selected_by_path[path] = self._sample_without_replacement(indices, target_count, rng)

            logger.info("  ✓ sampling_strategy=weighted (downsampled by dataset_weights).")

        else:
            raise ValueError(f"Unknown sampling strategy '{strategy}'.")

        global_indices = [
            (path, int(index))
            for path, indices in selected_by_path.items()
            for index in indices
        ]
        rng.shuffle(global_indices)
        self.selected_indices_by_path = {
            path: indices.copy() for path, indices in selected_by_path.items()
        }

        logger.info(f"Global index created with {len(global_indices)} total samples.")
        for metadata in self.metadata_list:
            available = len(all_valid_indices[metadata.path])
            selected = len(selected_by_path[metadata.path])
            if strategy == SamplingStrategy.WEIGHTED:
                weight = path_weights_for_logging[metadata.path]
                logger.info(
                    f"  {metadata.name}: {selected}/{available} samples (weight={weight:.4g})"
                )
            else:
                logger.info(f"  {metadata.name}: {selected}/{available} samples")

        return global_indices

    def _resolve_split_ratios(self, zarr_path: str) -> Tuple[float, float, str]:
        """Resolve train/val ratios for one dataset path."""
        per_dataset_split = self.config.get_split_config(zarr_path)
        if per_dataset_split is not None:
            return per_dataset_split[0], per_dataset_split[1], "dataset"

        train_ratio = float(self.config.split_ratio)
        val_ratio = 1.0 - train_ratio
        if train_ratio <= 0.0 or val_ratio <= 0.0:
            raise ValueError(f"split_ratio must be in (0, 1), got {self.config.split_ratio}.")
        return train_ratio, val_ratio, "global"

    @staticmethod
    def _compute_train_count(total_count: int, train_ratio: float, val_ratio: float) -> int:
        """Compute stable per-dataset train count with small-dataset safeguards."""
        if total_count <= 0:
            return 0
        if total_count == 1:
            return 1 if train_ratio >= val_ratio else 0

        train_count = int(round(total_count * train_ratio))
        if train_ratio > 0.0:
            train_count = max(1, train_count)
        if val_ratio > 0.0:
            train_count = min(total_count - 1, train_count)
        return int(np.clip(train_count, 0, total_count))

    def get_split_indices(self) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Apply per-dataset train/val splits, then merge into global train/val lists."""
        logger.info("Applying per-dataset train/val splits...")

        if not self.selected_indices_by_path:
            fallback: Dict[str, List[int]] = {m.path: [] for m in self.metadata_list}
            for path, det_idx in self.global_indices:
                fallback.setdefault(path, []).append(int(det_idx))
            self.selected_indices_by_path = {
                path: np.asarray(indices, dtype=int) for path, indices in fallback.items()
            }

        rng = np.random.default_rng(self.config.random_seed)
        train_indices: List[Tuple[str, int]] = []
        val_indices: List[Tuple[str, int]] = []

        for metadata in self.metadata_list:
            path = metadata.path
            selected = self.selected_indices_by_path.get(path, np.empty(0, dtype=int))
            total = int(selected.size)
            if total == 0:
                logger.info(f"  {metadata.name}: train=0 val=0 (no samples after sampling strategy)")
                continue

            train_ratio, val_ratio, split_source = self._resolve_split_ratios(path)
            shuffled = selected.copy()
            rng.shuffle(shuffled)

            train_count = self._compute_train_count(total, train_ratio, val_ratio)
            train_part = shuffled[:train_count]
            val_part = shuffled[train_count:]

            train_indices.extend((path, int(det_idx)) for det_idx in train_part)
            val_indices.extend((path, int(det_idx)) for det_idx in val_part)

            logger.info(
                f"  {metadata.name}: train={train_part.size} val={val_part.size} "
                f"(split={train_ratio:.3f}/{val_ratio:.3f}, source={split_source})"
            )

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        logger.info(
            f"Per-dataset split complete: train={len(train_indices)} samples, val={len(val_indices)} samples."
        )
        return train_indices, val_indices


# Rest of the code remains the same (ZarrYOLODataset and create_zarr_dataset)
class ZarrYOLODataset(Dataset):
    def __init__(self, config: ZarrDatasetConfig, mode: str = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode

        index_manager = GlobalIndexManager(config)
        train_indices, val_indices = index_manager.get_split_indices()
        self.indices = train_indices if mode == 'train' else val_indices
        
        self.metadata_map = {m.path: m for m in index_manager.metadata_list}
        if self.config.task == 'pose':
            first_metadata = index_manager.metadata_list[0] if index_manager.metadata_list else None
            labels = first_metadata.column_names if first_metadata and first_metadata.column_names else ['swim_bladder', 'eye_left', 'eye_right']
            self.keypoint_labels = labels
        else:
            self.keypoint_labels = []
        self.zarr_roots = {path: zarr.open(path, mode='r') for path in config.get_zarr_paths()}
        self.chunk_cache_size = max(0, int(self.config.chunk_cache_size or 0))
        self._chunk_cache_hits = 0
        self._chunk_cache_misses = 0

        self.target_size = self.config.target_size or (640 if self.config.task == 'detect' else 256)

        logger.info(f"Pre-caching labels for {self.mode} set ({self.config.task} task)...")
        self.labels = []
        
        # Cache metadata per task
        self.bbox_cache = {}
        self.frame_index_cache = {}
        self.kp_roi_norm_cache = {}
        self.kp_success_cache = {}
        self.kp_bbox_cache = {}
        self.kp_flat_cache = {}
        self.roi_size_cache = {}
        self.detect_frame_arrays = {}
        self.detect_frame_chunk_len = {}
        self.detect_frame_chunk_cache = OrderedDict()
        if self.config.task == 'detect':
            for zarr_path, root in self.zarr_roots.items():
                metadata = self.metadata_map.get(zarr_path)
                if metadata is None or not metadata.bbox_array_path:
                    raise KeyError(f"Missing metadata for detection dataset '{Path(zarr_path).name}'.")

                self.bbox_cache[zarr_path] = root[metadata.bbox_array_path][:]

                frame_indices = None
                if metadata.frame_indices_path:
                    try:
                        frame_indices = root[metadata.frame_indices_path][:]
                    except KeyError:
                        frame_indices = None

                if frame_indices is None:
                    detect_parent = root.get('detect_runs')
                    if detect_parent is not None and 'latest' in detect_parent.attrs:
                        detect_latest = detect_parent.attrs['latest']
                        frame_indices = root[f'detect_runs/{detect_latest}/frame_indices'][:]

                if frame_indices is None or frame_indices.shape[0] != self.bbox_cache[zarr_path].shape[0]:
                    frame_count = self.bbox_cache[zarr_path].shape[0]
                    if frame_indices is not None and frame_indices.shape[0] != frame_count:
                        logger.warning(
                            f"  ⚠ Frame index count mismatch for {Path(zarr_path).name}; falling back to sequential indices."
                        )
                    if frame_count == 0:
                        frame_indices = np.empty(0, dtype=np.int64)
                    else:
                        frame_indices = np.arange(frame_count, dtype=np.int64)

                self.frame_index_cache[zarr_path] = frame_indices
                frame_array = root[metadata.frame_array_path]
                self.detect_frame_arrays[zarr_path] = frame_array
                chunk_len = 1
                if frame_array.chunks and len(frame_array.chunks) > 0:
                    chunk_len = int(frame_array.chunks[0] or 1)
                self.detect_frame_chunk_len[zarr_path] = max(1, chunk_len)
                source_label = "crop_runs" if metadata.uses_crop_data else "detect_runs"
                logger.info(
                    f"  Cached {self.bbox_cache[zarr_path].shape[0]} bboxes from {Path(zarr_path).name} ({source_label})."
                )
            if self.chunk_cache_size > 0:
                logger.info(
                    f"  Enabled per-worker LRU chunk cache (chunk_cache_size={self.chunk_cache_size})."
                )
        else:
            for zarr_path in self.zarr_roots.keys():
                root = self.zarr_roots[zarr_path]
                latest_crop = root['crop_runs'].attrs['latest']
                crop_group = root[f'crop_runs/{latest_crop}']
                roi_shape = crop_group['roi_images'].shape[1:3]
                roi_h, roi_w = roi_shape
                self.roi_size_cache[zarr_path] = (roi_h, roi_w)

                metadata = self.metadata_map.get(zarr_path)
                kp_run_name = None
                if metadata is not None:
                    kp_run_name = metadata.keypoint_run
                if kp_run_name is None:
                    kp_run_name = root['keypoints_runs'].attrs.get('latest')
                if kp_run_name is None:
                    raise KeyError(f"No keypoint run available for {Path(zarr_path).name}; specify 'keypoint_run' in config.")

                kp_group = root[f'keypoints_runs/{kp_run_name}']
                kp_roi = kp_group['keypoints_roi'][:].astype(np.float32)
                kp_success = kp_group['detection_success'][:].astype(bool)

                finite_mask = np.isfinite(kp_roi).all(axis=(1, 2))
                valid_mask = kp_success & finite_mask
                self.kp_roi_norm_cache[zarr_path] = np.zeros_like(kp_roi, dtype=np.float32)

                if roi_w > 0 and roi_h > 0:
                    kp_roi_norm = kp_roi.copy()
                    kp_roi_norm[..., 0] /= float(roi_w)
                    kp_roi_norm[..., 1] /= float(roi_h)
                else:
                    kp_roi_norm = kp_roi

                kp_roi_norm = np.clip(kp_roi_norm, 0.0, 1.0)
                self.kp_roi_norm_cache[zarr_path] = kp_roi_norm
                self.kp_success_cache[zarr_path] = valid_mask

                # Compute bboxes only for valid rows to avoid all-NaN warnings
                kpts = kp_roi_norm.reshape(kp_roi_norm.shape[0], -1, 2)
                bboxes = np.zeros((kpts.shape[0], 4), dtype=np.float32)
                valid_idx = np.where(valid_mask)[0]
                if valid_idx.size > 0:
                    kpts_valid = kpts[valid_idx]
                    min_xy = np.nanmin(kpts_valid, axis=1)
                    max_xy = np.nanmax(kpts_valid, axis=1)
                    span = max_xy - min_xy
                    margin = span * 0.5
                    center = np.clip((min_xy + max_xy) / 2.0, 0.0, 1.0)
                    bbox_wh = np.clip(span + margin, 1e-6, 1.0)
                    bboxes_valid = np.concatenate([center, bbox_wh], axis=1).astype(np.float32)
                    bboxes[valid_idx] = bboxes_valid
                self.kp_bbox_cache[zarr_path] = bboxes

                visibility = np.full((kp_roi_norm.shape[0], kp_roi_norm.shape[1], 1), 2.0, dtype=np.float32)
                kpts_with_vis = np.concatenate([kp_roi_norm, visibility], axis=2).reshape(kp_roi_norm.shape[0], -1)
                self.kp_flat_cache[zarr_path] = kpts_with_vis
                logger.info(
                    f"  Cached {kp_roi_norm.shape[0]} keypoint entries from {Path(zarr_path).name} (run {kp_run_name})"
                )
        
        # Build labels using cached data
        label_fetcher = self._get_pose_data if self.config.task == 'pose' else self._get_bbox_data
        for zarr_path, idx in self.indices:
            self.labels.append(label_fetcher(zarr_path, idx))
        
        logger.info(f"Initialized '{mode}' dataset with {len(self.indices)} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def get_chunk_cache_stats(self) -> Dict[str, int]:
        """Expose per-worker cache stats for diagnostics/tests."""
        return {
            "chunk_cache_size": int(self.chunk_cache_size),
            "chunk_cache_hits": int(self._chunk_cache_hits),
            "chunk_cache_misses": int(self._chunk_cache_misses),
        }

    def _get_detect_frame(self, zarr_path: str, image_source_path: str, frame_idx: int) -> np.ndarray:
        frame_array = self.detect_frame_arrays.get(zarr_path)
        if frame_array is None:
            frame_array = self.zarr_roots[zarr_path][image_source_path]
            self.detect_frame_arrays[zarr_path] = frame_array
            chunk_len = 1
            if frame_array.chunks and len(frame_array.chunks) > 0:
                chunk_len = int(frame_array.chunks[0] or 1)
            self.detect_frame_chunk_len[zarr_path] = max(1, chunk_len)

        if self.chunk_cache_size <= 0:
            return frame_array[frame_idx]

        cache = self.detect_frame_chunk_cache
        chunk_len = self.detect_frame_chunk_len[zarr_path]
        chunk_start = (int(frame_idx) // chunk_len) * chunk_len
        cache_key = (zarr_path, chunk_start)
        chunk = cache.get(cache_key)

        if chunk is None:
            chunk_end = min(chunk_start + chunk_len, int(frame_array.shape[0]))
            chunk = np.asarray(frame_array[chunk_start:chunk_end])
            cache[cache_key] = chunk
            self._chunk_cache_misses += 1
            if len(cache) > self.chunk_cache_size:
                cache.popitem(last=False)
        else:
            cache.move_to_end(cache_key)
            self._chunk_cache_hits += 1

        local_idx = int(frame_idx) - chunk_start
        return chunk[local_idx]

    @staticmethod
    def _xywhn_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x, y, w, h = boxes.T
        x1 = (x - w / 2.0) * float(width)
        y1 = (y - h / 2.0) * float(height)
        x2 = (x + w / 2.0) * float(width)
        y2 = (y + h / 2.0) * float(height)
        return np.stack((x1, y1, x2, y2), axis=1).astype(np.float32)

    @staticmethod
    def _xyxy_to_xywhn(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = boxes.T
        cx = ((x1 + x2) / 2.0) / float(width)
        cy = ((y1 + y2) / 2.0) / float(height)
        bw = (x2 - x1) / float(width)
        bh = (y2 - y1) / float(height)
        out = np.stack((cx, cy, bw, bh), axis=1).astype(np.float32)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    @staticmethod
    def _apply_hsv_jitter(image: np.ndarray, aug: DetectAugmentConfig) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        if aug.hsv_h <= 0.0 and aug.hsv_s <= 0.0 and aug.hsv_v <= 0.0:
            return image

        dtype = image.dtype
        gains = np.random.uniform(-1.0, 1.0, 3) * np.array([aug.hsv_h, aug.hsv_s, aug.hsv_v], dtype=np.float32)
        x = np.arange(0, 256, dtype=np.float32)
        lut_h = ((x + gains[0] * 180.0) % 180.0).astype(np.uint8)
        lut_s = np.clip(x * (1.0 + gains[1]), 0, 255).astype(np.uint8)
        lut_v = np.clip(x * (1.0 + gains[2]), 0, 255).astype(np.uint8)
        lut_s[0] = 0

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        hsv_aug = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        out = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2RGB)
        return out.astype(dtype, copy=False)

    @staticmethod
    def _random_affine_matrix(image_h: int, image_w: int, aug: DetectAugmentConfig) -> np.ndarray:
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -image_w / 2.0
        center[1, 2] = -image_h / 2.0

        perspective = np.eye(3, dtype=np.float32)
        perspective[2, 0] = np.random.uniform(-aug.perspective, aug.perspective)
        perspective[2, 1] = np.random.uniform(-aug.perspective, aug.perspective)

        rotation = np.eye(3, dtype=np.float32)
        angle = np.random.uniform(-aug.degrees, aug.degrees)
        scale = np.random.uniform(1.0 - aug.scale, 1.0 + aug.scale)
        rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)

        shear = np.eye(3, dtype=np.float32)
        shear[0, 1] = np.tan(np.deg2rad(np.random.uniform(-aug.shear, aug.shear)))
        shear[1, 0] = np.tan(np.deg2rad(np.random.uniform(-aug.shear, aug.shear)))

        translate = np.eye(3, dtype=np.float32)
        translate[0, 2] = np.random.uniform(0.5 - aug.translate, 0.5 + aug.translate) * image_w
        translate[1, 2] = np.random.uniform(0.5 - aug.translate, 0.5 + aug.translate) * image_h

        return translate @ shear @ rotation @ perspective @ center

    @staticmethod
    def _transform_boxes_xyxy(
        boxes_xyxy: np.ndarray,
        matrix: np.ndarray,
        image_w: int,
        image_h: int,
        use_perspective: bool,
    ) -> np.ndarray:
        n = int(boxes_xyxy.shape[0])
        if n == 0:
            return boxes_xyxy

        corners = np.ones((n * 4, 3), dtype=np.float32)
        corners[:, :2] = boxes_xyxy[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        warped = corners @ matrix.T
        if use_perspective:
            warped = warped[:, :2] / warped[:, 2:3]
        else:
            warped = warped[:, :2]
        warped = warped.reshape(n, 8)
        x = warped[:, [0, 2, 4, 6]]
        y = warped[:, [1, 3, 5, 7]]
        out = np.stack((x.min(1), y.min(1), x.max(1), y.max(1)), axis=1).astype(np.float32)
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, float(image_w))
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, float(image_h))
        return out

    @staticmethod
    def _box_candidates(original_xyxy: np.ndarray, warped_xyxy: np.ndarray) -> np.ndarray:
        if original_xyxy.size == 0:
            return np.zeros((0,), dtype=bool)
        w1 = original_xyxy[:, 2] - original_xyxy[:, 0]
        h1 = original_xyxy[:, 3] - original_xyxy[:, 1]
        w2 = warped_xyxy[:, 2] - warped_xyxy[:, 0]
        h2 = warped_xyxy[:, 3] - warped_xyxy[:, 1]
        aspect = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > 2.0) & (h2 > 2.0) & ((w2 * h2) / (w1 * h1 + 1e-16) > 0.10) & (aspect < 100.0)

    @staticmethod
    def _apply_random_erasing(image: np.ndarray, prob: float) -> np.ndarray:
        if prob <= 0.0 or np.random.random() >= prob:
            return image
        h, w = image.shape[:2]
        image_area = h * w
        for _ in range(10):
            target_area = np.random.uniform(0.02, 0.2) * image_area
            aspect = np.random.uniform(0.3, 3.3)
            erase_h = int(round(np.sqrt(target_area / aspect)))
            erase_w = int(round(np.sqrt(target_area * aspect)))
            if erase_h <= 0 or erase_w <= 0 or erase_h >= h or erase_w >= w:
                continue
            y0 = np.random.randint(0, h - erase_h + 1)
            x0 = np.random.randint(0, w - erase_w + 1)
            if image.ndim == 3:
                image[y0:y0 + erase_h, x0:x0 + erase_w, :] = 114
            else:
                image[y0:y0 + erase_h, x0:x0 + erase_w] = 114
            break
        return image

    def _augment_detect_train_sample(
        self,
        image: np.ndarray,
        cls: np.ndarray,
        bboxes_xywhn: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aug = self.config.augmentation
        if aug is None:
            return image, cls, bboxes_xywhn

        out_img = np.ascontiguousarray(image.copy())
        out_cls = cls.astype(np.float32, copy=True)
        out_boxes = bboxes_xywhn.astype(np.float32, copy=True)
        image_h, image_w = out_img.shape[:2]

        if aug.uses_affine():
            matrix = self._random_affine_matrix(image_h, image_w, aug)
            use_perspective = aug.perspective > 0.0
            if use_perspective:
                out_img = cv2.warpPerspective(out_img, matrix, dsize=(image_w, image_h), borderValue=(114, 114, 114))
            else:
                out_img = cv2.warpAffine(out_img, matrix[:2], dsize=(image_w, image_h), borderValue=(114, 114, 114))
                if out_img.ndim == 2:
                    out_img = out_img[..., None]

            boxes_xyxy = self._xywhn_to_xyxy(out_boxes, image_w, image_h)
            warped_xyxy = self._transform_boxes_xyxy(boxes_xyxy, matrix, image_w, image_h, use_perspective)
            keep = self._box_candidates(boxes_xyxy, warped_xyxy)
            out_cls = out_cls[keep]
            warped_xyxy = warped_xyxy[keep]
            out_boxes = self._xyxy_to_xywhn(warped_xyxy, image_w, image_h)

        if aug.fliplr > 0.0 and np.random.random() < aug.fliplr:
            out_img = np.ascontiguousarray(np.fliplr(out_img))
            if out_boxes.size > 0:
                out_boxes[:, 0] = 1.0 - out_boxes[:, 0]

        if aug.flipud > 0.0 and np.random.random() < aug.flipud:
            out_img = np.ascontiguousarray(np.flipud(out_img))
            if out_boxes.size > 0:
                out_boxes[:, 1] = 1.0 - out_boxes[:, 1]

        out_img = self._apply_hsv_jitter(out_img, aug)
        out_img = self._apply_random_erasing(out_img, aug.erasing)

        return out_img, out_cls, out_boxes

    def _get_bbox_data(self, zarr_path: str, det_idx: int) -> Dict:
        bbox_coords = self.bbox_cache.get(zarr_path)
        
        if bbox_coords is not None and det_idx < bbox_coords.shape[0]:
            bbox = bbox_coords[det_idx]
            bbox_x, bbox_y, bbox_w, bbox_h = bbox
            
            if not any(np.isnan([bbox_x, bbox_y, bbox_w, bbox_h])):
                return {
                    "cls": np.array([0]), 
                    "bboxes": np.array([[bbox_x, bbox_y, bbox_w, bbox_h]])
                }
        
        return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}

    def _get_pose_data(self, zarr_path: str, det_idx: int) -> Dict:
        """Get pose data using pre-cached keypoints."""
        try:
            # Use cached data instead of recalculating
            if zarr_path not in self.kp_success_cache:
                return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}
            
            # Check if this detection has valid keypoints
            if det_idx >= len(self.kp_success_cache[zarr_path]):
                return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}
            
            if not self.kp_success_cache[zarr_path][det_idx]:
                return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}
            
            # Get cached bbox and keypoints
            bbox = self.kp_bbox_cache[zarr_path][det_idx]
            kpts_flat = self.kp_flat_cache[zarr_path][det_idx]
            
            # Verify data is valid
            if np.isnan(bbox).any() or np.isnan(kpts_flat).any():
                return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}
            
            return {
                "cls": np.array([0]),
                "bboxes": bbox.reshape(1, 4).astype(np.float32),
                "keypoints": kpts_flat.astype(np.float32)  # Already shape (9,)
            }
        except (KeyError, IndexError) as e:
            return {"cls": np.zeros((0,), dtype=np.float32), "bboxes": np.zeros((0, 4), dtype=np.float32)}
    
    def __getitem__(self, index: int) -> Dict:
        profile_cb = getattr(self, "_profile_callback", None)
        profile_enabled = callable(profile_cb)
        getitem_start = time.perf_counter() if profile_enabled else 0.0
        read_seconds = 0.0
        augment_seconds = 0.0

        zarr_path, det_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        metadata = self.metadata_map.get(zarr_path)
        if metadata is None:
            raise KeyError(f"Metadata missing for dataset '{Path(zarr_path).name}'.")
        
        if self.config.task == 'detect':
            image_source_path = metadata.frame_array_path
        else:
            image_source_path = f"crop_runs/{root['crop_runs'].attrs['latest']}/roi_images"
        
        frame_idx = None
        read_start = time.perf_counter() if profile_enabled else 0.0
        if self.config.task == 'detect':
            frame_array = self.detect_frame_arrays.get(zarr_path)
            if frame_array is None:
                frame_array = root[image_source_path]
                self.detect_frame_arrays[zarr_path] = frame_array
            frame_indices = self.frame_index_cache[zarr_path]
            frame_idx = int(frame_indices[det_idx]) if det_idx < len(frame_indices) else None
            if frame_idx is None or frame_idx >= frame_array.shape[0]:
                # fallback: skip label/image mismatch
                frame_idx = 0
                image = np.zeros_like(frame_array[0])
            else:
                image = self._get_detect_frame(zarr_path, image_source_path, frame_idx)
        else:
            roi_idx = det_idx
            if roi_idx >= root[image_source_path].shape[0]:
                image = np.zeros_like(root[image_source_path][0])
                roi_idx = 0
            else:
                image = root[image_source_path][roi_idx]
        if profile_enabled:
            read_seconds = max(0.0, time.perf_counter() - read_start)

        if self.config.task == 'detect' and metadata.input_format == 'rgb' and image.ndim == 3:
            image_3ch = image
        else:
            if image.ndim == 2:
                image_3ch = np.stack([image] * 3, axis=-1)
            else:
                image_3ch = image
        
        label_info = self.labels[index]
        cls_arr = label_info.get('cls', np.zeros((0,), dtype=np.float32)).astype(np.float32)
        bboxes_arr = label_info.get('bboxes', np.zeros((0, 4), dtype=np.float32)).astype(np.float32)
        if self.config.task == 'detect' and self.mode == 'train':
            augment_start = time.perf_counter() if profile_enabled else 0.0
            image_3ch, cls_arr, bboxes_arr = self._augment_detect_train_sample(
                image=image_3ch,
                cls=cls_arr,
                bboxes_xywhn=bboxes_arr,
            )
            if profile_enabled:
                augment_seconds = max(0.0, time.perf_counter() - augment_start)

        ori_shape = (image_3ch.shape[0], image_3ch.shape[1])
        
        im_identifier = (f"{Path(zarr_path).stem}_frame_{frame_idx}"
                         if self.config.task == 'detect'
                         else f"{Path(zarr_path).stem}_roi_{det_idx}")

        sample = {
            "img": image_3ch.transpose(2, 0, 1),
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "keypoints": label_info.get('keypoints', np.zeros((0, 9), dtype=np.float32)).astype(np.float32),
            "im_file": im_identifier,
            "ori_shape": ori_shape,
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "segments": np.zeros((0, 0), dtype=np.float32)
        }

        if profile_enabled:
            try:
                profile_cb(
                    {
                        "samples": 1,
                        "zarr_read_s": float(read_seconds),
                        "augment_preprocess_s": float(augment_seconds),
                        "getitem_total_s": float(max(0.0, time.perf_counter() - getitem_start)),
                    }
                )
            except Exception:
                # Profiling must never disrupt training.
                pass

        return sample


def create_zarr_dataset(config: Union[ZarrDatasetConfig, Dict], mode: str) -> ZarrYOLODataset:
    """Factory function to create a ZarrYOLODataset."""
    if isinstance(config, dict):
        config = ZarrDatasetConfig(**config)
    return ZarrYOLODataset(config, mode)
