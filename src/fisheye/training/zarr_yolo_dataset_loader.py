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

from ..pose.schema import resolve_keypoint_labels_from_attrs
from ..shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)
from ..shared.refined_detect_curation import (
    REFINED_DETECT_STATUS_CODE_MAP,
    build_curated_detection_source_array,
    has_curated_refined_detect_surface,
    has_sparse_curated_refined_detect_instances_arrays,
)
from ..shared.zarr_run_completion import resolve_authoritative_run_name
from ..shared.zarr_metadata import get_downsample_array_path, get_downsample_formats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FRAME_INDEX_OOB_DROP_RATE_LIMIT = 0.01

def _latest_complete(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    return resolve_authoritative_run_name(parent)


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
    source_type: str = 'refined'  # 'detect', 'refined', 'filtered', 'interpolated', or 'manual'
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
                source_type = 'detect' if self.filter_interpolated else 'refined'
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
                value = getattr(config, "source_type", "refined")
                if hasattr(value, "value"):
                    value = value.value
                text = str(value).strip().lower()
                return text if text else "refined"
        return 'refined'  # Default

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
    requested_source_type: str = 'refined'  # NEW: What user requested
    roi_shape: Tuple[int, int] = (0, 0)
    keypoint_run: Optional[str] = None
    requested_keypoint_run: Optional[str] = None
    bbox_array_path: str = ""
    frame_indices_path: Optional[str] = None
    detection_source_path: Optional[str] = None
    status_codes_path: Optional[str] = None
    artifact_train_indices_path: Optional[str] = None
    artifact_val_indices_path: Optional[str] = None
    uses_crop_data: bool = True
    input_format: str = "gray"
    frame_array_path: str = "raw_video/images_ds"


class GlobalIndexManager:
    """Builds and manages a global index across all specified Zarr files."""
    def __init__(self, config: ZarrDatasetConfig):
        self.config = config
        self.metadata_list = self._validate_and_get_metadata()
        self.selected_indices_by_path: Dict[str, np.ndarray] = {}
        self.frame_index_oob_drops: Dict[str, Dict[str, Any]] = {}
        self._frame_index_in_bounds_mask_by_path: Dict[str, np.ndarray] = {}
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

    @staticmethod
    def _resolve_training_split_paths(root: zarr.Group) -> Tuple[Optional[str], Optional[str]]:
        zarr_purpose = str(root.attrs.get("zarr_purpose") or "").strip().lower()
        if zarr_purpose != "training":
            return None, None
        split_group = root.get("splits")
        if split_group is None or not hasattr(split_group, "__contains__"):
            return None, None
        if "train_indices" in split_group and "val_indices" in split_group:
            return "splits/train_indices", "splits/val_indices"
        return None, None

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
                latest_crop = _latest_complete(crop_parent)
                crop_group = None
                uses_crop_data = False
                if crop_parent is not None and latest_crop and latest_crop in crop_parent:
                    crop_group = crop_parent[latest_crop]
                    uses_crop_data = True

                detect_parent = root.get('detect_runs')
                frame_array_path: Optional[str] = None
                if self.config.task == 'detect':
                    frame_array_rel = get_downsample_array_path(root, format_hint=requested_input_format)
                    if frame_array_rel is not None:
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
                status_codes_path: Optional[str] = None
                artifact_train_indices_path: Optional[str] = None
                artifact_val_indices_path: Optional[str] = None
                actual_source_type = requested_source
                has_interpolated = False
                n_real_rois = 0
                n_interpolated_rois = 0
                artifact_train_indices_path, artifact_val_indices_path = self._resolve_training_split_paths(root)

                if self.config.task == 'detect':
                    refined_parent_name = None
                    refined_parent = None
                    if "refined_detect_runs" in root:
                        refined_parent_name = "refined_detect_runs"
                        refined_parent = root["refined_detect_runs"]
                    elif "refined_runs" in root:
                        refined_parent_name = "refined_runs"
                        refined_parent = root["refined_runs"]

                    selected_source_type: Optional[str] = None
                    selected_group_path: Optional[str] = None
                    selected_group: Optional[zarr.Group] = None
                    selected_detect_run: Optional[str] = None

                    if requested_source == "detect":
                        detect_run_name = _latest_complete(detect_parent)
                        if detect_parent is not None and detect_run_name and detect_run_name in detect_parent:
                            selected_source_type = "detect"
                            selected_group_path = f"detect_runs/{detect_run_name}"
                            selected_group = detect_parent[detect_run_name]
                            selected_detect_run = str(detect_run_name)
                    else:
                        refined_run_name = _latest_complete(refined_parent)
                        if refined_parent is not None and refined_run_name and refined_run_name in refined_parent:
                            refined_run = refined_parent[refined_run_name]
                            preferred_group: Optional[str] = None
                            if requested_source == "refined" and has_curated_refined_detect_surface(refined_run):
                                selected_source_type = "refined"
                                if has_sparse_curated_refined_detect_instances_arrays(refined_run):
                                    selected_group_path = f"{refined_parent_name}/{refined_run_name}/instances"
                                    selected_group = refined_run["instances"]
                                else:
                                    selected_group_path = f"{refined_parent_name}/{refined_run_name}"
                                    selected_group = refined_run
                                source_detect_attr = refined_run.attrs.get("source_detect_run")
                                selected_detect_run = str(source_detect_attr) if source_detect_attr else None
                            elif requested_source == "manual":
                                manual_latest = refined_run.attrs.get("manual_review_latest")
                                if manual_latest and manual_latest in refined_run:
                                    preferred_group = str(manual_latest)
                                elif "manual" in refined_run:
                                    preferred_group = "manual"
                            elif requested_source in {"filtered", "interpolated"} and requested_source in refined_run:
                                preferred_group = requested_source

                            if preferred_group is not None:
                                selected_source_type = requested_source
                                selected_group_path = (
                                    f"{refined_parent_name}/{refined_run_name}/{preferred_group}"
                                )
                                selected_group = refined_run[preferred_group]
                                source_detect_attr = refined_run.attrs.get("source_detect_run")
                                selected_detect_run = str(source_detect_attr) if source_detect_attr else None
                            else:
                                resolved = resolve_refined_detect_group(
                                    refined_run,
                                    preference=DEFAULT_DETECT_GROUP_PREFERENCE,
                                )
                                resolved_group = resolved.group
                                if resolved_group and resolved_group in refined_run:
                                    selected_source_type = str(
                                        (resolved.label or resolved_group)
                                    ).lower()
                                    selected_group_path = (
                                        f"{refined_parent_name}/{refined_run_name}/{resolved_group}"
                                    )
                                    selected_group = refined_run[resolved_group]
                                    selected_detect_run = (
                                        str(resolved.source_detect_run)
                                        if resolved.source_detect_run
                                        else None
                                    )
                                elif resolved.label == "raw":
                                    detect_run_name = (
                                        resolved.source_detect_run
                                        or _latest_complete(detect_parent)
                                    )
                                    if detect_parent is not None and detect_run_name and detect_run_name in detect_parent:
                                        selected_source_type = "detect"
                                        selected_group_path = f"detect_runs/{detect_run_name}"
                                        selected_group = detect_parent[detect_run_name]
                                        selected_detect_run = str(detect_run_name)

                    # Legacy fallback for merged/training artifacts that only carry crop labels.
                    if selected_group is None and uses_crop_data:
                        selected_source_type = str(
                            crop_group.attrs.get("detection_source_type", "detect")
                        ).strip().lower()
                        selected_group_path = f"crop_runs/{latest_crop}"
                        selected_group = crop_group
                        if frame_array_path is None and "roi_images" in crop_group:
                            frame_array_path = f"crop_runs/{latest_crop}/roi_images"
                        logger.warning(
                            "  ⚠ %s: falling back to crop_runs/%s for detect labels (legacy source resolution).",
                            Path(path_str).name,
                            latest_crop,
                        )

                    if selected_group is None or selected_group_path is None:
                        available_formats = get_downsample_formats(root)
                        raise KeyError(
                            f"{Path(path_str).name}: unable to resolve detect label source "
                            f"for requested source_type '{requested_source}'. "
                            f"Available downsample formats: {available_formats or 'none'}."
                        )

                    actual_source_type = selected_source_type or requested_source
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
                                f"{message} Re-run refine/review to produce requested source, "
                                "or set allow_source_mismatch=true (or pass --allow-source-mismatch)."
                            )

                    bbox_array_path = f"{selected_group_path}/bbox_norm_coords"
                    status_codes_path = None
                    if "frame_indices" in selected_group:
                        frame_indices_path = f"{selected_group_path}/frame_indices"
                    if has_curated_refined_detect_surface(selected_group):
                        status_codes_path = f"{selected_group_path}/status_codes"
                    if "detection_source" in selected_group:
                        detection_source_path = f"{selected_group_path}/detection_source"

                    if frame_array_path is None and uses_crop_data and "roi_images" in crop_group:
                        frame_array_path = f"crop_runs/{latest_crop}/roi_images"
                    if frame_array_path is None:
                        available_formats = get_downsample_formats(root)
                        raise KeyError(
                            f"Downsampled frames for format '{requested_input_format}' not found in {Path(path_str).name}. "
                            f"Available formats: {available_formats or 'none'}."
                        )

                    if has_curated_refined_detect_surface(selected_group):
                        n_total = int(
                            np.sum(
                                np.asarray(selected_group["status_codes"][:], dtype=np.int8).reshape(-1)
                                == REFINED_DETECT_STATUS_CODE_MAP["present"]
                            )
                        )
                        detection_source = build_curated_detection_source_array(selected_group, present_only=True)
                        n_interpolated_rois = int(np.count_nonzero(detection_source != 0))
                        n_real_rois = int(max(0, detection_source.shape[0] - n_interpolated_rois))
                    elif (
                        actual_source_type == "refined"
                        and "bbox_norm_coords" in selected_group
                        and "source_kind_codes" in selected_group
                    ):
                        n_total = int(selected_group["bbox_norm_coords"].shape[0])
                        source_kind_codes = np.asarray(selected_group["source_kind_codes"][:], dtype=np.int8)
                        n_interpolated_rois = int(
                            np.count_nonzero(source_kind_codes == 2)
                        )
                        n_real_rois = int(max(0, n_total - n_interpolated_rois))
                    elif detection_source_path and detection_source_path in root:
                        n_total = int(selected_group["bbox_norm_coords"].shape[0])
                        detection_source = np.asarray(root[detection_source_path][:], dtype=np.int8)
                        n_interpolated_rois = int(np.count_nonzero(detection_source != 0))
                        n_real_rois = int(max(0, detection_source.shape[0] - n_interpolated_rois))
                    else:
                        n_total = int(selected_group["bbox_norm_coords"].shape[0])
                        n_interpolated_rois = 0
                        n_real_rois = n_total

                    has_interpolated = bool(
                        actual_source_type == "interpolated" or n_interpolated_rois > 0
                    )
                    roi_shape = (0, 0)
                    total_frames = n_total
                elif uses_crop_data:
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
                else:
                    raise KeyError(f"Could not find 'crop_runs' in {Path(path_str).name}")

                # Warn if requested != available
                if self.config.task != "detect" and requested_source != actual_source_type:
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
                        latest_attr = _latest_complete(kp_parent)
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
                    column_names = list(
                        resolve_keypoint_labels_from_attrs(
                            dict(kp_group.attrs),
                            keypoint_count=int(kp_group['keypoints_roi'].shape[1]),
                        )
                    )
                    if not column_names:
                        raise KeyError(
                            f"Keypoint run '{keypoint_run_name}' in {Path(path_str).name} is missing "
                            "keypoint label metadata (expected keypoint_labels or pose_schema nodes)."
                        )
                    success_arr = kp_group['detection_success'][:]
                    if total_frames > 0:
                        tracking_success_rate = float(np.mean(success_arr) * 100.0)

                if not bbox_array_path:
                    raise KeyError(f"Unable to determine bbox source for {Path(path_str).name}.")
                if not frame_array_path:
                    raise KeyError(f"Unable to determine frame image source for {Path(path_str).name}.")

                source_coords = root[bbox_array_path][:]
                valid_mask = np.zeros(source_coords.shape[0], dtype=bool) if source_coords.size == 0 else ~np.isnan(source_coords[:, 0])
                if status_codes_path and status_codes_path in root:
                    status_codes = np.asarray(root[status_codes_path][:], dtype=np.int8)
                    valid_mask = valid_mask & (status_codes == REFINED_DETECT_STATUS_CODE_MAP["present"])

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
                        status_codes_path=status_codes_path,
                        artifact_train_indices_path=artifact_train_indices_path,
                        artifact_val_indices_path=artifact_val_indices_path,
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


    def _resolve_detection_frame_indices(
        self,
        root: zarr.Group,
        metadata: DatasetMetadata,
        row_count: int,
    ) -> np.ndarray:
        frame_indices = None
        if metadata.frame_indices_path:
            try:
                frame_indices = np.asarray(root[metadata.frame_indices_path][:])
            except KeyError:
                frame_indices = None

        if frame_indices is None:
            detect_parent = root.get('detect_runs')
            detect_latest = _latest_complete(detect_parent) if detect_parent is not None else None
            if detect_latest:
                frame_indices = np.asarray(root[f'detect_runs/{detect_latest}/frame_indices'][:])

        if frame_indices is None:
            return np.arange(row_count, dtype=np.int64)

        frame_indices = np.asarray(frame_indices)
        if frame_indices.ndim != 1:
            raise ValueError(
                f"{metadata.name}: frame_indices must be 1D for training sample mapping; "
                f"got shape {frame_indices.shape}."
            )
        if frame_indices.shape[0] != row_count:
            raise ValueError(
                f"{metadata.name}: frame_indices length mismatch for training sample mapping "
                f"({frame_indices.shape[0]} frame indices for {row_count} label rows)."
            )
        if not np.issubdtype(frame_indices.dtype, np.integer):
            raise ValueError(
                f"{metadata.name}: frame_indices must be integer dtype for training sample mapping; "
                f"got {frame_indices.dtype}."
            )
        return frame_indices.astype(np.int64, copy=False)

    def _get_frame_index_in_bounds_mask(
        self,
        root: zarr.Group,
        metadata: DatasetMetadata,
        row_count: int,
    ) -> np.ndarray:
        cached = self._frame_index_in_bounds_mask_by_path.get(metadata.path)
        if cached is not None:
            return cached

        if self.config.task != 'detect':
            mask = np.ones(row_count, dtype=bool)
            self._frame_index_in_bounds_mask_by_path[metadata.path] = mask
            return mask

        if not metadata.frame_array_path:
            raise KeyError(f"{metadata.name}: no frame array path recorded for detection dataset.")
        frame_array = root[metadata.frame_array_path]
        frame_count = int(frame_array.shape[0])
        frame_indices = self._resolve_detection_frame_indices(root, metadata, row_count)
        bad_mask = (frame_indices < 0) | (frame_indices >= frame_count)
        mask = ~bad_mask

        bad_count = int(np.sum(bad_mask))
        if bad_count:
            bad_rows = np.flatnonzero(bad_mask)
            first_rows = [int(v) for v in bad_rows[:10]]
            first_frames = [int(v) for v in frame_indices[bad_rows[:10]]]
            drop_rate = bad_count / float(max(1, row_count))
            detail = {
                "dropped": bad_count,
                "rows": int(row_count),
                "drop_rate": float(drop_rate),
                "frame_count": int(frame_count),
                "first_bad_rows": first_rows,
                "first_bad_frame_indices": first_frames,
            }
            self.frame_index_oob_drops[metadata.path] = detail
            logger.warning(
                "%s: dropped %d/%d training rows with frame index out of bounds "
                "for %s frames (first rows=%s, frame_indices=%s).",
                metadata.name,
                bad_count,
                row_count,
                frame_count,
                first_rows,
                first_frames,
            )
            if drop_rate > FRAME_INDEX_OOB_DROP_RATE_LIMIT:
                raise ValueError(
                    f"{metadata.name}: frame-index out-of-bounds drop rate {drop_rate:.3%} "
                    f"exceeds {FRAME_INDEX_OOB_DROP_RATE_LIMIT:.1%} "
                    f"({bad_count}/{row_count} rows; frame_count={frame_count}; "
                    f"first_bad_rows={first_rows}; first_bad_frame_indices={first_frames})."
                )

        self._frame_index_in_bounds_mask_by_path[metadata.path] = mask
        return mask

    def _get_valid_indices(self, metadata: DatasetMetadata) -> np.ndarray:
        """Get valid frame indices, optionally filtering out interpolated data."""
        root = zarr.open(metadata.path, mode='r')
        
        if not metadata.bbox_array_path:
            raise KeyError(f"No bbox path recorded for dataset '{metadata.name}'.")

        coords = root[metadata.bbox_array_path][:]
        if coords.size == 0:
            return np.empty(0, dtype=int)
        valid_mask = ~np.isnan(coords[:, 0])
        if metadata.status_codes_path and metadata.status_codes_path in root:
            status_codes = np.asarray(root[metadata.status_codes_path][:], dtype=np.int8)
            valid_mask = valid_mask & (status_codes == REFINED_DETECT_STATUS_CODE_MAP["present"])
        if self.config.task == 'detect':
            frame_mask = self._get_frame_index_in_bounds_mask(root, metadata, coords.shape[0])
            valid_mask = valid_mask & frame_mask
        
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
            kp_run = metadata.keypoint_run or _latest_complete(root['keypoints_runs'])
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

    def _load_artifact_split_indices(self, metadata: DatasetMetadata, split_name: str) -> np.ndarray:
        path = (
            metadata.artifact_train_indices_path
            if split_name == "train"
            else metadata.artifact_val_indices_path
            if split_name == "val"
            else None
        )
        if not path:
            return np.empty(0, dtype=np.int64)

        root = zarr.open(metadata.path, mode="r")
        raw_values = np.asarray(root[path][:])
        if raw_values.ndim != 1:
            raise ValueError(f"{metadata.name}: {path} must be 1D, got shape {raw_values.shape}.")
        if not np.issubdtype(raw_values.dtype, np.integer):
            raise ValueError(f"{metadata.name}: {path} must be integer dtype, got {raw_values.dtype}.")
        values = raw_values.astype(np.int64, copy=False)

        row_count = int(root[metadata.bbox_array_path].shape[0])
        if values.size > 0:
            min_idx = int(values.min())
            max_idx = int(values.max())
            if min_idx < 0 or max_idx >= row_count:
                raise ValueError(
                    f"{metadata.name}: {path} contains out-of-bounds sample indices "
                    f"(min={min_idx}, max={max_idx}, rows={row_count})."
                )
        if self.config.task == 'detect' and values.size > 0:
            frame_mask = self._get_frame_index_in_bounds_mask(root, metadata, row_count)
            values = values[frame_mask[values]]
        return values

    def _load_artifact_train_val_indices(self, metadata: DatasetMetadata) -> Tuple[np.ndarray, np.ndarray]:
        train = self._load_artifact_split_indices(metadata, "train")
        val = self._load_artifact_split_indices(metadata, "val")
        if train.size > 0 and val.size > 0:
            overlap = np.intersect1d(train, val)
            if overlap.size > 0:
                raise ValueError(
                    f"{metadata.name}: artifact train/val splits overlap at sample indices "
                    f"{overlap[:10].tolist()}."
                )
        return train, val

    @staticmethod
    def _sample_without_replacement(indices: np.ndarray, sample_count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample deterministic subset without replacement."""
        if sample_count <= 0 or indices.size == 0:
            return np.empty(0, dtype=int)
        if sample_count >= indices.size:
            return indices.copy()
        return rng.choice(indices, size=sample_count, replace=False)

    def _resolve_weight_map(self, metadata_rows: Optional[List[DatasetMetadata]] = None) -> Dict[str, float]:
        """Resolve dataset weights keyed by zarr path."""
        raw_weights = self.config.dataset_weights or {}
        if not raw_weights:
            raise ValueError("dataset_weights is required when sampling_strategy='weighted'.")

        rows = metadata_rows if metadata_rows is not None else self.metadata_list
        path_to_dataset_name = {
            metadata.path: self.config.get_dataset_name(metadata.path)
            for metadata in rows
        }
        matched_keys = set()
        path_weights: Dict[str, float] = {}

        for metadata in rows:
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

        unknown_keys = sorted(
            key for key in set(raw_weights.keys()) - matched_keys
            if key not in {self.config.get_dataset_name(metadata.path) for metadata in self.metadata_list}
            and key not in {metadata.name for metadata in self.metadata_list}
            and key not in {metadata.path for metadata in self.metadata_list}
        )
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
        
        artifact_split_rows = [
            m for m in self.metadata_list
            if m.artifact_train_indices_path and m.artifact_val_indices_path
        ]
        artifact_paths = {m.path for m in artifact_split_rows}
        regular_rows = [m for m in self.metadata_list if m.path not in artifact_paths]
        all_valid_indices = {m.path: self._get_valid_indices(m) for m in regular_rows}
        rng = np.random.default_rng(self.config.random_seed)

        strategy = self.config.sampling_strategy
        selected_by_path: Dict[str, np.ndarray] = {m.path: np.empty(0, dtype=int) for m in self.metadata_list}
        path_weights_for_logging: Dict[str, float] = {}

        for metadata in artifact_split_rows:
            train_part, val_part = self._load_artifact_train_val_indices(metadata)
            selected_by_path[metadata.path] = np.concatenate([train_part, val_part]).astype(int, copy=False)
            logger.info(
                f"  ✓ {metadata.name}: using artifact splits "
                f"(train={train_part.size}, val={val_part.size})."
            )

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
            available_counts = {
                path: indices.size
                for path, indices in all_valid_indices.items()
                if indices.size > 0
            }
            if not available_counts:
                logger.warning("  ⚠ No valid samples available after filtering.")
                if not artifact_split_rows:
                    self.selected_indices_by_path = {
                        path: indices.copy() for path, indices in selected_by_path.items()
                    }
                    return []
            else:
                path_weights = self._resolve_weight_map(regular_rows)
                path_weights_for_logging = path_weights

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
            if metadata.path not in all_valid_indices:
                logger.info(
                    f"  {metadata.name}: {len(selected_by_path[metadata.path])} samples "
                    "(artifact split-controlled)"
                )
                continue
            available = len(all_valid_indices[metadata.path])
            selected = len(selected_by_path[metadata.path])
            if strategy == SamplingStrategy.WEIGHTED:
                weight = path_weights_for_logging.get(metadata.path)
                if weight is not None:
                    logger.info(
                        f"  {metadata.name}: {selected}/{available} samples (weight={weight:.4g})"
                    )
                else:
                    logger.info(f"  {metadata.name}: {selected}/{available} samples")
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
            if metadata.artifact_train_indices_path and metadata.artifact_val_indices_path:
                train_part, val_part = self._load_artifact_train_val_indices(metadata)
                self.selected_indices_by_path[path] = np.concatenate([train_part, val_part]).astype(int, copy=False)
                train_indices.extend((path, int(det_idx)) for det_idx in train_part)
                val_indices.extend((path, int(det_idx)) for det_idx in val_part)
                logger.info(
                    f"  {metadata.name}: train={train_part.size} val={val_part.size} "
                    "(source=artifact_splits)"
                )
                continue

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


    def frame_index_oob_drop_summary(self) -> Dict[str, Any]:
        total_dropped = sum(int(row.get("dropped", 0)) for row in self.frame_index_oob_drops.values())
        total_rows = sum(int(row.get("rows", 0)) for row in self.frame_index_oob_drops.values())
        return {
            "total_dropped": int(total_dropped),
            "total_rows": int(total_rows),
            "datasets": {
                str(path): dict(row)
                for path, row in self.frame_index_oob_drops.items()
            },
        }


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
        summary_fn = getattr(index_manager, "frame_index_oob_drop_summary", None)
        self.frame_index_oob_drop_summary = (
            summary_fn()
            if callable(summary_fn)
            else {"total_dropped": 0, "total_rows": 0, "datasets": {}}
        )
        self.frame_index_oob_drops = dict(getattr(index_manager, "frame_index_oob_drops", {}))
        if self.config.task == 'pose':
            label_sets = {
                tuple(metadata.column_names)
                for metadata in index_manager.metadata_list
                if metadata.column_names
            }
            if not label_sets:
                raise ValueError(
                    "Pose dataset metadata is missing keypoint_labels; expected live run attrs or pose_schema nodes."
                )
            if len(label_sets) != 1:
                raise ValueError(
                    "Mixed keypoint_labels across configured pose datasets are not supported; "
                    f"observed signatures: {[list(labels) for labels in sorted(label_sets)]}"
                )
            self.keypoint_labels = list(next(iter(label_sets)))
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
        self.kp_box_only_cache = {}
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
                    if detect_parent is not None:
                        detect_latest = _latest_complete(detect_parent)
                    else:
                        detect_latest = None
                    if detect_latest:
                        frame_indices = root[f'detect_runs/{detect_latest}/frame_indices'][:]

                if frame_indices is None:
                    frame_count = self.bbox_cache[zarr_path].shape[0]
                    frame_indices = np.arange(frame_count, dtype=np.int64)
                elif frame_indices.shape[0] != self.bbox_cache[zarr_path].shape[0]:
                    frame_count = self.bbox_cache[zarr_path].shape[0]
                    raise ValueError(
                        f"{Path(zarr_path).name}: frame_indices length mismatch during dataset initialization "
                        f"({frame_indices.shape[0]} frame indices for {frame_count} label rows)."
                    )

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
                latest_crop = _latest_complete(root['crop_runs'])
                if latest_crop is None:
                    raise KeyError(f"No complete crop run available for {Path(zarr_path).name}.")
                crop_group = root[f'crop_runs/{latest_crop}']
                roi_shape = crop_group['roi_images'].shape[1:3]
                roi_h, roi_w = roi_shape
                self.roi_size_cache[zarr_path] = (roi_h, roi_w)

                metadata = self.metadata_map.get(zarr_path)
                kp_run_name = None
                if metadata is not None:
                    kp_run_name = metadata.keypoint_run
                if kp_run_name is None:
                    kp_run_name = _latest_complete(root['keypoints_runs'])
                if kp_run_name is None:
                    raise KeyError(f"No keypoint run available for {Path(zarr_path).name}; specify 'keypoint_run' in config.")

                kp_group = root[f'keypoints_runs/{kp_run_name}']
                kp_roi = kp_group['keypoints_roi'][:].astype(np.float32)
                kp_success = kp_group['detection_success'][:].astype(bool)
                if "keypoint_box_only" in kp_group:
                    kp_box_only = kp_group["keypoint_box_only"][:].astype(bool)
                    if kp_box_only.shape[0] != kp_roi.shape[0]:
                        raise ValueError(
                            f"{Path(zarr_path).name}: keypoint_box_only length {kp_box_only.shape[0]} "
                            f"does not match keypoints_roi length {kp_roi.shape[0]}"
                        )
                else:
                    kp_box_only = np.zeros(kp_roi.shape[0], dtype=bool)
                self.kp_box_only_cache[zarr_path] = kp_box_only

                finite_mask = np.isfinite(kp_roi).all(axis=(1, 2))
                full_supervision_mask = ~kp_box_only
                valid_full_mask = kp_success & finite_mask & full_supervision_mask
                self.kp_roi_norm_cache[zarr_path] = np.zeros_like(kp_roi, dtype=np.float32)

                if roi_w > 0 and roi_h > 0:
                    kp_roi_norm = kp_roi.copy()
                    kp_roi_norm[..., 0] /= float(roi_w)
                    kp_roi_norm[..., 1] /= float(roi_h)
                else:
                    kp_roi_norm = kp_roi

                kp_roi_norm = np.clip(kp_roi_norm, 0.0, 1.0)
                self.kp_roi_norm_cache[zarr_path] = kp_roi_norm
                bbox_path = None
                if metadata is not None and metadata.bbox_array_path:
                    bbox_path = metadata.bbox_array_path
                if not bbox_path:
                    bbox_path = f"crop_runs/{latest_crop}/bbox_norm_coords"
                crop_bbox = root[bbox_path][:].astype(np.float32)
                if crop_bbox.shape[0] != kp_roi.shape[0]:
                    raise ValueError(
                        f"{Path(zarr_path).name}: bbox array length {crop_bbox.shape[0]} "
                        f"does not match keypoints length {kp_roi.shape[0]}"
                    )
                valid_box_only_mask = kp_box_only & np.isfinite(crop_bbox).all(axis=1)
                valid_mask = valid_full_mask | valid_box_only_mask
                self.kp_success_cache[zarr_path] = valid_mask

                # Compute bboxes from keypoints for full rows and from crop bboxes for box-only rows.
                kpts = kp_roi_norm.reshape(kp_roi_norm.shape[0], -1, 2)
                bboxes = np.zeros((kpts.shape[0], 4), dtype=np.float32)
                full_idx = np.where(valid_full_mask)[0]
                if full_idx.size > 0:
                    kpts_valid = kpts[full_idx]
                    min_xy = np.nanmin(kpts_valid, axis=1)
                    max_xy = np.nanmax(kpts_valid, axis=1)
                    span = max_xy - min_xy
                    margin = span * 0.5
                    center = np.clip((min_xy + max_xy) / 2.0, 0.0, 1.0)
                    bbox_wh = np.clip(span + margin, 1e-6, 1.0)
                    bboxes_valid = np.concatenate([center, bbox_wh], axis=1).astype(np.float32)
                    bboxes[full_idx] = bboxes_valid
                box_only_idx = np.where(valid_box_only_mask)[0]
                if box_only_idx.size > 0:
                    bboxes[box_only_idx] = crop_bbox[box_only_idx]
                self.kp_bbox_cache[zarr_path] = bboxes

                visibility = np.full((kp_roi_norm.shape[0], kp_roi_norm.shape[1], 1), 2.0, dtype=np.float32)
                if np.any(kp_box_only):
                    visibility[kp_box_only, :, :] = 0.0
                kp_for_labels = kp_roi_norm.copy()
                if np.any(kp_box_only):
                    # Box-only rows should not carry NaNs because _get_pose_data rejects NaN labels.
                    kp_for_labels[kp_box_only] = np.nan_to_num(
                        kp_for_labels[kp_box_only],
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                kpts_with_vis = np.concatenate([kp_for_labels, visibility], axis=2).reshape(
                    kp_roi_norm.shape[0],
                    -1,
                )
                self.kp_flat_cache[zarr_path] = kpts_with_vis
                box_only_count = int(np.sum(valid_box_only_mask))
                logger.info(
                    f"  Cached {kp_roi_norm.shape[0]} keypoint entries from {Path(zarr_path).name} "
                    f"(run {kp_run_name}, box_only={box_only_count})"
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
            latest_crop = _latest_complete(root['crop_runs'])
            if latest_crop is None:
                raise KeyError(f"No complete crop run available for {Path(zarr_path).name}.")
            image_source_path = f"crop_runs/{latest_crop}/roi_images"
        
        frame_idx = None
        read_start = time.perf_counter() if profile_enabled else 0.0
        if self.config.task == 'detect':
            frame_array = self.detect_frame_arrays.get(zarr_path)
            if frame_array is None:
                frame_array = root[image_source_path]
                self.detect_frame_arrays[zarr_path] = frame_array
            frame_indices = self.frame_index_cache[zarr_path]
            frame_idx = int(frame_indices[det_idx]) if det_idx < len(frame_indices) else None
            if frame_idx is None or frame_idx < 0 or frame_idx >= frame_array.shape[0]:
                raise IndexError(
                    f"{Path(zarr_path).name}: selected label row {det_idx} maps to frame index "
                    f"{frame_idx}, outside frame array '{image_source_path}' with "
                    f"{frame_array.shape[0]} frames. Invalid rows must be dropped during index construction."
                )
            image = self._get_detect_frame(zarr_path, image_source_path, frame_idx)
        else:
            roi_idx = det_idx
            if roi_idx < 0 or roi_idx >= root[image_source_path].shape[0]:
                raise IndexError(
                    f"{Path(zarr_path).name}: selected label row {det_idx} is outside "
                    f"'{image_source_path}' with {root[image_source_path].shape[0]} ROI images."
                )
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
            "keypoints": label_info.get(
                'keypoints',
                np.zeros((0, len(self.keypoint_labels) * 3), dtype=np.float32),
            ).astype(np.float32),
            "num_keypoints": len(self.keypoint_labels) if self.config.task == "pose" else 0,
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
