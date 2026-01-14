# zarr_yolo_dataset_loader.py

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass, field
from enum import Enum
import yaml
from typing import Union, Any
from datetime import datetime

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
    source_type: str = 'filtered'  # 'detect', 'filtered', or 'interpolated'
    input_format: str = 'gray'  # 'gray' or 'rgb'
    split: Optional[Dict[str, float]] = None  # {'train': 0.8, 'val': 0.2}
    keypoint_run: Optional[str] = None  # Optional specific keypoints run

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
    target_size: Optional[int] = None
    min_confidence: float = 0.0
    filter_interpolated: bool = False  # DEPRECATED: Use source_type instead

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
    
    def get_zarr_paths(self) -> List[str]:
        """Get list of all zarr paths from datasets."""
        return [config.zarr_path for config in self.datasets.values()]
    
    def get_source_type(self, zarr_path: str) -> str:
        """Get source_type for a specific zarr path."""
        for config in self.datasets.values():
            if config.zarr_path == zarr_path:
                return config.source_type
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
                    logger.warning(
                        f"  ⚠ {Path(path_str).name}: Requested '{requested_source}' but data is '{actual_source_type}'. "
                        f"Using available '{actual_source_type}' data."
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
                    column_names = list(kp_group.attrs.get('keypoint_labels', ['bladder', 'eye_left', 'eye_right']))
                    success_arr = kp_group['detection_success'][:]
                    if total_frames > 0:
                        tracking_success_rate = float(np.mean(success_arr) * 100.0)

                if not bbox_array_path:
                    raise KeyError(f"Unable to determine bbox source for {Path(path_str).name}.")

                source_coords = root[bbox_array_path][:]
                valid_mask = np.zeros(source_coords.shape[0], dtype=bool) if source_coords.size == 0 else ~np.isnan(source_coords[:, 0])

                if requested_source in ['filtered', 'detect'] and has_interpolated and detection_source_path:
                    detection_source = root[detection_source_path][:]
                    real_mask = (detection_source == 0)
                    valid_mask = valid_mask & real_mask

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
        
        # Filter out interpolated data if source_type is 'filtered' or 'detect'
        if (
            metadata.requested_source_type in ['filtered', 'detect']
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
            if 'detection_success' not in kp_group:
                raise KeyError(f"Keypoint run '{kp_run}' missing 'detection_success' array.")
            success_mask = kp_group['detection_success'][:]
            valid_mask &= success_mask
        
        return np.where(valid_mask)[0]

    def _build_global_index(self) -> List[Tuple[str, int]]:
        """Build global index, filtering based on source_type."""
        logger.info("Building global sample index...")
        
        # Check if any dataset is using filtered/detect source types
        any_filtering = any(m.requested_source_type in ['filtered', 'detect'] for m in self.metadata_list)
        if any_filtering:
            logger.info("  ℹ Using only real detections (filtering interpolated data)")
        
        all_valid_indices = {m.path: self._get_valid_indices(m) for m in self.metadata_list}
        
        global_indices = [(path, index) for path, indices in all_valid_indices.items() for index in indices]
        
        np.random.seed(self.config.random_seed)
        np.random.shuffle(global_indices)

        logger.info(f"Global index created with {len(global_indices)} total samples.")
        
        # Log statistics per dataset
        for metadata in self.metadata_list:
            n_samples = len(all_valid_indices[metadata.path])
            logger.info(f"  {metadata.name}: {n_samples} samples")
        
        return global_indices

    def get_split_indices(self) -> Tuple[List, List]:
        return train_test_split(
            self.global_indices, train_size=self.config.split_ratio, random_state=self.config.random_seed
        )


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
            labels = first_metadata.column_names if first_metadata and first_metadata.column_names else ['bladder', 'eye_left', 'eye_right']
            self.keypoint_labels = labels
        else:
            self.keypoint_labels = []
        self.zarr_roots = {path: zarr.open(path, mode='r') for path in config.get_zarr_paths()}

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
                source_label = "crop_runs" if metadata.uses_crop_data else "detect_runs"
                logger.info(
                    f"  Cached {self.bbox_cache[zarr_path].shape[0]} bboxes from {Path(zarr_path).name} ({source_label})."
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
        if self.config.task == 'detect':
            frame_indices = self.frame_index_cache[zarr_path]
            frame_idx = int(frame_indices[det_idx]) if det_idx < len(frame_indices) else None
            if frame_idx is None or frame_idx >= root[image_source_path].shape[0]:
                # fallback: skip label/image mismatch
                frame_idx = 0
                image = np.zeros_like(root[image_source_path][0])
            else:
                image = root[image_source_path][frame_idx]
        else:
            roi_idx = det_idx
            if roi_idx >= root[image_source_path].shape[0]:
                image = np.zeros_like(root[image_source_path][0])
                roi_idx = 0
            else:
                image = root[image_source_path][roi_idx]

        if self.config.task == 'detect' and metadata.input_format == 'rgb' and image.ndim == 3:
            image_3ch = image
        else:
            if image.ndim == 2:
                image_3ch = np.stack([image] * 3, axis=-1)
            else:
                image_3ch = image
        
        label_info = self.labels[index]
        ori_shape = (image_3ch.shape[0], image_3ch.shape[1])
        
        im_identifier = (f"{Path(zarr_path).stem}_frame_{frame_idx}"
                         if self.config.task == 'detect'
                         else f"{Path(zarr_path).stem}_roi_{det_idx}")

        return {
            "img": image_3ch.transpose(2, 0, 1),
            "cls": label_info.get('cls', np.zeros((0,), dtype=np.float32)).astype(np.float32),
            "bboxes": label_info.get('bboxes', np.zeros((0, 4), dtype=np.float32)).astype(np.float32),
            "keypoints": label_info.get('keypoints', np.zeros((0, 9), dtype=np.float32)).astype(np.float32),
            "im_file": im_identifier,
            "ori_shape": ori_shape,
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "segments": np.zeros((0, 0), dtype=np.float32)
        }


def create_zarr_dataset(config: Union[ZarrDatasetConfig, Dict], mode: str) -> ZarrYOLODataset:
    """Factory function to create a ZarrYOLODataset."""
    if isinstance(config, dict):
        config = ZarrDatasetConfig(**config)
    return ZarrYOLODataset(config, mode)
