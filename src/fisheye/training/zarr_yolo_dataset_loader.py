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
    split: Optional[Dict[str, float]] = None  # {'train': 0.8, 'val': 0.2}

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


class GlobalIndexManager:
    """Builds and manages a global index across all specified Zarr files."""
    def __init__(self, config: ZarrDatasetConfig):
        self.config = config
        self.metadata_list = self._validate_and_get_metadata()
        self.global_indices = self._build_global_index()

    def _validate_and_get_metadata(self) -> List[DatasetMetadata]:
        zarr_paths = self.config.get_zarr_paths()
        logger.info(f"Validating {len(zarr_paths)} Zarr files...")
        metadata_list = []
        
        for path_str in zarr_paths:
            try:
                root = zarr.open(path_str, mode='r')
                
                # Get requested source type for this dataset
                requested_source = self.config.get_source_type(path_str)
                
                # Check for crop_runs (required for all source types)
                if 'crop_runs' not in root or 'latest' not in root['crop_runs'].attrs:
                    raise KeyError(f"Could not find 'crop_runs' in {Path(path_str).name}")
                
                latest_crop = root['crop_runs'].attrs['latest']
                crop_group = root[f'crop_runs/{latest_crop}']
                
                # Get actual crop source info
                actual_source_type = crop_group.attrs.get('detection_source_type', 'detect')
                has_interpolated = crop_group.attrs.get('includes_interpolated', False)
                n_real_rois = crop_group.attrs.get('n_real_detections', 0)
                n_interpolated_rois = crop_group.attrs.get('n_interpolated_detections', 0)
                
                # Validate that requested source matches what's available
                if requested_source != actual_source_type:
                    logger.warning(
                        f"  ⚠ {Path(path_str).name}: Requested '{requested_source}' but crops are from '{actual_source_type}'. "
                        f"Using available '{actual_source_type}' data."
                    )
                
                # Check for tracking data (ONLY required for pose task)
                column_names: List[str] = []
                total_frames = 0
                tracking_success_rate = 0.0
                
                if self.config.task == 'pose':
                    if 'keypoints_runs' not in root or 'latest' not in root['keypoints_runs'].attrs:
                        raise KeyError("Pose task requires 'keypoints_runs' with 'latest' attribute.")

                    latest_run_name = root['keypoints_runs'].attrs['latest']
                    kp_group = root[f'keypoints_runs/{latest_run_name}']
                    if 'keypoints_roi' not in kp_group:
                        raise KeyError(f"Keypoint run '{latest_run_name}' missing 'keypoints_roi' array.")

                    total_frames = kp_group['keypoints_roi'].shape[0]
                    column_names = list(kp_group.attrs.get('keypoint_labels', ['bladder', 'eye_left', 'eye_right']))
                    success_arr = kp_group['detection_success'][:]
                    if total_frames > 0:
                        tracking_success_rate = float(np.mean(success_arr) * 100.0)
                else:
                    # Detect task only needs crops
                    if 'roi_images' not in crop_group:
                        raise KeyError(f"No ROI images found in crop run {latest_crop}")
                    
                    total_frames = crop_group['roi_images'].shape[0]
                
                # Determine source coordinates
                if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs:
                    latest_refine_run = root['refine_runs'].attrs['latest']
                    source_coords = root[f'refine_runs/{latest_refine_run}/refined_bbox_norm_coords']
                elif 'crop_runs' in root:
                    source_coords = crop_group['bbox_norm_coords']
                else:
                    raise KeyError("No valid coordinates found")
                
                valid_mask = ~np.isnan(source_coords[:, 0])
                if requested_source in ['filtered', 'detect'] and has_interpolated:
                    if 'detection_source' in crop_group:
                        real_mask = (crop_group['detection_source'][:] == 0)
                        valid_mask = valid_mask & real_mask
                if self.config.task == 'pose':
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
                    roi_shape=crop_group['roi_images'].shape[1:3] if 'roi_images' in crop_group else (0, 0)
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
        
        # Get source coordinates
        source_coords_path = (f"refine_runs/{root['refine_runs'].attrs['latest']}/refined_bbox_norm_coords" 
                              if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs 
                              else f"crop_runs/{root['crop_runs'].attrs['latest']}/bbox_norm_coords")
        valid_mask = ~np.isnan(root[source_coords_path][:, 0])
        
        # Filter out interpolated data if source_type is 'filtered' or 'detect'
        if metadata.requested_source_type in ['filtered', 'detect'] and metadata.has_interpolated:
            latest_crop = root['crop_runs'].attrs['latest']
            crop_group = root[f'crop_runs/{latest_crop}']
            
            if 'detection_source' in crop_group:
                detection_source = crop_group['detection_source'][:]
                # Only keep real detections (0), filter out interpolated (1)
                real_mask = (detection_source == 0)
                valid_mask = valid_mask & real_mask
                
                n_filtered = np.sum(detection_source == 1)
                logger.info(f"    Filtered out {n_filtered} interpolated ROIs from {metadata.name}")

        # For pose task, also check keypoints validity
        if self.config.task == 'pose':
            latest_kp_run = root['keypoints_runs'].attrs['latest']
            kp_group = root[f'keypoints_runs/{latest_kp_run}']
            if 'detection_success' not in kp_group:
                raise KeyError(f"Keypoint run '{latest_kp_run}' missing 'detection_success' array.")
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
        
        # Cache detection metadata per zarr file for detect task (load once per file)
        self.bbox_cache = {}
        self.frame_index_cache = {}
        if self.config.task == 'detect':
            for zarr_path in self.zarr_roots.keys():
                root = self.zarr_roots[zarr_path]
                latest_crop = root['crop_runs'].attrs['latest']
                crop_group = root[f'crop_runs/{latest_crop}']
                self.bbox_cache[zarr_path] = crop_group['bbox_norm_coords'][:]  # Load once
                if 'frame_indices' in crop_group:
                    self.frame_index_cache[zarr_path] = crop_group['frame_indices'][:]
                else:
                    # Fallback to detect run
                    detect_latest = root['detect_runs'].attrs['latest']
                    self.frame_index_cache[zarr_path] = root[f'detect_runs/{detect_latest}/frame_indices'][:]
                logger.info(f"  Cached {self.bbox_cache[zarr_path].shape[0]} bboxes from {Path(zarr_path).name}")
        
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
        
        return {"cls": np.array([]), "bboxes": np.empty((0, 4))}

    def _get_pose_data(self, zarr_path: str, det_idx: int) -> Dict:
        try:
            root = self.zarr_roots[zarr_path]
            latest_kp_run = root['keypoints_runs'].attrs['latest']
            kp_group = root[f'keypoints_runs/{latest_kp_run}']

            if 'keypoints_norm' not in kp_group or 'detection_success' not in kp_group:
                return {"cls": np.array([]), "bboxes": np.empty((0, 4))}
            
            if det_idx >= kp_group['keypoints_norm'].shape[0]:
                return {"cls": np.array([]), "bboxes": np.empty((0, 4))}
            
            if not bool(kp_group['detection_success'][det_idx]):
                return {"cls": np.array([]), "bboxes": np.empty((0, 4))}

            kpts_norm = kp_group['keypoints_norm'][det_idx].astype(np.float32)
            if np.isnan(kpts_norm).any():
                return {"cls": np.array([]), "bboxes": np.empty((0, 4))}

            kpts_flat = kpts_norm.reshape(-1)
            keypoints_x = kpts_flat[0::2]
            keypoints_y = kpts_flat[1::2]
            
            min_x, max_x = np.min(keypoints_x), np.max(keypoints_x)
            min_y, max_y = np.min(keypoints_y), np.max(keypoints_y)

            span_x = max_x - min_x
            span_y = max_y - min_y
            margin_x = span_x * 0.5
            margin_y = span_y * 0.5

            bbox_x = float(np.clip((min_x + max_x) / 2.0, 0.0, 1.0))
            bbox_y = float(np.clip((min_y + max_y) / 2.0, 0.0, 1.0))
            bbox_w = float(np.clip(span_x + margin_x, 1e-6, 1.0))
            bbox_h = float(np.clip(span_y + margin_y, 1e-6, 1.0))

            kpts_with_visibility = np.array([
                kpts_flat[0], kpts_flat[1], 2,
                kpts_flat[2], kpts_flat[3], 2,
                kpts_flat[4], kpts_flat[5], 2
            ], dtype=np.float32).reshape(1, -1)

            return {
                "cls": np.array([0]),
                "bboxes": np.array([[bbox_x, bbox_y, bbox_w, bbox_h]], dtype=np.float32),
                "keypoints": kpts_with_visibility
            }
        except (KeyError, IndexError):
            return {"cls": np.array([]), "bboxes": np.empty((0, 4))}
    
    def __getitem__(self, index: int) -> Dict:
        zarr_path, det_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        
        image_source_path = ('raw_video/images_ds' if self.config.task == 'detect' 
                             else f"crop_runs/{root['crop_runs'].attrs['latest']}/roi_images")
        
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

        image_3ch = np.stack([image] * 3, axis=-1)
        
        if image_3ch.shape[0] != self.target_size or image_3ch.shape[1] != self.target_size:
            import cv2
            image_3ch = cv2.resize(image_3ch, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
        label_info = self.labels[index]
        
        im_identifier = (f"{Path(zarr_path).stem}_frame_{frame_idx}"
                         if self.config.task == 'detect'
                         else f"{Path(zarr_path).stem}_roi_{det_idx}")

        return {
            "img": image_3ch.transpose(2, 0, 1),
            "cls": label_info.get('cls', np.array([])),
            "bboxes": label_info.get('bboxes', np.empty((0, 4))),
            "keypoints": label_info.get('keypoints', np.empty((0, 9))),
            "im_file": im_identifier,
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (None, None) 
        }


def create_zarr_dataset(config: Union[ZarrDatasetConfig, Dict], mode: str) -> ZarrYOLODataset:
    """Factory function to create a ZarrYOLODataset."""
    if isinstance(config, dict):
        config = ZarrDatasetConfig(**config)
    return ZarrYOLODataset(config, mode)
