# enhanced_multi_zarr_dataset.py (Final Version with all fixes)

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
from typing import Union, Any
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Utility Functions ---

def detect_data_format(column_names: List[str]) -> str:
    """Detects if the Zarr file uses the 'enhanced' or 'original' format."""
    if 'bbox_x_norm_ds' in column_names:
        return 'enhanced'
    if 'bbox_x_norm' in column_names:
        return 'original'
    raise ValueError("Could not determine data format. Missing 'bbox_x_norm_ds' or 'bbox_x_norm'.")

def get_column_mappings(column_names: List[str], data_format: str) -> Dict:
    """Gets a dictionary mapping coordinate names to their column index."""
    col_map = {name: i for i, name in enumerate(column_names)}
    if data_format == 'enhanced':
        return {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm_ds'],
            'bbox_y': col_map['bbox_y_norm_ds'],
            'bbox_width': col_map['bbox_width_norm_ds'],
            'bbox_height': col_map['bbox_height_norm_ds'],
            'confidence': col_map.get('confidence_score')
        }
    return { # Original format
        'heading': col_map['heading_degrees'],
        'bbox_x': col_map['bbox_x_norm'],
        'bbox_y': col_map['bbox_y_norm'],
    }


# --- Configuration ---

class SamplingStrategy(Enum):
    """Sampling strategies for combining multiple datasets."""
    BALANCED = "balanced"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"

@dataclass
class MultiDatasetConfig:
    """Configuration for the multi-Zarr dataset."""
    zarr_paths: List[str]
    task: str = 'detect'
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED
    split_ratio: float = 0.8
    random_seed: int = 42
    dataset_weights: Optional[Dict[str, float]] = None
    target_size: Optional[int] = None
    min_confidence: float = 0.0

    @classmethod
    def from_yaml(cls, path: str):
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        strategy_str = config_dict.get('sampling_strategy', 'balanced')
        try:
            config_dict['sampling_strategy'] = SamplingStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Unknown sampling strategy '{strategy_str}', defaulting to 'balanced'.")
            config_dict['sampling_strategy'] = SamplingStrategy.BALANCED
            
        known_keys = cls.__annotations__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_keys}
        
        return cls(**filtered_dict)


# --- Core Dataset and Helper Classes ---

@dataclass
class DatasetMetadata:
    """Holds metadata extracted from a single Zarr file."""
    path: str
    name: str
    total_frames: int
    valid_frames: int
    data_format: str
    column_names: List[str]
    tracking_success_rate: float = 0.0

class GlobalIndexManager:
    """Builds and manages a global index across all specified Zarr files."""
    def __init__(self, config: MultiDatasetConfig):
        self.config = config
        self.metadata_list = self._validate_and_get_metadata()
        self.global_indices = self._build_global_index()

    def _validate_and_get_metadata(self) -> List[DatasetMetadata]:
        logger.info(f"🔍 Validating {len(self.config.zarr_paths)} Zarr files...")
        metadata_list = []
        for path_str in self.config.zarr_paths:
            try:
                root = zarr.open(path_str, mode='r')
                if 'tracking_runs' not in root or 'latest' not in root['tracking_runs'].attrs:
                    raise KeyError("Could not find 'tracking_runs' with 'latest' attribute.")
                
                latest_run_name = root['tracking_runs'].attrs['latest']
                tracking_results_path = f'tracking_runs/{latest_run_name}/tracking_results'
                if tracking_results_path not in root:
                    raise KeyError(f"Latest tracking results not found at '{tracking_results_path}'")
                
                tracking_results = root[tracking_results_path]
                
                column_names = tracking_results.attrs['column_names']
                data_format = detect_data_format(column_names)
                
                if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs:
                    latest_refine_run = root['refine_runs'].attrs['latest']
                    source_coords = root[f'refine_runs/{latest_refine_run}/refined_bbox_norm_coords']
                else:
                    latest_crop_run = root['crop_runs'].attrs['latest']
                    source_coords = root[f'crop_runs/{latest_crop_run}/bbox_norm_coords']
                
                valid_mask = ~np.isnan(source_coords[:, 0])
                valid_frames = np.sum(valid_mask)
                total_frames = len(tracking_results)

                metadata_list.append(DatasetMetadata(
                    path=path_str, name=Path(path_str).stem, total_frames=total_frames,
                    valid_frames=valid_frames, data_format=data_format, column_names=column_names,
                    tracking_success_rate=(np.sum(~np.isnan(tracking_results[:, 0])) / total_frames * 100) if total_frames > 0 else 0
                ))
            except Exception as e:
                raise IOError(f"Failed to process Zarr file at '{path_str}': {e}")
        
        if len({m.data_format for m in metadata_list}) > 1:
            raise ValueError("All Zarr files must have the same data format.")
        
        logger.info("✅ All Zarr files are compatible.")
        return metadata_list

    def _get_valid_indices(self, metadata: DatasetMetadata) -> np.ndarray:
        root = zarr.open(metadata.path, mode='r')
        
        if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs:
            latest_refine_run = root['refine_runs'].attrs['latest']
            source_coords = root[f'refine_runs/{latest_refine_run}/refined_bbox_norm_coords']
        else:
            latest_crop_run = root['crop_runs'].attrs['latest']
            source_coords = root[f'crop_runs/{latest_crop_run}/bbox_norm_coords']
            
        valid_mask = ~np.isnan(source_coords[:, 0])
        
        if self.config.min_confidence > 0 and metadata.data_format == 'enhanced':
            latest_track_run = root['tracking_runs'].attrs['latest']
            tracking_data = root[f'tracking_runs/{latest_track_run}/tracking_results']
            col_map = {name: i for i, name in enumerate(metadata.column_names)}
            conf_idx = col_map.get('confidence_score')
            if conf_idx is not None:
                confidence_data = tracking_data[:, conf_idx]
                valid_mask &= (np.isnan(confidence_data) | (confidence_data >= self.config.min_confidence))
        
        return np.where(valid_mask)[0]

    def _apply_sampling(self, all_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        strategy = self.config.sampling_strategy
        if strategy == SamplingStrategy.PROPORTIONAL:
            return all_indices

        if strategy == SamplingStrategy.BALANCED:
            if not all_indices or not any(len(v) > 0 for v in all_indices.values()):
                return {}
            min_samples = min(len(indices) for indices in all_indices.values() if len(indices) > 0)
            logger.info(f"⚖️ Balancing datasets to {min_samples} samples each.")
            for path, indices in all_indices.items():
                if len(indices) > min_samples:
                    np.random.seed(self.config.random_seed)
                    all_indices[path] = np.random.choice(indices, min_samples, replace=False)
            return all_indices
            
        return all_indices

    def _build_global_index(self) -> List[Tuple[str, int]]:
        logger.info("🔧 Building global sample index...")
        all_valid_indices = {m.path: self._get_valid_indices(m) for m in self.metadata_list}
        sampled_indices = self._apply_sampling(all_valid_indices)
        
        global_indices = [(path, index) for path, indices in sampled_indices.items() for index in indices]
        
        np.random.seed(self.config.random_seed)
        np.random.shuffle(global_indices)
        
        logger.info(f"✅ Global index created with {len(global_indices)} total samples.")
        return global_indices

    def get_split_indices(self) -> Tuple[List, List]:
        return train_test_split(
            self.global_indices, train_size=self.config.split_ratio, random_state=self.config.random_seed
        )

class EnhancedMultiZarrYOLODataset(Dataset):
    def __init__(self, config: MultiDatasetConfig, mode: str = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode

        index_manager = GlobalIndexManager(config)
        train_indices, val_indices = index_manager.get_split_indices()
        self.indices = train_indices if mode == 'train' else val_indices
        
        self.metadata_map = {m.path: m for m in index_manager.metadata_list}
        # Access metadata_list from the manager ---
        self.data_format = index_manager.metadata_list[0].data_format if index_manager.metadata_list else 'enhanced'
        self.column_mappings = get_column_mappings(index_manager.metadata_list[0].column_names, self.data_format)
        
        self.zarr_roots = {path: zarr.open(path, mode='r') for path in config.zarr_paths}
        
        self.target_size = self.config.target_size or (640 if self.config.task == 'detect' else 320)
        
        logger.info(f"Pre-caching labels for {self.mode} set...")
        self.labels = []
        for zarr_path, frame_idx in self.indices:
            label = self._get_bbox_data(zarr_path, frame_idx)
            self.labels.append({
                "cls": label[:, 0].astype(np.float32),
                "bboxes": label[:, 1:5].astype(np.float32)
            })
        
        logger.info(f"✅ Initialized '{mode}' dataset with {len(self.indices)} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def _get_bbox_data(self, zarr_path: str, frame_idx: int) -> np.ndarray:
        """
        Gets bounding box data. Prefers tight, keypoint-derived tracking data, 
        and uses a tight fallback for crop-only data.
        """
        root = self.zarr_roots[zarr_path]
        latest_track_run = root['tracking_runs'].attrs['latest']
        tracking_data = root[f'tracking_runs/{latest_track_run}/tracking_results'][frame_idx]
        
        # 1. Prioritize the tight bounding box from the tracking stage
        if not np.isnan(tracking_data[self.column_mappings['heading']]):
            bbox_x = tracking_data[self.column_mappings['bbox_x']]
            bbox_y = tracking_data[self.column_mappings['bbox_y']]
            bbox_w = tracking_data[self.column_mappings['bbox_width']]
            bbox_h = tracking_data[self.column_mappings['bbox_height']]
            if not any(np.isnan([bbox_x, bbox_y, bbox_w, bbox_h])):
                return np.array([[0, bbox_x, bbox_y, bbox_w, bbox_h]], dtype=np.float32)

        # 2. If tracking failed, use the refined/cropped center point
        if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs:
            latest_refine_run = root['refine_runs'].attrs['latest']
            crop_coords = root[f'refine_runs/{latest_refine_run}/refined_bbox_norm_coords'][frame_idx]
        else:
            latest_crop_run = root['crop_runs'].attrs['latest']
            crop_coords = root[f'crop_runs/{latest_crop_run}/bbox_norm_coords'][frame_idx]

        bbox_x, bbox_y = crop_coords[0], crop_coords[1]
        if not any(np.isnan([bbox_x, bbox_y])):
            # Use a small, tight default size of bounding box
            default_width = 0.08  # 10% of image width
            default_height = 0.10  # 10% of image height (fish are often longer than they are wide)
            return np.array([[0, bbox_x, bbox_y, default_width, default_height]], dtype=np.float32)

        # 3. Final fallback for completely failed frames
        return np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)

    def __getitem__(self, index: int) -> Dict:
        zarr_path, frame_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        
        # Locate the ROI images
        if self.config.task == 'detect':
            image_source_path = 'raw_video/images_ds'
        else: # pose
            latest_crop_run = root['crop_runs'].attrs['latest']
            image_source_path = f'crop_runs/{latest_crop_run}/roi_images'
        
        image = root[image_source_path][frame_idx]
        image_3ch = np.stack([image] * 3, axis=-1).astype(np.uint8)
        
        if image_3ch.shape[0] != self.target_size:
            import cv2
            image_3ch = cv2.resize(image_3ch, (self.target_size, self.target_size))
            
        image_tensor = image_3ch.transpose(2, 0, 1)

        label_info = self.labels[index]

        return {
            "img": image_tensor,
            "cls": label_info['cls'],
            "bboxes": label_info['bboxes'],
            "im_file": f"{Path(zarr_path).stem}_frame_{frame_idx}",
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (1.0, (0.0, 0.0))
        }

def create_multi_zarr_dataset(config: Union[MultiDatasetConfig, Dict], mode: str) -> EnhancedMultiZarrYOLODataset:
    if isinstance(config, dict):
        config = MultiDatasetConfig(**config)
    return EnhancedMultiZarrYOLODataset(config, mode)