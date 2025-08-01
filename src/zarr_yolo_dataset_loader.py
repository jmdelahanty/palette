# zarr_yolo_dataset_loader.py

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


# Utility Functions

def get_column_mappings(column_names: List[str]) -> Dict:
    """Gets a dictionary mapping coordinate names to their column index for the multi-scale format."""
    col_map = {name: i for i, name in enumerate(column_names)}
    return {
        'heading': col_map['heading_degrees'],
        'bbox_x': col_map['bbox_x_norm_ds'],
        'bbox_y': col_map['bbox_y_norm_ds'],
        'bbox_width': col_map['bbox_width_norm_ds'],
        'bbox_height': col_map['bbox_height_norm_ds'],
        'bladder_x': col_map.get('bladder_x_roi_norm'),
        'bladder_y': col_map.get('bladder_y_roi_norm'),
        'eye_l_x': col_map.get('eye_l_x_roi_norm'),
        'eye_l_y': col_map.get('eye_l_y_roi_norm'),
        'eye_r_x': col_map.get('eye_r_x_roi_norm'),
        'eye_r_y': col_map.get('eye_r_y_roi_norm'),
        'confidence': col_map.get('confidence_score')
    }


# Configuration
class SamplingStrategy(Enum):
    """Sampling strategies for combining multiple datasets."""
    BALANCED = "balanced"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"

@dataclass
class ZarrDatasetConfig:
    """Configuration for the Zarr dataset loader."""
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


# Core Dataset and Helper Classes

@dataclass
class DatasetMetadata:
    """Holds metadata extracted from a single Zarr file."""
    path: str
    name: str
    total_frames: int
    valid_frames: int
    column_names: List[str]
    tracking_success_rate: float = 0.0

class GlobalIndexManager:
    """Builds and manages a global index across all specified Zarr files."""
    def __init__(self, config: ZarrDatasetConfig):
        self.config = config
        self.metadata_list = self._validate_and_get_metadata()
        self.global_indices = self._build_global_index()

    def _validate_and_get_metadata(self) -> List[DatasetMetadata]:
        logger.info(f"Validating {len(self.config.zarr_paths)} Zarr files...")
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
                    valid_frames=valid_frames, column_names=column_names,
                    tracking_success_rate=(np.sum(~np.isnan(tracking_results[:, 0])) / total_frames * 100) if total_frames > 0 else 0
                ))
            except Exception as e:
                raise IOError(f"Failed to process Zarr file at '{path_str}': {e}")
        
        logger.info("All Zarr files are compatible!")
        return metadata_list

    def _get_valid_indices(self, metadata: DatasetMetadata) -> np.ndarray:
        root = zarr.open(metadata.path, mode='r')
        
        source_coords_path = (f"refine_runs/{root['refine_runs'].attrs['latest']}/refined_bbox_norm_coords" 
                              if 'refine_runs' in root and 'latest' in root['refine_runs'].attrs 
                              else f"crop_runs/{root['crop_runs'].attrs['latest']}/bbox_norm_coords")
        valid_mask = ~np.isnan(root[source_coords_path][:, 0])

        if self.config.task == 'pose':
            latest_track_run = root['tracking_runs'].attrs['latest']
            tracking_data = root[f'tracking_runs/{latest_track_run}/tracking_results']
            col_map = {name: i for i, name in enumerate(metadata.column_names)}
            
            kpt_indices = [
                col_map.get('bladder_x_roi_norm'), col_map.get('bladder_y_roi_norm'),
                col_map.get('eye_l_x_roi_norm'), col_map.get('eye_l_y_roi_norm'),
                col_map.get('eye_r_x_roi_norm'), col_map.get('eye_r_y_roi_norm')
            ]

            if all(idx is not None for idx in kpt_indices):
                valid_mask &= ~np.any(np.isnan(tracking_data.oindex[:, kpt_indices]), axis=1)
        
        return np.where(valid_mask)[0]

    def _build_global_index(self) -> List[Tuple[str, int]]:
        logger.info("Building global sample index...")
        all_valid_indices = {m.path: self._get_valid_indices(m) for m in self.metadata_list}
        
        global_indices = [(path, index) for path, indices in all_valid_indices.items() for index in indices]
        
        np.random.seed(self.config.random_seed)
        np.random.shuffle(global_indices)

        logger.info(f"Global index created with {len(global_indices)} total samples.")
        return global_indices

    def get_split_indices(self) -> Tuple[List, List]:
        return train_test_split(
            self.global_indices, train_size=self.config.split_ratio, random_state=self.config.random_seed
        )

class ZarrYOLODataset(Dataset):
    def __init__(self, config: ZarrDatasetConfig, mode: str = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode

        index_manager = GlobalIndexManager(config)
        train_indices, val_indices = index_manager.get_split_indices()
        self.indices = train_indices if mode == 'train' else val_indices
        
        self.metadata_map = {m.path: m for m in index_manager.metadata_list}
        self.column_mappings = get_column_mappings(index_manager.metadata_list[0].column_names)
        self.zarr_roots = {path: zarr.open(path, mode='r') for path in config.zarr_paths}

        self.target_size = self.config.target_size or (640 if self.config.task == 'detect' else 256)

        logger.info(f"Pre-caching labels for {self.mode} set ({self.config.task} task)...")
        self.labels = []
        label_fetcher = self._get_pose_data if self.config.task == 'pose' else self._get_bbox_data
        for zarr_path, frame_idx in self.indices:
            self.labels.append(label_fetcher(zarr_path, frame_idx))
        
        logger.info(f"Initialized '{mode}' dataset with {len(self.indices)} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def _get_bbox_data(self, zarr_path: str, frame_idx: int) -> Dict:
        root = self.zarr_roots[zarr_path]
        latest_track_run = root['tracking_runs'].attrs['latest']
        tracking_data = root[f'tracking_runs/{latest_track_run}/tracking_results'][frame_idx]
        
        if not np.isnan(tracking_data[self.column_mappings['heading']]):
            bbox_x = tracking_data[self.column_mappings['bbox_x']]
            bbox_y = tracking_data[self.column_mappings['bbox_y']]
            bbox_w = tracking_data[self.column_mappings['bbox_width']]
            bbox_h = tracking_data[self.column_mappings['bbox_height']]
            if not any(np.isnan([bbox_x, bbox_y, bbox_w, bbox_h])):
                return {"cls": np.array([0]), "bboxes": np.array([[bbox_x, bbox_y, bbox_w, bbox_h]])}

        source_coords_path = (f"refine_runs/{root['refine_runs'].attrs['latest']}/refined_bbox_norm_coords" 
                              if 'refine_runs' in root else f"crop_runs/{root['crop_runs'].attrs['latest']}/bbox_norm_coords")
        crop_coords = root[source_coords_path][frame_idx]
        bbox_x, bbox_y = crop_coords[0], crop_coords[1]
        if not any(np.isnan([bbox_x, bbox_y])):
            return {"cls": np.array([0]), "bboxes": np.array([[bbox_x, bbox_y, 0.08, 0.10]])}

        return {"cls": np.array([]), "bboxes": np.empty((0, 4))}

    def _get_pose_data(self, zarr_path: str, frame_idx: int) -> Dict:
        try:
            root = self.zarr_roots[zarr_path]
            latest_track_run = root['tracking_runs'].attrs['latest']
            tracking_data = root[f'tracking_runs/{latest_track_run}/tracking_results'][frame_idx]

            kpts_flat = np.array([
                tracking_data[self.column_mappings['bladder_x']], tracking_data[self.column_mappings['bladder_y']],
                tracking_data[self.column_mappings['eye_l_x']], tracking_data[self.column_mappings['eye_l_y']],
                tracking_data[self.column_mappings['eye_r_x']], tracking_data[self.column_mappings['eye_r_y']],
            ], dtype=np.float32)

            if np.isnan(kpts_flat).any():
                return {"cls": np.array([]), "bboxes": np.empty((0, 4))}

            keypoints_x = kpts_flat[0::2]
            keypoints_y = kpts_flat[1::2]
            
            min_x, max_x = np.min(keypoints_x), np.max(keypoints_x)
            min_y, max_y = np.min(keypoints_y), np.max(keypoints_y)

            margin_x = (max_x - min_x) * 0.5
            margin_y = (max_y - min_y) * 0.5

            bbox_x = (min_x + max_x) / 2.0
            bbox_y = (min_y + max_y) / 2.0
            bbox_w = (max_x - min_x) + margin_x
            bbox_h = (max_y - min_y) + margin_y

            kpts_with_visibility = np.array([kpts_flat[0], kpts_flat[1], 2, kpts_flat[2], kpts_flat[3], 2, kpts_flat[4], kpts_flat[5], 2]).reshape(1, -1)

            return {
                "cls": np.array([0]),
                "bboxes": np.array([[bbox_x, bbox_y, bbox_w, bbox_h]]),
                "keypoints": kpts_with_visibility
            }
        except (KeyError, IndexError):
            return {"cls": np.array([]), "bboxes": np.empty((0, 4))}
    
    def __getitem__(self, index: int) -> Dict:
        zarr_path, frame_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        
        image_source_path = ('raw_video/images_ds' if self.config.task == 'detect' 
                             else f"crop_runs/{root['crop_runs'].attrs['latest']}/roi_images")
        
        image = root[image_source_path][frame_idx]
        image_3ch = np.stack([image] * 3, axis=-1)
        
        if image_3ch.shape[0] != self.target_size or image_3ch.shape[1] != self.target_size:
            import cv2
            image_3ch = cv2.resize(image_3ch, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            
        label_info = self.labels[index]
        
        return {
            "img": image_3ch.transpose(2, 0, 1),
            "cls": label_info.get('cls', np.array([])),
            "bboxes": label_info.get('bboxes', np.empty((0, 4))),
            "keypoints": label_info.get('keypoints', np.empty((0, 9))),
            "im_file": f"{Path(zarr_path).stem}_frame_{frame_idx}",
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (None, None) 
        }

def create_zarr_dataset(config: Union[ZarrDatasetConfig, Dict], mode: str) -> ZarrYOLODataset:
    if isinstance(config, dict): config = ZarrDatasetConfig(**config)
    return ZarrYOLODataset(config, mode)