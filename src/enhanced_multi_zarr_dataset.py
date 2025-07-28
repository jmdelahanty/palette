# enhanced_multi_zarr_dataset.py
# Professional-grade multi-zarr dataset implementation

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SamplingStrategy(Enum):
    """Sampling strategies for multi-dataset training."""
    BALANCED = "balanced"           # Equal representation from each dataset
    WEIGHTED = "weighted"           # Custom weights per dataset
    PROPORTIONAL = "proportional"   # Proportional to dataset size
    QUALITY_WEIGHTED = "quality"    # Weight by tracking success rate

@dataclass
class DatasetMetadata:
    """Metadata for a single zarr dataset."""
    path: str
    name: str
    total_frames: int
    valid_frames: int
    tracking_success_rate: float
    data_format: str
    image_shape: Tuple[int, int]
    column_names: List[str]
    enhanced_features: bool = False

@dataclass
class MultiDatasetConfig:
    """Comprehensive configuration for multi-dataset YOLO training."""
    
    # ===== REQUIRED FIELDS =====
    zarr_paths: List[str]
    
    # ===== YOLO TRAINING PARAMETERS =====
    # Standard YOLO parameters (passed to ultralytics)
    nc: int = 1  # Number of classes
    names: List[str] = None  # Class names
    train: str = "./"  # Placeholder - handled by custom dataset
    val: str = "./"    # Placeholder - handled by custom dataset
    
    # ===== MULTI-ZARR DATASET PARAMETERS =====
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED
    dataset_weights: Optional[Dict[str, float]] = None
    split_ratio: float = 0.8
    random_seed: int = 42
    task: str = 'detect'  # 'detect' or 'pose'
    target_size: Optional[int] = None
    min_confidence: float = 0.0
    balance_across_videos: bool = True
    
    # ===== TRAINING CONFIGURATION =====
    training: Optional[Dict] = None  # Training recommendations (epochs, batch_size, etc.)
    
    # ===== PERFORMANCE OPTIMIZATION =====
    performance: Optional[Dict] = None  # Performance settings
    
    # ===== MONITORING AND DEBUGGING =====
    monitoring: Optional[Dict] = None  # Monitoring settings
    
    # ===== DATA QUALITY CONTROLS =====
    quality_controls: Optional[Dict] = None  # Quality control settings
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default class names if not provided
        if self.names is None:
            self.names = ['fish']
        
        # Ensure we have the right number of class names
        if len(self.names) != self.nc:
            if self.nc == 1 and not self.names:
                self.names = ['fish']
            elif len(self.names) != self.nc:
                raise ValueError(f"Number of class names ({len(self.names)}) doesn't match nc ({self.nc})")
        
        # Set default training parameters
        if self.training is None:
            self.training = {
                'epochs': 100,
                'batch_size': 16,
                'patience': 50,
                'amp': True
            }
        
        # Set default performance parameters
        if self.performance is None:
            self.performance = {
                'recommended_batch_sizes': {
                    'gpu_8gb': 16,
                    'gpu_12gb': 24,
                    'gpu_16gb': 32,
                    'gpu_24gb': 48
                }
            }
        
        # Set default monitoring parameters
        if self.monitoring is None:
            self.monitoring = {
                'track_per_dataset_metrics': True,
                'log_sampling_stats': True,
                'create_plots': True
            }
        
        # Set default quality controls
        if self.quality_controls is None:
            self.quality_controls = {
                'require_valid_keypoints': False
            }
    
    def get_yolo_params(self) -> Dict:
        """Extract parameters for YOLO trainer."""
        return {
            'nc': self.nc,
            'names': self.names,
            'train': self.train,
            'val': self.val
        }
    
    def get_dataset_params(self) -> Dict:
        """Extract parameters for dataset creation."""
        return {
            'zarr_paths': self.zarr_paths,
            'sampling_strategy': self.sampling_strategy,
            'dataset_weights': self.dataset_weights,
            'split_ratio': self.split_ratio,
            'random_seed': self.random_seed,
            'task': self.task,
            'target_size': self.target_size,
            'min_confidence': self.min_confidence,
            'balance_across_videos': self.balance_across_videos
        }
    
    def get_training_params(self) -> Dict:
        """Extract training parameters."""
        return self.training or {}
    
    def save_full_config(self, path: str):
        """Save complete configuration with documentation."""
        config_dict = {
            '# COMPREHENSIVE MULTI-ZARR YOLO CONFIGURATION': None,
            '# This file contains all parameters for training': None,
            '# Generated at': datetime.utcnow().isoformat(),
            
            '# ===== YOLO TRAINING PARAMETERS =====': None,
            'nc': self.nc,
            'names': self.names,
            'train': self.train,
            'val': self.val,
            
            '# ===== MULTI-ZARR DATASET PARAMETERS =====': None,
            'zarr_paths': self.zarr_paths,
            'task': self.task,
            'target_size': self.target_size,
            'split_ratio': self.split_ratio,
            'random_seed': self.random_seed,
            'sampling_strategy': self.sampling_strategy.value if hasattr(self.sampling_strategy, 'value') else self.sampling_strategy,
            'min_confidence': self.min_confidence,
            'balance_across_videos': self.balance_across_videos,
            
            '# ===== TRAINING CONFIGURATION =====': None,
            'training': self.training,
            
            '# ===== PERFORMANCE OPTIMIZATION =====': None,
            'performance': self.performance,
            
            '# ===== MONITORING AND DEBUGGING =====': None,
            'monitoring': self.monitoring,
            
            '# ===== DATA QUALITY CONTROLS =====': None,
            'quality_controls': self.quality_controls
        }
        
        # Remove comment keys and save
        clean_dict = {k: v for k, v in config_dict.items() if not k.startswith('#') and v is not None}
        
        with open(path, 'w') as f:
            yaml.dump(clean_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"📝 Complete configuration saved to: {path}")

def load_training_config(config_path: str) -> MultiDatasetConfig:
    """Load multi-zarr training configuration."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['zarr_paths']
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Missing required config field: {field}")
    
    # Define the parameters that MultiDatasetConfig actually accepts
    known_params = {
        'zarr_paths', 'sampling_strategy', 'dataset_weights', 'split_ratio', 
        'random_seed', 'task', 'target_size', 'min_confidence', 'balance_across_videos'
    }
    
    # Filter config dict to only include known parameters
    filtered_config = {k: v for k, v in config_dict.items() if k in known_params}
    
    # Log what we're filtering out for transparency
    filtered_out = {k: v for k, v in config_dict.items() if k not in known_params and k not in ['nc', 'names', 'train', 'val']}
    if filtered_out:
        logger.info(f"📝 Additional config sections (preserved in file): {list(filtered_out.keys())}")
    
    # Convert sampling strategy if provided
    if 'sampling_strategy' in filtered_config:
        if isinstance(filtered_config['sampling_strategy'], str):
            try:
                filtered_config['sampling_strategy'] = SamplingStrategy(filtered_config['sampling_strategy'])
            except ValueError:
                logger.warning(f"Unknown sampling strategy: {filtered_config['sampling_strategy']}, using balanced")
                filtered_config['sampling_strategy'] = SamplingStrategy.BALANCED
    
    # Set defaults for MultiDatasetConfig
    filtered_config.setdefault('sampling_strategy', SamplingStrategy.BALANCED)
    filtered_config.setdefault('task', 'detect')
    filtered_config.setdefault('split_ratio', 0.8)
    filtered_config.setdefault('random_seed', 42)
    
    return MultiDatasetConfig(**filtered_config)

class CompatibilityValidator:
    """Validates compatibility across multiple zarr files."""
    
    @staticmethod
    def validate_zarr_compatibility(zarr_paths: List[str]) -> Dict:
        """Comprehensive compatibility validation."""
        logger.info(f"🔍 Validating compatibility of {len(zarr_paths)} zarr files...")
        
        metadata_list = []
        compatibility_issues = []
        
        for i, zarr_path in enumerate(zarr_paths):
            try:
                metadata = CompatibilityValidator._extract_metadata(zarr_path, i)
                metadata_list.append(metadata)
                logger.info(f"  ✅ {metadata.name}: {metadata.valid_frames}/{metadata.total_frames} valid frames "
                           f"({metadata.tracking_success_rate:.1f}% success)")
            except Exception as e:
                compatibility_issues.append(f"Zarr {i} ({Path(zarr_path).name}): {e}")
                logger.error(f"  ❌ Failed to process zarr {i}: {e}")
        
        if len(metadata_list) < 2:
            compatibility_issues.append("Need at least 2 valid zarr files")
        
        # Check format consistency
        if len(metadata_list) >= 2:
            formats = [m.data_format for m in metadata_list]
            if len(set(formats)) > 1:
                compatibility_issues.append(f"Mixed data formats: {set(formats)}")
            
            # Check image shape consistency
            shapes = [m.image_shape for m in metadata_list]
            if len(set(shapes)) > 1:
                compatibility_issues.append(f"Mixed image shapes: {set(shapes)}")
            
            # Check column consistency for enhanced format
            if all(m.data_format == 'enhanced' for m in metadata_list):
                all_columns = [set(m.column_names) for m in metadata_list]
                if not all(cols == all_columns[0] for cols in all_columns):
                    compatibility_issues.append("Enhanced format column names don't match")
        
        total_valid_frames = sum(m.valid_frames for m in metadata_list)
        
        return {
            'compatible': len(compatibility_issues) == 0,
            'issues': compatibility_issues,
            'metadata': metadata_list,
            'total_valid_frames': total_valid_frames,
            'common_format': metadata_list[0].data_format if metadata_list else None,
            'common_image_shape': metadata_list[0].image_shape if metadata_list else None
        }
    
    @staticmethod
    def _extract_metadata(zarr_path: str, index: int) -> DatasetMetadata:
        """Extract metadata from a single zarr file."""
        try:
            root = zarr.open(zarr_path, mode='r')
            
            # Check required structure
            required_paths = ['raw_video/images_ds', 'tracking/tracking_results']
            missing_paths = [path for path in required_paths if path not in root]
            if missing_paths:
                raise ValueError(f"Missing required paths: {missing_paths}")
            
            # Get tracking data
            tracking_results = root['tracking/tracking_results']
            column_names = tracking_results.attrs.get('column_names', [])
            
            if not column_names:
                raise ValueError("No column names found in tracking results")
            
            # Determine data format
            data_format = 'enhanced' if 'bbox_x_norm_ds' in column_names else 'original'
            
            # Calculate valid frames
            data = tracking_results[:]
            total_frames = data.shape[0]
            
            # Use same filtering logic as dataset
            valid_mask = ~np.isnan(data[:, 0])  # Valid heading
            if data_format == 'enhanced':
                col_map = {name: i for i, name in enumerate(column_names)}
                bbox_x_idx = col_map.get('bbox_x_norm_ds', col_map.get('bbox_x_norm'))
                bbox_y_idx = col_map.get('bbox_y_norm_ds', col_map.get('bbox_y_norm'))
                if bbox_x_idx is not None and bbox_y_idx is not None:
                    bbox_valid = ~np.isnan(data[:, bbox_x_idx]) & ~np.isnan(data[:, bbox_y_idx])
                    valid_mask = valid_mask & bbox_valid
            
            valid_frames = np.sum(valid_mask)
            tracking_success_rate = (valid_frames / total_frames) * 100 if total_frames > 0 else 0
            
            # Get image shape
            image_shape = root['raw_video/images_ds'].shape[1:]
            
            return DatasetMetadata(
                path=zarr_path,
                name=Path(zarr_path).stem,
                total_frames=total_frames,
                valid_frames=valid_frames,
                tracking_success_rate=tracking_success_rate,
                data_format=data_format,
                image_shape=image_shape,
                column_names=column_names,
                enhanced_features='enhanced_features' in root.get('tracking', {}).attrs
            )
            
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")

class GlobalIndexManager:
    """Manages global indexing across multiple datasets."""
    
    def __init__(self, metadata_list: List[DatasetMetadata], config: MultiDatasetConfig):
        self.metadata_list = metadata_list
        self.config = config
        self.global_indices = []
        self.dataset_stats = {}
        
        self._build_global_index()
    
    def _build_global_index(self):
        """Build global index mapping across all datasets."""
        logger.info("🔧 Building global index across datasets...")
        
        # Extract valid indices from each dataset
        raw_indices = {}
        for metadata in self.metadata_list:
            valid_indices = self._get_valid_indices(metadata)
            raw_indices[metadata.path] = valid_indices
            
            logger.info(f"  📊 {metadata.name}: {len(valid_indices)} valid samples")
        
        # Apply sampling strategy
        sampled_indices = self._apply_sampling_strategy(raw_indices)
        
        # Create global index list
        self.global_indices = []
        for zarr_path, indices in sampled_indices.items():
            for local_idx in indices:
                self.global_indices.append((zarr_path, local_idx))
        
        # Shuffle for good mixing
        np.random.seed(self.config.random_seed)
        np.random.shuffle(self.global_indices)
        
        # Calculate stats
        self._calculate_stats(sampled_indices)
        
        logger.info(f"  🎯 Total global samples: {len(self.global_indices)}")
    
    def _get_valid_indices(self, metadata: DatasetMetadata) -> List[int]:
        """Get valid frame indices for a dataset."""
        try:
            root = zarr.open(metadata.path, mode='r')
            tracking_data = root['tracking/tracking_results'][:]
            
            # Apply same filtering as dataset
            valid_mask = ~np.isnan(tracking_data[:, 0])  # Valid heading
            
            if metadata.data_format == 'enhanced':
                col_map = {name: i for i, name in enumerate(metadata.column_names)}
                bbox_x_idx = col_map.get('bbox_x_norm_ds', col_map.get('bbox_x_norm'))
                bbox_y_idx = col_map.get('bbox_y_norm_ds', col_map.get('bbox_y_norm'))
                
                if bbox_x_idx is not None and bbox_y_idx is not None:
                    bbox_valid = (~np.isnan(tracking_data[:, bbox_x_idx]) & 
                                 ~np.isnan(tracking_data[:, bbox_y_idx]))
                    valid_mask = valid_mask & bbox_valid
            
            # Apply confidence filtering if specified
            if self.config.min_confidence > 0 and metadata.data_format == 'enhanced':
                col_map = {name: i for i, name in enumerate(metadata.column_names)}
                conf_idx = col_map.get('confidence_score')
                if conf_idx is not None:
                    conf_valid = tracking_data[:, conf_idx] >= self.config.min_confidence
                    valid_mask = valid_mask & ~np.isnan(tracking_data[:, conf_idx]) & conf_valid
            
            return np.where(valid_mask)[0].tolist()
            
        except Exception as e:
            logger.warning(f"Error extracting valid indices from {metadata.name}: {e}")
            return []
    
    def _apply_sampling_strategy(self, raw_indices: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Apply the configured sampling strategy."""
        strategy = self.config.sampling_strategy
        
        if strategy == SamplingStrategy.PROPORTIONAL:
            # Use all available samples (proportional to dataset size)
            return raw_indices
        
        elif strategy == SamplingStrategy.BALANCED:
            # Balance across datasets
            min_samples = min(len(indices) for indices in raw_indices.values())
            balanced_indices = {}
            
            for zarr_path, indices in raw_indices.items():
                if len(indices) > min_samples:
                    # Subsample to balance
                    np.random.seed(self.config.random_seed)
                    sampled = np.random.choice(indices, min_samples, replace=False)
                    balanced_indices[zarr_path] = sampled.tolist()
                else:
                    balanced_indices[zarr_path] = indices
            
            return balanced_indices
        
        elif strategy == SamplingStrategy.WEIGHTED:
            # Apply custom weights
            if not self.config.dataset_weights:
                logger.warning("Weighted sampling requested but no weights provided, using proportional")
                return raw_indices
            
            weighted_indices = {}
            for zarr_path, indices in raw_indices.items():
                dataset_name = Path(zarr_path).stem
                weight = self.config.dataset_weights.get(dataset_name, 1.0)
                
                n_samples = int(len(indices) * weight)
                if n_samples > 0 and n_samples <= len(indices):
                    np.random.seed(self.config.random_seed)
                    sampled = np.random.choice(indices, n_samples, replace=False)
                    weighted_indices[zarr_path] = sampled.tolist()
                else:
                    weighted_indices[zarr_path] = indices
            
            return weighted_indices
        
        elif strategy == SamplingStrategy.QUALITY_WEIGHTED:
            # Weight by tracking success rate
            total_samples = sum(len(indices) for indices in raw_indices.values())
            quality_weighted_indices = {}
            
            for metadata in self.metadata_list:
                zarr_path = metadata.path
                indices = raw_indices[zarr_path]
                
                # Weight by success rate (higher success = more samples)
                quality_weight = metadata.tracking_success_rate / 100.0
                n_samples = int(len(indices) * quality_weight)
                
                if n_samples > 0:
                    np.random.seed(self.config.random_seed)
                    sampled = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
                    quality_weighted_indices[zarr_path] = sampled.tolist()
                else:
                    quality_weighted_indices[zarr_path] = indices[:1] if indices else []
            
            return quality_weighted_indices
        
        else:
            logger.warning(f"Unknown sampling strategy: {strategy}, using proportional")
            return raw_indices
    
    def _calculate_stats(self, sampled_indices: Dict[str, List[int]]):
        """Calculate dataset statistics."""
        total_samples = sum(len(indices) for indices in sampled_indices.values())
        
        self.dataset_stats = {
            'total_samples': total_samples,
            'per_dataset_counts': {},
            'per_dataset_percentages': {}
        }
        
        for zarr_path, indices in sampled_indices.items():
            dataset_name = Path(zarr_path).stem
            count = len(indices)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            
            self.dataset_stats['per_dataset_counts'][dataset_name] = count
            self.dataset_stats['per_dataset_percentages'][dataset_name] = percentage
    
    def get_train_val_split(self) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split global indices into train/val sets."""
        train_indices, val_indices = train_test_split(
            self.global_indices,
            train_size=self.config.split_ratio,
            random_state=self.config.random_seed,
            shuffle=True
        )
        
        return train_indices, val_indices

class EnhancedMultiZarrYOLODataset(Dataset):
    """
    Professional-grade multi-zarr YOLO dataset with advanced features.
    
    Features:
    - Runtime combination of multiple zarr files
    - Intelligent sampling strategies
    - Comprehensive validation and error handling
    - Performance monitoring and debugging
    - Flexible configuration system
    """
    
    def __init__(self, config: Union[MultiDatasetConfig, Dict], mode: str = 'train'):
        """
        Initialize the multi-zarr dataset.
        
        Args:
            config: Dataset configuration (MultiDatasetConfig or dict)
            mode: 'train' or 'val'
        """
        super().__init__()
        
        # Handle config
        if isinstance(config, dict):
            config = MultiDatasetConfig(**config)
        self.config = config
        self.mode = mode
        
        # Validate inputs
        if mode not in ['train', 'val']:
            raise ValueError("Mode must be 'train' or 'val'")
        
        if not config.zarr_paths:
            raise ValueError("No zarr paths provided")
        
        # Validate compatibility
        logger.info(f"🚀 Initializing Enhanced Multi-Zarr YOLO Dataset ({mode} mode)")
        self.compatibility_report = CompatibilityValidator.validate_zarr_compatibility(config.zarr_paths)
        
        if not self.compatibility_report['compatible']:
            raise ValueError(f"Zarr files not compatible: {self.compatibility_report['issues']}")
        
        # Setup dataset properties
        self.metadata_list = self.compatibility_report['metadata']
        self.data_format = self.compatibility_report['common_format']
        
        # Set task-specific properties
        self._setup_task_properties()
        
        # Open zarr files
        self.zarr_roots = {}
        for metadata in self.metadata_list:
            self.zarr_roots[metadata.path] = zarr.open(metadata.path, mode='r')
        
        # Setup column mappings
        self._setup_column_mappings()
        
        # Build global index
        self.index_manager = GlobalIndexManager(self.metadata_list, self.config)
        train_indices, val_indices = self.index_manager.get_train_val_split()
        
        # Set mode-specific indices
        if mode == 'train':
            self.indices = train_indices
        else:
            self.indices = val_indices
        
        # Log initialization summary
        self._log_initialization_summary()
    
    def _setup_task_properties(self):
        """Setup task-specific properties."""
        if self.config.task == 'detect':
            self.image_source = 'raw_video/images_ds'  # 640x640
            self.target_size = self.config.target_size or 640
        elif self.config.task == 'pose':
            self.image_source = 'crop_data/roi_images'  # 320x320
            self.target_size = self.config.target_size or 320
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
    
    def _setup_column_mappings(self):
        """Setup column mappings based on data format."""
        first_metadata = self.metadata_list[0]
        col_map = {name: i for i, name in enumerate(first_metadata.column_names)}
        
        if self.data_format == 'enhanced':
            self.coord_mappings = {
                'heading': col_map['heading_degrees'],
                'bbox_x': col_map['bbox_x_norm_ds'],
                'bbox_y': col_map['bbox_y_norm_ds'],
                'bbox_width': col_map['bbox_width_norm_ds'],
                'bbox_height': col_map['bbox_height_norm_ds'],
                'confidence': col_map.get('confidence_score')
            }
        else:  # original
            self.coord_mappings = {
                'heading': col_map['heading_degrees'],
                'bbox_x': col_map['bbox_x_norm'],
                'bbox_y': col_map['bbox_y_norm'],
                'bbox_width': None,
                'bbox_height': None,
                'confidence': None
            }
    
    def _log_initialization_summary(self):
        """Log comprehensive initialization summary."""
        stats = self.index_manager.dataset_stats
        
        logger.info(f"✅ Enhanced Multi-Zarr Dataset initialized successfully!")
        logger.info(f"   📊 Mode: {self.mode}")
        logger.info(f"   🎯 Task: {self.config.task}")
        logger.info(f"   📐 Target size: {self.target_size}x{self.target_size}")
        logger.info(f"   🎭 Sampling strategy: {self.config.sampling_strategy.value}")
        logger.info(f"   📈 Data format: {self.data_format}")
        logger.info(f"   📋 Total {self.mode} samples: {len(self.indices)}")
        
        logger.info(f"   📊 Dataset distribution:")
        for name, count in stats['per_dataset_counts'].items():
            percentage = stats['per_dataset_percentages'][name]
            logger.info(f"      {name}: {count} samples ({percentage:.1f}%)")
    
    def _get_bbox_data(self, zarr_path: str, local_frame_idx: int) -> Optional[np.ndarray]:
        """Extract bounding box data for a specific frame."""
        try:
            tracking_data = self.zarr_roots[zarr_path]['tracking/tracking_results'][local_frame_idx]
            
            # Get basic data
            heading = tracking_data[self.coord_mappings['heading']]
            bbox_x = tracking_data[self.coord_mappings['bbox_x']]
            bbox_y = tracking_data[self.coord_mappings['bbox_y']]
            
            if np.isnan(heading) or np.isnan(bbox_x) or np.isnan(bbox_y):
                return None
            
            if self.data_format == 'enhanced':
                # Enhanced format has pre-calculated dimensions
                bbox_width = tracking_data[self.coord_mappings['bbox_width']]
                bbox_height = tracking_data[self.coord_mappings['bbox_height']]
                
                if np.isnan(bbox_width) or np.isnan(bbox_height):
                    return None
                
                confidence = 1.0
                if self.coord_mappings['confidence'] is not None:
                    confidence = tracking_data[self.coord_mappings['confidence']]
                    if np.isnan(confidence):
                        confidence = 1.0
            else:
                # Original format - use reasonable defaults
                bbox_width = 0.05
                bbox_height = 0.05
                confidence = 1.0
            
            return np.array([0, bbox_x, bbox_y, bbox_width, bbox_height, confidence], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error extracting bbox data from {Path(zarr_path).stem} frame {local_frame_idx}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, index: int) -> Dict:
        """Get a training sample."""
        # Get global index
        zarr_path, local_frame_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        
        try:
            # Get image based on task
            image = root[self.image_source][local_frame_idx]
            
            # Prepare image for YOLO
            if image.ndim == 2:  # Grayscale
                image_3ch = np.stack([image, image, image], axis=-1)
            else:
                image_3ch = image
            
            if image_3ch.dtype != np.uint8:
                image_3ch = image_3ch.astype(np.uint8)
            
            # Resize if needed
            if image.shape[0] != self.target_size:
                import cv2
                image_3ch = cv2.resize(image_3ch, (self.target_size, self.target_size))
            
            # Transpose to CHW for PyTorch
            image_tensor = image_3ch.transpose(2, 0, 1)
            
            # Get bounding box data
            bbox_data = self._get_bbox_data(zarr_path, local_frame_idx)
            
            if bbox_data is not None:
                label = bbox_data[:5].reshape(1, -1)
            else:
                # Fallback for invalid data
                logger.warning(f"Using fallback bbox for {Path(zarr_path).stem} frame {local_frame_idx}")
                label = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
            
            # Ensure proper tensor shapes
            cls_labels = label[:, 0].astype(np.float32)
            bbox_coords = label[:, 1:5].astype(np.float32)
            
            # Validation
            assert cls_labels.ndim == 1, f"cls_labels must be 1D, got {cls_labels.ndim}D"
            assert bbox_coords.ndim == 2 and bbox_coords.shape[1] == 4, f"bbox_coords must be (N, 4)"
            
            # Add metadata
            dataset_name = Path(zarr_path).stem
            
            return {
                "img": image_tensor,
                "cls": cls_labels,
                "bboxes": bbox_coords,
                "im_file": f"{dataset_name}_frame_{local_frame_idx}",
                "ori_shape": (self.target_size, self.target_size),
                "ratio_pad": (1.0, (0.0, 0.0)),
                # Enhanced metadata
                "dataset_name": dataset_name,
                "zarr_path": zarr_path,
                "local_frame_idx": local_frame_idx
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {index} from {Path(zarr_path).stem}: {e}")
            # Return a safe fallback sample
            return self._get_fallback_sample(index)
    
    def _get_fallback_sample(self, index: int) -> Dict:
        """Generate a safe fallback sample."""
        # Create a black image
        image_tensor = np.zeros((3, self.target_size, self.target_size), dtype=np.float32)
        
        # Create fallback label
        cls_labels = np.array([0], dtype=np.float32)
        bbox_coords = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        
        return {
            "img": image_tensor,
            "cls": cls_labels,
            "bboxes": bbox_coords,
            "im_file": f"fallback_sample_{index}",
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (1.0, (0.0, 0.0)),
            "dataset_name": "fallback",
            "zarr_path": "fallback",
            "local_frame_idx": -1
        }
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        base_stats = self.index_manager.dataset_stats.copy()
        
        # Add mode-specific stats
        mode_stats = {'total_samples': 0, 'per_dataset_counts': {}, 'per_dataset_percentages': {}}
        
        for zarr_path, local_idx in self.indices:
            dataset_name = Path(zarr_path).stem
            mode_stats['per_dataset_counts'][dataset_name] = mode_stats['per_dataset_counts'].get(dataset_name, 0) + 1
        
        mode_stats['total_samples'] = len(self.indices)
        
        for name, count in mode_stats['per_dataset_counts'].items():
            mode_stats['per_dataset_percentages'][name] = (count / mode_stats['total_samples'] * 100) if mode_stats['total_samples'] > 0 else 0
        
        return {
            'mode': self.mode,
            'task': self.config.task,
            'sampling_strategy': self.config.sampling_strategy.value,
            'mode_specific': mode_stats,
            'global_stats': base_stats,
            'metadata': [
                {
                    'name': m.name,
                    'total_frames': m.total_frames,
                    'valid_frames': m.valid_frames,
                    'success_rate': m.tracking_success_rate
                }
                for m in self.metadata_list
            ]
        }
    
    def save_dataset_info(self, output_path: str):
        """Save comprehensive dataset information."""
        info = {
            'dataset_type': 'EnhancedMultiZarrYOLODataset',
            'config': {
                'zarr_paths': self.config.zarr_paths,
                'sampling_strategy': self.config.sampling_strategy.value,
                'dataset_weights': self.config.dataset_weights,
                'split_ratio': self.config.split_ratio,
                'random_seed': self.config.random_seed,
                'task': self.config.task,
                'target_size': self.config.target_size,
                'min_confidence': self.config.min_confidence,
                'balance_across_videos': self.config.balance_across_videos
            },
            'compatibility_report': self.compatibility_report,
            'statistics': self.get_dataset_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        logger.info(f"📋 Dataset info saved to: {output_path}")

# Factory function for easy dataset creation
def create_multi_zarr_dataset(zarr_paths: List[str], mode: str = 'train', **kwargs) -> EnhancedMultiZarrYOLODataset:
    """
    Factory function to create enhanced multi-zarr dataset.
    
    Args:
        zarr_paths: List of paths to zarr files
        mode: 'train' or 'val'
        **kwargs: Additional configuration options
    
    Returns:
        EnhancedMultiZarrYOLODataset instance
    """
    config = MultiDatasetConfig(zarr_paths=zarr_paths, **kwargs)
    return EnhancedMultiZarrYOLODataset(config, mode)