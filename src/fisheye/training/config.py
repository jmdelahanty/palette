# src/fisheye/training/config.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pathlib import Path
import yaml

class SamplingStrategy(str, Enum):
    BALANCED = "balanced"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"

class SourceType(str, Enum):
    """Detection source types"""
    DETECT = "detect"
    FILTERED = "filtered"
    INTERPOLATED = "interpolated"

class DatasetSplit(BaseModel):
    """Train/val split configuration"""
    train: float = Field(0.8, gt=0.0, lt=1.0)
    val: float = Field(0.2, gt=0.0, lt=1.0)
    
    @field_validator('val')
    @classmethod
    def check_split_sum(cls, v, info):
        train = info.data.get('train', 0.8)
        if abs(train + v - 1.0) > 0.001:
            raise ValueError(f"train ({train}) + val ({v}) must equal 1.0")
        return v

class DatasetConfig(BaseModel):
    """Configuration for a single dataset"""
    zarr_path: Path
    source_type: SourceType = SourceType.FILTERED
    split: Optional[DatasetSplit] = None
    
    @field_validator('zarr_path')
    @classmethod
    def check_zarr_path(cls, v):
        v = Path(v)
        if not v.is_dir():
            raise ValueError(f"Path '{v}' is not a valid directory")
        if not (v / 'zarr.json').exists() and not (v / '.zgroup').exists():
            raise ValueError(f"Path '{v}' is not a valid Zarr directory")
        return v

class TrainingParams(BaseModel):
    """Model & Training Hyperparameters"""
    model: str
    epochs: int = Field(..., gt=0)
    batch: int = Field(..., gt=0)
    imgsz: int = Field(..., gt=0)
    lr0: float = Field(..., gt=0)
    momentum: float
    weight_decay: float
    patience: int
    device: str
    project: Optional[str] = None

class DetectConfig(BaseModel):
    """Flat configuration for detection training"""
    # Dummy YOLO fields
    train: Path
    val: Path
    nc: int
    names: List[str]
    
    # Dataset configuration (flat, not nested!)
    datasets: Dict[str, DatasetConfig]
    task: str = Field(..., pattern="^(detect|pose)$")
    random_seed: int = 42
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED
    dataset_weights: Optional[Dict[str, float]] = None
    
    # Training parameters
    training_params: TrainingParams
    
    @field_validator('datasets')
    @classmethod
    def check_datasets_not_empty(cls, v):
        if v is None or len(v) == 0:
            raise ValueError("'datasets' cannot be empty")
        return v
    
    @field_validator('dataset_weights')
    @classmethod
    def check_weights_match_strategy(cls, v, info):
        strategy = info.data.get('sampling_strategy')
        if strategy == SamplingStrategy.WEIGHTED:
            if v is None or len(v) == 0:
                raise ValueError("dataset_weights required when using 'weighted' sampling")
        return v

    @classmethod
    def from_yaml(cls, path: Path):
        """Loads and validates configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class PoseConfig(DetectConfig):
    """Configuration for pose estimation task"""
    kpt_shape: Tuple[int, int]