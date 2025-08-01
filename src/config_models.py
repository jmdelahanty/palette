from pydantic import BaseModel, Field, validator, FilePath, DirectoryPath
from typing import List, Optional, Tuple
from enum import Enum
import yaml

class SamplingStrategy(str, Enum):
    BALANCED = "balanced"
    PROPORTIONAL = "proportional"
    WEIGHTED = "weighted"

class DataConfig(BaseModel):
    """Configuration for the Zarr dataset loader."""
    zarr_paths: List[FilePath]
    task: str = Field(..., pattern="^(detect|pose)$")
    split_ratio: float = Field(0.8, gt=0.0, lt=1.0)
    random_seed: int = 42
    sampling_strategy: SamplingStrategy = SamplingStrategy.BALANCED

    @validator('zarr_paths', each_item=True)
    def check_zarr_path(cls, v):
        if not v.is_dir() or not (v / '.zgroup').exists():
            raise ValueError(f"Path '{v}' is not a valid Zarr directory.")
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

class BaseConfig(BaseModel):
    """Base model for the main configuration."""
    train: FilePath
    val: FilePath
    nc: int
    names: List[str]
    data_config: DataConfig
    training_params: TrainingParams

    @classmethod
    def from_yaml(cls, path: FilePath):
        """Loads and validates configuration from a YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class DetectConfig(BaseConfig):
    """Validated configuration for the detection task."""
    pass

class PoseConfig(BaseConfig):
    """Validated configuration for the pose estimation task."""
    kpt_shape: Tuple[int, int]