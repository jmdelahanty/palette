# src/fisheye/training/config.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
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
    MANUAL = "manual"

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
    input_format: Literal["gray", "rgb"] = "gray"
    keypoint_run: Optional[str] = None
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
    imgsz: Union[int, Tuple[int, int], List[int]] = Field(...)
    lr0: float = Field(..., gt=0)
    momentum: float
    weight_decay: float
    patience: int
    device: str
    rect: bool = False
    project: Optional[str] = None
    max_det: Optional[int] = Field(None, ge=1)
    conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    seg_loss: Optional[str] = Field(
        default=None,
        description="Optional segmentation loss override (e.g., 'bce_dice').",
    )
    bce_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Blend weight for BCE term when using BCE+Dice loss.",
    )

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, value):
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("imgsz must be positive")
            return value
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("imgsz list/tuple must specify [height, width].")
            ints = [int(v) for v in value]
            if any(v <= 0 for v in ints):
                raise ValueError("imgsz height/width must be positive.")
            return ints
        raise TypeError("imgsz must be an int or a [height, width] list/tuple.")


class EyeMaskTrainingParams(TrainingParams):
    """Training parameters for eye-mask segmentation pipelines."""

    label_source: Literal["yolo", "manual"] = "yolo"
    label_mode: Literal["union", "lr"] = "union"
    eye_masks_run: Optional[str] = None
    eye_masks_method: Optional[str] = None


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
    allow_source_mismatch: bool = False
    
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


class CacheBackend(str, Enum):
    THREAD = "thread"
    PROCESS = "process"


class EyeMaskDatasetConfig(BaseModel):
    """Configuration for an eye-mask segmentation dataset sourced from a Zarr store."""
    zarr_path: Path
    crop_run: Optional[str] = None
    mask_run: Optional[str] = None
    mask_preference: Literal["auto", "refined_probs", "raw_probs", "binary"] = "auto"
    split: Optional[DatasetSplit] = None
    min_positive_area: int = Field(0, ge=0)
    include_empty: bool = True
    require_both_eyes: bool = False
    include_background_negatives: bool = False
    background_negative_ratio: float = Field(1.0, gt=0.0)
    background_run: Optional[str] = None
    background_from_downsampled: bool = False

    @field_validator('zarr_path')
    @classmethod
    def check_zarr_path(cls, v: Path) -> Path:
        v = Path(v)
        if not v.is_dir():
            raise ValueError(f"Path '{v}' is not a valid directory")
        if not (v / 'zarr.json').exists() and not (v / '.zgroup').exists():
            raise ValueError(f"Path '{v}' is not a valid Zarr directory")
        return v


class EyeMaskCacheConfig(BaseModel):
    """Optional cache settings for eye-mask training data."""

    enabled: bool = False
    directory: Path = Path("runs/cache/eye_masks")
    reuse_existing: bool = True
    workers: int = Field(4, ge=1)
    backend: CacheBackend = CacheBackend.THREAD

    @field_validator("directory")
    @classmethod
    def ensure_directory(cls, v: Path, info) -> Path:
        v = Path(v)
        enabled = info.data.get("enabled", False)
        if enabled:
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)
            if not v.is_dir():
                raise ValueError(f"Cache directory '{v}' is not a directory")
        return v


class EyeMaskTrainingConfig(BaseModel):
    """Configuration for training eye-mask segmentation models."""
    datasets: Dict[str, EyeMaskDatasetConfig]
    names: List[str] = Field(default_factory=lambda: ["eye"])
    nc: int = Field(1, gt=0)
    random_seed: int = 42
    target_size: int = Field(160, gt=0)
    default_split: DatasetSplit = DatasetSplit()
    training_params: EyeMaskTrainingParams
    num_workers: int = Field(8, ge=0)
    cache: EyeMaskCacheConfig = EyeMaskCacheConfig()
    mask_smoothing_kernel: int = Field(0, ge=0, description="Morphological kernel size (odd) for optional mask smoothing")
    edge_enhancement: Optional[str] = Field(
        default=None,
        description="Optional edge map added as an extra channel ('sobel', 'laplacian', 'canny')",
    )
    binarization_threshold: Optional[int] = Field(
        default=None,
        ge=0,
        le=255,
        description="Optional fixed intensity threshold (0-255) for an additional binarized input channel",
    )

    @field_validator('datasets')
    @classmethod
    def validate_datasets(cls, v: Dict[str, EyeMaskDatasetConfig]) -> Dict[str, EyeMaskDatasetConfig]:
        if not v:
            raise ValueError("'datasets' cannot be empty")
        return v

    @field_validator('names')
    @classmethod
    def validate_names(cls, v: List[str], info) -> List[str]:
        nc = info.data.get('nc', 1)
        if len(v) != nc:
            raise ValueError(f"Length of 'names' ({len(v)}) must match 'nc' ({nc})")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "EyeMaskTrainingConfig":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
