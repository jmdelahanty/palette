from pydantic import BaseModel, Field, validator, DirectoryPath
from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"

class FrameData(BaseModel):
    frame_idx: int = Field(..., ge=0)
    keypoints: List[KeypointData]
    has_ball: bool = Field(default=False)
    ball_keypoint: Optional[KeypointData] = None
    split: DataSplit = Field(default=DataSplit.TRAIN)

class YOLOBoundingBox(BaseModel):
    """Pydantic model for YOLO format bounding box"""
    center_x: float = Field(..., ge=0.0, le=1.0)
    center_y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

class KeypointData(BaseModel):
    """Pydantic model for keypoint information"""
    x: float = Field(..., ge=0.0)
    y: float = Field(..., ge=0.0)
    visibility: int = Field(default=1, ge=0, le=2)

class ConverterConfig(BaseModel):
    """Pydantic model for converter configuration"""
    input_folder: Path = Field(
        ...,
        description="Input folder containing videos and labels"
    )
    camera_name: str = Field(
        ...,
        min_length=1,
        description="Name of the camera"
    )
    output_folder: Path = Field(
        ...,
        description="Output folder for YOLO format data"
    )
    num_keypoints: int = Field(
        ...,
        gt=0,
        description="Number of keypoints to process"
    )
    margin: int = Field(
        default=40,
        gt=0,
        description="Margin for bounding box"
    )
    ball_bb_size: int = Field(
        default=55,
        gt=0,
        description="Ball bounding box size"
    )
    train_split: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Train/validation split ratio"
    )
    
    @validator('input_folder', 'output_folder')
    def validate_folders(cls, v):
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('camera_name')
    def validate_camera(cls, v, values):
        input_folder = values.get('input_folder')
        if input_folder:
            video_path = input_folder / "movies" / f"{v}.mp4"
            if not video_path.exists():
                raise ValueError(f"Video file not found: {video_path}")
        return v
    
    @validator('train_split')
    def validate_train_split(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("Train split must be between 0 and 1")
        return v

class VideoData(BaseModel):
    """Pydantic model for video properties"""
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0)
    total_frames: int = Field(..., gt=0)

    class Config:
        arbitrary_types_allowed = True

class FrameData(BaseModel):
    """Pydantic model for frame data"""
    frame_idx: int = Field(..., ge=0)
    keypoints: List[KeypointData]
    has_ball: bool = Field(default=False)
    ball_keypoint: Optional[KeypointData] = None
    
    @validator('keypoints')
    def validate_keypoints(cls, v):
        if not v:
            raise ValueError("Keypoints list cannot be empty")
        return v

class YOLOConverter:
    def __init__(self, config: Union[ConverterConfig, dict]):
        """
        Initialize the YOLO data converter with validated configuration
        """
        self.logger = logging.getLogger(__name__)
        # Validate config if dict is provided
        self.config = (
            config if isinstance(config, ConverterConfig)
            else ConverterConfig(**config)
        )
        
        # Initialize other attributes
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.labels: Dict = {}
        self.video_reader = None
    
    def setup(self) -> None:
        """Set up directories and load data"""
        self.logger.info("Setting up YOLO converter...")
        
        # Create directories
        self._create_directories()
        
        # Load video and labels
        self._load_data()
        
        self.logger.info("Setup complete")
        
        # Log some statistics
        self.logger.info(f"Found {len(self.labels)} labeled frames")
        self.logger.info(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        
    def validate_frame_data(self, 
                          frame: np.ndarray, 
                          labels_one_frame: np.ndarray,
                          frame_idx: int) -> FrameData:
        """Validate frame data using Pydantic model"""
        keypoints = [
            KeypointData(
                x=float(x),
                y=float(y),
                visibility=1
            )
            for x, y in labels_one_frame[:self.config.num_keypoints-1]
        ]
        
        return FrameData(
            frame_idx=frame_idx,
            keypoints=keypoints,
            has_ball=has_ball,
            ball_keypoint=ball_keypoint
        )
    
    def _create_yolo_rat_line(self, frame_data: FrameData) -> str:
        """Create YOLO format line for rat keypoints with validated data"""
        keypoints = np.array([[kp.x, kp.y] for kp in frame_data.keypoints])
        
        # Normalize coordinates
        rat_x = keypoints[:, 0] / self.frame_width
        rat_y = keypoints[:, 1] / self.frame_height
        
        margin_x = self.config.margin / self.frame_width
        margin_y = self.config.margin / self.frame_height
        
        # Calculate bounding box with validation
        bbox = YOLOBoundingBox(
            center_x=((np.min(rat_x) + np.max(rat_x)) / 2.0),
            center_y=((np.min(rat_y) + np.max(rat_y)) / 2.0),
            width=(np.max(rat_x) - np.min(rat_x) + 2 * margin_x),
            height=(np.max(rat_y) - np.min(rat_y) + 2 * margin_y)
        )
        
        # Create YOLO format line
        line = f"0 {bbox.center_x} {bbox.center_y} {bbox.width} {bbox.height}"
        for x, y in zip(rat_x, rat_y):
            line += f" {x} {y} 1"
        return line
    
    def _create_yolo_ball_line(self, ball_keypoint: KeypointData) -> str:
        """Create YOLO format line for ball keypoint with validated data"""
        ball_center_x = ball_keypoint.x / self.frame_width
        ball_center_y = ball_keypoint.y / self.frame_height
        ball_w = self.config.ball_bb_size / self.frame_width
        ball_h = self.config.ball_bb_size / self.frame_height
        
        bbox = YOLOBoundingBox(
            center_x=ball_center_x,
            center_y=ball_center_y,
            width=ball_w,
            height=ball_h
    )
    
    line = f"1 {bbox.center_x} {bbox.center_y} {bbox.width} {bbox.height}"
    # Add keypoints (only first point is the ball, rest are zeros)
    line += f" {ball_center_x} {ball_center_y} 1"
    line += " 0 0 0" * (self.config.num_keypoints - 1)
    return line

    def convert_frame(self, 
                     frame: np.ndarray, 
                     labels_one_frame: np.ndarray,
                     frame_idx: int,
                     split: str = 'train',
                     save_format: str = 'jpg') -> None:
        """Convert a single frame to YOLO format with validation"""
        try:
            # Validate frame data
            frame_data = self.validate_frame_data(
                frame, labels_one_frame, frame_idx
            )
            
            image_save_dir = self.config.output_folder / 'images' / split
            label_save_dir = self.config.output_folder / 'labels' / split
            
            # Save image
            image_path = image_save_dir / f"{frame_idx}.{save_format}"
            cv2.imwrite(str(image_path), frame)
            
            # Save labels with validation
            with open(label_save_dir / f"{frame_idx}.txt", 'w') as f:
                # Write rat keypoints
                rat_line = self._create_yolo_rat_line(frame_data)
                f.write(rat_line + '\n')
                
                # Write ball keypoint if present
                if frame_data.has_ball and frame_data.ball_keypoint:
                    ball_line = self._create_yolo_ball_line(frame_data.ball_keypoint)
                    f.write(ball_line + '\n')
                    
        except ValueError as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert video and labels to YOLO format')
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-k', '--num_keypoints', type=int, required=True)
    parser.add_argument('--margin', type=int, default=40)
    parser.add_argument('--ball_bb_size', type=int, default=55)
    parser.add_argument('--train_split', type=float, default=0.9)
    
    args = parser.parse_args()
    
    try:
        # Validate configuration using Pydantic
        config = ConverterConfig(
            input_folder=Path(args.input_folder),
            camera_name=args.name,
            output_folder=Path(args.output_folder),
            num_keypoints=args.num_keypoints,
            margin=args.margin,
            ball_bb_size=args.ball_bb_size,
            train_split=args.train_split
        )
        
        converter = YOLOConverter(config)
        converter.setup()
        converter.convert_dataset()
        print(f"Data saved to: {args.output_folder}")
        
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()