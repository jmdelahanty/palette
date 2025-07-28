# zarr_yolo_dataset_bbox.py (Updated for Enhanced Pipeline - Index-based Access)

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
from pydantic import BaseModel, ValidationError, field_validator
from sklearn.model_selection import train_test_split


class TrackingSummary(BaseModel):
    total_frames: int; frames_tracked: int; percent_tracked: float

class TrackingResultsAttrs(BaseModel):
    column_names: List[str]
    @field_validator('column_names')
    @classmethod
    def check_column_count(cls, v: List[str]) -> List[str]:
        if len(v) < 9: raise ValueError(f"Expected at least 9 columns, found {len(v)}."); return v

def validate_zarr_structure(root: zarr.hierarchy.Group):
    print("🔬 Running Zarr structure validation...")
    required_paths = {
        'raw_video': 'group', 'raw_video/images_ds': 'dataset',
        'crop_data': 'group', 'crop_data/roi_images': 'dataset', 
        'tracking': 'group', 'tracking/tracking_results': 'dataset'
    }
    for path, type_name in required_paths.items():
        if path not in root: raise ValueError(f"Validation Error: Required {type_name} '{path}' not found.")
    
    # Validate dimensions
    if root['raw_video/images_ds'].ndim != 3: raise ValueError(f"Validation Error: 'images_ds' should have 3 dimensions, has {root['raw_video/images_ds'].ndim}.")
    if root['crop_data/roi_images'].ndim != 3: raise ValueError(f"Validation Error: 'roi_images' should have 3 dimensions, has {root['crop_data/roi_images'].ndim}.")
    if root['tracking/tracking_results'].ndim != 2: raise ValueError(f"Validation Error: 'tracking_results' should have 2 dimensions, has {root['tracking/tracking_results'].ndim}.")
    
    try:
        if 'summary_statistics' not in root['tracking'].attrs: raise ValueError("Missing 'summary_statistics' in '/tracking'.")
        TrackingSummary.model_validate(root['tracking'].attrs['summary_statistics'])
        if 'column_names' not in root['tracking/tracking_results'].attrs: raise ValueError("Missing 'column_names' in '/tracking/tracking_results'.")
        TrackingResultsAttrs.model_validate(root['tracking/tracking_results'].attrs)
    except ValidationError as e: raise ValueError(f"Zarr attribute validation failed:\n{e}")
    print("✅ Zarr structure is valid.")


def detect_data_format(column_names):
    """Detect whether this is original or enhanced pipeline data."""
    if 'bbox_x_norm_ds' in column_names:
        return 'enhanced'
    elif 'bbox_x_norm' in column_names:
        return 'original'
    else:
        raise ValueError("Unknown data format - missing expected bbox coordinate columns")


def get_column_mappings(column_names, data_format):
    """Get column index mappings for the detected data format."""
    col_map = {name: i for i, name in enumerate(column_names)}
    
    if data_format == 'enhanced':
        return {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm_ds'],      # Use downsampled coords for 640x640 training
            'bbox_y': col_map['bbox_y_norm_ds'],
            'bbox_width': col_map['bbox_width_norm_ds'],
            'bbox_height': col_map['bbox_height_norm_ds'],
            'confidence': col_map.get('confidence_score', None)  # May not exist in all versions
        }
    else:  # original format
        return {
            'heading': col_map['heading_degrees'],
            'bbox_x': col_map['bbox_x_norm'],
            'bbox_y': col_map['bbox_y_norm'],
            'bbox_width': None,  # Not available in original format
            'bbox_height': None,  # Not available in original format
            'confidence': None   # Not available in original format
        }


class ZarrYOLODataset(Dataset):
    """
    Enhanced YOLO dataset with index-based access to comprehensive tracking data.
    
    Supports two main use cases:
    - Detection: 640x640 full images for fish detection anywhere in the scene
    - Pose: 320x320 ROI crops for detailed keypoint estimation on pre-located fish
    
    Works with both original and enhanced pipeline formats.
    """
    def __init__(self, zarr_path: str, mode: str = 'train', split_ratio: float = 0.8, 
                 random_seed: int = 42, task: str = 'detect', target_size: int = None):
        super().__init__()
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'."
        assert task in ['detect', 'pose'], "Task must be 'detect' (640x640 full images) or 'pose' (320x320 ROI crops)."
        
        self.root = zarr.open(zarr_path, mode='r')
        validate_zarr_structure(self.root)
        
        self.task = task
        
        # Set appropriate defaults based on task
        if task == 'detect':
            self.image_source = 'full'  # 640x640 full images
            self.target_size = target_size if target_size else 640
            print(f"🎯 Detection task: Using full 640×640 images for fish detection")
        else:  # pose
            self.image_source = 'roi'   # 320x320 ROI crops  
            self.target_size = target_size if target_size else 320
            print(f"🧘 Pose task: Using 320×320 ROI crops for keypoint estimation")
        
        # Detect data format and get column mappings
        tracking_results = self.root['tracking/tracking_results']
        column_names = tracking_results.attrs['column_names']
        self.data_format = detect_data_format(column_names)
        self.col_map = get_column_mappings(column_names, self.data_format)
        
        print(f"📊 Detected {self.data_format} pipeline format")
        
        # Find valid tracking indices
        data = tracking_results[:]
        valid_mask = ~np.isnan(data[:, self.col_map['heading']])  # Valid heading indicates successful tracking
        
        # For enhanced format, also check bbox data availability
        if self.data_format == 'enhanced':
            bbox_valid = ~np.isnan(data[:, self.col_map['bbox_x']]) & ~np.isnan(data[:, self.col_map['bbox_y']])
            valid_mask = valid_mask & bbox_valid
        
        all_valid_indices = np.where(valid_mask)[0]
        print(f"📈 Found {len(all_valid_indices)} frames with valid tracking data")
        
        # Split into train/val
        train_indices, val_indices = train_test_split(
            all_valid_indices,
            train_size=split_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        if mode == 'train':
            self.indices = train_indices
            print(f"🚂 Initialized 'train' dataset with {len(self.indices)} samples.")
        else: # mode == 'val'
            self.indices = val_indices
            print(f"✅ Initialized 'val' dataset with {len(self.indices)} samples.")

        # Pre-cache labels for plotting utility (lightweight - just indices and basic info)
        print(f"💾 Pre-caching {len(self.indices)} sample metadata...")
        self.labels = []
        
        for i in range(len(self.indices)):
            zarr_index = self.indices[i]
            bbox_data = self._get_bbox_data(zarr_index)
            
            if bbox_data is not None:
                cls = np.array([0.])  # Fish class
                bboxes = bbox_data[1:5].reshape(1, -1)  # [x, y, w, h]
                self.labels.append({"bboxes": bboxes, "cls": cls})
            else:
                # Fallback for invalid data
                fallback_bbox = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
                self.labels.append({"bboxes": fallback_bbox, "cls": np.array([0.])})
        
        print(f"✅ Dataset ready with {len(self.labels)} samples from {self.data_format} pipeline")

    def _get_bbox_data(self, zarr_index):
        """Extract bounding box data for a given frame index."""
        tracking_data = self.root['tracking/tracking_results'][zarr_index]
        
        # Get basic data
        heading = tracking_data[self.col_map['heading']]
        bbox_x = tracking_data[self.col_map['bbox_x']]
        bbox_y = tracking_data[self.col_map['bbox_y']]
        
        if np.isnan(heading) or np.isnan(bbox_x) or np.isnan(bbox_y):
            return None
        
        if self.data_format == 'enhanced':
            # Enhanced format has pre-calculated dimensions
            bbox_width = tracking_data[self.col_map['bbox_width']]
            bbox_height = tracking_data[self.col_map['bbox_height']]
            
            if np.isnan(bbox_width) or np.isnan(bbox_height):
                return None
                
            confidence = 1.0  # Default confidence
            if self.col_map['confidence'] is not None:
                confidence = tracking_data[self.col_map['confidence']]
                if np.isnan(confidence):
                    confidence = 1.0
                    
        else:
            # Original format - need to calculate dimensions
            # Use a reasonable default size (this could be improved with ROI analysis)
            bbox_width = 0.05   # 5% of image width
            bbox_height = 0.05  # 5% of image height
            confidence = 1.0
        
        # Return [class, center_x, center_y, width, height, confidence]
        return np.array([0, bbox_x, bbox_y, bbox_width, bbox_height, confidence], dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        # Map the item index to the actual Zarr frame index
        zarr_index = self.indices[index]
        
        # Get the appropriate image based on task
        if self.task == 'detect':
            # Detection: Use full 640x640 downsampled images
            image = self.root['raw_video/images_ds'][zarr_index]
            image_shape = image.shape  # (640, 640)
        else:  # pose
            # Pose: Use ROI 320x320 crop images  
            image = self.root['crop_data/roi_images'][zarr_index]
            image_shape = image.shape  # (320, 320)
        
        # Create 3-channel image in HWC format
        image_3ch = np.stack([image, image, image], axis=-1)
        
        # Resize if needed
        if image.shape[0] != self.target_size:
            import cv2
            image_3ch = cv2.resize(image_3ch, (self.target_size, self.target_size))
            # Update image shape for coordinate scaling
            scale_factor = self.target_size / image.shape[0]
        else:
            scale_factor = 1.0
        
        # Transpose from HWC to CHW format for PyTorch
        image_tensor = image_3ch.transpose(2, 0, 1)
        
        # Get bounding box data
        bbox_data = self._get_bbox_data(zarr_index)
        
        if bbox_data is not None:
            # Scale bounding box if image was resized
            if scale_factor != 1.0 and self.task == 'pose':
                # For pose task ROI images, bbox coordinates might need adjustment
                # But since they're normalized, they should be fine as-is
                pass
            
            label = bbox_data[:5].reshape(1, -1)  # [class, x, y, w, h]
        else:
            # Fallback for invalid data (shouldn't happen since we pre-filter)
            print(f"Warning: No valid fish data for frame {zarr_index}, using fallback")
            label = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)

        # CRITICAL FIX: Ensure cls and bboxes are always properly shaped arrays
        cls_labels = label[:, 0].astype(np.float32)  # Shape: (1,) - always 1D
        bbox_coords = label[:, 1:5].astype(np.float32)  # Shape: (1, 4) - always 2D
        
        # Validate dimensions
        assert cls_labels.ndim == 1, f"cls_labels should be 1D, got {cls_labels.ndim}D with shape {cls_labels.shape}"
        assert bbox_coords.ndim == 2 and bbox_coords.shape[1] == 4, f"bbox_coords should be (N, 4), got {bbox_coords.shape}"

        # Return complete dictionary for YOLO training
        return {
            "img": image_tensor,
            "cls": cls_labels,        # Guaranteed 1D: [0] 
            "bboxes": bbox_coords,    # Guaranteed 2D: [[x, y, w, h]]
            "im_file": f"zarr_frame_{zarr_index}",
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (1.0, (0.0, 0.0))
        }

    def get_sample_info(self, index: int):
        """Get detailed information about a sample (useful for debugging)."""
        zarr_index = self.indices[index]
        tracking_data = self.root['tracking/tracking_results'][zarr_index]
        bbox_data = self._get_bbox_data(zarr_index)
        
        info = {
            'zarr_index': zarr_index,
            'data_format': self.data_format,
            'task': self.task,
            'image_source': 'full_640x640' if self.task == 'detect' else 'roi_320x320',
            'valid_bbox': bbox_data is not None
        }
        
        if bbox_data is not None:
            info.update({
                'bbox_center': (bbox_data[1], bbox_data[2]),
                'bbox_size': (bbox_data[3], bbox_data[4]),
                'confidence': bbox_data[5] if len(bbox_data) > 5 else 1.0
            })
        
        return info


def main():
    """Main function for testing the dataset with command line arguments."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Test ZarrYOLODataset with enhanced pipeline data",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Test detection task (640x640 full images)
  python zarr_yolo_dataset_bbox.py /path/to/video.zarr --task detect
  
  # Test pose task (320x320 ROI crops)  
  python zarr_yolo_dataset_bbox.py /path/to/video.zarr --task pose
  
  # Test both tasks with custom split
  python zarr_yolo_dataset_bbox.py /path/to/video.zarr --task both --split-ratio 0.9
        """
    )
    
    parser.add_argument('zarr_path', type=str, help='Path to the Zarr file (e.g., video.zarr)')
    parser.add_argument('--task', choices=['detect', 'pose', 'both'], default='both',
                       help='Task to test: detect (640x640), pose (320x320), or both')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--max-samples', type=int, default=3,
                       help='Maximum samples to show in detail (default: 3)')
    
    args = parser.parse_args()
    
    # Validate input path
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    print(f"🔬 Testing Enhanced Index-Based Dataset")
    print(f"📁 Zarr path: {zarr_path}")
    print(f"🎯 Task(s): {args.task}")
    print(f"📊 Split ratio: {args.split_ratio}")
    print("=" * 60)
    
    try:
        # Test detection task
        if args.task in ['detect', 'both']:
            print("\n🎯 DETECTION TASK (640×640 Full Images)")
            print("-" * 45)
            
            detect_train_set = ZarrYOLODataset(
                zarr_path=str(zarr_path), 
                mode='train', 
                task='detect',
                split_ratio=args.split_ratio,
                random_seed=args.random_seed
            )
            
            detect_val_set = ZarrYOLODataset(
                zarr_path=str(zarr_path), 
                mode='val', 
                task='detect',
                split_ratio=args.split_ratio,
                random_seed=args.random_seed
            )

            print(f"📊 Training samples: {len(detect_train_set)}")
            print(f"📊 Validation samples: {len(detect_val_set)}")
            
            # Show sample details
            print(f"\n📋 Sample Detection Data:")
            for i in range(min(args.max_samples, len(detect_train_set))):
                sample = detect_train_set[i]
                sample_info = detect_train_set.get_sample_info(i)
                bbox = sample['bboxes'][0]
                cls = sample['cls'][0]
                img_shape = sample['img'].shape
                
                print(f"  Sample {i+1}:")
                print(f"    Zarr index: {sample_info['zarr_index']}")
                print(f"    Image shape: {img_shape}")
                print(f"    Bbox (YOLO): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
                print(f"    Class: {cls}")
                if sample_info['valid_bbox']:
                    print(f"    Confidence: {sample_info['confidence']:.3f}")
        
        # Test pose task  
        if args.task in ['pose', 'both']:
            print(f"\n🧘 POSE TASK (320×320 ROI Crops)")
            print("-" * 40)
            
            pose_train_set = ZarrYOLODataset(
                zarr_path=str(zarr_path), 
                mode='train', 
                task='pose',
                split_ratio=args.split_ratio,
                random_seed=args.random_seed
            )
            
            pose_val_set = ZarrYOLODataset(
                zarr_path=str(zarr_path), 
                mode='val', 
                task='pose',
                split_ratio=args.split_ratio,
                random_seed=args.random_seed
            )
            
            print(f"📊 Training samples: {len(pose_train_set)}")
            print(f"📊 Validation samples: {len(pose_val_set)}")
            
            # Show sample details
            print(f"\n📋 Sample Pose Data:")
            for i in range(min(args.max_samples, len(pose_train_set))):
                sample = pose_train_set[i]
                sample_info = pose_train_set.get_sample_info(i)
                bbox = sample['bboxes'][0]
                cls = sample['cls'][0]
                img_shape = sample['img'].shape
                
                print(f"  Sample {i+1}:")
                print(f"    Zarr index: {sample_info['zarr_index']}")
                print(f"    Image shape: {img_shape}")
                print(f"    Bbox (YOLO): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
                print(f"    Class: {cls}")
        
        print(f"\n✅ DATASET TESTING COMPLETE")
        print(f"🎯 Detection task: Ready for YOLO object detection training")
        print(f"🧘 Pose task: Ready for YOLO pose estimation training")
        print(f"💡 Use with your training scripts: python train_yolo.py --zarr-path {zarr_path}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()