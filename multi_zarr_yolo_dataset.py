# multi_zarr_yolo_dataset.py
# Enhanced YOLO dataset that combines multiple zarr files

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

def validate_zarr_compatibility(zarr_paths: List[str]) -> Dict:
    """
    Validate that multiple zarr files are compatible for training.
    
    Returns:
        dict: Compatibility report and merged metadata
    """
    print("🔍 Validating zarr file compatibility...")
    
    zarr_info = {}
    compatibility_issues = []
    
    for i, zarr_path in enumerate(zarr_paths):
        try:
            root = zarr.open(zarr_path, mode='r')
            
            # Check required structure
            required_paths = [
                'raw_video/images_ds',
                'tracking/tracking_results'
            ]
            
            missing_paths = [path for path in required_paths if path not in root]
            if missing_paths:
                compatibility_issues.append(f"Zarr {i} missing: {missing_paths}")
                continue
            
            # Get metadata
            tracking_results = root['tracking/tracking_results']
            column_names = tracking_results.attrs.get('column_names', [])
            
            zarr_info[zarr_path] = {
                'total_frames': tracking_results.shape[0],
                'num_columns': len(column_names),
                'column_names': column_names,
                'image_shape': root['raw_video/images_ds'].shape[1:],
                'data_format': 'enhanced' if 'bbox_x_norm_ds' in column_names else 'original',
                'enhanced_features': 'enhanced_features' in root['tracking'].attrs
            }
            
            print(f"  ✅ Zarr {i}: {zarr_info[zarr_path]['total_frames']} frames, "
                  f"{zarr_info[zarr_path]['data_format']} format")
            
        except Exception as e:
            compatibility_issues.append(f"Zarr {i} error: {e}")
    
    # Check compatibility
    if len(zarr_info) < 2:
        compatibility_issues.append("Need at least 2 valid zarr files")
        return {'compatible': False, 'issues': compatibility_issues}
    
    # Check format consistency
    formats = [info['data_format'] for info in zarr_info.values()]
    if len(set(formats)) > 1:
        compatibility_issues.append(f"Mixed data formats: {set(formats)}")
    
    # Check image shape consistency
    shapes = [info['image_shape'] for info in zarr_info.values()]
    if len(set(shapes)) > 1:
        compatibility_issues.append(f"Mixed image shapes: {set(shapes)}")
    
    # Check column consistency (for enhanced format)
    if 'enhanced' in formats:
        all_columns = [set(info['column_names']) for info in zarr_info.values()]
        if not all(cols == all_columns[0] for cols in all_columns):
            compatibility_issues.append("Enhanced format column names don't match")
    
    compatible = len(compatibility_issues) == 0
    
    return {
        'compatible': compatible,
        'issues': compatibility_issues,
        'zarr_info': zarr_info,
        'total_frames': sum(info['total_frames'] for info in zarr_info.values()),
        'common_format': formats[0] if len(set(formats)) == 1 else None,
        'common_image_shape': shapes[0] if len(set(shapes)) == 1 else None
    }

class MultiZarrYOLODataset(Dataset):
    """
    YOLO dataset that combines data from multiple zarr files.
    
    Features:
    - Combines multiple videos seamlessly
    - Maintains per-video metadata and provenance
    - Supports both original and enhanced data formats
    - Handles train/val splitting across all videos
    - Optional per-video weighting
    """
    
    def __init__(self, zarr_paths: List[str], mode: str = 'train', 
                 split_ratio: float = 0.8, random_seed: int = 42,
                 task: str = 'detect', target_size: int = None,
                 video_weights: Dict[str, float] = None):
        """
        Initialize multi-zarr dataset.
        
        Args:
            zarr_paths: List of paths to zarr files
            mode: 'train' or 'val'
            split_ratio: Train/validation split ratio
            random_seed: Random seed for reproducible splits
            task: 'detect' (640x640) or 'pose' (320x320 ROI)
            target_size: Override default target size
            video_weights: Optional weights for sampling from each video
        """
        super().__init__()
        
        self.zarr_paths = [str(Path(p).resolve()) for p in zarr_paths]
        self.mode = mode
        self.task = task
        self.video_weights = video_weights or {}
        
        # Validate compatibility
        self.compatibility = validate_zarr_compatibility(self.zarr_paths)
        if not self.compatibility['compatible']:
            raise ValueError(f"Zarr files not compatible: {self.compatibility['issues']}")
        
        # Set target size based on task
        if task == 'detect':
            self.image_source = 'raw_video/images_ds'  # 640x640
            self.target_size = target_size if target_size else 640
        else:  # pose
            self.image_source = 'crop_data/roi_images'  # 320x320
            self.target_size = target_size if target_size else 320
        
        # Load and combine data
        self._load_combined_data(split_ratio, random_seed)
        
        print(f"🎯 Multi-Zarr {task} dataset initialized:")
        print(f"   📁 Videos: {len(self.zarr_paths)}")
        print(f"   📊 Total valid frames: {len(self.global_indices)}")
        print(f"   🎭 Mode: {mode} ({len(self.indices)} samples)")
        print(f"   📐 Target size: {self.target_size}x{self.target_size}")
    
    def _load_combined_data(self, split_ratio: float, random_seed: int):
        """Load and combine data from all zarr files."""
        
        # Open all zarr files
        self.zarr_roots = {}
        self.video_metadata = {}
        
        for zarr_path in self.zarr_paths:
            root = zarr.open(zarr_path, mode='r')
            self.zarr_roots[zarr_path] = root
            
            # Store metadata for each video
            video_name = Path(zarr_path).stem
            self.video_metadata[zarr_path] = {
                'name': video_name,
                'total_frames': root['tracking/tracking_results'].shape[0],
                'zarr_path': zarr_path
            }
        
        # Detect data format from first zarr
        first_root = list(self.zarr_roots.values())[0]
        tracking_results = first_root['tracking/tracking_results']
        column_names = tracking_results.attrs['column_names']
        self.data_format = self.compatibility['common_format']
        self.col_map = {name: i for i, name in enumerate(column_names)}
        
        # Get column mappings
        if self.data_format == 'enhanced':
            self.coord_mappings = {
                'heading': self.col_map['heading_degrees'],
                'bbox_x': self.col_map['bbox_x_norm_ds'],
                'bbox_y': self.col_map['bbox_y_norm_ds'],
                'bbox_width': self.col_map['bbox_width_norm_ds'],
                'bbox_height': self.col_map['bbox_height_norm_ds'],
                'confidence': self.col_map.get('confidence_score', None)
            }
        else:  # original
            self.coord_mappings = {
                'heading': self.col_map['heading_degrees'],
                'bbox_x': self.col_map['bbox_x_norm'],
                'bbox_y': self.col_map['bbox_y_norm'],
                'bbox_width': None,
                'bbox_height': None,
                'confidence': None
            }
        
        # Find valid frames across all videos
        self.global_indices = []  # List of (zarr_path, local_frame_idx) tuples
        
        for zarr_path, root in self.zarr_roots.items():
            tracking_data = root['tracking/tracking_results'][:]
            
            # Apply same filtering logic as single-zarr dataset
            valid_mask = ~np.isnan(tracking_data[:, self.coord_mappings['heading']])
            
            if self.data_format == 'enhanced':
                bbox_valid = (~np.isnan(tracking_data[:, self.coord_mappings['bbox_x']]) & 
                             ~np.isnan(tracking_data[:, self.coord_mappings['bbox_y']]))
                valid_mask = valid_mask & bbox_valid
            
            valid_local_indices = np.where(valid_mask)[0]
            
            # Add to global index with video provenance
            video_global_indices = [(zarr_path, idx) for idx in valid_local_indices]
            self.global_indices.extend(video_global_indices)
            
            print(f"   📹 {Path(zarr_path).stem}: {len(valid_local_indices)} valid frames")
        
        # Apply video weighting if specified
        if self.video_weights:
            self._apply_video_weighting()
        
        # Split into train/val
        train_indices, val_indices = train_test_split(
            self.global_indices,
            train_size=split_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        if self.mode == 'train':
            self.indices = train_indices
        else:  # val
            self.indices = val_indices
    
    def _apply_video_weighting(self):
        """Apply video weighting to balance sampling."""
        if not self.video_weights:
            return
        
        weighted_indices = []
        
        for zarr_path, weight in self.video_weights.items():
            if zarr_path not in self.zarr_roots:
                continue
            
            # Get indices for this video
            video_indices = [(zp, idx) for zp, idx in self.global_indices if zp == zarr_path]
            
            # Sample based on weight
            n_samples = int(len(video_indices) * weight)
            if n_samples > 0:
                np.random.seed(42)  # Reproducible sampling
                sampled_indices = np.random.choice(len(video_indices), n_samples, replace=False)
                weighted_indices.extend([video_indices[i] for i in sampled_indices])
        
        if weighted_indices:
            self.global_indices = weighted_indices
            print(f"   ⚖️  Applied video weighting: {len(weighted_indices)} samples")
    
    def _get_bbox_data(self, zarr_path: str, local_frame_idx: int):
        """Extract bounding box data for a specific frame."""
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
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index: int):
        # Get global index (zarr_path, local_frame_idx)
        zarr_path, local_frame_idx = self.indices[index]
        root = self.zarr_roots[zarr_path]
        
        # Get image based on task
        if self.task == 'detect':
            image = root['raw_video/images_ds'][local_frame_idx]
        else:  # pose
            image = root['crop_data/roi_images'][local_frame_idx]
        
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
            # Fallback
            label = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        
        # Ensure proper tensor shapes
        cls_labels = label[:, 0].astype(np.float32)
        bbox_coords = label[:, 1:5].astype(np.float32)
        
        assert cls_labels.ndim == 1, f"cls_labels must be 1D, got {cls_labels.ndim}D"
        assert bbox_coords.ndim == 2 and bbox_coords.shape[1] == 4, f"bbox_coords must be (N, 4)"
        
        # Add video metadata to sample
        video_name = Path(zarr_path).stem
        
        return {
            "img": image_tensor,
            "cls": cls_labels,
            "bboxes": bbox_coords,
            "im_file": f"{video_name}_frame_{local_frame_idx}",
            "ori_shape": (self.target_size, self.target_size),
            "ratio_pad": (1.0, (0.0, 0.0)),
            # Additional metadata
            "video_name": video_name,
            "zarr_path": zarr_path,
            "local_frame_idx": local_frame_idx
        }
    
    def get_video_statistics(self) -> Dict:
        """Get statistics about video representation in the dataset."""
        video_counts = {}
        
        for zarr_path, _ in self.indices:
            video_name = Path(zarr_path).stem
            video_counts[video_name] = video_counts.get(video_name, 0) + 1
        
        total_samples = len(self.indices)
        
        stats = {
            'total_samples': total_samples,
            'videos': len(video_counts),
            'per_video_counts': video_counts,
            'per_video_percentages': {
                name: (count / total_samples) * 100 
                for name, count in video_counts.items()
            }
        }
        
        return stats
    
    def save_dataset_info(self, output_path: str):
        """Save comprehensive dataset information."""
        info = {
            'dataset_type': f'MultiZarr{self.task.title()}Dataset',
            'mode': self.mode,
            'data_format': self.data_format,
            'target_size': self.target_size,
            'zarr_paths': self.zarr_paths,
            'video_metadata': self.video_metadata,
            'compatibility_report': self.compatibility,
            'statistics': self.get_video_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        print(f"📋 Dataset info saved to: {output_path}")


def create_multi_zarr_config(zarr_paths: List[str], output_path: str = 'multi_zarr_config.yaml'):
    """Create YOLO config file for multi-zarr dataset."""
    
    config_content = f"""# Multi-Zarr YOLO Configuration
# Generated for {len(zarr_paths)} video files

# Training and validation paths (handled by custom dataset)
train: ./
val: ./

# Number of classes
nc: 1

# Class names
names:
  - fish

# Multi-Zarr specific configuration
zarr_paths:
{chr(10).join(f'  - {path}' for path in zarr_paths)}

# Dataset parameters
task: detect  # or 'pose'
target_size: 640  # or 320 for pose
split_ratio: 0.8
random_seed: 42

# Optional video weighting (uncomment and adjust as needed)
# video_weights:
#   video1: 1.0
#   video2: 0.5
#   video3: 2.0
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Multi-zarr config saved to: {output_path}")
    return output_path


# Example usage and testing
def main():
    """Test the multi-zarr dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Multi-Zarr YOLO Dataset")
    parser.add_argument("zarr_paths", nargs='+', help="Paths to zarr files")
    parser.add_argument("--mode", choices=['train', 'val'], default='train')
    parser.add_argument("--task", choices=['detect', 'pose'], default='detect')
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--max-samples", type=int, default=5, 
                       help="Max samples to show in test")
    
    args = parser.parse_args()
    
    print("🧪 Testing Multi-Zarr YOLO Dataset")
    print("=" * 50)
    
    try:
        # Create dataset
        dataset = MultiZarrYOLODataset(
            zarr_paths=args.zarr_paths,
            mode=args.mode,
            task=args.task,
            split_ratio=args.split_ratio
        )
        
        # Show statistics
        stats = dataset.get_video_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Videos: {stats['videos']}")
        for video, count in stats['per_video_counts'].items():
            percentage = stats['per_video_percentages'][video]
            print(f"   {video}: {count} samples ({percentage:.1f}%)")
        
        # Test sample loading
        print(f"\n🔬 Testing sample loading:")
        for i in range(min(args.max_samples, len(dataset))):
            sample = dataset[i]
            print(f"   Sample {i}:")
            print(f"     Video: {sample['video_name']}")
            print(f"     Image shape: {sample['img'].shape}")
            print(f"     Classes: {sample['cls']}")
            print(f"     Bboxes: {sample['bboxes']}")
            print(f"     Frame: {sample['local_frame_idx']}")
        
        # Save dataset info
        dataset.save_dataset_info('multi_zarr_dataset_info.json')
        
        # Create config file
        create_multi_zarr_config(args.zarr_paths)
        
        print(f"\n✅ Multi-Zarr dataset test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()