#!/usr/bin/env python3
"""
Test Multi-Zarr Dataset
Test the runtime combination of your two zarr files before training.
"""

import sys
import os
from pathlib import Path

# Add the current directory to path so we can import our modules
sys.path.append('.')

from multi_zarr_yolo_dataset import MultiZarrYOLODataset, validate_zarr_compatibility

def test_your_zarr_files():
    """Test your specific zarr files for compatibility and dataset creation."""
    
    # Your zarr file paths
    zarr_paths = [
        "/home/delahantyj@hhmi.org/Desktop/concentricOMR3/longer_edge.zarr",
        "/home/delahantyj@hhmi.org/Desktop/concentricOMR3/video.zarr"
    ]
    
    print("🧪 TESTING YOUR MULTI-ZARR SETUP")
    print("=" * 50)
    print(f"📁 Testing {len(zarr_paths)} zarr files:")
    for i, path in enumerate(zarr_paths):
        print(f"   {i+1}. {Path(path).name}")
    print()
    
    # Step 1: Validate compatibility
    print("STEP 1: Compatibility Check")
    print("-" * 30)
    
    try:
        compatibility = validate_zarr_compatibility(zarr_paths)
        
        if compatibility['compatible']:
            print("✅ Zarr files are compatible!")
            print(f"   📊 Total frames across all videos: {compatibility['total_frames']}")
            print(f"   📋 Data format: {compatibility['common_format']}")
            print(f"   📐 Image shape: {compatibility['common_image_shape']}")
            
            # Show per-video info
            for zarr_path, info in compatibility['zarr_info'].items():
                video_name = Path(zarr_path).stem
                print(f"   📹 {video_name}: {info['total_frames']} frames ({info['data_format']} format)")
        else:
            print("❌ Compatibility issues found:")
            for issue in compatibility['issues']:
                print(f"   • {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Error during compatibility check: {e}")
        return False
    
    print()
    
    # Step 2: Create dataset instances
    print("STEP 2: Dataset Creation Test")
    print("-" * 35)
    
    try:
        # Test both train and val datasets
        for mode in ['train', 'val']:
            print(f"🔄 Creating {mode} dataset...")
            
            dataset = MultiZarrYOLODataset(
                zarr_paths=zarr_paths,
                mode=mode,
                split_ratio=0.8,
                random_seed=42,
                task='detect'  # 640x640 detection task
            )
            
            print(f"✅ {mode.title()} dataset created successfully!")
            
            # Show statistics
            stats = dataset.get_video_statistics()
            print(f"   📊 Total {mode} samples: {stats['total_samples']}")
            
            for video_name, count in stats['per_video_counts'].items():
                percentage = stats['per_video_percentages'][video_name]
                print(f"      {video_name}: {count} samples ({percentage:.1f}%)")
            
            # Test loading a few samples
            print(f"   🧪 Testing sample loading...")
            for i in range(min(3, len(dataset))):
                try:
                    sample = dataset[i]
                    print(f"      Sample {i}: video={sample['video_name']}, "
                          f"frame={sample['local_frame_idx']}, "
                          f"img_shape={sample['img'].shape}, "
                          f"bbox_shape={sample['bboxes'].shape}")
                except Exception as e:
                    print(f"      ❌ Error loading sample {i}: {e}")
                    return False
            
            print()
    
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Save dataset info
    print("STEP 3: Saving Dataset Information")
    print("-" * 40)
    
    try:
        # Create a final test dataset to save info
        final_dataset = MultiZarrYOLODataset(
            zarr_paths=zarr_paths,
            mode='train',
            split_ratio=0.8,
            random_seed=42,
            task='detect'
        )
        
        final_dataset.save_dataset_info('multi_zarr_dataset_info.json')
        print("✅ Dataset info saved to 'multi_zarr_dataset_info.json'")
        
    except Exception as e:
        print(f"⚠️  Could not save dataset info: {e}")
    
    print()
    print("🎉 SUCCESS! Your multi-zarr setup is ready for training!")
    print()
    print("📋 Next steps:")
    print("   1. Review the config file: multi_zarr_config.yaml")
    print("   2. Start training with:")
    print("      python multi_zarr_yolo_trainer.py multi_zarr_config.yaml --epochs 100")
    print("   3. Monitor training progress and adjust parameters as needed")
    
    return True

if __name__ == "__main__":
    test_your_zarr_files()