#!/usr/bin/env python3
"""
Enhanced Multi-Zarr Test Script
Comprehensive validation of your multi-zarr setup before training.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import torch
from typing import List, Dict
import warnings
import time
import argparse

# Add current directory to path for imports
sys.path.append('.')

try:
    from enhanced_multi_zarr_dataset import (
        EnhancedMultiZarrYOLODataset, 
        MultiDatasetConfig, 
        SamplingStrategy,
        CompatibilityValidator,
        create_multi_zarr_dataset
    )
    from enhanced_multi_zarr_trainer import load_training_config
    
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure enhanced_multi_zarr_dataset.py and enhanced_multi_zarr_trainer.py are in the current directory")
    IMPORTS_SUCCESS = False

def test_imports():
    """Test that all required imports work."""
    print("🔍 TESTING IMPORTS")
    print("-" * 30)
    
    if not IMPORTS_SUCCESS:
        print("❌ Failed to import enhanced multi-zarr modules")
        return False
    
    print("✅ Enhanced multi-zarr modules imported successfully")
    
    # Test other critical imports
    try:
        import zarr
        print("✅ zarr imported successfully")
    except ImportError:
        print("❌ zarr not available - install with: pip install zarr")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        if torch.cuda.is_available():
            print(f"🎯 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - training will use CPU")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError:
        print("❌ Ultralytics not available - install with: pip install ultralytics")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print("✅ scikit-learn imported successfully")
    except ImportError:
        print("❌ scikit-learn not available - install with: pip install scikit-learn")
        return False
    
    print("✅ All required imports successful!")
    return True

def test_zarr_files_direct(zarr_paths: List[str]):
    """Test zarr files directly without our dataset classes."""
    print("\n🔍 TESTING ZARR FILES DIRECTLY")
    print("-" * 35)
    
    if not zarr_paths:
        print("❌ No zarr paths provided")
        return False
    
    for i, zarr_path in enumerate(zarr_paths):
        print(f"\n📁 Testing zarr file {i+1}: {Path(zarr_path).name}")
        
        if not Path(zarr_path).exists():
            print(f"   ❌ File not found: {zarr_path}")
            continue
        
        try:
            import zarr
            root = zarr.open(zarr_path, mode='r')
            
            # Check basic structure
            required_paths = ['raw_video/images_ds', 'tracking/tracking_results']
            for path in required_paths:
                if path in root:
                    print(f"   ✅ {path}: {root[path].shape}")
                else:
                    print(f"   ❌ Missing: {path}")
            
            # Check tracking results
            if 'tracking/tracking_results' in root:
                tracking = root['tracking/tracking_results']
                column_names = tracking.attrs.get('column_names', [])
                print(f"   📊 Tracking columns: {len(column_names)}")
                
                if 'bbox_x_norm_ds' in column_names:
                    print(f"   ✅ Enhanced format detected")
                else:
                    print(f"   📊 Original format detected")
                
                # Check for valid data
                data = tracking[:]
                valid_mask = ~np.isnan(data[:, 0])
                valid_frames = np.sum(valid_mask)
                total_frames = data.shape[0]
                
                print(f"   📈 Valid frames: {valid_frames}/{total_frames} ({valid_frames/total_frames*100:.1f}%)")
            
            print(f"   ✅ Zarr file {i+1} looks good!")
            
        except Exception as e:
            print(f"   ❌ Error reading zarr file: {e}")
            return False
    
    return True

def test_compatibility_validation(zarr_paths: List[str]):
    """Test our compatibility validation system."""
    print("\n🔍 TESTING COMPATIBILITY VALIDATION")
    print("-" * 40)
    
    try:
        compatibility = CompatibilityValidator.validate_zarr_compatibility(zarr_paths)
        
        if compatibility['compatible']:
            print("✅ Zarr files are compatible!")
            print(f"   📊 Total valid frames: {compatibility['total_valid_frames']}")
            print(f"   📋 Common format: {compatibility['common_format']}")
            print(f"   📐 Common image shape: {compatibility['common_image_shape']}")
            
            # Show per-file details
            for metadata in compatibility['metadata']:
                print(f"   📹 {metadata.name}:")
                print(f"      Valid frames: {metadata.valid_frames}/{metadata.total_frames}")
                print(f"      Success rate: {metadata.tracking_success_rate:.1f}%")
                print(f"      Data format: {metadata.data_format}")
        else:
            print("❌ Compatibility issues found:")
            for issue in compatibility['issues']:
                print(f"   • {issue}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in compatibility validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation(zarr_paths: List[str]):
    """Test dataset creation with different configurations."""
    print("\n🔍 TESTING DATASET CREATION")
    print("-" * 35)
    
    # Test different sampling strategies
    strategies = [
        SamplingStrategy.BALANCED,
        SamplingStrategy.PROPORTIONAL,
        SamplingStrategy.QUALITY_WEIGHTED
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value} sampling strategy:")
        
        try:
            config = MultiDatasetConfig(
                zarr_paths=zarr_paths,
                sampling_strategy=strategy,
                split_ratio=0.8,
                random_seed=42,
                task='detect'
            )
            
            # Create train dataset
            train_dataset = EnhancedMultiZarrYOLODataset(config, mode='train')
            val_dataset = EnhancedMultiZarrYOLODataset(config, mode='val')
            
            print(f"   ✅ Created datasets successfully")
            print(f"   📊 Train samples: {len(train_dataset)}")
            print(f"   📊 Val samples: {len(val_dataset)}")
            
            # Show distribution
            train_stats = train_dataset.get_dataset_statistics()
            print(f"   📈 Train distribution:")
            for name, count in train_stats['mode_specific']['per_dataset_counts'].items():
                pct = train_stats['mode_specific']['per_dataset_percentages'][name]
                print(f"      {name}: {count} samples ({pct:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Error with {strategy.value} strategy: {e}")
            return False
    
    return True

def test_sample_loading(zarr_paths: List[str], num_samples: int = 5):
    """Test loading individual samples."""
    print("\n🔍 TESTING SAMPLE LOADING")
    print("-" * 30)
    
    try:
        # Create a simple dataset
        config = MultiDatasetConfig(
            zarr_paths=zarr_paths,
            sampling_strategy=SamplingStrategy.BALANCED,
            split_ratio=0.8,
            random_seed=42,
            task='detect'
        )
        
        dataset = EnhancedMultiZarrYOLODataset(config, mode='train')
        
        print(f"🧪 Testing {min(num_samples, len(dataset))} samples...")
        
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                
                # Validate sample structure
                required_keys = ['img', 'cls', 'bboxes', 'im_file', 'dataset_name']
                missing_keys = [key for key in required_keys if key not in sample]
                
                if missing_keys:
                    print(f"   ❌ Sample {i}: Missing keys: {missing_keys}")
                    continue
                
                # Validate tensor shapes
                img_shape = sample['img'].shape
                cls_shape = sample['cls'].shape
                bbox_shape = sample['bboxes'].shape
                
                if img_shape[0] != 3:
                    print(f"   ❌ Sample {i}: Image should have 3 channels, got {img_shape}")
                    continue
                
                if len(cls_shape) != 1:
                    print(f"   ❌ Sample {i}: cls should be 1D, got shape {cls_shape}")
                    continue
                
                if len(bbox_shape) != 2 or bbox_shape[1] != 4:
                    print(f"   ❌ Sample {i}: bboxes should be (N, 4), got {bbox_shape}")
                    continue
                
                print(f"   ✅ Sample {i}: {sample['dataset_name']}, img={img_shape}, cls={cls_shape}, bbox={bbox_shape}")
                
            except Exception as e:
                print(f"   ❌ Sample {i}: Error loading - {e}")
                return False
        
        print("✅ All sample loading tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in sample loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_integration(zarr_paths: List[str], batch_size: int = 4):
    """Test PyTorch DataLoader integration."""
    print("\n🔍 TESTING DATALOADER INTEGRATION")
    print("-" * 40)
    
    try:
        # Create dataset
        config = MultiDatasetConfig(
            zarr_paths=zarr_paths,
            sampling_strategy=SamplingStrategy.BALANCED,
            random_seed=42,
            task='detect'
        )
        
        dataset = EnhancedMultiZarrYOLODataset(config, mode='train')
        
        # Create a simple collate function (not the enhanced one)
        def simple_collate_fn(batch):
            images = torch.from_numpy(np.stack([s['img'] for s in batch]))
            
            # Collect all labels
            all_cls = []
            all_bboxes = []
            all_batch_idx = []
            
            for batch_i, sample in enumerate(batch):
                cls = sample['cls']
                bboxes = sample['bboxes']
                
                # Ensure proper shapes
                if cls.ndim == 0:
                    cls = np.array([cls])
                if bboxes.ndim == 1:
                    bboxes = bboxes[None, :]
                
                all_cls.extend(cls)
                all_bboxes.extend(bboxes)
                all_batch_idx.extend([batch_i] * len(cls))
            
            return {
                'img': images,
                'cls': torch.tensor(all_cls, dtype=torch.float32),
                'bboxes': torch.tensor(all_bboxes, dtype=torch.float32),
                'batch_idx': torch.tensor(all_batch_idx, dtype=torch.long),
                'im_file': [s['im_file'] for s in batch],
            }
        
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=simple_collate_fn,
            num_workers=0  # Avoid multiprocessing issues in testing
        )
        
        print(f"🧪 Testing DataLoader with batch_size={batch_size}...")
        
        # Test a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test 3 batches
                break
            
            # Validate batch structure
            required_keys = ['img', 'cls', 'bboxes', 'batch_idx']
            missing_keys = [key for key in required_keys if key not in batch]
            
            if missing_keys:
                print(f"   ❌ Batch {i}: Missing keys: {missing_keys}")
                return False
            
            # Validate tensor shapes and properties
            img_tensor = batch['img']
            cls_tensor = batch['cls']
            bbox_tensor = batch['bboxes']
            batch_idx_tensor = batch['batch_idx']
            
            # Check shapes
            if img_tensor.ndim != 4:
                print(f"   ❌ Batch {i}: img should be 4D (B,C,H,W), got {img_tensor.shape}")
                return False
            
            if cls_tensor.ndim != 1:
                print(f"   ❌ Batch {i}: cls should be 1D, got {cls_tensor.shape}")
                return False
            
            if bbox_tensor.ndim != 2 or bbox_tensor.shape[1] != 4:
                print(f"   ❌ Batch {i}: bboxes should be (N,4), got {bbox_tensor.shape}")
                return False
            
            print(f"   ✅ Batch {i}: img={img_tensor.shape}, cls={cls_tensor.shape}, "
                  f"bbox={bbox_tensor.shape}, batch_idx={batch_idx_tensor.shape}")
        
        print("✅ DataLoader integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in DataLoader test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading(config_path: str):
    """Test configuration loading."""
    print("\n🔍 TESTING CONFIG LOADING")
    print("-" * 30)
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        config = load_training_config(config_path)
        
        print("✅ Config loaded successfully!")
        print(f"   📁 Zarr paths: {len(config.zarr_paths)}")
        print(f"   🎯 Task: {config.task}")
        print(f"   📊 Sampling strategy: {config.sampling_strategy.value}")
        print(f"   📈 Split ratio: {config.split_ratio}")
        
        # Validate paths exist
        missing_paths = [p for p in config.zarr_paths if not Path(p).exists()]
        if missing_paths:
            print(f"   ❌ Missing zarr files in config: {missing_paths}")
            return False
        
        print("✅ All zarr paths in config exist!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def run_comprehensive_test(config_path: str = None, zarr_paths: List[str] = None):
    """Run comprehensive test suite."""
    print("🧪 ENHANCED MULTI-ZARR COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Imports
    print("\n" + "="*60)
    result = test_imports()
    test_results.append(("Imports", result))
    if not result:
        print("❌ Cannot proceed without successful imports")
        return False
    
    # Determine zarr paths
    if config_path and Path(config_path).exists():
        try:
            config = load_training_config(config_path)
            zarr_paths = config.zarr_paths
            print(f"\n📋 Using zarr paths from config: {config_path}")
        except:
            print(f"\n⚠️  Could not load zarr paths from config, using provided paths")
    
    if not zarr_paths:
        # Default paths for your setup
        zarr_paths = [
            "/home/delahantyj@hhmi.org/Desktop/concentricOMR3/longer_edge.zarr",
            "/home/delahantyj@hhmi.org/Desktop/concentricOMR3/video.zarr"
        ]
        print(f"\n📋 Using default zarr paths")
    
    print(f"📁 Testing with {len(zarr_paths)} zarr files:")
    for i, path in enumerate(zarr_paths):
        print(f"   {i+1}. {Path(path).name}")
    
    # Test 2: Direct zarr file access
    print("\n" + "="*60)
    result = test_zarr_files_direct(zarr_paths)
    test_results.append(("Direct Zarr Access", result))
    
    # Test 3: Compatibility validation
    print("\n" + "="*60)
    result = test_compatibility_validation(zarr_paths)
    test_results.append(("Compatibility Validation", result))
    if not result:
        print("❌ Cannot proceed with incompatible zarr files")
        return False
    
    # Test 4: Dataset creation
    print("\n" + "="*60)
    result = test_dataset_creation(zarr_paths)
    test_results.append(("Dataset Creation", result))
    
    # Test 5: Sample loading
    print("\n" + "="*60)
    result = test_sample_loading(zarr_paths)
    test_results.append(("Sample Loading", result))
    
    # Test 6: DataLoader integration
    print("\n" + "="*60)
    result = test_dataloader_integration(zarr_paths)
    test_results.append(("DataLoader Integration", result))
    
    # Test 7: Config loading (if config provided)
    if config_path:
        print("\n" + "="*60)
        result = test_config_loading(config_path)
        test_results.append(("Config Loading", result))
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 TEST SUMMARY")
    print("-" * 20)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n💡 NEXT STEPS:")
        print("   1. Your enhanced multi-zarr setup is ready!")
        print("   2. Start training with:")
        if config_path:
            print(f"      python enhanced_multi_zarr_trainer.py {config_path} --epochs 50")
        else:
            print(f"      python enhanced_multi_zarr_trainer.py enhanced_multi_zarr_config.yaml --epochs 50")
        print("   3. Monitor training progress and per-dataset metrics")
        print("   4. Experiment with different sampling strategies")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Review failed tests above")
        print("   2. Check zarr file paths and permissions")
        print("   3. Ensure all dependencies are installed")
        print("   4. Verify zarr files contain valid tracking data")
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(
        description="Test Enhanced Multi-Zarr Setup",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--zarr-paths', nargs='+', help='Zarr file paths to test')
    
    args = parser.parse_args()
    
    if args.config and not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return
    
    # Run the comprehensive test
    success = run_comprehensive_test(
        config_path=args.config,
        zarr_paths=args.zarr_paths
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()