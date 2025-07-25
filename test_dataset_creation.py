#!/usr/bin/env python3
"""
Test Dataset Creation Logic
Actually instantiate the dataset and check every single sample for issues.
"""

import numpy as np
import argparse
from pathlib import Path
from zarr_yolo_dataset_bbox import ZarrYOLODataset
import torch

def test_every_dataset_sample(zarr_path, mode='train', max_samples=None):
    """
    Test every single sample in the dataset to find any problematic ones.
    """
    print(f"🧪 TESTING EVERY DATASET SAMPLE")
    print(f"📁 Zarr path: {zarr_path}")
    print(f"📊 Mode: {mode}")
    print("=" * 60)
    
    try:
        # Create dataset
        dataset = ZarrYOLODataset(zarr_path=zarr_path, mode=mode, task='detect')
        total_samples = len(dataset)
        
        print(f"✅ Created {mode} dataset with {total_samples} samples")
        
        if max_samples:
            test_samples = min(max_samples, total_samples)
            print(f"🎯 Testing first {test_samples} samples (limited)")
        else:
            test_samples = total_samples
            print(f"🎯 Testing ALL {test_samples} samples")
        
        print()
        
        # Track issues
        issues = {
            'cls_wrong_dim': [],
            'cls_wrong_type': [],
            'cls_nan_values': [],
            'bbox_wrong_dim': [],
            'bbox_wrong_type': [],
            'bbox_nan_values': [],
            'image_issues': [],
            'fallback_used': [],
            'other_errors': []
        }
        
        successful_samples = 0
        
        # Test each sample
        print("Testing samples...")
        for i in range(test_samples):
            try:
                # Get the sample
                sample = dataset[i]
                zarr_index = dataset.indices[i]
                
                # Check image
                img = sample['img']
                if not isinstance(img, np.ndarray):
                    issues['image_issues'].append((i, zarr_index, f"img is {type(img)}, not ndarray"))
                    continue
                    
                if img.shape != (3, 640, 640):
                    issues['image_issues'].append((i, zarr_index, f"img shape is {img.shape}, not (3, 640, 640)"))
                    continue
                    
                if np.any(np.isnan(img)):
                    issues['image_issues'].append((i, zarr_index, "img contains NaN values"))
                    continue
                
                # Check cls
                cls = sample['cls']
                if not isinstance(cls, np.ndarray):
                    issues['cls_wrong_type'].append((i, zarr_index, f"cls is {type(cls)}, not ndarray"))
                    continue
                    
                if cls.ndim != 1:
                    issues['cls_wrong_dim'].append((i, zarr_index, f"cls.ndim is {cls.ndim}, not 1"))
                    continue
                    
                if len(cls) != 1:
                    issues['cls_wrong_dim'].append((i, zarr_index, f"cls length is {len(cls)}, not 1"))
                    continue
                    
                if np.any(np.isnan(cls)):
                    issues['cls_nan_values'].append((i, zarr_index, f"cls contains NaN: {cls}"))
                    continue
                
                # Check bboxes
                bboxes = sample['bboxes']
                if not isinstance(bboxes, np.ndarray):
                    issues['bbox_wrong_type'].append((i, zarr_index, f"bboxes is {type(bboxes)}, not ndarray"))
                    continue
                    
                if bboxes.ndim != 2:
                    issues['bbox_wrong_dim'].append((i, zarr_index, f"bboxes.ndim is {bboxes.ndim}, not 2"))
                    continue
                    
                if bboxes.shape != (1, 4):
                    issues['bbox_wrong_dim'].append((i, zarr_index, f"bboxes shape is {bboxes.shape}, not (1, 4)"))
                    continue
                    
                if np.any(np.isnan(bboxes)):
                    issues['bbox_nan_values'].append((i, zarr_index, f"bboxes contains NaN: {bboxes}"))
                    continue
                
                # Check for fallback data (indicates _get_bbox_data returned None)
                bbox_coords = bboxes[0]
                if (np.allclose(bbox_coords, [0.5, 0.5, 0.1, 0.1]) and 
                    cls[0] == 0.0):
                    issues['fallback_used'].append((i, zarr_index, f"Using fallback bbox: {bbox_coords}"))
                    continue
                
                # Check bbox coordinate ranges
                x, y, w, h = bbox_coords
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    issues['bbox_wrong_dim'].append((i, zarr_index, f"bbox coords out of range: {bbox_coords}"))
                    continue
                
                successful_samples += 1
                
                # Show progress for first few and every 100th sample
                if i < 10 or i % 100 == 0:
                    print(f"  ✅ Sample {i} (zarr {zarr_index}): cls={cls}, bbox={bbox_coords}")
                    
            except Exception as e:
                issues['other_errors'].append((i, getattr(dataset, 'indices', [None])[i] if hasattr(dataset, 'indices') else None, str(e)))
                if len(issues['other_errors']) <= 10:  # Show first 10 errors
                    print(f"  ❌ Sample {i}: Exception: {e}")
        
        # Print summary
        print()
        print("🎯 TESTING RESULTS")
        print("-" * 20)
        print(f"✅ Successful samples: {successful_samples}/{test_samples}")
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        print(f"❌ Total issues found: {total_issues}")
        
        if total_issues > 0:
            print()
            print("📋 ISSUE BREAKDOWN:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"   {issue_type}: {len(issue_list)} samples")
                    # Show first few examples
                    for sample_idx, zarr_idx, details in issue_list[:3]:
                        print(f"      Sample {sample_idx} (zarr {zarr_idx}): {details}")
                    if len(issue_list) > 3:
                        print(f"      ... and {len(issue_list) - 3} more")
        
        # Critical issues that would cause 0-d tensor errors
        critical_issues = (len(issues['cls_wrong_dim']) + 
                          len(issues['cls_wrong_type']) + 
                          len(issues['cls_nan_values']))
        
        if critical_issues > 0:
            print()
            print("🚨 CRITICAL ISSUES FOUND:")
            print(f"   {critical_issues} samples have cls tensor problems")
            print("   These could cause the 0-d tensor error during training!")
            return False
        
        fallback_count = len(issues['fallback_used'])
        if fallback_count > 0:
            print()
            print("⚠️  FALLBACK DATA USAGE:")
            print(f"   {fallback_count} samples are using fallback bbox data")
            print("   This indicates _get_bbox_data() returned None for these frames")
            print("   Training might be suboptimal but shouldn't crash")
        
        return total_issues == 0
        
    except Exception as e:
        print(f"❌ Failed to create or test dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collate_function(zarr_path, batch_size=16):
    """
    Test the collate function with actual data to see if it produces valid batches.
    """
    print()
    print("🔧 TESTING COLLATE FUNCTION")
    print("-" * 30)
    
    try:
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = ZarrYOLODataset(zarr_path=zarr_path, mode='train', task='detect')
        
        # Simple collate function (similar to what we use in training)
        def test_collate_fn(batch):
            try:
                # Stack images
                images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                
                # Process labels
                total_labels = sum(len(s['cls']) for s in batch)
                
                if total_labels == 0:
                    return {
                        'img': images,
                        'batch_idx': torch.zeros((0,), dtype=torch.long),
                        'cls': torch.zeros((0,), dtype=torch.float32),
                        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                        'im_file': [s['im_file'] for s in batch],
                        'ori_shape': [s['ori_shape'] for s in batch],
                        'ratio_pad': [s['ratio_pad'] for s in batch]
                    }
                
                cls_array = np.zeros(total_labels, dtype=np.float32)
                bboxes_array = np.zeros((total_labels, 4), dtype=np.float32)
                batch_idx_array = np.zeros(total_labels, dtype=np.int64)
                
                current_idx = 0
                for batch_i, sample in enumerate(batch):
                    sample_cls = np.asarray(sample['cls'], dtype=np.float32)
                    sample_bboxes = np.asarray(sample['bboxes'], dtype=np.float32)
                    
                    # Ensure cls is 1D
                    if sample_cls.ndim == 0:
                        sample_cls = np.array([sample_cls], dtype=np.float32)
                    sample_cls = sample_cls.flatten()
                    
                    # Ensure bboxes is 2D
                    if sample_bboxes.ndim == 1 and len(sample_bboxes) == 4:
                        sample_bboxes = sample_bboxes[None, :]
                    
                    n_labels = len(sample_cls)
                    if n_labels > 0 and sample_bboxes.shape[-1] == 4:
                        end_idx = current_idx + n_labels
                        cls_array[current_idx:end_idx] = sample_cls
                        bboxes_array[current_idx:end_idx] = sample_bboxes.reshape(-1, 4)
                        batch_idx_array[current_idx:end_idx] = batch_i
                        current_idx = end_idx
                
                return {
                    'img': images,
                    'batch_idx': torch.from_numpy(batch_idx_array),
                    'cls': torch.from_numpy(cls_array),
                    'bboxes': torch.from_numpy(bboxes_array),
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch]
                }
            except Exception as e:
                print(f"Collate error: {e}")
                raise
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=test_collate_fn,
            num_workers=0  # Single threaded for testing
        )
        
        print(f"✅ Created DataLoader with batch_size={batch_size}")
        
        # Test first few batches
        batch_issues = []
        for batch_idx, batch in enumerate(dataloader):
            try:
                img = batch['img']
                cls = batch['cls'] 
                bboxes = batch['bboxes']
                batch_idx_tensor = batch['batch_idx']
                
                # Validate batch
                assert img.ndim == 4, f"img should be 4D, got {img.ndim}D"
                assert cls.ndim == 1, f"cls should be 1D, got {cls.ndim}D"
                assert bboxes.ndim == 2, f"bboxes should be 2D, got {bboxes.ndim}D"
                assert len(cls) == len(bboxes), f"cls/bbox length mismatch: {len(cls)} vs {len(bboxes)}"
                
                print(f"  ✅ Batch {batch_idx}: img={img.shape}, cls={cls.shape}, bboxes={bboxes.shape}")
                
                if batch_idx >= 5:  # Test first 5 batches
                    break
                    
            except Exception as e:
                batch_issues.append((batch_idx, str(e)))
                print(f"  ❌ Batch {batch_idx}: {e}")
                if len(batch_issues) >= 3:  # Stop after 3 batch errors
                    break
        
        if batch_issues:
            print(f"\n❌ Found {len(batch_issues)} batch processing issues")
            return False
        else:
            print(f"\n✅ Collate function test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Collate function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test dataset creation logic thoroughly")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--mode", choices=['train', 'val'], default='train', help="Dataset mode to test")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to test (default: all)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for collate testing")
    
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    # Test individual samples
    samples_ok = test_every_dataset_sample(args.zarr_path, args.mode, args.max_samples)
    
    # Test collate function
    collate_ok = test_collate_function(args.zarr_path, args.batch_size)
    
    print()
    print("🎯 FINAL VERDICT")
    print("-" * 15)
    if samples_ok and collate_ok:
        print("✅ All tests passed! Dataset should work for YOLO training.")
    else:
        print("❌ Issues found! Dataset may cause training problems.")
        if not samples_ok:
            print("   - Individual sample issues detected")
        if not collate_ok:
            print("   - Batch collation issues detected")

if __name__ == "__main__":
    main()