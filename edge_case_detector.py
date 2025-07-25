#!/usr/bin/env python3
"""
Edge Case Detection Script
Look for subtle edge cases that might cause YOLO validation to fail.
"""

import zarr
import numpy as np
import argparse
from zarr_yolo_dataset_bbox import ZarrYOLODataset
import torch
from torch.utils.data import DataLoader

def check_for_edge_cases(zarr_path):
    """
    Look for edge cases that might cause YOLO validation issues.
    """
    print(f"🔍 CHECKING FOR SUBTLE EDGE CASES")
    print(f"📁 Zarr path: {zarr_path}")
    print("=" * 60)
    
    # Test both train and validation datasets
    for mode in ['train', 'val']:
        print(f"\n🔬 TESTING {mode.upper()} DATASET")
        print("-" * 30)
        
        try:
            dataset = ZarrYOLODataset(zarr_path=zarr_path, mode=mode, task='detect')
            print(f"✅ Created {mode} dataset: {len(dataset)} samples")
            
            # Check for potential edge cases
            edge_cases = {
                'extremely_small_bbox': [],
                'extremely_large_bbox': [],
                'bbox_at_boundaries': [],
                'unusual_coordinates': [],
                'tensor_dtype_issues': [],
                'empty_or_zero_data': []
            }
            
            # Test a substantial sample
            test_size = min(200, len(dataset))
            print(f"Testing {test_size} samples for edge cases...")
            
            for i in range(test_size):
                try:
                    sample = dataset[i]
                    zarr_idx = dataset.indices[i]
                    
                    cls = sample['cls']
                    bboxes = sample['bboxes']
                    
                    # Check tensor properties
                    if cls.dtype != np.float32:
                        edge_cases['tensor_dtype_issues'].append((i, zarr_idx, f"cls dtype: {cls.dtype}"))
                    
                    if bboxes.dtype != np.float32:
                        edge_cases['tensor_dtype_issues'].append((i, zarr_idx, f"bbox dtype: {bboxes.dtype}"))
                    
                    # Check bbox coordinates
                    x, y, w, h = bboxes[0]
                    
                    # Extremely small bboxes (might cause numerical issues)
                    if w < 0.001 or h < 0.001:
                        edge_cases['extremely_small_bbox'].append((i, zarr_idx, f"tiny bbox: w={w:.6f}, h={h:.6f}"))
                    
                    # Extremely large bboxes
                    if w > 0.5 or h > 0.5:
                        edge_cases['extremely_large_bbox'].append((i, zarr_idx, f"large bbox: w={w:.6f}, h={h:.6f}"))
                    
                    # Bboxes at image boundaries (might cause issues)
                    if x < 0.01 or x > 0.99 or y < 0.01 or y > 0.99:
                        edge_cases['bbox_at_boundaries'].append((i, zarr_idx, f"boundary bbox: x={x:.3f}, y={y:.3f}"))
                    
                    # Check for unusual coordinate patterns
                    if x == 0.5 and y == 0.5 and w == 0.1 and h == 0.1:
                        edge_cases['unusual_coordinates'].append((i, zarr_idx, "exact fallback coordinates"))
                    
                    # Check for zeros or empty data
                    if np.any(cls == 0) and (w == 0 or h == 0):
                        edge_cases['empty_or_zero_data'].append((i, zarr_idx, f"zero dimension: w={w}, h={h}"))
                    
                    # Show progress
                    if i < 5 or i % 50 == 0:
                        print(f"  Sample {i}: bbox=({x:.4f}, {y:.4f}, {w:.6f}, {h:.6f})")
                        
                except Exception as e:
                    print(f"  ❌ Error testing sample {i}: {e}")
            
            # Report edge cases
            total_edge_cases = sum(len(cases) for cases in edge_cases.values())
            if total_edge_cases > 0:
                print(f"\n⚠️  Found {total_edge_cases} edge cases in {mode} dataset:")
                for case_type, cases in edge_cases.items():
                    if cases:
                        print(f"   {case_type}: {len(cases)} samples")
                        for sample_idx, zarr_idx, details in cases[:3]:
                            print(f"      Sample {sample_idx} (zarr {zarr_idx}): {details}")
                        if len(cases) > 3:
                            print(f"      ... and {len(cases) - 3} more")
            else:
                print(f"✅ No edge cases found in {mode} dataset")
                
        except Exception as e:
            print(f"❌ Error testing {mode} dataset: {e}")
            import traceback
            traceback.print_exc()

def test_yolo_validation_scenario(zarr_path, batch_size=16):
    """
    Try to reproduce the exact scenario that causes YOLO validation to fail.
    """
    print(f"\n🎯 TESTING YOLO VALIDATION SCENARIO")
    print("-" * 40)
    
    try:
        # Create validation dataset (this is where the error occurs)
        val_dataset = ZarrYOLODataset(zarr_path=zarr_path, mode='val', task='detect')
        print(f"✅ Created validation dataset: {len(val_dataset)} samples")
        
        # Create the exact collate function used in training
        def yolo_collate_fn(batch):
            """Exact replica of the collate function used in YOLO training"""
            images = torch.from_numpy(np.stack([s['img'] for s in batch]))
            
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
                
                # Critical: Ensure proper dimensions
                if sample_cls.ndim == 0:
                    sample_cls = np.array([sample_cls], dtype=np.float32)
                sample_cls = sample_cls.flatten()
                
                if sample_bboxes.ndim == 1 and len(sample_bboxes) == 4:
                    sample_bboxes = sample_bboxes[None, :]
                
                n_labels = len(sample_cls)
                if n_labels > 0 and sample_bboxes.shape[-1] == 4:
                    end_idx = current_idx + n_labels
                    cls_array[current_idx:end_idx] = sample_cls
                    bboxes_array[current_idx:end_idx] = sample_bboxes.reshape(-1, 4)
                    batch_idx_array[current_idx:end_idx] = batch_i
                    current_idx = end_idx
            
            # Final validation
            cls_tensor = torch.from_numpy(cls_array)
            bbox_tensor = torch.from_numpy(bboxes_array)
            batch_idx_tensor = torch.from_numpy(batch_idx_array)
            
            # This is where YOLO validation might fail - let's check dimensions
            print(f"    Batch validation: cls.ndim={cls_tensor.ndim}, cls.shape={cls_tensor.shape}")
            
            # Simulate what YOLO's validation code does
            try:
                cls_len = len(cls_tensor)  # This line causes "len() of a 0-d tensor" if cls_tensor is 0-d
                print(f"    YOLO len() test passed: len(cls) = {cls_len}")
            except Exception as e:
                print(f"    ❌ YOLO len() test failed: {e}")
                print(f"    cls_tensor details: shape={cls_tensor.shape}, ndim={cls_tensor.ndim}, type={type(cls_tensor)}")
                raise
            
            return {
                'img': images,
                'batch_idx': batch_idx_tensor,
                'cls': cls_tensor,
                'bboxes': bbox_tensor,
                'im_file': [s['im_file'] for s in batch],
                'ori_shape': [s['ori_shape'] for s in batch],
                'ratio_pad': [s['ratio_pad'] for s in batch]
            }
        
        # Create DataLoader with the exact same settings as training
        dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Validation doesn't shuffle
            collate_fn=yolo_collate_fn,
            num_workers=0,  # Single-threaded for testing
            drop_last=False  # Don't drop incomplete batches in validation
        )
        
        print(f"✅ Created validation DataLoader with batch_size={batch_size}")
        
        # Process several batches to find the problematic one
        problematic_batches = []
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                print(f"  Testing validation batch {batch_idx}...")
                
                # Simulate YOLO's validation processing
                cls = batch['cls']
                bboxes = batch['bboxes']
                
                # This is the exact line that fails in YOLO validation
                if len(cls):  # ← This line causes the error if cls is 0-d
                    print(f"    ✅ Batch {batch_idx}: cls.shape={cls.shape}, len(cls)={len(cls)}")
                else:
                    print(f"    ⚠️  Batch {batch_idx}: Empty batch (len(cls)=0)")
                
                if batch_idx >= 10:  # Test first 10 batches
                    break
                    
            except TypeError as e:
                if "len() of a 0-d tensor" in str(e):
                    problematic_batches.append(batch_idx)
                    print(f"    ❌ Batch {batch_idx}: FOUND THE PROBLEM! {e}")
                    print(f"    cls details: type={type(batch['cls'])}, shape={batch['cls'].shape}, ndim={batch['cls'].ndim}")
                    
                    # Analyze the problematic batch
                    print(f"    Analyzing problematic batch...")
                    for sample_idx in range(len(batch['im_file'])):
                        print(f"      Sample {sample_idx}: {batch['im_file'][sample_idx]}")
                    
                    break
                else:
                    print(f"    ❌ Batch {batch_idx}: Unexpected error: {e}")
                    break
            
            except Exception as e:
                print(f"    ❌ Batch {batch_idx}: Other error: {e}")
                problematic_batches.append(batch_idx)
                break
        
        if problematic_batches:
            print(f"\n🚨 FOUND PROBLEMATIC BATCHES: {problematic_batches}")
            print("This explains the 0-d tensor error during YOLO validation!")
            return False
        else:
            print(f"\n✅ All validation batches processed successfully")
            print("The 0-d tensor error might be happening elsewhere...")
            return True
            
    except Exception as e:
        print(f"❌ Error in YOLO validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Detect edge cases that might cause YOLO validation issues")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for testing")
    
    args = parser.parse_args()
    
    # Run edge case detection
    check_for_edge_cases(args.zarr_path)
    
    # Test YOLO validation scenario
    validation_ok = test_yolo_validation_scenario(args.zarr_path, args.batch_size)
    
    print(f"\n🎯 FINAL DIAGNOSIS")
    print("-" * 20)
    if validation_ok:
        print("❓ No obvious issues found. The 0-d tensor error might be:")
        print("   1. A race condition in multi-threaded data loading")
        print("   2. An issue in YOLO's internal validation logic")
        print("   3. A PyTorch/YOLO version compatibility problem")
        print("\n💡 Try running training with:")
        print("   - num_workers=0 (single-threaded)")
        print("   - smaller batch size")
        print("   - val=False (disable validation temporarily)")
    else:
        print("🎯 Found the root cause of the 0-d tensor error!")
        print("Check the problematic batch analysis above.")

if __name__ == "__main__":
    main()