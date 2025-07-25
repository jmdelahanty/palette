#!/usr/bin/env python3
"""
Debug Invalid Samples
Quick script to examine specific samples that might be causing the 0-d tensor error.
"""

import zarr
import numpy as np
import argparse
from zarr_yolo_dataset_bbox import ZarrYOLODataset

def debug_dataset_samples(zarr_path, num_samples=10):
    """
    Debug the first few samples from the dataset to see what's going wrong.
    """
    print(f"🔍 Debugging dataset samples from: {zarr_path}")
    print("=" * 60)
    
    try:
        # Create dataset instances
        train_dataset = ZarrYOLODataset(zarr_path=zarr_path, mode='train', task='detect')
        val_dataset = ZarrYOLODataset(zarr_path=zarr_path, mode='val', task='detect')
        
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Val dataset: {len(val_dataset)} samples")
        print()
        
        # Debug training samples
        print("🚂 TRAINING SAMPLES DEBUG:")
        print("-" * 30)
        
        for i in range(min(num_samples, len(train_dataset))):
            try:
                sample = train_dataset[i]
                zarr_index = train_dataset.indices[i]
                
                print(f"Train Sample {i} (zarr_index {zarr_index}):")
                print(f"  img shape: {sample['img'].shape}")
                print(f"  cls: {sample['cls']} (type: {type(sample['cls'])}, shape: {sample['cls'].shape if hasattr(sample['cls'], 'shape') else 'N/A'})")
                print(f"  bboxes: {sample['bboxes']} (shape: {sample['bboxes'].shape if hasattr(sample['bboxes'], 'shape') else 'N/A'})")
                print(f"  cls dtype: {sample['cls'].dtype if hasattr(sample['cls'], 'dtype') else 'N/A'}")
                print(f"  cls ndim: {sample['cls'].ndim if hasattr(sample['cls'], 'ndim') else 'N/A'}")
                
                # Check if cls is 0-dimensional
                if hasattr(sample['cls'], 'ndim') and sample['cls'].ndim == 0:
                    print(f"  ❌ PROBLEM: cls is 0-dimensional! Value: {sample['cls'].item()}")
                elif hasattr(sample['cls'], 'ndim') and sample['cls'].ndim == 1:
                    print(f"  ✅ OK: cls is 1-dimensional with {len(sample['cls'])} elements")
                else:
                    print(f"  ⚠️  UNKNOWN: cls dimension check failed")
                
                print()
                
            except Exception as e:
                print(f"  ❌ ERROR loading train sample {i}: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        # Debug validation samples
        print("✅ VALIDATION SAMPLES DEBUG:")
        print("-" * 32)
        
        for i in range(min(num_samples, len(val_dataset))):
            try:
                sample = val_dataset[i]
                zarr_index = val_dataset.indices[i]
                
                print(f"Val Sample {i} (zarr_index {zarr_index}):")
                print(f"  img shape: {sample['img'].shape}")
                print(f"  cls: {sample['cls']} (type: {type(sample['cls'])}, shape: {sample['cls'].shape if hasattr(sample['cls'], 'shape') else 'N/A'})")
                print(f"  bboxes: {sample['bboxes']} (shape: {sample['bboxes'].shape if hasattr(sample['bboxes'], 'shape') else 'N/A'})")
                
                # Check if cls is 0-dimensional
                if hasattr(sample['cls'], 'ndim') and sample['cls'].ndim == 0:
                    print(f"  ❌ PROBLEM: cls is 0-dimensional! Value: {sample['cls'].item()}")
                elif hasattr(sample['cls'], 'ndim') and sample['cls'].ndim == 1:
                    print(f"  ✅ OK: cls is 1-dimensional with {len(sample['cls'])} elements")
                else:
                    print(f"  ⚠️  UNKNOWN: cls dimension check failed")
                
                print()
                
            except Exception as e:
                print(f"  ❌ ERROR loading val sample {i}: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        # Test the collate function manually
        print("🔧 COLLATE FUNCTION TEST:")
        print("-" * 25)
        
        try:
            # Get a small batch from training
            batch_samples = []
            for i in range(min(4, len(train_dataset))):
                sample = train_dataset[i]
                batch_samples.append(sample)
            
            print(f"Testing collate with {len(batch_samples)} samples...")
            
            # Manual collate (simplified version)
            images = np.stack([s['img'] for s in batch_samples])
            all_cls = [s['cls'] for s in batch_samples]
            all_bboxes = [s['bboxes'] for s in batch_samples]
            
            print(f"Images stacked: {images.shape}")
            
            for i, (cls, bbox) in enumerate(zip(all_cls, all_bboxes)):
                print(f"  Sample {i}: cls {cls} (ndim: {cls.ndim}), bbox {bbox.shape}")
                if cls.ndim == 0:
                    print(f"    ❌ Found 0-d tensor in batch sample {i}!")
            
            # Try to process like the collate function would
            total_labels = sum(len(s['cls']) for s in batch_samples)
            print(f"Total labels to process: {total_labels}")
            
            # Check each sample's cls processing
            for i, sample in enumerate(batch_samples):
                sample_cls = np.asarray(sample['cls'], dtype=np.float32)
                print(f"  Sample {i} cls processing:")
                print(f"    Original: {sample['cls']} (ndim: {sample['cls'].ndim})")
                print(f"    After asarray: {sample_cls} (ndim: {sample_cls.ndim})")
                
                if sample_cls.ndim == 0:
                    fixed_cls = np.array([sample_cls], dtype=np.float32)
                    print(f"    After 0-d fix: {fixed_cls} (ndim: {fixed_cls.ndim})")
                
                final_cls = sample_cls.flatten() if sample_cls.ndim > 0 else np.array([sample_cls], dtype=np.float32).flatten()
                print(f"    Final: {final_cls} (ndim: {final_cls.ndim}, len: {len(final_cls)})")
                print()
            
        except Exception as e:
            print(f"❌ Error in collate test: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        import traceback
        traceback.print_exc()

def check_zarr_data_directly(zarr_path, sample_indices=[0, 1, 2, 3, 4]):
    """
    Check the raw Zarr data to see if the issue is in data generation or dataset loading.
    """
    print()
    print("🔬 DIRECT ZARR DATA CHECK:")
    print("-" * 30)
    
    try:
        root = zarr.open(zarr_path, mode='r')
        tracking_results = root['tracking/tracking_results']
        
        # Check the tracking results directly
        for idx in sample_indices:
            if idx >= tracking_results.shape[0]:
                continue
                
            data = tracking_results[idx]
            print(f"Frame {idx} raw data:")
            print(f"  Data: {data}")
            print(f"  Data shape: {data.shape}")
            print(f"  Data dtype: {data.dtype}")
            print(f"  Any NaN?: {np.any(np.isnan(data))}")
            print(f"  Heading (col 0): {data[0]}")
            
            # Check what _get_bbox_data would return
            heading = data[0]
            if not np.isnan(heading):
                # Mock the enhanced format bbox extraction
                if tracking_results.shape[1] >= 20:  # Enhanced format
                    bbox_x = data[7]    # bbox_x_norm_ds
                    bbox_y = data[8]    # bbox_y_norm_ds  
                    bbox_w = data[9]    # bbox_width_norm_ds
                    bbox_h = data[10]   # bbox_height_norm_ds
                    confidence = data[19] if not np.isnan(data[19]) else 1.0
                    
                    bbox_data = np.array([0, bbox_x, bbox_y, bbox_w, bbox_h, confidence], dtype=np.float32)
                    print(f"  Enhanced bbox_data: {bbox_data}")
                else:
                    # Original format
                    bbox_x = data[1]    # bbox_x_norm
                    bbox_y = data[2]    # bbox_y_norm
                    bbox_data = np.array([0, bbox_x, bbox_y, 0.05, 0.05, 1.0], dtype=np.float32)
                    print(f"  Original bbox_data: {bbox_data}")
                
                # What would be returned as cls and bboxes
                label = bbox_data[:5].reshape(1, -1)
                cls_labels = label[:, 0].astype(np.float32)
                bbox_coords = label[:, 1:5].astype(np.float32)
                
                print(f"  Would return cls: {cls_labels} (shape: {cls_labels.shape}, ndim: {cls_labels.ndim})")
                print(f"  Would return bbox: {bbox_coords} (shape: {bbox_coords.shape})")
            else:
                print(f"  ❌ Invalid heading - would be skipped")
            
            print()
            
    except Exception as e:
        print(f"❌ Error checking Zarr data: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Debug dataset samples for 0-d tensor issues")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to debug")
    
    args = parser.parse_args()
    
    debug_dataset_samples(args.zarr_path, args.num_samples)
    check_zarr_data_directly(args.zarr_path)

if __name__ == "__main__":
    main()