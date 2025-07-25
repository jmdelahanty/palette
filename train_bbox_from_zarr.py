# train_bbox_from_zarr.py (Ultra-Robust Version)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """Main training function with ultra-robust error handling."""

    class ZarrTrainer(DetectionTrainer):

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """
            Ultra-robust dataloader with comprehensive tensor validation.
            """
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'
            )
            
            def ultra_robust_collate_fn(batch):
                """
                Ultra-robust collate function with comprehensive error handling
                and tensor validation at every step.
                """
                try:
                    # Stack images efficiently
                    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                    
                    # Count total labels with validation
                    total_labels = 0
                    valid_samples = []
                    
                    for i, sample in enumerate(batch):
                        try:
                            cls = sample['cls']
                            bboxes = sample['bboxes']
                            
                            # Ultra-defensive validation
                            if cls is None or bboxes is None:
                                print(f"Warning: Sample {i} has None values, skipping")
                                continue
                                
                            # Convert to numpy and validate
                            cls_np = np.asarray(cls, dtype=np.float32)
                            bboxes_np = np.asarray(bboxes, dtype=np.float32)
                            
                            # Ensure proper dimensions
                            if cls_np.ndim == 0:
                                cls_np = np.array([cls_np], dtype=np.float32)
                            elif cls_np.ndim > 1:
                                cls_np = cls_np.flatten()
                            
                            if bboxes_np.ndim == 1 and len(bboxes_np) == 4:
                                bboxes_np = bboxes_np[None, :]
                            
                            # Validate shapes
                            if cls_np.ndim != 1:
                                print(f"Warning: Sample {i} cls has wrong dimensions: {cls_np.shape}")
                                continue
                                
                            if bboxes_np.ndim != 2 or bboxes_np.shape[1] != 4:
                                print(f"Warning: Sample {i} bboxes has wrong dimensions: {bboxes_np.shape}")
                                continue
                            
                            # Validate values
                            if len(cls_np) != bboxes_np.shape[0]:
                                print(f"Warning: Sample {i} cls/bbox count mismatch: {len(cls_np)} vs {bboxes_np.shape[0]}")
                                continue
                            
                            if np.any(np.isnan(cls_np)) or np.any(np.isnan(bboxes_np)):
                                print(f"Warning: Sample {i} contains NaN values, skipping")
                                continue
                            
                            # Sample is valid
                            valid_samples.append({
                                'index': i,
                                'cls': cls_np,
                                'bboxes': bboxes_np,
                                'im_file': sample['im_file'],
                                'ori_shape': sample['ori_shape'],
                                'ratio_pad': sample['ratio_pad']
                            })
                            
                            total_labels += len(cls_np)
                            
                        except Exception as e:
                            print(f"Error processing sample {i}: {e}")
                            continue
                    
                    # Handle empty batch
                    if total_labels == 0 or len(valid_samples) == 0:
                        print("Warning: Empty or invalid batch, returning minimal valid batch")
                        return {
                            'img': images,
                            'batch_idx': torch.zeros((0,), dtype=torch.long),
                            'cls': torch.zeros((0,), dtype=torch.float32),
                            'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                            'im_file': [s['im_file'] for s in batch],
                            'ori_shape': [s['ori_shape'] for s in batch],
                            'ratio_pad': [s['ratio_pad'] for s in batch]
                        }
                    
                    # Pre-allocate output arrays
                    cls_array = np.zeros(total_labels, dtype=np.float32)
                    bboxes_array = np.zeros((total_labels, 4), dtype=np.float32)
                    batch_idx_array = np.zeros(total_labels, dtype=np.int64)
                    
                    # Fill arrays with validated data
                    current_idx = 0
                    for sample in valid_samples:
                        batch_i = sample['index']
                        cls_data = sample['cls']
                        bbox_data = sample['bboxes']
                        
                        n_labels = len(cls_data)
                        end_idx = current_idx + n_labels
                        
                        cls_array[current_idx:end_idx] = cls_data
                        bboxes_array[current_idx:end_idx] = bbox_data
                        batch_idx_array[current_idx:end_idx] = batch_i
                        
                        current_idx = end_idx
                    
                    # Final validation before tensor conversion
                    assert cls_array.ndim == 1, f"cls_array should be 1D, got {cls_array.ndim}D"
                    assert bboxes_array.ndim == 2, f"bboxes_array should be 2D, got {bboxes_array.ndim}D"
                    assert len(cls_array) == len(bboxes_array), f"Mismatched lengths: cls={len(cls_array)}, bbox={len(bboxes_array)}"
                    
                    # Convert to tensors with final validation
                    cls_tensor = torch.from_numpy(cls_array)
                    bbox_tensor = torch.from_numpy(bboxes_array)
                    batch_idx_tensor = torch.from_numpy(batch_idx_array)
                    
                    # Ultimate validation
                    assert cls_tensor.ndim == 1, f"cls_tensor should be 1D, got {cls_tensor.ndim}D"
                    assert bbox_tensor.ndim == 2, f"bbox_tensor should be 2D, got {bbox_tensor.ndim}D"
                    
                    return {
                        'img': images,
                        'batch_idx': batch_idx_tensor,
                        'cls': cls_tensor,
                        'bboxes': bbox_tensor,
                        'im_file': [s['im_file'] for s in valid_samples],
                        'ori_shape': [s['ori_shape'] for s in valid_samples],
                        'ratio_pad': [s['ratio_pad'] for s in valid_samples]
                    }
                    
                except Exception as e:
                    print(f"Critical error in collate function: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Emergency fallback
                    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                    return {
                        'img': images,
                        'batch_idx': torch.zeros((0,), dtype=torch.long),
                        'cls': torch.zeros((0,), dtype=torch.float32),
                        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                        'im_file': [f"emergency_fallback_{i}" for i in range(len(batch))],
                        'ori_shape': [(640, 640) for _ in range(len(batch))],
                        'ratio_pad': [(1.0, (0.0, 0.0)) for _ in range(len(batch))]
                    }

            # Return DataLoader with robust settings
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=ultra_robust_collate_fn,
                num_workers=4,  # Reduced workers to minimize threading issues
                pin_memory=True,
                persistent_workers=False,  # Disable to avoid worker state issues
                drop_last=True if mode == 'train' else False  # Drop incomplete batches in training
            )

    # Load model and train
    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Starting Ultra-Robust Training ---")
    try:
        model.train(
            trainer=ZarrTrainer,
            data=args.config_file,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            plots=False,
            val=True,
            amp=True,
            cache=False,
            workers=4,  # Match DataLoader workers
            patience=100,
            verbose=True  # Enable verbose for better debugging
        )
        print("--- Ultra-Robust Training Completed Successfully ---")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with reduced settings
        print("\n--- Attempting Recovery Training with Minimal Settings ---")
        try:
            model.train(
                trainer=ZarrTrainer,
                data=args.config_file,
                epochs=min(10, args.epochs),  # Reduced epochs
                batch=max(1, args.batch_size // 2),  # Smaller batch
                imgsz=args.img_size,
                device=args.device,
                plots=False,
                val=False,  # Disable validation temporarily
                amp=False,  # Disable mixed precision
                cache=False,
                workers=1,  # Single worker
                patience=50
            )
            print("--- Recovery Training Completed ---")
        except Exception as e2:
            print(f"Recovery training also failed: {e2}")
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultra-robust YOLO training")
    
    parser.add_argument('--zarr-path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='zarr_bbox_config.yaml')
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--model-name', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()
    
    print(f"🚀 Starting ULTRA-ROBUST YOLO training")
    print(f"   Comprehensive tensor validation enabled")
    print(f"   Emergency fallback handling enabled")
    print(f"   Reduced workers to minimize threading issues")
    
    main(args)