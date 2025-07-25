# minimal_debug_trainer.py (Minimal settings to isolate the issue)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """Minimal training setup to isolate the 0-d tensor issue."""

    class MinimalTrainer(DetectionTrainer):

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """Ultra-minimal dataloader setup."""
            
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'
            )
            
            # Absolutely minimal collate function
            def minimal_collate_fn(batch):
                """Minimal collate with maximum error checking."""
                
                print(f"🔧 Processing batch with {len(batch)} samples")
                
                try:
                    # Stack images
                    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                    print(f"   Images: {images.shape}")
                    
                    # Collect all labels
                    all_cls = []
                    all_bboxes = []
                    all_batch_idx = []
                    
                    for batch_i, sample in enumerate(batch):
                        cls = sample['cls']
                        bboxes = sample['bboxes']
                        
                        print(f"   Sample {batch_i}: cls={cls} (shape={cls.shape}, ndim={cls.ndim}), bbox={bboxes.shape}")
                        
                        # Ensure cls is always 1D
                        if cls.ndim == 0:
                            print(f"   ⚠️  Converting 0-d cls to 1-d for sample {batch_i}")
                            cls = np.array([cls])
                        
                        all_cls.extend(cls)
                        all_bboxes.extend(bboxes)
                        all_batch_idx.extend([batch_i] * len(cls))
                    
                    # Convert to tensors
                    cls_tensor = torch.tensor(all_cls, dtype=torch.float32)
                    bbox_tensor = torch.tensor(all_bboxes, dtype=torch.float32)
                    batch_idx_tensor = torch.tensor(all_batch_idx, dtype=torch.long)
                    
                    print(f"   Final tensors: cls={cls_tensor.shape} (ndim={cls_tensor.ndim}), bbox={bbox_tensor.shape}")
                    
                    # Critical assertion
                    assert cls_tensor.ndim == 1, f"cls_tensor must be 1D, got {cls_tensor.ndim}D"
                    assert len(cls_tensor) > 0, f"cls_tensor cannot be empty"
                    
                    result = {
                        'img': images,
                        'batch_idx': batch_idx_tensor,
                        'cls': cls_tensor,
                        'bboxes': bbox_tensor,
                        'im_file': [s['im_file'] for s in batch],
                        'ori_shape': [s['ori_shape'] for s in batch],
                        'ratio_pad': [s['ratio_pad'] for s in batch]
                    }
                    
                    print(f"   ✅ Batch processed successfully")
                    return result
                    
                except Exception as e:
                    print(f"   ❌ Collate error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Ultra-minimal DataLoader settings
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=minimal_collate_fn,
                num_workers=0,  # Single-threaded
                pin_memory=False,
                persistent_workers=False,
                drop_last=False
            )

    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Starting Minimal Debug Training ---")
    print("🔧 Settings: Single-threaded, no AMP, minimal batch size, verbose logging")
    
    try:
        model.train(
            trainer=MinimalTrainer,
            data=args.config_file,
            epochs=1,  # Just one epoch to trigger the error
            batch=2,   # Tiny batch size
            imgsz=args.img_size,
            device=args.device,
            plots=False,
            val=True,  # Keep validation to trigger the error
            amp=False,  # No mixed precision
            cache=False,
            workers=0,  # Single-threaded
            patience=10,
            verbose=True,
            save=False  # Don't save checkpoints
        )
        
        print("--- Training completed without errors! ---")
        
    except TypeError as e:
        if "len() of a 0-d tensor" in str(e):
            print(f"\n🎯 CAUGHT THE 0-D TENSOR ERROR!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            print(f"\n🔍 This confirms the error is in YOLO's validation logic, not our data")
            return False
        else:
            print(f"Different error: {e}")
            raise
            
    except Exception as e:
        print(f"Other error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Minimal debug training to isolate 0-d tensor error")
    
    parser.add_argument('--zarr-path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='zarr_bbox_config.yaml')
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--model-name', type=str, default='yolov8n.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()
    
    print(f"🔬 MINIMAL DEBUG TRAINING")
    print(f"   Goal: Isolate the exact source of the 0-d tensor error")
    print(f"   Strategy: Minimal settings with maximum logging")
    
    success = main(args)
    
    if not success:
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Try different YOLO version: pip install ultralytics==8.0.0")
        print(f"   2. Try different PyTorch version")
        print(f"   3. Disable validation entirely with val=False")
        print(f"   4. Check if this is a known YOLO bug")
    else:
        print(f"\n🎉 SUCCESS! The minimal setup worked without errors.")
        print(f"   Try gradually increasing complexity to find the trigger.")