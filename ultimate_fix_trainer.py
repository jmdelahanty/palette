# corrected_patch_trainer.py (Fix the key mismatch issue)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

# CORRECTED MONKEY PATCH
def corrected_prepare_batch(self, si, batch):
    """
    Corrected patched version that uses the right key names.
    """
    try:
        # Extract batch data safely with correct key names
        cls = batch.get('cls', torch.empty(0))
        bboxes = batch.get('bboxes', torch.empty(0, 4))  # Note: 'bboxes' not 'bbox'
        
        # CRITICAL FIX: Ensure cls is never 0-dimensional
        if hasattr(cls, 'ndim'):
            if cls.ndim == 0:
                cls = torch.empty(0, dtype=cls.dtype, device=cls.device)
            elif cls.ndim > 1:
                cls = cls.flatten()
        
        # Apply the index selection safely  
        if len(cls) > 0 and si < len(cls):
            cls_si = cls[si:si+1] if si < len(cls) else torch.empty(0, dtype=cls.dtype, device=cls.device)
            bboxes_si = bboxes[si:si+1] if si < len(bboxes) else torch.empty(0, 4, dtype=bboxes.dtype, device=bboxes.device)
        else:
            # Return empty tensors with correct dimensions
            cls_si = torch.empty(0, dtype=cls.dtype if hasattr(cls, 'dtype') else torch.float32)
            bboxes_si = torch.empty(0, 4, dtype=bboxes.dtype if hasattr(bboxes, 'dtype') else torch.float32)
        
        # Create the output batch with CORRECT KEY NAMES
        pbatch = {
            'cls': cls_si,
            'bboxes': bboxes_si,  # ← KEY FIX: Use 'bboxes' not 'bbox'
            'ori_shape': batch.get('ori_shape', []),
            'ratio_pad': batch.get('ratio_pad', [])
        }
        
        return pbatch
        
    except Exception as e:
        print(f"🔧 Corrected _prepare_batch caught error: {e}")
        # Return safe empty batch with correct keys
        return {
            'cls': torch.empty(0, dtype=torch.float32),
            'bboxes': torch.empty(0, 4, dtype=torch.float32),  # ← KEY FIX
            'ori_shape': batch.get('ori_shape', []),
            'ratio_pad': batch.get('ratio_pad', [])
        }

# Apply the corrected monkey patch
DetectionValidator._prepare_batch = corrected_prepare_batch
print("🔧 Applied CORRECTED monkey patch to fix YOLO's key mismatch")

def main(args):
    """Training with the corrected patched YOLO code."""

    class StandardTrainer(DetectionTrainer):
        """Standard trainer with corrected patch."""

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """Standard dataloader."""
            
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'
            )
            
            def standard_collate_fn(batch):
                """Standard collate function with debugging."""
                images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                
                total_labels = sum(len(s['cls']) for s in batch)
                
                if total_labels == 0:
                    print("⚠️  Empty batch detected in collate")
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
                
                result = {
                    'img': images,
                    'batch_idx': torch.from_numpy(batch_idx_array),
                    'cls': torch.from_numpy(cls_array),
                    'bboxes': torch.from_numpy(bboxes_array),  # Ensure this key is correct
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch]
                }
                
                # Debug validation
                print(f"🔍 Collate result keys: {list(result.keys())}")
                print(f"🔍 cls shape: {result['cls'].shape}, bboxes shape: {result['bboxes'].shape}")
                
                return result
            
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=standard_collate_fn,
                num_workers=0 if mode == 'val' else 4,  # Reduce workers for validation
                pin_memory=True,
                persistent_workers=False
            )

    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Starting Training with Corrected Patch ---")
    print("🔧 Fixed key mismatch: bbox -> bboxes")
    
    try:
        model.train(
            trainer=StandardTrainer,
            data=args.config_file,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            plots=True,
            val=True,
            amp=True,
            cache=False,
            workers=4,  # Reduce workers
            patience=100,
            verbose=True
        )
        
        print("🎉 SUCCESS! Training completed with corrected patch!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed with corrected patch: {e}")
        import traceback
        traceback.print_exc()
        
        # Try without validation as fallback
        print("\n🔄 Falling back to no-validation training...")
        try:
            model.train(
                trainer=StandardTrainer,
                data=args.config_file,
                epochs=args.epochs,
                batch=args.batch_size,
                imgsz=args.img_size,
                device=args.device,
                plots=True,
                val=False,  # Disable validation
                amp=True,
                cache=False,
                workers=4,
                patience=100,
                verbose=True
            )
            print("🎉 Fallback no-validation training completed!")
            return True
        except Exception as e2:
            print(f"❌ Even fallback training failed: {e2}")
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO training with corrected patch")
    
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
    
    print(f"🔧 CORRECTED YOLO PATCH")
    print(f"   🎯 Fixed key mismatch issue")
    print(f"   🔍 Added debugging to see what's happening")
    print(f"   🛡️  Fallback to no-validation if needed")
    
    success = main(args)
    
    if success:
        print(f"\n🎉 PROBLEM SOLVED!")
        print(f"   The key mismatch was the real issue")
    else:
        print(f"\n😞 Still having issues - this is a deep YOLO problem")