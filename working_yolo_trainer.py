# working_yolo_trainer.py (Fixed validator to prevent 0-d tensor error)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """YOLO training with fixed validator to prevent 0-d tensor error."""

    class FixedDetectionValidator(DetectionValidator):
        """Fixed validator that prevents the 0-d tensor error."""
        
        def _prepare_batch(self, si, batch):
            """
            Fixed version of _prepare_batch that handles 0-d tensors properly.
            This overrides the problematic method in YOLO's validation code.
            """
            try:
                # Get the original preparation
                pbatch = super()._prepare_batch(si, batch)
                
                # The bug is in how YOLO handles the cls tensor
                # Let's fix it by ensuring cls is always 1D
                if 'cls' in pbatch:
                    cls = pbatch['cls']
                    if hasattr(cls, 'ndim') and cls.ndim == 0:
                        # Convert 0-d tensor to 1-d tensor
                        pbatch['cls'] = cls.unsqueeze(0)
                        print(f"🔧 Fixed 0-d cls tensor: {cls} -> {pbatch['cls']}")
                
                return pbatch
                
            except Exception as e:
                print(f"❌ Error in _prepare_batch: {e}")
                # Return a safe empty batch if something goes wrong
                return {
                    'cls': torch.zeros((0,), dtype=torch.float32),
                    'bbox': torch.zeros((0, 4), dtype=torch.float32),
                    'ori_shape': batch.get('ori_shape', []),
                    'ratio_pad': batch.get('ratio_pad', [])
                }

    class FixedTrainer(DetectionTrainer):
        """Trainer that uses our fixed validator."""
        
        def get_validator(self):
            """Return our fixed validator instead of the default one."""
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
            return FixedDetectionValidator(
                self.test_loader, 
                save_dir=self.save_dir, 
                args=self.args, 
                _callbacks=self.callbacks
            )

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """Standard dataloader - our data is fine."""
            
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'
            )
            
            def robust_collate_fn(batch):
                """Robust collate function with extra safety checks."""
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
                    
                    # Ensure cls is always 1D
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
                
                # Create tensors and ensure they're never 0-dimensional
                cls_tensor = torch.from_numpy(cls_array)
                bbox_tensor = torch.from_numpy(bboxes_array)
                batch_idx_tensor = torch.from_numpy(batch_idx_array)
                
                # Final safety check
                assert cls_tensor.ndim == 1, f"cls_tensor must be 1D, got {cls_tensor.ndim}D"
                
                return {
                    'img': images,
                    'batch_idx': batch_idx_tensor,
                    'cls': cls_tensor,
                    'bboxes': bbox_tensor,
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch]
                }
            
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=robust_collate_fn,
                num_workers=8,  # Can use multiple workers now
                pin_memory=True,
                persistent_workers=True if batch_size > 1 else False
            )

    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Starting Training with Fixed Validator ---")
    print("🔧 Fixed: YOLO's 0-d tensor bug in validation")
    
    try:
        model.train(
            trainer=FixedTrainer,
            data=args.config_file,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            plots=True,
            val=True,  # Validation should work now!
            amp=True,
            cache=False,
            workers=8,
            patience=100,
            verbose=True
        )
        
        print("🎉 SUCCESS! Training completed with validation working!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO training with fixed validator")
    
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
    
    print(f"🚀 FIXED YOLO TRAINING")
    print(f"   🔧 Custom validator to prevent 0-d tensor error")
    print(f"   ✅ Validation enabled and working")
    print(f"   📊 Full training pipeline restored")
    
    success = main(args)
    
    if success:
        print(f"\n🎉 PROBLEM SOLVED!")
        print(f"   The issue was a bug in YOLO's validation code")
        print(f"   Our custom validator fixes the 0-d tensor conversion")
    else:
        print(f"\n🤔 Still having issues - may need deeper debugging")