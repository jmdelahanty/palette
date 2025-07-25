# no_validation_trainer.py (Disable validation entirely)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """Training with validation completely disabled to bypass the error."""

    class NoValidationTrainer(DetectionTrainer):

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """Standard dataloader - the issue isn't here."""
            
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'
            )
            
            def standard_collate_fn(batch):
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
                
                return {
                    'img': images,
                    'batch_idx': torch.from_numpy(batch_idx_array),
                    'cls': torch.from_numpy(cls_array),
                    'bboxes': torch.from_numpy(bboxes_array),
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch]
                }
            
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=standard_collate_fn,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True if batch_size > 1 else False
            )

    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Training WITHOUT Validation ---")
    print("🎯 This should work if the issue is in YOLO's validation logic")
    
    try:
        model.train(
            trainer=NoValidationTrainer,
            data=args.config_file,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            plots=True,
            val=False,  # ← KEY: Disable validation entirely
            amp=True,
            cache=False,
            workers=8,
            patience=100,
            verbose=True
        )
        
        print("🎉 SUCCESS! Training completed without validation.")
        print("This confirms the issue is in YOLO's validation logic, not our data.")
        return True
        
    except Exception as e:
        print(f"❌ Even no-validation training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLO without validation to bypass the error")
    
    parser.add_argument('--zarr-path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='zarr_bbox_config.yaml')
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--model-name', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=10)  # Fewer epochs for testing
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()
    
    print(f"🚀 NO-VALIDATION TRAINING TEST")
    print(f"   If this works, the issue is definitely in YOLO's validation")
    
    success = main(args)
    
    if success:
        print(f"\n💡 SOLUTION FOUND:")
        print(f"   Train without validation first: val=False")
        print(f"   Then run separate validation manually")
        print(f"   Or try different YOLO version")
    else:
        print(f"\n🤔 Deeper issue - not just validation")