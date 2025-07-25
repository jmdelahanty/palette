# train_bbox_from_zarr.py (Performance Optimized - No More Warnings!)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
# Import the specific trainer we need to subclass
from ultralytics.models.yolo.detect import DetectionTrainer
# Import our updated dataset class
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """Main training function."""

    # 1. DEFINE THE CUSTOM TRAINER *INSIDE* MAIN
    class ZarrTrainer(DetectionTrainer):

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """
            Overrides the default data loader to use our custom ZarrYOLODataset
            with an optimized collate function.
            """
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed,
                task='detect'  # Detection task for bounding box training
            )
            
            # --- PERFORMANCE OPTIMIZED COLLATE FUNCTION ---
            def collate_fn(batch):
                """
                High-performance collate function that eliminates tensor creation warnings.
                Pre-allocates numpy arrays and converts to tensors efficiently.
                """
                # Stack images efficiently (numpy first, then tensor)
                images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                
                # Count total labels for efficient pre-allocation
                total_labels = sum(len(s['cls']) for s in batch)
                
                if total_labels == 0:
                    # Handle empty batch efficiently
                    return {
                        'img': images,
                        'batch_idx': torch.zeros((0,), dtype=torch.long),
                        'cls': torch.zeros((0,), dtype=torch.float32),
                        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                        'im_file': [s['im_file'] for s in batch],
                        'ori_shape': [s['ori_shape'] for s in batch],
                        'ratio_pad': [s['ratio_pad'] for s in batch]
                    }
                
                # Pre-allocate arrays for maximum efficiency
                cls_array = np.zeros(total_labels, dtype=np.float32)
                bboxes_array = np.zeros((total_labels, 4), dtype=np.float32)
                batch_idx_array = np.zeros(total_labels, dtype=np.int64)
                
                # Fill arrays in one pass
                current_idx = 0
                for batch_i, sample in enumerate(batch):
                    # Get sample data efficiently
                    sample_cls = np.asarray(sample['cls'], dtype=np.float32).flatten()
                    sample_bboxes = np.asarray(sample['bboxes'], dtype=np.float32)
                    
                    # Ensure proper dimensions
                    if sample_bboxes.ndim == 1 and len(sample_bboxes) == 4:
                        sample_bboxes = sample_bboxes[None, :]
                    
                    n_labels = len(sample_cls)
                    if n_labels > 0 and sample_bboxes.shape[-1] == 4:
                        end_idx = current_idx + n_labels
                        cls_array[current_idx:end_idx] = sample_cls
                        bboxes_array[current_idx:end_idx] = sample_bboxes.reshape(-1, 4)
                        batch_idx_array[current_idx:end_idx] = batch_i
                        current_idx = end_idx
                
                # Trim to actual size if needed
                if current_idx < total_labels:
                    cls_array = cls_array[:current_idx]
                    bboxes_array = bboxes_array[:current_idx]
                    batch_idx_array = batch_idx_array[:current_idx]
                
                # Single tensor conversion (efficient!)
                return {
                    'img': images,
                    'batch_idx': torch.from_numpy(batch_idx_array),
                    'cls': torch.from_numpy(cls_array),
                    'bboxes': torch.from_numpy(bboxes_array),
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch]
                }

            # Return optimized DataLoader
            return torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(mode == 'train'), 
                collate_fn=collate_fn,
                num_workers=min(8, batch_size),  # Optimize workers
                pin_memory=True,  # Faster GPU transfer
                persistent_workers=True if batch_size > 1 else False
            )

    # 2. LOAD THE MODEL
    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    # 3. START OPTIMIZED TRAINING
    print("\n--- Starting Optimized Training ---")
    model.train(
        trainer=ZarrTrainer,
        data=args.config_file,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        plots=False,
        val=True,
        # Performance optimizations
        amp=True,          # Mixed precision for speed
        cache=False,       # Don't cache (we're using Zarr)
        workers=8,         # Parallel data loading
        patience=100       # Early stopping patience
    )

    print("--- Optimized Training Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLO model on Zarr dataset (Performance Optimized)")
    
    # Dataset arguments
    parser.add_argument('--zarr-path', type=str, required=True, help="Path to the input Zarr archive.")
    parser.add_argument('--config-file', type=str, default='zarr_bbox_config.yaml', help="Path to the YOLOv8 data config YAML file.")
    parser.add_argument('--split-ratio', type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument('--random-seed', type=int, default=42, help="Seed for the random train/val split.")
    
    # Training hyperparameters
    parser.add_argument('--model-name', type=str, default='yolov8n.pt', help="Name of the base YOLO model.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--img-size', type=int, default=640, help="Image size for training.")
    parser.add_argument('--device', type=str, default='0', help="Device to run on, e.g., '0' for GPU 0 or 'cpu'.")

    args = parser.parse_args()
    
    print(f"🚀 Starting optimized YOLO training with:")
    print(f"   Zarr path: {args.zarr_path}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Image size: {args.img_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Split ratio: {args.split_ratio}")
    
    main(args)