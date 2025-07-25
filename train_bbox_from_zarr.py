# train_bbox_from_zarr.py (Corrected with a closure for the custom trainer)

import argparse
import torch
from ultralytics import YOLO
import numpy as np
# Import the specific trainer we need to subclass
from ultralytics.models.yolo.detect import DetectionTrainer
# Import our custom dataset class
from zarr_yolo_dataset_bbox import ZarrYOLODataset 

def main(args):
    """Main training function."""

    # 1. DEFINE THE CUSTOM TRAINER *INSIDE* MAIN
    # This creates a "closure", giving ZarrTrainer access to the 'args' object
    # from the main function's scope.
    class ZarrTrainer(DetectionTrainer):

        def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
            """
            Overrides the default data loader to use our custom ZarrYOLODataset
            and a custom collate function.
            """
            dataset = ZarrYOLODataset(
                zarr_path=args.zarr_path,
                mode=mode,
                split_ratio=args.split_ratio,
                random_seed=args.random_seed
            )
            
            # --- THIS IS THE FINAL PIECE OF THE PUZZLE ---
            def collate_fn(batch):
                """
                A custom collate function that manually builds the batch dictionary
                to ensure all keys and shapes are exactly as expected.
                """
                # Get components from each sample in the batch
                images = [s['img'] for s in batch]
                cls_list = [s['cls'] for s in batch]
                bboxes_list = [s['bboxes'] for s in batch]
                
                # Manually build metadata lists (these are not tensors)
                im_files = [s['im_file'] for s in batch]
                ori_shapes = [s['ori_shape'] for s in batch]
                ratio_pads = [s['ratio_pad'] for s in batch]
                
                # Stack images into a single tensor
                images = torch.from_numpy(np.stack(images))
                
                # --- THIS IS THE FIX ---
                # Use CONCATENATE to create flat 2D tensors for labels
                cls = torch.from_numpy(np.concatenate(cls_list))
                bboxes = torch.from_numpy(np.concatenate(bboxes_list))
                # --- END OF FIX ---
                
                # Create the batch_idx tensor
                batch_idx = []
                for i, s in enumerate(batch):
                    batch_idx.append(torch.full((len(s['cls']),), i))
                
                # Return the complete batch dictionary
                return {
                    'img': images,
                    'batch_idx': torch.cat(batch_idx, 0),
                    'cls': cls,
                    'bboxes': bboxes,
                    'im_file': im_files,
                    'ori_shape': ori_shapes,
                    'ratio_pad': ratio_pads
                }

            # Pass our custom collate function to the DataLoader
            return torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               shuffle=(mode == 'train'), 
                                               collate_fn=collate_fn)

    # 2. LOAD THE MODEL
    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    # 3. START TRAINING
    print("\n--- Starting Training ---")
    # We only need to override the trainer. The custom arguments are handled by the closure.
    # The YAML file is still needed for class names, etc.
    model.train(
        trainer=ZarrTrainer, # Pass our custom trainer class
        data=args.config_file,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        plots=False
    )

    print("--- Training Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLO model on a Zarr dataset.")
    
    # Add our custom arguments to the parser
    parser.add_argument('--zarr-path', type=str, required=True, help="Path to the input Zarr archive.")
    parser.add_argument('--config-file', type=str, default='zarr_bbox_config.yaml', help="Path to the YOLOv8 data config YAML file.")
    parser.add_argument('--split-ratio', type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument('--random-seed', type=int, default=42, help="Seed for the random train/val split.")
    
    # Standard training hyperparameters
    parser.add_argument('--model-name', type=str, default='yolov8n.pt', help="Name of the base YOLO model.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--img-size', type=int, default=320, help="Image size for training.")
    parser.add_argument('--device', type=str, default='0', help="Device to run on, e.g., '0' for GPU 0 or 'cpu'.")

    args = parser.parse_args()
    main(args)