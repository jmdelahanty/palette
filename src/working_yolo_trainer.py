# working_yolo_trainer.py

import argparse
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from zarr_yolo_dataset_bbox import ZarrYOLODataset

# --- Reusable Components ---

class YoloCompatibleDataLoader(DataLoader):
    """
    A DataLoader wrapper that is compatible with the Ultralytics trainer.
    It adds a dummy 'reset()' method to satisfy the trainer's API when
    mosaic augmentation is turned off in the final epochs.
    """
    def reset(self):
        """A dummy reset method to do nothing, preventing a crash."""
        pass

def robust_collate_fn(batch):
    """
    Fully robust collate function that handles batches with no labels by ensuring
    all expected keys are always present.
    """
    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
    im_files = [s['im_file'] for s in batch]
    ori_shapes = [s['ori_shape'] for s in batch]
    ratio_pads = [s['ratio_pad'] for s in batch]

    cls_list, bboxes_list, batch_idx_list = [], [], []
    for i, sample in enumerate(batch):
        cls_labels = np.asarray(sample['cls'])
        if cls_labels.size > 0:
            cls_list.append(torch.from_numpy(cls_labels))
            bboxes_list.append(torch.from_numpy(sample['bboxes']))
            batch_idx_list.append(torch.full((len(cls_labels),), i))

    if not batch_idx_list:
        return {'img': images, 'batch_idx': torch.empty(0, dtype=torch.long), 'cls': torch.empty(0, dtype=torch.float32), 'bboxes': torch.empty(0, 4, dtype=torch.float32), 'im_file': im_files, 'ori_shape': ori_shapes, 'ratio_pad': ratio_pads}

    return {'img': images, 'batch_idx': torch.cat(batch_idx_list, 0), 'cls': torch.cat(cls_list, 0), 'bboxes': torch.cat(bboxes_list, 0), 'im_file': im_files, 'ori_shape': ori_shapes, 'ratio_pad': ratio_pads}

class FixedDetectionValidator(DetectionValidator):
    """Custom validator to fix a 0-d tensor bug in the YOLOv8 validator."""
    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        if 'cls' in pbatch and hasattr(pbatch['cls'], 'ndim') and pbatch['cls'].ndim == 0:
            pbatch['cls'] = pbatch['cls'].unsqueeze(0)
        return pbatch

class FixedTrainer(DetectionTrainer):
    """Custom trainer that uses our fixed validator and the compatible dataloader."""
    def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
        """Dataloader for the single-zarr trainer."""
        dataset = ZarrYOLODataset(
            zarr_path=self.args.zarr_path, mode=mode, task='detect',
            split_ratio=self.args.split_ratio, random_seed=self.args.random_seed
        )
        return YoloCompatibleDataLoader(
            dataset, batch_size=batch_size, shuffle=(mode == 'train'),
            collate_fn=robust_collate_fn, num_workers=8, pin_memory=True
        )


# --- Main Execution Block ---

def main(args):
    """Main function to run the single-dataset training pipeline."""
    print("\n--- Loading Model ---")
    model = YOLO(args.model_name)

    print("\n--- Starting Training with Fixed Validator ---")
    
    # The 'trainer=FixedTrainer' argument tells YOLO to use our custom classes.
    # We pass zarr_path and other custom args here so they become part of `self.args`
    # inside the FixedTrainer instance.
    model.train(
        trainer=FixedTrainer,
        data=args.config_file,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        zarr_path=args.zarr_path,
        split_ratio=args.split_ratio,
        random_seed=args.random_seed
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO training with fixed validator for a single Zarr dataset")
    
    parser.add_argument('--zarr-path', type=str, required=True, help="Path to the single Zarr file for training.")
    parser.add_argument('--config-file', type=str, default='src/zarr_bbox_config.yaml', help="Path to the YOLO dataset config YAML.")
    parser.add_argument('--split-ratio', type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument('--random-seed', type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument('--model-name', type=str, default='yolov8n.pt', help="YOLO model to use (e.g., yolov8n.pt).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--img-size', type=int, default=640, help="Image size for training.")
    parser.add_argument('--device', type=str, default='0', help="GPU device to use (e.g., '0' or 'cpu').")

    args = parser.parse_args()
    main(args)