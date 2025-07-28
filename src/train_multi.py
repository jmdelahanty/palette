# train_multi.py (Final Version)

import argparse
import torch
from ultralytics import YOLO
from enhanced_multi_zarr_dataset import create_multi_zarr_dataset, MultiDatasetConfig
from working_yolo_trainer import FixedTrainer, robust_collate_fn

def main(args):
    """Main training function."""
    print("🚀 Starting Enhanced Multi-Zarr YOLO Training...")

    try:
        config = MultiDatasetConfig.from_yaml(args.config_path)
        print(f"✅ Loaded configuration from: {args.config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    def get_multi_dataloader(self, dataset_path, batch_size=16, **kwargs):
        """Replaces the default dataloader to use our multi-zarr dataset."""
        mode = kwargs.get('mode', 'train')
        dataset = create_multi_zarr_dataset(config=config, mode=mode)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            collate_fn=robust_collate_fn, # Use the robust collate function
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

    # Override the get_dataloader method in our FixedTrainer class
    FixedTrainer.get_dataloader = get_multi_dataloader
    
    model = YOLO(args.model_name)

    # Start training
    model.train(
        trainer=FixedTrainer,
        data=args.config_path, 
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        project="runs/detect",
        name="multi_zarr_train"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Zarr YOLO Trainer")
    parser.add_argument("config_path", type=str, help="Path to multi-zarr config YAML")
    parser.add_argument("--model-name", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    main(args)