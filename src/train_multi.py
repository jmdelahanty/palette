#!/usr/bin/env python3
"""
Enhanced Multi-Zarr YOLO Trainer with Automated Metadata Logging
This script trains a YOLO model on multiple Zarr datasets and saves a complete,
reproducible configuration file with the training results.
"""

import argparse
import torch
import yaml
from pathlib import Path
import time
import platform
from ultralytics import YOLO, __version__ as ultralytics_version

# Import our custom classes and the git info utility
from enhanced_multi_zarr_dataset import create_multi_zarr_dataset, MultiDatasetConfig
from working_yolo_trainer import FixedTrainer, robust_collate_fn
from tracker import get_git_info # For reproducibility

def main(args):
    """Main training function with post-training metadata logging."""
    print("🚀 Starting Enhanced Multi-Zarr YOLO Training...")

    # --- 1. Load Configuration ---
    try:
        with open(args.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        config = MultiDatasetConfig(**full_config['data_config'])
        print(f"✅ Loaded data configuration from: {args.config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # --- 2. Set up Custom Dataloader ---
    def get_multi_dataloader(self, dataset_path, batch_size=16, **kwargs):
        mode = kwargs.get('mode', 'train')
        dataset = create_multi_zarr_dataset(config=config, mode=mode)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=(mode == 'train'),
            collate_fn=robust_collate_fn, num_workers=8,
            pin_memory=True, persistent_workers=True
        )

    FixedTrainer.get_dataloader = get_multi_dataloader
    
    # --- 3. Initialize and Train the Model ---
    training_params = full_config.get('training_config', {})
    model = YOLO(training_params.get('model_name', 'yolov8n.pt'))
    
    training_start_time = time.time()
    
    # The 'train' method returns a results object with all the metrics
    results = model.train(
        trainer=FixedTrainer,
        data=args.config_path,
        epochs=training_params.get('epochs', 50),
        batch=training_params.get('batch_size', 16),
        device=training_params.get('device', '0'),
        project="runs/detect",
        name=args.run_name or "multi_zarr_train"
    )
    
    training_duration_seconds = time.time() - training_start_time

    # --- 4. Log Metadata and Save Final Config ---
    print("\n📝 Logging training history and metadata...")
    try:
        # Get git and environment info
        git_info = get_git_info()

        # Populate the training history section
        full_config['training_history'] = {
            'training_run_name': results.save_dir.name,
            'output_directory': str(results.save_dir),
            'final_model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'training_start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time)),
            'training_duration_hours': round(training_duration_seconds / 3600, 2),
            'git_commit_hash': git_info.get('commit_hash', 'N/A'),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'ultralytics_version': ultralytics_version,
            'final_metrics': {
                'box_loss': results.box_loss,
                'cls_loss': results.cls_loss,
                'dfl_loss': results.dfl_loss,
                'precision': results.metrics.precision,
                'recall': results.metrics.recall,
                'mAP50': results.metrics.map50,
                'mAP50_95': results.metrics.map
            }
        }
        
        # Save the updated, complete config file to the run directory
        final_config_path = results.save_dir / "final_comprehensive_config.yaml"
        with open(final_config_path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Successfully saved final config to: {final_config_path}")

    except Exception as e:
        print(f"❌ Could not save final training report: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Zarr YOLO Trainer")
    parser.add_argument("config_path", type=str, help="Path to the comprehensive multi-zarr config YAML")
    parser.add_argument("--run-name", type=str, help="Optional name for the training run directory.")
    args = parser.parse_args()
    main(args)