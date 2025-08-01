# src/train_pose.py

import argparse
import torch
import yaml
from pathlib import Path
import time
import platform
import traceback
import pandas as pd
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.utils import DEFAULT_CFG
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.panel import Panel
import json
import zarr

from config_models import PoseConfig
from zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
from tracker import get_git_info

class YoloCompatibleDataLoader(DataLoader):
    def reset(self): pass

def pose_collate_fn(batch):
    """
    Custom collate function to handle batches of pose data, including keypoints.
    This safely handles cases where some samples in a batch may not have labels and reshapes keypoints.
    """
    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
    im_files = [s.get('im_file', f'image_{i}.jpg') for i, s in enumerate(batch)]
    ori_shapes = [s.get('ori_shape', (images.shape[2], images.shape[3])) for s in batch]
    ratio_pads = [s.get('ratio_pad', (None, None)) for s in batch]
    
    cls_list, bboxes_list, keypoints_list, batch_idx_list = [], [], [], []
    
    for i, sample in enumerate(batch):
        cls_labels = np.atleast_1d(sample['cls'])
        if cls_labels.size > 0 and cls_labels[0] is not None:
            num_instances = len(cls_labels)
            cls_list.append(torch.from_numpy(cls_labels))
            bboxes_list.append(torch.from_numpy(sample['bboxes']))
            
            if 'keypoints' in sample and sample['keypoints'].size > 0:
                kpts = torch.from_numpy(sample['keypoints'])
                kpts = kpts.view(num_instances, -1, 3) 
                keypoints_list.append(kpts)
            else:
                num_kpts = 3 
                keypoints_list.append(torch.zeros((num_instances, num_kpts, 3)))

            batch_idx_list.append(torch.full((num_instances,), i))

    if not batch_idx_list:
        return {
            'img': images, 
            'batch_idx': torch.empty(0, dtype=torch.long), 
            'cls': torch.empty(0, dtype=torch.float32), 
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'keypoints': torch.empty(0, 3, 3, dtype=torch.float32), 
            'im_file': im_files, 
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }
        
    return {
        'img': images, 
        'batch_idx': torch.cat(batch_idx_list, 0), 
        'cls': torch.cat(cls_list, 0).float(),
        'bboxes': torch.cat(bboxes_list, 0).float(),
        'keypoints': torch.cat(keypoints_list, 0).float(),
        'im_file': im_files, 
        'ori_shape': ori_shapes,
        'ratio_pad': ratio_pads
    }

class ZarrPoseValidator(PoseValidator):
    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        if 'cls' in pbatch and hasattr(pbatch['cls'], 'ndim') and pbatch['cls'].ndim == 0:
            pbatch['cls'] = pbatch['cls'].unsqueeze(0)
        return pbatch

class ZarrPoseTrainer(PoseTrainer):
    def get_validator(self):
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return ZarrPoseValidator(self.test_loader, save_dir=self.save_dir, args=self.args, _callbacks=self.callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, mode='train', rank=0):
        try:
            global config 
            dataset = create_zarr_dataset(config=config, mode=mode)
            return YoloCompatibleDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(mode == 'train'),
                collate_fn=pose_collate_fn,
                num_workers=8,
                pin_memory=True,
                persistent_workers=False
            )
        except Exception as e:
            print(f"Error creating dataloader: {e}")
            raise

def get_zarr_metadata(zarr_paths):
    """Inspects Zarr files to get cropping and tracking success rates."""
    metadata = {}
    for path in zarr_paths:
        try:
            root = zarr.open(path, mode='r')
            path_name = Path(path).name
            crop_stats = root['crop_runs'].attrs.get('best', {})
            track_stats = root['tracking_runs'].attrs.get('best', {})
            metadata[path_name] = {
                'cropping_success_rate': crop_stats.get('percent_cropped', 'N/A'),
                'tracking_success_rate': track_stats.get('percent_tracked', 'N/A'),
                'best_crop_run': crop_stats.get('run_name', 'N/A'),
                'best_track_run': track_stats.get('run_name', 'N/A')
            }
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
    return metadata

def main(args):
    console = Console()
    console.print("[bold cyan]Starting Zarr YOLO Pose Training...[/bold cyan]")
    
    global config
    try:
        # Use the Pydantic model to load and validate the config
        full_config = PoseConfig.from_yaml(args.config_path)
        config = ZarrDatasetConfig(**full_config.data_config.model_dump())
        console.print(f"[bold green]Loaded and validated data configuration from:[/bold green] {args.config_path}")
    except Exception as e:
        console.print(f"[bold red]Error loading or validating config:[/bold red] {e}"); return
    
    zarr_metadata = get_zarr_metadata(config.zarr_paths)
    console.print("\n[bold cyan]Zarr Dataset Metadata[/bold cyan]")
    for name, meta in zarr_metadata.items():
        console.print(f"  [green]{name}[/green]:")
        console.print(f"    - Cropping Success: {meta.get('cropping_success_rate', 'N/A')}%")
        console.print(f"    - Tracking Success: {meta.get('tracking_success_rate', 'N/A')}%")

    # Access attributes using dot notation and convert to dict for YOLO
    training_params = full_config.training_params.model_dump()
    model_name = training_params.get('model', 'yolov8n-pose.pt')
    model = YOLO(model_name)

    params_str = json.dumps(training_params, indent=2)
    console.print(Panel(params_str, title="[bold yellow]Training Hyperparameters[/bold yellow]", expand=False))
    
    training_start_time = time.time()
    results = model.train(
        trainer=ZarrPoseTrainer,
        data=args.config_path,
        name=args.run_name or "multi_zarr_pose_train",
        **training_params
    )
    training_duration_seconds = time.time() - training_start_time

    console.print("\n[bold cyan]Logging training history and metadata...[/bold cyan]")
    try:
        git_info = get_git_info()
        results_df = pd.read_csv(results.save_dir / 'results.csv')
        results_df.columns = results_df.columns.str.strip()
        last_epoch_metrics = results_df.iloc[-1]

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
        final_config_filename = f"{timestamp}_pose_training_report.yaml"
        final_config_path = results.save_dir / final_config_filename
        
        # Create a dictionary for the final report
        final_report = full_config.model_dump()
        final_report['training_history'] = {
            'source_zarr_metadata': zarr_metadata,
            'training_run_name': results.save_dir.name,
            'output_directory': str(results.save_dir),
            'final_model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'training_start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time)),
            'training_duration_hours': round(training_duration_seconds / 3600, 2),
            'git_commit_hash': git_info.get('commit_hash', 'N/A'),
            'python_version': platform.python_version(),
            'torch_version': str(torch.__version__),
            'ultralytics_version': str(ultralytics_version),
            'final_training_losses': {
                'box_loss': float(last_epoch_metrics.get('train/box_loss')),
                'pose_loss': float(last_epoch_metrics.get('train/pose_loss')),
                'dfl_loss': float(last_epoch_metrics.get('train/dfl_loss')),
            },
            'final_validation_metrics': {
                'precision': float(last_epoch_metrics.get('metrics/precision(B)')),
                'recall': float(last_epoch_metrics.get('metrics/recall(B)')),
                'mAP50': float(last_epoch_metrics.get('metrics/mAP50(B)')),
                'mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(B)'))
            }
        }
        
        with open(final_config_path, 'w') as f:
            yaml.dump(final_report, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[bold green]Successfully saved final config to:[/bold green] {final_config_path}")

    except Exception as e:
        console.print(f"[bold red]Could not save final training report:[/bold red] {e}")
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Zarr YOLO Pose Trainer")
    parser.add_argument("config_path", type=str, help="Path to the pose training config YAML (e.g., src/pose_config.yaml)")
    parser.add_argument("--run-name", type=str, help="Optional name for the training run directory.")
    args = parser.parse_args()
    main(args)