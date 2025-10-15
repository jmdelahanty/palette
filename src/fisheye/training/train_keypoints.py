# src/fisheye/training/train_keypoints.py

"""
Keypoint/Pose YOLO Trainer from Zarr files with Enhanced Metadata Logging

Features:
- Trains pose estimation models on ROI images with keypoint annotations
- Requires tracking_runs with keypoint data (bladder, eye_l, eye_r)
- Tracks crop source (detect/filtered/interpolated)
- Complete provenance tracking
- Enhanced training reports with tracking success rates

Usage:
    python -m fisheye.training.train_keypoints path/to/pose_config.yaml --run-name my_pose_model
"""

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
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json
import zarr

from .config import PoseConfig
from .zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
from ..utils.system import get_git_info


# Custom DataLoader to ensure compatibility with Ultralytics YOLO's expected interface
class YoloCompatibleDataLoader(DataLoader):
    """Minimal wrapper to add reset() method required by YOLO."""
    def reset(self):
        pass


def pose_collate_fn(batch):
    """
    Custom collate function to handle batches of pose data, including keypoints.
    This safely handles cases where some samples may not have labels and reshapes keypoints.
    
    Args:
        batch: List of sample dicts with keys: 'img', 'cls', 'bboxes', 'keypoints', etc.
    
    Returns:
        Dict with batched tensors for training
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
            
            # Handle keypoints - reshape to (num_instances, num_keypoints, 3)
            # Expected format: [x, y, visibility] for each keypoint
            if 'keypoints' in sample and sample['keypoints'].size > 0:
                kpts = torch.from_numpy(sample['keypoints'])
                kpts = kpts.view(num_instances, -1, 3)  # Reshape to (N, K, 3)
                keypoints_list.append(kpts)
            else:
                # Create empty keypoints if none provided (shouldn't happen with pose task)
                num_kpts = 3  # Default: bladder, eye_l, eye_r
                keypoints_list.append(torch.zeros((num_instances, num_kpts, 3)))

            batch_idx_list.append(torch.full((num_instances,), i))

    # Handle empty batch (all samples had no valid labels)
    if not batch_idx_list:
        return {
            'img': images, 
            'batch_idx': torch.empty(0, dtype=torch.long), 
            'cls': torch.empty(0, dtype=torch.float32), 
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'keypoints': torch.empty(0, 3, 3, dtype=torch.float32),  # (0, 3 keypoints, 3 coords)
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
    """Custom validator that handles edge cases in pose data."""
    
    def _prepare_batch(self, si, batch):
        """Prepare batch for validation, handling shape issues."""
        pbatch = super()._prepare_batch(si, batch)
        
        # Handle cls shape issues (same as detection validator)
        if 'cls' in pbatch and hasattr(pbatch['cls'], 'ndim') and pbatch['cls'].ndim == 0:
            pbatch['cls'] = pbatch['cls'].unsqueeze(0)
        
        return pbatch


class ZarrPoseTrainer(PoseTrainer):
    """Custom pose trainer that uses Zarr dataset loader."""
    
    def get_validator(self):
        """Return custom validator with proper loss names."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return ZarrPoseValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=self.args, 
            _callbacks=self.callbacks
        )

    def get_dataloader(self, dataset_path, batch_size=16, mode='train', rank=0):
        """
        Create DataLoader for Zarr dataset.
        
        This method is called by YOLO during training setup.
        The config is accessed through the global variable set in main().
        """
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


def get_zarr_metadata(zarr_paths, console=None):
    """
    Extract comprehensive metadata from zarr files including crop and tracking info.
    
    Args:
        zarr_paths: List of paths to zarr files
        console: Optional Rich console for output
        
    Returns:
        Dictionary of metadata per zarr file
    """
    metadata = {}
    
    for path in zarr_paths:
        try:
            root = zarr.open(path, mode='r')
            path_name = Path(path).name
            
            zarr_meta = {
                'path': str(path),
                'video_frames': 0,
                'crop_info': {},
                'tracking_info': {},
                'data_quality': {}
            }
            
            # Get video info
            if 'raw_video' in root:
                if 'images_full' in root['raw_video']:
                    zarr_meta['video_frames'] = root['raw_video/images_full'].shape[0]
                zarr_meta['fps'] = root['raw_video'].attrs.get('fps', 'N/A')
            
            # Get crop info
            if 'crop_runs' in root and 'latest' in root['crop_runs'].attrs:
                latest_crop = root['crop_runs'].attrs['latest']
                crop_group = root[f'crop_runs/{latest_crop}']
                
                zarr_meta['crop_info'] = {
                    'run_name': latest_crop,
                    'source_type': crop_group.attrs.get('detection_source_type', 'unknown'),
                    'n_rois': crop_group['roi_images'].shape[0] if 'roi_images' in crop_group else 0,
                    'roi_size': tuple(crop_group['roi_images'].shape[1:3]) if 'roi_images' in crop_group else (0, 0),
                    'includes_interpolated': crop_group.attrs.get('includes_interpolated', False),
                    'n_real_detections': crop_group.attrs.get('n_real_detections', 0),
                    'n_interpolated_detections': crop_group.attrs.get('n_interpolated_detections', 0)
                }
            
            # Get tracking info (CRITICAL for pose task)
            if 'tracking_runs' in root and 'latest' in root['tracking_runs'].attrs:
                latest_track = root['tracking_runs'].attrs['latest']
                track_group = root[f'tracking_runs/{latest_track}']
                
                if 'tracking_results' in track_group:
                    tracking_data = track_group['tracking_results']
                    n_frames = len(tracking_data)
                    
                    # Calculate tracking success rate (non-NaN keypoints)
                    column_names = tracking_data.attrs.get('column_names', [])
                    if 'bladder_x_roi_norm' in column_names:
                        bladder_x_idx = column_names.index('bladder_x_roi_norm')
                        n_valid = np.sum(~np.isnan(tracking_data[:, bladder_x_idx]))
                        success_rate = (n_valid / n_frames * 100) if n_frames > 0 else 0
                    else:
                        success_rate = 0
                    
                    zarr_meta['tracking_info'] = {
                        'run_name': latest_track,
                        'n_frames': n_frames,
                        'n_valid_keypoints': int(n_valid) if 'n_valid' in locals() else 0,
                        'tracking_success_rate': round(success_rate, 2),
                        'keypoint_columns': column_names[:6] if len(column_names) >= 6 else column_names
                    }
                else:
                    zarr_meta['tracking_info'] = {'error': 'tracking_results not found'}
            else:
                zarr_meta['tracking_info'] = {'error': 'tracking_runs not found or no latest run'}
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
    
    return metadata


def main(args):
    """Main training function."""
    console = Console()
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]      Zarr YOLO Pose Training - Fisheye Module        [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Load and validate config
    global config
    try:
        full_config = PoseConfig.from_yaml(args.config_path)
        config = ZarrDatasetConfig(**full_config.model_dump())
        console.print(f"[bold green]✓ Loaded configuration:[/bold green] {args.config_path}\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error loading config:[/bold red] {e}")
        traceback.print_exc()
        return
    
    # Get zarr metadata
    zarr_paths = config.get_zarr_paths()
    zarr_metadata = get_zarr_metadata(zarr_paths, console)
    
    # Display dataset info
    console.print("[bold cyan]Dataset Information[/bold cyan]")
    console.rule()
    
    for name, meta in zarr_metadata.items():
        if 'error' in meta:
            console.print(f"  [red]✗ {name}[/red]: {meta['error']}")
            continue
        
        console.print(f"\n  [green]{name}[/green]")
        console.print(f"    • Video: {meta.get('video_frames', 0)} frames @ {meta.get('fps', 'N/A')} fps")
        
        crop_info = meta.get('crop_info', {})
        console.print(f"    • Crops: {crop_info.get('n_rois', 0)} ROIs ({crop_info.get('roi_size', 'N/A')})")
        console.print(f"      - Source: {crop_info.get('source_type', 'unknown')}")
        if crop_info.get('includes_interpolated'):
            console.print(f"      - Real: {crop_info.get('n_real_detections', 0)} | "
                         f"Interpolated: {crop_info.get('n_interpolated_detections', 0)}")
        
        track_info = meta.get('tracking_info', {})
        if 'error' not in track_info:
            console.print(f"    • Tracking: {track_info.get('n_valid_keypoints', 0)}/{track_info.get('n_frames', 0)} valid "
                         f"({track_info.get('tracking_success_rate', 0):.1f}%)")
        else:
            console.print(f"    [red]• Tracking: {track_info['error']}[/red]")
    
    console.print()
    
    # Verify all datasets have tracking data
    missing_tracking = [name for name, meta in zarr_metadata.items() 
                       if 'error' in meta.get('tracking_info', {})]
    if missing_tracking:
        console.print(f"[bold red]✗ ERROR: The following datasets are missing tracking data:[/bold red]")
        for name in missing_tracking:
            console.print(f"  - {name}")
        console.print("\n[yellow]Run keypoint detection first:[/yellow]")
        console.print("  python -m fisheye.detection.detect_keypoints_traditional <zarr_path>")
        return
    
    # Get training params
    training_params = full_config.training_params.model_dump()
    model_name = training_params.get('model', 'yolov8n-pose.pt')
    
    # Display hyperparameters
    console.print("[bold yellow]Training Hyperparameters[/bold yellow]")
    console.rule()
    params_table = Table(show_header=False, box=None, padding=(0, 2))
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="yellow")
    
    for key, value in training_params.items():
        params_table.add_row(key, str(value))
    
    console.print(params_table)
    console.print()
    
    # Initialize model
    console.print(f"[bold]Loading model:[/bold] {model_name}")
    model = YOLO(model_name)
    
    # Monkey-patch the trainer's get_dataloader method
    def get_zarr_dataloader(trainer_self, dataset_path, batch_size, mode, rank=0):
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
    
    ZarrPoseTrainer.get_dataloader = get_zarr_dataloader
    
    # Start training
    console.print("\n[bold green]⚡ Starting Training...[/bold green]\n")
    training_start_time = time.time()
    
    try:
        results = model.train(
            trainer=ZarrPoseTrainer,
            data=args.config_path,
            name=args.run_name or "zarr_pose_train",
            **training_params
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Training failed:[/bold red] {e}")
        traceback.print_exc()
        return
    
    training_duration_seconds = time.time() - training_start_time
    
    # Log training metadata
    console.print("\n[bold cyan]Generating Training Report...[/bold cyan]")
    try:
        git_info = get_git_info()
        results_df = pd.read_csv(results.save_dir / 'results.csv')
        results_df.columns = results_df.columns.str.strip()
        last_epoch_metrics = results_df.iloc[-1]
        
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
        final_config_filename = f"{timestamp}_pose_training_report.yaml"
        final_config_path = results.save_dir / final_config_filename
        
        # Build comprehensive training report
        final_report = full_config.model_dump()
        final_report['training_history'] = {
            'source_zarr_metadata': zarr_metadata,
            'training_run_name': results.save_dir.name,
            'output_directory': str(results.save_dir),
            'final_model_path': str(results.save_dir / 'weights' / 'best.pt'),
            'training_start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time)),
            'training_duration_hours': round(training_duration_seconds / 3600, 2),
            'git_commit_hash': git_info.get('commit_hash', 'N/A'),
            'git_branch': git_info.get('branch', 'N/A'),
            'python_version': platform.python_version(),
            'torch_version': str(torch.__version__),
            'ultralytics_version': str(ultralytics_version),
            'cuda_available': torch.cuda.is_available(),
            'final_training_losses': {
                'box_loss': float(last_epoch_metrics.get('train/box_loss', 0)),
                'pose_loss': float(last_epoch_metrics.get('train/pose_loss', 0)),
                'kobj_loss': float(last_epoch_metrics.get('train/kobj_loss', 0)),
                'cls_loss': float(last_epoch_metrics.get('train/cls_loss', 0)),
                'dfl_loss': float(last_epoch_metrics.get('train/dfl_loss', 0)),
            },
            'final_validation_metrics': {
                'precision': float(last_epoch_metrics.get('metrics/precision(B)', 0)),
                'recall': float(last_epoch_metrics.get('metrics/recall(B)', 0)),
                'mAP50': float(last_epoch_metrics.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(B)', 0)),
                'pose_mAP50': float(last_epoch_metrics.get('metrics/mAP50(P)', 0)),
                'pose_mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(P)', 0))
            }
        }
        
        # Save report
        with open(final_config_path, 'w') as f:
            yaml.dump(final_report, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[bold green]✓ Training report saved:[/bold green] {final_config_path}\n")
        
        # Display final metrics
        metrics_table = Table(title="Final Training Metrics", title_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="yellow", justify="right")
        
        # Detection metrics
        metrics_table.add_row("Box Precision", f"{final_report['training_history']['final_validation_metrics']['precision']:.3f}")
        metrics_table.add_row("Box Recall", f"{final_report['training_history']['final_validation_metrics']['recall']:.3f}")
        metrics_table.add_row("Box mAP50", f"{final_report['training_history']['final_validation_metrics']['mAP50']:.3f}")
        metrics_table.add_row("Box mAP50-95", f"{final_report['training_history']['final_validation_metrics']['mAP50_95']:.3f}")
        
        # Pose metrics
        metrics_table.add_row("─" * 20, "─" * 10)
        metrics_table.add_row("Pose mAP50", f"{final_report['training_history']['final_validation_metrics']['pose_mAP50']:.3f}")
        metrics_table.add_row("Pose mAP50-95", f"{final_report['training_history']['final_validation_metrics']['pose_mAP50_95']:.3f}")
        
        # Training info
        metrics_table.add_row("─" * 20, "─" * 10)
        metrics_table.add_row("Training Time", f"{final_report['training_history']['training_duration_hours']:.2f}h")
        metrics_table.add_row("Model Path", Path(final_report['training_history']['final_model_path']).name)
        
        console.print(metrics_table)
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]✗ Could not save training report:[/bold red] {e}")
        traceback.print_exc()
    
    console.print("[bold green]✓ Training Complete![/bold green]")
    console.print(f"[dim]Results saved to: {results.save_dir}[/dim]\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-Zarr YOLO Pose Trainer with Enhanced Metadata Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python -m fisheye.training.train_keypoints configs/pose_config.yaml
  
  # With custom run name
  python -m fisheye.training.train_keypoints configs/pose_config.yaml --run-name fish_pose_v1
  
  # After training, use the model
  python -m fisheye.inference.predict_pose runs/pose/fish_pose_v1/weights/best.pt video.zarr
        """
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the pose training configuration YAML"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional name for the training run directory"
    )
    args = parser.parse_args()
    main(args)