# src/fisheye/training/train_detection.py

"""
Detection YOLO Trainer from zarrs with Enhanced Metadata Logging

Features:
- Tracks crop source (detect/filtered/interpolated)
- Option to filter out interpolated data
- Complete provenance tracking
- Enhanced training reports
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
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from torch.utils.data import DataLoader
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json
import zarr

from config_models import DetectConfig
from training.zarr_yolo_dataset_loader import create_zarr_dataset, ZarrDatasetConfig
from tracker import get_git_info


# Custom DataLoader to ensure compatibility with Ultralytics YOLO's expected interface
class YoloCompatibleDataLoader(DataLoader):
    def reset(self):
        pass


def det_collate_fn(batch):
    """Collate function for detection data."""
    images = torch.from_numpy(np.stack([s['img'] for s in batch]))
    im_files = [s['im_file'] for s in batch]
    ori_shapes = [s['ori_shape'] for s in batch]
    ratio_pads = [s['ratio_pad'] for s in batch]
    cls_list, bboxes_list, batch_idx_list = [], [], []
    
    for i, sample in enumerate(batch):
        cls_labels = np.atleast_1d(sample['cls'])
        if cls_labels.size > 0 and cls_labels[0] is not None:
            cls_list.append(torch.from_numpy(cls_labels))
            bboxes_list.append(torch.from_numpy(sample['bboxes']))
            batch_idx_list.append(torch.full((len(cls_labels),), i))
    
    if not batch_idx_list:
        return {
            'img': images,
            'batch_idx': torch.empty(0, dtype=torch.long),
            'cls': torch.empty(0, dtype=torch.float32),
            'bboxes': torch.empty(0, 4, dtype=torch.float32),
            'im_file': im_files,
            'ori_shape': ori_shapes,
            'ratio_pad': ratio_pads
        }
    
    return {
        'img': images,
        'batch_idx': torch.cat(batch_idx_list, 0),
        'cls': torch.cat(cls_list, 0),
        'bboxes': torch.cat(bboxes_list, 0),
        'im_file': im_files,
        'ori_shape': ori_shapes,
        'ratio_pad': ratio_pads
    }


class DetValidator(DetectionValidator):
    def _prepare_batch(self, si, batch):
        pbatch = super()._prepare_batch(si, batch)
        if 'cls' in pbatch and hasattr(pbatch['cls'], 'ndim') and pbatch['cls'].ndim == 0:
            pbatch['cls'] = pbatch['cls'].unsqueeze(0)
        return pbatch


class DetTrainer(DetectionTrainer):
    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks
        )


def get_zarr_metadata(zarr_paths, console=None):
    """
    Extract comprehensive metadata from zarr files including crop source info.
    
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
                'detection_info': {},
                'data_quality': {}
            }
            
            # Get video info
            if 'raw_video' in root:
                if 'images_full' in root['raw_video']:
                    zarr_meta['video_frames'] = root['raw_video/images_full'].shape[0]
                zarr_meta['fps'] = root['raw_video'].attrs.get('fps', 'N/A')
            
            # Get detection info
            if 'detect_runs' in root:
                latest_detect = root['detect_runs'].attrs.get('latest')
                if latest_detect:
                    detect_group = root[f'detect_runs/{latest_detect}']
                    if 'summary_statistics' in detect_group.attrs:
                        stats = detect_group.attrs['summary_statistics']
                        zarr_meta['detection_info'] = {
                            'run_name': latest_detect,
                            'total_detections': stats.get('total_detections', 0),
                            'frames_with_detections': stats.get('frames_with_detections', 0),
                            'detection_rate': stats.get('frames_with_detections', 0) / max(stats.get('total_frames', 1), 1) * 100
                        }
            
            # Get crop info with source tracking
            if 'crop_runs' in root:
                latest_crop = root['crop_runs'].attrs.get('latest')
                if latest_crop:
                    crop_group = root[f'crop_runs/{latest_crop}']
                    
                    # Get crop source information
                    crop_source_type = crop_group.attrs.get('detection_source_type', 'detect')
                    crop_source_path = crop_group.attrs.get('detection_source_path', 'unknown')
                    includes_interpolated = crop_group.attrs.get('includes_interpolated', False)
                    
                    zarr_meta['crop_info'] = {
                        'run_name': latest_crop,
                        'source_type': crop_source_type,
                        'source_path': crop_source_path,
                        'includes_interpolated': includes_interpolated
                    }
                    
                    # Get statistics
                    if 'summary_statistics' in crop_group.attrs:
                        stats = crop_group.attrs['summary_statistics']
                        zarr_meta['crop_info'].update({
                            'total_rois': stats.get('total_rois_cropped', 0),
                            'frames_with_crops': stats.get('frames_with_crops', 0),
                            'roi_size': stats.get('roi_size', [256, 256])
                        })
                    
                    # If interpolated, get breakdown
                    if includes_interpolated:
                        zarr_meta['crop_info']['n_real'] = crop_group.attrs.get('n_real_detections', 0)
                        zarr_meta['crop_info']['n_interpolated'] = crop_group.attrs.get('n_interpolated_detections', 0)
            
            # Get refinement info if available
            if 'refined_runs' in root:
                latest_refined = root['refined_runs'].attrs.get('latest')
                if latest_refined:
                    refined_group = root[f'refined_runs/{latest_refined}']
                    
                    zarr_meta['data_quality']['has_refinement'] = True
                    zarr_meta['data_quality']['refined_run'] = latest_refined
                    
                    # Check what stages exist
                    if 'filtered' in refined_group:
                        filtered_grp = refined_group['filtered']
                        zarr_meta['data_quality']['filtered_detections'] = filtered_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['jumps_removed'] = filtered_grp.attrs.get('dropped_detections', 0)
                    
                    if 'interpolated' in refined_group:
                        interp_grp = refined_group['interpolated']
                        zarr_meta['data_quality']['interpolated_detections'] = interp_grp.attrs.get('total_detections', 0)
                        zarr_meta['data_quality']['gaps_filled'] = interp_grp.attrs.get('gaps_filled', 0)
            
            metadata[path_name] = zarr_meta
            
        except Exception as e:
            metadata[path_name] = {'error': str(e)}
            if console:
                console.print(f"[yellow]Warning: Could not read metadata from {path_name}: {e}[/yellow]")
    
    return metadata


def display_zarr_metadata(metadata, console):
    """Display zarr metadata in a nice table."""
    
    for zarr_name, meta in metadata.items():
        if 'error' in meta:
            console.print(f"[red]✗ {zarr_name}: {meta['error']}[/red]")
            continue
        
        # Create info table
        table = Table(title=f"📦 {zarr_name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Video info
        table.add_row("Video Frames", str(meta.get('video_frames', 'N/A')))
        table.add_row("FPS", str(meta.get('fps', 'N/A')))
        
        # Detection info
        if meta.get('detection_info'):
            det_info = meta['detection_info']
            table.add_row("Total Detections", f"{det_info.get('total_detections', 0):,}")
            table.add_row("Detection Rate", f"{det_info.get('detection_rate', 0):.1f}%")
        
        # Crop info with source
        if meta.get('crop_info'):
            crop_info = meta['crop_info']
            source_type = crop_info.get('source_type', 'unknown')
            
            # Color code the source
            if source_type == 'detect':
                source_display = f"[cyan]{source_type}[/cyan] (original)"
            elif source_type == 'filtered':
                source_display = f"[yellow]{source_type}[/yellow] (jumps removed)"
            elif source_type == 'interpolated':
                source_display = f"[magenta]{source_type}[/magenta] (gaps filled)"
            else:
                source_display = source_type
            
            table.add_row("Crop Source", source_display)
            table.add_row("Total ROIs", f"{crop_info.get('total_rois', 0):,}")
            
            # If interpolated, show breakdown
            if crop_info.get('includes_interpolated'):
                n_real = crop_info.get('n_real', 0)
                n_interp = crop_info.get('n_interpolated', 0)
                table.add_row("  └─ Real ROIs", f"{n_real:,}")
                table.add_row("  └─ Interpolated ROIs", f"{n_interp:,}")
        
        # Data quality info
        if meta.get('data_quality', {}).get('has_refinement'):
            quality = meta['data_quality']
            if 'jumps_removed' in quality:
                table.add_row("Jumps Removed", str(quality['jumps_removed']))
            if 'gaps_filled' in quality:
                table.add_row("Gaps Filled", str(quality['gaps_filled']))
        
        console.print(table)
        console.print()


def main(args):
    console = Console()
    console.print("[bold cyan]🚀 Starting YOLO Detection Training[/bold cyan]\n")

    try:
        # Load and validate config
        full_config = DetectConfig.from_yaml(args.config_path)
        config = ZarrDatasetConfig(**full_config.data_config.model_dump())
        console.print(f"[bold green]✓ Loaded config:[/bold green] {args.config_path}\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error loading config:[/bold red] {e}")
        return

    # Get comprehensive zarr metadata
    console.print("[bold cyan]📊 Analyzing Zarr Files...[/bold cyan]\n")
    zarr_metadata = get_zarr_metadata(config.zarr_paths, console)
    display_zarr_metadata(zarr_metadata, console)
    
    # Check for interpolated data
    has_interpolated = any(
        meta.get('crop_info', {}).get('includes_interpolated', False) 
        for meta in zarr_metadata.values() 
        if 'error' not in meta
    )
    
    if has_interpolated:
        console.print("[yellow]⚠ Warning: Some datasets include interpolated (synthetic) data[/yellow]")
        console.print("[dim]  To filter these out, add 'filter_interpolated: true' to your config[/dim]\n")

    # Setup dataloader
    def get_zarr_dataloader(self, dataset_path, batch_size=16, **kwargs):
        mode = kwargs.get('mode', 'train')
        dataset = create_zarr_dataset(config=config, mode=mode)
        return YoloCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            collate_fn=det_collate_fn,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False
        )

    DetTrainer.get_dataloader = get_zarr_dataloader

    # Get training params
    training_params = full_config.training_params.model_dump()
    model_name = training_params.get('model', 'yolov8n.pt')
    model = YOLO(model_name)

    # Display hyperparameters
    params_str = json.dumps(training_params, indent=2)
    console.print(Panel(
        params_str,
        title="[bold yellow]Training Hyperparameters[/bold yellow]",
        expand=False
    ))
    console.print()

    # Start training
    console.print("[bold green]🏋️ Starting Training...[/bold green]\n")
    training_start_time = time.time()
    
    results = model.train(
        trainer=DetTrainer,
        data=args.config_path,
        name=args.run_name or "multi_zarr_train",
        project="runs/detect",
        **training_params
    )
    
    training_duration_seconds = time.time() - training_start_time

    # Log training metadata
    console.print("\n[bold cyan]📝 Logging Training Report...[/bold cyan]")
    try:
        git_info = get_git_info()
        results_df = pd.read_csv(results.save_dir / 'results.csv')
        results_df.columns = results_df.columns.str.strip()
        last_epoch_metrics = results_df.iloc[-1]

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(training_start_time))
        final_config_filename = f"{timestamp}_detection_training_report.yaml"
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
                'cls_loss': float(last_epoch_metrics.get('train/cls_loss', 0)),
                'dfl_loss': float(last_epoch_metrics.get('train/dfl_loss', 0)),
            },
            'final_validation_metrics': {
                'precision': float(last_epoch_metrics.get('metrics/precision(B)', 0)),
                'recall': float(last_epoch_metrics.get('metrics/recall(B)', 0)),
                'mAP50': float(last_epoch_metrics.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(last_epoch_metrics.get('metrics/mAP50-95(B)', 0))
            }
        }
        
        # Save report
        with open(final_config_path, 'w') as f:
            yaml.dump(final_report, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[bold green]✓ Training report saved:[/bold green] {final_config_path}")
        
        # Display final metrics
        metrics_table = Table(title="📈 Final Training Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics_table.add_row("Precision", f"{final_report['training_history']['final_validation_metrics']['precision']:.3f}")
        metrics_table.add_row("Recall", f"{final_report['training_history']['final_validation_metrics']['recall']:.3f}")
        metrics_table.add_row("mAP50", f"{final_report['training_history']['final_validation_metrics']['mAP50']:.3f}")
        metrics_table.add_row("mAP50-95", f"{final_report['training_history']['final_validation_metrics']['mAP50_95']:.3f}")
        metrics_table.add_row("Training Time", f"{final_report['training_history']['training_duration_hours']:.2f} hours")
        
        console.print(metrics_table)

    except Exception as e:
        console.print(f"[bold red]✗ Could not save training report:[/bold red] {e}")
        traceback.print_exc()
    
    console.print("\n[bold green]✓ Training Complete![/bold green]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-Zarr YOLO Detection Trainer with Enhanced Metadata Tracking"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the training configuration YAML"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional name for the training run directory"
    )
    args = parser.parse_args()
    main(args)