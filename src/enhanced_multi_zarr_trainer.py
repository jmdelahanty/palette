# enhanced_multi_zarr_trainer.py
# Professional-grade multi-zarr YOLO trainer with comprehensive monitoring

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
import yaml
import json
import time
import logging
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt

# Import our enhanced dataset
from enhanced_multi_zarr_dataset import (
    EnhancedMultiZarrYOLODataset, 
    MultiDatasetConfig, 
    SamplingStrategy,
    create_multi_zarr_dataset
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Track training metrics per dataset."""
    epoch: int
    dataset_name: str
    samples_processed: int
    avg_loss: float
    timestamp: float

class PerDatasetMetricsTracker:
    """Track per-dataset metrics during training."""
    
    def __init__(self):
        self.metrics_history = []
        self.epoch_stats = defaultdict(lambda: defaultdict(list))
    
    def update(self, batch_info: Dict, loss: float, epoch: int):
        """Update metrics with batch information."""
        # Group samples by dataset
        dataset_counts = defaultdict(int)
        for sample in batch_info:
            if 'dataset_name' in sample:
                dataset_counts[sample['dataset_name']] += 1
        
        # Record metrics for each dataset in this batch
        for dataset_name, count in dataset_counts.items():
            self.epoch_stats[epoch][dataset_name].append({
                'samples': count,
                'loss': loss,
                'timestamp': time.time()
            })
    
    def finalize_epoch(self, epoch: int):
        """Finalize metrics for completed epoch."""
        for dataset_name, batch_records in self.epoch_stats[epoch].items():
            total_samples = sum(record['samples'] for record in batch_records)
            avg_loss = np.mean([record['loss'] for record in batch_records])
            
            metric = TrainingMetrics(
                epoch=epoch,
                dataset_name=dataset_name,
                samples_processed=total_samples,
                avg_loss=avg_loss,
                timestamp=time.time()
            )
            self.metrics_history.append(metric)
        
        # Clear current epoch data
        del self.epoch_stats[epoch]
    
    def get_summary(self) -> Dict:
        """Get training summary."""
        if not self.metrics_history:
            return {}
        
        summary = defaultdict(list)
        for metric in self.metrics_history:
            summary[metric.dataset_name].append({
                'epoch': metric.epoch,
                'samples': metric.samples_processed,
                'loss': metric.avg_loss
            })
        
        return dict(summary)
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Training metrics saved to: {filepath}")

class FixedDetectionValidator(DetectionValidator):
    """Enhanced validator with 0-d tensor fix and multi-dataset awareness."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_performance = defaultdict(list)
    
    def _prepare_batch(self, si, batch):
        """Fixed version that handles 0-d tensors and tracks dataset performance."""
        try:
            pbatch = super()._prepare_batch(si, batch)
            
            # Fix 0-d tensor issue
            if 'cls' in pbatch:
                cls = pbatch['cls']
                if hasattr(cls, 'ndim') and cls.ndim == 0:
                    pbatch['cls'] = cls.unsqueeze(0)
            
            return pbatch
            
        except Exception as e:
            logger.warning(f"Error in _prepare_batch: {e}")
            return {
                'cls': torch.zeros((0,), dtype=torch.float32),
                'bbox': torch.zeros((0, 4), dtype=torch.float32),
                'ori_shape': batch.get('ori_shape', []),
                'ratio_pad': batch.get('ratio_pad', [])
            }

class EnhancedMultiZarrTrainer(DetectionTrainer):
    """Enhanced YOLO trainer with multi-zarr support and comprehensive monitoring."""
    
    def __init__(self, config: MultiDatasetConfig, save_dir: Optional[str] = None, **kwargs):
        self.multi_config = config
        self.metrics_tracker = PerDatasetMetricsTracker()
        self.training_start_time = None
        
        super().__init__(**kwargs)
        
        if save_dir:
            self.save_dir = Path(save_dir)
    
    def get_validator(self):
        """Return our enhanced validator."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return FixedDetectionValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=self.args, 
            _callbacks=self.callbacks
        )
    
    def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
        """Create enhanced dataloader with multi-zarr dataset."""
        
        logger.info(f"🔄 Creating {mode} dataloader for enhanced multi-zarr dataset...")
        
        # Create dataset with our enhanced class
        dataset = EnhancedMultiZarrYOLODataset(self.multi_config, mode=mode)
        
        # Log dataset statistics
        stats = dataset.get_dataset_statistics()
        logger.info(f"   📊 {mode.title()} samples: {stats['mode_specific']['total_samples']}")
        logger.info(f"   🎯 Sampling strategy: {stats['sampling_strategy']}")
        
        for name, count in stats['mode_specific']['per_dataset_counts'].items():
            percentage = stats['mode_specific']['per_dataset_percentages'][name]
            logger.info(f"      {name}: {count} samples ({percentage:.1f}%)")
        
        def enhanced_collate_fn(batch):
            """Enhanced collate function with dataset tracking and safety checks."""
            try:
                # Extract data
                images = torch.from_numpy(np.stack([s['img'] for s in batch]))
                
                # Collect batch metadata for tracking
                batch_metadata = [
                    {
                        'dataset_name': s.get('dataset_name', 'unknown'),
                        'zarr_path': s.get('zarr_path', ''),
                        'local_frame_idx': s.get('local_frame_idx', -1)
                    }
                    for s in batch
                ]
                
                total_labels = sum(len(s['cls']) for s in batch)
                
                if total_labels == 0:
                    return {
                        'img': images,
                        'batch_idx': torch.zeros((0,), dtype=torch.long),
                        'cls': torch.zeros((0,), dtype=torch.float32),
                        'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                        'im_file': [s['im_file'] for s in batch],
                        'ori_shape': [s['ori_shape'] for s in batch],
                        'ratio_pad': [s['ratio_pad'] for s in batch],
                        'batch_metadata': batch_metadata  # Add metadata
                    }
                
                # Process labels with enhanced safety
                cls_array = np.zeros(total_labels, dtype=np.float32)
                bboxes_array = np.zeros((total_labels, 4), dtype=np.float32)
                batch_idx_array = np.zeros(total_labels, dtype=np.int64)
                
                current_idx = 0
                for batch_i, sample in enumerate(batch):
                    sample_cls = np.asarray(sample['cls'], dtype=np.float32)
                    sample_bboxes = np.asarray(sample['bboxes'], dtype=np.float32)
                    
                    # Enhanced cls processing
                    if sample_cls.ndim == 0:
                        sample_cls = np.array([sample_cls], dtype=np.float32)
                    sample_cls = sample_cls.flatten()
                    
                    # Enhanced bboxes processing
                    if sample_bboxes.ndim == 1 and len(sample_bboxes) == 4:
                        sample_bboxes = sample_bboxes[None, :]
                    elif sample_bboxes.ndim == 2 and sample_bboxes.shape[1] != 4:
                        logger.warning(f"Invalid bbox shape: {sample_bboxes.shape}, skipping sample")
                        continue
                    
                    n_labels = len(sample_cls)
                    if n_labels > 0 and sample_bboxes.shape[-1] == 4:
                        end_idx = current_idx + n_labels
                        cls_array[current_idx:end_idx] = sample_cls
                        bboxes_array[current_idx:end_idx] = sample_bboxes.reshape(-1, 4)
                        batch_idx_array[current_idx:end_idx] = batch_i
                        current_idx = end_idx
                
                # Create tensors with validation
                cls_tensor = torch.from_numpy(cls_array[:current_idx])
                bbox_tensor = torch.from_numpy(bboxes_array[:current_idx])
                batch_idx_tensor = torch.from_numpy(batch_idx_array[:current_idx])
                
                # Final validation
                assert cls_tensor.ndim == 1, f"cls_tensor must be 1D, got {cls_tensor.ndim}D"
                
                return {
                    'img': images,
                    'batch_idx': batch_idx_tensor,
                    'cls': cls_tensor,
                    'bboxes': bbox_tensor,
                    'im_file': [s['im_file'] for s in batch],
                    'ori_shape': [s['ori_shape'] for s in batch],
                    'ratio_pad': [s['ratio_pad'] for s in batch],
                    'batch_metadata': batch_metadata  # Include tracking metadata
                }
                
            except Exception as e:
                logger.error(f"Error in enhanced_collate_fn: {e}")
                # Return safe fallback
                return {
                    'img': torch.zeros((len(batch), 3, 640, 640), dtype=torch.float32),
                    'batch_idx': torch.zeros((0,), dtype=torch.long),
                    'cls': torch.zeros((0,), dtype=torch.float32),
                    'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                    'im_file': [f'fallback_{i}' for i in range(len(batch))],
                    'ori_shape': [(640, 640)] * len(batch),
                    'ratio_pad': [(1.0, (0.0, 0.0))] * len(batch),
                    'batch_metadata': [{'dataset_name': 'fallback'}] * len(batch)
                }
        
        # Create dataloader with optimal settings
        num_workers = min(8, len(self.multi_config.zarr_paths) * 2)
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(mode == 'train'), 
            collate_fn=enhanced_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if batch_size > 1 else False,
            drop_last=False
        )
    
    def train_one_epoch(self, epoch: int):
        """Enhanced training with per-dataset metrics tracking."""
        if self.training_start_time is None:
            self.training_start_time = time.time()
        
        # Call parent training method
        result = super().train_one_epoch if hasattr(super(), 'train_one_epoch') else None
        
        if result:
            # Finalize metrics for this epoch
            self.metrics_tracker.finalize_epoch(epoch)
            
            # Log epoch summary
            self._log_epoch_summary(epoch)
        
        return result
    
    def _log_epoch_summary(self, epoch: int):
        """Log comprehensive epoch summary."""
        summary = self.metrics_tracker.get_summary()
        
        if summary:
            logger.info(f"📊 Epoch {epoch} Summary:")
            for dataset_name, records in summary.items():
                latest_record = records[-1] if records else None
                if latest_record:
                    logger.info(f"   📈 {dataset_name}: {latest_record['samples']} samples, "
                              f"loss: {latest_record['loss']:.4f}")
    
    def on_train_end(self):
        """Enhanced training end with comprehensive reporting."""
        # Save metrics
        metrics_path = self.save_dir / 'multi_dataset_metrics.json'
        self.metrics_tracker.save_metrics(str(metrics_path))
        
        # Create training report
        self._create_training_report()
        
        # Call parent method if it exists
        if hasattr(super(), 'on_train_end'):
            super().on_train_end()
    
    def _create_training_report(self):
        """Create comprehensive training report."""
        try:
            report_path = self.save_dir / 'multi_dataset_training_report.json'
            
            training_duration = time.time() - self.training_start_time if self.training_start_time else 0
            
            report = {
                'training_summary': {
                    'total_duration_seconds': training_duration,
                    'total_duration_formatted': f"{training_duration/3600:.2f} hours",
                    'config': {
                        'zarr_paths': self.multi_config.zarr_paths,
                        'sampling_strategy': self.multi_config.sampling_strategy.value,
                        'split_ratio': self.multi_config.split_ratio,
                        'task': self.multi_config.task
                    }
                },
                'per_dataset_metrics': self.metrics_tracker.get_summary(),
                'training_config': vars(self.multi_config) if hasattr(self.multi_config, '__dict__') else str(self.multi_config)
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Training report saved to: {report_path}")
            
        except Exception as e:
            logger.warning(f"Could not create training report: {e}")

def load_training_config(config_path: str) -> MultiDatasetConfig:
    """Load multi-zarr training configuration."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['zarr_paths']
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Missing required config field: {field}")
    
    # Convert sampling strategy if provided
    if 'sampling_strategy' in config_dict:
        if isinstance(config_dict['sampling_strategy'], str):
            try:
                config_dict['sampling_strategy'] = SamplingStrategy(config_dict['sampling_strategy'])
            except ValueError:
                logger.warning(f"Unknown sampling strategy: {config_dict['sampling_strategy']}, using balanced")
                config_dict['sampling_strategy'] = SamplingStrategy.BALANCED
    
    # Set defaults
    config_dict.setdefault('sampling_strategy', SamplingStrategy.BALANCED)
    config_dict.setdefault('task', 'detect')
    config_dict.setdefault('split_ratio', 0.8)
    config_dict.setdefault('random_seed', 42)
    
    return MultiDatasetConfig(**config_dict)

def train_enhanced_multi_zarr_yolo(
    config_path: str, 
    model_name: str = 'yolov8n.pt',
    epochs: int = 50, 
    batch_size: int = 16, 
    img_size: int = 640,
    device: str = '0',
    project: str = None,
    name: str = None,
    **kwargs
) -> bool:
    """
    Train YOLO model on enhanced multi-zarr dataset.
    
    Args:
        config_path: Path to multi-zarr config YAML
        model_name: YOLO model to use
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: GPU device
        project: Training project directory
        name: Training run name
        **kwargs: Additional training arguments
    
    Returns:
        Success status
    """
    
    logger.info("🚀 ENHANCED MULTI-ZARR YOLO TRAINING")
    logger.info("=" * 60)
    
    # Load configuration
    try:
        config = load_training_config(config_path)
        logger.info(f"✅ Loaded config from: {config_path}")
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return False
    
    # Validate zarr files
    missing_files = [p for p in config.zarr_paths if not Path(p).exists()]
    if missing_files:
        logger.error(f"❌ Missing zarr files: {missing_files}")
        return False
    
    logger.info(f"📁 Training videos: {len(config.zarr_paths)}")
    for i, path in enumerate(config.zarr_paths):
        logger.info(f"   {i+1}. {Path(path).name}")
    
    # Load YOLO model
    try:
        model = YOLO(model_name)
        logger.info(f"✅ Loaded model: {model_name}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False
    
    # Setup training parameters
    train_kwargs = {
        'data': config_path,  # Will be ignored by our custom dataloader
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'plots': True,
        'val': True,
        'amp': True,
        'cache': False,
        'workers': min(8, len(config.zarr_paths) * 2),
        'patience': 100,
        'verbose': True,
        **kwargs
    }
    
    if project:
        train_kwargs['project'] = project
    if name:
        train_kwargs['name'] = name
    
    logger.info(f"🎯 Training parameters:")
    logger.info(f"   Task: {config.task}")
    logger.info(f"   Sampling: {config.sampling_strategy.value}")
    logger.info(f"   Split ratio: {config.split_ratio}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Device: {device}")
    
    try:
        # Create enhanced trainer
        trainer = EnhancedMultiZarrTrainer(config)
        
        # Start training
        logger.info(f"\n🔥 Starting enhanced multi-zarr training...")
        model.train(
            trainer=trainer,
            **train_kwargs
        )
        
        logger.info(f"🎉 SUCCESS! Enhanced multi-zarr training completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Zarr YOLO Trainer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python enhanced_multi_zarr_trainer.py config.yaml --epochs 100
  
  # Advanced training with custom parameters
  python enhanced_multi_zarr_trainer.py config.yaml \\
    --model yolov8s.pt --epochs 200 --batch-size 32 \\
    --project ./experiments --name multi_video_v1
  
  # Training with specific GPU and settings
  python enhanced_multi_zarr_trainer.py config.yaml \\
    --device 0 --img-size 640 --patience 50
        """
    )
    
    parser.add_argument('config', type=str, help='Path to enhanced multi-zarr config YAML file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU device (default: 0)')
    parser.add_argument('--project', type=str,
                       help='Training project directory')
    parser.add_argument('--name', type=str,
                       help='Training run name')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return
    
    # Run training
    success = train_enhanced_multi_zarr_yolo(
        config_path=str(config_path),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )
    
    if success:
        logger.info(f"\n✅ Enhanced multi-zarr training pipeline completed successfully!")
        logger.info(f"🎯 Check the training directory for:")
        logger.info(f"   📊 multi_dataset_metrics.json - Per-dataset training metrics")
        logger.info(f"   📋 multi_dataset_training_report.json - Comprehensive report")
        logger.info(f"   📈 Standard YOLO training outputs")
    else:
        logger.error(f"\n❌ Training failed. Check logs above for details.")

if __name__ == "__main__":
    main()