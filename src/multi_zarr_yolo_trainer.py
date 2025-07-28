# multi_zarr_yolo_trainer.py
# YOLO trainer with multi-zarr dataset support

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from multi_zarr_yolo_dataset import MultiZarrYOLODataset
import yaml

class FixedDetectionValidator(DetectionValidator):
    """Fixed validator that prevents the 0-d tensor error."""
    
    def _prepare_batch(self, si, batch):
        """Fixed version that handles 0-d tensors properly."""
        try:
            pbatch = super()._prepare_batch(si, batch)
            
            if 'cls' in pbatch:
                cls = pbatch['cls']
                if hasattr(cls, 'ndim') and cls.ndim == 0:
                    pbatch['cls'] = cls.unsqueeze(0)
            
            return pbatch
            
        except Exception as e:
            print(f"❌ Error in _prepare_batch: {e}")
            return {
                'cls': torch.zeros((0,), dtype=torch.float32),
                'bbox': torch.zeros((0, 4), dtype=torch.float32),
                'ori_shape': batch.get('ori_shape', []),
                'ratio_pad': batch.get('ratio_pad', [])
            }

class MultiZarrTrainer(DetectionTrainer):
    """YOLO trainer with multi-zarr dataset support."""
    
    def __init__(self, zarr_paths: List[str], split_ratio: float = 0.8, 
                 random_seed: int = 42, task: str = 'detect',
                 video_weights: Dict[str, float] = None, **kwargs):
        
        self.zarr_paths = zarr_paths
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.task = task
        self.video_weights = video_weights
        
        super().__init__(**kwargs)
    
    def get_validator(self):
        """Return our fixed validator."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return FixedDetectionValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=self.args, 
            _callbacks=self.callbacks
        )
    
    def get_dataloader(self, dataset_path, batch_size, mode="train", **kwargs):
        """Create dataloader with multi-zarr dataset."""
        
        print(f"🔄 Creating {mode} dataloader for multi-zarr dataset...")
        print(f"   📁 Zarr files: {len(self.zarr_paths)}")
        print(f"   🎯 Task: {self.task}")
        
        dataset = MultiZarrYOLODataset(
            zarr_paths=self.zarr_paths,
            mode=mode,
            split_ratio=self.split_ratio,
            random_seed=self.random_seed,
            task=self.task,
            video_weights=self.video_weights
        )
        
        # Show dataset statistics
        stats = dataset.get_video_statistics()
        print(f"   📊 {mode.title()} samples: {stats['total_samples']}")
        for video, count in stats['per_video_counts'].items():
            percentage = stats['per_video_percentages'][video]
            print(f"      {video}: {count} ({percentage:.1f}%)")
        
        def robust_collate_fn(batch):
            """Robust collate function with extra safety checks."""
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
                
                # Ensure cls is always 1D
                if sample_cls.ndim == 0:
                    sample_cls = np.array([sample_cls], dtype=np.float32)
                sample_cls = sample_cls.flatten()
                
                # Ensure bboxes is 2D
                if sample_bboxes.ndim == 1 and len(sample_bboxes) == 4:
                    sample_bboxes = sample_bboxes[None, :]
                
                n_labels = len(sample_cls)
                if n_labels > 0 and sample_bboxes.shape[-1] == 4:
                    end_idx = current_idx + n_labels
                    cls_array[current_idx:end_idx] = sample_cls
                    bboxes_array[current_idx:end_idx] = sample_bboxes.reshape(-1, 4)
                    batch_idx_array[current_idx:end_idx] = batch_i
                    current_idx = end_idx
            
            cls_tensor = torch.from_numpy(cls_array)
            bbox_tensor = torch.from_numpy(bboxes_array)
            batch_idx_tensor = torch.from_numpy(batch_idx_array)
            
            assert cls_tensor.ndim == 1, f"cls_tensor must be 1D, got {cls_tensor.ndim}D"
            
            return {
                'img': images,
                'batch_idx': batch_idx_tensor,
                'cls': cls_tensor,
                'bboxes': bbox_tensor,
                'im_file': [s['im_file'] for s in batch],
                'ori_shape': [s['ori_shape'] for s in batch],
                'ratio_pad': [s['ratio_pad'] for s in batch]
            }
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(mode == 'train'), 
            collate_fn=robust_collate_fn,
            num_workers=min(8, len(self.zarr_paths) * 2),  # Scale workers with number of videos
            pin_memory=True,
            persistent_workers=True if batch_size > 1 else False
        )

def load_config(config_path: str) -> Dict:
    """Load multi-zarr training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['zarr_paths', 'nc', 'names']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Set defaults
    config.setdefault('task', 'detect')
    config.setdefault('target_size', 640 if config['task'] == 'detect' else 320)
    config.setdefault('split_ratio', 0.8)
    config.setdefault('random_seed', 42)
    
    return config

def train_multi_zarr_yolo(config_path: str, model_name: str = 'yolov8n.pt',
                         epochs: int = 50, batch_size: int = 16, 
                         img_size: int = 640, device: str = '0',
                         save_dir: str = None):
    """
    Train YOLO model on multi-zarr dataset.
    
    Args:
        config_path: Path to multi-zarr config YAML
        model_name: YOLO model to use
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: GPU device
        save_dir: Save directory override
    """
    
    print("🚀 MULTI-ZARR YOLO TRAINING")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"✅ Loaded config from: {config_path}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Validate zarr files
    zarr_paths = config['zarr_paths']
    missing_files = [p for p in zarr_paths if not Path(p).exists()]
    if missing_files:
        print(f"❌ Missing zarr files: {missing_files}")
        return False
    
    print(f"📁 Training videos: {len(zarr_paths)}")
    for i, path in enumerate(zarr_paths):
        print(f"   {i+1}. {Path(path).name}")
    
    # Extract video weights if specified
    video_weights = config.get('video_weights', None)
    if video_weights:
        print(f"⚖️  Video weights: {video_weights}")
    
    # Load YOLO model
    try:
        model = YOLO(model_name)
        print(f"✅ Loaded model: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
        'workers': min(8, len(zarr_paths) * 2),
        'patience': 100,
        'verbose': True
    }
    
    if save_dir:
        train_kwargs['project'] = save_dir
    
    print(f"🎯 Training parameters:")
    print(f"   Task: {config['task']}")
    print(f"   Target size: {config.get('target_size', img_size)}")
    print(f"   Split ratio: {config['split_ratio']}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    try:
        # Create custom trainer
        trainer = MultiZarrTrainer(
            zarr_paths=zarr_paths,
            split_ratio=config['split_ratio'],
            random_seed=config['random_seed'],
            task=config['task'],
            video_weights=video_weights
        )
        
        # Start training
        print(f"\n🔥 Starting multi-zarr training...")
        model.train(
            trainer=trainer,
            **train_kwargs
        )
        
        print(f"🎉 SUCCESS! Multi-zarr training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO with multi-zarr dataset",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  python multi_zarr_yolo_trainer.py multi_zarr_config.yaml --epochs 100
  
  # Train with specific model and parameters
  python multi_zarr_yolo_trainer.py config.yaml --model yolov8s.pt --batch-size 32
  
  # Train with custom save directory
  python multi_zarr_yolo_trainer.py config.yaml --save-dir ./multi_video_training
        """
    )
    
    parser.add_argument('config', type=str, help='Path to multi-zarr config YAML file')
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
    parser.add_argument('--save-dir', type=str,
                       help='Save directory (optional)')
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    # Run training
    success = train_multi_zarr_yolo(
        config_path=str(config_path),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        save_dir=args.save_dir
    )
    
    if success:
        print(f"\n✅ Multi-zarr training pipeline completed successfully!")
    else:
        print(f"\n❌ Training failed. Check logs above for details.")

if __name__ == "__main__":
    main()