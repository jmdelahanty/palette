"""
Video downsampling stage for FishEye pipeline.
Creates downsampled versions of imported video data from Zarr arrays.
"""

import zarr
from zarr.codecs import Blosc, BytesCodec
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
import time

from ..utils.system import get_environment_info

def downsample_video(
    zarr_path: str,
    config: Dict[str, Any],
    console: Optional[Console] = None
) -> Path:
    """
    Downsample video data from existing Zarr array.
    
    Args:
        zarr_path: Path to Zarr archive with raw video
        config: Configuration dictionary
        console: Rich console for output
        force: Force re-downsampling even if it exists
    """
    if console is None:
        console = Console()
    
    zarr_path = Path(zarr_path)
    console.rule("[bold]Video Downsampling[/bold]")
    start_time = time.perf_counter()
    
    # Open existing Zarr
    root = zarr.open_group(zarr_path, mode='r+')
    
    # Check if raw video exists
    if 'raw_video' not in root:
        raise ValueError(f"No raw_video group found in {zarr_path}")
    
    raw = root['raw_video']
    if 'images_full' not in raw:
        raise ValueError(f"No full resolution images found in {zarr_path}")
    
    # Check if already downsampled
    if 'images_ds' in raw and not config.get('force_recompute', False):
        console.print("[yellow]Downsampled data already exists. Use --force to re-downsample.[/yellow]")
        return zarr_path
    
    # Get downsample configuration
    ds_config = config.get('downsample', {})
    
    # Extract parameters with defaults
    target_size = tuple(map(int, ds_config.get('size', [256, 256])))
    batch_size = ds_config.get('batch_size', 128)
    chunk_size = ds_config.get('chunk_size', 128)
    method = ds_config.get('method', 'bilinear')
    align_corners = ds_config.get('align_corners', False)
    use_gpu = ds_config.get('use_gpu', True) and torch.cuda.is_available()
    gpu_fp16 = ds_config.get('gpu_fp16', False)
    compression = ds_config.get('compression', 'lz4')
    compression_level = ds_config.get('compression_level', 1)
    skip_if_exists = ds_config.get('skip_if_exists', True)
    
    # Get source array info
    images_full = raw['images_full']
    n_frames, full_h, full_w = images_full.shape
    
    console.print(Panel.fit(
        f"[cyan]Source:[/cyan] {n_frames} frames @ {full_h}×{full_w}\n"
        f"[cyan]Target:[/cyan] {target_size[0]}×{target_size[1]}\n"
        f"[cyan]Method:[/cyan] {method}\n"
        f"[cyan]Device:[/cyan] {'GPU' if use_gpu else 'CPU'}\n"
        f"[cyan]Batch size:[/cyan] {batch_size} frames",
        title="Downsample Configuration"
    ))
    
    # Create downsampled array
    if 'images_ds' in raw and skip_if_exists and not config.get('force_recompute', False):
        console.print("[yellow]Downsampled data already exists. Set force_recompute=true to re-downsample.[/yellow]")
        return zarr_path
    
    if compression == 'none':
        compressors = None
    else:
        compressors = Blosc(cname=compression, clevel=compression_level, shuffle='bitshuffle')

    images_ds = raw.create_array(
        'images_ds',
        shape=(n_frames, target_size[0], target_size[1]),
        chunks=(chunk_size, target_size[0], target_size[1]),
        dtype=np.uint8,
        compressors=compressors
    )
    
    # Setup device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        console.print("[green]Using GPU acceleration[/green]")
    
    # Process in batches
    with Progress(
        TextColumn("[bold blue]Downsampling"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[green]{task.completed}/{task.total} frames"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing", total=n_frames)
        
        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            
            # Read batch from Zarr
            batch_np = images_full[i:end_idx]
            
            # Convert to tensor and add channel dimension
            batch_tensor = torch.from_numpy(batch_np).float().unsqueeze(1)
            
            if device == 'cuda':
                batch_tensor = batch_tensor.cuda()
            
            # Downsample
            with torch.no_grad():
                ds_tensor = F.interpolate(
                    batch_tensor,
                    size=target_size,
                    mode=method,
                    align_corners=align_corners if method in ['bilinear', 'bicubic'] else None
                )
            
            # Convert back to uint8 and numpy
            ds_np = ds_tensor.squeeze(1).byte()
            if device == 'cuda':
                ds_np = ds_np.cpu()
            ds_np = ds_np.numpy()
            
            # Write to Zarr
            images_ds[i:end_idx] = ds_np
            
            progress.update(task, advance=end_idx - i)
            
            # Clear GPU memory periodically
            if device == 'cuda' and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    
    # Add metadata with full provenance
    duration = time.perf_counter() - start_time
    env_info = get_environment_info(include_all_packages=False)
    
    downsample_metadata = {
        'downsample_timestamp': datetime.now(timezone.utc).isoformat(),
        'downsample_duration_seconds': duration,
        'downsample_source_shape': (n_frames, full_h, full_w),
        'downsample_target_shape': (n_frames, target_size[0], target_size[1]),
        'downsample_method': method,
        'downsample_device': device,
        'downsample_batch_size': batch_size,
        'downsample_throughput_fps': n_frames / duration,
        'downsample_reduction_factor': (full_h * full_w) / (target_size[0] * target_size[1]),
        'downsample_git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'downsample_git_branch': env_info['git'].get('branch', 'unknown'),
        'downsample_hostname': env_info['platform']['hostname'],
    }
    
    # Update raw group metadata
    raw.attrs.update(downsample_metadata)
    
    # Also store config used
    if 'pipeline_stages' not in root.attrs:
        root.attrs['pipeline_stages'] = {}
    root.attrs['pipeline_stages']['downsample'] = {
        'config': ds_config,
        'metadata': downsample_metadata
    }
    
    console.print(Panel(
        f"[green]✓ Downsampling completed[/green]\n\n"
        f"[yellow]Performance:[/yellow]\n"
        f"  Time: {duration:.1f}s\n"
        f"  Throughput: {n_frames/duration:.1f} fps\n"
        f"  Reduction: {downsample_metadata['downsample_reduction_factor']:.1f}x\n\n"
        f"[yellow]Output:[/yellow]\n"
        f"  Array: raw_video/images_ds\n"
        f"  Shape: ({n_frames}, {target_size[0]}, {target_size[1]})",
        title="Downsample Complete"
    ))
    
    return zarr_path