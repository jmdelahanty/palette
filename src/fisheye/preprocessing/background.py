"""
Background model computation for video analysis.
Computes static background from sampled frames using mode or median.
"""

import random
import time
import numpy as np
import zarr
from datetime import datetime, timezone
from typing import Dict, Optional, List, Literal
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..shared.system_metadata import get_git_info, get_platform_info
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)


def fast_mode_bincount(data_chunk: np.ndarray) -> np.ndarray:
    """
    Calculate mode using bincount - fast for uint8 data.
    
    Args:
        data_chunk: Array of shape (n_samples, height, width)
    
    Returns:
        Background array of shape (height, width)
    """
    n_samples, height, width = data_chunk.shape
    result = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            pixel_values = data_chunk[:, i, j]
            counts = np.bincount(pixel_values, minlength=256)
            result[i, j] = np.argmax(counts)
    
    return result


def compute_median(data_chunk: np.ndarray) -> np.ndarray:
    """Calculate background using median - more robust but slower."""
    return np.median(data_chunk, axis=0).astype(np.uint8)


def compute_background(
    zarr_path: str,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    method: Optional[Literal['mode', 'median']] = None,
    compute_full: Optional[bool] = None,
    compute_ds: Optional[bool] = None,
    config_path: Optional[str] = "pipeline_config.yaml",
    console: Optional[Console] = None
) -> Dict:
    """
    Compute background model from imported video data.
    Creates a new timestamped run for each execution.
    
    Args:
        zarr_path: Path to zarr archive with imported video
        sample_size: Number of frames to sample (-1 for all). If None, uses config.
        seed: Random seed for frame sampling. If None, uses config.
        method: Background computation method. If None, uses config.
        compute_full: Whether to compute full resolution. If None, uses config/auto.
        compute_ds: Whether to compute downsampled. If None, uses config/auto.
        config_path: Path to config file (used if parameters are None)
        console: Rich console for output
    
    Returns:
        Dictionary with computation metadata
    """
    if console is None:
        console = Console()
    
    # Load config if needed
    config_params = {}
    if any(p is None for p in [sample_size, seed, method]):
        try:
            import yaml
            from pathlib import Path
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                    config_params = config.get('background', {})
        except Exception:
            pass  # Use defaults if config can't be loaded
    
    # Get git info using system.py
    git_info = get_git_info()
    
    # Get minimal platform info (not full environment like import)
    platform_info = get_platform_info(collect_ip=False, disk_path=zarr_path)

    # Open zarr root for checking import commit
    root = zarr.open_group(zarr_path, mode='r+')
    
    # Check if running on different commit than import
    import_commit = root['raw_video'].attrs.get('git_commit_hash', 'unknown')
    if import_commit != 'unknown' and git_info['commit_hash'] != 'unknown':
        if import_commit != git_info['commit_hash']:
            console.print(f"[yellow]⚠ Running on different commit than import[/yellow]")
            console.print(f"  Import:  {import_commit[:8]}")
            console.print(f"  Current: {git_info['short_hash']} ({git_info['branch']})")
            if git_info['is_dirty']:
                console.print(f"  [yellow]Working directory has uncommitted changes[/yellow]")


    # Use provided params, then config, then defaults
    sample_size = sample_size if sample_size is not None else config_params.get('sample_size', 100)
    seed = seed if seed is not None else config_params.get('seed', 42)
    method = method if method is not None else config_params.get('method', 'mode')
    
    console.rule("[bold]Computing Background Model[/bold]")
    start_time = time.perf_counter()
    
    # Open zarr
    root = zarr.open_group(zarr_path, mode='r+')
    
    # Check for raw video data
    if 'raw_video' not in root:
        raise ValueError("No raw_video group found. Run import first.")
    
    raw_video = root['raw_video']
    
    # Check what arrays are available
    has_full = 'images_full' in raw_video
    has_ds = 'images_ds' in raw_video
    
    if not has_full and not has_ds:
        raise ValueError("No video arrays found in raw_video")
    
    # Handle compute flags - use provided, then config, then auto
    if compute_full is None:
        compute_full = config_params.get('compute_full', has_full)
    if compute_ds is None:
        compute_ds = config_params.get('compute_ds', has_ds)
    
    # Get array info and frame count
    if has_ds:
        ds_array = raw_video['images_ds']
        n_frames = ds_array.shape[0]
        console.print(f"Found downsampled array: {ds_array.shape}")
    
    if has_full:
        full_array = raw_video['images_full']
        n_frames = full_array.shape[0]
        console.print(f"Found full resolution array: {full_array.shape}")
    
    # Determine frames to use
    if sample_size == -1 or sample_size >= n_frames:
        console.print(f"Using [yellow]all {n_frames}[/yellow] frames for background model")
        frame_indices = list(range(n_frames))
    else:
        random.seed(seed)
        sample_count = min(sample_size, n_frames)
        console.print(f"Randomly sampling [yellow]{sample_count}[/yellow] frames (seed={seed})")
        frame_indices = sorted(random.sample(range(n_frames), sample_count))
    
    # Select computation function
    compute_func = fast_mode_bincount if method == 'mode' else compute_median
    console.print(f"Using [cyan]{method}[/cyan] method")
    
    # Create timestamped run group (matching the old tracker.py pattern)
    parent_group = require_runs_parent(root, 'background_runs')
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"background_{timestamp}"
    bg_group = parent_group.create_group(run_name)
    mark_run_started(bg_group, run_name=run_name, stage="background")
    note_pending_latest(parent_group, run_name)
    
    console.print(f"Created new run group: [cyan]background_runs/{run_name}[/cyan]")
    
    results = {}
    
    # Compute backgrounds
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        if compute_ds and has_ds:
            task = progress.add_task("Computing downsampled background...", total=1)
            
            sample_data = ds_array.get_orthogonal_selection((
                frame_indices, slice(None), slice(None)
            ))
            
            background_ds = compute_func(sample_data)
            
            bg_ds = bg_group.create_array('background_ds',
                                        data=background_ds,
                                        compressors=None,
                                        overwrite=True)
            
            results['background_ds'] = background_ds.shape
            progress.update(task, completed=1)
            console.print(f" Downsampled background computed: {background_ds.shape}")
        
        if compute_full and has_full:
            task = progress.add_task("Computing full resolution background...", total=1)
            
            sample_data = full_array.get_orthogonal_selection((
                frame_indices, slice(None), slice(None)
            ))
            
            background_full = compute_func(sample_data)
            
            bg_full = bg_group.create_array('background_full', 
                                        data=background_full,
                                        compressors=None,
                                        overwrite=True)
            
            results['background_full'] = background_full.shape
            progress.update(task, completed=1)
            console.print(f" Full resolution background computed: {background_full.shape}")
    
    # Store run-specific metadata
    duration = time.perf_counter() - start_time
    bg_group.attrs.update({
        'background_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': duration,
        'parameters': {
            'sample_size': sample_size,
            'seed': seed,
            'method': method,
            'compute_full': compute_full,
            'compute_ds': compute_ds
        },
        'n_frames_used': len(frame_indices),
        'total_frames': n_frames,
        'source_frame_indices': frame_indices if len(frame_indices) < 100 else f"sampled_{len(frame_indices)}_frames",
        'code_version': {
            # Git info
            'git_commit': git_info['commit_hash'],
            'git_short': git_info['short_hash'],
            'git_branch': git_info['branch'],
            'git_dirty': git_info['is_dirty'],
            # Machine info (subset of platform_info)
            'hostname': platform_info['hostname'],
            'system': platform_info['system'],
            # Critical library version for background computation
            'numpy_version': np.__version__,
            # HPC info if running on cluster
            'lsf_job_id': platform_info.get('lsf', {}).get('job_id'),
            'slurm_job_id': platform_info.get('slurm', {}).get('job_id'),
        }
    })
    
    results['duration_seconds'] = duration
    results['frames_used'] = len(frame_indices)
    results['run_name'] = run_name
    mark_run_complete(bg_group, parent_group=parent_group, run_name=run_name)
    
    console.print(f"\n[green]Background computation completed in {duration:.2f} seconds[/green]")
    console.print(f"Results stored in: background_runs/{run_name}/background_{{full,ds}}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute background model from video")
    parser.add_argument("zarr_path", help="Path to zarr archive")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of frames to sample (-1 for all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for frame sampling")
    parser.add_argument("--method", choices=['mode', 'median'], default='mode',
                       help="Background computation method")
    parser.add_argument("--skip-full", action="store_true",
                       help="Skip full resolution background")
    parser.add_argument("--skip-ds", action="store_true",
                       help="Skip downsampled background")
    
    args = parser.parse_args()
    
    console = Console()
    compute_background(
        args.zarr_path,
        sample_size=args.sample_size,
        seed=args.seed,
        method=args.method,
        compute_full=not args.skip_full,
        compute_ds=not args.skip_ds,
        console=console
    )
