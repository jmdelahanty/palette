"""
Fish detection module using blob detection on background-subtracted frames.
Updated to use zarr-first parameter resolution.
"""

import argparse
import time
import sys
import numpy as np
import zarr
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

import skimage
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    resolve_latest_complete_run_name,
)
from ..utils.system import get_git_info, get_platform_info
from ..refinement.detect_quality import analyze_detect_quality, save_quality_report

def _require_imported_detection_inputs(root: zarr.Group, latest_bg_run: str) -> None:
    raw_video = root.get("raw_video")
    if raw_video is None or "images_ds" not in raw_video:
        raise ValueError(
            "Traditional detection requires imported downsampled frames at raw_video/images_ds."
        )

    background_parent = root.get("background_runs")
    if background_parent is None or latest_bg_run not in background_parent:
        raise ValueError(
            f"Traditional detection requires background_runs/{latest_bg_run}."
        )
    if "background_ds" not in background_parent[latest_bg_run]:
        raise ValueError(
            f"Traditional detection requires background_runs/{latest_bg_run}/background_ds."
        )


def get_detection_parameters(
    root: zarr.Group,
    config: Dict[str, Any],
    console: Optional[Console] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Get detection parameters with zarr-first resolution.
    
    Priority order:
    1. Zarr analysis_metadata (tuned parameters)
    2. Config file defaults
    
    Args:
        root: Zarr root group
        config: Loaded config dictionary
        console: Rich console for output
    
    Returns:
        Tuple of (parameters dict, source string)
        source is either 'zarr_tuned' or 'config_default'
    """
    # Start with config defaults
    detect_params = config.get('detect', {}).copy()
    detect_params.setdefault('ds_thresh', 55)
    detect_params.setdefault('se1_radius', 1)
    detect_params.setdefault('se4_radius', 4)
    detect_params.setdefault('min_area', 50)
    detect_params.setdefault('max_area', 500)
    detect_params.setdefault('max_fish', 20)
    
    param_source = 'config_default'

    # Check for tuned parameters in zarr
    if 'analysis_metadata' in root:
        analysis_meta = root['analysis_metadata']
        
        # Check for blob detection tuning (current method)
        if 'detection_tuning' in analysis_meta.attrs:
            tuning_data = analysis_meta.attrs['detection_tuning']
            tuned_params = tuning_data.get('tuned_parameters', {})
            
            if tuned_params:
                # Merge tuned parameters over config defaults
                detect_params.update(tuned_params)
                param_source = 'zarr_tuned'
                
                if console:
                    tuned_date = tuning_data.get('tuned_timestamp', 'unknown')[:10]
                    console.print(f"[green]✓ Using tuned parameters from zarr[/green] (tuned: {tuned_date})")
                    console.print(f"  Threshold: {detect_params['ds_thresh']}, "
                                f"Min area: {detect_params['min_area']}, "
                                f"Max area: {detect_params['max_area']}, "
                                f"Max fish: {detect_params['max_fish']}")
        
        # Check for mask tuning
        if 'dish_mask' in analysis_meta.attrs:
            mask_attr = analysis_meta.attrs['dish_mask']
            mask_data = dict(mask_attr)
            shape = mask_data.get('shape')
            if not shape:
                if 'detected_circle' in mask_data:
                    shape = 'circle'
                elif 'rectangle' in mask_data:
                    shape = 'rectangle'
            if 'dish_mask' not in detect_params:
                detect_params['dish_mask'] = {}
            if shape == 'rectangle' and 'rectangle' in mask_data:
                roi = mask_data['rectangle'].get('roi')
                if roi:
                    detect_params['dish_mask'].update({
                        'shape': 'rectangle',
                        'roi': [int(v) for v in roi],
                    })
                    if console:
                        console.print(f"[green]✓ Using rectangular dish mask from zarr[/green]")
            elif shape == 'circle' and 'detected_circle' in mask_data:
                circle = mask_data['detected_circle']
                detect_params['dish_mask'].update({
                    'shape': 'circle',
                    'center': circle['center'],
                    'radius': circle['radius'],
                })
                if console:
                    console.print(f"[green]✓ Using circular dish mask from zarr[/green]")
    
    # No tuned parameters found - using config
    if param_source == 'config_default' and console:
        console.print(f"[yellow]⚠️  Using default config parameters - consider tuning first[/yellow]")
        console.print(f"  Run: python -m fisheye --tune mask")
        console.print(f"  Run: python -m fisheye --tune detect")
    
    return detect_params, param_source


def create_dish_mask(mask_params: Dict, img_shape: Tuple[int, int], console: Optional[Console] = None) -> Optional[np.ndarray]:
    """
    Create a dish mask based on parameters from config or analysis_metadata.
    
    Args:
        mask_params: Dictionary containing mask parameters
        img_shape: Tuple of (height, width) for the image
        console: Rich console for output
    
    Returns:
        numpy array mask or None if no mask defined
    """
    import cv2
    
    if not mask_params:
        return None
    
    dish_shape = mask_params.get('shape', 'circle')
    mask = None
    
    if dish_shape == 'rectangle' and 'roi' in mask_params:
        x, y, w, h = mask_params['roi']
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        if console:
            console.print(f"  Using rectangular mask: x={x}, y={y}, w={w}, h={h}")
            
    elif dish_shape == 'circle':
        center = mask_params.get('center')
        radius = mask_params.get('radius')
        
        if center and radius:
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.circle(mask, tuple(center), radius, 255, -1)
            if console:
                console.print(f"  Using circular mask: center={center}, radius={radius}")
    
    return mask


@delayed
def detect_chunk_delayed(zarr_path: str, chunk_slice: slice, detect_params: Dict, 
                         dish_mask: Optional[np.ndarray], latest_bg_run: str) -> Tuple:
    """
    Detects fish in a chunk of downsampled frames.
    
    Args:
        zarr_path: Path to zarr archive
        chunk_slice: Slice of frames to process
        detect_params: Detection parameters
        dish_mask: Optional mask for dish region
        latest_bg_run: Name of background run to use
    
    Returns:
        Tuple of (chunk_slice, frame_indices_list, bbox_norms_list)
    """
    se1 = disk(detect_params['se1_radius'])
    se4 = disk(detect_params['se4_radius'])

    root = zarr.open(zarr_path, mode='r')
    images_ds_chunk = root['raw_video/images_ds'][chunk_slice]
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    ds_img_shape = images_ds_chunk.shape[1:]

    chunk_len = images_ds_chunk.shape[0]
    frame_start = chunk_slice.start
    
    all_bbox_norms = []
    all_frame_indices = []

    _require_imported_detection_inputs(root, latest_bg_run)

    for i in range(chunk_len):
        frame_idx = frame_start + i
        
        diff_ds = np.clip(background_ds.astype(np.int16) - images_ds_chunk[i].astype(np.int16), 0, 255).astype(np.uint8)
        if dish_mask is not None:
            diff_ds[dish_mask == 0] = 0

        current_thresh = detect_params['ds_thresh']
        valid_blobs = []
        
        # Adaptive thresholding - try up to 5 threshold values
        for _ in range(5):
            im_ds = erosion(dilation(erosion(diff_ds >= current_thresh, se1), se4), se1)
            all_blobs = regionprops(label(im_ds))
            valid_blobs = [r for r in all_blobs if detect_params['min_area'] <= r.area <= detect_params['max_area']]
            if valid_blobs:
                break
            current_thresh -= 5
        
        if not valid_blobs:
            continue

        # Keep only top N fish by area
        sorted_blobs = sorted(valid_blobs, key=lambda r: r.area, reverse=True)[:detect_params.get('max_fish', 20)]

        for blob in sorted_blobs:
            min_r, min_c, max_r, max_c = blob.bbox
            center_y, center_x = (min_r + max_r) / 2, (min_c + max_c) / 2
            height, width = max_r - min_r, max_c - min_c
            
            # Normalized coordinates
            center_norm = np.array([center_x / ds_img_shape[1], center_y / ds_img_shape[0]])
            size_norm = np.array([width / ds_img_shape[1], height / ds_img_shape[0]])
            all_bbox_norms.append([*center_norm, *size_norm])
            all_frame_indices.append(frame_idx)
            
    return chunk_slice, all_frame_indices, all_bbox_norms


def detect_fish(
    zarr_path: str,
    config_path: Optional[str] = "pipeline_config.yaml",
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Detect fish in video frames using blob detection."""
    if console is None:
        console = Console()
    
    console.rule("[bold]Fish Detection[/bold]")
    start_time = time.perf_counter()
    
    # Load config
    import yaml
    from pathlib import Path
    
    config = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Open zarr
    root = zarr.open_group(zarr_path, mode='r+')
    
    if 'background_runs' not in root:
        raise ValueError("Background stage not run. Run background computation first.")
    
    # Get detection parameters with zarr-first resolution
    detect_params, param_source = get_detection_parameters(root, config, console)
    
    # Get version info
    git_info = get_git_info()
    platform_info = get_platform_info(collect_ip=False, disk_path=zarr_path)
    
    latest_bg_run = resolve_latest_complete_run_name(root["background_runs"])
    if latest_bg_run is None:
        raise ValueError("Traditional detection requires a complete background run.")
    _require_imported_detection_inputs(root, latest_bg_run)
    console.print(f"Using background: [cyan]{latest_bg_run}[/cyan]")
    
    # Create detect runs group
    parent_group = require_runs_parent(root, 'detect_runs')
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"detect_{timestamp}"
    detect_group = parent_group.create_group(run_name)
    mark_run_started(detect_group, run_name=run_name, stage="detect")
    note_pending_latest(parent_group, run_name)
    
    console.print(f"Created new run: [cyan]detect_runs/{run_name}[/cyan]")
    
    # Create dish mask
    mask_params = detect_params.get('dish_mask', {})
    background_ds = root[f'background_runs/{latest_bg_run}/background_ds'][:]
    ds_img_shape = background_ds.shape
    mask = create_dish_mask(mask_params, ds_img_shape, console)
    
    # Get video info
    num_images = root['raw_video/images_ds'].shape[0]
    chunk_size = root['raw_video/images_ds'].chunks[0]
    
    console.print(f"Processing {num_images} frames in chunks of {chunk_size}")
    
    # Create dask tasks
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for detection...")
    
    delayed_tasks = [detect_chunk_delayed(zarr_path, s, detect_params, mask, latest_bg_run) for s in chunk_slices]
    
    if show_progress:
        with ProgressBar():
            results = dask.compute(*delayed_tasks)
    else:
        results = dask.compute(*delayed_tasks)
    
    console.print("Writing detection results to Zarr...")
    
    # Collect all results
    all_frame_indices: List[int] = []
    all_bboxes: List[List[float]] = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        aggregate_task = progress.add_task(
            "[cyan]Aggregating detections...", total=len(results)
        )
        for slc, frame_idxs, bboxes in results:
            if frame_idxs:
                all_frame_indices.extend(frame_idxs)
                all_bboxes.extend(bboxes)
            progress.advance(aggregate_task)
    
    total_detections = len(all_frame_indices)
    console.print(f"Aggregating {total_detections} total detections...")
    
    # Write core detection arrays
    if total_detections > 0:
        frame_indices_np = np.array(all_frame_indices, dtype='i4')
        bboxes_np = np.array(all_bboxes, dtype='f8')
        
        det_chunk = min(chunk_size * 4, total_detections)

        detect_group.create_array(
            'frame_indices',
            data=frame_indices_np,
            chunks=(det_chunk,),
            overwrite=True
        )
        
        detect_group.create_array(
            'bbox_norm_coords',
            data=bboxes_np,
            chunks=(det_chunk, 4),
            overwrite=True
        )
        detect_group.create_array(
            'scores',
            data=np.ones(total_detections, dtype=np.float32),
            chunks=(det_chunk,),
            overwrite=True
        )
        detect_group.create_array(
            'class_ids',
            data=np.zeros(total_detections, dtype=np.int32),
            chunks=(det_chunk,),
            overwrite=True
        )
        
        # Compute and store frame counts for visualization
        console.print("Computing frame counts for visualization...")
        frame_counts = np.bincount(frame_indices_np, minlength=num_images)
        detect_group.create_array(
            'frame_counts',
            data=frame_counts,
            chunks=(min(chunk_size * 4, num_images),),
            overwrite=True
        )
        detect_group.create_array(
            'n_detections',
            data=frame_counts,
            chunks=(min(chunk_size * 4, num_images),),
            overwrite=True
        )
        
    else:
        # No detections found
        detect_group.create_array('frame_indices', data=np.empty((0,), dtype='i4'), overwrite=True)
        detect_group.create_array('bbox_norm_coords', data=np.empty((0, 4), dtype='f8'), overwrite=True)
        detect_group.create_array('scores', data=np.empty((0,), dtype=np.float32), overwrite=True)
        detect_group.create_array('class_ids', data=np.empty((0,), dtype=np.int32), overwrite=True)
        frame_counts_empty = np.zeros(num_images, dtype='i4')
        chunks = (min(chunk_size * 4, num_images),) if num_images else None
        detect_group.create_array(
            'frame_counts',
            data=frame_counts_empty,
            chunks=chunks,
            overwrite=True
        )

    # Calculate statistics
    if total_detections > 0:
        frame_counts = detect_group['frame_counts'][:]
        frames_with_detections = int(np.sum(frame_counts > 0))
        percent_detected = (frames_with_detections / num_images) * 100
        max_detections = int(np.max(frame_counts))
        mean_detections = float(np.mean(frame_counts[frame_counts > 0]))
        
        # Distribution breakdown
        distribution = {
            'frames_with_0': int(np.sum(frame_counts == 0)),
            'frames_with_1': int(np.sum(frame_counts == 1)),
            'frames_with_2': int(np.sum(frame_counts == 2)),
            'frames_with_3_to_5': int(np.sum((frame_counts >= 3) & (frame_counts <= 5))),
            'frames_with_6_plus': int(np.sum(frame_counts >= 6)),
        }
    else:
        frames_with_detections = 0
        percent_detected = 0.0
        max_detections = 0
        mean_detections = 0.0
        distribution = {
            'frames_with_0': num_images,
            'frames_with_1': 0,
            'frames_with_2': 0,
            'frames_with_3_to_5': 0,
            'frames_with_6_plus': 0,
        }
    
    # Store metadata (following unified spec)
    duration = time.perf_counter() - start_time
    
    detect_group.attrs.update({
        # Core detection metadata (per unified spec)
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': float(duration),
        'method': 'blob',
        'detection_method': 'blob',
        'detection_source': 'zarr_video',  # Blob uses imported video in zarr
        'total_frames': num_images,
        'has_raw_video': True,
        
        # Detection parameters (method-specific)
        'parameters': detect_params,
        'parameter_source': param_source,
        
        # Background subtraction info (method-specific)
        'source_background_run': latest_bg_run,
        
        # Processing info
        'dask_scheduler': scheduler,
        
        # Summary statistics
        'summary_statistics': {
            'total_frames': num_images,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'percent_frames_with_detections': round(percent_detected, 2),
            'max_detections_per_frame': max_detections,
            'mean_detections_per_frame': round(mean_detections, 2),
            'distribution': distribution
        },
        
        # Code version
        'code_version': {
            'git_commit': git_info['commit_hash'],
            'git_short': git_info['short_hash'],
            'git_branch': git_info['branch'],
            'git_dirty': git_info['is_dirty'],
            'hostname': platform_info['hostname'],
            'system': platform_info['system'],
            'numpy_version': np.__version__,
            'scikit_image_version': skimage.__version__,
            'dask_version': dask.__version__,
            'lsf_job_id': platform_info.get('lsf', {}).get('job_id'),
            'slurm_job_id': platform_info.get('slurm', {}).get('job_id'),
        }
    })
    
    console.print(f"[green]Detection completed in {duration:.2f} seconds[/green]")
    console.print(f"Found [green]{total_detections}[/green] fish in [green]{frames_with_detections}/{num_images}[/green] frames ([cyan]{percent_detected:.2f}%[/cyan])")
    
    if total_detections > 0:
        console.print(f"  Max detections/frame: [yellow]{max_detections}[/yellow], Mean: [yellow]{mean_detections:.1f}[/yellow]")
        console.print(f"  Distribution: 0=[cyan]{distribution['frames_with_0']}[/cyan], "
                     f"1=[cyan]{distribution['frames_with_1']}[/cyan], "
                     f"2=[cyan]{distribution['frames_with_2']}[/cyan], "
                     f"3-5=[cyan]{distribution['frames_with_3_to_5']}[/cyan], "
                     f"6+=[cyan]{distribution['frames_with_6_plus']}[/cyan]")

    provenance_record = build_stage_provenance(
        stage='detect',
        command=' '.join(sys.argv),
        created_at_utc=str(detect_group.attrs.get('detect_timestamp_utc') or datetime.now(timezone.utc).isoformat()),
        version=git_info.get('short_hash') or git_info.get('commit_hash'),
        git={
            'commit': git_info.get('commit_hash'),
            'short': git_info.get('short_hash'),
            'branch': git_info.get('branch'),
            'is_dirty': git_info.get('is_dirty'),
            'remote': git_info.get('remote_url'),
        },
        platform=platform_info,
        parameters={
            **dict(detect_params),
            'parameter_source': param_source,
            'scheduler': scheduler,
            'config_path': config_path,
        },
        inputs={
            'frame_source': 'zarr',
            'source_video_path': root.attrs.get('source_video_path'),
            'source_background_run': latest_bg_run,
        },
    )
    write_stage_provenance(detect_group, provenance_record)
    mark_run_complete(detect_group, parent_group=parent_group, run_name=run_name)
    
    # Run quality analysis
    console.print("\n[bold]Running detection quality analysis...[/bold]")
    try:
        quality_report = analyze_detect_quality(
            zarr_path=zarr_path,
            run_name=run_name,
            jump_threshold=100.0
        )
        
        save_quality_report(zarr_path, quality_report, console=console)
        
        # Print quick quality summary
        score = quality_report['quality_score']
        console.print(f"Quality Grade: [bold]{score['grade']}[/bold] ({score['overall_score']:.1f}/100)")
        console.print(f"  Artifacts found: {quality_report['artifacts']['total_artifacts']}")
        
    except Exception as e:
        console.print(f"[yellow]Quality analysis failed: {e}[/yellow]")

    return {
        'duration_seconds': duration,
        'total_detections': total_detections,
        'frames_with_detections': frames_with_detections,
        'run_name': run_name,
        'parameter_source': param_source
    }


# ---------------- CLI ---------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run traditional blob-based detection on a Palette Zarr archive."
    )
    parser.add_argument(
        "--zarr-path",
        required=True,
        help="Path to the Palette Zarr archive.",
    )
    parser.add_argument(
        "--config",
        default="pipeline_config.yaml",
        help="Optional pipeline configuration YAML (default: pipeline_config.yaml).",
    )
    parser.add_argument(
        "--scheduler",
        default="processes",
        choices=["threads", "processes", "single-threaded"],
        help="Dask scheduler to use (default: processes).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Optional worker count hint for multi-processing schedulers.",
    )
    parser.add_argument(
        "--no-dask-progress",
        action="store_true",
        help="Disable the Dask progress bar (useful when embedding in other progress displays).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console()
    try:
        result = detect_fish(
            zarr_path=args.zarr_path,
            config_path=args.config,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=console,
            show_progress=not args.no_dask_progress,
        )
    except Exception as exc:
        console.print(f"[bold red]Detection failed:[/bold red] {exc}")
        return 1

    console.print(
        f"[green]Done.[/green] Run '{result['run_name']}' "
        f"with {result['total_detections']} detections "
        f"({result['frames_with_detections']} frames)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
