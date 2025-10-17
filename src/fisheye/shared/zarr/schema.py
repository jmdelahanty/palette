"""
Standardized Zarr schema for the Palette ecosystem using Zarr v3.

This schema defines the structure that all packages expect and produce.
"""

from typing import Dict, Any, Optional, Tuple, List
import zarr
from zarr.storage import LocalStore
import zarr.codecs
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import platform
from rich.console import Console

# Import our existing system utilities
from fisheye.utils.system import (
    get_git_info,
    get_platform_info,
    get_gpu_info,
    get_software_versions,
    get_environment_summary
)

ZARR_SCHEMA_VERSION = "3.0.0"

ZARR_SCHEMA = {
    "version": ZARR_SCHEMA_VERSION,
    "zarr_format": 3,
    
    "root_attributes": {
        "schema_version": "Schema version string (3.0)",
        "zarr_format": "Zarr format version (3)",
        "created_at": "ISO timestamp of creation",
        "pipeline_version": "Version of fisheye used",
        "source_video": "Original video filename",
        "fps": "Frames per second",
        "total_frames": "Total number of frames",
        "width": "Video width in pixels",
        "height": "Video height in pixels",
        "processing_history": "List of processing steps applied"
    },
    
    "groups": {
        "raw_video": {
            "description": "Original video data",
            "arrays": {
                "images_full": {
                    "description": "Full resolution frames (optional)",
                    "dtype": "uint8",
                    "dimensions": ["frame", "height", "width"],
                    "chunks": "Auto-calculated based on size"
                },
                "images_ds": {
                    "description": "Downsampled frames for processing",
                    "dtype": "uint8", 
                    "dimensions": ["frame", "height", "width"],
                    "chunks": "(100, height, width)"
                },
                "timestamps": {
                    "description": "Frame timestamps",
                    "dtype": "float64",
                    "dimensions": ["frame"]
                }
            }
        },
        
        "processing": {
            "description": "Processing pipeline results organized by run",
            "subgroups": {
                "background": {
                    "description": "Background subtraction runs",
                    "run_structure": {
                        "background_{timestamp}": {
                            "arrays": {
                                "background": "Computed background image",
                                "background_std": "Standard deviation"
                            },
                            "attributes": {
                                "created_at": "ISO timestamp",
                                "method": "median/mean/mode",
                                "num_samples": "Number of frames sampled",
                                "parameters": "Method-specific parameters"
                            }
                        }
                    }
                },
                
                "detection": {
                    "description": "Detection results from various methods",
                    "run_structure": {
                        "detect_{timestamp}": {
                            "arrays": {
                                "n_detections": {
                                    "shape": "(n_frames,)",
                                    "dtype": "int32"
                                },
                                "bboxes": {
                                    "shape": "(n_total_detections, 4)",
                                    "dtype": "float32",
                                    "description": "Normalized [x, y, w, h]"
                                },
                                "scores": {
                                    "shape": "(n_total_detections,)",
                                    "dtype": "float32"
                                },
                                "class_ids": {
                                    "shape": "(n_total_detections,)",
                                    "dtype": "int32"
                                }
                            },
                            "attributes": {
                                "method": "blob/yolo_detect/yolo_pose",
                                "model_version": "For ML methods",
                                "parameters": "Detection parameters"
                            }
                        }
                    }
                },
                
                "tracking": {
                    "description": "Tracking and keypoint results",
                    "run_structure": {
                        "track_{timestamp}": {
                            "arrays": {
                                "keypoints": {
                                    "shape": "(n_detections, n_keypoints, 2)",
                                    "dtype": "float32",
                                    "description": "Normalized keypoint coordinates"
                                },
                                "identities": {
                                    "shape": "(n_detections,)",
                                    "dtype": "int32",
                                    "description": "Fish/ROI identity assignments"
                                }
                            }
                        }
                    }
                }
            }
        },
        
        "analysis": {
            "description": "Analysis results",
            "subgroups": {
                "metrics": {
                    "description": "Computed behavioral metrics",
                    "arrays": {
                        "speed": "Frame-by-frame speed",
                        "distance": "Cumulative distance",
                        "approach_metrics": "Chaser-specific metrics"
                    }
                },
                "filtered": {
                    "description": "Filtered/cleaned tracking data"
                },
                "interpolated": {
                    "description": "Gap-filled tracking data"
                }
            }
        },
        
        "metadata": {
            "description": "Experiment and processing metadata",
            "attributes": {
                "experiment_config": "Original experiment configuration",
                "processing_config": "Pipeline configuration used",
                "roi_definitions": "ROI boundaries if applicable",
                "stimulus_events": "Stimulus timing if applicable"
            }
        }
    }
}
def _auto_shard_frames(height: int, width: int, bytes_per_pixel: int = 1,
                       target_mb: int = 128, min_frames: int = 1, max_frames: int = 64) -> int:
    """Choose frames/shard so each shard ≈ target_mb (clamped)."""
    bytes_per_frame = height * width * bytes_per_pixel
    if bytes_per_frame == 0:
        return min_frames
    frames = max(min_frames, int((target_mb * 1024 * 1024) // bytes_per_frame))
    return max(min_frames, min(max_frames, frames))


def create_palette_zarr(
    path: str,
    video_metadata: Dict[str, Any],
    config: Dict[str, Any],
    use_sharding: bool = False,
    cli_args: Optional[Dict[str, Any]] = None
) -> zarr.Group:
    store = LocalStore(path)
    root = zarr.open_group(store=store, mode='w', zarr_format=3)

    # ---- Root attrs ---------------------------------------------------------
    if cli_args:
        root.attrs['command_line_args'] = cli_args
    root.attrs['git_info'] = get_git_info()
    root.attrs['platform_info'] = get_platform_info(collect_ip=False, disk_path=path)
    root.attrs['source_video_metadata'] = video_metadata

    env_summary = get_environment_summary()
    root.attrs['software_versions'] = env_summary.get('key_packages', {})
    root.attrs['environment'] = {
        'type': env_summary.get('environment_type', 'unknown'),
        'name': env_summary.get('environment_name', 'none'),
        'python_version': env_summary.get('python_version', platform.python_version()),
    }
    root.attrs['pipeline_version'] = '2.0-multi-fish'
    root.attrs['zarr_format'] = 3
    root.attrs['zarr_python'] = zarr.__version__
    root.attrs['schema_version'] = '3.0.0'  # keep consistent with your constant

    # ---- Pipeline params ----------------------------------------------------
    param_group = root.create_group('pipeline_params')
    for stage, stage_params in config.items():
        param_group.attrs[stage] = stage_params

    # ---- Groups -------------------------------------------------------------
    metadata = root.create_group('metadata')
    raw_video = root.create_group('raw_video')

    height  = int(video_metadata.get('height', 1080))
    width   = int(video_metadata.get('width', 1920))
    n_frames = int(video_metadata.get('total_frames', 1))

    # Import config
    import_config = config.get('import', {})
    ds_height, ds_width = import_config.get('downsample_size', [640, 640])

    # ---- compression mapping ------------------------------------------------
    comp_name = (import_config.get('compression', 'lz4') or 'none').lower()
    clevel    = int(import_config.get('compression_level', 1))

    def compressors_for(name: str, level: int):
        if name == 'none':
            return []
        if name in ('lz4', 'zstd'):
            return [zarr.codecs.BloscCodec(cname=name, clevel=level, shuffle='bitshuffle')]
        if name == 'blosc':  # treat 'blosc' as lz4 default
            return [zarr.codecs.BloscCodec(cname='lz4', clevel=level, shuffle='bitshuffle')]
        # fallback: no compression
        return []

    compressors_full = compressors_for(comp_name, clevel)
    compressors_ds   = compressors_for(comp_name, clevel)

    # ---- shard sizes ----------------------------------
    shard_size_full = int(import_config.get('shard_size', _auto_shard_frames(height, width, 1)))
    shard_size_ds   = int(import_config.get('shard_size_ds', shard_size_full))

    # ---- serializer (CRC enabled; set checksum=False to disable) ------------
    ser = zarr.codecs.BytesCodec()  # or BytesCodec(checksum=False) while debugging

    # Record sharding choices
    raw_video.attrs['sharding'] = {
        'images_full': {'frames_per_shard': shard_size_full, 'shard_shape': (shard_size_full, height,  width)},
        'images_ds':   {'frames_per_shard': shard_size_ds,   'shard_shape': (shard_size_ds,   ds_height, ds_width)},
    }

    # Raw-video attrs (reflect actual compressor choice)
    raw_video.attrs.update({
        'import_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'original_resolution': (height, width),
        'downsampled_resolution': (ds_height, ds_width),
        'fps': video_metadata.get('fps', 30),
        'total_frames': n_frames,
        'source_video': video_metadata.get('source_video', 'unknown'),
        'decoding_device': (get_gpu_info().get('devices', [{'name': 'N/A'}])[0].get('name', 'N/A')),
        'compressor': {'name': comp_name, 'clevel': clevel, 'shuffle': 'bitshuffle'},
        'duration_seconds': None,
    })

    # ---- arrays -------------------------------------------------------------
    raw_video.create_array(
        name='images_full',
        shape=(n_frames, height, width),
        chunks=(1, height, width),
        shards=(shard_size_full, height, width),
        dtype=np.uint8,
        fill_value=0,
        serializer=ser,
        compressors=compressors_full,
    )
    raw_video.create_array(
        name='images_ds',
        shape=(n_frames, ds_height, ds_width),
        chunks=(1, ds_height, ds_width),
        shards=(shard_size_ds, ds_height, ds_width),
        dtype=np.uint8,
        fill_value=0,
        serializer=ser,
        compressors=compressors_ds,
    )
    raw_video.create_array(
        name='timestamps',
        shape=(n_frames,),
        chunks=(min(1000, n_frames),),
        dtype=np.float64,
        fill_value=np.nan,
        serializer=ser,
        compressors=[],  # no compression for small 1D
    )

    # Processing / runs
    processing = root.create_group('processing')
    for g in ('background', 'detection', 'tracking',
              'background_runs', 'detect_runs', 'crop_runs',
              'tracking_runs', 'id_assignments_runs'):
        processing.create_group(g)
    for g in ('background_runs', 'detect_runs', 'crop_runs', 'tracking_runs', 'id_assignments_runs'):
        processing[g].attrs['latest'] = None
    processing['tracking_runs'].attrs['best'] = None

    results = root.create_group('results')
    for g in ('detections', 'tracks', 'keypoints'):
        results.create_group(g)

    gi = get_git_info()
    metadata.attrs.update({
        'git_commit': gi.get('commit_hash', 'N/A'),
        'git_branch': gi.get('branch', 'N/A'),
        'git_dirty': gi.get('is_dirty', False),
    })
    pi = root.attrs['platform_info']
    metadata.attrs.update({
        'hostname': pi.get('hostname', 'unknown'),
        'system': pi.get('system', 'unknown'),
        'cpu_cores': pi.get('cpu_cores', 1),
    })
    return root




def get_run_group(
    root: zarr.Group, 
    stage_name: str, 
    console: Optional[Console] = None,
    create_new: bool = True
) -> Tuple[zarr.Group, str]:
    """
    Get or create a run group for a pipeline stage with timestamp.
    
    Args:
        root: Zarr root group
        stage_name: Stage name (e.g., 'detect', 'crop', 'keypoints')
        console: Rich console for output (optional)
        create_new: Whether to create a new run group
        
    Returns:
        Tuple of (run_group, run_group_name)
    """
    from datetime import datetime, timezone
    
    parent_group_name = f'{stage_name}_runs'
    
    if parent_group_name not in root:
        parent_group = root.create_group(parent_group_name)
    else:
        parent_group = root[parent_group_name]
    
    if create_new:
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
        base_name = f'{stage_name}_{timestamp}'
        run_group_name = base_name
        suffix = 1
        while run_group_name in parent_group:
            run_group_name = f"{base_name}_{suffix:03d}"
            suffix += 1
        run_group = parent_group.create_group(run_group_name)
        parent_group.attrs['latest'] = run_group_name
        
        if console:
            console.print(f"Created run group: [cyan]{parent_group_name}/{run_group_name}[/cyan]")
    else:
        run_group_name = parent_group.attrs.get('latest')
        if not run_group_name:
            raise ValueError(f"No existing run found for {stage_name}")
        run_group = parent_group[run_group_name]
    
    return run_group, run_group_name

def create_background_arrays(
    bg_group: zarr.Group,
    img_shape: Tuple[int, int],
    ds_img_shape: Tuple[int, int] | None = None,
    source_frame_indices: Any = None,
) -> tuple[zarr.Array, zarr.Array]:
    """
    Create arrays for storing background calculation results (Zarr v3 API).

    Returns:
        (background_full, background_ds)
    """
    ser = zarr.codecs.BytesCodec()
    lz4 = zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")

    if ds_img_shape is None:
        ds_img_shape = img_shape

    # Single-chunk scalar images are fine for backgrounds.
    background_full = bg_group.create_array(
        "background_full",
        shape=img_shape,
        chunks=img_shape,
        dtype="uint8",
        fill_value=0,
        serializer=ser,
        compressors=[lz4],
    )

    background_ds = bg_group.create_array(
        "background_ds",
        shape=ds_img_shape,
        chunks=ds_img_shape,
        dtype="uint8",
        fill_value=0,
        serializer=ser,
        compressors=[lz4],
    )

    # Store frame indices used
    if source_frame_indices is not None:
        if isinstance(source_frame_indices, str) and source_frame_indices == "all":
            bg_group.attrs["source_frame_indices"] = "all"
        else:
            if hasattr(source_frame_indices, "tolist"):
                source_frame_indices = source_frame_indices.tolist()
            bg_group.attrs["source_frame_indices"] = source_frame_indices

    return background_full, background_ds


def create_detection_arrays(
    detect_group: zarr.Group,
    n_frames: int,
    chunk_size: int = 32,
) -> tuple[zarr.Array, zarr.Array]:
    """
    Create arrays for storing detection results (Zarr v3 API).

    Returns:
        (n_detections, bbox_norm_coords)
    """
    ser = zarr.codecs.BytesCodec()
    lz4 = zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")

    # Per-frame counts
    n_detections = detect_group.create_array(
        "n_detections",
        shape=(n_frames,),
        chunks=(max(1, chunk_size * 4),),
        dtype="i4",
        fill_value=0,
        serializer=ser,
        compressors=[lz4],
    )

    # Growable table of boxes: start with 0 rows; each row = [x, y, w, h] (normalized)
    bbox_norm_coords = detect_group.create_array(
        "bbox_norm_coords",
        shape=(0, 4),                            # start empty; use .resize(new_rows, 4) when appending
        chunks=(max(1, chunk_size * 4), 4),     # chunks must be all integers (no None)
        dtype="f8",
        fill_value=np.nan,
        serializer=ser,
        compressors=[lz4],
    )

    return n_detections, bbox_norm_coords


def create_tracking_arrays(
    track_group: zarr.Group,
    n_frames: int,
    total_detections: int = 0,
    chunk_size: int = 32,
) -> tuple[zarr.Array, zarr.Array]:
    """
    Create arrays for storing tracking results (Zarr v3 API).

    Returns:
        (n_detections, tracking_results)
    """
    ser = zarr.codecs.BytesCodec()
    lz4 = zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")

    # Per-frame tracked counts
    n_tracked = track_group.create_array(
        "n_detections",
        shape=(n_frames,),
        chunks=(max(1, chunk_size * 4),),
        dtype="i4",
        fill_value=0,
        serializer=ser,
        compressors=[lz4],
    )

    # Growable table of tracking features; start empty
    # If you already know an upper bound, you can start with (total_detections, 21) instead.
    tracking_results = track_group.create_array(
        "tracking_results",
        shape=(0, 21),
        chunks=(max(1, chunk_size * 4), 21),
        dtype="f8",
        fill_value=np.nan,
        serializer=ser,
        compressors=[lz4],
    )

    tracking_results.attrs["column_names"] = [
        "heading_degrees", "bladder_x_roi_norm", "bladder_y_roi_norm",
        "eye_l_x_roi_norm", "eye_l_y_roi_norm", "eye_r_x_roi_norm", "eye_r_y_roi_norm",
        "bbox_x_norm_ds", "bbox_y_norm_ds", "bbox_width_norm_ds", "bbox_height_norm_ds",
        "bbox_x_norm_full", "bbox_y_norm_full", "bbox_width_norm_full", "bbox_height_norm_full",
        "roi_x1_full", "roi_y1_full", "roi_x1_ds", "roi_y1_ds",
        "confidence_score", "effective_threshold",
    ]

    return n_tracked, tracking_results


def add_processing_run(
    root: zarr.Group,
    stage: str,
    parameters: Dict[str, Any],
    source_runs: Optional[Dict[str, str]] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
    extra_attrs: Optional[Dict[str, Any]] = None
) -> zarr.Group:
    """
    Add a new processing run with full metadata.
    
    Args:
        root: Root zarr group
        stage: Stage name (e.g., 'background', 'detect', 'track')
        parameters: Stage parameters used
        source_runs: Dict of source stage names to run names
        summary_stats: Optional summary statistics
        duration_seconds: Processing duration
        extra_attrs: Additional stage-specific attributes
    
    Returns:
        The created run group
    """
    run_group = get_run_group(root, stage, create_new=True)
    
    # Add standard metadata
    run_group.attrs.update({
        f'{stage}_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': parameters,
    })
    
    # Add source run references
    if source_runs:
        for source_stage, source_run in source_runs.items():
            run_group.attrs[f'source_{source_stage}_run'] = source_run
    
    # Add summary statistics
    if summary_stats:
        run_group.attrs['summary_statistics'] = summary_stats
    
    # Add duration
    if duration_seconds is not None:
        run_group.attrs['duration_seconds'] = duration_seconds
    
    # Add any extra stage-specific attributes
    if extra_attrs:
        run_group.attrs.update(extra_attrs)
    
    # Add system info at time of processing
    run_group.attrs['processing_host'] = platform.node()
    
    # Check if GPU was used
    gpu_info = get_gpu_info()
    if gpu_info and gpu_info.get('available'):
        run_group.attrs['gpu_used'] = True
        run_group.attrs['gpu_device'] = gpu_info['devices'][0].get('name', 'Unknown')
    else:
        run_group.attrs['gpu_used'] = False
    
    return run_group


def get_latest_run(root: zarr.Group, stage: str) -> Optional[zarr.Group]:
    """
    Get the latest run for a given stage.
    
    Args:
        root: Root zarr group
        stage: Stage name
    
    Returns:
        Latest run group or None if no runs exist
    """
    parent_group_name = f"{stage}_runs"
    
    # Navigate to the runs group
    if 'processing' in root and parent_group_name in root['processing']:
        parent_group = root['processing'][parent_group_name]
    elif parent_group_name in root:
        parent_group = root[parent_group_name]
    else:
        return None
    
    if 'latest' not in parent_group.attrs:
        return None
    
    latest_run_name = parent_group.attrs['latest']
    if not latest_run_name or latest_run_name not in parent_group:
        return None
    
    return parent_group[latest_run_name]


def update_import_duration(root: zarr.Group, duration_seconds: float) -> None:
    """
    Update the raw_video group with import duration after completion.
    
    Args:
        root: Root zarr group
        duration_seconds: Time taken for import in seconds
    """
    if 'raw_video' in root:
        root['raw_video'].attrs['duration_seconds'] = duration_seconds


def validate_zarr_structure(zarr_path: str) -> Dict[str, Any]:
    """
    Validate that a zarr store has the expected structure.
    
    Args:
        zarr_path: Path to zarr store
    
    Returns:
        Dict with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        store = LocalStore(zarr_path)
        root = zarr.open_group(store=store, mode='r', zarr_format=3)
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Could not open zarr store: {e}")
        return results
    
    # Check required groups
    required_groups = ['raw_video', 'metadata', 'processing', 'results', 'pipeline_params']
    for group_name in required_groups:
        if group_name not in root:
            results['errors'].append(f"Missing required group: {group_name}")
            results['valid'] = False
    
    # Check required attributes
    required_attrs = ['git_info', 'platform_info', 'software_versions', 'source_video_metadata']
    for attr in required_attrs:
        if attr not in root.attrs:
            results['warnings'].append(f"Missing recommended attribute: {attr}")
    
    # Check raw_video arrays
    if 'raw_video' in root:
        raw_video = root['raw_video']
        required_arrays = ['images_full', 'images_ds']
        for array_name in required_arrays:
            if array_name not in raw_video:
                results['errors'].append(f"Missing required array: raw_video/{array_name}")
                results['valid'] = False
            else:
                array = raw_video[array_name]
                results['info'][f'{array_name}_shape'] = array.shape
                results['info'][f'{array_name}_dtype'] = str(array.dtype)
    
    # Check processing run groups
    if 'processing' in root:
        processing = root['processing']
        run_groups = ['background_runs', 'detect_runs', 'tracking_runs']
        for run_group in run_groups:
            if run_group in processing:
                group = processing[run_group]
                if 'latest' in group.attrs:
                    results['info'][f'{run_group}_latest'] = group.attrs['latest']
                    
                    # Count total runs
                    num_runs = len([k for k in group.keys() if not k.startswith('.')])
                    results['info'][f'{run_group}_count'] = num_runs
    
    return results
