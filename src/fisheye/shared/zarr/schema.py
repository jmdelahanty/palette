"""
Standardized Zarr schema for the Palette ecosystem using Zarr v3.

Legacy schema helpers for broad archive structure and metadata.

Per-stage array contracts are maintained in `fisheye.shared.zarr.stage_arrays`.
"""

from typing import Dict, Any, Optional, Tuple, List
import warnings
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
    get_environment_summary,
    get_environment_info
)

ZARR_SCHEMA_VERSION = "3.0.0"

ZARR_SCHEMA = {
    "version": ZARR_SCHEMA_VERSION,
    "zarr_format": 3,
    "status": "legacy-metadata",
    "array_contract_source": "fisheye.shared.zarr.stage_arrays",
    "root_attributes": {
        "schema_version": "Schema version string (3.x)",
        "zarr_format": "Zarr format version (3)",
        "created_at": "ISO timestamp of creation",
        "pipeline_version": "Version of fisheye used",
        "command_line_args": "CLI arguments used to create the archive (optional)",
        "git_info": "Git commit/branch/dirty state",
        "platform_info": "Host/system metadata",
        "software_versions": "Key package versions",
        "environment": "Condensed runtime environment info",
        "source_video_metadata": "Source video metadata dict (fps, width, height, etc.)",
        "source_video_path": "Absolute path to source video (optional)",
        "fps": "Frames per second",
        "total_frames": "Total number of frames",
        "width": "Video width in pixels",
        "height": "Video height in pixels",
        "has_raw_video": "True if raw_video frames are stored in the archive",
        "processing_history": "List of processing steps applied (optional)",
    },
    "groups": {
        "raw_video": {
            "description": "Imported video data + import metadata",
            "arrays": {
                "images_full": "Full resolution frames (optional)",
                "images_ds": "Downsampled frames (optional)",
                "images_ds_rgb": "Downsampled RGB frames (optional)",
                "timestamps": "Frame timestamps (optional)",
                "original_frame_indices": "Full-video indices for sampled imports (optional)",
            },
        },
        "pipeline_params": {"description": "Config snapshot per pipeline stage"},
        "background_runs": {"description": "Background subtraction runs (latest attr)"},
        "detect_runs": {"description": "Detection runs (latest attr)"},
        "refined_detect_runs": {
            "description": "Refined detect runs with canonical sparse curated instances and source_detections",
            "parent_attributes": {
                "latest": "Latest refined detect run name",
                "detect_review_status_latest": "Refined run name with the latest review status (optional)",
            },
            "run_attributes": {
                "source_detect_run": "Upstream detect run name",
                "source_quality_run": "Upstream detect quality run name (optional)",
                "refinement_timestamp": "ISO timestamp when refinement ran",
                "operations": "List of refinement operations (e.g., ['filter'] or ['passthrough'])",
                "parameters": "Refine parameters (filters, refine_mode, sampled_import, interpolation_enabled=False, etc.)",
                "coverage_comparison": "Coverage stats for original/refined",
                "coverage_frames_total": "Frame universe used for coverage percent",
                "coverage_frame_source": "Coverage frame source (full or sampled)",
                "coverage_frames_full": "Full frame count when sampled coverage is used",
                "manual_review_latest": "Legacy manual/retune subgroup label for sparse archives (optional)",
                "detect_review_status": "Review status payload (state/method/intended_use/resolved_group/etc.)",
                "retune_params": "Mapping of retune_id → parameter sets",
                "curated_row_storage": "Canonical refined detect storage mode",
                "entity_assignment_policy": "How per-frame local instance ids are assigned for sparse views",
                "coordinate_space": "Canonical bbox coordinate space for refined instances",
                "row_identity_policy": "Policy used for refined_row_ids on sparse instances",
                "status_code_map": "Mapping for detect status labels used during slot-based editing/materialization",
                "source_kind_code_map": "Mapping for instance/source provenance labels",
                "review_state_code_map": "Mapping for detect review state labels stored in attrs",
                "summary_statistics": "Summary statistics for the canonical sparse refined surfaces",
                "curation_provenance": "Provenance payload for the canonical sparse refined write",
                "curation_updated_at_utc": "Last timestamp when sparse refined surfaces were rewritten",
            },
        },
        "crop_runs": {
            "description": "ROI crop runs (latest attr)",
            "parent_attributes": {
                "crop_review_status_latest": "Crop run name with the latest review status (optional)",
            },
            "run_attributes": {
                "detection_source_type": "detect/filtered/interpolated/manual/refined/auto (resolved)",
                "detection_source_path": "Zarr path to the detection source used",
                "detection_selection_policy": "Policy label used when resolving auto source selection",
                "detect_review_status": "Snapshot of refined detect review status (optional)",
                "detect_review_status_ref": "Reference path to refined run holding review status (optional)",
                "crop_signature": "Signature of crop inputs (source path/type, roi size, parameters hash)",
                "crop_review_status": "Review status payload for crop run (optional)",
                "crop_review_signature": "Signature snapshot stored when crop review was set (optional)",
                "source_detect_run": "Upstream detect run name (when available)",
                "source_refined_run": "Upstream refined detect run name (when available)",
                "source_refined_row_ids_available": "Whether crop rows carry source_refined_row_ids lineage",
                "source_refined_row_id_policy": "Policy for copied refined detection row IDs",
                "source_detect_row_index_available": "Whether crop rows carry raw detect row lineage from refined instances",
            },
        },
        "keypoints_runs": {"description": "Keypoint detection runs (latest attr)"},
        "refined_keypoints_runs": {
            "description": "Refined keypoint runs (latest attr)",
            "parent_attributes": {
                "keypoint_review_status_latest": "Refined run name with the latest keypoint review status (optional)",
            },
            "run_attributes": {
                "keypoint_signature": "Signature of keypoint inputs (source runs, parameters hash)",
                "keypoint_review_status": "Review status payload for refined keypoints (optional)",
                "keypoint_review_signature": "Signature snapshot stored when keypoint review was set (optional)",
            },
        },
        "eye_masks_runs": {"description": "Eye mask segmentation runs (latest attr)"},
        "refined_eye_masks_runs": {"description": "Refined eye mask runs (latest attr)"},
        "subject_mask_runs": {"description": "Generalized subject-mask runs (latest attr)"},
        "refined_subject_masks_runs": {"description": "Refined subject-mask runs (latest attr)"},
        "refined_online_runs": {"description": "Online refined detection runs (latest attr)"},
        "tracking_runs": {"description": "Tracking runs (latest attr)"},
        "arena_assignment_runs": {"description": "Arena assignment runs (latest attr)"},
        "analysis": {"description": "Analysis outputs"},
        "analysis_metadata": {"description": "Calibration/tuning metadata"},
        "calibration": {"description": "Calibration data"},
    },
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
    root.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
    root.attrs['fps'] = video_metadata.get('fps')
    root.attrs['width'] = int(video_metadata.get('width', 0))
    root.attrs['height'] = int(video_metadata.get('height', 0))
    root.attrs['total_frames'] = int(video_metadata.get('total_frames', 0))
    root.attrs['has_raw_video'] = True
    if cli_args and cli_args.get("training_data"):
        root.attrs["zarr_purpose"] = "training"
    else:
        root.attrs["zarr_purpose"] = "analysis"
    source_path = video_metadata.get('source_path') or video_metadata.get('source_video_path')
    if source_path:
        root.attrs['source_video_path'] = str(source_path)

    # ---- Pipeline params ----------------------------------------------------
    param_group = root.create_group('pipeline_params')
    for stage, stage_params in config.items():
        param_group.attrs[stage] = stage_params

    # ---- Groups -------------------------------------------------------------
    raw_video = root.create_group('raw_video')

    height  = int(video_metadata.get('height', 1080))
    width   = int(video_metadata.get('width', 1920))
    n_frames = int(video_metadata.get('total_frames', 1))

    # Import config
    import_config = config.get('import', {}) if isinstance(config, dict) else {}
    down_cfg = import_config.get('downsampled', {}) if isinstance(import_config, dict) else {}
    ds_size = down_cfg.get('size') or import_config.get('downsample_size') or [640, 640]
    ds_height, ds_width = [int(x) for x in ds_size]

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
    use_sharding = bool(import_config.get('use_sharding', use_sharding))
    chunks_per_shard = int(import_config.get('chunks_per_shard', 1))
    full_chunk = int(import_config.get('chunk_size', 1))
    ds_chunk = int(down_cfg.get('chunk_size', full_chunk))
    if use_sharding:
        shard_size_full = int(import_config.get('shard_size', max(1, full_chunk * chunks_per_shard)))
        shard_size_ds = int(import_config.get('shard_size_ds', max(1, ds_chunk * chunks_per_shard)))
    else:
        shard_size_full = None
        shard_size_ds = None

    # ---- serializer (CRC enabled; set checksum=False to disable) ------------
    ser = zarr.codecs.BytesCodec()  # or BytesCodec(checksum=False) while debugging

    # Record sharding choices
    if use_sharding and shard_size_full and shard_size_ds:
        raw_video.attrs['sharding'] = {
            'images_full': {'frames_per_shard': shard_size_full, 'shard_shape': (shard_size_full, height, width)},
            'images_ds': {'frames_per_shard': shard_size_ds, 'shard_shape': (shard_size_ds, ds_height, ds_width)},
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
    full_kwargs = {}
    ds_kwargs = {}
    if use_sharding and shard_size_full:
        full_kwargs["shards"] = (shard_size_full, height, width)
    if use_sharding and shard_size_ds:
        ds_kwargs["shards"] = (shard_size_ds, ds_height, ds_width)

    raw_video.create_array(
        name='images_full',
        shape=(n_frames, height, width),
        chunks=(max(1, full_chunk), height, width),
        dtype=np.uint8,
        fill_value=0,
        serializer=ser,
        compressors=compressors_full,
        **full_kwargs,
    )
    raw_video.create_array(
        name='images_ds',
        shape=(n_frames, ds_height, ds_width),
        chunks=(max(1, ds_chunk), ds_height, ds_width),
        dtype=np.uint8,
        fill_value=0,
        serializer=ser,
        compressors=compressors_ds,
        **ds_kwargs,
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

    # Run groups at root
    run_groups = (
        'background_runs',
        'detect_runs',
        'refined_detect_runs',
        'crop_runs',
        'keypoints_runs',
        'refined_keypoints_runs',
        'eye_masks_runs',
        'refined_eye_masks_runs',
        'subject_mask_runs',
        'refined_subject_masks_runs',
        'refined_online_runs',
        'tracking_runs',
        'arena_assignment_runs',
    )
    for group_name in run_groups:
        group = root.require_group(group_name)
        if 'latest' not in group.attrs:
            group.attrs['latest'] = None

    # Non-run groups
    root.require_group('analysis')
    root.require_group('analysis_metadata')
    root.require_group('calibration')
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
    warnings.warn(
        "create_detection_arrays() is legacy and incomplete for modern detect runs; "
        "use stage-specific writers and fisheye.shared.zarr.stage_arrays contracts instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    warnings.warn(
        "create_tracking_arrays() is legacy and uses a historical 21-column layout; "
        "prefer current stage-specific writers and contracts in fisheye.shared.zarr.stage_arrays.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    extra_attrs: Optional[Dict[str, Any]] = None,
    env_info: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    run_name: Optional[str] = None,
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
    if env_info is None:
        env_info = get_environment_info()

    parent_group_name = f"{stage}_runs"
    parent_group = root.require_group(parent_group_name)

    if run_name:
        if run_name in parent_group:
            raise ValueError(f"{parent_group_name}/{run_name} already exists")
        run_group = parent_group.create_group(run_name)
        parent_group.attrs["latest"] = run_name
        if console:
            console.print(f"Created run group: [cyan]{parent_group_name}/{run_name}[/cyan]")
    else:
        run_group, run_name = get_run_group(root, stage, create_new=True, console=console)
    
    # Add standard metadata
    run_group.attrs.update({
        f'{stage}_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': parameters,
        'run_name': run_name,
        'run_stage': stage,
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
    platform_info = env_info.get('platform', {}) if isinstance(env_info, dict) else {}
    run_group.attrs['processing_host'] = platform_info.get('hostname', platform.node())
    run_group.attrs['processing_platform'] = platform_info

    git_info = env_info.get('git', {}) if isinstance(env_info, dict) else {}
    run_group.attrs['git_commit'] = git_info.get('commit_hash', 'unknown')
    run_group.attrs['git_branch'] = git_info.get('branch', 'unknown')

    gpu_info = env_info.get('gpu', {}) if isinstance(env_info, dict) else {}
    run_group.attrs['gpu_used'] = bool(gpu_info.get('available'))
    if gpu_info.get('devices'):
        primary_gpu = gpu_info['devices'][0]
        run_group.attrs['gpu_device'] = primary_gpu.get('name', 'Unknown')
        run_group.attrs['gpu_compute_capability'] = primary_gpu.get('compute_capability')
        run_group.attrs['gpu_total_memory_gb'] = primary_gpu.get('total_memory_gb')
    else:
        run_group.attrs['gpu_device'] = 'None'
    run_group.attrs['environment'] = env_info.get('environment', {})

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
    required_groups = ['pipeline_params']
    for group_name in required_groups:
        if group_name not in root:
            results['errors'].append(f"Missing required group: {group_name}")
            results['valid'] = False

    # Check recommended groups
    recommended_groups = [
        'raw_video',
        'background_runs',
        'detect_runs',
        'refined_detect_runs',
        'crop_runs',
        'keypoints_runs',
        'refined_keypoints_runs',
        'eye_masks_runs',
        'refined_eye_masks_runs',
        'tracking_runs',
        'arena_assignment_runs',
        'analysis_metadata',
    ]
    for group_name in recommended_groups:
        if group_name not in root:
            results['warnings'].append(f"Missing recommended group: {group_name}")

    # Check required attributes
    required_attrs = [
        'schema_version',
        'zarr_format',
        'pipeline_version',
        'source_video_metadata',
        'fps',
        'width',
        'height',
        'total_frames',
    ]
    for attr in required_attrs:
        if attr not in root.attrs:
            results['warnings'].append(f"Missing recommended attribute: {attr}")

    # Check raw_video arrays if raw video is present
    has_raw = root.attrs.get('has_raw_video', True)
    if has_raw and 'raw_video' in root:
        raw_video = root['raw_video']
        if 'images_full' not in raw_video and 'images_ds' not in raw_video:
            results['errors'].append("Missing raw_video arrays (images_full/images_ds)")
            results['valid'] = False
        else:
            for array_name in ('images_full', 'images_ds'):
                if array_name in raw_video:
                    array = raw_video[array_name]
                    results['info'][f'{array_name}_shape'] = array.shape
                    results['info'][f'{array_name}_dtype'] = str(array.dtype)

    # Check run groups at root
    run_groups = [
        'background_runs',
        'detect_runs',
        'refined_detect_runs',
        'crop_runs',
        'keypoints_runs',
        'refined_keypoints_runs',
        'eye_masks_runs',
        'refined_eye_masks_runs',
        'tracking_runs',
        'arena_assignment_runs',
    ]
    for run_group in run_groups:
        if run_group in root:
            group = root[run_group]
            if 'latest' in group.attrs:
                results['info'][f'{run_group}_latest'] = group.attrs['latest']
                num_runs = len([k for k in group.keys() if not k.startswith('.')])
                results['info'][f'{run_group}_count'] = num_runs
    
    return results
