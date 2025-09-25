"""
Standardized Zarr schema for the Palette ecosystem using Zarr v3.

This schema defines the structure that all packages expect and produce.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np
import zarr
from zarr.storage import LocalStore
import zarr.codecs

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
        "experiment_type": "Type of experiment (chaser/grating/etc)",
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

def create_palette_zarr(
    path: str,
    video_metadata: Dict[str, Any],
    config: Dict[str, Any],
    use_sharding: bool = False,
    cli_args: Optional[Dict[str, Any]] = None
) -> zarr.Group:
    """
    Create a Zarr store with Palette/FishEye structure using Zarr v3.
    
    Args:
        path: Path to create the zarr store
        video_metadata: Dictionary containing video information
        config: Pipeline configuration dictionary
        use_sharding: Whether to use sharding (not yet implemented in v3)
        cli_args: Optional command line arguments to store
    
    Returns:
        Root zarr group
    """
    # Create store using LocalStore for v3
    store = LocalStore(path)
    root = zarr.open_group(store=store, mode='w')
    
    # Set root attributes
    root.attrs.update({
        'schema_version': ZARR_SCHEMA_VERSION,
        'zarr_version': zarr.__version__,
        'created_at': datetime.utcnow().isoformat(),
        'fisheye_version': '0.1.0',
        'pipeline_version': config.get('pipeline_version', '2.0-multi-fish'),
        **video_metadata
    })
    
    # Store command line args if provided
    if cli_args:
        root.attrs['command_line_args'] = cli_args
    
    # Create main groups
    raw_video = root.create_group('raw_video')
    processing = root.create_group('processing')
    analysis = root.create_group('analysis')
    metadata = root.create_group('metadata')
    
    # Create pipeline_params group to store configuration
    pipeline_params = root.create_group('pipeline_params')
    for stage_name, stage_config in config.items():
        pipeline_params.attrs[stage_name] = stage_config
    
    # Create processing subgroups and run groups
    processing.create_group('background')
    processing.create_group('detection')
    processing.create_group('tracking')
    
    # Create run groups that tracker.py expects
    processing.create_group('background_runs')
    processing.create_group('detect_runs')
    processing.create_group('crop_runs')
    processing.create_group('tracking_runs')
    processing.create_group('id_assignments_runs')
    
    # Create analysis subgroups
    analysis.create_group('metrics')
    analysis.create_group('filtered')
    analysis.create_group('interpolated')
    
    # Get dimensions
    height = video_metadata.get('height', 1080)
    width = video_metadata.get('width', 1920)
    n_frames = video_metadata.get('total_frames', 100)
    
    import_config = config.get('import', {})
    ds_height, ds_width = import_config.get('downsample_size', [540, 960])
    chunk_size = import_config.get('chunk_size', 100)
    
    # Add raw_video attributes for import stage compatibility
    raw_video.attrs.update({
        'import_timestamp_utc': datetime.utcnow().isoformat(),
        'original_resolution': (height, width),
        'downsampled_resolution': (ds_height, ds_width),
        'fps': video_metadata.get('fps', 30),
        'total_frames': n_frames,
        'source_video': video_metadata.get('source_video', 'unknown')
    })
    
    # Setup compressors for v3 (use 'compressors' for single codec)
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=1, shuffle='bitshuffle')
    
    # Create arrays using v3 API with compressors
    raw_video.create_array(
        'images_full',
        shape=(n_frames, height, width),
        chunks=(chunk_size, height, width),
        dtype=np.uint8,
        compressors=compressors,
        fill_value=0
    )
    
    raw_video.create_array(
        'images_ds',
        shape=(n_frames, ds_height, ds_width),
        chunks=(chunk_size, ds_height, ds_width),
        dtype=np.uint8,
        compressors=compressors,
        fill_value=0
    )
    
    raw_video.create_array(
        'timestamps',
        shape=(n_frames,),
        chunks=(n_frames,),
        dtype=np.float64,
        compressors=None,  # No compression for small 1D array
        fill_value=0.0
    )
    
    return root


def get_run_group(root: zarr.Group, run_name: str, stage: Optional[str] = None) -> zarr.Group:
    """
    Get or create a run group in the processing section.
    
    Args:
        root: Root zarr group
        run_name: Name of the run
        stage: Optional stage name (e.g., 'detection', 'tracking')
    
    Returns:
        Run group
    """
    processing = root['processing']
    
    # If stage is specified, get the stage group first
    if stage:
        if stage not in processing:
            raise ValueError(f"Stage '{stage}' not found in processing groups")
        parent_group = processing[stage]
    else:
        parent_group = processing
    
    # Get or create the run group
    if run_name in parent_group:
        return parent_group[run_name]
    return parent_group.create_group(run_name)


def create_detection_arrays(
    run_group: zarr.Group,
    n_frames: int,
    n_rois: int,
    chunk_size: int = 100
) -> Dict[str, zarr.Array]:
    """
    Create arrays for detection data in a run group.
    
    Args:
        run_group: The run group to create arrays in
        n_frames: Number of frames
        n_rois: Number of ROIs/detections per frame
        chunk_size: Chunk size for arrays
    
    Returns:
        Dictionary of created arrays
    """
    # Setup compressors for v3
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')
    
    arrays = {}
    
    # Create bounding boxes array
    arrays['bboxes'] = run_group.create_array(
        'bboxes',
        shape=(n_frames, n_rois, 4),
        dtype=np.float32,
        chunks=(chunk_size, n_rois, 4),
        compressors=compressors,
        fill_value=np.nan
    )
    
    # Create scores array
    arrays['scores'] = run_group.create_array(
        'scores',
        shape=(n_frames, n_rois),
        dtype=np.float32,
        chunks=(chunk_size, n_rois),
        compressors=compressors,
        fill_value=0.0
    )
    
    # Create ROI IDs array
    arrays['roi_ids'] = run_group.create_array(
        'roi_ids',
        shape=(n_frames, n_rois),
        dtype=np.int32,
        chunks=(chunk_size, n_rois),
        compressors=compressors,
        fill_value=-1
    )
    
    return arrays


def create_tracking_arrays(
    run_group: zarr.Group,
    n_frames: int,
    n_rois: int,
    n_keypoints: int = 3,
    chunk_size: int = 100
) -> Dict[str, zarr.Array]:
    """
    Create arrays for tracking data using Zarr v3 compressors.
    
    Args:
        run_group: Run group to create arrays in
        n_frames: Number of frames
        n_rois: Number of ROIs  
        n_keypoints: Number of keypoints per detection
        chunk_size: Chunk size for arrays
    
    Returns:
        Dictionary of created arrays
    """
    compressors = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle='bitshuffle')
    
    arrays = {}
    
    # Keypoint coordinates (x, y) per keypoint per ROI per frame
    arrays['keypoints'] = run_group.create_array(
        'keypoints',
        shape=(n_frames, n_rois, n_keypoints, 2),
        dtype=np.float32,
        chunks=(chunk_size, n_rois, n_keypoints, 2),
        compressors=compressors,
        fill_value=np.nan
    )
    
    # Keypoint confidence scores
    arrays['keypoint_scores'] = run_group.create_array(
        'keypoint_scores',
        shape=(n_frames, n_rois, n_keypoints),
        dtype=np.float32,
        chunks=(chunk_size, n_rois, n_keypoints),
        compressors=compressors,
        fill_value=0.0
    )
    
    # Track IDs for identity tracking
    arrays['track_ids'] = run_group.create_array(
        'track_ids',
        shape=(n_frames, n_rois),
        dtype=np.int32,
        chunks=(chunk_size, n_rois),
        compressors=compressors,
        fill_value=-1
    )
    
    return arrays


def add_calibration_data(
    root: zarr.Group,
    pixel_to_mm: float,
    arena_info: Optional[Dict[str, Any]] = None
) -> zarr.Group:
    """
    Add calibration information to the zarr store.
    
    Args:
        root: Root zarr group
        pixel_to_mm: Conversion factor from pixels to millimeters
        arena_info: Optional arena information (diameter, shape, etc.)
    
    Returns:
        Calibration group
    """
    if 'calibration' not in root:
        calib_group = root.create_group('calibration')
    else:
        calib_group = root['calibration']
    
    calib_group.attrs['pixel_to_mm'] = pixel_to_mm
    calib_group.attrs['pixels_per_mm'] = 1.0 / pixel_to_mm
    calib_group.attrs['calibrated_at'] = datetime.utcnow().isoformat()
    
    if arena_info:
        calib_group.attrs.update(arena_info)
    
    return calib_group


def validate_zarr_structure(path: str) -> Dict[str, Any]:
    """
    Validate that a zarr store follows the expected schema.
    
    Args:
        path: Path to zarr store
    
    Returns:
        Validation report dictionary
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'schema_version': None,
        'zarr_version': None
    }
    
    try:
        # Open the store in read mode using LocalStore for v3
        store = LocalStore(path)
        root = zarr.open_group(store=store, mode='r')
        
        # Check schema version
        if 'schema_version' not in root.attrs:
            report['errors'].append('Missing schema_version in root attributes')
            report['valid'] = False
        else:
            report['schema_version'] = root.attrs['schema_version']
        
        # Check zarr version
        if 'zarr_version' in root.attrs:
            report['zarr_version'] = root.attrs['zarr_version']
        
        # Check required groups
        required_groups = ['raw_video', 'processing', 'analysis', 'metadata']
        for group_name in required_groups:
            if group_name not in root:
                report['errors'].append(f'Missing required group: {group_name}')
                report['valid'] = False
        
        # Check raw_video arrays
        if 'raw_video' in root:
            expected_arrays = ['images_full', 'images_ds', 'timestamps']
            for array_name in expected_arrays:
                if array_name not in root['raw_video']:
                    report['warnings'].append(f'Missing expected array: raw_video/{array_name}')
        
        # Check processing subgroups
        if 'processing' in root:
            expected_subgroups = ['background', 'detection', 'tracking']
            for subgroup_name in expected_subgroups:
                if subgroup_name not in root['processing']:
                    report['warnings'].append(f'Missing expected processing subgroup: {subgroup_name}')
        
        # Check analysis subgroups
        if 'analysis' in root:
            expected_subgroups = ['metrics', 'filtered', 'interpolated']
            for subgroup_name in expected_subgroups:
                if subgroup_name not in root['analysis']:
                    report['warnings'].append(f'Missing expected analysis subgroup: {subgroup_name}')
                    
    except Exception as e:
        report['errors'].append(f'Failed to open zarr store: {str(e)}')
        report['valid'] = False
    
    return report


def add_processing_run(
    root: zarr.Group,
    stage: str,
    run_name: str,
    parameters: Dict[str, Any]
) -> zarr.Group:
    """
    Add a new processing run to the zarr store.
    
    Args:
        root: Root zarr group
        stage: Stage name ('detection', 'tracking', etc.)
        run_name: Name for this run
        parameters: Parameters used for this run
    
    Returns:
        The created run group
    """
    run_group = get_run_group(root, run_name, stage)
    
    # Add run metadata
    run_group.attrs.update({
        'created_at': datetime.utcnow().isoformat(),
        'stage': stage,
        'parameters': parameters
    })
    
    return run_group


def get_latest_run(root: zarr.Group, stage: str) -> Optional[zarr.Group]:
    """
    Get the most recent run for a given stage.
    
    Args:
        root: Root zarr group
        stage: Stage name
    
    Returns:
        Latest run group or None if no runs exist
    """
    if 'processing' not in root or stage not in root['processing']:
        return None
    
    stage_group = root['processing'][stage]
    
    # Get all run groups with timestamps
    runs_with_time = []
    for run_name in stage_group.group_keys():
        run_group = stage_group[run_name]
        if 'created_at' in run_group.attrs:
            runs_with_time.append((run_name, run_group.attrs['created_at']))
    
    if not runs_with_time:
        return None
    
    # Sort by timestamp and return the latest
    runs_with_time.sort(key=lambda x: x[1])
    latest_run_name = runs_with_time[-1][0]
    
    return stage_group[latest_run_name]