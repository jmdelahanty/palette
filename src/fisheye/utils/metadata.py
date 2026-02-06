"""
Simple metadata helpers for unified spec compliance.

These lightweight utilities help read metadata fields consistently across
the pipeline, following the unified metadata specification.
"""

from typing import Optional
import zarr


def get_total_frames(root: zarr.Group, detect_group: Optional[zarr.Group] = None) -> Optional[int]:
    """
    Get total frame count following unified spec.
    
    Priority:
    1. detect_group.attrs['total_frames'] (most specific)
    2. root.attrs['total_frames'] (standard location)
    3. raw_video array shapes (if has_raw_video=True)
    4. None (caller should infer from detection data)
    
    Args:
        root: Zarr root group
        detect_group: Optional detect_runs group for more specific lookup
    
    Returns:
        Total number of frames, or None if not found in metadata
    
    Example:
        >>> root = zarr.open_group('data.zarr', mode='r')
        >>> detect_group = root['detect_runs/detect_2025-10-14_10-30-00']
        >>> num_frames = get_total_frames(root, detect_group)
        >>> if num_frames is None:
        ...     num_frames = int(frame_indices.max() + 1)  # Infer from data
    """
    # Try detect group first (most specific)
    if detect_group is not None:
        n = detect_group.attrs.get('total_frames')
        if n is not None:
            return int(n)

    # Prefer imported frame counts for sampled training data
    raw = root.get("raw_video") if "raw_video" in root else None
    if raw is not None:
        import_mode = raw.attrs.get("import_mode")
        import_purpose = raw.attrs.get("import_purpose")
        if import_mode == "sampled" or import_purpose == "training_data":
            imported = raw.attrs.get("imported_frame_count")
            if imported is not None:
                return int(imported)
            if "images_ds" in raw:
                return int(raw["images_ds"].shape[0])
            if "images_full" in raw:
                return int(raw["images_full"].shape[0])
    
    # Try root attrs (standard location per unified spec)
    n = root.attrs.get('total_frames')
    if n is not None:
        return int(n)
    
    # Try raw_video arrays (for blob detection)
    if has_raw_video(root) and 'raw_video' in root:
        if 'images_ds' in root['raw_video']:
            return int(root['raw_video/images_ds'].shape[0])
        elif 'images_full' in root['raw_video']:
            return int(root['raw_video/images_full'].shape[0])
    
    # Not found - caller should infer from data
    return None


def get_detection_method(detect_group: zarr.Group) -> str:
    """
    Get detection method following unified spec.
    
    Args:
        detect_group: detect_runs group
    
    Returns:
        Detection method string: 'yolo', 'blob', 'unknown', etc.
    
    Example:
        >>> detect_group = root['detect_runs/detect_2025-10-14_10-30-00']
        >>> method = get_detection_method(detect_group)
        >>> if method == 'yolo':
        ...     print("YOLO detections - no video in zarr")
        >>> elif method == 'blob':
        ...     print("Blob detections - requires video in zarr")
    """
    return detect_group.attrs.get('detection_method', 'unknown')


def has_raw_video(root: zarr.Group) -> bool:
    """
    Check if video data is stored in zarr following unified spec.
    
    Args:
        root: Zarr root group
    
    Returns:
        True if video frames are stored in zarr, False otherwise
    
    Example:
        >>> root = zarr.open_group('data.zarr', mode='r')
        >>> if has_raw_video(root):
        ...     # Can access raw_video/images_ds or raw_video/images_full
        ...     frames = root['raw_video/images_ds'][:]
        >>> else:
        ...     # YOLO-style inference, no video stored
        ...     print("Video not stored in zarr")
    """
    # Check metadata first (canonical per unified spec)
    has_raw = root.attrs.get('has_raw_video')
    if has_raw is not None:
        return bool(has_raw)
    
    # Fallback: check for actual arrays
    if 'raw_video' not in root:
        return False
    
    return 'images_ds' in root['raw_video'] or 'images_full' in root['raw_video']


def get_video_source_path(root: zarr.Group) -> Optional[str]:
    """
    Get source video path following unified spec.
    
    Args:
        root: Zarr root group
    
    Returns:
        Path to source video file, or None if not found
    
    Example:
        >>> root = zarr.open_group('data.zarr', mode='r')
        >>> video_path = get_video_source_path(root)
        >>> if video_path and Path(video_path).exists():
        ...     # Can re-open video if needed
        ...     cap = cv2.VideoCapture(video_path)
    """
    return root.attrs.get('source_video_path')


def get_frame_source(root: zarr.Group) -> Dict[str, Optional[str]]:
    """
    Resolve the frame source for this archive.

    Returns:
        dict with:
          - type: 'zarr' | 'external' | 'unknown'
          - path: external video path if available
    """
    if has_raw_video(root):
        return {"type": "zarr", "path": None}
    source_path = get_video_source_path(root)
    if source_path:
        return {"type": "external", "path": source_path}
    return {"type": "unknown", "path": None}


def get_fps(root: zarr.Group) -> Optional[float]:
    """
    Get video frame rate following unified spec.
    
    Args:
        root: Zarr root group
    
    Returns:
        Frames per second, or None if not found
    
    Example:
        >>> root = zarr.open_group('data.zarr', mode='r')
        >>> fps = get_fps(root)
        >>> if fps:
        ...     duration = num_frames / fps
    """
    fps = root.attrs.get('fps')
    return float(fps) if fps is not None else None


def get_pipeline_type(root: zarr.Group) -> str:
    """
    Get pipeline type following unified spec.
    
    Args:
        root: Zarr root group
    
    Returns:
        Pipeline type: 'video_import', 'yolo_inference', 'blob_detection', etc.
    
    Example:
        >>> root = zarr.open_group('data.zarr', mode='r')
        >>> pipeline = get_pipeline_type(root)
        >>> if pipeline == 'yolo_inference':
        ...     print("Created via YOLO inference")
    """
    return root.attrs.get('pipeline_type', 'unknown')
