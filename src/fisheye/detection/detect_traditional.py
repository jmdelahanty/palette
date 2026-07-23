"""
Fish detection module using blob detection on background-subtracted frames.
Updated to use zarr-first parameter resolution.
"""

import argparse
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import time
import sys
import numpy as np
import zarr
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

import skimage
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.detection_producer_lifecycle import (
    DETECTION_ARTIFACT_RUN_FAMILY,
    DetectionProducerAttempt,
    UNBOUND_ARTIFACT_RUN_BINDING_KEY,
    build_unbound_artifact_run_binding,
    publish_artifact_payload_inventory_seal,
    publish_empty_artifact_observation_proof,
    stamp_unbound_artifact_numeric_semantics,
)
from ..shared.zarr.chunk_profiles import create_geometry_preload_array
from ..shared.zarr_run_completion import (
    resolve_authoritative_run_name,
)
from ..shared.system_metadata import get_git_info, get_platform_info


_ACTIVE_DETECTION_ATTEMPT: ContextVar[DetectionProducerAttempt | None] = (
    ContextVar("traditional_detection_attempt", default=None)
)


def _with_detection_attempt_rollback(function):
    """Restore exact selector state if a bound producer attempt raises."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _ACTIVE_DETECTION_ATTEMPT.set(None)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            attempt = _ACTIVE_DETECTION_ATTEMPT.get()
            if attempt is not None:
                attempt.fail(exc)
            raise
        finally:
            _ACTIVE_DETECTION_ATTEMPT.reset(token)

    return wrapped

_ARRAY_FINGERPRINT_SCHEMA = "palette.ndarray_axis0_sha256.v1"
_TRADITIONAL_SOURCE_LINEAGE_ATTR = "artifact_source_lineage"
_TRADITIONAL_SOURCE_LINEAGE_SCHEMA = (
    "palette.traditional_detection_artifact_source_lineage.v1"
)


def _canonical_mapping_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stamp_traditional_artifact_semantics(
    run: Any,
    *,
    reference_width: int,
    reference_height: int,
    source_frame_count: int,
    source_lineage_sha256: str,
) -> None:
    profiles = {
        "artifact_row_id": "traditional.artifact_row_id.v1",
        "frame_indices": "traditional.frame_indices.v1",
        "bbox_norm_coords": "traditional.bbox_norm_cxcywh.v1",
        "scores": "traditional.scores.v1",
        "class_ids": "traditional.class_ids.v1",
        "frame_counts": "traditional.frame_counts.v1",
        "n_detections": "traditional.n_detections.v1",
    }
    if set(run.keys()) != set(profiles):
        raise ValueError(
            "Traditional artifact semantic inventory does not match live arrays."
        )
    for name, profile_id in profiles.items():
        stamp_unbound_artifact_numeric_semantics(
            run[name],
            semantic_profile_id=profile_id,
            reference_node_path="raw_video/images_ds",
            reference_width=reference_width,
            reference_height=reference_height,
            source_frame_count=source_frame_count,
            source_sha256=source_lineage_sha256,
        )


def _require_imported_detection_inputs(
    root: zarr.Group,
    latest_bg_run: str,
) -> tuple[Any, Any, tuple[int, int]]:
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
    images_ds = raw_video["images_ds"]
    background_ds = background_parent[latest_bg_run]["background_ds"]
    images_shape = tuple(int(value) for value in images_ds.shape)
    background_shape = tuple(int(value) for value in background_ds.shape)
    if len(images_shape) != 3 or len(background_shape) != 2:
        raise ValueError(
            "Traditional detection requires images_ds shaped (frames, height, width) "
            "and background_ds shaped (height, width)."
        )
    if images_shape[0] <= 0:
        raise ValueError(
            "Traditional detection requires raw_video/images_ds to contain at "
            "least one source frame."
        )
    image_hw = images_shape[-2:]
    if background_shape != image_hw:
        raise ValueError(
            "Traditional detection requires exact images_ds/background_ds H/W "
            f"equality; images_ds={image_hw}, background_ds={background_shape}."
        )
    return images_ds, background_ds, image_hw


def _array_content_fingerprint(array: Any) -> str:
    """Hash one stable array payload without materializing the full video."""

    before_shape = tuple(int(value) for value in array.shape)
    before_dtype = np.dtype(array.dtype)
    if before_dtype.hasobject:
        raise ValueError("Traditional source arrays cannot use object dtype.")
    header = {
        "schema_id": _ARRAY_FINGERPRINT_SCHEMA,
        "dtype": before_dtype.str,
        "shape": list(before_shape),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    digest.update(b"\x00")
    if before_shape:
        raw_chunks = getattr(array, "chunks", None)
        first_chunk = (
            int(raw_chunks[0])
            if isinstance(raw_chunks, (tuple, list)) and raw_chunks
            else 1
        )
        step = max(1, first_chunk)
        for start in range(0, before_shape[0], step):
            stop = min(start + step, before_shape[0])
            values = np.asarray(array[start:stop])
            expected_shape = (stop - start, *before_shape[1:])
            if values.shape != expected_shape or values.dtype != before_dtype:
                raise ValueError(
                    "Traditional source array changed shape or dtype during fingerprinting."
                )
            digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    else:
        values = np.asarray(array[...])
        if values.shape != () or values.dtype != before_dtype:
            raise ValueError(
                "Traditional scalar source changed during fingerprinting."
            )
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    if (
        tuple(int(value) for value in array.shape) != before_shape
        or np.dtype(array.dtype) != before_dtype
    ):
        raise ValueError(
            "Traditional source array metadata changed during fingerprinting."
        )
    return digest.hexdigest()


def _require_array_fingerprint(
    array: Any,
    *,
    expected: str,
    label: str,
) -> None:
    if _array_content_fingerprint(array) != expected:
        raise ValueError(
            f"{label} changed while traditional detection was running; refusing "
            "to persist lineage for unstable pixels."
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
                        console.print("[green]✓ Using rectangular dish mask from zarr[/green]")
            elif shape == 'circle' and 'detected_circle' in mask_data:
                circle = mask_data['detected_circle']
                detect_params['dish_mask'].update({
                    'shape': 'circle',
                    'center': circle['center'],
                    'radius': circle['radius'],
                })
                if console:
                    console.print("[green]✓ Using circular dish mask from zarr[/green]")
    
    # No tuned parameters found - using config
    if param_source == 'config_default' and console:
        console.print("[yellow]⚠️  Using default config parameters - consider tuning first[/yellow]")
        console.print("  Run: python -m fisheye --tune mask")
        console.print("  Run: python -m fisheye --tune detect")
    
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
    images_ds, background_node, _image_hw = _require_imported_detection_inputs(
        root,
        latest_bg_run,
    )
    images_ds_chunk = images_ds[chunk_slice]
    background_ds = background_node[:]
    ds_img_shape = images_ds_chunk.shape[1:]

    chunk_len = images_ds_chunk.shape[0]
    frame_start = chunk_slice.start
    
    all_bbox_norms = []
    all_frame_indices = []

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


def _require_exact_detection_chunk_plan(
    chunk_slices: list[slice],
    *,
    total_frames: int,
) -> None:
    """Require one ordered, nonoverlapping plan covering the full source domain."""

    frame_count = int(total_frames)
    if frame_count <= 0:
        raise ValueError("Traditional detection requires a positive source frame domain.")
    if frame_count - 1 > np.iinfo(np.int32).max:
        raise ValueError(
            "Traditional detection frame_indices cannot represent this source domain "
            "exactly as int32."
        )
    if not chunk_slices:
        raise ValueError("Traditional detection chunk plan is empty.")
    cursor = 0
    for index, chunk_slice in enumerate(chunk_slices):
        if type(chunk_slice) is not slice or chunk_slice.step not in (None, 1):
            raise ValueError(
                f"Traditional detection chunk {index} is not a unit-step slice."
            )
        if type(chunk_slice.start) is not int or type(chunk_slice.stop) is not int:
            raise ValueError(
                f"Traditional detection chunk {index} lacks exact integer bounds."
            )
        if chunk_slice.start != cursor or not (
            chunk_slice.start < chunk_slice.stop <= frame_count
        ):
            raise ValueError(
                "Traditional detection chunks must be ordered, nonoverlapping, and "
                "exactly contiguous over the source frame domain."
            )
        cursor = chunk_slice.stop
    if cursor != frame_count:
        raise ValueError(
            "Traditional detection chunks do not cover the exact source frame domain."
        )


def _validate_detection_chunk_result(
    result: Any,
    *,
    expected_slice: slice,
    total_frames: int,
    result_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one worker result before it can contribute rows or identity."""

    if type(result) is not tuple or len(result) != 3:
        raise ValueError(
            f"Traditional detection worker result {result_index} must be one "
            "three-item tuple."
        )
    returned_slice, raw_frames, raw_bboxes = result
    if (
        type(returned_slice) is not slice
        or returned_slice.start != expected_slice.start
        or returned_slice.stop != expected_slice.stop
        or returned_slice.step not in (None, 1)
    ):
        raise ValueError(
            f"Traditional detection worker result {result_index} returned the "
            "wrong source slice."
        )

    frames_raw = np.asarray(raw_frames)
    if frames_raw.ndim != 1:
        raise ValueError(
            f"Traditional detection worker result {result_index} frame indices "
            "must be rank 1."
        )
    if frames_raw.size and frames_raw.dtype.kind not in "iu":
        raise ValueError(
            f"Traditional detection worker result {result_index} frame indices "
            "must be exact integers."
        )
    frames = frames_raw.astype(np.int64, copy=False)

    bboxes_raw = np.asarray(raw_bboxes)
    if frames.size == 0 and bboxes_raw.size == 0:
        bboxes = np.empty((0, 4), dtype=np.float64)
    else:
        if (
            bboxes_raw.ndim != 2
            or bboxes_raw.shape[1] != 4
            or bboxes_raw.dtype.kind not in "fiu"
        ):
            raise ValueError(
                f"Traditional detection worker result {result_index} bboxes must "
                "have exact numeric shape (N, 4)."
            )
        bboxes = bboxes_raw.astype(np.float64, copy=False)
    if bboxes.shape[0] != frames.shape[0]:
        raise ValueError(
            f"Traditional detection worker result {result_index} bbox/frame "
            "cardinality does not agree."
        )
    if frames.size:
        if (
            np.any(frames < int(expected_slice.start))
            or np.any(frames >= int(expected_slice.stop))
            or np.any(frames < 0)
            or np.any(frames >= int(total_frames))
        ):
            raise ValueError(
                f"Traditional detection worker result {result_index} contains a "
                "frame outside its exact source slice/domain."
            )
        if frames.size > 1 and np.any(np.diff(frames) < 0):
            raise ValueError(
                f"Traditional detection worker result {result_index} frame rows "
                "are not ordered by source frame."
            )
    if bboxes.size:
        if not np.isfinite(bboxes).all():
            raise ValueError(
                f"Traditional detection worker result {result_index} contains "
                "non-finite normalized bbox values."
            )
        cx, cy, width, height = bboxes.T
        tolerance = 1e-12
        invalid = (
            (width <= 0.0)
            | (height <= 0.0)
            | (width > 1.0 + tolerance)
            | (height > 1.0 + tolerance)
            | (cx < 0.0)
            | (cx > 1.0)
            | (cy < 0.0)
            | (cy > 1.0)
            | (cx - width * 0.5 < -tolerance)
            | (cx + width * 0.5 > 1.0 + tolerance)
            | (cy - height * 0.5 < -tolerance)
            | (cy + height * 0.5 > 1.0 + tolerance)
        )
        if np.any(invalid):
            raise ValueError(
                f"Traditional detection worker result {result_index} contains an "
                "invalid normalized bbox extent."
            )
    return frames, bboxes


def _validate_and_aggregate_detection_results(
    results: Any,
    *,
    chunk_slices: list[slice],
    total_frames: int,
    console: Console,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact worker coverage, then concatenate trusted numeric rows."""

    _require_exact_detection_chunk_plan(
        chunk_slices,
        total_frames=total_frames,
    )
    if type(results) is not tuple or len(results) != len(chunk_slices):
        raise ValueError(
            "Traditional detection requires exactly one worker result for every "
            "requested source slice."
        )
    frame_chunks: list[np.ndarray] = []
    bbox_chunks: list[np.ndarray] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        aggregate_task = progress.add_task(
            "[cyan]Validating detections...", total=len(results)
        )
        for result_index, (expected_slice, result) in enumerate(
            zip(chunk_slices, results, strict=True)
        ):
            frames, bboxes = _validate_detection_chunk_result(
                result,
                expected_slice=expected_slice,
                total_frames=total_frames,
                result_index=result_index,
            )
            frame_chunks.append(frames)
            bbox_chunks.append(bboxes)
            progress.advance(aggregate_task)
    if not frame_chunks:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0, 4), dtype=np.float64),
        )
    frames = np.concatenate(frame_chunks).astype(np.int32, copy=False)
    bboxes = np.concatenate(bbox_chunks).astype(np.float64, copy=False)
    return frames, bboxes


@_with_detection_attempt_rollback
def detect_fish(
    zarr_path: str,
    config_path: Optional[str] = "pipeline_config.yaml",
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    show_progress: bool = True,
    artifact_only: bool = False,
) -> Dict[str, Any]:
    """Detect fish in video frames using blob detection."""
    if not artifact_only:
        raise ValueError(
            "Traditional detection cannot publish a selectable canonical detect run: "
            "raw_video/images_ds lacks exact acquisition transform authority. "
            "Pass artifact_only=True (CLI: --artifact-only) to write an explicit "
            "nonselector detection_artifact_runs output."
        )
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
    
    latest_bg_run = resolve_authoritative_run_name(root["background_runs"])
    if latest_bg_run is None:
        raise ValueError("Traditional detection requires a complete background run.")
    images_ds, background_node, ds_img_shape = _require_imported_detection_inputs(
        root,
        latest_bg_run,
    )
    images_ds_sha256 = _array_content_fingerprint(images_ds)
    background_ds_sha256 = _array_content_fingerprint(background_node)
    console.print(f"Using background: [cyan]{latest_bg_run}[/cyan]")
    
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"detect_{timestamp}"
    
    # Create dish mask
    mask_params = detect_params.get('dish_mask', {})
    mask = create_dish_mask(mask_params, ds_img_shape, console)
    
    # Get video info
    num_images = images_ds.shape[0]
    chunk_size = images_ds.chunks[0]
    
    console.print(f"Processing {num_images} frames in chunks of {chunk_size}")
    
    # Create dask tasks
    chunk_slices = [slice(i, min(i + chunk_size, num_images)) for i in range(0, num_images, chunk_size)]
    _require_exact_detection_chunk_plan(
        chunk_slices,
        total_frames=num_images,
    )
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for detection...")
    
    delayed_tasks = [detect_chunk_delayed(zarr_path, s, detect_params, mask, latest_bg_run) for s in chunk_slices]
    
    if show_progress:
        with ProgressBar():
            results = dask.compute(*delayed_tasks)
    else:
        results = dask.compute(*delayed_tasks)
    
    console.print("Writing detection results to Zarr...")
    
    frame_indices_np, bboxes_np = _validate_and_aggregate_detection_results(
        results,
        chunk_slices=chunk_slices,
        total_frames=num_images,
        console=console,
    )
    total_detections = int(frame_indices_np.shape[0])
    console.print(f"Aggregating {total_detections} total detections...")
    _require_array_fingerprint(
        images_ds,
        expected=images_ds_sha256,
        label="raw_video/images_ds",
    )
    _require_array_fingerprint(
        background_node,
        expected=background_ds_sha256,
        label=f"background_runs/{latest_bg_run}/background_ds",
    )
    source_lineage = {
        "schema_id": _TRADITIONAL_SOURCE_LINEAGE_SCHEMA,
        "status": "unbound_artifact_provenance_only",
        "frame_source": {
            "node_path": "raw_video/images_ds",
            "content_sha256": images_ds_sha256,
            "content_sha256_kind": _ARRAY_FINGERPRINT_SCHEMA,
            "shape": [int(value) for value in images_ds.shape],
            "dtype": np.dtype(images_ds.dtype).str,
        },
        "background_source": {
            "node_path": f"background_runs/{latest_bg_run}/background_ds",
            "content_sha256": background_ds_sha256,
            "content_sha256_kind": _ARRAY_FINGERPRINT_SCHEMA,
            "shape": [int(value) for value in background_node.shape],
            "dtype": np.dtype(background_node.dtype).str,
        },
        "algorithm": "traditional_blob_detection_unchanged_v1",
        "source_camera_binding_status": "unbound",
        UNBOUND_ARTIFACT_RUN_BINDING_KEY: build_unbound_artifact_run_binding(
            manifest_id="traditional_detection.v1",
            reference_node_path="raw_video/images_ds",
            reference_width=int(ds_img_shape[1]),
            reference_height=int(ds_img_shape[0]),
            source_frame_count=int(num_images),
        ),
    }
    source_lineage_sha256 = _canonical_mapping_sha256(source_lineage)

    # ``images_ds`` does not yet carry one array-bound transform into the
    # published acquisition camera authority.  Keep the scientifically useful
    # blob output as an explicit nonselector artifact; do not let normalized
    # downsample coordinates masquerade as a normal source-camera detect run.
    attempt = DetectionProducerAttempt.begin_unbound_artifact(
        root,
        run_name=run_name,
        semantic_manifest_id="traditional_detection.v1",
        strict_integrity_required=True,
    )
    _ACTIVE_DETECTION_ATTEMPT.set(attempt)
    detect_group = attempt.run
    console.print(
        f"Created nonselector artifact: [cyan]{DETECTION_ARTIFACT_RUN_FAMILY}/{run_name}[/cyan]"
    )
    
    artifact_row_id = np.arange(total_detections, dtype=np.uint64)
    artifact_row_chunk = max(1, min(max(1, total_detections), 16384))
    detect_group.create_array(
        "artifact_row_id",
        data=artifact_row_id,
        chunks=(artifact_row_chunk,),
        overwrite=True,
    )

    # Write core detection arrays
    if total_detections > 0:
        class_ids_np = np.zeros(total_detections, dtype=np.int32)
        
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
            data=class_ids_np,
            chunks=(det_chunk,),
            overwrite=True
        )
        # Compute and store frame counts for visualization
        console.print("Computing frame counts for visualization...")
        frame_counts = np.bincount(
            frame_indices_np,
            minlength=num_images,
        ).astype(np.int32)
        create_geometry_preload_array(
            detect_group,
            'frame_counts',
            data=frame_counts,
            overwrite=True
        )
        create_geometry_preload_array(
            detect_group,
            'n_detections',
            data=frame_counts,
            overwrite=True
        )
        
    else:
        # No detections found
        detect_group.create_array('frame_indices', data=np.empty((0,), dtype='i4'), overwrite=True)
        detect_group.create_array(
            'bbox_norm_coords',
            data=np.empty((0, 4), dtype='f8'),
            overwrite=True,
        )
        detect_group.create_array('scores', data=np.empty((0,), dtype=np.float32), overwrite=True)
        detect_group.create_array('class_ids', data=np.empty((0,), dtype=np.int32), overwrite=True)
        frame_counts_empty = np.zeros(num_images, dtype='i4')
        create_geometry_preload_array(
            detect_group,
            'frame_counts',
            data=frame_counts_empty,
            overwrite=True
        )
        create_geometry_preload_array(
            detect_group,
            'n_detections',
            data=frame_counts_empty,
            overwrite=True,
        )

    detect_group.attrs[_TRADITIONAL_SOURCE_LINEAGE_ATTR] = source_lineage
    detect_group.attrs[f"{_TRADITIONAL_SOURCE_LINEAGE_ATTR}_sha256"] = (
        source_lineage_sha256
    )
    _stamp_traditional_artifact_semantics(
        detect_group,
        reference_width=int(ds_img_shape[1]),
        reference_height=int(ds_img_shape[0]),
        source_frame_count=int(num_images),
        source_lineage_sha256=source_lineage_sha256,
    )

    if total_detections == 0:
        publish_empty_artifact_observation_proof(
            detect_group,
            source_frame_count=int(num_images),
            row_array_names=(
                "artifact_row_id",
                "frame_indices",
                "bbox_norm_coords",
                "scores",
                "class_ids",
            ),
            full_domain_evidence={
                "coverage_status": "full_source_domain_validated",
                "source_array_ref": "/raw_video/images_ds",
                "source_array_content_sha256": images_ds_sha256,
                "background_array_ref": (
                    f"/background_runs/{latest_bg_run}/background_ds"
                ),
                "background_array_content_sha256": background_ds_sha256,
                "source_frame_count": int(num_images),
                "worker_slice_plan": [
                    [int(chunk.start), int(chunk.stop)] for chunk in chunk_slices
                ],
                "validated_worker_result_count": len(results),
                "validated_observation_row_count": 0,
                "algorithm": "traditional_blob_detection_unchanged_v1",
            },
        )
    payload_inventory = publish_artifact_payload_inventory_seal(
        detect_group,
        source_frame_count=int(num_images),
    )
    payload_inventory_sha256 = detect_group.attrs[
        "artifact_payload_inventory_seal_sha256"
    ]

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
        'source_geometry_status': 'unbound_downsample_artifact',
        'source_geometry_reason': (
            'raw_video/images_ds lacks an exact persisted transform into the '
            'acquisition source-camera authority'
        ),
        'total_frames': num_images,
        'has_raw_video': True,
        # Detection parameters (method-specific)
        'parameters': detect_params,
        'parameter_source': param_source,
        
        # Background subtraction info (method-specific)
        'source_background_run': latest_bg_run,
        'source_images_ds_content_sha256': images_ds_sha256,
        'source_background_ds_content_sha256': background_ds_sha256,
        'source_array_fingerprint_schema': _ARRAY_FINGERPRINT_SCHEMA,
        
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
            'frame_source': 'raw_video/images_ds',
            'frame_source_content_sha256': images_ds_sha256,
            'source_video_path': root.attrs.get('source_video_path'),
            'source_background_run': latest_bg_run,
            'source_background_array': (
                f'background_runs/{latest_bg_run}/background_ds'
            ),
            'source_background_content_sha256': background_ds_sha256,
        },
        artifacts={
            'run_path': f'{DETECTION_ARTIFACT_RUN_FAMILY}/{run_name}',
            'bbox_norm_coords': (
                f'{DETECTION_ARTIFACT_RUN_FAMILY}/{run_name}/bbox_norm_coords'
            ),
            'artifact_payload_inventory_seal_sha256': (
                payload_inventory_sha256
            ),
            'artifact_payload_row_count': payload_inventory['row_count'],
        },
    )
    write_stage_provenance(detect_group, provenance_record)
    _require_array_fingerprint(
        images_ds,
        expected=images_ds_sha256,
        label="raw_video/images_ds",
    )
    _require_array_fingerprint(
        background_node,
        expected=background_ds_sha256,
        label=f"background_runs/{latest_bg_run}/background_ds",
    )
    attempt.complete(
        run_provenance=build_run_provenance_from_stage_record(provenance_record),
    )
    console.print(
        "[yellow]Skipped canonical detect-quality publication:[/yellow] "
        "traditional geometry is retained as a nonselector artifact."
    )

    return {
        'duration_seconds': duration,
        'total_detections': total_detections,
        'frames_with_detections': frames_with_detections,
        'run_name': run_name,
        'run_path': f'{DETECTION_ARTIFACT_RUN_FAMILY}/{run_name}',
        'output_parent': DETECTION_ARTIFACT_RUN_FAMILY,
        'stage_selector_eligible': False,
        'parameter_source': param_source
    }


# ---------------- CLI ---------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run traditional blob detection as a nonselector detection artifact. "
            "The path cannot publish a normal detect run until images_ds carries "
            "an exact acquisition transform."
        )
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
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help=(
            "Explicitly permit an unbound nonselector detection_artifact_runs output."
        ),
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
            artifact_only=bool(args.artifact_only),
        )
    except Exception as exc:
        console.print(f"[bold red]Detection failed:[/bold red] {exc}")
        return 1

    console.print(
        f"[green]Done.[/green] Artifact '{result['run_path']}' "
        f"with {result['total_detections']} detections "
        f"({result['frames_with_detections']} frames)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
