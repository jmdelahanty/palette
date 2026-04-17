#!/usr/bin/env python3
"""
Keypoint Detection Tuner - Interactive tool for optimizing anatomical keypoint detection.

This tuner helps find optimal parameters for detecting the swim bladder and eyes
in fish ROI images. The goal is to consistently detect exactly 3 blobs that can be
identified as bladder, left eye, and right eye.

Usage:
    python -m fisheye.tune.keypoint_tuner data.zarr [start_frame]
    python -m fisheye.tune.keypoint_review data.zarr --retune  # failure retune

Controls:
    - Arrow keys: Navigate frames
    - Trackbars: Adjust detection parameters
    - 's': Save parameters to Zarr metadata
    - 'd': Toggle difference image
    - 'g': Toggle geometry visualization
    - 'q' or ESC: Quit
    
Failure retune mode (via keypoint_review):
    - 'e': Evaluate params on remaining failures
    - 'a': Apply params to remaining failures
"""

import cv2
import numpy as np
import zarr
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import yaml
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from skimage.morphology import disk, erosion, dilation
from skimage.measure import label, regionprops

try:
    from ..detection.detect_keypoints_traditional import detect_keypoints_traditional
    from ..pose.heading import compute_heading_from_attrs, compute_heading_from_spec
    from ..pose.heuristics import (
        heuristic_profile_from_package,
        require_blob_assignment,
        require_geometry_qc,
    )
    from ..pose.schema import schema_from_package
    from ..refinement.keypoint_quality import compute_geometry_metrics
    from ..refinement.refine_keypoints import _detect_eye_flip
    from ..shared.keypoint_temporal_heading import refresh_refined_keypoint_heading_fields
except ImportError:  # pragma: no cover - fallback for script execution
    from fisheye.detection.detect_keypoints_traditional import detect_keypoints_traditional
    from fisheye.pose.heading import compute_heading_from_attrs, compute_heading_from_spec
    from fisheye.pose.heuristics import (
        heuristic_profile_from_package,
        require_blob_assignment,
        require_geometry_qc,
    )
    from fisheye.pose.schema import schema_from_package
    from fisheye.refinement.keypoint_quality import compute_geometry_metrics
    from fisheye.refinement.refine_keypoints import _detect_eye_flip
    from fisheye.shared.keypoint_temporal_heading import refresh_refined_keypoint_heading_fields

MIN_AREA_SLIDER_MAX = 1000
MIN_TRI_AREA_SLIDER_MAX = 2000
MAX_TRI_AREA_SLIDER_MAX = 200000
MIN_ANGLE_SLIDER_MAX = 90
MAX_ANGLE_SLIDER_MAX = 180
EVAL_SAMPLE_DEFAULT = 300
APPLY_BATCH_DEFAULT = 128
APPLY_WORKERS_DEFAULT = max(1, min(4, os.cpu_count() or 1))
TRADITIONAL_POSE_SCHEMA = schema_from_package("traditional_v1")
TRADITIONAL_HEURISTIC_PROFILE = heuristic_profile_from_package(
    "traditional_pose", TRADITIONAL_POSE_SCHEMA.name
)
TRADITIONAL_BLOB_ASSIGNMENT = require_blob_assignment(
    TRADITIONAL_HEURISTIC_PROFILE,
    family="triangle_3blob",
)
TRADITIONAL_GEOMETRY_QC = require_geometry_qc(TRADITIONAL_HEURISTIC_PROFILE)


def _required_geometry_default(value: Optional[float], field_name: str) -> float:
    if value is None:
        raise RuntimeError(
            "Traditional pose heuristic profile is missing "
            f"'geometry_qc.{field_name}'."
        )
    return float(value)


DEFAULT_MIN_VALID_ANGLE = _required_geometry_default(
    TRADITIONAL_GEOMETRY_QC.min_triangle_angle_deg,
    "min_triangle_angle_deg",
)
DEFAULT_MAX_VALID_ANGLE = _required_geometry_default(
    TRADITIONAL_GEOMETRY_QC.max_triangle_angle_deg,
    "max_triangle_angle_deg",
)
DEFAULT_MIN_TRIANGLE_AREA = _required_geometry_default(
    TRADITIONAL_GEOMETRY_QC.min_triangle_area_px,
    "min_triangle_area_px",
)
DEFAULT_MAX_TRIANGLE_AREA = (
    float(TRADITIONAL_GEOMETRY_QC.max_triangle_area_px)
    if TRADITIONAL_GEOMETRY_QC.max_triangle_area_px is not None
    else None
)
DEFAULT_MAX_TRIANGLE_AREA_SLIDER = (
    int(round(DEFAULT_MAX_TRIANGLE_AREA))
    if DEFAULT_MAX_TRIANGLE_AREA is not None and DEFAULT_MAX_TRIANGLE_AREA > 0
    else 0
)

# Global variables for trackbar values
current_frame = 1
min_valid_angle = int(round(DEFAULT_MIN_VALID_ANGLE))
max_valid_angle = int(round(DEFAULT_MAX_VALID_ANGLE))
min_triangle_area = int(round(DEFAULT_MIN_TRIANGLE_AREA))
max_triangle_area = DEFAULT_MAX_TRIANGLE_AREA_SLIDER
current_detection = 0
roi_thresh = 50
se1_radius = 1
se2_radius = 2
min_area = 5
use_difference = 1  # Default to using difference (matches actual pipeline)
show_geometry = 1   # Show triangle geometry analysis

def update_frame(val):
    global current_frame
    current_frame = val

def update_detection(val):
    global current_detection
    current_detection = val

def update_roi_thresh(val):
    global roi_thresh
    roi_thresh = val

def update_se1(val):
    global se1_radius
    se1_radius = max(1, val)

def update_se2(val):
    global se2_radius
    se2_radius = max(1, val)

def update_min_area(val):
    global min_area
    min_area = max(1, val)

def update_use_difference(val):
    global use_difference
    use_difference = val

def update_show_geometry(val):
    global show_geometry
    show_geometry = val

def update_min_valid_angle(val):
    global min_valid_angle
    min_valid_angle = max(1, val)

def update_max_valid_angle(val):
    global max_valid_angle
    max_valid_angle = max(1, val)

def update_min_triangle_area(val):
    global min_triangle_area
    min_triangle_area = max(1, val)

def update_max_triangle_area(val):
    global max_triangle_area
    max_triangle_area = max(0, val)


def _apply_keypoint_params(params: Dict[str, Any]) -> None:
    global roi_thresh, se1_radius, se2_radius
    global min_area, min_triangle_area, max_triangle_area, min_valid_angle, max_valid_angle
    if params is None:
        return
    if 'roi_thresh' in params:
        roi_thresh = int(params['roi_thresh'])
    if 'se1_radius' in params:
        se1_radius = int(params['se1_radius'])
    if 'se2_radius' in params:
        se2_radius = int(params['se2_radius'])
    if 'min_area' in params:
        min_area = int(params['min_area'])
    if 'min_triangle_area' in params:
        min_triangle_area = int(params['min_triangle_area'])
    if 'max_triangle_area' in params:
        value = params['max_triangle_area']
        if value is None:
            max_triangle_area = 0
        else:
            try:
                max_triangle_area = max(0, int(value))
            except (TypeError, ValueError):
                max_triangle_area = 0
    if 'min_valid_angle' in params:
        min_valid_angle = int(params['min_valid_angle'])
    if 'max_valid_angle' in params:
        max_valid_angle = int(params['max_valid_angle'])


def _load_initial_params(config_path: Path) -> None:
    if not config_path.exists():
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        kp_params = config.get('keypoints', {})
        _apply_keypoint_params(kp_params)
    print(f"✓ Loaded initial parameters from {config_path}")


def _load_tuned_params_from_zarr(root: zarr.Group) -> Optional[Dict[str, Any]]:
    if 'analysis_metadata' not in root:
        return None
    analysis_meta = root['analysis_metadata']
    tuning = analysis_meta.attrs.get('keypoint_tuning')
    if not isinstance(tuning, dict):
        return None
    tuned_params = tuning.get('tuned_parameters')
    if not isinstance(tuned_params, dict) or not tuned_params:
        return None
    return tuned_params

def calculate_triangle_area(p1, p2, p3):
    """Calculate area of triangle using cross product method."""
    # Vectors from p1 to p2 and p1 to p3
    v1 = p2 - p1
    v2 = p3 - p1
    # Area = 0.5 * |cross product|
    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return area

def process_roi_for_keypoints(roi_image, background_roi, params):
    """
    Process an ROI image to detect keypoints using morphology + geometry validation.
    Returns processed image, all regions, and identified keypoints.
    """
    if roi_image is None or roi_image.size == 0:
        return None, [], None, 0
    
    # Use difference image
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
    else:
        diff_roi = roi_image
    
    # Morphology with geometry validation
    se1 = disk(params['se1_radius'])
    
    base_thresh = params['roi_thresh']
    current_se2_radius = params['se2_radius']
    keypoint_stats = []
    last_roi_stats = []
    effective_se2_radius = current_se2_radius
    min_angle_threshold = params.get('min_valid_angle', DEFAULT_MIN_VALID_ANGLE)
    max_angle_threshold = params.get('max_valid_angle', DEFAULT_MAX_VALID_ANGLE)
    min_triangle_area = params.get('min_triangle_area', DEFAULT_MIN_TRIANGLE_AREA)
    max_triangle_area = params.get('max_triangle_area', DEFAULT_MAX_TRIANGLE_AREA)
    angle_min = min(min_angle_threshold, max_angle_threshold)
    angle_max = max(min_angle_threshold, max_angle_threshold)
    
    se2_radius_int = max(1, int(round(current_se2_radius)))
    se2 = disk(se2_radius_int)
    # Apply morphological operations
    im_roi = erosion(dilation(erosion(
        diff_roi >= base_thresh, se1), se2), se1
    )

    # Find regions
    roi_stat = [r for r in regionprops(label(im_roi))
                if r.area > params['min_area']]
    last_roi_stats = roi_stat

    if len(roi_stat) >= 3:
        # Take top 3 by area
        candidate_stats = sorted(roi_stat, key=lambda r: r.area, reverse=True)[:3]

        # Calculate triangle geometry for validation
        centroids = np.array([r.centroid for r in candidate_stats])
        angles = calculate_triangle_angles(centroids[0], centroids[1], centroids[2])
        tri_area = calculate_triangle_area(centroids[0], centroids[1], centroids[2])

        # Check if angles form a valid triangle AND triangle is large enough
        max_ok = (
            max_triangle_area is None
            or max_triangle_area <= 0
            or tri_area <= max_triangle_area
        )
        if (
            np.all(angles >= angle_min)
            and np.all(angles <= angle_max)
            and tri_area >= min_triangle_area
            and max_ok
        ):
            keypoint_stats = candidate_stats
            effective_se2_radius = se2_radius_int
    
    # Try to identify keypoints if we have valid 3 blobs
    keypoint_id = None
    if len(keypoint_stats) == 3:
        keypoint_id = identify_keypoints_by_geometry(keypoint_stats)
        if keypoint_id:
            keypoint_id['effective_thresh'] = base_thresh
            keypoint_id['effective_se2_radius'] = effective_se2_radius
    
    # Return final processed image
    final_se2 = disk(max(1, int(round(effective_se2_radius))))
    final_processed = erosion(dilation(erosion(
        diff_roi >= base_thresh, se1), final_se2), se1
    )
    
    return final_processed, keypoint_stats, keypoint_id, len(last_roi_stats)

def save_keypoint_params(zarr_path, params):
    """
    Save keypoint detection parameters to Zarr metadata ONLY.
    
    Config file remains as a template - tuned parameters go in zarr.
    
    Args:
        zarr_path: Path to zarr file
        params: Dictionary of parameters including roi_thresh, se1_radius, etc.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Save to Zarr analysis_metadata
        root = zarr.open_group(zarr_path, mode='r+')
        
        if 'analysis_metadata' not in root:
            analysis_meta = root.create_group('analysis_metadata')
        else:
            analysis_meta = root['analysis_metadata']
        
        # Get existing attrs or create new dict
        metadata = dict(analysis_meta.attrs) if analysis_meta.attrs else {}
        
        # Add/update keypoint tuning data
        metadata['keypoint_tuning'] = {
            'method': 'threshold_morphology',
            'version': '2.0',
            'tuned_timestamp': datetime.now(timezone.utc).isoformat(),
            'tuned_parameters': {
                'roi_thresh': params['roi_thresh'],
                'se1_radius': params['se1_radius'],
                'se2_radius': params['se2_radius'],
                'min_area': params['min_area'],
                'min_valid_angle': params.get('min_valid_angle', DEFAULT_MIN_VALID_ANGLE),
                'max_valid_angle': params.get('max_valid_angle', DEFAULT_MAX_VALID_ANGLE),
                'min_triangle_area': params.get(
                    'min_triangle_area', DEFAULT_MIN_TRIANGLE_AREA
                ),
                'max_triangle_area': params.get(
                    'max_triangle_area', DEFAULT_MAX_TRIANGLE_AREA
                ),
            },
            'tuned_on_frame': params.get('frame_index', None),
            'tuned_on_detection': params.get('detection_index', None)
        }
        
        # Save back to attrs
        analysis_meta.attrs.update(metadata)
        
        print(f"\n✓ Parameters saved to zarr: {zarr_path}")
        print(f"   Location: analysis_metadata/attrs['keypoint_tuning']")
        print(f"   roi_thresh: {params['roi_thresh']}")
        print(f"   se1_radius: {params['se1_radius']}")
        print(f"   se2_radius: {params['se2_radius']}")
        print(f"   min_area: {params['min_area']}")
        print(f"   min_valid_angle: {params.get('min_valid_angle', 'N/A')}")
        print(f"   min_triangle_area: {params['min_triangle_area']}")
        print(f"   max_triangle_area: {params.get('max_triangle_area')}")
        print(f"   max_valid_angle: {params.get('max_valid_angle', 'N/A')}")
        print(f"   Tuned on frame: {params.get('frame_index', 'N/A')}")
        print(f"   Tuned on detection: {params.get('detection_index', 'N/A')}")
        
        return True, "Parameters saved to zarr"
        
    except Exception as e:
        return False, f"Error saving parameters: {e}"


def calculate_triangle_angles(p1, p2, p3):
    """Calculate angles at each vertex of a triangle."""
    # Calculate side lengths
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    
    angles = np.zeros(3)
    
    if b * c > 0:
        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        angles[0] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    if a * c > 0:
        cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        angles[1] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    if a * b > 0:
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        angles[2] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.rad2deg(angles)


def _load_failure_indices(refined: zarr.Group) -> np.ndarray:
    if "refined_success" in refined:
        success = np.asarray(refined["refined_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "source_success" in refined:
        success = np.asarray(refined["source_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "failure_indices" in refined:
        failures = np.asarray(refined["failure_indices"][:], dtype="i4")
    else:
        return np.zeros(0, dtype="i4")

    reason_arr = refined.get("reason")
    if reason_arr is None or failures.size == 0:
        return failures
    try:
        reason_vals = np.asarray(reason_arr[:], dtype=object)
    except Exception:
        return failures
    if reason_vals.size == 0:
        return failures
    keep_mask = []
    for idx in failures:
        try:
            text = str(reason_vals[int(idx)]) if reason_vals[int(idx)] is not None else ""
        except Exception:
            text = ""
        keep_mask.append(
            "fish_present_no_keypoints" not in text
            and "detection_issue" not in text
        )
    return failures[np.array(keep_mask, dtype=bool)]


def _ensure_retune_id_array(refined: zarr.Group, chunks: Sequence[int]) -> zarr.Array:
    if "retune_id" in refined:
        return refined["retune_id"]
    total_rois = refined["keypoints_roi"].shape[0]
    return refined.create_array(
        "retune_id",
        shape=(total_rois,),
        chunks=chunks,
        dtype="i4",
        fill_value=-1,
        overwrite=True,
    )


def _get_or_create_retune_id(refined: zarr.Group, params: Dict[str, Any]) -> int:
    existing = refined.attrs.get("retune_params")
    retune_params = existing if isinstance(existing, dict) else {}

    def signature(values: Dict[str, Any]) -> tuple:
        return tuple(sorted(values.items()))

    target = signature(params)
    for key, value in retune_params.items():
        if isinstance(value, dict) and signature(value) == target:
            try:
                return int(key)
            except ValueError:
                continue

    existing_ids = [int(k) for k in retune_params.keys() if str(k).isdigit()]
    next_id = max(existing_ids, default=0) + 1
    retune_params[str(next_id)] = params
    refined.attrs["retune_params"] = retune_params
    return next_id


def _merge_reason(existing: str, tags: Sequence[str]) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag and tag != "detection_failed"]
    merged = sorted(set(existing_tags + list(tags)))
    return "|".join(merged) if merged else "clean"


def _extract_background_roi(
    background: Optional[np.ndarray], roi_coord: np.ndarray, roi_shape: tuple
) -> Optional[np.ndarray]:
    if background is None or roi_coord[0] == -1:
        return None
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    y1, x1 = int(roi_coord[1]), int(roi_coord[0])
    y2, x2 = y1 + roi_h, x1 + roi_w

    bg_h, bg_w = background.shape[:2]
    vy1, vx1 = max(y1, 0), max(x1, 0)
    vy2, vx2 = min(y2, bg_h), min(x2, bg_w)

    if vy1 >= vy2 or vx1 >= vx2:
        return np.zeros((roi_h, roi_w), dtype=background.dtype)

    if y1 >= 0 and x1 >= 0 and y2 <= bg_h and x2 <= bg_w:
        roi = background[y1:y2, x1:x2]
        if roi.shape[:2] == (roi_h, roi_w):
            return roi

    if background.ndim == 2:
        roi = np.zeros((roi_h, roi_w), dtype=background.dtype)
    else:
        roi = np.zeros((roi_h, roi_w, background.shape[2]), dtype=background.dtype)

    py1 = max(0, -y1)
    px1 = max(0, -x1)
    py2 = py1 + (vy2 - vy1)
    px2 = px1 + (vx2 - vx1)
    roi[py1:py2, px1:px2] = background[vy1:vy2, vx1:vx2]
    return roi


def _sanitize_reason_array(reason_arr: zarr.Array) -> None:
    try:
        raw = reason_arr[:]
    except Exception:
        return
    if raw.size == 0:
        return

    def coerce(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.item())
            return "|".join(str(item) for item in val.tolist())
        return str(val)

    cleaned = np.array([coerce(v) for v in raw], dtype=object)
    reason_arr[:] = cleaned


def _load_frame_flags(path: Path) -> dict[str, list[Dict[str, Optional[int]]]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: dict[str, list[Dict[str, Optional[int]]]] = {}
    for key, value in data.items():
        entries: list[Dict[str, Optional[int]]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    frame_val = item.get("frame_idx")
                    roi_val = item.get("roi_idx")
                    try:
                        frame_idx = int(frame_val) if frame_val is not None else None
                    except (TypeError, ValueError):
                        frame_idx = None
                    try:
                        roi_idx = int(roi_val) if roi_val is not None else None
                    except (TypeError, ValueError):
                        roi_idx = None
                    if frame_idx is not None:
                        entries.append({"frame_idx": frame_idx, "roi_idx": roi_idx})
                else:
                    try:
                        frame_idx = int(item)
                    except (TypeError, ValueError):
                        continue
                    entries.append({"frame_idx": frame_idx, "roi_idx": None})
        parsed[str(key)] = entries
    return parsed


def _append_flagged_frame(
    flag_path: Path,
    zarr_path: str,
    frame_idx: int,
    roi_idx: Optional[int],
) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_frame_flags(flag_path)
    entries = data.get(zarr_path, [])
    dedupe = {(entry.get("frame_idx"), entry.get("roi_idx")) for entry in entries}
    key = (int(frame_idx), int(roi_idx) if roi_idx is not None else None)
    if key in dedupe:
        return
    entries.append({"frame_idx": int(frame_idx), "roi_idx": key[1]})
    entries.sort(key=lambda item: (item.get("frame_idx") or 0, item.get("roi_idx") or -1))
    data[zarr_path] = entries
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_detection_frame_flags(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: dict[str, list[int]] = {}
    for key, value in data.items():
        frames: list[int] = []
        if isinstance(value, list):
            for item in value:
                try:
                    frames.append(int(item))
                except (TypeError, ValueError):
                    continue
        parsed[str(key)] = sorted(set(frames))
    return parsed


def _append_detection_frame(flag_path: Path, zarr_path: str, frame_idx: int) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_detection_frame_flags(flag_path)
    frames = set(data.get(zarr_path, []))
    frames.add(int(frame_idx))
    data[zarr_path] = sorted(frames)
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _append_flagged_path(flag_path: Path, zarr_path: str) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if flag_path.exists():
        with flag_path.open("r", encoding="utf-8") as handle:
            existing = {line.strip() for line in handle if line.strip()}
    if zarr_path in existing:
        return
    with flag_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{zarr_path}\n")


def identify_keypoints_by_geometry(keypoint_stats):
    """
    Identify which blob is the bladder and which are eyes.
    
    Uses the packaged traditional pose heuristic profile to map the three
    candidate blobs onto swim bladder / eye labels.
    """
    if len(keypoint_stats) != 3:
        return None
    
    # Get centroids
    centroids = np.array([r.centroid for r in keypoint_stats])
    
    # Calculate angles at each vertex
    angles = calculate_triangle_angles(centroids[0], centroids[1], centroids[2])
    
    if TRADITIONAL_BLOB_ASSIGNMENT.bladder_vertex_rule != "smallest_angle":
        raise ValueError(
            "Unsupported traditional blob assignment rule "
            f"'{TRADITIONAL_BLOB_ASSIGNMENT.bladder_vertex_rule}'."
        )

    bladder_idx = int(np.argmin(angles))
    eye_indices = [i for i in range(3) if i != bladder_idx]
    
    bladder = centroids[bladder_idx]
    eyes = centroids[eye_indices]
    eye_mid = eyes.mean(axis=0)
    dy = float(eye_mid[0] - bladder[0])
    dx = float(eye_mid[1] - bladder[1])

    heading_deg = compute_heading_from_spec(
        TRADITIONAL_POSE_SCHEMA.metadata.get("heading_computation"),
        labels=TRADITIONAL_POSE_SCHEMA.node_names,
        points=np.asarray(
            [
                [bladder[1], bladder[0]],
                [centroids[eye_indices[0]][1], centroids[eye_indices[0]][0]],
                [centroids[eye_indices[1]][1], centroids[eye_indices[1]][0]],
            ],
            dtype=np.float64,
        ),
    )

    left_eye_idx = None
    right_eye_idx = None
    if TRADITIONAL_BLOB_ASSIGNMENT.left_right_rule == "heading_relative" and (dx or dy):
        # Determine left/right relative to heading so rotated view is stable.
        left_vec = np.array([-dx, dy], dtype=np.float64)  # (y, x) vector for "left" side
        eye_vecs = eyes - bladder
        dots = eye_vecs @ left_vec
        if dots[0] != dots[1]:
            left_eye_idx = eye_indices[int(np.argmax(dots))]
            right_eye_idx = eye_indices[int(np.argmin(dots))]
    elif TRADITIONAL_BLOB_ASSIGNMENT.left_right_rule != "heading_relative":
        raise ValueError(
            "Unsupported traditional blob assignment rule "
            f"'{TRADITIONAL_BLOB_ASSIGNMENT.left_right_rule}'."
        )

    if left_eye_idx is None or right_eye_idx is None:
        if TRADITIONAL_BLOB_ASSIGNMENT.fallback_rule != "image_x_order":
            raise ValueError(
                "Unsupported traditional blob assignment fallback "
                f"'{TRADITIONAL_BLOB_ASSIGNMENT.fallback_rule}'."
            )
        eye_x = [centroids[i][1] for i in eye_indices]
        if eye_x[0] < eye_x[1]:
            left_eye_idx, right_eye_idx = eye_indices[0], eye_indices[1]
        else:
            left_eye_idx, right_eye_idx = eye_indices[1], eye_indices[0]
    
    return {
        'bladder': keypoint_stats[bladder_idx],
        'left_eye': keypoint_stats[left_eye_idx],
        'right_eye': keypoint_stats[right_eye_idx],
        'bladder_idx': bladder_idx,
        'left_eye_idx': left_eye_idx,
        'right_eye_idx': right_eye_idx,
        'angles': angles,
        'heading': heading_deg
    }


def create_keypoint_dashboard(roi_image, background_roi, params, frame_num, det_num, roi_idx, mode_label=None):
    """
    Create comprehensive visualization for keypoint detection tuning.
    """
    display_size = (400, 400)
    
    if roi_image is None:
        # Create blank dashboard
        blank = np.zeros((display_size[1] * 2, display_size[0] * 2, 3), dtype=np.uint8)
        cv2.putText(blank, "No detection at this frame", (50, display_size[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank
    
    # Process the ROI
    processed, keypoint_stats, keypoint_id, raw_blob_count = process_roi_for_keypoints(
        roi_image, background_roi, params
    )
    
    # Panel 1: Original ROI
    panel1 = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel1, "Original ROI", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Panel 2: Background ROI (if available)
    if background_roi is not None:
        panel2 = cv2.cvtColor(background_roi, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "Background ROI", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        panel2 = np.zeros_like(panel1)
        cv2.putText(panel2, "No Background", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Panel 3: Difference image (if using diff mode)
    if params['use_diff'] and background_roi is not None:
        diff_roi = np.clip(
            background_roi.astype(np.int16) - roi_image.astype(np.int16), 
            0, 255
        ).astype(np.uint8)
        panel3 = cv2.cvtColor(diff_roi, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel3, "Difference (BG - ROI)", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        panel3 = np.zeros_like(panel1)
        cv2.putText(panel3, "Diff mode OFF", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Panel 4: Processed with keypoint detection
    panel4 = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Panel 5 placeholder (rotated view)
    panel5 = np.zeros_like(panel1)
    
    # Draw detected keypoints
    color_map = {
        'bladder': (0, 255, 0),      # Green
        'left_eye': (255, 0, 0),     # Blue
        'right_eye': (0, 0, 255)     # Red
    }
    
    if keypoint_id:
        # Draw labeled keypoints
        for key, stat in [('bladder', keypoint_id['bladder']),
                         ('left_eye', keypoint_id['left_eye']),
                         ('right_eye', keypoint_id['right_eye'])]:
            y, x = map(int, stat.centroid)
            color = color_map[key]
            cv2.circle(panel4, (x, y), 5, color, -1)
            cv2.putText(panel4, key.split('_')[0][:4].upper(), (x + 8, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw triangle if geometry is enabled
        if params['show_geometry']:
            pts = np.array([
                [int(keypoint_id['bladder'].centroid[1]), int(keypoint_id['bladder'].centroid[0])],
                [int(keypoint_id['left_eye'].centroid[1]), int(keypoint_id['left_eye'].centroid[0])],
                [int(keypoint_id['right_eye'].centroid[1]), int(keypoint_id['right_eye'].centroid[0])]
            ])
            cv2.polylines(panel4, [pts], True, (255, 255, 0), 1)
            
            # Display angles
            angles = keypoint_id['angles']
            y_offset = roi_image.shape[0] - 40
            cv2.putText(panel4, f"Angles: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}", 
                       (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        effective_se2 = keypoint_id.get('effective_se2_radius', params['se2_radius'])
        status = "IDENTIFIED"
        status_color = (0, 255, 0)

        # Build rotated fish frame visualization
        heading_deg = float(keypoint_id.get('heading', 0.0))
        if np.isfinite(heading_deg):
            center = (roi_image.shape[1] / 2.0, roi_image.shape[0] / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, heading_deg, 1.0)
            rotated_roi = cv2.warpAffine(
                roi_image,
                rot_mat,
                (roi_image.shape[1], roi_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            # Apply a circular window to avoid showing rotating edges.
            circle_mask = np.zeros_like(rotated_roi, dtype=np.uint8)
            center_pt = (rotated_roi.shape[1] // 2, rotated_roi.shape[0] // 2)
            radius = min(center_pt)
            cv2.circle(circle_mask, center_pt, radius, 255, -1)
            rotated_roi = np.where(circle_mask > 0, rotated_roi, 0)
            panel5 = cv2.cvtColor(rotated_roi.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            def rotate_point(stat):
                y, x = stat.centroid
                vec = np.array([x, y, 1.0])
                rx, ry = rot_mat @ vec
                return int(round(rx)), int(round(ry))

            for key, stat in [('bladder', keypoint_id['bladder']),
                              ('left_eye', keypoint_id['left_eye']),
                              ('right_eye', keypoint_id['right_eye'])]:
                rx, ry = rotate_point(stat)
                color = color_map[key]
                cv2.circle(panel5, (rx, ry), 5, color, -1)
                cv2.putText(panel5, key.split('_')[0][:4].upper(), (rx + 8, ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(panel5, "Rotated ROI", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(panel5, "Rotated ROI (heading NA)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        # Draw unidentified blobs
        for stat in keypoint_stats:
            y, x = map(int, stat.centroid)
            cv2.circle(panel4, (x, y), 5, (128, 128, 128), -1)
        
        effective_se2 = params['se2_radius']
        status = f"Found {raw_blob_count} blobs (need 3)"
        status_color = (0, 165, 255)
        cv2.putText(panel5, "Rotated ROI (unavailable)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    cv2.putText(panel4, status, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
    
    max_tri = params.get('max_triangle_area')
    max_tri_label = "off" if max_tri is None or max_tri <= 0 else f"{int(max_tri)}"
    slider_info = (
        f"Thresh:{params['roi_thresh']} | SE1:{params['se1_radius']} "
        f"| SE2 slider:{params['se2_radius']} | SE2 eff:{effective_se2} "
        f"| MinArea:{params['min_area']} | Ang:{params['min_valid_angle']}-{params.get('max_valid_angle', DEFAULT_MAX_VALID_ANGLE)} "
        f"| MinTri:{params['min_triangle_area']} | MaxTri:{max_tri_label}"
    )
    cv2.putText(panel4, slider_info, (10, 38),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Resize all panels
    panel1 = cv2.resize(panel1, display_size)
    panel2 = cv2.resize(panel2, display_size)
    panel3 = cv2.resize(panel3, display_size)
    panel4 = cv2.resize(panel4, display_size)
    panel5 = cv2.resize(panel5, display_size)
    blank_panel = np.zeros_like(panel1)
    if mode_label:
        for i, line in enumerate(str(mode_label).split("\n")):
            y = 30 + i * 22
            cv2.putText(blank_panel, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Add frame info to panel 1
    info_text = f"Frame: {frame_num}, Det: {det_num}, ROI: {roi_idx}"
    cv2.putText(panel1, info_text, (10, panel1.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Combine into 2x3 grid
    top_row = np.hstack((panel1, panel2, panel5))
    bottom_row = np.hstack((panel3, panel4, blank_panel))
    dashboard = np.vstack((top_row, bottom_row))
    
    return dashboard


def run_failure_tuner(
    zarr_path: str,
    refined_run: Optional[str],
    start_index: int,
    *,
    apply_batch_size: int = APPLY_BATCH_DEFAULT,
    apply_workers: int = APPLY_WORKERS_DEFAULT,
    target_frames: Optional[Sequence[int]] = None,
    frame_flag_file: Optional[str] = None,
    detect_flag_file: Optional[str] = None,
    detect_frame_flag_file: Optional[str] = None,
) -> None:
    global current_frame, roi_thresh, se1_radius, se2_radius
    global min_area, min_triangle_area, max_triangle_area, use_difference, show_geometry
    global min_valid_angle, max_valid_angle

    current_frame = max(1, start_index)
    use_difference = 1

    config_path = Path("configs/fisheye/default.yaml")
    _load_initial_params(config_path)

    try:
        zarr_root = zarr.open_group(zarr_path, mode='a')
    except Exception as e:
        print(f"Error opening Zarr: {e}")
        return

    tuned_params = _load_tuned_params_from_zarr(zarr_root)
    if tuned_params:
        _apply_keypoint_params(tuned_params)
        print("✓ Loaded tuned keypoint parameters from analysis_metadata")

    refined_parent = zarr_root.get("refined_keypoints_runs")
    if refined_parent is None:
        print("Error: No refined_keypoints_runs found")
        return
    refined_run = refined_run or refined_parent.attrs.get("latest")
    if not refined_run or refined_run not in refined_parent:
        print("Error: Refined keypoint run not found")
        return
    refined = refined_parent[refined_run]

    failures = _load_failure_indices(refined)

    crop_run = refined.attrs.get("source_crop_run")
    if not crop_run and "crop_runs" in zarr_root:
        crop_run = zarr_root["crop_runs"].attrs.get("latest")
    if not crop_run:
        print("Error: No crop run found for failures")
        return
    crop_group = zarr_root[f"crop_runs/{crop_run}"]
    roi_images = crop_group["roi_images"]
    roi_coords = crop_group["roi_coordinates_full"]
    frame_indices = crop_group["frame_indices"][:]
    det_indices = crop_group.get("detection_indices")

    targeted = False
    if target_frames:
        target_arr = np.array(sorted({int(f) for f in target_frames}), dtype=np.int64)
        if target_arr.size:
            targeted = True
            failures = failures[np.isin(frame_indices[failures], target_arr)]

    filtered_flags = 0
    if frame_flag_file and failures.size > 0:
        try:
            flag_path = Path(frame_flag_file).expanduser()
            flag_data = _load_frame_flags(flag_path)
            flagged = flag_data.get(zarr_path, [])
            flagged_frames = {
                entry.get("frame_idx")
                for entry in flagged
                if isinstance(entry, dict) and entry.get("frame_idx") is not None and entry.get("roi_idx") is None
            }
            flagged_pairs = {
                (entry.get("frame_idx"), entry.get("roi_idx"))
                for entry in flagged
                if isinstance(entry, dict) and entry.get("frame_idx") is not None and entry.get("roi_idx") is not None
            }
            if flagged_frames or flagged_pairs:
                keep_mask = []
                for roi_idx in failures:
                    frame_idx = int(frame_indices[roi_idx])
                    pair = (frame_idx, int(roi_idx))
                    if frame_idx in flagged_frames:
                        keep_mask.append(False)
                    elif pair in flagged_pairs:
                        keep_mask.append(False)
                    else:
                        keep_mask.append(True)
                keep_mask_arr = np.array(keep_mask, dtype=bool)
                filtered_flags = int(np.sum(~keep_mask_arr))
                failures = failures[keep_mask_arr]
        except Exception as exc:
            print(f"Warning: failed to apply frame flags: {exc}")
    if failures.size == 0:
        if targeted:
            print("No failed keypoints to retune for requested frames.")
        else:
            print("No failed keypoints to retune.")
        return

    current_frame = max(1, min(current_frame, len(failures)))

    if "background_runs" not in zarr_root:
        print("Error: Run background stage first")
        return
    latest_bg = zarr_root["background_runs"].attrs.get("latest")
    if not latest_bg:
        print("Error: No latest background run found")
        return
    bg_group = zarr_root[f"background_runs/{latest_bg}"]
    if "background_full" in bg_group:
        background = bg_group["background_full"][:]
    else:
        print("Error: Background full-resolution array missing")
        return

    if "raw_video" in zarr_root and "images_full" in zarr_root["raw_video"]:
        full_h, full_w = zarr_root["raw_video/images_full"].shape[1:]
    elif "raw_video" in zarr_root and "images_ds" in zarr_root["raw_video"]:
        full_h, full_w = zarr_root["raw_video/images_ds"].shape[1:]
    else:
        print("Error: No raw_video images found")
        return
    norm_factor = np.array([full_w, full_h], dtype=np.float64)

    kp_roi_arr = refined.get("keypoints_roi")
    kp_img_arr = refined.get("keypoints_img")
    kp_norm_arr = refined.get("keypoints_norm")
    heading_arr = refined.get("heading")
    confidence_arr = refined.get("confidence")
    kp_conf_arr = refined.get("keypoint_confidences")
    thresh_arr = refined.get("effective_threshold")
    se2_arr = refined.get("effective_se2_radius")
    triangle_area_arr = refined.get("triangle_area")
    min_angle_arr = refined.get("min_angle")
    triangle_angles_arr = refined.get("triangle_angles")
    refined_success_arr = refined.get("refined_success")
    flip_corrected_arr = refined.get("flip_corrected")
    quality_labels_arr = refined.get("quality_labels")
    confidence_valid_arr = refined.get("confidence_valid")
    geometry_valid_arr = refined.get("geometry_valid")
    usable_arr = refined.get("usable_keypoints")
    reason_arr = refined.get("reason")
    heading_finite_arr = refined.get("heading_finite")
    heading_usable_arr = refined.get("heading_usable")
    detection_source_arr = refined.get("detection_source")

    retune_id_arr = _ensure_retune_id_array(
        refined, heading_arr.chunks or (min(1024, kp_roi_arr.shape[0]),)
    )
    if reason_arr is not None:
        _sanitize_reason_array(reason_arr)

    summary_raw = refined.attrs.get("summary_statistics", {})
    summary = summary_raw.get("refine", summary_raw) if isinstance(summary_raw, dict) else {}
    confidence_threshold = float(summary.get("confidence_threshold", 0.3))
    min_triangle_angle = float(summary.get("min_triangle_angle", DEFAULT_MIN_VALID_ANGLE))
    min_triangle_area = float(summary.get("min_triangle_area", DEFAULT_MIN_TRIANGLE_AREA))
    max_summary = summary.get("max_triangle_area")
    try:
        max_triangle_area = (
            float(max_summary)
            if max_summary is not None
            else DEFAULT_MAX_TRIANGLE_AREA_SLIDER
        )
    except (TypeError, ValueError):
        max_triangle_area = DEFAULT_MAX_TRIANGLE_AREA_SLIDER

    apply_batch_size = max(1, int(apply_batch_size))
    apply_workers = max(1, int(apply_workers))

    print("\nKeypoint Failure Retune")
    print(f"  Zarr: {zarr_path}")
    print(f"  Refined run: {refined_run}")
    print(f"  Crop run: {crop_run}")
    print(f"  Failures to retune: {len(failures)}")
    if filtered_flags:
        print(f"  Skipped flagged frames: {filtered_flags}")
    print(f"  Apply batch: {apply_batch_size} | Workers: {apply_workers}")
    flag_path = Path(frame_flag_file).expanduser() if frame_flag_file else None
    if flag_path is not None:
        print(f"  Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
    detect_flag_path = Path(detect_flag_file).expanduser() if detect_flag_file else None
    detect_frame_flag_path = (
        Path(detect_frame_flag_file).expanduser() if detect_frame_flag_file else None
    )
    if detect_flag_path is not None:
        print(f"  Detection flag file: {detect_flag_path.expanduser().resolve(strict=False)}")
    if detect_frame_flag_path is not None:
        print(f"  Detection frame flag file: {detect_frame_flag_path.expanduser().resolve(strict=False)}")

    window_name = "Keypoint Failure Retune"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)

    def update_failure_slider(val: int) -> None:
        global current_frame
        current_frame = max(1, int(val) + 1)

    cv2.createTrackbar(
        "Failure",
        window_name,
        max(0, current_frame - 1),
        max(0, len(failures) - 1),
        update_failure_slider,
    )
    cv2.createTrackbar("Threshold", window_name, roi_thresh, 255, update_roi_thresh)
    cv2.createTrackbar("SE1 Radius", window_name, se1_radius, 10, update_se1)
    cv2.createTrackbar("SE2 Radius", window_name, se2_radius, 10, update_se2)

    min_area = int(min(min_area, MIN_AREA_SLIDER_MAX))
    min_triangle_area = int(min(min_triangle_area, MIN_TRI_AREA_SLIDER_MAX))
    max_triangle_area = int(min(max_triangle_area, MAX_TRI_AREA_SLIDER_MAX))
    min_valid_angle = int(min(min_valid_angle, MIN_ANGLE_SLIDER_MAX))
    max_valid_angle = int(min(max_valid_angle, MAX_ANGLE_SLIDER_MAX))

    cv2.createTrackbar("Min Area", window_name, int(min_area), MIN_AREA_SLIDER_MAX, update_min_area)
    cv2.createTrackbar("Min Angle", window_name, min_valid_angle, MIN_ANGLE_SLIDER_MAX, update_min_valid_angle)
    cv2.createTrackbar("Max Angle", window_name, max_valid_angle, MAX_ANGLE_SLIDER_MAX, update_max_valid_angle)
    cv2.createTrackbar("Min Tri Area", window_name, int(min_triangle_area), MIN_TRI_AREA_SLIDER_MAX, update_min_triangle_area)
    cv2.createTrackbar("Max Tri Area", window_name, int(max_triangle_area), MAX_TRI_AREA_SLIDER_MAX, update_max_triangle_area)
    cv2.createTrackbar("Show Geometry", window_name, show_geometry, 1, update_show_geometry)

    print("\nControls:")
    print("  Arrow keys: Navigate failures")
    print("  e: Quick eval on a sample of remaining failures")
    print("  E: Eval all remaining failures (slow)")
    print("  a: Apply params to remaining failures")
    print("  b: flag current frame for follow-up (writes --frame-flag-file)")
    print("  d: flag detection issue (writes detection retune flags)")
    print("  g: Toggle geometry display")
    print("  q/ESC: Quit")

    def current_params() -> Dict[str, Any]:
        max_tri = max_triangle_area if max_triangle_area > 0 else None
        return {
            "roi_thresh": roi_thresh,
            "se1_radius": se1_radius,
            "se2_radius": se2_radius,
            "min_area": min_area,
            "min_valid_angle": min_valid_angle,
            "max_valid_angle": max_valid_angle,
            "min_triangle_area": min_triangle_area,
            "max_triangle_area": max_tri,
        }

    def evaluate_params(full: bool = False) -> None:
        params = current_params()
        total_failures = len(failures)
        if total_failures == 0:
            print("Eval: no failures remaining.")
            return

        if full or total_failures <= EVAL_SAMPLE_DEFAULT:
            sample_indices = failures
            label = f"all {total_failures}"
        else:
            sample_size = min(EVAL_SAMPLE_DEFAULT, total_failures)
            rng = np.random.default_rng(0)
            sample_indices = rng.choice(failures, size=sample_size, replace=False)
            label = f"sample {sample_size}/{total_failures}"

        success = 0
        total = len(sample_indices)
        for roi_idx in sample_indices:
            roi_idx = int(roi_idx)
            roi_image = roi_images[roi_idx]
            roi_coord = roi_coords[roi_idx]
            background_roi = _extract_background_roi(background, roi_coord, roi_image.shape)
            if background_roi is None:
                continue
            if detect_keypoints_traditional(roi_image, background_roi, **params) is not None:
                success += 1
        rate = (success / total * 100.0) if total else 0.0
        print(f"Eval ({label}): {success}/{total} would pass ({rate:.1f}%)")

    def apply_params() -> None:
        nonlocal failures
        global current_frame
        params = current_params()
        retune_id = _get_or_create_retune_id(refined, params)
        updated = 0
        total = len(failures)
        failures_list = [int(idx) for idx in failures]
        total_batches = (total + apply_batch_size - 1) // apply_batch_size if total else 0

        executor = ThreadPoolExecutor(max_workers=apply_workers) if apply_workers > 1 else None
        try:
            for batch_idx, start in enumerate(range(0, total, apply_batch_size), start=1):
                batch_indices = failures_list[start:start + apply_batch_size]
                batch_items: List[Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
                for roi_idx in batch_indices:
                    roi_image = roi_images[roi_idx]
                    roi_coord = roi_coords[roi_idx]
                    background_roi = _extract_background_roi(background, roi_coord, roi_image.shape)
                    batch_items.append((roi_idx, roi_image, roi_coord, background_roi))
                    retune_id_arr[roi_idx] = retune_id

                results: List[Optional[Dict[str, Any]]] = [None] * len(batch_items)
                if executor is None:
                    for i, (_, roi_image, _, background_roi) in enumerate(batch_items):
                        if background_roi is not None:
                            results[i] = detect_keypoints_traditional(
                                roi_image, background_roi, **params
                            )
                else:
                    futures = []
                    for i, (_, roi_image, _, background_roi) in enumerate(batch_items):
                        if background_roi is None:
                            continue
                        futures.append(
                            (i, executor.submit(
                                detect_keypoints_traditional, roi_image, background_roi, **params
                            ))
                        )
                    for i, future in futures:
                        results[i] = future.result()

                for (roi_idx, _, roi_coord, _), keypoints in zip(batch_items, results):
                    if keypoints is None:
                        continue

                    points = np.array(
                        [keypoints["bladder"], keypoints["eye_left"], keypoints["eye_right"]],
                        dtype=np.float64,
                    )
                    heading_for_flip = compute_heading_from_attrs(
                        refined.attrs,
                        labels=labels,
                        points=points,
                    )
                    flip_detected = _detect_eye_flip(points[0], points[1], points[2], heading_for_flip)
                    if flip_detected:
                        points[[1, 2]] = points[[2, 1]]
                    heading_val = compute_heading_from_attrs(
                        refined.attrs,
                        labels=labels,
                        points=points,
                    )

                    conf_missing = True
                    conf_ok = False
                    if kp_conf_arr is not None:
                        conf_vals = np.array(keypoints["keypoint_confidences"], dtype=np.float64)
                        if flip_detected:
                            conf_vals[[1, 2]] = conf_vals[[2, 1]]
                        kp_conf_arr[roi_idx] = conf_vals
                        if np.all(np.isfinite(conf_vals)):
                            conf_missing = False
                            conf_ok = bool(np.all(conf_vals >= confidence_threshold))

                    metrics = compute_geometry_metrics(points)
                    max_ok = max_triangle_area <= 0 or metrics.area <= max_triangle_area
                    geom_ok = bool(
                        np.isfinite(metrics.min_angle)
                        and np.isfinite(metrics.area)
                        and metrics.min_angle >= min_triangle_angle
                        and metrics.area >= min_triangle_area
                        and max_ok
                    )

                    if kp_roi_arr is not None:
                        kp_roi_arr[roi_idx] = points
                    full_points = points + roi_coord
                    if kp_img_arr is not None:
                        kp_img_arr[roi_idx] = full_points
                    if kp_norm_arr is not None:
                        kp_norm_arr[roi_idx] = full_points / norm_factor

                    if heading_arr is not None:
                        heading_arr[roi_idx] = heading_val
                    if confidence_arr is not None:
                        confidence_arr[roi_idx] = float(keypoints.get("confidence", 1.0))
                    if thresh_arr is not None:
                        thresh_arr[roi_idx] = float(keypoints.get("effective_threshold", params["roi_thresh"]))
                    if se2_arr is not None:
                        se2_arr[roi_idx] = float(keypoints.get("effective_se2_radius", params["se2_radius"]))

                    if triangle_area_arr is not None:
                        triangle_area_arr[roi_idx] = metrics.area
                    if min_angle_arr is not None:
                        min_angle_arr[roi_idx] = metrics.min_angle
                    if triangle_angles_arr is not None:
                        triangle_angles_arr[roi_idx] = metrics.angles

                    if refined_success_arr is not None:
                        refined_success_arr[roi_idx] = True
                    if flip_corrected_arr is not None:
                        flip_corrected_arr[roi_idx] = flip_detected
                    if quality_labels_arr is not None:
                        quality_labels_arr[roi_idx] = 6 if flip_detected else 0
                    if confidence_valid_arr is not None:
                        confidence_valid_arr[roi_idx] = conf_ok
                    if geometry_valid_arr is not None:
                        geometry_valid_arr[roi_idx] = geom_ok
                    if usable_arr is not None:
                        usable_arr[roi_idx] = conf_ok and geom_ok
                    heading_is_finite = bool(np.isfinite(heading_val))
                    if heading_finite_arr is not None:
                        heading_finite_arr[roi_idx] = heading_is_finite
                    if heading_usable_arr is not None:
                        det_src = int(detection_source_arr[roi_idx]) if detection_source_arr is not None else 0
                        heading_usable_arr[roi_idx] = det_src == 0 and heading_is_finite
                    if reason_arr is not None:
                        tags = []
                        if flip_detected:
                            tags.append("flip_corrected")
                        if conf_missing:
                            tags.append("confidence_missing")
                        elif not conf_ok:
                            tags.append("low_confidence")
                        if not geom_ok:
                            tags.append("geometry_issue")
                        existing_val = reason_arr[roi_idx]
                        existing = "" if existing_val is None else str(existing_val)
                        reason_value = str(_merge_reason(existing, tags))
                        reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

                    updated += 1

                if total_batches > 1:
                    print(f"  Batch {batch_idx}/{total_batches} processed ({updated}/{total} updated)")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        failures = _load_failure_indices(refined)
        remaining = len(failures)
        rate = (updated / total * 100.0) if total else 0.0
        print(f"Applied retune {retune_id}: {updated}/{total} updated ({rate:.1f}%)")
        print(f"Remaining failures: {remaining}")
        if remaining <= 0:
            cv2.setTrackbarMax("Failure", window_name, 0)
            cv2.setTrackbarPos("Failure", window_name, 0)
            current_frame = 1
            return

        new_max = max(0, remaining - 1)
        cv2.setTrackbarMax("Failure", window_name, new_max)
        current_frame = min(current_frame, remaining)
        cv2.setTrackbarPos("Failure", window_name, max(0, current_frame - 1))

    while True:
        if len(failures) == 0:
            print("No failures remaining.")
            break

        current_frame = max(1, min(current_frame, len(failures)))
        failure_pos = current_frame - 1
        roi_idx = int(failures[failure_pos])
        roi_image = roi_images[roi_idx]
        roi_coord = roi_coords[roi_idx]
        background_roi = _extract_background_roi(background, roi_coord, roi_image.shape)

        frame_idx = int(frame_indices[roi_idx]) if frame_indices is not None else roi_idx
        det_num = int(det_indices[roi_idx]) if det_indices is not None else 0

        params = {
            "roi_thresh": roi_thresh,
            "se1_radius": se1_radius,
            "se2_radius": se2_radius,
            "min_area": min_area,
            "min_valid_angle": min_valid_angle,
            "max_valid_angle": max_valid_angle,
            "use_diff": 1,
            "show_geometry": show_geometry,
            "min_triangle_area": min_triangle_area,
            "max_triangle_area": max_triangle_area if max_triangle_area > 0 else None,
        }

        mode_label = f"RETUNE FAILURES\nCorrecting {failure_pos + 1}/{len(failures)}"
        dashboard = create_keypoint_dashboard(
            roi_image, background_roi, params, frame_idx, det_num, roi_idx, mode_label
        )

        cv2.imshow(window_name, dashboard)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        if key == ord('e'):
            evaluate_params(full=False)
        elif key == ord('E'):
            evaluate_params(full=True)
        elif key == ord('a'):
            apply_params()
            if len(failures) == 0:
                break
            cv2.setTrackbarPos("Failure", window_name, max(0, min(current_frame, len(failures)) - 1))
        elif key == ord('b'):
            if flag_path is None:
                print("No frame flag file configured. Pass --frame-flag-file to enable frame flagging.")
            else:
                try:
                    _append_flagged_frame(flag_path, zarr_path, frame_idx, roi_idx)
                    print(f"Flagged frame {frame_idx} (ROI {roi_idx}) for keypoint follow-up.")
                except Exception as exc:
                    print(f"Failed to flag frame: {exc}")
        elif key == ord('d'):
            if detect_frame_flag_path is None and detect_flag_path is None:
                print("No detection flag files configured. Pass --detect-flag-file/--detect-frame-flag-file.")
            else:
                try:
                    if detect_frame_flag_path is not None:
                        _append_detection_frame(detect_frame_flag_path, zarr_path, frame_idx)
                    if detect_flag_path is not None:
                        _append_flagged_path(detect_flag_path, zarr_path)
                    if reason_arr is not None:
                        existing_val = reason_arr[roi_idx]
                        existing = "" if existing_val is None else str(existing_val)
                        reason_value = str(_merge_reason(existing, ["detection_issue"]))
                        reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
                    if refined_success_arr is not None:
                        refined_success_arr[roi_idx] = False
                    if confidence_valid_arr is not None:
                        confidence_valid_arr[roi_idx] = False
                    if geometry_valid_arr is not None:
                        geometry_valid_arr[roi_idx] = False
                    if usable_arr is not None:
                        usable_arr[roi_idx] = False
                    if heading_finite_arr is not None:
                        heading_finite_arr[roi_idx] = False
                    if heading_usable_arr is not None:
                        heading_usable_arr[roi_idx] = False

                    failure_pos = current_frame - 1
                    failures = np.delete(failures, failure_pos)
                    if failures.size == 0:
                        print("No failures remaining.")
                        break
                    new_max = max(0, len(failures) - 1)
                    cv2.setTrackbarMax("Failure", window_name, new_max)
                    current_frame = min(current_frame, len(failures))
                    cv2.setTrackbarPos("Failure", window_name, max(0, current_frame - 1))
                except Exception as exc:
                    print(f"Failed to flag detection issue: {exc}")
        elif key == ord('g'):
            show_geometry = 1 - show_geometry
            cv2.setTrackbarPos("Show Geometry", window_name, show_geometry)
        elif key == 83:  # Right arrow
            current_frame = min(len(failures), current_frame + 1)
            cv2.setTrackbarPos("Failure", window_name, max(0, current_frame - 1))
        elif key == 81:  # Left arrow
            current_frame = max(1, current_frame - 1)
            cv2.setTrackbarPos("Failure", window_name, max(0, current_frame - 1))

    cv2.destroyAllWindows()
    refresh_refined_keypoint_heading_fields(refined, root=zarr_root)


def main(zarr_path, start_frame=1):
    global current_frame, roi_thresh, se1_radius, se2_radius
    global min_area, min_triangle_area, max_triangle_area, use_difference, show_geometry
    global min_valid_angle, max_valid_angle
    
    current_frame = start_frame
    
    # Load config for initial values
    config_path = Path("configs/fisheye/default.yaml")
    _load_initial_params(config_path)
    
    # Open zarr
    try:
        zarr_root = zarr.open_group(zarr_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr: {e}")
        return
    
    # Check prerequisites
    if 'crop_runs' not in zarr_root:
        print("Error: Run crop stage first")
        return
    if 'background_runs' not in zarr_root:
        print("Error: Run background stage first")
        return
    
    # Get latest runs
    latest_crop = zarr_root['crop_runs'].attrs['latest']
    latest_bg = zarr_root['background_runs'].attrs['latest']

    crop_group = zarr_root[f'crop_runs/{latest_crop}']
    crop_source_path = crop_group.attrs.get('detection_source_path', 'unknown')
    crop_source_type = crop_group.attrs.get('detection_source_type', 'unknown')

    print(f"Using crop: {latest_crop}")
    print(f"Using background: {latest_bg}")
    print(f"Crop detection source: {crop_source_type} ({crop_source_path})")
    
    # Load background
    bg_group = zarr_root[f'background_runs/{latest_bg}']
    if 'background_full' in bg_group:
        background = bg_group['background_full'][:]
    else:
        print("Warning: No full background found")
        background = None
    
    # Load data
    roi_images = crop_group['roi_images']
    roi_coords = crop_group['roi_coordinates_full']

    # Load per-frame detection counts (prefer crop-derived counts so refined runs are honored)
    n_detections = None
    if 'frame_counts' in crop_group:
        n_detections = crop_group['frame_counts'][:]
        print("Using crop frame_counts for detection availability")
    elif 'frame_indices' in crop_group:
        frame_indices = crop_group['frame_indices'][:]
        num_frames = crop_group.attrs.get('total_frames')
        if num_frames is None:
            if 'raw_video' in zarr_root and 'images_ds' in zarr_root['raw_video']:
                num_frames = zarr_root['raw_video/images_ds'].shape[0]
            elif 'raw_video' in zarr_root and 'images_full' in zarr_root['raw_video']:
                num_frames = zarr_root['raw_video/images_full'].shape[0]
        if num_frames is not None:
            n_detections = np.bincount(frame_indices, minlength=int(num_frames))
            print("Using crop frame_indices for detection availability")

    # Fallback to detect run counts if crop counts are missing
    if n_detections is None:
        if 'detect_runs' not in zarr_root:
            print("Error: No detect_runs found in zarr (needed for detection counts)")
            return
        latest_detect = zarr_root['detect_runs'].attrs.get('latest')
        if not latest_detect:
            print("Error: No latest detect_run found")
            return
        if 'n_detections' in zarr_root[f'detect_runs/{latest_detect}']:
            n_detections = zarr_root[f'detect_runs/{latest_detect}/n_detections'][:]
        elif 'frame_counts' in zarr_root[f'detect_runs/{latest_detect}']:
            n_detections = zarr_root[f'detect_runs/{latest_detect}/frame_counts'][:]
        else:
            print("Error: Neither 'n_detections' nor 'frame_counts' found in detect run")
            return
        print(f"Using detect run counts from {latest_detect}")
    
    total_rois = roi_images.shape[0]
    num_frames = len(n_detections)
    max_dets = int(n_detections.max()) if n_detections.max() > 0 else 1
    
    print(f"\nData: {total_rois} ROIs, {num_frames} frames")
    
    # Create window
    window_name = "Keypoint Detection Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 800)
    
    # Create trackbars
    cv2.createTrackbar("Frame", window_name, current_frame, num_frames, update_frame)
    cv2.createTrackbar("Detection", window_name, 0, max(1, max_dets - 1), update_detection)
    cv2.createTrackbar("Threshold", window_name, roi_thresh, 255, update_roi_thresh)
    cv2.createTrackbar("SE1 Radius", window_name, se1_radius, 10, update_se1)
    cv2.createTrackbar("SE2 Radius", window_name, se2_radius, 10, update_se2)
    # Clamp globals to slider limits before creating trackbars
    min_area = int(min(min_area, MIN_AREA_SLIDER_MAX))
    min_triangle_area = int(min(min_triangle_area, MIN_TRI_AREA_SLIDER_MAX))
    max_triangle_area = int(min(max_triangle_area, MAX_TRI_AREA_SLIDER_MAX))
    min_valid_angle = int(min(min_valid_angle, MIN_ANGLE_SLIDER_MAX))
    max_valid_angle = int(min(max_valid_angle, MAX_ANGLE_SLIDER_MAX))

    cv2.createTrackbar("Min Area", window_name, int(min_area), MIN_AREA_SLIDER_MAX, update_min_area)
    cv2.createTrackbar("Min Angle", window_name, min_valid_angle, MIN_ANGLE_SLIDER_MAX, update_min_valid_angle)
    cv2.createTrackbar("Max Angle", window_name, max_valid_angle, MAX_ANGLE_SLIDER_MAX, update_max_valid_angle)
    cv2.createTrackbar("Min Tri Area", window_name, int(min_triangle_area), MIN_TRI_AREA_SLIDER_MAX, update_min_triangle_area)
    cv2.createTrackbar("Max Tri Area", window_name, int(max_triangle_area), MAX_TRI_AREA_SLIDER_MAX, update_max_triangle_area)
    if background is not None:
        cv2.createTrackbar("Use Diff", window_name, use_difference, 1, update_use_difference)
    cv2.createTrackbar("Show Geometry", window_name, show_geometry, 1, update_show_geometry)
    
    print("\nControls:")
    print("  Arrow keys: Navigate frames")
    print("  s: Save parameters to Zarr metadata")
    print("  d: Toggle difference mode")
    print("  g: Toggle geometry display")
    print("  q/ESC: Quit")
    
    while True:
        frame_idx = current_frame - 1
        n_dets_frame = n_detections[frame_idx] if frame_idx < num_frames else 0
        
        if n_dets_frame > 0:
            # Calculate ROI index
            cumulative_dets = np.cumsum(np.insert(n_detections[:frame_idx+1], 0, 0))
            det_idx = min(current_detection, n_dets_frame - 1)
            roi_idx = cumulative_dets[frame_idx] + det_idx
            
            if roi_idx < total_rois:
                roi_image = roi_images[roi_idx]
                roi_coord = roi_coords[roi_idx]
                background_roi = _extract_background_roi(background, roi_coord, roi_image.shape)
            else:
                roi_image = None
                background_roi = None
        else:
            roi_image = None
            background_roi = None
            roi_idx = -1
        
        # Create dashboard
        params = {
            'roi_thresh': roi_thresh,
            'se1_radius': se1_radius,
            'se2_radius': se2_radius,
            'min_area': min_area,
            'min_valid_angle': min_valid_angle,
            'max_valid_angle': max_valid_angle,
            'use_diff': use_difference,
            'show_geometry': show_geometry,
            'min_triangle_area': min_triangle_area,
            'max_triangle_area': max_triangle_area if max_triangle_area > 0 else None
        }
        
        dashboard = create_keypoint_dashboard(
            roi_image, background_roi, params,
            current_frame, current_detection, roi_idx
        )
        
        cv2.imshow(window_name, dashboard)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Save parameters to Zarr
            try:
                params = {
                    'roi_thresh': roi_thresh,
                    'se1_radius': se1_radius,
                    'se2_radius': se2_radius,
                    'min_area': min_area,
                    'min_valid_angle': min_valid_angle,
                    'max_valid_angle': max_valid_angle,
                    'min_triangle_area': min_triangle_area,
                    'max_triangle_area': max_triangle_area if max_triangle_area > 0 else None,
                    'frame_index': current_frame,
                    'detection_index': current_detection
                }
                
                success, message = save_keypoint_params(zarr_path, params)
                
                if success:
                    print(f"✓ {message}")
                else:
                    print(f"✗ {message}")
                    
            except Exception as e:
                print(f"✗ Error saving: {e}")
        elif key == ord('d'):
            use_difference = 1 - use_difference
            if background is not None:
                cv2.setTrackbarPos("Use Diff", window_name, use_difference)
        elif key == ord('g'):
            show_geometry = 1 - show_geometry
            cv2.setTrackbarPos("Show Geometry", window_name, show_geometry)
        elif key == 83:  # Right arrow
            current_frame = min(num_frames, current_frame + 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
        elif key == 81:  # Left arrow
            current_frame = max(1, current_frame - 1)
            cv2.setTrackbarPos("Frame", window_name, current_frame)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keypoint Detection Tuner")
    parser.add_argument("zarr_path", help="Path to Zarr archive")
    parser.add_argument("start_frame", type=int, nargs='?', default=1,
                       help="Starting frame or failure index (default: 1)")
    args = parser.parse_args()

    main(args.zarr_path, args.start_frame)
