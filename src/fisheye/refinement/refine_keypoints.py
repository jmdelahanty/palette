"""
Keypoint refinement pipeline.

Detects and corrects left/right eye swaps in keypoint detections and
records diagnostic metrics without re-running the geometry filters that
already execute during detection.
"""

from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
import argparse
import yaml

import dask
from dask import delayed

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:
    LocalCluster = None  # type: ignore
    Client = None  # type: ignore
    HAVE_DISTRIBUTED = False

from .keypoint_quality import KeypointGeometryMetrics, compute_geometry_metrics
from ..shared.detect_reason_codec import write_reason_columns
from ..utils.system import get_environment_info, get_git_info

REFINED_KEYPOINT_GROUP = "refined_keypoints_runs"
LEGACY_KEYPOINT_GROUP = "keypoints_refined_runs"


def _write_reason_arrays(group: zarr.Group, reason: np.ndarray, chunk_size: int) -> None:
    """Write reason labels in both text and Crimson-compatible byte formats."""
    write_reason_columns(
        group,
        np.asarray(reason, dtype=object),
        chunk_size,
        include_reason_text=True,
        overwrite=True,
    )


@dataclass
class KeypointRefinementParams:
    """Configuration values controlling keypoint refinement."""

    chunk_size: int = 1024
    scheduler: str = "processes"
    num_workers: Optional[int] = None
    memory_limit: Optional[str] = None
    confidence_threshold: float = 0.3
    min_triangle_angle: float = 10.0
    min_triangle_area: float = 100.0
    max_triangle_area: Optional[float] = None

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]],
    ) -> Tuple["KeypointRefinementParams", str]:
        """
        Instantiate parameters from pipeline config subtree.
        Returns (params, source_label).
        """
        source = "defaults"
        params = cls()
        if config:
            if config.get("chunk_size") is not None:
                params.chunk_size = int(config["chunk_size"])
            if config.get("scheduler") is not None:
                params.scheduler = str(config["scheduler"])
            if config.get("num_workers") is not None:
                params.num_workers = int(config["num_workers"])
            if config.get("memory_limit") is not None:
                params.memory_limit = str(config["memory_limit"])
            if config.get("confidence_threshold") is not None:
                params.confidence_threshold = float(config["confidence_threshold"])
            if config.get("min_triangle_angle") is not None:
                params.min_triangle_angle = float(config["min_triangle_angle"])
            if config.get("min_triangle_area") is not None:
                params.min_triangle_area = float(config["min_triangle_area"])
            if config.get("max_triangle_area") is not None:
                params.max_triangle_area = float(config["max_triangle_area"])
            source = config.get("parameter_source", "config")
        return params, source


def _ensure_group(root: zarr.Group, name: str) -> zarr.Group:
    if name in root:
        return root[name]
    return root.create_group(name)


def _copy_array(src: zarr.Array, dst_group: zarr.Group, name: str) -> None:
    """Shallow copy helper for metadata arrays."""
    dst_group.create_array(
        name,
        data=src[:],
        chunks=src.chunks,
        overwrite=True,
    )


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(
    attrs: Dict[str, object],
    parameters: Optional[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "signature_version": 1,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "parameter_source": parameters.get("parameter_source") if parameters else None,
        "parameters_hash": _hash_parameters(parameters),
    }


def _compute_heading_from_points(
    bladder: np.ndarray, eye_left: np.ndarray, eye_right: np.ndarray
) -> float:
    """Compute heading (degrees) from keypoint geometry."""
    eye_mean = (eye_left + eye_right) / 2.0
    head_vec = eye_mean - bladder
    if not np.all(np.isfinite(head_vec)) or np.linalg.norm(head_vec) == 0:
        return float("nan")
    return float(np.rad2deg(np.arctan2(-head_vec[1], head_vec[0])))


def _detect_eye_flip(
    bladder: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    heading_deg: float,
) -> bool:
    """Return True if left/right eye assignments appear swapped."""
    if not np.isfinite(heading_deg):
        heading_deg = _compute_heading_from_points(bladder, eye_left, eye_right)
    if not np.isfinite(heading_deg):
        return False

    theta = np.deg2rad(heading_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    eye_mean = (eye_left + eye_right) / 2.0
    pts = np.vstack([bladder, eye_left, eye_right])
    rotated = (pts - eye_mean) @ rotation.T

    left_y = rotated[1, 1]
    right_y = rotated[2, 1]
    if not (np.isfinite(left_y) and np.isfinite(right_y)):
        return False
    # In the detector, the right eye ends up with the larger y value in this frame.
    return left_y > right_y


def _process_refinement_chunk(
    zarr_path: str,
    keypoint_run: str,
    start: int,
    end: int,
    params_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Process a slice of keypoints and return refinement outputs."""
    root = zarr.open(zarr_path, mode="r")
    kp_source = root[f"keypoints_runs/{keypoint_run}"]

    idx = slice(start, end)

    kp_roi_src = kp_source["keypoints_roi"][idx]
    kp_img_src = kp_source["keypoints_img"][idx]
    kp_norm_src = kp_source["keypoints_norm"][idx]
    heading_src = kp_source["heading"][idx]
    confidence_src = kp_source["confidence"][idx]
    kp_conf_src = (
        kp_source["keypoint_confidences"][idx]
        if "keypoint_confidences" in kp_source
        else None
    )
    thresh_src = (
        kp_source["effective_threshold"][idx]
        if "effective_threshold" in kp_source
        else None
    )
    se2_src = (
        kp_source["effective_se2_radius"][idx]
        if "effective_se2_radius" in kp_source
        else None
    )
    success_chunk = kp_source["detection_success"][idx]

    length = end - start

    roi_out = np.full_like(kp_roi_src, np.nan)
    img_out = np.full_like(kp_img_src, np.nan)
    norm_out = np.full_like(kp_norm_src, np.nan)
    heading_out = np.full_like(heading_src, np.nan)
    confidence_out = np.full_like(confidence_src, np.nan)
    kp_conf_out = (
        np.full_like(kp_conf_src, np.nan) if kp_conf_src is not None else None
    )
    thresh_out = (
        np.full_like(thresh_src, np.nan) if thresh_src is not None else None
    )
    se2_out = np.full_like(se2_src, np.nan) if se2_src is not None else None

    area_out = np.full(length, np.nan, dtype=np.float64)
    min_angle_out = np.full(length, np.nan, dtype=np.float64)
    triangle_angles_out = np.full((length, 3), np.nan, dtype=np.float64)
    quality_out = np.zeros(length, dtype=np.int8)
    refined_success_out = np.zeros(length, dtype=bool)
    flip_flags_out = np.zeros(length, dtype=bool)
    confidence_valid_out = np.zeros(length, dtype=bool)
    geometry_valid_out = np.zeros(length, dtype=bool)
    usable_out = np.zeros(length, dtype=bool)
    reason_out = np.full(length, "", dtype=object)

    confidence_threshold = float(params_dict.get("confidence_threshold", 0.3))
    min_triangle_angle = float(params_dict.get("min_triangle_angle", 10.0))
    min_triangle_area = float(params_dict.get("min_triangle_area", 100.0))
    max_tri_val = params_dict.get("max_triangle_area")
    try:
        max_triangle_area = float(max_tri_val) if max_tri_val is not None else None
    except (TypeError, ValueError):
        max_triangle_area = None

    stats = {
        "refined_success": 0,
        "source_success": int(np.sum(success_chunk)),
        "source_failures": int(len(success_chunk) - int(np.sum(success_chunk))),
        "flips_corrected": 0,
        "low_confidence": 0,
        "confidence_missing": 0,
        "geometry_issues": 0,
        "clean": 0,
        "usable": 0,
    }

    for i in range(length):
        if not success_chunk[i]:
            quality_out[i] = 4  # source detection failed
            reason_out[i] = "detection_failed"
            continue

        metrics: KeypointGeometryMetrics = compute_geometry_metrics(kp_roi_src[i])
        area_out[i] = metrics.area
        min_angle_out[i] = metrics.min_angle
        triangle_angles_out[i] = metrics.angles

        roi_out[i] = kp_roi_src[i]
        img_out[i] = kp_img_src[i]
        norm_out[i] = kp_norm_src[i]
        confidence_out[i] = confidence_src[i]
        if kp_conf_out is not None and kp_conf_src is not None:
            kp_conf_out[i] = kp_conf_src[i]

        heading_val = heading_src[i]
        heading_val = _compute_heading_from_points(
            roi_out[i][0], roi_out[i][1], roi_out[i][2]
        ) if not np.isfinite(heading_val) else heading_val
        heading_out[i] = heading_val

        if thresh_out is not None and thresh_src is not None:
            thresh_out[i] = thresh_src[i]
        if se2_out is not None and se2_src is not None:
            se2_out[i] = se2_src[i]

        flip_detected = _detect_eye_flip(
            roi_out[i][0], roi_out[i][1], roi_out[i][2], heading_val
        )
        if flip_detected:
            # Swap left/right eyes in all coordinate spaces
            roi_out[i][[1, 2]] = roi_out[i][[2, 1]]
            img_out[i][[1, 2]] = img_out[i][[2, 1]]
            norm_out[i][[1, 2]] = norm_out[i][[2, 1]]
            if kp_conf_out is not None:
                kp_conf_out[i][[1, 2]] = kp_conf_out[i][[2, 1]]
            flip_flags_out[i] = True
            quality_out[i] = 6  # Flag flip correction
            stats["flips_corrected"] += 1
        else:
            quality_out[i] = 0

        conf_missing = False
        conf_ok = False
        if kp_conf_out is None:
            conf_missing = True
        else:
            conf_vals = kp_conf_out[i]
            if not np.all(np.isfinite(conf_vals)):
                conf_missing = True
            else:
                conf_ok = bool(np.all(conf_vals >= confidence_threshold))
        confidence_valid_out[i] = conf_ok

        max_ok = max_triangle_area is None or metrics.area <= max_triangle_area
        geom_ok = bool(
            np.isfinite(metrics.min_angle)
            and np.isfinite(metrics.area)
            and metrics.min_angle >= min_triangle_angle
            and metrics.area >= min_triangle_area
            and max_ok
        )
        geometry_valid_out[i] = geom_ok

        tags: List[str] = []
        if flip_detected:
            tags.append("flip_corrected")
        if conf_missing:
            tags.append("confidence_missing")
            stats["confidence_missing"] += 1
        elif not conf_ok:
            tags.append("low_confidence")
            stats["low_confidence"] += 1
        if not geom_ok:
            tags.append("geometry_issue")
            stats["geometry_issues"] += 1

        if tags:
            reason_out[i] = "|".join(tags)
        else:
            reason_out[i] = "clean"
            stats["clean"] += 1

        usable = conf_ok and geom_ok
        usable_out[i] = usable
        if usable:
            stats["usable"] += 1

        refined_success_out[i] = True
        stats["refined_success"] += 1

    return {
        "start": start,
        "end": end,
        "quality": quality_out,
        "refined_success": refined_success_out,
        "roi": roi_out,
        "img": img_out,
        "norm": norm_out,
        "heading": heading_out,
        "confidence": confidence_out,
        "kp_conf": kp_conf_out,
        "thresh": thresh_out,
        "se2": se2_out,
        "flip_flags": flip_flags_out,
        "area": area_out,
        "min_angle": min_angle_out,
        "triangle_angles": triangle_angles_out,
        "confidence_valid": confidence_valid_out,
        "geometry_valid": geometry_valid_out,
        "usable": usable_out,
        "reason": reason_out,
        "stats": stats,
    }


def create_refined_keypoint_run(
    zarr_path: str,
    keypoint_run: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    *,
    command: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> str:
    """
    Create a refined keypoint run with geometry-based validation.

    Returns name of the created run.
    """
    if console is None:
        console = Console()

    console.rule("[bold]Keypoint Refinement[/bold]")
    start_time = time.perf_counter()

    root = zarr.open(zarr_path, mode="a")
    if "keypoints_runs" not in root or root["keypoints_runs"].attrs.get("latest") is None:
        raise RuntimeError("No keypoint runs found. Run keypoint detection first.")

    if keypoint_run is None:
        keypoint_run = root["keypoints_runs"].attrs["latest"]

    kp_source = root[f"keypoints_runs/{keypoint_run}"]
    console.print(f"Source keypoint run: [cyan]{keypoint_run}[/cyan]")

    params_config = (config or {}).get("refine_keypoints", {}) if config else {}
    params, param_source = KeypointRefinementParams.from_config(params_config)

    console.print(f"Parameter source: [cyan]{param_source}[/cyan]")
    console.print(f"  Chunk size: {params.chunk_size}")
    console.print(f"  Scheduler: {params.scheduler}")
    if params.num_workers is not None:
        console.print(f"  Num workers: {params.num_workers}")
    console.print(f"  Confidence threshold: {params.confidence_threshold}")
    console.print(f"  Min triangle angle: {params.min_triangle_angle}")
    console.print(f"  Min triangle area: {params.min_triangle_area}")
    if params.max_triangle_area is not None:
        console.print(f"  Max triangle area: {params.max_triangle_area}")

    total_rois = kp_source["keypoints_roi"].shape[0]
    console.print(f"Total ROI keypoints: {total_rois}")
    if total_rois == 0:
        raise RuntimeError("Keypoint run contains zero ROIs; nothing to refine.")

    # Prepare destination group
    kp_refined_root = _ensure_group(root, REFINED_KEYPOINT_GROUP)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"refined_keypoints_{timestamp}"
    kp_refined = kp_refined_root.create_group(run_name)
    kp_refined_root.attrs["latest"] = run_name

    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    source_crop_run = kp_source.attrs.get("source_crop_run")
    source_detect_run = kp_source.attrs.get("source_detect_run")

    kp_refined.attrs.update(
        {
            "source_keypoints_run": keypoint_run,
            "chunk_size": params.chunk_size,
            "scheduler": params.scheduler,
            "num_workers": params.num_workers,
            "memory_limit": params.memory_limit,
            "refinement_role": "left_right_eye_check",
            "created_utc": created_timestamp,
        }
    )
    if source_crop_run:
        kp_refined.attrs["source_crop_run"] = source_crop_run
    if source_detect_run:
        kp_refined.attrs["source_detect_run"] = source_detect_run

    if "keypoint_labels" in kp_source.attrs:
        kp_refined.attrs["keypoint_labels"] = kp_source.attrs["keypoint_labels"]

    # Copy pose schema if present in source run
    if "pose_schema" in kp_source.attrs:
        kp_refined.attrs["pose_schema"] = kp_source.attrs["pose_schema"]

    # Copy metadata arrays (frame indices, counts, etc.)
    for meta_name in ("frame_indices", "n_rois", "frame_counts", "detection_indices"):
        if meta_name in kp_source:
            _copy_array(kp_source[meta_name], kp_refined, meta_name)

    heading_chunks = kp_source["heading"].chunks or (min(1024, total_rois),)
    if "detection_source" in kp_source:
        source_detection_array = kp_source["detection_source"]
        detection_source_values = source_detection_array[:].astype("i1", copy=False)
        det_chunks = source_detection_array.chunks or heading_chunks
    else:
        detection_source_values = np.zeros(total_rois, dtype="i1")
        det_chunks = heading_chunks
    detection_source_dst = kp_refined.create_array(
        "detection_source",
        shape=(total_rois,),
        chunks=det_chunks,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )
    detection_source_dst[:] = detection_source_values
    kp_refined.create_array(
        "retune_id",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="i4",
        fill_value=-1,
        overwrite=True,
    )

    # Prepare output arrays
    kp_roi_dst = kp_refined.create_array(
        "keypoints_roi",
        shape=kp_source["keypoints_roi"].shape,
        chunks=kp_source["keypoints_roi"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_img_dst = kp_refined.create_array(
        "keypoints_img",
        shape=kp_source["keypoints_img"].shape,
        chunks=kp_source["keypoints_img"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_norm_dst = kp_refined.create_array(
        "keypoints_norm",
        shape=kp_source["keypoints_norm"].shape,
        chunks=kp_source["keypoints_norm"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )

    heading_dst = kp_refined.create_array(
        "heading",
        shape=kp_source["heading"].shape,
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    confidence_dst = kp_refined.create_array(
        "confidence",
        shape=kp_source["confidence"].shape,
        chunks=kp_source["confidence"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    kp_conf_dst = None
    if "keypoint_confidences" in kp_source:
        kp_conf_dst = kp_refined.create_array(
            "keypoint_confidences",
            shape=kp_source["keypoint_confidences"].shape,
            chunks=kp_source["keypoint_confidences"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )
    thresh_dst = None
    if "effective_threshold" in kp_source:
        thresh_dst = kp_refined.create_array(
            "effective_threshold",
            shape=kp_source["effective_threshold"].shape,
            chunks=kp_source["effective_threshold"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )
    se2_dst = None
    if "effective_se2_radius" in kp_source:
        se2_dst = kp_refined.create_array(
            "effective_se2_radius",
            shape=kp_source["effective_se2_radius"].shape,
            chunks=kp_source["effective_se2_radius"].chunks,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        )

    # Diagnostics
    geom_area_dst = kp_refined.create_array(
        "triangle_area",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    geom_min_angle_dst = kp_refined.create_array(
        "min_angle",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )
    geom_angles_dst = kp_refined.create_array(
        "triangle_angles",
        shape=(total_rois, 3),
        chunks=(
            kp_source["heading"].chunks[0]
            if kp_source["heading"].chunks
            else min(1024, total_rois),
            3,
        ),
        dtype="f8",
        fill_value=np.nan,
        overwrite=True,
    )

    quality_dst = kp_refined.create_array(
        "quality_labels",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )
    refined_success_dst = kp_refined.create_array(
        "refined_success",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    confidence_valid_dst = kp_refined.create_array(
        "confidence_valid",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    geometry_valid_dst = kp_refined.create_array(
        "geometry_valid",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    usable_dst = kp_refined.create_array(
        "usable_keypoints",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    reason_chunk = int(heading_chunks[0]) if heading_chunks else max(1, min(1024, total_rois))
    reason_values = np.full(int(total_rois), "", dtype=object)
    source_success = kp_source["detection_success"][:]
    kp_refined.create_array(
        "source_success",
        data=source_success,
        chunks=kp_source["detection_success"].chunks,
        overwrite=True,
    )
    flip_dst = kp_refined.create_array(
        "flip_corrected",
        shape=(total_rois,),
        chunks=kp_source["heading"].chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    heading_finite_dst = kp_refined.create_array(
        "heading_finite",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )
    heading_usable_dst = kp_refined.create_array(
        "heading_usable",
        shape=(total_rois,),
        chunks=heading_chunks,
        dtype="bool",
        fill_value=False,
        overwrite=True,
    )

    stats = {
        "total": int(total_rois),
        "source_success": int(np.sum(source_success)),
        "refined_success": 0,
        "source_failures": int(total_rois - np.sum(source_success)),
        "flips_corrected": 0,
        "low_confidence": 0,
        "confidence_missing": 0,
        "geometry_issues": 0,
        "clean": 0,
        "usable": 0,
    }

    console.print("\nEvaluating keypoints for eye flips...")
    console.print(
        f"  Scheduler: {params.scheduler}"
        + (f" | Workers: {params.num_workers or 'default'}" if params.scheduler != "single-threaded" else "")
    )

    chunk_len = max(1, params.chunk_size)
    chunk_tasks = []
    for start in range(0, total_rois, chunk_len):
        end = min(start + chunk_len, total_rois)
        chunk_tasks.append(
            delayed(_process_refinement_chunk)(
                zarr_path,
                keypoint_run,
                int(start),
                int(end),
                {
                    "confidence_threshold": params.confidence_threshold,
                    "min_triangle_angle": params.min_triangle_angle,
                    "min_triangle_area": params.min_triangle_area,
                    "max_triangle_area": params.max_triangle_area,
                },
            )
        )

    cluster = None
    client = None
    results_list: List[Dict[str, Any]] = []
    if chunk_tasks:
        try:
            if params.scheduler == "distributed":
                if not HAVE_DISTRIBUTED:
                    raise RuntimeError(
                        "Dask distributed is not available. Install dask[distributed] "
                        "or choose a different scheduler (e.g. 'processes' or 'threads')."
                    )
                cluster_kwargs: Dict[str, Any] = {}
                if params.num_workers is not None:
                    cluster_kwargs["n_workers"] = params.num_workers
                if params.memory_limit is not None:
                    cluster_kwargs["memory_limit"] = params.memory_limit
                cluster = LocalCluster(**cluster_kwargs)
                client = Client(cluster)
                futures = client.compute(chunk_tasks)
                results_list = list(client.gather(futures))
            else:
                compute_kwargs: Dict[str, Any] = {"scheduler": params.scheduler}
                if params.num_workers is not None:
                    compute_kwargs["num_workers"] = params.num_workers
                compute_result = dask.compute(*chunk_tasks, **compute_kwargs)
                results_list = list(compute_result) if isinstance(compute_result, tuple) else list(compute_result)
        finally:
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

    for result in results_list:
        start = result["start"]
        end = result["end"]
        idx = slice(start, end)

        kp_roi_dst[idx] = result["roi"]
        kp_img_dst[idx] = result["img"]
        kp_norm_dst[idx] = result["norm"]
        heading_dst[idx] = result["heading"]
        confidence_dst[idx] = result["confidence"]
        if kp_conf_dst is not None and result.get("kp_conf") is not None:
            kp_conf_dst[idx] = result["kp_conf"]
        if thresh_dst is not None and result["thresh"] is not None:
            thresh_dst[idx] = result["thresh"]
        if se2_dst is not None and result["se2"] is not None:
            se2_dst[idx] = result["se2"]
        geom_area_dst[idx] = result["area"]
        geom_min_angle_dst[idx] = result["min_angle"]
        geom_angles_dst[idx] = result["triangle_angles"]
        quality_dst[idx] = result["quality"]
        refined_success_dst[idx] = result["refined_success"]
        confidence_valid_dst[idx] = result["confidence_valid"]
        geometry_valid_dst[idx] = result["geometry_valid"]
        usable_dst[idx] = result["usable"]
        reason_values[idx] = np.asarray(result["reason"], dtype=object)
        flip_dst[idx] = result["flip_flags"]

        stats["refined_success"] += result["stats"]["refined_success"]
        stats["flips_corrected"] += result["stats"]["flips_corrected"]
        stats["low_confidence"] += result["stats"]["low_confidence"]
        stats["confidence_missing"] += result["stats"]["confidence_missing"]
        stats["geometry_issues"] += result["stats"]["geometry_issues"]
        stats["clean"] += result["stats"]["clean"]
        stats["usable"] += result["stats"]["usable"]

    _write_reason_arrays(kp_refined, reason_values, reason_chunk)

    duration = time.perf_counter() - start_time

    pass_rate = stats["refined_success"] / stats["total"] * 100 if stats["total"] else 0.0
    kp_refined.attrs["summary_statistics"] = {
        "total_rois": stats["total"],
        "source_success": stats["source_success"],
        "source_failures": stats["source_failures"],
        "refined_success": stats["refined_success"],
        "flips_corrected": stats["flips_corrected"],
        "low_confidence": stats["low_confidence"],
        "confidence_missing": stats["confidence_missing"],
        "geometry_issues": stats["geometry_issues"],
        "clean": stats["clean"],
        "usable_keypoints": stats["usable"],
        "confidence_threshold": params.confidence_threshold,
        "min_triangle_angle": params.min_triangle_angle,
        "min_triangle_area": params.min_triangle_area,
        "max_triangle_area": params.max_triangle_area,
        "pass_rate_percent": pass_rate,
        "duration_seconds": duration,
    }

    failure_indices = np.where(~source_success)[0].astype("i4", copy=False)
    failure_chunk = (max(1, min(10000, failure_indices.size)),)
    kp_refined.create_array(
        "failure_indices",
        data=failure_indices,
        chunks=failure_chunk,
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    environment_summary = env_info.get("environment")

    scheduler_info: Optional[Dict[str, object]] = None
    if params.scheduler:
        scheduler_info = {"type": params.scheduler}
        if params.num_workers is not None:
            scheduler_info["num_workers"] = int(params.num_workers)
        if params.memory_limit is not None:
            scheduler_info["memory_limit"] = params.memory_limit

    platform_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
        "machine": env_info["platform"].get("machine", "unknown"),
    }

    parameters_info = {
        "chunk_size": params.chunk_size,
        "scheduler": params.scheduler,
        "num_workers": params.num_workers,
        "memory_limit": params.memory_limit,
        "parameter_source": param_source,
        "confidence_threshold": params.confidence_threshold,
        "min_triangle_angle": params.min_triangle_angle,
        "min_triangle_area": params.min_triangle_area,
        "max_triangle_area": params.max_triangle_area,
    }
    kp_refined.attrs["parameter_source"] = param_source
    kp_refined.attrs["parameters"] = parameters_info

    artifact_keys = [
        "model_checkpoint",
        "model_name",
        "model_version",
        "source_checkpoint",
        "inference_device",
        "inference_batch_size",
        "dataset_meta",
    ]
    artifact_info = {key: kp_source.attrs[key] for key in artifact_keys if key in kp_source.attrs}

    frame_source = "zarr"
    source_video_path = root.attrs.get("source_video_path")
    if source_crop_run:
        crop_group = root.get(f"crop_runs/{source_crop_run}")
        if crop_group is not None:
            frame_source = crop_group.attrs.get("video_source_type", frame_source)
            source_video_path = crop_group.attrs.get("video_source_path") or source_video_path

    provenance_record = {
        "stage": "refine_keypoints",
        "command": command or "unknown",
        "created_at_utc": created_timestamp,
        "version": git_info.get("short_hash") or git_info.get("commit_hash"),
        "git": {
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        "environment": environment_summary,
        "platform": platform_info,
        "scheduler": scheduler_info,
        "parameters": parameters_info,
        "inputs": {
            "keypoints_run": keypoint_run,
            "source_crop_run": source_crop_run,
            "frame_source": frame_source,
            "source_video_path": source_video_path,
        },
        "artifacts": artifact_info,
    }
    provenance_record = {k: v for k, v in provenance_record.items() if v is not None}

    kp_refined.attrs["provenance"] = provenance_record
    kp_refined.attrs["keypoint_signature"] = _build_keypoint_signature(
        kp_refined.attrs, parameters_info
    )

    refined_success_values = refined_success_dst[:]
    heading_values = np.asarray(heading_dst[:], dtype=np.float64)
    heading_finite_values = np.isfinite(heading_values)
    heading_usable_values = refined_success_values.astype(bool)
    if detection_source_values.size:
        heading_usable_values &= (detection_source_values == 0)
    heading_usable_values &= heading_finite_values
    heading_finite_dst[:] = heading_finite_values
    heading_usable_dst[:] = heading_usable_values

    report_lines = [
        "[bold]Results[/bold]",
        f"  Total ROIs: {stats['total']}",
        f"  Source success: {stats['source_success']}",
        f"  Refined success: {stats['refined_success']}",
        f"  Source failures: {stats['source_failures']}",
        f"  Flips corrected: {stats['flips_corrected']}",
        f"  Low confidence: {stats['low_confidence']}",
        f"  Confidence missing: {stats['confidence_missing']}",
        f"  Geometry issues: {stats['geometry_issues']}",
        f"  Clean: {stats['clean']}",
        f"  Usable keypoints: {stats['usable']}",
        f"  Pass rate: {pass_rate:.1f}%",
        f"  Duration: {duration:.2f}s",
    ]

    console.print("\n".join(report_lines))
    console.print(f"[green]✓[/green] Saved refined keypoints run: [cyan]{run_name}[/cyan]")

    return run_name


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Refine keypoint detections to correct eye swaps and produce diagnostics."
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run to refine (defaults to latest in keypoints_runs).",
    )
    parser.add_argument(
        "--config",
        default="configs/fisheye/default.yaml",
        help="Pipeline configuration file (default: configs/fisheye/default.yaml).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override refinement chunk size.",
    )
    parser.add_argument(
        "--scheduler",
        choices={"processes", "threads", "distributed"},
        help="Override Dask scheduler.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override number of workers.",
    )
    parser.add_argument(
        "--memory-limit",
        help="Override Dask worker memory limit (e.g. '32GB').",
    )
    parser.add_argument(
        "--review-failures",
        action="store_true",
        help="Launch the manual keypoint failure review tool after refinement.",
    )

    args = parser.parse_args(argv)
    console = Console()

    config: Dict[str, Any] = {}
    overrides_applied = False

    cfg_path = Path(args.config) if args.config else None
    if cfg_path and cfg_path.exists():
        with cfg_path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                config = loaded
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Config file '{cfg_path}' did not contain a mapping; ignoring."
                )
    elif cfg_path:
        console.print(
            f"[yellow]Warning:[/yellow] Config file '{cfg_path}' not found; using defaults."
        )

    refine_cfg = config.setdefault("refine_keypoints", {})

    if args.chunk_size is not None:
        refine_cfg["chunk_size"] = args.chunk_size
        overrides_applied = True
    if args.scheduler is not None:
        refine_cfg["scheduler"] = args.scheduler
        overrides_applied = True
    if args.num_workers is not None:
        refine_cfg["num_workers"] = args.num_workers
        overrides_applied = True
    if args.memory_limit is not None:
        refine_cfg["memory_limit"] = args.memory_limit
        overrides_applied = True

    config_to_use = config if (config or overrides_applied) else None

    run_name = create_refined_keypoint_run(
        zarr_path=args.zarr_path,
        keypoint_run=args.keypoint_run,
        config=config_to_use,
        console=console,
    )

    console.print(
        f"[green]✓[/green] Refined keypoints written to "
        f"[bold]refined_keypoints_runs/{run_name}[/bold]"
    )

    if args.review_failures:
        try:
            from ..tune.keypoint_review import run_manual_review
        except Exception as exc:  # pragma: no cover - optional UI dependency
            console.print(f"[yellow]Warning:[/yellow] Review tool unavailable: {exc}")
            return
        console.print("[blue]Launching keypoint review (manual)...[/blue]")
        run_manual_review(args.zarr_path, refined_run=run_name)


if __name__ == "__main__":  # pragma: no cover
    main()
