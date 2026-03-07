"""YOLO-based keypoint detection for FishEye Zarr archives.

This mirrors :mod:`fisheye.detection.detect_keypoints_traditional` but uses a
trained Ultralytics YOLO pose model on existing ROI crops. It creates a new
``keypoints_runs`` group without overwriting prior runs and records metadata so
downstream tooling can distinguish between traditional and YOLO-derived
keypoints.
"""

from __future__ import annotations

import math
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
from datetime import datetime, timezone
import time

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from ultralytics import YOLO, __version__ as ultralytics_version

from ..registry.db import RegistryPaths
from ..shared.crop_image_source import CropImageSource
from ..shared.registry_stage_complete import emit_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info
from ..pose.schema import schema_from_package

# Load the traditional 3-point pose schema (swim_bladder + left/right eyes)
TRADITIONAL_POSE_SCHEMA = schema_from_package("traditional_v1")
_KEYPOINT_STEP_NAME = "keypoints"
_KEYPOINT_STATUS_SOURCE = "runtime_keypoints_detect"


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
) -> Tuple[zarr.Group, str]:
    parent = root.require_group("keypoints_runs")
    if run_name:
        if run_name in parent:
            raise ValueError(f"keypoints_runs/{run_name} already exists")
        run_group = parent.create_group(run_name)
        parent.attrs["latest"] = run_name
        console.print(f"Created run group: [cyan]keypoints_runs/{run_name}[/cyan]")
        return run_group, run_name
    return get_run_group(root, "keypoints", console=console, create_new=True)


def _resolve_registry_path(registry: Optional[Path]) -> Optional[Path]:
    if registry is not None:
        return registry.expanduser().resolve()
    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not inferred.exists():
        return None
    return inferred


def _emit_keypoint_step_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    run_name: str,
    method: Optional[str],
    coverage_pct: Optional[float],
    details: Dict[str, object],
    console: Optional[Console],
    registry: Optional[Path],
) -> None:
    registry_path = _resolve_registry_path(registry)
    if registry_path is None:
        return
    emit_stage_completion(
        root,
        zarr_path,
        step_name=_KEYPOINT_STEP_NAME,
        status="ok",
        source=_KEYPOINT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        details_json=details,
        console=console,
        registry=registry_path,
        auto_registry_from_env=False,
        trigger_run_name=run_name,
    )


def _create_output_arrays(group: zarr.Group, total_rois: int, chunk_hint: int) -> Dict[str, zarr.Array]:
    chunk_len = min(max(chunk_hint, 1), total_rois) if total_rois > 0 else 1
    data_chunk = (chunk_len, 3, 2)
    scalar_chunk = (chunk_len,)

    arrays = {
        "keypoints_roi": group.create_array(
            "keypoints_roi",
            shape=(total_rois, 3, 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoints_img": group.create_array(
            "keypoints_img",
            shape=(total_rois, 3, 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoints_norm": group.create_array(
            "keypoints_norm",
            shape=(total_rois, 3, 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "heading": group.create_array(
            "heading",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "confidence": group.create_array(
            "confidence",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoint_confidences": group.create_array(
            "keypoint_confidences",
            shape=(total_rois, 3),
            chunks=(chunk_len, 3),
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "detection_success": group.create_array(
            "detection_success",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "heading_finite": group.create_array(
            "heading_finite",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "heading_usable": group.create_array(
            "heading_usable",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "effective_threshold": group.create_array(
            "effective_threshold",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "effective_se2_radius": group.create_array(
            "effective_se2_radius",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
    }
    return arrays


def _prepare_refined_roi_overrides(
    root: zarr.Group,
    crop_group: zarr.Group,
    total_rois: int,
    roi_shape: Tuple[int, int],
    console: Console,
) -> Optional[Dict[str, Any]]:
    path = crop_group.attrs.get("refined_roi_path")
    if not path:
        return None
    if path not in root:
        console.print(
            f"[yellow]Refined ROI group '{path}' not found; continuing with original crops.[/yellow]"
        )
        return None
    group = root[path]
    required = {"detection_indices", "roi_images", "roi_coordinates_full"}
    if not required.issubset(set(group.keys())):
        console.print(
            f"[yellow]Refined ROI group '{path}' missing {required - set(group.keys())}; skipping overrides.[/yellow]"
        )
        return None

    detection_indices = group["detection_indices"][:].astype(np.int64, copy=False)
    if detection_indices.size == 0:
        return None
    if detection_indices.min(initial=0) < 0 or detection_indices.max(initial=0) >= total_rois:
        raise ValueError("Refined ROI detection indices out of range for current crop run.")

    refined_rois = group["roi_images"][:]
    if refined_rois.shape[1:3] != roi_shape:
        raise ValueError(
            f"Refined ROI shape {refined_rois.shape[1:3]} does not match crop ROI shape {roi_shape}."
        )
    refined_coords = group["roi_coordinates_full"][:]

    override_map = np.full(total_rois, -1, dtype=np.int64)
    override_map[detection_indices] = np.arange(detection_indices.size, dtype=np.int64)

    frame_indices_override = (
        group["frame_indices"][:].astype(np.int64, copy=False)
        if "frame_indices" in group
        else None
    )

    decoder = (
        group.attrs.get("video_device")
        or group.attrs.get("refined_roi_decoder")
        or crop_group.attrs.get("refined_roi_decoder")
    )
    duration = (
        group.attrs.get("duration_seconds")
        or crop_group.attrs.get("refined_roi_generation_duration_seconds")
    )

    return {
        "path": path,
        "count": detection_indices.size,
        "indices": detection_indices,
        "map": override_map,
        "rois": refined_rois,
        "coords": refined_coords,
        "frame_indices": frame_indices_override,
        "decoder": decoder,
        "duration": duration,
    }


def _compute_heading(bladder: np.ndarray, eye_left: np.ndarray, eye_right: np.ndarray) -> float:
    eye_mean = (eye_left + eye_right) / 2.0
    head_vec = eye_mean - bladder
    if np.allclose(head_vec, 0.0):
        return float("nan")
    angle = math.degrees(math.atan2(-head_vec[1], head_vec[0]))
    return float(angle)


def _repeat_to_rgb(batch: np.ndarray) -> List[np.ndarray]:
    if batch.ndim != 3:
        raise ValueError("ROI images should have shape (N, H, W)")
    return [np.repeat(img[..., None], 3, axis=2) for img in batch]


def _select_detection(result) -> Optional[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes is False:
        return None
    if boxes.conf is None or boxes.conf.numel() == 0:
        return 0 if boxes.xyxy.shape[0] > 0 else None
    conf = boxes.conf.detach().cpu().numpy()
    return int(conf.argmax())


def _extract_keypoint_confidences(keypoints, det_idx: int, *, n_keypoints: int = 3) -> np.ndarray:
    """Extract per-keypoint confidences for one detection.

    Returns a float array of length ``n_keypoints``. Missing/conf-unavailable
    values are left as NaN.
    """
    out = np.full(n_keypoints, np.nan, dtype=np.float64)
    kp_conf = getattr(keypoints, "conf", None)
    if kp_conf is None:
        return out
    try:
        conf_np = kp_conf[det_idx].detach().cpu().numpy()
    except Exception:
        return out
    conf_flat = np.asarray(conf_np, dtype=np.float64).reshape(-1)
    if conf_flat.size == 0:
        return out
    take = min(n_keypoints, conf_flat.size)
    out[:take] = conf_flat[:take]
    return out


def detect_keypoints_yolo(
    zarr_path: str,
    model_path: str,
    *,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    verbose: bool = False,
    mask_threshold: float = 0.5,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    registry: Optional[Path] = None,
    console: Optional[Console] = None,
) -> str:
    """Run YOLO pose inference and record outputs in ``keypoints_runs``.

    Returns the name of the created run group.
    """

    console = console or Console()
    console.rule("[bold cyan]YOLO Pose Inference[/bold cyan]")

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = YOLO(str(model_path))
    if device:
        model.to(device)
    try:
        model_device = str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path_resolved = model_path.resolve()

    root = zarr.open(str(zarr_path), mode="a")
    crop_source = CropImageSource.open(
        root,
        crop_run=crop_run,
        zarr_path=zarr_path,
        roi_cache_policy=roi_cache_policy,
        roi_cache_dir=roi_cache_dir,
        console=console,
    )
    crop_group = crop_source.crop_group
    latest_crop = crop_source.crop_run_name

    roi_coords = crop_source.roi_coordinates_full.copy()
    frame_indices = crop_source.frame_indices.astype(np.int64, copy=True)
    total_rois = crop_source.total_rois
    if total_rois == 0:
        console.print("[yellow]No ROIs found in crop run; nothing to process[/yellow]")
        crop_source.close()
        return ""

    roi_h, roi_w = crop_source.roi_shape
    source_detect_run = crop_group.attrs.get("source_detect_run")
    source_refined_run = crop_group.attrs.get("source_refined_run")
    override_data = _prepare_refined_roi_overrides(
        root, crop_group, total_rois, (roi_h, roi_w), console
    )
    override_map: Optional[np.ndarray] = None
    override_rois: Optional[np.ndarray] = None
    if override_data is not None:
        indices = override_data["indices"]
        roi_coords[indices] = override_data["coords"]
        frame_override = override_data["frame_indices"]
        if frame_override is not None:
            frame_indices[indices] = frame_override.astype(frame_indices.dtype, copy=False)
        override_map = override_data["map"]
        override_rois = override_data["rois"]
        console.print(
            f"[cyan]Applying refined ROI overrides:[/cyan] {override_data['count']} detections"
        )

    imgsz = imgsz or max(roi_h, roi_w)

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)
    run_group.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run_group.attrs["keypoint_confidence_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run_group.attrs["pose_schema"] = {
        "name": TRADITIONAL_POSE_SCHEMA.name,
        "nodes": TRADITIONAL_POSE_SCHEMA.node_names,
        "edges": TRADITIONAL_POSE_SCHEMA.edges,
        "metadata": TRADITIONAL_POSE_SCHEMA.metadata,
        "source": "configs/fisheye/pose_schemas/traditional_v1.json"
    }
    root.attrs["current_keypoint_group_path"] = run_group.path

    arrays = _create_output_arrays(run_group, total_rois, chunk_hint=batch_size * 4)

    frame_chunks = (min(batch_size * 4, len(frame_indices)),) if frame_indices.size > 0 else None
    run_group.create_array(
        "frame_indices",
        data=frame_indices,
        chunks=frame_chunks,
        overwrite=True,
    )

    if "detection_indices" in crop_group:
        det_idx = crop_group["detection_indices"][:].astype("i4", copy=False)
        det_chunks = (min(batch_size * 4, det_idx.shape[0]),) if det_idx.size > 0 else None
        run_group.create_array(
            "detection_indices",
            data=det_idx,
            chunks=det_chunks,
            overwrite=True,
        )
    else:
        console.print("[yellow]Crop run missing 'detection_indices'; YOLO keypoint run will omit them.[/yellow]")

    total_frames_attr = (
        root.attrs.get("total_frames")
        or root.attrs.get("n_frames")
        or crop_group.attrs.get("total_frames")
    )
    total_frames: Optional[int] = int(total_frames_attr) if total_frames_attr is not None else None

    try:
        images_full = root["raw_video/images_full"]
        frame_dim, img_h, img_w = images_full.shape
        full_img_shape = (img_h, img_w)
        if total_frames is None:
            total_frames = int(frame_dim)
    except KeyError:
        img_w = (
            root.attrs.get("video_width")
            or root.attrs.get("palette_video_width")
            or root.attrs.get("source_full_width")
            or root.attrs.get("source_video_width")
        )
        img_h = (
            root.attrs.get("video_height")
            or root.attrs.get("palette_video_height")
            or root.attrs.get("source_full_height")
            or root.attrs.get("source_video_height")
        )
        if img_w is None or img_h is None:
            raise ValueError(
                "Unable to determine full-resolution image dimensions. "
                "Expected raw_video/images_full dataset or root attrs 'video_width'/'video_height'."
            )
        full_img_shape = (int(img_h), int(img_w))

    norm_factor = np.array([full_img_shape[1], full_img_shape[0]], dtype="f8")

    if total_frames is None:
        total_frames = int(frame_indices.max() + 1) if frame_indices.size > 0 else 0

    frame_counts_total = (
        np.bincount(frame_indices, minlength=total_frames).astype("i4", copy=False)
        if frame_indices.size > 0
        else np.zeros(total_frames, dtype="i4")
    )
    count_chunks = (min(len(frame_counts_total), batch_size * 4),) if frame_counts_total.size > 0 else None
    run_group.create_array(
        "n_rois",
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True,
    )
    run_group.create_array(
        "frame_counts",
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True,
    )

    crop_detection_source = crop_group.get("detection_source")
    if crop_detection_source is not None and crop_detection_source.shape[0] != total_rois:
        raise ValueError(
            f"Crop run detection_source length {crop_detection_source.shape[0]} does not match ROI count {total_rois}"
        )
    scalar_chunk = arrays["heading"].chunks
    detection_source_dst = run_group.create_array(
        "detection_source",
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )

    success_total = 0
    confidence_accum: List[float] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    start_time = time.time()

    with progress:
        task = progress.add_task("[cyan]Predicting keypoints...", total=total_rois)
        for start in range(0, total_rois, batch_size):
            end = min(start + batch_size, total_rois)
            batch_coords = roi_coords[start:end]
            batch_roi_np = crop_source.read_slice(start, end)
            if override_map is not None and override_rois is not None:
                local_map = override_map[start:end]
                valid = local_map >= 0
                if np.any(valid):
                    batch_roi_np[valid] = override_rois[local_map[valid]]

            rgb_inputs = _repeat_to_rgb(batch_roi_np)
            results = model.predict(
                rgb_inputs,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=verbose,
                stream=False,
            )

            batch_keypoints_roi = np.full((len(rgb_inputs), 3, 2), np.nan, dtype=np.float64)
            batch_keypoints_img = np.full_like(batch_keypoints_roi, np.nan)
            batch_keypoints_norm = np.full_like(batch_keypoints_roi, np.nan)
            batch_heading = np.full(len(rgb_inputs), np.nan, dtype=np.float64)
            batch_conf = np.full(len(rgb_inputs), np.nan, dtype=np.float64)
            batch_keypoint_conf = np.full((len(rgb_inputs), 3), np.nan, dtype=np.float64)
            batch_success = np.zeros(len(rgb_inputs), dtype=bool)

            for i, (res, top_left) in enumerate(zip(results, batch_coords)):
                det_idx = _select_detection(res)
                if det_idx is None:
                    continue
                keypoints = getattr(res, "keypoints", None)
                if keypoints is None or keypoints.xy is None:
                    continue
                kp_xy = keypoints.xy
                if kp_xy is None or kp_xy.ndim != 3 or kp_xy.shape[0] == 0:
                    continue

                kp = kp_xy[det_idx].detach().cpu().numpy()
                if kp.shape[0] < 3:
                    continue

                kp[:, 0] = np.clip(kp[:, 0], 0.0, roi_w - 1)
                kp[:, 1] = np.clip(kp[:, 1], 0.0, roi_h - 1)

                batch_keypoints_roi[i] = kp
                top_left = np.asarray(top_left, dtype=np.float64)
                kp_img = kp + np.array([top_left[0], top_left[1]])
                batch_keypoints_img[i] = kp_img
                batch_keypoints_norm[i] = kp_img / norm_factor
                batch_heading[i] = _compute_heading(kp[0], kp[1], kp[2])
                batch_keypoint_conf[i] = _extract_keypoint_confidences(keypoints, det_idx, n_keypoints=3)

                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.conf is not None and boxes.conf.numel() > 0:
                    det_conf = float(boxes.conf[det_idx].detach().cpu())
                else:
                    kp_conf = getattr(keypoints, "conf", None)
                    det_conf = float(kp_conf[det_idx].detach().cpu().mean()) if kp_conf is not None else 0.0

                batch_conf[i] = det_conf
                batch_success[i] = True
                success_total += 1
                confidence_accum.append(det_conf)

            arrays["keypoints_roi"][start:end] = batch_keypoints_roi
            arrays["keypoints_img"][start:end] = batch_keypoints_img
            arrays["keypoints_norm"][start:end] = batch_keypoints_norm
            arrays["heading"][start:end] = batch_heading
            arrays["confidence"][start:end] = batch_conf
            arrays["keypoint_confidences"][start:end] = batch_keypoint_conf
            arrays["detection_success"][start:end] = batch_success
            arrays["effective_threshold"][start:end] = np.nan
            arrays["effective_se2_radius"][start:end] = np.nan

            if crop_detection_source is not None:
                source_chunk = crop_detection_source[start:end].astype("i1", copy=False)
            else:
                source_chunk = np.zeros(end - start, dtype="i1")
            detection_source_dst[start:end] = source_chunk
            heading_finite_chunk = np.isfinite(batch_heading)
            arrays["heading_finite"][start:end] = heading_finite_chunk
            arrays["heading_usable"][start:end] = np.logical_and(
                np.logical_and(batch_success, source_chunk == 0),
                heading_finite_chunk,
            )

            progress.update(task, advance=end - start)

    total_time = time.time() - start_time
    inference_rate = success_total / total_time if total_time > 0 else 0.0

    success_rate = (success_total / total_rois * 100.0) if total_rois > 0 else 0.0
    failure_total = total_rois - success_total

    if total_frames is not None:
        full_frame_count = int(total_frames)
    elif frame_indices.size > 0:
        full_frame_count = int(frame_indices.max() + 1)
    else:
        full_frame_count = 0
    if success_total > 0:
        success_mask = arrays["detection_success"][:]
        success_counts = np.bincount(frame_indices[success_mask], minlength=full_frame_count).astype("i4", copy=False)
    else:
        success_counts = np.zeros(full_frame_count, dtype="i4")
    success_chunks = (min(len(success_counts), batch_size * 4),) if success_counts.size > 0 else None
    run_group.create_array(
        "n_keypoints",
        data=success_counts,
        chunks=success_chunks,
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})

    run_group.attrs.update({
        "method": "yolo_pose",
        "keypoints_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path_resolved),
        "model_name": model_path.name,
        "ultralytics_version": ultralytics_version,
        "device": model_device,
        "source_crop_run": latest_crop,
        "source_crop_storage_mode": crop_source.storage_mode,
        "source_crop_signature": normalize_attr(crop_group.attrs.get("crop_signature")),
        "source_crop_revision": crop_group.attrs.get("crop_revision"),
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "source_roi_cache_used": bool(crop_source.roi_cache_used),
        "source_detect_run": source_detect_run or "unknown",
        "keypoints_processed": total_rois,
        "success_rate": round(success_rate, 2),
        "parameters": {
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "max_det": max_det,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "device": model_device,
            # Maintained for API compatibility with pipeline/batch configs.
            "mask_threshold": float(mask_threshold),
        },
        "model_names": getattr(model.model, "names", None),
        "summary_statistics": {
            "total_rois": int(total_rois),
            "successful_detections": int(success_total),
            "failed_detections": int(failure_total),
            "success_rate_percent": round(success_rate, 2),
            "mean_confidence": float(np.mean(confidence_accum)) if confidence_accum else 0.0,
        },
        "git_commit": git_info.get("commit_hash", "unknown"),
        "git_branch": git_info.get("branch", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "inference_duration_seconds": float(total_time),
        "inference_poses_per_second": float(inference_rate),
    })
    if crop_source.roi_cache_key:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    if source_refined_run:
        run_group.attrs["source_refined_run"] = source_refined_run
    provenance_record = build_stage_provenance(
        stage="keypoints_detect",
        command=" ".join(sys.argv),
        created_at_utc=str(run_group.attrs.get("keypoints_timestamp_utc")),
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters=dict(run_group.attrs.get("parameters") or {}),
        inputs={
            "source_crop_run": latest_crop,
            "source_crop_storage_mode": crop_source.storage_mode,
            "source_crop_signature": normalize_attr(crop_group.attrs.get("crop_signature")),
            "source_crop_revision": crop_group.attrs.get("crop_revision"),
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "roi_cache_key": crop_source.roi_cache_key,
            "source_detect_run": source_detect_run or "unknown",
            "source_refined_run": source_refined_run,
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
        },
        artifacts={
            "model_path": str(model_path_resolved),
            "model_name": model_path.name,
            "ultralytics_version": ultralytics_version,
            "device": model_device,
        },
    )
    write_stage_provenance(run_group, provenance_record)
    if override_data is not None:
        run_group.attrs["refined_roi_overrides"] = int(override_data["count"])
        run_group.attrs["refined_roi_source"] = override_data["path"]
        if override_data["decoder"]:
            run_group.attrs["refined_roi_decoder"] = override_data["decoder"]
        if override_data["duration"] is not None:
            run_group.attrs["refined_roi_generation_duration_seconds"] = float(override_data["duration"])

    status_details: Dict[str, object] = {
        "reason": "present",
        "run_group": "keypoints_runs",
        "source_crop_run": latest_crop,
        "source_crop_storage_mode": crop_source.storage_mode,
        "source_crop_signature": normalize_attr(crop_group.attrs.get("crop_signature")),
        "source_crop_revision": crop_group.attrs.get("crop_revision"),
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "roi_cache_used": bool(crop_source.roi_cache_used),
        "source_detect_run": source_detect_run or "unknown",
        "total_rois": int(total_rois),
        "successful_detections": int(success_total),
        "failed_detections": int(failure_total),
        "success_rate_percent": round(float(success_rate), 2),
    }
    if source_refined_run:
        status_details["source_refined_run"] = source_refined_run
    if override_data is not None:
        status_details["refined_roi_overrides"] = int(override_data["count"])
        status_details["refined_roi_source"] = str(override_data["path"])

    try:
        _emit_keypoint_step_status(
            root=root,
            zarr_path=zarr_path.resolve(),
            run_name=resolved_run_name,
            method=normalize_attr(run_group.attrs.get("method")) or "yolo_pose",
            coverage_pct=float(success_rate),
            details=status_details,
            console=console,
            registry=registry,
        )
    finally:
        crop_source.close()

    summary_lines = [
        "[green]✓[/green] Pose inference complete",
        "",
        f"[bold]Run:[/bold] keypoints_runs/{resolved_run_name}",
        f"[bold]Total ROIs:[/bold] {total_rois}",
        f"[bold]Successful:[/bold] {success_total} ({success_rate:.2f}%)",
        f"[bold]Failed:[/bold] {failure_total}",
        f"[bold]Model:[/bold] {model_path_resolved}",
        f"[bold]Duration:[/bold] {total_time:.1f}s ({inference_rate:.1f} poses/s)",
    ]
    if override_data is not None:
        summary_lines.append(
            f"[dim]Refined ROI overrides: {override_data['count']} from {override_data['path']}[/dim]"
        )
    completion = Panel(
        "\n".join(summary_lines),
        title="YOLO Pose Inference",
        border_style="green",
    )
    console.print("\n")
    console.print(completion)

    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-based keypoint inference on Palette Zarr crops")
    parser.add_argument("zarr_path", type=str, help="Path to the Palette Zarr archive")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO pose weights (.pt)")
    parser.add_argument("--run-name", help="Optional custom run name for keypoints_runs")
    parser.add_argument("--crop-run", help="Optional crop run override (defaults to latest)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size for YOLO inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. '0' or 'cuda:0')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=1, help="Maximum detections per ROI")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Compatibility parameter recorded in run metadata (not used for pose decoding).",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Optional registry SQLite path.")
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    detect_keypoints_yolo(
        zarr_path=args.zarr_path,
        model_path=args.model,
        run_name=args.run_name,
        crop_run=args.crop_run,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=args.verbose,
        mask_threshold=args.mask_threshold,
        roi_cache_policy=args.roi_cache_policy,
        roi_cache_dir=args.roi_cache_dir,
        registry=args.registry,
    )


__all__ = ["detect_keypoints_yolo", "main"]


if __name__ == "__main__":
    main()
