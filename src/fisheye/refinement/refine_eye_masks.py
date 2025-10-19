# src/fisheye/refinement/refine_eye_masks.py
"""Refine eye-mask segmentation runs using keypoint geometry.

This post-processes an existing ``eye_masks_runs`` entry (typically produced by
YOLO segmentation) and rewrites left/right mask channels so they align with the
keypoint pipeline's anatomical labels. The output mirrors the traditional
segmentation stage – same arrays, same dtypes – but is written as a new run to
preserve provenance.

Usage::

    python -m fisheye.refinement.refine_eye_masks /path/to/archive.zarr \
        --source-run yolo_2025_01_01 \
        --run-name refined_from_yolo
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from skimage import measure

from ..segmentation.eye_segmentation import (
    _extract_contour,
    _feret_mask_from_region,
)
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


@dataclass
class ROIOutput:
    """Container for refined measurements of a single ROI."""

    masks: np.ndarray  # (2, H, W) uint8
    ellipse_params: np.ndarray  # (2, 5) float32
    ellipse_success: np.ndarray  # (2,) bool
    feret_major: np.ndarray  # (2, 4) float32
    feret_minor: np.ndarray  # (2, 4) float32
    feret_roundness: np.ndarray  # (2,) float32
    centroids: np.ndarray  # (2, 2) float32 (x, y) or nan
    contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    eye_separation: float
    used_original_order: bool
    reason: Optional[str]


def _prepare_run_group(root: zarr.Group, run_name: Optional[str], console: Console) -> Tuple[zarr.Group, str]:
    parent = root.require_group("eye_masks_runs")
    if run_name:
        if run_name in parent:
            raise ValueError(f"eye_masks_runs/{run_name} already exists")
        run_group = parent.create_group(run_name)
        parent.attrs["latest"] = run_name
        console.print(f"Created run group: [cyan]eye_masks_runs/{run_name}[/cyan]")
        return run_group, run_name
    return get_run_group(root, "eye_masks", console=console, create_new=True)


def _split_mask_by_keypoints(
    union_mask: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a union mask into left/right halves using closest-keypoint distance."""

    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)

    ys, xs = np.nonzero(union_mask)
    if ys.size == 0:
        return left_mask, right_mask

    if not (np.all(np.isfinite(eye_left)) and np.all(np.isfinite(eye_right))):
        return left_mask, right_mask

    if np.allclose(eye_left, eye_right):
        return left_mask, right_mask

    x_coords = xs.astype(np.float32)
    y_coords = ys.astype(np.float32)

    dist_left = (x_coords - float(eye_left[0])) ** 2 + (y_coords - float(eye_left[1])) ** 2
    dist_right = (x_coords - float(eye_right[0])) ** 2 + (y_coords - float(eye_right[1])) ** 2

    assign_left = dist_left <= dist_right
    left_mask[ys[assign_left], xs[assign_left]] = True
    right_mask[ys[~assign_left], xs[~assign_left]] = True
    return left_mask, right_mask


def _rotation_matrix(angle_degrees: float) -> np.ndarray:
    """Create a 2×2 rotation matrix matching the keypoint pipeline."""
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)


def _split_mask_by_heading(
    union_mask: np.ndarray,
    heading_deg: float,
    center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: split mask using a heading-aligned midpoint plane."""

    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)

    ys, xs = np.nonzero(union_mask)
    if ys.size == 0 or not np.isfinite(heading_deg):
        return left_mask, right_mask

    center = center.astype(np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1) - center
    rot = _rotation_matrix(float(heading_deg))
    rotated = coords @ rot.T

    split_mask = rotated[:, 1] <= 0.0
    left_mask[ys[split_mask], xs[split_mask]] = True
    right_mask[ys[~split_mask], xs[~split_mask]] = True
    return left_mask, right_mask


def _measure_mask(mask: np.ndarray, min_contour_points: int = 5) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, Optional[np.ndarray]]:
    """Extract metrics from a binary mask."""

    if mask.sum() == 0:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        feret_major = np.full(4, np.nan, dtype=np.float32)
        feret_minor = np.full(4, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, feret_major, feret_minor, float("nan"), centroid, None

    region_mask = mask.astype(np.uint8)
    props = measure.regionprops(region_mask)
    if not props:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        feret_major = np.full(4, np.nan, dtype=np.float32)
        feret_minor = np.full(4, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, feret_major, feret_minor, float("nan"), centroid, None

    region = props[0]
    centroid = np.array([float(region.centroid[1]), float(region.centroid[0])], dtype=np.float32)
    ellipse = np.array(
        [
            centroid[0],
            centroid[1],
            float(region.major_axis_length),
            float(region.minor_axis_length),
            float(np.rad2deg(region.orientation)),
        ],
        dtype=np.float32,
    )

    feret_major = np.full(4, np.nan, dtype=np.float32)
    feret_minor = np.full(4, np.nan, dtype=np.float32)
    feret_roundness = float("nan")

    feret_mask, info = _feret_mask_from_region(region_mask, 0.0)
    if feret_mask is not None and info:
        feret_roundness = float(info.get("roundness", float("nan")))
        major_pts = info.get("major_pts")
        minor_pts = info.get("minor_pts")
        if isinstance(major_pts, np.ndarray) and major_pts.shape == (2, 2):
            feret_major = np.array(
                [major_pts[0, 0], major_pts[0, 1], major_pts[1, 0], major_pts[1, 1]],
                dtype=np.float32,
            )
        if isinstance(minor_pts, np.ndarray) and minor_pts.shape == (2, 2):
            feret_minor = np.array(
                [minor_pts[0, 0], minor_pts[0, 1], minor_pts[1, 0], minor_pts[1, 1]],
                dtype=np.float32,
            )

    contour = _extract_contour(mask.astype(float), min_contour_points)
    if contour is not None:
        contour = contour.astype(np.float32)

    return True, ellipse, feret_major, feret_minor, feret_roundness, centroid, contour


def _refine_roi(
    source_masks: np.ndarray,
    keypoints_roi: np.ndarray,
    heading_deg: float,
    success_flag: bool,
) -> ROIOutput:
    """Refine a single ROI's mask assignment."""

    roi_h, roi_w = source_masks.shape[1:]
    masks_out = np.zeros((2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros(2, dtype=bool)
    feret_major = np.full((2, 4), np.nan, dtype=np.float32)
    feret_minor = np.full((2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full(2, np.nan, dtype=np.float32)
    centroids = np.full((2, 2), np.nan, dtype=np.float32)

    def _copy_original(reason: str) -> ROIOutput:
        contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        for eye_idx, mask_raw in enumerate(source_masks):
            mask_bool = mask_raw > 0
            masks_out[eye_idx] = mask_bool.astype(np.uint8)
            (
                success,
                ellipse,
                major,
                minor,
                roundness,
                centroid,
                contour,
            ) = _measure_mask(mask_bool.astype(np.uint8))
            ellipse_success[eye_idx] = success
            ellipse_params[eye_idx] = ellipse
            feret_major[eye_idx] = major
            feret_minor[eye_idx] = minor
            feret_roundness[eye_idx] = roundness
            centroids[eye_idx] = centroid
            if contour is not None:
                if eye_idx == 0:
                    contours = (contour, contours[1])
                else:
                    contours = (contours[0], contour)

        if np.all(ellipse_success):
            separation = float(
                math.hypot(
                    float(centroids[0, 0] - centroids[1, 0]),
                    float(centroids[0, 1] - centroids[1, 1]),
                )
            )
        else:
            separation = float("nan")

        return ROIOutput(
            masks_out,
            ellipse_params,
            ellipse_success,
            feret_major,
            feret_minor,
            feret_roundness,
            centroids,
            contours,
            separation,
            True,
            reason,
        )

    if not success_flag:
        return _copy_original("keypoint_fail")

    union_mask = (source_masks[0] > 0) | (source_masks[1] > 0)
    if union_mask.sum() == 0:
        return _copy_original("empty_union")

    eye_left = keypoints_roi[1]
    eye_right = keypoints_roi[2]

    left_mask, right_mask = _split_mask_by_keypoints(union_mask, eye_left, eye_right)
    used_original_order = False
    reason = None

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        # Try heading-based split around midpoint of keypoints.
        midpoint = np.array([(eye_left[0] + eye_right[0]) / 2.0, (eye_left[1] + eye_right[1]) / 2.0], dtype=np.float32)
        fallback_left, fallback_right = _split_mask_by_heading(union_mask, heading_deg, midpoint)
        if fallback_left.sum() > 0 and fallback_right.sum() > 0:
            left_mask, right_mask = fallback_left, fallback_right
            reason = "heading_split"
        else:
            # Give up and retain original ordering.
            left_mask = source_masks[0] > 0
            right_mask = source_masks[1] > 0
            used_original_order = True
            reason = "original_order"

    masks_out[0] = left_mask.astype(np.uint8)
    masks_out[1] = right_mask.astype(np.uint8)

    contours: List[Optional[np.ndarray]] = [None, None]
    for eye_idx, mask in enumerate((left_mask, right_mask)):
        (
            success,
            ellipse,
            major,
            minor,
            roundness,
            centroid,
            contour,
        ) = _measure_mask(mask.astype(np.uint8))

        ellipse_success[eye_idx] = success
        ellipse_params[eye_idx] = ellipse
        feret_major[eye_idx] = major
        feret_minor[eye_idx] = minor
        feret_roundness[eye_idx] = roundness
        centroids[eye_idx] = centroid
        contours[eye_idx] = contour

    if np.all(ellipse_success):
        eye_separation = float(
            math.hypot(
                float(centroids[0, 0] - centroids[1, 0]),
                float(centroids[0, 1] - centroids[1, 1]),
            )
        )
    else:
        eye_separation = float("nan")

    return ROIOutput(
        masks_out,
        ellipse_params,
        ellipse_success,
        feret_major,
        feret_minor,
        feret_roundness,
        centroids,
        (contours[0], contours[1]),
        eye_separation,
        used_original_order,
        reason,
    )


def refine_eye_masks(
    zarr_path: str,
    source_run: Optional[str] = None,
    run_name: Optional[str] = None,
    *,
    keypoint_run: Optional[str] = None,
    chunk_size: int = 1024,
    console: Optional[Console] = None,
) -> str:
    """Refine an eye-mask run and return the name of the new run."""

    console = console or Console()
    stage_start = time.perf_counter()

    root = zarr.open(zarr_path, mode="a")

    if "eye_masks_runs" not in root:
        raise ValueError("Zarr archive missing eye_masks_runs; run segmentation first.")
    eye_parent = root["eye_masks_runs"]
    src_run_name = source_run or eye_parent.attrs.get("latest")
    if src_run_name is None or src_run_name not in eye_parent:
        raise ValueError("Source eye mask run not found.")
    src_run = eye_parent[src_run_name]

    if "masks_roi" not in src_run:
        raise ValueError(f"eye_masks_runs/{src_run_name} lacks 'masks_roi'.")

    crop_run_name = src_run.attrs.get("source_crop_run") or root.get("crop_runs", {}).attrs.get("latest")
    if crop_run_name is None:
        raise ValueError("Unable to determine crop run (missing attribute 'source_crop_run').")

    kp_parent = root.require_group("keypoints_runs")
    keypoint_run_name = (
        keypoint_run
        or src_run.attrs.get("source_keypoints_run")
        or kp_parent.attrs.get("latest")
    )
    if keypoint_run_name is None or keypoint_run_name not in kp_parent:
        raise ValueError("Keypoint run required for refinement (set --keypoint-run).")
    kp_group = kp_parent[keypoint_run_name]

    required_kp = ["keypoints_roi", "heading", "detection_success"]
    for arr in required_kp:
        if arr not in kp_group:
            raise ValueError(f"Keypoint run '{keypoint_run_name}' missing '{arr}'.")

    masks_src = src_run["masks_roi"]
    total_rois, _, roi_h, roi_w = masks_src.shape

    kp_roi = kp_group["keypoints_roi"]
    headings = kp_group["heading"]
    success_flags = kp_group["detection_success"]

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    masks_out = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    feret_major = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_minor = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full((total_rois, 2), np.nan, dtype=np.float32)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_len = np.zeros((total_rois,), dtype=np.int32)

    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    stats = {
        "total": total_rois,
        "refined": 0,
        "fallback_heading": 0,
        "copied_original": 0,
        "keypoint_fail": 0,
        "empty_union": 0,
        "original_order": 0,
    }

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Refining eye masks[/cyan]", total=total_rois)
        for start in range(0, total_rois, max(1, chunk_size)):
            end = min(total_rois, start + max(1, chunk_size))
            masks_chunk = masks_src[start:end]
            kp_chunk = kp_roi[start:end]
            heading_chunk = headings[start:end]
            success_chunk = success_flags[start:end]

            for local_idx, global_idx in enumerate(range(start, end)):
                result = _refine_roi(
                    np.asarray(masks_chunk[local_idx]),
                    np.asarray(kp_chunk[local_idx]),
                    float(heading_chunk[local_idx]),
                    bool(success_chunk[local_idx]),
                )

                masks_out[global_idx] = result.masks
                ellipse_params[global_idx] = result.ellipse_params
                ellipse_success[global_idx] = result.ellipse_success
                feret_major[global_idx] = result.feret_major
                feret_minor[global_idx] = result.feret_minor
                feret_roundness[global_idx] = result.feret_roundness
                eye_separation[global_idx] = result.eye_separation

                if result.contours[0] is not None:
                    contour = result.contours[0]
                    contour_len = contour.shape[0]
                    left_ptr[global_idx] = left_total
                    left_len[global_idx] = contour_len
                    left_points.append(contour)
                    left_total += contour_len
                if result.contours[1] is not None:
                    contour = result.contours[1]
                    contour_len = contour.shape[0]
                    right_ptr[global_idx] = right_total
                    right_len[global_idx] = contour_len
                    right_points.append(contour)
                    right_total += contour_len

                reason = result.reason or "refined"
                if reason == "heading_split":
                    stats["fallback_heading"] += 1
                    stats["refined"] += 1
                elif reason == "original_order":
                    stats["original_order"] += 1
                    stats["copied_original"] += 1
                elif reason == "keypoint_fail":
                    stats["keypoint_fail"] += 1
                    stats["copied_original"] += 1
                elif reason == "empty_union":
                    stats["empty_union"] += 1
                    stats["copied_original"] += 1
                else:
                    stats["refined"] += 1

            progress.update(task, advance=end - start)

    left_concat = np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)

    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    chunk_rois = min(512, total_rois) if total_rois > 0 else 1

    run_group.create_array(
        "masks_roi",
        data=masks_out,
        chunks=(chunk_rois, 2, roi_h, roi_w),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_params",
        data=ellipse_params,
        chunks=(chunk_rois, 2, 5),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(chunk_rois, 2),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_major",
        data=feret_major,
        chunks=(chunk_rois, 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_minor",
        data=feret_minor,
        chunks=(chunk_rois, 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_roundness",
        data=feret_roundness,
        chunks=(chunk_rois, 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_left_ptr",
        data=left_ptr,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_left_len",
        data=left_len,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_right_ptr",
        data=right_ptr,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_right_len",
        data=right_len,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contours_left",
        data=left_store,
        chunks=(max(1, min(4096, left_store.shape[0])), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_right",
        data=right_store,
        chunks=(max(1, min(4096, right_store.shape[0])), 2),
        overwrite=True,
    )

    source_method = src_run.attrs.get("method", "unknown")
    eye_labels = src_run.attrs.get("eye_labels", ["eye_left", "eye_right"])

    total_eyes = int(ellipse_success.sum())
    successful_pairs = int(np.sum(ellipse_success.all(axis=1)))
    pair_rate = float(successful_pairs / total_rois) if total_rois else float("nan")

    git_info = get_git_info()
    env_info = get_environment_info()
    duration = time.perf_counter() - stage_start

    run_group.attrs.update(
        {
            "method": "refine_eye_masks",
            "source_eye_masks_run": src_run_name,
            "source_eye_masks_method": source_method,
            "source_keypoints_run": keypoint_run_name,
            "source_crop_run": crop_run_name,
            "total_rois": total_rois,
            "successful_eyes": total_eyes,
            "successful_roi_pairs": successful_pairs,
            "successful_roi_pair_rate": pair_rate,
            "refine_stats": stats,
            "duration_seconds": duration,
            "eye_labels": eye_labels,
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )

    console.print(
        f"[green]✓[/green] Refined eye masks saved to [cyan]eye_masks_runs/{resolved_run_name}[/cyan] "
        f"({successful_pairs}/{total_rois} ROI pairs refined)"
    )
    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine eye-mask segmentation outputs.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--source-run",
        help="Eye mask run name to refine (default: latest in eye_masks_runs).",
    )
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run providing headings (default: infer from source or latest).",
    )
    parser.add_argument(
        "--run-name",
        help="Name for the new refined run (default: auto-generated).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Number of ROIs to refine per chunk (default: 1024).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    try:
        refine_eye_masks(
            args.zarr_path,
            source_run=args.source_run,
            run_name=args.run_name,
            keypoint_run=args.keypoint_run,
            chunk_size=args.chunk_size,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]✗[/red] Failed to refine eye masks: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
