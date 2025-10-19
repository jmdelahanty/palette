"""YOLO-based eye segmentation for Palette ROI crops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import zarr
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from skimage import measure

from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


@dataclass
class EyeCandidate:
    mask: np.ndarray
    centroid_xy: Tuple[float, float]
    ellipse_row: np.ndarray
    contour_xy: Optional[np.ndarray]
    area: float


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
) -> Tuple[zarr.Group, str]:
    parent = root.require_group("eye_masks_runs")
    if run_name:
        if run_name in parent:
            raise ValueError(f"eye_masks_runs/{run_name} already exists")
        run_group = parent.create_group(run_name)
        parent.attrs["latest"] = run_name
        console.print(f"Created run group: [cyan]eye_masks_runs/{run_name}[/cyan]")
        return run_group, run_name
    return get_run_group(root, "eye_masks", console=console, create_new=True)


def _repeat_to_rgb(batch: np.ndarray) -> List[np.ndarray]:
    if batch.ndim != 3:
        raise ValueError("ROI images expected shape (N, H, W)")
    return [np.repeat(img[..., None], 3, axis=2) for img in batch]


def _candidate_from_mask(mask: np.ndarray) -> Optional[EyeCandidate]:
    if mask.sum() <= 0:
        return None
    props = measure.regionprops(mask.astype(np.uint8))
    if not props:
        return None
    region = max(props, key=lambda p: p.area)
    centroid = (float(region.centroid[1]), float(region.centroid[0]))
    ellipse_row = np.array(
        [
            centroid[0],
            centroid[1],
            float(region.major_axis_length),
            float(region.minor_axis_length),
            float(math.degrees(region.orientation)),
        ],
        dtype=np.float32,
    )
    contour = measure.find_contours(mask.astype(float), 0.5)
    contour_xy: Optional[np.ndarray] = None
    if contour:
        best = max(contour, key=lambda arr: arr.shape[0])
        contour_xy = best[:, ::-1].astype(np.float32)
    return EyeCandidate(
        mask=mask,
        centroid_xy=centroid,
        ellipse_row=ellipse_row,
        contour_xy=contour_xy,
        area=float(region.area),
    )


def _extract_candidates(
    result,
    roi_shape: Tuple[int, int],
    mask_threshold: float,
) -> List[EyeCandidate]:
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return []

    candidate_list: List[EyeCandidate] = []
    data = masks.data.detach().cpu().numpy()
    for mask in data:
        resized = cv2.resize(mask, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_LINEAR)
        binary = resized >= mask_threshold
        candidate = _candidate_from_mask(binary.astype(np.uint8))
        if candidate is not None:
            candidate_list.append(candidate)

    candidate_list.sort(key=lambda c: (c.area, c.centroid_xy[0]), reverse=True)
    return candidate_list[:2]


def _assign_left_right(candidates: Sequence[EyeCandidate]) -> List[Optional[EyeCandidate]]:
    if not candidates:
        return [None, None]
    ordered = sorted(candidates, key=lambda c: c.centroid_xy[0])
    result: List[Optional[EyeCandidate]] = [None, None]
    if ordered:
        result[0] = ordered[0]
    if len(ordered) > 1:
        result[1] = ordered[1]
    return result


def segment_eye_masks_yolo(
    zarr_path: str,
    model_path: str,
    *,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 4,
    mask_threshold: float = 0.5,
    verbose: bool = False,
    console: Optional[Console] = None,
) -> str:
    """Run a YOLO segmentation model to generate binary eye masks."""

    from ultralytics import YOLO, __version__ as ultralytics_version

    console = console or Console()
    console.rule("[bold cyan]YOLO Eye Segmentation[/bold cyan]")

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
    if "crop_runs" not in root:
        raise ValueError("Zarr archive missing crop_runs; run cropping first")

    crop_run_name = crop_run or root["crop_runs"].attrs.get("latest")
    if crop_run_name is None:
        raise ValueError("No crop run available; cannot perform eye segmentation")
    crop_group = root[f"crop_runs/{crop_run_name}"]
    roi_images = crop_group["roi_images"]

    total_rois = int(roi_images.shape[0])
    if total_rois == 0:
        console.print("[yellow]No ROIs available; nothing to segment[/yellow]")
        return ""

    roi_h, roi_w = int(roi_images.shape[1]), int(roi_images.shape[2])

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    feret_axes_major = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_axes_minor = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full((total_rois, 2), np.nan, dtype=np.float32)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_len = np.zeros((total_rois,), dtype=np.int32)
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    successful_pairs = 0

    timer = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with timer:
        task_id = timer.add_task("[cyan]Running YOLO segmentation...[/cyan]", total=total_rois)
        for start in range(0, total_rois, batch_size):
            end = min(start + batch_size, total_rois)
            batch = np.asarray(roi_images[start:end])
            rgb_batch = _repeat_to_rgb(batch)

            results = model(
                rgb_batch,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=verbose,
            )

            for idx, result in enumerate(results):
                global_idx = start + idx
                candidates = _extract_candidates(result, (roi_h, roi_w), mask_threshold)
                left_right = _assign_left_right(candidates)

                centroids: List[Optional[Tuple[float, float]]] = [None, None]

                for eye_idx, candidate in enumerate(left_right):
                    if candidate is None:
                        continue
                    bin_mask = candidate.mask.astype(np.uint8)
                    masks[global_idx, eye_idx] = bin_mask
                    ellipse_params[global_idx, eye_idx] = candidate.ellipse_row
                    ellipse_success[global_idx, eye_idx] = True
                    centroids[eye_idx] = candidate.centroid_xy
                    if candidate.contour_xy is not None:
                        contour = candidate.contour_xy
                        if eye_idx == 0:
                            left_ptr[global_idx] = left_total
                            left_len[global_idx] = contour.shape[0]
                            left_points.append(contour)
                            left_total += contour.shape[0]
                        else:
                            right_ptr[global_idx] = right_total
                            right_len[global_idx] = contour.shape[0]
                            right_points.append(contour)
                            right_total += contour.shape[0]

                if all(point is not None for point in centroids):
                    left_pt = centroids[0]
                    right_pt = centroids[1]
                    separation = math.hypot(left_pt[0] - right_pt[0], left_pt[1] - right_pt[1])
                    eye_separation[global_idx] = float(separation)
                    successful_pairs += 1

            timer.update(task_id, advance=end - start)

    left_concat = (
        np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    )
    right_concat = (
        np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)
    )
    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    chunk_rois = min(512, total_rois) if total_rois > 0 else 1
    run_group.create_array(
        "masks_roi",
        data=masks,
        chunks=(chunk_rois, 2, roi_h, roi_w),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_params",
        data=ellipse_params,
        chunks=(min(1024, total_rois), 2, 5),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_major",
        data=feret_axes_major,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_minor",
        data=feret_axes_minor,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_roundness",
        data=feret_roundness,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    run_group.create_array("contour_left_ptr", data=left_ptr, overwrite=True)
    run_group.create_array("contour_left_len", data=left_len, overwrite=True)
    run_group.create_array("contour_right_ptr", data=right_ptr, overwrite=True)
    run_group.create_array("contour_right_len", data=right_len, overwrite=True)
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

    git_info = get_git_info()
    env_info = get_environment_info()
    total_successful_eyes = int(ellipse_success.sum())
    pair_rate = float(successful_pairs / total_rois) if total_rois > 0 else float("nan")

    run_group.attrs.update(
        {
            "method": "yolo_eye_segmentation",
            "model_path": str(model_path_resolved),
            "model_device": model_device,
            "ultralytics_version": ultralytics_version,
            "config": {
                "batch_size": batch_size,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "mask_threshold": mask_threshold,
            },
            "source_crop_run": crop_run_name,
            "total_rois": total_rois,
            "successful_eyes": total_successful_eyes,
            "successful_roi_pairs": int(successful_pairs),
            "successful_roi_pair_rate": pair_rate,
            "eye_labels": ["eye_left", "eye_right"],
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )
    run_group.attrs["rejected_overlap"] = 0
    run_group.attrs["rejected_too_close"] = 0
    run_group.attrs["rejected_too_far"] = 0

    console.print(
        f"[green]✓[/green] Eye masks saved as [cyan]eye_masks_runs/{resolved_run_name}[/cyan] "
        f"({total_successful_eyes} successful eyes, {successful_pairs}/{total_rois} ROI pairs)"
    )

    return resolved_run_name


__all__ = ["segment_eye_masks_yolo"]
