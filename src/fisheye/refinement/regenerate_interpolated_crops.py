#!/usr/bin/env python3
"""
Regenerate ROI crops for interpolated detections in a refined detection run.

This script reads `refined_detect_runs/<run>/interpolated`, identifies detections
marked as interpolated (`detection_source == 1`), and crops the corresponding
regions from `raw_video/images_full`. The regenerated crops are stored alongside
the refined run so downstream stages (keypoints, ID assignment, etc.) can prefer
them over the original false-positive crops.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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

try:  # pragma: no cover - decord optional at runtime
    import decord
    from decord import VideoReader, cpu, gpu  # type: ignore
    DECORD_AVAILABLE = True
except ImportError:  # pragma: no cover
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = gpu = None  # type: ignore
    DECORD_AVAILABLE = False

try:  # pragma: no cover - torch optional
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ..utils.system import get_environment_info, get_git_info

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
DEFAULT_REFINED_ROI_GROUP = "interpolated_rois"


class CropConfigError(RuntimeError):
    """Raised when required prerequisites for regeneration are missing."""


def _maybe_int(value: Optional[object]) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _maybe_float(value: Optional[object]) -> Optional[float]:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return None


def _collect_expected_video_meta(root: zarr.Group, refined_group: zarr.Group) -> dict:
    expected: dict[str, Optional[object]] = {}

    width = (
        _maybe_int(root.attrs.get("source_video_width"))
        or _maybe_int(root.attrs.get("source_full_width"))
        or _maybe_int(root.attrs.get("video_width"))
        or _maybe_int(root.attrs.get("palette_video_width"))
    )
    height = (
        _maybe_int(root.attrs.get("source_video_height"))
        or _maybe_int(root.attrs.get("source_full_height"))
        or _maybe_int(root.attrs.get("video_height"))
        or _maybe_int(root.attrs.get("palette_video_height"))
    )
    total_frames = (
        _maybe_int(root.attrs.get("total_frames"))
        or _maybe_int(root.attrs.get("n_frames"))
        or _maybe_int(root.attrs.get("palette_total_frames"))
    )
    fps = (
        _maybe_float(root.attrs.get("fps"))
        or _maybe_float(root.attrs.get("video_fps"))
        or _maybe_float(root.attrs.get("palette_fps"))
    )

    detect_run = refined_group.attrs.get("source_detect_run")
    detect_parent = root.get("detect_runs")
    if detect_run and isinstance(detect_parent, zarr.Group) and detect_run in detect_parent:
        detect_attrs = detect_parent[detect_run].attrs
        width = width or _maybe_int(detect_attrs.get("source_video_width")) or _maybe_int(
            detect_attrs.get("inference_width")
        )
        height = height or _maybe_int(detect_attrs.get("source_video_height")) or _maybe_int(
            detect_attrs.get("inference_height")
        )
        total_frames = total_frames or _maybe_int(detect_attrs.get("total_frames")) or _maybe_int(
            detect_attrs.get("n_frames")
        )
        fps = fps or _maybe_float(detect_attrs.get("fps"))
        expected["video_path"] = detect_attrs.get("source_video_path")

    expected["width"] = width
    expected["height"] = height
    expected["total_frames"] = total_frames
    expected["fps"] = fps
    return expected


def _verify_video_metadata(expected: dict, actual: dict) -> None:
    mismatches = []
    width_exp = expected.get("width")
    height_exp = expected.get("height")
    frames_exp = expected.get("total_frames")
    fps_exp = expected.get("fps")

    width_act = actual.get("width")
    height_act = actual.get("height")
    frames_act = actual.get("total_frames")
    fps_act = actual.get("fps")
    path_exp = expected.get("video_path")
    path_act = actual.get("path")

    if width_exp is not None and width_act is not None and width_exp != width_act:
        mismatches.append(f"width expected {width_exp} but video reports {width_act}")
    if height_exp is not None and height_act is not None and height_exp != height_act:
        mismatches.append(f"height expected {height_exp} but video reports {height_act}")
    if frames_exp is not None and frames_act is not None and frames_exp != frames_act:
        mismatches.append(
            f"total_frames expected {frames_exp} but video reports {frames_act}"
        )
    if fps_exp is not None and fps_act is not None and not math.isclose(
        fps_exp, fps_act, rel_tol=1e-3, abs_tol=1e-2
    ):
        mismatches.append(f"fps expected {fps_exp:.3f} but video reports {fps_act:.3f}")
    if path_exp and path_act and Path(str(path_exp)).resolve() != Path(str(path_act)).resolve():
        mismatches.append(f"video path expected '{path_exp}' but opened '{path_act}'")

    if mismatches:
        raise CropConfigError(
            "Source video metadata does not match archive provenance: "
            + "; ".join(mismatches)
        )


def _load_refined_group(root: zarr.Group, run_name: Optional[str], console: Console) -> zarr.Group:
    if REFINED_DETECT_GROUP in root:
        parent = root[REFINED_DETECT_GROUP]
    elif LEGACY_REFINED_DETECT_GROUP in root:
        parent = root[LEGACY_REFINED_DETECT_GROUP]
    else:
        raise CropConfigError("Zarr archive has no refined detection runs (refined_detect_runs).")
    if run_name is None:
        run_name = parent.attrs.get("latest")
        if not run_name:
            raise CropConfigError("No refined detection run specified and parent group has no 'latest'.")
        console.print(f"[cyan]Using refined detection run:[/cyan] {run_name} (latest)")
    elif run_name not in parent:
        raise CropConfigError(f"Refined detection run '{run_name}' not found under refined_detect_runs/.")

    return parent[run_name]


def _resolve_roi_shape(
    root: zarr.Group,
    crop_run: Optional[str],
    explicit_shape: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    if explicit_shape is not None:
        return explicit_shape

    candidates = []
    if crop_run:
        crop_parent = root.get("crop_runs")
        if isinstance(crop_parent, zarr.Group) and crop_run in crop_parent:
            roi_arr = crop_parent[crop_run].get("roi_images")
            if isinstance(roi_arr, zarr.Array) and roi_arr.ndim == 3:
                candidates.append((int(roi_arr.shape[1]), int(roi_arr.shape[2])))

    if not candidates and "crop_runs" in root:
        crop_parent = root["crop_runs"]
        latest = crop_parent.attrs.get("latest")
        if latest and latest in crop_parent:
            roi_arr = crop_parent[latest].get("roi_images")
            if isinstance(roi_arr, zarr.Array) and roi_arr.ndim == 3:
                candidates.append((int(roi_arr.shape[1]), int(roi_arr.shape[2])))

    if candidates:
        return candidates[0]

    raise CropConfigError(
        "Unable to determine ROI size. Re-run with --roi-size H W or specify --source-crop-run."
    )


def _open_decord_reader(video_path: Path, console: Console) -> dict:
    if not DECORD_AVAILABLE:
        raise CropConfigError(
            "Decord is required to read the source video; install decord or import raw_video/images_full."
        )
    if not video_path.exists():
        raise CropConfigError(f"Video file '{video_path}' does not exist.")

    reader = None
    device = "cpu"

    gpu_available = (
        DECORD_AVAILABLE
        and gpu is not None
        and torch is not None
        and torch.cuda.is_available()
    )

    if gpu_available:
        try:  # pragma: no cover - GPU path not covered in tests
            decord.bridge.set_bridge("torch")
            reader = VideoReader(str(video_path), ctx=gpu(0))
            device = "gpu"
        except Exception as exc:
            reader = None
            device = "cpu"
            try:
                decord.bridge.set_bridge("native")
            except Exception:
                pass
            console.print(
                f"[yellow]Decord GPU decode failed ({exc}); falling back to CPU[/yellow]"
            )

    if reader is None:
        try:
            decord.bridge.set_bridge("native")
        except Exception:
            pass
        try:
            ctx = cpu() if cpu is not None else None
            reader = VideoReader(str(video_path), ctx=ctx)
            device = "cpu"
        except Exception as exc:  # pragma: no cover
            raise CropConfigError(f"Failed to open video '{video_path}' with decord: {exc}") from exc

    try:
        first = reader[0]
        if hasattr(first, "asnumpy"):
            frame = first.asnumpy()
        elif hasattr(first, "cpu"):
            frame = first.cpu().numpy()
        else:
            frame = np.asarray(first)
        height, width = int(frame.shape[0]), int(frame.shape[1])
    except Exception as exc:  # pragma: no cover
        raise CropConfigError(f"Unable to read first frame from '{video_path}': {exc}") from exc

    fps = float(reader.get_avg_fps()) if hasattr(reader, "get_avg_fps") else float("nan")
    total_frames = len(reader)

    console.print(
        f"[cyan]Opened source video:[/cyan] {video_path} ({total_frames} frames, device={device})"
    )

    return {
        "reader": reader,
        "height": height,
        "width": width,
        "fps": fps,
        "total_frames": total_frames,
        "path": str(video_path),
        "device": device,
    }


def _ensure_target_group(
    refined_group: zarr.Group,
    group_name: str,
    overwrite: bool,
) -> zarr.Group:
    if group_name in refined_group:
        if not overwrite:
            raise CropConfigError(
                f"Target group '{refined_group.path}/{group_name}' already exists. "
                "Use --overwrite to replace it."
            )
        del refined_group[group_name]
    return refined_group.create_group(group_name)


def _crop_roi(
    frame: np.ndarray,
    cx_norm: float,
    cy_norm: float,
    roi_h: int,
    roi_w: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop a centered window from the frame, padding with zeros if needed."""
    height, width = frame.shape[-2], frame.shape[-1]
    cx = int(round(cx_norm * width))
    cy = int(round(cy_norm * height))

    x1 = cx - roi_w // 2
    y1 = cy - roi_h // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    vy1 = max(0, y1)
    vy2 = min(height, y2)
    vx1 = max(0, x1)
    vx2 = min(width, x2)

    if vy1 >= vy2 or vx1 >= vx2:
        roi = np.zeros((roi_h, roi_w), dtype=frame.dtype)
    elif vy2 - vy1 == roi_h and vx2 - vx1 == roi_w and 0 <= y1 and 0 <= x1 and y2 <= height and x2 <= width:
        roi = frame[vy1:vy2, vx1:vx2]
    else:
        roi = np.zeros((roi_h, roi_w), dtype=frame.dtype)
        py1 = max(0, -y1)
        px1 = max(0, -x1)
        roi[py1 : py1 + vy2 - vy1, px1 : px1 + vx2 - vx1] = frame[vy1:vy2, vx1:vx2]

    return roi, (x1, y1)


def _regenerate_interpolated_rois(
    root: zarr.Group,
    refined_group: zarr.Group,
    roi_shape: Tuple[int, int],
    target_group_name: str,
    batch_size: int,
    overwrite: bool,
    console: Console,
) -> Tuple[int, int]:
    interpolated = refined_group.get("interpolated")
    if interpolated is None:
        raise CropConfigError(f"Refined run '{refined_group.name}' has no 'interpolated' group.")

    detection_source = (
        interpolated["detection_source"][:] if "detection_source" in interpolated else None
    )
    if detection_source is None:
        raise CropConfigError(
            "Interpolated group missing 'detection_source'; cannot distinguish interpolated detections."
        )

    interp_idx = np.where(detection_source == 1)[0]
    if interp_idx.size == 0:
        console.print("[green]No interpolated detections found; nothing to regenerate.[/green]")
        return 0, 0

    bbox_norm = interpolated["bbox_norm_coords"][:]
    frame_indices = interpolated["frame_indices"][:].astype(np.int64, copy=False)

    roi_h, roi_w = roi_shape
    console.print(
        f"[cyan]Regenerating {interp_idx.size} interpolated crops "
        f"({roi_h}×{roi_w})[/cyan]"
    )

    raw_video = root.get("raw_video")
    images_full = None
    scale_factor = None
    if raw_video is not None and "images_full" in raw_video:
        images_full = raw_video["images_full"]
        if images_full.ndim != 3:
            raise CropConfigError("Expected raw_video/images_full to be (frames, H, W).")
        ds_images = raw_video.get("images_ds")
        if ds_images is not None and ds_images.ndim == 3:
            scale_factor = ds_images.shape[1] / images_full.shape[1]

    video_reader = None
    video_height = None
    video_width = None
    video_fps = None
    total_frames: Optional[int] = None
    expected_meta = _collect_expected_video_meta(root, refined_group)
    video_metadata = None
    if images_full is None:
        source_path = refined_group.attrs.get("source_video_path")
        if not source_path:
            detect_run = refined_group.attrs.get("source_detect_run")
            detect_parent = root.get("detect_runs")
            if detect_run and isinstance(detect_parent, zarr.Group) and detect_run in detect_parent:
                source_path = detect_parent[detect_run].attrs.get("source_video_path")
        if not source_path:
            source_path = refined_group.attrs.get("video_path")
        if not source_path:
            raise CropConfigError(
                "raw_video/images_full missing and refined run lacks 'source_video_path'. "
                "Provide --video-path when running the script."
            )

        video_metadata = _open_decord_reader(Path(source_path), console)
        video_reader = video_metadata["reader"]
        video_height = video_metadata["height"]
        video_width = video_metadata["width"]
        video_fps = video_metadata["fps"]
        total_frames = video_metadata["total_frames"]
        video_device = video_metadata.get("device", "cpu")
        _verify_video_metadata(expected_meta, video_metadata)
    else:
        video_height = images_full.shape[1]
        video_width = images_full.shape[2]
        total_frames = images_full.shape[0]
        video_device = "zarr"
        _verify_video_metadata(
            expected_meta,
            {
                "width": video_width,
                "height": video_height,
                "total_frames": total_frames,
                "fps": video_fps,
            },
        )

    target_group = _ensure_target_group(refined_group, target_group_name, overwrite)

    chunk = min(batch_size, interp_idx.size)
    roi_storage = target_group.create_array(
        "roi_images",
        shape=(interp_idx.size, roi_h, roi_w),
        chunks=(chunk, roi_h, roi_w),
        dtype="uint8",
        overwrite=True,
    )
    coords_full = target_group.create_array(
        "roi_coordinates_full",
        shape=(interp_idx.size, 2),
        chunks=(chunk, 2),
        dtype="i4",
        overwrite=True,
    )
    target_group.create_array(
        "frame_indices",
        data=frame_indices[interp_idx].astype("i4", copy=False),
        chunks=(chunk,),
        overwrite=True,
    )
    target_group.create_array(
        "detection_indices",
        data=interp_idx.astype("i8", copy=False),
        chunks=(chunk,),
        overwrite=True,
    )

    if scale_factor is not None:
        coords_ds = target_group.create_array(
            "roi_coordinates_ds",
            shape=(interp_idx.size, 2),
            chunks=(chunk, 2),
            dtype="i4",
            overwrite=True,
        )
    else:
        coords_ds = None
    cropped = 0
    unique_frames = np.unique(frame_indices[interp_idx])
    if total_frames:
        console.print(
            f"[dim]Affected frames: {unique_frames.size} / {total_frames} "
            f"({unique_frames.size / total_frames * 100:.2f}%)[/dim]"
        )
    else:
        console.print(f"[dim]Affected frames: {unique_frames.size}[/dim]")

    start_time = time.perf_counter()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task(
            "Regenerating interpolated crops", total=interp_idx.size
        )

        for batch_start in range(0, interp_idx.size, batch_size):
            batch_end = min(batch_start + batch_size, interp_idx.size)
            batch_ids = interp_idx[batch_start:batch_end]

            rois = np.zeros((batch_ids.size, roi_h, roi_w), dtype=np.uint8)
            coords = np.zeros((batch_ids.size, 2), dtype=np.int32)
            if coords_ds is not None:
                coords_down = np.zeros((batch_ids.size, 2), dtype=np.int32)

            for offset, det_idx in enumerate(batch_ids):
                frame_idx = int(frame_indices[det_idx])
                if frame_idx < 0:
                    raise CropConfigError(f"Negative frame index encountered: {frame_idx}")

                if images_full is not None:
                    if frame_idx >= images_full.shape[0]:
                        raise CropConfigError(
                            f"Frame index {frame_idx} exceeds raw_video/images_full length ({images_full.shape[0]})."
                        )
                    frame = images_full[frame_idx]
                else:
                    if total_frames is not None and frame_idx >= total_frames:
                        raise CropConfigError(
                            f"Frame index {frame_idx} exceeds video length ({total_frames})."
                        )
                    frame_array = video_reader[frame_idx]
                    if hasattr(frame_array, "asnumpy"):
                        frame_array = frame_array.asnumpy()
                    elif hasattr(frame_array, "cpu"):
                        frame_array = frame_array.cpu().numpy()
                    frame = frame_array
                    if frame.ndim == 3:
                        frame = (
                            0.299 * frame[:, :, 0]
                            + 0.587 * frame[:, :, 1]
                            + 0.114 * frame[:, :, 2]
                        )
                    frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

                cx_norm, cy_norm = bbox_norm[det_idx, 0], bbox_norm[det_idx, 1]
                roi, top_left = _crop_roi(frame, cx_norm, cy_norm, roi_h, roi_w)

                rois[offset] = roi
                coords[offset] = top_left
                if coords_ds is not None and scale_factor is not None:
                    dx = int(math.floor(top_left[0] * scale_factor))
                    dy = int(math.floor(top_left[1] * scale_factor))
                    coords_down[offset] = (dx, dy)

            roi_storage[batch_start:batch_end] = rois
            coords_full[batch_start:batch_end] = coords
            if coords_ds is not None and scale_factor is not None:
                coords_ds[batch_start:batch_end] = coords_down

            cropped += batch_ids.size
            progress.advance(task_id, batch_ids.size)

    elapsed = time.perf_counter() - start_time

    target_group.attrs.update(
        {
            "total_crops": int(interp_idx.size),
            "roi_shape": [int(roi_h), int(roi_w)],
            "source": "refine_detect_interpolated",
            "video_device": video_device,
            "duration_seconds": float(elapsed),
        }
    )

    return cropped, unique_frames.size, elapsed, video_device


def _update_metadata(
    refined_group: zarr.Group,
    target_group: str,
    cropped: int,
    unique_frames: int,
    duration_seconds: float,
    video_device: str,
    crop_run: Optional[str],
    env_info: dict,
    console: Console,
) -> None:
    refined_group.attrs["interpolated_roi_path"] = f"{refined_group.path}/{target_group}"
    refined_group.attrs["interpolated_roi_count"] = int(cropped)
    refined_group.attrs["interpolated_roi_frames"] = int(unique_frames)
    refined_group.attrs["interpolated_roi_duration_seconds"] = float(duration_seconds)
    refined_group.attrs["interpolated_roi_decoder"] = video_device

    git_info = get_git_info()
    refined_group.attrs["interpolated_roi_git_commit"] = git_info.get("commit_hash", "unknown")

    platform_info = env_info.get("platform", {})
    refined_group.attrs["interpolated_roi_host"] = platform_info.get("hostname", "unknown")

    if crop_run:
        crop_parent = refined_group._root.get("crop_runs")
        if isinstance(crop_parent, zarr.Group) and crop_run in crop_parent:
            crop_group = crop_parent[crop_run]
            crop_group.attrs["refined_roi_path"] = refined_group.attrs["interpolated_roi_path"]
            crop_group.attrs["refined_roi_count"] = int(cropped)
            crop_group.attrs["refined_roi_frames"] = int(unique_frames)
            crop_group.attrs["refined_roi_generation_duration_seconds"] = float(duration_seconds)
            crop_group.attrs["refined_roi_decoder"] = video_device
            console.print(
                f"[cyan]Linked refined crops to crop_runs/{crop_run}[/cyan]"
            )


def _infer_crop_run(
    root: zarr.Group,
    refined_group: zarr.Group,
    explicit: Optional[str],
    console: Console,
) -> Optional[str]:
    if explicit:
        crop_parent = root.get("crop_runs")
        if isinstance(crop_parent, zarr.Group) and explicit in crop_parent:
            return explicit
        raise CropConfigError(f"Specified crop run '{explicit}' not found in archive.")

    crop_parent = root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group):
        raise CropConfigError(
            "No crop runs found. Provide --source-crop-run to select a crop run."
        )

    source_detect = refined_group.attrs.get("source_detect_run")
    refined_path = f"{refined_group.path}/interpolated"
    expected_paths: List[str] = [refined_path]
    if source_detect:
        expected_paths.append(f"detect_runs/{source_detect}")

    matches = [
        name
        for name in crop_parent.group_keys()
        if crop_parent[name].attrs.get("detection_source_path") in expected_paths
    ]

    if not matches:
        raise CropConfigError(
            "Unable to infer crop run. "
            "Specify --source-crop-run to link regenerated crops to a specific crop run."
        )
    if len(matches) > 1:
        raise CropConfigError(
            "Multiple crop runs reference these detections: "
            + ", ".join(matches)
            + ". Specify --source-crop-run to disambiguate."
        )

    console.print(
        f"[cyan]Inferred crop run for refined detections:[/cyan] {matches[0]}"
    )
    return matches[0]


def regenerate_interpolated_crops(
    zarr_path: Path,
    refined_run: Optional[str],
    target_group_name: str,
    roi_size: Optional[Tuple[int, int]],
    video_path: Optional[Path],
    source_crop_run: Optional[str],
    batch_size: int,
    overwrite: bool,
    console: Console,
) -> None:
    root = zarr.open(str(zarr_path), mode="a")
    refined_group = _load_refined_group(root, refined_run, console)
    if video_path is not None:
        refined_group.attrs["source_video_path"] = str(video_path)
    resolved_crop_run = _infer_crop_run(root, refined_group, source_crop_run, console)
    roi_shape = _resolve_roi_shape(root, resolved_crop_run, roi_size)

    cropped, unique_frames, elapsed, video_device = _regenerate_interpolated_rois(
        root=root,
        refined_group=refined_group,
        roi_shape=roi_shape,
        target_group_name=target_group_name,
        batch_size=batch_size,
        overwrite=overwrite,
        console=console,
    )

    if cropped == 0:
        return

    env_info = get_environment_info()
    _update_metadata(
        refined_group=refined_group,
        target_group=target_group_name,
        cropped=cropped,
        unique_frames=unique_frames,
        duration_seconds=elapsed,
        video_device=video_device,
        crop_run=resolved_crop_run,
        env_info=env_info,
        console=console,
    )

    console.print(
        f"[green]✓[/green] Regenerated {cropped} interpolated crops "
        f"across {unique_frames} frames in {elapsed:.2f}s (decoder={video_device})."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate ROI crops for interpolated detections.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--refined-run",
        help="Specific refined detection run (defaults to latest).",
    )
    parser.add_argument(
        "--target-group",
        default=DEFAULT_REFINED_ROI_GROUP,
        help=f"Name of the group to store regenerated crops (default: {DEFAULT_REFINED_ROI_GROUP}).",
    )
    parser.add_argument(
        "--roi-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Explicit ROI size if not inferrable from an existing crop run.",
    )
    parser.add_argument(
        "--source-crop-run",
        help="Existing crop run to mirror metadata from (optional).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Path to the original video when raw_video/images_full is unavailable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of interpolated detections to crop per batch (default: 512).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the target group if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    console = Console()
    console.rule("[bold]Regenerate Interpolated Crops[/bold]")

    roi_shape = tuple(args.roi_size) if args.roi_size else None

    try:
        regenerate_interpolated_crops(
            zarr_path=args.zarr_path,
            refined_run=args.refined_run,
            target_group_name=args.target_group,
            roi_size=roi_shape,
            video_path=args.video_path,
            source_crop_run=args.source_crop_run,
            batch_size=max(1, int(args.batch_size)),
            overwrite=bool(args.overwrite),
            console=console,
        )
    except CropConfigError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
