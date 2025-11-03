#!/usr/bin/env python3
"""
Compute chaser ↔ fish geometric metrics using offline keypoints and store them in analysis.

Metrics:
    - distance between fish centroid and chaser (pixels, millimetres if calibration available)
    - signed / unsigned angle between fish heading vector and chaser displacement vector

Inputs:
    • Zarr archive containing:
        - analysis/stimulus_runs/<run>/tracking_data/chaser_states
        - keypoints_runs/<keypoint_run>

Outputs:
    • analysis/chaser_fish_metrics/<output_run>/ with 1-D arrays aligned to ROI indices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import zarr
from rich.console import Console

from .chaser_state_interpolator import load_structured_dataset
from ..utils.calibration import load_run_calibration


DEFAULT_FISH_LABEL_PRIORITY = ("bladder", "body", "center", "midpoint", "spine", "spine_base")


def _resolve_zarr(path: Path) -> zarr.Group:
    try:
        return zarr.open(str(path), mode="a")
    except Exception as exc:  # pragma: no cover - handled via CLI message
        raise RuntimeError(f"Unable to open Zarr store at {path}: {exc}") from exc


def _list_latest(group: zarr.Group) -> Optional[str]:
    latest = group.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    return latest if isinstance(latest, str) and latest in group else None


def _require_group(parent: zarr.Group, path: str) -> zarr.Group:
    try:
        return parent[path]
    except Exception as exc:
        raise ValueError(f"Required group '{path}' not found in Zarr store.") from exc


def _structured_to_dict(array: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: array[name] for name in array.dtype.names} if array.dtype.names else {}


def _get_field(data: Dict[str, np.ndarray], *names: str, required: bool = True) -> Optional[np.ndarray]:
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    if required:
        raise ValueError(f"chaser_states missing required field (tried {', '.join(names)})")
    return None


def _resolve_texture_scale(
    zarr_root: zarr.Group,
    stim_group: zarr.Group,
    stimulus_run: str,
) -> Tuple[float, Optional[float]]:
    calibration = load_run_calibration(zarr_root, stimulus_run)
    scale = calibration.texture_to_camera_scale or 1.0
    texture_width = calibration.texture_width_px

    if (scale == 1.0 or not np.isfinite(scale)) and calibration.camera_width_px and texture_width:
        try:
            scale = float(calibration.camera_width_px) / float(texture_width)
        except (TypeError, ValueError, ZeroDivisionError):
            scale = 1.0

    if texture_width is None:
        texture_width = _load_texture_width_from_h5(stim_group)

    if texture_width is None:
        width = zarr_root.attrs.get("width")
        if width is None:
            meta = zarr_root.attrs.get("source_video_metadata", {})
            width = meta.get("width") or meta.get("full_width")
        try:
            texture_width = float(width)
        except (TypeError, ValueError):
            texture_width = None

    return scale, texture_width


def _load_texture_width_from_h5(stim_group: zarr.Group) -> Optional[float]:
    source_h5 = stim_group.attrs.get("source_h5")
    if not source_h5:
        return None
    try:
        path = Path(source_h5)
    except TypeError:
        return None
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as handle:
            if "stimulus_coordinates" not in handle:
                return None
            coord_root = handle["stimulus_coordinates"]
            for name in coord_root:
                node = coord_root[name]
                if isinstance(node, h5py.Group):
                    width = node.attrs.get("texture_width_px")
                    if width is not None:
                        try:
                            return float(width)
                        except (TypeError, ValueError):
                            continue
    except Exception:
        return None
    return None


def _resolve_struct_field(array: np.ndarray, *candidates: str) -> str:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(
        f"Structured array missing expected field (tried {', '.join(candidates)})"
    )


def _build_stimulus_to_camera_map(stim_group: zarr.Group) -> Dict[int, int]:
    meta_group = _require_group(stim_group, "video_metadata")
    frame_metadata, _ = load_structured_dataset(meta_group, "frame_metadata")
    stim_field = _resolve_struct_field(frame_metadata, "stimulus_frame_num", "frame_number", "stim_frame_num")
    camera_field = _resolve_struct_field(
        frame_metadata,
        "triggering_camera_frame_id",
        "camera_frame_id",
    )

    mapping: Dict[int, int] = {}
    for record in frame_metadata:
        stim = int(record[stim_field])
        cam = int(record[camera_field])
        mapping.setdefault(stim, cam)
    return mapping


def _safe_mean(points: np.ndarray) -> Optional[np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    mask = np.isfinite(points).all(axis=1)
    if not np.any(mask):
        return None
    return points[mask].mean(axis=0)


def _heading_vector(deg: float) -> Optional[np.ndarray]:
    if not np.isfinite(deg):
        return None
    rad = np.deg2rad(float(deg))
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


def _signed_angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[float, float]:
    """Return (unsigned_deg, signed_deg) between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return (np.nan, np.nan)
    a_unit = vec_a / norm_a
    b_unit = vec_b / norm_b
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    unsigned = float(np.degrees(np.arccos(dot)))
    cross_z = float(a_unit[0] * b_unit[1] - a_unit[1] * b_unit[0])
    signed = float(np.degrees(np.arctan2(cross_z, dot)))
    return unsigned, signed


def _resolve_fish_position(
    keypoints: np.ndarray,
    labels: Sequence[str],
) -> Optional[np.ndarray]:
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        return None

    # Prefer specific labels if available
    label_map = {label: idx for idx, label in enumerate(labels)}
    for label in DEFAULT_FISH_LABEL_PRIORITY:
        idx = label_map.get(label)
        if idx is not None and idx < keypoints.shape[0]:
            pt = keypoints[idx]
            if np.isfinite(pt).all():
                return pt.astype(np.float64)

    # Fallback to average of all finite keypoints
    return _safe_mean(keypoints.astype(np.float64, copy=False))


def _prepare_output_group(
    analysis_root: zarr.Group,
    run_name: str,
    overwrite: bool,
) -> zarr.Group:
    parent = analysis_root.require_group("chaser_fish_metrics")
    if run_name in parent:
        if not overwrite:
            raise ValueError(
                f"analysis/chaser_fish_metrics/{run_name} already exists. "
                "Use --overwrite to replace the run."
            )
        del parent[run_name]
    return parent.create_group(run_name)


def compute_metrics(
    zarr_root: zarr.Group,
    stimulus_run: str,
    keypoints_path: str,
    output_run: str,
    *,
    chaser_index: int,
    overwrite: bool,
    console: Console,
) -> None:
    analysis_root = zarr_root.require_group("analysis")
    stim_group = _require_group(analysis_root, f"stimulus_runs/{stimulus_run}")
    tracking_group = _require_group(stim_group, "tracking_data")
    if "chaser_states" not in tracking_group:
        raise ValueError(
            f"Run analysis/stimulus_runs/{stimulus_run} does not contain tracking_data/chaser_states."
        )
    chaser_states, _ = load_structured_dataset(tracking_group, "chaser_states")
    texture_scale, texture_width = _resolve_texture_scale(zarr_root, stim_group, stimulus_run)
    chaser_dict = _structured_to_dict(chaser_states)
    frame_numbers_arr = _get_field(chaser_dict, "frame_number", "stimulus_frame_num")
    try:
        camera_frame_arr = _get_field(chaser_dict, "camera_frame_id", "triggering_camera_frame_id")
    except ValueError:
        stim_to_camera = _build_stimulus_to_camera_map(stim_group)
        camera_frame_arr = np.array(
            [stim_to_camera.get(int(stim), -1) for stim in frame_numbers_arr],
            dtype=np.int64,
        )
    pos_x_arr = _get_field(chaser_dict, "pos_x_px", "chaser_pos_x")
    pos_y_arr = _get_field(chaser_dict, "pos_y_px", "chaser_pos_y")

    mask_index = chaser_dict.get("chaser_index")
    if mask_index is not None:
        chaser_rows = mask_index == chaser_index
    else:
        chaser_rows = np.ones_like(frame_numbers_arr, dtype=bool)
    if not np.any(chaser_rows):
        raise ValueError(f"No chaser_states rows found for chaser_index={chaser_index}.")

    camera_frames = np.asarray(camera_frame_arr)[chaser_rows].astype(np.int64)
    chaser_pos = np.stack(
        [pos_x_arr[chaser_rows], pos_y_arr[chaser_rows]],
        axis=1,
    ).astype(np.float64)
    if texture_scale != 1.0:
        chaser_pos *= texture_scale
    pixels_per_mm = chaser_dict.get("pixels_per_mm")
    ppm_values = pixels_per_mm[chaser_rows].astype(np.float64) if pixels_per_mm is not None else None

    chaser_by_frame: Dict[int, Tuple[np.ndarray, Optional[float]]] = {}
    for idx, cam_frame in enumerate(camera_frames):
        if cam_frame < 0:
            continue
        chaser_by_frame[int(cam_frame)] = (
            chaser_pos[idx],
            float(ppm_values[idx]) if ppm_values is not None else None,
        )

    kp_group = _require_group(zarr_root, keypoints_path)
    kp_frame_indices = np.asarray(kp_group["frame_indices"], dtype=np.int64)
    kp_heading = np.asarray(kp_group.get("heading", np.full_like(kp_frame_indices, np.nan, dtype=np.float64)))

    crop_run = kp_group.attrs.get("source_crop_run")
    if not crop_run:
        raise ValueError(
            "keypoints run is missing source_crop_run attribute; unable to derive detection centroids"
        )

    crop_group = _require_group(zarr_root, f"crop_runs/{crop_run}")
    bbox_norm = np.asarray(crop_group["bbox_norm_coords"], dtype=np.float64)
    if bbox_norm.shape[0] != kp_frame_indices.shape[0]:
        raise ValueError(
            "bbox_norm_coords length does not match keypoint ROI count; expected "
            f"{kp_frame_indices.shape[0]}, found {bbox_norm.shape[0]}"
        )

    full_images = zarr_root["raw_video/images_full"]
    full_height, full_width = full_images.shape[1], full_images.shape[2]
    bbox_centers = np.empty((bbox_norm.shape[0], 2), dtype=np.float64)
    bbox_centers[:, 0] = bbox_norm[:, 0] * full_width
    bbox_centers[:, 1] = bbox_norm[:, 1] * full_height

    n_rois = kp_frame_indices.shape[0]
    metrics_mask = np.zeros(n_rois, dtype=bool)
    fish_centroids = np.full((n_rois, 2), np.nan, dtype=np.float32)
    chaser_positions = np.full((n_rois, 2), np.nan, dtype=np.float32)
    distance_px = np.full(n_rois, np.nan, dtype=np.float32)
    distance_mm = np.full(n_rois, np.nan, dtype=np.float32)
    angle_unsigned = np.full(n_rois, np.nan, dtype=np.float32)
    angle_signed = np.full(n_rois, np.nan, dtype=np.float32)

    skipped_no_chaser = skipped_no_detection = 0
    missing_heading = 0
    for idx in range(n_rois):
        if not np.all(np.isfinite(bbox_centers[idx])):
            skipped_no_detection += 1
            continue

        # Always store fish centroid and heading when detection is valid
        fish_point = bbox_centers[idx]
        frame = int(kp_frame_indices[idx])
        metrics_mask[idx] = True
        fish_centroids[idx] = fish_point.astype(np.float32)

        # Check for chaser data
        chaser_entry = chaser_by_frame.get(frame)
        if chaser_entry is None:
            skipped_no_chaser += 1
            # Leave chaser-specific metrics (distance, angle, chaser_position) as NaN
            continue

        # Compute chaser-specific metrics when chaser is present
        chaser_point, ppm = chaser_entry
        displacement = chaser_point - fish_point
        dist_px = float(np.linalg.norm(displacement))

        chaser_positions[idx] = chaser_point.astype(np.float32)
        distance_px[idx] = dist_px
        if ppm is not None and np.isfinite(ppm) and ppm > 0:
            distance_mm[idx] = dist_px / ppm

        heading_vec = _heading_vector(float(kp_heading[idx])) if np.isfinite(kp_heading[idx]) else None
        if heading_vec is None:
            missing_heading += 1
            # Leave angles as NaN when heading is missing
            continue

        unsigned_deg, signed_deg = _signed_angle_deg(heading_vec, displacement)
        angle_unsigned[idx] = unsigned_deg
        angle_signed[idx] = signed_deg

    valid_count = int(metrics_mask.sum())
    total_frames = len(chaser_by_frame)

    console.print(
        f"[bold green]✓ Processed {valid_count} / {n_rois} ROIs with valid detections[/bold green]"
    )
    console.print(
        f"  {skipped_no_detection} frames skipped (invalid detection centroid)"
    )
    console.print(
        f"  {skipped_no_chaser} frames missing chaser data (chaser metrics = NaN)"
    )
    console.print(
        f"  {missing_heading} frames missing heading data (angle metrics = NaN)"
    )

    out_group = _prepare_output_group(analysis_root, output_run, overwrite=overwrite)

    def _write_array(name: str, data: np.ndarray, *, dtype: Optional[str] = None) -> None:
        arr = data.astype(dtype) if dtype else data
        if arr.ndim == 0:
            chunks = None
        else:
            first_dim = arr.shape[0] if arr.shape[0] > 0 else 1
            chunk_first = max(1, min(4096, first_dim))
            chunks = (chunk_first,) + arr.shape[1:]
        out_group.create_array(
            name,
            data=arr,
            chunks=chunks,
            overwrite=True,
        )

    _write_array("frame_indices", kp_frame_indices.astype(np.int64), dtype="int64")
    _write_array("valid_mask", metrics_mask.astype(np.bool_), dtype="bool")
    _write_array("fish_centroid_px", fish_centroids.astype(np.float32), dtype="float32")
    _write_array("chaser_position_px", chaser_positions.astype(np.float32), dtype="float32")
    _write_array("distance_px", distance_px.astype(np.float32), dtype="float32")
    _write_array("distance_mm", distance_mm.astype(np.float32), dtype="float32")
    _write_array("heading_deg", kp_heading.astype(np.float32), dtype="float32")
    _write_array("angle_unsigned_deg", angle_unsigned.astype(np.float32), dtype="float32")
    _write_array("angle_signed_deg", angle_signed.astype(np.float32), dtype="float32")

    out_group.attrs.update(
        {
            "source_stimulus_run": stimulus_run,
            "source_keypoints_run": keypoints_path,
            "chaser_index": int(chaser_index),
            "total_rois": int(n_rois),
            "valid_rois": int(valid_count),
            "total_chaser_frames": int(total_frames),
            "metrics_version": "1.0.0",
            "texture_to_camera_scale": float(texture_scale),
        }
    )
    if texture_width:
        out_group.attrs["texture_width_px"] = float(texture_width)

    console.print(
        f"[bold cyan]Stored metrics in analysis/chaser_fish_metrics/{output_run}[/bold cyan]"
    )

    analysis_root.require_group("chaser_fish_metrics").attrs["latest"] = output_run


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute chaser ↔ fish geometric metrics using offline keypoints."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--stimulus-run",
        dest="stimulus_run",
        help="analysis/stimulus_runs/<name> to use (default: latest).",
    )
    parser.add_argument(
        "--keypoints-run",
        dest="keypoints_run",
        help="keypoints_runs/<name> providing YOLO detections (default: latest).",
    )
    parser.add_argument(
        "--output-run",
        dest="output_run",
        help="Name for analysis/chaser_fish_metrics/<name> (default: derived from keypoints run).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to use from chaser_states (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metrics run if present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    console = Console()

    root = _resolve_zarr(Path(args.zarr_path))
    analysis_root = root.require_group("analysis")

    stim_parent = analysis_root.require_group("stimulus_runs")
    stimulus_run = args.stimulus_run or _list_latest(stim_parent)
    if stimulus_run is None:
        raise ValueError("Unable to resolve stimulus run (no runs present and --stimulus-run not provided).")
    if stimulus_run not in stim_parent:
        raise ValueError(f"Stimulus run '{stimulus_run}' not found in analysis/stimulus_runs.")

    kp_arg = args.keypoints_run.strip("/") if args.keypoints_run else None
    if kp_arg:
        if kp_arg.startswith("keypoints_runs/") or kp_arg.startswith("refined_keypoints_runs/"):
            keypoints_path = kp_arg
        else:
            keypoints_path = f"keypoints_runs/{kp_arg}"
    else:
        refined_parent = root.get("refined_keypoints_runs")
        refined_latest = _list_latest(refined_parent) if refined_parent is not None else None
        if refined_latest:
            keypoints_path = f"refined_keypoints_runs/{refined_latest}"
        else:
            kp_parent = root.require_group("keypoints_runs")
            latest_raw = _list_latest(kp_parent)
            if latest_raw is None:
                raise ValueError(
                    "Unable to resolve keypoints run (no runs present and --keypoints-run not provided)."
                )
            keypoints_path = f"keypoints_runs/{latest_raw}"

    try:
        _ = _require_group(root, keypoints_path)
    except ValueError as exc:
        raise ValueError(f"Keypoints run '{keypoints_path}' not found in zarr store") from exc

    keypoints_name = keypoints_path.split("/")[-1]
    output_run = args.output_run or f"{keypoints_name}_chaser_metrics"

    console.print(
        f"[bold]Computing metrics[/bold]\n"
        f"  Stimulus run : {stimulus_run}\n"
        f"  Keypoints run: {keypoints_path}\n"
        f"  Output run   : {output_run}\n"
        f"  Chaser index : {args.chaser_index}\n"
    )

    compute_metrics(
        zarr_root=root,
        stimulus_run=stimulus_run,
        keypoints_path=keypoints_path,
        output_run=output_run,
        chaser_index=args.chaser_index,
        overwrite=args.overwrite,
        console=console,
    )


if __name__ == "__main__":
    main()
