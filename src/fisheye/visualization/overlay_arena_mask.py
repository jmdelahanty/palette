#!/usr/bin/env python3
"""
Visualize the experimental arena mask overlaid on raw video frames.

Given a Palette Zarr archive created via ``import_stimulus_to_zarr.py``,
this script:

1. Loads the stimulus run (defaulting to ``analysis/stimulus_runs`` latest).
2. Regenerates the projector-space arena mask from the source Citrus HDF5.
3. Warps the mask into camera coordinates using the stored homography.
4. Overlays the mask on the first raw frame (full-resolution and downsampled).
5. Writes PNG images with the overlays for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import yaml
import zarr

DEFAULT_PROJECTOR_SIZE = (1920, 1080)  # width, height fallback


def _load_arena_config(h5: h5py.File) -> dict:
    """Load arena configuration JSON from Citrus HDF5."""
    arena_path = "/calibration_snapshot/arena_config_json"
    if arena_path not in h5:
        raise KeyError(f"{arena_path} not found in {h5.filename}")
    raw = h5[arena_path][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _load_homography(h5: h5py.File) -> Tuple[np.ndarray, str]:
    """Load homography matrix for the first camera in calibration snapshot."""
    calib_group = h5["/calibration_snapshot"]
    cam_keys = [key for key in calib_group.keys() if key != "arena_config_json"]
    if not cam_keys:
        raise KeyError("No camera calibration groups found in /calibration_snapshot.")
    camera_id = cam_keys[0]
    yml_dataset = f"/calibration_snapshot/{camera_id}/homography_matrix_yml"
    if yml_dataset not in h5:
        raise KeyError(f"{yml_dataset} not found in {h5.filename}")
    yml_text = h5[yml_dataset][()]
    if isinstance(yml_text, bytes):
        yml_text = yml_text.decode("utf-8")
    yml_data = yaml.safe_load(yml_text) or {}
    matrix = np.array(yml_data["homography_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    return matrix, camera_id


def _projector_dimensions(arena_config: dict) -> Tuple[int, int]:
    """Infer projector canvas size in pixels."""
    width = arena_config.get("projector_width_px")
    height = arena_config.get("projector_height_px")
    if width and height:
        return int(width), int(height)
    return DEFAULT_PROJECTOR_SIZE


def create_projector_mask(arena_config: dict, dims: Tuple[int, int]) -> np.ndarray:
    """Create a binary mask in projector space based on arena configuration."""
    width, height = dims
    mask = np.zeros((height, width), dtype=np.uint8)

    center_x = arena_config.get("experimental_area_center_x_px")
    center_y = arena_config.get("experimental_area_center_y_px")
    if center_x is None or center_y is None:
        raise ValueError("Arena config missing center coordinates.")

    if arena_config.get("experimental_area_shape") == "CIRCLE":
        radius = arena_config.get("experimental_area_radius_px")
        if radius is None:
            raise ValueError("Arena config missing radius for circle shape.")
        cv2.circle(mask, (int(round(center_x)), int(round(center_y))), int(round(radius)), 255, -1)
        return mask

    if arena_config.get("experimental_area_shape") == "RECTANGLE":
        width_px = arena_config.get("experimental_area_width_px")
        height_px = arena_config.get("experimental_area_height_px")
        if width_px is None or height_px is None:
            raise ValueError("Arena config missing rectangle dimensions.")
        corner_radius = int(round(arena_config.get("experimental_area_corner_radius_px", 0) or 0))
        x1 = int(round(center_x - width_px / 2.0))
        y1 = int(round(center_y - height_px / 2.0))
        x2 = int(round(center_x + width_px / 2.0))
        y2 = int(round(center_y + height_px / 2.0))
        if corner_radius <= 0:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask
        # Rounded rectangle via rectangles + quarter circles.
        cv2.rectangle(mask, (x1 + corner_radius, y1), (x2 - corner_radius, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1 + corner_radius), (x2, y2 - corner_radius), 255, -1)
        cv2.circle(mask, (x1 + corner_radius, y1 + corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (x2 - corner_radius, y1 + corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (x1 + corner_radius, y2 - corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (x2 - corner_radius, y2 - corner_radius), corner_radius, 255, -1)
        return mask

    raise ValueError(f"Unsupported experimental_area_shape: {arena_config.get('experimental_area_shape')}")


def warp_mask_to_camera(projector_mask: np.ndarray, homography: np.ndarray, camera_size: Tuple[int, int]) -> np.ndarray:
    """Warp projector mask into camera pixel coordinates using inverse homography."""
    camera_width, camera_height = camera_size
    h_inv = np.linalg.inv(homography)
    warped = cv2.warpPerspective(projector_mask, h_inv, (camera_width, camera_height))
    return warped


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    """Overlay binary mask onto grayscale image with outline."""
    base = image
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    mask_bool = mask > 127
    if mask_bool.any():
        color_layer = np.zeros_like(base, dtype=np.uint8)
        color_layer[mask_bool] = color
        cv2.addWeighted(color_layer, alpha, overlay, 1.0, 0.0, overlay)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), thickness=2)

    return overlay


def _resolve_stimulus_run(root: zarr.Group, run_name: Optional[str]) -> Tuple[zarr.Group, str]:
    """Pick a stimulus run group and return it along with its name."""
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise KeyError("analysis/stimulus_runs not found in Zarr archive.")
    stim_root = analysis["stimulus_runs"]
    chosen = run_name or stim_root.attrs.get("latest")
    if not chosen:
        keys = sorted(k for k in stim_root.group_keys())
        if not keys:
            raise KeyError("No runs in analysis/stimulus_runs.")
        chosen = keys[0]
    if chosen not in stim_root:
        raise KeyError(f"Stimulus run '{chosen}' not present in analysis/stimulus_runs.")
    return stim_root[chosen], chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay experimental arena mask on raw frames.")
    parser.add_argument("zarr", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to visualize.")
    parser.add_argument("--output-dir", type=Path, help="Directory to store rendered PNGs.")
    parser.add_argument(
        "--calibration-h5",
        type=Path,
        help="Optional path to Citrus stimulus H5. Overrides run attrs['source_h5'] if provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zarr_path = args.zarr.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    root = zarr.open(zarr_path, mode="r")
    run_group, run_name = _resolve_stimulus_run(root, args.stimulus_run)

    source_h5_path = args.calibration_h5 or run_group.attrs.get("source_h5")
    if not source_h5_path:
        raise ValueError("Stimulus run missing source_h5 attribute. Provide --calibration-h5.")
    source_h5_path = Path(source_h5_path).expanduser().resolve()
    if not source_h5_path.exists():
        raise FileNotFoundError(f"Source H5 not found: {source_h5_path}")

    with h5py.File(source_h5_path, "r") as h5:
        arena_config = _load_arena_config(h5)
        projector_dims = _projector_dimensions(arena_config)
        projector_mask = create_projector_mask(arena_config, projector_dims)
        homography, camera_id = _load_homography(h5)

    raw_group = root.get("raw_video")
    if raw_group is None:
        raise KeyError("raw_video group missing from Zarr archive.")

    overlays = []

    if "images_full" in raw_group:
        full_frames = raw_group["images_full"]
        if args.frame_index >= len(full_frames):
            raise IndexError(f"Frame index {args.frame_index} out of bounds for images_full.")
        full_frame = np.asarray(full_frames[args.frame_index])
        camera_shape = (full_frame.shape[1], full_frame.shape[0])
        camera_mask = warp_mask_to_camera(projector_mask, homography, camera_shape)
        overlay_full = _overlay_mask(full_frame, camera_mask)
        overlays.append(("full", overlay_full))

    if "images_ds" in raw_group:
        ds_frames = raw_group["images_ds"]
        if args.frame_index >= len(ds_frames):
            raise IndexError(f"Frame index {args.frame_index} out of bounds for images_ds.")
        ds_frame = np.asarray(ds_frames[args.frame_index])
        if "images_full" in raw_group:
            ds_mask = cv2.resize(camera_mask, (ds_frame.shape[1], ds_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            camera_shape = (ds_frame.shape[1], ds_frame.shape[0])
            ds_mask = warp_mask_to_camera(projector_mask, homography, camera_shape)
        overlay_ds = _overlay_mask(ds_frame, ds_mask)
        overlays.append(("downsampled", overlay_ds))

    if not overlays:
        raise KeyError("Neither images_full nor images_ds datasets are present in raw_video group.")

    output_dir = args.output_dir or (Path.cwd() / "arena_mask_overlays")
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag, image in overlays:
        filename = output_dir / f"{run_name}_{tag}_frame{args.frame_index:04d}.png"
        cv2.imwrite(str(filename), image)
        print(f"Wrote overlay: {filename} (camera_id={camera_id})")


if __name__ == "__main__":
    main()
