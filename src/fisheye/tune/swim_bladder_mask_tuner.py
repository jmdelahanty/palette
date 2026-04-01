#!/usr/bin/env python3
"""Interactive tuner for keypoint-centered traditional swim-bladder segmentation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import zarr
from skimage import measure

from ..diagnostics.preview_eye_mask_background_subtraction import _prepare_panel, _resolve_run_name
from ..utils.zarr_io import open_zarr_root
from ..visualization.visualize_swim_bladder_mask_patches import (
    _extract_patch_bounds,
    _resolve_swim_bladder_center_with_source,
)
from . import eye_mask_tuner as eye_mask_ops
from . import subject_mask_tuner as subject_tuner

try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)


WINDOW_NAME = "Swim Bladder Mask Tuner"
THRESHOLD_METHOD_FAMILY = "swim_bladder_patch_threshold_v1"
POLAR_BOUNDARY_METHOD_FAMILY = "swim_bladder_polar_boundary_v1"
DEFAULT_METHOD_FAMILY = THRESHOLD_METHOD_FAMILY

DEFAULT_ROI_PADDING = 18
DEFAULT_PRE_THRESHOLD: Optional[int] = None
DEFAULT_SOBEL_STRENGTH = 0.0
DEFAULT_MIN_AREA = 8
DEFAULT_MAX_AREA: Optional[int] = None
DEFAULT_MIN_CIRCULARITY: Optional[float] = None
DEFAULT_CLOSING_RADIUS = 1
DEFAULT_OPENING_RADIUS = 0

SLIDER_MAX_PADDING = 64
SLIDER_MAX_PRETHRESH = 255
SLIDER_MAX_SOBEL = 100
SLIDER_MAX_AREA = 2000
SLIDER_MAX_RADIUS = 12
SLIDER_MAX_CIRCULARITY = 100
SLIDER_MAX_ANGLE_STEP = 60
SLIDER_MAX_RESPONSE = 100
SLIDER_MAX_GAP_DEGREES = 180
SLIDER_MAX_VALID_FRAC = 100
SLIDER_MAX_SIGMA = 100
DEFAULT_PANEL_SIZE = 280


def _normalize_method_family(value: Optional[object]) -> str:
    if value is None:
        return DEFAULT_METHOD_FAMILY
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "threshold": THRESHOLD_METHOD_FAMILY,
        "blob": THRESHOLD_METHOD_FAMILY,
        "patch_threshold": THRESHOLD_METHOD_FAMILY,
        "threshold_blob": THRESHOLD_METHOD_FAMILY,
        THRESHOLD_METHOD_FAMILY: THRESHOLD_METHOD_FAMILY,
        "polar": POLAR_BOUNDARY_METHOD_FAMILY,
        "polar_boundary": POLAR_BOUNDARY_METHOD_FAMILY,
        POLAR_BOUNDARY_METHOD_FAMILY: POLAR_BOUNDARY_METHOD_FAMILY,
    }
    return aliases.get(text, DEFAULT_METHOD_FAMILY)


def _normalize_gradient_mode(value: Optional[object]) -> str:
    if value is None:
        return "sobel_magnitude"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sobel": "sobel_magnitude",
        "sobel_magnitude": "sobel_magnitude",
        "scharr": "scharr_magnitude",
        "scharr_magnitude": "scharr_magnitude",
        "laplacian": "laplacian_abs",
        "laplacian_abs": "laplacian_abs",
    }
    return aliases.get(text, "sobel_magnitude")


def _default_swim_bladder_params(*, method_family: str = DEFAULT_METHOD_FAMILY) -> Dict[str, Any]:
    normalized_family = _normalize_method_family(method_family)
    if normalized_family == POLAR_BOUNDARY_METHOD_FAMILY:
        return {
            "roi_padding": DEFAULT_ROI_PADDING,
            "angle_step_degrees": 8,
            "min_radius_px": 3,
            "max_radius_px": DEFAULT_ROI_PADDING,
            "smoothing_sigma": 1.5,
            "response_threshold": 0.12,
            "max_missing_gap_degrees": 40,
            "min_valid_ray_fraction": 0.55,
            "gradient_mode": "sobel_magnitude",
            "prefilter_sigma": 1.0,
        }
    return {
        "roi_padding": DEFAULT_ROI_PADDING,
        "pre_threshold": DEFAULT_PRE_THRESHOLD,
        "sobel_strength": DEFAULT_SOBEL_STRENGTH,
        "min_area": DEFAULT_MIN_AREA,
        "max_area": DEFAULT_MAX_AREA,
        "min_circularity": DEFAULT_MIN_CIRCULARITY,
        "closing_radius": DEFAULT_CLOSING_RADIUS,
        "opening_radius": DEFAULT_OPENING_RADIUS,
    }


def _normalize_swim_bladder_params(
    params: Mapping[str, Any],
    *,
    method_family: str = DEFAULT_METHOD_FAMILY,
) -> Dict[str, Any]:
    normalized_family = _normalize_method_family(method_family)

    def _optional_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 else None

    def _optional_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0.0 else None

    if normalized_family == POLAR_BOUNDARY_METHOD_FAMILY:
        angle_step = max(1, int(params.get("angle_step_degrees", 8)))
        min_radius = max(1, int(params.get("min_radius_px", 3)))
        max_radius = max(min_radius + 1, int(params.get("max_radius_px", DEFAULT_ROI_PADDING)))
        return {
            "roi_padding": max(max_radius, int(params.get("roi_padding", DEFAULT_ROI_PADDING)), 1),
            "angle_step_degrees": angle_step,
            "min_radius_px": min_radius,
            "max_radius_px": max_radius,
            "smoothing_sigma": max(0.0, float(params.get("smoothing_sigma", 1.5))),
            "response_threshold": float(np.clip(params.get("response_threshold", 0.12), 0.0, 1.0)),
            "max_missing_gap_degrees": max(0, int(params.get("max_missing_gap_degrees", 40))),
            "min_valid_ray_fraction": float(np.clip(params.get("min_valid_ray_fraction", 0.55), 0.0, 1.0)),
            "gradient_mode": _normalize_gradient_mode(params.get("gradient_mode")),
            "prefilter_sigma": max(0.0, float(params.get("prefilter_sigma", 1.0))),
        }

    return {
        "roi_padding": max(1, int(params.get("roi_padding", DEFAULT_ROI_PADDING))),
        "pre_threshold": _optional_int(params.get("pre_threshold", DEFAULT_PRE_THRESHOLD)),
        "sobel_strength": max(0.0, float(params.get("sobel_strength", DEFAULT_SOBEL_STRENGTH))),
        "min_area": max(1, int(params.get("min_area", DEFAULT_MIN_AREA))),
        "max_area": _optional_int(params.get("max_area", DEFAULT_MAX_AREA)),
        "min_circularity": _optional_float(params.get("min_circularity", DEFAULT_MIN_CIRCULARITY)),
        "closing_radius": max(0, int(params.get("closing_radius", DEFAULT_CLOSING_RADIUS))),
        "opening_radius": max(0, int(params.get("opening_radius", DEFAULT_OPENING_RADIUS))),
    }


def _build_swim_bladder_params(
    tuned_params: Optional[Mapping[str, Any]],
    *,
    method_family: str = DEFAULT_METHOD_FAMILY,
) -> Dict[str, Any]:
    params = _default_swim_bladder_params(method_family=method_family)
    if isinstance(tuned_params, Mapping):
        params.update(_normalize_swim_bladder_params(tuned_params, method_family=method_family))
    return params


def _resolve_subject_source(root: zarr.Group, subject_run: Optional[str]) -> Optional[subject_tuner.SubjectMaskSource]:
    source = subject_tuner._load_subject_mask_source(root, subject_run)
    if source is None:
        return None
    component_index = subject_tuner._find_subject_label_index(source, "swim_bladder")
    if component_index is None:
        return None
    if not bool(source.available_channels[component_index]):
        return None
    return source


def _circular_gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0 or values.size == 0:
        return np.asarray(values, dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    weights = np.exp(-(offsets.astype(np.float32) ** 2) / (2.0 * float(sigma) * float(sigma)))
    weights /= float(weights.sum())
    smoothed = np.zeros_like(values, dtype=np.float32)
    for offset, weight in zip(offsets.tolist(), weights.tolist()):
        smoothed += np.roll(values, shift=int(offset)) * float(weight)
    return smoothed


def _longest_nan_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    mask = np.isnan(values)
    if not np.any(mask):
        return 0
    doubled = np.concatenate([mask, mask], axis=0)
    longest = current = 0
    for flag in doubled.tolist():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        if longest >= int(mask.size):
            return int(mask.size)
    return min(int(longest), int(mask.size))


def _interpolate_circular_nans(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    if np.all(valid):
        return arr
    x = np.arange(arr.size, dtype=np.float32)
    x_valid = x[valid]
    y_valid = arr[valid]
    x_ext = np.concatenate([x_valid - float(arr.size), x_valid, x_valid + float(arr.size)], axis=0)
    y_ext = np.concatenate([y_valid, y_valid, y_valid], axis=0)
    interpolated = np.interp(x, x_ext, y_ext).astype(np.float32)
    arr[~valid] = interpolated[~valid]
    return arr


def _sync_polar_radius_constraints(
    patch_pad: int,
    min_radius: int,
    max_radius: int,
) -> tuple[int, int, int]:
    synced_pad = max(2, int(patch_pad))
    synced_max = max(2, min(int(max_radius), synced_pad))
    synced_min = max(1, min(int(min_radius), synced_max - 1))
    return synced_pad, synced_min, synced_max


def _compute_polar_boundary_preview(
    patch_uint8: np.ndarray,
    *,
    center_xy: tuple[float, float],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    sigma = float(params.get("prefilter_sigma", 1.0))
    gradient_mode = _normalize_gradient_mode(params.get("gradient_mode"))
    patch_float = patch_uint8.astype(np.float32) / 255.0
    if sigma > 0.0:
        prefiltered = cv2.GaussianBlur(patch_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        prefiltered = patch_float

    if gradient_mode == "laplacian_abs":
        response = np.abs(cv2.Laplacian(prefiltered, cv2.CV_32F, ksize=3)).astype(np.float32, copy=False)
    elif gradient_mode == "scharr_magnitude":
        grad_x = cv2.Scharr(prefiltered, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(prefiltered, cv2.CV_32F, 0, 1)
        response = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32, copy=False)
    else:
        grad_x = cv2.Sobel(prefiltered, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(prefiltered, cv2.CV_32F, 0, 1, ksize=3)
        response = np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32, copy=False)
    max_response = float(response.max()) if response.size else 0.0
    if max_response > 0.0:
        response_norm = response / max_response
    else:
        response_norm = np.zeros_like(response, dtype=np.float32)

    angle_step = max(1, int(params.get("angle_step_degrees", 8)))
    angles_deg = np.arange(0, 360, angle_step, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)
    min_radius = max(1, int(params.get("min_radius_px", 3)))
    max_radius = max(min_radius + 1, int(params.get("max_radius_px", DEFAULT_ROI_PADDING)))
    radii = np.arange(min_radius, max_radius + 1, dtype=np.float32)
    response_threshold = float(np.clip(params.get("response_threshold", 0.12), 0.0, 1.0))

    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    chosen_radii = np.full((angles_rad.shape[0],), np.nan, dtype=np.float32)
    chosen_scores = np.full((angles_rad.shape[0],), np.nan, dtype=np.float32)
    ray_hits = np.zeros_like(patch_uint8, dtype=np.uint8)

    for angle_idx, angle in enumerate(angles_rad.tolist()):
        xs = center_x + radii * float(np.cos(angle))
        ys = center_y + radii * float(np.sin(angle))
        xs_i = np.rint(xs).astype(np.int32)
        ys_i = np.rint(ys).astype(np.int32)
        valid = (
            (xs_i >= 0)
            & (xs_i < int(patch_uint8.shape[1]))
            & (ys_i >= 0)
            & (ys_i < int(patch_uint8.shape[0]))
        )
        if not np.any(valid):
            continue
        xs_valid = xs_i[valid]
        ys_valid = ys_i[valid]
        values = response_norm[ys_valid, xs_valid]
        best_idx = int(np.argmax(values))
        best_score = float(values[best_idx])
        if best_score < response_threshold:
            continue
        chosen_radii[angle_idx] = float(radii[valid][best_idx])
        chosen_scores[angle_idx] = best_score
        ray_hits[int(ys_valid[best_idx]), int(xs_valid[best_idx])] = 1

    valid_fraction = float(np.mean(np.isfinite(chosen_radii))) if chosen_radii.size else 0.0
    max_missing_gap = int(_longest_nan_run(chosen_radii) * angle_step)

    proposal_mask = np.zeros_like(patch_uint8, dtype=np.uint8)
    bbox_xyxy = None
    centroid_xy = None
    selected_area = 0

    min_valid_fraction = float(np.clip(params.get("min_valid_ray_fraction", 0.55), 0.0, 1.0))
    max_gap_allowed = max(0, int(params.get("max_missing_gap_degrees", 40)))
    if (
        chosen_radii.size >= 3
        and valid_fraction >= min_valid_fraction
        and max_missing_gap <= max_gap_allowed
    ):
        filled_radii = _interpolate_circular_nans(chosen_radii)
        smoothed_radii = _circular_gaussian_smooth(
            filled_radii,
            float(params.get("smoothing_sigma", 1.5)),
        )
        polygon = np.stack(
            [
                center_x + smoothed_radii * np.cos(angles_rad),
                center_y + smoothed_radii * np.sin(angles_rad),
            ],
            axis=1,
        )
        polygon_i = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(proposal_mask, [polygon_i], 1)

        labeled = measure.label(proposal_mask.astype(np.uint8), connectivity=1)
        props = measure.regionprops(labeled)
        if props:
            region = max(props, key=lambda item: item.area)
            selected_area = int(region.area)
            min_row, min_col, max_row, max_col = region.bbox
            bbox_xyxy = [int(min_col), int(min_row), int(max_col), int(max_row)]
            centroid_xy = [float(region.centroid[1]), float(region.centroid[0])]

    response_uint8 = (np.clip(response_norm, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    prefiltered_uint8 = (np.clip(prefiltered, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    return {
        "filtered_patch": response_uint8,
        "filtered_patch_title": "Boundary Response",
        "sobel_panel": prefiltered_uint8,
        "sobel_panel_title": "Prefiltered Patch",
        "seed_mask": ray_hits,
        "seed_mask_title": "Ray Hits",
        "proposal_mask": proposal_mask,
        "probability_patch": np.asarray(response_norm, dtype=np.float32),
        "stats": {
            "selected_area": selected_area,
            "circularity": None,
            "threshold_value": None,
            "bbox_xyxy": bbox_xyxy,
            "selected_centroid_xy": centroid_xy,
            "valid_ray_fraction": valid_fraction,
            "max_missing_gap_degrees": max_missing_gap,
            "max_response": float(np.nanmax(chosen_scores)) if np.any(np.isfinite(chosen_scores)) else 0.0,
        },
    }


def _compute_threshold_patch_preview(
    patch: np.ndarray,
    *,
    center_xy: tuple[float, float],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    patch_uint8 = np.asarray(patch, dtype=np.uint8)
    filtered_patch, sobel_panel = eye_mask_ops.apply_sobel_filter(
        patch_uint8,
        float(params["sobel_strength"]),
    )
    _binary_seed, threshold_value = eye_mask_ops.global_mask(filtered_patch)
    binary = np.asarray(filtered_patch, dtype=np.uint8) <= int(round(float(threshold_value)))
    pre_threshold = params.get("pre_threshold")
    if pre_threshold is not None:
        binary = np.logical_and(binary, filtered_patch < int(pre_threshold))

    region_mask = eye_mask_ops.select_region(
        binary,
        center_xy,
        int(params["min_area"]),
        int(params["max_area"]) if params.get("max_area") is not None else None,
        float(params["min_circularity"]) if params.get("min_circularity") is not None else None,
        int(params["closing_radius"]),
        int(params["opening_radius"]),
    )

    proposal_mask = np.zeros_like(patch_uint8, dtype=np.uint8)
    stats: Dict[str, Any] = {
        "selected_area": 0,
        "circularity": None,
        "threshold_value": float(threshold_value),
        "bbox_xyxy": None,
        "selected_centroid_xy": None,
    }
    if region_mask is not None:
        proposal_mask[region_mask] = 1
        labeled = measure.label(region_mask.astype(np.uint8), connectivity=1)
        props = measure.regionprops(labeled)
        if props:
            region = props[0]
            perimeter = float(region.perimeter)
            circularity = (
                float((4.0 * np.pi * float(region.area)) / (perimeter * perimeter))
                if perimeter > 0.0
                else None
            )
            min_row, min_col, max_row, max_col = region.bbox
            stats.update(
                {
                    "selected_area": int(region.area),
                    "circularity": circularity,
                    "bbox_xyxy": [int(min_col), int(min_row), int(max_col), int(max_row)],
                    "selected_centroid_xy": [float(region.centroid[1]), float(region.centroid[0])],
                }
            )

    return {
        "filtered_patch": np.asarray(filtered_patch, dtype=np.uint8),
        "filtered_patch_title": "Filtered Patch",
        "sobel_panel": np.asarray(sobel_panel, dtype=np.uint8) if sobel_panel is not None else None,
        "sobel_panel_title": "Sobel Response",
        "seed_mask": np.asarray(binary, dtype=np.uint8),
        "seed_mask_title": "Binary Seed",
        "proposal_mask": proposal_mask,
        "probability_patch": 1.0 - (np.asarray(filtered_patch, dtype=np.float32) / 255.0),
        "stats": stats,
    }


def _compute_swim_bladder_patch_preview(
    patch: np.ndarray,
    *,
    center_xy: tuple[float, float],
    params: Mapping[str, Any],
    method_family: str = DEFAULT_METHOD_FAMILY,
) -> Dict[str, Any]:
    normalized_family = _normalize_method_family(method_family)
    patch_uint8 = np.asarray(patch, dtype=np.uint8)
    if normalized_family == POLAR_BOUNDARY_METHOD_FAMILY:
        preview = _compute_polar_boundary_preview(
            patch_uint8,
            center_xy=center_xy,
            params=params,
        )
    else:
        preview = _compute_threshold_patch_preview(
            patch_uint8,
            center_xy=center_xy,
            params=params,
        )
    preview["method_family"] = normalized_family
    return preview


def _draw_patch_overlay(patch: np.ndarray, mask: np.ndarray, *, center_xy: tuple[float, float]) -> np.ndarray:
    base = cv2.cvtColor(np.asarray(patch, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[np.asarray(mask, dtype=np.uint8) > 0] = (255, 220, 0)
    out = cv2.addWeighted(overlay, 0.45, base, 0.55, 0.0)
    cv2.drawMarker(
        out,
        (int(round(center_xy[0])), int(round(center_xy[1]))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        10,
        1,
    )
    return out


def _draw_roi_context(
    roi_img: np.ndarray,
    *,
    center_xy: tuple[float, float],
    bounds_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    panel = cv2.cvtColor(np.asarray(roi_img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    x0, x1, y0, y1 = bounds_xyxy
    cv2.rectangle(panel, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 220, 0), 1)
    cv2.drawMarker(
        panel,
        (int(round(center_xy[0])), int(round(center_xy[1]))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        10,
        1,
    )
    return panel


def _build_canvas(
    *,
    roi_img: np.ndarray,
    patch: np.ndarray,
    filtered_patch: np.ndarray,
    seed_mask: np.ndarray,
    proposal_mask: np.ndarray,
    center_xy: tuple[float, float],
    patch_center_xy: tuple[float, float],
    patch_bounds: tuple[int, int, int, int],
    existing_patch: Optional[np.ndarray],
    existing_title: str,
    sobel_panel: Optional[np.ndarray],
    filtered_patch_title: str,
    sobel_panel_title: str,
    seed_mask_title: str,
    panel_size: int,
    footer_lines: Sequence[str],
) -> np.ndarray:
    proposal_overlay = _draw_patch_overlay(patch, proposal_mask, center_xy=patch_center_xy)
    roi_context = _draw_roi_context(roi_img, center_xy=center_xy, bounds_xyxy=patch_bounds)

    panels_top = [
        _prepare_panel(roi_context, "Crop ROI", panel_size),
        _prepare_panel(patch, "Swim Bladder Patch", panel_size),
        _prepare_panel(
            np.asarray(existing_patch, dtype=np.uint8) * 255 if existing_patch is not None else np.zeros_like(patch),
            existing_title,
            panel_size,
            nearest=True,
        ),
        _prepare_panel(proposal_overlay, "Proposal Overlay", panel_size),
    ]
    panels_bottom = [
        _prepare_panel(filtered_patch, filtered_patch_title, panel_size),
        _prepare_panel(
            sobel_panel if sobel_panel is not None else np.zeros_like(filtered_patch),
            sobel_panel_title,
            panel_size,
        ),
        _prepare_panel(seed_mask * 255, seed_mask_title, panel_size, nearest=True),
        _prepare_panel(proposal_mask * 255, "Proposal Mask", panel_size, nearest=True),
    ]

    top_row = cv2.hconcat(panels_top)
    bottom_row = cv2.hconcat(panels_bottom)
    grid = cv2.vconcat([top_row, bottom_row])

    footer = np.zeros((110, grid.shape[1], 3), dtype=np.uint8)
    for idx, line in enumerate(footer_lines[:4]):
        cv2.putText(
            footer,
            str(line),
            (10, 24 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return cv2.vconcat([grid, footer])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune local traditional swim-bladder segmentation parameters.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette zarr archive.")
    parser.add_argument("--crop-run", help="Crop run to use (default: latest).")
    parser.add_argument("--keypoint-run", help="Optional keypoint run to anchor swim bladder patch center.")
    parser.add_argument("--subject-run", help="Optional subject_mask_runs/<run> to preview stored swim-bladder masks.")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index.")
    parser.add_argument(
        "--method-family",
        help=(
            "Optional swim-bladder tuning family override. "
            "Examples: threshold_blob, patch_threshold, polar_boundary."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    root = open_zarr_root(zarr_path, mode="r")
    crop_parent = root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group):
        raise RuntimeError("No crop_runs group found.")
    crop_run = _resolve_run_name(root, "crop_runs", args.crop_run)
    crop_group = crop_parent[crop_run]
    roi_images = crop_group.get("roi_images")
    if roi_images is None:
        raise RuntimeError(f"crop_runs/{crop_run} missing roi_images.")
    total_rois = int(roi_images.shape[0])
    if total_rois <= 0:
        raise RuntimeError("No ROIs available in crop run.")

    frame_indices = np.asarray(crop_group["frame_indices"][:], dtype=np.int32) if "frame_indices" in crop_group else None
    detection_indices = (
        np.asarray(crop_group["detection_indices"][:], dtype=np.int32) if "detection_indices" in crop_group else None
    )

    keypoint_source = subject_tuner._resolve_eye_keypoint_source(root, args.keypoint_run)
    keypoints_roi = np.asarray(keypoint_source.keypoints_roi[:], dtype=np.float32)
    keypoint_labels_raw = keypoint_source.group.attrs.get("keypoint_labels")
    keypoint_labels = (
        [str(item) for item in keypoint_labels_raw]
        if isinstance(keypoint_labels_raw, (list, tuple))
        else None
    )
    if int(keypoints_roi.shape[0]) != total_rois:
        raise RuntimeError(
            f"Keypoint rows {int(keypoints_roi.shape[0])} do not match crop ROI rows {total_rois}."
        )

    subject_source = _resolve_subject_source(root, args.subject_run)
    subject_component_index = (
        subject_tuner._find_subject_label_index(subject_source, "swim_bladder")
        if subject_source is not None
        else None
    )

    tuning_entry = subject_tuner._load_subject_tuning_entry_from_root(root, "swim_bladder")
    saved_method_family = (
        tuning_entry.get("subject_method_family")
        if isinstance(tuning_entry, Mapping)
        else None
    )
    active_method_family = _normalize_method_family(args.method_family or saved_method_family)
    tuned_params = tuning_entry.get("tuned_parameters") if isinstance(tuning_entry, Mapping) else None
    params = _build_swim_bladder_params(tuned_params if isinstance(tuned_params, Mapping) else None, method_family=active_method_family)

    subject_tuner._require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1000)

    def _noop(_value: int) -> None:
        return

    cv2.createTrackbar("ROI Index", WINDOW_NAME, max(0, min(int(args.roi_index), total_rois - 1)), total_rois - 1, _noop)
    cv2.createTrackbar("Patch Pad", WINDOW_NAME, min(int(params["roi_padding"]), SLIDER_MAX_PADDING), SLIDER_MAX_PADDING, _noop)
    if active_method_family == POLAR_BOUNDARY_METHOD_FAMILY:
        cv2.createTrackbar(
            "Angle Step",
            WINDOW_NAME,
            min(int(params["angle_step_degrees"]), SLIDER_MAX_ANGLE_STEP),
            SLIDER_MAX_ANGLE_STEP,
            _noop,
        )
        cv2.createTrackbar(
            "Min Radius",
            WINDOW_NAME,
            min(int(params["min_radius_px"]), SLIDER_MAX_PADDING),
            SLIDER_MAX_PADDING,
            _noop,
        )
        cv2.createTrackbar(
            "Max Radius",
            WINDOW_NAME,
            min(int(params["max_radius_px"]), SLIDER_MAX_PADDING),
            SLIDER_MAX_PADDING,
            _noop,
        )
        cv2.createTrackbar(
            "Smooth %",
            WINDOW_NAME,
            int(round(float(params["smoothing_sigma"]) * 10.0)),
            SLIDER_MAX_SIGMA,
            _noop,
        )
        cv2.createTrackbar(
            "Resp %",
            WINDOW_NAME,
            int(round(float(params["response_threshold"]) * 100.0)),
            SLIDER_MAX_RESPONSE,
            _noop,
        )
        cv2.createTrackbar(
            "Max Gap",
            WINDOW_NAME,
            min(int(params["max_missing_gap_degrees"]), SLIDER_MAX_GAP_DEGREES),
            SLIDER_MAX_GAP_DEGREES,
            _noop,
        )
        cv2.createTrackbar(
            "Min Valid %",
            WINDOW_NAME,
            int(round(float(params["min_valid_ray_fraction"]) * 100.0)),
            SLIDER_MAX_VALID_FRAC,
            _noop,
        )
        cv2.createTrackbar(
            "Prefilter %",
            WINDOW_NAME,
            int(round(float(params["prefilter_sigma"]) * 10.0)),
            SLIDER_MAX_SIGMA,
            _noop,
        )
        synced_pad, synced_min, synced_max = _sync_polar_radius_constraints(
            cv2.getTrackbarPos("Patch Pad", WINDOW_NAME),
            cv2.getTrackbarPos("Min Radius", WINDOW_NAME),
            cv2.getTrackbarPos("Max Radius", WINDOW_NAME),
        )
        cv2.setTrackbarPos("Patch Pad", WINDOW_NAME, synced_pad)
        cv2.setTrackbarPos("Min Radius", WINDOW_NAME, synced_min)
        cv2.setTrackbarPos("Max Radius", WINDOW_NAME, synced_max)
    else:
        cv2.createTrackbar(
            "PreThresh",
            WINDOW_NAME,
            int(params["pre_threshold"]) if params.get("pre_threshold") is not None else 0,
            SLIDER_MAX_PRETHRESH,
            _noop,
        )
        cv2.createTrackbar(
            "Sobel %",
            WINDOW_NAME,
            int(round(float(params["sobel_strength"]) * SLIDER_MAX_SOBEL)),
            SLIDER_MAX_SOBEL,
            _noop,
        )
        cv2.createTrackbar("Min Area", WINDOW_NAME, min(int(params["min_area"]), SLIDER_MAX_AREA), SLIDER_MAX_AREA, _noop)
        cv2.createTrackbar(
            "Max Area",
            WINDOW_NAME,
            int(params["max_area"]) if params.get("max_area") is not None else 0,
            SLIDER_MAX_AREA,
            _noop,
        )
        cv2.createTrackbar(
            "Min Circ %",
            WINDOW_NAME,
            int(round(float(params["min_circularity"]) * SLIDER_MAX_CIRCULARITY))
            if params.get("min_circularity") is not None
            else 0,
            SLIDER_MAX_CIRCULARITY,
            _noop,
        )
        cv2.createTrackbar(
            "Closing r",
            WINDOW_NAME,
            min(int(params["closing_radius"]), SLIDER_MAX_RADIUS),
            SLIDER_MAX_RADIUS,
            _noop,
        )
        cv2.createTrackbar(
            "Opening r",
            WINDOW_NAME,
            min(int(params["opening_radius"]), SLIDER_MAX_RADIUS),
            SLIDER_MAX_RADIUS,
            _noop,
        )

    current_roi = max(0, min(int(args.roi_index), total_rois - 1))

    print("\nStarting Swim Bladder Mask Tuner")
    print(f"  crop_run={crop_run}")
    print(f"  keypoint_run={keypoint_source.group_name}/{keypoint_source.run_name}")
    print(f"  method_family={active_method_family}")
    if subject_source is not None:
        if int(subject_source.masks_roi.shape[0]) != total_rois:
            raise RuntimeError(
                f"Subject-mask rows {int(subject_source.masks_roi.shape[0])} do not match crop ROI rows {total_rois}."
            )
        print(f"  subject_run={subject_source.run_name}")
    print("Controls:")
    print("  - ROI trackbar or n/p, j/k to navigate crops")
    if active_method_family == POLAR_BOUNDARY_METHOD_FAMILY:
        print("  - Adjust patch, radial boundary, and response sliders")
        print("  - Patch Pad is the hard upper bound for Max Radius")
    else:
        print("  - Adjust patch, threshold, Sobel, morphology, and area sliders")
    print("  - Press 's' to save swim-bladder tuning to analysis_metadata")
    print("  - Press 'q' or ESC to quit")

    while True:
        roi_idx = int(cv2.getTrackbarPos("ROI Index", WINDOW_NAME))
        if roi_idx != current_roi:
            current_roi = roi_idx

        if active_method_family == POLAR_BOUNDARY_METHOD_FAMILY:
            pad_value, min_radius_value, max_radius_value = _sync_polar_radius_constraints(
                cv2.getTrackbarPos("Patch Pad", WINDOW_NAME),
                cv2.getTrackbarPos("Min Radius", WINDOW_NAME),
                cv2.getTrackbarPos("Max Radius", WINDOW_NAME),
            )
            if pad_value != cv2.getTrackbarPos("Patch Pad", WINDOW_NAME):
                cv2.setTrackbarPos("Patch Pad", WINDOW_NAME, pad_value)
            if min_radius_value != cv2.getTrackbarPos("Min Radius", WINDOW_NAME):
                cv2.setTrackbarPos("Min Radius", WINDOW_NAME, min_radius_value)
            if max_radius_value != cv2.getTrackbarPos("Max Radius", WINDOW_NAME):
                cv2.setTrackbarPos("Max Radius", WINDOW_NAME, max_radius_value)
            current_params = _normalize_swim_bladder_params(
                {
                    "roi_padding": pad_value,
                    "angle_step_degrees": cv2.getTrackbarPos("Angle Step", WINDOW_NAME),
                    "min_radius_px": min_radius_value,
                    "max_radius_px": max_radius_value,
                    "smoothing_sigma": float(cv2.getTrackbarPos("Smooth %", WINDOW_NAME)) / 10.0,
                    "response_threshold": float(cv2.getTrackbarPos("Resp %", WINDOW_NAME)) / 100.0,
                    "max_missing_gap_degrees": cv2.getTrackbarPos("Max Gap", WINDOW_NAME),
                    "min_valid_ray_fraction": float(cv2.getTrackbarPos("Min Valid %", WINDOW_NAME)) / 100.0,
                    "prefilter_sigma": float(cv2.getTrackbarPos("Prefilter %", WINDOW_NAME)) / 10.0,
                    "gradient_mode": "sobel_magnitude",
                },
                method_family=active_method_family,
            )
        else:
            current_params = _normalize_swim_bladder_params(
                {
                    "roi_padding": cv2.getTrackbarPos("Patch Pad", WINDOW_NAME),
                    "pre_threshold": cv2.getTrackbarPos("PreThresh", WINDOW_NAME) or None,
                    "sobel_strength": float(cv2.getTrackbarPos("Sobel %", WINDOW_NAME)) / float(SLIDER_MAX_SOBEL),
                    "min_area": cv2.getTrackbarPos("Min Area", WINDOW_NAME),
                    "max_area": cv2.getTrackbarPos("Max Area", WINDOW_NAME) or None,
                    "min_circularity": (
                        float(cv2.getTrackbarPos("Min Circ %", WINDOW_NAME)) / float(SLIDER_MAX_CIRCULARITY)
                        if cv2.getTrackbarPos("Min Circ %", WINDOW_NAME) > 0
                        else None
                    ),
                    "closing_radius": cv2.getTrackbarPos("Closing r", WINDOW_NAME),
                    "opening_radius": cv2.getTrackbarPos("Opening r", WINDOW_NAME),
                },
                method_family=active_method_family,
            )

        roi_img = np.asarray(roi_images[current_roi], dtype=np.uint8)
        keypoints_row = keypoints_roi[current_roi]
        center_xy, center_source = _resolve_swim_bladder_center_with_source(
            keypoints_row,
            keypoint_labels,
            np.zeros(roi_img.shape, dtype=np.uint8),
            tuple(roi_img.shape),
        )
        x0, x1, y0, y1 = _extract_patch_bounds(tuple(roi_img.shape), center_xy, int(current_params["roi_padding"]))
        patch = np.asarray(roi_img[y0:y1, x0:x1], dtype=np.uint8)
        patch_center_xy = (float(center_xy[0]) - float(x0), float(center_xy[1]) - float(y0))
        preview = _compute_swim_bladder_patch_preview(
            patch,
            center_xy=patch_center_xy,
            params=current_params,
            method_family=active_method_family,
        )

        existing_patch = None
        existing_title = "Stored Mask (none)"
        if subject_source is not None and subject_component_index is not None:
            existing_mask = np.asarray(subject_source.masks_roi[current_roi, subject_component_index], dtype=np.uint8)
            existing_patch = existing_mask[y0:y1, x0:x1]
            existing_title = f"Stored Mask ({subject_source.run_name})"

        stats = preview["stats"]
        if active_method_family == POLAR_BOUNDARY_METHOD_FAMILY:
            footer_lines = [
                (
                    f"ROI {current_roi + 1}/{total_rois} | frame="
                    f"{int(frame_indices[current_roi]) if frame_indices is not None else '-'} | detection="
                    f"{int(detection_indices[current_roi]) if detection_indices is not None else '-'} | center={center_source}"
                ),
                (
                    f"pad={int(current_params['roi_padding'])} | angle_step={int(current_params['angle_step_degrees'])} | "
                    f"radius=[{int(current_params['min_radius_px'])},{int(current_params['max_radius_px'])}] | "
                    f"prefilter={float(current_params['prefilter_sigma']):.1f}"
                ),
                (
                    f"resp_thr={float(current_params['response_threshold']):.2f} | "
                    f"smooth={float(current_params['smoothing_sigma']):.1f} | "
                    f"max_gap={int(current_params['max_missing_gap_degrees'])} | "
                    f"min_valid={float(current_params['min_valid_ray_fraction']):.2f}"
                ),
                (
                    f"selected_area={int(stats['selected_area'])} | "
                    f"valid_frac={float(stats.get('valid_ray_fraction') or 0.0):.2f} | "
                    f"max_gap={int(stats.get('max_missing_gap_degrees') or 0)} | "
                    "Controls: n/p next-prev, j/k +/-10 ROI, s save tuning, q/ESC quit"
                ),
            ]
        else:
            min_circularity_text = (
                f"{float(current_params['min_circularity']):.2f}"
                if current_params["min_circularity"] is not None
                else "None"
            )
            selected_circularity_text = (
                f"{float(stats['circularity']):.2f}"
                if stats["circularity"] is not None
                else "None"
            )
            footer_lines = [
                (
                    f"ROI {current_roi + 1}/{total_rois} | frame="
                    f"{int(frame_indices[current_roi]) if frame_indices is not None else '-'} | detection="
                    f"{int(detection_indices[current_roi]) if detection_indices is not None else '-'} | center={center_source}"
                ),
                (
                    f"pad={int(current_params['roi_padding'])} | pre_thr={current_params['pre_threshold'] or 'None'} | "
                    f"sobel={float(current_params['sobel_strength']):.2f} | close={int(current_params['closing_radius'])} | "
                    f"open={int(current_params['opening_radius'])}"
                ),
                (
                    f"min_area={int(current_params['min_area'])} | max_area={current_params['max_area'] or 'None'} | "
                    f"min_circ={min_circularity_text} | "
                    f"otsu={float(stats['threshold_value']):.1f}"
                ),
                (
                    f"selected_area={int(stats['selected_area'])} | "
                    f"circularity={selected_circularity_text} | "
                    "Controls: n/p next-prev, j/k +/-10 ROI, s save tuning, q/ESC quit"
                ),
            ]

        canvas = _build_canvas(
            roi_img=roi_img,
            patch=patch,
            filtered_patch=preview["filtered_patch"],
            seed_mask=preview["seed_mask"],
            proposal_mask=preview["proposal_mask"],
            center_xy=center_xy,
            patch_center_xy=patch_center_xy,
            patch_bounds=(x0, x1, y0, y1),
            existing_patch=existing_patch,
            existing_title=existing_title,
            sobel_panel=preview["sobel_panel"],
            filtered_patch_title=str(preview.get("filtered_patch_title") or "Filtered Patch"),
            sobel_panel_title=str(preview.get("sobel_panel_title") or "Debug Panel"),
            seed_mask_title=str(preview.get("seed_mask_title") or "Seed Mask"),
            panel_size=DEFAULT_PANEL_SIZE,
            footer_lines=footer_lines,
        )
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and current_roi < total_rois - 1:
            current_roi += 1
            cv2.setTrackbarPos("ROI Index", WINDOW_NAME, current_roi)
            continue
        if key == ord("p") and current_roi > 0:
            current_roi -= 1
            cv2.setTrackbarPos("ROI Index", WINDOW_NAME, current_roi)
            continue
        if key == ord("j"):
            current_roi = max(0, current_roi - 10)
            cv2.setTrackbarPos("ROI Index", WINDOW_NAME, current_roi)
            continue
        if key == ord("k"):
            current_roi = min(total_rois - 1, current_roi + 10)
            cv2.setTrackbarPos("ROI Index", WINDOW_NAME, current_roi)
            continue
        if key == ord("s"):
            method_label = (
                "polar_boundary_center_seed"
                if active_method_family == POLAR_BOUNDARY_METHOD_FAMILY
                else "global_threshold_otsu"
            )
            save_ok, message = subject_tuner.save_subject_mask_params(
                zarr_path,
                current_params,
                context={
                    "roi_index": int(current_roi),
                    "roi_index_one_based": int(current_roi) + 1,
                    "total_rois": int(total_rois),
                    "crop_run": str(crop_run),
                    "keypoint_run": str(keypoint_source.run_name),
                    "keypoint_source": str(keypoint_source.group_name),
                    "frame_index": int(frame_indices[current_roi]) if frame_indices is not None else None,
                    "detection_index": int(detection_indices[current_roi]) if detection_indices is not None else None,
                    "roi_success": bool(np.all(np.isfinite(np.asarray(center_xy, dtype=np.float32)))),
                    "patch_center_source": str(center_source),
                    "storage_component_name": "swim_bladder",
                    "subject_preview_run": str(subject_source.run_name) if subject_source is not None else None,
                },
                component_name="swim_bladder",
                method=method_label,
                version="1.0",
                extra_entry_fields={
                    "subject_method_family": active_method_family,
                    "output_labels": ["swim_bladder"],
                    "storage_component": "swim_bladder",
                },
                normalize_params=False,
            )
            print(f"✓ {message}" if save_ok else f"✗ {message}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
