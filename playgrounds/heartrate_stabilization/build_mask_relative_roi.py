from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    cfg_path,
    cfg_value,
    ensure_output_dir,
    get_video_info,
    load_config,
    load_keypoint_data,
    read_crop_meta,
    selected_crop_rows,
)
from map_pixel_band_contributions import (
    _load_mask_component,
    _split_components,
    _stable_component_mask_counts,
    _video_shape,
)


def _occupancy_fraction(counts: np.ndarray, frames: int, *, component: str) -> np.ndarray:
    if int(frames) <= 0:
        raise ValueError(f"No valid mask projections for {component}.")
    return counts.astype(np.float32) / float(frames)


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if int(pixels) <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    return ndimage.binary_erosion(np.asarray(mask, dtype=bool), iterations=int(pixels))


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if int(pixels) <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), iterations=int(pixels))


def _clipped_span(center_or_start: float, length: int, limit: int, *, centered: bool) -> tuple[int, int]:
    length = max(1, int(length))
    limit = max(1, int(limit))
    if centered:
        start = int(round(float(center_or_start) - float(length) / 2.0))
    else:
        start = int(round(float(center_or_start)))
    stop = start + length
    if start < 0:
        stop -= start
        start = 0
    if stop > limit:
        start = max(0, start - (stop - limit))
        stop = limit
    if stop <= start:
        raise ValueError(f"Could not fit span length={length} inside limit={limit}.")
    return int(start), int(stop)


def _median_x_in_rows(mask: np.ndarray, y0: int, y1: int) -> float | None:
    y0 = max(0, int(y0))
    y1 = min(mask.shape[0], int(y1))
    if y1 <= y0:
        return None
    _yy, xx = np.nonzero(mask[y0:y1])
    if xx.size == 0:
        return None
    return float(np.median(xx))


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if xx.size == 0:
        return None
    return float(np.mean(xx)), float(np.mean(yy))


def _eye_midpoint_center(
    *,
    eye_component_fractions: dict[str, np.ndarray],
    eye_threshold: float,
) -> tuple[float | None, dict[str, Any]]:
    centroids: list[tuple[str, float, float, int]] = []
    for component, fraction in eye_component_fractions.items():
        mask = np.asarray(fraction >= float(eye_threshold), dtype=bool)
        centroid = _mask_centroid(mask)
        if centroid is None:
            continue
        x, y = centroid
        centroids.append((component, x, y, int(np.count_nonzero(mask))))
    details = {
        "component_centroids": [
            {"component": name, "x": float(x), "y": float(y), "pixels": int(pixels)}
            for name, x, y, pixels in centroids
        ],
        "component_count_used": int(len(centroids)),
    }
    if len(centroids) < 2:
        return None, details
    x_values = np.asarray([x for _name, x, _y, _pixels in centroids], dtype=np.float64)
    y_values = np.asarray([y for _name, _x, y, _pixels in centroids], dtype=np.float64)
    details["midpoint_x"] = float(np.mean(x_values))
    details["midpoint_y"] = float(np.mean(y_values))
    return float(np.mean(x_values)), details


def _body_center_fallback(
    *,
    body_mask: np.ndarray,
    top_y: int,
    height_px: int,
    center_window_px: int,
) -> tuple[float, str]:
    center_window = max(0, int(center_window_px))
    center_x = _median_x_in_rows(body_mask, top_y - center_window, top_y + center_window + 1)
    center_source = "body_rows_near_top"
    if center_x is None:
        center_x = _median_x_in_rows(body_mask, top_y, top_y + int(height_px))
        center_source = "body_rows_inside_roi_height"
    if center_x is None:
        lower_body = body_mask.copy()
        lower_body[: max(0, top_y), :] = False
        _yy, xx = np.nonzero(lower_body)
        if xx.size:
            center_x = float(np.median(xx))
            center_source = "body_rows_below_top"
    if center_x is None:
        _yy, xx = np.nonzero(body_mask)
        center_x = float(np.median(xx))
        center_source = "all_body_rows"
    return float(center_x), center_source


def _derive_mask_relative_roi(
    *,
    body_fraction: np.ndarray,
    eye_fraction: np.ndarray,
    eye_component_fractions: dict[str, np.ndarray],
    body_threshold: float,
    eye_threshold: float,
    eye_bottom_quantile: float,
    top_offset_px: float,
    width_px: int,
    height_px: int,
    center_mode: str,
    center_window_px: int,
    body_erode_px: int,
    eye_dilate_px: int,
) -> dict[str, Any]:
    body_mask = np.asarray(body_fraction >= float(body_threshold), dtype=bool)
    eye_mask = np.asarray(eye_fraction >= float(eye_threshold), dtype=bool)
    if int(np.count_nonzero(body_mask)) == 0:
        raise ValueError("The projected body occupancy mask is empty at the requested threshold.")
    if int(np.count_nonzero(eye_mask)) == 0:
        raise ValueError("The projected eye occupancy mask is empty at the requested threshold.")

    eye_y, _eye_x = np.nonzero(eye_mask)
    q = float(np.clip(eye_bottom_quantile, 0.0, 1.0))
    eye_bottom_y = float(np.quantile(eye_y.astype(np.float64), q))
    top_y = int(np.ceil(eye_bottom_y + float(top_offset_px)))

    normalized_center_mode = str(center_mode).strip().lower()
    if normalized_center_mode not in {"eye_midpoint", "body_rows_near_top"}:
        raise ValueError(f"Unsupported center_mode={center_mode!r}")
    eye_center_x, eye_center_details = _eye_midpoint_center(
        eye_component_fractions=eye_component_fractions,
        eye_threshold=float(eye_threshold),
    )
    if normalized_center_mode == "eye_midpoint" and eye_center_x is not None:
        center_x = float(eye_center_x)
        center_source = "eye_mask_midpoint"
    else:
        center_x, center_source = _body_center_fallback(
            body_mask=body_mask,
            top_y=top_y,
            height_px=int(height_px),
            center_window_px=int(center_window_px),
        )
        if normalized_center_mode == "eye_midpoint":
            center_source = f"{center_source}_fallback"

    height, width = body_mask.shape
    x0, x1 = _clipped_span(center_x, int(width_px), width, centered=True)
    y0, y1 = _clipped_span(top_y, int(height_px), height, centered=False)
    diagnostic_rect_mask = np.zeros_like(body_mask, dtype=bool)
    diagnostic_rect_mask[y0:y1, x0:x1] = True

    body_interior = _erode(body_mask, int(body_erode_px))
    eye_exclusion = _dilate(eye_mask, int(eye_dilate_px))
    candidate_mask = diagnostic_rect_mask & body_interior & ~eye_exclusion
    if int(np.count_nonzero(candidate_mask)) == 0 and int(body_erode_px) > 0:
        relaxed_body = body_mask
        relaxed_mask = diagnostic_rect_mask & relaxed_body & ~eye_exclusion
        if int(np.count_nonzero(relaxed_mask)) > 0:
            candidate_mask = relaxed_mask
            body_erode_used = 0
        else:
            body_erode_used = int(body_erode_px)
    else:
        body_erode_used = int(body_erode_px)
    if int(np.count_nonzero(candidate_mask)) == 0:
        raise ValueError(
            "The mask-relative ROI contains no sample pixels. "
            "Try reducing --body-erode-px/--eye-dilate-px or moving --top-offset-px farther down."
        )

    return {
        "candidate_mask": candidate_mask,
        "body_mask": body_mask,
        "body_interior": body_interior if body_erode_used == int(body_erode_px) else body_mask,
        "eye_mask": eye_mask,
        "eye_exclusion": eye_exclusion,
        "diagnostic_rect_mask": diagnostic_rect_mask,
        "bbox_xyxy": np.asarray([x0, y0, x1, y1], dtype=np.int32),
        "roi_rect_xywh": np.asarray([x0, y0, x1 - x0, y1 - y0], dtype=np.float32),
        "eye_bottom_y": float(eye_bottom_y),
        "roi_top_y": int(y0),
        "center_x": float(center_x),
        "center_mode": normalized_center_mode,
        "center_source": center_source,
        "eye_center": eye_center_details,
        "body_erode_px_used": int(body_erode_used),
    }


def _draw_diagnostic(
    *,
    path: Path,
    body_fraction: np.ndarray,
    eye_fraction: np.ndarray,
    body_mask: np.ndarray,
    eye_mask: np.ndarray,
    candidate_mask: np.ndarray,
    diagnostic_rect_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    eye_bottom_y: float,
) -> None:
    import cv2

    ensure_output_dir(path.parent)
    base = np.clip(body_fraction, 0.0, 1.0)
    image = np.repeat((base * 110.0).astype(np.uint8)[:, :, None], 3, axis=2)
    image[body_mask] = np.maximum(image[body_mask], np.asarray([70, 70, 70], dtype=np.uint8))
    eye_alpha = np.clip(eye_fraction, 0.0, 1.0)
    eye_pixels = eye_alpha > 0
    image[eye_pixels, 0] = np.maximum(image[eye_pixels, 0], (eye_alpha[eye_pixels] * 230).astype(np.uint8))
    image[eye_mask] = np.asarray([255, 80, 40], dtype=np.uint8)
    image[diagnostic_rect_mask] = np.maximum(
        image[diagnostic_rect_mask],
        np.asarray([30, 130, 130], dtype=np.uint8),
    )
    image[candidate_mask] = np.asarray([0, 255, 255], dtype=np.uint8)

    x0, y0, x1, y1 = [int(value) for value in bbox_xyxy.tolist()]
    cv2.rectangle(image, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 1)
    y_eye = int(round(float(eye_bottom_y)))
    if 0 <= y_eye < image.shape[0]:
        cv2.line(image, (0, y_eye), (image.shape[1] - 1, y_eye), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def _projection_context(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))
    frame_array = str(cfg_value(config, "alignment", "keypoint_frame_array", "frame_indices"))
    keypoint_array = str(cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img"))
    valid_array = str(cfg_value(config, "alignment", "keypoint_valid_array", "usable_keypoints"))
    stable_width = int(cfg_value(config, "alignment", "stable_width", 256))
    stable_height = int(cfg_value(config, "alignment", "stable_height", 256))
    stable_center_x = float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0))
    stable_center_y = float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0))
    origin = str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    if target_forward.strip().lower() != "up":
        raise ValueError(
            "build_mask_relative_roi.py currently assumes canonical target_forward='up', "
            f"but config has {target_forward!r}."
        )

    crop_rows = read_crop_meta(crop_meta_csv)
    frame_count = int(args.frame_count)
    if frame_count <= 0:
        frame_count = len(crop_rows)
    selected_all = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(frame_count)),
        stride=max(1, int(args.stride)),
    )
    selected = selected_all[:: max(1, int(args.mask_projection_stride))]
    if not selected:
        raise ValueError("No crop rows were selected for mask projection.")

    return {
        "crop_video": crop_video,
        "zarr_path": zarr_path,
        "keypoint_group": keypoint_group,
        "frame_id_column": frame_id_column,
        "keypoint_array": keypoint_array,
        "video": get_video_info(crop_video),
        "keypoints": load_keypoint_data(
            zarr_path,
            keypoint_group,
            frame_array=frame_array,
            keypoint_array=keypoint_array,
            valid_array=valid_array,
        ),
        "selected_all": selected_all,
        "selected": selected,
        "stable_width": stable_width,
        "stable_height": stable_height,
        "stable_center_x": stable_center_x,
        "stable_center_y": stable_center_y,
        "origin": origin,
        "target_forward": target_forward,
        "scale": float(cfg_value(config, "alignment", "scale", 1.0)),
        "min_forward": float(cfg_value(config, "alignment", "min_forward_length_px", 8.0)),
        "min_eye_span": float(cfg_value(config, "alignment", "min_eye_span_px", 4.0)),
    }


def _project_component_fraction(
    *,
    context: dict[str, Any],
    mask_parent: str,
    mask_run: str,
    component_name: str,
    dilate_px: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = _load_mask_component(
        context["zarr_path"],
        parent=mask_parent,
        run_name=mask_run,
        component_name=component_name,
    )
    counts, frames, reasons = _stable_component_mask_counts(
        mask_data=data,
        selected=context["selected"],
        keypoints=context["keypoints"],
        video=context["video"],
        frame_id_column=context["frame_id_column"],
        keypoint_coordinate_array=context["keypoint_array"],
        stable_width=context["stable_width"],
        stable_height=context["stable_height"],
        stable_center_x=context["stable_center_x"],
        stable_center_y=context["stable_center_y"],
        origin=context["origin"],
        target_forward=context["target_forward"],
        scale=context["scale"],
        min_forward=context["min_forward"],
        min_eye_span=context["min_eye_span"],
        dilate_px=int(dilate_px),
    )
    fraction = _occupancy_fraction(counts, frames, component=component_name)
    return fraction, {
        "source_path": data.source_path,
        "source_crop_run": data.source_crop_run,
        "storage_surface": data.storage_surface,
        "valid_projection_frames": int(frames),
        "reason_counts": reasons,
        "occupancy_pixels_any": int(np.count_nonzero(counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed candidate-pixel mask in canonical stabilized coordinates "
            "using projected subject body and eye masks."
        )
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, default=None, help="Optional stabilized video used only to validate output shape.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=6000, help="Crop-video frames to use; <=0 means all available rows.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mask-projection-stride", type=int, default=10)
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--body-occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--eye-occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--eye-bottom-quantile", type=float, default=0.98)
    parser.add_argument("--top-offset-px", type=float, default=2.0)
    parser.add_argument("--width-px", type=int, default=20)
    parser.add_argument("--height-px", type=int, default=12)
    parser.add_argument(
        "--center-mode",
        choices=("eye_midpoint", "body_rows_near_top"),
        default="eye_midpoint",
        help="How to choose the ROI x center. eye_midpoint uses projected eye-mask centroids with body fallback.",
    )
    parser.add_argument("--center-window-px", type=int, default=4)
    parser.add_argument("--body-erode-px", type=int, default=2)
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/mask_relative_roi"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    context = _projection_context(config, args)
    stable_shape = (int(context["stable_height"]), int(context["stable_width"]))
    if args.video is not None:
        video_shape = _video_shape(args.video)
        if tuple(video_shape) != tuple(stable_shape):
            raise ValueError(f"Video shape {video_shape} does not match configured stabilized shape {stable_shape}.")

    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))

    body_fraction, body_summary = _project_component_fraction(
        context=context,
        mask_parent=mask_parent,
        mask_run=mask_run,
        component_name=str(args.body_component),
    )
    eye_components = _split_components(args.eye_components)
    if not eye_components:
        raise ValueError("--eye-components must name at least one mask component.")
    eye_fractions: list[np.ndarray] = []
    eye_component_fractions: dict[str, np.ndarray] = {}
    eye_summaries: dict[str, Any] = {}
    for component in eye_components:
        fraction, component_summary = _project_component_fraction(
            context=context,
            mask_parent=mask_parent,
            mask_run=mask_run,
            component_name=component,
        )
        eye_fractions.append(fraction)
        eye_component_fractions[component] = fraction
        eye_summaries[component] = component_summary
    eye_fraction = np.maximum.reduce(eye_fractions)

    derived = _derive_mask_relative_roi(
        body_fraction=body_fraction,
        eye_fraction=eye_fraction,
        eye_component_fractions=eye_component_fractions,
        body_threshold=float(args.body_occupancy_threshold),
        eye_threshold=float(args.eye_occupancy_threshold),
        eye_bottom_quantile=float(args.eye_bottom_quantile),
        top_offset_px=float(args.top_offset_px),
        width_px=int(args.width_px),
        height_px=int(args.height_px),
        center_mode=str(args.center_mode),
        center_window_px=int(args.center_window_px),
        body_erode_px=int(args.body_erode_px),
        eye_dilate_px=int(args.eye_dilate_px),
    )

    output_prefix = Path(args.output_prefix)
    ensure_output_dir(output_prefix.parent)
    npz_path = output_prefix.with_suffix(".mask_relative_roi.npz")
    roi_json_path = output_prefix.with_suffix(".mask_relative_roi.roi.json")
    diagnostic_path = output_prefix.with_suffix(".mask_relative_roi.png")
    summary_path = output_prefix.with_suffix(".mask_relative_roi.json")

    candidate_mask = np.asarray(derived["candidate_mask"], dtype=bool)
    yy, xx = np.nonzero(candidate_mask)
    np.savez_compressed(
        npz_path,
        roi_mask=candidate_mask.astype(np.uint8),
        roi_x=xx.astype(np.int32),
        roi_y=yy.astype(np.int32),
        candidate_mask=candidate_mask.astype(np.uint8),
        body_mask=np.asarray(derived["body_mask"], dtype=np.uint8),
        body_interior=np.asarray(derived["body_interior"], dtype=np.uint8),
        eye_mask=np.asarray(derived["eye_mask"], dtype=np.uint8),
        eye_exclusion=np.asarray(derived["eye_exclusion"], dtype=np.uint8),
        diagnostic_rect_mask=np.asarray(derived["diagnostic_rect_mask"], dtype=np.uint8),
        body_fraction=body_fraction.astype(np.float32),
        eye_fraction=eye_fraction.astype(np.float32),
        bbox_xyxy=np.asarray(derived["bbox_xyxy"], dtype=np.int32),
        roi_rect_xywh=np.asarray(derived["roi_rect_xywh"], dtype=np.float32),
    )
    roi_json = {
        "roi_rect_stable_xywh": [float(value) for value in np.asarray(derived["roi_rect_xywh"]).tolist()],
        "mask_npz": str(npz_path),
        "coordinate_space": "canonical_stabilized",
        "candidate_mask_is_authoritative": True,
        "center_mode": str(derived["center_mode"]),
        "center_source": str(derived["center_source"]),
    }
    with roi_json_path.open("w") as handle:
        json.dump(roi_json, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _draw_diagnostic(
        path=diagnostic_path,
        body_fraction=body_fraction,
        eye_fraction=eye_fraction,
        body_mask=np.asarray(derived["body_mask"], dtype=bool),
        eye_mask=np.asarray(derived["eye_mask"], dtype=bool),
        candidate_mask=candidate_mask,
        diagnostic_rect_mask=np.asarray(derived["diagnostic_rect_mask"], dtype=bool),
        bbox_xyxy=np.asarray(derived["bbox_xyxy"], dtype=np.int32),
        eye_bottom_y=float(derived["eye_bottom_y"]),
    )

    summary = {
        "config": str(args.config),
        "crop_video": str(context["crop_video"]),
        "zarr_path": str(context["zarr_path"]),
        "keypoint_group": str(context["keypoint_group"]),
        "mask_parent": mask_parent,
        "mask_run": mask_run,
        "canonical_target_forward": str(context["target_forward"]),
        "stable_shape_hw": [int(stable_shape[0]), int(stable_shape[1])],
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "stride": int(args.stride),
        "selected_frames": int(len(context["selected_all"])),
        "projected_frames": int(len(context["selected"])),
        "mask_projection_stride": int(args.mask_projection_stride),
        "thresholds": {
            "body_occupancy": float(args.body_occupancy_threshold),
            "eye_occupancy": float(args.eye_occupancy_threshold),
            "eye_bottom_quantile": float(args.eye_bottom_quantile),
        },
        "geometry": {
            "top_offset_px": float(args.top_offset_px),
            "width_px": int(args.width_px),
            "height_px": int(args.height_px),
            "center_mode": str(derived["center_mode"]),
            "center_window_px": int(args.center_window_px),
            "body_erode_px_requested": int(args.body_erode_px),
            "body_erode_px_used": int(derived["body_erode_px_used"]),
            "eye_dilate_px": int(args.eye_dilate_px),
            "eye_bottom_y": float(derived["eye_bottom_y"]),
            "roi_top_y": int(derived["roi_top_y"]),
            "center_x": float(derived["center_x"]),
            "center_source": str(derived["center_source"]),
            "eye_center": derived["eye_center"],
            "bbox_xyxy": [int(value) for value in np.asarray(derived["bbox_xyxy"]).tolist()],
            "roi_rect_xywh": [float(value) for value in np.asarray(derived["roi_rect_xywh"]).tolist()],
        },
        "pixel_counts": {
            "body_mask": int(np.count_nonzero(derived["body_mask"])),
            "body_interior": int(np.count_nonzero(derived["body_interior"])),
            "eye_mask": int(np.count_nonzero(derived["eye_mask"])),
            "eye_exclusion": int(np.count_nonzero(derived["eye_exclusion"])),
            "diagnostic_rect": int(np.count_nonzero(derived["diagnostic_rect_mask"])),
            "candidate_mask": int(np.count_nonzero(candidate_mask)),
        },
        "components": {
            str(args.body_component): {"role": "include_body", **body_summary},
            **{name: {"role": "eye_anchor_and_exclusion", **value} for name, value in eye_summaries.items()},
        },
        "outputs": {
            "mask_npz": str(npz_path),
            "roi_json": str(roi_json_path),
            "diagnostic_png": str(diagnostic_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_path}")
    print(f"mask_npz: {npz_path}")
    print(f"roi_json: {roi_json_path}")
    print(f"diagnostic_png: {diagnostic_path}")
    print(f"candidate_pixels: {summary['pixel_counts']['candidate_mask']}")
    print(f"roi_rect_xywh: {summary['geometry']['roi_rect_xywh']}")


if __name__ == "__main__":
    main()
