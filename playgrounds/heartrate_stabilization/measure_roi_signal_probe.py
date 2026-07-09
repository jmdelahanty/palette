from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from _common import (
    cfg_path,
    cfg_value,
    compute_body_transform,
    crop_row_frame_id,
    ensure_output_dir,
    get_video_info,
    keypoints_to_crop_pixels,
    load_config,
    load_keypoint_data,
    load_subject_mask_data,
    mask_mean_intensity,
    polygon_mask,
    project_subject_mask_to_crop_frame,
    read_crop_meta,
    resolve_roi_rect,
    roi_rect_corners,
    selected_crop_rows,
    SubjectMaskUnavailable,
    transform_points,
)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Measure a stabilized heart ROI from original crop-video pixels.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Stabilized-view rectangle as x,y,width,height. Defaults to [roi].rect from config.",
    )
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--debug-frames", type=int, default=0, help="Write first N debug PNGs with transformed ROI.")
    parser.add_argument("--mask", action="store_true", help="Intersect the ROI with a projected subject-mask component.")
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--mask-component", type=str, default=None, help="Semantic mask component to use.")
    parser.add_argument("--require-mask", action="store_true", help="Mark frames invalid instead of falling back to ROI-only.")
    args = parser.parse_args()

    config = load_config(args.config)
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
    origin = str(cfg_value(config, "alignment", "origin", "eye_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_enabled = bool(args.mask or cfg_value(config, "mask", "enabled", False))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))
    mask_component = str(args.mask_component or cfg_value(config, "mask", "component", "subject_body"))
    require_mask = bool(args.require_mask or cfg_value(config, "mask", "require_mask", False))
    frame_start = args.frame_start
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))
    frame_count = int(args.frame_count or cfg_value(config, "probe", "frame_count", 120))
    output = args.output or Path(str(cfg_value(config, "probe", "output_dir", "outputs"))) / "roi_signal.csv"
    ensure_output_dir(output.parent)
    roi = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_corners_stable = roi_rect_corners(roi)

    video = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=frame_start,
        frame_count=frame_count,
        stride=max(1, int(args.stride)),
    )
    mask_data = None
    mask_setup_reason = "mask_disabled"
    if mask_enabled:
        candidate_parents = (
            ("refined_subject_masks_runs", "subject_mask_runs")
            if mask_parent.strip().lower() == "auto"
            else tuple(part.strip() for part in mask_parent.split(",") if part.strip())
        )
        mask_errors: list[str] = []
        for candidate_parent in candidate_parents:
            try:
                mask_data = load_subject_mask_data(
                    zarr_path,
                    parent=candidate_parent,
                    run_name=mask_run,
                    component_name=mask_component,
                )
                mask_setup_reason = "ok"
                break
            except SubjectMaskUnavailable as exc:
                mask_errors.append(f"{candidate_parent}:{exc}")
        if mask_data is None and mask_errors:
            mask_setup_reason = ";".join(mask_errors)
        if mask_data is None and require_mask:
            raise SubjectMaskUnavailable(mask_setup_reason)

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {crop_video}")

    fieldnames = [
        "crop_video_frame_index",
        "frame_id_column",
        "frame_id",
        "valid",
        "reason",
        "mean_intensity",
        "roi_pixel_count",
        "roi_unmasked_mean_intensity",
        "roi_unmasked_pixel_count",
        "mask_enabled",
        "mask_available",
        "mask_source_path",
        "mask_component",
        "mask_storage_surface",
        "mask_valid",
        "mask_reason",
        "mask_row",
        "mask_source_crop_row_id",
        "mask_projected_pixel_count",
        "origin_crop_x",
        "origin_crop_y",
        "forward_angle_deg",
        "roi_crop_x0",
        "roi_crop_y0",
        "roi_crop_x1",
        "roi_crop_y1",
        "roi_crop_x2",
        "roi_crop_y2",
        "roi_crop_x3",
        "roi_crop_y3",
    ]
    debug_written = 0
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        try:
            for crop_video_index, crop_row in selected:
                frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
                keypoint_row = keypoints.frame_to_row.get(frame_id)
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
                ok, frame = capture.read()
                if not ok:
                    writer.writerow(
                        {
                            "crop_video_frame_index": crop_video_index,
                            "frame_id_column": frame_id_column,
                            "frame_id": frame_id,
                            "valid": 0,
                            "reason": "video_read_failed",
                            "mask_enabled": int(mask_enabled),
                            "mask_available": int(mask_data is not None),
                            "mask_source_path": mask_data.source_path if mask_data is not None else "",
                            "mask_component": mask_component,
                            "mask_storage_surface": mask_data.storage_surface if mask_data is not None else "",
                            "mask_valid": 0,
                            "mask_reason": mask_setup_reason,
                            "mask_row": -1,
                            "mask_source_crop_row_id": -1,
                            "mask_projected_pixel_count": 0,
                        }
                    )
                    continue

                roi_unmasked_mean = math.nan
                roi_unmasked_pixel_count = 0
                mask_valid = False
                mask_reason = "mask_disabled" if not mask_enabled else mask_setup_reason
                mask_row = -1
                mask_source_crop_row_id = -1
                mask_projected_pixel_count = 0
                debug_projected_mask = None
                debug_sample_mask = None
                if keypoint_row is None:
                    valid = False
                    reason = "missing_keypoint_frame"
                    mean_intensity = math.nan
                    pixel_count = 0
                    polygon_crop = np.full((4, 2), np.nan, dtype=np.float64)
                    origin_xy = np.full(2, np.nan, dtype=np.float64)
                    angle = math.nan
                elif not bool(keypoints.valid[keypoint_row]):
                    valid = False
                    reason = "invalid_keypoints"
                    mean_intensity = math.nan
                    pixel_count = 0
                    polygon_crop = np.full((4, 2), np.nan, dtype=np.float64)
                    origin_xy = np.full(2, np.nan, dtype=np.float64)
                    angle = math.nan
                else:
                    kp_crop = keypoints_to_crop_pixels(
                        keypoints.keypoints_img[keypoint_row],
                        crop_row,
                        video_width=video.width,
                        video_height=video.height,
                    )
                    transform = compute_body_transform(
                        kp_crop,
                        stable_width=stable_width,
                        stable_height=stable_height,
                        stable_center_x=stable_center_x,
                        stable_center_y=stable_center_y,
                        origin=origin,
                        target_forward=target_forward,
                        scale=scale,
                        min_forward_length_px=min_forward,
                        min_eye_span_px=min_eye_span,
                    )
                    valid = bool(transform.valid)
                    reason = transform.reason
                    origin_xy = transform.origin_crop_xy
                    angle = transform.forward_angle_deg
                    if valid:
                        polygon_crop = transform_points(transform.stable_to_crop, roi_corners_stable)
                        roi_mask = polygon_mask(frame.shape[:2], polygon_crop)
                        roi_unmasked_mean, roi_unmasked_pixel_count = mask_mean_intensity(frame, roi_mask)
                        mean_intensity = roi_unmasked_mean
                        pixel_count = roi_unmasked_pixel_count
                        valid = bool(pixel_count > 0 and np.isfinite(mean_intensity))
                        if not valid:
                            reason = "roi_outside_crop"
                        elif mask_enabled:
                            if mask_data is None:
                                mask_reason = mask_setup_reason
                                if require_mask:
                                    valid = False
                                    reason = mask_reason
                                    mean_intensity = math.nan
                                    pixel_count = 0
                            else:
                                projected = project_subject_mask_to_crop_frame(
                                    mask_data,
                                    frame_id=frame_id,
                                    crop_row=crop_row,
                                    video_width=video.width,
                                    video_height=video.height,
                                )
                                mask_valid = bool(projected.valid)
                                mask_reason = projected.reason
                                mask_row = int(projected.mask_row)
                                mask_source_crop_row_id = int(projected.source_crop_row_id)
                                mask_projected_pixel_count = int(projected.mask_pixel_count)
                                debug_projected_mask = projected.mask
                                if projected.valid and projected.mask is not None:
                                    sample_mask = roi_mask & projected.mask
                                    debug_sample_mask = sample_mask
                                    mean_intensity, pixel_count = mask_mean_intensity(frame, sample_mask)
                                    valid = bool(pixel_count > 0 and np.isfinite(mean_intensity))
                                    if not valid:
                                        reason = "masked_roi_empty"
                                elif require_mask:
                                    valid = False
                                    reason = projected.reason
                                    mean_intensity = math.nan
                                    pixel_count = 0
                    else:
                        polygon_crop = np.full((4, 2), np.nan, dtype=np.float64)
                        mean_intensity = math.nan
                        pixel_count = 0

                flat_polygon = polygon_crop.reshape(-1)
                writer.writerow(
                    {
                        "crop_video_frame_index": crop_video_index,
                        "frame_id_column": frame_id_column,
                        "frame_id": frame_id,
                        "valid": int(valid),
                        "reason": reason,
                        "mean_intensity": mean_intensity,
                        "roi_pixel_count": pixel_count,
                        "roi_unmasked_mean_intensity": roi_unmasked_mean,
                        "roi_unmasked_pixel_count": roi_unmasked_pixel_count,
                        "mask_enabled": int(mask_enabled),
                        "mask_available": int(mask_data is not None),
                        "mask_source_path": mask_data.source_path if mask_data is not None else "",
                        "mask_component": mask_data.component_name if mask_data is not None else mask_component,
                        "mask_storage_surface": mask_data.storage_surface if mask_data is not None else "",
                        "mask_valid": int(mask_valid),
                        "mask_reason": mask_reason,
                        "mask_row": mask_row,
                        "mask_source_crop_row_id": mask_source_crop_row_id,
                        "mask_projected_pixel_count": mask_projected_pixel_count,
                        "origin_crop_x": origin_xy[0],
                        "origin_crop_y": origin_xy[1],
                        "forward_angle_deg": angle,
                        "roi_crop_x0": flat_polygon[0],
                        "roi_crop_y0": flat_polygon[1],
                        "roi_crop_x1": flat_polygon[2],
                        "roi_crop_y1": flat_polygon[3],
                        "roi_crop_x2": flat_polygon[4],
                        "roi_crop_y2": flat_polygon[5],
                        "roi_crop_x3": flat_polygon[6],
                        "roi_crop_y3": flat_polygon[7],
                    }
                )

                if args.debug_frames > 0 and debug_written < args.debug_frames and np.isfinite(polygon_crop).all():
                    debug = frame.copy()
                    if debug.ndim == 2:
                        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
                    if debug_projected_mask is not None:
                        overlay = np.zeros_like(debug)
                        overlay[np.asarray(debug_projected_mask, dtype=bool)] = (255, 0, 255)
                        debug = cv2.addWeighted(debug, 0.78, overlay, 0.22, 0.0)
                    if debug_sample_mask is not None:
                        overlay = np.zeros_like(debug)
                        overlay[np.asarray(debug_sample_mask, dtype=bool)] = (0, 255, 255)
                        debug = cv2.addWeighted(debug, 0.82, overlay, 0.18, 0.0)
                    cv2.polylines(
                        debug,
                        [np.round(polygon_crop).astype(np.int32).reshape(-1, 1, 2)],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    debug_path = output.parent / f"roi_debug_{crop_video_index:06d}_{frame_id}.png"
                    cv2.imwrite(str(debug_path), debug)
                    debug_written += 1
        finally:
            capture.release()

    print(f"wrote: {output}")
    if mask_enabled and mask_data is None:
        print(f"mask: unavailable ({mask_setup_reason}); used ROI-only measurements unless --require-mask was set")
    elif mask_enabled and mask_data is not None:
        print(
            "mask: "
            f"{mask_data.source_path} component={mask_data.component_name} "
            f"surface={mask_data.storage_surface}"
        )


if __name__ == "__main__":
    main()
