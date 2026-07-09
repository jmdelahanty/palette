from __future__ import annotations

import argparse
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
    project_subject_mask_to_crop_frame,
    read_crop_meta,
    resolve_roi_rect,
    roi_rect_corners,
    selected_crop_rows,
    SubjectMaskUnavailable,
    transform_points,
)
from render_stabilization_probe import _draw_keypoints, _label_panel


def _draw_polygon(image: np.ndarray, points_xy: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] >= 3 and np.isfinite(points).all():
        cv2.polylines(
            out,
            [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return out


def _blank_panel(width: int, height: int, label: str) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    return _label_panel(panel, label)


def _tint_mask(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.25) -> np.ndarray:
    import cv2

    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != out.shape[:2] or not np.any(mask_bool):
        return out
    overlay = np.zeros_like(out)
    overlay[mask_bool] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(out, 1.0 - float(alpha), overlay, float(alpha), 0.0)


def _apply_circular_mask(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_px: float,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    import cv2

    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 0.5 * float(min(out.shape[0], out.shape[1]))
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center_x)), int(round(center_y))),
        int(round(radius)),
        255,
        -1,
        lineType=cv2.LINE_AA,
    )
    bg = np.zeros_like(out)
    bg[:, :] = np.asarray(background, dtype=np.uint8)
    return np.where(mask[:, :, None] > 0, out, bg)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Create a short crop-vs-stabilized diagnostic clip.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--roi", type=str, default=None, help="Optional stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--mask", action="store_true", help="Overlay a projected subject-mask component.")
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--mask-component", type=str, default=None, help="Semantic mask component to overlay.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path. Defaults to [probe].output_dir/stabilization_subset_clip.mp4.",
    )
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
    stable_circular_mask = bool(cfg_value(config, "alignment", "stable_circular_mask", True))
    stable_mask_radius = float(cfg_value(config, "alignment", "stable_mask_radius_px", min(stable_width, stable_height) / 2.0))
    mask_enabled = bool(args.mask or cfg_value(config, "mask", "enabled", False))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))
    mask_component = str(args.mask_component or cfg_value(config, "mask", "component", "subject_body"))
    frame_start = args.frame_start
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))
    frame_count = int(args.frame_count or min(int(cfg_value(config, "probe", "frame_count", 120)), 300))
    stride = max(1, int(args.stride))
    output = args.output or Path(str(cfg_value(config, "probe", "output_dir", "outputs"))) / "stabilization_subset_clip.mp4"
    ensure_output_dir(output.parent)

    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_stable = roi_rect_corners(roi_rect)

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
        stride=stride,
    )
    if not selected:
        raise ValueError("No crop rows selected for subset clip.")
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

    output_fps = max(1.0, float(video.fps) / float(stride)) if np.isfinite(video.fps) and video.fps > 0 else 30.0
    panel_width = int(video.width) + stable_width
    panel_height = max(int(video.height), stable_height)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (panel_width, panel_height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output writer: {output}")

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        writer.release()
        raise ValueError(f"Could not open video: {crop_video}")

    rendered = 0
    valid_transforms = 0
    try:
        for crop_video_index, crop_row in selected:
            frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
            keypoint_row = keypoints.frame_to_row.get(frame_id)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
            ok, frame = capture.read()
            if not ok:
                continue

            if keypoint_row is None or not bool(keypoints.valid[keypoint_row]):
                crop_panel = _label_panel(frame, f"crop frame={frame_id} no valid keypoints")
                stable_panel = _blank_panel(stable_width, stable_height, "stable unavailable")
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
                crop_panel = _draw_keypoints(frame, kp_crop, color=(0, 255, 0))
                if transform.valid:
                    valid_transforms += 1
                    projected_mask = None
                    if mask_data is not None:
                        projected = project_subject_mask_to_crop_frame(
                            mask_data,
                            frame_id=frame_id,
                            crop_row=crop_row,
                            video_width=video.width,
                            video_height=video.height,
                        )
                        if projected.valid and projected.mask is not None:
                            projected_mask = projected.mask
                            crop_panel = _tint_mask(crop_panel, projected_mask, color=(255, 0, 255), alpha=0.25)
                    roi_crop = transform_points(transform.stable_to_crop, roi_stable)
                    crop_panel = _draw_polygon(crop_panel, roi_crop, color=(0, 255, 255))
                    crop_panel = _label_panel(crop_panel, f"crop frame={frame_id}")
                    stable = cv2.warpAffine(
                        frame,
                        transform.crop_to_stable.astype(np.float32),
                        (stable_width, stable_height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    if projected_mask is not None:
                        stable_mask = cv2.warpAffine(
                            projected_mask.astype(np.uint8) * 255,
                            transform.crop_to_stable.astype(np.float32),
                            (stable_width, stable_height),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        stable = _tint_mask(stable, stable_mask > 0, color=(255, 0, 255), alpha=0.25)
                    if stable_circular_mask:
                        stable = _apply_circular_mask(
                            stable,
                            center_x=stable_center_x,
                            center_y=stable_center_y,
                            radius_px=stable_mask_radius,
                        )
                    kp_stable = transform_points(transform.crop_to_stable, kp_crop)
                    stable_panel = _draw_keypoints(stable, kp_stable, color=(0, 255, 0))
                    stable_panel = _draw_polygon(stable_panel, roi_stable, color=(0, 255, 255))
                    stable_panel = _label_panel(
                        stable_panel,
                        f"stable fwd=up angle={transform.forward_angle_deg:.1f}",
                    )
                else:
                    crop_panel = _label_panel(crop_panel, f"crop frame={frame_id}")
                    stable_panel = _blank_panel(stable_width, stable_height, f"stable invalid: {transform.reason}")

            if crop_panel.shape[0] != panel_height:
                crop_canvas = np.zeros((panel_height, crop_panel.shape[1], 3), dtype=np.uint8)
                crop_canvas[: crop_panel.shape[0], : crop_panel.shape[1], :] = crop_panel[:, :, :3]
                crop_panel = crop_canvas
            if stable_panel.shape[0] != panel_height:
                stable_canvas = np.zeros((panel_height, stable_panel.shape[1], 3), dtype=np.uint8)
                stable_canvas[: stable_panel.shape[0], : stable_panel.shape[1], :] = stable_panel[:, :, :3]
                stable_panel = stable_canvas
            writer.write(np.hstack([crop_panel[:, :, :3], stable_panel[:, :, :3]]))
            rendered += 1
    finally:
        capture.release()
        writer.release()

    if rendered == 0:
        raise ValueError("No clip frames were rendered.")
    print(f"wrote: {output}")
    print(f"frames_rendered: {rendered}")
    print(f"valid_transforms: {valid_transforms}")
    print(f"fps: {output_fps:g}")
    if mask_enabled and mask_data is None:
        print(f"mask: unavailable ({mask_setup_reason})")
    elif mask_enabled and mask_data is not None:
        print(
            "mask: "
            f"{mask_data.source_path} component={mask_data.component_name} "
            f"surface={mask_data.storage_surface}"
        )


if __name__ == "__main__":
    main()
