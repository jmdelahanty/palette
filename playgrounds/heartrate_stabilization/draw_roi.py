from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

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
    parse_roi_rect,
    read_crop_meta,
    roi_rect_corners,
    selected_crop_rows,
)
from make_subset_clip import _apply_circular_mask, _draw_polygon
from render_stabilization_probe import _draw_keypoints, _label_panel


def _load_reference_frame(
    config: dict[str, Any],
    *,
    frame_start: int | None,
    draw_keypoints: bool,
) -> dict[str, Any]:
    import cv2

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
    stable_mask_radius = float(
        cfg_value(config, "alignment", "stable_mask_radius_px", min(stable_width, stable_height) / 2.0)
    )
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))

    video = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=frame_start,
        frame_count=1,
        stride=1,
    )
    if not selected:
        raise ValueError(f"No crop metadata row found at or after frame_start={frame_start}.")
    crop_video_index, crop_row = selected[0]
    frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)

    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    keypoint_row = keypoints.frame_to_row.get(frame_id)
    if keypoint_row is None:
        raise ValueError(f"No keypoint row found for {frame_id_column}={frame_id}.")
    if not bool(keypoints.valid[keypoint_row]):
        raise ValueError(f"Keypoint row is invalid for {frame_id_column}={frame_id}.")

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {crop_video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Could not read crop video frame {crop_video_index}.")

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
    if not transform.valid:
        raise ValueError(f"Body transform is invalid for {frame_id_column}={frame_id}: {transform.reason}")

    stable = cv2.warpAffine(
        frame,
        transform.crop_to_stable.astype(np.float32),
        (stable_width, stable_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if stable_circular_mask:
        stable = _apply_circular_mask(
            stable,
            center_x=stable_center_x,
            center_y=stable_center_y,
            radius_px=stable_mask_radius,
        )
    if draw_keypoints:
        kp_stable = (np.column_stack([kp_crop, np.ones(kp_crop.shape[0])]) @ transform.crop_to_stable.T).reshape(
            kp_crop.shape
        )
        stable = _draw_keypoints(stable, kp_stable, color=(0, 255, 0))
    stable = _label_panel(stable, f"draw ROI frame={frame_id}")
    return {
        "image": stable,
        "crop_video": crop_video,
        "crop_meta_csv": crop_meta_csv,
        "zarr_path": zarr_path,
        "keypoint_group": keypoint_group,
        "frame_id_column": frame_id_column,
        "frame_id": int(frame_id),
        "crop_video_frame_index": int(crop_video_index),
        "stable_width": int(stable_width),
        "stable_height": int(stable_height),
        "stable_center_x": float(stable_center_x),
        "stable_center_y": float(stable_center_y),
        "stable_circular_mask": bool(stable_circular_mask),
        "stable_mask_radius_px": float(stable_mask_radius),
        "alignment_origin": str(origin),
        "body_transform_reason": transform.reason,
        "forward_angle_deg": float(transform.forward_angle_deg),
        "origin_crop_xy": transform.origin_crop_xy.astype(float).tolist(),
    }


def _scaled_select_roi(image: np.ndarray, *, display_scale: float, window_name: str) -> tuple[float, float, float, float]:
    import cv2

    scale = float(display_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    display = image
    if scale != 1.0:
        display = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
    print("Draw ROI, then press Enter or Space to accept. Press Esc/C to cancel.")
    x, y, w, h = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if w <= 0 or h <= 0:
        raise ValueError("ROI selection was cancelled or empty.")
    return (float(x) / scale, float(y) / scale, float(w) / scale, float(h) / scale)


def _write_roi_json(
    path: Path,
    *,
    roi_rect: tuple[float, float, float, float],
    reference: dict[str, Any],
    config_path: Path,
) -> None:
    payload = {
        "schema_id": "palette.playground.heartrate_stabilized_roi.v1",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "coordinate_space": "stabilized_fish_body_frame_pixels",
        "source_config": str(config_path),
        "source_crop_video": str(reference["crop_video"]),
        "source_crop_meta_csv": str(reference["crop_meta_csv"]),
        "source_zarr_path": str(reference["zarr_path"]),
        "source_keypoint_group": str(reference["keypoint_group"]),
        "frame_id_column": str(reference["frame_id_column"]),
        "frame_id": int(reference["frame_id"]),
        "crop_video_frame_index": int(reference["crop_video_frame_index"]),
        "stable_frame_shape_hw": [int(reference["stable_height"]), int(reference["stable_width"])],
        "stable_center_xy": [float(reference["stable_center_x"]), float(reference["stable_center_y"])],
        "stable_circular_mask": bool(reference["stable_circular_mask"]),
        "stable_mask_radius_px": float(reference["stable_mask_radius_px"]),
        "alignment_origin": str(reference["alignment_origin"]),
        "body_transform_reason": str(reference["body_transform_reason"]),
        "forward_angle_deg": float(reference["forward_angle_deg"]),
        "origin_crop_xy": reference["origin_crop_xy"],
        "measurement_note": "Map this stabilized ROI back to each crop-video frame and sample source crop pixels.",
    }
    ensure_output_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Draw a stabilized heart ROI rectangle and save it as JSON.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument("--roi", type=str, default=None, help="Use x,y,width,height instead of opening the selector.")
    parser.add_argument("--display-scale", type=float, default=3.0)
    parser.add_argument("--no-keypoints", action="store_true", help="Hide keypoint markers while drawing.")
    parser.add_argument(
        "--save-frame-only",
        action="store_true",
        help="Write the stabilized reference PNG and exit without opening an interactive window.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dir(Path(str(cfg_value(config, "probe", "output_dir", "outputs"))))
    output = args.output or output_dir / "heart_roi.json"
    preview_output = args.preview_output or output.with_name(f"{output.stem}_preview.png")
    reference = _load_reference_frame(config, frame_start=args.frame_start, draw_keypoints=not args.no_keypoints)
    image = reference["image"]

    ensure_output_dir(preview_output.parent)
    if args.roi is None and args.save_frame_only:
        cv2.imwrite(str(preview_output), image)
        print(f"wrote reference frame: {preview_output}")
        return

    if args.roi is not None:
        roi_rect = parse_roi_rect(args.roi)
    else:
        roi_rect = _scaled_select_roi(
            image,
            display_scale=float(args.display_scale),
            window_name="heartrate stabilized ROI",
        )
    roi_preview = _draw_polygon(image, roi_rect_corners(roi_rect), color=(0, 255, 255))
    cv2.imwrite(str(preview_output), roi_preview)
    _write_roi_json(output, roi_rect=roi_rect, reference=reference, config_path=args.config)
    print(f"wrote ROI JSON: {output}")
    print(f"wrote preview PNG: {preview_output}")
    print(f"roi_rect_stable_xywh={roi_rect[0]:.2f},{roi_rect[1]:.2f},{roi_rect[2]:.2f},{roi_rect[3]:.2f}")


if __name__ == "__main__":
    main()
