from __future__ import annotations

import argparse
import json
import math
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
    read_crop_meta,
    resolve_roi_rect,
    roi_rect_corners,
)
from map_pixel_band_contributions import _draw_polygon, _load_mask_component, _split_components, _video_shape
from measure_live_mask_stability import _dilate, _load_fixed_masks, _mask_metrics, _stable_project_component
from render_roi_mask_overlay_video import _read_status_csv


def _parse_center_frames(raw: str) -> list[int]:
    frames = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not frames:
        raise ValueError("--center-frames must contain at least one frame index.")
    return sorted(set(frames))


def _selected_frames(*, centers: list[int], context_frames: int, stride: int, max_frame_count: int) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    seen: set[int] = set()
    for center in centers:
        start = max(0, int(center) - int(context_frames))
        stop = min(int(max_frame_count), int(center) + int(context_frames) + 1)
        for frame_index in range(start, stop, max(1, int(stride))):
            if frame_index in seen:
                continue
            seen.add(frame_index)
            selected.append((frame_index, int(center)))
    return sorted(selected)


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = image[:, :, :3].copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return out
    base = out[mask_bool].astype(np.float64)
    target = np.asarray(color, dtype=np.float64)
    out[mask_bool] = np.clip(base * (1.0 - float(alpha)) + target * float(alpha), 0, 255).astype(np.uint8)
    return out


def _draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    import cv2

    out = image.copy()
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    if int(np.count_nonzero(mask_uint8)) == 0:
        return out
    contours, _hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, int(thickness), cv2.LINE_AA)
    return out


def _draw_text(
    image: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    scale: float = 0.42,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _crop_zoom(image: np.ndarray, *, bbox_xyxy: np.ndarray, pad_px: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(value) for value in np.asarray(bbox_xyxy).reshape(-1).tolist()]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(image.shape[1], x1 + int(pad_px))
    y1 = min(image.shape[0], y1 + int(pad_px))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Zoom crop is empty: {(x0, y0, x1, y1)}")
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if xx.size == 0:
        raise ValueError("Cannot derive a bounding box from an empty mask.")
    return np.asarray([int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1], dtype=np.int32)


def _resize_panel(image: np.ndarray, *, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(image[:, :, :3], (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def _read_frame(capture: Any, *, frame_index: int, next_expected: int | None) -> tuple[bool, np.ndarray | None, int]:
    import cv2

    if next_expected is None or int(frame_index) != int(next_expected):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return bool(ok), frame, int(frame_index) + 1


def _overlay(
    frame: np.ndarray,
    *,
    fixed: dict[str, np.ndarray],
    live_body: np.ndarray | None,
    live_eye: np.ndarray | None,
    live_eye_exclusion: np.ndarray | None,
    roi_polygon: np.ndarray,
    metrics: dict[str, float],
    frame_index: int,
    center_frame: int,
    status_valid: bool,
) -> np.ndarray:
    out = frame[:, :, :3].copy()
    roi_mask = np.asarray(fixed["roi_mask"], dtype=bool)
    out = _blend_mask(out, roi_mask, color=(0, 255, 255), alpha=0.28)
    if live_body is not None and live_eye_exclusion is not None:
        not_live_usable = roi_mask & (~np.asarray(live_body, dtype=bool) | np.asarray(live_eye_exclusion, dtype=bool))
        out = _blend_mask(out, not_live_usable, color=(0, 0, 255), alpha=0.60)

    # Fixed consensus contours.
    out = _draw_contours(out, fixed["body_mask"], color=(60, 170, 60), thickness=1)
    out = _draw_contours(out, fixed["eye_mask"], color=(0, 140, 255), thickness=1)
    out = _draw_contours(out, fixed["roi_mask"], color=(0, 255, 255), thickness=1)

    # Live framewise contours.
    if live_body is not None:
        out = _draw_contours(out, live_body, color=(255, 0, 255), thickness=1)
    if live_eye is not None:
        out = _draw_contours(out, live_eye, color=(255, 255, 0), thickness=1)

    out = _draw_polygon(out, roi_polygon, color=(255, 255, 255))
    label = "valid" if status_valid else "invalid"
    _draw_text(out, f"frame {frame_index}  center {center_frame}  {label}", origin=(8, 18), scale=0.43)
    _draw_text(out, "fixed: body green, eyes orange, ROI yellow", origin=(8, out.shape[0] - 28), scale=0.38)
    _draw_text(out, "live: body magenta, eyes cyan; red=ROI not live-usable", origin=(8, out.shape[0] - 10), scale=0.38)
    usable = metrics.get("roi_live_usable_fraction", math.nan)
    body_shift = metrics.get("body_centroid_shift_px", math.nan)
    eye_shift = metrics.get("eye_union_centroid_shift_px", math.nan)
    body_iou = metrics.get("body_iou", math.nan)
    eye_iou = metrics.get("eye_union_iou", math.nan)
    _draw_text(
        out,
        f"usable={usable:.3f}  body shift={body_shift:.2f}px IoU={body_iou:.3f}  eye shift={eye_shift:.2f}px IoU={eye_iou:.3f}",
        origin=(8, 36),
        scale=0.36,
    )
    return out


def _failure_panel(*, shape_hw: tuple[int, int], reason: str, frame_index: int, center_frame: int) -> np.ndarray:
    panel = np.zeros((*shape_hw, 3), dtype=np.uint8)
    _draw_text(panel, f"frame {frame_index} center {center_frame}", origin=(8, 24), scale=0.5)
    _draw_text(panel, f"live mask unavailable: {reason[:80]}", origin=(8, 46), scale=0.42)
    return panel


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render live-vs-fixed mask contours around selected stabilized frames.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to render.")
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--center-frames", type=str, required=True, help="Comma-separated crop/stabilized frame indices.")
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--zoom-pad-px", type=int, default=12)
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/live_vs_fixed_mask_overlay.mp4"),
    )
    parser.add_argument("--summary-json", type=Path, default=None)
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
    origin = str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))

    shape_hw = _video_shape(args.video)
    if tuple(shape_hw) != (stable_height, stable_width):
        raise ValueError(f"Video shape {shape_hw} does not match configured stable shape {(stable_height, stable_width)}")
    fixed = _load_fixed_masks(args.mask_npz, shape_hw=shape_hw)
    roi_mask = np.asarray(fixed["roi_mask"], dtype=bool)
    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels <= 0:
        raise ValueError(f"{args.mask_npz} roi_mask is empty.")
    roi_bbox_xyxy = _bbox_from_mask(roi_mask)
    roi_rect = resolve_roi_rect(config, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    centers = _parse_center_frames(args.center_frames)
    status = _read_status_csv(args.status_csv)

    stable_video_info = get_video_info(args.video)
    crop_video_info = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    selected = _selected_frames(
        centers=centers,
        context_frames=max(0, int(args.context_frames)),
        stride=max(1, int(args.stride)),
        max_frame_count=min(int(stable_video_info.frame_count), len(crop_rows)),
    )
    if not selected:
        raise ValueError("No frames selected.")
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    body_data = _load_mask_component(
        zarr_path,
        parent=mask_parent,
        run_name=mask_run,
        component_name=str(args.body_component),
    )
    eye_data = {
        component: _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=component,
        )
        for component in _split_components(args.eye_components)
    }
    if not eye_data:
        raise ValueError("--eye-components must name at least one component.")

    output = Path(args.output)
    ensure_output_dir(output.parent)
    panel_size = int(args.panel_size)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        (panel_size * 2, panel_size),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output writer: {output}")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        writer.release()
        raise ValueError(f"Could not open video: {args.video}")

    rows: list[dict[str, Any]] = []
    preview: np.ndarray | None = None
    rendered = 0
    read_failures = 0
    live_failures = 0
    next_expected: int | None = None
    try:
        for frame_index, center_frame in selected:
            ok, frame, next_expected = _read_frame(capture, frame_index=frame_index, next_expected=next_expected)
            if not ok or frame is None:
                read_failures += 1
                frame = np.zeros((*shape_hw, 3), dtype=np.uint8)
            crop_row = crop_rows[int(frame_index)]
            frame_id = crop_row_frame_id(int(frame_index), crop_row, frame_id_column)
            status_valid = status.get(int(frame_index), True)
            metrics: dict[str, float] = {}
            reason = "ok"
            live_body = None
            live_eye = None
            live_eye_exclusion = None

            keypoint_row = keypoints.frame_to_row.get(frame_id)
            if keypoint_row is None:
                reason = "missing_keypoint_frame"
            elif not bool(keypoints.valid[keypoint_row]):
                reason = "invalid_keypoints"
            else:
                kp_crop = keypoints_to_crop_pixels(
                    keypoints.keypoints_img[keypoint_row],
                    crop_row,
                    video_width=crop_video_info.width,
                    video_height=crop_video_info.height,
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
                    reason = transform.reason
                else:
                    live_body, body_reason = _stable_project_component(
                        mask_data=body_data,
                        frame_id=frame_id,
                        crop_row=crop_row,
                        transform=transform,
                        video=crop_video_info,
                        stable_width=stable_width,
                        stable_height=stable_height,
                    )
                    if live_body is None:
                        reason = f"body:{body_reason}"
                    else:
                        live_eye = np.zeros((stable_height, stable_width), dtype=bool)
                        eye_reasons: dict[str, str] = {}
                        for component, data in eye_data.items():
                            component_mask, component_reason = _stable_project_component(
                                mask_data=data,
                                frame_id=frame_id,
                                crop_row=crop_row,
                                transform=transform,
                                video=crop_video_info,
                                stable_width=stable_width,
                                stable_height=stable_height,
                            )
                            eye_reasons[component] = component_reason
                            if component_mask is not None:
                                live_eye |= component_mask
                        if not np.any(live_eye):
                            reason = "empty_eye_union:" + ",".join(f"{name}={value}" for name, value in eye_reasons.items())
                            live_eye = None
                        else:
                            live_eye_exclusion = _dilate(live_eye, int(args.eye_dilate_px))
                            live_usable = live_body & ~live_eye_exclusion
                            metrics = {
                                **_mask_metrics("body", live_body, fixed["body_mask"]),
                                **_mask_metrics("eye_union", live_eye, fixed["eye_mask"]),
                                **_mask_metrics("eye_exclusion", live_eye_exclusion, fixed["eye_exclusion"]),
                                "roi_body_coverage_fraction": float(np.count_nonzero(roi_mask & live_body) / roi_pixels),
                                "roi_eye_overlap_fraction": float(np.count_nonzero(roi_mask & live_eye) / roi_pixels),
                                "roi_eye_exclusion_overlap_fraction": float(np.count_nonzero(roi_mask & live_eye_exclusion) / roi_pixels),
                                "roi_live_usable_fraction": float(np.count_nonzero(roi_mask & live_usable) / roi_pixels),
                            }

            if reason != "ok":
                live_failures += 1
                overlay = _failure_panel(shape_hw=shape_hw, reason=reason, frame_index=frame_index, center_frame=center_frame)
            else:
                overlay = _overlay(
                    frame,
                    fixed=fixed,
                    live_body=live_body,
                    live_eye=live_eye,
                    live_eye_exclusion=live_eye_exclusion,
                    roi_polygon=roi_polygon,
                    metrics=metrics,
                    frame_index=frame_index,
                    center_frame=center_frame,
                    status_valid=bool(status_valid),
                )
            zoom, (x0, y0, _x1, _y1) = _crop_zoom(
                overlay,
                bbox_xyxy=roi_bbox_xyxy,
                pad_px=int(args.zoom_pad_px),
            )
            zoom_polygon = roi_polygon - np.asarray([x0, y0], dtype=np.float64)
            zoom = _draw_polygon(zoom, zoom_polygon, color=(255, 255, 255))
            _draw_text(zoom, "ROI zoom", origin=(8, 18), scale=0.45, color=(255, 255, 255))
            combined = np.concatenate(
                [
                    _resize_panel(overlay, size=panel_size),
                    _resize_panel(zoom, size=panel_size),
                ],
                axis=1,
            )
            if preview is None:
                preview = combined.copy()
            writer.write(combined)
            rendered += 1
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "center_frame": int(center_frame),
                    "frame_id": int(frame_id),
                    "status_valid": int(bool(status_valid)),
                    "reason": reason,
                    **metrics,
                }
            )
    finally:
        capture.release()
        writer.release()

    preview_path = output.with_suffix(".preview.png")
    if preview is not None:
        cv2.imwrite(str(preview_path), preview)
    csv_path = output.with_suffix(".csv")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        import csv

        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    valid_rows = [row for row in rows if row.get("reason") == "ok"]
    def finite_values(key: str) -> list[float]:
        return [float(row[key]) for row in valid_rows if key in row and np.isfinite(float(row[key]))]

    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json),
        "mask_npz": str(args.mask_npz),
        "center_frames": centers,
        "context_frames": int(args.context_frames),
        "stride": int(args.stride),
        "selected_frames": int(len(selected)),
        "rendered_frames": int(rendered),
        "read_failures": int(read_failures),
        "live_failures": int(live_failures),
        "output_video": str(output),
        "preview_png": str(preview_path) if preview is not None else None,
        "csv": str(csv_path),
        "roi_mask_pixels": roi_pixels,
        "roi_bbox_xyxy": [int(value) for value in roi_bbox_xyxy.tolist()],
        "metrics": {
            key: {
                "min": float(np.min(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for key, values in {
                "body_iou": finite_values("body_iou"),
                "body_centroid_shift_px": finite_values("body_centroid_shift_px"),
                "eye_union_iou": finite_values("eye_union_iou"),
                "eye_union_centroid_shift_px": finite_values("eye_union_centroid_shift_px"),
                "roi_live_usable_fraction": finite_values("roi_live_usable_fraction"),
            }.items()
        },
    }
    summary_path = args.summary_json or output.with_suffix(".summary.json")
    with Path(summary_path).open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"output_video: {output}")
    print(f"preview_png: {preview_path}")
    print(f"summary_json: {summary_path}")
    print(f"csv: {csv_path}")
    print(f"rendered_frames: {rendered}")
    print(f"live_failures: {live_failures}")


if __name__ == "__main__":
    main()
