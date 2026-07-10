from __future__ import annotations

import argparse
import csv
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
    selected_crop_rows,
)
from map_pixel_band_contributions import (
    _draw_polygon,
    _load_mask_component,
    _split_components,
    _stable_component_mask_counts,
    _video_shape,
)
from measure_live_mask_stability import _dilate, _load_fixed_masks, _mask_metrics, _stable_project_component
from render_live_vs_fixed_mask_overlay_video import (
    _bbox_from_mask,
    _blend_mask,
    _crop_zoom,
    _draw_contours,
    _draw_text,
    _parse_center_frames,
    _read_frame,
    _resize_panel,
    _selected_frames,
)
from render_roi_mask_overlay_video import _read_status_csv


def _edge_anchor(mask: np.ndarray, *, quantile: float, side: str, band_px: int) -> np.ndarray | None:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if xx.size == 0:
        return None
    qy = float(np.quantile(yy.astype(np.float64), float(np.clip(quantile, 0.0, 1.0))))
    band = max(0, int(band_px))
    if side == "upper":
        keep = yy <= qy + band
    elif side == "lower":
        keep = yy >= qy - band
    else:
        raise ValueError(f"Unsupported side={side!r}")
    if not np.any(keep):
        keep = np.ones_like(yy, dtype=bool)
    return np.asarray([float(np.median(xx[keep])), float(np.median(yy[keep]))], dtype=np.float64)


def _mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if xx.size == 0:
        return None
    return np.asarray([float(np.mean(xx)), float(np.mean(yy))], dtype=np.float64)


def _component_edge_midpoint(
    component_masks: list[np.ndarray],
    *,
    fallback_mask: np.ndarray,
    quantile: float,
    side: str,
    band_px: int,
) -> np.ndarray | None:
    anchors: list[np.ndarray] = []
    for mask in component_masks:
        anchor = _edge_anchor(mask, quantile=quantile, side=side, band_px=band_px)
        if anchor is not None:
            anchors.append(anchor)
    if len(anchors) >= 2:
        return np.mean(np.asarray(anchors, dtype=np.float64), axis=0)
    return _edge_anchor(fallback_mask, quantile=quantile, side=side, band_px=band_px)


def _swim_anchor(mask: np.ndarray, *, mode: str, band_px: int) -> np.ndarray | None:
    normalized = str(mode).strip().lower()
    if normalized == "upper_edge":
        return _edge_anchor(mask, quantile=0.05, side="upper", band_px=band_px)
    if normalized == "centroid":
        return _mask_centroid(mask)
    raise ValueError(f"Unsupported swim anchor mode: {mode!r}")


def _rigid_from_anchor_pair(
    *,
    source_posterior: np.ndarray,
    source_anterior: np.ndarray,
    target_posterior: np.ndarray,
    target_anterior: np.ndarray,
    allow_scale: bool,
    min_axis_length_px: float,
) -> tuple[np.ndarray | None, dict[str, float | str]]:
    src_tail = np.asarray(source_posterior, dtype=np.float64)
    src_head = np.asarray(source_anterior, dtype=np.float64)
    dst_tail = np.asarray(target_posterior, dtype=np.float64)
    dst_head = np.asarray(target_anterior, dtype=np.float64)
    if not (np.isfinite(src_tail).all() and np.isfinite(src_head).all() and np.isfinite(dst_tail).all() and np.isfinite(dst_head).all()):
        return None, {"reason": "nonfinite_anchor"}

    src_vec = src_head - src_tail
    dst_vec = dst_head - dst_tail
    src_len = float(np.linalg.norm(src_vec))
    dst_len = float(np.linalg.norm(dst_vec))
    if src_len < float(min_axis_length_px) or dst_len < float(min_axis_length_px):
        return None, {
            "reason": "axis_too_short",
            "source_axis_length_px": src_len,
            "target_axis_length_px": dst_len,
        }

    src_unit = src_vec / src_len
    dst_unit = dst_vec / dst_len
    cosine = float(np.clip(np.dot(src_unit, dst_unit), -1.0, 1.0))
    sine = float(src_unit[0] * dst_unit[1] - src_unit[1] * dst_unit[0])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    scale = float(dst_len / src_len) if allow_scale else 1.0
    linear = scale * rotation
    src_mid = 0.5 * (src_tail + src_head)
    dst_mid = 0.5 * (dst_tail + dst_head)
    translation = dst_mid - linear @ src_mid
    matrix = np.column_stack([linear, translation])
    return matrix.astype(np.float64), {
        "reason": "ok",
        "source_axis_length_px": src_len,
        "target_axis_length_px": dst_len,
        "local_scale": scale,
        "local_rotation_deg": float(math.degrees(math.atan2(sine, cosine))),
        "local_translation_px": float(np.linalg.norm(translation)),
        "source_anterior_x": float(src_head[0]),
        "source_anterior_y": float(src_head[1]),
        "source_posterior_x": float(src_tail[0]),
        "source_posterior_y": float(src_tail[1]),
        "target_anterior_x": float(dst_head[0]),
        "target_anterior_y": float(dst_head[1]),
        "target_posterior_x": float(dst_tail[0]),
        "target_posterior_y": float(dst_tail[1]),
    }


def _warp_mask(mask: np.ndarray, matrix: np.ndarray, *, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return (
        cv2.warpAffine(
            np.asarray(mask, dtype=np.uint8) * 255,
            np.asarray(matrix, dtype=np.float32),
            (int(shape_hw[1]), int(shape_hw[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def _warp_frame(frame: np.ndarray, matrix: np.ndarray, *, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.warpAffine(
        frame[:, :, :3],
        np.asarray(matrix, dtype=np.float32),
        (int(shape_hw[1]), int(shape_hw[0])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _draw_axis(
    image: np.ndarray,
    *,
    posterior: np.ndarray | None,
    anterior: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    import cv2

    if posterior is None or anterior is None:
        return
    if not (np.isfinite(posterior).all() and np.isfinite(anterior).all()):
        return
    p0 = tuple(np.round(posterior).astype(int).tolist())
    p1 = tuple(np.round(anterior).astype(int).tolist())
    cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)
    cv2.circle(image, p0, 2, color, -1, cv2.LINE_AA)
    cv2.circle(image, p1, 2, color, -1, cv2.LINE_AA)


def _metric_set(
    *,
    body: np.ndarray,
    eye: np.ndarray,
    eye_exclusion: np.ndarray,
    swim: np.ndarray,
    fixed: dict[str, np.ndarray],
    fixed_swim: np.ndarray,
    roi_mask: np.ndarray,
) -> dict[str, float]:
    roi_pixels = max(1, int(np.count_nonzero(roi_mask)))
    usable = np.asarray(body, dtype=bool) & ~np.asarray(eye_exclusion, dtype=bool)
    return {
        **_mask_metrics("body", body, fixed["body_mask"]),
        **_mask_metrics("eye_union", eye, fixed["eye_mask"]),
        **_mask_metrics("eye_exclusion", eye_exclusion, fixed["eye_exclusion"]),
        **_mask_metrics("swim_bladder", swim, fixed_swim),
        "roi_body_coverage_fraction": float(np.count_nonzero(roi_mask & body) / roi_pixels),
        "roi_eye_overlap_fraction": float(np.count_nonzero(roi_mask & eye) / roi_pixels),
        "roi_eye_exclusion_overlap_fraction": float(np.count_nonzero(roi_mask & eye_exclusion) / roi_pixels),
        "roi_live_usable_fraction": float(np.count_nonzero(roi_mask & usable) / roi_pixels),
    }


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _overlay_panel(
    frame: np.ndarray,
    *,
    title: str,
    fixed: dict[str, np.ndarray],
    fixed_swim: np.ndarray,
    live_body: np.ndarray,
    live_eye: np.ndarray,
    live_swim: np.ndarray,
    live_eye_exclusion: np.ndarray,
    roi_polygon: np.ndarray,
    metrics: dict[str, float],
    frame_index: int,
    center_frame: int,
    status_valid: bool,
    fixed_anterior: np.ndarray | None,
    fixed_posterior: np.ndarray | None,
    live_anterior: np.ndarray | None,
    live_posterior: np.ndarray | None,
) -> np.ndarray:
    out = frame[:, :, :3].copy()
    roi_mask = np.asarray(fixed["roi_mask"], dtype=bool)
    out = _blend_mask(out, roi_mask, color=(0, 255, 255), alpha=0.24)
    not_live_usable = roi_mask & (~np.asarray(live_body, dtype=bool) | np.asarray(live_eye_exclusion, dtype=bool))
    out = _blend_mask(out, not_live_usable, color=(0, 0, 255), alpha=0.58)

    out = _draw_contours(out, fixed["body_mask"], color=(60, 170, 60), thickness=1)
    out = _draw_contours(out, fixed["eye_mask"], color=(0, 140, 255), thickness=1)
    out = _draw_contours(out, fixed_swim, color=(255, 140, 0), thickness=1)
    out = _draw_contours(out, fixed["roi_mask"], color=(0, 255, 255), thickness=1)
    out = _draw_contours(out, live_body, color=(255, 0, 255), thickness=1)
    out = _draw_contours(out, live_eye, color=(255, 255, 0), thickness=1)
    out = _draw_contours(out, live_swim, color=(180, 0, 255), thickness=1)
    out = _draw_polygon(out, roi_polygon, color=(255, 255, 255))

    _draw_axis(out, posterior=fixed_posterior, anterior=fixed_anterior, color=(255, 255, 255))
    _draw_axis(out, posterior=live_posterior, anterior=live_anterior, color=(0, 255, 0))

    label = "valid" if status_valid else "invalid"
    _draw_text(out, f"{title}  frame {frame_index}  center {center_frame}  {label}", origin=(8, 18), scale=0.39)
    usable = metrics.get("roi_live_usable_fraction", math.nan)
    body_shift = metrics.get("body_centroid_shift_px", math.nan)
    eye_shift = metrics.get("eye_union_centroid_shift_px", math.nan)
    swim_shift = metrics.get("swim_bladder_centroid_shift_px", math.nan)
    _draw_text(
        out,
        f"usable={usable:.3f}  body={body_shift:.1f}px  eyes={eye_shift:.1f}px  swim={swim_shift:.1f}px",
        origin=(8, 36),
        scale=0.35,
    )
    _draw_text(out, "fixed contours: body green, eyes orange, swim blue", origin=(8, out.shape[0] - 28), scale=0.35)
    _draw_text(out, "live contours: body magenta, eyes cyan, swim violet; red=rejected ROI", origin=(8, out.shape[0] - 10), scale=0.35)
    return out


def _reference_component_mask(
    *,
    mask_data: Any,
    crop_rows: list[dict[str, str]],
    keypoints: Any,
    video: Any,
    frame_id_column: str,
    keypoint_coordinate_array: str,
    stable_width: int,
    stable_height: int,
    stable_center_x: float,
    stable_center_y: float,
    origin: str,
    target_forward: str,
    scale: float,
    min_forward: float,
    min_eye_span: float,
    frame_start: int,
    frame_count: int,
    stride: int,
    occupancy_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=int(frame_start),
        frame_count=int(frame_count),
        stride=max(1, int(stride)),
    )
    if not selected:
        raise ValueError("No frames selected for the local reference mask.")
    counts, frames, reasons = _stable_component_mask_counts(
        mask_data=mask_data,
        selected=selected,
        keypoints=keypoints,
        video=video,
        frame_id_column=frame_id_column,
        keypoint_coordinate_array=keypoint_coordinate_array,
        stable_width=int(stable_width),
        stable_height=int(stable_height),
        stable_center_x=float(stable_center_x),
        stable_center_y=float(stable_center_y),
        origin=origin,
        target_forward=target_forward,
        scale=float(scale),
        min_forward=float(min_forward),
        min_eye_span=float(min_eye_span),
    )
    if int(frames) <= 0:
        raise ValueError(f"No valid projected frames for reference mask: {reasons}")
    fraction = counts.astype(np.float32) / float(frames)
    mask = fraction >= float(occupancy_threshold)
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(
            f"Reference mask is empty at occupancy_threshold={occupancy_threshold}; "
            f"valid_frames={frames}, reasons={reasons}"
        )
    return mask, {
        "frame_start": int(frame_start),
        "frame_count": int(frame_count),
        "stride": int(stride),
        "selected_frames": int(len(selected)),
        "valid_projection_frames": int(frames),
        "occupancy_threshold": float(occupancy_threshold),
        "reason_counts": reasons,
        "mask_pixels": int(np.count_nonzero(mask)),
    }


def _failure_panel(*, shape_hw: tuple[int, int], reason: str, frame_index: int, center_frame: int) -> np.ndarray:
    panel = np.zeros((*shape_hw, 3), dtype=np.uint8)
    _draw_text(panel, f"frame {frame_index} center {center_frame}", origin=(8, 24), scale=0.5)
    _draw_text(panel, f"local correction unavailable: {reason[:80]}", origin=(8, 46), scale=0.42)
    return panel


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Compare keypoint-stabilized masks to a local rostral-segment rigid correction.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Existing keypoint-stabilized video.")
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True, help="Fixed mask-relative ROI NPZ.")
    parser.add_argument("--center-frames", type=str, required=True)
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--panel-size", type=int, default=384)
    parser.add_argument("--zoom-pad-px", type=int, default=12)
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--swim-component", type=str, default="swim_bladder")
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument("--anchor-band-px", type=int, default=2)
    parser.add_argument("--eye-occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--swim-anchor-mode", choices=("upper_edge", "centroid"), default="upper_edge")
    parser.add_argument("--min-axis-length-px", type=float, default=8.0)
    parser.add_argument("--max-local-rotation-deg", type=float, default=50.0)
    parser.add_argument("--max-local-translation-px", type=float, default=150.0)
    parser.add_argument("--allow-local-scale", action="store_true")
    parser.add_argument("--reference-frame-start", type=int, default=30000)
    parser.add_argument("--reference-frame-count", type=int, default=3000)
    parser.add_argument("--reference-stride", type=int, default=10)
    parser.add_argument("--swim-occupancy-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/local_rostral_alignment_comparison.mp4"),
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
    swim_data = _load_mask_component(
        zarr_path,
        parent=mask_parent,
        run_name=mask_run,
        component_name=str(args.swim_component),
    )
    if not eye_data:
        raise ValueError("--eye-components must name at least one component.")

    fixed_swim, fixed_swim_summary = _reference_component_mask(
        mask_data=swim_data,
        crop_rows=crop_rows,
        keypoints=keypoints,
        video=crop_video_info,
        frame_id_column=frame_id_column,
        keypoint_coordinate_array=keypoint_array,
        stable_width=stable_width,
        stable_height=stable_height,
        stable_center_x=stable_center_x,
        stable_center_y=stable_center_y,
        origin=origin,
        target_forward=target_forward,
        scale=scale,
        min_forward=min_forward,
        min_eye_span=min_eye_span,
        frame_start=int(args.reference_frame_start),
        frame_count=int(args.reference_frame_count),
        stride=int(args.reference_stride),
        occupancy_threshold=float(args.swim_occupancy_threshold),
    )
    fixed_eye_components: dict[str, np.ndarray] = {}
    fixed_eye_component_summaries: dict[str, Any] = {}
    for component, data in eye_data.items():
        component_mask, component_summary = _reference_component_mask(
            mask_data=data,
            crop_rows=crop_rows,
            keypoints=keypoints,
            video=crop_video_info,
            frame_id_column=frame_id_column,
            keypoint_coordinate_array=keypoint_array,
            stable_width=stable_width,
            stable_height=stable_height,
            stable_center_x=stable_center_x,
            stable_center_y=stable_center_y,
            origin=origin,
            target_forward=target_forward,
            scale=scale,
            min_forward=min_forward,
            min_eye_span=min_eye_span,
            frame_start=int(args.reference_frame_start),
            frame_count=int(args.reference_frame_count),
            stride=int(args.reference_stride),
            occupancy_threshold=float(args.eye_occupancy_threshold),
        )
        fixed_eye_components[component] = component_mask
        fixed_eye_component_summaries[component] = component_summary
    fixed_anterior = _component_edge_midpoint(
        list(fixed_eye_components.values()),
        fallback_mask=fixed["eye_mask"],
        quantile=0.98,
        side="lower",
        band_px=int(args.anchor_band_px),
    )
    fixed_posterior = _swim_anchor(fixed_swim, mode=str(args.swim_anchor_mode), band_px=int(args.anchor_band_px))
    if fixed_anterior is None or fixed_posterior is None:
        raise ValueError("Could not compute fixed local anchors.")

    output = Path(args.output)
    ensure_output_dir(output.parent)
    panel_size = int(args.panel_size)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        (panel_size * 2, panel_size * 2),
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
    correction_failures = 0
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
            reason = "ok"
            row: dict[str, Any] = {
                "frame_index": int(frame_index),
                "center_frame": int(center_frame),
                "frame_id": int(frame_id),
                "status_valid": int(bool(status_valid)),
            }

            live_body = live_eye = live_swim = live_eye_exclusion = None
            local_body = local_eye = local_swim = local_eye_exclusion = None
            live_eye_components: dict[str, np.ndarray] = {}
            live_anterior = live_posterior = None
            local_anterior = local_posterior = None
            corrected_frame = frame
            local_metrics: dict[str, float] = {}
            current_metrics: dict[str, float] = {}

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
                    live_swim, swim_reason = _stable_project_component(
                        mask_data=swim_data,
                        frame_id=frame_id,
                        crop_row=crop_row,
                        transform=transform,
                        video=crop_video_info,
                        stable_width=stable_width,
                        stable_height=stable_height,
                    )
                    if live_body is None:
                        reason = f"body:{body_reason}"
                    elif live_swim is None:
                        reason = f"swim:{swim_reason}"
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
                                live_eye_components[component] = component_mask
                                live_eye |= component_mask
                        if not np.any(live_eye):
                            reason = "empty_eye_union:" + ",".join(f"{name}={value}" for name, value in eye_reasons.items())
                        else:
                            live_eye_exclusion = _dilate(live_eye, int(args.eye_dilate_px))
                            current_metrics = _metric_set(
                                body=live_body,
                                eye=live_eye,
                                eye_exclusion=live_eye_exclusion,
                                swim=live_swim,
                                fixed=fixed,
                                fixed_swim=fixed_swim,
                                roi_mask=roi_mask,
                            )
                            live_anterior = _component_edge_midpoint(
                                list(live_eye_components.values()),
                                fallback_mask=live_eye,
                                quantile=0.98,
                                side="lower",
                                band_px=int(args.anchor_band_px),
                            )
                            live_posterior = _swim_anchor(
                                live_swim,
                                mode=str(args.swim_anchor_mode),
                                band_px=int(args.anchor_band_px),
                            )
                            if live_anterior is None or live_posterior is None:
                                reason = "missing_live_anchor"
                            else:
                                local_matrix, local_details = _rigid_from_anchor_pair(
                                    source_posterior=live_posterior,
                                    source_anterior=live_anterior,
                                    target_posterior=fixed_posterior,
                                    target_anterior=fixed_anterior,
                                    allow_scale=bool(args.allow_local_scale),
                                    min_axis_length_px=float(args.min_axis_length_px),
                                )
                                row.update(local_details)
                                correction_status = str(local_details.get("reason", "ok"))
                                if local_matrix is None:
                                    correction_failures += 1
                                    row["local_correction_status"] = correction_status
                                else:
                                    rotation = abs(float(local_details.get("local_rotation_deg", math.nan)))
                                    translation = float(local_details.get("local_translation_px", math.nan))
                                    if np.isfinite(rotation) and rotation > float(args.max_local_rotation_deg):
                                        local_matrix = None
                                        correction_status = "rejected_rotation_limit"
                                        correction_failures += 1
                                    elif np.isfinite(translation) and translation > float(args.max_local_translation_px):
                                        local_matrix = None
                                        correction_status = "rejected_translation_limit"
                                        correction_failures += 1
                                    row["local_correction_status"] = correction_status
                                    if local_matrix is not None:
                                        local_body = _warp_mask(live_body, local_matrix, shape_hw=shape_hw)
                                        local_eye = _warp_mask(live_eye, local_matrix, shape_hw=shape_hw)
                                        local_swim = _warp_mask(live_swim, local_matrix, shape_hw=shape_hw)
                                        local_eye_exclusion = _dilate(local_eye, int(args.eye_dilate_px))
                                        corrected_frame = _warp_frame(frame, local_matrix, shape_hw=shape_hw)
                                        local_anterior = _edge_anchor(
                                            local_eye,
                                            quantile=0.98,
                                            side="lower",
                                            band_px=int(args.anchor_band_px),
                                        )
                                        local_posterior = _swim_anchor(
                                            local_swim,
                                            mode=str(args.swim_anchor_mode),
                                            band_px=int(args.anchor_band_px),
                                        )
                                        local_metrics = _metric_set(
                                            body=local_body,
                                            eye=local_eye,
                                            eye_exclusion=local_eye_exclusion,
                                            swim=local_swim,
                                            fixed=fixed,
                                            fixed_swim=fixed_swim,
                                            roi_mask=roi_mask,
                                        )

            row["reason"] = reason
            row.update(_prefixed("current_", current_metrics))
            row.update(_prefixed("local_", local_metrics))
            if current_metrics and local_metrics:
                row["delta_roi_live_usable_fraction"] = (
                    float(local_metrics["roi_live_usable_fraction"]) - float(current_metrics["roi_live_usable_fraction"])
                )
                row["delta_body_centroid_shift_px"] = (
                    float(local_metrics["body_centroid_shift_px"]) - float(current_metrics["body_centroid_shift_px"])
                )
                row["delta_eye_union_centroid_shift_px"] = (
                    float(local_metrics["eye_union_centroid_shift_px"]) - float(current_metrics["eye_union_centroid_shift_px"])
                )
                row["delta_swim_bladder_centroid_shift_px"] = (
                    float(local_metrics["swim_bladder_centroid_shift_px"]) - float(current_metrics["swim_bladder_centroid_shift_px"])
                )

            if reason != "ok":
                live_failures += 1
                current_panel = _failure_panel(shape_hw=shape_hw, reason=reason, frame_index=frame_index, center_frame=center_frame)
                local_panel = current_panel.copy()
            else:
                current_panel = _overlay_panel(
                    frame,
                    title="keypoint stable",
                    fixed=fixed,
                    fixed_swim=fixed_swim,
                    live_body=live_body,
                    live_eye=live_eye,
                    live_swim=live_swim,
                    live_eye_exclusion=live_eye_exclusion,
                    roi_polygon=roi_polygon,
                    metrics=current_metrics,
                    frame_index=frame_index,
                    center_frame=center_frame,
                    status_valid=bool(status_valid),
                    fixed_anterior=fixed_anterior,
                    fixed_posterior=fixed_posterior,
                    live_anterior=live_anterior,
                    live_posterior=live_posterior,
                )
                if local_body is None or local_eye is None or local_swim is None or local_eye_exclusion is None:
                    local_panel = _failure_panel(
                        shape_hw=shape_hw,
                        reason=str(row.get("local_correction_status", "local_correction_unavailable")),
                        frame_index=frame_index,
                        center_frame=center_frame,
                    )
                else:
                    local_panel = _overlay_panel(
                        corrected_frame,
                        title="local rostral corrected",
                        fixed=fixed,
                        fixed_swim=fixed_swim,
                        live_body=local_body,
                        live_eye=local_eye,
                        live_swim=local_swim,
                        live_eye_exclusion=local_eye_exclusion,
                        roi_polygon=roi_polygon,
                        metrics=local_metrics,
                        frame_index=frame_index,
                        center_frame=center_frame,
                        status_valid=bool(status_valid),
                        fixed_anterior=fixed_anterior,
                        fixed_posterior=fixed_posterior,
                        live_anterior=local_anterior,
                        live_posterior=local_posterior,
                    )

            current_zoom, (x0, y0, _x1, _y1) = _crop_zoom(current_panel, bbox_xyxy=roi_bbox_xyxy, pad_px=int(args.zoom_pad_px))
            local_zoom, (lx0, ly0, _lx1, _ly1) = _crop_zoom(local_panel, bbox_xyxy=roi_bbox_xyxy, pad_px=int(args.zoom_pad_px))
            current_zoom = _draw_polygon(current_zoom, roi_polygon - np.asarray([x0, y0], dtype=np.float64), color=(255, 255, 255))
            local_zoom = _draw_polygon(local_zoom, roi_polygon - np.asarray([lx0, ly0], dtype=np.float64), color=(255, 255, 255))
            _draw_text(current_zoom, "keypoint ROI zoom", origin=(8, 18), scale=0.42)
            _draw_text(local_zoom, "local-corrected ROI zoom", origin=(8, 18), scale=0.42)

            top = np.concatenate(
                [_resize_panel(current_panel, size=panel_size), _resize_panel(local_panel, size=panel_size)],
                axis=1,
            )
            bottom = np.concatenate(
                [_resize_panel(current_zoom, size=panel_size), _resize_panel(local_zoom, size=panel_size)],
                axis=1,
            )
            combined = np.concatenate([top, bottom], axis=0)
            if preview is None:
                preview = combined.copy()
            writer.write(combined)
            rendered += 1
            rows.append(row)
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
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    valid_rows = [row for row in rows if row.get("reason") == "ok"]

    def finite_values(key: str) -> list[float]:
        values: list[float] = []
        for row in valid_rows:
            if key not in row or row[key] in ("", None):
                continue
            value = float(row[key])
            if np.isfinite(value):
                values.append(value)
        return values

    metric_keys = (
        "current_roi_live_usable_fraction",
        "local_roi_live_usable_fraction",
        "delta_roi_live_usable_fraction",
        "current_body_centroid_shift_px",
        "local_body_centroid_shift_px",
        "delta_body_centroid_shift_px",
        "current_eye_union_centroid_shift_px",
        "local_eye_union_centroid_shift_px",
        "delta_eye_union_centroid_shift_px",
        "current_swim_bladder_centroid_shift_px",
        "local_swim_bladder_centroid_shift_px",
        "delta_swim_bladder_centroid_shift_px",
        "current_body_iou",
        "local_body_iou",
        "current_eye_union_iou",
        "local_eye_union_iou",
        "current_swim_bladder_iou",
        "local_swim_bladder_iou",
        "local_rotation_deg",
        "local_translation_px",
    )
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
        "correction_failures": int(correction_failures),
        "output_video": str(output),
        "preview_png": str(preview_path) if preview is not None else None,
        "csv": str(csv_path),
        "roi_bbox_xyxy": [int(value) for value in roi_bbox_xyxy.tolist()],
        "fixed_anchors": {
            "anterior_eye_bottom_xy": [float(value) for value in fixed_anterior.tolist()],
            "posterior_swim_rostral_xy": [float(value) for value in fixed_posterior.tolist()],
        },
        "fixed_swim_reference": fixed_swim_summary,
        "fixed_eye_component_references": fixed_eye_component_summaries,
        "swim_anchor_mode": str(args.swim_anchor_mode),
        "local_correction_limits": {
            "max_rotation_deg": float(args.max_local_rotation_deg),
            "max_translation_px": float(args.max_local_translation_px),
        },
        "metrics": {
            key: {
                "min": float(np.min(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for key in metric_keys
            for values in [finite_values(key)]
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
    print(f"correction_failures: {correction_failures}")


if __name__ == "__main__":
    main()
