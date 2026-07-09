from __future__ import annotations

import argparse
from collections import Counter
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
from make_subset_clip import _apply_circular_mask


def _read_status_csv(path: Path | None) -> dict[int, bool]:
    if path is None:
        return {}
    status: dict[int, bool] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "output_frame_index" not in (reader.fieldnames or []):
            raise ValueError(f"status CSV lacks output_frame_index: {path}")
        for row in reader:
            idx = int(row["output_frame_index"])
            status[idx] = str(row.get("valid", "1")).strip() not in {"", "0", "false", "False"}
    return status


def _read_frame(capture: Any, *, frame_index: int, next_expected_index: int | None) -> tuple[bool, np.ndarray | None, int]:
    import cv2

    if next_expected_index is None or int(frame_index) != int(next_expected_index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return bool(ok), frame if ok else None, int(frame_index) + 1


def _gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.ndim == 2:
        return np.asarray(frame, dtype=np.uint8)
    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)


def _mean_rgb(sum_image: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(sum_image.shape, dtype=np.uint8)
    return np.clip(sum_image / float(count), 0, 255).astype(np.uint8)


def _mean_from_sum(sum_values: np.ndarray, count_values: np.ndarray) -> np.ndarray:
    out = np.zeros(sum_values.shape, dtype=np.float32)
    mask = count_values > 0
    out[mask] = (sum_values[mask] / count_values[mask]).astype(np.float32)
    return out


def _draw_polygon(image: np.ndarray, points_xy: np.ndarray, *, color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
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
            thickness=int(thickness),
            lineType=cv2.LINE_AA,
        )
    return out


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float = 0.65,
) -> np.ndarray:
    import cv2

    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != out.shape[:2] or not np.any(mask_bool):
        return out
    overlay = np.zeros_like(out)
    overlay[mask_bool] = np.asarray(color, dtype=np.uint8)
    blended = cv2.addWeighted(out, 1.0 - float(alpha), overlay, float(alpha), 0.0)
    out[mask_bool] = blended[mask_bool]
    return out


def _used_pixels_image(counts: np.ndarray, *, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    out = np.zeros((*counts.shape[:2], 3), dtype=np.uint8)
    out[np.asarray(counts) > 0] = np.asarray(color, dtype=np.uint8)
    return out


def _write_count_products(
    *,
    output_prefix: Path,
    stable_count: np.ndarray,
    stable_sum: np.ndarray,
    stable_base_sum: np.ndarray,
    stable_base_count: int,
    roi_stable: np.ndarray,
    crop_count: np.ndarray | None = None,
    crop_sum: np.ndarray | None = None,
    crop_base_sum: np.ndarray | None = None,
    crop_base_count: int = 0,
    representative_crop_roi: np.ndarray | None = None,
) -> dict[str, str]:
    import cv2

    ensure_output_dir(output_prefix.parent)
    outputs: dict[str, str] = {}

    stable_mean = _mean_from_sum(stable_sum, stable_count)
    stable_base = _mean_rgb(stable_base_sum, stable_base_count)
    stable_max = int(np.max(stable_count)) if stable_count.size else 0
    stable_norm = np.zeros(stable_count.shape, dtype=np.uint8)
    if stable_max > 0:
        stable_norm = np.clip(stable_count.astype(np.float32) * (255.0 / stable_max), 0, 255).astype(np.uint8)
    stable_heat = cv2.applyColorMap(stable_norm, cv2.COLORMAP_TURBO)
    stable_overlay = stable_base.copy()
    mask = stable_count > 0
    if np.any(mask):
        blended = cv2.addWeighted(stable_overlay, 0.60, stable_heat, 0.40, 0.0)
        stable_overlay[mask] = blended[mask]
    stable_overlay = _draw_polygon(stable_overlay, roi_stable, color=(0, 255, 255), thickness=1)
    stable_used = _draw_polygon(
        _used_pixels_image(stable_count, color=(0, 255, 255)),
        roi_stable,
        color=(255, 255, 255),
        thickness=1,
    )

    stable_count_png = output_prefix.with_suffix(".stable_count.png")
    stable_overlay_png = output_prefix.with_suffix(".stable_overlay.png")
    stable_used_png = output_prefix.with_suffix(".stable_used_pixels.png")
    stable_mean_png = output_prefix.with_suffix(".stable_mean_intensity.png")
    cv2.imwrite(str(stable_count_png), stable_heat)
    cv2.imwrite(str(stable_overlay_png), stable_overlay)
    cv2.imwrite(str(stable_used_png), stable_used)
    cv2.imwrite(str(stable_mean_png), np.clip(stable_mean, 0, 255).astype(np.uint8))
    outputs["stable_count_png"] = str(stable_count_png)
    outputs["stable_overlay_png"] = str(stable_overlay_png)
    outputs["stable_used_pixels_png"] = str(stable_used_png)
    outputs["stable_mean_intensity_png"] = str(stable_mean_png)

    npz_payload: dict[str, np.ndarray] = {
        "stable_count": stable_count.astype(np.uint32),
        "stable_intensity_sum": stable_sum.astype(np.float64),
        "stable_mean_intensity": stable_mean.astype(np.float32),
    }

    if crop_count is not None and crop_sum is not None and crop_base_sum is not None:
        crop_mean = _mean_from_sum(crop_sum, crop_count)
        crop_base = _mean_rgb(crop_base_sum, crop_base_count)
        crop_max = int(np.max(crop_count)) if crop_count.size else 0
        crop_norm = np.zeros(crop_count.shape, dtype=np.uint8)
        if crop_max > 0:
            crop_norm = np.clip(crop_count.astype(np.float32) * (255.0 / crop_max), 0, 255).astype(np.uint8)
        crop_heat = cv2.applyColorMap(crop_norm, cv2.COLORMAP_TURBO)
        crop_overlay = crop_base.copy()
        crop_mask = crop_count > 0
        if np.any(crop_mask):
            blended = cv2.addWeighted(crop_overlay, 0.60, crop_heat, 0.40, 0.0)
            crop_overlay[crop_mask] = blended[crop_mask]
        if representative_crop_roi is not None:
            crop_overlay = _draw_polygon(crop_overlay, representative_crop_roi, color=(0, 255, 255), thickness=1)
        crop_used = _used_pixels_image(crop_count, color=(0, 255, 255))
        if representative_crop_roi is not None:
            crop_used = _draw_polygon(crop_used, representative_crop_roi, color=(255, 255, 255), thickness=1)

        crop_count_png = output_prefix.with_suffix(".crop_count.png")
        crop_overlay_png = output_prefix.with_suffix(".crop_overlay.png")
        crop_used_png = output_prefix.with_suffix(".crop_used_pixels.png")
        crop_mean_png = output_prefix.with_suffix(".crop_mean_intensity.png")
        cv2.imwrite(str(crop_count_png), crop_heat)
        cv2.imwrite(str(crop_overlay_png), crop_overlay)
        cv2.imwrite(str(crop_used_png), crop_used)
        cv2.imwrite(str(crop_mean_png), np.clip(crop_mean, 0, 255).astype(np.uint8))
        outputs["crop_count_png"] = str(crop_count_png)
        outputs["crop_overlay_png"] = str(crop_overlay_png)
        outputs["crop_used_pixels_png"] = str(crop_used_png)
        outputs["crop_mean_intensity_png"] = str(crop_mean_png)
        npz_payload.update(
            {
                "crop_count": crop_count.astype(np.uint32),
                "crop_intensity_sum": crop_sum.astype(np.float64),
                "crop_mean_intensity": crop_mean.astype(np.float32),
            }
        )

    maps_npz = output_prefix.with_suffix(".maps.npz")
    np.savez_compressed(maps_npz, **npz_payload)
    outputs["maps_npz"] = str(maps_npz)
    return outputs


def _write_summary(output_prefix: Path, payload: dict[str, Any]) -> Path:
    path = output_prefix.with_suffix(".summary.json")
    ensure_output_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _map_stabilized_video(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    import cv2

    if args.stabilized_video is None:
        raise ValueError("--stabilized-video is required for stabilized-video mode")
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_stable = roi_rect_corners(roi_rect)

    capture = cv2.VideoCapture(str(args.stabilized_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open stabilized video: {args.stabilized_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    status = _read_status_csv(args.status_csv)

    frame_start = max(0, int(args.frame_start or 0))
    frame_count = int(args.frame_count) if args.frame_count is not None else max(0, total_frames - frame_start)
    stride = max(1, int(args.stride))
    stop = frame_start + max(0, frame_count)

    roi_mask = polygon_mask((height, width), roi_stable)
    stable_count = np.zeros((height, width), dtype=np.uint32)
    stable_sum = np.zeros((height, width), dtype=np.float64)
    stable_base_sum = np.zeros((height, width, 3), dtype=np.float64)
    reasons = Counter()
    valid_frames = 0
    total_sample_pixels = 0
    debug_paths: list[str] = []
    debug_limit = max(0, int(args.debug_frames))
    next_expected: int | None = None
    try:
        for frame_index in range(frame_start, stop, stride):
            ok, frame, next_expected = _read_frame(
                capture,
                frame_index=frame_index,
                next_expected_index=next_expected,
            )
            if not ok or frame is None:
                reasons["video_read_failed"] += 1
                break
            if not status.get(frame_index, True):
                reasons["invalid_status_frame"] += 1
                continue
            if not np.any(roi_mask):
                reasons["empty_roi"] += 1
                continue
            gray = _gray(frame)
            stable_count[roi_mask] += 1
            stable_sum[roi_mask] += gray[roi_mask].astype(np.float64)
            stable_base_sum += frame[:, :, :3].astype(np.float64)
            if len(debug_paths) < debug_limit:
                debug = _overlay_mask(frame[:, :, :3], roi_mask, color=(0, 255, 255), alpha=0.75)
                debug = _draw_polygon(debug, roi_stable, color=(255, 255, 255), thickness=1)
                debug_path = args.output_prefix.with_suffix(f".debug_stable_frame_{frame_index:06d}.png")
                cv2.imwrite(str(debug_path), debug)
                debug_paths.append(str(debug_path))
            valid_frames += 1
            total_sample_pixels += int(np.count_nonzero(roi_mask))
            reasons["ok"] += 1
    finally:
        capture.release()

    outputs = _write_count_products(
        output_prefix=args.output_prefix,
        stable_count=stable_count,
        stable_sum=stable_sum,
        stable_base_sum=stable_base_sum,
        stable_base_count=valid_frames,
        roi_stable=roi_stable,
    )
    return {
        "mode": "stabilized_video",
        "source_video": str(args.stabilized_video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "frame_start": frame_start,
        "frame_count_requested": frame_count,
        "stride": stride,
        "frames_used": int(valid_frames),
        "total_sample_pixels": int(total_sample_pixels),
        "mean_sample_pixels_per_used_frame": float(total_sample_pixels / valid_frames) if valid_frames else math.nan,
        "max_stable_pixel_contribution_count": int(np.max(stable_count)) if stable_count.size else 0,
        "reason_counts": dict(reasons),
        "debug_frame_pngs": debug_paths,
        "outputs": outputs,
    }


def _load_mask_data(
    *,
    zarr_path: Path,
    mask_enabled: bool,
    mask_parent: str,
    mask_run: str,
    mask_component: str,
    require_mask: bool,
) -> tuple[Any | None, str]:
    if not mask_enabled:
        return None, "mask_disabled"
    candidate_parents = (
        ("refined_subject_masks_runs", "subject_mask_runs")
        if mask_parent.strip().lower() == "auto"
        else tuple(part.strip() for part in mask_parent.split(",") if part.strip())
    )
    mask_errors: list[str] = []
    for candidate_parent in candidate_parents:
        try:
            return (
                load_subject_mask_data(
                    zarr_path,
                    parent=candidate_parent,
                    run_name=mask_run,
                    component_name=mask_component,
                ),
                "ok",
            )
        except SubjectMaskUnavailable as exc:
            mask_errors.append(f"{candidate_parent}:{exc}")
    reason = ";".join(mask_errors) if mask_errors else "mask_unavailable"
    if require_mask:
        raise SubjectMaskUnavailable(reason)
    return None, reason


def _map_source_crop(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
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
    mask_enabled = bool(args.mask or cfg_value(config, "mask", "enabled", False))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))
    mask_component = str(args.mask_component or cfg_value(config, "mask", "component", "subject_body"))
    require_mask = bool(args.require_mask or cfg_value(config, "mask", "require_mask", False))
    frame_start = args.frame_start
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))
    frame_count = int(args.frame_count or cfg_value(config, "probe", "frame_count", 120))
    stride = max(1, int(args.stride))

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
    mask_data, mask_setup_reason = _load_mask_data(
        zarr_path=zarr_path,
        mask_enabled=mask_enabled,
        mask_parent=mask_parent,
        mask_run=mask_run,
        mask_component=mask_component,
        require_mask=require_mask,
    )

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {crop_video}")

    crop_count = np.zeros((int(video.height), int(video.width)), dtype=np.uint32)
    crop_sum = np.zeros((int(video.height), int(video.width)), dtype=np.float64)
    crop_base_sum = np.zeros((int(video.height), int(video.width), 3), dtype=np.float64)
    stable_count = np.zeros((stable_height, stable_width), dtype=np.uint32)
    stable_sum = np.zeros((stable_height, stable_width), dtype=np.float64)
    stable_base_sum = np.zeros((stable_height, stable_width, 3), dtype=np.float64)
    reasons = Counter()
    mask_reasons = Counter()
    valid_frames = 0
    valid_transform_frames = 0
    total_sample_pixels = 0
    representative_crop_roi: np.ndarray | None = None
    debug_paths: list[dict[str, str]] = []
    debug_limit = max(0, int(args.debug_frames))
    next_expected: int | None = None

    try:
        for crop_video_index, crop_row in selected:
            frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
            keypoint_row = keypoints.frame_to_row.get(frame_id)
            ok, frame, next_expected = _read_frame(
                capture,
                frame_index=crop_video_index,
                next_expected_index=next_expected,
            )
            if not ok or frame is None:
                reasons["video_read_failed"] += 1
                continue
            if keypoint_row is None:
                reasons["missing_keypoint_frame"] += 1
                continue
            if not bool(keypoints.valid[keypoint_row]):
                reasons["invalid_keypoints"] += 1
                continue

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
                reasons[transform.reason] += 1
                continue

            valid_transform_frames += 1
            stable_frame = cv2.warpAffine(
                frame,
                transform.crop_to_stable.astype(np.float32),
                (stable_width, stable_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if stable_circular_mask:
                stable_frame = _apply_circular_mask(
                    stable_frame,
                    center_x=stable_center_x,
                    center_y=stable_center_y,
                    radius_px=stable_mask_radius,
                )
            stable_base_sum += stable_frame[:, :, :3].astype(np.float64)
            crop_base_sum += frame[:, :, :3].astype(np.float64)

            roi_crop = transform_points(transform.stable_to_crop, roi_stable)
            if representative_crop_roi is None:
                representative_crop_roi = roi_crop
            roi_mask = polygon_mask(frame.shape[:2], roi_crop)
            sample_mask = roi_mask
            mask_reason = "mask_disabled" if not mask_enabled else mask_setup_reason
            projected_mask: np.ndarray | None = None
            if mask_enabled:
                if mask_data is None:
                    if require_mask:
                        reasons[mask_setup_reason] += 1
                        mask_reasons[mask_setup_reason] += 1
                        continue
                    mask_reasons[mask_setup_reason] += 1
                else:
                    projected = project_subject_mask_to_crop_frame(
                        mask_data,
                        frame_id=frame_id,
                        crop_row=crop_row,
                        video_width=video.width,
                        video_height=video.height,
                    )
                    mask_reason = projected.reason
                    mask_reasons[mask_reason] += 1
                    if projected.valid and projected.mask is not None:
                        projected_mask = projected.mask
                        sample_mask = roi_mask & projected.mask
                    elif require_mask:
                        reasons[projected.reason] += 1
                        continue

            mean_intensity, pixel_count = mask_mean_intensity(frame, sample_mask)
            if pixel_count <= 0 or not np.isfinite(mean_intensity):
                reasons["masked_roi_empty" if mask_enabled else "roi_outside_crop"] += 1
                continue

            gray = _gray(frame)
            sample_mask_bool = np.asarray(sample_mask, dtype=bool)
            crop_count[sample_mask_bool] += 1
            crop_sum[sample_mask_bool] += gray[sample_mask_bool].astype(np.float64)

            stable_sample_mask = cv2.warpAffine(
                sample_mask_bool.astype(np.uint8) * 255,
                transform.crop_to_stable.astype(np.float32),
                (stable_width, stable_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            stable_gray = _gray(stable_frame)
            stable_count[stable_sample_mask] += 1
            stable_sum[stable_sample_mask] += stable_gray[stable_sample_mask].astype(np.float64)

            if len(debug_paths) < debug_limit:
                crop_debug = frame[:, :, :3].copy()
                if projected_mask is not None:
                    crop_debug = _overlay_mask(crop_debug, projected_mask, color=(255, 0, 255), alpha=0.20)
                crop_debug = _overlay_mask(crop_debug, sample_mask_bool, color=(0, 255, 255), alpha=0.80)
                crop_debug = _draw_polygon(crop_debug, roi_crop, color=(255, 255, 255), thickness=1)

                stable_debug = stable_frame[:, :, :3].copy()
                stable_debug = _overlay_mask(stable_debug, stable_sample_mask, color=(0, 255, 255), alpha=0.80)
                stable_debug = _draw_polygon(stable_debug, roi_stable, color=(255, 255, 255), thickness=1)

                crop_debug_path = args.output_prefix.with_suffix(
                    f".debug_crop_frame_{crop_video_index:06d}.png"
                )
                stable_debug_path = args.output_prefix.with_suffix(
                    f".debug_stable_frame_{crop_video_index:06d}.png"
                )
                cv2.imwrite(str(crop_debug_path), crop_debug)
                cv2.imwrite(str(stable_debug_path), stable_debug)
                debug_paths.append(
                    {
                        "crop_frame": str(crop_debug_path),
                        "stable_frame": str(stable_debug_path),
                        "crop_video_frame_index": str(int(crop_video_index)),
                        "frame_id": str(int(frame_id)),
                        "sample_pixel_count": str(int(pixel_count)),
                        "mask_reason": str(mask_reason),
                    }
                )

            valid_frames += 1
            total_sample_pixels += int(pixel_count)
            reasons["ok"] += 1
    finally:
        capture.release()

    outputs = _write_count_products(
        output_prefix=args.output_prefix,
        stable_count=stable_count,
        stable_sum=stable_sum,
        stable_base_sum=stable_base_sum,
        stable_base_count=valid_transform_frames,
        roi_stable=roi_stable,
        crop_count=crop_count,
        crop_sum=crop_sum,
        crop_base_sum=crop_base_sum,
        crop_base_count=valid_transform_frames,
        representative_crop_roi=representative_crop_roi,
    )
    return {
        "mode": "source_crop",
        "source_crop_video": str(crop_video),
        "source_crop_meta_csv": str(crop_meta_csv),
        "source_zarr_path": str(zarr_path),
        "source_keypoint_group": keypoint_group,
        "frame_id_column": frame_id_column,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "frame_start": int(frame_start),
        "frame_count_requested": int(frame_count),
        "stride": stride,
        "frames_selected": int(len(selected)),
        "frames_with_valid_transform": int(valid_transform_frames),
        "frames_used": int(valid_frames),
        "total_sample_pixels": int(total_sample_pixels),
        "mean_sample_pixels_per_used_frame": float(total_sample_pixels / valid_frames) if valid_frames else math.nan,
        "max_stable_pixel_contribution_count": int(np.max(stable_count)) if stable_count.size else 0,
        "max_crop_pixel_contribution_count": int(np.max(crop_count)) if crop_count.size else 0,
        "mask_enabled": bool(mask_enabled),
        "mask_require": bool(require_mask),
        "mask_setup_reason": mask_setup_reason,
        "mask_source_path": mask_data.source_path if mask_data is not None else None,
        "mask_component": mask_data.component_name if mask_data is not None else mask_component,
        "mask_storage_surface": mask_data.storage_surface if mask_data is not None else None,
        "reason_counts": dict(reasons),
        "mask_reason_counts": dict(mask_reasons),
        "debug_frame_pngs": debug_paths,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map which pixels contribute to a stabilized ROI intensity signal.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--roi", type=str, default=None, help="Stabilized-view rectangle as x,y,width,height.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--stabilized-video",
        type=Path,
        default=None,
        help="If supplied, map the fixed ROI pixels used directly from this stabilized video.",
    )
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video valid-frame CSV.")
    parser.add_argument("--mask", action="store_true", help="Intersect source-crop ROI with a projected subject mask.")
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--mask-component", type=str, default=None, help="Semantic mask component to use.")
    parser.add_argument("--require-mask", action="store_true", help="Skip frames instead of falling back to ROI-only.")
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="Write this many per-frame overlays showing the exact sampled pixels.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_contribution_map"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dir(args.output_prefix.parent)
    if args.stabilized_video is not None:
        summary = _map_stabilized_video(args, config)
    else:
        summary = _map_source_crop(args, config)
    summary_path = _write_summary(args.output_prefix, summary)

    print(f"summary_json: {summary_path}")
    for key, value in summary["outputs"].items():
        print(f"{key}: {value}")
    print(f"mode: {summary['mode']}")
    print(f"frames_used: {summary['frames_used']}")
    print(f"mean_sample_pixels_per_used_frame: {summary['mean_sample_pixels_per_used_frame']:.3f}")


if __name__ == "__main__":
    main()
