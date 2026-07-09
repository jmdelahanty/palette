from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    polygon_mask,
    project_subject_mask_to_crop_frame,
    read_crop_meta,
    resolve_roi_rect,
    roi_rect_corners,
    selected_crop_rows,
    SubjectMaskUnavailable,
)


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


def _gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.ndim == 2:
        return np.asarray(frame, dtype=np.uint8)
    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)


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


def _video_shape(path: Path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not resolve video dimensions: {path}")
    return height, width


def _split_components(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _load_mask_component(
    zarr_path: Path,
    *,
    parent: str,
    run_name: str,
    component_name: str,
) -> Any:
    candidate_parents = (
        ("refined_subject_masks_runs", "subject_mask_runs")
        if parent.strip().lower() == "auto"
        else tuple(part.strip() for part in parent.split(",") if part.strip())
    )
    errors: list[str] = []
    for candidate_parent in candidate_parents:
        try:
            return load_subject_mask_data(
                zarr_path,
                parent=candidate_parent,
                run_name=run_name,
                component_name=component_name,
            )
        except SubjectMaskUnavailable as exc:
            errors.append(f"{candidate_parent}:{exc}")
    detail = ";".join(errors) if errors else "no candidate mask parents"
    raise SubjectMaskUnavailable(f"{component_name}:{detail}")


def _stable_component_mask_counts(
    *,
    mask_data: Any,
    selected: list[tuple[int, Any]],
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
    dilate_px: int = 0,
) -> tuple[np.ndarray, int, dict[str, int]]:
    import cv2

    _ = keypoint_coordinate_array
    counts = np.zeros((stable_height, stable_width), dtype=np.uint32)
    reasons: dict[str, int] = {}
    valid_frames = 0
    kernel = None
    if int(dilate_px) > 0:
        radius = int(dilate_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for crop_video_index, crop_row in selected:
        frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
        keypoint_row = keypoints.frame_to_row.get(frame_id)
        if keypoint_row is None:
            reasons["missing_keypoint_frame"] = reasons.get("missing_keypoint_frame", 0) + 1
            continue
        if not bool(keypoints.valid[keypoint_row]):
            reasons["invalid_keypoints"] = reasons.get("invalid_keypoints", 0) + 1
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
            reasons[transform.reason] = reasons.get(transform.reason, 0) + 1
            continue
        projected = project_subject_mask_to_crop_frame(
            mask_data,
            frame_id=frame_id,
            crop_row=crop_row,
            video_width=video.width,
            video_height=video.height,
        )
        if not projected.valid or projected.mask is None:
            reasons[projected.reason] = reasons.get(projected.reason, 0) + 1
            continue
        stable_mask = cv2.warpAffine(
            projected.mask.astype(np.uint8) * 255,
            transform.crop_to_stable.astype(np.float32),
            (stable_width, stable_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if kernel is not None:
            stable_mask = cv2.dilate(stable_mask, kernel, iterations=1)
        stable_mask_bool = stable_mask > 0
        counts[stable_mask_bool] += 1
        valid_frames += 1
        reasons["ok"] = reasons.get("ok", 0) + 1
    return counts, valid_frames, reasons


def _write_mask_filter_images(
    *,
    output_prefix: Path,
    roi_polygon: np.ndarray,
    base_roi_mask: np.ndarray,
    final_mask: np.ndarray,
    include_mask: np.ndarray | None,
    exclude_mask: np.ndarray | None,
) -> dict[str, str]:
    import cv2

    outputs: dict[str, str] = {}
    ensure_output_dir(output_prefix.parent)
    final_img = np.zeros((*base_roi_mask.shape, 3), dtype=np.uint8)
    final_img[base_roi_mask] = (48, 48, 48)
    final_img[final_mask] = (0, 255, 255)
    final_img = _draw_polygon(final_img, roi_polygon, color=(255, 255, 255))
    final_path = output_prefix.with_suffix(".mask_filtered_valid_pixels.png")
    cv2.imwrite(str(final_path), final_img)
    outputs["mask_filtered_valid_pixels_png"] = str(final_path)

    if exclude_mask is not None:
        exclude_img = np.zeros((*base_roi_mask.shape, 3), dtype=np.uint8)
        exclude_img[base_roi_mask] = (48, 48, 48)
        exclude_img[exclude_mask & base_roi_mask] = (0, 0, 255)
        exclude_img[final_mask] = (0, 255, 255)
        exclude_img = _draw_polygon(exclude_img, roi_polygon, color=(255, 255, 255))
        exclude_path = output_prefix.with_suffix(".mask_filter_excluded_pixels.png")
        cv2.imwrite(str(exclude_path), exclude_img)
        outputs["mask_filter_excluded_pixels_png"] = str(exclude_path)

    if include_mask is not None:
        include_img = np.zeros((*base_roi_mask.shape, 3), dtype=np.uint8)
        include_img[base_roi_mask] = (48, 48, 48)
        include_img[include_mask & base_roi_mask] = (0, 255, 0)
        include_img[final_mask] = (0, 255, 255)
        include_img = _draw_polygon(include_img, roi_polygon, color=(255, 255, 255))
        include_path = output_prefix.with_suffix(".mask_filter_included_pixels.png")
        cv2.imwrite(str(include_path), include_img)
        outputs["mask_filter_included_pixels_png"] = str(include_path)
    return outputs


def _build_stable_valid_pixel_mask(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    base_roi_mask: np.ndarray,
    roi_polygon: np.ndarray,
    stable_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    include_component = str(args.include_mask_component or "").strip()
    exclude_components = _split_components(args.exclude_mask_components)
    if not include_component and not exclude_components:
        return base_roi_mask, {"enabled": False}, {}

    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))
    frame_array = str(cfg_value(config, "alignment", "keypoint_frame_array", "frame_indices"))
    keypoint_array = str(cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img"))
    valid_array = str(cfg_value(config, "alignment", "keypoint_valid_array", "usable_keypoints"))
    stable_width = int(stable_shape_hw[1])
    stable_height = int(stable_shape_hw[0])
    stable_center_x = float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0))
    stable_center_y = float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0))
    origin = str(cfg_value(config, "alignment", "origin", "eye_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))
    mask_projection_stride = max(1, int(args.mask_projection_stride))

    video = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    selected_all = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
    )
    selected = selected_all[::mask_projection_stride]

    include_mask = None
    exclude_mask = None
    details: dict[str, Any] = {
        "enabled": True,
        "mask_parent": mask_parent,
        "mask_run": mask_run,
        "selected_frames": int(len(selected_all)),
        "projected_frames": int(len(selected)),
        "mask_projection_stride": mask_projection_stride,
        "include_component": include_component or None,
        "exclude_components": list(exclude_components),
        "include_occupancy_threshold": float(args.include_mask_occupancy_threshold),
        "exclude_occupancy_threshold": float(args.exclude_mask_occupancy_threshold),
        "exclude_mask_dilate_px": int(args.exclude_mask_dilate_px),
        "components": {},
    }

    final_mask = base_roi_mask.copy()
    if include_component:
        data = _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=include_component,
        )
        counts, frames, reasons = _stable_component_mask_counts(
            mask_data=data,
            selected=selected,
            keypoints=keypoints,
            video=video,
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
            dilate_px=0,
        )
        if frames <= 0:
            raise SubjectMaskUnavailable(f"No valid include-mask projections for {include_component}")
        include_fraction = counts.astype(np.float32) / float(frames)
        include_mask = include_fraction >= float(args.include_mask_occupancy_threshold)
        final_mask &= include_mask
        details["components"][include_component] = {
            "role": "include",
            "source_path": data.source_path,
            "valid_projection_frames": int(frames),
            "reason_counts": reasons,
            "included_roi_pixels": int(np.count_nonzero(include_mask & base_roi_mask)),
        }

    excluded_union = np.zeros_like(base_roi_mask, dtype=bool)
    for component in exclude_components:
        data = _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=component,
        )
        counts, frames, reasons = _stable_component_mask_counts(
            mask_data=data,
            selected=selected,
            keypoints=keypoints,
            video=video,
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
            dilate_px=int(args.exclude_mask_dilate_px),
        )
        if frames <= 0:
            raise SubjectMaskUnavailable(f"No valid exclude-mask projections for {component}")
        fraction = counts.astype(np.float32) / float(frames)
        component_excluded = fraction >= float(args.exclude_mask_occupancy_threshold)
        excluded_union |= component_excluded
        details["components"][component] = {
            "role": "exclude",
            "source_path": data.source_path,
            "valid_projection_frames": int(frames),
            "reason_counts": reasons,
            "excluded_roi_pixels": int(np.count_nonzero(component_excluded & base_roi_mask)),
        }
    if exclude_components:
        exclude_mask = excluded_union
        final_mask &= ~exclude_mask

    details["base_roi_pixels"] = int(np.count_nonzero(base_roi_mask))
    details["final_valid_roi_pixels"] = int(np.count_nonzero(final_mask))
    details["excluded_roi_pixels_union"] = (
        int(np.count_nonzero(exclude_mask & base_roi_mask)) if exclude_mask is not None else 0
    )
    if details["final_valid_roi_pixels"] <= 0:
        raise ValueError("Mask filtering removed every ROI pixel.")
    outputs = _write_mask_filter_images(
        output_prefix=args.output_prefix,
        roi_polygon=roi_polygon,
        base_roi_mask=base_roi_mask,
        final_mask=final_mask,
        include_mask=include_mask,
        exclude_mask=exclude_mask,
    )
    return final_mask, details, outputs


def _load_roi_pixel_traces(
    *,
    video_path: Path,
    roi_polygon: np.ndarray,
    status_csv: Path | None,
    frame_start: int,
    frame_count: int,
    stride: int,
    sample_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = max(0, total_frames - int(frame_start))
    frame_indices = np.arange(int(frame_start), int(frame_start) + int(frame_count), max(1, int(stride)), dtype=np.int64)
    if frame_indices.size == 0:
        raise ValueError("No frames selected.")

    roi_mask = polygon_mask((height, width), roi_polygon)
    if sample_mask is not None:
        provided = np.asarray(sample_mask, dtype=bool)
        if provided.shape != roi_mask.shape:
            raise ValueError(f"sample mask shape {provided.shape} does not match video shape {roi_mask.shape}")
        roi_mask &= provided
    yy, xx = np.nonzero(roi_mask)
    if xx.size == 0:
        raise ValueError("ROI mask contains no pixels.")
    traces = np.full((int(frame_indices.size), int(xx.size)), np.nan, dtype=np.float32)
    valid = np.zeros(int(frame_indices.size), dtype=bool)
    mean_frame_sum = np.zeros((height, width, 3), dtype=np.float64)
    mean_frame_count = 0
    status = _read_status_csv(status_csv)
    next_expected: int | None = None
    try:
        for out_row, frame_index in enumerate(frame_indices.tolist()):
            if next_expected is None or int(frame_index) != int(next_expected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            next_expected = int(frame_index) + 1
            if not ok:
                break
            if not status.get(int(frame_index), True):
                continue
            gray = _gray(frame)
            traces[out_row, :] = gray[yy, xx].astype(np.float32)
            valid[out_row] = True
            mean_frame_sum += frame[:, :, :3].astype(np.float64)
            mean_frame_count += 1
    finally:
        capture.release()

    if int(np.count_nonzero(valid)) < 16:
        raise ValueError("Fewer than 16 valid frames were loaded.")
    mean_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if mean_frame_count > 0:
        mean_frame = np.clip(mean_frame_sum / float(mean_frame_count), 0, 255).astype(np.uint8)
    return {
        "frame_indices": frame_indices,
        "traces": traces,
        "valid": valid,
        "roi_mask": roi_mask,
        "roi_x": xx.astype(np.int32),
        "roi_y": yy.astype(np.int32),
        "mean_frame": mean_frame,
        "width": width,
        "height": height,
    }


def _interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, *, max_gap: int) -> tuple[np.ndarray, int]:
    out = values.astype(np.float64, copy=True)
    finite = valid & np.isfinite(out).all(axis=1)
    if int(np.count_nonzero(finite)) < 16:
        raise ValueError("Not enough valid rows to interpolate traces.")
    invalid = ~finite
    interpolated_rows = 0
    idx = 0
    while idx < len(finite):
        if finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(finite) and not finite[idx]:
            idx += 1
        stop = idx
        if start == 0 or stop >= len(finite) or stop - start > int(max_gap):
            continue
        left = out[start - 1]
        right = out[stop]
        steps = float(stop - start + 1)
        for offset, row in enumerate(range(start, stop), start=1):
            frac = float(offset) / steps
            out[row] = (1.0 - frac) * left + frac * right
            finite[row] = True
            interpolated_rows += 1
    kept = finite
    first = int(np.flatnonzero(kept)[0])
    last = int(np.flatnonzero(kept)[-1]) + 1
    contiguous = kept[first:last]
    if not np.all(contiguous):
        # Fall back to the longest contiguous segment after short-gap interpolation.
        best_start = first
        best_stop = first + 1
        run_start: int | None = None
        for row, ok in enumerate(kept):
            if ok and run_start is None:
                run_start = row
            if (not ok or row == len(kept) - 1) and run_start is not None:
                run_stop = row if not ok else row + 1
                if run_stop - run_start > best_stop - best_start:
                    best_start, best_stop = run_start, run_stop
                run_start = None
        first, last = best_start, best_stop
    return out[first:last], interpolated_rows


def _bandpass_matrix(
    traces: np.ndarray,
    *,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> np.ndarray:
    from scipy import signal

    sample_rate = float(fps)
    if sample_rate <= 0 or not np.isfinite(sample_rate):
        raise ValueError(f"Invalid fps: {fps}")
    nyquist = sample_rate / 2.0
    low = max(0.001, float(band_min_hz))
    high = min(float(band_max_hz), nyquist * 0.98)
    if low >= high:
        raise ValueError(f"Invalid band after Nyquist clamp: {low}..{high} Hz")
    centered = traces - np.nanmean(traces, axis=0, keepdims=True)
    detrended = signal.detrend(centered, axis=0, type="linear")
    sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, detrended, axis=0)


def _safe_corr_with_reference(traces: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = traces - np.mean(traces, axis=0, keepdims=True)
    y = reference.astype(np.float64) - float(np.mean(reference))
    numerator = np.sum(x * y[:, None], axis=0)
    denom = np.sqrt(np.sum(x * x, axis=0) * float(np.sum(y * y)))
    out = np.zeros(traces.shape[1], dtype=np.float64)
    ok = denom > 0
    out[ok] = numerator[ok] / denom[ok]
    return np.clip(out, -1.0, 1.0)


def _scatter_to_image(
    values: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    shape_hw: tuple[int, int],
    fill: float = math.nan,
) -> np.ndarray:
    image = np.full(shape_hw, float(fill), dtype=np.float32)
    image[y, x] = np.asarray(values, dtype=np.float32)
    return image


def _write_maps(
    *,
    output_prefix: Path,
    mean_frame: np.ndarray,
    roi_polygon: np.ndarray,
    roi_mask: np.ndarray,
    roi_x: np.ndarray,
    roi_y: np.ndarray,
    band_power: np.ndarray,
    correlation: np.ndarray,
    signed_covariance: np.ndarray,
    roi_signal: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(output_prefix.parent)
    shape = roi_mask.shape
    power_image = _scatter_to_image(band_power, x=roi_x, y=roi_y, shape_hw=shape)
    corr_image = _scatter_to_image(correlation, x=roi_x, y=roi_y, shape_hw=shape)
    cov_image = _scatter_to_image(signed_covariance, x=roi_x, y=roi_y, shape_hw=shape)

    pad = 8
    y0 = max(0, int(np.min(roi_y)) - pad)
    y1 = min(shape[0], int(np.max(roi_y)) + pad + 1)
    x0 = max(0, int(np.min(roi_x)) - pad)
    x1 = min(shape[1], int(np.max(roi_x)) + pad + 1)
    extent = [x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5]

    figure_path = output_prefix.with_suffix(".pixel_band_maps.png")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    mean_with_roi = _draw_polygon(mean_frame, roi_polygon, color=(0, 255, 255))
    axes[0, 0].imshow(cv2.cvtColor(mean_with_roi, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Mean stabilized frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(mean_with_roi[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=extent)
    axes[0, 1].set_title("ROI zoom")
    axes[0, 1].set_xlim(x0 - 0.5, x1 - 0.5)
    axes[0, 1].set_ylim(y1 - 0.5, y0 - 0.5)

    used = np.zeros((*shape, 3), dtype=np.uint8)
    used[roi_mask] = (255, 255, 0)
    used = _draw_polygon(used, roi_polygon, color=(255, 255, 255))
    axes[0, 2].imshow(cv2.cvtColor(used[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=extent)
    axes[0, 2].set_title("Pixels included in ROI")
    axes[0, 2].set_xlim(x0 - 0.5, x1 - 0.5)
    axes[0, 2].set_ylim(y1 - 0.5, y0 - 0.5)

    power_crop = np.ma.masked_invalid(power_image[y0:y1, x0:x1])
    corr_crop = np.ma.masked_invalid(corr_image[y0:y1, x0:x1])
    cov_crop = np.ma.masked_invalid(cov_image[y0:y1, x0:x1])

    im = axes[1, 0].imshow(power_crop, cmap="magma", interpolation="nearest", extent=extent)
    axes[1, 0].set_title(f"Per-pixel band power {band_min_hz:g}-{band_max_hz:g} Hz")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(corr_crop, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", extent=extent)
    axes[1, 1].set_title("Correlation with band-passed ROI mean")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    max_abs_cov = float(np.nanmax(np.abs(signed_covariance))) if signed_covariance.size else 1.0
    if not np.isfinite(max_abs_cov) or max_abs_cov <= 0:
        max_abs_cov = 1.0
    im = axes[1, 2].imshow(
        cov_crop,
        cmap="coolwarm",
        vmin=-max_abs_cov,
        vmax=max_abs_cov,
        interpolation="nearest",
        extent=extent,
    )
    axes[1, 2].set_title("Signed covariance with ROI rhythm")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    for axis in axes[1]:
        axis.set_xlim(x0 - 0.5, x1 - 0.5)
        axis.set_ylim(y1 - 0.5, y0 - 0.5)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    trace_path = output_prefix.with_suffix(".roi_band_trace.png")
    t = np.arange(len(roi_signal), dtype=np.float64) / float(fps)
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.plot(t, roi_signal, lw=0.8)
    ax.set_title(f"Band-passed ROI mean signal, {band_min_hz:g}-{band_max_hz:g} Hz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("intensity delta")
    fig.savefig(trace_path, dpi=160)
    plt.close(fig)

    npz_path = output_prefix.with_suffix(".pixel_band_maps.npz")
    np.savez_compressed(
        npz_path,
        roi_mask=roi_mask.astype(np.uint8),
        roi_x=roi_x.astype(np.int32),
        roi_y=roi_y.astype(np.int32),
        band_power=band_power.astype(np.float32),
        correlation=correlation.astype(np.float32),
        signed_covariance=signed_covariance.astype(np.float32),
        power_image=power_image.astype(np.float32),
        correlation_image=corr_image.astype(np.float32),
        signed_covariance_image=cov_image.astype(np.float32),
        roi_bandpassed_signal=roi_signal.astype(np.float32),
    )
    return {
        "pixel_band_maps_png": str(figure_path),
        "roi_band_trace_png": str(trace_path),
        "pixel_band_maps_npz": str(npz_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map per-pixel heartbeat-band fluctuations inside a stabilized ROI.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to sample.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video status CSV.")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=6000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument(
        "--include-mask-component",
        type=str,
        default=None,
        help="Optional subject-mask component that valid ROI pixels must occupy, e.g. subject_body.",
    )
    parser.add_argument(
        "--exclude-mask-components",
        type=str,
        default="",
        help="Comma-separated subject-mask components to exclude from valid ROI pixels, e.g. eye_left,eye_right.",
    )
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument(
        "--include-mask-occupancy-threshold",
        type=float,
        default=0.5,
        help="Stable pixel must be inside the include component in at least this fraction of projected mask frames.",
    )
    parser.add_argument(
        "--exclude-mask-occupancy-threshold",
        type=float,
        default=0.05,
        help="Stable pixel is excluded if it is inside an exclude component in at least this fraction of projected mask frames.",
    )
    parser.add_argument(
        "--exclude-mask-dilate-px",
        type=int,
        default=2,
        help="Dilate projected exclusion masks by this many stabilized pixels before thresholding.",
    )
    parser.add_argument(
        "--mask-projection-stride",
        type=int,
        default=1,
        help="Use every Nth selected frame when projecting masks to build the stable validity map.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_pixel_band_contributions"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    stable_shape_hw = _video_shape(args.video)
    base_roi_mask = polygon_mask(stable_shape_hw, roi_polygon)
    sample_mask, mask_filter_summary, mask_filter_outputs = _build_stable_valid_pixel_mask(
        config=config,
        args=args,
        base_roi_mask=base_roi_mask,
        roi_polygon=roi_polygon,
        stable_shape_hw=stable_shape_hw,
    )
    loaded = _load_roi_pixel_traces(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
    )
    traces, interpolated_rows = _interpolate_short_gaps(
        loaded["traces"],
        loaded["valid"],
        max_gap=max(0, int(args.max_interpolated_gap_samples)),
    )
    effective_fps = float(args.fps) / max(1, int(args.stride))
    bandpassed = _bandpass_matrix(
        traces,
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )
    roi_signal = np.mean(bandpassed, axis=1)
    band_power = np.mean(bandpassed * bandpassed, axis=0)
    correlation = _safe_corr_with_reference(bandpassed, roi_signal)
    signed_covariance = np.mean((bandpassed - np.mean(bandpassed, axis=0)) * (roi_signal[:, None] - np.mean(roi_signal)), axis=0)

    outputs = _write_maps(
        output_prefix=args.output_prefix,
        mean_frame=loaded["mean_frame"],
        roi_polygon=roi_polygon,
        roi_mask=loaded["roi_mask"],
        roi_x=loaded["roi_x"],
        roi_y=loaded["roi_y"],
        band_power=band_power,
        correlation=correlation,
        signed_covariance=signed_covariance,
        roi_signal=roi_signal,
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )
    outputs.update(mask_filter_outputs)

    top_count = min(12, int(band_power.size))
    top_by_power = np.argsort(band_power)[::-1][:top_count]
    top_by_correlation = np.argsort(correlation)[::-1][:top_count]
    summary: dict[str, Any] = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "stride": int(args.stride),
        "fps": effective_fps,
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "loaded_frames": int(loaded["traces"].shape[0]),
        "valid_frames": int(np.count_nonzero(loaded["valid"])),
        "analysis_frames": int(traces.shape[0]),
        "interpolated_rows": int(interpolated_rows),
        "roi_pixel_count": int(band_power.size),
        "mask_filter": mask_filter_summary,
        "band_power_min": float(np.min(band_power)),
        "band_power_median": float(np.median(band_power)),
        "band_power_max": float(np.max(band_power)),
        "correlation_min": float(np.min(correlation)),
        "correlation_median": float(np.median(correlation)),
        "correlation_max": float(np.max(correlation)),
        "top_pixels_by_band_power": [
            {
                "x": int(loaded["roi_x"][idx]),
                "y": int(loaded["roi_y"][idx]),
                "band_power": float(band_power[idx]),
                "correlation": float(correlation[idx]),
                "signed_covariance": float(signed_covariance[idx]),
            }
            for idx in top_by_power
        ],
        "top_pixels_by_correlation": [
            {
                "x": int(loaded["roi_x"][idx]),
                "y": int(loaded["roi_y"][idx]),
                "band_power": float(band_power[idx]),
                "correlation": float(correlation[idx]),
                "signed_covariance": float(signed_covariance[idx]),
            }
            for idx in top_by_correlation
        ],
        "outputs": outputs,
    }
    summary_path = args.output_prefix.with_suffix(".summary.json")
    ensure_output_dir(summary_path.parent)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_path}")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    print(f"analysis_frames: {summary['analysis_frames']}")
    print(f"roi_pixel_count: {summary['roi_pixel_count']}")
    print(f"band_power_max: {summary['band_power_max']:.6g}")
    print(f"correlation_max: {summary['correlation_max']:.6g}")


if __name__ == "__main__":
    main()
