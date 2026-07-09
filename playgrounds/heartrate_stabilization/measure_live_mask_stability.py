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
    project_subject_mask_to_crop_frame,
    read_crop_meta,
    selected_crop_rows,
)
from map_pixel_band_contributions import _load_mask_component, _split_components


def _load_fixed_masks(path: Path, *, shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ("roi_mask", "body_mask", "eye_mask")
        for key in required:
            if key not in data:
                raise ValueError(f"{path} lacks required mask array {key!r}")
        fixed = {
            "roi_mask": np.asarray(data["roi_mask"], dtype=bool),
            "body_mask": np.asarray(data["body_mask"], dtype=bool),
            "eye_mask": np.asarray(data["eye_mask"], dtype=bool),
            "eye_exclusion": np.asarray(data["eye_exclusion"], dtype=bool)
            if "eye_exclusion" in data
            else np.asarray(data["eye_mask"], dtype=bool),
        }
    for key, value in fixed.items():
        if tuple(value.shape) != tuple(shape_hw):
            raise ValueError(f"{path} {key} shape {value.shape} does not match stable shape {shape_hw}")
    return fixed


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if int(pixels) <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), iterations=int(pixels))


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if xx.size == 0:
        return None
    return float(np.mean(xx)), float(np.mean(yy))


def _mask_metrics(prefix: str, live: np.ndarray, fixed: np.ndarray) -> dict[str, float]:
    live_mask = np.asarray(live, dtype=bool)
    fixed_mask = np.asarray(fixed, dtype=bool)
    live_area = int(np.count_nonzero(live_mask))
    fixed_area = int(np.count_nonzero(fixed_mask))
    intersection = int(np.count_nonzero(live_mask & fixed_mask))
    union = int(np.count_nonzero(live_mask | fixed_mask))
    denom = live_area + fixed_area
    live_centroid = _centroid(live_mask)
    fixed_centroid = _centroid(fixed_mask)
    dx = math.nan
    dy = math.nan
    dist = math.nan
    if live_centroid is not None and fixed_centroid is not None:
        dx = float(live_centroid[0] - fixed_centroid[0])
        dy = float(live_centroid[1] - fixed_centroid[1])
        dist = float(math.hypot(dx, dy))
    return {
        f"{prefix}_live_area_px": float(live_area),
        f"{prefix}_fixed_area_px": float(fixed_area),
        f"{prefix}_area_ratio": float(live_area / fixed_area) if fixed_area else math.nan,
        f"{prefix}_intersection_px": float(intersection),
        f"{prefix}_union_px": float(union),
        f"{prefix}_iou": float(intersection / union) if union else math.nan,
        f"{prefix}_dice": float((2.0 * intersection) / denom) if denom else math.nan,
        f"{prefix}_centroid_dx_px": dx,
        f"{prefix}_centroid_dy_px": dy,
        f"{prefix}_centroid_shift_px": dist,
    }


def _stable_project_component(
    *,
    mask_data: Any,
    frame_id: int,
    crop_row: dict[str, str],
    transform: Any,
    video: Any,
    stable_width: int,
    stable_height: int,
) -> tuple[np.ndarray | None, str]:
    import cv2

    projected = project_subject_mask_to_crop_frame(
        mask_data,
        frame_id=frame_id,
        crop_row=crop_row,
        video_width=video.width,
        video_height=video.height,
    )
    if not projected.valid or projected.mask is None:
        return None, projected.reason
    stable_mask = cv2.warpAffine(
        projected.mask.astype(np.uint8) * 255,
        transform.crop_to_stable.astype(np.float32),
        (int(stable_width), int(stable_height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return stable_mask > 0, "ok"


def _reason_add(reasons: dict[str, int], reason: str) -> None:
    reasons[str(reason)] = reasons.get(str(reason), 0) + 1


def _value_summary(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _write_plot(
    path: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(path.parent)
    time_s = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)

    def vals(key: str) -> np.ndarray:
        return np.asarray([float(row.get(key, math.nan)) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(time_s, vals("body_iou"), lw=0.8, label="body")
    axes[0].plot(time_s, vals("eye_union_iou"), lw=0.8, label="eye union")
    axes[0].plot(time_s, vals("eye_exclusion_iou"), lw=0.8, label="eye exclusion")
    axes[0].set_ylabel("IoU")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")

    axes[1].plot(time_s, vals("body_centroid_shift_px"), lw=0.8, label="body")
    axes[1].plot(time_s, vals("eye_union_centroid_shift_px"), lw=0.8, label="eye union")
    axes[1].set_ylabel("centroid shift (px)")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_s, vals("body_area_ratio"), lw=0.8, label="body")
    axes[2].plot(time_s, vals("eye_union_area_ratio"), lw=0.8, label="eye union")
    axes[2].axhline(1.0, color="0.3", lw=0.6)
    axes[2].set_ylabel("live / fixed area")
    axes[2].legend(loc="upper right")

    axes[3].plot(time_s, vals("roi_body_coverage_fraction"), lw=0.8, label="ROI covered by live body")
    axes[3].plot(time_s, vals("roi_eye_exclusion_overlap_fraction"), lw=0.8, label="ROI overlapped by live eye exclusion")
    axes[3].plot(time_s, vals("roi_live_usable_fraction"), lw=0.8, label="ROI live usable")
    axes[3].set_ylabel("fraction")
    axes[3].set_xlabel("time in selected clip (s)")
    axes[3].set_ylim(-0.02, 1.02)
    axes[3].legend(loc="lower right")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure framewise live-mask stability in canonical stabilized coordinates.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--mask-npz", type=Path, required=True, help="Consensus mask-relative ROI NPZ.")
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--mask-parent", type=str, default=None, help="Mask parent group, or 'auto'.")
    parser.add_argument("--mask-run", type=str, default=None, help="Mask run name, or 'latest'.")
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/live_mask_stability"),
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
    origin = str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))

    fixed = _load_fixed_masks(args.mask_npz, shape_hw=(stable_height, stable_width))
    roi_mask = np.asarray(fixed["roi_mask"], dtype=bool)
    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels <= 0:
        raise ValueError(f"{args.mask_npz} roi_mask is empty.")

    video = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
    )
    if not selected:
        raise ValueError("No crop rows were selected.")
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

    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    first_crop_index = int(selected[0][0])
    for crop_video_index, crop_row in selected:
        frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
        base_row: dict[str, Any] = {
            "crop_video_frame_index": int(crop_video_index),
            "frame_id": int(frame_id),
            "time_s": float((int(crop_video_index) - first_crop_index) / float(video.fps or 100.0)),
            "valid": 0,
            "reason": "",
        }
        keypoint_row = keypoints.frame_to_row.get(frame_id)
        if keypoint_row is None:
            base_row["reason"] = "missing_keypoint_frame"
            _reason_add(reason_counts, str(base_row["reason"]))
            rows.append(base_row)
            continue
        if not bool(keypoints.valid[keypoint_row]):
            base_row["reason"] = "invalid_keypoints"
            _reason_add(reason_counts, str(base_row["reason"]))
            rows.append(base_row)
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
            base_row["reason"] = transform.reason
            _reason_add(reason_counts, str(base_row["reason"]))
            rows.append(base_row)
            continue

        live_body, body_reason = _stable_project_component(
            mask_data=body_data,
            frame_id=frame_id,
            crop_row=crop_row,
            transform=transform,
            video=video,
            stable_width=stable_width,
            stable_height=stable_height,
        )
        if live_body is None:
            base_row["reason"] = f"body:{body_reason}"
            _reason_add(reason_counts, str(base_row["reason"]))
            rows.append(base_row)
            continue
        live_eye = np.zeros((stable_height, stable_width), dtype=bool)
        eye_component_reasons: dict[str, str] = {}
        for component, data in eye_data.items():
            component_mask, reason = _stable_project_component(
                mask_data=data,
                frame_id=frame_id,
                crop_row=crop_row,
                transform=transform,
                video=video,
                stable_width=stable_width,
                stable_height=stable_height,
            )
            eye_component_reasons[component] = reason
            if component_mask is not None:
                live_eye |= component_mask
        if not np.any(live_eye):
            base_row["reason"] = "empty_eye_union:" + ",".join(f"{name}={reason}" for name, reason in eye_component_reasons.items())
            _reason_add(reason_counts, "empty_eye_union")
            rows.append(base_row)
            continue

        live_eye_exclusion = _dilate(live_eye, int(args.eye_dilate_px))
        live_usable = live_body & ~live_eye_exclusion
        base_row.update(
            {
                "valid": 1,
                "reason": "ok",
                **_mask_metrics("body", live_body, fixed["body_mask"]),
                **_mask_metrics("eye_union", live_eye, fixed["eye_mask"]),
                **_mask_metrics("eye_exclusion", live_eye_exclusion, fixed["eye_exclusion"]),
                "roi_body_coverage_fraction": float(np.count_nonzero(roi_mask & live_body) / roi_pixels),
                "roi_eye_overlap_fraction": float(np.count_nonzero(roi_mask & live_eye) / roi_pixels),
                "roi_eye_exclusion_overlap_fraction": float(np.count_nonzero(roi_mask & live_eye_exclusion) / roi_pixels),
                "roi_live_usable_fraction": float(np.count_nonzero(roi_mask & live_usable) / roi_pixels),
            }
        )
        _reason_add(reason_counts, "ok")
        rows.append(base_row)

    output_prefix = Path(args.output_prefix)
    ensure_output_dir(output_prefix.parent)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".summary.json")
    plot_path = output_prefix.with_suffix(".png")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if int(row.get("valid", 0)) == 1]
    if valid_rows:
        _write_plot(plot_path, rows=valid_rows)

    metric_keys = [
        "body_iou",
        "body_centroid_shift_px",
        "body_area_ratio",
        "eye_union_iou",
        "eye_union_centroid_shift_px",
        "eye_union_area_ratio",
        "eye_exclusion_iou",
        "eye_exclusion_centroid_shift_px",
        "roi_body_coverage_fraction",
        "roi_eye_overlap_fraction",
        "roi_eye_exclusion_overlap_fraction",
        "roi_live_usable_fraction",
    ]
    summary = {
        "config": str(args.config),
        "mask_npz": str(args.mask_npz),
        "crop_video": str(crop_video),
        "zarr_path": str(zarr_path),
        "keypoint_group": keypoint_group,
        "mask_parent": mask_parent,
        "mask_run": mask_run,
        "body_component": str(args.body_component),
        "eye_components": list(eye_data.keys()),
        "eye_dilate_px": int(args.eye_dilate_px),
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "stride": int(args.stride),
        "selected_frames": int(len(selected)),
        "valid_frames": int(len(valid_rows)),
        "reason_counts": reason_counts,
        "roi_mask_pixels": roi_pixels,
        "fixed_mask_pixels": {
            "body": int(np.count_nonzero(fixed["body_mask"])),
            "eye": int(np.count_nonzero(fixed["eye_mask"])),
            "eye_exclusion": int(np.count_nonzero(fixed["eye_exclusion"])),
        },
        "metrics": {
            key: _value_summary([float(row.get(key, math.nan)) for row in valid_rows])
            for key in metric_keys
        },
        "outputs": {
            "csv": str(csv_path),
            "summary_json": str(json_path),
            "plot_png": str(plot_path) if valid_rows else None,
        },
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"plot_png: {plot_path if valid_rows else '<not written>'}")
    print(f"valid_frames: {len(valid_rows)}/{len(selected)}")
    if valid_rows:
        print(f"body_iou_median: {summary['metrics']['body_iou']['median']:.4f}")
        print(f"body_shift_median_px: {summary['metrics']['body_centroid_shift_px']['median']:.4f}")
        print(f"eye_iou_median: {summary['metrics']['eye_union_iou']['median']:.4f}")
        print(f"eye_shift_median_px: {summary['metrics']['eye_union_centroid_shift_px']['median']:.4f}")


if __name__ == "__main__":
    main()
