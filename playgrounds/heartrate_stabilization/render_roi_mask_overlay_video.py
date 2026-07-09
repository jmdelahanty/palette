from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, resolve_roi_rect, roi_rect_corners
from map_pixel_band_contributions import _draw_polygon, _video_shape


def _read_status_csv(path: Path | None) -> dict[int, bool]:
    if path is None:
        return {}
    status: dict[int, bool] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "output_frame_index" not in (reader.fieldnames or []):
            raise ValueError(f"status CSV lacks output_frame_index: {path}")
        for row in reader:
            status[int(row["output_frame_index"])] = str(row.get("valid", "1")).strip() not in {"", "0", "false", "False"}
    return status


def _load_masks(path: Path, *, shape_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if "roi_mask" not in data:
            raise ValueError(f"{path} lacks roi_mask")
        masks = {
            "roi_mask": np.asarray(data["roi_mask"], dtype=bool),
            "diagnostic_rect_mask": np.asarray(data["diagnostic_rect_mask"], dtype=bool)
            if "diagnostic_rect_mask" in data
            else np.asarray(data["roi_mask"], dtype=bool),
            "body_mask": np.asarray(data["body_mask"], dtype=bool) if "body_mask" in data else np.zeros(shape_hw, bool),
            "eye_mask": np.asarray(data["eye_mask"], dtype=bool) if "eye_mask" in data else np.zeros(shape_hw, bool),
            "eye_exclusion": np.asarray(data["eye_exclusion"], dtype=bool) if "eye_exclusion" in data else np.zeros(shape_hw, bool),
        }
        if "bbox_xyxy" in data:
            masks["bbox_xyxy"] = np.asarray(data["bbox_xyxy"], dtype=np.int32).reshape(-1)
        else:
            yy, xx = np.nonzero(masks["roi_mask"])
            if xx.size == 0:
                raise ValueError(f"{path} roi_mask is empty")
            masks["bbox_xyxy"] = np.asarray([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=np.int32)
    for key, value in masks.items():
        if key == "bbox_xyxy":
            continue
        if tuple(value.shape) != tuple(shape_hw):
            raise ValueError(f"{path} {key} shape {value.shape} does not match video shape {shape_hw}")
    return masks


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
    scale: float = 0.45,
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


def _overlay_geometry(
    frame: np.ndarray,
    *,
    roi_polygon: np.ndarray,
    masks: dict[str, np.ndarray],
    frame_index: int,
    valid: bool,
) -> np.ndarray:
    out = frame[:, :, :3].copy()
    out = _blend_mask(out, masks["body_mask"], color=(50, 90, 50), alpha=0.12)
    out = _blend_mask(out, masks["eye_exclusion"], color=(255, 80, 20), alpha=0.20)
    out = _blend_mask(out, masks["roi_mask"], color=(0, 255, 255), alpha=0.55)
    out = _draw_contours(out, masks["body_mask"], color=(70, 150, 70), thickness=1)
    out = _draw_contours(out, masks["eye_mask"], color=(255, 80, 20), thickness=1)
    out = _draw_contours(out, masks["diagnostic_rect_mask"], color=(0, 255, 255), thickness=1)
    out = _draw_contours(out, masks["roi_mask"], color=(0, 255, 255), thickness=1)
    out = _draw_polygon(out, roi_polygon, color=(255, 255, 255))
    label = "valid" if valid else "invalid"
    _draw_text(out, f"frame {frame_index}  {label}", origin=(8, 20), scale=0.45, color=(255, 255, 255))
    return out


def _crop_zoom(
    image: np.ndarray,
    *,
    bbox_xyxy: np.ndarray,
    pad_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(value) for value in np.asarray(bbox_xyxy).reshape(-1).tolist()]
    x0 = max(0, x0 - int(pad_px))
    y0 = max(0, y0 - int(pad_px))
    x1 = min(image.shape[1], x1 + int(pad_px))
    y1 = min(image.shape[0], y1 + int(pad_px))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Zoom crop is empty: {(x0, y0, x1, y1)}")
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _resize_panel(image: np.ndarray, *, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(image[:, :, :3], (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render a geometry-only ROI/mask overlay video on a stabilized clip.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to render.")
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--zoom-pad-px", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_mask_overlay_video.mp4"),
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    shape_hw = _video_shape(args.video)
    masks = _load_masks(args.mask_npz, shape_hw=shape_hw)
    status = _read_status_csv(args.status_csv)

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

    frame_indices = np.arange(
        max(0, int(args.frame_start)),
        max(0, int(args.frame_start)) + max(0, int(args.frame_count)),
        max(1, int(args.stride)),
        dtype=np.int64,
    )
    if frame_indices.size == 0:
        capture.release()
        writer.release()
        raise ValueError("No frames selected.")

    rendered = 0
    read_failures = 0
    invalid_status = 0
    preview: np.ndarray | None = None
    next_expected: int | None = None
    try:
        for frame_index in frame_indices.tolist():
            if next_expected is None or int(frame_index) != int(next_expected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            next_expected = int(frame_index) + 1
            if not ok or frame is None:
                read_failures += 1
                frame = np.zeros((*shape_hw, 3), dtype=np.uint8)
            valid = status.get(int(frame_index), True)
            if not valid:
                invalid_status += 1
            overlay = _overlay_geometry(
                frame,
                roi_polygon=roi_polygon,
                masks=masks,
                frame_index=int(frame_index),
                valid=bool(valid),
            )
            zoom, (x0, y0, _x1, _y1) = _crop_zoom(overlay, bbox_xyxy=masks["bbox_xyxy"], pad_px=int(args.zoom_pad_px))
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
    finally:
        capture.release()
        writer.release()

    preview_path = output.with_suffix(".preview.png")
    if preview is not None:
        cv2.imwrite(str(preview_path), preview)
    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json),
        "mask_npz": str(args.mask_npz),
        "output_video": str(output),
        "preview_png": str(preview_path) if preview is not None else None,
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "stride": int(args.stride),
        "playback_fps": float(args.playback_fps),
        "selected_output_frames": int(frame_indices.size),
        "rendered_frames": int(rendered),
        "read_failures": int(read_failures),
        "invalid_status_frames": int(invalid_status),
        "roi_rect_stable_xywh": [float(value) for value in roi_rect],
        "roi_mask_pixels": int(np.count_nonzero(masks["roi_mask"])),
        "diagnostic_rect_pixels": int(np.count_nonzero(masks["diagnostic_rect_mask"])),
        "body_mask_pixels": int(np.count_nonzero(masks["body_mask"])),
        "eye_mask_pixels": int(np.count_nonzero(masks["eye_mask"])),
        "bbox_xyxy": [int(value) for value in masks["bbox_xyxy"].tolist()],
    }
    summary_path = args.summary_json or output.with_suffix(".summary.json")
    with Path(summary_path).open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"output_video: {output}")
    print(f"preview_png: {preview_path}")
    print(f"summary_json: {summary_path}")
    print(f"rendered_frames: {rendered}")
    print(f"roi_mask_pixels: {summary['roi_mask_pixels']}")


if __name__ == "__main__":
    main()
