"""Interactive preview of background subtraction on crop ROIs for eye-mask analysis.

This tool helps evaluate whether ``background - crop`` would improve traditional
eye-mask segmentation. It displays per-ROI panels for:

- crop ROI
- matched background ROI
- positive subtraction (background - crop, clipped at 0)
- thresholded subtraction mask
- eye-centered raw/diff patches (when keypoints are available)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import zarr
from rich.console import Console
from ..shared.roi_background import (
    BackgroundSource,
    KeypointSource,
    _extract_background_roi_ds,
    _extract_background_roi_full,
    _extract_patch,
    _prepare_panel,
    _resolve_background_source,
    _resolve_keypoint_source,
    _resolve_run_name,
    _to_bgr,
)


def _draw_keypoints(image: np.ndarray, keypoints_roi: Optional[np.ndarray], keypoint_success: bool) -> np.ndarray:
    out = _to_bgr(image)
    if keypoints_roi is None or keypoints_roi.shape[0] < 3:
        return out
    if not keypoint_success:
        return out

    colors = [
        (255, 255, 0),  # bladder
        (80, 255, 80),  # left
        (80, 80, 255),  # right
    ]
    for idx, color in enumerate(colors):
        pt = keypoints_roi[idx]
        if not np.all(np.isfinite(pt)):
            continue
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=11, thickness=1)
    return out


def _render_roi_preview(
    *,
    roi_image: np.ndarray,
    background_roi: np.ndarray,
    threshold: int,
    panel_size: int,
    eye_padding: int,
    keypoints_roi: Optional[np.ndarray],
    keypoint_success: bool,
) -> np.ndarray:
    diff = np.clip(background_roi.astype(np.int16) - roi_image.astype(np.int16), 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold_mask = (diff >= int(np.clip(threshold, 0, 255))).astype(np.uint8) * 255

    roi_kp = _draw_keypoints(roi_image, keypoints_roi, keypoint_success)
    diff_kp = _draw_keypoints(diff, keypoints_roi, keypoint_success)

    top = [
        _prepare_panel(roi_kp, "Crop ROI", panel_size),
        _prepare_panel(background_roi, "Background ROI", panel_size),
        _prepare_panel(diff_kp, "BG - Crop", panel_size),
        _prepare_panel(threshold_mask, f"Diff >= {threshold}", panel_size, nearest=True),
    ]

    left_center = keypoints_roi[1] if keypoints_roi is not None and keypoints_roi.shape[0] > 1 else None
    right_center = keypoints_roi[2] if keypoints_roi is not None and keypoints_roi.shape[0] > 2 else None

    if keypoint_success and left_center is not None and np.all(np.isfinite(left_center)):
        left_raw = _extract_patch(roi_image, left_center, eye_padding)
        left_diff = _extract_patch(diff, left_center, eye_padding)
    else:
        size = eye_padding * 2 + 1
        left_raw = np.zeros((size, size), dtype=np.uint8)
        left_diff = np.zeros((size, size), dtype=np.uint8)

    if keypoint_success and right_center is not None and np.all(np.isfinite(right_center)):
        right_raw = _extract_patch(roi_image, right_center, eye_padding)
        right_diff = _extract_patch(diff, right_center, eye_padding)
    else:
        size = eye_padding * 2 + 1
        right_raw = np.zeros((size, size), dtype=np.uint8)
        right_diff = np.zeros((size, size), dtype=np.uint8)

    bottom = [
        _prepare_panel(left_raw, "Left Eye Raw", panel_size, nearest=True),
        _prepare_panel(left_diff, "Left Eye BG-Diff", panel_size, nearest=True),
        _prepare_panel(right_raw, "Right Eye Raw", panel_size, nearest=True),
        _prepare_panel(right_diff, "Right Eye BG-Diff", panel_size, nearest=True),
    ]

    row_top = cv2.hconcat(top)
    row_bottom = cv2.hconcat(bottom)
    grid = cv2.vconcat([row_top, row_bottom])

    footer = np.zeros((88, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        footer,
        f"Manual threshold={int(threshold)} | Otsu threshold={int(otsu_threshold)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Controls: n/p next-prev, j/k +/-10 ROI, ]/[ threshold +/-1, s save, q/ESC quit",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([grid, footer])


def _save_panel(save_dir: Path, roi_idx: int, threshold: int, panel: np.ndarray) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"eye_bgsub_roi_{roi_idx:05d}_thr_{threshold:03d}.png"
    cv2.imwrite(str(out), panel)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--crop-run", help="Crop run name (default: latest).")
    parser.add_argument("--background-run", help="Background run name (default: latest).")
    parser.add_argument(
        "--keypoint-run",
        help="Optional keypoint run name from refined_keypoints_runs/keypoints_runs (default: latest refined, else latest raw).",
    )
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index (default: 0).")
    parser.add_argument("--threshold", type=int, default=50, help="Initial diff threshold in [0,255] (default: 50).")
    parser.add_argument("--eye-padding", type=int, default=12, help="Half-width for eye patch crops (default: 12).")
    parser.add_argument("--panel-size", type=int, default=280, help="Size of each panel in pixels (default: 280).")
    parser.add_argument("--window-name", default="Eye BG-Sub Preview", help="OpenCV window name.")
    parser.add_argument("--save-dir", type=Path, help="Optional directory for saving current panel via 's' key.")
    return parser


def run(args: argparse.Namespace) -> int:
    console = Console()
    root = zarr.open_group(str(args.zarr_path), mode="r", use_consolidated=False)

    crop_run = _resolve_run_name(root, "crop_runs", args.crop_run)
    background_run = _resolve_run_name(root, "background_runs", args.background_run)
    crop_group = root[f"crop_runs/{crop_run}"]
    bg_source = _resolve_background_source(root, background_run)
    bg_group = root[f"background_runs/{background_run}"]

    if "roi_images" not in crop_group:
        raise ValueError(f"crop_runs/{crop_run} missing roi_images.")
    if "roi_coordinates_full" not in crop_group:
        raise ValueError(f"crop_runs/{crop_run} missing roi_coordinates_full.")

    roi_images = crop_group["roi_images"]
    roi_coords = crop_group["roi_coordinates_full"]
    total_rois = int(roi_images.shape[0])
    if total_rois <= 0:
        raise ValueError(f"crop_runs/{crop_run}/roi_images is empty.")

    keypoint_source = _resolve_keypoint_source(root, args.keypoint_run)
    keypoints_ds = None
    success_ds = None
    if keypoint_source is not None:
        kp_group = root[f"{keypoint_source.group_name}/{keypoint_source.run_name}"]
        keypoints_ds = kp_group.get("keypoints_roi")
        if keypoints_ds is None:
            keypoint_source = None
        elif keypoint_source.success_name:
            success_ds = kp_group.get(keypoint_source.success_name)
            if success_ds is not None and int(success_ds.shape[0]) != total_rois:
                raise ValueError(
                    f"{keypoint_source.group_name}/{keypoint_source.run_name}/{keypoint_source.success_name} "
                    f"has {success_ds.shape[0]} rows, expected {total_rois}."
                )
        if keypoints_ds is not None and int(keypoints_ds.shape[0]) != total_rois:
            raise ValueError(
                f"{keypoint_source.group_name}/{keypoint_source.run_name}/keypoints_roi has "
                f"{keypoints_ds.shape[0]} rows, expected {total_rois}."
            )

    bg_array = bg_group[bg_source.array_name]
    roi_idx = int(np.clip(args.roi_index, 0, max(0, total_rois - 1)))
    threshold = int(np.clip(args.threshold, 0, 255))
    eye_padding = max(1, int(args.eye_padding))
    panel_size = max(96, int(args.panel_size))
    save_dir = args.save_dir.expanduser() if args.save_dir else None

    console.print(f"[bold]Zarr:[/bold] {args.zarr_path}")
    console.print(
        f"[bold]Runs:[/bold] crop={crop_run} background={background_run} ({bg_source.array_name})"
    )
    if keypoint_source is None:
        console.print("[yellow]No keypoint run available for eye patch overlays.[/yellow]")
    else:
        success_name = keypoint_source.success_name or "none"
        console.print(
            f"[bold]Keypoints:[/bold] {keypoint_source.group_name}/{keypoint_source.run_name} "
            f"(success={success_name})"
        )

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    try:
        while True:
            roi_img = np.asarray(roi_images[roi_idx], dtype=np.uint8)
            coord_xy = np.asarray(roi_coords[roi_idx], dtype=np.float64)
            roi_shape = (int(roi_img.shape[0]), int(roi_img.shape[1]))

            if bg_source.array_name == "background_full":
                bg_roi = _extract_background_roi_full(bg_array, coord_xy, roi_shape)
            else:
                assert bg_source.full_shape is not None
                bg_roi = _extract_background_roi_ds(
                    bg_array,
                    coord_xy,
                    roi_shape,
                    bg_source.full_shape,
                )
            bg_roi = np.asarray(bg_roi, dtype=np.uint8)

            kp_row = None
            kp_success = True
            if keypoints_ds is not None:
                kp_row = np.asarray(keypoints_ds[roi_idx], dtype=np.float32)
            if success_ds is not None:
                kp_success = bool(success_ds[roi_idx])

            panel = _render_roi_preview(
                roi_image=roi_img,
                background_roi=bg_roi,
                threshold=threshold,
                panel_size=panel_size,
                eye_padding=eye_padding,
                keypoints_roi=kp_row,
                keypoint_success=kp_success,
            )

            status = (
                f"ROI {roi_idx + 1}/{total_rois} | "
                f"coord=({int(round(float(coord_xy[0])))} , {int(round(float(coord_xy[1])))}) | "
                f"threshold={threshold} | kp_success={'yes' if kp_success else 'no'}"
            )
            if hasattr(cv2, "setWindowTitle"):
                cv2.setWindowTitle(args.window_name, status)
            cv2.imshow(args.window_name, panel)
            key = cv2.waitKeyEx(0)

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                roi_idx = min(total_rois - 1, roi_idx + 1)
                continue
            if key in (ord("p"), ord("P")):
                roi_idx = max(0, roi_idx - 1)
                continue
            if key == ord("j"):
                roi_idx = min(total_rois - 1, roi_idx + 10)
                continue
            if key == ord("k"):
                roi_idx = max(0, roi_idx - 10)
                continue
            if key == ord("]"):
                threshold = min(255, threshold + 1)
                continue
            if key == ord("["):
                threshold = max(0, threshold - 1)
                continue
            if key in (ord("s"), ord("S")):
                if save_dir is None:
                    console.print("[yellow]Set --save-dir to enable panel saves.[/yellow]")
                else:
                    out = _save_panel(save_dir, roi_idx, threshold, panel)
                    console.print(f"[green]Saved:[/green] {out}")
                continue
    finally:
        cv2.destroyWindow(args.window_name)

    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
