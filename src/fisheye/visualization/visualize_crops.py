#!/usr/bin/env python3
"""
Quick viewer for crop runs stored in Palette Zarr archives.

This tool loads a crop run (defaulting to the latest) and presents the ROI crops
with an interactive slider and keyboard navigation.  It is intentionally
lightweight so it can be used during annotation / QA to spot issues in the crop
stage without bringing keypoints or masks into the picture.

Note: The viewer locks contrast to 0–255 for grayscale crops so intensity
comparisons are consistent across frames (avoids per-frame auto-scaling).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import Slider

from ..shared.crop_signature import build_crop_signature as _build_shared_crop_signature


def get_latest_run(root: zarr.Group, group_name: str, explicit: Optional[str]) -> str:
    """
    Resolve a run name inside ``<group_name>_runs``.

    Mirrors the helper used in other visualization modules so behaviour stays
    consistent.
    """

    group_path = f"{group_name}_runs"
    if group_path not in root:
        raise ValueError(f"No '{group_path}' group found in archive.")

    run_root = root[group_path]
    if explicit:
        if explicit not in run_root:
            raise ValueError(f"Run '{explicit}' not found in {group_path}.")
        return explicit

    latest = run_root.attrs.get("latest")
    if latest:
        return latest

    runs = sorted(run_root.group_keys())
    if not runs:
        raise ValueError(f"No runs stored under {group_path}.")
    return runs[-1]


def _prepare_image(raw: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """
    Convert a crop image to something Matplotlib can display.

    Returns the possibly transformed array plus the cmap to use (``None`` for RGB).
    """

    img = np.asarray(raw)
    if img.ndim == 2:
        return img, "gray"
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0], "gray"
        if img.shape[2] >= 3:
            return img[:, :, :3], None
    raise ValueError(f"Unsupported crop image shape {img.shape!r}")


def _format_info(
    idx: int,
    frame: int,
    coords_full: Optional[np.ndarray],
    review_label: Optional[str] = None,
    write_intended_use: Optional[str] = None,
) -> str:
    lines = [f"Crop index: {idx}", f"Frame index: {frame}"]
    if coords_full is not None and coords_full.size >= 4:
        x0, y0, x1, y1 = coords_full
        width = x1 - x0
        height = y1 - y0
        lines.append(f"Full-frame box: x0={x0:.1f}, y0={y0:.1f}, w={width:.1f}, h={height:.1f}")
    if review_label:
        lines.append(f"Review: {review_label}")
    if write_intended_use:
        lines.append(f"Write intended_use: {write_intended_use}")
    return "\n".join(lines)


def _cycle_intended_use(current: str) -> str:
    values = ("training", "full_recording")
    normalized = str(current or "").strip().lower()
    if normalized not in values:
        return values[0]
    idx = values.index(normalized)
    return values[(idx + 1) % len(values)]


def _build_crop_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    return dict(_build_shared_crop_signature(attrs))


def _format_review_status(status: Optional[Dict[str, object]]) -> Optional[str]:
    if not status:
        return None
    state = str(status.get("state", "")).strip()
    method = str(status.get("method", "")).strip()
    intended_use = str(status.get("intended_use", "")).strip()
    parts = []
    if method:
        parts.append(method)
    if intended_use:
        parts.append(intended_use)
    if parts:
        return f"{state or 'review'} ({', '.join(parts)})"
    return state or "review"


def _apply_crop_review_status(
    zarr_path: Path,
    crop_run: str,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> Dict[str, object]:
    root = zarr.open(str(zarr_path), mode="a")
    if "crop_runs" not in root:
        raise RuntimeError("No crop_runs found in archive.")
    crop_parent = root["crop_runs"]
    if crop_run not in crop_parent:
        raise RuntimeError(f"Crop run '{crop_run}' not found.")
    crop_group = crop_parent[crop_run]

    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes

    crop_group.attrs["crop_review_status"] = payload

    signature = crop_group.attrs.get("crop_signature")
    if not isinstance(signature, dict):
        signature = _build_crop_signature(crop_group.attrs)
        crop_group.attrs["crop_signature"] = signature
    crop_group.attrs["crop_review_signature"] = signature
    crop_parent.attrs["crop_review_status_latest"] = crop_run
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize ROI crops stored in crop_runs.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--crop-run",
        type=str,
        help="Specific crop run to load (default: latest).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the window (useful for tests).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status messages.",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when pressing 'a' (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = zarr.open(str(args.zarr_path), mode="r")
    crop_run = get_latest_run(root, "crop", args.crop_run)
    crop_group = root[f"crop_runs/{crop_run}"]
    review_status = crop_group.attrs.get("crop_review_status")
    review_label = _format_review_status(review_status if isinstance(review_status, dict) else None)
    current_intended_use = str(args.review_intended_use)

    roi_images = crop_group["roi_images"]
    frame_indices = crop_group["frame_indices"][:]
    coords_full = crop_group.get("roi_coordinates_full")

    total = roi_images.shape[0]
    if total == 0:
        raise ValueError(f"Crop run '{crop_run}' has no crops to display.")

    if not args.quiet:
        print(f"Loaded crop run '{crop_run}' ({total} crops).")

    first_img, first_cmap = _prepare_image(roi_images[0])
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.22)
    ax.set_axis_off()
    image_artist = ax.imshow(first_img, cmap=first_cmap, vmin=0, vmax=255)

    info_ax = fig.add_axes([0.02, 0.02, 0.4, 0.16])
    info_ax.set_axis_off()
    info_text = info_ax.text(
        0.0,
        0.98,
        _format_info(
            0,
            int(frame_indices[0]),
            coords_full[0] if coords_full is not None else None,
            review_label,
            current_intended_use,
        ),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    slider_ax = fig.add_axes([0.15, 0.05, 0.75, 0.03])
    slider = Slider(slider_ax, "Crop index", 0, total - 1, valinit=0, valstep=1)

    def _update_title() -> None:
        title = f"Crop viewer - run {crop_run} | target_use: {current_intended_use}"
        if review_label:
            title = f"{title} | review: {review_label}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

    def update(idx_float: float) -> None:
        idx = int(idx_float)
        img, cmap = _prepare_image(roi_images[idx])
        image_artist.set_data(img)
        if cmap:
            image_artist.set_cmap(cmap)
        image_artist.set_clim(0, 255)
        info_text.set_text(
            _format_info(
                idx,
                int(frame_indices[idx]),
                coords_full[idx] if coords_full is not None else None,
                review_label,
                current_intended_use,
            )
        )
        _update_title()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event) -> None:
        nonlocal review_label, current_intended_use
        def apply_state(state: str) -> None:
            nonlocal review_label
            reviewer = args.reviewer or os.environ.get("USER")
            payload = _apply_crop_review_status(
                args.zarr_path,
                crop_run,
                state=state,
                method=args.review_method,
                intended_use=current_intended_use,
                reviewer=reviewer,
                notes=args.review_notes,
            )
            review_label = _format_review_status(payload)
            if not args.quiet:
                print(f"✓ Crop review set: {review_label}")
            update(slider.val)

        if event.key in {"left", "down", "pageup"}:
            slider.set_val(max(slider.val - 1, 0))
        elif event.key in {"right", "up", "pagedown"}:
            slider.set_val(min(slider.val + 1, total - 1))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(total - 1)
        elif event.key == "a":
            try:
                apply_state(args.review_state)
            except Exception as exc:
                if not args.quiet:
                    print(f"⚠️ Failed to set crop review: {exc}")
        elif event.key == "r":
            try:
                apply_state("rejected")
            except Exception as exc:
                if not args.quiet:
                    print(f"⚠️ Failed to set crop review: {exc}")
        elif event.key == "n":
            try:
                apply_state("needs_review")
            except Exception as exc:
                if not args.quiet:
                    print(f"⚠️ Failed to set crop review: {exc}")
        elif event.key == "p":
            try:
                apply_state("pending")
            except Exception as exc:
                if not args.quiet:
                    print(f"⚠️ Failed to set crop review: {exc}")
        elif event.key == "u":
            current_intended_use = _cycle_intended_use(current_intended_use)
            if not args.quiet:
                print(f"↻ Review intended_use set to: {current_intended_use}")
            update(slider.val)

    fig.canvas.mpl_connect("key_press_event", on_key)
    _update_title()

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
