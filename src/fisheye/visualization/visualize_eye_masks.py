"""Interactive visualizer for eye mask segmentation runs.

Displays ROI crops with eye masks overlaid, using the masks produced
by the `eye_masks` pipeline stage. Handles both traditional left/right masks and
YOLO index-ordered masks, highlighting when refinement has not yet been applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import Button, Slider


def open_zarr(zarr_path: Path) -> zarr.Group:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")
    return zarr.open_group(str(zarr_path), mode="r")


def get_latest_run(root: zarr.Group, run_group: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    runs_name = f"{run_group}_runs"
    if runs_name not in root:
        raise RuntimeError(f"No '{runs_name}' group found in Zarr store.")
    group = root[runs_name]
    latest = group.attrs.get("latest")
    if not latest:
        raise RuntimeError(f"No runs recorded under '{runs_name}'.")
    return latest


def normalize_roi(roi: np.ndarray) -> np.ndarray:
    """Return ROI normalized to 0..1 range for visualization."""
    roi = roi.astype(np.float32)
    min_val = roi.min()
    max_val = roi.max()
    if max_val <= min_val:
        return np.zeros_like(roi, dtype=np.float32)
    return (roi - min_val) / (max_val - min_val)


class EyeMaskViewer:
    def __init__(
        self,
        root: zarr.Group,
        eye_run: str,
        crop_run: str,
        keypoint_run: str,
    ) -> None:
        eye_group = root[f"eye_masks_runs/{eye_run}"]
        crop_group = root[f"crop_runs/{crop_run}"]
        kp_group = root[f"keypoints_runs/{keypoint_run}"]

        self.roi_images = crop_group["roi_images"]
        self.masks = eye_group["masks_roi"]
        self.mask_probs = eye_group["mask_probs_roi"] if "mask_probs_roi" in eye_group else None
        self.ellipse_params = eye_group["ellipse_params"]
        self.ellipse_success = eye_group["ellipse_success"][:]
        self.success_flags = kp_group["detection_success"][:]
        self.feret_major = (
            np.asarray(eye_group["feret_axes_major"][:]) if "feret_axes_major" in eye_group else None
        )
        self.feret_minor = (
            np.asarray(eye_group["feret_axes_minor"][:]) if "feret_axes_minor" in eye_group else None
        )
        self.feret_roundness = (
            np.asarray(eye_group["feret_roundness"][:]) if "feret_roundness" in eye_group else None
        )
        self.total = self.roi_images.shape[0]

        if self.masks.shape[0] != self.total:
            raise ValueError(
                f"Mask count ({self.masks.shape[0]}) does not match ROI count ({self.total})"
            )

        self.channel_count = int(self.masks.shape[1])
        raw_labels = eye_group.attrs.get("eye_labels")
        if isinstance(raw_labels, (list, tuple)) and len(raw_labels) >= self.channel_count:
            self.eye_labels = [str(raw_labels[i]) for i in range(self.channel_count)]
        else:
            default_labels = ["eye_left", "eye_right"]
            self.eye_labels = [
                default_labels[i] if i < len(default_labels) else f"eye_{i}"
                for i in range(self.channel_count)
            ]

        normalized = [label.lower() for label in self.eye_labels]
        refined_template = ["eye_left", "eye_right"]
        self.is_refined = (
            self.channel_count == 2 and normalized == refined_template[: self.channel_count]
        )

        self.display_names = [
            self._friendly_label(label, idx) for idx, label in enumerate(self.eye_labels)
        ]

        if self.is_refined:
            base_rgb = [
                np.array([0.1, 0.4, 0.95], dtype=np.float32),
                np.array([0.95, 0.2, 0.2], dtype=np.float32),
            ]
            base_hex = ["#1a66f3", "#f85151"]
        else:
            base_rgb = [
                np.array([0.2, 0.75, 0.45], dtype=np.float32),
                np.array([0.85, 0.6, 0.15], dtype=np.float32),
            ]
            base_hex = ["#2fcc72", "#d89d1a"]
        self.channel_colors = [base_rgb[i % len(base_rgb)] for i in range(self.channel_count)]
        self.channel_hex = [base_hex[i % len(base_hex)] for i in range(self.channel_count)]
        self.overlay_alpha = 0.45
        self._unrefined_note = (
            None
            if self.is_refined
            else (
                "Channels reflect YOLO index order "
                f"(eye_labels={', '.join(self.eye_labels)}); run refinement to align left/right."
            )
        )

    @staticmethod
    def _friendly_label(label: Optional[str], idx: int) -> str:
        if label is None:
            return f"Eye {idx}"
        lower = str(label).lower()
        if lower in {"eye_left", "left", "left_eye"}:
            return "Left"
        if lower in {"eye_right", "right", "right_eye"}:
            return "Right"
        if lower.startswith("eye_"):
            suffix = lower[4:]
            if suffix.isdigit():
                return f"Eye {int(suffix)}"
            return suffix.replace("_", " ").title()
        return str(label).title()

    @staticmethod
    def _format_measure(value: Optional[float], precision: int = 1) -> str:
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.{precision}f}"

    def _axis_endpoints(
        self, cx: float, cy: float, major: float, minor: float, orientation_deg: float
    ) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        if not (
            np.isfinite(cx)
            and np.isfinite(cy)
            and np.isfinite(major)
            and np.isfinite(minor)
            and np.isfinite(orientation_deg)
        ):
            return {}

        theta = np.deg2rad(orientation_deg)

        # Skimage orientation is measured from the horizontal axis with positive values
        # indicating counter-clockwise rotation, but image Y coordinates increase downward.
        # Follow skimage's own example for drawing axes.
        dx_major = np.cos(theta) * (major / 2.0)
        dy_major = np.sin(theta) * (major / 2.0)

        major_pts = (
            (cx + dx_major, cy - dy_major),
            (cx - dx_major, cy + dy_major),
        )

        theta_minor = theta + np.pi / 2.0
        dx_minor = np.cos(theta_minor) * (minor / 2.0)
        dy_minor = np.sin(theta_minor) * (minor / 2.0)

        minor_pts = (
            (cx + dx_minor, cy - dy_minor),
            (cx - dx_minor, cy + dy_minor),
        )

        return {"major": major_pts, "minor": minor_pts}

    @staticmethod
    def _length_from_segment(segment: np.ndarray) -> Optional[float]:
        if segment is None or segment.shape[0] != 4 or not np.all(np.isfinite(segment)):
            return None
        p1 = segment[:2]
        p2 = segment[2:]
        return float(np.linalg.norm(p1 - p2))

    def make_overlay(
        self,
        idx: int,
    ) -> tuple[
        np.ndarray,
        str,
        List[dict[str, tuple[tuple[float, float], tuple[float, float]]]],
        Optional[List[np.ndarray]],
        np.ndarray,
        List[np.ndarray],
    ]:
        roi = np.asarray(self.roi_images[idx])
        base = normalize_roi(roi)
        overlay = np.dstack([base, base, base])

        success_flag = bool(self.success_flags[idx])
        summary_lines = [
            f"ROI {idx + 1}/{self.total} | keypoints: {'ok' if success_flag else 'fail'}"
        ]

        mask_list: List[np.ndarray] = []
        axes_data: List[dict[str, tuple[tuple[float, float], tuple[float, float]]]] = []

        ellipse_info = np.asarray(self.ellipse_params[idx])

        for ch_idx in range(self.channel_count):
            mask = np.asarray(self.masks[idx, ch_idx])
            mask_list.append(mask)
            mask_bool = mask > 0
            color = self.channel_colors[ch_idx]
            overlay[mask_bool] = (1 - self.overlay_alpha) * overlay[mask_bool] + self.overlay_alpha * color

            area = int(mask.sum())

            success = (
                bool(self.ellipse_success[idx, ch_idx])
                if ch_idx < self.ellipse_success.shape[1]
                else False
            )
            ellipse_row = (
                ellipse_info[ch_idx]
                if ch_idx < ellipse_info.shape[0]
                else np.full(5, np.nan, dtype=np.float32)
            )
            cx, cy, major_len_raw, minor_len_raw, theta = (
                float(ellipse_row[0]),
                float(ellipse_row[1]),
                float(ellipse_row[2]),
                float(ellipse_row[3]),
                float(ellipse_row[4]),
            )

            channel_axes: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
            feret_len = None
            feret_minor_len = None
            feret_round_val = None

            if self.feret_major is not None and ch_idx < self.feret_major.shape[1]:
                major_seg = self.feret_major[idx, ch_idx]
                if np.all(np.isfinite(major_seg)):
                    channel_axes["major"] = (
                        (major_seg[0], major_seg[1]),
                        (major_seg[2], major_seg[3]),
                    )
                    feret_len = self._length_from_segment(major_seg)
            if self.feret_minor is not None and ch_idx < self.feret_minor.shape[1]:
                minor_seg = self.feret_minor[idx, ch_idx]
                if np.all(np.isfinite(minor_seg)):
                    channel_axes["minor"] = (
                        (minor_seg[0], minor_seg[1]),
                        (minor_seg[2], minor_seg[3]),
                    )
                    feret_minor_len = self._length_from_segment(minor_seg)
            if self.feret_roundness is not None and ch_idx < self.feret_roundness.shape[1]:
                feret_round_val = float(self.feret_roundness[idx, ch_idx])

            if success and "major" not in channel_axes:
                channel_axes = self._axis_endpoints(cx, cy, major_len_raw, minor_len_raw, theta)

            axes_data.append(channel_axes)

            display_name = self.display_names[ch_idx]
            major_len = feret_len if feret_len is not None else major_len_raw
            minor_len = feret_minor_len if feret_minor_len is not None else minor_len_raw
            summary_line = (
                f"{display_name}: area={area} success={success} "
                f"major={self._format_measure(major_len)} minor={self._format_measure(minor_len)}"
            )
            if feret_round_val is not None and np.isfinite(feret_round_val):
                summary_line += f" round={feret_round_val:.2f}"
            summary_lines.append(summary_line)

        if self._unrefined_note:
            summary_lines.append(self._unrefined_note)

        summary = "\n".join(summary_lines)
        prob_maps: Optional[List[np.ndarray]] = None
        if self.mask_probs is not None:
            prob_maps = [
                np.asarray(self.mask_probs[idx, ch_idx], dtype=np.float32)
                for ch_idx in range(self.channel_count)
            ]
        return overlay, summary, axes_data, prob_maps, base, mask_list


def create_viewer(zarr_path: Path, eye_run: Optional[str], crop_run: Optional[str], keypoint_run: Optional[str]) -> None:
    root = open_zarr(zarr_path)
    eye_run = get_latest_run(root, "eye_masks", eye_run)
    crop_run = get_latest_run(root, "crop", crop_run)
    keypoint_run = get_latest_run(root, "keypoints", keypoint_run)

    viewer = EyeMaskViewer(root, eye_run, crop_run, keypoint_run)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.15], width_ratios=[1, 1, 1])

    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_overlay.set_axis_off()

    ax_raw = fig.add_subplot(gs[0, 1])
    ax_raw.set_title("Raw ROI")
    ax_raw.set_axis_off()

    prob_panel = gs[0, 2]

    slider_ax = fig.add_subplot(gs[1, :])

    overlay, summary, axes, probs, base_roi, bin_masks = viewer.make_overlay(0)
    image_artist = ax_overlay.imshow(overlay, interpolation="nearest")
    info_text = ax_overlay.text(
        0.02,
        0.02,
        summary,
        color="white",
        fontsize=9,
        transform=ax_overlay.transAxes,
        verticalalignment="bottom",
        bbox=dict(facecolor="black", alpha=0.4, pad=4),
    )

    raw_artist = ax_raw.imshow(base_roi, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    line_major: List = []
    line_minor: List = []
    for ch_idx in range(viewer.channel_count):
        color = viewer.channel_hex[ch_idx % len(viewer.channel_hex)]
        axis_entry = axes[ch_idx] if ch_idx < len(axes) else {}
        major_pts = axis_entry.get("major", ((np.nan, np.nan), (np.nan, np.nan)))
        minor_pts = axis_entry.get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
        (major_line,) = ax_overlay.plot(
            [major_pts[0][0], major_pts[1][0]],
            [major_pts[0][1], major_pts[1][1]],
            color=color,
            linewidth=1.8,
        )
        (minor_line,) = ax_overlay.plot(
            [minor_pts[0][0], minor_pts[1][0]],
            [minor_pts[0][1], minor_pts[1][1]],
            color=color,
            linewidth=1.2,
            linestyle="--",
        )
        line_major.append(major_line)
        line_minor.append(minor_line)

    slider = Slider(
        ax=slider_ax,
        label="ROI Index",
        valmin=0,
        valmax=viewer.total - 1,
        valinit=0,
        valfmt="%0.0f",
    )

    prob_images: List = []
    bin_images: List = []
    prob_axes_list = []
    prob_titles = [f"{name} Probability" for name in viewer.display_names]
    bin_titles = [f"{name} Binary Mask" for name in viewer.display_names]

    sub_gs = prob_panel.subgridspec(2, max(1, viewer.channel_count), hspace=0.25, wspace=0.2)
    if probs is None:
        prob_maps_init = [np.zeros_like(base_roi, dtype=np.float32) for _ in range(viewer.channel_count)]
    else:
        prob_maps_init = probs

    for col, title in enumerate(prob_titles):
        prob_map = prob_maps_init[col] if col < len(prob_maps_init) else np.zeros_like(base_roi, dtype=np.float32)
        ax_prob = fig.add_subplot(sub_gs[0, col])
        ax_prob.set_title(title, fontsize=9)
        ax_prob.set_axis_off()
        im = ax_prob.imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        prob_images.append(im)
        prob_axes_list.append(ax_prob)
    if prob_axes_list:
        fig.colorbar(prob_images[0], ax=prob_axes_list, fraction=0.046, pad=0.04)

    for col, title in enumerate(bin_titles):
        bin_map = bin_masks[col] if col < len(bin_masks) else np.zeros_like(base_roi)
        ax_bin = fig.add_subplot(sub_gs[1, col])
        ax_bin.set_title(title, fontsize=9)
        ax_bin.set_axis_off()
        im = ax_bin.imshow(bin_map, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        bin_images.append(im)

    def update_from_slider(val: float) -> None:
        idx = int(round(val))
        idx = max(0, min(viewer.total - 1, idx))
        overlay, info, axes, prob_maps, base_img, mask_pair = viewer.make_overlay(idx)
        image_artist.set_data(overlay)
        info_text.set_text(info)
        raw_artist.set_data(base_img)
        for ch_idx in range(len(line_major)):
            axis_entry = axes[ch_idx] if ch_idx < len(axes) else {}
            major_pts = axis_entry.get("major", ((np.nan, np.nan), (np.nan, np.nan)))
            minor_pts = axis_entry.get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
            line_major[ch_idx].set_data(
                [major_pts[0][0], major_pts[1][0]],
                [major_pts[0][1], major_pts[1][1]],
            )
            line_minor[ch_idx].set_data(
                [minor_pts[0][0], minor_pts[1][0]],
                [minor_pts[0][1], minor_pts[1][1]],
            )
        if prob_images:
            if prob_maps is None:
                prob_seq = [np.zeros_like(im.get_array()) for im in prob_images]
            else:
                prob_seq = [
                    prob_maps[i] if i < len(prob_maps) else np.zeros_like(prob_images[i].get_array())
                    for i in range(len(prob_images))
                ]
            for im, prob_map in zip(prob_images, prob_seq):
                im.set_data(prob_map)
        for idx_img, im in enumerate(bin_images):
            if idx_img < len(mask_pair):
                mask_map = mask_pair[idx_img]
            else:
                mask_map = np.zeros_like(bin_images[0].get_array()) if bin_images else np.zeros_like(base_img)
            im.set_data(mask_map)
        fig.canvas.draw_idle()

    slider.on_changed(update_from_slider)

    def step(delta: int) -> None:
        idx = int(round(slider.val)) + delta
        idx = max(0, min(viewer.total - 1, idx))
        slider.set_val(idx)

    ax_prev = fig.add_axes([0.2, 0.015, 0.1, 0.04])
    ax_next = fig.add_axes([0.75, 0.015, 0.1, 0.04])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    btn_prev.on_clicked(lambda _: step(-1))
    btn_next.on_clicked(lambda _: step(1))

    def on_key(event) -> None:
        if event.key in {"left", "j"}:
            step(-1)
        elif event.key in {"right", "l"}:
            step(1)
        elif event.key in {"down"}:
            step(-5)
        elif event.key in {"up"}:
            step(5)

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.canvas.manager.set_window_title(
        f"Eye Mask Viewer | eye_run={eye_run}, crop_run={crop_run}, keypoint_run={keypoint_run}"
    )
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize eye mask segmentation results.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr store")
    parser.add_argument("--eye-run", help="Specific eye mask run name")
    parser.add_argument("--crop-run", help="Specific crop run name")
    parser.add_argument("--keypoint-run", help="Specific keypoint run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_viewer(args.zarr_path, args.eye_run, args.crop_run, args.keypoint_run)


if __name__ == "__main__":
    main()
