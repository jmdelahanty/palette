"""Interactive visualizer for eye mask segmentation runs.

Displays ROI crops with left/right eye masks overlaid, using the masks produced
by the `eye_masks` pipeline stage. Useful for spot-checking segmentation quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

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

    def make_overlay(self, idx: int) -> tuple[
        np.ndarray,
        str,
        dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]],
    ]:
        roi = np.asarray(self.roi_images[idx])
        mask_left = np.asarray(self.masks[idx, 0])
        mask_right = np.asarray(self.masks[idx, 1])
        success_left = bool(self.ellipse_success[idx, 0])
        success_right = bool(self.ellipse_success[idx, 1])

        base = normalize_roi(roi)
        overlay = np.dstack([base, base, base])

        alpha = 0.45
        blue = np.array([0.1, 0.4, 0.95], dtype=np.float32)
        red = np.array([0.95, 0.2, 0.2], dtype=np.float32)

        left_mask = mask_left > 0
        right_mask = mask_right > 0

        overlay[left_mask] = (1 - alpha) * overlay[left_mask] + alpha * blue
        overlay[right_mask] = (1 - alpha) * overlay[right_mask] + alpha * red

        left_area = int(mask_left.sum())
        right_area = int(mask_right.sum())

        ellipse_info = self.ellipse_params[idx]
        left_major, left_minor = ellipse_info[0, 2:4]
        right_major, right_minor = ellipse_info[1, 2:4]
        left_cx, left_cy, left_theta = ellipse_info[0, 0], ellipse_info[0, 1], ellipse_info[0, 4]
        right_cx, right_cy, right_theta = ellipse_info[1, 0], ellipse_info[1, 1], ellipse_info[1, 4]

        axes_data: dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]] = {
            "left": {},
            "right": {},
        }

        feret_lengths = {"left": None, "right": None}
        feret_minor_lengths = {"left": None, "right": None}
        feret_roundness = {"left": None, "right": None}

        for side_idx, side in enumerate(("left", "right")):
            axes_data[side] = {}
            if self.feret_major is not None:
                major_seg = self.feret_major[idx, side_idx]
                if np.all(np.isfinite(major_seg)):
                    axes_data[side]["major"] = (
                        (major_seg[0], major_seg[1]),
                        (major_seg[2], major_seg[3]),
                    )
                    feret_lengths[side] = self._length_from_segment(major_seg)
            if self.feret_minor is not None:
                minor_seg = self.feret_minor[idx, side_idx]
                if np.all(np.isfinite(minor_seg)):
                    axes_data[side]["minor"] = (
                        (minor_seg[0], minor_seg[1]),
                        (minor_seg[2], minor_seg[3]),
                    )
                    feret_minor_lengths[side] = self._length_from_segment(minor_seg)
            if self.feret_roundness is not None:
                feret_roundness[side] = float(self.feret_roundness[idx, side_idx])

        if success_left and "major" not in axes_data["left"]:
            axes_data["left"] = self._axis_endpoints(left_cx, left_cy, left_major, left_minor, left_theta)
        if success_right and "major" not in axes_data["right"]:
            axes_data["right"] = self._axis_endpoints(right_cx, right_cy, right_major, right_minor, right_theta)

        success_flag = bool(self.success_flags[idx])
        left_major_len = feret_lengths["left"] or left_major
        left_minor_len = feret_minor_lengths["left"] or left_minor
        right_major_len = feret_lengths["right"] or right_major
        right_minor_len = feret_minor_lengths["right"] or right_minor
        left_round = feret_roundness["left"]
        right_round = feret_roundness["right"]

        summary_lines = [
            f"ROI {idx + 1}/{self.total} | keypoints: {'ok' if success_flag else 'fail'}",
            f"Left: area={left_area} success={success_left} major={left_major_len:.1f} minor={left_minor_len:.1f}"
            + (f" round={left_round:.2f}" if left_round is not None and np.isfinite(left_round) else ""),
            f"Right: area={right_area} success={success_right} major={right_major_len:.1f} minor={right_minor_len:.1f}"
            + (f" round={right_round:.2f}" if right_round is not None and np.isfinite(right_round) else ""),
        ]
        summary = "\n".join(summary_lines)
        return overlay, summary, axes_data


def create_viewer(zarr_path: Path, eye_run: Optional[str], crop_run: Optional[str], keypoint_run: Optional[str]) -> None:
    root = open_zarr(zarr_path)
    eye_run = get_latest_run(root, "eye_masks", eye_run)
    crop_run = get_latest_run(root, "crop", crop_run)
    keypoint_run = get_latest_run(root, "keypoints", keypoint_run)

    viewer = EyeMaskViewer(root, eye_run, crop_run, keypoint_run)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.22)
    ax.set_axis_off()

    overlay, summary, axes = viewer.make_overlay(0)
    image_artist = ax.imshow(overlay, interpolation="nearest")
    info_text = ax.text(
        0.02,
        0.02,
        summary,
        color="white",
        fontsize=9,
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox=dict(facecolor="black", alpha=0.4, pad=4),
    )

    colors = {"left": "#1a66f3", "right": "#f85151"}
    line_major = {}
    line_minor = {}
    for side in ("left", "right"):
        if axes[side]:
            major_pts = axes[side].get("major", ((np.nan, np.nan), (np.nan, np.nan)))
            minor_pts = axes[side].get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
        else:
            major_pts = ((np.nan, np.nan), (np.nan, np.nan))
            minor_pts = ((np.nan, np.nan), (np.nan, np.nan))
        (major_line,) = ax.plot(
            [major_pts[0][0], major_pts[1][0]],
            [major_pts[0][1], major_pts[1][1]],
            color=colors[side],
            linewidth=1.8,
        )
        (minor_line,) = ax.plot(
            [minor_pts[0][0], minor_pts[1][0]],
            [minor_pts[0][1], minor_pts[1][1]],
            color=colors[side],
            linewidth=1.2,
            linestyle="--",
        )
        line_major[side] = major_line
        line_minor[side] = minor_line

    slider_ax = fig.add_axes([0.2, 0.08, 0.65, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="ROI Index",
        valmin=0,
        valmax=viewer.total - 1,
        valinit=0,
        valfmt="%0.0f",
    )

    def update_from_slider(val: float) -> None:
        idx = int(round(val))
        idx = max(0, min(viewer.total - 1, idx))
        overlay, info, axes = viewer.make_overlay(idx)
        image_artist.set_data(overlay)
        info_text.set_text(info)
        for side in ("left", "right"):
            if axes[side]:
                major_pts = axes[side].get("major", ((np.nan, np.nan), (np.nan, np.nan)))
                minor_pts = axes[side].get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
            else:
                major_pts = ((np.nan, np.nan), (np.nan, np.nan))
                minor_pts = ((np.nan, np.nan), (np.nan, np.nan))
            line_major[side].set_data(
                [major_pts[0][0], major_pts[1][0]],
                [major_pts[0][1], major_pts[1][1]],
            )
            line_minor[side].set_data(
                [minor_pts[0][0], minor_pts[1][0]],
                [minor_pts[0][1], minor_pts[1][1]],
            )
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
