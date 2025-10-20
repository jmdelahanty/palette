"""Interactive visualizer for eye mask segmentation runs.

Displays ROI crops with eye masks overlaid, using the masks produced
by the `eye_masks` pipeline stage. Handles both traditional left/right masks and
YOLO index-ordered masks, highlighting when refinement has not yet been applied.
If matching entries exist in ``refined_eye_masks_runs`` (or are specified via
``--refined-run``), their results are loaded as additional variants that can be
inspected alongside the original segmentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Patch
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


_REFINED_COLORS_RGB = [
    np.array([0.1, 0.4, 0.95], dtype=np.float32),
    np.array([0.95, 0.2, 0.2], dtype=np.float32),
    np.array([0.55, 0.2, 0.85], dtype=np.float32),
    np.array([0.15, 0.7, 0.8], dtype=np.float32),
]
_REFINED_COLORS_HEX = ["#1a66f3", "#f85151", "#8c33d9", "#26b2cc"]

_UNREFINED_COLORS_RGB = [
    np.array([0.2, 0.75, 0.45], dtype=np.float32),
    np.array([0.85, 0.6, 0.15], dtype=np.float32),
    np.array([0.6, 0.3, 0.1], dtype=np.float32),
    np.array([0.3, 0.6, 0.8], dtype=np.float32),
]
_UNREFINED_COLORS_HEX = ["#2fcc72", "#d89d1a", "#99511a", "#4c99cc"]


def _cycle_colors(
    rgb_palette: Sequence[np.ndarray], hex_palette: Sequence[str], count: int
) -> tuple[List[np.ndarray], List[str]]:
    rgb_out: List[np.ndarray] = []
    hex_out: List[str] = []
    for i in range(count):
        rgb_out.append(rgb_palette[i % len(rgb_palette)])
        hex_out.append(hex_palette[i % len(hex_palette)])
    return rgb_out, hex_out


def _friendly_eye_label(label: Optional[str], idx: int) -> str:
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


def _build_variant_summary(group: zarr.Group) -> List[str]:
    summary: List[str] = []
    method = group.attrs.get("method")
    if method:
        summary.append(f"Method: {method}")
    source = group.attrs.get("source_eye_masks_run")
    if source:
        summary.append(f"Source: {source}")
    smoothing = group.attrs.get("smoothing")
    if isinstance(smoothing, dict) and smoothing.get("enabled"):
        closing = smoothing.get("closing_radius")
        opening = smoothing.get("opening_radius")
        channels_modified = smoothing.get("channels_modified")
        summary.append(
            "Smoothing: closing={closing} opening={opening} modified={channels}".format(
                closing=closing,
                opening=opening,
                channels=channels_modified,
            )
        )
        if smoothing.get("probabilities_available"):
            threshold = smoothing.get("probability_threshold")
            splits = smoothing.get("probability_splits")
            summary.append(
                "Probability split: threshold={threshold:.2f} ROIs={splits}".format(
                    threshold=float(threshold) if threshold is not None else float("nan"),
                    splits=int(splits) if splits is not None else 0,
                )
            )
    return summary


@dataclass
class MaskVariant:
    name: str
    group_path: str
    masks: np.ndarray
    mask_probs: Optional[np.ndarray]
    ellipse_params: np.ndarray
    ellipse_success: np.ndarray
    feret_major: Optional[np.ndarray]
    feret_minor: Optional[np.ndarray]
    feret_roundness: Optional[np.ndarray]
    eye_labels: List[str]
    display_names: List[str]
    channel_colors: List[np.ndarray]
    channel_hex: List[str]
    is_refined: bool
    unrefined_note: Optional[str]
    summary_lines: List[str]

    @property
    def channel_count(self) -> int:
        return int(self.masks.shape[1])


def build_mask_variant(root: zarr.Group, group_path: str, name: str) -> MaskVariant:
    group = root[group_path]
    masks = group["masks_roi"]
    mask_probs = group["mask_probs_roi"] if "mask_probs_roi" in group else None
    ellipse_params = group["ellipse_params"]
    ellipse_success = group["ellipse_success"]
    feret_major = group["feret_axes_major"] if "feret_axes_major" in group else None
    feret_minor = group["feret_axes_minor"] if "feret_axes_minor" in group else None
    feret_roundness = group["feret_roundness"] if "feret_roundness" in group else None

    eye_labels_attr = group.attrs.get("eye_labels")
    if isinstance(eye_labels_attr, (list, tuple)):
        eye_labels = [str(val) for val in eye_labels_attr]
    else:
        eye_labels = ["eye_left", "eye_right"]

    channel_count = int(masks.shape[1])
    if len(eye_labels) < channel_count:
        eye_labels = [
            eye_labels[i] if i < len(eye_labels) else f"eye_{i}"
            for i in range(channel_count)
        ]

    display_names = [_friendly_eye_label(label, idx) for idx, label in enumerate(eye_labels)]
    normalized = [label.lower() for label in eye_labels]
    is_refined = channel_count == 2 and normalized[:2] == ["eye_left", "eye_right"]

    if is_refined:
        channel_colors, channel_hex = _cycle_colors(_REFINED_COLORS_RGB, _REFINED_COLORS_HEX, channel_count)
        unrefined_note = None
    else:
        channel_colors, channel_hex = _cycle_colors(_UNREFINED_COLORS_RGB, _UNREFINED_COLORS_HEX, channel_count)
        unrefined_note = (
            "Channels reflect YOLO index order "
            f"(eye_labels={', '.join(eye_labels)}); run refinement to align left/right."
        )

    summary_lines = _build_variant_summary(group)

    return MaskVariant(
        name=name,
        group_path=group_path,
        masks=masks,
        mask_probs=mask_probs,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        feret_major=feret_major,
        feret_minor=feret_minor,
        feret_roundness=feret_roundness,
        eye_labels=eye_labels,
        display_names=display_names,
        channel_colors=channel_colors,
        channel_hex=channel_hex,
        is_refined=is_refined,
        unrefined_note=unrefined_note,
        summary_lines=summary_lines,
    )


class EyeMaskViewer:
    def __init__(
        self,
        root: zarr.Group,
        variants: Sequence[MaskVariant],
        roi_images: np.ndarray,
        success_flags: np.ndarray,
        keypoints: np.ndarray,
        keypoint_labels: Sequence[str],
    ) -> None:
        if not variants:
            raise ValueError("No mask variants available to visualize.")
        self.root = root
        self.variants = list(variants)
        self.roi_images = roi_images
        self.total = int(self.roi_images.shape[0])
        self.success_flags = success_flags
        self.keypoints = np.asarray(keypoints)
        if self.keypoints.shape[0] != self.total:
            raise ValueError(
                f"Keypoint count ({self.keypoints.shape[0]}) does not match ROI count ({self.total})."
            )
        if self.keypoints.ndim != 3 or self.keypoints.shape[2] != 2:
            raise ValueError("Keypoints array must have shape (num_roi, num_keypoints, 2).")
        self.keypoint_labels = list(keypoint_labels)
        self.keypoint_count = int(self.keypoints.shape[1]) if self.keypoints.ndim >= 2 else 0
        for variant in self.variants:
            if variant.masks.shape[0] != self.total:
                raise ValueError(
                    f"Variant '{variant.name}' mask count ({variant.masks.shape[0]}) does not match ROI count ({self.total})."
                )
        self.variant_index = 0
        self.max_channels = max(variant.channel_count for variant in self.variants)
        self._keypoint_colors = [
            self._keypoint_color(self._keypoint_label(idx)) for idx in range(self.keypoint_count)
        ]

    @staticmethod
    def _format_measure(value: Optional[float], precision: int = 1) -> str:
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.{precision}f}"

    @staticmethod
    def _axis_endpoints(
        cx: float, cy: float, major: float, minor: float, orientation_deg: float
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

    def _keypoint_label(self, idx: int) -> str:
        if idx < len(self.keypoint_labels):
            return str(self.keypoint_labels[idx])
        return f"kp_{idx}"

    @staticmethod
    def _keypoint_color(label: str) -> str:
        lower = label.lower()
        if "left" in lower:
            return "#00d1b2"  # teal
        if "right" in lower:
            return "#ff6b6b"  # coral
        if "bladder" in lower:
            return "#ffd166"
        if "mouth" in lower or "nose" in lower:
            return "#ffe066"
        return "#f4f1bb"  # soft yellow

    def keypoint_color(self, idx: int) -> str:
        if 0 <= idx < len(self._keypoint_colors):
            return self._keypoint_colors[idx]
        return "#f4f1bb"

    def make_overlay(
        self,
        idx: int,
        variant_idx: int,
    ) -> tuple[
        np.ndarray,
        str,
        List[dict[str, tuple[tuple[float, float], tuple[float, float]]]],
        Optional[List[np.ndarray]],
        np.ndarray,
        List[np.ndarray],
        MaskVariant,
        np.ndarray,
        np.ndarray,
    ]:
        variant = self.variants[variant_idx]
        roi = np.asarray(self.roi_images[idx])
        base = normalize_roi(roi)
        overlay = np.dstack([base, base, base])

        success_flag = bool(self.success_flags[idx])
        summary_lines = [
            f"ROI {idx + 1}/{self.total} | keypoints: {'ok' if success_flag else 'fail'}",
            f"Variant: {variant.name}",
        ]
        summary_lines.extend(variant.summary_lines)

        kp_array = np.asarray(self.keypoints[idx], dtype=np.float32)
        kp_valid = np.all(np.isfinite(kp_array), axis=1)
        kp_summaries = []
        for kp_idx in range(self.keypoint_count):
            if kp_idx >= kp_array.shape[0]:
                break
            if not kp_valid[kp_idx]:
                continue
            label = self._keypoint_label(kp_idx)
            x, y = kp_array[kp_idx]
            kp_summaries.append(f"{label}=({x:.1f}, {y:.1f})")
        if kp_summaries:
            summary_lines.append("Keypoints: " + ", ".join(kp_summaries))

        mask_list: List[np.ndarray] = []
        axes_data: List[dict[str, tuple[tuple[float, float], tuple[float, float]]]] = []
        ellipse_info = np.asarray(variant.ellipse_params[idx])

        for ch_idx in range(variant.channel_count):
            mask = np.asarray(variant.masks[idx, ch_idx])
            mask_list.append(mask)
            mask_bool = mask > 0
            color = variant.channel_colors[ch_idx]
            overlay[mask_bool] = (1 - 0.45) * overlay[mask_bool] + 0.45 * color

            area = int(mask.sum())

            success = (
                bool(np.asarray(variant.ellipse_success[idx, ch_idx]))
                if ch_idx < variant.ellipse_success.shape[1]
                else False
            )
            ellipse_row = (
                np.asarray(ellipse_info[ch_idx])
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

            if variant.feret_major is not None and ch_idx < variant.feret_major.shape[1]:
                major_seg = np.asarray(variant.feret_major[idx, ch_idx])
                if np.all(np.isfinite(major_seg)):
                    channel_axes["major"] = (
                        (major_seg[0], major_seg[1]),
                        (major_seg[2], major_seg[3]),
                    )
                    feret_len = self._length_from_segment(major_seg)
            if variant.feret_minor is not None and ch_idx < variant.feret_minor.shape[1]:
                minor_seg = np.asarray(variant.feret_minor[idx, ch_idx])
                if np.all(np.isfinite(minor_seg)):
                    channel_axes["minor"] = (
                        (minor_seg[0], minor_seg[1]),
                        (minor_seg[2], minor_seg[3]),
                    )
                    feret_minor_len = self._length_from_segment(minor_seg)
            if variant.feret_roundness is not None and ch_idx < variant.feret_roundness.shape[1]:
                feret_round_val = float(np.asarray(variant.feret_roundness[idx, ch_idx]))

            if success and "major" not in channel_axes:
                channel_axes = self._axis_endpoints(cx, cy, major_len_raw, minor_len_raw, theta)

            axes_data.append(channel_axes)

            display_name = variant.display_names[ch_idx]
            major_len = feret_len if feret_len is not None else major_len_raw
            minor_len = feret_minor_len if feret_minor_len is not None else minor_len_raw
            summary_line = (
                f"{display_name}: area={area} success={success} "
                f"major={self._format_measure(major_len)} minor={self._format_measure(minor_len)}"
            )
            if feret_round_val is not None and np.isfinite(feret_round_val):
                summary_line += f" round={feret_round_val:.2f}"
            summary_lines.append(summary_line)

        if variant.unrefined_note:
            summary_lines.append(variant.unrefined_note)

        summary = "\n".join(summary_lines)
        prob_maps: Optional[List[np.ndarray]] = None
        if variant.mask_probs is not None:
            prob_maps = [
                np.asarray(variant.mask_probs[idx, ch_idx], dtype=np.float32)
                for ch_idx in range(variant.channel_count)
            ]
        return overlay, summary, axes_data, prob_maps, base, mask_list, variant, kp_array, kp_valid


def create_viewer(
    zarr_path: Path,
    eye_run: Optional[str],
    crop_run: Optional[str],
    keypoint_run: Optional[str],
    refined_runs: Optional[List[str]] = None,
) -> None:
    root = open_zarr(zarr_path)
    eye_run = get_latest_run(root, "eye_masks", eye_run)
    crop_run = get_latest_run(root, "crop", crop_run)
    keypoint_run = get_latest_run(root, "keypoints", keypoint_run)

    roi_images = root[f"crop_runs/{crop_run}/roi_images"]
    kp_group = root[f"keypoints_runs/{keypoint_run}"]
    success_flags = np.asarray(kp_group["detection_success"][:])
    keypoints = np.asarray(kp_group["keypoints_roi"][:])
    keypoint_labels = list(kp_group.attrs.get("keypoint_labels", []))

    variants: List[MaskVariant] = []
    base_group_path = f"eye_masks_runs/{eye_run}"
    variants.append(build_mask_variant(root, base_group_path, f"Original: {eye_run}"))

    refined_parent = root.get("refined_eye_masks_runs")
    refined_names: List[str] = []
    if refined_parent is not None:
        if refined_runs:
            for name in refined_runs:
                if name not in refined_parent:
                    raise ValueError(f"Refined eye-mask run '{name}' not found.")
                refined_names.append(name)
        else:
            for name in refined_parent.keys():
                group = refined_parent[name]
                if isinstance(group, zarr.Group) and group.attrs.get("source_eye_masks_run") == eye_run:
                    refined_names.append(name)

    for name in refined_names:
        group_path = f"refined_eye_masks_runs/{name}"
        variants.append(build_mask_variant(root, group_path, f"Refined: {name}"))

    viewer = EyeMaskViewer(root, variants, roi_images, success_flags, keypoints, keypoint_labels)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.18], width_ratios=[1, 1, 1])

    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_overlay.set_axis_off()

    ax_raw = fig.add_subplot(gs[0, 1])
    ax_raw.set_title("Raw ROI")
    ax_raw.set_axis_off()

    prob_panel = gs[0, 2]
    slider_ax = fig.add_subplot(gs[1, :])

    overlay, summary, axes, probs, base_roi, bin_masks, variant, kp_coords, kp_valid = viewer.make_overlay(0, viewer.variant_index)
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
    variant_text = fig.text(
        0.02,
        0.95,
        f"Variant 1/{len(viewer.variants)}: {variant.name}",
        color="black",
        fontsize=10,
        transform=fig.transFigure,
    )

    raw_artist = ax_raw.imshow(base_roi, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    line_major: List = []
    line_minor: List = []
    for ch_idx in range(viewer.max_channels):
        if ch_idx < variant.channel_count:
            color = variant.channel_hex[ch_idx]
            axis_entry = axes[ch_idx] if ch_idx < len(axes) else {}
            major_pts = axis_entry.get("major", ((np.nan, np.nan), (np.nan, np.nan)))
            minor_pts = axis_entry.get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
        else:
            color = "#999999"
            major_pts = ((np.nan, np.nan), (np.nan, np.nan))
            minor_pts = ((np.nan, np.nan), (np.nan, np.nan))
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

    keypoint_artists: List = []
    for kp_idx in range(viewer.keypoint_count):
        color = viewer.keypoint_color(kp_idx)
        (artist,) = ax_overlay.plot(
            [],
            [],
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="black",
            linestyle="None",
            label=viewer._keypoint_label(kp_idx),
        )
        keypoint_artists.append(artist)

    for kp_idx, artist in enumerate(keypoint_artists):
        if kp_idx < kp_coords.shape[0] and kp_valid[kp_idx]:
            x, y = kp_coords[kp_idx]
            artist.set_data([x], [y])
            artist.set_visible(True)
        else:
            artist.set_visible(False)

    slider = Slider(
        ax=slider_ax,
        label="ROI Index",
        valmin=0,
        valmax=viewer.total - 1,
        valinit=0,
        valfmt="%0.0f",
    )

    mask_legend = None
    keypoint_legend = None

    def update_mask_legend(variant_local: MaskVariant) -> None:
        nonlocal mask_legend
        if mask_legend is not None:
            try:
                mask_legend.remove()
            except ValueError:
                pass
            mask_legend = None
        handles = [
            Patch(facecolor=variant_local.channel_hex[ch_idx], edgecolor="black", label=variant_local.display_names[ch_idx])
            for ch_idx in range(variant_local.channel_count)
        ]
        if handles:
            mask_legend = fig.legend(
                handles,
                [h.get_label() for h in handles],
                loc="upper left",
                bbox_to_anchor=(0.02, 0.88),
                fontsize=8,
                framealpha=0.6,
                handlelength=1.5,
                facecolor="white",
                title="Masks",
            )
            mask_legend.get_title().set_fontsize(9)

    def update_keypoint_legend(kp_valid_mask: np.ndarray) -> None:
        nonlocal keypoint_legend
        if keypoint_legend is not None:
            keypoint_legend.remove()
            keypoint_legend = None
        handles = []
        labels = []
        for kp_idx, artist in enumerate(keypoint_artists):
            if kp_idx < kp_valid_mask.shape[0] and kp_valid_mask[kp_idx]:
                handles.append(artist)
                labels.append(viewer._keypoint_label(kp_idx))
        if handles:
            keypoint_legend = ax_overlay.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=8,
                framealpha=0.6,
                handlelength=1.5,
                facecolor="white",
                title="Keypoints",
            )
            keypoint_legend.get_title().set_fontsize(9)

    prob_images: List = []
    prob_axes_list: List = []
    bin_images: List = []
    bin_axes_list: List = []

    sub_gs = prob_panel.subgridspec(2, max(1, viewer.max_channels), hspace=0.25, wspace=0.2)
    zeros_prob = np.zeros_like(base_roi, dtype=np.float32)
    zeros_bin = np.zeros_like(base_roi, dtype=np.float32)
    for col in range(viewer.max_channels):
        ax_prob = fig.add_subplot(sub_gs[0, col])
        ax_prob.set_axis_off()
        im_prob = ax_prob.imshow(zeros_prob, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        prob_axes_list.append(ax_prob)
        prob_images.append(im_prob)

        ax_bin = fig.add_subplot(sub_gs[1, col])
        ax_bin.set_axis_off()
        im_bin = ax_bin.imshow(zeros_bin, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        bin_axes_list.append(ax_bin)
        bin_images.append(im_bin)

    any_probs = any(v.mask_probs is not None for v in viewer.variants)
    prob_colorbar = None
    if any_probs and prob_axes_list:
        prob_colorbar = fig.colorbar(prob_images[0], ax=prob_axes_list, fraction=0.046, pad=0.04)

    def update_display(idx: int, variant_idx: int) -> None:
        (
            overlay,
            info,
            axes_data,
            prob_maps,
            base_img,
            mask_pair,
            variant_local,
            kp_array,
            kp_valid_mask,
        ) = viewer.make_overlay(idx, variant_idx)
        image_artist.set_data(overlay)
        info_text.set_text(info)
        raw_artist.set_data(base_img)
        variant_text.set_text(
            f"Variant {variant_idx + 1}/{len(viewer.variants)}: {variant_local.name}"
        )

        for ch_idx in range(viewer.max_channels):
            if ch_idx < variant_local.channel_count:
                axis_entry = axes_data[ch_idx] if ch_idx < len(axes_data) else {}
                major_pts = axis_entry.get("major", ((np.nan, np.nan), (np.nan, np.nan)))
                minor_pts = axis_entry.get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
                line_color = variant_local.channel_hex[ch_idx]
                line_major[ch_idx].set_color(line_color)
                line_minor[ch_idx].set_color(line_color)
                line_major[ch_idx].set_data(
                    [major_pts[0][0], major_pts[1][0]],
                    [major_pts[0][1], major_pts[1][1]],
                )
                line_minor[ch_idx].set_data(
                    [minor_pts[0][0], minor_pts[1][0]],
                    [minor_pts[0][1], minor_pts[1][1]],
                )
                line_major[ch_idx].set_visible(True)
                line_minor[ch_idx].set_visible(True)
            else:
                line_major[ch_idx].set_visible(False)
                line_minor[ch_idx].set_visible(False)

        for kp_idx, artist in enumerate(keypoint_artists):
            if kp_idx < kp_array.shape[0] and kp_valid_mask[kp_idx]:
                x, y = kp_array[kp_idx]
                artist.set_data([x], [y])
                artist.set_markerfacecolor(viewer.keypoint_color(kp_idx))
                artist.set_visible(True)
            else:
                artist.set_visible(False)

        update_mask_legend(variant_local)
        update_keypoint_legend(kp_valid_mask)

        for col in range(viewer.max_channels):
            if col < variant_local.channel_count:
                bin_axes_list[col].set_visible(True)
                bin_axes_list[col].set_axis_off()
                bin_axes_list[col].set_title(
                    f"{variant_local.display_names[col]} Binary Mask", fontsize=9
                )
                bin_images[col].set_data(mask_pair[col])
            else:
                bin_axes_list[col].set_title("")
                bin_axes_list[col].set_visible(False)

        if prob_images:
            if prob_maps is None:
                for col in range(viewer.max_channels):
                    prob_axes_list[col].set_title("")
                    prob_axes_list[col].set_visible(False)
            else:
                for col in range(viewer.max_channels):
                    if col < variant_local.channel_count:
                        prob_axes_list[col].set_visible(True)
                        prob_axes_list[col].set_axis_off()
                        prob_axes_list[col].set_title(
                            f"{variant_local.display_names[col]} Probability", fontsize=9
                        )
                        prob_images[col].set_data(prob_maps[col])
                    else:
                        prob_axes_list[col].set_title("")
                        prob_axes_list[col].set_visible(False)
        if prob_colorbar is not None:
            prob_colorbar.ax.set_visible(prob_maps is not None)

        fig.canvas.draw_idle()

    update_display(0, viewer.variant_index)

    def on_slider(val: float) -> None:
        idx = int(round(val))
        idx = max(0, min(viewer.total - 1, idx))
        update_display(idx, viewer.variant_index)

    slider.on_changed(on_slider)

    def step_roi(delta: int) -> None:
        idx = int(round(slider.val)) + delta
        idx = max(0, min(viewer.total - 1, idx))
        slider.set_val(idx)

    ax_prev = fig.add_axes([0.16, 0.015, 0.12, 0.04])
    ax_next = fig.add_axes([0.72, 0.015, 0.12, 0.04])
    btn_prev = Button(ax_prev, "Prev ROI")
    btn_next = Button(ax_next, "Next ROI")

    btn_prev.on_clicked(lambda _: step_roi(-1))
    btn_next.on_clicked(lambda _: step_roi(1))

    ax_variant_prev = fig.add_axes([0.40, 0.015, 0.1, 0.04])
    ax_variant_next = fig.add_axes([0.52, 0.015, 0.1, 0.04])
    btn_variant_prev = Button(ax_variant_prev, "Prev Variant")
    btn_variant_next = Button(ax_variant_next, "Next Variant")

    def step_variant(delta: int) -> None:
        viewer.variant_index = (viewer.variant_index + delta) % len(viewer.variants)
        update_display(int(round(slider.val)), viewer.variant_index)

    btn_variant_prev.on_clicked(lambda _: step_variant(-1))
    btn_variant_next.on_clicked(lambda _: step_variant(1))

    def on_key(event) -> None:
        if event.key in {"left", "j"}:
            step_roi(-1)
        elif event.key in {"right", "l"}:
            step_roi(1)
        elif event.key in {"down"}:
            step_roi(-5)
        elif event.key in {"up"}:
            step_roi(5)
        elif event.key in {"v", "m"}:
            step_variant(1)

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
    parser.add_argument(
        "--refined-run",
        action="append",
        help="Refined eye mask run to include (can be repeated). If omitted, any refined runs referencing the source will be included automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_viewer(args.zarr_path, args.eye_run, args.crop_run, args.keypoint_run, args.refined_run)


if __name__ == "__main__":
    main()
