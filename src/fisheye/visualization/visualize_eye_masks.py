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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider

from ..shared.crop_image_source import CropImageSource
from ..shared.mask_source import load_mask_bundle


def open_zarr(zarr_path: Path) -> zarr.Group:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)


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


def _load_frame_flags(path: Path) -> dict[str, list[dict[str, Optional[int]]]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: dict[str, list[dict[str, Optional[int]]]] = {}
    for key, value in data.items():
        entries: list[dict[str, Optional[int]]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    frame_val = item.get("frame_idx")
                    roi_val = item.get("roi_idx")
                    try:
                        frame_idx = int(frame_val) if frame_val is not None else None
                    except (TypeError, ValueError):
                        frame_idx = None
                    try:
                        roi_idx = int(roi_val) if roi_val is not None else None
                    except (TypeError, ValueError):
                        roi_idx = None
                    if frame_idx is not None:
                        entries.append({"frame_idx": frame_idx, "roi_idx": roi_idx})
                else:
                    try:
                        frame_idx = int(item)
                    except (TypeError, ValueError):
                        continue
                    entries.append({"frame_idx": frame_idx, "roi_idx": None})
        parsed[str(key)] = entries
    return parsed


def _append_flagged_frame(
    flag_path: Path,
    zarr_path: str,
    frame_idx: int,
    roi_idx: Optional[int],
) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_frame_flags(flag_path)
    entries = data.get(zarr_path, [])
    dedupe = {(entry.get("frame_idx"), entry.get("roi_idx")) for entry in entries}
    key = (int(frame_idx), int(roi_idx) if roi_idx is not None else None)
    if key in dedupe:
        return
    entries.append({"frame_idx": int(frame_idx), "roi_idx": key[1]})
    entries.sort(key=lambda item: (item.get("frame_idx") or 0, item.get("roi_idx") or -1))
    data[zarr_path] = entries
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _is_refined_variant(group_path: str) -> bool:
    return str(group_path).startswith("refined_eye_masks_runs/")


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


def _format_attr_value(key: str, value: object, max_items: int = 4) -> Optional[str]:
    if isinstance(value, dict):
        if not value:
            return f"{key}: {{}}"
        if key == "provenance":
            parts = []
            for sub_key in ("command", "created_at_utc"):
                if sub_key in value:
                    parts.append(f"{sub_key}={value[sub_key]}")
            if not parts:
                parts.append(f"keys={list(value.keys())[:max_items]}")
            return f"{key}: " + ", ".join(parts)
        items = []
        for idx, (sub_key, sub_val) in enumerate(value.items()):
            if idx >= max_items:
                items.append("…")
                break
            if isinstance(sub_val, (dict, list, tuple)):
                items.append(f"{sub_key}=…")
            else:
                items.append(f"{sub_key}={sub_val}")
        return f"{key}: " + ", ".join(items)
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{key}: []"
        if len(value) <= max_items and all(
            isinstance(elem, (str, int, float, bool)) for elem in value
        ):
            return f"{key}: {value}"
        return f"{key}: len={len(value)}"
    if isinstance(value, (np.generic,)):
        value = value.item()
    return f"{key}: {value}"


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
    skip_keys = {
        "method",
        "source_eye_masks_run",
        "source_keypoints_run",
        "source_crop_run",
        "smoothing",
        "refine_stats",
        "mask_probabilities_available",
        "mask_probability_threshold",
        "mask_probability_source",
        "mask_probability_policy",
        "dask_scheduler",
        "dask_num_workers",
        "dask_chunk_size",
        "dask_version",
        "git_commit",
        "git_branch",
        "hostname",
    }
    for key in sorted(group.attrs.keys()):
        value = group.attrs[key]
        if key in skip_keys or key.startswith("_"):
            continue
        if key == "source_eye_masks_method" and isinstance(value, str):
            summary.append(f"Source method: {value}")
            continue
        formatted = _format_attr_value(key, value)
        if formatted:
            summary.append(formatted)
    return summary


@dataclass
class MaskVariant:
    name: str
    group_path: str
    masks: object  # zarr.Array | dask.Array | np.ndarray
    mask_probs: Optional[object]
    ellipse_params: Optional[object]
    ellipse_success: Optional[object]
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
    threshold_attr = group.attrs.get("mask_probability_threshold", 0.5)
    try:
        threshold = float(threshold_attr)
    except (TypeError, ValueError):
        threshold = 0.5

    bundle = load_mask_bundle(
        group,
        threshold=threshold,
        prefer_probs=True,
        materialize=False,
        lazy=True,
    )
    masks = bundle.binary
    mask_probs = bundle.probs

    channel_count = int(masks.shape[1])

    ellipse_params = group["ellipse_params"] if "ellipse_params" in group else None
    ellipse_success = group["ellipse_success"] if "ellipse_success" in group else None

    eye_labels_attr = group.attrs.get("eye_labels")
    if isinstance(eye_labels_attr, (list, tuple)):
        eye_labels = [str(val) for val in eye_labels_attr]
    else:
        eye_labels = ["eye_left", "eye_right"]

    if channel_count == 1:
        eye_labels = ["union"]
    elif len(eye_labels) < channel_count:
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
        crop_source: CropImageSource,
        success_flags: np.ndarray,
        keypoints: np.ndarray,
        keypoint_labels: Sequence[str],
    ) -> None:
        if not variants:
            raise ValueError("No mask variants available to visualize.")
        self.root = root
        self.variants = list(variants)
        self.crop_source = crop_source
        self.total = int(self.crop_source.total_rois)
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
    def _ellipse_curve(
        cx: float,
        cy: float,
        major: float,
        minor: float,
        orientation_deg: float,
        num_points: int = 100,
    ) -> np.ndarray:
        if not (
            np.isfinite(cx)
            and np.isfinite(cy)
            and np.isfinite(major)
            and np.isfinite(minor)
            and np.isfinite(orientation_deg)
            and major > 0
            and minor > 0
        ):
            return np.zeros((0, 2), dtype=np.float32)

        theta = np.deg2rad(orientation_deg)
        t = np.linspace(0.0, 2.0 * np.pi, num=max(16, int(num_points)), endpoint=True)
        a = major / 2.0
        b = minor / 2.0

        # y is inverted in image coordinates (imshow), so apply the same
        # sign convention used by _axis_endpoints.
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        x = cx + (a * cos_t * np.cos(theta)) - (b * sin_t * np.sin(theta))
        y = cy - ((a * cos_t * np.sin(theta)) + (b * sin_t * np.cos(theta)))
        return np.column_stack([x, y]).astype(np.float32, copy=False)

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
        List[np.ndarray],
        Optional[List[np.ndarray]],
        np.ndarray,
        List[np.ndarray],
        MaskVariant,
        np.ndarray,
        np.ndarray,
    ]:
        variant = self.variants[variant_idx]
        roi = np.asarray(self.crop_source.read_slice(idx, idx + 1)[0])
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
        ellipse_curves: List[np.ndarray] = []
        if variant.ellipse_params is not None:
            ellipse_info = np.asarray(variant.ellipse_params[idx])
        else:
            ellipse_info = np.full((variant.channel_count, 5), np.nan, dtype=np.float32)

        for ch_idx in range(variant.channel_count):
            mask = np.asarray(variant.masks[idx, ch_idx])
            mask_list.append(mask)
            mask_bool = mask > 0
            color = variant.channel_colors[ch_idx]
            overlay[mask_bool] = (1 - 0.45) * overlay[mask_bool] + 0.45 * color

            area = int(mask.sum())

            if (
                variant.ellipse_success is not None
                and variant.ellipse_success.shape[0] > idx
                and variant.ellipse_success.shape[1] > ch_idx
            ):
                success = bool(np.asarray(variant.ellipse_success[idx, ch_idx]))
            else:
                success = False
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
            channel_curve = np.zeros((0, 2), dtype=np.float32)

            if success:
                channel_axes = self._axis_endpoints(cx, cy, major_len_raw, minor_len_raw, theta)
                channel_curve = self._ellipse_curve(cx, cy, major_len_raw, minor_len_raw, theta)

            axes_data.append(channel_axes)
            ellipse_curves.append(channel_curve)

            display_name = variant.display_names[ch_idx]
            summary_line = (
                f"{display_name}: area={area} success={success} "
                f"major={self._format_measure(major_len_raw)} minor={self._format_measure(minor_len_raw)}"
            )
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
        return (
            overlay,
            summary,
            axes_data,
            ellipse_curves,
            prob_maps,
            base,
            mask_list,
            variant,
            kp_array,
            kp_valid,
        )


def create_viewer(
    zarr_path: Path,
    eye_run: Optional[str],
    crop_run: Optional[str],
    keypoint_run: Optional[str],
    refined_runs: Optional[List[str]] = None,
    frame_flag_file: Optional[str] = None,
) -> None:
    root = open_zarr(zarr_path)
    eye_run = get_latest_run(root, "eye_masks", eye_run)
    crop_run = get_latest_run(root, "crop", crop_run)
    keypoint_run = get_latest_run(root, "keypoints", keypoint_run)

    crop_group = root[f"crop_runs/{crop_run}"]
    crop_source = CropImageSource.open(root, crop_run=crop_run, zarr_path=zarr_path)
    frame_indices_ds = crop_group.get("frame_indices")
    frame_indices: Optional[np.ndarray] = None
    if frame_indices_ds is not None:
        frame_indices = np.asarray(frame_indices_ds[:], dtype=np.int64)
        if frame_indices.shape[0] != crop_source.total_rois:
            raise ValueError(
                "crop frame_indices length ({}) does not match roi_images rows ({}).".format(
                    frame_indices.shape[0], crop_source.total_rois
                )
            )

    kp_group = root[f"keypoints_runs/{keypoint_run}"]
    success_flags = np.asarray(kp_group["detection_success"][:])
    keypoints = np.asarray(kp_group["keypoints_roi"][:])
    keypoint_labels = list(kp_group.attrs.get("keypoint_labels", []))
    flag_path = Path(frame_flag_file).expanduser() if frame_flag_file else None

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
            latest = refined_parent.attrs.get("latest")
            if latest and latest in refined_parent:
                group = refined_parent[latest]
                if isinstance(group, zarr.Group) and group.attrs.get("source_eye_masks_run") == eye_run:
                    refined_names.append(latest)
            if not refined_names:
                for name in refined_parent.keys():
                    group = refined_parent[name]
                    if isinstance(group, zarr.Group) and group.attrs.get("source_eye_masks_run") == eye_run:
                        refined_names.append(name)

    for name in refined_names:
        group_path = f"refined_eye_masks_runs/{name}"
        variants.append(build_mask_variant(root, group_path, f"Refined: {name}"))

    viewer = EyeMaskViewer(root, variants, crop_source, success_flags, keypoints, keypoint_labels)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.35, 0.12], width_ratios=[1, 1, 1])

    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_overlay.set_axis_off()

    ax_raw = fig.add_subplot(gs[0, 1])
    ax_raw.set_title("Raw ROI")
    ax_raw.set_axis_off()

    prob_panel = gs[0, 2]
    info_ax = fig.add_subplot(gs[1, :])
    info_ax.set_axis_off()
    info_ax.set_facecolor("#f5f5f5")
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    slider_ax = fig.add_subplot(gs[2, :])

    (
        overlay,
        summary,
        axes,
        ellipse_curves,
        probs,
        base_roi,
        bin_masks,
        variant,
        kp_coords,
        kp_valid,
    ) = viewer.make_overlay(0, viewer.variant_index)
    image_artist = ax_overlay.imshow(overlay, interpolation="nearest")
    metadata_artist = info_ax.text(
        0.01,
        0.99,
        summary,
        color="black",
        fontsize=9,
        verticalalignment="top",
        transform=info_ax.transAxes,
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
    line_ellipse: List = []
    for ch_idx in range(viewer.max_channels):
        if ch_idx < variant.channel_count:
            color = variant.channel_hex[ch_idx]
            axis_entry = axes[ch_idx] if ch_idx < len(axes) else {}
            curve = (
                ellipse_curves[ch_idx]
                if ch_idx < len(ellipse_curves)
                else np.zeros((0, 2), dtype=np.float32)
            )
            major_pts = axis_entry.get("major", ((np.nan, np.nan), (np.nan, np.nan)))
            minor_pts = axis_entry.get("minor", ((np.nan, np.nan), (np.nan, np.nan)))
        else:
            color = "#999999"
            curve = np.zeros((0, 2), dtype=np.float32)
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
        if curve.size > 0:
            x_curve = curve[:, 0]
            y_curve = curve[:, 1]
        else:
            x_curve = np.array([], dtype=np.float32)
            y_curve = np.array([], dtype=np.float32)
        (ellipse_line,) = ax_overlay.plot(
            x_curve,
            y_curve,
            color=color,
            linewidth=1.0,
            alpha=0.95,
        )
        line_major.append(major_line)
        line_minor.append(minor_line)
        line_ellipse.append(ellipse_line)

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
            ellipse_data,
            prob_maps,
            base_img,
            mask_pair,
            variant_local,
            kp_array,
            kp_valid_mask,
        ) = viewer.make_overlay(idx, variant_idx)
        image_artist.set_data(overlay)
        metadata_artist.set_text(info)
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
                curve = (
                    ellipse_data[ch_idx]
                    if ch_idx < len(ellipse_data)
                    else np.zeros((0, 2), dtype=np.float32)
                )
                if curve.size > 0:
                    line_ellipse[ch_idx].set_data(curve[:, 0], curve[:, 1])
                    line_ellipse[ch_idx].set_visible(True)
                else:
                    line_ellipse[ch_idx].set_data([], [])
                    line_ellipse[ch_idx].set_visible(False)
                line_major[ch_idx].set_visible(True)
                line_minor[ch_idx].set_visible(True)
            else:
                line_major[ch_idx].set_visible(False)
                line_minor[ch_idx].set_visible(False)
                line_ellipse[ch_idx].set_visible(False)

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
        elif event.key in {"b"}:
            if flag_path is None:
                print("No frame flag file configured. Pass --frame-flag-file to enable cleanup flagging.")
                return
            roi_idx = int(round(slider.val))
            roi_idx = max(0, min(viewer.total - 1, roi_idx))
            variant_local = viewer.variants[viewer.variant_index]
            if not _is_refined_variant(variant_local.group_path):
                print("Frame flagging is for refined variants only. Press 'v' to switch variants.")
                return
            if frame_indices is None:
                print(f"crop_runs/{crop_run} is missing frame_indices; cannot flag cleanup frames.")
                return
            frame_idx = int(frame_indices[roi_idx])
            try:
                _append_flagged_frame(flag_path, str(zarr_path), frame_idx, roi_idx)
                print(
                    "Flagged cleanup frame {frame_idx} (roi {roi_idx}) from {group}".format(
                        frame_idx=frame_idx,
                        roi_idx=roi_idx,
                        group=variant_local.group_path,
                    )
                )
                print(f"Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
            except Exception as exc:
                print(f"Failed to flag cleanup frame: {exc}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.canvas.manager.set_window_title(
        f"Eye Mask Viewer | eye_run={eye_run}, crop_run={crop_run}, keypoint_run={keypoint_run}"
    )
    try:
        plt.show()
    finally:
        crop_source.close()

def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--frame-flag-file",
        default="eye_mask_frame_flags.json",
        help="JSON file to append cleanup flags when pressing 'b' (default: eye_mask_frame_flags.json).",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    create_viewer(
        args.zarr_path,
        args.eye_run,
        args.crop_run,
        args.keypoint_run,
        args.refined_run,
        args.frame_flag_file,
    )


if __name__ == "__main__":
    main()
