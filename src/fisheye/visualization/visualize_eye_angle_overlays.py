#!/usr/bin/env python3
"""
Interactive explorer that overlays refined eye masks and signed eye-angle
metrics on the ROI crops used by the eye pipeline. Supports toggling between
major-axis (nasal-positive) and minor-axis (temporal-positive) angle
interpretations for side-by-side evaluation.

Loads a Palette archive, pulls the specified (or latest) eye-angle analysis run
along with its corresponding refined eye masks and keypoint crops, then renders
each ROI with:
  * Grayscale crop
  * Colored mask overlays (left/right or per-channel)
  * Signed left/right angles, vergence, version
  * QA flags and ellipse diagnostics

Use a slider (or ←/→ buttons) to scrub through detections. Optional export path
lets you save individual overlays as PNGs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import Button, Slider


DEFAULT_LEFT_COLOR = np.array([0.90, 0.23, 0.31], dtype=np.float32)   # reddish
DEFAULT_RIGHT_COLOR = np.array([0.18, 0.61, 0.95], dtype=np.float32)  # blue-ish
ADDITIONAL_COLORS = [
    np.array([0.47, 0.72, 0.31], dtype=np.float32),
    np.array([0.56, 0.35, 0.74], dtype=np.float32),
]


def _open_zarr(zarr_path: Path) -> zarr.Group:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr archive not found: {zarr_path}")
    return zarr.open_group(str(zarr_path), mode="r")


def _get_latest(root: zarr.Group, base: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    runs_group = f"{base}_runs"
    if runs_group not in root or "latest" not in root[runs_group].attrs:
        raise RuntimeError(f"No '{runs_group}' group or latest attribute found.")
    return root[runs_group].attrs["latest"]


def _normalize_image(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    min_val = float(img.min())
    max_val = float(img.max())
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_val) / (max_val - min_val)


def _compose_overlay(
    base_image: np.ndarray,
    masks: Sequence[np.ndarray],
    colors: Sequence[np.ndarray],
    alpha: float,
) -> np.ndarray:
    base_norm = _normalize_image(base_image)
    overlay = np.dstack([base_norm, base_norm, base_norm])
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        color = colors[idx % len(colors)]
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def _decode_reasons(code: int, mapping: Dict[int, str]) -> List[str]:
    names: List[str] = []
    for key, label in mapping.items():
        if code & key:
            names.append(label)
    return names


@dataclass
class AngleRecord:
    left: float
    right: float
    left_signed: float
    right_signed: float
    heading_deg: float
    vergence: float
    vergence_signed: float
    version: float
    valid_left: bool
    valid_right: bool
    valid_frame: bool
    reason_code: int
    ellipse_major: Optional[float]
    ellipse_minor: Optional[float]
    ellipse_ratio: Optional[float]
    left_minor_signed: float = float("nan")
    right_minor_signed: float = float("nan")
    vergence_minor_signed: float = float("nan")
    version_minor: float = float("nan")
    reason_names: List[str] = field(default_factory=list)
    summary_text: str = ""
    using_smoothed: bool = False


class EyeAngleOverlayViewer:
    def __init__(
        self,
        roi_images: zarr.Array,
        masks: zarr.Array,
        angles: Sequence[AngleRecord],
        keypoints_roi: Optional[zarr.Array],
        ellipse_params: Optional[zarr.Array],
        vergence_threshold: float,
    ) -> None:
        self._roi_images = roi_images
        self._masks = masks
        self._angles = list(angles)
        self._keypoints = keypoints_roi
        self._ellipse_params = ellipse_params
        self._vergence_threshold = float(vergence_threshold)
        self.total = int(roi_images.shape[0])
        self.channel_count = int(masks.shape[1]) if masks.ndim >= 4 else 0

        if self.total == 0 or self.channel_count == 0:
            raise ValueError("No ROI images or masks available to display.")

        self.colors = [DEFAULT_LEFT_COLOR, DEFAULT_RIGHT_COLOR] + ADDITIONAL_COLORS
        self.alpha = 0.55
        self.index = 0
        self.line_artists: List[plt.Artist] = []

        self.modes: List[str] = []
        if any(np.isfinite(rec.left_signed) and np.isfinite(rec.right_signed) for rec in self._angles):
            self.modes.append("ellipse_major")
        if any(np.isfinite(rec.left_minor_signed) and np.isfinite(rec.right_minor_signed) for rec in self._angles):
            self.modes.append("ellipse_minor")
        if not self.modes:
            self.modes.append("ellipse_major")
        else:
            self.modes = list(dict.fromkeys(self.modes))
        self.angle_mode: str = self.modes[0]

        self.fig, self.ax = plt.subplots(figsize=(6.5, 6.5))
        plt.subplots_adjust(bottom=0.22)

        self.image_artist = self.ax.imshow(
            _compose_overlay(
                np.asarray(self._roi_images[0]),
                [np.asarray(self._masks[0, ch]) for ch in range(self.channel_count)],
                self.colors,
                self.alpha,
            )
        )
        self.ax.axis("off")
        self.text_artist = self.ax.text(
            0.02,
            0.02,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
        )

        slider_ax = self.fig.add_axes([0.12, 0.10, 0.76, 0.04])
        self.slider = Slider(
            slider_ax,
            "ROI",
            0,
            self.total - 1,
            valinit=0,
            valstep=1,
            color="#1f77b4",
        )
        self.slider.on_changed(self._on_slider)

        prev_ax = self.fig.add_axes([0.12, 0.02, 0.08, 0.05])
        next_ax = self.fig.add_axes([0.80, 0.02, 0.08, 0.05])
        self.prev_button = Button(prev_ax, "Prev")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(lambda _event: self._step(-1))
        self.next_button.on_clicked(lambda _event: self._step(1))

        mode_ax = self.fig.add_axes([0.32, 0.02, 0.32, 0.05])
        self.mode_button = Button(mode_ax, f"Mode: {self._mode_label()}", color="#dddddd", hovercolor="#bbbbbb")
        self.mode_button.on_clicked(self._toggle_mode)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_display()

    def _fetch_masks(self, idx: int) -> List[np.ndarray]:
        return [np.asarray(self._masks[idx, ch]) for ch in range(self.channel_count)]

    def _format_angles(self, idx: int) -> str:
        record = self._angles[idx]
        def _fmt(value: Optional[float]) -> str:
            if value is None:
                return "–"
            try:
                if np.isfinite(value):
                    return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return "–"
            return "–"

        def _fmt_pair(signed_val: float, unsigned_val: float) -> str:
            return f"{_fmt(signed_val)}° / {_fmt(unsigned_val)}°"

        def _fmt_abs_pair(signed_val: float) -> str:
            return f"{_fmt(signed_val)}° / {_fmt(np.abs(signed_val))}°"

        lines = [
            f"ROI {idx + 1}/{self.total}",
            f"Left major (signed/unsigned): {_fmt_pair(record.left_signed, record.left)}",
            f"Left minor (signed/unsigned): {_fmt_abs_pair(record.left_minor_signed)}",
            f"Right major (signed/unsigned): {_fmt_pair(record.right_signed, record.right)}",
            f"Right minor (signed/unsigned): {_fmt_abs_pair(record.right_minor_signed)}",
            f"Vergence (signed/unsigned): {_fmt_pair(record.vergence_signed, record.vergence)}",
            f"Vergence minor (signed/unsigned): {_fmt_abs_pair(record.vergence_minor_signed)}",
            f"Version (signed/unsigned): {_fmt_pair(record.version, np.abs(record.version))}",
            f"Version minor (signed/unsigned): {_fmt_pair(record.version_minor, np.abs(record.version_minor))}",
            f"Valid (L/R/frame): {int(record.valid_left)}/{int(record.valid_right)}/{int(record.valid_frame)}",
            f"Heading: {_fmt(record.heading_deg)}°",
        ]
        if record.ellipse_major is not None and np.isfinite(record.ellipse_major):
            lines.append(
                f"Ellipse major/minor: {_fmt(record.ellipse_major)} / {_fmt(record.ellipse_minor)} px | ratio={_fmt(record.ellipse_ratio)}"
            )
        if record.reason_code and record.reason_names:
            lines.append("Reasons: " + ", ".join(record.reason_names))
        lines.append(f"Active axis: {self.angle_mode.capitalize()} (press 'm' to toggle)")
        if self._vergence_threshold > 0:
            lines.append(f"'v' jumps to next vergence ≥ {self._vergence_threshold:.1f}°")
        return "\n".join(lines)

    def _update_display(self) -> None:
        img = np.asarray(self._roi_images[self.index])
        masks = self._fetch_masks(self.index)
        overlay = _compose_overlay(img, masks, self.colors, self.alpha)

        self.image_artist.set_data(overlay)
        self.text_artist.set_text(self._format_angles(self.index))
        self._draw_vectors(self.index)
        self.ax.set_title(
            f"Eye mask + angles | ROI {self.index + 1}/{self.total} | mode: {self.angle_mode.capitalize()}"
        )
        self.fig.canvas.draw_idle()

    def _on_slider(self, value: float) -> None:
        self.index = int(value)
        self._update_display()

    def _step(self, delta: int) -> None:
        new_index = (self.index + delta) % self.total
        self.slider.set_val(new_index)

    def _mode_label(self) -> str:
        return {
            "ellipse_major": "Ellipse Major",
            "ellipse_minor": "Ellipse Minor",
        }.get(self.angle_mode, self.angle_mode.title())

    def _toggle_mode(self, _event=None) -> None:
        if len(self.modes) <= 1:
            print("Only one axis mode available; toggle ignored.")
            return
        idx = self.modes.index(self.angle_mode)
        self.angle_mode = self.modes[(idx + 1) % len(self.modes)]
        self.mode_button.label.set_text(f"Mode: {self._mode_label()}")
        self._update_display()

    def _on_key(self, event) -> None:
        if event.key in {"left", "left arrow"}:
            self._step(-1)
        elif event.key in {"right", "right arrow"}:
            self._step(1)
        elif event.key in {"m", "M"}:
            self._toggle_mode()
        elif event.key in {"v", "V"}:
            self._jump_to_vergence()

    def _jump_to_vergence(self) -> None:
        if not self._angles:
            return
        start = self.index
        threshold = self._vergence_threshold
        if threshold <= 0:
            print("Vergence threshold ≤ 0; shortcut ignored.")
            return
        total = self.total
        idx = (start + 1) % total
        while idx != start:
            record = self._angles[idx]
            value = float(record.vergence) if np.isfinite(record.vergence) else float("nan")
            if np.isfinite(value) and value >= threshold:
                self.slider.set_val(idx)
                return
            idx = (idx + 1) % total
        print(f"No vergence ≥ {threshold:.1f}° found after ROI {start + 1}.")

    def _draw_heading(self, idx: int) -> None:
        if self._keypoints is None:
            return
        try:
            kp = np.asarray(self._keypoints[idx])
        except Exception:
            return
        if kp.ndim < 2 or kp.shape[0] < 3 or kp.shape[1] < 2:
            return
        bladder = kp[0, :2].astype(np.float32)
        eye_left = kp[1, :2].astype(np.float32)
        eye_right = kp[2, :2].astype(np.float32)
        if not (np.all(np.isfinite(bladder)) and np.all(np.isfinite(eye_left)) and np.all(np.isfinite(eye_right))):
            return
        center = 0.5 * (eye_left + eye_right)
        vec = center - bladder
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm == 0.0:
            return
        color = (0.98, 0.82, 0.15)
        line = self.ax.plot(
            [bladder[0], center[0]],
            [bladder[1], center[1]],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            linestyle="-",
        )[0]
        marker = self.ax.scatter(
            [center[0]],
            [center[1]],
            color=color,
            s=30,
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )
        artists = [line, marker]
        heading_val = self._angles[idx].heading_deg if idx < len(self._angles) else float("nan")
        if np.isfinite(heading_val):
            text = self.ax.text(
                center[0],
                center[1],
                f"{heading_val:.1f}°",
                color="black",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
                zorder=8,
            )
            artists.append(text)
        self.line_artists.extend(artists)

    def _draw_vectors(self, idx: int) -> None:
        for artist in self.line_artists:
            artist.remove()
        self.line_artists.clear()

        self._draw_heading(idx)

        mode = self.angle_mode
        record = self._angles[idx]

        if mode.startswith("ellipse"):
            if self._ellipse_params is None:
                return
            params = np.asarray(self._ellipse_params[idx])
            if params.ndim != 2 or params.shape[1] < 5:
                return

            for ch in range(min(self.channel_count, params.shape[0])):
                row = params[ch]
                if not np.all(np.isfinite(row[:5])):
                    continue
                cx, cy, major_len, minor_len, theta_deg = row[:5]
                theta_rad = np.deg2rad(theta_deg)
                base_vec = (
                    np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float32)
                    if mode == "ellipse_major"
                    else np.array([-np.sin(theta_rad), np.cos(theta_rad)], dtype=np.float32)
                )
                signed_value = {
                    "ellipse_major": [record.left_signed, record.right_signed],
                    "ellipse_minor": [record.left_minor_signed, record.right_minor_signed],
                }[mode][ch] if ch < 2 else np.nan
                if not np.isfinite(signed_value):
                    continue
                length = float(major_len if mode == "ellipse_major" else minor_len) * 0.5
                if not np.isfinite(length) or length <= 0:
                    length = 1.5
                vec = base_vec * (1.0 if signed_value >= 0 else -1.0)
                x0 = float(cx)
                y0 = float(cy)
                x1 = x0 + vec[0] * length
                y1 = y0 + vec[1] * length
                color = self.colors[ch % len(self.colors)]
                line = self.ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.8, alpha=0.9)[0]
                marker = self.ax.scatter([x1], [y1], color=color, s=22, edgecolors="white", linewidths=0.5, zorder=6)
                self.line_artists.extend([line, marker])


def _load_roi_images(root: zarr.Group, keypoint_run: str) -> zarr.Array:
    kp_group = root[f"keypoints_runs/{keypoint_run}"]
    if "roi_images" in kp_group:
        return kp_group["roi_images"]
    crop_run = kp_group.attrs.get("source_crop_run")
    if crop_run and f"crop_runs/{crop_run}/roi_images" in root:
        return root[f"crop_runs/{crop_run}/roi_images"]
    raise RuntimeError(
        "Could not locate ROI images. Ensure keypoints run stores 'roi_images' or its source crop run does."
    )


def _load_keypoints(root: zarr.Group, keypoint_run: str) -> Optional[zarr.Array]:
    kp_group = root[f"keypoints_runs/{keypoint_run}"]
    return kp_group["keypoints_roi"] if "keypoints_roi" in kp_group else None


def _load_masks(root: zarr.Group, refined_run: str) -> tuple[zarr.Array, Optional[zarr.Array], Optional[zarr.Array], Optional[zarr.Array]]:
    group_path = f"refined_eye_masks_runs/{refined_run}"
    if group_path not in root:
        raise RuntimeError(f"Refined eye mask run '{refined_run}' not found.")
    group = root[group_path]
    if "masks_roi" not in group:
        raise RuntimeError(f"Group '{group_path}' missing 'masks_roi' dataset.")
    ellipse = group["ellipse_params"] if "ellipse_params" in group else None
    return group["masks_roi"], ellipse


def _load_angles(run_group: zarr.Group, prefer_smoothed: bool) -> tuple[List[AngleRecord], Dict[int, str]]:
    angles_grp = run_group["angles"]["roi"]
    qa_grp = run_group["qa"]["roi"]
    support = run_group.get("support")

    reason_map_raw = run_group.attrs.get("reason_code_map", {}) or {}
    reason_mapping = {int(k): str(v) for k, v in reason_map_raw.items()}

    def _pick_series(name: str, fallback: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool]:
        raw = (
            np.asarray(angles_grp[name][:], dtype=np.float32)
            if name in angles_grp
            else (
                np.asarray(fallback, dtype=np.float32)
                if fallback is not None
                else None
            )
        )
        if raw is None:
            raise KeyError(f"Required dataset '{name}' missing in eye angle run.")
        smoothed_name = f"{name}_smoothed"
        smoothed = (
            np.asarray(angles_grp[smoothed_name][:], dtype=np.float32)
            if prefer_smoothed and smoothed_name in angles_grp
            else None
        )
        if smoothed is not None:
            return smoothed, True
        return raw, False

    left, left_used_smoothed = _pick_series("left_deg")
    right, right_used_smoothed = _pick_series("right_deg")
    vergence, vergence_used_smoothed = _pick_series("vergence_deg")

    left_signed, left_signed_smoothed = _pick_series("left_signed_deg", fallback=left)
    right_signed, right_signed_smoothed = _pick_series("right_signed_deg", fallback=right)
    vergence_signed, vergence_signed_smoothed = _pick_series("vergence_signed_deg", fallback=vergence)
    version, version_smoothed = _pick_series("version_deg", fallback=np.full_like(left, np.nan, dtype=np.float32))
    left_minor_signed, left_minor_smoothed = _pick_series(
        "left_minor_signed_deg", fallback=np.full_like(left, np.nan, dtype=np.float32)
    )
    right_minor_signed, right_minor_smoothed = _pick_series(
        "right_minor_signed_deg", fallback=np.full_like(left, np.nan, dtype=np.float32)
    )
    vergence_minor_signed, vergence_minor_smoothed = _pick_series(
        "vergence_minor_signed_deg", fallback=np.full_like(left, np.nan, dtype=np.float32)
    )
    version_minor, version_minor_smoothed = _pick_series(
        "version_minor_deg", fallback=np.full_like(left, np.nan, dtype=np.float32)
    )

    heading = (
        np.asarray(angles_grp["heading_deg"][:], dtype=np.float32)
        if "heading_deg" in angles_grp
        else np.full_like(left, np.nan, dtype=np.float32)
    )

    any_smoothed_available = any(
        [
            left_used_smoothed,
            right_used_smoothed,
            vergence_used_smoothed,
            left_signed_smoothed,
            right_signed_smoothed,
            vergence_signed_smoothed,
            version_smoothed,
            left_minor_smoothed,
            right_minor_smoothed,
            vergence_minor_smoothed,
            version_minor_smoothed,
        ]
    )

    valid_left = np.asarray(qa_grp["valid_left"][:], dtype=bool)
    valid_right = np.asarray(qa_grp["valid_right"][:], dtype=bool)
    valid_frame = np.asarray(qa_grp["valid_frame"][:], dtype=bool)
    reason_codes = np.asarray(qa_grp["reason_codes"][:], dtype=np.uint16)

    ellipse_major = (
        np.asarray(support["ellipse_major"][:], dtype=np.float32)
        if support and "ellipse_major" in support
        else None
    )
    ellipse_minor = (
        np.asarray(support["ellipse_minor"][:], dtype=np.float32)
        if support and "ellipse_minor" in support
        else None
    )
    ellipse_ratio = (
        np.asarray(support["ellipse_ratio"][:], dtype=np.float32)
        if support and "ellipse_ratio" in support
        else None
    )

    total = left.shape[0]
    records: List[AngleRecord] = []
    for idx in range(total):
        reason_list = _decode_reasons(int(reason_codes[idx]), reason_mapping)
        def _fmt(val: float) -> str:
            return f"{val:.1f}" if np.isfinite(val) else "–"

        header = f"ROI {idx + 1}/{total}"
        if any_smoothed_available:
            header += " [smoothed]"
        lines = [
            header,
            f"Left signed (maj/min): {_fmt(left_signed[idx])}° / {_fmt(left_minor_signed[idx])}°",
            f"Right signed (maj/min): {_fmt(right_signed[idx])}° / {_fmt(right_minor_signed[idx])}°",
            f"Vergence signed (maj/min): {_fmt(vergence_signed[idx])}° / {_fmt(vergence_minor_signed[idx])}°",
            f"Version (maj/min): {_fmt(version[idx])}° / {_fmt(version_minor[idx])}°",
            f"Valid (L/R/frame): {int(valid_left[idx])}/{int(valid_right[idx])}/{int(valid_frame[idx])}",
            f"Heading: {_fmt(heading[idx])}°",
        ]
        if ellipse_major is not None and np.isfinite(ellipse_major[idx]):
            lines.append(
                f"Ellipse major/minor: {_fmt(ellipse_major[idx])} / {_fmt(ellipse_minor[idx])} px | ratio={_fmt(ellipse_ratio[idx])}"
            )
        if reason_list:
            lines.append("Reasons: " + ", ".join(reason_list))

        clean_lines = [line for line in lines if line]
        record = AngleRecord(
            left=float(left[idx]),
            right=float(right[idx]),
            left_signed=float(left_signed[idx]),
            right_signed=float(right_signed[idx]),
            heading_deg=float(heading[idx]),
            left_minor_signed=float(left_minor_signed[idx]),
            right_minor_signed=float(right_minor_signed[idx]),
            vergence=float(vergence[idx]),
            vergence_signed=float(vergence_signed[idx]),
            vergence_minor_signed=float(vergence_minor_signed[idx]),
            version=float(version[idx]),
            version_minor=float(version_minor[idx]),
            valid_left=bool(valid_left[idx]),
            valid_right=bool(valid_right[idx]),
            valid_frame=bool(valid_frame[idx]),
            reason_code=int(reason_codes[idx]),
            ellipse_major=None if ellipse_major is None else float(ellipse_major[idx]),
            ellipse_minor=None if ellipse_minor is None else float(ellipse_minor[idx]),
            ellipse_ratio=None if ellipse_ratio is None else float(ellipse_ratio[idx]),
            reason_names=reason_list,
            summary_text="\n".join(clean_lines),
            using_smoothed=any_smoothed_available,
        )
        records.append(record)
    return records, reason_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay refined eye masks with signed eye-angle metrics on ROI crops.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--eye-angle-run", dest="eye_angle_run", help="Specific analysis/eye_angle_runs/<run> to visualize.")
    parser.add_argument("--refined-eye-run", dest="refined_run", help="Specific refined_eye_masks_runs/<run> to use.")
    parser.add_argument("--keypoint-run", dest="keypoint_run", help="Specific keypoints_runs/<run> providing ROI images.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Mask overlay alpha (default 0.55).")
    parser.add_argument(
        "--angle-series",
        choices=["raw", "smoothed"],
        default="raw",
        help="Which eye-angle series to prioritise when smoothed data is available (default: raw).",
    )
    parser.add_argument(
        "--vergence-threshold",
        type=float,
        default=40.0,
        help="If >0, pressing 'v' jumps to the next ROI whose unsigned vergence meets/exceeds this threshold (default 40°).",
    )
    parser.add_argument("--no-show", action="store_true", help="Generate figure without displaying (useful for tests).")
    parser.add_argument("--save", type=Path, help="Optional path to save the current ROI overlay as PNG.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = _open_zarr(args.zarr_path)
    eye_angle_parent = root["analysis"]["eye_angle_runs"]
    eye_angle_run = args.eye_angle_run or eye_angle_parent.attrs.get("latest")
    if not eye_angle_run or eye_angle_run not in eye_angle_parent:
        raise RuntimeError("Eye angle run not found; specify --eye-angle-run.")
    run_group = eye_angle_parent[eye_angle_run]

    refined_run = args.refined_run or run_group.attrs.get("source_refined_eye_run")
    if not refined_run:
        raise RuntimeError("Could not determine refined eye mask run; pass --refined-eye-run.")
    refined_run = _get_latest(root, "refined_eye_masks", refined_run) if refined_run == "latest" else refined_run

    keypoint_run = args.keypoint_run or run_group.attrs.get("source_keypoint_run")
    if not keypoint_run:
        raise RuntimeError("Could not determine keypoint run; pass --keypoint-run.")
    keypoint_run = _get_latest(root, "keypoints", keypoint_run) if keypoint_run == "latest" else keypoint_run

    roi_images = _load_roi_images(root, keypoint_run)
    keypoints_roi = _load_keypoints(root, keypoint_run)
    masks, ellipse_params = _load_masks(root, refined_run)

    if roi_images.shape[0] != masks.shape[0]:
        raise ValueError(
            f"ROI image count ({roi_images.shape[0]}) does not match masks count ({masks.shape[0]})."
        )
    if keypoints_roi is not None and keypoints_roi.shape[0] != roi_images.shape[0]:
        raise ValueError(
            f"Keypoint count ({keypoints_roi.shape[0]}) does not match ROI image count ({roi_images.shape[0]})."
        )

    prefer_smoothed = args.angle_series == "smoothed"

    angle_records, _ = _load_angles(run_group, prefer_smoothed)
    if len(angle_records) != roi_images.shape[0]:
        raise ValueError(
            f"Angle record count ({len(angle_records)}) does not match ROI count ({roi_images.shape[0]})."
        )

    viewer = EyeAngleOverlayViewer(
        roi_images,
        masks,
        angle_records,
        keypoints_roi,
        ellipse_params,
        args.vergence_threshold,
    )
    viewer.alpha = float(np.clip(args.alpha, 0.05, 0.95))
    viewer._update_display()

    if args.save:
        overlay = viewer.image_artist.get_array()
        plt.imsave(args.save, overlay)

    if not args.no_show:
        plt.show()
    else:
        plt.close(viewer.fig)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
