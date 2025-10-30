#!/usr/bin/env python3
"""
Visualize eye angle analysis outputs stored in analysis/eye_angle_runs.

Builds a dashboard highlighting per-eye timelines, distributions, and QA
summary derived from ``fisheye.analysis.eye_angle_analysis``.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zarr
try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None

try:
    from PIL import Image, PngImagePlugin
except ImportError:  # pragma: no cover
    Image = None
    PngImagePlugin = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
MAX_TIMELINE_POINTS = 8000
SCATTER_DECIMATE = 10


def _serialize_xmp(payload: Dict[str, object]) -> str:
    json_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        '<?xpacket begin="\\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:palette="https://palette.hhmi.org/ns/analysis/">\n'
        f'   <palette:eye_angle_provenance>{json_payload}</palette:eye_angle_provenance>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )


def _save_with_metadata(fig: plt.Figure, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)

    if Image is None or PngImagePlugin is None:  # pragma: no cover
        with path.open("wb") as fh:
            fh.write(buffer.getvalue())
        return

    image = Image.open(buffer)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("XML:com.adobe.xmp", _serialize_xmp(metadata))
    image.save(path, pnginfo=pnginfo)


def _load_eye_angle_run(zarr_path: Path, run_name: Optional[str]) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    root = zarr.open(str(zarr_path), mode="r")
    analysis = root.get("analysis")
    if analysis is None or "eye_angle_runs" not in analysis:
        raise ValueError("Archive is missing analysis/eye_angle_runs.")

    parent = analysis["eye_angle_runs"]
    if run_name is None:
        run_name = parent.attrs.get("latest")
        if not run_name:
            raise ValueError("No eye_angle_runs found; specify --run.")
    if run_name not in parent:
        raise ValueError(f"Run '{run_name}' not found in analysis/eye_angle_runs.")

    run_group = parent[run_name]
    angles = run_group["angles"]
    qa = run_group["qa"]
    support = run_group.get("support")

    roi_group = angles["roi"]

    def _maybe(name: str) -> Optional[np.ndarray]:
        return roi_group[name][:] if name in roi_group else None

    roi_angles = {
        "left": roi_group["left_deg"][:],
        "right": roi_group["right_deg"][:],
        "vergence": roi_group["vergence_deg"][:],
        "left_signed": _maybe("left_signed_deg"),
        "right_signed": _maybe("right_signed_deg"),
        "vergence_signed": _maybe("vergence_signed_deg"),
        "version": _maybe("version_deg"),
        "left_minor_signed": _maybe("left_minor_signed_deg"),
        "right_minor_signed": _maybe("right_minor_signed_deg"),
        "vergence_minor_signed": _maybe("vergence_minor_signed_deg"),
        "version_minor": _maybe("version_minor_deg"),
        "left_feret_major_signed": _maybe("left_feret_major_signed_deg"),
        "right_feret_major_signed": _maybe("right_feret_major_signed_deg"),
        "vergence_feret_major_signed": _maybe("vergence_feret_major_signed_deg"),
        "version_feret_major": _maybe("version_feret_major_deg"),
        "left_feret_minor_signed": _maybe("left_feret_minor_signed_deg"),
        "right_feret_minor_signed": _maybe("right_feret_minor_signed_deg"),
        "vergence_feret_minor_signed": _maybe("vergence_feret_minor_signed_deg"),
        "version_feret_minor": _maybe("version_feret_minor_deg"),
        "left_speed": _maybe("left_speed_deg_s"),
        "right_speed": _maybe("right_speed_deg_s"),
        "vergence_speed": _maybe("vergence_speed_deg_s"),
        "vergence_signed_speed": _maybe("vergence_signed_speed_deg_s"),
        "version_speed": _maybe("version_speed_deg_s"),
    }

    qa_roi = {
        "valid_left": qa["roi"]["valid_left"][:],
        "valid_right": qa["roi"]["valid_right"][:],
        "valid_frame": qa["roi"]["valid_frame"][:],
        "reason_codes": qa["roi"]["reason_codes"][:],
    }

    frame_data = {}
    if "frame" in angles:
        frame_group = angles["frame"]
        def _frame_maybe(name: str) -> Optional[np.ndarray]:
            return frame_group[name][:] if name in frame_group else None
        frame_data["left"] = frame_group["left_deg"][:]
        frame_data["right"] = frame_group["right_deg"][:]
        frame_data["vergence"] = frame_group["vergence_deg"][:]
        frame_data["vergence_signed"] = _frame_maybe("vergence_signed_deg")
        frame_data["vergence_minor_signed"] = _frame_maybe("vergence_minor_signed_deg")
        frame_data["vergence_feret_major_signed"] = _frame_maybe("vergence_feret_major_signed_deg")
        frame_data["vergence_feret_minor_signed"] = _frame_maybe("vergence_feret_minor_signed_deg")
        frame_data["version"] = _frame_maybe("version_deg")
        frame_data["version_minor"] = _frame_maybe("version_minor_deg")
        frame_data["version_feret_major"] = _frame_maybe("version_feret_major_deg")
        frame_data["version_feret_minor"] = _frame_maybe("version_feret_minor_deg")
    if "frame" in qa:
        frame_data["frame_valid"] = qa["frame"]["valid_frame"][:]
        frame_data["frame_reason"] = qa["frame"]["reason_codes"][:]

    support_data = {}
    if support is not None:
        if "time_seconds" in support:
            support_data["time_seconds"] = support["time_seconds"][:]
        if "frame_indices" in support:
            support_data["frame_indices"] = support["frame_indices"][:]
        if "frame_time_seconds" in support:
            support_data["frame_time_seconds"] = support["frame_time_seconds"][:]
        if "ellipse_major" in support:
            support_data["ellipse_major"] = support["ellipse_major"][:]
        if "ellipse_minor" in support:
            support_data["ellipse_minor"] = support["ellipse_minor"][:]
        if "ellipse_ratio" in support:
            support_data["ellipse_ratio"] = support["ellipse_ratio"][:]

    attrs = dict(run_group.attrs)
    attrs["run_name"] = run_name

    data = {
        "roi_angles": roi_angles,
        "qa": qa_roi,
        "frame": frame_data,
        "support": support_data,
    }
    return attrs, data


def _downsample(values: Optional[np.ndarray], max_points: int) -> np.ndarray:
    if values is None:
        return np.array([])
    if values.size <= max_points:
        return values
    idx = np.linspace(0, values.size - 1, max_points, dtype=int)
    return values[idx]


def _find_bimodal_valley(x: np.ndarray, nbins: int = 256) -> float:
    x = x[np.isfinite(x)]
    if x.size < 200:
        return np.nan
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan
    grid = np.linspace(lo, hi, nbins)
    if gaussian_kde is None:
        hist, edges = np.histogram(x, bins=nbins, range=(lo, hi), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        pdf = hist
        grid_eval = centers
    else:
        kde = gaussian_kde(x)
        pdf = kde(grid)
        grid_eval = grid
    peaks = np.r_[False, (pdf[1:-1] > pdf[:-2]) & (pdf[1:-1] > pdf[2:]), False]
    peak_idx = np.flatnonzero(peaks)
    if peak_idx.size < 2:
        return np.nan
    i0, i1 = peak_idx[0], peak_idx[1]
    if i0 >= i1:
        return np.nan
    valley_rel = np.argmin(pdf[i0:i1 + 1])
    valley_idx = i0 + valley_rel
    return float(grid_eval[valley_idx])


def _prepare_timeline_data(
    time: Optional[np.ndarray],
    frame_indices: Optional[np.ndarray],
    fps: Optional[float],
) -> Tuple[np.ndarray, str]:
    if time is not None and np.isfinite(time).any():
        return time, "Time (s)"
    if frame_indices is not None:
        label = "Frame"
        if fps and fps > 0:
            return frame_indices.astype(np.float64) / float(fps), "Time (s)"
        return frame_indices.astype(np.float64), label
    return np.arange(0, 0), "Sample"


def _reason_counts(reason_codes: np.ndarray, mapping: Dict[str, str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for code_str, label in mapping.items():
        code = int(code_str)
        mask = (reason_codes & code) > 0
        counts[label] = int(mask.sum())
    return counts


def _select_angle_variant(
    roi_angles: Dict[str, Optional[np.ndarray]],
    source: str,
) -> Tuple[Dict[str, Optional[np.ndarray]], str]:
    """Return the angle series to visualize along with a human-friendly label."""

    def _require(name: str) -> np.ndarray:
        arr = roi_angles.get(name)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Dataset '{name}' required for angle source '{source}' is missing.")
        return arr

    source = source.lower()
    if source == "ellipse":
        variant = {
            "left": roi_angles["left"],
            "right": roi_angles["right"],
            "vergence": roi_angles["vergence"],
            "left_signed": roi_angles.get("left_signed"),
            "right_signed": roi_angles.get("right_signed"),
            "vergence_signed": roi_angles.get("vergence_signed"),
            "version": roi_angles.get("version"),
        }
        label = "Ellipse major axis"
    elif source == "minor":
        left_signed = _require("left_minor_signed")
        right_signed = _require("right_minor_signed")
        vergence_signed = _require("vergence_minor_signed")
        variant = {
            "left": np.abs(left_signed),
            "right": np.abs(right_signed),
            "vergence": np.abs(vergence_signed),
            "left_signed": left_signed,
            "right_signed": right_signed,
            "vergence_signed": vergence_signed,
            "version": roi_angles.get("version_minor"),
        }
        label = "Ellipse minor axis"
    elif source == "feret_major":
        left_signed = _require("left_feret_major_signed")
        right_signed = _require("right_feret_major_signed")
        vergence_signed = _require("vergence_feret_major_signed")
        variant = {
            "left": np.abs(left_signed),
            "right": np.abs(right_signed),
            "vergence": np.abs(vergence_signed),
            "left_signed": left_signed,
            "right_signed": right_signed,
            "vergence_signed": vergence_signed,
            "version": roi_angles.get("version_feret_major"),
        }
        label = "Feret major axis"
    elif source == "feret_minor":
        left_signed = _require("left_feret_minor_signed")
        right_signed = _require("right_feret_minor_signed")
        vergence_signed = _require("vergence_feret_minor_signed")
        variant = {
            "left": np.abs(left_signed),
            "right": np.abs(right_signed),
            "vergence": np.abs(vergence_signed),
            "left_signed": left_signed,
            "right_signed": right_signed,
            "vergence_signed": vergence_signed,
            "version": roi_angles.get("version_feret_minor"),
        }
        label = "Feret minor axis"
    else:
        raise ValueError(f"Unknown angle source '{source}'.")

    return variant, label


def _format_summary_lines(attrs: Dict[str, object], counts: Dict[str, int], roi_valid: np.ndarray, frame_valid: Optional[np.ndarray]) -> str:
    total = int(attrs.get("num_detections", roi_valid.size))
    valid = int(np.sum(roi_valid))
    total_frames = int(attrs.get("num_frames", frame_valid.size if frame_valid is not None else 0))
    valid_frames = int(np.sum(frame_valid)) if frame_valid is not None else None

    lines = [
        f"Run: {attrs.get('run_name', 'unknown')}",
        f"Created: {attrs.get('provenance', {}).get('timestamp_utc', attrs.get('timestamp_utc', 'unknown'))}",
        f"Detections: {valid}/{total} valid ({(valid/total*100.0) if total else 0:.1f}%)",
    ]
    if valid_frames is not None:
        lines.append(f"Frames: {valid_frames}/{total_frames} valid ({(valid_frames/total_frames*100.0) if total_frames else 0:.1f}%)")
    fps = attrs.get("fps")
    if fps:
        lines.append(f"FPS: {fps:.3f}")
    counts_lines = ", ".join(f"{name}: {count}" for name, count in counts.items() if count > 0)
    if counts_lines:
        lines.append(f"Reasons: {counts_lines}")
    return "\n".join(lines)


def _plot_eye_angle_dashboard(
    attrs: Dict[str, object],
    data: Dict[str, np.ndarray],
    title: Optional[str],
    angle_variant: Dict[str, Optional[np.ndarray]],
    variant_label: str,
) -> plt.Figure:
    roi = data["roi_angles"]
    qa = data["qa"]
    support = data["support"]
    frame = data["frame"]

    fps = attrs.get("fps")
    time_axis_raw = support.get("time_seconds")
    frame_indices = support.get("frame_indices")
    time_axis, time_label = _prepare_timeline_data(time_axis_raw, frame_indices, fps)

    left = angle_variant["left"]
    right = angle_variant["right"]
    vergence = angle_variant["vergence"]
    left_signed = angle_variant.get("left_signed")
    right_signed = angle_variant.get("right_signed")
    vergence_signed = angle_variant.get("vergence_signed")
    version = angle_variant.get("version")
    valid_detection = qa["valid_frame"]

    left_valid = left[np.isfinite(left)]
    right_valid = right[np.isfinite(right)]
    vergence_valid = vergence[np.isfinite(vergence)]
    vergence_signed_valid = (
        vergence_signed[np.isfinite(vergence_signed)] if isinstance(vergence_signed, np.ndarray) else np.array([])
    )
    version_valid = version[np.isfinite(version)] if isinstance(version, np.ndarray) else np.array([])

    reason_map_raw = attrs.get("reason_code_map", {}) or {}
    reason_map = {str(key): str(value) for key, value in reason_map_raw.items()}
    reason_counts = _reason_counts(qa["reason_codes"], reason_map) if reason_map else {}
    frame_valid = frame.get("frame_valid")

    time_ds = _downsample(time_axis, MAX_TIMELINE_POINTS) if time_axis.size else time_axis
    left_ds = _downsample(left, MAX_TIMELINE_POINTS)
    right_ds = _downsample(right, MAX_TIMELINE_POINTS)
    vergence_plot = vergence_signed if isinstance(vergence_signed, np.ndarray) else vergence
    vergence_plot_ds = _downsample(vergence_plot, MAX_TIMELINE_POINTS)

    ellipse_ratio = support.get("ellipse_ratio")
    ellipse_ratio_valid = ellipse_ratio[np.isfinite(ellipse_ratio)] if isinstance(ellipse_ratio, np.ndarray) else np.array([])

    threshold = _find_bimodal_valley(vergence_signed_valid) if vergence_signed_valid.size else np.nan

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.1, 1.0, 0.8, 1.1],
        hspace=0.35,
        wspace=0.3,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    if time_ds.size:
        ax_left.plot(time_ds, left_ds, color="#1f77b4", linewidth=0.8)
    else:
        ax_left.plot(left_ds, color="#1f77b4", linewidth=0.8)
    ax_left.set_title("Left Eye Over Time")
    ax_left.set_xlabel(time_label)
    ax_left.set_ylabel("Angle (deg)")
    ax_left.set_ylim(-5, 95)
    ax_left.grid(alpha=0.2)

    ax_right = fig.add_subplot(gs[0, 1])
    if time_ds.size:
        ax_right.plot(time_ds, right_ds, color="#ff7f0e", linewidth=0.8)
    else:
        ax_right.plot(right_ds, color="#ff7f0e", linewidth=0.8)
    ax_right.set_title("Right Eye Over Time")
    ax_right.set_xlabel(time_label)
    ax_right.set_ylabel("Angle (deg)")
    ax_right.set_ylim(-5, 95)
    ax_right.grid(alpha=0.2)

    ax_vergence = fig.add_subplot(gs[0, 2])
    plotted_series = False
    if vergence_plot_ds.size == 0:
        ax_vergence.text(0.5, 0.5, "No vergence data", ha="center", va="center")
    elif time_ds.size:
        ax_vergence.plot(time_ds, vergence_plot_ds, color="#2ca02c", linewidth=0.8, label="Vergence (signed)")
        plotted_series = True
    else:
        ax_vergence.plot(vergence_plot_ds, color="#2ca02c", linewidth=0.8, label="Vergence (signed)")
        plotted_series = True
    ax_vergence.set_title("Vergence (signed) Over Time")
    ax_vergence.set_xlabel(time_label)
    ax_vergence.set_ylabel("Angle (deg)")
    ax_vergence.set_ylim(-5, 95)
    if np.isfinite(threshold):
        ax_vergence.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0, label=f"threshold ≈ {threshold:.1f}°")
        plotted_series = True
    if plotted_series:
        ax_vergence.legend(loc="upper right")
    ax_vergence.grid(alpha=0.2)

    ax_scatter = fig.add_subplot(gs[1, 0])
    if isinstance(left_signed, np.ndarray) and isinstance(right_signed, np.ndarray):
        scatter_mask = np.isfinite(left_signed) & np.isfinite(right_signed)
        if scatter_mask.any():
            idx = np.flatnonzero(scatter_mask)
            if idx.size > SCATTER_DECIMATE:
                idx = idx[::SCATTER_DECIMATE]
            ax_scatter.scatter(
                left_signed[idx],
                right_signed[idx],
                s=6,
                alpha=0.25,
                color="#9467bd",
                edgecolors="none",
            )
            ax_scatter.plot([0, 90], [0, 90], "k--", linewidth=0.8)
        else:
            ax_scatter.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax_scatter.text(0.5, 0.5, "Signed angles unavailable", ha="center", va="center")
    ax_scatter.set_title("Left vs Right (signed)")
    ax_scatter.set_xlabel("Left (deg)")
    ax_scatter.set_ylabel("Right (deg)")
    ax_scatter.set_xlim(-5, 95)
    ax_scatter.set_ylim(-5, 95)
    ax_scatter.grid(alpha=0.2)

    ax_hist_lr = fig.add_subplot(gs[1, 1])
    if left_valid.size:
        ax_hist_lr.hist(left_valid, bins=60, alpha=0.6, label="Left", color="#1f77b4")
    if right_valid.size:
        ax_hist_lr.hist(right_valid, bins=60, alpha=0.6, label="Right", color="#ff7f0e")
    ax_hist_lr.set_title("Left vs Right Angle Distribution")
    ax_hist_lr.set_xlabel("Angle (deg)")
    ax_hist_lr.set_ylabel("Count")

    ax_hist_v = fig.add_subplot(gs[1, 2])
    if vergence_valid.size:
        ax_hist_v.hist(vergence_valid, bins=60, color="#2ca02c", alpha=0.5, label="Unsigned")
    if vergence_signed_valid.size:
        ax_hist_v.hist(
            vergence_signed_valid,
            bins=60,
            histtype="step",
            color="#d62728",
            linewidth=1.4,
            label="Signed",
        )
    if np.isfinite(threshold):
        ax_hist_v.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"threshold ≈ {threshold:.1f}°")
    ax_hist_v.set_title("Vergence Distribution")
    ax_hist_v.set_xlabel("Angle (deg)")
    ax_hist_v.set_ylabel("Count")
    handles, labels = ax_hist_v.get_legend_handles_labels()
    if handles:
        ax_hist_v.legend(loc="upper right")
    ax_hist_v.grid(alpha=0.2)

    ax_version = fig.add_subplot(gs[2, 0])
    if version_valid.size:
        ax_version.hist(version_valid, bins=60, color="#8c564b", alpha=0.8)
    ax_version.set_title("Version Distribution")
    ax_version.set_xlabel("Angle (deg)")
    ax_version.set_ylabel("Count")
    ax_version.grid(alpha=0.2)

    ax_reasons = fig.add_subplot(gs[2, 1])
    if reason_counts:
        labels = list(reason_counts.keys())
        counts = [reason_counts[label] for label in labels]
        positions = np.arange(len(labels))
        ax_reasons.bar(positions, counts, color="#9467bd")
        ax_reasons.set_ylabel("Count")
        ax_reasons.set_xticks(positions)
        ax_reasons.set_xticklabels(labels, rotation=20, ha="right")
    else:
        ax_reasons.text(0.5, 0.5, "No reason codes available", ha="center", va="center")
    ax_reasons.set_title("Invalid Reason Counts")
    ax_reasons.grid(axis="y", alpha=0.2)

    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis("off")
    summary_text = _format_summary_lines(attrs, reason_counts, valid_detection, frame_valid)
    extra_lines: List[str] = [f"Angle source: {variant_label}"]
    if np.isfinite(threshold):
        extra_lines.append(f"Convergence threshold ≈ {threshold:.1f}°")
    if ellipse_ratio_valid.size:
        extra_lines.append(
            f"Ellipse ratio mean±std: {float(np.mean(ellipse_ratio_valid)):.2f} ± {float(np.std(ellipse_ratio_valid)):.2f}"
        )
    if extra_lines:
        summary_text = summary_text + "\n" + "\n".join(extra_lines)
    ax_summary.text(0.0, 0.95, summary_text, ha="left", va="top", fontsize=10, family="monospace")

    ax_hex = fig.add_subplot(gs[3, :])
    if vergence_signed_valid.size and version_valid.size and (np.isfinite(vergence_signed_valid) & np.isfinite(version_valid)).any():
        mask_hex = np.isfinite(vergence_signed) & np.isfinite(version)
        verg_valid = vergence_signed[mask_hex]
        vers_valid = version[mask_hex]
        if verg_valid.size:
            verg_clamped = np.clip(verg_valid, -90.0, 90.0)
            vers_clamped = np.clip(vers_valid, -90.0, 90.0)
            hb = ax_hex.hexbin(
                verg_clamped,
                vers_clamped,
                gridsize=80,
                extent=[-90, 90, -90, 90],
                cmap="viridis",
                mincnt=1,
            )
            fig.colorbar(hb, ax=ax_hex, label="Count", fraction=0.046, pad=0.04)
            ax_hex.set_xlim(-90, 90)
            ax_hex.set_ylim(-90, 90)
            ax_hex.axhline(0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
            if np.isfinite(threshold):
                ax_hex.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.9)
            v0 = 5.0
            pct_converged = float(np.mean(verg_valid >= threshold) * 100.0) if np.isfinite(threshold) else np.nan
            pct_symmetric = float(np.mean(np.abs(vers_valid) <= v0) * 100.0)
            stats_lines = []
            if np.isfinite(pct_converged):
                stats_lines.append(f"% converged ≥ {threshold:.1f}°: {pct_converged:.1f}%")
            else:
                stats_lines.append("% converged: n/a")
            stats_lines.append(f"% symmetric (|version| ≤ {v0:.1f}°): {pct_symmetric:.1f}%")
            if np.isfinite(threshold):
                q_labels = ["UL", "UR", "LL", "LR"]
                quadrants = {
                    "UL": np.sum((verg_valid < threshold) & (vers_valid >= 0)),
                    "UR": np.sum((verg_valid >= threshold) & (vers_valid >= 0)),
                    "LL": np.sum((verg_valid < threshold) & (vers_valid < 0)),
                    "LR": np.sum((verg_valid >= threshold) & (vers_valid < 0)),
                }
                total_q = sum(quadrants.values())
                quad_text = ", ".join(f"{label}: {count} ({(count/total_q*100.0):.1f}%)" for label, count in quadrants.items() if total_q)
                stats_lines.append(f"Quadrants: {quad_text}")
            ax_hex.text(
                0.02,
                0.98,
                "\n".join(stats_lines),
                transform=ax_hex.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
    else:
        ax_hex.text(0.5, 0.5, "Vergence/version data unavailable", ha="center", va="center")
        ax_hex.set_xlim(-90, 90)
        ax_hex.set_ylim(-90, 90)
    ax_hex.set_title("Vergence × Version")
    ax_hex.set_xlabel("Vergence (signed, deg)")
    ax_hex.set_ylabel("Version (deg)")
    ax_hex.grid(alpha=0.2, linestyle=":", linewidth=0.5)

    if title:
        fig_title = title
    else:
        fig_title = f"Eye Angle Summary ({attrs.get('run_name', 'latest')}) – {variant_label}"
    fig.suptitle(fig_title, fontsize=16, fontweight="bold", y=0.98)
    provenance = attrs.get("provenance", {})
    git_info = provenance.get("git", {})
    footer_lines = [
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Script: fisheye.visualization.visualize_eye_angles",
    ]
    if git_info:
        footer_lines.append(f"Data commit: {git_info.get('commit_hash', git_info.get('short_hash', 'unknown'))}")
    fig.text(0.01, 0.01, "\n".join(footer_lines), fontsize=8, va="bottom", ha="left")
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize eye angle runs stored in analysis/eye_angle_runs.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--run",
        type=str,
        help="Specific eye_angle_runs/<name> to load (default: latest).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional PNG output path (default: auto-generated in visualization/plots).",
    )
    parser.add_argument(
        "--angle-source",
        choices=["ellipse", "minor", "feret_major", "feret_minor"],
        default="ellipse",
        help="Which angle series to treat as primary (default: ellipse).",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Render dashboards for every angle variant (ellipse, minor, feret_major, feret_minor).",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Override figure title.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate plot without displaying.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status messages.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    attrs, data = _load_eye_angle_run(args.zarr_path, args.run)
    if not args.quiet:
        print(f"Loaded eye angle run '{attrs['run_name']}' from {args.zarr_path}")

    requested_sources = (
        ["ellipse", "minor", "feret_major", "feret_minor"]
        if args.all_variants
        else [args.angle_source]
    )

    figs: List[plt.Figure] = []

    for angle_source in requested_sources:
        try:
            angle_variant, variant_label = _select_angle_variant(data["roi_angles"], angle_source)
        except ValueError as exc:
            if angle_source != "ellipse":
                if not args.quiet:
                    print(f"Warning: {exc} Skipping variant '{angle_source}'.")
                continue
            raise

        fig = _plot_eye_angle_dashboard(attrs, data, args.title, angle_variant, variant_label)
        figs.append(fig)

        provenance = attrs.get("provenance", {})
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_zarr": str(args.zarr_path),
            "run_name": attrs.get("run_name"),
            "provenance": provenance,
            "angle_source": angle_source,
            "angle_source_label": variant_label,
        }

        if args.output:
            output_path = Path(args.output)
            if args.all_variants:
                base = output_path.stem
                output_path = output_path.with_name(f"{base}_{angle_source}{output_path.suffix}")
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = (
                DEFAULT_OUTPUT_DIR
                / f"eye_angle_dashboard_{attrs.get('run_name', 'latest')}_{angle_source}_{timestamp}.png"
            )

        _save_with_metadata(fig, output_path, metadata)
        if not args.quiet:
            print(f"Saved dashboard ({angle_source}) to {output_path}")

        if args.no_show:
            plt.close(fig)

    if not figs:
        if not args.quiet:
            print("No figures generated.")
        return 1

    if not args.no_show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
