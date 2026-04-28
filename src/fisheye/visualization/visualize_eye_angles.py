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

from fisheye.utils.zarr_io import open_zarr_root

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
    root = open_zarr_root(zarr_path, mode="r")
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
        "left_smoothed": _maybe("left_deg_smoothed"),
        "right": roi_group["right_deg"][:],
        "right_smoothed": _maybe("right_deg_smoothed"),
        "vergence": roi_group["vergence_deg"][:],
        "vergence_smoothed": _maybe("vergence_deg_smoothed"),
        "left_signed": _maybe("left_signed_deg"),
        "left_signed_smoothed": _maybe("left_signed_deg_smoothed"),
        "right_signed": _maybe("right_signed_deg"),
        "right_signed_smoothed": _maybe("right_signed_deg_smoothed"),
        "vergence_signed": _maybe("vergence_signed_deg"),
        "vergence_signed_smoothed": _maybe("vergence_signed_deg_smoothed"),
        "version": _maybe("version_deg"),
        "version_smoothed": _maybe("version_deg_smoothed"),
        "left_minor_signed": _maybe("left_minor_signed_deg"),
        "left_minor_signed_smoothed": _maybe("left_minor_signed_deg_smoothed"),
        "right_minor_signed": _maybe("right_minor_signed_deg"),
        "right_minor_signed_smoothed": _maybe("right_minor_signed_deg_smoothed"),
        "vergence_minor_signed": _maybe("vergence_minor_signed_deg"),
        "vergence_minor_signed_smoothed": _maybe("vergence_minor_signed_deg_smoothed"),
        "version_minor": _maybe("version_minor_deg"),
        "version_minor_smoothed": _maybe("version_minor_deg_smoothed"),
        "left_speed": _maybe("left_speed_deg_s"),
        "right_speed": _maybe("right_speed_deg_s"),
        "vergence_speed": _maybe("vergence_speed_deg_s"),
        "vergence_signed_speed": _maybe("vergence_signed_speed_deg_s"),
        "version_speed": _maybe("version_speed_deg_s"),
        # Centroid-based angles (paper-comparable)
        "left_centroid": _maybe("left_centroid_deg"),
        "left_centroid_smoothed": _maybe("left_centroid_deg_smoothed"),
        "right_centroid": _maybe("right_centroid_deg"),
        "right_centroid_smoothed": _maybe("right_centroid_deg_smoothed"),
        "vergence_centroid": _maybe("vergence_centroid_deg"),
        "vergence_centroid_smoothed": _maybe("vergence_centroid_deg_smoothed"),
    }

    roi_deltas = {
        "left": _maybe("left_delta_deg"),
        "left_smoothed": _maybe("left_delta_deg_smoothed"),
        "right": _maybe("right_delta_deg"),
        "right_smoothed": _maybe("right_delta_deg_smoothed"),
        "vergence": _maybe("vergence_delta_deg"),
        "vergence_smoothed": _maybe("vergence_delta_deg_smoothed"),
        "left_signed": _maybe("left_signed_delta_deg"),
        "left_signed_smoothed": _maybe("left_signed_delta_deg_smoothed"),
        "right_signed": _maybe("right_signed_delta_deg"),
        "right_signed_smoothed": _maybe("right_signed_delta_deg_smoothed"),
        "vergence_signed": _maybe("vergence_signed_delta_deg"),
        "vergence_signed_smoothed": _maybe("vergence_signed_delta_deg_smoothed"),
        "version": _maybe("version_delta_deg"),
        "version_smoothed": _maybe("version_delta_deg_smoothed"),
        "left_minor_signed": _maybe("left_minor_signed_delta_deg"),
        "left_minor_signed_smoothed": _maybe("left_minor_signed_delta_deg_smoothed"),
        "right_minor_signed": _maybe("right_minor_signed_delta_deg"),
        "right_minor_signed_smoothed": _maybe("right_minor_signed_delta_deg_smoothed"),
        "vergence_minor_signed": _maybe("vergence_minor_signed_delta_deg"),
        "vergence_minor_signed_smoothed": _maybe("vergence_minor_signed_delta_deg_smoothed"),
        "version_minor": _maybe("version_minor_delta_deg"),
        "version_minor_smoothed": _maybe("version_minor_delta_deg_smoothed"),
        # Centroid-based deltas
        "left_centroid": _maybe("left_centroid_delta_deg"),
        "left_centroid_smoothed": _maybe("left_centroid_delta_deg_smoothed"),
        "right_centroid": _maybe("right_centroid_delta_deg"),
        "right_centroid_smoothed": _maybe("right_centroid_delta_deg_smoothed"),
        "vergence_centroid": _maybe("vergence_centroid_delta_deg"),
        "vergence_centroid_smoothed": _maybe("vergence_centroid_delta_deg_smoothed"),
    }

    qa_roi = {
        "valid_left": qa["roi"]["valid_left"][:],
        "valid_right": qa["roi"]["valid_right"][:],
        "valid_frame": qa["roi"]["valid_frame"][:],
        "reason_codes": qa["roi"]["reason_codes"][:],
    }

    frame_data: Dict[str, Optional[np.ndarray]] = {}
    frame_deltas: Dict[str, Optional[np.ndarray]] = {}
    if "frame" in angles:
        frame_group = angles["frame"]
        def _frame_maybe(name: str) -> Optional[np.ndarray]:
            return frame_group[name][:] if name in frame_group else None
        frame_data["left"] = frame_group["left_deg"][:]
        frame_data["left_smoothed"] = _frame_maybe("left_deg_smoothed")
        frame_data["right"] = frame_group["right_deg"][:]
        frame_data["right_smoothed"] = _frame_maybe("right_deg_smoothed")
        frame_data["vergence"] = frame_group["vergence_deg"][:]
        frame_data["vergence_smoothed"] = _frame_maybe("vergence_deg_smoothed")
        frame_data["vergence_signed"] = _frame_maybe("vergence_signed_deg")
        frame_data["vergence_signed_smoothed"] = _frame_maybe("vergence_signed_deg_smoothed")
        frame_data["vergence_minor_signed"] = _frame_maybe("vergence_minor_signed_deg")
        frame_data["vergence_minor_signed_smoothed"] = _frame_maybe("vergence_minor_signed_deg_smoothed")
        frame_data["version"] = _frame_maybe("version_deg")
        frame_data["version_smoothed"] = _frame_maybe("version_deg_smoothed")
        frame_data["version_minor"] = _frame_maybe("version_minor_deg")
        frame_data["version_minor_smoothed"] = _frame_maybe("version_minor_deg_smoothed")
        frame_deltas["left"] = _frame_maybe("left_delta_deg")
        frame_deltas["left_smoothed"] = _frame_maybe("left_delta_deg_smoothed")
        frame_deltas["right"] = _frame_maybe("right_delta_deg")
        frame_deltas["right_smoothed"] = _frame_maybe("right_delta_deg_smoothed")
        frame_deltas["vergence"] = _frame_maybe("vergence_delta_deg")
        frame_deltas["vergence_smoothed"] = _frame_maybe("vergence_delta_deg_smoothed")
        frame_deltas["vergence_signed"] = _frame_maybe("vergence_signed_delta_deg")
        frame_deltas["vergence_signed_smoothed"] = _frame_maybe("vergence_signed_delta_deg_smoothed")
        frame_deltas["vergence_minor_signed"] = _frame_maybe("vergence_minor_signed_delta_deg")
        frame_deltas["vergence_minor_signed_smoothed"] = _frame_maybe("vergence_minor_signed_delta_deg_smoothed")
        frame_deltas["version"] = _frame_maybe("version_delta_deg")
        frame_deltas["version_smoothed"] = _frame_maybe("version_delta_deg_smoothed")
        frame_deltas["version_minor"] = _frame_maybe("version_minor_delta_deg")
        frame_deltas["version_minor_smoothed"] = _frame_maybe("version_minor_delta_deg_smoothed")
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
        "roi_deltas": roi_deltas,
        "qa": qa_roi,
        "frame": frame_data,
        "frame_deltas": frame_deltas,
        "support": support_data,
    }
    return attrs, data


def _eye_angle_contract_metadata(attrs: Dict[str, object]) -> Dict[str, object]:
    """Return stable contract/source metadata for summaries and PNG XMP."""

    return {
        "schema_id": attrs.get("schema_id"),
        "schema_version": attrs.get("schema_version"),
        "method": attrs.get("method"),
        "row_axis": attrs.get("row_axis"),
        "source_geometry_kind": attrs.get("source_geometry_kind"),
        "source_eye_geometry_stage": attrs.get("source_eye_geometry_stage"),
        "source_eye_geometry_run": attrs.get("source_eye_geometry_run"),
        "source_subject_shape_run": attrs.get("source_subject_shape_run"),
        "source_refined_subject_masks_run": attrs.get("source_refined_subject_masks_run"),
        "source_refined_eye_run": attrs.get("source_refined_eye_run"),
        "source_keypoints_run": attrs.get("source_keypoints_run") or attrs.get("source_keypoint_run"),
        "eye_angle_output_schema": attrs.get("eye_angle_output_schema"),
    }


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
    roi_deltas: Dict[str, Optional[np.ndarray]],
    source: str,
) -> Tuple[Dict[str, Optional[np.ndarray]], str, Dict[str, Dict[str, object]]]:
    """Return the angle series to visualize along with a human-friendly label."""

    deltas: Dict[str, Optional[np.ndarray]] = {}
    smoothed_flags: Dict[str, bool] = {}
    series_lookup: Dict[str, str] = {}
    used_smoothed: set[str] = set()

    def _delta_for(name: str, use_smoothed: bool) -> Optional[np.ndarray]:
        key = f"{name}_smoothed" if use_smoothed else name
        arr = roi_deltas.get(key)
        if isinstance(arr, np.ndarray):
            return arr
        fallback_key = name if use_smoothed else f"{name}_smoothed"
        arr_fallback = roi_deltas.get(fallback_key)
        if isinstance(arr_fallback, np.ndarray):
            return arr_fallback
        return None

    def _pick(name: str, *, required: bool = True) -> Optional[np.ndarray]:
        smoothed = roi_angles.get(f"{name}_smoothed")
        if isinstance(smoothed, np.ndarray):
            used_smoothed.add(name)
            smoothed_flags[name] = True
            deltas[name] = _delta_for(name, True)
            return smoothed
        arr = roi_angles.get(name)
        if required and not isinstance(arr, np.ndarray):
            raise ValueError(f"Dataset '{name}' required for angle source '{source}' is missing.")
        smoothed_flags[name] = False
        if isinstance(arr, np.ndarray):
            deltas[name] = _delta_for(name, False)
        else:
            deltas[name] = None
        return arr  # may be None if not required

    source = source.lower()
    if source == "ellipse":
        variant = {
            "left": _pick("left"),
            "right": _pick("right"),
            "vergence": _pick("vergence"),
            "left_signed": _pick("left_signed", required=False),
            "right_signed": _pick("right_signed", required=False),
            "vergence_signed": _pick("vergence_signed", required=False),
            "version": _pick("version", required=False),
        }
        series_lookup.update(
            {
                "left": "left",
                "right": "right",
                "vergence": "vergence",
                "left_signed": "left_signed",
                "right_signed": "right_signed",
                "vergence_signed": "vergence_signed",
                "version": "version",
            }
        )
        label = "Ellipse major axis"
    elif source == "minor":
        left_signed = _pick("left_minor_signed")
        right_signed = _pick("right_minor_signed")
        vergence_signed = _pick("vergence_minor_signed")
        variant = {
            "left": np.abs(left_signed),
            "right": np.abs(right_signed),
            "vergence": np.abs(vergence_signed),
            "left_signed": left_signed,
            "right_signed": right_signed,
            "vergence_signed": vergence_signed,
            "version": _pick("version_minor", required=False),
        }
        series_lookup.update(
            {
                "left": "left_minor_signed",
                "right": "right_minor_signed",
                "vergence": "vergence_minor_signed",
                "left_signed": "left_minor_signed",
                "right_signed": "right_minor_signed",
                "vergence_signed": "vergence_minor_signed",
                "version": "version_minor",
            }
        )
        label = "Ellipse minor axis"
    elif source == "centroid":
        # Centroid-based angles measure eye position (not orientation) in fish frame
        left_centroid = _pick("left_centroid")
        right_centroid = _pick("right_centroid")
        vergence_centroid = _pick("vergence_centroid")
        variant = {
            "left": np.abs(left_centroid),
            "right": np.abs(right_centroid),
            "vergence": vergence_centroid,  # already |L| + |R|
            "left_signed": left_centroid,
            "right_signed": right_centroid,
            "vergence_signed": vergence_centroid,
            "version": None,  # not computed for centroid method
        }
        series_lookup.update(
            {
                "left": "left_centroid",
                "right": "right_centroid",
                "vergence": "vergence_centroid",
                "left_signed": "left_centroid",
                "right_signed": "right_centroid",
                "vergence_signed": "vergence_centroid",
                "version": None,
            }
        )
        label = "Centroid position (paper)"
    else:
        raise ValueError(f"Unknown angle source '{source}'.")

    if used_smoothed:
        label = f"{label} (smoothed)"

    variant_meta: Dict[str, Dict[str, object]] = {
        "deltas": deltas,
        "smoothed": smoothed_flags,
        "series_lookup": series_lookup,
    }

    return variant, label, variant_meta


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
    schema_id = attrs.get("schema_id")
    schema_version = attrs.get("schema_version")
    if schema_id:
        schema_text = str(schema_id)
        if schema_version is not None:
            schema_text = f"{schema_text} v{schema_version}"
        lines.append(f"Schema: {schema_text}")
    method = attrs.get("method")
    if method:
        lines.append(f"Method: {method}")
    geometry_kind = attrs.get("source_geometry_kind")
    geometry_stage = attrs.get("source_eye_geometry_stage")
    geometry_run = attrs.get("source_eye_geometry_run")
    if geometry_kind or geometry_stage or geometry_run:
        geometry_desc = str(geometry_kind or "unknown_eye_geometry")
        if geometry_stage or geometry_run:
            geometry_desc += f" ({geometry_stage or '?'} / {geometry_run or '?'})"
        lines.append(f"Eye geometry: {geometry_desc}")
    source_keypoints = attrs.get("source_keypoints_run") or attrs.get("source_keypoint_run")
    if source_keypoints:
        lines.append(f"Keypoints: {source_keypoints}")
    fps = attrs.get("fps")
    if fps:
        lines.append(f"FPS: {fps:.3f}")
    smoothing_method = attrs.get("angle_smoothing_method")
    if smoothing_method:
        det_window = attrs.get("angle_smoothing_window_detections")
        frame_window = attrs.get("angle_smoothing_window_frames")
        window_parts = []
        if det_window:
            window_parts.append(f"detections={det_window}")
        if frame_window:
            window_parts.append(f"frames={frame_window}")
        window_desc = f" ({', '.join(window_parts)})" if window_parts else ""
        lines.append(f"Smoothing: {smoothing_method}{window_desc}")
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
    variant_meta: Dict[str, Dict[str, object]],
) -> plt.Figure:
    roi = data["roi_angles"]
    qa = data["qa"]
    support = data["support"]
    frame = data["frame"]
    frame_deltas = data.get("frame_deltas", {})

    delta_meta = variant_meta.get("deltas", {})
    smoothed_flags = variant_meta.get("smoothed", {})
    series_lookup = variant_meta.get("series_lookup", {})

    fps = attrs.get("fps")
    time_axis_raw = support.get("time_seconds")
    frame_indices = support.get("frame_indices")
    time_axis, time_label = _prepare_timeline_data(time_axis_raw, frame_indices, fps)
    frame_time_axis_raw = support.get("frame_time_seconds")
    frame_time_axis, frame_time_label = _prepare_timeline_data(frame_time_axis_raw, frame_indices, fps)

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
    frame_time_ds = _downsample(frame_time_axis, MAX_TIMELINE_POINTS) if frame_time_axis.size else frame_time_axis
    left_ds = _downsample(left, MAX_TIMELINE_POINTS)
    right_ds = _downsample(right, MAX_TIMELINE_POINTS)
    vergence_plot = vergence_signed if isinstance(vergence_signed, np.ndarray) else vergence
    vergence_plot_ds = _downsample(vergence_plot, MAX_TIMELINE_POINTS)

    ellipse_ratio = support.get("ellipse_ratio")
    ellipse_ratio_valid = ellipse_ratio[np.isfinite(ellipse_ratio)] if isinstance(ellipse_ratio, np.ndarray) else np.array([])

    threshold = _find_bimodal_valley(vergence_signed_valid) if vergence_signed_valid.size else np.nan

    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(
        5,
        3,
        figure=fig,
        height_ratios=[1.1, 0.6, 1.0, 0.8, 1.1],
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
    ax_left.set_ylim(-5, 185)
    ax_left.grid(alpha=0.2)

    ax_right = fig.add_subplot(gs[0, 1])
    if time_ds.size:
        ax_right.plot(time_ds, right_ds, color="#ff7f0e", linewidth=0.8)
    else:
        ax_right.plot(right_ds, color="#ff7f0e", linewidth=0.8)
    ax_right.set_title("Right Eye Over Time")
    ax_right.set_xlabel(time_label)
    ax_right.set_ylabel("Angle (deg)")
    ax_right.set_ylim(-5, 185)
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
    ax_vergence.set_ylim(-5, 185)
    if np.isfinite(threshold):
        ax_vergence.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0, label=f"threshold ≈ {threshold:.1f}°")
        plotted_series = True
    if plotted_series:
        ax_vergence.legend(loc="upper right")
    ax_vergence.grid(alpha=0.2)

    ax_delta_det = fig.add_subplot(gs[1, :2])
    ax_delta_frame = fig.add_subplot(gs[1, 2])

    detection_delta_series: List[Tuple[str, np.ndarray, str]] = []
    detection_delta_stats: Dict[str, np.ndarray] = {}
    frame_delta_series: List[Tuple[str, np.ndarray, str]] = []
    frame_delta_stats: Dict[str, np.ndarray] = {}

    def _maybe_add_detection_delta(label: str, variant_key: str, color: str) -> None:
        dataset_key = series_lookup.get(variant_key)
        if not dataset_key:
            return
        arr = delta_meta.get(dataset_key)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return
        detection_delta_series.append((label, arr, color))
        finite = arr[np.isfinite(arr)]
        if finite.size:
            detection_delta_stats[label] = finite

    def _get_frame_delta(dataset_key: Optional[str]) -> Optional[np.ndarray]:
        if not dataset_key:
            return None
        use_smoothed = bool(smoothed_flags.get(dataset_key))
        keys_to_try = []
        if use_smoothed:
            keys_to_try.append(f"{dataset_key}_smoothed")
        keys_to_try.append(dataset_key)
        if not use_smoothed:
            keys_to_try.append(f"{dataset_key}_smoothed")
        for key in keys_to_try:
            arr = frame_deltas.get(key)
            if isinstance(arr, np.ndarray) and arr.size:
                return arr
        return None

    def _maybe_add_frame_delta(label: str, variant_key: str, color: str) -> None:
        dataset_key = series_lookup.get(variant_key)
        arr = _get_frame_delta(dataset_key)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return
        frame_delta_series.append((label, arr, color))
        finite = arr[np.isfinite(arr)]
        if finite.size:
            frame_delta_stats[label] = finite

    _maybe_add_detection_delta("Left Δ", "left", "#1f77b4")
    _maybe_add_detection_delta("Right Δ", "right", "#ff7f0e")
    vergence_label = "Vergence (signed) Δ" if isinstance(vergence_signed, np.ndarray) else "Vergence Δ"
    vergence_key = "vergence_signed" if isinstance(vergence_signed, np.ndarray) else "vergence"
    _maybe_add_detection_delta(vergence_label, vergence_key, "#2ca02c")

    _maybe_add_frame_delta("Left Δ", "left", "#1f77b4")
    _maybe_add_frame_delta("Right Δ", "right", "#ff7f0e")
    _maybe_add_frame_delta(vergence_label, vergence_key, "#2ca02c")

    if detection_delta_series:
        for label, series, color in detection_delta_series:
            series_ds = _downsample(series, MAX_TIMELINE_POINTS)
            if time_ds.size:
                x = time_ds
                if x.size != series_ds.size:
                    count = min(x.size, series_ds.size)
                    ax_delta_det.plot(x[:count], series_ds[:count], label=label, color=color, linewidth=0.8)
                else:
                    ax_delta_det.plot(x, series_ds, label=label, color=color, linewidth=0.8)
            else:
                x = np.arange(series_ds.size)
                ax_delta_det.plot(x, series_ds, label=label, color=color, linewidth=0.8)
        ax_delta_det.legend(loc="upper right")
    else:
        ax_delta_det.text(0.5, 0.5, "Delta data unavailable", ha="center", va="center")
    ax_delta_det.set_title("Detection Δ Angles")
    ax_delta_det.set_xlabel(time_label if time_ds.size else "Detection Index")
    ax_delta_det.set_ylabel("Δ Angle (deg)")
    ax_delta_det.grid(alpha=0.2)

    if frame_delta_series:
        for label, series, color in frame_delta_series:
            series_ds = _downsample(series, MAX_TIMELINE_POINTS)
            if frame_time_ds.size:
                x = frame_time_ds
                if x.size != series_ds.size:
                    count = min(x.size, series_ds.size)
                    ax_delta_frame.plot(x[:count], series_ds[:count], label=label, color=color, linewidth=0.8)
                else:
                    ax_delta_frame.plot(x, series_ds, label=label, color=color, linewidth=0.8)
            else:
                x = np.arange(series_ds.size)
                ax_delta_frame.plot(x, series_ds, label=label, color=color, linewidth=0.8)
        ax_delta_frame.legend(loc="upper right")
    else:
        ax_delta_frame.text(0.5, 0.5, "Frame delta data unavailable", ha="center", va="center")
    ax_delta_frame.set_title("Frame Δ Angles")
    ax_delta_frame.set_xlabel(frame_time_label if frame_time_ds.size else "Frame Index")
    ax_delta_frame.set_ylabel("Δ Angle (deg)")
    ax_delta_frame.grid(alpha=0.2)

    ax_scatter = fig.add_subplot(gs[2, 0])
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
            ax_scatter.plot([-180, 180], [-180, 180], "k--", linewidth=0.8)
        else:
            ax_scatter.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax_scatter.text(0.5, 0.5, "Signed angles unavailable", ha="center", va="center")
    ax_scatter.set_title("Left vs Right (signed)")
    ax_scatter.set_xlabel("Left (deg)")
    ax_scatter.set_ylabel("Right (deg)")
    ax_scatter.set_xlim(-185, 185)
    ax_scatter.set_ylim(-185, 185)
    ax_scatter.grid(alpha=0.2)

    ax_hist_lr = fig.add_subplot(gs[2, 1])
    if left_valid.size:
        ax_hist_lr.hist(left_valid, bins=60, alpha=0.6, label="Left", color="#1f77b4")
    if right_valid.size:
        ax_hist_lr.hist(right_valid, bins=60, alpha=0.6, label="Right", color="#ff7f0e")
    ax_hist_lr.set_title("Left vs Right Angle Distribution")
    ax_hist_lr.set_xlabel("Angle (deg)")
    ax_hist_lr.set_ylabel("Count")

    ax_hist_v = fig.add_subplot(gs[2, 2])
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

    ax_version = fig.add_subplot(gs[3, 0])
    if version_valid.size:
        ax_version.hist(version_valid, bins=60, color="#8c564b", alpha=0.8)
    ax_version.set_title("Version Distribution")
    ax_version.set_xlabel("Angle (deg)")
    ax_version.set_ylabel("Count")
    ax_version.grid(alpha=0.2)

    ax_reasons = fig.add_subplot(gs[3, 1])
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

    ax_summary = fig.add_subplot(gs[3, 2])
    ax_summary.axis("off")
    summary_text = _format_summary_lines(attrs, reason_counts, valid_detection, frame_valid)
    extra_lines: List[str] = [f"Angle source: {variant_label}"]
    if np.isfinite(threshold):
        extra_lines.append(f"Convergence threshold ≈ {threshold:.1f}°")
    if ellipse_ratio_valid.size:
        extra_lines.append(
            f"Ellipse ratio mean±std: {float(np.mean(ellipse_ratio_valid)):.2f} ± {float(np.std(ellipse_ratio_valid)):.2f}"
        )
    def _format_delta_summary(stats: Dict[str, np.ndarray]) -> Optional[str]:
        if not stats:
            return None
        parts: List[str] = []
        for name, values in stats.items():
            mean_val = float(np.mean(values))
            max_val = float(np.max(values))
            clean_name = name.replace(" Δ", "")
            parts.append(f"{clean_name}: μ={mean_val:.2f}°, max={max_val:.2f}°")
        return "; ".join(parts)

    detection_summary = _format_delta_summary(detection_delta_stats)
    if detection_summary:
        extra_lines.append(f"Detection Δ mean/max: {detection_summary}")
    frame_summary = _format_delta_summary(frame_delta_stats)
    if frame_summary:
        extra_lines.append(f"Frame Δ mean/max: {frame_summary}")
    if extra_lines:
        summary_text = summary_text + "\n" + "\n".join(extra_lines)
    ax_summary.text(0.0, 0.95, summary_text, ha="left", va="top", fontsize=10, family="monospace")

    ax_hex = fig.add_subplot(gs[4, :])
    if vergence_signed_valid.size and version_valid.size and (np.isfinite(vergence_signed_valid) & np.isfinite(version_valid)).any():
        mask_hex = np.isfinite(vergence_signed) & np.isfinite(version)
        verg_valid = vergence_signed[mask_hex]
        vers_valid = version[mask_hex]
        if verg_valid.size:
            verg_clamped = np.clip(verg_valid, -180.0, 180.0)
            vers_clamped = np.clip(vers_valid, -180.0, 180.0)
            hb = ax_hex.hexbin(
                verg_clamped,
                vers_clamped,
                gridsize=80,
                extent=[-180, 180, -180, 180],
                cmap="viridis",
                mincnt=1,
            )
            fig.colorbar(hb, ax=ax_hex, label="Count", fraction=0.046, pad=0.04)
            ax_hex.set_xlim(-180, 180)
            ax_hex.set_ylim(-180, 180)
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
        ax_hex.set_xlim(-180, 180)
        ax_hex.set_ylim(-180, 180)
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
    parser = argparse.ArgumentParser(
        description="Visualize eye angle runs stored in analysis/eye_angle_runs, including per-step delta timelines when available.",
        epilog="Delta plots automatically follow the selected smoothed/raw variant; legacy runs without *_delta datasets skip the panels.",
    )
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
        choices=["ellipse", "minor", "centroid"],
        default="ellipse",
        help="Which angle series to treat as primary (default: ellipse).",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Render dashboards for every angle variant (ellipse, minor, centroid).",
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
        ["ellipse", "minor", "centroid"]
        if args.all_variants
        else [args.angle_source]
    )

    figs: List[plt.Figure] = []

    for angle_source in requested_sources:
        try:
            angle_variant, variant_label, variant_meta = _select_angle_variant(
                data["roi_angles"],
                data.get("roi_deltas", {}),
                angle_source,
            )
        except ValueError as exc:
            if angle_source != "ellipse":
                if not args.quiet:
                    print(f"Warning: {exc} Skipping variant '{angle_source}'.")
                continue
            raise

        fig = _plot_eye_angle_dashboard(attrs, data, args.title, angle_variant, variant_label, variant_meta)
        figs.append(fig)

        provenance = attrs.get("provenance", {})
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_zarr": str(args.zarr_path),
            "run_name": attrs.get("run_name"),
            "provenance": provenance,
            "angle_source": angle_source,
            "angle_source_label": variant_label,
            "eye_angle_contract": _eye_angle_contract_metadata(attrs),
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
