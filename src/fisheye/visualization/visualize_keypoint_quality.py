#!/usr/bin/env python3
"""Render keypoint refinement quality overview artifacts."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels


REFINED_PARENT_NAMES = ("refined_keypoints_runs", "keypoints_refined_runs")
QUALITY_ARTIFACT_NAME = "keypoint_quality_overview_png"
REFINEMENT_PIPELINE_ARTIFACT_NAME = "keypoint_refinement_pipeline_overview_png"


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _safe_mapping(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_summary_statistics(value: object) -> Dict[str, object]:
    """Normalize legacy/nested summary payloads to a flat dict.

    Some refined keypoint runs persist summary_statistics as:
      {"refine": {...}, "postprocess": {...}}
    while newer runs may store a flat dict directly.
    """
    payload = _safe_mapping(value)
    if not payload:
        return {}

    refine = _safe_mapping(payload.get("refine"))
    postprocess = _safe_mapping(payload.get("postprocess"))
    if not refine and not postprocess:
        return payload

    merged: Dict[str, object] = {}
    merged.update(refine)
    merged.update(postprocess)
    return merged


def _review_timestamp(review: Mapping[str, object]) -> Optional[str]:
    for key in ("timestamp_utc", "timestamp", "reviewed_at_utc", "reviewed_at", "updated_utc"):
        text = _decode_text(review.get(key))
        if text:
            return text
    return None


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_refined_parent(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    for parent_name in REFINED_PARENT_NAMES:
        if parent_name in root:
            return parent_name, root[parent_name]
    return None, None


def _resolve_refined_run(
    parent_group: zarr.Group,
    requested_refined_run: Optional[str],
    *,
    allow_legacy_latest_fallback: bool = False,
) -> Optional[str]:
    if requested_refined_run:
        return requested_refined_run if requested_refined_run in parent_group else None
    if not allow_legacy_latest_fallback:
        raise ValueError(
            "Refined-keypoint quality diagnostics require one exact run name. "
            "Implicit latest/sorted-child selection is legacy inference; opt in "
            "explicitly only for historical diagnosis."
        )
    latest = _decode_text(parent_group.attrs.get("latest"))
    if latest and latest in parent_group:
        return latest
    try:
        names = sorted(parent_group.group_keys())
    except Exception:
        names = sorted(parent_group.keys())
    if not names:
        return None
    return str(names[-1])


def _array_or_none(group: zarr.Group, name: str, *, dtype: object | None = None) -> Optional[np.ndarray]:
    arr = group.get(name)
    if arr is None:
        return None
    data = np.asarray(arr[:])
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    return data


def _nonempty_finite(values: Optional[np.ndarray]) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    return data[np.isfinite(data)]


def _decode_text_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = _decode_text(item)
        if text is not None:
            out.append(text)
    return out


def _derive_edge_labels(
    *,
    edge_pairs: Optional[np.ndarray],
    explicit_labels: object,
    keypoint_labels: object,
) -> list[str]:
    if edge_pairs is None:
        return []
    labels = _decode_text_list(explicit_labels)
    edge_count = int(edge_pairs.shape[0]) if edge_pairs.ndim == 2 else 0
    if edge_count <= 0:
        return []
    if len(labels) == edge_count:
        return labels

    keypoint_names = _decode_text_list(keypoint_labels)
    derived: list[str] = []
    for edge_idx in range(edge_count):
        src = int(edge_pairs[edge_idx, 0])
        dst = int(edge_pairs[edge_idx, 1])
        if 0 <= src < len(keypoint_names) and 0 <= dst < len(keypoint_names):
            derived.append(f"{keypoint_names[src]}-{keypoint_names[dst]}")
        else:
            derived.append(f"{src}-{dst}")
    return derived


def _edge_distance_panel_data(quality_data: Mapping[str, object]) -> tuple[list[dict[str, object]], str]:
    edge_pairs = quality_data.get("edge_pairs")
    if not isinstance(edge_pairs, np.ndarray) or edge_pairs.ndim != 2 or edge_pairs.shape[1] < 2:
        return [], "missing"

    norm_values = quality_data.get("edge_distances_norm")
    raw_values = quality_data.get("edge_distances")
    values: Optional[np.ndarray] = None
    mode = "missing"
    if isinstance(norm_values, np.ndarray) and norm_values.ndim == 2 and norm_values.shape[1] == edge_pairs.shape[0]:
        values = np.asarray(norm_values, dtype=np.float64)
        mode = "normalized"
    elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 2 and raw_values.shape[1] == edge_pairs.shape[0]:
        values = np.asarray(raw_values, dtype=np.float64)
        mode = "raw"
    if values is None:
        return [], "missing"

    valid = quality_data.get("edge_distance_valid")
    if isinstance(valid, np.ndarray) and valid.shape == values.shape:
        valid_mask = np.asarray(valid, dtype=bool)
    else:
        valid_mask = np.isfinite(values)

    labels_raw = quality_data.get("edge_labels")
    labels = [str(item) for item in labels_raw] if isinstance(labels_raw, list) else []
    if len(labels) != int(edge_pairs.shape[0]):
        labels = [f"{int(pair[0])}-{int(pair[1])}" for pair in edge_pairs[:, :2].tolist()]

    total_rows = int(values.shape[0])
    items: list[dict[str, object]] = []
    for edge_idx in range(int(edge_pairs.shape[0])):
        mask = np.asarray(valid_mask[:, edge_idx], dtype=bool) & np.isfinite(values[:, edge_idx])
        valid_count = int(np.count_nonzero(mask))
        if valid_count <= 0:
            continue
        p50 = float(np.percentile(values[mask, edge_idx], 50))
        items.append(
            {
                "label": labels[edge_idx],
                "p50": p50,
                "valid_count": valid_count,
                "valid_rate": float(valid_count / total_rows) if total_rows > 0 else 0.0,
            }
        )
    return items, mode


def _count_reason_tokens(reason_labels: Optional[np.ndarray]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if reason_labels is None:
        return counts
    for raw in np.asarray(reason_labels, dtype=object):
        text = str(raw).strip()
        if not text:
            continue
        for token in text.split("|"):
            label = token.strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1
    return counts


def _compute_detection_source_counts(values: Optional[np.ndarray]) -> Dict[str, int]:
    if values is None:
        return {}
    source = np.asarray(values, dtype=np.int32)
    return {
        "manual_or_clean": int(np.sum(source == 0)),
        "interpolated": int(np.sum(source == 1)),
        "other": int(np.sum((source != 0) & (source != 1))),
    }


def _build_rate_items(quality: Mapping[str, object], total_rows: int) -> list[tuple[str, float]]:
    usable = np.asarray(quality.get("usable_keypoints") if "usable_keypoints" in quality else [], dtype=bool)
    confidence_valid = np.asarray(quality.get("confidence_valid") if "confidence_valid" in quality else [], dtype=bool)
    geometry_valid = np.asarray(quality.get("geometry_valid") if "geometry_valid" in quality else [], dtype=bool)
    flip_corrected = np.asarray(quality.get("flip_corrected") if "flip_corrected" in quality else [], dtype=bool)

    summary = _safe_mapping(quality.get("summary_statistics"))
    refined_success = max(1, _safe_int(summary.get("refined_success"), default=total_rows if total_rows > 0 else 1))

    items: list[tuple[str, float]] = []
    if usable.size:
        items.append(("usable", float(np.mean(usable) * 100.0)))
    if confidence_valid.size:
        items.append(("confidence_valid", float(np.mean(confidence_valid) * 100.0)))
    if geometry_valid.size:
        items.append(("geometry_valid", float(np.mean(geometry_valid) * 100.0)))
    if flip_corrected.size:
        items.append(("flip_corrected", float(np.mean(flip_corrected) * 100.0)))

    if not items:
        items = [
            (
                "usable",
                float(_safe_int(summary.get("usable_keypoints")) / float(refined_success) * 100.0),
            )
        ]
    return items


def load_keypoint_quality_report(
    zarr_path: str,
    refined_run: Optional[str] = None,
    *,
    allow_legacy_latest_fallback: bool = False,
) -> Dict[str, object]:
    """Load refined keypoint quality arrays and metadata for plotting."""
    root = zarr.open_group(zarr_path, mode="r")

    parent_name, parent_group = _resolve_refined_parent(root)
    if parent_name is None or parent_group is None:
        raise ValueError("No refined keypoint runs found.")

    selected_run = _resolve_refined_run(
        parent_group,
        refined_run,
        allow_legacy_latest_fallback=allow_legacy_latest_fallback,
    )
    if selected_run is None:
        raise ValueError("No refined keypoint run available.")

    run_group = parent_group[selected_run]
    summary = _normalize_summary_statistics(run_group.attrs.get("summary_statistics"))
    params = _safe_mapping(run_group.attrs.get("parameters"))
    review = _safe_mapping(run_group.attrs.get("keypoint_review_status"))

    quality_labels = _array_or_none(run_group, "quality_labels", dtype=np.int16)
    usable_keypoints = _array_or_none(run_group, "usable_keypoints", dtype=bool)
    confidence_valid = _array_or_none(run_group, "confidence_valid", dtype=bool)
    geometry_valid = _array_or_none(run_group, "geometry_valid", dtype=bool)
    flip_corrected = _array_or_none(run_group, "flip_corrected", dtype=bool)
    triangle_area = _array_or_none(run_group, "triangle_area", dtype=np.float64)
    min_angle = _array_or_none(run_group, "min_angle", dtype=np.float64)
    confidence = _array_or_none(run_group, "confidence", dtype=np.float64)
    detection_source = _array_or_none(run_group, "detection_source", dtype=np.int16)
    edge_pairs = _array_or_none(run_group, "edge_pairs", dtype=np.int32)
    edge_distances = _array_or_none(run_group, "edge_distances", dtype=np.float64)
    edge_distances_norm = _array_or_none(run_group, "edge_distances_norm", dtype=np.float64)
    edge_distance_valid = _array_or_none(run_group, "edge_distance_valid", dtype=bool)
    reason_labels = read_reason_labels(run_group)
    edge_labels = _derive_edge_labels(
        edge_pairs=edge_pairs,
        explicit_labels=run_group.attrs.get("edge_distance_labels"),
        keypoint_labels=run_group.attrs.get("keypoint_labels"),
    )

    if usable_keypoints is not None:
        total_rows = int(usable_keypoints.size)
    elif quality_labels is not None:
        total_rows = int(quality_labels.size)
    else:
        total_rows = _safe_int(summary.get("total_rois"), default=0)

    quality_data: Dict[str, object] = {
        "zarr_path": str(zarr_path),
        "parent_name": parent_name,
        "refined_run": selected_run,
        "coordinate_verification": "legacy_unverified_diagnostic",
        "source_camera_overlay_authority": False,
        "source_keypoints_run": _decode_text(run_group.attrs.get("source_keypoints_run")),
        "source_crop_run": _decode_text(run_group.attrs.get("source_crop_run")),
        "source_detect_run": _decode_text(run_group.attrs.get("source_detect_run")),
        "summary_statistics": summary,
        "parameters": params,
        "review_status": review,
        "total_rows": total_rows,
        "quality_labels": quality_labels,
        "usable_keypoints": usable_keypoints,
        "confidence_valid": confidence_valid,
        "geometry_valid": geometry_valid,
        "flip_corrected": flip_corrected,
        "triangle_area": triangle_area,
        "min_angle": min_angle,
        "confidence": confidence,
        "detection_source": detection_source,
        "edge_pairs": edge_pairs,
        "edge_distances": edge_distances,
        "edge_distances_norm": edge_distances_norm,
        "edge_distance_valid": edge_distance_valid,
        "edge_labels": edge_labels,
        "edge_distance_normalization": _safe_mapping(run_group.attrs.get("edge_distance_normalization")),
        "reason_labels": reason_labels,
        "reason_counts": _count_reason_tokens(reason_labels),
        "detection_source_counts": _compute_detection_source_counts(detection_source),
    }
    return quality_data


def create_keypoint_quality_visualization(quality_data: Mapping[str, object]) -> plt.Figure:
    """Create a quality dashboard figure for a refined keypoint run."""
    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        "UNVERIFIED DIAGNOSTIC — Keypoint Refinement Quality Overview",
        fontsize=16,
        fontweight="bold",
    )
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.26)

    summary = _normalize_summary_statistics(quality_data.get("summary_statistics"))
    params = _safe_mapping(quality_data.get("parameters"))
    review = _safe_mapping(quality_data.get("review_status"))
    source_counts = dict(quality_data.get("detection_source_counts") or {})
    edge_items, edge_mode = _edge_distance_panel_data(quality_data)
    normalization = _safe_mapping(quality_data.get("edge_distance_normalization"))
    normalization_mode = _decode_text(normalization.get("mode")) if normalization else None
    total_rows = _safe_int(quality_data.get("total_rows"))
    refined_success = _safe_int(summary.get("refined_success"), default=total_rows)
    usable_total = _safe_int(summary.get("usable_keypoints"))
    usable_rate = float(usable_total / refined_success * 100.0) if refined_success > 0 else 0.0

    ax_summary = fig.add_subplot(gs[0, 0])
    ax_summary.axis("off")
    summary_text = (
        "Run Metadata\n\n"
        f"Refined run: {quality_data.get('refined_run') or '—'}\n"
        f"Source keypoint run: {quality_data.get('source_keypoints_run') or '—'}\n"
        f"Source crop run: {quality_data.get('source_crop_run') or '—'}\n"
        f"Source detect run: {quality_data.get('source_detect_run') or '—'}\n\n"
        f"Total rows: {total_rows}\n"
        f"Refined success: {refined_success}\n"
        f"Usable keypoints: {usable_total} ({usable_rate:.1f}%)\n"
        f"Geometry issues: {_safe_int(summary.get('geometry_issues'))}\n"
        f"Low confidence: {_safe_int(summary.get('low_confidence'))}\n"
        f"Confidence missing: {_safe_int(summary.get('confidence_missing'))}\n"
        f"Flips corrected: {_safe_int(summary.get('flips_corrected', summary.get('flip_corrected')))}\n\n"
        f"Review: {_decode_text(review.get('state')) or '—'}/"
        f"{_decode_text(review.get('intended_use')) or '—'}\n"
        f"Edge metrics: {len(edge_items)} "
        f"({normalization_mode or edge_mode or '—'})"
    )
    ax_summary.text(
        0.02,
        0.98,
        summary_text,
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "whitesmoke", "edgecolor": "lightgray"},
    )

    ax_rates = fig.add_subplot(gs[0, 1])
    rate_items = _build_rate_items(quality_data, total_rows)
    labels = [item[0] for item in rate_items]
    values = [item[1] for item in rate_items]
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"][: len(values)]
    ax_rates.bar(labels, values, color=colors)
    ax_rates.set_ylim(0.0, 100.0)
    ax_rates.set_ylabel("Percent")
    ax_rates.set_title("Quality Gate Rates")
    ax_rates.grid(axis="y", alpha=0.25)

    ax_reasons = fig.add_subplot(gs[0, 2])
    reason_counts = dict(quality_data.get("reason_counts") or {})
    if reason_counts:
        ranked = sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        y = np.arange(len(ranked), dtype=np.int32)
        ax_reasons.barh(y, [int(count) for _, count in ranked], color="#9C755F")
        ax_reasons.set_yticks(y, labels=[name for name, _ in ranked])
        ax_reasons.invert_yaxis()
        ax_reasons.set_xlabel("Count")
        ax_reasons.set_title("Reason Labels")
        ax_reasons.grid(axis="x", alpha=0.25)
    else:
        ax_reasons.axis("off")
        ax_reasons.text(0.5, 0.5, "No reason labels", ha="center", va="center", fontsize=11)

    ax_edges = fig.add_subplot(gs[0, 3])
    if edge_items:
        labels = [str(item["label"]) for item in edge_items]
        p50_values = np.asarray([float(item["p50"]) for item in edge_items], dtype=np.float64)
        y = np.arange(len(edge_items), dtype=np.int32)
        ax_edges.barh(y, p50_values, color="#6D9DC5", edgecolor="#2C4A63", linewidth=0.7)
        ax_edges.set_yticks(y, labels=labels)
        ax_edges.invert_yaxis()
        ax_edges.set_xlabel("P50 distance")
        panel_title = "Edge Distance P50"
        if normalization_mode:
            panel_title = f"Edge Distance P50 ({normalization_mode})"
        elif edge_mode == "normalized":
            panel_title = "Edge Distance P50 (normalized)"
        ax_edges.set_title(panel_title)
        ax_edges.grid(axis="x", alpha=0.25)
        max_val = float(np.max(p50_values)) if p50_values.size else 0.0
        for idx, item in enumerate(edge_items):
            rate = float(item.get("valid_rate") or 0.0) * 100.0
            x_text = float(item["p50"]) + (0.01 * max(1.0, max_val))
            ax_edges.text(x_text, idx, f"{rate:.1f}%", va="center", fontsize=8)
    else:
        ax_edges.axis("off")
        ax_edges.text(0.5, 0.5, "No edge-distance metrics", ha="center", va="center", fontsize=11)

    area_values = _nonempty_finite(quality_data.get("triangle_area"))
    angle_values = _nonempty_finite(quality_data.get("min_angle"))
    confidence_values = _nonempty_finite(quality_data.get("confidence"))

    ax_area = fig.add_subplot(gs[1, 0])
    if area_values.size:
        ax_area.hist(area_values, bins=32, color="#76B7B2", edgecolor="black", linewidth=0.35)
        area_threshold = params.get("min_triangle_area")
        if area_threshold is not None:
            ax_area.axvline(float(area_threshold), color="red", linestyle="--", linewidth=1.2, label="min_area")
            ax_area.legend(loc="upper right", fontsize=8)
    ax_area.set_title("Triangle Area Distribution")
    ax_area.set_xlabel("Area")
    ax_area.set_ylabel("Count")
    ax_area.grid(alpha=0.2)

    ax_angle = fig.add_subplot(gs[1, 1])
    if angle_values.size:
        ax_angle.hist(angle_values, bins=32, color="#EDC948", edgecolor="black", linewidth=0.35)
        min_angle_threshold = params.get("min_triangle_angle")
        if min_angle_threshold is not None:
            ax_angle.axvline(
                float(min_angle_threshold),
                color="red",
                linestyle="--",
                linewidth=1.2,
                label="min_angle",
            )
            ax_angle.legend(loc="upper right", fontsize=8)
    ax_angle.set_title("Minimum Angle Distribution")
    ax_angle.set_xlabel("Angle (deg)")
    ax_angle.set_ylabel("Count")
    ax_angle.grid(alpha=0.2)

    ax_conf = fig.add_subplot(gs[1, 2])
    if confidence_values.size:
        ax_conf.hist(confidence_values, bins=32, color="#B07AA1", edgecolor="black", linewidth=0.35)
        conf_threshold = params.get("confidence_threshold")
        if conf_threshold is not None:
            ax_conf.axvline(
                float(conf_threshold),
                color="red",
                linestyle="--",
                linewidth=1.2,
                label="confidence_threshold",
            )
            ax_conf.legend(loc="upper right", fontsize=8)
    ax_conf.set_title("Confidence Distribution")
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Count")
    ax_conf.grid(alpha=0.2)

    ax_source = fig.add_subplot(gs[1, 3])
    if source_counts:
        labels = ["manual_or_clean", "interpolated", "other"]
        values = [int(source_counts.get(label, 0)) for label in labels]
        ax_source.bar(labels, values, color=["#4E79A7", "#F28E2B", "#9C755F"])
        ax_source.set_title("Detection Source Counts")
        ax_source.set_ylabel("Count")
        ax_source.tick_params(axis="x", rotation=20)
        ax_source.grid(axis="y", alpha=0.2)
    else:
        ax_source.axis("off")
        ax_source.text(0.5, 0.5, "No detection_source", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    return fig


def create_keypoint_refinement_pipeline_visualization(quality_data: Mapping[str, object]) -> plt.Figure:
    """Create a refinement-pipeline summary figure for a refined keypoint run."""
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        "UNVERIFIED DIAGNOSTIC — Keypoint Refinement Pipeline Overview",
        fontsize=16,
        fontweight="bold",
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.15], wspace=0.24)

    summary = _normalize_summary_statistics(quality_data.get("summary_statistics"))
    review = _safe_mapping(quality_data.get("review_status"))
    source_counts = dict(quality_data.get("detection_source_counts") or {})
    edge_items, edge_mode = _edge_distance_panel_data(quality_data)
    normalization = _safe_mapping(quality_data.get("edge_distance_normalization"))
    normalization_mode = _decode_text(normalization.get("mode")) if normalization else None

    ax_meta = fig.add_subplot(gs[0, 0])
    ax_meta.axis("off")
    meta_text = (
        "Lineage + Review\n\n"
        f"Refined run: {quality_data.get('refined_run') or '—'}\n"
        f"Source keypoint run: {quality_data.get('source_keypoints_run') or '—'}\n"
        f"Source crop run: {quality_data.get('source_crop_run') or '—'}\n"
        f"Source detect run: {quality_data.get('source_detect_run') or '—'}\n\n"
        f"Review state: {_decode_text(review.get('state')) or '—'}\n"
        f"Review intended_use: {_decode_text(review.get('intended_use')) or '—'}\n"
        f"Review method: {_decode_text(review.get('method')) or '—'}\n"
        f"Review timestamp: {_review_timestamp(review) or '—'}\n\n"
        f"Edge metrics: {len(edge_items)} edges "
        f"({normalization_mode or edge_mode or '—'})\n"
    )
    ax_meta.text(
        0.02,
        0.98,
        meta_text,
        transform=ax_meta.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "whitesmoke", "edgecolor": "lightgray"},
    )

    ax_counts = fig.add_subplot(gs[0, 1])
    names = [
        "total_rois",
        "source_success",
        "refined_success",
        "usable_keypoints",
        "flips_corrected",
    ]
    values = [
        _safe_int(summary.get("total_rois")),
        _safe_int(summary.get("source_success")),
        _safe_int(summary.get("refined_success")),
        _safe_int(summary.get("usable_keypoints")),
        _safe_int(summary.get("flips_corrected")),
    ]
    ax_counts.bar(names, values, color=["#4E79A7", "#59A14F", "#F28E2B", "#76B7B2", "#E15759"])
    ax_counts.set_ylabel("Count")
    ax_counts.set_title("Pipeline Counts")
    ax_counts.grid(axis="y", alpha=0.25)
    ax_counts.tick_params(axis="x", rotation=20)

    if source_counts:
        summary_line = (
            f"detection_source: manual_or_clean={source_counts.get('manual_or_clean', 0)} "
            f"interpolated={source_counts.get('interpolated', 0)} "
            f"other={source_counts.get('other', 0)}"
        )
        ax_counts.text(
            0.01,
            -0.18,
            summary_line,
            transform=ax_counts.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
        )

    ax_edges = fig.add_subplot(gs[0, 2])
    if edge_items:
        labels = [str(item["label"]) for item in edge_items]
        p50_values = np.asarray([float(item["p50"]) for item in edge_items], dtype=np.float64)
        y = np.arange(len(edge_items), dtype=np.int32)
        ax_edges.barh(y, p50_values, color="#6D9DC5", edgecolor="#2C4A63", linewidth=0.7)
        ax_edges.set_yticks(y, labels=labels)
        ax_edges.invert_yaxis()
        ax_edges.set_xlabel("P50 distance")
        panel_title = "Edge Distance P50"
        if normalization_mode:
            panel_title = f"Edge Distance P50 ({normalization_mode})"
        elif edge_mode == "normalized":
            panel_title = "Edge Distance P50 (normalized)"
        ax_edges.set_title(panel_title)
        ax_edges.grid(axis="x", alpha=0.25)
        max_val = float(np.max(p50_values)) if p50_values.size else 0.0
        for idx, item in enumerate(edge_items):
            rate = float(item.get("valid_rate") or 0.0) * 100.0
            x_text = float(item["p50"]) + (0.01 * max(1.0, max_val))
            ax_edges.text(x_text, idx, f"{rate:.1f}%", va="center", fontsize=8)
    else:
        ax_edges.axis("off")
        ax_edges.text(0.5, 0.5, "No edge-distance metrics", ha="center", va="center", fontsize=11)

    fig.tight_layout()
    return fig


def render_keypoint_quality_png(
    zarr_path: str,
    refined_run: Optional[str] = None,
    *,
    dpi: int = 150,
    show: bool = False,
    allow_legacy_latest_fallback: bool = False,
) -> tuple[bytes, Dict[str, object]]:
    """Render keypoint quality overview PNG bytes for a refined run."""
    quality_data = load_keypoint_quality_report(
        zarr_path,
        refined_run=refined_run,
        allow_legacy_latest_fallback=allow_legacy_latest_fallback,
    )
    fig = create_keypoint_quality_visualization(quality_data)
    if show:
        plt.show()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    meta = {
        "refined_run": quality_data.get("refined_run"),
        "summary_statistics": quality_data.get("summary_statistics"),
        "parameters": quality_data.get("parameters"),
        "coordinate_verification": quality_data.get("coordinate_verification"),
        "source_camera_overlay_authority": False,
    }
    return buffer.getvalue(), meta


def render_keypoint_refinement_pipeline_png(
    zarr_path: str,
    refined_run: Optional[str] = None,
    *,
    dpi: int = 150,
    show: bool = False,
    allow_legacy_latest_fallback: bool = False,
) -> tuple[bytes, Dict[str, object]]:
    """Render keypoint refinement-pipeline overview PNG bytes for a refined run."""
    quality_data = load_keypoint_quality_report(
        zarr_path,
        refined_run=refined_run,
        allow_legacy_latest_fallback=allow_legacy_latest_fallback,
    )
    fig = create_keypoint_refinement_pipeline_visualization(quality_data)
    if show:
        plt.show()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    meta = {
        "refined_run": quality_data.get("refined_run"),
        "summary_statistics": quality_data.get("summary_statistics"),
        "review_status": quality_data.get("review_status"),
        "coordinate_verification": quality_data.get("coordinate_verification"),
        "source_camera_overlay_authority": False,
    }
    return buffer.getvalue(), meta


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to archive zarr.")
    parser.add_argument(
        "--refined-run",
        help="Exact refined keypoint run required unless legacy fallback is enabled.",
    )
    parser.add_argument(
        "--allow-legacy-latest-fallback",
        action="store_true",
        help=(
            "Permit historical latest/sorted-child selection for an explicitly "
            "unverified diagnostic rendering."
        ),
    )
    parser.add_argument(
        "--artifact",
        choices=[QUALITY_ARTIFACT_NAME, REFINEMENT_PIPELINE_ARTIFACT_NAME],
        default=QUALITY_ARTIFACT_NAME,
        help=f"Artifact to render (default: {QUALITY_ARTIFACT_NAME}).",
    )
    parser.add_argument("--output", type=Path, help="Optional PNG output path (default: show interactive plot).")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI when writing output.")
    args = parser.parse_args(argv)

    try:
        if args.artifact == QUALITY_ARTIFACT_NAME:
            png_bytes, _meta = render_keypoint_quality_png(
                str(args.zarr_path),
                refined_run=args.refined_run,
                dpi=int(args.dpi),
                show=args.output is None,
                allow_legacy_latest_fallback=args.allow_legacy_latest_fallback,
            )
        else:
            png_bytes, _meta = render_keypoint_refinement_pipeline_png(
                str(args.zarr_path),
                refined_run=args.refined_run,
                dpi=int(args.dpi),
                show=args.output is None,
                allow_legacy_latest_fallback=args.allow_legacy_latest_fallback,
            )
    except Exception as exc:
        print(f"Keypoint quality visualization failed: {exc}")
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(png_bytes)
        print(f"Wrote keypoint quality visualization: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
