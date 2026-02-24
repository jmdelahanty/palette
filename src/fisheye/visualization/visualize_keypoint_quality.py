#!/usr/bin/env python3
"""Render keypoint refinement quality overview artifacts."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

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


def _resolve_refined_run(parent_group: zarr.Group, requested_refined_run: Optional[str]) -> Optional[str]:
    if requested_refined_run:
        return requested_refined_run if requested_refined_run in parent_group else None
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
) -> Dict[str, object]:
    """Load refined keypoint quality arrays and metadata for plotting."""
    root = zarr.open_group(zarr_path, mode="r")

    parent_name, parent_group = _resolve_refined_parent(root)
    if parent_name is None or parent_group is None:
        raise ValueError("No refined keypoint runs found.")

    selected_run = _resolve_refined_run(parent_group, refined_run)
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
    reason_labels = read_reason_labels(run_group)

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
        "reason_labels": reason_labels,
        "reason_counts": _count_reason_tokens(reason_labels),
        "detection_source_counts": _compute_detection_source_counts(detection_source),
    }
    return quality_data


def create_keypoint_quality_visualization(quality_data: Mapping[str, object]) -> plt.Figure:
    """Create a 2x3 quality dashboard figure for a refined keypoint run."""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Keypoint Refinement Quality Overview", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    summary = _normalize_summary_statistics(quality_data.get("summary_statistics"))
    params = _safe_mapping(quality_data.get("parameters"))
    review = _safe_mapping(quality_data.get("review_status"))
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
        f"{_decode_text(review.get('intended_use')) or '—'}"
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

    fig.tight_layout()
    return fig


def create_keypoint_refinement_pipeline_visualization(quality_data: Mapping[str, object]) -> plt.Figure:
    """Create a refinement-pipeline summary figure for a refined keypoint run."""
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Keypoint Refinement Pipeline Overview", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.24)

    summary = _normalize_summary_statistics(quality_data.get("summary_statistics"))
    review = _safe_mapping(quality_data.get("review_status"))
    source_counts = dict(quality_data.get("detection_source_counts") or {})

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
        f"Review timestamp: {_review_timestamp(review) or '—'}\n"
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

    fig.tight_layout()
    return fig


def render_keypoint_quality_png(
    zarr_path: str,
    refined_run: Optional[str] = None,
    *,
    dpi: int = 150,
    show: bool = False,
) -> tuple[bytes, Dict[str, object]]:
    """Render keypoint quality overview PNG bytes for a refined run."""
    quality_data = load_keypoint_quality_report(zarr_path, refined_run=refined_run)
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
    }
    return buffer.getvalue(), meta


def render_keypoint_refinement_pipeline_png(
    zarr_path: str,
    refined_run: Optional[str] = None,
    *,
    dpi: int = 150,
    show: bool = False,
) -> tuple[bytes, Dict[str, object]]:
    """Render keypoint refinement-pipeline overview PNG bytes for a refined run."""
    quality_data = load_keypoint_quality_report(zarr_path, refined_run=refined_run)
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
    }
    return buffer.getvalue(), meta


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to archive zarr.")
    parser.add_argument("--refined-run", help="Specific refined keypoint run (default: latest).")
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
            )
        else:
            png_bytes, _meta = render_keypoint_refinement_pipeline_png(
                str(args.zarr_path),
                refined_run=args.refined_run,
                dpi=int(args.dpi),
                show=args.output is None,
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
