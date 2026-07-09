from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, resolve_roi_rect, roi_rect_corners
from localized_periodic_signal_probe import (
    _extract_events,
    _inband_snr,
    _json_ready,
    _load_roi_stack,
    _mask_from_npz,
    _nuisance_regressors,
    _prepare_stack,
    _residualize,
    _trace_spike_metrics,
    _video_shape,
)


def _load_selected_mask(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        selected = np.asarray(data["selected_mask"], dtype=bool)
        candidate = np.asarray(data["candidate_mask"], dtype=bool)
        bbox = tuple(int(value) for value in np.asarray(data["bbox_xyxy"]).reshape(-1).tolist())
    if selected.shape != candidate.shape:
        raise ValueError(f"{path} selected_mask shape {selected.shape} differs from candidate_mask {candidate.shape}")
    if len(bbox) != 4:
        raise ValueError(f"{path} bbox_xyxy must have four values")
    return {
        "path": str(path),
        "selected_mask": selected,
        "candidate_mask": candidate,
        "bbox_xyxy": bbox,
    }


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return math.nan, math.nan
    return float(np.mean(yy)), float(np.mean(xx))


def _component_metrics(mask: np.ndarray) -> dict[str, Any]:
    from scipy import ndimage

    labels, n_components = ndimage.label(mask)
    if int(n_components) == 0:
        return {"n_components": 0, "largest_size": 0, "largest_fraction": 0.0}
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, int(n_components) + 1))
    largest = int(np.max(sizes))
    return {
        "n_components": int(n_components),
        "largest_size": largest,
        "largest_fraction": float(largest / max(1, int(np.count_nonzero(mask)))),
    }


def _dilated(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 2)
    return ndimage.binary_dilation(mask, structure=structure, iterations=int(radius))


def _spatial_comparison(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Cannot compare masks with shapes {mask_a.shape} and {mask_b.shape}")
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    count_a = int(np.count_nonzero(mask_a))
    count_b = int(np.count_nonzero(mask_b))
    centroid_a = _mask_centroid(mask_a)
    centroid_b = _mask_centroid(mask_b)
    centroid_distance = (
        float(np.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]))
        if np.isfinite(centroid_a).all() and np.isfinite(centroid_b).all()
        else math.nan
    )
    out = {
        "count_a": count_a,
        "count_b": count_b,
        "intersection": intersection,
        "union": union,
        "iou": float(intersection / union) if union else math.nan,
        "a_fraction_overlapping_b": float(intersection / count_a) if count_a else math.nan,
        "b_fraction_overlapping_a": float(intersection / count_b) if count_b else math.nan,
        "centroid_a_yx": [float(centroid_a[0]), float(centroid_a[1])],
        "centroid_b_yx": [float(centroid_b[0]), float(centroid_b[1])],
        "centroid_distance_px": centroid_distance,
        "mask_a_components": _component_metrics(mask_a),
        "mask_b_components": _component_metrics(mask_b),
    }
    for radius in (1, 2):
        dil_a = _dilated(mask_a, radius)
        dil_b = _dilated(mask_b, radius)
        dil_intersection = int(np.count_nonzero(dil_a & dil_b))
        dil_union = int(np.count_nonzero(dil_a | dil_b))
        out[f"dilated_{radius}px_iou"] = float(dil_intersection / dil_union) if dil_union else math.nan
        out[f"a_fraction_within_b_dilated_{radius}px"] = (
            float(np.count_nonzero(mask_a & dil_b) / count_a) if count_a else math.nan
        )
        out[f"b_fraction_within_a_dilated_{radius}px"] = (
            float(np.count_nonzero(mask_b & dil_a) / count_b) if count_b else math.nan
        )
    return out


def _load_segment(
    *,
    args: argparse.Namespace,
    frame_start: int,
    frame_count: int,
    roi_polygon: np.ndarray,
    sample_mask: np.ndarray | None,
) -> dict[str, Any]:
    loaded = _load_roi_stack(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(frame_start)),
        frame_count=max(0, int(frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=float(args.min_roi_mean_intensity),
        pad_px=max(0, int(args.pad_px)),
        duplicate_mean_tol=float(args.duplicate_mean_tol),
    )
    stack, finite_valid, segment, interpolated = _prepare_stack(
        loaded["stack"],
        loaded["valid"],
        max_interpolated_gap=max(0, int(args.max_interpolated_gap_samples)),
        min_segment_samples=max(32, int(args.min_segment_samples)),
    )
    segment_start, segment_stop = segment
    mask = loaded["candidate_crop_mask"]
    mask_flat = mask.reshape(-1)
    traces = stack.reshape(stack.shape[0], -1)[:, mask_flat].T.astype(np.float64)
    motion_pred, scalar, mean_image = _nuisance_regressors(stack, mask)
    residual = _residualize(traces, motion_pred, scalar)
    frame_indices = loaded["frame_indices"][segment_start:segment_stop]
    return {
        "stack": stack,
        "valid": finite_valid,
        "analysis_segment_rows": [int(segment_start), int(segment_stop)],
        "frame_indices": frame_indices,
        "candidate_mask": mask,
        "bbox_xyxy": tuple(int(value) for value in loaded["bbox_xyxy"]),
        "residual": residual,
        "mean_image": mean_image,
        "interpolated_short_gap_frame_count": int(interpolated),
        "valid_loaded_frame_count": int(np.count_nonzero(loaded["valid"])),
        "loaded_frame_count": int(loaded["stack"].shape[0]),
    }


def _event_summary(events: dict[str, np.ndarray | float | int]) -> dict[str, Any]:
    rates = np.asarray(events["event_rate_per_min"], dtype=np.float64)
    finite_rates = rates[np.isfinite(rates)]
    intervals = np.asarray(events["intervals_s"], dtype=np.float64)
    return {
        "event_count": int(np.asarray(events["peaks"], dtype=np.int64).size),
        "event_rate_per_min_median": float(np.median(finite_rates)) if finite_rates.size else math.nan,
        "event_rate_per_min_mean": float(np.mean(finite_rates)) if finite_rates.size else math.nan,
        "event_rate_per_min_iqr": float(np.percentile(finite_rates, 75) - np.percentile(finite_rates, 25))
        if finite_rates.size
        else math.nan,
        "fraction_intervals_in_band": float(events["fraction_intervals_in_band"]),
        "rejected_interval_count": int(events["rejected_interval_count"]),
        "interval_s_median": float(np.median(intervals)) if intervals.size else math.nan,
    }


def _fixed_mask_metrics(
    *,
    residual: np.ndarray,
    candidate_mask: np.ndarray,
    fixed_mask: np.ndarray,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
    event_prominence_scale: float,
) -> dict[str, Any]:
    if fixed_mask.shape != candidate_mask.shape:
        raise ValueError(f"fixed mask shape {fixed_mask.shape} does not match candidate mask {candidate_mask.shape}")
    usable = np.asarray(fixed_mask, dtype=bool) & np.asarray(candidate_mask, dtype=bool)
    selected_idx = np.flatnonzero(usable.reshape(-1)[candidate_mask.reshape(-1)])
    if selected_idx.size == 0:
        raise ValueError("Fixed mask has no pixels inside this segment's candidate mask.")
    trace = np.median(residual[selected_idx], axis=0)
    filtered_snr = _inband_snr(
        trace[None, :],
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
    )[0]
    events = _extract_events(
        trace,
        fps=fps,
        band_min_hz=band_min_hz,
        band_max_hz=band_max_hz,
        prominence_scale=event_prominence_scale,
    )
    return {
        "fixed_pixel_count": int(selected_idx.size),
        "trace": trace,
        "filtered": np.asarray(events["filtered"], dtype=np.float64),
        "inband_snr": float(filtered_snr),
        "trace_std": float(np.std(trace)),
        "filtered_std": float(np.std(np.asarray(events["filtered"], dtype=np.float64))),
        "spike_metrics": _trace_spike_metrics(trace),
        "events": events,
        "event_summary": _event_summary(events),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 3:
        return math.nan
    xv = x[valid] - float(np.mean(x[valid]))
    yv = y[valid] - float(np.mean(y[valid]))
    denom = float(np.linalg.norm(xv) * np.linalg.norm(yv))
    if denom <= np.finfo(float).eps:
        return math.nan
    return float(np.dot(xv, yv) / denom)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    fieldnames = [
        "segment_label",
        "mask_label",
        "fixed_pixel_count",
        "inband_snr",
        "trace_std",
        "filtered_std",
        "event_count",
        "event_rate_per_min_median",
        "event_rate_per_min_mean",
        "event_rate_per_min_iqr",
        "fraction_intervals_in_band",
        "rejected_interval_count",
        "interval_s_median",
        "selected_trace_abs_robust_z_p99",
        "selected_trace_spike_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, math.nan) for key in fieldnames})


def _write_figure(
    path: Path,
    *,
    segment_a: dict[str, Any],
    segment_b: dict[str, Any],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    result_lookup: dict[tuple[str, str], dict[str, Any]],
    label_a: str,
    label_b: str,
    fps: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_output_dir(path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    def overlay(axis: Any, mean_image: np.ndarray, title: str) -> None:
        axis.imshow(mean_image, cmap="gray", interpolation="nearest")
        axis.contour(segment_a["candidate_mask"].astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.7)
        axis.contour(mask_a.astype(np.uint8), levels=[0.5], colors="#e41a1c", linewidths=1.0)
        axis.contour(mask_b.astype(np.uint8), levels=[0.5], colors="#377eb8", linewidths=1.0)
        axis.set_title(title)
        axis.axis("off")

    overlay(axes[0, 0], segment_a["mean_image"], f"{label_a}: fixed masks")
    overlay(axes[0, 1], segment_b["mean_image"], f"{label_b}: fixed masks")
    overlap = np.zeros((*mask_a.shape, 3), dtype=np.float32)
    overlap[segment_a["candidate_mask"]] = (0.22, 0.22, 0.22)
    overlap[mask_a] = (0.95, 0.10, 0.08)
    overlap[mask_b] = (0.10, 0.30, 0.95)
    overlap[mask_a & mask_b] = (0.95, 0.10, 0.95)
    axes[0, 2].imshow(overlap, interpolation="nearest")
    axes[0, 2].set_title("Mask overlap")
    axes[0, 2].axis("off")

    for axis, segment_label in ((axes[1, 0], label_a), (axes[1, 1], label_b)):
        time = None
        for mask_label, color in ((label_a, "#e41a1c"), (label_b, "#377eb8")):
            result = result_lookup[(segment_label, mask_label)]
            filtered = np.asarray(result["filtered"], dtype=np.float64)
            z = filtered / (np.std(filtered) + np.finfo(float).eps)
            if time is None:
                time = np.arange(z.size, dtype=np.float64) / float(fps)
            axis.plot(time, z, linewidth=0.7, color=color, label=f"{mask_label} mask")
        corr = _correlation(
            result_lookup[(segment_label, label_a)]["filtered"],
            result_lookup[(segment_label, label_b)]["filtered"],
        )
        axis.set_title(f"{segment_label}: fixed-mask filtered traces, r={corr:.3f}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("z")
        axis.legend(fontsize=8)

    axes[1, 2].axis("off")
    lines = ["Fixed-mask performance"]
    for segment_label in (label_a, label_b):
        for mask_label in (label_a, label_b):
            result = result_lookup[(segment_label, mask_label)]
            event_summary = result["event_summary"]
            lines.append(
                f"{segment_label} / {mask_label}: "
                f"{event_summary['event_count']} events, "
                f"{event_summary['event_rate_per_min_median']:.1f}/min, "
                f"SNR {result['inband_snr']:.2f}, "
                f"in-band {event_summary['fraction_intervals_in_band'] * 100:.0f}%"
            )
    axes[1, 2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-apply fixed localized periodic-signal masks across two segments.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--roi-json", type=Path, default=None)
    parser.add_argument("--roi", type=str, default=None)
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--mask-npz", type=Path, default=None)
    parser.add_argument("--mask-a-npz", type=Path, required=True)
    parser.add_argument("--mask-b-npz", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="early")
    parser.add_argument("--label-b", type=str, default="late")
    parser.add_argument("--segment-a-start", type=int, required=True)
    parser.add_argument("--segment-a-count", type=int, required=True)
    parser.add_argument("--segment-b-start", type=int, required=True)
    parser.add_argument("--segment-b-count", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--pad-px", type=int, default=4)
    parser.add_argument("--min-roi-mean-intensity", type=float, default=1.0)
    parser.add_argument("--duplicate-mean-tol", type=float, default=1e-6)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--min-segment-samples", type=int, default=512)
    parser.add_argument("--event-prominence-scale", type=float, default=0.5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).with_name("outputs") / "localized_periodic_mask_cross_apply",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video)) if args.mask_npz is not None else None
    mask_a_info = _load_selected_mask(args.mask_a_npz)
    mask_b_info = _load_selected_mask(args.mask_b_npz)
    if mask_a_info["bbox_xyxy"] != mask_b_info["bbox_xyxy"]:
        raise ValueError(f"Mask bbox mismatch: {mask_a_info['bbox_xyxy']} vs {mask_b_info['bbox_xyxy']}")
    if mask_a_info["candidate_mask"].shape != mask_b_info["candidate_mask"].shape:
        raise ValueError("Mask candidate shapes differ.")

    segment_a = _load_segment(
        args=args,
        frame_start=int(args.segment_a_start),
        frame_count=int(args.segment_a_count),
        roi_polygon=roi_polygon,
        sample_mask=sample_mask,
    )
    segment_b = _load_segment(
        args=args,
        frame_start=int(args.segment_b_start),
        frame_count=int(args.segment_b_count),
        roi_polygon=roi_polygon,
        sample_mask=sample_mask,
    )
    expected_bbox = mask_a_info["bbox_xyxy"]
    for label, segment in ((args.label_a, segment_a), (args.label_b, segment_b)):
        if tuple(segment["bbox_xyxy"]) != expected_bbox:
            raise ValueError(f"{label} segment bbox {segment['bbox_xyxy']} does not match mask bbox {expected_bbox}")

    mask_a = np.asarray(mask_a_info["selected_mask"], dtype=bool)
    mask_b = np.asarray(mask_b_info["selected_mask"], dtype=bool)
    effective_fps = float(args.fps) / max(1, int(args.stride))
    result_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for segment_label, segment in ((args.label_a, segment_a), (args.label_b, segment_b)):
        for mask_label, fixed_mask in ((args.label_a, mask_a), (args.label_b, mask_b)):
            result = _fixed_mask_metrics(
                residual=segment["residual"],
                candidate_mask=segment["candidate_mask"],
                fixed_mask=fixed_mask,
                fps=effective_fps,
                band_min_hz=float(args.band_min_hz),
                band_max_hz=float(args.band_max_hz),
                event_prominence_scale=float(args.event_prominence_scale),
            )
            result_lookup[(segment_label, mask_label)] = result
            row = {
                "segment_label": segment_label,
                "mask_label": mask_label,
                "fixed_pixel_count": result["fixed_pixel_count"],
                "inband_snr": result["inband_snr"],
                "trace_std": result["trace_std"],
                "filtered_std": result["filtered_std"],
                **result["event_summary"],
                **result["spike_metrics"],
            }
            rows.append(row)

    trace_correlations = {
        args.label_a: _correlation(
            result_lookup[(args.label_a, args.label_a)]["filtered"],
            result_lookup[(args.label_a, args.label_b)]["filtered"],
        ),
        args.label_b: _correlation(
            result_lookup[(args.label_b, args.label_a)]["filtered"],
            result_lookup[(args.label_b, args.label_b)]["filtered"],
        ),
    }
    spatial = _spatial_comparison(mask_a, mask_b)

    output_prefix = args.output_prefix
    ensure_output_dir(output_prefix.parent)
    summary_path = output_prefix.with_suffix(".summary.json")
    csv_path = output_prefix.with_suffix(".csv")
    figure_path = output_prefix.with_suffix(".png")
    _write_csv(csv_path, rows)
    _write_figure(
        figure_path,
        segment_a=segment_a,
        segment_b=segment_b,
        mask_a=mask_a,
        mask_b=mask_b,
        result_lookup=result_lookup,
        label_a=args.label_a,
        label_b=args.label_b,
        fps=effective_fps,
    )
    summary = {
        "mask_a_npz": str(args.mask_a_npz),
        "mask_b_npz": str(args.mask_b_npz),
        "label_a": str(args.label_a),
        "label_b": str(args.label_b),
        "segment_a": {
            "requested": [int(args.segment_a_start), int(args.segment_a_count)],
            "analysis_frame_indices": [int(segment_a["frame_indices"][0]), int(segment_a["frame_indices"][-1])],
            "analysis_frame_count": int(segment_a["frame_indices"].size),
            "valid_loaded_frame_count": int(segment_a["valid_loaded_frame_count"]),
        },
        "segment_b": {
            "requested": [int(args.segment_b_start), int(args.segment_b_count)],
            "analysis_frame_indices": [int(segment_b["frame_indices"][0]), int(segment_b["frame_indices"][-1])],
            "analysis_frame_count": int(segment_b["frame_indices"].size),
            "valid_loaded_frame_count": int(segment_b["valid_loaded_frame_count"]),
        },
        "spatial": spatial,
        "trace_correlations": trace_correlations,
        "fixed_mask_results": rows,
        "outputs": {
            "summary_json": str(summary_path),
            "csv": str(csv_path),
            "figure_png": str(figure_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"summary_json: {summary_path}")
    print(f"csv: {csv_path}")
    print(f"figure_png: {figure_path}")
    print(f"mask_iou: {spatial['iou']:.3f}")
    print(f"dilated_2px_iou: {spatial['dilated_2px_iou']:.3f}")
    print(f"{args.label_a}_trace_correlation: {trace_correlations[args.label_a]:.3f}")
    print(f"{args.label_b}_trace_correlation: {trace_correlations[args.label_b]:.3f}")
    for row in rows:
        print(
            f"{row['segment_label']} / {row['mask_label']}: "
            f"events={row['event_count']} median={row['event_rate_per_min_median']:.3f}/min "
            f"snr={row['inband_snr']:.3f} in_band={row['fraction_intervals_in_band']:.3f}"
        )


if __name__ == "__main__":
    main()
