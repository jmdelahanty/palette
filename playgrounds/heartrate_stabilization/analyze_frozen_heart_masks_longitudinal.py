from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.dynamic_heart_support import analyze_dynamic_heart_support
from fisheye.analysis.local_rostral_heartrate import (
    HeartrateConfig,
    LocalCoordinateDataset,
    analyze_heartrate,
)


_FRAME_FIELDS = (
    "frame_indices",
    "timestamps_s",
    "traces",
    "pixel_valid",
    "frame_valid",
    "source_xy",
    "bilinear_weights",
    "body_occupancy",
    "eye_occupancy",
    "gradient_magnitude",
    "motion_prediction",
    "nuisance_values",
    "transform_uncertainty",
)


def _read_mask(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=bool)


def _window_dataset(
    dataset: LocalCoordinateDataset,
    start: int,
    stop: int,
) -> LocalCoordinateDataset:
    if not (0 <= int(start) < int(stop) <= dataset.frame_count):
        raise ValueError("invalid longitudinal window bounds")
    updates = {
        name: np.asarray(getattr(dataset, name))[int(start) : int(stop)]
        for name in _FRAME_FIELDS
    }
    updates["metadata"] = {
        **dict(dataset.metadata),
        "longitudinal_window_row_start": int(start),
        "longitudinal_window_row_stop_exclusive": int(stop),
    }
    return replace(dataset, **updates).validated()


def _window_ranges(
    timestamps_s: np.ndarray,
    *,
    window_seconds: float,
    step_seconds: float,
    min_window_seconds: float,
) -> list[tuple[int, int]]:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    ranges: list[tuple[int, int]] = []
    start_time = float(timestamps[0])
    final_time = float(timestamps[-1])
    while start_time <= final_time:
        stop_time = start_time + float(window_seconds)
        start = int(np.searchsorted(timestamps, start_time, side="left"))
        stop = int(np.searchsorted(timestamps, stop_time, side="left"))
        stop = min(stop, timestamps.size)
        if stop - start >= 2:
            duration = float(timestamps[stop - 1] - timestamps[start])
            if duration >= float(min_window_seconds):
                ranges.append((start, stop))
        start_time += float(step_seconds)
    return ranges


def _write_plot(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    mask_names: tuple[str, ...],
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "original_38": "#777777",
        "consensus_9": "#0072B2",
        "intersection_8": "#009E73",
        "union_39": "#D55E00",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    for name in mask_names:
        selected = [row for row in rows if row["mask"] == name]
        if not selected:
            continue
        time_min = np.asarray([row["window_mid_s"] for row in selected]) / 60.0
        candidate_rate = np.asarray(
            [
                row["candidate_cycles_per_min"] if row["status"] == "ok" else np.nan
                for row in selected
            ],
            dtype=np.float64,
        )
        latent_score = np.asarray(
            [row["latent_score"] if row["status"] == "ok" else np.nan for row in selected],
            dtype=np.float64,
        )
        color = colors.get(name)
        axes[0].plot(
            time_min,
            candidate_rate,
            marker="o",
            ms=3,
            lw=1.2,
            color=color,
            label=name,
        )
        axes[1].plot(
            time_min,
            latent_score,
            marker="o",
            ms=3,
            lw=1.2,
            color=color,
            label=name,
        )
    coverage_rows = [row for row in rows if row["mask"] == mask_names[0]]
    axes[2].plot(
        np.asarray([row["window_mid_s"] for row in coverage_rows]) / 60.0,
        [row["valid_frame_fraction"] for row in coverage_rows],
        color="#444444",
        marker="o",
        ms=3,
        lw=1.2,
    )
    axes[0].set_ylabel("candidate cycles/min")
    axes[0].set_ylim(120.0, 240.0)
    axes[0].set_title(title)
    axes[1].set_ylabel("cross-fit latent score")
    axes[2].set_ylabel("valid frame fraction")
    axes[2].set_xlabel("recording time (min)")
    axes[2].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncol=2)
    axes[1].legend(loc="best", ncol=2)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def _finite_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": None, "median": None, "max": None, "iqr": None}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track frozen-mask candidate oscillation across a complete local-coordinate cache."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--step-seconds", type=float, default=60.0)
    parser.add_argument("--min-window-seconds", type=float, default=30.0)
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    if original.shape != dataset.image_shape_hw or consensus.shape != dataset.image_shape_hw:
        raise ValueError("frozen masks must match the dataset canonical image shape")
    masks = {
        "original_38": original,
        "consensus_9": consensus,
        "intersection_8": original & consensus,
        "union_39": original | consensus,
    }
    config = HeartrateConfig(
        band_min_hz=float(args.frequency_min_hz),
        band_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        surrogate_count=0,
        random_seed=int(args.seed),
    ).validated()
    ranges = _window_ranges(
        dataset.timestamps_s,
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.step_seconds),
        min_window_seconds=float(args.min_window_seconds),
    )
    frequencies = (
        float(args.frequency_min_hz)
        + np.arange(
            int(
                round(
                    (float(args.frequency_max_hz) - float(args.frequency_min_hz))
                    / float(args.frequency_step_hz)
                )
            )
            + 1,
            dtype=np.float64,
        )
        * float(args.frequency_step_hz)
    )
    score_curves = {
        name: np.full((len(ranges), frequencies.size), np.nan, dtype=np.float32)
        for name in masks
    }
    rows: list[dict[str, Any]] = []
    for window_index, (start, stop) in enumerate(ranges):
        window = _window_dataset(dataset, start, stop)
        frame_start = int(window.frame_indices[0])
        frame_stop = int(window.frame_indices[-1])
        start_s = float(window.timestamps_s[0] - dataset.timestamps_s[0])
        stop_s = float(window.timestamps_s[-1] - dataset.timestamps_s[0])
        valid_fraction = float(np.mean(window.frame_valid))
        try:
            base = analyze_heartrate(window, config)
        except (RuntimeError, ValueError) as exc:
            for name, mask in masks.items():
                rows.append(
                    {
                        "window_index": window_index,
                        "window_start_s": start_s,
                        "window_stop_s": stop_s,
                        "window_mid_s": 0.5 * (start_s + stop_s),
                        "window_frame_start": frame_start,
                        "window_frame_stop_inclusive": frame_stop,
                        "valid_frame_fraction": valid_fraction,
                        "mask": name,
                        "pixel_count": int(np.count_nonzero(mask)),
                        "status": f"base_failed:{type(exc).__name__}",
                        "candidate_frequency_hz": math.nan,
                        "candidate_cycles_per_min": math.nan,
                        "support_score": math.nan,
                        "shared_phase_score": math.nan,
                        "latent_score": math.nan,
                        "control_ratio": math.nan,
                    }
                )
            print(f"window {window_index + 1}/{len(ranges)} base_failed")
            continue
        for name, mask in masks.items():
            row = {
                "window_index": window_index,
                "window_start_s": start_s,
                "window_stop_s": stop_s,
                "window_mid_s": 0.5 * (start_s + stop_s),
                "window_frame_start": frame_start,
                "window_frame_stop_inclusive": frame_stop,
                "valid_frame_fraction": valid_fraction,
                "mask": name,
                "pixel_count": int(np.count_nonzero(mask)),
            }
            try:
                dynamic = analyze_dynamic_heart_support(
                    window,
                    config,
                    base,
                    heart_mask=mask,
                    mask_is_independent=True,
                    frequency_min_hz=float(args.frequency_min_hz),
                    frequency_max_hz=float(args.frequency_max_hz),
                    surrogate_count=0,
                )
                if not np.allclose(dynamic.frequency_grid_hz, frequencies):
                    raise ValueError("window frequency grid does not match output grid")
                score_curves[name][window_index] = dynamic.frequency_latent_scores
                row.update(
                    {
                        "status": "ok",
                        "candidate_frequency_hz": float(dynamic.frequency_hz),
                        "candidate_cycles_per_min": float(dynamic.frequency_hz * 60.0),
                        "support_score": float(dynamic.support_score),
                        "shared_phase_score": float(dynamic.shared_phase_score),
                        "latent_score": float(dynamic.latent_score),
                        "control_ratio": float(dynamic.control_ratio),
                    }
                )
            except (RuntimeError, ValueError) as exc:
                row.update(
                    {
                        "status": f"dynamic_failed:{type(exc).__name__}",
                        "candidate_frequency_hz": math.nan,
                        "candidate_cycles_per_min": math.nan,
                        "support_score": math.nan,
                        "shared_phase_score": math.nan,
                        "latent_score": math.nan,
                        "control_ratio": math.nan,
                    }
                )
            rows.append(row)
        ok_count = sum(
            row["status"] == "ok" and row["window_index"] == window_index
            for row in rows
        )
        print(
            f"window {window_index + 1}/{len(ranges)} frames={frame_start}:{frame_stop} "
            f"valid={valid_fraction:.3f} masks_ok={ok_count}/{len(masks)}"
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".longitudinal.csv")
    json_path = output_prefix.with_suffix(".longitudinal.summary.json")
    arrays_path = output_prefix.with_suffix(".longitudinal.arrays.npz")
    figure_path = output_prefix.with_suffix(".longitudinal.png")
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        arrays_path,
        frequency_grid_hz=frequencies.astype(np.float32),
        window_row_start=np.asarray([start for start, _stop in ranges], dtype=np.int64),
        window_row_stop_exclusive=np.asarray([stop for _start, stop in ranges], dtype=np.int64),
        **{
            f"{name}_frequency_latent_scores": values
            for name, values in score_curves.items()
        },
    )
    summary: dict[str, Any] = {
        "interpretation": "descriptive_frozen_mask_candidate_frequency_not_validated_heart_rate",
        "dataset_npz": str(args.dataset_npz),
        "frame_count": dataset.frame_count,
        "elapsed_seconds": float(dataset.timestamps_s[-1] - dataset.timestamps_s[0]),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "window_count": len(ranges),
        "frequency_bounds_hz": [
            float(args.frequency_min_hz),
            float(args.frequency_max_hz),
        ],
        "frequency_step_hz": float(args.frequency_step_hz),
        "per_window_p_values": "not_computed",
        "masks": {},
    }
    for name in masks:
        selected = [row for row in rows if row["mask"] == name and row["status"] == "ok"]
        summary["masks"][name] = {
            "pixel_count": int(np.count_nonzero(masks[name])),
            "successful_window_count": len(selected),
            "candidate_cycles_per_min": _finite_summary(
                [float(row["candidate_cycles_per_min"]) for row in selected]
            ),
            "candidate_frequency_hz": _finite_summary(
                [float(row["candidate_frequency_hz"]) for row in selected]
            ),
            "latent_score": _finite_summary(
                [float(row["latent_score"]) for row in selected]
            ),
            "control_ratio": _finite_summary(
                [float(row["control_ratio"]) for row in selected]
            ),
        }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_plot(
        figure_path,
        rows,
        mask_names=tuple(masks),
        title="Frozen-mask candidate oscillation across complete recording",
    )
    print(f"summary_json: {json_path}")
    print(f"longitudinal_csv: {csv_path}")
    print(f"arrays_npz: {arrays_path}")
    print(f"diagnostic_png: {figure_path}")


if __name__ == "__main__":
    main()
