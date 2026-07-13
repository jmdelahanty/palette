from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.dynamic_heart_support import (
    analyze_dynamic_heart_support,
    reconstruct_crossfit_heart_phase,
)
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, analyze_heartrate
from fisheye.analysis.regional_phase_delay import analyze_regional_phase_delay


def _read_windows(path: Path, mask_name: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mask"] == mask_name]
    if not rows:
        raise ValueError(f"no longitudinal rows found for mask {mask_name!r}")
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _circular_mean(angles_rad: np.ndarray) -> tuple[float, float]:
    values = np.asarray(angles_rad, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    vector = np.mean(np.exp(1j * values))
    return float(np.angle(vector)), float(np.abs(vector))


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _plot_diagnostic(
    path: Path,
    dataset: Any,
    upper: np.ndarray,
    lower: np.ndarray,
    window_rows: Sequence[Mapping[str, Any]],
    block_rows: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    upper_analytic: np.ndarray,
    lower_analytic: np.ndarray,
    frame_valid: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    mean_image = np.full(dataset.image_shape_hw, np.nan, dtype=np.float64)
    mean_image[xy[:, 1], xy[:, 0]] = np.nanmedian(dataset.traces, axis=0)
    upper_image = np.zeros(dataset.image_shape_hw, dtype=bool)
    lower_image = np.zeros(dataset.image_shape_hw, dtype=bool)
    upper_image[xy[upper, 1], xy[upper, 0]] = True
    lower_image[xy[lower, 1], xy[lower, 0]] = True
    support_y, support_x = np.nonzero(upper_image | lower_image)
    margin = 2
    x0, x1 = int(support_x.min()) - margin, int(support_x.max()) + margin + 1
    y0, y1 = int(support_y.min()) - margin, int(support_y.max()) + margin + 1

    ok_windows = [row for row in window_rows if row["status"] == "ok"]
    mid_min = np.asarray([float(row["window_mid_s"]) / 60.0 for row in ok_windows])
    phase_deg = np.asarray([float(row["mean_phase_deg"]) for row in ok_windows])
    lag_fraction = np.asarray([float(row["mean_lag_cycle_fraction"]) for row in ok_windows])
    window_plv = np.asarray([float(row["across_block_plv"]) for row in ok_windows])
    block_mid_min = np.asarray(
        [0.5 * (float(row["start_s"]) + float(row["stop_s"])) / 60.0 for row in block_rows]
    )
    block_phase = np.asarray([float(row["mean_phase_deg"]) for row in block_rows])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes[0, 0].imshow(mean_image, cmap="gray", interpolation="nearest")
    overlay = np.zeros((*mean_image.shape, 4), dtype=np.float64)
    overlay[upper_image] = (0.85, 0.18, 0.16, 0.62)
    overlay[lower_image] = (0.12, 0.35, 0.82, 0.62)
    axes[0, 0].imshow(overlay, interpolation="nearest")
    axes[0, 0].set_xlim(x0 - 0.5, x1 - 0.5)
    axes[0, 0].set_ylim(y1 - 0.5, y0 - 0.5)
    axes[0, 0].set_title(f"Frozen split: upper {np.count_nonzero(upper)}, lower {np.count_nonzero(lower)}")
    axes[0, 0].set_axis_off()

    axes[0, 1].scatter(block_mid_min, block_phase, s=12, alpha=0.45, color="0.45", label="blocks")
    axes[0, 1].plot(mid_min, phase_deg, "o-", color="black", lw=1.0, label="window mean")
    axes[0, 1].axhline(0.0, color="0.65", ls="--", lw=0.8)
    axes[0, 1].axhline(180.0, color="0.8", ls=":", lw=0.8)
    axes[0, 1].axhline(-180.0, color="0.8", ls=":", lw=0.8)
    axes[0, 1].set_ylim(-190.0, 190.0)
    axes[0, 1].set_xlabel("recording time (min)")
    axes[0, 1].set_ylabel("lower minus upper phase (deg)")
    axes[0, 1].set_title("Direction and phase offset by window")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(mid_min, lag_fraction, "o-", color="#3a6ea5", lw=1.0)
    axes[0, 2].axhline(0.0, color="0.65", ls="--", lw=0.8)
    axes[0, 2].set_ylim(-0.52, 0.52)
    axes[0, 2].set_xlabel("recording time (min)")
    axes[0, 2].set_ylabel("lower lag / local period")
    axes[0, 2].set_title("Peak-equivalent lag; sign gives direction")

    cycle_fraction = np.asarray([float(row["lag_cycle_fraction"]) for row in cycle_rows])
    axes[1, 0].hist(cycle_fraction, bins=np.linspace(-0.5, 0.5, 31), color="#5f82b5", edgecolor="white")
    axes[1, 0].axvline(0.0, color="black", ls="--", lw=0.8)
    axes[1, 0].set_xlabel("paired-cycle lower lag / local period")
    axes[1, 0].set_ylabel("cycle count")
    axes[1, 0].set_title("Paired phase-crossing offsets")

    axes[1, 1].plot(mid_min, window_plv, "o-", color="#222222", label="across-block PLV")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_xlabel("recording time (min)")
    axes[1, 1].set_ylabel("phase-locking value")
    axes[1, 1].set_title("Within-window delay repeatability")

    valid_rows = np.flatnonzero(np.asarray(frame_valid, dtype=bool))
    if valid_rows.size:
        center = int(valid_rows[valid_rows.size // 2])
        timestamps = np.asarray(dataset.timestamps_s, dtype=np.float64)
        show = (
            np.asarray(frame_valid, dtype=bool)
            & (timestamps >= timestamps[center] - 1.5)
            & (timestamps <= timestamps[center] + 1.5)
        )
        relative = timestamps[show] - timestamps[center]
        upper_trace = np.asarray(upper_analytic.real[show], dtype=np.float64)
        lower_trace = np.asarray(lower_analytic.real[show], dtype=np.float64)
        for trace, color, label in (
            (upper_trace, "#c9362e", "upper"),
            (lower_trace, "#2458b8", "lower"),
        ):
            scale = np.nanmedian(np.abs(trace - np.nanmedian(trace)))
            scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
            axes[1, 2].plot(relative, trace / scale, color=color, lw=1.1, label=label)
        axes[1, 2].axhline(0.0, color="0.7", lw=0.8)
        axes[1, 2].set_xlabel("time around example segment (s)")
        axes[1, 2].set_ylabel("analytic real component / scale")
        axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("Example held-out regional oscillations")
    fig.suptitle("Frozen original-38 upper/lower phase-delay diagnostic; not cardiac event validation")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure frozen upper/lower phase delay in longitudinal windows."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--regions-npz", type=Path, required=True)
    parser.add_argument("--upper-key", default="upper_mask")
    parser.add_argument("--lower-key", default="lower_mask")
    parser.add_argument("--frequency-source-mask", default="intersection_8")
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--surrogate-count", type=int, default=999)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    intersection = original & consensus
    original_pixels = _mask_at_pixels(original, dataset.pixel_xy)
    upper_image = _read_mask(args.regions_npz, args.upper_key)
    lower_image = _read_mask(args.regions_npz, args.lower_key)
    upper_pixels = _mask_at_pixels(upper_image, dataset.pixel_xy) & original_pixels
    lower_pixels = _mask_at_pixels(lower_image, dataset.pixel_xy) & original_pixels
    if np.any(upper_pixels & lower_pixels):
        raise ValueError("frozen upper/lower regions overlap")
    if not np.array_equal(upper_pixels | lower_pixels, original_pixels):
        raise ValueError("frozen upper/lower regions do not partition original mask")

    windows = _read_windows(args.longitudinal_csv, str(args.frequency_source_mask))
    config = HeartrateConfig(
        band_min_hz=float(args.frequency_min_hz),
        band_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        surrogate_count=0,
        random_seed=int(args.seed),
    ).validated()
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    full_upper = np.full(dataset.frame_count, np.nan + 0j, dtype=np.complex64)
    full_lower = np.full(dataset.frame_count, np.nan + 0j, dtype=np.complex64)
    full_lag_ms = np.full(dataset.frame_count, np.nan, dtype=np.float32)
    full_valid = np.zeros(dataset.frame_count, dtype=bool)
    window_output: list[dict[str, Any]] = []
    block_output: list[dict[str, Any]] = []
    cycle_output: list[dict[str, Any]] = []

    for window in windows:
        common = {
            "window_index": int(window["window_index"]),
            "window_mid_s": float(window["window_mid_s"]),
            "status": str(window["status"]),
        }
        if window["status"] != "ok":
            window_output.append(common)
            continue
        frame_start = int(window["window_frame_start"])
        frame_stop = int(window["window_frame_stop_inclusive"])
        start = int(np.searchsorted(frame_indices, frame_start, side="left"))
        stop = int(np.searchsorted(frame_indices, frame_stop, side="right"))
        local = _window_dataset(dataset, start, stop)
        base = analyze_heartrate(local, config)
        dynamic = analyze_dynamic_heart_support(
            local,
            config,
            base,
            heart_mask=intersection,
            mask_is_independent=True,
            frequency_min_hz=float(args.frequency_min_hz),
            frequency_max_hz=float(args.frequency_max_hz),
            surrogate_count=0,
        )
        frozen_frequency = float(window["candidate_frequency_hz"])
        groups = dict(dynamic.pixel_groups)
        groups["heart_support"] = original_pixels.copy()
        dynamic = replace(dynamic, frequency_hz=frozen_frequency, pixel_groups=groups)
        phase = reconstruct_crossfit_heart_phase(local, config, base, dynamic)
        regional = analyze_regional_phase_delay(
            local,
            phase,
            upper_pixels=upper_pixels,
            lower_pixels=lower_pixels,
            regions_independent=True,
            surrogate_count=0,
            alpha=float(config.alpha),
            max_gap_factor=float(config.max_timestamp_gap_factor),
            seed=int(args.seed) + int(window["window_index"]),
        )
        full_upper[start:stop] = regional.upper_analytic.astype(np.complex64)
        full_lower[start:stop] = regional.lower_analytic.astype(np.complex64)
        full_lag_ms[start:stop] = regional.lower_lag_ms.astype(np.float32)
        full_valid[start:stop] = regional.frame_valid
        mean_phase_rad = math.radians(regional.across_block_mean_phase_deg)
        period_ms = 1000.0 / frozen_frequency
        row = {
            **common,
            "frequency_hz": frozen_frequency,
            "period_ms": period_ms,
            "valid_frame_count": int(np.count_nonzero(regional.frame_valid)),
            "block_count": len(regional.block_summary),
            "paired_cycle_count": len(regional.cycle_rows),
            "mean_phase_deg": regional.across_block_mean_phase_deg,
            "mean_lag_ms": regional.across_block_lower_lag_ms,
            "mean_lag_cycle_fraction": -mean_phase_rad / (2.0 * np.pi),
            "across_block_plv": regional.across_block_phase_locking_value,
            "median_within_block_plv": regional.median_within_block_phase_locking_value,
        }
        window_output.append(row)
        for block in regional.block_summary:
            phase_deg = float(block["mean_phase_offset_deg_lower_minus_upper"])
            block_output.append(
                {
                    "window_index": common["window_index"],
                    "frequency_hz": frozen_frequency,
                    **dict(block),
                    "mean_phase_deg": phase_deg,
                    "lag_cycle_fraction": -math.radians(phase_deg) / (2.0 * np.pi),
                }
            )
        for cycle in regional.cycle_rows:
            lag_ms = float(cycle["lower_minus_upper_ms"])
            cycle_output.append(
                {
                    "window_index": common["window_index"],
                    "frequency_hz": frozen_frequency,
                    **dict(cycle),
                    "lag_cycle_fraction": lag_ms / period_ms,
                }
            )
        print(
            f"regional_window index={window['window_index']} frequency={frozen_frequency:.2f} "
            f"blocks={len(regional.block_summary)} phase={regional.across_block_mean_phase_deg:.1f}deg",
            flush=True,
        )

    ok_output = [row for row in window_output if row["status"] == "ok" and row.get("block_count", 0)]
    window_angles = np.radians([float(row["mean_phase_deg"]) for row in ok_output])
    mean_phase, across_window_plv = _circular_mean(window_angles)
    rng = np.random.default_rng(int(args.seed) + 40009)
    null_plv = np.zeros(int(args.surrogate_count), dtype=np.float64)
    for index in range(null_plv.size):
        randomized = window_angles + rng.uniform(-np.pi, np.pi, window_angles.size)
        _mean, null_plv[index] = _circular_mean(randomized)
    p_value = (
        float(1 + np.count_nonzero(null_plv >= across_window_plv)) / float(null_plv.size + 1)
        if null_plv.size
        else 1.0
    )
    cycle_fraction = np.asarray(
        [float(row["lag_cycle_fraction"]) for row in cycle_output], dtype=np.float64
    )
    window_phase_deg = _wrap_degrees(np.degrees(window_angles))
    summary = {
        "diagnostic_only": True,
        "event_validation": False,
        "interpretation": "exploratory_full_recording_frozen_upper_lower_delay",
        "frequency_source_mask": str(args.frequency_source_mask),
        "phase_support_mask": "original_38",
        "regions_source": str(args.regions_npz),
        "upper_pixel_count": int(np.count_nonzero(upper_pixels)),
        "lower_pixel_count": int(np.count_nonzero(lower_pixels)),
        "scorable_window_count": len(ok_output),
        "block_count": len(block_output),
        "paired_cycle_count": len(cycle_output),
        "phase_valid_frame_count": int(np.count_nonzero(full_valid)),
        "across_window_mean_phase_deg_lower_minus_upper": float(np.degrees(mean_phase)),
        "across_window_phase_locking_value": across_window_plv,
        "across_window_random_rotation_p_value": p_value,
        "lower_leading_window_count": int(np.count_nonzero(window_phase_deg > 0.0)),
        "upper_leading_window_count": int(np.count_nonzero(window_phase_deg < 0.0)),
        "median_absolute_antiphase_deviation_deg": float(
            np.median(np.abs(np.abs(window_phase_deg) - 180.0))
        ),
        "median_cycle_lag_fraction": float(np.median(cycle_fraction)),
        "cycle_lag_fraction_mad": float(
            np.median(np.abs(cycle_fraction - np.median(cycle_fraction)))
        ),
        "paired_cycle_lower_leading_fraction": float(np.mean(cycle_fraction < 0.0)),
        "null_contract": "independent uniform phase rotation of each one-minute window mean",
        "lag_sign_contract": "positive lag means lower follows upper; positive phase means lower leads upper",
        "caveat": "The narrow-band phase crossings are peak-equivalent diagnostics, not independently validated cardiac events.",
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    windows_path = output_prefix.with_suffix(".regional_longitudinal.windows.csv")
    blocks_path = output_prefix.with_suffix(".regional_longitudinal.blocks.csv")
    cycles_path = output_prefix.with_suffix(".regional_longitudinal.cycles.csv")
    arrays_path = output_prefix.with_suffix(".regional_longitudinal.arrays.npz")
    summary_path = output_prefix.with_suffix(".regional_longitudinal.summary.json")
    figure_path = output_prefix.with_suffix(".regional_longitudinal.png")
    _write_rows(windows_path, window_output)
    _write_rows(blocks_path, block_output)
    _write_rows(cycles_path, cycle_output)
    np.savez_compressed(
        arrays_path,
        frame_indices=frame_indices,
        timestamps_s=np.asarray(dataset.timestamps_s, dtype=np.float64),
        upper_analytic=full_upper,
        lower_analytic=full_lower,
        lower_lag_ms=full_lag_ms,
        frame_valid=full_valid.astype(np.uint8),
        upper_pixels=upper_pixels.astype(np.uint8),
        lower_pixels=lower_pixels.astype(np.uint8),
        null_across_window_plv=null_plv.astype(np.float32),
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot_diagnostic(
        figure_path,
        dataset,
        upper_pixels,
        lower_pixels,
        window_output,
        block_output,
        cycle_output,
        full_upper,
        full_lower,
        full_valid,
    )
    print(f"summary_json: {summary_path}")
    print(f"diagnostic_png: {figure_path}")


if __name__ == "__main__":
    main()
