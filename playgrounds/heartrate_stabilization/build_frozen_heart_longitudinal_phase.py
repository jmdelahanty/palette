from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import warnings

import numpy as np

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.dynamic_heart_support import (
    analyze_dynamic_heart_support,
    reconstruct_crossfit_heart_phase,
)
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, analyze_heartrate


def _read_windows(path: Path, mask_name: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["mask"] == mask_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-window cross-fitted frozen-mask phase for longitudinal rendering."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument(
        "--phase-mask",
        choices=("intersection", "consensus", "original"),
        default="intersection",
        help="Frozen mask whose pixels receive phase estimates.",
    )
    parser.add_argument(
        "--frequency-source-mask",
        default="intersection_8",
        help="Longitudinal CSV mask whose frozen per-window frequency is used.",
    )
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    intersection = original & consensus
    phase_masks = {
        "intersection": intersection,
        "consensus": consensus,
        "original": original,
    }
    phase_mask = phase_masks[str(args.phase_mask)]
    phase_mask_name = {
        "intersection": "intersection_8",
        "consensus": "consensus_9",
        "original": "original_38",
    }[str(args.phase_mask)]
    selected = _mask_at_pixels(phase_mask, dataset.pixel_xy)
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size < 3:
        raise ValueError(f"{phase_mask_name} has fewer than three dataset pixels")
    windows = _read_windows(args.longitudinal_csv, str(args.frequency_source_mask))
    if not windows:
        raise ValueError(
            f"no longitudinal rows found for frequency source {args.frequency_source_mask!r}"
        )
    config = HeartrateConfig(
        band_min_hz=float(args.frequency_min_hz),
        band_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        surrogate_count=0,
        random_seed=int(args.seed),
    ).validated()
    phase_rad = np.full(
        (dataset.frame_count, selected_indices.size),
        np.nan,
        dtype=np.float32,
    )
    phase_alpha = np.zeros_like(phase_rad)
    phase_valid = np.zeros(dataset.frame_count, dtype=bool)
    frequency_hz = np.full(dataset.frame_count, np.nan, dtype=np.float32)
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    completed = 0
    for window in windows:
        if window["status"] != "ok":
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
        phase_groups = dict(dynamic.pixel_groups)
        phase_groups["heart_support"] = selected.copy()
        dynamic = replace(
            dynamic,
            frequency_hz=frozen_frequency,
            pixel_groups=phase_groups,
        )
        phase = reconstruct_crossfit_heart_phase(local, config, base, dynamic)
        analytic = np.asarray(phase.analytic_residual[:, selected_indices], dtype=np.complex128)
        amplitude = np.abs(analytic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_amplitude = np.nanmedian(amplitude, axis=0)
        median_amplitude = np.where(
            np.isfinite(median_amplitude) & (median_amplitude > np.finfo(float).eps),
            median_amplitude,
            1.0,
        )
        relative_amplitude = np.clip(amplitude / (1.5 * median_amplitude[None, :]), 0.0, 1.0)
        local_alpha = np.zeros_like(amplitude, dtype=np.float64)
        for row_index, fold_index in enumerate(phase.model_fold_indices):
            if int(fold_index) < 0:
                continue
            weights = np.asarray(
                phase.fold_loading_weights[int(fold_index), selected_indices],
                dtype=np.float64,
            )
            local_alpha[row_index] = weights * relative_amplitude[row_index]
        finite = np.isfinite(analytic) & np.asarray(phase.frame_valid, dtype=bool)[:, None]
        visible_alpha = np.zeros_like(local_alpha)
        visible_alpha[finite] = 0.35 + 0.65 * local_alpha[finite]
        phase_rad[start:stop] = np.where(finite, np.angle(analytic), np.nan).astype(np.float32)
        phase_alpha[start:stop] = visible_alpha.astype(np.float32)
        phase_valid[start:stop] = np.asarray(phase.frame_valid, dtype=bool)
        frequency_hz[start:stop] = float(dynamic.frequency_hz)
        completed += 1
        print(
            f"phase_window {completed} index={window['window_index']} "
            f"frames={frame_start}:{frame_stop} frequency={frozen_frequency:.2f}",
            flush=True,
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output_prefix.with_suffix(".longitudinal_phase.arrays.npz")
    summary_path = output_prefix.with_suffix(".longitudinal_phase.summary.json")
    np.savez_compressed(
        arrays_path,
        frame_indices=np.asarray(dataset.frame_indices, dtype=np.int64),
        pixel_indices=selected_indices.astype(np.int32),
        phase_rad=phase_rad,
        phase_alpha=phase_alpha,
        phase_valid=phase_valid.astype(np.uint8),
        frequency_hz=frequency_hz,
        phase_mask_name=np.asarray(phase_mask_name),
        frequency_source_mask=np.asarray(str(args.frequency_source_mask)),
    )
    summary = {
        "dataset_npz": str(args.dataset_npz),
        "longitudinal_csv": str(args.longitudinal_csv),
        "mask": phase_mask_name,
        "frequency_source_mask": str(args.frequency_source_mask),
        "pixel_count": int(selected_indices.size),
        "completed_window_count": completed,
        "phase_valid_frame_count": int(np.count_nonzero(phase_valid)),
        "phase_valid_fraction": float(np.mean(phase_valid)),
        "frequency_bounds_hz": [
            float(args.frequency_min_hz),
            float(args.frequency_max_hz),
        ],
        "phase_encoding": "twilight_shifted_cyclic_hue",
        "alpha_encoding": "crossfit_loading_weight_times_relative_analytic_amplitude",
        "caveat": "Phase is a window-cross-fitted diagnostic of the candidate oscillator, not biological contraction phase.",
        "arrays_npz": str(arrays_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"arrays_npz: {arrays_path}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
