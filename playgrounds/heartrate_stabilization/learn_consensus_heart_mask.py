from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.consensus_heart_mask import (
    ConsensusHeartMaskResult,
    ConsensusMaskConfig,
    learn_consensus_heart_mask,
)
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _fold_rows(result: ConsensusHeartMaskResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in result.outer_folds:
        rows.append(
            {
                "fold_index": fold.fold_index,
                "confirmation_start_s": fold.confirmation_interval_s[0],
                "confirmation_stop_s": fold.confirmation_interval_s[1],
                "discovery_frame_count": fold.discovery_frame_count,
                "confirmation_frame_count": fold.confirmation_frame_count,
                "candidate_frequency_hz": fold.candidate.frequency_hz,
                "candidate_pixel_count": int(fold.candidate.pixel_indices.size),
                "candidate_cluster_mass": fold.candidate.cluster_mass,
                "confirmation_score_definition": "held_out_phase_aligned_latent_score",
                "confirmation_score": fold.confirmation_score,
                "confirmation_p_value": fold.confirmation_p_value,
                "confirmation_chunk_count": fold.confirmation_chunk_count,
                "control_ratio": fold.control_ratio,
                "confirmed": fold.confirmed,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, dataset, result: ConsensusHeartMaskResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    count_image = np.full(dataset.image_shape_hw, np.nan, dtype=np.float64)
    p_image = np.full(dataset.image_shape_hw, np.nan, dtype=np.float64)
    for pixel, (x, y) in enumerate(xy.tolist()):
        if 0 <= x < dataset.image_shape_hw[1] and 0 <= y < dataset.image_shape_hw[0]:
            count_image[y, x] = result.selection_counts[pixel]
            p_image[y, x] = -np.log10(max(result.selection_p_values[pixel], 1e-12))
    finite_xy = xy[
        (xy[:, 0] >= 0)
        & (xy[:, 0] < dataset.image_shape_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < dataset.image_shape_hw[0])
    ]
    x0, y0 = np.min(finite_xy, axis=0) - 2
    x1, y1 = np.max(finite_xy, axis=0) + 3
    extent = (max(0, x0), min(dataset.image_shape_hw[1], x1), max(0, y0), min(dataset.image_shape_hw[0], y1))
    ys = slice(extent[2], extent[3])
    xs = slice(extent[0], extent[1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    first = axes[0].imshow(count_image[ys, xs], cmap="viridis", vmin=0, vmax=len(result.outer_folds))
    axes[0].set_title("Outer-fold selection count")
    fig.colorbar(first, ax=axes[0], fraction=0.046)
    second = axes[1].imshow(p_image[ys, xs], cmap="magma", vmin=0)
    axes[1].set_title("Max-null calibrated -log10(p)")
    fig.colorbar(second, ax=axes[1], fraction=0.046)
    axes[2].imshow(result.consensus_mask[ys, xs], cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title(f"Consensus support ({np.count_nonzero(result.consensus_pixels)} pixels)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"Five-minute consensus mask | detected={result.detected} | "
        f"outer confirmations={result.confirmed_outer_fold_count}/{len(result.outer_folds)}"
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_outputs(
    output_prefix: Path,
    dataset,
    result: ConsensusHeartMaskResult,
    analysis_config: HeartrateConfig,
    consensus_config: ConsensusMaskConfig,
) -> dict[str, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_suffix(".consensus_mask.summary.json")
    folds_path = output_prefix.with_suffix(".consensus_mask.folds.csv")
    pixels_path = output_prefix.with_suffix(".consensus_mask.pixels.csv")
    arrays_path = output_prefix.with_suffix(".consensus_mask.arrays.npz")
    figure_path = output_prefix.with_suffix(".consensus_mask.diagnostic.png")
    folds = _fold_rows(result)
    summary = {
        "detected": result.detected,
        "reason": result.reason,
        "consensus_pixel_count": int(np.count_nonzero(result.consensus_pixels)),
        "confirmed_outer_fold_count": result.confirmed_outer_fold_count,
        "outer_fold_count": len(result.outer_folds),
        "median_candidate_frequency_hz": result.median_candidate_frequency_hz,
        "null_selection_count_threshold": result.null_selection_count_threshold,
        "consensus_null": (
            "independent_shape_preserving_cluster_translations_within_physically_eligible_roi"
        ),
        "analysis_config": vars(analysis_config),
        "consensus_config": vars(consensus_config),
        "folds": folds,
        "interpretation": (
            "exploratory_five_minute_mask_discovery_requires_untouched_interval_confirmation"
        ),
    }
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    _write_csv(folds_path, folds)
    pixel_rows = [
        {
            "pixel_index": pixel,
            "canonical_x": float(dataset.pixel_xy[pixel, 0]),
            "canonical_y": float(dataset.pixel_xy[pixel, 1]),
            "selection_count": int(result.selection_counts[pixel]),
            "selection_fraction": float(result.selection_fractions[pixel]),
            "max_null_p_value": float(result.selection_p_values[pixel]),
            "consensus_selected": bool(result.consensus_pixels[pixel]),
        }
        for pixel in range(dataset.pixel_count)
    ]
    _write_csv(pixels_path, pixel_rows)
    np.savez_compressed(
        arrays_path,
        pixel_xy=np.asarray(dataset.pixel_xy, dtype=np.float32),
        selection_counts=np.asarray(result.selection_counts, dtype=np.int16),
        selection_fractions=np.asarray(result.selection_fractions, dtype=np.float32),
        selection_p_values=np.asarray(result.selection_p_values, dtype=np.float32),
        consensus_pixels=np.asarray(result.consensus_pixels, dtype=np.uint8),
        consensus_mask=np.asarray(result.consensus_mask, dtype=np.uint8),
        null_max_selection_counts=np.asarray(result.null_max_selection_counts, dtype=np.int16),
    )
    _write_figure(figure_path, dataset, result)
    return {
        "summary_json": summary_path,
        "folds_csv": folds_path,
        "pixels_csv": pixels_path,
        "arrays_npz": arrays_path,
        "diagnostic_png": figure_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a null-calibrated consensus heart mask across contiguous outer folds."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--band-min-hz", type=float, default=3.0)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--outer-fold-count", type=int, default=5)
    parser.add_argument("--outer-guard-seconds", type=float, default=1.0)
    parser.add_argument("--min-selection-folds", type=int, default=3)
    parser.add_argument("--min-confirmed-outer-folds", type=int, default=3)
    parser.add_argument("--consensus-surrogate-count", type=int, default=199)
    parser.add_argument("--heldout-surrogate-count", type=int, default=39)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pixel-score-threshold-z", type=float, default=1.5)
    parser.add_argument("--min-cluster-pixels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_npz)
    analysis_config = HeartrateConfig(
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        pixel_score_threshold_z=float(args.pixel_score_threshold_z),
        min_cluster_pixels=int(args.min_cluster_pixels),
        surrogate_count=0,
        alpha=float(args.alpha),
        random_seed=int(args.seed),
    ).validated()
    consensus_config = ConsensusMaskConfig(
        outer_fold_count=int(args.outer_fold_count),
        outer_guard_seconds=float(args.outer_guard_seconds),
        min_selection_folds=int(args.min_selection_folds),
        min_confirmed_outer_folds=int(args.min_confirmed_outer_folds),
        consensus_surrogate_count=int(args.consensus_surrogate_count),
        heldout_surrogate_count=int(args.heldout_surrogate_count),
        alpha=float(args.alpha),
        random_seed=int(args.seed),
    ).validated()
    result = learn_consensus_heart_mask(dataset, analysis_config, consensus_config)
    outputs = write_outputs(
        args.output_prefix,
        dataset,
        result,
        analysis_config,
        consensus_config,
    )
    print(f"detected: {result.detected}")
    print(f"reason: {result.reason}")
    print(f"consensus_pixels: {int(np.count_nonzero(result.consensus_pixels))}")
    print(
        f"confirmed_outer_folds: {result.confirmed_outer_fold_count}/{len(result.outer_folds)}"
    )
    print(f"median_candidate_frequency_hz: {result.median_candidate_frequency_hz:.6f}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
