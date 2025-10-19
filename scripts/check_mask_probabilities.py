#!/usr/bin/env python3
"""Sample stored YOLO eye-mask probabilities and report mid-range statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import zarr


class RunningStats:
    """Aggregate min/max/moments and mid-range probability metrics."""

    def __init__(
        self,
        *,
        tol: float,
        max_unique: int,
        hist_bins: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        self._tol = tol
        self._max_unique = max_unique
        self._rng = rng
        self._unique_set: set[float] = set()
        self.count = 0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.sum_val = 0.0
        self.sum_sq = 0.0
        self.mid_count = 0
        self.near_zero = 0
        self.near_one = 0
        self._hist_bins = hist_bins
        self._hist_counts = np.zeros(len(hist_bins) - 1, dtype=np.float64) if hist_bins is not None else None

    def update(self, data: np.ndarray) -> None:
        flat = np.asarray(data, dtype=np.float32).ravel()
        if flat.size == 0:
            return
        self.count += flat.size
        self.min_val = min(self.min_val, float(flat.min()))
        self.max_val = max(self.max_val, float(flat.max()))
        self.sum_val += float(flat.sum(dtype=np.float64))
        self.sum_sq += float(np.square(flat, dtype=np.float64).sum())
        tol = self._tol
        self.mid_count += int(np.count_nonzero((flat > tol) & (flat < 1.0 - tol)))
        self.near_zero += int(np.count_nonzero(flat <= tol))
        self.near_one += int(np.count_nonzero(flat >= 1.0 - tol))

        if self._hist_counts is not None:
            counts, _ = np.histogram(flat, bins=self._hist_bins)
            self._hist_counts += counts

        if len(self._unique_set) < self._max_unique:
            uniques = np.unique(flat)
            remaining = self._max_unique - len(self._unique_set)
            if uniques.size > remaining:
                uniques = self._rng.choice(uniques, size=remaining, replace=False)
            for val in np.asarray(uniques, dtype=np.float32):
                cast_val = float(val)
                if len(self._unique_set) >= self._max_unique:
                    break
                if cast_val not in self._unique_set:
                    self._unique_set.add(cast_val)

    def as_dict(self) -> dict:
        if self.count == 0:
            return {}
        mean = self.sum_val / self.count
        variance = (self.sum_sq / self.count) - (mean * mean)
        std = np.sqrt(max(variance, 0.0))
        return {
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": mean,
            "std": std,
            "mid_fraction": self.mid_count / self.count,
            "near_zero_fraction": self.near_zero / self.count,
            "near_one_fraction": self.near_one / self.count,
            "unique_values": sorted(self._unique_set),
            "hist_counts": self._hist_counts,
        }


def _choose_indices(total: int, sample: int, rng: np.random.Generator) -> np.ndarray:
    if sample <= 0 or sample >= total:
        return np.arange(total, dtype=np.int64)
    indices = rng.choice(total, size=sample, replace=False)
    indices.sort()
    return indices.astype(np.int64, copy=False)


def _select_run(group: zarr.Group, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in group:
            raise ValueError(f"Run '{explicit}' not found under eye_masks_runs")
        return explicit
    latest = group.attrs.get("latest")
    if not latest:
        raise ValueError("No runs recorded under eye_masks_runs and no --run provided.")
    return latest


def _build_stats_collectors(
    *,
    num_eyes: int,
    tol: float,
    max_unique: int,
    bins: Optional[int],
    rng: np.random.Generator,
    per_eye: bool,
) -> tuple[RunningStats, List[RunningStats]]:
    hist_bins = None
    if bins and bins > 0:
        hist_bins = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    overall = RunningStats(tol=tol, max_unique=max_unique, hist_bins=hist_bins, rng=rng)
    per_eye_stats: List[RunningStats] = []
    if per_eye:
        per_eye_stats = [
            RunningStats(tol=tol, max_unique=max_unique, hist_bins=hist_bins, rng=rng) for _ in range(num_eyes)
        ]
    return overall, per_eye_stats


def _summarize(stats: RunningStats, label: str) -> None:
    summary = stats.as_dict()
    if not summary:
        print(f"{label}: no data")
        return
    print(f"\n{label}:")
    print(f"  count={summary['count']}")
    print(f"  min={summary['min']:.6f} max={summary['max']:.6f}")
    print(f"  mean={summary['mean']:.6f} std={summary['std']:.6f}")
    print(f"  mid_fraction={summary['mid_fraction']:.6f}")
    print(f"  near_zero_fraction={summary['near_zero_fraction']:.6f}")
    print(f"  near_one_fraction={summary['near_one_fraction']:.6f}")
    uniques = summary["unique_values"]
    if uniques:
        print(f"  sample_unique_values (≤{len(uniques)}): {', '.join(f'{u:.6f}' for u in uniques)}")
    hist_counts = summary.get("hist_counts")
    if hist_counts is not None:
        total = hist_counts.sum()
        if total > 0:
            fractions = hist_counts / total
            frac_str = ", ".join(f"{f:.4f}" for f in fractions)
            print(f"  histogram_fraction={frac_str}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample mask_probs_roi from a YOLO eye segmentation run and report probability statistics."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--run", help="Specific eye_masks_runs/<run>. Defaults to latest.")
    parser.add_argument(
        "--sample",
        type=int,
        default=128,
        help="Number of ROI entries to sample (use 0 or negative to scan the full run).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed when sampling.")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for treating probabilities as exactly 0 or 1 when computing mid-range fractions.",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=20,
        help="Maximum number of representative unique values to report.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=0,
        help="Optional number of bins for a [0,1] histogram (0 disables histogram).",
    )
    parser.add_argument(
        "--per-eye",
        action="store_true",
        help="Report statistics separately for each eye channel.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of ROI entries to process per batch (improves throughput for large samples).",
    )
    args = parser.parse_args(argv)

    if not args.zarr_path.exists():
        raise FileNotFoundError(args.zarr_path)

    root = zarr.open(str(args.zarr_path), mode="r")
    if "eye_masks_runs" not in root:
        raise ValueError("Zarr archive has no 'eye_masks_runs' group.")
    runs_group = root["eye_masks_runs"]
    run_name = _select_run(runs_group, args.run)
    run_group = runs_group[run_name]

    if "mask_probs_roi" not in run_group:
        raise ValueError(f"Run '{run_name}' does not contain mask_probs_roi.")

    arr = run_group["mask_probs_roi"]
    num_rois = arr.shape[0]
    if num_rois == 0:
        raise ValueError(f"Run '{run_name}' has no ROI entries.")
    num_eyes = arr.shape[1] if arr.ndim >= 2 else 1

    rng = np.random.default_rng(args.seed)
    indices = _choose_indices(num_rois, args.sample, rng)
    overall_stats, per_eye_stats = _build_stats_collectors(
        num_eyes=num_eyes,
        tol=args.tol,
        max_unique=args.max_unique,
        bins=args.hist_bins,
        rng=rng,
        per_eye=args.per_eye,
    )

    print(f"Inspecting eye_masks_runs/{run_name}")
    print(f"Total ROIs: {num_rois}, sampled: {len(indices)}")

    batch_size = max(1, args.batch_size)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch_idx = indices[start:stop]
        selection = (batch_idx,) + (slice(None),) * (arr.ndim - 1)
        try:
            getter = arr.get_orthogonal_selection
        except AttributeError:
            block = np.asarray(arr.oindex[batch_idx], dtype=np.float32)
        else:
            block = np.asarray(getter(selection), dtype=np.float32)
        overall_stats.update(block)
        if per_eye_stats:
            for eye_idx, eye_stats in enumerate(per_eye_stats):
                eye_stats.update(block[:, eye_idx])

    _summarize(overall_stats, "mask_probs_roi (combined)")
    if per_eye_stats:
        for eye_idx, eye_stats in enumerate(per_eye_stats):
            _summarize(eye_stats, f"mask_probs_roi eye_index={eye_idx}")


if __name__ == "__main__":
    main()
