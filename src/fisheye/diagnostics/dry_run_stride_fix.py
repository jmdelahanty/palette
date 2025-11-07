#!/usr/bin/env python3
"""Dry-run correction for high stimulus-frame strides in Palette H5 metadata.

This tool inspects ``/video_metadata/frame_metadata`` inside a stimulus H5 file,
detects the current camera→stimulus mapping cadence, and simulates the
sequential stimulus-frame numbering we would assign if we compressed the stride
back down so that every metadata row simply receives the next integer.

It never modifies the input file.  Instead it prints statistics comparing the
existing stimulus_frame_num cadence against the simulated one and, optionally,
exports a CSV that maps each record's original stimulus_frame_num to the
proposed replacement so the fix can be applied manually or by another script.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


def _resolve_field(names: Sequence[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Could not locate any of {candidates} in dataset fields: {names}")


def _pick_timestamp_field(names: Sequence[str]) -> Optional[str]:
    for candidate in ("timestamp_ns", "timestamp_ns_session", "relative_timestamp_ns"):
        if candidate in names:
            return candidate
    return None


def _compute_ratio_summary(camera_ids: np.ndarray, first_stim: np.ndarray) -> Optional[Dict[str, float]]:
    if camera_ids.size < 2:
        return None
    diffs_cam = np.diff(camera_ids.astype(np.int64, copy=False))
    diffs_stim = np.diff(first_stim.astype(np.int64, copy=False))
    valid = diffs_cam != 0
    if not np.any(valid):
        return None
    ratios = diffs_stim[valid] / diffs_cam[valid]
    return {
        "count": int(ratios.size),
        "min": float(ratios.min(initial=np.nan)),
        "max": float(ratios.max(initial=np.nan)),
        "median": float(np.median(ratios)),
        "mean": float(ratios.mean()),
        "zeros": int(np.count_nonzero(diffs_stim == 0)),
        "negatives": int(np.count_nonzero(diffs_stim < 0)),
    }


def _print_ratio(label: str, summary: Optional[Dict[str, float]]) -> None:
    if not summary:
        print(f"{label}: insufficient data for ratio statistics.")
        return
    print(f"{label}:")
    print(f"  samples: {summary['count']}")
    print(f"  min / median / max: {summary['min']:.4f} / {summary['median']:.4f} / {summary['max']:.4f}")
    print(f"  mean: {summary['mean']:.4f}")
    if summary["zeros"] or summary["negatives"]:
        print(f"  zero stimulus deltas: {summary['zeros']}")
        print(f"  negative stimulus deltas: {summary['negatives']}")


def _assign_sequential_values(counts: np.ndarray, start_value: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequential stimulus assignments per record and per camera."""
    total = int(counts.sum())
    new_values = np.empty(total, dtype=np.int64)
    per_camera_first = np.empty(counts.size, dtype=np.int64)
    per_camera_last = np.empty(counts.size, dtype=np.int64)
    cursor = 0
    next_value = int(start_value)
    for idx, count in enumerate(counts):
        span = slice(cursor, cursor + int(count))
        seq = np.arange(next_value, next_value + int(count), dtype=np.int64)
        new_values[span] = seq
        per_camera_first[idx] = seq[0]
        per_camera_last[idx] = seq[-1]
        next_value += int(count)
        cursor += int(count)
    return new_values, per_camera_first, per_camera_last


def _export_csv(
    target: Path,
    order: np.ndarray,
    cam_sorted: np.ndarray,
    stim_sorted: np.ndarray,
    new_stim_sorted: np.ndarray,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["record_index", "camera_frame_id", "original_stimulus_frame", "proposed_stimulus_frame"]
        )
        for sorted_idx, original_idx in enumerate(order):
            writer.writerow(
                [
                    int(original_idx),
                    int(cam_sorted[sorted_idx]),
                    int(stim_sorted[sorted_idx]),
                    int(new_stim_sorted[sorted_idx]),
                ]
            )
    print(f"[dry-run] Wrote proposed mapping to {target}")


def _preview_mappings(
    camera_ids: np.ndarray,
    start_indices: np.ndarray,
    counts: np.ndarray,
    stim_sorted: np.ndarray,
    new_sorted: np.ndarray,
    limit: int,
) -> None:
    if limit <= 0:
        return
    limit = min(limit, camera_ids.size)
    print(f"\nPreview of first {limit} camera frames:")
    print("  cam_id   count   original (first→last)   proposed (first→last)")
    for i in range(limit):
        start = int(start_indices[i])
        count = int(counts[i])
        stop = start + count
        old_vals = stim_sorted[start:stop]
        new_vals = new_sorted[start:stop]
        print(
            f"  {int(camera_ids[i]):6d}   {count:5d}   "
            f"{int(old_vals[0]):6d} → {int(old_vals[-1]):6d}   "
            f"{int(new_vals[0]):6d} → {int(new_vals[-1]):6d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a stride correction for stimulus frame numbers inside a Palette H5 file."
    )
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus H5 file.")
    parser.add_argument(
        "--stimulus-field",
        help="Override the stimulus frame field (default: auto-detect common names).",
    )
    parser.add_argument(
        "--camera-field",
        help="Override the camera frame field (default: auto-detect common names).",
    )
    parser.add_argument(
        "--timestamp-field",
        help="Override the timestamp field for intra-camera ordering (default: auto-detect).",
    )
    parser.add_argument(
        "--start-stim",
        type=int,
        default=1,
        help="Stimulus frame number to assign to the first metadata record (default: 1).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="How many camera frames to preview in the console output (default: 10).",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Optional path to write a CSV mapping (record index, camera frame, original stim, proposed stim).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.h5_path.exists():
        raise SystemExit(f"H5 file not found: {args.h5_path}")

    with h5py.File(args.h5_path, "r") as handle:
        if "/video_metadata/frame_metadata" not in handle:
            raise SystemExit("Dataset '/video_metadata/frame_metadata' not found in H5.")
        frame_meta = handle["/video_metadata/frame_metadata"]
        dtype_names = frame_meta.dtype.names or ()

        stim_field = args.stimulus_field or _resolve_field(
            dtype_names,
            "stimulus_frame_num",
            "frame_number",
            "stim_frame_num",
        )
        camera_field = args.camera_field or _resolve_field(
            dtype_names,
            "triggering_camera_frame_id",
            "camera_frame_id",
        )
        timestamp_field = (
            args.timestamp_field
            if args.timestamp_field is not None
            else _pick_timestamp_field(dtype_names)
        )

        stimulus = np.asarray(frame_meta[stim_field], dtype=np.int64)
        camera = np.asarray(frame_meta[camera_field], dtype=np.int64)
        if timestamp_field is not None:
            timestamps = np.asarray(frame_meta[timestamp_field], dtype=np.int64)
        else:
            timestamps = np.arange(stimulus.shape[0], dtype=np.int64)

    order = np.lexsort((timestamps, camera))
    cam_sorted = camera[order]
    stim_sorted = stimulus[order]

    unique_cams, start_indices, counts = np.unique(
        cam_sorted, return_index=True, return_counts=True
    )
    old_first = stim_sorted[start_indices]
    old_ratio = _compute_ratio_summary(unique_cams, old_first)

    new_values_sorted, new_first, new_last = _assign_sequential_values(counts, args.start_stim)
    new_ratio = _compute_ratio_summary(unique_cams, new_first)

    print(f"Loaded metadata records: {stimulus.size:,}")
    print(f"Unique camera frames: {unique_cams.size:,}")
    print(
        f"Stimulus frame range: {int(stimulus.min())} → {int(stimulus.max())} "
        f"(proposed range {int(new_first[0])} → {int(new_last[-1])})"
    )
    print(f"Camera frame range: {int(unique_cams.min())} → {int(unique_cams.max())}")

    unique_counts, count_freq = np.unique(counts, return_counts=True)
    distribution = ", ".join(f"{int(val)}×{int(freq)}" for val, freq in zip(unique_counts, count_freq))
    print(f"Records per camera distribution: {distribution}")
    print(f"Average records per camera frame: {counts.mean():.3f}")

    print("\nExisting stride statistics (first stimulus per camera):")
    _print_ratio("  current", old_ratio)

    print("\nSimulated stride statistics after sequential reassignment:")
    _print_ratio("  proposed", new_ratio)

    _preview_mappings(
        unique_cams,
        start_indices,
        counts,
        stim_sorted,
        new_values_sorted,
        args.preview,
    )

    if args.export_csv:
        _export_csv(
            args.export_csv,
            order,
            cam_sorted,
            stim_sorted,
            new_values_sorted,
        )


if __name__ == "__main__":
    main()
