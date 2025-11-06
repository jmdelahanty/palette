#!/usr/bin/env python3
"""Check stimulus frame alignment in Palette stimulus H5 files and Zarr archives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import zarr


@dataclass
class AlignmentSummary:
    has_data: bool
    total_rows: int = 0
    unique_camera_frames: int = 0
    camera_min: Optional[int] = None
    camera_max: Optional[int] = None
    missing_camera_frames: int = 0
    unique_camera: np.ndarray | None = None
    first_stimulus: np.ndarray | None = None
    per_camera_counts: np.ndarray | None = None
    ratios: np.ndarray | None = None
    delta_camera: np.ndarray | None = None
    delta_stimulus: np.ndarray | None = None
    preview_camera: np.ndarray | None = None
    preview_stimulus: np.ndarray | None = None
    sort_order: np.ndarray | None = None


def _compute_alignment_summary(
    camera_frames: np.ndarray,
    stimulus_frames: np.ndarray,
    preview: int,
) -> AlignmentSummary:
    if camera_frames.size == 0 or stimulus_frames.size == 0:
        return AlignmentSummary(has_data=False)

    camera_frames = camera_frames.astype(np.int64, copy=False)
    stimulus_frames = stimulus_frames.astype(np.int64, copy=False)

    order = np.argsort(camera_frames, kind="mergesort")
    camera_sorted = camera_frames[order]
    stimulus_sorted = stimulus_frames[order].astype(np.int64, copy=False)

    unique_camera, first_indices = np.unique(camera_sorted, return_index=True)
    unique_camera = unique_camera.astype(np.int64, copy=False)
    counts_per_camera = np.diff(np.append(first_indices, camera_sorted.size))

    first_stimulus = stimulus_sorted[first_indices].astype(np.int64, copy=False)
    if unique_camera.size >= 2:
        delta_camera = np.diff(unique_camera)
        delta_stimulus = np.diff(first_stimulus)
        valid = delta_camera != 0
        ratios = delta_stimulus[valid] / delta_camera[valid] if np.any(valid) else np.zeros(0, dtype=np.float64)
    else:
        delta_camera = np.zeros(0, dtype=np.int64)
        delta_stimulus = np.zeros(0, dtype=np.int64)
        ratios = np.zeros(0, dtype=np.float64)

    cam_min = int(unique_camera[0])
    cam_max = int(unique_camera[-1])
    expected_span = cam_max - cam_min + 1
    missing = expected_span - unique_camera.size

    preview_limit = min(preview, camera_sorted.size)
    preview_camera = camera_sorted[:preview_limit]
    preview_stimulus = stimulus_sorted[:preview_limit]

    return AlignmentSummary(
        has_data=True,
        total_rows=int(camera_sorted.size),
        unique_camera_frames=int(unique_camera.size),
        camera_min=cam_min,
        camera_max=cam_max,
        missing_camera_frames=int(missing),
        unique_camera=unique_camera,
        first_stimulus=first_stimulus,
        per_camera_counts=counts_per_camera,
        ratios=ratios,
        delta_camera=delta_camera,
        delta_stimulus=delta_stimulus,
        preview_camera=preview_camera,
        preview_stimulus=preview_stimulus,
        sort_order=order,
    )


def _print_ratio_stats(
    label: str,
    summary: AlignmentSummary,
    expected_ratio: float,
    warn_threshold: float,
) -> None:
    if not summary.has_data:
        print(f"\n[{label}] No camera/stimulus pairs found.")
        return

    print(f"\n[{label}]")
    print(f"  Rows: {summary.total_rows}")
    print(f"  Unique camera frames: {summary.unique_camera_frames}")
    print(
        f"  Camera frame range: {summary.camera_min} – {summary.camera_max} "
        f"(missing {summary.missing_camera_frames})"
    )

    if summary.per_camera_counts is not None and summary.per_camera_counts.size:
        counts = summary.per_camera_counts.astype(np.int64, copy=False)
        mean_count = float(np.mean(counts))
        median_count = float(np.median(counts))
        print(
            "  Stimulus frames per camera: "
            f"min {counts.min()} | median {median_count:.2f} | mean {mean_count:.2f} | max {counts.max()}"
        )
        over_expected = np.count_nonzero(counts > expected_ratio + warn_threshold)
        under_expected = np.count_nonzero(counts < max(1.0, expected_ratio - warn_threshold))
        print(
            f"  Cameras over expected ({expected_ratio}+{warn_threshold}): {over_expected} | "
            f"under: {under_expected}"
        )
    else:
        print("  Stimulus frames per camera: unavailable (insufficient data)")

    ratios = summary.ratios if summary.ratios is not None else np.zeros(0, dtype=np.float64)
    if ratios.size:
        print(
            "  Δstimulus / Δcamera ratio stats: "
            f"min {ratios.min():.3f} | median {np.median(ratios):.3f} | mean {ratios.mean():.3f} | max {ratios.max():.3f}"
        )
        deviation = np.abs(ratios - expected_ratio)
        warn = np.count_nonzero(deviation > warn_threshold)
        negatives = np.count_nonzero(ratios < 0)
        zeros = np.count_nonzero(np.isclose(ratios, 0.0))
        print(
            f"  Ratios outside ±{warn_threshold} of {expected_ratio}: {warn} | "
            f"zero-steps: {zeros} | negatives: {negatives}"
        )
        if warn and summary.unique_camera is not None and summary.first_stimulus is not None:
            unique = summary.unique_camera
            first_stim = summary.first_stimulus
            delta_cam = summary.delta_camera if summary.delta_camera is not None else np.zeros(0, dtype=np.int64)
            delta_stim = summary.delta_stimulus if summary.delta_stimulus is not None else np.zeros(0, dtype=np.int64)
            warn_indices = np.flatnonzero(deviation > warn_threshold)
            top_idx = warn_indices[np.argsort(deviation[warn_indices])][-5:][::-1]
            if top_idx.size:
                print("  Worst ratio deviations (camera_a→camera_b | Δstim | ratio):")
                for idx in top_idx:
                    if idx >= ratios.size:
                        continue
                    cam_a = int(unique[idx])
                    cam_b = int(unique[idx + 1])
                    stim_a = int(first_stim[idx])
                    stim_b = int(first_stim[idx + 1])
                    stim_delta = int(delta_stim[idx]) if idx < delta_stim.size else stim_b - stim_a
                    ratio_val = ratios[idx]
                    print(f"    {cam_a} → {cam_b} | Δstim {stim_delta:+d} | ratio {ratio_val:.3f}")
    else:
        print("  Not enough unique camera frames to compute ratio statistics.")

    if summary.preview_camera is not None and summary.preview_camera.size:
        print("  Preview pairs:")
        for cam, stim in zip(summary.preview_camera, summary.preview_stimulus):
            print(f"    cam {int(cam):6d} → stim {int(stim):6d}")
    else:
        print("  Preview pairs: unavailable")


def analyze_h5(
    h5_path: Path,
    expected_ratio: float,
    warn_threshold: float,
    preview: int,
) -> None:
    if not h5_path.exists():
        print(f"\n[H5] File not found: {h5_path}")
        return

    try:
        with h5py.File(h5_path, "r") as handle:
            if "/video_metadata/frame_metadata" not in handle:
                print(f"\n[H5] Dataset /video_metadata/frame_metadata missing in {h5_path}")
                return
            data = handle["/video_metadata/frame_metadata"][:]
    except Exception as exc:  # pragma: no cover
        print(f"\n[H5] Failed to read {h5_path}: {exc}")
        return

    dtype_names: Sequence[str] = data.dtype.names or ()
    if "triggering_camera_frame_id" not in dtype_names or "stimulus_frame_num" not in dtype_names:
        print("\n[H5] Required fields not present (need triggering_camera_frame_id and stimulus_frame_num).")
        return

    camera = data["triggering_camera_frame_id"].astype(np.int64, copy=False)
    stimulus = data["stimulus_frame_num"].astype(np.int64, copy=False)

    if "timestamp_ns" in dtype_names:
        timestamps = data["timestamp_ns"].astype(np.int64, copy=False)
        order = np.lexsort((timestamps, camera))
        camera = camera[order]
        stimulus = stimulus[order]

    summary = _compute_alignment_summary(camera, stimulus, preview)
    _print_ratio_stats(f"H5 {h5_path.name}", summary, expected_ratio, warn_threshold)


def _detect_latest_run(group: zarr.Group) -> Optional[str]:
    for attr in ("latest", "latest_completed", "latest_success", "latest_run"):
        value = group.attrs.get(attr)
        if isinstance(value, str) and value in group:
            return value
    keys: List[str] = sorted(group.group_keys())
    return keys[-1] if keys else None


def analyze_zarr(
    zarr_path: Path,
    run_name: Optional[str],
    expected_ratio: float,
    warn_threshold: float,
    preview: int,
) -> None:
    if not zarr_path.exists():
        print(f"\n[Zarr] Path not found: {zarr_path}")
        return

    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:  # pragma: no cover
        print(f"\n[Zarr] Failed to open {zarr_path}: {exc}")
        return

    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        print("\n[Zarr] analysis/stimulus_runs group not found.")
        return

    runs_parent = analysis["stimulus_runs"]
    chosen = run_name or _detect_latest_run(runs_parent)
    if chosen is None or chosen not in runs_parent:
        print("\n[Zarr] Unable to determine stimulus run (use --stimulus-run).")
        return

    run_group = runs_parent[chosen]
    print(f"\n[Zarr] Using stimulus run: {chosen}")

    meta_group = run_group.get("video_metadata")
    if meta_group is None or "frame_metadata" not in meta_group:
        print("  frame_metadata dataset missing under video_metadata.")
        return
    frame_meta = meta_group["frame_metadata"]
    stimulus_frames = frame_meta["stimulus_frame_num"][:]
    camera_frames = frame_meta["triggering_camera_frame_id"][:]

    fa_group = run_group.get("frame_alignment")
    if fa_group is None or "camera_to_metadata_index" not in fa_group:
        print("  frame_alignment/camera_to_metadata_index missing.")
        return

    camera_to_meta = fa_group["camera_to_metadata_index"][:].astype(np.int64, copy=False)
    valid_mask = (camera_to_meta >= 0) & (camera_to_meta < stimulus_frames.shape[0])
    if not np.any(valid_mask):
        print("  No valid camera→metadata mappings present.")
        return

    cam_indices = np.nonzero(valid_mask)[0].astype(np.int64, copy=False)
    stim_values = stimulus_frames[camera_to_meta[valid_mask]].astype(np.int64, copy=False)
    summary = _compute_alignment_summary(cam_indices, stim_values, preview)
    _print_ratio_stats(f"Zarr {zarr_path.name}", summary, expected_ratio, warn_threshold)

    preview_limit = summary.preview_camera.size if summary.preview_camera is not None else 0
    if preview_limit and summary.sort_order is not None:
        meta_sorted = camera_to_meta[valid_mask][summary.sort_order]
        print("  Preview with metadata indices:")
        for cam, stim, meta_idx in zip(
            summary.preview_camera,
            summary.preview_stimulus,
            meta_sorted[:preview_limit],
        ):
            print(f"    cam {int(cam):6d} → meta {int(meta_idx):7d} → stim {int(stim):6d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect stimulus frame alignment ratios in Palette H5 and Zarr data.",
    )
    parser.add_argument("--h5", type=Path, help="Stimulus H5 file to inspect.")
    parser.add_argument("--zarr", type=Path, help="Palette Zarr root directory to inspect.")
    parser.add_argument(
        "--stimulus-run",
        type=str,
        help="Stimulus run name inside the Zarr archive (defaults to the latest* attribute).",
    )
    parser.add_argument(
        "--expected-ratio",
        type=float,
        default=2.0,
        help="Expected stimulus frame delta per camera frame delta (default: 2.0 for 120Hz stimulus vs 60fps camera).",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.5,
        help="Allowed deviation from expected ratio before reporting warnings (default: 0.5).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of mapping pairs to preview in each report (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.h5 and not args.zarr:
        print("Provide at least one of --h5 or --zarr.")
        return

    if args.h5:
        analyze_h5(args.h5, args.expected_ratio, args.warn_threshold, max(1, args.preview))

    if args.zarr:
        analyze_zarr(args.zarr, args.stimulus_run, args.expected_ratio, args.warn_threshold, max(1, args.preview))


if __name__ == "__main__":  # pragma: no cover
    main()
