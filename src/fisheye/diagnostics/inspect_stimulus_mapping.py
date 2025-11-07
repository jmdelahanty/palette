#!/usr/bin/env python3
"""
Inspect camera→stimulus mapping arrays to verify alignment.

This script reports:
  * First/last real mappings and observed ratio
  * Drift from expected ratio (anchored at first non-interpolated frame)
  * Distribution of per-camera stimulus deltas
  * Whether the corrected mapping matches actual `stimulus_frame_num`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


def _resolve_run(root: zarr.Group, name: Optional[str]) -> tuple[zarr.Group, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise KeyError("analysis/stimulus_runs missing.")
    runs = analysis["stimulus_runs"]
    if name:
        if name not in runs:
            raise KeyError(f"Stimulus run '{name}' not found.")
        return runs[name], name
    latest = runs.attrs.get("latest")
    if latest and latest in runs:
        return runs[latest], latest
    keys = sorted(runs.group_keys())
    if not keys:
        raise KeyError("No stimulus runs present.")
    return runs[keys[-1]], keys[-1]


def _first_last(mapping: np.ndarray, mask: Optional[np.ndarray]) -> tuple[int, float, int, float]:
    valid = mapping >= 0
    if mask is not None and mask.shape == mapping.shape:
        valid &= ~mask.astype(bool)
    indices = np.nonzero(valid)[0]
    if indices.size == 0:
        raise ValueError("No valid mappings found.")
    first_cam = int(indices[0])
    last_cam = int(indices[-1])
    return first_cam, float(mapping[first_cam]), last_cam, float(mapping[last_cam])


def _ratio(first_cam: float, first_stim: float, last_cam: float, last_stim: float) -> float:
    span = max(1.0, last_cam - first_cam)
    return (last_stim - first_stim) / span


def _drift_report(mapping: np.ndarray, mask: Optional[np.ndarray], expected_ratio: float) -> None:
    valid = mapping >= 0
    valid_idx = np.nonzero(valid)[0]
    if valid_idx.size < 2:
        print("  Not enough valid frames for drift analysis.")
        return

    if mask is not None and mask.shape == mapping.shape:
        anchor_candidates = np.nonzero(valid & (~mask.astype(bool)))[0]
        anchor_indices = anchor_candidates if anchor_candidates.size >= 2 else valid_idx
    else:
        anchor_indices = valid_idx

    first_cam = float(anchor_indices[0])
    first_stim = mapping[anchor_indices[0]]
    expected = first_stim + (valid_idx - first_cam) * expected_ratio
    drift = mapping[valid_idx] - expected
    abs_drift = np.abs(drift)

    print("  Drift stats (frames):")
    print(f"    Median {np.median(drift):+.3f} | 95th {np.percentile(abs_drift, 95):.3f} | Max {drift[np.argmax(abs_drift)]:+.3f}")


def _inspect(zarr_path: Path, run_name: Optional[str], ratio: float) -> None:
    root = zarr.open(zarr_path, mode="r")
    run, run_id = _resolve_run(root, run_name)
    print(f"Inspecting stimulus mapping for run: {run_id}")

    meta_group = run["video_metadata"]["frame_metadata"]
    stim = meta_group["stimulus_frame_num"][:].astype(np.int64, copy=False)
    stim_corr = meta_group.get("stimulus_frame_num_corrected")
    stim_corr = stim_corr[:] if stim_corr is not None else None

    alignment = run["frame_alignment"]
    cam_to_meta_corr = alignment.get("camera_to_metadata_index_corrected")
    cam_to_meta_corr = cam_to_meta_corr[:] if cam_to_meta_corr is not None else None
    cam_to_stim_corr = alignment.get("camera_to_stimulus_frame_corrected")
    cam_to_stim_corr = cam_to_stim_corr[:] if cam_to_stim_corr is not None else None
    cam_interp = alignment.get("camera_stimulus_frame_interpolated")
    cam_interp = cam_interp[:] if cam_interp is not None else None

    if cam_to_stim_corr is None:
        raise KeyError("camera_to_stimulus_frame_corrected missing.")

    try:
        first_cam, first_stim, last_cam, last_stim = _first_last(cam_to_stim_corr, cam_interp)
    except ValueError:
        raise RuntimeError("No valid entries in camera_to_stimulus_frame_corrected.")

    obs_ratio = _ratio(first_cam, first_stim, last_cam, last_stim)
    print(f"  First real mapping: camera {first_cam} → stimulus {first_stim:.0f}")
    print(f"  Last real mapping:  camera {last_cam} → stimulus {last_stim:.0f}")
    print(f"  Observed avg ratio: {obs_ratio:.4f} (expected {ratio:.4f})")

    _drift_report(cam_to_stim_corr.astype(np.float64, copy=False), cam_interp, ratio)

    if cam_to_meta_corr is not None:
        valid_meta = (cam_to_meta_corr >= 0) & (cam_to_meta_corr < stim.shape[0])
        subset = cam_to_meta_corr[valid_meta]
        mismatch = stim[subset] - cam_to_stim_corr[valid_meta]
        if stim_corr is not None:
            mismatch_corr = stim_corr[subset] - cam_to_stim_corr[valid_meta]
        else:
            mismatch_corr = None

        print("\n  Metadata vs corrected mapping:")
        print(
            f"    Using original stimulus_frame_num: median {np.median(mismatch):+.1f}, max {mismatch.max():+.1f}, min {mismatch.min():+.1f}"
        )
        if mismatch_corr is not None:
            print(
                f"    Using stimulus_frame_num_corrected: median {np.median(mismatch_corr):+.1f}, max {mismatch_corr.max():+.1f}, min {mismatch_corr.min():+.1f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect camera→stimulus mapping arrays.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="Expected stimulus frames per camera frame (default: 2.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _inspect(args.zarr_path, args.stimulus_run, ratio=args.ratio)


if __name__ == "__main__":  # pragma: no cover
    main()
