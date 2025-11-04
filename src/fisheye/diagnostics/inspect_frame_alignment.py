#!/usr/bin/env python3
"""
Inspect camera ↔ stimulus frame alignment stored in a Palette Zarr archive.

This utility mirrors the diagnostic steps outlined in the data-generation prompt,
but uses the ``zarr`` library directly instead of TensorStore.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import zarr


def _load_latest_stimulus_run(root: zarr.Group) -> tuple[zarr.Group, str]:
    """Return the stimulus run group and its name (preferring attrs['latest'])."""
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise KeyError("analysis/stimulus_runs group not found in Zarr archive.")

    runs = analysis["stimulus_runs"]
    run_name = runs.attrs.get("latest")
    if run_name:
        run_name = str(run_name)
    else:
        keys = sorted(runs.group_keys())
        if not keys:
            raise KeyError("No runs available under analysis/stimulus_runs.")
        run_name = keys[0]

    if run_name not in runs:
        raise KeyError(f"Stimulus run '{run_name}' not found in analysis/stimulus_runs.")

    return runs[run_name], run_name


def _resolve_stimulus_run(root: zarr.Group, name: Optional[str]) -> tuple[zarr.Group, str]:
    if name is None:
        return _load_latest_stimulus_run(root)

    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise KeyError("analysis/stimulus_runs group not found in Zarr archive.")

    runs = analysis["stimulus_runs"]
    if name not in runs:
        raise KeyError(f"Stimulus run '{name}' not found in analysis/stimulus_runs.")
    return runs[name], name


def _extract_columnar(root: zarr.Group, dataset: str) -> np.ndarray:
    if dataset not in root:
        raise KeyError(f"Dataset '{dataset}' not found in {root.path}.")
    array = root[dataset]
    return np.asarray(array[:])


def analyze_alignment(
    zarr_path: Path,
    run_name: Optional[str],
    *,
    preview_frames: int = 100,
) -> None:
    """Run alignment diagnostics and print a textual report."""
    store = zarr.open(zarr_path, mode="r")
    run_group, chosen_name = _resolve_stimulus_run(store, run_name)

    fa_group = run_group.get("frame_alignment")
    if fa_group is None:
        raise KeyError(f"frame_alignment group missing in stimulus run '{chosen_name}'.")

    cam_to_meta = _extract_columnar(fa_group, "camera_to_metadata_index")
    cam_mask = _extract_columnar(fa_group, "camera_interpolation_mask")

    meta_group = run_group.require_group("video_metadata").require_group("frame_metadata")
    stim_frames = _extract_columnar(meta_group, "stimulus_frame_num")
    trig_frames = _extract_columnar(meta_group, "triggering_camera_frame_id")

    max_preview = min(preview_frames, cam_to_meta.shape[0])
    print(f"Zarr: {zarr_path}")
    print(f"Stimulus run: {chosen_name}")
    print(f"Previewing first {max_preview} camera frames\n")

    for cam_idx in range(max_preview):
        meta_idx = cam_to_meta[cam_idx]
        mask_val = cam_mask[cam_idx] if cam_idx < cam_mask.shape[0] else None
        if 0 <= meta_idx < stim_frames.shape[0]:
            stim_frame = int(stim_frames[meta_idx])
            trig_frame = int(trig_frames[meta_idx])
            print(
                f"  cam {cam_idx:5d} → meta {meta_idx:5d} → stimulus {stim_frame:6d} "
                f"(trigger {trig_frame:6d}, interpolated={bool(mask_val)})"
            )
        else:
            print(f"  cam {cam_idx:5d} → meta {meta_idx:5d} → INVALID (interpolated={bool(mask_val)})")

    print("\nSummary:")
    valid_mask = (cam_to_meta >= 0) & (cam_to_meta < stim_frames.shape[0])
    valid_cam_indices = np.flatnonzero(valid_mask)
    valid_indices = cam_to_meta[valid_mask]
    if valid_indices.size:
        first_meta_idx = int(valid_indices[0])
        first_stim_frame = int(stim_frames[first_meta_idx])
        first_cam_frame = int(valid_cam_indices[0])
        print(f"  First valid mapping: camera {first_cam_frame} → meta {first_meta_idx} → stimulus {first_stim_frame}")
    else:
        print("  No valid camera-to-metadata mappings found.")

    valid_stimulus = stim_frames[stim_frames >= 0]
    if valid_stimulus.size:
        min_stim = int(valid_stimulus.min())
        print(f"  Minimum stimulus frame number: {min_stim}")

        if valid_indices.size:
            stim_matches = np.flatnonzero(stim_frames[valid_indices] == min_stim)
            if stim_matches.size:
                print(f"  Stimulus frame {min_stim} first observed at camera frame {int(valid_cam_indices[stim_matches[0]])}")
    else:
        print("  No non-negative stimulus frame numbers found.")

    offset_attr = fa_group.attrs.get("camera_frame_offset", None)
    if offset_attr is not None:
        offset_val = int(offset_attr)
        print(f"  camera_frame_offset attribute: {offset_val}")
    else:
        print("  camera_frame_offset attribute not present.")
        offset_val = 0

    uses_offset = False
    if valid_cam_indices.size:
        first_idx = valid_cam_indices[0]
        if offset_attr is not None and first_idx == 0 and offset_val != 0:
            uses_offset = True

    mismatch = []
    max_check = min(cam_to_meta.shape[0], trig_frames.shape[0])
    for cam_idx in range(max_check):
        meta_idx = cam_to_meta[cam_idx]
        if 0 <= meta_idx < trig_frames.shape[0]:
            trig_frame = trig_frames[meta_idx]
            expected = cam_idx + offset_val if uses_offset else cam_idx
            if trig_frame != expected:
                mismatch.append(cam_idx)
                if len(mismatch) >= 5:
                    break
    if mismatch:
        print(f"  First mismatched trigger indices (up to 5): {mismatch}")
    else:
        print("  Trigger frame IDs align with camera indices + offset for checked range.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect camera-to-stimulus frame alignment inside a Palette Zarr archive.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr root directory.")
    parser.add_argument("--stimulus-run", type=str, help="Stimulus run name (defaults to latest).")
    parser.add_argument(
        "--preview-frames",
        type=int,
        default=100,
        help="Number of initial camera frames to print (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zarr_path = args.zarr_path.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    analyze_alignment(
        zarr_path=zarr_path,
        run_name=args.stimulus_run,
        preview_frames=max(1, args.preview_frames),
    )


if __name__ == "__main__":
    main()
