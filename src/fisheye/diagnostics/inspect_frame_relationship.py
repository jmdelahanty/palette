"""Inspect the relationship between stimulus frame numbers and camera frame IDs.

This diagnostic helps confirm whether multiple stimulus frames map to the same
camera frame (or vice versa) by analyzing ``video_metadata/frame_metadata`` inside
an imported Palette zarr archive.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset


def _resolve_run(stim_parent: zarr.Group, requested: Optional[str]) -> str:
    if requested and requested != "latest":
        if requested not in stim_parent:
            raise ValueError(f"Stimulus run '{requested}' not found under analysis/stimulus_runs")
        return requested

    latest = stim_parent.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in stim_parent:
        return latest

    run_names = sorted(stim_parent.keys())
    if not run_names:
        raise ValueError("No stimulus runs available in analysis/stimulus_runs")
    return run_names[-1]


def _resolve_field(array: np.ndarray, *candidates: str) -> str:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Structured array missing expected field (tried {', '.join(candidates)})")


def summarize_relationship(stimulus: np.ndarray, camera: np.ndarray, console: Console) -> None:
    if stimulus.size == 0:
        console.print("[red]frame_metadata dataset is empty[/red]")
        return

    console.print(f"[bold]Total frame_metadata records:[/bold] {stimulus.size:,}")
    console.print(f"[bold]Stimulus frame range:[/bold] {int(stimulus.min())} → {int(stimulus.max())}")
    console.print(f"[bold]Camera frame range:[/bold] {int(camera.min())} → {int(camera.max())}")

    unique_stim = np.unique(stimulus)
    unique_cam = np.unique(camera)
    console.print(f"[bold]Unique stimulus frames:[/bold] {unique_stim.size:,}")
    console.print(f"[bold]Unique camera frames:[/bold] {unique_cam.size:,}")

    stim_deltas = np.diff(stimulus)
    cam_deltas = np.diff(camera)
    delta_pairs = Counter(zip(stim_deltas.tolist(), cam_deltas.tolist()))

    console.print("\n[bold]Delta distribution (stimulus Δ → camera Δ):[/bold]")
    table = Table(show_header=True)
    table.add_column("Stim Δ", justify="right")
    table.add_column("Cam Δ", justify="right")
    table.add_column("Count", justify="right")
    for (d_stim, d_cam), count in delta_pairs.most_common(10):
        table.add_row(str(d_stim), str(d_cam), f"{count:,}")
    if len(delta_pairs) > 10:
        table.caption = f"Showing top 10 of {len(delta_pairs)} unique delta pairs"
    console.print(table)

    camera_to_stim: Dict[int, List[int]] = defaultdict(list)
    for stim, cam in zip(stimulus, camera):
        camera_to_stim[int(cam)].append(int(stim))

    multiplicities = Counter(len(stims) for stims in camera_to_stim.values())
    console.print("\n[bold]Stimulus-per-camera multiplicity:[/bold]")
    multiplicity_table = Table(show_header=True)
    multiplicity_table.add_column("# Stimulus per Camera", justify="right")
    multiplicity_table.add_column("Camera Frames", justify="right")
    for multiplicity, count in sorted(multiplicities.items()):
        multiplicity_table.add_row(str(multiplicity), f"{count:,}")
    console.print(multiplicity_table)

    samples = sorted(camera_to_stim.items())[:10]
    sample_table = Table(show_header=True, title="Sample camera→stimulus mapping")
    sample_table.add_column("Camera frame", justify="right")
    sample_table.add_column("Stimulus frames")
    for cam_frame, stim_frames in samples:
        stim_list = ", ".join(str(s) for s in stim_frames)
        sample_table.add_row(str(cam_frame), stim_list)
    console.print(sample_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run to inspect (default: latest)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    root = zarr.open(str(args.zarr_path), mode="r")
    analysis = root.require_group("analysis")
    stim_parent = analysis.require_group("stimulus_runs")

    run_name = _resolve_run(stim_parent, args.stimulus_run)
    console.print(f"[bold]Inspecting stimulus run:[/bold] {run_name}")
    run_group = stim_parent[run_name]

    meta_group = run_group.require_group("video_metadata")
    frame_metadata, _ = load_structured_dataset(meta_group, "frame_metadata")
    stim_field = _resolve_field(frame_metadata, "stimulus_frame_num", "frame_number")
    camera_field = _resolve_field(frame_metadata, "triggering_camera_frame_id", "camera_frame_id")

    stimulus = np.asarray(frame_metadata[stim_field], dtype=np.int64)
    camera = np.asarray(frame_metadata[camera_field], dtype=np.int64)

    summarize_relationship(stimulus, camera, console)


if __name__ == "__main__":  # pragma: no cover
    main()
