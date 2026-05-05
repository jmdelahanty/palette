"""Extract a real swim-bout window from a fully-processed recording.

Pulls 8 s of track id_0 from
``/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding`` (the
``tk_hyst4_low2_s005`` kinematics run + matching swim_bout_filtered run)
and saves it as a ``.npz`` cache that conforms to the same schema as
``synthetic_speeds.npz`` so the existing scenes can switch over by
loading a different file.

Window choice: ``[205.5 s, 213.5 s]`` of the recording. Contains 10
detected bouts including a dramatic 79.7 mm/s peak around t=210.28 s.
Both edges land in quiet inter-bout intervals, so no bout is bisected
by the window.

Usage:
    PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python \
      -m apps.manim.data.build_real_cache
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


RECORDING_ZARR = Path(
    "/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/"
    "2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr"
)
KINEMATICS_RUN = "analysis/track_kinematics_runs/offline/tk_hyst4_low2_s005"
BOUTS_RUN = (
    "analysis/swim_bout_runs/"
    "bouts_tk_hyst4_low2_s005_filtered/speed_filtered/bouts"
)
TRACK_GROUP = "tracks/id_0"

WINDOW_START_S = 205.5
WINDOW_END_S = 213.5
RECORDING_LABEL = "2026-01-28T23-15-10Z_arena_2_Feeding (track 0)"

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "real_swim_bout.npz"


def _slice_array(group: zarr.Group, name: str, lo: int, hi: int) -> np.ndarray:
    return np.asarray(group[name][lo:hi])


def build_real_cache(
    *,
    out_path: Path = CACHE_FILE,
    window_start_s: float = WINDOW_START_S,
    window_end_s: float = WINDOW_END_S,
) -> Path:
    z = zarr.open_group(str(RECORDING_ZARR), mode="r")
    track = z[f"{KINEMATICS_RUN}/{TRACK_GROUP}"]
    run_attrs = dict(z[KINEMATICS_RUN].attrs)

    fps = float(run_attrs["fps"])
    pixel_to_mm = float(run_attrs["pixel_to_mm"])
    if pixel_to_mm <= 0:
        raise RuntimeError("pixel_to_mm in run attrs is non-positive")
    px_per_mm = 1.0 / pixel_to_mm

    smooth_seconds = float(run_attrs["smoothing_seconds"])
    distance_smooth_seconds = float(
        run_attrs.get("distance_interpolation_seconds") or smooth_seconds
    )
    hysteresis_high_px = float(run_attrs["hysteresis_high_px"])
    hysteresis_low_px = float(run_attrs["hysteresis_low_px"])
    hysteresis_min_frames = int(run_attrs["hysteresis_min_frames"])

    # Slice frames covering the window.
    time_seconds = np.asarray(track["time_seconds"][:])
    keep = np.where((time_seconds >= window_start_s) & (time_seconds <= window_end_s))[0]
    if keep.size == 0:
        raise RuntimeError("No samples in window")
    lo, hi = int(keep[0]), int(keep[-1] + 1)

    # Pull arrays. Speed/distance are stored in both _px and _mm; we keep _px to
    # match the synthetic schema and the scene loader divides by px_per_mm.
    frames = _slice_array(track, "frame_indices", lo, hi).astype(np.int64)
    t = _slice_array(track, "time_seconds", lo, hi).astype(np.float64)
    position_xy = _slice_array(track, "positions_px", lo, hi).astype(np.float64)
    heading_deg = _slice_array(track, "heading_degrees", lo, hi).astype(np.float64)
    delta_frames = _slice_array(track, "delta_frames", lo, hi).astype(np.int32)
    delta_seconds = _slice_array(track, "delta_seconds", lo, hi).astype(np.float64)
    transition_valid = _slice_array(track, "transition_valid", lo, hi).astype(bool)
    transition_reason_code = _slice_array(track, "transition_reason_code", lo, hi).astype(np.int16)

    speed_raw = _slice_array(track, "speed_raw_px", lo, hi).astype(np.float32)
    speed_filtered = _slice_array(track, "speed_filtered_px", lo, hi).astype(np.float32)
    speed_smoothed = _slice_array(track, "speed_smoothed_px", lo, hi).astype(np.float32)
    speed_averaged = _slice_array(track, "speed_averaged_px", lo, hi).astype(np.float32)
    fpd_raw = _slice_array(track, "frame_path_distance_raw_px", lo, hi).astype(np.float32)
    fpd_filt = _slice_array(track, "frame_path_distance_filtered_px", lo, hi).astype(np.float32)
    fpd_smooth = _slice_array(track, "frame_path_distance_smoothed_px", lo, hi).astype(np.float32)
    cum = _slice_array(track, "cumulative_path_distance_px", lo, hi).astype(np.float32)
    # Re-zero cumulative so it starts at 0 at the window's left edge.
    cum = (cum - cum[0]).astype(np.float32)

    # Bouts inside the window.
    bouts_grp = z[BOUTS_RUN]
    bout_starts = np.asarray(bouts_grp["start_time_s"][:])
    bout_ends = np.asarray(bouts_grp["end_time_s"][:])
    bout_in_window = (bout_starts >= window_start_s) & (bout_ends <= window_end_s)
    ground_truth_bouts = np.column_stack(
        [bout_starts[bout_in_window], bout_ends[bout_in_window]]
    ).astype(np.float64)

    # Build region labels by stitching pre/bout/inter/post intervals.
    regions_labels: list[str] = []
    regions_starts: list[float] = []
    regions_ends: list[float] = []
    cursor = window_start_s
    bout_count = 0
    for s, e in ground_truth_bouts:
        if s > cursor:
            regions_labels.append("inter_bout" if bout_count > 0 else "pre_bout")
            regions_starts.append(cursor)
            regions_ends.append(float(s))
        regions_labels.append(f"bout_{bout_count}")
        regions_starts.append(float(s))
        regions_ends.append(float(e))
        cursor = float(e)
        bout_count += 1
    if cursor < window_end_s:
        regions_labels.append("post_bout")
        regions_starts.append(cursor)
        regions_ends.append(window_end_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        # Source identification
        source_recording=np.array(RECORDING_LABEL, dtype=object),
        # Trace inputs
        fps=np.float64(fps),
        px_per_mm=np.float64(px_per_mm),
        frames=frames,
        t=t,
        position_xy=position_xy,
        heading_deg=heading_deg,
        # Pipeline params actually used in production
        smooth_seconds=np.float64(smooth_seconds),
        distance_smooth_seconds=np.float64(distance_smooth_seconds),
        hysteresis_high_px=np.float64(hysteresis_high_px),
        hysteresis_low_px=np.float64(hysteresis_low_px),
        hysteresis_min_frames=np.int32(hysteresis_min_frames),
        # Pipeline outputs (already in cache from production run)
        delta_frames=delta_frames,
        delta_seconds=delta_seconds.astype(np.float32),
        transition_valid=transition_valid,
        transition_reason_code=transition_reason_code,
        speed_raw=speed_raw,
        speed_filtered=speed_filtered,
        speed_smoothed=speed_smoothed,
        speed_averaged=speed_averaged,
        frame_path_distance_raw=fpd_raw,
        frame_path_distance_filtered=fpd_filt,
        frame_path_distance_smoothed=fpd_smooth,
        cumulative_path_distance=cum,
        # Region labels
        region_labels=np.array(regions_labels, dtype=object),
        region_starts_s=np.array(regions_starts, dtype=np.float64),
        region_ends_s=np.array(regions_ends, dtype=np.float64),
        ground_truth_bouts=ground_truth_bouts,
    )
    return out_path


if __name__ == "__main__":
    out = build_real_cache()
    print(f"Wrote {out}")
    data = np.load(out, allow_pickle=True)
    print(f"  frames: {data['frames'].shape}")
    print(f"  t: {float(data['t'][0]):0.3f}..{float(data['t'][-1]):0.3f} s")
    print(f"  fps: {float(data['fps'])}")
    print(f"  px_per_mm: {float(data['px_per_mm']):0.2f}")
    sf = data["speed_filtered"].astype(np.float64) / float(data["px_per_mm"])
    print(f"  speed_filtered max: {float(np.nanmax(sf)):0.1f} mm/s")
    print(f"  ground_truth_bouts: {data['ground_truth_bouts'].shape[0]} intervals")
    print(f"  regions: {len(data['region_labels'])} ({list(data['region_labels'])[:6]}...)")
    print(f"  hysteresis (px): high={float(data['hysteresis_high_px'])}, low={float(data['hysteresis_low_px'])}, min_frames={int(data['hysteresis_min_frames'])}")
