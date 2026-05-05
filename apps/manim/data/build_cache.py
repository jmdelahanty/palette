"""Run the synthetic trace through the real pipeline and cache outputs.

Run this once after editing the synthetic trace or after changing pipeline
parameters. Scenes never import ``fisheye`` directly — they load the
``.npz`` cache so that the manim render env only needs ``manim`` + numpy.

Usage:
    PYTHONPATH=src ~/miniconda3/envs/palette/bin/python -m apps.manim.data.build_cache
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apps.manim.data.synthetic_trace import (
    PX_PER_MM_DEFAULT,
    SyntheticTrace,
    generate_synthetic_trace,
)


CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "synthetic_speeds.npz"


def build_cache(
    trace: SyntheticTrace | None = None,
    *,
    smooth_seconds: float = 0.10,
    distance_smooth_seconds: float = 0.05,
    hysteresis_high_px: float = 2.0,
    hysteresis_low_px: float = 1.0,
    hysteresis_min_frames: int = 3,
    out_path: Path = CACHE_FILE,
) -> Path:
    """Build the .npz cache and return its path."""
    from fisheye.analysis.compute_speed import compute_track_speed

    if trace is None:
        trace = generate_synthetic_trace()

    speeds = compute_track_speed(
        frames=trace.frames,
        positions=trace.position_xy,
        fps=trace.fps,
        smooth_seconds=smooth_seconds,
        distance_smooth_seconds=distance_smooth_seconds,
        hysteresis_high_px=hysteresis_high_px,
        hysteresis_low_px=hysteresis_low_px,
        hysteresis_min_frames=hysteresis_min_frames,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        # Trace inputs
        fps=np.float64(trace.fps),
        px_per_mm=np.float64(trace.px_per_mm),
        frames=trace.frames,
        t=trace.t,
        position_xy=trace.position_xy,
        heading_deg=trace.heading_deg,
        # Pipeline params used
        smooth_seconds=np.float64(smooth_seconds),
        distance_smooth_seconds=np.float64(distance_smooth_seconds),
        hysteresis_high_px=np.float64(hysteresis_high_px),
        hysteresis_low_px=np.float64(hysteresis_low_px),
        hysteresis_min_frames=np.int32(hysteresis_min_frames),
        # Pipeline outputs
        delta_frames=speeds.delta_frames,
        delta_seconds=speeds.delta_seconds,
        transition_valid=speeds.transition_valid,
        transition_reason_code=speeds.transition_reason_code,
        speed_raw=speeds.speed_raw,
        speed_filtered=speeds.speed_filtered,
        speed_smoothed=speeds.speed_smoothed,
        speed_averaged=speeds.speed_averaged,
        frame_path_distance_raw=speeds.frame_path_distance_raw,
        frame_path_distance_filtered=speeds.frame_path_distance_filtered,
        frame_path_distance_smoothed=speeds.frame_path_distance_smoothed,
        cumulative_path_distance=speeds.cumulative_path_distance,
        # Region labels (object array of strings)
        region_labels=np.array([r.label for r in trace.regions], dtype=object),
        region_starts_s=np.array([r.start_s for r in trace.regions], dtype=np.float64),
        region_ends_s=np.array([r.end_s for r in trace.regions], dtype=np.float64),
        ground_truth_bouts=np.array(trace.ground_truth_bouts, dtype=np.float64),
    )
    return out_path


def load_cache(path: Path = CACHE_FILE) -> dict:
    """Load the cache as a plain dict; convertible to scene-local containers."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found at {path}. Build it via "
            f"`PYTHONPATH=src ~/miniconda3/envs/palette/bin/python -m apps.manim.data.build_cache`."
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    out = build_cache()
    cache = load_cache(out)
    print(f"Wrote {out}")
    print(f"Contains {len(cache)} arrays:")
    for key in sorted(cache.keys()):
        value = cache[key]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            print(f"  {key}: shape={value.shape} dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    # Sanity peek
    px_per_mm = float(cache["px_per_mm"])
    speed_filtered_mm_s = cache["speed_filtered"].astype(np.float64) / px_per_mm
    print(f"\nFiltered max mm/s: {np.nanmax(speed_filtered_mm_s):.2f}")
