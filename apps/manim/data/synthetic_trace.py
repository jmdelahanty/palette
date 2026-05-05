"""Synthetic single-track trajectory used as input to manim teaching scenes.

The trace is engineered to exercise every branch of the speed pipeline in
``src/fisheye/analysis/compute_speed.py`` so animations can show the real
function output rather than a hand-tuned cartoon.

Layout (default seed = 0, 60 fps, 8 s, calibration 51 px/mm):

  region            t window     contents
  ----------------- ------------ --------------------------------------------
  still_pre         [0.0, 1.0)   stationary, sub-pixel detection jitter
  clean_bout        [1.0, 2.5)   single gaussian-shaped swim (~120 mm/s peak)
  noisy_threshold   [2.5, 4.0)   hovering near hysteresis high; per-frame
                                 detection noise occasionally crosses
  multi_peak_bout   [4.0, 6.0)   two close peaks with shallow valley between
                                 (peak-event refinement target)
  gap               [6.0, 6.5)   missing frames (detection failure)
  still_post        [6.5, 8.0]   stationary again

The trace returns frames with the gap region's frames *removed*, so the
``frames`` array has int spacing == 1 within each segment but a jump at the
gap. ``compute_track_speed`` flags that jump as ``TRANSITION_REASON_FRAME_GAP``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


PX_PER_MM_DEFAULT = 51.0
FPS_DEFAULT = 60.0
DURATION_S_DEFAULT = 8.0


@dataclass(frozen=True)
class Region:
    """Named time interval inside the synthetic trace."""

    label: str
    start_s: float
    end_s: float


@dataclass
class SyntheticTrace:
    """Single-track synthetic trajectory and metadata.

    Attributes
    ----------
    fps:
        Frames per second of the source clip.
    px_per_mm:
        Calibration constant — number of pixels in one millimeter at the
        focal plane. Multiply pixel speeds by ``1 / px_per_mm`` to get
        mm/s.
    frames:
        ``int64`` frame indices. Strictly increasing; a contiguous gap
        appears at the ``gap`` region.
    t:
        ``float64`` seconds (= frames / fps).
    position_xy:
        ``(N, 2) float64`` per-frame pixel positions of the keypoint we
        feed to ``compute_track_speed``.
    heading_deg:
        ``(N,) float64`` per-frame body heading in degrees, CCW from +x.
    regions:
        Named time intervals (see module docstring).
    ground_truth_bouts:
        ``[(start_s, end_s), ...]`` intervals where the fish is actually
        swimming, for sanity-check overlays in scenes.
    """

    fps: float
    px_per_mm: float
    frames: np.ndarray
    t: np.ndarray
    position_xy: np.ndarray
    heading_deg: np.ndarray
    regions: List[Region]
    ground_truth_bouts: List[Tuple[float, float]]


def _gaussian_envelope(
    t: np.ndarray, peak_s: float, peak_mm_s: float, width_s: float
) -> np.ndarray:
    """Smooth gaussian speed envelope centered at ``peak_s``."""
    return peak_mm_s * np.exp(-0.5 * ((t - peak_s) / width_s) ** 2)


def _build_speed_envelope_mm_s(
    t: np.ndarray,
    *,
    clean_peak_s: float = 1.75,
    clean_peak_mm_s: float = 120.0,
    clean_width_s: float = 0.18,
    multi_peak_a_s: float = 4.55,
    multi_peak_b_s: float = 5.35,
    multi_peak_mm_s: float = 150.0,
    multi_width_s: float = 0.20,
) -> np.ndarray:
    """True instantaneous speed (mm/s) along the synthetic trajectory.

    Stationary regions are exactly zero. Bouts are gaussian envelopes; the
    multi-peak bout is a sum of two gaussians whose valley sits well above
    the hysteresis floor but well below the peak — the prominence/width
    refinement target.
    """
    speed = np.zeros_like(t)
    # Single bout
    speed += _gaussian_envelope(t, clean_peak_s, clean_peak_mm_s, clean_width_s)
    # Two-peak bout
    speed += _gaussian_envelope(t, multi_peak_a_s, multi_peak_mm_s, multi_width_s)
    speed += _gaussian_envelope(t, multi_peak_b_s, multi_peak_mm_s, multi_width_s)
    # Zero out outside [1.0, 2.5) and [4.0, 6.0) — the gaussian tails are
    # negligible there but make the still regions exactly flat for clarity.
    swim_mask = ((t >= 1.0) & (t < 2.5)) | ((t >= 4.0) & (t < 6.0))
    speed = np.where(swim_mask, speed, 0.0)
    return speed


def _build_heading_deg(t: np.ndarray) -> np.ndarray:
    """Slowly varying heading: small turn during each bout, otherwise flat."""
    heading = np.zeros_like(t)
    # During the clean bout, drift heading by 30 deg
    in_clean = (t >= 1.0) & (t < 2.5)
    if np.any(in_clean):
        local_t = t[in_clean]
        progress = (local_t - 1.0) / 1.5
        heading[in_clean] = 30.0 * progress
    heading[t >= 2.5] = 30.0
    # During the multi-peak bout, swing back the other way
    in_multi = (t >= 4.0) & (t < 6.0)
    if np.any(in_multi):
        local_t = t[in_multi]
        progress = (local_t - 4.0) / 2.0
        heading[in_multi] = 30.0 - 60.0 * progress
    heading[t >= 6.0] = -30.0
    return heading


def _integrate_position_px(
    speed_mm_s: np.ndarray,
    heading_deg: np.ndarray,
    fps: float,
    px_per_mm: float,
    start_xy_px: Tuple[float, float] = (200.0, 200.0),
) -> np.ndarray:
    """Integrate (speed, heading) into a pixel-space trajectory."""
    dt = 1.0 / fps
    heading_rad = np.deg2rad(heading_deg)
    velocity_px_per_s = speed_mm_s * px_per_mm
    vx = velocity_px_per_s * np.cos(heading_rad)
    vy = velocity_px_per_s * np.sin(heading_rad)
    # Forward Euler integration; speed envelope is smooth so this is fine
    dx = vx * dt
    dy = vy * dt
    x = np.cumsum(dx) + start_xy_px[0]
    y = np.cumsum(dy) + start_xy_px[1]
    return np.stack([x, y], axis=1)


def _add_detection_noise(
    position_xy: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    *,
    still_sigma_px: float = 0.45,
    noisy_sigma_px: float = 1.6,
) -> np.ndarray:
    """Add per-frame detection noise. Heavier in the noisy_threshold region."""
    sigma = np.full(t.shape, still_sigma_px)
    in_noisy = (t >= 2.5) & (t < 4.0)
    sigma[in_noisy] = noisy_sigma_px
    noise = rng.normal(loc=0.0, scale=1.0, size=position_xy.shape) * sigma[:, None]
    return position_xy + noise


def generate_synthetic_trace(
    *,
    seed: int = 0,
    fps: float = FPS_DEFAULT,
    duration_s: float = DURATION_S_DEFAULT,
    px_per_mm: float = PX_PER_MM_DEFAULT,
) -> SyntheticTrace:
    """Generate the deterministic synthetic single-track trace.

    See module docstring for the layout.
    """
    rng = np.random.default_rng(seed)

    # Dense (no-gap) frame timeline first
    n_frames_total = int(round(fps * duration_s))
    frames_full = np.arange(n_frames_total, dtype=np.int64)
    t_full = frames_full / fps

    speed_mm_s = _build_speed_envelope_mm_s(t_full)
    heading_deg = _build_heading_deg(t_full)
    position_clean = _integrate_position_px(
        speed_mm_s, heading_deg, fps=fps, px_per_mm=px_per_mm
    )
    position_noisy = _add_detection_noise(position_clean, t_full, rng)

    # Drop the gap region [6.0, 6.5) — frames 360..389 inclusive at 60 fps
    gap_start_s = 6.0
    gap_end_s = 6.5
    keep_mask = (t_full < gap_start_s) | (t_full >= gap_end_s)
    frames = frames_full[keep_mask]
    t = t_full[keep_mask]
    position_xy = position_noisy[keep_mask]
    heading_deg = heading_deg[keep_mask]

    regions = [
        Region("still_pre", 0.0, 1.0),
        Region("clean_bout", 1.0, 2.5),
        Region("noisy_threshold", 2.5, 4.0),
        Region("multi_peak_bout", 4.0, 6.0),
        Region("gap", gap_start_s, gap_end_s),
        Region("still_post", gap_end_s, duration_s),
    ]
    ground_truth_bouts = [(1.0, 2.5), (4.0, 6.0)]

    return SyntheticTrace(
        fps=fps,
        px_per_mm=px_per_mm,
        frames=frames,
        t=t,
        position_xy=position_xy,
        heading_deg=heading_deg,
        regions=regions,
        ground_truth_bouts=ground_truth_bouts,
    )


def smoke_check(trace: SyntheticTrace | None = None) -> dict:
    """Run the trace through the real compute_track_speed and return summaries.

    Useful as a sanity tool before building scene visuals on top of it.
    Returns a dict of region-level summary stats (max/mean speed in mm/s)
    plus the count of frames flagged as frame-gap transitions.
    """
    from fisheye.analysis.compute_speed import compute_track_speed

    if trace is None:
        trace = generate_synthetic_trace()

    speeds = compute_track_speed(
        frames=trace.frames,
        positions=trace.position_xy,
        fps=trace.fps,
        smooth_seconds=0.10,
        distance_smooth_seconds=0.05,
        hysteresis_high_px=2.0,
        hysteresis_low_px=1.0,
        hysteresis_min_frames=3,
    )

    speed_filtered_mm_s = speeds.speed_filtered.astype(np.float64) / trace.px_per_mm
    speed_smoothed_mm_s = speeds.speed_smoothed.astype(np.float64) / trace.px_per_mm

    summary: dict = {}
    for region in trace.regions:
        in_region = (trace.t >= region.start_s) & (trace.t < region.end_s)
        if not np.any(in_region):
            continue
        summary[region.label] = {
            "n_frames": int(np.sum(in_region)),
            "filtered_max_mm_s": float(np.nanmax(speed_filtered_mm_s[in_region])),
            "filtered_mean_mm_s": float(np.nanmean(speed_filtered_mm_s[in_region])),
            "smoothed_max_mm_s": float(np.nanmax(speed_smoothed_mm_s[in_region])),
        }

    from fisheye.analysis.compute_speed import TRANSITION_REASON_FRAME_GAP

    summary["frame_gap_count"] = int(
        np.sum(speeds.transition_reason_code == TRANSITION_REASON_FRAME_GAP)
    )
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(smoke_check(), indent=2))
