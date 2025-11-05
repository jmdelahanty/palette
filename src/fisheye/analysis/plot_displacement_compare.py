#!/usr/bin/env python3
"""
Plot raw vs smoothed per-frame displacement (pixels & millimeters), plus the
corresponding instantaneous speeds (raw vs displacement-smoothed and moving
average) for a selected track.

Usage:
    python -m fisheye.analysis.plot_displacement_compare <archive.zarr> [--speed-run RUN] [--track-id ID]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import zarr
from numpy.typing import ArrayLike


def _coerce_scalar(value: ArrayLike) -> Optional[float]:
    """Convert assorted scalar-like values (including numpy types) to float."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (bytes, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.asarray(value).astype(np.float64).ravel()[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _coerce_scalar(value[0])
    return None


def _extract_mm_factor_from_attrs(attrs) -> Optional[float]:
    """Return millimeters-per-pixel factor from attribute container if present."""
    for key in ("pixel_to_mm", "mm_per_pixel", "millimeters_per_pixel"):
        if key in attrs:
            val = _coerce_scalar(attrs[key])
            if val and val > 0:
                return float(val)
    for key in ("pixels_per_mm_camera", "pixels_per_mm", "pixels_per_millimeter"):
        if key in attrs:
            val = _coerce_scalar(attrs[key])
            if val and val > 0:
                return 1.0 / float(val)
    return None


def _search_mm_factor(group: Optional[zarr.Group], visited: Optional[set[int]] = None) -> Optional[float]:
    if group is None or not isinstance(group, zarr.Group):
        return None
    if visited is None:
        visited = set()
    ident = id(group)
    if ident in visited:
        return None
    visited.add(ident)

    mm_factor = _extract_mm_factor_from_attrs(group.attrs)
    if mm_factor:
        return mm_factor

    # Prefer deterministic order
    for name in getattr(group, "group_keys", lambda: [])():
        sub = group[name]
        if isinstance(sub, zarr.Group):
            mm_factor = _search_mm_factor(sub, visited)
            if mm_factor:
                return mm_factor
    return None


def _infer_pixel_to_mm(
    root: zarr.Group,
    run_group: zarr.Group,
    track_group: zarr.Group,
    *,
    pixels_per_mm_override: Optional[float],
    pixel_to_mm_override: Optional[float],
) -> Optional[float]:
    """Determine mm-per-pixel conversion factor."""
    if pixel_to_mm_override and pixel_to_mm_override > 0:
        return float(pixel_to_mm_override)
    if pixels_per_mm_override and pixels_per_mm_override > 0:
        return 1.0 / float(pixels_per_mm_override)

    for attrs in (track_group.attrs, run_group.attrs, root.attrs):
        mm_factor = _extract_mm_factor_from_attrs(attrs)
        if mm_factor:
            return mm_factor

    # Search dedicated calibration groups
    analysis_group = root.get("analysis")
    for candidate in (
        root.get("calibration"),
        analysis_group.get("calibration") if isinstance(analysis_group, zarr.Group) else None,
    ):
        mm_factor = _search_mm_factor(candidate)
        if mm_factor:
            return mm_factor

    if isinstance(analysis_group, zarr.Group):
        stim_root = analysis_group.get("stimulus_runs")
        if isinstance(stim_root, zarr.Group):
            # Try latest run first
            preferred = []
            latest = stim_root.attrs.get("latest")
            if latest and latest in stim_root:
                preferred.append(latest)
            preferred.extend(sorted(name for name in stim_root.group_keys() if name not in preferred))
            for run_name in preferred:
                mm_factor = _search_mm_factor(stim_root[run_name].get("calibration"))
                if mm_factor:
                    return mm_factor

    return None


def _resolve_speed_run(root: zarr.Group, requested: Optional[str]) -> Tuple[zarr.Group, str]:
    analysis = root.get("analysis")
    if analysis is None or "speed_runs" not in analysis:
        raise KeyError("analysis/speed_runs group not found. Compute speeds first via fisheye.analysis.compute_speed.")

    parent = analysis["speed_runs"]
    run_name = requested or parent.attrs.get("latest")
    if run_name and run_name in parent:
        return parent[run_name], str(run_name)

    if requested:
        raise KeyError(f"Speed run '{requested}' not found under analysis/speed_runs.")

    run_names = sorted(parent.group_keys())
    if not run_names:
        raise KeyError("analysis/speed_runs exists but contains no runs.")
    return parent[run_names[0]], run_names[0]


def _resolve_track(run_group: zarr.Group, requested: Optional[int]) -> Tuple[zarr.Group, int]:
    track_ids = run_group.get("track_ids")
    if track_ids is None:
        raise KeyError("speed run missing track_ids array.")
    track_ids = np.asarray(track_ids[:], dtype=int)
    if track_ids.size == 0:
        raise ValueError("Speed run contains no tracks.")

    if requested is not None:
        if requested not in track_ids:
            raise KeyError(f"Track {requested} not present in run (available: {track_ids.tolist()}).")
        track_id = requested
    else:
        track_id = int(track_ids[0])

    group_path = f"tracks/id_{track_id}"
    if group_path not in run_group:
        raise KeyError(f"Track group '{group_path}' missing in speed run.")
    return run_group[group_path], track_id


def plot_displacement_and_speed(
    frame_indices: np.ndarray,
    raw_displacement_px: np.ndarray,
    smoothed_displacement_px: np.ndarray,
    speed_raw_px: np.ndarray,
    speed_filtered_px: Optional[np.ndarray],
    speed_smoothed_px: Optional[np.ndarray],
    raw_displacement_mm: Optional[np.ndarray],
    smoothed_displacement_mm: Optional[np.ndarray],
    speed_raw_mm: Optional[np.ndarray],
    speed_filtered_mm: Optional[np.ndarray],
    speed_smoothed_mm: Optional[np.ndarray],
    *,
    title: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
    save_path: Optional[Path],
) -> None:
    if start_frame is not None or end_frame is not None:
        mask = np.ones(frame_indices.shape, dtype=bool)
        if start_frame is not None:
            mask &= frame_indices >= start_frame
        if end_frame is not None:
            mask &= frame_indices <= end_frame
        frame_indices = frame_indices[mask]
        raw_displacement_px = raw_displacement_px[mask]
        smoothed_displacement_px = smoothed_displacement_px[mask]
        speed_raw_px = speed_raw_px[mask]
        if speed_filtered_px is not None:
            speed_filtered_px = speed_filtered_px[mask]
        if speed_smoothed_px is not None:
            speed_smoothed_px = speed_smoothed_px[mask]

        def _mask_optional(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if array is None:
                return None
            return array[mask]

        raw_displacement_mm = _mask_optional(raw_displacement_mm)
        smoothed_displacement_mm = _mask_optional(smoothed_displacement_mm)
        speed_raw_mm = _mask_optional(speed_raw_mm)
        speed_filtered_mm = _mask_optional(speed_filtered_mm)
        speed_smoothed_mm = _mask_optional(speed_smoothed_mm)

    has_mm = (
        raw_displacement_mm is not None
        and smoothed_displacement_mm is not None
        and speed_raw_mm is not None
    )

    if has_mm:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="row")
        ax_disp_px, ax_disp_mm = axes[0]
        ax_speed_px, ax_speed_mm = axes[1]
    else:
        fig, (ax_disp_px, ax_speed_px) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax_disp_mm = ax_speed_mm = None

    ax_disp_px.plot(frame_indices, raw_displacement_px, label="Raw displacement", color="tab:orange", alpha=0.7)
    ax_disp_px.plot(
        frame_indices,
        smoothed_displacement_px,
        label="Smoothed displacement",
        color="tab:blue",
        linewidth=1.3,
    )
    ax_disp_px.set_ylabel("Displacement (px)")
    if has_mm:
        ax_disp_px.set_title(f"{title}\nPixel space")
    else:
        ax_disp_px.set_title(title)
    ax_disp_px.legend(loc="upper right")
    ax_disp_px.grid(True, alpha=0.25)

    ax_speed_px.plot(
        frame_indices,
        speed_raw_px,
        label="Instantaneous speed (raw)",
        color="tab:red",
        alpha=0.6,
    )
    if speed_filtered_px is not None:
        ax_speed_px.plot(
            frame_indices,
            speed_filtered_px,
            label="Instantaneous speed (displacement-smoothed)",
            color="tab:green",
            linewidth=1.1,
        )
    if speed_smoothed_px is not None:
        ax_speed_px.plot(
            frame_indices,
            speed_smoothed_px,
            label="Moving-average speed",
            color="tab:purple",
            linewidth=1.2,
        )
    ax_speed_px.set_xlabel("Camera frame")
    ax_speed_px.set_ylabel("Speed (px/s)")
    ax_speed_px.legend(loc="upper right")
    ax_speed_px.grid(True, alpha=0.25)

    if has_mm and ax_disp_mm is not None and ax_speed_mm is not None:
        ax_disp_mm.plot(
            frame_indices,
            raw_displacement_mm,
            label="Raw displacement",
            color="tab:orange",
            alpha=0.7,
        )
        ax_disp_mm.plot(
            frame_indices,
            smoothed_displacement_mm,
            label="Smoothed displacement",
            color="tab:blue",
            linewidth=1.3,
        )
        ax_disp_mm.set_ylabel("Displacement (mm)")
        ax_disp_mm.set_title("Millimeter space")
        ax_disp_mm.legend(loc="upper right")
        ax_disp_mm.grid(True, alpha=0.25)

        ax_speed_mm.plot(
            frame_indices,
            speed_raw_mm,
            label="Instantaneous speed (raw)",
            color="tab:red",
            alpha=0.6,
        )
        if speed_filtered_mm is not None:
            ax_speed_mm.plot(
                frame_indices,
                speed_filtered_mm,
                label="Instantaneous speed (displacement-smoothed)",
                color="tab:green",
                linewidth=1.1,
            )
        if speed_smoothed_mm is not None:
            ax_speed_mm.plot(
                frame_indices,
                speed_smoothed_mm,
                label="Moving-average speed",
                color="tab:purple",
                linewidth=1.2,
            )
        ax_speed_mm.set_xlabel("Camera frame")
        ax_speed_mm.set_ylabel("Speed (mm/s)")
        ax_speed_mm.legend(loc="upper right")
        ax_speed_mm.grid(True, alpha=0.25)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-frame displacement before and after smoothing.")
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--speed-run", help="Speed run name (defaults to latest).")
    parser.add_argument("--track-id", type=int, help="Track ID to plot (defaults to first in run).")
    parser.add_argument("--start-frame", type=int, help="Optional starting camera frame.")
    parser.add_argument("--end-frame", type=int, help="Optional ending camera frame.")
    parser.add_argument("--save", type=Path, help="Optional path to save the figure instead of showing.")
    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        help="Override pixels-per-millimeter calibration (camera space).",
    )
    parser.add_argument(
        "--pixel-to-mm",
        type=float,
        help="Override millimeters-per-pixel calibration (camera space).",
    )
    args = parser.parse_args()
    if args.pixels_per_mm and args.pixel_to_mm:
        parser.error("Specify only one of --pixels-per-mm or --pixel-to-mm.")

    archive = args.zarr_path.expanduser().resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Zarr archive not found: {archive}")

    root = zarr.open(archive, mode="r")
    run_group, run_name = _resolve_speed_run(root, args.speed_run)
    track_group, track_id = _resolve_track(run_group, args.track_id)

    frame_indices = np.asarray(track_group["frame_indices"][:], dtype=np.int64)
    smoothed = np.asarray(track_group["distance_per_frame"][:], dtype=np.float64)
    if "distance_per_frame_raw" in track_group:
        raw = np.asarray(track_group["distance_per_frame_raw"][:], dtype=np.float64)
    else:
        print("Warning: distance_per_frame_raw missing; falling back to smoothed values.")
        raw = smoothed.copy()

    smoothed_mm = (
        np.asarray(track_group["distance_per_frame_mm"][:], dtype=np.float64)
        if "distance_per_frame_mm" in track_group
        else None
    )
    raw_mm = (
        np.asarray(track_group["distance_per_frame_raw_mm"][:], dtype=np.float64)
        if "distance_per_frame_raw_mm" in track_group
        else None
    )
    if smoothed_mm is not None and raw_mm is None:
        raw_mm = smoothed_mm.copy()

    speed_raw = np.asarray(track_group["instantaneous_speed"][:], dtype=np.float64)
    speed_filtered = (
        np.asarray(track_group["instantaneous_speed_filtered"][:], dtype=np.float64)
        if "instantaneous_speed_filtered" in track_group
        else None
    )
    speed_smoothed = (
        np.asarray(track_group["smoothed_speed"][:], dtype=np.float64)
        if "smoothed_speed" in track_group
        else None
    )
    speed_raw_mm = (
        np.asarray(track_group["instantaneous_speed_mm"][:], dtype=np.float64)
        if "instantaneous_speed_mm" in track_group
        else None
    )
    speed_filtered_mm = (
        np.asarray(track_group["instantaneous_speed_filtered_mm"][:], dtype=np.float64)
        if "instantaneous_speed_filtered_mm" in track_group
        else None
    )
    speed_smoothed_mm = (
        np.asarray(track_group["smoothed_speed_mm"][:], dtype=np.float64)
        if "smoothed_speed_mm" in track_group
        else None
    )

    if smoothed_mm is None or raw_mm is None or speed_raw_mm is None:
        mm_factor = _infer_pixel_to_mm(
            root,
            run_group,
            track_group,
            pixels_per_mm_override=args.pixels_per_mm,
            pixel_to_mm_override=args.pixel_to_mm,
        )
        if mm_factor and mm_factor > 0:
            smoothed_mm = smoothed * mm_factor
            raw_mm = raw * mm_factor
            speed_raw_mm = speed_raw * mm_factor
            speed_filtered_mm = speed_filtered * mm_factor if speed_filtered is not None else None
            speed_smoothed_mm = speed_smoothed * mm_factor if speed_smoothed is not None else None
        else:
            if args.pixels_per_mm or args.pixel_to_mm:
                print("Warning: provided calibration values were invalid; skipping mm plots.")
            elif args.pixels_per_mm is None and args.pixel_to_mm is None:
                print("Info: Could not determine pixel-to-mm calibration; showing pixel plots only.")
            smoothed_mm = raw_mm = speed_raw_mm = speed_filtered_mm = speed_smoothed_mm = None

    title = f"Displacement comparison — run {run_name}, track {track_id}"
    plot_displacement_and_speed(
        frame_indices,
        raw,
        smoothed,
        speed_raw,
        speed_filtered,
        speed_smoothed,
        raw_mm,
        smoothed_mm,
        speed_raw_mm,
        speed_filtered_mm,
        speed_smoothed_mm,
        title=title,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
