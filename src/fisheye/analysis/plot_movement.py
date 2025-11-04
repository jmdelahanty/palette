"""Quick plotting helpers for movement_runs outputs.

Generates speed-vs-time, heading-vs-time, and position heatmaps for a
selected track within an analysis/movement_runs entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console

from .chaser_metrics_loader import load_chaser_metrics


@dataclass
class _PersistedMetricsBundle:
    camera_frame_ids: np.ndarray
    has_offline: np.ndarray


def _extract_smoothed_series(
    run_group: zarr.Group,
    unit: str,
    indices: np.ndarray,
) -> Optional[np.ndarray]:
    if indices.size == 0:
        return None

    if unit == "mm":
        candidates = (
            "distance_to_target_smoothed_mm",
            "distance_to_target_interpolated_mm",
        )
    else:
        candidates = (
            "distance_to_target_smoothed_px",
            "distance_to_target_interpolated_px",
        )

    max_idx = int(indices.max()) if indices.size else -1
    for name in candidates:
        if name not in run_group:
            continue
        series = np.asarray(run_group[name], dtype=np.float32)
        if series.ndim == 0 or series.shape[0] <= max_idx:
            continue
        values = series[indices]
        if values.size == 0:
            continue
        return values.astype(np.float64)
    return None


def resolve_movement_runs(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
    *,
    include_online: bool,
    include_offline: bool,
) -> List[Tuple[str, zarr.Group]]:
    if "analysis" not in root or "movement_runs" not in root["analysis"]:
        raise ValueError("No movement_runs group found under analysis/.")

    parent = root["analysis/movement_runs"]
    online_parent = parent.get("online")
    offline_parent = parent.get("offline")

    def get_run(group: Optional[zarr.Group], name: str) -> Optional[zarr.Group]:
        if group is None:
            return None
        return group.get(name)

    def iter_runs() -> Iterable[Tuple[str, str, zarr.Group]]:
        if include_online and online_parent is not None:
            for name in online_parent.group_keys():
                yield ("online", name, online_parent[name])
        if include_offline and offline_parent is not None:
            for name in offline_parent.group_keys():
                yield ("offline", name, offline_parent[name])

    def resolve_requested(name: str) -> Optional[Tuple[str, str, zarr.Group]]:
        if "/" in name:
            prefix, run = name.split("/", 1)
            if prefix == "online":
                group = get_run(online_parent, run)
                if group is not None:
                    return ("online", run, group)
            elif prefix == "offline":
                group = get_run(offline_parent, run)
                if group is not None:
                    return ("offline", run, group)
            return None
        # Search both parents for bare run names
        if include_online:
            group = get_run(online_parent, name)
            if group is not None:
                return ("online", name, group)
        if include_offline:
            group = get_run(offline_parent, name)
            if group is not None:
                return ("offline", name, group)
        return None

    runs: List[Tuple[str, zarr.Group]] = []

    if requested:
        resolved = resolve_requested(requested)
        if resolved is None:
            raise ValueError(f"Movement run '{requested}' not found.")
        run_type, name, group = resolved
        console.print(f"Using movement run: [cyan]{run_type}/{name}[/cyan]")
        runs.append((f"{run_type}/{name}", group))
        return runs

    preferred: List[Tuple[str, str, zarr.Group]] = []
    if include_online and online_parent is not None:
        latest_online = online_parent.attrs.get("latest")
        if latest_online and latest_online in online_parent:
            preferred.append(("online", latest_online, online_parent[latest_online]))
    if include_offline and offline_parent is not None:
        latest_offline = offline_parent.attrs.get("latest")
        if latest_offline and latest_offline in offline_parent:
            preferred.append(("offline", latest_offline, offline_parent[latest_offline]))

    # legacy attribute storing path like 'online/run'
    legacy_latest = parent.attrs.get("latest")
    if legacy_latest and isinstance(legacy_latest, str):
        resolved = resolve_requested(legacy_latest)
        if resolved and resolved not in preferred:
            preferred.append(resolved)

    seen = set()
    for run_type, name, group in preferred:
        key = (run_type, name)
        if key in seen:
            continue
        seen.add(key)
        console.print(f"Using movement run: [cyan]{run_type}/{name}[/cyan]")
        runs.append((f"{run_type}/{name}", group))

    if not runs:
        console.print("[yellow]No movement runs resolved via 'latest'; scanning available runs.[/yellow]")
        for run_type, name, group in iter_runs():
            runs.append((f"{run_type}/{name}", group))

    return runs


def resolve_track(group: zarr.Group, track_id: Optional[int], console: Console) -> tuple[int, zarr.Group]:
    track_ids = group["track_ids"][:]
    if track_ids.size == 0:
        raise ValueError("Movement run contains no tracks.")

    if track_id is None:
        track_id = int(track_ids[0])
        console.print(f"Using track id: [cyan]{track_id}[/cyan]")
    elif track_id not in track_ids:
        raise ValueError(f"Track id {track_id} not found in run. Available IDs: {track_ids.tolist()}")

    track_group_name = f"id_{int(track_id)}"
    if track_group_name not in group["tracks"]:
        raise ValueError(f"Track group '{track_group_name}' missing under movement run.")
    return int(track_id), group["tracks"][track_group_name]


def pick_units(run_group: zarr.Group, track_group: zarr.Group) -> tuple[str, np.ndarray, np.ndarray]:
    pixel_to_mm = run_group.attrs.get("pixel_to_mm")
    positions_mm = track_group["positions_mm"][:]
    if pixel_to_mm and np.isfinite(positions_mm).any():
        return "mm", positions_mm[:, 0], positions_mm[:, 1]
    positions_px = track_group["positions_px"][:]
    return "px", positions_px[:, 0], positions_px[:, 1]


def resolve_swim_bout_spans(
    root: zarr.Group,
    requested: Optional[str],
    console: Console,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    swim_parent = root.get("analysis/swim_bout_runs")
    if swim_parent is None:
        raise ValueError("No swim_bout_runs group found under analysis/.")

    run_name = requested
    if not run_name or run_name.lower() == "latest":
        run_name = swim_parent.attrs.get("latest")
    if not run_name:
        raise ValueError("No swim_bout_runs entries available.")
    if run_name not in swim_parent:
        raise ValueError(f"Swim bout run '{run_name}' not found.")

    run_group = swim_parent[run_name]
    if "bouts" not in run_group:
        raise ValueError(f"Swim bout run '{run_name}' lacks 'bouts' dataset.")

    bout_array = run_group["bouts"][:]
    spans: List[Tuple[float, float]] = []
    if bout_array.size == 0:
        return spans, run_name

    names = bout_array.dtype.names or ()
    if "start_time_s" in names and "end_time_s" in names:
        starts = bout_array["start_time_s"]
        ends = bout_array["end_time_s"]
    elif "start_frame" in names and "end_frame" in names:
        fps: Optional[float] = None
        provenance = run_group.attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters") or {}
            if isinstance(params, dict):
                fps_val = params.get("fps")
                if fps_val is not None:
                    fps = float(fps_val)
        if not fps:
            raise ValueError(
                f"Swim bout run '{run_name}' lacks time fields and FPS information required to convert frames."
            )
        starts = bout_array["start_frame"] / float(fps)
        ends = bout_array["end_frame"] / float(fps)
    else:
        raise ValueError(f"Swim bout run '{run_name}' lacks usable start/end fields for plotting.")

    for start, stop in zip(starts, ends):
        if not (np.isfinite(start) and np.isfinite(stop)):
            continue
        if stop < start:
            continue
        spans.append((float(start), float(stop)))

    return spans, run_name


def resolve_stimulus_run(root: zarr.Group, run_group: zarr.Group) -> Optional[str]:
    inputs = run_group.attrs.get("inputs")
    if isinstance(inputs, dict):
        for key in ("stimulus_run", "source_stimulus_run"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value
    alt = run_group.attrs.get("stimulus_run")
    if isinstance(alt, str) and alt:
        return alt
    stim_parent = root.get("analysis/stimulus_runs")
    if stim_parent is not None:
        latest = stim_parent.attrs.get("latest")
        if isinstance(latest, str) and latest:
            return latest
    return None


def _resolve_metrics_run(run_group: zarr.Group) -> Optional[str]:
    direct = run_group.attrs.get("source_metrics_run")
    if isinstance(direct, str) and direct:
        return direct

    inputs = run_group.attrs.get("inputs")
    if isinstance(inputs, dict):
        for key in ("source_metrics_run", "metrics_run"):
            value = inputs.get(key)
            if isinstance(value, str) and value:
                return value

    provenance = run_group.attrs.get("provenance")
    if isinstance(provenance, dict):
        provenance_inputs = provenance.get("inputs")
        if isinstance(provenance_inputs, dict):
            for key in ("source_metrics_run", "metrics_run"):
                value = provenance_inputs.get(key)
                if isinstance(value, str) and value:
                    return value
        parameters = provenance.get("parameters")
        if isinstance(parameters, dict):
            for key in ("source_metrics_run", "metrics_run"):
                value = parameters.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _resolve_track_times(
    run_group: zarr.Group,
    track_group: zarr.Group,
    frame_indices: np.ndarray,
) -> np.ndarray:
    time_seconds = np.asarray(track_group["time_seconds"][:], dtype=np.float64)
    if time_seconds.shape[0] == frame_indices.shape[0]:
        return time_seconds

    fps = run_group.attrs.get("fps")
    if fps and np.isfinite(fps) and fps > 0:
        return frame_indices.astype(np.float64) / float(fps)

    return np.arange(frame_indices.shape[0], dtype=np.float64)


def _map_distance_to_track(
    bundle,
    run_group: zarr.Group,
    track_group: zarr.Group,
    distance_values: np.ndarray,
    availability: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = track_group["frame_indices"][:].astype(np.int64, copy=False)
    if frame_indices.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    time_seconds = _resolve_track_times(run_group, track_group, frame_indices)
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(bundle.camera_frame_ids)}

    times: List[float] = []
    distances: List[float] = []
    indices: List[int] = []
    for frame, timestamp in zip(frame_indices, time_seconds):
        idx = frame_to_idx.get(int(frame))
        if idx is None:
            continue
        if availability is not None:
            if idx >= availability.shape[0] or not availability[idx]:
                continue
        value = distance_values[idx]
        if not np.isfinite(value):
            continue
        times.append(float(timestamp))
        distances.append(float(value))
        indices.append(int(idx))

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def compute_online_chaser_distance(
    zarr_path: Path,
    stim_run: str,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
    metrics_run = _resolve_metrics_run(run_group)
    try:
        bundle = load_chaser_metrics(
            zarr_path,
            stimulus_run=stim_run,
            metrics_run=metrics_run,
            chaser_index=track_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load chaser metrics ({exc}); skipping distance plot."
        )
        return None

    online = bundle.online
    distance_mm = online.get("distance_to_target_mm")
    distance_px = online.get("distance_to_target_px")

    unit: Optional[str] = None
    distances: Optional[np.ndarray] = None
    if distance_mm is not None:
        candidate = np.asarray(distance_mm, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "mm"
            distances = candidate
    if distances is None and distance_px is not None:
        candidate = np.asarray(distance_px, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "px"
            distances = candidate

    if distances is None or unit is None:
        console.print("[yellow]Warning:[/yellow] Online distance metrics unavailable; skipping distance plot.")
        return None

    times, values, _ = _map_distance_to_track(bundle, run_group, track_group, distances)
    if times.size == 0:
        console.print("[yellow]Warning:[/yellow] No overlapping online metrics for this track.")
        return None

    return times, values, None, unit


def compute_offline_chaser_distance(
    zarr_path: Path,
    stim_run: str,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
    distance_px = run_group.get("distance_to_target_px")
    distance_mm = run_group.get("distance_to_target_mm")
    camera_frames = run_group.get("camera_frame_ids")
    availability = run_group.get("has_offline")

    if distance_px is not None or distance_mm is not None:
        if camera_frames is None or availability is None:
            console.print(
                "[yellow]Warning:[/yellow] Offline movement run is missing camera_frame_ids/has_offline; falling back to metrics load."
            )
        else:
            unit: Optional[str] = None
            distances: Optional[np.ndarray] = None
            if distance_mm is not None:
                candidate = np.asarray(distance_mm, dtype=np.float64)
                if np.isfinite(candidate).any():
                    unit = "mm"
                    distances = candidate
            if distances is None and distance_px is not None:
                candidate = np.asarray(distance_px, dtype=np.float64)
                if np.isfinite(candidate).any():
                    unit = "px"
                    distances = candidate

            if distances is not None and unit is not None:
                bundle_like = _PersistedMetricsBundle(
                    camera_frame_ids=np.asarray(camera_frames, dtype=np.int64),
                    has_offline=np.asarray(availability, dtype=bool),
                )
                times, values, indices = _map_distance_to_track(
                    bundle_like,
                    run_group,
                    track_group,
                    distances,
                    availability=bundle_like.has_offline,
                )
                if times.size == 0:
                    console.print("[yellow]Warning:[/yellow] No overlapping offline metrics for this track.")
                    return None

                smoothed = _extract_smoothed_series(run_group, unit, indices)
                console.print("[dim]Using offline metrics embedded in movement run.[/dim]")
                return times, values, smoothed, unit

    metrics_run = _resolve_metrics_run(run_group)
    try:
        bundle = load_chaser_metrics(
            zarr_path,
            stimulus_run=stim_run,
            metrics_run=metrics_run,
            chaser_index=track_id,
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load chaser metrics ({exc}); skipping distance plot."
        )
        return None

    offline = bundle.offline
    has_offline_raw = offline.get("has_offline")
    if has_offline_raw is None:
        console.print(
            "[yellow]Warning:[/yellow] Offline chaser metrics not available. Run compute_chaser_fish_metrics to enable distance plotting."
        )
        return None

    has_offline = np.asarray(has_offline_raw, dtype=bool)
    if not has_offline.any():
        console.print(
            "[yellow]Warning:[/yellow] Offline chaser metrics missing for this run. Run compute_chaser_fish_metrics to enable distance plotting."
        )
        return None

    distance_mm = offline.get("distance_mm")
    distance_px = offline.get("distance_px")

    unit: Optional[str] = None
    distances: Optional[np.ndarray] = None
    if distance_mm is not None:
        candidate = np.asarray(distance_mm, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "mm"
            distances = candidate
    if distances is None and distance_px is not None:
        candidate = np.asarray(distance_px, dtype=np.float64)
        if np.isfinite(candidate).any():
            unit = "px"
            distances = candidate

    if distances is None or unit is None:
        console.print("[yellow]Warning:[/yellow] Offline distance metrics unavailable; skipping distance plot.")
        return None

    times, values, indices = _map_distance_to_track(
        bundle,
        run_group,
        track_group,
        distances,
        availability=has_offline,
    )
    if times.size == 0:
        console.print("[yellow]Warning:[/yellow] No overlapping offline metrics for this track.")
        return None

    metrics_label = bundle.provenance.get("metrics_run")
    if isinstance(metrics_label, str) and metrics_label:
        console.print(f"[dim]Using offline metrics run: {metrics_label}[/dim]")

    smoothed = _extract_smoothed_series(run_group, unit, indices)

    return times, values, smoothed, unit

def plot_track(
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    save_path: Optional[Path],
    bins: int,
    console: Console,
    run_name: Optional[str] = None,
    swim_bouts: Optional[List[Tuple[float, float]]] = None,
    swim_bout_label: Optional[str] = None,
    distance_series: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]] = None,
) -> None:
    time_seconds = track_group["time_seconds"][:]
    smoothed_speed_px = track_group["smoothed_speed_px"][:]
    smoothed_speed_mm = track_group["smoothed_speed_mm"][:]
    instantaneous_speed_px = track_group["instantaneous_speed_px"][:]
    instantaneous_speed_mm = track_group["instantaneous_speed_mm"][:]
    smoothed_heading_deg = track_group["smoothed_heading_degrees"][:]
    smoothed_accel_px = track_group["smoothed_acceleration_px"][:]
    smoothed_accel_mm = track_group["smoothed_acceleration_mm"][:]

    unit_label, pos_x, pos_y = pick_units(run_group, track_group)
    speed_smoothed = smoothed_speed_mm if unit_label == "mm" and np.isfinite(smoothed_speed_mm).any() else smoothed_speed_px
    speed_raw = instantaneous_speed_mm if unit_label == "mm" and np.isfinite(instantaneous_speed_mm).any() else instantaneous_speed_px
    speed_label = f"Speed ({unit_label}/s)" if unit_label in {"mm", "px"} else "Speed"
    accel = (
        smoothed_accel_mm
        if unit_label == "mm" and np.isfinite(smoothed_accel_mm).any()
        else smoothed_accel_px
    )
    accel_label = f"Acceleration ({unit_label}/s^2)" if unit_label in {"mm", "px"} else "Acceleration"

    num_rows = 5 + (1 if distance_series is not None else 0)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3.2 * num_rows))
    if num_rows == 1:
        axes = [axes]
    ax_idx = 0

    # Plot both raw and smoothed speed
    speed_ax = axes[ax_idx]
    ax_idx += 1
    speed_ax.plot(time_seconds, speed_raw, color="tab:gray", linewidth=0.8, alpha=0.5, label="Instantaneous")
    speed_ax.plot(time_seconds, speed_smoothed, color="tab:blue", linewidth=1.2, label="Smoothed")
    speed_ax.set_xlabel("Time (s)")
    speed_ax.set_ylabel(speed_label)
    speed_ax.set_title(f"Track {track_id}: Speed over time")
    speed_ax.grid(alpha=0.3)
    if swim_bouts:
        label_added = False
        for start, stop in swim_bouts:
            if not np.isfinite(start) or not np.isfinite(stop):
                continue
            speed_ax.axvspan(
                start,
                stop,
                color="tab:orange",
                alpha=0.18,
                linewidth=0,
                label="Swim bout" if not label_added else None,
            )
            label_added = True
        if label_added and swim_bout_label:
            speed_ax.set_title(f"Track {track_id}: Speed over time (swim bouts: {swim_bout_label})")
    speed_ax.legend(loc="upper right")

    accel_ax = axes[ax_idx]
    ax_idx += 1
    accel_ax.plot(time_seconds, accel, color="tab:red", linewidth=1.0)
    accel_ax.set_xlabel("Time (s)")
    accel_ax.set_ylabel(accel_label)
    accel_ax.set_title("Smoothed acceleration over time")
    accel_ax.grid(alpha=0.3)

    if distance_series is not None:
        dist_time, dist_values, dist_smoothed, dist_unit = distance_series
        distance_ax = axes[ax_idx]
        ax_idx += 1
        distance_ax.plot(
            dist_time,
            dist_values,
            color="tab:purple",
            linewidth=1.0,
            alpha=0.6,
            label="Raw",
        )
        if dist_smoothed is not None and dist_smoothed.size == dist_time.size:
            distance_ax.plot(
                dist_time,
                dist_smoothed,
                color="tab:pink",
                linewidth=1.2,
                label="Smoothed",
            )
        distance_ax.set_xlabel("Time (s)")
        distance_ax.set_ylabel(f"Distance to target ({dist_unit})")
        distance_ax.set_title("Chaser → target distance")
        distance_ax.grid(alpha=0.3)
        distance_ax.set_xlim(speed_ax.get_xlim())
        if (dist_smoothed is not None and dist_smoothed.size == dist_time.size) or dist_values.size:
            distance_ax.legend(loc="upper right")

    heading_ax = axes[ax_idx]
    ax_idx += 1
    heading_ax.plot(time_seconds, smoothed_heading_deg, color="tab:orange", linewidth=1.0)
    heading_ax.set_xlabel("Time (s)")
    heading_ax.set_ylabel("Heading (deg)")
    heading_ax.set_title("Heading over time (smoothed)")
    heading_ax.grid(alpha=0.3)

    cumulative = track_group["cumulative_distance_mm"][:]
    if not (np.isfinite(cumulative).any() and unit_label == "mm"):
        cumulative = track_group["cumulative_distance_px"][:]
        cumulative_label = "Cumulative distance (px)"
    else:
        cumulative_label = "Cumulative distance (mm)"
    cumulative_ax = axes[ax_idx]
    ax_idx += 1
    cumulative_ax.plot(time_seconds, cumulative, color="tab:green", linewidth=1.2)
    cumulative_ax.set_xlabel("Time (s)")
    cumulative_ax.set_ylabel(cumulative_label)
    cumulative_ax.set_title("Cumulative distance over time")
    cumulative_ax.grid(alpha=0.3)

    # Filter out NaN positions for histogram
    valid_pos = np.isfinite(pos_x) & np.isfinite(pos_y)
    if np.any(valid_pos):
        heatmap_ax = axes[ax_idx]
        heat = heatmap_ax.hist2d(pos_x[valid_pos], pos_y[valid_pos], bins=bins, cmap="inferno")
        heatmap_ax.set_xlabel(f"X ({unit_label})")
        heatmap_ax.set_ylabel(f"Y ({unit_label})")
        heatmap_ax.set_title(f"Position density ({valid_pos.sum()}/{len(pos_x)} valid)")
        cbar = fig.colorbar(heat[3], ax=heatmap_ax)
        cbar.set_label("Counts")
    else:
        heatmap_ax = axes[ax_idx]
        heatmap_ax.text(0.5, 0.5, "No valid positions", ha="center", va="center", transform=heatmap_ax.transAxes)
        heatmap_ax.set_xlabel(f"X ({unit_label})")
        heatmap_ax.set_ylabel(f"Y ({unit_label})")
        heatmap_ax.set_title("Position density (no valid data)")

    # Build title with run type (online/offline) if available
    title = f"Movement summary – track {track_id}"
    if run_name:
        # Extract online/offline prefix if present
        if run_name.startswith("online/"):
            title = f"Movement summary (online) – track {track_id}"
        elif run_name.startswith("offline/"):
            title = f"Movement summary (offline) – track {track_id}"
        else:
            title = f"Movement summary ({run_name}) – track {track_id}"

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path)
        console.print(f"[green]Saved plot to {save_path}[/green]")
    else:
        plt.show()

    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot movement_run track metrics.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument("--movement-run", help="Movement run name (defaults to latest).")
    parser.add_argument("--track-id", type=int, help="Track ID to visualize.")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins for position heatmap (default: 200).")
    parser.add_argument("--save", help="Path to save the figure instead of showing interactively.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only plot movement runs derived from offline metrics.",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Only plot detection-based movement runs.",
    )
    parser.add_argument(
        "--swim-bout-run",
        help="Swim-bout run under analysis/swim_bout_runs to overlay (use 'latest' for the most recent run).",
    )

    args = parser.parse_args(argv)

    console = Console()
    zarr_path = Path(args.zarr_path)
    root = zarr.open(str(zarr_path), mode="r")

    include_online = not args.offline_only
    include_offline = not args.online_only
    if args.offline_only and args.online_only:
        include_online = include_offline = True
    if not include_online and not include_offline:
        include_online = include_offline = True

    runs = resolve_movement_runs(root, args.movement_run, console, include_online=include_online, include_offline=include_offline)
    if not runs:
        console.print("[yellow]No movement runs matched the requested filters.[/yellow]")
        return

    save_path = Path(args.save) if args.save else None
    swim_spans: Optional[List[Tuple[float, float]]] = None
    swim_label: Optional[str] = None
    if args.swim_bout_run is not None:
        try:
            swim_spans, swim_label = resolve_swim_bout_spans(root, args.swim_bout_run, console)
            if swim_spans:
                console.print(
                    f"[dim]Overlaying {len(swim_spans)} swim bouts from swim_bout_runs/{swim_label}.[/dim]"
                )
            else:
                console.print(
                    f"[yellow]Swim bout run '{swim_label}' contains no bouts to overlay.[/yellow]"
                )
        except ValueError as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")
            swim_spans = None
            swim_label = None

    for run_name, run_group in runs:
        console.print(f"\n[bold]Plotting movement run:[/bold] {run_name}")
        try:
            track_id, track_group = resolve_track(run_group, args.track_id, console)
        except ValueError as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")
            continue
        dest = save_path
        if dest:
            # Replace slashes in run_name to avoid invalid filenames
            safe_run_name = run_name.replace("/", "_")
            dest = dest.with_name(f"{dest.stem}_{safe_run_name}{dest.suffix}")

        distance_series = None
        stim_run_name = resolve_stimulus_run(root, run_group)
        if stim_run_name:
            if run_name.startswith("online/"):
                distance_series = compute_online_chaser_distance(
                    zarr_path,
                    stim_run_name,
                    run_group,
                    track_group,
                    track_id,
                    console,
                )
            elif run_name.startswith("offline/"):
                distance_series = compute_offline_chaser_distance(
                    zarr_path,
                    stim_run_name,
                    run_group,
                    track_group,
                    track_id,
                    console,
                )
        else:
            console.print(
                "[yellow]Warning:[/yellow] Unable to resolve stimulus run for distance overlay."
            )

        plot_track(
            run_group,
            track_group,
            track_id,
            dest,
            args.bins,
            console,
            run_name,
            swim_bouts=swim_spans,
            swim_bout_label=swim_label,
            distance_series=distance_series,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
