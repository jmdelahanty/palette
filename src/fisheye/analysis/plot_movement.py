"""Quick plotting helpers for movement_runs outputs.

Generates speed-vs-time, heading-vs-time, and position heatmaps for a
selected track within an analysis/movement_runs entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console

from .chaser_state_interpolator import load_structured_dataset


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


def _parse_coordinate_transform(attr: Any) -> Dict[str, Any]:
    if isinstance(attr, dict):
        return attr
    if isinstance(attr, (bytes, bytearray)):
        try:
            attr = attr.decode("utf-8")
        except Exception:
            return {}
    if isinstance(attr, str):
        try:
            return json.loads(attr)
        except Exception:
            return {}
    return {}


def compute_texture_to_camera_transform(
    root: zarr.Group, stim_run: str
) -> Tuple[float, float, float]:
    stim_group = root[f"analysis/stimulus_runs/{stim_run}"]
    coord_attr = stim_group.attrs.get("coordinate_transform")
    coord_info = _parse_coordinate_transform(coord_attr)

    texture_dims = coord_info.get("texture_dimensions")
    camera_dims = coord_info.get("camera_dimensions")
    scale = coord_info.get("texture_to_camera_scale")
    if scale is None:
        try:
            if isinstance(camera_dims, (list, tuple)) and isinstance(texture_dims, (list, tuple)):
                if camera_dims and texture_dims and texture_dims[0]:
                    scale = float(camera_dims[0]) / float(texture_dims[0])
        except Exception:
            scale = None
    if scale is None:
        scale = 1.0

    offset_x = offset_y = 0.0
    calib = root.get("calibration")
    if calib is not None:
        offset_x = float(calib.attrs.get("stimulus_offset_x", 0.0) or 0.0)
        offset_y = float(calib.attrs.get("stimulus_offset_y", 0.0) or 0.0)
        primary_cam = calib.attrs.get("primary_camera_id")
        if isinstance(primary_cam, (bytes, bytearray)):
            primary_cam = primary_cam.decode("utf-8", "ignore")
        if (
            isinstance(primary_cam, str)
            and "cameras" in calib
            and primary_cam in calib["cameras"]
        ):
            cam_attrs = calib["cameras"][primary_cam].attrs
            offset_x = float(cam_attrs.get("stimulus_offset_x", offset_x) or offset_x)
            offset_y = float(cam_attrs.get("stimulus_offset_y", offset_y) or offset_y)
    return float(scale), float(offset_x), float(offset_y)

def load_chaser_states(
    root: zarr.Group,
    stim_run: str,
    console: Console,
) -> Optional[np.ndarray]:
    path = f"analysis/stimulus_runs/{stim_run}/tracking_data"
    if path not in root or "chaser_states" not in root[path]:
        console.print(
            f"[yellow]Warning:[/yellow] Stimulus run '{stim_run}' lacks tracking_data/chaser_states."
        )
        return None
    try:
        chaser_arr, _ = load_structured_dataset(root[path], "chaser_states")
        return chaser_arr
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load stimulus chaser states ({exc})."
        )
        return None


def compute_online_chaser_distance(
    root: zarr.Group,
    stim_run: str,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    chaser_arr = load_chaser_states(root, stim_run, console)
    if chaser_arr is None or chaser_arr.size == 0:
        return None

    names = chaser_arr.dtype.names or ()
    if "distance_to_target_mm" in names:
        distances = np.asarray(chaser_arr["distance_to_target_mm"], dtype=np.float64)
        unit = "mm"
    elif "distance_to_target_px" in names:
        distances = np.asarray(chaser_arr["distance_to_target_px"], dtype=np.float64)
        unit = "px"
    else:
        console.print(
            "[yellow]Warning:[/yellow] Stimulus chaser_states lacks distance columns; skipping distance plot."
        )
        return None

    if "timestamp_ns_session" in names:
        times = np.asarray(chaser_arr["timestamp_ns_session"], dtype=np.float64) / 1e9
    elif "stimulus_frame_num" in names:
        stim_group = root[f"analysis/stimulus_runs/{stim_run}"]
        fps = stim_group.attrs.get("fps")
        if fps and np.isfinite(fps) and fps > 0:
            times = np.asarray(chaser_arr["stimulus_frame_num"], dtype=np.float64) / float(fps)
        else:
            times = np.arange(distances.shape[0], dtype=np.float64)
    else:
        times = np.arange(distances.shape[0], dtype=np.float64)

    if times.size:
        times = times - times[0]

    valid = np.isfinite(times) & np.isfinite(distances)
    if not np.any(valid):
        return None

    return times[valid], distances[valid], unit


def compute_offline_chaser_distance(
    root: zarr.Group,
    run_group: zarr.Group,
    track_group: zarr.Group,
    track_id: int,
    console: Console,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    stim_run = resolve_stimulus_run(root, run_group)
    if not stim_run:
        console.print(
            "[yellow]Warning:[/yellow] Unable to resolve stimulus run for offline distance plot."
        )
        return None

    chaser_arr = load_chaser_states(root, stim_run, console)
    if chaser_arr is None or chaser_arr.size == 0:
        return None

    names = chaser_arr.dtype.names or ()
    if "stimulus_frame_num" not in names:
        console.print(
            "[yellow]Warning:[/yellow] Stimulus chaser_states lacks 'stimulus_frame_num'; skipping distance plot."
        )
        return None

    mask = np.ones(chaser_arr.shape[0], dtype=bool)
    if "chaser_index" in names:
        chaser_indices = chaser_arr["chaser_index"]
        if track_id in np.unique(chaser_indices):
            mask = chaser_indices == track_id
        else:
            mask = chaser_indices == chaser_indices[0]
    chaser_filtered = chaser_arr[mask]
    if chaser_filtered.size == 0:
        console.print(
            "[yellow]Warning:[/yellow] No chaser states matched track id; skipping distance plot."
        )
        return None

    try:
        scale, offset_x, offset_y = compute_texture_to_camera_transform(root, stim_run)
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Unable to derive texture→camera transform ({exc}); skipping distance plot."
        )
        return None

    stim_group = root[f"analysis/stimulus_runs/{stim_run}"]
    try:
        frame_meta, _ = load_structured_dataset(stim_group["video_metadata"], "frame_metadata")
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] Unable to load frame_metadata for {stim_run} ({exc}); skipping distance plot."
        )
        return None

    meta_names = frame_meta.dtype.names or ()
    if "triggering_camera_frame_id" not in meta_names or "stimulus_frame_num" not in meta_names:
        console.print(
            f"[yellow]Warning:[/yellow] frame_metadata lacks required fields; skipping distance plot."
        )
        return None

    cam_frames = frame_meta["triggering_camera_frame_id"].astype(np.int64, copy=False)
    stim_frames = frame_meta["stimulus_frame_num"].astype(np.int64, copy=False)
    cam_to_stim: Dict[int, int] = {int(cam): int(stim) for cam, stim in zip(cam_frames, stim_frames)}

    stim_frames_chaser = chaser_filtered["stimulus_frame_num"].astype(np.int64, copy=False)
    chaser_x_tex = np.asarray(chaser_filtered["chaser_pos_x"], dtype=np.float64)
    chaser_y_tex = np.asarray(chaser_filtered["chaser_pos_y"], dtype=np.float64)
    chaser_cam_x = chaser_x_tex * scale + offset_x
    chaser_cam_y = chaser_y_tex * scale + offset_y

    stim_to_pos: Dict[int, Tuple[float, float]] = {
        int(stim): (float(x_cam), float(y_cam))
        for stim, x_cam, y_cam in zip(stim_frames_chaser, chaser_cam_x, chaser_cam_y)
    }

    frame_indices = track_group["frame_indices"][:].astype(np.int64, copy=False)
    positions_px = track_group["positions_px"][:]
    if positions_px.shape[0] != frame_indices.shape[0]:
        console.print(
            "[yellow]Warning:[/yellow] Position array length mismatch; skipping distance plot."
        )
        return None

    chaser_camera = np.full_like(positions_px, np.nan, dtype=np.float64)
    for idx, cam_frame in enumerate(frame_indices):
        stim_frame = cam_to_stim.get(int(cam_frame))
        if stim_frame is None:
            continue
        pos = stim_to_pos.get(stim_frame)
        if pos is None:
            continue
        chaser_camera[idx, 0] = pos[0]
        chaser_camera[idx, 1] = pos[1]

    valid = np.isfinite(chaser_camera).all(axis=1) & np.isfinite(positions_px).all(axis=1)
    if not np.any(valid):
        console.print(
            "[yellow]Warning:[/yellow] No overlapping frames between detections and chaser states; skipping distance plot."
        )
        return None

    delta = positions_px[valid] - chaser_camera[valid]
    distance_px = np.hypot(delta[:, 0], delta[:, 1])

    time_seconds = track_group["time_seconds"][:]
    if time_seconds.shape[0] != frame_indices.shape[0]:
        fps = run_group.attrs.get("fps")
        if fps and np.isfinite(fps) and fps > 0:
            time_seconds = frame_indices.astype(np.float64) / float(fps)
        else:
            time_seconds = np.arange(frame_indices.shape[0], dtype=np.float64)

    time_valid = time_seconds[valid]

    pixel_to_mm = run_group.attrs.get("pixel_to_mm")
    if pixel_to_mm and np.isfinite(pixel_to_mm):
        distance_vals = distance_px * float(pixel_to_mm)
        distance_unit = "mm"
    else:
        distance_vals = distance_px
        distance_unit = "px"

    return time_valid, distance_vals, distance_unit

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
    distance_series: Optional[Tuple[np.ndarray, np.ndarray, str]] = None,
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
        dist_time, dist_values, dist_unit = distance_series
        distance_ax = axes[ax_idx]
        ax_idx += 1
        distance_ax.plot(dist_time, dist_values, color="tab:purple", linewidth=1.0)
        distance_ax.set_xlabel("Time (s)")
        distance_ax.set_ylabel(f"Distance to target ({dist_unit})")
        distance_ax.set_title("Chaser → target distance")
        distance_ax.grid(alpha=0.3)
        distance_ax.set_xlim(speed_ax.get_xlim())

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
    root = zarr.open(args.zarr_path, mode="r")

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
        if run_name.startswith('online/') and stim_run_name:
            distance_series = compute_online_chaser_distance(root, stim_run_name, console)
        elif run_name.startswith('offline/') and stim_run_name:
            distance_series = compute_offline_chaser_distance(root, run_group, track_group, track_id, console)

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