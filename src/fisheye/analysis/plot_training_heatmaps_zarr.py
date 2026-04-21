#!/usr/bin/env python3
"""
Plot pre/training/post spatial heatmaps using data stored inside a Palette Zarr archive.

This script assumes stimulus metadata has been imported via
``fisheye.analysis.import_stimulus_to_zarr`` and that a track kinematics run exists under
``analysis/track_kinematics_runs``. It mirrors the functionality of
``chaser_analysis/training_heatmap_analyzer.py`` but reads exclusively from the
Zarr store (no separate analysis H5 required).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import zarr
from rich.console import Console
from scipy.ndimage import gaussian_filter

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics
from fisheye.shared.citrus_enums import EXPERIMENT_EVENT_TYPE, EVENT_NAME_TO_ID


@dataclass
class TrainingPeriods:
    pre_start: int
    pre_end: int
    train_start: int
    train_end: int
    post_start: int
    post_end: int
    pre_start_ns: Optional[int] = None
    pre_end_ns: Optional[int] = None
    train_start_ns: Optional[int] = None
    train_end_ns: Optional[int] = None
    post_start_ns: Optional[int] = None
    post_end_ns: Optional[int] = None


def _to_python(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _load_structured_group(node: Union[zarr.Group, zarr.Array]) -> Dict[str, np.ndarray]:
    if hasattr(node, "array_keys"):
        field_names = node.attrs.get("field_names")
        if not field_names:
            field_names = list(node.array_keys())
        return {name: node[name][:] for name in field_names}

    array = node[:]
    if array.dtype.names:
        return {name: array[name] for name in array.dtype.names}
    raise ValueError("Expected a structured array with named fields for events dataset")


def _load_events(stim_run: zarr.Group) -> List[Dict[str, object]]:
    events_group = stim_run.get("events")
    if events_group is None:
        raise ValueError("Stimulus run missing 'events' group.")
    arrays = _load_structured_group(events_group)
    field_names = list(arrays.keys())
    total = len(arrays[field_names[0]])
    events: List[Dict[str, object]] = []
    for idx in range(total):
        record = {}
        for name in field_names:
            value = arrays[name][idx]
            value = _to_python(value)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore").rstrip("\x00")
            record[name] = value
        # Normalize event type fields
        if "event_type_id" in record:
            record["event_type_id"] = int(record["event_type_id"])
            record["event_type"] = EXPERIMENT_EVENT_TYPE.get(record["event_type_id"], "UNKNOWN")
        events.append(record)
    return events


def _load_frame_metadata(stim_run: zarr.Group) -> Dict[str, np.ndarray]:
    vm_group = stim_run.get("video_metadata")
    if vm_group is None or "frame_metadata" not in vm_group:
        raise ValueError("Stimulus run missing video_metadata/frame_metadata.")
    fm_group = vm_group["frame_metadata"]
    data = _load_structured_group(fm_group)
    return data


def _build_stim_to_camera_map(frame_metadata: Dict[str, np.ndarray]) -> Dict[int, int]:
    stim_frames = frame_metadata.get("stimulus_frame_num")
    camera_frames = frame_metadata.get("triggering_camera_frame_id")
    if stim_frames is None or camera_frames is None:
        return {}
    stim_to_camera: Dict[int, int] = {}
    for stim, cam in zip(stim_frames, camera_frames):
        stim_i = int(stim)
        cam_i = int(cam)
        stim_to_camera.setdefault(stim_i, cam_i)
    return stim_to_camera


def _resolve_event_camera_frame(event: Dict[str, object], stim_to_camera: Dict[int, int]) -> Optional[int]:
    camera_frame = event.get("camera_frame_id")
    if camera_frame is not None:
        camera_frame = int(camera_frame)
        if camera_frame >= 0:
            return camera_frame
    stim_frame = event.get("stimulus_frame_num") or event.get("stimulus_frame")
    if stim_frame is None:
        return None
    return stim_to_camera.get(int(stim_frame))


def _determine_periods(
    events: List[Dict[str, object]],
    stim_to_camera: Dict[int, int],
    camera_frames_all: np.ndarray,
    console: Console,
) -> TrainingPeriods:
    min_cam = int(np.min(camera_frames_all))
    max_cam = int(np.max(camera_frames_all))
    total_frames = max_cam - min_cam + 1

    frame_markers: Dict[int, int] = {}
    time_markers: Dict[int, int] = {}
    for event in events:
        etype = int(event.get("event_type_id", -1))
        cam_frame = _resolve_event_camera_frame(event, stim_to_camera)
        if cam_frame is not None and etype not in frame_markers:
            frame_markers[etype] = cam_frame
        ts_ns = event.get("timestamp_ns_session")
        if ts_ns is not None and etype not in time_markers:
            time_markers[etype] = int(ts_ns)

    def frame_or_default(event_id: int, default: Optional[int] = None) -> Optional[int]:
        return frame_markers.get(event_id, default)

    def time_or_default(event_id: int, default: Optional[int] = None) -> Optional[int]:
        return time_markers.get(event_id, default)

    protocol_start_frame = frame_or_default(EVENT_NAME_TO_ID["CHASER_PRE_PERIOD_START"])
    if protocol_start_frame is None:
        protocol_start_frame = frame_or_default(EVENT_NAME_TO_ID["PROTOCOL_START"], min_cam)
    training_start_frame = frame_or_default(EVENT_NAME_TO_ID["CHASER_TRAINING_START"])
    post_start_frame = frame_or_default(EVENT_NAME_TO_ID["CHASER_POST_PERIOD_START"])
    protocol_finish_frame = frame_or_default(EVENT_NAME_TO_ID["PROTOCOL_FINISH"], max_cam)

    protocol_start_ns = time_or_default(EVENT_NAME_TO_ID["PROTOCOL_START"])
    training_start_ns = time_or_default(EVENT_NAME_TO_ID["CHASER_TRAINING_START"])
    post_start_ns = time_or_default(EVENT_NAME_TO_ID["CHASER_POST_PERIOD_START"])
    protocol_finish_ns = time_or_default(EVENT_NAME_TO_ID["PROTOCOL_FINISH"])

    # Fallback to proportional mapping using timestamps if frames are missing.
    def proportional_frame(target_ns: Optional[int]) -> Optional[int]:
        if target_ns is None or protocol_start_ns is None or protocol_finish_ns is None:
            return None
        total_duration = protocol_finish_ns - protocol_start_ns
        if total_duration <= 0:
            return None
        fraction = (target_ns - protocol_start_ns) / total_duration
        fraction = max(0.0, min(1.0, fraction))
        return int(round(min_cam + fraction * (total_frames - 1)))

    if training_start_frame is None:
        training_start_frame = proportional_frame(training_start_ns)
    if post_start_frame is None:
        post_start_frame = proportional_frame(post_start_ns)
    if protocol_finish_frame is None:
        protocol_finish_frame = proportional_frame(protocol_finish_ns) or max_cam

    if training_start_frame is None:
        raise ValueError("Training start event not found; cannot partition periods.")
    if post_start_frame is None:
        console.log("[yellow]Post-training start missing; treating end of protocol as post period start.[/yellow]")
        post_start_frame = protocol_finish_frame

    pre_start = protocol_start_frame or min_cam
    pre_end = max(training_start_frame - 1, pre_start)
    train_start = training_start_frame
    train_end = max(post_start_frame - 1, train_start)
    post_start = post_start_frame
    post_end = max(protocol_finish_frame, post_start)

    console.log(
        "[cyan]Training periods (camera frames):[/cyan] "
        f"pre {pre_start}–{pre_end}, training {train_start}–{train_end}, post {post_start}–{post_end}"
    )

    return TrainingPeriods(
        pre_start=pre_start,
        pre_end=pre_end,
        train_start=train_start,
        train_end=train_end,
        post_start=post_start,
        post_end=post_end,
        pre_start_ns=protocol_start_ns,
        pre_end_ns=training_start_ns,
        train_start_ns=training_start_ns,
        train_end_ns=post_start_ns,
        post_start_ns=post_start_ns,
        post_end_ns=protocol_finish_ns,
    )


def _collect_positions(movement_run: zarr.Group) -> Tuple[np.ndarray, np.ndarray]:
    tracks_group = movement_run["tracks"]
    track_ids = movement_run["track_ids"][:]
    frames_list: List[np.ndarray] = []
    positions_list: List[np.ndarray] = []
    for track_id in track_ids:
        track = tracks_group[f"id_{int(track_id)}"]
        frames = track["frame_indices"][:]
        positions = track["positions_px"][:]
        frames_list.append(frames.astype(np.int64, copy=False))
        positions_list.append(positions.astype(np.float64, copy=False))
    if not frames_list:
        raise ValueError("Track kinematics run contains no track data.")
    frames = np.concatenate(frames_list)
    positions = np.concatenate(positions_list)
    order = np.argsort(frames)
    frames = frames[order]
    positions = positions[order]
    return frames, positions


def _compute_period_heatmaps(
    *,
    frames: np.ndarray,
    positions: np.ndarray,
    period_specs: Sequence[Tuple[str, int, int, Optional[int], Optional[int]]],
    width: int,
    height: int,
    bin_size: int,
    smooth_sigma: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int, float]]]:
    heatmaps: Dict[str, np.ndarray] = {}
    coverage_info: Dict[str, Tuple[int, int, float]] = {}

    for label, start, end, _, _ in period_specs:
        mask = (frames >= start) & (frames <= end)
        frames_period = frames[mask]
        positions_period = positions[mask]

        if positions_period.ndim != 2 or positions_period.shape[1] != 2:
            positions_period = positions_period.reshape((-1, 2)) if positions_period.size else positions_period

        valid = np.isfinite(positions_period).all(axis=1) if positions_period.size else np.array([], dtype=bool)
        positions_valid = positions_period[valid]
        frames_valid = frames_period[valid]

        heatmaps[label] = _create_heatmap(positions_valid, width, height, bin_size, smooth_sigma)

        unique_frames = np.unique(frames_valid)
        total_span = max(end - start + 1, 1)
        coverage_ratio = unique_frames.size / total_span if total_span else 0.0
        coverage_info[label] = (positions_valid.shape[0], unique_frames.size, coverage_ratio)

    return heatmaps, coverage_info


def _render_heatmap_grid(
    *,
    heatmaps: Dict[str, np.ndarray],
    coverage_info: Dict[str, Tuple[int, int, float]],
    period_specs: Sequence[Tuple[str, int, int, Optional[int], Optional[int]]],
    width: int,
    height: int,
    title: str,
) -> plt.Figure:
    labels = [label for label, _, _, _, _ in period_specs if label in heatmaps]
    if not labels:
        raise ValueError("No periods available for plotting heatmaps.")

    n_periods = len(labels)
    fig, axes = plt.subplots(2, n_periods, figsize=(6 * n_periods, 10))
    if n_periods == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    cmap_main = "inferno"
    base_label = labels[0]
    base_heatmap = heatmaps[base_label]
    spec_lookup = {
        label: (start, end, start_ns, end_ns)
        for label, start, end, start_ns, end_ns in period_specs
    }

    for idx, label in enumerate(labels):
        heatmap = heatmaps[label]
        ax_main = axes[0, idx]
        im = ax_main.imshow(
            heatmap,
            cmap=cmap_main,
            origin="lower",
            extent=[0, width, 0, height],
            vmin=0.0,
            vmax=1.0,
        )
        ax_main.set_title(label.replace("_", " ").title())
        ax_main.set_xlabel("X (px)")
        ax_main.set_ylabel("Y (px)")
        fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label="Normalized occupancy")

        detections, covered_frames, coverage_ratio = coverage_info.get(label, (0, 0, 0.0))
        _, _, start_ns, end_ns = spec_lookup[label]
        duration_text = _format_duration(start_ns, end_ns)
        ax_main.text(
            0.02,
            0.98,
            f"detections: {detections}\n"
            f"covered frames: {covered_frames}\n"
            f"coverage: {coverage_ratio*100:.1f}%\n"
            f"duration: {duration_text}",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            fontsize=9,
        )

        ax_diff = axes[1, idx]
        if label == base_label:
            diff = heatmap
            im_diff = ax_diff.imshow(
                diff,
                cmap=cmap_main,
                origin="lower",
                extent=[0, width, 0, height],
                vmin=0.0,
                vmax=1.0,
            )
            ax_diff.set_title("Baseline occupancy")
            fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label="Normalized occupancy")
        else:
            diff = heatmap - base_heatmap
            vmax = np.max(np.abs(diff)) if np.any(diff) else 1.0
            im_diff = ax_diff.imshow(
                diff,
                cmap="RdBu_r",
                origin="lower",
                extent=[0, width, 0, height],
                vmin=-vmax,
                vmax=vmax,
            )
            ax_diff.set_title("Change from pre-training")
            fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label="Δ occupancy")
        ax_diff.set_xlabel("X (px)")
        ax_diff.set_ylabel("Y (px)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def _save_or_show(
    fig: plt.Figure,
    base_path: Optional[Path],
    *,
    suffix: str,
    show: bool,
    console: Optional[Console] = None,
) -> None:
    if base_path:
        if suffix:
            target = base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")
        else:
            target = base_path
        fig.savefig(target, dpi=150, bbox_inches="tight")
        if console is not None:
            console.log(f"[green]Saved figure to {target}[/green]")
    if show and not base_path:
        plt.show()
    else:
        plt.close(fig)


def _stack_online_positions(online_fields: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    pos_x = online_fields.get("chaser_pos_x")
    pos_y = online_fields.get("chaser_pos_y")
    if pos_x is None or pos_y is None:
        return None
    return np.column_stack([pos_x, pos_y])


def _create_heatmap(
    positions: np.ndarray,
    width: int,
    height: int,
    bin_size: int,
    smooth_sigma: float,
) -> np.ndarray:
    if positions.size == 0:
        x_bins = np.arange(0, width + bin_size, bin_size)
        y_bins = np.arange(0, height + bin_size, bin_size)
        return np.zeros((y_bins.size - 1, x_bins.size - 1), dtype=np.float64)

    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]
    x_bins = np.arange(0, width + bin_size, bin_size)
    y_bins = np.arange(0, height + bin_size, bin_size)
    heatmap, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=[x_bins, y_bins],
    )
    if smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap.T  # transpose for imshow (y first)


def _resolve_latest(parent: zarr.Group, run_name: Optional[str], label: str, console: Console) -> Tuple[zarr.Group, str]:
    if label == "track_kinematics_run":
        return _resolve_track_kinematics_run(parent, run_name, console)

    if run_name:
        if run_name not in parent:
            raise ValueError(f"{label} '{run_name}' not found.")
        console.log(f"[cyan]Using {label}:[/cyan] {run_name}")
        return parent[run_name], run_name

    latest = parent.attrs.get("latest")
    if not latest:
        raise ValueError(f"{label} has no 'latest' attribute; specify --{label.replace('_', '-')}.")
    if latest not in parent:
        raise ValueError(f"{label} '{latest}' referenced by 'latest' is missing.")
    console.log(f"[cyan]Using {label}:[/cyan] {latest} (latest)")
    return parent[latest], latest


def _resolve_track_kinematics_run(
    parent: zarr.Group, run_name: Optional[str], console: Console
) -> Tuple[zarr.Group, str]:
    online_parent = parent.get("online")
    offline_parent = parent.get("offline")

    def lookup(sub_parent: Optional[zarr.Group], name: str) -> Optional[zarr.Group]:
        if sub_parent is None:
            return None
        return sub_parent.get(name)

    def resolve_name(name: str) -> Optional[Tuple[zarr.Group, str]]:
        if "/" in name:
            prefix, rest = name.split("/", 1)
            if prefix == "online":
                group = lookup(online_parent, rest)
                if group is not None:
                    return group, f"online/{rest}"
            elif prefix == "offline":
                group = lookup(offline_parent, rest)
                if group is not None:
                    return group, f"offline/{rest}"
            return None
        group = lookup(online_parent, name)
        if group is not None:
            return group, f"online/{name}"
        group = lookup(offline_parent, name)
        if group is not None:
            return group, f"offline/{name}"
        return None

    if run_name:
        resolved = resolve_name(run_name)
        if resolved is None:
            raise ValueError(f"track_kinematics_run '{run_name}' not found.")
        group, label = resolved
        console.log(f"[cyan]Using track_kinematics_run:[/cyan] {label}")
        return group, label

    candidates: list[Tuple[zarr.Group, str]] = []

    if online_parent is not None:
        latest_online = online_parent.attrs.get("latest")
        if isinstance(latest_online, str):
            resolved = resolve_name(f"online/{latest_online}")
            if resolved is not None:
                candidates.append(resolved)

    if offline_parent is not None:
        latest_offline = offline_parent.attrs.get("latest")
        if isinstance(latest_offline, str):
            resolved = resolve_name(f"offline/{latest_offline}")
            if resolved is not None:
                candidates.append(resolved)

    legacy_latest = parent.attrs.get("latest")
    if isinstance(legacy_latest, str):
        resolved = resolve_name(legacy_latest)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)

    if not candidates:
        console.log("[yellow]No track kinematics runs referenced by 'latest'; scanning available runs.[/yellow]")
        for subgroup, prefix in ((online_parent, "online"), (offline_parent, "offline")):
            if subgroup is None:
                continue
            for name in subgroup.group_keys():
                candidates.append((subgroup[name], f"{prefix}/{name}"))

    if not candidates:
        raise ValueError("No track kinematics runs found under analysis/track_kinematics_runs.")

    group, label = candidates[0]
    console.log(f"[cyan]Using track_kinematics_run:[/cyan] {label}")
    return group, label


def _format_duration(start_ns: Optional[int], end_ns: Optional[int]) -> str:
    if start_ns is None or end_ns is None:
        return "n/a"
    seconds = (end_ns - start_ns) / 1e9
    return f"{seconds:.1f}s"


def plot_training_heatmaps(
    zarr_path: Path,
    track_kinematics_run_name: Optional[str],
    stimulus_run_name: Optional[str],
    bin_size: int,
    smooth_sigma: float,
    save_path: Optional[Path],
    show: bool,
    include_chaser: bool,
    chaser_index: int,
) -> None:
    console = Console()
    console.log(f"[bold]Opening Zarr archive:[/bold] {zarr_path}")
    root = zarr.open(zarr_path, mode="r")

    analysis = root.get("analysis")
    if analysis is None:
        raise ValueError("Archive missing 'analysis' group.")

    movement_parent = analysis.get("track_kinematics_runs")
    if movement_parent is None or not movement_parent:
        raise ValueError("No track kinematics runs found under analysis/track_kinematics_runs.")
    movement_run_group, track_kinematics_run_name = _resolve_latest(
        movement_parent,
        track_kinematics_run_name,
        "track_kinematics_run",
        console,
    )

    stimulus_parent = analysis.get("stimulus_runs")
    if stimulus_parent is None or not stimulus_parent:
        raise ValueError("No stimulus runs found under analysis/stimulus_runs. Import the stimulus H5 first.")
    stimulus_run_group, stimulus_run_name = _resolve_latest(
        stimulus_parent, stimulus_run_name, "stimulus_run", console
    )

    console.log("[bold]Loading stimulus metadata...[/bold]")
    events = _load_events(stimulus_run_group)
    frame_metadata = _load_frame_metadata(stimulus_run_group)
    stim_to_camera = _build_stim_to_camera_map(frame_metadata)
    camera_frames_all = frame_metadata.get("triggering_camera_frame_id")
    if camera_frames_all is None:
        raise ValueError("Stimulus metadata missing 'triggering_camera_frame_id'.")
    camera_frames_all = camera_frames_all.astype(np.int64, copy=False)

    console.log("[bold]Loading track kinematics run positions...[/bold]")
    frames, positions = _collect_positions(movement_run_group)

    periods = _determine_periods(events, stim_to_camera, camera_frames_all, console)

    console.log("[bold]Computing heatmaps...[/bold]")
    width = int(root.attrs.get("width", 4512))
    height = int(root.attrs.get("height", 4512))

    period_specs = [
        ("pre_training", periods.pre_start, periods.pre_end, periods.pre_start_ns, periods.pre_end_ns),
        ("training", periods.train_start, periods.train_end, periods.train_start_ns, periods.train_end_ns),
        ("post_training", periods.post_start, periods.post_end, periods.post_start_ns, periods.post_end_ns),
    ]

    heatmaps, coverage_info = _compute_period_heatmaps(
        frames=frames,
        positions=positions,
        period_specs=period_specs,
        width=width,
        height=height,
        bin_size=bin_size,
        smooth_sigma=smooth_sigma,
    )
    spec_lookup = {label: (start, end, start_ns, end_ns) for label, start, end, start_ns, end_ns in period_specs}
    for label, metrics in coverage_info.items():
        detections, covered_frames, coverage_ratio = metrics
        start, end, _, _ = spec_lookup[label]
        console.log(
            f"[blue]{label}:[/blue] frames {start}-{end}, detections={detections}, "
            f"covered_frames={covered_frames}, coverage={coverage_ratio*100:.1f}%"
        )
    fig = _render_heatmap_grid(
        heatmaps=heatmaps,
        coverage_info=coverage_info,
        period_specs=period_specs,
        width=width,
        height=height,
        title=(
            f"Training Heatmaps – track kinematics run {track_kinematics_run_name}, stimulus run {stimulus_run_name}\n"
            f"bin size {bin_size}px, smoothing σ={smooth_sigma}"
        ),
    )
    _save_or_show(fig, save_path, suffix="", show=show, console=console)

    if include_chaser:
        console.log("[bold]Loading chaser metrics bundle...[/bold]")
        bundle = load_chaser_metrics(
            zarr_path,
            stimulus_run=stimulus_run_name,
            chaser_index=chaser_index,
        )
        chaser_frames = bundle.camera_frame_ids
        online_positions = _stack_online_positions(bundle.online)
        offline_positions = bundle.offline.get("chaser_position_px")

        dataset_specs = [
            ("online", online_positions, "Online chaser telemetry"),
            ("offline", offline_positions, "Offline chaser metrics"),
        ]
        for suffix_name, positions_dataset, description in dataset_specs:
            if positions_dataset is None:
                console.log(f"[yellow]Skipping {suffix_name} chaser heatmaps – positions unavailable.[/yellow]")
                continue
            heatmaps_ds, coverage_ds = _compute_period_heatmaps(
                frames=chaser_frames,
                positions=positions_dataset,
                period_specs=period_specs,
                width=width,
                height=height,
                bin_size=bin_size,
                smooth_sigma=smooth_sigma,
            )
            for label, metrics in coverage_ds.items():
                detections, covered_frames, coverage_ratio = metrics
                start, end, _, _ = spec_lookup[label]
                console.log(
                    f"[blue]{label} ({suffix_name}):[/blue] frames {start}-{end}, "
                    f"detections={detections}, covered_frames={covered_frames}, "
                    f"coverage={coverage_ratio*100:.1f}%"
                )
            fig_ds = _render_heatmap_grid(
                heatmaps=heatmaps_ds,
                coverage_info=coverage_ds,
                period_specs=period_specs,
                width=width,
                height=height,
                title=(
                    f"Chaser Heatmaps ({description}) – stimulus run {stimulus_run_name}, chaser index {chaser_index}\n"
                    f"bin size {bin_size}px, smoothing σ={smooth_sigma}"
                ),
            )
            suffix_path = f"_chaser_{suffix_name}"
            _save_or_show(fig_ds, save_path, suffix=suffix_path, show=show, console=console)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training heatmaps from a Palette Zarr archive.")
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--track-kinematics-run",
        dest="track_kinematics_run",
        help="Specific track kinematics run name (defaults to latest).",
    )
    parser.add_argument("--stimulus-run", help="Specific stimulus run name under analysis/stimulus_runs.")
    parser.add_argument("--bin-size", type=int, default=80, help="Heatmap bin size in pixels (default: 80).")
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.2,
        help="Gaussian smoothing sigma for heatmaps (default: 1.2; set 0 to disable).",
    )
    parser.add_argument("--save", type=Path, help="Path to save the figure instead of showing interactively.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot interactively.")
    parser.add_argument(
        "--include-chaser",
        action="store_true",
        help="Generate additional heatmaps for chaser positions using online/offline metrics.",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to visualize when plotting chaser heatmaps (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plot_training_heatmaps(
        zarr_path=args.zarr_path,
        track_kinematics_run_name=args.track_kinematics_run,
        stimulus_run_name=args.stimulus_run,
        bin_size=args.bin_size,
        smooth_sigma=args.smooth_sigma,
        save_path=args.save,
        show=not args.no_show and args.save is None,
        include_chaser=args.include_chaser,
        chaser_index=args.chaser_index,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
