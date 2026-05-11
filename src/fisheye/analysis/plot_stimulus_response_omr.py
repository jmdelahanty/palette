"""Persist OMR stimulus-response review visualizations.

The numeric arrays in ``stimulus_response_runs`` remain the source of truth.
This module renders a compact PNG review snapshot and a small interactive spec
that points viewers back to the canonical OMR arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import zarr
from rich.console import Console

from fisheye.shared.plot_artifacts import (
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.stage_provenance import build_stage_provenance
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.analysis.stimulus_response_io import moving_grating_omr_steps
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root


STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID = "palette.stimulus_response.omr_summary_plot.v1"
STIMULUS_RESPONSE_OMR_PLOT_RENDERER = "palette-stimulus-response-omr-summary-v1"
OMR_SUMMARY_PNG_ARTIFACT_NAME = "stimulus_response_omr_summary_png"
OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME = "stimulus_response_omr_summary_interactive"
OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME = "stimulus_response_omr_bout_trajectory_png"
OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME = "stimulus_response_omr_bout_trajectory_interactive"


@dataclass(frozen=True)
class OMRStepSummary:
    step_key: str
    step_index: int
    step_name: str
    start_frame: int
    end_frame: int
    duration_s: float
    stimulus_direction_deg: float
    arena_center_mm: Optional[tuple[float, float]]
    arena_axis_extent_mm: Optional[float]
    arena_geometry_source: str
    per_fish: Mapping[str, np.ndarray]
    per_bout: Mapping[str, np.ndarray]
    windows: Mapping[str, np.ndarray]
    early_windows: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class TrackSeries:
    fish_id: int
    frame_indices: np.ndarray
    time_seconds: np.ndarray
    positions_mm: np.ndarray
    heading_degrees: np.ndarray


_json_safe = json_attr_safe


def _artifact_signature(payload: Mapping[str, Any]) -> str:
    data = strict_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_array(group: zarr.Group, name: str) -> np.ndarray:
    if name not in group:
        return np.asarray([], dtype=np.float32)
    return np.asarray(group[name][:])


def _as_float_pair(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    return float(arr[0]), float(arr[1])


def _nanmean(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _metric_by_step(steps: Sequence[OMRStepSummary], name: str) -> np.ndarray:
    return np.asarray(
        [_nanmean(step.per_fish.get(name, np.asarray([], dtype=np.float32))) for step in steps],
        dtype=np.float64,
    )


def _finite_limits(*arrays: np.ndarray, fallback: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    finite_parts = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    finite = np.concatenate([a[np.isfinite(a)] for a in finite_parts]) if finite_parts else np.asarray([])
    if finite.size == 0:
        return fallback
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1e-9:
        pad = 1.0 if abs(hi) < 1e-9 else abs(hi) * 0.1
    else:
        pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _step_label(step: OMRStepSummary) -> str:
    return f"{step.step_index}"


def _stimulus_response_layout(run_group: zarr.Group) -> str:
    return str(run_group.attrs.get("layout") or run_group.attrs.get("storage_layout") or "hierarchical_v1")


def _has_path(group: zarr.Group, path: str) -> bool:
    try:
        group[path]
    except (KeyError, ValueError, TypeError):
        return False
    return True


def _summary_source_paths(run_group: zarr.Group, run_path: str, layout: str) -> dict[str, str]:
    if layout == "compact_tabular_v2":
        paths = {
            "run": run_path,
        }
        for name in (
            "step_index",
            "global_omr_per_fish",
            "moving_grating_omr_per_fish",
            "moving_grating_omr_per_bout",
            "moving_grating_omr_windows",
            "moving_grating_omr_early_windows",
        ):
            if name in run_group:
                paths[name] = f"{run_path}/{name}"
        return paths
    paths = {
        "run": run_path,
        "steps": f"{run_path}/steps",
    }
    if _has_path(run_group, "global/omr"):
        paths["global_omr"] = f"{run_path}/global/omr"
    return paths


def _summary_source_filters(steps: Sequence[OMRStepSummary], layout: str) -> dict[str, Any]:
    step_indices = [int(step.step_index) for step in steps]
    filters: dict[str, Any] = {
        "step_indices": step_indices,
        "stimulus_family": "moving_grating",
        "metric_family": "moving_grating_omr",
    }
    if layout == "compact_tabular_v2":
        filters["compact_tables_filtered_by"] = ["step_index", "metric_family"]
    return filters


def _step_spec_paths(
    run_path: str,
    step: OMRStepSummary,
    layout: str,
    source_paths: Mapping[str, Any],
) -> dict[str, Any]:
    if layout == "compact_tabular_v2":
        return {
            "step_path": source_paths.get("step_index", f"{run_path}/step_index"),
            "omr_path": source_paths.get("moving_grating_omr_per_fish"),
            "per_fish_path": source_paths.get("moving_grating_omr_per_fish"),
            "per_bout_path": source_paths.get("moving_grating_omr_per_bout"),
            "windows_path": source_paths.get("moving_grating_omr_windows"),
            "early_windows_path": source_paths.get("moving_grating_omr_early_windows"),
            "source_filters": {
                "step_index": int(step.step_index),
                "stimulus_family": "moving_grating",
                "metric_family": "moving_grating_omr",
            },
        }
    omr_path = f"{run_path}/steps/{step.step_key}/grating/omr"
    return {
        "step_path": f"{run_path}/steps/{step.step_key}",
        "omr_path": omr_path,
        "per_fish_path": f"{omr_path}/per_fish",
        "per_bout_path": f"{omr_path}/per_bout",
        "windows_path": f"{omr_path}/windows",
        "early_windows_path": f"{omr_path}/early_windows",
        "source_filters": {"step_index": int(step.step_index)},
    }


def load_omr_step_summaries(run_group: zarr.Group) -> list[OMRStepSummary]:
    """Load MOVING_GRATING OMR summaries from a stimulus_response run."""

    summaries: list[OMRStepSummary] = []
    for step in moving_grating_omr_steps(run_group):
        omr = step.moving_grating_omr
        if omr is None:
            continue
        attrs = omr.attrs
        arena_extent = attrs.get("arena_axis_extent_mm")
        summaries.append(
            OMRStepSummary(
                step_key=step.step_key,
                step_index=step.step_index,
                step_name=step.step_name,
                start_frame=int(step.start_frame or 0),
                end_frame=int(step.end_frame or 0),
                duration_s=float(step.duration_s or 0.0),
                stimulus_direction_deg=float(attrs.get("stimulus_direction_deg", float("nan"))),
                arena_center_mm=_as_float_pair(attrs.get("arena_center_mm")),
                arena_axis_extent_mm=(
                    float(arena_extent)
                    if arena_extent is not None and np.isfinite(float(arena_extent))
                    else None
                ),
                arena_geometry_source=str(attrs.get("arena_geometry_source", "unknown")),
                per_fish=omr.per_fish,
                per_bout=omr.per_bout,
                windows=omr.windows,
                early_windows=omr.early_windows,
            )
        )
    summaries.sort(key=lambda s: s.step_index)
    return summaries


def _select_window_length(steps: Sequence[OMRStepSummary]) -> Optional[float]:
    lengths: list[float] = []
    for step in steps:
        arr = step.windows.get("window_length_s")
        if arr is None:
            continue
        for value in np.asarray(arr, dtype=np.float64):
            if np.isfinite(value) and value > 0:
                # Round to avoid showing separate keys for float32 noise.
                lengths.append(round(float(value), 6))
    if not lengths:
        return None
    return min(lengths)


def _grating_direction_xy(direction_deg: float) -> np.ndarray:
    rad = math.radians(float(direction_deg))
    return np.asarray([math.cos(rad), math.sin(rad)], dtype=np.float64)


def _heading_to_vector(heading_deg: np.ndarray | float) -> np.ndarray:
    rad = np.deg2rad(np.asarray(heading_deg, dtype=np.float64))
    return np.stack([np.cos(rad), np.sin(rad)], axis=-1)


def _load_track_series(
    root: zarr.Group,
    *,
    kinematics_type: str,
    kinematics_run: str,
    track_id: Optional[int] = None,
) -> TrackSeries:
    kin_group, resolved_run = resolve_zarr_run(
        root,
        f"analysis/track_kinematics_runs/{kinematics_type}",
        run_name=kinematics_run,
    )
    tracks_group = kin_group["tracks"]
    track_names = sorted(
        (name for name in tracks_group.group_keys() if str(name).startswith("id_")),
        key=lambda name: int(str(name).split("_", 1)[1]),
    )
    if not track_names:
        raise ValueError(f"No tracks found in track_kinematics run: {kinematics_type}/{resolved_run}")
    if track_id is None:
        selected_name = track_names[0]
    else:
        selected_name = f"id_{int(track_id)}"
        if selected_name not in tracks_group:
            available = ", ".join(track_names)
            raise ValueError(f"Track {track_id} not found in {kinematics_type}/{resolved_run}; available: {available}")
    track_group = tracks_group[selected_name]
    fish_id = int(selected_name.split("_", 1)[1])
    frame_indices = _read_array(track_group, "frame_indices").astype(np.int64, copy=False)
    time_seconds = _read_array(track_group, "time_seconds").astype(np.float64, copy=False)
    positions_mm = _read_array(track_group, "positions_mm").astype(np.float64, copy=False)
    heading = _read_array(track_group, "smoothed_heading_degrees")
    if heading.size == 0:
        heading = _read_array(track_group, "heading_degrees")
    heading_degrees = heading.astype(np.float64, copy=False)
    if not (frame_indices.size == time_seconds.size == positions_mm.shape[0] == heading_degrees.size):
        raise ValueError(f"Track arrays have inconsistent lengths for {kinematics_type}/{resolved_run}/{selected_name}")
    return TrackSeries(
        fish_id=fish_id,
        frame_indices=frame_indices,
        time_seconds=time_seconds,
        positions_mm=positions_mm,
        heading_degrees=heading_degrees,
    )


def _track_slice_for_step(track: TrackSeries, step: OMRStepSummary) -> np.ndarray:
    return np.flatnonzero(
        (track.frame_indices >= int(step.start_frame))
        & (track.frame_indices < int(step.end_frame))
        & np.isfinite(track.positions_mm).all(axis=1)
    )


def _slice_for_frame_range(track: TrackSeries, start_frame: int, end_frame: int) -> np.ndarray:
    return np.flatnonzero(
        (track.frame_indices >= int(start_frame))
        & (track.frame_indices <= int(end_frame))
        & np.isfinite(track.positions_mm).all(axis=1)
    )


def _label_color(label: int) -> str:
    if label > 0:
        return "#2a9d8f"
    if label < 0:
        return "#e76f51"
    return "#8d99ae"


def _label_name(label: int) -> str:
    if label > 0:
        return "aligned"
    if label < 0:
        return "opposing"
    return "ambiguous"


def _plot_arena_reference(ax: Any, step: OMRStepSummary) -> None:
    if step.arena_center_mm is None or step.arena_axis_extent_mm is None:
        return
    try:
        import matplotlib.patches as patches
    except Exception:  # pragma: no cover - matplotlib import failure is caught by caller tests.
        return
    center = step.arena_center_mm
    radius = float(step.arena_axis_extent_mm)
    if not np.isfinite(radius) or radius <= 0:
        return
    circle = patches.Circle(
        center,
        radius,
        fill=False,
        edgecolor="#d62828",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.add_patch(circle)


def render_omr_bout_trajectory_png(
    *,
    run_name: str,
    steps: Sequence[OMRStepSummary],
    track: TrackSeries,
    artifact_dpi: int = 150,
    max_steps: int = 6,
    max_heading_arrows_per_step: int = 24,
) -> bytes:
    """Render a spatial OMR bout trajectory review PNG."""

    if not steps:
        raise ValueError("No OMR step summaries found in stimulus_response run")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected_steps = list(steps[: max(1, int(max_steps))])
    n_steps = len(selected_steps)
    n_cols = min(3, n_steps)
    n_rows = int(math.ceil(n_steps / n_cols))
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.6 * n_cols, 5.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"OMR Bout Trajectories: {run_name} | track {track.fish_id}",
        fontsize=15,
        fontweight="bold",
    )

    for ax in axes_arr.ravel()[n_steps:]:
        ax.axis("off")

    for ax, step in zip(axes_arr.ravel(), selected_steps):
        step_idx = _track_slice_for_step(track, step)
        ax.set_title(f"Step {step.step_index}: {step.step_name}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")
        _plot_arena_reference(ax, step)

        if step_idx.size >= 2:
            step_pos = track.positions_mm[step_idx]
            ax.plot(step_pos[:, 0], step_pos[:, 1], color="#1f77b4", linewidth=1.2, alpha=0.55, label="track")

        direction_xy = _grating_direction_xy(step.stimulus_direction_deg)
        if step_idx.size:
            origin = np.nanmean(track.positions_mm[step_idx], axis=0)
        elif step.arena_center_mm is not None:
            origin = np.asarray(step.arena_center_mm, dtype=np.float64)
        else:
            origin = np.asarray([0.0, 0.0], dtype=np.float64)
        arrow_len = 0.25 * float(step.arena_axis_extent_mm or 10.0)
        ax.arrow(
            origin[0],
            origin[1],
            direction_xy[0] * arrow_len,
            direction_xy[1] * arrow_len,
            width=0.025 * max(arrow_len, 1.0),
            head_width=0.16 * max(arrow_len, 1.0),
            length_includes_head=True,
            color="#f4a261",
            alpha=0.9,
            label="stimulus direction",
        )

        bout_fish = np.asarray(step.per_bout.get("fish_id", []), dtype=np.int64)
        bout_start = np.asarray(step.per_bout.get("start_frame", []), dtype=np.int64)
        bout_end = np.asarray(step.per_bout.get("end_frame", []), dtype=np.int64)
        bout_score = np.asarray(step.per_bout.get("per_bout_omr_score", []), dtype=np.float64)
        bout_label = np.asarray(step.per_bout.get("correct_label", []), dtype=np.int64)
        if bout_label.size != bout_start.size:
            bout_label = np.zeros(bout_start.size, dtype=np.int64)
        if bout_score.size != bout_start.size:
            bout_score = np.full(bout_start.size, np.nan, dtype=np.float64)

        matching = (
            np.flatnonzero(bout_fish == int(track.fish_id))
            if bout_fish.size == bout_start.size
            else np.arange(bout_start.size)
        )
        arrow_drawn = 0
        arrow_stride = max(1, int(math.ceil(max(1, matching.size) / max(1, max_heading_arrows_per_step))))
        label_seen: set[int] = set()
        for local_count, idx in enumerate(matching):
            segment_idx = _slice_for_frame_range(track, int(bout_start[idx]), int(bout_end[idx]))
            if segment_idx.size < 2:
                continue
            pos = track.positions_mm[segment_idx]
            label = int(bout_label[idx])
            color = _label_color(label)
            line_label = _label_name(label) if label not in label_seen else None
            label_seen.add(label)
            ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2.8, alpha=0.95, label=line_label)

            if local_count % arrow_stride == 0 and arrow_drawn < max_heading_arrows_per_step:
                start = segment_idx[0]
                heading_vec = _heading_to_vector(track.heading_degrees[start])
                arrow_scale = 0.08 * float(step.arena_axis_extent_mm or 10.0)
                ax.arrow(
                    track.positions_mm[start, 0],
                    track.positions_mm[start, 1],
                    heading_vec[0] * arrow_scale,
                    heading_vec[1] * arrow_scale,
                    head_width=0.04 * float(step.arena_axis_extent_mm or 10.0),
                    length_includes_head=True,
                    color="black",
                    alpha=0.75,
                )
                arrow_drawn += 1
            if np.isfinite(bout_score[idx]) and segment_idx.size >= 2:
                mid = segment_idx[segment_idx.size // 2]
                ax.scatter(
                    track.positions_mm[mid, 0],
                    track.positions_mm[mid, 1],
                    c=[color],
                    s=18,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.4,
                )

        if step_idx.size:
            pos = track.positions_mm[step_idx]
            lo_x, hi_x = _finite_limits(pos[:, 0], fallback=(-1.0, 1.0))
            lo_y, hi_y = _finite_limits(pos[:, 1], fallback=(-1.0, 1.0))
            if step.arena_center_mm is not None and step.arena_axis_extent_mm is not None:
                cx, cy = step.arena_center_mm
                radius = step.arena_axis_extent_mm
                lo_x = min(lo_x, cx - radius)
                hi_x = max(hi_x, cx + radius)
                lo_y = min(lo_y, cy - radius)
                hi_y = max(hi_y, cy + radius)
            ax.set_xlim(lo_x, hi_x)
            ax.set_ylim(lo_y, hi_y)

        ax.text(
            0.02,
            0.02,
            (
                f"bouts={matching.size}\n"
                f"direction={step.stimulus_direction_deg:g} deg\n"
                f"frames={step.start_frame}-{step.end_frame}"
            ),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            custom = [
                Line2D([0], [0], color="#1f77b4", lw=1.2, alpha=0.55, label="track"),
                Line2D([0], [0], color="#2a9d8f", lw=2.8, label="aligned bout"),
                Line2D([0], [0], color="#e76f51", lw=2.8, label="opposing bout"),
                Line2D([0], [0], color="#8d99ae", lw=2.8, label="ambiguous bout"),
                Line2D([0], [0], color="#f4a261", lw=2.8, label="stimulus direction"),
            ]
            ax.legend(handles=custom, loc="best", fontsize=7)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(artifact_dpi), bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_omr_summary_png(
    *,
    run_name: str,
    steps: Sequence[OMRStepSummary],
    artifact_dpi: int = 150,
) -> bytes:
    """Render the OMR summary PNG and return encoded bytes."""

    if not steps:
        raise ValueError("No OMR step summaries found in stimulus_response run")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.arange(len(steps), dtype=np.float64)
    labels = [_step_label(step) for step in steps]
    path_index = _metric_by_step(steps, "omr_path_index")
    bout_path_index = _metric_by_step(steps, "bout_path_index")
    bout_choice = _metric_by_step(steps, "bout_choice_index")
    time_choice = _metric_by_step(steps, "time_choice_index")
    start_pos = _metric_by_step(steps, "start_position_axis_norm")
    mean_pos = _metric_by_step(steps, "mean_position_axis_norm")
    end_pos = _metric_by_step(steps, "end_position_axis_norm")
    correct_side = _metric_by_step(steps, "fraction_time_correct_side")
    aligned_latency = _metric_by_step(steps, "first_aligned_bout_latency_s")
    classified_latency = _metric_by_step(steps, "first_classified_bout_latency_s")
    opposing_latency = _metric_by_step(steps, "first_opposing_bout_latency_s")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    fig.suptitle(f"OMR Stimulus Response Summary: {run_name}", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    width = 0.2
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.bar(x - 1.5 * width, path_index, width=width, label="Path index", color="#2a9d8f")
    ax.bar(x - 0.5 * width, bout_path_index, width=width, label="Bout path index", color="#43aa8b")
    ax.bar(x + 0.5 * width, bout_choice, width=width, label="Bout count choice", color="#e76f51")
    ax.bar(x + 1.5 * width, time_choice, width=width, label="Time choice index", color="#457b9d")
    ax.set_title("Direction Selectivity by Step")
    ax.set_ylabel("Signed index [-1, 1]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.axhspan(-1.0, 1.0, color="#edf2f4", alpha=0.85, label="Nominal arena span")
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.plot(x, start_pos, marker="o", label="Start", color="#1d3557")
    ax.plot(x, mean_pos, marker="s", label="Mean", color="#f4a261")
    ax.plot(x, end_pos, marker="^", label="End", color="#e63946")
    ax2 = ax.twinx()
    ax2.plot(x, correct_side, marker="D", linestyle="--", color="#6a994e", label="Correct side fraction")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Fraction")
    ax.set_title("Arena-Axis Occupancy / Opportunity")
    ax.set_ylabel("Position along stimulus axis (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*_finite_limits(start_pos, mean_pos, end_pos, np.asarray([-1.0, 1.0]), fallback=(-1.25, 1.25)))
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(x, classified_latency, marker="o", label="First classified", color="#264653")
    ax.plot(x, aligned_latency, marker="s", label="First aligned", color="#2a9d8f")
    ax.plot(x, opposing_latency, marker="^", label="First opposing", color="#e76f51")
    ax.set_title("First Bout Latencies")
    ax.set_ylabel("Latency from step start (s)")
    ax.set_xlabel("Step index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*_finite_limits(classified_latency, aligned_latency, opposing_latency, np.asarray([0.0]), fallback=(0.0, 1.0)))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    window_length = _select_window_length(steps)
    if window_length is None:
        ax.text(0.5, 0.5, "No windowed OMR metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        for step in steps:
            lengths = np.asarray(step.windows.get("window_length_s", []), dtype=np.float64)
            starts = np.asarray(step.windows.get("start_time_s", []), dtype=np.float64)
            values = np.asarray(step.windows.get("omr_path_index", []), dtype=np.float64)
            fish_ids = np.asarray(step.windows.get("fish_id", []))
            if lengths.size == 0 or starts.size == 0 or values.size == 0:
                continue
            mask = np.isclose(lengths, window_length, rtol=1e-5, atol=1e-5)
            if not np.any(mask):
                continue
            # Average across fish per window start for a compact step-level trace.
            step_starts = starts[mask]
            step_values = values[mask]
            step_fish = fish_ids[mask] if fish_ids.size == values.size else np.zeros_like(step_starts)
            unique_starts = np.unique(step_starts)
            y_vals = []
            for start in unique_starts:
                start_mask = np.isclose(step_starts, start, rtol=1e-5, atol=1e-5)
                _ = step_fish[start_mask]  # retained so the grouping intent is explicit.
                y_vals.append(_nanmean(step_values[start_mask]))
            ax.plot(unique_starts, y_vals, marker=".", linewidth=1.1, label=f"step {step.step_index}")
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_title(f"Windowed Path Index ({window_length:g} s windows)")
        ax.set_ylabel("OMR path index")
        ax.set_xlabel("Time from step start (s)")
        ax.set_ylim(-1.1, 1.1)
        if len(steps) <= 8:
            ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    footer_parts = []
    geometry_sources = sorted({s.arena_geometry_source for s in steps})
    if geometry_sources:
        footer_parts.append(f"arena geometry: {', '.join(geometry_sources)}")
    directions = [
        f"step {s.step_index}: {s.stimulus_direction_deg:g} deg"
        for s in steps
        if np.isfinite(s.stimulus_direction_deg)
    ]
    if directions:
        footer_parts.append("directions: " + "; ".join(directions[:6]))
    if footer_parts:
        fig.text(0.01, 0.005, " | ".join(footer_parts), fontsize=8, color="0.35")

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(artifact_dpi), bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _build_interactive_spec(
    *,
    run_name: str,
    run_path: str,
    steps: Sequence[OMRStepSummary],
    layout: str,
    source_paths: Mapping[str, Any],
    source_filters: Mapping[str, Any],
) -> Mapping[str, Any]:
    step_specs = []
    for step in steps:
        step_paths = _step_spec_paths(run_path, step, layout, source_paths)
        step_specs.append(
            {
                "step_index": step.step_index,
                "step_name": step.step_name,
                "step_path": step_paths["step_path"],
                "omr_path": step_paths["omr_path"],
                "stimulus_direction_deg": step.stimulus_direction_deg,
                "per_fish_path": step_paths["per_fish_path"],
                "per_bout_path": step_paths["per_bout_path"],
                "windows_path": step_paths["windows_path"],
                "early_windows_path": step_paths["early_windows_path"],
                "source_filters": step_paths["source_filters"],
                "primary_fields": [
                    "omr_path_index",
                    "bout_path_index",
                    "bout_fraction_correct_weighted_by_path",
                    "bout_choice_index",
                    "time_choice_index",
                    "start_position_axis_norm",
                    "end_position_axis_norm",
                    "fraction_time_correct_side",
                    "first_aligned_bout_latency_s",
                    "first_classified_bout_latency_s",
                ],
            }
        )
    return {
        "schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
        "renderer": STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
        "run_name": run_name,
        "run_path": run_path,
        "layout": layout,
        "artifact_family": "stimulus_response_omr_summary",
        "source_paths": source_paths,
        "source_filters": source_filters,
        "panels": [
            "direction_selectivity_by_step",
            "arena_axis_occupancy_opportunity",
            "first_bout_latencies",
            "windowed_path_index",
            "early_response_windows",
        ],
        "steps": step_specs,
    }


def _build_bout_trajectory_interactive_spec(
    *,
    run_name: str,
    run_path: str,
    steps: Sequence[OMRStepSummary],
    track: TrackSeries,
    kinematics_type: str,
    kinematics_run: str,
    layout: str,
    source_paths: Mapping[str, Any],
    source_filters: Mapping[str, Any],
) -> Mapping[str, Any]:
    step_specs = []
    for step in steps:
        step_paths = _step_spec_paths(run_path, step, layout, source_paths)
        step_specs.append(
            {
                "step_index": step.step_index,
                "step_name": step.step_name,
                "step_path": step_paths["step_path"],
                "omr_per_bout_path": step_paths["per_bout_path"],
                "source_filters": step_paths["source_filters"],
                "stimulus_direction_deg": step.stimulus_direction_deg,
                "start_frame": step.start_frame,
                "end_frame": step.end_frame,
            }
        )
    track_path = (
        f"analysis/track_kinematics_runs/{kinematics_type}/"
        f"{kinematics_run}/tracks/id_{track.fish_id}"
    )
    return {
        "schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
        "renderer": STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
        "run_name": run_name,
        "run_path": run_path,
        "layout": layout,
        "artifact_family": "stimulus_response_omr_bout_trajectory",
        "source_paths": source_paths,
        "source_filters": source_filters,
        "track_id": int(track.fish_id),
        "track_path": track_path,
        "track_fields": ["frame_indices", "positions_mm", "heading_degrees", "smoothed_heading_degrees"],
        "bout_fields": ["start_frame", "end_frame", "per_bout_omr_score", "correct_label"],
        "steps": step_specs,
    }


def write_omr_summary_visualization(
    root: zarr.Group,
    *,
    run_name: Optional[str] = None,
    zarr_path: Optional[Path] = None,
    artifact_dpi: int = 150,
    track_id: Optional[int] = None,
    write_bout_trajectory: bool = True,
    max_bout_trajectory_steps: int = 6,
    save_path: Optional[Path] = None,
    bout_trajectory_save_path: Optional[Path] = None,
    command: Optional[str] = None,
    console: Optional[Console] = None,
) -> dict[str, Any]:
    """Write OMR summary PNG/spec artifacts for a stimulus_response run."""

    console = console or Console()
    run_group, resolved_run = resolve_zarr_run(
        root,
        "analysis/stimulus_response_runs",
        run_name=run_name,
    )
    run_path = f"analysis/stimulus_response_runs/{resolved_run}"
    layout = _stimulus_response_layout(run_group)
    steps = load_omr_step_summaries(run_group)
    if not steps:
        raise ValueError(f"Stimulus response run has no OMR step groups: {resolved_run}")

    png_bytes = render_omr_summary_png(
        run_name=resolved_run,
        steps=steps,
        artifact_dpi=artifact_dpi,
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(png_bytes)
        console.print(f"[green]Saved OMR summary PNG to {save_path}[/green]")

    source_paths = _summary_source_paths(run_group, run_path, layout)
    source_filters = _summary_source_filters(steps, layout)
    source_runs = {
        "stimulus_response": resolved_run,
        "track_kinematics": run_group.attrs.get("source_track_kinematics_run"),
        "stimulus": run_group.attrs.get("source_stimulus_run"),
        "swim_bout": run_group.attrs.get("source_bout_run"),
    }
    parameters = {
        "artifact_dpi": int(artifact_dpi),
        "n_omr_steps": len(steps),
        "plot_schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
        "renderer": STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
        "layout": layout,
    }
    signature = _artifact_signature(
        {
            "schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
            "run_name": resolved_run,
            "layout": layout,
            "source_paths": source_paths,
            "source_filters": source_filters,
            "source_runs": source_runs,
            "parameters": parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(
        disk_path=str(zarr_path) if zarr_path is not None else None,
        capture_env_vars=False,
    )
    provenance = build_stage_provenance(
        stage="stimulus_response_omr_visualization",
        created_at_utc=created_at_utc,
        parameters=parameters,
        inputs={
            "zarr_path": str(zarr_path) if zarr_path is not None else None,
            "source_paths": source_paths,
            "source_filters": source_filters,
            "source_runs": source_runs,
            "run_name": resolved_run,
        },
        command=command,
        version=STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{OMR_SUMMARY_PNG_ARTIFACT_NAME}",
            "interactive_artifact": f"visualizations/{OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME}",
            "artifact_signature": signature,
        },
    )

    png_result = write_png_visualization_artifact(
        run_group,
        OMR_SUMMARY_PNG_ARTIFACT_NAME,
        png_bytes,
        description="OMR stimulus-response summary PNG",
        created_by="fisheye.analysis.plot_stimulus_response_omr",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=parameters,
        extra_attrs={
            "plot_schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
            "run_name": resolved_run,
            "n_omr_steps": len(steps),
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    spec = _build_interactive_spec(
        run_name=resolved_run,
        run_path=run_path,
        steps=steps,
        layout=layout,
        source_paths=source_paths,
        source_filters=source_filters,
    )
    spec_result = write_interactive_plot_spec_artifact(
        run_group,
        OMR_SUMMARY_INTERACTIVE_ARTIFACT_NAME,
        spec,
        description="OMR stimulus-response interactive plot spec",
        created_by="fisheye.analysis.plot_stimulus_response_omr",
        renderer=STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=OMR_SUMMARY_PNG_ARTIFACT_NAME,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=parameters,
        extra_attrs={
            "plot_schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
            "run_name": resolved_run,
            "n_omr_steps": len(steps),
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    console.print(
        "[green]Wrote OMR visualization artifacts:[/green] "
        f"{png_result.path}, {spec_result.path}"
    )

    result: dict[str, Any] = {
        "run_name": resolved_run,
        "png_artifact": png_result.path,
        "interactive_artifact": spec_result.path,
        "n_omr_steps": len(steps),
    }
    if write_bout_trajectory:
        try:
            kinematics_type = str(run_group.attrs.get("source_track_kinematics_type", "offline"))
            kinematics_run = str(run_group.attrs.get("source_track_kinematics_run"))
            if not kinematics_run or kinematics_run == "None":
                raise ValueError("stimulus_response run has no source_track_kinematics_run attr")
            track = _load_track_series(
                root,
                kinematics_type=kinematics_type,
                kinematics_run=kinematics_run,
                track_id=track_id,
            )
            trajectory_png = render_omr_bout_trajectory_png(
                run_name=resolved_run,
                steps=steps,
                track=track,
                artifact_dpi=artifact_dpi,
                max_steps=max_bout_trajectory_steps,
            )
            if bout_trajectory_save_path is not None:
                bout_trajectory_save_path.parent.mkdir(parents=True, exist_ok=True)
                bout_trajectory_save_path.write_bytes(trajectory_png)
                console.print(f"[green]Saved OMR bout trajectory PNG to {bout_trajectory_save_path}[/green]")

            track_path = (
                f"analysis/track_kinematics_runs/{kinematics_type}/"
                f"{kinematics_run}/tracks/id_{track.fish_id}"
            )
            trajectory_source_paths = {
                **source_paths,
                "track": track_path,
                "track_positions": f"{track_path}/positions_mm",
                "track_headings": f"{track_path}/heading_degrees",
            }
            trajectory_source_filters = {
                **source_filters,
                "track_id": int(track.fish_id),
            }
            trajectory_parameters = {
                **parameters,
                "track_id": int(track.fish_id),
                "max_bout_trajectory_steps": int(max_bout_trajectory_steps),
                "trajectory_panel": "spatial_only_no_yaw_trace",
            }
            trajectory_signature = _artifact_signature(
                {
                    "schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
                    "artifact_family": "stimulus_response_omr_bout_trajectory",
                    "run_name": resolved_run,
                    "layout": layout,
                    "source_paths": trajectory_source_paths,
                    "source_filters": trajectory_source_filters,
                    "source_runs": source_runs,
                    "parameters": trajectory_parameters,
                }
            )
            trajectory_provenance = build_stage_provenance(
                stage="stimulus_response_omr_bout_trajectory_visualization",
                created_at_utc=created_at_utc,
                parameters=trajectory_parameters,
                inputs={
                    "zarr_path": str(zarr_path) if zarr_path is not None else None,
                    "source_paths": trajectory_source_paths,
                    "source_filters": trajectory_source_filters,
                    "source_runs": source_runs,
                    "run_name": resolved_run,
                    "track_id": int(track.fish_id),
                },
                command=command,
                version=STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
                git=get_git_info(),
                environment=env_info.get("environment"),
                platform=env_info.get("platform"),
                artifacts={
                    "png_artifact": f"visualizations/{OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME}",
                    "interactive_artifact": f"visualizations/{OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME}",
                    "artifact_signature": trajectory_signature,
                },
            )
            trajectory_png_result = write_png_visualization_artifact(
                run_group,
                OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
                trajectory_png,
                description="OMR bout trajectory summary PNG",
                created_by="fisheye.analysis.plot_stimulus_response_omr",
                artifact_signature=trajectory_signature,
                created_at_utc=created_at_utc,
                source_paths=trajectory_source_paths,
                source_runs=source_runs,
                parameters=trajectory_parameters,
                extra_attrs={
                    "plot_schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
                    "run_name": resolved_run,
                    "track_id": int(track.fish_id),
                    "n_omr_steps": len(steps),
                    "source_filters": trajectory_source_filters,
                    "provenance": trajectory_provenance,
                },
            )
            trajectory_spec = _build_bout_trajectory_interactive_spec(
                run_name=resolved_run,
                run_path=run_path,
                steps=steps,
                track=track,
                kinematics_type=kinematics_type,
                kinematics_run=kinematics_run,
                layout=layout,
                source_paths=trajectory_source_paths,
                source_filters=trajectory_source_filters,
            )
            trajectory_spec_result = write_interactive_plot_spec_artifact(
                run_group,
                OMR_BOUT_TRAJECTORY_INTERACTIVE_ARTIFACT_NAME,
                trajectory_spec,
                description="OMR bout trajectory interactive plot spec",
                created_by="fisheye.analysis.plot_stimulus_response_omr",
                renderer=STIMULUS_RESPONSE_OMR_PLOT_RENDERER,
                artifact_signature=trajectory_signature,
                created_at_utc=created_at_utc,
                snapshot_artifact=OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
                source_paths=trajectory_source_paths,
                source_runs=source_runs,
                parameters=trajectory_parameters,
                extra_attrs={
                    "plot_schema_id": STIMULUS_RESPONSE_OMR_PLOT_SCHEMA_ID,
                    "run_name": resolved_run,
                    "track_id": int(track.fish_id),
                    "n_omr_steps": len(steps),
                    "source_filters": trajectory_source_filters,
                    "provenance": trajectory_provenance,
                },
            )
            console.print(
                "[green]Wrote OMR bout trajectory artifacts:[/green] "
                f"{trajectory_png_result.path}, {trajectory_spec_result.path}"
            )
            result["bout_trajectory_png_artifact"] = trajectory_png_result.path
            result["bout_trajectory_interactive_artifact"] = trajectory_spec_result.path
        except ValueError as exc:
            console.print(f"  [yellow]OMR bout trajectory visualization skipped: {exc}[/yellow]")

    return result


def main(argv: Optional[Iterable[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(
        description="Persist OMR summary visualization artifacts for a stimulus_response run.",
    )
    parser.add_argument("zarr_path", help="Path to the Palette analysis Zarr archive.")
    parser.add_argument(
        "--run",
        dest="run_name",
        default=None,
        help="stimulus_response run name (default: latest).",
    )
    parser.add_argument(
        "--artifact-dpi",
        type=int,
        default=150,
        help="DPI for the persisted PNG artifact (default: 150).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Track ID for the OMR bout trajectory plot (default: first track).",
    )
    parser.add_argument(
        "--no-bout-trajectory",
        action="store_true",
        help="Only write the aggregate OMR summary artifacts, not the spatial bout trajectory artifacts.",
    )
    parser.add_argument(
        "--max-bout-trajectory-steps",
        type=int,
        default=6,
        help="Maximum number of OMR steps to show in the bout trajectory PNG (default: 6).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional filesystem path to also write the rendered PNG.",
    )
    parser.add_argument(
        "--save-bout-trajectory",
        type=Path,
        default=None,
        help="Optional filesystem path to also write the rendered bout trajectory PNG.",
    )
    args = parser.parse_args(argv_list)

    command = (
        " ".join(sys.argv)
        if argv_list is None
        else " ".join(["fisheye.analysis.plot_stimulus_response_omr", *map(str, argv_list)])
    )
    console = Console()
    root = open_zarr_root(args.zarr_path, mode="a")
    write_omr_summary_visualization(
        root,
        run_name=args.run_name,
        zarr_path=Path(args.zarr_path),
        artifact_dpi=args.artifact_dpi,
        track_id=args.track_id,
        write_bout_trajectory=not args.no_bout_trajectory,
        max_bout_trajectory_steps=args.max_bout_trajectory_steps,
        save_path=args.save,
        bout_trajectory_save_path=args.save_bout_trajectory,
        command=command,
        console=console,
    )


if __name__ == "__main__":
    main()
