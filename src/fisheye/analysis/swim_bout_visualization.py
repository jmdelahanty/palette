#!/usr/bin/env python3
"""
Matplotlib-based visualization for swim bout statistics stored in Palette archives.

Loads the structured datasets written by ``swim_bout_statistics`` under
``analysis/swim_bout_runs/<run>`` and produces a dashboard summarising:
  • bout-duration / distance / speed distributions
  • per-trial bout rates and active time
  • bout start/end scatter across time
  • inter-bout interval distribution and summary

Figures can be displayed interactively or saved to disk with provenance metadata
embedded via an XMP packet, matching the behaviour of other Palette visualizers.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.analysis.swim_bout_io import (
    SPEED_LEVEL_ALIASES,
    SwimBoutIOError,
    load_default_swim_bout_tables,
    load_swim_bout_tables,
)
from fisheye.shared.json_safety import decode_null_terminated_text

try:
    from PIL import Image, PngImagePlugin
except ImportError:  # pragma: no cover
    Image = None
    PngImagePlugin = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DISPLAY_SPEED_LEVEL_ORDER = ("raw", "filtered", "smoothed", "averaged", "exponential")


def _decode_bytes(arr: np.ndarray) -> np.ndarray:
    """Decode byte-string columns to unicode strings."""
    if arr.dtype.kind == "S":
        values = [decode_null_terminated_text(value) for value in arr.reshape(-1)]
        return np.asarray(values, dtype=str).reshape(arr.shape)
    return arr


def _optional_table(records: np.ndarray) -> Optional[np.ndarray]:
    """Return ``None`` for missing optional tables represented by an empty dtype."""
    if records.size == 0 and records.dtype.names == ():
        return None
    return records


def _display_speed_level(speed_level: str) -> str:
    """Return the CLI/display spelling for a canonical ``speed_*`` level."""
    return str(speed_level).replace("speed_", "", 1)


def _sort_display_speed_levels(speed_levels: Iterable[str]) -> list[str]:
    """Sort canonical speed levels in the dashboard's historical display order."""
    display_levels = {_display_speed_level(level) for level in speed_levels if level}
    order = {name: index for index, name in enumerate(DISPLAY_SPEED_LEVEL_ORDER)}
    return sorted(display_levels, key=lambda name: (order.get(name, len(order)), name))


def _load_swim_bout_run(
    zarr_path: Path,
    run_name: Optional[str],
    speed_level: str = "smoothed",
) -> Tuple[Dict[str, Any], Dict[str, Optional[np.ndarray]]]:
    """Load swim-bout datasets through the shared logical resolver."""
    root = zarr.open(str(zarr_path), mode="r")
    selector = run_name or "latest"
    try:
        payload = load_swim_bout_tables(root, run_name=selector, speed_level=speed_level)
    except SwimBoutIOError as exc:
        if "Speed level" not in str(exc):
            raise ValueError(str(exc)) from exc
        payload = load_default_swim_bout_tables(root, run_name=selector)
        requested = SPEED_LEVEL_ALIASES.get(speed_level, f"speed_{speed_level}")
        print(
            f"Warning: Speed level '{requested}' not found, "
            f"using default '{payload.signal.speed_level or payload.signal.signal_name}'"
        )

    signal_level = payload.signal.speed_level
    is_hierarchical = bool(signal_level)
    attrs = dict(payload.run_attrs)
    attrs["signal_attrs"] = dict(payload.signal_attrs)
    attrs["run_name"] = (
        f"{payload.run_name} ({signal_level})" if is_hierarchical else payload.run_name
    )
    attrs["speed_level"] = signal_level
    attrs["speed_signal_name"] = payload.signal.signal_name
    attrs["is_hierarchical"] = is_hierarchical
    attrs["source_swim_bout_run"] = payload.run_name
    attrs["source_swim_bout_path"] = payload.level_path
    attrs["source_swim_bout_candidate_id"] = payload.candidate.candidate_id
    attrs["source_swim_bout_signal_id"] = payload.signal.signal_id
    attrs["available_speed_levels"] = [
        signal.speed_level for signal in payload.candidate.signals if signal.speed_level
    ]

    datasets: Dict[str, Optional[np.ndarray]] = {
        "bouts": payload.bouts,
        "global_metrics": _optional_table(payload.global_metrics),
        "trials": _optional_table(payload.trials),
        "bout_points": _optional_table(payload.bout_points),
        "inter_bout_intervals": _optional_table(payload.inter_bout_intervals),
        "inter_bout_interval_histogram": _optional_table(payload.inter_bout_interval_histogram),
    }
    return attrs, datasets


def _serialize_xmp(payload: Dict[str, Any]) -> str:
    """Serialize payload into an XMP packet stored in a PNG tEXt chunk."""
    json_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return (
        '<?xpacket begin="\\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:palette="https://palette.hhmi.org/ns/analysis/">\n'
        f'   <palette:swim_bout_provenance>{json_payload}</palette:swim_bout_provenance>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )


def _save_figure_with_metadata(fig: plt.Figure, output_path: Path, metadata: Dict[str, Any]) -> None:
    """Save figure to PNG with embedded provenance XMP metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    if Image is not None and PngImagePlugin is not None:
        image = Image.open(buf)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("XML:com.adobe.xmp", _serialize_xmp(metadata))
        image.save(output_path, pnginfo=pnginfo)
    else:  # pragma: no cover
        # Fall back to raw bytes without metadata
        with open(output_path, "wb") as handle:
            handle.write(buf.getvalue())


def _format_provenance_text(attrs: Dict[str, Any]) -> str:
    """Format provenance summary for figure annotation."""
    provenance = attrs.get("provenance", {})
    lines = [
        f"Run: {attrs.get('run_name', 'unknown')}",
        f"Created: {attrs.get('created_at_utc', 'unknown')}",
    ]
    git = provenance.get("git", {})
    if git:
        lines.append(f"Git: {git.get('commit_hash', git.get('short_hash', 'unknown'))}")
    args = provenance.get("arguments", {})
    if args:
        threshold = args.get("threshold_px") or args.get("threshold_mm") or args.get("threshold_bl")
        if threshold:
            lines.append(f"Speed threshold: {threshold}")
        lines.append(f"Source: {args.get('source', 'latest')}")
    detect_source = provenance.get("detect_source")
    if detect_source:
        lines.append(f"Detect run: {detect_source.get('name')}")
    if provenance.get("stimulus_run"):
        lines.append(f"Stimulus run: {provenance['stimulus_run']}")
    return "\n".join(lines)


def _prepare_point_series(points: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Split bout_points dataset into separate start/end arrays for plotting."""
    if points is None or points.size == 0:
        return {
            "start_times": np.array([]),
            "start_positions": np.array([]),
            "end_times": np.array([]),
            "end_positions": np.array([]),
        }
    point_types = np.array([pt.decode("utf-8") for pt in points["point_type"]])
    start_mask = point_types == "start"
    end_mask = point_types == "end"

    # Support both pixel (x_px, y_px) and mm (x_mm, y_mm) fields
    if 'x_px' in points.dtype.names:
        x_field, y_field = 'x_px', 'y_px'
    elif 'x_mm' in points.dtype.names:
        x_field, y_field = 'x_mm', 'y_mm'
    else:
        raise ValueError("bout_points must have either (x_px, y_px) or (x_mm, y_mm) fields")

    return {
        "start_times": points["time_s"][start_mask],
        "start_positions": np.vstack((points[x_field][start_mask], points[y_field][start_mask])) if np.any(start_mask) else np.empty((2, 0)),
        "end_times": points["time_s"][end_mask],
        "end_positions": np.vstack((points[x_field][end_mask], points[y_field][end_mask])) if np.any(end_mask) else np.empty((2, 0)),
    }


def _extract_pixel_to_mm(attrs: Dict[str, Any]) -> Optional[float]:
    """Return pixel-to-mm scale from provenance, if available."""
    provenance = attrs.get("provenance", {})
    calibration = provenance.get("calibration") or {}
    pixel_to_mm = calibration.get("pixel_to_mm")
    if pixel_to_mm and pixel_to_mm > 0:
        return float(pixel_to_mm)
    return None


def _first_existing_field(records: Optional[np.ndarray], candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate field present in a structured array."""
    if records is None or records.dtype.names is None:
        return None
    fields = set(records.dtype.names)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def _finite_field_values(records: np.ndarray, field: Optional[str]) -> np.ndarray:
    """Read finite numeric values for a possibly absent structured field."""
    if field is None:
        return np.array([], dtype=float)
    values = np.asarray(records[field], dtype=float)
    return values[np.isfinite(values)]


def create_swim_bout_dashboard(
    attrs: Dict[str, Any],
    datasets: Dict[str, Optional[np.ndarray]],
    title: Optional[str] = None,
) -> plt.Figure:
    """Create the swim bout dashboard figure."""
    global_metrics = datasets["global_metrics"]
    trials = datasets["trials"]
    bouts = datasets["bouts"]
    inter_bout_intervals = datasets.get("inter_bout_intervals")
    inter_bout_histogram = datasets.get("inter_bout_interval_histogram")
    point_series = _prepare_point_series(datasets["bout_points"])
    pixel_to_mm = _extract_pixel_to_mm(attrs)

    distance_mm_field = _first_existing_field(bouts, ("path_length_mm", "distance"))
    distance_px_field = _first_existing_field(bouts, ("path_length_px", "distance_px"))
    speed_mm_field = _first_existing_field(bouts, ("mean_speed_mm_s", "mean_speed"))
    speed_px_field = _first_existing_field(bouts, ("mean_speed_px_s",))

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[2, 1.2, 0.9],
        width_ratios=[1, 1, 1.2],
        hspace=0.4,
        wspace=0.32,
    )

    # Histograms
    ax_duration = fig.add_subplot(gs[0, 0])
    durations = bouts["duration_s"]
    durations = durations[~np.isnan(durations)]
    if durations.size:
        ax_duration.hist(durations, bins=30, color="#1f77b4", alpha=0.8)
    ax_duration.set_title("Bout Duration Distribution")
    ax_duration.set_xlabel("Duration (s)")
    ax_duration.set_ylabel("Count")

    ax_distance = fig.add_subplot(gs[0, 1])
    if distance_mm_field:
        distances = _finite_field_values(bouts, distance_mm_field)
        if distances.size:
            ax_distance.hist(distances, bins=30, color="#ff7f0e", alpha=0.8)
        ax_distance.set_title("Bout Distance Distribution")
        ax_distance.set_xlabel("Distance (mm)")
        ax_distance.set_ylabel("Count")
    else:
        distances = _finite_field_values(bouts, distance_px_field)
        if distances.size:
            ax_distance.hist(distances, bins=30, color="#ff7f0e", alpha=0.8)
        ax_distance.set_title("Bout Distance Distribution")
        ax_distance.set_xlabel("Distance (px)")
        ax_distance.set_ylabel("Count")
        if pixel_to_mm:
            def px_to_mm_func(px):
                return px * pixel_to_mm

            def mm_to_px_func(mm):
                return mm / pixel_to_mm

            sec_ax = ax_distance.secondary_xaxis("top", functions=(px_to_mm_func, mm_to_px_func))
            sec_ax.set_xlabel("Distance (mm)")

    ax_speed = fig.add_subplot(gs[0, 2])
    if speed_mm_field:
        mean_speeds = _finite_field_values(bouts, speed_mm_field)
        if mean_speeds.size:
            ax_speed.hist(mean_speeds, bins=30, color="#2ca02c", alpha=0.8)
        ax_speed.set_title("Mean Bout Speed Distribution")
        ax_speed.set_xlabel("Speed (mm/s)")
        ax_speed.set_ylabel("Count")
    else:
        mean_speeds = _finite_field_values(bouts, speed_px_field)
        if mean_speeds.size:
            ax_speed.hist(mean_speeds, bins=30, color="#2ca02c", alpha=0.8)
        ax_speed.set_title("Mean Bout Speed Distribution")
        ax_speed.set_xlabel("Speed (px/s)")
        ax_speed.set_ylabel("Count")
        if pixel_to_mm:
            def pxs_to_mms(px_per_s):
                return px_per_s * pixel_to_mm

            def mms_to_pxs(mm_per_s):
                return mm_per_s / pixel_to_mm

            sec_ax_speed = ax_speed.secondary_xaxis("top", functions=(pxs_to_mms, mms_to_pxs))
            sec_ax_speed.set_xlabel("Speed (mm/s)")

    # Timeline scatter
    ax_timeline = fig.add_subplot(gs[1, :2])
    if point_series["start_times"].size:
        ax_timeline.scatter(
            point_series["start_times"],
            point_series["start_positions"][0],
            s=12,
            alpha=0.5,
            label="Bout start",
            color="#1f77b4",
        )
    if point_series["end_times"].size:
        ax_timeline.scatter(
            point_series["end_times"],
            point_series["end_positions"][0],
            s=12,
            alpha=0.5,
            label="Bout end",
            color="#ff7f0e",
        )
    ax_timeline.set_title("Bout Start/End Positions Over Time")
    ax_timeline.set_xlabel("Time (s)")
    ax_timeline.set_ylabel("X Position (px)")
    if point_series["start_times"].size or point_series["end_times"].size:
        ax_timeline.legend(loc="upper right")
    ax_timeline.grid(alpha=0.2)
    if pixel_to_mm:
        sec_ax_timeline = ax_timeline.secondary_yaxis(
            "right",
            functions=(lambda px: px * pixel_to_mm, lambda mm: mm / pixel_to_mm),
        )
        sec_ax_timeline.set_ylabel("X Position (mm)")

    # Trial bar chart
    ax_trials = fig.add_subplot(gs[1, 2])
    if trials is not None and trials.size:
        bout_rates = trials["bout_rate_per_min"]
        active_pct = trials["percent_active"]
        indices = np.arange(trials.shape[0])
        width = 0.4
        ax_trials.bar(indices - width / 2, bout_rates, width=width, color="#1f77b4", alpha=0.8, label="Bout rate (per min)")
        ax_trials.bar(indices + width / 2, active_pct, width=width, color="#ff7f0e", alpha=0.8, label="Active time (%)")
        ax_trials.set_xticks(indices)
        ax_trials.set_xticklabels(trials["trial_id"])
        ax_trials.legend(loc="upper right", fontsize=8)
    ax_trials.set_title("Per-Trial Summary")
    ax_trials.set_xlabel("Trial ID")
    ax_trials.set_ylabel("Value")
    ax_trials.grid(alpha=0.2, axis="y")

    # Inter-bout interval histogram
    ax_intervals = fig.add_subplot(gs[2, :2])
    interval_values = None
    if inter_bout_intervals is not None and getattr(inter_bout_intervals, "size", 0):
        interval_values = inter_bout_intervals["interval_s"]
        interval_values = interval_values[np.isfinite(interval_values)]
    if interval_values is not None and interval_values.size:
        if inter_bout_histogram is not None and getattr(inter_bout_histogram, "size", 0):
            bin_left = inter_bout_histogram["bin_left_edge_s"]
            bin_right = inter_bout_histogram["bin_right_edge_s"]
            counts = inter_bout_histogram["count"]
            bin_widths = bin_right - bin_left
            bin_centers = bin_left + bin_widths / 2.0
            ax_intervals.bar(
                bin_centers,
                counts,
                width=bin_widths,
                align="center",
                color="#9467bd",
                alpha=0.8,
                edgecolor="#3b2070",
                linewidth=0.6,
            )
        else:
            ax_intervals.hist(interval_values, bins="auto", color="#9467bd", alpha=0.8)
    ax_intervals.set_title("Inter-Bout Interval Distribution")
    ax_intervals.set_xlabel("Interval (s)")
    ax_intervals.set_ylabel("Count")
    ax_intervals.grid(alpha=0.2, axis="y")

    # Inter-bout interval summary text
    ax_intervals_summary = fig.add_subplot(gs[2, 2])
    ax_intervals_summary.axis("off")
    summary_lines = ["Inter-Bout Interval Stats"]
    if global_metrics is not None and global_metrics.size:
        metrics = global_metrics[0]
        def _get_metric(name: str) -> Optional[float]:
            if getattr(metrics, "dtype", None) and name in metrics.dtype.names:
                value = metrics[name]
            else:  # pragma: no cover - defensive for dict-like records
                value = metrics.get(name) if isinstance(metrics, dict) else None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)

        count = _get_metric("inter_bout_interval_count")
        mean_val = _get_metric("inter_bout_interval_mean_s")
        std_val = _get_metric("inter_bout_interval_std_s")
        median_val = _get_metric("inter_bout_interval_median_s")
        min_val = _get_metric("inter_bout_interval_min_s")
        max_val = _get_metric("inter_bout_interval_max_s")

        if count is not None:
            summary_lines.append(f"Count: {int(count)}")
        if mean_val is not None and std_val is not None:
            summary_lines.append(f"Mean ± SD: {mean_val:.3f} ± {std_val:.3f} s")
        elif mean_val is not None:
            summary_lines.append(f"Mean: {mean_val:.3f} s")
        if median_val is not None:
            summary_lines.append(f"Median: {median_val:.3f} s")
        if min_val is not None and max_val is not None:
            summary_lines.append(f"Range: {min_val:.3f} – {max_val:.3f} s")
        if len(summary_lines) == 1:
            summary_lines.append("No inter-bout intervals available.")
    else:
        summary_lines.append("No inter-bout intervals available.")
    ax_intervals_summary.text(
        0.0,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    # Overall title and provenance text
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    else:
        fig.suptitle("Swim Bout Summary", fontsize=16, fontweight="bold", y=0.98)

    provenance_text = _format_provenance_text(attrs)
    fig.text(0.01, 0.01, provenance_text, fontsize=9, va="bottom", ha="left", family="monospace")

    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize swim bout statistics stored in analysis/swim_bout_runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--run",
        type=str,
        help="Specific swim_bout_runs/<name> to load (default: latest).",
    )
    parser.add_argument(
        "--speed-level",
        type=str,
        choices=["raw", "filtered", "smoothed", "averaged", "exponential"],
        default="smoothed",
        help="Speed level to visualize in hierarchical runs (default: smoothed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional PNG file to save the dashboard (default: auto-generated in analysis/plots).",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Override figure title.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate plot without displaying it interactively.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console messages.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    # First load to check if hierarchical
    attrs_initial, _ = _load_swim_bout_run(args.zarr_path, args.run, speed_level=args.speed_level)
    is_hierarchical = attrs_initial.get("is_hierarchical", False)

    if is_hierarchical:
        # Generate a plot for each signal level available in this run.
        speed_levels = _sort_display_speed_levels(attrs_initial.get("available_speed_levels", []))
        if not speed_levels:
            speed_levels = [args.speed_level]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for level in speed_levels:
            attrs, datasets = _load_swim_bout_run(args.zarr_path, args.run, speed_level=level)
            if not args.quiet:
                print(f"Generating plot for speed level: {level}")

            # Create title with speed level
            title = args.title if args.title else f"Swim Bout Summary - {level.capitalize()}"
            fig = create_swim_bout_dashboard(attrs, datasets, title=title)

            provenance = attrs.get("provenance", {})
            metadata = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_zarr": str(args.zarr_path),
                "run_name": attrs.get("run_name"),
                "speed_level": attrs.get("speed_level"),
                "provenance": provenance,
            }

            if args.output:
                # If user specified output, append speed level to filename
                output_stem = args.output.stem
                output_suffix = args.output.suffix
                output_path = args.output.parent / f"{output_stem}_{level}{output_suffix}"
            else:
                run_name = attrs.get("source_swim_bout_run", attrs.get("run_name", "latest"))
                output_path = DEFAULT_OUTPUT_DIR / f"swim_bout_dashboard_{run_name}_{level}_{timestamp}.png"

            _save_figure_with_metadata(fig, output_path, metadata)
            if not args.quiet:
                print(f"  Saved to {output_path}")

        if not args.no_show:
            plt.show()
    else:
        # Legacy: single plot
        attrs, datasets = _load_swim_bout_run(args.zarr_path, args.run, speed_level=args.speed_level)
        if not args.quiet:
            print(f"Loaded swim bout run '{attrs['run_name']}' from {args.zarr_path}")

        fig = create_swim_bout_dashboard(attrs, datasets, title=args.title)

        provenance = attrs.get("provenance", {})
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_zarr": str(args.zarr_path),
            "run_name": attrs.get("run_name"),
            "provenance": provenance,
        }

        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = DEFAULT_OUTPUT_DIR / f"swim_bout_dashboard_{attrs.get('run_name', 'latest')}_{timestamp}.png"

        _save_figure_with_metadata(fig, output_path, metadata)
        if not args.quiet:
            print(f"Saved dashboard to {output_path}")

        if not args.no_show:
            plt.show()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
