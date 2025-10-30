#!/usr/bin/env python3
"""
Visualize experiment timeline with all event categories combined on one axis.

This script mirrors the data loading and categorization logic from
`visualize_experiment_timeline.py`, but renders every event on a single
timeline instead of splitting by category.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

try:
    # When executed inside the package (e.g. python -m fisheye...).
    from fisheye.visualization.visualize_experiment_timeline import (
        EVENT_COLORS,
        categorize_event,
        diagnose_calibration_test_shape_events,
        load_enum_mappings,
        load_events,
        list_stimulus_runs,
    )
except ImportError:
    # When executed as a standalone script (e.g. python src/fisheye/.../script.py).
    from visualize_experiment_timeline import (  # type: ignore
        EVENT_COLORS,
        categorize_event,
        diagnose_calibration_test_shape_events,
        load_enum_mappings,
        load_events,
        list_stimulus_runs,
    )


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(value).rstrip("\x00")


def _assign_event_colors(event_names: list[str]) -> dict[str, str]:
    """
    Generate distinct colors for each event name. Uses the tab20 colormap first and
    falls back to evenly spaced HSV colors if there are more unique events than tab20 supports.
    """
    if not event_names:
        return {}

    base_cmap = cm.get_cmap("tab20")
    max_base = base_cmap.N

    colors: dict[str, str] = {}
    if len(event_names) <= max_base:
        for idx, name in enumerate(event_names):
            colors[name] = mcolors.to_hex(base_cmap(idx))
        return colors

    # More labels than tab20; spread colors over HSV.
    for idx, name in enumerate(event_names):
        hue = idx / len(event_names)
        colors[name] = mcolors.to_hex(mcolors.hsv_to_rgb((hue, 0.65, 0.9)))
    return colors


EVENT_HIGHLIGHTS = [
    {
        "start": "CHASER_POSITIONING_START",
        "end": "CHASER_POSITIONING_END",
        "color": "#FFD166",
        "alpha": 0.25,
        "label": "Chaser Positioning",
    },
]


def _compute_highlight_spans(
    event_records: list[dict[str, object]],
) -> list[tuple[dict[str, object], float, float]]:
    """Identify time spans between configured start/end event pairs."""
    if not EVENT_HIGHLIGHTS:
        return []

    active: dict[int, list[float]] = {idx: [] for idx in range(len(EVENT_HIGHLIGHTS))}
    spans: list[tuple[dict[str, object], float, float]] = []

    for record in event_records:
        event_type = str(record.get("event_type"))
        time_value = float(record.get("time_s", 0.0))

        for idx, highlight in enumerate(EVENT_HIGHLIGHTS):
            if event_type == highlight["start"]:
                active[idx].append(time_value)
            elif event_type == highlight["end"] and active[idx]:
                start_time = active[idx].pop(0)
                spans.append((highlight, start_time, time_value))

    return spans


def prepare_event_records(
    events_dict: dict[str, np.ndarray],
    event_type_mappings: dict[int, str],
    stimulus_mode_mappings: dict[int, str],
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    dict[str, set[str]],
    list[dict[str, object]],
    dict[str, int],
    dict[str, str],
    list[str],
]:
    """
    Build per-event records with decoded names, categories, and timestamps.
    Returns the event list along with category counts, unique event types per category,
    detailed OTHER-category entries, and per-event-type metadata for plotting.
    """
    timestamp_field = next(
        (
            field_name
            for field_name in (
                "timestamp_ns_session",
                "timestamp_ns_epoch",
                "relative_timestamp_ns",
                "timestamp_ns",
            )
            if field_name in events_dict
        ),
        None,
    )

    if timestamp_field is None:
        raise ValueError("Could not find timestamp field in events")

    event_type_field = next(
        (field_name for field_name in ("event_type_id", "event_type") if field_name in events_dict),
        None,
    )
    if event_type_field is None:
        raise ValueError("Could not find event type field in events")

    event_names_field = next(
        (field_name for field_name in ("name_or_context", "event_name") if field_name in events_dict),
        None,
    )

    timestamps = events_dict[timestamp_field]
    event_types = events_dict[event_type_field]
    event_names = events_dict[event_names_field] if event_names_field else None

    mode_ids = events_dict.get("stimulus_mode_id")
    mode_names = None
    if mode_ids is not None:
        mode_names = np.array([stimulus_mode_mappings.get(int(mid), "UNKNOWN") for mid in mode_ids])

    time_seconds = (timestamps / 1e9) - (timestamps[0] / 1e9)

    category_counts: dict[str, int] = defaultdict(int)
    category_event_names: dict[str, set[str]] = defaultdict(set)
    other_records: list[dict[str, object]] = []
    event_type_counts: dict[str, int] = defaultdict(int)
    event_type_categories: dict[str, str] = {}
    event_type_order: list[str] = []
    records: list[dict[str, object]] = []

    for idx, time in enumerate(time_seconds):
        event_type_id = int(event_types[idx])
        event_type_label = event_type_mappings.get(event_type_id, f"UNKNOWN_{event_type_id}")

        context_name: Optional[str] = None
        if event_names is not None:
            decoded_name = _decode_string(event_names[idx])
            if decoded_name:
                context_name = decoded_name

        mode_name = None
        if mode_names is not None:
            mode_name = mode_names[idx]

        category = categorize_event(event_type_label, mode_name)

        category_counts[category] += 1
        category_event_names[category].add(event_type_label)
        event_type_counts[event_type_label] += 1
        if event_type_label not in event_type_categories:
            event_type_categories[event_type_label] = category
            event_type_order.append(event_type_label)

        record = {
            "time_s": float(time),
            "category": category,
            "event_type": event_type_label,
            "event_type_id": event_type_id,
            "mode": mode_name,
            "context": context_name,
        }
        records.append(record)

        if category == "OTHER":
            other_records.append(record.copy())

    return (
        records,
        category_counts,
        category_event_names,
        other_records,
        event_type_counts,
        event_type_categories,
        event_type_order,
    )


def plot_combined_timeline(
    event_records: list[dict[str, object]],
    category_counts: dict[str, int],
    category_event_names: dict[str, set[str]],
    event_type_counts: dict[str, int],
    event_type_categories: dict[str, str],
    event_type_order: list[str],
    output_path: Optional[Path] = None,
) -> None:
    """Render all events on a single horizontal timeline, colored by category and grouped by event type."""
    if not event_records:
        raise ValueError("No events available to plot")

    times_by_event: dict[str, list[float]] = defaultdict(list)
    for record in event_records:
        times_by_event[str(record["event_type"])].append(record["time_s"])

    highlight_spans = _compute_highlight_spans(event_records)

    fig, ax = plt.subplots(figsize=(14, 4))

    baseline = 0.0
    ax.axhline(0, color="0.75", linewidth=1, linestyle="--", zorder=1)

    highlight_handles: list[Patch] = []
    seen_highlights: set[str] = set()
    for highlight, start, end in highlight_spans:
        ax.axvspan(start, end, color=highlight["color"], alpha=highlight["alpha"], zorder=0.5)
        if highlight["label"] not in seen_highlights:
            highlight_handles.append(
                Patch(
                    facecolor=highlight["color"],
                    edgecolor="none",
                    alpha=highlight["alpha"],
                    label=highlight["label"],
                )
            )
            seen_highlights.add(highlight["label"])

    legend_handles: list[Patch] = []
    seen_for_legend: set[str] = set()
    event_colors = _assign_event_colors(event_type_order)
    for event_name in event_type_order:
        event_times = times_by_event.get(event_name, [])
        if not event_times:
            continue
        color = event_colors.get(
            event_name,
            EVENT_COLORS.get(event_type_categories.get(event_name, "OTHER"), EVENT_COLORS["OTHER"]),
        )
        ymin = baseline - 0.35
        ymax = baseline + 0.35
        ax.vlines(event_times, ymin, ymax, colors=color, linewidth=2, alpha=0.7, zorder=2)
        ax.scatter(
            event_times,
            np.full(len(event_times), baseline),
            c=color,
            s=32,
            alpha=0.95,
            edgecolor="none",
            zorder=3,
            label=event_name,
        )
        if event_name not in seen_for_legend:
            legend_handles.append(
                Patch(color=color, label=f"{event_name} ({event_type_counts[event_name]})")
            )
            seen_for_legend.add(event_name)

    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel("Events", fontsize=12, fontweight="bold")
    ax.set_title("Experiment Timeline (Combined)", fontsize=14, fontweight="bold", pad=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    sorted_categories = sorted(category_counts.keys())
    category_handles = [
        Patch(color=EVENT_COLORS.get(category, EVENT_COLORS["OTHER"]), label=f"{category} ({category_counts[category]})")
        for category in sorted_categories
    ]
    legend_anchor_y = 0.98
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, title="Event Types", ncol=1)
    if category_handles:
        fig.legend(
            handles=category_handles,
            loc="upper left",
            frameon=False,
            title="Categories",
            bbox_to_anchor=(0.01, legend_anchor_y),
        )
        legend_anchor_y -= 0.08
    if highlight_handles:
        fig.legend(
            handles=highlight_handles,
            loc="upper left",
            frameon=False,
            title="Intervals",
            bbox_to_anchor=(0.01, legend_anchor_y),
        )

    all_times = [record["time_s"] for record in event_records]
    total_events = len(all_times)
    unique_event_types = len(event_type_counts)
    total_duration = max(all_times) - min(all_times) if all_times else 0.0

    fig.text(
        0.01,
        0.01,
        f"Total Events: {total_events} | Unique Event Types: {unique_event_types} | Duration: {total_duration:.2f}s",
        ha="left",
        va="bottom",
        fontsize=10,
        style="italic",
        alpha=0.75,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved combined timeline to: {output_path}")
    else:
        plt.show()


def dump_other_records(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    import json

    path.write_text(json.dumps(records, indent=2))
    print(f"Saved OTHER event details to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize experiment timeline from zarr events on a single timeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_experiment_timeline_combined.py experiment.zarr
  python visualize_experiment_timeline_combined.py experiment.zarr --run-name run_20250101_120000
  python visualize_experiment_timeline_combined.py experiment.zarr -o combined_timeline.png
        """,
    )

    parser.add_argument("zarr_path", type=Path, help="Path to the zarr file containing events")
    parser.add_argument("--run-name", type=str, help="Specific stimulus run to visualize (default: latest)")
    parser.add_argument("-o", "--output", type=Path, help="Output file path for the plot")
    parser.add_argument(
        "--dump-other-json",
        type=Path,
        help="Optional path to write OTHER-category events as JSON",
    )
    parser.add_argument(
        "--skip-calibration-diagnostic",
        action="store_true",
        help="Skip the CALIBRATION_TEST_SHAPE diagnostic report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Opening zarr file: {args.zarr_path}")
    root = zarr.open(str(args.zarr_path), mode="r")

    print("\nLoading enum mappings from HDF5...")
    enum_mappings = load_enum_mappings(root)

    event_type_mappings = enum_mappings.get("events", {})
    if not event_type_mappings:
        print("Warning: No event type mappings found. Events will show as UNKNOWN.")
    else:
        print(f"✓ Ready to decode {len(event_type_mappings)} event types")

    stimulus_mode_mappings = enum_mappings.get("stimulus_modes", {})
    if stimulus_mode_mappings:
        print(f"✓ Loaded {len(stimulus_mode_mappings)} stimulus modes")

    available_runs = list_stimulus_runs(root)
    if not available_runs:
        raise ValueError("No stimulus runs found in zarr file")

    run_name = args.run_name if args.run_name else available_runs[-1]
    if run_name not in available_runs:
        raise ValueError(f"Run '{run_name}' not found. Available: {', '.join(available_runs)}")

    print(f"\nLoading events from run: {run_name}")
    events_dict = load_events(root, run_name)

    if not args.skip_calibration_diagnostic:
        diagnose_calibration_test_shape_events(events_dict, event_type_mappings, stimulus_mode_mappings)

    (
        records,
        category_counts,
        category_event_names,
        other_records,
        event_type_counts,
        event_type_categories,
        event_type_order,
    ) = prepare_event_records(
        events_dict,
        event_type_mappings,
        stimulus_mode_mappings,
    )

    plot_combined_timeline(
        records,
        category_counts,
        category_event_names,
        event_type_counts,
        event_type_categories,
        event_type_order,
        args.output,
    )

    if args.dump_other_json:
        dump_other_records(args.dump_other_json, other_records)


if __name__ == "__main__":
    main()
