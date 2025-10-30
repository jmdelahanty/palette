#!/usr/bin/env python3
"""
Visualize experiment events directly from a stimulus H5 file on a single timeline.

This is the H5 counterpart to ``visualize_experiment_timeline_combined.py`` and
avoids importing events into a Zarr archive first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from fisheye.visualization.visualize_experiment_timeline import (
    diagnose_calibration_test_shape_events,
)
from fisheye.visualization.visualize_experiment_timeline_combined import (
    dump_other_records,
    prepare_event_records,
    plot_combined_timeline,
)
from fisheye.utils.patch_legacy_h5 import (  # Provides fallback enum values.
    EVENT_TYPE_MAPPINGS as DEFAULT_EVENT_TYPES,
    STIMULUS_MODE_MAPPINGS as DEFAULT_STIMULUS_MODES,
)


def _decode_bytes(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(value).rstrip("\x00")


def load_enum_mappings_h5(h5: h5py.File) -> Dict[str, Dict[int, str]]:
    """Read enum mapping tables from the H5 /enums group."""
    mappings: Dict[str, Dict[int, str]] = {}

    enums_group = h5.get("enums")
    if enums_group is None:
        print("Warning: No /enums group found in H5 file. Using empty mappings.")
        return mappings

    for name, dataset in enums_group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        data = dataset[:]
        if data.size == 0:
            mappings[name] = {}
            continue

        mapping_dict: Dict[int, str] = {}
        for record in data:
            enum_id = int(record["id"])
            enum_name_raw = record["name"]
            mapping_dict[enum_id] = _decode_bytes(enum_name_raw)

        mappings[name] = mapping_dict
        print(f"✓ Loaded {len(mapping_dict)} {name} mappings from H5")

    return mappings


def resolve_events_dataset(h5: h5py.File, run_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Determine the dataset path containing events.

    Returns the dataset path (relative to root) and the resolved run name if applicable.
    """
    if run_name:
        candidate = f"analysis/stimulus_runs/{run_name}/events"
        if candidate in h5:
            return candidate, run_name
        if run_name in h5:
            candidate = f"{run_name}/events"
            if candidate in h5:
                return candidate, run_name
        raise ValueError(f"Run '{run_name}' not found in H5 file.")

    if "events" in h5:
        return "events", None

    runs_group = h5.get("analysis/stimulus_runs")
    if runs_group is not None:
        available_runs = [
            name
            for name in runs_group.keys()
            if isinstance(runs_group[name], h5py.Group) and "events" in runs_group[name]
        ]
        if not available_runs:
            raise ValueError("No events dataset found under analysis/stimulus_runs in H5.")

        latest = runs_group.attrs.get("latest")
        if isinstance(latest, bytes):
            latest = latest.decode("utf-8", errors="ignore").rstrip("\x00")
        if isinstance(latest, str) and latest in available_runs:
            return f"analysis/stimulus_runs/{latest}/events", latest

        available_runs.sort()
        selected = available_runs[-1]
        return f"analysis/stimulus_runs/{selected}/events", selected

    raise ValueError("No events dataset located in H5 file.")


def load_events_dataset(h5: h5py.File, dataset_path: str) -> Dict[str, np.ndarray]:
    """Load structured events dataset and convert to a dict of numpy arrays."""
    dataset = h5[dataset_path]
    data = dataset[:]

    if data.dtype.names:
        return {field: data[field] for field in data.dtype.names}

    raise TypeError(f"Unsupported events dataset dtype at '{dataset_path}'. Expected structured array.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stimulus events timeline directly from an H5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_experiment_timeline_combined_h5.py path/to/stimulus.h5
  python visualize_experiment_timeline_combined_h5.py path/to/stimulus.h5 --run-name run_20250101_120000
  python visualize_experiment_timeline_combined_h5.py stimulus.h5 -o combined_timeline.png
        """,
    )
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus H5 file containing events.")
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name inside analysis/stimulus_runs (if the H5 stores multiple runs).",
    )
    parser.add_argument("-o", "--output", type=Path, help="Optional output path for the plot.")
    parser.add_argument(
        "--dump-other-json",
        type=Path,
        help="Optional path to write OTHER-category events as JSON.",
    )
    parser.add_argument(
        "--skip-calibration-diagnostic",
        action="store_true",
        help="Skip the CALIBRATION_TEST_SHAPE diagnostic report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with h5py.File(args.h5_path, "r") as h5:
        print(f"Opened H5 file: {args.h5_path}")

        enum_mappings = load_enum_mappings_h5(h5)
        event_type_mappings = {**DEFAULT_EVENT_TYPES, **enum_mappings.get("events", {})}
        print(f"✓ Ready to decode {len(event_type_mappings)} event types (includes fallbacks)")

        stimulus_mode_mappings = {**DEFAULT_STIMULUS_MODES, **enum_mappings.get("stimulus_modes", {})}
        print(f"✓ Loaded {len(stimulus_mode_mappings)} stimulus modes (includes fallbacks)")

        dataset_path, resolved_run = resolve_events_dataset(h5, args.run_name)
        if resolved_run:
            print(f"Using events dataset from run: {resolved_run}")
        else:
            print(f"Using events dataset at root path: {dataset_path}")

        events_dict = load_events_dataset(h5, dataset_path)

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
