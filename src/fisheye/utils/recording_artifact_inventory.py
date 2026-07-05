#!/usr/bin/env python3
"""Inventory per-recording Palette Zarr runs, artifacts, and sidecar mirrors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.recording_artifact_inventory import build_recording_artifact_inventory
from fisheye.shared.zarr_io import open_zarr_root


def _print_text_summary(inventory: dict[str, object]) -> None:
    print(f"Zarr: {inventory.get('zarr_path') or '(open root)'}")
    print(f"Use: {inventory.get('zarr_use') or 'unknown'}")
    root_attrs = inventory.get("root_attrs")
    if isinstance(root_attrs, dict):
        recording_id = root_attrs.get("recording_id") or root_attrs.get("recording_name")
        if recording_id:
            print(f"Recording: {recording_id}")
    print(f"Run families: {inventory.get('run_family_count', 0)}")
    print(f"Runs: {inventory.get('run_count', 0)}")
    print(f"Visualization artifacts: {inventory.get('visualization_artifact_count', 0)}")

    acquisition = inventory.get("acquisition_video_streams")
    if isinstance(acquisition, dict) and acquisition.get("available"):
        print(
            "Acquisition streams: "
            f"{acquisition.get('stream_count', 0)} "
            f"status={acquisition.get('inventory_status') or 'unknown'}"
        )

    for section_name, label in (
        ("root_run_families", "Root"),
        ("analysis_run_families", "Analysis"),
        ("nested_report_families", "Nested"),
    ):
        families = inventory.get(section_name)
        if not isinstance(families, list) or not families:
            continue
        print(f"\n{label} run families:")
        for family in families:
            if not isinstance(family, dict):
                continue
            parent = family.get("parent") if isinstance(family.get("parent"), dict) else {}
            resolved = parent.get("resolved_authoritative") or parent.get("resolved_latest_complete") or "-"
            print(
                f"  {family.get('run_parent_path')}: "
                f"{family.get('run_count', 0)} runs, "
                f"resolved={resolved}"
            )
            for run in family.get("runs", []) if isinstance(family.get("runs"), list) else []:
                if not isinstance(run, dict):
                    continue
                print(
                    f"    - {run.get('name')} "
                    f"status={run.get('completion_status')} "
                    f"visualizations={run.get('visualization_count', 0)}"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Recording analysis/training Zarr to inventory.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON inventory.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation when --json is set.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = open_zarr_root(args.zarr_path, mode="r")
    inventory = build_recording_artifact_inventory(root, zarr_path=args.zarr_path)
    if args.json:
        print(json.dumps(inventory, indent=args.indent, sort_keys=True))
    else:
        _print_text_summary(inventory)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
