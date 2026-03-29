"""Inspect track kinematics runs and report their crop provenance.

This diagnostic walks the ``analysis/track_kinematics_runs`` tree and attempts to locate
the crop run referenced by each track kinematics run. It is useful when the
visualization tooling cannot resolve ROI imagery because the track kinematics run does
not record which crop archive it depends on.

Example
-------
python -m fisheye.diagnostics.check_track_kinematics_crop_links data.zarr
python -m fisheye.diagnostics.check_track_kinematics_crop_links data.zarr --run offline/track_kinematics_20250101_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import zarr
from rich.console import Console
from rich.table import Table


def _sorted_group_keys(group: zarr.Group) -> Iterable[str]:
    """Return the child group keys in sorted order."""
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else list(group.keys())
    except Exception:
        keys = list(group.keys())
    return sorted(key for key in keys if isinstance(key, str))


def _as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    """Attempt to coerce an attribute payload into a mapping."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _first_truthy_strings(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed
    return None


def _extract_provenance(attrs: zarr.attrs.Attrs) -> Dict[str, Optional[str]]:
    """Extract crop, keypoint, and detection references from run attrs."""
    crop_candidates = []
    keypoint_candidates = []
    detection_candidates = []
    notes = []

    # Direct attributes
    for attr_name in ("crop_run", "source_crop_run"):
        crop_candidates.append(attrs.get(attr_name))
    for attr_name in ("keypoint_run", "source_keypoint_run"):
        keypoint_candidates.append(attrs.get(attr_name))
    for attr_name in ("detection_run", "source_detection_run", "source_detect_run"):
        detection_candidates.append(attrs.get(attr_name))

    # Inputs block (preferred)
    inputs = _as_mapping(attrs.get("inputs"))
    if inputs:
        crop_candidates.append(inputs.get("crop_run"))
        crop_candidates.append(inputs.get("source_crop_run"))
        keypoint_candidates.append(inputs.get("keypoint_run"))
        keypoint_candidates.append(inputs.get("base_keypoint_run"))
        detection_candidates.append(inputs.get("detection_run"))
        detection_candidates.append(inputs.get("source_detect_run"))
        detection_candidates.append(inputs.get("detection_path"))
        if "tracking_metadata" in inputs:
            meta = inputs["tracking_metadata"]
            if isinstance(meta, dict):
                detection_candidates.append(meta.get("source_detect_run"))

    # Provenance payload (legacy runs may only record here)
    provenance = _as_mapping(attrs.get("provenance"))
    if provenance:
        prov_inputs = _as_mapping(provenance.get("inputs"))
        if prov_inputs:
            crop_candidates.append(prov_inputs.get("crop_run"))
            crop_candidates.append(prov_inputs.get("source_crop_run"))
            keypoint_candidates.append(prov_inputs.get("keypoint_run"))
            keypoint_candidates.append(prov_inputs.get("base_keypoint_run"))
            detection_candidates.append(prov_inputs.get("detection_run"))
            detection_candidates.append(prov_inputs.get("source_detect_run"))
            detection_candidates.append(prov_inputs.get("detection_path"))

    crop = _first_truthy_strings(crop_candidates)
    keypoint = _first_truthy_strings(keypoint_candidates)
    detection = _first_truthy_strings(detection_candidates)

    if crop is None:
        notes.append("missing crop reference")
    if detection is None:
        notes.append("missing detection reference")
    if keypoint is None:
        notes.append("missing keypoint reference")

    return {
        "crop_run": crop,
        "keypoint_run": keypoint,
        "detection_ref": detection,
        "status": ", ".join(notes) if notes else "",
    }


def _iter_track_kinematics_runs(
    movement_root: zarr.Group, run_filter: Optional[str]
) -> Iterator[Tuple[str, str, zarr.Group]]:
    for run_type in ("online", "offline"):
        type_group = movement_root.get(run_type)
        if not isinstance(type_group, zarr.Group):
            continue
        for name in _sorted_group_keys(type_group):
            full_name = f"{run_type}/{name}"
            if run_filter and run_filter not in (name, full_name):
                continue
            yield run_type, name, type_group[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive")
    parser.add_argument(
        "--run",
        help="Specific track kinematics run to inspect (accepts 'offline/<name>' or just the run name)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"{args.zarr_path} does not exist")

    root = zarr.open_group(str(args.zarr_path), mode="r")
    analysis = root.get("analysis")
    if not isinstance(analysis, zarr.Group):
        raise ValueError("Archive has no 'analysis' group; run track_kinematics first.")
    movement_root = analysis.get("track_kinematics_runs")
    if not isinstance(movement_root, zarr.Group):
        raise ValueError("Archive has no 'analysis/track_kinematics_runs' group.")

    runs = list(_iter_track_kinematics_runs(movement_root, args.run))
    if not runs:
        if args.run:
            raise ValueError(f"No track kinematics runs matched selector '{args.run}'.")
        console.print("[yellow]No track kinematics runs found under analysis/track_kinematics_runs.[/yellow]")
        return

    table = Table("Type", "Run", "Crop Run", "Keypoint Run", "Detection", "Notes")
    for run_type, run_name, run_group in runs:
        provenance = _extract_provenance(run_group.attrs)
        table.add_row(
            run_type,
            run_name,
            provenance["crop_run"] or "[red]∅[/red]",
            provenance["keypoint_run"] or "[red]∅[/red]",
            provenance["detection_ref"] or "[red]∅[/red]",
            provenance["status"],
        )

    console.print(f"[bold]Track kinematics crop provenance for[/bold] {args.zarr_path}")
    console.print(table)


if __name__ == "__main__":
    main()
