#!/usr/bin/env python3
"""
Import stimulus H5 data into a Palette detection Zarr archive.

This script mirrors the functionality of ``create_analysis_h5.py`` but writes the
output directly inside the Zarr hierarchy under ``analysis/stimulus_runs``.

It copies frame metadata (with interpolation), chaser states, events, protocol
snapshots, and calibration information so downstream tooling no longer relies on
a separate analysis H5.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import zarr
from rich.console import Console

from .calibration_manager import CalibrationManager
from .chaser_state_interpolator import (
    analyze_frame_gaps,
    interpolate_metadata,
    interpolate_run,
    pick_chunks,
    store_array,
)


def _log(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.log(message)


def _compute_camera_alignment(
    metadata: np.ndarray,
    metadata_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Derive helper arrays to map between camera frame IDs and metadata indices.
    """
    camera_ids = metadata["triggering_camera_frame_id"].astype(np.int64)
    min_camera = int(camera_ids.min())
    max_camera = int(camera_ids.max())
    span = max_camera - min_camera + 1

    camera_to_index = np.full(span, -1, dtype=np.int64)
    camera_mask = np.ones(span, dtype=bool)

    for idx, cam in enumerate(camera_ids):
        offset = cam - min_camera
        if camera_to_index[offset] == -1:
            camera_to_index[offset] = idx
        camera_mask[offset] &= metadata_mask[idx]

    return {
        "camera_frame_offset": int(min_camera),
        "camera_to_metadata_index": camera_to_index,
        "camera_interpolation_mask": camera_mask,
    }


def _ensure_utf8_column(values: np.ndarray) -> np.ndarray:
    """
    Decode bytes/object string arrays to Unicode for stable UTF-8 storage.
    """
    dtype_kind = values.dtype.kind
    if dtype_kind == "S":
        decoded = np.char.decode(values, "utf-8", errors="ignore")
        return np.char.rstrip(decoded, "\x00")
    if dtype_kind == "U":
        return np.char.rstrip(values, "\x00")
    if dtype_kind == "O":
        cleaned = []
        for item in values:
            if isinstance(item, bytes):
                cleaned.append(item.decode("utf-8", errors="ignore").rstrip("\x00"))
            elif item is None:
                cleaned.append("")
            else:
                cleaned.append(str(item).rstrip("\x00"))
        return np.asarray(cleaned, dtype="U")
    return values


def _copy_h5_dataset(
    h5_group: h5py.Group,
    zarr_group: zarr.Group,
    name: str,
) -> None:
    """Copy a dataset from H5 into Zarr if present."""
    if name not in h5_group:
        return
    data = h5_group[name][:]
    attrs = {attr_name: attr_value for attr_name, attr_value in h5_group[name].attrs.items()}
    store_array(zarr_group, name, data, attrs)


def _copy_enums(h5: h5py.File, analysis_group: zarr.Group, console: Optional[Console]) -> None:
    """Copy /enums datasets from H5 into analysis/enums/events."""
    if "/enums" not in h5:
        _log(console, "[dim]/enums group not found in H5; skipping enum import.[/dim]")
        return

    enums_src = h5["/enums"]
    enums_dst = analysis_group.require_group("enums").require_group("events")

    copied = 0
    existing = set(enums_dst.array_keys())
    for leftover in existing - set(enums_src.keys()):
        del enums_dst[leftover]
    for name in enums_src.keys():
        data = enums_src[name][:]
        attrs = {attr_name: attr_value for attr_name, attr_value in enums_src[name].attrs.items()}
        store_array(enums_dst, name, data, attrs)
        copied += 1

    if copied:
        _log(console, f"[green]✓ Imported {copied} enum tables into analysis/enums/events[/green]")

def import_stimulus_to_zarr(
    stimulus_h5: Optional[Path],
    zarr_path: Path,
    *,
    run_name: Optional[str],
    overwrite: bool,
    verbose: bool,
    repair_chaser_gaps: bool = True,
) -> str:
    """Main import routine."""
    console = Console() if verbose else None

    calib_manager = CalibrationManager(str(zarr_path), verbose=verbose)

    resolved_h5: Optional[Path] = stimulus_h5
    if resolved_h5 is None:
        auto_h5 = calib_manager.find_default_h5()
        if auto_h5:
            resolved_h5 = auto_h5
            _log(console, f"[dim]Auto-detected stimulus H5: {auto_h5}[/dim]")
    if resolved_h5 is None or not resolved_h5.exists():
        raise FileNotFoundError(
            "Stimulus H5 not specified and no .h5 file found alongside the zarr. "
            "Provide one explicitly."
        )

    # Ensure calibration metadata is populated before ingesting stimulus data
    try:
        existing_calib = calib_manager.get_calibration()
        if existing_calib:
            _log(console, "[dim]Calibration already present in zarr; skipping auto-import.[/dim]")
        else:
            calib_data = calib_manager.extract_from_h5(str(resolved_h5))
            if calib_data:
                saved = calib_manager.save_calibration(calib_data, overwrite=False)
                if saved:
                    _log(console, "[green]✓ Imported calibration metadata from stimulus H5[/green]")
    except Exception as exc:
        _log(console, f"[yellow]Warning:[/yellow] Unable to import calibration automatically ({exc})")

    root = zarr.open(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    runs_parent = analysis.require_group("stimulus_runs")

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("stimulus_%Y%m%d_%H%M%S")

    if run_name in runs_parent:
        if not overwrite:
            raise ValueError(
                f"analysis/stimulus_runs/{run_name} already exists. "
                "Use --overwrite to replace the existing run."
            )
        del runs_parent[run_name]

    run_group = runs_parent.create_group(run_name)
    with h5py.File(resolved_h5, "r") as h5:
        _copy_enums(h5, analysis, console)
        if "/video_metadata/frame_metadata" not in h5:
            raise ValueError("Stimulus H5 missing /video_metadata/frame_metadata dataset.")

        frame_metadata = h5["/video_metadata/frame_metadata"][:]
        stats = analyze_frame_gaps(frame_metadata, console)
        combined_metadata, interpolation_mask = interpolate_metadata(frame_metadata, stats, console)
        alignment = _compute_camera_alignment(combined_metadata, interpolation_mask)

        meta_group = run_group.create_group("video_metadata")
        meta_attrs = {}
        if "/video_metadata/frame_metadata" in h5:
            meta_attrs.update({attr_name: attr_value for attr_name, attr_value in h5["/video_metadata/frame_metadata"].attrs.items()})
        meta_attrs["interpolated"] = bool(stats.missing_frames > 0 if stats else False)
        meta_attrs["original_records"] = int(frame_metadata.shape[0])
        meta_attrs["total_records"] = int(combined_metadata.shape[0])
        store_array(
            meta_group,
            "frame_metadata",
            combined_metadata,
            meta_attrs,
        )

        mask_chunks = pick_chunks(interpolation_mask.shape)
        run_group.create_array(
            "interpolation_mask",
            data=interpolation_mask,
            chunks=mask_chunks,
            overwrite=True,
        )

        # Alignment helpers
        align_group = run_group.create_group("frame_alignment")
        camera_offset = alignment.pop("camera_frame_offset", None)
        if camera_offset is not None:
            align_group.attrs["camera_frame_offset"] = int(camera_offset)

        for key, value in alignment.items():
            chunks = pick_chunks(value.shape)
            align_group.create_array(
                key,
                data=value,
                chunks=chunks,
                overwrite=True,
            )

        # Copy tracking data
        if "/tracking_data" in h5:
            track_group = run_group.create_group("tracking_data")
            _copy_h5_dataset(h5["/tracking_data"], track_group, "chaser_states")
            _copy_h5_dataset(h5["/tracking_data"], track_group, "bounding_boxes")

        # Events
        if "/events" in h5:
            events_data = h5["/events"][:]
            events_attrs = {attr: val for attr, val in h5["/events"].attrs.items()}

            if "events" in run_group:
                del run_group["events"]
            events_group = run_group.create_group("events")
            if events_attrs:
                events_group.attrs.update(events_attrs)

            if events_data.dtype.names:
                field_names = list(events_data.dtype.names)
                events_group.attrs["field_names"] = field_names
                events_group.attrs["storage_layout"] = "columnar"
                for field_name in field_names:
                    field_values = np.asarray(events_data[field_name])
                    if field_values.dtype.kind in ("S", "U", "O"):
                        field_values = np.asarray(_ensure_utf8_column(field_values))
                    store_array(events_group, field_name, field_values, {})
            else:
                values = np.asarray(events_data)
                if values.dtype.kind in ("S", "U", "O"):
                    values = np.asarray(_ensure_utf8_column(values))
                store_array(events_group, "values", values, {})

        # Protocol snapshot
        if "/protocol_snapshot/protocol_json" in h5:
            proto_bytes = h5["/protocol_snapshot/protocol_json"][()]
            if isinstance(proto_bytes, bytes):
                run_group.attrs["protocol_json"] = proto_bytes.decode("utf-8")
            else:
                run_group.attrs["protocol_json"] = proto_bytes

        # Calibration snapshot
        if "/calibration_snapshot/arena_config_json" in h5:
            calib_bytes = h5["/calibration_snapshot/arena_config_json"][()]
            try:
                calib_json = calib_bytes.decode("utf-8")
            except AttributeError:
                calib_json = str(calib_bytes)
            run_group.attrs["arena_config_json"] = calib_json

            try:
                arena_config = json.loads(calib_json)
            except json.JSONDecodeError:
                arena_config = {}

            coord_info = {
                "texture_dimensions": [358, 358],
                "camera_dimensions": [4512, 4512],
                "texture_to_camera_scale": 4512 / 358,
                "coordinate_note": "Chaser positions are in texture space (358x358); fish in camera space (4512x4512).",
            }
            try:
                cam_calib = arena_config.get("camera_calibrations", [{}])[0]
                width = cam_calib.get("native_width_px")
                height = cam_calib.get("native_height_px")
                if width and height:
                    coord_info["camera_dimensions"] = [int(width), int(height)]
                    coord_info["texture_to_camera_scale"] = width / 358
            except Exception:
                pass
            run_group.attrs["coordinate_transform"] = json.dumps(coord_info)

        if stats:
            run_group.attrs.update(asdict(stats))

        if repair_chaser_gaps and stats and stats.missing_frames:
            _log(console, "[cyan]Detected chaser stimulus gaps; interpolating chaser states...[/cyan]")
            interpolate_run(
                zarr_path=zarr_path,
                run_name=run_name,
                update_metadata=False,
                update_chaser=True,
                verbose=False,
                console=console,
            )

    run_group.attrs.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_h5": str(resolved_h5),
            "import_version": "1.0.0",
        }
    )

    runs_parent.attrs["latest"] = run_name
    _log(console, f"\n[bold green] Imported stimulus data to analysis/stimulus_runs/{run_name}[/bold green]")
    return run_name


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import stimulus H5 contents into a Palette detection Zarr archive.",
    )
    parser.add_argument(
        "stimulus_h5",
        type=Path,
        nargs="?",
        help="Path to the raw stimulus H5 file. If omitted, the tool will search alongside the zarr.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive to update.")
    parser.add_argument("--run-name", help="Optional run name inside analysis/stimulus_runs/.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing stimulus run with the same name.",
    )
    parser.add_argument(
        "--skip-chaser-repair",
        action="store_true",
        help="Skip the post-import chaser state interpolation step.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    import_stimulus_to_zarr(
        stimulus_h5=args.stimulus_h5,
        zarr_path=args.zarr_path,
        run_name=args.run_name,
        overwrite=args.overwrite,
        verbose=not args.quiet,
        repair_chaser_gaps=not args.skip_chaser_repair,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
