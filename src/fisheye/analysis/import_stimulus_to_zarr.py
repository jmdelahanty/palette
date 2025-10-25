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
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8
from rich.console import Console

from .calibration_manager import CalibrationManager


@dataclass
class InterpolationStats:
    """Summary statistics for frame interpolation."""

    total_camera_frames: int
    original_frames: int
    missing_frames: int
    interpolated_frames: int
    gap_ranges: List[Tuple[int, int]]
    largest_gap: int
    interpolation_method: str = "linear"
    timestamp: str = ""


def _log(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.log(message)


def _analyze_frame_gaps(
    frame_metadata: np.ndarray,
    console: Optional[Console],
) -> Optional[InterpolationStats]:
    """Inspect camera frame coverage and identify gaps."""

    _log(console, "\n[bold cyan]Analyzing frame gaps...[/bold cyan]")
    if frame_metadata.size == 0:
        _log(console, "[yellow]frame_metadata is empty; skipping interpolation.[/yellow]")
        return None

    camera_ids = frame_metadata["triggering_camera_frame_id"]
    unique_ids = np.unique(camera_ids)
    sorted_ids = np.sort(unique_ids)
    full_range = np.arange(sorted_ids[0], sorted_ids[-1] + 1)
    missing_frames = np.setdiff1d(full_range, sorted_ids)

    gap_ranges: List[Tuple[int, int]] = []
    if missing_frames.size:
        splits = np.where(np.diff(missing_frames) != 1)[0] + 1
        for gap in np.split(missing_frames, splits):
            if gap.size:
                gap_ranges.append((int(gap[0]), int(gap[-1])))

    stats = InterpolationStats(
        total_camera_frames=int(sorted_ids[-1] - sorted_ids[0] + 1),
        original_frames=int(unique_ids.size),
        missing_frames=int(missing_frames.size),
        interpolated_frames=0,
        gap_ranges=gap_ranges,
        largest_gap=max((end - start + 1) for start, end in gap_ranges) if gap_ranges else 0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    _log(console, f"[green]Frame range:[/green] {sorted_ids[0]} → {sorted_ids[-1]}")
    _log(console, f"[green]Original frames:[/green] {stats.original_frames}")
    if stats.missing_frames:
        _log(console, f"[yellow]Missing frames:[/yellow] {stats.missing_frames}")
        _log(console, f"[yellow]Number of gaps:[/yellow] {len(gap_ranges)}")
        _log(console, f"[yellow]Largest gap:[/yellow] {stats.largest_gap}")
    else:
        _log(console, "[green]No missing camera frames detected.[/green]")

    return stats


def _interpolate_metadata(
    frame_metadata: np.ndarray,
    stats: Optional[InterpolationStats],
    console: Optional[Console],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill missing camera frames by interpolating stimulus metadata.

    Returns combined metadata and a boolean mask (True = original record).
    """
    if frame_metadata.size == 0 or stats is None or stats.missing_frames == 0:
        mask = np.ones(frame_metadata.shape[0], dtype=bool)
        return frame_metadata, mask

    _log(console, "\n[bold cyan]Interpolating missing frames...[/bold cyan]")

    camera_to_stim: Dict[int, List[int]] = {}
    camera_to_ts: Dict[int, List[int]] = {}
    stimulus_to_camera: Dict[int, int] = {}

    for record in frame_metadata:
        cam = int(record["triggering_camera_frame_id"])
        stim = int(record["stimulus_frame_num"])
        ts = int(record["timestamp_ns"])
        camera_to_stim.setdefault(cam, []).append(stim)
        camera_to_ts.setdefault(cam, []).append(ts)
        stimulus_to_camera[stim] = cam

    stim_per_camera = np.mean([len(vals) for vals in camera_to_stim.values()])
    _log(console, f"[blue]Stimulus frames per camera frame:[/blue] {stim_per_camera:.2f}")

    camera_ids = sorted(camera_to_stim.keys())
    full_range = range(camera_ids[0], camera_ids[-1] + 1)
    missing_frames = [cam for cam in full_range if cam not in camera_to_stim]

    interpolated_records: List[np.ndarray] = []
    dtype = frame_metadata.dtype

    for missing_cam in missing_frames:
        prev_frames = [cam for cam in camera_ids if cam < missing_cam]
        next_frames = [cam for cam in camera_ids if cam > missing_cam]
        if not prev_frames or not next_frames:
            _log(console, f"[yellow]Cannot interpolate boundary frame {missing_cam}[/yellow]")
            continue
        prev_cam = max(prev_frames)
        next_cam = min(next_frames)
        gap = next_cam - prev_cam
        weight = (missing_cam - prev_cam) / gap

        prev_time = camera_to_ts[prev_cam][-1]
        next_time = camera_to_ts[next_cam][0]
        interp_time = int(prev_time + (next_time - prev_time) * weight)

        prev_max_stim = max(camera_to_stim[prev_cam])
        next_min_stim = min(camera_to_stim[next_cam])
        stim_gap = next_min_stim - prev_max_stim - 1
        stim_per_gap_frame = stim_gap / gap if gap else 0
        stim_offset = int((missing_cam - prev_cam) * stim_per_gap_frame)
        base_stim = prev_max_stim + stim_offset + 1

        num_stim = max(1, int(round(stim_per_camera)))

        for i in range(num_stim):
            new_record = np.zeros((), dtype=dtype)
            new_record["stimulus_frame_num"] = base_stim + i
            new_record["triggering_camera_frame_id"] = missing_cam
            new_record["timestamp_ns"] = interp_time + i * 8333333  # ~8.33ms for 120Hz
            interpolated_records.append(new_record)

    if interpolated_records:
        combined = np.concatenate([frame_metadata, np.array(interpolated_records, dtype=dtype)])
        order = np.argsort(combined["stimulus_frame_num"])
        combined = combined[order]

        original_pairs = {
            (int(rec["stimulus_frame_num"]), int(rec["triggering_camera_frame_id"]))
            for rec in frame_metadata
        }
        mask = np.array(
            [
                (int(rec["stimulus_frame_num"]), int(rec["triggering_camera_frame_id"])) in original_pairs
                for rec in combined
            ],
            dtype=bool,
        )
        interpolated_count = mask.size - int(mask.sum())
        stats.interpolated_frames = interpolated_count
        _log(
            console,
            f"[green]Created {interpolated_count} interpolated records for {len(missing_frames)} camera frames.[/green]",
        )
        return combined, mask

    _log(console, "[green]No interpolated records created (no gaps detected).[/green]")
    mask = np.ones(frame_metadata.shape[0], dtype=bool)
    return frame_metadata, mask


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


def _pick_chunks(shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Choose a reasonable chunk layout for storing data; returns None for empty arrays."""
    if len(shape) == 0:
        return None
    if shape[0] == 0:
        return None
    if len(shape) == 1:
        return (min(4096, shape[0]),)
    first_dim = min(1024, shape[0])
    if first_dim <= 0:
        return None
    return (first_dim,) + shape[1:]


def _to_string_list(data: np.ndarray) -> List[str]:
    """Convert string-like array to a list of UTF-8 strings."""
    strings: List[str] = []
    for value in data:
        if isinstance(value, bytes):
            strings.append(value.rstrip(b"\x00").decode("utf-8", errors="ignore"))
        elif isinstance(value, str):
            strings.append(value.rstrip("\x00"))
        elif value is None:
            strings.append("")
        else:
            strings.append(str(value))
    return strings


def _store_array(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
) -> None:
    """Store numpy array (structured or standard) into Zarr."""
    if name in parent:
        del parent[name]

    if data.dtype.names:  # structured dtype -> columnar storage
        subgroup = parent.create_group(name)
        subgroup.attrs["original_dtype"] = str(data.dtype)
        subgroup.attrs["field_names"] = list(data.dtype.names)
        field_types = {}
        for field_name in data.dtype.names:
            field_data = data[field_name]
            field_types[field_name] = str(field_data.dtype)
            if field_data.dtype.kind in ("S", "O", "U"):
                values = _to_string_list(field_data)
                shape = (len(values),)
                chunks = _pick_chunks(shape)
                arr = subgroup.create_array(
                    field_name,
                    shape=shape,
                    chunks=chunks,
                    dtype=VariableLengthUTF8(),
                    fill_value="",
                    overwrite=True,
                )
                if values:
                    arr[:] = values
            else:
                chunks = _pick_chunks(field_data.shape)
                subgroup.create_array(
                    field_name,
                    data=field_data,
                    chunks=chunks,
                    overwrite=True,
                )
        subgroup.attrs["field_dtypes"] = field_types
        if attrs:
            for attr_name, attr_value in attrs.items():
                subgroup.attrs[attr_name] = attr_value
    else:
        if data.dtype.kind in ("S", "O", "U"):
            values = _to_string_list(data)
            shape = (len(values),)
            chunks = _pick_chunks(shape)
            arr = parent.create_array(
                name,
                shape=shape,
                chunks=chunks,
                dtype=VariableLengthUTF8(),
                fill_value="",
                overwrite=True,
            )
            if values:
                arr[:] = values
        else:
            chunks = _pick_chunks(data.shape)
            arr = parent.create_array(
                name,
                data=data,
                chunks=chunks,
                overwrite=True,
            )
        if attrs:
            for attr_name, attr_value in attrs.items():
                arr.attrs[attr_name] = attr_value


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
    _store_array(zarr_group, name, data, attrs)


def import_stimulus_to_zarr(
    stimulus_h5: Optional[Path],
    zarr_path: Path,
    *,
    run_name: Optional[str],
    overwrite: bool,
    verbose: bool,
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
        if "/video_metadata/frame_metadata" not in h5:
            raise ValueError("Stimulus H5 missing /video_metadata/frame_metadata dataset.")

        frame_metadata = h5["/video_metadata/frame_metadata"][:]
        stats = _analyze_frame_gaps(frame_metadata, console)
        combined_metadata, interpolation_mask = _interpolate_metadata(frame_metadata, stats, console)
        alignment = _compute_camera_alignment(combined_metadata, interpolation_mask)

        meta_group = run_group.create_group("video_metadata")
        meta_attrs = {}
        if "/video_metadata/frame_metadata" in h5:
            meta_attrs.update({attr_name: attr_value for attr_name, attr_value in h5["/video_metadata/frame_metadata"].attrs.items()})
        meta_attrs["interpolated"] = bool(stats.missing_frames > 0 if stats else False)
        meta_attrs["original_records"] = int(frame_metadata.shape[0])
        meta_attrs["total_records"] = int(combined_metadata.shape[0])
        _store_array(
            meta_group,
            "frame_metadata",
            combined_metadata,
            meta_attrs,
        )

        mask_chunks = _pick_chunks(interpolation_mask.shape)
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
            chunks = _pick_chunks(value.shape)
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
            _store_array(run_group, "events", events_data, events_attrs)

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
    )


if __name__ == "__main__":  # pragma: no cover
    main()
