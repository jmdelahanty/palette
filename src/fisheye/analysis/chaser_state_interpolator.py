#!/usr/bin/env python3
"""Utilities for repairing stimulus metadata and chaser states inside Palette Zarr archives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from zarr.core.dtype import VariableLengthUTF8
from zarr import Array as ZarrArray, Group as ZarrGroup

from fisheye.shared.json_safety import decode_null_terminated_text


DEFAULT_TIMESTAMP_DELTA_NS = 8_333_333  # ~8.33ms for 120 Hz stimulus rendering
TIMESTAMP_FIELD_CANDIDATES = (
    "timestamp_ns",
    "timestamp_ns_session",
    "relative_timestamp_ns",
    "timestamp_ns_epoch",
)


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


@dataclass
class ChaserInterpolationReport:
    """Per-chaser summary of interpolation work."""

    chaser_index: int
    missing_frames: int
    interpolated_frames: int
    skipped_at_edges: int


@dataclass
class InterpolationOutcome:
    """Combined outcome from metadata and chaser interpolation."""

    metadata_stats: Optional[InterpolationStats]
    chaser_reports: List[ChaserInterpolationReport]
    metadata_records_total: int
    chaser_records_total: int
    new_metadata_records: int
    new_chaser_records: int


def _log(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.log(message)


def _resolve_field(names: Sequence[str], candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise KeyError(f"None of the expected fields {candidates} found in {list(names)}")
    return None


def _pick_chunks(shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Choose a reasonable chunk layout for storing data."""

    if len(shape) == 0:
        return None
    if shape[0] == 0:
        return (1,) + shape[1:]
    if len(shape) == 1:
        return (min(4096, shape[0]),)
    first_dim = min(1024, shape[0])
    if first_dim <= 0:
        return None
    return (first_dim,) + shape[1:]


def _to_string_list(data: np.ndarray) -> List[str]:
    """Convert a string-like numpy array to a list of text values."""

    strings: List[str] = []
    for value in data:
        if isinstance(value, (bytes, np.bytes_, str)):
            strings.append(decode_null_terminated_text(value, errors="ignore"))
        elif value is None:
            strings.append("")
        else:
            strings.append(str(value))
    return strings


def pick_chunks(shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Public wrapper around the chunk picker."""

    return _pick_chunks(shape)


def store_array(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
) -> zarr.Array:
    """Store numpy array (structured or standard) into Zarr, replacing any existing entry."""

    if name in parent:
        del parent[name]

    if data.dtype.names:
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
        return arr

    if data.dtype.kind in ("S", "O", "U"):
        values = _to_string_list(data)

        # Encode strings as 2D uint8 array for TensorStore compatibility
        # Determine max length needed (with reasonable upper bound)
        if values:
            max_len = min(max(len(str(v).encode('utf-8')) for v in values), 512)
            # Round up to next power of 2 for efficiency
            max_len = 2 ** (max_len - 1).bit_length()
        else:
            max_len = 128

        # Create 2D uint8 array (num_strings, max_len)
        encoded = np.zeros((len(values), max_len), dtype=np.uint8)
        for i, v in enumerate(values):
            byte_data = str(v).encode('utf-8')[:max_len]
            encoded[i, :len(byte_data)] = np.frombuffer(byte_data, dtype=np.uint8)

        # Store as 2D uint8 array - TensorStore can read this directly
        arr = parent.create_array(
            name,
            data=encoded,
            chunks=_pick_chunks(encoded.shape),
            overwrite=True,
        )
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

    return arr


def write_columnar_dataset(
    parent: zarr.Group,
    name: str,
    data: np.ndarray,
    attrs: Optional[Dict[str, object]] = None,
) -> zarr.Group:
    """Store a structured array as a columnar Zarr group."""
    if data.dtype.names is None:
        raise ValueError("write_columnar_dataset requires a structured dtype.")

    if name in parent:
        del parent[name]

    group = parent.create_group(name)
    field_names = list(data.dtype.names)
    group.attrs["storage_layout"] = "columnar"
    group.attrs["field_names"] = field_names

    # Store the original dtype string for each field to preserve exact type information
    field_dtypes = {}
    for field in field_names:
        field_dtypes[field] = str(data.dtype.fields[field][0])
    group.attrs["field_dtypes"] = field_dtypes

    if attrs:
        for attr_name, attr_value in attrs.items():
            group.attrs[attr_name] = attr_value

    for field in field_names:
        field_data = np.asarray(data[field])
        store_array(group, field, field_data, None)

    return group


def read_columnar_dataset(group: zarr.Group) -> np.ndarray:
    """Load a columnar Zarr group into a structured numpy array."""
    if not isinstance(group, ZarrGroup):
        raise TypeError("Expected a Zarr group for columnar dataset.")
    field_names = list(group.attrs.get("field_names", []))
    if not field_names:
        raise ValueError("Columnar group missing 'field_names' attribute.")

    # Get stored field dtypes if available
    field_dtypes = group.attrs.get("field_dtypes", {})

    arrays = []
    dtype = []
    for field in field_names:
        arr = group[field][:]

        # Use stored dtype if available, otherwise infer from loaded array
        if field in field_dtypes:
            field_dtype = np.dtype(field_dtypes[field])
            dtype.append((field, field_dtype))

            # Special handling for fixed-length byte strings stored as 2D uint8
            if field_dtype.kind == 'S' and arr.ndim == 2 and arr.dtype == np.uint8:
                # Decode 2D uint8 array back to fixed-length byte strings
                # Each row is a string encoded as uint8 bytes
                n_strings = arr.shape[0]
                decoded = np.empty(n_strings, dtype=field_dtype)
                for i in range(n_strings):
                    # Convert uint8 row to bytes, strip null bytes
                    byte_string = arr[i].tobytes().rstrip(b'\x00')
                    decoded[i] = byte_string
                arrays.append(decoded)
            else:
                arrays.append(arr)
        else:
            dtype.append((field, arr.dtype))
            arrays.append(arr)

    structured = np.empty(len(arrays[0]), dtype=dtype)
    for field, arr in zip(field_names, arrays):
        structured[field] = arr
    return structured


def load_structured_dataset(parent: zarr.Group, name: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load a dataset that may be stored as columnar group or structured array."""
    node = parent.get(name)
    if node is None:
        raise KeyError(f"Dataset '{name}' not found in group '{parent.path}'.")
    if isinstance(node, ZarrArray):
        return node[:], dict(node.attrs)
    if isinstance(node, ZarrGroup):
        return read_columnar_dataset(node), dict(node.attrs)
    raise TypeError(f"Unsupported node type for '{name}': {type(node)}")


def analyze_frame_gaps(
    frame_metadata: np.ndarray,
    console: Optional[Console],
) -> Optional[InterpolationStats]:
    """Inspect camera frame coverage and identify gaps."""

    _log(console, "\n[bold cyan]Analyzing frame gaps...[/bold cyan]")
    if frame_metadata.size == 0:
        _log(console, "[yellow]frame_metadata is empty; skipping interpolation.[/yellow]")
        return None

    name = _resolve_field(frame_metadata.dtype.names or (), ("triggering_camera_frame_id", "camera_frame_id"))
    camera_ids = frame_metadata[name]
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


def interpolate_metadata(
    frame_metadata: np.ndarray,
    stats: Optional[InterpolationStats],
    console: Optional[Console],
    *,
    existing_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill missing camera frames by interpolating stimulus metadata."""

    if frame_metadata.size == 0 or stats is None or stats.missing_frames == 0:
        if existing_mask is not None and existing_mask.shape[0] == frame_metadata.shape[0]:
            return frame_metadata, existing_mask
        mask = np.ones(frame_metadata.shape[0], dtype=bool)
        return frame_metadata, mask

    _log(console, "\n[bold cyan]Interpolating missing metadata frames...[/bold cyan]")

    stim_field = _resolve_field(frame_metadata.dtype.names or (), ("stimulus_frame_num", "frame_number", "stim_frame_num"))
    cam_field = _resolve_field(frame_metadata.dtype.names or (), ("triggering_camera_frame_id", "camera_frame_id"))
    ts_field = _resolve_field(frame_metadata.dtype.names or (), TIMESTAMP_FIELD_CANDIDATES)

    camera_to_stim: Dict[int, List[int]] = {}
    camera_to_ts: Dict[int, List[int]] = {}

    for record in frame_metadata:
        cam = int(record[cam_field])
        stim = int(record[stim_field])
        ts = int(record[ts_field])
        camera_to_stim.setdefault(cam, []).append(stim)
        camera_to_ts.setdefault(cam, []).append(ts)

    stim_per_camera = np.mean([len(vals) for vals in camera_to_stim.values()]) if camera_to_stim else 0.0
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
            new_record[stim_field] = base_stim + i
            new_record[cam_field] = missing_cam
            new_record[ts_field] = interp_time + i * DEFAULT_TIMESTAMP_DELTA_NS
            interpolated_records.append(new_record)

    if not interpolated_records:
        _log(console, "[green]No interpolated metadata records created (no gaps detected).[/green]")
        if existing_mask is not None and existing_mask.shape[0] == frame_metadata.shape[0]:
            return frame_metadata, existing_mask
        mask = np.ones(frame_metadata.shape[0], dtype=bool)
        return frame_metadata, mask

    combined = np.concatenate([frame_metadata, np.array(interpolated_records, dtype=dtype)])
    order = np.argsort(combined[stim_field])
    combined = combined[order]

    original_pairs: set[Tuple[int, int]] = set()
    if existing_mask is not None and existing_mask.shape[0] == frame_metadata.shape[0]:
        for rec, keep in zip(frame_metadata, existing_mask):
            if keep:
                original_pairs.add((int(rec[stim_field]), int(rec[cam_field])))
    else:
        original_pairs = {
            (int(rec[stim_field]), int(rec[cam_field]))
            for rec in frame_metadata
        }

    mask = np.array(
        [
            (int(rec[stim_field]), int(rec[cam_field])) in original_pairs
            for rec in combined
        ],
        dtype=bool,
    )

    interpolated_count = mask.size - int(mask.sum())
    stats.interpolated_frames = interpolated_count
    _log(
        console,
        f"[green]Created {interpolated_count} interpolated metadata records for {len(missing_frames)} camera frames.[/green]",
    )

    return combined, mask


def _interpolate_numeric(prev: float, next_: float, weight: float) -> float:
    if prev < 0 or next_ < 0:
        return -1.0
    return prev + (next_ - prev) * weight


def interpolate_chaser_states(
    metadata: np.ndarray,
    metadata_mask: np.ndarray,
    chaser_states: np.ndarray,
    console: Optional[Console],
    *,
    existing_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[ChaserInterpolationReport]]:
    """Fill in missing stimulus frames for chaser state logs."""

    if chaser_states.size == 0:
        _log(console, "[yellow]chaser_states is empty; skipping interpolation.[/yellow]")
        if existing_mask is not None and existing_mask.shape[0] == 0:
            return chaser_states, existing_mask, []
        return chaser_states, np.zeros(0, dtype=bool), []

    meta_stim_field = _resolve_field(metadata.dtype.names or (), ("stimulus_frame_num", "frame_number", "stim_frame_num"))
    meta_cam_field = _resolve_field(metadata.dtype.names or (), ("triggering_camera_frame_id", "camera_frame_id"))

    stim_field = _resolve_field(chaser_states.dtype.names or (), ("stimulus_frame_num", "frame_number", "stim_frame_num"))
    timestamp_field = _resolve_field(
        chaser_states.dtype.names or (),
        TIMESTAMP_FIELD_CANDIDATES,
        required=False,
    )
    camera_field = _resolve_field(
        chaser_states.dtype.names or (),
        ("camera_frame_id", "triggering_camera_frame_id"),
        required=False,
    )
    index_field = _resolve_field(
        chaser_states.dtype.names or (),
        ("chaser_index", "chaser_id"),
        required=False,
    )

    metadata_by_stim: Dict[int, np.void] = {}
    for record, keep in zip(metadata, metadata_mask):
        stim = int(record[meta_stim_field])
        # Prefer original metadata rows when duplicates exist
        if keep or stim not in metadata_by_stim:
            metadata_by_stim[stim] = record

    target_stims = sorted(metadata_by_stim.keys())

    if index_field is None:
        chaser_indices = [0]
        chaser_map: Dict[int, Dict[int, np.void]] = {0: {}}
        for record in chaser_states:
            stim = int(record[stim_field])
            chaser_map[0][stim] = record
    else:
        chaser_indices = sorted(set(int(idx) for idx in chaser_states[index_field]))
        chaser_map = {index: {} for index in chaser_indices}
        for record in chaser_states:
            index = int(record[index_field])
            stim = int(record[stim_field])
            chaser_map[index][stim] = record

    original_keys: set[Tuple[int, int]] = set()
    if existing_mask is not None and existing_mask.shape[0] == chaser_states.shape[0]:
        for record, keep in zip(chaser_states, existing_mask):
            key = (
                int(record[index_field]) if index_field is not None else 0,
                int(record[stim_field]),
            )
            if keep:
                original_keys.add(key)
    else:
        for record in chaser_states:
            key = (
                int(record[index_field]) if index_field is not None else 0,
                int(record[stim_field]),
            )
            original_keys.add(key)

    interpolated_records: List[np.void] = []
    reports: List[ChaserInterpolationReport] = []

    for index in chaser_indices:
        existing = chaser_map[index]
        existing_stims = sorted(existing.keys())
        existing_set = set(existing_stims)
        missing_for_index = [stim for stim in target_stims if stim not in existing_set]
        if not missing_for_index:
            reports.append(ChaserInterpolationReport(index, 0, 0, 0))
            continue

        added = 0
        skipped = 0

        for stim in missing_for_index:
            prev_candidates = [s for s in existing_stims if s < stim]
            next_candidates = [s for s in existing_stims if s > stim]
            if not prev_candidates or not next_candidates:
                skipped += 1
                continue

            prev_stim = prev_candidates[-1]
            next_stim = next_candidates[0]
            span = next_stim - prev_stim
            if span == 0:
                skipped += 1
                continue

            weight = (stim - prev_stim) / span

            prev_record = existing[prev_stim]
            next_record = existing[next_stim]
            new_record = np.zeros((), dtype=chaser_states.dtype)

            for field in chaser_states.dtype.names or ():
                kind = chaser_states.dtype[field].kind

                if field == stim_field:
                    new_record[field] = stim
                elif field == timestamp_field:
                    prev_val = float(prev_record[field])
                    next_val = float(next_record[field])
                    new_record[field] = int(round(prev_val + (next_val - prev_val) * weight))
                elif field == camera_field:
                    if stim in metadata_by_stim:
                        new_record[field] = metadata_by_stim[stim][meta_cam_field]
                    else:
                        prev_val = float(prev_record[field])
                        next_val = float(next_record[field])
                        new_record[field] = int(round(prev_val + (next_val - prev_val) * weight))
                elif field == index_field:
                    new_record[field] = index
                elif kind == "f":
                    prev_val = float(prev_record[field])
                    next_val = float(next_record[field])
                    new_record[field] = _interpolate_numeric(prev_val, next_val, weight)
                elif kind in ("i", "u"):
                    prev_val = int(prev_record[field])
                    next_val = int(next_record[field])
                    if prev_val == next_val:
                        new_record[field] = prev_val
                    else:
                        new_record[field] = int(round(prev_val + (next_val - prev_val) * weight))
                elif kind == "?":
                    new_record[field] = bool(prev_record[field]) or bool(next_record[field])
                else:
                    new_record[field] = prev_record[field]

            interpolated_records.append(new_record)
            added += 1

        reports.append(
            ChaserInterpolationReport(
                chaser_index=index,
                missing_frames=len(missing_for_index),
                interpolated_frames=added,
                skipped_at_edges=skipped,
            )
        )

    if not interpolated_records:
        _log(console, "[green]No chaser states required interpolation.[/green]")
        if existing_mask is not None and existing_mask.shape[0] == chaser_states.shape[0]:
            return chaser_states, existing_mask, reports
        mask = np.ones(chaser_states.shape[0], dtype=bool)
        return chaser_states, mask, reports

    combined = np.concatenate([chaser_states, np.array(interpolated_records, dtype=chaser_states.dtype)])
    order = np.argsort(combined[stim_field])
    combined = combined[order]

    mask = np.array(
        [
            (
                int(rec[index_field]) if index_field is not None else 0,
                int(rec[stim_field]),
            ) in original_keys
            for rec in combined
        ],
        dtype=bool,
    )

    new_rows = mask.size - int(mask.sum())
    _log(console, f"[green]Interpolated {new_rows} chaser state entries across {len(chaser_indices)} chaser indices.[/green]")

    return combined, mask, reports


def _interpolate_run(
    zarr_path: Path,
    run_name: str,
    *,
    update_metadata: bool,
    update_chaser: bool,
    console: Optional[Console],
    dry_run: bool,
) -> InterpolationOutcome:
    root = zarr.open(zarr_path, mode="a" if not dry_run else "r")
    if "analysis" not in root:
        raise KeyError(f"analysis group not present in {zarr_path}")
    analysis = root["analysis"]
    if "stimulus_runs" not in analysis:
        raise KeyError(f"analysis/stimulus_runs group not present in {zarr_path}")
    runs_parent = analysis["stimulus_runs"]
    if run_name not in runs_parent:
        raise KeyError(f"analysis/stimulus_runs/{run_name} not found in {zarr_path}")

    run_group = runs_parent[run_name]

    if "video_metadata" not in run_group:
        raise KeyError(
            f"Run '{run_name}' is missing the video_metadata group (expected analysis/stimulus_runs/{run_name}/video_metadata)"
        )
    meta_group = run_group["video_metadata"]
    metadata_arr, metadata_attrs = load_structured_dataset(meta_group, "frame_metadata")
    metadata_mask = run_group["interpolation_mask"][:] if "interpolation_mask" in run_group else None

    stats: Optional[InterpolationStats] = None
    combined_metadata = metadata_arr
    combined_metadata_mask = metadata_mask

    if update_metadata:
        stats = analyze_frame_gaps(metadata_arr, console)
        combined_metadata, combined_metadata_mask = interpolate_metadata(
            metadata_arr,
            stats,
            console,
            existing_mask=metadata_mask,
        )
        if not dry_run:
            metadata_attrs["interpolated"] = bool(stats.missing_frames > 0 if stats else False)
            metadata_attrs["original_records"] = int(metadata_arr.shape[0])
            metadata_attrs["total_records"] = int(combined_metadata.shape[0])
            write_columnar_dataset(meta_group, "frame_metadata", combined_metadata, metadata_attrs)
            if combined_metadata_mask is not None:
                mask_chunks = _pick_chunks(combined_metadata_mask.shape)
                arr = run_group.create_array(
                    "interpolation_mask",
                    data=combined_metadata_mask,
                    chunks=mask_chunks,
                    overwrite=True,
                )
                arr.attrs["description"] = "True for original metadata rows, False for interpolated"

    if combined_metadata_mask is None:
        combined_metadata_mask = np.ones(combined_metadata.shape[0], dtype=bool)

    if "tracking_data" not in run_group:
        raise KeyError(f"Run '{run_name}' lacks tracking_data group")
    tracking_group = run_group["tracking_data"]
    if "chaser_states" not in tracking_group:
        raise KeyError(f"Run '{run_name}' lacks tracking_data/chaser_states")

    chaser_arr, chaser_attrs = load_structured_dataset(tracking_group, "chaser_states")
    chaser_mask = (
        tracking_group["chaser_interpolation_mask"][:]
        if "chaser_interpolation_mask" in tracking_group
        else None
    )

    combined_chaser = chaser_arr
    combined_chaser_mask = chaser_mask
    reports: List[ChaserInterpolationReport] = []

    if update_chaser:
        combined_chaser, combined_chaser_mask, reports = interpolate_chaser_states(
            combined_metadata,
            combined_metadata_mask,
            chaser_arr,
            console,
            existing_mask=chaser_mask,
        )
        if not dry_run:
            chaser_attrs["interpolated"] = bool(len(reports) and any(r.interpolated_frames for r in reports))
            chaser_attrs["original_records"] = int(chaser_arr.shape[0])
            chaser_attrs["total_records"] = int(combined_chaser.shape[0])
            write_columnar_dataset(tracking_group, "chaser_states", combined_chaser, chaser_attrs)
            if combined_chaser_mask is not None:
                mask_chunks = _pick_chunks(combined_chaser_mask.shape)
                arr = tracking_group.create_array(
                    "chaser_interpolation_mask",
                    data=combined_chaser_mask,
                    chunks=mask_chunks,
                    overwrite=True,
                )
                arr.attrs["description"] = "True for original chaser rows, False for interpolated"

    if not dry_run:
        run_group.attrs["chaser_interpolation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if stats is not None:
            run_group.attrs.update(asdict(stats))

    metadata_new = int(combined_metadata.shape[0] - metadata_arr.shape[0]) if update_metadata else 0
    chaser_new = int(combined_chaser.shape[0] - chaser_arr.shape[0]) if update_chaser else 0

    return InterpolationOutcome(
        metadata_stats=stats,
        chaser_reports=reports,
        metadata_records_total=int(combined_metadata.shape[0]),
        chaser_records_total=int(combined_chaser.shape[0]),
        new_metadata_records=metadata_new,
        new_chaser_records=chaser_new,
    )


def interpolate_run(
    zarr_path: Path,
    run_name: str,
    *,
    update_metadata: bool = True,
    update_chaser: bool = True,
    verbose: bool = True,
    dry_run: bool = False,
    console: Optional[Console] = None,
) -> InterpolationOutcome:
    """
    High-level helper to repair a stimulus run inside a Palette Zarr archive.

    This function fills gaps in stimulus metadata and chaser tracking states by
    generating interpolated rows, creating **clean, contiguous datasets** suitable
    for downstream analysis tools that expect frame-by-frame data.

    Design Philosophy
    -----------------
    The goal is to transform raw experimental data (which may contain gaps due to
    dropped frames, logging issues, or sensor failures) into a **working dataset**
    where every stimulus frame has corresponding metadata and chaser positions.

    **What Gets Overwritten:**
        - ``video_metadata/frame_metadata`` - Replaced with original + interpolated rows
        - ``tracking_data/chaser_states`` - Replaced with original + interpolated rows

    **What Gets Preserved:**
        - Boolean masks identifying original vs synthetic data:
            - ``interpolation_mask`` (metadata level)
            - ``chaser_interpolation_mask`` (chaser_states level)
        - Attributes recording the interpolation:
            - ``interpolated`` (bool) - Whether interpolation occurred
            - ``original_records`` (int) - Count before interpolation
            - ``total_records`` (int) - Count after interpolation

    **Filtering Back to Original Data:**
        Downstream tools can filter to ground-truth measurements only:

        .. code-block:: python

            chaser_states = run["tracking_data"]["chaser_states"]
            mask = run["tracking_data"]["chaser_interpolation_mask"][:]
            original_only = chaser_states[mask]  # True = original H5 data

    **Re-run Safety:**
        This function is designed to be called during initial import from H5 to Zarr.
        If called on already-interpolated data, it will attempt to preserve the
        existing mask and avoid re-interpolating original rows, but this is an
        advanced use case. The typical workflow is:

        1. ``import_stimulus_to_zarr`` creates a new timestamped run from H5
        2. ``interpolate_run`` is called automatically if gaps are detected
        3. Each run is independent; no risk of compounding interpolation errors

    Parameters
    ----------
    zarr_path : Path
        Path to the Palette Zarr archive.
    run_name : str
        Name of the stimulus run under ``analysis/stimulus_runs/`` to repair.
    update_metadata : bool, default=True
        If True, interpolate stimulus metadata (frame_metadata).
    update_chaser : bool, default=True
        If True, interpolate chaser tracking states (chaser_states).
    verbose : bool, default=True
        If True, print progress and summary statistics.
    dry_run : bool, default=False
        If True, analyze and report gaps without writing changes.
    console : Optional[Console], default=None
        Rich console for logging. If None, creates one if verbose=True.

    Returns
    -------
    InterpolationOutcome
        Summary of what was interpolated (record counts, reports per chaser).

    See Also
    --------
    interpolate_metadata : Low-level metadata interpolation
    interpolate_chaser_states : Low-level chaser state interpolation
    import_stimulus_to_zarr : High-level import that calls this function automatically
    """

    local_console = console if console is not None else (Console() if verbose else None)
    outcome = _interpolate_run(
        Path(zarr_path),
        run_name,
        update_metadata=update_metadata,
        update_chaser=update_chaser,
        console=local_console,
        dry_run=dry_run,
    )

    if console is None and local_console is not None:
        summary_lines = [
            f"[bold green]Interpolation summary for {run_name}[/bold green]",
            f"Metadata: +{outcome.new_metadata_records} rows (total {outcome.metadata_records_total})",
            f"Chaser: +{outcome.new_chaser_records} rows (total {outcome.chaser_records_total})",
        ]
        if outcome.chaser_reports:
            for report in outcome.chaser_reports:
                summary_lines.append(
                    f"  Index {report.chaser_index}: filled {report.interpolated_frames} / {report.missing_frames} gaps (skipped {report.skipped_at_edges})"
                )
        for line in summary_lines:
            local_console.log(line)

    return outcome


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate stimulus metadata and chaser states inside a Palette Zarr archive.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("run_name", help="Name of the run under analysis/stimulus_runs/ to repair.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only interpolate metadata; skip chaser states.",
    )
    parser.add_argument(
        "--chaser-only",
        action="store_true",
        help="Only interpolate chaser states; keep metadata untouched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report without writing changes back to the archive.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.metadata_only and args.chaser_only:
        raise ValueError("--metadata-only and --chaser-only cannot be used together")

    outcome = interpolate_run(
        zarr_path=args.zarr_path,
        run_name=args.run_name,
        update_metadata=not args.chaser_only,
        update_chaser=not args.metadata_only,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry-run summary:")
        print(f"  New metadata rows: {outcome.new_metadata_records}")
        print(f"  New chaser rows: {outcome.new_chaser_records}")


if __name__ == "__main__":  # pragma: no cover
    main()
