#!/usr/bin/env python3
"""Benchmark eye-angle column locality without mutating the source archive.

The source eye-angle run is opened read-only.  A bounded row prefix is copied
into disposable Zarr-v3 candidates under ``--output-root``.  Candidates differ
only in physical column order, inner chunks, and outer shards; every workload
resolves channels by the authoritative channel-name index and exact decoded
values are checked before a report is accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.analysis.eye_angle_analysis import semantic_angle_channel_order
from fisheye.shared.json_safety import decode_null_terminated_text


REPORT_SCHEMA_ID = "palette.eye_angle_column_layout_benchmark.v1"


@dataclass(frozen=True)
class LayoutCandidate:
    name: str
    semantic_order: bool
    chunks: tuple[int, int]
    shards: tuple[int, int]


CANDIDATES = (
    LayoutCandidate(
        name="current_all_columns",
        semantic_order=False,
        chunks=(8_192, -1),
        shards=(262_144, -1),
    ),
    LayoutCandidate(
        name="recommended_semantic_8",
        semantic_order=True,
        chunks=(2_048, 8),
        shards=(131_072, 32),
    ),
    LayoutCandidate(
        name="balanced_semantic_16",
        semantic_order=True,
        chunks=(4_096, 16),
        shards=(131_072, 32),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_names(run: zarr.Group, count: int) -> list[str]:
    index = run.get("angle_channel_index")
    if not isinstance(index, zarr.Group) or "name" not in index:
        raise ValueError("Eye-angle run lacks angle_channel_index/name.")
    values = np.asarray(index["name"][:])
    names = [str(decode_null_terminated_text(value)) for value in values]
    if len(names) != int(count) or len(set(names)) != len(names):
        raise ValueError("Eye-angle channel-name index is invalid or non-unique.")
    return names


def _resolve_run(root: zarr.Group, requested: str | None) -> tuple[zarr.Group, str]:
    parent = root.get("analysis/eye_angle_runs")
    if not isinstance(parent, zarr.Group):
        raise ValueError("Archive has no analysis/eye_angle_runs group.")
    run_name = str(requested or parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "")
    if not run_name or run_name not in parent:
        raise ValueError(f"Eye-angle run {requested!r} is unavailable.")
    run = parent[run_name]
    if str(run.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError("Column-layout benchmark requires a completed immutable eye-angle run.")
    if "frame_angles" not in run:
        raise ValueError("Eye-angle run has no compact frame_angles table.")
    return run, run_name


def _effective_grid(
    requested: tuple[int, int],
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int] | None = None,
) -> tuple[int, int]:
    values = (
        tuple(max(1, int(value)) for value in requested)
        if chunks is not None
        else tuple(
            min(max(1, int(value)), max(1, int(dimension)))
            for value, dimension in zip(requested, shape)
        )
    )
    if chunks is None:
        return values
    return tuple(
        int(math.ceil(value / chunk) * chunk)
        for value, chunk in zip(values, chunks)
    )


def _storage_stats(path: Path) -> dict[str, int]:
    file_count = 0
    apparent_bytes = 0
    allocated_bytes = 0
    for directory, _children, filenames in os.walk(path):
        for filename in filenames:
            stat = (Path(directory) / filename).stat()
            file_count += 1
            apparent_bytes += int(stat.st_size)
            allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
    return {
        "file_count": file_count,
        "apparent_bytes": apparent_bytes,
        "allocated_bytes": allocated_bytes,
    }


def _estimated_decoded_bytes(
    shape: tuple[int, int],
    chunks: tuple[int, int],
    *,
    row_count: int,
    column_indexes: Sequence[int],
    itemsize: int,
) -> int:
    if row_count <= 0 or not column_indexes:
        return 0
    row_chunks = range(0, min(int(row_count), shape[0]), chunks[0])
    column_chunk_starts = sorted({(int(index) // chunks[1]) * chunks[1] for index in column_indexes})
    decoded_elements = 0
    for row_start in row_chunks:
        row_width = min(chunks[0], shape[0] - row_start)
        for column_start in column_chunk_starts:
            column_width = min(chunks[1], shape[1] - column_start)
            decoded_elements += row_width * column_width
    return int(decoded_elements * int(itemsize))


def _read_columns(
    array: zarr.Array,
    *,
    rows: int,
    indexes: Sequence[int],
) -> np.ndarray:
    selection = (slice(0, int(rows)), list(int(index) for index in indexes))
    try:
        values = array.get_orthogonal_selection(selection)
    except (AttributeError, TypeError, IndexError):
        values = np.column_stack(
            [np.asarray(array[: int(rows), int(index)]) for index in indexes]
        )
    result = np.asarray(values)
    return result.reshape(int(rows), len(indexes))


def _measure_workload(
    array: zarr.Array,
    *,
    names: Sequence[str],
    selected_names: Sequence[str],
    rows: int,
    repeats: int,
) -> dict[str, Any]:
    indexes = [names.index(name) for name in selected_names]
    durations: list[float] = []
    decoded_return_bytes = 0
    for _repeat in range(max(1, int(repeats))):
        started = time.perf_counter()
        values = _read_columns(array, rows=rows, indexes=indexes)
        durations.append(float(time.perf_counter() - started))
        decoded_return_bytes = int(values.nbytes)
        del values
    shape = tuple(int(value) for value in array.shape)
    chunks = tuple(int(value) for value in array.chunks)
    return {
        "rows": int(rows),
        "channel_names": list(selected_names),
        "channel_count": len(selected_names),
        "repeats": max(1, int(repeats)),
        "seconds": durations,
        "median_seconds": float(statistics.median(durations)),
        "returned_decoded_bytes": decoded_return_bytes,
        "estimated_inner_chunk_decoded_bytes": _estimated_decoded_bytes(
            shape,
            chunks,
            row_count=rows,
            column_indexes=indexes,
            itemsize=np.dtype(array.dtype).itemsize,
        ),
    }


def _candidate_names(source_names: Sequence[str], candidate: LayoutCandidate) -> list[str]:
    if not candidate.semantic_order:
        return list(source_names)
    return semantic_angle_channel_order(source_names, block_width=candidate.chunks[1])


def _write_candidate(
    path: Path,
    *,
    source_values: np.ndarray,
    source_names: Sequence[str],
    candidate: LayoutCandidate,
) -> tuple[list[str], dict[str, Any]]:
    names = _candidate_names(source_names, candidate)
    source_indexes = [source_names.index(name) for name in names]
    values = np.ascontiguousarray(source_values[:, source_indexes])
    shape = tuple(int(value) for value in values.shape)
    requested_chunks = (
        candidate.chunks[0],
        shape[1] if candidate.chunks[1] < 0 else candidate.chunks[1],
    )
    chunks = _effective_grid(requested_chunks, shape=shape)
    requested_shards = (
        candidate.shards[0],
        shape[1] if candidate.shards[1] < 0 else candidate.shards[1],
    )
    shards = _effective_grid(requested_shards, shape=shape, chunks=chunks)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": REPORT_SCHEMA_ID,
            "benchmark_only": True,
            "channel_names": names,
            "logical_lookup": "name",
        }
    )
    started = time.perf_counter()
    root.create_array(
        "frame_angles",
        data=values,
        chunks=chunks,
        shards=shards,
        overwrite=True,
    )
    write_seconds = float(time.perf_counter() - started)
    return names, {
        "shape": list(shape),
        "chunks": list(chunks),
        "shards": list(shards),
        "write_seconds": write_seconds,
        "storage": _storage_stats(path),
    }


def run_benchmark(
    source_zarr: str | Path,
    *,
    output_root: str | Path,
    run_name: str | None = None,
    max_rows: int = 120_000,
    narrow_rows: int = 1_800,
    repeats: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_zarr).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Benchmark output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    root = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    run, resolved_run = _resolve_run(root, run_name)
    source = run["frame_angles"]
    source_names = _decode_names(run, int(source.shape[1]))
    rows = min(max(1, int(max_rows)), int(source.shape[0]))
    source_values = np.asarray(source[:rows, :])
    preferred = [
        name
        for name in (
            "left_eye_angle_deg",
            "right_eye_angle_deg",
            "vergence_eye_angle_deg",
        )
        if name in source_names
    ]
    if len(preferred) < 3:
        preferred = source_names[: min(3, len(source_names))]
    six_channel = list(preferred)
    for name in (
        "left_gaze_signed_deg",
        "right_gaze_signed_deg",
        "vergence_gaze_deg",
    ):
        if name in source_names and name not in six_channel:
            six_channel.append(name)

    candidate_reports: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_path = output_path / candidate.name
        names, layout = _write_candidate(
            candidate_path,
            source_values=source_values,
            source_names=source_names,
            candidate=candidate,
        )
        candidate_root = zarr.open_group(str(candidate_path), mode="r", use_consolidated=False)
        array = candidate_root["frame_angles"]
        restored = _read_columns(
            array,
            rows=rows,
            indexes=[names.index(name) for name in source_names],
        )
        exact = bool(np.array_equal(restored, source_values, equal_nan=True))
        del restored
        if not exact:
            raise RuntimeError(f"Decoded-value validation failed for {candidate.name}.")
        workloads = {
            "narrow_common_three": _measure_workload(
                array,
                names=names,
                selected_names=preferred,
                rows=min(rows, max(1, int(narrow_rows))),
                repeats=repeats,
            ),
            "narrow_common_six": _measure_workload(
                array,
                names=names,
                selected_names=six_channel,
                rows=min(rows, max(1, int(narrow_rows))),
                repeats=repeats,
            ),
            "full_duration_common_three": _measure_workload(
                array,
                names=names,
                selected_names=preferred,
                rows=rows,
                repeats=repeats,
            ),
            "bounded_full_table": _measure_workload(
                array,
                names=names,
                selected_names=names,
                rows=rows,
                repeats=repeats,
            ),
        }
        candidate_reports.append(
            {
                "candidate": asdict(candidate),
                "layout": layout,
                "exact_values_by_name": exact,
                "workloads": workloads,
            }
        )

    report = {
        "schema_id": REPORT_SCHEMA_ID,
        "created_at_utc": _utc_now(),
        "source_zarr": str(source_path),
        "source_access": "read_only",
        "eye_angle_run": resolved_run,
        "source_shape": [int(value) for value in source.shape],
        "benchmarked_rows": rows,
        "source_channel_count": len(source_names),
        "candidate_output_root": str(output_path),
        "timing_scope": "warm_local_candidate_reads; use estimated decoded bytes for layout comparison",
        "candidates": candidate_reports,
    }
    (output_path / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--max-rows", type=int, default=120_000)
    parser.add_argument("--narrow-rows", type=int, default=1_800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_benchmark(
        args.source_zarr,
        output_root=args.output_root,
        run_name=args.run_name,
        max_rows=args.max_rows,
        narrow_rows=args.narrow_rows,
        repeats=args.repeats,
        overwrite=args.overwrite,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(args.output_root.expanduser().resolve() / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
