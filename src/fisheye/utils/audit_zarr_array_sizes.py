"""Audit Zarr array size, chunking, and read/write strategy hints.

This utility intentionally reads Zarr metadata files directly instead of
opening stores with ``zarr.open_group``. That keeps it usable in sandboxed or
remote-storage contexts where synchronous Zarr opens can hang or consolidated
metadata can be stale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_V2_METADATA_FILENAMES = {
    ".zarray",
    ".zattrs",
    ".zgroup",
    ".zmetadata",
}

_V3_METADATA_FILENAMES = {
    "zarr.json",
}

_DENSE_PIXEL_TOKENS = (
    "images_full",
    "images_ds",
    "roi_images",
    "mask_probs_roi",
    "masks_roi",
    "source_seed_masks_roi",
)

_RAGGED_TOKENS = (
    "/contours/",
    "/mask_rle/",
    "points_xy",
    "counts",
    "ptr",
    "len",
)

_GEOMETRY_TOKENS = (
    "bbox",
    "xyxy",
    "scores",
    "class_ids",
    "frame_indices",
    "frame_index",
    "frame_numbers",
    "frame_counts",
    "roi_coordinates",
    "source_crop_row_ids",
    "source_refined_row_ids",
    "source_detect_row_ids",
    "track_ids",
)

_KEYPOINT_TOKENS = (
    "keypoint",
    "points_img",
    "points_roi",
    "confidence",
    "confidences",
    "skeleton",
)

_EDITABLE_SURFACE_TOKENS = (
    "detect_runs/",
    "refined_detect_runs/",
    "keypoints_runs/",
    "refined_keypoints_runs/",
    "subject_mask_runs/",
    "refined_subject_masks_runs/",
)


@dataclass(frozen=True)
class ZarrArrayAuditRow:
    zarr_path: str
    array_path: str
    zarr_format: int
    shape: tuple[int, ...]
    dtype: str
    dtype_itemsize: int | None
    logical_bytes: int | None
    chunk_shape: tuple[int, ...] | None
    chunk_count: int | None
    chunk_logical_bytes: int | None
    shard_shape: tuple[int, ...] | None
    shard_count: int | None
    physical_layout: str
    physical_bytes: int | None
    physical_file_count: int | None
    compression_ratio_logical_to_physical: float | None
    surface_family: str
    memory_strategy: str
    write_strategy: str
    recommendation: str


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _array_path(zarr_path: Path, array_dir: Path) -> str:
    try:
        rel = array_dir.relative_to(zarr_path)
    except ValueError:
        return str(array_dir)
    return "" if str(rel) == "." else rel.as_posix()


def _coerce_int_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def _dtype_to_name(value: object) -> str:
    if isinstance(value, Mapping):
        name = value.get("name") or value.get("data_type") or value.get("dtype")
        return str(name) if name is not None else json.dumps(value, sort_keys=True)
    return str(value)


def _dtype_itemsize(dtype_name: str) -> int | None:
    lowered = dtype_name.lower()
    aliases = {
        "bool": 1,
        "bool_": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
    }
    if lowered in aliases:
        return aliases[lowered]
    try:
        return int(np.dtype(dtype_name).itemsize)
    except TypeError:
        return None


def _product(values: Sequence[int]) -> int:
    return int(math.prod(int(v) for v in values))


def _ceil_chunk_count(shape: Sequence[int], chunks: Sequence[int] | None) -> int | None:
    if not chunks:
        return None
    if len(shape) != len(chunks):
        return None
    total = 1
    for size, chunk in zip(shape, chunks):
        if chunk <= 0:
            return None
        total *= int(math.ceil(int(size) / int(chunk)))
    return total


def _logical_bytes(shape: Sequence[int], itemsize: int | None) -> int | None:
    if itemsize is None:
        return None
    return _product(shape) * int(itemsize)


def _count_physical_chunk_files(array_dir: Path, *, zarr_format: int) -> tuple[int, int]:
    if zarr_format == 3:
        data_root = array_dir / "c"
        if not data_root.exists():
            return 0, 0
        excluded = set(_V3_METADATA_FILENAMES)
    else:
        data_root = array_dir
        excluded = set(_V2_METADATA_FILENAMES)

    count = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(data_root):
        if zarr_format == 2:
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in _V2_METADATA_FILENAMES
            ]
        for filename in filenames:
            if filename in excluded:
                continue
            if filename.startswith(".z"):
                continue
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            count += 1
            total += int(stat.st_size)
    return count, total


def _extract_v3_array_metadata(payload: Mapping[str, Any]) -> tuple[
    tuple[int, ...] | None,
    str,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
]:
    shape = _coerce_int_tuple(payload.get("shape"))
    dtype = _dtype_to_name(payload.get("data_type"))
    chunk_grid = payload.get("chunk_grid")
    grid_shape = None
    if isinstance(chunk_grid, Mapping):
        configuration = chunk_grid.get("configuration")
        if isinstance(configuration, Mapping):
            grid_shape = _coerce_int_tuple(configuration.get("chunk_shape"))

    # In Zarr v3 a sharded array uses the regular chunk grid for physical
    # shards. The logical inner chunk shape lives in the sharding codec. Keep
    # those concepts separate: worker-write safety and object count depend on
    # the outer shard, while decoded over-read depends on the inner chunk.
    codecs = payload.get("codecs")
    if isinstance(codecs, Sequence) and not isinstance(codecs, (str, bytes, bytearray)):
        for codec in codecs:
            if not isinstance(codec, Mapping) or codec.get("name") != "sharding_indexed":
                continue
            configuration = codec.get("configuration")
            if isinstance(configuration, Mapping):
                inner_chunk = _coerce_int_tuple(configuration.get("chunk_shape"))
                if inner_chunk is not None:
                    return shape, dtype, inner_chunk, grid_shape
    return shape, dtype, grid_shape, None


def _extract_v2_array_metadata(payload: Mapping[str, Any]) -> tuple[
    tuple[int, ...] | None,
    str,
    tuple[int, ...] | None,
    tuple[int, ...] | None,
]:
    shape = _coerce_int_tuple(payload.get("shape"))
    dtype = _dtype_to_name(payload.get("dtype"))
    chunk_shape = _coerce_int_tuple(payload.get("chunks"))
    return shape, dtype, chunk_shape, None


def _surface_family(path: str) -> str:
    lowered = path.lower()
    if any(token in lowered for token in ("images_full", "images_ds")):
        return "video_pixels"
    if "roi_images" in lowered:
        return "roi_pixels"
    if any(token in lowered for token in ("mask_probs_roi", "masks_roi", "source_seed_masks_roi")):
        return "dense_masks"
    if any(token in lowered for token in _RAGGED_TOKENS):
        return "ragged_mask_geometry"
    if any(token in lowered for token in _KEYPOINT_TOKENS):
        return "keypoints"
    if any(token in lowered for token in _GEOMETRY_TOKENS):
        return "detection_geometry"
    if "reason" in lowered or "status" in lowered or "label" in lowered:
        return "labels_status"
    if "metric" in lowered or "qc" in lowered:
        return "metrics"
    return "unknown"


def _is_editable_surface(path: str) -> bool:
    lowered = path.lower()
    return any(token in lowered for token in _EDITABLE_SURFACE_TOKENS)


def _classify_strategy(
    *,
    path: str,
    family: str,
    logical_bytes: int | None,
    chunk_logical_bytes: int | None,
    physical_file_count: int | None,
    preload_threshold_bytes: int,
    large_chunk_threshold_bytes: int,
) -> tuple[str, str, str]:
    editable = _is_editable_surface(path)
    huge_dense = family in {"video_pixels", "roi_pixels", "dense_masks"}
    ragged = family == "ragged_mask_geometry"
    logical = logical_bytes if logical_bytes is not None else preload_threshold_bytes + 1
    chunk_bytes = chunk_logical_bytes or 0

    notes: list[str] = []
    if huge_dense:
        memory_strategy = "lazy_chunked"
        write_strategy = "chunked_surface_writes"
        notes.append("large dense pixel/mask surface; do not preload for Crimson startup")
    elif ragged:
        if logical <= preload_threshold_bytes:
            memory_strategy = "preload_index_or_small_ragged"
        else:
            memory_strategy = "lazy_ragged_indexed"
        write_strategy = "component_or_run_level_rewrite"
        notes.append("ragged geometry is seekable through ptr/len, but random single-row edits can force repacking")
    elif logical <= preload_threshold_bytes:
        memory_strategy = "preload_candidate"
        if editable:
            write_strategy = "preload_read_cache_with_row_or_overlay_writes"
            notes.append("small enough to cache for random seeks; keep edits row-granular or in review overlays")
        else:
            write_strategy = "read_only_preload_ok"
            notes.append("small read-mostly surface; full Crimson startup preload is reasonable")
    else:
        memory_strategy = "lazy_chunked"
        write_strategy = "row_chunked_or_sharded_writes" if editable else "lazy_read_only"
        notes.append("larger than preload threshold; use chunk-aware lazy reads")

    if chunk_bytes > large_chunk_threshold_bytes:
        notes.append("chunk is large for random seek; cache misses may over-read")
    if physical_file_count is not None and physical_file_count > 10000:
        notes.append("many physical chunk files; metadata/open overhead may matter on PRFS/NRS")

    return memory_strategy, write_strategy, "; ".join(notes)


def _row_from_array_metadata(
    *,
    zarr_path: Path,
    array_dir: Path,
    zarr_format: int,
    shape: tuple[int, ...],
    dtype: str,
    chunk_shape: tuple[int, ...] | None,
    shard_shape: tuple[int, ...] | None,
    collect_physical: bool,
    preload_threshold_bytes: int,
    large_chunk_threshold_bytes: int,
) -> ZarrArrayAuditRow:
    itemsize = _dtype_itemsize(dtype)
    logical = _logical_bytes(shape, itemsize)
    chunk_count = _ceil_chunk_count(shape, chunk_shape)
    chunk_logical = _logical_bytes(chunk_shape, itemsize) if chunk_shape else None
    shard_count = _ceil_chunk_count(shape, shard_shape)
    physical_file_count = None
    physical_bytes = None
    if collect_physical:
        physical_file_count, physical_bytes = _count_physical_chunk_files(
            array_dir,
            zarr_format=zarr_format,
        )
    compression_ratio = None
    if logical is not None and physical_bytes not in (None, 0):
        compression_ratio = float(logical) / float(physical_bytes)
    array_path = _array_path(zarr_path, array_dir)
    family = _surface_family(array_path)
    memory_strategy, write_strategy, recommendation = _classify_strategy(
        path=array_path,
        family=family,
        logical_bytes=logical,
        chunk_logical_bytes=chunk_logical,
        physical_file_count=physical_file_count,
        preload_threshold_bytes=preload_threshold_bytes,
        large_chunk_threshold_bytes=large_chunk_threshold_bytes,
    )
    return ZarrArrayAuditRow(
        zarr_path=str(zarr_path),
        array_path=array_path,
        zarr_format=zarr_format,
        shape=shape,
        dtype=dtype,
        dtype_itemsize=itemsize,
        logical_bytes=logical,
        chunk_shape=chunk_shape,
        chunk_count=chunk_count,
        chunk_logical_bytes=chunk_logical,
        shard_shape=shard_shape,
        shard_count=shard_count,
        physical_layout="sharded" if shard_shape is not None else "regular",
        physical_bytes=physical_bytes,
        physical_file_count=physical_file_count,
        compression_ratio_logical_to_physical=compression_ratio,
        surface_family=family,
        memory_strategy=memory_strategy,
        write_strategy=write_strategy,
        recommendation=recommendation,
    )


def iter_array_metadata_dirs(zarr_path: Path) -> Iterable[tuple[Path, int, Mapping[str, Any]]]:
    """Yield array metadata files without descending into Zarr chunk payloads."""

    for dirpath, dirnames, filenames in os.walk(zarr_path):
        root = Path(dirpath)
        # Zarr v3 chunk payload directories can be extremely large.
        dirnames[:] = [dirname for dirname in dirnames if dirname != "c"]

        if "zarr.json" in filenames:
            payload = _read_json(root / "zarr.json")
            if payload is not None and payload.get("node_type") == "array":
                yield root, 3, payload
                dirnames[:] = []
                continue
        if ".zarray" in filenames:
            payload = _read_json(root / ".zarray")
            if payload is not None:
                yield root, 2, payload
                dirnames[:] = []


def scan_zarr_array_sizes(
    zarr_path: Path,
    *,
    collect_physical: bool = False,
    include: re.Pattern[str] | None = None,
    exclude: re.Pattern[str] | None = None,
    preload_threshold_bytes: int = 256 * 1024 * 1024,
    large_chunk_threshold_bytes: int = 64 * 1024 * 1024,
) -> list[ZarrArrayAuditRow]:
    zarr_path = zarr_path.expanduser().resolve()
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr path is not a directory: {zarr_path}")

    rows: list[ZarrArrayAuditRow] = []
    for array_dir, zarr_format, payload in iter_array_metadata_dirs(zarr_path):
        if zarr_format == 3:
            shape, dtype, chunk_shape, shard_shape = _extract_v3_array_metadata(payload)
        else:
            shape, dtype, chunk_shape, shard_shape = _extract_v2_array_metadata(payload)
        if shape is None:
            continue
        array_path = _array_path(zarr_path, array_dir)
        if include is not None and include.search(array_path) is None:
            continue
        if exclude is not None and exclude.search(array_path) is not None:
            continue
        rows.append(
            _row_from_array_metadata(
                zarr_path=zarr_path,
                array_dir=array_dir,
                zarr_format=zarr_format,
                shape=shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                collect_physical=collect_physical,
                preload_threshold_bytes=preload_threshold_bytes,
                large_chunk_threshold_bytes=large_chunk_threshold_bytes,
            )
        )
    return rows


def discover_zarr_roots(paths: Iterable[Path], *, recursive: bool = False) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        candidates: list[Path] = []
        if path.is_dir() and ((path / "zarr.json").is_file() or (path / ".zgroup").is_file()):
            candidates = [path]
        elif recursive and path.is_dir():
            for dirpath, dirnames, _filenames in os.walk(path):
                root = Path(dirpath)
                for dirname in list(dirnames):
                    candidate = root / dirname
                    if dirname.endswith(".zarr") and (
                        (candidate / "zarr.json").is_file()
                        or (candidate / ".zgroup").is_file()
                    ):
                        candidates.append(candidate)
                        dirnames.remove(dirname)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found)


def _as_jsonable(row: ZarrArrayAuditRow) -> dict[str, Any]:
    payload = asdict(row)
    for key in ("shape", "chunk_shape", "shard_shape"):
        if payload[key] is not None:
            payload[key] = list(payload[key])
    return payload


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def _format_tuple(value: tuple[int, ...] | None) -> str:
    if value is None:
        return "-"
    return "x".join(str(v) for v in value)


def _sort_rows(rows: list[ZarrArrayAuditRow], sort_key: str) -> list[ZarrArrayAuditRow]:
    if sort_key == "path":
        return sorted(rows, key=lambda row: row.array_path)
    if sort_key == "physical":
        return sorted(rows, key=lambda row: (-(row.physical_bytes or 0), row.array_path))
    if sort_key == "chunk-files":
        return sorted(rows, key=lambda row: (-(row.physical_file_count or 0), row.array_path))
    if sort_key == "chunk-bytes":
        return sorted(rows, key=lambda row: (-(row.chunk_logical_bytes or 0), row.array_path))
    return sorted(rows, key=lambda row: (-(row.logical_bytes or 0), row.array_path))


def _print_table(rows: Sequence[ZarrArrayAuditRow]) -> None:
    print(
        "\t".join(
            [
                "logical",
                "physical",
                "chunk",
                "chunk_bytes",
                "chunks",
                "shard",
                "shards",
                "files",
                "family",
                "memory",
                "write",
                "path",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                [
                    _format_bytes(row.logical_bytes),
                    _format_bytes(row.physical_bytes),
                    _format_tuple(row.chunk_shape),
                    _format_bytes(row.chunk_logical_bytes),
                    str(row.chunk_count) if row.chunk_count is not None else "-",
                    _format_tuple(row.shard_shape),
                    str(row.shard_count) if row.shard_count is not None else "-",
                    str(row.physical_file_count) if row.physical_file_count is not None else "-",
                    row.surface_family,
                    row.memory_strategy,
                    row.write_strategy,
                    row.array_path,
                ]
            )
        )


def _summary(rows: Sequence[ZarrArrayAuditRow]) -> dict[str, Any]:
    logical_total = sum(row.logical_bytes or 0 for row in rows)
    physical_values = [row.physical_bytes for row in rows if row.physical_bytes is not None]
    physical_total = sum(int(value) for value in physical_values)
    sharded_array_count = sum(row.shard_shape is not None for row in rows)
    family_counts: dict[str, int] = {}
    memory_counts: dict[str, int] = {}
    write_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row.surface_family] = family_counts.get(row.surface_family, 0) + 1
        memory_counts[row.memory_strategy] = memory_counts.get(row.memory_strategy, 0) + 1
        write_counts[row.write_strategy] = write_counts.get(row.write_strategy, 0) + 1
    return {
        "array_count": len(rows),
        "logical_bytes": logical_total,
        "physical_bytes": physical_total if physical_values else None,
        "sharded_array_count": sharded_array_count,
        "surface_family_counts": dict(sorted(family_counts.items())),
        "memory_strategy_counts": dict(sorted(memory_counts.items())),
        "write_strategy_counts": dict(sorted(write_counts.items())),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Zarr array logical size, chunking, optional physical chunk bytes, "
            "and Crimson read/edit strategy hints without opening Zarr stores."
        ),
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr path(s), or roots with --recursive.")
    parser.add_argument("--recursive", action="store_true", help="Discover *.zarr stores below input roots.")
    parser.add_argument("--physical", action="store_true", help="Walk chunk files and report physical bytes/files.")
    parser.add_argument("--include", help="Only include array paths matching this regex.")
    parser.add_argument("--exclude", help="Exclude array paths matching this regex.")
    parser.add_argument("--top", type=int, default=40, help="Limit rows printed for table/json output; <=0 means all.")
    parser.add_argument(
        "--format",
        choices=("table", "json", "jsonl"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--sort",
        choices=("logical", "physical", "chunk-files", "chunk-bytes", "path"),
        default="logical",
        help="Sort key.",
    )
    parser.add_argument(
        "--preload-threshold-mib",
        type=float,
        default=256.0,
        help="Logical-size threshold for preload_candidate recommendations.",
    )
    parser.add_argument(
        "--large-chunk-threshold-mib",
        type=float,
        default=64.0,
        help="Chunk logical-size threshold for random-seek over-read warnings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    roots = discover_zarr_roots(args.paths, recursive=bool(args.recursive))
    if not roots:
        raise SystemExit("No Zarr stores found.")

    include = re.compile(args.include) if args.include else None
    exclude = re.compile(args.exclude) if args.exclude else None
    preload_threshold_bytes = int(float(args.preload_threshold_mib) * 1024 * 1024)
    large_chunk_threshold_bytes = int(float(args.large_chunk_threshold_mib) * 1024 * 1024)

    rows: list[ZarrArrayAuditRow] = []
    for root in roots:
        rows.extend(
            scan_zarr_array_sizes(
                root,
                collect_physical=bool(args.physical),
                include=include,
                exclude=exclude,
                preload_threshold_bytes=preload_threshold_bytes,
                large_chunk_threshold_bytes=large_chunk_threshold_bytes,
            )
        )
    rows = _sort_rows(rows, args.sort)
    output_rows = rows if int(args.top) <= 0 else rows[: int(args.top)]

    if args.format == "jsonl":
        for row in output_rows:
            print(json.dumps(_as_jsonable(row), sort_keys=True))
    elif args.format == "json":
        payload = {
            "summary": _summary(rows),
            "rows": [_as_jsonable(row) for row in output_rows],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = _summary(rows)
        physical = _format_bytes(summary["physical_bytes"])
        print(
            "arrays={array_count} logical={logical} physical={physical}".format(
                array_count=summary["array_count"],
                logical=_format_bytes(summary["logical_bytes"]),
                physical=physical,
            )
        )
        _print_table(output_rows)
        if int(args.top) > 0 and len(rows) > int(args.top):
            print(f"... {len(rows) - int(args.top)} arrays not shown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
