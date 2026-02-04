#!/usr/bin/env python3
"""Report Zarr storage footprint (file counts, data bytes, sharding hints)."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import zarr


META_FILENAMES = {
    ".zarray",
    ".zattrs",
    ".zgroup",
    ".zmetadata",
    "zarr.json",
}


@dataclass
class ArrayReport:
    name: str
    path: Path
    shape: Tuple[int, ...]
    chunks: Optional[Tuple[int, ...]]
    shards: Optional[Tuple[int, ...]]
    expected_chunks: int
    expected_shards: int
    data_files: int
    data_bytes: int

    @property
    def avg_file_bytes(self) -> float:
        if self.data_files <= 0:
            return 0.0
        return self.data_bytes / self.data_files


@dataclass
class ZarrReport:
    path: Path
    arrays: List[ArrayReport]
    total_files: int
    total_bytes: int

    @property
    def data_files(self) -> int:
        return sum(arr.data_files for arr in self.arrays)

    @property
    def data_bytes(self) -> int:
        return sum(arr.data_bytes for arr in self.arrays)

    @property
    def metadata_files(self) -> int:
        return max(0, self.total_files - self.data_files)

    @property
    def metadata_bytes(self) -> int:
        return max(0, self.total_bytes - self.data_bytes)

    @property
    def sharded_arrays(self) -> int:
        return sum(1 for arr in self.arrays if arr.shards)


def _is_zarr_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.endswith(".zarr"):
        return True
    if (path / "zarr.json").exists() or (path / ".zgroup").exists():
        return True
    if (path / ".zarray").exists():
        return True
    return False


def _find_zarr_roots(paths: Iterable[Path], recursive: bool) -> List[Path]:
    found: List[Path] = []
    seen = set()

    for base in paths:
        if _is_zarr_root(base):
            if base not in seen:
                found.append(base)
                seen.add(base)
            continue
        if not recursive or not base.is_dir():
            continue
        for root, dirs, _files in os.walk(base):
            root_path = Path(root)
            for d in list(dirs):
                candidate = root_path / d
                if _is_zarr_root(candidate):
                    if candidate not in seen:
                        found.append(candidate)
                        seen.add(candidate)
                    dirs.remove(d)
    return sorted(found)


def _count_files(root: Path, exclude_meta: bool) -> Tuple[int, int]:
    count = 0
    total_bytes = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if exclude_meta:
                if name in META_FILENAMES or name.startswith(".z"):
                    continue
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            count += 1
            total_bytes += stat.st_size
    return count, total_bytes


def _expected_chunks(shape: Tuple[int, ...], chunks: Optional[Tuple[int, ...]]) -> int:
    if not shape or not chunks:
        return 0
    return int(math.prod(math.ceil(s / c) for s, c in zip(shape, chunks)))


def _array_data_stats(array_path: Path) -> Tuple[int, int]:
    # Zarr v3 stores chunk/shard data under array_path/c
    v3_data_dir = array_path / "c"
    if v3_data_dir.exists():
        return _count_files(v3_data_dir, exclude_meta=False)
    # Zarr v2 stores chunks alongside metadata in the array directory.
    return _count_files(array_path, exclude_meta=True)


def _iter_arrays(root: zarr.Group) -> Iterable[Tuple[str, zarr.Array]]:
    for name, item in root.items():
        if isinstance(item, zarr.Array):
            yield name, item
        elif isinstance(item, zarr.Group):
            for sub_name, sub_item in _iter_arrays(item):
                yield f"{name}/{sub_name}", sub_item


def _build_array_report(zarr_root: Path, name: str, arr: zarr.Array) -> ArrayReport:
    array_path = zarr_root / arr.path
    chunks = tuple(arr.chunks) if getattr(arr, "chunks", None) else None
    shards = tuple(arr.shards) if getattr(arr, "shards", None) else None
    expected_chunks = _expected_chunks(arr.shape, chunks)
    expected_shards = _expected_chunks(arr.shape, shards) if shards else 0
    data_files, data_bytes = _array_data_stats(array_path)
    return ArrayReport(
        name=name,
        path=array_path,
        shape=tuple(arr.shape),
        chunks=chunks,
        shards=shards,
        expected_chunks=expected_chunks,
        expected_shards=expected_shards,
        data_files=data_files,
        data_bytes=data_bytes,
    )


def _build_report(zarr_path: Path) -> ZarrReport:
    store = zarr.storage.LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode="r")
    arrays = [_build_array_report(zarr_path, name, arr) for name, arr in _iter_arrays(root)]
    total_files, total_bytes = _count_files(zarr_path, exclude_meta=False)
    return ZarrReport(
        path=zarr_path,
        arrays=arrays,
        total_files=total_files,
        total_bytes=total_bytes,
    )


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} EB"


def _print_report(report: ZarrReport, show_arrays: bool, max_arrays: Optional[int]) -> None:
    print(f"Zarr: {report.path}")
    print(f"  arrays: {len(report.arrays)} (sharded: {report.sharded_arrays})")
    print(
        "  total files: {files} ({size})".format(
            files=report.total_files,
            size=_format_bytes(report.total_bytes),
        )
    )
    print(
        "  data files: {files} ({size})".format(
            files=report.data_files,
            size=_format_bytes(report.data_bytes),
        )
    )
    print(
        "  metadata files: {files} ({size})".format(
            files=report.metadata_files,
            size=_format_bytes(report.metadata_bytes),
        )
    )

    if not show_arrays:
        print("")
        return

    rows = report.arrays
    if max_arrays is not None:
        rows = rows[:max_arrays]

    header = (
        "  array | data_files | data_bytes | avg_file | chunks | shards | expected_chunks | expected_shards"
    )
    print(header)
    for arr in rows:
        print(
            "  {name} | {files} | {bytes} | {avg} | {chunks} | {shards} | {exp_chunks} | {exp_shards}".format(
                name=arr.name,
                files=arr.data_files,
                bytes=_format_bytes(arr.data_bytes),
                avg=_format_bytes(int(arr.avg_file_bytes)),
                chunks=arr.chunks if arr.chunks else "-",
                shards=arr.shards if arr.shards else "-",
                exp_chunks=arr.expected_chunks,
                exp_shards=arr.expected_shards,
            )
        )
    if max_arrays is not None and len(report.arrays) > max_arrays:
        print(f"  ... {len(report.arrays) - max_arrays} arrays not shown")
    print("")


def _report_to_dict(report: ZarrReport) -> Dict[str, object]:
    return {
        "path": str(report.path),
        "total_files": report.total_files,
        "total_bytes": report.total_bytes,
        "data_files": report.data_files,
        "data_bytes": report.data_bytes,
        "metadata_files": report.metadata_files,
        "metadata_bytes": report.metadata_bytes,
        "arrays": [
            {
                "name": arr.name,
                "path": str(arr.path),
                "shape": list(arr.shape),
                "chunks": list(arr.chunks) if arr.chunks else None,
                "shards": list(arr.shards) if arr.shards else None,
                "expected_chunks": arr.expected_chunks,
                "expected_shards": arr.expected_shards,
                "data_files": arr.data_files,
                "data_bytes": arr.data_bytes,
                "avg_file_bytes": arr.avg_file_bytes,
            }
            for arr in report.arrays
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report Zarr storage file counts and data footprint.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or directories to scan.")
    parser.add_argument("--recursive", action="store_true", help="Search for .zarr stores recursively.")
    parser.add_argument("--arrays", action="store_true", help="Include per-array stats.")
    parser.add_argument(
        "--max-arrays",
        type=int,
        default=None,
        help="Limit number of arrays shown in per-array output.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    zarr_roots = _find_zarr_roots(args.paths, args.recursive)
    if not zarr_roots:
        print("No Zarr stores found.")
        return 1

    reports = [_build_report(path) for path in zarr_roots]

    if args.json:
        payload = [_report_to_dict(report) for report in reports]
        print(json.dumps(payload, indent=2))
        return 0

    for report in reports:
        _print_report(report, args.arrays, args.max_arrays)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
