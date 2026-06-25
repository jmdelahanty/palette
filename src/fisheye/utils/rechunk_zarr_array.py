"""Rechunk one Zarr array in place with validation.

Default mode is dry-run. Use ``--apply`` to create a temporary sibling array,
copy data, validate equality, and move the temporary array over the original.

This is intentionally a narrow local-filesystem canary utility, not a whole-store
migration tool. It is meant for low-risk arrays whose chunking is clearly too
fine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr.chunk_profiles import GEOMETRY_PRELOAD_STORAGE_PROFILE_ID

DEFAULT_STORAGE_PROFILE_ID = GEOMETRY_PRELOAD_STORAGE_PROFILE_ID


@dataclass(frozen=True)
class RechunkArraySummary:
    zarr_path: str
    array_path: str
    status: str
    shape: tuple[int, ...]
    dtype: str
    old_chunks: tuple[int, ...] | None
    new_chunks: tuple[int, ...]
    old_chunk_count: int | None
    new_chunk_count: int | None
    storage_profile_id: str
    applied: bool
    temp_name: str | None = None
    reason: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array_data_type(arr: zarr.Array) -> Any:
    metadata = getattr(arr, "metadata", None)
    data_type = getattr(metadata, "data_type", None)
    return data_type if data_type is not None else arr.dtype


def _iter_chunk_slices(
    shape: Sequence[int],
    chunks: Sequence[int] | None,
) -> Iterable[tuple[slice, ...]]:
    if not chunks:
        yield tuple(slice(0, int(dim)) for dim in shape)
        return
    if len(shape) != len(chunks):
        raise ValueError(f"Chunk rank {len(chunks)} does not match shape rank {len(shape)}")
    grid = [int(math.ceil(int(dim) / int(chunk))) for dim, chunk in zip(shape, chunks)]
    for idx in np.ndindex(*grid):
        slices: list[slice] = []
        for axis, chunk_idx in enumerate(idx):
            start = int(chunk_idx) * int(chunks[axis])
            stop = min(start + int(chunks[axis]), int(shape[axis]))
            slices.append(slice(start, stop))
        yield tuple(slices)


def _chunk_count(shape: Sequence[int], chunks: Sequence[int] | None) -> int | None:
    if chunks is None:
        return None
    if len(shape) != len(chunks):
        return None
    total = 1
    for dim, chunk in zip(shape, chunks):
        if int(chunk) <= 0:
            return None
        total *= int(math.ceil(int(dim) / int(chunk)))
    return total


def _parse_chunks(text: str) -> tuple[int, ...]:
    parts = [part for part in text.replace("x", ",").split(",") if part]
    if not parts:
        raise argparse.ArgumentTypeError("chunks must be a comma- or x-separated integer tuple")
    try:
        chunks = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(chunk <= 0 for chunk in chunks):
        raise argparse.ArgumentTypeError("all chunk dimensions must be positive")
    return chunks


def _resolve_chunks(
    *,
    shape: tuple[int, ...],
    old_chunks: tuple[int, ...] | None,
    chunks: tuple[int, ...] | None,
    row_chunk: int | None,
) -> tuple[int, ...]:
    if chunks is not None and row_chunk is not None:
        raise ValueError("Specify either chunks or row_chunk, not both.")
    if chunks is not None:
        if len(chunks) != len(shape):
            raise ValueError(f"New chunks rank {len(chunks)} does not match array rank {len(shape)}")
        return tuple(min(int(chunk), max(1, int(dim))) for chunk, dim in zip(chunks, shape))
    if row_chunk is None:
        raise ValueError("Either chunks or row_chunk is required.")
    if not shape:
        raise ValueError("row_chunk cannot be used for scalar arrays.")
    if old_chunks is None:
        trailing = tuple(int(dim) for dim in shape[1:])
    else:
        trailing = tuple(int(chunk) for chunk in old_chunks[1:])
    return (min(int(row_chunk), max(1, int(shape[0]))), *trailing)


def _copy_attrs(src_attrs: Any, dest_attrs: Any) -> None:
    dest_attrs.update(dict(src_attrs))


def _create_destination_array(
    *,
    src: zarr.Array,
    parent: zarr.Group,
    temp_name: str,
    new_chunks: tuple[int, ...],
) -> zarr.Array:
    kwargs: dict[str, Any] = {
        "shape": src.shape,
        "dtype": _array_data_type(src),
        "chunks": tuple(int(v) for v in new_chunks),
        "overwrite": True,
    }
    fill_value = getattr(src, "fill_value", None)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    compressors = getattr(src, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    filters = getattr(src, "filters", None)
    if filters:
        kwargs["filters"] = filters
    serializer = getattr(src, "serializer", None)
    if serializer is not None:
        kwargs["serializer"] = serializer
    return parent.create_array(temp_name, **kwargs)


def _arrays_equal(src: zarr.Array, dest: zarr.Array) -> bool:
    src_chunks = tuple(int(v) for v in src.chunks) if getattr(src, "chunks", None) else None
    for slc in _iter_chunk_slices(tuple(int(v) for v in src.shape), src_chunks):
        left = np.asarray(src[slc])
        right = np.asarray(dest[slc])
        if left.dtype.kind == "f" or right.dtype.kind == "f":
            if not np.array_equal(left, right, equal_nan=True):
                return False
        elif not np.array_equal(left, right):
            return False
    return True


def _split_parent_array_path(array_path: str) -> tuple[str, str]:
    cleaned = array_path.strip("/")
    if not cleaned:
        raise ValueError("array_path must not be empty")
    if "/" not in cleaned:
        return "", cleaned
    parent_path, name = cleaned.rsplit("/", 1)
    return parent_path, name


def _array_dir(zarr_root_path: Path, parent_path: str, array_name: str) -> Path:
    if parent_path:
        return zarr_root_path / parent_path / array_name
    return zarr_root_path / array_name


def rechunk_zarr_array(
    zarr_path: Path | str,
    array_path: str,
    *,
    chunks: tuple[int, ...] | None = None,
    row_chunk: int | None = None,
    storage_profile_id: str = DEFAULT_STORAGE_PROFILE_ID,
    reason: str | None = None,
    apply: bool = False,
) -> RechunkArraySummary:
    zarr_root_path = Path(zarr_path).expanduser().resolve()
    mode = "a" if apply else "r"
    root = zarr.open_group(str(zarr_root_path), mode=mode, use_consolidated=False)
    parent_path, array_name = _split_parent_array_path(array_path)
    parent = root[parent_path] if parent_path else root
    src = parent[array_name]
    if not isinstance(src, zarr.Array):
        raise TypeError(f"Path is not an array: {array_path}")

    shape = tuple(int(v) for v in src.shape)
    old_chunks = tuple(int(v) for v in src.chunks) if getattr(src, "chunks", None) else None
    new_chunks = _resolve_chunks(
        shape=shape,
        old_chunks=old_chunks,
        chunks=chunks,
        row_chunk=row_chunk,
    )
    summary = RechunkArraySummary(
        zarr_path=str(zarr_root_path),
        array_path=array_path.strip("/"),
        status="planned",
        shape=shape,
        dtype=str(src.dtype),
        old_chunks=old_chunks,
        new_chunks=new_chunks,
        old_chunk_count=_chunk_count(shape, old_chunks),
        new_chunk_count=_chunk_count(shape, new_chunks),
        storage_profile_id=storage_profile_id,
        applied=False,
        reason=reason,
    )
    if old_chunks == new_chunks:
        return RechunkArraySummary(**{**asdict(summary), "status": "already_matching"})
    if not apply:
        return summary

    temp_name = f"{array_name}__rechunk_tmp_{os.getpid()}"
    if temp_name in parent:
        del parent[temp_name]
    dest = _create_destination_array(
        src=src,
        parent=parent,
        temp_name=temp_name,
        new_chunks=new_chunks,
    )
    _copy_attrs(src.attrs, dest.attrs)
    dest.attrs.update(
        {
            "storage_profile_id": storage_profile_id,
            "chunk_policy_version": storage_profile_id,
            "rechunk_provenance": {
                "tool": "fisheye.utils.rechunk_zarr_array",
                "created_at_utc": _utc_now_iso(),
                "source_array_path": array_path.strip("/"),
                "old_chunk_shape": list(old_chunks) if old_chunks is not None else None,
                "new_chunk_shape": list(new_chunks),
                "reason": reason,
            },
        }
    )

    copy_chunks = old_chunks if old_chunks is not None else new_chunks
    for slc in _iter_chunk_slices(shape, copy_chunks):
        dest[slc] = src[slc]
    if not _arrays_equal(src, dest):
        del parent[temp_name]
        raise RuntimeError(f"Validation failed after rechunk copy for {array_path}")

    target_dir = _array_dir(zarr_root_path, parent_path, array_name)
    temp_dir = _array_dir(zarr_root_path, parent_path, temp_name)
    del parent[array_name]
    if target_dir.exists():
        raise RuntimeError(f"Expected deleted array directory to be absent: {target_dir}")
    temp_dir.rename(target_dir)

    return RechunkArraySummary(
        zarr_path=str(zarr_root_path),
        array_path=array_path.strip("/"),
        status="updated",
        shape=shape,
        dtype=str(src.dtype),
        old_chunks=old_chunks,
        new_chunks=new_chunks,
        old_chunk_count=_chunk_count(shape, old_chunks),
        new_chunk_count=_chunk_count(shape, new_chunks),
        storage_profile_id=storage_profile_id,
        applied=True,
        temp_name=temp_name,
        reason=reason,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Zarr root containing the array.")
    parser.add_argument("array_path", help="Array path inside the Zarr root.")
    parser.add_argument(
        "--chunks",
        type=_parse_chunks,
        help="Full new chunk shape, e.g. 16384,1 or 16384x1.",
    )
    parser.add_argument(
        "--row-chunk",
        type=int,
        help="New first-axis chunk size. Trailing chunk dimensions are preserved.",
    )
    parser.add_argument(
        "--storage-profile-id",
        default=DEFAULT_STORAGE_PROFILE_ID,
        help="Storage/chunking profile stamp to write when applying.",
    )
    parser.add_argument("--reason", help="Human-readable reason stored in rechunk provenance.")
    parser.add_argument("--apply", action="store_true", help="Rewrite the array. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def _print_summary(summary: RechunkArraySummary) -> None:
    print(
        "{status}\tarray={array}\tshape={shape}\told_chunks={old}\tnew_chunks={new}\told_chunks_n={old_n}\tnew_chunks_n={new_n}\tapplied={applied}".format(
            status=summary.status,
            array=summary.array_path,
            shape=summary.shape,
            old=summary.old_chunks,
            new=summary.new_chunks,
            old_n=summary.old_chunk_count,
            new_n=summary.new_chunk_count,
            applied=summary.applied,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = rechunk_zarr_array(
        args.zarr_path,
        args.array_path,
        chunks=args.chunks,
        row_chunk=args.row_chunk,
        storage_profile_id=args.storage_profile_id,
        reason=args.reason,
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(asdict(summary), sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
