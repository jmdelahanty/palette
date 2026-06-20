"""Benchmark compact RLE storage for existing dense binary mask arrays.

This diagnostic is read-only by default. It scans ``masks_roi`` arrays, encodes
selected rows with Palette's exact typed-array RLE contract, and reports dense
versus compact logical bytes. With ``--write-temp-zarr`` it also writes a
temporary ``mask_rle`` Zarr group to measure physical compressed size.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import zarr

from ..shared.mask_rle import (
    EncodedMaskComponentStack,
    EncodedMaskStack,
    component_stack_from_flat_rle,
    concatenate_encoded_mask_stacks,
    decode_binary_mask_rle,
    encode_mask_stack_rle,
)
from ..shared.mask_store import write_encoded_component_rle_mask_store

DEFAULT_FAMILIES = (
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
)


@dataclass(frozen=True)
class MaskRleBenchmarkResult:
    archive: str
    family: str
    run: str
    run_path: str
    source_array: str
    status: str
    shape: tuple[int, ...] | None
    dtype: str | None
    chunks: tuple[int, ...] | None
    dense_logical_bytes: int | None
    dense_physical_bytes: int | None
    sampled_rows: int
    total_rows: int | None
    channel_count: int | None
    mask_shape_hw: tuple[int, int] | None
    sample_is_exact: bool
    rle_counts_count: int | None
    rle_layout: str | None
    rle_logical_bytes: int | None
    rle_estimated_total_logical_bytes: int | None
    rle_physical_bytes: int | None
    rle_temp_zarr_path: str | None
    dense_to_rle_logical_ratio: float | None
    dense_to_rle_physical_ratio: float | None
    encode_seconds: float | None
    encode_rows_per_second: float | None
    decode_benchmark_rows: int
    decode_seconds: float | None
    decode_rows_per_second: float | None
    encode_workers: int
    encode_backend: str
    notes: tuple[str, ...]


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _array_stored_bytes(array: Any) -> int | None:
    stored = getattr(array, "nbytes_stored", None)
    if callable(stored):
        try:
            stored = stored()
        except TypeError:
            stored = None
    if stored is None or isinstance(stored, str):
        try:
            stored = int(getattr(array, "nbytes", 0))
        except Exception:
            return None
    try:
        return int(stored)
    except Exception:
        return None


def _group_stored_bytes(group: Any) -> int:
    total = 0
    for key in group.array_keys():
        stored = _array_stored_bytes(group[key])
        if stored is not None:
            total += int(stored)
    for key in group.group_keys():
        total += _group_stored_bytes(group[key])
    return int(total)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) == 3:
        n_rows, height, width = dims
        return (n_rows, 1, height, width)
    if len(dims) == 4:
        n_rows, n_channels, height, width = dims
        return (n_rows, n_channels, height, width)
    raise ValueError(f"Expected mask array with shape (N,H,W) or (N,C,H,W), got {dims}.")


def _component_names(run_group: Any, channel_count: int) -> tuple[str, ...]:
    attrs = dict(getattr(run_group, "attrs", {}) or {})
    for key in ("mask_labels", "component_names", "labels"):
        value = attrs.get(key)
        if isinstance(value, (list, tuple)) and len(value) == int(channel_count):
            return tuple(str(item) for item in value)
    return tuple(f"component_{idx}" for idx in range(int(channel_count)))


def _parse_sample_rows(value: str, total_rows: int) -> np.ndarray:
    text = str(value).strip().lower()
    if text == "all":
        return np.arange(int(total_rows), dtype=np.int64)
    if not text:
        raise ValueError("--sample-rows must be 'all', an integer, or a fraction in (0, 1].")
    if "." in text:
        fraction = float(text)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"Sample fraction must be in (0, 1], got {value!r}.")
        count = max(1, int(math.ceil(float(total_rows) * fraction)))
    else:
        count = int(text)
        if count <= 0:
            raise ValueError(f"Sample row count must be positive, got {value!r}.")
    if count >= total_rows:
        return np.arange(int(total_rows), dtype=np.int64)
    return np.linspace(0, int(total_rows) - 1, num=count, dtype=np.int64)


def _iter_contiguous_batches(rows: np.ndarray, *, max_batch_rows: int) -> Iterable[np.ndarray]:
    if rows.size == 0:
        return
    start = 0
    while start < rows.size:
        stop = start + 1
        while stop < rows.size and int(rows[stop]) == int(rows[stop - 1]) + 1 and (stop - start) < max_batch_rows:
            stop += 1
        yield rows[start:stop]
        start = stop


def _read_rows(array: Any, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0,), dtype=np.uint8)
    if rows.size == 1:
        return np.asarray(array[int(rows[0]) : int(rows[0]) + 1])
    if np.all(np.diff(rows) == 1):
        return np.asarray(array[int(rows[0]) : int(rows[-1]) + 1])
    return np.stack([np.asarray(array[int(row)]) for row in rows], axis=0)


def _empty_encoded_for_array(array: Any) -> EncodedMaskStack:
    _, n_channels, height, width = _normalize_shape(array.shape)
    return EncodedMaskStack(
        counts=np.zeros((0,), dtype=np.uint32),
        indptr=np.asarray([0], dtype=np.int64),
        present=np.zeros((0, n_channels), dtype=bool),
        area_px=np.zeros((0, n_channels), dtype=np.int32),
        bbox_xyxy=np.zeros((0, n_channels, 4), dtype=np.int32),
        shape_hw=(height, width),
    )


def _encode_selected_rows_serial(array: Any, rows: np.ndarray, *, row_batch_size: int) -> EncodedMaskStack:
    shards: list[EncodedMaskStack] = []
    shape_hw: tuple[int, int] | None = None
    for batch_rows in _iter_contiguous_batches(rows, max_batch_rows=row_batch_size):
        dense = _read_rows(array, batch_rows)
        encoded = encode_mask_stack_rle(dense)
        if shape_hw is None:
            shape_hw = encoded.shape_hw
        elif shape_hw != encoded.shape_hw:
            raise RuntimeError(f"Unexpected mask shape drift: {shape_hw!r} vs {encoded.shape_hw!r}.")
        shards.append(encoded)
    if not shards:
        return _empty_encoded_for_array(array)
    return concatenate_encoded_mask_stacks(shards)


def _split_rows_into_shards(rows: np.ndarray, *, shard_count: int) -> list[np.ndarray]:
    if rows.size == 0:
        return []
    count = max(1, min(int(shard_count), int(rows.size)))
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(rows, count) if part.size]


def _encode_rows_process_worker(
    archive_path: str,
    run_path: str,
    source_array: str,
    row_values: list[int],
    row_batch_size: int,
) -> EncodedMaskStack:
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    run_group = root[run_path]
    array = run_group[source_array]
    rows = np.asarray(row_values, dtype=np.int64)
    return _encode_selected_rows_serial(array, rows, row_batch_size=int(row_batch_size))


def _encode_selected_rows(
    array: Any,
    rows: np.ndarray,
    *,
    row_batch_size: int,
    encode_workers: int,
    archive_path: Path | None = None,
    run_path: str | None = None,
    source_array: str | None = None,
) -> tuple[EncodedMaskStack, float, str]:
    start_time = time.perf_counter()
    if int(encode_workers) <= 1:
        encoded = _encode_selected_rows_serial(array, rows, row_batch_size=row_batch_size)
        return encoded, float(time.perf_counter() - start_time), "serial"

    if archive_path is None or run_path is None or source_array is None:
        raise ValueError("Parallel encoding requires archive_path, run_path, and source_array.")
    row_shards = _split_rows_into_shards(rows, shard_count=int(encode_workers))
    if not row_shards:
        encoded = _empty_encoded_for_array(array)
        return encoded, float(time.perf_counter() - start_time), "process_shards"
    with ProcessPoolExecutor(max_workers=int(encode_workers)) as pool:
        futures = [
            pool.submit(
                _encode_rows_process_worker,
                str(archive_path),
                str(run_path),
                str(source_array),
                [int(value) for value in shard_rows.tolist()],
                int(row_batch_size),
            )
            for shard_rows in row_shards
        ]
        shards = [future.result() for future in futures]
    encoded = concatenate_encoded_mask_stacks(shards)
    return encoded, float(time.perf_counter() - start_time), "process_shards"


def _benchmark_decode(encoded: EncodedMaskStack, *, rows: int) -> tuple[int, float | None, float | None]:
    n_rows = min(int(rows), int(encoded.present.shape[0]))
    if n_rows <= 0:
        return (0, None, None)
    n_channels = int(encoded.present.shape[1])
    decoded = 0
    start_time = time.perf_counter()
    for row_idx in range(n_rows):
        for channel_idx in range(n_channels):
            flat_idx = row_idx * n_channels + channel_idx
            start = int(encoded.indptr[flat_idx])
            stop = int(encoded.indptr[flat_idx + 1])
            _ = decode_binary_mask_rle(encoded.counts[start:stop], encoded.shape_hw)
        decoded += 1
    seconds = time.perf_counter() - start_time
    return (n_rows, float(seconds), float(n_rows / seconds) if seconds > 0 else None)


def _write_temp_rle_zarr(
    *,
    temp_root: Path,
    archive_path: Path,
    family: str,
    run_name: str,
    encoded: EncodedMaskComponentStack,
    count_chunk_bytes: int,
) -> tuple[Path, int]:
    safe_name = "__".join(
        part.replace("/", "_").replace(" ", "_")
        for part in (archive_path.stem or "archive", family, run_name, _utc_now_slug())
    )
    target = Path(tempfile.mkdtemp(prefix=f"{safe_name}__", suffix=".zarr", dir=str(temp_root)))
    root = zarr.open_group(str(target), mode="w")
    write_encoded_component_rle_mask_store(
        root,
        encoded,
        overwrite=True,
        count_chunk_bytes=count_chunk_bytes,
        extra_attrs={
            "source_archive": str(archive_path),
            "source_family": str(family),
            "source_run": str(run_name),
        },
    )
    physical_bytes = _group_stored_bytes(root)
    return target, int(physical_bytes)


def _component_rle_logical_bytes(encoded: EncodedMaskComponentStack) -> int:
    total = 0
    for component in encoded.components:
        total += int(
            component.counts.nbytes
            + component.indptr.nbytes
            + component.present.nbytes
            + component.area_px.nbytes
            + component.bbox_xyxy.nbytes
        )
    return int(total)


def _resolve_run_names(parent: Any, runs_arg: str) -> list[str]:
    requested = str(runs_arg).strip()
    if requested == "all":
        return sorted(str(key) for key in parent.group_keys())
    if requested == "latest":
        latest = parent.attrs.get("latest")
        if latest is None:
            return []
        return [str(latest)]
    return [part.strip() for part in requested.split(",") if part.strip()]


def benchmark_mask_array(
    *,
    archive_path: Path,
    family: str,
    run_name: str,
    run_group: Any,
    source_array: str,
    sample_rows: str,
    row_batch_size: int,
    decode_benchmark_rows: int,
    write_temp_zarr: bool,
    temp_root: Path,
    count_chunk_bytes: int,
    delete_temp: bool,
    encode_workers: int,
) -> MaskRleBenchmarkResult:
    run_path = f"{family}/{run_name}"
    if source_array not in run_group:
        return MaskRleBenchmarkResult(
            archive=str(archive_path),
            family=family,
            run=run_name,
            run_path=run_path,
            source_array=source_array,
            status="missing_source_array",
            shape=None,
            dtype=None,
            chunks=None,
            dense_logical_bytes=None,
            dense_physical_bytes=None,
            sampled_rows=0,
            total_rows=None,
            channel_count=None,
            mask_shape_hw=None,
            sample_is_exact=False,
            rle_counts_count=None,
            rle_layout=None,
            rle_logical_bytes=None,
            rle_estimated_total_logical_bytes=None,
            rle_physical_bytes=None,
            rle_temp_zarr_path=None,
            dense_to_rle_logical_ratio=None,
            dense_to_rle_physical_ratio=None,
            encode_seconds=None,
            encode_rows_per_second=None,
            decode_benchmark_rows=0,
            decode_seconds=None,
            decode_rows_per_second=None,
            encode_workers=0,
            encode_backend="not_run",
            notes=("source array missing",),
        )

    array = run_group[source_array]
    notes: list[str] = []
    try:
        total_rows, channel_count, height, width = _normalize_shape(array.shape)
    except ValueError as exc:
        return MaskRleBenchmarkResult(
            archive=str(archive_path),
            family=family,
            run=run_name,
            run_path=run_path,
            source_array=source_array,
            status="invalid_shape",
            shape=tuple(int(v) for v in array.shape),
            dtype=str(array.dtype),
            chunks=tuple(int(v) for v in getattr(array, "chunks", ()) or ()),
            dense_logical_bytes=int(getattr(array, "nbytes", 0)),
            dense_physical_bytes=_array_stored_bytes(array),
            sampled_rows=0,
            total_rows=None,
            channel_count=None,
            mask_shape_hw=None,
            sample_is_exact=False,
            rle_counts_count=None,
            rle_layout=None,
            rle_logical_bytes=None,
            rle_estimated_total_logical_bytes=None,
            rle_physical_bytes=None,
            rle_temp_zarr_path=None,
            dense_to_rle_logical_ratio=None,
            dense_to_rle_physical_ratio=None,
            encode_seconds=None,
            encode_rows_per_second=None,
            decode_benchmark_rows=0,
            decode_seconds=None,
            decode_rows_per_second=None,
            encode_workers=0,
            encode_backend="not_run",
            notes=(str(exc),),
        )

    rows = _parse_sample_rows(sample_rows, total_rows)
    encoded, encode_seconds, encode_backend = _encode_selected_rows(
        array,
        rows,
        row_batch_size=row_batch_size,
        encode_workers=int(encode_workers),
        archive_path=archive_path,
        run_path=run_path,
        source_array=source_array,
    )
    sampled_rows = int(rows.size)
    sample_is_exact = sampled_rows == int(total_rows)
    dense_logical_bytes = int(getattr(array, "nbytes", 0))
    dense_physical_bytes = _array_stored_bytes(array)
    names = _component_names(run_group, channel_count)
    component_encoded = component_stack_from_flat_rle(encoded, component_names=names)
    rle_logical_bytes = _component_rle_logical_bytes(component_encoded)
    if sample_is_exact:
        rle_estimated_total_logical_bytes = rle_logical_bytes
    else:
        scale = float(total_rows) / float(max(1, sampled_rows))
        rle_estimated_total_logical_bytes = int(round(float(rle_logical_bytes) * scale))
        notes.append("RLE totals are estimated from sampled rows.")

    decode_rows, decode_seconds, decode_rows_per_second = _benchmark_decode(
        encoded,
        rows=decode_benchmark_rows,
    )

    rle_physical_bytes = None
    temp_path: str | None = None
    if write_temp_zarr:
        target, rle_physical_bytes = _write_temp_rle_zarr(
            temp_root=temp_root,
            archive_path=archive_path,
            family=family,
            run_name=run_name,
            encoded=component_encoded,
            count_chunk_bytes=count_chunk_bytes,
        )
        temp_path = str(target)
        if delete_temp:
            shutil.rmtree(target, ignore_errors=True)
            temp_path = None

    dense_to_rle_logical_ratio = (
        float(dense_logical_bytes) / float(rle_estimated_total_logical_bytes)
        if rle_estimated_total_logical_bytes and rle_estimated_total_logical_bytes > 0
        else None
    )
    dense_to_rle_physical_ratio = (
        float(dense_physical_bytes) / float(rle_physical_bytes)
        if dense_physical_bytes and rle_physical_bytes and rle_physical_bytes > 0
        else None
    )
    return MaskRleBenchmarkResult(
        archive=str(archive_path),
        family=family,
        run=run_name,
        run_path=run_path,
        source_array=source_array,
        status="ok",
        shape=tuple(int(v) for v in array.shape),
        dtype=str(array.dtype),
        chunks=tuple(int(v) for v in getattr(array, "chunks", ()) or ()),
        dense_logical_bytes=dense_logical_bytes,
        dense_physical_bytes=dense_physical_bytes,
        sampled_rows=sampled_rows,
        total_rows=int(total_rows),
        channel_count=int(channel_count),
        mask_shape_hw=(int(height), int(width)),
        sample_is_exact=sample_is_exact,
        rle_counts_count=int(encoded.counts.size),
        rle_layout="component_groups",
        rle_logical_bytes=rle_logical_bytes,
        rle_estimated_total_logical_bytes=rle_estimated_total_logical_bytes,
        rle_physical_bytes=rle_physical_bytes,
        rle_temp_zarr_path=temp_path,
        dense_to_rle_logical_ratio=dense_to_rle_logical_ratio,
        dense_to_rle_physical_ratio=dense_to_rle_physical_ratio,
        encode_seconds=float(encode_seconds),
        encode_rows_per_second=(float(sampled_rows) / float(encode_seconds) if encode_seconds > 0 else None),
        decode_benchmark_rows=int(decode_rows),
        decode_seconds=decode_seconds,
        decode_rows_per_second=decode_rows_per_second,
        encode_workers=int(encode_workers),
        encode_backend=encode_backend,
        notes=tuple(notes),
    )


def run_benchmark(
    archive_path: Path,
    *,
    families: Sequence[str],
    runs: str,
    source_array: str,
    sample_rows: str,
    row_batch_size: int,
    decode_benchmark_rows: int,
    write_temp_zarr: bool,
    temp_root: Path,
    count_chunk_bytes: int,
    delete_temp: bool,
    encode_workers: int,
) -> list[MaskRleBenchmarkResult]:
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    results: list[MaskRleBenchmarkResult] = []
    for family in families:
        if family not in root:
            continue
        parent = root[family]
        for run_name in _resolve_run_names(parent, runs):
            if run_name not in parent:
                results.append(
                    MaskRleBenchmarkResult(
                        archive=str(archive_path),
                        family=family,
                        run=run_name,
                        run_path=f"{family}/{run_name}",
                        source_array=source_array,
                        status="missing_run",
                        shape=None,
                        dtype=None,
                        chunks=None,
                        dense_logical_bytes=None,
                        dense_physical_bytes=None,
                        sampled_rows=0,
                        total_rows=None,
                        channel_count=None,
                        mask_shape_hw=None,
                        sample_is_exact=False,
                        rle_counts_count=None,
                        rle_layout=None,
                        rle_logical_bytes=None,
                        rle_estimated_total_logical_bytes=None,
                        rle_physical_bytes=None,
                        rle_temp_zarr_path=None,
                        dense_to_rle_logical_ratio=None,
                        dense_to_rle_physical_ratio=None,
                        encode_seconds=None,
                        encode_rows_per_second=None,
                        decode_benchmark_rows=0,
                        decode_seconds=None,
                        decode_rows_per_second=None,
                        encode_workers=0,
                        encode_backend="not_run",
                        notes=("run listed but not found",),
                    )
                )
                continue
            results.append(
                benchmark_mask_array(
                    archive_path=archive_path,
                    family=family,
                    run_name=run_name,
                    run_group=parent[run_name],
                    source_array=source_array,
                    sample_rows=sample_rows,
                    row_batch_size=row_batch_size,
                    decode_benchmark_rows=decode_benchmark_rows,
                    write_temp_zarr=write_temp_zarr,
                    temp_root=temp_root,
                    count_chunk_bytes=count_chunk_bytes,
                    delete_temp=delete_temp,
                    encode_workers=int(encode_workers),
                )
            )
    return results


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def render_markdown(results: Sequence[MaskRleBenchmarkResult]) -> str:
    lines = [
        "# Mask RLE Storage Benchmark",
        "",
        "| family | run | status | encoder | layout | shape | sampled | dense logical | dense physical | RLE logical | RLE physical | logical ratio | physical ratio | encode rows/s | decode rows/s |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        shape = "x".join(str(v) for v in result.shape) if result.shape else "-"
        sampled = f"{result.sampled_rows}/{result.total_rows}" if result.total_rows is not None else str(result.sampled_rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    result.family,
                    result.run,
                    result.status,
                    f"{result.encode_backend}/{result.encode_workers}",
                    result.rle_layout or "-",
                    shape,
                    sampled,
                    _format_bytes(result.dense_logical_bytes),
                    _format_bytes(result.dense_physical_bytes),
                    _format_bytes(result.rle_estimated_total_logical_bytes),
                    _format_bytes(result.rle_physical_bytes),
                    _format_ratio(result.dense_to_rle_logical_ratio),
                    _format_ratio(result.dense_to_rle_physical_ratio),
                    "-" if result.encode_rows_per_second is None else f"{result.encode_rows_per_second:.1f}",
                    "-" if result.decode_rows_per_second is None else f"{result.decode_rows_per_second:.1f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, results: Sequence[MaskRleBenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_id": "palette_mask_rle_storage_benchmark_v1",
        "results": [_json_safe(asdict(result)) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, results: Sequence[MaskRleBenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(_json_safe(asdict(result)), sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Analysis or training Zarr archive to inspect.")
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        help="Mask run parent groups to scan.",
    )
    parser.add_argument(
        "--runs",
        default="latest",
        help="'latest', 'all', or comma-separated run names within each family.",
    )
    parser.add_argument("--source-array", default="masks_roi", help="Dense binary mask array name.")
    parser.add_argument("--sample-rows", default="all", help="'all', an integer row count, or a fraction in (0,1].")
    parser.add_argument("--row-batch-size", type=int, default=256, help="Rows to read per contiguous batch.")
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=1,
        help="Number of row-sharded worker processes for RLE encoding. Use 1 for serial encoding.",
    )
    parser.add_argument("--decode-benchmark-rows", type=int, default=1024, help="Rows to decode for timing.")
    parser.add_argument("--write-temp-zarr", action="store_true", help="Write temporary mask_rle Zarrs to measure physical size.")
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp/palette_mask_rle_benchmark"), help="Temporary output root.")
    parser.add_argument("--count-chunk-bytes", type=int, default=4 * 1024 * 1024, help="Target 1D counts chunk size.")
    parser.add_argument("--delete-temp", action="store_true", help="Delete temporary RLE Zarrs after measuring them.")
    parser.add_argument("--json-report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--jsonl-report", type=Path, help="Optional JSONL report path, one run per line.")
    parser.add_argument("--markdown-report", type=Path, help="Optional markdown report path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    archive = Path(args.archive)
    if not archive.exists():
        parser.error(f"Archive not found: {archive}")
    if int(args.row_batch_size) <= 0:
        parser.error("--row-batch-size must be positive.")
    if int(args.encode_workers) <= 0:
        parser.error("--encode-workers must be positive.")
    if int(args.decode_benchmark_rows) < 0:
        parser.error("--decode-benchmark-rows must be non-negative.")
    if int(args.count_chunk_bytes) <= 0:
        parser.error("--count-chunk-bytes must be positive.")

    args.tmp_root.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        archive,
        families=tuple(str(value) for value in args.families),
        runs=str(args.runs),
        source_array=str(args.source_array),
        sample_rows=str(args.sample_rows),
        row_batch_size=int(args.row_batch_size),
        decode_benchmark_rows=int(args.decode_benchmark_rows),
        write_temp_zarr=bool(args.write_temp_zarr),
        temp_root=Path(args.tmp_root),
        count_chunk_bytes=int(args.count_chunk_bytes),
        delete_temp=bool(args.delete_temp),
        encode_workers=int(args.encode_workers),
    )

    if args.json_report:
        _write_json(Path(args.json_report), results)
    if args.jsonl_report:
        _write_jsonl(Path(args.jsonl_report), results)
    markdown = render_markdown(results)
    if args.markdown_report:
        Path(args.markdown_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_report).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
