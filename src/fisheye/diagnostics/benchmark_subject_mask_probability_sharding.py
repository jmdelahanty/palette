"""Build one exact subject-mask probability sharding benchmark variant.

The tool copies a contiguous row sample from one raw subject-mask run into a
small benchmark-only Zarr. It writes either the current regular chunk layout or
a Zarr v3 indexed-sharded layout, validates exact uint8 equality, and records
physical bytes, object count, timings, and process peak RSS.

Run one variant per process so ``ru_maxrss`` remains attributable to that
layout. The destination is always a new benchmark artifact; this tool never
modifies the source archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_filesystem import describe_filesystem, require_storage_tier
from fisheye.shared.batch_logging import utc_now


def _variant_name(layout: str, shard_rows: int | None) -> str:
    if layout == "regular":
        return "regular"
    if shard_rows is None:
        raise ValueError("shard_rows is required for sharded layout")
    return f"shard_{int(shard_rows):05d}"


def _validate_layout(
    *,
    layout: str,
    inner_chunk_rows: int,
    shard_rows: int | None,
) -> None:
    if layout not in {"regular", "sharded"}:
        raise ValueError("layout must be regular or sharded")
    if int(inner_chunk_rows) <= 0:
        raise ValueError("inner_chunk_rows must be positive")
    if layout == "regular":
        if shard_rows is not None:
            raise ValueError("shard_rows is only valid for sharded layout")
        return
    if shard_rows is None or int(shard_rows) <= 0:
        raise ValueError("shard_rows must be positive for sharded layout")
    if int(shard_rows) % int(inner_chunk_rows) != 0:
        raise ValueError("shard_rows must be an integer multiple of inner_chunk_rows")


def _iter_ranges(total_rows: int, rows_per_block: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(total_rows), int(rows_per_block)):
        yield int(start), min(int(total_rows), int(start) + int(rows_per_block))


def _copy_codec_kwargs(source: zarr.Array) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    compressors = getattr(source, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    filters = getattr(source, "filters", None)
    if filters:
        kwargs["filters"] = filters
    serializer = getattr(source, "serializer", None)
    if serializer is not None:
        kwargs["serializer"] = serializer
    return kwargs


def _write_variant(
    source: zarr.Array,
    destination: zarr.Array,
    *,
    sample_start: int,
    sample_rows: int,
    write_block_rows: int,
) -> None:
    channel_count = int(source.shape[1])
    for channel in range(channel_count):
        for local_start, local_stop in _iter_ranges(sample_rows, write_block_rows):
            source_start = int(sample_start) + int(local_start)
            source_stop = int(sample_start) + int(local_stop)
            values = np.asarray(source[source_start:source_stop, channel, :, :])
            destination[local_start:local_stop, channel, :, :] = values


def _array_digest(
    array: zarr.Array,
    *,
    start_row: int,
    total_rows: int,
    inner_chunk_rows: int,
) -> str:
    digest = hashlib.sha256()
    for channel in range(int(array.shape[1])):
        for local_start, local_stop in _iter_ranges(total_rows, inner_chunk_rows):
            values = np.asarray(
                array[
                    int(start_row) + int(local_start) : int(start_row) + int(local_stop),
                    channel,
                    :,
                    :,
                ]
            )
            digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _storage_stats(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    file_count = 0
    byte_count = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        byte_count += int(item.stat().st_size)
    return {
        "file_count": int(file_count),
        "stored_bytes": int(byte_count),
        "inventory_seconds": float(time.perf_counter() - started),
    }


def write_benchmark_set_manifest(output_root: Path | str) -> Path:
    output_path = Path(output_root).expanduser().resolve()
    variants: list[dict[str, Any]] = []
    for summary_path in sorted(output_path.glob("*.summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("schema_id") != "palette.subject_mask_probability_sharding_benchmark_summary.v1":
            continue
        variants.append(payload)
    source_runs = sorted({str(item.get("source_run") or "") for item in variants})
    destination_storage_tiers = sorted(
        {
            str((item.get("destination_filesystem") or {}).get("storage_tier") or "unknown")
            for item in variants
        }
    )
    sample_ranges = sorted(
        {
            (int(item.get("sample_start") or 0), int(item.get("sample_rows") or 0))
            for item in variants
        }
    )
    manifest = {
        "schema_id": "palette.subject_mask_probability_sharding_benchmark_set.v1",
        "updated_utc": utc_now(),
        "output_root": str(output_path),
        "variant_count": len(variants),
        "source_runs": source_runs,
        "destination_storage_tiers": destination_storage_tiers,
        "sample_ranges": [list(item) for item in sample_ranges],
        "all_exact_match": bool(variants) and all(bool(item.get("exact_match")) for item in variants),
        "variants": variants,
    }
    manifest_path = output_path / "benchmark_set.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _read_benchmarks(
    array: zarr.Array,
    *,
    batch_rows: int,
    random_read_count: int,
    random_seed: int,
) -> dict[str, Any]:
    total_rows = int(array.shape[0])
    batch = max(1, min(int(batch_rows), total_rows))

    sequential_bytes = 0
    started = time.perf_counter()
    for start, stop in _iter_ranges(total_rows, batch):
        values = np.asarray(array[start:stop, 0, :, :])
        sequential_bytes += int(values.nbytes)
    sequential_seconds = float(time.perf_counter() - started)

    rng = np.random.default_rng(int(random_seed))
    count = max(0, int(random_read_count))
    rows = rng.integers(0, total_rows, size=count, endpoint=False) if count else np.asarray([], dtype=np.int64)
    started = time.perf_counter()
    random_bytes = 0
    checksum = 0
    for row in rows.tolist():
        values = np.asarray(array[int(row), 0, :, :])
        random_bytes += int(values.nbytes)
        checksum = (checksum + int(values.sum(dtype=np.uint64))) % (2**63 - 1)
    random_seconds = float(time.perf_counter() - started)

    return {
        "sequential_batch_rows": int(batch),
        "sequential_component_bytes": int(sequential_bytes),
        "sequential_component_seconds": sequential_seconds,
        "sequential_component_mib_per_second": (
            float(sequential_bytes) / (1024.0 * 1024.0) / sequential_seconds
            if sequential_seconds > 0
            else None
        ),
        "random_row_reads": int(count),
        "random_row_bytes": int(random_bytes),
        "random_row_seconds": random_seconds,
        "random_row_milliseconds_per_read": (
            random_seconds * 1000.0 / float(count) if count else None
        ),
        "random_row_checksum": int(checksum),
    }


def build_probability_sharding_variant(
    source_run: Path | str,
    *,
    output_root: Path | str,
    layout: str,
    shard_rows: int | None = None,
    sample_start: int = 0,
    sample_rows: int = 8192,
    inner_chunk_rows: int = 32,
    batch_rows: int = 256,
    random_read_count: int = 32,
    random_seed: int = 0,
    require_source_storage_tier: str | None = None,
    require_destination_storage_tier: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _validate_layout(
        layout=str(layout),
        inner_chunk_rows=int(inner_chunk_rows),
        shard_rows=int(shard_rows) if shard_rows is not None else None,
    )
    source_path = Path(source_run).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    source_filesystem = describe_filesystem(source_path)
    destination_filesystem = describe_filesystem(output_path)
    require_storage_tier(
        source_filesystem,
        require_source_storage_tier,
        label="Benchmark source",
    )
    require_storage_tier(
        destination_filesystem,
        require_destination_storage_tier,
        label="Benchmark destination",
    )
    source_group = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    if "mask_probs_roi" not in source_group:
        raise ValueError(f"{source_path} does not contain mask_probs_roi")
    source = source_group["mask_probs_roi"]
    if int(source.ndim) != 4:
        raise ValueError(f"Expected a four-dimensional probability array, got shape={source.shape}")

    start = int(sample_start)
    rows = int(sample_rows)
    if start < 0 or rows <= 0 or start + rows > int(source.shape[0]):
        raise ValueError(
            f"Invalid sample range [{start}, {start + rows}) for source with {int(source.shape[0])} rows"
        )

    variant = _variant_name(str(layout), int(shard_rows) if shard_rows is not None else None)
    destination_path = output_path / f"{variant}.zarr"
    summary_path = output_path / f"{variant}.summary.json"
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        shutil.rmtree(destination_path)
    if summary_path.exists() and overwrite:
        summary_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    shape = (rows, int(source.shape[1]), int(source.shape[2]), int(source.shape[3]))
    chunks = (int(inner_chunk_rows), 1, int(source.shape[2]), int(source.shape[3]))
    shards = (
        (int(shard_rows), 1, int(source.shape[2]), int(source.shape[3]))
        if layout == "sharded" and shard_rows is not None
        else None
    )

    root = zarr.open_group(str(destination_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": "palette.subject_mask_probability_sharding_benchmark.v1",
            "created_utc": utc_now(),
            "source_run": str(source_path),
            "source_array": "mask_probs_roi",
            "source_sample_start": int(start),
            "source_sample_rows": int(rows),
            "layout": str(layout),
            "inner_chunk_rows": int(inner_chunk_rows),
            "shard_rows": int(shard_rows) if shard_rows is not None else None,
            "mask_labels": list(source_group.attrs.get("mask_labels") or []),
            "probabilities_encoding": source_group.attrs.get("probabilities_encoding"),
            "benchmark_destination_storage_tier": destination_filesystem["storage_tier"],
        }
    )
    create_kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": source.dtype,
        "chunks": chunks,
        "fill_value": source.fill_value,
        "overwrite": True,
        **_copy_codec_kwargs(source),
    }
    if shards is not None:
        create_kwargs["shards"] = shards
    destination = root.create_array("mask_probs_roi", **create_kwargs)
    destination.attrs.update(dict(source.attrs))

    write_block_rows = int(shard_rows) if shards is not None and shard_rows is not None else int(inner_chunk_rows)
    write_started = time.perf_counter()
    _write_variant(
        source,
        destination,
        sample_start=start,
        sample_rows=rows,
        write_block_rows=write_block_rows,
    )
    write_seconds = float(time.perf_counter() - write_started)

    validate_started = time.perf_counter()
    source_digest = _array_digest(
        source,
        start_row=start,
        total_rows=rows,
        inner_chunk_rows=int(inner_chunk_rows),
    )
    destination_digest = _array_digest(
        destination,
        start_row=0,
        total_rows=rows,
        inner_chunk_rows=int(inner_chunk_rows),
    )
    validation_seconds = float(time.perf_counter() - validate_started)
    exact_match = source_digest == destination_digest
    if not exact_match:
        raise RuntimeError(
            f"Probability digest mismatch for {variant}: source={source_digest} destination={destination_digest}"
        )

    storage = _storage_stats(destination_path)
    reads = _read_benchmarks(
        destination,
        batch_rows=int(batch_rows),
        random_read_count=int(random_read_count),
        random_seed=int(random_seed),
    )
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    logical_bytes = int(math.prod(shape) * np.dtype(source.dtype).itemsize)
    summary: dict[str, Any] = {
        "schema_id": "palette.subject_mask_probability_sharding_benchmark_summary.v1",
        "created_utc": utc_now(),
        "variant": variant,
        "source_run": str(source_path),
        "destination_zarr": str(destination_path),
        "sample_start": int(start),
        "sample_rows": int(rows),
        "shape": list(shape),
        "dtype": str(source.dtype),
        "chunks": list(chunks),
        "shards": list(shards) if shards is not None else None,
        "write_block_rows": int(write_block_rows),
        "logical_bytes": int(logical_bytes),
        "stored_bytes": int(storage["stored_bytes"]),
        "file_count": int(storage["file_count"]),
        "logical_to_stored_ratio": (
            float(logical_bytes) / float(storage["stored_bytes"])
            if storage["stored_bytes"] > 0
            else None
        ),
        "write_seconds": write_seconds,
        "logical_write_mib_per_second": (
            float(logical_bytes) / (1024.0 * 1024.0) / write_seconds if write_seconds > 0 else None
        ),
        "validation_seconds": validation_seconds,
        "source_sha256": source_digest,
        "destination_sha256": destination_digest,
        "exact_match": bool(exact_match),
        "peak_rss_kib": int(peak_rss_kib),
        "storage_inventory_seconds": float(storage["inventory_seconds"]),
        "source_filesystem": source_filesystem,
        "destination_filesystem": destination_filesystem,
        "read_benchmarks": reads,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_benchmark_set_manifest(output_path)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "{variant}: files={files} stored_mib={stored:.2f} write_s={write:.3f} "
        "peak_rss_mib={rss:.1f} exact={exact}".format(
            variant=summary["variant"],
            files=int(summary["file_count"]),
            stored=float(summary["stored_bytes"]) / (1024.0 * 1024.0),
            write=float(summary["write_seconds"]),
            rss=float(summary["peak_rss_kib"]) / 1024.0,
            exact=bool(summary["exact_match"]),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path, help="Raw subject-mask run containing mask_probs_roi.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("regular", "sharded"), required=True)
    parser.add_argument("--shard-rows", type=int)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--inner-chunk-rows", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--random-read-count", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--require-source-storage-tier",
        choices=("prfs", "network", "local"),
        help="Fail before writing unless source-run resolves to this storage tier.",
    )
    parser.add_argument(
        "--require-destination-storage-tier",
        choices=("prfs", "network", "local"),
        help="Fail before writing unless output-root resolves to this storage tier.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = build_probability_sharding_variant(
        args.source_run,
        output_root=args.output_root,
        layout=str(args.layout),
        shard_rows=args.shard_rows,
        sample_start=int(args.sample_start),
        sample_rows=int(args.sample_rows),
        inner_chunk_rows=int(args.inner_chunk_rows),
        batch_rows=int(args.batch_rows),
        random_read_count=int(args.random_read_count),
        random_seed=int(args.random_seed),
        require_source_storage_tier=args.require_source_storage_tier,
        require_destination_storage_tier=args.require_destination_storage_tier,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
