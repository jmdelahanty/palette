"""Benchmark validated row/window/full scans of one immutable crop snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_filesystem import describe_filesystem
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.crop_consumer import build_crop_run_reference
from fisheye.shared.zarr.crop_manifest import validate_crop_run_manifest


_WINDOW_ARRAYS = (
    "instance_key",
    "bbox_img_xyxy",
    "roi_coordinates_full",
    "source_row_signature",
)


def _ranges(total: int, rows: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(total), int(rows)):
        yield start, min(int(total), start + int(rows))


def _request_cache_eviction(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    result: dict[str, Any] = {
        "method": "posix_fadvise_POSIX_FADV_DONTNEED",
        "supported": bool(supported),
        "files_advised": 0,
        "errors": 0,
    }
    if supported:
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            descriptor: int | None = None
            try:
                descriptor = os.open(item, os.O_RDONLY)
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                result["files_advised"] += 1
            except OSError:
                result["errors"] += 1
            finally:
                if descriptor is not None:
                    os.close(descriptor)
    result["seconds"] = float(time.perf_counter() - started)
    return result


def _open_run(
    archive: Path,
    *,
    run_id: str,
    consolidated: bool,
) -> tuple[Any, float]:
    started = time.perf_counter()
    root = zarr.open_group(
        str(archive),
        mode="r",
        use_consolidated=consolidated,
    )
    run = root["crop_runs"][run_id]
    return run, float(time.perf_counter() - started)


def _window_read(run: Any, *, start: int, stop: int) -> dict[str, Any]:
    started = time.perf_counter()
    logical_bytes = 0
    digest = hashlib.sha256()
    for path in _WINDOW_ARRAYS:
        values = np.ascontiguousarray(run[path][start:stop])
        logical_bytes += int(values.nbytes)
        digest.update(values.tobytes(order="C"))
    seconds = float(time.perf_counter() - started)
    return {
        "start": int(start),
        "stop": int(stop),
        "rows": int(stop - start),
        "logical_bytes": logical_bytes,
        "seconds": seconds,
        "mib_per_second": (
            logical_bytes / 1024**2 / seconds if seconds > 0 else None
        ),
        "digest": digest.hexdigest(),
    }


def _full_scan(
    run: Any,
    *,
    manifest: dict[str, Any],
    batch_rows: int,
) -> dict[str, Any]:
    expected = manifest["payload"]["logical_content"]["document"]["arrays"]
    started = time.perf_counter()
    logical_bytes = 0
    observed: dict[str, str] = {}
    for path, declaration in expected.items():
        array = run[path]
        digest = hashlib.sha256()
        for start, stop in _ranges(int(array.shape[0]), int(batch_rows)):
            values = np.ascontiguousarray(array[start:stop])
            logical_bytes += int(values.nbytes)
            digest.update(values.tobytes(order="C"))
        observed[path] = digest.hexdigest()
        if observed[path] != declaration["sha256"]:
            raise RuntimeError(f"Decoded crop digest mismatch for {path!r}.")
    seconds = float(time.perf_counter() - started)
    return {
        "batch_rows": int(batch_rows),
        "logical_bytes": logical_bytes,
        "seconds": seconds,
        "mib_per_second": (
            logical_bytes / 1024**2 / seconds if seconds > 0 else None
        ),
        "array_sha256": observed,
    }


def _measure_pass(
    archive: Path,
    *,
    run_id: str,
    batch_rows: int,
    window_rows: int,
    evict_cache: bool,
) -> dict[str, Any]:
    eviction = (
        _request_cache_eviction(archive)
        if evict_cache
        else {
            "method": "none",
            "supported": False,
            "files_advised": 0,
            "errors": 0,
            "seconds": 0.0,
        }
    )
    direct, direct_open_seconds = _open_run(
        archive,
        run_id=run_id,
        consolidated=False,
    )
    consolidated, consolidated_open_seconds = _open_run(
        archive,
        run_id=run_id,
        consolidated=True,
    )
    manifest = dict(direct.attrs["run_manifest"])
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise RuntimeError("Invalid crop run manifest: " + "; ".join(errors))
    direct_reference = build_crop_run_reference(direct, run_id=run_id)
    consolidated_reference = build_crop_run_reference(consolidated, run_id=run_id)
    if direct_reference != consolidated_reference:
        raise RuntimeError("Direct and consolidated crop references differ.")
    n_rows = int(manifest["payload"]["logical_schema"]["dimensions"]["n_instances"])
    count = min(int(window_rows), n_rows)
    starts = sorted({0, max(0, (n_rows - count) // 2), max(0, n_rows - count)})
    windows = [
        _window_read(consolidated, start=start, stop=start + count)
        for start in starts
    ]
    full_scan = _full_scan(
        consolidated,
        manifest=manifest,
        batch_rows=batch_rows,
    )
    return {
        "cache_eviction": eviction,
        "direct_open_seconds": direct_open_seconds,
        "consolidated_open_seconds": consolidated_open_seconds,
        "run_reference": direct_reference,
        "windows": windows,
        "full_scan": full_scan,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    return {
        "minimum": min(rows),
        "median": statistics.median(rows),
        "mean": statistics.fmean(rows),
        "maximum": max(rows),
    }


def benchmark_crop_snapshot_reads(
    archive: str | Path,
    *,
    run_id: str,
    repetitions: int = 3,
    batch_rows: int = 131_072,
    window_rows: int = 1_024,
    evict_cache: bool = True,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(archive).expanduser().resolve()
    if int(repetitions) <= 0 or int(batch_rows) <= 0 or int(window_rows) <= 0:
        raise ValueError("repetitions, batch_rows, and window_rows must be positive.")
    passes = [
        _measure_pass(
            path,
            run_id=run_id,
            batch_rows=int(batch_rows),
            window_rows=int(window_rows),
            evict_cache=bool(evict_cache and index == 0),
        )
        for index in range(int(repetitions))
    ]
    payload = {
        "schema_id": "palette.crop_snapshot_read_benchmark",
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": utc_now(),
        "archive": str(path),
        "run_id": run_id,
        "filesystem": describe_filesystem(path),
        "repetitions": int(repetitions),
        "batch_rows": int(batch_rows),
        "window_rows": int(window_rows),
        "passes": passes,
        "summary": {
            "direct_open_seconds": _distribution(
                [item["direct_open_seconds"] for item in passes]
            ),
            "consolidated_open_seconds": _distribution(
                [item["consolidated_open_seconds"] for item in passes]
            ),
            "full_scan_seconds": _distribution(
                [item["full_scan"]["seconds"] for item in passes]
            ),
            "full_scan_mib_per_second": _distribution(
                [item["full_scan"]["mib_per_second"] for item in passes]
            ),
        },
    }
    if output_json is not None:
        write_json_atomic(Path(output_json), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-rows", type=int, default=131_072)
    parser.add_argument("--window-rows", type=int, default=1_024)
    parser.add_argument("--no-cache-eviction", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = benchmark_crop_snapshot_reads(
        args.zarr,
        run_id=args.run_id,
        repetitions=args.repetitions,
        batch_rows=args.batch_rows,
        window_rows=args.window_rows,
        evict_cache=not args.no_cache_eviction,
        output_json=args.output_json,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
