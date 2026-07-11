"""Repeat cold/warm reads across a probability-sharding benchmark set."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_filesystem import describe_filesystem, require_storage_tier
from fisheye.shared.batch_logging import utc_now


def _iter_ranges(total_rows: int, batch_rows: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(total_rows), int(batch_rows)):
        yield int(start), min(int(total_rows), int(start) + int(batch_rows))


def _request_cache_eviction(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    summary: dict[str, Any] = {
        "supported": int(bool(supported)),
        "files_advised": 0,
        "errors": 0,
    }
    if not supported:
        summary["seconds"] = float(time.perf_counter() - started)
        return summary
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        fd: int | None = None
        try:
            fd = os.open(item, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            summary["files_advised"] += 1
        except OSError:
            summary["errors"] += 1
        finally:
            if fd is not None:
                os.close(fd)
    summary["seconds"] = float(time.perf_counter() - started)
    return summary


def _scan_component(
    array: zarr.Array,
    *,
    component: int,
    batch_rows: int,
) -> dict[str, Any]:
    logical_bytes = 0
    checksum = 0
    started = time.perf_counter()
    for start, stop in _iter_ranges(int(array.shape[0]), int(batch_rows)):
        values = np.asarray(array[start:stop, int(component), :, :])
        logical_bytes += int(values.nbytes)
        if values.size:
            row_indexes = np.linspace(
                0,
                int(values.shape[0]) - 1,
                num=min(8, int(values.shape[0])),
                dtype=np.intp,
            )
            y_indexes = np.asarray(
                [
                    int(values.shape[1]) // 4,
                    int(values.shape[1]) // 2,
                    3 * int(values.shape[1]) // 4,
                ],
                dtype=np.intp,
            )
            x_indexes = np.asarray(
                [
                    int(values.shape[2]) // 4,
                    int(values.shape[2]) // 2,
                    3 * int(values.shape[2]) // 4,
                ],
                dtype=np.intp,
            )
            sampled = values[np.ix_(row_indexes, y_indexes, x_indexes)]
            checksum = (
                checksum + int(sampled.sum(dtype=np.uint64))
            ) % (2**63 - 1)
    seconds = float(time.perf_counter() - started)
    return {
        "seconds": seconds,
        "logical_bytes": int(logical_bytes),
        "mib_per_second": (
            float(logical_bytes) / (1024.0 * 1024.0) / seconds if seconds > 0 else None
        ),
        "checksum": int(checksum),
    }


def _measure_variant(
    variant_path: Path,
    *,
    component: int,
    batch_rows: int,
    evict_cache: bool,
) -> dict[str, Any]:
    eviction = (
        _request_cache_eviction(variant_path)
        if evict_cache
        else {"supported": 0, "files_advised": 0, "errors": 0, "seconds": 0.0}
    )
    open_started = time.perf_counter()
    root = zarr.open_group(str(variant_path), mode="r", use_consolidated=False)
    array = root["mask_probs_roi"]
    open_seconds = float(time.perf_counter() - open_started)
    cold = _scan_component(array, component=int(component), batch_rows=int(batch_rows))
    warm = _scan_component(array, component=int(component), batch_rows=int(batch_rows))
    if int(cold["checksum"]) != int(warm["checksum"]):
        raise RuntimeError(f"Cold/warm checksum mismatch for {variant_path}")
    return {
        "cache_eviction": eviction,
        "open_seconds": open_seconds,
        "cold": cold,
        "warm": warm,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}
    return {
        "minimum": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "maximum": float(ordered[-1]),
    }


def benchmark_probability_sharding_reads(
    benchmark_root: Path | str,
    *,
    repeats: int = 7,
    batch_rows: int = 256,
    component: int = 0,
    random_seed: int = 0,
    evict_cache: bool = True,
    require_storage_tier_name: str | None = None,
    output_json: Path | str | None = None,
) -> dict[str, Any]:
    root_path = Path(benchmark_root).expanduser().resolve()
    benchmark_filesystem = describe_filesystem(root_path)
    require_storage_tier(
        benchmark_filesystem,
        require_storage_tier_name,
        label="Read benchmark root",
    )
    set_path = root_path / "benchmark_set.json"
    benchmark_set = json.loads(set_path.read_text(encoding="utf-8"))
    variants = [
        {
            "variant": str(item["variant"]),
            "path": Path(str(item["destination_zarr"])).expanduser().resolve(),
        }
        for item in benchmark_set.get("variants", [])
    ]
    if not variants:
        raise ValueError(f"No variants found in {set_path}")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive")

    rng = random.Random(int(random_seed))
    rounds: list[dict[str, Any]] = []
    for repeat_index in range(int(repeats)):
        order = list(variants)
        rng.shuffle(order)
        measurements: list[dict[str, Any]] = []
        for ordinal, item in enumerate(order):
            result = _measure_variant(
                item["path"],
                component=int(component),
                batch_rows=int(batch_rows),
                evict_cache=bool(evict_cache),
            )
            measurements.append(
                {
                    "variant": item["variant"],
                    "order": int(ordinal),
                    **result,
                }
            )
        rounds.append(
            {
                "repeat_index": int(repeat_index),
                "variant_order": [str(item["variant"]) for item in order],
                "measurements": measurements,
            }
        )

    summaries: list[dict[str, Any]] = []
    for item in variants:
        variant = str(item["variant"])
        rows = [
            measurement
            for round_payload in rounds
            for measurement in round_payload["measurements"]
            if measurement["variant"] == variant
        ]
        cold_rates = [float(row["cold"]["mib_per_second"]) for row in rows]
        warm_rates = [float(row["warm"]["mib_per_second"]) for row in rows]
        cold_seconds = [float(row["cold"]["seconds"]) for row in rows]
        warm_seconds = [float(row["warm"]["seconds"]) for row in rows]
        open_seconds = [float(row["open_seconds"]) for row in rows]
        eviction_seconds = [float(row["cache_eviction"]["seconds"]) for row in rows]
        summaries.append(
            {
                "variant": variant,
                "cold_mib_per_second": _distribution(cold_rates),
                "warm_mib_per_second": _distribution(warm_rates),
                "cold_seconds": _distribution(cold_seconds),
                "warm_seconds": _distribution(warm_seconds),
                "metadata_open_seconds": _distribution(open_seconds),
                "cache_eviction_seconds": _distribution(eviction_seconds),
                "cache_eviction_supported_all": all(
                    bool(row["cache_eviction"]["supported"]) for row in rows
                ),
                "cache_eviction_error_count": int(
                    sum(int(row["cache_eviction"]["errors"]) for row in rows)
                ),
            }
        )

    result = {
        "schema_id": "palette.subject_mask_probability_sharding_read_benchmark.v1",
        "created_utc": utc_now(),
        "benchmark_root": str(root_path),
        "repeats": int(repeats),
        "batch_rows": int(batch_rows),
        "component": int(component),
        "random_seed": int(random_seed),
        "cache_eviction_requested": bool(evict_cache),
        "benchmark_filesystem": benchmark_filesystem,
        "variant_summaries": summaries,
        "rounds": rounds,
    }
    output_path = (
        Path(output_json).expanduser().resolve()
        if output_json is not None
        else root_path / "read_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _print_summary(result: dict[str, Any]) -> None:
    print(
        f"repeats={result['repeats']} batch_rows={result['batch_rows']} "
        f"component={result['component']} "
        f"storage_tier={result['benchmark_filesystem']['storage_tier']}"
    )
    for row in result["variant_summaries"]:
        cold = row["cold_mib_per_second"]
        warm = row["warm_mib_per_second"]
        print(
            f"{row['variant']}: cold_median={float(cold['median']):.1f} MiB/s "
            f"warm_median={float(warm['median']):.1f} MiB/s "
            f"eviction_errors={int(row['cache_eviction_error_count'])}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--no-evict-cache", action="store_true")
    parser.add_argument(
        "--require-storage-tier",
        choices=("prfs", "network", "local"),
        help="Fail unless benchmark-root resolves to this storage tier.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write results here instead of benchmark-root/read_benchmark.json.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = benchmark_probability_sharding_reads(
        args.benchmark_root,
        repeats=int(args.repeats),
        batch_rows=int(args.batch_rows),
        component=int(args.component),
        random_seed=int(args.random_seed),
        evict_cache=not bool(args.no_evict_cache),
        require_storage_tier_name=args.require_storage_tier,
        output_json=args.output_json,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
