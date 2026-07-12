"""Audit ordinary and indexed-sharded YOLO detection runs for exact parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr


DETECT_ARRAYS = (
    "frame_indices",
    "bbox_norm_coords",
    "scores",
    "class_ids",
    "instance_key",
    "n_detections",
    "frame_counts",
)


def _digest_array(array: zarr.Array, *, row_step: int = 262_144) -> str:
    digest = hashlib.sha256()
    step = max(1, int(row_step))
    for start in range(0, int(array.shape[0]), step):
        stop = min(start + step, int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop, ...])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _physical_stats(path: Path) -> dict[str, int]:
    files = 0
    payload_files = 0
    apparent_bytes = 0
    allocated_bytes = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            item = Path(root) / name
            stat = item.stat()
            files += 1
            apparent_bytes += int(stat.st_size)
            allocated_bytes += int(stat.st_blocks * 512)
            if name != "zarr.json":
                payload_files += 1
    return {
        "files": files,
        "payload_files": payload_files,
        "apparent_bytes": apparent_bytes,
        "allocated_bytes": allocated_bytes,
    }


def audit_detection_runs(
    zarr_path: Path,
    *,
    regular_run: str,
    sharded_run: str,
) -> dict[str, Any]:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    regular = root[f"detect_runs/{regular_run}"]
    sharded = root[f"detect_runs/{sharded_run}"]
    arrays: dict[str, dict[str, Any]] = {}
    all_exact = True
    for name in DETECT_ARRAYS:
        source = regular[name]
        candidate = sharded[name]
        source_hash = _digest_array(source)
        candidate_hash = _digest_array(candidate)
        exact = (
            tuple(source.shape) == tuple(candidate.shape)
            and str(source.dtype) == str(candidate.dtype)
            and source_hash == candidate_hash
        )
        all_exact = all_exact and exact
        arrays[name] = {
            "shape": [int(value) for value in source.shape],
            "dtype": str(source.dtype),
            "regular_chunks": [int(value) for value in source.chunks],
            "regular_shards": (
                [int(value) for value in source.shards] if source.shards is not None else None
            ),
            "sharded_chunks": [int(value) for value in candidate.chunks],
            "sharded_shards": (
                [int(value) for value in candidate.shards]
                if candidate.shards is not None
                else None
            ),
            "regular_sha256": source_hash,
            "sharded_sha256": candidate_hash,
            "exact": exact,
        }

    regular_path = zarr_path / "detect_runs" / regular_run
    sharded_path = zarr_path / "detect_runs" / sharded_run
    report = {
        "schema_id": "palette.yolo_detection_sharding_ab.v1",
        "zarr_path": str(zarr_path),
        "regular_run": regular_run,
        "sharded_run": sharded_run,
        "all_arrays_exact": all_exact,
        "arrays": arrays,
        "regular_physical": _physical_stats(regular_path),
        "sharded_physical": _physical_stats(sharded_path),
        "regular_attrs": {
            "inference_duration_seconds": regular.attrs.get("inference_duration_seconds"),
            "zarr_write_seconds_total": regular.attrs.get("zarr_write_seconds_total"),
            "summary_statistics": regular.attrs.get("summary_statistics"),
            "detect_storage_layout": regular.attrs.get("detect_storage_layout"),
        },
        "sharded_attrs": {
            "inference_duration_seconds": sharded.attrs.get("inference_duration_seconds"),
            "zarr_write_seconds_total": sharded.attrs.get("zarr_write_seconds_total"),
            "summary_statistics": sharded.attrs.get("summary_statistics"),
            "detect_storage_layout": sharded.attrs.get("detect_storage_layout"),
            "detect_shard_write": sharded.attrs.get("detect_shard_write"),
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--regular-run", required=True)
    parser.add_argument("--sharded-run", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    report = audit_detection_runs(
        args.zarr_path.expanduser().resolve(),
        regular_run=args.regular_run,
        sharded_run=args.sharded_run,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_arrays_exact"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
