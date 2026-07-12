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


def replay_detection_run_as_sharded(
    zarr_path: Path,
    *,
    source_run: str,
    destination_run: str,
    detect_row_shard_rows: int = 262_144,
    detect_frame_shard_rows: int = 262_144,
) -> dict[str, Any]:
    """Replay one materialized detect table through the production shard writer."""

    from fisheye.detection.detect_yolo import _write_detection_output_arrays

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    parent = root["detect_runs"]
    if destination_run in parent:
        raise ValueError(f"Destination detect run already exists: {destination_run}")
    source = parent[source_run]
    destination = parent.create_group(destination_run)
    summary = _write_detection_output_arrays(
        destination,
        frame_indices=source["frame_indices"][:],
        bbox_coords=source["bbox_norm_coords"][:],
        scores=source["scores"][:],
        class_ids=source["class_ids"][:],
        instance_keys=source["instance_key"][:],
        frame_counts=source["frame_counts"][:],
        det_chunk=int(source["frame_indices"].chunks[0]),
        detect_row_shard_rows=int(detect_row_shard_rows),
        detect_frame_shard_rows=int(detect_frame_shard_rows),
    )
    attrs = dict(source.attrs)
    attrs.update(
        {
            "benchmark_only": True,
            "detect_storage_layout": "indexed_sharding_v1",
            "detect_row_shard_rows": int(detect_row_shard_rows),
            "detect_frame_shard_rows": int(detect_frame_shard_rows),
            "detect_shard_write": summary,
            "sharded_replay_source_run": source_run,
        }
    )
    destination.attrs.put(attrs)
    return summary or {}


def _semantic_frame_diff(regular: zarr.Group, sharded: zarr.Group) -> dict[str, Any]:
    regular_frames = np.asarray(regular["frame_indices"][:], dtype=np.int64)
    sharded_frames = np.asarray(sharded["frame_indices"][:], dtype=np.int64)
    regular_unique, regular_index = np.unique(regular_frames, return_index=True)
    sharded_unique, sharded_index = np.unique(sharded_frames, return_index=True)
    shared, regular_shared_pos, sharded_shared_pos = np.intersect1d(
        regular_unique,
        sharded_unique,
        assume_unique=True,
        return_indices=True,
    )
    regular_rows = regular_index[regular_shared_pos]
    sharded_rows = sharded_index[sharded_shared_pos]
    shared_exact = {
        name: bool(
            np.array_equal(
                np.asarray(regular[name][:])[regular_rows, ...],
                np.asarray(sharded[name][:])[sharded_rows, ...],
            )
        )
        for name in ("bbox_norm_coords", "scores", "class_ids", "instance_key")
    }
    only_regular = np.setdiff1d(regular_unique, sharded_unique, assume_unique=True)
    only_sharded = np.setdiff1d(sharded_unique, regular_unique, assume_unique=True)
    return {
        "regular_detection_rows": int(regular_frames.shape[0]),
        "sharded_detection_rows": int(sharded_frames.shape[0]),
        "shared_detection_frames": int(shared.shape[0]),
        "regular_only_detection_frames": [int(value) for value in only_regular[:100]],
        "sharded_only_detection_frames": [int(value) for value in only_sharded[:100]],
        "shared_rows_exact_by_array": shared_exact,
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
        "semantic_frame_diff": _semantic_frame_diff(regular, sharded),
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
    parser.add_argument(
        "--replay-sharded-run",
        action="store_true",
        help="Create --sharded-run by replaying --regular-run through the production shard writer.",
    )
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.replay_sharded_run:
        replay_detection_run_as_sharded(
            args.zarr_path.expanduser().resolve(),
            source_run=args.regular_run,
            destination_run=args.sharded_run,
        )
    report = audit_detection_runs(
        args.zarr_path.expanduser().resolve(),
        regular_run=args.regular_run,
        sharded_run=args.sharded_run,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_arrays_exact"] or args.allow_mismatch else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
