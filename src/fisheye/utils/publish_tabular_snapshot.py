#!/usr/bin/env python3
"""Publish a completed keypoint/detection run as an immutable sharded snapshot.

The source run is never modified. Default mode is dry-run; ``--apply`` writes a
new sibling run, validates decoded values while streaming complete outer
shards, and only then promotes the new snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_PROVENANCE_ATTR,
    is_run_complete,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
)


SNAPSHOT_SCHEMA = "palette.immutable_tabular_snapshot.v1"
DEFAULT_TABULAR_SHARD_ROWS = 262_144
SUPPORTED_FAMILIES = frozenset({"keypoints_runs", "refined_keypoints_runs", "refined_detect_runs"})


@dataclass(frozen=True)
class ArrayPlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    shards: tuple[int, ...] | None


def _data_type(array: zarr.Array) -> Any:
    metadata = getattr(array, "metadata", None)
    return getattr(metadata, "data_type", None) or array.dtype


def _numeric_dtype(array: zarr.Array) -> bool:
    try:
        return np.dtype(array.dtype).kind in "biufc"
    except (TypeError, ValueError):
        return False


def _aligned_shards(array: zarr.Array, shard_rows: int) -> tuple[int, ...] | None:
    if int(array.ndim) < 1 or not _numeric_dtype(array):
        return None
    chunks = tuple(int(value) for value in (array.chunks or ()))
    if not chunks:
        return None
    requested_rows = min(
        int(shard_rows),
        max(int(chunks[0]), int(array.shape[0])),
    )
    rows = int(math.ceil(requested_rows / int(chunks[0])) * int(chunks[0]))
    return (rows, *chunks[1:])


def _walk_arrays(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Array]]:
    out = [
        (f"{prefix}/{name}" if prefix else str(name), array)
        for name, array in group.arrays()
    ]
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        out.extend(_walk_arrays(child, child_prefix))
    return sorted(out, key=lambda item: item[0])


def build_plan(source: zarr.Group, *, shard_rows: int) -> tuple[ArrayPlan, ...]:
    if int(shard_rows) <= 0:
        raise ValueError("shard_rows must be positive.")
    plans: list[ArrayPlan] = []
    for path, array in _walk_arrays(source):
        plans.append(
            ArrayPlan(
                path=path,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                chunks=(
                    tuple(int(value) for value in array.chunks)
                    if array.chunks is not None
                    else None
                ),
                shards=_aligned_shards(array, int(shard_rows)),
            )
        )
    return tuple(plans)


def _require_group_path(root: zarr.Group, path: str) -> zarr.Group:
    group = root
    for part in [value for value in str(path).split("/") if value]:
        group = group.require_group(part)
    return group


def _create_array_like(
    target_run: zarr.Group,
    path: str,
    source: zarr.Array,
    plan: ArrayPlan,
) -> zarr.Array:
    parts = path.split("/")
    parent = _require_group_path(target_run, "/".join(parts[:-1]))
    kwargs: dict[str, Any] = {
        "shape": source.shape,
        "dtype": _data_type(source),
        "overwrite": True,
    }
    if plan.chunks is not None:
        kwargs["chunks"] = plan.chunks
    if plan.shards is not None:
        kwargs["shards"] = plan.shards
    fill_value = getattr(source, "fill_value", None)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    compressors = getattr(source, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    filters = getattr(source, "filters", None)
    if filters:
        kwargs["filters"] = filters
    serializer = getattr(source, "serializer", None)
    if serializer is not None:
        kwargs["serializer"] = serializer
    destination = parent.create_array(parts[-1], **kwargs)
    destination.attrs.update(dict(source.attrs))
    return destination


def _update_digest(digest: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    if contiguous.dtype.kind in "biufc":
        digest.update(contiguous.view(np.uint8))
        return
    digest.update(
        json.dumps(contiguous.tolist(), ensure_ascii=False, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )


def _copy_array(
    source: zarr.Array,
    destination: zarr.Array,
    plan: ArrayPlan,
) -> tuple[str, str]:
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    if int(source.ndim) == 0:
        source_values = np.asarray(source[...])
        destination[...] = source_values
        destination_values = np.asarray(destination[...])
        _update_digest(source_digest, source_values)
        _update_digest(destination_digest, destination_values)
        return source_digest.hexdigest(), destination_digest.hexdigest()

    row_step = (
        int(plan.shards[0])
        if plan.shards is not None
        else max(1, int(plan.chunks[0]) if plan.chunks else int(source.shape[0]) or 1)
    )
    trailing = (slice(None),) * (int(source.ndim) - 1)
    for start in range(0, int(source.shape[0]), row_step):
        stop = min(start + row_step, int(source.shape[0]))
        selection = (slice(start, stop), *trailing)
        source_values = np.asarray(source[selection])
        destination[selection] = source_values
        destination_values = np.asarray(destination[selection])
        _update_digest(source_digest, source_values)
        _update_digest(destination_digest, destination_values)
    return source_digest.hexdigest(), destination_digest.hexdigest()


def _copy_group_attrs(source: zarr.Group, target: zarr.Group) -> None:
    target.attrs.update(dict(source.attrs))
    for name, child in source.groups():
        target_child = target.require_group(str(name))
        target_child.attrs.update(dict(child.attrs))
        _copy_group_attrs(child, target_child)


def publish_tabular_snapshot(
    *,
    zarr_path: str | Path,
    family: str,
    source_run: str,
    output_run: str,
    shard_rows: int = DEFAULT_TABULAR_SHARD_ROWS,
    apply: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported family {family!r}; choose from {sorted(SUPPORTED_FAMILIES)}")
    if not source_run or not output_run or "/" in source_run or "/" in output_run:
        raise ValueError("source_run and output_run must be non-empty single group names.")
    root = open_zarr_root(archive, mode="a" if apply else "r")
    parent = root.get(family)
    if parent is None or source_run not in parent:
        raise ValueError(f"Missing source run {family}/{source_run}.")
    source = parent[source_run]
    if not is_run_complete(source):
        raise ValueError(f"Source run {family}/{source_run} is not complete.")
    plans = build_plan(source, shard_rows=int(shard_rows))
    result: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "planned" if not apply else "running",
        "zarr_path": str(archive),
        "family": family,
        "source_run": source_run,
        "source_run_path": f"{family}/{source_run}",
        "output_run": output_run,
        "output_run_path": f"{family}/{output_run}",
        "shard_rows": int(shard_rows),
        "promote": bool(promote),
        "arrays": [asdict(plan) for plan in plans],
        "sharded_array_count": sum(plan.shards is not None for plan in plans),
    }
    if not apply:
        return result
    if output_run in parent:
        raise ValueError(f"Output run already exists: {family}/{output_run}.")

    target = parent.create_group(output_run)
    _copy_group_attrs(source, target)
    mark_run_started(target, run_name=output_run, stage=f"{family}_immutable_snapshot")
    if promote:
        note_pending_latest(parent, output_run)
    try:
        validation: list[dict[str, Any]] = []
        plans_by_path = {plan.path: plan for plan in plans}
        for path, source_array in _walk_arrays(source):
            plan = plans_by_path[path]
            destination = _create_array_like(target, path, source_array, plan)
            source_sha256, destination_sha256 = _copy_array(source_array, destination, plan)
            if source_sha256 != destination_sha256:
                raise RuntimeError(f"Decoded digest mismatch for {path}.")
            validation.append(
                {
                    "path": path,
                    "source_sha256": source_sha256,
                    "destination_sha256": destination_sha256,
                    "exact": True,
                }
            )

        target.attrs.update(
            {
                "snapshot_schema": SNAPSHOT_SCHEMA,
                "artifact_mutability": "immutable_snapshot",
                "snapshot_source_run": source_run,
                "snapshot_source_run_path": f"{family}/{source_run}",
                "tabular_storage_layout": "indexed_sharding_v1",
                "tabular_shard_rows": int(shard_rows),
                "tabular_snapshot_validation": {
                    "status": "complete",
                    "array_count": len(validation),
                    "sharded_array_count": sum(plan.shards is not None for plan in plans),
                    "exact": True,
                },
            }
        )
        provenance = build_writer_run_provenance(
            command="fisheye.utils.publish_tabular_snapshot",
            params={
                "zarr_path": str(archive),
                "family": family,
                "source_run": source_run,
                "output_run": output_run,
                "shard_rows": int(shard_rows),
                "promote": bool(promote),
            },
            input_run_ids={"source_run": f"{family}/{source_run}"},
            cwd=Path.cwd(),
        )
        target.attrs[RUN_PROVENANCE_ATTR] = provenance
        mark_run_complete(
            target,
            parent_group=parent if promote else None,
            run_name=output_run,
            run_provenance=provenance,
        )
        result.update({"status": "complete", "validation": validation})
        return result
    except Exception as exc:
        mark_run_failed(
            target,
            parent_group=parent if promote else None,
            run_name=output_run,
            error=str(exc),
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--family", choices=sorted(SUPPORTED_FAMILIES), required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_TABULAR_SHARD_ROWS)
    parser.add_argument("--apply", action="store_true", help="Write the snapshot; default is dry-run.")
    parser.add_argument("--no-promote", action="store_true", help="Complete without changing latest pointers.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = publish_tabular_snapshot(
        zarr_path=args.zarr_path,
        family=args.family,
        source_run=args.source_run,
        output_run=args.output_run,
        shard_rows=int(args.shard_rows),
        apply=bool(args.apply),
        promote=not bool(args.no_promote),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: {result['source_run_path']} -> {result['output_run_path']} "
            f"sharded_arrays={result['sharded_array_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
