"""Migrate selected immutable YOLO arrays to indexed Zarr v3 shards.

The migration preserves run names, selectors, decoded array values, inner
chunks, codecs, and array attributes.  It stages and validates every rewritten
array before publishing anything, retains the ordinary arrays as same-directory
backups until the complete selected run has passed a second validation, and
rolls back on ordinary Python failures.

Default mode is a read-only plan.  ``--apply`` is intended for a single-process
LSF maintenance allocation, never a login node.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Iterator, Sequence
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.zarr_helpers import reconsolidate_zarr_metadata


MIGRATION_ID = "palette.immutable_yolo_indexed_sharding.v1"
MIGRATION_TOOL = "fisheye.utils.migrate_immutable_yolo_sharding"
REPORT_SCHEMA = "palette.immutable_yolo_sharding_migration_report.v1"
DEFAULT_DETECT_ROW_SHARD_ROWS = 262_144
DEFAULT_DETECT_FRAME_SHARD_ROWS = 262_144
DEFAULT_KEYPOINT_ROI_SHARD_ROWS = 262_144
DEFAULT_KEYPOINT_FRAME_SHARD_ROWS = 262_144
FRAME_ARRAY_NAMES = {
    "detect": frozenset({"frame_counts", "n_detections"}),
    "keypoints": frozenset({"frame_counts", "n_keypoints", "n_rois"}),
}
PARENT_NAMES = {
    "detect": "detect_runs",
    "keypoints": "keypoints_runs",
}
_NUMERIC_KINDS = frozenset("biufc")
_TEMP_PREFIX = "_palette_shard_stage_"
_BACKUP_PREFIX = "_palette_shard_backup_"
_FAILED_PREFIX = "_palette_shard_failed_"
_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zattrs", ".zgroup"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _effective_shard_rows(requested: int, inner_rows: int) -> int:
    if requested <= 0 or inner_rows <= 0:
        raise ValueError("Shard and chunk row sizes must be positive.")
    return int(math.ceil(int(requested) / int(inner_rows)) * int(inner_rows))


def _data_type(array: zarr.Array) -> Any:
    metadata = getattr(array, "metadata", None)
    return getattr(metadata, "data_type", None) or array.dtype


def _array_shards(array: zarr.Array) -> tuple[int, ...] | None:
    value = getattr(array, "shards", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _storage_stats(path: Path) -> dict[str, int]:
    totals = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    if not path.exists():
        return totals
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(root) / filename
            result = file_path.stat()
            totals["file_count"] += 1
            if filename in _METADATA_NAMES:
                totals["metadata_file_count"] += 1
            else:
                totals["payload_file_count"] += 1
            totals["apparent_bytes"] += int(result.st_size)
            totals["allocated_bytes"] += int(getattr(result, "st_blocks", 0)) * 512
    return totals


@dataclass(frozen=True)
class ArrayPlan:
    name: str
    action: str
    domain: str
    shape: tuple[int, ...]
    dtype: str
    inner_chunks: tuple[int, ...] | None
    outer_shards: tuple[int, ...] | None


@dataclass
class StagePlan:
    stage: str
    parent_name: str
    run_name: str
    run_path: str
    arrays: list[ArrayPlan] = field(default_factory=list)
    repair_noop_provenance: bool = False

    @property
    def migrated_arrays(self) -> list[ArrayPlan]:
        return [item for item in self.arrays if item.action == "migrate"]


@dataclass
class ArchivePlan:
    zarr_path: Path
    stages: list[StagePlan]

    def summary(self) -> dict[str, Any]:
        return {
            "zarr_path": str(self.zarr_path),
            "selected_runs": {item.stage: item.run_name for item in self.stages},
            "stages": [
                {
                    "stage": item.stage,
                    "parent_name": item.parent_name,
                    "run_name": item.run_name,
                    "run_path": item.run_path,
                    "arrays": [asdict(array) for array in item.arrays],
                    "arrays_to_migrate": len(item.migrated_arrays),
                    "repair_noop_provenance": item.repair_noop_provenance,
                }
                for item in self.stages
            ],
        }


def _group_at(root: Any, group_path: str) -> Any:
    group = root
    for part in group_path.split("/"):
        group = group[part]
    return group


def _selected_complete_run(root: zarr.Group, parent_name: str) -> tuple[str, zarr.Group]:
    if parent_name not in root:
        raise ValueError(f"Archive is missing {parent_name}.")
    parent = root[parent_name]
    name = parent.attrs.get("latest_complete") or parent.attrs.get("latest")
    if not name or str(name) not in parent:
        raise ValueError(f"{parent_name} has no resolvable selected run.")
    run_name = str(name)
    run = parent[run_name]
    completion = str(run.attrs.get("palette_run_completion_status") or "").strip().lower()
    if completion != "complete":
        raise ValueError(
            f"Refusing incomplete selected run {parent_name}/{run_name}: "
            f"palette_run_completion_status={completion or '<missing>'}."
        )
    return run_name, run


def _requested_rows(stage: str, domain: str) -> int:
    if stage == "detect":
        return (
            DEFAULT_DETECT_FRAME_SHARD_ROWS
            if domain == "frame"
            else DEFAULT_DETECT_ROW_SHARD_ROWS
        )
    return (
        DEFAULT_KEYPOINT_FRAME_SHARD_ROWS
        if domain == "frame"
        else DEFAULT_KEYPOINT_ROI_SHARD_ROWS
    )


def _plan_stage(root: zarr.Group, stage: str) -> StagePlan:
    parent_name = PARENT_NAMES[stage]
    run_name, run = _selected_complete_run(root, parent_name)
    run_path = f"{parent_name}/{run_name}"
    arrays: list[ArrayPlan] = []
    for name, array in sorted(run.arrays(), key=lambda item: item[0]):
        shape = tuple(int(value) for value in array.shape)
        dtype = np.dtype(array.dtype)
        domain = "frame" if str(name) in FRAME_ARRAY_NAMES[stage] else "row"
        chunks_value = getattr(array, "chunks", None)
        chunks = tuple(int(value) for value in chunks_value) if chunks_value else None
        existing_shards = _array_shards(array)
        if int(array.ndim) < 1 or dtype.kind not in _NUMERIC_KINDS:
            action = "preserve_ordinary"
            target_shards = existing_shards
        elif chunks is None:
            raise ValueError(f"{run_path}/{name} has no inner chunk contract.")
        else:
            requested = _requested_rows(stage, domain)
            target_shards = (
                _effective_shard_rows(requested, chunks[0]),
                *chunks[1:],
            )
            if existing_shards is None:
                action = "migrate"
            elif existing_shards == target_shards:
                action = "verify_existing_sharded"
            else:
                raise ValueError(
                    f"{run_path}/{name} is already sharded as {existing_shards}, "
                    f"not the default target {target_shards}."
                )
        arrays.append(
            ArrayPlan(
                name=str(name),
                action=action,
                domain=domain,
                shape=shape,
                dtype=str(dtype),
                inner_chunks=chunks,
                outer_shards=target_shards,
            )
        )
    if not arrays:
        raise ValueError(f"Selected run {run_path} has no direct arrays.")
    prefix = "detect" if stage == "detect" else "keypoint"
    migration_summary = run.attrs.get(f"{prefix}_storage_migration")
    migration_hashes = (
        dict(migration_summary).get("source_sha256_by_array")
        if isinstance(migration_summary, dict)
        else None
    )
    provenance = run.attrs.get("provenance")
    artifacts = dict(provenance).get("artifacts") if isinstance(provenance, dict) else None
    writer_summary = (
        dict(artifacts).get(f"{prefix}_shard_write")
        if isinstance(artifacts, dict)
        else None
    )
    repair_noop_provenance = bool(
        not any(item.action == "migrate" for item in arrays)
        and run.attrs.get(f"{prefix}_storage_policy") == "migrated_indexed_sharding_v1"
        and migration_hashes == {}
        and isinstance(writer_summary, dict)
        and writer_summary.get("exact_match") is True
    )
    return StagePlan(
        stage=stage,
        parent_name=parent_name,
        run_name=run_name,
        run_path=run_path,
        arrays=arrays,
        repair_noop_provenance=repair_noop_provenance,
    )


def build_plan(zarr_path: Path | str, *, stages: Sequence[str]) -> ArchivePlan:
    path = Path(zarr_path).expanduser().resolve()
    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    planned = [_plan_stage(root, stage) for stage in stages]
    return ArchivePlan(zarr_path=path, stages=planned)


def _digest_array(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    trailing = (slice(None),) * max(0, int(array.ndim) - 1)
    for start in range(0, int(array.shape[0]), max(1, int(row_step))):
        stop = min(start + max(1, int(row_step)), int(array.shape[0]))
        values = np.ascontiguousarray(array[(slice(start, stop), *trailing)])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _create_staged_array(
    source: zarr.Array,
    run: zarr.Group,
    *,
    temp_name: str,
    plan: ArrayPlan,
) -> zarr.Array:
    if plan.inner_chunks is None or plan.outer_shards is None:
        raise ValueError(f"Array {plan.name} has no sharded destination geometry.")
    kwargs: dict[str, Any] = {
        "shape": source.shape,
        "dtype": _data_type(source),
        "chunks": plan.inner_chunks,
        "shards": plan.outer_shards,
        "overwrite": False,
    }
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
    destination = run.create_array(temp_name, **kwargs)
    destination.attrs.update(dict(source.attrs))
    return destination


def _copy_complete_shards(
    source: zarr.Array,
    destination: zarr.Array,
    plan: ArrayPlan,
) -> tuple[str, float]:
    if plan.outer_shards is None:
        raise ValueError(f"Array {plan.name} has no outer shard plan.")
    digest = hashlib.sha256()
    started = time.perf_counter()
    rows = int(plan.outer_shards[0])
    trailing = (slice(None),) * max(0, int(source.ndim) - 1)
    for start in range(0, int(source.shape[0]), rows):
        stop = min(start + rows, int(source.shape[0]))
        selection = (slice(start, stop), *trailing)
        values = np.ascontiguousarray(source[selection])
        digest.update(values.view(np.uint8))
        destination[selection] = values
    return digest.hexdigest(), float(time.perf_counter() - started)


def _replace_attrs(group: zarr.Group, values: dict[str, Any]) -> None:
    for key in list(group.attrs.keys()):
        del group.attrs[key]
    group.attrs.update(values)


def _assert_no_orphan_transactions(plan: ArchivePlan) -> None:
    for stage in plan.stages:
        run_dir = plan.zarr_path.joinpath(*stage.run_path.split("/"))
        names = [
            child.name
            for child in run_dir.iterdir()
            if child.name.startswith((_TEMP_PREFIX, _BACKUP_PREFIX, _FAILED_PREFIX))
        ]
        if names:
            raise RuntimeError(
                f"{stage.run_path} contains retained migration artifacts {sorted(names)}. "
                "Inspect/recover them before another migration."
            )


@contextmanager
def _archive_lock(zarr_path: Path) -> Iterator[None]:
    lock_path = zarr_path.parent / f".{zarr_path.name}.immutable-yolo-sharding.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at_utc={_utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stage_attrs(stage: StagePlan, array_results: list[dict[str, Any]]) -> dict[str, Any]:
    now = _utc_now()
    hashes = {
        str(item["name"]): str(item["destination_sha256"])
        for item in array_results
        if item.get("action") == "migrate"
    }
    summary = {
        "schema_id": MIGRATION_ID,
        "status": "complete",
        "write_mode": "migration_complete_shards",
        "source_sha256_by_array": hashes,
        "destination_sha256_by_array": hashes,
        "exact_match": True,
        "migrated_at_utc": now,
    }
    if stage.stage == "detect":
        return {
            "detect_storage_layout": "indexed_sharding_v1",
            "detect_storage_policy": "migrated_indexed_sharding_v1",
            "detect_row_shard_rows": DEFAULT_DETECT_ROW_SHARD_ROWS,
            "detect_frame_shard_rows": DEFAULT_DETECT_FRAME_SHARD_ROWS,
            "detect_shard_write": summary,
            "detect_storage_migration": summary,
        }
    return {
        "keypoint_storage_layout": "indexed_sharding_v1",
        "keypoint_storage_policy": "migrated_indexed_sharding_v1",
        "keypoint_roi_shard_rows": DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
        "keypoint_frame_shard_rows": DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
        "keypoint_shard_write": summary,
        "keypoint_storage_migration": summary,
    }


def _restore_noop_writer_provenance(run: zarr.Group, stage: StagePlan) -> None:
    """Undo the early canary bug that relabeled an already-sharded no-op stage."""

    prefix = "detect" if stage.stage == "detect" else "keypoint"
    provenance = run.attrs.get("provenance")
    artifacts = dict(provenance).get("artifacts") if isinstance(provenance, dict) else None
    if not isinstance(artifacts, dict):
        raise ValueError(f"{stage.run_path} lacks preserved writer artifact provenance.")
    writer_summary = artifacts.get(f"{prefix}_shard_write")
    if not isinstance(writer_summary, dict) or writer_summary.get("exact_match") is not True:
        raise ValueError(f"{stage.run_path} lacks a validated preserved writer shard summary.")
    layout = artifacts.get(f"{prefix}_storage_layout")
    policy = artifacts.get(f"{prefix}_storage_policy")
    if layout != "indexed_sharding_v1" or policy != "default_indexed_sharding_v1":
        raise ValueError(
            f"{stage.run_path} preserved writer provenance is not default indexed sharding."
        )
    run.attrs.update(
        {
            f"{prefix}_storage_layout": layout,
            f"{prefix}_storage_policy": policy,
            f"{prefix}_shard_write": writer_summary,
        }
    )
    for key in (
        f"{prefix}_storage_migration",
        "immutable_yolo_sharding_migration_id",
        "immutable_yolo_sharding_migration_tool",
        "immutable_yolo_sharding_migration_status",
    ):
        if key in run.attrs:
            del run.attrs[key]


def apply_plan(plan: ArchivePlan) -> dict[str, Any]:
    """Apply one archive plan with stage-all, validate-all, publish, and rollback."""

    started_at = _utc_now()
    token = uuid4().hex
    staged: list[dict[str, Any]] = []
    published: list[dict[str, Any]] = []
    original_run_attrs: dict[str, dict[str, Any]] = {}
    original_root_attrs: dict[str, Any] = {}
    before_stats = {
        stage.run_path: _storage_stats(plan.zarr_path.joinpath(*stage.run_path.split("/")))
        for stage in plan.stages
    }

    with _archive_lock(plan.zarr_path):
        _assert_no_orphan_transactions(plan)
        root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
        original_root_attrs = dict(root.attrs)
        for stage in plan.stages:
            original_run_attrs[stage.run_path] = dict(_group_at(root, stage.run_path).attrs)
        root.attrs.update(
            {
                "immutable_yolo_sharding_migration_status": "in_progress",
                "immutable_yolo_sharding_migration_id": MIGRATION_ID,
                "immutable_yolo_sharding_migration_started_at_utc": started_at,
                "immutable_yolo_sharding_migration_selected_runs": {
                    stage.stage: stage.run_name for stage in plan.stages
                },
            }
        )
        del root

        try:
            stage_results: dict[str, list[dict[str, Any]]] = {}
            for stage in plan.stages:
                stage_results[stage.run_path] = []
                for array_plan in stage.arrays:
                    result: dict[str, Any] = {
                        "name": array_plan.name,
                        "action": array_plan.action,
                        "domain": array_plan.domain,
                        "inner_chunks": array_plan.inner_chunks,
                        "outer_shards": array_plan.outer_shards,
                    }
                    if array_plan.action != "migrate":
                        stage_results[stage.run_path].append(result)
                        continue
                    root = zarr.open_group(
                        str(plan.zarr_path), mode="a", use_consolidated=False
                    )
                    run = _group_at(root, stage.run_path)
                    source = run[array_plan.name]
                    temp_name = f"{_TEMP_PREFIX}{token}_{array_plan.name}"
                    backup_name = f"{_BACKUP_PREFIX}{token}_{array_plan.name}"
                    destination = _create_staged_array(
                        source,
                        run,
                        temp_name=temp_name,
                        plan=array_plan,
                    )
                    source_digest, copy_seconds = _copy_complete_shards(
                        source, destination, array_plan
                    )
                    validation_started = time.perf_counter()
                    destination_digest = _digest_array(
                        destination,
                        row_step=int(array_plan.outer_shards[0]),  # type: ignore[index]
                    )
                    validation_seconds = float(time.perf_counter() - validation_started)
                    if source_digest != destination_digest:
                        raise RuntimeError(
                            f"Staged digest mismatch for {stage.run_path}/{array_plan.name}: "
                            f"{source_digest} != {destination_digest}."
                        )
                    result.update(
                        {
                            "source_sha256": source_digest,
                            "destination_sha256": destination_digest,
                            "copy_seconds": copy_seconds,
                            "validation_seconds": validation_seconds,
                            "temp_name": temp_name,
                            "backup_name": backup_name,
                        }
                    )
                    staged.append(
                        {
                            "stage": stage,
                            "plan": array_plan,
                            "_result_ref": result,
                            **result,
                        }
                    )
                    stage_results[stage.run_path].append(result)
                    del destination, source, run, root

            for item in staged:
                stage = item["stage"]
                run_dir = plan.zarr_path.joinpath(*stage.run_path.split("/"))
                destination_dir = run_dir / item["name"]
                temp_dir = run_dir / item["temp_name"]
                backup_dir = run_dir / item["backup_name"]
                if not destination_dir.is_dir() or not temp_dir.is_dir() or backup_dir.exists():
                    raise RuntimeError(
                        f"Unsafe publish state for {stage.run_path}/{item['name']}."
                    )
                os.replace(destination_dir, backup_dir)
                try:
                    os.replace(temp_dir, destination_dir)
                except Exception:
                    os.replace(backup_dir, destination_dir)
                    raise
                published.append(item)

            root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
            for stage in plan.stages:
                run = _group_at(root, stage.run_path)
                if stage.migrated_arrays:
                    run.attrs.update(_stage_attrs(stage, stage_results[stage.run_path]))
                    run.attrs.update(
                        {
                            "immutable_yolo_sharding_migration_id": MIGRATION_ID,
                            "immutable_yolo_sharding_migration_tool": MIGRATION_TOOL,
                            "immutable_yolo_sharding_migration_status": "complete",
                        }
                    )
                elif stage.repair_noop_provenance:
                    _restore_noop_writer_provenance(run, stage)
            completed_at = _utc_now()
            root.attrs.update(
                {
                    "immutable_yolo_sharding_migration_status": "complete",
                    "immutable_yolo_sharding_migration_id": MIGRATION_ID,
                    "immutable_yolo_sharding_migration_tool": MIGRATION_TOOL,
                    "immutable_yolo_sharding_migration_completed_at_utc": completed_at,
                }
            )
            del root

            for item in published:
                stage = item["stage"]
                root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
                array = _group_at(root, stage.run_path)[item["name"]]
                if _array_shards(array) != tuple(item["outer_shards"]):
                    raise RuntimeError(
                        f"Published shard geometry mismatch for {stage.run_path}/{item['name']}."
                    )
                digest = _digest_array(array, row_step=int(item["outer_shards"][0]))
                if digest != item["source_sha256"]:
                    raise RuntimeError(
                        f"Published digest mismatch for {stage.run_path}/{item['name']}."
                    )
                item["published_sha256"] = digest
                item["_result_ref"]["published_sha256"] = digest
                del array, root

            for item in published:
                stage = item["stage"]
                backup = plan.zarr_path.joinpath(
                    *stage.run_path.split("/"), item["backup_name"]
                )
                shutil.rmtree(backup)

            consolidation = [
                reconsolidate_zarr_metadata(
                    plan.zarr_path,
                    group_path=stage.run_path,
                    policy="immutable_yolo_sharding_migration_v1",
                    fail_on_error=False,
                )
                for stage in plan.stages
                if stage.migrated_arrays or stage.repair_noop_provenance
            ]
            after_stats = {
                stage.run_path: _storage_stats(
                    plan.zarr_path.joinpath(*stage.run_path.split("/"))
                )
                for stage in plan.stages
            }
            return {
                **plan.summary(),
                "status": "complete",
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "stages": [
                    {
                        **next(
                            item
                            for item in plan.summary()["stages"]
                            if item["run_path"] == stage.run_path
                        ),
                        "array_results": stage_results[stage.run_path],
                    }
                    for stage in plan.stages
                ],
                "storage_before": before_stats,
                "storage_after": after_stats,
                "metadata_consolidation": consolidation,
            }
        except Exception as exc:
            rollback_errors: list[str] = []
            for item in reversed(published):
                stage = item["stage"]
                run_dir = plan.zarr_path.joinpath(*stage.run_path.split("/"))
                destination = run_dir / item["name"]
                backup = run_dir / item["backup_name"]
                failed = run_dir / f"{_FAILED_PREFIX}{token}_{item['name']}"
                try:
                    if backup.exists():
                        if destination.exists():
                            os.replace(destination, failed)
                        os.replace(backup, destination)
                        if failed.exists():
                            shutil.rmtree(failed)
                except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
                    rollback_errors.append(
                        f"{stage.run_path}/{item['name']}: {rollback_exc}"
                    )
            for item in staged:
                stage = item["stage"]
                temp = plan.zarr_path.joinpath(
                    *stage.run_path.split("/"), item["temp_name"]
                )
                if temp.exists():
                    shutil.rmtree(temp)
            try:
                root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
                for group_path, attrs in original_run_attrs.items():
                    _replace_attrs(_group_at(root, group_path), attrs)
                _replace_attrs(root, original_root_attrs)
                root.attrs.update(
                    {
                        "immutable_yolo_sharding_migration_status": "error",
                        "immutable_yolo_sharding_migration_id": MIGRATION_ID,
                        "immutable_yolo_sharding_migration_failed_at_utc": _utc_now(),
                        "immutable_yolo_sharding_migration_error": str(exc),
                        "immutable_yolo_sharding_migration_rollback_errors": rollback_errors,
                    }
                )
            except Exception as attrs_exc:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(f"attribute_restore: {attrs_exc}")
            if rollback_errors:
                raise RuntimeError(
                    f"Migration failed ({exc}); rollback also failed: {rollback_errors}"
                ) from exc
            raise


def _normalize_stages(values: Sequence[str] | None) -> tuple[str, ...]:
    raw = tuple(values or ("detect", "keypoints"))
    expanded: list[str] = []
    for value in raw:
        candidates = ("detect", "keypoints") if value == "both" else (value,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_paths", nargs="+", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        choices=("detect", "keypoints", "both"),
        help="Selected raw YOLO stage to migrate; repeatable (default: both).",
    )
    parser.add_argument("--apply", action="store_true", help="Apply the migration.")
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    stages = _normalize_stages(args.stage)
    archives: list[dict[str, Any]] = []
    for raw_path in args.zarr_paths:
        path = raw_path.expanduser().resolve()
        try:
            plan = build_plan(path, stages=stages)
            report = apply_plan(plan) if args.apply else {**plan.summary(), "status": "dry_run"}
        except Exception as exc:
            report = {
                "zarr_path": str(path),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        archives.append(report)
        print(json.dumps(report, allow_nan=False, sort_keys=True))

    payload = {
        "schema": REPORT_SCHEMA,
        "migration_id": MIGRATION_ID,
        "apply": bool(args.apply),
        "stages": list(stages),
        "archives": archives,
        "archives_ok": sum(
            report.get("status") in {"dry_run", "complete"} for report in archives
        ),
        "archives_error": sum(report.get("status") == "error" for report in archives),
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    if payload["archives_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
