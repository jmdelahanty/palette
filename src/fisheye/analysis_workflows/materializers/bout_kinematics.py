"""Compute or rematerialize bout kinematics locally and publish atomically.

Fresh computation opens authoritative inputs read-only and writes its complete
run into a node-local Zarr before validation and atomic publication. The
storage-only mode copies an existing completed run into the same compact shard
profile and publishes a non-promoted candidate for review.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ...analysis import bout_kinematics as bout_writer
from ...analysis.bout_kinematics import (
    LAYOUT_COMPACT_TABULAR_V2,
    resolve_bout_kinematics_tables,
)
from ...shared.json_safety import json_attr_safe
from ...shared.run_provenance import build_run_provenance
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent
from ...shared.zarr_sharded_copy import (
    SHARD_POLICY_MULTI_CHUNK_CAPPED,
    copy_completed_run_to_sharded,
)
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.bout_kinematics_storage_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.bout_kinematics_storage_publish.v1"
COMPUTE_MATERIALIZATION_SCHEMA_ID = "palette.bout_kinematics_compute_materialization.v1"
COMPUTE_PUBLISH_SCHEMA_ID = "palette.bout_kinematics_compute_publish.v1"
PROMOTION_SCHEMA_ID = "palette.bout_kinematics_candidate_promotion.v1"
DEFAULT_OUTPUT_SHARD_ROWS = 262_144
COMMON_LOCK_SUFFIX = "bout-kinematics-publish"
MANAGED_COMPUTE_WRITER_ARGUMENTS = {
    "--output-zarr-path",
    "--output-shard-rows",
    "--overwrite",
    "--run-name",
}


@dataclass(frozen=True)
class BoutKinematicsStoragePlan:
    source_zarr: Path
    source_run_name: str
    source_run_path: Path
    scratch_root: Path
    local_run_path: Path
    run_name: str
    target_run_path: Path
    output_shard_rows: int
    workers: int
    latest_before: str | None
    latest_complete_before: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "source_run_name": self.source_run_name,
            "source_run_path": str(self.source_run_path),
            "scratch_root": str(self.scratch_root),
            "local_run_path": str(self.local_run_path),
            "run_name": self.run_name,
            "target_run_path": str(self.target_run_path),
            "output_shard_rows": int(self.output_shard_rows),
            "workers": int(self.workers),
            "shard_policy": SHARD_POLICY_MULTI_CHUNK_CAPPED,
            "latest_before": self.latest_before,
            "latest_complete_before": self.latest_complete_before,
            "promotion_policy": "publish_named_candidate_without_pointer_update",
        }


@dataclass(frozen=True)
class BoutKinematicsComputePlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    target_run_path: Path
    output_shard_rows: int
    writer_arguments: tuple[str, ...]

    @property
    def local_run_path(self) -> Path:
        return (
            self.local_zarr
            / "analysis"
            / "bout_kinematics_runs"
            / self.run_name
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": COMPUTE_MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "source_access": "read_only_during_compute",
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "run_name": self.run_name,
            "target_run_path": str(self.target_run_path),
            "output_shard_rows": int(self.output_shard_rows),
            "writer_arguments": list(self.writer_arguments),
            "publication_policy": "atomic_publish_then_complete_and_promote",
        }


def _safe_run_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def build_bout_kinematics_storage_plan(
    source_zarr: str | Path,
    *,
    source_run: str,
    scratch_root: str | Path,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    workers: int = 1,
) -> BoutKinematicsStoragePlan:
    """Build a read-only rematerialization plan without creating paths."""

    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative Zarr.")
    if int(output_shard_rows) <= 0 or int(workers) <= 0:
        raise ValueError("output_shard_rows and workers must be positive.")

    root = open_zarr_root(source, mode="r")
    parent = root.get("analysis/bout_kinematics_runs")
    if not isinstance(parent, zarr.Group):
        raise KeyError("Missing analysis/bout_kinematics_runs parent.")
    requested_source = str(source_run).strip()
    if requested_source in {"", "latest", "latest_complete"}:
        resolved_source = str(parent.attrs.get("latest_complete", "")).strip()
        if not resolved_source:
            raise ValueError("Bout-kinematics parent has no latest_complete pointer.")
    else:
        resolved_source = _safe_run_name(requested_source, label="source run name")
    if resolved_source not in parent:
        raise KeyError(f"Bout-kinematics source run not found: {resolved_source!r}.")
    source_group = parent[resolved_source]
    if str(source_group.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError(f"Source run {resolved_source!r} is not complete.")
    if str(source_group.attrs.get("layout", "")) != LAYOUT_COMPACT_TABULAR_V2:
        raise ValueError(
            "Storage rematerialization currently requires compact_tabular_v2; "
            f"found {source_group.attrs.get('layout')!r}."
        )

    target_name = _safe_run_name(run_name, label="target run name")
    target = source / "analysis" / "bout_kinematics_runs" / target_name
    if target.exists() or target_name in parent:
        raise FileExistsError(f"Refusing to replace existing target run: {target}")
    return BoutKinematicsStoragePlan(
        source_zarr=source,
        source_run_name=resolved_source,
        source_run_path=(
            source / "analysis" / "bout_kinematics_runs" / resolved_source
        ),
        scratch_root=scratch,
        local_run_path=scratch / "bout-kinematics-run",
        run_name=target_name,
        target_run_path=target,
        output_shard_rows=int(output_shard_rows),
        workers=int(workers),
        latest_before=(
            str(parent.attrs.get("latest"))
            if parent.attrs.get("latest") is not None
            else None
        ),
        latest_complete_before=(
            str(parent.attrs.get("latest_complete"))
            if parent.attrs.get("latest_complete") is not None
            else None
        ),
    )


def build_bout_kinematics_compute_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    writer_arguments: Sequence[str] = (),
) -> BoutKinematicsComputePlan:
    """Build a read-only fresh-computation plan without creating paths."""

    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative Zarr.")
    if int(output_shard_rows) <= 0:
        raise ValueError("output_shard_rows must be positive.")
    name = _safe_run_name(run_name, label="target run name")
    forwarded = tuple(str(value) for value in writer_arguments)
    forbidden = sorted(
        argument.split("=", 1)[0]
        for argument in forwarded
        if argument.split("=", 1)[0] in MANAGED_COMPUTE_WRITER_ARGUMENTS
    )
    if forbidden:
        raise ValueError(
            "Bout-kinematics compute materializer owns these writer arguments: "
            + ", ".join(forbidden)
        )
    target = source / "analysis" / "bout_kinematics_runs" / name
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target}"
        )
    return BoutKinematicsComputePlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "bout-output.zarr",
        run_name=name,
        target_run_path=target,
        output_shard_rows=int(output_shard_rows),
        writer_arguments=forwarded,
    )


def _iter_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, child_prefix)


def _payload_stats(path: Path) -> dict[str, int]:
    metadata_names = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}
    file_count = 0
    metadata_file_count = 0
    payload_file_count = 0
    physical_bytes = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        file_count += 1
        physical_bytes += int(candidate.stat().st_size)
        if candidate.name in metadata_names:
            metadata_file_count += 1
        else:
            payload_file_count += 1
    return {
        "file_count": int(file_count),
        "metadata_file_count": int(metadata_file_count),
        "payload_file_count": int(payload_file_count),
        "physical_bytes": int(physical_bytes),
    }


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        json_attr_safe(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_digest(records: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(records.shape).encode("ascii"))
    names = records.dtype.names or ()
    digest.update(json.dumps(list(names)).encode("utf-8"))
    if not names:
        values = np.ascontiguousarray(records)
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(values.tobytes(order="C"))
        return digest.hexdigest()
    for name in names:
        values = np.ascontiguousarray(records[name])
        digest.update(str(name).encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _logical_fingerprint(group: zarr.Group) -> dict[str, Any]:
    records_by_level, _level_attrs, table_attrs = resolve_bout_kinematics_tables(
        group
    )
    levels: dict[str, Any] = {}
    for level, records in sorted(records_by_level.items()):
        levels[str(level)] = {
            "row_count": int(len(records)),
            "field_names": list(records.dtype.names or ()),
            "records_sha256": _records_digest(records),
            "table_attrs_sha256": _canonical_json_digest(table_attrs.get(level, {})),
        }
    scientific_attrs = {
        key: group.attrs.get(key)
        for key in (
            "schema_id",
            "schema_version",
            "method",
            "method_version",
            "layout",
            "parameters",
            "source_refs",
            "heading_levels",
            "analysis_levels",
            "default_heading_level",
        )
    }
    payload = {
        "levels": levels,
        "scientific_attrs_sha256": _canonical_json_digest(scientific_attrs),
    }
    payload["logical_sha256"] = _canonical_json_digest(payload)
    return payload


def _validate_bout_run(
    path: Path,
    *,
    expected_logical_sha256: str,
    expected_run_name: str | None,
    require_small_run_profile: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    group = open_zarr_root(path, mode="r")
    if str(group.attrs.get("palette_run_completion_status", "")) != "complete":
        errors.append("run is not complete")
    if str(group.attrs.get("layout", "")) != LAYOUT_COMPACT_TABULAR_V2:
        errors.append("run is not compact_tabular_v2")
    if expected_run_name is not None and str(
        group.attrs.get("palette_run_name", "")
    ) != str(expected_run_name):
        errors.append("palette_run_name mismatch")

    logical = _logical_fingerprint(group)
    if str(logical["logical_sha256"]) != str(expected_logical_sha256):
        errors.append("resolver-level logical fingerprint mismatch")
    if not logical["levels"]:
        errors.append("resolver returned no logical analysis levels")

    array_count = 0
    sharded_array_count = 0
    regular_array_count = 0
    for array_path, array in _iter_arrays(group):
        array_count += 1
        chunks = tuple(int(value) for value in array.chunks)
        shards = getattr(array, "shards", None)
        logical_row_chunks = (
            int(math.ceil(int(array.shape[0]) / int(chunks[0])))
            if int(array.ndim) >= 1 and int(array.shape[0]) > 0
            else 0
        )
        if shards is not None:
            sharded_array_count += 1
            outer = tuple(int(value) for value in shards)
            if any(outer[index] % chunks[index] for index in range(len(chunks))):
                errors.append(f"{array_path}: shard grid is not chunk aligned")
            if require_small_run_profile and logical_row_chunks <= 1:
                errors.append(f"{array_path}: single-chunk array should remain regular")
        else:
            regular_array_count += 1
            if require_small_run_profile and logical_row_chunks > 1:
                errors.append(f"{array_path}: multi-chunk array is not sharded")

    layout = group.attrs.get("physical_storage_layout")
    if require_small_run_profile:
        if not isinstance(layout, Mapping):
            errors.append("missing physical_storage_layout")
        elif str(layout.get("shard_policy")) != SHARD_POLICY_MULTI_CHUNK_CAPPED:
            errors.append("unexpected physical_storage_layout shard policy")
    return {
        "valid": not errors,
        "errors": errors,
        "logical_fingerprint": logical,
        "array_count": int(array_count),
        "sharded_array_count": int(sharded_array_count),
        "regular_array_count": int(regular_array_count),
        "storage": _payload_stats(path),
    }


def _copy_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "schema_id",
            "status",
            "source_run",
            "destination_run",
            "requested_shard_rows",
            "effective_shard_rows",
            "shard_policy",
            "worker_count",
            "worker_task_count",
            "worker_ownership",
            "array_count",
            "sharded_array_count",
            "regular_array_count",
            "decoded_bytes_copied",
            "duration_seconds",
            "decoded_mib_per_second",
            "exact_decoded_validation",
        )
    }


def publish_bout_kinematics_candidate(
    plan: BoutKinematicsStoragePlan,
    *,
    source_logical_sha256: str,
    materialization_payload: Mapping[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    """Atomically publish one validated candidate without moving pointers."""

    def validate(path: Path) -> dict[str, Any]:
        return _validate_bout_run(
            path,
            expected_logical_sha256=source_logical_sha256,
            expected_run_name=plan.run_name,
            require_small_run_profile=True,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "bout_kinematics_runs",
            ),
        )

    def complete(
        _root: zarr.Group,
        _parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        mark_run_complete(
            run_group,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=run_group.attrs.get("run_provenance"),
        )

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/bout_kinematics_runs"]
        if parent.attrs.get("latest") != plan.latest_before or parent.attrs.get(
            "latest_complete"
        ) != plan.latest_complete_before:
            raise RuntimeError(
                "Non-promoting publication changed bout-kinematics parent pointers."
            )
        candidate = parent.get(plan.run_name)
        if not isinstance(candidate, zarr.Group):
            raise RuntimeError("Published bout-kinematics candidate is missing.")

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix=COMMON_LOCK_SUFFIX,
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_small_run_sharding_atomic_nonpromoting_publish",
            rollback_policy=(
                "remove_new_target_and_restore_parent_attrs_on_any_failure"
            ),
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "source_run": plan.source_run_name,
            "source_run_path": str(plan.source_run_path),
            "copy_backend": copy_backend,
            "promotion_policy": "named_candidate_only_parent_pointers_unchanged",
            "materialization": json_attr_safe(dict(materialization_payload)),
        },
    )


def publish_computed_bout_kinematics_run(
    plan: BoutKinematicsComputePlan,
    *,
    logical_sha256: str,
    materialization_payload: Mapping[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    """Publish one freshly computed local run and promote it after validation."""

    def validate(path: Path) -> dict[str, Any]:
        return _validate_bout_run(
            path,
            expected_logical_sha256=logical_sha256,
            expected_run_name=plan.run_name,
            require_small_run_profile=True,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "bout_kinematics_runs",
            ),
        )

    def complete(
        _root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=run_group.attrs.get("run_provenance"),
        )

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/bout_kinematics_runs"]
        if str(parent.attrs.get("latest")) != plan.run_name or str(
            parent.attrs.get("latest_complete")
        ) != plan.run_name:
            raise RuntimeError(
                "Bout-kinematics parent pointers were not updated to the "
                "published computed run."
            )

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix=COMMON_LOCK_SUFFIX,
            publish_schema_id=COMPUTE_PUBLISH_SCHEMA_ID,
            policy="node_local_compute_atomic_run_group_publish",
            rollback_policy=(
                "remove_new_target_and_restore_parent_attrs_on_any_failure"
            ),
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "copy_backend": copy_backend,
            "promotion_policy": "completion_last_then_latest_pointer_update",
            "materialization": json_attr_safe(dict(materialization_payload)),
        },
    )


def materialize_bout_kinematics_compute(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    writer_arguments: Sequence[str] = (),
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Compute locally from read-only inputs and atomically publish the run."""

    plan = build_bout_kinematics_compute_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
        output_shard_rows=output_shard_rows,
        writer_arguments=writer_arguments,
    )
    result: dict[str, Any] = {
        "schema_id": COMPUTE_MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        writer_argv = [
            str(plan.source_zarr),
            "--output-zarr-path",
            str(plan.local_zarr),
            "--run-name",
            plan.run_name,
            "--output-shard-rows",
            str(plan.output_shard_rows),
            *plan.writer_arguments,
        ]
        compute_started = time.perf_counter()
        bout_writer.main(writer_argv)
        compute_seconds = float(time.perf_counter() - compute_started)

        local_group = open_zarr_root(plan.local_run_path, mode="a")
        logical = _logical_fingerprint(local_group)
        local_validation = _validate_bout_run(
            plan.local_run_path,
            expected_logical_sha256=str(logical["logical_sha256"]),
            expected_run_name=plan.run_name,
            require_small_run_profile=True,
        )
        if not local_validation["valid"]:
            raise RuntimeError(
                f"Local computed bout-kinematics run is invalid: {local_validation}"
            )
        local_payload = {
            "schema_id": COMPUTE_MATERIALIZATION_SCHEMA_ID,
            "source_access": "authoritative_zarr_read_only",
            "compute_output": "node_local_zarr",
            "scientific_values_recomputed": True,
            "compute_duration_seconds": compute_seconds,
            "writer_arguments": writer_argv,
            "logical_fingerprint": logical,
            "local_validation": local_validation,
        }
        local_group.attrs["node_local_materialization"] = json_attr_safe(
            local_payload
        )
        publish = publish_computed_bout_kinematics_run(
            plan,
            logical_sha256=str(logical["logical_sha256"]),
            materialization_payload=local_payload,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
                "local_materialization": local_payload,
                "publish": publish,
                "promoted": True,
            }
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def promote_bout_kinematics_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    apply: bool = False,
    approved_by: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Validate and promote an already-published complete candidate in place."""

    source = Path(source_zarr).expanduser().resolve()
    name = _safe_run_name(run_name, label="candidate run name")
    candidate_path = source / "analysis" / "bout_kinematics_runs" / name
    if not candidate_path.is_dir():
        raise FileNotFoundError(f"Bout-kinematics candidate not found: {candidate_path}")

    def inspect_candidate() -> tuple[dict[str, Any], dict[str, Any]]:
        group = open_zarr_root(candidate_path, mode="r")
        logical = _logical_fingerprint(group)
        validation = _validate_bout_run(
            candidate_path,
            expected_logical_sha256=str(logical["logical_sha256"]),
            expected_run_name=name,
            require_small_run_profile=True,
        )
        if not validation["valid"]:
            raise RuntimeError(
                f"Refusing to promote invalid bout-kinematics candidate: {validation}"
            )
        rematerialization = group.attrs.get("storage_rematerialization")
        if isinstance(rematerialization, Mapping):
            expected = rematerialization.get("source_logical_fingerprint")
            expected_sha = (
                expected.get("logical_sha256")
                if isinstance(expected, Mapping)
                else None
            )
            if expected_sha is not None and str(expected_sha) != str(
                logical["logical_sha256"]
            ):
                raise RuntimeError(
                    "Candidate no longer matches its persisted source logical fingerprint."
                )
        return logical, validation

    root = open_zarr_root(source, mode="r")
    parent = root["analysis/bout_kinematics_runs"]
    logical, validation = inspect_candidate()
    result: dict[str, Any] = {
        "schema_id": PROMOTION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "run_name": name,
        "candidate_path": str(candidate_path),
        "logical_sha256": str(logical["logical_sha256"]),
        "validation": validation,
        "latest_before": parent.attrs.get("latest"),
        "latest_complete_before": parent.attrs.get("latest_complete"),
    }
    if not apply:
        return result

    lock_path = source.parent / f".{source.name}.{COMMON_LOCK_SUFFIX}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            write_root = open_zarr_root(source, mode="a")
            write_parent = write_root["analysis/bout_kinematics_runs"]
            write_candidate = write_parent[name]
            parent_before = dict(write_parent.attrs)
            candidate_before = dict(write_candidate.attrs)
            try:
                logical, validation = inspect_candidate()
                promoted_at = datetime.now(timezone.utc).isoformat()
                receipt = {
                    "schema_id": PROMOTION_SCHEMA_ID,
                    "promoted_at_utc": promoted_at,
                    "approved_by": str(
                        approved_by
                        or os.environ.get("USER")
                        or getpass.getuser()
                    ),
                    "host": socket.gethostname(),
                    "note": str(note or ""),
                    "run_name": name,
                    "previous_latest": parent_before.get("latest"),
                    "previous_latest_complete": parent_before.get(
                        "latest_complete"
                    ),
                    "logical_sha256": str(logical["logical_sha256"]),
                }
                write_candidate.attrs["storage_promotion"] = json_attr_safe(receipt)
                write_parent.attrs["latest_complete"] = name
                write_parent.attrs["latest"] = name

                verify_root = open_zarr_root(source, mode="r")
                verify_parent = verify_root["analysis/bout_kinematics_runs"]
                if str(verify_parent.attrs.get("latest")) != name or str(
                    verify_parent.attrs.get("latest_complete")
                ) != name:
                    raise RuntimeError(
                        "Bout-kinematics candidate pointer verification failed."
                    )
            except BaseException:
                write_parent.attrs.clear()
                write_parent.attrs.update(parent_before)
                write_candidate.attrs.clear()
                write_candidate.attrs.update(candidate_before)
                raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    result.update(
        {
            "status": "complete",
            "promoted": True,
            "promotion": receipt,
            "latest_after": name,
            "latest_complete_after": name,
            "validation": validation,
        }
    )
    return result


def materialize_bout_kinematics_storage(
    source_zarr: str | Path,
    *,
    source_run: str,
    scratch_root: str | Path,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    workers: int = 1,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    plan = build_bout_kinematics_storage_plan(
        source_zarr,
        source_run=source_run,
        scratch_root=scratch_root,
        run_name=run_name,
        output_shard_rows=output_shard_rows,
        workers=workers,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "promotion_requested": False,
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        source_group = open_zarr_root(plan.source_run_path, mode="r")
        source_logical = _logical_fingerprint(source_group)
        source_validation = _validate_bout_run(
            plan.source_run_path,
            expected_logical_sha256=str(source_logical["logical_sha256"]),
            expected_run_name=plan.source_run_name,
            require_small_run_profile=False,
        )
        if not source_validation["valid"]:
            raise RuntimeError(
                f"Source bout-kinematics run is invalid: {source_validation}"
            )

        copied = copy_completed_run_to_sharded(
            plan.source_run_path,
            plan.local_run_path,
            row_count_array=None,
            shard_rows=plan.output_shard_rows,
            shard_policy=SHARD_POLICY_MULTI_CHUNK_CAPPED,
            workers=plan.workers,
        )
        provenance = build_run_provenance(
            command=command or " ".join(sys.argv) or "unknown",
            params={
                "operation": "storage_only_rematerialization",
                "scientific_values_recomputed": False,
                "output_shard_rows": int(plan.output_shard_rows),
                "shard_policy": SHARD_POLICY_MULTI_CHUNK_CAPPED,
                "promotion_requested": False,
            },
            input_run_ids={
                "source_bout_kinematics_run": plan.source_run_name,
                "source_logical_sha256": source_logical["logical_sha256"],
            },
        )
        materialization_payload = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_access": "authoritative_completed_run_read_only",
            "source_run": plan.source_run_name,
            "source_run_path": str(plan.source_run_path),
            "source_logical_fingerprint": source_logical,
            "scientific_values_recomputed": False,
            "physical_layout_only_change": True,
            "promotion_requested": False,
            "sharded_copy": _copy_summary(copied),
        }
        local_group = open_zarr_root(plan.local_run_path, mode="a")
        local_group.attrs["palette_run_name"] = plan.run_name
        local_group.attrs["run_provenance"] = json_attr_safe(provenance)
        local_group.attrs["storage_rematerialization"] = json_attr_safe(
            materialization_payload
        )
        local_group.attrs["node_local_materialization"] = json_attr_safe(
            materialization_payload
        )

        local_validation = _validate_bout_run(
            plan.local_run_path,
            expected_logical_sha256=str(source_logical["logical_sha256"]),
            expected_run_name=plan.run_name,
            require_small_run_profile=True,
        )
        if not local_validation["valid"]:
            raise RuntimeError(
                f"Local rematerialized bout-kinematics run is invalid: {local_validation}"
            )
        publish = publish_bout_kinematics_candidate(
            plan,
            source_logical_sha256=str(source_logical["logical_sha256"]),
            materialization_payload=materialization_payload,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
                "source_validation": source_validation,
                "local_validation": local_validation,
                "local_materialization": materialization_payload,
                "publish": publish,
                "promoted": False,
            }
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_bout_kinematics_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_bout_kinematics_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--compute",
        action="store_true",
        help="Compute a fresh node-local run and atomically publish/promote it.",
    )
    mode.add_argument(
        "--promote-candidate",
        action="store_true",
        help="Validate and promote an existing named candidate without copying it.",
    )
    parser.add_argument("--source-run", default="latest_complete")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--note")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    if remaining and not args.compute:
        parser.error("writer arguments after -- are supported only with --compute")
    if remaining and remaining[0] != "--":
        parser.error(
            "unrecognized materializer arguments; place bout-writer arguments after --"
        )
    writer_arguments = tuple(remaining)
    if writer_arguments[:1] == ("--",):
        writer_arguments = writer_arguments[1:]

    if args.promote_candidate:
        result = promote_bout_kinematics_candidate(
            args.zarr_path,
            run_name=args.run_name,
            apply=bool(args.apply),
            approved_by=args.approved_by,
            note=args.note,
        )
    elif args.compute:
        result = materialize_bout_kinematics_compute(
            args.zarr_path,
            scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
            run_name=args.run_name,
            output_shard_rows=args.output_shard_rows,
            writer_arguments=writer_arguments,
            copy_backend=args.copy_backend,
            apply=bool(args.apply),
            keep_scratch=bool(args.keep_scratch),
        )
    else:
        result = materialize_bout_kinematics_storage(
            args.zarr_path,
            source_run=args.source_run,
            scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
            run_name=args.run_name,
            output_shard_rows=args.output_shard_rows,
            workers=args.workers,
            copy_backend=args.copy_backend,
            apply=bool(args.apply),
            keep_scratch=bool(args.keep_scratch),
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
