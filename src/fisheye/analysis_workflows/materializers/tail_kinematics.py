"""Stage subject-shape inputs locally, materialize tail kinematics, and publish.

The authoritative recording Zarr is never used as the compute output store.
Required subject-shape physical files are copied to a minimal node-local Zarr,
the bounded writer runs there, and the completed run group is copied to a
hidden shared-storage sibling before an atomic rename and parent-pointer update.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import zarr

from ...analysis.tail_kinematics_runs import (
    DEFAULT_BLOCK_ROWS,
    DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    ROW_LINEAGE_NAMES,
    SOURCE_REVISION_ARRAY_NAMES,
    SUBJECT_SHAPE_BODY_ARRAY_NAMES,
    SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES,
    TAIL_KINEMATICS_COMPUTE_KERNEL,
    TAIL_KINEMATICS_SCHEMA_ID,
    _resolve_tail_kinematics_sources,
    write_tail_kinematics_run_group,
)
from ...shared.json_safety import json_attr_safe
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent

MATERIALIZATION_SCHEMA_ID = "palette.tail_kinematics_materialization.v1"
STAGING_SCHEMA_ID = "palette.tail_kinematics_source_staging.v1"
PUBLISH_SCHEMA_ID = "palette.tail_kinematics_run_publish.v1"
GROUP_METADATA_NAMES = ("zarr.json", ".zgroup", ".zattrs")
DEFAULT_CAPACITY_MARGIN_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PhysicalFile:
    """One selected physical source file relative to a Zarr root."""

    relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class TailKinematicsMaterializationPlan:
    """Immutable plan for one staged tail-kinematics materialization."""

    source_zarr: Path
    scratch_root: Path
    staged_zarr: Path
    shape_run: str
    run_name: str
    row_count: int
    tail_angle_sample_count: int
    requested_block_rows: int
    execution_backend: str
    requested_num_workers: int
    selected_paths: tuple[str, ...]
    physical_files: tuple[PhysicalFile, ...]
    source_bytes: int
    estimated_output_bytes: int
    inventory_sha256: str
    source_metadata_sha256: str
    source_contract: dict[str, Any]

    @property
    def files_manifest_path(self) -> Path:
        return self.scratch_root / "source-files.txt"

    @property
    def staging_manifest_path(self) -> Path:
        return self.scratch_root / "staging-manifest.json"

    @property
    def local_run_path(self) -> Path:
        return self.staged_zarr / "analysis" / "tail_kinematics_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "tail_kinematics_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "staged_zarr": str(self.staged_zarr),
            "shape_run": self.shape_run,
            "run_name": self.run_name,
            "row_count": int(self.row_count),
            "tail_angle_sample_count": int(self.tail_angle_sample_count),
            "requested_block_rows": int(self.requested_block_rows),
            "execution_backend": self.execution_backend,
            "requested_num_workers": int(self.requested_num_workers),
            "selected_paths": list(self.selected_paths),
            "physical_file_count": len(self.physical_files),
            "source_bytes": int(self.source_bytes),
            "estimated_output_bytes": int(self.estimated_output_bytes),
            "inventory_sha256": self.inventory_sha256,
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_contract": json_attr_safe(self.source_contract),
            "files_manifest_path": str(self.files_manifest_path),
            "staging_manifest_path": str(self.staging_manifest_path),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe tail-kinematics run name: {run_name!r}.")
    return value


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _add_group_metadata(source_zarr: Path, relative_group: str, selected: set[Path]) -> None:
    group = source_zarr if relative_group in {"", "."} else source_zarr / relative_group
    for name in GROUP_METADATA_NAMES:
        candidate = group / name
        if candidate.is_file():
            selected.add(candidate)


def _add_array_tree(
    source_zarr: Path,
    relative_array: str,
    selected: set[Path],
    selected_paths: list[str],
    *,
    required: bool,
) -> None:
    path = source_zarr / relative_array
    if not path.is_dir():
        if required:
            raise FileNotFoundError(f"Required staged source array is missing: {path}")
        return
    selected_paths.append(relative_array)
    for candidate in path.rglob("*"):
        if candidate.is_file():
            selected.add(candidate)


def _inventory_digest(files: Sequence[PhysicalFile]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(item.size_bytes)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata_content_digest(source_zarr: Path, files: Sequence[PhysicalFile]) -> str:
    """Fingerprint selected Zarr metadata content without hashing data chunks."""

    digest = hashlib.sha256()
    metadata_names = set(GROUP_METADATA_NAMES)
    for item in files:
        if Path(item.relative_path).name not in metadata_names:
            continue
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        with (source_zarr / item.relative_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def _estimated_output_bytes(row_count: int, sample_count: int) -> int:
    # Six K-wide float32 surfaces, ten scalar float32 surfaces, validity,
    # fixed-width reasons, frame index, and a conservative lineage allowance.
    per_row = (6 * int(sample_count) * 4) + (10 * 4) + 1 + 64 + 8 + 64
    return int(row_count) * int(per_row)


def _selected_source_files(
    source_zarr: Path,
    *,
    shape_run: str,
) -> tuple[tuple[str, ...], tuple[PhysicalFile, ...]]:
    selected: set[Path] = set()
    selected_paths: list[str] = []
    run_prefix = f"analysis/subject_shape_runs/{shape_run}"

    for group_path in (
        ".",
        "analysis",
        "analysis/subject_shape_runs",
        run_prefix,
        f"{run_prefix}/components",
        f"{run_prefix}/components/subject_body",
        f"{run_prefix}/body_frame",
        f"{run_prefix}/row_index",
        f"{run_prefix}/source_refined_subject_masks",
    ):
        _add_group_metadata(source_zarr, group_path, selected)

    required_body = {
        "tail_sample_s",
        "tail_sample_xy",
        "tail_tangent_xy",
        "tail_curvature_px_inv",
        "tail_sample_valid",
        "bspline_valid",
        "tail_base_xy",
    }
    for name in SUBJECT_SHAPE_BODY_ARRAY_NAMES:
        _add_array_tree(
            source_zarr,
            f"{run_prefix}/components/subject_body/{name}",
            selected,
            selected_paths,
            required=name in required_body,
        )
    required_body_frame = {"forward_axis_xy", "left_axis_xy", "valid"}
    for name in SUBJECT_SHAPE_BODY_FRAME_ARRAY_NAMES:
        _add_array_tree(
            source_zarr,
            f"{run_prefix}/body_frame/{name}",
            selected,
            selected_paths,
            required=name in required_body_frame,
        )
    for name in ROW_LINEAGE_NAMES:
        _add_array_tree(
            source_zarr,
            f"{run_prefix}/row_index/{name}",
            selected,
            selected_paths,
            required=False,
        )
    for name in SOURCE_REVISION_ARRAY_NAMES:
        _add_array_tree(
            source_zarr,
            f"{run_prefix}/source_refined_subject_masks/{name}",
            selected,
            selected_paths,
            required=False,
        )

    files = tuple(
        PhysicalFile(relative_path=_relative_path(source_zarr, path), size_bytes=int(path.stat().st_size))
        for path in sorted(selected)
    )
    if not files:
        raise RuntimeError(f"No physical source files selected from {source_zarr}.")
    return tuple(selected_paths), files


def build_tail_kinematics_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    shape_run: str | None,
    run_name: str,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
) -> TailKinematicsMaterializationPlan:
    """Build a read-only physical-file staging plan for one recording."""

    source = Path(source_zarr).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    scratch = Path(scratch_root).expanduser().resolve()
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative source Zarr.")
    if int(tail_angle_sample_count) < 2:
        raise ValueError("tail_angle_sample_count must be >= 2.")
    if int(block_rows) <= 0:
        raise ValueError("block_rows must be positive.")
    backend = str(execution_backend).strip().lower()
    if backend not in {"serial", "process_shards"}:
        raise ValueError(f"Unsupported execution backend: {execution_backend!r}.")
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive.")

    root = open_zarr_root(source, mode="r")
    resolved_shape_run, shape_group, sources = _resolve_tail_kinematics_sources(root, shape_run)
    target_run = source / "analysis" / "tail_kinematics_runs" / _validate_run_name(run_name)
    if target_run.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target_run}")
    selected_paths, files = _selected_source_files(source, shape_run=resolved_shape_run)
    return TailKinematicsMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        staged_zarr=scratch / "source-subset.zarr",
        shape_run=resolved_shape_run,
        run_name=_validate_run_name(run_name),
        row_count=int(sources.row_count),
        tail_angle_sample_count=int(tail_angle_sample_count),
        requested_block_rows=int(block_rows),
        execution_backend=backend,
        requested_num_workers=int(num_workers),
        selected_paths=selected_paths,
        physical_files=files,
        source_bytes=sum(int(item.size_bytes) for item in files),
        estimated_output_bytes=_estimated_output_bytes(
            int(sources.row_count),
            int(tail_angle_sample_count),
        ),
        inventory_sha256=_inventory_digest(files),
        source_metadata_sha256=_metadata_content_digest(source, files),
        source_contract=json_attr_safe(
            {
                key: shape_group.attrs[key]
                for key in (
                    "schema_id",
                    "schema_version",
                    "method",
                    "method_version",
                    "palette_run_completion_status",
                    "source_refined_subject_masks_run",
                    "body_frame_schema_id",
                    "tail_geometry_schema_id",
                )
                if key in shape_group.attrs
            }
        ),
    )


def _write_files_manifest(plan: TailKinematicsMaterializationPlan) -> None:
    plan.files_manifest_path.write_text(
        "".join(f"{item.relative_path}\n" for item in plan.physical_files),
        encoding="utf-8",
    )


def _copy_selected_files_python(plan: TailKinematicsMaterializationPlan) -> None:
    for item in plan.physical_files:
        source = plan.source_zarr / item.relative_path
        target = plan.staged_zarr / item.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_selected_files_rsync(plan: TailKinematicsMaterializationPlan) -> None:
    command = [
        "rsync",
        "--archive",
        f"--files-from={plan.files_manifest_path}",
        f"{plan.source_zarr}/",
        f"{plan.staged_zarr}/",
    ]
    subprocess.run(command, check=True)


def _validate_file_inventory(root: Path, expected: Sequence[PhysicalFile]) -> dict[str, Any]:
    observed: list[PhysicalFile] = []
    missing: list[str] = []
    size_mismatches: list[str] = []
    for item in expected:
        path = root / item.relative_path
        if not path.is_file():
            missing.append(item.relative_path)
            continue
        size = int(path.stat().st_size)
        observed.append(PhysicalFile(relative_path=item.relative_path, size_bytes=size))
        if size != int(item.size_bytes):
            size_mismatches.append(item.relative_path)
    observed_digest = _inventory_digest(observed)
    return {
        "valid": not missing and not size_mismatches and len(observed) == len(expected),
        "expected_file_count": len(expected),
        "observed_file_count": len(observed),
        "expected_bytes": sum(int(item.size_bytes) for item in expected),
        "observed_bytes": sum(int(item.size_bytes) for item in observed),
        "expected_inventory_sha256": _inventory_digest(expected),
        "observed_inventory_sha256": observed_digest,
        "missing": missing,
        "size_mismatches": size_mismatches,
    }


def stage_tail_kinematics_sources(
    plan: TailKinematicsMaterializationPlan,
    *,
    copy_backend: str = "rsync",
    check_capacity: bool = True,
) -> dict[str, Any]:
    """Copy the complete selected physical source surface into node-local scratch."""

    if plan.staged_zarr.exists():
        raise FileExistsError(f"Refusing existing staged Zarr: {plan.staged_zarr}")
    plan.scratch_root.mkdir(parents=True, exist_ok=False)
    plan.staged_zarr.mkdir(parents=True)
    _write_files_manifest(plan)
    required_bytes = int(plan.source_bytes + plan.estimated_output_bytes + DEFAULT_CAPACITY_MARGIN_BYTES)
    free_bytes = int(shutil.disk_usage(plan.scratch_root).free)
    if check_capacity and free_bytes < required_bytes:
        raise OSError(
            f"Insufficient scratch capacity: need approximately {required_bytes} bytes, "
            f"found {free_bytes} bytes at {plan.scratch_root}."
        )

    started_at = _utc_now()
    started = time.perf_counter()
    if copy_backend == "rsync":
        _copy_selected_files_rsync(plan)
    elif copy_backend == "python":
        _copy_selected_files_python(plan)
    else:
        raise ValueError(f"Unsupported copy backend: {copy_backend!r}.")
    duration = float(time.perf_counter() - started)
    validation = _validate_file_inventory(plan.staged_zarr, plan.physical_files)
    if not bool(validation["valid"]):
        raise RuntimeError(f"Staged source inventory validation failed: {validation}")

    staged_root = open_zarr_root(plan.staged_zarr, mode="r")
    staged_shape_run, _shape_group, staged_sources = _resolve_tail_kinematics_sources(
        staged_root,
        plan.shape_run,
    )
    if staged_shape_run != plan.shape_run or int(staged_sources.row_count) != int(plan.row_count):
        raise RuntimeError("Staged subject-shape logical validation did not match the staging plan.")

    payload = {
        "schema_id": STAGING_SCHEMA_ID,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "duration_seconds": duration,
        "mib_per_second": (
            (int(plan.source_bytes) / (1024 * 1024)) / duration if duration > 0.0 else None
        ),
        "copy_backend": copy_backend,
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "source_zarr": str(plan.source_zarr),
        "staged_zarr": str(plan.staged_zarr),
        "shape_run": plan.shape_run,
        "row_count": int(plan.row_count),
        "selected_paths": list(plan.selected_paths),
        "source_metadata_sha256": plan.source_metadata_sha256,
        "source_contract": json_attr_safe(plan.source_contract),
        "inventory": validation,
        "capacity": {
            "check_enabled": bool(check_capacity),
            "free_bytes_before_copy": free_bytes,
            "required_bytes_estimate": required_bytes,
            "estimated_output_bytes": int(plan.estimated_output_bytes),
            "margin_bytes": DEFAULT_CAPACITY_MARGIN_BYTES,
        },
    }
    _write_json_atomic(plan.staging_manifest_path, payload)
    return payload


def _tree_inventory(root: Path) -> tuple[tuple[PhysicalFile, ...], str]:
    files = tuple(
        PhysicalFile(relative_path=path.relative_to(root).as_posix(), size_bytes=int(path.stat().st_size))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    return files, _inventory_digest(files)


def _copy_run_tree(source: Path, target: Path, *, copy_backend: str) -> None:
    if copy_backend == "rsync":
        target.mkdir(parents=True)
        subprocess.run(["rsync", "--archive", f"{source}/", f"{target}/"], check=True)
    elif copy_backend == "python":
        shutil.copytree(source, target)
    else:
        raise ValueError(f"Unsupported copy backend: {copy_backend!r}.")


def _validate_tail_run(path: Path, *, row_count: int, sample_count: int) -> dict[str, Any]:
    group = open_zarr_root(path, mode="r")
    attrs = group.attrs
    errors: list[str] = []
    if str(attrs.get("schema_id")) != TAIL_KINEMATICS_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if str(attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if str(attrs.get("compute_kernel")) != TAIL_KINEMATICS_COMPUTE_KERNEL:
        errors.append("compute_kernel mismatch")
    expected_shapes = {
        "valid": (int(row_count),),
        "failure_reason_bytes": (int(row_count), 64),
        "tail_angle_sample_s": (int(sample_count),),
        "tail_angle_sample_xy": (int(row_count), int(sample_count), 2),
        "tail_angle_rad": (int(row_count), int(sample_count)),
        "tail_angle_deg": (int(row_count), int(sample_count)),
        "tail_curvature_px_inv": (int(row_count), int(sample_count)),
        "tail_tip_angle_rad": (int(row_count),),
        "integrated_abs_tail_curvature": (int(row_count),),
    }
    for name, expected_shape in expected_shapes.items():
        item = group.get(name)
        if not isinstance(item, zarr.Array):
            errors.append(f"missing array {name}")
        elif tuple(int(value) for value in item.shape) != expected_shape:
            errors.append(f"shape mismatch for {name}")
    valid_count = int(attrs.get("valid_row_count", -1))
    invalid_count = int(attrs.get("invalid_row_count", -1))
    if valid_count + invalid_count != int(row_count):
        errors.append("valid/invalid accounting mismatch")
    if int(attrs.get("completed_block_count", -1)) != int(attrs.get("block_count", -2)):
        errors.append("block completion mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(row_count),
        "sample_count": int(sample_count),
        "valid_row_count": valid_count,
        "invalid_row_count": invalid_count,
        "output_row_chunk": attrs.get("output_row_chunk"),
        "output_shard_rows": attrs.get("output_shard_rows"),
    }


def _serialized_authoritative_publish(function):
    """Serialize run-group rename and parent-pointer mutation per recording Zarr."""

    @functools.wraps(function)
    def wrapped(plan: TailKinematicsMaterializationPlan, *args, **kwargs):
        lock_path = plan.source_zarr.parent / (
            f".{plan.source_zarr.name}.tail-kinematics-publish.lock"
        )
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                return function(plan, *args, **kwargs)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    return wrapped


@_serialized_authoritative_publish
def publish_tail_kinematics_run(
    plan: TailKinematicsMaterializationPlan,
    *,
    staging_payload: dict[str, Any],
    copy_backend: str = "rsync",
) -> dict[str, Any]:
    """Validate and atomically publish one completed local run group."""

    source_run = plan.local_run_path
    if not source_run.is_dir():
        raise FileNotFoundError(f"Local materialized run not found: {source_run}")
    local_validation = _validate_tail_run(
        source_run,
        row_count=plan.row_count,
        sample_count=plan.tail_angle_sample_count,
    )
    if not bool(local_validation["valid"]):
        raise RuntimeError(f"Local tail run validation failed: {local_validation}")

    canonical_root = open_zarr_root(plan.source_zarr, mode="a")
    analysis = canonical_root.require_group("analysis")
    parent = require_runs_parent(analysis, "tail_kinematics_runs")
    if plan.run_name in parent or plan.target_run_path.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {plan.target_run_path}")

    target_parent = plan.target_run_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    temporary = target_parent / f".{plan.run_name}.publish_tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Refusing existing publish temporary path: {temporary}")

    source_files, source_digest = _tree_inventory(source_run)
    copy_started = time.perf_counter()
    try:
        _copy_run_tree(source_run, temporary, copy_backend=copy_backend)
        temporary_validation = _validate_file_inventory(temporary, source_files)
        logical_validation = _validate_tail_run(
            temporary,
            row_count=plan.row_count,
            sample_count=plan.tail_angle_sample_count,
        )
        if not bool(temporary_validation["valid"]) or not bool(logical_validation["valid"]):
            raise RuntimeError(
                "Temporary publish validation failed: "
                f"inventory={temporary_validation}, logical={logical_validation}"
            )
        os.replace(temporary, plan.target_run_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    copy_duration = float(time.perf_counter() - copy_started)

    canonical_root = open_zarr_root(plan.source_zarr, mode="a")
    parent = require_runs_parent(canonical_root.require_group("analysis"), "tail_kinematics_runs")
    run_group = parent[plan.run_name]
    publish_payload = {
        "schema_id": PUBLISH_SCHEMA_ID,
        "policy": "node_local_source_and_output_atomic_run_group_publish",
        "serialization_policy": "per_recording_advisory_file_lock",
        "published_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "source_zarr": str(plan.source_zarr),
        "staged_zarr": str(plan.staged_zarr),
        "source_run_path": str(source_run),
        "target_run_path": str(plan.target_run_path),
        "copy_backend": copy_backend,
        "copy_duration_seconds": copy_duration,
        "physical_file_count": len(source_files),
        "physical_bytes": sum(int(item.size_bytes) for item in source_files),
        "inventory_sha256": source_digest,
        "source_staging": staging_payload,
        "local_validation": local_validation,
    }
    run_group.attrs["cluster_output_staging"] = publish_payload
    command = " ".join(sys.argv) if sys.argv else "unknown"
    mark_run_complete(
        run_group,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=build_run_provenance_from_stage_record(
            run_group.attrs.get("provenance", {}),
            fallback_command=command,
        ),
    )
    final_validation = _validate_tail_run(
        plan.target_run_path,
        row_count=plan.row_count,
        sample_count=plan.tail_angle_sample_count,
    )
    if not bool(final_validation["valid"]):
        raise RuntimeError(f"Published tail run validation failed: {final_validation}")
    publish_payload["final_validation"] = final_validation
    return publish_payload


def materialize_tail_kinematics(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    shape_run: str | None,
    run_name: str,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    check_capacity: bool = True,
    stage_command: str | None = None,
) -> dict[str, Any]:
    """Execute or plan the complete staged materialization workflow."""

    plan = build_tail_kinematics_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        shape_run=shape_run,
        run_name=run_name,
        tail_angle_sample_count=tail_angle_sample_count,
        block_rows=block_rows,
        execution_backend=execution_backend,
        num_workers=num_workers,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        return result

    succeeded = False
    try:
        staging = stage_tail_kinematics_sources(
            plan,
            copy_backend=copy_backend,
            check_capacity=check_capacity,
        )
        local_root = open_zarr_root(plan.staged_zarr, mode="a")
        local_summary = write_tail_kinematics_run_group(
            local_root,
            shape_run=plan.shape_run,
            run_name=plan.run_name,
            tail_angle_sample_count=plan.tail_angle_sample_count,
            block_rows=plan.requested_block_rows,
            execution_backend=plan.execution_backend,
            num_workers=plan.requested_num_workers,
            worker_zarr_path=plan.staged_zarr,
            overwrite=False,
            dry_run=False,
            stage_command=stage_command or (" ".join(sys.argv) if sys.argv else "unknown"),
        )
        local_run = local_root["analysis"]["tail_kinematics_runs"][plan.run_name]
        local_run.attrs["node_local_source_staging"] = staging
        publish = publish_tail_kinematics_run(
            plan,
            staging_payload=staging,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
                "staging": staging,
                "local_materialization": local_summary,
                "publish": publish,
            }
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.is_dir():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_tail_kinematics_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / f"palette_tail_kinematics_{job_id}_{run_name}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage subject-shape arrays locally, materialize tail kinematics, and atomically publish."
    )
    parser.add_argument("zarr_path", type=Path, help="Authoritative Palette analysis Zarr on shared storage.")
    parser.add_argument("--shape-run", help="Exact subject-shape source run; defaults to latest.")
    parser.add_argument("--run-name", required=True, help="New authoritative tail-kinematics run name.")
    parser.add_argument("--scratch-root", type=Path, help="Unique node-local staging directory.")
    parser.add_argument(
        "--tail-angle-sample-count",
        type=int,
        default=DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    )
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument(
        "--execution-backend",
        choices=("serial", "process_shards"),
        default="serial",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true", help="Execute; default is a read-only plan.")
    parser.add_argument("--keep-scratch", action="store_true", help="Keep scratch after successful publication.")
    parser.add_argument("--no-capacity-check", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON report path, normally on shared storage.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    scratch_root = args.scratch_root or _default_scratch_root(args.run_name)
    result = materialize_tail_kinematics(
        args.zarr_path,
        scratch_root=scratch_root,
        shape_run=args.shape_run,
        run_name=args.run_name,
        tail_angle_sample_count=int(args.tail_angle_sample_count),
        block_rows=int(args.block_rows),
        execution_backend=str(args.execution_backend),
        num_workers=int(args.num_workers),
        copy_backend=str(args.copy_backend),
        apply=bool(args.apply),
        keep_scratch=bool(args.keep_scratch),
        check_capacity=not bool(args.no_capacity_check),
    )
    if args.report is not None:
        _write_json_atomic(args.report.expanduser().resolve(), result)
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
