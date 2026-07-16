"""Stage exact eye-angle inputs, compute locally, shard, and publish atomically.

The production eye-angle path consumes completed subject-shape eye geometry and
completed refined keypoints.  Only the physical files backing those resolved
arrays are copied to node-local storage.  The existing scientific writer then
runs entirely against that staged Zarr, its completed output is converted to
indexed Zarr v3 shards with decoded validation, and the shared atomic publisher
installs the result in the authoritative recording.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import zarr

from ...analysis import eye_angle_analysis as eye_writer
from ...shared.eye_geometry_source import EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
from ...shared.json_safety import json_attr_safe
from ...shared.metadata import get_fps
from ...shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent
from ...shared.zarr_sharded_copy import copy_completed_run_to_sharded
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.eye_angle_materialization.v1"
STAGING_SCHEMA_ID = "palette.eye_angle_source_staging.v1"
PUBLISH_SCHEMA_ID = "palette.eye_angle_run_publish.v1"
SOURCE_REVISION_AUDIT_SCHEMA_ID = "palette.eye_angle_source_revision_audit.v1"
GROUP_METADATA_NAMES = ("zarr.json", ".zgroup", ".zattrs")
DEFAULT_CHUNK_ROWS = 8_192
DEFAULT_OUTPUT_SHARD_ROWS = 262_144
DEFAULT_NUM_WORKERS = 8
DEFAULT_SHARD_WORKERS = 8
DEFAULT_NATIVE_THREADS = 1
DEFAULT_CAPACITY_MARGIN_BYTES = 1024 * 1024 * 1024
ESTIMATED_OUTPUT_BYTES_PER_DETECTION = 2_048
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class PhysicalFile:
    """One selected physical source file relative to a Zarr root."""

    relative_path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class EyeAngleMaterializationPlan:
    """Immutable read-only plan for one eye-angle materialization."""

    source_zarr: Path
    scratch_root: Path
    staged_zarr: Path
    sharded_run: Path
    subject_shape_run: str
    keypoint_run: str
    source_keypoint_run: str | None
    run_name: str
    row_count: int
    frame_count: int
    chunk_rows: int
    output_shard_rows: int
    execution_backend: str
    scheduler: str
    num_workers: int
    shard_workers: int
    native_threads: int
    fps: float | None
    fps_source: str
    smoothing_window: int | None
    selected_arrays: tuple[str, ...]
    physical_files: tuple[PhysicalFile, ...]
    source_bytes: int
    estimated_output_bytes: int
    inventory_sha256: str
    revision_inventory_sha256: str
    source_metadata_sha256: str
    source_contract_sha256: str
    source_contracts: dict[str, Any]

    @property
    def files_manifest_path(self) -> Path:
        return self.scratch_root / "source-files.txt"

    @property
    def staging_manifest_path(self) -> Path:
        return self.scratch_root / "staging-manifest.json"

    @property
    def local_run_path(self) -> Path:
        return self.staged_zarr / "analysis" / "eye_angle_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "eye_angle_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return json_attr_safe(
            {
                "schema_id": MATERIALIZATION_SCHEMA_ID,
                "source_zarr": str(self.source_zarr),
                "source_access_policy": "authoritative_shared_read_only",
                "scratch_root": str(self.scratch_root),
                "staged_zarr": str(self.staged_zarr),
                "sharded_run": str(self.sharded_run),
                "local_run_path": str(self.local_run_path),
                "target_run_path": str(self.target_run_path),
                "subject_shape_run": self.subject_shape_run,
                "keypoint_run": self.keypoint_run,
                "source_keypoint_run": self.source_keypoint_run,
                "run_name": self.run_name,
                "row_count": self.row_count,
                "frame_count": self.frame_count,
                "chunk_rows": self.chunk_rows,
                "output_shard_rows": self.output_shard_rows,
                "execution_backend": self.execution_backend,
                "scheduler": self.scheduler,
                "num_workers": self.num_workers,
                "shard_workers": self.shard_workers,
                "native_threads": self.native_threads,
                "fps": self.fps,
                "fps_source": self.fps_source,
                "smoothing_window": self.smoothing_window,
                "selected_arrays": list(self.selected_arrays),
                "physical_file_count": len(self.physical_files),
                "source_bytes": self.source_bytes,
                "estimated_output_bytes": self.estimated_output_bytes,
                "inventory_sha256": self.inventory_sha256,
                "revision_inventory_sha256": self.revision_inventory_sha256,
                "source_metadata_sha256": self.source_metadata_sha256,
                "source_contract_sha256": self.source_contract_sha256,
                "source_contracts": self.source_contracts,
                "full_source_data_content_hash": False,
                "source_revision_assurance": (
                    "completed immutable input runs plus logical source contracts, "
                    "physical path/size/mtime inventory, and selected metadata SHA-256"
                ),
            }
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe eye-angle run name: {run_name!r}.")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _configure_native_threads(native_threads: int) -> dict[str, str]:
    value = str(max(1, int(native_threads)))
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value
    return {name: value for name in NATIVE_THREAD_ENV_VARS}


def _add_group_metadata(
    source_zarr: Path,
    relative_group: str,
    selected: set[Path],
) -> None:
    group = source_zarr if relative_group in {"", "."} else source_zarr / relative_group
    for name in GROUP_METADATA_NAMES:
        candidate = group / name
        if candidate.is_file():
            selected.add(candidate)


def _ancestor_groups(relative_path: str) -> tuple[str, ...]:
    parts = tuple(part for part in str(relative_path).split("/") if part)
    return tuple("/".join(parts[:index]) for index in range(0, len(parts)))


def _add_array_tree(
    source_zarr: Path,
    relative_array: str,
    selected: set[Path],
) -> None:
    path = source_zarr / relative_array
    if not path.is_dir():
        raise FileNotFoundError(f"Required staged source array is missing: {path}")
    for group_path in _ancestor_groups(relative_array):
        _add_group_metadata(source_zarr, group_path, selected)
    for candidate in path.rglob("*"):
        if candidate.is_file():
            selected.add(candidate)


def _physical_files(source_zarr: Path, selected: set[Path]) -> tuple[PhysicalFile, ...]:
    return tuple(
        PhysicalFile(
            relative_path=path.relative_to(source_zarr).as_posix(),
            size_bytes=int(path.stat().st_size),
            mtime_ns=int(path.stat().st_mtime_ns),
        )
        for path in sorted(selected)
    )


def _inventory_digest(
    files: Sequence[PhysicalFile],
    *,
    include_mtime: bool,
) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(item.size_bytes)).encode("ascii"))
        if include_mtime:
            digest.update(b"\0")
            digest.update(str(int(item.mtime_ns)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata_content_digest(source_zarr: Path, files: Sequence[PhysicalFile]) -> str:
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


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(
        json_attr_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_complete_source(group: zarr.Group, *, label: str) -> None:
    if str(group.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError(f"{label} must be a completed immutable input run.")


def _frame_count(frame_indices: zarr.Array) -> int:
    """Return max(nonnegative frame index) + 1 with bounded read memory."""

    rows = int(frame_indices.shape[0])
    chunk_rows = max(1, int(frame_indices.chunks[0]))
    maximum = -1
    for start in range(0, rows, chunk_rows):
        values = np.asarray(
            frame_indices[start : min(rows, start + chunk_rows)],
            dtype=np.int64,
        )
        nonnegative = values[values >= 0]
        if nonnegative.size:
            maximum = max(maximum, int(nonnegative.max()))
    return maximum + 1


def _resolve_source_plan(
    source_zarr: Path,
    *,
    subject_shape_run: str | None,
    keypoint_run: str | None,
) -> tuple[
    Any,
    dict[str, Any],
    tuple[str, ...],
    tuple[PhysicalFile, ...],
    float | None,
    int,
]:
    root = open_zarr_root(source_zarr, mode="r")
    context = eye_writer._resolve_eye_angle_inputs(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=None,
        keypoint_run=keypoint_run,
    )
    geometry = context.eye_geometry
    if geometry.stage_group != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE:
        raise ValueError(
            "The production eye-angle materializer requires completed "
            "analysis/subject_shape_runs eye geometry."
        )
    _require_complete_source(geometry.group, label="Subject-shape source")
    _require_complete_source(context.kp_group, label="Refined-keypoint source")

    source_contracts = json_attr_safe(eye_writer._eye_angle_source_contracts(context))
    component_contracts = source_contracts["eye_geometry"]["components"]
    selected_arrays = [
        str(component["ellipse_params_path"])
        for component in component_contracts
    ]
    selected_arrays.extend(
        str(component["ellipse_success_path"])
        for component in component_contracts
    )
    selected_arrays.append(f"{geometry.group_path}/relations/eye_pair/separation_px")
    selected_arrays.extend(
        str(path)
        for path in source_contracts["resolved_arrays"].values()
        if path
    )
    selected_arrays = sorted(set(selected_arrays))

    selected_files: set[Path] = set()
    _add_group_metadata(source_zarr, ".", selected_files)
    for array_path in selected_arrays:
        _add_array_tree(source_zarr, array_path, selected_files)
    if context.source_kp_group_path:
        for group_path in _ancestor_groups(context.source_kp_group_path):
            _add_group_metadata(source_zarr, group_path, selected_files)
        _add_group_metadata(source_zarr, context.source_kp_group_path, selected_files)

    files = _physical_files(source_zarr, selected_files)
    if not files:
        raise RuntimeError(f"No physical eye-angle source files selected from {source_zarr}.")
    return (
        context,
        source_contracts,
        tuple(selected_arrays),
        files,
        get_fps(root),
        _frame_count(context.frame_indices_source["frame_indices"]),
    )


def build_eye_angle_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    run_name: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = eye_writer.DASK_WORKER_EXECUTION_BACKEND,
    scheduler: str = "processes",
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_workers: int = DEFAULT_SHARD_WORKERS,
    native_threads: int = DEFAULT_NATIVE_THREADS,
    fps: float | None = None,
    smoothing_window: int | None = None,
) -> EyeAngleMaterializationPlan:
    """Resolve exact inputs without creating scratch or mutating the archive."""

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
    positive_values = (
        chunk_rows,
        output_shard_rows,
        num_workers,
        shard_workers,
        native_threads,
    )
    if min(int(value) for value in positive_values) <= 0:
        raise ValueError("Chunk, shard, worker, and native-thread values must be positive.")
    backend = eye_writer._normalize_execution_backend(execution_backend)
    scheduler_key = eye_writer._normalize_scheduler(scheduler)
    if fps is not None and float(fps) <= 0:
        raise ValueError("fps must be positive when supplied.")
    if smoothing_window is not None and int(smoothing_window) <= 0:
        raise ValueError("smoothing_window must be positive when supplied.")

    (
        context,
        contracts,
        selected_arrays,
        files,
        metadata_fps,
        frame_count,
    ) = _resolve_source_plan(
        source,
        subject_shape_run=subject_shape_run,
        keypoint_run=keypoint_run,
    )
    resolved_name = _validate_run_name(run_name)
    target = source / "analysis" / "eye_angle_runs" / resolved_name
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target}")
    resolved_fps = float(fps) if fps is not None else (
        float(metadata_fps) if metadata_fps is not None and float(metadata_fps) > 0 else None
    )
    row_count = int(context.eye_geometry.ellipse_params.shape[0])
    estimated_output_bytes = max(1, row_count + frame_count) * (
        ESTIMATED_OUTPUT_BYTES_PER_DETECTION // 2
    )
    return EyeAngleMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        staged_zarr=scratch / "eye-inputs-and-output.zarr",
        sharded_run=scratch / "eye-angle-sharded-run",
        subject_shape_run=context.eye_geometry.run_name,
        keypoint_run=context.keypoint_run_name,
        source_keypoint_run=context.source_kp_run_name,
        run_name=resolved_name,
        row_count=row_count,
        frame_count=frame_count,
        chunk_rows=int(chunk_rows),
        output_shard_rows=int(output_shard_rows),
        execution_backend=backend,
        scheduler=scheduler_key,
        num_workers=int(num_workers),
        shard_workers=int(shard_workers),
        native_threads=int(native_threads),
        fps=resolved_fps,
        fps_source="cli_override" if fps is not None else (
            "authoritative_recording_metadata" if resolved_fps is not None else "unavailable"
        ),
        smoothing_window=None if smoothing_window is None else int(smoothing_window),
        selected_arrays=selected_arrays,
        physical_files=files,
        source_bytes=sum(item.size_bytes for item in files),
        estimated_output_bytes=int(estimated_output_bytes),
        inventory_sha256=_inventory_digest(files, include_mtime=False),
        revision_inventory_sha256=_inventory_digest(files, include_mtime=True),
        source_metadata_sha256=_metadata_content_digest(source, files),
        source_contract_sha256=_json_digest(contracts),
        source_contracts=contracts,
    )


def _write_files_manifest(plan: EyeAngleMaterializationPlan) -> None:
    plan.files_manifest_path.write_text(
        "".join(f"{item.relative_path}\n" for item in plan.physical_files),
        encoding="utf-8",
    )


def _copy_selected_files_python(plan: EyeAngleMaterializationPlan) -> None:
    for item in plan.physical_files:
        source = plan.source_zarr / item.relative_path
        target = plan.staged_zarr / item.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_selected_files_rsync(plan: EyeAngleMaterializationPlan) -> None:
    subprocess.run(
        [
            "rsync",
            "--archive",
            f"--files-from={plan.files_manifest_path}",
            f"{plan.source_zarr}/",
            f"{plan.staged_zarr}/",
        ],
        check=True,
    )


def _validate_file_inventory(
    root: Path,
    expected: Sequence[PhysicalFile],
) -> dict[str, Any]:
    observed: list[PhysicalFile] = []
    missing: list[str] = []
    size_mismatches: list[str] = []
    mtime_mismatches: list[str] = []
    for item in expected:
        path = root / item.relative_path
        if not path.is_file():
            missing.append(item.relative_path)
            continue
        stat = path.stat()
        observed_item = PhysicalFile(
            relative_path=item.relative_path,
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
        observed.append(observed_item)
        if observed_item.size_bytes != item.size_bytes:
            size_mismatches.append(item.relative_path)
        if observed_item.mtime_ns != item.mtime_ns:
            mtime_mismatches.append(item.relative_path)
    return {
        "valid": (
            not missing
            and not size_mismatches
            and not mtime_mismatches
            and len(observed) == len(expected)
        ),
        "expected_file_count": len(expected),
        "observed_file_count": len(observed),
        "expected_bytes": sum(item.size_bytes for item in expected),
        "observed_bytes": sum(item.size_bytes for item in observed),
        "expected_inventory_sha256": _inventory_digest(expected, include_mtime=False),
        "observed_inventory_sha256": _inventory_digest(observed, include_mtime=False),
        "expected_revision_inventory_sha256": _inventory_digest(expected, include_mtime=True),
        "observed_revision_inventory_sha256": _inventory_digest(observed, include_mtime=True),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "mtime_mismatches": mtime_mismatches,
    }


def audit_eye_angle_source_revision(plan: EyeAngleMaterializationPlan) -> dict[str, Any]:
    """Verify that resolved authoritative inputs still match the read-only plan."""

    inventory = _validate_file_inventory(plan.source_zarr, plan.physical_files)
    errors: list[str] = []
    if not inventory["valid"]:
        errors.append("physical source inventory changed")
    observed_metadata_sha256 = _metadata_content_digest(
        plan.source_zarr,
        plan.physical_files,
    )
    if observed_metadata_sha256 != plan.source_metadata_sha256:
        errors.append("selected source metadata changed")
    try:
        context, contracts, arrays, _files, _fps, frame_count = _resolve_source_plan(
            plan.source_zarr,
            subject_shape_run=plan.subject_shape_run,
            keypoint_run=plan.keypoint_run,
        )
        observed_contract_sha256 = _json_digest(contracts)
        if context.eye_geometry.run_name != plan.subject_shape_run:
            errors.append("resolved subject-shape run changed")
        if context.keypoint_run_name != plan.keypoint_run:
            errors.append("resolved refined-keypoint run changed")
        if int(frame_count) != int(plan.frame_count):
            errors.append("resolved frame count changed")
        if tuple(arrays) != tuple(plan.selected_arrays):
            errors.append("resolved source array set changed")
        if observed_contract_sha256 != plan.source_contract_sha256:
            errors.append("logical source contract changed")
    except Exception as exc:  # fail closed and preserve the exact resolver error
        observed_contract_sha256 = None
        errors.append(f"source resolution failed: {type(exc).__name__}: {exc}")
    return json_attr_safe(
        {
            "schema_id": SOURCE_REVISION_AUDIT_SCHEMA_ID,
            "status": "current" if not errors else "changed",
            "checked_at_utc": _utc_now(),
            "authoritative_source_zarr": str(plan.source_zarr),
            "subject_shape_run": plan.subject_shape_run,
            "keypoint_run": plan.keypoint_run,
            "inventory": inventory,
            "expected_source_metadata_sha256": plan.source_metadata_sha256,
            "observed_source_metadata_sha256": observed_metadata_sha256,
            "expected_source_contract_sha256": plan.source_contract_sha256,
            "observed_source_contract_sha256": observed_contract_sha256,
            "full_source_data_content_hash": False,
            "errors": errors,
        }
    )


def stage_eye_angle_sources(
    plan: EyeAngleMaterializationPlan,
    *,
    copy_backend: str,
    check_capacity: bool,
) -> dict[str, Any]:
    """Copy and logically validate the exact resolved source surface."""

    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    plan.staged_zarr.mkdir(parents=True)
    _write_files_manifest(plan)
    required_bytes = (
        plan.source_bytes
        + 2 * plan.estimated_output_bytes
        + DEFAULT_CAPACITY_MARGIN_BYTES
    )
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
    inventory = _validate_file_inventory(plan.staged_zarr, plan.physical_files)
    if not inventory["valid"]:
        raise RuntimeError(f"Staged source inventory validation failed: {inventory}")
    staged_metadata_sha256 = _metadata_content_digest(
        plan.staged_zarr,
        plan.physical_files,
    )
    if staged_metadata_sha256 != plan.source_metadata_sha256:
        raise RuntimeError("Staged source metadata content differs from the plan.")
    (
        staged_context,
        staged_contracts,
        staged_arrays,
        _files,
        _fps,
        staged_frame_count,
    ) = _resolve_source_plan(
        plan.staged_zarr,
        subject_shape_run=plan.subject_shape_run,
        keypoint_run=plan.keypoint_run,
    )
    if (
        staged_context.eye_geometry.run_name != plan.subject_shape_run
        or staged_context.keypoint_run_name != plan.keypoint_run
        or int(staged_frame_count) != int(plan.frame_count)
        or tuple(staged_arrays) != plan.selected_arrays
        or _json_digest(staged_contracts) != plan.source_contract_sha256
    ):
        raise RuntimeError("Staged logical source contract differs from the plan.")
    source_revision = audit_eye_angle_source_revision(plan)
    if source_revision["status"] != "current":
        raise RuntimeError(
            f"Authoritative inputs changed during source staging: {source_revision}"
        )
    payload = json_attr_safe(
        {
            "schema_id": STAGING_SCHEMA_ID,
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "duration_seconds": duration,
            "mib_per_second": (
                (plan.source_bytes / (1024**2)) / duration if duration > 0 else None
            ),
            "copy_backend": copy_backend,
            "host": socket.gethostname(),
            "lsb_jobid": os.environ.get("LSB_JOBID"),
            "authoritative_source_zarr": str(plan.source_zarr),
            "node_local_staged_zarr": str(plan.staged_zarr),
            "subject_shape_run": plan.subject_shape_run,
            "keypoint_run": plan.keypoint_run,
            "source_keypoint_run": plan.source_keypoint_run,
            "row_count": plan.row_count,
            "frame_count": plan.frame_count,
            "selected_arrays": list(plan.selected_arrays),
            "source_contracts": plan.source_contracts,
            "source_contract_sha256": plan.source_contract_sha256,
            "source_metadata_sha256": plan.source_metadata_sha256,
            "inventory": inventory,
            "source_revision_audit": source_revision,
            "capacity": {
                "check_enabled": bool(check_capacity),
                "free_bytes_before_copy": free_bytes,
                "required_bytes_estimate": required_bytes,
                "source_bytes": plan.source_bytes,
                "estimated_output_bytes_per_copy": plan.estimated_output_bytes,
                "estimated_output_copy_count": 2,
                "margin_bytes": DEFAULT_CAPACITY_MARGIN_BYTES,
            },
        }
    )
    _write_json_atomic(plan.staging_manifest_path, payload)
    return payload


def _iter_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in group.arrays():
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, child_prefix)


def _decode_text_index(array: zarr.Array) -> tuple[str, ...]:
    values = np.asarray(array[:])
    if values.ndim == 1 and values.dtype.kind in {"S", "U"}:
        return tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values.tolist()
        )
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("Unsupported eye-angle text-index encoding.")
    decoded: list[str] = []
    for row in values:
        raw = bytes(np.asarray(row, dtype=np.uint8).tolist()).split(b"\0", 1)[0]
        decoded.append(raw.decode("utf-8"))
    return tuple(decoded)


def _validate_eye_angle_run(
    path: Path,
    *,
    row_count: int,
    frame_count: int,
    expected_source_contract_sha256: str,
    require_sharded: bool,
) -> dict[str, Any]:
    group = open_zarr_root(path, mode="r")
    attrs = group.attrs
    errors: list[str] = []
    if str(attrs.get("schema_id")) != eye_writer.EYE_ANGLE_RUN_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if int(attrs.get("schema_version", -1)) != eye_writer.EYE_ANGLE_RUN_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if str(attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if str(attrs.get("layout")) != eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        errors.append("layout is not compact_dense_v2")
    if int(attrs.get("num_detections", -1)) != int(row_count):
        errors.append("num_detections mismatch")
    observed_frame_count = int(attrs.get("num_frames", -1))
    if observed_frame_count != int(frame_count):
        errors.append("num_frames mismatch")

    output_schema = attrs.get("eye_angle_output_schema")
    if not isinstance(output_schema, dict) or (
        str(output_schema.get("schema_id")) != eye_writer.EYE_ANGLE_OUTPUT_SCHEMA_ID
        or int(output_schema.get("schema_version", -1))
        != eye_writer.EYE_ANGLE_OUTPUT_SCHEMA_VERSION
    ):
        errors.append("eye_angle_output_schema contract mismatch")
    algorithm = attrs.get("eye_angle_algorithm_contract")
    if not isinstance(algorithm, dict) or (
        str(algorithm.get("schema_id"))
        != eye_writer.EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_ID
        or int(algorithm.get("schema_version", -1))
        != eye_writer.EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_VERSION
    ):
        errors.append("eye_angle_algorithm_contract mismatch")
    elif (
        str(algorithm.get("method")) != eye_writer.EYE_ANGLE_METHOD
        or str(algorithm.get("method_version")) != eye_writer.EYE_ANGLE_METHOD_VERSION
    ):
        errors.append("eye-angle method contract mismatch")
    source_contracts = attrs.get("eye_angle_source_contracts")
    observed_contract_sha256 = (
        _json_digest(source_contracts) if isinstance(source_contracts, dict) else None
    )
    if observed_contract_sha256 != expected_source_contract_sha256:
        errors.append("persisted source contracts differ from the materialization plan")

    index_specs = {
        "angle_channel_index": ("roi_angles", "frame_angles"),
        "qa_channel_index": ("roi_qa", "frame_qa"),
    }
    channel_names: dict[str, tuple[str, ...]] = {}
    for index_path, array_paths in index_specs.items():
        index_group = group.get(index_path)
        if not isinstance(index_group, zarr.Group):
            errors.append(f"missing group {index_path}")
            continue
        name_array = index_group.get("name")
        if not isinstance(name_array, zarr.Array):
            errors.append(f"missing array {index_path}/name")
            continue
        try:
            names = _decode_text_index(name_array)
        except ValueError as exc:
            errors.append(f"invalid {index_path}/name: {exc}")
            continue
        channel_names[index_path] = names
        if int(index_group.attrs.get("channel_count", -1)) != len(names):
            errors.append(f"channel_count mismatch for {index_path}")
        for array_path in array_paths:
            array = group.get(array_path)
            if not isinstance(array, zarr.Array):
                errors.append(f"missing array {array_path}")
                continue
            expected_rows = (
                row_count if array_path.startswith("roi_") else observed_frame_count
            )
            if tuple(int(value) for value in array.shape) != (expected_rows, len(names)):
                errors.append(f"shape mismatch for {array_path}")

    required_angles = {
        "left_eye_angle_deg",
        "right_eye_angle_deg",
        "vergence_eye_angle_deg",
        "left_gaze_signed_deg",
        "right_gaze_signed_deg",
        "left_nasal_gaze_deg",
        "right_nasal_gaze_deg",
    }
    if not required_angles.issubset(set(channel_names.get("angle_channel_index", ()))):
        errors.append("required biological eye-angle channels are missing")
    required_qa = {"valid_left", "valid_right", "valid_frame", "reason_codes"}
    if not required_qa.issubset(set(channel_names.get("qa_channel_index", ()))):
        errors.append("required eye-angle QA channels are missing")

    vector_index = group.get("vector_channel_index")
    roi_vectors = group.get("roi_vectors")
    if not isinstance(vector_index, zarr.Group) or not isinstance(roi_vectors, zarr.Array):
        errors.append("missing gaze vector table or index")
    else:
        name_array = vector_index.get("name")
        try:
            vector_names = (
                _decode_text_index(name_array)
                if isinstance(name_array, zarr.Array)
                else ()
            )
        except ValueError:
            vector_names = ()
        if not {"left_gaze_xy", "right_gaze_xy"}.issubset(set(vector_names)):
            errors.append("required gaze vectors are missing")
        if tuple(int(value) for value in roi_vectors.shape) != (
            row_count,
            len(vector_names),
            2,
        ):
            errors.append("shape mismatch for roi_vectors")

    required_support = {
        "support/frame_indices": (row_count,),
        "support/time_seconds": (row_count,),
        "support/body_frame/origin_xy": (row_count, 2),
        "support/body_frame/forward_axis_xy": (row_count, 2),
        "support/body_frame/left_axis_xy": (row_count, 2),
        "support/body_frame/heading_deg": (row_count,),
        "support/body_frame/valid": (row_count,),
    }
    for array_path, expected_shape in required_support.items():
        array = group.get(array_path)
        if not isinstance(array, zarr.Array):
            errors.append(f"missing array {array_path}")
        elif tuple(int(value) for value in array.shape) != expected_shape:
            errors.append(f"shape mismatch for {array_path}")

    array_count = 0
    sharded_array_count = 0
    for array_path, array in _iter_arrays(group):
        array_count += 1
        shards = getattr(array, "shards", None)
        if shards is None:
            if require_sharded and int(array.ndim) >= 1:
                errors.append(f"{array_path}: expected indexed sharding")
            continue
        sharded_array_count += 1
        chunks = tuple(int(value) for value in array.chunks)
        outer = tuple(int(value) for value in shards)
        if any(outer[index] % chunks[index] for index in range(len(chunks))):
            errors.append(f"{array_path}: shard grid is not chunk aligned")
    physical_layout = attrs.get("physical_storage_layout")
    if require_sharded and not isinstance(physical_layout, dict):
        errors.append("physical_storage_layout provenance is missing")
    materialization = attrs.get("node_local_materialization")
    if not isinstance(materialization, dict):
        errors.append("node_local_materialization provenance is missing")
    return json_attr_safe(
        {
            "valid": not errors,
            "errors": errors,
            "row_count": row_count,
            "frame_count": observed_frame_count,
            "angle_channel_count": len(channel_names.get("angle_channel_index", ())),
            "qa_channel_count": len(channel_names.get("qa_channel_index", ())),
            "array_count": array_count,
            "sharded_array_count": sharded_array_count,
            "require_sharded": bool(require_sharded),
            "source_contract_sha256": observed_contract_sha256,
            "algorithm_contract_sha256": (
                _json_digest(algorithm) if isinstance(algorithm, dict) else None
            ),
            "output_schema_sha256": (
                _json_digest(output_schema) if isinstance(output_schema, dict) else None
            ),
            "physical_storage_layout": physical_layout,
        }
    )


def publish_eye_angle_run(
    plan: EyeAngleMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    """Validate and publish one completed sharded eye-angle run."""

    def validate(path: Path) -> dict[str, Any]:
        return _validate_eye_angle_run(
            path,
            row_count=plan.row_count,
            frame_count=plan.frame_count,
            expected_source_contract_sha256=plan.source_contract_sha256,
            require_sharded=True,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(root.require_group("analysis"), "eye_angle_runs"),
        )

    def after_rename(_root: zarr.Group, run_group: zarr.Group) -> dict[str, Any]:
        source_revision = audit_eye_angle_source_revision(plan)
        if source_revision["status"] != "current":
            raise RuntimeError(
                "Eye-angle inputs changed during materialization: "
                f"{source_revision}"
            )
        write_best_effort_run_lineage_attrs(run_group, run_family="eye_angle_run")
        return {"source_revision_audit": source_revision}

    def complete(
        _root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="eye_angle_materializer",
            ),
        )

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/eye_angle_runs"]
        if str(parent.attrs.get("latest")) != plan.run_name or str(
            parent.attrs.get("latest_complete")
        ) != plan.run_name:
            raise RuntimeError("Eye-angle parent pointers were not updated consistently.")

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.sharded_run,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="eye-angle-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=(
                "exact_source_subset_staged_local_compute_then_shard_then_"
                "atomic_run_group_publish"
            ),
            rollback_policy=(
                "remove_new_target_and_restore_parent_attrs_on_post_rename_failure"
            ),
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        after_rename=after_rename,
        payload_metadata={
            "authoritative_source_zarr": str(plan.source_zarr),
            "node_local_staged_zarr": str(plan.staged_zarr),
            "node_local_regular_run": str(plan.local_run_path),
            "node_local_sharded_run": str(plan.sharded_run),
            "materialization": json_attr_safe(materialization_payload),
        },
    )


def materialize_eye_angles(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    run_name: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = eye_writer.DASK_WORKER_EXECUTION_BACKEND,
    scheduler: str = "processes",
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_workers: int = DEFAULT_SHARD_WORKERS,
    native_threads: int = DEFAULT_NATIVE_THREADS,
    fps: float | None = None,
    smoothing_window: int | None = None,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    check_capacity: bool = True,
    stage_command: str | None = None,
) -> dict[str, Any]:
    """Plan or execute the complete staged eye-angle materialization."""

    plan = build_eye_angle_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        subject_shape_run=subject_shape_run,
        keypoint_run=keypoint_run,
        run_name=run_name,
        chunk_rows=chunk_rows,
        output_shard_rows=output_shard_rows,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        shard_workers=shard_workers,
        native_threads=native_threads,
        fps=fps,
        smoothing_window=smoothing_window,
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
    native_environment = _configure_native_threads(plan.native_threads)
    try:
        staging = stage_eye_angle_sources(
            plan,
            copy_backend=copy_backend,
            check_capacity=check_capacity,
        )
        writer_argv = [
            str(plan.staged_zarr),
            "--subject-shape-run",
            plan.subject_shape_run,
            "--keypoint-run",
            plan.keypoint_run,
            "--run-name",
            plan.run_name,
            "--chunk-size",
            str(plan.chunk_rows),
            "--execution-backend",
            plan.execution_backend,
            "--scheduler",
            plan.scheduler,
            "--num-workers",
            str(plan.num_workers),
            "--layout",
            eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
            "--quiet",
        ]
        if plan.fps is not None:
            writer_argv.extend(("--fps", str(plan.fps)))
        if plan.smoothing_window is not None:
            writer_argv.extend(("--smoothing-window", str(plan.smoothing_window)))
        compute_started = time.perf_counter()
        eye_writer.main(writer_argv)
        compute_seconds = float(time.perf_counter() - compute_started)

        regular_validation = _validate_eye_angle_run(
            plan.local_run_path,
            row_count=plan.row_count,
            frame_count=plan.frame_count,
            expected_source_contract_sha256=plan.source_contract_sha256,
            require_sharded=False,
        )
        # The validation contract requires materialization provenance, which is
        # appended immediately after validating the scientific writer surface.
        non_provenance_errors = [
            error
            for error in regular_validation["errors"]
            if error != "node_local_materialization provenance is missing"
        ]
        if non_provenance_errors:
            raise RuntimeError(
                "Node-local regular eye-angle run is invalid: "
                f"{regular_validation}"
            )

        materialization_payload = json_attr_safe(
            {
                "schema_id": MATERIALIZATION_SCHEMA_ID,
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "authoritative_source_zarr": str(plan.source_zarr),
                "node_local_staged_zarr": str(plan.staged_zarr),
                "source_access_policy": "authoritative_shared_read_only_then_exact_local_subset",
                "source_staging": staging,
                "compute": {
                    "writer": "fisheye.analysis.eye_angle_analysis",
                    "writer_arguments": writer_argv,
                    "stage_command": stage_command or (
                        " ".join(sys.argv) if sys.argv else "unknown"
                    ),
                    "duration_seconds": compute_seconds,
                    "chunk_rows": plan.chunk_rows,
                    "execution_backend": plan.execution_backend,
                    "scheduler": plan.scheduler,
                    "num_workers": plan.num_workers,
                    "native_thread_environment": native_environment,
                    "fps": plan.fps,
                    "fps_source": plan.fps_source,
                    "smoothing_window": plan.smoothing_window,
                    "layout": eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
                },
                "regular_validation": {
                    **regular_validation,
                    "valid": not non_provenance_errors,
                    "errors": non_provenance_errors,
                },
                "source_contract_sha256": plan.source_contract_sha256,
                "source_metadata_sha256": plan.source_metadata_sha256,
                "algorithm_contract": {
                    "schema_id": eye_writer.EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_ID,
                    "schema_version": eye_writer.EYE_ANGLE_ALGORITHM_CONTRACT_SCHEMA_VERSION,
                    "persisted_run_attr": "eye_angle_algorithm_contract",
                    "sha256": regular_validation["algorithm_contract_sha256"],
                },
                "output_contract": {
                    "schema_id": eye_writer.EYE_ANGLE_OUTPUT_SCHEMA_ID,
                    "schema_version": eye_writer.EYE_ANGLE_OUTPUT_SCHEMA_VERSION,
                    "persisted_run_attr": "eye_angle_output_schema",
                    "sha256": regular_validation["output_schema_sha256"],
                },
            }
        )
        regular_run = open_zarr_root(plan.local_run_path, mode="a")
        regular_run.attrs["node_local_materialization"] = materialization_payload
        provenance = dict(regular_run.attrs.get("provenance", {}))
        provenance["materialization"] = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "authoritative_source_zarr": str(plan.source_zarr),
            "node_local_staged_zarr": str(plan.staged_zarr),
            "source_contract_sha256": plan.source_contract_sha256,
            "source_metadata_sha256": plan.source_metadata_sha256,
            "selected_arrays": list(plan.selected_arrays),
            "compute_arguments": writer_argv,
        }
        regular_run.attrs["provenance"] = json_attr_safe(provenance)

        sharding = copy_completed_run_to_sharded(
            plan.local_run_path,
            plan.sharded_run,
            row_count_array=None,
            shard_rows=plan.output_shard_rows,
            workers=plan.shard_workers,
        )
        sharding_summary = {
            key: value
            for key, value in sharding.items()
            if key not in {"arrays", "shards", "static_arrays"}
        }
        sharded = open_zarr_root(plan.sharded_run, mode="a")
        materialization_payload["sharding"] = json_attr_safe(sharding_summary)
        sharded.attrs["node_local_materialization"] = materialization_payload
        sharded_provenance = dict(sharded.attrs.get("provenance", {}))
        sharded_provenance["materialization"]["sharding"] = json_attr_safe(
            sharding_summary
        )
        sharded.attrs["provenance"] = json_attr_safe(sharded_provenance)

        sharded_validation = _validate_eye_angle_run(
            plan.sharded_run,
            row_count=plan.row_count,
            frame_count=plan.frame_count,
            expected_source_contract_sha256=plan.source_contract_sha256,
            require_sharded=True,
        )
        if not sharded_validation["valid"]:
            raise RuntimeError(
                f"Node-local sharded eye-angle run is invalid: {sharded_validation}"
            )
        materialization_payload["sharded_validation"] = sharded_validation
        sharded.attrs["node_local_materialization"] = materialization_payload
        publish = publish_eye_angle_run(
            plan,
            materialization_payload=materialization_payload,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
                "staging": staging,
                "local_materialization": materialization_payload,
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
        return scratch_user / job_id / f"palette_eye_angles_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_eye_angles_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--subject-shape-run")
    parser.add_argument("--keypoint-run")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
    )
    parser.add_argument(
        "--execution-backend",
        choices=eye_writer.EXECUTION_BACKENDS,
        default=eye_writer.DASK_WORKER_EXECUTION_BACKEND,
    )
    parser.add_argument(
        "--scheduler",
        choices=eye_writer.SUPPORTED_SCHEDULERS,
        default="processes",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--shard-workers", type=int, default=DEFAULT_SHARD_WORKERS)
    parser.add_argument("--native-threads", type=int, default=DEFAULT_NATIVE_THREADS)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--smoothing-window", type=int)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--no-capacity-check", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_eye_angles(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        subject_shape_run=args.subject_shape_run,
        keypoint_run=args.keypoint_run,
        run_name=args.run_name,
        chunk_rows=args.chunk_size,
        output_shard_rows=args.output_shard_rows,
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        shard_workers=args.shard_workers,
        native_threads=args.native_threads,
        fps=args.fps,
        smoothing_window=args.smoothing_window,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
        check_capacity=not args.no_capacity_check,
    )
    if args.report is not None:
        _write_json_atomic(args.report.expanduser().resolve(), result)
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
