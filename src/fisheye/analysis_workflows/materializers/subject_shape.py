"""Materialize subject shape locally, shard it, and publish atomically.

The authoritative refined subject masks are opened read-only.  Computation
writes ordinary logical chunks into a node-local Zarr, a second node-local pass
assembles complete indexed outer shards with exact decoded validation, and only
the completed sharded run is copied back to shared storage.
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

from ...analysis.subject_shape_runs import (
    CENTERLINE_SAMPLE_COUNT,
    COMPONENT_ORDER,
    DASK_WORKER_EXECUTION_BACKEND,
    SUBJECT_SHAPE_SCHEMA_ID,
    TAIL_SAMPLE_COUNT,
    audit_subject_shape_source_revisions_group,
    write_subject_shape_run_group,
)
from ...shared.json_safety import json_attr_safe
from ...shared.refined_subject_masks_io import (
    load_refined_subject_masks_run_tables,
    resolve_refined_subject_masks_run,
)
from ...shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent
from ...shared.zarr_sharded_copy import copy_completed_run_to_sharded


MATERIALIZATION_SCHEMA_ID = "palette.subject_shape_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.subject_shape_run_publish.v1"
DEFAULT_BLOCK_ROWS = 1_024
DEFAULT_OUTPUT_SHARD_ROWS = 131_072
DEFAULT_NATIVE_THREADS = 1
DEFAULT_NUM_WORKERS = 32
DEFAULT_SHARD_COPY_WORKERS = 16
DEFAULT_CAPACITY_MARGIN_BYTES = 1024 * 1024 * 1024
ESTIMATED_BYTES_PER_ROW_PER_COPY = 4096
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class SubjectShapeMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    compute_zarr: Path
    sharded_run: Path
    refined_run: str
    run_name: str
    row_count: int
    component_names: tuple[str, ...]
    block_rows: int
    output_shard_rows: int
    execution_backend: str
    scheduler: str
    num_workers: int
    shard_copy_workers: int
    native_threads: int
    estimated_scratch_bytes: int
    source_contract: dict[str, Any]

    @property
    def compute_run_path(self) -> Path:
        return self.compute_zarr / "analysis" / "subject_shape_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "subject_shape_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return json_attr_safe(
            {
                "schema_id": MATERIALIZATION_SCHEMA_ID,
                "source_zarr": str(self.source_zarr),
                "source_access_policy": "authoritative_shared_read_only",
                "scratch_root": str(self.scratch_root),
                "compute_zarr": str(self.compute_zarr),
                "compute_run_path": str(self.compute_run_path),
                "sharded_run": str(self.sharded_run),
                "target_run_path": str(self.target_run_path),
                "refined_run": self.refined_run,
                "run_name": self.run_name,
                "row_count": self.row_count,
                "component_names": list(self.component_names),
                "block_rows": self.block_rows,
                "output_shard_rows": self.output_shard_rows,
                "execution_backend": self.execution_backend,
                "scheduler": self.scheduler,
                "num_workers": self.num_workers,
                "shard_copy_workers": self.shard_copy_workers,
                "native_threads": self.native_threads,
                "centerline_crop_to_foreground": True,
                "estimated_scratch_bytes": self.estimated_scratch_bytes,
                "source_contract": self.source_contract,
            }
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe subject-shape run name: {run_name!r}.")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _configure_native_threads(native_threads: int) -> dict[str, str]:
    value = str(max(1, int(native_threads)))
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = value
    return {name: value for name in NATIVE_THREAD_ENV_VARS}


def build_subject_shape_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    refined_run: str | None,
    run_name: str,
    components: Sequence[str] | None = None,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = DASK_WORKER_EXECUTION_BACKEND,
    scheduler: str = "processes",
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_copy_workers: int = DEFAULT_SHARD_COPY_WORKERS,
    native_threads: int = DEFAULT_NATIVE_THREADS,
) -> SubjectShapeMaterializationPlan:
    """Resolve a read-only plan without creating scratch or mutating the archive."""

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
    if int(block_rows) <= 0 or int(output_shard_rows) <= 0:
        raise ValueError("Block and output-shard row counts must be positive.")
    if int(num_workers) <= 0 or int(shard_copy_workers) <= 0 or int(native_threads) <= 0:
        raise ValueError("Worker and native-thread counts must be positive.")
    backend = str(execution_backend).strip().lower()
    if backend not in {"serial_driver", DASK_WORKER_EXECUTION_BACKEND}:
        raise ValueError(f"Unsupported execution backend: {execution_backend!r}.")
    scheduler_key = str(scheduler).strip().lower().replace("_", "-")
    if scheduler_key not in {"single-threaded", "threads", "processes", "distributed"}:
        raise ValueError(f"Unsupported scheduler: {scheduler!r}.")

    root = open_zarr_root(source, mode="r")
    refined_group, resolved_refined_run, _path = resolve_refined_subject_masks_run(root, refined_run)
    tables = load_refined_subject_masks_run_tables(
        root,
        run_name=resolved_refined_run,
        component_names=components,
        include_masks_roi=True,
        include_metrics=False,
        include_components=False,
        include_relations=False,
    )
    mask_store = tables.require_mask_store()
    available = tuple(str(value) for value in refined_group.attrs.get("mask_labels") or ())
    selected = (
        tuple(str(value) for value in components)
        if components
        else tuple(name for name in COMPONENT_ORDER if name in available)
    )
    if not selected:
        raise ValueError("No known subject-shape components are available in the refined run.")
    target_name = _validate_run_name(run_name)
    target = source / "analysis" / "subject_shape_runs" / target_name
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target}")
    row_count = int(mask_store.n_rows)
    estimated = (
        2 * row_count * ESTIMATED_BYTES_PER_ROW_PER_COPY
        + DEFAULT_CAPACITY_MARGIN_BYTES
    )
    return SubjectShapeMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        compute_zarr=scratch / "compute.zarr",
        sharded_run=scratch / "subject-shape-sharded-run",
        refined_run=resolved_refined_run,
        run_name=target_name,
        row_count=row_count,
        component_names=selected,
        block_rows=int(block_rows),
        output_shard_rows=int(output_shard_rows),
        execution_backend=backend,
        scheduler=scheduler_key,
        num_workers=int(num_workers),
        shard_copy_workers=int(shard_copy_workers),
        native_threads=int(native_threads),
        estimated_scratch_bytes=int(estimated),
        source_contract=json_attr_safe(
            {
                "schema_id": refined_group.attrs.get("schema_id"),
                "schema_version": refined_group.attrs.get("schema_version"),
                "method": refined_group.attrs.get("method"),
                "method_version": refined_group.attrs.get("method_version"),
                "palette_run_completion_status": refined_group.attrs.get(
                    "palette_run_completion_status"
                ),
                "mask_labels": list(available),
                "mask_store_encoding": mask_store.encoding,
                "mask_storage_surface": mask_store.storage_surface,
            }
        ),
    )


def _validate_subject_shape_run(
    path: Path,
    *,
    row_count: int,
    require_sharded: bool,
) -> dict[str, Any]:
    group = open_zarr_root(path, mode="r")
    errors: list[str] = []
    if str(group.attrs.get("schema_id")) != SUBJECT_SHAPE_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if str(group.attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if not bool(group.attrs.get("centerline_crop_to_foreground")):
        errors.append("foreground-cropped centerline acceleration not recorded")
    expected = {
        "row_index/frame_indices": (int(row_count),),
        "components/subject_body/centerline_xy": (
            int(row_count),
            CENTERLINE_SAMPLE_COUNT,
            2,
        ),
        "components/subject_body/centerline_valid": (int(row_count),),
        "components/subject_body/tail_sample_xy": (
            int(row_count),
            TAIL_SAMPLE_COUNT,
            2,
        ),
    }
    for name, shape in expected.items():
        node = group.get(name)
        if not isinstance(node, zarr.Array):
            errors.append(f"missing array {name}")
            continue
        if tuple(int(value) for value in node.shape) != shape:
            errors.append(f"shape mismatch for {name}")
        if require_sharded and getattr(node, "shards", None) is None:
            errors.append(f"array is not physically sharded: {name}")
    body_frame = group.get("body_frame/heading_deg")
    if isinstance(body_frame, zarr.Array):
        if tuple(int(value) for value in body_frame.shape) != (int(row_count),):
            errors.append("shape mismatch for body_frame/heading_deg")
        if require_sharded and getattr(body_frame, "shards", None) is None:
            errors.append("array is not physically sharded: body_frame/heading_deg")
    layout = group.attrs.get("physical_storage_layout")
    if require_sharded and not isinstance(layout, dict):
        errors.append("physical storage layout provenance missing")
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(row_count),
        "require_sharded": bool(require_sharded),
        "physical_storage_layout": layout,
    }


def _file_content_digest(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                total += len(block)
        digest.update(b"\n")
        count += 1
    return count, total, digest.hexdigest()


def _copy_and_checksum(source: Path, target: Path, *, copy_backend: str) -> dict[str, Any]:
    if copy_backend == "rsync":
        target.mkdir(parents=True)
        subprocess.run(["rsync", "--archive", f"{source}/", f"{target}/"], check=True)
        check = subprocess.run(
            [
                "rsync",
                "--archive",
                "--dry-run",
                "--checksum",
                "--delete",
                "--itemize-changes",
                f"{source}/",
                f"{target}/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if check.stdout.strip():
            raise RuntimeError(f"Rsync checksum validation found differences: {check.stdout}")
        count, total, digest = _file_content_digest(source)
        return {
            "backend": "rsync",
            "verification": "rsync_checksum_dry_run",
            "file_count": count,
            "physical_bytes": total,
            "source_tree_sha256": digest,
        }
    if copy_backend == "python":
        shutil.copytree(source, target)
        source_count, source_total, source_digest = _file_content_digest(source)
        target_count, target_total, target_digest = _file_content_digest(target)
        if (source_count, source_total, source_digest) != (
            target_count,
            target_total,
            target_digest,
        ):
            raise RuntimeError("Python publish copy failed exact physical checksum validation.")
        return {
            "backend": "python",
            "verification": "sha256_all_physical_files",
            "file_count": source_count,
            "physical_bytes": source_total,
            "source_tree_sha256": source_digest,
        }
    raise ValueError(f"Unsupported copy backend: {copy_backend!r}.")


def _restore_attrs(group: zarr.Group, attrs: dict[str, Any]) -> None:
    group.attrs.clear()
    group.attrs.update(attrs)


def _serialized_publish(function):
    @functools.wraps(function)
    def wrapped(plan: SubjectShapeMaterializationPlan, *args, **kwargs):
        lock_path = plan.source_zarr.parent / f".{plan.source_zarr.name}.subject-shape-publish.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return function(plan, *args, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return wrapped


@_serialized_publish
def publish_subject_shape_run(
    plan: SubjectShapeMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    local_validation = _validate_subject_shape_run(
        plan.sharded_run,
        row_count=plan.row_count,
        require_sharded=True,
    )
    if not local_validation["valid"]:
        raise RuntimeError(f"Local sharded run validation failed: {local_validation}")

    root = open_zarr_root(plan.source_zarr, mode="a")
    parent = require_runs_parent(root.require_group("analysis"), "subject_shape_runs")
    if plan.run_name in parent or plan.target_run_path.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {plan.target_run_path}")
    parent_attrs_before = dict(parent.attrs)
    plan.target_run_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.target_run_path.parent / f".{plan.run_name}.publish_tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Refusing existing publish temporary: {temporary}")

    published = False
    started = time.perf_counter()
    try:
        physical = _copy_and_checksum(plan.sharded_run, temporary, copy_backend=copy_backend)
        temporary_validation = _validate_subject_shape_run(
            temporary,
            row_count=plan.row_count,
            require_sharded=True,
        )
        if not temporary_validation["valid"]:
            raise RuntimeError(f"Temporary publish logical validation failed: {temporary_validation}")
        os.replace(temporary, plan.target_run_path)
        published = True

        root = open_zarr_root(plan.source_zarr, mode="a")
        parent = require_runs_parent(root.require_group("analysis"), "subject_shape_runs")
        run_group = parent[plan.run_name]
        source_revision_audit = audit_subject_shape_source_revisions_group(
            root,
            shape_run=plan.run_name,
            refined_run=plan.refined_run,
        )
        if str(source_revision_audit.get("status")) != "current":
            raise RuntimeError(
                "Refined subject-mask revisions changed during materialization: "
                f"{source_revision_audit}"
            )
        payload = {
            "schema_id": PUBLISH_SCHEMA_ID,
            "policy": "node_local_compute_then_shard_then_atomic_run_group_publish",
            "serialization_policy": "per_recording_advisory_file_lock",
            "published_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "lsb_jobid": os.environ.get("LSB_JOBID"),
            "source_zarr": str(plan.source_zarr),
            "local_sharded_run": str(plan.sharded_run),
            "target_run_path": str(plan.target_run_path),
            "copy_duration_seconds": float(time.perf_counter() - started),
            "physical_copy": physical,
            "local_validation": local_validation,
            "source_revision_audit": source_revision_audit,
            "materialization": materialization_payload,
        }
        run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)
        write_best_effort_run_lineage_attrs(run_group, run_family="subject_shape_run")
        pre_pointer_validation = _validate_subject_shape_run(
            plan.target_run_path,
            row_count=plan.row_count,
            require_sharded=True,
        )
        if not pre_pointer_validation["valid"]:
            raise RuntimeError(
                f"Published run failed pre-pointer validation: {pre_pointer_validation}"
            )
        payload["pre_pointer_validation"] = pre_pointer_validation
        run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="subject_shape_materializer",
            ),
        )
        final_validation = _validate_subject_shape_run(
            plan.target_run_path,
            row_count=plan.row_count,
            require_sharded=True,
        )
        if not final_validation["valid"]:
            raise RuntimeError(f"Published run validation failed: {final_validation}")
        if str(parent.attrs.get("latest")) != plan.run_name or str(
            parent.attrs.get("latest_complete")
        ) != plan.run_name:
            raise RuntimeError("Subject-shape parent pointers were not updated to the published run.")
        payload["final_validation"] = final_validation
        run_group.attrs["cluster_output_staging"] = json_attr_safe(payload)
        return payload
    except BaseException:
        if published and plan.target_run_path.exists():
            shutil.rmtree(plan.target_run_path)
        root = open_zarr_root(plan.source_zarr, mode="a")
        parent = require_runs_parent(root.require_group("analysis"), "subject_shape_runs")
        _restore_attrs(parent, parent_attrs_before)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def materialize_subject_shape(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    refined_run: str | None,
    run_name: str,
    components: Sequence[str] | None = None,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = DASK_WORKER_EXECUTION_BACKEND,
    scheduler: str = "processes",
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_copy_workers: int = DEFAULT_SHARD_COPY_WORKERS,
    native_threads: int = DEFAULT_NATIVE_THREADS,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    check_capacity: bool = True,
    stage_command: str | None = None,
) -> dict[str, Any]:
    plan = build_subject_shape_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        refined_run=refined_run,
        run_name=run_name,
        components=components,
        block_rows=block_rows,
        output_shard_rows=output_shard_rows,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        shard_copy_workers=shard_copy_workers,
        native_threads=native_threads,
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
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    free_bytes = int(shutil.disk_usage(plan.scratch_root).free)
    if check_capacity and free_bytes < plan.estimated_scratch_bytes:
        raise OSError(
            f"Insufficient scratch capacity: need approximately {plan.estimated_scratch_bytes} bytes, "
            f"found {free_bytes} bytes at {plan.scratch_root}."
        )
    native_environment = _configure_native_threads(plan.native_threads)
    try:
        source_root = open_zarr_root(plan.source_zarr, mode="r")
        compute_root = zarr.open_group(str(plan.compute_zarr), mode="w", zarr_format=3)
        compute_summary = write_subject_shape_run_group(
            source_root,
            zarr_path=plan.source_zarr,
            output_root=compute_root,
            output_zarr_path=plan.compute_zarr,
            refined_run=plan.refined_run,
            run_name=plan.run_name,
            components=plan.component_names,
            chunk_size=plan.block_rows,
            execution_backend=plan.execution_backend,
            scheduler=plan.scheduler,
            num_workers=plan.num_workers,
            overwrite=False,
            dry_run=False,
            centerline_crop_to_foreground=True,
            native_threads=plan.native_threads,
            stage_command=stage_command or (" ".join(sys.argv) if sys.argv else "unknown"),
        )
        compute_validation = _validate_subject_shape_run(
            plan.compute_run_path,
            row_count=plan.row_count,
            require_sharded=False,
        )
        if not compute_validation["valid"]:
            raise RuntimeError(f"Node-local compute validation failed: {compute_validation}")
        sharding = copy_completed_run_to_sharded(
            plan.compute_run_path,
            plan.sharded_run,
            row_count_array="row_index/frame_indices",
            shard_rows=plan.output_shard_rows,
            workers=plan.shard_copy_workers,
        )
        sharding_summary = {
            key: value
            for key, value in sharding.items()
            if key not in {"arrays", "shards", "static_arrays"}
        }
        materialization_payload = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "source_access_policy": "authoritative_shared_read_only",
            "node_local_compute": compute_summary,
            "node_local_compute_validation": compute_validation,
            "node_local_sharding": sharding_summary,
            "native_thread_environment": native_environment,
            "capacity": {
                "check_enabled": bool(check_capacity),
                "free_bytes_before_compute": free_bytes,
                "required_bytes_estimate": plan.estimated_scratch_bytes,
            },
        }
        publish = publish_subject_shape_run(
            plan,
            materialization_payload=materialization_payload,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
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
        return scratch_user / job_id / f"palette_subject_shape_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / f"palette_subject_shape_{job_id}_{run_name}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--refined-run")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--component", action="append", dest="components")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--output-shard-rows", type=int, default=DEFAULT_OUTPUT_SHARD_ROWS)
    parser.add_argument(
        "--execution-backend",
        choices=("serial_driver", DASK_WORKER_EXECUTION_BACKEND),
        default=DASK_WORKER_EXECUTION_BACKEND,
    )
    parser.add_argument(
        "--scheduler",
        choices=("single-threaded", "threads", "processes", "distributed"),
        default="processes",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--shard-copy-workers", type=int, default=DEFAULT_SHARD_COPY_WORKERS)
    parser.add_argument("--native-threads", type=int, default=DEFAULT_NATIVE_THREADS)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--no-capacity-check", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    scratch = args.scratch_root or _default_scratch_root(args.run_name)
    result = materialize_subject_shape(
        args.zarr_path,
        scratch_root=scratch,
        refined_run=args.refined_run,
        run_name=args.run_name,
        components=args.components,
        block_rows=int(args.block_rows),
        output_shard_rows=int(args.output_shard_rows),
        execution_backend=str(args.execution_backend),
        scheduler=str(args.scheduler),
        num_workers=int(args.num_workers),
        shard_copy_workers=int(args.shard_copy_workers),
        native_threads=int(args.native_threads),
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
