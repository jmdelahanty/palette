"""Stage subject-shape inputs locally, materialize tail kinematics, and publish.

The authoritative recording Zarr is never used as the compute output store.
Required subject-shape physical files are copied to a minimal node-local Zarr,
the bounded writer runs there, and the completed run group is copied to a
hidden shared-storage sibling before an atomic rename and parent-pointer update.
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
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import zarr

from ...analysis import tail_kinematics_runs as tail_mod
from ...analysis.tail_kinematics_runs import (
    DEFAULT_BLOCK_ROWS,
    DEFAULT_OUTPUT_SHARD_ROWS,
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
from ...analysis_workflows.tail_kinematics_candidate_execution import (
    build_tail_kinematics_logical_hashes_from_array_digests,
)
from ...shared.json_safety import json_attr_safe
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared import tail_coordinate_publication as tail_publication_mod
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr.benchmark_runtime import storage_stats
from ...shared.zarr.manifest_digest import canonical_json_sha256
from ...shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from ...shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)
from ...shared.zarr.storage_profiles import StorageProfile, get_storage_profile
from ...shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.runtime_telemetry import PhaseTelemetry

MATERIALIZATION_SCHEMA_ID = "palette.tail_kinematics_materialization.v1"
STAGING_SCHEMA_ID = "palette.tail_kinematics_source_staging.v1"
PUBLISH_SCHEMA_ID = "palette.tail_kinematics_run_publish.v1"
GROUP_METADATA_NAMES = ("zarr.json", ".zgroup", ".zattrs")
DEFAULT_CAPACITY_MARGIN_BYTES = 512 * 1024 * 1024
TAIL_KINEMATICS_EXECUTION_PHASE_ORDER = (
    "plan",
    "source_staging",
    "scientific_compute",
    "payload_receipt_scan",
    "local_validation",
    "local_consolidation",
    "local_direct_consolidated_comparison",
    "atomic_publication",
    "published_validation",
    "published_direct_consolidated_comparison",
    "decoded_equality",
    "physical_inventory",
    "publication_acceptance_validation",
)
EXECUTION_BINDING_ATTR = "analysis_candidate_execution_binding"
EXECUTION_FAILURE_TOMBSTONE_ATTR = "analysis_candidate_execution_tombstone"
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


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
    requested_output_shard_rows: int
    execution_backend: str
    requested_num_workers: int
    storage_profile_id: str | None
    selected_paths: tuple[str, ...]
    physical_files: tuple[PhysicalFile, ...]
    source_bytes: int
    estimated_output_bytes: int
    inventory_sha256: str
    source_metadata_sha256: str
    source_contract: dict[str, Any]
    staged_source_authority: dict[str, Any]

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
            "requested_output_shard_rows": int(self.requested_output_shard_rows),
            "execution_backend": self.execution_backend,
            "requested_num_workers": int(self.requested_num_workers),
            "storage_profile_id": self.storage_profile_id,
            "byte_planner_candidate": self.storage_profile_id is not None,
            "selected_paths": list(self.selected_paths),
            "physical_file_count": len(self.physical_files),
            "source_bytes": int(self.source_bytes),
            "estimated_output_bytes": int(self.estimated_output_bytes),
            "inventory_sha256": self.inventory_sha256,
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_contract": json_attr_safe(self.source_contract),
            "staged_source_authority_sha256": self.staged_source_authority.get(
                "record_sha256"
            ),
            "staged_source_authority": json_attr_safe(self.staged_source_authority),
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe tail-kinematics run name: {run_name!r}.")
    return value


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _add_group_metadata(
    source_zarr: Path, relative_group: str, selected: set[Path]
) -> None:
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
    required_body_frame = {"forward_axis_xy", "left_axis_xy", "axis_valid"}
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
            f"{run_prefix}/{name}",
            selected,
            selected_paths,
            required=True,
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
        PhysicalFile(
            relative_path=_relative_path(source_zarr, path),
            size_bytes=int(path.stat().st_size),
        )
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
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
    storage_profile: StorageProfile | None = None,
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
        raise ValueError(
            "Scratch root must not be inside the authoritative source Zarr."
        )
    if int(tail_angle_sample_count) < 2:
        raise ValueError("tail_angle_sample_count must be >= 2.")
    if int(block_rows) <= 0:
        raise ValueError("block_rows must be positive.")
    if int(output_shard_rows) <= 0:
        raise ValueError("output_shard_rows must be positive.")
    backend = str(execution_backend).strip().lower()
    if backend not in {"serial", "process_shards"}:
        raise ValueError(f"Unsupported execution backend: {execution_backend!r}.")
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive.")
    if storage_profile is not None:
        if not isinstance(storage_profile, StorageProfile):
            raise TypeError("storage_profile must be an explicit StorageProfile.")
        if storage_profile.profile_id != tail_mod.TAIL_KINEMATICS_CANDIDATE_PROFILE_ID:
            raise ValueError(
                "Tail-kinematics materialization supports only the explicit "
                f"{tail_mod.TAIL_KINEMATICS_CANDIDATE_PROFILE_ID!r} candidate."
            )
        if backend != "serial" or int(num_workers) != 1:
            raise ValueError(
                "Tail-kinematics byte-planner candidate materialization requires "
                "one serial writer."
            )

    root = open_zarr_root(source, mode="r")
    resolved_shape_run, shape_group, sources = _resolve_tail_kinematics_sources(
        root, shape_run
    )
    target_run = (
        source / "analysis" / "tail_kinematics_runs" / _validate_run_name(run_name)
    )
    if target_run.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target_run}"
        )
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
        requested_output_shard_rows=int(output_shard_rows),
        execution_backend=backend,
        requested_num_workers=int(num_workers),
        storage_profile_id=(
            storage_profile.profile_id if storage_profile is not None else None
        ),
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
            | {
                "canonical_publication_manifest_sha256": (
                    sources.source_publication_manifest_sha256
                ),
                "staged_source_authority_sha256": sources.source_authority.get(
                    "record_sha256"
                ),
            }
        ),
        staged_source_authority=json_attr_safe(sources.source_authority),
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


def _validate_file_inventory(
    root: Path, expected: Sequence[PhysicalFile]
) -> dict[str, Any]:
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
    required_bytes = int(
        plan.source_bytes + plan.estimated_output_bytes + DEFAULT_CAPACITY_MARGIN_BYTES
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
    validation = _validate_file_inventory(plan.staged_zarr, plan.physical_files)
    if not bool(validation["valid"]):
        raise RuntimeError(f"Staged source inventory validation failed: {validation}")

    staged_root = open_zarr_root(plan.staged_zarr, mode="r")
    staged_shape_run, _shape_group, staged_sources = _resolve_tail_kinematics_sources(
        staged_root,
        plan.shape_run,
        _staged_source_authority=plan.staged_source_authority,
        _verify_staged_payload=False,
    )
    if staged_shape_run != plan.shape_run or int(staged_sources.row_count) != int(
        plan.row_count
    ):
        raise RuntimeError(
            "Staged subject-shape logical validation did not match the staging plan."
        )
    effective_block_rows = tail_mod._effective_block_rows(
        row_count=int(plan.row_count),
        requested_block_rows=int(plan.requested_block_rows),
    )
    staged_input_integrity_receipt = (
        tail_mod.build_tail_kinematics_staged_input_integrity_receipt(
            staged_sources.shape_group,
            run_name=plan.shape_run,
            authority=plan.staged_source_authority,
            chunk_rows=int(effective_block_rows),
            read_workers=max(1, min(4, int(plan.requested_num_workers))),
        )
    )

    payload = {
        "schema_id": STAGING_SCHEMA_ID,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "duration_seconds": duration,
        "mib_per_second": (
            (int(plan.source_bytes) / (1024 * 1024)) / duration
            if duration > 0.0
            else None
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
        "staged_source_authority_sha256": plan.staged_source_authority.get(
            "record_sha256"
        ),
        "staged_source_authority": json_attr_safe(plan.staged_source_authority),
        "staged_input_integrity_receipt_sha256": (
            staged_input_integrity_receipt["record_sha256"]
        ),
        "staged_input_integrity_receipt": json_attr_safe(
            staged_input_integrity_receipt
        ),
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


def _validate_tail_run(
    path: Path, *, row_count: int, sample_count: int
) -> dict[str, Any]:
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
        "instance_key": (int(row_count),),
        "source_crop_row_ids": (int(row_count),),
        "source_acquisition_frame_index": (int(row_count),),
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
    if str(attrs.get("execution_backend")) == "process_shards" and int(
        attrs.get("completed_worker_task_count", -1)
    ) != int(attrs.get("worker_task_count", -2)):
        errors.append("worker task completion mismatch")
    if attrs.get("stage_selector_eligible") is True and not isinstance(
        attrs.get("tail_coordinate_publication_manifest_sha256"), str
    ):
        errors.append("selector-eligible run lacks tail coordinate publication seal")
    if attrs.get("byte_planner_adopted") is True:
        errors.extend(tail_mod.validate_tail_kinematics_storage_receipt(group))
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(row_count),
        "sample_count": int(sample_count),
        "valid_row_count": valid_count,
        "invalid_row_count": invalid_count,
        "output_row_chunk": attrs.get("output_row_chunk"),
        "requested_output_shard_rows": attrs.get("requested_output_shard_rows"),
        "effective_output_shard_rows": attrs.get("effective_output_shard_rows"),
        "output_shard_rows": attrs.get("output_shard_rows"),
        "output_shard_count": attrs.get("output_shard_count"),
        "completed_worker_task_count": attrs.get("completed_worker_task_count"),
    }


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index for index, name in enumerate(TAIL_KINEMATICS_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Tail-kinematics telemetry contains an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


def tombstone_tail_kinematics_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one exact execution-owned candidate without touching selectors."""

    archive = Path(source_zarr).expanduser().resolve()
    name = _validate_run_name(run_name)
    expected_binding = json_attr_safe(dict(expected_execution_binding))
    if not expected_binding:
        raise ValueError("expected_execution_binding must be nonempty.")
    payload = {
        "schema_id": "palette.analysis_candidate_execution_tombstone",
        "schema_version": 1,
        "execution_binding": expected_binding,
        "failure_phase": str(failure_phase),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }
    tombstone = {**payload, "payload_sha256": canonical_json_sha256(payload)}
    run_path = f"analysis/tail_kinematics_runs/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root["analysis/tail_kinematics_runs"]
        run = parent.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        if run.attrs.get(EXECUTION_BINDING_ATTR) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone a tail-kinematics candidate owned by another execution."
            )
        if run.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(
                "Refusing to tombstone a selector-eligible tail-kinematics run."
            )
        existing = run.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        status = run.attrs.get(RUN_COMPLETION_STATUS_ATTR)
        if status == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError(
                    "Existing tail-kinematics execution tombstone differs."
                )
        else:
            if status != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Tail-kinematics execution candidate is neither complete nor failed."
                )
            mark_run_failed(
                run,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            run.attrs["stage_selector_eligible"] = False
            run.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        consolidate_metadata_capture_expected_warnings(archive)
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        fresh = open_zarr_root(archive, mode="r")[run_path]
        if (
            fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
            or fresh.attrs.get("stage_selector_eligible") is not False
            or fresh.attrs.get(EXECUTION_BINDING_ATTR) != expected_binding
            or fresh.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR) != tombstone
        ):
            raise RuntimeError(
                "Tail-kinematics execution tombstone did not persist exactly."
            )
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": receipt.declarations_sha256,
    }


def publish_tail_kinematics_run(
    plan: TailKinematicsMaterializationPlan,
    *,
    staging_payload: dict[str, Any],
    payload_scan_receipt: Mapping[str, Any],
    copy_backend: str = "rsync",
    telemetry: PhaseTelemetry | None = None,
    expected_source_logical_hashes: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one completed local run group."""

    byte_planner_candidate = plan.storage_profile_id is not None
    if (
        expected_source_logical_hashes is not None
        or publication_acceptance_validator is not None
    ) and not byte_planner_candidate:
        raise ValueError(
            "Execution acceptance is valid for tail-kinematics candidates only."
        )

    def phase(name: str):
        return telemetry.phase(name) if telemetry is not None else nullcontext()

    source_run = plan.local_run_path
    source_owner = open_zarr_root(source_run, mode="r").attrs.get(
        tail_publication_mod.TAIL_PUBLICATION_OWNER_ATTR
    )
    try:
        expected_publication_owner_uuid = str(uuid.UUID(str(source_owner)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            "Tail-kinematics publication source lacks one valid publication owner."
        ) from exc
    if source_owner != expected_publication_owner_uuid:
        raise RuntimeError(
            "Tail-kinematics publication source owner is not one canonical UUID."
        )
    deferred_activation: list[Any] = []
    publication_acceptance: dict[str, Any] = {}

    def validate(path: Path) -> dict[str, Any]:
        return _validate_tail_run(
            path,
            row_count=plan.row_count,
            sample_count=plan.tail_angle_sample_count,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "tail_kinematics_runs",
            ),
        )

    def after_rename(
        root: zarr.Group,
        run_group: zarr.Group,
        physical_copy: Mapping[str, Any],
    ) -> dict[str, Any]:
        if "tail_coordinate_publication_deferred" not in run_group.attrs:
            raise RuntimeError(
                "Tail-kinematics authoritative binding lacks its staged marker."
            )
        del run_group.attrs["tail_coordinate_publication_deferred"]
        run_group.attrs["stage_selector_eligible"] = False
        publication = tail_mod.publish_tail_kinematics_coordinate_surfaces(
            root,
            run_group,
            payload_scan_receipt=payload_scan_receipt,
            payload_run_path=plan.target_run_path,
            verified_physical_copy=physical_copy,
        )
        return {
            "tail_coordinate_publication": {
                "manifest_sha256": publication.manifest.record_sha256,
                "payload_scan_receipt_sha256": payload_scan_receipt.get(
                    "record_sha256"
                ),
            }
        }

    def complete(
        root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        command = " ".join(sys.argv) if sys.argv else "unknown"
        if "tail_coordinate_publication_deferred" in run_group.attrs:
            raise RuntimeError(
                "Tail-kinematics completion ran before authoritative binding."
            )
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command=command,
            ),
        )
        if not byte_planner_candidate:
            activation = (
                tail_publication_mod.defer_tail_coordinate_publication_activation(
                    root,
                    parent,
                    run_group,
                    run_name=plan.run_name,
                    expected_publication_owner_uuid=expected_publication_owner_uuid,
                )
            )
            deferred_activation[:] = [activation]

    def activate(
        root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if byte_planner_candidate:
            if run_group.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError(
                    "Tail-kinematics storage candidate became selector eligible."
                )
            run_path = f"analysis/tail_kinematics_runs/{plan.run_name}"
            with phase("published_validation"):
                published_validation = validate(plan.target_run_path)
                if not published_validation["valid"]:
                    raise RuntimeError(
                        "Published tail-kinematics candidate is invalid: "
                        f"{published_validation}"
                    )
            tail_mod.consolidate_and_validate_tail_kinematics_metadata(
                root,
                run_path=run_path,
            )
            with phase("published_direct_consolidated_comparison"):
                published_metadata = validate_direct_consolidated_subtree(
                    plan.source_zarr,
                    subtree_path=run_path,
                )
            with phase("decoded_equality"):
                proof = tail_publication_mod.validate_sealed_tail_publication_metadata(
                    root,
                    run_path,
                    expected_selector_eligible=False,
                    expected_publication_owner=expected_publication_owner_uuid,
                    expected_kind="tail_kinematics",
                    payload_run_path=plan.target_run_path,
                )
                published_hashes = (
                    build_tail_kinematics_logical_hashes_from_array_digests(
                        run_group,
                        proof.array_content_sha256,
                    )
                )
                if (
                    expected_source_logical_hashes is not None
                    and published_hashes != dict(expected_source_logical_hashes)
                ):
                    raise RuntimeError(
                        "Published tail-kinematics decoded values differ from the source run."
                    )
            with phase("physical_inventory"):
                published_storage = storage_stats(plan.target_run_path)
                if (
                    published_storage["file_count"] < 1
                    or published_storage["apparent_bytes"] < 1
                    or published_storage["allocated_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published tail-kinematics candidate has no physical payload."
                    )
            publication_acceptance.update(
                {
                    "published_validation": published_validation,
                    "published_direct_consolidated_array_count": int(
                        published_metadata.array_count
                    ),
                    "published_logical_hashes": published_hashes,
                    "published_logical_manifest_sha256": canonical_json_sha256(
                        published_hashes
                    ),
                    "output_storage": published_storage,
                }
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    publication_acceptance["caller_acceptance"] = json_attr_safe(
                        dict(publication_acceptance_validator(root, parent, run_group))
                    )
            return
        if len(deferred_activation) != 1:
            raise RuntimeError(
                "Tail-kinematics publication lacks one deferred activation receipt."
            )
        tail_publication_mod.commit_deferred_tail_coordinate_publication_activation(
            deferred_activation[0],
            root=root,
            parent=parent,
            run=run_group,
        )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=f"analysis/tail_kinematics_runs/{plan.run_name}",
        )
        consolidated_root = zarr.open_group(
            str(plan.source_zarr),
            mode="r",
            use_consolidated=True,
        )
        tail_publication_mod.validate_sealed_tail_publication_metadata(
            consolidated_root,
            f"analysis/tail_kinematics_runs/{plan.run_name}",
            expected_selector_eligible=True,
            expected_publication_owner=expected_publication_owner_uuid,
            expected_kind="tail_kinematics",
            payload_run_path=plan.target_run_path,
        )

    def rollback_activation() -> None:
        if not byte_planner_candidate and deferred_activation:
            tail_publication_mod.rollback_deferred_tail_coordinate_publication_activation(
                deferred_activation[0]
            )

    def repair_failed_candidate_visibility(_target_path: Path) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/tail_kinematics_runs"]
        if byte_planner_candidate:
            if (
                parent.attrs.get("latest") == plan.run_name
                or parent.attrs.get("latest_complete") == plan.run_name
            ):
                raise RuntimeError(
                    "Tail-kinematics storage candidate updated a parent selector."
                )
            return
        if (
            str(parent.attrs.get("latest")) != plan.run_name
            or str(parent.attrs.get("latest_complete")) != plan.run_name
        ):
            raise RuntimeError(
                "Tail-kinematics parent pointers were not updated to the published run."
            )

    result = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=source_run,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="tail-kinematics-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_source_and_output_atomic_run_group_publish",
            rollback_policy=(
                "retain_owner_bound_failed_public_tombstone_and_"
                "stage_specific_receipt_rollback_only"
            ),
            publication_owner_attr=tail_publication_mod.TAIL_PUBLICATION_OWNER_ATTR,
            content_checksum=True,
            # The sealed stage receipt is the only selector rollback authority.
            # A generic pre-copy snapshot can predate an intervening publication.
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        rollback_activation=(None if byte_planner_candidate else rollback_activation),
        repair_failed_publication_visibility=(repair_failed_candidate_visibility),
        after_rename=after_rename,
        payload_metadata={
            "staged_zarr": str(plan.staged_zarr),
            "source_run_path": str(source_run),
            "copy_backend": copy_backend,
            "source_staging": json_attr_safe(staging_payload),
        },
        accept_persisted_activation_on_callback_error=False,
    )
    if byte_planner_candidate:
        result.update(publication_acceptance)
        result["registry_updated"] = False
    return result


def materialize_tail_kinematics(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    shape_run: str | None,
    run_name: str,
    tail_angle_sample_count: int = DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = "serial",
    num_workers: int = 1,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    check_capacity: bool = True,
    stage_command: str | None = None,
    storage_profile: StorageProfile | None = None,
    execution_binding: Mapping[str, Any] | None = None,
    expected_source_logical_hashes: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Execute or plan the complete staged materialization workflow."""

    execution_requested = any(
        value is not None
        for value in (
            execution_binding,
            expected_source_logical_hashes,
            publication_acceptance_validator,
        )
    )
    if execution_requested and storage_profile is None:
        raise ValueError(
            "Typed execution evidence requires one explicit storage candidate profile."
        )
    if execution_requested and not isinstance(execution_binding, Mapping):
        raise ValueError("Typed execution requires one nonempty execution binding.")
    normalized_binding = (
        json_attr_safe(dict(execution_binding))
        if isinstance(execution_binding, Mapping)
        else None
    )
    if execution_requested and not normalized_binding:
        raise ValueError("Typed execution binding must be nonempty.")
    telemetry = PhaseTelemetry(
        materializer="tail_kinematics_candidate",
        context={
            "run_name": str(run_name),
            "shape_run": shape_run,
            "storage_profile_id": (
                storage_profile.profile_id if storage_profile is not None else None
            ),
            "execution_requested": execution_requested,
        },
    )
    with telemetry.phase("plan"):
        plan = build_tail_kinematics_materialization_plan(
            source_zarr,
            scratch_root=scratch_root,
            shape_run=shape_run,
            run_name=run_name,
            tail_angle_sample_count=tail_angle_sample_count,
            block_rows=block_rows,
            output_shard_rows=output_shard_rows,
            execution_backend=execution_backend,
            num_workers=num_workers,
            storage_profile=storage_profile,
        )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        result["runtime_telemetry"] = _ordered_runtime_telemetry(telemetry)
        return result

    succeeded = False
    try:
        with telemetry.phase("source_staging"):
            staging = stage_tail_kinematics_sources(
                plan,
                copy_backend=copy_backend,
                check_capacity=check_capacity,
            )
        with telemetry.phase("scientific_compute"):
            local_root = open_zarr_root(plan.staged_zarr, mode="a")
            local_summary = write_tail_kinematics_run_group(
                local_root,
                shape_run=plan.shape_run,
                run_name=plan.run_name,
                tail_angle_sample_count=plan.tail_angle_sample_count,
                block_rows=plan.requested_block_rows,
                output_shard_rows=plan.requested_output_shard_rows,
                execution_backend=plan.execution_backend,
                num_workers=plan.requested_num_workers,
                worker_zarr_path=plan.staged_zarr,
                overwrite=False,
                dry_run=False,
                stage_command=stage_command
                or (" ".join(sys.argv) if sys.argv else "unknown"),
                storage_profile=(
                    get_storage_profile(plan.storage_profile_id)
                    if plan.storage_profile_id is not None
                    else None
                ),
                _staged_source_authority=plan.staged_source_authority,
                _staged_input_integrity_receipt=staging[
                    "staged_input_integrity_receipt"
                ],
            )
        local_run = local_root["analysis"]["tail_kinematics_runs"][plan.run_name]
        local_run.attrs["node_local_source_staging"] = staging
        if normalized_binding is not None:
            local_run.attrs[EXECUTION_BINDING_ATTR] = normalized_binding
            local_run.attrs["storage_candidate_profile_promoted"] = False
        with telemetry.phase("payload_receipt_scan"):
            payload_scan_receipt = (
                tail_publication_mod.build_tail_kinematics_payload_scan_receipt(
                    local_run,
                    workers=max(1, min(4, int(plan.requested_num_workers))),
                )
            )
        local_validation: dict[str, Any] | None = None
        local_hashes: dict[str, object] | None = None
        local_metadata: Any | None = None
        if plan.storage_profile_id is not None:
            with telemetry.phase("local_validation"):
                local_validation = _validate_tail_run(
                    plan.local_run_path,
                    row_count=plan.row_count,
                    sample_count=plan.tail_angle_sample_count,
                )
                if not local_validation["valid"]:
                    raise RuntimeError(
                        "Local tail-kinematics candidate is invalid: "
                        f"{local_validation}"
                    )
                local_hashes = build_tail_kinematics_logical_hashes_from_array_digests(
                    local_run,
                    payload_scan_receipt["array_content_sha256"],
                )
                if expected_source_logical_hashes is not None and local_hashes != dict(
                    expected_source_logical_hashes
                ):
                    raise RuntimeError(
                        "Local tail-kinematics decoded values differ from the source run."
                    )
            with telemetry.phase("local_consolidation"):
                consolidate_metadata_capture_expected_warnings(plan.staged_zarr)
            with telemetry.phase("local_direct_consolidated_comparison"):
                local_metadata = validate_direct_consolidated_subtree(
                    plan.staged_zarr,
                    subtree_path=f"analysis/tail_kinematics_runs/{plan.run_name}",
                )
        with telemetry.phase("atomic_publication"):
            publish = publish_tail_kinematics_run(
                plan,
                staging_payload=staging,
                payload_scan_receipt=payload_scan_receipt,
                copy_backend=copy_backend,
                telemetry=telemetry,
                expected_source_logical_hashes=expected_source_logical_hashes,
                publication_acceptance_validator=(publication_acceptance_validator),
            )
        result.update(
            {
                "status": "complete",
                "staging": staging,
                "local_materialization": local_summary,
                "payload_scan_receipt_sha256": payload_scan_receipt.get(
                    "record_sha256"
                ),
                "local_validation": local_validation,
                "local_direct_consolidated_array_count": (
                    int(local_metadata.array_count)
                    if local_metadata is not None
                    else None
                ),
                "source_logical_manifest_sha256": (
                    canonical_json_sha256(expected_source_logical_hashes)
                    if expected_source_logical_hashes is not None
                    else (
                        canonical_json_sha256(local_hashes)
                        if local_hashes is not None
                        else None
                    )
                ),
                "local_logical_manifest_sha256": (
                    canonical_json_sha256(local_hashes)
                    if local_hashes is not None
                    else None
                ),
                "published_logical_manifest_sha256": publish.get(
                    "published_logical_manifest_sha256"
                ),
                "published_direct_consolidated_array_count": publish.get(
                    "published_direct_consolidated_array_count"
                ),
                "output_storage": publish.get("output_storage"),
                "caller_acceptance": publish.get("caller_acceptance"),
                "publish": publish,
            }
        )
        result["runtime_telemetry"] = _ordered_runtime_telemetry(telemetry)
        succeeded = True
        return result
    except BaseException as exc:
        try:
            setattr(
                exc, "palette_runtime_telemetry", _ordered_runtime_telemetry(telemetry)
            )
        except BaseException:
            pass
        raise
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.is_dir():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_tail_kinematics_{run_name}"
    return (
        Path(os.environ.get("TMPDIR") or "/tmp")
        / f"palette_tail_kinematics_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage subject-shape arrays locally, materialize tail kinematics, and atomically publish."
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Authoritative Palette analysis Zarr on shared storage.",
    )
    parser.add_argument(
        "--shape-run", help="Exact subject-shape source run; defaults to latest."
    )
    parser.add_argument(
        "--run-name", required=True, help="New authoritative tail-kinematics run name."
    )
    parser.add_argument(
        "--scratch-root", type=Path, help="Unique node-local staging directory."
    )
    parser.add_argument(
        "--tail-angle-sample-count",
        type=int,
        default=DEFAULT_TAIL_ANGLE_SAMPLE_COUNT,
    )
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
    )
    parser.add_argument(
        "--execution-backend",
        choices=("serial", "process_shards"),
        default="serial",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--storage-profile",
        choices=(tail_mod.TAIL_KINEMATICS_CANDIDATE_PROFILE_ID,),
        help=(
            "Opt into the unpromoted byte-planned candidate. This requires one "
            "serial writer and publishes no latest/latest_complete selector."
        ),
    )
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument(
        "--apply", action="store_true", help="Execute; default is a read-only plan."
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep scratch after successful publication.",
    )
    parser.add_argument("--no-capacity-check", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path, normally on shared storage.",
    )
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
        output_shard_rows=int(args.output_shard_rows),
        execution_backend=str(args.execution_backend),
        num_workers=int(args.num_workers),
        copy_backend=str(args.copy_backend),
        apply=bool(args.apply),
        keep_scratch=bool(args.keep_scratch),
        check_capacity=not bool(args.no_capacity_check),
        storage_profile=(
            get_storage_profile(args.storage_profile)
            if args.storage_profile is not None
            else None
        ),
    )
    if args.report is not None:
        _write_json_atomic(args.report.expanduser().resolve(), result)
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
