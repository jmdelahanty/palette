"""Stage exact eye-angle inputs, compute locally, shard, and publish atomically.

The production eye-angle path consumes completed subject-shape eye geometry and
the exact canonical base keypoints sealed by that publication. Only the physical
files backing those resolved
arrays are copied to node-local storage.  The existing scientific writer then
runs entirely against that staged Zarr. The established path converts its
completed output to indexed Zarr v3 shards; the explicit access-aware candidate
is already written through the shared byte planner and is not reshared. Both
paths receive decoded validation before the shared atomic publisher installs
the result in the authoritative recording.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import zarr

from ...analysis import eye_angle_analysis as eye_writer
from ...analysis.eye_angle_schema import (
    EyeAngleDimensions,
    validate_eye_angle_compact_run,
    validate_eye_angle_value_aliases,
)
from ...analysis.eye_angle_storage import (
    EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    EYE_ANGLE_STORAGE_PROFILE_CHOICES,
    is_eye_angle_storage_candidate,
    validate_eye_angle_candidate_storage,
    validate_eye_angle_direct_consolidated_storage,
)
from ..eye_angle_candidate_execution import compute_eye_angle_logical_hashes
from ...shared.derived_analysis_registry_status import (
    emit_eye_angle_stage_completion,
)
from ...shared.eye_geometry_source import EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
from ...shared.json_safety import json_attr_safe
from ...shared.metadata import get_fps
from ...shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr.benchmark_runtime import storage_stats
from ...shared.zarr.manifest_digest import canonical_json_sha256
from ...shared.zarr.metadata_equivalence import validate_direct_consolidated_subtree
from ...shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)
from ...shared.zarr_sharded_copy import ShardedArrayLayout, copy_completed_run_to_sharded
from ...shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from .atomic_run_publisher import (
    ATOMIC_PUBLICATION_TOMBSTONE_ATTR,
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from .runtime_telemetry import PhaseTelemetry, require_runtime_telemetry


MATERIALIZATION_SCHEMA_ID = "palette.eye_angle_materialization.v1"
STAGING_SCHEMA_ID = "palette.eye_angle_source_staging.v1"
PUBLISH_SCHEMA_ID = "palette.eye_angle_run_publish.v1"
SOURCE_REVISION_AUDIT_SCHEMA_ID = "palette.eye_angle_source_revision_audit.v1"
GROUP_METADATA_NAMES = ("zarr.json", ".zgroup", ".zattrs")
DEFAULT_CHUNK_ROWS = 8_192
DEFAULT_ANGLE_CHUNK_ROWS = eye_writer.EYE_ANGLE_DENSE_CHUNK_ROWS
DEFAULT_ANGLE_CHUNK_COLUMNS = eye_writer.EYE_ANGLE_DENSE_CHUNK_COLUMNS
DEFAULT_OUTPUT_SHARD_ROWS = 131_072
DEFAULT_ANGLE_SHARD_COLUMNS = 32
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
EYE_ANGLE_EXECUTION_PHASE_ORDER = (
    "plan",
    "source_staging",
    "scientific_compute",
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
    storage_profile_id: str
    latest_before: str | None
    latest_complete_before: str | None
    row_count: int
    frame_count: int
    chunk_rows: int
    angle_chunk_rows: int
    angle_chunk_columns: int
    output_shard_rows: int
    angle_shard_columns: int
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
    staged_input_integrity_receipt: dict[str, Any]

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

    @property
    def publication_run_path(self) -> Path:
        """Return the already-final physical run selected for publication."""

        if is_eye_angle_storage_candidate(self.storage_profile_id):
            return self.local_run_path
        return self.sharded_run

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
                "storage_profile_id": self.storage_profile_id,
                "publication_run_path": str(self.publication_run_path),
                "latest_before": self.latest_before,
                "latest_complete_before": self.latest_complete_before,
                "row_count": self.row_count,
                "frame_count": self.frame_count,
                "chunk_rows": self.chunk_rows,
                "angle_chunk_rows": self.angle_chunk_rows,
                "angle_chunk_columns": self.angle_chunk_columns,
                "output_shard_rows": self.output_shard_rows,
                "angle_shard_columns": self.angle_shard_columns,
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
                "staged_input_integrity_receipt_sha256": (
                    self.staged_input_integrity_receipt.get("record_sha256")
                ),
                "staged_input_integrity_receipt": (
                    self.staged_input_integrity_receipt
                ),
                "full_selected_scientific_input_content_hash": True,
                "source_revision_assurance": (
                    "canonical subject-shape authority plus a two-pass, exact, "
                    "chunked content receipt for every staged worker input"
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


def _sealed_output_identity_digests(
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    """Return output-axis digests from the validated canonical keypoint proof."""

    if not isinstance(receipt, Mapping):
        raise ValueError("Eye-angle materialization lacks its staged input receipt.")
    authority = eye_writer._canonical_staged_keypoint_authority(
        receipt.get("canonical_keypoint_authority")
    )
    if (
        receipt.get("canonical_keypoint_authority_sha256")
        != authority["record_sha256"]
    ):
        raise ValueError(
            "Eye-angle materialization receipt names another keypoint authority."
        )
    arrays = authority["arrays"]
    return {
        "instance_key": str(arrays["instance_key"]["content_sha256"]),
        "source_acquisition_frame_index": str(
            arrays["source_acquisition_frame_index"]["content_sha256"]
        ),
    }


def _require_complete_source(group: zarr.Group, *, label: str) -> None:
    if str(group.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError(f"{label} must be a completed immutable input run.")


def _resolve_source_plan(
    source_zarr: Path,
    *,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    staged_input_integrity_receipt: Mapping[str, Any] | None = None,
    staged_subject_shape_subset: bool = False,
    verify_staged_payload: bool = True,
) -> tuple[
    Any,
    dict[str, Any],
    tuple[str, ...],
    tuple[PhysicalFile, ...],
    float | None,
    int,
]:
    root = open_zarr_root(source_zarr, mode="r")
    staged_subject_shape_authority = (
        eye_writer._staged_subject_shape_authority_from_input_receipt(
            staged_input_integrity_receipt
        )
        if staged_input_integrity_receipt is not None
        and staged_subject_shape_subset
        else None
    )
    staged_keypoint_authority = (
        eye_writer._staged_keypoint_authority_from_input_receipt(
            staged_input_integrity_receipt
        )
        if staged_input_integrity_receipt is not None
        and staged_subject_shape_subset
        else None
    )
    context = eye_writer._resolve_eye_angle_inputs(
        root,
        subject_shape_run=subject_shape_run,
        refined_subject_run=None,
        keypoint_run=keypoint_run,
        _staged_subject_shape_authority=staged_subject_shape_authority,
        _staged_keypoint_authority=staged_keypoint_authority,
        _verify_staged_payload=verify_staged_payload,
    )
    if staged_input_integrity_receipt is not None:
        eye_writer._validate_staged_eye_angle_input_integrity_receipt(
            context,
            staged_input_integrity_receipt,
            verify_payload=verify_staged_payload,
        )
    geometry = context.eye_geometry
    if geometry.stage_group != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE:
        raise ValueError(
            "The production eye-angle materializer requires completed "
            "analysis/subject_shape_runs eye geometry."
        )
    _require_complete_source(geometry.group, label="Subject-shape source")
    if context.keypoint_source_mode != eye_writer.EYE_ANGLE_KEYPOINT_SOURCE_CANONICAL:
        raise ValueError(
            "The production eye-angle materializer accepts canonical base keypoints only."
        )
    _require_complete_source(context.kp_group, label="Canonical base-keypoint source")

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
    if staged_input_integrity_receipt is not None:
        verified_frame_indices = eye_writer._load_validated_staged_frame_indices(
            context,
            staged_input_integrity_receipt,
        )
        if verified_frame_indices.shape[0] != int(
            context.eye_geometry.ellipse_params.shape[0]
        ):
            raise RuntimeError("Verified acquisition-frame row count changed.")
        resolved_frame_count = int(context.source_total_frames or 0)
        resolved_metadata_fps = staged_input_integrity_receipt[
            "scientific_parameters"
        ]["fps"]
    else:
        resolved_frame_count = int(context.source_total_frames or 0)
        resolved_metadata_fps = get_fps(root)
    return (
        context,
        source_contracts,
        tuple(selected_arrays),
        files,
        resolved_metadata_fps,
        resolved_frame_count,
    )


def build_eye_angle_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    run_name: str,
    storage_profile: str = EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    angle_chunk_rows: int = DEFAULT_ANGLE_CHUNK_ROWS,
    angle_chunk_columns: int = DEFAULT_ANGLE_CHUNK_COLUMNS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    angle_shard_columns: int = DEFAULT_ANGLE_SHARD_COLUMNS,
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
    scratch_inside_source = False
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        scratch_inside_source = True
    source_inside_scratch = False
    try:
        source.relative_to(scratch)
    except ValueError:
        pass
    else:
        source_inside_scratch = True
    if scratch_inside_source or source_inside_scratch:
        raise ValueError(
            "Scratch root and authoritative source Zarr must be disjoint after "
            "resolving symlinks; equality and either containment direction are "
            "unsafe."
        )
    positive_values = (
        chunk_rows,
        angle_chunk_rows,
        angle_chunk_columns,
        output_shard_rows,
        angle_shard_columns,
        num_workers,
        shard_workers,
        native_threads,
    )
    if min(int(value) for value in positive_values) <= 0:
        raise ValueError("Chunk, shard, worker, and native-thread values must be positive.")
    if int(angle_chunk_columns) < 3:
        raise ValueError(
            "angle_chunk_columns must be at least 3 to preserve left/right/binocular bundles."
        )
    backend = eye_writer._normalize_execution_backend(execution_backend)
    scheduler_key = eye_writer._normalize_scheduler(scheduler)
    storage_profile_id = str(storage_profile)
    if storage_profile_id not in EYE_ANGLE_STORAGE_PROFILE_CHOICES:
        raise ValueError(
            f"Unsupported eye-angle storage profile {storage_profile_id!r}; "
            f"expected one of {EYE_ANGLE_STORAGE_PROFILE_CHOICES!r}."
        )
    if is_eye_angle_storage_candidate(storage_profile_id) and (
        backend != eye_writer.SERIAL_EXECUTION_BACKEND
    ):
        raise ValueError(
            "The byte-planned eye-angle candidate requires serial_driver so "
            "one writer owns every complete physical shard."
        )
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
    source_root = open_zarr_root(source, mode="r")
    existing_parent = source_root.get("analysis/eye_angle_runs")
    latest_before = (
        existing_parent.attrs.get("latest")
        if isinstance(existing_parent, zarr.Group)
        else None
    )
    latest_complete_before = (
        existing_parent.attrs.get("latest_complete")
        if isinstance(existing_parent, zarr.Group)
        else None
    )
    target = source / "analysis" / "eye_angle_runs" / resolved_name
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target}")
    resolved_fps = float(fps) if fps is not None else (
        float(metadata_fps) if metadata_fps is not None and float(metadata_fps) > 0 else None
    )
    resolved_fps_source = (
        "cli_override"
        if fps is not None
        else "authoritative_recording_metadata"
        if resolved_fps is not None
        else "unavailable"
    )
    row_count = int(context.eye_geometry.ellipse_params.shape[0])
    staged_subject_shape_authority = context.eye_geometry.source_authority
    if (
        context.eye_geometry.source_authority_mode != "canonical_publication"
        or not isinstance(staged_subject_shape_authority, Mapping)
        or not isinstance(staged_subject_shape_authority.get("record_sha256"), str)
    ):
        raise ValueError(
            "The production eye-angle materializer requires one digest-bound "
            "canonical subject-shape source authority."
        )
    staged_input_integrity_receipt = dict(
        json_attr_safe(
            eye_writer._build_staged_eye_angle_input_integrity_receipt(
                context,
                chunk_rows=int(chunk_rows),
                fps=resolved_fps,
                fps_source=resolved_fps_source,
            )
        )
    )
    if staged_input_integrity_receipt.get("source_contract_sha256") != _json_digest(
        contracts
    ):
        raise RuntimeError(
            "Staged input integrity receipt differs from the resolved source contract."
        )
    sealed_frame_indices = eye_writer._load_validated_staged_frame_indices(
        context,
        staged_input_integrity_receipt,
    )
    if sealed_frame_indices.shape != (row_count,):
        raise RuntimeError("Sealed acquisition-frame index row count changed.")
    if context.source_total_frames is None:
        raise RuntimeError("Canonical eye inputs lack a sealed full-video frame extent.")
    frame_count = int(context.source_total_frames)
    if fps is None:
        confirmed_metadata_fps = get_fps(open_zarr_root(source, mode="r"))
        confirmed_resolved_fps = (
            float(confirmed_metadata_fps)
            if confirmed_metadata_fps is not None
            and float(confirmed_metadata_fps) > 0
            else None
        )
        if confirmed_resolved_fps != resolved_fps:
            raise RuntimeError(
                "Recording FPS changed while the staged input receipt was built."
            )
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
        source_keypoint_run=context.keypoint_run_name,
        run_name=resolved_name,
        storage_profile_id=storage_profile_id,
        latest_before=(
            str(latest_before) if latest_before is not None else None
        ),
        latest_complete_before=(
            str(latest_complete_before)
            if latest_complete_before is not None
            else None
        ),
        row_count=row_count,
        frame_count=frame_count,
        chunk_rows=int(chunk_rows),
        angle_chunk_rows=int(angle_chunk_rows),
        angle_chunk_columns=int(angle_chunk_columns),
        output_shard_rows=int(output_shard_rows),
        angle_shard_columns=int(angle_shard_columns),
        execution_backend=backend,
        scheduler=scheduler_key,
        num_workers=int(num_workers),
        shard_workers=int(shard_workers),
        native_threads=int(native_threads),
        fps=resolved_fps,
        fps_source=resolved_fps_source,
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
        staged_input_integrity_receipt=staged_input_integrity_receipt,
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
            staged_input_integrity_receipt=plan.staged_input_integrity_receipt,
            staged_subject_shape_subset=False,
            verify_staged_payload=True,
        )
        observed_contract_sha256 = _json_digest(contracts)
        if context.eye_geometry.run_name != plan.subject_shape_run:
            errors.append("resolved subject-shape run changed")
        if context.keypoint_run_name != plan.keypoint_run:
            errors.append("resolved canonical base-keypoint run changed")
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
            "full_selected_scientific_input_content_hash": True,
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
        staged_input_integrity_receipt=plan.staged_input_integrity_receipt,
        staged_subject_shape_subset=True,
        verify_staged_payload=True,
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
            "source_authority_mode": "digest_bound_staged_subset",
            "staged_input_integrity_receipt_sha256": (
                plan.staged_input_integrity_receipt.get("record_sha256")
            ),
            "staged_input_integrity_receipt": (
                plan.staged_input_integrity_receipt
            ),
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
    expected_instance_key_sha256: str,
    expected_acquisition_frame_index_sha256: str,
    require_sharded: bool,
    expected_angle_chunk_rows: int | None,
    expected_angle_chunk_columns: int | None,
    expected_angle_shard_rows: int | None,
    expected_angle_shard_columns: int | None,
    storage_profile_id: str = EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
) -> dict[str, Any]:
    group = open_zarr_root(path, mode="r")
    attrs = group.attrs
    errors: list[str] = []
    exact_schema_issues = validate_eye_angle_compact_run(group)
    errors.extend(
        "exact compact-v7 contract: "
        f"{issue.code}:{issue.path}:{issue.message}"
        for issue in exact_schema_issues
    )
    if str(attrs.get("schema_id")) != eye_writer.EYE_ANGLE_RUN_SCHEMA_ID:
        errors.append("schema_id mismatch")
    if int(attrs.get("schema_version", -1)) != eye_writer.EYE_ANGLE_RUN_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if str(attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if str(attrs.get("layout")) != eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        errors.append("layout is not compact_dense_v2")
    column_order = attrs.get("angle_column_order_contract")
    if not isinstance(column_order, dict) or (
        str(column_order.get("schema_id"))
        != eye_writer.EYE_ANGLE_COLUMN_ORDER_SCHEMA_ID
        or str(column_order.get("profile"))
        != eye_writer.EYE_ANGLE_COLUMN_ORDER_PROFILE
        or bool(column_order.get("physical_index_semantics", True))
    ):
        errors.append("angle column-order contract mismatch")
    if int(attrs.get("num_detections", -1)) != int(row_count):
        errors.append("num_detections mismatch")
    observed_frame_count = int(attrs.get("num_frames", -1))
    if observed_frame_count != int(frame_count):
        errors.append("num_frames mismatch")

    output_schema = attrs.get("eye_angle_output_schema")
    algorithm = attrs.get("eye_angle_algorithm_contract")
    errors.extend(eye_writer.validate_eye_angle_persisted_contract_manifests(attrs))
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
        if index_path == "angle_channel_index" and (
            str(index_group.attrs.get("logical_lookup")) != "name"
            or str(index_group.attrs.get("physical_order_profile"))
            != eye_writer.EYE_ANGLE_COLUMN_ORDER_PROFILE
        ):
            errors.append("angle_channel_index does not declare name-based lookup")
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
                continue
            if (
                index_path == "angle_channel_index"
                and expected_angle_chunk_rows is not None
                and expected_angle_chunk_columns is not None
            ):
                expected_chunks = (
                    min(max(1, int(expected_angle_chunk_rows)), max(1, expected_rows)),
                    min(max(1, int(expected_angle_chunk_columns)), max(1, len(names))),
                )
                observed_chunks = tuple(int(value) for value in array.chunks)
                if observed_chunks != expected_chunks:
                    errors.append(
                        f"{array_path}: expected angle chunks {expected_chunks}, "
                        f"observed {observed_chunks}"
                    )
                if (
                    require_sharded
                    and expected_angle_shard_rows is not None
                    and expected_angle_shard_columns is not None
                ):
                    requested_outer = (
                        max(1, int(expected_angle_shard_rows)),
                        max(1, int(expected_angle_shard_columns)),
                    )
                    expected_outer = tuple(
                        int(math.ceil(requested / chunk) * chunk)
                        for requested, chunk in zip(requested_outer, expected_chunks)
                    )
                    observed_outer = tuple(int(value) for value in array.shards)
                    if observed_outer != expected_outer:
                        errors.append(
                            f"{array_path}: expected angle shards {expected_outer}, "
                            f"observed {observed_outer}"
                        )

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
        "support/instance_key": (row_count,),
        "support/source_acquisition_frame_index": (row_count,),
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
    errors.extend(
        f"{issue.code}:{issue.path}:{issue.message}"
        for issue in validate_eye_angle_value_aliases(group)
    )
    instance_key = group.get("support/instance_key")
    acquisition_index = group.get("support/source_acquisition_frame_index")
    frame_alias = group.get("support/frame_indices")
    observed_instance_key_sha256 = None
    observed_acquisition_frame_index_sha256 = None
    observed_frame_alias_sha256 = None
    if isinstance(instance_key, zarr.Array):
        instance_dtype_valid = np.dtype(instance_key.dtype) == np.dtype("<u8")
        if not instance_dtype_valid:
            errors.append("support/instance_key must be exact uint64")
        if (
            instance_key.attrs.get("identity_domain") != "observation_instance"
            or instance_key.attrs.get("identity_mode") != "instance_key"
            or instance_key.attrs.get("row_axis") != eye_writer.EYE_ANGLE_ROW_AXIS
        ):
            errors.append("support/instance_key identity attrs mismatch")
        if instance_dtype_valid:
            observed_instance_key_sha256 = eye_writer.array_values_sha256(
                instance_key
            )
            if observed_instance_key_sha256 != expected_instance_key_sha256:
                errors.append(
                    "support/instance_key differs from sealed canonical source"
                )
    if isinstance(acquisition_index, zarr.Array):
        acquisition_dtype_valid = np.dtype(acquisition_index.dtype) == np.dtype(
            "<i8"
        )
        if not acquisition_dtype_valid:
            errors.append(
                "support/source_acquisition_frame_index must be exact int64"
            )
        if (
            acquisition_index.attrs.get("value_kind")
            != "source_acquisition_frame_index"
            or acquisition_index.attrs.get("row_axis")
            != eye_writer.EYE_ANGLE_ROW_AXIS
        ):
            errors.append(
                "support/source_acquisition_frame_index attrs mismatch"
            )
        if acquisition_dtype_valid:
            observed_acquisition_frame_index_sha256 = (
                eye_writer.array_values_sha256(acquisition_index)
            )
            if (
                observed_acquisition_frame_index_sha256
                != expected_acquisition_frame_index_sha256
            ):
                errors.append(
                    "support/source_acquisition_frame_index differs from sealed canonical source"
                )
    if isinstance(frame_alias, zarr.Array):
        alias_dtype_valid = np.dtype(frame_alias.dtype) == np.dtype("<i8")
        if not alias_dtype_valid:
            errors.append("support/frame_indices must be exact int64")
        if (
            frame_alias.attrs.get("compatibility_alias_of")
            != "support/source_acquisition_frame_index"
            or frame_alias.attrs.get("values_must_equal_canonical") is not True
            or frame_alias.attrs.get("value_kind")
            != "source_acquisition_frame_index"
            or frame_alias.attrs.get("row_axis")
            != eye_writer.EYE_ANGLE_ROW_AXIS
        ):
            errors.append("support/frame_indices compatibility attrs mismatch")
        if alias_dtype_valid:
            observed_frame_alias_sha256 = eye_writer.array_values_sha256(frame_alias)
            if observed_frame_alias_sha256 != expected_acquisition_frame_index_sha256:
                errors.append(
                    "support/frame_indices differs from sealed canonical acquisition frames"
                )
    if (
        observed_acquisition_frame_index_sha256 is not None
        and observed_frame_alias_sha256 is not None
        and observed_acquisition_frame_index_sha256 != observed_frame_alias_sha256
    ):
        errors.append(
            "support/frame_indices differs from canonical source_acquisition_frame_index"
        )

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
    candidate_storage_issues = ()
    if is_eye_angle_storage_candidate(storage_profile_id):
        column_order_contract = attrs.get("angle_column_order_contract")
        observed_bundle_width = (
            column_order_contract.get("semantic_bundle_width")
            if isinstance(column_order_contract, Mapping)
            else eye_writer.EYE_ANGLE_DENSE_CHUNK_COLUMNS
        )
        semantic_bundle_width = (
            observed_bundle_width
            if type(observed_bundle_width) is int and observed_bundle_width >= 3
            else eye_writer.EYE_ANGLE_DENSE_CHUNK_COLUMNS
        )
        candidate_storage_issues = validate_eye_angle_candidate_storage(
            group,
            dimensions=EyeAngleDimensions(
                n_roi_rows=int(row_count),
                n_frames=int(frame_count),
                angle_block_width=int(semantic_bundle_width),
            ),
        )
        errors.extend(
            f"candidate storage: {issue.code}:{issue.path}:{issue.message}"
            for issue in candidate_storage_issues
        )
    return json_attr_safe(
        {
            "valid": not errors,
            "errors": errors,
            "exact_compact_v7_valid": not exact_schema_issues,
            "row_count": row_count,
            "frame_count": observed_frame_count,
            "angle_channel_count": len(channel_names.get("angle_channel_index", ())),
            "qa_channel_count": len(channel_names.get("qa_channel_index", ())),
            "array_count": array_count,
            "sharded_array_count": sharded_array_count,
            "require_sharded": bool(require_sharded),
            "source_contract_sha256": observed_contract_sha256,
            "instance_key_sha256": observed_instance_key_sha256,
            "source_acquisition_frame_index_sha256": (
                observed_acquisition_frame_index_sha256
            ),
            "algorithm_contract_sha256": (
                _json_digest(algorithm) if isinstance(algorithm, dict) else None
            ),
            "output_schema_sha256": (
                _json_digest(output_schema) if isinstance(output_schema, dict) else None
            ),
            "physical_storage_layout": physical_layout,
            "storage_profile_id": storage_profile_id,
            "candidate_storage_valid": not candidate_storage_issues,
        }
    )


def _require_candidate_direct_consolidated_equivalence(
    direct_run: zarr.Group,
    consolidated_run: zarr.Group,
    *,
    dimensions: EyeAngleDimensions,
    label: str,
) -> int:
    """Require final attrs and all 41 array declarations in both views."""

    if dict(direct_run.attrs) != dict(consolidated_run.attrs):
        raise RuntimeError(
            f"{label} eye-angle candidate direct/consolidated attributes differ."
        )
    issues = validate_eye_angle_direct_consolidated_storage(
        direct_run,
        consolidated_run,
        dimensions=dimensions,
    )
    if issues:
        raise RuntimeError(
            f"{label} eye-angle candidate direct/consolidated metadata differs: "
            + "; ".join(
                f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
            )
        )
    return 41


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {name: index for index, name in enumerate(EYE_ANGLE_EXECUTION_PHASE_ORDER)}
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Eye-angle telemetry contains an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    require_runtime_telemetry(
        result,
        expected_materializer="eye_angle_candidate",
        allowed_phase_order=EYE_ANGLE_EXECUTION_PHASE_ORDER,
    )
    return result


def tombstone_eye_angle_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one owned eye-angle benchmark candidate after runner finalization."""

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
    run_path = f"analysis/eye_angle_runs/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root["analysis/eye_angle_runs"]
        run = parent.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        if run.attrs.get(EXECUTION_BINDING_ATTR) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone an eye-angle candidate owned by another execution."
            )
        if run.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Refusing to tombstone a selector-eligible eye-angle run.")
        existing = run.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        status = run.attrs.get(RUN_COMPLETION_STATUS_ATTR)
        if status == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError("Existing eye-angle execution tombstone differs.")
        else:
            if status != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Eye-angle execution candidate is neither complete nor failed."
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
            raise RuntimeError("Eye-angle execution tombstone did not persist exactly.")
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": receipt.declarations_sha256,
    }


def publish_eye_angle_run(
    plan: EyeAngleMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
    telemetry: PhaseTelemetry | None = None,
    expected_source_logical_hashes: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish one final physical eye-angle run."""

    sealed_identity = _sealed_output_identity_digests(
        plan.staged_input_integrity_receipt
    )
    storage_candidate = is_eye_angle_storage_candidate(plan.storage_profile_id)
    if (
        expected_source_logical_hashes is not None
        or publication_acceptance_validator is not None
    ) and not storage_candidate:
        raise ValueError("Execution acceptance is valid for eye-angle candidates only.")
    publication_pointer_snapshot: dict[str, tuple[bool, Any]] = {}
    publication_acceptance: dict[str, Any] = {}

    def phase(name: str):
        return telemetry.phase(name) if telemetry is not None else nullcontext()

    def candidate_dimensions() -> EyeAngleDimensions:
        return EyeAngleDimensions(
            n_roi_rows=int(plan.row_count),
            n_frames=int(plan.frame_count),
            angle_block_width=int(plan.angle_chunk_columns),
        )

    def validate(path: Path) -> dict[str, Any]:
        return _validate_eye_angle_run(
            path,
            row_count=plan.row_count,
            frame_count=plan.frame_count,
            expected_source_contract_sha256=plan.source_contract_sha256,
            expected_instance_key_sha256=sealed_identity["instance_key"],
            expected_acquisition_frame_index_sha256=sealed_identity[
                "source_acquisition_frame_index"
            ],
            require_sharded=not storage_candidate,
            expected_angle_chunk_rows=(
                None if storage_candidate else plan.angle_chunk_rows
            ),
            expected_angle_chunk_columns=(
                None if storage_candidate else plan.angle_chunk_columns
            ),
            expected_angle_shard_rows=(
                None if storage_candidate else plan.output_shard_rows
            ),
            expected_angle_shard_columns=(
                None if storage_candidate else plan.angle_shard_columns
            ),
            storage_profile_id=plan.storage_profile_id,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        parent = require_runs_parent(
            root.require_group("analysis"), "eye_angle_runs"
        )
        if storage_candidate and not publication_pointer_snapshot:
            publication_pointer_snapshot.update(
                {
                    name: (name in parent.attrs, parent.attrs.get(name))
                    for name in ("latest", "latest_complete")
                }
            )
        return (parent,)

    def candidate_pointers_unchanged(parent: zarr.Group) -> bool:
        return bool(publication_pointer_snapshot) and all(
            (name in parent.attrs, parent.attrs.get(name)) == expected
            for name, expected in publication_pointer_snapshot.items()
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
            parent_group=None if storage_candidate else parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="eye_angle_materializer",
            ),
        )
        run_group.attrs["stage_selector_eligible"] = False
        if not storage_candidate:
            parent.attrs["latest_complete"] = plan.run_name
            parent.attrs["latest"] = plan.run_name

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/eye_angle_runs"]
        run_group = parent[plan.run_name]
        pointers_valid = (
            candidate_pointers_unchanged(parent)
            if storage_candidate
            else (
                parent.attrs.get("latest") == plan.run_name
                and parent.attrs.get("latest_complete") == plan.run_name
            )
        )
        if (
            not pointers_valid
            or run_group.attrs.get("palette_run_completion_status") != "complete"
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Eye-angle run was not persisted complete and ineligible behind "
                "the expected parent pointer state."
            )

    archive_consolidated_counts: list[int] = []

    def finalize_visibility_boundary(
        _root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if storage_candidate:
            if (
                not candidate_pointers_unchanged(parent)
                or run_group.attrs.get("palette_run_completion_status")
                != "complete"
                or run_group.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Eye-angle candidate lost its complete, selector-ineligible, "
                    "pointer-preserving state before metadata consolidation."
                )
            with phase("published_validation"):
                published_validation = validate(plan.target_run_path)
                if not published_validation["valid"]:
                    raise RuntimeError(
                        "Published eye-angle candidate is invalid: "
                        f"{published_validation}"
                    )
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            direct_root = zarr.open_group(
                str(plan.source_zarr), mode="r", use_consolidated=False
            )
            consolidated_root = zarr.open_group(
                str(plan.source_zarr), mode="r", use_consolidated=True
            )
            direct_parent = direct_root["analysis/eye_angle_runs"]
            consolidated_parent = consolidated_root["analysis/eye_angle_runs"]
            if not candidate_pointers_unchanged(
                direct_parent
            ) or not candidate_pointers_unchanged(consolidated_parent):
                raise RuntimeError(
                    "Eye-angle candidate consolidated metadata does not preserve "
                    "the publication-lock selector snapshot."
                )
            with phase("published_direct_consolidated_comparison"):
                published_compared = _require_candidate_direct_consolidated_equivalence(
                    direct_root[f"analysis/eye_angle_runs/{plan.run_name}"],
                    consolidated_root[f"analysis/eye_angle_runs/{plan.run_name}"],
                    dimensions=candidate_dimensions(),
                    label="Authoritative",
                )
            archive_consolidated_counts.append(published_compared)
            with phase("decoded_equality"):
                published_hashes = compute_eye_angle_logical_hashes(run_group)
                if (
                    expected_source_logical_hashes is not None
                    and published_hashes != dict(expected_source_logical_hashes)
                ):
                    raise RuntimeError(
                        "Published eye-angle decoded values differ from the source run."
                    )
            with phase("physical_inventory"):
                published_storage = storage_stats(plan.target_run_path)
                if (
                    published_storage["file_count"] < 1
                    or published_storage["apparent_bytes"] < 1
                    or published_storage["allocated_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published eye-angle candidate has no physical payload."
                    )
            publication_acceptance.update(
                published_validation=published_validation,
                published_direct_consolidated_array_count=published_compared,
                published_hashes=published_hashes,
                output_storage=published_storage,
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    publication_acceptance["caller_acceptance"] = json_attr_safe(
                        dict(
                            publication_acceptance_validator(
                                _root,
                                parent,
                                run_group,
                            )
                        )
                    )
            return
        if (
            str(parent.attrs.get("latest")) != plan.run_name
            or str(parent.attrs.get("latest_complete")) != plan.run_name
            or run_group.attrs.get("palette_run_completion_status") != "complete"
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Eye-angle activation requires one complete, ineligible run."
            )
        try:
            run_group.attrs["stage_selector_eligible"] = True
        except BaseException:
            if run_group.attrs.get("stage_selector_eligible") is True:
                return
            raise

    def repair_failed_candidate_visibility(target_run_path: Path) -> None:
        """Make one owned candidate tombstone identical in both metadata views."""

        if target_run_path.resolve() != plan.target_run_path.resolve():
            raise RuntimeError(
                "Eye-angle failed-publication repair received an unexpected target."
            )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        direct_root = zarr.open_group(
            str(plan.source_zarr), mode="r", use_consolidated=False
        )
        consolidated_root = zarr.open_group(
            str(plan.source_zarr), mode="r", use_consolidated=True
        )
        relative_run_path = f"analysis/eye_angle_runs/{plan.run_name}"
        direct_run = direct_root[relative_run_path]
        consolidated_run = consolidated_root[relative_run_path]
        direct_attrs = dict(direct_run.attrs)
        consolidated_attrs = dict(consolidated_run.attrs)
        if direct_attrs != consolidated_attrs:
            raise RuntimeError(
                "Eye-angle failed candidate differs between direct and "
                "consolidated metadata views."
            )
        tombstone = direct_attrs.get(ATOMIC_PUBLICATION_TOMBSTONE_ATTR)
        expected_tombstone_fields = frozenset(
            {
                "schema_id",
                "schema_version",
                "failed_at_utc",
                "publication_owner_attr",
                "publication_owner_uuid",
                "run_name",
                "run_path",
                "public_path_retained",
                "selector_eligible",
                "retry_policy",
                "failure_type",
                "failure",
            }
        )
        if (
            direct_attrs.get("palette_run_completion_status") != "failed"
            or direct_attrs.get("stage_selector_eligible") is not False
            or "palette_run_completed_at_utc" in direct_attrs
            or not isinstance(tombstone, Mapping)
            or frozenset(tombstone) != expected_tombstone_fields
            or tombstone.get("schema_id")
            != "palette.atomic_publication_tombstone"
            or tombstone.get("schema_version") != 1
            or not isinstance(tombstone.get("failed_at_utc"), str)
            or not tombstone.get("failed_at_utc")
            or not isinstance(tombstone.get("publication_owner_attr"), str)
            or direct_attrs.get(str(tombstone.get("publication_owner_attr")))
            != tombstone.get("publication_owner_uuid")
            or tombstone.get("run_name") != plan.run_name
            or Path(str(tombstone.get("run_path"))).resolve()
            != plan.target_run_path.resolve()
            or tombstone.get("public_path_retained") is not True
            or tombstone.get("selector_eligible") is not False
            or tombstone.get("retry_policy")
            != "new_immutable_run_name_required"
            or not isinstance(tombstone.get("failure_type"), str)
            or direct_attrs.get("palette_run_error") != tombstone.get("failure")
            or (
                "publication_status" in direct_attrs
                and direct_attrs.get("publication_status") != "failed"
            )
        ):
            raise RuntimeError(
                "Eye-angle failed candidate is not the exact failed/ineligible "
                "atomic-publication tombstone."
            )

    result = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.publication_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="eye-angle-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=(
                "exact_source_subset_staged_local_compute_then_shard_then_"
                "atomic_run_group_publish"
            ),
            rollback_policy=(
                "retain_failed_public_tombstone_leave_unleased_parent_state_untouched"
            ),
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize_visibility_boundary,
        repair_failed_publication_visibility=(
            repair_failed_candidate_visibility if storage_candidate else None
        ),
        after_rename=after_rename,
        payload_metadata={
            "authoritative_source_zarr": str(plan.source_zarr),
            "node_local_staged_zarr": str(plan.staged_zarr),
            "node_local_regular_run": str(plan.local_run_path),
            "node_local_sharded_run": (
                None if storage_candidate else str(plan.sharded_run)
            ),
            "node_local_publication_run": str(plan.publication_run_path),
            "storage_profile_id": plan.storage_profile_id,
            "metadata_visibility_policy": (
                {
                    "authoritative_root_consolidation": (
                        "after_final_publisher_metadata_write"
                    ),
                    "direct_consolidated_group_attrs_required": True,
                    "direct_consolidated_array_declarations_required": 41,
                    "consolidated_parent_selectors_must_match_publication_snapshot": True,
                }
                if storage_candidate
                else None
            ),
            "promotion_policy": (
                "immutable_named_candidate_no_pointer_or_registry_activation"
                if storage_candidate
                else "complete_ineligible_then_pointers_then_eligibility_final"
            ),
            "materialization": json_attr_safe(materialization_payload),
        },
    )
    if storage_candidate and archive_consolidated_counts != [41]:
        raise RuntimeError(
            "Eye-angle candidate archive metadata was not consolidated exactly once."
        )
    result["archive_direct_consolidated_array_count"] = (
        archive_consolidated_counts[0] if archive_consolidated_counts else None
    )
    if storage_candidate:
        result.update(publication_acceptance)
        result["registry_updated"] = False
        return result
    authoritative_root = open_zarr_root(plan.source_zarr, mode="r")
    result["registry_updated"] = emit_eye_angle_stage_completion(
        authoritative_root,
        plan.source_zarr,
        run_group=authoritative_root[
            f"analysis/eye_angle_runs/{plan.run_name}"
        ],
        run_name=plan.run_name,
        source="eye_angle_atomic_materializer",
    )
    return result


def materialize_eye_angles(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    run_name: str,
    storage_profile: str = EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    angle_chunk_rows: int = DEFAULT_ANGLE_CHUNK_ROWS,
    angle_chunk_columns: int = DEFAULT_ANGLE_CHUNK_COLUMNS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    angle_shard_columns: int = DEFAULT_ANGLE_SHARD_COLUMNS,
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
    execution_binding: Mapping[str, Any] | None = None,
    expected_source_logical_hashes: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Plan or execute the complete staged eye-angle materialization."""

    telemetry = PhaseTelemetry(
        materializer="eye_angle_candidate",
        context={
            "run_name": run_name,
            "subject_shape_run": subject_shape_run,
            "keypoint_run": keypoint_run,
            "storage_profile_id": storage_profile,
        },
    )
    with telemetry.phase("plan"):
        plan = build_eye_angle_materialization_plan(
            source_zarr,
            scratch_root=scratch_root,
            subject_shape_run=subject_shape_run,
            keypoint_run=keypoint_run,
            run_name=run_name,
            storage_profile=storage_profile,
            chunk_rows=chunk_rows,
            angle_chunk_rows=angle_chunk_rows,
            angle_chunk_columns=angle_chunk_columns,
            output_shard_rows=output_shard_rows,
            angle_shard_columns=angle_shard_columns,
            execution_backend=execution_backend,
            scheduler=scheduler,
            num_workers=num_workers,
            shard_workers=shard_workers,
            native_threads=native_threads,
            fps=fps,
            smoothing_window=smoothing_window,
        )
    execution_mode = any(
        value is not None
        for value in (
            execution_binding,
            expected_source_logical_hashes,
            publication_acceptance_validator,
        )
    )
    if execution_mode and not is_eye_angle_storage_candidate(plan.storage_profile_id):
        raise ValueError("Shared execution hooks require an eye-angle candidate profile.")
    binding: dict[str, Any] | None = None
    if execution_binding is not None:
        binding = json_attr_safe(dict(execution_binding))
        if not binding:
            raise ValueError("execution_binding must be one nonempty mapping.")
    if execution_mode and (
        binding is None
        or expected_source_logical_hashes is None
        or publication_acceptance_validator is None
    ):
        raise ValueError(
            "Eye-angle shared execution requires binding, source equality, and "
            "atomic acceptance together."
        )
    sealed_identity = _sealed_output_identity_digests(
        plan.staged_input_integrity_receipt
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
    native_environment = _configure_native_threads(plan.native_threads)
    try:
        with telemetry.phase("source_staging"):
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
            "--dense-chunk-rows",
            str(plan.angle_chunk_rows),
            "--dense-chunk-columns",
            str(plan.angle_chunk_columns),
            "--execution-backend",
            plan.execution_backend,
            "--scheduler",
            plan.scheduler,
            "--num-workers",
            str(plan.num_workers),
            "--layout",
            eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
            "--storage-profile",
            plan.storage_profile_id,
            "--quiet",
        ]
        if plan.fps is not None:
            writer_argv.extend(("--fps", str(plan.fps)))
        if plan.smoothing_window is not None:
            writer_argv.extend(("--smoothing-window", str(plan.smoothing_window)))
        with telemetry.phase("scientific_compute"):
            compute_started = time.perf_counter()
            eye_writer.main(
                writer_argv,
                _staged_input_integrity_receipt=(
                    plan.staged_input_integrity_receipt
                ),
            )
            compute_seconds = float(time.perf_counter() - compute_started)

        with telemetry.phase("local_validation"):
            regular_run = open_zarr_root(plan.local_run_path, mode="a")
            if binding is not None:
                regular_run.attrs[EXECUTION_BINDING_ATTR] = binding
            regular_validation = _validate_eye_angle_run(
                plan.local_run_path,
            row_count=plan.row_count,
            frame_count=plan.frame_count,
            expected_source_contract_sha256=plan.source_contract_sha256,
            expected_instance_key_sha256=sealed_identity["instance_key"],
            expected_acquisition_frame_index_sha256=sealed_identity[
                "source_acquisition_frame_index"
            ],
            require_sharded=False,
            expected_angle_chunk_rows=(
                None
                if is_eye_angle_storage_candidate(plan.storage_profile_id)
                else plan.angle_chunk_rows
            ),
            expected_angle_chunk_columns=(
                None
                if is_eye_angle_storage_candidate(plan.storage_profile_id)
                else plan.angle_chunk_columns
            ),
            expected_angle_shard_rows=(
                None
                if is_eye_angle_storage_candidate(plan.storage_profile_id)
                else plan.output_shard_rows
            ),
            expected_angle_shard_columns=(
                None
                if is_eye_angle_storage_candidate(plan.storage_profile_id)
                else plan.angle_shard_columns
            ),
            storage_profile_id=plan.storage_profile_id,
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
                    "angle_chunk_rows": plan.angle_chunk_rows,
                    "angle_chunk_columns": plan.angle_chunk_columns,
                    "execution_backend": plan.execution_backend,
                    "scheduler": plan.scheduler,
                    "num_workers": plan.num_workers,
                    "native_thread_environment": native_environment,
                    "fps": plan.fps,
                    "fps_source": plan.fps_source,
                    "smoothing_window": plan.smoothing_window,
                    "layout": eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
                    "storage_profile_id": plan.storage_profile_id,
                    "angle_column_order_profile": eye_writer.EYE_ANGLE_COLUMN_ORDER_PROFILE,
                },
                "regular_validation": {
                    **regular_validation,
                    "valid": not non_provenance_errors,
                    "errors": non_provenance_errors,
                },
                "source_contract_sha256": plan.source_contract_sha256,
                "source_metadata_sha256": plan.source_metadata_sha256,
                "staged_input_integrity_receipt_sha256": (
                    plan.staged_input_integrity_receipt.get("record_sha256")
                ),
                "staged_input_integrity_receipt": (
                    plan.staged_input_integrity_receipt
                ),
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
        regular_run.attrs["node_local_materialization"] = materialization_payload
        provenance = dict(regular_run.attrs.get("provenance", {}))
        provenance["materialization"] = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "authoritative_source_zarr": str(plan.source_zarr),
            "node_local_staged_zarr": str(plan.staged_zarr),
            "source_contract_sha256": plan.source_contract_sha256,
            "source_metadata_sha256": plan.source_metadata_sha256,
            "staged_input_integrity_receipt_sha256": (
                plan.staged_input_integrity_receipt.get("record_sha256")
            ),
            "selected_arrays": list(plan.selected_arrays),
            "compute_arguments": writer_argv,
        }
        regular_run.attrs["provenance"] = json_attr_safe(provenance)

        if is_eye_angle_storage_candidate(plan.storage_profile_id):
            candidate_validation = _validate_eye_angle_run(
                plan.local_run_path,
                row_count=plan.row_count,
                frame_count=plan.frame_count,
                expected_source_contract_sha256=plan.source_contract_sha256,
                expected_instance_key_sha256=sealed_identity["instance_key"],
                expected_acquisition_frame_index_sha256=sealed_identity[
                    "source_acquisition_frame_index"
                ],
                require_sharded=False,
                expected_angle_chunk_rows=None,
                expected_angle_chunk_columns=None,
                expected_angle_shard_rows=None,
                expected_angle_shard_columns=None,
                storage_profile_id=plan.storage_profile_id,
            )
            if not candidate_validation["valid"]:
                raise RuntimeError(
                    "Node-local final eye-angle candidate is invalid: "
                    f"{candidate_validation}"
                )
            local_hashes = compute_eye_angle_logical_hashes(regular_run)
            if (
                expected_source_logical_hashes is not None
                and local_hashes != dict(expected_source_logical_hashes)
            ):
                raise RuntimeError(
                    "Node-local eye-angle decoded values differ from the source run."
                )
            materialization_payload.update(
                {
                    "local_direct_consolidated_array_count": 41,
                    "final_physical_validation": candidate_validation,
                }
            )
            regular_run.attrs["node_local_materialization"] = materialization_payload

        if is_eye_angle_storage_candidate(plan.storage_profile_id):
            with telemetry.phase("local_consolidation"):
                consolidate_metadata_capture_expected_warnings(plan.staged_zarr)
            direct_root = zarr.open_group(
                str(plan.staged_zarr), mode="r", use_consolidated=False
            )
            consolidated_root = zarr.open_group(
                str(plan.staged_zarr), mode="r", use_consolidated=True
            )
            with telemetry.phase("local_direct_consolidated_comparison"):
                local_compared = _require_candidate_direct_consolidated_equivalence(
                    direct_root[f"analysis/eye_angle_runs/{plan.run_name}"],
                    consolidated_root[f"analysis/eye_angle_runs/{plan.run_name}"],
                    dimensions=EyeAngleDimensions(
                        n_roi_rows=plan.row_count,
                        n_frames=plan.frame_count,
                        angle_block_width=plan.angle_chunk_columns,
                    ),
                    label="Node-local",
                )
            with telemetry.phase("atomic_publication"):
                publish_kwargs: dict[str, Any] = {
                    "materialization_payload": materialization_payload,
                    "copy_backend": copy_backend,
                }
                if execution_mode:
                    publish_kwargs.update(
                        telemetry=telemetry,
                        expected_source_logical_hashes=(
                            expected_source_logical_hashes
                        ),
                        publication_acceptance_validator=(
                            publication_acceptance_validator
                        ),
                    )
                publish = publish_eye_angle_run(plan, **publish_kwargs)
            published_hashes = publish.get("published_hashes")
            result.update(
                {
                    "status": "complete",
                    "staging": staging,
                    "local_materialization": materialization_payload,
                    "publish": publish,
                    "source_logical_manifest_sha256": (
                        None
                        if expected_source_logical_hashes is None
                        else canonical_json_sha256(expected_source_logical_hashes)
                    ),
                    "local_logical_manifest_sha256": canonical_json_sha256(
                        local_hashes
                    ),
                    "published_logical_manifest_sha256": canonical_json_sha256(
                        published_hashes
                    ),
                    "local_direct_consolidated_array_count": local_compared,
                    "published_validation": publish.get("published_validation"),
                    "published_direct_consolidated_array_count": publish.get(
                        "published_direct_consolidated_array_count"
                    ),
                    "output_storage": publish.get("output_storage"),
                    "caller_acceptance": publish.get("caller_acceptance"),
                    "runtime_telemetry": _ordered_runtime_telemetry(telemetry),
                }
            )
            succeeded = True
            return result

        sharding = copy_completed_run_to_sharded(
            plan.local_run_path,
            plan.sharded_run,
            row_count_array=None,
            shard_rows=plan.output_shard_rows,
            array_layouts={
                array_name: ShardedArrayLayout(
                    inner_chunks=(plan.angle_chunk_rows, plan.angle_chunk_columns),
                    outer_shards=(plan.output_shard_rows, plan.angle_shard_columns),
                    layout_profile=eye_writer.EYE_ANGLE_COLUMN_ORDER_PROFILE,
                )
                for array_name in ("roi_angles", "frame_angles")
            },
            workers=plan.shard_workers,
        )
        sharding_summary = {
            key: value
            for key, value in sharding.items()
            if key not in {"arrays", "shards", "static_arrays"}
        }
        sharding_summary["angle_array_layouts"] = [
            item
            for item in sharding["arrays"]
            if item["path"] in {"roi_angles", "frame_angles"}
        ]
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
            expected_instance_key_sha256=sealed_identity["instance_key"],
            expected_acquisition_frame_index_sha256=sealed_identity[
                "source_acquisition_frame_index"
            ],
            require_sharded=True,
            expected_angle_chunk_rows=plan.angle_chunk_rows,
            expected_angle_chunk_columns=plan.angle_chunk_columns,
            expected_angle_shard_rows=plan.output_shard_rows,
            expected_angle_shard_columns=plan.angle_shard_columns,
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
    except BaseException as exc:
        try:
            setattr(exc, "palette_runtime_telemetry", _ordered_runtime_telemetry(telemetry))
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
        return scratch_user / job_id / f"palette_eye_angles_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_eye_angles_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--subject-shape-run")
    parser.add_argument(
        "--keypoint-run",
        help=(
            "Optional exact base keypoint run assertion. The selected subject-shape "
            "publication remains the authority; refined keypoints are unsupported."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--storage-profile",
        choices=EYE_ANGLE_STORAGE_PROFILE_CHOICES,
        default=EYE_ANGLE_LEGACY_EXPLICIT_STORAGE,
        help=(
            "Explicit physical storage profile. The access-aware profile is an "
            "immutable selector-ineligible candidate and requires serial_driver."
        ),
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--angle-chunk-rows",
        type=int,
        default=DEFAULT_ANGLE_CHUNK_ROWS,
    )
    parser.add_argument(
        "--angle-chunk-columns",
        type=int,
        default=DEFAULT_ANGLE_CHUNK_COLUMNS,
    )
    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
    )
    parser.add_argument(
        "--angle-shard-columns",
        type=int,
        default=DEFAULT_ANGLE_SHARD_COLUMNS,
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
        storage_profile=args.storage_profile,
        chunk_rows=args.chunk_size,
        angle_chunk_rows=args.angle_chunk_rows,
        angle_chunk_columns=args.angle_chunk_columns,
        output_shard_rows=args.output_shard_rows,
        angle_shard_columns=args.angle_shard_columns,
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
