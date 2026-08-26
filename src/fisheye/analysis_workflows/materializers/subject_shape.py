"""Materialize subject shape locally, shard it, and publish atomically.

The authoritative refined subject masks are opened read-only.  Computation
writes ordinary logical chunks into a node-local Zarr, a second node-local pass
assembles complete indexed outer shards with exact decoded validation, and only
the completed sharded run is copied back to shared storage.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr

from ...analysis.subject_shape_runs import (
    CENTERLINE_SAMPLE_COUNT,
    COMPONENT_ORDER,
    DASK_WORKER_EXECUTION_BACKEND,
    SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
    SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR,
    SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS,
    SUBJECT_SHAPE_SCHEMA_ID,
    SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
    TAIL_SAMPLE_COUNT,
    bind_staged_subject_shape_run,
    complete_bound_subject_shape_candidate_run,
    complete_bound_subject_shape_run_for_deferred_activation,
    refresh_unbound_subject_shape_manifest_after_storage_materialization,
    write_subject_shape_run_group,
)
from ...analysis.subject_shape_storage import (
    SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES,
    finalize_bound_subject_shape_storage_receipt,
    is_subject_shape_access_aware_storage,
    is_subject_shape_storage_candidate,
    materialize_subject_shape_access_aware_storage,
    set_subject_shape_metadata_visibility_policy,
    subject_shape_access_aware_storage_profile,
    validate_subject_shape_access_aware_storage,
    validate_subject_shape_direct_consolidated_storage,
    validate_subject_shape_storage_source_manifest_link,
)
from ...shared.json_safety import json_attr_safe
from ...shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    commit_deferred_subject_shape_coordinate_activation,
    load_completed_ineligible_subject_shape_coordinate_publication,
    load_persisted_subject_shape_coordinate_publication,
    rollback_deferred_subject_shape_coordinate_activation,
)
from ...shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from ...shared.zarr_helpers import archive_metadata_publication_lock
from ...shared.refined_subject_masks_io import (
    load_refined_subject_masks_run_tables,
    resolve_refined_subject_masks_run,
)
from ...shared.refined_subject_mask_coordinate_publication import (
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from ...shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import require_runs_parent
from ...shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_failed,
)
from ...shared.zarr.benchmark_runtime import storage_stats
from ...shared.zarr.manifest_digest import canonical_json_sha256
from ...shared.zarr.subject_shape_bundle_source import (
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    load_subject_shape_bundle_source,
)
from ...shared.zarr_sharded_copy import copy_completed_run_to_sharded
from fisheye.shared.atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from fisheye.shared.atomic_run_publisher import ATOMIC_PUBLICATION_TOMBSTONE_ATTR
from fisheye.shared.runtime_telemetry import PhaseTelemetry

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

SUBJECT_SHAPE_EXECUTION_PHASE_ORDER = (
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
SUBJECT_SHAPE_EXECUTION_BINDING_ATTR = "analysis_candidate_execution_binding"
SUBJECT_SHAPE_EXECUTION_FAILURE_TOMBSTONE_ATTR = (
    "analysis_candidate_execution_tombstone"
)
SubjectShapePublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


@dataclass(frozen=True)
class SubjectShapeMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    compute_zarr: Path
    sharded_run: Path
    refined_run: str
    run_name: str
    storage_profile_id: str
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
    subject_mask_bundle_id: str | None = None
    allow_inactive_subject_mask_bundle: bool = False
    assignment_keypoint_rebinding_run_id: str | None = None

    @property
    def compute_run_path(self) -> Path:
        return self.compute_zarr / "analysis" / "subject_shape_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "subject_shape_runs" / self.run_name

    @property
    def publication_run_path(self) -> Path:
        return self.sharded_run

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
                "storage_profile_id": self.storage_profile_id,
                "publication_run_path": str(self.publication_run_path),
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
                "subject_mask_bundle_id": self.subject_mask_bundle_id,
                "allow_inactive_subject_mask_bundle": (
                    self.allow_inactive_subject_mask_bundle
                ),
                "assignment_keypoint_rebinding_run_id": (
                    self.assignment_keypoint_rebinding_run_id
                ),
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    storage_profile: str = SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    components: Sequence[str] | None = None,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    execution_backend: str = DASK_WORKER_EXECUTION_BACKEND,
    scheduler: str = "processes",
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_copy_workers: int = DEFAULT_SHARD_COPY_WORKERS,
    native_threads: int = DEFAULT_NATIVE_THREADS,
    subject_mask_bundle_id: str | None = None,
    allow_inactive_subject_mask_bundle: bool = False,
    assignment_keypoint_rebinding_run_id: str | None = None,
) -> SubjectShapeMaterializationPlan:
    """Resolve a read-only plan without creating scratch or mutating the archive."""

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
            "resolving symlinks; equality and either containment direction are unsafe."
        )
    if int(block_rows) <= 0 or int(output_shard_rows) <= 0:
        raise ValueError("Block and output-shard row counts must be positive.")
    if (
        int(num_workers) <= 0
        or int(shard_copy_workers) <= 0
        or int(native_threads) <= 0
    ):
        raise ValueError("Worker and native-thread counts must be positive.")
    backend = str(execution_backend).strip().lower()
    if backend not in {"serial_driver", DASK_WORKER_EXECUTION_BACKEND}:
        raise ValueError(f"Unsupported execution backend: {execution_backend!r}.")
    scheduler_key = str(scheduler).strip().lower().replace("_", "-")
    if scheduler_key not in {"single-threaded", "threads", "processes", "distributed"}:
        raise ValueError(f"Unsupported scheduler: {scheduler!r}.")
    storage_profile_id = str(storage_profile)
    if storage_profile_id not in SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES:
        raise ValueError(
            f"Unsupported subject-shape storage profile {storage_profile_id!r}; "
            f"expected one of {SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES!r}."
        )

    if type(allow_inactive_subject_mask_bundle) is not bool:
        raise TypeError("allow_inactive_subject_mask_bundle must be an exact bool.")
    root = open_zarr_root(source, mode="r")
    bundle_source = None
    if subject_mask_bundle_id is not None:
        if not is_subject_shape_access_aware_storage(storage_profile_id):
            raise ValueError(
                "Recording-bundle subject shape requires one explicit access-aware "
                "storage profile."
            )
        bundle_source = load_subject_shape_bundle_source(
            source,
            bundle_id=str(subject_mask_bundle_id),
            allow_inactive=allow_inactive_subject_mask_bundle,
            assignment_keypoint_rebinding_run_id=(
                assignment_keypoint_rebinding_run_id
            ),
        )
        refined_path = bundle_source.authority.refined_run_path
        prefix = "refined_subject_masks_runs/"
        if not refined_path.startswith(prefix) or "/" in refined_path[len(prefix) :]:
            raise ValueError("Subject-mask bundle refined member path is invalid.")
        resolved_refined_run = refined_path[len(prefix) :]
        if refined_run is not None and str(refined_run) != resolved_refined_run:
            raise ValueError(
                "Explicit refined_run differs from the selected subject-mask bundle member."
            )
        refined_group = root[refined_path]
        source_contract = {
            "source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
            "bundle_id": bundle_source.bundle_id,
            "bundle_active": bundle_source.active,
            "source_binding_sha256": bundle_source.source_digest,
            "bundle_coordinate_authority_digest": (
                bundle_source.authority.authority_digest
            ),
            "bundle_manifest_payload_digest": (
                bundle_source.authority.bundle_manifest["payload_digest"]
            ),
            "refined_manifest_payload_digest": (
                bundle_source.authority.refined_manifest["payload_digest"]
            ),
            "assignment_keypoint_rebinding_run_id": (
                bundle_source.assignment_keypoint_rebinding_run_id
            ),
            "assignment_keypoint_rebinding_manifest_payload_digest": (
                bundle_source.assignment_keypoint_rebinding_manifest[
                    "payload_digest"
                ]
                if bundle_source.assignment_keypoint_rebinding_manifest is not None
                else None
            ),
        }
    else:
        if allow_inactive_subject_mask_bundle:
            raise ValueError(
                "allow_inactive_subject_mask_bundle requires subject_mask_bundle_id."
            )
        if assignment_keypoint_rebinding_run_id is not None:
            raise ValueError(
                "assignment_keypoint_rebinding_run_id requires "
                "subject_mask_bundle_id."
            )
        refined_group, resolved_refined_run, _path = resolve_refined_subject_masks_run(
            root, refined_run
        )
        refined_coordinates = load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            f"refined_subject_masks_runs/{resolved_refined_run}",
        )
        if refined_coordinates.context._run_group.path != refined_group.path:
            raise ValueError(
                "Logical refined-mask selection differs from its canonical authority."
            )
        source_contract = {
            "schema_id": refined_group.attrs.get("schema_id"),
            "schema_version": refined_group.attrs.get("schema_version"),
            "method": refined_group.attrs.get("method"),
            "method_version": refined_group.attrs.get("method_version"),
            "palette_run_completion_status": refined_group.attrs.get(
                "palette_run_completion_status"
            ),
            "coordinate_context_sha256": (
                refined_coordinates.context.context_record.record_sha256
            ),
            "surface_inventory_sha256": refined_coordinates.inventory.record_sha256,
            "component_qc_inventory_sha256": (
                refined_coordinates.component_qc_inventory.record_sha256
            ),
        }
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
    available = tuple(
        str(value) for value in refined_group.attrs.get("mask_labels") or ()
    )
    selected = (
        tuple(str(value) for value in components)
        if components
        else tuple(name for name in COMPONENT_ORDER if name in available)
    )
    if not selected:
        raise ValueError(
            "No known subject-shape components are available in the refined run."
        )
    if selected != COMPONENT_ORDER:
        raise ValueError(
            "Canonical subject-shape materialization requires the exact component "
            f"order {COMPONENT_ORDER!r}; got {selected!r}. Historical component "
            "variants are inspection/migration inputs, not maintained publications."
        )
    missing_required = sorted(set(COMPONENT_ORDER) - set(selected))
    if missing_required:
        raise ValueError(
            "Canonical subject-shape materialization requires the full component "
            f"anchor set; missing {missing_required!r}."
        )
    target_name = _validate_run_name(run_name)
    target = source / "analysis" / "subject_shape_runs" / target_name
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target}"
        )
    row_count = int(mask_store.n_rows)
    estimated = (
        2 * row_count * ESTIMATED_BYTES_PER_ROW_PER_COPY + DEFAULT_CAPACITY_MARGIN_BYTES
    )
    return SubjectShapeMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        compute_zarr=scratch / "compute.zarr",
        sharded_run=scratch / "subject-shape-sharded-run",
        refined_run=resolved_refined_run,
        run_name=target_name,
        storage_profile_id=storage_profile_id,
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
                **source_contract,
                "mask_labels": list(available),
                "mask_store_encoding": mask_store.encoding,
                "mask_storage_surface": mask_store.storage_surface,
            }
        ),
        subject_mask_bundle_id=(
            bundle_source.bundle_id if bundle_source is not None else None
        ),
        allow_inactive_subject_mask_bundle=(
            allow_inactive_subject_mask_bundle if bundle_source is not None else False
        ),
        assignment_keypoint_rebinding_run_id=(
            bundle_source.assignment_keypoint_rebinding_run_id
            if bundle_source is not None
            else None
        ),
    )


def _validate_subject_shape_run(
    path: Path,
    *,
    row_count: int,
    require_sharded: bool,
    expected_binding_status: str,
    require_complete: bool,
    expected_selector_eligible: bool,
    storage_profile_id: str = SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
) -> dict[str, Any]:
    group = open_zarr_root(path, mode="r")
    errors: list[str] = []
    if str(group.attrs.get("schema_id")) != SUBJECT_SHAPE_SCHEMA_ID:
        errors.append("schema_id mismatch")
    expected_completion = "complete" if require_complete else "running"
    if str(group.attrs.get("palette_run_completion_status")) != expected_completion:
        errors.append("run completion status mismatch")
    if group.attrs.get("stage_selector_eligible") is not expected_selector_eligible:
        errors.append("stage selector eligibility mismatch")
    if (
        group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != expected_binding_status
    ):
        errors.append("coordinate binding status mismatch")
    if not bool(group.attrs.get("centerline_crop_to_foreground")):
        errors.append("foreground-cropped centerline acceleration not recorded")
    expected = {
        "row_index/instance_key": (int(row_count),),
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
    if expected_binding_status == SUBJECT_SHAPE_BOUND_CANONICAL_STATUS:
        if group.attrs.get("coordinate_contract") != "canonical_v2":
            errors.append("bound run lacks canonical_v2 coordinate contract")
    else:
        for name in (
            "coordinate_contract",
            "subject_shape_coordinate_derivation",
            "subject_shape_publication_manifest",
            "source_row_temporal_authority",
            "row_identity_contract",
        ):
            if name in group.attrs:
                errors.append(f"unbound run contains canonical attr {name}")
        if "coordinate_records" in group:
            errors.append("unbound run contains coordinate_records")
    storage_access_aware = is_subject_shape_access_aware_storage(storage_profile_id)
    storage_candidate = is_subject_shape_storage_candidate(storage_profile_id)
    if storage_access_aware:
        phase = (
            "bound"
            if expected_binding_status == SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
            else "unbound"
        )
        errors.extend(
            validate_subject_shape_access_aware_storage(
                group,
                phase=phase,
                expected_profile_id=storage_profile_id,
            )
        )
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(row_count),
        "require_sharded": bool(require_sharded),
        "binding_status": expected_binding_status,
        "physical_storage_layout": layout,
        "storage_profile_id": storage_profile_id,
        "storage_access_aware": storage_access_aware,
        "storage_candidate": storage_candidate,
    }


def publish_subject_shape_run(
    plan: SubjectShapeMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    storage_profile_id = getattr(
        plan,
        "storage_profile_id",
        SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    )
    publication_run_path = getattr(
        plan,
        "publication_run_path",
        plan.sharded_run,
    )
    storage_access_aware = is_subject_shape_access_aware_storage(storage_profile_id)
    storage_candidate = is_subject_shape_storage_candidate(storage_profile_id)
    transaction = {
        "binding_complete": False,
        "completion_published": False,
        "publication_owner_uuid": None,
    }
    deferred_activation: list[Any] = []
    publication_pointer_snapshot: dict[str, tuple[bool, Any]] = {}

    def validate(path: Path) -> dict[str, Any]:
        structural = _validate_subject_shape_run(
            path,
            row_count=plan.row_count,
            require_sharded=not storage_access_aware,
            expected_binding_status=(
                SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
                if transaction["binding_complete"]
                else SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS
            ),
            require_complete=bool(transaction["completion_published"]),
            expected_selector_eligible=False,
            storage_profile_id=storage_profile_id,
        )
        if not structural["valid"] or not transaction["binding_complete"]:
            return structural
        if path.resolve() != plan.target_run_path.resolve():
            structural["valid"] = False
            structural["errors"].append(
                "canonical validation is permitted only at the exact authoritative path"
            )
            return structural
        if transaction["completion_published"]:
            try:
                root = open_zarr_root(plan.source_zarr, mode="r")
                proof = load_completed_ineligible_subject_shape_coordinate_publication(
                    root,
                    f"analysis/subject_shape_runs/{plan.run_name}",
                    expected_publication_owner=str(
                        transaction["publication_owner_uuid"]
                    ),
                )
                structural["canonical_validation"] = {
                    "valid": True,
                    "run_name": plan.run_name,
                    "row_count": int(proof.row_identity.leading_dimension),
                    "manifest_sha256": proof.manifest.record_sha256,
                }
            except Exception as exc:
                structural["valid"] = False
                structural["errors"].append(
                    f"strict canonical validation failed: {exc}"
                )
        return structural

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        parent = require_runs_parent(
            root.require_group("analysis"),
            "subject_shape_runs",
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

    def after_rename(
        root: zarr.Group,
        run_group: zarr.Group,
    ) -> dict[str, Any]:
        if transaction["binding_complete"] or transaction["completion_published"]:
            raise RuntimeError("Subject-shape publication state is inconsistent.")
        owner = run_group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
        if not isinstance(owner, str):
            raise RuntimeError("Subject-shape publication owner is missing.")
        transaction["publication_owner_uuid"] = owner
        write_best_effort_run_lineage_attrs(run_group, run_family="subject_shape_run")
        if storage_access_aware:
            source_link_issues = validate_subject_shape_storage_source_manifest_link(
                run_group,
                phase="unbound",
                verify_content=True,
                block_rows=int(getattr(plan, "block_rows", 1_024)),
            )
            if source_link_issues:
                raise RuntimeError(
                    "Subject-shape access-aware run differs from its original producer "
                    "seal before binding: " + "; ".join(source_link_issues)
                )
            refresh_unbound_subject_shape_manifest_after_storage_materialization(
                run_group
            )
        binding = bind_staged_subject_shape_run(
            root,
            run_group,
            expected_refined_run=plan.refined_run,
            expected_run_name=plan.run_name,
            expected_subject_mask_bundle_id=getattr(
                plan,
                "subject_mask_bundle_id",
                None,
            ),
        )
        if binding.get("valid") is not True:
            raise RuntimeError(f"Final-path subject-shape binding failed: {binding!r}")
        if storage_access_aware:
            receipt = finalize_bound_subject_shape_storage_receipt(run_group)
            array_count = int(
                receipt["payload"]["object_estimate"]["array_metadata_objects"]
            )
            set_subject_shape_metadata_visibility_policy(
                run_group,
                expected_array_count=array_count,
            )
        else:
            receipt = None
        transaction["binding_complete"] = True
        return {
            "canonical_binding": binding,
            "storage_plan": receipt,
        }

    def complete(
        root: zarr.Group,
        _parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if not transaction["binding_complete"] or transaction["completion_published"]:
            raise RuntimeError(
                "Subject-shape completion requires exactly one successful binding."
            )
        owner = transaction["publication_owner_uuid"]
        if not isinstance(owner, str):
            raise RuntimeError("Subject-shape completion lacks its exact owner.")
        if storage_candidate:
            completion = complete_bound_subject_shape_candidate_run(
                root,
                run_group,
                expected_run_name=plan.run_name,
                publication_owner=owner,
            )
            activation = None
        else:
            completion, activation = (
                complete_bound_subject_shape_run_for_deferred_activation(
                    root,
                    run_group,
                    expected_run_name=plan.run_name,
                    publication_owner=owner,
                )
            )
        if completion.get("valid") is not True:
            raise RuntimeError(
                f"Deferred subject-shape completion failed: {completion!r}."
            )
        if activation is not None:
            deferred_activation[:] = [activation]
        transaction["completion_published"] = True

    def activate(
        root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if storage_candidate:
            if (
                not candidate_pointers_unchanged(parent)
                or run_group.attrs.get("palette_run_completion_status") != "complete"
                or run_group.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Subject-shape candidate lost its complete, ineligible, "
                    "pointer-preserving state before consolidation."
                )
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            direct_root = zarr.open_group(
                str(plan.source_zarr),
                mode="r",
                use_consolidated=False,
            )
            consolidated_root = zarr.open_group(
                str(plan.source_zarr),
                mode="r",
                use_consolidated=True,
            )
            direct_parent = direct_root["analysis/subject_shape_runs"]
            consolidated_parent = consolidated_root["analysis/subject_shape_runs"]
            if not candidate_pointers_unchanged(
                direct_parent
            ) or not candidate_pointers_unchanged(consolidated_parent):
                raise RuntimeError(
                    "Subject-shape consolidated metadata changed the frozen "
                    "candidate pointer snapshot."
                )
            issues = validate_subject_shape_direct_consolidated_storage(
                plan.source_zarr,
                run_path=f"analysis/subject_shape_runs/{plan.run_name}",
                phase="bound",
                expected_profile_id=storage_profile_id,
            )
            if issues:
                raise RuntimeError(
                    "Subject-shape direct/consolidated candidate declarations differ: "
                    + "; ".join(issues)
                )
            return
        if len(deferred_activation) != 1:
            raise RuntimeError(
                "Subject-shape publication lacks one deferred activation receipt."
            )
        expected_run_attrs = json_attr_safe(dict(run_group.attrs))
        staging = expected_run_attrs.get("cluster_output_staging")
        final_validation = (
            staging.get("final_validation") if isinstance(staging, Mapping) else None
        )
        if (
            not isinstance(staging, Mapping)
            or staging.get("schema_id") != PUBLISH_SCHEMA_ID
            or not isinstance(final_validation, Mapping)
            or final_validation.get("valid") is not True
        ):
            raise RuntimeError(
                "Subject-shape activation lacks its exact successful final "
                "validation payload."
            )
        commit_deferred_subject_shape_coordinate_activation(
            deferred_activation[0],
            root=root,
            parent=parent,
            run=run_group,
            expected_run_attrs=expected_run_attrs,
        )
        if storage_access_aware:
            # Selector-visible immutable publications expose their committed
            # lifecycle only after the root consolidated generation is
            # refreshed and independently reopened.
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            issues = validate_subject_shape_direct_consolidated_storage(
                plan.source_zarr,
                run_path=f"analysis/subject_shape_runs/{plan.run_name}",
                phase="bound",
                expected_profile_id=storage_profile_id,
            )
            if issues:
                raise RuntimeError(
                    "Supported subject-shape publication differs between direct "
                    "and consolidated metadata: " + "; ".join(issues)
                )
            consolidated_root = zarr.open_group(
                str(plan.source_zarr),
                mode="r",
                use_consolidated=True,
            )
            proof = load_persisted_subject_shape_coordinate_publication(
                consolidated_root,
                f"analysis/subject_shape_runs/{plan.run_name}",
                expected_publication_owner=str(
                    transaction["publication_owner_uuid"]
                ),
            )
            if proof.manifest.record_sha256 != deferred_activation[0].manifest_sha256:
                raise RuntimeError(
                    "Consolidated supported subject-shape authority changed during "
                    "selector activation."
                )

    def rollback_activation() -> None:
        if not storage_candidate and deferred_activation:
            rollback_deferred_subject_shape_coordinate_activation(
                deferred_activation[0]
            )

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/subject_shape_runs"]
        run = parent[plan.run_name]
        pointers_valid = (
            candidate_pointers_unchanged(parent)
            if storage_candidate
            else (
                str(parent.attrs.get("latest")) == plan.run_name
                and str(parent.attrs.get("latest_complete")) == plan.run_name
            )
        )
        if (
            not pointers_valid
            or run.attrs.get("stage_selector_eligible") is not False
            or run.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
            != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
        ):
            raise RuntimeError(
                "Subject-shape complete run does not match its expected pointer "
                "and selector-ineligible state."
            )

    def repair_failed_candidate_visibility(target_run_path: Path) -> None:
        if target_run_path.resolve() != plan.target_run_path.resolve():
            raise RuntimeError(
                "Subject-shape failed-publication repair received another target."
            )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        direct_root = zarr.open_group(
            str(plan.source_zarr), mode="r", use_consolidated=False
        )
        consolidated_root = zarr.open_group(
            str(plan.source_zarr), mode="r", use_consolidated=True
        )
        relative = f"analysis/subject_shape_runs/{plan.run_name}"
        direct = direct_root[relative]
        consolidated = consolidated_root[relative]
        direct_attrs = dict(direct.attrs)
        if direct_attrs != dict(consolidated.attrs):
            raise RuntimeError(
                "Subject-shape failed candidate differs between metadata views."
            )
        tombstone = direct_attrs.get(ATOMIC_PUBLICATION_TOMBSTONE_ATTR)
        if (
            direct_attrs.get("palette_run_completion_status") != "failed"
            or direct_attrs.get("stage_selector_eligible") is not False
            or "palette_run_completed_at_utc" in direct_attrs
            or not isinstance(tombstone, Mapping)
            or tombstone.get("schema_id") != "palette.atomic_publication_tombstone"
            or tombstone.get("schema_version") != 1
            or tombstone.get("run_name") != plan.run_name
            or Path(str(tombstone.get("run_path"))).resolve()
            != plan.target_run_path.resolve()
            or tombstone.get("selector_eligible") is not False
            or tombstone.get("retry_policy") != "new_immutable_run_name_required"
        ):
            raise RuntimeError(
                "Subject-shape failed candidate is not an exact failed tombstone."
            )

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=publication_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="subject-shape-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=(
                "read_only_compute_unbound_stage_byte_plan_final_path_bind_then_publish"
                if storage_access_aware
                else "read_only_compute_unbound_stage_shard_final_path_bind_then_publish"
            ),
            rollback_policy=(
                "retain_owner_bound_failed_public_tombstone_and_"
                "stage_specific_receipt_rollback_only"
            ),
            content_checksum=True,
            publication_owner_attr=SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
            # The sealed stage receipt is the only selector rollback authority.
            # A generic pre-copy snapshot can predate an intervening publication.
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        rollback_activation=(None if storage_candidate else rollback_activation),
        repair_failed_publication_visibility=(
            repair_failed_candidate_visibility if storage_candidate else None
        ),
        after_rename=after_rename,
        payload_metadata={
            "local_publication_run": str(publication_run_path),
            "storage_profile_id": storage_profile_id,
            "promotion_policy": (
                "immutable_named_candidate_no_pointer_or_registry_activation"
                if storage_candidate
                else "complete_ineligible_then_pointers_then_eligibility_final"
            ),
            "materialization": json_attr_safe(materialization_payload),
        },
        accept_persisted_activation_on_callback_error=not storage_access_aware,
    )


def materialize_subject_shape(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    refined_run: str | None,
    run_name: str,
    storage_profile: str = SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
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
    subject_mask_bundle_id: str | None = None,
    allow_inactive_subject_mask_bundle: bool = False,
    assignment_keypoint_rebinding_run_id: str | None = None,
) -> dict[str, Any]:
    plan = build_subject_shape_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        refined_run=refined_run,
        run_name=run_name,
        storage_profile=storage_profile,
        components=components,
        block_rows=block_rows,
        output_shard_rows=output_shard_rows,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        shard_copy_workers=shard_copy_workers,
        native_threads=native_threads,
        subject_mask_bundle_id=subject_mask_bundle_id,
        allow_inactive_subject_mask_bundle=allow_inactive_subject_mask_bundle,
        assignment_keypoint_rebinding_run_id=(
            assignment_keypoint_rebinding_run_id
        ),
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
            stage_command=stage_command
            or (" ".join(sys.argv) if sys.argv else "unknown"),
            subject_mask_bundle_id=getattr(plan, "subject_mask_bundle_id", None),
            allow_inactive_subject_mask_bundle=(
                getattr(plan, "allow_inactive_subject_mask_bundle", False)
            ),
            assignment_keypoint_rebinding_run_id=getattr(
                plan,
                "assignment_keypoint_rebinding_run_id",
                None,
            ),
            _unbound_coordinate_stage=True,
        )
        compute_validation = _validate_subject_shape_run(
            plan.compute_run_path,
            row_count=plan.row_count,
            require_sharded=False,
            expected_binding_status=SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
            require_complete=True,
            expected_selector_eligible=False,
        )
        if not compute_validation["valid"]:
            raise RuntimeError(
                f"Node-local compute validation failed: {compute_validation}"
            )
        storage_access_aware = is_subject_shape_access_aware_storage(
            plan.storage_profile_id
        )
        if storage_access_aware:
            compute_run = compute_root[f"analysis/subject_shape_runs/{plan.run_name}"]
            sharding_summary = materialize_subject_shape_access_aware_storage(
                compute_run,
                plan.sharded_run,
                profile=subject_shape_access_aware_storage_profile(
                    plan.storage_profile_id
                ),
                phase="unbound",
                copy_block_rows=plan.block_rows,
            )
            sharding_summary["exact_decoded_validation"] = True
        else:
            sharding = copy_completed_run_to_sharded(
                plan.compute_run_path,
                plan.sharded_run,
                row_count_array="row_index/instance_key",
                shard_rows=plan.output_shard_rows,
                workers=plan.shard_copy_workers,
            )
            sharding_summary = {
                key: value
                for key, value in sharding.items()
                if key not in {"arrays", "shards", "static_arrays"}
            }
        sharded = open_zarr_root(plan.sharded_run, mode="a")
        if (
            sharded.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
            != SUBJECT_SHAPE_UNBOUND_STAGE_STATUS
            or sharded.attrs.get("palette_run_completion_status") != "complete"
        ):
            raise RuntimeError(
                "Sharded subject-shape copy did not preserve the exact unbound stage."
            )
        sharded.attrs[SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR] = (
            SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS
        )
        sharded.attrs["palette_run_completion_status"] = "running"
        if "palette_run_completed_at_utc" in sharded.attrs:
            del sharded.attrs["palette_run_completed_at_utc"]
        publishing_validation = _validate_subject_shape_run(
            plan.sharded_run,
            row_count=plan.row_count,
            require_sharded=not storage_access_aware,
            expected_binding_status=SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS,
            require_complete=False,
            expected_selector_eligible=False,
            storage_profile_id=plan.storage_profile_id,
        )
        if not publishing_validation["valid"]:
            raise RuntimeError(
                "Sharded subject-shape publishing stage is invalid: "
                f"{publishing_validation!r}"
            )
        materialization_payload = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "source_access_policy": "authoritative_shared_read_only",
            "node_local_compute": compute_summary,
            "node_local_compute_validation": compute_validation,
            "node_local_sharding": sharding_summary,
            "publishing_validation": publishing_validation,
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


def _ordered_execution_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index for index, name in enumerate(SUBJECT_SHAPE_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Subject-shape execution telemetry has an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


def _copy_archive_snapshot(source: Path, target: Path, *, backend: str) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing staged archive: {target}.")
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Subject-shape execution source archive must be symlink-free.")
    if backend == "python":
        shutil.copytree(source, target, symlinks=False)
    elif backend == "rsync":
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--no-links", f"{source}/", f"{target}/"],
            check=True,
        )
    else:
        raise ValueError("copy_backend must be python or rsync.")
    if target.is_symlink() or any(path.is_symlink() for path in target.rglob("*")):
        raise RuntimeError("Staged subject-shape archive contains a symlink.")


def _selector_snapshot(parent: zarr.Group) -> dict[str, tuple[bool, Any]]:
    return {
        name: (name in parent.attrs, json_attr_safe(parent.attrs.get(name)))
        for name in ("latest", "latest_complete")
    }


def _subject_shape_execution_binding(
    run: zarr.Group,
) -> Mapping[str, Any] | None:
    receipt = run.attrs.get("cluster_output_staging")
    if not isinstance(receipt, Mapping):
        return None
    binding = receipt.get(SUBJECT_SHAPE_EXECUTION_BINDING_ATTR)
    return binding if isinstance(binding, Mapping) else None


def tombstone_subject_shape_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one execution-owned, selector-ineligible subject-shape candidate."""

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
    run_path = f"analysis/subject_shape_runs/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root["analysis/subject_shape_runs"]
        pointers_before = _selector_snapshot(parent)
        run = parent.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        if _subject_shape_execution_binding(run) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone a subject-shape candidate owned by another "
                "execution."
            )
        if run.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(
                "Refusing to tombstone a selector-eligible subject-shape run."
            )
        candidate = run.attrs.get("subject_shape_storage_candidate")
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("promotion_status") != "unpromoted_candidate"
        ):
            raise RuntimeError("Refusing to tombstone a promoted subject-shape run.")
        existing = run.attrs.get(SUBJECT_SHAPE_EXECUTION_FAILURE_TOMBSTONE_ATTR)
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError(
                    "Existing subject-shape execution tombstone differs."
                )
        else:
            if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Subject-shape execution target is neither complete nor failed."
                )
            mark_run_failed(
                run,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            run.attrs["stage_selector_eligible"] = False
            run.attrs[SUBJECT_SHAPE_EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        if _selector_snapshot(parent) != pointers_before:
            raise RuntimeError("Subject-shape execution tombstone changed selectors.")
        consolidate_metadata_capture_expected_warnings(archive)
        direct = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )[run_path]
        consolidated = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=True
        )[run_path]
        if (
            dict(direct.attrs) != dict(consolidated.attrs)
            or direct.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
            or direct.attrs.get("stage_selector_eligible") is not False
            or direct.attrs.get(SUBJECT_SHAPE_EXECUTION_FAILURE_TOMBSTONE_ATTR)
            != tombstone
        ):
            raise RuntimeError(
                "Subject-shape execution tombstone did not persist exactly."
            )
        from ...shared.zarr.metadata_equivalence import (
            validate_direct_consolidated_subtree,
        )

        metadata = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": metadata.declarations_sha256,
    }


def materialize_subject_shape_execution_candidate(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    block_rows: int,
    output_shard_rows: int,
    execution_backend: str,
    scheduler: str,
    num_workers: int,
    shard_copy_workers: int,
    native_threads: int,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
    execution_binding: Mapping[str, Any],
    expected_source_logical_hashes: Mapping[str, str],
    publication_acceptance_validator: (
        SubjectShapePublicationAcceptanceValidator | None
    ) = None,
) -> dict[str, Any]:
    """Recompute one v4 candidate from a node-local immutable archive snapshot.

    This execution-only wrapper intentionally leaves the maintained producer
    untouched.  Its first publication occurs inside the node-local snapshot;
    a second atomic copy publishes that already validated, bound, immutable
    candidate to the benchmark archive without touching selectors.
    """

    from ...analysis.subject_shape_storage import (
        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    )
    from ...shared.zarr.metadata_equivalence import (
        validate_direct_consolidated_subtree,
    )
    from ..subject_shape_candidate_execution import (
        compute_subject_shape_logical_hashes,
    )

    archive = Path(source_zarr).expanduser().resolve()
    source_path = str(source_run).strip().strip("/")
    if (
        not source_path.startswith("analysis/subject_shape_runs/")
        or source_path.count("/") != 2
    ):
        raise ValueError("source_run must be one explicit subject-shape run path.")
    source_name = _validate_run_name(source_path.rsplit("/", 1)[1])
    candidate_name = _validate_run_name(run_name)
    if source_name == candidate_name:
        raise ValueError("Source and candidate subject-shape names must differ.")
    binding = json_attr_safe(dict(execution_binding))
    if not binding:
        raise ValueError("execution_binding must be one nonempty mapping.")
    expected_hashes = dict(expected_source_logical_hashes)
    telemetry = PhaseTelemetry(
        materializer="subject_shape_execution_candidate",
        context={
            "source_run": source_path,
            "run_name": candidate_name,
            "source_staging_mode": "archive_snapshot_copy_v1",
        },
    )
    with telemetry.phase("plan"):
        direct_root = open_zarr_root(archive, mode="r")
        source_group = direct_root[source_path]
        source_publication = load_persisted_subject_shape_coordinate_publication(
            direct_root,
            source_path,
        )
        if compute_subject_shape_logical_hashes(source_group) != expected_hashes:
            raise ValueError(
                "Subject-shape source logical hashes differ from the execution request."
            )
        refined_run = source_group.attrs.get("source_refined_subject_masks_run")
        if type(refined_run) is not str:
            raise ValueError("Subject-shape source refined authority is absent.")
        plan = build_subject_shape_materialization_plan(
            archive,
            scratch_root=scratch_root,
            refined_run=refined_run,
            run_name=candidate_name,
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            components=COMPONENT_ORDER,
            block_rows=block_rows,
            output_shard_rows=output_shard_rows,
            execution_backend=execution_backend,
            scheduler=scheduler,
            num_workers=num_workers,
            shard_copy_workers=shard_copy_workers,
            native_threads=native_threads,
        )
        root_parent = direct_root["analysis/subject_shape_runs"]
        selectors_before = _selector_snapshot(root_parent)
    result: dict[str, Any] = {
        "schema_id": "palette.subject_shape_execution_materialization.v1",
        "status": "running",
        "mutates_archive": True,
        "plan": plan.to_json(),
    }
    succeeded = False
    try:
        if plan.scratch_root.exists() or plan.scratch_root.is_symlink():
            raise FileExistsError(
                f"Refusing existing scratch root: {plan.scratch_root}."
            )
        plan.scratch_root.mkdir(parents=True)
        if check_capacity:
            free = int(shutil.disk_usage(plan.scratch_root).free)
            if free < plan.estimated_scratch_bytes:
                raise OSError(
                    "Insufficient scratch capacity for subject-shape execution: "
                    f"need {plan.estimated_scratch_bytes}, found {free}."
                )
        staged_archive = plan.scratch_root / "staged-source.zarr"
        with telemetry.phase("source_staging"):
            _copy_archive_snapshot(archive, staged_archive, backend=copy_backend)
            staged_root = open_zarr_root(staged_archive, mode="r")
            staged_source = staged_root[source_path]
            staged_publication = load_persisted_subject_shape_coordinate_publication(
                staged_root,
                source_path,
            )
            if (
                compute_subject_shape_logical_hashes(staged_source) != expected_hashes
                or staged_publication.manifest.record_sha256
                != source_publication.manifest.record_sha256
                or staged_publication.source.scientific_manifest.record_sha256
                != source_publication.source.scientific_manifest.record_sha256
            ):
                raise RuntimeError(
                    "Node-local subject-shape source staging differs from authority."
                )
        nested_scratch = plan.scratch_root / "scientific-compute"
        with telemetry.phase("scientific_compute"):
            local_compute = materialize_subject_shape(
                staged_archive,
                scratch_root=nested_scratch,
                refined_run=refined_run,
                run_name=candidate_name,
                storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
                components=COMPONENT_ORDER,
                block_rows=block_rows,
                output_shard_rows=output_shard_rows,
                execution_backend=execution_backend,
                scheduler=scheduler,
                num_workers=num_workers,
                shard_copy_workers=shard_copy_workers,
                native_threads=native_threads,
                copy_backend=copy_backend,
                apply=True,
                keep_scratch=False,
                check_capacity=check_capacity,
                stage_command="typed subject-shape execution candidate",
            )
        candidate_path = f"analysis/subject_shape_runs/{candidate_name}"
        local_candidate_path = staged_archive.joinpath(*candidate_path.split("/"))
        with telemetry.phase("local_validation"):
            local_root = open_zarr_root(staged_archive, mode="r")
            local_candidate = local_root[candidate_path]
            owner = local_candidate.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
            if type(owner) is not str:
                raise RuntimeError("Local subject-shape candidate lacks its owner.")
            local_publication = (
                load_completed_ineligible_subject_shape_coordinate_publication(
                    local_root,
                    candidate_path,
                    expected_publication_owner=owner,
                )
            )
            local_validation = _validate_subject_shape_run(
                local_candidate_path,
                row_count=plan.row_count,
                require_sharded=False,
                expected_binding_status=SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
                require_complete=True,
                expected_selector_eligible=False,
                storage_profile_id=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            )
            local_hashes = compute_subject_shape_logical_hashes(local_candidate)
            if not local_validation["valid"] or local_hashes != expected_hashes:
                raise RuntimeError(
                    "Local subject-shape candidate failed exact validation/equality."
                )
            if (
                local_publication.source.scientific_manifest.record_sha256
                != source_publication.source.scientific_manifest.record_sha256
            ):
                raise RuntimeError(
                    "Local subject-shape candidate changed refined authority."
                )
        with telemetry.phase("local_consolidation"):
            consolidate_metadata_capture_expected_warnings(staged_archive)
        with telemetry.phase("local_direct_consolidated_comparison"):
            local_metadata = validate_direct_consolidated_subtree(
                staged_archive,
                subtree_path=candidate_path,
            )

        def validate(path: Path) -> dict[str, Any]:
            return _validate_subject_shape_run(
                path,
                row_count=plan.row_count,
                require_sharded=False,
                expected_binding_status=SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
                require_complete=True,
                expected_selector_eligible=False,
                storage_profile_id=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            return (
                require_runs_parent(
                    root.require_group("analysis"),
                    "subject_shape_runs",
                ),
            )

        def complete(
            _root: zarr.Group,
            _parent: zarr.Group,
            run: zarr.Group,
        ) -> None:
            if (
                run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Copied subject-shape candidate lost its complete/ineligible state."
                )

        def verify(root: zarr.Group) -> None:
            parent = root["analysis/subject_shape_runs"]
            if _selector_snapshot(parent) != selectors_before:
                raise RuntimeError("Subject-shape execution changed parent selectors.")
            run = parent[candidate_name]
            if (
                run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise RuntimeError(
                    "Published subject-shape candidate is not complete/ineligible."
                )

        accepted: dict[str, Any] = {}

        def accept(
            root: zarr.Group,
            parent: zarr.Group,
            run: zarr.Group,
        ) -> None:
            if _selector_snapshot(parent) != selectors_before:
                raise RuntimeError(
                    "Subject-shape selectors changed before final acceptance."
                )
            with telemetry.phase("published_validation"):
                published_owner = run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
                if type(published_owner) is not str:
                    raise RuntimeError(
                        "Published subject-shape candidate lacks its owner."
                    )
                published_publication = (
                    load_completed_ineligible_subject_shape_coordinate_publication(
                        root,
                        candidate_path,
                        expected_publication_owner=published_owner,
                    )
                )
                stable_source = load_persisted_subject_shape_coordinate_publication(
                    root,
                    source_path,
                )
                if (
                    stable_source.manifest.record_sha256
                    != source_publication.manifest.record_sha256
                    or stable_source.source.scientific_manifest.record_sha256
                    != source_publication.source.scientific_manifest.record_sha256
                    or published_publication.source.scientific_manifest.record_sha256
                    != stable_source.source.scientific_manifest.record_sha256
                ):
                    raise RuntimeError(
                        "Subject-shape source authority changed during publication."
                    )
                published_validation = validate(plan.target_run_path)
                if not published_validation["valid"]:
                    raise RuntimeError(
                        f"Published subject-shape candidate is invalid: {published_validation}."
                    )
            consolidate_metadata_capture_expected_warnings(archive)
            with telemetry.phase("published_direct_consolidated_comparison"):
                issues = validate_subject_shape_direct_consolidated_storage(
                    archive,
                    run_path=candidate_path,
                    phase="bound",
                )
                if issues:
                    raise RuntimeError(
                        "Published subject-shape direct/consolidated metadata differs: "
                        + "; ".join(issues)
                    )
                published_metadata = validate_direct_consolidated_subtree(
                    archive,
                    subtree_path=candidate_path,
                )
            with telemetry.phase("decoded_equality"):
                published_hashes = compute_subject_shape_logical_hashes(run)
                if published_hashes != expected_hashes:
                    raise RuntimeError(
                        "Published subject-shape decoded arrays differ from source."
                    )
            with telemetry.phase("physical_inventory"):
                output_storage = storage_stats(plan.target_run_path)
                if (
                    output_storage["file_count"] < 1
                    or output_storage["apparent_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published subject-shape candidate has no physical payload."
                    )
            accepted.update(
                published_owner=published_owner,
                published_manifest_sha256=(
                    published_publication.manifest.record_sha256
                ),
                published_validation=published_validation,
                published_hashes=published_hashes,
                published_metadata=published_metadata.to_json(),
                output_storage=output_storage,
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    accepted["caller_acceptance"] = json_attr_safe(
                        dict(publication_acceptance_validator(root, parent, run))
                    )

        def repair(_target: Path) -> None:
            consolidate_metadata_capture_expected_warnings(archive)

        with telemetry.phase("atomic_publication"):
            publication = atomic_publish_run_group(
                AtomicRunPublishSpec(
                    source_zarr=archive,
                    local_run_path=local_candidate_path,
                    target_run_path=plan.target_run_path,
                    run_name=candidate_name,
                    lock_suffix="subject-shape-execution-candidate",
                    publish_schema_id=PUBLISH_SCHEMA_ID,
                    policy=(
                        "subject_shape_v4_scientific_recompute_atomic_nonpromoting"
                    ),
                    rollback_policy=(
                        "retain_failed_public_tombstone_leave_parent_selectors_untouched"
                    ),
                    content_checksum=True,
                    publication_owner_attr=SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
                ),
                copy_backend=copy_backend,
                validate_run=validate,
                prepare_parents=prepare,
                complete_run=complete,
                verify_pointers=verify,
                activate_run=accept,
                repair_failed_publication_visibility=repair,
                payload_metadata={
                    SUBJECT_SHAPE_EXECUTION_BINDING_ATTR: binding,
                    "source_run": source_name,
                    "source_run_path": source_path,
                    "source_staging_mode": "archive_snapshot_copy_v1",
                    "storage_profile_id": (
                        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
                    ),
                    "promotion_policy": (
                        "immutable_named_candidate_no_pointer_registry_or_profile_activation"
                    ),
                    "local_direct_consolidated": local_metadata.to_json(),
                    "nested_scientific_materialization": local_compute,
                },
            )
        result.update(
            status="complete",
            source_logical_manifest_sha256=canonical_json_sha256(expected_hashes),
            published_logical_manifest_sha256=canonical_json_sha256(
                accepted["published_hashes"]
            ),
            local_validation=local_validation,
            local_direct_consolidated=local_metadata.to_json(),
            published_validation=accepted["published_validation"],
            published_direct_consolidated=accepted["published_metadata"],
            published_manifest_sha256=accepted["published_manifest_sha256"],
            output_storage=accepted["output_storage"],
            caller_acceptance=accepted.get("caller_acceptance"),
            publication=publication,
            runtime_telemetry=_ordered_execution_telemetry(telemetry),
        )
        succeeded = True
        return json_attr_safe(result)
    except BaseException as exc:
        try:
            setattr(
                exc,
                "palette_runtime_telemetry",
                _ordered_execution_telemetry(telemetry),
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
        return scratch_user / job_id / f"palette_subject_shape_{run_name}"
    return (
        Path(os.environ.get("TMPDIR") or "/tmp")
        / f"palette_subject_shape_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--refined-run")
    parser.add_argument("--subject-mask-bundle-id")
    parser.add_argument(
        "--allow-inactive-subject-mask-bundle",
        action="store_true",
        help="Authorize exactly the named inactive bundle for a selector-ineligible canary.",
    )
    parser.add_argument("--assignment-keypoint-rebinding-run")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--storage-profile",
        choices=SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES,
        default=SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    )
    parser.add_argument("--component", action="append", dest="components")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument(
        "--output-shard-rows", type=int, default=DEFAULT_OUTPUT_SHARD_ROWS
    )
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
    parser.add_argument(
        "--shard-copy-workers", type=int, default=DEFAULT_SHARD_COPY_WORKERS
    )
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
        storage_profile=str(args.storage_profile),
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
        subject_mask_bundle_id=args.subject_mask_bundle_id,
        allow_inactive_subject_mask_bundle=(
            args.allow_inactive_subject_mask_bundle
        ),
        assignment_keypoint_rebinding_run_id=(
            args.assignment_keypoint_rebinding_run
        ),
    )
    if args.report is not None:
        _write_json_atomic(args.report.expanduser().resolve(), result)
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
