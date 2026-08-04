"""Node-local compute and atomic publication for tail-posture v3 candidates.

The maintained production writer remains a guarded direct writer.  This
module is an execution-only boundary: it snapshots the exact logical inputs
into process-local arrays, computes and writes a byte-planned candidate in
node-local scratch, and publishes only that immutable run child through the
shared atomic publisher.  It never activates a selector or promotes a profile.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import zarr

from fisheye.analysis import tail_posture_view_runs as tail_mod
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    build_direct_writer_storage_receipt,
    create_direct_writer_arrays_from_receipt,
    persist_direct_writer_storage_receipt,
)
from fisheye.analysis.tail_posture_view_schema import (
    TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
    TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
    TAIL_POSTURE_VIEW_FILL_VALUES,
    TailPostureViewDimensions,
    validate_tail_posture_view_arrays,
    write_tail_posture_view_array_schema_manifest,
)
from fisheye.analysis_workflows.tail_posture_candidate_execution import (
    build_tail_posture_scientific_identity,
    compute_tail_posture_logical_hashes,
    infer_tail_posture_dimensions,
)
from fisheye.shared import tail_coordinate_publication as tail_publication
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.zarr.benchmark_runtime import storage_stats
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import StorageProfile
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from .runtime_telemetry import PhaseTelemetry

MATERIALIZATION_SCHEMA_ID = "palette.tail_posture_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.tail_posture_candidate_publish.v1"
SOURCE_STAGING_SCHEMA_ID = "palette.tail_posture_logical_source_snapshot.v1"
TAIL_POSTURE_EXECUTION_PHASE_ORDER = (
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
)
EXECUTION_BINDING_ATTR = "analysis_candidate_execution_binding"
EXECUTION_FAILURE_TOMBSTONE_ATTR = "analysis_candidate_execution_tombstone"
RUN_PARENT = "analysis/tail_posture_view_runs"
DEFAULT_CAPACITY_MARGIN_BYTES = 64 * 1024 * 1024
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


@dataclass(frozen=True)
class TailPostureMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    source_run_name: str
    source_run_path: str
    run_name: str
    target_run_path: Path
    subject_shape_run: str
    source_subject_shape_manifest_sha256: str
    source_tail_posture_manifest_sha256: str
    source_tail_kinematics_run: str | None
    source_tail_kinematics_manifest_sha256: str | None
    view_family: str
    head_source: str
    keypoint_count: int
    row_count: int
    storage_profile_id: str
    estimated_output_bytes: int

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr / RUN_PARENT / self.run_name

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "source_run_name": self.source_run_name,
            "source_run_path": self.source_run_path,
            "run_name": self.run_name,
            "target_run_path": str(self.target_run_path),
            "subject_shape_run": self.subject_shape_run,
            "source_subject_shape_manifest_sha256": (
                self.source_subject_shape_manifest_sha256
            ),
            "source_tail_posture_manifest_sha256": (
                self.source_tail_posture_manifest_sha256
            ),
            "source_tail_kinematics_run": self.source_tail_kinematics_run,
            "source_tail_kinematics_manifest_sha256": (
                self.source_tail_kinematics_manifest_sha256
            ),
            "view_family": self.view_family,
            "head_source": self.head_source,
            "keypoint_count": self.keypoint_count,
            "row_count": self.row_count,
            "storage_profile_id": self.storage_profile_id,
            "source_staging_mode": "logical_array_snapshot_v1",
            "estimated_output_bytes": self.estimated_output_bytes,
        }


@dataclass(frozen=True)
class TailPostureSourceSnapshot:
    shape_group: zarr.Group
    shape_tables: Any
    source_arrays: Mapping[str, np.ndarray]
    lineage_arrays: Mapping[str, np.ndarray]
    source_bytes: int
    source_subject_shape_manifest_sha256: str


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return name


def _estimated_output_bytes(*, rows: int, keypoints: int) -> int:
    angles = keypoints - 1
    # Three lineage columns, validity/reason, head point/yaw, tail points, and
    # the two angle representations.  Metadata/copy overhead uses a separate
    # fixed safety margin.
    per_row = (8 + 8 + 8) + 1 + 64 + (2 * 4) + 4
    per_row += keypoints * 2 * 4
    per_row += angles * 4 * 2
    return max(1, rows * per_row)


def build_tail_posture_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    source_run_name: str,
    run_name: str,
    subject_shape_run: str,
    source_subject_shape_manifest_sha256: str,
    source_tail_posture_manifest_sha256: str,
    source_tail_kinematics_run: str | None,
    source_tail_kinematics_manifest_sha256: str | None,
    view_family: str,
    head_source: str,
    keypoint_count: int,
    storage_profile: StorageProfile,
) -> TailPostureMaterializationPlan:
    archive = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Tail-posture source archive not found: {archive}")
    try:
        scratch.relative_to(archive)
    except ValueError:
        pass
    else:
        raise ValueError("Tail-posture scratch must be outside the source archive")
    source_name = _safe_name(source_run_name, label="source run name")
    target_name = _safe_name(run_name, label="candidate run name")
    shape_name = _safe_name(subject_shape_run, label="subject-shape run name")
    if source_name == target_name:
        raise ValueError("Tail-posture source and candidate names must differ")
    if view_family != "megabouts_compatible":
        raise ValueError("Typed tail-posture execution requires megabouts_compatible")
    if head_source not in {"head_endpoint_xy", "snout_tip_xy"}:
        raise ValueError("Tail-posture head_source is unsupported")
    if type(keypoint_count) is not int or keypoint_count < 2:
        raise ValueError("Tail-posture keypoint_count must be an integer >= 2")
    if not isinstance(storage_profile, StorageProfile):
        raise TypeError("Tail-posture storage profile must be explicit")
    if storage_profile.profile_id != "published_http_v1":
        raise ValueError("Tail-posture execution supports only published_http_v1")

    root = open_zarr_root(archive, mode="r")
    source_path = f"{RUN_PARENT}/{source_name}"
    source = root.get(source_path)
    if not isinstance(source, zarr.Group):
        raise ValueError("Tail-posture source run is missing")
    dimensions = infer_tail_posture_dimensions(source)
    if dimensions.n_keypoints != keypoint_count:
        raise ValueError("Tail-posture source keypoint count differs")
    if source.attrs.get("source_subject_shape_run") != shape_name:
        raise ValueError("Tail-posture source subject-shape binding differs")
    target = archive / RUN_PARENT / target_name
    if target.exists() or root.get(f"{RUN_PARENT}/{target_name}") is not None:
        raise FileExistsError(f"Tail-posture candidate already exists: {target_name}")
    if source_tail_kinematics_run is not None:
        _safe_name(source_tail_kinematics_run, label="tail-kinematics run name")
        if source_tail_kinematics_manifest_sha256 is None:
            raise ValueError("Tail-kinematics run requires its manifest digest")
    elif source_tail_kinematics_manifest_sha256 is not None:
        raise ValueError("Tail-kinematics digest requires its source run")
    return TailPostureMaterializationPlan(
        source_zarr=archive,
        scratch_root=scratch,
        local_zarr=scratch / "tail-posture-output.zarr",
        source_run_name=source_name,
        source_run_path=source_path,
        run_name=target_name,
        target_run_path=target,
        subject_shape_run=shape_name,
        source_subject_shape_manifest_sha256=(
            str(source_subject_shape_manifest_sha256)
        ),
        source_tail_posture_manifest_sha256=str(source_tail_posture_manifest_sha256),
        source_tail_kinematics_run=source_tail_kinematics_run,
        source_tail_kinematics_manifest_sha256=(source_tail_kinematics_manifest_sha256),
        view_family=view_family,
        head_source=head_source,
        keypoint_count=keypoint_count,
        row_count=dimensions.n_rows,
        storage_profile_id=storage_profile.profile_id,
        estimated_output_bytes=_estimated_output_bytes(
            rows=dimensions.n_rows,
            keypoints=keypoint_count,
        ),
    )


def snapshot_tail_posture_sources(
    plan: TailPostureMaterializationPlan,
    *,
    check_capacity: bool,
) -> TailPostureSourceSnapshot:
    """Read the exact scientific inputs into process-local immutable arrays."""

    if plan.scratch_root.exists():
        raise FileExistsError(
            f"Tail-posture scratch already exists: {plan.scratch_root}"
        )
    parent = plan.scratch_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(parent).free)
    required = int(plan.estimated_output_bytes + DEFAULT_CAPACITY_MARGIN_BYTES)
    if check_capacity and free < required:
        raise OSError(
            f"Insufficient scratch capacity: need {required} bytes, found {free}"
        )
    plan.scratch_root.mkdir(parents=False, exist_ok=False)
    plan.local_zarr.mkdir(parents=False, exist_ok=False)

    root = open_zarr_root(plan.source_zarr, mode="r")
    shape_name, shape_group, shape_tables = tail_mod._resolve_subject_shape_tables(
        root,
        plan.subject_shape_run,
        head_source=plan.head_source,
    )
    if shape_name != plan.subject_shape_run:
        raise ValueError("Resolved subject-shape run differs from the plan")
    source_arrays, row_count = tail_mod._read_sources(
        shape_tables,
        head_source=plan.head_source,
    )
    if row_count != plan.row_count:
        raise ValueError("Subject-shape row count differs from tail-posture source")
    shape_manifest = tail_mod._require_fresh_subject_shape_manifest(root, shape_tables)
    if shape_manifest != plan.source_subject_shape_manifest_sha256:
        raise ValueError("Subject-shape publication differs from the plan")
    lineage = tail_mod._read_tail_posture_lineage_arrays(
        shape_group,
        row_count=row_count,
    )
    frozen_sources = {
        name: np.asarray(values).copy() for name, values in source_arrays.items()
    }
    frozen_lineage = {
        name: np.asarray(values).copy() for name, values in lineage.items()
    }
    source_bytes = sum(value.nbytes for value in frozen_sources.values()) + sum(
        value.nbytes for value in frozen_lineage.values()
    )
    return TailPostureSourceSnapshot(
        shape_group=shape_group,
        shape_tables=shape_tables,
        source_arrays=frozen_sources,
        lineage_arrays=frozen_lineage,
        source_bytes=int(source_bytes),
        source_subject_shape_manifest_sha256=shape_manifest,
    )


def _candidate_arrays(
    snapshot: TailPostureSourceSnapshot,
    batch: tail_mod.TailPostureViewBatch,
) -> dict[str, np.ndarray]:
    return {
        **{
            name: np.asarray(values) for name, values in snapshot.lineage_arrays.items()
        },
        "valid": batch.valid.astype(bool, copy=False),
        "failure_reason_bytes": batch.failure_reason_bytes.astype(np.uint8, copy=False),
        "head_xy": batch.head_xy.astype(np.float32, copy=False),
        "head_yaw_rad": batch.head_yaw_rad.astype(np.float32, copy=False),
        "tail_keypoints_xy": batch.tail_keypoints_xy.astype(np.float32, copy=False),
        "tail_angle_rad": batch.tail_angle_rad.astype(np.float32, copy=False),
        "tail_angle_deg": batch.tail_angle_deg.astype(np.float32, copy=False),
    }


def write_local_tail_posture_candidate(
    plan: TailPostureMaterializationPlan,
    snapshot: TailPostureSourceSnapshot,
    *,
    batch: tail_mod.TailPostureViewBatch,
    storage_profile: StorageProfile,
    execution_binding: Mapping[str, Any],
    stage_command: str,
) -> tuple[zarr.Group, dict[str, object]]:
    """Write one running, unsealed candidate into the node-local archive."""

    local_root = open_zarr_root(plan.local_zarr, mode="a")
    owner = str(uuid.uuid4())
    run = tail_mod._prepare_run_group(
        local_root,
        target_run=plan.run_name,
        shape_run_name=plan.subject_shape_run,
        shape_group=snapshot.shape_group,
        source_subject_shape_publication_manifest_sha256=(
            snapshot.source_subject_shape_manifest_sha256
        ),
        row_count=plan.row_count,
        view_family=plan.view_family,
        head_source=plan.head_source,
        keypoint_count=plan.keypoint_count,
        source_tail_kinematics_run=plan.source_tail_kinematics_run,
        stage_command=stage_command,
        publication_owner_uuid=owner,
        overwrite=False,
        legacy_storage_layout=False,
    )
    arrays = _candidate_arrays(snapshot, batch)
    dimensions = TailPostureViewDimensions(
        n_rows=plan.row_count,
        n_keypoints=plan.keypoint_count,
        n_angles=plan.keypoint_count - 1,
    )
    receipt = build_direct_writer_storage_receipt(
        declarations=TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
        arrays_by_path=arrays,
        access_unit_semantics=TAIL_POSTURE_VIEW_ACCESS_UNIT_SEMANTICS,
        profile=storage_profile,
        dimensions=dimensions.contract_dimensions,
    )
    persist_direct_writer_storage_receipt(run, receipt)
    create_direct_writer_arrays_from_receipt(
        run,
        receipt=receipt,
        arrays_by_path=arrays,
        fill_values=TAIL_POSTURE_VIEW_FILL_VALUES,
    )
    write_tail_posture_view_array_schema_manifest(
        run,
        n_rows=dimensions.n_rows,
        n_keypoints=dimensions.n_keypoints,
        n_angles=dimensions.n_angles,
        byte_planner_adopted=True,
    )
    run.attrs[EXECUTION_BINDING_ATTR] = json_attr_safe(dict(execution_binding))
    run.attrs["storage_candidate_profile_promoted"] = False
    run.attrs["node_local_source_staging"] = json_attr_safe(
        {
            "schema_id": SOURCE_STAGING_SCHEMA_ID,
            "mode": "logical_array_snapshot_v1",
            "source_zarr": str(plan.source_zarr),
            "source_tail_posture_run": plan.source_run_path,
            "source_subject_shape_run": (
                f"analysis/subject_shape_runs/{plan.subject_shape_run}"
            ),
            "source_subject_shape_manifest_sha256": (
                snapshot.source_subject_shape_manifest_sha256
            ),
            "source_bytes": snapshot.source_bytes,
            "array_count": len(snapshot.source_arrays) + len(snapshot.lineage_arrays),
        }
    )
    return run, receipt.as_manifest()


def _validate_candidate_group(
    run: zarr.Group,
    *,
    expected_status: set[str],
    expected_execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    dimensions = infer_tail_posture_dimensions(run)
    issues = validate_tail_posture_view_arrays(run, dimensions=dimensions)
    errors = [f"{item.code}:{item.path}:{item.message}" for item in issues]
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) not in expected_status:
        errors.append("completion status differs")
    if (
        run.attrs.get("stage_selector_eligible") is not False
        or run.attrs.get("storage_candidate_profile_promoted") is not False
        or run.attrs.get(EXECUTION_BINDING_ATTR)
        != json_attr_safe(dict(expected_execution_binding))
        or not isinstance(run.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR), Mapping)
    ):
        errors.append("candidate lifecycle or execution binding differs")
    try:
        build_tail_posture_scientific_identity(run)
        logical = compute_tail_posture_logical_hashes(run)
    except Exception as exc:
        errors.append(f"logical validation failed: {exc}")
        logical = None
    return {
        "valid": not errors,
        "errors": errors,
        "dimensions": dimensions.contract_dimensions,
        "logical_hashes": logical,
    }


def _validate_candidate_path(
    path: Path,
    *,
    expected_execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    run = open_zarr_root(path, mode="r")
    return _validate_candidate_group(
        run,
        expected_status={RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE},
        expected_execution_binding=expected_execution_binding,
    )


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index for index, name in enumerate(TAIL_POSTURE_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Tail-posture telemetry contains an unknown phase")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


def tombstone_tail_posture_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail only the exact execution-owned ineligible candidate."""

    archive = Path(source_zarr).expanduser().resolve()
    name = _safe_name(run_name, label="candidate run name")
    binding = json_attr_safe(dict(expected_execution_binding))
    if not binding:
        raise ValueError("Tail-posture execution binding must be nonempty")
    payload = {
        "schema_id": "palette.analysis_candidate_execution_tombstone",
        "schema_version": 1,
        "execution_binding": binding,
        "failure_phase": str(failure_phase),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }
    tombstone = {**payload, "payload_sha256": canonical_json_sha256(payload)}
    run_path = f"{RUN_PARENT}/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        run = root.get(run_path)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        if run.attrs.get(EXECUTION_BINDING_ATTR) != binding:
            raise RuntimeError("Refusing to tombstone a foreign tail-posture run")
        if run.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Refusing to tombstone an eligible tail-posture run")
        existing = run.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        status = run.attrs.get(RUN_COMPLETION_STATUS_ATTR)
        if status == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError("Existing tail-posture tombstone differs")
        elif status == RUN_STATUS_COMPLETE:
            mark_run_failed(
                run,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            run.attrs["stage_selector_eligible"] = False
            run.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        else:
            raise RuntimeError("Tail-posture candidate is neither complete nor failed")
        consolidate_metadata_capture_expected_warnings(archive)
        metadata = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        fresh = open_zarr_root(archive, mode="r")[run_path]
        if (
            fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
            or fresh.attrs.get("stage_selector_eligible") is not False
            or fresh.attrs.get(EXECUTION_BINDING_ATTR) != binding
            or fresh.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR) != tombstone
        ):
            raise RuntimeError("Tail-posture tombstone did not persist exactly")
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": metadata.declarations_sha256,
    }


def publish_tail_posture_candidate(
    plan: TailPostureMaterializationPlan,
    *,
    execution_binding: Mapping[str, Any],
    expected_source_logical_hashes: Mapping[str, Any],
    copy_backend: str,
    telemetry: PhaseTelemetry,
    publication_acceptance_validator: PublicationAcceptanceValidator,
) -> dict[str, Any]:
    """Atomically publish and accept one selector-ineligible candidate."""

    binding = json_attr_safe(dict(execution_binding))
    expected_hashes = dict(expected_source_logical_hashes)
    acceptance: dict[str, Any] = {}

    def phase(name: str):
        return telemetry.phase(name) if telemetry is not None else nullcontext()

    def validate(path: Path) -> dict[str, Any]:
        return _validate_candidate_path(
            path,
            expected_execution_binding=binding,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"), "tail_posture_view_runs"
            ),
        )

    def complete(
        root: zarr.Group,
        parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        run.attrs["stage_selector_eligible"] = False
        tail_mod.publish_tail_posture_coordinate_surfaces(root, run)
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run.attrs.get("provenance", {}),
                fallback_command="tail-posture typed candidate execution",
            ),
        )

    def verify(root: zarr.Group) -> None:
        parent = root[RUN_PARENT]
        if (
            parent.attrs.get("latest") == plan.run_name
            or parent.attrs.get("latest_complete") == plan.run_name
            or parent.attrs.get("latest_megabouts_compatible") == plan.run_name
        ):
            raise RuntimeError("Tail-posture candidate changed a parent selector")

    def activate(
        root: zarr.Group,
        parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        run_path = f"{RUN_PARENT}/{plan.run_name}"
        with phase("published_validation"):
            validation = _validate_candidate_group(
                run,
                expected_status={RUN_STATUS_COMPLETE},
                expected_execution_binding=binding,
            )
            if not validation["valid"]:
                raise RuntimeError(
                    f"Published tail-posture candidate is invalid: {validation}"
                )
            publication = (
                tail_publication._load_tail_coordinate_publication(  # noqa: SLF001
                    root,
                    run_path,
                    expected_selector_eligible=False,
                    expected_kind="tail_posture_view",
                    require_complete=True,
                )
            )
            if publication.source.manifest.record_sha256 != (
                plan.source_subject_shape_manifest_sha256
            ):
                raise RuntimeError("Published tail-posture source authority differs")
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        with phase("published_direct_consolidated_comparison"):
            metadata = validate_direct_consolidated_subtree(
                plan.source_zarr,
                subtree_path=run_path,
            )
        with phase("decoded_equality"):
            hashes = compute_tail_posture_logical_hashes(run)
            if hashes != expected_hashes:
                raise RuntimeError("Published tail-posture decoded values differ")
        with phase("physical_inventory"):
            physical = storage_stats(plan.target_run_path)
            if (
                physical["file_count"] < 1
                or physical["apparent_bytes"] < 1
                or physical["allocated_bytes"] < 1
            ):
                raise RuntimeError("Published tail-posture candidate has no payload")
        acceptance.update(
            {
                "published_validation": validation,
                "published_coordinate_manifest_sha256": (
                    publication.manifest.record_sha256
                ),
                "published_direct_consolidated_array_count": metadata.array_count,
                "published_logical_hashes": hashes,
                "published_logical_manifest_sha256": canonical_json_sha256(hashes),
                "output_storage": physical,
                "caller_acceptance": json_attr_safe(
                    dict(publication_acceptance_validator(root, parent, run))
                ),
            }
        )

    def repair(_target: Path) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)

    result = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="tail-posture-candidate-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_tail_posture_candidate_atomic_publish",
            rollback_policy=(
                "retain_owner_bound_failed_public_tombstone_without_selector_rollback"
            ),
            publication_owner_attr=tail_publication.TAIL_PUBLICATION_OWNER_ATTR,
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        repair_failed_publication_visibility=repair,
        payload_metadata={
            "source_run_path": plan.source_run_path,
            "source_subject_shape_run": plan.subject_shape_run,
            "source_subject_shape_manifest_sha256": (
                plan.source_subject_shape_manifest_sha256
            ),
            "source_tail_posture_manifest_sha256": (
                plan.source_tail_posture_manifest_sha256
            ),
            "source_staging_mode": "logical_array_snapshot_v1",
        },
    )
    result.update(acceptance)
    result["registry_updated"] = False
    return result


def materialize_tail_posture_candidate(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    source_run_name: str,
    run_name: str,
    subject_shape_run: str,
    source_subject_shape_manifest_sha256: str,
    source_tail_posture_manifest_sha256: str,
    source_tail_kinematics_run: str | None,
    source_tail_kinematics_manifest_sha256: str | None,
    view_family: str,
    head_source: str,
    keypoint_count: int,
    storage_profile: StorageProfile,
    copy_backend: str,
    keep_scratch: bool,
    check_capacity: bool,
    execution_binding: Mapping[str, Any],
    expected_source_logical_hashes: Mapping[str, Any],
    publication_acceptance_validator: PublicationAcceptanceValidator,
    stage_command: str | None = None,
) -> dict[str, Any]:
    """Execute the complete nonpromoting tail-posture candidate workflow."""

    if copy_backend not in {"python", "rsync"}:
        raise ValueError("Tail-posture copy backend is unsupported")
    binding = json_attr_safe(dict(execution_binding))
    if not binding or not expected_source_logical_hashes:
        raise ValueError("Typed tail-posture execution bindings must be nonempty")
    telemetry = PhaseTelemetry(
        materializer="tail_posture_candidate",
        context={
            "source_run_name": source_run_name,
            "run_name": run_name,
            "storage_profile_id": storage_profile.profile_id,
        },
    )
    with telemetry.phase("plan"):
        plan = build_tail_posture_materialization_plan(
            source_zarr,
            scratch_root=scratch_root,
            source_run_name=source_run_name,
            run_name=run_name,
            subject_shape_run=subject_shape_run,
            source_subject_shape_manifest_sha256=(source_subject_shape_manifest_sha256),
            source_tail_posture_manifest_sha256=(source_tail_posture_manifest_sha256),
            source_tail_kinematics_run=source_tail_kinematics_run,
            source_tail_kinematics_manifest_sha256=(
                source_tail_kinematics_manifest_sha256
            ),
            view_family=view_family,
            head_source=head_source,
            keypoint_count=keypoint_count,
            storage_profile=storage_profile,
        )
    succeeded = False
    try:
        with telemetry.phase("source_staging"):
            snapshot = snapshot_tail_posture_sources(
                plan,
                check_capacity=check_capacity,
            )
        with telemetry.phase("scientific_compute"):
            batch = tail_mod.compute_tail_posture_view_from_subject_shape_arrays(
                **snapshot.source_arrays,
                keypoint_count=plan.keypoint_count,
            )
            local_run, storage_receipt = write_local_tail_posture_candidate(
                plan,
                snapshot,
                batch=batch,
                storage_profile=storage_profile,
                execution_binding=binding,
                stage_command=stage_command
                or (" ".join(sys.argv) if sys.argv else "unknown"),
            )
        with telemetry.phase("local_validation"):
            local_validation = _validate_candidate_group(
                local_run,
                expected_status={RUN_STATUS_RUNNING},
                expected_execution_binding=binding,
            )
            if not local_validation["valid"]:
                raise RuntimeError(
                    f"Local tail-posture candidate is invalid: {local_validation}"
                )
            local_hashes = local_validation["logical_hashes"]
            if local_hashes != dict(expected_source_logical_hashes):
                raise RuntimeError("Local tail-posture decoded values differ")
        with telemetry.phase("local_consolidation"):
            consolidate_metadata_capture_expected_warnings(plan.local_zarr)
        with telemetry.phase("local_direct_consolidated_comparison"):
            local_metadata = validate_direct_consolidated_subtree(
                plan.local_zarr,
                subtree_path=f"{RUN_PARENT}/{plan.run_name}",
            )
        with telemetry.phase("atomic_publication"):
            published = publish_tail_posture_candidate(
                plan,
                execution_binding=binding,
                expected_source_logical_hashes=expected_source_logical_hashes,
                copy_backend=copy_backend,
                telemetry=telemetry,
                publication_acceptance_validator=publication_acceptance_validator,
            )
        result = {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "status": "complete",
            "mutates_archive": True,
            "plan": plan.as_manifest(),
            "source_staging": {
                "schema_id": SOURCE_STAGING_SCHEMA_ID,
                "source_bytes": snapshot.source_bytes,
                "source_subject_shape_manifest_sha256": (
                    snapshot.source_subject_shape_manifest_sha256
                ),
            },
            "storage_plan": storage_receipt,
            "local_direct_consolidated_array_count": local_metadata.array_count,
            "source_logical_manifest_sha256": canonical_json_sha256(
                expected_source_logical_hashes
            ),
            "local_logical_manifest_sha256": canonical_json_sha256(local_hashes),
            "published_logical_manifest_sha256": published[
                "published_logical_manifest_sha256"
            ],
            "published_direct_consolidated_array_count": published[
                "published_direct_consolidated_array_count"
            ],
            "output_storage": published["output_storage"],
            "caller_acceptance": published["caller_acceptance"],
            "publish": published,
        }
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


__all__ = [
    "EXECUTION_BINDING_ATTR",
    "EXECUTION_FAILURE_TOMBSTONE_ATTR",
    "TAIL_POSTURE_EXECUTION_PHASE_ORDER",
    "TailPostureMaterializationPlan",
    "build_tail_posture_materialization_plan",
    "materialize_tail_posture_candidate",
    "publish_tail_posture_candidate",
    "snapshot_tail_posture_sources",
    "tombstone_tail_posture_execution_candidate",
]
