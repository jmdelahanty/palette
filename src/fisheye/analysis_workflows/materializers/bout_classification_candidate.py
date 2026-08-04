"""Replay and atomically publish one bout-classification storage candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import zarr

from fisheye.analysis.bout_classification_runs import (
    BOUT_CLASSIFICATION_RUN_PARENT,
    resolve_bout_classification_run,
    validate_staged_bout_classification_run,
)
from fisheye.analysis.megabouts_classifier import write_megabouts_classification_run
from fisheye.analysis_workflows.bout_classification_candidate_execution import (
    BOUT_CLASSIFICATION_ARRAY_COUNT,
    BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID,
    bout_classification_logical_manifest_sha256,
    build_bout_classification_coordinate_evidence,
    build_bout_classification_scientific_identity,
    reconstruct_bout_classification_writer_inputs,
    require_exact_bout_classification_run,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import storage_stats
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from .runtime_telemetry import PhaseTelemetry

MATERIALIZATION_SCHEMA_ID = "palette.bout_classification_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.bout_classification_atomic_publish.v1"
BOUT_CLASSIFICATION_EXECUTION_PHASE_ORDER = (
    "plan",
    "source_staging",
    "logical_rematerialization",
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
_NODE_LOCAL_ROOTS = tuple(
    Path(value)
    for value in ("/tmp", "/var/tmp", "/scratch", "/dev/shm", "/local", "/lscratch")
)
_CAPACITY_MARGIN_BYTES = 64 * 1024 * 1024
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


@dataclass(frozen=True)
class BoutClassificationCandidatePlan:
    archive: Path
    source_run_name: str
    run_name: str
    scratch_root: Path
    staged_source_path: Path
    local_archive: Path
    source_tree_bytes: int
    latest_before: str | None
    latest_complete_before: str | None
    source_identity_sha256: str
    source_logical_manifest_sha256: str

    @property
    def source_run_path(self) -> str:
        return f"{BOUT_CLASSIFICATION_RUN_PARENT}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{BOUT_CLASSIFICATION_RUN_PARENT}/{self.run_name}"

    @property
    def source_tree_path(self) -> Path:
        return self.archive.joinpath(*self.source_run_path.split("/"))

    @property
    def local_run_path(self) -> Path:
        return self.local_archive.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.archive.joinpath(*self.run_path.split("/"))


def _safe_name(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string")
    if (
        not value
        or value in {".", "..", "latest", "latest_complete"}
        or "/" in value
        or "\\" in value
        or value != value.strip()
    ):
        raise ValueError(f"{label} must be one explicit immutable run name")
    return value


def _require_node_local(path: Path) -> None:
    if not any(path == root or path.is_relative_to(root) for root in _NODE_LOCAL_ROOTS):
        raise ValueError("bout-classification scratch must be node-local")


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _require_symlink_free(path: Path) -> None:
    if path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("bout-classification source tree contains a symlink")


def _copy_tree(source: Path, target: Path, *, backend: str) -> None:
    _require_symlink_free(source)
    if backend == "python":
        shutil.copytree(source, target, symlinks=False)
    elif backend == "rsync":
        target.mkdir(parents=True)
        import subprocess

        subprocess.run(
            ["rsync", "--archive", "--copy-links", f"{source}/", f"{target}/"],
            check=True,
        )
    else:
        raise ValueError("copy_backend must be python or rsync")
    _require_symlink_free(target)


def _ordered_runtime(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index
        for index, name in enumerate(BOUT_CLASSIFICATION_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("bout-classification telemetry phase differs")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


def build_bout_classification_candidate_plan(
    archive: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
) -> BoutClassificationCandidatePlan:
    source_archive = Path(archive).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    _require_node_local(scratch)
    if not source_archive.is_dir():
        raise FileNotFoundError(f"analysis archive does not exist: {source_archive}")
    if scratch == source_archive or scratch.is_relative_to(source_archive):
        raise ValueError("scratch must be outside the authoritative archive")
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(run_name, label="run_name")
    if source_name == candidate_name:
        raise ValueError("source and candidate names must differ")
    root = open_zarr_root(source_archive, mode="r")
    source_group, resolved, _path = resolve_bout_classification_run(root, source_name)
    if resolved != source_name:
        raise ValueError("bout-classification source resolution differs")
    require_exact_bout_classification_run(source_group)
    parent = root[BOUT_CLASSIFICATION_RUN_PARENT]
    if (
        candidate_name in parent
        or source_archive.joinpath(
            *BOUT_CLASSIFICATION_RUN_PARENT.split("/"), candidate_name
        ).exists()
    ):
        raise FileExistsError("bout-classification candidate already exists")
    source_path = source_archive.joinpath(
        *BOUT_CLASSIFICATION_RUN_PARENT.split("/"), source_name
    )
    _require_symlink_free(source_path)
    return BoutClassificationCandidatePlan(
        archive=source_archive,
        source_run_name=source_name,
        run_name=candidate_name,
        scratch_root=scratch,
        staged_source_path=scratch / "staged-source-run",
        local_archive=scratch / "candidate.zarr",
        source_tree_bytes=_tree_bytes(source_path),
        latest_before=parent.attrs.get("latest"),
        latest_complete_before=parent.attrs.get("latest_complete"),
        source_identity_sha256=canonical_json_sha256(
            build_bout_classification_scientific_identity(source_group)
        ),
        source_logical_manifest_sha256=(
            bout_classification_logical_manifest_sha256(source_group)
        ),
    )


def _validate_candidate(
    group: Any,
    *,
    source_group: Any,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        require_exact_bout_classification_run(group)
        if group.attrs.get("stage_selector_eligible") is not False:
            errors.append("candidate is selector eligible")
        if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            errors.append("candidate is not complete")
        coordinate = build_bout_classification_coordinate_evidence(source_group, group)
        source_hash = bout_classification_logical_manifest_sha256(source_group)
        candidate_hash = bout_classification_logical_manifest_sha256(group)
        if source_hash != candidate_hash:
            errors.append("decoded logical arrays differ from source")
    except Exception as exc:
        errors.append(str(exc))
        coordinate = None
        source_hash = bout_classification_logical_manifest_sha256(source_group)
        candidate_hash = None
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": BOUT_CLASSIFICATION_ARRAY_COUNT,
        "source_logical_manifest_sha256": source_hash,
        "candidate_logical_manifest_sha256": candidate_hash,
        "coordinate_evidence": coordinate,
    }


def materialize_bout_classification_candidate(
    archive: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    check_capacity: bool = True,
    execution_binding: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Replay the exact source through the writer and publish atomically."""

    if profile_id != BOUT_CLASSIFICATION_EXECUTION_PROFILE_ID:
        raise ValueError("bout-classification candidate profile differs")
    telemetry = PhaseTelemetry(
        materializer="bout_classification_candidate",
        context={"source_run": source_run, "run_name": run_name},
    )
    with telemetry.phase("plan"):
        plan = build_bout_classification_candidate_plan(
            archive,
            source_run=source_run,
            run_name=run_name,
            scratch_root=scratch_root,
        )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "source_run_path": plan.source_run_path,
        "run_path": plan.run_path,
        "source_identity_sha256": plan.source_identity_sha256,
        "source_logical_manifest_sha256": plan.source_logical_manifest_sha256,
    }
    if not apply:
        result["runtime_telemetry"] = _ordered_runtime(telemetry)
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"scratch root already exists: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        with telemetry.phase("source_staging"):
            required = plan.source_tree_bytes * 3 + _CAPACITY_MARGIN_BYTES
            free = shutil.disk_usage(plan.scratch_root).free
            if check_capacity and free < required:
                raise OSError(
                    f"insufficient scratch: require {required}, available {free}"
                )
            _copy_tree(
                plan.source_tree_path,
                plan.staged_source_path,
                backend=copy_backend,
            )
            staged_source = open_zarr_root(plan.staged_source_path, mode="r")
            require_exact_bout_classification_run(staged_source)
            if (
                bout_classification_logical_manifest_sha256(staged_source)
                != plan.source_logical_manifest_sha256
            ):
                raise ValueError("staged bout-classification source differs")

        with telemetry.phase("logical_rematerialization"):
            pack, classified = reconstruct_bout_classification_writer_inputs(
                staged_source
            )
            local_root = zarr.open_group(
                str(plan.local_archive), mode="w-", zarr_format=3
            )
            write_megabouts_classification_run(
                local_root,
                run_name=plan.run_name,
                pack=pack,
                result=classified,
                storage_profile=get_storage_profile(profile_id),
                command="bout_classification_exact_result_direct_writer_replay_v1",
            )
            local_group = local_root[plan.run_path]
            local_group.attrs["storage_candidate_source_run"] = plan.source_run_name
            local_group.attrs["storage_candidate_source_run_path"] = (
                plan.source_run_path
            )
            local_group.attrs["storage_candidate_profile_promoted"] = False
            if execution_binding is not None:
                local_group.attrs[EXECUTION_BINDING_ATTR] = json_attr_safe(
                    dict(execution_binding)
                )

        with telemetry.phase("local_validation"):
            local_validation = _validate_candidate(
                local_group,
                source_group=staged_source,
            )
            if not local_validation["valid"]:
                raise ValueError(f"local candidate is invalid: {local_validation}")
        with telemetry.phase("local_consolidation"):
            consolidate_metadata_capture_expected_warnings(plan.local_archive)
        with telemetry.phase("local_direct_consolidated_comparison"):
            local_metadata = validate_direct_consolidated_subtree(
                plan.local_archive,
                subtree_path=plan.run_path,
            )
            if local_metadata.array_count < BOUT_CLASSIFICATION_ARRAY_COUNT:
                raise ValueError("local consolidated metadata omits arrays")

        authoritative_root = open_zarr_root(plan.archive, mode="r")
        authoritative_source = authoritative_root[plan.source_run_path]

        def validate(path: Path) -> dict[str, Any]:
            group = open_zarr_root(path, mode="r")
            return _validate_candidate(group, source_group=authoritative_source)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, "bout_classification_runs"),)

        def complete(
            _root: zarr.Group,
            _parent: zarr.Group,
            group: zarr.Group,
        ) -> None:
            mark_run_complete(
                group,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=group.attrs.get("run_provenance"),
            )
            group.attrs["stage_selector_eligible"] = False

        def verify(root: zarr.Group) -> None:
            parent = root[BOUT_CLASSIFICATION_RUN_PARENT]
            if (
                parent.attrs.get("latest") != plan.latest_before
                or parent.attrs.get("latest_complete") != plan.latest_complete_before
            ):
                raise ValueError("bout-classification publication changed selectors")
            group = parent[plan.run_name]
            if (
                group.attrs.get("stage_selector_eligible") is not False
                or group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            ):
                raise ValueError("published candidate lifecycle differs")

        acceptance: dict[str, Any] = {}

        def activate(
            root: zarr.Group,
            parent: zarr.Group,
            group: zarr.Group,
        ) -> None:
            with telemetry.phase("published_validation"):
                published = _validate_candidate(
                    group,
                    source_group=root[plan.source_run_path],
                )
                if not published["valid"]:
                    raise ValueError(f"published candidate is invalid: {published}")
                staged_validation = validate_staged_bout_classification_run(
                    root, plan.run_name, strict=True
                )
                if not staged_validation["ok"]:
                    raise ValueError(
                        f"published staged reader rejected candidate: {staged_validation}"
                    )
            consolidate_metadata_capture_expected_warnings(plan.archive)
            with telemetry.phase("published_direct_consolidated_comparison"):
                published_metadata = validate_direct_consolidated_subtree(
                    plan.archive,
                    subtree_path=plan.run_path,
                )
                if published_metadata.array_count < BOUT_CLASSIFICATION_ARRAY_COUNT:
                    raise ValueError("published consolidated metadata omits arrays")
            with telemetry.phase("decoded_equality"):
                if (
                    published["candidate_logical_manifest_sha256"]
                    != plan.source_logical_manifest_sha256
                ):
                    raise ValueError("published decoded values differ")
            with telemetry.phase("physical_inventory"):
                output_storage = storage_stats(plan.target_run_path)
                if (
                    output_storage["file_count"] < 1
                    or output_storage["apparent_bytes"] < 1
                ):
                    raise ValueError("published candidate has no physical payload")
            acceptance.update(
                published_validation=published,
                published_direct_consolidated_array_count=(
                    published_metadata.array_count
                ),
                output_storage=output_storage,
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    acceptance["caller_acceptance"] = json_attr_safe(
                        dict(publication_acceptance_validator(root, parent, group))
                    )

        def repair(_target: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.archive)

        with telemetry.phase("atomic_publication"):
            publication = atomic_publish_run_group(
                AtomicRunPublishSpec(
                    source_zarr=plan.archive,
                    local_run_path=plan.local_run_path,
                    target_run_path=plan.target_run_path,
                    run_name=plan.run_name,
                    lock_suffix="bout-classification-storage-candidate",
                    publish_schema_id=PUBLISH_SCHEMA_ID,
                    policy="bout_classification_atomic_nonpromoting_v1",
                    rollback_policy=(
                        "retain_failed_public_tombstone_leave_selectors_untouched"
                    ),
                ),
                copy_backend=copy_backend,
                validate_run=validate,
                prepare_parents=prepare,
                complete_run=complete,
                verify_pointers=verify,
                activate_run=activate,
                repair_failed_publication_visibility=repair,
                payload_metadata={
                    "profile_id": profile_id,
                    "source_run": plan.source_run_name,
                    "source_logical_manifest_sha256": (
                        plan.source_logical_manifest_sha256
                    ),
                    "local_direct_consolidated_array_count": (
                        local_metadata.array_count
                    ),
                },
            )
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_metadata.array_count,
            published_validation=acceptance["published_validation"],
            published_logical_manifest_sha256=acceptance["published_validation"][
                "candidate_logical_manifest_sha256"
            ],
            published_direct_consolidated_array_count=acceptance[
                "published_direct_consolidated_array_count"
            ],
            output_storage=acceptance["output_storage"],
            caller_acceptance=acceptance.get("caller_acceptance"),
            publication=publication,
            runtime_telemetry=_ordered_runtime(telemetry),
        )
        succeeded = True
        return json_attr_safe(result)
    except BaseException as exc:
        try:
            setattr(exc, "palette_runtime_telemetry", _ordered_runtime(telemetry))
        except BaseException:
            pass
        raise
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def tombstone_bout_classification_execution_candidate(
    archive: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one execution-owned, selector-ineligible public candidate."""

    source = Path(archive).expanduser().resolve()
    name = _safe_name(run_name, label="run_name")
    binding = json_attr_safe(dict(expected_execution_binding))
    payload = {
        "schema_id": "palette.analysis_candidate_execution_tombstone",
        "schema_version": 1,
        "execution_binding": binding,
        "failure_phase": failure_phase,
        "error_type": error_type,
        "error_message": error_message,
    }
    tombstone = {**payload, "payload_sha256": canonical_json_sha256(payload)}
    with archive_metadata_publication_lock(source):
        root = open_zarr_root(source, mode="a")
        parent = root[BOUT_CLASSIFICATION_RUN_PARENT]
        group = parent.get(name)
        if not isinstance(group, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        if group.attrs.get(EXECUTION_BINDING_ATTR) != binding:
            raise ValueError("refusing to tombstone another execution's candidate")
        if group.attrs.get("stage_selector_eligible") is not False:
            raise ValueError("refusing to tombstone selector-eligible candidate")
        if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED:
            mark_run_failed(
                group,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            group.attrs["stage_selector_eligible"] = False
            group.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        elif group.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR) != tombstone:
            raise ValueError("existing execution tombstone differs")
        consolidate_metadata_capture_expected_warnings(source)
    return {"target_present": True, "tombstoned": True}


__all__ = [
    "BOUT_CLASSIFICATION_EXECUTION_PHASE_ORDER",
    "EXECUTION_BINDING_ATTR",
    "EXECUTION_FAILURE_TOMBSTONE_ATTR",
    "build_bout_classification_candidate_plan",
    "materialize_bout_classification_candidate",
    "tombstone_bout_classification_execution_candidate",
]
