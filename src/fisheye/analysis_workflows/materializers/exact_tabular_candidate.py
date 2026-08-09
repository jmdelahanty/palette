"""Publish selector-ineligible byte-planned compact analysis candidates.

This finalization boundary currently supports the exact swim-bout v8 and
bout-kinematics v7 schemas.  It reads an explicit completed authority, writes a
complete candidate on node-local scratch through the shared byte planner,
validates exact decoded equality and local consolidated metadata, and then
uses the common atomic run-group publisher.  Parent selectors and production
profiles are never changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.bout_kinematics_schema import (
    build_bout_kinematics_array_declarations,
    validate_bout_kinematics_array_manifest,
    write_bout_kinematics_array_manifest,
)
from fisheye.analysis.detection_occupancy_schema import (
    build_occupancy_array_declarations,
    validate_occupancy_array_manifest,
    write_occupancy_array_manifest,
)
from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
    persist_exact_tabular_storage_receipt,
    rematerialize_exact_tabular_candidate,
    validate_exact_tabular_storage_receipt,
)
from fisheye.analysis.swim_bout_schema import (
    build_swim_bout_array_declarations,
    validate_swim_bout_array_manifest,
    write_swim_bout_array_manifest,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr.benchmark_runtime import storage_stats
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)

from fisheye.shared.atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from fisheye.shared.runtime_telemetry import PhaseTelemetry


MATERIALIZATION_SCHEMA_ID = "palette.exact_tabular_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.exact_tabular_candidate_publish.v1"
SUPPORTED_PROFILE_ID = "published_http_v1"
_NODE_LOCAL_SCRATCH_ROOTS = tuple(
    Path(value)
    for value in (
        "/tmp",
        "/var/tmp",
        "/scratch",
        "/dev/shm",
        "/local",
        "/lscratch",
    )
)
EXACT_TABULAR_EXECUTION_PHASE_ORDER = (
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
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


@dataclass(frozen=True)
class _Family:
    family_id: str
    parent_path: str
    validate_manifest: Callable[..., tuple[str, ...]]
    build_declarations: Callable[..., tuple[Any, ...]]
    write_manifest: Callable[..., Mapping[str, Any]]


_FAMILIES = {
    "swim_bouts": _Family(
        family_id="swim_bouts",
        parent_path="analysis/swim_bout_runs",
        validate_manifest=validate_swim_bout_array_manifest,
        build_declarations=build_swim_bout_array_declarations,
        write_manifest=write_swim_bout_array_manifest,
    ),
    "bout_kinematics": _Family(
        family_id="bout_kinematics",
        parent_path="analysis/bout_kinematics_runs",
        validate_manifest=validate_bout_kinematics_array_manifest,
        build_declarations=build_bout_kinematics_array_declarations,
        write_manifest=write_bout_kinematics_array_manifest,
    ),
    "detection_occupancy": _Family(
        family_id="detection_occupancy",
        parent_path="analysis/detection_occupancy_runs",
        validate_manifest=lambda group, **kwargs: validate_occupancy_array_manifest(
            group, session=False, **kwargs
        ),
        build_declarations=lambda group, **kwargs: build_occupancy_array_declarations(
            group, session=False, **kwargs
        ),
        write_manifest=lambda group, **kwargs: write_occupancy_array_manifest(
            group, session=False, **kwargs
        ),
    ),
    "session_occupancy": _Family(
        family_id="session_occupancy",
        parent_path="analysis/session_occupancy_runs",
        validate_manifest=lambda group, **kwargs: validate_occupancy_array_manifest(
            group, session=True, **kwargs
        ),
        build_declarations=lambda group, **kwargs: build_occupancy_array_declarations(
            group, session=True, **kwargs
        ),
        write_manifest=lambda group, **kwargs: write_occupancy_array_manifest(
            group, session=True, **kwargs
        ),
    ),
}


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", "..", "latest", "latest_complete"}:
        raise ValueError(f"{label} must be one explicit immutable run name.")
    if "/" in name or "\\" in name:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def _family(value: str) -> _Family:
    try:
        return _FAMILIES[str(value)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported exact compact family {value!r}; expected {sorted(_FAMILIES)!r}."
        ) from exc


def _require_node_local_scratch(path: Path) -> None:
    if not any(
        path == root or path.is_relative_to(root)
        for root in _NODE_LOCAL_SCRATCH_ROOTS
    ):
        raise ValueError(
            "Scratch root must be below a recognized node-local filesystem."
        )


def _require_symlink_free_tree(path: Path, *, label: str) -> None:
    links: list[str] = []
    if path.is_symlink():
        links.append(str(path))
    for root, directories, filenames in os.walk(path, followlinks=False):
        for name in (*directories, *filenames):
            candidate = Path(root) / name
            if candidate.is_symlink():
                links.append(str(candidate))
                if len(links) >= 8:
                    break
        if len(links) >= 8:
            break
    if links:
        raise ValueError(
            f"{label} must be self-contained and symlink-free; found {links!r}."
        )


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {
        name: index for index, name in enumerate(EXACT_TABULAR_EXECUTION_PHASE_ORDER)
    }
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Exact-tabular telemetry contains an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


@dataclass(frozen=True)
class ExactTabularCandidatePlan:
    source_zarr: Path
    family_id: str
    parent_path: str
    source_run_name: str
    run_name: str
    scratch_root: Path
    local_zarr: Path
    profile_id: str
    latest_before: str | None
    latest_complete_before: str | None

    @property
    def source_run_path(self) -> str:
        return f"{self.parent_path}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{self.parent_path}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "family_id": self.family_id,
            "parent_path": self.parent_path,
            "source_run_name": self.source_run_name,
            "source_run_path": self.source_run_path,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "profile_id": self.profile_id,
            "latest_before": self.latest_before,
            "latest_complete_before": self.latest_complete_before,
            "publication_policy": (
                "atomic_named_candidate_selector_ineligible_no_pointer_update"
            ),
        }


def build_exact_tabular_candidate_plan(
    source_zarr: str | Path,
    *,
    family_id: str,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
) -> ExactTabularCandidatePlan:
    """Validate source identity and build a read-only candidate plan."""

    family = _family(family_id)
    if profile_id != SUPPORTED_PROFILE_ID:
        raise ValueError(
            f"Exact compact candidates require profile {SUPPORTED_PROFILE_ID!r}."
        )
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}.")
    _require_node_local_scratch(scratch)
    if scratch == source or scratch.is_relative_to(source):
        raise ValueError("Scratch root must be outside the authoritative archive.")
    source_name = _safe_name(source_run, label="source run")
    target_name = _safe_name(run_name, label="candidate run")
    if source_name == target_name:
        raise ValueError("Source and candidate run names must differ.")

    root = open_zarr_root(source, mode="r")
    parent = root.get(family.parent_path)
    if not isinstance(parent, zarr.Group):
        raise KeyError(f"Missing exact compact parent {family.parent_path!r}.")
    source_group = parent.get(source_name)
    if not isinstance(source_group, zarr.Group):
        raise KeyError(f"Source run {source_name!r} does not exist.")
    if source_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError("Source exact compact run is not complete.")
    if source_group.attrs.get("stage_selector_eligible") is not True:
        raise ValueError("Source exact compact run is not selector eligible.")
    manifest_errors = family.validate_manifest(source_group)
    if manifest_errors:
        raise ValueError(
            "Source exact compact manifest is invalid: " + "; ".join(manifest_errors)
        )
    if target_name in parent or source.joinpath(
        *family.parent_path.split("/"), target_name
    ).exists():
        raise FileExistsError(f"Candidate run {target_name!r} already exists.")
    return ExactTabularCandidatePlan(
        source_zarr=source,
        family_id=family.family_id,
        parent_path=family.parent_path,
        source_run_name=source_name,
        run_name=target_name,
        scratch_root=scratch,
        local_zarr=scratch / "exact-tabular-candidate.zarr",
        profile_id=profile_id,
        latest_before=(
            None if parent.attrs.get("latest") is None else str(parent.attrs["latest"])
        ),
        latest_complete_before=(
            None
            if parent.attrs.get("latest_complete") is None
            else str(parent.attrs["latest_complete"])
        ),
    )


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def compute_exact_tabular_logical_hashes(
    group: Any,
    declarations: Sequence[Any],
) -> dict[str, str]:
    """Hash every declared decoded array in one deterministic logical order."""

    hashes: dict[str, str] = {}
    for declaration in declarations:
        array = _array_at_path(group, declaration.path)
        digest = hashlib.sha256()
        digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        if array.ndim == 0:
            digest.update(np.ascontiguousarray(array[...]).tobytes(order="C"))
        else:
            block_rows = max(1, min(int(array.shape[0]) or 1, 65_536))
            for start in range(0, int(array.shape[0]), block_rows):
                values = np.ascontiguousarray(array[start : start + block_rows])
                digest.update(values.tobytes(order="C"))
        hashes[declaration.path] = digest.hexdigest()
    return hashes


def _consolidate_archive_metadata(path: Path) -> None:
    consolidate_metadata_capture_expected_warnings(path)


def _direct_consolidated_check(
    local_zarr: Path,
    *,
    run_path: str,
    declarations: Sequence[Any],
) -> int:
    receipt = validate_direct_consolidated_subtree(
        local_zarr,
        subtree_path=run_path,
    )
    if receipt.array_count < len(declarations):
        raise ValueError(
            "Direct/consolidated candidate subtree omits declared arrays: "
            f"expected at least {len(declarations)}, got {receipt.array_count}."
        )
    # The execution receipt's equality surface is the exact declared schema.
    # Explicitly non-authoritative report/visualization arrays may coexist in
    # the subtree and are still checked by the full metadata comparison, but
    # they do not increase the logical compared-array count.
    return len(declarations)


def _local_direct_consolidated_check(
    local_zarr: Path,
    *,
    run_path: str,
    declarations: Sequence[Any],
) -> int:
    """Compatibility wrapper for callers that require both operations."""

    _consolidate_archive_metadata(local_zarr)
    return _direct_consolidated_check(
        local_zarr,
        run_path=run_path,
        declarations=declarations,
    )


def _copy_source_run_to_scratch(
    source: Path,
    target: Path,
    *,
    backend: str,
) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing existing staged source: {target}.")
    _require_symlink_free_tree(source, label="Authoritative source run")
    target.parent.mkdir(parents=True, exist_ok=True)
    if backend == "python":
        shutil.copytree(source, target, symlinks=False)
        _require_symlink_free_tree(target, label="Staged source run")
        return
    if backend != "rsync":
        raise ValueError(f"Unsupported source-staging backend: {backend!r}.")
    target.mkdir()
    subprocess.run(
        ["rsync", "-aL", "--", f"{source}/", f"{target}/"],
        check=True,
    )
    _require_symlink_free_tree(target, label="Staged source run")


def _validate_candidate_group(
    group: Any,
    *,
    family: _Family,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    errors = list(
        family.validate_manifest(group, byte_planner_adopted=True)
    )
    declarations = family.build_declarations(
        group,
        byte_planner_adopted=True,
    )
    errors.extend(
        validate_exact_tabular_storage_receipt(
            group,
            declarations=declarations,
        )
    )
    if group.attrs.get("stage_selector_eligible") is not False:
        errors.append("candidate is not selector-ineligible")
    if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("candidate is not complete")
    hashes = compute_exact_tabular_logical_hashes(group, declarations)
    if dict(hashes) != dict(expected_hashes):
        errors.append("candidate decoded logical hashes differ from source")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": hashes,
    }


def tombstone_exact_tabular_execution_candidate(
    source_zarr: str | Path,
    *,
    family_id: str,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one exact owned benchmark candidate after runner finalization fails."""

    family = _family(family_id)
    archive = Path(source_zarr).expanduser().resolve()
    name = _safe_name(run_name, label="candidate run")
    expected_binding = json_attr_safe(dict(expected_execution_binding))
    if not expected_binding:
        raise ValueError("expected_execution_binding must be nonempty.")
    tombstone_payload = {
        "schema_id": "palette.analysis_candidate_execution_tombstone",
        "schema_version": 1,
        "execution_binding": expected_binding,
        "failure_phase": str(failure_phase),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }
    tombstone = {
        **tombstone_payload,
        "payload_sha256": canonical_json_sha256(tombstone_payload),
    }
    run_path = f"{family.parent_path}/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root[family.parent_path]
        run = parent.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        attrs = run.attrs
        if attrs.get(EXECUTION_BINDING_ATTR) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone a candidate owned by another execution."
            )
        if attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Refusing to tombstone a selector-eligible run.")
        existing = attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        if attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError("Existing execution tombstone differs.")
        else:
            if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Execution candidate is neither complete nor already failed."
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
            raise RuntimeError("Execution failure tombstone did not persist exactly.")
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": receipt.declarations_sha256,
    }


def materialize_exact_tabular_candidate(
    source_zarr: str | Path,
    *,
    family_id: str,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    stage_source_to_scratch: bool = False,
    execution_binding: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Create and optionally atomically publish one named physical candidate."""

    telemetry = PhaseTelemetry(
        materializer="exact_tabular_candidate",
        context={
            "family_id": family_id,
            "source_run": source_run,
            "run_name": run_name,
            "stage_source_to_scratch": bool(stage_source_to_scratch),
        },
    )
    with telemetry.phase("plan"):
        plan = build_exact_tabular_candidate_plan(
            source_zarr,
            family_id=family_id,
            source_run=source_run,
            run_name=run_name,
            scratch_root=scratch_root,
            profile_id=profile_id,
        )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.as_dict(),
    }
    if not apply:
        result["runtime_telemetry"] = _ordered_runtime_telemetry(telemetry)
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}.")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    family = _family(plan.family_id)
    try:
        with telemetry.phase("source_staging"):
            source_root = open_zarr_root(plan.source_zarr, mode="r")
            authoritative_source_group = source_root[plan.source_run_path]
            source_declarations = family.build_declarations(
                authoritative_source_group,
                byte_planner_adopted=False,
            )
            source_hashes = compute_exact_tabular_logical_hashes(
                authoritative_source_group,
                source_declarations,
            )
            if stage_source_to_scratch:
                staged_source_path = plan.scratch_root / "staged-source-run"
                authoritative_source_path = plan.source_zarr.joinpath(
                    *plan.source_run_path.split("/")
                )
                _copy_source_run_to_scratch(
                    authoritative_source_path,
                    staged_source_path,
                    backend=copy_backend,
                )
                source_group = open_zarr_root(staged_source_path, mode="r")
                staged_errors = family.validate_manifest(source_group)
                if staged_errors:
                    raise ValueError(
                        "Staged exact compact source is invalid: "
                        + "; ".join(staged_errors)
                    )
                staged_declarations = family.build_declarations(
                    source_group,
                    byte_planner_adopted=False,
                )
                if (
                    compute_exact_tabular_logical_hashes(
                        source_group,
                        staged_declarations,
                    )
                    != source_hashes
                ):
                    raise ValueError(
                        "Staged exact compact source differs from authoritative source."
                    )
            else:
                source_group = authoritative_source_group
            candidate_declarations = family.build_declarations(
                source_group,
                byte_planner_adopted=True,
            )
            receipt = build_exact_tabular_storage_receipt(
                source_group,
                declarations=candidate_declarations,
                profile=get_storage_profile(plan.profile_id),
            )

        with telemetry.phase("logical_rematerialization"):
            local_root = zarr.open_group(
                str(plan.local_zarr), mode="w-", zarr_format=3
            )
            local_parent = local_root
            for component in plan.parent_path.split("/"):
                local_parent = local_parent.require_group(component)
            local_group = local_parent.create_group(plan.run_name)
            rematerialize_exact_tabular_candidate(
                source_group,
                local_group,
                receipt=receipt,
            )
            local_group.attrs[RUN_NAME_ATTR] = plan.run_name
            local_group.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_RUNNING
            local_group.attrs.pop(RUN_COMPLETED_AT_ATTR, None)
            local_group.attrs["stage_selector_eligible"] = False
            local_group.attrs["storage_candidate_source_run"] = plan.source_run_name
            local_group.attrs["storage_candidate_source_run_path"] = plan.source_run_path
            if execution_binding is not None:
                binding = json_attr_safe(dict(execution_binding))
                if not binding:
                    raise ValueError("execution_binding must be one nonempty mapping.")
                local_group.attrs[EXECUTION_BINDING_ATTR] = binding
            local_group.attrs["storage_candidate_profile_promoted"] = False
            family.write_manifest(local_group, byte_planner_adopted=True)
            persist_exact_tabular_storage_receipt(local_group, receipt)
            mark_run_complete(
                local_group,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=build_run_provenance_from_stage_record(
                    source_group.attrs.get("provenance", {}),
                    fallback_command="exact_tabular_candidate_materializer",
                ),
            )
        with telemetry.phase("local_validation"):
            local_validation = _validate_candidate_group(
                local_group,
                family=family,
                expected_hashes=source_hashes,
            )
            if not local_validation["valid"]:
                raise RuntimeError(
                    "Local exact compact candidate is invalid: "
                    f"{local_validation}."
                )
        with telemetry.phase("local_consolidation"):
            _consolidate_archive_metadata(plan.local_zarr)
        with telemetry.phase("local_direct_consolidated_comparison"):
            compared = _direct_consolidated_check(
                plan.local_zarr,
                run_path=plan.run_path,
                declarations=candidate_declarations,
            )
        materialization_seconds = float(
            sum(
                telemetry.duration_seconds(name) or 0.0
                for name in (
                    "logical_rematerialization",
                    "local_validation",
                    "local_consolidation",
                    "local_direct_consolidated_comparison",
                )
            )
        )

        def validate(path: Path) -> dict[str, Any]:
            return _validate_candidate_group(
                open_zarr_root(path, mode="r"),
                family=family,
                expected_hashes=source_hashes,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (
                require_runs_parent(
                    analysis,
                    plan.parent_path.split("/", 1)[1],
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
            run_group.attrs["stage_selector_eligible"] = False

        def verify(root: zarr.Group) -> None:
            parent = root[plan.parent_path]
            if (
                parent.attrs.get("latest") != plan.latest_before
                or parent.attrs.get("latest_complete")
                != plan.latest_complete_before
            ):
                raise RuntimeError("Exact compact candidate changed parent selectors.")
            candidate = parent.get(plan.run_name)
            if not isinstance(candidate, zarr.Group):
                raise RuntimeError("Published exact compact candidate is missing.")
            if (
                candidate.attrs.get("stage_selector_eligible") is not False
                or candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Published exact compact candidate is not complete and ineligible."
                )

        publication_acceptance: dict[str, Any] = {}

        def consolidate_archive(
            _root: zarr.Group,
            _parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            if (
                run_group.attrs.get("stage_selector_eligible") is not False
                or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Exact compact candidate lost its complete ineligible state "
                    "before metadata consolidation."
                )
            with telemetry.phase("published_validation"):
                published_validation = _validate_candidate_group(
                    run_group,
                    family=family,
                    expected_hashes=source_hashes,
                )
                if not published_validation["valid"]:
                    raise RuntimeError(
                        "Published exact compact candidate is invalid: "
                        f"{published_validation}."
                    )
            _consolidate_archive_metadata(plan.source_zarr)
            with telemetry.phase("published_direct_consolidated_comparison"):
                published_compared = _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declarations=candidate_declarations,
                )
            with telemetry.phase("decoded_equality"):
                published_hashes = compute_exact_tabular_logical_hashes(
                    run_group,
                    candidate_declarations,
                )
                if published_hashes != source_hashes:
                    raise RuntimeError(
                        "Published exact compact decoded values differ from source."
                    )
            with telemetry.phase("physical_inventory"):
                published_storage = storage_stats(plan.target_run_path)
                if (
                    published_storage["file_count"] < 1
                    or published_storage["apparent_bytes"] < 1
                    or published_storage["allocated_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published exact compact candidate has no physical payload."
                    )
            publication_acceptance.update(
                archive_direct_consolidated_array_count=published_compared,
                published_validation=published_validation,
                published_direct_consolidated_array_count=published_compared,
                published_hashes=published_hashes,
                output_storage=published_storage,
            )
            if publication_acceptance_validator is not None:
                with telemetry.phase("publication_acceptance_validation"):
                    caller_acceptance = dict(
                        publication_acceptance_validator(_root, _parent, run_group)
                    )
                publication_acceptance["caller_acceptance"] = json_attr_safe(
                    caller_acceptance
                )

        def repair_failed_visibility(_target_path: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)

        with telemetry.phase("atomic_publication"):
            publication = atomic_publish_run_group(
                AtomicRunPublishSpec(
                    source_zarr=plan.source_zarr,
                    local_run_path=plan.local_run_path,
                    target_run_path=plan.target_run_path,
                    run_name=plan.run_name,
                    lock_suffix=f"{plan.family_id}-storage-candidate",
                    publish_schema_id=PUBLISH_SCHEMA_ID,
                    policy="exact_tabular_byte_planned_atomic_nonpromoting_publish",
                    rollback_policy=(
                        "retain_failed_public_tombstone_leave_parent_selectors_untouched"
                    ),
                ),
                copy_backend=copy_backend,
                validate_run=validate,
                prepare_parents=prepare,
                complete_run=complete,
                verify_pointers=verify,
                activate_run=consolidate_archive,
                repair_failed_publication_visibility=repair_failed_visibility,
                payload_metadata={
                    "profile_id": plan.profile_id,
                    "source_run": plan.source_run_name,
                    "source_logical_hashes": source_hashes,
                    "local_direct_consolidated_array_count": compared,
                    "materialization_seconds": materialization_seconds,
                },
            )
        published_hashes = publication_acceptance["published_hashes"]
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=compared,
            archive_direct_consolidated_array_count=publication_acceptance[
                "archive_direct_consolidated_array_count"
            ],
            published_validation=publication_acceptance["published_validation"],
            published_direct_consolidated_array_count=publication_acceptance[
                "published_direct_consolidated_array_count"
            ],
            source_logical_manifest_sha256=canonical_json_sha256(source_hashes),
            published_logical_manifest_sha256=canonical_json_sha256(
                published_hashes
            ),
            output_storage=publication_acceptance["output_storage"],
            caller_acceptance=publication_acceptance.get("caller_acceptance"),
            materialization_seconds=materialization_seconds,
            publication=publication,
            runtime_telemetry=_ordered_runtime_telemetry(telemetry),
        )
        succeeded = True
        return json_attr_safe(result)
    except BaseException as exc:
        try:
            setattr(
                exc,
                "palette_runtime_telemetry",
                _ordered_runtime_telemetry(telemetry),
            )
        except BaseException:
            pass
        raise
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_exact_tabular_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_exact_tabular_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--family", choices=tuple(sorted(_FAMILIES)), required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--profile", choices=(SUPPORTED_PROFILE_ID,), default=SUPPORTED_PROFILE_ID)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--stage-source-to-scratch",
        action="store_true",
        help=(
            "Copy and verify the exact source run on node-local scratch before "
            "rematerialization."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_exact_tabular_candidate(
        args.zarr_path,
        family_id=args.family,
        source_run=args.source_run,
        run_name=args.run_name,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        profile_id=args.profile,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
        stage_source_to_scratch=args.stage_source_to_scratch,
    )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
