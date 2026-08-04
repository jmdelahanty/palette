"""Publish one selector-ineligible byte-planned track-kinematics v2 candidate."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence

import zarr

from fisheye.analysis.track_kinematics_storage import (
    TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
    flat_candidate_logical_hashes,
    persist_flat_candidate_contract,
    rematerialize_flat_candidate,
    source_flat_projection_hashes,
    validate_flat_candidate,
)
from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
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
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    mark_run_failed,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from .runtime_telemetry import PhaseTelemetry

MATERIALIZATION_SCHEMA_ID = "palette.track_kinematics_flat_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.track_kinematics_flat_candidate_publish.v1"
SUPPORTED_PROFILE_ID = "published_http_v1"
PARENT_PATH = "analysis/track_kinematics_runs"
RUN_TYPE = "offline"
TRACK_FLAT_EXECUTION_PHASE_ORDER = (
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
)
EXECUTION_BINDING_ATTR = "analysis_candidate_execution_binding"
EXECUTION_FAILURE_TOMBSTONE_ATTR = "analysis_candidate_execution_tombstone"
PublicationAcceptanceValidator = Callable[
    [zarr.Group, zarr.Group, zarr.Group], Mapping[str, Any]
]


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", "..", "latest", "latest_complete"}:
        raise ValueError(f"{label} must be one explicit immutable run name.")
    if "/" in name or "\\" in name:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def _published_root(path: Path) -> zarr.Group:
    """Open an immutable source through its published consolidated generation."""

    return zarr.open_group(
        str(path),
        mode="r",
        zarr_format=3,
        use_consolidated=True,
    )


def _direct_consolidated_check(
    archive: Path,
    *,
    run_path: str,
    declaration_paths: Sequence[str],
) -> int:
    receipt = validate_direct_consolidated_subtree(
        archive,
        subtree_path=run_path,
    )
    if receipt.array_count != len(declaration_paths):
        raise ValueError(
            "Direct/consolidated track candidate array inventory differs: "
            f"expected {len(declaration_paths)}, got {receipt.array_count}."
        )
    return receipt.array_count


def _ordered_runtime_telemetry(telemetry: PhaseTelemetry) -> dict[str, Any]:
    result = telemetry.to_json()
    order = {name: index for index, name in enumerate(TRACK_FLAT_EXECUTION_PHASE_ORDER)}
    phases = list(result["phases"])
    if any(phase["name"] not in order for phase in phases):
        raise RuntimeError("Track-flat telemetry contains an unknown phase.")
    phases.sort(key=lambda phase: order[phase["name"]])
    result["phases"] = phases
    return result


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


def _copy_source_run_to_scratch(
    source: Path,
    target: Path,
    *,
    backend: str,
) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing existing staged source: {target}.")
    _require_symlink_free_tree(source, label="Authoritative track source run")
    target.parent.mkdir(parents=True, exist_ok=True)
    if backend == "python":
        shutil.copytree(source, target, symlinks=False)
        _require_symlink_free_tree(target, label="Staged track source run")
        return
    if backend != "rsync":
        raise ValueError(f"Unsupported source-staging backend: {backend!r}.")
    target.mkdir()
    subprocess.run(
        ["rsync", "-aL", "--", f"{source}/", f"{target}/"],
        check=True,
    )
    _require_symlink_free_tree(target, label="Staged track source run")


@dataclass(frozen=True)
class TrackKinematicsFlatCandidatePlan:
    source_zarr: Path
    source_run_name: str
    run_name: str
    scratch_root: Path
    local_zarr: Path
    profile_id: str
    parent_pointers_before: tuple[tuple[str, object], ...]
    offline_pointers_before: tuple[tuple[str, object], ...]

    @property
    def source_run_path(self) -> str:
        return f"{PARENT_PATH}/{RUN_TYPE}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{RUN_TYPE}/{self.run_name}"

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
            "source_run_name": self.source_run_name,
            "source_run_path": self.source_run_path,
            "run_name": self.run_name,
            "run_path": self.run_path,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "profile_id": self.profile_id,
            "parent_pointers_before": dict(self.parent_pointers_before),
            "offline_pointers_before": dict(self.offline_pointers_before),
            "publication_policy": (
                "atomic_named_flat_lineage_candidate_selector_ineligible_no_pointer_update"
            ),
        }


def _pointer_snapshot(
    group: Any, names: Sequence[str]
) -> tuple[tuple[str, object], ...]:
    return tuple((name, group.attrs.get(name)) for name in names)


def build_track_kinematics_flat_candidate_plan(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
) -> TrackKinematicsFlatCandidatePlan:
    """Build a read-only candidate plan from one published v1 authority."""

    if profile_id != SUPPORTED_PROFILE_ID:
        raise ValueError(
            f"Track flat candidates require profile {SUPPORTED_PROFILE_ID!r}."
        )
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}.")
    if (
        scratch == source
        or scratch.is_relative_to(source)
        or source.is_relative_to(scratch)
    ):
        raise ValueError(
            "Scratch and authoritative archive must not contain one another."
        )
    source_name = _safe_name(source_run, label="source run")
    candidate_name = _safe_name(run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")

    root = _published_root(source)
    parent = root.get(PARENT_PATH)
    offline = root.get(f"{PARENT_PATH}/{RUN_TYPE}")
    if not isinstance(parent, zarr.Group) or not isinstance(offline, zarr.Group):
        raise KeyError("Published track-kinematics offline parent is missing.")
    source_group = offline.get(source_name)
    if not isinstance(source_group, zarr.Group):
        raise KeyError(f"Published source run {source_name!r} is missing.")
    if (
        source_group.attrs.get("schema_id") != "analysis.track_kinematics_runs"
        or source_group.attrs.get("schema_version") != 1
    ):
        raise ValueError("Source run is not the exact maintained v1 track authority.")
    if source_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError("Source track run is not complete.")
    if source_group.attrs.get("stage_selector_eligible") is not True:
        raise ValueError("Source track run is not selector-eligible authority.")
    build_flat_candidate_declarations(source_group)
    if (
        candidate_name in offline
        or source.joinpath(*PARENT_PATH.split("/"), RUN_TYPE, candidate_name).exists()
    ):
        raise FileExistsError(f"Candidate run {candidate_name!r} already exists.")
    return TrackKinematicsFlatCandidatePlan(
        source_zarr=source,
        source_run_name=source_name,
        run_name=candidate_name,
        scratch_root=scratch,
        local_zarr=scratch / "track-flat-candidate.zarr",
        profile_id=profile_id,
        parent_pointers_before=_pointer_snapshot(
            parent, ("latest", "latest_complete", "latest_offline")
        ),
        offline_pointers_before=_pointer_snapshot(
            offline, ("latest", "latest_complete")
        ),
    )


def tombstone_track_kinematics_execution_candidate(
    source_zarr: str | Path,
    *,
    run_name: str,
    expected_execution_binding: Mapping[str, Any],
    failure_phase: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Fail one complete benchmark candidate owned by the named execution."""

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
    run_path = f"{PARENT_PATH}/{RUN_TYPE}/{name}"
    with archive_metadata_publication_lock(archive):
        root = open_zarr_root(archive, mode="a")
        parent = root[PARENT_PATH]
        offline = parent[RUN_TYPE]
        parent_before = _pointer_snapshot(
            parent, ("latest", "latest_complete", "latest_offline")
        )
        offline_before = _pointer_snapshot(offline, ("latest", "latest_complete"))
        run = offline.get(name)
        if not isinstance(run, zarr.Group):
            return {"target_present": False, "tombstoned": False}
        attrs = run.attrs
        if attrs.get(EXECUTION_BINDING_ATTR) != expected_binding:
            raise RuntimeError(
                "Refusing to tombstone a track candidate owned by another execution."
            )
        if attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError("Refusing to tombstone a selector-eligible track run.")
        if attrs.get("storage_candidate_profile_promoted") is not False:
            raise RuntimeError("Refusing to tombstone a promoted track profile.")
        existing = attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR)
        if attrs.get(RUN_COMPLETION_STATUS_ATTR) == RUN_STATUS_FAILED:
            if existing != tombstone:
                raise RuntimeError("Existing track execution tombstone differs.")
        else:
            if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                raise RuntimeError(
                    "Track execution candidate is neither complete nor failed."
                )
            mark_run_failed(
                run,
                parent_group=None,
                run_name=name,
                error=f"{error_type}: {error_message}",
            )
            run.attrs["stage_selector_eligible"] = False
            run.attrs["storage_candidate_profile_promoted"] = False
            run.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR] = tombstone
        if (
            _pointer_snapshot(parent, ("latest", "latest_complete", "latest_offline"))
            != parent_before
            or _pointer_snapshot(offline, ("latest", "latest_complete"))
            != offline_before
        ):
            raise RuntimeError("Track execution tombstone changed selector state.")
        consolidate_metadata_capture_expected_warnings(archive)
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        direct = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )[run_path]
        consolidated = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=True
        )[run_path]
        for fresh in (direct, consolidated):
            if (
                fresh.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_FAILED
                or fresh.attrs.get("stage_selector_eligible") is not False
                or fresh.attrs.get("storage_candidate_profile_promoted") is not False
                or fresh.attrs.get(EXECUTION_BINDING_ATTR) != expected_binding
                or fresh.attrs.get(EXECUTION_FAILURE_TOMBSTONE_ATTR) != tombstone
            ):
                raise RuntimeError(
                    "Track execution failure tombstone did not persist exactly."
                )
    return {
        "target_present": True,
        "tombstoned": True,
        "metadata_declarations_sha256": receipt.declarations_sha256,
    }


def materialize_track_kinematics_flat_candidate(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
    stage_source_to_scratch: bool = False,
    exclude_physical_bundle: bool = False,
    execution_binding: Mapping[str, Any] | None = None,
    publication_acceptance_validator: PublicationAcceptanceValidator | None = None,
) -> dict[str, Any]:
    """Rematerialize and atomically publish one non-promoting v2 candidate."""

    telemetry = PhaseTelemetry(
        materializer="track_kinematics_flat_candidate",
        context={
            "source_run": source_run,
            "run_name": run_name,
            "stage_source_to_scratch": bool(stage_source_to_scratch),
            "exclude_physical_bundle": bool(exclude_physical_bundle),
        },
    )
    with telemetry.phase("plan"):
        plan = build_track_kinematics_flat_candidate_plan(
            source_zarr,
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
    succeeded = False
    try:
        if plan.scratch_root.exists():
            raise FileExistsError(
                f"Refusing existing scratch root: {plan.scratch_root}."
            )
        plan.scratch_root.mkdir(parents=True)
        with telemetry.phase("source_staging"):
            source_root = _published_root(plan.source_zarr)
            authoritative_source_group = source_root[plan.source_run_path]
            authoritative_declarations = build_flat_candidate_declarations(
                authoritative_source_group
            )
            if exclude_physical_bundle and any(
                declaration.path.endswith("/positions_mm")
                for declaration in authoritative_declarations
            ):
                raise ValueError(
                    "track_flat_v1 explicitly excludes the physical track bundle."
                )
            source_hashes = source_flat_projection_hashes(
                authoritative_source_group,
                authoritative_declarations,
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
                source_group = zarr.open_group(
                    str(staged_source_path),
                    mode="r",
                    zarr_format=3,
                    use_consolidated=False,
                )
                declarations = build_flat_candidate_declarations(source_group)
                staged_hashes = source_flat_projection_hashes(
                    source_group,
                    declarations,
                )
                if [item.as_manifest() for item in declarations] != [
                    item.as_manifest() for item in authoritative_declarations
                ] or staged_hashes != source_hashes:
                    raise ValueError(
                        "Staged track source differs from the authoritative projection."
                    )
            else:
                source_group = authoritative_source_group
                declarations = authoritative_declarations
            paths = tuple(declaration.path for declaration in declarations)
            receipt = build_flat_candidate_storage_receipt(
                source_group,
                profile=get_storage_profile(plan.profile_id),
            )

        with telemetry.phase("logical_rematerialization"):
            local_root = zarr.open_group(
                str(plan.local_zarr),
                mode="w-",
                zarr_format=3,
                use_consolidated=False,
            )
            local_parent = local_root
            for component in (PARENT_PATH + "/" + RUN_TYPE).split("/"):
                local_parent = local_parent.require_group(component)
            local_group = local_parent.create_group(plan.run_name)
            rematerialize_flat_candidate(
                source_group,
                local_group,
                receipt=receipt,
            )
            local_group.attrs.update(
                {
                    "schema_id": TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
                    "schema_version": TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
                    "method": (
                        "track_kinematics_v1_exact_flat_lineage_rematerialization"
                    ),
                    "method_version": "track_kinematics.flat_lineage_candidate.v2",
                    RUN_NAME_ATTR: plan.run_name,
                    RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
                    "stage_selector_eligible": False,
                    "storage_candidate_profile_promoted": False,
                    "storage_candidate_source_run": plan.source_run_name,
                    "storage_candidate_source_run_path": plan.source_run_path,
                    "legacy_compatibility_policy": (
                        "v1_structured_source_explicit_only_no_dtype_probe"
                    ),
                    "physical_bundle_mode": (
                        "excluded_from_flat_candidate_v1"
                        if exclude_physical_bundle
                        else "source_layout_preserved"
                    ),
                }
            )
            if execution_binding is not None:
                binding = json_attr_safe(dict(execution_binding))
                if not binding:
                    raise ValueError("execution_binding must be one nonempty mapping.")
                local_group.attrs[EXECUTION_BINDING_ATTR] = binding
            local_group.attrs.pop(RUN_COMPLETED_AT_ATTR, None)
            persist_flat_candidate_contract(
                local_group,
                receipt=receipt,
                declarations=declarations,
                source_run_path=plan.source_run_path,
                source_projection_hashes=source_hashes,
            )
            mark_run_complete(
                local_group,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=build_run_provenance_from_stage_record(
                    source_group.attrs.get("provenance", {}),
                    fallback_command="track_kinematics_flat_candidate_materializer",
                ),
            )
        with telemetry.phase("local_validation"):
            local_validation = validate_flat_candidate(
                local_group,
                source_group=source_group,
            )
            if not local_validation["valid"]:
                raise RuntimeError(
                    f"Local track flat candidate is invalid: {local_validation}."
                )
        with telemetry.phase("local_consolidation"):
            consolidate_metadata_capture_expected_warnings(plan.local_zarr)
        with telemetry.phase("local_direct_consolidated_comparison"):
            local_compared = _direct_consolidated_check(
                plan.local_zarr,
                run_path=plan.run_path,
                declaration_paths=paths,
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
            group = zarr.open_group(
                str(path), mode="r", zarr_format=3, use_consolidated=False
            )
            authoritative = open_zarr_root(plan.source_zarr, mode="r")
            return validate_flat_candidate(
                group,
                source_group=authoritative[plan.source_run_path],
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group, zarr.Group]:
            parent = root[PARENT_PATH]
            offline = parent[RUN_TYPE]
            return parent, offline

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
            run_group.attrs["storage_candidate_profile_promoted"] = False

        def verify(root: zarr.Group) -> None:
            parent = root[PARENT_PATH]
            offline = parent[RUN_TYPE]
            if (
                _pointer_snapshot(
                    parent, ("latest", "latest_complete", "latest_offline")
                )
                != plan.parent_pointers_before
                or _pointer_snapshot(offline, ("latest", "latest_complete"))
                != plan.offline_pointers_before
            ):
                raise RuntimeError("Track flat candidate changed parent selectors.")
            candidate = offline.get(plan.run_name)
            if not isinstance(candidate, zarr.Group):
                raise RuntimeError("Published track flat candidate is missing.")
            if (
                candidate.attrs.get("stage_selector_eligible") is not False
                or candidate.attrs.get("storage_candidate_profile_promoted")
                is not False
                or candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Published track flat candidate is not complete and ineligible."
                )

        publication_acceptance: dict[str, Any] = {}

        def consolidate_archive(
            root: zarr.Group,
            _parent: zarr.Group,
            run_group: zarr.Group,
        ) -> None:
            if (
                run_group.attrs.get("stage_selector_eligible") is not False
                or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Track flat candidate lost its complete ineligible state."
                )
            authoritative = root[plan.source_run_path]
            with telemetry.phase("published_validation"):
                published_validation = validate_flat_candidate(
                    run_group,
                    source_group=authoritative,
                )
                if not published_validation["valid"]:
                    raise RuntimeError(
                        "Published track flat candidate is invalid: "
                        f"{published_validation}."
                    )
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            with telemetry.phase("published_direct_consolidated_comparison"):
                published_compared = _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declaration_paths=paths,
                )
            with telemetry.phase("decoded_equality"):
                published_hashes = flat_candidate_logical_hashes(
                    run_group,
                    declarations,
                )
                if published_hashes != source_hashes:
                    raise RuntimeError(
                        "Published track flat decoded values differ from source."
                    )
            with telemetry.phase("physical_inventory"):
                published_storage = storage_stats(plan.target_run_path)
                if (
                    published_storage["file_count"] < 1
                    or published_storage["apparent_bytes"] < 1
                    or published_storage["allocated_bytes"] < 1
                ):
                    raise RuntimeError(
                        "Published track flat candidate has no physical payload."
                    )
            publication_acceptance.update(
                archive_direct_consolidated_array_count=published_compared,
                published_validation=published_validation,
                published_direct_consolidated_array_count=published_compared,
                published_hashes=published_hashes,
                output_storage=published_storage,
            )
            if publication_acceptance_validator is not None:
                caller_acceptance = dict(
                    publication_acceptance_validator(root, _parent, run_group)
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
                    lock_suffix="track-flat-lineage-storage-candidate",
                    publish_schema_id=PUBLISH_SCHEMA_ID,
                    policy=(
                        "track_flat_lineage_byte_planned_atomic_nonpromoting_publish"
                    ),
                    rollback_policy=(
                        "retain_failed_public_tombstone_leave_track_selectors_untouched"
                    ),
                    content_checksum=True,
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
                    "source_projection_hashes": source_hashes,
                    "local_direct_consolidated_array_count": local_compared,
                    "materialization_seconds": materialization_seconds,
                },
            )
        published_hashes = publication_acceptance["published_hashes"]
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_compared,
            archive_direct_consolidated_array_count=publication_acceptance[
                "archive_direct_consolidated_array_count"
            ],
            published_validation=publication_acceptance["published_validation"],
            published_direct_consolidated_array_count=publication_acceptance[
                "published_direct_consolidated_array_count"
            ],
            source_logical_manifest_sha256=canonical_json_sha256(source_hashes),
            published_logical_manifest_sha256=canonical_json_sha256(published_hashes),
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
        return scratch_user / job_id / f"palette_track_flat_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_track_flat_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument(
        "--profile", choices=(SUPPORTED_PROFILE_ID,), default=SUPPORTED_PROFILE_ID
    )
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_track_kinematics_flat_candidate(
        args.zarr_path,
        source_run=args.source_run,
        run_name=args.run_name,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        profile_id=args.profile,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
