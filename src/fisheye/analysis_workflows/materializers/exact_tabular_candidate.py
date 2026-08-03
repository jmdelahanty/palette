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
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.bout_kinematics_schema import (
    build_bout_kinematics_array_declarations,
    validate_bout_kinematics_array_manifest,
    write_bout_kinematics_array_manifest,
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
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    require_runs_parent,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.exact_tabular_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.exact_tabular_candidate_publish.v1"
SUPPORTED_PROFILE_ID = "published_http_v1"


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


def _logical_hashes(group: Any, declarations: Sequence[Any]) -> dict[str, str]:
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


def _metadata_declarations(group: Any, declarations: Sequence[Any]) -> dict[str, Any]:
    return {
        declaration.path: _array_at_path(group, declaration.path).metadata.to_dict()
        for declaration in declarations
    }


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {
            str(key): _normalize_metadata(child) for key, child in value.items()
        }
        if result.get("node_type") == "group":
            envelope = result.get("consolidated_metadata")
            if envelope is None or envelope == {
                "kind": "inline",
                "must_understand": False,
                "metadata": {},
            }:
                result.pop("consolidated_metadata", None)
        return result
    if isinstance(value, (tuple, list)):
        return [_normalize_metadata(child) for child in value]
    if value == "NaN" or (
        isinstance(value, (float, np.floating)) and np.isnan(float(value))
    ):
        return {"palette_exact_float": "nan"}
    return value


def _local_direct_consolidated_check(
    local_zarr: Path,
    *,
    run_path: str,
    declarations: Sequence[Any],
) -> int:
    consolidate_metadata_capture_expected_warnings(local_zarr)
    direct_root = zarr.open_group(
        str(local_zarr), mode="r", use_consolidated=False
    )
    consolidated_root = zarr.open_group(
        str(local_zarr), mode="r", use_consolidated=True
    )
    direct_run = direct_root[run_path]
    consolidated_run = consolidated_root[run_path]
    # A consolidated group node embeds descendant declarations in its
    # ``consolidated_metadata`` envelope, while the direct group node does not.
    # Compare the run's semantic attributes here and every exact array below;
    # do not confuse that representation-only envelope with contract drift.
    if _normalize_metadata(dict(direct_run.attrs)) != _normalize_metadata(
        dict(consolidated_run.attrs)
    ):
        raise ValueError("Direct and consolidated candidate group attributes differ.")
    direct = _metadata_declarations(direct_run, declarations)
    consolidated = _metadata_declarations(consolidated_run, declarations)
    if _normalize_metadata(direct) != _normalize_metadata(consolidated):
        raise ValueError("Direct and consolidated candidate declarations differ.")
    return len(direct)


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
    hashes = _logical_hashes(group, declarations)
    if dict(hashes) != dict(expected_hashes):
        errors.append("candidate decoded logical hashes differ from source")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": hashes,
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
) -> dict[str, Any]:
    """Create and optionally atomically publish one named physical candidate."""

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
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}.")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    family = _family(plan.family_id)
    try:
        source_root = open_zarr_root(plan.source_zarr, mode="r")
        source_group = source_root[plan.source_run_path]
        source_declarations = family.build_declarations(
            source_group,
            byte_planner_adopted=False,
        )
        candidate_declarations = family.build_declarations(
            source_group,
            byte_planner_adopted=True,
        )
        source_hashes = _logical_hashes(source_group, source_declarations)
        receipt = build_exact_tabular_storage_receipt(
            source_group,
            declarations=candidate_declarations,
            profile=get_storage_profile(plan.profile_id),
        )

        started = time.perf_counter()
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
        local_validation = _validate_candidate_group(
            local_group,
            family=family,
            expected_hashes=source_hashes,
        )
        if not local_validation["valid"]:
            raise RuntimeError(f"Local exact compact candidate is invalid: {local_validation}.")
        compared = _local_direct_consolidated_check(
            plan.local_zarr,
            run_path=plan.run_path,
            declarations=candidate_declarations,
        )
        materialization_seconds = float(time.perf_counter() - started)

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

        archive_consolidated_counts: list[int] = []

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
            archive_consolidated_counts.append(
                _local_direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declarations=candidate_declarations,
                )
            )

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
            payload_metadata={
                "profile_id": plan.profile_id,
                "source_run": plan.source_run_name,
                "source_logical_hashes": source_hashes,
                "local_direct_consolidated_array_count": compared,
                "materialization_seconds": materialization_seconds,
            },
        )
        if archive_consolidated_counts != [len(candidate_declarations)]:
            raise RuntimeError(
                "Exact compact archive metadata was not consolidated exactly once."
            )
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=compared,
            archive_direct_consolidated_array_count=(
                archive_consolidated_counts[0]
            ),
            materialization_seconds=materialization_seconds,
            publication=publication,
        )
        succeeded = True
        return json_attr_safe(result)
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
    )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
