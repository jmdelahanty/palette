"""Materialize and atomically publish one exact stimulus-epoch v2 candidate.

The established v1 writer remains a compatibility producer.  This boundary
reads one explicit complete v1 run, validates its complete logical table,
rematerializes all twelve arrays on node-local scratch through the shared byte
planner and Zarr v3 factory, and publishes a selector-ineligible immutable v2
candidate without changing parent pointers or production policy.
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
    persist_exact_tabular_storage_receipt,
    rematerialize_exact_tabular_candidate,
    validate_exact_tabular_storage_receipt,
)
from fisheye.analysis.stimulus_epoch_schema import (
    LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
    LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
    STIMULUS_EPOCH_LAYOUT,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    build_stimulus_epoch_array_declarations,
    validate_legacy_stimulus_epoch_source,
    validate_stimulus_epoch_array_manifest,
    write_stimulus_epoch_array_manifest,
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
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    mark_run_complete,
    require_runs_parent,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


PARENT_PATH = "analysis/stimulus_epoch_runs"
SUPPORTED_PROFILE_ID = "published_http_v1"
MATERIALIZATION_SCHEMA_ID = "palette.stimulus_epoch_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.stimulus_epoch_candidate_publish.v1"


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", "..", "latest", "latest_complete"}:
        raise ValueError(f"{label} must be one explicit immutable run name.")
    if "/" in name or "\\" in name or any(character.isspace() for character in name):
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def _require_contained(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the authoritative archive.") from exc


@dataclass(frozen=True)
class StimulusEpochCandidatePlan:
    source_zarr: Path
    source_run_name: str
    run_name: str
    scratch_root: Path
    local_zarr: Path
    profile_id: str
    latest_before: Any
    latest_complete_before: Any

    @property
    def source_run_path(self) -> str:
        return f"{PARENT_PATH}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

    @property
    def source_run_physical_path(self) -> Path:
        return self.source_zarr.joinpath(*self.source_run_path.split("/"))

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
            "profile_id": self.profile_id,
            "latest_before": self.latest_before,
            "latest_complete_before": self.latest_complete_before,
            "source_schema": {
                "schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
                "schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
                "role": "explicit_legacy_input_only",
            },
            "candidate_schema": {
                "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
                "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
                "layout": STIMULUS_EPOCH_LAYOUT,
            },
            "publication_policy": (
                "atomic_named_candidate_selector_ineligible_no_pointer_update"
            ),
        }


def build_stimulus_epoch_candidate_plan(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
) -> StimulusEpochCandidatePlan:
    if profile_id != SUPPORTED_PROFILE_ID:
        raise ValueError(
            f"Stimulus-epoch candidates require profile {SUPPORTED_PROFILE_ID!r}."
        )
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}.")
    if scratch == source or scratch.is_relative_to(source) or source.is_relative_to(scratch):
        raise ValueError(
            "Scratch and authoritative archive trees must be disjoint in both directions."
        )
    source_name = _safe_name(source_run, label="source run")
    candidate_name = _safe_name(run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")

    source_run_path = source.joinpath(*PARENT_PATH.split("/"), source_name)
    if source_run_path.is_symlink():
        raise ValueError("Source stimulus-epoch run cannot be a symbolic link.")
    _require_contained(source_run_path, source, label="source run path")
    if any(path.is_symlink() for path in source_run_path.rglob("*")):
        raise ValueError("Source stimulus-epoch run cannot contain symbolic links.")
    target_run_path = source.joinpath(*PARENT_PATH.split("/"), candidate_name)
    _require_contained(target_run_path.parent, source, label="candidate parent path")

    root = open_zarr_root(source, mode="r")
    parent = root.get(PARENT_PATH)
    if not isinstance(parent, zarr.Group):
        raise KeyError(f"Missing stimulus-epoch parent {PARENT_PATH!r}.")
    source_group = parent.get(source_name)
    if not isinstance(source_group, zarr.Group):
        raise KeyError(f"Source stimulus-epoch run {source_name!r} does not exist.")
    if source_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError("Source stimulus-epoch run is not explicitly complete.")
    if source_group.attrs.get("stage_selector_eligible") is False:
        raise ValueError("Source stimulus-epoch run is explicitly selector-ineligible.")
    errors = validate_legacy_stimulus_epoch_source(source_group)
    if errors:
        raise ValueError("Legacy stimulus-epoch source is invalid: " + "; ".join(errors))
    if candidate_name in parent or target_run_path.exists() or target_run_path.is_symlink():
        raise FileExistsError(f"Candidate run {candidate_name!r} already exists.")

    return StimulusEpochCandidatePlan(
        source_zarr=source,
        source_run_name=source_name,
        run_name=candidate_name,
        scratch_root=scratch,
        local_zarr=scratch / "stimulus-epoch-candidate.zarr",
        profile_id=profile_id,
        latest_before=parent.attrs.get("latest"),
        latest_complete_before=parent.attrs.get("latest_complete"),
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
        values = np.ascontiguousarray(array[...])
        digest = hashlib.sha256()
        digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        hashes[declaration.path] = digest.hexdigest()
    return hashes


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


def _metadata_declarations(group: Any, declarations: Sequence[Any]) -> dict[str, Any]:
    return {
        declaration.path: _array_at_path(group, declaration.path).metadata.to_dict()
        for declaration in declarations
    }


def _direct_consolidated_check(
    zarr_path: Path,
    *,
    run_path: str,
    declarations: Sequence[Any],
) -> int:
    consolidate_metadata_capture_expected_warnings(zarr_path)
    direct_root = zarr.open_group(
        str(zarr_path), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated_root = zarr.open_group(
        str(zarr_path), mode="r", zarr_format=3, use_consolidated=True
    )
    direct = direct_root[run_path]
    consolidated = consolidated_root[run_path]
    if _normalize_metadata(dict(direct.attrs)) != _normalize_metadata(
        dict(consolidated.attrs)
    ):
        raise ValueError("Direct and consolidated stimulus-epoch attributes differ.")
    direct_arrays = _metadata_declarations(direct, declarations)
    consolidated_arrays = _metadata_declarations(consolidated, declarations)
    if _normalize_metadata(direct_arrays) != _normalize_metadata(consolidated_arrays):
        raise ValueError("Direct and consolidated stimulus-epoch arrays differ.")
    return len(direct_arrays)


def _validate_candidate_group(
    group: Any,
    *,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    errors = list(
        validate_stimulus_epoch_array_manifest(
            group,
            byte_planner_adopted=True,
        )
    )
    try:
        declarations = build_stimulus_epoch_array_declarations(
            group,
            byte_planner_adopted=True,
        )
        errors.extend(
            validate_exact_tabular_storage_receipt(
                group,
                declarations=declarations,
            )
        )
        hashes = _logical_hashes(group, declarations)
        if dict(hashes) != dict(expected_hashes):
            errors.append("candidate decoded logical hashes differ from source")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        declarations = ()
        hashes = {}
    if group.attrs.get("stage_selector_eligible") is not False:
        errors.append("candidate is not selector-ineligible")
    if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("candidate is not complete")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": hashes,
    }


def materialize_stimulus_epoch_candidate(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_stimulus_epoch_candidate_plan(
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
        return result
    if plan.scratch_root.exists() or plan.scratch_root.is_symlink():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}.")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        source_root = open_zarr_root(plan.source_zarr, mode="r")
        source_group = source_root[plan.source_run_path]
        source_declarations = build_stimulus_epoch_array_declarations(
            source_group,
            byte_planner_adopted=False,
        )
        candidate_declarations = build_stimulus_epoch_array_declarations(
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
        local_root = zarr.open_group(str(plan.local_zarr), mode="w-", zarr_format=3)
        local_parent = local_root
        for component in PARENT_PATH.split("/"):
            local_parent = local_parent.require_group(component)
        local_group = local_parent.create_group(plan.run_name)
        rematerialize_exact_tabular_candidate(
            source_group,
            local_group,
            receipt=receipt,
        )
        local_group.attrs.update(
            {
                "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
                "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
                "layout": STIMULUS_EPOCH_LAYOUT,
                "row_axis": "epoch_windows",
                "legacy_source_schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
                "legacy_source_schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
                "storage_candidate_source_run": plan.source_run_name,
                "storage_candidate_source_run_path": plan.source_run_path,
                "storage_candidate_profile_promoted": False,
                "stage_selector_eligible": False,
                RUN_NAME_ATTR: plan.run_name,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
            }
        )
        local_group.attrs.pop(RUN_COMPLETED_AT_ATTR, None)
        write_stimulus_epoch_array_manifest(
            local_group,
            byte_planner_adopted=True,
        )
        persist_exact_tabular_storage_receipt(local_group, receipt)
        mark_run_complete(
            local_group,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                source_group.attrs.get("provenance", {}),
                fallback_command="stimulus_epoch_candidate_materializer",
            ),
        )
        local_validation = _validate_candidate_group(
            local_group,
            expected_hashes=source_hashes,
        )
        if not local_validation["valid"]:
            raise RuntimeError(
                f"Local stimulus-epoch candidate is invalid: {local_validation}."
            )
        local_compared = _direct_consolidated_check(
            plan.local_zarr,
            run_path=plan.run_path,
            declarations=candidate_declarations,
        )
        materialization_seconds = float(time.perf_counter() - started)

        def validate(path: Path) -> dict[str, Any]:
            return _validate_candidate_group(
                open_zarr_root(path, mode="r"),
                expected_hashes=source_hashes,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, "stimulus_epoch_runs"),)

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
            parent = root[PARENT_PATH]
            if (
                parent.attrs.get("latest") != plan.latest_before
                or parent.attrs.get("latest_complete") != plan.latest_complete_before
            ):
                raise RuntimeError("Stimulus-epoch candidate changed parent selectors.")
            candidate = parent.get(plan.run_name)
            if not isinstance(candidate, zarr.Group):
                raise RuntimeError("Published stimulus-epoch candidate is missing.")
            if (
                candidate.attrs.get("stage_selector_eligible") is not False
                or candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR)
                != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError(
                    "Published stimulus-epoch candidate is not complete and ineligible."
                )

        archive_compared: list[int] = []

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
                    "Stimulus-epoch candidate lost its fail-closed state before "
                    "consolidation."
                )
            archive_compared.append(
                _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declarations=candidate_declarations,
                )
            )

        def repair_failed_visibility(_target_path: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=plan.local_run_path,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="stimulus-epoch-storage-candidate",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy="stimulus_epoch_v2_byte_planned_atomic_nonpromoting_publish",
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
                "local_direct_consolidated_array_count": local_compared,
                "materialization_seconds": materialization_seconds,
            },
        )
        if archive_compared != [len(candidate_declarations)]:
            raise RuntimeError(
                "Stimulus-epoch archive metadata was not consolidated exactly once."
            )
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_compared,
            archive_direct_consolidated_array_count=archive_compared[0],
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
        return scratch_user / job_id / f"palette_stimulus_epoch_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_stimulus_epoch_{job_id}_{run_name}"
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
    result = materialize_stimulus_epoch_candidate(
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


__all__ = [
    "MATERIALIZATION_SCHEMA_ID",
    "PUBLISH_SCHEMA_ID",
    "SUPPORTED_PROFILE_ID",
    "StimulusEpochCandidatePlan",
    "build_stimulus_epoch_candidate_plan",
    "materialize_stimulus_epoch_candidate",
]
