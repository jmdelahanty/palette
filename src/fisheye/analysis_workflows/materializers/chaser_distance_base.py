"""Publish one atomic, selector-ineligible chaser-distance base candidate."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Optional, Sequence

import zarr

from fisheye.analysis.chaser_distance_base_schema import (
    CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID,
    CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION,
    build_chaser_distance_base_declarations,
)
from fisheye.analysis.chaser_distance_base_storage import (
    base_logical_hashes,
    build_base_storage_receipt,
    build_source_authority_binding,
    persist_base_candidate_contract,
    rematerialize_base_candidate,
    validate_base_candidate,
)
from fisheye.analysis.chaser_distance_coordinate_publication import (
    load_bound_chaser_distance_run,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
    mark_run_complete,
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.chaser_distance_base_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.chaser_distance_base_candidate_publish.v1"
SUPPORTED_PROFILE_ID = "published_http_v1"
SOURCE_PARENT_PATH = "analysis/chaser_distance_runs"
CANDIDATE_PARENT_PATH = "analysis/chaser_distance_storage_candidates"


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if (
        not name
        or name in {".", "..", "latest", "latest_complete"}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return name


def _published_root(path: Path) -> zarr.Group:
    return zarr.open_group(
        str(path), mode="r", zarr_format=3, use_consolidated=True
    )


def _pointer_snapshot(group: Any) -> tuple[tuple[str, object], ...]:
    names = (
        "latest", "latest_complete", "latest_pending", "authoritative_run",
        "authoritative_run_provenance", "publication_policy",
        "publication_generation", "chaser_distance_publication_lease",
    )
    return tuple((name, group.attrs.get(name)) for name in names)


def _direct_consolidated_check(
    archive: Path,
    *,
    run_path: str,
    expected_arrays: int,
) -> int:
    consolidate_metadata_capture_expected_warnings(archive)
    receipt = validate_direct_consolidated_subtree(
        archive,
        subtree_path=run_path,
    )
    if receipt.array_count != expected_arrays:
        raise ValueError(
            "Direct/consolidated chaser-distance candidate inventory differs: "
            f"expected {expected_arrays}, got {receipt.array_count}."
        )
    return receipt.array_count


@dataclass(frozen=True)
class ChaserDistanceBaseCandidatePlan:
    source_zarr: Path
    source_run_name: str
    run_name: str
    scratch_root: Path
    local_zarr: Path
    profile_id: str
    source_parent_pointers_before: tuple[tuple[str, object], ...]
    candidate_parent_attrs_before: tuple[tuple[str, object], ...] | None

    @property
    def source_run_path(self) -> str:
        return f"{SOURCE_PARENT_PATH}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{CANDIDATE_PARENT_PATH}/{self.run_name}"

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
            "source_parent_pointers_before": dict(self.source_parent_pointers_before),
            "candidate_parent_attrs_before": (
                None
                if self.candidate_parent_attrs_before is None
                else dict(self.candidate_parent_attrs_before)
            ),
            "publication_policy": (
                "atomic_named_selector_ineligible_nonpromoting_sealed_base_projection"
            ),
        }


def build_chaser_distance_base_candidate_plan(
    source_zarr: str | Path,
    *,
    source_run: str,
    run_name: str,
    scratch_root: str | Path,
    profile_id: str = SUPPORTED_PROFILE_ID,
) -> ChaserDistanceBaseCandidatePlan:
    if profile_id != SUPPORTED_PROFILE_ID:
        raise ValueError(f"Only profile {SUPPORTED_PROFILE_ID!r} is supported.")
    archive = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}.")
    if scratch == archive or scratch.is_relative_to(archive) or archive.is_relative_to(scratch):
        raise ValueError("Scratch and authoritative archive must be disjoint.")
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(run_name, label="run_name")
    source_path = archive.joinpath(*SOURCE_PARENT_PATH.split("/"), source_name)
    if source_path.is_symlink() or source_path.resolve() != source_path:
        raise ValueError("Canonical chaser-distance source path must not be a symlink.")
    root = _published_root(archive)
    source_group = root[f"{SOURCE_PARENT_PATH}/{source_name}"]
    # This is the crucial authority gate: it re-derives the source against its
    # live detection/stimulus authorities and rejects stale publication seals.
    bound = load_bound_chaser_distance_run(
        root,
        f"{SOURCE_PARENT_PATH}/{source_name}",
    )
    build_source_authority_binding(bound, source_group=source_group)
    build_chaser_distance_base_declarations(source_group)
    source_parent = root[SOURCE_PARENT_PATH]
    candidate_parent = root.get(CANDIDATE_PARENT_PATH)
    candidate_attrs = (
        None
        if candidate_parent is None
        else tuple(sorted(dict(candidate_parent.attrs).items()))
    )
    target = archive.joinpath(*CANDIDATE_PARENT_PATH.split("/"), candidate_name)
    if target.exists():
        raise FileExistsError(f"Candidate already exists: {target}.")
    return ChaserDistanceBaseCandidatePlan(
        source_zarr=archive,
        source_run_name=source_name,
        run_name=candidate_name,
        scratch_root=scratch,
        local_zarr=scratch / "candidate_analysis.zarr",
        profile_id=profile_id,
        source_parent_pointers_before=_pointer_snapshot(source_parent),
        candidate_parent_attrs_before=candidate_attrs,
    )


def materialize_chaser_distance_base_candidate(
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
    plan = build_chaser_distance_base_candidate_plan(
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
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}.")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        source_root = _published_root(plan.source_zarr)
        source_group = source_root[plan.source_run_path]
        bound = load_bound_chaser_distance_run(source_root, plan.source_run_path)
        source_binding = build_source_authority_binding(
            bound, source_group=source_group
        )
        declarations = build_chaser_distance_base_declarations(source_group)
        source_hashes = base_logical_hashes(source_group, declarations)
        receipt = build_base_storage_receipt(
            source_group,
            profile=get_storage_profile(plan.profile_id),
        )
        started = time.perf_counter()
        local_root = zarr.open_group(
            str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
        )
        local_parent = local_root
        for component in CANDIDATE_PARENT_PATH.split("/"):
            local_parent = local_parent.require_group(component)
        local_group = local_parent.create_group(plan.run_name)
        rematerialize_base_candidate(source_group, local_group, receipt=receipt)
        local_group.attrs.update(
            {
                "schema_id": CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID,
                "schema_version": CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION,
                "method": "sealed_chaser_distance_base_exact_rematerialization",
                "method_version": "1",
                RUN_NAME_ATTR: plan.run_name,
                RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_RUNNING,
                "stage_selector_eligible": False,
                "storage_candidate_profile_promoted": False,
            }
        )
        persist_base_candidate_contract(
            local_group,
            receipt=receipt,
            declarations=declarations,
            source_binding=source_binding,
            source_hashes=source_hashes,
        )
        mark_run_complete(
            local_group,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                source_group.attrs.get("run_provenance", source_group.attrs.get("provenance", {})),
                fallback_command="chaser_distance_base_candidate_materializer",
            ),
        )
        local_validation = validate_base_candidate(
            local_group,
            source_group=source_group,
            expected_source_binding=source_binding,
        )
        if not local_validation["valid"]:
            raise RuntimeError(f"Local chaser-distance candidate is invalid: {local_validation}.")
        local_compared = _direct_consolidated_check(
            plan.local_zarr,
            run_path=plan.run_path,
            expected_arrays=len(declarations),
        )
        materialization_seconds = float(time.perf_counter() - started)

        def verified_source() -> tuple[Any, dict[str, Any]]:
            current_root = _published_root(plan.source_zarr)
            current_bound = load_bound_chaser_distance_run(
                current_root, plan.source_run_path
            )
            current_group = current_root[plan.source_run_path]
            return current_group, build_source_authority_binding(
                current_bound,
                source_group=current_group,
            )

        def validate(path: Path) -> dict[str, Any]:
            group = zarr.open_group(
                str(path), mode="r", zarr_format=3, use_consolidated=False
            )
            current_source, current_binding = verified_source()
            return validate_base_candidate(
                group,
                source_group=current_source,
                expected_source_binding=current_binding,
            )

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (analysis.require_group("chaser_distance_storage_candidates"),)

        def complete(
            _root: zarr.Group, _parent: zarr.Group, run_group: zarr.Group
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
            source_parent = root[SOURCE_PARENT_PATH]
            if _pointer_snapshot(source_parent) != plan.source_parent_pointers_before:
                raise RuntimeError("Candidate publication changed chaser-distance selectors.")
            parent = root[CANDIDATE_PARENT_PATH]
            before = plan.candidate_parent_attrs_before
            if before is not None and tuple(sorted(dict(parent.attrs).items())) != before:
                raise RuntimeError("Candidate publication changed candidate-parent attrs.")
            candidate = parent.get(plan.run_name)
            if candidate is None or candidate.attrs.get("stage_selector_eligible") is not False:
                raise RuntimeError("Published candidate is absent or selector eligible.")
            if candidate.attrs.get("storage_candidate_profile_promoted") is not False:
                raise RuntimeError("Published candidate profile became promoted.")

        compared: list[int] = []

        def consolidate(
            _root: zarr.Group, _parent: zarr.Group, run_group: zarr.Group
        ) -> None:
            if (
                run_group.attrs.get("stage_selector_eligible") is not False
                or run_group.attrs.get("storage_candidate_profile_promoted") is not False
                or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            ):
                raise RuntimeError("Candidate lost its complete nonpromoting state.")
            compared.append(
                _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    expected_arrays=len(declarations),
                )
            )

        def repair(_target: Path) -> None:
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=plan.local_run_path,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="chaser-distance-base-storage-candidate",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy="sealed_base_byte_planned_atomic_nonpromoting_publish",
                rollback_policy=(
                    "retain_failed_public_tombstone_leave_all_selectors_untouched"
                ),
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            activate_run=consolidate,
            repair_failed_publication_visibility=repair,
            payload_metadata={
                "profile_id": plan.profile_id,
                "source_run_path": plan.source_run_path,
                "source_binding": source_binding,
                "source_logical_hashes": source_hashes,
                "local_direct_consolidated_array_count": local_compared,
                "materialization_seconds": materialization_seconds,
            },
        )
        if compared != [len(declarations)]:
            raise RuntimeError("Archive candidate metadata was not consolidated once.")
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_compared,
            archive_direct_consolidated_array_count=compared[0],
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
    job = os.environ.get("LSB_JOBID") or "manual"
    root = Path("/scratch") / user
    if root.is_dir() and os.access(root, os.W_OK | os.X_OK):
        return root / job / f"palette_chaser_distance_base_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_chaser_distance_base_{job}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
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
    result = materialize_chaser_distance_base_candidate(
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
    "ChaserDistanceBaseCandidatePlan",
    "build_chaser_distance_base_candidate_plan",
    "materialize_chaser_distance_base_candidate",
]
