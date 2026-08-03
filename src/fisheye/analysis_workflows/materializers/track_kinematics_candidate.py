"""Publish one selector-ineligible byte-planned track-kinematics v2 candidate."""

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

from fisheye.analysis.track_kinematics_storage import (
    TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
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
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
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
)

from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.track_kinematics_flat_candidate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.track_kinematics_flat_candidate_publish.v1"
SUPPORTED_PROFILE_ID = "published_http_v1"
PARENT_PATH = "analysis/track_kinematics_runs"
RUN_TYPE = "offline"


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
    consolidate_metadata_capture_expected_warnings(archive)
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
) -> dict[str, Any]:
    """Rematerialize and atomically publish one non-promoting v2 candidate."""

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
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}.")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        source_root = _published_root(plan.source_zarr)
        source_group = source_root[plan.source_run_path]
        declarations = build_flat_candidate_declarations(source_group)
        paths = tuple(declaration.path for declaration in declarations)
        source_hashes = source_flat_projection_hashes(source_group, declarations)
        receipt = build_flat_candidate_storage_receipt(
            source_group,
            profile=get_storage_profile(plan.profile_id),
        )

        started = time.perf_counter()
        local_root = zarr.open_group(
            str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
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
                "method": "track_kinematics_v1_exact_flat_lineage_rematerialization",
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
            }
        )
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
        local_validation = validate_flat_candidate(
            local_group,
            source_group=source_group,
        )
        if not local_validation["valid"]:
            raise RuntimeError(
                f"Local track flat candidate is invalid: {local_validation}."
            )
        local_compared = _direct_consolidated_check(
            plan.local_zarr,
            run_path=plan.run_path,
            declaration_paths=paths,
        )
        materialization_seconds = float(time.perf_counter() - started)

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
                    "Track flat candidate lost its complete ineligible state."
                )
            archive_consolidated_counts.append(
                _direct_consolidated_check(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    declaration_paths=paths,
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
                lock_suffix="track-flat-lineage-storage-candidate",
                publish_schema_id=PUBLISH_SCHEMA_ID,
                policy="track_flat_lineage_byte_planned_atomic_nonpromoting_publish",
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
        if archive_consolidated_counts != [len(declarations)]:
            raise RuntimeError(
                "Track flat archive metadata was not consolidated exactly once."
            )
        result.update(
            status="complete",
            local_validation=local_validation,
            local_direct_consolidated_array_count=local_compared,
            archive_direct_consolidated_array_count=archive_consolidated_counts[0],
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
