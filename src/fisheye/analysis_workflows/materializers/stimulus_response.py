"""Compute stimulus-response tables locally and publish the run atomically."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import zarr

from ...analysis import stimulus_response as response_writer
from ...analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
    consolidate_and_validate_stimulus_response_metadata,
    validate_stimulus_response_metadata_equivalence,
    validate_stimulus_response_storage_receipt,
)
from ...analysis.stimulus_response_coordinate_authority import (
    STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID,
    STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION,
)
from ...shared.json_safety import json_attr_safe
from ...shared.stimulus_coordinate_contract import canonical_mapping_digest
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent
from ...shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_ID,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
    validate_stimulus_response_v3_run,
)
from ...shared.zarr.storage_profiles import get_storage_profile
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group

MATERIALIZATION_SCHEMA_ID = "palette.stimulus_response_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.stimulus_response_run_publish.v1"
MANAGED_WRITER_ARGUMENTS = {
    "--layout",
    "--output-zarr-path",
    "--overwrite",
    "--run-name",
    "--storage-profile",
}


@dataclass(frozen=True)
class StimulusResponseMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    layout: str
    storage_profile_id: str | None
    writer_arguments: tuple[str, ...]

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr / "analysis" / "stimulus_response_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "stimulus_response_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "run_name": self.run_name,
            "layout": self.layout,
            "storage_profile_id": self.storage_profile_id,
            "writer_arguments": list(self.writer_arguments),
        }


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe stimulus-response run name: {run_name!r}.")
    return value


def build_stimulus_response_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    layout: str = "compact_tabular_v2",
    storage_profile_id: str | None = None,
    writer_arguments: Sequence[str] = (),
) -> StimulusResponseMaterializationPlan:
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Scratch root must not be inside the authoritative source Zarr."
        )
    name = _validate_run_name(run_name)
    if layout not in {"compact_tabular_v2", STIMULUS_RESPONSE_LAYOUT}:
        raise ValueError(
            f"Unsupported stimulus-response materializer layout: {layout!r}."
        )
    if storage_profile_id is not None:
        if type(storage_profile_id) is not str:
            raise TypeError("storage_profile_id must be an exact string or None.")
        if layout != STIMULUS_RESPONSE_LAYOUT:
            raise ValueError(
                "Stimulus-response storage candidate requires compact-tabular-v3."
            )
        if storage_profile_id != STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID:
            raise ValueError(
                "Unsupported stimulus-response storage candidate profile: "
                f"{storage_profile_id!r}."
            )
    forwarded = tuple(str(value) for value in writer_arguments)
    forbidden = sorted(
        argument.split("=", 1)[0]
        for argument in forwarded
        if argument.split("=", 1)[0] in MANAGED_WRITER_ARGUMENTS
    )
    if forbidden:
        raise ValueError(
            "Stimulus-response materializer owns these writer arguments: "
            + ", ".join(forbidden)
        )
    target = source / "analysis" / "stimulus_response_runs" / name
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target}"
        )
    return StimulusResponseMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "stimulus-response-output.zarr",
        run_name=name,
        layout=layout,
        storage_profile_id=storage_profile_id,
        writer_arguments=forwarded,
    )


def _validate_stimulus_response_run(
    path: Path,
    *,
    required_layout: str = "compact_tabular_v2",
    required_storage_profile_id: str | None = None,
    require_metadata_equivalence: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    group = open_zarr_root(path, mode="r")
    attrs = dict(group.attrs)
    if str(attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if attrs.get("schema_id") != STIMULUS_RESPONSE_SCHEMA_ID:
        errors.append("invalid schema_id")
    required_version = (
        STIMULUS_RESPONSE_SCHEMA_VERSION
        if required_layout == STIMULUS_RESPONSE_LAYOUT
        else 2
    )
    if (
        type(attrs.get("schema_version")) is not int
        or attrs.get("schema_version") != required_version
    ):
        errors.append("invalid schema_version")
    if attrs.get("layout") != required_layout:
        errors.append(f"stimulus-response layout must be {required_layout}")
    if str(attrs.get("method_version")) != "stimulus_response.v3":
        errors.append("invalid method_version")
    if str(attrs.get("row_axis")) != "stimulus_steps":
        errors.append("invalid row_axis")
    source_refs = attrs.get("source_refs")
    if not isinstance(source_refs, dict):
        errors.append("missing source_refs")
    else:
        coordinate_lineage = source_refs.get("stimulus_coordinate_lineage")
        if not isinstance(coordinate_lineage, dict):
            errors.append("missing stimulus_coordinate_lineage")
        elif (
            coordinate_lineage.get("schema_id")
            != STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_ID
            or coordinate_lineage.get("schema_version")
            != STIMULUS_RESPONSE_COORDINATE_LINEAGE_SCHEMA_VERSION
            or not isinstance(coordinate_lineage.get("record_sha256"), str)
        ):
            errors.append("invalid stimulus_coordinate_lineage")
        else:
            lineage_payload = dict(coordinate_lineage)
            lineage_digest = lineage_payload.pop("record_sha256")
            if canonical_mapping_digest(lineage_payload) != lineage_digest:
                errors.append("stale stimulus_coordinate_lineage digest")
    if not isinstance(attrs.get("parameters"), dict):
        errors.append("missing parameters")
    if required_layout == STIMULUS_RESPONSE_LAYOUT:
        errors.extend(validate_stimulus_response_v3_run(group))
        if required_storage_profile_id is not None:
            if (
                attrs.get("analysis_storage_profile_id")
                != required_storage_profile_id
            ):
                errors.append("invalid analysis_storage_profile_id")
            errors.extend(validate_stimulus_response_storage_receipt(group))
            if require_metadata_equivalence:
                errors.extend(
                    validate_stimulus_response_metadata_equivalence(group)
                )
        elif attrs.get("analysis_storage_profile_role") is not None:
            errors.append("unexpected stimulus-response storage candidate metadata")
    else:
        for required in ("step_index", "global_per_fish"):
            if group.get(required) is None:
                errors.append(f"missing compact table {required}")
    return {
        "valid": not errors,
        "errors": errors,
        "schema_version": attrs.get("schema_version"),
        "layout": attrs.get("layout"),
        "n_steps": attrs.get("n_steps"),
        "n_fish": attrs.get("n_fish"),
        "storage_profile_id": attrs.get("analysis_storage_profile_id"),
    }


def publish_stimulus_response_run(
    plan: StimulusResponseMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    selector_ineligible = plan.layout == STIMULUS_RESPONSE_LAYOUT
    byte_planner_candidate = plan.storage_profile_id is not None

    def validate(path: Path) -> dict[str, Any]:
        return _validate_stimulus_response_run(
            path,
            required_layout=plan.layout,
            required_storage_profile_id=plan.storage_profile_id,
            # Atomic publication adds owner/completion/provenance metadata after
            # each validation pass. The final candidate callback below owns
            # reconsolidation and the strict current-generation equality gate.
            require_metadata_equivalence=False,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "stimulus_response_runs",
            ),
        )

    def complete(
        _root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="stimulus_response_materializer",
            ),
        )
        if not selector_ineligible:
            parent.attrs["latest_complete"] = plan.run_name
            parent.attrs["latest"] = plan.run_name

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/stimulus_response_runs"]
        run_group = parent[plan.run_name]
        pointers_valid = (
            str(parent.attrs.get("latest")) != plan.run_name
            and str(parent.attrs.get("latest_complete")) != plan.run_name
            if selector_ineligible
            else (
                str(parent.attrs.get("latest")) == plan.run_name
                and str(parent.attrs.get("latest_complete")) == plan.run_name
            )
        )
        if (
            not pointers_valid
            or run_group.attrs.get("palette_run_completion_status") != "complete"
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Stimulus-response run was not persisted complete and ineligible "
                "under its requested publication policy."
            )

    def activate(
        _root: zarr.Group,
        parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if (
            str(parent.attrs.get("latest")) != plan.run_name
            or str(parent.attrs.get("latest_complete")) != plan.run_name
            or run_group.attrs.get("palette_run_completion_status") != "complete"
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Stimulus-response activation requires one complete, ineligible run."
            )
        try:
            run_group.attrs["stage_selector_eligible"] = True
        except BaseException:
            if run_group.attrs.get("stage_selector_eligible") is True:
                return
            raise

    def finalize_candidate_metadata(
        root: zarr.Group,
        _parent: zarr.Group,
        run_group: zarr.Group,
    ) -> None:
        if not byte_planner_candidate:
            raise RuntimeError("Non-candidate reached candidate metadata finalization.")
        if run_group.attrs.get("stage_selector_eligible") is not False:
            raise RuntimeError(
                "Stimulus-response storage candidate became selector eligible."
            )
        consolidate_and_validate_stimulus_response_metadata(
            root,
            run_path=f"analysis/stimulus_response_runs/{plan.run_name}",
        )
        final_direct_root = zarr.open_group(
            root.store,
            mode="r",
            use_consolidated=False,
        )
        final_errors = validate_stimulus_response_metadata_equivalence(
            final_direct_root[
                f"analysis/stimulus_response_runs/{plan.run_name}"
            ]
        )
        if final_errors:
            raise RuntimeError(
                "Final stimulus-response candidate metadata equality failed: "
                + "; ".join(final_errors)
            )

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="stimulus-response-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_compute_atomic_run_group_publish",
            rollback_policy=(
                "retain_failed_public_tombstone_leave_unleased_parent_state_untouched"
            ),
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=(
            finalize_candidate_metadata
            if byte_planner_candidate
            else None
            if selector_ineligible
            else activate
        ),
        payload_metadata={
            "copy_backend": copy_backend,
            "promotion_policy": (
                "complete_selector_ineligible_no_parent_pointer_mutation"
                if selector_ineligible
                else "complete_ineligible_then_pointers_then_eligibility_final"
            ),
            "storage_profile_id": plan.storage_profile_id,
            "materialization": json_attr_safe(materialization_payload),
        },
    )


def materialize_stimulus_response(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    layout: str = "compact_tabular_v2",
    storage_profile_id: str | None = None,
    writer_arguments: Sequence[str] = (),
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_stimulus_response_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
        layout=layout,
        storage_profile_id=storage_profile_id,
        writer_arguments=writer_arguments,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        writer_argv = [
            str(plan.source_zarr),
            "--output-zarr-path",
            str(plan.local_zarr),
            "--run-name",
            plan.run_name,
            "--layout",
            plan.layout,
        ]
        if plan.storage_profile_id is not None:
            # Resolve now as a second fail-closed check against the shared,
            # versioned profile registry before invoking the scientific writer.
            get_storage_profile(plan.storage_profile_id)
            writer_argv.extend(
                [
                    "--storage-profile",
                    plan.storage_profile_id,
                    "--no-write-zarr-artifacts",
                ]
            )
        writer_argv.extend(plan.writer_arguments)
        started = time.perf_counter()
        response_writer.main(writer_argv)
        compute_seconds = float(time.perf_counter() - started)
        validation = _validate_stimulus_response_run(
            plan.local_run_path,
            required_layout=plan.layout,
            required_storage_profile_id=plan.storage_profile_id,
            require_metadata_equivalence=False,
        )
        if not validation["valid"]:
            raise RuntimeError(f"Local stimulus-response run is invalid: {validation}")
        payload = {
            "source_access": "authoritative_zarr_read_only",
            "compute_output": "node_local_zarr",
            "compute_duration_seconds": compute_seconds,
            "writer_arguments": writer_argv,
            "local_validation": validation,
        }
        local_group = open_zarr_root(plan.local_run_path, mode="a")
        local_group.attrs["node_local_materialization"] = json_attr_safe(payload)
        if plan.storage_profile_id is not None:
            local_root = open_zarr_root(plan.local_zarr, mode="a")
            equivalence = consolidate_and_validate_stimulus_response_metadata(
                local_root,
                run_path=f"analysis/stimulus_response_runs/{plan.run_name}",
            )
            payload["local_metadata_equivalence"] = equivalence
            final_local_validation = _validate_stimulus_response_run(
                plan.local_run_path,
                required_layout=plan.layout,
                required_storage_profile_id=plan.storage_profile_id,
                require_metadata_equivalence=True,
            )
            if not final_local_validation["valid"]:
                raise RuntimeError(
                    "Local stimulus-response candidate metadata is invalid: "
                    f"{final_local_validation}"
                )
            payload["local_validation"] = final_local_validation
        publish = publish_stimulus_response_run(
            plan,
            materialization_payload=payload,
            copy_backend=copy_backend,
        )
        result.update(
            status="complete",
            local_materialization=payload,
            publish=publish,
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_stimulus_response_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_stimulus_response_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--layout",
        choices=("compact_tabular_v2", STIMULUS_RESPONSE_LAYOUT),
        default="compact_tabular_v2",
        help=(
            "compact_tabular_v3 is an opt-in selector-ineligible contract "
            "checkpoint; compact_tabular_v2 retains legacy materializer behavior"
        ),
    )
    parser.add_argument(
        "--storage-profile",
        choices=(STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,),
        default=None,
        help=(
            "Opt compact-tabular-v3 into the explicit selector-ineligible "
            "byte-planned candidate."
        ),
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] != "--":
        parser.error(
            "unrecognized materializer arguments; place stimulus-response writer arguments after --"
        )
    writer_arguments = tuple(remaining[1:] if remaining[:1] == ["--"] else remaining)
    result = materialize_stimulus_response(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        run_name=args.run_name,
        layout=args.layout,
        storage_profile_id=args.storage_profile,
        writer_arguments=writer_arguments,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
