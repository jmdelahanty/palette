"""Compute swim-bout candidates locally and publish the run atomically."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from ...analysis import detect_bouts_multi_level as bout_writer
from ...analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_SCHEMA_ID,
    canonical_frame_axis_sha256,
    resolve_swim_bout_frame_axis,
)
from ...analysis.swim_bout_schema import validate_swim_bout_array_manifest
from ...shared.json_safety import json_attr_safe
from ...shared.run_provenance import build_run_provenance_from_stage_record
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import mark_run_complete, require_runs_parent
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group

MATERIALIZATION_SCHEMA_ID = "palette.swim_bout_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.swim_bout_run_publish.v1"
MANAGED_WRITER_ARGUMENTS = {
    "--output-zarr-path",
    "--overwrite",
    "--run-name",
}


@dataclass(frozen=True)
class SwimBoutMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    writer_arguments: tuple[str, ...]

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr / "analysis" / "swim_bout_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "analysis" / "swim_bout_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "run_name": self.run_name,
            "writer_arguments": list(self.writer_arguments),
        }


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe swim-bout run name: {run_name!r}.")
    return value


def build_swim_bout_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    writer_arguments: Sequence[str] = (),
) -> SwimBoutMaterializationPlan:
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
    forwarded = tuple(str(value) for value in writer_arguments)
    forbidden = sorted(
        argument.split("=", 1)[0]
        for argument in forwarded
        if argument.split("=", 1)[0] in MANAGED_WRITER_ARGUMENTS
    )
    if forbidden:
        raise ValueError(
            "Swim-bout materializer owns these writer arguments: "
            + ", ".join(forbidden)
        )
    target = source / "analysis" / "swim_bout_runs" / name
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target}"
        )
    return SwimBoutMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "swim-bout-output.zarr",
        run_name=name,
        writer_arguments=forwarded,
    )


def _validate_swim_bout_run(path: Path, *, source_zarr: Path) -> dict[str, Any]:
    errors: list[str] = []
    group = open_zarr_root(path, mode="r")
    attrs = dict(group.attrs)
    if str(attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if str(attrs.get("schema_id")) != "palette.swim_bout_runs":
        errors.append("invalid schema_id")
    if int(attrs.get("schema_version", -1)) != 8:
        errors.append("production compact swim-bout schema must be version 8")
    if str(attrs.get("layout")) != "compact_tabular_v2":
        errors.append("production swim-bout layout must be compact_tabular_v2")
    if str(attrs.get("method_version")) != str(bout_writer.METHOD_VERSION):
        errors.append("invalid method_version")
    if str(attrs.get("row_axis")) != "swim_bout_rows":
        errors.append("invalid row_axis")
    errors.extend(validate_swim_bout_array_manifest(group))
    if not isinstance(attrs.get("parameters"), dict):
        errors.append("missing parameters")
    if not isinstance(attrs.get("source_refs"), dict):
        errors.append("missing source_refs")
    for required in (
        "indexes/candidates",
        "indexes/signal_variants",
        "tables/bouts",
        "signals/detector_signal_mm_s",
        "signals/detector_signal_signal_ids",
    ):
        if group.get(required) is None:
            errors.append(f"missing {required}")

    detector = group.get("signals/detector_signal_mm_s")
    frame_count = None
    if isinstance(detector, zarr.Array) and detector.ndim == 2:
        frame_count = int(detector.shape[1])
    elif detector is not None:
        errors.append("detector signal must be a two-dimensional array")

    default_signal_id = attrs.get("default_signal_id")
    default_detector_row = None
    default_detector_finite_count = None
    detector_signal_ids = group.get("signals/detector_signal_signal_ids")
    if default_signal_id is None:
        errors.append("missing default_signal_id")
    elif not isinstance(detector_signal_ids, zarr.Array):
        errors.append("detector signal IDs must be an array")
    elif not isinstance(detector, zarr.Array) or detector.ndim != 2:
        pass
    elif detector_signal_ids.ndim != 1 or int(detector_signal_ids.shape[0]) != int(
        detector.shape[0]
    ):
        errors.append("detector signal IDs do not align with detector rows")
    else:
        try:
            selected_signal_id = int(default_signal_id)
            signal_ids = np.asarray(detector_signal_ids[:], dtype=np.int64)
            matching_rows = np.flatnonzero(signal_ids == selected_signal_id)
            if matching_rows.size != 1:
                errors.append(
                    "default_signal_id must select exactly one detector row"
                )
            else:
                default_detector_row = int(matching_rows[0])
                default_values = np.asarray(
                    detector[default_detector_row, :],
                    dtype=np.float32,
                )
                default_detector_finite_count = int(
                    np.count_nonzero(np.isfinite(default_values))
                )
                if default_detector_finite_count == 0:
                    errors.append(
                        "default detector signal has no finite physical samples"
                    )
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"invalid default detector selection: {exc}")

    contract = attrs.get("frame_axis_contract")
    if not isinstance(contract, dict):
        errors.append("missing frame_axis_contract")
    elif contract.get("schema_id") != FRAME_AXIS_CONTRACT_SCHEMA_ID:
        errors.append("invalid frame_axis_contract schema")

    axis_count = None
    if frame_count is not None and isinstance(contract, dict):
        try:
            source_root = open_zarr_root(source_zarr, mode="r")
            axis = resolve_swim_bout_frame_axis(
                source_root,
                group,
                expected_length=frame_count,
            )
            axis_count = int(axis.size) if axis is not None else None
            if axis is None:
                errors.append("frame axis did not resolve")
            elif canonical_frame_axis_sha256(axis) != contract.get("content_sha256"):
                errors.append("authoritative frame-axis content hash mismatch")
        except Exception as exc:
            errors.append(f"frame-axis validation failed: {type(exc).__name__}: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "schema_version": attrs.get("schema_version"),
        "layout": attrs.get("layout"),
        "detector_frame_count": frame_count,
        "default_signal_id": default_signal_id,
        "default_detector_row": default_detector_row,
        "default_detector_finite_count": default_detector_finite_count,
        "resolved_axis_count": axis_count,
    }


def publish_swim_bout_run(
    plan: SwimBoutMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    def validate(path: Path) -> dict[str, Any]:
        return _validate_swim_bout_run(path, source_zarr=plan.source_zarr)

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (require_runs_parent(root.require_group("analysis"), "swim_bout_runs"),)

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
                fallback_command="swim_bout_materializer",
            ),
        )
        parent.attrs["latest_complete"] = plan.run_name
        parent.attrs["latest"] = plan.run_name

    def verify(root: zarr.Group) -> None:
        parent = root["analysis/swim_bout_runs"]
        run_group = parent[plan.run_name]
        if (
            str(parent.attrs.get("latest")) != plan.run_name
            or str(parent.attrs.get("latest_complete")) != plan.run_name
            or run_group.attrs.get("palette_run_completion_status") != "complete"
            or run_group.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Swim-bout run was not persisted complete and ineligible behind "
                "its parent pointers."
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
                "Swim-bout activation requires one complete, ineligible run."
            )
        try:
            run_group.attrs["stage_selector_eligible"] = True
        except BaseException:
            if run_group.attrs.get("stage_selector_eligible") is True:
                return
            raise

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="swim-bout-publish",
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
        activate_run=activate,
        payload_metadata={
            "copy_backend": copy_backend,
            "promotion_policy": (
                "complete_ineligible_then_pointers_then_eligibility_final"
            ),
            "materialization": json_attr_safe(materialization_payload),
        },
    )


def materialize_swim_bouts(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    writer_arguments: Sequence[str] = (),
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_swim_bout_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
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
            *plan.writer_arguments,
        ]
        started = time.perf_counter()
        exit_code = bout_writer.main(writer_argv)
        if exit_code not in (None, 0):
            raise RuntimeError(f"Swim-bout writer exited with status {exit_code}.")
        compute_seconds = float(time.perf_counter() - started)
        validation = _validate_swim_bout_run(
            plan.local_run_path,
            source_zarr=plan.source_zarr,
        )
        if not validation["valid"]:
            raise RuntimeError(f"Local swim-bout run is invalid: {validation}")
        payload = {
            "source_access": "authoritative_zarr_read_only",
            "compute_output": "node_local_zarr",
            "compute_duration_seconds": compute_seconds,
            "writer_arguments": writer_argv,
            "local_validation": validation,
        }
        local_group = open_zarr_root(plan.local_run_path, mode="a")
        local_group.attrs["node_local_materialization"] = json_attr_safe(payload)
        publish = publish_swim_bout_run(
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
        return scratch_user / job_id / f"palette_swim_bouts_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_swim_bouts_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--run-name", required=True)
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
            "unrecognized materializer arguments; place swim-bout writer arguments after --"
        )
    writer_arguments = tuple(remaining[1:] if remaining[:1] == ["--"] else remaining)
    result = materialize_swim_bouts(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        run_name=args.run_name,
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
