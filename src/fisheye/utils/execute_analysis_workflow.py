"""Execute selected analysis DAG nodes inside an LSF allocation.

The default is a read-only dry run.  ``--apply`` is rejected unless the process
is running under LSF, and every new run is verified complete before a dependent
command is allowed to start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from fisheye.analysis_workflows import (
    AnalysisWorkflow,
    EXECUTION_SCHEMA_ID,
    EXECUTION_SCHEMA_VERSION,
    StageCommand,
    WorkflowExecutionError,
    WorkflowExecutionPlan,
    build_workflow_execution_plan,
    default_core_behavior_profile_path,
    load_analysis_workflow,
    plan_analysis_workflow,
    stage_run_relative_path,
)
from fisheye.analysis_workflows.runtime_verification import (
    RuntimeVerificationSession,
    verify_persisted_stage_output,
)
from fisheye.analytics_exports.publication import export_manifest_path
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_run,
)
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.system_metadata import get_git_info
from fisheye.utils.plan_analysis_workflow import build_availability


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_mapping(values: Iterable[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        key, separator, value = str(raw).partition("=")
        if not separator or not key.strip() or not value.strip():
            raise WorkflowExecutionError(f"{label} must use STAGE=RUN: {raw!r}")
        stage_id = canonical_stage_id(key.strip())
        if stage_id in parsed:
            raise WorkflowExecutionError(f"{label} repeats stage {stage_id!r}")
        parsed[stage_id] = value.strip()
    return parsed


def _export_mapping(
    values: Iterable[str],
    *,
    workflow: AnalysisWorkflow,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        node_id, separator, value = str(raw).partition("=")
        node_id = node_id.strip()
        value = value.strip()
        node = workflow.node_by_id.get(node_id)
        if not separator or node is None or node.kind != "export" or not value:
            raise WorkflowExecutionError(
                f"--export-run must use EXPORT_NODE=RUN for a declared export: {raw!r}"
            )
        if node_id in parsed:
            raise WorkflowExecutionError(f"--export-run repeats node {node_id!r}")
        parsed[node_id] = value
    return parsed


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _preflight_new_outputs(
    zarr_path: Path,
    execution_plan: WorkflowExecutionPlan,
) -> None:
    for command in execution_plan.commands:
        if command.admission is not None:
            receipt_path = Path(command.admission.receipt_path)
            if receipt_path.is_symlink() or receipt_path.exists():
                raise WorkflowExecutionError(
                    "refusing existing admission receipt for "
                    f"{command.node_id}: {receipt_path}"
                )
        if command.output_kind == "parquet_export":
            manifest = export_manifest_path(
                Path(command.output_root), command.output_run
            )
            if manifest.exists():
                raise WorkflowExecutionError(
                    f"refusing existing export run for {command.node_id}: {manifest}"
                )
            continue
        stage_id = command.stage_id
        if stage_id is None:
            raise WorkflowExecutionError(
                f"Zarr-stage command {command.node_id!r} has no canonical stage ID"
            )
        if stage_id == "track_kinematics_visualization":
            # The visualization publisher owns a unique immutable render child
            # below the source-run-specific sibling parent. Existing renders
            # therefore do not collide with a new requested render.
            continue
        relative = stage_run_relative_path(stage_id, command.output_run)
        if (zarr_path / relative / "zarr.json").exists():
            raise WorkflowExecutionError(
                f"refusing existing output run for {stage_id}: {relative}"
            )


def _initial_results(execution_plan: WorkflowExecutionPlan) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for node in execution_plan.workflow_plan.nodes:
        if node.action == "reuse":
            results.append(
                {
                    "node_id": node.node_id,
                    "stage_id": node.stage_id,
                    "status": "reused",
                    "run_name": node.selected_run,
                    "artifact_path": node.artifact_path,
                }
            )
    for command in execution_plan.commands:
        results.append(
            {
                "node_id": command.node_id,
                "stage_id": command.stage_id,
                "status": "pending",
                "run_name": command.output_run,
            }
        )
    return results


def _result_for_node(
    results: list[dict[str, object]],
    node_id: str,
) -> dict[str, object]:
    for result in results:
        if result.get("node_id") == node_id:
            return result
    raise KeyError(node_id)


def _write_report(path: Path | None, payload: Mapping[str, object]) -> None:
    if path is not None:
        write_json_atomic(path, payload)


_FAILED_NODE_STATUSES = frozenset(
    {"blocked_dependency", "failed", "failed_admission", "failed_verification"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _failed_dependencies(
    results: list[dict[str, object]],
    execution_plan: WorkflowExecutionPlan,
    node_id: str,
) -> list[str]:
    node = execution_plan.workflow_plan.node_by_id[node_id]
    failed: list[str] = []
    for dependency in node.depends_on:
        dependency_result = _result_for_node(results, dependency)
        if dependency_result.get("status") in _FAILED_NODE_STATUSES:
            failed.append(dependency)
    return failed


def _remember_failure(payload: dict[str, object], message: str) -> None:
    payload["status"] = "failed"
    if payload.get("error") is None:
        payload["error"] = message


def _revalidate_reused_stage_inputs(
    zarr_path: Path,
    execution_plan: WorkflowExecutionPlan,
    results: list[dict[str, object]],
    *,
    payload: dict[str, object],
    report_path: Path | None,
) -> None:
    """Dynamically admit exact reused artifacts before any command mutates state."""

    session = RuntimeVerificationSession(zarr_path)
    for node in execution_plan.workflow_plan.nodes:
        if node.action != "reuse" or node.stage_id is None:
            continue
        result = _result_for_node(results, node.node_id)
        failed_dependencies: list[str] = []
        dependency_runs: dict[str, str] = {}
        for dependency in node.depends_on:
            try:
                dependency_result = _result_for_node(results, dependency)
            except KeyError:
                # A persisted downstream authority can close a structural branch;
                # its strict resolver remains responsible for that sealed lineage.
                continue
            if dependency_result.get("status") in _FAILED_NODE_STATUSES:
                failed_dependencies.append(dependency)
                continue
            dependency_run = dependency_result.get("run_name")
            if isinstance(dependency_run, str) and dependency_run:
                dependency_runs[dependency] = dependency_run
        if failed_dependencies:
            result["status"] = "blocked_dependency"
            result["blocked_by"] = failed_dependencies
            result["completed_at_utc"] = _utc_now()
            _remember_failure(
                payload,
                f"reused node {node.node_id} was blocked by failed dependencies: "
                + ", ".join(failed_dependencies),
            )
            _write_report(report_path, payload)
            continue

        run_name = result.get("run_name")
        if not isinstance(run_name, str) or not run_name:
            message = f"reused node {node.node_id} has no exact selected run"
            result["status"] = "failed_verification"
            result["completed_at_utc"] = _utc_now()
            result["error"] = message
            _remember_failure(payload, message)
            _write_report(report_path, payload)
            continue
        availability = verify_persisted_stage_output(
            zarr_path,
            node.stage_id,
            requested_run=run_name,
            dependency_runs=dependency_runs,
            session=session,
        )
        result["verification"] = availability.to_dict()
        result["completed_at_utc"] = _utc_now()
        if not availability.available:
            message = (
                f"reused node {node.node_id} failed dynamic admission: "
                f"{availability.reason}"
            )
            result["status"] = "failed_verification"
            result["error"] = message
            _remember_failure(payload, message)
        _write_report(report_path, payload)


def _run_admission_command(
    command: StageCommand,
    result: dict[str, object],
    *,
    environment: Mapping[str, str],
    payload: dict[str, object],
    report_path: Path | None,
) -> str | None:
    admission = command.admission
    if admission is None:
        return None
    receipt_path = Path(admission.receipt_path)
    admission_result: dict[str, object] = {
        "status": "running",
        "receipt_path": str(receipt_path),
        "started_at_utc": _utc_now(),
        "argv": list(admission.argv),
    }
    result["status"] = "admitting"
    result["admission"] = admission_result
    _write_report(report_path, payload)
    print(f"workflow_node={command.node_id}", flush=True)
    print(f"workflow_admission_command={shlex.join(admission.argv)}", flush=True)
    try:
        completed = subprocess.run(admission.argv, check=False, env=environment)
    except OSError as exc:
        admission_result["status"] = "failed"
        admission_result["completed_at_utc"] = _utc_now()
        admission_result["error"] = f"{type(exc).__name__}: {exc}"
        return f"could not start admission for node {command.node_id}: {exc}"
    admission_result["returncode"] = int(completed.returncode)
    admission_result["completed_at_utc"] = _utc_now()
    if completed.returncode != 0:
        admission_result["status"] = "failed"
        return (
            f"admission for node {command.node_id} failed with exit code "
            f"{completed.returncode}"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        admission_result["status"] = "failed"
        admission_result["error"] = (
            "admission command did not create one regular receipt"
        )
        return (
            f"admission for node {command.node_id} returned successfully without "
            f"creating its receipt: {receipt_path}"
        )
    try:
        admission_result["receipt_sha256"] = _sha256_file(receipt_path)
    except OSError as exc:
        admission_result["status"] = "failed"
        admission_result["error"] = f"{type(exc).__name__}: {exc}"
        return f"could not hash admission receipt for node {command.node_id}: {exc}"
    admission_result["status"] = "complete"
    _write_report(report_path, payload)
    return None


def execute_workflow_plan(
    zarr_path: Path,
    execution_plan: WorkflowExecutionPlan,
    *,
    workflow_payload: Mapping[str, object],
    apply: bool,
    report_path: Path | None,
    defer_registry_writes: bool = True,
) -> dict[str, object]:
    """Run topologically, blocking failed descendants but continuing independent nodes."""

    _preflight_new_outputs(zarr_path, execution_plan)
    if apply and not os.environ.get("LSB_JOBID"):
        raise WorkflowExecutionError(
            "--apply is allowed only inside an LSF allocation (LSB_JOBID is unset)"
        )
    if apply and report_path is None:
        raise WorkflowExecutionError(
            "--apply requires --report outside the analysis Zarr"
        )
    if report_path is not None:
        if _path_is_within(report_path, zarr_path):
            raise WorkflowExecutionError(
                "execution report must be outside the analysis Zarr"
            )
        if report_path.exists():
            raise WorkflowExecutionError(
                f"refusing to overwrite existing execution report: {report_path}"
            )

    results = _initial_results(execution_plan)
    payload: dict[str, object] = {
        "schema_id": EXECUTION_SCHEMA_ID,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "execution_id": execution_plan.execution_id,
        "mode": "apply" if apply else "dry_run",
        "status": "running" if apply else "planned",
        "created_at_utc": _utc_now(),
        "completed_at_utc": None,
        "host": os.uname().nodename,
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "palette_git": get_git_info(Path(__file__).resolve().parents[3]),
        "zarr_path": str(zarr_path),
        "registry_write_mode": (
            "deferred_to_serial_finalizer"
            if defer_registry_writes
            else "inline_explicit"
        ),
        "workflow": dict(workflow_payload),
        "execution_plan": execution_plan.to_dict(),
        "node_results": results,
        "error": None,
    }

    if not apply:
        for result in results:
            if result.get("status") == "pending":
                result["status"] = "planned"
        payload["completed_at_utc"] = _utc_now()
        _write_report(report_path, payload)
        return payload

    _write_report(report_path, payload)
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR",
        str((report_path or Path("/tmp/workflow.json")).parent / "matplotlib"),
    )
    if defer_registry_writes:
        environment["PALETTE_DISABLE_REGISTRY_WRITES"] = "1"
    else:
        environment.pop("PALETTE_DISABLE_REGISTRY_WRITES", None)

    _revalidate_reused_stage_inputs(
        zarr_path,
        execution_plan,
        results,
        payload=payload,
        report_path=report_path,
    )

    for command in execution_plan.commands:
        result = _result_for_node(results, command.node_id)
        failed_dependencies = _failed_dependencies(
            results,
            execution_plan,
            command.node_id,
        )
        if failed_dependencies:
            result["status"] = "blocked_dependency"
            result["blocked_by"] = failed_dependencies
            result["completed_at_utc"] = _utc_now()
            _remember_failure(
                payload,
                f"node {command.node_id} was blocked by failed dependencies: "
                + ", ".join(failed_dependencies),
            )
            _write_report(report_path, payload)
            continue

        admission_failure = _run_admission_command(
            command,
            result,
            environment=environment,
            payload=payload,
            report_path=report_path,
        )
        if admission_failure is not None:
            result["status"] = "failed_admission"
            result["completed_at_utc"] = _utc_now()
            result["error"] = admission_failure
            _remember_failure(payload, admission_failure)
            _write_report(report_path, payload)
            continue

        result["status"] = "running"
        result["started_at_utc"] = _utc_now()
        result["argv"] = list(command.argv)
        _write_report(report_path, payload)
        print(f"workflow_node={command.node_id}", flush=True)
        print(f"workflow_command={shlex.join(command.argv)}", flush=True)
        try:
            completed = subprocess.run(command.argv, check=False, env=environment)
        except OSError as exc:
            message = f"could not start node {command.node_id}: {exc}"
            result["status"] = "failed"
            result["completed_at_utc"] = _utc_now()
            result["error"] = f"{type(exc).__name__}: {exc}"
            _remember_failure(payload, message)
            _write_report(report_path, payload)
            continue
        result["returncode"] = int(completed.returncode)
        result["completed_at_utc"] = _utc_now()
        if completed.returncode != 0:
            message = (
                f"node {command.node_id} failed with exit code {completed.returncode}"
            )
            result["status"] = "failed"
            _remember_failure(payload, message)
            _write_report(report_path, payload)
            continue

        if command.output_kind == "parquet_export":
            try:
                export_validation = validate_export_run(
                    Path(command.output_root),
                    command.output_run,
                )
            except (
                ExportValidationError,
                FileNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                message = (
                    f"node {command.node_id} returned successfully but its export "
                    f"manifest failed validation: {exc}"
                )
                result["status"] = "failed_verification"
                result["verification"] = {
                    "available": False,
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _remember_failure(payload, message)
                _write_report(report_path, payload)
                continue
            result["verification"] = {
                **export_validation,
                "available": export_validation.get("status") == "valid",
            }
            output_available = export_validation.get("status") == "valid"
            unavailable_reason = "manifest-selected export is not valid"
        else:
            if command.stage_id is None:
                raise WorkflowExecutionError(
                    f"Zarr-stage command {command.node_id!r} has no canonical stage ID"
                )
            availability = verify_persisted_stage_output(
                zarr_path,
                command.stage_id,
                requested_run=command.output_run,
                dependency_runs=command.dependency_runs,
            )
            result["verification"] = availability.to_dict()
            output_available = availability.available
            unavailable_reason = availability.reason
        if not output_available:
            message = (
                f"node {command.node_id} returned successfully but output run is not "
                f"complete: {unavailable_reason}"
            )
            result["status"] = "failed_verification"
            _remember_failure(payload, message)
            _write_report(report_path, payload)
            continue
        result["status"] = "complete"
        _write_report(report_path, payload)

    if payload["status"] != "failed":
        payload["status"] = "complete"
    payload["completed_at_utc"] = _utc_now()
    _write_report(report_path, payload)
    return payload


def _default_num_workers() -> int:
    raw = os.environ.get("LSB_DJOB_NUMPROC") or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_core_behavior_profile_path(),
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="Executable analysis or implemented export target; repeatable.",
    )
    parser.add_argument("--stage-run", action="append", default=[], metavar="STAGE=RUN")
    parser.add_argument(
        "--output-run", action="append", default=[], metavar="STAGE=RUN"
    )
    parser.add_argument(
        "--export-run",
        action="append",
        default=[],
        metavar="EXPORT_NODE=RUN",
        help="Override the immutable export-run ID for one runnable export node.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        help="Destination root for immutable manifest-selected Parquet exports.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="Explicit node-local scratch root required by runnable export nodes.",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        metavar="STAGE",
        help="Force an otherwise reusable analysis stage to produce a new run.",
    )
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--kinematics-sample-rate-hz", type=float)
    parser.add_argument("--activity-spatial-bin-size-s", type=float)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute serially inside LSF. Default renders a read-only dry run.",
    )
    parser.add_argument(
        "--inline-registry",
        action="store_true",
        help=(
            "Allow stage subprocesses to update SQLite directly. The production "
            "default defers writes to a dependent serial registry finalizer."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        zarr_path = args.zarr_path.expanduser()
        if not (zarr_path / "zarr.json").is_file():
            raise WorkflowExecutionError(
                f"analysis Zarr metadata is missing: {zarr_path}"
            )
        workflow = load_analysis_workflow(args.config)
        stage_runs = _stage_mapping(args.stage_run, label="--stage-run")
        if stage_runs:
            workflow = workflow.with_run_selection(stage_runs)
        workflow = workflow.with_temporal_overrides(
            kinematics_sample_rate_hz=args.kinematics_sample_rate_hz,
            activity_spatial_bin_size_s=args.activity_spatial_bin_size_s,
        )
        availability = build_availability(
            workflow,
            zarr_path,
            forced_unavailable=args.force_stage,
        )
        workflow_plan = plan_analysis_workflow(
            workflow,
            availability,
            targets=tuple(args.target),
        )
        report_path = (
            args.report.expanduser().resolve() if args.report is not None else None
        )
        admission_receipt_root = (
            report_path.parent / "admission_receipts"
            if report_path is not None
            else Path(os.environ.get("TMPDIR") or "/tmp")
            / "palette_analysis_workflow_admission_receipts"
            / str(args.execution_id)
        )
        execution_plan = build_workflow_execution_plan(
            workflow,
            workflow_plan,
            zarr_path=zarr_path,
            execution_id=args.execution_id,
            num_workers=int(args.num_workers),
            output_run_overrides=_stage_mapping(
                args.output_run,
                label="--output-run",
            ),
            export_run_overrides=_export_mapping(
                args.export_run,
                workflow=workflow,
            ),
            export_root=args.export_root,
            scratch_root=args.scratch_root,
            admission_receipt_root=admission_receipt_root,
            python_executable=sys.executable,
        )
        payload = execute_workflow_plan(
            zarr_path,
            execution_plan,
            workflow_payload=workflow.to_dict(),
            apply=bool(args.apply),
            report_path=report_path,
            defer_registry_writes=not bool(args.inline_registry),
        )
    except (KeyError, ValueError, WorkflowExecutionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"execution_id={payload['execution_id']}")
        print(f"mode={payload['mode']}")
        print(f"status={payload['status']}")
        if args.report is not None:
            print(f"report={args.report.expanduser()}")
        for command in execution_plan.commands:
            if command.admission is not None:
                print(
                    f"admission[{command.node_id}]="
                    f"{shlex.join(command.admission.argv)}"
                )
            print(f"command[{command.node_id}]={shlex.join(command.argv)}")
        if payload.get("error"):
            print(f"error={payload['error']}", file=sys.stderr)
    return 0 if payload["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
