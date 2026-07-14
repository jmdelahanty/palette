"""Execute selected analysis DAG nodes inside an LSF allocation.

The default is a read-only dry run.  ``--apply`` is rejected unless the process
is running under LSF, and every new run is verified complete before a dependent
command is allowed to start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from fisheye.analysis_workflows import (
    EXECUTION_SCHEMA_ID,
    EXECUTION_SCHEMA_VERSION,
    WorkflowExecutionError,
    WorkflowExecutionPlan,
    build_workflow_execution_plan,
    default_core_behavior_profile_path,
    discover_stage_availability,
    load_analysis_workflow,
    plan_analysis_workflow,
    stage_run_relative_path,
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
    for stage_id, run_name in execution_plan.output_runs.items():
        relative = stage_run_relative_path(stage_id, run_name)
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


def execute_workflow_plan(
    zarr_path: Path,
    execution_plan: WorkflowExecutionPlan,
    *,
    workflow_payload: Mapping[str, object],
    apply: bool,
    report_path: Path | None,
) -> dict[str, object]:
    """Run commands serially, verifying completion before advancing."""

    _preflight_new_outputs(zarr_path, execution_plan)
    if apply and not os.environ.get("LSB_JOBID"):
        raise WorkflowExecutionError(
            "--apply is allowed only inside an LSF allocation (LSB_JOBID is unset)"
        )
    if apply and report_path is None:
        raise WorkflowExecutionError("--apply requires --report outside the analysis Zarr")
    if report_path is not None:
        if _path_is_within(report_path, zarr_path):
            raise WorkflowExecutionError("execution report must be outside the analysis Zarr")
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

    for command in execution_plan.commands:
        result = _result_for_node(results, command.node_id)
        result["status"] = "running"
        result["started_at_utc"] = _utc_now()
        result["argv"] = list(command.argv)
        _write_report(report_path, payload)
        print(f"workflow_node={command.node_id}", flush=True)
        print(f"workflow_command={shlex.join(command.argv)}", flush=True)
        try:
            completed = subprocess.run(command.argv, check=False, env=environment)
        except OSError as exc:
            result["status"] = "failed"
            result["completed_at_utc"] = _utc_now()
            result["error"] = f"{type(exc).__name__}: {exc}"
            payload["status"] = "failed"
            payload["error"] = f"could not start node {command.node_id}: {exc}"
            payload["completed_at_utc"] = _utc_now()
            _write_report(report_path, payload)
            return payload
        result["returncode"] = int(completed.returncode)
        result["completed_at_utc"] = _utc_now()
        if completed.returncode != 0:
            result["status"] = "failed"
            payload["status"] = "failed"
            payload["error"] = (
                f"node {command.node_id} failed with exit code {completed.returncode}"
            )
            payload["completed_at_utc"] = _utc_now()
            _write_report(report_path, payload)
            return payload

        availability = discover_stage_availability(
            zarr_path,
            command.stage_id,
            requested_run=command.output_run,
        )
        result["verification"] = availability.to_dict()
        if not availability.available:
            result["status"] = "failed_verification"
            payload["status"] = "failed"
            payload["error"] = (
                f"node {command.node_id} returned successfully but output run is not "
                f"complete: {availability.reason}"
            )
            payload["completed_at_utc"] = _utc_now()
            _write_report(report_path, payload)
            return payload
        result["status"] = "complete"
        _write_report(report_path, payload)

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
        help="Executable analysis target; repeatable. Export targets are not yet supported.",
    )
    parser.add_argument("--stage-run", action="append", default=[], metavar="STAGE=RUN")
    parser.add_argument("--output-run", action="append", default=[], metavar="STAGE=RUN")
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
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        zarr_path = args.zarr_path.expanduser()
        if not (zarr_path / "zarr.json").is_file():
            raise WorkflowExecutionError(f"analysis Zarr metadata is missing: {zarr_path}")
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
            python_executable=sys.executable,
        )
        payload = execute_workflow_plan(
            zarr_path,
            execution_plan,
            workflow_payload=workflow.to_dict(),
            apply=bool(args.apply),
            report_path=args.report.expanduser() if args.report is not None else None,
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
            print(f"command[{command.node_id}]={shlex.join(command.argv)}")
        if payload.get("error"):
            print(f"error={payload['error']}", file=sys.stderr)
    return 0 if payload["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
