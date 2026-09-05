"""Plan a configurable recording-analysis DAG without mutating the recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from fisheye.analysis_workflows import (
    AnalysisWorkflow,
    StageAvailability,
    default_core_behavior_profile_path,
    discover_stage_availability,
    load_analysis_workflow,
    plan_analysis_workflow,
)
from fisheye.analysis_workflows.dag import topological_order
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.json_safety import write_json_atomic


def _key_value(values: Iterable[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = str(value).partition("=")
        if not separator or not key.strip() or not item.strip():
            raise ValueError(f"{label} must use STAGE=VALUE: {value!r}")
        parsed[canonical_stage_id(key.strip())] = item.strip()
    return parsed


def _forced_available(values: Iterable[str]) -> dict[str, StageAvailability]:
    parsed: dict[str, StageAvailability] = {}
    for value in values:
        stage, separator, path = str(value).partition("=")
        stage_id = canonical_stage_id(stage.strip())
        parsed[stage_id] = StageAvailability(
            stage_id=stage_id,
            available=True,
            artifact_path=path.strip() if separator and path.strip() else None,
            reason="declared available by planner override",
        )
    return parsed


def build_availability(
    workflow: AnalysisWorkflow,
    zarr_path: Path,
    *,
    forced_available: Mapping[str, StageAvailability] | None = None,
    forced_unavailable: Iterable[str] = (),
) -> dict[str, StageAvailability]:
    """Resolve all persisted workflow stages using metadata files only."""

    unavailable = {canonical_stage_id(value) for value in forced_unavailable}
    forced = dict(forced_available or {})
    statuses: dict[str, StageAvailability] = {}
    for node_id in topological_order(workflow):
        node = workflow.node_by_id[node_id]
        if node.stage_id is None or node.stage_id in statuses:
            continue
        stage_id = node.stage_id
        if stage_id in unavailable:
            statuses[stage_id] = StageAvailability(
                stage_id=stage_id,
                available=False,
                reason="declared unavailable by planner override",
            )
        elif stage_id in forced:
            statuses[stage_id] = forced[stage_id]
        else:
            requested_run = workflow.run_selection.get(stage_id)
            dependency_runs: dict[str, str] = {}
            missing_dependencies: list[str] = []
            for dependency_id in node.depends_on:
                dependency_node = workflow.node_by_id[dependency_id]
                dependency_stage = dependency_node.stage_id
                dependency_status = (
                    statuses.get(dependency_stage)
                    if dependency_stage is not None
                    else None
                )
                if (
                    dependency_status is None
                    or not dependency_status.available
                    or not dependency_status.run_name
                ):
                    missing_dependencies.append(dependency_id)
                else:
                    dependency_runs[dependency_id] = dependency_status.run_name
            if missing_dependencies:
                statuses[stage_id] = StageAvailability(
                    stage_id=stage_id,
                    available=False,
                    reason=(
                        "workflow dependency inputs are unavailable: "
                        + ", ".join(missing_dependencies)
                    ),
                )
                continue
            if node.output_run_from is not None:
                requested_run = dependency_runs[node.output_run_from]
            statuses[stage_id] = discover_stage_availability(
                zarr_path,
                stage_id,
                requested_run=requested_run,
                dependency_runs=dependency_runs,
            )
    return statuses


def build_plan_payload(
    workflow: AnalysisWorkflow,
    zarr_path: Path,
    *,
    targets: Sequence[str] | None = None,
    forced_available: Mapping[str, StageAvailability] | None = None,
    forced_unavailable: Iterable[str] = (),
) -> dict[str, object]:
    availability = build_availability(
        workflow,
        zarr_path,
        forced_available=forced_available,
        forced_unavailable=forced_unavailable,
    )
    plan = plan_analysis_workflow(workflow, availability, targets=targets)
    return {
        "schema_id": "palette.analysis_workflow_plan",
        "schema_version": 1,
        "mode": "read_only_plan",
        "zarr_path": str(zarr_path),
        "workflow": workflow.to_dict(),
        "availability": {
            stage_id: status.to_dict()
            for stage_id, status in sorted(availability.items())
        },
        "plan": plan.to_dict(),
    }


def _print_human(payload: Mapping[str, object]) -> None:
    plan = payload["plan"]
    if not isinstance(plan, Mapping):
        return
    temporal = plan.get("temporal_policy")
    print(f"workflow: {plan.get('workflow_id')}")
    print(f"zarr_path: {payload.get('zarr_path')}")
    print(f"mode: {payload.get('mode')}")
    print(f"ready: {str(bool(plan.get('ready'))).lower()}")
    if isinstance(temporal, Mapping):
        kinematics = temporal.get("kinematics")
        summaries = temporal.get("activity_spatial")
        eyes = temporal.get("eye_traces")
        tail = temporal.get("tail_traces")
        if isinstance(kinematics, Mapping):
            print(f"kinematics_resolution: {kinematics.get('resolution')}")
            if "sample_rate_hz" in kinematics:
                print(
                    "kinematics_sample_rate_hz: "
                    f"{kinematics.get('sample_rate_hz')}"
                )
        if isinstance(summaries, Mapping):
            print(f"activity_spatial_bin_size_s: {summaries.get('bin_size_s')}")
        if isinstance(eyes, Mapping):
            print(f"eye_trace_resolution: {eyes.get('resolution')}")
        if isinstance(tail, Mapping):
            print(f"tail_trace_resolution: {tail.get('resolution')}")
    print("\norder\taction\tnode\tstage\tselected_run\treason")
    rows = plan.get("nodes")
    if isinstance(rows, Sequence):
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                continue
            print(
                "\t".join(
                    (
                        str(index),
                        str(row.get("action") or ""),
                        str(row.get("node_id") or ""),
                        str(row.get("stage_id") or "-"),
                        str(row.get("selected_run") or "-"),
                        str(row.get("reason") or ""),
                    )
                )
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr_path", type=Path, help="Analysis Zarr to inspect read-only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_core_behavior_profile_path(),
        help="Analysis workflow YAML (default: packaged core_behavior_v1 profile).",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Limit planning to a target; repeatable.",
    )
    parser.add_argument(
        "--stage-run",
        action="append",
        default=[],
        metavar="STAGE=RUN",
        help="Pin one persisted stage run instead of its latest pointer; repeatable.",
    )
    parser.add_argument(
        "--available-stage",
        action="append",
        default=[],
        metavar="STAGE[=PATH]",
        help="Declare a stage available when planning from an external registry; repeatable.",
    )
    parser.add_argument(
        "--unavailable-stage",
        action="append",
        default=[],
        metavar="STAGE",
        help="Force a stage unavailable for what-if planning; repeatable.",
    )
    parser.add_argument(
        "--kinematics-sample-rate-hz",
        type=float,
        help=(
            "Explicitly downsample the framewise kinematic export to this rate; "
            "the profile default preserves every source frame."
        ),
    )
    parser.add_argument(
        "--activity-spatial-bin-size-s",
        type=float,
        help="Override activity/spatial summary bin width (profile default: 5 seconds).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full machine-readable plan."
    )
    parser.add_argument(
        "--json-output", type=Path, help="Optionally write the plan as JSON."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    zarr_path = args.zarr_path.expanduser()
    if not (zarr_path / "zarr.json").is_file():
        raise ValueError(f"analysis Zarr metadata was not found: {zarr_path}")
    workflow = load_analysis_workflow(args.config)
    run_overrides = _key_value(args.stage_run, label="--stage-run")
    if run_overrides:
        workflow = workflow.with_run_selection(run_overrides)
    workflow = workflow.with_temporal_overrides(
        kinematics_sample_rate_hz=args.kinematics_sample_rate_hz,
        activity_spatial_bin_size_s=args.activity_spatial_bin_size_s,
    )
    payload = build_plan_payload(
        workflow,
        zarr_path,
        targets=tuple(args.target) if args.target else None,
        forced_available=_forced_available(args.available_stage),
        forced_unavailable=args.unavailable_stage,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.json_output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
