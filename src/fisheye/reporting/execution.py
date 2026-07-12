"""Explicit, allowlisted execution of actions emitted by the report planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from fisheye.shared.zarr_helpers import zarr_child_group
from fisheye.shared.zarr_io import open_zarr_root

from .models import PlanStatus, ReportPlan, VisualizationPlanItem


@dataclass(frozen=True)
class ExecutionResult:
    recording_id: str
    visualization_id: str
    entity_id: str | None
    action: str
    status: str
    detail: str


@dataclass(frozen=True)
class ExecutionContext:
    zarr_path: Path
    item: VisualizationPlanItem
    overwrite_analysis: bool


Executor = Callable[[ExecutionContext], str]


def _render_track_kinematics(context: ExecutionContext) -> str:
    from fisheye.analysis import plot_track_kinematics

    run = context.item.source_run
    if run is None or context.item.entity_id is None:
        raise ValueError("Track rendering requires a concrete run and track entity.")
    parts = run.path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected track-kinematics run path: {run.path}")
    scope = parts[-2]
    args = [
        str(context.zarr_path),
        "--track-kinematics-run",
        f"{scope}/{run.run_id}",
        "--track-id",
        str(context.item.entity_id),
        "--swim-bout-run",
        "none",
        "--write-zarr-artifacts",
    ]
    args.append("--offline-only" if scope == "offline" else "--online-only")
    plot_track_kinematics.main(args)
    return "rendered contracted track overview and X/Y trace"


def _render_swim_bouts(context: ExecutionContext) -> str:
    from fisheye.analysis import swim_bout_visualization

    run = context.item.source_run
    if run is None:
        raise ValueError("Swim-bout rendering requires a concrete analysis run.")
    attrs, datasets = swim_bout_visualization._load_swim_bout_run(
        context.zarr_path,
        run.run_id,
        speed_level="smoothed",
    )
    speed_level = swim_bout_visualization._display_speed_level(
        str(attrs.get("speed_level") or "smoothed")
    )
    root = open_zarr_root(context.zarr_path, mode="a")
    run_group = zarr_child_group(root, run.path)
    if run_group is None:
        raise ValueError(f"Resolved swim-bout run disappeared: {run.path}")
    swim_bout_visualization.write_swim_bout_visualization_artifact(
        run_group=run_group,
        run_name=run.run_id,
        attrs=attrs,
        datasets=datasets,
        speed_level=speed_level,
    )
    return f"rendered contracted swim-bout summary at speed level {speed_level}"


def _render_eye_angles(context: ExecutionContext) -> str:
    from fisheye.visualization import visualize_eye_angles

    run = context.item.source_run
    if run is None:
        raise ValueError("Eye-angle rendering requires a concrete analysis run.")
    result = visualize_eye_angles.main(
        [
            str(context.zarr_path),
            "--run",
            run.run_id,
            "--angle-source",
            "eye_frame",
            "--no-show",
            "--quiet",
            "--no-filesystem-output",
            "--write-zarr-artifact",
        ]
    )
    if result != 0:
        raise RuntimeError(f"Eye-angle renderer returned {result}")
    return "rendered contracted eye-frame angle and convergence summary"


def _render_moving_grating(context: ExecutionContext) -> str:
    from fisheye.analysis import plot_stimulus_response_omr

    run = context.item.source_run
    if run is None:
        raise ValueError("Moving-grating rendering requires a stimulus-response run.")
    plot_stimulus_response_omr.main(
        [
            str(context.zarr_path),
            "--run",
            run.run_id,
        ]
    )
    return "rendered contracted moving-grating OMR artifacts"


def _render_chaser_egocentric_bearing(context: ExecutionContext) -> str:
    from fisheye.analysis import chaser_egocentric_bearing

    run = context.item.source_run
    if run is None:
        raise ValueError("Egocentric-bearing rendering requires a chaser-distance run.")
    result = chaser_egocentric_bearing.main(
        [
            str(context.zarr_path),
            "--chaser-distance-run",
            run.run_id,
            "--apply",
            "--overwrite",
        ]
    )
    if result != 0:
        raise RuntimeError(f"Egocentric-bearing renderer returned {result}")
    return "materialized and rendered contracted egocentric-bearing component"


def _analyze_chaser_distance(context: ExecutionContext) -> str:
    from fisheye.analysis import chaser_distance_runs

    args = [
        str(context.zarr_path),
        "--run-name",
        "chaser_distance_reporting_v1",
        "--apply",
    ]
    if context.overwrite_analysis:
        args.append("--overwrite")
    result = chaser_distance_runs.main(args)
    if result != 0:
        raise RuntimeError(f"Chaser-distance analysis returned {result}")
    return "materialized contracted chaser-distance analysis and visualizations"


def _analyze_session_occupancy(context: ExecutionContext) -> str:
    from fisheye.analysis.detection_occupancy_runs import (
        build_session_occupancy_result,
        write_session_occupancy_run,
    )

    run_name = "session_occupancy_reporting_v1"
    result = build_session_occupancy_result(
        context.zarr_path,
        run_name=run_name,
    )
    path = write_session_occupancy_run(
        context.zarr_path,
        result,
        overwrite=context.overwrite_analysis,
        write_png=True,
    )
    return f"materialized {path}"


RENDER_EXECUTORS: dict[str, tuple[str, Executor]] = {
    "core.track_kinematics.overview": ("render_track_kinematics", _render_track_kinematics),
    "core.position.xy_trace": ("render_track_kinematics", _render_track_kinematics),
    "core.swim_bouts.overview": ("render_swim_bouts", _render_swim_bouts),
    "core.eye_angles.overview": ("render_eye_angles", _render_eye_angles),
    "stimulus.moving_grating.omr_summary": ("render_moving_grating", _render_moving_grating),
    "stimulus.moving_grating.bout_trajectory": ("render_moving_grating", _render_moving_grating),
    "stimulus.chaser.egocentric_bearing": (
        "render_chaser_egocentric_bearing",
        _render_chaser_egocentric_bearing,
    ),
}


ANALYSIS_EXECUTORS: dict[str, tuple[str, Executor]] = {
    "core.position.occupancy": ("analyze_session_occupancy", _analyze_session_occupancy),
    "stimulus.chaser.distance_trace": (
        "analyze_chaser_distance",
        _analyze_chaser_distance,
    ),
    "stimulus.chaser.distance_distribution": (
        "analyze_chaser_distance",
        _analyze_chaser_distance,
    ),
    "stimulus.chaser.egocentric_bearing": (
        "analyze_chaser_distance",
        _analyze_chaser_distance,
    ),
}


def execution_result_to_dict(result: ExecutionResult) -> dict[str, object]:
    return asdict(result)


def execute_report_plan(
    plan: ReportPlan,
    *,
    render_missing: bool,
    apply_analysis: bool,
    refresh_contract_mismatches: bool = False,
    visualization_ids: Iterable[str] = (),
    overwrite_analysis: bool = False,
    continue_on_error: bool = False,
) -> tuple[ExecutionResult, ...]:
    """Execute only explicitly enabled, allowlisted planner actions."""

    if not render_missing and not apply_analysis:
        raise ValueError("Enable --render-missing and/or --apply-analysis.")
    selected_ids = {str(value) for value in visualization_ids}
    results: list[ExecutionResult] = []
    executed_keys: set[tuple[str, str, str | None, str | None]] = set()

    for recording_plan in plan.recordings:
        for item in recording_plan.items:
            if selected_ids and item.visualization_id not in selected_ids:
                continue
            executor_entry: tuple[str, Executor] | None = None
            action = ""
            if apply_analysis and item.status == PlanStatus.NEEDS_ANALYSIS.value:
                executor_entry = ANALYSIS_EXECUTORS.get(item.visualization_id)
                action = "apply_analysis"
            elif render_missing and (
                item.status == PlanStatus.NEEDS_RENDER.value
                or (
                    refresh_contract_mismatches
                    and item.status == PlanStatus.CONTRACT_MISMATCH.value
                )
            ):
                executor_entry = RENDER_EXECUTORS.get(item.visualization_id)
                action = "render"
            else:
                continue

            if executor_entry is None:
                results.append(
                    ExecutionResult(
                        recording_id=recording_plan.recording.recording_id,
                        visualization_id=item.visualization_id,
                        entity_id=item.entity_id,
                        action=action,
                        status="unsupported",
                        detail="No allowlisted executor is registered for this planned action.",
                    )
                )
                continue

            executor_id, executor = executor_entry
            run_id = item.source_run.run_id if item.source_run is not None else None
            dedupe_key = (
                recording_plan.recording.recording_id,
                executor_id,
                item.entity_id,
                run_id,
            )
            if dedupe_key in executed_keys:
                results.append(
                    ExecutionResult(
                        recording_id=recording_plan.recording.recording_id,
                        visualization_id=item.visualization_id,
                        entity_id=item.entity_id,
                        action=action,
                        status="deduplicated",
                        detail=f"Already handled by executor {executor_id}.",
                    )
                )
                continue

            try:
                detail = executor(
                    ExecutionContext(
                        zarr_path=Path(recording_plan.recording.zarr_path),
                        item=item,
                        overwrite_analysis=overwrite_analysis,
                    )
                )
            except Exception as exc:
                results.append(
                    ExecutionResult(
                        recording_id=recording_plan.recording.recording_id,
                        visualization_id=item.visualization_id,
                        entity_id=item.entity_id,
                        action=action,
                        status="failed",
                        detail=str(exc),
                    )
                )
                if not continue_on_error:
                    return tuple(results)
            else:
                executed_keys.add(dedupe_key)
                results.append(
                    ExecutionResult(
                        recording_id=recording_plan.recording.recording_id,
                        visualization_id=item.visualization_id,
                        entity_id=item.entity_id,
                        action=action,
                        status="executed",
                        detail=detail,
                    )
                )
    return tuple(results)
