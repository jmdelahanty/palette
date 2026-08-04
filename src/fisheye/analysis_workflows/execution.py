"""Fail-closed command rendering for analysis-workflow DAGs.

This module turns a validated read-only workflow plan into exact subprocess
commands.  It does not submit jobs or execute commands; the CLI runner owns
those side effects and verifies each completed run before advancing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping

from fisheye.registry.stage_catalog import canonical_stage_id

from .contracts import AnalysisWorkflow
from .dag import NodePlan, WorkflowPlan

EXECUTION_SCHEMA_ID = "palette.analysis_workflow_execution"
EXECUTION_LEGACY_SCHEMA_VERSION = 1
EXECUTION_SCHEMA_VERSION = 2
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WorkflowExecutionError(RuntimeError):
    """Raised when a plan cannot be rendered or executed safely."""


@dataclass(frozen=True)
class StageCommandContext:
    zarr_path: Path
    node: NodePlan
    output_run: str
    dependency_runs: Mapping[str, str]
    python_executable: str
    num_workers: int
    export_root: Path | None = None
    scratch_root: Path | None = None

    def dependency_run(self, node_id: str) -> str:
        try:
            return self.dependency_runs[node_id]
        except KeyError as exc:
            raise WorkflowExecutionError(
                f"node {self.node.node_id!r} has no resolved run for dependency "
                f"{node_id!r}"
            ) from exc


@dataclass(frozen=True)
class StageCommand:
    node_id: str
    node_kind: str
    stage_id: str | None
    output_run: str
    output_kind: str
    output_root: str
    dependency_runs: Mapping[str, str]
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "stage_id": self.stage_id,
            "output_run": self.output_run,
            "output_kind": self.output_kind,
            "output_root": self.output_root,
            "dependency_runs": dict(self.dependency_runs),
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class WorkflowExecutionPlan:
    execution_id: str
    workflow_plan: WorkflowPlan
    output_runs: Mapping[str, str]
    export_runs: Mapping[str, str]
    commands: tuple[StageCommand, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": EXECUTION_SCHEMA_ID,
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "workflow_plan": self.workflow_plan.to_dict(),
            "output_runs": dict(self.output_runs),
            "export_runs": dict(self.export_runs),
            "commands": [command.to_dict() for command in self.commands],
        }


StageCommandBuilder = Callable[[StageCommandContext], tuple[str, ...]]


def _module_command(context: StageCommandContext, module: str) -> list[str]:
    return [
        context.python_executable,
        "-m",
        module,
        str(context.zarr_path),
    ]


def _track_kinematics_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.track_kinematics",
    )
    command.extend(
        (
            "--keypoint-run",
            f"refined/{context.dependency_run('refined_keypoints')}",
            "--run-name",
            context.output_run,
            "--output-shard-rows",
            "262144",
            "--shard-workers",
            str(max(1, context.num_workers)),
            "--apply",
            "--",
            "--hysteresis-high-px",
            "4.0",
            "--hysteresis-low-px",
            "2.0",
            "--hysteresis-min-frames",
            "3",
            "--hysteresis-band-policy",
            "latch",
            "--smooth-seconds",
            "0.05",
            "--smoothing-alignment",
            "causal",
        )
    )
    return tuple(command)


def _swim_bout_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.swim_bouts",
    )
    command.extend(
        (
            "--run-name",
            context.output_run,
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
            "--",
            "--track-kinematics-run",
            context.dependency_run("track_kinematics"),
            "--track-id",
            "0",
            "--method",
            "peak_event",
            "--layout",
            "compact_v2",
            "--default-level",
            "exponential",
            "--exponential-source-level",
            "filtered",
            "--exponential-tau-s",
            "0.025",
            "--min-peak-prominence-mm-s",
            "4.0",
            "--min-peak-distance-s",
            "0.10",
            "--peak-width-rel-height",
            "0.98",
            "--peak-event-boundary-mode",
            "relative_prominence_width",
            "--shape-split-policy",
            "none",
            "--min-bout-duration",
            "0.05",
            "--min-gap-duration",
            "0.1",
            "--gap-merge-policy",
            "sampled_frame_gap",
        )
    )
    return tuple(command)


def _track_kinematics_visualization_command(
    context: StageCommandContext,
) -> tuple[str, ...]:
    track_run = context.dependency_run("track_kinematics")
    if context.output_run != track_run:
        raise WorkflowExecutionError(
            "track-kinematics visualization output must inherit its track run"
        )
    command = _module_command(context, "fisheye.analysis.plot_track_kinematics")
    command.extend(
        (
            "--track-kinematics-run",
            track_run,
            "--track-id",
            "0",
            "--offline-only",
            "--write-zarr-artifacts",
            "--swim-bout-run",
            context.dependency_run("swim_bouts"),
            "--speed-level",
            "exponential",
        )
    )
    return tuple(command)


def _bout_kinematics_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.bout_kinematics",
    )
    command.extend(
        (
            "--compute",
            "--run-name",
            context.output_run,
            "--output-shard-rows",
            "262144",
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
            "--",
            "--track-kinematics-run",
            context.dependency_run("track_kinematics"),
            "--track-scope",
            "offline",
            "--track-id",
            "0",
            "--swim-bout-run",
            context.dependency_run("swim_bouts"),
            "--speed-level",
            "exponential",
            "--include-eye-gaze",
            "--eye-angle-run",
            context.dependency_run("eye_angles"),
            "--eye-angle-family",
            "gaze",
            "--pre-post-mode",
            "interbout_epoch",
            "--pre-window-s",
            "0.05",
            "--post-window-s",
            "0.05",
            "--within-window",
            "bout_start_end",
            "--physical-active-signal-level",
            "filtered",
            "--physical-active-threshold-mm-s",
            "0.01",
            "--physical-active-boundary-constraint",
            "search_with_margin",
            "--physical-active-boundary-margin-s",
            "0.05",
            "--layout",
            "compact_tabular_v2",
            "--write-zarr-artifacts",
        )
    )
    return tuple(command)


def _eye_angle_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.eye_angles",
    )
    command.extend(
        (
            "--subject-shape-run",
            context.dependency_run("subject_shape"),
            "--run-name",
            context.output_run,
            "--chunk-size",
            "8192",
            "--execution-backend",
            "dask_worker_chunks",
            "--scheduler",
            "processes",
            "--num-workers",
            str(context.num_workers),
            "--angle-chunk-rows",
            "4096",
            "--angle-chunk-columns",
            "16",
            "--output-shard-rows",
            "131072",
            "--angle-shard-columns",
            "32",
            "--shard-workers",
            str(min(context.num_workers, 16)),
            "--native-threads",
            "1",
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
        )
    )
    return tuple(command)


def _subject_shape_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.subject_shape",
    )
    command.extend(
        (
            "--refined-run",
            context.dependency_run("refined_subject_masks"),
            "--run-name",
            context.output_run,
            "--block-rows",
            "1024",
            "--output-shard-rows",
            "131072",
            "--execution-backend",
            "dask_worker_chunks",
            "--scheduler",
            "processes",
            "--num-workers",
            str(context.num_workers),
            "--shard-copy-workers",
            str(min(context.num_workers, 16)),
            "--native-threads",
            "1",
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
        )
    )
    return tuple(command)


def _tail_kinematics_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.tail_kinematics",
    )
    command.extend(
        (
            "--shape-run",
            context.dependency_run("subject_shape"),
            "--run-name",
            context.output_run,
            "--block-rows",
            "16384",
            "--output-shard-rows",
            "262144",
            "--execution-backend",
            "process_shards",
            "--num-workers",
            str(context.num_workers),
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
        )
    )
    return tuple(command)


def _tail_posture_view_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis.tail_posture_view_runs",
    )
    command.extend(
        (
            "--subject-shape-run",
            context.dependency_run("subject_shape"),
            "--run-name",
            context.output_run,
        )
    )
    tail_run = context.dependency_runs.get("tail_kinematics")
    if tail_run is not None:
        command.extend(("--source-tail-kinematics-run", tail_run))
    command.append("--json")
    return tuple(command)


def _bout_classification_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis.megabouts_classifier",
    )
    command.extend(
        (
            "--run-name",
            context.output_run,
            "--tail-posture-view-run",
            context.dependency_run("tail_posture_view"),
            "--track-kinematics-run",
            context.dependency_run("track_kinematics"),
            "--track-scope",
            "offline",
            "--track-id",
            "0",
            "--swim-bout-run",
            context.dependency_run("swim_bouts"),
            "--speed-level",
            "exponential",
            "--device",
            "auto",
            "--json",
        )
    )
    return tuple(command)


def _stimulus_response_command(context: StageCommandContext) -> tuple[str, ...]:
    command = _module_command(
        context,
        "fisheye.analysis_workflows.materializers.stimulus_response",
    )
    command.extend(
        (
            "--run-name",
            context.output_run,
            "--layout",
            "compact_tabular_v3",
            "--copy-backend",
            "rsync",
            "--apply",
            "--json",
            "--",
            "--track-kinematics-run",
            context.dependency_run("track_kinematics"),
            "--stimulus-run",
            context.dependency_run("stimulus"),
            "--bout-run",
            context.dependency_run("swim_bouts"),
        )
    )
    return tuple(command)


def _eye_trace_export_command(context: StageCommandContext) -> tuple[str, ...]:
    if context.export_root is None or context.scratch_root is None:
        raise WorkflowExecutionError(
            "eye-trace export requires explicit export and node-local scratch roots"
        )
    if context.node.temporal_policy != {
        "resolution": "framewise",
        "source_authority": "framewise_zarr",
    }:
        raise WorkflowExecutionError(
            "eye-trace export supports only the exact framewise temporal policy"
        )
    command = _module_command(
        context,
        "fisheye.utils.export_eye_trace_samples",
    )
    command.extend(
        (
            "--eye-angle-run",
            context.dependency_run("eye_angles"),
            "--output-root",
            str(context.export_root),
            "--export-run-id",
            context.output_run,
            "--scratch-root",
            str(context.scratch_root),
            "--row-group-rows",
            "65536",
            "--json",
        )
    )
    return tuple(command)


STAGE_COMMAND_BUILDERS: Mapping[str, StageCommandBuilder] = MappingProxyType(
    {
        "track_kinematics": _track_kinematics_command,
        "swim_bouts": _swim_bout_command,
        "track_kinematics_visualization": _track_kinematics_visualization_command,
        "bout_kinematics": _bout_kinematics_command,
        "eye_angles": _eye_angle_command,
        "subject_shape": _subject_shape_command,
        "tail_kinematics": _tail_kinematics_command,
        "tail_posture_view": _tail_posture_view_command,
        "bout_classification": _bout_classification_command,
        "stimulus_response": _stimulus_response_command,
    }
)

EXPORT_COMMAND_BUILDERS: Mapping[str, StageCommandBuilder] = MappingProxyType(
    {
        "eye_traces": _eye_trace_export_command,
    }
)


def _safe_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not SAFE_RUN_NAME.fullmatch(name):
        raise WorkflowExecutionError(f"unsafe {label}: {value!r}")
    return name


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def default_output_run_name(*, execution_id: str, node_id: str) -> str:
    return _safe_name(f"{node_id}_{execution_id}", label="generated output run name")


def build_workflow_execution_plan(
    workflow: AnalysisWorkflow,
    workflow_plan: WorkflowPlan,
    *,
    zarr_path: str | Path,
    execution_id: str,
    num_workers: int,
    output_run_overrides: Mapping[str, str] | None = None,
    export_run_overrides: Mapping[str, str] | None = None,
    export_root: str | Path | None = None,
    scratch_root: str | Path | None = None,
    python_executable: str,
) -> WorkflowExecutionPlan:
    """Render exact commands for runnable analysis nodes in topological order."""

    execution_id = _safe_name(execution_id, label="execution id")
    if int(num_workers) < 1:
        raise WorkflowExecutionError("num_workers must be a positive integer")
    if not workflow_plan.ready:
        blocked = [
            node.node_id for node in workflow_plan.nodes if node.action == "blocked"
        ]
        raise WorkflowExecutionError("workflow plan is blocked: " + ", ".join(blocked))

    planned_by_id = workflow_plan.node_by_id
    workflow_nodes = workflow.node_by_id
    overrides: dict[str, str] = {}
    for raw_stage_id, run_name in dict(output_run_overrides or {}).items():
        stage_id = canonical_stage_id(str(raw_stage_id))
        if stage_id in overrides:
            raise WorkflowExecutionError(
                f"output-run overrides repeat canonical stage {stage_id!r}"
            )
        overrides[stage_id] = _safe_name(
            run_name,
            label=f"output run for {stage_id}",
        )
    run_nodes = [node for node in workflow_plan.nodes if node.action == "run"]
    run_stage_ids = {node.stage_id for node in run_nodes if node.stage_id is not None}
    unused_overrides = sorted(set(overrides) - run_stage_ids)
    if unused_overrides:
        raise WorkflowExecutionError(
            "output-run override does not select a runnable stage: "
            + ", ".join(unused_overrides)
        )

    export_overrides: dict[str, str] = {}
    for raw_node_id, run_name in dict(export_run_overrides or {}).items():
        node_id = str(raw_node_id).strip()
        workflow_node = workflow_nodes.get(node_id)
        if workflow_node is None or workflow_node.kind != "export":
            raise WorkflowExecutionError(
                f"export-run override does not name an export node: {node_id!r}"
            )
        if node_id in export_overrides:
            raise WorkflowExecutionError(
                f"export-run override repeats node {node_id!r}"
            )
        export_overrides[node_id] = _safe_name(
            run_name,
            label=f"export run for {node_id}",
        )
    run_export_node_ids = {node.node_id for node in run_nodes if node.kind == "export"}
    unimplemented_export_nodes = sorted(
        run_export_node_ids - set(EXPORT_COMMAND_BUILDERS)
    )
    if unimplemented_export_nodes:
        raise WorkflowExecutionError(
            "export node(s) require an execution adapter that is not implemented: "
            + ", ".join(unimplemented_export_nodes)
        )
    unused_export_overrides = sorted(set(export_overrides) - run_export_node_ids)
    if unused_export_overrides:
        raise WorkflowExecutionError(
            "export-run override does not select a runnable export: "
            + ", ".join(unused_export_overrides)
        )
    resolved_export_root = (
        Path(export_root).expanduser().resolve() if export_root is not None else None
    )
    resolved_scratch_root = (
        Path(scratch_root).expanduser().resolve() if scratch_root is not None else None
    )
    if run_export_node_ids and (
        resolved_export_root is None or resolved_scratch_root is None
    ):
        raise WorkflowExecutionError(
            "runnable export nodes require explicit export_root and scratch_root"
        )
    if run_export_node_ids:
        assert resolved_export_root is not None and resolved_scratch_root is not None
        source_root = Path(zarr_path).expanduser().resolve()
        if _path_is_within(resolved_export_root, source_root):
            raise WorkflowExecutionError(
                "export_root must remain outside the analysis Zarr"
            )
        if _path_is_within(resolved_scratch_root, source_root):
            raise WorkflowExecutionError(
                "scratch_root must remain outside the analysis Zarr"
            )
        if _path_is_within(resolved_scratch_root, resolved_export_root):
            raise WorkflowExecutionError(
                "scratch_root must remain outside the immutable export root"
            )
        if _path_is_within(resolved_export_root, resolved_scratch_root):
            raise WorkflowExecutionError(
                "export_root must remain outside node-local scratch"
            )

    output_runs: dict[str, str] = {}
    export_runs: dict[str, str] = {}
    resolved_runs: dict[str, str] = {}
    for node in workflow_plan.nodes:
        if node.action == "reuse" and node.stage_id is not None:
            if not node.selected_run:
                raise WorkflowExecutionError(
                    f"reused node {node.node_id!r} has no selected run"
                )
            resolved_runs[node.node_id] = node.selected_run
        elif node.action == "run" and node.kind == "export":
            output_run = export_overrides.get(node.node_id) or default_output_run_name(
                execution_id=execution_id,
                node_id=node.node_id,
            )
            export_runs[node.node_id] = output_run
            resolved_runs[node.node_id] = output_run
        elif node.action == "run" and node.stage_id is not None:
            workflow_node = workflow_nodes[node.node_id]
            if workflow_node.output_run_from is not None:
                if node.stage_id in overrides:
                    raise WorkflowExecutionError(
                        f"stage {node.stage_id!r} inherits its output run from "
                        f"{workflow_node.output_run_from!r}; --output-run is not allowed"
                    )
                try:
                    output_run = resolved_runs[workflow_node.output_run_from]
                except KeyError as exc:
                    raise WorkflowExecutionError(
                        f"node {node.node_id!r} cannot resolve inherited output run "
                        f"from {workflow_node.output_run_from!r}"
                    ) from exc
            else:
                output_run = overrides.get(node.stage_id) or default_output_run_name(
                    execution_id=execution_id,
                    node_id=node.node_id,
                )
            output_runs[node.stage_id] = output_run
            resolved_runs[node.node_id] = output_run

    commands: list[StageCommand] = []
    for node_id in workflow_plan.execution_order:
        node = planned_by_id[node_id]
        if node.kind == "export":
            builder = EXPORT_COMMAND_BUILDERS.get(node.node_id)
            if builder is None:
                raise WorkflowExecutionError(
                    f"export node {node_id!r} requires an execution adapter that is not implemented"
                )
            output_run = export_runs[node.node_id]
            output_kind = "parquet_export"
            assert resolved_export_root is not None
            output_root = str(resolved_export_root)
        elif node.kind in {"analysis", "visualization"} and node.stage_id is not None:
            builder = STAGE_COMMAND_BUILDERS.get(node.stage_id)
            if builder is None:
                raise WorkflowExecutionError(
                    f"stage {node.stage_id!r} has no command adapter"
                )
            output_run = output_runs[node.stage_id]
            output_kind = "zarr_stage"
            output_root = str(Path(zarr_path).expanduser().resolve())
        else:
            raise WorkflowExecutionError(
                f"node {node_id!r} requires an execution adapter that is not implemented"
            )
        dependency_runs = {
            dependency: resolved_runs[dependency]
            for dependency in workflow_nodes[node_id].depends_on
            if dependency in resolved_runs
        }
        context = StageCommandContext(
            zarr_path=Path(zarr_path),
            node=node,
            output_run=output_run,
            dependency_runs=MappingProxyType(dependency_runs),
            python_executable=str(python_executable),
            num_workers=int(num_workers),
            export_root=resolved_export_root,
            scratch_root=resolved_scratch_root,
        )
        commands.append(
            StageCommand(
                node_id=node.node_id,
                node_kind=node.kind,
                stage_id=node.stage_id,
                output_run=output_run,
                output_kind=output_kind,
                output_root=output_root,
                dependency_runs=MappingProxyType(dependency_runs),
                argv=builder(context),
            )
        )

    return WorkflowExecutionPlan(
        execution_id=execution_id,
        workflow_plan=workflow_plan,
        output_runs=MappingProxyType(output_runs),
        export_runs=MappingProxyType(export_runs),
        commands=tuple(commands),
    )


__all__ = [
    "EXECUTION_SCHEMA_ID",
    "EXECUTION_LEGACY_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "EXPORT_COMMAND_BUILDERS",
    "STAGE_COMMAND_BUILDERS",
    "StageCommand",
    "WorkflowExecutionError",
    "WorkflowExecutionPlan",
    "build_workflow_execution_plan",
    "default_output_run_name",
]
