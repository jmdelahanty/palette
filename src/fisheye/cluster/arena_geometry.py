"""Composable selected-geometry and registered detection-gate DAG fragments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import LsfResources, LsfWorkflowFragment
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN
from fisheye.cluster.recording_layout import RecordingTarget


@dataclass(frozen=True)
class ArenaGeometrySelectionFragmentInputs:
    workflow_id: str
    family: str
    target: RecordingTarget
    repo: Path
    run_root: Path
    candidate_run: str
    selection_run: str
    selected_by: str
    decision_reason: str
    decision_source: str = "manual_review"
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, RecordingTarget):
            raise TypeError("Geometry selection target must be a RecordingTarget.")
        for name in (
            "workflow_id",
            "family",
            "candidate_run",
            "selection_run",
            "selected_by",
            "decision_reason",
            "decision_source",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Geometry selection {name} cannot be empty.")


@dataclass(frozen=True)
class RegisteredDetectionGateFragmentInputs:
    workflow_id: str
    family: str
    target: RecordingTarget
    repo: Path
    run_root: Path
    source_detection_group_path: str
    selection_run: str
    output_run: str
    inner_rows: int = 16_384
    shard_rows: int = 131_072
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, RecordingTarget):
            raise TypeError("Registered gate target must be a RecordingTarget.")
        for name in (
            "workflow_id",
            "family",
            "source_detection_group_path",
            "selection_run",
            "output_run",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Registered gate {name} cannot be empty.")
        if int(self.inner_rows) <= 0 or int(self.shard_rows) <= 0:
            raise ValueError("Registered gate chunk and shard rows must be positive.")


@dataclass(frozen=True)
class ArenaGeometrySelectionFragmentOutputs:
    target_id: str
    selection_run: str
    selection_group_path: str
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "selection_run": self.selection_run,
            "selection_group_path": self.selection_group_path,
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class RegisteredDetectionGateFragmentOutputs:
    target_id: str
    source_detection_group_path: str
    selection_run: str
    gate_run: str
    gate_group_path: str
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "source_detection_group_path": self.source_detection_group_path,
            "selection_run": self.selection_run,
            "gate_run": self.gate_run,
            "gate_group_path": self.gate_group_path,
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class ArenaGeometrySelectionWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: ArenaGeometrySelectionFragmentOutputs


@dataclass(frozen=True)
class RegisteredDetectionGateWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: RegisteredDetectionGateFragmentOutputs


def build_arena_geometry_selection_fragment(
    inputs: ArenaGeometrySelectionFragmentInputs,
) -> ArenaGeometrySelectionWorkflowModule:
    """Plan one explicit reviewed selection, independent of video layout."""

    safe = safe_component(inputs.target.target_id, default="target", max_length=56)
    job_key = f"arena_geometry_selection:{safe}"
    group_path = f"analysis/arena_geometry_selection/{inputs.selection_run}"
    scratch = (
        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/arena_geometry_selection"
    )
    job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=job_key,
        stage="arena_geometry_selection",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.publish_arena_geometry_selection",
            str(inputs.target.analysis_zarr),
            "--candidate-run",
            inputs.candidate_run,
            "--selected-by",
            inputs.selected_by,
            "--decision-reason",
            inputs.decision_reason,
            "--decision-source",
            inputs.decision_source,
            "--expected-selection-run",
            inputs.selection_run,
            "--scratch-root",
            scratch,
            "--apply",
        ),
        resources=LsfResources(queue="short", ncores=1, mem_gb=4, walltime="1:00"),
        upstream=inputs.upstream_job_keys,
        expected_outputs=(inputs.target.analysis_zarr / group_path / "zarr.json",),
        cleanup_paths=(scratch,),
    )
    artifact_key = f"arena_geometry_selection:{safe}"
    outputs = ArenaGeometrySelectionFragmentOutputs(
        target_id=inputs.target.target_id,
        selection_run=inputs.selection_run,
        selection_group_path=group_path,
        terminal_job_key=job_key,
        artifact_key=artifact_key,
    )
    return ArenaGeometrySelectionWorkflowModule(
        fragment=LsfWorkflowFragment(
            fragment_id=artifact_key,
            jobs=(job,),
            requires=inputs.required_artifacts,
            provides=(artifact_key,),
            metadata={
                "module": "arena_geometry_selection",
                "recording_layout": inputs.target.layout.value,
                "candidate_mutated": False,
                "outputs": outputs.to_json(),
            },
        ),
        outputs=outputs,
    )


def build_registered_detection_gate_fragment(
    inputs: RegisteredDetectionGateFragmentInputs,
) -> RegisteredDetectionGateWorkflowModule:
    """Plan one keyed gate table for a whole-video or collection source."""

    safe = safe_component(inputs.target.target_id, default="target", max_length=56)
    job_key = f"registered_detection_gate:{safe}"
    group_path = f"analysis/detection_gate_runs/{inputs.output_run}"
    scratch = f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/registered_detection_gate"
    job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=job_key,
        stage="registered_detection_gate",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.materialize_registered_detection_gate",
            str(inputs.target.analysis_zarr),
            "--source-group",
            inputs.source_detection_group_path,
            "--selection-run",
            inputs.selection_run,
            "--output-run",
            inputs.output_run,
            "--inner-rows",
            str(int(inputs.inner_rows)),
            "--shard-rows",
            str(int(inputs.shard_rows)),
            "--scratch-root",
            scratch,
            "--apply",
        ),
        resources=LsfResources(queue="short", ncores=2, mem_gb=8, walltime="1:00"),
        upstream=inputs.upstream_job_keys,
        expected_outputs=(inputs.target.analysis_zarr / group_path / "zarr.json",),
        cleanup_paths=(scratch,),
    )
    artifact_key = f"registered_detection_gate:{safe}"
    outputs = RegisteredDetectionGateFragmentOutputs(
        target_id=inputs.target.target_id,
        source_detection_group_path=inputs.source_detection_group_path,
        selection_run=inputs.selection_run,
        gate_run=inputs.output_run,
        gate_group_path=group_path,
        terminal_job_key=job_key,
        artifact_key=artifact_key,
    )
    return RegisteredDetectionGateWorkflowModule(
        fragment=LsfWorkflowFragment(
            fragment_id=artifact_key,
            jobs=(job,),
            requires=inputs.required_artifacts,
            provides=(artifact_key,),
            metadata={
                "module": "registered_detection_gate",
                "recording_layout": inputs.target.layout.value,
                "source_detection_group_path": inputs.source_detection_group_path,
                "raw_detections_preserved": True,
                "row_identity": "instance_key",
                "outputs": outputs.to_json(),
            },
        ),
        outputs=outputs,
    )


__all__ = [
    "ArenaGeometrySelectionFragmentInputs",
    "ArenaGeometrySelectionWorkflowModule",
    "RegisteredDetectionGateFragmentInputs",
    "RegisteredDetectionGateWorkflowModule",
    "build_arena_geometry_selection_fragment",
    "build_registered_detection_gate_fragment",
]
