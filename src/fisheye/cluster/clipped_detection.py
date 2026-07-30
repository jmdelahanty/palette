"""Composable raw detection plus clipped detection-postprocessing fragments.

Raw detection dispatches from the layout-neutral recording contract: clipped
work units use artifact build/import publication, while whole videos use the
node-local atomic run publisher.  Collection quality/refinement remains a
clipped-only fragment until its source adapter is generalized separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_lsf import (
    build_execution_task,
    build_job,
    build_task_group_job,
    chain_commands,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
)
from fisheye.cluster.recording_layout import (
    RecordingLayout,
    RecordingTarget,
    VideoWorkUnit,
)
from fisheye.shared.detection_candidate import (
    DEFAULT_DETECT_FRAME_SHARD_ROWS,
    DEFAULT_DETECT_ROW_SHARD_ROWS,
)


@dataclass(frozen=True)
class DetectionModelSpec:
    """Exact registry-backed model identity consumed by detection."""

    set_id: str
    run_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class RawDetectionWorkUnitSpec:
    """One raw-detection run bound to a neutral video work unit."""

    work_unit: VideoWorkUnit
    detect_run: str
    detect_group_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.work_unit, VideoWorkUnit):
            raise TypeError("Detection work_unit must be a VideoWorkUnit.")
        for field_name in ("detect_run", "detect_group_path"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"Detection {field_name} cannot be empty.")
            object.__setattr__(self, field_name, value)

    @property
    def clip_id(self) -> str:
        """Compatibility label consumed by current clipped stage commands."""

        return self.work_unit.source_partition_id

    @property
    def clip_index(self) -> int:
        return self.work_unit.source_partition_index

    @property
    def camera_serial(self) -> str:
        return self.work_unit.camera_serial

    @property
    def video_path(self) -> Path:
        return self.work_unit.video_path


@dataclass(frozen=True)
class DetectionWorkUnitSpec(RawDetectionWorkUnitSpec):
    """Raw and refined run identities for clipped postprocessing."""

    refined_detect_run: str
    refined_detect_group_path: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        work_unit: VideoWorkUnit,
    ) -> DetectionWorkUnitSpec:
        return cls(
            work_unit=work_unit,
            detect_run=str(value["detect_run"]),
            detect_group_path=str(value["detect_group_path"]),
            refined_detect_run=str(value["refined_detect_run"]),
            refined_detect_group_path=str(value["refined_detect_group_path"]),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in ("refined_detect_run", "refined_detect_group_path"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"Detection {field_name} cannot be empty.")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class RawDetectionFragmentInputs:
    """Layout-neutral inputs required to plan raw detection publication."""

    workflow_id: str
    family: str
    target_label: str
    target: RecordingTarget
    repo: Path
    run_root: Path
    work_units: tuple[RawDetectionWorkUnitSpec, ...]
    model: DetectionModelSpec
    registry_path: Path | None = None
    resume_existing_detections: bool = False
    detect_array_concurrency: int = 8
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, RecordingTarget):
            raise TypeError("Raw detection target must be a RecordingTarget.")
        units = tuple(self.work_units)
        object.__setattr__(self, "work_units", units)
        if not units:
            raise ValueError("Raw detection requires at least one work unit.")
        if any(not isinstance(unit, RawDetectionWorkUnitSpec) for unit in units):
            raise TypeError(
                "Raw detection work_units must be RawDetectionWorkUnitSpec values."
            )
        observed_work_units = tuple(unit.work_unit for unit in units)
        if observed_work_units != self.target.work_units:
            raise ValueError(
                "Raw detection work units must exactly match target work-unit order."
            )
        group_paths = [unit.detect_group_path for unit in units]
        if len(set(group_paths)) != len(group_paths):
            raise ValueError("Raw detection output group paths must be unique.")
        if int(self.detect_array_concurrency) <= 0:
            raise ValueError("Raw detection array concurrency must be positive.")
        object.__setattr__(
            self,
            "detect_array_concurrency",
            int(self.detect_array_concurrency),
        )
        if self.registry_path is not None:
            object.__setattr__(
                self,
                "registry_path",
                self.registry_path.expanduser().resolve(),
            )

        if self.target.layout is RecordingLayout.WHOLE_VIDEO:
            if len(units) != 1:
                raise ValueError("Whole-video raw detection requires one work unit.")
            expected_group = f"detect_runs/{units[0].detect_run}"
            if units[0].detect_group_path != expected_group:
                raise ValueError(
                    "Whole-video raw detection must publish to "
                    f"{expected_group!r}, got {units[0].detect_group_path!r}."
                )
            if self.registry_path is None:
                raise ValueError(
                    "Whole-video atomic detection requires an explicit registry path."
                )
            if self.resume_existing_detections:
                raise ValueError(
                    "Whole-video raw-detection reuse needs a separate validation "
                    "contract and is not enabled."
                )

    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def analysis_zarr(self) -> Path:
        return self.target.analysis_zarr


@dataclass(frozen=True)
class DetectionFragmentInputs:
    """Typed inputs required to plan the complete detection subgraph."""

    workflow_id: str
    family: str
    target_label: str
    target: RecordingTarget
    repo: Path
    run_root: Path
    detection_plan_path: Path
    collection_id: str
    quality_source_run: str
    quality_run: str
    work_units: tuple[DetectionWorkUnitSpec, ...]
    model: DetectionModelSpec
    resume_existing_detections: bool = False
    detect_array_concurrency: int = 8
    refine_bundle_concurrency: int = 4
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, RecordingTarget):
            raise TypeError("Detection target must be a RecordingTarget.")
        if self.target.layout is not RecordingLayout.CLIPPED_COLLECTION:
            raise ValueError(
                "The current detection command renderer requires clipped work units."
            )
        if not self.work_units:
            raise ValueError("A detection fragment requires at least one clip.")
        if self.detect_array_concurrency <= 0 or self.refine_bundle_concurrency <= 0:
            raise ValueError("Detection concurrency limits must be positive.")
        clip_camera_keys = [
            (clip.clip_id, clip.camera_serial) for clip in self.work_units
        ]
        if len(set(clip_camera_keys)) != len(clip_camera_keys):
            raise ValueError(
                "Detection clip/camera identities must be unique within a target."
            )
        observed_work_units = tuple(unit.work_unit for unit in self.work_units)
        if observed_work_units != self.target.work_units:
            raise ValueError(
                "Detection work units must exactly match target work-unit order."
            )
        detect_groups = [clip.detect_group_path for clip in self.work_units]
        refined_groups = [clip.refined_detect_group_path for clip in self.work_units]
        if len(set(detect_groups)) != len(detect_groups):
            raise ValueError("Detection output group paths must be unique.")
        if len(set(refined_groups)) != len(refined_groups):
            raise ValueError("Refined-detection output group paths must be unique.")

    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def recording_id(self) -> str:
        return self.target.recording_id

    @property
    def recording_dir(self) -> Path:
        return self.target.recording_dir

    @property
    def analysis_zarr(self) -> Path:
        return self.target.analysis_zarr

    @property
    def expected_subject_count(self) -> int:
        return self.target.expected_subject_count

    @property
    def clips(self) -> tuple[DetectionWorkUnitSpec, ...]:
        """Compatibility view while clipped commands retain clip terminology."""

        return self.work_units


@dataclass(frozen=True)
class DetectionFragmentOutputs:
    """Validated artifacts and dependency handles published by the subgraph."""

    target_id: str
    collection_id: str
    raw_detection_group_paths: tuple[str, ...]
    quality_source_group_path: str
    quality_group_path: str
    refined_detection_group_paths: tuple[str, ...]
    finalized_collection_group_path: str
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "collection_id": self.collection_id,
            "raw_detection_group_paths": list(self.raw_detection_group_paths),
            "quality_source_group_path": self.quality_source_group_path,
            "quality_group_path": self.quality_group_path,
            "refined_detection_group_paths": list(self.refined_detection_group_paths),
            "finalized_collection_group_path": self.finalized_collection_group_path,
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class RawDetectionFragmentOutputs:
    """Raw detection artifacts produced before quality or refinement."""

    target_id: str
    raw_detection_group_paths: tuple[str, ...]
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "raw_detection_group_paths": list(self.raw_detection_group_paths),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class RawDetectionWorkflowModule:
    """One independently composable raw-detection publication fragment."""

    fragment: LsfWorkflowFragment
    outputs: RawDetectionFragmentOutputs


@dataclass(frozen=True)
class RawDetectionCohortWorkflowModule:
    """One bounded scheduler array publishing several whole-video runs."""

    fragment: LsfWorkflowFragment
    outputs: tuple[RawDetectionFragmentOutputs, ...]


@dataclass(frozen=True)
class DetectionWorkflowModule:
    """Layout-neutral raw and postprocess fragments with typed outputs."""

    fragments: tuple[LsfWorkflowFragment, ...]
    raw_outputs: RawDetectionFragmentOutputs
    outputs: DetectionFragmentOutputs

    @property
    def jobs(self) -> tuple[LsfJob, ...]:
        return tuple(job for fragment in self.fragments for job in fragment.jobs)


def _clipped_raw_detection_job(
    inputs: RawDetectionFragmentInputs,
    *,
    target_safe: str,
) -> LsfJob:
    config_path = inputs.repo / "configs" / "fisheye" / "yolo_detect_config.yaml"
    detect_gpu = LsfResources(
        queue="gpu_l4", ncores=8, mem_gb=120, gpus=1, walltime="2:00"
    )
    detect_reuse_cpu = LsfResources(queue="short", ncores=1, mem_gb=8, walltime="1:00")
    detect_tasks = []
    for clip in inputs.work_units:
        report = (
            inputs.run_root
            / "targets"
            / target_safe
            / "detection_reports"
            / f"{clip.clip_id}.json"
        )
        detect_command = [
            "scripts/py",
            "-m",
            "fisheye.utils.run_clipped_detection_work_unit",
            "--video",
            str(clip.video_path),
            "--target-zarr",
            str(inputs.analysis_zarr),
            "--target-group-path",
            clip.detect_group_path,
            "--model",
            str(inputs.model.path),
            "--model-sha256",
            inputs.model.sha256,
            "--model-registry-set-id",
            inputs.model.set_id,
            "--model-registry-run-id",
            inputs.model.run_id,
            "--config",
            str(config_path),
            "--workflow-id",
            inputs.target_label,
            "--recording-id",
            inputs.target.recording_id,
            "--clip-id",
            clip.clip_id,
            "--clip-index",
            str(clip.clip_index),
            "--camera-serial",
            clip.camera_serial,
            "--recording-frame-index",
            str(clip.work_unit.frame_mapping.recording_frame_index),
            "--run-name",
            clip.detect_run,
            "--report",
            str(report),
            "--batch-size",
            "16",
            "--decode-backend",
            "pynvvc_luma_rgb",
        ]
        if inputs.resume_existing_detections:
            detect_command.append("--reuse-existing")
        detect_tasks.append(
            build_execution_task(
                run_root=inputs.run_root,
                task_key=f"detect:{target_safe}:{clip.clip_id}",
                stage=(
                    "detect_reuse" if inputs.resume_existing_detections else "detect"
                ),
                command=detect_command,
                expected_outputs=(
                    inputs.analysis_zarr / clip.detect_group_path / "zarr.json",
                    report,
                ),
                cleanup_paths=(
                    ()
                    if inputs.resume_existing_detections
                    else (
                        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}_"
                        f"{RUNTIME_JOB_INDEX_TOKEN}/palette_clipped_detection",
                    )
                ),
                array_indexed=True,
            )
        )
    return build_task_group_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=f"detect_array:{target_safe}",
        stage=("detect_reuse" if inputs.resume_existing_detections else "detect"),
        tasks=detect_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=inputs.detect_array_concurrency,
        resources=detect_reuse_cpu if inputs.resume_existing_detections else detect_gpu,
        upstream=inputs.upstream_job_keys,
    )


def _whole_video_raw_detection_command(
    inputs: RawDetectionFragmentInputs,
    *,
    target_safe: str,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    unit = inputs.work_units[0]
    assert inputs.registry_path is not None
    result_json = (
        inputs.run_root
        / "targets"
        / target_safe
        / "detection_reports"
        / "whole_video.json"
    )
    command: tuple[str, ...] = (
        "scripts/py",
        "-m",
        "fisheye.utils.run_detection_local_publish",
        "--zarr",
        str(inputs.analysis_zarr),
        "--video",
        str(unit.video_path),
        "--model",
        str(inputs.model.path),
        "--model-sha256",
        inputs.model.sha256,
        "--model-run-id",
        inputs.model.run_id,
        "--model-set-id",
        inputs.model.set_id,
        "--run-name",
        unit.detect_run,
        "--registry",
        str(inputs.registry_path),
        "--config",
        str(inputs.repo / "configs" / "fisheye" / "yolo_detect_config.yaml"),
        "--batch-size",
        "16",
        "--resize-dims",
        "640",
        "640",
        "--decode-backend",
        "pynvvc_nv12_rgb",
        "--detect-row-shard-rows",
        str(DEFAULT_DETECT_ROW_SHARD_ROWS),
        "--detect-frame-shard-rows",
        str(DEFAULT_DETECT_FRAME_SHARD_ROWS),
        "--result-json",
        str(result_json),
    )
    return command, (
        inputs.analysis_zarr / unit.detect_group_path / "zarr.json",
        result_json,
    )


def _whole_video_raw_detection_job(
    inputs: RawDetectionFragmentInputs,
    *,
    target_safe: str,
) -> LsfJob:
    command, expected_outputs = _whole_video_raw_detection_command(
        inputs,
        target_safe=target_safe,
    )
    return build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=f"detect:{target_safe}",
        stage="detect",
        command=command,
        resources=LsfResources(
            queue="gpu_l4",
            ncores=8,
            mem_gb=120,
            gpus=1,
            walltime="2:00",
        ),
        upstream=inputs.upstream_job_keys,
        expected_outputs=expected_outputs,
    )


def build_raw_detection_fragment(
    inputs: RawDetectionFragmentInputs,
) -> RawDetectionWorkflowModule:
    """Build the layout-specific raw publisher behind one artifact contract."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    if inputs.target.layout is RecordingLayout.CLIPPED_COLLECTION:
        job = _clipped_raw_detection_job(inputs, target_safe=target_safe)
    elif inputs.target.layout is RecordingLayout.WHOLE_VIDEO:
        job = _whole_video_raw_detection_job(inputs, target_safe=target_safe)
    else:  # pragma: no cover - enum construction already fails closed
        raise ValueError(f"Unsupported recording layout: {inputs.target.layout!r}")
    artifact_key = f"raw_detection_work_units:{target_safe}"
    outputs = RawDetectionFragmentOutputs(
        target_id=inputs.target_id,
        raw_detection_group_paths=tuple(
            unit.detect_group_path for unit in inputs.work_units
        ),
        terminal_job_key=job.job_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"raw_detection:{target_safe}",
        jobs=(job,),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "raw_detection",
            "target_id": inputs.target_id,
            "recording_layout": inputs.target.layout.value,
            "work_unit_count": len(inputs.work_units),
            "resume_existing_detections": inputs.resume_existing_detections,
            "publication_policy": (
                "node_local_complete_run_then_atomic_prfs_publication_v1"
                if inputs.target.layout is RecordingLayout.WHOLE_VIDEO
                else "clip_artifact_build_import_validate_v1"
            ),
            "outputs": outputs.to_json(),
        },
    )
    return RawDetectionWorkflowModule(fragment=fragment, outputs=outputs)


def build_whole_video_raw_detection_cohort_fragment(
    inputs: Sequence[RawDetectionFragmentInputs],
    *,
    max_concurrent: int,
) -> RawDetectionCohortWorkflowModule:
    """Pack whole-video atomic publishers into one bounded LSF array.

    The scheduler array is an orchestration optimization only.  Every element
    retains one recording-bound source video, one analysis Zarr, one complete
    node-local candidate, and one atomic publication boundary.
    """

    cohort = tuple(inputs)
    if not cohort:
        raise ValueError("A whole-video detection cohort requires at least one target.")
    if int(max_concurrent) <= 0:
        raise ValueError("Whole-video detection cohort concurrency must be positive.")
    if any(item.target.layout is not RecordingLayout.WHOLE_VIDEO for item in cohort):
        raise ValueError(
            "Whole-video detection cohorts cannot contain clipped targets."
        )

    first = cohort[0]
    invariant_fields = (
        "workflow_id",
        "family",
        "repo",
        "run_root",
        "model",
        "registry_path",
        "upstream_job_keys",
        "required_artifacts",
    )
    for item in cohort[1:]:
        mismatched = [
            name
            for name in invariant_fields
            if getattr(item, name) != getattr(first, name)
        ]
        if mismatched:
            raise ValueError(
                "Whole-video cohort members must share scheduler/publication "
                f"bindings; mismatched fields: {mismatched!r}."
            )

    target_ids = [item.target_id for item in cohort]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Whole-video detection cohort target ids must be unique.")
    analysis_zarrs = [item.analysis_zarr for item in cohort]
    if len(set(analysis_zarrs)) != len(analysis_zarrs):
        raise ValueError("Whole-video detection cohort analysis Zarrs must be unique.")

    tasks = []
    target_safe_values: list[str] = []
    for item in cohort:
        target_safe = safe_component(item.target_id, default="target", max_length=56)
        target_safe_values.append(target_safe)
        command, expected_outputs = _whole_video_raw_detection_command(
            item,
            target_safe=target_safe,
        )
        tasks.append(
            build_execution_task(
                run_root=item.run_root,
                task_key=f"detect:{target_safe}",
                stage="detect",
                command=command,
                expected_outputs=expected_outputs,
                array_indexed=True,
            )
        )

    workflow_safe = safe_component(
        first.workflow_id,
        default="whole_video_detect",
        max_length=56,
    )
    job = build_task_group_job(
        workflow_id=first.workflow_id,
        family=first.family,
        repo=first.repo,
        run_root=first.run_root,
        job_key=f"detect_array:{workflow_safe}",
        stage="detect",
        tasks=tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=int(max_concurrent),
        resources=LsfResources(
            queue="gpu_l4",
            ncores=8,
            mem_gb=120,
            gpus=1,
            walltime="2:00",
        ),
        upstream=first.upstream_job_keys,
    )
    outputs = tuple(
        RawDetectionFragmentOutputs(
            target_id=item.target_id,
            raw_detection_group_paths=tuple(
                unit.detect_group_path for unit in item.work_units
            ),
            terminal_job_key=job.job_key,
            artifact_key=f"raw_detection_work_units:{target_safe}",
        )
        for item, target_safe in zip(cohort, target_safe_values, strict=True)
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"raw_detection_cohort:{workflow_safe}",
        jobs=(job,),
        requires=first.required_artifacts,
        provides=tuple(output.artifact_key for output in outputs),
        metadata={
            "module": "raw_detection_cohort",
            "recording_layout": RecordingLayout.WHOLE_VIDEO.value,
            "target_count": len(cohort),
            "scheduler_execution": "bounded_lsf_array",
            "max_concurrent": min(int(max_concurrent), len(cohort)),
            "publication_policy": (
                "one_complete_node_local_run_then_atomic_prfs_publication_per_element_v1"
            ),
            "outputs": [output.to_json() for output in outputs],
        },
    )
    return RawDetectionCohortWorkflowModule(fragment=fragment, outputs=outputs)


def compose_raw_detection_workflow(
    *,
    workflow_id: str,
    family: str,
    modules: tuple[RawDetectionWorkflowModule, ...],
    external_inputs: tuple[str, ...] = (),
) -> LsfWorkflow:
    """Compose raw detection for one or more recording-layout targets."""

    if not modules:
        raise ValueError("A raw-detection workflow requires at least one module.")
    return compose_lsf_workflow(
        workflow_id=workflow_id,
        family=family,
        fragments=tuple(module.fragment for module in modules),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "raw_detection_only",
            "target_count": len(modules),
            "outputs": [module.outputs.to_json() for module in modules],
        },
    )


def build_detection_fragment(
    inputs: DetectionFragmentInputs,
    *,
    raw_module: RawDetectionWorkflowModule | None = None,
) -> DetectionWorkflowModule:
    """Plan detect -> quality -> refine -> finalized collection for one target."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    refine_config = inputs.repo / "configs" / "fisheye" / "default.yaml"
    cpu = LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00")
    refine_bundle_cpu = LsfResources(
        queue="short",
        ncores=4 * int(inputs.refine_bundle_concurrency),
        mem_gb=32,
        walltime="1:00",
        span_hosts=1,
    )
    if raw_module is None:
        raw_module = build_raw_detection_fragment(
            RawDetectionFragmentInputs(
                workflow_id=inputs.workflow_id,
                family=inputs.family,
                target_label=inputs.target_label,
                target=inputs.target,
                repo=inputs.repo,
                run_root=inputs.run_root,
                work_units=inputs.work_units,
                model=inputs.model,
                resume_existing_detections=inputs.resume_existing_detections,
                detect_array_concurrency=inputs.detect_array_concurrency,
                upstream_job_keys=inputs.upstream_job_keys,
                required_artifacts=inputs.required_artifacts,
            )
        )
    expected_raw_paths = tuple(unit.detect_group_path for unit in inputs.work_units)
    if raw_module.outputs.target_id != inputs.target_id:
        raise ValueError("Raw and postprocess detection modules target different recordings.")
    if raw_module.outputs.raw_detection_group_paths != expected_raw_paths:
        raise ValueError(
            "Raw detection outputs do not match the postprocess source groups."
        )
    detect_array_key = raw_module.outputs.terminal_job_key
    jobs: list[LsfJob] = []

    quality_source_key = f"detect_quality_source:{target_safe}"
    quality_source_group_path = f"detect_collection_sources/{inputs.quality_source_run}"
    jobs.append(
        build_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=quality_source_key,
            stage="detect_quality_source",
            command=(
                "scripts/py",
                "-m",
                "fisheye.utils.materialize_clipped_detect_quality_source",
                str(inputs.analysis_zarr),
                "--plan",
                str(inputs.detection_plan_path),
                "--output-run",
                inputs.quality_source_run,
                "--recording-frame-index",
                str(inputs.recording_dir / "recording_frame_index.parquet"),
                "--shard-rows",
                "131072",
                "--inner-rows",
                "16384",
                "--apply",
                "--json",
            ),
            resources=cpu,
            upstream=(detect_array_key,),
            expected_outputs=(
                inputs.analysis_zarr / quality_source_group_path / "zarr.json",
            ),
        )
    )

    quality_key = f"detect_quality:{target_safe}"
    quality_group_path = f"detect_quality_runs/{inputs.quality_run}"
    quality_work_dir = (
        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/detect_quality"
    )
    jobs.append(
        build_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=quality_key,
            stage="detect_quality",
            command=(
                "scripts/py",
                "-m",
                "fisheye.refinement.detect_quality_collection",
                str(inputs.analysis_zarr),
                "--source-group",
                quality_source_group_path,
                "--output-run",
                inputs.quality_run,
                "--expected-subject-count",
                str(inputs.expected_subject_count),
                "--threshold-mode",
                "scaled",
                "--jump-threshold",
                "100.0",
                "--threshold-reference-width",
                "640.0",
                "--blip-gap-threshold",
                "10",
                "--shard-rows",
                "131072",
                "--row-chunk-rows",
                "16384",
                "--frame-chunk-rows",
                "16384",
                "--workers",
                "4",
                "--work-dir",
                quality_work_dir,
                "--apply",
                "--json",
            ),
            resources=cpu,
            upstream=(quality_source_key,),
            expected_outputs=(inputs.analysis_zarr / quality_group_path / "zarr.json",),
            cleanup_paths=(quality_work_dir,),
        )
    )

    refine_tasks = []
    for clip in inputs.clips:
        refine = (
            "scripts/py",
            "-m",
            "fisheye.refinement.refine_detect",
            str(inputs.analysis_zarr),
            "--detect-run",
            clip.detect_run,
            "--detect-family-path",
            str(Path(clip.detect_group_path).parent),
            "--refined-family-path",
            str(Path(clip.refined_detect_group_path).parent),
            "--quality-group-path",
            quality_group_path,
            "--config",
            str(refine_config),
            "--run-name",
            clip.refined_detect_run,
            "--per-frame-top-k",
            "1",
        )
        validate = (
            "scripts/py",
            "-m",
            "fisheye.utils.validate_refined_detect_run",
            str(inputs.analysis_zarr),
            "--target-group-path",
            clip.refined_detect_group_path,
        )
        refine_tasks.append(
            build_execution_task(
                run_root=inputs.run_root,
                task_key=f"detect_refine:{target_safe}:{clip.clip_id}",
                stage="detect_refine",
                command=chain_commands((refine, validate)),
                expected_outputs=(
                    inputs.analysis_zarr / clip.refined_detect_group_path / "zarr.json",
                ),
                array_indexed=False,
            )
        )
    refine_bundle_key = f"detect_refine_bundle:{target_safe}"
    jobs.append(
        build_task_group_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=refine_bundle_key,
            stage="detect_refine",
            tasks=refine_tasks,
            mode=LsfExecutionMode.BUNDLE,
            max_concurrent=inputs.refine_bundle_concurrency,
            resources=refine_bundle_cpu,
            upstream=(quality_key,),
        )
    )

    collection_key = f"detect_collection:{target_safe}"
    collection_report = (
        inputs.run_root / "targets" / target_safe / "detection_collection.json"
    )
    finalized_collection_group_path = (
        f"experiment_index/finalized_runs/{inputs.collection_id}"
    )
    jobs.append(
        build_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=collection_key,
            stage="detect_collection",
            command=(
                "scripts/py",
                "-m",
                "fisheye.utils.finalize_clipped_detect_refine_workflow",
                str(inputs.detection_plan_path),
                "--collection-id",
                inputs.collection_id,
                "--detect-quality-run",
                inputs.quality_run,
                "--detect-quality-group-path",
                quality_group_path,
                "--no-require-stage-status",
                "--apply",
                "--output-json",
                str(collection_report),
            ),
            resources=cpu,
            upstream=(refine_bundle_key,),
            expected_outputs=(
                inputs.analysis_zarr / finalized_collection_group_path / "zarr.json",
                collection_report,
            ),
        )
    )

    raw_outputs = raw_module.outputs
    raw_artifact_key = raw_outputs.artifact_key
    raw_fragment = raw_module.fragment

    artifact_key = f"finalized_detection_collection:{target_safe}"
    outputs = DetectionFragmentOutputs(
        target_id=inputs.target_id,
        collection_id=inputs.collection_id,
        raw_detection_group_paths=tuple(
            clip.detect_group_path for clip in inputs.clips
        ),
        quality_source_group_path=quality_source_group_path,
        quality_group_path=quality_group_path,
        refined_detection_group_paths=tuple(
            clip.refined_detect_group_path for clip in inputs.clips
        ),
        finalized_collection_group_path=finalized_collection_group_path,
        terminal_job_key=collection_key,
        artifact_key=artifact_key,
    )
    postprocess_fragment = LsfWorkflowFragment(
        fragment_id=f"detection_postprocess:{target_safe}",
        jobs=tuple(jobs),
        requires=(raw_artifact_key,),
        provides=(artifact_key,),
        metadata={
            "module": "detection_postprocess",
            "target_id": inputs.target_id,
            "work_unit_count": len(inputs.work_units),
            "raw_detection_inputs": raw_outputs.to_json(),
            "outputs": outputs.to_json(),
        },
    )
    return DetectionWorkflowModule(
        fragments=(raw_fragment, postprocess_fragment),
        raw_outputs=raw_outputs,
        outputs=outputs,
    )


def compose_detection_workflow(
    *,
    workflow_id: str,
    family: str,
    modules: tuple[DetectionWorkflowModule, ...],
    external_inputs: tuple[str, ...] = (),
) -> LsfWorkflow:
    """Compose one or more detection modules into a detection-only workflow."""

    if not modules:
        raise ValueError("A detection-only workflow requires at least one module.")
    return compose_lsf_workflow(
        workflow_id=workflow_id,
        family=family,
        fragments=tuple(
            fragment for module in modules for fragment in module.fragments
        ),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "detection_only",
            "target_count": len(modules),
            "outputs": [module.outputs.to_json() for module in modules],
        },
    )


__all__ = [
    "DetectionFragmentInputs",
    "DetectionFragmentOutputs",
    "DetectionModelSpec",
    "DetectionWorkUnitSpec",
    "DetectionWorkflowModule",
    "RawDetectionFragmentInputs",
    "RawDetectionFragmentOutputs",
    "RawDetectionWorkUnitSpec",
    "RawDetectionCohortWorkflowModule",
    "RawDetectionWorkflowModule",
    "build_detection_fragment",
    "build_raw_detection_fragment",
    "build_whole_video_raw_detection_cohort_fragment",
    "compose_detection_workflow",
    "compose_raw_detection_workflow",
]
