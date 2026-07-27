"""Composable detection fragments for clipped recording work units.

The target and work-unit inputs use the layout-neutral production contract.
The current command renderer remains clipped-specific until the whole-video
publisher is adapted in the next migration checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


@dataclass(frozen=True)
class DetectionModelSpec:
    """Exact registry-backed model identity consumed by detection."""

    set_id: str
    run_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class DetectionWorkUnitSpec:
    """Detection run identities bound to one neutral video work unit."""

    work_unit: VideoWorkUnit
    detect_run: str
    detect_group_path: str
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
        if not isinstance(self.work_unit, VideoWorkUnit):
            raise TypeError("Detection work_unit must be a VideoWorkUnit.")
        for field_name in (
            "detect_run",
            "detect_group_path",
            "refined_detect_run",
            "refined_detect_group_path",
        ):
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
class DetectionWorkflowModule:
    """Layout-neutral raw and postprocess fragments with typed outputs."""

    fragments: tuple[LsfWorkflowFragment, ...]
    raw_outputs: RawDetectionFragmentOutputs
    outputs: DetectionFragmentOutputs

    @property
    def jobs(self) -> tuple[LsfJob, ...]:
        return tuple(job for fragment in self.fragments for job in fragment.jobs)


def build_detection_fragment(inputs: DetectionFragmentInputs) -> DetectionWorkflowModule:
    """Plan detect -> quality -> refine -> finalized collection for one target."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    config_path = inputs.repo / "configs" / "fisheye" / "yolo_detect_config.yaml"
    refine_config = inputs.repo / "configs" / "fisheye" / "default.yaml"
    detect_gpu = LsfResources(
        queue="gpu_l4", ncores=8, mem_gb=120, gpus=1, walltime="2:00"
    )
    detect_reuse_cpu = LsfResources(
        queue="short", ncores=1, mem_gb=8, walltime="1:00"
    )
    cpu = LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00")
    refine_bundle_cpu = LsfResources(
        queue="short",
        ncores=4 * int(inputs.refine_bundle_concurrency),
        mem_gb=32,
        walltime="1:00",
        span_hosts=1,
    )
    jobs = []

    detect_array_key = f"detect_array:{target_safe}"
    detect_tasks = []
    for clip in inputs.clips:
        detect_key = f"detect:{target_safe}:{clip.clip_id}"
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
            inputs.recording_id,
            "--clip-id",
            clip.clip_id,
            "--clip-index",
            str(clip.clip_index),
            "--camera-serial",
            clip.camera_serial,
            "--recording-frame-index",
            str(inputs.recording_dir / "recording_frame_index.parquet"),
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
                task_key=detect_key,
                stage=("detect_reuse" if inputs.resume_existing_detections else "detect"),
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
    jobs.append(
        build_task_group_job(
            workflow_id=inputs.workflow_id,
            family=inputs.family,
            repo=inputs.repo,
            run_root=inputs.run_root,
            job_key=detect_array_key,
            stage=("detect_reuse" if inputs.resume_existing_detections else "detect"),
            tasks=detect_tasks,
            mode=LsfExecutionMode.ARRAY,
            max_concurrent=inputs.detect_array_concurrency,
            resources=detect_reuse_cpu if inputs.resume_existing_detections else detect_gpu,
            upstream=inputs.upstream_job_keys,
        )
    )

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

    raw_artifact_key = f"raw_detection_work_units:{target_safe}"
    raw_outputs = RawDetectionFragmentOutputs(
        target_id=inputs.target_id,
        raw_detection_group_paths=tuple(
            clip.detect_group_path for clip in inputs.clips
        ),
        terminal_job_key=detect_array_key,
        artifact_key=raw_artifact_key,
    )
    raw_fragment = LsfWorkflowFragment(
        fragment_id=f"raw_detection:{target_safe}",
        jobs=(jobs[0],),
        requires=inputs.required_artifacts,
        provides=(raw_artifact_key,),
        metadata={
            "module": "raw_detection",
            "target_id": inputs.target_id,
            "recording_layout": inputs.target.layout.value,
            "work_unit_count": len(inputs.work_units),
            "resume_existing_detections": inputs.resume_existing_detections,
            "outputs": raw_outputs.to_json(),
        },
    )

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
        jobs=tuple(jobs[1:]),
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
    "RawDetectionFragmentOutputs",
    "build_detection_fragment",
    "compose_detection_workflow",
]
