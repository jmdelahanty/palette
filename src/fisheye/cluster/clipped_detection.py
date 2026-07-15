"""Composable clipped detection, quality, refinement, and publication subgraph."""

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
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
)


@dataclass(frozen=True)
class DetectionModelSpec:
    """Exact registry-backed model identity consumed by detection."""

    set_id: str
    run_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class DetectionClipSpec:
    """One clip-local detection and refined-detection lineage."""

    clip_id: str
    clip_index: int
    camera_serial: str
    video_path: Path
    detect_run: str
    detect_group_path: str
    refined_detect_run: str
    refined_detect_group_path: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DetectionClipSpec:
        return cls(
            clip_id=str(value["clip_id"]),
            clip_index=int(value["clip_index"]),
            camera_serial=str(value["camera_serial"]),
            video_path=Path(str(value["video_path"])),
            detect_run=str(value["detect_run"]),
            detect_group_path=str(value["detect_group_path"]),
            refined_detect_run=str(value["refined_detect_run"]),
            refined_detect_group_path=str(value["refined_detect_group_path"]),
        )


@dataclass(frozen=True)
class DetectionFragmentInputs:
    """Typed inputs required to plan the complete detection subgraph."""

    workflow_id: str
    family: str
    target_id: str
    target_label: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    repo: Path
    run_root: Path
    detection_plan_path: Path
    collection_id: str
    quality_source_run: str
    quality_run: str
    clips: tuple[DetectionClipSpec, ...]
    model: DetectionModelSpec
    expected_subject_count: int = 1
    resume_existing_detections: bool = False
    detect_array_concurrency: int = 8
    refine_bundle_concurrency: int = 4
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("A detection fragment requires at least one clip.")
        if self.expected_subject_count <= 0:
            raise ValueError("expected_subject_count must be positive.")
        if self.detect_array_concurrency <= 0 or self.refine_bundle_concurrency <= 0:
            raise ValueError("Detection concurrency limits must be positive.")
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("Detection clip ids must be unique within a target.")
        detect_groups = [clip.detect_group_path for clip in self.clips]
        refined_groups = [clip.refined_detect_group_path for clip in self.clips]
        if len(set(detect_groups)) != len(detect_groups):
            raise ValueError("Detection output group paths must be unique.")
        if len(set(refined_groups)) != len(refined_groups):
            raise ValueError("Refined-detection output group paths must be unique.")


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
class DetectionWorkflowModule:
    """A composable LSF fragment paired with its typed artifact outputs."""

    fragment: LsfWorkflowFragment
    outputs: DetectionFragmentOutputs


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
        walltime="2:00",
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
    fragment = LsfWorkflowFragment(
        fragment_id=f"detection:{target_safe}",
        jobs=tuple(jobs),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "clipped_detection",
            "target_id": inputs.target_id,
            "clip_count": len(inputs.clips),
            "resume_existing_detections": inputs.resume_existing_detections,
            "outputs": outputs.to_json(),
        },
    )
    return DetectionWorkflowModule(fragment=fragment, outputs=outputs)


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
        fragments=tuple(module.fragment for module in modules),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "detection_only",
            "target_count": len(modules),
            "outputs": [module.outputs.to_json() for module in modules],
        },
    )


__all__ = [
    "DetectionClipSpec",
    "DetectionFragmentInputs",
    "DetectionFragmentOutputs",
    "DetectionModelSpec",
    "DetectionWorkflowModule",
    "build_detection_fragment",
    "compose_detection_workflow",
]
