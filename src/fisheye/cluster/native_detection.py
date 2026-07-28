"""Composable artifact-array to native canonical-detection workflow."""

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
    RUNTIME_USER_TOKEN,
)


@dataclass(frozen=True)
class NativeDetectionModelSpec:
    set_id: str
    run_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class NativeDetectionAuthoritySpec:
    record_ref: str
    record_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeDetectionAuthoritySpec:
        if set(value) != {"record_ref", "record_sha256"}:
            raise ValueError("Native detection authority requires exact ref/digest fields.")
        return cls(
            record_ref=str(value["record_ref"]),
            record_sha256=str(value["record_sha256"]),
        )


@dataclass(frozen=True)
class NativeDetectionClipSpec:
    work_unit_id: str
    clip_id: str
    clip_index: int
    camera_serial: str
    video_path: Path
    artifact_run_id: str
    artifact_group_path: str
    report_path: Path

    @classmethod
    def from_plan_work_unit(
        cls,
        value: Mapping[str, Any],
        *,
        report_path: Path,
    ) -> NativeDetectionClipSpec:
        run_names = value.get("run_names")
        zarr_paths = value.get("zarr_paths")
        source = value.get("source")
        if not all(isinstance(item, Mapping) for item in (run_names, zarr_paths, source)):
            raise ValueError("Detection plan work unit has incomplete typed sections.")
        artifact_run_id = str(run_names.get("detect") or "")
        artifact_group_path = str(
            zarr_paths.get("detection_artifact_target_group_path") or ""
        )
        if not artifact_group_path:
            raise ValueError(
                "Detection plan lacks detection_artifact_target_group_path."
            )
        return cls(
            work_unit_id=str(value.get("work_unit_id") or ""),
            clip_id=str(value.get("clip_id") or ""),
            clip_index=int(value.get("clip_index")),
            camera_serial=str(value.get("camera_serial") or ""),
            video_path=Path(str(source.get("video_path") or "")),
            artifact_run_id=artifact_run_id,
            artifact_group_path=artifact_group_path,
            report_path=report_path,
        )

    def __post_init__(self) -> None:
        parts = Path(self.artifact_group_path).parts
        if len(parts) < 2 or parts[-2] != "detection_artifact_runs":
            raise ValueError(
                "Native clip output must be an explicit detection_artifact_runs path."
            )
        if parts[-1] != self.artifact_run_id:
            raise ValueError("Artifact group path and run id disagree.")


@dataclass(frozen=True)
class NativeDetectionFragmentInputs:
    workflow_id: str
    family: str
    target_id: str
    recording_identity: str
    recording_dir: Path
    analysis_zarr: Path
    repo: Path
    run_root: Path
    canonical_run_id: str
    n_frames: int
    source_width: int
    source_height: int
    source_frame_authority: NativeDetectionAuthoritySpec
    source_pixel_authority: NativeDetectionAuthoritySpec
    producer_version: str
    clips: tuple[NativeDetectionClipSpec, ...]
    model: NativeDetectionModelSpec
    detect_array_concurrency: int = 8
    resume_existing_artifacts: bool = False
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("Native detection requires at least one clip artifact.")
        if self.detect_array_concurrency <= 0:
            raise ValueError("detect_array_concurrency must be positive.")
        for name in ("n_frames", "source_width", "source_height"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive exact integer.")
        for name, values in (
            ("work_unit_id", [clip.work_unit_id for clip in self.clips]),
            ("clip_id", [clip.clip_id for clip in self.clips]),
            ("clip_index", [clip.clip_index for clip in self.clips]),
            ("artifact_group_path", [clip.artifact_group_path for clip in self.clips]),
            ("report_path", [str(clip.report_path) for clip in self.clips]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Native detection clip {name} values must be unique.")


@dataclass(frozen=True)
class NativeDetectionFragmentOutputs:
    target_id: str
    artifact_group_paths: tuple[str, ...]
    canonical_run_id: str
    canonical_group_path: str
    publication_receipt_path: Path
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "artifact_group_paths": list(self.artifact_group_paths),
            "canonical_run_id": self.canonical_run_id,
            "canonical_group_path": self.canonical_group_path,
            "publication_receipt_path": str(self.publication_receipt_path),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "native_run_manifest_schema_version": 2,
            "logical_schema_version": 1,
            "selector_eligible": False,
        }


@dataclass(frozen=True)
class NativeDetectionWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: NativeDetectionFragmentOutputs


def build_native_detection_fragment(
    inputs: NativeDetectionFragmentInputs,
) -> NativeDetectionWorkflowModule:
    """Plan clip artifact inference followed by one atomic native publication."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    config_path = inputs.repo / "configs" / "fisheye" / "yolo_detect_config.yaml"
    detect_resources = LsfResources(
        queue="gpu_l4",
        ncores=8,
        mem_gb=120,
        gpus=1,
        walltime="2:00",
    )
    reuse_resources = LsfResources(queue="short", ncores=1, mem_gb=8, walltime="1:00")
    publish_resources = LsfResources(
        queue="short",
        ncores=4,
        mem_gb=32,
        walltime="1:00",
        span_hosts=1,
    )
    tasks = []
    for clip in inputs.clips:
        command = [
            "scripts/py",
            "-m",
            "fisheye.utils.run_clipped_detection_work_unit",
            "--video",
            str(clip.video_path),
            "--target-zarr",
            str(inputs.analysis_zarr),
            "--target-group-path",
            clip.artifact_group_path,
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
            inputs.workflow_id,
            "--recording-id",
            inputs.recording_identity,
            "--clip-id",
            clip.clip_id,
            "--clip-index",
            str(clip.clip_index),
            "--camera-serial",
            clip.camera_serial,
            "--recording-frame-index",
            str(inputs.recording_dir / "recording_frame_index.parquet"),
            "--run-name",
            clip.artifact_run_id,
            "--report",
            str(clip.report_path),
            "--batch-size",
            "16",
            "--decode-backend",
            "pynvvc_luma_rgb",
        ]
        if inputs.resume_existing_artifacts:
            command.append("--reuse-existing")
        tasks.append(
            build_execution_task(
                run_root=inputs.run_root,
                task_key=f"detect_artifact:{target_safe}:{clip.clip_id}",
                stage=(
                    "detect_artifact_reuse"
                    if inputs.resume_existing_artifacts
                    else "detect_artifact"
                ),
                command=command,
                expected_outputs=(
                    inputs.analysis_zarr / clip.artifact_group_path / "zarr.json",
                    clip.report_path,
                ),
                cleanup_paths=(
                    ()
                    if inputs.resume_existing_artifacts
                    else (
                        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}_"
                        f"{RUNTIME_JOB_INDEX_TOKEN}/palette_clipped_detection",
                    )
                ),
                array_indexed=True,
            )
        )
    array_key = f"detect_artifact_array:{target_safe}"
    array_job = build_task_group_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=array_key,
        stage=(
            "detect_artifact_reuse"
            if inputs.resume_existing_artifacts
            else "detect_artifact"
        ),
        tasks=tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=inputs.detect_array_concurrency,
        resources=reuse_resources if inputs.resume_existing_artifacts else detect_resources,
        upstream=inputs.upstream_job_keys,
    )

    receipt = inputs.run_root / "native_detection" / f"{target_safe}.publication.json"
    scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "native_canonical_detection"
    )
    candidate = f"{scratch}/{inputs.canonical_run_id}.zarr"
    assemble = [
        "scripts/py",
        "-m",
        "fisheye.utils.assemble_clipped_native_detection",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--recording-frame-index",
        str(inputs.recording_dir / "recording_frame_index.parquet"),
        "--recording-identity",
        inputs.recording_identity,
        "--n-frames",
        str(inputs.n_frames),
        "--source-width",
        str(inputs.source_width),
        "--source-height",
        str(inputs.source_height),
        "--run-id",
        inputs.canonical_run_id,
        "--candidate-zarr",
        candidate,
        "--producer-id",
        "fisheye.detection.detect_yolo",
        "--producer-version",
        inputs.producer_version,
        "--source-frame-record-ref",
        inputs.source_frame_authority.record_ref,
        "--source-frame-record-sha256",
        inputs.source_frame_authority.record_sha256,
        "--source-pixel-record-ref",
        inputs.source_pixel_authority.record_ref,
        "--source-pixel-record-sha256",
        inputs.source_pixel_authority.record_sha256,
        "--model-artifact-sha256",
        inputs.model.sha256,
        "--workflow-id",
        inputs.workflow_id,
        "--result-json",
        str(receipt),
    ]
    for clip in inputs.clips:
        assemble.extend(("--work-unit-report", str(clip.report_path)))
    publish_key = f"detect_native_publish:{target_safe}"
    publish_job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=publish_key,
        stage="detect_native_publish",
        command=chain_commands((("mkdir", "-p", scratch), tuple(assemble))),
        resources=publish_resources,
        upstream=(array_key,),
        expected_outputs=(
            inputs.analysis_zarr
            / "detect_runs"
            / inputs.canonical_run_id
            / "zarr.json",
            receipt,
        ),
        cleanup_paths=(scratch,),
    )

    artifact_key = f"canonical_detection:{target_safe}"
    outputs = NativeDetectionFragmentOutputs(
        target_id=inputs.target_id,
        artifact_group_paths=tuple(clip.artifact_group_path for clip in inputs.clips),
        canonical_run_id=inputs.canonical_run_id,
        canonical_group_path=f"detect_runs/{inputs.canonical_run_id}",
        publication_receipt_path=receipt,
        terminal_job_key=publish_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"native_detection:{target_safe}",
        jobs=(array_job, publish_job),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "native_detection",
            "target_id": inputs.target_id,
            "artifact_namespace": "detection_artifact_runs",
            "canonical_namespace": "detect_runs",
            "selector_activation": "deferred",
            "registry_update": False,
            "outputs": outputs.to_json(),
        },
    )
    return NativeDetectionWorkflowModule(fragment=fragment, outputs=outputs)


def compose_native_detection_workflow(
    *,
    workflow_id: str,
    family: str,
    modules: tuple[NativeDetectionWorkflowModule, ...],
    external_inputs: tuple[str, ...] = (),
) -> LsfWorkflow:
    if not modules:
        raise ValueError("A native detection workflow requires at least one module.")
    return compose_lsf_workflow(
        workflow_id=workflow_id,
        family=family,
        fragments=tuple(module.fragment for module in modules),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "native_canonical_detection",
            "target_count": len(modules),
            "selector_activation": "deferred",
            "outputs": [module.outputs.to_json() for module in modules],
        },
    )


__all__ = [
    "NativeDetectionAuthoritySpec",
    "NativeDetectionClipSpec",
    "NativeDetectionFragmentInputs",
    "NativeDetectionFragmentOutputs",
    "NativeDetectionModelSpec",
    "NativeDetectionWorkflowModule",
    "build_native_detection_fragment",
    "compose_native_detection_workflow",
]
