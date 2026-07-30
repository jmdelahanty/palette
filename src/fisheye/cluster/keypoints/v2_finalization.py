"""Composable clipped keypoint-v2 terminal-evidence and finalization fragment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    LsfWorkflowFragment,
)


@dataclass(frozen=True)
class ClipKeypointV2FinalizationInput:
    clip_id: str
    clip_index: int
    source_group_path: str
    input_package_manifest_path: Path

    def __post_init__(self) -> None:
        if not str(self.clip_id).strip() or "/" in str(self.clip_id):
            raise ValueError("clip_id must be one nonempty path-safe component.")
        if type(self.clip_index) is not int or self.clip_index < 0:
            raise ValueError("clip_index must be a nonnegative exact integer.")
        group = str(self.source_group_path).strip().strip("/")
        if not group or any(part in {"", ".", ".."} for part in group.split("/")):
            raise ValueError("source_group_path must be one safe relative Zarr path.")
        object.__setattr__(self, "source_group_path", group)


@dataclass(frozen=True)
class ClippedKeypointV2FinalizationInputs:
    workflow_id: str
    family: str
    target_id: str
    analysis_zarr: Path
    crop_run_id: str
    clips: tuple[ClipKeypointV2FinalizationInput, ...]
    pose_binding_path: Path
    preprocessing_path: Path
    bundle_root: Path
    raw_run_id: str
    quality_run_id: str
    refined_run_id: str
    body_frame_run_id: str
    recording_identity: str
    refined_lineage_id: str
    refined_snapshot_id: str
    repo: Path
    run_root: Path
    crop_archive: Path | None = None
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    receipt_array_concurrency: int = 8

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("Clipped keypoint-v2 finalization requires clip inputs.")
        clip_ids = [clip.clip_id for clip in self.clips]
        clip_indices = [clip.clip_index for clip in self.clips]
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("Clip ids must be unique.")
        if sorted(clip_indices) != list(range(len(self.clips))):
            raise ValueError("Clip indices must form [0, clip_count).")
        if self.receipt_array_concurrency <= 0:
            raise ValueError("receipt_array_concurrency must be positive.")


@dataclass(frozen=True)
class ClippedKeypointV2FinalizationOutputs:
    target_id: str
    crop_archive: Path
    crop_run_id: str
    clip_receipt_paths: tuple[Path, ...]
    bundle_root: Path
    result_path: Path
    finalization_receipt_path: Path
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "crop_archive": str(self.crop_archive),
            "crop_run_id": self.crop_run_id,
            "clip_receipt_paths": [str(path) for path in self.clip_receipt_paths],
            "bundle_root": str(self.bundle_root),
            "result_path": str(self.result_path),
            "finalization_receipt_path": str(self.finalization_receipt_path),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "selector_eligible": False,
            "registry_updated": False,
            "physical_layout_source": "shared_byte_planners",
        }


@dataclass(frozen=True)
class ClippedKeypointV2FinalizationModule:
    fragment: LsfWorkflowFragment
    outputs: ClippedKeypointV2FinalizationOutputs


def build_clipped_keypoint_v2_finalization_fragment(
    inputs: ClippedKeypointV2FinalizationInputs,
) -> ClippedKeypointV2FinalizationModule:
    """Plan terminal sidecars followed by one recording-level v2 bundle."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    receipt_dir = inputs.run_root / "keypoint_v2_clip_receipts" / target_safe
    terminal_key = f"keypoint_v2_terminal_receipts:{target_safe}"
    terminal_tasks = []
    receipt_paths: list[Path] = []
    for clip in sorted(inputs.clips, key=lambda value: value.clip_index):
        clip_safe = safe_component(clip.clip_id, default="clip", max_length=56)
        receipt = receipt_dir / f"{clip.clip_index:04d}_{clip_safe}.json"
        receipt_paths.append(receipt)
        command: list[str] = [
            "scripts/py",
            "-m",
            "fisheye.utils.write_keypoint_clip_terminal_receipt",
            "--analysis-zarr",
            str(inputs.analysis_zarr),
        ]
        if inputs.crop_archive is not None:
            command.extend(("--crop-archive", str(inputs.crop_archive)))
        command.extend(
            (
                "--crop-run",
                inputs.crop_run_id,
                "--source-group",
                clip.source_group_path,
                "--clip-id",
                clip.clip_id,
                "--clip-index",
                str(clip.clip_index),
                "--pose-binding",
                str(inputs.pose_binding_path),
                "--preprocessing",
                str(inputs.preprocessing_path),
                "--input-package-manifest",
                str(clip.input_package_manifest_path),
                "--output",
                str(receipt),
            )
        )
        terminal_tasks.append(
            build_execution_task(
                run_root=inputs.run_root,
                task_key=f"keypoint_v2_terminal:{target_safe}:{clip_safe}",
                stage="keypoint_v2_terminal_receipt",
                command=tuple(command),
                expected_outputs=(receipt,),
                array_indexed=True,
            )
        )
    receipt_resources = LsfResources(
        queue="short", ncores=2, mem_gb=16, walltime="1:00", span_hosts=1
    )
    terminal_job = build_task_group_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=terminal_key,
        stage="keypoint_v2_terminal_receipt",
        tasks=terminal_tasks,
        mode=LsfExecutionMode.ARRAY,
        max_concurrent=inputs.receipt_array_concurrency,
        resources=receipt_resources,
        upstream=inputs.upstream_job_keys,
    )

    final_key = f"keypoint_v2_finalize:{target_safe}"
    result_path = inputs.run_root / "keypoint_v2_finalization" / f"{target_safe}.json"
    command: list[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.finalize_clipped_keypoint_v2_bundle",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
    ]
    if inputs.crop_archive is not None:
        command.extend(("--crop-archive", str(inputs.crop_archive)))
    command.extend(
        [
            "--crop-run",
            inputs.crop_run_id,
            "--pose-binding",
            str(inputs.pose_binding_path),
            "--preprocessing",
            str(inputs.preprocessing_path),
            "--bundle-root",
            str(inputs.bundle_root),
            "--raw-run",
            inputs.raw_run_id,
            "--quality-run",
            inputs.quality_run_id,
            "--refined-run",
            inputs.refined_run_id,
            "--body-frame-run",
            inputs.body_frame_run_id,
            "--recording-identity",
            inputs.recording_identity,
            "--refined-lineage-id",
            inputs.refined_lineage_id,
            "--refined-snapshot-id",
            inputs.refined_snapshot_id,
            "--result-json",
            str(result_path),
        ]
    )
    for receipt in receipt_paths:
        command.extend(("--clip-receipt", str(receipt)))
    final_job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=final_key,
        stage="keypoint_v2_finalize",
        command=chain_commands(
            (
                ("mkdir", "-p", str(inputs.bundle_root.parent)),
                tuple(command),
            )
        ),
        resources=LsfResources(
            queue="short", ncores=4, mem_gb=48, walltime="2:00", span_hosts=1
        ),
        upstream=(terminal_key,),
        expected_outputs=(
            result_path,
            inputs.bundle_root / "finalization_receipt.json",
            inputs.bundle_root / "raw_keypoints.zarr" / "zarr.json",
            inputs.bundle_root / "keypoint_quality.zarr" / "zarr.json",
            inputs.bundle_root / "refined_keypoints.zarr" / "zarr.json",
            inputs.bundle_root / "body_frame.zarr" / "zarr.json",
        ),
    )
    artifact_key = f"selector_ineligible_keypoint_v2_chain:{target_safe}"
    outputs = ClippedKeypointV2FinalizationOutputs(
        target_id=inputs.target_id,
        crop_archive=(
            inputs.analysis_zarr if inputs.crop_archive is None else inputs.crop_archive
        ),
        crop_run_id=inputs.crop_run_id,
        clip_receipt_paths=tuple(receipt_paths),
        bundle_root=inputs.bundle_root,
        result_path=result_path,
        finalization_receipt_path=inputs.bundle_root / "finalization_receipt.json",
        terminal_job_key=final_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"keypoint_v2_finalization:{target_safe}",
        jobs=(terminal_job, final_job),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "clipped_keypoint_v2_finalization",
            "target_id": inputs.target_id,
            "clip_count": len(inputs.clips),
            "compute_partition": "clip_local",
            "publication_partition": "complete_recording_snapshot",
            "selector_activation": "none_direct_path_only",
            "registry_update": False,
            "outputs": outputs.to_json(),
        },
    )
    return ClippedKeypointV2FinalizationModule(fragment=fragment, outputs=outputs)


__all__ = [
    "ClipKeypointV2FinalizationInput",
    "ClippedKeypointV2FinalizationInputs",
    "ClippedKeypointV2FinalizationModule",
    "ClippedKeypointV2FinalizationOutputs",
    "build_clipped_keypoint_v2_finalization_fragment",
]
