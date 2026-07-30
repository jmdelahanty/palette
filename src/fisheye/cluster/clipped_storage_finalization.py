"""Composable strict refined-detection and crop-v2 recording finalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_lsf import build_job, chain_commands
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.keypoints.v2_finalization import (
    ClippedKeypointV2FinalizationInputs,
    ClippedKeypointV2FinalizationModule,
    build_clipped_keypoint_v2_finalization_fragment,
)
from fisheye.cluster.lsf import LsfResources, LsfWorkflowFragment


@dataclass(frozen=True)
class StrictClipRefinedDetectionInput:
    clip_index: int
    clip_id: str
    archive: Path
    run_id: str

    def __post_init__(self) -> None:
        if type(self.clip_index) is not int or self.clip_index < 0:
            raise ValueError("clip_index must be a nonnegative exact integer.")
        for name in ("clip_id", "run_id"):
            value = str(getattr(self, name)).strip()
            if not value or "/" in value:
                raise ValueError(f"{name} must be one path-safe component.")


@dataclass(frozen=True)
class ClippedStorageFinalizationInputs:
    workflow_id: str
    family: str
    target_id: str
    analysis_zarr: Path
    canonical_archive: Path
    canonical_run_id: str
    clips: tuple[StrictClipRefinedDetectionInput, ...]
    clipped_binding_path: Path
    bundle_root: Path
    refined_run_id: str
    refined_lineage_id: str
    refined_snapshot_id: str
    crop_run_id: str
    recording_identity: str
    crop_purpose: str
    roi_width: int
    roi_height: int
    camera_id: str
    repo: Path
    run_root: Path
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("Clipped storage finalization requires clip inputs.")
        ordered = sorted(self.clips, key=lambda item: item.clip_index)
        if [item.clip_index for item in ordered] != list(range(len(ordered))):
            raise ValueError("Clip indices must form [0, clip_count).")
        if len({item.clip_id for item in ordered}) != len(ordered):
            raise ValueError("Clip ids must be unique.")
        for name in ("roi_width", "roi_height"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer.")


@dataclass(frozen=True)
class ClippedStorageFinalizationOutputs:
    refined_archive: Path
    refined_run_id: str
    crop_archive: Path
    crop_run_id: str
    refined_result_path: Path
    crop_result_path: Path
    terminal_job_key: str
    refined_artifact_key: str
    crop_artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "refined_archive": str(self.refined_archive),
            "refined_run_id": self.refined_run_id,
            "crop_archive": str(self.crop_archive),
            "crop_run_id": self.crop_run_id,
            "refined_result_path": str(self.refined_result_path),
            "crop_result_path": str(self.crop_result_path),
            "terminal_job_key": self.terminal_job_key,
            "refined_artifact_key": self.refined_artifact_key,
            "crop_artifact_key": self.crop_artifact_key,
            "selector_eligible": False,
            "registry_updated": False,
            "publication_partition": "complete_recording_snapshot",
        }


@dataclass(frozen=True)
class ClippedStorageFinalizationModule:
    fragment: LsfWorkflowFragment
    outputs: ClippedStorageFinalizationOutputs


@dataclass(frozen=True)
class ClippedStorageKeypointChainModules:
    """Two composable fragments with their dependency edge frozen."""

    storage: ClippedStorageFinalizationModule
    keypoints: ClippedKeypointV2FinalizationModule


def build_clipped_storage_finalization_fragment(
    inputs: ClippedStorageFinalizationInputs,
) -> ClippedStorageFinalizationModule:
    """Plan strict refined publication followed by geometry-only crop-v2."""

    target = safe_component(inputs.target_id, default="target", max_length=56)
    refined_archive = inputs.bundle_root / "refined_detections.zarr"
    crop_archive = inputs.bundle_root / "crop_geometry.zarr"
    refined_result = inputs.run_root / "storage_finalization" / f"{target}.refined.json"
    crop_result = inputs.run_root / "storage_finalization" / f"{target}.crop.json"
    ordered = sorted(inputs.clips, key=lambda item: item.clip_index)

    refined_command: list[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.finalize_clipped_refined_detection_v1",
        "--canonical-archive",
        str(inputs.canonical_archive),
        "--canonical-run",
        inputs.canonical_run_id,
        "--clipped-binding",
        str(inputs.clipped_binding_path),
        "--output-archive",
        str(refined_archive),
        "--safe-root",
        str(inputs.bundle_root),
        "--output-run",
        inputs.refined_run_id,
        "--recording-identity",
        inputs.recording_identity,
        "--lineage-id",
        inputs.refined_lineage_id,
        "--snapshot-id",
        inputs.refined_snapshot_id,
        "--result-json",
        str(refined_result),
    ]
    for clip in ordered:
        refined_command.extend(("--clip-archive", str(clip.archive)))
        refined_command.extend(("--clip-run", clip.run_id))
    refined_key = f"clipped_refined_finalize:{target}"
    refined_job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=refined_key,
        stage="clipped_refined_detection_finalize",
        command=chain_commands(
            (
                ("mkdir", "-p", str(inputs.bundle_root)),
                tuple(refined_command),
            )
        ),
        resources=LsfResources(
            queue="short", ncores=4, mem_gb=48, walltime="2:00", span_hosts=1
        ),
        upstream=inputs.upstream_job_keys,
        expected_outputs=(
            refined_result,
            refined_archive / "zarr.json",
            refined_archive / "finalization_receipt.json",
        ),
    )

    crop_command: list[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.publish_clipped_crop_geometry_v2",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--refined-archive",
        str(refined_archive),
        "--refined-run",
        inputs.refined_run_id,
        "--destination",
        str(crop_archive),
        "--safe-root",
        str(inputs.bundle_root),
        "--run-id",
        inputs.crop_run_id,
        "--purpose",
        inputs.crop_purpose,
        "--roi-width",
        str(inputs.roi_width),
        "--roi-height",
        str(inputs.roi_height),
        "--camera-id",
        inputs.camera_id,
        "--result-json",
        str(crop_result),
    ]
    for clip in ordered:
        crop_command.extend(("--clip-archive", str(clip.archive)))
        crop_command.extend(("--clip-run", clip.run_id))
    crop_key = f"clipped_crop_finalize:{target}"
    crop_job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=crop_key,
        stage="clipped_crop_geometry_finalize",
        command=tuple(crop_command),
        resources=LsfResources(
            queue="short", ncores=4, mem_gb=32, walltime="1:00", span_hosts=1
        ),
        upstream=(refined_key,),
        expected_outputs=(crop_result, crop_archive / "zarr.json"),
    )
    refined_artifact = f"selector_ineligible_refined_detection:{target}"
    crop_artifact = f"selector_ineligible_crop_v2:{target}"
    outputs = ClippedStorageFinalizationOutputs(
        refined_archive=refined_archive,
        refined_run_id=inputs.refined_run_id,
        crop_archive=crop_archive,
        crop_run_id=inputs.crop_run_id,
        refined_result_path=refined_result,
        crop_result_path=crop_result,
        terminal_job_key=crop_key,
        refined_artifact_key=refined_artifact,
        crop_artifact_key=crop_artifact,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"clipped_storage_finalization:{target}",
        jobs=(refined_job, crop_job),
        requires=inputs.required_artifacts,
        provides=(refined_artifact, crop_artifact),
        metadata={
            "module": "clipped_storage_finalization",
            "target_id": inputs.target_id,
            "clip_count": len(ordered),
            "compute_partition": "clip_local",
            "publication_partition": "complete_recording_snapshot",
            "physical_layout_source": "shared_byte_planners",
            "selector_activation": "none_direct_path_only",
            "registry_update": False,
            "outputs": outputs.to_json(),
        },
    )
    return ClippedStorageFinalizationModule(fragment=fragment, outputs=outputs)


def build_clipped_storage_keypoint_chain_fragments(
    storage_inputs: ClippedStorageFinalizationInputs,
    keypoint_inputs: ClippedKeypointV2FinalizationInputs,
) -> ClippedStorageKeypointChainModules:
    """Bind the crop-v2 output to the recording-level keypoint finalizer."""

    storage = build_clipped_storage_finalization_fragment(storage_inputs)
    if keypoint_inputs.target_id != storage_inputs.target_id:
        raise ValueError("Storage and keypoint finalizers target different recordings.")
    if keypoint_inputs.analysis_zarr != storage_inputs.analysis_zarr:
        raise ValueError("Storage and keypoint finalizers use different archives.")
    keypoints = build_clipped_keypoint_v2_finalization_fragment(
        replace(
            keypoint_inputs,
            crop_archive=storage.outputs.crop_archive,
            crop_run_id=storage.outputs.crop_run_id,
            upstream_job_keys=tuple(
                dict.fromkeys(
                    (
                        *keypoint_inputs.upstream_job_keys,
                        storage.outputs.terminal_job_key,
                    )
                )
            ),
            required_artifacts=tuple(
                dict.fromkeys(
                    (
                        *keypoint_inputs.required_artifacts,
                        storage.outputs.crop_artifact_key,
                    )
                )
            ),
        )
    )
    return ClippedStorageKeypointChainModules(
        storage=storage,
        keypoints=keypoints,
    )


__all__ = [
    "ClippedStorageFinalizationInputs",
    "ClippedStorageFinalizationModule",
    "ClippedStorageFinalizationOutputs",
    "ClippedStorageKeypointChainModules",
    "StrictClipRefinedDetectionInput",
    "build_clipped_storage_finalization_fragment",
    "build_clipped_storage_keypoint_chain_fragments",
]
