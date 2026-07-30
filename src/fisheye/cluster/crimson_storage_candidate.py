"""Compose one selector-ineligible Crimson storage candidate workflow.

The recording may be computed clip-by-clip, but every public-facing artifact is
finalized as one recording-level immutable snapshot.  This module deliberately
contains no selector or registry operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_detection_evidence import (
    ClippedDetectionEvidenceInputs,
    ClippedDetectionStorageModules,
    build_clipped_detection_storage_fragments,
)
from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
)
from fisheye.cluster.keypoints.v2_finalization import (
    ClippedKeypointV2FinalizationInputs,
    ClippedKeypointV2FinalizationModule,
    RecordingAggregateKeypointV2AdapterInputs,
    build_clipped_keypoint_v2_finalization_fragment,
    build_recording_aggregate_keypoint_v2_adapter_fragment,
)
from fisheye.cluster.lsf import (
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)


class CrimsonCandidateScale(str, Enum):
    """Evidence scope; integration evidence cannot satisfy scale gates."""

    INTEGRATION = "integration_fixture"
    FULL_DURATION = "full_duration_fixture"


@dataclass(frozen=True)
class CrimsonStorageCandidateInputs:
    candidate_id: str
    scale: CrimsonCandidateScale
    expected_n_frames: int
    expected_n_instances: int
    evidence: ClippedDetectionEvidenceInputs
    storage: ClippedStorageFinalizationInputs
    keypoints: (
        ClippedKeypointV2FinalizationInputs | RecordingAggregateKeypointV2AdapterInputs
    )
    handoff_path: Path
    palette_commit: str
    crimson_contract_commit: str
    crimson_contract_sha256: str
    preparation_fragments: tuple[LsfWorkflowFragment, ...] = ()

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id or "/" in candidate_id:
            raise ValueError("candidate_id must be one path-safe component.")
        object.__setattr__(self, "candidate_id", candidate_id)
        if not isinstance(self.scale, CrimsonCandidateScale):
            raise TypeError("scale must be a CrimsonCandidateScale.")
        for name in ("expected_n_frames", "expected_n_instances"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer.")
        if self.scale is CrimsonCandidateScale.INTEGRATION:
            if self.expected_n_frames >= self.storage.roi_width * 100_000:
                raise ValueError("Integration evidence has an implausible frame count.")
        targets = {
            self.evidence.target_id,
            self.storage.target_id,
            self.keypoints.target_id,
        }
        if len(targets) != 1:
            raise ValueError("Candidate fragments target different recordings.")
        workflows = {
            self.evidence.workflow_id,
            self.storage.workflow_id,
            self.keypoints.workflow_id,
        }
        families = {
            self.evidence.family,
            self.storage.family,
            self.keypoints.family,
        }
        if len(workflows) != 1 or len(families) != 1:
            raise ValueError("Candidate fragments must share workflow and family ids.")
        repos = {
            self.evidence.repo.expanduser().resolve(),
            self.storage.repo.expanduser().resolve(),
            self.keypoints.repo.expanduser().resolve(),
        }
        run_roots = {
            self.evidence.run_root.expanduser().resolve(),
            self.storage.run_root.expanduser().resolve(),
            self.keypoints.run_root.expanduser().resolve(),
        }
        if len(repos) != 1 or len(run_roots) != 1:
            raise ValueError("Candidate fragments must share repo and run roots.")
        expected_keypoint_root = self.storage.bundle_root / "keypoints"
        if self.keypoints.bundle_root != expected_keypoint_root:
            raise ValueError(
                "keypoints.bundle_root must equal storage.bundle_root/'keypoints'."
            )
        expected_handoff = self.storage.bundle_root / "handoff_manifest.json"
        if self.handoff_path != expected_handoff:
            raise ValueError(
                "handoff_path must be storage.bundle_root/'handoff_manifest.json'."
            )
        for name in (
            "palette_commit",
            "crimson_contract_commit",
            "crimson_contract_sha256",
        ):
            value = str(getattr(self, name)).strip().lower()
            expected_length = 40 if name.endswith("commit") else 64
            if len(value) != expected_length:
                raise ValueError(f"{name} has the wrong hexadecimal length.")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be lowercase hexadecimal.")
        provided = [
            artifact
            for fragment in self.preparation_fragments
            for artifact in fragment.provides
        ]
        if len(provided) != len(set(provided)):
            raise ValueError("Preparation fragments must provide unique artifacts.")


@dataclass(frozen=True)
class CrimsonStorageCandidatePlan:
    workflow: LsfWorkflow
    preparation_fragments: tuple[LsfWorkflowFragment, ...]
    detection_storage: ClippedDetectionStorageModules
    keypoints: ClippedKeypointV2FinalizationModule
    handoff_fragment: LsfWorkflowFragment
    handoff_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow.to_json(),
            "handoff_path": str(self.handoff_path),
            "selector_eligible": False,
            "registry_updated": False,
            "production_state_changes": [],
        }


def build_crimson_storage_candidate_workflow(
    inputs: CrimsonStorageCandidateInputs,
) -> CrimsonStorageCandidatePlan:
    """Compose strict detections, crop, keypoints, and the final handoff gate."""

    detection_storage = build_clipped_detection_storage_fragments(
        inputs.evidence,
        inputs.storage,
    )
    storage = detection_storage.storage
    bound_keypoint_inputs = replace(
        inputs.keypoints,
        crop_archive=storage.outputs.crop_archive,
        refined_archive=storage.outputs.refined_archive,
        crop_run_id=storage.outputs.crop_run_id,
        upstream_job_keys=tuple(
            dict.fromkeys(
                (
                    *inputs.keypoints.upstream_job_keys,
                    storage.outputs.terminal_job_key,
                )
            )
        ),
        required_artifacts=tuple(
            dict.fromkeys(
                (
                    *inputs.keypoints.required_artifacts,
                    storage.outputs.crop_artifact_key,
                )
            )
        ),
    )
    if isinstance(bound_keypoint_inputs, RecordingAggregateKeypointV2AdapterInputs):
        keypoints = build_recording_aggregate_keypoint_v2_adapter_fragment(
            bound_keypoint_inputs
        )
    else:
        keypoints = build_clipped_keypoint_v2_finalization_fragment(
            bound_keypoint_inputs
        )

    handoff_key = f"crimson_storage_handoff:{inputs.candidate_id}"
    handoff_job = build_job(
        workflow_id=inputs.evidence.workflow_id,
        family=inputs.evidence.family,
        repo=inputs.evidence.repo,
        run_root=inputs.evidence.run_root,
        job_key=handoff_key,
        stage="crimson_storage_candidate_handoff",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.finalize_crimson_storage_candidate",
            "--candidate-id",
            inputs.candidate_id,
            "--classification",
            inputs.scale.value,
            "--expected-n-frames",
            str(inputs.expected_n_frames),
            "--expected-n-instances",
            str(inputs.expected_n_instances),
            "--canonical-archive",
            str(inputs.storage.canonical_archive),
            "--canonical-run",
            inputs.storage.canonical_run_id,
            "--refined-result",
            str(storage.outputs.refined_result_path),
            "--crop-result",
            str(storage.outputs.crop_result_path),
            "--keypoint-result",
            str(keypoints.outputs.result_path),
            "--crimson-contract-commit",
            inputs.crimson_contract_commit,
            "--crimson-contract-sha256",
            inputs.crimson_contract_sha256,
            "--expected-palette-commit",
            inputs.palette_commit,
            "--output",
            str(inputs.handoff_path),
        ),
        resources=LsfResources(
            queue="short", ncores=2, mem_gb=16, walltime="1:00", span_hosts=1
        ),
        upstream=(keypoints.outputs.terminal_job_key,),
        expected_outputs=(inputs.handoff_path,),
    )
    handoff_fragment = LsfWorkflowFragment(
        fragment_id=f"crimson_storage_candidate_handoff:{inputs.candidate_id}",
        jobs=(handoff_job,),
        requires=(
            storage.outputs.refined_artifact_key,
            storage.outputs.crop_artifact_key,
            keypoints.outputs.artifact_key,
        ),
        provides=(f"crimson_storage_candidate:{inputs.candidate_id}",),
        metadata={
            "module": "crimson_storage_candidate_handoff",
            "candidate_id": inputs.candidate_id,
            "classification": inputs.scale.value,
            "expected_n_frames": inputs.expected_n_frames,
            "expected_n_instances": inputs.expected_n_instances,
            "selector_activation": "none_direct_path_only",
            "registry_update": False,
            "handoff_path": str(inputs.handoff_path),
        },
    )
    preparation_provides = {
        artifact
        for fragment in inputs.preparation_fragments
        for artifact in fragment.provides
    }
    requested_external_inputs = tuple(
        dict.fromkeys(
            (
                *(
                    artifact
                    for fragment in inputs.preparation_fragments
                    for artifact in fragment.requires
                ),
                *inputs.evidence.required_artifacts,
                *inputs.storage.required_artifacts,
                *inputs.keypoints.required_artifacts,
            )
        )
    )
    workflow = compose_lsf_workflow(
        workflow_id=inputs.evidence.workflow_id,
        family=inputs.evidence.family,
        fragments=(
            *inputs.preparation_fragments,
            detection_storage.evidence.fragment,
            storage.fragment,
            keypoints.fragment,
            handoff_fragment,
        ),
        external_inputs=tuple(
            artifact
            for artifact in requested_external_inputs
            if artifact not in preparation_provides
        ),
        metadata={
            "module": "crimson_storage_candidate",
            "candidate_id": inputs.candidate_id,
            "classification": inputs.scale.value,
            "publication_partition": "complete_recording_snapshot",
            "physical_layout_source": "shared_versioned_byte_planners",
            "pixel_payload_in_analysis_archive": False,
            "selector_eligible": False,
            "registry_updated": False,
        },
    )
    return CrimsonStorageCandidatePlan(
        workflow=workflow,
        preparation_fragments=inputs.preparation_fragments,
        detection_storage=detection_storage,
        keypoints=keypoints,
        handoff_fragment=handoff_fragment,
        handoff_path=inputs.handoff_path,
    )


__all__ = [
    "CrimsonCandidateScale",
    "CrimsonStorageCandidateInputs",
    "CrimsonStorageCandidatePlan",
    "build_crimson_storage_candidate_workflow",
]
