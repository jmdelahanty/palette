"""Composable LSF fragment for immutable detection snapshot publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_lsf import build_job, chain_commands
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, RUNTIME_USER_TOKEN


@dataclass(frozen=True)
class DetectionSnapshotFragmentInputs:
    """Exact source and destination identities for one snapshot pair."""

    workflow_id: str
    family: str
    target_id: str
    analysis_zarr: Path
    recording_identity: str
    source_detect_group_path: str
    source_refined_group_path: str
    canonical_run_id: str
    refined_run_id: str
    repo: Path
    run_root: Path
    upstream_job_keys: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    allow_initialize_missing_source_keys: bool = False
    allow_manual_score_reset: bool = False
    resources: LsfResources = LsfResources(
        queue="short",
        ncores=4,
        mem_gb=32,
        walltime="2:00",
        span_hosts=1,
    )

    def __post_init__(self) -> None:
        if not str(self.target_id).strip():
            raise ValueError("Detection snapshot target_id cannot be empty.")
        if not str(self.recording_identity).strip():
            raise ValueError("Detection snapshot recording_identity cannot be empty.")


@dataclass(frozen=True)
class DetectionSnapshotFragmentOutputs:
    """Selector-ineligible run groups and the receipt proving their publication."""

    target_id: str
    canonical_run_id: str
    canonical_group_path: str
    refined_run_id: str
    refined_group_path: str
    receipt_path: Path
    terminal_job_key: str
    artifact_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "canonical_run_id": self.canonical_run_id,
            "canonical_group_path": self.canonical_group_path,
            "refined_run_id": self.refined_run_id,
            "refined_group_path": self.refined_group_path,
            "receipt_path": str(self.receipt_path),
            "terminal_job_key": self.terminal_job_key,
            "artifact_key": self.artifact_key,
            "selector_eligible": False,
        }


@dataclass(frozen=True)
class DetectionSnapshotWorkflowModule:
    fragment: LsfWorkflowFragment
    outputs: DetectionSnapshotFragmentOutputs


def build_detection_snapshot_fragment(
    inputs: DetectionSnapshotFragmentInputs,
) -> DetectionSnapshotWorkflowModule:
    """Build one CPU publication node after full-acquisition detect/refine."""

    target_safe = safe_component(inputs.target_id, default="target", max_length=56)
    job_key = f"detection_snapshot_publish:{target_safe}"
    canonical_group_path = f"detect_runs/{inputs.canonical_run_id}"
    refined_group_path = f"refined_detect_runs/{inputs.refined_run_id}"
    receipt = (
        inputs.run_root / "detection_snapshots" / f"{target_safe}.publication.json"
    )
    scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "detection_snapshot_publication"
    )
    command = [
        "scripts/py",
        "-m",
        "fisheye.utils.publish_detection_snapshots",
        "--analysis-zarr",
        str(inputs.analysis_zarr),
        "--source-detect-group",
        inputs.source_detect_group_path,
        "--source-refined-group",
        inputs.source_refined_group_path,
        "--recording-identity",
        inputs.recording_identity,
        "--canonical-run",
        inputs.canonical_run_id,
        "--refined-run",
        inputs.refined_run_id,
        "--scratch-root",
        scratch,
        "--result-json",
        str(receipt),
    ]
    if inputs.allow_initialize_missing_source_keys:
        command.append("--allow-initialize-missing-source-keys")
    if inputs.allow_manual_score_reset:
        command.append("--allow-manual-score-reset")

    job = build_job(
        workflow_id=inputs.workflow_id,
        family=inputs.family,
        repo=inputs.repo,
        run_root=inputs.run_root,
        job_key=job_key,
        stage="detection_snapshot_publish",
        command=chain_commands(
            (
                ("mkdir", "-p", scratch),
                tuple(command),
            )
        ),
        resources=inputs.resources,
        upstream=inputs.upstream_job_keys,
        expected_outputs=(
            inputs.analysis_zarr / canonical_group_path / "zarr.json",
            inputs.analysis_zarr / refined_group_path / "zarr.json",
            receipt,
        ),
        cleanup_paths=(scratch,),
    )
    artifact_key = f"detection_snapshot_pair:{target_safe}"
    outputs = DetectionSnapshotFragmentOutputs(
        target_id=inputs.target_id,
        canonical_run_id=inputs.canonical_run_id,
        canonical_group_path=canonical_group_path,
        refined_run_id=inputs.refined_run_id,
        refined_group_path=refined_group_path,
        receipt_path=receipt,
        terminal_job_key=job_key,
        artifact_key=artifact_key,
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"detection_snapshots:{target_safe}",
        jobs=(job,),
        requires=inputs.required_artifacts,
        provides=(artifact_key,),
        metadata={
            "module": "detection_snapshots",
            "target_id": inputs.target_id,
            "lineage_profile": "full_acquisition",
            "selector_activation": "deferred",
            "registry_update": False,
            "outputs": outputs.to_json(),
        },
    )
    return DetectionSnapshotWorkflowModule(fragment=fragment, outputs=outputs)


def compose_detection_snapshot_workflow(
    *,
    workflow_id: str,
    family: str,
    modules: tuple[DetectionSnapshotWorkflowModule, ...],
    external_inputs: tuple[str, ...] = (),
) -> LsfWorkflow:
    """Compose snapshot modules for one or many independent recordings."""

    if not modules:
        raise ValueError("A detection snapshot workflow requires at least one module.")
    return compose_lsf_workflow(
        workflow_id=workflow_id,
        family=family,
        fragments=tuple(module.fragment for module in modules),
        external_inputs=external_inputs,
        metadata={
            "workflow_scope": "selector_ineligible_detection_snapshot_publication",
            "target_count": len(modules),
            "selector_activation": "deferred",
            "outputs": [module.outputs.to_json() for module in modules],
        },
    )


__all__ = [
    "DetectionSnapshotFragmentInputs",
    "DetectionSnapshotFragmentOutputs",
    "DetectionSnapshotWorkflowModule",
    "build_detection_snapshot_fragment",
    "compose_detection_snapshot_workflow",
]
