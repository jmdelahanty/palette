"""Commit-pinned LSF workflow unlocked by one frozen geometry approval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.lsf import LsfResources, LsfWorkflow
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN
from fisheye.cluster.recording_detection_postprocess import (
    RecordingDetectionPostprocessInputs,
    build_recording_detection_postprocess_fragment,
)
from fisheye.cluster.recording_layout import whole_video_recording_target
from fisheye.registry.geometry_review_approval import GeometryReviewApprovalRequest

FAMILY = "geometry_review_approval"


@dataclass(frozen=True)
class GeometryReviewApprovalWorkflowPlan:
    request_id: str
    request_sha256: str
    repo_commit: str
    request_path: Path
    run_root: Path
    result_path: Path
    registry_result_path: Path
    registry_backup_path: Path
    workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.geometry_review_approval_workflow_plan",
            "schema_version": 1,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "repo_commit": self.repo_commit,
            "request_path": str(self.request_path),
            "run_root": str(self.run_root),
            "result_path": str(self.result_path),
            "registry_result_path": str(self.registry_result_path),
            "registry_backup_path": str(self.registry_backup_path),
            "workflow": self.workflow.to_json(),
        }


def build_geometry_review_approval_workflow(
    request: GeometryReviewApprovalRequest,
    *,
    request_path: str | Path,
    palette_repo: str | Path,
    run_root: str | Path,
) -> GeometryReviewApprovalWorkflowPlan:
    """Build publication -> quality -> refinement -> registry dependencies."""

    identity = request.payload["identity"]
    dataset = identity["dataset"]
    processing = identity["recording_processing"]
    detection = identity["detection_source"]
    pipeline = request.payload["pipeline"]
    repo = Path(palette_repo).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    frozen_request = Path(request_path).expanduser().resolve()
    repo_commit = str(identity["execution"]["palette_commit"])
    target = whole_video_recording_target(
        target_id=str(dataset["recording_id"]),
        recording_id=str(dataset["recording_id"]),
        recording_dir=Path(dataset["recording_dir"]),
        analysis_zarr=Path(dataset["analysis_zarr"]),
        video_path=Path(processing["video_path"]),
        camera_serial=str(dataset["camera_serial"]),
        frame_count=int(processing["frame_count"]),
        arena_id=str(dataset["arena_id"]),
        expected_subject_count=int(processing["expected_subject_count"]),
    )
    publication_key = f"geometry_review_publish:{request.request_id}"
    result_path = root / "approval" / f"{request.request_id}.result.json"
    scratch = (
        f"/scratch/__PALETTE_LSF_USER__/{RUNTIME_JOB_ID_TOKEN}/"
        "geometry_review_approval"
    )
    publication = build_job(
        workflow_id=request.request_id,
        family=FAMILY,
        repo=repo,
        run_root=root,
        job_key=publication_key,
        stage="geometry_review_approval_publication",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.apply_geometry_review_approval",
            "--request-json",
            str(frozen_request),
            "--palette-repo",
            str(repo),
            "--scratch-root",
            scratch,
            "--result-json",
            str(result_path),
            "--apply",
        ),
        resources=LsfResources(
            queue="short", ncores=1, mem_gb=8, walltime="1:00", span_hosts=1
        ),
        expected_outputs=(
            result_path,
            target.analysis_zarr
            / "analysis"
            / "detection_gate_runs"
            / str(pipeline["gate_run"])
            / "zarr.json",
        ),
        cleanup_paths=(scratch,),
    )
    postprocess = build_recording_detection_postprocess_fragment(
        RecordingDetectionPostprocessInputs(
            workflow_id=request.request_id,
            family=FAMILY,
            target=target,
            repo=repo,
            run_root=root,
            source_detect_run=str(detection["run_name"]),
            quality_run=str(pipeline["quality_run"]),
            refined_run=str(pipeline["refined_run"]),
            registered_gate_requirement="required",
            registered_gate_run=str(pipeline["gate_run"]),
            selection_policy_id=str(pipeline["selection_policy_id"]),
            require_active_canonical_source=True,
            expected_source_manifest_digest=str(
                detection["canonical_run_manifest_payload_digest"]
            ),
            upstream_job_keys=(publication_key,),
        )
    )
    registry_result = root / "registry_refresh.json"
    registry_backup = (
        root
        / "registry_backups"
        / f"palette_registry_before_{request.request_id}.sqlite"
    )
    registry_key = f"geometry_review_registry_refresh:{request.request_id}"
    registry_job = build_job(
        workflow_id=request.request_id,
        family=FAMILY,
        repo=repo,
        run_root=root,
        job_key=registry_key,
        stage="geometry_review_registry_refresh",
        command=(
            "scripts/py",
            "-m",
            "fisheye.utils.registry_rescan",
            "--registry",
            str(dataset["registry_path"]),
            "--result-json",
            str(registry_result),
            "--fail-on-error",
            "--reconcile-step-status",
            "--safe-shadow-publish",
            "--backup-path",
            str(registry_backup),
            str(target.analysis_zarr),
        ),
        resources=LsfResources(
            queue="short", ncores=1, mem_gb=8, walltime="1:00", span_hosts=1
        ),
        upstream=(postprocess.outputs.terminal_job_key,),
        expected_outputs=(registry_result, registry_backup),
    )
    workflow = LsfWorkflow(
        workflow_id=request.request_id,
        family=FAMILY,
        jobs=(
            publication,
            *postprocess.fragment.jobs,
            registry_job,
        ),
        metadata={
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "palette_commit": repo_commit,
            "dataset_id": dataset["dataset_id"],
            "recording_id": dataset["recording_id"],
            "analysis_zarr": dataset["analysis_zarr"],
            "selected_candidate_kind": identity["decision"]["selected_candidate_kind"],
            "source_detection_group_path": detection["group_path"],
            "registered_gate_run": pipeline["gate_run"],
            "processing_scope": "geometry_quality_refinement",
            "crop_submission": "deferred",
            "raw_detections_preserved": True,
            "browser_writes_canonical_zarr": False,
            "registry_publication_mode": "local_shadow_copy_atomic_replace",
        },
    )
    return GeometryReviewApprovalWorkflowPlan(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        repo_commit=repo_commit,
        request_path=frozen_request,
        run_root=root,
        result_path=result_path,
        registry_result_path=registry_result,
        registry_backup_path=registry_backup,
        workflow=workflow,
    )


__all__ = [
    "FAMILY",
    "GeometryReviewApprovalWorkflowPlan",
    "build_geometry_review_approval_workflow",
]
