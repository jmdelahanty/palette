"""LSF DAG for the selector-ineligible chaser proxy candidate chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.lsf import (
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.runtime import RUNTIME_JOB_ID_TOKEN, RUNTIME_USER_TOKEN


FAMILY = "chaser_proxy_candidate"


def _name(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{field} must be one exact bare run name.")
    return text


@dataclass(frozen=True)
class ChaserProxyCandidateWorkflowPlan:
    workflow: LsfWorkflow
    palette_commit: str
    analysis_zarr: Path
    proxy_run_name: str
    relative_frame_run_name: str
    receipt_path: Path
    run_root: Path

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": "palette.chaser_proxy_candidate_workflow_plan",
            "schema_version": 1,
            "palette_commit": self.palette_commit,
            "analysis_zarr": str(self.analysis_zarr),
            "proxy_run_name": self.proxy_run_name,
            "relative_frame_run_name": self.relative_frame_run_name,
            "receipt_path": str(self.receipt_path),
            "run_root": str(self.run_root),
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
            "workflow": self.workflow.to_json(),
        }


def build_chaser_proxy_candidate_workflow(
    *,
    workflow_id: str,
    repo: str | Path,
    run_root: str | Path,
    analysis_zarr: str | Path,
    source_run_name: str,
    proxy_run_name: str,
    relative_frame_run_name: str,
    analysis_profile_path: str | Path,
    palette_commit: str,
    expected_recording_id: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    resources: LsfResources | None = None,
) -> ChaserProxyCandidateWorkflowPlan:
    """Build the exact native -> proxy -> camera-frame -> receipt job graph."""

    workflow_id = _name(workflow_id, field="workflow_id")
    palette_commit = str(palette_commit or "").strip().lower()
    if len(palette_commit) != 40 or any(
        value not in "0123456789abcdef" for value in palette_commit
    ):
        raise ValueError("palette_commit must be one full lowercase Git SHA.")
    source_run_name = _name(source_run_name, field="source_run_name")
    proxy_run_name = _name(proxy_run_name, field="proxy_run_name")
    relative_frame_run_name = _name(
        relative_frame_run_name,
        field="relative_frame_run_name",
    )
    repo_path = Path(repo).expanduser().resolve()
    run_path = Path(run_root).expanduser().resolve()
    archive = Path(analysis_zarr).expanduser().resolve()
    profile = Path(analysis_profile_path).expanduser().resolve()
    if not repo_path.is_dir() or not (repo_path / "scripts/py").is_file():
        raise FileNotFoundError(f"Palette deployment is invalid: {repo_path}")
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr is unavailable: {archive}")
    if not profile.is_file():
        raise FileNotFoundError(f"Chaser analysis profile is unavailable: {profile}")
    if run_path == archive or archive in run_path.parents:
        raise ValueError("Workflow run_root must remain outside the analysis Zarr.")
    resources = resources or LsfResources(
        queue="short",
        ncores=1,
        mem_gb=8,
        gpus=0,
        walltime="1:00",
        span_hosts=1,
    )
    proxy_key = f"chaser_proxy:{proxy_run_name}"
    relative_key = f"chaser_relative_frame:{relative_frame_run_name}"
    receipt_key = f"chaser_proxy_receipt:{relative_frame_run_name}"
    proxy_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/chaser_proxy_candidate"
    )
    relative_scratch = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/chaser_relative_frame"
    )
    proxy_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.materialize_chaser_input_provenance_proxy",
        str(archive),
        "--source-run-name",
        source_run_name,
        "--output-run-name",
        proxy_run_name,
        "--scratch-root",
        proxy_scratch,
        "--apply",
        "--json",
    ]
    if expected_recording_id is not None:
        proxy_command.extend(("--expected-recording-id", expected_recording_id))
    if expected_source_manifest_sha256 is not None:
        proxy_command.extend(
            (
                "--expected-source-manifest-sha256",
                expected_source_manifest_sha256,
            )
        )
    proxy_job = build_job(
        workflow_id=workflow_id,
        family=FAMILY,
        repo=repo_path,
        run_root=run_path,
        job_key=proxy_key,
        stage="chaser_input_provenance_proxy_candidate",
        command=proxy_command,
        resources=resources,
        expected_outputs=(
            archive
            / "analysis/chaser_input_provenance_proxy_runs"
            / proxy_run_name
            / "zarr.json",
        ),
        cleanup_paths=(proxy_scratch,),
    )
    relative_command = [
        "scripts/py",
        "-m",
        "fisheye.utils.materialize_chaser_proxy_relative_frame",
        str(archive),
        "--proxy-run-name",
        proxy_run_name,
        "--output-run-name",
        relative_frame_run_name,
        "--scratch-root",
        relative_scratch,
        "--analysis-profile",
        str(profile),
        "--apply",
        "--json",
    ]
    if expected_recording_id is not None:
        relative_command.extend(("--expected-recording-id", expected_recording_id))
    relative_job = build_job(
        workflow_id=workflow_id,
        family=FAMILY,
        repo=repo_path,
        run_root=run_path,
        job_key=relative_key,
        stage="chaser_relative_frame_candidate",
        command=relative_command,
        resources=resources,
        upstream=(proxy_key,),
        expected_outputs=(
            archive
            / "analysis/chaser_relative_frame_runs"
            / relative_frame_run_name
            / "zarr.json",
        ),
        cleanup_paths=(relative_scratch,),
    )
    receipt_path = run_path / "candidate_chain_receipt.json"
    receipt_command = [
        "scripts/py",
        "-m",
        "fisheye.analysis_workflows.chaser_proxy_candidate_receipt",
        str(archive),
        "--proxy-run-name",
        proxy_run_name,
        "--relative-frame-run-name",
        relative_frame_run_name,
        "--analysis-profile",
        str(profile),
        "--palette-commit",
        palette_commit,
        "--output-json",
        str(receipt_path),
    ]
    if expected_recording_id is not None:
        receipt_command.extend(("--expected-recording-id", expected_recording_id))
    receipt_job = build_job(
        workflow_id=workflow_id,
        family=FAMILY,
        repo=repo_path,
        run_root=run_path,
        job_key=receipt_key,
        stage="chaser_proxy_candidate_readiness_receipt",
        command=receipt_command,
        resources=resources,
        upstream=(relative_key,),
        expected_outputs=(receipt_path,),
    )
    fragment = LsfWorkflowFragment(
        fragment_id=f"chaser_proxy_candidate:{relative_frame_run_name}",
        jobs=(proxy_job, relative_job, receipt_job),
        requires=(f"native_chaser_samples:{source_run_name}",),
        provides=(
            f"chaser_proxy_candidate:{proxy_run_name}",
            f"chaser_relative_frame_candidate:{relative_frame_run_name}",
            f"chaser_proxy_candidate_receipt:{relative_frame_run_name}",
        ),
        metadata={
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
            "physical_presentation_verified": False,
        },
    )
    workflow = compose_lsf_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        fragments=(fragment,),
        external_inputs=(f"native_chaser_samples:{source_run_name}",),
        metadata={
            "analysis_zarr": str(archive),
            "palette_repo": str(repo_path),
            "palette_commit": palette_commit,
            "workflow_scope": "exploratory_controller_input_provenance_proxy",
            "selector_eligible": False,
            "production_selector_activation": False,
            "registry_update": False,
            "required_ci_before_promotion": True,
        },
    )
    return ChaserProxyCandidateWorkflowPlan(
        workflow=workflow,
        palette_commit=palette_commit,
        analysis_zarr=archive,
        proxy_run_name=proxy_run_name,
        relative_frame_run_name=relative_frame_run_name,
        receipt_path=receipt_path,
        run_root=run_path,
    )


__all__ = [
    "FAMILY",
    "ChaserProxyCandidateWorkflowPlan",
    "build_chaser_proxy_candidate_workflow",
]
