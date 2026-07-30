"""Plan and optionally submit one full-duration Crimson storage candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from fisheye.cluster.clipped_detection_evidence import (
    ClipDetectionEvidenceInput,
    ClippedDetectionEvidenceInputs,
)
from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
    StrictClipRefinedDetectionInput,
)
from fisheye.cluster.crimson_storage_candidate import (
    CrimsonCandidateScale,
    CrimsonStorageCandidateInputs,
    CrimsonStorageCandidatePlan,
    build_crimson_storage_candidate_workflow,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.keypoints.v2_finalization import (
    RecordingAggregateKeypointV2AdapterInputs,
)
from fisheye.cluster.clipped_lsf import build_job
from fisheye.cluster.lsf import (
    LsfResources,
    LsfWorkflowFragment,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


PLAN_SCHEMA_ID = "palette.crimson.full_storage_candidate_plan"
PLAN_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_commit(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one full lowercase Git commit.")
    return text


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return text


def _clips_from_detection_plan(
    document: Mapping[str, Any],
    *,
    candidate_id: str,
) -> tuple[ClipDetectionEvidenceInput, ...]:
    work_units = document.get("work_units")
    if not isinstance(work_units, list) or not work_units:
        raise ValueError("Detection plan lacks work_units.")
    clips: list[ClipDetectionEvidenceInput] = []
    for expected_index, unit in enumerate(work_units):
        if not isinstance(unit, Mapping):
            raise ValueError("Detection work unit must be an object.")
        index = unit.get("clip_index")
        clip_id = unit.get("clip_id")
        zarr_paths = unit.get("zarr_paths")
        if index != expected_index or not isinstance(clip_id, str):
            raise ValueError("Detection plan clips must form ordered [0, clip_count).")
        if not isinstance(zarr_paths, Mapping):
            raise ValueError(f"Detection plan clip {clip_id!r} lacks zarr_paths.")
        detect_path = zarr_paths.get("detect_target_group_path")
        refined_path = zarr_paths.get("refined_group_path")
        if not isinstance(detect_path, str) or not isinstance(refined_path, str):
            raise ValueError(f"Detection plan clip {clip_id!r} paths are incomplete.")
        prefix = safe_component(
            f"{candidate_id}_{clip_id}", default=f"clip_{index:06d}", max_length=88
        )
        clips.append(
            ClipDetectionEvidenceInput(
                clip_index=index,
                clip_id=clip_id,
                source_detect_group_path=detect_path,
                source_refined_group_path=refined_path,
                canonical_run_id=f"canonical_{prefix}",
                refined_run_id=f"refined_{prefix}",
            )
        )
    declared = document.get("work_unit_count")
    if declared != len(clips):
        raise ValueError("Detection plan work_unit_count disagrees with work_units.")
    return tuple(clips)


@dataclass(frozen=True)
class CrimsonFullStorageCandidateRequest:
    candidate_id: str
    recording_dir: Path
    analysis_zarr: Path
    detection_plan_path: Path
    collection_id: str
    canonical_archive: Path
    canonical_run_id: str
    source_keypoint_group_path: str
    source_keypoint_metadata_sha256: str
    expected_detection_model_sha256: str
    expected_model_sha256: str
    expected_n_frames: int
    expected_n_instances: int
    output_root: Path
    palette_repo: Path
    palette_commit: str
    crimson_contract_commit: str
    crimson_contract_sha256: str
    camera_id: str
    roi_width: int = 512
    roi_height: int = 512


@dataclass(frozen=True)
class CrimsonFullStorageCandidateLsfPlan:
    candidate: CrimsonStorageCandidatePlan
    request: CrimsonFullStorageCandidateRequest
    plan_manifest: Mapping[str, Any]

    @property
    def plan_path(self) -> Path:
        return self.request.output_root / "candidate_plan.json"

    @property
    def lsf_plan_path(self) -> Path:
        return self.request.output_root / "lsf_plan.json"

    @property
    def submission_path(self) -> Path:
        return self.request.output_root / "lsf_submission.json"


def build_full_storage_candidate_plan(
    request: CrimsonFullStorageCandidateRequest,
) -> CrimsonFullStorageCandidateLsfPlan:
    candidate_id = safe_component(request.candidate_id, default="candidate", max_length=72)
    if candidate_id != request.candidate_id:
        raise ValueError("candidate_id must already be one canonical path-safe value.")
    root = request.output_root.expanduser().resolve()
    if ".palette_benchmarks" not in root.parts:
        raise ValueError("Full candidate output_root must be in .palette_benchmarks.")
    repository = request.palette_repo.expanduser().resolve()
    palette_commit = _require_commit(request.palette_commit, name="palette_commit")
    if _git(repository, "rev-parse", "HEAD") != palette_commit:
        raise ValueError("Deployed Palette checkout is not at palette_commit.")
    if _git(repository, "status", "--short"):
        raise ValueError("Deployed Palette checkout must be clean.")
    crimson_commit = _require_commit(
        request.crimson_contract_commit, name="crimson_contract_commit"
    )
    crimson_sha = _require_sha256(
        request.crimson_contract_sha256, name="crimson_contract_sha256"
    )
    source_metadata_sha = _require_sha256(
        request.source_keypoint_metadata_sha256,
        name="source_keypoint_metadata_sha256",
    )
    expected_model_sha = _require_sha256(
        request.expected_model_sha256, name="expected_model_sha256"
    )
    expected_detection_model_sha = _require_sha256(
        request.expected_detection_model_sha256,
        name="expected_detection_model_sha256",
    )
    analysis = request.analysis_zarr.expanduser().resolve()
    plan_path = request.detection_plan_path.expanduser().resolve()
    detection_plan = _read_json(plan_path)
    recording_identity = str(detection_plan.get("recording_id") or "").strip()
    if not recording_identity or recording_identity != request.recording_dir.name:
        raise ValueError("Detection plan and recording_dir identities differ.")
    if Path(str(detection_plan.get("analysis_zarr") or "")).resolve() != analysis:
        raise ValueError("Detection plan and requested analysis_zarr differ.")
    clips = _clips_from_detection_plan(detection_plan, candidate_id=candidate_id)
    frame_sum = sum(
        int(unit["frame_count"])
        for unit in detection_plan["work_units"]
        if isinstance(unit, Mapping)
    )
    if frame_sum != request.expected_n_frames:
        raise ValueError("Detection plan frame sum differs from expected_n_frames.")
    detection_model_path = Path(
        str(detection_plan.get("model") or "")
    ).expanduser().resolve()
    if not detection_model_path.is_file():
        raise FileNotFoundError(
            f"Pinned detection model not found: {detection_model_path}"
        )
    if _sha256_file(detection_model_path) != expected_detection_model_sha:
        raise ValueError("Detection plan model differs from its requested pin.")
    source_metadata_path = analysis / request.source_keypoint_group_path / "zarr.json"
    if _sha256_file(source_metadata_path) != source_metadata_sha:
        raise ValueError("Source keypoint metadata differs from its requested pin.")

    run_root = root / "lsf"
    evidence_root = root / "clip_evidence"
    canonical_archive = root / "canonical_detection.zarr"
    canonical_run_id = f"canonical_{candidate_id}"
    canonical_result = root / "canonical_detection_result.json"
    canonical_job_key = f"canonical_detection_adapter:{candidate_id}"
    canonical_artifact_key = f"canonical_detection:{candidate_id}"
    canonical_job = build_job(
        workflow_id=candidate_id,
        family="analysis.crimson_full_storage_candidate",
        repo=repository,
        run_root=run_root,
        job_key=canonical_job_key,
        stage="canonical_detection_benchmark_adapter",
        command=(
            "scripts/py",
            "-m",
            (
                "fisheye.utils."
                "finalize_recording_canonical_detection_benchmark_adapter"
            ),
            "--analysis-zarr",
            str(analysis),
            "--detection-plan",
            str(plan_path),
            "--recording-frame-index",
            str(request.recording_dir / "recording_frame_index.parquet"),
            "--recording-identity",
            recording_identity,
            "--canonical-anchor-archive",
            str(request.canonical_archive),
            "--canonical-anchor-run",
            request.canonical_run_id,
            "--expected-model-sha256",
            expected_detection_model_sha,
            "--expected-n-frames",
            str(request.expected_n_frames),
            "--destination",
            str(canonical_archive),
            "--benchmark-root",
            str(root),
            "--run-id",
            canonical_run_id,
            "--result-json",
            str(canonical_result),
        ),
        resources=LsfResources(
            queue="local", ncores=4, mem_gb=32, walltime="2:00", span_hosts=1
        ),
        expected_outputs=(
            canonical_archive / "zarr.json",
            canonical_result,
        ),
    )
    canonical_fragment = LsfWorkflowFragment(
        fragment_id=f"canonical_detection_adapter:{candidate_id}",
        jobs=(canonical_job,),
        provides=(canonical_artifact_key,),
        metadata={
            "module": "recording_canonical_detection_benchmark_adapter",
            "candidate_id": candidate_id,
            "canonical_archive": str(canonical_archive),
            "canonical_run_id": canonical_run_id,
            "canonical_anchor_archive": str(request.canonical_archive),
            "canonical_anchor_run_id": request.canonical_run_id,
            "node_local_materialization": True,
            "selector_activation": "none_direct_path_only",
            "registry_update": False,
        },
    )
    refined_lineage = str(
        uuid5(NAMESPACE_URL, f"{recording_identity}:{candidate_id}:refined-lineage")
    )
    refined_snapshot = str(
        uuid5(NAMESPACE_URL, f"{recording_identity}:{candidate_id}:refined-snapshot")
    )
    keypoint_lineage = str(
        uuid5(NAMESPACE_URL, f"{recording_identity}:{candidate_id}:keypoint-lineage")
    )
    keypoint_snapshot = str(
        uuid5(NAMESPACE_URL, f"{recording_identity}:{candidate_id}:keypoint-snapshot")
    )
    evidence = ClippedDetectionEvidenceInputs(
        workflow_id=candidate_id,
        family="analysis.crimson_full_storage_candidate",
        target_id=recording_identity,
        analysis_zarr=analysis,
        recording_canonical_archive=canonical_archive,
        recording_canonical_run_id=canonical_run_id,
        recording_identity=recording_identity,
        detection_plan_path=plan_path,
        collection_id=request.collection_id,
        recording_dir=request.recording_dir,
        bundle_root=evidence_root,
        clips=clips,
        repo=repository,
        run_root=run_root,
        max_concurrent=4,
        upstream_job_keys=(canonical_job_key,),
        required_artifacts=(canonical_artifact_key,),
    )
    placeholders = tuple(
        StrictClipRefinedDetectionInput(
            clip_index=clip.clip_index,
            clip_id=clip.clip_id,
            archive=evidence_root / f"placeholder_{clip.clip_index:06d}.zarr",
            run_id=clip.refined_run_id,
        )
        for clip in clips
    )
    storage = ClippedStorageFinalizationInputs(
        workflow_id=candidate_id,
        family=evidence.family,
        target_id=recording_identity,
        analysis_zarr=analysis,
        canonical_archive=canonical_archive,
        canonical_run_id=canonical_run_id,
        clips=placeholders,
        clipped_binding_path=root / "pending_binding.json",
        bundle_root=root,
        refined_run_id=f"refined_{candidate_id}",
        refined_lineage_id=refined_lineage,
        refined_snapshot_id=refined_snapshot,
        crop_run_id=f"crop_{candidate_id}",
        recording_identity=recording_identity,
        crop_purpose="pose",
        roi_width=request.roi_width,
        roi_height=request.roi_height,
        camera_id=request.camera_id,
        repo=repository,
        run_root=run_root,
    )
    keypoints = RecordingAggregateKeypointV2AdapterInputs(
        workflow_id=candidate_id,
        family=evidence.family,
        target_id=recording_identity,
        analysis_zarr=analysis,
        source_group_path=request.source_keypoint_group_path,
        source_group_metadata_sha256=source_metadata_sha,
        expected_model_sha256=expected_model_sha,
        expected_n_frames=request.expected_n_frames,
        expected_n_instances=request.expected_n_instances,
        crop_run_id=storage.crop_run_id,
        bundle_root=root / "keypoints",
        raw_run_id=f"raw_keypoints_{candidate_id}",
        quality_run_id=f"keypoint_quality_{candidate_id}",
        refined_run_id=f"refined_keypoints_{candidate_id}",
        body_frame_run_id=f"body_frame_{candidate_id}",
        recording_identity=recording_identity,
        refined_lineage_id=keypoint_lineage,
        refined_snapshot_id=keypoint_snapshot,
        repo=repository,
        run_root=run_root,
    )
    candidate = build_crimson_storage_candidate_workflow(
        CrimsonStorageCandidateInputs(
            candidate_id=candidate_id,
            scale=CrimsonCandidateScale.FULL_DURATION,
            expected_n_frames=request.expected_n_frames,
            expected_n_instances=request.expected_n_instances,
            evidence=evidence,
            storage=storage,
            keypoints=keypoints,
            handoff_path=root / "handoff_manifest.json",
            palette_commit=palette_commit,
            crimson_contract_commit=crimson_commit,
            crimson_contract_sha256=crimson_sha,
            preparation_fragments=(canonical_fragment,),
        )
    )
    payload = {
        "status": "planned",
        "candidate_id": candidate_id,
        "classification": "full_duration_fixture",
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "recording_identity": recording_identity,
        "dimensions": {
            "n_frames": request.expected_n_frames,
            "n_instances": request.expected_n_instances,
        },
        "inputs": {
            "analysis_zarr": str(analysis),
            "detection_plan": str(plan_path),
            "detection_plan_sha256": _sha256_file(plan_path),
            "canonical_archive": str(request.canonical_archive),
            "canonical_run_id": request.canonical_run_id,
            "canonical_anchor_role": "logical_equality_input_only",
            "canonical_output_archive": str(canonical_archive),
            "canonical_output_run_id": canonical_run_id,
            "expected_detection_model_sha256": expected_detection_model_sha,
            "source_keypoint_group_path": request.source_keypoint_group_path,
            "source_keypoint_metadata_sha256": source_metadata_sha,
            "expected_model_sha256": expected_model_sha,
        },
        "publication": {
            "output_root": str(root),
            "handoff_path": str(candidate.handoff_path),
            "video_copy_included": False,
            "crop_pixels_in_analysis_archives": False,
            "node_local_keypoint_materialization": True,
        },
        "palette": {
            "repository": str(repository),
            "commit": palette_commit,
            "worktree_clean": True,
        },
        "crimson_contract": {
            "commit": crimson_commit,
            "document_sha256": crimson_sha,
        },
        "workflow_digest": canonical_json_sha256(candidate.workflow.to_json()),
    }
    manifest = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    return CrimsonFullStorageCandidateLsfPlan(
        candidate=candidate,
        request=request,
        plan_manifest=manifest,
    )


def materialize_full_storage_candidate_plan(
    plan: CrimsonFullStorageCandidateLsfPlan,
) -> None:
    root = plan.request.output_root.expanduser().resolve()
    if plan.plan_path.exists():
        if _read_json(plan.plan_path) != plan.plan_manifest:
            raise FileExistsError("Candidate root contains a different plan manifest.")
        if _read_json(plan.lsf_plan_path) != plan.candidate.workflow.to_json():
            raise FileExistsError("Candidate root contains a different LSF plan.")
        return
    if root.exists():
        raise FileExistsError("Candidate root exists without an identical plan.")
    (root / "lsf" / "logs").mkdir(parents=True)
    (root / "lsf" / "status").mkdir(parents=True)
    write_json_snapshot(plan.lsf_plan_path, plan.candidate.workflow.to_json())
    write_json_snapshot(plan.plan_path, plan.plan_manifest)


def apply_full_storage_candidate_plan(
    plan: CrimsonFullStorageCandidateLsfPlan,
    *,
    submit_host: str = "login1-citrus-poller",
) -> dict[str, Any]:
    materialize_full_storage_candidate_plan(plan)
    if plan.submission_path.exists():
        raise FileExistsError("Candidate already has LSF submission evidence.")
    return submit_lsf_workflow(
        plan.candidate.workflow,
        cwd=plan.request.palette_repo,
        plan_path=plan.lsf_plan_path,
        submission_path=plan.submission_path,
        runner=build_ssh_bsub_runner(submit_host),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--detection-plan", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--canonical-archive", type=Path, required=True)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--source-keypoint-group", required=True)
    parser.add_argument("--source-keypoint-metadata-sha256", required=True)
    parser.add_argument("--expected-detection-model-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--expected-n-instances", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--palette-repo", type=Path, required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--crimson-contract-commit", required=True)
    parser.add_argument("--crimson-contract-sha256", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--roi-width", type=int, default=512)
    parser.add_argument("--roi-height", type=int, default=512)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_full_storage_candidate_plan(
        CrimsonFullStorageCandidateRequest(
            candidate_id=args.candidate_id,
            recording_dir=args.recording_dir,
            analysis_zarr=args.analysis_zarr,
            detection_plan_path=args.detection_plan,
            collection_id=args.collection_id,
            canonical_archive=args.canonical_archive,
            canonical_run_id=args.canonical_run,
            source_keypoint_group_path=args.source_keypoint_group,
            source_keypoint_metadata_sha256=args.source_keypoint_metadata_sha256,
            expected_detection_model_sha256=(
                args.expected_detection_model_sha256
            ),
            expected_model_sha256=args.expected_model_sha256,
            expected_n_frames=args.expected_n_frames,
            expected_n_instances=args.expected_n_instances,
            output_root=args.output_root,
            palette_repo=args.palette_repo,
            palette_commit=args.palette_commit,
            crimson_contract_commit=args.crimson_contract_commit,
            crimson_contract_sha256=args.crimson_contract_sha256,
            camera_id=args.camera_id,
            roi_width=args.roi_width,
            roi_height=args.roi_height,
        )
    )
    if args.apply:
        result = apply_full_storage_candidate_plan(
            plan, submit_host=args.submit_host
        )
    else:
        materialize_full_storage_candidate_plan(plan)
        result = dict(plan.plan_manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CrimsonFullStorageCandidateLsfPlan",
    "CrimsonFullStorageCandidateRequest",
    "PLAN_SCHEMA_ID",
    "PLAN_SCHEMA_VERSION",
    "apply_full_storage_candidate_plan",
    "build_full_storage_candidate_plan",
    "materialize_full_storage_candidate_plan",
]
