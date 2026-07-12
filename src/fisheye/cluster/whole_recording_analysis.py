"""Plan concurrent whole-recording keypoint and subject-mask inference on LSF."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fisheye.cluster.keypoints import whole_recording as keypoints
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfDependency,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    build_bsub_command,
    shell_join,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_USER_TOKEN,
    build_runtime_command,
)


PLAN_SCHEMA = "palette.whole_recording_analysis_bsub_plan.v1"


@dataclass(frozen=True)
class SubjectMaskRunNames:
    subject_mask_run: str
    refined_subject_mask_run: str

    def to_json(self) -> dict[str, str]:
        return {
            "subject_mask_run": self.subject_mask_run,
            "refined_subject_mask_run": self.refined_subject_mask_run,
        }


@dataclass(frozen=True)
class PlannedAnalysisTarget:
    target_id: str
    analysis_zarr: Path
    crop_run: str
    roi_cache_manifest: Path
    keypoint_run: str
    refined_keypoint_run: str
    subject_masks: SubjectMaskRunNames
    keypoint_prediction_job_key: str
    keypoint_refinement_job_key: str
    subject_mask_inference_job_key: str
    subject_mask_finalization_job_key: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "analysis_zarr": str(self.analysis_zarr),
            "crop_run": self.crop_run,
            "roi_cache_manifest": str(self.roi_cache_manifest),
            "keypoint_run": self.keypoint_run,
            "refined_keypoint_run": self.refined_keypoint_run,
            "subject_masks": self.subject_masks.to_json(),
            "jobs": {
                "keypoint_prediction": self.keypoint_prediction_job_key,
                "keypoint_refinement": self.keypoint_refinement_job_key,
                "subject_mask_inference": self.subject_mask_inference_job_key,
                "subject_mask_finalization": self.subject_mask_finalization_job_key,
            },
        }


@dataclass(frozen=True)
class WholeRecordingAnalysisPlan:
    run_label: str
    mask_run_label: str
    repo: Path
    registry: Path
    run_root: Path
    keypoint_plan: keypoints.WholeRecordingWorkflowPlan
    targets: tuple[PlannedAnalysisTarget, ...]
    lsf_workflow: LsfWorkflow

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "mask_run_label": self.mask_run_label,
            "repo": str(self.repo),
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "keypoint_plan_path": str(self.keypoint_plan.run_root / "plan.json"),
            "target_count": len(self.targets),
            "targets": [target.to_json() for target in self.targets],
            "lsf_workflow": self.lsf_workflow.to_json(),
        }


def build_subject_mask_run_names(run_label: str) -> SubjectMaskRunNames:
    label = safe_component(run_label, default="subject_masks")
    return SubjectMaskRunNames(
        subject_mask_run=f"subject_masks_unet_registry_{label}",
        refined_subject_mask_run=f"refined_subject_masks_smart_finalizer_{label}",
    )


def _refuse_mask_output_collisions(
    analysis_zarr: Path,
    run_names: SubjectMaskRunNames,
) -> None:
    outputs = (
        analysis_zarr / "subject_mask_runs" / run_names.subject_mask_run,
        analysis_zarr
        / "refined_subject_masks_runs"
        / run_names.refined_subject_mask_run,
    )
    collisions = [path for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Planned subject-mask output already exists: "
            + ", ".join(str(path) for path in collisions)
        )


def _build_subject_mask_inference_job(
    *,
    workflow_id: str,
    target: keypoints.PlannedWholeRecordingTarget,
    run_names: SubjectMaskRunNames,
    repo: Path,
    registry: Path,
    run_root: Path,
    resources: LsfResources,
    batch_size: int,
    model_coverage_class: str,
    model_component_coverage_key: str,
    model_label_schema_id: str,
    model_top_k: int,
    handoff_package_dir: Path | None,
) -> LsfJob:
    target_id = target.target.target_id
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"mask_infer:{target_id}"
    job_name = safe_component(
        f"sm_{workflow_id}_{safe_target}",
        default="subject_mask_inference",
        max_length=120,
    )
    scratch_stage = (
        f"/scratch/{RUNTIME_USER_TOKEN}/{RUNTIME_JOB_ID_TOKEN}/"
        "palette_subject_mask_roi_cache_stage"
    )
    worker: list[str] = [
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.subject_masks.recording",
        "inference",
        "--analysis-zarr",
        str(target.target.analysis_zarr),
        "--registry",
        str(registry),
        "--run-label",
        run_names.subject_mask_run.removeprefix("subject_masks_unet_registry_"),
        "--crop-run",
        target.cache.crop_run,
        "--roi-cache-manifest",
        str(target.cache.manifest_path),
        "--roi-cache-staging-dir",
        scratch_stage,
        "--batch-size",
        str(int(batch_size)),
        "--model-coverage-class",
        model_coverage_class,
        "--model-component-coverage-key",
        model_component_coverage_key,
        "--model-label-schema-id",
        model_label_schema_id,
        "--model-top-k",
        str(int(model_top_k)),
        "--progress-dir",
        str(run_root / "progress"),
        "--json",
    ]
    if handoff_package_dir is not None:
        worker.extend(["--handoff-package-dir", str(handoff_package_dir)])
    command = build_runtime_command(
        worker,
        status_path_template=(
            run_root
            / "status"
            / f"{safe_target}.subject_mask_inference.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=workflow_id,
        family="analysis.whole_recording",
        job_key=job_key,
        stage="subject_mask_inference",
        cwd=repo,
        cleanup_path_templates=(scratch_stage,),
        expected_output_templates=(
            str(
                target.target.analysis_zarr
                / "subject_mask_runs"
                / run_names.subject_mask_run
            ),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        metadata={
            "target_id": target_id,
            "analysis_zarr": str(target.target.analysis_zarr),
            "subject_mask_run": run_names.subject_mask_run,
            "roi_cache_manifest": str(target.cache.manifest_path),
            "assignment_keypoints": None,
        },
    )


def _build_subject_mask_finalization_job(
    *,
    workflow_id: str,
    target: keypoints.PlannedWholeRecordingTarget,
    run_names: SubjectMaskRunNames,
    repo: Path,
    registry: Path,
    run_root: Path,
    resources: LsfResources,
    num_workers: int,
    inference_job: LsfJob,
    refinement_job: LsfJob,
) -> LsfJob:
    target_id = target.target.target_id
    safe_target = safe_component(target_id, default="target", max_length=56)
    job_key = f"mask_finalize:{target_id}"
    job_name = safe_component(
        f"sm_finalize_{workflow_id}_{safe_target}",
        default="subject_mask_finalization",
        max_length=120,
    )
    worker = (
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.subject_masks.recording",
        "finalization",
        "--analysis-zarr",
        str(target.target.analysis_zarr),
        "--registry",
        str(registry),
        "--run-label",
        run_names.subject_mask_run.removeprefix("subject_masks_unet_registry_"),
        "--crop-run",
        target.cache.crop_run,
        "--refined-keypoint-run",
        target.run_names.refined_keypoint_run,
        "--finalize-num-workers",
        str(int(num_workers)),
        "--progress-dir",
        str(run_root / "progress"),
        "--json",
    )
    command = build_runtime_command(
        worker,
        status_path_template=(
            run_root
            / "status"
            / f"{safe_target}.subject_mask_finalization.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=workflow_id,
        family="analysis.whole_recording",
        job_key=job_key,
        stage="subject_mask_finalization",
        cwd=repo,
        expected_output_templates=(
            str(
                target.target.analysis_zarr
                / "refined_subject_masks_runs"
                / run_names.refined_subject_mask_run
            ),
        ),
        python_launcher=(str(repo / "scripts" / "py"),),
    )
    return LsfJob(
        job_key=job_key,
        job_name=job_name,
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{job_name}.%J.out",
        stderr_path=run_root / "logs" / f"{job_name}.%J.err",
        dependency=LsfDependency((inference_job.job_key, refinement_job.job_key)),
        metadata={
            "target_id": target_id,
            "analysis_zarr": str(target.target.analysis_zarr),
            "source_subject_mask_run": run_names.subject_mask_run,
            "refined_subject_mask_run": run_names.refined_subject_mask_run,
            "assignment_keypoint_group": "refined_keypoints_runs",
            "assignment_keypoint_run": target.run_names.refined_keypoint_run,
            "component_contours_requested": False,
            "sampled_component_contours_requested": True,
        },
    )


def build_plan(
    *,
    keypoint_plan: keypoints.WholeRecordingWorkflowPlan,
    run_root: Path,
    mask_run_label: str,
    mask_inference_resources: LsfResources,
    mask_finalization_resources: LsfResources,
    mask_batch_size: int = 128,
    mask_finalize_num_workers: int = 16,
    model_coverage_class: str = "dense_all_components",
    model_component_coverage_key: str = "body+eyes+swim_bladder",
    model_label_schema_id: str = "subject_v1_union",
    model_top_k: int = 5,
    handoff_package_dir: Path | None = None,
) -> WholeRecordingAnalysisPlan:
    run_root = run_root.expanduser().resolve()
    run_label = keypoint_plan.run_label
    mask_label = safe_component(mask_run_label, default=f"{run_label}_subject_masks")
    mask_names = build_subject_mask_run_names(mask_label)
    keypoint_finalizer = next(
        job
        for job in keypoint_plan.lsf_workflow.jobs
        if job.job_key == keypoint_plan.registry_finalizer_job_key
    )
    jobs = [
        job
        for job in keypoint_plan.lsf_workflow.jobs
        if job.job_key != keypoint_plan.registry_finalizer_job_key
    ]
    jobs_by_key = {job.job_key: job for job in jobs}
    planned_targets: list[PlannedAnalysisTarget] = []
    mask_finalizer_keys: list[str] = []

    for target in keypoint_plan.targets:
        _refuse_mask_output_collisions(target.target.analysis_zarr, mask_names)
        inference_job = _build_subject_mask_inference_job(
            workflow_id=run_label,
            target=target,
            run_names=mask_names,
            repo=keypoint_plan.repo,
            registry=keypoint_plan.registry,
            run_root=run_root,
            resources=mask_inference_resources,
            batch_size=mask_batch_size,
            model_coverage_class=model_coverage_class,
            model_component_coverage_key=model_component_coverage_key,
            model_label_schema_id=model_label_schema_id,
            model_top_k=model_top_k,
            handoff_package_dir=handoff_package_dir,
        )
        refinement_job = jobs_by_key[target.refinement_job_key]
        finalization_job = _build_subject_mask_finalization_job(
            workflow_id=run_label,
            target=target,
            run_names=mask_names,
            repo=keypoint_plan.repo,
            registry=keypoint_plan.registry,
            run_root=run_root,
            resources=mask_finalization_resources,
            num_workers=mask_finalize_num_workers,
            inference_job=inference_job,
            refinement_job=refinement_job,
        )
        jobs.extend((inference_job, finalization_job))
        mask_finalizer_keys.append(finalization_job.job_key)
        planned_targets.append(
            PlannedAnalysisTarget(
                target_id=target.target.target_id,
                analysis_zarr=target.target.analysis_zarr,
                crop_run=target.cache.crop_run,
                roi_cache_manifest=target.cache.manifest_path,
                keypoint_run=target.run_names.keypoint_run,
                refined_keypoint_run=target.run_names.refined_keypoint_run,
                subject_masks=mask_names,
                keypoint_prediction_job_key=target.prediction_job_key,
                keypoint_refinement_job_key=target.refinement_job_key,
                subject_mask_inference_job_key=inference_job.job_key,
                subject_mask_finalization_job_key=finalization_job.job_key,
            )
        )

    registry_job_key = "registry_finalize"
    registry_job_name = safe_component(
        f"analysis_registry_{run_label}",
        default="analysis_registry_finalize",
        max_length=120,
    )
    registry_output = (
        run_root / "registry" / f"registry_finalizer.{RUNTIME_JOB_ID_TOKEN}.json"
    )
    registry_worker = (
        str(keypoint_plan.repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.whole_recording_analysis_registry_finalize",
        str(run_root),
        "--registry",
        str(keypoint_plan.registry),
        "--apply",
        "--output-json",
        str(registry_output),
    )
    registry_command = build_runtime_command(
        registry_worker,
        status_path_template=(
            run_root / "status" / f"registry_finalizer.{RUNTIME_JOB_ID_TOKEN}.json"
        ),
        workflow_id=run_label,
        family="analysis.whole_recording",
        job_key=registry_job_key,
        stage="registry_finalization",
        cwd=keypoint_plan.repo,
        expected_output_templates=(str(registry_output),),
        python_launcher=(str(keypoint_plan.repo / "scripts" / "py"),),
    )
    jobs.append(
        LsfJob(
            job_key=registry_job_key,
            job_name=registry_job_name,
            command=registry_command,
            resources=keypoint_finalizer.resources,
            stdout_path=run_root / "logs" / f"{registry_job_name}.%J.out",
            stderr_path=run_root / "logs" / f"{registry_job_name}.%J.err",
            dependency=LsfDependency(tuple(mask_finalizer_keys)),
            metadata={
                "whole_recording_analysis_join": True,
                "target_count": len(planned_targets),
                "subject_mask_finalizer_job_keys": list(mask_finalizer_keys),
                "registry": str(keypoint_plan.registry),
            },
        )
    )
    workflow = LsfWorkflow(
        workflow_id=run_label,
        family="analysis.whole_recording",
        jobs=tuple(jobs),
        metadata={
            "plan_schema": PLAN_SCHEMA,
            "target_count": len(planned_targets),
            "fork_join_contract": (
                "keypoint_prediction_and_subject_mask_inference_parallel;"
                "subject_mask_finalization_joins_keypoint_refinement_and_mask_inference"
            ),
            "mask_run_label": mask_label,
        },
    )
    return WholeRecordingAnalysisPlan(
        run_label=run_label,
        mask_run_label=mask_label,
        repo=keypoint_plan.repo,
        registry=keypoint_plan.registry,
        run_root=run_root,
        keypoint_plan=keypoint_plan,
        targets=tuple(planned_targets),
        lsf_workflow=workflow,
    )


def materialize_plan_bundle(plan: WholeRecordingAnalysisPlan) -> dict[str, Any]:
    plan.run_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "progress", "status", "registry"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    keypoints.materialize_plan_bundle(plan.keypoint_plan)
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Run root contains a different plan: {plan_path}")
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(plan.run_root / "lsf_plan.json", plan.lsf_workflow.to_json())
    return payload


def apply_plan(
    plan: WholeRecordingAnalysisPlan,
    *,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission_path}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.lsf_workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=runner,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--mask-run-label")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=keypoints.DEFAULT_GROUPS_REPO)
    parser.add_argument("--registry", type=Path, default=keypoints.DEFAULT_REGISTRY)
    parser.add_argument("--model-set-id", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--pose-schema", default="traditional_v2")
    parser.add_argument("--batch-size-kp", type=int, default=256)
    parser.add_argument("--batch-size-sm", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mask-model-coverage-class", default="dense_all_components")
    parser.add_argument(
        "--mask-model-component-coverage-key",
        default="body+eyes+swim_bladder",
    )
    parser.add_argument("--mask-model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--mask-model-top-k", type=int, default=5)
    parser.add_argument("--handoff-package-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    keypoint_plan = keypoints.build_plan(
        manifest_path=args.manifest,
        run_label=args.run_label,
        repo=args.repo,
        registry=args.registry,
        run_root=run_root / "keypoints",
        model_set_id=args.model_set_id,
        model_run_id=args.model_run_id,
        pose_schema=args.pose_schema,
        min_roi_size=348,
        batch_size=args.batch_size_kp,
        device=args.device,
        input_mode="tensor",
        progress_every_batches=1,
        prediction_resources=LsfResources(queue="gpu_l4", ncores=4, mem_gb=32, gpus=1),
        refinement_resources=LsfResources(queue="short", ncores=4, mem_gb=16, walltime="1:00"),
        refine_chunk_size=2048,
        refine_scheduler="threads",
        refine_num_workers=4,
        refine_memory_limit=None,
        finalizer_resources=LsfResources(queue="short", ncores=1, mem_gb=8, walltime="1:00"),
    )
    plan = build_plan(
        keypoint_plan=keypoint_plan,
        run_root=run_root,
        mask_run_label=args.mask_run_label or f"{args.run_label}_subject_masks",
        mask_inference_resources=LsfResources(queue="gpu_l4", ncores=8, mem_gb=48, gpus=1),
        mask_finalization_resources=LsfResources(queue="short", ncores=16, mem_gb=32),
        mask_batch_size=args.batch_size_sm,
        mask_finalize_num_workers=16,
        model_coverage_class=args.mask_model_coverage_class,
        model_component_coverage_key=args.mask_model_component_coverage_key,
        model_label_schema_id=args.mask_model_label_schema_id,
        model_top_k=args.mask_model_top_k,
        handoff_package_dir=args.handoff_package_dir,
    )
    if args.apply:
        result = apply_plan(plan)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
        return 0
    materialize_plan_bundle(plan)
    if args.json:
        print(json.dumps(plan.to_json(), indent=2, sort_keys=True))
    else:
        print("Whole-recording concurrent keypoint/subject-mask DAG")
        for job in plan.lsf_workflow.topological_jobs():
            print(f"  {shell_join(build_bsub_command(job))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PLAN_SCHEMA",
    "PlannedAnalysisTarget",
    "SubjectMaskRunNames",
    "WholeRecordingAnalysisPlan",
    "apply_plan",
    "build_plan",
    "build_subject_mask_run_names",
    "main",
    "materialize_plan_bundle",
]
