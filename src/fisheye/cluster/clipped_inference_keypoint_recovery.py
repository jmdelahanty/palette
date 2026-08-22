"""Plan a clipped-inference recovery from completed caches and raw masks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import (
    FAMILY,
    PLAN_SCHEMA,
    SUPPORTED_PLAN_SCHEMAS,
    build_ssh_bsub_runner,
)
from fisheye.cluster.clipped_lsf import (
    build_execution_task as _execution_task,
    build_job as _job,
    build_task_group_job as _task_group_job,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfExecutionMode,
    LsfExecutionTask,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.utils.prepare_clipped_keypoint_recovery import prepare_keypoint_recovery

RECOVERY_SCHEMA = "palette.clipped_inference_keypoint_recovery.v1"


@dataclass(frozen=True)
class KeypointRecoveryPlan:
    payload: Mapping[str, Any]
    workflow: LsfWorkflow
    repo: Path
    run_root: Path

    def to_json(self) -> dict[str, Any]:
        return {**dict(self.payload), "lsf_workflow": self.workflow.to_json()}


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _inner_command(job: Mapping[str, Any]) -> tuple[str, ...]:
    command = job.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"Prior job has no command: {job.get('job_key')}")
    inner = (
        tuple(str(value) for value in command[command.index("--") + 1 :])
        if "--" in command
        else tuple(str(value) for value in command)
    )
    if not inner:
        raise ValueError(f"Prior job has an empty inner command: {job.get('job_key')}")
    return inner


def _prior_jobs_by_task_key(
    prior_jobs: Sequence[object],
) -> dict[str, Mapping[str, Any]]:
    """Index standalone scheduler jobs and grouped tasks through one view."""

    indexed: dict[str, Mapping[str, Any]] = {}
    for raw_job in prior_jobs:
        if not isinstance(raw_job, Mapping) or not raw_job.get("job_key"):
            continue
        job = dict(raw_job)
        indexed[str(job["job_key"])] = job
        group = job.get("execution_group")
        tasks = group.get("tasks") if isinstance(group, Mapping) else None
        if not isinstance(tasks, list):
            continue
        for raw_task in tasks:
            if not isinstance(raw_task, Mapping) or not raw_task.get("task_key"):
                continue
            task = dict(raw_task)
            task["job_key"] = str(task["task_key"])
            task["resources"] = job.get("resources")
            task["parent_execution_group"] = dict(group)
            indexed[str(task["task_key"])] = task
    return indexed


def _prior_group_limit(job: Mapping[str, Any], *, default: int) -> int:
    group = job.get("parent_execution_group")
    if not isinstance(group, Mapping):
        return int(default)
    return max(1, int(group.get("max_concurrent") or default))


def _resources(job: Mapping[str, Any]) -> LsfResources:
    payload = job.get("resources")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Prior job has no resources: {job.get('job_key')}")
    return LsfResources(
        queue=str(payload.get("queue") or ""),
        ncores=int(payload["ncores"]),
        mem_gb=int(payload["mem_gb"]),
        gpus=int(payload.get("gpus") or 0),
        walltime=(
            str(payload["walltime"]) if payload.get("walltime") is not None else None
        ),
        span_hosts=(
            int(payload["span_hosts"])
            if payload.get("span_hosts") is not None
            else None
        ),
        extra_lsf_args=tuple(
            str(value) for value in payload.get("extra_lsf_args") or ()
        ),
    )


def _replace_run_root(
    command: Sequence[str],
    *,
    prior_run_root: Path,
    recovery_run_root: Path,
) -> tuple[str, ...]:
    old = str(prior_run_root)
    new = str(recovery_run_root)
    return tuple(str(value).replace(old, new) for value in command)


def _recovery_keypoint_command(
    command: Sequence[str],
    *,
    prior_run_root: Path,
    recovery_run_root: Path,
) -> tuple[str, ...]:
    """Upgrade a prior collection-shard command to the required source label."""

    updated = list(
        _replace_run_root(
            command,
            prior_run_root=prior_run_root,
            recovery_run_root=recovery_run_root,
        )
    )
    module = "fisheye.utils.run_keypoints_with_registry_model"
    if module not in updated or "--output-parent" not in updated:
        raise ValueError("Prior keypoint recovery task is not a registry shard command.")
    output_parent_index = updated.index("--output-parent") + 1
    if output_parent_index >= len(updated) or updated[output_parent_index] != (
        "keypoint_shard_runs"
    ):
        raise ValueError("Keypoint recovery requires keypoint_shard_runs output.")
    flag = "--coordinate-contract-mode"
    if flag in updated:
        value_index = updated.index(flag) + 1
        if value_index >= len(updated) or updated[value_index] != "legacy_noncanonical":
            raise ValueError(
                "Keypoint recovery shard has an incompatible coordinate contract."
            )
    else:
        updated.extend((flag, "legacy_noncanonical"))
    return tuple(updated)


def build_plan(
    *,
    source_plan_path: Path,
    run_root: Path,
    recovery_label: str,
) -> KeypointRecoveryPlan:
    source_plan_path = source_plan_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    source = _read_json(source_plan_path)
    if (
        not isinstance(source, Mapping)
        or source.get("schema") not in SUPPORTED_PLAN_SCHEMAS
    ):
        raise ValueError(f"Unsupported clipped inference plan: {source_plan_path}")
    targets = source.get("targets")
    workflow_payload = source.get("lsf_workflow")
    prior_jobs = (
        workflow_payload.get("jobs") if isinstance(workflow_payload, Mapping) else None
    )
    if not isinstance(targets, list) or not targets or not isinstance(prior_jobs, list):
        raise ValueError("Source plan requires targets and an LSF workflow.")
    label = safe_component(recovery_label, default="keypoint_recovery", max_length=72)
    repo = Path(str(source["repo"])).expanduser().resolve()
    registry = Path(str(source["registry"])).expanduser().resolve()
    prior_run_root = Path(str(source["run_root"])).expanduser().resolve()
    prior_by_key = _prior_jobs_by_task_key(prior_jobs)
    preflight = prepare_keypoint_recovery(source_plan_path, apply=False)
    if preflight.get("status") != "ok":
        raise RuntimeError("Keypoint recovery preflight did not pass.")

    jobs: list[LsfJob] = []
    preparation_key = "prepare_keypoint_recovery"
    preparation_report = run_root / "recovery" / "preparation.json"
    jobs.append(
        _job(
            workflow_id=label,
            repo=repo,
            run_root=run_root,
            job_key=preparation_key,
            stage="keypoint_recovery_preparation",
            command=(
                "scripts/py",
                "-m",
                "fisheye.utils.prepare_clipped_keypoint_recovery",
                "--plan",
                str(source_plan_path),
                "--apply",
                "--output-json",
                str(preparation_report),
            ),
            resources=LsfResources(queue="short", ncores=2, mem_gb=16, walltime="1:00"),
            expected_outputs=(preparation_report,),
        )
    )

    validation_keys: list[str] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("Source plan target must be an object.")
        target_safe = safe_component(
            str(target["target_id"]), default="target", max_length=56
        )
        zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
        keypoint_tasks: list[LsfExecutionTask] = []
        keypoint_priors: list[Mapping[str, Any]] = []
        for clip in target["clips"]:
            clip_id = str(clip["clip_id"])
            key = f"keypoints:{target_safe}:{clip_id}"
            prior = prior_by_key[key]
            command = _recovery_keypoint_command(
                _inner_command(prior),
                prior_run_root=prior_run_root,
                recovery_run_root=run_root,
            )
            keypoint_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=key,
                    stage="keypoints",
                    command=command,
                    expected_outputs=(
                        zarr_path
                        / "keypoint_shard_runs"
                        / str(clip["keypoint_shard_run"])
                        / "zarr.json",
                    ),
                    array_indexed=True,
                )
            )
            keypoint_priors.append(prior)
        keypoint_array_key = f"keypoints_array:{target_safe}"
        jobs.append(
            _task_group_job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=keypoint_array_key,
                stage="keypoints",
                tasks=keypoint_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=_prior_group_limit(keypoint_priors[0], default=4),
                resources=_resources(keypoint_priors[0]),
                upstream=(preparation_key,),
            )
        )

        finalize_key = f"keypoint_finalize:{target_safe}"
        finalize_prior = prior_by_key[finalize_key]
        jobs.append(
            _job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=finalize_key,
                stage="keypoint_finalize",
                command=_replace_run_root(
                    _inner_command(finalize_prior),
                    prior_run_root=prior_run_root,
                    recovery_run_root=run_root,
                ),
                resources=_resources(finalize_prior),
                upstream=(keypoint_array_key,),
                expected_outputs=(
                    zarr_path
                    / "keypoints_runs"
                    / str(target["keypoint_run"])
                    / "zarr.json",
                ),
            )
        )
        refine_key = f"keypoint_refine:{target_safe}"
        refine_prior = prior_by_key[refine_key]
        jobs.append(
            _job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=refine_key,
                stage="keypoint_refine",
                command=_replace_run_root(
                    _inner_command(refine_prior),
                    prior_run_root=prior_run_root,
                    recovery_run_root=run_root,
                ),
                resources=_resources(refine_prior),
                upstream=(finalize_key,),
                expected_outputs=(
                    zarr_path
                    / "refined_keypoints_runs"
                    / str(target["refined_keypoint_run"])
                    / "zarr.json",
                ),
            )
        )

        package_tasks: list[LsfExecutionTask] = []
        package_priors: list[Mapping[str, Any]] = []
        for clip in target["clips"]:
            clip_id = str(clip["clip_id"])
            key = f"mask_package:{target_safe}:{clip_id}"
            prior = prior_by_key[key]
            package_tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=key,
                    stage="subject_mask_refine_package",
                    command=_replace_run_root(
                        _inner_command(prior),
                        prior_run_root=prior_run_root,
                        recovery_run_root=run_root,
                    ),
                    expected_outputs=(Path(str(clip["package_path"])),),
                    array_indexed=True,
                )
            )
            package_priors.append(prior)
        package_array_key = f"mask_package_array:{target_safe}"
        jobs.append(
            _task_group_job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=package_array_key,
                stage="subject_mask_refine_package",
                tasks=package_tasks,
                mode=LsfExecutionMode.ARRAY,
                max_concurrent=_prior_group_limit(package_priors[0], default=4),
                resources=_resources(package_priors[0]),
                upstream=(refine_key,),
            )
        )

        receipt_composed = (
            source.get("subject_mask_publication_profile") == "receipt_composed_v1"
        )
        terminal_mask_key = (
            f"mask_publish:{target_safe}"
            if receipt_composed
            else f"mask_import:{target_safe}"
        )
        terminal_prior = prior_by_key[terminal_mask_key]
        jobs.append(
            _job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=terminal_mask_key,
                stage=(
                    "subject_mask_collection_publication"
                    if receipt_composed
                    else "subject_mask_collection_import"
                ),
                command=_replace_run_root(
                    _inner_command(terminal_prior),
                    prior_run_root=prior_run_root,
                    recovery_run_root=run_root,
                ),
                resources=(
                    _resources(terminal_prior)
                    if receipt_composed
                    else LsfResources(
                        queue="local", ncores=8, mem_gb=32, walltime="3:00"
                    )
                ),
                upstream=(package_array_key,),
                expected_outputs=(
                    (
                        (
                            zarr_path
                            / "subject_mask_bundle_runs"
                            / str(target["subject_mask_bundle_id"])
                            / "zarr.json"
                        )
                        if receipt_composed
                        else (
                            zarr_path
                            / "refined_subject_masks_runs"
                            / str(target["refined_subject_mask_run"])
                            / "zarr.json"
                        )
                    ),
                ),
            )
        )
        validation_key = f"validate:{target_safe}"
        validation_report = run_root / "validation" / f"{target_safe}.json"
        validation_prior = prior_by_key[validation_key]
        jobs.append(
            _job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key=validation_key,
                stage="validation",
                command=(
                    "scripts/py",
                    "-m",
                    "fisheye.cluster.clipped_inference_validate",
                    "--plan",
                    str(run_root / "plan.json"),
                    "--target-id",
                    str(target["target_id"]),
                    "--output-json",
                    str(validation_report),
                ),
                resources=_resources(validation_prior),
                upstream=(terminal_mask_key,),
                expected_outputs=(validation_report,),
            )
        )
        validation_keys.append(validation_key)

    registry_report = run_root / "registry" / "reconcile.json"
    registry_prior = prior_by_key["registry_finalize"]
    jobs.append(
        _job(
            workflow_id=label,
            repo=repo,
            run_root=run_root,
            job_key="registry_finalize",
            stage="registry_finalize",
            command=(
                "scripts/py",
                "-m",
                "fisheye.cluster.clipped_inference_registry_finalize",
                "--plan",
                str(run_root / "plan.json"),
                "--output-json",
                str(registry_report),
            ),
            resources=_resources(registry_prior),
            upstream=tuple(validation_keys),
            expected_outputs=(registry_report,),
        )
    )
    if bool(source.get("cleanup_nrs_after_success")):
        cleanup_prior = prior_by_key["nrs_cleanup"]
        jobs.append(
            _job(
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                job_key="nrs_cleanup",
                stage="nrs_cleanup",
                command=_replace_run_root(
                    _inner_command(cleanup_prior),
                    prior_run_root=prior_run_root,
                    recovery_run_root=run_root,
                ),
                resources=_resources(cleanup_prior),
                upstream=("registry_finalize",),
                expected_outputs=(run_root / "cleanup" / "nrs_cleanup.json",),
            )
        )

    workflow = LsfWorkflow(
        workflow_id=label,
        family=FAMILY,
        jobs=tuple(jobs),
        metadata={
            "recovery_schema": RECOVERY_SCHEMA,
            "source_plan": str(source_plan_path),
            "reused_completed_raw_subject_masks": True,
            "reused_completed_roi_caches": True,
        },
    )
    payload = {
        **{key: value for key, value in source.items() if key != "lsf_workflow"},
        "schema": PLAN_SCHEMA,
        "run_label": label,
        "workflow_id": label,
        "run_root": str(run_root),
        "repo": str(repo),
        "registry": str(registry),
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "source_plan": str(source_plan_path),
            "preflight": preflight,
        },
    }
    return KeypointRecoveryPlan(
        payload=payload, workflow=workflow, repo=repo, run_root=run_root
    )


def materialize_plan(plan: KeypointRecoveryPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if (
            existing != payload
            or not lsf_path.is_file()
            or _read_json(lsf_path) != plan.workflow.to_json()
        ):
            raise FileExistsError(
                f"Recovery run root contains different immutable evidence: {plan.run_root}"
            )
        return existing
    for name in (
        "logs",
        "status",
        "progress",
        "recovery",
        "validation",
        "registry",
        "cleanup",
    ):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(plan: KeypointRecoveryPlan, *, submit_host: str) -> dict[str, Any]:
    materialize_plan(plan)
    submission = plan.run_root / "lsf_submission.json"
    if submission.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission}")
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission,
        runner=build_ssh_bsub_runner(submit_host),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--recovery-label", required=True)
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        source_plan_path=args.source_plan,
        run_root=args.run_root,
        recovery_label=args.recovery_label,
    )
    result = (
        apply_plan(plan, submit_host=args.submit_host)
        if args.apply
        else materialize_plan(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "job_count": len(plan.workflow.jobs),
        "plan_path": str(plan.run_root / "plan.json"),
        "submission_path": str(plan.run_root / "lsf_submission.json"),
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['job_count']} jobs; plan={summary['plan_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
