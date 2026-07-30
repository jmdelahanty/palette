"""Resume one clipped-inference DAG after a detect-quality geometry failure.

The recovery preserves the original immutable artifact names, validates and
repairs only the missing source geometry metadata, and resubmits the original
workflow tail from collection detection quality onward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import (
    DEFAULT_REPO,
    FAMILY,
    PLAN_SCHEMA,
    SUPPORTED_PLAN_SCHEMAS,
    build_ssh_bsub_runner,
)
from fisheye.cluster.clipped_inference_keypoint_recovery import (
    _inner_command,
    _resources,
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


RECOVERY_SCHEMA = "palette.clipped_inference_detect_quality_recovery.v1"


@dataclass(frozen=True)
class DetectQualityRecoveryPlan:
    payload: Mapping[str, Any]
    workflow: LsfWorkflow
    repo: Path
    run_root: Path
    source_detection_plan: Path
    recovery_detection_plan: Path

    def to_json(self) -> dict[str, Any]:
        return {**dict(self.payload), "lsf_workflow": self.workflow.to_json()}


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _rewrite_value(
    value: str,
    *,
    prior_repo: Path,
    repo: Path,
    prior_run_root: Path,
    run_root: Path,
) -> str:
    # The prior repo may itself be nested under the prior run root, so replace
    # that more-specific prefix first.
    return str(value).replace(str(prior_repo), str(repo)).replace(
        str(prior_run_root), str(run_root)
    )


def _rewrite_values(
    values: Sequence[object],
    *,
    prior_repo: Path,
    repo: Path,
    prior_run_root: Path,
    run_root: Path,
) -> tuple[str, ...]:
    return tuple(
        _rewrite_value(
            str(value),
            prior_repo=prior_repo,
            repo=repo,
            prior_run_root=prior_run_root,
            run_root=run_root,
        )
        for value in values
    )


def _runtime_values(job: Mapping[str, Any], flag: str) -> tuple[str, ...]:
    command = job.get("command")
    if not isinstance(command, list):
        return ()
    outer = command[: command.index("--")] if "--" in command else command
    return tuple(
        str(outer[index + 1])
        for index, value in enumerate(outer[:-1])
        if value == flag
    )


def _quality_array_contract(path: Path, *, shape: tuple[int, ...], dtype: str) -> None:
    payload = _read_json(path / "zarr.json")
    if payload.get("shape") != list(shape) or payload.get("data_type") != dtype:
        raise RuntimeError(
            f"Completed quality array contract mismatch at {path}: "
            f"shape={payload.get('shape')!r}, data_type={payload.get('data_type')!r}."
        )


def _validate_complete_quality(
    target: Mapping[str, Any],
    *,
    zarr_path: Path,
    source_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    quality_path = zarr_path / str(target["detect_quality_group_path"])
    metadata = quality_path / "zarr.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Completed quality output is missing: {metadata}")
    payload = _read_json(metadata)
    attrs = payload.get("attributes") if isinstance(payload, Mapping) else None
    if not isinstance(attrs, Mapping):
        raise RuntimeError(f"Completed quality output has no attributes: {metadata}")
    expected_source_path = str(target["detect_quality_source_group_path"])
    expected_rows = int(source_attrs.get("source_row_count") or -1)
    expected_frames = int(source_attrs.get("recording_frame_count") or -1)
    expected = {
        "palette_run_completion_status": "complete",
        "schema_id": "palette.detect_quality_collection.v2",
        "source_detection_group_path": expected_source_path,
        "source_row_count": expected_rows,
        "recording_frame_count": expected_frames,
        "source_video_width": int(source_attrs.get("source_video_width") or -1),
        "source_video_height": int(source_attrs.get("source_video_height") or -1),
    }
    mismatches = {
        key: {"expected": value, "observed": attrs.get(key)}
        for key, value in expected.items()
        if attrs.get(key) != value
    }
    validation = attrs.get("collection_quality_validation")
    if not isinstance(validation, Mapping):
        mismatches["collection_quality_validation"] = {
            "expected": "complete validation object",
            "observed": validation,
        }
    else:
        required_validation = {
            "status": "complete",
            "instance_key_exact": True,
            "instance_key_unique": True,
            "arrays_indexed_sharded": True,
            "trace_ranges_complete_nonoverlapping": True,
            "source_rows_canonical_frame_order": True,
            "row_count": expected_rows,
            "recording_frame_count": expected_frames,
        }
        for key, value in required_validation.items():
            if validation.get(key) != value:
                mismatches[f"collection_quality_validation.{key}"] = {
                    "expected": value,
                    "observed": validation.get(key),
                }
    if mismatches:
        raise RuntimeError(
            "Completed quality output failed recovery validation: "
            + json.dumps(mismatches, sort_keys=True)
        )
    _quality_array_contract(
        quality_path / "instance_key", shape=(expected_rows,), dtype="uint64"
    )
    _quality_array_contract(
        quality_path / "detection_quality_labels", shape=(expected_rows,), dtype="int8"
    )
    _quality_array_contract(
        quality_path / "quality_flags", shape=(expected_frames,), dtype="int8"
    )
    return {
        "status": "complete",
        "quality_group_path": str(quality_path.relative_to(zarr_path)),
        "source_group_path": expected_source_path,
        "row_count": expected_rows,
        "recording_frame_count": expected_frames,
        "instance_key_exact": True,
        "instance_key_unique": True,
        "arrays_indexed_sharded": True,
    }


def _preflight(
    source: Mapping[str, Any],
    *,
    reuse_complete_quality: bool = False,
) -> dict[str, Any]:
    targets = source.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], Mapping):
        raise ValueError("Detect-quality recovery currently requires one source-plan target.")
    target = targets[0]
    zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
    source_path = (
        zarr_path
        / "detect_collection_sources"
        / str(target["detect_quality_source_run"])
    )
    source_metadata = source_path / "zarr.json"
    if not source_metadata.is_file():
        raise FileNotFoundError(f"Completed detect-quality source is missing: {source_metadata}")
    source_payload = _read_json(source_metadata)
    attrs = source_payload.get("attributes") if isinstance(source_payload, Mapping) else None
    if not isinstance(attrs, Mapping) or attrs.get("palette_run_completion_status") != "complete":
        raise RuntimeError(f"Detect-quality source is not complete: {source_path}")
    if attrs.get("schema_id") not in {
        "palette.clipped_detect_quality_source.v1",
        "palette.clipped_detect_quality_source.v2",
    }:
        raise RuntimeError(f"Unsupported detect-quality source schema: {attrs.get('schema_id')!r}")
    if reuse_complete_quality:
        width = int(attrs.get("source_video_width") or -1)
        height = int(attrs.get("source_video_height") or -1)
        if width <= 0 or height <= 0:
            raise RuntimeError("Complete-quality reuse requires repaired source geometry.")
        if attrs.get("schema_id") == "palette.clipped_detect_quality_source.v1":
            repair = attrs.get("full_frame_geometry_repair")
            if not isinstance(repair, Mapping) or repair.get("status") != "complete":
                raise RuntimeError(
                    "Historical v1 source has no complete full-frame geometry repair."
                )
    quality_validation = (
        _validate_complete_quality(target, zarr_path=zarr_path, source_attrs=attrs)
        if reuse_complete_quality
        else None
    )

    planned_outputs: list[Path] = []
    for clip in target["clips"]:
        planned_outputs.extend(
            [
                zarr_path / str(clip["refined_detect_group_path"]),
                zarr_path / "crop_runs" / str(clip["proxy_crop_run"]),
                zarr_path / "keypoint_shard_runs" / str(clip["keypoint_shard_run"]),
                zarr_path / "subject_mask_shard_runs" / str(clip["subject_mask_shard_run"]),
                Path(str(clip["package_path"])),
            ]
        )
    planned_outputs.extend(
        [
            *(
                ()
                if reuse_complete_quality
                else (zarr_path / str(target["detect_quality_group_path"]),)
            ),
            zarr_path / "experiment_index" / "finalized_runs" / str(target["collection_id"]),
            zarr_path / "crop_runs" / str(target["merged_proxy_crop_run"]),
            zarr_path / "keypoints_runs" / str(target["keypoint_run"]),
            zarr_path / "refined_keypoints_runs" / str(target["refined_keypoint_run"]),
            zarr_path / "refined_subject_masks_runs" / str(target["refined_subject_mask_run"]),
            Path(str(target["cache_dir"])),
            Path(str(target["package_dir"])),
        ]
    )
    collisions = [str(path) for path in planned_outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Recovery refuses existing downstream outputs: " + ", ".join(collisions)
        )
    return {
        "status": "ok",
        "target_id": str(target["target_id"]),
        "source_group_path": str(source_path.relative_to(zarr_path)),
        "source_schema": str(attrs["schema_id"]),
        "source_completion_status": "complete",
        "downstream_output_count_checked": len(planned_outputs),
        "downstream_outputs_absent": True,
        "reused_complete_quality": bool(reuse_complete_quality),
        "quality_validation": quality_validation,
    }


def _resources_with_current_queue_limits(job: Mapping[str, Any]) -> LsfResources:
    resources = _resources(job)
    if resources.queue == "short" and resources.walltime != "1:00":
        return replace(resources, walltime="1:00")
    return resources


def _clone_prior_job(
    prior: Mapping[str, Any],
    *,
    workflow_id: str,
    repo: Path,
    run_root: Path,
    prior_repo: Path,
    prior_run_root: Path,
    upstream: Sequence[str],
) -> LsfJob:
    job_key = str(prior["job_key"])
    metadata = prior.get("metadata")
    stage = str(metadata.get("stage") if isinstance(metadata, Mapping) else job_key)
    resources = _resources_with_current_queue_limits(prior)
    group = prior.get("execution_group")
    if isinstance(group, Mapping):
        raw_tasks = group.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError(f"Prior execution group has no tasks: {job_key}")
        mode = LsfExecutionMode(str(group["mode"]))
        tasks: list[LsfExecutionTask] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"Prior execution group contains a non-object task: {job_key}")
            command = raw_task.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"Prior execution task has no command: {raw_task.get('task_key')}")
            tasks.append(
                _execution_task(
                    run_root=run_root,
                    task_key=str(raw_task["task_key"]),
                    stage=str(raw_task["stage"]),
                    command=_rewrite_values(
                        command,
                        prior_repo=prior_repo,
                        repo=repo,
                        prior_run_root=prior_run_root,
                        run_root=run_root,
                    ),
                    expected_outputs=tuple(
                        Path(value)
                        for value in _rewrite_values(
                            raw_task.get("expected_outputs") or (),
                            prior_repo=prior_repo,
                            repo=repo,
                            prior_run_root=prior_run_root,
                            run_root=run_root,
                        )
                    ),
                    cleanup_paths=_rewrite_values(
                        raw_task.get("cleanup_paths") or (),
                        prior_repo=prior_repo,
                        repo=repo,
                        prior_run_root=prior_run_root,
                        run_root=run_root,
                    ),
                    array_indexed=mode is LsfExecutionMode.ARRAY,
                )
            )
        return _task_group_job(
            workflow_id=workflow_id,
            repo=repo,
            run_root=run_root,
            job_key=job_key,
            stage=stage,
            tasks=tasks,
            mode=mode,
            max_concurrent=int(group["max_concurrent"]),
            resources=resources,
            upstream=upstream,
        )

    command = _rewrite_values(
        _inner_command(prior),
        prior_repo=prior_repo,
        repo=repo,
        prior_run_root=prior_run_root,
        run_root=run_root,
    )
    expected_outputs = tuple(
        Path(value)
        for value in _rewrite_values(
            _runtime_values(prior, "--expected-output"),
            prior_repo=prior_repo,
            repo=repo,
            prior_run_root=prior_run_root,
            run_root=run_root,
        )
    )
    cleanup_paths = _rewrite_values(
        _runtime_values(prior, "--cleanup-path"),
        prior_repo=prior_repo,
        repo=repo,
        prior_run_root=prior_run_root,
        run_root=run_root,
    )
    return _job(
        workflow_id=workflow_id,
        repo=repo,
        run_root=run_root,
        job_key=job_key,
        stage=stage,
        command=command,
        resources=resources,
        upstream=upstream,
        expected_outputs=expected_outputs,
        cleanup_paths=cleanup_paths,
    )


def build_plan(
    *,
    source_plan_path: Path,
    run_root: Path,
    recovery_label: str,
    repo: Path = DEFAULT_REPO,
    reuse_complete_quality: bool = False,
) -> DetectQualityRecoveryPlan:
    source_plan_path = source_plan_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    repo = repo.expanduser().resolve()
    source = _read_json(source_plan_path)
    if (
        not isinstance(source, Mapping)
        or source.get("schema") not in SUPPORTED_PLAN_SCHEMAS
    ):
        raise ValueError(f"Unsupported clipped inference plan: {source_plan_path}")
    workflow_payload = source.get("lsf_workflow")
    prior_jobs = workflow_payload.get("jobs") if isinstance(workflow_payload, Mapping) else None
    if not isinstance(prior_jobs, list) or not prior_jobs:
        raise ValueError("Source plan has no LSF workflow jobs.")
    if not (repo / "scripts" / "py").is_file():
        raise FileNotFoundError(f"Recovery repository is not deployed: {repo}")
    preflight = _preflight(source, reuse_complete_quality=reuse_complete_quality)
    target = source["targets"][0]
    target_safe = safe_component(str(target["target_id"]), default="target", max_length=56)
    label = safe_component(recovery_label, default="detect_quality_recovery", max_length=72)
    prior_repo = Path(str(source["repo"])).expanduser().resolve()
    prior_run_root = Path(str(source["run_root"])).expanduser().resolve()
    source_detection_plan = Path(str(target["detection_plan_path"])).expanduser().resolve()
    recovery_detection_plan = run_root / "targets" / target_safe / "detection_plan.json"
    if not source_detection_plan.is_file():
        raise FileNotFoundError(source_detection_plan)

    repair_key = f"detect_quality_source_geometry_repair:{target_safe}"
    repair_report = run_root / "recovery" / f"{target_safe}.geometry.json"
    zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
    jobs: list[LsfJob] = []
    if not reuse_complete_quality:
        jobs.append(_job(
            workflow_id=label,
            repo=repo,
            run_root=run_root,
            job_key=repair_key,
            stage="detect_quality_source_geometry_repair",
            command=(
                "scripts/py",
                "-m",
                "fisheye.utils.repair_clipped_detect_quality_source_geometry",
                str(zarr_path),
                "--plan",
                str(recovery_detection_plan),
                "--source-run",
                str(target["detect_quality_source_run"]),
                "--recording-frame-index",
                str(Path(str(target["recording_dir"])) / "recording_frame_index.parquet"),
                "--apply",
                "--output-json",
                str(repair_report),
                "--json",
            ),
            resources=LsfResources(queue="short", ncores=2, mem_gb=16, walltime="1:00"),
            expected_outputs=(repair_report,),
        ))

    quality_key = f"detect_quality:{target_safe}"
    start_key = (
        f"detect_refine_bundle:{target_safe}"
        if reuse_complete_quality
        else quality_key
    )
    start = next(
        (index for index, job in enumerate(prior_jobs) if isinstance(job, Mapping) and job.get("job_key") == start_key),
        None,
    )
    if start is None:
        raise ValueError(f"Source workflow has no {start_key!r} job.")
    selected = [job for job in prior_jobs[start:] if isinstance(job, Mapping)]
    selected_keys = {str(job["job_key"]) for job in selected}
    source_key = f"detect_quality_source:{target_safe}"
    for prior in selected:
        dependency = prior.get("dependency")
        prior_upstream = (
            tuple(str(value) for value in dependency.get("upstream_job_keys") or ())
            if isinstance(dependency, Mapping)
            else ()
        )
        upstream = tuple(
            repair_key if value == source_key else value
            for value in prior_upstream
            if not (reuse_complete_quality and value == quality_key)
        )
        unknown = set(upstream) - selected_keys - {repair_key}
        if unknown:
            raise ValueError(
                f"Recovery job {prior['job_key']!r} depends on excluded jobs: {sorted(unknown)}"
            )
        jobs.append(
            _clone_prior_job(
                prior,
                workflow_id=label,
                repo=repo,
                run_root=run_root,
                prior_repo=prior_repo,
                prior_run_root=prior_run_root,
                upstream=upstream,
            )
        )

    workflow = LsfWorkflow(
        workflow_id=label,
        family=FAMILY,
        jobs=tuple(jobs),
        metadata={
            "recovery_schema": RECOVERY_SCHEMA,
            "source_plan": str(source_plan_path),
            "reused_completed_raw_detections": True,
            "reused_completed_quality_source_arrays": True,
            "reused_completed_detect_quality": bool(reuse_complete_quality),
            "first_recomputed_stage": (
                "detect_refine" if reuse_complete_quality else "detect_quality"
            ),
        },
    )
    recovery_target = json.loads(json.dumps(target))
    recovery_target["detection_plan_path"] = str(recovery_detection_plan)
    payload = {
        **{key: value for key, value in source.items() if key not in {"lsf_workflow", "targets"}},
        "schema": PLAN_SCHEMA,
        "run_label": label,
        "workflow_id": label,
        "run_root": str(run_root),
        "repo": str(repo),
        "target_count": 1,
        "targets": [recovery_target],
        "detect_quality_recovery": {
            "schema": RECOVERY_SCHEMA,
            "source_plan": str(source_plan_path),
            "preflight": preflight,
            "preserved_artifact_names": True,
            "source_array_payload_rewritten": False,
            "reused_complete_quality": bool(reuse_complete_quality),
        },
    }
    return DetectQualityRecoveryPlan(
        payload=payload,
        workflow=workflow,
        repo=repo,
        run_root=run_root,
        source_detection_plan=source_detection_plan,
        recovery_detection_plan=recovery_detection_plan,
    )


def materialize_plan(plan: DetectQualityRecoveryPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    detection_payload = _read_json(plan.source_detection_plan)
    if plan_path.exists():
        existing = _read_json(plan_path)
        if (
            existing != payload
            or not lsf_path.is_file()
            or _read_json(lsf_path) != plan.workflow.to_json()
            or not plan.recovery_detection_plan.is_file()
            or _read_json(plan.recovery_detection_plan) != detection_payload
        ):
            raise FileExistsError(
                f"Recovery run root contains different immutable evidence: {plan.run_root}"
            )
        return existing
    for name in (
        "logs",
        "status",
        "progress",
        "targets",
        "cache_jobs",
        "recovery",
        "validation",
        "registry",
        "cleanup",
    ):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    plan.recovery_detection_plan.parent.mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan.recovery_detection_plan, detection_payload)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(plan: DetectQualityRecoveryPlan, *, submit_host: str) -> dict[str, Any]:
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
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    parser.add_argument("--reuse-complete-quality", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        source_plan_path=args.source_plan,
        run_root=args.run_root,
        recovery_label=args.recovery_label,
        repo=args.repo,
        reuse_complete_quality=args.reuse_complete_quality,
    )
    result = apply_plan(plan, submit_host=args.submit_host) if args.apply else materialize_plan(plan)
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
        print(f"{summary['status']}: {summary['job_count']} jobs; plan={summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
