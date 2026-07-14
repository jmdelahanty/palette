"""Recover clipped inference after a collection-level mask import failure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import (
    FAMILY,
    PLAN_SCHEMA,
    _job,
    build_ssh_bsub_runner,
)
from fisheye.cluster.clipped_inference_keypoint_recovery import (
    _inner_command,
    _replace_run_root,
    _resources,
)
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfJob,
    LsfResources,
    LsfWorkflow,
    submit_lsf_workflow,
    write_json_snapshot,
)


RECOVERY_SCHEMA = "palette.clipped_inference_import_recovery.v1"


@dataclass(frozen=True)
class ImportRecoveryPlan:
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


def _group_attrs(path: Path) -> Mapping[str, Any] | None:
    metadata = path / "zarr.json"
    if not metadata.is_file():
        return None
    payload = _read_json(metadata)
    attrs = payload.get("attributes") if isinstance(payload, Mapping) else None
    return attrs if isinstance(attrs, Mapping) else {}


def _preflight(
    source: Mapping[str, Any],
    *,
    require_complete_import: bool = False,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for target in source["targets"]:
        zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
        refined_keypoints = zarr_path / "refined_keypoints_runs" / str(
            target["refined_keypoint_run"]
        )
        keypoint_attrs = _group_attrs(refined_keypoints)
        if not keypoint_attrs or keypoint_attrs.get("palette_run_completion_status") != "complete":
            raise RuntimeError(f"Refined keypoints are not complete: {refined_keypoints}")

        packages: list[dict[str, Any]] = []
        for clip in target["clips"]:
            package = Path(str(clip["package_path"])).expanduser().resolve()
            if not package.is_file() or package.stat().st_size <= 0:
                raise FileNotFoundError(f"Completed mask package is missing: {package}")
            packages.append(
                {
                    "clip_id": str(clip["clip_id"]),
                    "path": str(package),
                    "size_bytes": int(package.stat().st_size),
                }
            )

        output = zarr_path / "refined_subject_masks_runs" / str(
            target["refined_subject_mask_run"]
        )
        output_attrs = _group_attrs(output)
        output_status = (
            str(output_attrs.get("palette_run_completion_status") or "")
            if output_attrs is not None
            else "absent"
        )
        if output_status == "complete" and not require_complete_import:
            raise RuntimeError(f"Refined subject-mask output is already complete: {output}")
        if require_complete_import and output_status != "complete":
            raise RuntimeError(
                f"Validation-only recovery requires a complete refined subject-mask output: "
                f"{output} (status={output_status!r})"
            )
        reports.append(
            {
                "target_id": str(target["target_id"]),
                "analysis_zarr": str(zarr_path),
                "package_count": len(packages),
                "package_bytes": sum(row["size_bytes"] for row in packages),
                "packages": packages,
                "refined_keypoints_status": "complete",
                "partial_output": str(output),
                "partial_output_status": output_status,
                "import_action": (
                    "reuse_complete_output"
                    if require_complete_import
                    else "overwrite_incomplete_output"
                ),
            }
        )
    return {"status": "ok", "target_count": len(reports), "targets": reports}


def _with_overwrite(command: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in command)
    return values if "--overwrite" in values else (*values, "--overwrite")


def _replace_package_paths(
    command: Sequence[str],
    replacements: Mapping[str, str],
) -> tuple[str, ...]:
    values = [str(value) for value in command]
    index = 0
    while index < len(values):
        if values[index] == "--package" and index + 1 < len(values):
            values[index + 1] = str(replacements.get(values[index + 1], values[index + 1]))
            index += 2
            continue
        index += 1
    if "--encoded-copy-workers" not in values:
        values.extend(["--encoded-copy-workers", "32"])
    return tuple(values)


def build_plan(
    *,
    source_plan_path: Path,
    run_root: Path,
    recovery_label: str,
    convert_packages_v2: bool = False,
    validate_existing_complete_import: bool = False,
) -> ImportRecoveryPlan:
    if convert_packages_v2 and validate_existing_complete_import:
        raise ValueError(
            "Package conversion and validation-only recovery are mutually exclusive."
        )
    source_plan_path = source_plan_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    source = _read_json(source_plan_path)
    if not isinstance(source, Mapping) or source.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported clipped inference plan: {source_plan_path}")
    targets = source.get("targets")
    workflow_payload = source.get("lsf_workflow")
    prior_jobs = workflow_payload.get("jobs") if isinstance(workflow_payload, Mapping) else None
    if not isinstance(targets, list) or not targets or not isinstance(prior_jobs, list):
        raise ValueError("Source plan requires targets and an LSF workflow.")

    label = safe_component(recovery_label, default="import_recovery", max_length=72)
    repo = Path(str(source["repo"])).expanduser().resolve()
    registry = Path(str(source["registry"])).expanduser().resolve()
    prior_run_root = Path(str(source["run_root"])).expanduser().resolve()
    prior_by_key = {
        str(job["job_key"]): job
        for job in prior_jobs
        if isinstance(job, Mapping) and job.get("job_key")
    }
    preflight = (
        _preflight(source, require_complete_import=True)
        if validate_existing_complete_import
        else _preflight(source)
    )

    jobs: list[LsfJob] = []
    validation_keys: list[str] = []
    recovery_targets = json.loads(json.dumps(targets))
    recovery_target_by_id = {
        str(target["target_id"]): target
        for target in recovery_targets
    }
    for target in targets:
        target_safe = safe_component(str(target["target_id"]), default="target", max_length=56)
        zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
        import_upstream: tuple[str, ...] = ()
        package_replacements: dict[str, str] = {}
        if convert_packages_v2:
            package_dir = Path(str(target["package_dir"])).expanduser().resolve()
            encoded_dir = package_dir / "encoded_v2"
            grid_manifest = encoded_dir / "global_mask_chunk_grid.json"
            grid_key = f"mask_grid:{target_safe}"
            jobs.append(
                _job(
                    workflow_id=label,
                    repo=repo,
                    run_root=run_root,
                    job_key=grid_key,
                    stage="subject_mask_global_chunk_grid_recovery",
                    command=(
                        "scripts/py", "-m", "fisheye.utils.prepare_refined_subject_mask_chunk_grid",
                        "--zarr", str(zarr_path),
                        "--crop-run", str(target["merged_proxy_crop_run"]),
                        "--output-manifest", str(grid_manifest),
                        "--mask-label", "subject_body",
                        "--mask-label", "eye_left",
                        "--mask-label", "eye_right",
                        "--mask-label", "swim_bladder",
                        "--mask-height", "512", "--mask-width", "512",
                        "--dense-mask-row-chunk", "128", "--json",
                    ),
                    resources=LsfResources(queue="short", ncores=2, mem_gb=16, walltime="1:00"),
                    expected_outputs=(grid_manifest,),
                )
            )
            conversion_keys: list[str] = []
            payload_target = recovery_target_by_id[str(target["target_id"])]
            payload_target["global_mask_grid_manifest"] = str(grid_manifest)
            payload_target["encoded_mask_packages"] = True
            for clip, payload_clip in zip(target["clips"], payload_target["clips"], strict=True):
                source_package = Path(str(clip["package_path"])).expanduser().resolve()
                output_package = encoded_dir / source_package.name
                package_replacements[str(source_package)] = str(output_package)
                payload_clip["source_package_path"] = str(source_package)
                payload_clip["package_path"] = str(output_package)
                clip_safe = safe_component(str(clip["clip_id"]), default="clip", max_length=40)
                conversion_key = f"mask_package_v2:{target_safe}:{clip_safe}"
                conversion_keys.append(conversion_key)
                jobs.append(
                    _job(
                        workflow_id=label,
                        repo=repo,
                        run_root=run_root,
                        job_key=conversion_key,
                        stage="subject_mask_package_v2_conversion",
                        command=(
                            "scripts/py", "-m", "fisheye.utils.convert_refined_subject_mask_clip_package_v2",
                            "--source-package", str(source_package),
                            "--output-package", str(output_package),
                            "--grid-manifest", str(grid_manifest),
                            "--copy-workers", "8", "--json",
                        ),
                        resources=LsfResources(queue="short", ncores=8, mem_gb=32, walltime="1:00"),
                        upstream=(grid_key,),
                        expected_outputs=(output_package,),
                    )
                )
            import_upstream = tuple(conversion_keys)
        validation_upstream: tuple[str, ...] = ()
        if not validate_existing_complete_import:
            import_key = f"mask_import:{target_safe}"
            import_prior = prior_by_key[import_key]
            import_command = _with_overwrite(
                _replace_run_root(
                    _inner_command(import_prior),
                    prior_run_root=prior_run_root,
                    recovery_run_root=run_root,
                )
            )
            if convert_packages_v2:
                import_command = _replace_package_paths(import_command, package_replacements)
            jobs.append(
                _job(
                    workflow_id=label,
                    repo=repo,
                    run_root=run_root,
                    job_key=import_key,
                    stage="subject_mask_collection_import_recovery",
                    command=import_command,
                    resources=LsfResources(
                        queue="local", ncores=8, mem_gb=32, walltime="3:00"
                    ),
                    upstream=import_upstream,
                    expected_outputs=(
                        zarr_path
                        / "refined_subject_masks_runs"
                        / str(target["refined_subject_mask_run"])
                        / "zarr.json",
                    ),
                )
            )
            validation_upstream = (import_key,)

        validation_key = f"validate:{target_safe}"
        validation_prior = prior_by_key[validation_key]
        validation_report = run_root / "validation" / f"{target_safe}.json"
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
                upstream=validation_upstream,
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
            "reused_completed_mask_packages": True,
            "converted_mask_packages_v2": bool(convert_packages_v2),
            "validation_only_complete_import": bool(validate_existing_complete_import),
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
        "targets": recovery_targets,
        "import_recovery": {
            "schema": RECOVERY_SCHEMA,
            "source_plan": str(source_plan_path),
            "preflight": preflight,
            "validation_only_complete_import": bool(validate_existing_complete_import),
        },
    }
    return ImportRecoveryPlan(payload=payload, workflow=workflow, repo=repo, run_root=run_root)


def materialize_plan(plan: ImportRecoveryPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if existing != payload or not lsf_path.is_file() or _read_json(lsf_path) != plan.workflow.to_json():
            raise FileExistsError(f"Import recovery contains different evidence: {plan.run_root}")
        return existing
    for name in ("logs", "status", "validation", "registry", "cleanup"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(plan: ImportRecoveryPlan, *, submit_host: str) -> dict[str, Any]:
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
    parser.add_argument("--convert-packages-v2", action="store_true")
    parser.add_argument("--validate-existing-complete-import", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        source_plan_path=args.source_plan,
        run_root=args.run_root,
        recovery_label=args.recovery_label,
        convert_packages_v2=bool(args.convert_packages_v2),
        validate_existing_complete_import=bool(args.validate_existing_complete_import),
    )
    result = apply_plan(plan, submit_host=args.submit_host) if args.apply else materialize_plan(plan)
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "job_count": len(plan.workflow.jobs),
        "plan_path": str(plan.run_root / "plan.json"),
        "submission_path": str(plan.run_root / "lsf_submission.json"),
        "result": result if args.apply else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
