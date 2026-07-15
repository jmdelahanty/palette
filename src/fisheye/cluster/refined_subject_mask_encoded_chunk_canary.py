"""Plan and submit a two-package decoded-versus-encoded mask import canary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.clipped_inference import DEFAULT_REPO, build_ssh_bsub_runner
from fisheye.cluster.clipped_lsf import build_job as _job
from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfJob,
    LsfResources,
    LsfWorkflow,
    submit_lsf_workflow,
    write_json_snapshot,
)


CANARY_SCHEMA = "palette.refined_subject_mask_encoded_chunk_canary.v1"
CANARY_FAMILY = "refined_subject_mask_encoded_chunk_canary"


@dataclass(frozen=True)
class EncodedChunkCanaryPlan:
    payload: Mapping[str, Any]
    workflow: LsfWorkflow
    repo: Path
    run_root: Path

    def to_json(self) -> dict[str, Any]:
        return {**dict(self.payload), "lsf_workflow": self.workflow.to_json()}


def _package_preflight(package_paths: Sequence[Path]) -> list[dict[str, Any]]:
    if len(package_paths) != 2:
        raise ValueError("The encoded-chunk A/B canary requires exactly two adjacent clip packages.")
    rows: list[dict[str, Any]] = []
    for package in package_paths:
        resolved = package.expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise FileNotFoundError(f"Canary source package is missing or empty: {resolved}")
        rows.append({"path": str(resolved), "size_bytes": int(resolved.stat().st_size)})
    return rows


def build_plan(
    *,
    package_paths: Sequence[Path],
    run_root: Path,
    encoded_package_dir: Path,
    canary_label: str,
    repo: Path = DEFAULT_REPO,
) -> EncodedChunkCanaryPlan:
    run_root = run_root.expanduser().resolve()
    encoded_package_dir = encoded_package_dir.expanduser().resolve()
    repo = repo.expanduser().resolve()
    label = safe_component(canary_label, default="encoded_mask_canary", max_length=72)
    packages = _package_preflight(package_paths)
    source_packages = [Path(str(row["path"])) for row in packages]
    canary_zarr = run_root / "canary.zarr"
    grid_manifest = run_root / "grid" / "global_mask_chunk_grid.json"
    report_path = run_root / "validation" / "ab_report.json"
    baseline_run = "refined_subject_masks_v1_decoded_baseline"
    encoded_run = "refined_subject_masks_v2_encoded"
    encoded_packages = [encoded_package_dir / source.name for source in source_packages]

    collisions = [canary_zarr, report_path, *encoded_packages]
    existing = [str(path) for path in collisions if path.exists()]
    if existing:
        raise FileExistsError(f"Encoded-chunk canary output collision: {existing!r}")

    prepare_command = [
        "scripts/py", "-m", "fisheye.utils.prepare_encoded_subject_mask_import_canary",
    ]
    for package in source_packages:
        prepare_command.extend(["--package", str(package)])
    prepare_command.extend(
        [
            "--output-zarr", str(canary_zarr),
            "--grid-manifest", str(grid_manifest),
            "--json",
        ]
    )
    jobs: list[LsfJob] = [
        _job(
            workflow_id=label,
            family=CANARY_FAMILY,
            repo=repo,
            run_root=run_root,
            job_key="prepare",
            stage="encoded_mask_canary_prepare",
            command=prepare_command,
            resources=LsfResources(queue="short", ncores=2, mem_gb=16, walltime="1:00"),
            expected_outputs=(canary_zarr / "zarr.json", grid_manifest),
        )
    ]

    baseline_command = [
        "scripts/py", "-m", "fisheye.utils.import_refined_subject_mask_clip_packages",
        "--zarr", str(canary_zarr),
        "--output-run", baseline_run,
        "--array-copy-workers", "8",
        "--json",
    ]
    for package in source_packages:
        baseline_command.extend(["--package", str(package)])
    jobs.append(
        _job(
            workflow_id=label,
            family=CANARY_FAMILY,
            repo=repo,
            run_root=run_root,
            job_key="baseline_import",
            stage="encoded_mask_canary_v1_import",
            command=baseline_command,
            resources=LsfResources(queue="local", ncores=8, mem_gb=48, walltime="3:00"),
            upstream=("prepare",),
            expected_outputs=(
                canary_zarr / "refined_subject_masks_runs" / baseline_run / "zarr.json",
            ),
        )
    )

    conversion_keys: list[str] = []
    for index, (source, output) in enumerate(zip(source_packages, encoded_packages, strict=True)):
        key = f"convert:{index:02d}"
        conversion_keys.append(key)
        jobs.append(
            _job(
                workflow_id=label,
                family=CANARY_FAMILY,
                repo=repo,
                run_root=run_root,
                job_key=key,
                stage="encoded_mask_canary_package_v2",
                command=(
                    "scripts/py", "-m", "fisheye.utils.convert_refined_subject_mask_clip_package_v2",
                    "--source-package", str(source),
                    "--output-package", str(output),
                    "--grid-manifest", str(grid_manifest),
                    "--copy-workers", "8",
                    "--json",
                ),
                resources=LsfResources(queue="short", ncores=8, mem_gb=32, walltime="1:00"),
                upstream=("prepare",),
                expected_outputs=(output,),
            )
        )

    encoded_command = [
        "scripts/py", "-m", "fisheye.utils.import_refined_subject_mask_clip_packages",
        "--zarr", str(canary_zarr),
        "--output-run", encoded_run,
        "--array-copy-workers", "8",
        "--encoded-copy-workers", "32",
        "--json",
    ]
    for package in encoded_packages:
        encoded_command.extend(["--package", str(package)])
    jobs.append(
        _job(
            workflow_id=label,
            family=CANARY_FAMILY,
            repo=repo,
            run_root=run_root,
            job_key="encoded_import",
            stage="encoded_mask_canary_v2_import",
            command=encoded_command,
            resources=LsfResources(queue="local", ncores=8, mem_gb=48, walltime="3:00"),
            upstream=("baseline_import", *conversion_keys),
            expected_outputs=(
                canary_zarr / "refined_subject_masks_runs" / encoded_run / "zarr.json",
            ),
        )
    )
    jobs.append(
        _job(
            workflow_id=label,
            family=CANARY_FAMILY,
            repo=repo,
            run_root=run_root,
            job_key="validate",
            stage="encoded_mask_canary_validate",
            command=(
                "scripts/py", "-m", "fisheye.utils.validate_encoded_subject_mask_import_canary",
                "--zarr", str(canary_zarr),
                "--baseline-run", baseline_run,
                "--encoded-run", encoded_run,
                "--sample-row-chunks", "16",
                "--output-json", str(report_path),
                "--json",
            ),
            resources=LsfResources(queue="short", ncores=4, mem_gb=32, walltime="1:00"),
            upstream=("encoded_import",),
            expected_outputs=(report_path,),
        )
    )

    workflow = LsfWorkflow(
        workflow_id=label,
        family=CANARY_FAMILY,
        jobs=tuple(jobs),
        metadata={
            "family": CANARY_FAMILY,
            "schema": CANARY_SCHEMA,
        },
    )
    payload = {
        "schema": CANARY_SCHEMA,
        "canary_label": label,
        "repo": str(repo),
        "run_root": str(run_root),
        "canary_zarr": str(canary_zarr),
        "grid_manifest": str(grid_manifest),
        "encoded_package_dir": str(encoded_package_dir),
        "source_packages": packages,
        "encoded_packages": [str(path) for path in encoded_packages],
        "baseline_run": baseline_run,
        "encoded_run": encoded_run,
        "report_path": str(report_path),
    }
    return EncodedChunkCanaryPlan(
        payload=payload,
        workflow=workflow,
        repo=repo,
        run_root=run_root,
    )


def materialize_plan(plan: EncodedChunkCanaryPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing_lsf = json.loads(lsf_path.read_text(encoding="utf-8")) if lsf_path.is_file() else None
        if existing != payload or existing_lsf != plan.workflow.to_json():
            raise FileExistsError(f"Canary run root contains different immutable evidence: {plan.run_root}")
        return existing
    for name in ("logs", "status", "grid", "validation"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(plan: EncodedChunkCanaryPlan, *, submit_host: str) -> dict[str, Any]:
    materialize_plan(plan)
    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Canary submission evidence already exists: {submission_path}")
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=build_ssh_bsub_runner(submit_host),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", dest="packages", action="append", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--encoded-package-dir", required=True, type=Path)
    parser.add_argument("--canary-label", required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        package_paths=args.packages,
        run_root=args.run_root,
        encoded_package_dir=args.encoded_package_dir,
        canary_label=args.canary_label,
        repo=args.repo,
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
