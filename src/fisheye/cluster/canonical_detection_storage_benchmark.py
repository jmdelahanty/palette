"""Plan and submit commit-pinned canonical detection storage benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from fisheye.cluster.lsf import (
    CommandRunner,
    LsfDependency,
    LsfExecutionGroup,
    LsfExecutionMode,
    LsfExecutionTask,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    build_runtime_command,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_matrix import (
    BenchmarkScale,
    StorageCandidateRequest,
)
from fisheye.shared.zarr.detection_benchmark_matrix import (
    plan_canonical_detection_benchmark_matrix,
    selectable_detection_candidate_requests,
)


FAMILY = "canonical_detection_storage_benchmark"


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


@dataclass(frozen=True)
class DetectionStorageBenchmarkLsfPlan:
    workflow_root: Path
    palette_repo: Path
    palette_commit: str
    matrix_manifest: dict[str, object]
    workflow: LsfWorkflow
    fixture_root: Path
    recording_identity: str

    @property
    def matrix_path(self) -> Path:
        return self.workflow_root / "matrix.json"

    @property
    def plan_path(self) -> Path:
        return self.workflow_root / "plan.json"

    @property
    def lsf_plan_path(self) -> Path:
        return self.workflow_root / "lsf_plan.json"

    @property
    def submission_path(self) -> Path:
        return self.workflow_root / "lsf_submission.json"

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.canonical_detection_storage_lsf_plan",
            "schema_version": 1,
            "workflow_id": self.workflow.workflow_id,
            "family": FAMILY,
            "workflow_root": str(self.workflow_root),
            "palette_repo": str(self.palette_repo),
            "palette_commit": self.palette_commit,
            "palette_clean": True,
            "fixture_root": str(self.fixture_root),
            "recording_identity": self.recording_identity,
            "matrix_path": str(self.matrix_path),
            "matrix_fingerprint": self.matrix_manifest["matrix_fingerprint"],
            "lsf_plan_path": str(self.lsf_plan_path),
            "submission_path": str(self.submission_path),
            "job_count": len(self.workflow.jobs),
            "block_count": len(self.matrix_manifest["repetitions"]),
            "candidate_count": len(self.matrix_manifest["candidates"]),
            "profile_promotion": False,
            "registry_updates": False,
            "selector_updates": False,
        }


def _require_safe_workflow_root(
    workflow_root: Path,
    *,
    benchmark_root: Path,
) -> Path:
    root = benchmark_root.expanduser().resolve()
    workflow = workflow_root.expanduser().resolve()
    if workflow == root or not workflow.is_relative_to(root):
        raise ValueError(f"Workflow root must be below {root}.")
    relative = workflow.relative_to(root)
    if not (
        "canonical_detection_storage" in relative.parts
        and "workflows" in relative.parts
    ):
        raise ValueError(
            "Workflow root must be below canonical_detection_storage/workflows."
        )
    return workflow


def build_plan(
    *,
    workflow_id: str,
    workflow_root: Path,
    benchmark_root: Path,
    fixture_root: Path,
    palette_repo: Path,
    recording_identity: str,
    scales: Sequence[BenchmarkScale],
    repetitions: int,
    repetition_start: int,
    seed: int,
    queue: str,
    ncores: int,
    mem_gb_per_slot: int,
    walltime: str,
    max_active_blocks: int,
    candidate_requests: Sequence[StorageCandidateRequest] | None = None,
    scratch_base: Path | None = None,
    keep_scratch: bool = False,
) -> DetectionStorageBenchmarkLsfPlan:
    workflow = _require_safe_workflow_root(
        workflow_root,
        benchmark_root=benchmark_root,
    )
    repo = palette_repo.expanduser().resolve()
    if not (repo / "scripts" / "py").is_file():
        raise ValueError(f"Palette repository is not runnable: {repo}")
    commit = _git_output(repo, "rev-parse", "HEAD")
    if _git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Palette repository must be clean: {repo}")
    fixture = fixture_root.expanduser().resolve()
    fixture_manifest_path = fixture / "fixture_manifest.json"
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    if (
        fixture_manifest.get("status") != "published_immutable"
        or fixture_manifest.get("benchmark_only") is not True
        or fixture_manifest.get("canonical") is not False
    ):
        raise ValueError("Fixture is not a published noncanonical benchmark source.")

    matrix = plan_canonical_detection_benchmark_matrix(
        matrix_id=workflow_id,
        scales=scales,
        destination_root=workflow / "candidates",
        repetitions=int(repetitions),
        repetition_start=int(repetition_start),
        seed=int(seed),
        candidate_requests=candidate_requests,
    )
    matrix_manifest = matrix.as_manifest()
    matrix_path = workflow / "matrix.json"
    lsf_plan_path = workflow / "lsf_plan.json"
    block_tasks: list[LsfExecutionTask] = []
    for repetition in matrix.repetitions:
        scale_id = repetition.scale_id
        repetition_index = repetition.repetition_index
        block_report = (
            workflow
            / "reports"
            / "blocks"
            / f"{scale_id}_repetition_{repetition_index:03d}.json"
        )
        command = [
            "/usr/bin/env",
            *(
                f"{key}={value}"
                for key, value in STORAGE_BENCHMARK_THREAD_ENVIRONMENT.items()
            ),
            str(repo / "scripts" / "py"),
            "-m",
            "fisheye.diagnostics.run_canonical_detection_storage_benchmark_block",
            "--matrix",
            str(matrix_path),
            "--fixture-root",
            str(fixture),
            "--workflow-root",
            str(workflow),
            "--block-report",
            str(block_report),
            "--scale-id",
            scale_id,
            "--repetition-index",
            str(repetition_index),
            "--recording-identity",
            recording_identity,
            "--palette-repo",
            str(repo),
            "--expected-commit",
            commit,
        ]
        if scratch_base is not None:
            command.extend(["--scratch-base", str(scratch_base.expanduser().resolve())])
        if keep_scratch:
            command.append("--keep-scratch")
        block_tasks.append(
            LsfExecutionTask(
                task_key=f"block:{scale_id}:{repetition_index:03d}",
                stage="storage_benchmark_block",
                command=tuple(command),
                status_path=(
                    workflow
                    / "status"
                    / "blocks"
                    / (
                        f"{scale_id}_{repetition_index:03d}."
                        f"{RUNTIME_JOB_ID_TOKEN}.{RUNTIME_JOB_INDEX_TOKEN}.json"
                    )
                ),
                expected_outputs=(str(block_report),),
                metadata={
                    "scale_id": scale_id,
                    "repetition_index": repetition_index,
                    "candidate_count": len(repetition.trials),
                    "balanced_order": [
                        trial.candidate_id for trial in repetition.trials
                    ],
                    "native_thread_environment": dict(
                        STORAGE_BENCHMARK_THREAD_ENVIRONMENT
                    ),
                },
            )
        )
    if not block_tasks:
        raise ValueError("Benchmark plan contains no execution blocks.")
    block_job = LsfJob(
        job_key="benchmark_blocks",
        job_name=f"det_storage_{workflow_id}",
        command=(
            str(repo / "scripts" / "py"),
            "-m",
            "fisheye.cluster.lsf.task_group",
            "--plan",
            str(lsf_plan_path),
            "--job-key",
            "benchmark_blocks",
            "--cwd",
            str(repo),
        ),
        resources=LsfResources(
            queue=queue,
            ncores=int(ncores),
            mem_gb=int(mem_gb_per_slot),
            gpus=0,
            walltime=walltime,
            span_hosts=1,
        ),
        stdout_path=workflow / "logs" / "blocks.%J.%I.out",
        stderr_path=workflow / "logs" / "blocks.%J.%I.err",
        execution_group=LsfExecutionGroup(
            mode=LsfExecutionMode.ARRAY,
            tasks=tuple(block_tasks),
            max_concurrent=min(int(max_active_blocks), len(block_tasks)),
        ),
        metadata={
            "palette_repo": str(repo),
            "palette_commit": commit,
            "fixture_id": fixture_manifest["fixture_id"],
            "node_local_compute_required": True,
            "gpus": 0,
        },
    )
    aggregate = workflow / "aggregate.json"
    finalizer_payload = (
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.diagnostics.finalize_canonical_detection_storage_benchmark",
        "--matrix",
        str(matrix_path),
        "--workflow-root",
        str(workflow),
        "--output",
        str(aggregate),
    )
    finalizer_job = LsfJob(
        job_key="finalize",
        job_name=f"det_storage_{workflow_id}_finalize",
        command=build_runtime_command(
            finalizer_payload,
            status_path_template=(
                workflow / "status" / f"finalize.{RUNTIME_JOB_ID_TOKEN}.json"
            ),
            workflow_id=workflow_id,
            family=FAMILY,
            job_key="finalize",
            stage="aggregate",
            cwd=repo,
            expected_output_templates=(aggregate,),
            python_launcher=(str(repo / "scripts" / "py"),),
        ),
        resources=LsfResources(
            queue=queue,
            ncores=1,
            mem_gb=4,
            gpus=0,
            walltime="0:15",
            span_hosts=1,
        ),
        stdout_path=workflow / "logs" / "finalize.%J.out",
        stderr_path=workflow / "logs" / "finalize.%J.err",
        dependency=LsfDependency(("benchmark_blocks",)),
        metadata={
            "validation_only": True,
            "registry_updates": False,
            "selector_updates": False,
            "profile_promotion": False,
        },
    )
    lsf_workflow = LsfWorkflow(
        workflow_id=workflow_id,
        family=FAMILY,
        jobs=(block_job, finalizer_job),
        metadata={
            "palette_repo": str(repo),
            "palette_commit": commit,
            "fixture_root": str(fixture),
            "matrix_fingerprint": matrix_manifest["matrix_fingerprint"],
            "scratch_compute": "node_local_only",
            "published_reads": "shared_prfs_explicit_workload_only",
        },
    )
    return DetectionStorageBenchmarkLsfPlan(
        workflow_root=workflow,
        palette_repo=repo,
        palette_commit=commit,
        matrix_manifest=matrix_manifest,
        workflow=lsf_workflow,
        fixture_root=fixture,
        recording_identity=recording_identity,
    )


def materialize_plan(plan: DetectionStorageBenchmarkLsfPlan) -> dict[str, object]:
    manifest = plan.as_manifest()
    if plan.plan_path.exists():
        if (
            json.loads(plan.plan_path.read_text(encoding="utf-8")) != manifest
            or json.loads(plan.matrix_path.read_text(encoding="utf-8"))
            != plan.matrix_manifest
            or json.loads(plan.lsf_plan_path.read_text(encoding="utf-8"))
            != plan.workflow.to_json()
        ):
            raise FileExistsError(
                f"Workflow root contains different plan evidence: {plan.workflow_root}"
            )
        return manifest
    if plan.workflow_root.exists():
        raise FileExistsError(
            f"Workflow root exists without an identical plan: {plan.workflow_root}"
        )
    for name in ("logs", "status/blocks", "reports/blocks", "candidates"):
        (plan.workflow_root / name).mkdir(parents=True, exist_ok=True)
    write_json_snapshot(plan.matrix_path, plan.matrix_manifest)
    write_json_snapshot(plan.lsf_plan_path, plan.workflow.to_json())
    write_json_snapshot(plan.plan_path, manifest)
    return manifest


def apply_plan(
    plan: DetectionStorageBenchmarkLsfPlan,
    *,
    submit_host: str | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    materialize_plan(plan)
    if plan.submission_path.exists():
        raise FileExistsError(
            f"Workflow already has submission evidence: {plan.submission_path}"
        )
    if runner is not None and submit_host is not None:
        raise ValueError("Provide runner or submit_host, not both.")
    selected_runner = (
        runner
        if runner is not None
        else build_ssh_bsub_runner(submit_host or "login1-citrus-poller")
    )
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.palette_repo,
        plan_path=plan.lsf_plan_path,
        submission_path=plan.submission_path,
        runner=selected_runner,
    )


def _parse_scale(value: str) -> BenchmarkScale:
    fields = value.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "scale must be ID:N_FRAMES:N_INSTANCES:SOURCE_WIDTH:SOURCE_HEIGHT"
        )
    scale_id, *raw = fields
    try:
        n_frames, n_instances, width, height = map(int, raw)
        return BenchmarkScale.from_mapping(
            scale_id,
            {
                "n_frames": n_frames,
                "n_instances": n_instances,
                "source_width": width,
                "source_height": height,
            },
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_candidate_label(value: str) -> StorageCandidateRequest:
    requests = {
        request.label: request
        for request in selectable_detection_candidate_requests()
    }
    try:
        return requests[value]
    except KeyError as exc:
        choices = ", ".join(sorted(requests))
        raise argparse.ArgumentTypeError(
            f"unknown candidate label {value!r}; choose one of: {choices}"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--palette-repo", required=True, type=Path)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--scale", required=True, action="append", type=_parse_scale)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--repetition-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument("--queue", default="short")
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--mem-gb-per-slot", type=int, default=8)
    parser.add_argument("--walltime", default="0:30")
    parser.add_argument("--max-active-blocks", type=int, default=1)
    parser.add_argument(
        "--candidate-label",
        action="append",
        type=_parse_candidate_label,
        help=(
            "Initial byte-budget candidate label to retain; repeat for a "
            "reviewed shortlist. The default retains the complete sweep."
        ),
    )
    parser.add_argument("--scratch-base", type=Path)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--submit-host", default="login1-citrus-poller")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        workflow_id=args.workflow_id,
        workflow_root=args.workflow_root,
        benchmark_root=args.benchmark_root,
        fixture_root=args.fixture_root,
        palette_repo=args.palette_repo,
        recording_identity=args.recording_identity,
        scales=args.scale,
        repetitions=args.repetitions,
        repetition_start=args.repetition_start,
        seed=args.seed,
        queue=args.queue,
        ncores=args.ncores,
        mem_gb_per_slot=args.mem_gb_per_slot,
        walltime=args.walltime,
        max_active_blocks=args.max_active_blocks,
        candidate_requests=args.candidate_label,
        scratch_base=args.scratch_base,
        keep_scratch=bool(args.keep_scratch),
    )
    result = (
        apply_plan(plan, submit_host=args.submit_host)
        if args.submit
        else materialize_plan(plan)
    )
    print(
        json.dumps(
            {
                "status": "submitted" if args.submit else "planned",
                "workflow_root": str(plan.workflow_root),
                "palette_repo": str(plan.palette_repo),
                "palette_commit": plan.palette_commit,
                "matrix_fingerprint": plan.matrix_manifest["matrix_fingerprint"],
                "block_count": len(plan.matrix_manifest["repetitions"]),
                "candidate_count": len(plan.matrix_manifest["candidates"]),
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
