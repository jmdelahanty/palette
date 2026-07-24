from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fisheye.cluster.canonical_detection_storage_benchmark import (
    apply_plan,
    build_plan,
    materialize_plan,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_matrix import BenchmarkScale


def _clean_palette_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "palette-worktree"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Palette Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "scripts/py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return repo


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmarks" / "canonical_detection_storage" / "fixtures" / "f1"
    root.mkdir(parents=True)
    (root / "fixture_manifest.json").write_text(
        json.dumps(
            {
                "fixture_id": "f1",
                "status": "published_immutable",
                "benchmark_only": True,
                "canonical": False,
            }
        ),
        encoding="utf-8",
    )
    return root


def _scale() -> BenchmarkScale:
    return BenchmarkScale.from_mapping(
        "frames_200k",
        {
            "n_frames": 200_000,
            "n_instances": 199_734,
            "source_width": 4512,
            "source_height": 4512,
        },
    )


def _plan(
    tmp_path: Path,
    *,
    repetitions: int = 1,
    repetition_start: int = 0,
):
    benchmark_root = tmp_path / "benchmarks"
    workflow_root = (
        benchmark_root
        / "canonical_detection_storage"
        / "workflows"
        / "smoke_01"
    )
    return build_plan(
        workflow_id="smoke_01",
        workflow_root=workflow_root,
        benchmark_root=benchmark_root,
        fixture_root=_fixture_root(tmp_path),
        palette_repo=_clean_palette_repo(tmp_path),
        recording_identity="sleepyfish_cam2010095",
        scales=(_scale(),),
        repetitions=repetitions,
        repetition_start=repetition_start,
        seed=20_260_724,
        queue="short",
        ncores=2,
        mem_gb_per_slot=8,
        walltime="1:00",
        max_active_blocks=1,
    )


def test_cluster_plan_is_one_cpu_array_block_plus_success_finalizer(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    assert plan.matrix_manifest["summary"] == {
        "requested_candidate_labels": 20,
        "unique_physical_candidates": 8,
        "removed_duplicate_labels": 12,
        "planned_trials": 8,
        "destination_collisions": 0,
        "payload_io_performed": False,
    }
    blocks, finalizer = plan.workflow.jobs
    assert blocks.resources.gpus == 0
    assert blocks.resources.ncores == 2
    assert blocks.resources.mem_gb == 8
    assert blocks.execution_group is not None
    assert len(blocks.execution_group.tasks) == 1
    task = blocks.execution_group.tasks[0]
    assert task.metadata["candidate_count"] == 8
    assert len(set(task.metadata["balanced_order"])) == 8
    assert task.command[:1] == ("/usr/bin/env",)
    assert set(
        task.command[1 : 1 + len(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)]
    ) == {
        f"{key}={value}"
        for key, value in STORAGE_BENCHMARK_THREAD_ENVIRONMENT.items()
    }
    assert task.metadata["native_thread_environment"] == (
        STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    )
    assert "__PALETTE_LSF_JOBID__" in str(task.status_path)
    assert "__PALETTE_LSF_JOBINDEX__" in str(task.status_path)
    assert finalizer.dependency is not None
    assert finalizer.dependency.upstream_job_keys == ("benchmark_blocks",)
    assert finalizer.metadata["profile_promotion"] is False


def test_cluster_plan_continues_balanced_repetitions_without_index_zero(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, repetitions=4, repetition_start=1)

    blocks, _finalizer = plan.workflow.jobs
    assert blocks.execution_group is not None
    assert [
        task.metadata["repetition_index"]
        for task in blocks.execution_group.tasks
    ] == [1, 2, 3, 4]
    assert all(
        "repetition_000" not in output
        for task in blocks.execution_group.tasks
        for output in task.expected_outputs
    )


def test_materialize_and_apply_submit_exact_dependency_with_fake_runner(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    manifest = materialize_plan(plan)

    assert json.loads(plan.plan_path.read_text(encoding="utf-8")) == manifest
    assert json.loads(plan.matrix_path.read_text(encoding="utf-8")) == (
        plan.matrix_manifest
    )
    assert json.loads(plan.lsf_plan_path.read_text(encoding="utf-8")) == (
        plan.workflow.to_json()
    )

    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        job_id = 100 + len(calls)
        return SimpleNamespace(
            returncode=0,
            stdout=f"Job <{job_id}> is submitted to queue <short>.\n",
            stderr="",
        )

    result = apply_plan(plan, runner=runner)

    assert result["status"] == "submitted"
    assert result["job_ids_by_key"] == {
        "benchmark_blocks": "101",
        "finalize": "102",
    }
    assert len(calls) == 2
    dependency_index = calls[1].index("-w")
    assert calls[1][dependency_index + 1] == "done(101)"
    assert json.loads(plan.submission_path.read_text(encoding="utf-8"))[
        "status"
    ] == "submitted"
