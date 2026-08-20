from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.chaser_proxy_candidate import (
    build_chaser_proxy_candidate_workflow,
)
from fisheye.cluster.lsf import LsfDependencyCondition


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/py").write_text("#!/bin/sh\n", encoding="utf-8")
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    profile = tmp_path / "profile.yaml"
    profile.write_text("profile_id: fixture\n", encoding="utf-8")
    return repo, archive, profile, tmp_path / "operations" / "candidate-1"


def test_candidate_workflow_is_exact_ordered_and_non_production(
    tmp_path: Path,
) -> None:
    repo, archive, profile, run_root = _fixture_paths(tmp_path)

    plan = build_chaser_proxy_candidate_workflow(
        workflow_id="candidate-1",
        repo=repo,
        run_root=run_root,
        analysis_zarr=archive,
        source_run_name="native-v1",
        proxy_run_name="proxy-v1",
        relative_frame_run_name="relative-v1",
        analysis_profile_path=profile,
        palette_commit="f" * 40,
        expected_recording_id="recording-1",
        expected_source_manifest_sha256="a" * 64,
    )

    jobs = list(plan.workflow.topological_jobs())
    assert [job.metadata["stage"] for job in jobs] == [
        "chaser_input_provenance_proxy_candidate",
        "chaser_relative_frame_candidate",
        "chaser_proxy_candidate_readiness_receipt",
    ]
    assert jobs[0].dependency is None
    assert jobs[1].dependency is not None
    assert jobs[1].dependency.upstream_job_keys == (jobs[0].job_key,)
    assert jobs[1].dependency.condition is LsfDependencyCondition.ALL_SUCCEEDED
    assert jobs[2].dependency is not None
    assert jobs[2].dependency.upstream_job_keys == (jobs[1].job_key,)
    assert jobs[2].dependency.condition is LsfDependencyCondition.ALL_SUCCEEDED

    rendered = [" ".join(job.command) for job in jobs]
    assert "fisheye.utils.materialize_chaser_input_provenance_proxy" in rendered[0]
    assert "--source-run-name native-v1" in rendered[0]
    assert "--expected-source-manifest-sha256 " + "a" * 64 in rendered[0]
    assert "fisheye.utils.materialize_chaser_proxy_relative_frame" in rendered[1]
    assert "--proxy-run-name proxy-v1" in rendered[1]
    assert "--palette-commit" not in rendered[1]
    assert "fisheye.analysis_workflows.chaser_proxy_candidate_receipt" in rendered[2]
    assert "--relative-frame-run-name relative-v1" in rendered[2]
    assert "--palette-commit " + "f" * 40 in rendered[2]

    metadata = plan.workflow.metadata
    assert metadata["selector_eligible"] is False
    assert metadata["production_selector_activation"] is False
    assert metadata["registry_update"] is False
    assert metadata["required_ci_before_promotion"] is True
    assert metadata["palette_commit"] == "f" * 40
    assert all("registry" not in job.metadata["stage"] for job in jobs)


def test_candidate_workflow_rejects_run_root_inside_archive(tmp_path: Path) -> None:
    repo, archive, profile, _ = _fixture_paths(tmp_path)

    with pytest.raises(ValueError, match="outside the analysis Zarr"):
        build_chaser_proxy_candidate_workflow(
            workflow_id="candidate-1",
            repo=repo,
            run_root=archive / "operations",
            analysis_zarr=archive,
            source_run_name="native-v1",
            proxy_run_name="proxy-v1",
            relative_frame_run_name="relative-v1",
            analysis_profile_path=profile,
            palette_commit="f" * 40,
        )
