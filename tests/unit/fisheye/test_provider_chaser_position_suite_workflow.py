from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.analysis.chaser_profiles import position_chaser_analysis_profile_path
from fisheye.cluster.provider_chaser_position_suite import (
    build_provider_chaser_position_suite_workflow,
)


def _plan(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/py").write_text("#!/bin/sh\n", encoding="utf-8")
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    run_root = tmp_path / "operations" / "run-v1"
    return build_provider_chaser_position_suite_workflow(
        workflow_id="position-suite-workflow-v1",
        repo=repo,
        run_root=run_root,
        analysis_zarr=archive,
        run_name="position-suite-v1",
        provider_run="provider-distance-v1",
        geometry_selection_run="geometry-selection-v1",
        expected_selection_record_sha256="a" * 64,
        expected_physical_authority_sha256="b" * 64,
        epoch_role_bindings=(("pre", 0), ("training", 1), ("post", 2)),
        analysis_profile_path=position_chaser_analysis_profile_path(),
        palette_commit="c" * 40,
        expected_recording_id="recording-1",
    )


def test_exact_workflow_orders_publication_then_nonmutating_readiness(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    payload = plan.to_json()
    jobs = plan.workflow.jobs

    assert payload["profile_id"] == "chaser_position_suite_v1"
    assert payload["selector_eligible"] is False
    assert payload["registry_update"] is False
    assert len(jobs) == 2
    assert jobs[0].metadata["stage"] == "provider_chaser_position_suite_candidate"
    assert jobs[1].metadata["stage"] == "provider_chaser_position_suite_readiness"
    assert jobs[1].dependency.upstream_job_keys == (jobs[0].job_key,)
    publication = list(jobs[0].command)
    assert publication[-6:] == [
        "--epoch-role",
        "pre=0",
        "--epoch-role",
        "training=1",
        "--epoch-role",
        "post=2",
    ]
    assert "--apply" in publication
    assert "--output-json" in publication
    assert "registry" not in " ".join(publication).lower()
    readiness = list(jobs[1].command)
    assert "fisheye.utils.provider_chaser_position_suite_readiness" in readiness
    assert "--apply" not in readiness
    readiness_path = str(plan.readiness_receipt_path)
    assert readiness_path in readiness
    assert readiness_path in " ".join(readiness)


def test_workflow_is_a_revealing_plan_and_creates_no_outputs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert not plan.run_root.exists()
    assert not plan.publication_result_path.exists()
    assert not plan.readiness_receipt_path.exists()
    assert plan.workflow.metadata["required_ci_before_promotion"] is True
    assert plan.workflow.metadata["production_selector_activation"] is False


def test_workflow_rejects_selector_aliases_and_duplicate_epoch_roles(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/py").write_text("#!/bin/sh\n", encoding="utf-8")
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    kwargs = {
        "workflow_id": "position-suite-workflow-v1",
        "repo": repo,
        "run_root": tmp_path / "operations",
        "analysis_zarr": archive,
        "provider_run": "provider-distance-v1",
        "geometry_selection_run": "geometry-selection-v1",
        "expected_selection_record_sha256": "a" * 64,
        "expected_physical_authority_sha256": "b" * 64,
        "analysis_profile_path": position_chaser_analysis_profile_path(),
        "palette_commit": "c" * 40,
    }
    with pytest.raises(ValueError, match="non-selector"):
        build_provider_chaser_position_suite_workflow(
            **kwargs,
            run_name="latest",
            epoch_role_bindings=(("pre", 0),),
        )
    with pytest.raises(ValueError, match="must each be unique"):
        build_provider_chaser_position_suite_workflow(
            **kwargs,
            run_name="position-suite-v1",
            epoch_role_bindings=(("pre", 0), ("pre", 1)),
        )


def test_workflow_rejects_invalid_scientific_parameters(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/py").write_text("#!/bin/sh\n", encoding="utf-8")
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    kwargs = {
        "workflow_id": "position-suite-workflow-v1",
        "repo": repo,
        "run_root": tmp_path / "operations",
        "analysis_zarr": archive,
        "run_name": "position-suite-v1",
        "provider_run": "provider-distance-v1",
        "geometry_selection_run": "geometry-selection-v1",
        "expected_selection_record_sha256": "a" * 64,
        "expected_physical_authority_sha256": "b" * 64,
        "epoch_role_bindings": (("pre", 0),),
        "analysis_profile_path": position_chaser_analysis_profile_path(),
        "palette_commit": "c" * 40,
    }
    with pytest.raises(ValueError, match="radial_bin_width_mm"):
        build_provider_chaser_position_suite_workflow(
            **kwargs, radial_bin_width_mm=float("nan")
        )
    with pytest.raises(ValueError, match="must exceed"):
        build_provider_chaser_position_suite_workflow(
            **kwargs, near_entry_radius_mm=6.0, near_exit_radius_mm=5.0
        )
