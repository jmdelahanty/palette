from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import arena_geometry_candidates
from fisheye.analysis_workflows.materializers import (
    arena_geometry_fit_review as fit_review,
)
from fisheye.diagnostics.probe_recording_dish_rim_fit import write_review_package
from fisheye.registry.registered_geometry_readiness import (
    project_registered_geometry_stages,
)
from tests.unit.fisheye.test_arena_geometry_candidates import (
    _palette_binding,
    _palette_fit_inputs,
)


def _review_package(path: Path) -> Path:
    path.mkdir(parents=True)
    _palette_fit_inputs(path)
    for index, name in enumerate(("early", "middle", "late")):
        image = np.full((32, 48, 3), 20 + index * 30, dtype=np.uint8)
        assert cv2.imwrite(str(path / f"{name}_palette_fit.png"), image)
    write_review_package(path, acquisition_revealed=False)
    return path


def test_fit_review_package_is_self_contained_selector_ineligible_zarr_evidence(
    tmp_path: Path,
) -> None:
    source_zarr = tmp_path / "recording_analysis.zarr"
    zarr.open_group(str(source_zarr), mode="w", zarr_format=3).require_group("analysis")
    package = _review_package(tmp_path / "review_package")
    plan = fit_review.build_arena_geometry_fit_review_plan(
        source_zarr,
        review_package_dir=package,
    )

    result = fit_review.publish_arena_geometry_fit_review(
        plan,
        scratch_root=tmp_path / "scratch",
    )

    assert result["published"] is True
    shutil.rmtree(package)
    evidence = fit_review.load_arena_geometry_fit_review_evidence(
        source_zarr,
        run_name=plan.run_name,
    )
    assert json.loads(evidence.fit_report_bytes)["windows"].keys() == {
        "early",
        "middle",
        "late",
    }
    assert evidence.montage_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert evidence.fit_report_ref.startswith(
        f"analysis/{fit_review.FIT_REVIEW_RUNS_PARENT}/{plan.run_name}/"
    )

    direct = zarr.open_group(
        str(source_zarr), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated = zarr.open_group(
        str(source_zarr), mode="r", zarr_format=3, use_consolidated=True
    )
    path = f"analysis/{fit_review.FIT_REVIEW_RUNS_PARENT}/{plan.run_name}"
    for root in (direct, consolidated):
        run = root[path]
        assert run.attrs["palette_run_completion_status"] == "complete"
        assert run.attrs["stage_selector_eligible"] is False
        assert run.attrs["review_status"] == "awaiting_explicit_human_review"
        assert set(run.attrs["visualizations"]) == {
            fit_review.MONTAGE_ARTIFACT,
            "source_panel_00_early_palette_fit_png",
            "source_panel_01_middle_palette_fit_png",
            "source_panel_02_late_palette_fit_png",
        }
        assert run.attrs["candidate_published"] is False
        assert run.attrs["candidate_selected"] is False
        assert run.attrs["detection_gate_applied"] is False


def test_reviewed_candidate_consumes_embedded_fit_review_after_staging_is_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_zarr = tmp_path / "recording_analysis.zarr"
    zarr.open_group(str(source_zarr), mode="w", zarr_format=3).require_group("analysis")
    package = _review_package(tmp_path / "review_package")
    fit_plan = fit_review.build_arena_geometry_fit_review_plan(
        source_zarr,
        review_package_dir=package,
    )
    fit_review.publish_arena_geometry_fit_review(
        fit_plan,
        scratch_root=tmp_path / "scratch",
    )
    shutil.rmtree(package)
    monkeypatch.setattr(
        arena_geometry_candidates,
        "_source_camera_candidate_binding",
        lambda *_args, **_kwargs: _palette_binding(),
    )

    candidate = arena_geometry_candidates.plan_reviewed_palette_geometry_candidate(
        source_zarr=source_zarr,
        fit_review_run=fit_plan.run_name,
        reviewer="delahantyj",
        reviewed_at_utc="2026-08-13T12:00:00Z",
    )

    source = candidate.candidate_record["palette_fit_source"]
    assert source["review_evidence_storage"] == "embedded_zarr_fit_review_run_v1"
    assert source["fit_review_run"] == fit_plan.run_name
    assert source["fit_review_record_sha256"] == fit_plan.review_record_sha256
    assert source["fit_report_path"].startswith(
        f"analysis/{fit_review.FIT_REVIEW_RUNS_PARENT}/{fit_plan.run_name}/"
    )
    assert (
        candidate.run_provenance["input_run_ids"]["arena_geometry_fit_review"]
        == fit_plan.run_name
    )


def test_fit_review_import_fails_closed_on_changed_panel(
    tmp_path: Path,
) -> None:
    source_zarr = tmp_path / "recording_analysis.zarr"
    zarr.open_group(str(source_zarr), mode="w", zarr_format=3)
    package = _review_package(tmp_path / "review_package")
    (package / "early_palette_fit.png").write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed after review-package creation"):
        fit_review.build_arena_geometry_fit_review_plan(
            source_zarr,
            review_package_dir=package,
        )


def test_registry_projection_exposes_embedded_fit_as_review_pending(
    tmp_path: Path,
) -> None:
    source_zarr = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(source_zarr), mode="w", zarr_format=3)
    root.require_group("analysis")
    package = _review_package(tmp_path / "review_package")
    plan = fit_review.build_arena_geometry_fit_review_plan(
        source_zarr,
        review_package_dir=package,
    )
    fit_review.publish_arena_geometry_fit_review(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    root = zarr.open_group(
        str(source_zarr), mode="r", zarr_format=3, use_consolidated=True
    )

    projections = project_registered_geometry_stages(
        root=root,
        analysis_group=root["analysis"],
        common_details={},
        raw_status="ok",
        calibration_status="ok",
        detect_status="ok",
        detect_quality_status="ok",
    )
    by_stage = {row.step_name: row for row in projections}
    offline = by_stage["arena_geometry_offline_fit"]
    comparison = by_stage["arena_geometry_comparison"]
    selection = by_stage["arena_geometry_selection"]
    assert offline.status == "ok"
    assert offline.run_name == plan.run_name
    assert offline.review_status == {
        "state": "evidence_complete_review_pending",
        "runs": [plan.run_name],
    }
    assert comparison.status == "review"
    assert comparison.review_status["fit_review_runs"] == [plan.run_name]
    assert selection.status == "review"
