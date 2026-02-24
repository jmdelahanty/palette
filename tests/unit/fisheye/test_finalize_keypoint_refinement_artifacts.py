from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.utils import finalize_keypoint_refinement_artifacts as mod


def _make_archive(
    root: Path,
    name: str,
    *,
    review_state: str = "approved",
    review_intended_use: str = "training",
) -> Path:
    zarr_path = root / f"{name}_analysis.zarr"
    group = zarr.open_group(str(zarr_path), mode="w")
    group.attrs["zarr_purpose"] = "analysis"

    refined_parent = group.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_keypoints_1"
    refined_run = refined_parent.create_group("refined_keypoints_1")
    refined_run.attrs["source_keypoints_run"] = "keypoints_1"
    refined_run.attrs["source_crop_run"] = "crop_1"
    refined_run.attrs["source_detect_run"] = "detect_1"
    refined_run.attrs["keypoint_review_status"] = {
        "state": review_state,
        "method": "manual",
        "intended_use": review_intended_use,
        "timestamp_utc": "2026-02-24T00:00:00+00:00",
        "reviewer": "tester",
    }
    return zarr_path


def test_dry_run_reports_only_approved_rows(tmp_path: Path) -> None:
    _make_archive(tmp_path, "rec_approved", review_state="approved")
    _make_archive(tmp_path, "rec_needs_review", review_state="needs_review")
    report_path = tmp_path / "report.json"

    rc = mod.main([str(tmp_path), "--recursive", "--json-report", str(report_path)])
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["summary"]["scanned"] == 2
    assert report["summary"]["eligible"] == 1
    assert report["summary"]["would_finalize"] == 1

    reasons = {row["reason"] for row in report["rows"]}
    assert "review_state_not_approved" in reasons


def test_apply_writes_finalized_visualizations(monkeypatch, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path, "rec_apply", review_state="approved")
    quality_calls = {"count": 0}
    refinement_calls = {"count": 0}

    def _fake_quality(zarr_path: str, refined_run: str | None = None, *, dpi: int = 150, show: bool = False):
        quality_calls["count"] += 1
        return b"\x89PNG\r\n\x1a\nKPQUALITY", {"summary_statistics": {"usable_keypoints": 1}}

    def _fake_refinement(zarr_path: str, refined_run: str | None = None, *, dpi: int = 150, show: bool = False):
        refinement_calls["count"] += 1
        return b"\x89PNG\r\n\x1a\nKPPIPE", {"refined_run": refined_run}

    monkeypatch.setattr(mod, "render_keypoint_quality_png", _fake_quality)
    monkeypatch.setattr(mod, "render_keypoint_refinement_pipeline_png", _fake_refinement)

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0
    assert quality_calls["count"] == 1
    assert refinement_calls["count"] == 1

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_keypoints_runs"]["refined_keypoints_1"]
    assert "visualizations" in run
    assert mod.QUALITY_ARTIFACT_NAME in run["visualizations"]
    assert mod.REFINEMENT_PIPELINE_ARTIFACT_NAME in run["visualizations"]

    quality_artifact = run["visualizations"][mod.QUALITY_ARTIFACT_NAME]
    assert quality_artifact.attrs.get("source_keypoints_run") == "keypoints_1"
    assert quality_artifact.attrs.get("artifact_signature")
    assert quality_artifact.attrs.get("usable_keypoints") == 1

    refinement_artifact = run["visualizations"][mod.REFINEMENT_PIPELINE_ARTIFACT_NAME]
    assert refinement_artifact.attrs.get("source_crop_run") == "crop_1"
    assert refinement_artifact.attrs.get("render_refined_run") == "refined_keypoints_1"
    assert refinement_artifact.attrs.get("artifact_signature")

    manifest = run.attrs.get("visualizations")
    assert isinstance(manifest, dict)
    assert manifest[mod.QUALITY_ARTIFACT_NAME]["path"] == mod.ARTIFACT_SPECS[mod.QUALITY_ARTIFACT_NAME]["path"]
    assert (
        manifest[mod.REFINEMENT_PIPELINE_ARTIFACT_NAME]["path"]
        == mod.ARTIFACT_SPECS[mod.REFINEMENT_PIPELINE_ARTIFACT_NAME]["path"]
    )


def test_apply_skips_when_signature_is_unchanged(monkeypatch, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path, "rec_skip", review_state="approved")

    def _fake_quality(zarr_path: str, refined_run: str | None = None, *, dpi: int = 150, show: bool = False):
        return b"\x89PNG\r\n\x1a\nKPQUALITY", {"summary_statistics": {"usable_keypoints": 1}}

    def _fake_refinement(zarr_path: str, refined_run: str | None = None, *, dpi: int = 150, show: bool = False):
        return b"\x89PNG\r\n\x1a\nKPPIPE", {"refined_run": refined_run}

    monkeypatch.setattr(mod, "render_keypoint_quality_png", _fake_quality)
    monkeypatch.setattr(mod, "render_keypoint_refinement_pipeline_png", _fake_refinement)
    assert mod.main([str(zarr_path), "--apply"]) == 0

    def _raise_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("render should not be called for unchanged signature")

    monkeypatch.setattr(mod, "render_keypoint_quality_png", _raise_if_called)
    monkeypatch.setattr(mod, "render_keypoint_refinement_pipeline_png", _raise_if_called)
    assert mod.main([str(zarr_path), "--apply"]) == 0
