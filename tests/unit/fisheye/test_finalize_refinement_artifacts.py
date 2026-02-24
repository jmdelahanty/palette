from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.utils import finalize_refinement_artifacts as mod


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

    refined_parent = group.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_1"
    refined_run = refined_parent.create_group("refined_detect_1")
    refined_run.attrs["source_detect_run"] = "detect_1"
    refined_run.attrs["source_quality_run"] = "quality_1"
    refined_run.attrs["manual_review_latest"] = "manual"
    refined_run.attrs["detect_review_status"] = {
        "state": review_state,
        "method": "manual",
        "intended_use": review_intended_use,
        "timestamp_utc": "2026-02-23T00:00:00+00:00",
        "resolved_group": "manual",
    }
    refined_run.create_group("manual")
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
    quality_calls: list[tuple[str, str, str | None, int]] = []
    refinement_calls: list[tuple[str, str]] = []

    def _fake_render(*, zarr_path: str, source_detect_run: str, source_quality_run: str | None, visuals_dpi: int):
        quality_calls.append((zarr_path, source_detect_run, source_quality_run, visuals_dpi))
        return b"\x89PNG\r\n\x1a\nFAKE", {"quality_score": {"grade": "A"}}

    def _fake_render_refinement(*, zarr_path: str, refined_run: str):
        refinement_calls.append((zarr_path, refined_run))
        return b"\x89PNG\r\n\x1a\nREFINE", {"refined_run": refined_run}

    monkeypatch.setattr(mod, "_render_quality_png", _fake_render)
    monkeypatch.setattr(mod, "_render_refinement_pipeline_png", _fake_render_refinement)

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0
    assert len(quality_calls) == 1
    assert len(refinement_calls) == 1

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_detect_runs"]["refined_detect_1"]
    assert "visualizations" in run
    assert mod.DETECT_QUALITY_ARTIFACT in run["visualizations"]
    assert mod.REFINEMENT_PIPELINE_ARTIFACT in run["visualizations"]
    quality_artifact = run["visualizations"][mod.DETECT_QUALITY_ARTIFACT]
    assert quality_artifact.attrs.get("source_detect_run") == "detect_1"
    assert quality_artifact.attrs.get("source_quality_run") == "quality_1"
    assert quality_artifact.attrs.get("quality_grade") == "A"
    assert quality_artifact.attrs.get("artifact_signature")
    refinement_artifact = run["visualizations"][mod.REFINEMENT_PIPELINE_ARTIFACT]
    assert refinement_artifact.attrs.get("source_detect_run") == "detect_1"
    assert refinement_artifact.attrs.get("source_quality_run") == "quality_1"
    assert refinement_artifact.attrs.get("render_refined_run") == "refined_detect_1"
    assert refinement_artifact.attrs.get("artifact_signature")

    manifest = run.attrs.get("visualizations")
    assert isinstance(manifest, dict)
    assert manifest[mod.DETECT_QUALITY_ARTIFACT]["path"] == mod.ARTIFACT_SPECS[mod.DETECT_QUALITY_ARTIFACT]["path"]
    assert (
        manifest[mod.REFINEMENT_PIPELINE_ARTIFACT]["path"]
        == mod.ARTIFACT_SPECS[mod.REFINEMENT_PIPELINE_ARTIFACT]["path"]
    )


def test_apply_skips_when_signature_is_unchanged(monkeypatch, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path, "rec_skip", review_state="approved")

    def _fake_render(*, zarr_path: str, source_detect_run: str, source_quality_run: str | None, visuals_dpi: int):
        return b"\x89PNG\r\n\x1a\nFAKE", {"quality_score": {"grade": "A"}}

    def _fake_render_refinement(*, zarr_path: str, refined_run: str):
        return b"\x89PNG\r\n\x1a\nREFINE", {"refined_run": refined_run}

    monkeypatch.setattr(mod, "_render_quality_png", _fake_render)
    monkeypatch.setattr(mod, "_render_refinement_pipeline_png", _fake_render_refinement)
    assert mod.main([str(zarr_path), "--apply"]) == 0

    def _raise_if_called(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("render should not be called for unchanged signature")

    monkeypatch.setattr(mod, "_render_quality_png", _raise_if_called)
    monkeypatch.setattr(mod, "_render_refinement_pipeline_png", _raise_if_called)
    assert mod.main([str(zarr_path), "--apply"]) == 0


def test_apply_force_rerenders_when_signature_matches(monkeypatch, tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path, "rec_force", review_state="approved")
    quality_calls = {"count": 0}
    refinement_calls = {"count": 0}

    def _fake_render(*, zarr_path: str, source_detect_run: str, source_quality_run: str | None, visuals_dpi: int):
        quality_calls["count"] += 1
        return b"\x89PNG\r\n\x1a\nFAKE", {"quality_score": {"grade": "A"}}

    def _fake_render_refinement(*, zarr_path: str, refined_run: str):
        refinement_calls["count"] += 1
        return b"\x89PNG\r\n\x1a\nREFINE", {"refined_run": refined_run}

    monkeypatch.setattr(mod, "_render_quality_png", _fake_render)
    monkeypatch.setattr(mod, "_render_refinement_pipeline_png", _fake_render_refinement)

    assert mod.main([str(zarr_path), "--apply"]) == 0
    assert mod.main([str(zarr_path), "--apply", "--force"]) == 0
    assert quality_calls["count"] == 2
    assert refinement_calls["count"] == 2
