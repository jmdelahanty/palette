from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.core.pipeline import Pipeline, PipelineConfig
from fisheye.refinement import refine_subject_masks as refine_subject_masks_mod


def _make_pipeline(tmp_path: Path) -> Pipeline:
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = PipelineConfig(
        zarr_path=str(tmp_path / "archive.zarr"),
        config_path=str(config_path),
    )
    return Pipeline(cfg)


def test_stage_order_and_dependency_include_refined_subject_masks() -> None:
    assert "refined_subject_masks" in Pipeline.STAGE_ORDER
    assert Pipeline.STAGE_DEPENDENCIES["refined_subject_masks"] == []


def test_resolve_dependencies_keeps_refined_subject_masks_standalone(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    resolved = pipeline._resolve_dependencies(["refined_subject_masks"])
    assert resolved == ["refined_subject_masks"]


def test_run_stage_dispatches_refined_subject_masks(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    called: list[str] = []
    monkeypatch.setattr(pipeline, "_is_stage_complete", lambda stage: False)
    monkeypatch.setattr(pipeline, "_run_refined_subject_masks", lambda: called.append("refined_subject_masks"))

    pipeline._run_stage("refined_subject_masks")

    assert called == ["refined_subject_masks"]
    assert "refined_subject_masks" in pipeline.stage_timings


def test_run_refined_subject_masks_respects_enabled_flag(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_subject_masks"] = {"enabled": False}

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("refine_subject_masks should not run when stage is disabled")

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", _unexpected_call)

    pipeline._run_refined_subject_masks()


def test_run_refined_subject_masks_explicit_stage_overrides_enabled_flag(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline._explicitly_requested_stages = {"refined_subject_masks"}
    pipeline.pipeline_params["refine_subject_masks"] = {
        "enabled": False,
        "run_name": "refined_subject_masks_001",
    }

    captured: dict[str, object] = {}

    def _fake_refine_subject_masks(**kwargs):
        captured.update(kwargs)
        return {
            "status": "updated",
            "refined_run": "refined_subject_masks_001",
            "changed_roi_count": 0,
            "noop_roi_count": 0,
        }

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", _fake_refine_subject_masks)

    pipeline._run_refined_subject_masks()

    assert captured["refined_run"] == "refined_subject_masks_001"


def test_run_refined_subject_masks_passes_config_and_preserves_supported_scheduler(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_subject_masks"] = {
        "enabled": True,
        "subject_run": "subject_masks_001",
        "refined_run": "refined_subject_masks_001",
        "components": ["subject_body", "swim_bladder"],
        "roi_indices": [3, 4, 5],
        "chunk_size": 128,
        "scheduler": "distributed",
        "num_workers": 7,
    }

    captured: dict[str, object] = {}

    def _fake_refine_subject_masks(**kwargs):
        captured.update(kwargs)
        return {
            "status": "updated",
            "refined_run": "refined_subject_masks_001",
            "changed_roi_count": 2,
            "noop_roi_count": 1,
        }

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", _fake_refine_subject_masks)

    pipeline._run_refined_subject_masks()

    assert captured["zarr_path"] == pipeline.config.zarr_path
    assert captured["subject_run"] == "subject_masks_001"
    assert captured["refined_run"] == "refined_subject_masks_001"
    assert captured["components"] == ["subject_body", "swim_bladder"]
    assert captured["roi_indices"] == [3, 4, 5]
    assert captured["chunk_size"] == 128
    assert captured["scheduler"] == "distributed"
    assert captured["num_workers"] == 7
    assert captured["console"] == pipeline.console
    assert pipeline.stage_results["refined_subject_masks"]["refined_run"] == "refined_subject_masks_001"


def test_run_refined_subject_masks_accepts_config_aliases(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_subject_masks"] = {
        "enabled": True,
        "source_run": "subject_masks_alias_001",
        "run_name": "refined_subject_masks_alias_001",
    }

    captured: dict[str, object] = {}

    def _fake_refine_subject_masks(**kwargs):
        captured.update(kwargs)
        return {
            "status": "updated",
            "refined_run": "refined_subject_masks_alias_001",
            "changed_roi_count": 0,
            "noop_roi_count": 0,
        }

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", _fake_refine_subject_masks)

    pipeline._run_refined_subject_masks()

    assert captured["subject_run"] == "subject_masks_alias_001"
    assert captured["refined_run"] == "refined_subject_masks_alias_001"


def test_run_refined_subject_masks_normalizes_invalid_scheduler(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_subject_masks"] = {
        "enabled": True,
        "scheduler": "bogus",
    }

    captured: dict[str, object] = {}

    def _fake_refine_subject_masks(**kwargs):
        captured.update(kwargs)
        return {
            "status": "updated",
            "refined_run": "refined_subject_masks_001",
            "changed_roi_count": 0,
            "noop_roi_count": 0,
        }

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", _fake_refine_subject_masks)

    pipeline._run_refined_subject_masks()

    assert captured["scheduler"] == "processes"


def test_run_refined_subject_masks_rejects_conflicting_aliases(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_subject_masks"] = {
        "enabled": True,
        "subject_run": "subject_masks_primary_001",
        "source_run": "subject_masks_alias_001",
    }

    monkeypatch.setattr(refine_subject_masks_mod, "refine_subject_masks", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="disagree"):
        pipeline._run_refined_subject_masks()
