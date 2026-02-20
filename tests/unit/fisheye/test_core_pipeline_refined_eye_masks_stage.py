from __future__ import annotations

from pathlib import Path

from fisheye.core.pipeline import Pipeline, PipelineConfig
from fisheye.refinement import refine_eye_masks as refine_eye_masks_mod


def _make_pipeline(tmp_path: Path) -> Pipeline:
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = PipelineConfig(
        zarr_path=str(tmp_path / "archive.zarr"),
        config_path=str(config_path),
    )
    return Pipeline(cfg)


def test_stage_order_and_dependency_include_refined_eye_masks() -> None:
    assert "refined_eye_masks" in Pipeline.STAGE_ORDER
    assert Pipeline.STAGE_DEPENDENCIES["refined_eye_masks"] == ["eye_masks"]
    assert Pipeline.STAGE_ORDER.index("eye_masks") < Pipeline.STAGE_ORDER.index("refined_eye_masks")


def test_resolve_dependencies_adds_eye_masks_for_refined_eye_masks(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    resolved = pipeline._resolve_dependencies(["refined_eye_masks"])
    assert "eye_masks" in resolved
    assert "refined_eye_masks" in resolved
    assert resolved.index("eye_masks") < resolved.index("refined_eye_masks")


def test_run_stage_dispatches_refined_eye_masks(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    called: list[str] = []
    monkeypatch.setattr(pipeline, "_is_stage_complete", lambda stage: False)
    monkeypatch.setattr(pipeline, "_run_refined_eye_masks", lambda: called.append("refined_eye_masks"))

    pipeline._run_stage("refined_eye_masks")

    assert called == ["refined_eye_masks"]
    assert "refined_eye_masks" in pipeline.stage_timings


def test_run_refined_eye_masks_respects_enabled_flag(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_eye_masks"] = {"enabled": False}

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("refine_eye_masks should not run when stage is disabled")

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", _unexpected_call)

    pipeline._run_refined_eye_masks()


def test_run_refined_eye_masks_passes_config_and_normalizes_scheduler(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_eye_masks"] = {
        "enabled": True,
        "source_run": "eye_001",
        "run_name": "refined_001",
        "keypoint_run": "kp_001",
        "chunk_size": 256,
        "scheduler": "distributed",
        "num_workers": 7,
        "area_filter_z": 3.5,
        "area_filter_mode": "both",
        "force_refine_traditional": True,
        "allow_latest_keypoint_fallback": True,
    }

    captured: dict[str, object] = {}

    def _fake_refine_eye_masks(**kwargs):
        captured.update(kwargs)
        return "refined_eye_masks_001"

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", _fake_refine_eye_masks)

    pipeline._run_refined_eye_masks()

    assert captured["zarr_path"] == pipeline.config.zarr_path
    assert captured["source_run"] == "eye_001"
    assert captured["run_name"] == "refined_001"
    assert captured["keypoint_run"] == "kp_001"
    assert captured["chunk_size"] == 256
    assert captured["scheduler"] == "processes"
    assert captured["num_workers"] == 7
    assert captured["area_filter_z"] == 3.5
    assert captured["area_filter_mode"] == "both"
    assert captured["force_refine_traditional"] is True
    assert captured["allow_latest_keypoint_fallback"] is True
    assert captured["command"] == "pipeline:refined_eye_masks"
    assert captured["created_at_utc"] is not None
