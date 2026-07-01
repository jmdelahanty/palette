from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

from fisheye.core import pipeline as pipeline_mod
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


def test_code_defaults_disable_refined_eye_masks(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        zarr_path=str(tmp_path / "archive.zarr"),
        config_path=str(tmp_path / "missing_config.yaml"),
    )
    pipeline = Pipeline(cfg)

    assert pipeline.pipeline_params["refine_eye_masks"]["enabled"] is False


def test_packaged_default_yaml_disables_refined_eye_masks() -> None:
    config = yaml.safe_load(Path("configs/fisheye/default.yaml").read_text(encoding="utf-8"))

    assert config["refine_eye_masks"]["enabled"] is False


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


def test_run_refined_eye_masks_runs_when_explicitly_requested(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline._explicitly_requested_stages = {"refined_eye_masks"}
    pipeline.pipeline_params["refine_eye_masks"] = {"enabled": False}
    captured: dict[str, object] = {}

    def _fake_refine_eye_masks(**kwargs):
        captured.update(kwargs)
        return "refined_eye_masks_001"

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", _fake_refine_eye_masks)

    pipeline._run_refined_eye_masks()

    assert captured["zarr_path"] == pipeline.config.zarr_path
    assert captured["command"] == "pipeline:refined_eye_masks"


def test_stage_completion_probe_logs_failures(monkeypatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    zarr_path = tmp_path / "archive.zarr"
    zarr_path.mkdir()
    pipeline = _make_pipeline(tmp_path)
    pipeline.config.zarr_path = str(zarr_path)

    def _raise_open(*_args, **_kwargs):
        raise PermissionError("blocked store")

    monkeypatch.setattr(pipeline_mod.zarr, "open", _raise_open)

    with caplog.at_level(logging.WARNING, logger=pipeline_mod.__name__):
        assert pipeline._is_stage_complete("detect") is False

    assert str(zarr_path) in caplog.text
    assert "blocked store" in caplog.text


def test_legacy_orchestrator_notice_in_docstring_and_main_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "current cluster workflow" in (pipeline_mod.__doc__ or "")

    monkeypatch.setattr(sys, "argv", ["fisheye", "--help"])
    with pytest.raises(SystemExit) as exc:
        pipeline_mod.main()

    assert exc.value.code == 0
    assert pipeline_mod.LEGACY_ORCHESTRATOR_NOTICE in capsys.readouterr().out


def test_run_refined_eye_masks_passes_config_and_preserves_supported_scheduler(
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
        "success_min_eye_area_px": 42.0,
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
    assert captured["scheduler"] == "distributed"
    assert captured["num_workers"] == 7
    assert captured["area_filter_z"] == 3.5
    assert captured["area_filter_mode"] == "both"
    assert captured["success_min_eye_area_px"] == pytest.approx(42.0)
    assert captured["force_refine_traditional"] is True
    assert captured["allow_latest_keypoint_fallback"] is True
    assert captured["command"] == "pipeline:refined_eye_masks"
    assert captured["created_at_utc"] is not None


def test_run_refined_eye_masks_exports_merged_and_auto_aggregates(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    merged_out = tmp_path / "merged_eye_training.zarr"
    pipeline.pipeline_params["refine_eye_masks"] = {
        "enabled": True,
        "merged_out_zarr": str(merged_out),
    }

    captured: dict[str, object] = {}

    def _fake_refine_eye_masks(**_kwargs):
        return "refined_eye_masks_001"

    def _fake_export_merged_eye_mask_training_zarr(**kwargs):
        captured.update(kwargs)
        return {"zarr_path": str(kwargs["out_zarr"])}

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", _fake_refine_eye_masks)
    monkeypatch.setattr(
        "fisheye.utils.export_eye_mask_training_zarr.export_merged_eye_mask_training_zarr",
        _fake_export_merged_eye_mask_training_zarr,
    )

    pipeline._run_refined_eye_masks()

    assert captured["source_zarr"] == Path(pipeline.config.zarr_path)
    assert captured["out_zarr"] == merged_out
    assert captured["eye_stage"] == "refined_eye_masks_runs"
    assert captured["eye_run"] == "refined_eye_masks_001"
    assert captured["aggregate_training_data_card"] is True
    assert pipeline.stage_results["eye_mask_merged_export"]["zarr_path"] == str(merged_out)


def test_run_refined_eye_masks_respects_no_aggregate_opt_out(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    merged_out = tmp_path / "merged_eye_training.zarr"
    pipeline.pipeline_params["refine_eye_masks"] = {
        "enabled": True,
        "merged_out_zarr": str(merged_out),
        "no_aggregate_training_data_card": True,
    }

    captured: dict[str, object] = {}

    def _fake_refine_eye_masks(**_kwargs):
        return "refined_eye_masks_001"

    def _fake_export_merged_eye_mask_training_zarr(**kwargs):
        captured.update(kwargs)
        return {"zarr_path": str(kwargs["out_zarr"])}

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", _fake_refine_eye_masks)
    monkeypatch.setattr(
        "fisheye.utils.export_eye_mask_training_zarr.export_merged_eye_mask_training_zarr",
        _fake_export_merged_eye_mask_training_zarr,
    )

    pipeline._run_refined_eye_masks()

    assert captured["aggregate_training_data_card"] is False


def test_run_refined_eye_masks_rejects_conflicting_aggregate_flags(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_eye_masks"] = {
        "enabled": True,
        "merged_out_zarr": str(tmp_path / "merged_eye_training.zarr"),
        "aggregate_training_data_card": True,
        "no_aggregate_training_data_card": True,
    }

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", lambda **_kwargs: "refined_eye_masks_001")

    with pytest.raises(ValueError, match="cannot be combined"):
        pipeline._run_refined_eye_masks()


def test_run_refined_eye_masks_requires_merged_export_for_aggregation(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.pipeline_params["refine_eye_masks"] = {
        "enabled": True,
        "aggregate_training_data_card": True,
    }

    monkeypatch.setattr(refine_eye_masks_mod, "refine_eye_masks", lambda **_kwargs: "refined_eye_masks_001")

    with pytest.raises(ValueError, match="Provide --eye-mask-merged-out-zarr"):
        pipeline._run_refined_eye_masks()
