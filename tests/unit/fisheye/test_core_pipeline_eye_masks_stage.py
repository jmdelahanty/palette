from __future__ import annotations

from pathlib import Path

from fisheye.core import pipeline as pipeline_mod
from fisheye.core.pipeline import Pipeline, PipelineConfig


def _make_pipeline(tmp_path: Path) -> Pipeline:
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    cfg = PipelineConfig(
        zarr_path=str(tmp_path / "archive.zarr"),
        config_path=str(config_path),
    )
    return Pipeline(cfg)


def test_run_eye_masks_delegates_to_shared_batch_orchestration(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    pipeline.zarr_root = object()
    pipeline.config.registry_path = str(tmp_path / "registry.sqlite")
    pipeline.pipeline_params["eye_masks"] = {
        "method": "yolo_eye_segmentation",
        "model_path": "/tmp/eye_model.pt",
        "batch_size": 64,
    }

    captured: dict[str, object] = {}
    reopened_root = object()

    monkeypatch.setattr(
        pipeline_mod.eye_mask_batch,
        "_validate_method_requirements",
        lambda config, method, *, refine_only: captured.setdefault(
            "validated",
            {"config": config, "method": method, "refine_only": refine_only},
        ),
    )
    def _fake_infer_recording_dir(zarr_path: Path) -> Path:
        captured["recording_dir_input"] = zarr_path
        return tmp_path / "recording"

    monkeypatch.setattr(pipeline_mod.eye_mask_batch, "_infer_recording_dir", _fake_infer_recording_dir)

    def _fake_run_plan(plan, **kwargs):  # noqa: ANN001
        captured["plan"] = plan
        captured["run_kwargs"] = kwargs
        return {
            "status": "ok",
            "method": "yolo",
            "eye_masks": {"run_name": "eye_masks_001"},
            "subject_masks": {"run_name": "subject_masks_001"},
            "registry_sync": {"synced": True},
        }

    monkeypatch.setattr(pipeline_mod.eye_mask_batch, "_run_plan", _fake_run_plan)
    monkeypatch.setattr(pipeline_mod.zarr, "open_group", lambda *_args, **_kwargs: reopened_root)

    pipeline._run_eye_masks()

    validated = captured.get("validated")
    assert isinstance(validated, dict)
    assert validated["config"] == pipeline.pipeline_params
    assert validated["method"] == "yolo"
    assert validated["refine_only"] is False

    plan = captured.get("plan")
    assert isinstance(plan, pipeline_mod.eye_mask_batch.EyeMaskPlan)
    assert plan.zarr_path == Path(pipeline.config.zarr_path).expanduser().resolve()
    assert plan.recording_dir == tmp_path / "recording"
    assert plan.h5_path is None
    assert plan.camera_id is None
    assert plan.status == "ok"

    run_kwargs = captured.get("run_kwargs")
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["config"] == pipeline.pipeline_params
    assert run_kwargs["method"] == "yolo"
    assert run_kwargs["scheduler"] == pipeline.config.scheduler
    assert run_kwargs["num_workers"] == pipeline.config.num_workers
    assert run_kwargs["quiet"] is True
    assert run_kwargs["refine"] is False
    assert run_kwargs["refine_only"] is False
    assert run_kwargs["registry_path_for_sync"] == Path(str(pipeline.config.registry_path)).expanduser().resolve()

    assert pipeline.stage_results["eye_masks"]["eye_masks"]["run_name"] == "eye_masks_001"
    assert pipeline.stage_results["eye_masks"]["subject_masks"]["run_name"] == "subject_masks_001"
    assert pipeline.zarr_root is reopened_root
