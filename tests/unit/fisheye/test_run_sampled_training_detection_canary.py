from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils import predict_training_detections as prediction
from fisheye.utils import run_sampled_training_detection_canary as canary


def _spec(model: Path) -> prediction.ModelInputSpec:
    return prediction.ModelInputSpec(
        artifact_kind="training",
        run_id="registered-detect-v1",
        set_id="detect-set-v1",
        task_type="detect",
        artifact_path=str(model),
        input_shape=[1, 3, 64, 64],
        input_layout="NCHW",
        input_channels=3,
        img_h=64,
        img_w=64,
        max_batch=1,
        dynamic_shapes=False,
        input_dtype="float32",
        input_color_space="rgb",
        input_shape_source="test",
        input_shape_status="explicit",
    )


def _archive(path: Path, *, model_run_id: str = "registered-detect-v1") -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    artifacts = root.create_group("detection_artifact_runs")
    artifact = artifacts.create_group("artifact-v1")
    artifact.attrs["model_registry_run_id"] = model_run_id


def test_reuse_existing_artifact_skips_inference_and_publication(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "training.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    model = tmp_path / "model.pt"
    model.touch()
    _archive(archive)

    monkeypatch.setattr(
        canary, "resolve_model_input_spec", lambda *_a, **_k: _spec(model)
    )
    monkeypatch.setattr(
        canary,
        "run_training_zarr_prediction",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("inference must not run during artifact recovery")
        ),
    )
    monkeypatch.setattr(
        canary,
        "publish_detection_artifact_run",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("artifact publication must not repeat during recovery")
        ),
    )
    monkeypatch.setattr(
        canary,
        "build_and_publish_sampled_training_detection",
        lambda **kwargs: {
            "status": "complete",
            "run_id": kwargs["run_id"],
            "cardinality_statistics": {"detection_rows": 3},
        },
    )

    result = canary.run_sampled_training_detection_canary(
        archive=archive,
        scratch_root=scratch,
        registry_path=registry,
        model_run_id="registered-detect-v1",
        model_path=None,
        model_set_id=None,
        artifact_kind="training",
        artifact_run_id="artifact-v1",
        detect_run_id="detect-v1",
        batch_size=8,
        conf=0.4,
        iou=0.45,
        max_det=20,
        cpu=False,
        copy_backend="python",
        reuse_existing_artifact=True,
    )

    assert result["source_copy"]["status"] == "skipped_reusing_existing_artifact"
    assert result["inference"]["status"] == "skipped_reusing_existing_artifact"
    assert result["artifact_publication"]["status"] == (
        "reused_existing_immutable_artifact"
    )
    assert result["detection_publication"]["run_id"] == "detect-v1"
    assert result["parameters"]["reuse_existing_artifact"] is True


def test_reuse_existing_artifact_rejects_missing_or_wrong_model(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "training.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    model = tmp_path / "model.pt"
    model.touch()
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    monkeypatch.setattr(
        canary, "resolve_model_input_spec", lambda *_a, **_k: _spec(model)
    )

    kwargs = {
        "archive": archive,
        "scratch_root": scratch,
        "registry_path": registry,
        "model_run_id": "registered-detect-v1",
        "model_path": None,
        "model_set_id": None,
        "artifact_kind": "training",
        "artifact_run_id": "artifact-v1",
        "detect_run_id": "detect-v1",
        "batch_size": 8,
        "conf": 0.4,
        "iou": 0.45,
        "max_det": 20,
        "cpu": False,
        "copy_backend": "python",
        "reuse_existing_artifact": True,
    }
    try:
        canary.run_sampled_training_detection_canary(**kwargs)
    except FileNotFoundError as exc:
        assert "requires the exact completed artifact" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Missing artifact unexpectedly entered recovery.")

    _archive(archive, model_run_id="different-model")
    try:
        canary.run_sampled_training_detection_canary(**kwargs)
    except ValueError as exc:
        assert "model identity differs" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Wrong-model artifact unexpectedly entered recovery.")
