from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fisheye.utils import run_eye_masks_with_registry_model as mod


def test_pick_best_candidate_enforces_unique_when_tied() -> None:
    candidates = [
        mod.Candidate(
            run_id="run_a",
            set_id="eye_masks_set_a",
            model_path="/tmp/a.pt",
            created_utc="2026-02-09T00:00:00+00:00",
            status="success",
            dataset_count=10,
            weighted_score=0.9,
            feature_match_counts={"rig_id": 10},
            feature_weights_used=1.0,
        ),
        mod.Candidate(
            run_id="run_b",
            set_id="eye_masks_set_b",
            model_path="/tmp/b.pt",
            created_utc="2026-02-09T00:00:01+00:00",
            status="success",
            dataset_count=10,
            weighted_score=0.9,
            feature_match_counts={"rig_id": 10},
            feature_weights_used=1.0,
        ),
    ]
    with pytest.raises(SystemExit, match="Top candidate score tied"):
        mod._pick_best_candidate(candidates, require_unique=True)  # noqa: SLF001


def test_write_model_resolution_provenance_updates_eye_mask_run_attrs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Attrs(dict):
        def put(self, mapping: dict[str, object]) -> None:
            self.clear()
            self.update(mapping)

    class _Group(dict):
        def __init__(self) -> None:
            super().__init__()
            self.attrs = _Attrs()

        def get(self, key: str, default: object = None) -> object:  # noqa: A003
            return super().get(key, default)

    root = _Group()
    eye_parent = _Group()
    eye_run = _Group()
    eye_run.attrs["provenance"] = {"stage": "eye_masks"}
    eye_parent["eye_masks_20260209_000000"] = eye_run
    root["eye_masks_runs"] = eye_parent
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root)

    payload = {
        "mode": "registry",
        "task": "eye_masks",
        "method": "yolo",
        "registry_path": "/nvme1/palette_registry.sqlite",
        "recording_id": "2026-01-28T19-22-28Z_arena_1",
        "resolved_at_utc": "2026-02-09T09:00:00+00:00",
        "selected": {
            "run_id": "eye_run_001",
            "set_id": "eye_set_001",
            "model_path": "/nvme1/models/eye_masks/model.pt",
            "score": 0.88,
            "created_utc": "2026-02-09T08:59:00+00:00",
        },
        "candidates": [{"run_id": "eye_run_001", "score": 0.88}],
    }

    mod._write_model_resolution_provenance(  # noqa: SLF001
        zarr_path=tmp_path / "sample_analysis.zarr",
        run_name="eye_masks_20260209_000000",
        payload=payload,
    )

    assert eye_run.attrs.get("model_resolution_mode") == "registry"
    assert eye_run.attrs.get("model_resolution_task") == "eye_masks"
    assert eye_run.attrs.get("model_resolution_method") == "yolo"
    assert eye_run.attrs.get("model_resolution_selected_run_id") == "eye_run_001"
    assert eye_run.attrs.get("model_resolution_selected_set_id") == "eye_set_001"
    provenance = eye_run.attrs.get("provenance")
    assert isinstance(provenance, dict)
    assert "model_resolution" in provenance


def test_main_runs_eye_mask_resolution_and_writes_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "out_analysis.zarr"
    registry_path = tmp_path / "registry.sqlite"

    calls: dict[str, object] = {}

    class _FakeRegistry:
        def __init__(self, _path: Path) -> None:
            calls["registry_opened"] = True

        def close(self) -> None:
            calls["registry_closed"] = True

    target = mod.TargetProfile(
        recording_id="rec_001",
        recording_type="behavior",
        recording_subtype="screen",
        behavior_mode="default",
        rig_id="rig_1",
        arena_id="arena_1",
        camera_id="cam_1",
        canvas_name="canvas_1",
        protocol_name="protocol_a",
        dish_design="cedar_shadow",
        cross_id=None,
        genotype=None,
        dpf_at_acquisition=None,
    )
    best = mod.Candidate(
        run_id="eye_run_123",
        set_id="eye_masks_set_123",
        model_path="/tmp/eye_model.pt",
        created_utc="2026-02-09T00:00:00+00:00",
        status="success",
        dataset_count=4,
        weighted_score=0.99,
        feature_match_counts={"rig_id": 4},
        feature_weights_used=1.0,
    )

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)
    monkeypatch.setattr(mod, "_resolve_recording_id", lambda *_args, **_kwargs: "rec_001")
    monkeypatch.setattr(mod, "_load_target_profile", lambda *_args, **_kwargs: target)

    def _fake_load_candidates(
        _registry,
        *,
        target: mod.TargetProfile,
        task: str,
        set_id_filter: str | None,
        include_non_success: bool,
    ) -> list[mod.Candidate]:
        calls["resolver_target_recording_id"] = target.recording_id
        calls["resolver_task"] = task
        calls["resolver_set_id"] = set_id_filter
        calls["resolver_include_non_success"] = include_non_success
        return [best]

    monkeypatch.setattr(mod, "_load_candidates", _fake_load_candidates)

    def _fake_segment_eye_masks_yolo(**kwargs: object) -> str:
        calls["segment_kwargs"] = kwargs
        return "eye_masks_001"

    monkeypatch.setattr(mod, "_segment_eye_masks_yolo", _fake_segment_eye_masks_yolo)

    def _fake_write_model_resolution_provenance(*, zarr_path: Path, run_name: str, payload: dict[str, object]) -> None:
        calls["write_zarr_path"] = zarr_path
        calls["write_run_name"] = run_name
        calls["write_payload"] = payload

    monkeypatch.setattr(mod, "_write_model_resolution_provenance", _fake_write_model_resolution_provenance)

    rc = mod.main(
        [
            "--recording-dir",
            str(recording_dir),
            "--output",
            str(output_path),
            "--registry",
            str(registry_path),
            "--set-id",
            "eye_masks_set_123",
            "--method",
            "yolo",
            "--include-non-success",
        ]
    )

    assert rc == 0
    assert calls.get("registry_opened") is True
    assert calls.get("registry_closed") is True
    assert calls.get("resolver_task") == "eye_masks"
    assert calls.get("resolver_set_id") == "eye_masks_set_123"
    assert calls.get("resolver_include_non_success") is True

    segment_kwargs = calls.get("segment_kwargs")
    assert isinstance(segment_kwargs, dict)
    assert segment_kwargs.get("zarr_path") == str(output_path.resolve())
    assert segment_kwargs.get("model_path") == "/tmp/eye_model.pt"

    assert calls.get("write_zarr_path") == output_path.resolve()
    assert calls.get("write_run_name") == "eye_masks_001"
    payload = calls.get("write_payload")
    assert isinstance(payload, dict)
    assert payload.get("task") == "eye_masks"
    assert payload.get("method") == "yolo"
    for key in ("contract", "command", "git", "environment", "platform", "parameters", "inputs", "artifacts"):
        assert key in payload


def test_run_yolo_forwards_registry_and_status_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_segment_eye_masks_yolo(**kwargs: object) -> str:
        captured.update(kwargs)
        return "eye_masks_abc"

    monkeypatch.setattr(mod, "_segment_eye_masks_yolo", _fake_segment_eye_masks_yolo)

    args = argparse.Namespace(
        run_name="eye_masks_custom",
        crop_run="crop_001",
        keypoints_run="kp_001",
        batch_size=64,
        device="cpu",
        imgsz=512,
        conf=0.1,
        iou=0.5,
        max_det=2,
        mask_threshold=0.05,
        adaptive_scale=0.6,
        adaptive_cap=0.6,
        no_retina_masks=False,
        proto_upsample_factor=2,
        legacy_masks=False,
        verbose=False,
    )
    registry = object()
    details = {"selected_run_id": "eye_train_01"}

    run_name = mod._run_yolo(  # noqa: SLF001
        tmp_path / "out.zarr",
        "/tmp/eye_model.pt",
        args,
        registry=registry,  # type: ignore[arg-type]
        status_details=details,
    )

    assert run_name == "eye_masks_abc"
    assert captured.get("registry") is registry
    assert captured.get("status_details") == details


def test_run_unet_forwards_registry_and_status_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_infer_unet_eye_masks(
        argv: list[str],
        *,
        registry: object | None = None,
        status_details: dict[str, object] | None = None,
    ) -> None:
        captured["argv"] = list(argv)
        captured["registry"] = registry
        captured["status_details"] = dict(status_details or {})

    monkeypatch.setattr(mod, "_infer_unet_eye_masks", _fake_infer_unet_eye_masks)
    monkeypatch.setattr(mod, "_latest_eye_masks_run", lambda _path: "eye_masks_unet_001")

    args = argparse.Namespace(
        run_name="eye_masks_unet_custom",
        crop_run="crop_001",
        keypoints_run="kp_001",
        batch_size=32,
        device="cpu",
        label_mode="union",
        write_binary_masks=True,
        no_use_crop=False,
    )
    registry = object()
    details = {"selected_set_id": "eye_masks_set_123"}

    run_name = mod._run_unet(  # noqa: SLF001
        tmp_path / "out.zarr",
        "/tmp/eye_unet.pt",
        args,
        registry=registry,  # type: ignore[arg-type]
        status_details=details,
    )

    assert run_name == "eye_masks_unet_001"
    assert captured.get("registry") is registry
    assert captured.get("status_details") == details
    argv = captured.get("argv")
    assert isinstance(argv, list)
    assert "--write-binary-masks" in argv
