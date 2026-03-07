from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.utils import run_keypoints_with_registry_model as mod


def test_pick_best_candidate_enforces_unique_when_tied() -> None:
    candidates = [
        mod.Candidate(
            run_id="run_a",
            set_id="pose_a",
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
            set_id="pose_b",
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


def test_write_model_resolution_provenance_updates_keypoint_run_attrs(
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
    keypoint_parent = _Group()
    keypoint_run = _Group()
    keypoint_run.attrs["provenance"] = {"stage": "keypoints_detect"}
    keypoint_parent["keypoints_20260209_000000"] = keypoint_run
    root["keypoints_runs"] = keypoint_parent
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root)

    payload = {
        "mode": "registry",
        "task": "pose",
        "registry_path": "/nvme1/palette_registry.sqlite",
        "recording_id": "2026-01-28T19-22-28Z_arena_1",
        "resolved_at_utc": "2026-02-09T09:00:00+00:00",
        "selected": {
            "run_id": "pose_run_001",
            "set_id": "pose_set_001",
            "model_path": "/nvme1/models/pose/model.pt",
            "score": 0.88,
            "created_utc": "2026-02-09T08:59:00+00:00",
        },
        "candidates": [{"run_id": "pose_run_001", "score": 0.88}],
    }

    mod._write_model_resolution_provenance(  # noqa: SLF001
        zarr_path=tmp_path / "sample_analysis.zarr",
        run_name="keypoints_20260209_000000",
        payload=payload,
    )

    assert keypoint_run.attrs.get("model_resolution_mode") == "registry"
    assert keypoint_run.attrs.get("model_resolution_task") == "pose"
    assert keypoint_run.attrs.get("model_resolution_selected_run_id") == "pose_run_001"
    assert keypoint_run.attrs.get("model_resolution_selected_set_id") == "pose_set_001"
    provenance = keypoint_run.attrs.get("provenance")
    assert isinstance(provenance, dict)
    assert "model_resolution" in provenance


def test_main_runs_pose_resolution_and_writes_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        run_id="pose_run_123",
        set_id="pose_set_123",
        model_path="/tmp/pose_model.pt",
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

    def _fake_detect_keypoints_yolo(**kwargs: object) -> str:
        calls["detect_kwargs"] = kwargs
        return "keypoints_001"

    monkeypatch.setattr(mod, "detect_keypoints_yolo", _fake_detect_keypoints_yolo)

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
            "pose_set_123",
            "--include-non-success",
            "--cpu",
            "--roi-cache-policy",
            "always",
            "--roi-cache-dir",
            str(tmp_path / "roi-cache"),
        ]
    )

    assert rc == 0
    assert calls.get("registry_opened") is True
    assert calls.get("registry_closed") is True
    assert calls.get("resolver_task") == "pose"
    assert calls.get("resolver_set_id") == "pose_set_123"
    assert calls.get("resolver_include_non_success") is True

    detect_kwargs = calls.get("detect_kwargs")
    assert isinstance(detect_kwargs, dict)
    assert detect_kwargs.get("zarr_path") == str(output_path.resolve())
    assert detect_kwargs.get("model_path") == "/tmp/pose_model.pt"
    assert detect_kwargs.get("device") == "cpu"
    assert detect_kwargs.get("roi_cache_policy") == "always"
    assert detect_kwargs.get("roi_cache_dir") == tmp_path / "roi-cache"
    assert Path(str(detect_kwargs.get("registry"))) == registry_path.resolve()

    assert calls.get("write_zarr_path") == output_path.resolve()
    assert calls.get("write_run_name") == "keypoints_001"
    payload = calls.get("write_payload")
    assert isinstance(payload, dict)
    assert payload.get("task") == "pose"
    parameters = payload.get("parameters")
    assert isinstance(parameters, dict)
    assert parameters.get("roi_cache_policy") == "always"
    for key in ("contract", "command", "git", "environment", "platform", "parameters", "inputs", "artifacts"):
        assert key in payload
