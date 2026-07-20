from __future__ import annotations

import json
import os
from inspect import signature
from pathlib import Path

import pytest

from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.utils import run_keypoints_with_registry_model as mod


def test_registry_model_runner_defaults_to_keypoint_sharding() -> None:
    params = signature(mod.run_keypoints_with_registry_model).parameters

    assert (
        params["keypoint_roi_shard_rows"].default
        == mod.DEFAULT_KEYPOINT_ROI_SHARD_ROWS
    )
    assert (
        params["keypoint_frame_shard_rows"].default
        == mod.DEFAULT_KEYPOINT_FRAME_SHARD_ROWS
    )


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
        mod.pick_best_keypoint_candidate(candidates, require_unique=True)


def test_default_roi_cache_staging_dir_falls_back_when_user_scratch_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "palette_keypoint_user_without_scratch")
    monkeypatch.setenv("LSB_JOBID", "67890")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    assert mod._default_roi_cache_staging_dir() == (  # noqa: SLF001
        tmp_path / f"palette_roi_cache_stage_{os.getpid()}"
    )


def test_default_roi_cache_staging_dir_is_unique_per_lsf_array_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "palette_array_fixture_user")
    monkeypatch.setenv("LSB_JOBID", "123")
    monkeypatch.setenv("LSB_JOBINDEX", "7")
    monkeypatch.setattr(mod.Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(mod.os, "access", lambda *_args: True)

    assert mod._default_roi_cache_staging_dir() == (  # noqa: SLF001
        Path("/scratch/palette_array_fixture_user/123_7/palette_roi_cache_stage")
    )


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
    open_kwargs: dict[str, object] = {}

    def _open_group(*_args: object, **kwargs: object) -> _Group:
        open_kwargs.update(kwargs)
        return root

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)

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

    mod.write_keypoint_model_resolution_provenance(
        zarr_path=tmp_path / "sample_analysis.zarr",
        run_name="keypoints_20260209_000000",
        payload=payload,
    )

    assert keypoint_run.attrs.get("model_resolution_mode") == "registry"
    assert keypoint_run.attrs.get("model_resolution_task") == "pose"
    assert keypoint_run.attrs.get("model_resolution_selected_run_id") == "pose_run_001"
    assert keypoint_run.attrs.get("model_resolution_selected_set_id") == "pose_set_001"
    assert open_kwargs.get("use_consolidated") is False
    provenance = keypoint_run.attrs.get("provenance")
    assert isinstance(provenance, dict)
    assert "model_resolution" in provenance


def test_write_model_resolution_provenance_can_target_keypoint_shard_parent(
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
    shard_parent = _Group()
    shard_run = _Group()
    shard_parent["keypoint_shard_001"] = shard_run
    root["keypoint_shard_runs"] = shard_parent
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root)

    payload = {
        "mode": "registry",
        "task": "pose",
        "registry_path": "/nvme1/palette_registry.sqlite",
        "recording_id": "rec_001",
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

    mod.write_keypoint_model_resolution_provenance(
        zarr_path=tmp_path / "sample_analysis.zarr",
        run_name="keypoint_shard_001",
        payload=payload,
        output_parent="keypoint_shard_runs",
    )

    assert shard_run.attrs.get("model_resolution_mode") == "registry"
    assert shard_run.attrs.get("model_resolution_selected_run_id") == "pose_run_001"
    provenance = shard_run.attrs.get("provenance")
    assert isinstance(provenance, dict)
    assert provenance["model_resolution"] == payload


def test_stage_flat_roi_cache_manifest_copies_payload_and_rewrites_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_bin = source_dir / "sample.flat_roi_cache.bin"
    source_bin.write_bytes(bytes(range(8)))
    source_manifest = source_dir / "sample.flat_roi_cache.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": "palette_roi_cache_flat_bin_v1",
                "layout": "flat_bin_v1",
                "cache_complete": True,
                "source": {
                    "archive_path": str(tmp_path / "recording_analysis.zarr"),
                    "crop_run_name": "crop_test",
                },
                "array": {
                    "bin_path": source_bin.name,
                    "dtype": "uint8",
                    "shape": [2, 2, 2],
                    "order": "C",
                },
            }
        ),
        encoding="utf-8",
    )

    local_manifest, details = mod._stage_flat_roi_cache_manifest(  # noqa: SLF001
        source_manifest,
        staging_dir=tmp_path / "scratch",
    )

    local_bin = local_manifest.with_suffix(".bin")
    assert local_manifest.exists()
    assert local_bin.read_bytes() == source_bin.read_bytes()
    assert details["staged"] is True
    assert details["policy"] == "node_scratch_staged_flat_cache"
    assert details["stage_to_scratch_requested"] is True
    assert details["requested_manifest_path"] == str(source_manifest.resolve())
    assert details["effective_manifest_path"] == str(local_manifest)
    assert details["payload_size_bytes"] == 8
    assert details["payload_copy"]["size_bytes"] == 8
    assert details["validation_status"] == "ok"
    assert details["staging_recommendation_min_bytes"] == mod.ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES

    staged_payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    assert staged_payload["array"]["bin_path"] == local_bin.name
    assert staged_payload["staging"]["policy"] == "node_scratch_staged_flat_cache"
    assert staged_payload["staging"]["manifest_publish_policy"] == "payload_first_manifest_last"


def test_prepare_roi_cache_manifest_requires_manifest_for_staging() -> None:
    with pytest.raises(ValueError, match="requires --roi-cache-manifest"):
        mod._prepare_roi_cache_manifest(  # noqa: SLF001
            None,
            stage_to_scratch=True,
            staging_dir=None,
        )


def test_staging_recommendation_marks_large_prfs_cache() -> None:
    payload = mod._staging_recommendation_payload(  # noqa: SLF001
        manifest_path=Path("/groups/johnson/johnsonlab/jeremy/cache/sample.flat_roi_cache.json"),
        source_tier="prfs_workflow_scratch",
        payload_size_bytes=mod.ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES,
    )

    assert payload["staging_recommended"] is True
    assert payload["staging_recommendation_reason"] == "large_prfs_flat_cache"
    assert "GoodCopBadCop L4 benchmark" in payload["staging_recommendation_basis"]


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
        model_sha256="e" * 64,
        created_utc="2026-02-09T00:00:00+00:00",
        status="success",
        dataset_count=4,
        weighted_score=0.99,
        feature_match_counts={"rig_id": 4},
        feature_weights_used=1.0,
    )

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)
    monkeypatch.setattr(mod, "resolve_recording_id", lambda *_args, **_kwargs: "rec_001")
    monkeypatch.setattr(mod, "load_target_profile", lambda *_args, **_kwargs: target)

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

    monkeypatch.setattr(mod, "load_candidates", _fake_load_candidates)
    model_pose_schema_binding = {
        "schema_id": "palette.pose_model_schema_binding",
        "binding_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        mod,
        "resolve_registered_pose_model_schema_binding",
        lambda *_args, **_kwargs: model_pose_schema_binding,
    )

    def _fake_detect_keypoints_yolo(**kwargs: object) -> str:
        calls["detect_kwargs"] = kwargs
        return "keypoints_001"

    monkeypatch.setattr(mod, "detect_keypoints_yolo", _fake_detect_keypoints_yolo)

    def _fake_write_model_resolution_provenance(
        *,
        zarr_path: Path,
        run_name: str,
        payload: dict[str, object],
        output_parent: str = "keypoints_runs",
    ) -> None:
        calls["write_zarr_path"] = zarr_path
        calls["write_run_name"] = run_name
        calls["write_payload"] = payload
        calls["write_output_parent"] = output_parent

    monkeypatch.setattr(mod, "write_keypoint_model_resolution_provenance", _fake_write_model_resolution_provenance)

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
            "--model-run-id",
            "pose_run_123",
            "--output-parent",
            "keypoint_shard_runs",
            "--pose-schema",
            "traditional_v2",
            "--keypoint-roi-shard-rows",
            "262144",
            "--keypoint-frame-shard-rows",
            "262144",
            "--include-non-success",
            "--cpu",
            "--roi-cache-policy",
            "always",
            "--roi-cache-dir",
            str(tmp_path / "roi-cache"),
            "--roi-cache-manifest",
            str(tmp_path / "flat-cache" / "sample.flat_roi_cache.json"),
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
    assert detect_kwargs.get("model_sha256") == "e" * 64
    assert (
        detect_kwargs.get("model_pose_schema_binding")
        == model_pose_schema_binding
    )
    assert detect_kwargs.get("output_parent") == "keypoint_shard_runs"
    assert detect_kwargs.get("pose_schema") == "traditional_v2"
    assert detect_kwargs.get("keypoint_roi_shard_rows") == 262144
    assert detect_kwargs.get("keypoint_frame_shard_rows") == 262144
    assert detect_kwargs.get("device") == "cpu"
    assert detect_kwargs.get("roi_cache_policy") == "always"
    assert detect_kwargs.get("roi_cache_dir") == tmp_path / "roi-cache"
    assert detect_kwargs.get("roi_cache_manifest") == (tmp_path / "flat-cache" / "sample.flat_roi_cache.json").resolve()
    assert detect_kwargs.get("roi_cache_staged_to_node_scratch") is False
    assert detect_kwargs.get("roi_cache_source_tier") == "node_scratch"
    staging_details = detect_kwargs.get("roi_cache_staging_details")
    assert isinstance(staging_details, dict)
    assert staging_details.get("staged") is False
    assert staging_details.get("policy") == "direct_manifest_read"
    assert staging_details.get("stage_to_scratch_requested") is False
    assert staging_details.get("staging_recommendation_min_bytes") == mod.ROI_CACHE_STAGING_RECOMMENDED_MIN_BYTES
    assert Path(str(detect_kwargs.get("registry"))) == registry_path.resolve()
    assert detect_kwargs.get("run_provenance") == detect_kwargs.get("cli_provenance")
    run_provenance = detect_kwargs.get("run_provenance")
    assert isinstance(run_provenance, dict)
    assert validate_run_provenance(run_provenance).valid is True
    assert run_provenance["command"] == "fisheye.utils.run_keypoints_with_registry_model"
    assert run_provenance["input_run_ids"] == {
        "crop_run": None,
        "model_run": "pose_run_123",
        "model_set": "pose_set_123",
    }

    assert calls.get("write_zarr_path") == output_path.resolve()
    assert calls.get("write_run_name") == "keypoints_001"
    assert calls.get("write_output_parent") == "keypoint_shard_runs"
    payload = calls.get("write_payload")
    assert isinstance(payload, dict)
    assert payload.get("task") == "pose"
    selected = payload.get("selected")
    assert isinstance(selected, dict)
    assert selected.get("model_sha256") == "e" * 64
    assert payload["artifacts"]["model_pose_schema_binding"] == (
        model_pose_schema_binding
    )
    parameters = payload.get("parameters")
    assert isinstance(parameters, dict)
    assert parameters.get("pose_schema") == "traditional_v2"
    assert parameters.get("model_run_id") == "pose_run_123"
    assert parameters.get("roi_cache_policy") == "always"
    assert parameters.get("roi_cache_manifest") == str((tmp_path / "flat-cache" / "sample.flat_roi_cache.json"))
    for key in ("contract", "command", "git", "environment", "platform", "parameters", "inputs", "artifacts"):
        assert key in payload
