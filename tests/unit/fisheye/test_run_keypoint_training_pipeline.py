"""Tests for keypoint pipeline orchestration wrapper."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import run_keypoint_training_pipeline as pipeline


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    db.upsert_dataset(
        "dataset_1",
        session_uuid="dataset_1",
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        "dataset_1",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )
    db.close()


def test_train_runs_keypoint_after_preflight(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "pose.yaml"
    out_manifest = tmp_path / "prep" / "pose.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: pose\n", encoding="utf-8")
        out_manifest.write_text('{"set_id":"pose_smoke_v001"}', encoding="utf-8")
        return 0

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--train",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert str(out_config) in train_cmd
    assert "--manifest" in train_cmd
    assert str(out_manifest) in train_cmd
    assert "--set-id" in train_cmd
    assert "pose_smoke_v001" in train_cmd
    assert "--registry" in train_cmd
    assert str(registry_path) in train_cmd


def test_train_requires_manifest_set_id(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "pose.yaml"
    out_manifest = tmp_path / "prep" / "pose.manifest.json"

    def fake_prepare(cli: list[str]) -> int:
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: pose\n", encoding="utf-8")
        out_manifest.write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)

    with pytest.raises(SystemExit, match="--train requires manifest set_id"):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--out-config",
                str(out_config),
                "--out-manifest",
                str(out_manifest),
                "--train",
            ]
        )


def test_train_cannot_combine_dry_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    with pytest.raises(SystemExit, match="--train cannot be combined with --dry-run"):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--train",
                "--dry-run",
            ]
        )


def test_auto_sets_out_manifest_when_missing(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")
    dataset_root = tmp_path / "datasets_root"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))
    monkeypatch.chdir(tmp_path)

    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        out_manifest = Path(cli[cli.index("--out-manifest") + 1])
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text('{"set_id":"pose_pose_smoke_v001"}', encoding="utf-8")
        out_config = Path(cli[cli.index("--out-config") + 1])
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: pose\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--set-name",
            "pose_smoke",
            "--set-version",
            "1",
        ]
    )
    assert rc == 0
    prepare_cli = calls["prepare"]
    manifest_path = Path(prepare_cli[prepare_cli.index("--out-manifest") + 1])
    expected_manifest = dataset_root / "pose_pose_smoke_v001" / "pose_pose_smoke_v001.manifest.json"
    assert manifest_path.resolve() == expected_manifest.resolve()


def test_export_merged_invokes_exporter(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "out.manifest.json"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        manifest_path.write_text(json.dumps({"set_id": "pose_set_v001"}), encoding="utf-8")
        return 0

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--export-merged",
            "--merge-out-zarr",
            str(tmp_path / "merged_pose.zarr"),
            "--merge-split",
            "0.7/0.3",
            "--merge-seed",
            "123",
            "--merge-overwrite",
        ]
    )
    assert rc == 0

    prepare_cli = calls["prepare"]
    export_cli = calls["export"]
    assert "--registry" in prepare_cli
    assert str(registry_path) in prepare_cli
    assert "--out-manifest" in prepare_cli
    assert "--merge" in export_cli
    assert "--manifest" in export_cli
    assert str(manifest_path) in export_cli
    assert "--registry" in export_cli
    assert str(registry_path) in export_cli
    assert "--out-zarr" in export_cli
    assert str(tmp_path / "merged_pose.zarr") in export_cli
    assert "--split" in export_cli
    assert "0.7/0.3" in export_cli
    assert "--seed" in export_cli
    assert "123" in export_cli
    assert "--overwrite" in export_cli
    assert "--no-aggregate-training-data-card" in export_cli


def test_passes_cross_method_review_fallback_flag_to_preflight(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")
    manifest_path = tmp_path / "pose.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"set_id":"pose_set_v001"}', encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--allow-cross-method-review-fallback",
        ]
    )
    assert rc == 0
    assert "--allow-cross-method-review-fallback" in calls["prepare"]


def test_passes_skeleton_id_to_preflight(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")
    manifest_path = tmp_path / "pose.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"set_id":"pose_set_v001"}', encoding="utf-8")
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--skeleton-id",
            "pose_skel_traditional_v2",
        ]
    )
    assert rc == 0
    prepare_cli = calls["prepare"]
    assert "--skeleton-id" in prepare_cli
    assert "pose_skel_traditional_v2" in prepare_cli


def test_train_after_merged_export_uses_merged_outputs(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    preflight_manifest = tmp_path / "prep" / "pose_v001.manifest.json"
    merged_out_dir = tmp_path / "datasets" / "pose_set_v001"
    merged_config = merged_out_dir / "pose_set_v001.yaml"
    merged_manifest = merged_out_dir / "pose_set_v001.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        preflight_manifest.parent.mkdir(parents=True, exist_ok=True)
        preflight_manifest.write_text(json.dumps({"set_id": "pose_set_v001"}), encoding="utf-8")
        return 0

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        merged_out_dir.mkdir(parents=True, exist_ok=True)
        merged_config.write_text("task: pose\n", encoding="utf-8")
        merged_manifest.write_text('{"set_id":"pose_set_v001"}', encoding="utf-8")
        return 0

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--set-name",
            "pose_set",
            "--set-version",
            "1",
            "--out-manifest",
            str(preflight_manifest),
            "--export-merged",
            "--merge-out-dir",
            str(merged_out_dir),
            "--train",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert str(merged_config) in train_cmd
    assert "--manifest" in train_cmd
    assert str(merged_manifest) in train_cmd
    assert "--set-id" in train_cmd
    assert "pose_set_v001" in train_cmd
    assert "--registry" in train_cmd
    assert str(registry_path) in train_cmd


def test_data_card_profile_flags_are_forwarded_to_aggregator(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")
    manifest_path = tmp_path / "pose.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"set_id": "pose_set_v001"}), encoding="utf-8")
        return 0

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    def fake_aggregate(*, cli: list[str], required: bool) -> int:
        del required
        calls["aggregate"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)
    monkeypatch.setattr(pipeline, "_run_keypoint_data_card_aggregation", fake_aggregate)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--export-merged",
            "--data-card-allow-profile-mtime-mismatch",
            "--data-card-allow-profile-fallback-scan",
        ]
    )
    assert rc == 0
    aggregate_cli = calls["aggregate"]
    assert "--allow-profile-mtime-mismatch" in aggregate_cli
    assert "--allow-profile-fallback-scan" in aggregate_cli
