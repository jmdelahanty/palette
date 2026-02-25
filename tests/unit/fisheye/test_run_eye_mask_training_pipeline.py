"""Tests for eye-mask pipeline orchestration wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import run_eye_mask_training_pipeline as pipeline


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
        context={"canvas_name": "DefaultScreen", "rig_id": "omnifin0"},
        protocol_name="DefaultScreen",
        protocol_hash=None,
        acquisition={
            "dish_design": "cedar",
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )
    db.close()


def _write_preflight_files(config_path: Path, manifest_path: Path, source_zarr: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "datasets:\n  test_merged:\n    zarr_path: /tmp/placeholder.zarr\ntraining_params:\n  label_mode: lr\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "set_id": "eye_mask_smoke_v001",
                "set_name": "eye_mask_smoke",
                "datasets": [
                    {
                        "name": "eye_mask_smoke_merged",
                        "dataset_id": "eye_mask_smoke_v001_merged",
                        "out_zarr": str(config_path.parent / "zarr" / "eye_mask_smoke_v001_merged.zarr"),
                        "run_name": "merged_eye_masks",
                        "export_status": "planned",
                    }
                ],
                "selected_sources": [
                    {
                        "dataset_id": "dataset_1",
                        "zarr_path": str(source_zarr),
                        "source_eye_stage": "refined_eye_masks_runs",
                        "source_eye_run": "refined_eye_masks_001",
                        "source_crop_run": "crop_001",
                    }
                ],
                "merged_export": {
                    "dataset_id": "eye_mask_smoke_v001_merged",
                    "dataset_name": "eye_mask_smoke_merged",
                    "zarr_path": str(config_path.parent / "zarr" / "eye_mask_smoke_v001_merged.zarr"),
                    "export_status": "planned",
                },
                "execution": {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0},
            }
        ),
        encoding="utf-8",
    )


def test_train_runs_eye_masks_after_preflight(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "eye_mask.yaml"
    out_manifest = tmp_path / "prep" / "eye_mask.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        _write_preflight_files(out_config, out_manifest, tmp_path / "dataset_1.zarr")
        return 0

    def fake_run(cmd: list[str], check: bool = False):
        del check
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
    assert "eye_mask_smoke_v001" in train_cmd
    assert "--registry" in train_cmd
    assert str(registry_path) in train_cmd


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


def test_export_merged_invokes_exporter(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    source_zarr = tmp_path / "dataset_1.zarr"
    _seed_registry(registry_path, source_zarr)

    out_config = tmp_path / "prep" / "eye_mask.yaml"
    out_manifest = tmp_path / "prep" / "eye_mask.manifest.json"
    merged_out = tmp_path / "merged_eye.zarr"
    calls: dict[str, object] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        _write_preflight_files(out_config, out_manifest, source_zarr)
        return 0

    def fake_export(*, source_specs, out_zarr, **kwargs):
        calls["export"] = {
            "source_specs": list(source_specs),
            "out_zarr": Path(out_zarr),
            "kwargs": dict(kwargs),
        }
        return {
            "zarr_path": str(out_zarr),
            "total_samples": 42,
            "source_count": len(list(source_specs)),
        }

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(
        pipeline.export_zarr,
        "export_merged_eye_mask_training_zarr_from_sources",
        fake_export,
    )

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--export-merged",
            "--merge-out-zarr",
            str(merged_out),
            "--merge-split",
            "0.7/0.3",
            "--merge-seed",
            "123",
            "--merge-overwrite",
            "--no-aggregate-training-data-card",
        ]
    )
    assert rc == 0

    export_call = calls["export"]
    assert export_call["out_zarr"] == merged_out
    assert export_call["kwargs"]["split_train"] == pytest.approx(0.7)
    assert export_call["kwargs"]["split_val"] == pytest.approx(0.3)
    assert export_call["kwargs"]["split_test"] == pytest.approx(0.0)
    assert export_call["kwargs"]["split_seed"] == 123
    assert export_call["kwargs"]["overwrite"] is True
    assert len(export_call["source_specs"]) == 1

    payload = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert payload["datasets"][0]["out_zarr"] == str(merged_out)
    assert payload["datasets"][0]["export_status"] == "succeeded"
    assert payload["execution"] == {"mode": "apply", "planned": 1, "succeeded": 1, "failed": 0}


def test_data_card_flags_are_forwarded_to_aggregator(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    source_zarr = tmp_path / "dataset_1.zarr"
    _seed_registry(registry_path, source_zarr)

    out_config = tmp_path / "prep" / "eye_mask.yaml"
    out_manifest = tmp_path / "prep" / "eye_mask.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        _write_preflight_files(out_config, out_manifest, source_zarr)
        return 0

    def fake_aggregate(cli: list[str]) -> int:
        calls["aggregate"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)
    monkeypatch.setattr(pipeline.aggregate_data_card, "main", fake_aggregate)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--aggregate-training-data-card",
            "--data-card-allow-profile-mtime-mismatch",
            "--data-card-allow-profile-fallback-scan",
            "--data-card-view",
            "--data-card-force-plots",
        ]
    )
    assert rc == 0
    aggregate_cli = calls["aggregate"]
    assert "--allow-profile-mtime-mismatch" in aggregate_cli
    assert "--allow-profile-fallback-scan" in aggregate_cli
    assert "--view" in aggregate_cli
    assert "--force" in aggregate_cli


def test_data_card_view_cannot_combine_no_plots(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    with pytest.raises(SystemExit) as exc:
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--aggregate-training-data-card",
                "--data-card-view",
                "--data-card-no-plots",
            ]
        )
    assert int(exc.value.code) == 2


def test_auto_sets_out_manifest_when_missing(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    source_zarr = tmp_path / "dataset_1.zarr"
    _seed_registry(registry_path, source_zarr)
    dataset_root = tmp_path / "datasets_root"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))
    monkeypatch.chdir(tmp_path)

    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> int:
        calls["prepare"] = list(cli)
        out_manifest = Path(cli[cli.index("--out-manifest") + 1])
        out_config = Path(cli[cli.index("--out-config") + 1])
        _write_preflight_files(out_config, out_manifest, source_zarr)
        return 0

    monkeypatch.setattr(pipeline.prepare_from_registry, "main", fake_prepare)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--set-name",
            "eye_smoke",
            "--set-version",
            "1",
        ]
    )
    assert rc == 0
    prepare_cli = calls["prepare"]
    manifest_path = Path(prepare_cli[prepare_cli.index("--out-manifest") + 1])
    expected_manifest = dataset_root / "eye_mask_eye_smoke_v001" / "eye_mask_eye_smoke_v001.manifest.json"
    assert manifest_path.resolve() == expected_manifest.resolve()
