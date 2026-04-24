"""Tests for subject-mask training pipeline orchestration wrapper."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import run_subject_mask_training_pipeline as pipeline


def _write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "datasets:\n"
            "  subject_merged:\n"
            "    zarr_path: /tmp/old_subject_masks.zarr\n"
            "    crop_run: old_crop_run\n"
            "    subject_mask_run: old_subject_run\n"
            "names: [subject_body, eyes_union, swim_bladder]\n"
            "nc: 3\n"
            "training_params:\n"
            "  model: unet_small\n"
            "  epochs: 1\n"
            "  batch_size: 2\n"
            "  imgsz: [64, 64]\n"
            "  lr0: 0.001\n"
            "  momentum: 0.9\n"
            "  weight_decay: 0.0005\n"
            "  patience: 2\n"
            "  device: cpu\n"
            "  subject_masks_run: old_global_subject\n"
            "  crop_run: old_global_crop\n"
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, source_zarr: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "set_id": "subject_mask_smoke_v001",
                "set_name": "subject_mask_smoke",
                "datasets": [
                    {
                        "name": "subject_mask_smoke_merged",
                        "dataset_id": "subject_mask_smoke_v001_merged",
                        "out_zarr": str(path.parent / "zarr" / "subject_mask_smoke_v001_merged.zarr"),
                        "run_name": "planned_subject_masks",
                        "export_status": "planned",
                    }
                ],
                "selected_sources": [
                    {
                        "dataset_id": "dataset_1",
                        "zarr_path": str(source_zarr),
                        "source_stage_group": "refined_subject_masks_runs",
                        "source_subject_mask_run": "refined_subject_masks_001",
                        "source_crop_run": "crop_001",
                    }
                ],
                "merged_export": {
                    "dataset_id": "subject_mask_smoke_v001_merged",
                    "dataset_name": "subject_mask_smoke_merged",
                    "zarr_path": str(path.parent / "zarr" / "subject_mask_smoke_v001_merged.zarr"),
                    "run_name": "planned_subject_masks",
                    "export_status": "planned",
                },
                "execution": {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0},
            }
        ),
        encoding="utf-8",
    )


def test_train_runs_subject_mask_trainer(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "subject_mask.yaml"
    manifest_path = tmp_path / "subject_mask.manifest.json"
    registry_path = tmp_path / "registry.sqlite"
    _write_config(config_path)
    _write_manifest(manifest_path, tmp_path / "dataset_1.zarr")
    calls: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], check: bool = False):
        del check
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--train",
            "--run-name",
            "subject_train_smoke",
            "--project",
            str(tmp_path / "models"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--no-progress",
            "--no-compile",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert "fisheye.segmentation.train_unet_subject_masks" in train_cmd
    assert str(config_path) in train_cmd
    assert "--manifest" in train_cmd
    assert train_cmd[train_cmd.index("--manifest") + 1] == str(manifest_path)
    assert "--set-id" in train_cmd
    assert train_cmd[train_cmd.index("--set-id") + 1] == "subject_mask_smoke_v001"
    assert "--registry" in train_cmd
    assert train_cmd[train_cmd.index("--registry") + 1] == str(registry_path)
    assert "--output-dir" in train_cmd
    assert train_cmd[train_cmd.index("--output-dir") + 1] == str(tmp_path / "models")
    assert "--device" in train_cmd and "cpu" in train_cmd
    assert "--epochs" in train_cmd and "1" in train_cmd
    assert "--no-progress" in train_cmd
    assert "--no-compile" in train_cmd


def test_export_merged_invokes_exporter_and_rewrites_outputs(tmp_path: Path, monkeypatch) -> None:
    source_zarr = tmp_path / "dataset_1.zarr"
    config_path = tmp_path / "prep" / "subject_mask.yaml"
    manifest_path = tmp_path / "prep" / "subject_mask.manifest.json"
    out_config = tmp_path / "merged" / "subject_mask.yaml"
    out_manifest = tmp_path / "merged" / "subject_mask.manifest.json"
    merged_out = tmp_path / "merged" / "subject_masks.zarr"
    registry_path = tmp_path / "registry.sqlite"
    _write_config(config_path)
    _write_manifest(manifest_path, source_zarr)
    calls: dict[str, object] = {}

    def fake_export(*, source_specs, out_zarr, **kwargs):
        calls["export"] = {
            "source_specs": list(source_specs),
            "out_zarr": Path(out_zarr),
            "kwargs": dict(kwargs),
        }
        return {
            "run_name": kwargs["run_name"],
            "zarr_path": str(out_zarr),
            "total_samples": 42,
            "label_schema_id": kwargs["subject_label_schema"],
            "source_count": len(list(source_specs)),
        }

    monkeypatch.setattr(
        pipeline.export_zarr,
        "export_merged_subject_mask_training_zarr_from_sources",
        fake_export,
    )

    rc = pipeline.main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--registry",
            str(registry_path),
            "--training-set-name",
            "Subject Mask Smoke",
            "--export-merged",
            "--merge-out-zarr",
            str(merged_out),
            "--merge-split",
            "0.7/0.2/0.1",
            "--merge-seed",
            "123",
            "--merge-run-name",
            "merged_subject_masks_test",
            "--subject-label-schema",
            "subject_v1_union",
            "--input-format",
            "gray",
            "--merge-overwrite",
        ]
    )
    assert rc == 0

    export_call = calls["export"]
    assert export_call["out_zarr"] == merged_out
    assert export_call["kwargs"]["subject_label_schema"] == "subject_v1_union"
    assert export_call["kwargs"]["input_format"] == "gray"
    assert export_call["kwargs"]["train_ratio"] == pytest.approx(0.7)
    assert export_call["kwargs"]["val_ratio"] == pytest.approx(0.2)
    assert export_call["kwargs"]["test_ratio"] == pytest.approx(0.1)
    assert export_call["kwargs"]["split_seed"] == 123
    assert export_call["kwargs"]["run_name"] == "merged_subject_masks_test"
    assert export_call["kwargs"]["overwrite"] is True
    assert export_call["kwargs"]["registry"] == registry_path
    assert export_call["kwargs"]["training_set_id"] == "subject_mask_smoke_v001"
    assert export_call["kwargs"]["training_set_name"] == "Subject Mask Smoke"
    source_specs = export_call["source_specs"]
    assert len(source_specs) == 1
    assert source_specs[0].source_zarr == source_zarr
    assert source_specs[0].subject_run == "refined_subject_masks_001"
    assert source_specs[0].crop_run == "crop_001"
    assert source_specs[0].stage_group == "refined_subject_masks_runs"

    manifest_payload = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["datasets"][0]["out_zarr"] == str(merged_out)
    assert manifest_payload["datasets"][0]["zarr_path"] == str(merged_out)
    assert manifest_payload["datasets"][0]["run_name"] == "merged_subject_masks_test"
    assert manifest_payload["datasets"][0]["export_status"] == "succeeded"
    assert manifest_payload["merged_export"]["zarr_path"] == str(merged_out)
    assert manifest_payload["merged_export"]["run_name"] == "merged_subject_masks_test"
    assert manifest_payload["execution"] == {"mode": "apply", "planned": 1, "succeeded": 1, "failed": 0}

    config_payload = pipeline.yaml.safe_load(out_config.read_text(encoding="utf-8"))
    dataset_cfg = config_payload["datasets"]["subject_merged"]
    assert dataset_cfg["zarr_path"] == str(merged_out)
    assert dataset_cfg["crop_run"] == "merged_subject_masks_test"
    assert dataset_cfg["subject_mask_run"] == "merged_subject_masks_test"
    assert config_payload["training_params"]["crop_run"] == "merged_subject_masks_test"
    assert config_payload["training_params"]["subject_masks_run"] == "merged_subject_masks_test"
    assert config_payload["training_params"]["label_schema_id"] == "subject_v1_union"

    original_config = pipeline.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert original_config["datasets"]["subject_merged"]["zarr_path"] == "/tmp/old_subject_masks.zarr"


def test_train_after_export_uses_rewritten_outputs(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "prep" / "subject_mask.yaml"
    manifest_path = tmp_path / "prep" / "subject_mask.manifest.json"
    out_config = tmp_path / "merged" / "subject_mask.yaml"
    out_manifest = tmp_path / "merged" / "subject_mask.manifest.json"
    merged_out = tmp_path / "merged" / "subject_masks.zarr"
    registry_path = tmp_path / "registry.sqlite"
    _write_config(config_path)
    _write_manifest(manifest_path, tmp_path / "dataset_1.zarr")
    calls: dict[str, object] = {}

    def fake_export(*, source_specs, out_zarr, **kwargs):
        calls["export"] = list(source_specs)
        return {
            "run_name": kwargs["run_name"],
            "zarr_path": str(out_zarr),
            "total_samples": 3,
            "label_schema_id": "subject_v1_union",
            "source_count": 1,
        }

    def fake_run(cmd: list[str], check: bool = False):
        del check
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(
        pipeline.export_zarr,
        "export_merged_subject_mask_training_zarr_from_sources",
        fake_export,
    )
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--registry",
            str(registry_path),
            "--export-merged",
            "--merge-out-zarr",
            str(merged_out),
            "--merge-run-name",
            "merged_subject_masks_test",
            "--train",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert str(out_config) in train_cmd
    assert "--manifest" in train_cmd
    assert train_cmd[train_cmd.index("--manifest") + 1] == str(out_manifest)
    assert "--set-id" in train_cmd
    assert train_cmd[train_cmd.index("--set-id") + 1] == "subject_mask_smoke_v001"
    assert "--registry" in train_cmd
    assert train_cmd[train_cmd.index("--registry") + 1] == str(registry_path)


def test_export_merged_requires_manifest() -> None:
    with pytest.raises(SystemExit, match="--export-merged requires --manifest"):
        pipeline.main(["--export-merged"])
