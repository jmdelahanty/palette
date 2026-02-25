"""Tests for eye-mask registry preflight manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_eye_mask_training_from_registry as wrapper


def _seed_dataset(db: Registry, *, dataset_id: str, zarr_path: Path) -> None:
    zarr_path.mkdir(parents=True, exist_ok=True)
    db.upsert_dataset(
        dataset_id,
        session_uuid=f"{dataset_id}_session",
        zarr_path=zarr_path,
        recording_id=f"{dataset_id}_recording",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    db.upsert_provenance(
        dataset_id,
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


def _seed_eye_profile(
    db: Registry,
    *,
    dataset_id: str,
    review_state: str,
    review_intended_use: str,
    usable_rate: float,
) -> None:
    db.upsert_eye_mask_data_profile(
        dataset_id=dataset_id,
        profile_run=f"profile_{dataset_id}",
        recording_id=f"{dataset_id}_recording",
        zarr_use="analysis",
        stage_group="refined_eye_masks_runs",
        eye_mask_method="traditional",
        source_eye_mask_path="refined_eye_masks_runs/refined_eye_masks_001",
        source_eye_mask_run="refined_eye_masks_001",
        source_keypoint_path="refined_keypoints_runs/refined_keypoints_001",
        source_keypoint_run="refined_keypoints_001",
        source_crop_run="crop_001",
        profile_created_utc="2026-02-25T00:00:00+00:00",
        rows_total=100,
        rows_usable=int(round(100 * float(usable_rate))),
        usable_rate=float(usable_rate),
        reviewed_rate=1.0,
        excluded_rate=max(0.0, 1.0 - float(usable_rate)),
        exclusion_reasons_json="{}",
        ellipse_success_rate=0.95,
        pair_success_rate=0.9,
        area_p10=None,
        area_p50=None,
        area_p90=None,
        major_axis_p10=None,
        major_axis_p50=None,
        major_axis_p90=None,
        minor_axis_p10=None,
        minor_axis_p50=None,
        minor_axis_p90=None,
        aspect_ratio_p10=None,
        aspect_ratio_p50=None,
        aspect_ratio_p90=None,
        eye_separation_p10=None,
        eye_separation_p50=None,
        eye_separation_p90=None,
        edge_proximity_rate=None,
        review_state=review_state,
        review_method="manual",
        review_intended_use=review_intended_use,
        review_timestamp_utc="2026-02-25T00:00:00+00:00",
        source_keypoint_stale_state="fresh",
        source_keypoint_stale_reason=None,
        source_keypoint_stale_timestamp_utc=None,
        source_keypoint_stale_json=None,
        rig_id="omnifin0",
        camera_id="cam_1",
        arena_id="arena_1",
        dish_design="cedar",
        canvas_name="DefaultScreen",
        protocol_name="DefaultScreen",
        genotype=None,
        dpf_at_acquisition=None,
        profile_json="{}",
        zarr_mtime_ns=None,
        updated_utc="2026-02-25T00:00:00+00:00",
    )


def _monkeypatch_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_selection(
        zarr_path: Path,
        *,
        crop_run: str | None,
        eye_stage: str,
        eye_run: str | None,
    ) -> wrapper.EyeSourceSelection:
        del zarr_path
        return wrapper.EyeSourceSelection(
            crop_run=crop_run or "crop_001",
            eye_stage="refined_eye_masks_runs" if eye_stage == "auto" else eye_stage,
            eye_run=eye_run or "refined_eye_masks_001",
            total_samples=42,
            channels=2,
            mask_probs_name="mask_probs_roi_refined",
            method="traditional",
            source_keypoints_run="refined_keypoints_001",
            source_keypoint_group="refined_keypoints_runs",
        )

    monkeypatch.setattr(wrapper, "_inspect_source_selection", _fake_selection)


def _monkeypatch_export_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_export(*args, **kwargs):
        del args, kwargs
        raise AssertionError("prepare_eye_mask_training_from_registry should not execute export in prepare-only mode")

    monkeypatch.setattr(wrapper.export_eye, "export_merged_eye_mask_training_zarr_from_sources", _fail_export)


def _write_base_config(path: Path) -> Path:
    base_config = {
        "datasets": {
            "template": {
                "zarr_path": "/tmp/template.zarr",
                "mask_preference": "raw_probs",
                "include_empty": False,
            }
        },
        "names": ["eye"],
        "nc": 1,
        "random_seed": 42,
        "target_size": 160,
        "default_split": {"train": 0.8, "val": 0.2},
        "training_params": {
            "model": "unet-small",
            "epochs": 2,
            "batch": 4,
            "imgsz": 160,
            "lr0": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 10,
            "device": "cpu",
            "project": "runs/eye_masks",
            "label_source": "yolo",
            "label_mode": "union",
        },
        "num_workers": 0,
        "cache": {
            "enabled": False,
            "directory": "runs/cache/eye_masks",
            "reuse_existing": True,
            "workers": 1,
            "backend": "thread",
        },
    }
    path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return path


def test_prepare_eye_mask_from_registry_auto_set_name_and_set_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "eye_base.yaml")
    dataset_root = tmp_path / "datasets"
    source_path = tmp_path / "source_a.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    db.close()

    _monkeypatch_selection(monkeypatch)
    _monkeypatch_export_not_called(monkeypatch)
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--set-version",
            "1",
        ]
    )
    assert rc == 0

    manifests = sorted(dataset_root.glob("eye_mask_*_v001/*.manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))

    set_name = payload["set_name"]
    set_id = payload["set_id"]
    assert set_name.startswith("cedar_defaultscreen_omnifin0_auto_gray_lr_")
    assert len(set_name.rsplit("_", 1)[-1]) == 8
    assert set_id == f"eye_mask_{set_name}_v001"
    out_config = dataset_root / set_id / f"{set_id}.yaml"
    assert payload["base_config_path"] == str(base_config_path)
    assert payload["output_config_path"] == str(out_config)
    assert out_config.exists()

    assert len(payload["selected_sources"]) == 1
    assert payload["selected_sources"][0]["dataset_id"] == "dataset_a"

    merged = payload["datasets"][0]
    assert merged["dataset_id"] == f"{set_id}_merged"
    assert merged["name"] == f"{set_id}_merged"
    assert merged["out_zarr"] == str(dataset_root / set_id / "zarr" / f"{set_id}_merged.zarr")
    assert merged["export_status"] == "planned"
    cmd = merged["export_command"]
    assert "--training-set-id" in cmd
    assert f"--training-set-id {set_id}" in cmd
    assert payload["merged_export"]["source_count"] == 1
    assert payload["merged_export"]["zarr_path"] == merged["out_zarr"]
    assert payload["execution"] == {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0}

    config = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    cfg_dataset = config["datasets"][f"{set_id}_merged"]
    assert cfg_dataset["zarr_path"] == merged["out_zarr"]
    assert cfg_dataset["crop_run"] == merged["run_name"]
    assert cfg_dataset["mask_run"] == merged["run_name"]
    assert cfg_dataset["mask_preference"] == "raw_probs"
    assert cfg_dataset["include_empty"] is False
    assert config["training_params"]["label_mode"] == "lr"
    assert config["random_seed"] == 42


def test_prepare_eye_mask_from_registry_quality_filters_select_matching_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "eye_base.yaml")
    source_a = tmp_path / "source_a.zarr"
    source_b = tmp_path / "source_b.zarr"
    manifest_path = tmp_path / "prepared.manifest.json"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_a)
    _seed_dataset(db, dataset_id="dataset_b", zarr_path=source_b)
    _seed_eye_profile(
        db,
        dataset_id="dataset_a",
        review_state="approved",
        review_intended_use="training",
        usable_rate=0.95,
    )
    _seed_eye_profile(
        db,
        dataset_id="dataset_b",
        review_state="pending",
        review_intended_use="training",
        usable_rate=0.95,
    )
    db.close()

    _monkeypatch_selection(monkeypatch)
    _monkeypatch_export_not_called(monkeypatch)

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-manifest",
            str(manifest_path),
            "--eye-mask-method",
            "traditional",
            "--min-usable-rate",
            "0.9",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["set_id"] is None
    assert len(payload["datasets"]) == 1
    assert len(payload["selected_sources"]) == 1
    assert payload["selected_sources"][0]["dataset_id"] == "dataset_a"
    assert payload["quality_exclusions"] == [
        {
            "dataset_id": "dataset_b",
            "zarr_path": str(source_b),
            "reason": "missing_or_mismatched_eye_mask_profile",
        }
    ]
    merged = payload["datasets"][0]
    cmd = merged["export_command"]
    assert "--training-set-id" not in str(cmd)
    assert payload["datasets"][0]["export_status"] == "planned"
    assert payload["execution"] == {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0}
    out_config = manifest_path.with_name("prepared.yaml")
    assert payload["output_config_path"] == str(out_config)
    assert out_config.exists()
    config = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert list(config["datasets"].keys()) == ["prepared_merged"]
    assert config["training_params"]["label_mode"] == "lr"
    assert config["random_seed"] == 42


def test_prepare_eye_mask_from_registry_explicit_set_identity_with_out_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "eye_base.yaml")
    source_path = tmp_path / "source_a.zarr"
    manifest_path = tmp_path / "manual.manifest.json"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    db.close()

    _monkeypatch_selection(monkeypatch)
    _monkeypatch_export_not_called(monkeypatch)

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-manifest",
            str(manifest_path),
            "--set-name",
            "eye_ops",
            "--set-version",
            "3",
        ]
    )
    assert rc == 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["set_name"] == "eye_ops"
    assert payload["set_version"] == 3
    assert payload["set_id"] == "eye_mask_eye_ops_v003"

    cmd = payload["datasets"][0]["export_command"]
    assert "--training-set-id eye_mask_eye_ops_v003" in cmd
    assert payload["execution"] == {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0}
    out_config = manifest_path.with_name("manual.yaml")
    assert payload["output_config_path"] == str(out_config)
    assert out_config.exists()
    config = yaml.safe_load(out_config.read_text(encoding="utf-8"))
    assert list(config["datasets"].keys()) == ["eye_mask_eye_ops_v003_merged"]
    assert config["training_params"]["label_mode"] == "lr"


def test_prepare_eye_mask_from_registry_rejects_orchestration_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "eye_base.yaml")
    source_path = tmp_path / "source_a.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    db.close()

    _monkeypatch_selection(monkeypatch)

    with pytest.raises(SystemExit, match="prepare-only"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--aggregate-training-data-card",
            ]
        )
