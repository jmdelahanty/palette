"""Tests for subject-mask registry preflight manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_subject_mask_training_from_registry as wrapper


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


def _seed_subject_run(
    db: Registry,
    *,
    dataset_id: str,
    stage_group: str = "subject_mask_runs",
    run_name: str = "subject_masks_001",
    source_subject_mask_run: str | None = None,
    review_state: str = "approved",
    review_intended_use: str = "training",
    available_components: tuple[str, ...] = ("subject_body", "eyes_union"),
    label_schema_id: str = "subject_v1_union",
    created_utc: str = "2026-04-01T00:00:00+00:00",
) -> None:
    all_components = ("subject_body", "eyes_union", "swim_bladder")
    db.upsert_subject_mask_performance(
        dataset_id=dataset_id,
        stage_group=stage_group,
        run_name=run_name,
        run_created_utc=created_utc,
        recording_id=f"{dataset_id}_recording",
        zarr_use="analysis",
        subject_mask_method="subject_unet",
        label_schema_id=label_schema_id,
        source_crop_run="crop_001",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        source_subject_mask_run=source_subject_mask_run,
        source_subject_mask_method=None,
        run_semantics="subject_mask_training_source",
        probability_semantics=None,
        source_background_run=None,
        source_background_array=None,
        source_dish_mask_array=None,
        tuning_source=None,
        tuning_timestamp=None,
        total_rois=100,
        rows_with_any_mask=80,
        coverage_percent=80.0,
        duration_seconds=None,
        rois_per_second=None,
        available_component_count=len(available_components),
        available_components_json=json.dumps(list(available_components)),
        unavailable_components_json=json.dumps(
            [component for component in all_components if component not in available_components]
        ),
        component_review_states_json=json.dumps(
            {component: review_state for component in available_components}
        ),
        eye_component_mode="union",
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state=review_state,
        review_method="manual",
        review_intended_use=review_intended_use,
        review_reviewer="pytest",
        review_timestamp_utc=created_utc,
        lifecycle_state=review_state,
        lifecycle_reason=review_state,
        zarr_mtime_ns=123,
    )
    for component in all_components:
        available = int(component in available_components)
        db.upsert_subject_mask_component_quality(
            dataset_id=dataset_id,
            stage_group=stage_group,
            run_name=run_name,
            component_name=component,
            component_family="eyes" if component == "eyes_union" else "subject",
            run_created_utc=created_utc,
            recording_id=f"{dataset_id}_recording",
            zarr_use="analysis",
            subject_mask_method="subject_unet",
            label_schema_id=label_schema_id,
            eye_component_mode="union",
            source_subject_mask_run=source_subject_mask_run,
            available=available,
            review_state=review_state if available else None,
            review_method="manual" if available else None,
            review_intended_use=review_intended_use if available else None,
            review_reviewer="pytest" if available else None,
            review_timestamp_utc=created_utc if available else None,
            total_rois=100,
            rows_with_component_mask=75 if available else 0,
            rows_with_component_mask_rate=0.75 if available else 0.0,
            lifecycle_state=review_state if available else "na",
            lifecycle_reason=review_state if available else "component_unavailable",
            quality_updated_utc=created_utc,
            zarr_mtime_ns=123,
        )


def _write_base_config(path: Path) -> Path:
    payload = {
        "datasets": {
            "template": {
                "zarr_path": "/tmp/template_subject_masks.zarr",
            }
        },
        "names": ["subject_body", "eyes_union", "swim_bladder"],
        "nc": 3,
        "random_seed": 42,
        "num_workers": 0,
        "training_params": {
            "model": "unet-small",
            "epochs": 2,
            "batch_size": 4,
            "imgsz": 160,
            "lr0": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "patience": 10,
            "device": "cpu",
            "project": "runs/subject_masks",
            "label_schema_id": "subject_v1_union",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_prepare_subject_mask_from_registry_writes_manifest_and_config(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "subject_base.yaml")
    dataset_root = tmp_path / "datasets"
    source_path = tmp_path / "source_a.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    _seed_subject_run(db, dataset_id="dataset_a")
    db.close()

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-dir",
            str(dataset_root),
            "--set-name",
            "subject_ops",
            "--set-version",
            "1",
            "--require-component",
            "subject_body",
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
        ]
    )
    assert rc == 0

    set_id = "subject_mask_subject_ops_v001"
    manifest_path = dataset_root / f"{set_id}.manifest.json"
    config_path = dataset_root / f"{set_id}.yaml"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["task"] == "subject_masks"
    assert payload["set_id"] == set_id
    assert payload["source_stage_group"] == "subject_mask_runs"
    assert payload["subject_label_schema"] == "subject_v1_union"
    assert len(payload["selected_sources"]) == 1
    source = payload["selected_sources"][0]
    assert source["dataset_id"] == "dataset_a"
    assert source["source_stage_group"] == "subject_mask_runs"
    assert source["source_subject_mask_run"] == "subject_masks_001"
    assert source["source_crop_run"] == "crop_001"
    assert source["available_components"] == ["eyes_union", "subject_body"]
    assert source["canonical_latest_non_exportable_components"] == []

    merged = payload["datasets"][0]
    assert merged["dataset_id"] == f"{set_id}_merged"
    assert merged["out_zarr"] == str(dataset_root / "zarr" / f"{set_id}_merged.zarr")
    assert "fisheye.utils.run_subject_mask_training_pipeline" in merged["export_command"]
    assert "--subject-label-schema subject_v1_union" in merged["export_command"]
    assert "--training-set-id subject_mask_subject_ops_v001" in merged["export_command"]
    assert payload["execution"] == {"mode": "planned", "planned": 1, "succeeded": 0, "failed": 0}

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg_dataset = config["datasets"][f"{set_id}_merged"]
    assert cfg_dataset["zarr_path"] == merged["out_zarr"]
    assert cfg_dataset["crop_run"] == merged["run_name"]
    assert cfg_dataset["subject_mask_run"] == merged["run_name"]
    assert config["training_params"]["crop_run"] == merged["run_name"]
    assert config["training_params"]["subject_masks_run"] == merged["run_name"]
    assert config["training_params"]["label_schema_id"] == "subject_v1_union"
    assert config["random_seed"] == 123


def test_prepare_subject_mask_filters_by_component_review_state(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "subject_base.yaml")
    source_a = tmp_path / "source_a.zarr"
    source_b = tmp_path / "source_b.zarr"
    manifest_path = tmp_path / "prepared.manifest.json"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_a)
    _seed_dataset(db, dataset_id="dataset_b", zarr_path=source_b)
    _seed_subject_run(db, dataset_id="dataset_a", review_state="approved")
    _seed_subject_run(db, dataset_id="dataset_b", review_state="pending")
    db.close()

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-manifest",
            str(manifest_path),
            "--require-review-state",
            "approved",
            "--require-review-intended-use",
            "training",
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(payload["selected_sources"]) == 1
    assert payload["selected_sources"][0]["dataset_id"] == "dataset_a"
    assert payload["quality_exclusions"] == [
        {
            "dataset_id": "dataset_b",
            "zarr_path": str(source_b),
            "stage_group": "subject_mask_runs",
            "run_name": "subject_masks_001",
            "reason": "review_state_mismatch",
        }
    ]
    out_config = manifest_path.with_name("prepared.yaml")
    assert payload["output_config_path"] == str(out_config)
    assert out_config.exists()


def test_prepare_subject_mask_selects_coherent_refined_latest_source(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "subject_base.yaml")
    source_path = tmp_path / "source_a.zarr"
    manifest_path = tmp_path / "prepared.manifest.json"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    _seed_subject_run(
        db,
        dataset_id="dataset_a",
        stage_group="subject_mask_runs",
        run_name="subject_masks_001",
        created_utc="2026-04-01T00:00:00+00:00",
    )
    _seed_subject_run(
        db,
        dataset_id="dataset_a",
        stage_group="refined_subject_masks_runs",
        run_name="refined_subject_masks_001",
        source_subject_mask_run="subject_masks_001",
        available_components=("subject_body", "eyes_union", "swim_bladder"),
        created_utc="2026-04-02T00:00:00+00:00",
    )
    db.close()

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-manifest",
            str(manifest_path),
            "--require-review-state",
            "approved",
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = payload["selected_sources"][0]

    assert payload["source_stage_group"] == "refined_subject_masks_runs"
    assert source["source_stage_group"] == "refined_subject_masks_runs"
    assert source["source_subject_mask_run"] == "refined_subject_masks_001"
    assert source["canonical_latest_non_exportable_components"] == []
    selected_latest = source["canonical_latest_selected_components"]
    assert selected_latest
    assert {row["stage_group"] for row in selected_latest} == {"refined_subject_masks_runs"}
    assert {row["run_name"] for row in selected_latest} == {"refined_subject_masks_001"}
    assert source["canonical_latest_requires_assembly"] is False


def test_prepare_subject_mask_flags_split_refined_latest_for_assembly(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "subject_base.yaml")
    source_path = tmp_path / "source_a.zarr"
    manifest_path = tmp_path / "prepared.manifest.json"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=source_path)
    _seed_subject_run(
        db,
        dataset_id="dataset_a",
        stage_group="subject_mask_runs",
        run_name="subject_masks_001",
        created_utc="2026-04-01T00:00:00+00:00",
    )
    _seed_subject_run(
        db,
        dataset_id="dataset_a",
        stage_group="refined_subject_masks_runs",
        run_name="refined_subject_body_001",
        source_subject_mask_run="subject_masks_001",
        available_components=("subject_body",),
        created_utc="2026-04-02T00:00:00+00:00",
    )
    _seed_subject_run(
        db,
        dataset_id="dataset_a",
        stage_group="refined_subject_masks_runs",
        run_name="refined_subject_eyes_001",
        source_subject_mask_run="subject_masks_001",
        available_components=("eyes_union",),
        created_utc="2026-04-03T00:00:00+00:00",
    )
    db.close()

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--base-config",
            str(base_config_path),
            "--out-manifest",
            str(manifest_path),
            "--require-review-state",
            "approved",
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = payload["selected_sources"][0]

    assert source["source_stage_group"] == "subject_mask_runs"
    assert source["source_subject_mask_run"] == "subject_masks_001"
    assert source["canonical_latest_requires_assembly"] is True
    latest_runs = {
        (row["component_name"], row["stage_group"], row["run_name"])
        for row in source["canonical_latest_components"]
        if row["available"] == 1
    }
    assert ("subject_body", "refined_subject_masks_runs", "refined_subject_body_001") in latest_runs
    assert ("eyes_union", "refined_subject_masks_runs", "refined_subject_eyes_001") in latest_runs


def test_prepare_subject_mask_rejects_unknown_component(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    base_config_path = _write_base_config(tmp_path / "subject_base.yaml")
    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", zarr_path=tmp_path / "source_a.zarr")
    _seed_subject_run(db, dataset_id="dataset_a")
    db.close()

    with pytest.raises(SystemExit, match="Unsupported component"):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--base-config",
                str(base_config_path),
                "--require-component",
                "not_a_component",
            ]
        )
