from __future__ import annotations

import json
from pathlib import Path

from fisheye.registry.db import Registry


def _subject_mask_final_metrics() -> dict[str, object]:
    summary = {
        "label_schema_id": "subject_v1_union",
        "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
        "coverage_class": "dense_all_components",
        "component_groups": ["body", "eyes", "swim_bladder"],
        "component_coverage_key": "body+eyes+swim_bladder",
        "contains_only_eye_masks": False,
        "available_labels": ["subject_body", "eyes_union", "swim_bladder"],
        "missing_labels": [],
        "supervised_row_counts": {
            "subject_body": 3153,
            "eyes_union": 3153,
            "swim_bladder": 3153,
        },
        "positive_row_counts": {
            "subject_body": 3153,
            "eyes_union": 3153,
            "swim_bladder": 3153,
        },
        "negative_row_counts": {
            "subject_body": 0,
            "eyes_union": 0,
            "swim_bladder": 0,
        },
        "unsupervised_row_counts": {
            "subject_body": 0,
            "eyes_union": 0,
            "swim_bladder": 0,
        },
    }
    return {
        "stage": "completed",
        "best_val_dice": 0.947,
        "best_epoch": 82,
        "epochs": 100,
        "train_samples": 2522,
        "val_samples": 631,
        "label_schema_id": "subject_v1_union",
        "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
        "subject_mask_model_summary": summary,
    }


def test_subject_mask_training_run_populates_model_discovery_metadata(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "best_model.pt"
    metrics_path = tmp_path / "training_history.json"
    model_path.write_text("weights", encoding="utf-8")
    metrics_path.write_text("[]", encoding="utf-8")

    registry.record_training_run(
        run_id="subject_masks_union_all_components_v001",
        set_id="subject_mask_training_set_v001",
        task_type="subject_masks",
        config_path=None,
        manifest_path=None,
        model_path=model_path,
        metrics_path=metrics_path,
        model_sha256="model_sha",
        metrics_sha256="metrics_sha",
        status="success",
        final_metrics=_subject_mask_final_metrics(),
    )

    row = registry.conn.execute(
        """
        SELECT
            task_type,
            label_schema_id,
            coverage_class,
            component_coverage_key,
            mask_labels_json,
            component_groups_json,
            best_metric_name,
            best_metric_value,
            best_epoch,
            metadata_json
        FROM training_models
        WHERE run_id = ?;
        """,
        ("subject_masks_union_all_components_v001",),
    ).fetchone()
    assert row is not None
    assert row["task_type"] == "subject_masks"
    assert row["label_schema_id"] == "subject_v1_union"
    assert row["coverage_class"] == "dense_all_components"
    assert row["component_coverage_key"] == "body+eyes+swim_bladder"
    assert json.loads(row["mask_labels_json"]) == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert json.loads(row["component_groups_json"]) == ["body", "eyes", "swim_bladder"]
    assert row["best_metric_name"] == "best_val_dice"
    assert row["best_metric_value"] == 0.947
    assert row["best_epoch"] == 82

    metadata = json.loads(row["metadata_json"])
    assert metadata["source"] == "training_runs"
    assert metadata["task_type"] == "subject_masks"
    assert metadata["train_samples"] == 2522
    assert metadata["val_samples"] == 631
    assert metadata["subject_mask_model_summary"]["component_coverage_key"] == (
        "body+eyes+swim_bladder"
    )

    view_row = registry.conn.execute(
        """
        SELECT run_id, model_path, best_metric_value, component_coverage_key
        FROM subject_mask_training_models
        WHERE run_id = ?;
        """,
        ("subject_masks_union_all_components_v001",),
    ).fetchone()
    assert view_row is not None
    assert view_row["model_path"] == str(model_path)
    assert view_row["best_metric_value"] == 0.947
    assert view_row["component_coverage_key"] == "body+eyes+swim_bladder"
    registry.close()


def test_subject_mask_model_discovery_backfill_preserves_existing_metadata(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="subject_mask_backfill_v001",
        set_id="subject_mask_training_set_v001",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics=_subject_mask_final_metrics(),
    )
    registry.conn.execute(
        """
        UPDATE training_models
        SET task_type = NULL,
            label_schema_id = NULL,
            coverage_class = NULL,
            component_coverage_key = NULL,
            mask_labels_json = NULL,
            component_groups_json = NULL,
            best_metric_name = NULL,
            best_metric_value = NULL,
            best_epoch = NULL,
            metadata_json = ?
        WHERE run_id = ?;
        """,
        (json.dumps({"source": "training_runs", "note": "legacy"}), "subject_mask_backfill_v001"),
    )

    registry._migration_040_subject_mask_training_model_discovery()

    row = registry.conn.execute(
        """
        SELECT task_type, label_schema_id, component_coverage_key, metadata_json
        FROM training_models
        WHERE run_id = ?;
        """,
        ("subject_mask_backfill_v001",),
    ).fetchone()
    assert row is not None
    assert row["task_type"] == "subject_masks"
    assert row["label_schema_id"] == "subject_v1_union"
    assert row["component_coverage_key"] == "body+eyes+swim_bladder"
    metadata = json.loads(row["metadata_json"])
    assert metadata["note"] == "legacy"
    assert metadata["subject_mask_model_summary"]["coverage_class"] == "dense_all_components"
    registry.close()


def test_subject_mask_model_discovery_derives_coverage_from_labels(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="subject_mask_legacy_summary_v001",
        set_id="subject_mask_training_set_v001",
        task_type="subject_masks",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={
            "stage": "completed",
            "best_val_dice": 0.12,
            "best_epoch": 20,
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
        },
    )

    row = registry.conn.execute(
        """
        SELECT coverage_class, component_coverage_key, component_groups_json, metadata_json
        FROM training_models
        WHERE run_id = ?;
        """,
        ("subject_mask_legacy_summary_v001",),
    ).fetchone()
    assert row is not None
    assert row["coverage_class"] == "dense_all_components"
    assert row["component_coverage_key"] == "body+eyes+swim_bladder"
    assert json.loads(row["component_groups_json"]) == ["body", "eyes", "swim_bladder"]
    metadata = json.loads(row["metadata_json"])
    assert metadata["available_labels"] == ["subject_body", "eyes_union", "swim_bladder"]
    assert metadata["missing_labels"] == []
    registry.close()
