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


def test_training_model_input_shape_fields_from_final_metrics(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "best.pt"
    model_path.write_text("weights", encoding="utf-8")

    registry.record_training_run(
        run_id="detect_run_640",
        set_id="detect_set",
        task_type="detect",
        config_path=None,
        manifest_path=None,
        model_path=model_path,
        metrics_path=None,
        model_sha256="model_sha",
        status="success",
        final_metrics={
            "stage": "completed",
            "imgsz_h": 640,
            "imgsz_w": 640,
        },
    )

    row = registry.conn.execute(
        """
        SELECT
            input_shape,
            input_layout,
            input_channels,
            img_h,
            img_w,
            max_batch,
            dynamic_shapes,
            input_dtype,
            input_color_space,
            input_shape_source,
            input_shape_status
        FROM training_models
        WHERE run_id = ?;
        """,
        ("detect_run_640",),
    ).fetchone()
    assert row is not None
    assert json.loads(row["input_shape"]) == [1, 3, 640, 640]
    assert row["input_layout"] == "NCHW"
    assert row["input_channels"] == 3
    assert row["img_h"] == 640
    assert row["img_w"] == 640
    assert row["max_batch"] == 1
    assert row["dynamic_shapes"] == 0
    assert row["input_dtype"] == "float32"
    assert row["input_color_space"] == "rgb"
    assert row["input_shape_source"] == "final_metrics.imgsz_h_imgsz_w"
    assert row["input_shape_status"] == "inferred_from_imgsz"

    view_row = registry.conn.execute(
        """
        SELECT artifact_kind, task_type, artifact_path, input_shape, img_h, img_w
        FROM model_input_shapes
        WHERE run_id = ? AND artifact_kind = 'training';
        """,
        ("detect_run_640",),
    ).fetchone()
    assert view_row is not None
    assert view_row["task_type"] == "detect"
    assert view_row["artifact_path"] == str(model_path)
    assert json.loads(view_row["input_shape"]) == [1, 3, 640, 640]
    assert view_row["img_h"] == 640
    assert view_row["img_w"] == 640
    registry.close()


def test_model_input_shapes_view_keeps_artifacts_separate(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    model_path = tmp_path / "best.pt"
    onnx_path = tmp_path / "best.onnx"
    trt_path = tmp_path / "best_fp16.engine"
    model_path.write_text("weights", encoding="utf-8")
    onnx_path.write_text("onnx", encoding="utf-8")
    trt_path.write_text("trt", encoding="utf-8")

    registry.record_training_run(
        run_id="detect_run_exports",
        set_id="detect_set",
        task_type="detect",
        config_path=None,
        manifest_path=None,
        model_path=model_path,
        metrics_path=None,
        status="success",
        final_metrics={"effective_imgsz": [640, 640]},
    )
    registry.record_onnx_model(
        run_id="detect_run_exports",
        set_id="detect_set",
        detection_model_run_id="detect_run_exports",
        path=onnx_path,
        sha256="onnx_sha",
        manifest_path=None,
        manifest_sha256=None,
        input_shape="[1, 3, 640, 640]",
        img_h=640,
        img_w=640,
        max_batch=1,
        dynamic_shapes=False,
    )
    registry.record_tensorrt_model(
        run_id="detect_run_exports",
        set_id="detect_set",
        detection_model_run_id="detect_run_exports",
        onnx_run_id="detect_run_exports",
        precision="fp16",
        path=trt_path,
        sha256="trt_sha",
        manifest_path=None,
        manifest_sha256=None,
        input_shape="[1, 3, 640, 640]",
        img_h=640,
        img_w=640,
        max_batch=8,
        dynamic_shapes=False,
    )

    rows = registry.conn.execute(
        """
        SELECT artifact_kind, artifact_precision, artifact_path, max_batch
        FROM model_input_shapes
        WHERE run_id = ?
        ORDER BY artifact_kind;
        """,
        ("detect_run_exports",),
    ).fetchall()
    assert [row["artifact_kind"] for row in rows] == ["onnx", "tensorrt", "training"]
    assert {row["artifact_path"] for row in rows} == {
        str(model_path),
        str(onnx_path),
        str(trt_path),
    }
    assert [row["artifact_precision"] for row in rows if row["artifact_kind"] == "tensorrt"] == ["fp16"]
    assert [row["max_batch"] for row in rows if row["artifact_kind"] == "tensorrt"] == [8]
    registry.close()


def test_model_input_shape_migration_backfills_and_flags_export_conflict(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="detect_conflict",
        set_id="detect_set",
        task_type="detect",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={"imgsz_h": 640, "imgsz_w": 640},
    )
    registry.record_onnx_model(
        run_id="detect_conflict",
        set_id="detect_set",
        detection_model_run_id="detect_conflict",
        path=tmp_path / "conflict.onnx",
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        input_shape="[1, 3, 320, 320]",
        img_h=320,
        img_w=320,
    )
    registry.conn.execute(
        """
        UPDATE training_models
        SET input_shape = NULL,
            input_layout = NULL,
            input_channels = NULL,
            img_h = NULL,
            img_w = NULL,
            max_batch = NULL,
            dynamic_shapes = NULL,
            input_dtype = NULL,
            input_color_space = NULL,
            input_shape_source = NULL,
            input_shape_status = NULL
        WHERE run_id = ?;
        """,
        ("detect_conflict",),
    )

    registry._migration_049_model_input_shape_registry()

    row = registry.conn.execute(
        """
        SELECT input_shape, img_h, img_w, input_shape_status, metadata_json
        FROM training_models
        WHERE run_id = ?;
        """,
        ("detect_conflict",),
    ).fetchone()
    assert row is not None
    assert json.loads(row["input_shape"]) == [1, 3, 640, 640]
    assert row["img_h"] == 640
    assert row["img_w"] == 640
    assert row["input_shape_status"] == "conflict"
    metadata = json.loads(row["metadata_json"])
    assert metadata["input_shape_conflict"]["export"]["img_h"] == 320
    assert metadata["input_shape_conflict"]["training"]["img_h"] == 640
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
