"""Unit tests for registry maintenance helpers."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.maintenance import (
    _backfill_keypoint_quality,
    _backfill_model_tables,
    _check_registry_integrity,
    _collect_empty_training_set_candidates,
    _collect_failed_run_candidates,
    _collect_invalid_dataset_candidates,
    _delete_training_set_ids,
    _delete_training_run_ids,
    _delete_dataset_ids,
    _is_nested_zarr_subpath,
    _normalize_status_values,
)


def _create_quality_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "quality_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["detection_source_type"] = "filtered"
    crop.create_array("roi_images", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    kp_parent = root.create_group("keypoints_runs")
    kp = kp_parent.create_group("kp_001")
    kp.attrs["method"] = "traditional_pose"
    kp.attrs["source_crop_run"] = "crop_001"
    kp.attrs["success_rate"] = 0.75
    kp.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), chunks=(1, 3, 2))
    refined_parent = root.create_group("refined_keypoints_runs")
    refined = refined_parent.create_group("refined_001")
    refined.attrs["source_keypoints_run"] = "kp_001"
    refined.attrs["created_utc"] = "2026-02-08T00:00:00+00:00"
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp_utc": "2026-02-08T00:00:00+00:00",
    }
    refined.create_array(
        "usable_keypoints",
        data=np.array([True, True, True, False], dtype=np.bool_),
        chunks=(4,),
    )


def test_is_nested_zarr_subpath() -> None:
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs")
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs/run_01")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr/subset.zarr")


def test_collect_and_delete_invalid_candidates(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    good = tmp_path / "good.zarr"
    nested = tmp_path / "good.zarr" / "detect_runs"
    nested_zarr = tmp_path / "good.zarr" / "subset.zarr"
    stale = tmp_path / "missing.zarr"

    registry.upsert_dataset("good", session_uuid="good", zarr_path=good)
    registry.upsert_dataset("nested", session_uuid=None, zarr_path=nested)
    registry.upsert_dataset("nested_zarr", session_uuid=None, zarr_path=nested_zarr)
    registry.upsert_dataset("stale", session_uuid=None, zarr_path=stale)
    registry.conn.execute("UPDATE datasets SET status = 'missing' WHERE dataset_id = 'stale';")
    registry.conn.commit()

    candidates = _collect_invalid_dataset_candidates(registry)
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"nested", "stale"}
    assert by_id["nested"].reasons == ("nested_zarr_subpath",)
    assert by_id["stale"].reasons == ("status_missing",)

    # Dry run must not delete.
    assert _delete_dataset_ids(registry, ["nested", "stale"], dry_run=True) == 0
    still_present = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert still_present == {"good", "nested", "nested_zarr", "stale"}

    deleted = _delete_dataset_ids(registry, ["nested", "stale"], dry_run=False)
    assert deleted == 2
    remaining = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert remaining == {"good", "nested_zarr"}
    registry.close()


def test_collect_candidates_can_infer_missing_without_status(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("missing_active", session_uuid=None, zarr_path=tmp_path / "not_there.zarr")
    nested_path = tmp_path / "recording.zarr" / "detect_runs"
    nested_path.mkdir(parents=True)
    (nested_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")
    registry.upsert_dataset(
        "nested_active",
        session_uuid=None,
        zarr_path=nested_path,
    )

    candidates = _collect_invalid_dataset_candidates(
        registry,
        include_missing_scan=True,
    )
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"missing_active", "nested_active"}
    assert by_id["missing_active"].reasons == ("status_missing",)
    assert by_id["nested_active"].reasons == ("nested_zarr_subpath",)
    registry.close()


def test_collect_and_delete_failed_training_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="run_failed",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="failed",
    )
    registry.record_training_run(
        run_id="run_failed_caps",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="FAILED",
    )
    registry.record_training_run(
        run_id="run_success",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_in_progress",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
    )
    registry.record_model_export(run_id="run_failed", export_type="onnx", path=tmp_path / "run_failed.onnx")
    registry.record_model_export(run_id="run_success", export_type="onnx", path=tmp_path / "run_success.onnx")

    status_values = _normalize_status_values(["failed"])
    candidates = _collect_failed_run_candidates(registry, status_values=status_values)
    candidate_ids = {candidate.run_id for candidate in candidates}
    assert candidate_ids == {"run_failed", "run_failed_caps"}

    # Dry run must not delete.
    assert _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert still_present == {"run_failed", "run_failed_caps", "run_success", "run_in_progress"}

    deleted = _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert remaining == {"run_success", "run_in_progress"}

    export_rows = registry.conn.execute(
        "SELECT run_id FROM onnx_models ORDER BY run_id;"
    ).fetchall()
    assert [row["run_id"] for row in export_rows] == ["run_success"]
    registry.close()


def test_collect_and_delete_empty_training_sets(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="set_empty_1",
        name="empty one",
        query_filter=None,
        dataset_ids=["dataset_a"],
    )
    registry.upsert_training_set(
        set_id="set_linked",
        name="linked",
        query_filter=None,
        dataset_ids=["dataset_b"],
    )
    registry.upsert_training_set(
        set_id="set_empty_2",
        name="empty two",
        query_filter=None,
        dataset_ids=[],
    )
    registry.record_training_run(
        run_id="run_linked",
        set_id="set_linked",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_unlinked",
        set_id=None,
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )

    candidates = _collect_empty_training_set_candidates(registry)
    candidate_ids = {candidate.set_id for candidate in candidates}
    assert candidate_ids == {"set_empty_1", "set_empty_2"}

    assert _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert still_present == {"set_empty_1", "set_empty_2", "set_linked"}

    deleted = _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert remaining == {"set_linked"}
    registry.close()


def test_backfill_model_tables_from_legacy_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    trt = tmp_path / "best_fp16.engine"
    config.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")

    registry.record_training_run(
        run_id="run_a",
        set_id="set_a",
        config_path=config,
        manifest_path=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    # Legacy model_exports rows that backfill reads from.
    registry.conn.execute(
        """
        INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
        VALUES (?, 'onnx', ?, NULL, ?, datetime('now'));
        """,
        ("run_a", str(onnx), '{"sha256":"onnx_sha","manifest_sha256":"onnx_manifest_sha"}'),
    )
    registry.conn.execute(
        """
        INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
        VALUES (?, 'tensorrt', ?, NULL, ?, datetime('now'));
        """,
        ("run_a", str(trt), '{"sha256":"trt_sha","manifest_sha256":"trt_manifest_sha","precision":"fp16"}'),
    )
    registry.conn.commit()

    # Simulate pre-migration registry by removing new tables.
    registry.conn.execute("DELETE FROM detection_models;")
    registry.conn.execute("DELETE FROM onnx_models;")
    registry.conn.execute("DELETE FROM tensorrt_models;")
    registry.conn.commit()

    dry = _backfill_model_tables(registry, dry_run=True)
    assert dry["detection_missing"] == 1
    assert dry["onnx_missing"] == 1
    assert dry["tensorrt_missing"] == 1
    assert dry["detection_inserted"] == 0
    assert dry["onnx_inserted"] == 0
    assert dry["tensorrt_inserted"] == 0

    applied = _backfill_model_tables(registry, dry_run=False)
    assert applied["detection_inserted"] == 1
    assert applied["onnx_inserted"] == 1
    assert applied["tensorrt_inserted"] == 1

    detection_count = registry.conn.execute("SELECT COUNT(*) AS n FROM detection_models;").fetchone()["n"]
    onnx_count = registry.conn.execute("SELECT COUNT(*) AS n FROM onnx_models;").fetchone()["n"]
    trt_count = registry.conn.execute("SELECT COUNT(*) AS n FROM tensorrt_models;").fetchone()["n"]
    assert detection_count == 1
    assert onnx_count == 1
    assert trt_count == 1

    # Idempotent on repeat.
    repeat = _backfill_model_tables(registry, dry_run=False)
    assert repeat["detection_inserted"] == 0
    assert repeat["onnx_inserted"] == 0
    assert repeat["tensorrt_inserted"] == 0
    registry.close()


def test_backfill_keypoint_quality_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "quality_sample.zarr"
    _create_quality_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute("DELETE FROM keypoint_quality WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_keypoint_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_keypoint_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        "SELECT review_state, review_intended_use, usable_keypoints_rate FROM keypoint_quality_current WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert row["review_state"] == "approved"
    assert row["review_intended_use"] == "training"
    assert float(row["usable_keypoints_rate"]) == 0.75
    registry.close()


def test_refresh_keypoint_quality_deletes_stale_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "quality_refresh.zarr"
    _create_quality_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute(
        """
        INSERT INTO keypoint_quality (
            dataset_id, refined_run, source_keypoint_run, quality_updated_utc
        ) VALUES (?, 'stale_refined', 'kp_old', datetime('now'));
        """,
        (dataset_id,),
    )
    registry.conn.commit()

    summary = _backfill_keypoint_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert summary["rows_deleted"] >= 1
    stale = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_quality WHERE dataset_id = ? AND refined_run = 'stale_refined';",
        (dataset_id,),
    ).fetchone()["n"]
    assert stale == 0
    registry.close()


def test_check_registry_integrity_passes_for_valid_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    onnx_manifest = tmp_path / "best.onnx.manifest.json"
    trt = tmp_path / "best_fp16.engine"
    trt_manifest = tmp_path / "best_fp16.tensorrt.manifest.json"
    cfg.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    onnx_manifest.write_text("{}", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")
    trt_manifest.write_text("{}", encoding="utf-8")

    registry.record_training_run(
        run_id="run_ok",
        set_id="set_ok",
        config_path=cfg,
        manifest_path=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_ok",
        export_type="onnx",
        path=onnx,
        manifest_path=onnx_manifest,
        metadata={"sha256": "onnx_sha", "manifest_sha256": "onnx_manifest_sha"},
    )
    registry.record_model_export(
        run_id="run_ok",
        export_type="tensorrt",
        path=trt,
        manifest_path=trt_manifest,
        metadata={"sha256": "trt_sha", "manifest_sha256": "trt_manifest_sha", "precision": "fp16"},
    )

    issues = _check_registry_integrity(registry)
    assert issues == []
    registry.close()


def test_check_registry_integrity_reports_missing_detection_model_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cfg", encoding="utf-8")
    registry.record_training_run(
        run_id="run_missing_dm",
        set_id="set_a",
        config_path=cfg,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "start"},
    )
    # Simulate inconsistent state.
    registry.conn.execute("DELETE FROM detection_models WHERE run_id = 'run_missing_dm';")
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    assert any(issue.code == "missing_detection_model_row" and issue.run_id == "run_missing_dm" for issue in issues)
    registry.close()


def test_check_registry_integrity_reports_missing_artifact_files(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cfg", encoding="utf-8")
    model = tmp_path / "missing_best.pt"
    metrics = tmp_path / "missing_results.csv"
    onnx = tmp_path / "missing_best.onnx"
    trt = tmp_path / "missing_best_fp16.engine"

    registry.record_training_run(
        run_id="run_missing_files",
        set_id="set_missing",
        config_path=cfg,
        manifest_path=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_missing_files",
        export_type="onnx",
        path=onnx,
        metadata={"sha256": "onnx_sha"},
    )
    registry.record_model_export(
        run_id="run_missing_files",
        export_type="tensorrt",
        path=trt,
        metadata={"sha256": "trt_sha", "precision": "fp16"},
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "missing_model_file" in codes
    assert "missing_metrics_file" in codes
    assert "onnx_file_missing" in codes
    assert "trt_file_missing" in codes
    registry.close()


def test_check_registry_integrity_reports_trt_plugin_contract_mismatch(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    onnx_manifest = tmp_path / "best.onnx.manifest.json"
    trt = tmp_path / "best_fp16.engine"
    trt_manifest = tmp_path / "best_fp16.tensorrt.manifest.json"
    cfg.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    onnx_manifest.write_text("{}", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")
    trt_manifest.write_text("{}", encoding="utf-8")

    registry.record_training_run(
        run_id="run_plugin_mismatch",
        set_id="set_ok",
        config_path=cfg,
        manifest_path=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_plugin_mismatch",
        export_type="onnx",
        path=onnx,
        manifest_path=onnx_manifest,
        metadata={
            "sha256": "onnx_sha",
            "manifest_sha256": "onnx_manifest_sha",
            "requires_plugins": True,
            "plugin_ops": ["TRT::EfficientNMS_TRT"],
            "plugin_versions": {"TRT::EfficientNMS_TRT": "1"},
        },
    )
    registry.record_model_export(
        run_id="run_plugin_mismatch",
        export_type="tensorrt",
        path=trt,
        manifest_path=trt_manifest,
        metadata={
            "sha256": "trt_sha",
            "manifest_sha256": "trt_manifest_sha",
            "precision": "fp16",
            # Deliberately incomplete to trigger integrity findings.
            "requires_plugins": True,
        },
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "trt_plugins_missing_ops" in codes
    assert "trt_plugin_contract_mismatch" in codes
    registry.close()
