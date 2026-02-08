"""Unit tests for training registry status rendering helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.check_training_registry import (
    KEYPOINT_GATE_MIN_RATE,
    _fetch_exports,
    _keypoint_exclusion_reason,
    _metrics_summary_from_json,
    _onnx_plugin_details,
    _run_id_style,
    _sum_manifest_rois,
    _status_with_details,
    _summarize_keypoint_quality_rows,
)


def test_metrics_summary_from_json_prefers_common_fields() -> None:
    payload = (
        '{"mAP50": 0.95231, "mAP50_95": 0.72111, '
        '"precision": 0.91001, "recall": 0.83002}'
    )
    assert (
        _metrics_summary_from_json(payload)
        == "mAP50=0.952, mAP50-95=0.721, P=0.910, R=0.830"
    )


def test_metrics_summary_from_json_supports_alt_keys() -> None:
    payload = '{"map50": 0.9, "map50-95": 0.6}'
    assert _metrics_summary_from_json(payload) == "mAP50=0.900, mAP50-95=0.600"


def test_status_with_details_appends_suffix_only_for_ok() -> None:
    assert _status_with_details(True, "mAP50=0.900", rich=False) == "OK (mAP50=0.900)"
    assert _status_with_details(False, "mAP50=0.900", rich=False) == "MISS"
    assert _status_with_details(None, "mAP50=0.900", rich=False) == "—"


def test_fetch_exports_prefers_new_format_tables(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("cfg", encoding="utf-8")
    run_id = "run_exports_001"
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "start"},
    )

    onnx_new = tmp_path / "new.onnx"
    trt_new = tmp_path / "new_fp16.engine"
    onnx_new.write_text("onnx", encoding="utf-8")
    trt_new.write_text("trt", encoding="utf-8")
    registry.record_onnx_model(
        run_id=run_id,
        set_id="detect_set_v001",
        detection_model_run_id=run_id,
        path=onnx_new,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        requires_plugins=True,
        plugin_ops=["TRT::EfficientNMS_TRT"],
        plugin_versions={"TRT::EfficientNMS_TRT": "1"},
        metadata=None,
    )
    registry.record_tensorrt_model(
        run_id=run_id,
        set_id="detect_set_v001",
        detection_model_run_id=run_id,
        onnx_run_id=run_id,
        precision="fp16",
        path=trt_new,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        metadata=None,
    )

    # Legacy table intentionally points elsewhere; reader ignores legacy fallback.
    registry.conn.execute(
        "INSERT INTO model_exports (run_id, export_type, path, created_utc) VALUES (?, 'onnx', ?, datetime('now')) "
        "ON CONFLICT(run_id, export_type) DO UPDATE SET path=excluded.path;",
        (run_id, str(tmp_path / "legacy.onnx")),
    )
    registry.conn.execute(
        "INSERT INTO model_exports (run_id, export_type, path, created_utc) VALUES (?, 'tensorrt', ?, datetime('now')) "
        "ON CONFLICT(run_id, export_type) DO UPDATE SET path=excluded.path;",
        (run_id, str(tmp_path / "legacy.engine")),
    )
    registry.conn.commit()

    exports = _fetch_exports(registry, run_id)
    assert exports["onnx_path"] == str(onnx_new)
    assert exports["trt_path"] == str(trt_new)
    assert exports["onnx_source"] == "new"
    assert exports["trt_source"] == "new"
    assert exports["onnx_requires_plugins"] == 1
    assert exports["onnx_plugin_ops_json"] == '["TRT::EfficientNMS_TRT"]'
    assert exports["onnx_plugin_versions_json"] == '{"TRT::EfficientNMS_TRT": "1"}'
    registry.close()


def test_fetch_exports_does_not_fallback_to_legacy_sources(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("cfg", encoding="utf-8")
    run_id = "run_exports_mixed_001"
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_set_v001",
        config_path=config_path,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "start"},
    )

    onnx_new = tmp_path / "new.onnx"
    onnx_new.write_text("onnx", encoding="utf-8")
    registry.record_onnx_model(
        run_id=run_id,
        set_id="detect_set_v001",
        detection_model_run_id=run_id,
        path=onnx_new,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        metadata=None,
    )
    registry.conn.execute(
        "INSERT INTO model_exports (run_id, export_type, path, created_utc) VALUES (?, 'tensorrt', ?, datetime('now')) "
        "ON CONFLICT(run_id, export_type) DO UPDATE SET path=excluded.path;",
        (run_id, str(tmp_path / "legacy.engine")),
    )
    registry.conn.commit()

    exports = _fetch_exports(registry, run_id)
    assert exports["onnx_path"] == str(onnx_new)
    assert exports["onnx_source"] == "new"
    assert exports["onnx_requires_plugins"] is None
    assert exports["onnx_plugin_ops_json"] is None
    assert exports["onnx_plugin_versions_json"] is None
    assert exports["trt_path"] is None
    assert exports["trt_source"] is None
    registry.close()


def test_onnx_plugin_details_formats_ops_and_versions() -> None:
    assert (
        _onnx_plugin_details(
            1,
            '["TRT::EfficientNMS_TRT"]',
            '{"TRT::EfficientNMS_TRT":"1"}',
        )
        == "plugins=TRT::EfficientNMS_TRT@1"
    )
    assert _onnx_plugin_details(0, None, None) is None


def test_keypoint_quality_summary_counts_and_buckets() -> None:
    rows = [
        {
            "dataset_id": "a",
            "review_state": "approved",
            "review_intended_use": "training",
            "usable_keypoints_rate": 0.9,
        },
        {
            "dataset_id": "b",
            "review_state": None,
            "review_intended_use": "training",
            "usable_keypoints_rate": 0.95,
        },
        {
            "dataset_id": "c",
            "review_state": "pending",
            "review_intended_use": "training",
            "usable_keypoints_rate": 0.95,
        },
        {
            "dataset_id": "d",
            "review_state": "approved",
            "review_intended_use": "training",
            "usable_keypoints_rate": KEYPOINT_GATE_MIN_RATE - 0.01,
        },
        {
            "dataset_id": "e",
            "review_state": "approved",
            "review_intended_use": "training",
            "usable_keypoints_rate": None,
        },
    ]
    summary = _summarize_keypoint_quality_rows(rows)
    assert summary.total_rows == 5
    assert summary.passing_rows == 1
    assert summary.excluded_rows == 4
    assert summary.exclusion_reasons == {
        "low rate": 1,
        "missing rate": 1,
        "missing review": 1,
        "wrong state/use": 1,
    }


def test_keypoint_exclusion_reason_precedence() -> None:
    assert (
        _keypoint_exclusion_reason(
            {
                "review_state": None,
                "review_intended_use": "training",
                "usable_keypoints_rate": 0.2,
            }
        )
        == "missing review"
    )
    assert (
        _keypoint_exclusion_reason(
            {
                "review_state": "pending",
                "review_intended_use": "training",
                "usable_keypoints_rate": 0.95,
            }
        )
        == "wrong state/use"
    )
    assert (
        _keypoint_exclusion_reason(
            {
                "review_state": "approved",
                "review_intended_use": "training",
                "usable_keypoints_rate": None,
            }
        )
        == "missing rate"
    )


def test_run_id_style_contract_and_legacy() -> None:
    assert (
        _run_id_style("omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb")
        == "contract"
    )
    assert _run_id_style("cedar_shadow_pose_20260208-030800_505915") == "legacy"


def test_sum_manifest_rois_supports_detect_and_pose_fields(tmp_path: Path) -> None:
    detect_manifest = tmp_path / "detect.manifest.json"
    detect_manifest.write_text(
        '{"datasets":[{"total_bboxes":10},{"total_bboxes":5}]}',
        encoding="utf-8",
    )
    assert _sum_manifest_rois(str(detect_manifest)) == 15

    pose_manifest = tmp_path / "pose.manifest.json"
    pose_manifest.write_text(
        '{"datasets":[{"keypoints_total":7},{"keypoints_total":4}]}',
        encoding="utf-8",
    )
    assert _sum_manifest_rois(str(pose_manifest)) == 11
