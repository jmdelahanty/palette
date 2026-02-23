"""Unit tests for training registry status rendering helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.check_training_registry import (
    KEYPOINT_GATE_MIN_RATE,
    RecordingVocabRow,
    _fetch_exports,
    _load_onnx_rows,
    _load_tensorrt_rows,
    _format_recording_vocab_inline,
    _group_recording_vocab_rows,
    _keypoint_exclusion_reason,
    main as check_training_registry_main,
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
        skeleton_id=None,
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
        skeleton_id=None,
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
        skeleton_id=None,
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
        skeleton_id=None,
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
        skeleton_id=None,
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


def test_onnx_tensorrt_rows_expose_nms_columns(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("cfg", encoding="utf-8")
    run_id = "run_nms_view_001"
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_set_v001",
        config_path=config_path,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={"mAP50": 0.9},
    )

    onnx_new = tmp_path / "nms.onnx"
    trt_new = tmp_path / "nms_fp16.engine"
    onnx_new.write_text("onnx", encoding="utf-8")
    trt_new.write_text("trt", encoding="utf-8")

    registry.record_onnx_model(
        run_id=run_id,
        set_id="detect_set_v001",
        skeleton_id=None,
        detection_model_run_id=run_id,
        path=onnx_new,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        nms_conf=0.81,
        nms_iou=0.66,
        nms_topk=2,
        metadata=None,
    )
    registry.record_tensorrt_model(
        run_id=run_id,
        set_id="detect_set_v001",
        skeleton_id=None,
        detection_model_run_id=run_id,
        onnx_run_id=run_id,
        precision="fp16",
        path=trt_new,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        nms_conf=0.79,
        nms_iou=0.64,
        nms_topk=3,
        metadata=None,
    )

    onnx_rows = _load_onnx_rows(registry, set_filter=None, limit=None, hide_unlinked=False)
    trt_rows = _load_tensorrt_rows(registry, set_filter=None, limit=None, hide_unlinked=False)
    onnx_row = next((row for row in onnx_rows if row.run_id == run_id), None)
    trt_row = next((row for row in trt_rows if row.run_id == run_id), None)
    assert onnx_row is not None
    assert trt_row is not None
    assert onnx_row.nms_conf is not None and round(float(onnx_row.nms_conf), 2) == 0.81
    assert onnx_row.nms_iou is not None and round(float(onnx_row.nms_iou), 2) == 0.66
    assert onnx_row.nms_topk == 2
    assert trt_row.nms_conf is not None and round(float(trt_row.nms_conf), 2) == 0.79
    assert trt_row.nms_iou is not None and round(float(trt_row.nms_iou), 2) == 0.64
    assert trt_row.nms_topk == 3
    registry.close()


def test_view_outputs_include_nms_summary(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("cfg", encoding="utf-8")
    run_id = "run_nms_output_001"
    registry.record_training_run(
        run_id=run_id,
        set_id="detect_set_v001",
        config_path=config_path,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={"mAP50": 0.9},
    )

    onnx_path = tmp_path / "output.onnx"
    trt_path = tmp_path / "output_fp16.engine"
    onnx_path.write_text("onnx", encoding="utf-8")
    trt_path.write_text("trt", encoding="utf-8")

    registry.record_onnx_model(
        run_id=run_id,
        set_id="detect_set_v001",
        skeleton_id=None,
        detection_model_run_id=run_id,
        path=onnx_path,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        nms_conf=0.8,
        nms_iou=0.65,
        nms_topk=1,
        metadata=None,
    )
    registry.record_tensorrt_model(
        run_id=run_id,
        set_id="detect_set_v001",
        skeleton_id=None,
        detection_model_run_id=run_id,
        onnx_run_id=run_id,
        precision="fp16",
        path=trt_path,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        nms_conf=0.8,
        nms_iou=0.65,
        nms_topk=1,
        metadata=None,
    )
    registry.close()

    rc_onnx = check_training_registry_main(
        ["--registry", str(tmp_path / "registry.sqlite"), "--view", "onnx", "--no-rich"]
    )
    out_onnx = capsys.readouterr().out
    assert rc_onnx == 0
    assert "nms: c=0.800 i=0.650 k=1" in out_onnx

    rc_trt = check_training_registry_main(
        ["--registry", str(tmp_path / "registry.sqlite"), "--view", "tensorrt", "--no-rich"]
    )
    out_trt = capsys.readouterr().out
    assert rc_trt == 0
    assert "nms: c=0.800 i=0.650 k=1" in out_trt


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


def test_group_and_format_recording_vocab_rows_filters_empty_and_no_brackets() -> None:
    grouped = _group_recording_vocab_rows(
        [
            RecordingVocabRow(recording_type="behavior", recording_subtype="free"),
            RecordingVocabRow(recording_type="behavior", recording_subtype="embedded"),
            RecordingVocabRow(recording_type="behavior", recording_subtype="free"),
            RecordingVocabRow(recording_type=" ", recording_subtype="free"),
            RecordingVocabRow(recording_type="histology", recording_subtype="section"),
        ]
    )
    assert grouped == {
        "behavior": ["embedded", "free"],
        "histology": ["section"],
    }
    inline = _format_recording_vocab_inline(grouped)
    assert inline == "behavior:embedded,free; histology:section"
    assert "[" not in inline
    assert "]" not in inline


def test_recording_summary_mode_does_not_require_training_sets(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "rec_001",
            "session_001",
            "rec_001",
            str(tmp_path / "recordings" / "rec_001"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        ["--registry", str(tmp_path / "registry.sqlite"), "--recording-summary", "--no-rich"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recording Type/Subtype Summary" in out
    assert "behavior / free: 1" in out


def test_recording_overview_view_works_without_training_sets(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "rec_ov_001",
            "session_ov_001",
            "rec_ov_001",
            str(tmp_path / "recordings" / "rec_ov_001"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.upsert_dataset(
        "session_ov_001:zabc",
        session_uuid="session_ov_001",
        zarr_path=tmp_path / "recordings" / "rec_ov_001" / "zarr" / "rec_ov_001_training.zarr",
        recording_id="rec_ov_001",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.close()

    rc = check_training_registry_main(
        ["--registry", str(tmp_path / "registry.sqlite"), "--view", "recording-overview", "--no-rich"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recording Overview" in out
    assert "rec_ov_001" in out


def test_recording_steps_view_outputs_overview_rows(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_steps_a",
        session_uuid="session_steps_a",
        zarr_path=tmp_path / "recordings" / "rec_steps_001" / "zarr" / "rec_steps_001_training.zarr",
        recording_id="rec_steps_001",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_steps_b",
        session_uuid="session_steps_b",
        zarr_path=tmp_path / "recordings" / "rec_steps_001" / "zarr" / "rec_steps_001_analysis.zarr",
        recording_id="rec_steps_001",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                "dataset_steps_a",
                "rec_steps_001",
                "detect",
                "ok",
                "detect_a",
                "unit_test",
                "2026-02-22T00:00:00+00:00",
            ),
            (
                "dataset_steps_b",
                "rec_steps_001",
                "detect",
                "ok",
                "detect_b",
                "unit_test",
                "2026-02-22T00:00:01+00:00",
            ),
            (
                "dataset_steps_a",
                "rec_steps_001",
                "keypoints",
                "missing",
                None,
                "unit_test",
                "2026-02-22T00:00:02+00:00",
            ),
        ],
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        ["--registry", str(tmp_path / "registry.sqlite"), "--view", "recording-steps", "--no-rich"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recording Step Overview" in out
    assert "rec_steps_001" in out
    assert "keypoints" in out
    assert "Recording Step Details" not in out


def test_recording_steps_view_show_details_outputs_step_rows(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_steps_detail",
        session_uuid="session_steps_detail",
        zarr_path=tmp_path / "recordings" / "rec_steps_002" / "zarr" / "rec_steps_002_analysis.zarr",
        recording_id="rec_steps_002",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_steps_detail",
            "rec_steps_002",
            "detect",
            "ok",
            "detect_detail",
            "yolo",
            98.5,
            "unit_test",
            "2026-02-22T00:10:00+00:00",
        ),
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "recording-steps",
            "--show-recording-step-details",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recording Step Overview" in out
    assert "Recording Step Details" in out
    assert "dataset_steps_detail" in out
    assert "detect" in out
    assert "yolo" in out


def test_recording_steps_wide_view_outputs_rows(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_steps_wide",
        session_uuid="session_steps_wide",
        zarr_path=tmp_path / "recordings" / "rec_steps_wide_001" / "zarr" / "rec_steps_wide_001_training.zarr",
        recording_id="rec_steps_wide_001",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "detect",
                "ok",
                "detect_run",
                "blob",
                73.6,
                "unit_test",
                "2026-02-22T02:00:00+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "refined_detect",
                "ok",
                "refined_detect_run",
                "passthrough",
                None,
                "unit_test",
                "2026-02-22T02:00:01+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "id_assignment",
                "missing",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:02+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "tracks",
                "missing",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:03+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "dish_mask",
                "ok",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:04+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "detection_tuning",
                "ok",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:05+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "keypoint_tuning",
                "ok",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:06+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "eye_mask_tuning",
                "ok",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:07+00:00",
            ),
            (
                "dataset_steps_wide",
                "rec_steps_wide_001",
                "subdish_mask_tuning",
                "na",
                None,
                None,
                None,
                "unit_test",
                "2026-02-22T02:00:08+00:00",
            ),
        ],
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "recording-steps-wide",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Recording Step Status (Wide)" in out
    assert "rec_steps_wide_001" in out
    assert "OK (73.6%, registry, blob)" in out
    assert "100% (passthrough)" in out
    assert "4/5" in out


def test_recording_steps_wide_view_respects_zarr_use_filter(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_steps_wide_training",
        session_uuid="session_steps_wide_training",
        zarr_path=tmp_path / "recordings" / "rec_steps_wide_training" / "zarr" / "rec_steps_wide_training.zarr",
        recording_id="rec_steps_wide_training",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_steps_wide_analysis",
        session_uuid="session_steps_wide_analysis",
        zarr_path=tmp_path / "recordings" / "rec_steps_wide_analysis" / "zarr" / "rec_steps_wide_analysis.zarr",
        recording_id="rec_steps_wide_analysis",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                "dataset_steps_wide_training",
                "rec_steps_wide_training",
                "detect",
                "ok",
                "detect_training",
                "blob",
                80.0,
                "unit_test",
                "2026-02-22T03:00:00+00:00",
            ),
            (
                "dataset_steps_wide_analysis",
                "rec_steps_wide_analysis",
                "detect",
                "ok",
                "detect_analysis",
                "blob",
                70.0,
                "unit_test",
                "2026-02-22T03:00:01+00:00",
            ),
        ],
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "recording-steps-wide",
            "--zarr-use",
            "training",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "rec_steps_wide_training" in out
    assert "rec_steps_wide_analysis" not in out
